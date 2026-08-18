#include "Tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

namespace litemind {
namespace {

constexpr std::uint32_t bos_token_id = 100000U;
constexpr char pair_separator = '\x1f';

class JsonCursor final {
public:
    explicit JsonCursor(std::string_view source) : source_(source) {}

    [[nodiscard]] bool find_key(const std::string_view key) {
        const std::string needle = '"' + std::string(key) + '"';
        const std::size_t location = source_.find(needle, position_);
        if (location == std::string_view::npos) {
            return false;
        }
        position_ = location + needle.size();
        skip_space();
        return consume(':');
    }

    [[nodiscard]] bool consume(const char expected) {
        skip_space();
        if (position_ < source_.size() && source_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    [[nodiscard]] std::optional<std::string> string() {
        skip_space();
        if (position_ == source_.size() || source_[position_] != '"') {
            return std::nullopt;
        }
        ++position_;
        std::string output;
        while (position_ < source_.size()) {
            const char character = source_[position_++];
            if (character == '"') {
                return output;
            }
            if (character != '\\') {
                output += character;
                continue;
            }
            if (position_ == source_.size()) {
                return std::nullopt;
            }
            const char escaped = source_[position_++];
            switch (escaped) {
                case '"': output += '"'; break;
                case '\\': output += '\\'; break;
                case '/': output += '/'; break;
                case 'b': output += '\b'; break;
                case 'f': output += '\f'; break;
                case 'n': output += '\n'; break;
                case 'r': output += '\r'; break;
                case 't': output += '\t'; break;
                default: return std::nullopt;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::uint32_t> unsigned_number() {
        skip_space();
        const std::size_t start = position_;
        while (position_ < source_.size()
               && std::isdigit(static_cast<unsigned char>(source_[position_])) != 0) {
            ++position_;
        }
        if (start == position_) {
            return std::nullopt;
        }
        try {
            const auto parsed = std::stoull(std::string(source_.substr(start, position_ - start)));
            if (parsed > std::numeric_limits<std::uint32_t>::max()) {
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(parsed);
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

private:
    void skip_space() {
        while (position_ < source_.size()
               && std::isspace(static_cast<unsigned char>(source_[position_])) != 0) {
            ++position_;
        }
    }

    std::string_view source_;
    std::size_t position_{};
};

void append_utf8(std::string& output, const std::uint32_t code_point) {
    if (code_point <= 0x7fU) {
        output += static_cast<char>(code_point);
    } else if (code_point <= 0x7ffU) {
        output += static_cast<char>(0xc0U | (code_point >> 6U));
        output += static_cast<char>(0x80U | (code_point & 0x3fU));
    } else {
        output += static_cast<char>(0xe0U | (code_point >> 12U));
        output += static_cast<char>(0x80U | ((code_point >> 6U) & 0x3fU));
        output += static_cast<char>(0x80U | (code_point & 0x3fU));
    }
}

[[nodiscard]] std::uint32_t next_utf8_code_point(const std::string_view text, std::size_t& position) {
    const auto first = static_cast<unsigned char>(text[position++]);
    if (first < 0x80U || position >= text.size()) {
        return first;
    }
    if ((first & 0xe0U) == 0xc0U) {
        return ((first & 0x1fU) << 6U) | (static_cast<unsigned char>(text[position++]) & 0x3fU);
    }
    if ((first & 0xf0U) == 0xe0U && position + 1U < text.size()) {
        const auto second = static_cast<unsigned char>(text[position++]);
        const auto third = static_cast<unsigned char>(text[position++]);
        return ((first & 0x0fU) << 12U) | ((second & 0x3fU) << 6U) | (third & 0x3fU);
    }
    return first;
}

[[nodiscard]] bool is_word_byte(const unsigned char value) noexcept {
    return std::isalpha(value) != 0 || value >= 0x80U;
}

[[nodiscard]] constexpr bool is_direct_byte_code_point(const std::uint32_t byte) noexcept {
    return (byte >= 33U && byte <= 126U) || (byte >= 161U && byte <= 172U) || byte >= 174U;
}

/**
 * GPT-2/ByteLevel BPE represents every byte as a printable Unicode code point.
 * Keeping this table as the sole encoding definition prevents decode drift.
 */
[[nodiscard]] const std::array<std::uint32_t, 256U>& byte_to_code_point() noexcept {
    static const std::array<std::uint32_t, 256U> table = [] {
        std::array<std::uint32_t, 256U> result{};
        std::uint32_t generated_code_point = 256U;
        for (std::uint32_t byte = 0; byte < result.size(); ++byte) {
            result[byte] = is_direct_byte_code_point(byte) ? byte : generated_code_point++;
        }
        return result;
    }();
    return table;
}

[[nodiscard]] const std::array<std::int16_t, 324U>& code_point_to_byte() noexcept {
    static const std::array<std::int16_t, 324U> table = [] {
        std::array<std::int16_t, 324U> result{};
        result.fill(-1);
        const auto& forward = byte_to_code_point();
        for (std::size_t byte = 0; byte < forward.size(); ++byte) {
            result[forward[byte]] = static_cast<std::int16_t>(byte);
        }
        return result;
    }();
    return table;
}

}  // namespace

bool Tokenizer::load(const std::filesystem::path& tokenizer_path, std::string& error) {
    vocabulary_.clear();
    merge_ranks_.clear();
    reverse_vocabulary_.clear();

    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file) {
        error = "Unable to open tokenizer: " + tokenizer_path.string();
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string document = buffer.str();

    JsonCursor vocabulary_cursor(document);
    if (!vocabulary_cursor.find_key("vocab") || !vocabulary_cursor.consume('{')) {
        error = "Tokenizer has no BPE vocabulary object.";
        return false;
    }
    while (!vocabulary_cursor.consume('}')) {
        const auto token = vocabulary_cursor.string();
        if (!token || !vocabulary_cursor.consume(':')) {
            error = "Tokenizer vocabulary entry is malformed.";
            return false;
        }
        const auto identifier = vocabulary_cursor.unsigned_number();
        if (!identifier || !vocabulary_cursor.consume(',')) {
            if (!identifier || !vocabulary_cursor.consume('}')) {
                error = "Tokenizer vocabulary identifier is malformed.";
                return false;
            }
            vocabulary_[*token] = *identifier;
            break;
        }
        vocabulary_[*token] = *identifier;
    }

    JsonCursor merge_cursor(document);
    if (!merge_cursor.find_key("merges") || !merge_cursor.consume('[')) {
        error = "Tokenizer has no BPE merge list.";
        return false;
    }
    std::uint32_t rank{};
    while (!merge_cursor.consume(']')) {
        const auto merge = merge_cursor.string();
        if (!merge) {
            error = "Tokenizer merge entry is malformed.";
            return false;
        }
        const std::size_t separator = merge->find(' ');
        if (separator == std::string::npos) {
            error = "Tokenizer merge does not contain a pair.";
            return false;
        }
        merge_ranks_[merge->substr(0, separator) + pair_separator + merge->substr(separator + 1U)] = rank++;
        if (!merge_cursor.consume(',')) {
            if (!merge_cursor.consume(']')) {
                error = "Tokenizer merge list is malformed.";
                return false;
            }
            break;
        }
    }

    reverse_vocabulary_.resize(vocabulary_.size());
    for (const auto& [token, identifier] : vocabulary_) {
        if (identifier >= reverse_vocabulary_.size()) {
            error = "Tokenizer vocabulary IDs are not contiguous.";
            vocabulary_.clear();
            return false;
        }
        reverse_vocabulary_[identifier] = token;
    }
    return true;
}

std::vector<std::uint32_t> Tokenizer::encode(const std::string_view text, const bool add_bos) const {
    std::vector<std::uint32_t> ids;
    if (!ready()) {
        return ids;
    }
    if (add_bos) {
        ids.push_back(bos_token_id);
    }
    for (const std::string& piece : pre_tokenize(text)) {
        for (const std::string& symbol : apply_bpe(byte_encode(piece))) {
            const auto token = vocabulary_.find(symbol);
            if (token == vocabulary_.end()) {
                return {};
            }
            ids.push_back(token->second);
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<std::uint32_t>& token_ids) const {
    std::string byte_level_text;
    for (const std::uint32_t identifier : token_ids) {
        if (identifier == bos_token_id || identifier >= reverse_vocabulary_.size()) {
            continue;
        }
        byte_level_text += reverse_vocabulary_[identifier];
    }

    std::string output;
    const auto& inverse_byte_table = code_point_to_byte();
    for (std::size_t position = 0; position < byte_level_text.size();) {
        const std::uint32_t code_point = next_utf8_code_point(byte_level_text, position);
        if (code_point >= inverse_byte_table.size() || inverse_byte_table[code_point] < 0) {
            continue;
        }
        output += static_cast<char>(inverse_byte_table[code_point]);
    }
    return output;
}

bool Tokenizer::ready() const noexcept { return !vocabulary_.empty() && !merge_ranks_.empty(); }
std::size_t Tokenizer::vocabulary_size() const noexcept { return vocabulary_.size(); }

std::vector<std::string> Tokenizer::pre_tokenize(const std::string_view text) const {
    std::vector<std::string> pieces;
    std::size_t position{};
    while (position < text.size()) {
        const std::size_t start = position;
        if (text[position] == '\r' || text[position] == '\n') {
            ++position;
        } else if (std::isspace(static_cast<unsigned char>(text[position])) != 0) {
            ++position;
            if (position < text.size() && is_word_byte(static_cast<unsigned char>(text[position]))) {
                while (position < text.size() && is_word_byte(static_cast<unsigned char>(text[position]))) {
                    ++position;
                }
            }
        } else if (std::isdigit(static_cast<unsigned char>(text[position])) != 0) {
            ++position;
        } else if (is_word_byte(static_cast<unsigned char>(text[position]))) {
            while (position < text.size() && is_word_byte(static_cast<unsigned char>(text[position]))) {
                ++position;
            }
        } else {
            ++position;
        }
        pieces.emplace_back(text.substr(start, position - start));
    }
    return pieces;
}

std::vector<std::string> Tokenizer::byte_encode(const std::string_view text) const {
    std::vector<std::string> symbols;
    const auto& forward_byte_table = byte_to_code_point();
    for (const unsigned char byte : text) {
        std::string symbol;
        append_utf8(symbol, forward_byte_table[byte]);
        symbols.push_back(std::move(symbol));
    }
    return symbols;
}

std::vector<std::string> Tokenizer::apply_bpe(std::vector<std::string> symbols) const {
    while (symbols.size() > 1U) {
        std::optional<std::size_t> best_position;
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        for (std::size_t index = 0; index + 1U < symbols.size(); ++index) {
            const auto merge = merge_ranks_.find(symbols[index] + pair_separator + symbols[index + 1U]);
            if (merge != merge_ranks_.end() && merge->second < best_rank) {
                best_rank = merge->second;
                best_position = index;
            }
        }
        if (!best_position) {
            break;
        }
        symbols[*best_position] += symbols[*best_position + 1U];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(*best_position + 1U));
    }
    return symbols;
}

}  // namespace litemind
