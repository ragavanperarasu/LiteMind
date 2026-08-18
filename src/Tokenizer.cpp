#include "Tokenizer.hpp"

#include "Json.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

namespace litemind {
namespace {

/** Separates the two halves of a merge rule inside the rank map's key. */
constexpr char pair_separator = '\x1f';

// ── Byte-level alphabet ──────────────────────────────────────────────────────

[[nodiscard]] constexpr bool is_printable_byte(const std::uint32_t byte) noexcept {
    return (byte >= 33U && byte <= 126U) || (byte >= 161U && byte <= 172U) || byte >= 174U;
}

void append_utf8(std::string& output, const std::uint32_t code_point) {
    if (code_point <= 0x7fU) {
        output += static_cast<char>(code_point);
    } else if (code_point <= 0x7ffU) {
        output += static_cast<char>(0xc0U | (code_point >> 6U));
        output += static_cast<char>(0x80U | (code_point & 0x3fU));
    } else if (code_point <= 0xffffU) {
        output += static_cast<char>(0xe0U | (code_point >> 12U));
        output += static_cast<char>(0x80U | ((code_point >> 6U) & 0x3fU));
        output += static_cast<char>(0x80U | (code_point & 0x3fU));
    } else {
        output += static_cast<char>(0xf0U | (code_point >> 18U));
        output += static_cast<char>(0x80U | ((code_point >> 12U) & 0x3fU));
        output += static_cast<char>(0x80U | ((code_point >> 6U) & 0x3fU));
        output += static_cast<char>(0x80U | (code_point & 0x3fU));
    }
}

/**
 * GPT-2's byte alphabet: every one of the 256 byte values becomes a printable
 * code point, so the merge table only ever sees text and no token is unknown.
 */
[[nodiscard]] const std::array<std::uint32_t, 256U>& byte_to_code_point() noexcept {
    static const std::array<std::uint32_t, 256U> table = [] {
        std::array<std::uint32_t, 256U> result{};
        std::uint32_t generated = 256U;
        for (std::uint32_t byte = 0; byte < 256U; ++byte) {
            result[byte] = is_printable_byte(byte) ? byte : generated++;
        }
        return result;
    }();
    return table;
}

[[nodiscard]] const std::array<std::int16_t, 512U>& code_point_to_byte() noexcept {
    static const std::array<std::int16_t, 512U> table = [] {
        std::array<std::int16_t, 512U> result{};
        result.fill(-1);
        const auto& forward = byte_to_code_point();
        for (std::size_t byte = 0; byte < forward.size(); ++byte) {
            result[forward[byte]] = static_cast<std::int16_t>(byte);
        }
        return result;
    }();
    return table;
}

// ── UTF-8 scanning ───────────────────────────────────────────────────────────

/** Decodes one code point and advances position. Invalid bytes decode as themselves. */
[[nodiscard]] std::uint32_t decode_utf8(const std::string_view text, std::size_t& position) noexcept {
    const auto first = static_cast<unsigned char>(text[position]);
    const std::size_t available = text.size() - position;

    if (first < 0x80U) {
        ++position;
        return first;
    }
    if ((first & 0xe0U) == 0xc0U && available >= 2U) {
        const auto second = static_cast<unsigned char>(text[position + 1U]);
        position += 2U;
        return ((first & 0x1fU) << 6U) | (second & 0x3fU);
    }
    if ((first & 0xf0U) == 0xe0U && available >= 3U) {
        const auto second = static_cast<unsigned char>(text[position + 1U]);
        const auto third = static_cast<unsigned char>(text[position + 2U]);
        position += 3U;
        return ((first & 0x0fU) << 12U) | ((second & 0x3fU) << 6U) | (third & 0x3fU);
    }
    if ((first & 0xf8U) == 0xf0U && available >= 4U) {
        const auto second = static_cast<unsigned char>(text[position + 1U]);
        const auto third = static_cast<unsigned char>(text[position + 2U]);
        const auto fourth = static_cast<unsigned char>(text[position + 3U]);
        position += 4U;
        return ((first & 0x07U) << 18U) | ((second & 0x3fU) << 12U) | ((third & 0x3fU) << 6U)
             | (fourth & 0x3fU);
    }
    ++position;
    return first;
}

/** The number of bytes a UTF-8 sequence starting with this byte occupies. */
[[nodiscard]] std::size_t utf8_sequence_length(const unsigned char first) noexcept {
    if (first < 0x80U) return 1U;
    if ((first & 0xe0U) == 0xc0U) return 2U;
    if ((first & 0xf0U) == 0xe0U) return 3U;
    if ((first & 0xf8U) == 0xf0U) return 4U;
    return 1U;  // A stray continuation byte; emit it rather than stalling.
}

// ── Pre-tokenizer character classes ──────────────────────────────────────────
//
// These mirror the Split rules in DeepSeek's tokenizer.json. Its letter class
// is an explicit list of Unicode letter ranges that deliberately excludes CJK,
// which the following rule handles separately; the ranges below cover the
// scripts that list names. Its punctuation class is written as the ASCII ranges
// !-/ and :-~, which also contain the letters — but the letter rule runs first
// and has already claimed them, so only true punctuation ever reaches it.

[[nodiscard]] constexpr bool is_cjk(const std::uint32_t code_point) noexcept {
    return (code_point >= 0x2E80U && code_point <= 0x303FU)   // CJK radicals and punctuation
        || (code_point >= 0x3040U && code_point <= 0x30FFU)   // Kana
        || (code_point >= 0x3400U && code_point <= 0x4DBFU)   // CJK extension A
        || (code_point >= 0x4E00U && code_point <= 0x9FFFU)   // CJK unified ideographs
        || (code_point >= 0xAC00U && code_point <= 0xD7AFU)   // Hangul syllables
        || (code_point >= 0xF900U && code_point <= 0xFAFFU)   // CJK compatibility
        || (code_point >= 0x20000U && code_point <= 0x3FFFFU);
}

[[nodiscard]] constexpr bool is_letter(const std::uint32_t code_point) noexcept {
    if (is_cjk(code_point)) {
        return false;
    }
    return (code_point >= 'A' && code_point <= 'Z')
        || (code_point >= 'a' && code_point <= 'z')
        || code_point == 0xB5U                                       // micro sign
        || (code_point >= 0xC0U && code_point <= 0xFFU && code_point != 0xD7U && code_point != 0xF7U)
        || (code_point >= 0x100U && code_point <= 0x2AFU)            // Latin extended, IPA
        || (code_point >= 0x370U && code_point <= 0x3FFU)            // Greek
        || (code_point >= 0x400U && code_point <= 0x52FU)            // Cyrillic
        || (code_point >= 0x531U && code_point <= 0x58FU)            // Armenian
        || (code_point >= 0x5D0U && code_point <= 0x5EAU)            // Hebrew
        || (code_point >= 0x620U && code_point <= 0x64AU)            // Arabic
        || (code_point >= 0x900U && code_point <= 0x97FU)            // Devanagari
        || (code_point >= 0xB80U && code_point <= 0xBFFU)            // Tamil
        || (code_point >= 0x10A0U && code_point <= 0x10FFU)          // Georgian
        || (code_point >= 0x1E00U && code_point <= 0x1FFFU)          // Latin/Greek additional
        || (code_point >= 0xFF21U && code_point <= 0xFF3AU)          // Fullwidth A-Z
        || (code_point >= 0xFF41U && code_point <= 0xFF5AU);         // Fullwidth a-z
}

[[nodiscard]] constexpr bool is_punctuation(const std::uint32_t code_point) noexcept {
    return (code_point >= 0x21U && code_point <= 0x2FU)
        || (code_point >= 0x3AU && code_point <= 0x40U)
        || (code_point >= 0x5BU && code_point <= 0x60U)
        || (code_point >= 0x7BU && code_point <= 0x7EU)
        || (code_point >= 0x2018U && code_point <= 0x201FU)          // Curly quotes
        || (code_point >= 0x3000U && code_point <= 0x3002U)          // Ideographic space and stops
        || (code_point >= 0xFF01U && code_point <= 0xFF0FU)
        || (code_point >= 0xFF1AU && code_point <= 0xFF20U)
        || (code_point >= 0xFF3BU && code_point <= 0xFF40U)
        || (code_point >= 0xFF5BU && code_point <= 0xFF5EU);
}

[[nodiscard]] constexpr bool is_digit(const std::uint32_t code_point) noexcept {
    return code_point >= '0' && code_point <= '9';
}

/** Whitespace other than the line breaks, which the first Split rule isolates. */
[[nodiscard]] constexpr bool is_inline_space(const std::uint32_t code_point) noexcept {
    return code_point == ' ' || code_point == '\t' || code_point == 0x0BU || code_point == 0x0CU
        || code_point == 0xA0U || code_point == 0x3000U;
}

[[nodiscard]] constexpr bool is_line_break(const std::uint32_t code_point) noexcept {
    return code_point == '\r' || code_point == '\n';
}

/** Peeks at the code point after position without consuming it. */
[[nodiscard]] std::uint32_t peek(const std::string_view text, const std::size_t position) noexcept {
    if (position >= text.size()) {
        return 0U;
    }
    std::size_t scan = position;
    return decode_utf8(text, scan);
}

}  // namespace

// ── Loading ──────────────────────────────────────────────────────────────────

bool Tokenizer::load(const std::filesystem::path& path, std::string& error) {
    vocabulary_.clear();
    merge_ranks_.clear();
    reverse_vocabulary_.clear();
    special_flags_.clear();
    added_tokens_.clear();
    has_bos_ = false;

    const std::filesystem::path file =
        std::filesystem::is_directory(path) ? path / "tokenizer.json" : path;
    if (!std::filesystem::exists(file)) {
        error = "tokenizer.json was not found at: " + file.string();
        return false;
    }

    Json document;
    if (!Json::parse_file(file.string(), document, error)) {
        return false;
    }

    if (!load_vocabulary(document, error) || !load_merges(document, error)) {
        error = file.string() + ": " + error;
        vocabulary_.clear();
        return false;
    }
    load_added_tokens(document);
    load_tokenizer_config(file.parent_path());

    source_path_ = file;
    return true;
}

bool Tokenizer::load_vocabulary(const Json& document, std::string& error) {
    const Json* model = document.find("model");
    const Json* vocabulary = model != nullptr ? model->find("vocab") : nullptr;
    if (vocabulary == nullptr || !vocabulary->is_object()) {
        error = "the tokenizer has no model.vocab object.";
        return false;
    }

    std::uint32_t highest = 0U;
    for (const auto& [token, identifier] : vocabulary->members()) {
        if (!identifier.is_number()) {
            error = "vocabulary entry '" + token + "' does not map to a number.";
            return false;
        }
        const auto id = static_cast<std::uint32_t>(identifier.number_value());
        vocabulary_.emplace(token, id);
        highest = std::max(highest, id);
    }
    if (vocabulary_.empty()) {
        error = "the tokenizer's vocabulary is empty.";
        return false;
    }

    reverse_vocabulary_.assign(static_cast<std::size_t>(highest) + 1U, std::string{});
    special_flags_.assign(reverse_vocabulary_.size(), false);
    for (const auto& [token, id] : vocabulary_) {
        reverse_vocabulary_[id] = token;
    }
    return true;
}

bool Tokenizer::load_merges(const Json& document, std::string& error) {
    const Json* model = document.find("model");
    const Json* merges = model != nullptr ? model->find("merges") : nullptr;
    if (merges == nullptr || !merges->is_array()) {
        error = "the tokenizer has no model.merges array.";
        return false;
    }

    std::uint32_t rank = 0U;
    for (const Json& entry : merges->elements()) {
        std::string left;
        std::string right;

        if (entry.is_string()) {
            // The classic format: the two halves separated by a space.
            const std::string& text = entry.string_value();
            const std::size_t separator = text.find(' ');
            if (separator == std::string::npos) {
                error = "merge rule '" + text + "' is not a space-separated pair.";
                return false;
            }
            left = text.substr(0U, separator);
            right = text.substr(separator + 1U);
        } else if (entry.is_array() && entry.elements().size() == 2U
                   && entry.elements()[0].is_string() && entry.elements()[1].is_string()) {
            // The newer format, which stores each half as its own string.
            left = entry.elements()[0].string_value();
            right = entry.elements()[1].string_value();
        } else {
            error = "a merge rule is neither a string nor a two-element array.";
            return false;
        }

        merge_ranks_.emplace(left + pair_separator + right, rank++);
    }

    if (merge_ranks_.empty()) {
        error = "the tokenizer's merge table is empty.";
        return false;
    }
    return true;
}

void Tokenizer::load_added_tokens(const Json& document) {
    const Json* added = document.find("added_tokens");
    if (added == nullptr || !added->is_array()) {
        return;
    }

    for (const Json& entry : added->elements()) {
        const Json* content = entry.find("content");
        const Json* identifier = entry.find("id");
        if (content == nullptr || !content->is_string() || identifier == nullptr
            || !identifier->is_number()) {
            continue;
        }
        const auto id = static_cast<std::uint32_t>(identifier->number_value());
        const std::string& text = content->string_value();

        if (id >= reverse_vocabulary_.size()) {
            reverse_vocabulary_.resize(static_cast<std::size_t>(id) + 1U);
            special_flags_.resize(reverse_vocabulary_.size(), false);
        }
        // Added tokens carry their literal text, not a byte-level spelling.
        reverse_vocabulary_[id] = text;
        special_flags_[id] = entry.boolean_or("special", true);
        added_tokens_.emplace_back(text, id);
    }

    // Longest first, so a token whose text contains another still matches whole.
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const auto& left, const auto& right) {
                  return left.first.size() > right.first.size();
              });
}

void Tokenizer::load_tokenizer_config(const std::filesystem::path& directory) {
    Json document;
    std::string error;
    const std::filesystem::path path = directory / "tokenizer_config.json";
    if (!std::filesystem::exists(path) || !Json::parse_file(path.string(), document, error)) {
        return;
    }

    const auto resolve = [this](const Json* node, std::uint32_t& out) {
        if (node == nullptr) {
            return false;
        }
        // The field is either the token's text or an object holding it.
        const std::string text = node->is_string() ? node->string_value() : node->string_or("content", "");
        if (text.empty()) {
            return false;
        }
        for (const auto& [spelling, id] : added_tokens_) {
            if (spelling == text) {
                out = id;
                return true;
            }
        }
        const auto entry = vocabulary_.find(text);
        if (entry != vocabulary_.end()) {
            out = entry->second;
            return true;
        }
        return false;
    };

    has_bos_ = resolve(document.find("bos_token"), bos_token_id_)
            && document.boolean_or("add_bos_token", true);
    static_cast<void>(resolve(document.find("eos_token"), eos_token_id_));
}

// ── Encoding ─────────────────────────────────────────────────────────────────

std::vector<std::uint32_t> Tokenizer::encode(const std::string_view text, const bool add_bos) const {
    std::vector<std::uint32_t> token_ids;
    if (!ready()) {
        return token_ids;
    }
    if (add_bos && has_bos_) {
        token_ids.push_back(bos_token_id_);
    }

    // Added tokens are matched literally and never run through BPE, so a
    // control token written out in a prompt encodes to its own single ID.
    std::size_t position = 0U;
    std::size_t ordinary_start = 0U;
    while (position < text.size()) {
        bool matched = false;
        for (const auto& [spelling, id] : added_tokens_) {
            if (!spelling.empty() && text.compare(position, spelling.size(), spelling) == 0) {
                if (position > ordinary_start) {
                    encode_ordinary(text.substr(ordinary_start, position - ordinary_start), token_ids);
                }
                token_ids.push_back(id);
                position += spelling.size();
                ordinary_start = position;
                matched = true;
                break;
            }
        }
        if (!matched) {
            ++position;
        }
    }
    if (ordinary_start < text.size()) {
        encode_ordinary(text.substr(ordinary_start), token_ids);
    }
    return token_ids;
}

void Tokenizer::encode_ordinary(const std::string_view text, std::vector<std::uint32_t>& output) const {
    for (const std::string& piece : pre_tokenize(text)) {
        apply_bpe(byte_encode(piece), output);
    }
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string_view text) const {
    std::vector<std::string> pieces;
    std::size_t position = 0U;

    while (position < text.size()) {
        const std::size_t start = position;
        std::size_t scan = position;
        const std::uint32_t first = decode_utf8(text, scan);

        if (is_line_break(first)) {
            // Rule 1: each carriage return or newline stands alone.
            position = scan;
        } else if (is_letter(first)
                   || (is_inline_space(first) && is_letter(peek(text, scan)))) {
            // Rule 2: an optional single space, then a run of letters.
            position = scan;
            while (position < text.size()) {
                std::size_t next = position;
                if (!is_letter(decode_utf8(text, next))) {
                    break;
                }
                position = next;
            }
        } else if (is_punctuation(first)
                   || (is_inline_space(first) && is_punctuation(peek(text, scan)))) {
            // Rule 3: an optional single space, then a run of punctuation.
            position = scan;
            while (position < text.size()) {
                std::size_t next = position;
                if (!is_punctuation(decode_utf8(text, next))) {
                    break;
                }
                position = next;
            }
        } else if (is_cjk(first)) {
            // Rule 5: a run of ideographs.
            position = scan;
            while (position < text.size()) {
                std::size_t next = position;
                if (!is_cjk(decode_utf8(text, next))) {
                    break;
                }
                position = next;
            }
        } else if (is_digit(first)) {
            // Rule 6: digits are always split one at a time.
            position = scan;
        } else if (is_inline_space(first)) {
            // Whatever whitespace the letter and punctuation rules left behind.
            position = scan;
            while (position < text.size()) {
                std::size_t next = position;
                if (!is_inline_space(decode_utf8(text, next))) {
                    break;
                }
                position = next;
            }
        } else {
            // Symbols and anything unclassified, grouped into one run.
            position = scan;
            while (position < text.size()) {
                std::size_t next = position;
                const std::uint32_t code_point = decode_utf8(text, next);
                if (is_letter(code_point) || is_punctuation(code_point) || is_digit(code_point)
                    || is_cjk(code_point) || is_inline_space(code_point) || is_line_break(code_point)) {
                    break;
                }
                position = next;
            }
        }

        if (position == start) {
            ++position;  // Defensive: never loop without consuming input.
        }
        pieces.emplace_back(text.substr(start, position - start));
    }
    return pieces;
}

std::string Tokenizer::byte_encode(const std::string_view text) const {
    std::string encoded;
    encoded.reserve(text.size() * 2U);
    const auto& forward = byte_to_code_point();
    for (const char character : text) {
        append_utf8(encoded, forward[static_cast<unsigned char>(character)]);
    }
    return encoded;
}

void Tokenizer::apply_bpe(const std::string& piece, std::vector<std::uint32_t>& output) const {
    if (piece.empty()) {
        return;
    }

    // Start from one symbol per byte-alphabet character.
    std::vector<std::string> symbols;
    for (std::size_t position = 0; position < piece.size();) {
        const std::size_t length = utf8_sequence_length(static_cast<unsigned char>(piece[position]));
        symbols.push_back(piece.substr(position, length));
        position += length;
    }

    // Repeatedly merge the adjacent pair with the lowest rank, which is the
    // order the merge table was learned in.
    std::string key;
    while (symbols.size() > 1U) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t best_position = symbols.size();

        for (std::size_t index = 0; index + 1U < symbols.size(); ++index) {
            key.assign(symbols[index]).append(1U, pair_separator).append(symbols[index + 1U]);
            const auto merge = merge_ranks_.find(key);
            if (merge != merge_ranks_.end() && merge->second < best_rank) {
                best_rank = merge->second;
                best_position = index;
            }
        }
        if (best_position == symbols.size()) {
            break;
        }

        symbols[best_position] += symbols[best_position + 1U];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_position + 1U));
    }

    for (const std::string& symbol : symbols) {
        const auto entry = vocabulary_.find(symbol);
        if (entry != vocabulary_.end()) {
            output.push_back(entry->second);
            continue;
        }
        // Unreachable for a well-formed byte-level vocabulary, but falling back
        // to single-character tokens keeps encoding total rather than failing.
        for (std::size_t position = 0; position < symbol.size();) {
            const std::size_t length = utf8_sequence_length(static_cast<unsigned char>(symbol[position]));
            const auto single = vocabulary_.find(symbol.substr(position, length));
            if (single != vocabulary_.end()) {
                output.push_back(single->second);
            }
            position += length;
        }
    }
}

// ── Decoding ─────────────────────────────────────────────────────────────────

std::string Tokenizer::decode(const std::span<const std::uint32_t> token_ids) const {
    std::string output;
    for (const std::uint32_t token_id : token_ids) {
        if (token_id >= reverse_vocabulary_.size()) {
            continue;
        }
        if (is_special(token_id)) {
            continue;  // Control tokens are not part of the produced text.
        }

        const std::string& spelling = reverse_vocabulary_[token_id];
        const auto& inverse = code_point_to_byte();
        for (std::size_t position = 0; position < spelling.size();) {
            const std::uint32_t code_point = decode_utf8(spelling, position);
            if (code_point < inverse.size() && inverse[code_point] >= 0) {
                output += static_cast<char>(inverse[code_point]);
            }
        }
    }
    return output;
}

bool Tokenizer::is_special(const std::uint32_t token_id) const {
    return token_id < special_flags_.size() && special_flags_[token_id];
}

std::string Tokenizer::token_text(const std::uint32_t token_id) const {
    if (token_id >= reverse_vocabulary_.size()) {
        return "<out-of-range:" + std::to_string(token_id) + ">";
    }
    return reverse_vocabulary_[token_id];
}

std::string Tokenizer::StreamDecoder::push(const std::uint32_t token_id) {
    const std::array<std::uint32_t, 1U> single{token_id};
    pending_ += tokenizer_->decode(single);

    // Emit everything up to the last complete UTF-8 sequence and keep the rest.
    std::size_t emitted = 0U;
    while (emitted < pending_.size()) {
        const std::size_t length = utf8_sequence_length(static_cast<unsigned char>(pending_[emitted]));
        if (emitted + length > pending_.size()) {
            break;
        }
        emitted += length;
    }

    std::string ready = pending_.substr(0U, emitted);
    pending_.erase(0U, emitted);
    return ready;
}

std::string Tokenizer::StreamDecoder::flush() {
    std::string remainder = std::move(pending_);
    pending_.clear();
    return remainder;
}

}  // namespace litemind
