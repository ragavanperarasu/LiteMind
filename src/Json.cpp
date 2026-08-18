#include "Json.hpp"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace litemind {
namespace {

/** Appends one Unicode code point to a UTF-8 string. */
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

[[nodiscard]] bool hex_digit(const char character, std::uint32_t& value) noexcept {
    if (character >= '0' && character <= '9') {
        value = static_cast<std::uint32_t>(character - '0');
    } else if (character >= 'a' && character <= 'f') {
        value = static_cast<std::uint32_t>(character - 'a') + 10U;
    } else if (character >= 'A' && character <= 'F') {
        value = static_cast<std::uint32_t>(character - 'A') + 10U;
    } else {
        return false;
    }
    return true;
}

}  // namespace

/**
 * A recursive-descent JSON parser.
 *
 * Depth is bounded so that a malformed or hostile document cannot exhaust the
 * call stack; tokenizer.json nests only a few levels deep in practice.
 */
class JsonParser final {
public:
    explicit JsonParser(const std::string_view source) : source_(source) {}

    [[nodiscard]] bool parse_document(Json& value, std::string& error) {
        skip_whitespace();
        if (!parse_value(value, 0U, error)) {
            return false;
        }
        skip_whitespace();
        if (position_ != source_.size()) {
            error = "JSON document contains trailing characters at offset "
                  + std::to_string(position_) + ".";
            return false;
        }
        return true;
    }

private:
    static constexpr std::size_t maximum_depth = 128U;

    [[nodiscard]] bool parse_value(Json& value, const std::size_t depth, std::string& error) {
        if (depth > maximum_depth) {
            error = "JSON document nests deeper than LiteMind supports.";
            return false;
        }
        skip_whitespace();
        if (position_ >= source_.size()) {
            error = "JSON document ended where a value was expected.";
            return false;
        }

        switch (source_[position_]) {
            case '{': return parse_object(value, depth, error);
            case '[': return parse_array(value, depth, error);
            case '"': {
                value.kind_ = Json::Kind::String;
                return parse_string(value.string_, error);
            }
            case 't': return parse_literal("true", Json::Kind::Boolean, true, value, error);
            case 'f': return parse_literal("false", Json::Kind::Boolean, false, value, error);
            case 'n': return parse_literal("null", Json::Kind::Null, false, value, error);
            default: return parse_number(value, error);
        }
    }

    [[nodiscard]] bool parse_object(Json& value, const std::size_t depth, std::string& error) {
        value.kind_ = Json::Kind::Object;
        ++position_;  // consume '{'
        skip_whitespace();
        if (consume('}')) {
            return true;
        }

        while (true) {
            skip_whitespace();
            std::string key;
            if (!parse_string(key, error)) {
                return false;
            }
            if (!consume(':')) {
                error = "JSON object member is missing its colon at offset "
                      + std::to_string(position_) + ".";
                return false;
            }

            Json member;
            if (!parse_value(member, depth + 1U, error)) {
                return false;
            }
            value.object_.insert_or_assign(std::move(key), std::move(member));

            skip_whitespace();
            if (consume(',')) {
                continue;
            }
            if (consume('}')) {
                return true;
            }
            error = "JSON object is missing a comma or closing brace at offset "
                  + std::to_string(position_) + ".";
            return false;
        }
    }

    [[nodiscard]] bool parse_array(Json& value, const std::size_t depth, std::string& error) {
        value.kind_ = Json::Kind::Array;
        ++position_;  // consume '['
        skip_whitespace();
        if (consume(']')) {
            return true;
        }

        while (true) {
            Json element;
            if (!parse_value(element, depth + 1U, error)) {
                return false;
            }
            value.array_.push_back(std::move(element));

            skip_whitespace();
            if (consume(',')) {
                continue;
            }
            if (consume(']')) {
                return true;
            }
            error = "JSON array is missing a comma or closing bracket at offset "
                  + std::to_string(position_) + ".";
            return false;
        }
    }

    [[nodiscard]] bool parse_string(std::string& output, std::string& error) {
        output.clear();
        skip_whitespace();
        if (position_ >= source_.size() || source_[position_] != '"') {
            error = "JSON string is missing its opening quote at offset "
                  + std::to_string(position_) + ".";
            return false;
        }
        ++position_;

        while (position_ < source_.size()) {
            const char character = source_[position_++];
            if (character == '"') {
                return true;
            }
            if (character != '\\') {
                output += character;
                continue;
            }
            if (position_ >= source_.size()) {
                break;
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
                case 'u': {
                    std::uint32_t code_point{};
                    if (!parse_hex_quad(code_point)) {
                        error = "JSON string contains a malformed \\u escape.";
                        return false;
                    }
                    // Combine a UTF-16 surrogate pair into one code point.
                    if (code_point >= 0xd800U && code_point <= 0xdbffU
                        && position_ + 1U < source_.size() && source_[position_] == '\\'
                        && source_[position_ + 1U] == 'u') {
                        const std::size_t saved = position_;
                        position_ += 2U;
                        std::uint32_t low{};
                        if (parse_hex_quad(low) && low >= 0xdc00U && low <= 0xdfffU) {
                            code_point = 0x10000U + ((code_point - 0xd800U) << 10U) + (low - 0xdc00U);
                        } else {
                            position_ = saved;
                        }
                    }
                    append_utf8(output, code_point);
                    break;
                }
                default:
                    error = "JSON string contains an unsupported escape sequence.";
                    return false;
            }
        }

        error = "JSON string is missing its closing quote.";
        return false;
    }

    [[nodiscard]] bool parse_hex_quad(std::uint32_t& code_point) noexcept {
        if (position_ + 4U > source_.size()) {
            return false;
        }
        code_point = 0U;
        for (std::size_t index = 0; index < 4U; ++index) {
            std::uint32_t digit{};
            if (!hex_digit(source_[position_ + index], digit)) {
                return false;
            }
            code_point = (code_point << 4U) | digit;
        }
        position_ += 4U;
        return true;
    }

    [[nodiscard]] bool parse_number(Json& value, std::string& error) {
        const std::size_t start = position_;
        if (position_ < source_.size() && (source_[position_] == '-' || source_[position_] == '+')) {
            ++position_;
        }
        while (position_ < source_.size()) {
            const char character = source_[position_];
            const bool numeric = (character >= '0' && character <= '9') || character == '.'
                              || character == 'e' || character == 'E' || character == '-'
                              || character == '+';
            if (!numeric) {
                break;
            }
            ++position_;
        }
        if (start == position_) {
            error = "JSON value at offset " + std::to_string(start) + " is not recognised.";
            return false;
        }

        const std::string text(source_.substr(start, position_ - start));
        char* parse_end = nullptr;
        errno = 0;
        const double parsed = std::strtod(text.c_str(), &parse_end);
        if (parse_end != text.c_str() + text.size()) {
            error = "JSON number '" + text + "' is malformed.";
            return false;
        }
        value.kind_ = Json::Kind::Number;
        value.number_ = parsed;
        return true;
    }

    [[nodiscard]] bool parse_literal(const std::string_view literal, const Json::Kind kind,
                                     const bool boolean, Json& value, std::string& error) {
        if (source_.compare(position_, literal.size(), literal) != 0) {
            error = "JSON literal at offset " + std::to_string(position_) + " is malformed.";
            return false;
        }
        position_ += literal.size();
        value.kind_ = kind;
        value.boolean_ = boolean;
        return true;
    }

    void skip_whitespace() noexcept {
        while (position_ < source_.size()) {
            const char character = source_[position_];
            if (character != ' ' && character != '\t' && character != '\n' && character != '\r') {
                break;
            }
            ++position_;
        }
    }

    [[nodiscard]] bool consume(const char expected) noexcept {
        skip_whitespace();
        if (position_ < source_.size() && source_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    std::string_view source_;
    std::size_t position_{};
};

bool Json::parse(const std::string_view text, Json& value, std::string& error) {
    value = Json{};
    JsonParser parser(text);
    return parser.parse_document(value, error);
}

bool Json::parse_file(const std::string& path, Json& value, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "Unable to open JSON file: " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    // Skip a UTF-8 byte-order mark, which some Windows editors add on save.
    constexpr std::string_view byte_order_mark = "\xEF\xBB\xBF";
    if (text.size() >= byte_order_mark.size() && text.compare(0, byte_order_mark.size(), byte_order_mark) == 0) {
        text.erase(0, byte_order_mark.size());
    }
    if (!parse(text, value, error)) {
        error = path + ": " + error;
        return false;
    }
    return true;
}

const Json* Json::find(const std::string_view key) const {
    if (kind_ != Kind::Object) {
        return nullptr;
    }
    const auto member = object_.find(key);
    return member == object_.end() ? nullptr : &member->second;
}

double Json::number_or(const std::string_view key, const double fallback) const {
    const Json* member = find(key);
    return (member != nullptr && member->is_number()) ? member->number_ : fallback;
}

std::uint64_t Json::unsigned_or(const std::string_view key, const std::uint64_t fallback) const {
    const Json* member = find(key);
    if (member == nullptr || !member->is_number() || member->number_ < 0.0) {
        return fallback;
    }
    return static_cast<std::uint64_t>(member->number_);
}

bool Json::boolean_or(const std::string_view key, const bool fallback) const {
    const Json* member = find(key);
    if (member == nullptr) {
        return fallback;
    }
    if (member->kind_ == Kind::Boolean) {
        return member->boolean_;
    }
    if (member->is_number()) {
        return member->number_ != 0.0;
    }
    return fallback;
}

std::string Json::string_or(const std::string_view key, const std::string_view fallback) const {
    const Json* member = find(key);
    return (member != nullptr && member->is_string()) ? member->string_ : std::string(fallback);
}

}  // namespace litemind
