#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace litemind {

/**
 * @brief A small, complete JSON document model.
 *
 * LiteMind reads several JSON files that ship with a Hugging Face model
 * directory (config.json, tokenizer.json, generation_config.json). Substring
 * searching over those files is unreliable, because the same key name occurs in
 * more than one object. This parser produces a real tree so lookups are exact.
 */
class Json final {
public:
    enum class Kind { Null, Boolean, Number, String, Array, Object };

    Json() = default;

    /** Parses an entire JSON document. Returns false and sets error on failure. */
    [[nodiscard]] static bool parse(std::string_view text, Json& value, std::string& error);

    /** Parses an entire JSON file from disk. */
    [[nodiscard]] static bool parse_file(const std::string& path, Json& value, std::string& error);

    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    [[nodiscard]] bool is_null() const noexcept { return kind_ == Kind::Null; }
    [[nodiscard]] bool is_object() const noexcept { return kind_ == Kind::Object; }
    [[nodiscard]] bool is_array() const noexcept { return kind_ == Kind::Array; }
    [[nodiscard]] bool is_string() const noexcept { return kind_ == Kind::String; }
    [[nodiscard]] bool is_number() const noexcept { return kind_ == Kind::Number; }

    /** Returns the named member, or nullptr when absent or when this is not an object. */
    [[nodiscard]] const Json* find(std::string_view key) const;

    [[nodiscard]] const std::vector<Json>& elements() const noexcept { return array_; }
    [[nodiscard]] const std::map<std::string, Json, std::less<>>& members() const noexcept { return object_; }

    [[nodiscard]] const std::string& string_value() const noexcept { return string_; }
    [[nodiscard]] double number_value() const noexcept { return number_; }
    [[nodiscard]] bool boolean_value() const noexcept { return boolean_; }

    /**
     * Typed lookups that fall back to a default. A JSON null also yields the
     * fallback, which matters because DeepSeek configurations use null for
     * optional fields such as q_lora_rank.
     */
    [[nodiscard]] double number_or(std::string_view key, double fallback) const;
    [[nodiscard]] std::uint64_t unsigned_or(std::string_view key, std::uint64_t fallback) const;
    [[nodiscard]] bool boolean_or(std::string_view key, bool fallback) const;
    [[nodiscard]] std::string string_or(std::string_view key, std::string_view fallback) const;

private:
    Kind kind_{Kind::Null};
    bool boolean_{};
    double number_{};
    std::string string_;
    std::vector<Json> array_;
    std::map<std::string, Json, std::less<>> object_;

    friend class JsonParser;
};

}  // namespace litemind
