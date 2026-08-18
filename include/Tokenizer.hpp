#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace litemind {

/**
 * @brief Byte-level BPE tokenizer for the bundled DeepSeek-V2 tokenizer.json.
 *
 * The implementation reads the tokenizer's vocabulary and merge table directly;
 * it does not depend on Python, Hugging Face, or any inference library.
 */
class Tokenizer final {
public:
    [[nodiscard]] bool load(const std::filesystem::path& tokenizer_path, std::string& error);
    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text,
                                                     bool add_bos = true) const;
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& token_ids) const;
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept;

private:
    [[nodiscard]] std::vector<std::string> pre_tokenize(std::string_view text) const;
    [[nodiscard]] std::vector<std::string> byte_encode(std::string_view text) const;
    [[nodiscard]] std::vector<std::string> apply_bpe(std::vector<std::string> symbols) const;

    std::unordered_map<std::string, std::uint32_t> vocabulary_;
    std::unordered_map<std::string, std::uint32_t> merge_ranks_;
    std::vector<std::string> reverse_vocabulary_;
};

}  // namespace litemind
