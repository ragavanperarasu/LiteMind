#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace litemind {

/**
 * @brief The byte-level BPE tokenizer described by a model's tokenizer.json.
 *
 * Reads the vocabulary, the merge table and the added special tokens directly,
 * with no dependency on Python or Hugging Face. Every byte maps to a printable
 * code point first, so arbitrary binary input round-trips and no token is ever
 * unknown: worst case a piece falls back to its 256 single-byte tokens.
 */
class Tokenizer final {
public:
    [[nodiscard]] bool load(const std::filesystem::path& path, std::string& error);

    /** Encodes text, optionally prefixing the beginning-of-sequence token. */
    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text, bool add_bos = true) const;

    /** Decodes a token sequence back to UTF-8 text. */
    [[nodiscard]] std::string decode(std::span<const std::uint32_t> token_ids) const;

    [[nodiscard]] bool ready() const noexcept { return !vocabulary_.empty(); }
    [[nodiscard]] std::size_t vocabulary_size() const noexcept { return reverse_vocabulary_.size(); }

    [[nodiscard]] std::uint32_t bos_token_id() const noexcept { return bos_token_id_; }
    [[nodiscard]] std::uint32_t eos_token_id() const noexcept { return eos_token_id_; }
    [[nodiscard]] bool has_bos() const noexcept { return has_bos_; }

    /**
     * The chat_template stored in tokenizer_config.json, empty on a base
     * checkpoint. Its presence is what distinguishes an instruction-tuned
     * checkpoint from the base model it was fine-tuned from.
     */
    [[nodiscard]] const std::string& chat_template() const noexcept { return chat_template_; }

    /** True when token_id is an added control token rather than ordinary text. */
    [[nodiscard]] bool is_special(std::uint32_t token_id) const;

    /** The token's byte-level spelling, for inspecting a tokenisation. */
    [[nodiscard]] std::string token_text(std::uint32_t token_id) const;

    /** The path this tokenizer was loaded from. */
    [[nodiscard]] const std::filesystem::path& source_path() const noexcept { return source_path_; }

    /**
     * @brief Emits decoded text incrementally without splitting a UTF-8 character.
     *
     * Byte-level BPE happily puts the first byte of a multi-byte character in
     * one token and the rest in the next. Printing each token as it arrives
     * would write invalid UTF-8 to the console, so partial sequences are held
     * back until they complete.
     */
    class StreamDecoder final {
    public:
        explicit StreamDecoder(const Tokenizer& tokenizer) : tokenizer_(&tokenizer) {}

        /** Returns the text that became printable after adding token_id. */
        [[nodiscard]] std::string push(std::uint32_t token_id);

        /** Returns any held-back bytes at the end of a generation. */
        [[nodiscard]] std::string flush();

    private:
        const Tokenizer* tokenizer_;
        std::string pending_;
    };

private:
    [[nodiscard]] bool load_vocabulary(const class Json& document, std::string& error);
    [[nodiscard]] bool load_merges(const class Json& document, std::string& error);
    void load_added_tokens(const class Json& document);
    void load_tokenizer_config(const std::filesystem::path& directory);

    [[nodiscard]] std::vector<std::string> pre_tokenize(std::string_view text) const;
    [[nodiscard]] std::string byte_encode(std::string_view text) const;
    void apply_bpe(const std::string& piece, std::vector<std::uint32_t>& output) const;
    void encode_ordinary(std::string_view text, std::vector<std::uint32_t>& output) const;

    std::filesystem::path source_path_;
    std::unordered_map<std::string, std::uint32_t> vocabulary_;
    std::unordered_map<std::string, std::uint32_t> merge_ranks_;
    std::vector<std::string> reverse_vocabulary_;
    std::vector<bool> special_flags_;

    /** Added tokens matched literally in the input, longest spelling first. */
    std::vector<std::pair<std::string, std::uint32_t>> added_tokens_;

    std::string chat_template_;
    std::uint32_t bos_token_id_{};
    std::uint32_t eos_token_id_{};
    bool has_bos_{false};
};

}  // namespace litemind
