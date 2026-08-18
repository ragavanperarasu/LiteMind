#pragma once

#include "Tokenizer.hpp"

#include <filesystem>
#include <string>

namespace litemind {

/**
 * @brief End-to-end autoregressive decoder for DeepSeek-V2 (small).
 *
 * Loads all weights from the model directory's SafeTensors shards on
 * construction, then generates text token-by-token using the full
 * DeepSeek-V2 architecture: Multi-head Latent Attention with RoPE and
 * SwiGLU dense / Mixture-of-Experts feed-forward layers.
 */
class DeepSeekRunner final {
public:
    /**
     * Loads all model weights from model_directory.
     * Sets error and leaves the object invalid on failure.
     */
    explicit DeepSeekRunner(const std::filesystem::path& model_directory, std::string& error);

    /** Returns true if the model was loaded successfully. */
    [[nodiscard]] bool ready() const noexcept;

    /**
     * Generates up to max_new_tokens tokens following prompt_tokens.
     * Stops early on EOS. Streams each token to stdout as it is produced.
     * Returns the generated text (not including the prompt).
     */
    std::string generate(const Tokenizer& tokenizer,
                         const std::vector<std::uint32_t>& prompt_tokens,
                         std::size_t max_new_tokens = 200) const;

private:
    struct Impl;
    // Intentionally use a raw pointer to avoid incomplete-type issues with unique_ptr in headers.
    // Ownership is managed via constructor/destructor.
    Impl* impl_{nullptr};

public:
    ~DeepSeekRunner();
    DeepSeekRunner(const DeepSeekRunner&) = delete;
    DeepSeekRunner& operator=(const DeepSeekRunner&) = delete;
};

}  // namespace litemind
