#pragma once

#include "Attention.hpp"
#include "Config.hpp"
#include "ExpertCache.hpp"
#include "KvCache.hpp"
#include "MoeRouter.hpp"
#include "Rope.hpp"
#include "Sampler.hpp"
#include "Threading.hpp"
#include "Tokenizer.hpp"
#include "WeightStore.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace litemind {

/** How the runtime is set up before any prompt is seen. */
struct RunnerOptions final {
    std::size_t threads{0U};              ///< Zero means one per hardware thread.
    std::size_t context_length{1024U};    ///< Prompt plus generated tokens.
    std::uint64_t expert_budget_bytes{0U};///< Zero leaves expert residency to the OS.
    bool warm_start{false};               ///< Pre-stream the always-hot weights at load.
    bool verbose{true};
};

/** How one generation call behaves. */
struct GenerationOptions final {
    std::size_t max_new_tokens{128U};
    SamplingOptions sampling{};
    /** Called with each newly decoded fragment of text, for live output. */
    std::function<void(std::string_view)> on_text;

    /**
     * Text that ends the reply as soon as the model produces it.
     *
     * An instruction-tuned checkpoint has no reason to stop after answering; the
     * frame it was trained on simply continues into the next turn, so left alone
     * it plays both sides of the conversation. The marker that opens that next
     * turn is where its answer actually ends.
     *
     * Matched text is removed from the result and never reaches on_text, and a
     * fragment that might still grow into a match is held back until it is
     * settled either way.
     */
    std::vector<std::string> stop_sequences;
    bool show_progress{true};

    /**
     * When non-zero, record the N highest logits produced for the final prompt
     * token. These are the model's raw prediction before any sampling, so they
     * can be compared directly against a reference implementation to tell a
     * loading or arithmetic fault from a sampling one.
     */
    std::size_t top_logits{0U};
};

/** What one generation call produced. */
struct GenerationResult final {
    std::string text;
    std::vector<std::uint32_t> tokens;
    std::size_t prompt_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    std::string stop_reason{"length"};

    /** The highest logits after prefill, most likely first. See top_logits. */
    std::vector<std::pair<std::uint32_t, float>> prompt_top_logits;

    [[nodiscard]] double tokens_per_second() const noexcept {
        return decode_seconds > 0.0 ? static_cast<double>(tokens.size()) / decode_seconds : 0.0;
    }
};

/**
 * @brief End-to-end DeepSeek-V2 decoding on the CPU, streaming weights from SSD.
 *
 * Loading maps every shard and resolves each weight to a pointer inside its
 * mapping; no weight is copied into RAM except the small norm vectors and the
 * router gates. A decoding step touches the embedding row for one token, every
 * attention matrix, and only the experts the router selected, so resident
 * memory tracks the working set rather than the 31 GB checkpoint.
 */
class DeepSeekRunner final {
public:
    DeepSeekRunner();
    ~DeepSeekRunner();

    DeepSeekRunner(const DeepSeekRunner&) = delete;
    DeepSeekRunner& operator=(const DeepSeekRunner&) = delete;

    /**
     * Reads config.json and tokenizer.json, maps every shard and resolves every
     * weight the architecture needs. Returns false with a specific message when
     * the checkpoint and the configuration disagree.
     */
    [[nodiscard]] bool load(const std::filesystem::path& model_directory, const RunnerOptions& options,
                            std::string& error);

    [[nodiscard]] bool ready() const noexcept;

    [[nodiscard]] const Config& config() const noexcept;
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept;
    [[nodiscard]] const WeightStore& weights() const noexcept;
    [[nodiscard]] const ExpertCache& expert_cache() const noexcept;

    /** Clears conversation state so the next prompt starts from position zero. */
    void reset();

    /** Runs prefill and then decoding. Returns false and sets error on failure. */
    [[nodiscard]] bool generate(const std::vector<std::uint32_t>& prompt_tokens,
                                const GenerationOptions& options, GenerationResult& result,
                                std::string& error);

    /** A multi-line report of memory layout and residency, for the console. */
    [[nodiscard]] std::string memory_report() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace litemind
