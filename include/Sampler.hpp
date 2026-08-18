#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace litemind {

/** How the next token is drawn from the logit distribution. */
enum class SamplingMethod { Greedy, Temperature };

/** Decoding controls supplied on the command line. */
struct SamplingOptions final {
    SamplingMethod method{SamplingMethod::Greedy};
    float temperature{0.7F};
    std::size_t top_k{40U};     ///< Zero disables the top-k cut.
    float top_p{0.95F};         ///< One disables nucleus filtering.
    float repetition_penalty{1.0F};
    std::uint64_t seed{0U};     ///< Zero draws a seed from the system entropy source.
};

/**
 * @brief Converts decoder logits into the next token ID.
 *
 * Greedy decoding is the default because it is reproducible, which matters most
 * while a checkpoint is still being validated: the same prompt must give the
 * same continuation every run, or a regression cannot be told from sampling noise.
 */
class Sampler final {
public:
    Sampler() = default;
    explicit Sampler(const SamplingOptions& options);

    /** Draws the next token. history is used only by the repetition penalty. */
    [[nodiscard]] std::uint32_t next(std::span<const float> logits,
                                     std::span<const std::uint32_t> history);

    /** Argmax decoding, kept as a free-standing entry point for tests. */
    [[nodiscard]] static std::size_t select_next(std::span<const float> logits,
                                                  SamplingMethod method = SamplingMethod::Greedy);

    /** The seed actually used, so a run can be reproduced. */
    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

private:
    SamplingOptions options_{};
    std::uint64_t seed_{};
    std::mt19937_64 generator_{};
    std::vector<std::pair<float, std::uint32_t>> candidates_;
};

}  // namespace litemind
