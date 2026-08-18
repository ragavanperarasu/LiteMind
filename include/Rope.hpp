#pragma once

#include "Config.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace litemind {

/**
 * @brief DeepSeek-V2's YaRN-scaled rotary position embedding.
 *
 * Two details here are easy to get wrong and both destroy output quality
 * silently rather than loudly:
 *
 * 1. DeepSeek permutes the rope channels before rotating them. The reference
 *    implementation reshapes [d] to [d/2, 2] and transposes, so channel pairs
 *    stored interleaved as (a0, b0, a1, b1, ...) become (a0, a1, ..., b0, b1, ...).
 *    Rotating the raw interleaved layout pairs the wrong channels together.
 *    The permuted layout is kept afterwards, which is safe because queries and
 *    keys are permuted identically and their dot product is unchanged by it.
 *
 * 2. The frequencies are YaRN-interpolated, not plain inverse powers of theta.
 *    Low-frequency channels are divided by the scaling factor, high-frequency
 *    channels are left alone, and a linear ramp blends the band between.
 */
class RotaryEmbedding final {
public:
    RotaryEmbedding() = default;

    /** Builds cosine and sine tables for positions [0, max_positions). */
    void build(const Config& config, std::size_t max_positions);

    /**
     * Rotates one head's rope channels in place. The span must hold exactly
     * qk_rope_head_dim values, and position must be below max_positions().
     */
    void apply(float* values, std::size_t position) const noexcept;

    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t max_positions() const noexcept { return max_positions_; }

    /** A one-line description of the frequency schedule, for the console. */
    [[nodiscard]] std::string summary() const;

private:
    std::size_t dimension_{};
    std::size_t half_{};
    std::size_t max_positions_{};
    double magnitude_scale_{1.0};
    bool yarn_{false};
    std::vector<float> cosine_;  ///< [max_positions_ * half_]
    std::vector<float> sine_;    ///< [max_positions_ * half_]
    std::vector<double> inverse_frequency_;
};

}  // namespace litemind
