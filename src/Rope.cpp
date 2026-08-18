#include "Rope.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace litemind {
namespace {

constexpr double pi = 3.14159265358979323846;

/** yarn_get_mscale: the amplitude correction YaRN applies at a given scale. */
[[nodiscard]] double yarn_get_mscale(const double scale, const double mscale) noexcept {
    if (scale <= 1.0) {
        return 1.0;
    }
    return 0.1 * mscale * std::log(scale) + 1.0;
}

/**
 * yarn_find_correction_dim: the channel index whose wavelength completes
 * num_rotations turns across the original training context.
 */
[[nodiscard]] double correction_dimension(const double num_rotations, const std::size_t dimension,
                                          const double base,
                                          const std::size_t original_max_positions) noexcept {
    const double numerator =
        static_cast<double>(dimension)
        * std::log(static_cast<double>(original_max_positions) / (num_rotations * 2.0 * pi));
    return numerator / (2.0 * std::log(base));
}

}  // namespace

void RotaryEmbedding::build(const Config& config, const std::size_t max_positions) {
    dimension_ = config.qk_rope_head_dim;
    half_ = dimension_ / 2U;
    max_positions_ = std::max<std::size_t>(max_positions, 1U);
    yarn_ = config.rope_scaling.enabled;

    const double base = static_cast<double>(config.rope_theta);
    inverse_frequency_.assign(half_, 0.0);

    if (!yarn_) {
        for (std::size_t index = 0; index < half_; ++index) {
            const double exponent = 2.0 * static_cast<double>(index) / static_cast<double>(dimension_);
            inverse_frequency_[index] = 1.0 / std::pow(base, exponent);
        }
        magnitude_scale_ = 1.0;
    } else {
        const RopeScaling& scaling = config.rope_scaling;
        const double factor = scaling.factor <= 0.0 ? 1.0 : scaling.factor;

        // The band of channels that YaRN blends, in channel-index units.
        double low = std::floor(
            correction_dimension(scaling.beta_fast, dimension_, base, scaling.original_max_position_embeddings));
        double high = std::ceil(
            correction_dimension(scaling.beta_slow, dimension_, base, scaling.original_max_position_embeddings));
        low = std::max(low, 0.0);
        high = std::min(high, static_cast<double>(dimension_) - 1.0);
        if (high - low < 1e-6) {
            high = low + 0.001;  // Matches the reference guard against a zero-width ramp.
        }

        for (std::size_t index = 0; index < half_; ++index) {
            const double exponent = 2.0 * static_cast<double>(index) / static_cast<double>(dimension_);
            const double frequency_extrapolation = 1.0 / std::pow(base, exponent);
            const double frequency_interpolation = frequency_extrapolation / factor;

            const double ramp = std::clamp((static_cast<double>(index) - low) / (high - low), 0.0, 1.0);
            const double mask = 1.0 - ramp;  // 1 keeps the original frequency, 0 uses the scaled one.
            inverse_frequency_[index] =
                frequency_interpolation * (1.0 - mask) + frequency_extrapolation * mask;
        }

        // Both halves of this ratio are 1.0 when mscale equals mscale_all_dim,
        // which is the case for DeepSeek-V2-Lite. The attention softmax scale in
        // Config carries the amplitude correction instead.
        magnitude_scale_ = yarn_get_mscale(factor, scaling.mscale)
                         / yarn_get_mscale(factor, scaling.mscale_all_dim);
    }

    cosine_.assign(max_positions_ * half_, 0.0F);
    sine_.assign(max_positions_ * half_, 0.0F);
    for (std::size_t position = 0; position < max_positions_; ++position) {
        float* const cosine_row = cosine_.data() + position * half_;
        float* const sine_row = sine_.data() + position * half_;
        for (std::size_t index = 0; index < half_; ++index) {
            const double angle = static_cast<double>(position) * inverse_frequency_[index];
            cosine_row[index] = static_cast<float>(std::cos(angle) * magnitude_scale_);
            sine_row[index] = static_cast<float>(std::sin(angle) * magnitude_scale_);
        }
    }
}

void RotaryEmbedding::apply(float* const values, const std::size_t position) const noexcept {
    if (values == nullptr || half_ == 0U || position >= max_positions_) {
        return;
    }
    const float* const cosine_row = cosine_.data() + position * half_;
    const float* const sine_row = sine_.data() + position * half_;

    // The permutation cannot run in place: writing the second half of the
    // output would clobber interleaved inputs that later channels still read.
    // One scratch buffer per thread keeps this allocation-free after the first
    // call while staying safe if attention is ever parallelised over heads.
    thread_local std::vector<float> source;
    source.assign(values, values + dimension_);

    for (std::size_t index = 0; index < half_; ++index) {
        const float even = source[2U * index];
        const float odd = source[2U * index + 1U];
        const float cosine = cosine_row[index];
        const float sine = sine_row[index];
        values[index] = even * cosine - odd * sine;
        values[index + half_] = odd * cosine + even * sine;
    }
}

std::string RotaryEmbedding::summary() const {
    std::ostringstream text;
    text << "rope dim=" << dimension_ << " positions=" << max_positions_;
    if (yarn_) {
        text << " yarn (magnitude x" << magnitude_scale_ << ")";
    } else {
        text << " unscaled";
    }
    return text.str();
}

}  // namespace litemind
