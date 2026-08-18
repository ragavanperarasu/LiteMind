#include "Sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litemind {

Sampler::Sampler(const SamplingOptions& options) : options_(options) {
    seed_ = options_.seed;
    if (seed_ == 0U) {
        std::random_device device;
        seed_ = (static_cast<std::uint64_t>(device()) << 32U) | device();
    }
    generator_.seed(seed_);
}

std::uint32_t Sampler::next(const std::span<const float> logits,
                            const std::span<const std::uint32_t> history) {
    if (logits.empty()) {
        throw std::invalid_argument("Sampling requires at least one vocabulary logit.");
    }

    if (options_.method == SamplingMethod::Greedy && options_.repetition_penalty == 1.0F) {
        return static_cast<std::uint32_t>(select_next(logits, SamplingMethod::Greedy));
    }

    candidates_.resize(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
        candidates_[index] = {logits[index], static_cast<std::uint32_t>(index)};
    }

    // Penalise tokens already produced. Dividing a positive logit and
    // multiplying a negative one both push the value down, which is the
    // convention the penalty was defined with.
    if (options_.repetition_penalty != 1.0F) {
        for (const std::uint32_t token : history) {
            if (token < candidates_.size()) {
                float& value = candidates_[token].first;
                value = value > 0.0F ? value / options_.repetition_penalty
                                     : value * options_.repetition_penalty;
            }
        }
    }

    if (options_.method == SamplingMethod::Greedy) {
        const auto best = std::max_element(candidates_.begin(), candidates_.end(),
                                           [](const auto& left, const auto& right) {
                                               return left.first < right.first;
                                           });
        return best->second;
    }

    const float temperature = std::max(options_.temperature, 1e-4F);
    for (auto& candidate : candidates_) {
        candidate.first /= temperature;
    }

    // Keep only the top-k logits before the softmax so the exponentials are
    // computed over a short list rather than the whole 102k-entry vocabulary.
    std::size_t keep = candidates_.size();
    if (options_.top_k != 0U && options_.top_k < keep) {
        keep = options_.top_k;
        std::partial_sort(candidates_.begin(),
                          candidates_.begin() + static_cast<std::ptrdiff_t>(keep), candidates_.end(),
                          [](const auto& left, const auto& right) { return left.first > right.first; });
        candidates_.resize(keep);
    } else {
        std::sort(candidates_.begin(), candidates_.end(),
                  [](const auto& left, const auto& right) { return left.first > right.first; });
    }

    const float maximum = candidates_.front().first;
    float total = 0.0F;
    for (auto& candidate : candidates_) {
        candidate.first = std::exp(candidate.first - maximum);
        total += candidate.first;
    }
    for (auto& candidate : candidates_) {
        candidate.first /= total;
    }

    // Nucleus filtering: keep the shortest prefix whose mass reaches top_p.
    if (options_.top_p < 1.0F) {
        float cumulative = 0.0F;
        std::size_t cutoff = 0U;
        for (; cutoff < candidates_.size(); ++cutoff) {
            cumulative += candidates_[cutoff].first;
            if (cumulative >= options_.top_p) {
                ++cutoff;
                break;
            }
        }
        candidates_.resize(std::max<std::size_t>(cutoff, 1U));

        float remaining = 0.0F;
        for (const auto& candidate : candidates_) {
            remaining += candidate.first;
        }
        for (auto& candidate : candidates_) {
            candidate.first /= remaining;
        }
    }

    std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
    float target = distribution(generator_);
    for (const auto& candidate : candidates_) {
        target -= candidate.first;
        if (target <= 0.0F) {
            return candidate.second;
        }
    }
    return candidates_.back().second;
}

std::size_t Sampler::select_next(const std::span<const float> logits, const SamplingMethod method) {
    if (logits.empty()) {
        throw std::invalid_argument("Sampling requires at least one vocabulary logit.");
    }
    if (method != SamplingMethod::Greedy) {
        throw std::invalid_argument("Only greedy decoding is available without a Sampler instance.");
    }
    return static_cast<std::size_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

}  // namespace litemind
