#pragma once

#include "Config.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace litemind {

/** One routed-expert decision for a token in a mixture-of-experts layer. */
struct ExpertSelection final {
    std::size_t expert_index{};
    float weight{};
};

/**
 * @brief Turns router logits into the experts a token is sent to.
 *
 * This follows DeepSeek-V2's gate exactly, including two details that are easy
 * to invert: the softmax runs over all experts before the top-k cut, not after
 * it, and routed_scaling_factor is applied only when the selected weights are
 * left unnormalised. DeepSeek-V2-Lite sets norm_topk_prob to false and the
 * scaling factor to 1.0, so its weights are raw softmax probabilities that do
 * not sum to one.
 */
class MoeRouter final {
public:
    MoeRouter() = default;
    explicit MoeRouter(const Config& config);

    /** Selects experts for one token from its router logits. */
    void select(std::span<const float> gate_logits, std::vector<ExpertSelection>& selection) const;

    /** Convenience overload that returns a fresh vector. */
    [[nodiscard]] std::vector<ExpertSelection> select(std::span<const float> gate_logits) const;

    /** The original interface, kept for tests that exercise routing directly. */
    [[nodiscard]] static std::vector<ExpertSelection> select_top_k(std::span<const float> gate_logits,
                                                                   std::size_t expert_count,
                                                                   std::size_t top_k,
                                                                   bool normalize_selected_weights);

private:
    std::size_t expert_count_{64U};
    std::size_t top_k_{6U};
    std::size_t n_group_{1U};
    std::size_t topk_group_{1U};
    bool group_limited_{false};
    bool normalize_{false};
    float routed_scaling_factor_{1.0F};

    mutable std::vector<float> scores_;
    mutable std::vector<float> group_scores_;
};

}  // namespace litemind
