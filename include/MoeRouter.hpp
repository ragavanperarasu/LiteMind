#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace litemind {

/** One routed-expert decision for a token in an MoE layer. */
struct ExpertSelection final {
    std::size_t expert_index{};
    float weight{};
};

/**
 * @brief Selects the highest-scoring experts from a router-logit vector.
 *
 * DeepSeek-V2 uses softmax scores and selects six routed experts per token.
 * This component is independent of the gate projection that produces logits.
 */
class MoeRouter final {
public:
    [[nodiscard]] static std::vector<ExpertSelection> select_top_k(
        std::span<const float> gate_logits, std::size_t expert_count, std::size_t top_k,
        bool normalize_selected_weights);
};

}  // namespace litemind
