#include "MoeRouter.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace litemind {

std::vector<ExpertSelection> MoeRouter::select_top_k(const std::span<const float> gate_logits,
                                                      const std::size_t expert_count,
                                                      const std::size_t top_k,
                                                      const bool normalize_selected_weights) {
    if (expert_count == 0U || top_k == 0U || top_k > expert_count || gate_logits.size() != expert_count) {
        throw std::invalid_argument("MoE routing requires one logit per expert and a valid top-k value.");
    }

    const float maximum = *std::max_element(gate_logits.begin(), gate_logits.end());
    std::vector<float> scores(expert_count);
    float denominator{};
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        scores[expert] = std::exp(gate_logits[expert] - maximum);
        denominator += scores[expert];
    }
    for (float& score : scores) {
        score /= denominator;
    }

    std::vector<std::size_t> indices(expert_count);
    std::iota(indices.begin(), indices.end(), 0U);
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(top_k), indices.end(),
                      [&scores](const std::size_t left, const std::size_t right) {
                          return scores[left] > scores[right];
                      });

    std::vector<ExpertSelection> selection;
    selection.reserve(top_k);
    float selected_sum{};
    for (std::size_t index = 0; index < top_k; ++index) {
        const std::size_t expert = indices[index];
        selection.push_back({expert, scores[expert]});
        selected_sum += scores[expert];
    }
    if (normalize_selected_weights) {
        for (ExpertSelection& item : selection) {
            item.weight /= selected_sum;
        }
    }
    return selection;
}

}  // namespace litemind
