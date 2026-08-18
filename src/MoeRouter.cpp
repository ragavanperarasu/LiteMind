#include "MoeRouter.hpp"

#include "Gemm.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace litemind {
namespace {

/** Selects the top_k highest-scoring entries of scores, ignoring masked ones. */
void top_k_indices(const std::vector<float>& scores, const std::vector<bool>& allowed,
                   const std::size_t top_k, std::vector<std::size_t>& order) {
    order.resize(scores.size());
    std::iota(order.begin(), order.end(), 0U);

    const auto middle = order.begin() + static_cast<std::ptrdiff_t>(std::min(top_k, order.size()));
    std::partial_sort(order.begin(), middle, order.end(),
                      [&scores, &allowed](const std::size_t left, const std::size_t right) {
                          const bool left_allowed = allowed.empty() || allowed[left];
                          const bool right_allowed = allowed.empty() || allowed[right];
                          if (left_allowed != right_allowed) {
                              return left_allowed;
                          }
                          if (scores[left] != scores[right]) {
                              return scores[left] > scores[right];
                          }
                          // A stable tie-break keeps routing reproducible run to run.
                          return left < right;
                      });
    order.resize(std::min(top_k, order.size()));
}

}  // namespace

MoeRouter::MoeRouter(const Config& config)
    : expert_count_(config.n_routed_experts),
      top_k_(config.num_experts_per_tok),
      n_group_(std::max<std::size_t>(config.n_group, 1U)),
      topk_group_(std::max<std::size_t>(config.topk_group, 1U)),
      group_limited_(config.topk_method == "group_limited_greedy"),
      normalize_(config.norm_topk_prob),
      routed_scaling_factor_(config.routed_scaling_factor) {
    if (expert_count_ % n_group_ != 0U) {
        // A group that does not divide the experts cannot be masked coherently,
        // so fall back to plain greedy routing rather than mis-routing silently.
        group_limited_ = false;
        n_group_ = 1U;
    }
}

void MoeRouter::select(const std::span<const float> gate_logits,
                       std::vector<ExpertSelection>& selection) const {
    selection.clear();
    if (gate_logits.size() != expert_count_ || top_k_ == 0U || expert_count_ == 0U) {
        return;
    }

    // Softmax over every expert, then cut. Taking the softmax after the cut
    // would renormalise the weights and change the residual's magnitude.
    scores_.assign(gate_logits.begin(), gate_logits.end());
    gemm::softmax(scores_.data(), scores_.size());

    std::vector<bool> allowed;
    if (group_limited_ && n_group_ > 1U) {
        const std::size_t experts_per_group = expert_count_ / n_group_;
        group_scores_.assign(n_group_, -std::numeric_limits<float>::infinity());
        for (std::size_t group = 0; group < n_group_; ++group) {
            const auto first = scores_.begin() + static_cast<std::ptrdiff_t>(group * experts_per_group);
            group_scores_[group] = *std::max_element(first, first + static_cast<std::ptrdiff_t>(experts_per_group));
        }

        std::vector<std::size_t> group_order;
        top_k_indices(group_scores_, {}, std::min(topk_group_, n_group_), group_order);

        allowed.assign(expert_count_, false);
        for (const std::size_t group : group_order) {
            for (std::size_t offset = 0; offset < experts_per_group; ++offset) {
                allowed[group * experts_per_group + offset] = true;
            }
        }
    }

    std::vector<std::size_t> order;
    top_k_indices(scores_, allowed, top_k_, order);

    selection.reserve(order.size());
    float selected_sum = 0.0F;
    for (const std::size_t expert : order) {
        selection.push_back(ExpertSelection{expert, scores_[expert]});
        selected_sum += scores_[expert];
    }

    // DeepSeek normalises the selected weights or scales them, never both.
    if (top_k_ > 1U && normalize_) {
        const float denominator = selected_sum + 1e-20F;
        for (ExpertSelection& item : selection) {
            item.weight /= denominator;
        }
    } else if (routed_scaling_factor_ != 1.0F) {
        for (ExpertSelection& item : selection) {
            item.weight *= routed_scaling_factor_;
        }
    }
}

std::vector<ExpertSelection> MoeRouter::select(const std::span<const float> gate_logits) const {
    std::vector<ExpertSelection> selection;
    select(gate_logits, selection);
    return selection;
}

std::vector<ExpertSelection> MoeRouter::select_top_k(const std::span<const float> gate_logits,
                                                      const std::size_t expert_count,
                                                      const std::size_t top_k,
                                                      const bool normalize_selected_weights) {
    if (expert_count == 0U || top_k == 0U || top_k > expert_count || gate_logits.size() != expert_count) {
        throw std::invalid_argument("MoE routing requires one logit per expert and a valid top-k value.");
    }

    Config config;
    config.n_routed_experts = expert_count;
    config.num_experts_per_tok = top_k;
    config.norm_topk_prob = normalize_selected_weights;
    config.routed_scaling_factor = 1.0F;
    config.topk_method = "greedy";
    config.n_group = 1U;
    config.topk_group = 1U;

    return MoeRouter(config).select(gate_logits);
}

}  // namespace litemind
