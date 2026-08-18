#include "MoeRouter.hpp"
#include "Sampler.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace test_support;

namespace {

[[nodiscard]] litemind::Config lite_routing() {
    litemind::Config config;
    config.n_routed_experts = 8U;
    config.num_experts_per_tok = 3U;
    config.norm_topk_prob = false;      // DeepSeek-V2-Lite leaves weights unnormalised.
    config.routed_scaling_factor = 1.0F;
    config.topk_method = "greedy";
    config.n_group = 1U;
    config.topk_group = 1U;
    return config;
}

}  // namespace

int main() {
    // ── Greedy top-k over a full softmax ────────────────────────────────────
    const std::vector<float> logits{0.0F, 5.0F, 1.0F, 4.0F, 2.0F, 3.0F, -1.0F, -2.0F};
    const litemind::MoeRouter router(lite_routing());
    const auto selection = router.select(logits);

    check(selection.size() == 3U, "three experts are selected");
    check(selection[0].expert_index == 1U, "the highest logit is chosen first");
    check(selection[1].expert_index == 3U, "the second highest follows");
    check(selection[2].expert_index == 5U, "the third highest follows");
    check(selection[0].weight > selection[1].weight, "weights follow the score order");

    // The softmax runs over all eight experts before the cut, so the three
    // selected weights must sum to less than one. Softmaxing after the cut
    // would make them sum to exactly one and inflate the layer's output.
    const double total = selection[0].weight + selection[1].weight + selection[2].weight;
    check(total < 1.0, "unnormalised weights sum to less than one");
    check(total > 0.5, "the selected experts still carry most of the mass");

    // The weight must equal the full softmax probability of that expert.
    double denominator = 0.0;
    for (const float logit : logits) {
        denominator += std::exp(static_cast<double>(logit) - 5.0);
    }
    check_close(selection[0].weight, std::exp(0.0) / denominator, 1e-6,
                "the top weight is the full-softmax probability");

    // ── Normalisation, when a configuration asks for it ─────────────────────
    litemind::Config normalising = lite_routing();
    normalising.norm_topk_prob = true;
    const auto normalised = litemind::MoeRouter(normalising).select(logits);
    const double normalised_total =
        normalised[0].weight + normalised[1].weight + normalised[2].weight;
    check_close(normalised_total, 1.0, 1e-5, "normalised weights sum to one");

    // ── routed_scaling_factor applies only when weights are left unnormalised ─
    litemind::Config scaled = lite_routing();
    scaled.routed_scaling_factor = 2.0F;
    const auto scaled_selection = litemind::MoeRouter(scaled).select(logits);
    check_close(scaled_selection[0].weight, selection[0].weight * 2.0, 1e-6,
                "the scaling factor multiplies unnormalised weights");

    litemind::Config both = lite_routing();
    both.norm_topk_prob = true;
    both.routed_scaling_factor = 2.0F;
    const auto both_selection = litemind::MoeRouter(both).select(logits);
    const double both_total = both_selection[0].weight + both_selection[1].weight
                            + both_selection[2].weight;
    check_close(both_total, 1.0, 1e-5, "normalisation wins over the scaling factor");

    // ── Group-limited routing keeps its choices inside the selected groups ──
    litemind::Config grouped = lite_routing();
    grouped.topk_method = "group_limited_greedy";
    grouped.n_group = 2U;      // experts 0-3 and 4-7
    grouped.topk_group = 1U;   // only the stronger group may be used
    grouped.num_experts_per_tok = 2U;
    const auto grouped_selection = litemind::MoeRouter(grouped).select(logits);
    check(grouped_selection.size() == 2U, "group-limited routing still selects top-k");
    for (const auto& choice : grouped_selection) {
        check(choice.expert_index < 4U, "the chosen experts stay in the winning group");
    }

    // ── The legacy entry point still behaves ────────────────────────────────
    const auto legacy = litemind::MoeRouter::select_top_k(logits, 8U, 3U, true);
    check_close(legacy[0].weight + legacy[1].weight + legacy[2].weight, 1.0, 1e-5,
                "select_top_k normalises when asked");

    // ── Sampling ────────────────────────────────────────────────────────────
    const std::vector<float> vocabulary_logits{0.1F, 9.0F, 0.2F, 0.3F};
    check(litemind::Sampler::select_next(vocabulary_logits) == 1U, "argmax finds the peak");

    // Greedy decoding must be reproducible: the same logits, the same token.
    litemind::SamplingOptions greedy;
    greedy.method = litemind::SamplingMethod::Greedy;
    litemind::Sampler first(greedy);
    litemind::Sampler second(greedy);
    check(first.next(vocabulary_logits, {}) == second.next(vocabulary_logits, {}),
          "greedy decoding is deterministic");

    // Seeded sampling must also be reproducible.
    litemind::SamplingOptions sampled;
    sampled.method = litemind::SamplingMethod::Temperature;
    sampled.temperature = 1.0F;
    sampled.top_k = 4U;
    sampled.top_p = 1.0F;
    sampled.seed = 42U;
    litemind::Sampler seeded_a(sampled);
    litemind::Sampler seeded_b(sampled);
    bool identical = true;
    for (int trial = 0; trial < 20; ++trial) {
        identical = identical && seeded_a.next(vocabulary_logits, {}) == seeded_b.next(vocabulary_logits, {});
    }
    check(identical, "a seeded sampler repeats its sequence");

    // A very low temperature must collapse onto the argmax.
    litemind::SamplingOptions cold = sampled;
    cold.temperature = 0.01F;
    litemind::Sampler cold_sampler(cold);
    check(cold_sampler.next(vocabulary_logits, {}) == 1U, "a cold sampler picks the peak");

    // The repetition penalty must push an already-used token down.
    litemind::SamplingOptions penalised;
    penalised.method = litemind::SamplingMethod::Greedy;
    penalised.repetition_penalty = 100.0F;
    litemind::Sampler penalising(penalised);
    const std::vector<std::uint32_t> history{1U};
    check(penalising.next(vocabulary_logits, history) != 1U,
          "the repetition penalty demotes a repeated token");

    return report("MoeRouterSamplerTest");
}
