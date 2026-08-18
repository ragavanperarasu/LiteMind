#include "MoeRouter.hpp"
#include "Sampler.hpp"

#include <array>
#include <cmath>
#include <iostream>

int main() {
    constexpr std::array<float, 4U> gate_logits{0.0F, 3.0F, 1.0F, 2.0F};
    const auto experts = litemind::MoeRouter::select_top_k(gate_logits, 4U, 2U, true);
    if (experts.size() != 2U || experts[0].expert_index != 1U || experts[1].expert_index != 3U
        || std::fabs(experts[0].weight + experts[1].weight - 1.0F) > 0.0001F) {
        std::cerr << "MoE top-k routing produced unexpected selections.\n";
        return 1;
    }

    constexpr std::array<float, 4U> logits{-2.0F, 5.0F, 4.0F, 1.0F};
    if (litemind::Sampler::select_next(logits) != 1U) {
        std::cerr << "Greedy sampling selected an unexpected token.\n";
        return 1;
    }
    return 0;
}
