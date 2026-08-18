#include "Attention.hpp"

#include <array>
#include <cmath>
#include <iostream>

int main() {
    litemind::KvCache cache(2U);
    constexpr std::array<float, 2U> first_key{1.0F, 0.0F};
    constexpr std::array<float, 2U> first_value{1.0F, 10.0F};
    constexpr std::array<float, 2U> second_key{0.0F, 1.0F};
    constexpr std::array<float, 2U> second_value{3.0F, 30.0F};
    constexpr std::array<float, 2U> query{1.0F, 0.0F};
    cache.append(first_key, first_value);
    cache.append(second_key, second_value);

    const auto output = litemind::Attention::attend(query, cache);
    if (output.size() != 2U || std::fabs(output[0] - 1.660476F) > 0.0001F
        || std::fabs(output[1] - 16.604765F) > 0.0001F) {
        std::cerr << "Causal attention output is incorrect.\n";
        return 1;
    }
    return 0;
}
