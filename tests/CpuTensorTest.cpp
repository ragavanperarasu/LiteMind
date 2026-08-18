#include "CpuTensor.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

[[nodiscard]] bool approximately_equal(const float left, const float right) {
    return std::fabs(left - right) < 0.0001F;
}

}  // namespace

int main() {
    const litemind::CpuTensor left({2U, 3U}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const litemind::CpuTensor right({3U, 2U}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
    litemind::CpuTensor product = litemind::CpuTensor::matmul(left, right);
    const std::vector<float> expected_product{58.0F, 64.0F, 139.0F, 154.0F};
    if (!std::equal(product.values().begin(), product.values().end(), expected_product.begin())) {
        std::cerr << "Matrix multiplication produced unexpected values.\n";
        return 1;
    }

    product.softmax_last_dimension_inplace();
    for (std::size_t row = 0; row < 2U; ++row) {
        const auto values = product.values();
        const float sum = values[row * 2U] + values[row * 2U + 1U];
        if (!approximately_equal(sum, 1.0F)) {
            std::cerr << "Softmax row does not sum to one.\n";
            return 1;
        }
    }

    litemind::CpuTensor normalized({1U, 2U}, {3.0F, 4.0F});
    constexpr std::array<float, 2U> rms_weight{1.0F, 1.0F};
    normalized.rms_norm_inplace(rms_weight, 0.000001F);
    const auto normalized_values = normalized.values();
    if (!approximately_equal(normalized_values[0], 0.848528F)
        || !approximately_equal(normalized_values[1], 1.131370F)) {
        std::cerr << "RMSNorm produced unexpected values.\n";
        return 1;
    }

    constexpr std::array<float, 3U> logits{-1.0F, 4.0F, 2.0F};
    if (litemind::CpuTensor::argmax(logits) != 1U) {
        std::cerr << "Argmax produced an unexpected index.\n";
        return 1;
    }
    return 0;
}
