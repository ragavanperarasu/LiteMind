#pragma once

#include "Tensor.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace litemind {

/**
 * @brief A contiguous float32 CPU tensor used by the first-principles runtime.
 *
 * Model weights are BF16 on disk but are widened to float32 for the initial
 * portable CPU implementation. Optimised or quantized backends can later sit
 * behind the same operation boundaries.
 */
class CpuTensor final {
public:
    explicit CpuTensor(std::vector<std::size_t> shape);
    CpuTensor(std::vector<std::size_t> shape, std::vector<float> values);

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;
    [[nodiscard]] std::size_t element_count() const noexcept;
    [[nodiscard]] std::span<const float> values() const noexcept;
    [[nodiscard]] std::span<float> values() noexcept;

    /** Adds another tensor with identical shape in place. */
    void add_inplace(const CpuTensor& other);
    /** Applies the SiLU activation in place. */
    void silu_inplace() noexcept;
    /** Applies RMS normalisation over the final dimension in place. */
    void rms_norm_inplace(std::span<const float> weight, float epsilon);
    /** Applies softmax over the final dimension in place. */
    void softmax_last_dimension_inplace();

    /** Computes a rank-2 matrix product: [M, K] × [K, N] = [M, N]. */
    [[nodiscard]] static CpuTensor matmul(const CpuTensor& left, const CpuTensor& right);
    [[nodiscard]] static std::size_t argmax(std::span<const float> values);

private:
    [[nodiscard]] static std::size_t checked_element_count(std::span<const std::size_t> shape);

    std::vector<std::size_t> shape_;
    std::vector<float> values_;
};

}  // namespace litemind
