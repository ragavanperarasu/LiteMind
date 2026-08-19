#include "Gemm.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace litemind {
namespace gemm {
namespace {

/** Rows handled by one scheduling unit; large enough to amortise task overhead. */
constexpr std::size_t rows_per_task = 32U;

[[nodiscard]] inline float widen(const std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

#if defined(__AVX2__)
/** Horizontally sums the eight lanes of an AVX register. */
[[nodiscard]] inline float horizontal_sum(const __m256 vector) noexcept {
    const __m128 high = _mm256_extractf128_ps(vector, 1);
    const __m128 low = _mm256_castps256_ps128(vector);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    return _mm_cvtss_f32(sum);
}

/** Widens eight BF16 values to float32 by shifting them into the high half. */
[[nodiscard]] inline __m256 widen8(const std::uint16_t* source) noexcept {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
}
#endif

/**
 * One row of a BF16 matrix dotted with a float32 vector.
 *
 * Four independent accumulators hide the FMA latency, which otherwise
 * serialises this loop on every current x86 core.
 */
[[nodiscard]] float dot_bf16(const std::uint16_t* __restrict weights,
                             const float* __restrict input, const std::size_t count) noexcept {
#if defined(__AVX2__)
    __m256 accumulator0 = _mm256_setzero_ps();
    __m256 accumulator1 = _mm256_setzero_ps();
    __m256 accumulator2 = _mm256_setzero_ps();
    __m256 accumulator3 = _mm256_setzero_ps();

    std::size_t index = 0U;
    for (; index + 32U <= count; index += 32U) {
#if defined(__FMA__)
        accumulator0 = _mm256_fmadd_ps(widen8(weights + index), _mm256_loadu_ps(input + index), accumulator0);
        accumulator1 = _mm256_fmadd_ps(widen8(weights + index + 8U), _mm256_loadu_ps(input + index + 8U), accumulator1);
        accumulator2 = _mm256_fmadd_ps(widen8(weights + index + 16U), _mm256_loadu_ps(input + index + 16U), accumulator2);
        accumulator3 = _mm256_fmadd_ps(widen8(weights + index + 24U), _mm256_loadu_ps(input + index + 24U), accumulator3);
#else
        accumulator0 = _mm256_add_ps(accumulator0, _mm256_mul_ps(widen8(weights + index), _mm256_loadu_ps(input + index)));
        accumulator1 = _mm256_add_ps(accumulator1, _mm256_mul_ps(widen8(weights + index + 8U), _mm256_loadu_ps(input + index + 8U)));
        accumulator2 = _mm256_add_ps(accumulator2, _mm256_mul_ps(widen8(weights + index + 16U), _mm256_loadu_ps(input + index + 16U)));
        accumulator3 = _mm256_add_ps(accumulator3, _mm256_mul_ps(widen8(weights + index + 24U), _mm256_loadu_ps(input + index + 24U)));
#endif
    }
    for (; index + 8U <= count; index += 8U) {
#if defined(__FMA__)
        accumulator0 = _mm256_fmadd_ps(widen8(weights + index), _mm256_loadu_ps(input + index), accumulator0);
#else
        accumulator0 = _mm256_add_ps(accumulator0, _mm256_mul_ps(widen8(weights + index), _mm256_loadu_ps(input + index)));
#endif
    }

    accumulator0 = _mm256_add_ps(accumulator0, accumulator1);
    accumulator2 = _mm256_add_ps(accumulator2, accumulator3);
    float total = horizontal_sum(_mm256_add_ps(accumulator0, accumulator2));
    for (; index < count; ++index) {
        total += widen(weights[index]) * input[index];
    }
    return total;
#else
    float accumulator0 = 0.0F;
    float accumulator1 = 0.0F;
    float accumulator2 = 0.0F;
    float accumulator3 = 0.0F;

    std::size_t index = 0U;
    for (; index + 4U <= count; index += 4U) {
        accumulator0 += widen(weights[index]) * input[index];
        accumulator1 += widen(weights[index + 1U]) * input[index + 1U];
        accumulator2 += widen(weights[index + 2U]) * input[index + 2U];
        accumulator3 += widen(weights[index + 3U]) * input[index + 3U];
    }
    float total = (accumulator0 + accumulator1) + (accumulator2 + accumulator3);
    for (; index < count; ++index) {
        total += widen(weights[index]) * input[index];
    }
    return total;
#endif
}

}  // namespace

float dot(const float* left, const float* right, std::size_t count);

std::string kernel_description() {
    // The hand-written dot product is AVX2. Naming AVX-512 here would claim a
    // kernel that does not exist; the most it does on such a machine is let the
    // compiler widen the surrounding scalar loops.
#if defined(__AVX2__) && defined(__FMA__)
#if defined(__AVX512F__)
    return "AVX2 + FMA BF16 kernels (AVX-512 available but unused)";
#else
    return "AVX2 + FMA BF16 kernels";
#endif
#elif defined(__AVX2__)
    return "AVX2 BF16 kernels (no FMA)";
#else
    return "portable scalar BF16 kernels (rebuild with -march=native for AVX2)";
#endif
}

void matvec_bf16_serial(const std::uint16_t* const weights, const float* const input, float* const out,
                        const std::size_t out_dim, const std::size_t in_dim) {
    for (std::size_t row = 0; row < out_dim; ++row) {
        out[row] = dot_bf16(weights + row * in_dim, input, in_dim);
    }
}

void matvec_f32(const float* const weights, const float* const input, float* const out,
                const std::size_t out_dim, const std::size_t in_dim) {
    for (std::size_t row = 0; row < out_dim; ++row) {
        out[row] = dot(weights + row * in_dim, input, in_dim);
    }
}

void matvec_bf16(ThreadPool& pool, const std::uint16_t* const weights, const float* const input,
                 float* const out, const std::size_t out_dim, const std::size_t in_dim) {
    const std::size_t task_count = (out_dim + rows_per_task - 1U) / rows_per_task;
    if (task_count <= 1U) {
        matvec_bf16_serial(weights, input, out, out_dim, in_dim);
        return;
    }

    pool.parallel_for(task_count, [&](const std::size_t task) {
        const std::size_t first = task * rows_per_task;
        const std::size_t last = std::min(first + rows_per_task, out_dim);
        for (std::size_t row = first; row < last; ++row) {
            out[row] = dot_bf16(weights + row * in_dim, input, in_dim);
        }
    });
}

void matvec_bf16_accumulate(ThreadPool& pool, const std::uint16_t* const weights, const float* const input,
                            float* const out, const std::size_t out_dim, const std::size_t in_dim,
                            const float scale) {
    const std::size_t task_count = (out_dim + rows_per_task - 1U) / rows_per_task;
    if (task_count <= 1U) {
        for (std::size_t row = 0; row < out_dim; ++row) {
            out[row] += scale * dot_bf16(weights + row * in_dim, input, in_dim);
        }
        return;
    }

    // Every task owns a disjoint row range of out, so no synchronisation is needed.
    pool.parallel_for(task_count, [&](const std::size_t task) {
        const std::size_t first = task * rows_per_task;
        const std::size_t last = std::min(first + rows_per_task, out_dim);
        for (std::size_t row = first; row < last; ++row) {
            out[row] += scale * dot_bf16(weights + row * in_dim, input, in_dim);
        }
    });
}

void widen_bf16(const std::uint16_t* const source, float* const destination, const std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        destination[index] = widen(source[index]);
    }
}

float dot(const float* const left, const float* const right, const std::size_t count) {
    float total = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
        total += left[index] * right[index];
    }
    return total;
}

void axpy(const float scale, const float* const source, float* const destination, const std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        destination[index] += scale * source[index];
    }
}

void rms_norm(float* const values, const float* const weight, const std::size_t count, const float epsilon) {
    double sum_of_squares = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        sum_of_squares += static_cast<double>(values[index]) * static_cast<double>(values[index]);
    }
    const auto inverse = static_cast<float>(
        1.0 / std::sqrt(sum_of_squares / static_cast<double>(count) + static_cast<double>(epsilon)));
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = values[index] * inverse * weight[index];
    }
}

void softmax(float* const values, const std::size_t count) {
    if (count == 0U) {
        return;
    }
    const float maximum = *std::max_element(values, values + count);
    float total = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = std::exp(values[index] - maximum);
        total += values[index];
    }
    const float inverse = total > 0.0F ? 1.0F / total : 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
        values[index] *= inverse;
    }
}

void silu_multiply(const float* const gate, const float* const up, float* const destination,
                   const std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        const float value = gate[index];
        destination[index] = value / (1.0F + std::exp(-value)) * up[index];
    }
}

}  // namespace gemm
}  // namespace litemind
