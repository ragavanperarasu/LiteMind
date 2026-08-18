#pragma once

#include "Threading.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace litemind {

/**
 * @brief The dense kernels the decoder runs on the CPU.
 *
 * Weights stay BF16 in their memory mapping and are widened to float32 inside
 * the inner loop. That halves the bytes each matrix-vector product pulls from
 * RAM or the SSD, and on this workload bandwidth, not arithmetic, is the limit.
 * Widening BF16 costs one shift because it shares float32's exponent layout.
 */
namespace gemm {

/** Describes the vector instruction set this binary was compiled for. */
[[nodiscard]] std::string kernel_description();

/**
 * out = weights * input, where weights is a row-major [out_dim, in_dim] BF16
 * matrix. Rows are distributed across the pool.
 */
void matvec_bf16(ThreadPool& pool, const std::uint16_t* weights, const float* input, float* out,
                 std::size_t out_dim, std::size_t in_dim);

/**
 * out += scale * (weights * input). Used to accumulate mixture-of-experts
 * contributions into a shared residual buffer without a temporary per expert.
 */
void matvec_bf16_accumulate(ThreadPool& pool, const std::uint16_t* weights, const float* input,
                            float* out, std::size_t out_dim, std::size_t in_dim, float scale);

/** Single-threaded matrix-vector product, for small matrices such as the router. */
void matvec_bf16_serial(const std::uint16_t* weights, const float* input, float* out,
                        std::size_t out_dim, std::size_t in_dim);

/**
 * out = weights * input for a row-major float32 matrix. Used for the router
 * gate, which is small enough to keep widened in RAM and whose stored element
 * type differs between DeepSeek checkpoints.
 */
void matvec_f32(const float* weights, const float* input, float* out, std::size_t out_dim,
                std::size_t in_dim);

/** Widens a contiguous BF16 block to float32. */
void widen_bf16(const std::uint16_t* source, float* destination, std::size_t count);

/** A float32 dot product. */
[[nodiscard]] float dot(const float* left, const float* right, std::size_t count);

/** destination += scale * source. */
void axpy(float scale, const float* source, float* destination, std::size_t count);

/**
 * RMS normalisation in place: x = x / sqrt(mean(x^2) + epsilon) * weight.
 *
 * The sum of squares accumulates in double precision. At hidden_size 2048 the
 * float32 error is small but systematic, and it compounds across 27 layers.
 */
void rms_norm(float* values, const float* weight, std::size_t count, float epsilon);

/** Softmax in place over a contiguous span, using the max-subtraction form. */
void softmax(float* values, std::size_t count);

/** destination[i] = silu(gate[i]) * up[i], the SwiGLU non-linearity. */
void silu_multiply(const float* gate, const float* up, float* destination, std::size_t count);

}  // namespace gemm
}  // namespace litemind
