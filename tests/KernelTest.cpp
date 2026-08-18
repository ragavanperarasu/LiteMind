#include "Gemm.hpp"
#include "TestSupport.hpp"
#include "Threading.hpp"

#include <bit>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using namespace test_support;

namespace {

[[nodiscard]] std::uint16_t to_bf16(const float value) {
    // Round to nearest even, matching how the checkpoint was written.
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounded = bits + 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}

}  // namespace

int main() {
    std::cout << "  kernels: " << litemind::gemm::kernel_description() << "\n";

    // ── matvec against a reference computed in double precision ──────────────
    constexpr std::size_t out_dim = 67U;   // Deliberately not a multiple of the
    constexpr std::size_t in_dim = 253U;   // vector width or the task block size.

    std::mt19937 generator(12345U);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    std::vector<float> weights_f32(out_dim * in_dim);
    std::vector<std::uint16_t> weights_bf16(out_dim * in_dim);
    for (std::size_t index = 0; index < weights_f32.size(); ++index) {
        weights_bf16[index] = to_bf16(distribution(generator));
        // Compare against the BF16 values themselves, so the test measures the
        // kernel rather than the precision lost when the weights were rounded.
        weights_f32[index] = litemind::gemm::kernel_description().empty()
                                 ? 0.0F
                                 : std::bit_cast<float>(static_cast<std::uint32_t>(weights_bf16[index]) << 16U);
    }

    std::vector<float> input(in_dim);
    for (float& value : input) {
        value = distribution(generator);
    }

    std::vector<double> expected(out_dim, 0.0);
    for (std::size_t row = 0; row < out_dim; ++row) {
        for (std::size_t column = 0; column < in_dim; ++column) {
            expected[row] += static_cast<double>(weights_f32[row * in_dim + column])
                           * static_cast<double>(input[column]);
        }
    }

    litemind::ThreadPool pool(4U);
    std::vector<float> actual(out_dim, 0.0F);
    litemind::gemm::matvec_bf16(pool, weights_bf16.data(), input.data(), actual.data(), out_dim, in_dim);
    for (std::size_t row = 0; row < out_dim; ++row) {
        check_close(actual[row], expected[row], 1e-3, "threaded matvec row " + std::to_string(row));
    }

    // The serial path must agree with the threaded one exactly in structure.
    std::vector<float> serial(out_dim, 0.0F);
    litemind::gemm::matvec_bf16_serial(weights_bf16.data(), input.data(), serial.data(), out_dim, in_dim);
    for (std::size_t row = 0; row < out_dim; ++row) {
        check_close(serial[row], actual[row], 1e-4, "serial matches threaded at row " + std::to_string(row));
    }

    // Accumulation must add to what is already there, scaled.
    std::vector<float> accumulated(out_dim, 1.0F);
    litemind::gemm::matvec_bf16_accumulate(pool, weights_bf16.data(), input.data(),
                                           accumulated.data(), out_dim, in_dim, 0.5F);
    for (std::size_t row = 0; row < out_dim; ++row) {
        check_close(accumulated[row], 1.0 + 0.5 * expected[row], 1e-3,
                    "accumulating matvec row " + std::to_string(row));
    }

    // ── RMSNorm ─────────────────────────────────────────────────────────────
    std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> weight(4U, 2.0F);
    litemind::gemm::rms_norm(values.data(), weight.data(), values.size(), 1e-6F);
    // mean of squares is 7.5, so the scale is 2 / sqrt(7.5).
    const double scale = 2.0 / std::sqrt(7.5 + 1e-6);
    for (std::size_t index = 0; index < values.size(); ++index) {
        check_close(values[index], (static_cast<double>(index) + 1.0) * scale, 1e-5,
                    "rms_norm element " + std::to_string(index));
    }

    // ── Softmax ─────────────────────────────────────────────────────────────
    std::vector<float> scores{1.0F, 2.0F, 3.0F};
    litemind::gemm::softmax(scores.data(), scores.size());
    check_close(std::accumulate(scores.begin(), scores.end(), 0.0F), 1.0, 1e-6, "softmax sums to one");
    check(scores[2] > scores[1] && scores[1] > scores[0], "softmax preserves the ordering");

    // A large offset must not overflow, which is why the maximum is subtracted.
    std::vector<float> extreme{1000.0F, 1001.0F};
    litemind::gemm::softmax(extreme.data(), extreme.size());
    check(std::isfinite(extreme[0]) && std::isfinite(extreme[1]), "softmax is stable at large logits");
    check_close(extreme[0] + extreme[1], 1.0, 1e-6, "the stable softmax still sums to one");

    // ── SiLU times up ───────────────────────────────────────────────────────
    const std::vector<float> gate{0.0F, 1.0F};
    const std::vector<float> up{3.0F, 2.0F};
    std::vector<float> product(2U);
    litemind::gemm::silu_multiply(gate.data(), up.data(), product.data(), 2U);
    check_close(product[0], 0.0, 1e-6, "silu(0) is zero");
    check_close(product[1], (1.0 / (1.0 + std::exp(-1.0))) * 2.0, 1e-6, "silu(1) times up");

    // ── Thread pool ─────────────────────────────────────────────────────────
    // Every index must run exactly once, whatever the worker count.
    for (const std::size_t workers : {1U, 2U, 8U}) {
        litemind::ThreadPool sized(workers);
        std::vector<int> visits(1000U, 0);
        sized.parallel_for(visits.size(), [&visits](const std::size_t index) { visits[index] = 1; });
        const int total = std::accumulate(visits.begin(), visits.end(), 0);
        check(total == static_cast<int>(visits.size()),
              "parallel_for visits every index with " + std::to_string(workers) + " workers");
    }

    return report("KernelTest");
}
