#include "Config.hpp"
#include "Rope.hpp"
#include "TestSupport.hpp"

#include <cmath>
#include <vector>

using namespace test_support;

namespace {

/** A DeepSeek-V2-Lite shaped configuration, built without touching the disk. */
[[nodiscard]] litemind::Config lite_config() {
    litemind::Config config;
    config.hidden_size = 2048U;
    config.num_attention_heads = 16U;
    config.qk_nope_head_dim = 128U;
    config.qk_rope_head_dim = 64U;
    config.v_head_dim = 128U;
    config.kv_lora_rank = 512U;
    config.rope_theta = 10000.0F;
    config.rope_scaling.enabled = true;
    config.rope_scaling.type = "yarn";
    config.rope_scaling.factor = 40.0;
    config.rope_scaling.beta_fast = 32.0;
    config.rope_scaling.beta_slow = 1.0;
    config.rope_scaling.mscale = 0.707;
    config.rope_scaling.mscale_all_dim = 0.707;
    config.rope_scaling.original_max_position_embeddings = 4096U;
    return config;
}

}  // namespace

int main() {
    const litemind::Config config = lite_config();

    // ── The softmax scale must carry the YaRN amplitude correction ──────────
    // mscale = 0.1 * 0.707 * ln(40) + 1, and the scale is multiplied by its square.
    const double mscale = 0.1 * 0.707 * std::log(40.0) + 1.0;
    const double expected_scale = (1.0 / std::sqrt(192.0)) * mscale * mscale;
    check_close(config.softmax_scale(), expected_scale, 1e-6,
                "softmax_scale includes mscale squared");
    check(config.softmax_scale() > 1.0 / std::sqrt(192.0),
          "the YaRN correction sharpens attention rather than leaving it flat");

    litemind::RotaryEmbedding rope;
    rope.build(config, 128U);
    check(rope.dimension() == 64U, "the rope table covers qk_rope_head_dim");

    // ── Position zero must be the identity, up to the de-interleave ─────────
    // At position 0 every cosine is 1 and every sine is 0, so the values are
    // only permuted: even channels first, then odd ones.
    std::vector<float> values(64U);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(index);
    }
    rope.apply(values.data(), 0U);
    for (std::size_t index = 0; index < 32U; ++index) {
        check_close(values[index], static_cast<double>(2U * index), 1e-5,
                    "position zero moves even channel " + std::to_string(index) + " to the front");
        check_close(values[index + 32U], static_cast<double>(2U * index + 1U), 1e-5,
                    "position zero moves odd channel " + std::to_string(index) + " to the back");
    }

    // ── Rotation must preserve each channel pair's magnitude ────────────────
    std::vector<float> pairs(64U, 0.0F);
    pairs[0] = 3.0F;   // even channel 0
    pairs[1] = 4.0F;   // odd channel 0, so this pair has magnitude 5
    rope.apply(pairs.data(), 7U);
    const double magnitude = std::hypot(pairs[0], pairs[32]);
    // The magnitude scale is one here because mscale equals mscale_all_dim.
    check_close(magnitude, 5.0, 1e-4, "rotation preserves the pair's magnitude");

    // ── A dot product must depend only on the relative position ─────────────
    // This is the defining property of rotary embedding: rotating a query at
    // position p and a key at position q leaves their dot product a function of
    // p - q alone. It fails immediately if the de-interleave is wrong.
    std::vector<float> query_a(64U);
    std::vector<float> key_a(64U);
    for (std::size_t index = 0; index < 64U; ++index) {
        query_a[index] = std::sin(static_cast<float>(index) * 0.3F);
        key_a[index] = std::cos(static_cast<float>(index) * 0.17F);
    }
    std::vector<float> query_b = query_a;
    std::vector<float> key_b = key_a;

    rope.apply(query_a.data(), 20U);
    rope.apply(key_a.data(), 12U);
    rope.apply(query_b.data(), 33U);
    rope.apply(key_b.data(), 25U);  // The same gap of eight positions.

    double dot_a = 0.0;
    double dot_b = 0.0;
    for (std::size_t index = 0; index < 64U; ++index) {
        dot_a += static_cast<double>(query_a[index]) * key_a[index];
        dot_b += static_cast<double>(query_b[index]) * key_b[index];
    }
    check_close(dot_b, dot_a, 1e-3, "the dot product depends only on the relative position");

    // A different gap must give a different result, or the check above is vacuous.
    std::vector<float> query_c = query_a;
    std::vector<float> key_c = key_a;
    rope.apply(query_c.data(), 20U);
    rope.apply(key_c.data(), 3U);
    double dot_c = 0.0;
    for (std::size_t index = 0; index < 64U; ++index) {
        dot_c += static_cast<double>(query_c[index]) * key_c[index];
    }
    check(std::abs(dot_c - dot_a) > 1e-3, "a different relative position gives a different result");

    // ── YaRN must actually change the frequencies ───────────────────────────
    litemind::Config unscaled = lite_config();
    unscaled.rope_scaling.enabled = false;
    litemind::RotaryEmbedding plain;
    plain.build(unscaled, 128U);

    std::vector<float> scaled_values(64U, 1.0F);
    std::vector<float> plain_values(64U, 1.0F);
    rope.apply(scaled_values.data(), 100U);
    plain.apply(plain_values.data(), 100U);

    bool differs = false;
    for (std::size_t index = 0; index < 64U; ++index) {
        differs = differs || std::abs(scaled_values[index] - plain_values[index]) > 1e-4;
    }
    check(differs, "YaRN interpolation changes the low-frequency channels");

    return report("RopeTest");
}
