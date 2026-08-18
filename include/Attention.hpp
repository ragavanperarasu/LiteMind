#pragma once

#include "Config.hpp"
#include "KvCache.hpp"
#include "Rope.hpp"
#include "Threading.hpp"
#include "WeightStore.hpp"

#include <vector>

namespace litemind {

/** The projection matrices of one multi-head latent attention block. */
struct AttentionWeights final {
    /** Used when q_lora_rank is zero: [heads * qk_head_dim, hidden]. */
    WeightView q_proj;

    /** Used when q_lora_rank is non-zero: hidden -> rank -> heads * qk_head_dim. */
    WeightView q_a_proj;
    std::vector<float> q_a_layernorm;
    WeightView q_b_proj;

    /** [kv_lora_rank + qk_rope_head_dim, hidden]; the tail is the shared rope key. */
    WeightView kv_a_proj;
    std::vector<float> kv_a_layernorm;

    /** [heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]. */
    WeightView kv_b_proj;

    /** [hidden, heads * v_head_dim]. */
    WeightView o_proj;
};

/**
 * @brief DeepSeek-V2 multi-head latent attention for a single position.
 *
 * Keys and values are produced from one low-rank latent vector, and the rotary
 * part of the key is decoupled: a single qk_rope_head_dim vector shared by every
 * head, rather than one per head. Queries carry both parts, concatenated as
 * qk_nope_head_dim non-rotary channels followed by qk_rope_head_dim rotary ones.
 *
 * One instance owns the scratch buffers and is reused across layers and steps,
 * so a decoding step performs no heap allocation.
 */
class Attention final {
public:
    Attention() = default;

    /** Binds the block to a configuration, a thread pool and a rotary table. */
    void configure(const Config& config, ThreadPool& pool, const RotaryEmbedding& rope);

    /**
     * Runs attention for the token at position, appending its key and value to
     * cache and writing hidden_size values to output.
     *
     * Returns false only when the cache is full, which the caller reports as a
     * context-length limit rather than an error.
     */
    [[nodiscard]] bool forward(const float* input, const AttentionWeights& weights, KvCache& cache,
                               std::size_t position, float* output);

private:
    const Config* config_{nullptr};
    ThreadPool* pool_{nullptr};
    const RotaryEmbedding* rope_{nullptr};

    std::vector<float> query_;          ///< [heads * qk_head_dim]
    std::vector<float> query_latent_;   ///< [q_lora_rank]
    std::vector<float> kv_latent_;      ///< [kv_lora_rank + qk_rope_head_dim]
    std::vector<float> kv_expanded_;    ///< [heads * (qk_nope_head_dim + v_head_dim)]
    std::vector<float> scores_;         ///< [cache capacity]
    std::vector<float> context_;        ///< [heads * v_head_dim]
};

}  // namespace litemind
