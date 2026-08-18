#include "Attention.hpp"

#include "Gemm.hpp"

#include <algorithm>
#include <cstring>

namespace litemind {

void Attention::configure(const Config& config, ThreadPool& pool, const RotaryEmbedding& rope) {
    config_ = &config;
    pool_ = &pool;
    rope_ = &rope;

    query_.assign(config.num_attention_heads * config.qk_head_dim(), 0.0F);
    query_latent_.assign(std::max<std::size_t>(config.q_lora_rank, 1U), 0.0F);
    kv_latent_.assign(config.kv_lora_rank + config.qk_rope_head_dim, 0.0F);
    kv_expanded_.assign(config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim), 0.0F);
    context_.assign(config.num_attention_heads * config.v_head_dim, 0.0F);
}

bool Attention::forward(const float* const input, const AttentionWeights& weights, KvCache& cache,
                        const std::size_t position, float* const output) {
    const Config& config = *config_;
    const std::size_t heads = config.num_attention_heads;
    const std::size_t qk_nope = config.qk_nope_head_dim;
    const std::size_t qk_rope = config.qk_rope_head_dim;
    const std::size_t qk_head = config.qk_head_dim();
    const std::size_t v_head = config.v_head_dim;
    const std::size_t kv_rank = config.kv_lora_rank;

    // ── Queries ──────────────────────────────────────────────────────────────
    if (config.q_lora_rank == 0U) {
        gemm::matvec_bf16(*pool_, weights.q_proj.as_bf16(), input, query_.data(), heads * qk_head,
                          config.hidden_size);
    } else {
        gemm::matvec_bf16(*pool_, weights.q_a_proj.as_bf16(), input, query_latent_.data(),
                          config.q_lora_rank, config.hidden_size);
        gemm::rms_norm(query_latent_.data(), weights.q_a_layernorm.data(), config.q_lora_rank,
                       config.rms_norm_eps);
        gemm::matvec_bf16(*pool_, weights.q_b_proj.as_bf16(), query_latent_.data(), query_.data(),
                          heads * qk_head, config.q_lora_rank);
    }

    // Each head's rotary channels sit after its non-rotary ones.
    for (std::size_t head = 0; head < heads; ++head) {
        rope_->apply(query_.data() + head * qk_head + qk_nope, position);
    }

    // ── Compressed keys and values ───────────────────────────────────────────
    gemm::matvec_bf16(*pool_, weights.kv_a_proj.as_bf16(), input, kv_latent_.data(),
                      kv_rank + qk_rope, config.hidden_size);

    // Only the latent half is normalised; the rotary key tail passes through.
    gemm::rms_norm(kv_latent_.data(), weights.kv_a_layernorm.data(), kv_rank, config.rms_norm_eps);
    rope_->apply(kv_latent_.data() + kv_rank, position);

    gemm::matvec_bf16(*pool_, weights.kv_b_proj.as_bf16(), kv_latent_.data(), kv_expanded_.data(),
                      heads * (qk_nope + v_head), kv_rank);

    const std::size_t slot = cache.append();
    if (slot >= cache.capacity()) {
        return false;
    }

    // kv_b_proj emits each head's key and value contiguously; split them apart
    // so attention reads keys and values with independent strides.
    float* const cached_key = cache.key_nope(slot);
    float* const cached_value = cache.value(slot);
    for (std::size_t head = 0; head < heads; ++head) {
        const float* const source = kv_expanded_.data() + head * (qk_nope + v_head);
        std::memcpy(cached_key + head * qk_nope, source, qk_nope * sizeof(float));
        std::memcpy(cached_value + head * v_head, source + qk_nope, v_head * sizeof(float));
    }
    std::memcpy(cache.key_rope(slot), kv_latent_.data() + kv_rank, qk_rope * sizeof(float));

    // ── Causal attention over every cached position ──────────────────────────
    const std::size_t length = cache.token_count();
    scores_.resize(length);
    std::fill(context_.begin(), context_.end(), 0.0F);
    const float scale = config.softmax_scale();

    for (std::size_t head = 0; head < heads; ++head) {
        const float* const query_nope = query_.data() + head * qk_head;
        const float* const query_rope = query_nope + qk_nope;

        for (std::size_t token = 0; token < length; ++token) {
            // The rotary half of the key is shared by all heads.
            scores_[token] = scale * (gemm::dot(query_nope, cache.key_nope(token, head), qk_nope)
                                    + gemm::dot(query_rope, cache.key_rope(token), qk_rope));
        }
        gemm::softmax(scores_.data(), length);

        float* const head_context = context_.data() + head * v_head;
        for (std::size_t token = 0; token < length; ++token) {
            gemm::axpy(scores_[token], cache.value(token, head), head_context, v_head);
        }
    }

    gemm::matvec_bf16(*pool_, weights.o_proj.as_bf16(), context_.data(), output, config.hidden_size,
                      heads * v_head);
    return true;
}

}  // namespace litemind
