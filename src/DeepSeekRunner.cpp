#include "DeepSeekRunner.hpp"

#include "Gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace litemind {
namespace {

/** The three matrices of one SwiGLU feed-forward block. */
struct FeedForwardWeights final {
    WeightView gate;
    WeightView up;
    WeightView down;
    std::size_t intermediate_size{};
};

/** Everything one decoder layer needs, as views into the memory mappings. */
struct LayerWeights final {
    std::vector<float> input_layernorm;
    std::vector<float> post_attention_layernorm;
    AttentionWeights attention;

    bool mixture_of_experts{false};
    FeedForwardWeights dense;                  ///< Used when mixture_of_experts is false.
    std::vector<float> router;                 ///< [n_routed_experts, hidden], widened to float32.
    std::vector<FeedForwardWeights> experts;
    FeedForwardWeights shared;
};

[[nodiscard]] std::string layer_prefix(const std::size_t layer) {
    return "model.layers." + std::to_string(layer) + ".";
}

[[nodiscard]] double seconds_since(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

[[nodiscard]] std::string format_bytes(const std::uint64_t bytes) {
    constexpr double unit = 1024.0;
    const char* const names[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    std::size_t name = 0U;
    while (value >= unit && name + 1U < std::size(names)) {
        value /= unit;
        ++name;
    }
    std::ostringstream text;
    text << std::fixed << std::setprecision(value < 10.0 ? 2 : 1) << value << ' ' << names[name];
    return text.str();
}

}  // namespace

struct DeepSeekRunner::Impl final {
    RunnerOptions options;
    Config config;
    Tokenizer tokenizer;
    WeightStore store;
    ExpertCache experts;
    MoeRouter router;
    RotaryEmbedding rope;
    Attention attention;
    std::unique_ptr<ThreadPool> pool;

    WeightView embedding;
    WeightView lm_head;
    std::vector<float> final_layernorm;
    std::vector<LayerWeights> layers;
    std::vector<KvCache> caches;

    // Scratch reused across every step so decoding allocates nothing.
    std::vector<float> hidden_state;
    std::vector<float> normalised;
    std::vector<float> block_output;
    std::vector<float> feed_forward_output;
    std::vector<float> gate_buffer;
    std::vector<float> up_buffer;
    std::vector<float> activation_buffer;
    std::vector<float> router_logits;
    std::vector<float> logits;
    std::vector<ExpertSelection> selection;

    std::uint64_t hot_bytes{};
    std::uint64_t expert_bytes{};
    bool loaded{false};

    [[nodiscard]] bool resolve_attention(std::size_t layer, AttentionWeights& weights, std::string& error);
    [[nodiscard]] bool resolve_feed_forward(const std::string& prefix, std::size_t intermediate_size,
                                            FeedForwardWeights& weights, std::string& error);
    [[nodiscard]] bool resolve_layer(std::size_t layer, LayerWeights& weights, std::string& error);

    void feed_forward(const FeedForwardWeights& weights, const float* input, float* output, float scale);
    [[nodiscard]] bool forward(std::uint32_t token_id, std::size_t position, bool need_logits);
    void warm_start();
};

// ── Weight resolution ────────────────────────────────────────────────────────

bool DeepSeekRunner::Impl::resolve_attention(const std::size_t layer, AttentionWeights& weights,
                                             std::string& error) {
    const std::string prefix = layer_prefix(layer) + "self_attn.";
    const std::size_t hidden = config.hidden_size;
    const std::size_t heads = config.num_attention_heads;
    const std::size_t qk_head = config.qk_head_dim();

    if (config.q_lora_rank == 0U) {
        weights.q_proj = store.require(prefix + "q_proj.weight", {heads * qk_head, hidden},
                                       DataType::BFloat16, error);
        if (!weights.q_proj.valid()) {
            return false;
        }
    } else {
        weights.q_a_proj = store.require(prefix + "q_a_proj.weight", {config.q_lora_rank, hidden},
                                         DataType::BFloat16, error);
        if (!weights.q_a_proj.valid()) {
            return false;
        }
        if (!store.read_float32(prefix + "q_a_layernorm.weight", {config.q_lora_rank},
                                weights.q_a_layernorm, error)) {
            return false;
        }
        weights.q_b_proj = store.require(prefix + "q_b_proj.weight", {heads * qk_head, config.q_lora_rank},
                                         DataType::BFloat16, error);
        if (!weights.q_b_proj.valid()) {
            return false;
        }
    }

    weights.kv_a_proj = store.require(prefix + "kv_a_proj_with_mqa.weight",
                                      {config.kv_lora_rank + config.qk_rope_head_dim, hidden},
                                      DataType::BFloat16, error);
    if (!weights.kv_a_proj.valid()) {
        return false;
    }
    if (!store.read_float32(prefix + "kv_a_layernorm.weight", {config.kv_lora_rank},
                            weights.kv_a_layernorm, error)) {
        return false;
    }
    weights.kv_b_proj = store.require(
        prefix + "kv_b_proj.weight",
        {heads * (config.qk_nope_head_dim + config.v_head_dim), config.kv_lora_rank},
        DataType::BFloat16, error);
    if (!weights.kv_b_proj.valid()) {
        return false;
    }
    weights.o_proj = store.require(prefix + "o_proj.weight", {hidden, heads * config.v_head_dim},
                                   DataType::BFloat16, error);
    return weights.o_proj.valid();
}

bool DeepSeekRunner::Impl::resolve_feed_forward(const std::string& prefix,
                                                const std::size_t intermediate_size,
                                                FeedForwardWeights& weights, std::string& error) {
    const std::size_t hidden = config.hidden_size;
    weights.intermediate_size = intermediate_size;

    weights.gate = store.require(prefix + "gate_proj.weight", {intermediate_size, hidden},
                                 DataType::BFloat16, error);
    if (!weights.gate.valid()) {
        return false;
    }
    weights.up = store.require(prefix + "up_proj.weight", {intermediate_size, hidden},
                               DataType::BFloat16, error);
    if (!weights.up.valid()) {
        return false;
    }
    // down_proj maps back the other way, so its rows are the intermediate axis.
    weights.down = store.require(prefix + "down_proj.weight", {hidden, intermediate_size},
                                 DataType::BFloat16, error);
    return weights.down.valid();
}

bool DeepSeekRunner::Impl::resolve_layer(const std::size_t layer, LayerWeights& weights,
                                         std::string& error) {
    const std::string prefix = layer_prefix(layer);
    if (!store.read_float32(prefix + "input_layernorm.weight", {config.hidden_size},
                            weights.input_layernorm, error)
        || !store.read_float32(prefix + "post_attention_layernorm.weight", {config.hidden_size},
                               weights.post_attention_layernorm, error)) {
        return false;
    }
    if (!resolve_attention(layer, weights.attention, error)) {
        return false;
    }

    weights.mixture_of_experts = config.is_moe_layer(layer);
    if (!weights.mixture_of_experts) {
        return resolve_feed_forward(prefix + "mlp.", config.intermediate_size, weights.dense, error);
    }

    // The router gate is small and read every step, so it is widened once into
    // RAM. Its stored element type varies between DeepSeek releases, and
    // read_float32 accepts BF16, F16 and F32 alike.
    if (!store.read_float32(prefix + "mlp.gate.weight", {config.n_routed_experts, config.hidden_size},
                            weights.router, error)) {
        return false;
    }

    weights.experts.resize(config.n_routed_experts);
    for (std::size_t expert = 0; expert < config.n_routed_experts; ++expert) {
        const std::string expert_prefix = prefix + "mlp.experts." + std::to_string(expert) + ".";
        if (!resolve_feed_forward(expert_prefix, config.moe_intermediate_size, weights.experts[expert],
                                  error)) {
            return false;
        }
        expert_bytes += weights.experts[expert].gate.byte_size()
                      + weights.experts[expert].up.byte_size()
                      + weights.experts[expert].down.byte_size();
    }

    if (config.n_shared_experts != 0U) {
        if (!resolve_feed_forward(prefix + "mlp.shared_experts.",
                                  config.shared_expert_intermediate_size(), weights.shared, error)) {
            return false;
        }
    }
    return true;
}

// ── Forward pass ─────────────────────────────────────────────────────────────

void DeepSeekRunner::Impl::feed_forward(const FeedForwardWeights& weights, const float* const input,
                                        float* const output, const float scale) {
    const std::size_t intermediate = weights.intermediate_size;
    gemm::matvec_bf16(*pool, weights.gate.as_bf16(), input, gate_buffer.data(), intermediate,
                      config.hidden_size);
    gemm::matvec_bf16(*pool, weights.up.as_bf16(), input, up_buffer.data(), intermediate,
                      config.hidden_size);
    gemm::silu_multiply(gate_buffer.data(), up_buffer.data(), activation_buffer.data(), intermediate);

    // Accumulating straight into the residual avoids a temporary per expert.
    gemm::matvec_bf16_accumulate(*pool, weights.down.as_bf16(), activation_buffer.data(), output,
                                 config.hidden_size, intermediate, scale);
}

bool DeepSeekRunner::Impl::forward(const std::uint32_t token_id, const std::size_t position,
                                   const bool need_logits) {
    const std::size_t hidden = config.hidden_size;

    // Embedding lookup: one row of a 419 MB table, so only its pages are touched.
    const std::uint16_t* const embedding_row =
        embedding.as_bf16() + static_cast<std::uint64_t>(token_id) * hidden;
    gemm::widen_bf16(embedding_row, hidden_state.data(), hidden);

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const LayerWeights& weights = layers[layer];

        // Attention sub-layer.
        std::copy(hidden_state.begin(), hidden_state.end(), normalised.begin());
        gemm::rms_norm(normalised.data(), weights.input_layernorm.data(), hidden, config.rms_norm_eps);
        std::fill(block_output.begin(), block_output.end(), 0.0F);
        if (!attention.forward(normalised.data(), weights.attention, caches[layer], position,
                               block_output.data())) {
            return false;
        }
        gemm::axpy(1.0F, block_output.data(), hidden_state.data(), hidden);

        // Feed-forward sub-layer.
        std::copy(hidden_state.begin(), hidden_state.end(), normalised.begin());
        gemm::rms_norm(normalised.data(), weights.post_attention_layernorm.data(), hidden,
                       config.rms_norm_eps);
        std::fill(feed_forward_output.begin(), feed_forward_output.end(), 0.0F);

        if (!weights.mixture_of_experts) {
            feed_forward(weights.dense, normalised.data(), feed_forward_output.data(), 1.0F);
        } else {
            gemm::matvec_f32(weights.router.data(), normalised.data(), router_logits.data(),
                             config.n_routed_experts, hidden);
            router.select(router_logits, selection);

            for (const ExpertSelection& choice : selection) {
                const FeedForwardWeights& expert = weights.experts[choice.expert_index];
                // Declare the expert needed before touching it, so the cache can
                // stream it in and make room by evicting a colder one.
                experts.touch(ExpertCache::make_key(layer, choice.expert_index),
                              ExpertCache::Block{expert.gate, expert.up, expert.down});
                feed_forward(expert, normalised.data(), feed_forward_output.data(), choice.weight);
            }
            if (config.n_shared_experts != 0U) {
                // Shared experts run for every token, so they stay hot naturally.
                feed_forward(weights.shared, normalised.data(), feed_forward_output.data(), 1.0F);
            }
        }
        gemm::axpy(1.0F, feed_forward_output.data(), hidden_state.data(), hidden);
    }

    if (!need_logits) {
        return true;  // Prefill only needs logits for the final prompt token.
    }

    std::copy(hidden_state.begin(), hidden_state.end(), normalised.begin());
    gemm::rms_norm(normalised.data(), final_layernorm.data(), hidden, config.rms_norm_eps);
    gemm::matvec_bf16(*pool, lm_head.as_bf16(), normalised.data(), logits.data(), config.vocab_size,
                      hidden);
    return true;
}

void DeepSeekRunner::Impl::warm_start() {
    // Stream in everything a step touches regardless of routing. The experts are
    // deliberately left out: they are the part that must stay on the SSD.
    embedding.prefetch();
    lm_head.prefetch();
    for (const LayerWeights& weights : layers) {
        weights.attention.q_proj.prefetch();
        weights.attention.q_a_proj.prefetch();
        weights.attention.q_b_proj.prefetch();
        weights.attention.kv_a_proj.prefetch();
        weights.attention.kv_b_proj.prefetch();
        weights.attention.o_proj.prefetch();
        weights.shared.gate.prefetch();
        weights.shared.up.prefetch();
        weights.shared.down.prefetch();
        weights.dense.gate.prefetch();
        weights.dense.up.prefetch();
        weights.dense.down.prefetch();
    }
}

// ── Public interface ─────────────────────────────────────────────────────────

DeepSeekRunner::DeepSeekRunner() : impl_(std::make_unique<Impl>()) {}
DeepSeekRunner::~DeepSeekRunner() = default;

bool DeepSeekRunner::ready() const noexcept { return impl_->loaded; }
const Config& DeepSeekRunner::config() const noexcept { return impl_->config; }
const Tokenizer& DeepSeekRunner::tokenizer() const noexcept { return impl_->tokenizer; }
const WeightStore& DeepSeekRunner::weights() const noexcept { return impl_->store; }
const ExpertCache& DeepSeekRunner::expert_cache() const noexcept { return impl_->experts; }

bool DeepSeekRunner::load(const std::filesystem::path& model_directory, const RunnerOptions& options,
                          std::string& error) {
    Impl& state = *impl_;
    state.options = options;
    state.loaded = false;

    if (!state.config.load(model_directory, error) || !state.tokenizer.load(model_directory, error)
        || !state.store.open(model_directory, error)) {
        return false;
    }

    if (state.tokenizer.vocabulary_size() > state.config.vocab_size) {
        error = "The tokenizer defines " + std::to_string(state.tokenizer.vocabulary_size())
              + " tokens but config.json declares a vocabulary of "
              + std::to_string(state.config.vocab_size)
              + ". The tokenizer and the weights come from different models.";
        return false;
    }

    const Config& config = state.config;
    const std::size_t hidden = config.hidden_size;

    state.pool = std::make_unique<ThreadPool>(options.threads);
    state.rope.build(config, options.context_length);
    state.attention.configure(config, *state.pool, state.rope);
    state.router = MoeRouter(config);
    state.experts.configure(options.expert_budget_bytes);

    state.embedding = state.store.require("model.embed_tokens.weight", {config.vocab_size, hidden},
                                          DataType::BFloat16, error);
    if (!state.embedding.valid()) {
        return false;
    }
    if (!state.store.read_float32("model.norm.weight", {hidden}, state.final_layernorm, error)) {
        return false;
    }

    // A tied head shares the embedding table rather than storing a second copy.
    if (config.tie_word_embeddings || !state.store.contains("lm_head.weight")) {
        state.lm_head = state.embedding;
    } else {
        state.lm_head = state.store.require("lm_head.weight", {config.vocab_size, hidden},
                                            DataType::BFloat16, error);
        if (!state.lm_head.valid()) {
            return false;
        }
    }

    state.expert_bytes = 0U;
    state.layers.assign(config.num_hidden_layers, LayerWeights{});
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        if (!state.resolve_layer(layer, state.layers[layer], error)) {
            error = "layer " + std::to_string(layer) + ": " + error;
            return false;
        }
        if (options.verbose) {
            std::cout << "\r  Resolving weights: layer " << (layer + 1U) << " of "
                      << config.num_hidden_layers << std::flush;
        }
    }
    if (options.verbose) {
        std::cout << "\r  Resolving weights: " << config.num_hidden_layers << " layers ready.        \n";
    }

    state.hot_bytes = state.store.statistics().mapped_bytes - state.expert_bytes;

    // Size every scratch buffer once. The largest feed-forward width decides the
    // activation buffers, so a dense layer and an expert can share them.
    const std::size_t widest = std::max({config.intermediate_size, config.moe_intermediate_size,
                                         config.shared_expert_intermediate_size()});
    state.hidden_state.assign(hidden, 0.0F);
    state.normalised.assign(hidden, 0.0F);
    state.block_output.assign(hidden, 0.0F);
    state.feed_forward_output.assign(hidden, 0.0F);
    state.gate_buffer.assign(widest, 0.0F);
    state.up_buffer.assign(widest, 0.0F);
    state.activation_buffer.assign(widest, 0.0F);
    state.router_logits.assign(std::max<std::size_t>(config.n_routed_experts, 1U), 0.0F);
    state.logits.assign(config.vocab_size, 0.0F);

    state.caches.resize(config.num_hidden_layers);
    for (KvCache& cache : state.caches) {
        cache.configure(config.num_attention_heads, config.qk_nope_head_dim, config.qk_rope_head_dim,
                        config.v_head_dim, options.context_length);
    }

    if (options.warm_start) {
        if (options.verbose) {
            std::cout << "  Warming the always-resident weights from disk...\n";
        }
        state.warm_start();
    }

    state.loaded = true;
    return true;
}

void DeepSeekRunner::reset() {
    for (KvCache& cache : impl_->caches) {
        cache.clear();
    }
}

bool DeepSeekRunner::generate(const std::vector<std::uint32_t>& prompt_tokens,
                              const GenerationOptions& options, GenerationResult& result,
                              std::string& error) {
    Impl& state = *impl_;
    if (!state.loaded) {
        error = "The model is not loaded.";
        return false;
    }
    if (prompt_tokens.empty()) {
        error = "The prompt encoded to zero tokens.";
        return false;
    }
    if (prompt_tokens.size() >= state.options.context_length) {
        error = "The prompt is " + std::to_string(prompt_tokens.size()) + " tokens but the context is "
              + std::to_string(state.options.context_length)
              + ". Raise it with --context or shorten the prompt.";
        return false;
    }
    for (const std::uint32_t token : prompt_tokens) {
        if (token >= state.config.vocab_size) {
            error = "Prompt token " + std::to_string(token) + " is outside the model's vocabulary.";
            return false;
        }
    }

    reset();
    result = GenerationResult{};
    result.prompt_tokens = prompt_tokens.size();

    // ── Prefill ──────────────────────────────────────────────────────────────
    // Positions are processed one at a time. Only the last one needs logits,
    // which skips a 419 MB matrix read for every earlier prompt token.
    const auto prefill_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < prompt_tokens.size(); ++index) {
        const bool last = index + 1U == prompt_tokens.size();
        if (!state.forward(prompt_tokens[index], index, last)) {
            error = "The key/value cache filled during prefill.";
            return false;
        }
        if (options.show_progress && prompt_tokens.size() > 1U) {
            std::cout << "\r  Reading the prompt: " << (index + 1U) << " / " << prompt_tokens.size()
                      << " tokens" << std::flush;
        }
    }
    result.prefill_seconds = seconds_since(prefill_start);
    if (options.show_progress && prompt_tokens.size() > 1U) {
        std::cout << "\r  Read the prompt: " << prompt_tokens.size() << " tokens in " << std::fixed
                  << std::setprecision(1) << result.prefill_seconds << " s.            \n";
    }

    // The prediction for the token after the prompt, before any sampling.
    if (options.top_logits != 0U) {
        std::vector<std::pair<std::uint32_t, float>> ranked(state.logits.size());
        for (std::size_t index = 0; index < state.logits.size(); ++index) {
            ranked[index] = {static_cast<std::uint32_t>(index), state.logits[index]};
        }
        const std::size_t keep = std::min(options.top_logits, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(keep),
                          ranked.end(), [](const auto& left, const auto& right) {
                              return left.second > right.second;
                          });
        ranked.resize(keep);
        result.prompt_top_logits = std::move(ranked);
    }

    // ── Decode ───────────────────────────────────────────────────────────────
    Sampler sampler(options.sampling);
    Tokenizer::StreamDecoder decoder(state.tokenizer);
    std::vector<std::uint32_t> history = prompt_tokens;

    const auto decode_start = std::chrono::steady_clock::now();
    std::size_t position = prompt_tokens.size();

    while (result.tokens.size() < options.max_new_tokens) {
        const std::uint32_t token = sampler.next(state.logits, history);

        if (state.config.is_eos(token) || token == state.tokenizer.eos_token_id()) {
            result.stop_reason = "end-of-sequence";
            break;
        }
        if (token >= state.config.vocab_size) {
            result.stop_reason = "invalid token";
            break;
        }

        result.tokens.push_back(token);
        history.push_back(token);

        const std::string fragment = decoder.push(token);
        result.text += fragment;
        if (options.on_text && !fragment.empty()) {
            options.on_text(fragment);
        }

        if (position >= state.options.context_length) {
            result.stop_reason = "context full";
            break;
        }
        if (!state.forward(token, position, true)) {
            result.stop_reason = "context full";
            break;
        }
        ++position;
    }

    const std::string remainder = decoder.flush();
    if (!remainder.empty()) {
        result.text += remainder;
        if (options.on_text) {
            options.on_text(remainder);
        }
    }
    result.decode_seconds = seconds_since(decode_start);
    return true;
}

std::string DeepSeekRunner::memory_report() const {
    const Impl& state = *impl_;
    const StoreStatistics& statistics = state.store.statistics();

    std::uint64_t cache_bytes = 0U;
    for (const KvCache& cache : state.caches) {
        cache_bytes += cache.reserved_bytes();
    }

    std::ostringstream text;
    text << "Memory\n"
         << "  Checkpoint mapped:   " << format_bytes(statistics.mapped_bytes) << " across "
         << statistics.shards << " shard(s), " << statistics.tensors << " tensors\n"
         << "  Always-hot weights:  " << format_bytes(state.hot_bytes)
         << "  (embeddings, attention, norms, shared experts, head)\n"
         << "  Routed experts:      " << format_bytes(state.expert_bytes)
         << "  (streamed from SSD on demand)\n"
         << "  Key/value cache:     " << format_bytes(cache_bytes) << " for "
         << state.options.context_length << " positions\n";

    if (state.experts.enabled()) {
        text << "  Expert budget:       " << format_bytes(state.experts.budget_bytes());
        // Before the first prompt there is no traffic to report, and printing a
        // row of zeroes only looks like something went wrong.
        if (state.experts.loads() + state.experts.hits() == 0U) {
            text << ", nothing resident yet\n";
        } else {
            text << ", " << format_bytes(state.experts.resident_bytes()) << " resident across "
                 << state.experts.resident_experts() << " expert(s)\n"
                 << "  Expert traffic:      " << state.experts.loads() << " loads, "
                 << state.experts.hits() << " hits, " << state.experts.evictions() << " evictions, "
                 << format_bytes(state.experts.bytes_streamed()) << " streamed\n";
        }
    } else {
        text << "  Expert residency:    managed by the operating system page cache "
                "(set --expert-cache to bound it)\n";
    }
    return text.str();
}

}  // namespace litemind
