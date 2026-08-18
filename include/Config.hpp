#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace litemind {

/** YaRN rotary-scaling parameters, as stored under config.json's rope_scaling. */
struct RopeScaling final {
    bool enabled{false};
    std::string type{"yarn"};
    double factor{1.0};
    double beta_fast{32.0};
    double beta_slow{1.0};
    double mscale{1.0};
    double mscale_all_dim{0.0};
    std::size_t original_max_position_embeddings{4096U};
};

/**
 * @brief The DeepSeek-V2 architecture description parsed from config.json.
 *
 * Every shape the runtime uses comes from here rather than from compiled-in
 * constants, so the same binary runs DeepSeek-V2-Lite and DeepSeek-V2 without
 * a rebuild, and a mismatched checkpoint fails with a clear message instead of
 * silently producing noise.
 */
class Config final {
public:
    Config() = default;

    /** Reads and validates config.json from a model directory or an explicit file path. */
    [[nodiscard]] bool load(const std::filesystem::path& path, std::string& error);

    /** Returns the path that load() actually read. */
    [[nodiscard]] const std::filesystem::path& source_path() const noexcept { return source_path_; }

    // ── Core dimensions ──────────────────────────────────────────────────────
    std::string model_type{"deepseek_v2"};
    std::size_t hidden_size{2048U};
    std::size_t num_hidden_layers{27U};
    std::size_t num_attention_heads{16U};
    std::size_t vocab_size{102400U};
    std::size_t max_position_embeddings{163840U};
    float rms_norm_eps{1e-6F};
    bool tie_word_embeddings{false};

    // ── Multi-head latent attention ──────────────────────────────────────────
    std::size_t kv_lora_rank{512U};
    std::size_t q_lora_rank{0U};  ///< Zero means q_proj is a single dense matrix.
    std::size_t qk_nope_head_dim{128U};
    std::size_t qk_rope_head_dim{64U};
    std::size_t v_head_dim{128U};

    // ── Feed-forward and mixture of experts ──────────────────────────────────
    std::size_t intermediate_size{10944U};
    std::size_t moe_intermediate_size{1408U};
    std::size_t n_routed_experts{64U};
    std::size_t n_shared_experts{2U};
    std::size_t num_experts_per_tok{6U};
    std::size_t first_k_dense_replace{1U};
    std::size_t moe_layer_freq{1U};
    bool norm_topk_prob{false};
    float routed_scaling_factor{1.0F};
    std::string scoring_func{"softmax"};
    std::string topk_method{"greedy"};
    std::size_t n_group{1U};
    std::size_t topk_group{1U};

    // ── Rotary embedding ─────────────────────────────────────────────────────
    float rope_theta{10000.0F};
    RopeScaling rope_scaling{};

    // ── Special tokens ───────────────────────────────────────────────────────
    std::uint32_t bos_token_id{100000U};
    std::vector<std::uint32_t> eos_token_ids{100001U};

    // ── Derived quantities ───────────────────────────────────────────────────

    /** Per-head query/key width: qk_nope_head_dim + qk_rope_head_dim. */
    [[nodiscard]] std::size_t qk_head_dim() const noexcept {
        return qk_nope_head_dim + qk_rope_head_dim;
    }

    /**
     * The attention logit scale.
     *
     * This is q_head_dim^-0.5, multiplied by the squared YaRN magnitude
     * correction when rope_scaling supplies a non-zero mscale_all_dim. Omitting
     * that correction leaves attention roughly 1.6x too flat on DeepSeek-V2.
     */
    [[nodiscard]] float softmax_scale() const noexcept;

    /** True when layer_index uses a mixture-of-experts feed-forward block. */
    [[nodiscard]] bool is_moe_layer(std::size_t layer_index) const noexcept;

    /** The shared-expert block's intermediate width. */
    [[nodiscard]] std::size_t shared_expert_intermediate_size() const noexcept {
        return moe_intermediate_size * n_shared_experts;
    }

    /** True when token_id ends generation. */
    [[nodiscard]] bool is_eos(std::uint32_t token_id) const noexcept;

    /** A one-line human-readable summary for the console. */
    [[nodiscard]] std::string summary() const;

private:
    [[nodiscard]] bool validate(std::string& error) const;

    std::filesystem::path source_path_;
};

}  // namespace litemind
