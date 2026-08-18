#include "Config.hpp"

#include "Json.hpp"

#include <cmath>
#include <sstream>

namespace litemind {
namespace {

/** yarn_get_mscale from the DeepSeek-V2 reference implementation. */
[[nodiscard]] double yarn_get_mscale(const double scale, const double mscale) noexcept {
    if (scale <= 1.0) {
        return 1.0;
    }
    return 0.1 * mscale * std::log(scale) + 1.0;
}

void read_eos_tokens(const Json& document, std::vector<std::uint32_t>& out) {
    const Json* member = document.find("eos_token_id");
    if (member == nullptr) {
        return;
    }
    out.clear();
    if (member->is_number()) {
        out.push_back(static_cast<std::uint32_t>(member->number_value()));
        return;
    }
    if (member->is_array()) {
        for (const Json& element : member->elements()) {
            if (element.is_number()) {
                out.push_back(static_cast<std::uint32_t>(element.number_value()));
            }
        }
    }
}

}  // namespace

bool Config::load(const std::filesystem::path& path, std::string& error) {
    const std::filesystem::path file = std::filesystem::is_directory(path) ? path / "config.json" : path;
    if (!std::filesystem::exists(file)) {
        error = "config.json was not found at: " + file.string();
        return false;
    }

    Json document;
    if (!Json::parse_file(file.string(), document, error)) {
        return false;
    }
    if (!document.is_object()) {
        error = file.string() + ": the top level of config.json must be a JSON object.";
        return false;
    }

    model_type = document.string_or("model_type", model_type);
    hidden_size = static_cast<std::size_t>(document.unsigned_or("hidden_size", hidden_size));
    num_hidden_layers = static_cast<std::size_t>(document.unsigned_or("num_hidden_layers", num_hidden_layers));
    num_attention_heads = static_cast<std::size_t>(document.unsigned_or("num_attention_heads", num_attention_heads));
    vocab_size = static_cast<std::size_t>(document.unsigned_or("vocab_size", vocab_size));
    max_position_embeddings =
        static_cast<std::size_t>(document.unsigned_or("max_position_embeddings", max_position_embeddings));
    rms_norm_eps = static_cast<float>(document.number_or("rms_norm_eps", rms_norm_eps));
    tie_word_embeddings = document.boolean_or("tie_word_embeddings", tie_word_embeddings);

    kv_lora_rank = static_cast<std::size_t>(document.unsigned_or("kv_lora_rank", kv_lora_rank));
    // q_lora_rank is null on DeepSeek-V2-Lite, which means the undecomposed q_proj path.
    q_lora_rank = static_cast<std::size_t>(document.unsigned_or("q_lora_rank", 0U));
    qk_nope_head_dim = static_cast<std::size_t>(document.unsigned_or("qk_nope_head_dim", qk_nope_head_dim));
    qk_rope_head_dim = static_cast<std::size_t>(document.unsigned_or("qk_rope_head_dim", qk_rope_head_dim));
    v_head_dim = static_cast<std::size_t>(document.unsigned_or("v_head_dim", v_head_dim));

    intermediate_size = static_cast<std::size_t>(document.unsigned_or("intermediate_size", intermediate_size));
    moe_intermediate_size =
        static_cast<std::size_t>(document.unsigned_or("moe_intermediate_size", moe_intermediate_size));
    n_routed_experts = static_cast<std::size_t>(document.unsigned_or("n_routed_experts", n_routed_experts));
    n_shared_experts = static_cast<std::size_t>(document.unsigned_or("n_shared_experts", n_shared_experts));
    num_experts_per_tok = static_cast<std::size_t>(document.unsigned_or("num_experts_per_tok", num_experts_per_tok));
    first_k_dense_replace =
        static_cast<std::size_t>(document.unsigned_or("first_k_dense_replace", first_k_dense_replace));
    moe_layer_freq = static_cast<std::size_t>(document.unsigned_or("moe_layer_freq", moe_layer_freq));
    norm_topk_prob = document.boolean_or("norm_topk_prob", norm_topk_prob);
    routed_scaling_factor = static_cast<float>(document.number_or("routed_scaling_factor", routed_scaling_factor));
    scoring_func = document.string_or("scoring_func", scoring_func);
    topk_method = document.string_or("topk_method", topk_method);
    n_group = static_cast<std::size_t>(document.unsigned_or("n_group", n_group));
    topk_group = static_cast<std::size_t>(document.unsigned_or("topk_group", topk_group));

    rope_theta = static_cast<float>(document.number_or("rope_theta", rope_theta));
    if (const Json* scaling = document.find("rope_scaling"); scaling != nullptr && scaling->is_object()) {
        rope_scaling.enabled = true;
        rope_scaling.type = scaling->string_or("type", scaling->string_or("rope_type", "yarn"));
        rope_scaling.factor = scaling->number_or("factor", rope_scaling.factor);
        rope_scaling.beta_fast = scaling->number_or("beta_fast", rope_scaling.beta_fast);
        rope_scaling.beta_slow = scaling->number_or("beta_slow", rope_scaling.beta_slow);
        rope_scaling.mscale = scaling->number_or("mscale", rope_scaling.mscale);
        rope_scaling.mscale_all_dim = scaling->number_or("mscale_all_dim", rope_scaling.mscale_all_dim);
        rope_scaling.original_max_position_embeddings = static_cast<std::size_t>(
            scaling->unsigned_or("original_max_position_embeddings",
                                 rope_scaling.original_max_position_embeddings));
    }

    bos_token_id = static_cast<std::uint32_t>(document.unsigned_or("bos_token_id", bos_token_id));
    read_eos_tokens(document, eos_token_ids);

    if (!validate(error)) {
        error = file.string() + ": " + error;
        return false;
    }
    source_path_ = file;
    return true;
}

float Config::softmax_scale() const noexcept {
    double scale = 1.0 / std::sqrt(static_cast<double>(qk_head_dim()));
    if (rope_scaling.enabled && rope_scaling.mscale_all_dim != 0.0) {
        const double mscale = yarn_get_mscale(rope_scaling.factor, rope_scaling.mscale_all_dim);
        scale *= mscale * mscale;
    }
    return static_cast<float>(scale);
}

bool Config::is_moe_layer(const std::size_t layer_index) const noexcept {
    if (n_routed_experts == 0U || layer_index < first_k_dense_replace) {
        return false;
    }
    const std::size_t frequency = moe_layer_freq == 0U ? 1U : moe_layer_freq;
    return layer_index % frequency == 0U;
}

bool Config::is_eos(const std::uint32_t token_id) const noexcept {
    for (const std::uint32_t candidate : eos_token_ids) {
        if (candidate == token_id) {
            return true;
        }
    }
    return false;
}

std::string Config::summary() const {
    std::ostringstream text;
    std::size_t moe_layers = 0U;
    for (std::size_t layer = 0; layer < num_hidden_layers; ++layer) {
        moe_layers += is_moe_layer(layer) ? 1U : 0U;
    }
    text << model_type << ": hidden=" << hidden_size << " layers=" << num_hidden_layers << " ("
         << (num_hidden_layers - moe_layers) << " dense, " << moe_layers << " MoE)"
         << " heads=" << num_attention_heads << " kv_lora=" << kv_lora_rank
         << " qk=" << qk_nope_head_dim << "+" << qk_rope_head_dim << " v=" << v_head_dim
         << " experts=" << n_routed_experts << "(top-" << num_experts_per_tok << ")+"
         << n_shared_experts << " shared"
         << " vocab=" << vocab_size;
    return text.str();
}

bool Config::validate(std::string& error) const {
    if (hidden_size == 0U || num_hidden_layers == 0U || num_attention_heads == 0U || vocab_size == 0U) {
        error = "hidden_size, num_hidden_layers, num_attention_heads and vocab_size must all be positive.";
        return false;
    }
    if (kv_lora_rank == 0U || qk_nope_head_dim == 0U || qk_rope_head_dim == 0U || v_head_dim == 0U) {
        error = "this build implements multi-head latent attention and needs kv_lora_rank, "
                "qk_nope_head_dim, qk_rope_head_dim and v_head_dim.";
        return false;
    }
    if (qk_rope_head_dim % 2U != 0U) {
        error = "qk_rope_head_dim must be even because rotary embedding pairs its channels.";
        return false;
    }
    if (n_routed_experts != 0U && num_experts_per_tok > n_routed_experts) {
        error = "num_experts_per_tok cannot exceed n_routed_experts.";
        return false;
    }
    if (scoring_func != "softmax") {
        error = "unsupported scoring_func '" + scoring_func + "'; this build implements 'softmax'.";
        return false;
    }
    if (topk_method != "greedy" && topk_method != "group_limited_greedy") {
        error = "unsupported topk_method '" + topk_method + "'.";
        return false;
    }
    if (rope_scaling.enabled && rope_scaling.type != "yarn") {
        error = "unsupported rope_scaling type '" + rope_scaling.type + "'; this build implements 'yarn'.";
        return false;
    }
    if (model_type.rfind("deepseek", 0U) != 0U) {
        error = "model_type '" + model_type + "' is not a DeepSeek architecture.";
        return false;
    }
    return true;
}

}  // namespace litemind
