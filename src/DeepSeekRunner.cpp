/**
 * DeepSeekRunner.cpp — end-to-end DeepSeek-V2 (small) inference.
 *
 * Memory : weights stored as BF16 (uint16_t) ~30 GB, plus float32 LM head.
 * Compute: fused BF16→F32 dot products; compiler auto-vectorises with AVX2/FMA.
 * CBLAS  : used only for sdot / saxpy in attention (small vectors, already F32).
 *
 * config.json key values:
 *   hidden=2048  layers=27(1 dense+26 MoE)  heads=16
 *   kv_lora_rank=512  qk_nope=128  qk_rope=64  v_head=128
 *   dense_inter=10944  moe_inter=1408
 *   routed_experts=64  shared=2  top_k=6
 *   vocab=102400  rms_eps=1e-6  rope_theta=10000 (YaRN mscale=0.707 factor=40)
 */
#include "DeepSeekRunner.hpp"

#include "MoeRouter.hpp"
#include "SafeTensor.hpp"
#include "Sampler.hpp"
#include "WeightReader.hpp"

#include <cblas.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace litemind {
namespace {

// ─── Constants ───────────────────────────────────────────────────────────────
constexpr std::size_t   kH      = 2048;
constexpr std::size_t   kNL     = 27;
constexpr std::size_t   kNH     = 16;
constexpr std::size_t   kKvR    = 512;
constexpr std::size_t   kDnope  = 128;
constexpr std::size_t   kDrope  = 64;
constexpr std::size_t   kDq     = kDnope + kDrope;   // 192
constexpr std::size_t   kDv     = 128;
constexpr std::size_t   kID     = 10944;              // dense inter
constexpr std::size_t   kIM     = 1408;               // moe inter
constexpr std::size_t   kNExp   = 64;
constexpr std::size_t   kNSh    = 2;
constexpr std::size_t   kTopK   = 6;
constexpr std::size_t   kVocab  = 102400;
constexpr float         kEps    = 1e-6F;
constexpr float         kTheta  = 10000.0F;
constexpr float         kMs     = 0.707F;
constexpr float         kFac    = 40.0F;
constexpr std::size_t   kOrig   = 4096;
constexpr std::uint32_t kEos    = 100001;
constexpr std::size_t   kMaxPos = 2048;

// ─── BF16 → F32 ──────────────────────────────────────────────────────────────
inline float b2f(std::uint16_t b) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16U);
}

// ─── Matrix × vector (BF16 weights, F32 input/output) ────────────────────────
// Each output element is computed as a fused BF16-widening dot product.
// At -O3 the inner loop auto-vectorises with AVX2 (vpmovsxwd + vfmadd).
void mv(const std::uint16_t* __restrict__ W,
        const float*          __restrict__ x,
        float*                __restrict__ y,
        std::size_t out, std::size_t in) noexcept {
    for (std::size_t o = 0; o < out; ++o) {
        const std::uint16_t* row = W + o * in;
        float acc = 0.0F;
        for (std::size_t i = 0; i < in; ++i)
            acc += b2f(row[i]) * x[i];
        y[o] = acc;
    }
}

// ─── RMSNorm ──────────────────────────────────────────────────────────────────
void rms_norm(float* x, const float* w, std::size_t n) noexcept {
    float ss = cblas_sdot(static_cast<blasint>(n), x, 1, x, 1);
    const float inv = 1.0F / std::sqrt(ss / static_cast<float>(n) + kEps);
    for (std::size_t i = 0; i < n; ++i) x[i] *= inv * w[i];
}

// ─── RoPE ─────────────────────────────────────────────────────────────────────
std::vector<float> build_rope(std::size_t dim, std::size_t max_pos) {
    const float ms = kMs * (1.0F + std::log(kFac) / std::log(static_cast<float>(kOrig)));
    std::vector<float> t(max_pos * dim);
    for (std::size_t p = 0; p < max_pos; ++p)
        for (std::size_t i = 0; i < dim / 2; ++i) {
            const float a = static_cast<float>(p) / std::pow(kTheta, 2.0F * i / dim);
            t[p * dim + i]           = ms * std::cos(a);
            t[p * dim + i + dim / 2] = ms * std::sin(a);
        }
    return t;
}

void apply_rope(float* v, std::size_t pos,
                const std::vector<float>& tbl, std::size_t dim) noexcept {
    const float* row = tbl.data() + pos * dim;
    const std::size_t half = dim / 2;
    for (std::size_t i = 0; i < half; ++i) {
        const float x = v[i], y = v[i + half];
        v[i]        = x * row[i]        - y * row[i + half];
        v[i + half] = x * row[i + half] + y * row[i];
    }
}

// ─── SiLU ─────────────────────────────────────────────────────────────────────
inline float silu(float x) noexcept { return x / (1.0F + std::exp(-x)); }

// ─── SwiGLU FFN ───────────────────────────────────────────────────────────────
// out += scale * down( silu(gate(x)) * up(x) )
// All weight matrices are BF16.
void swiglu(const std::uint16_t* wg, const std::uint16_t* wu, const std::uint16_t* wd,
            const float* x, float* out, std::size_t inter, float scale = 1.0F) {
    std::vector<float> gate(inter), up(inter), mid(inter);
    mv(wg, x, gate.data(), inter, kH);
    mv(wu, x, up.data(),   inter, kH);
    for (std::size_t i = 0; i < inter; ++i) mid[i] = silu(gate[i]) * up[i];
    // down: [kH, inter] BF16
    for (std::size_t o = 0; o < kH; ++o) {
        const std::uint16_t* row = wd + o * inter;
        float acc = 0.0F;
        for (std::size_t i = 0; i < inter; ++i) acc += b2f(row[i]) * mid[i];
        out[o] += scale * acc;
    }
}

// ─── Weight containers ────────────────────────────────────────────────────────
struct AttnW {
    std::vector<std::uint16_t> q;       // [kNH*kDq, kH]
    std::vector<std::uint16_t> kv_a;    // [kKvR+kDrope, kH]
    std::vector<float>         kv_a_ln; // [kKvR]
    std::vector<std::uint16_t> kv_b;    // [kNH*(kDnope+kDv), kKvR]
    std::vector<std::uint16_t> o;       // [kH, kNH*kDv]
};

struct ExpertW { std::vector<std::uint16_t> g, u, d; };

struct LayerW {
    std::vector<float> ln1, ln2;
    AttnW              attn;
    bool               moe{false};
    // dense FFN
    std::vector<std::uint16_t> wg, wu, wd;
    // MoE
    std::vector<std::uint16_t> router;
    std::vector<ExpertW>       experts;
    std::vector<std::uint16_t> sh_g, sh_u, sh_d;
};

// KV cache: pre-expanded per-token entries (avoid re-running kv_b per query)
struct KvEntry {
    std::vector<float> k_nope; // [kNH * kDnope]
    std::vector<float> k_rope; // [kDrope]
    std::vector<float> v;      // [kNH * kDv]
};
using KvCache = std::vector<KvEntry>;

// ─── Weight store ─────────────────────────────────────────────────────────────
class WStore {
public:
    bool open(const std::filesystem::path& dir, std::string& err) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            if (ec || !e.is_regular_file(ec) || e.path().extension() != ".safetensors") continue;
            SafeTensor shard;
            std::string se;
            if (!shard.open(e.path(), se)) { err = "open shard: " + se; return false; }
            for (const Tensor& t : shard.tensors()) { idx_[t.name()] = shards_.size(); tensors_.push_back(t); }
            shards_.push_back(std::move(shard));
        }
        if (shards_.empty()) { err = "no .safetensors in " + dir.string(); return false; }
        return true;
    }

    std::vector<std::uint16_t> bf16(const std::string& name, std::string& err) const {
        auto it = idx_.find(name);
        if (it == idx_.end()) { err = "missing tensor: " + name; return {}; }
        const Tensor* m = nullptr;
        for (const Tensor& t : tensors_) if (t.name() == name) { m = &t; break; }
        WeightReader r;
        std::vector<std::byte> bytes;
        if (!r.read_bytes(shards_[it->second], *m, bytes, err)) return {};
        std::vector<std::uint16_t> out(bytes.size() / 2);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = static_cast<std::uint16_t>(std::to_integer<unsigned char>(bytes[i * 2]))
                   | (static_cast<std::uint16_t>(std::to_integer<unsigned char>(bytes[i * 2 + 1])) << 8U);
        return out;
    }

    std::vector<float> f32(const std::string& name, std::string& err) const {
        auto raw = bf16(name, err);
        std::vector<float> out(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) out[i] = b2f(raw[i]);
        return out;
    }

private:
    std::vector<SafeTensor>                      shards_;
    std::vector<Tensor>                          tensors_;
    std::unordered_map<std::string, std::size_t> idx_;
};

// ─── Layer loading ────────────────────────────────────────────────────────────
bool load_layer(std::size_t idx, LayerW& lw, const WStore& ws, std::string& err) {
    const std::string p = "model.layers." + std::to_string(idx) + ".";
    lw.ln1 = ws.f32(p + "input_layernorm.weight", err);           if (lw.ln1.empty()) return false;
    lw.ln2 = ws.f32(p + "post_attention_layernorm.weight", err);   if (lw.ln2.empty()) return false;

    lw.attn.q      = ws.bf16(p + "self_attn.q_proj.weight", err);             if (lw.attn.q.empty())      return false;
    lw.attn.kv_a   = ws.bf16(p + "self_attn.kv_a_proj_with_mqa.weight", err); if (lw.attn.kv_a.empty())   return false;
    lw.attn.kv_a_ln= ws.f32 (p + "self_attn.kv_a_layernorm.weight", err);     if (lw.attn.kv_a_ln.empty())return false;
    lw.attn.kv_b   = ws.bf16(p + "self_attn.kv_b_proj.weight", err);          if (lw.attn.kv_b.empty())   return false;
    lw.attn.o      = ws.bf16(p + "self_attn.o_proj.weight", err);              if (lw.attn.o.empty())      return false;

    lw.moe = (idx >= 1);
    if (!lw.moe) {
        lw.wg = ws.bf16(p + "mlp.gate_proj.weight", err); if (lw.wg.empty()) return false;
        lw.wu = ws.bf16(p + "mlp.up_proj.weight",   err); if (lw.wu.empty()) return false;
        lw.wd = ws.bf16(p + "mlp.down_proj.weight", err); if (lw.wd.empty()) return false;
    } else {
        lw.router = ws.bf16(p + "mlp.gate.weight", err); if (lw.router.empty()) return false;
        lw.experts.resize(kNExp);
        for (std::size_t e = 0; e < kNExp; ++e) {
            const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
            lw.experts[e].g = ws.bf16(ep + "gate_proj.weight", err); if (lw.experts[e].g.empty()) return false;
            lw.experts[e].u = ws.bf16(ep + "up_proj.weight",   err); if (lw.experts[e].u.empty()) return false;
            lw.experts[e].d = ws.bf16(ep + "down_proj.weight", err); if (lw.experts[e].d.empty()) return false;
        }
        lw.sh_g = ws.bf16(p + "mlp.shared_experts.gate_proj.weight", err); if (lw.sh_g.empty()) return false;
        lw.sh_u = ws.bf16(p + "mlp.shared_experts.up_proj.weight",   err); if (lw.sh_u.empty()) return false;
        lw.sh_d = ws.bf16(p + "mlp.shared_experts.down_proj.weight", err); if (lw.sh_d.empty()) return false;
    }
    return true;
}

// ─── Attention forward ────────────────────────────────────────────────────────
void attn_fwd(const float* x, const AttnW& aw, KvCache& kvc,
              std::size_t pos, const std::vector<float>& rope, float* out) {
    // Q projection + per-head RoPE on rope slice
    std::vector<float> q(kNH * kDq);
    mv(aw.q.data(), x, q.data(), kNH * kDq, kH);
    for (std::size_t h = 0; h < kNH; ++h)
        apply_rope(q.data() + h * kDq + kDnope, pos, rope, kDrope);

    // KV compression: kv_a → latent c_kv + decoupled rope key
    std::vector<float> kva(kKvR + kDrope);
    mv(aw.kv_a.data(), x, kva.data(), kKvR + kDrope, kH);
    std::vector<float> c_kv(kva.begin(), kva.begin() + static_cast<std::ptrdiff_t>(kKvR));
    std::vector<float> k_rope_raw(kva.begin() + static_cast<std::ptrdiff_t>(kKvR), kva.end());
    rms_norm(c_kv.data(), aw.kv_a_ln.data(), kKvR);
    apply_rope(k_rope_raw.data(), pos, rope, kDrope);

    // Pre-expand K_nope and V for all heads: kv_b [kNH*(kDnope+kDv), kKvR]
    KvEntry entry;
    entry.k_nope.resize(kNH * kDnope);
    entry.k_rope = k_rope_raw;
    entry.v.resize(kNH * kDv);
    {
        std::vector<float> flat(kNH * (kDnope + kDv));
        mv(aw.kv_b.data(), c_kv.data(), flat.data(), kNH * (kDnope + kDv), kKvR);
        for (std::size_t h = 0; h < kNH; ++h) {
            const float* src = flat.data() + h * (kDnope + kDv);
            std::copy(src,          src + kDnope, entry.k_nope.data() + h * kDnope);
            std::copy(src + kDnope, src + kDnope + kDv, entry.v.data() + h * kDv);
        }
    }
    kvc.push_back(std::move(entry));
    const std::size_t T = kvc.size();

    // Causal dot-product attention (F32 K/V already expanded)
    std::vector<float> ctx(kNH * kDv, 0.0F);
    std::vector<float> scores(T);
    const float scale = 1.0F / std::sqrt(static_cast<float>(kDnope + kDrope));
    const auto TI = static_cast<blasint>(T);

    for (std::size_t h = 0; h < kNH; ++h) {
        const float* qn = q.data() + h * kDq;
        const float* qr = qn + kDnope;
        for (std::size_t t = 0; t < T; ++t) {
            const auto& e = kvc[t];
            scores[t] = scale * (
                cblas_sdot(static_cast<blasint>(kDnope), qn, 1, e.k_nope.data() + h * kDnope, 1) +
                cblas_sdot(static_cast<blasint>(kDrope), qr, 1, e.k_rope.data(), 1));
        }
        float mx = *std::max_element(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(T));
        float sm = 0.0F;
        for (std::size_t t = 0; t < T; ++t) { scores[t] = std::exp(scores[t] - mx); sm += scores[t]; }
        for (std::size_t t = 0; t < T; ++t) scores[t] /= sm;

        float* ch = ctx.data() + h * kDv;
        for (std::size_t t = 0; t < T; ++t)
            cblas_saxpy(static_cast<blasint>(kDv), scores[t], kvc[t].v.data() + h * kDv, 1, ch, 1);

        static_cast<void>(TI);
    }
    mv(aw.o.data(), ctx.data(), out, kH, kNH * kDv);
}

// ─── Layer forward ────────────────────────────────────────────────────────────
void layer_fwd(const float* xi, const LayerW& lw, KvCache& kvc,
               std::size_t pos, const std::vector<float>& rope, float* xo) {
    std::copy(xi, xi + kH, xo);

    // Attention block
    std::vector<float> h(xo, xo + kH);
    rms_norm(h.data(), lw.ln1.data(), kH);
    std::vector<float> ao(kH, 0.0F);
    attn_fwd(h.data(), lw.attn, kvc, pos, rope, ao.data());
    cblas_saxpy(static_cast<blasint>(kH), 1.0F, ao.data(), 1, xo, 1);

    // FFN block
    std::vector<float> fi(xo, xo + kH);
    rms_norm(fi.data(), lw.ln2.data(), kH);
    std::vector<float> fo(kH, 0.0F);

    if (!lw.moe) {
        swiglu(lw.wg.data(), lw.wu.data(), lw.wd.data(), fi.data(), fo.data(), kID);
    } else {
        std::vector<float> logits(kNExp);
        mv(lw.router.data(), fi.data(), logits.data(), kNExp, kH);
        const auto sel = MoeRouter::select_top_k(logits, kNExp, kTopK, false);
        for (const auto& s : sel)
            swiglu(lw.experts[s.expert_index].g.data(),
                   lw.experts[s.expert_index].u.data(),
                   lw.experts[s.expert_index].d.data(),
                   fi.data(), fo.data(), kIM, s.weight);
        swiglu(lw.sh_g.data(), lw.sh_u.data(), lw.sh_d.data(),
               fi.data(), fo.data(), kNSh * kIM);
    }
    cblas_saxpy(static_cast<blasint>(kH), 1.0F, fo.data(), 1, xo, 1);
}

} // namespace

// ─── Impl ─────────────────────────────────────────────────────────────────────
struct DeepSeekRunner::Impl {
    std::vector<std::uint16_t> embed;       // [kVocab, kH] BF16
    std::vector<LayerW>        layers;
    std::vector<float>         final_ln;    // [kH]
    std::vector<float>         lm_head_f32; // [kVocab, kH] F32 (pre-converted)
    std::vector<float>         rope;
    bool loaded{false};
};

// ─── Constructor ──────────────────────────────────────────────────────────────
DeepSeekRunner::DeepSeekRunner(const std::filesystem::path& dir, std::string& err) {
    impl_ = new Impl();
    Impl& m = *impl_;

    std::cout << "Loading weights" << std::flush;
    WStore ws;
    if (!ws.open(dir, err)) return;

    m.embed    = ws.bf16("model.embed_tokens.weight", err); if (m.embed.empty())    return;
    m.final_ln = ws.f32 ("model.norm.weight",         err); if (m.final_ln.empty()) return;
    {
        // Pre-convert LM head to F32 once — avoids per-step BF16→F32 conversion
        // for this [102400 × 2048] matrix (largest hot matrix in the model).
        auto raw = ws.bf16("lm_head.weight", err);
        if (raw.empty()) return;
        m.lm_head_f32.resize(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) m.lm_head_f32[i] = b2f(raw[i]);
    }
    m.rope = build_rope(kDrope, kMaxPos);

    m.layers.resize(kNL);
    for (std::size_t i = 0; i < kNL; ++i) {
        std::cout << " " << i << std::flush;
        if (!load_layer(i, m.layers[i], ws, err)) {
            std::cout << "\nFailed at layer " << i << ": " << err << "\n";
            return;
        }
    }
    std::cout << "\nAll weights loaded.\n";
    m.loaded = true;
}

DeepSeekRunner::~DeepSeekRunner() { delete impl_; }
bool DeepSeekRunner::ready() const noexcept { return impl_ && impl_->loaded; }

// ─── Generate ─────────────────────────────────────────────────────────────────
std::string DeepSeekRunner::generate(const Tokenizer& tok,
                                     const std::vector<std::uint32_t>& prompt,
                                     std::size_t max_new) const {
    if (!ready()) return "";
    const Impl& m = *impl_;

    std::vector<KvCache> kvc(kNL);
    std::vector<float>   h(kH), buf(kH), norm(kH), logits(kVocab);
    std::vector<std::uint32_t> gen;
    gen.reserve(max_new);

    for (std::size_t step = 0; step < prompt.size() + max_new; ++step) {
        std::uint32_t tid;
        if (step < prompt.size()) {
            tid = prompt[step];
        } else {
            tid = static_cast<std::uint32_t>(Sampler::select_next(logits));
            if (tid == kEos) break;
            gen.push_back(tid);
            std::cout << tok.decode({tid}) << std::flush;
            if (gen.size() >= max_new) break;
        }

        // Embed token
        const std::size_t off = static_cast<std::size_t>(tid) * kH;
        for (std::size_t i = 0; i < kH; ++i) h[i] = b2f(m.embed[off + i]);

        // Transformer layers
        for (std::size_t l = 0; l < kNL; ++l) {
            layer_fwd(h.data(), m.layers[l], kvc[l], step, m.rope, buf.data());
            std::swap(h, buf);
        }

        // Final norm + LM head (F32, uses CBLAS SGEMV directly)
        std::copy(h.begin(), h.end(), norm.begin());
        rms_norm(norm.data(), m.final_ln.data(), kH);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<blasint>(kVocab), static_cast<blasint>(kH),
                    1.0F, m.lm_head_f32.data(), static_cast<blasint>(kH),
                    norm.data(), 1, 0.0F, logits.data(), 1);
    }

    if (gen.empty()) return "";
    return tok.decode(gen);
}

} // namespace litemind
