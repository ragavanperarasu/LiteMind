# 4. Attention

[`src/Attention.cpp`](../src/Attention.cpp) · [`src/KvCache.cpp`](../src/KvCache.cpp)

DeepSeek-V2 uses **multi-head latent attention** (MLA), which differs from
conventional attention in a way that had to be implemented exactly.

## The idea

Conventional attention projects the hidden state into a key and a value *per
head* and caches both. MLA instead compresses the hidden state into **one
low-rank latent vector** of `kv_lora_rank` (512) values, and expands that into
every head's key and value through `kv_b_proj`.

```
hidden [2048]
   |
   +-- kv_a_proj --> latent [512 + 64]
   |                   |         |
   |                   |         +-- shared rope key (all heads)
   |                   |
   |                   +-- kv_b_proj --> per head: key_nope [128], value [128]
   |
   +-- q_proj -------> per head: query [128 nope + 64 rope]
```

## The decoupled rotary key

This is the detail most likely to be got wrong. The rotary part of the key is
**not** per head. It is a single `qk_rope_head_dim` (64) vector shared by every
head, concatenated onto each head's non-rotary key.

So a query has 192 channels — 128 non-rotary plus 64 rotary — while the cached
key is 128 per-head channels plus 64 channels shared across all heads. In the
cache that shows up directly:

```cpp
std::vector<float> key_nope_;  // [capacity, heads, qk_nope_head_dim]
std::vector<float> key_rope_;  // [capacity, qk_rope_head_dim]  <- no head axis
std::vector<float> value_;     // [capacity, heads, v_head_dim]
```

## The weights

| Weight | Shape | Note |
|---|---|---|
| `q_proj` | `[heads * 192, 2048]` | Used when `q_lora_rank == 0`, as in V2-Lite |
| `q_a_proj` / `q_b_proj` | via rank | Used when the query is also low-rank |
| `kv_a_proj` | `[512 + 64, 2048]` | The tail 64 rows are the shared rope key |
| `kv_b_proj` | `[heads * (128 + 128), 512]` | Expands the latent |
| `o_proj` | `[2048, heads * 128]` | Output projection |

## The KV cache

`KvCache` stores keys and values **expanded**, not as the latent.

That is a deliberate trade. Caching the latent would use about seven times less
memory, but it would re-run `kv_b_proj` for every past position on every step —
turning a cheap quadratic into an expensive one. Expanding once at append time
and caching the result removes that matrix product from the inner loop.

Cost per position, at V2-Lite's shapes:

| Component | Floats | |
|---|---|---|
| `key_nope` | 16 heads × 128 | 2048 |
| `key_rope` | 64, shared | 64 |
| `value` | 16 heads × 128 | 2048 |
| **Per layer** | | **4160 floats = 16.25 KiB** |
| **× 27 layers** | | **438.75 KiB per position** |

At the default `--context 1024` that is **438.75 MiB** — the one thing that
scales with context rather than with the model. Raise `--context` only as far as
you need.

> **Not implemented.** The "absorbed" MLA formulation avoids both the memory of
> expanded caching and the recomputation of latent caching, by folding
> `kv_b_proj` into the query projection. It is the obvious next optimisation.

## Softmax scale

The attention scale is not the usual `1/sqrt(head_dim)`. YaRN applies a squared
magnitude correction on top — see [page 5](05-rope.md). Without it, attention on
this model is about 1.6× too flat, which produces plausible but degraded output.

`Config::softmax_scale()` returns 0.114721 for V2-Lite, which `--inspect` prints
so you can check it.

## Execution

One `Attention` instance owns its scratch buffers and is reused across every
layer and every step, so **a decoding step performs no heap allocation**. Head
loops are distributed across the thread pool.

`forward()` returns `false` only when the KV cache is full, which the caller
reports as a context-length limit rather than an error.

Next: [Rotary embedding](05-rope.md)
