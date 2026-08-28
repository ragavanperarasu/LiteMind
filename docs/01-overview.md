# 1. Overview

## The problem

An open model of useful quality needs 32–80 GB of GPU memory. The alternative
is a metered cloud API, which bills per token and sends every prompt to someone
else's server. Neither works for a college lab, a rural clinic or a small NGO:
no GPU, intermittent internet, and data that must not leave the room.

The gap is not model quality — the weights are free to download. The gap is
that **every runtime assumes the whole model fits in fast memory at once.**

## The idea

DeepSeek-V2-Lite is a *mixture-of-experts* model. Each of its 26 MoE layers
holds 64 independent feed-forward "experts", and a small router picks **6** of
them per token. Roughly a tenth of the expert weights decide any single token.

So residency can track the *working set* instead of the *checkpoint*:

| | Size | Behaviour |
|---|---|---|
| Always-hot weights | 2.44 GiB | Touched every step, stay resident |
| Routed experts | 26.81 GiB | ~10% touched per token, streamed |

LiteMind memory-maps the checkpoint and lets the router's decision drive paging.
A token pays for six experts, not for the model.

## How a prompt flows

```
  your prompt
      |
      v
  [1] TOKENIZE            byte-level BPE, from tokenizer.json
      |                   "The capital of" -> [100000, 549, 6077, 280]
      v
  [2] PREFILL             one position at a time, filling the KV cache
      |                   only the final position needs logits
      v
  [3] PER LAYER, PER TOKEN   (27 layers)
      |
      +-- attention       multi-head latent attention; q / kv_a / kv_b / o are
      |                   read straight from the mapping and stay resident
      |
      +-- router          64 logits -> softmax -> pick the top 6
      |
      +-- LOAD            the 6 chosen experts arrive from the SSD
      |                   (gate, up, down: 16.5 MiB per expert)
      |
      +-- EXECUTE         SwiGLU, multi-threaded BF16 kernels
      |
      +-- RELEASE         experts beyond the budget go back to the SSD
      v
  [4] SAMPLE              greedy by default, or temperature / top-k / top-p
      |
      v
  [5] STREAM              decoded and printed as each token arrives
      |
      +--> back to [3] for the next token
```

## The modules

| Layer | Files | Page |
|---|---|---|
| Reading the checkpoint | `Json`, `Config`, `MappedFile`, `SafeTensor`, `WeightStore` | [2](02-loading.md) |
| Text in and out | `Tokenizer` | [3](03-tokenizer.md) |
| The model | `Attention`, `KvCache`, `Rope`, `MoeRouter` | [4](04-attention.md), [5](05-rope.md), [6](06-mixture-of-experts.md) |
| Memory policy | `ExpertCache` | [7](07-expert-cache.md) |
| Arithmetic | `Gemm`, `Threading` | [8](08-kernels.md) |
| Choosing a token | `Sampler` | [9](09-sampling.md) |
| Driving it | `DeepSeekRunner`, `Cli` | [10](10-cli-and-config.md) |

`DeepSeekRunner` is the one that ties them together: it resolves every weight at
load, then runs prefill and decode.

## Constraints the design accepts

- **C++20, no third-party libraries.** No BLAS, no vcpkg, no Python at runtime.
  Every dependency is the standard library, an OS header, or compiler intrinsics.
- **BF16 only.** The large matrices must be BF16, which is how DeepSeek-V2
  ships. A quantised checkpoint is rejected by name rather than misread.
- **One sequence at a time.** No batching, no server mode.
- **Prefill is sequential.** A prompt token costs about what a generated token
  costs, so a long prompt takes minutes before any output appears.
