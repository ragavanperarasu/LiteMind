# LiteMind documentation

One page per part of the system. Read them in order the first time; after that
each stands on its own.

| # | Page | What it covers |
|---|---|---|
| 1 | [Overview](01-overview.md) | The problem, the idea, and how a prompt flows end to end |
| 2 | [Loading the checkpoint](02-loading.md) | `Json`, `Config`, `MappedFile`, `SafeTensor`, `WeightStore` |
| 3 | [Tokenizer](03-tokenizer.md) | Byte-level BPE, added tokens, streaming decode |
| 4 | [Attention](04-attention.md) | Multi-head latent attention and the KV cache |
| 5 | [Rotary embedding](05-rope.md) | YaRN scaling and DeepSeek's channel permutation |
| 6 | [Mixture of experts](06-mixture-of-experts.md) | The router, expert selection, SwiGLU |
| 7 | [Expert cache](07-expert-cache.md) | Streaming experts from SSD and the bounded arena |
| 8 | [Kernels and threading](08-kernels.md) | BF16 matrix-vector products, the thread pool |
| 9 | [Sampling](09-sampling.md) | Greedy, temperature, top-k, top-p, repetition penalty |
| 10 | [Command line and settings](10-cli-and-config.md) | `litemind.json`, `run.ps1`, the flags |
| 11 | [Testing](11-testing.md) | The suites, the synthetic model, reference logits |
| 12 | [Performance](12-performance.md) | Measured numbers and the memory/traffic trade |
| 13 | [The chat template](13-chat-template.md) | Why a chat checkpoint needs its User/Assistant frame |
| 14 | [The web interface](14-web-ui.md) | The JSON event stream, the Node backend, the React front end |

[`model-info.json`](model-info.json) is an annotated dump of the architecture
itself — layer counts, expert counts, where the parameters sit.

## The short version

DeepSeek-V2-Lite holds 15.7 billion parameters but runs only 2.4 billion of
them for any one token, because each mixture-of-experts layer sends a token to
6 of its 64 experts. LiteMind keeps the whole 29.3 GiB checkpoint on the SSD as
a memory mapping and lets that routing decision decide what to bring into RAM.
The result runs on a laptop with no GPU.

Everything else in this documentation is a consequence of that one idea.
