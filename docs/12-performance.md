# 12. Performance

All figures measured on a Windows laptop: **8 hardware threads**, AVX2 + FMA
kernels, NVMe SSD, DeepSeek-V2-Lite.

## Where the time goes

Every token reads about 2.4 B active parameters as BF16 — roughly 5 GB of
traffic. **Throughput is a bandwidth question, not an arithmetic one.** RAM is
about ten times faster than an NVMe SSD, so the split between cached and cold
experts decides everything; the CPU barely matters.

## Measured

| Configuration | Hit rate | Weight reads | Throughput |
|---|---|---|---|
| No budget (OS page cache) | — | — | **3.47 tok/s** |
| `--expert-cache 4` | 19.7% | 76.7 GiB | 0.89 tok/s |
| `--expert-cache 2` | 0.0% | 95.5 GiB | 0.76 tok/s |

Prompt: `"The capital of France is"`, `-n 32`. All three produced
**byte-identical output**.

Other measurements from the same runs:

| | |
|---|---|
| Load time, 29.3 GiB checkpoint | **0.2 s** |
| Prefill | 6 tokens in 8.3–9.5 s (~1.5 s/token) |
| Expert requests | 5,928 for 38 forward passes (156/token) |

## Load time is 0.2 seconds

Mapping is lazy. Opening the checkpoint parses four SafeTensors headers and
resolves 5,291 tensor names to pointers; **no payload is read**. This is the
clearest single demonstration that nothing is being copied.

## Prefill is sequential

Prompt tokens are processed one position at a time, not batched, so a 20-token
prompt costs about what 20 generated tokens cost. A long prompt takes minutes
before any output appears.

`Reading the prompt: N / M tokens` tracks it, so progress can be told from a
hang. Start short.

Note the asymmetry in the numbers above: **1.5 s per prompt token against 0.7 s
per generated token**. That is the cold-versus-warm expert story showing up
directly — by decode time the frequently chosen experts have settled in cache.

## The counter-intuitive result

**The bounded arena is about 4× slower than letting the OS do it.**

This is not a bug, and it is worth understanding:

1. A machine with 32 GB of RAM caches far more than a 4 GiB arena ever could.
2. The arena's `release()` after copying actively *discards* the page-cache
   entries that made the fast path fast.
3. On top of that it pays a `memcpy` that mmap gave away free.

So the two modes serve different machines:

| Situation | Use |
|---|---|
| RAM to spare | **No budget.** The page cache wins. |
| RAM-constrained | **A budget ≥ 4 GiB.** Bounded memory is the difference between running and not. |
| Anything below ~2.6 GiB | **Nothing.** Worst of both — see below. |

## The 2.51 GiB threshold

A single token touches 156 experts × 16.5 MiB = **2.51 GiB** of expert weights.

A budget below that cannot hold one token's working set, so LRU evicts every
expert before the token that wanted it has finished. The result is a **0% hit
rate by construction** — which is exactly what the 2 GiB row above shows, at 124
slots against 156 needed.

## What would make it faster

| Idea | Expected effect |
|---|---|
| **Quantisation** (INT8/INT4) | The biggest win by far — halves or quarters the bytes per token, and the workload is bandwidth-bound |
| **Batched prefill** | Long prompts stop costing a token each |
| **Absorbed MLA** | ~7× smaller KV cache; folds `kv_b_proj` into the query projection |
| More threads | Little. The limit is arrival, not arithmetic |

## Reproducing these numbers

```powershell
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 32
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 32 --expert-cache 4
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 32 --expert-cache 2
```

Read the `Experts:` line at the end of each run for the hit rate and bytes read.
To confirm the outputs match:

```powershell
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 32 -q > out-none.txt
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 32 -q --expert-cache 4 > out-c4.txt
fc.exe out-none.txt out-c4.txt
```

Use `fc.exe`, not `fc` — in PowerShell bare `fc` is an alias for `Format-Custom`.

Sampling is greedy by default, so this is reproducible without a seed. Hit rates
climb across repeated runs as the page cache warms, so compare runs from the
same session.
