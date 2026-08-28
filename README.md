# LiteMind

DeepSeek-V2 mixture-of-experts inference on the CPU, in C++20, with no
third-party libraries. Weights stay on the SSD and are paged in as the router
asks for them, so a 31 GB checkpoint runs on a laptop that has nowhere near
31 GB of RAM.

Built and tested against
[deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
(15.7 B parameters, 2.4 B active per token).

**[Read the documentation as a website](https://ragavanperarasu.github.io/LiteMind/)**
— searchable, cross-linked, built from `docs/` by [`site/`](site/).

**[Or read the same pages in `docs/`](docs/README.md)** — one page per part:
[loading](docs/02-loading.md), [tokenizer](docs/03-tokenizer.md),
[attention](docs/04-attention.md), [rotary embedding](docs/05-rope.md),
[mixture of experts](docs/06-mixture-of-experts.md),
[expert cache](docs/07-expert-cache.md), [kernels](docs/08-kernels.md),
[sampling](docs/09-sampling.md), [the CLI](docs/10-cli-and-config.md),
[testing](docs/11-testing.md), [performance](docs/12-performance.md),
[the chat template](docs/13-chat-template.md) and [the web interface](docs/14-web-ui.md).

**New here? Start with [Setup and prerequisites](docs/00-setup.md)** — what to
install, how to verify it, how to get the checkpoint, and what to do when a step
fails.

---

## What happens when you type a prompt

This is the flow the project is built around.

```
  your prompt
      |
      v
  [1] TOKENIZE            byte-level BPE, from tokenizer.json
      |                   "The capital of" -> [100000, 651, 6884, 280]
      v
  [2] PREFILL             one position at a time, filling the KV cache
      |                   only the final position needs logits
      v
  [3] PER LAYER, PER TOKEN
      |
      +-- attention       q / kv_a / kv_b / o read straight from the
      |                   memory mapping; these are touched every step,
      |                   so the OS keeps them resident
      |
      +-- router          64 logits -> softmax -> pick the top 6
      |
      +-- LOAD FROM SSD   the 6 chosen experts are paged in
      |                   (gate, up, down: about 17 MB per expert)
      |
      +-- EXECUTE ON CPU  SwiGLU, multi-threaded BF16 kernels
      |
      +-- RELEASE TO SSD  under --expert-cache the expert is copied into a
      |                   fixed arena and the file pages are dropped; the
      |                   least recently used expert is evicted to make room
      v
  [4] SAMPLE              greedy by default, or temperature / top-k / top-p
      |
      v
  [5] STREAM              decoded and printed as each token arrives
      |
      +--> back to [3] for the next token
```

Steps 3 and 4 are the point. DeepSeek-V2-Lite holds 64 experts per layer but
sends each token to 6 of them. Roughly a tenth of the expert weights decide any
one token, so residency can track the working set instead of the checkpoint.

---

## Quick start on Windows 11

### 0. Check what you have

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_environment.ps1
```

Reports compilers and their versions, CMake, Python, MSYS2, disk space and RAM
in one pass, and says what to fix. It changes nothing. Paste its output when
asking for help.

### 1. Install a compiler

LiteMind is C++20 and 64-bit only. That means **GCC 10 or newer**, Clang 12 or
newer, or Visual Studio 2019 16.11 or newer.

> **If you have `C:\MinGW`, it will not work.** That is the old MinGW.org
> toolchain, typically GCC 6.3 from 2016. It is 32-bit and predates C++20, and
> a 32-bit process cannot memory-map a multi-gigabyte checkpoint regardless.
> Install MSYS2 alongside it — there is no need to remove it. `scripts\build.ps1`
> searches known toolchain locations before falling back to PATH, so a stale
> `C:\MinGW` will not be picked up.

Either toolchain works. MSYS2 is the smaller download.

**MSYS2 / MinGW-w64**

```powershell
winget install MSYS2.MSYS2
```

Then open **MSYS2 MINGW64** from the Start menu — not "MSYS2 MSYS" and not
MINGW32:

```bash
pacman -Syu
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
```

You only need that shell for the install. `scripts\build.ps1` finds the
compiler from any shell afterwards.

**Visual Studio 2022** — install the "Desktop development with C++" workload.
CMake comes with it.

Check what you have:

```powershell
g++ --version
g++ -dumpmachine    # must contain x86_64
```

### 2. Build

```powershell
git clone <your-repo-url> LiteMind
cd LiteMind
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -RunTests
```

Or by hand:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

There is nothing to install first. No BLAS, no OpenBLAS, no vcpkg.

### 3. Check the build before downloading 31 GB

```powershell
python tools\make_test_model.py models-test
.\build\bin\LiteMind.exe models-test --inspect
.\build\bin\LiteMind.exe models-test -p "hello" -n 8
```

This writes a 100 KB checkpoint with the same tensor names and shape
relationships as the real model. The weights are random, so the text is
meaningless — what a successful run proves is that the loader, the shapes and
the forward pass all work. Fix any failure here before downloading the weights.

### 4. Download the real weights

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_model.ps1
```

About 31 GB. The download resumes if it is interrupted. To put it on another
drive:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_model.ps1 -Destination D:\models\deepseek
```

### 5. Run

One command builds whatever changed and then asks for a prompt:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

That is the command for every run, not just the first. The build is
incremental, so nothing is recompiled when no source has changed.

Everything it can be told is in `litemind.json` at the top of the repository.
Edit that file and run the same command again — there are no flags to remember:

```json
{
  "model": "models",
  "max_tokens": 32,
  "context": 1024,
  "threads": 0,
  "expert_cache_gb": 0,
  "warm": false,
  "temperature": 0,
  "top_k": 40,
  "top_p": 0.95,
  "repeat_penalty": 1.0,
  "seed": 0,
  "show_plan": true,
  "show_tokens": false
}
```

Each prompt reports what it is about to cost before it answers:

```
Enter prompt (/exit to quit)> The capital of France is

  This prompt needs
    Tokens       6 in the prompt, up to 32 to generate  (38 forward passes)
    Experts      6 of 64 per layer across 26 MoE layers = 156 per token
                 5,928 expert executions, 16.5 MiB each, 95.5 GiB of weight reads
    Parameters   15.71 B in the model, 2.45 B active per token (15.6%)
    Settings     8 threads, context 1024, greedy sampling, experts left to the page cache

  Answer
     Paris.
```

Those figures come from `config.json`, not from a measurement, so they are
available before the work starts. They match what the run then reports.

[`docs/model-info.json`](docs/model-info.json) is an annotated breakdown of the
model itself — layer counts, expert counts, what multi-head latent attention
stores, where the 15.7 B parameters sit and which 2.4 B of them run per token.
It is documentation; nothing reads it.

The command line still works and always wins over the settings file, which is
what makes it useful for one-off comparisons:

```powershell
.\build\bin\LiteMind.exe models --inspect
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 32
.\build\bin\LiteMind.exe models -i --expert-cache 4 --warm
```

---

## Memory

By default nothing is copied out of the mapping except the norm vectors and the
router gates, which together are a few tens of megabytes. `--expert-cache`
changes that for the routed experts; see below.

| What | Size (V2-Lite) | Where it lives |
|---|---|---|
| Embeddings and output head | ~840 MB | mapped, touched every step |
| Attention, all 27 layers | ~740 MB | mapped, touched every step |
| Shared experts and the dense layer | ~1.0 GB | mapped, touched every step |
| **Routed experts** | **~29 GB** | **mapped, ~10% touched per token; copied into the arena under `--expert-cache`** |
| Norms and router gates | ~15 MB | copied to RAM as float32 |
| KV cache | ~450 KB per position | RAM |

So the always-hot part is roughly 2.6 GB, and the 29 GB of routed experts is
what streaming exists for.

`--expert-cache N` caps the resident routed experts at N gigabytes, and the cap
is a real one. An arena of N gigabytes is allocated once at load. Each expert
the router selects is copied into it and the file pages are released
immediately, so an expert is never held twice; when the arena is full, the
least recently used expert is evicted to make room. Eviction is bookkeeping in
memory LiteMind owns, so the ceiling holds regardless of what the operating
system decides to cache.

Without the flag nothing is copied. Views address the mapping directly and
residency is left to the page cache, which is faster when the machine has
enough RAM to hold the working set anyway — there is no copy to pay for, and no
budget to enforce.

The two paths produce identical output. A smaller budget buys lower peak memory
with more SSD traffic, and nothing else:

```powershell
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 16 --expert-cache 8
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 16 --expert-cache 2
```

The `Experts:` line at the end of each run reports the hit rate and the bytes
read, which is where that trade shows up.

The KV cache is the one thing that scales with context rather than with the
model, at about 450 KB per position across all layers. `--context` defaults to
1024 positions, or roughly 460 MB. Raise it only as far as you need.

`--warm` streams the always-hot weights in before the first prompt, which moves
the wait out of the first token.

### On speed

Every token reads about 2.4 B active parameters as BF16, roughly 5 GB of
traffic. Throughput is therefore a bandwidth question, not an arithmetic one,
and where that bandwidth comes from decides everything: RAM is about ten times
faster than an NVMe SSD, so the split between cached and cold experts matters
far more than the CPU does.

On a laptop with 32 GB of RAM, expect:

- **The first few tokens to be slow** — several seconds each, sometimes more.
  Nothing is cached yet and almost every expert comes off the SSD.
- **Then roughly 1 to 2 tokens per second**, once the hot weights and the
  frequently chosen experts have settled in the page cache.
- **Less RAM to mean steadier SSD traffic** and a lower steady-state rate,
  not a failure. It still runs.

Two things follow from this that are worth knowing before the first run:

**Prefill is sequential.** Prompt tokens are processed one position at a time,
so a 20-token prompt costs about what 20 generated tokens cost. A long prompt
takes minutes before any output appears. `Reading the prompt: N / M tokens`
tracks it, so you can tell progress from a hang. Start short.

**`--warm` moves the wait, it does not remove it.** It streams roughly 2.6 GB
of always-hot weights at startup, which takes a while on its own but stops the
first token from absorbing all of it.

A sensible first run:

```powershell
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 16 --warm
```

Then watch the `Experts:` line at the end. A hit rate that climbs across runs
means the page cache is doing its job.

---

## Command line

```
LiteMind [model-directory] [options]

Prompting
  -p, --prompt TEXT       Run one prompt and exit
  -i, --interactive       Keep asking for prompts until you type /exit
  -n, --max-tokens N      Tokens to generate (default 128)
      --context N         Prompt plus generated tokens (default 1024)

Sampling (greedy by default, which is reproducible)
      --temp T            Sample with temperature T instead of greedily
      --top-k K           Keep only the K most likely tokens (default 40)
      --top-p P           Nucleus sampling threshold (default 0.95)
      --repeat-penalty R  Penalise tokens already produced (default 1.0)
      --seed S            Seed the sampler for a reproducible run

Memory and speed
  -t, --threads N         Worker threads (default: one per core)
      --expert-cache GB   Cap the resident routed experts at GB gigabytes
      --warm              Stream the always-hot weights in at load

Diagnostics
      --config PATH       Read settings from PATH instead of litemind.json
      --no-plan           Skip the summary of what a prompt will cost
      --inspect           Report what is in the model directory and exit
      --show-tokens       Print the token IDs the prompt encoded to
      --top-logits N      Print the N highest logits predicted after the prompt
  -q, --quiet             Suppress progress output
  -h, --help              Show the usage text
```

---

## If it prints nothing at all

Running the executable and getting *no output whatsoever* — no text, no error,
straight back to the prompt — almost always means Windows could not start the
process because a runtime DLL was missing. PowerShell does not report that as
an error, which is why it looks like a program that ran and said nothing.

Confirm it by checking the exit code:

```powershell
.\build\bin\LiteMind.exe --help
$LASTEXITCODE
```

- `-1073741515` is `STATUS_DLL_NOT_FOUND`. That is this problem.
- `0` means it really did run, so you are looking at a stale executable from an
  older build. Rebuild.

The build links the C++ runtime statically so the executable is self-contained
and this cannot happen. If you built before that was in place, rebuild from
scratch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Clean -RunTests
```

`build.ps1` now runs the executable after building it and reports this
explicitly rather than leaving you to guess.

If static linking fails because your MinGW installation has no static
libraries, turn it off and put the MinGW `bin` directory on PATH instead:

```powershell
cmake -S . -B build -DLITEMIND_STATIC_RUNTIME=OFF -DCMAKE_BUILD_TYPE=Release
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
```

### `winget install MSYS2.MSYS2` fails with exit code 1

MSYS2 is already installed. The installer refuses to write into a directory
that already holds an installation and exits with 1 — it reports
`TargetDirectoryInUse` in its log.

Do not try to reinstall it. Install the toolchain package into the copy you
already have:

```powershell
& "C:\msys64\usr\bin\bash.exe" -lc "pacman -Sy --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-ninja"
```

If pacman complains about a stale keyring or a corrupt database:

```powershell
& "C:\msys64\usr\bin\bash.exe" -lc "pacman -Syu --noconfirm"
```

### "does not support this" / "requires the language dialect CXX20"

The compiler is too old. Check which one CMake picked — the line reading
`Check for working CXX compiler:` in its output names the path. If it says
`C:/MinGW/bin/c++.exe`, see the compiler note in step 1.

`scripts\build.ps1` reports every compiler it found, with the reason each was
rejected, and picks a usable one automatically. To name one yourself:

```powershell
powershell -File scripts\build.ps1 -CompilerPath C:\msys64\mingw64\bin\g++.exe
```

### "Does not match the generator used previously"

The build directory remembers how it was configured and CMake will not switch
generators in place. `build.ps1` detects this and clears the directory for you.
By hand:

```powershell
Remove-Item -Recurse -Force build
```

---

## If the output looks wrong

Work through this in order. Each step rules out one layer of the stack.

**1. Does the checkpoint load?**

```powershell
.\build\bin\LiteMind.exe models --inspect
```

Every shape is checked against `config.json` at load, so a mismatch names the
tensor, the shape found and the shape expected rather than producing noise.

**2. Are the tokens right?**

```powershell
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 1 --show-tokens
```

Compare against Hugging Face:

```bash
python3 tools/reference_logits.py models --prompt "The capital of France is"
```

If the token IDs differ, the tokenizer is the problem and nothing downstream
can match. Fix that first.

**3. Are the logits right?**

```powershell
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 1 --top-logits 10
```

`tools/reference_logits.py` prints the same list from the reference
implementation. Logits are what the model predicts before any sampling
decision, so this separates a loading or arithmetic fault from a sampling one.

Compare the *identity and order* of the top tokens, not the third decimal
place. This runs float32 arithmetic over BF16 weights and the reference may
accumulate differently, so small numeric differences are expected. If the
ordering agrees, the model is loading and computing correctly.

**4. Is it just sampling?**

Greedy decoding is the default precisely because it is reproducible. If greedy
output is sensible and sampled output is not, lower `--temp` or `--top-p`.

Note that DeepSeek-V2-Lite is a **base** model, not an instruction-tuned one.
It continues text; it does not answer questions. Prompt it accordingly:

- good: `The capital of France is`
- good: `def fibonacci(n):`
- poor: `What is the capital of France?`

---

## What the code does

The architecture is DeepSeek-V2, which differs from a conventional transformer
in two ways that both had to be implemented exactly.

**Multi-head latent attention.** Keys and values are not stored per head.
Instead the hidden state is compressed to a `kv_lora_rank` latent vector, and
`kv_b_proj` expands that into every head's key and value. The rotary part of
the key is *decoupled*: a single `qk_rope_head_dim` vector shared by all heads,
concatenated onto each head's non-rotary key.

**Rotary embedding with a permutation.** DeepSeek reshapes the rotary channels
from `[d]` to `[d/2, 2]` and transposes before rotating, so channels stored
interleaved as `(a0, b0, a1, b1, …)` become `(a0, a1, …, b0, b1, …)`. Rotating
the raw interleaved layout pairs the wrong channels together and produces
fluent-looking noise. See `src/Rope.cpp`; `tests/RopeTest.cpp` checks the
defining property, that a query-key dot product depends only on the relative
position.

**YaRN frequency scaling.** The frequencies are interpolated, not plain inverse
powers of theta: low-frequency channels are divided by the scaling factor, high
frequency channels are left alone, and a linear ramp blends the band between.
The attention softmax scale also carries a squared magnitude correction —
without it, attention is about 1.6x too flat on this model.

**Mixture-of-experts routing.** The softmax runs over all 64 experts *before*
the top-6 cut, and `routed_scaling_factor` applies only when the selected
weights are left unnormalised. DeepSeek-V2-Lite sets `norm_topk_prob` to false,
so its expert weights are raw softmax probabilities that deliberately do not
sum to one. Softmaxing after the cut would renormalise them and change every
layer's output magnitude.

### Layout

| File | What it does |
|---|---|
| `src/Json.cpp` | JSON parser, so config lookups are structural rather than substring matches |
| `src/Config.cpp` | Parses and validates `config.json`; every shape comes from here |
| `src/MappedFile.cpp` | Cross-platform mmap with prefetch and release hints |
| `src/SafeTensor.cpp` | Maps one shard and parses its header |
| `src/WeightStore.cpp` | Resolves a tensor name to a pointer, enforcing shape and dtype |
| `src/ExpertCache.cpp` | The bounded expert working set: a fixed-slot RAM arena with LRU eviction |
| `src/Gemm.cpp` | BF16 kernels, hand-vectorised for AVX2 with a portable fallback |
| `src/Threading.cpp` | A fixed thread pool with a blocking `parallel_for` |
| `src/Rope.cpp` | YaRN rotary embedding with DeepSeek's channel permutation |
| `src/Attention.cpp` | Multi-head latent attention for one position |
| `src/MoeRouter.cpp` | Expert selection, following DeepSeek's gate exactly |
| `src/Tokenizer.cpp` | Byte-level BPE with added tokens and streaming decode |
| `src/DeepSeekRunner.cpp` | Weight resolution, the forward pass, prefill and decode |
| `src/Cli.cpp` | Argument parsing, inspection, the interactive loop |

### Tests

```powershell
ctest --test-dir build -C Release --output-on-failure
```

`JsonTest`, `SafeTensorTest`, `KernelTest`, `RopeTest`,
`MoeRouterSamplerTest`, `ExpertCacheTest` and `AttentionTest` need no model
files.
`TokenizerRoundTrip` runs only when `models/tokenizer.json` is present.

---

## Known limits

- **BF16 only.** The large matrices must be BF16, which is how DeepSeek-V2
  ships. A float16 or quantised checkpoint is rejected with a message naming
  the type it found. There is no quantisation yet.
- **Prefill is sequential.** Prompt tokens are processed one position at a
  time rather than batched, so a long prompt costs about as much as generating
  the same number of tokens.
- **The KV cache stores expanded keys and values.** This trades memory for
  speed: caching the latent instead would use about seven times less memory but
  would re-run `kv_b_proj` for every past position on every step. The
  "absorbed" formulation avoids both costs and is not implemented here.
- **The pre-tokenizer is hand-written.** It follows the split rules in
  DeepSeek's `tokenizer.json` and is exact for ASCII. Coverage of less common
  scripts is approximate; `--show-tokens` exists so you can check.
- **One sequence at a time.** No batching, no server mode.

## License

MIT. See `LICENSE`.
