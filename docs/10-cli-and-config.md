# 10. Command line and settings

[`src/Cli.cpp`](../src/Cli.cpp) · [`src/DeepSeekRunner.cpp`](../src/DeepSeekRunner.cpp) ·
[`litemind.json`](../litemind.json) · [`scripts/run.ps1`](../scripts/run.ps1)

## One command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

Builds whatever changed, then asks for a prompt. This is the command for every
run, not just the first — the build is incremental, so nothing is recompiled
when no source has changed.

## The settings file

Everything configurable lives in `litemind.json` at the repository root. Edit it
and run the same command again.

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

**Precedence is file, then command line.** Anything passed as an argument
overrides the file, which is what makes the flags useful for one-off
comparisons without editing anything.

`--config other.json` reads a different file. Naming one that does not exist
fails with exit code 2 rather than silently falling back to defaults.

## What each prompt reports

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

These figures are **derived from `config.json`, not measured**, so they are
available before the work starts. They match what the run then reports —
5,928 executions and 95.5 GiB are exactly what the expert cache counts.

`--no-plan`, or `"show_plan": false`, turns it off.

## The flags

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
      --expert-cache GB   Copy routed experts into a bounded GB-gigabyte arena
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

## `--inspect`

Reports what is in a model directory **without loading any weights**:

```
config.json
  deepseek_v2: hidden=2048 layers=27 (1 dense, 26 MoE) heads=16 kv_lora=512 ...
  rope_scaling: yarn factor=40 original_context=4096 mscale=0.707/0.707
  routing: greedy, norm_topk_prob=false, routed_scaling_factor=1

Weight shards
  model-00001-of-000004.safetensors  8.00 GiB, 1329 tensors
  ...
  total: 4 shard(s), 5291 tensors, 29.3 GiB
```

This is the first thing to run against a new checkpoint, and the first thing to
paste when asking for help.

## `--warm`

Streams the 2.44 GiB of always-hot weights in before the first prompt. It
**moves the wait, it does not remove it** — startup takes longer, but the first
token no longer absorbs all of it.

## `DeepSeekRunner`

The class the CLI drives. `load()` maps every shard and resolves every weight
the architecture needs, failing with a specific message when the checkpoint and
`config.json` disagree. `generate()` runs prefill then decode. `memory_report()`
produces the memory block printed at startup.

Next: [Testing](11-testing.md)
