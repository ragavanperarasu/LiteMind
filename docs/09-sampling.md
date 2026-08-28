# 9. Sampling

[`src/Sampler.cpp`](../src/Sampler.cpp) · tested by
[`tests/MoeRouterSamplerTest.cpp`](../tests/MoeRouterSamplerTest.cpp)

Turning the final layer's logits into the next token ID.

## Greedy is the default, deliberately

Greedy decoding — take the highest logit — is **reproducible**. The same prompt
gives the same continuation every run.

That matters most while a checkpoint is still being validated: if output varies
run to run, a real regression cannot be told from sampling noise. Every
comparison in this project (`fc.exe` between expert-cache budgets, reference
logit checks) depends on it.

## Sampling instead

Naming a temperature is what asks for sampling:

```powershell
--temp 0.7 --top-k 40 --top-p 0.95
```

The pipeline, in order:

1. **Repetition penalty** — divide the logits of tokens already produced by
   `--repeat-penalty`. Default 1.0, which disables it.
2. **Temperature** — divide all logits by `T`. Lower is more decisive.
3. **Top-k** — keep only the `k` highest. `0` disables.
4. **Top-p (nucleus)** — keep the smallest set whose probabilities sum to `p`.
   `1.0` disables.
5. **Draw** from what remains.

| Option | Default | Effect |
|---|---|---|
| `--temp` | (greedy) | Above 0 switches to sampling |
| `--top-k` | 40 | Candidate cut by rank |
| `--top-p` | 0.95 | Candidate cut by cumulative probability |
| `--repeat-penalty` | 1.0 | Discourage repeats |
| `--seed` | 0 | 0 draws from system entropy |

`seed()` returns the seed actually used, so a run drawn from system entropy can
still be reproduced afterwards.

## Reading raw logits

`--top-logits N` prints the N highest logits for the token after the prompt —
the model's prediction **before any sampling decision**:

```
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 1 --top-logits 10
```

```
8913  25.5393  ĠParis
 245  24.7099  Ġa
 634  24.3080  Ġone
 254  24.1904  Ġthe
```

This is the single most useful diagnostic in the project, because it **separates
a loading or arithmetic fault from a sampling one**. If the ordering here is
right, the model is loading and computing correctly, and anything odd in the
output is a sampling question.

Compare against `tools/reference_logits.py`, and compare the **identity and
order** of the top tokens, not the third decimal place — this runs float32
arithmetic over BF16 weights and the reference may accumulate differently.

## A note on the model

DeepSeek-V2-Lite is a **base** model, not instruction-tuned. It continues text;
it does not answer questions.

- good: `The capital of France is`
- good: `def fibonacci(n):`
- poor: `What is the capital of France?`

If greedy output is sensible and sampled output is not, lower `--temp` or
`--top-p`. If greedy output is wrong, the problem is upstream of sampling.

Next: [Command line and settings](10-cli-and-config.md)
