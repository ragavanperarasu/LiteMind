# 6. Mixture of experts

[`src/MoeRouter.cpp`](../src/MoeRouter.cpp) · the MoE block in
[`src/DeepSeekRunner.cpp`](../src/DeepSeekRunner.cpp)

This is the part the whole project is built around.

## The structure

Layer 0 is a normal dense feed-forward layer. The other **26 layers** replace it
with a mixture of experts:

- **64 routed experts**, of which the router picks **6** per token
- **2 shared experts**, which run for every token regardless
- Each expert is a SwiGLU block: three matrices `gate`, `up`, `down`

```
normalised hidden [2048]
      |
      +-- router gate [64, 2048] --> 64 logits --> softmax --> top 6
      |                                                         |
      |        +------------------------------------------------+
      |        v
      +--> expert 17 (weight 0.31) --+
      +--> expert  3 (weight 0.22) --+
      +--> expert 51 (weight 0.19) --+--> summed into the residual
      +--> ... 3 more               --+
      |
      +--> shared expert 0 ----------+   (always, weight 1.0)
      +--> shared expert 1 ----------+
```

## The gate, exactly

Two details invert easily, and `MoeRouter` follows DeepSeek's definition on both.

### The softmax runs *before* the cut

```
64 logits -> softmax over all 64 -> take the top 6
```

**Not** "take the top 6, then softmax over those". Softmaxing after the cut
renormalises the six weights so they sum to 1, which changes every layer's
output magnitude.

### `norm_topk_prob` is false

DeepSeek-V2-Lite sets `norm_topk_prob: false`, so the six selected weights are
**raw softmax probabilities that deliberately do not sum to one**. They typically
sum to something like 0.3–0.6, and that is correct.

`routed_scaling_factor` applies *only* when the weights are left unnormalised.
For V2-Lite it is 1.0, so it changes nothing here — but the conditional matters
for other DeepSeek checkpoints.

```cpp
struct ExpertSelection {
    std::size_t expert_index;
    float weight;        // raw softmax probability, not renormalised
};
```

## Group-limited routing

Full DeepSeek-V2 restricts selection to a subset of expert *groups*
(`n_group`, `topk_group`). V2-Lite sets both to 1, which disables it. The code
supports it so the same binary runs both.

## One expert

| | V2-Lite |
|---|---|
| Matrices | `gate`, `up`, `down` |
| `moe_intermediate_size` | 1408 |
| Parameters | 3 × 1408 × 2048 = 8,650,752 |
| Size, BF16 | **16.5 MiB** |

SwiGLU is `down · (silu(gate · x) * (up · x))`, accumulated into the shared
residual buffer scaled by the expert's weight — with no temporary per expert.

## The numbers that follow

| | |
|---|---|
| Routed experts in the model | 26 × 64 = **1664** |
| Expert executions per token | 6 × 26 = **156** |
| Fraction of experts touched | **9.4%** |
| Expert weights read per token | 156 × 16.5 MiB = **2.51 GiB** |
| Total routed expert weights | **26.81 GiB** |

That 9.4% is the sparsity the whole design exploits. It is also why
`--expert-cache` below ~2.6 GiB thrashes — see [page 7](07-expert-cache.md).

## Why the shared experts matter

The 2 shared experts run for every token, so they are always hot and the OS
keeps them resident naturally. They are counted in the 2.44 GiB of always-hot
weights, not in the 26.81 GiB that is streamed.

Next: [Expert cache](07-expert-cache.md)
