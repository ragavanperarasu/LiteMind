# 5. Rotary embedding

[`src/Rope.cpp`](../src/Rope.cpp) · tested by [`tests/RopeTest.cpp`](../tests/RopeTest.cpp)

Two details here are easy to get wrong, and **both degrade output silently
rather than loudly**. That is what makes them dangerous: the model still emits
fluent English, just worse, so there is nothing obvious to debug.

## Detail 1: the channel permutation

DeepSeek's reference implementation reshapes the rotary channels from `[d]` to
`[d/2, 2]` and transposes before rotating. Channels stored interleaved as

```
(a0, b0, a1, b1, a2, b2, ...)
```

become

```
(a0, a1, a2, ..., b0, b1, b2, ...)
```

Rotating the raw interleaved layout pairs the **wrong channels** together. The
arithmetic completes, the shapes match, nothing errors — and the output is
subtly wrong.

The permuted layout is kept afterwards rather than undone. That is safe because
queries and keys are permuted *identically*, and their dot product is unchanged
by a shared permutation.

## Detail 2: YaRN frequency interpolation

The frequencies are not plain inverse powers of `rope_theta`. YaRN interpolates:

- **Low-frequency channels** are divided by the scaling factor (40 for V2-Lite)
- **High-frequency channels** are left alone
- **A linear ramp** blends the band between, bounded by `beta_fast` and
  `beta_slow`

This is what extends the usable context from `original_max_position_embeddings`
(4096) to `max_position_embeddings` (163840).

## Detail 3: the magnitude correction

YaRN also scales the attention softmax. `mscale` and `mscale_all_dim` (both
0.707 for V2-Lite) combine into a squared correction folded into
`Config::softmax_scale()`.

**Without it attention is about 1.6× too flat.** The model still produces text;
it is simply less decisive than it should be.

## The test

`RopeTest` does not compare against golden numbers. It asserts the *defining
property* of rotary embedding:

> A query–key dot product depends only on the **relative** position of the two,
> never on their absolute positions.

So rotating query at position 5 and key at position 3 must give the same dot
product as positions 105 and 103. A permutation error, a frequency error or an
off-by-one in the table all break this property, which makes it a much stronger
test than checking a few values.

## Configuration

Everything comes from `config.json`'s `rope_scaling`:

```json
"rope_scaling": {
  "type": "yarn", "factor": 40,
  "beta_fast": 32, "beta_slow": 1,
  "mscale": 0.707, "mscale_all_dim": 0.707,
  "original_max_position_embeddings": 4096
}
```

`--inspect` prints the schedule that was built, so you can confirm it was read
as intended:

```
rope_scaling: yarn factor=40 original_context=4096 mscale=0.707/0.707
```

Cosine and sine tables are built once at load for every position up to the
configured context, then indexed during decoding.

Next: [Mixture of experts](06-mixture-of-experts.md)
