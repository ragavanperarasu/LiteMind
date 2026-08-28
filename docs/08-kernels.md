# 8. Kernels and threading

[`src/Gemm.cpp`](../src/Gemm.cpp) · [`src/Threading.cpp`](../src/Threading.cpp) ·
tested by [`tests/KernelTest.cpp`](../tests/KernelTest.cpp)

## The workload is bandwidth-bound, not compute-bound

Every token reads about 2.4 B active parameters as BF16 — roughly 5 GB of
traffic. The arithmetic is trivial by comparison. So the kernels are written
around one goal: **move fewer bytes**.

That is why weights stay BF16 in the mapping and are widened to float32 *inside
the inner loop* rather than converted up front. It halves the bytes each product
pulls from RAM or SSD.

Widening BF16 costs **one shift**: BF16 is the top 16 bits of a float32, sharing
its exponent layout exactly.

```cpp
float widen(std::uint16_t bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    // reinterpret as float
}
```

That is the whole conversion. It is also why BF16 was chosen over float16 for
this kind of work.

## The kernels

| Function | Use |
|---|---|
| `matvec_bf16` | `out = W · x`, rows spread across the pool |
| `matvec_bf16_accumulate` | `out += scale · (W · x)` — MoE contributions, no temporary per expert |
| `matvec_bf16_serial` | Small matrices where threading would cost more than it saves |
| `matvec_f32` | The router gate, kept widened in RAM |
| `rms_norm` | Normalisation, in place |
| `softmax` | Max-subtraction form |
| `dot`, `axpy`, `widen_bf16` | Building blocks |

`matvec_bf16_accumulate` exists specifically for the MoE block: six experts
accumulate into one shared residual buffer, each scaled by its router weight,
without allocating a temporary for each.

## Vectorisation

Hand-written AVX2 + FMA intrinsics with a **portable scalar fallback**. There is
no BLAS, no OpenBLAS, no MKL.

`kernel_description()` reports what the binary was compiled for, and the CLI
prints it at startup:

```
AVX2 + FMA BF16 kernels (AVX-512 available but unused), 8 threads
```

or, on a build without native tuning:

```
portable scalar BF16 kernels (rebuild with -march=native for AVX2)
```

The fallback is not decorative — it is what makes the project build and run
anywhere, and `KernelTest` checks both paths agree.

## One precision decision worth knowing

`rms_norm` accumulates its sum of squares in **double** precision:

> At `hidden_size` 2048 the float32 error is small but systematic, and it
> compounds across 27 layers.

This is the kind of thing that produces "the output is nearly right but not
quite" and is almost impossible to find later. The cost is negligible — one
reduction per normalisation.

## The thread pool

Decoding one token runs a few hundred matrix-vector products. Spawning threads
per product would cost more than the arithmetic, so `ThreadPool` creates its
workers once and parks them on a condition variable between calls.

```cpp
pool.parallel_for(row_count, [&](std::size_t row) { /* ... */ });
```

`parallel_for` blocks until every index has been processed. With a single worker
it runs inline, with no synchronisation at all.

`-t/--threads` sets the worker count; the default is one per hardware thread.
On a bandwidth-bound workload more threads stop helping fairly quickly — the
limit is how fast weights arrive, not how fast they are multiplied.

Next: [Sampling](09-sampling.md)
