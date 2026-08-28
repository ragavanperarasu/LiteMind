# 7. Expert cache

[`src/ExpertCache.cpp`](../src/ExpertCache.cpp) · tested by
[`tests/ExpertCacheTest.cpp`](../tests/ExpertCacheTest.cpp)

Turning "a token needs only 6 of 64 experts" into a memory policy.

## Two modes

### Default: zero-copy, OS-managed

Without `--expert-cache`, nothing is copied. `WeightView`s address the mapping
directly, and residency is the operating system's page cache. The cache issues
`prefetch()` before an expert is used so the read arrives as one sequential
request instead of a storm of page faults.

This is **faster on a machine with RAM to spare**, because the OS will happily
cache far more than any budget you would set, and there is no copy to pay for.

### With `--expert-cache N`: a bounded arena

An arena of N gigabytes is allocated once at load. Then, per selected expert:

```
prefetch(gate, up, down)     ask the OS for one sequential read
memcpy -> arena slot         take our own copy
release(gate, up, down)      drop the file pages
execute from the arena
```

The `release` is the important step. Without it the expert would sit in the
arena *and* in the page cache — counted against RAM twice.

Eviction is least-recently-used and is **pure bookkeeping**: the slot returns to
a free list and the next load overwrites it. Nothing is asked of the OS, which
is what makes the budget a ceiling rather than a request.

## Why the arena exists

`release()` is advisory. `VirtualUnlock` on Windows only removes pages from the
working set; the OS may keep them, and may equally evict experts you believe are
resident. So a hint-based budget is a *claim you cannot enforce*.

The arena is memory LiteMind owns and evicts itself. **The budget becomes a
number you can prove.**

> **One honest caveat.** The arena is a hard cap on what LiteMind holds. It is
> not a hard cap on what Windows caches underneath — the OS may still hold the
> released file pages in its standby list.

## The slot allocator

Every routed expert in a DeepSeek-V2 model has the same shape, so one slot size
serves all of them. `reserve()` takes a representative expert at load time and
computes a layout with each matrix aligned to 64 bytes:

```
capacity = budget / stride
```

Knowing capacity before the first prompt is why `reserve()` is called at load
rather than lazily — the memory report can then say "room for 248 expert(s)"
instead of appearing part-way through a token.

The arena is allocated with `new std::byte[]` rather than a vector, so a 4 GB
budget does not spend a multi-gigabyte zeroing pass at startup. Pages commit as
experts are copied in.

## The threshold that matters

**A single token touches 156 experts × 16.5 MiB = 2.51 GiB.**

A budget below that evicts every expert before it can be reused — a **0% hit
rate by construction**. Measured on a 27-layer V2-Lite run:

| Budget | Slots | Hit rate | Read | Throughput |
|---|---|---|---|---|
| none | — | — | — | **3.47 tok/s** |
| 4 GiB | 248 | 19.7% | 76.7 GiB | 0.89 tok/s |
| 2 GiB | 124 | **0.0%** | 95.5 GiB | 0.76 tok/s |

124 slots is fewer than the 156 one token needs. 248 clears it with room for
cross-token reuse, hence 19.7%.

**So use 0, or at least 4.** A value between 0 and ~2.6 is the worst of both.

## Correctness

The two paths produce **identical output** — verified with `fc.exe` across no
budget, 2 GiB and 4 GiB. A budget buys lower peak memory with more SSD traffic
and nothing else.

`ExpertCacheTest` proves the copy is real by overwriting the source buffer
behind the cache's back and checking the resident copy is unaffected, then
checks LRU ordering, capacity and that residency never exceeds the budget.

Next: [Kernels and threading](08-kernels.md)
