# 2. Loading the checkpoint

Five components turn a directory of Hugging Face files into typed pointers the
decoder can multiply with. None of them copies a weight.

```
config.json ──> Config ──┐
                         ├──> WeightStore ──> WeightView (pointer + shape + dtype)
*.safetensors ─> SafeTensor ─> MappedFile
```

## `Json` — [`src/Json.cpp`](../src/Json.cpp)

A complete JSON document model: null, boolean, number, string, array, object.

**Why not substring matching.** A model directory holds several JSON files, and
the same key name appears in more than one object — `type` occurs both at the
top level and inside `rope_scaling`. Searching for `"factor"` in the raw text
finds whichever comes first, which is a bug waiting to happen. A real tree makes
lookups exact.

The accessors take a fallback so a missing key is not an error:

```cpp
document.unsigned_or("hidden_size", 2048U);
document.number_or("rms_norm_eps", 1e-6);
const Json* scaling = document.find("rope_scaling");
```

`null` falls through to the fallback rather than becoming `0` — DeepSeek-V2-Lite
stores `"q_lora_rank": null`, and reading it as zero happens to be correct here,
but only by accident. The distinction is kept deliberately.

## `Config` — [`src/Config.cpp`](../src/Config.cpp)

Parses `config.json` into typed fields and validates them. **Every shape the
runtime uses comes from here**, never from a compiled-in constant, so the same
binary runs V2-Lite and full V2 without a rebuild.

Key fields:

| Field | V2-Lite | Meaning |
|---|---|---|
| `hidden_size` | 2048 | Width of the residual stream |
| `num_hidden_layers` | 27 | Decoder layers |
| `first_k_dense_replace` | 1 | Layer 0 is dense; the rest are MoE |
| `n_routed_experts` | 64 | Experts held per MoE layer |
| `num_experts_per_tok` | 6 | Experts a token is sent to |
| `n_shared_experts` | 2 | Experts every token runs |
| `moe_intermediate_size` | 1408 | Width inside one routed expert |
| `kv_lora_rank` | 512 | Latent width for keys and values |
| `qk_nope_head_dim` / `qk_rope_head_dim` | 128 / 64 | Non-rotary and rotary query-key channels |

`is_moe_layer(i)` answers whether layer `i` is dense or mixture-of-experts, and
`softmax_scale()` applies YaRN's magnitude correction (see [page 5](05-rope.md)).

## `MappedFile` — [`src/MappedFile.cpp`](../src/MappedFile.cpp)

A read-only memory mapping of an entire file, with a Windows branch and a POSIX
branch behind one interface.

**Why mapping and not reading.** Mapping reserves address space up front, but a
page costs physical memory only once it is touched, and the OS reclaims cold
pages on its own. Opening a 9 GB shard costs milliseconds. Reading it would cost
9 GB of RAM you do not have.

Two hints steer that behaviour:

| Call | Windows | POSIX |
|---|---|---|
| `prefetch(offset, length)` | `PrefetchVirtualMemory` | `madvise(MADV_WILLNEED)` |
| `release(offset, length)` | `VirtualUnlock` | `madvise(MADV_DONTNEED)` |

`PrefetchVirtualMemory` arrived in Windows 8 and MinGW does not always declare
it, so it is resolved at runtime through `GetProcAddress`; prefetching is only a
hint, so a system without it still works.

**Both calls are advisory.** They ask; the OS decides. That is exactly why the
bounded arena in [page 7](07-expert-cache.md) exists.

## `SafeTensor` — [`src/SafeTensor.cpp`](../src/SafeTensor.cpp)

One mapped shard. The SafeTensors format is:

```
[8 bytes little-endian header length][JSON header][payload]
```

The header names every tensor with its dtype, shape and byte range. Opening a
shard maps the file and parses only the header — payload pages arrive when a
weight is first touched.

A declared byte range that falls outside the file returns `nullptr` rather than
handing back a pointer into nothing.

## `WeightStore` — [`src/WeightStore.cpp`](../src/WeightStore.cpp)

Indexes every `.safetensors` file in the model directory and resolves a tensor
name to a `WeightView`:

```cpp
struct WeightView {
    const SafeTensor* shard;   // which mapping
    const Tensor* meta;        // shape, dtype, byte range
    const std::byte* bytes;    // the payload, inside the mapping
};
```

`require(name, expected_shape, expected_type, error)` is the important one. It
checks the shape and dtype and, on a mismatch, fills `error` with the tensor
name, the shape found and the shape expected. **This is what turns a wrong
checkpoint into a clear message instead of fluent-looking noise.**

`read_float32` is the one exception to zero-copy: the small norm vectors and the
router gates are widened to float32 and kept in RAM, about 15 MB in total. They
are touched every step and are too small to be worth re-widening.

## What ends up in RAM

| | Size | Where |
|---|---|---|
| Norm vectors, router gates | ~15 MB | Copied, float32 |
| Everything else | 29.3 GiB | Pointers into the mapping |

Next: [Tokenizer](03-tokenizer.md)
