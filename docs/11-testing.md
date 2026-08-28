# 11. Testing

```powershell
ctest --test-dir build -C Release --output-on-failure
```

or, as part of a build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -RunTests
```

## The framework

There isn't one. [`tests/TestSupport.hpp`](../tests/TestSupport.hpp) is a
dependency-free check helper: each test is its own executable reporting through
its exit code, so CTest needs nothing else. No GoogleTest, no Catch2 — the tests
are as self-contained as the executable.

```cpp
check(condition, "what should be true");
check_close(actual, expected, tolerance, "what should be close");
return report("MyTest");
```

## The suites

| Suite | What it establishes |
|---|---|
| `JsonTest` | Parses a real `config.json` shape, including `null` and nested objects |
| `SafeTensorTest` | Header parsing, byte ranges, out-of-range rejection |
| `KernelTest` | BF16 widening, matvec, rms_norm, softmax; AVX2 and scalar agree |
| `RopeTest` | The relative-position property (see [page 5](05-rope.md)) |
| `MoeRouterSamplerTest` | Softmax-before-cut, unnormalised weights, sampling |
| `ExpertCacheTest` | The copy is real, LRU order, capacity is never exceeded |
| `AttentionTest` | Latent attention shapes and the decoupled rope key |
| `TokenizerRoundTrip` | Encode→decode returns the input exactly |

The first seven **need no model files** and run everywhere.
`TokenizerRoundTrip` is registered only when `models/tokenizer.json` is present;
otherwise it is built but not run, so a fresh clone still gets a clean test run.

## Testing without 31 GB

```powershell
python tools\make_test_model.py models-test
.\build\bin\LiteMind.exe models-test --inspect
.\build\bin\LiteMind.exe models-test -p "hello" -n 8
```

This writes a **100 KB checkpoint** with the same tensor names and shape
*relationships* as the real model — 3 layers, 4 routed experts, top-2. The
weights are random, so the text is meaningless.

What a successful run proves is that the loader, the shapes and the forward pass
all work. **Fix any failure here before downloading the weights.**

It is also how the expert cache is exercised end to end: the same prompt under
no budget and under a tiny budget must produce byte-identical output.

## Checking the maths against the reference

```bash
python3 tools/reference_logits.py models --prompt "The capital of France is"
```

Prints token IDs and top logits from the Hugging Face implementation. This is
the only part of the project with an external dependency — `torch` and
`transformers` — and it is **development-only**: nothing in the build or the
runtime needs them.

Work through a suspected problem in this order, because each step rules out one
layer of the stack:

1. **Does the checkpoint load?** `--inspect`. Every shape is checked against
   `config.json`, so a mismatch names the tensor and both shapes.
2. **Are the tokens right?** `--show-tokens`. If the IDs differ from the
   reference, the tokenizer is the problem and nothing downstream can match.
3. **Are the logits right?** `--top-logits 10`. Compare the *identity and order*
   of the top tokens, not the third decimal — this runs float32 arithmetic over
   BF16 weights and the reference may accumulate differently.
4. **Is it just sampling?** Greedy is the default precisely so this is a
   meaningful question. If greedy is sensible and sampled is not, lower `--temp`.

## Windows code paths

```bash
python3 tools/check_win32_syntax.py
```

Compiles the Windows branches of `MappedFile.cpp` and `main.cpp` against a stub
`windows.h`, so a change that breaks the Windows build can be caught from a
POSIX machine.

Next: [Performance](12-performance.md)
