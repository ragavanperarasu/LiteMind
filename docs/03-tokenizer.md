# 3. Tokenizer

[`src/Tokenizer.cpp`](../src/Tokenizer.cpp) · [`include/Tokenizer.hpp`](../include/Tokenizer.hpp)

Byte-level BPE, read directly from `tokenizer.json`. No Python, no Hugging Face
`tokenizers` library.

## Why byte-level

Every input byte maps to a printable code point *before* any merging happens.
Two consequences:

- **No token is ever unknown.** Worst case, a piece falls back to its 256
  single-byte tokens.
- **Arbitrary binary round-trips.** Encoding then decoding returns the input
  exactly, which is what `TokenizerRoundTrip` checks.

This is why token text looks odd when printed: `Ġcapital` is " capital", where
`Ġ` is the printable stand-in for a space.

## What it reads

| From `tokenizer.json` | Used for |
|---|---|
| `model.vocab` | Token string to ID |
| `model.merges` | The BPE merge table, in priority order |
| `added_tokens` | Special tokens such as `<｜begin▁of▁sentence｜>` |

V2-Lite has **100002 tokenizer entries** against a `vocab_size` of 102400 in
`config.json`. The extra room in the embedding matrix is unused; the two extra
tokenizer entries are the added control tokens. BOS is 100000, EOS is 100001.

## Encoding

1. **Pre-tokenize** — split the text on the rules in `tokenizer.json`.
2. **Byte-encode** each piece to printable code points.
3. **Merge** greedily by merge priority until no pair can be merged.
4. **Look up** each final piece in the vocabulary.

`encode(text, add_bos = true)` prefixes the BOS token by default.

> **Known limit.** The pre-tokenizer is hand-written. It follows the split rules
> in DeepSeek's `tokenizer.json` and is exact for ASCII; coverage of less common
> scripts is approximate. `--show-tokens` exists so you can check a tokenisation
> against the reference before blaming the model.

## Streaming decode

`Tokenizer::StreamDecoder` is a small but necessary piece. Byte-level BPE will
happily put the first byte of a multi-byte UTF-8 character in one token and the
rest in the next. Printing each token as it arrives would write invalid UTF-8 to
the console.

```cpp
StreamDecoder decoder(tokenizer);
std::string text = decoder.push(token_id);   // "" until the character completes
...
text += decoder.flush();                     // any held-back bytes at the end
```

`push` returns only what became printable; incomplete sequences are held back.

## Checking a tokenisation

```powershell
.\build\bin\LiteMind.exe models -p "The capital of France is" -n 1 --show-tokens
```

```
100000  <｜begin▁of▁sentence｜>
   549  The
  6077  Ġcapital
   280  Ġof
  7239  ĠFrance
   317  Ġis
```

Compare against the reference:

```bash
python3 tools/reference_logits.py models --prompt "The capital of France is"
```

**If the token IDs differ, stop there.** Nothing downstream can match, and every
later comparison will mislead you.

Next: [Attention](04-attention.md)
