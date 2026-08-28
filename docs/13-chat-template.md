# 13 · The chat template

**Source:** [`include/ChatTemplate.hpp`](../include/ChatTemplate.hpp), [`src/ChatTemplate.cpp`](../src/ChatTemplate.cpp)
**Test:** [`tests/ChatTemplateTest.cpp`](../tests/ChatTemplateTest.cpp)

## Why a prompt is not sent as typed

A base checkpoint is trained to continue text. An instruction-tuned one is
trained on a fixed conversational frame, and it only answers when it sees that
frame. Handed a bare `hi`, a chat model has no assistant turn to complete, so it
does the only thing its pre-training taught it — it continues the fragment as
ordinary prose.

That failure is worth dwelling on, because it does not look like a formatting
problem. The replies are fluent, grammatical and full of accurate world
knowledge; they simply answer a question nobody asked. It is easy to mistake for
a routing fault, a weight-loading fault, or a broken checkpoint, and to go
looking for the bug in the expert selection. The bug is in the prompt.

## What the frame is

Hugging Face stores it in `tokenizer_config.json` under `chat_template`, as a
Jinja program. DeepSeek's reduces, for a single user turn, to:

```
<bos>User: {prompt}\n\nAssistant:
```

The trailing `Assistant:` carries no space. The leading space of the reply is a
token the model was trained to produce, and supplying it here would put the
model one token off the distribution it learned.

A system message, when there is one, is emitted bare and first, with no marker
of its own:

```
<bos>{system}\n\nUser: {prompt}\n\nAssistant:
```

The beginning-of-sequence token is left to the tokenizer, which already prepends
it, so `ChatTemplate::apply` returns the text from `User:` onward.

## Why the Jinja is not interpreted

Interpreting the stored template properly would mean shipping a Jinja engine,
which is a dependency and a parser this project has no other use for. Instead
the frames are written out in C++ and the stored program is matched against them
by its distinguishing literals — `'User: '`, `'Assistant: '`, `'Assistant:'`.

An unrecognised template is **refused rather than guessed at**. Formatting a
prompt with the wrong family's frame is worse than not formatting it at all: it
fails silently, degrading output without any error to notice. So a checkpoint
carrying a template LiteMind does not know produces a warning and raw prompts,
which is at least a known state.

## Deciding once, from the checkpoint

The decision is made when the model loads, not per prompt:

| Checkpoint | `chat` setting | Result |
|---|---|---|
| Base (no template) | `true` | Raw prompt — nothing to apply |
| Chat (recognised) | `true` | Frame applied |
| Chat (recognised) | `false` | Raw prompt, reported on stderr |
| Chat (unrecognised) | either | Raw prompt, with a warning |

Leaving `chat` on therefore costs a base checkpoint nothing. Turning it off is
what the logit comparison in [`tools/reference_logits.py`](../tools/reference_logits.py)
needs, since that script feeds the model unformatted text.

## Seeing it work

`--show-tokens` prints what actually reached the model:

```
$ LiteMind models -p "hi" -n 1 --show-tokens
  Chat template: applying the checkpoint's User/Assistant frame.
  7 prompt tokens:
  100000  <bos>
    2724  User
      27  :
    3555  Ġhi
     185  Ċ
     185  Ċ
   16094  ĠAssistant
```

Without the frame the same prompt is two tokens, and the model has nothing to
answer.

## What this does not do

Only one turn is formatted. The interactive loop treats every prompt as a fresh
conversation and re-runs the prefill, so the model has no memory of the previous
exchange. Multi-turn history would mean carrying the KV cache across prompts and
appending `Assistant: {reply}<eos>` for each completed turn — the frame above
already describes how, but nothing in the runner keeps that state yet.
