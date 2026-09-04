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

## Where the reply ends

Applying the frame is only half of it. Nothing in the frame tells the model to
*stop* after answering — the training data simply continues into the next turn,
so left alone the model answers your question and then cheerfully plays both
sides of the conversation:

```
 Hi, how can I help you?
User: Can you tell me the weather for tomorrow in Dallas?
Assistant: It looks like it will be a sunny day...
```

Everything from `User:` onward is the model talking to itself. The marker that
opens the next turn is where its answer actually ended, so those markers are
stop sequences: `\nUser:` and `\nAssistant:`.

The leading newline is deliberate. It anchors the marker to a turn boundary, so
a reply that happens to contain the words mid-sentence — "ask the User: politely"
— is not truncated.

### Why the text cannot simply be printed as it arrives

A marker is generated one token at a time, and it can straddle two fragments.
Printing eagerly would leak `\nUser` into the reply whenever the `:` had not
arrived yet. So `StopScanner` releases text only once it can no longer become
part of a marker:

| Accumulated text | Held back | Why |
|---|---|---|
| `done.` | 0 bytes | Nothing here can start a marker |
| `done.\n` | 1 byte | A newline could still open a turn |
| `done.\nUser` | 5 bytes | A `:` would complete the marker |
| `done.\nUser:` | 0 bytes | Complete — the reply is cut here instead |

When a marker completes, the text before it is emitted, the result is truncated
at the marker, and the decoder's held-back bytes are discarded rather than
flushed: they belong to text that is no longer part of the reply.

## Remembering an earlier turn

The model has no memory. Nothing is carried between prompts — not the KV cache,
not the process, nothing — so remembering an earlier exchange means putting that
exchange back into the prompt. That is all a conversation is here: the same
frame, with the finished turns in front of the new question.

```
User: capital of France?

Assistant: Paris.<｜end▁of▁sentence｜>User: and Japan?

Assistant:
```

Each completed reply is closed with the checkpoint's end-of-sequence text,
because that is what the Jinja template appends after an assistant message and
nowhere else. It is the difference between the model reading a settled turn and
reading one it is expected to keep writing. Only the last `Assistant:` is left
open.

Two ways in:

| | |
|---|---|
| `--history PATH` | A JSON array of `{"role", "content"}` objects, alternating user and assistant. The roles must alternate and the array must end on a finished reply — the question being asked now goes to `--prompt`. |
| `--remember` | In interactive mode, each finished exchange is carried into the next prompt. `/reset` forgets them without ending the session. |

A malformed transcript is refused rather than repaired, and it is read before
the checkpoint is mapped, so the error arrives immediately instead of after the
load.

### When it no longer fits

The conversation grows and the context does not. Before each prompt the whole
frame is encoded and measured against `--context`; while the prompt plus
`--max-tokens` would overflow it, the **oldest** exchange is dropped and the
frame rebuilt. A conversation that quietly forgot its beginning reads like a
model that stopped paying attention, so the count is reported rather than
swallowed — as a line on the console, and as `history_dropped` in the `plan`
event that the web interface turns into a warning chip.

The default 1,024-token context holds only a few turns of real text. Raising
`--context` costs about 450 KB of KV cache per position.
