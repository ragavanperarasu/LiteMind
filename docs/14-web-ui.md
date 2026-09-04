# 14 · The web interface

**Source:** [`ui/server/`](../ui/server/), [`ui/web/`](../ui/web/)
**Launcher:** [`scripts/ui.ps1`](../scripts/ui.ps1)

```
browser  ──HTTP──▶  Node (node:http)  ──spawn──▶  LiteMind.exe --json
   ▲                                                      │
   └──────── server-sent events ◀── newline JSON ──────────┘
```

## Running it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ui.ps1
```

That builds the engine, installs the interface packages on first run, builds the
bundle and serves everything on <http://localhost:5174>. `-Dev` runs the Vite dev
server with hot reload instead; `-SkipBuild` leaves the C++ alone.

## Why the engine emits JSON

The console report exists to be read by a person, and it changes whenever the
wording improves. A program that scraped it would break on a reworded heading, a
renamed column or a progress bar drawn with carriage returns.

So `--json` gives the same information as a stream of objects, one per line:

| Event | When | Carries |
|---|---|---|
| `ready` | after the checkpoint loads | architecture, context, whether a chat template applies |
| `plan` | before each prompt runs | token counts, expert counts, parameter counts, how many remembered exchanges went in and how many were dropped |
| `token` | per decoded fragment | the text |
| `done` | when the reply ends | timings, throughput, stop reason |
| `memory` | after `ready` | mapped, hot, routed, KV-cache and expert-cache bytes |
| `routing` | per generated token, with `--routing` | the experts that token used |
| `error` | on failure | the message |

`--json` also silences the report, because both would otherwise be writing to
stdout at once.

### The output has to be valid UTF-8

JSON is defined over Unicode text, but what reaches the writer is model output,
and a byte-level tokenizer can emit any byte at all. A truncated sequence or a
stray continuation byte would produce a document no parser will accept, killing a
reply mid-stream for a reason nowhere near where it appears.

`JsonWriter::quote` therefore validates as it escapes, rejecting over-long forms,
surrogate halves and anything past U+10FFFF, and substituting U+FFFD. Malformed
bytes become visible replacement characters rather than a broken stream.

## Why a process per request

A 29.3 GiB checkpoint sounds far too expensive to load per request. It is not:
the weights are memory mapped, so a load costs about a third of a second, and the
expert pages stay in the operating system's cache between processes — the same
mechanism that makes streaming work at all.

What it buys is that a cancelled request leaves nothing behind. Killing the
process resets the KV cache, the sampler and the expert residency together, with
no state to unwind by hand.

It also means the conversation cannot live in the engine, because the process
that answers one question is not the one that answered the last. It lives in the
page instead: the browser keeps the transcript and sends the finished exchanges
back with each new prompt, and the server writes them to a temporary file for
`--history`. A transcript is unbounded and a command line is not — Windows caps
one at 32,767 characters, which a dozen turns can reach — and a file also keeps
the text out of the process table. The file is deleted when the process exits.

Only settled turns are sent. A reply that was stopped or that failed has no
answer to remember, and feeding half a reply back would make it context for
everything after it: the model would carry on from the break rather than treat
it as something already said. Both halves of such a turn are dropped together,
because a question with no answer is not a turn.

Requests are serialised. The engine already spreads one token across every core,
so two prompts at once would halve the threads available to each and make both
slower than running them in turn. A second request gets `409` rather than
competing for the cores.

## Why the backend has no packages

The engine's claim is that it has no third-party dependencies. The two lines of
routing here do not justify spending that claim on Express, so the server is
`node:http` and nothing else.

The browser bundle is a different matter. Material UI *is* a dependency, and
React and Vite come with it — but that is a property of the interface, not of the
inference engine, which is unchanged and still builds and runs with no packages
at all. `scripts\run.ps1` remains the dependency-free path.

## Streaming, twice

The reply crosses two boundaries, and both can split a multi-byte character:

1. **Engine → Node**, over a pipe that breaks wherever the buffer ends.
2. **Node → browser**, over a network that breaks wherever a packet ends.

Both sides keep a `TextDecoder` across chunks with `{ stream: true }`, so a
character split across a boundary is held until it is complete. Decoding per
chunk instead would corrupt any reply that is not pure ASCII — which is to say
any reply in Chinese, or containing an emoji or a dash.

Server-sent events carry it over the second hop. It is the one streaming
transport that needs no library on either end. `EventSource` cannot be used
directly because it only issues GET requests and a prompt needs a POST body, so
the response body is read and split by hand — the framing is identical.

## What the interface shows

Alongside the answer, the plan panel restates the cost of the prompt, because
those numbers are easy to misread. `26 layers × 6 experts = 156 per token` and
`79,248 expert executions` invite the conclusion that 79,248 experts get loaded.
They do not: there are only **1,664** routed experts in the whole model, and the
executions reuse them roughly forty-eight times each.

The panel names the pool size and the reuse factor next to the execution count
for exactly that reason, and swaps the planned ceilings for what the reply
actually cost once it finishes.

## The routing map

`--routing` reports the expert indices each token was routed to, layer by layer:
`moe_layers × top-k` numbers per token, 156 of them for DeepSeek-V2-Lite. It is
off unless asked for, and the recording itself is skipped entirely when nobody
is observing — an unset callback turns the `push_back` off rather than
collecting data to discard.

The interface draws that as one dot per routed expert: a column is a layer, a
row is an expert within it, 26 × 64 = **1,664 dots**. Six light up per column
per token.

That picture is the argument of the whole project in one image. Almost the
entire grid stays dark while the model answers, and the dark part is weight that
never left the disk. Brightness accumulates over a reply, so it also shows which
experts a particular prompt kept returning to — and the coverage figure says how
much of the pool a reply reached at all.

## What the usage panel can and cannot say

Host CPU and memory come from `node:os`. Per-core utilisation is a **difference
between two readings** — `os.cpus()` reports time accumulated since boot, not a
rate — so the first sample after startup reads zero and the second is the first
real one.

The panel deliberately does not claim a memory figure for the engine process.
The weights are a mapping of a file: most of what the engine "uses" is page
cache the operating system owns and will reclaim the moment something else needs
it. A resident-set number would look precise and mean very little. What is shown
instead is the engine's own accounting — what it mapped, what stays hot, what is
streamed — which is the part that is actually true.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Is the engine built, is a checkpoint present, is one running |
| `GET` | `/api/settings` | `litemind.json`, without its comment keys |
| `PUT` | `/api/settings` | Merge changes in, preserving the comments |
| `POST` | `/api/generate` | Run a prompt, streaming events back. Takes `prompt`, an optional `history` of `{role, content}` messages, and optional `settings` |
| `GET` | `/api/usage` | Host CPU and memory, sampled between calls |
| `POST` | `/api/cancel` | Kill the running generation |

Settings are read from and written to the same `litemind.json` the command line
uses, so a change made in the browser applies to `scripts\run.ps1` and back.

Next: [Comparison and requirements](15-comparison.md)
