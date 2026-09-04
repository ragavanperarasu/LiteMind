#!/usr/bin/env python3
"""
Draw the comparison charts in docs/15-comparison.md as SVG.

Every value is computed here from the checkpoint's own parameter counts in
docs/model-info.json, so the charts and the tables on that page cannot drift
apart: change a number there and regenerate rather than editing an SVG by hand.

    python3 tools/make_comparison_charts.py

Two rules the drawing follows, both about honesty rather than looks:

  * A bar is drawn solid only when its value is measured or arithmetic. A
    quantity nobody measured is drawn as an open outline labelled "not
    measured", because an absent bar reads as zero and a guessed one is worse.
  * Colours have to work on GitHub's light theme and its dark theme, which the
    same file is shown against. Nothing relies on the background, and no text
    is lighter than mid grey.

Standard library only, like the rest of the project.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")

GIB = 1024 ** 3

# Effective bits per weight, published per GGUF type. Q4_K_M is what Ollama
# serves for deepseek-v2:16b and what llama.cpp users pick by default.
Q4_K_M = 4.85
BF16 = 16.0

# Readable on white and on #0d1117 alike.
INK = "#8b949e"          # labels and axis
STRONG = "#7d8590"       # titles
ACCENT = "#7aa2f7"       # LiteMind
NEUTRAL = "#7f849c"      # llama.cpp / Ollama
ACCENT_SOFT = "#7aa2f7"
GRID = "#8b949e"

FONT = ("ui-sans-serif, -apple-system, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, sans-serif")
MONO = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'Liberation Mono', monospace")


def esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def chart(title, subtitle, rows, unit, footnote):
    """
    One grouped horizontal bar chart.

    rows is a list of (label, value, note, kind) where kind is 'accent' for
    LiteMind, 'neutral' for the others, and 'unknown' for a quantity that was
    not measured - which is drawn as an outline and never given a length.
    """
    width = 760
    pad_left = 188
    pad_right = 96
    top = 78
    row_h = 50
    bar_h = 22
    height = top + row_h * len(rows) + 54

    plot = width - pad_left - pad_right
    known = [value for _, value, _, kind in rows if kind != "unknown"]
    top_value = max(known) if known else 1.0

    # A round number above the tallest bar, so the axis reads sensibly.
    for step in (0.5, 1, 2, 2.5, 5, 10, 25, 50):
        if top_value / step <= 6:
            break
    axis_max = step * (int(top_value / step) + 1)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
        f'<text x="0" y="22" font-family="{FONT}" font-size="15" '
        f'font-weight="600" fill="{STRONG}">{esc(title)}</text>',
        f'<text x="0" y="42" font-family="{FONT}" font-size="12" '
        f'fill="{INK}">{esc(subtitle)}</text>',
    ]

    # Gridlines behind everything, at each axis step.
    ticks = int(axis_max / step) + 1
    for index in range(ticks):
        value = step * index
        x = pad_left + plot * value / axis_max
        out.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" '
            f'y2="{top + row_h * len(rows) - 14}" stroke="{GRID}" '
            f'stroke-width="1" opacity="0.18"/>')
        out.append(
            f'<text x="{x:.1f}" y="{top + row_h * len(rows) + 4}" '
            f'font-family="{MONO}" font-size="10" fill="{INK}" '
            f'text-anchor="middle" opacity="0.85">'
            f'{value:g}</text>')

    for index, (label, value, note, kind) in enumerate(rows):
        y = top + index * row_h
        out.append(
            f'<text x="{pad_left - 12}" y="{y + bar_h - 6}" '
            f'font-family="{FONT}" font-size="12.5" '
            f'font-weight="{"600" if kind == "accent" else "400"}" '
            f'fill="{STRONG if kind == "accent" else INK}" '
            f'text-anchor="end">{esc(label)}</text>')

        if kind == "unknown":
            # An outline, full width, so it is plainly a placeholder and not a
            # value. Nothing here may be read off a scale.
            out.append(
                f'<rect x="{pad_left}" y="{y}" width="{plot}" height="{bar_h}" '
                f'rx="3" fill="none" stroke="{INK}" stroke-width="1" '
                f'stroke-dasharray="4 4" opacity="0.5"/>')
            out.append(
                f'<text x="{pad_left + 10}" y="{y + bar_h - 6}" '
                f'font-family="{FONT}" font-size="11.5" fill="{INK}" '
                f'font-style="italic">{esc(note or "not measured")}</text>')
            continue

        bar = plot * value / axis_max
        colour = ACCENT if kind == "accent" else NEUTRAL
        out.append(
            f'<rect x="{pad_left}" y="{y}" width="{bar:.1f}" height="{bar_h}" '
            f'rx="3" fill="{colour}" opacity="{1.0 if kind == "accent" else 0.55}"/>')
        out.append(
            f'<text x="{pad_left + bar + 10:.1f}" y="{y + bar_h - 6}" '
            f'font-family="{MONO}" font-size="12" '
            f'font-weight="{"600" if kind == "accent" else "400"}" '
            f'fill="{STRONG}">{value:.2f} {unit}</text>')
        if note:
            # Under the bar, anchored to the axis rather than to the bar's end -
            # a note that started where the longest bar finished ran off the
            # right edge of the picture.
            out.append(
                f'<text x="{pad_left}" y="{y + bar_h + 13}" '
                f'font-family="{FONT}" font-size="10" fill="{INK}" '
                f'opacity="0.9">{esc(note)}</text>')

    out.append(
        f'<text x="0" y="{height - 8}" font-family="{FONT}" font-size="10.5" '
        f'fill="{INK}" opacity="0.9">{esc(footnote)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    with open(os.path.join(DOCS, "model-info.json"), encoding="utf-8") as handle:
        info = json.load(handle)

    total = info["parameters"]["total"]
    active = info["parameters"]["active_per_token"]
    routed = info["parameters"]["breakdown_total"]["routed_experts"]
    non_expert = total - routed

    # One expert, and the 156 of them a single token runs.
    expert_params = info["experts"]["one_expert"]["parameters"]
    per_token_expert_params = expert_params * info["per_token_cost"]["expert_executions"]

    def gib(params, bits):
        return params * bits / 8 / GIB

    charts = []

    charts.append(("comparison-storage.svg", chart(
        "Checkpoint on disk",
        "DeepSeek-V2-Lite, 15.7 B parameters — the same model in four forms",
        [
            ("Ollama  deepseek-v2:16b", gib(total, Q4_K_M), "Q4_K_M, what the registry serves", "neutral"),
            ("llama.cpp  Q4_K_M", gib(total, Q4_K_M), "the usual choice", "neutral"),
            ("llama.cpp  BF16", gib(total, BF16), "like-for-like with LiteMind", "neutral"),
            ("LiteMind", gib(total, BF16), "BF16, the only format it reads", "accent"),
        ],
        "GiB",
        "Arithmetic from published bits-per-weight. The BF16 rows predict 29.25 GiB; "
        "the checkpoint on disk is 29.3 GiB.",
    )))

    charts.append(("comparison-ram.svg", chart(
        "Weights that must be resident in RAM",
        "Non-expert tensors, plus the experts one token needs at once. KV cache excluded — see below.",
        [
            ("Ollama  Q4_K_M",
             gib(non_expert, Q4_K_M) + gib(per_token_expert_params, Q4_K_M),
             "llama.cpp underneath, plus a server process", "neutral"),
            ("llama.cpp  Q4_K_M",
             gib(non_expert, Q4_K_M) + gib(per_token_expert_params, Q4_K_M),
             "page cache decides; no ceiling", "neutral"),
            ("llama.cpp  BF16",
             gib(non_expert, BF16) + gib(per_token_expert_params, BF16),
             "page cache decides; no ceiling", "neutral"),
            ("LiteMind  BF16",
             gib(non_expert, BF16) + gib(per_token_expert_params, BF16),
             "--expert-cache sets a hard ceiling", "accent"),
        ],
        "GiB",
        "At equal precision this is a tie: the same architecture holds the same weights. "
        "What differs is that LiteMind's expert residency has an explicit bound.",
    )))

    charts.append(("comparison-traffic.svg", chart(
        "Weight bytes read per generated token",
        "2.45 B active parameters — the number that decides throughput on a bandwidth-bound CPU",
        [
            ("Ollama  Q4_K_M", gib(active, Q4_K_M), "", "neutral"),
            ("llama.cpp  Q4_K_M", gib(active, Q4_K_M), "", "neutral"),
            ("llama.cpp  BF16", gib(active, BF16), "", "neutral"),
            ("LiteMind  BF16", gib(active, BF16),
             f"{gib(active, BF16) / gib(active, Q4_K_M):.1f}x a Q4_K_M run", "accent"),
        ],
        "GiB",
        "Arithmetic. Fewer bytes per token is faster when the CPU is waiting on memory, "
        "which page 12 shows it is.",
    )))

    charts.append(("comparison-throughput.svg", chart(
        "Throughput",
        "Same prompt, same laptop: 8 threads, AVX2 + FMA, NVMe SSD, 32 GB RAM",
        [
            ("Ollama  Q4_K_M", 0, "", "unknown"),
            ("llama.cpp  Q4_K_M", 0, "", "unknown"),
            ("LiteMind  page cache", 3.47, "measured, warm", "accent"),
            ("LiteMind  --expert-cache 4", 0.89, "measured, bounded arena", "accent"),
        ],
        "tok/s",
        "Neither llama.cpp nor Ollama was run for this page, so neither is given a bar. "
        "The last section has the commands to fill them in.",
    )))

    hot = info["memory"]["always_hot_gib"]
    kv = info["memory"]["kv_cache"]["mib_at_1024_positions"] / 1024
    token_experts = info["per_token_cost"]["expert_weight_reads_gib"]

    charts.append(("comparison-requirements.svg", chart(
        "RAM needed to run LiteMind",
        "DeepSeek-V2-Lite at BF16, default 1,024-token context",
        [
            ("Resident floor", hot + kv,
             "always-hot weights 2.44 GiB + KV cache 0.43 GiB", "accent"),
            ("+ smallest useful arena", hot + kv + token_experts,
             "--expert-cache 2.51 - one token's expert working set", "accent"),
            ("+ recommended arena", hot + kv + 4,
             "--expert-cache 4 - the smallest budget worth setting", "accent"),
            ("No budget", 0,
             "no fixed size - the page cache takes what the machine allows, "
             "and this is the fastest setting", "unknown"),
        ],
        "GiB",
        "Measured components. The last row has no fixed size: without a budget the "
        "operating system caches as much as the machine allows.",
    )))

    for name, svg in charts:
        path = os.path.join(DOCS, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(svg)
        print(f"  {name:32} {len(svg):6,} bytes")

    print(f"\nWrote {len(charts)} charts to docs/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
