#!/usr/bin/env python3
"""
Print the logits Hugging Face predicts, to compare against LiteMind's.

If LiteMind's output looks like noise, the question is whether the fault is in
loading, in the arithmetic, or only in sampling. Comparing the raw logits for
the token right after a prompt answers that directly, because those come out of
the model before any sampling decision is made.

Run both on the same prompt:

    python3 tools/reference_logits.py models --prompt "The capital of France is"
    LiteMind models -p "The capital of France is" -n 1 --top-logits 10 --show-tokens

Then compare, in this order:

  1. The token IDs. If they differ, the tokenizer is the problem and nothing
     downstream can match. Fix that first.
  2. The identity and order of the top tokens. If these agree, the model is
     loading and computing correctly; anything left is a sampling setting.
  3. The logit values. Small differences are expected: this runs in float32
     from BF16 weights, and the reference may accumulate differently. Ordering
     matters more than the third decimal place.

This needs torch and transformers, which are only for checking. LiteMind itself
depends on neither. Run it on any machine with enough RAM, not necessarily the
one running LiteMind.

    pip install torch transformers accelerate
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="the model directory, the same one LiteMind is given")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="the prompt to score")
    parser.add_argument("--top", type=int, default=10, help="how many logits to print")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                        help="float32 matches LiteMind's accumulation most closely")
    arguments = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as failure:
        print(f"This script needs torch and transformers: {failure}", file=sys.stderr)
        print("  pip install torch transformers accelerate", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(arguments.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model,
        trust_remote_code=True,
        torch_dtype=getattr(torch, arguments.dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()

    encoded = tokenizer(arguments.prompt, return_tensors="pt")
    token_ids = encoded["input_ids"][0].tolist()

    print(f"prompt: {arguments.prompt!r}")
    print(f"{len(token_ids)} prompt tokens:")
    for token in token_ids:
        print(f"  {token:6d}  {tokenizer.convert_ids_to_tokens([token])[0]}")

    with torch.no_grad():
        output = model(**encoded)

    # The last position's logits predict the token that follows the prompt,
    # which is exactly what LiteMind's --top-logits reports.
    logits = output.logits[0, -1].float()
    values, indices = torch.topk(logits, arguments.top)

    print(f"\nTop {arguments.top} logits for the token after the prompt:")
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"  {index:6d}  {value:10.4f}  {tokenizer.convert_ids_to_tokens([index])[0]}")

    print("\nCompare against:")
    print(f"  LiteMind {arguments.model} -p {arguments.prompt!r} "
          f"-n 1 --top-logits {arguments.top} --show-tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
