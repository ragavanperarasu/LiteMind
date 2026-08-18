#!/usr/bin/env python3
"""
Build a tiny, randomly weighted DeepSeek-V2 checkpoint.

The real DeepSeek-V2-Lite download is about 31 GB, which is a slow way to find
out that a build is broken. This writes a structurally identical model a few
megabytes in size: the same tensor names, the same relationships between the
shapes, one dense layer followed by mixture-of-experts layers, and the same
BF16 element type. LiteMind loads and runs it through exactly the code path it
uses for the real checkpoint.

The weights are random, so the generated text is meaningless. What it proves is
that every tensor was found, every shape agreed with config.json, and the
forward pass ran end to end.

    python3 tools/make_test_model.py models-test
    ./build/bin/LiteMind models-test -p "hello" -n 8

Standard library only: no numpy, no torch.
"""

import argparse
import json
import os
import struct
import sys

# A deliberately small architecture that still exercises every code path.
CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "vocab_size": 300,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "kv_lora_rank": 16,
    "q_lora_rank": None,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 4,
    "v_head_dim": 8,
    "intermediate_size": 64,
    "moe_intermediate_size": 16,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "scoring_func": "softmax",
    "topk_method": "greedy",
    "n_group": 1,
    "topk_group": 1,
    "rope_theta": 10000,
    "rope_scaling": {
        "type": "yarn",
        "factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
        "original_max_position_embeddings": 4096,
    },
    "bos_token_id": 290,
    "eos_token_id": 291,
    "torch_dtype": "bfloat16",
}


class Random:
    """A small reproducible generator, so two runs write identical bytes."""

    def __init__(self, seed=20240816):
        self.state = seed & 0xFFFFFFFF

    def next_float(self):
        # Numerical Recipes' linear congruential parameters.
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.state / 0xFFFFFFFF) * 2.0 - 1.0


def to_bfloat16(value):
    """Truncates a float32 to its high 16 bits, which is exactly BF16."""
    packed = struct.pack("<f", value)
    return packed[2:4]


def make_tensor(random, count, scale=0.08):
    """Builds `count` BF16 values small enough to keep activations in range."""
    return b"".join(to_bfloat16(random.next_float() * scale) for _ in range(count))


def make_ones(count):
    """Norm weights start at one, as they do in a trained checkpoint."""
    return to_bfloat16(1.0) * count


def build_tensors():
    random = Random()
    config = CONFIG
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    qk_head = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    kv_rank = config["kv_lora_rank"]
    v_head = config["v_head_dim"]
    moe_inter = config["moe_intermediate_size"]

    tensors = {}

    def add(name, shape, data):
        expected = 2
        for dimension in shape:
            expected *= dimension
        assert len(data) == expected, f"{name}: {len(data)} bytes, expected {expected}"
        tensors[name] = (shape, data)

    add("model.embed_tokens.weight", [config["vocab_size"], hidden],
        make_tensor(random, config["vocab_size"] * hidden))

    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        add(prefix + "input_layernorm.weight", [hidden], make_ones(hidden))
        add(prefix + "post_attention_layernorm.weight", [hidden], make_ones(hidden))

        attn = prefix + "self_attn."
        add(attn + "q_proj.weight", [heads * qk_head, hidden],
            make_tensor(random, heads * qk_head * hidden))
        add(attn + "kv_a_proj_with_mqa.weight",
            [kv_rank + config["qk_rope_head_dim"], hidden],
            make_tensor(random, (kv_rank + config["qk_rope_head_dim"]) * hidden))
        add(attn + "kv_a_layernorm.weight", [kv_rank], make_ones(kv_rank))
        add(attn + "kv_b_proj.weight",
            [heads * (config["qk_nope_head_dim"] + v_head), kv_rank],
            make_tensor(random, heads * (config["qk_nope_head_dim"] + v_head) * kv_rank))
        add(attn + "o_proj.weight", [hidden, heads * v_head],
            make_tensor(random, hidden * heads * v_head))

        is_moe = layer >= config["first_k_dense_replace"]
        if not is_moe:
            inter = config["intermediate_size"]
            add(prefix + "mlp.gate_proj.weight", [inter, hidden], make_tensor(random, inter * hidden))
            add(prefix + "mlp.up_proj.weight", [inter, hidden], make_tensor(random, inter * hidden))
            add(prefix + "mlp.down_proj.weight", [hidden, inter], make_tensor(random, hidden * inter))
            continue

        add(prefix + "mlp.gate.weight", [config["n_routed_experts"], hidden],
            make_tensor(random, config["n_routed_experts"] * hidden))
        for expert in range(config["n_routed_experts"]):
            expert_prefix = prefix + f"mlp.experts.{expert}."
            add(expert_prefix + "gate_proj.weight", [moe_inter, hidden],
                make_tensor(random, moe_inter * hidden))
            add(expert_prefix + "up_proj.weight", [moe_inter, hidden],
                make_tensor(random, moe_inter * hidden))
            add(expert_prefix + "down_proj.weight", [hidden, moe_inter],
                make_tensor(random, hidden * moe_inter))

        # The shared block's width is n_shared_experts times the expert width.
        shared_inter = moe_inter * config["n_shared_experts"]
        shared_prefix = prefix + "mlp.shared_experts."
        add(shared_prefix + "gate_proj.weight", [shared_inter, hidden],
            make_tensor(random, shared_inter * hidden))
        add(shared_prefix + "up_proj.weight", [shared_inter, hidden],
            make_tensor(random, shared_inter * hidden))
        add(shared_prefix + "down_proj.weight", [hidden, shared_inter],
            make_tensor(random, hidden * shared_inter))

    add("model.norm.weight", [hidden], make_ones(hidden))
    add("lm_head.weight", [config["vocab_size"], hidden],
        make_tensor(random, config["vocab_size"] * hidden))
    return tensors


def write_safetensors(path, tensors):
    """Writes the 8-byte header length, the JSON header, then the payload."""
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        # Every tensor starts on an 8-byte boundary, as the writers do.
        while len(payload) % 8 != 0:
            payload.append(0)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, start + len(data)]}

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    while len(header_bytes) % 8 != 0:
        header_bytes += b" "

    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(bytes(payload))
    return 8 + len(header_bytes) + len(payload)


def byte_alphabet():
    """GPT-2's mapping from each byte value to a printable code point."""
    printable = [b for b in range(256) if (33 <= b <= 126) or (161 <= b <= 172) or b >= 174]
    table = {}
    generated = 256
    for byte in range(256):
        if byte in printable:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(generated)
            generated += 1
    return table


def write_tokenizer(directory):
    """Writes a byte-level BPE tokenizer with a handful of real merge rules."""
    alphabet = byte_alphabet()
    vocab = {alphabet[byte]: byte for byte in range(256)}

    # A few merges over common English fragments, so BPE has work to do.
    space = alphabet[ord(" ")]
    merges = []
    next_id = 256
    for left, right in [
        ("t", "h"), ("th", "e"), (space, "t"), (f"{space}t", "he"),
        ("i", "n"), ("i", "s"), ("o", "f"), (space, "a"),
        ("e", "r"), ("o", "n"), ("a", "n"), ("an", "d"),
        (space, "i"), (f"{space}i", "s"), (space, "o"), (f"{space}o", "f"),
        ("h", "e"), ("l", "l"), ("l", "o"), ("he", "llo"),
    ]:
        merges.append(f"{left} {right}")
        merged = left + right
        if merged not in vocab:
            vocab[merged] = next_id
            next_id += 1

    tokenizer = {
        "version": "1.0",
        "added_tokens": [
            {"id": 290, "content": "<|begin_of_sentence|>", "special": True,
             "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
            {"id": 291, "content": "<|end_of_sentence|>", "special": True,
             "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        ],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }
    with open(os.path.join(directory, "tokenizer.json"), "w", encoding="utf-8") as handle:
        json.dump(tokenizer, handle, ensure_ascii=False)

    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<|begin_of_sentence|>",
        "eos_token": "<|end_of_sentence|>",
        "model_max_length": 4096,
        "tokenizer_class": "LlamaTokenizerFast",
    }
    with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
        json.dump(tokenizer_config, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", default="models-test",
                        help="where to write the checkpoint (default: models-test)")
    arguments = parser.parse_args()

    directory = arguments.directory
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(CONFIG, handle, indent=2)

    write_tokenizer(directory)
    tensors = build_tensors()
    size = write_safetensors(os.path.join(directory, "model.safetensors"), tensors)

    print(f"Wrote {len(tensors)} tensors ({size / 1024.0:.1f} KiB) to {directory}/")
    print(f"  {CONFIG['num_hidden_layers']} layers, "
          f"{CONFIG['n_routed_experts']} routed experts, "
          f"top-{CONFIG['num_experts_per_tok']}")
    print()
    print("Run it with:")
    print(f"  LiteMind {directory} --inspect")
    print(f"  LiteMind {directory} -p \"hello\" -n 8")
    print()
    print("The weights are random, so the text is meaningless. A run that")
    print("completes proves the loader, the shapes and the forward pass work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
