#!/usr/bin/env python3
"""Export the Inworld realtime voice-isolator checkpoint to a portable binary.

Writes, into <out_dir>:
  weights.bin    all tensors as little-endian float32, 64-byte aligned
  weights.txt    one line per tensor:  <name> <rank> <dims...> <offset_floats>
  config.txt     resolved architecture values the C++ runtime needs

Usage:
  python tools/export_weights.py [--out model] [--lookahead 4]
"""
import argparse
import json
import os
import struct

import numpy as np
import safetensors

REPO = ("/Users/san/.cache/huggingface/hub/models--inworld--voice-isolator/"
        "snapshots/9b0285b3fe8403aa8e2f1dbb2165a80a4294c970/"
        "denoiser_realtime.safetensors")
ALIGN = 64  # floats


def band_edges(cfg):
    """model.band_edges, verbatim."""
    n_bins = cfg["n_fft"] // 2 + 1
    hz = cfg["sample_rate"] / cfg["n_fft"]
    edges, start = [], 0
    for width_hz, upto_hz in cfg["band_plan"]:
        width = max(1, round(width_hz / hz))
        upto = min(n_bins, round(upto_hz / hz))
        while start < upto:
            end = min(start + width, upto)
            edges.append((start, end))
            start = end
    if start < n_bins:
        edges.append((start, n_bins))
    return edges


def torch_round(x):
    """C round-half-away-from-zero vs python round-half-even: torch uses the
    former in band_edges (python's round() is what the package uses, so match
    python)."""
    return round(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="model")
    p.add_argument("--lookahead", type=int, default=4)
    p.add_argument("--ckpt", default=REPO)
    args = p.parse_args()

    with safetensors.safe_open(args.ckpt, framework="np") as f:
        cfg = json.loads(f.metadata()["config"])
        tensors = {k: np.asarray(f.get_tensor(k), dtype=np.float32)
                   for k in f.keys()}

    edges = band_edges(cfg)
    choices = cfg["lookahead_frames_choices"]
    assert args.lookahead in choices, (args.lookahead, choices)
    index = choices.index(args.lookahead)

    os.makedirs(args.out, exist_ok=True)
    blob_path = os.path.join(args.out, "weights.bin")
    lines, offset = [], 0
    with open(blob_path, "wb") as out:
        for name in sorted(tensors):
            t = np.ascontiguousarray(tensors[name], dtype="<f4")
            pad = (-offset) % ALIGN
            if pad:
                out.write(b"\0" * (4 * pad))
            offset += pad
            out.write(t.tobytes())
            lines.append(f"{name} {t.ndim} " +
                         " ".join(str(d) for d in t.shape) + f" {offset}")
            offset += t.size
    with open(os.path.join(args.out, "weights.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    total = sum(t.size for t in tensors.values())
    with open(os.path.join(args.out, "config.txt"), "w") as f:
        f.write(f"n_fft {cfg['n_fft']}\n")
        f.write(f"hop_length {cfg['hop_length']}\n")
        f.write(f"sample_rate {cfg['sample_rate']}\n")
        f.write(f"width {cfg['width']}\n")
        f.write(f"num_heads {cfg['num_heads']}\n")
        f.write(f"mamba_state {cfg['mamba_state']}\n")
        f.write(f"mamba_conv {cfg['mamba_conv']}\n")
        f.write(f"mamba_expand {cfg['mamba_expand']}\n")
        f.write(f"correction_bound {cfg['correction_bound']}\n")
        f.write(f"chunk_seconds {cfg['chunk_seconds']}\n")
        f.write(f"attention_window_frames {cfg['attention_window_frames']}\n")
        f.write(f"lookahead_frames {args.lookahead}\n")
        f.write(f"lookahead_index {index}\n")
        f.write(f"num_blocks {len(cfg['block_types'])}\n")
        f.write(f"block_types {' '.join(cfg['block_types'])}\n")
        f.write(f"num_bands {len(edges)}\n")
        f.write("band_edges " + " ".join(f"{a} {b}" for a, b in edges) + "\n")

    print(f"{len(tensors)} tensors, {total} params -> {blob_path}")
    print(f"{len(edges)} bands, lookahead {args.lookahead} (index {index})")


if __name__ == "__main__":
    main()
