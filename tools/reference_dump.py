#!/usr/bin/env python3
"""Dump Python-reference tensors for verifying the C++ port.

Subcommands:
  stages  -- plain (unwindowed) stream over a short clip, every intermediate
  final   -- stream over a clip, the released samples only (windowed or not)
  full    -- stream over the whole file given on the command line

Everything is written as raw little-endian float32 under --out.
"""
import argparse
import json
import os

import numpy as np
import torch

from inworld.voice_isolation import isolator
from inworld.voice_isolation import model as modeling
from inworld.voice_isolation import streaming as streaming_lib

AUDIO = ("audio_7082000Hz_18-59-39_25-06-2024-L20-endfed-maticna.wav")


def load_mono(path, seconds=None):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 48000, sr
    y = x.mean(1).astype(np.float32)
    if seconds:
        y = y[:int(seconds * sr)]
    return y


def save(path, array):
    np.ascontiguousarray(array, dtype="<f4").tofile(path)


class Recorder:
    """Records the intermediates of a Stream run, in frame order."""

    def __init__(self, model):
        self.model = model
        self.reset()
        self.ids = [id(b) for b in model.blocks]
        self.time_mamba_out = []
        self.time_proj_in = []
        self.time_proj_out = []
        self.freq_mamba_out = []
        self.orig_time = streaming_lib._time_mamba
        self.orig_attend = streaming_lib._attend
        self.orig_scan = streaming_lib._scan
        self.orig_run = streaming_lib.Stream._run_block
        self.orig_analyze = streaming_lib.Stream._analyze

        def attend(block, features, states, cache):
            out = self.orig_attend(block, features, states, cache)
            self.blocks.append((self.ids.index(id(block)), features, out))
            return out

        def scan(block, features, states):
            out = self.orig_scan(block, features, states)
            self.blocks.append((self.ids.index(id(block)), features, out))
            return out

        def time_mamba(mamba, inputs, states):
            out = self.orig_time(mamba, inputs, states)
            bands = len(self.model.edges)
            self.time_mamba_out.append(
                (mamba, out.detach().reshape(bands, -1, out.shape[-1])))
            return out

        streaming_lib._time_mamba = time_mamba
        for index, block in enumerate(model.blocks):
            if not hasattr(block, "freq_mamba"):
                continue
            block.time_mamba.output_proj.register_forward_pre_hook(
                lambda m, args, i=index: self.time_proj_in.append(
                    (i, args[0].detach().reshape(len(self.model.edges), -1,
                                                 args[0].shape[-1]))))
            block.time_mamba.output_proj.register_forward_hook(
                lambda m, args, out, i=index: self.time_proj_out.append(
                    (i, out.detach().reshape(len(self.model.edges), -1,
                                             out.shape[-1]))))
        streaming_lib._attend, streaming_lib._scan = attend, scan
        orig_run = self.orig_run

        def run_block(stream, spectra):
            self.spectra.append(spectra.detach().clone())
            return orig_run(stream, spectra)

        streaming_lib.Stream._run_block = run_block
        self.packed_encode = None
        self.packed_decode = None

    def reset(self):
        self.spectra = []
        self.encoded = []
        self.blocks = []
        self.raw = []

    def install_freq_hooks(self):
        for index, block in enumerate(self.model.blocks):
            if not hasattr(block, "freq_mamba"):
                continue
            hook = lambda m, args, out, i=index: self.freq_mamba_out.append(
                (i, out.detach().clone()))
            block.freq_mamba.register_forward_hook(hook)

    def install_packed(self, stream):
        bands = stream._bands
        self.orig_encode, self.orig_decode = bands.encode, bands.decode
        model = self.model

        def encode(m, spectrum):
            out = self.orig_encode(m, spectrum)
            self.encoded.append(out.detach().clone())
            return out

        def decode(features):
            out = self.orig_decode(features)
            self.raw.append(out.detach().clone())
            return out

        bands.encode, bands.decode = encode, decode

    def _run_block(self, stream, spectra):
        self.spectra.append(spectra.detach().clone())
        return self.orig_run(stream, spectra)

    def dump(self, root, diagnostics=False):
        os.makedirs(root, exist_ok=True)

        def cat(chunks, name, frames_dim=0):
            if not chunks:
                return
            parts = []
            for c in chunks:
                c = c.float()
                order = [frames_dim] + [d for d in range(c.ndim)
                                        if d != frames_dim]
                parts.append(c.permute(*order))
            t = torch.cat(parts, dim=0)
            save(os.path.join(root, name), t.float().cpu().numpy())

        cat(self.spectra, "spectra.bin", 0)   # [frames, batch, bins]
        cat(self.encoded, "encoded.bin", 2)   # [frames, batch, bands, width]
        # Mamba block halves, keyed by block index (diagnostics only).
        if diagnostics:
            time_chunks = {}
            for module, out in self.time_mamba_out:
                time_chunks.setdefault(id(module), []).append(out)
            for index, block in enumerate(self.model.blocks):
                if not hasattr(block, "freq_mamba"):
                    continue
                key = id(block.time_mamba)
                if key in time_chunks:
                    cat(time_chunks[key], f"time_mamba{index}.bin", 1)
            for name, store in (("time_proj_in", self.time_proj_in),
                                ("time_proj_out", self.time_proj_out)):
                chunks = {}
                for index, out in store:
                    chunks.setdefault(index, []).append(out)
                for index, outs in sorted(chunks.items()):
                    cat(outs, f"{name}{index}.bin", 1)
            freq_chunks = {}
            for index, out in self.freq_mamba_out:
                freq_chunks.setdefault(index, []).append(out)
            for index, outs in sorted(freq_chunks.items()):
                cat(outs, f"freq_mamba{index}.bin", 0)

        for index in range(len(self.ids)):
            mine = [(f, o) for (i, f, o) in self.blocks if i == index]
            if mine:
                cat([m[0] for m in mine], f"block{index}_in.bin", 2)
                cat([m[1] for m in mine], f"block{index}_out.bin", 2)
        print(f"dumped to {root}")


def run(model, audio, window, crossfade, lookahead, block, piece, recorder=None):
    isolator_ = model
    stream = streaming_lib.Stream(
        model,
        block_frames=block,
        window_frames=None if not window else window,
        crossfade_frames=None if not window else crossfade)
    if recorder is not None:
        recorder.install_packed(stream)
        recorder.install_freq_hooks()
    out = []
    for start in range(0, len(audio), piece):
        out.append(np.asarray(stream.push(audio[start:start + piece])))
    out.append(np.asarray(stream.flush()))
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["stages", "final", "full"])
    p.add_argument("--out", default=None)
    p.add_argument("--audio", default=AUDIO)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--crossfade", type=int, default=0)
    p.add_argument("--lookahead", type=int, default=4)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--piece", type=int, default=24000)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--diagnostics", action="store_true",
                   help="also dump the mamba halves and projections")
    args = p.parse_args()

    model = isolator.load_model(
        isolator.registry.resolve("voice-isolator-v1-realtime", "streaming"),
        args.device)
    model.set_lookahead(args.lookahead)

    audio = load_mono(args.audio, None if args.cmd == "full" else args.seconds)
    if args.start:
        audio = audio[int(args.start * 48000):]

    out = args.out or {
        "stages": "ref/stages",
        "final": "ref/final",
        "full": "ref/full",
    }[args.cmd]
    os.makedirs(out, exist_ok=True)

    rec = Recorder(model) if args.cmd == "stages" else None
    y = run(model, audio, args.window, args.crossfade, args.lookahead,
            args.block, args.piece, rec)
    save(os.path.join(out, "input.bin"), audio)
    save(os.path.join(out, "output.bin"), y)
    if rec is not None:
        rec.dump(out, args.diagnostics)
    meta = dict(window=args.window, crossfade=args.crossfade,
                lookahead=args.lookahead, block=args.block, piece=args.piece,
                samples=len(audio), out_samples=len(y))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
