#!/usr/bin/env python3
"""Validate the Mamba layer math independently of C++.

A numpy mirror of the scan (the same recurrence the C++ layer runs) is compared
against the torch module on the same weights and input, stage by stage. They
agree to ~133 dB. Usage: python tools/mamba_probe.py
"""
import numpy as np
import torch
import safetensors.torch

from inworld.voice_isolation import isolator
from inworld.voice_isolation import scan

CKPT = ("/Users/san/.cache/huggingface/hub/models--inworld--voice-isolator/"
        "snapshots/9b0285b3fe8403aa8e2f1dbb2165a80a4294c970/"
        "denoiser_realtime.safetensors")
F32 = np.float32


def mirror_scan_y(x, w):
    """The scan readout only, silu applied once (clean version)."""
    tasks, length, _ = x.shape
    inner = w["in_proj.weight"].shape[0] // 2
    state_n = w["A_log"].shape[1]
    dt_rank = w["dt_proj.weight"].shape[1]
    xz = x @ w["in_proj.weight"].T
    u, z = xz[..., :inner], xz[..., inner:]
    taps = w["conv1d.weight"].shape[2] - 1
    history = np.concatenate([np.zeros((tasks, taps, inner), F32), u], axis=1)
    kernel = w["conv1d.weight"][:, 0, :]
    convolved = np.zeros((tasks, length, inner), F32)
    for j in range(taps + 1):
        convolved += history[:, j:j + length, :] * kernel[None, None, :, j]
    convolved += w["conv1d.bias"][None, None, :]
    u2 = convolved / (1.0 + np.exp(-convolved))
    xp = u2 @ w["x_proj.weight"].T
    dt, b, c = xp[..., :dt_rank], xp[..., dt_rank:dt_rank + state_n], xp[..., dt_rank + state_n:]
    pre = dt @ w["dt_proj.weight"].T + w["dt_proj.bias"]
    delta = np.where(pre > 20.0, pre, np.log1p(np.exp(np.minimum(pre, 20.0))))
    a = -np.exp(w["A_log"])
    h = np.zeros((tasks, inner, state_n), F32)
    y = np.zeros((tasks, length, inner), F32)
    for t in range(length):
        d = delta[:, t, :, None]
        h = np.exp(d * a[None]) * h + (d * u2[:, t, :, None]) * b[:, t, None, :]
        y[:, t] = (h * c[:, t, None, :]).sum(-1) + w["D"][None, :] * u2[:, t, :]
    y = y * (z / (1.0 + np.exp(-z)))
    return y, h, (u2, delta, b, c, z)


def to_np(t):
    return t.detach().float().cpu().numpy()


def main():
    model = isolator.load_model(CKPT, "cpu")
    sd = safetensors.torch.load_file(CKPT)
    for which in ("time_mamba.forward_blocks", "freq_mamba.forward_blocks",
                  "freq_mamba.backward_blocks"):
        prefix = "blocks.1." + which + "."
        w = {k[len(prefix):]: to_np(v) for k, v in sd.items()
             if k.startswith(prefix)}
        module = model.blocks[1]
        for part in which.split("."):
            module = getattr(module, part)
        if which.startswith("time"):
            x = np.random.RandomState(0).randn(46, 4, 512).astype(F32) * 0.3
        else:
            x = np.random.RandomState(1).randn(4, 46, 512).astype(F32) * 0.3

        ref_out, (ref_tail, ref_ssm) = module(torch.tensor(x))
        y_ref, ssm_ref, (u2, delta, b, c, z) = mirror_scan_y(x, w)
        out_ref = to_np(module.out_proj(torch.tensor(y_ref)))
        print(f"{which}")
        print(f"   scan readout rms {np.sqrt((y_ref**2).mean()):.4g}")
        print(f"   module out vs mirror out  max "
              f"{np.abs(to_np(ref_out) - out_ref).max():.2e}  snr "
              f"{20*np.log10(np.sqrt((to_np(ref_out)**2).mean())/np.sqrt(((to_np(ref_out)-out_ref)**2).mean())):.1f} dB")
        print(f"   ssm state rms {np.sqrt((ssm_ref**2).mean()):.4g}")
        # what torch's scan gives for the same inputs
        y_t, ssm_t = scan.selective_scan(
            torch.tensor(u2), torch.tensor(delta), torch.tensor(-np.exp(w["A_log"])),
            torch.tensor(b), torch.tensor(c), torch.tensor(w["D"]),
            torch.tensor(z), None)
        print(f"   scan(y) vs mirror(y) max {np.abs(to_np(y_t) - y_ref).max():.2e}")
        print(f"   scan ssm vs mirror ssm max {np.abs(to_np(ssm_t) - ssm_ref).max():.2e}")
        ssm_ref = ssm_ref
        # dump for the C++ side
        tag = which.replace(".", "_")
        x.tofile(f"/tmp/probe_{tag}_x.f32")
        y_ref.tofile(f"/tmp/probe_{tag}_y.f32")
        out_ref.tofile(f"/tmp/probe_{tag}_out.f32")
        u2.tofile(f"/tmp/probe_{tag}_u2.f32")
        delta.tofile(f"/tmp/probe_{tag}_delta.f32")
        b.tofile(f"/tmp/probe_{tag}_b.f32")
        c.tofile(f"/tmp/probe_{tag}_c.f32")
        z.tofile(f"/tmp/probe_{tag}_z.f32")


if __name__ == "__main__":
    main()
