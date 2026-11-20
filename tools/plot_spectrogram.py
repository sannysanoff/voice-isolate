#!/usr/bin/env python3
"""Left-to-right spectrogram of one channel of a WAV file (0..3 kHz, auto brightness)."""

import argparse

import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-c", "--channel", type=int, default=0)
    ap.add_argument("--fmax", type=float, default=3000.0)
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--cmap", default="magma")
    args = ap.parse_args()

    x, fs = sf.read(args.wav, dtype="float32", always_2d=True)
    sig = x[:, args.channel]

    # STFT (Hann window, hop = nfft/8)
    win = np.hanning(args.nfft).astype(np.float32)
    n_frames = 1 + (len(sig) - args.nfft) // args.hop
    idx = np.arange(args.nfft)[None, :] + args.hop * np.arange(n_frames)[:, None]
    frames = sig[idx] * win
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real**2 + spec.imag**2).T  # (freq, time)

    freqs = np.fft.rfftfreq(args.nfft, 1.0 / fs)
    keep = freqs <= args.fmax
    power = power[keep]
    freqs = freqs[keep]

    db = 10.0 * np.log10(power + 1e-12)
    # auto brightness: robust percentile stretch of the content itself
    lo, hi = np.percentile(db, [5.0, 99.9])
    if hi - lo < 1.0:
        hi = lo + 1.0
    db = np.clip((db - lo) / (hi - lo), 0.0, 1.0)

    t0 = args.nfft / 2.0 / fs
    t1 = t0 + (n_frames - 1) * args.hop / fs

    out = args.out or f"{args.wav.rsplit('.', 1)[0]}_ch{args.channel}_spec.pdf"
    fig, ax = plt.subplots(figsize=(22, 5.2), dpi=150)
    ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        cmap=args.cmap,
        extent=[t0, t1, freqs[0], freqs[-1]],
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("time [s]  →")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(
        f"{args.wav}  channel {args.channel}/{x.shape[1]}  fs={fs} Hz  "
        f"nfft={args.nfft} hop={args.hop}  dB range {lo:.1f}..{hi:.1f}"
    )
    ax.set_ylim(0, args.fmax)
    ax.set_xlim(t0, t1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
