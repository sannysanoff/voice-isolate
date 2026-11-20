#!/usr/bin/env python3
"""Measure the harmonic tuning error of a voice recording.

A lower-sideband demodulator tuned to the wrong carrier shifts every component
of the voice by the same amount, so the partials sit at h*f0 + offset instead of
h*f0.  This tool estimates (f0, offset) per frame by fitting a harmonic lattice
to the detected spectral peaks, and reports how well the frame sits on that
lattice.  That number is the property pitchfix-cli is supposed to improve.

    python3 tools/pitchfix_analyze.py voice.wav
    python3 tools/pitchfix_analyze.py voice.wav --compare voice-fixed.wav

The offset is only observable modulo f0: offset and offset - m*f0 describe the
same lattice.  The reported series is alias-reduced around the file median.

Needs numpy only (no scipy).
"""
import argparse
import struct
import sys

import numpy as np


def load_wav(path):
    """Read PCM16 or float32 WAV as mono float64. Returns (samples, rate)."""
    with open(path, "rb") as file:
        data = file.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SystemExit(f"{path}: not a RIFF/WAVE file")
    pos, fmt, samples, rate = 12, None, None, 48000
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = (tag, channels, bits)
        elif chunk_id == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    if fmt is None or samples is None:
        raise SystemExit(f"{path}: missing fmt or data chunk")
    tag, channels, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(samples, dtype="<f4").astype(np.float64)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(samples, dtype="<i2").astype(np.float64) / 32768.0
    else:
        raise SystemExit(f"{path}: unsupported format tag {tag}, {bits} bit")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def hann(n):
    return np.hanning(n + 1)[:-1]


def spectrum(seg, n, rate):
    """Log magnitude with the running 300 Hz trend removed, and the magnitudes."""
    mag = np.abs(np.fft.rfft(seg * hann(n)))
    log_mag = np.log(mag + 1e-6)
    width = 13
    pad = np.pad(log_mag, (width // 2, width // 2), mode="edge")
    trend = np.median(np.lib.stride_tricks.sliding_window_view(pad, width), axis=1)
    return np.clip(log_mag - trend, -3.0, 6.0), mag, rate / n


def comb_contrast(w, df, f0, delta, lo=110.0, hi=3600.0, harmonics=8):
    """Mean excess on h*f0+delta minus the midpoints: the lattice fit score."""
    on, mid = [], []
    for h in range(1, harmonics + 1):
        for f, bucket in ((h * f0 + delta, on), ((h + 0.5) * f0 + delta, mid)):
            if lo <= f <= hi:
                i = min(int(f / df), len(w) - 2)
                t = f / df - i
                bucket.append(w[i] * (1 - t) + w[i + 1] * t)
    if len(on) < 4 or len(mid) < 3:
        return -9.0
    return float(np.mean(on) - np.mean(mid))


def make_grid(max_shift, f0_range, df, bins, harmonics=8, step_f0=4.0, step_d=12.0):
    """Precompute the comb look-up tables: one build, reused for every frame."""
    f0 = np.arange(f0_range[0], f0_range[1] + 1e-9, step_f0)[:, None, None]
    delta = np.arange(-max_shift, max_shift + 1e-9, step_d)[None, :, None]
    h = np.arange(1, harmonics + 1)[None, None, :]
    grid = {}
    for name, positions in (("on", h * f0 + delta), ("mid", (h + 0.5) * f0 + delta)):
        valid = (positions >= 110.0) & (positions <= 3600.0)
        index = np.clip((positions / df).astype(int), 0, bins - 2)
        grid[name] = (index, positions / df - index, valid, np.maximum(valid.sum(-1), 1))
    grid["f0"] = f0[:, 0, 0]
    grid["delta"] = delta[0, :, 0]
    grid["ok"] = (grid["on"][3] >= 4) & (grid["mid"][3] >= 3)
    return grid


def comb_surface(w, grid):
    out = []
    for index, frac, valid, count in (grid["on"], grid["mid"]):
        value = w[index] * (1 - frac) + w[index + 1] * frac
        out.append(np.where(valid, value, 0.0).sum(-1) / count)
    return np.where(grid["ok"], out[0] - out[1], -9.0)


def coarse_grid(w, df, max_shift, f0_range):
    grid = make_grid(max_shift, f0_range, df, len(w))
    surface = comb_surface(w, grid)
    i, j = np.unravel_index(np.argmax(surface), surface.shape)
    best = (float(surface[i, j]), float(grid["f0"][i]), float(grid["delta"][j]))
    # 1 Hz / 2 Hz refinement around the coarse winner.
    fine = make_grid(12.0, (grid["f0"][i] - 4.0, grid["f0"][i] + 4.0), df, len(w),
                     step_f0=1.0, step_d=2.0)
    fine["ok"] = (fine["on"][3] >= 4) & (fine["mid"][3] >= 3)
    surface = comb_surface(w, fine)
    i, j = np.unravel_index(np.argmax(surface), surface.shape)
    candidate = (float(surface[i, j]), float(fine["f0"][i]), float(fine["delta"][j]))
    return max(best, candidate)


def peaks(w, mag, df, lo=110.0, hi=3600.0):
    """Prominent partials as (Hz, weight), interpolated exactly for a Hann peak."""
    out = []
    for k in range(2, len(w) - 2):
        if not (w[k] >= w[k - 1] and w[k] > w[k + 1]) or w[k] < 0.6:
            continue
        f = k * df
        if not (lo <= f <= hi):
            continue
        valley = min(w[max(0, k - 5):k].min(), w[k + 1:k + 6].min())
        if w[k] - valley < 0.4:
            continue
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        den = a + 2 * b + c
        frac = 2 * (c - a) / den if den > 0 else 0.0
        out.append(((k + frac) * df, min(max(w[k], 0.5), 4.0)))
    return out


def fit_lattice(peak_list, f0, delta, iterations=4):
    for _ in range(iterations):
        hs, ps, ws = [], [], []
        for p, weight in peak_list:
            h = round((p - delta) / f0)
            if h < 1 or h > 14 or abs(p - (h * f0 + delta)) > 0.22 * f0:
                continue
            hs.append(h)
            ps.append(p)
            ws.append(weight)
        if len(hs) < 3:
            return f0, delta, len(hs)
        a = np.array(hs, dtype=float)
        p = np.array(ps)
        w = np.array(ws)
        design = np.vstack([a, np.ones_like(a)]).T
        f0_new, delta_new = np.linalg.lstsq(design * w[:, None], p * w, rcond=None)[0]
        if not 0.5 * f0 < f0_new < 2.0 * f0:
            return f0, delta, len(hs)
        f0, delta = float(f0_new), float(delta_new)
    return f0, delta, len(hs)


def analyse(x, rate, window=2048, hop=480, max_shift=300.0, f0_range=(80.0, 320.0),
            min_confidence=0.4):
    """Per-frame (f0, offset, contrast) measurements."""
    rows = []
    for start in range(0, len(x) - window, hop):
        w, mag, df = spectrum(x[start:start + window], window, rate)
        score, f0, delta = coarse_grid(w, df, max_shift, f0_range)
        if score < min_confidence:
            continue
        peak_list = peaks(w, mag, df)
        f0, delta, used = fit_lattice(peak_list, f0, delta)
        if used < 3:
            continue
        rows.append((start / rate, f0, delta, comb_contrast(w, df, f0, delta), used))
    return np.array(rows, dtype=float).reshape(-1, 5)


def report(name, x, rate, args):
    rows = analyse(x, rate, args.window, args.hop, args.max_shift, (args.f0_min, args.f0_max))
    if len(rows) == 0:
        print(f"{name}: no harmonic frames found")
        return None
    t, f0, delta, contrast = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    # Canonical alias: the smallest residual against the observed spacing.
    reduced = delta - f0 * np.round(delta / f0)
    print(f"\n{name}: {len(rows)} harmonic frames of {len(x) / rate:.2f} s")
    print(f"  f0            median {np.median(f0):5.1f} Hz  "
          f"p10..p90 {np.percentile(f0, 10):5.1f}..{np.percentile(f0, 90):5.1f}")
    print(f"  offset        median {np.median(reduced):+6.1f} Hz  "
          f"p10..p90 {np.percentile(reduced, 10):+6.1f}..{np.percentile(reduced, 90):+6.1f}")
    print(f"  median per 3 s offset: " +
          " ".join(f"{np.median(reduced[(t >= s) & (t < s + 3)]):+5.0f}"
                   for s in range(0, int(t[-1]) + 1, 3)
                   if np.any((t >= s) & (t < s + 3))))
    print(f"  lattice contrast  median {np.median(contrast):5.2f} "
          f"(higher = more energy exactly on the harmonic comb)")
    return dict(f0=float(np.median(f0)), offset=float(np.median(reduced)),
                contrast=float(np.median(contrast)), frames=len(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--compare", metavar="WAV", help="second file, e.g. the corrected output")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=480)
    ap.add_argument("--max-shift", type=float, default=300.0)
    ap.add_argument("--f0-min", type=float, default=80.0)
    ap.add_argument("--f0-max", type=float, default=320.0)
    args = ap.parse_args()

    x, rate = load_wav(args.wav)
    before = report(args.wav, x, rate, args)
    if args.compare:
        y, rate2 = load_wav(args.compare)
        if rate2 != rate:
            raise SystemExit("sample rates differ")
        after = report(args.compare, y, rate, args)
        if before and after:
            print(f"\ncontrast {before['contrast']:.2f} -> {after['contrast']:.2f} "
                  f"({after['contrast'] - before['contrast']:+.2f})")
            print(f"offset   {before['offset']:+.1f} -> {after['offset']:+.1f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
