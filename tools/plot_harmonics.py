#!/usr/bin/env python3
"""Per-window vocal harmonic detection on a WAV channel.

Model per FFT frame:  f_k = phi + k * f0     (equal spacing f0, unknown offset phi)

phi is not arbitrary per frame: it drifts slowly, with occasional jumps of at
most one per J seconds. So the estimation runs in three passes:

1. raw  : per frame, seed f0 from the time-domain autocorrelation, seed phi
          from the circular mean of (peak_freq mod f0), then joint weighted
          least squares of (phi, f0) over the detected spectral peaks.
2. track: follow phi with slow adaptation; a sustained deviation (> tol for
          jump_persist frames) is a jump. Jumps closer than J seconds are
          suppressed (the rule says they cannot happen) and flagged.
3. verify: test the comb f_k = phi_tracked + k*f0 against the spectrum, keep
          the lines that are really there, score 0..1 confidence.
"""

import argparse

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import find_peaks

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def stft_analyze(sig, fs, nfft, hop, fmax):
    """dB spectrogram (frames x bins) + per-frame time-ACF pitch and peakiness."""
    win = np.hanning(nfft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(sig, nfft)[::hop]
    nfr = frames.shape[0]
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = freqs <= fmax

    S_db = np.empty((nfr, int(keep.sum())), np.float32)
    acf_peak = np.zeros(nfr, np.float32)
    f0_acf = np.zeros(nfr, np.float32)

    lag_lo, lag_hi = int(fs / 400.0), int(fs / 60.0)
    bias = 1.0 / (1.0 - np.arange(lag_lo, lag_hi + 1) / nfft)

    block = 256
    for b0 in range(0, nfr, block):
        b1 = min(nfr, b0 + block)
        X = np.fft.rfft(frames[b0:b1] * win, axis=1)
        P = X.real**2 + X.imag**2
        S_db[b0:b1] = (10.0 * np.log10(P[:, keep] + 1e-12)).astype(np.float32)

        acf = np.fft.irfft(P, n=nfft, axis=1)
        acf /= acf[:, :1] + 1e-20
        seg = acf[:, lag_lo : lag_hi + 1] * bias
        mx = seg.max(axis=1)
        j = np.clip((seg >= 0.88 * mx[:, None]).argmax(axis=1), 1, seg.shape[1] - 2)

        r = np.arange(len(j))
        ym, y0, yp = seg[r, j - 1], seg[r, j], seg[r, j + 1]
        den = ym - 2 * y0 + yp
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (ym - yp) / np.where(den == 0, 1, den), 0.0)

        f0_acf[b0:b1] = fs / (lag_lo + j + np.clip(delta, -1, 1))
        acf_peak[b0:b1] = mx

    return S_db, freqs[keep], f0_acf, acf_peak, nfr


def raw_fit(S_db, freqs, f0_seed, present_db, min_peaks, n_div=4, base_hz=150.0):
    """Pass 1: per-frame (phi, f0) comb fit against the spectral peaks.

    The autocorrelation seed can pick a multiple of the true spacing (every second
    line), so after the least-squares fit the spacings s/1 .. s/n_div are all fitted
    and scored. The score is the on-grid minus half-spacing-shifted presence inside
    the observed band: a spacing that skips real interior lines scores ~0, because
    the shifted grid lands on the lines it skipped.
    """
    nfr = S_db.shape[0]
    binw = freqs[1] - freqs[0]
    base = median_filter(S_db, size=(1, max(3, int(round(base_hz / binw)) | 1)))
    smax = maximum_filter1d(S_db, size=3, axis=1)

    phi = np.zeros(nfr)
    f0 = np.zeros(nfr)
    fit_q = np.zeros(nfr)
    coh = np.zeros(nfr)
    n_match = np.zeros(nfr, int)

    def lsq(frame, fp, w, f_seed):
        if not (55.0 <= f_seed <= 420.0):
            return None
        z = np.sum(w * np.exp(1j * 2.0 * np.pi * fp / f_seed))
        if abs(z) < 1e-9:
            return None
        ph = (np.angle(z) / (2.0 * np.pi)) * f_seed
        fc = f_seed
        for _ in range(5):
            m = np.round((fp - ph) / fc)
            sel = np.abs(fp - (ph + m * fc)) < 0.18 * fc
            if sel.sum() < min_peaks:
                return None
            sw = np.sqrt(w[sel])
            X = np.stack([np.ones(int(sel.sum())), m[sel]], axis=1)
            (ph, fc), *_ = np.linalg.lstsq(X * sw[:, None], fp[sel] * sw, rcond=None)
            ph, fc = float(ph), float(fc)
            if not (55.0 <= fc <= 420.0):
                return None
        # phi from the residues of the matched peaks, spacing from a 1-D fit with
        # the harmonic index held fixed. phi never enters the design matrix, so it
        # cannot be dragged towards 0 by the fit.
        for _ in range(3):
            k = np.round((fp - ph) / fc)
            sel = np.abs(fp - (ph + k * fc)) < 0.18 * fc
            if sel.sum() < min_peaks:
                return None
            res = fp[sel] - k[sel] * fc
            ang = 2.0 * np.pi * res / fc
            zz = np.sum(w[sel] * np.exp(1j * ang))
            if abs(zz) < 1e-12:
                return None
            ph = (np.angle(zz) / (2.0 * np.pi)) * fc
            k = np.round((fp[sel] - ph) / fc)
            ww = np.sqrt(w[sel])
            X = k[:, None]
            slope, *_ = np.linalg.lstsq(X * ww[:, None], (fp[sel] - ph) * ww, rcond=None)
            fc_new = float(slope[0])
            if not (55.0 <= fc_new <= 420.0):
                break
            fc = fc_new
        k = np.round((fp - (ph)) / fc)
        sel = np.abs(fp - (ph + k * fc)) < 0.18 * fc
        resid = fp[sel] - (ph + k[sel] * fc) if sel.any() else fp - ph
        coh = float(abs(zz) / max(np.sum(w[sel]), 1e-12))
        return ph, fc, int(sel.sum()), float(np.mean(np.abs(resid))), coh

    for i in range(nfr):
        pk, props = find_peaks(S_db[i], prominence=present_db, distance=2)
        inner = pk[(pk > 0) & (pk < S_db.shape[1] - 1)]
        if len(inner) < min_peaks:
            continue
        y0, y1, y2 = S_db[i, inner - 1], S_db[i, inner], S_db[i, inner + 1]
        den = y0 - 2 * y1 + y2
        d = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
        fp = freqs[inner] + np.clip(d, -0.5, 0.5) * binw
        w = props["prominences"][np.isin(pk, inner)].astype(float)

        seed = float(f0_seed[i])
        if not np.isfinite(seed) or not (55.0 <= seed <= 420.0):
            continue

        # seeds: the autocorrelation value plus the step seen between the peaks
        # themselves. The ACF can be a multiple of the true spacing, and on a
        # gliding (vibrato) comb it can miss entirely, so both are tried.
        dd = np.diff(np.sort(fp))
        dd = dd[dd > 0]
        seeds = [seed, seed / 2.0, seed / 3.0]
        if len(dd):
            med = float(np.median(dd))
            dd2 = dd[(dd > 0.6 * med) & (dd < 1.6 * med)]
            d_seed = float(np.median(dd2)) if len(dd2) else med
            seeds += [d_seed, d_seed / 2.0, d_seed * 2.0, d_seed / 3.0]
        def gather_fits(seed_list):
            out = []
            for sd in seed_list:
                r = lsq(i, fp, w, sd)
                if r is not None:
                    out.append(r)
                    for div in range(2, n_div + 1):
                        r2 = lsq(i, fp, w, r[1] / div)
                        if r2 is not None:
                            out.append(r2)
            return out

        fits = gather_fits(seeds)

        # score: on-grid presence minus half-spacing presence, inside the peak span
        lo, hi = float(fp.min()), float(fp.max())
        best = None
        for (ph, fc, nm, resid, c) in fits:
            ks = np.arange(np.ceil((lo - ph) / fc), np.floor((hi - ph) / fc) + 1)
            if len(ks) < min_peaks:
                continue
            on = ph + ks * fc
            off = on + 0.5 * fc
            idx = np.searchsorted(freqs, on).clip(0, len(freqs) - 1)
            idx2 = np.searchsorted(freqs, off).clip(0, len(freqs) - 1)
            p_on = np.clip(smax[i, idx] - base[i, idx], 0, None)
            p_off = np.clip(smax[i, idx2] - base[i, idx2], 0, None).mean()
            cover = float(np.mean(p_on > present_db))
            score = float((p_on.mean() - p_off) * cover)
            if best is None or score > best[0] + 0.3:
                best = (score, ph, fc, nm, resid, c)

        # a gliding comb can defeat every seed: sweep spacings and keep the best
        if best is None:
            for (ph, fc, nm, resid, c) in gather_fits(np.arange(60.0, 260.0, 2.5)):
                ks = np.arange(np.ceil((lo - ph) / fc), np.floor((hi - ph) / fc) + 1)
                if len(ks) < min_peaks:
                    continue
                on = ph + ks * fc
                off = on + 0.5 * fc
                idx = np.searchsorted(freqs, on).clip(0, len(freqs) - 1)
                idx2 = np.searchsorted(freqs, off).clip(0, len(freqs) - 1)
                p_on = np.clip(smax[i, idx] - base[i, idx], 0, None)
                p_off = np.clip(smax[i, idx2] - base[i, idx2], 0, None).mean()
                cover = float(np.mean(p_on > present_db))
                score = float((p_on.mean() - p_off) * cover)
                if best is None or score > best[0] + 0.3:
                    best = (score, ph, fc, nm, resid, c)

        if best is None:
            continue
        _, ph, fc, nm, resid, c = best
        phi[i], f0[i] = ph % fc, fc
        n_match[i] = nm
        coh[i] = c
        fit_q[i] = np.clip(1.0 - 3.0 * resid / fc, 0.0, 1.0)

    return phi, f0, fit_q, coh, n_match


def verify(S_db, freqs, phi, f0, fmax, present_db=7.0, fmin_line=50.0, n_lines=5,
           base_hz=150.0):
    """Pass 3: on-grid vs half-spacing-shifted comb test; returns present lines.

    A line counts as present when it stands `present_db` above its *local*
    baseline (median level over +-base_hz/2), not above a global floor.
    """
    nfr = S_db.shape[0]
    f_hi = 0.97 * fmax
    binw = freqs[1] - freqs[0]
    base = median_filter(S_db, size=(1, max(3, int(round(base_hz / binw)) | 1)))
    smax = maximum_filter1d(S_db, size=3, axis=1)

    mt = np.arange(0, n_lines)[None, :].astype(float)
    lines = phi[:, None] + mt * f0[:, None]
    ok = (f0 > 0)[:, None]
    valid = ok & (lines >= fmin_line) & (lines <= f_hi)

    def gather(f):
        idx = np.searchsorted(freqs, f).clip(0, len(freqs) - 1)
        return np.clip(smax[np.arange(nfr)[:, None], idx] - base[np.arange(nfr)[:, None], idx],
                       0.0, None)

    p = np.where(valid, gather(lines), 0.0)
    p_off = np.where(valid, gather(lines + 0.5 * f0[:, None]), 0.0)

    nv = np.maximum(valid.sum(axis=1), 1)
    present = (p > present_db) & valid
    coverage = present.sum(axis=1) / nv
    p_al, p_off_m = p.sum(axis=1) / nv, p_off.sum(axis=1) / nv
    comb_ratio = np.clip((p_al - p_off_m) / np.maximum(p_al, 1e-9), 0.0, 1.0)
    strength = np.clip(p_al / 18.0, 0.0, 1.0)
    return present, coverage, comb_ratio, strength, nv


def follow_stable(cand, green, dt, conf=None, window_s=3.0, value_s=0.2, tol_green=8.0,
                  margin=0.65, switch_hold=0.4, lo=-1e9, hi=1e9, conf_min=0.10,
                  slew=30.0):
    """Follow the most stable candidate population, lazily.

    cand  : (frames, K) candidate positions, NaN where unavailable
    green : (frames, lines) verified harmonic positions, NaN where absent
    Stability is the median absolute deviation of each population over the last
    window_s seconds. A population is dropped while it overlaps verified harmonics
    (it cannot be told apart from a harmonic) and while it is out of view.
    Low-confidence frames hold the current population instead of switching, and the
    drawn value is glided at most `slew` Hz/s, so it cannot step.
    """
    nfr, K = cand.shape
    W = max(3, int(round(window_s / dt)))
    V = max(1, int(round(value_s / dt)))
    hold = max(1, int(round(switch_hold / dt)))

    spread = np.full((nfr, K), np.inf)
    for k in range(K):
        v = cand[:, k]
        for i in range(nfr):
            w = v[max(0, i - W + 1):i + 1]
            w = w[np.isfinite(w)]
            if w.size >= max(8, W // 10):
                spread[i, k] = np.median(np.abs(w - np.median(w)))

    near_green = np.zeros((nfr, K), bool)
    for i in range(nfr):
        g = green[i][np.isfinite(green[i])]
        if not g.size:
            continue
        for k in range(K):
            if np.isfinite(cand[i, k]) and np.min(np.abs(g - cand[i, k])) < tol_green:
                near_green[i, k] = True
    # per frame: a candidate sitting on a verified harmonic right now is ambiguous.
    # k = 0 is the phi population itself, which sits on the m = 0 harmonic by
    # construction, so it is not treated as ambiguous.
    mixed = near_green.copy()
    mixed[:, 0] = False
    valid = np.isfinite(cand) & ~mixed & (cand >= lo) & (cand <= hi)

    out = np.full(nfr, np.nan)
    cur = -1
    alt = -1
    alt_run = 0
    last = np.nan
    step = slew * dt if slew > 0 else np.inf  # slew <= 0: sharp moves allowed
    for i in range(nfr):
        quiet = (conf is not None) and (conf[i] < conf_min)
        s = np.where(valid[i], spread[i], np.inf)
        if not quiet:  # low confidence: keep the population, do not switch
            if cur < 0 or not valid[i, cur]:
                if np.isfinite(s).any():
                    cur = int(np.argmin(s))
                    alt, alt_run = -1, 0
            else:
                a = int(np.argmin(s))
                if a != cur and s[a] < margin * spread[i, cur]:
                    if a == alt:
                        alt_run += 1
                    else:
                        alt, alt_run = a, 1
                    if alt_run >= hold:
                        cur, alt, alt_run = a, -1, 0
                else:
                    alt, alt_run = -1, 0
        if cur >= 0:
            w = cand[max(0, i - V + 1):i + 1, cur]
            w = w[np.isfinite(w)]
            if w.size:
                tgt = float(np.median(w))
                if not np.isfinite(last):
                    last = tgt
                else:
                    last += float(np.clip(tgt - last, -step, step))
                out[i] = last
        if np.isfinite(out[i]):
            last = out[i]
    return out, spread, mixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-c", "--channel", type=int, default=0)
    ap.add_argument("--fmax", type=float, default=3000.0)
    ap.add_argument("--nfft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--cmap", default="magma")
    ap.add_argument("--lines", type=int, default=5, help="max harmonics drawn per frame")
    ap.add_argument("--cand", type=int, default=4, help="candidate f0 lines below f1")
    ap.add_argument("--follow-s", type=float, default=3.0, dest="follow_s",
                    help="seconds of history used to judge which population is stable")
    ap.add_argument("--slew", type=float, default=0.0,
                    help="max speed of the stable-population line [Hz/s]; 0 = unlimited")
    ap.add_argument("--conf-min", type=float, default=0.10, dest="conf_min",
                    help="below this confidence the line holds and does not switch")
    ap.add_argument("--disp-max", type=float, default=1000.0, dest="disp_max",
                    help="displayed frequency range, upper limit [Hz]")
    ap.add_argument("--disp-min", type=float, default=-200.0, dest="disp_min",
                    help="displayed frequency range, lower limit of chart 2 [Hz]")
    ap.add_argument("--ytick", type=float, default=100.0, help="y grid step on both charts [Hz]")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    x, fs = sf.read(args.wav, dtype="float32", always_2d=True)
    sig = x[:, args.channel]

    S_db, freqs, f0_seed, acf_peak, nfr = stft_analyze(sig, fs, args.nfft, args.hop, args.fmax)

    # 1) raw per-frame comb fit
    phi_raw, f0, fit_q, coh, n_match = raw_fit(
        S_db, freqs, f0_seed, present_db=6.0, min_peaks=3)

    # coarse per-frame quality, used only to gate the tracking
    _, cov0, comb0, str0, _ = verify(S_db, freqs, phi_raw, f0, args.fmax, n_lines=args.lines)
    q0 = np.clip((comb0 - 0.05) / 0.45, 0, 1) * (0.55 * cov0 + 0.45 * str0) * fit_q

    # 2) phi is per slice: it comes from the residues of the matched peaks of that
    #    slice. It is never smoothed and never tied to the spacing.
    phi = phi_raw
    jump_flag = np.zeros(nfr, bool)
    blocked = np.zeros(nfr, bool)

    # 3) verify comb on the tracked offset
    present, coverage, comb_ratio, strength, n_lines = verify(
        S_db, freqs, phi, f0, args.fmax, n_lines=args.lines)

    comb_q = np.clip((comb_ratio - 0.05) / 0.45, 0.0, 1.0)
    voicing = np.clip((acf_peak - 0.15) / 0.45, 0.0, 1.0)
    conf = comb_q * (0.55 * coverage + 0.45 * strength) * (0.35 + 0.65 * fit_q) * voicing
    conf = np.where(acf_peak < 0.25, 0.0, conf)
    conf = np.where(blocked, conf * 0.5, conf)
    conf = np.clip(conf, 0.0, 1.0)

    dt = args.hop / fs

    if args.debug:
        for name, v in [("coverage", coverage), ("comb", comb_q), ("strength", strength),
                        ("fit", fit_q), ("voicing", voicing), ("f0", f0), ("phi", phi),
                        ("conf", conf)]:
            print(f"{name:9s} p05={np.percentile(v,5):7.3f} p50={np.percentile(v,50):7.3f} "
                  f"p75={np.percentile(v,75):7.3f} p95={np.percentile(v,95):7.3f}")
        print(f"frames={nfr} dur={nfr*args.hop/fs:.2f}s  conf>0.5: {(conf>0.5).sum()} "
              f"({100*(conf>0.5).mean():.1f}%)")
        ok = conf > 0.5
        if ok.any():
            print(f"median f0={np.median(f0[ok]):.1f} Hz  phi/s: "
                  f"p50={np.median((phi[ok]/f0[ok])%1.0):.2f}  phi: "
                  f"p05={np.percentile(phi[ok],5):.1f} p50={np.percentile(phi[ok],50):.1f} "
                  f"p95={np.percentile(phi[ok],95):.1f} Hz  "
                  f"coherence p50={np.median(coh[ok]):.2f}")

    dt = args.hop / fs
    t = (args.nfft / 2.0) / fs + np.arange(nfr) * dt
    t0, t1 = t[0], t[-1] + dt

    disp = freqs <= args.disp_max  # auto-brightness taken from the displayed band
    lo, hi = np.percentile(S_db[:, disp], [5.0, 99.9])
    if hi - lo < 1.0:
        hi = lo + 1.0
    img = np.clip((S_db - lo) / (hi - lo), 0.0, 1.0).T

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, 9.5), dpi=150, sharex=True,
        gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.12),
    )
    ax1.imshow(img, origin="lower", aspect="auto", cmap=args.cmap,
               extent=[t0, t1, freqs[0], freqs[-1]], interpolation="nearest",
               vmin=0.0, vmax=1.0)
    ax1.set_ylabel("frequency [Hz]")
    ax1.set_ylim(0, args.disp_max)
    ax1.set_yticks(np.arange(0.0, args.disp_max + 0.5 * args.ytick, args.ytick))
    ax1.grid(axis="y", color="white", alpha=0.4, linewidth=0.7)
    ax1.set_title(
        f"{args.wav}  ch {args.channel}/{x.shape[1]}  fs={fs} Hz  nfft={args.nfft} "
        f"hop={args.hop}  model f_k = phi(t) + k*f0(t)"
    )

    # green: detected harmonics f_k = phi(tracked) + k*f0, at most --lines per frame
    mt = np.arange(0, args.lines)[None, :].astype(float)
    lines = phi[:, None] + mt * f0[:, None]
    fi, ki = np.nonzero(present)
    segs = np.stack([
        np.stack([t[fi], lines[fi, ki]], axis=1),
        np.stack([t[fi] + dt, lines[fi, ki]], axis=1),
    ], axis=1)
    ax2.add_collection(LineCollection(segs, colors="#00ff5a", linewidths=1.1, antialiaseds=False))
    ax2.set_facecolor("#050505")
    ax2.set_ylim(args.disp_min, args.disp_max)
    ax2.set_xlim(t0, t1)
    ax2.set_xlabel("time [s]  →")
    ax2.set_ylabel("harmonic frequency [Hz]")

    # red: candidate f0 positions = the same comb continued below the lowest visible
    # harmonic, i.e. phi + k*s for k = 0, -1, -2, ...  The step is exactly the green
    # step, so these are where the hidden low harmonics would sit. Negative
    # frequencies are drawn too (the y axis is extended to fit them).
    # k = 0 is the base element phi itself; the negative k continue below it
    kl = -np.arange(0, args.cand)[None, :].astype(float)
    y_red = np.mod(phi, f0)[:, None] + kl * f0[:, None]
    dc = np.abs(y_red) < 1.5  # only true DC is meaningless as a candidate
    thin = ((conf > 0.15)[:, None] & (f0 > 0)[:, None] & np.ones((1, args.cand), bool)
            & ~dc)
    cpos = np.where(thin, y_red, np.nan)
    y_lo = 0.0
    for n in range(args.cand):
        ri = np.nonzero(thin[:, n])[0]
        if not len(ri):
            continue
        y = y_red[ri, n]
        y_lo = min(y_lo, float(y.min()))
        seg_n = np.stack([np.stack([t[ri], y], axis=1),
                          np.stack([t[ri] + dt, y], axis=1)], axis=1)
        ax2.add_collection(LineCollection(seg_n, colors="#ff3b3b", linewidths=0.9, alpha=0.9))
    # thick, pale cyan line: the most stable candidate population, followed lazily
    follow, _, _ = follow_stable(cpos, np.where(present, lines, np.nan), dt, conf=conf,
                                 window_s=args.follow_s, conf_min=args.conf_min,
                                 slew=args.slew,
                                 lo=args.disp_min + 0.02 * args.disp_max,
                                 hi=args.disp_max)
    ax2.plot(t, follow, color="#3ef0ff", lw=3.0, alpha=0.35, solid_capstyle="round",
             zorder=9)
    ax2.set_yticks(np.arange(args.disp_min, args.disp_max + 0.5 * args.ytick, args.ytick))
    if y_lo < args.disp_min:
        print(f"note: lowest drawn candidate {y_lo:.0f} Hz is below the display window")

    axc = ax2.twinx()
    axc.fill_between(t, conf, color="#4fd6ff", alpha=0.16, linewidth=0, step="mid")
    axc.plot(t, conf, color="#7fe9ff", linewidth=0.7, alpha=0.28)
    axc.set_ylim(0, 1.0)
    axc.set_ylabel("confidence (0..1)", color="#7fe9ff")
    axc.tick_params(axis="y", colors="#7fe9ff")
    axc.grid(False)
    for s in axc.spines.values():
        s.set_color("#7fe9ff")
        s.set_alpha(0.5)

    ax2.legend(
        handles=[Line2D([], [], color="#00ff5a", lw=1.4, label="harmonic f_k = phi + k*f0"),
                 Line2D([], [], color="#ff3b3b", lw=1.0, label="f0 candidates phi - k*s"),
                 Line2D([], [], color="#3ef0ff", lw=3.0, alpha=0.5,
                        label=f"stable population ({args.follow_s:g}s)"),
                 Line2D([], [], color="#7fe9ff", lw=1.2, alpha=0.6, label="confidence"),
                 Line2D([], [], color="#ffd39b", lw=1.0, label="phi of that frame")],
        loc="upper right", facecolor="black", edgecolor="#444", labelcolor="white", fontsize=8,
    )
    ax2.grid(color="white", alpha=0.3, linewidth=0.7)

    # every 0.2 s: print the absolute frequency of three verified harmonics, stacked
    # one under another, all shifted right by the same amount while the measured
    # points stay put; a pale blue pointer leads from each number to its point.
    lab_every = 0.2
    lab_win = max(1, int(round(0.2 / dt)))
    for tl in np.arange(t[0], t[-1], lab_every):
        i0 = int(np.clip(np.searchsorted(t, tl), 0, nfr - 1))
        i = None
        for j in range(i0, min(nfr, i0 + lab_win + 1)):
            if conf[j] > 0.2 and present[j].sum() >= 3:
                i = j
                break
        if i is None:
            continue
        ys = np.sort(np.where(present[i], lines[i], np.nan))
        ys = ys[np.isfinite(ys)][:3]
        for yv in ys:
            ax2.annotate(f"{yv:.0f}", xy=(t[i], yv), xytext=(2.6, 0),
                         textcoords="offset points", fontsize=2.4, color="#cfe0ff",
                         ha="left", va="center", zorder=12,
                         arrowprops=dict(arrowstyle="-", color="#5f9bff", lw=0.35,
                                         alpha=0.85, shrinkA=0, shrinkB=0.5))
        if len(ys):
            ax2.annotate(f"\u03c6{phi[i]:.0f}", xy=(t[i], ys.max()), xytext=(2.6, 4.0),
                         textcoords="offset points", fontsize=2.4, color="#ffd39b",
                         ha="left", va="center", zorder=12)

    fig.subplots_adjust(left=0.05, right=0.94, top=0.94, bottom=0.07, hspace=0.12)

    out = args.out or f"{args.wav.rsplit('.', 1)[0]}_ch{args.channel}_harmonics.pdf"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
