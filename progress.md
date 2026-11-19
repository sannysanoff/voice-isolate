# Progress: SIMD and memory-layout optimisation

Speed is always reported **relative to realtime, from `user` CPU time** — the
honest measure, independent of how many threads happen to run. `user / audio
seconds` is the ratio: 1.0 means one CPU-second per audio second, 0.12 means the
job costs 8.3 CPU-seconds per audio second.

Test signal: `audio_7082000Hz_18-59-39_25-06-2024-L20-endfed-maticna.wav`
(44.29 s) and its 5 s excerpt `/tmp/short5.wav` (deliberately kept in /tmp, it is
only a slice of the same recording).

## Baseline — before this round of optimisation

fp32, `-O3 -fno-math-errno`, no `-march=native`, Eigen 3.4.

| threads | audio | wall | user | sys | user/realtime |
|---|---|---|---|---|---|
| 1 | 5.00 s | 41.35 s | 39.5 s | ~1 s | **7.9×** (0.13× realtime) |
| 6 | 5.00 s | 14.12 s | ~43 s | 1.5 s | **8.6×** (0.12× realtime) |
| 8 | 44.29 s | 163.5 s | 456 s | 18 s | **10.3×** (0.097× realtime) |

Stage split, 5 s clip, 1 thread (wall): encode 0.03, time attention 12.92,
band attention 10.57, mamba 15.23, decode 0.70 — total 39.5 s.

Stage split, 5 s clip, 6 threads (wall): encode 0.01, time attention 4.33,
band attention 3.29, mamba 5.71, decode 0.27 — total 13.6 s.

So: ~40 % of the time is not GEMM, and the pool delivers only ~2.9× on 6 cores.
Both are targeted below.

## Planned work, in order of expected gain

1. **K/V cache layout**: `[band][slot][head*hd]` → `[band][head][slot][hd]`.
   Each head currently reads 600 keys with a 2 KB stride and uses 256 B of every
   line — **87.5 % of the cache-line traffic is waste**.
2. **Band attention q/k/v** → `[frame][head][band][hd]`, same reason (6 KB stride).
3. **Mamba scan**: fuse `exp(delta*A)` into the state update, drop the 64 KB
   `decay` buffer per (task, step), vectorise the recurrence with xsimd.
4. **Depthwise convolution**: vectorise, and fold the SiLU into it (one pass
   fewer).
5. **LayerNorm**: still a scalar loop — vectorise.
6. **`linear_par`**: `x_proj`/`dt_proj` have `n = 64`, so the column split
   degenerates to 1 chunk and they run serial. Split rows for small `n`.
7. **RoPE, softmax sums, residual adds**: vectorise.
8. **`-march=native` by default** (desktop only from now on). On x86-64 the
   baseline is SSE2, so without this Eigen loses ~4× on the target Intel box.

## Status

- [x] `vecmath`: `v_scale`, `v_sum`, `v_scan_step` (fused decay), vectorised RoPE
- [ ] time attention on a head-major K/V cache
- [ ] band attention on head-major q/k/v
- [ ] mamba conv + fused scan
- [ ] vectorised LayerNorm
- [ ] row split for small-`n` GEMMs
- [ ] build flags
- [ ] re-verify against the reference dumps, re-measure

## Verification gate

Every stage must stay ≥ 100 dB SNR against `ref/stages`, and the whole file
≥ 100 dB; the 44 s file must stay within ~1e-4 of `…-maticna.clean2.wav`.
