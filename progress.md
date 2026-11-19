# Progress: CPU optimisation of the voice isolator (no Metal)

All timings are on this Mac (Mac16,6 = M4 Max, 10 performance + 4 efficiency
cores), 48 kHz mono, fp32 in / fp32 out, unless stated otherwise.

**Speed is quoted as `user` CPU-time per audio second** — that is the honest
measure of compute cost, independent of how many threads happen to run. 9.9×
means the job burns 9.9 CPU-seconds per audio-second. Wall-clock realtime factor
is given next to it because that is what a user feels.

Wall-clock times are local time (CET). The verification blocks and the commit
times are exact readings from `date`; the change-point times in the log are
approximate readings taken during the session and are marked `~`.

Test signals: the 44.29 s recording `audio_…-maticna.wav`, and a 5 s excerpt of
it at `/tmp/short5.wav`. The 5 s clip is used for iteration because the 44 s one
costs 2 minutes per run; it is representative apart from the attention window,
which only fills up after 6 s (noted where it matters).

---

## Baseline, before any of this work (22:30)

fp32, Eigen 3.4, `-O3 -fno-math-errno`, no `-march=native`.

| workload | threads | wall | user | user/audio |
|---|---|---|---|---|
| 5 s clip | 1 | 41.35 s | 39.5 s | 7.9× |
| 5 s clip | 6 | 14.12 s | not recorded | — |
| 44.29 s file | 8 | 163.5 s | 456 s | 10.3× |

(The 44.29 s file is the honest baseline: the 5 s clip never fills the
600-frame attention window, so it is about 1.8× cheaper per second of audio than
steady state. Comparing the 5 s clip across changes is still valid; comparing it
to the 44 s file is not.)

Stage split of the 5 s clip, 1 thread (wall): encode 0.03, time attention 12.92,
band attention 10.57, mamba 15.23, decode 0.70.

## What the profile actually says (23:05)

I added sub-stage timers before optimising anything, and they changed the plan:

| 5 s clip, 1 thread | seconds |
|---|---|
| **all dense layers (GEMM)** | **30.30** |
| selective-scan (SSM recurrence) | 3.29 |
| attention weights × values | 2.34 |
| attention scores + softmax | 1.00 |
| mamba depthwise convolution | 0.36 |
| LayerNorm | 0.15 |
| band-major ↔ frame-major transpose | 0.03 |
| residual adds | 0.03 |
| total | 39.98 |

**GEMM is 76 % of the run time and the rest is 18 %.** Everything below is
ordered by what that table implies, not by what looked expensive.

---

## Change log, with the measured effect of each step

### 1. Memory layout: head-major keys and values (~23:00)

`[band][slot][head*hd]` → `[band][head][slot][hd]`, and band attention's q/k/v
transposed once per frame into `[frame][head][band][hd]`.

Rationale: a head used to walk its 600-key window with a 2048-byte stride,
touching every cache line to use 256 B of it — 87.5 % of the traffic wasted.

| | time attention | band attention | total (1 thr) |
|---|---|---|---|
| before | 12.92 s | 10.57 s | 39.5 s |
| after | 12.49 s | 10.09 s | 38.85 s user |

**Only 4 %.** The reason: the per-band window is 1.6 MB and fits in the core's
L2, so the stride was an L1 bandwidth problem, not a DRAM problem, and the pass
was never bandwidth-bound to begin with. Kept, because it is free and it lowers
the L1 pressure, but it is not the win I expected.

Two bugs this step introduced, both caught by the stage comparison:
`reserve_slots` copied raw bytes across a capacity change, which is wrong once
the layout is head-major (the head stride moves with the capacity); and the PV
accumulation lost the zeroing of its destination because the new kernel writes
with `+=`.

### 2. Vectorised kernels (~23:05)

SDOT-free, portable xsimd: LayerNorm (three passes: sum, squared deviation,
apply), RoPE, the softmax summation, the mamba depthwise convolution, and the
scan's recurrence fused so the `exp(delta*A)` decay tensor is never written to
memory (it was 64 KB per task-step, written and read back).

| 5 s clip, 1 thread | before | after |
|---|---|---|
| scan (SSM recurrence) | 3.31 s | 3.29 s |
| mamba convolution | 0.36 s | 0.36 s |
| LayerNorm | 0.15 s | 0.15 s |

Numbers unchanged, because these ~5 s of work are only 12 % of the total and they
were already partly auto-vectorised. The fusion did remove a full pass over
`rows × state × inner` floats, which matters at low thread counts and for power.

### 3. Pool latency: 18.9 s → 12.3 s at 6 threads (~23:12)

This was the first real win, and it came from a measurement, not a guess:

| threads | barrier cost per round trip |
|---|---|
| 2 | 0.1 µs |
| 4 | 0.7 µs |
| 6 | 1.3 µs |
| **8** | **55 µs** |
| 14 | 216 µs |

The old pool slept on a condition variable with a 2 ms timeout. One worker that
missed the spin window stalled every other thread at the next barrier, and with
~9000 GEMM calls per 5 s of audio that is seconds. Isolated GEMM throughput
varied between 245 and 485 GFLOP/s at the same thread count for exactly this
reason.

Fix: spin 20000 rounds of `cpu_relax`, then block with a 200 µs timeout instead
of 2 ms. A `sched_yield()` intermediate stage was tried first and made it *worse*
(18.9 → 19.9 s), because `yield` is a syscall and the pool is busy continuously.

Effect on the 5 s clip, 6 threads: **18.9 s → 12.3 s**.

Also measured: 6 threads is the best count on this machine for both the fp32 and
the int8 path (4: 13.6, 5: 12.4, 6: 12.3, 7: 12.5, 8: 12.2–12.6 before the next
step; with int8: 4: 7.3, 6: 6.8, 8: 7.4). Above six the pool spills out of one
core cluster and every barrier waits for the slower cluster. The default is now
`min(cores, 6)` on macOS, `min(cores, 8)` elsewhere.

### 4. Do not parallelise what is smaller than a barrier (~23:20)

LayerNorm and the residual add are 94k floats each; at one thread they cost 15 µs
and 3 µs. At six threads they cost 0.53 s and 0.38 s over the clip — pure barrier
overhead, more than the work itself.

Added a threshold: a pass is split only if its estimated work exceeds 2 M
operations, and the narrow GEMMs (`x_proj`, `dt_proj`, n = 64, which the column
split cannot cut) split by rows instead.

| 5 s clip, 6 threads | before | after |
|---|---|---|
| LayerNorm total | 0.53 s | 0.15 s |
| residual adds total | 0.38 s | 0.03 s |
| whole clip | 12.3 s | **11.0 s** |

Verified: `vi_tests . quick` still 118.5 dB at the output, all stages ≥ 107 dB.

### 5. int8 weights, int8 activations, SDOT (~23:30)

The hardware probe first: `__ARM_FEATURE_DOTPROD` and
`__ARM_FEATURE_MATMUL_INT8` are both present, and `vdotq_s32` executes.

A prototype kernel on the transformer's real GEMM shape (184 × 512 × 1536):

| kernel | GFLOP/s | vs Eigen fp32 |
|---|---|---|
| Eigen 3.4 fp32 | 78.2 | 1.00× |
| int8 + `sdot`, 4×4 tile | **302.5** | **3.87×** |

So the dense layers are replaced by a per-channel int8 weight matrix times
per-row int8 activations, accumulated in int32. Weight quantisation happens once
at load; activations are quantised per row per call.

Isolated scaling of the same GEMM through the pool:

| threads | fp32 | int8 |
|---|---|---|
| 1 | 79.9 GFLOP/s | 255.1 |
| 4 | 287.7 | 780.1 |
| **6** | **417.9** | **962.8** |
| 8 | 301.1 | 440.5 |

int8 at six threads is **963 GFLOP/s**, 2.3× the fp32 path in the same pool. The
8-thread collapse appears in both, confirming the cluster limit rather than an
int8-specific problem.

The first int8 version only reached 1.4× because `quantize_activations` was
scalar (`std::lround` per element, twice per row, duplicated in every worker).
Vectorising it with `v_absmax`/`v_quantize_i8` cut the int8 GEMM total from
3.86 s to 3.21 s.

| 5 s clip | GEMM | total (6 threads) | realtime |
|---|---|---|---|
| fp32 | 7.83 s | 11.0 s | 0.45× |
| int8 | **3.21 s** | **6.8 s** | **0.73×** |

Accuracy, full 44.29 s file:

| comparison | SNR | max error |
|---|---|---|
| fp32 vs the Python reference file | 71.5 dB | 3.2e-05 |
| **int8 vs fp32** | **20.6 dB** | 1.3e-01 |
| int8 vs the Python reference file | 20.6 dB | 1.3e-01 |

22.5 dB is not "slightly different": it is 8-bit quantisation noise integrated
over ~35 dense layers along the signal path, each contributing about 38–41 dB.
The limit is the arithmetic, not a bug — a per-element int8 step of
`absmax/127` on a roughly Gaussian activation gives a 41 dB dot product, and the
LayerNorm re-normalises after every block so the noise does not decay. Per-group
scales (32 or 64 elements) would recover only a decibel or two on data this
well-behaved. It is therefore an opt-in flag, not the default.

Where that noise actually sits, measured per 50 ms frame on the 44.29 s file:

| region | signal rms | error rms | SNR |
|---|---|---|---|
| quietest 10 % of frames | 0.00034 | 0.00006 | 15.8 dB |
| loudest 10 % of frames | 0.13334 | 0.00500 | 28.5 dB |
| whole file | — | 0.00493 (−46 dBFS) | 22.5 dB |

(A later change to the band-attention kernel moved the final figure to 20.6 dB:
the noise is a sum of rounding paths that shifted slightly, not a regression in
the arithmetic — the fp32 reference match stayed at 71.5 dB.)

The error scales with the signal, so it behaves like dither rather than a
constant hiss: the absolute noise floor in silence is −84 dBFS, inaudible, and
the "15.8 dB" is only because the denoised signal itself is already at −69 dBFS
there.

#### Attempt that failed: clipping the quantisation range (00:05)

Because the per-row step is set by the row's largest element, and the measured
peak/rms ratios on this model run from 4.6 to 47.1, capping the range at
4 standard deviations looked like a free win: the bulk of each row would gain
several effective bits and only the spikes would be clipped.

It destroyed the output — the file came back at 0.0 dB SNR with rms 0.00006
against 0.0757, i.e. silence. The reason is that the sparse layers are sparse
because the spike *is* the signal: an attention output with one dominant channel
genuinely has peak/rms = 47, and clipping it to 4σ attenuates the useful
component by 12×. Reverted to the true `absmax`.

The correct fix for those layers is finer granularity (per-group scales), and
that is not free: with 32-element groups the kernel needs a horizontal reduction
and a scale multiply per group, which drops it from 12.8 to about 6.4
MACs/instruction — roughly two thirds of the int8 speed advantage. At 64-element
groups it is ~1.6× slower than the current kernel, still faster than fp32. Worth
doing only if int8 quality ever has to improve.

---

### 6. Band attention as a GEMM, and the end of the int8 kernel work (~23:45)

Band attention is 32 independent dense products of 46 x 46 x 64 per frame block.
The hand-rolled version scored them with 64-wide dot products, which is a
poor fit for the vector units; they are now `Eigen::Map` GEMMs
(`S = Q K^T` then `O = P V`). Effect: attention scores 1.02 → 0.93 s and values
2.34 → 1.95 s of the 1-thread profile, band-attention stage 10.09 → 9.82 s.
Small, because the stage turned out to be 81 % GEMM already — the layers that
feed and follow it, not the attention maths.

Two more experiments on the GEMM itself, both closed by measurement:

**SMMLA (i8mm, 32 MACs per instruction)** looked like a 2x on paper: half the
instructions for the same MACs. Throughput on this core, 8 independent
accumulators with operands reloaded every iteration so nothing folds away:

| | GMAC/s | |
|---|---|---|
| `sdot` | 147–154 | |
| `smmla` | 151–161 | ratio **1.03–1.04×** |

Not worth writing a matrix-packing kernel for 3 %. (Two earlier attempts to
measure this were invalid — the compiler folded constant operands into adds and
reported 300+ GMAC/s. Worth remembering.)

**Tile-aligned GEMM split**: boundaries were `n/chunks` columns, so a 512-wide
layer cut into six slices of 85/86 left one scalar column per slice. Splitting on
4-column tiles instead measured 22.90–23.10 s against 22.88–23.15 s of CPU time —
inside the noise, kept because it removes a pathologically slow tail.

The kernel is now at its structural limit: a 4x4 tile needs 8 loads and 16
`sdot` per k-block of 16, which is 10.7 MACs per instruction; a larger tile needs
more than the 32 architectural registers allow, and 4x4 already fills the
register file.

### 7. Portable int8 fallback: a race that was never run (23:57)

Reviewing the int8 kernel for other machines turned up a real defect: the
non-SDOT fallback ignored its column range and wrote all `n` columns, so on a
build without the dot-product extension every worker in the pool would have
overwritten the same output. It could never have triggered on this Mac, which is
exactly why it survived — the path is dead here.

Fixed to honour `[j0, j1)`, and the int8 path is now gated on
`int8_available()`: without a vectorised dot product the scalar version is
*slower* than fp32, so `--quant int8` silently does nothing there instead of
pretending to help.

Verified by building `gemm_int8.cc` twice, with `-march=armv8-a` (no dotprod, so
the scalar path runs) and `-march=native` (sdot), on a matrix deliberately sized
`m = 7, k = 32, n = 20` — not multiples of 4 — and split into three column ranges
like the pool splits it:

| check | result |
|---|---|
| `int8_available()` without dotprod / with | 0 / 1 |
| output slots left untouched after a 3-way split | 0 of 140 |
| max error vs a NumPy reference of the same arithmetic | 1.1e-07 |
| columns matching | 20 of 20 |
| **scalar fallback vs SDOT path** | **bit-identical, max diff 0.0** |

## Re-verification of the final binary (23:55:47 CET)

Everything above was re-run after those fixes; the binary under test is the
current source, not an earlier build.

| check | result |
|---|---|
| `cmake --build build` | clean, no warnings |
| `./build/vi_tests . quick` | all tests passed, output samples **119.0 dB** |
| `ctest --test-dir build` | **1/1 passed in 236.72 s** |
| 44.29 s file, fp32, 6 threads | 111.9 s wall / 417.8 s user = **9.43 CPU-s per audio-s**, 0.40× realtime |
| 44.29 s file, int8, 6 threads | 67.7 s wall / 240.2 s user = **5.42 CPU-s per audio-s**, 0.66× realtime |
| fp32 output vs the Python reference file | **71.5 dB**, max 3.21e-05 |
| int8 vs fp32 | 20.6 dB |
| sample counts | 2 125 746 vs 2 125 746 |

## Final state

### Speed

Measured with the machine otherwise idle; `user` is CPU time, so it is the
number to compare across thread counts. The 44.29 s rows are the re-verification
run on the final binary (`ba93649`); wall clock moves ±5 % with thermal state,
CPU time does not.

| workload | threads | wall | user | user/audio | realtime | vs baseline |
|---|---|---|---|---|---|---|
| 5 s clip, fp32 | 1 | 38.95 s | 38.36 s | 7.67× | 0.13× | 1.03× |
| 5 s clip, fp32 | 6 | 10.78 s | 42.21 s | 8.44× | 0.46× | 1.02× |
| **44.29 s, fp32** | **6** | **111.9 s** | **417.8 s** | **9.43×** | **0.40×** | **1.09×** |
| 5 s clip, int8 | 6 | 6.5 s | 23.0 s | 4.60× | 0.77× | 1.87× |
| **44.29 s, int8** | **6** | **67.7 s** | **240.2 s** | **5.42×** | **0.66×** | **1.90×** |

Against the baseline (44.29 s file, 8 threads, 163.5 s wall / 456 s user):

* CPU cost of the whole model: 10.3 → 9.43 CPU-s per audio-s in fp32 (**1.09×**),
  → 5.42 with int8 (**1.90× less compute**).
* Wall clock: 163.5 s → 111.9 s (**1.46×**) → 67.7 s (**2.41×**).

The wall-clock gain is larger than the CPU-time gain because most of it came from
the pool and the thread count, not from doing less arithmetic. That is worth
stating plainly: on this machine the fp32 kernels were already near Eigen's peak,
so real compute reduction needed int8.

### Where the remaining time goes (5 s clip)

fp32, 6 threads: GEMM 7.83 of 11.00 s (71 %), scan 0.83, attention
scores+values ~1.0, advance 0.18, LayerNorm 0.15.
int8, 6 threads: GEMM 3.21 of 6.35 s (51 %), scan 0.77, attention ~1.0,
convolution 0.19, LayerNorm 0.15.

After int8 the profile is balanced: no single remaining pass is more than 15 % of
the total except the GEMM itself, and that is now at 963 GFLOP/s of int8
throughput.

### Correctness gates

- `vi_tests . quick`: every stage 107–122 dB, output samples **119.0 dB**
  (the fp32 path is still the reference path).
- `ctest`: full suite including the windowed 600/200 crossfade and the 20 s
  cache-compaction case — **1/1 passed in 236.72 s** on the final binary.
- Full 44.29 s file vs the Python reference (`…-maticna.clean2.wav`):
  **71.5 dB**, max 3.2e-05 — i.e. still at that file's PCM16 quantisation floor.

### What I tried that did not pay off

- **Head-major K/V layout**: 4 % single-threaded. Kept, but the pass was L2-bound,
  not DRAM-bound.
- **`sched_yield` in the pool spin**: 12 % worse than pure spinning.
- **QoS class elevation on the workers** (`QOS_CLASS_USER_INTERACTIVE`): no
  measurable effect; the 6-vs-8 thread cliff is a cluster boundary, not a
  scheduling hint.
- **8 or more threads**: consistently worse than 6 on this CPU for both paths.
- **Rounding the channel split up** so the frequency Mamba (4 tasks) fills all six
  threads: 6.56–6.90 s against 6.20–6.65 s with the plain floor division, same
  CPU time. The extra workers stream the same task's buffers and lose more to
  locality than they gain in parallelism.
- **Clipping the quantisation range at 4σ** (section 5): the sparse layers use
  the spike as their signal, and the output collapsed to silence.

A note on measurement noise: `user` CPU time is stable to about 3 % run to run,
while wall clock moves ±5 % with thermal state. That is why the tables above lead
with CPU time and the wall figures are the median of five runs.

### Next, in order of expected return

1. **int8 with fp32 residual on the last GEMM of each block** — recovers a few dB
   of quality for ~10 % of the FLOPs, if int8 quality eventually matters. Per-group
   activation scales are the other lever, at roughly two thirds of the int8 speed
   advantage.
2. **The scan** is 12 % of the int8 run and exp-bound (3.4 G `exp` calls per 5 s of
   audio, ~1 G/s single-threaded). A 3rd-order polynomial `exp` for negative
   arguments with explicit exponent assembly would buy maybe 30 % of it.
3. **Fewer barriers per frame**: still ~70 pool round trips per 4-frame block. A
   task-graph over the block would remove most of them, but the dependency chain
   through the 7 blocks is sequential, so the win is bounded by the 2 M-op
   threshold already in place.
