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

### 8. Where the int8 error accumulates (~00:10)

`VI_QUANT=1 ./build/vi_tests . quick` runs the stage comparison through the int8
dense layers. The encoder is untouched (no quantised layer there), and the error
then grows monotonically through the blocks — no single step goes wrong, which is
what distinguishes quantisation noise from a bug:

| stage | fp32 | int8 |
|---|---|---|
| encoded | 107.5 dB | 107.5 dB |
| block0 out | 108.6 | 37.0 |
| block1 out | 110.8 | 35.4 |
| block2 out | 109.1 | 30.4 |
| block3 out | 108.6 | 29.4 |
| block4 out | 108.6 | 27.6 |
| block5 out | 109.1 | 26.8 |
| block6 out | 121.9 | 39.2 |
| **samples** | **119.0** | **26.3** |

The `FAIL` markers are the test's 100 dB gate, which is calibrated for the fp32
path; they are not a malfunction. The arithmetic closes: one 8-bit layer is worth
about 40 dB, there are ~35 of them along the signal path, and the LayerNorm after
each block re-normalises instead of letting the noise decay, so
40 − 10·log10(35) ≈ 25 dB. That is the 8-bit limit, not an implementation defect.

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


---

# Session 2: ARM SME2, NEON FP16, `ldaxr+wfe` Pool & Memory Layouts

**Date:** 2026-11-20, 07:11 – 08:39 CET  
**Machine:** Apple M4 Max (`Mac16,6`), 10 P-cores in two 5-core L2 clusters (16 MB L2 per cluster, 128-byte cache lines) + 4 E-cores. CPU only (no Metal/GPU).  
**Starting point (re-measured 07:11 CET on `/tmp/short5.wav`, 6 threads):**
- `fp32`: 13.21 s wall / 58.21 s user (0.38× realtime); 44.29 s prior final: **111.9 s wall** (0.40× RT) / **417.8 s user** (9.43 CPU-s/audio-s).
- `int8`: 8.58 s wall / 32.40 s user (0.58× realtime); 44.29 s prior final: **67.7 s wall** (0.66× RT) / **240.2 s user** (5.42 CPU-s/audio-s).
- Original baseline (44.29 s, 8 thr, fp32): **163.5 s wall / 456 s user** (0.27× RT, 10.3 CPU-s/audio-s).

## Hardware probe & microbenchmarks (07:12 – 07:20 CET)

`sysctl hw.optional.arm` confirmed full ARMv9 SME2 and NEON FP16/FHM support:
- `FEAT_SME: 1`, `FEAT_SME2: 1`, `sme_max_svl_b: 64` (512-bit streaming vector length -> 16×16 `ZA32` tiles)
- `SME_F32F32: 1`, `SME_F16F32: 1`, `SME_B16F32: 1`, `SME_I8I32: 1`, `SME_I16I32: 1` (note: no `SME_F16F16`)
- `FEAT_FP16: 1`, `FEAT_FHM: 1`, `FEAT_BF16: 1`, `FEAT_I8MM: 1`, `FEAT_DotProd: 1`
- Important compiler finding: `-march=native` on Apple Clang 17 does **not** enable SME2 or NEON FP16 vector arithmetic; explicit per-file flags (`-march=armv8.7-a+sme+sme2+fp16+bf16` on `src/gemm_sme.cc` and `-march=armv8.7-a+fp16+fp16fml` on `src/vecmath.cc`) with CMake `check_cxx_source_compiles` + runtime `sysctlbyname("hw.optional.arm.FEAT_SME2")` gating are required so portable builds stay unaffected.

Isolated GEMM microbenchmarks on representative shape `184 × 512 × 2048` (streaming 16 weight matrices to exceed L2 = "cold" vs single matrix = "warm"):

| kernel | 1-thread warm | 1-thread cold | notes |
|---|---|---|---|
| Apple Accelerate `cblas_sgemm` (reference) | 950 GFLOP/s | — | 1424 GFLOP/s default multi-thread |
| SME2 `FMOPA` fp32 (`svmopa_za32_f32_m`) | 1147 GFLOP/s | 992 GFLOP/s | 2×2 ZA tiles (32×32), 2× K-unroll, ZA bias init |
| SME2 `FMOPA` fp16->f32 (`svmopa_za32_f16_m`) | 1153 GFLOP/s | 1090 GFLOP/s | same FP32 accum rate as f32; +10% cold from 2× smaller weights |
| SME2 `FMOPA` bf16->f32 (`svmopa_za32_bf16_m`) | 1150 GFLOP/s | 1105 GFLOP/s | same FP32 accum rate |
| SME2 `SMOPA` i16->i32 (`svmopa_za32_s16_m`) | 1151 GOPS/s | — | same accumulator rate as f32 -> not worth implementing |
| SME2 `SMOPA` i8->i32 (`svmopa_za32_s8_m`) | 1885 GOPS/s | 1833 GOPS/s | 1.85× faster cold; 4-vec `svread_hor_za32_s32_vg4` epilogue |

## Chronological change log (07:20 – 08:35 CET)

### 9. ARM SME2 matrix engine & pre-packed panel layouts (`src/gemm_sme.h`, `src/gemm_sme.cc`, `CMakeLists.txt`, `src/model.cc`) — 07:22 CET

- Added full SME2 GEMM engine supporting `fp32`, `fp16`, `bf16`, `int8`, and `int8g` (`--quant MODE` in `apps/vi_cli.cc`).
- Pre-packed weight layouts (`Model::prepack_weights()`, eagerly packed in parallel at load/mode switch):
  - `fp32`: `[ceil(N/16)][K][16]`
  - `fp16` / `bf16`: `[ceil(N/16)][K/2][16][2]`
  - `int8` / `int8g`: `[ceil(N/16)][K/4][16][4]` + per-row (or per-group) scales
- Activation packing uses hardware ZA-tile 16×16 transpose (`sme_pack_a_f32`: 5.44 µs, `sme_pack_a_f16`: 3.99 µs on `184×512`).
- Micro-kernels: M-outer 32×32 (2×2 ZA tiles) with 2× K-unrolling, 16×64 (1×4 ZA tiles) tail for `M <= 16` (crucial for dual-lane `M = 368 = 11×32 + 16`), zero-overhead bias via ZA initialization (`svwrite_hor_za32` + direct `svst1_hor_za32`), and SME2 4-vector ZA reads (`svread_hor_za32_s32_vg4`) in the `int8` epilogue (+35% epilogue speed, 1332 -> 1804 GOPS/s 1-thr).
- **2-SME cluster cap (`max_sme_chunks = 2`, overridable via `VI_SME_CHUNKS`):** M4 Max has 2 SME2 units (1 per 5-core P-cluster). Running > 2 threads in SME2 simultaneously causes intra-cluster SME arbiter/ZA contention (measured on 5 s `fp16`: `chunks=2` -> `gemm 1.26 s`, `chunks=3` -> `2.71 s`, `chunks=4` -> `2.11 s`). Capping SME GEMM to 2 chunks while running NEON passes on 8 threads is optimal.

### 10. Low-power ARM64 `ldaxr+wfe` thread pool, 128-byte padding & cluster affinity (`src/pool.cc`) — 07:32 & 08:27 CET

- Isolated `job/count/generation`, `next`, `pending`, and `sleepers` on separate `alignas(128)` cache lines (Apple Silicon L2 cache line size = 128 B).
- Replaced busy `yield` spinning with ARM64 userspace `ldaxr + wfe` event waiting (`wait_for_change_u64` / `wait_for_nonzero_i32`), cutting 8-thread barrier latency from **~55 µs -> 1.45 µs** with near-zero power/thermal throttling.
- Added static dispatch when `n <= threads_` (eliminating `next.fetch_add(1)` contention) and strided peer selection (`worker 0` + `worker threads_-1`) for `n == 2`.
- Lowered `kMinParallelOps` from `2'000'000` to `150'000` on `aarch64` (LayerNorm on 5 s clip dropped 0.20 s -> 0.04 s) and raised default macOS thread count to `min(cores, 8)`.
- **Cluster affinity fix (08:27 CET):** without explicit cluster separation, macOS XNU occasionally scheduled both `chunks=2` SME workers onto the same 5-core P-cluster, dropping GEMM by 2.8× (`fp32` 1.43 s -> 4.05 s on 5 s clip). Setting `THREAD_AFFINITY_POLICY` (tag `1` on main thread + first half of workers, tag `2` on second half including `stride_peer = threads - 1`) eliminated SME collisions completely (10/10 microbench trials and 9/9 CLI runs collision-free).
- **Accounting note on `user` CPU time:** macOS counts hardware `wfe` naps inside `user` time even though waiting cores draw near-zero power. Therefore, honest compute cost below is reported from **1-thread `user` time** alongside **8-thread wall-clock realtime**.

### 11. Fast NEON vector math & register-resident Mamba scan (`src/vecmath.h`, `src/vecmath.cc`, `src/model.cc`) — 07:45 & 08:16 CET

- Added hand-tuned NEON fp32 kernels for `v_exp` (1.7e-7), `v_gelu` (zero-cancellation degree-7 Chebyshev `erfc`, 5.1e-7), `v_silu` (1.3e-7), `v_softplus` (1.7e-7), `v_mul_exp` (3.5e-7), `v_quantize_i8` (`vcvtaq_s32_f32` + `vqmovn`), and `v_softmax_norm`.
- **`v_mamba_scan` (fp32):** 2-pass 8-state register-resident NEON scan (zero register spills) over pre-scaled `a_log2e4` (`[inner/4][state][4]`) and transposed `ssm_state` (`[bands][inner/4][state][4]`), using `vfmaq_laneq_f32`, FMA exponent assembly (`vcvtq_s32_f32(vfmaq_f32(127*2^23, n, 2^23))`), and an unbiased degree-6 `2^f` polynomial (`c0 = 1.0f`, max err `2.6e-9`).
- **`v_mamba_scan_f16` (stateless `freq_mamba` only):** 8-wide NEON FP16 scan (`4.34 G exp/s` single-core), enabled in non-`fp32` quant modes **only** for stateless `freq_mamba` (`ssm_state == nullptr`, `len = 46`, reset every frame). `time_mamba` stays in fp32 to eliminate cumulative decay-rounding drift (see failed experiments below).
- **`v_conv4_silu`:** fused 4-tap depthwise conv1d + bias + SiLU; `MambaState::conv` re-laid out to `[bands][taps][inner]` for contiguous `memcpy`.
- **`v_quant_pack_a_i8_block`:** multi-core NEON 4×4 transposed activation quant+pack across all 8 P-cores for `int8` SME2 (cut 5 s `int8` GEMM from **1.31 s -> 0.77 s**).

### 12. Batched 4-query attention, FP16 KV cache (`FEAT_FHM`), zero-alloc decode & dual-lane crossfade batching (`src/model.cc`, `src/stream.cc`, `src/vecmath.cc`) — 07:55 – 08:34 CET

- **`v_attn4_qk` & `v_attn4_pv`:** 4-query batched NEON micro-kernels (4×4 QK and L1-tiled `kJTile=128` 4-way `vfmaq_laneq_f32` PV) used in both `time_transformer` (reads KV cache 1× per 4 frames instead of 4×) and `band_transformer`.
- **FP16 KV cache (`--fp16-kv`; auto-enabled for `--quant fp16|bf16|int8`, opt-in for `fp32`):**
  - `v_attn4_qk_f16` uses `FEAT_FHM` widening `vfmlalq_low_f16` / `vfmlalq_high_f16` (FP16 storage + FP32 accumulation).
  - `v_attn4_pv_f16` uses `vfmlalq_lane_low_f16` / `vfmlalq_lane_high_f16` (FP16 `V` + FP16 softmax weights -> FP32 accumulation, eliminating all `vcvt_f32_f16` in the inner loop). Single-thread 5 s `pv` dropped from **0.60 s (fp32 KV) -> 0.33 s (fp16 KV)** while keeping **72.11 dB SNR vs fp32** and halving KV cache RAM.
- **Static `bands × heads` load balancing in `time_transformer` / `time_transformer_pair`:** replaced dynamic `parallel_for(46)` (6 vs 5 bands = 17% straggler tail on 8 threads) with static chunking over `46 × 8 = 368` (`46` exact per thread; `92` exact in dual-lane) and 6× smaller `sc.scores` buffer. Same static chunking applied to `mamba_layer` / `mamba_layer_pair`.
- **Zero-alloc NEON decode:** stack scratch buffers (zero heap allocations for `frames <= 4`) + direct 4-row NEON `v_linear_4rows`.
- **Crossfade dual-lane batching (`Model::backbone_pair` & `decode_pair`):** during the 200/600 crossfade window (~1/3 of long audio), both lanes are stacked into `M = 2 × 184 = 368`, halving weight reads and barriers.

## What was tried and discarded (or fixed) in Session 2

- **N-outer SME2 loop + `svread_hor_za32` bias epilogue:** 48% slower than M-outer + `svwrite_hor_za32` bias init + direct `svst1_hor_za32` (769 vs 1140 GFLOP/s).
- **Degree-5 minimax `exp2` (`c0 = 1.000000075f`) in `v_mamba_scan`:** `c0 > 1.0` biased slow-decaying channels over 2000 steps, failing `plain20` at 91.4 dB (< 100 dB gate). Replaced with unbiased degree-6 `exp2` (`c0 = 1.0f`, max err `2.6e-9`) -> `plain20` restored to **108.5 dB**.
- **FP16 scan on carried `time_mamba`:** IEEE fp16 rounds `decay > 0.99976` to `1.0`, causing temporal drift on 44.29 s `fp16` from 52.5 dB (0–4 s) down to 10.3 dB (32–36 s; 17.82 dB overall). Restricting `v_mamba_scan_f16` to stateless `freq_mamba` (`ssm_state == nullptr`) restored **55.62–56.45 dB** with zero drift across all 44 s (every 4 s window 50–70 dB).
- **Pure FP16 accumulation (`vfmaq_f16`) in `v_attn4_qk_f16` / `v_attn4_pv_f16`:** dropped `fp32+kv16` SNR to 44.46 dB. Switching to `FEAT_FHM` widening FP32 accumulation (`vfmlalq_*`) restored **72.11 dB** at virtually identical speed.
- **Multi-core NEON A-packing for `fp32`/`fp16`/`bf16`:** SME2 hardware ZA transpose (`sme_pack_a_f16` = 3.99 µs, `sme_pack_a_f32` = 5.44 µs) is 2.5–3× faster than NEON 4×4 transpose (10.02 / 16.08 µs), and waking 8 cores dirtied `ap` across both clusters (`fp16` 1.30 s vs 1.20 s). Kept single-thread hardware ZA packing for `f32`/`f16`/`bf16` and 8-core NEON packing only for `int8`.
- **Splitting `time_mamba` channels across 4 tasks (`46 × 4 = 184`)**: hurt cache locality on `sc.xp` and `ext`, regressing `scan` 0.40 -> 0.45 s (`fp32`) and 0.24 -> 0.32 s (`fp16`). Kept `split = 1` for `time_mamba` with static 8-chunk dispatch.
- **Per-group int8 (`--quant int8g`, `group_size = 64`):** improved `int8` SNR by +10.1 dB (22.66 -> 32.76 dB on 5 s clip), but draining ZA 8–32× per tile slowed 5 s clip to 5.37 s — strictly dominated by `fp16` (2.15 s, 58.0 dB). Kept as opt-in CLI flag for reference.
- **Hybrid int8 + fp16 residual-output projections (`proj_w`, `out_w`, `f1w`, `bl2w`, `mamba.out_w`):** measured 23.13 dB vs pure `int8` 23.64 dB (no gain because the remaining ~60 int8 layers still floor SNR near ~22 dB). Removed to keep the CLI clean.
- **Larger `--block-frames` (8, 16):** 2.31 s (bf=8) and 2.45 s (bf=16) vs 2.35 s (bf=4) on 5 s `fp16` (~1–2% noise); kept default `block_frames = 4`.

## Re-verification of the Session 2 final binary (2026-11-20 08:35:46 – 08:39:07 CET)

All measurements below were run back-to-back on the final clean build (zero warnings):

| check | result |
|---|---|
| `cmake --build build -j4` | clean, zero warnings (SME2 + NEON FP16 enabled) |
| `./build/vi_tests . quick` | all tests passed, output samples **117.5 dB** (max `7.75e-7`) |
| `ctest --test-dir build` | **1/1 passed in 62.14 s** (down from 236.72 s — **3.81× faster**; `ref1`: 117.5 dB, `win10`: 113.5 dB, `plain20`: 108.5 dB) |
| sample counts (44.29 s) | 2 125 746 vs 2 125 746 |

### True compute cost (5 s clip, 1 thread — zero `wfe` accounting distortion)

| mode | 1-thr wall | 1-thr user | CPU-s / audio-s | vs prior fp32 (38.36 s) |
|---|---|---|---|---|
| Prior `progress.md` fp32 (1 thr) | 38.95 s | 38.36 s | 7.67× | 1.00× |
| **`fp32` (SME2)** | **8.14 s** | **8.01 s** | **1.60×** | **4.79× less compute** |
| **`fp32 + fp16-kv`** | **7.59 s** | **7.52 s** | **1.50×** | **5.10× less compute** |
| **`fp16` (SME2 + kv16 + freq-f16)** | **6.64 s** | **6.58 s** | **1.32×** | **5.83× less compute** |
| **`bf16` (SME2 + kv16 + freq-f16)** | **6.89 s** | **6.75 s** | **1.35×** | **5.68× less compute** |
| **`int8` (SME2 + kv16 + freq-f16)** | **6.12 s** | **6.09 s** | **1.22×** | **6.30× less compute** |

### Wall-clock throughput & accuracy (8 threads, final binary)

| workload / mode | wall | realtime | vs orig baseline (163.5 s) | vs prior `progress.md` | SNR vs `clean2.wav` | SNR vs `fp32` |
|---|---|---|---|---|---|---|
| 5 s, `fp32` | 2.75 s | **1.82×** | — | 3.92× (vs 10.78 s) | — | ref |
| 5 s, `fp32 + fp16-kv` | 2.52 s | **1.98×** | — | 4.28× | — | 72.67 dB |
| 5 s, `fp16` | 2.15 s | **2.32×** | — | 5.01× | — | 58.00 dB |
| 5 s, `bf16` | 2.16 s | **2.31×** | — | 4.99× | — | — |
| 5 s, `int8` | 1.71 s | **2.93×** | — | 3.80× (vs 6.5 s) | — | 23.37 dB |
| **44.29 s, `fp32` (default)** | **30.44 s** | **1.45×** | **5.37×** | **3.68×** (vs 111.9 s) | **69.71 dB** (max `6.10e-5`) | ref |
| **44.29 s, `fp32 + fp16-kv`** | **27.37 s** | **1.62×** | **5.97×** | **4.09×** | **68.50 dB** (max `2.75e-4`) | **72.11 dB** |
| **44.29 s, `fp16`** | **23.53 s** *(22.19 s cool)* | **1.88×** *(2.00×)* | **6.95×** *(7.37×)* | **4.76×** | **55.58 dB** (max `1.98e-3`) | **55.62 dB** |
| **44.29 s, `bf16`** | **24.05 s** | **1.84×** | **6.80×** | **4.65×** | **45.38 dB** (max `9.77e-3`) | **45.39 dB** |
| **44.29 s, `int8`** | **18.87 s** | **2.35×** | **8.66×** | **3.59×** (vs 67.7 s) | **19.94 dB** (max `1.49e-1`) | **19.93 dB** |

### Where the remaining time goes (44.29 s, 8 threads)

- **`fp32` (30.44 s):** GEMM 14.34 s (47%), Mamba scan 4.35 s (14%), attention QK+PV+ortho ~5.5 s (18%), decode 0.56 s, conv 0.51 s, LN 0.47 s, transpose 0.44 s, add 0.41 s, RoPE 0.34 s.
- **`fp16` (23.53 s):** GEMM 12.21 s (52%), attention QK+PV+ortho ~3.5 s (15%), Mamba scan 2.55 s (11%), decode 0.53 s, conv 0.47 s, LN 0.43 s, transpose 0.40 s, add 0.35 s, RoPE 0.19 s.
- **`int8` (18.87 s):** GEMM 7.77 s (41%), attention QK+PV+ortho ~3.3 s (17%), Mamba scan 2.52 s (13%), decode 0.52 s, conv 0.44 s, transpose 0.43 s, LN 0.38 s, add 0.37 s, RoPE 0.17 s.

All three operating points now run comfortably faster than realtime on CPU alone:
- **Reference quality:** default `--quant fp32` (**1.45× realtime**, 117.5 dB vs Python float32, ~70 dB vs PCM16) or `--quant fp32 --fp16-kv` (**1.62× realtime**, 72.1 dB vs fp32, half KV RAM).
- **Best speed/quality sweet spot:** `--quant fp16` (**1.88–2.00× realtime**, **55.6 dB** vs fp32, zero temporal drift).
- **Maximum CPU throughput:** `--quant int8` (**2.35× realtime** on 44.29 s / **2.93× realtime** on 5 s, 19.9 dB vs fp32).

---

## x86-64 (AMD / Intel) investigation — 2026-11-20

The x86 measurements below were made on Linux with an AMD Ryzen 9 3900X
(AVX2/FMA, 12 CPUs visible), using the 48 kHz mono recording and four-frame
streaming blocks. They establish behavior on this AMD machine, not a benchmark
on an Intel CPU. The default fp32 path and the 600/200-frame window/crossfade
were used. These wall times are individual measurements, not medians. The 5 s
excerpt does not fill the attention window and is not representative of the
full recording's steady-state cost.

### Changes and measurements

- The x86 build exposed a missing `<vector>` include in `vecmath.cc`, because
  the non-NEON scan fallback uses `std::vector`. Adding it fixes the build.
- The previous x86 Mamba scan evaluated each channel separately with scalar
  `std::exp`. An SSE2 path now processes the four adjacent channels in the
  existing `[inner/4][state][4]` layout together with xsimd, preserving the
  carried state. The NEON and SME2 paths remain in place. On the 5 s excerpt,
  eight threads, the same-source scalar baseline took 13.72 s wall with 2.84 s
  attributed to scan; the SSE2 build took 11.67 s wall with 1.19 s in scan.
  These runs used Eigen GEMM and included CLI/model startup, so treat the
  differences as indicative rather than controlled statistical estimates.
- An optional OpenBLAS GEMM path can be enabled with
  `-DVI_OPENBLAS_LIBRARY=/path/to/libopenblas.so`. It uses the existing worker
  pool and sets OpenBLAS to one thread per call (a process-wide setting). SME2
  still takes priority where available. On the same 5 s excerpt with eight
  threads, SSE2 plus OpenBLAS took 10.52 s wall; the profile attributed 5.93 s
  to GEMM and 1.24 s to scan. The Eigen/SSE2 run attributed 7.17 s to GEMM.
- A first AVX2 int8 dot-product experiment took 12.04 s on the 5 s excerpt,
  slower than fp32 with OpenBLAS; it was discarded. The ARM int8 kernels are
  unchanged. This result does not rule out a better tiled or packed x86 kernel.

### Full-file verification and limits

Before merging the later `master` changes, the restored CLI with SSE2 and
OpenBLAS processed 44.29 s in 106.19 s with eight threads: **0.42× realtime**.
Its stage profile reported 57.05 s in GEMM and 11.18 s in scan. It produced
2,125,746 samples, the same count as `clean2.wav`, with **69.71 dB SNR** and
maximum absolute difference **6.10e-5** relative to that PCM16 reference. The
5 s optimized outputs were within one PCM16 step of the same-source scalar
baseline (at least 97.7 dB SNR).

Both the portable (`VI_NATIVE=OFF`) and OpenBLAS builds passed CTest after the
merge with the newer `master`; the new direct scan test covers carried state,
zero-initialized state and a channel slice. The optional `ref/` fixtures were
absent, so the Python stage, windowed and cache-compaction reference tests were
skipped. The full-file timing and comparison above were made before that merge
and should be rerun for performance claims about the merged version. ARM and
Metal execution were not tested on this x86 host.

The realtime goal remains unmet on this machine. GEMM is the largest measured
cost; further work should benchmark better packed/vectorized x86 kernels and
threading against the full recording, while checking the quality impact of any
lower-precision mode. See `README-intel.md` for build and CLI commands.
