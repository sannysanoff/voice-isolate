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

## Ryzen AVX2 int8 follow-up

Starting revision `3e49c57`, Ryzen 9 3900X, 12 visible cores, GCC 12.3,
Release `-O3 -march=native -fno-math-errno`. The timed builds used Eigen
3.4.0, xsimd 13.2.0 and the same optional OpenBLAS 0.3.15 as the handoff.
All runs used four-frame blocks, a 600-frame window and a 200-frame crossfade.

The AVX2 dense kernel packs 16 columns by four depth values in each aligned
64-byte panel. A 4-by-16 register tile shares its weights across four rows.
Weights are cached on first use. Each worker quantizes disjoint activation rows
once and corrects the unsigned activation offset in the integer accumulator.
A per-output weight scale bounds same-sign byte pairs so the int16 intermediate
cannot saturate. Eight-wide AVX2/FMA scan, SIMD attention scores, depthwise
convolution and the four-row decoder also use the existing fp32 model layout.

| Full recording, 8 threads, one run each | CLI processing (s) | User CPU (s) | GEMM (s) | Scan (s) |
|---|---:|---:|---:|---:|
| Starting revision, fp32 | 113.31 | 859.65 | 59.99 | 12.53 |
| First packed int8 build with vector scan and attention scores | 116.33 | 883.49 | 61.71 | 6.33 |

The int8 run produced 2,125,746 samples. Against the same-source fp32 PCM16
output it measured 19.67 dB SNR and 0.12793 peak absolute error (full scale
1). SNR ranged from 11.49 to 27.45 dB across six-second windows including
the final partial window. Sorting 100 ms segments by fp32 RMS, the quietest
quartile had -54.02 dBFS RMS error and the loudest had -44.93 dBFS. The CLI
peak-normalized both files. The fp32 baseline measured 69.71 dB against the
supplied PCM16 reference.

In the first five-second experiment with packed GEMM and the original scan,
4/6/8 threads took 14.59/11.92/9.50 s of processing respectively. A later
column-split version with a separate parallel quantization pass took 9.26
and 9.87 s at eight threads. The retained row split needs one dispatch per
dense layer. Small matrix tests of inline assembly and alternate row tile
sizes gave inconsistent rankings across runs, so the kernel uses intrinsics.

Subsequent sustained runs hit severe host throttling. Linux k10temp reported
95.2°C with all reported core clocks below 1 GHz, reaching 103.5°C after the
benchmarks stopped. The later trials cannot establish a sustained speedup.
The full int8 quality and timing above precede the final decoder/convolution
and attention-value loop changes. The Python reference fixtures were absent.

## x86 int8 optimization — Ryzen 7 7435HS (2026-11-20, 8 threads)

Machine: AMD Ryzen 7 7435HS, 8 cores / 16 SMT threads, AVX2+FMA, **no
AVX-512 / VNNI / fp16**, ~4.2 GHz under load, no thermal throttling during
runs. GCC 13.3, Release `-O3 -march=native -fno-math-errno`. Source at this
revision; full 44.29 s WAV, `--quant int8 --threads 8`, window 600 / crossfade
200. Times are processing time (model load excluded).

| Step | Processing | Realtime | int8 SNR vs matching fp32 |
|---|---:|---:|---:|
| Starting build (bf4, pool.cc spin tweak, no inlining) | 51.5 s | 0.86x | 23.32 dB |
| `tile()` forced `always_inline` | 48.8 s | 0.91x | — |
| + column-major loop nesting + one-thread-per-core affinity | 44.2 s | 1.00x | 23.32 dB |
| + default `--block-frames 6` | 41.1-42.4 s | **1.04-1.08x** | 23.40 dB |

Diagnosis: `nm` showed `tile<4>` emitted as a standalone symbol (GCC refused to
inline the `target("avx2")` function into `gemm_avx2_tiles`), which cost ~37% of
kernel throughput (187 -> 253 GOPS on 184x512x1536). AVX2 k-unrolling, 8-row
split tiles, AVX2 attention QK tiling, LTO, and bias-template specialization
were all tried and gave no gain. Affinity (one worker per physical core) saved
~2 s; `VI_NO_AFFINITY=1` disables it. Block 8 is faster still (40.0 s, 1.11x)
but int8 SNR drops to 21.6 dB, so block 6 was chosen as the default.

Output sample count is unchanged (2,125,746). int8 remains a low-precision mode
(~23 dB against fp32); fp32 is the reference-quality path. ARM/Metal backends
were not exercised on this host.

### Final verification of the 7435HS binary

Re-run after the last refactor (bias epilogue moved out of the lambda so the
portable build compiles), machine otherwise idle:

| check | result |
|---|---|
| `cmake --build build-opt -j8` (native) | clean |
| `cmake --build build-portable2 -j8` (`VI_NATIVE=OFF`) | clean |
| `ctest --test-dir build-opt --output-on-failure` | 1/1 passed |
| `ctest --test-dir build-portable2 --output-on-failure` | 1/1 passed |
| `nm` on `gemm_avx2.cc.o` | no standalone `tile<4>` symbol (inlined) |
| portable CLI, 5 s excerpt, int8, 8 threads | 4.68 s incl. startup |
| 44.29 s, int8, 8 threads, default bf6 (two runs) | **41.06 / 40.91 s processing = 1.08x** |
| profiled run | stages 38.64 of 41.35 s; GEMM 16.05, scan 2.93 s |
| sample count | 2,125,746 (matches `clean2.wav`) |
| int8 vs matching fp32 bf6 output | 23.40 dB, max 0.09048 |
| int8 vs `clean2.wav` PCM16 | 23.40 dB, max 0.09051 |

Outputs kept at `/tmp/opencode/F1.wav`, `/tmp/opencode/F2.wav` and
`/tmp/opencode/FINAL-int8.wav`. The `ref/` fixtures were still absent, so the
Python stage/windowed/cache-compaction reference checks are skipped rather than
passed. The handoff document now has a matching final stage with the changed
bf6 default and reproduction command.

---

## x86 int8 towards 1.9x — session 2026-11-20 (Ryzen 7 7435HS, same host)

Goal: 1.9x realtime (44.29 s audio in <= 23.3 s) on this CPU with the int8
path. Starting point re-measured: 41.02 s processing, 8 threads, bf6 (1.08x).
fp32 stays the reference path; int8 is labelled low precision.

Environment caveats (all runs below): `powersave` governor (amd-pstate-epp,
`balance_performance`), ~2.9 GHz observed under 8-core AVX2 load vs 4.55 GHz
max; no sudo so the governor cannot be changed. Box is shared (Minecraft JVM,
Brave, Telegram); loadavg seen 0.84–6.70. `user` CPU time is the stable metric;
wall moves with ambient load and clocks. Single-thread AVX2 int8 kernel now
benches 169–259 GOPS shape-dependent (vs 253 on 184x512x1536 at full boost),
i.e. clocks are derated ~0.67x in this environment.

### Diagnosis added (kept in tree)

- Worker-side atomic accumulators `attn_qk_w / attn_soft_w / attn_pv_w /
  attn_ortho_w / attn_store_w` (flushed once per pool chunk) + `elem` timer
  for GELU/softplus/SiLU + `gemm_flops` and `gemm_fp32_flops` counters.
- Full-file finding: attn math ~31 s CPU (QK 14.4 + PV 11.4 + softmax 4–5),
  GEMM 16 s wall at ~1380–1425 GFLOPS aggregate (99.9% int8, fp32 fallback
  only 0.1% of FLOPs), scan 2.9, elem 1.3, LN 1.1, transpose 0.6. GEMM runs
  at ~85% of the derated single-thread peak, so the ceiling from here is
  mostly clocks, not kernel efficiency.

### Changes (all measured on 5 s excerpt + full 44.29 s file, 8 threads, bf6)

1. **Remainder batching**: the `frames % 4` tail (2 frames at bf6) ran as
   single queries reloading the whole 600-key window each. Pad to a 4-batch
   by repeating the last frame (padded outputs rewrite it identically; only
   `rem` outputs go through the non-idempotent ortho step). Pair path too.
   Band tail (window 46) deliberately left as singles: negligible traffic,
   and score-buffer aliasing would break softmax.
2. **x86 int8 KV cache** (`--quant int8` auto-enables on AVX2+FMA, explicit
   `--int8-kv` flag; `--fp16-kv` still forces fp32 KV on x86 for now):
   per-token symmetric int8 for K and V + one fp32 scale per slot, fp32
   accumulation in AVX2 QK/PV kernels with on-the-fly dequant; fp32 KV path
   untouched. Insert quantizes after RoPE; cache-trim path dequantizes,
   inverse-rotates and requantizes (cold). 5 s clip: QK 1.77->0.89 s CPU,
   PV 1.94->0.70 s CPU (PV includes a prescale fold of vscale into weights,
   ~+7%). Quality on 5 s: int8+fp32KV vs fp32 = 24.82 dB; int8+int8KV vs
   fp32 = 24.51 dB — KV quant costs **0.3 dB** total, far below the 23 dB
   GEMM floor. Per-token int8 suffices; no Hadamard needed at int8.
3. **Column-split AVX2 GEMM**: quantize all rows once (row-parallel), then one
   worker per 16-column tile subset (weights stream 1x instead of 8x).
   Fallback to the old row split when `n/16 < threads`. Neutral on 5 s
   (compute-bound shapes), kept for L3 pressure on full file.
4. **x86 `kMinParallelOps` 2M -> 100k** (barriers ~1-2 us on hot spinning
   workers): LayerNorm 0.11->0.02 s on 5 s clip.
5. **Parallel strided moves**: `to_frame_major`/`to_band_major` over bands,
   mamba reversal + cat over frames (single + pair paths), scalar
   `sc.trans += feat` -> `add_par`.
6. **Decode tails**: 1-3 leftover rows via padded `v_attn4_qk` + bias
   (`linear_tail4`) instead of Eigen. Decode 0.24->0.18 s on 5 s clip.

### Rejected (measured, reverted or not taken)

- Scan as 4 passes of 4 states (register pressure): **slower** (scan
  0.32->0.40 s on 5 s) — L1-resident spills are nearly free, extra passes
  add real work. Reverted.
- Integer PV via `vpmaddubsw`: wrong reduction — madd/maddubs reduce across
  bytes (group sums), but PV needs per-element outputs. Caught before
  running; float dequant path kept.
- Masked band tail: score-buffer aliasing breaks softmax (not idempotent);
  window-46 traffic negligible anyway. Reverted.
- fp16 KV on x86: not needed — int8 KV is strictly better here and quality
  holds; `--fp16-kv` on x86 keeps meaning fp32 KV (documented above).
- Integer QK dot: needs quantized Q + offset correction via per-key sums
  (~25% overhead for ~15% uop win) — not worth it.
- Larger blocks: bf12 38.1 s vs 41.0 s on the old code but int8 SNR
  23.4->21.6 dB at bf8; not taken (kept bf6 default + comparability).
- LTO, AVX2 k-unrolling, 8-row tiles, 4/6/7 threads: no gain (prior session,
  still true).

### Results so far (8 threads, bf6, window 600 / crossfade 200)

| build step (full 44.29 s, int8) | processing | realtime |
|---|---|---:|
| session start | 41.02 s | 1.08x |
| + remainder batch + int8 KV + colsplit + PV fold | 33.72 s | 1.31x |
| + threshold + parallel moves + decode tails | **31.71 s** | **1.40x** |

5 s excerpt int8: 4.19 s -> 3.22 s (1.55x). `vi_tests quick` passes, `ctest`
passes; fp32 path outputs unchanged in character (full fp32 re-verify pending
at the end). int8 quality MUST still be re-verified full-file vs matching
fp32 and vs `clean2.wav` (5 s: 25.0 dB vs fp32).

One full-file run at 52.25 s was discarded as invalid: a foreground parameter
sweep ran concurrently and oversubscribed the cores (user 255 s vs 235 s,
wall/user ratio collapsed). Never benchmark concurrently on this box.

Remaining gap: 31.7 s -> 23.3 s (8.4 s). GEMM is at ~85-95% of the derated
peak so the rest must come from attention math, scan, decode, data movement
and quiet-machine clocks. Next: band-QK/PV costs, `bf` re-check with current
code + quality, quiet-window full-file runs.

### Session continued (2026-11-20, later)

Determinism check: int8 bf6 run twice -> **341.57 dB, maxabs 0.0**
(bit-identical). No pool race; workers write disjoint outputs and shared
accumulators are order-independent. The earlier masked-batching
double-compute bug (singles loop left alongside the masked batch) was found
and removed; it had inflated one full-file run to 41.63 s.

bf quality, current code: int8 bf8 vs matching fp32 bf8 = **23.64 dB**
(the old 21.6 dB was a mismatched-config artifact), fp32 bf6 vs fp32 bf8 =
98.16 dB. bf8 is quality-neutral vs bf6 (23.64 vs 23.16 dB) and ~12% faster
on full file (30.60 s vs 34.23 s). int8 bf6 vs int8 bf8 differ by ~25 dB:
expected cascade amplification — tiny fp32 summation-order seeds (Eigen
blocking by M, batching) flip int8 rounding thresholds through ~35
quantized layers. Same phenomenon explains why any non-bit-identical change
in int8 mode lands at ~23-26 dB inter-variant SNR while staying ~23-24 dB
vs fp32. Not a bug (determinism proven); document, don't chase.

Power finding: with `performance` governor + EPP performance (enabled via
sudo by the operator), single-thread AVX2 int8 went 169 -> **254-256 GOPS**
(the handoff peak), but full-file wall did not move (30.60 -> 30.55 s).
8-core AVX2 is power-capped (~2.9 GHz sustained vs 4.2+ single-core); the
governor cannot fix it. GEMM in-situ 1355-1425 GFLOPS aggregate = ~95% of
the 8-core power ceiling, i.e. at-peak. Clocks are out; only less work
counts now.

x86 int8 KV quality (5 s): int8+fp32KV vs fp32 = 24.82 dB; int8+int8KV vs
fp32 = 24.51 dB (KV costs 0.3 dB). Per-token int8 suffices; no Hadamard.

GELU/softplus epilogue fusion into the AVX2 int8 GEMM (raw AVX2 erf via
A&S 7.1.26 + exp2 reuse, threshold-20 softplus via degree-13 log1p series;
portable-safe, no xsimd which needs -mavx2 for the explicit arch):
- GELU fusion is quality-clean (25.77 dB vs 25.60 unfused on 5 s, elem pass
  eliminated). Kept.
- Softplus fusion cost ~2 dB on 5 s (24.57 dB; finer log1p recovered half of
  it). Root cause: 1e-6..1e-8 delta perturbations accumulate in time-mamba
  scan state over thousands of steps, while GELU perturbations (no carry)
  do not. Dropped (saves only ~0.15 s anyway); GELU-only fusion kept.

Also done: `kMinParallelOps` x86 2M->100k (LN 0.11->0.02 s on 5 s),
parallel to_frame/to_band_major + mamba reversal/cat (single+pair),
`sc.trans+=feat` -> add_par, decode tails via padded `v_attn4_qk`
(decode 0.24->0.15 s on 5 s), int8 KV P prescale fold (~+7% PV),
scratch reserve-no-zero pattern (q8/q8s/bscores/bqkv/mamba_ext; removed
resizes of the never-read decay/h slabs), madvise(MADV_HUGEPAGE) on the
weight blob + AVX2 panels (THP `madvise` mode), column-split AVX2 GEMM with
row-split fallback for n/16 < threads, fp32-fallback FLOP counter (only
0.1% of FLOPs — small layers correctly stay fp32), `linear_par` now returns
whether the epilogue fused (fp32/SME paths return false, elementwise runs).

Rejected since last entry: scan as 4x4-state passes (slower: more passes
over L1-hot data beat the spill saving, reverted), integer PV via maddubs
(wrong reduction for per-element outputs, caught before running), integer
QK dot (offset-correction overhead negates the uop win), band int8 QK/PV
(quantize overhead dwarfs the tiny n=46 compute), 2D GEMM tiling (marginal
vs colsplit at L3-resident weights), softplus fusion (above), prefetch/unroll
in QK/PV (streams are HW-prefetched; kernel is uop-bound, not latency-bound).
`perf` unblocked via sysctl (paranoid=1) for future counter checks;
valgrind/AMDuProf absent, THP is `madvise` mode.

Epilogue fusion follow-up: GELU-only fusion verified quality-clean
(25.77 dB vs 25.60 unfused on 5 s, elem pass eliminated) but full-file
(30.90 s vs 30.67) showed net-zero/negative — the hand-rolled AVX2 erf
(~25 uops/8) is slower than xsimd's erf in the separate pass, so moving the
ALU into the GEMM critical path outweighed the saved memory passes. Fully
reverted (bit-identical to pre-fusion: 338 dB on 5 s). Lesson: the elem pass
is ALU-bound (xsimd erf/exp), not memory-bound; fusion only helps if the
epilogue ALU matches xsimd speed.

Power-wall conclusion (multiple independent confirmations): single-thread
AVX2 int8 254-256 GOPS (4.2 GHz boost) vs in-situ 169/core = exactly
254x2.9/4.2 — in-situ runs at the 8-core AVX power cap (~2.9 GHz), with zero
parallel scaling loss. Attention matches the same way (61 -> 42 GFLOPS).
`performance` governor + EPP performance changed nothing vs `powersave`
(30.60 -> 30.55 s): the limit is PPT, not policy. GEMM 1355-1425 GFLOPS
aggregate is ~95% of the 8-core ceiling. No code change can exceed it;
only less work or more power/clocks.

## What worked (full-file int8, bf8 unless noted; baseline 41.02 s / 1.08x)

- x86 int8 KV cache, per-token symmetric + fp32 accum (`--quant int8`
  auto-enables on AVX2+FMA): QK 2.0x, PV 2.6x on the math; full file
  ~41 -> ~34 s with the rest. Costs 0.3 dB total (24.82 -> 24.51 dB vs
  fp32 on 5 s). With prescale fold in PV (+~7%).
- Masked remainder batching (4+2 pattern): -33% KV traffic on bf6 tails.
- Column-split AVX2 GEMM (weights 1x instead of 8x): neutral-to-positive;
  kept for L3 pressure on full file. Falls back to row split for n/16 < threads.
- Block size 8 as default (was 6): +12% vs bf6 (30.6 vs 34.2 s), quality
  neutral-to-better (23.64 vs 23.16 dB vs matching fp32). bf12 (32.8 s) and
  bf16 (33.9 s) are SLOWER — L3 thrashing past M~736 (pair rows2), so the
  optimum is confirmed from both sides, not just picked.
- x86 `kMinParallelOps` 2M -> 100k + parallel transpose/mamba-scalar/cat
  loops + `sc.trans+=feat` -> add_par: LN 1.06->0.23 s, transpose/add down.
- Scratch reserve-no-zero pattern (q8/q8s/bscores/bqkv/mamba_ext); removed
  resizes of the never-read decay/h slabs; madvise(HUGEPAGE) on weight blob
  + AVX2 panels.
- Decode tails via padded `v_attn4_qk` (decode 0.24->0.15 s on 5 s).
- Worker-side attn breakdown timers + FLOP counters (diagnostic, kept;
  zero cost with --profile off).
- Determinism proven (341.57 dB rerun); fp32 reference intact
  (69.75 dB vs clean2.wav, was 69.71).

## What did NOT work (measured, reverted or not taken)

- GELU/softplus epilogue fusion into the AVX2 GEMM: net-zero/negative
  (30.67 -> 30.90 s). Hand-rolled AVX2 erf (~25 uops/8) is slower than
  xsimd's erf in the separate pass; moving ALU into the GEMM critical path
  outweighed saved memory passes. Fully reverted (bit-identical: 338 dB).
  Softplus fusion additionally showed ~2 dB quality cost (1e-6 seed
  accumulates in scan state); GELU-only was clean but still net-zero. Lesson:
  the elem pass is ALU-bound, not memory-bound.
- Scan as 4 passes of 4 states: slower (more passes over L1-hot data beat
  the spill saving). L1-resident spills are nearly free.
- Integer PV via `maddubs`: wrong reduction (group sums, but PV needs
  per-element outputs). Caught before running. Integer QK dot: offset
  correction overhead negates the uop win.
- Band int8 QK/PV: quantize overhead dwarfs the n=46 compute.
- 2D GEMM tiling, QK/PV prefetch+unroll: streams are HW-prefetched, kernel
  is uop-bound at peak already.
- bf12/bf16 defaults: slower (L3 thrash), rejected; bf8 stays.
- Masked band tail: score-buffer aliasing breaks softmax; negligible win.
- Hadamard rotation for KV: unnecessary — plain per-token int8 holds
  quality (0.3 dB cost).
- `-march=native` PGO/fast-math, LTO, k-unrolling, 8-row tiles, 4/6/7
  threads: no gain (prior sessions + re-verified where cheap).
- Governor/performance+EPP: no effect under 8-core AVX (power-capped).

Current best (this session, code as of now): 30.55-30.67 s full-file int8
bf8 = **1.44-1.45x realtime** (user CPU ~229 s). Remaining gap to 1.9x
(23.3 s) is ~7.3 s against a proven power wall: GEMM at ~95% of the 8-core
ceiling, attention/scan/elem at their uop peaks. Only less work (model or
measurement config) or more power/clocks can close it — both are decisions,
not code.

## Pool affinity bug (found 2026-11-20, fixed)

`--threads 6` ran at ~1 core (full file 576 s, 5 s clip 63 s, user≈wall;
attention sub-kernels looked normal, everything pool-parallel exploded).
Root cause, proven by strace: the default pool is lazily built as Pool(8)
and pins main+workers to cpus [0,2,...,14]. `set_default_pool_threads(6)`
then constructs Pool(6), whose `one_cpu_per_core` probe sees only the
pinned main-thread mask {0}, fails its `size == threads` check, skips
pinning — and all 5 new workers inherit the {0} mask (return codes
unchecked, no fallback). All threads time-share one core.
Fix (`src/pool.cc`): capture the pre-pinning allowed set once; on size
mismatch explicitly restore the full mask on main+workers instead of
inheriting. Verified: threads-6 5 s clip 63.05 s -> 4.02 s (user 20.7,
ratio 5.2x), SNR vs fp32 25.19 dB (quality-neutral), vs t8 29.89 dB
(chunk-order seeds only), `vi_tests` green. t8 path untouched (early-return,
same pinned-8 pool). Lesson: the threads-6 full-file number (576 s) was
never a data point — thread-count tuning is closed, 8 stays (6 threads
cannot beat 8 on wall: same user CPU over fewer cores).

## Frequency-domain cropping: measured and rejected (2026-11-20)

Hypothesis: for band-limited radio input (USB voice, nothing useful outside
0.3–4 kHz) runtime band cropping could cut compute roughly in proportion to
bands kept (B=23 → ~×0.53 work, B=19 → ~×0.45; band attention quadratic;
FFT/OLA fixed). All 46 bands cost the same (width-512 rows per band), so the
arithmetic was clean — if the model tolerated it.

Zero-code A/B: brick-wall filter the INPUT at a band boundary, keeping float
values bit-identical in the kept range (8-bit source re-encoded into 16-bit
containers; `read_wav`'s 8-bit path is `(v−128)/128`, so values match
exactly), run the full fp32 model (44.29 s, 8 threads, bf8, window 600/200),
compare outputs against the unfiltered run.

| arm: input removed | overall SNR vs full-input run | note |
|---|---:|---|
| control: no filtering, container only | bit-identical, max diff 0.0 | methodology + determinism validated |
| >13.03 kHz (SDR noise floor only, −38 dB) | 3.4 dB | collapse |
| >7.99 kHz | 4.3 dB | collapse |
| >6.47 kHz | 3.6 dB | collapse |
| >4.0 kHz | 4.2 dB | 0–3 kHz energy −9.2 dB vs −5.8 dB |
| >3.07 kHz | ~2.6 dB | collapse |
| <281 Hz (hum) | 2.0 dB | 0.3–3 kHz energy −11.0 dB |
| keep only 281 Hz–3.07 kHz | 0.1 dB | near-silence (−28.2 dB) |

Listening (operator): kept speech sounds clean, but the model mutes most
borderline speech ("holes"). The collapse is FLAT across cutoffs: removing
only the 13–24 kHz noise floor, or only the sub-300 Hz hum, produces the
same conservative gating as removing everything. The whole spectrum is
load-bearing: the model reads noise-only and hum-only regions as its
speech/noise reference (per-band LayerNorm removes absolute level, so it is
the *texture* of those regions that carries the evidence). This is likely
why it gates so well on noisy radio and never hallucinates.

Caveat: these arms zero the removed bands' input, so their features become
degenerate constants inside band attention / freq mamba. A true runtime crop
(bands absent from attention entirely, `--bands B`) is a different
perturbation and was NOT implemented — with eight arms collapsing
identically, the prior for the crop being clean is too low to spend the day.
No retraining pipeline exists for this model, so a retrained narrow-band
variant is not available either.

Conclusion: frequency-domain compute reduction is CLOSED for this model
without retraining. Remaining lever on this hardware: the streaming window
architecture — restart+crossfade costs ×4/3 on all frame work; a sliding
window without restarts is ~−25% wall, testable with zero code via
`--window-frames 0 --cache-frames N`.

Artifacts: `/tmp/opencode/ab_{orig,control,f13,f8,f6,filt,filt3,hp300,bp}.wav`
and the filtered inputs
`/tmp/opencode/{control16,lowpass*,highpass281,bandpass_281_3070}.wav`.

## Sliding window: measured, operator-validated, adopted as default (2026-11-20)

With frequency cropping closed (previous section), the restart+crossfade
tax became the main remaining lever: lanes 600/200 recompute 200 of every
600 frames → ×4/3 on all frame work. The alternative needs no code:
`--window-frames 0` runs ONE continuous lane (mamba state never resets, KV
trimmed by the existing compaction path) with `--cache-frames N`.

Full 44.29 s file, 8 threads, bf8, arms run back to back (shared box, wall
indicative):

| arm | processing | realtime |
|---|---:|---:|
| fp32 restart 600/200 (old default) | 71.5 s | 0.62x |
| fp32 sliding, cache 1200 | 58.98 s | 0.75x |
| fp32 sliding, cache 600 | 54.08 s | 0.82x |
| fp32 sliding, cache 400 | 51.68 s | 0.86x |
| fp32 sliding, cache 300 | 50.55 s | 0.88x |
| int8 restart 600/200 (old default) | 30.6 s | 1.44x |
| **int8 sliding, cache 400** | **22.33 s** | **1.98x** |

Quality diagnosis (sliding cache 400 vs restart): bit-identical for the
first ~3 s, divergence begins exactly at the first lane boundary (restart
starts its second lane at frame 400); no time shift (0.0 ms best alignment
lag); the difference spectrum tracks the signal spectrum (−1.5 dB vs −1.2 dB
share in the 0.3–1.5 kHz band) — different but legitimate denoising
decisions, not added noise and not muting (0.3–3 kHz energy −4.7 dB vs
restart's −5.6 dB). `clean2.wav` was produced with restart semantics, so
sliding-vs-clean2 reads ~8 dB BY DESIGN and is not a gate (the restart
reference itself matches clean2 at 69.7 dB).

Operator listening verdict on the radio material: fp32 cache 400, fp32
cache 300 and int8 cache 400 all satisfactory.

Decision (operator-approved): sliding is the new default — `src/stream.h`
now has `window_frames = 0`, `cache_frames = 400`; `--help` and
README-intel.md updated. Restart mode remains reproducible via
`--window-frames 600 --cache-frames 1200` (the configuration all historical
quality gates were anchored on). The ×4/3 tax is platform-independent, so
the same win is expected on the M4 Max (unmeasured there: int8 18.9 → ~14 s
est., fp32 30.4 → ~23 s est.).

Verification (final binaries, new defaults, 2026-11-20): clean rebuild;
`vi_tests . quick` all green (FFT roundtrip, carried-state selective scan,
AVX2 int8 strided/tile-slice with and without bias, four-query attention,
vector math; the `ref/` stage comparison stays skipped — fixtures absent
throughout this workspace). The new-default fp32 run is **bit-identical**
to the validated `ab_sw400` arm, and the new-default int8 run is
**bit-identical** to `ab_sw400_i8` (determinism rerun also bit-identical):
the default flip is pure configuration, zero change in computation. `ctest`
1/1 passed for both the native build and the `VI_NATIVE=OFF` portable
build. New-default timings on this box: fp32 51.51 s (0.86x), int8 22.32 s
(1.98x). One pre-existing warning from the in-flight WIP surfaced in the
portable build (`vecmath.cc:1237` unused `inv_s` in `v_quant_pack_a_i8_block`)
— left for its owner to fix with their commit.

## ARM build fix + new recording, 2026-11-20 14:38 CET (M4 Max, Mac16,6)

The x86 session left the tree unbuildable on ARM: `v_quantize_kv_row` calls
`quantize_kv_row_scalar` unconditionally, but the definition sat inside the
`__x86_64__` guard, so `build-native` failed with
`use of undeclared identifier 'quantize_kv_row_scalar'` at `vecmath.cc:1875`.
Moved the scalar helper into its own anonymous namespace ahead of the guard — it
is the non-AVX2 fallback and the tail handler on x86, and the only path on ARM.
No arithmetic changed.

Fresh build: `cmake -S . -B build-native -DVI_NATIVE=ON -DVI_METAL=OFF`,
clean, and `./build-native/vi_tests . quick` all green (samples 117.5 dB).

### `audio_7133500Hz_14-34-23_20-11-2026.wav` (25.16 s, 48 kHz, stereo PCM16)

Requested run: `--quant int8 --window-frames 0 --threads 8`, everything else
default (block 8, lookback 400, capacity 600). CPU only (`device cpu`),
`int8 weights (SME2)`.

| arm | processing | realtime | user CPU |
|---|---:|---:|---:|
| **int8, 8 threads, sliding** | **7.76 s** | **3.24x** | 61.25 s |
| fp32, 8 threads, sliding | 11.75 s | 2.14x | 90.39 s |

Output `…-2026.clean2.wav`: 48 kHz mono PCM16, 1 207 680 samples — same count as
the input, 25.160 s, peak 0.805, rms 0.0285 against 0.0949 at the input
(17.8 dB down on the quietest and the loudest tenth of 50 ms frames alike, so
it is not a global gain).

Checks: `int8 vs fp32` on the same input **15.7 dB** (lower than the 20.6 dB on
the older 44 s file; this material is harder, and the per-frame split is
unchanged in character). Sliding default vs restart 600/1200 diverges by
**7.0 dB by design** from the first lane boundary — the restart arm was run only
as a sanity check and is not the deliverable.
