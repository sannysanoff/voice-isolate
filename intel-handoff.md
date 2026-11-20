# x86-64 optimization handoff

## Objective and current state

Make four-frame streaming voice isolation approach realtime on 4–8 CPU threads,
while retaining buildable source and the existing ARM/NEON/SME2 and Metal
implementations. Preserve the fp32 reference-quality mode; clearly label and
measure any lower-precision mode. Do not assume that an AMD benchmark predicts
performance on every Intel processor, or vice versa.

As of this handoff, `master` includes the x86 work through merge `5a84d14` and
this investigation is recorded in `progress.md` (commit `f7d50b1`). The x86
performance goal is **not yet met**. The latest remote `master` changes were
merged after the full-file x86 benchmark; measure the current source again
before treating the older measurements as its baseline. Check `git status`,
branch and remote history before starting, and preserve any subsequent work.

> **Status update (2026-11-20):** the realtime goal **is met** for the int8
> streaming path on the second x86 machine below (Ryzen 7 7435HS, 8 threads,
> **1.08× realtime**). It remains unmet on the original Ryzen 9 3900X and for
> fp32 on x86, and no Intel CPU has been tested. The final stage, including the
> changed default block size that affects all future comparisons, is appended
> at the end of this document. The statement above is kept as the historical
> starting point.

## Workload and machine

The measurements so far used Linux on an **AMD Ryzen 9 3900X**, with 12 CPUs
visible and AVX2/FMA. The recording is 48 kHz mono, 2,125,746 samples, about
44.29 s. Files in this workspace (they may not be in the Git repository):

- Input: `audio_7082000Hz_18-59-39_25-06-2024-L20-endfed-maticna.wav`
- PCM16 reference: `audio_7082000Hz_18-59-39_25-06-2024-L20-endfed-maticna.clean2.wav`
- Weights: `model/weights.bin`, `model/weights.txt`, `model/config.txt`
- Latest measured output: the same input basename with `.optimized-intel.wav`

Use the default fp32 mode, `--block-frames 4`, `--window-frames 600`, and
`--crossfade-frames 200` for comparisons. The CLI accepts `--threads N` and
`--profile`; see `--help`. A five-second excerpt is useful for iteration but
never fills the attention window, so its speed is not the full-file steady-state
speed. The default build uses `-march=native`; use `-DVI_NATIVE=OFF` to check a
portable build, and record the CPU features for every benchmark.

## Measurements already made

These are single wall-clock measurements, not medians, from before the later
`master` merge. Profile numbers are reported wall-time accumulators, and should
not be added across overlapping stages. The short-clip CLI measurements include
model startup; the full-file CLI reports processing time. They are useful
starting points, not a substitute for rerunning a controlled baseline.

| 5 s excerpt, 8 threads | Elapsed | GEMM profile | Scan profile |
|---|---:|---:|---:|
| Scalar x86 scan, Eigen (same-source experimental baseline) | 13.72 s | 7.48 s | 2.84 s |
| SSE2 scan, Eigen | 11.67 s | 7.17 s | 1.19 s |
| SSE2 scan, optional OpenBLAS | 10.52 s | 5.93 s | 1.24 s |
| Experimental AVX2 int8, OpenBLAS for remaining work | 12.04 s | 7.38 s | 1.28 s |

The full recording using SSE2 scan and OpenBLAS, eight threads, took **106.19 s
for 44.29 s of audio (0.42× realtime)**. GEMM accounted for 57.05 s and scan
11.18 s in the stage profile. Output had 2,125,746 samples and measured
**69.71 dB SNR**, maximum absolute error **6.10e-5**, versus the existing PCM16
reference. The five-second optimized outputs were within one PCM16 step of the
same-source scalar baseline (at least 97.7 dB SNR). Output file comparison
includes PCM16 rounding; it is not a substitute for float-stage checks.

The AVX2 int8 experiment was discarded. It calculated one output dot product
at a time with `maddubs`/`madd`, with no output tiling or packing, and paid
activation quantization costs. It was slower than fp32 OpenBLAS; the specific
cause was not isolated, and its audio quality was not independently validated.
Do not infer that a well-tiled x86 int8 kernel is necessarily slower. Existing
ARM int8 quality results in `progress.md` show a substantial quality tradeoff,
not a guarantee of the same error for a new x86 implementation.

## Relevant code

- `src/model.cc`: `Model::linear_par`, `linear_par_i8`, `linear_par_sme`, GEMM
  splitting, thresholds, profiling and attention/Mamba orchestration. Inspect
  the current dispatch order and actual tensor shapes before changing kernels.
- `src/gemm_int8.cc`, `src/gemm_int8.h`: ARM SDOT kernel, quantization and
  scalar fallback. `int8_available()` currently reports true for ARM SDOT,
  not x86. A new x86 path needs correct feature gating and must honor column
  slices and tails; do not silently use the slow scalar fallback as an
  optimization. `linear_par_i8` currently quantizes activations per worker.
- `src/vecmath.cc`: `v_mamba_scan` has a four-lane SSE2 implementation over
  `[inner/4][state][4]`, the existing NEON implementation, and scalar fallback.
  `src/vecmath.h` describes state and array layouts. Preserve state across
  calls and handle null state and partial channel ranges correctly.
- `src/pool.cc`: thread pool and default thread count. Profile synchronization,
  oversubscription and cache behavior before altering thresholds or splitting.
- `src/gemm_sme.cc`, `src/metal_backend.mm`: ARM SME2 and Metal backends. Keep
  their dispatch and build guards intact; x86 measurements cannot validate them.
- `CMakeLists.txt`: `VI_NATIVE`, optional `VI_OPENBLAS_LIBRARY`, ARM feature
  checks and Metal configuration. The optional OpenBLAS path currently sets its
  thread count to one **process-wide**, while the pool supplies parallelism.
- `tests/vi_tests.cc`: FFT/vector tests, direct selective-scan test, and optional
  Python reference tests. `README.md`, `README-intel.md` and `progress.md`
  explain dependencies, CLI and previous ARM/x86 experiments.

## Suggested workflow

1. Establish the **current** full-file baseline with the exact CPU, compiler,
   flags, dependency versions, thread count and source revision recorded. Build
   both Eigen and OpenBLAS variants in separate build directories. Run on an
   otherwise idle system and use several runs/medians when comparing changes;
   measure startup separately if it affects short clips. Record wall time and
   CPU time, and retain the profiler output. Sweep 4, 6 and 8 threads rather
   than assuming the best count. Include warm-up and window/crossfade effects.
2. Use a 5 s excerpt for fast iteration, then validate promising changes on the
   full 44.29 s input. The GEMM stage is the largest observed target. Consider
   profiling actual GEMM shapes and CPU/cache use; compare packed/tiled x86
   kernels or appropriate optimized libraries, including their packing and
   quantization cost. Check CPU feature support and provide safe fallback paths.
   The failed simple int8 experiment is only one design point.
3. Compare float outputs or intermediate stages against fp32 when available,
   as well as the supplied PCM16 file. Report sample count, SNR and max error;
   for lower precision, assess quality over time and quiet/loud regions. Test
   matrix slices, tails, non-multiple dimensions, multithreading and numerical
   extremes against an independent scalar reference. Scan tests should cover
   carried and null state, channel slices and multi-step recurrence.
4. Run CTest for native and `VI_NATIVE=OFF` builds and any optional backend
   build. The `ref/` fixtures were absent in this workspace: CTest can pass
   while stage, windowed and cache-compaction comparisons are **skipped**. Check
   test output, obtain fixtures if available, and do not report skipped tests
   as passed reference checks. On x86, ARM/Metal correctness remains untested;
   avoid claiming those backends were verified without their hardware.
5. Keep measured results and rejected experiments in `progress.md`. Preserve
   working input/reference WAVs and choose a new output filename for each run.

Example OpenBLAS build and full-file benchmark (replace the library path):

```sh
cmake -S . -B build-openblas -DVI_OPENBLAS_LIBRARY=/path/to/libopenblas.so
cmake --build build-openblas -j 4
ctest --test-dir build-openblas --output-on-failure
/usr/bin/time -p ./build-openblas/voice-isolate-cli \
  audio_7082000Hz_18-59-39_25-06-2024-L20-endfed-maticna.wav \
  /tmp/voice-isolate-x86-output.wav --threads 8 --block-frames 4 --profile
```

The normal Eigen build omits `VI_OPENBLAS_LIBRARY`; because CMake caches this
option, use a separate build directory or explicitly clear it. OpenBLAS is an
optional external dependency, not bundled. In the measured environment the
library used was `/usr/local/lib/ollama/mlx_cuda_v13/libopenblas.so.0` (version
0.3.15), which is machine-specific and should not be hard-coded in source.

---

## Final stage: x86 int8 realtime goal met (Ryzen 7 7435HS, 2026-11-20)

### What changed since the previous handoff

A second x86 host was used after the throttled 3900X runs: **AMD Ryzen 7
7435HS**, 8 physical cores / 16 SMT threads, AVX2+FMA, **no AVX-512, no
AVX-VNNI, no fp16 arithmetic**, ~4.2 GHz under load, no thermal throttling
during the timed runs. GCC 13.3.0, Release `-O3 -march=native -fno-math-errno`,
Eigen 3.4.0 / xsimd 13.2.0, no OpenBLAS. All runs used eight threads, the
44.29 s WAV, 600-frame window and 200-frame crossfade; times below are CLI
processing time (model load excluded). Full detail and rejected experiments
are in `progress.md`.

**Important for future comparisons:** the default `StreamOptions::block_frames`
changed from **4 to 6** (`src/stream.h`, CLI help updated). To reproduce any
earlier bf4 number you must now pass `--block-frames 4` explicitly. Six frames
is a throughput-only change in this codebase, but it is not bit-identical to
bf4; measured int8 SNR vs the matching fp32 output was 23.32 dB at bf4 and
23.40 dB at bf6 (bf8 was 21.62 dB, so bf8 was rejected despite being faster).

The changes that mattered, in order of measured effect:

1. `src/gemm_avx2.cc`: forced `tile<Rows>()` to `always_inline`. `nm` showed
   GCC emitting it as a standalone symbol (the `target("avx2")` attribute
   prevented inlining), which cost ~37% of kernel throughput: 187 -> 253 GOPS
   on an isolated 184x512x1536 layer. The epilogue is now specialized through a
   `gemm_tiles_impl<WithBias>()` helper rather than a lambda, so the
   `VI_NATIVE=OFF` build still compiles (a lambda loses the target attribute).
2. `src/pool.cc`: on Linux x86, pin one worker per physical core using
   `/sys/.../topology/core_id` and the process's allowed CPU mask. SMT placement
   otherwise stacked two workers on one core. `VI_NO_AFFINITY=1` disables it;
   the fallback is unchanged when fewer than `threads` distinct cores exist.
   The pre-existing x86 "stay parked across timeout checks" spin behaviour was
   preserved.
3. `src/gemm_avx2.cc`: load the 16 per-column offset values once per tile
   instead of once per row. Kept the original column-major j-outer loop; a
   row-major nesting was slower in the real per-worker workload.
4. `src/stream.h`: default block size 6 (see caveat above).

### Result and verification

| Step (int8, 8 threads, full file) | Processing | Realtime |
|---|---:|---:|
| Starting build (bf4, no inlining, default affinity) | 51.5 s | 0.86x |
| `tile()` forced inline | 48.8 s | 0.91x |
| + column-major nesting + one-core-per-thread affinity | 44.2 s | 1.00x |
| + default bf6 (current) | 40.9-41.1 s | **1.08x** |

- Output: 2,125,746 samples (matches `clean2.wav`).
- Quality: 23.40 dB SNR, max abs error 0.0905, both vs matching-fp32 and vs the
  supplied PCM16 reference. int8 is clearly labelled low precision; fp32 stays
  the reference path.
- `ctest --output-on-failure` passes for both `-march=native` and
  `VI_NATIVE=OFF` builds. As before, the optional `ref/` fixtures were absent,
  so the Python stage/windowed/cache-compaction comparisons are skipped, not
  passed. No ARM or Metal backend was exercised on this host.
- Reproduce with:
  ```sh
  cmake -S . -B build-opt -DCMAKE_BUILD_TYPE=Release
  cmake --build build-opt -j8
  ./build-opt/voice-isolate-cli \
    audio_7082000Hz_18-59-39_25-06-2024-L20-endfed-maticna.wav \
    /tmp/voice-isolate-x86-int8-final.wav \
    --quant int8 --threads 8 --profile
  ```

### What was tried and rejected here

AVX2 k-unrolling, 8-row split tiles, hand-written AVX2 2-key/4-key attention QK
tiling (slower than xsimd's existing 60 GOPS path), LTO, bias-template
specialization alone, 4/6/7 thread counts (8 with affinity is best), bf8
(speed 1.11x but 1.7 dB quality loss), and changing the pool spin reset
(negligible wall time, more user CPU). The kernel is near the AVX2 ceiling:
~253 of ~270 GOPS single-threaded, since `vpmaddubsw`/`vpmaddwd` is the only
int8 multiply available without VNNI.

### Open items

- No actual **Intel** CPU was measured; all x86 numbers are AMD. The original
  3900X remains at 0.42x (OpenBLAS fp32) / 0.38-0.39x (early int8) and is throttled.
- fp32 on x86 is still far from realtime (~80 s on the 7435HS); only int8
  reaches the target.
- The `ref/` fixtures are still needed to turn the skipped CTest comparisons
  into real reference checks.
