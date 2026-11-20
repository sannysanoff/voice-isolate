# x86-64 build and four-frame streaming

Install the dependencies described in [README.md](README.md), then build:

```sh
cmake -S . -B build-intel
cmake --build build-intel -j
./build-intel/voice-isolate-cli input.wav output-intel.wav --threads 8
```

The normal build uses Eigen for matrix multiplication. An optional OpenBLAS
library can be selected by passing its full path at configure time, for example:

```sh
cmake -S . -B build-intel -DVI_OPENBLAS_LIBRARY=/path/to/libopenblas.so
cmake --build build-intel -j
./build-intel/voice-isolate-cli input.wav output-intel.wav --threads 8 --profile
```

OpenBLAS is restricted to one thread per call; the existing worker pool divides
matrix work across the selected threads. This changes the OpenBLAS thread setting
process-wide. The ARM SME2 kernels retain priority when available.

The default streaming configuration uses six frames per block, a 600-frame
window and a 200-frame crossfade. See `--help` for the filter mode and options.
The input sample rate must match the model (48 kHz); channels are averaged to
mono. The model directory defaults to `model` relative to the current directory.
`--block-frames 4` remains available for bit-comparable older runs; 6 is the
measured throughput optimum on 8-core x86 (throughput only, no latency change).

`VI_NATIVE=ON` tunes the build for the build machine; use `-DVI_NATIVE=OFF` for
an x86-64 binary intended for other machines. Run
`ctest --test-dir build-intel --output-on-failure` for available tests; reference
tests are skipped when the optional `ref/` data is absent.

For quantized inference on x86-64, select the AVX2 backend with the existing
int8 option. The CLI prints `int8 weights (AVX2)` when it detects support.

```sh
./build-intel/voice-isolate-cli input.wav output-int8.wav --quant int8 --threads 8 --profile
```

The model packs eligible dense weights on first use into 64-byte panels.
Each worker quantizes its own activation rows once and computes 16 output
channels per tile. Weight scaling bounds adjacent pairs to prevent saturation
in the AVX2 byte multiply-add. Small or incompatible layers use fp32. The
Mamba scan separately checks for both AVX2 and FMA. A `VI_NATIVE=OFF` build
includes both runtime dispatches and runs the fallback paths on older CPUs.

### x86 throughput work (8-core AMD Ryzen 7 7435HS, AVX2/FMA, no AVX-512/VNNI)

Full 44.29 s recording, `--quant int8 --threads 8`, default window/crossfade,
one run each. Processing time excludes model load:

| Configuration | Processing | Realtime | int8 SNR vs matching fp32 |
|---|---:|---:|---:|
| Before (block 4, non-inlined tile, default affinity) | 51.5 s | 0.86x | 23.3 dB |
| `tile()` force-inlined into `gemm_avx2_tiles` | 48.8 s | 0.91x | 23.3 dB |
| + original column-major nesting + one-per-core affinity | 44.2 s | 1.00x | 23.3 dB |
| + default block 6 (current, streaming) | ~41-42 s | **~1.05-1.08x** | **23.4 dB** |

The main win was forcing the AVX2 `tile()` template to inline: GCC emitted it
as a standalone call, costing ~37% of kernel throughput (187 -> 253 GOPS on a
184x512x1536 layer). Thread affinity pins one worker per physical core on Linux
x86 SMT machines (`VI_NO_AFFINITY=1` disables it). Six frames per block amortizes
per-dispatch overhead without changing int8 accuracy. No AVX-512 or AVX-VNNI is
available on this CPU, so the int8 kernel uses `vpmaddubsw`/`vpmaddwd` and is
already near the AVX2 peak (~253 of ~270 GOPS single-threaded).
