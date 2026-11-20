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

The default streaming configuration uses four frames per block, a 600-frame
window and a 200-frame crossfade. See `--help` for the filter mode and options.
The input sample rate must match the model (48 kHz); channels are averaged to
mono. The model directory defaults to `model` relative to the current directory.

`VI_NATIVE=ON` tunes the build for the build machine; use `-DVI_NATIVE=OFF` for
an x86-64 binary intended for other machines. Run
`ctest --test-dir build-intel --output-on-failure` for available tests; reference
tests are skipped when the optional `ref/` data is absent.
