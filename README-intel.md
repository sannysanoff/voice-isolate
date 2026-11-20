# x86-64 build and four-frame streaming

On Linux with CMake, a C++17 compiler and pthreads:

```sh
cmake -S . -B build-intel
cmake --build build-intel -j
./build-intel/voice-isolate-cli input.wav output-intel.wav model
```

The model directory defaults to `model` relative to the current directory. The
input must have the model's sample rate (48 kHz); channels are averaged to mono.
The output is mono 16-bit PCM. Processing uses the streaming defaults: four
frames per block, a 600-frame window, and a 200-frame crossfade. The executable
reads the input in chunks and flushes the stream at the end.

By default `VI_NATIVE=ON` tunes the build for the machine compiling it. For a
binary intended for other x86-64 machines, configure with `-DVI_NATIVE=OFF`.
The ARM SME2 and dot-product kernels are disabled on x86; the fp32 path is used.
Run `ctest --test-dir build-intel --output-on-failure` for the available tests.
Some reference tests are skipped if the optional `ref/` data is absent.
