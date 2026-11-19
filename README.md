# voice-isolate (C++)

A portable C++17 port of [Inworld's Voice Isolator v1 realtime
model](https://huggingface.co/inworld/voice-isolator) as a **sample-in,
sample-out streaming filter**: 48 kHz mono float in, isolated voice out, no
Python and no GPU.

The reference implementation lives in the installed Python package
(`.venv/lib/python3.12/site-packages/inworld/voice_isolation/`); this port is
numerically faithful to it, verified stage by stage.

| | |
|---|---|
| sample rate | 48 000 Hz, mono |
| algorithmic latency | **3 968 samples = 82.7 ms** (2048-sample window + 4 frames × 480 look-ahead) |
| parameters | 58 178 056, fp32 (233 MB) |
| peak RSS | 1.07 GB (weights + streaming state) |
| speed on Apple M4 Max, 8 threads | **0.27–0.35× realtime** (~165 s for 44 s of audio) |
| speed, 1 thread | 0.11× realtime |
| output vs the Python reference | 71.4 dB SNR on a 44 s file (PCM16 floor ≈ 77.5 dB) |

The model is GPU-class (the Python reference itself runs on MPS at ~4×
realtime). On a CPU it is compute-bound; see [Performance](#performance).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Dependencies are vendored, header-only, and need no network:

| | used for | licence |
|---|---|---|
| [Eigen 3.4](https://gitlab.com/libeigen/eigen) | GEMM, `third_party/Eigen` | MPL-2.0 |
| [pocketfft](https://github.com/mreineck/pocketfft) | the 2048-point real FFT (what NumPy/SciPy use) | BSD-3 |
| [xsimd 13](https://github.com/xtensor-stack/xsimd) | portable SIMD `exp`/`erf`/`tanh`/`log1p` for the state-space scan and activations | BSD-3 |

`-DVI_NATIVE=ON` adds `-march=native` (drops binary portability).

## Weights

The runtime reads a directory with a packed blob and a manifest:

```bash
python tools/export_weights.py --out model          # needs the HF checkpoint cached
```

This writes `model/weights.bin` (all 665 tensors, float32, 64-byte aligned),
`model/weights.txt` (name, shape, offset) and `model/config.txt` (the
architecture read from the checkpoint). `--lookahead 4` selects the trained
streaming position (the checkpoint offers 0, 2 and 4 frames).

## Usage

### File

```bash
./build/voice-isolate-cli in.wav out.wav --float32
```

Options: `--block-frames N` (throughput only, default 4), `--window-frames N`
(time-attention window, default 600, `0` disables the windows),
`--crossfade-frames N` (default 200), `--cache-frames N`, `--threads N`,
`--float32` (default output is PCM16, peak-normalised), `--profile`.

### As a filter — samples in, samples out

```bash
./build/voice-isolate-cli --filter --chunk 480 < in.f32 > out.f32
```

Raw little-endian float32 mono at 48 kHz on stdin, the same on stdout. The first
`latency_samples()` outputs lag the input by 82.7 ms; every later chunk comes
out as soon as it is final. Output is **independent of how the input is
chunked** (measured: 126 dB agreement between 480-sample and 24 000-sample
pushes), because the checkpoint's attention window keeps the state bounded.

### C++ API

```cpp
#include "stream.h"

auto model = std::make_shared<vi::Model>("model");   // loads weights once
vi::StreamOptions options;                            // defaults = the CLI
vi::Stream stream(model, options);

// from a microphone or a socket, in pieces of any size
for (;;) {
  const int got = capture(buffer, kMax);              // 48 kHz mono float
  const std::vector<float> out = stream.process(buffer, got);
  playback(out.data(), out.size());
}
const std::vector<float> tail = stream.flush();       // end of source
```

`stream.latency_samples()` is 3968, `stream.sample_rate()` 48000. One `Model`
serves one `Stream` at a time (the working buffers are reused).

## Structure of the port

The network is a band-split dual-path model over a 48 kHz STFT
(`n_fft` 2048, hop 480 → 100 frames/s), 46 bands, width 512, 8 heads:

```
STFT (hann 2048, hop 480, no centre pad)
  → 46 band tokens/frame: LayerNorm(2·bins) → Linear(·,512) + band/position embedding
  → 7 dual-path blocks: T M T M T M T
      transformer: time attention (per band, RoPE, causal, exclusive) + FFN,
                   then band attention (per frame, 46 tokens) + FFN
      mamba:       selective scan along time (carried state), then two-sided
                   scan across bands (per frame)
  → per-band decoder heads: LN → 512 → GELU → 4·bins (complex mask + residual)
  → correction of the spectrum, iSTFT, overlap-add, output = input + change
```

Streaming semantics reproduced exactly from `streaming.py`: a frame's decoder
output is read 4 frames later, keys are rotated once at the slot they are
written (the slot *is* the rotary position), the time Mamba carries its
convolution tail and state, and a second lane starts 200 frames before each
600-frame window boundary with the state cloned and the attention empty,
crossfading into the boundary.

Source map: `src/model.cc` (network), `src/stream.cc` (filter, lanes,
overlap-add), `src/vecmath.cc` (xsimd kernels), `src/fft.cc` (pocketfft),
`src/pool.cc` (thread pool + vector math), `src/wav.cc`, `apps/vi_cli.cc`.

## Verification

`tools/reference_dump.py` runs the Python reference and dumps every
intermediate; `vi_tests` runs the same input through C++ and compares. Against
a 1 s clip, float32:

| stage | max abs error | SNR |
|---|---|---|
| STFT spectra | — | exact |
| encoder output | 5.2e-05 | 107.5 dB |
| block 0 (transformer) out | 9.3e-04 | 108.6 dB |
| block 1 (mamba) out | 9.3e-04 | 110.8 dB |
| block 5 (mamba) out | 4.2e-04 | 109.0 dB |
| decoder raw | 1.6e-05 | 123.1 dB |
| **output samples** | **7.5e-07** | **118.1 dB** |

Whole-file checks:

* 10 s, windowed 600/200 (the CLI default): 113.6 dB
* 20 s, no windows (exercises the attention-cache compaction): 107.9 dB
* the 44 s file against `…-maticna.clean2.wav` produced by
  `voice-isolate … --mode streaming --lookahead 4`: **71.4 dB SNR, max
  difference 6.3e-05** — at that file's own PCM16 quantisation floor.
* filter mode vs file mode: 126 dB.

Two real bugs this process found, both worth remembering: the decoder's FiLM
`condition` is **not** identity in the released checkpoint (its weights are
non-zero), and the wrappers' `output_proj` layers have biases.

## Performance

Measured on an M4 Max (10 P + 4 E cores), 44.29 s file:

```
encode 0.10 s   time_attn 50.6 s   band_attn 37.2 s   mamba 63.0 s
decode 2.8 s    (stages 153.7 s of 165.4 s wall = 93%)
```

* 0.11× realtime on one thread, **0.35× on 6 threads**, 0.31× on 8, worse above
  that: the pool re-synchronises at every GEMM (~70 barriers per frame), so
  extra cores return less than the barrier and memory-traffic cost. 8 threads is
  the default (`min(cores, 8)`); `--threads` overrides it.
* The kernels are the limiting factor, not the FFT: per frame the model is
  ~3.2 GFLOP, and realtime at 100 frames/s would need ~320 GFLOP/s sustained —
  about half of the whole M4 Max CPU's fp32 peak. Expect realtime only on a GPU.

Reaching realtime on CPU would need quantisation (next step) plus a wider
compute backend. The single biggest present win is done already: GEMMs are
split by output columns so threads stream disjoint weight slices.

## Tests

```bash
ctest --test-dir build --output-on-failure      # 4 min (regenerates nothing)
./build/vi_tests . quick                        # FFT, vector math, stage compare
```

The reference dumps under `ref/` are derived data, regenerable with:

```bash
python tools/reference_dump.py stages --seconds 1 --window 0
python tools/reference_dump.py final --seconds 10 --window 600 --crossfade 200 --out ref/win10
python tools/reference_dump.py final --seconds 20 --window 0 --out ref/plain20
```

Memory safety: `cmake -DCMAKE_CXX_FLAGS=-fsanitize=address` and run
`vi_tests . quick` / `--filter`; both are ASan-clean.

`tools/mamba_probe.py` compares a numpy mirror of the Mamba layer against the
torch module (they agree to 133 dB); it is how the scan implementation was
validated independently of C++.
