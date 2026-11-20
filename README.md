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
| speed on Apple M4 Max, fp32, 6 threads | **0.39× realtime** (112 s for 44 s of audio; 9.49 CPU-s per audio-s) |
| speed, int8 weights (`--quant int8`), 6 threads | **0.67× realtime** (66 s for 44 s; 5.47 CPU-s per audio-s) |
| speed, 1 thread, fp32 | 0.13× realtime |
| output vs the Python reference, fp32 | 71.5 dB SNR on a 44 s file (PCM16 floor ≈ 77.5 dB) |
| output vs fp32, `--quant int8` | 20.6 dB SNR (8-bit noise, −84 dBFS floor in silence) |

The model is GPU-class (the Python reference itself runs on MPS at ~4×
realtime). On a CPU it is compute-bound; see [Performance](#performance).

## Build

```bash
tools/fetch_third_party.sh          # one time, network needed
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The three dependencies are header-only. The repository references them, it
does not vendor them; `tools/fetch_third_party.sh` downloads the pinned
versions into `third_party/` (git-ignored):

| | used for | licence |
|---|---|---|
| [Eigen 3.4.0](https://gitlab.com/libeigen/eigen) | GEMM, `third_party/Eigen` | MPL-2.0 |
| [pocketfft](https://github.com/mreineck/pocketfft) commit `4d865cb` | the 2048-point real FFT (what NumPy/SciPy use) | BSD-3 |
| [xsimd 13.2.0](https://github.com/xtensor-stack/xsimd) | portable SIMD `exp`/`erf`/`tanh`/`log1p` for the state-space scan and activations | BSD-3 |

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
`--float32` (default output is PCM16, peak-normalised), `--quant int8`,
`--profile`.

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

Measured on an M4 Max (10 performance + 4 efficiency cores), 44.29 s file,
6 threads — the best thread count on this machine for both paths, since above
six the pool spills out of one core cluster and every barrier waits for the
slower one.

| path | wall | user | CPU-s per audio-s | realtime |
|---|---|---|---|---|
| fp32 | 112.3 s | 420.3 s | 9.49 | 0.39× |
| `--quant int8` | 66.1 s | 242.1 s | 5.47 | 0.67× |
| fp32, 1 thread | — | 39.5 s per 5 s | 7.9 | 0.13× |

Where the time goes, 5 s clip, 6 threads (`--profile`):

```
fp32:  gemm 7.83   time_attn 3.12  band_attn 2.52  mamba 4.61  decode 0.15
int8:  gemm 3.21   time_attn 1.86  band_attn 1.21  mamba 2.78  decode 0.13
       (the attention and mamba figures include their own GEMMs)
```

The dense layers are 76 % of the fp32 run, so that is where the work went:

* They are split by **output columns**, so threads stream disjoint weight slices.
* The pool spins for 20 000 `cpu_relax` rounds before blocking for 200 µs. The
  old 2 ms condition-variable sleep let one late worker stall every barrier;
  isolated GEMM throughput used to vary between 245 and 485 GFLOP/s run to run.
* Passes estimated at under 2 M operations are **not** split: LayerNorm and the
  residual add are 94 k floats each and cost more in barriers than in work.
* **int8 weights and activations** with the NEON `sdot` instruction: 16 MACs per
  instruction instead of 4, and a quarter of the weight stream. 963 GFLOP/s
  against 418 GFLOP/s for fp32 at six threads (2.3×).

Sample-exact kernels are not the limit: the FFT is 0.1 s of the run, pocketfft
is the same code NumPy uses, and single-thread Eigen reaches 80 GFLOP/s on the
model's shapes (82 GFLOP/s on a 1024³ square), which is Eigen's practical fp32
peak here.

`--quant int8` is opt-in because it is lossy: 20.6 dB against the fp32 output,
which is signal-correlated 8-bit noise rather than a hiss (loudest 10 % of frames
28.5 dB, quietest 10 % 15.8 dB, absolute floor −84 dBFS in silence). Per-group
scales would improve it at roughly two thirds of the int8 speed advantage.

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
