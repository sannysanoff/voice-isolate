# Harmonic pitch fixer (`pitchfix-cli`)

Fixes a voice whose partials sit at `h*f0 + offset` instead of `h*f0`.

## What is wrong with this recording

`audio_7133500Hz_14-34-23_20-11-2026.wav` is a 48 kHz stereo capture of a radio
receiver. A lower-sideband demodulator tuned to the wrong carrier shifts *every*
frequency by the same amount. The spacing of the voice partials is unchanged;
only their absolute positions are wrong. At t=1.47 s the peaks are 314, 522,
722, 943 and 1155 Hz: spacing about 212 Hz, all 100 Hz above the harmonic
lattice. A resampler would be the wrong fix — it would scale the spacing too.

Two independent sources are present in this file:

| component | tuning error | level (rms) |
|---|---|---:|
| isolated voice (`.clean.wav`) | about **+108 Hz**, stable over the file (±3 Hz per 3 s block) | 0.043 |
| background bed (`.clean_background.wav`) | about 0 Hz (in tune) | 0.077 |

The bed is louder, so a blind global statistic on the raw mix mostly measures
the music. The filter runs on the raw file as requested and follows the voice
comb because the estimate tracks a local lattice from frame to frame; a second
source with a different offset cannot be corrected by one shift. Shifting the
whole file therefore detunes the music bed by the same amount. This is the
"some components are lost" trade-off of a single global correction.

## Build and run

```sh
cmake -S . -B build-pitchfix -DCMAKE_BUILD_TYPE=Release -DVI_METAL=OFF
cmake --build build-pitchfix --target pitchfix-cli -j8
./build-pitchfix/pitchfix-cli audio_7133500Hz_14-34-23_20-11-2026.wav /tmp/raw-fixed.wav --report
```

The input must be 48 kHz; the filter is fixed to that rate. It uses a 4096-sample
Hann window, a 1024-sample hop and delays the signal by one window (4096
samples). The WAV output is that delayed stream flushed with 4096 zeros, so it
holds the leading delay followed by the complete processed input. `--filter`
does raw float32 stdin/stdout with the delay left intact.

Options: `--max-shift 300` (Hz, also the estimate search range), `--tau-ms 500`
(easing time constant), `--fixed HZ` (skip estimation), `--seed HZ` (pin the
alias branch), `--f0-min/--f0-max`, `--min-confidence`, `--block`, `--report`.

The measured offset is only defined modulo the harmonic spacing: `offset` and
`offset - m*f0` put the partials on the same lattice. The filter unwraps each
measurement against its running estimate so the correction does not jump by a
whole spacing. `--seed 108` pins the branch when the operator knows the error.

## How it works

1. **Estimate** (2048-sample analysis frame, every 1024 samples): remove the
   300 Hz log-spectral trend, score a 2D grid of `(f0, delta)` with a comb
   contrast (energy on `h*f0+delta` minus the midpoints, which rejects
   subharmonics), then fit the lattice to the prominent spectral peaks. The
   peak positions use exact three-bin Hann interpolation, so the fit is
   unbiased on clean partials.
2. **Smooth**: median of recent fits (0.15 s) for outlier rejection, then a
   one-pole easing with tau = 0.5 s. Speed is proportional to the remaining
   distance, so the correction moves fast at first and eases in. Clamped to
   `±max_shift_hz`.
3. **Shift**: overlap-add single-sideband translation,
   `y = x*cos(theta) + H{x}*sin(theta)`, with H from the FFT of each window and
   an accumulated oscillator phase. No resampling: spacing is preserved, and
   components pushed past 0 Hz are dropped.

## Verification

`python3 tools/pitchfix_analyze.py before.wav --compare after.wav` reports the
harmonic spacing, the offset reduced into `±f0/2`, and a lattice-contrast score.
The metric used below is the per-frame median canonical offset of the isolated
voice; the corrected file was re-isolated before measurement.

| file | canonical voice offset | median per 3 s |
|---|---:|---|
| raw input | -25 Hz | -26, -38, -30, -31, -4, -34, -32, -15, +27 |
| auto-corrected raw input | **-1.5 Hz** | -10, +8, -2, -8, -5, +3, -4, -5, -2 |
| raw input, `--fixed 108` | -2.5 Hz | -2, -2, -2, -3, -13, -2, -1, -0, -4 |

Unit tests (`ctest -R pitchfix_tests`) cover delay/gain, exact translation,
offset tracking, the easing time constant, clamping, streaming determinism
across block sizes 1..5000, and silence. The filter runs about 70x realtime on
one core of this machine.
