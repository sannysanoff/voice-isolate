# pitchfix handoff

State: **unfinished**. The filter core works. The automatic tuning-error detector
does not work on `waving.wav`, so `waving-fixed.wav` is not a correct fix.

## 1. Goal

`waving.wav` is a 48 kHz stereo capture of a voice whose harmonics are detuned:
the partials sit at `h*f0 + offset` instead of `h*f0`. A single-sideband
demodulator tuned to the wrong carrier does this. Every partial moves by the same
amount, and the spacing between partials does not change.

The utility must, in real time, sample in and sample out:

1. Detect the tuning error from the signal itself.
2. Translate the frequency of the whole signal by `-offset`, with a maximum of
   +/-300 Hz (configurable).
3. Move the correction smoothly. Speed proportional to the remaining distance,
   about 0.5 s of easing. No jumps.
4. Bring both voices of the recording back to normal at all times.
5. Work on the raw file, not on a `.clean.wav`.

Frequency translation, not resampling. Resampling would scale the spacing as
well as the position.

## 2. Files

| path | role |
|---|---|
| `src/pitchfix.h` | `PitchFixOptions`, `PitchFixStats`, fixed constants, streaming API |
| `src/pitchfix.cc` | estimator, easing, median smoothing, SSB shifter, alias unwrap |
| `apps/pitchfix_cli.cc` | WAV and raw float32 CLI, delayed output, `--report` |
| `tests/pitchfix_tests.cc` | unit tests, all passing |
| `tools/pitchfix_analyze.py` | harmonic-lattice analysis and source/after comparison |
| `CMakeLists.txt` | modified: adds `pitchfix`, `pitchfix-cli`, `pitchfix_tests` targets |
| `README-pitchfix.md` | older findings and usage, describes a different recording |
| `waving.wav` | input, 48 kHz stereo, 27.275 s, 1309200 frames per channel |
| `waving-fixed.wav` | current output, mono, 27.36 s (4096-sample delay plus flush) |

Everything is uncommitted on `master`. `git status`: `M CMakeLists.txt`, and the
five pitchfix files are untracked.

## 3. Design as implemented

Fixed constants: `kPitchFixSampleRate = 48000`, `kPitchFixWindow = 4096`,
`kPitchFixHop = 1024`, latency = one window = 4096 samples. All sizes are in
samples so the FFT and Hann lengths stay powers of two.

Signal path per 1024-sample block:

1. Slide `hist_` (last 4096 samples) forward by one hop.
2. Analyse the newest 2048 samples: `analyse()`.
3. `shift_block()`: Hann window, rFFT, build the one-sided analytic spectrum
   `-i*X` (DC and Nyquist zeroed), inverse transform to get `H{x}`, then
   `y = x*cos(theta) + H{x}*sin(theta)`. Overlap-add with the Hann window again,
   scale by `2/3` (COLA of the squared window at 75% overlap).
4. `theta` is accumulated by `hop * dphi`, so a changing offset does not click at
   block edges.

Estimator (`analyse`):

1. Log magnitude of the 2048-point frame, minus a 13-bin (~300 Hz) running
   median. This removes the low-frequency tilt and the vowel formant envelope.
2. Score a grid of `(f0, delta)` with a comb contrast: mean log magnitude on
   `h*f0 + delta` minus the mean on the half-harmonic midpoints, `h = 1..8`,
   band 110..3600 Hz. The midpoint term is what rejects sub-multiples.
3. Refine with least squares over prominent spectral peaks, using three-bin Hann
   interpolation for each peak position.
4. Reduce the measurement modulo `f0`, then ease it.

Smoothing, two stages:

- A 0.15 s sliding median (`recent_`, 7 frames) rejects per-frame fit outliers and
  is the value that anchors alias unwrapping.
- A 0.5 s sliding median (`slow_history_`, 23 frames) feeds a one-pole easing with
  `tau = 0.5 s`: `offset += alpha*(target-offset)`,
  `alpha = 1 - exp(-hop/(fs*tau))`. Speed is proportional to the remaining
  distance, so it arrives fast and eases in.

The shifter reads `slow_offset_` in estimate mode and `offset_` in `--fixed` mode.
In practice the two are now equal, because `update_offset` assigns
`slow_offset_ = offset_` at the end.

## 4. What is verified

The **shifter** is correct. Measured on synthetic input through `--fixed`:

| test | input | output |
|---|---|---|
| single harmonic tone, `--fixed 100` | 440, 880, 1320 Hz | 340, 780, 1180 Hz |
| harmonic comb, `--fixed 100` | 300, 500, 700... Hz | 200, 400, 600... Hz, no residue at 300/500/700 |
| comb, `--fixed 50`, direct DFT | 250, 450, 650 Hz | 200, 400, 600 Hz |

Also verified: block-size invariance (1..5000 samples per call gives identical
output), silence stays silent and finite, `--max-shift` clamps, the easing moves
no more than ~3.2 Hz per block even on a 130 Hz step.

The detector **worked once**, on the earlier recording
`audio_7133500Hz_14-34-23_20-11-2026.wav`, where the isolated voice had a stable
+108 Hz error. Canonical voice offset went from -25 Hz to -1.5 Hz in auto mode,
and to -2.5 Hz with `--fixed 108`. That file has one voice and one stable offset.
`waving.wav` does not.

## 5. Open bugs

### 5.1 Detector reports ~0 Hz and applies no shift (blocking)

Symptom on `waving.wav`: `--report` gives offsets that wander from +45 Hz to
-170 Hz, the final summary says `offset -141.4 Hz (range -171.6..+67.0 over 2349
frames)`, and the output sounds unchanged. A spectral peak comparison between
source and output shows the partials did not move as a group.

Cause: the lattice fit leaves `f0` free on every frame. With `f0` free, a
harmonic comb is harmonic at its own spacing by construction, so `delta` shrinks
to the small residual and cannot express the tuning error. Any tuning is absorbed
into `f0`. The estimator therefore measures inharmonicity, not detuning.

This is the central design flaw. It cannot be fixed by smoothing, clamping, or
alias unwrapping.

### 5.2 Alias ambiguity modulo `f0` (inherent, under-handled)

`delta` and `delta - m*f0` align the same comb, only the harmonic indices shift.
Unavoidable. The code unwraps toward the running estimate, and `--seed` pins the
first branch, but the anchor itself is the broken estimate from 5.1, so the unwrap
has nothing trustworthy to follow.

### 5.3 Two voices, two different offsets (structural)

`waving.wav` alternates between two voices. Each has its own `f0` and its own
tuning error, and they overlap in time. One global frequency translation cannot
correct two different offsets at once. Correcting voice A detunes voice B by the
difference. Any single-shift design has this limit. Options: track per-voice
offset and cross-fade the shift when the dominant voice changes, or run two
shifters with a separator, or correct only the dominant voice and accept the
other.

### 5.4 Median pipeline adds about a second of lag

The applied value passes through a 0.5 s median and then a 0.5 s easing. On a
smoothly swept synthetic offset, a target of 200 Hz was still at 278, 246, 231 Hz
at t = 1, 2, 3 s. The correction converges, but it trails a moving error. If the
wobble is fast, the residual wobble stays significant even when detection is
right.

### 5.5 Minor code defects

- `src/pitchfix.cc:82`: the pre-fill `if (have_offset_) std::fill(...)` never
  runs. `have_offset_` is still false at that point; it is set at lines 84-90.
  Dead code, so the slow median does start empty.
- `src/pitchfix.cc:387`: `if (!have_slow_) { offset_ = target; ... }` uses the raw
  clip of the measurement, not `slow_target`. First slow frame bypasses the
  median.
- `src/pitchfix.cc:73-76`: comment says "one-second median", the span is 0.5 s.
- `src/pitchfix.cc:299-311`: two nested `fit_lattice` calls with the same
  condition. Redundant; one call is enough.
- `PITCHFIX_DEBUG` grid print (line 261) is left in production code.
- `slow_offset_` and `offset_` are now the same value, so the two-stage split is
  vestigial. Either remove one or restore a real second stage.

### 5.6 Misleading diagnostics that cost time

Two earlier conclusions were wrong and are worth not repeating.

- The synthetic "f0 = 200, offset = +50" test signal was built as
  `h*(f0+off) = 250, 500, 750`. That is a **rescaled** voice with spacing 250, not
  a detuned voice. The detector correctly reported `f0 = 250, delta = 0`. The
  "spacing/offset degeneracy" conclusion drawn from it does not hold for real
  partial positions: `250, 450, 650` is harmonic only at `f0 = 200, delta = 50`.
  Build test signals as `h*f0 + off`, never `h*(f0+off)`.
- The measured source-to-output "shift" of +592, +410, +486, +709, -322 Hz is not
  evidence of a broken shifter. Those windows are unvoiced or mixed, and the
  matched peaks are different partials. The shifter is verified separately by
  Section 4.

## 6. Facts about `waving.wav`

- 48 kHz, stereo, 27.275 s. Mono content; the CLI uses channel 0 only.
- Strongest peaks in the 100..1200 Hz band are often 250, 500, 750 Hz, i.e.
  spacing 250. That is a valid harmonic comb, so the estimator reports offset 0.
- Per-frame medians over the whole file: `f0` 150.8 Hz (p10..p90 100.7..226.2),
  offset -0.6 Hz (p10..p90 -63.9..+58.2). The `f0` spread is the tell: the
  spacing is not stable across the file, which means the fit is following
  different voices or different partial sets, not one voice.
- Per-3 s medians of the canonical offset: +26, -1, -19, -12, -12, +3, -1, -1,
  +21, -13. Near zero, as expected from 5.1.
- The user supplied a spectrogram screenshot labelled "waving". It was not
  readable by this tool, so its content is unverified. The user reports that the
  "fixed" output looks like the source.

## 7. Repro

Build and test:

```sh
cmake -S . -B build-pitchfix -DCMAKE_BUILD_TYPE=Release -DVI_METAL=OFF
cmake --build build-pitchfix --target pitchfix-cli pitchfix_tests -j8
./build-pitchfix/pitchfix_tests          # all tests pass
```

Run on the file:

```sh
./build-pitchfix/pitchfix-cli waving.wav waving-fixed.wav --report
```

Detector trace per block, including the raw grid hit:

```sh
PITCHFIX_DEBUG=1 ./build-pitchfix/pitchfix-cli waving.wav /tmp/x.wav --report
```

Analysis helpers:

```sh
python3 tools/pitchfix_analyze.py waving.wav
python3 tools/pitchfix_analyze.py waving.wav --compare waving-fixed.wav
```

Ephemeral C++ probes used during debugging (in the session temp dir, may not
persist): `dbg_shift.cc` (fixed-mode shifter proof), `dbg_est.cc` (detector trace
on a synthetic comb), `dbg_grid.cc` (contrast surface print), `dbg_slow.cc`
(easing trace), `offset_per_sec.cc`, `dump_frames.cc`.

## 8. Test coverage

Covered: delay equals one window, unity gain in the pass band, exact translation
of a single tone, offset tracking on a synthetic error, easing time constant,
`--max-shift` clamping, streaming determinism across block sizes 1..5000,
silence.

Not covered, and this is the gap that let the bug ship:

- Two interfering voices with different offsets.
- A voice whose spacing is close to the offset magnitude, where `f0` can absorb
  the error.
- Any assertion that the *absolute* partial frequency ends up at a known target.
  Every existing test checks relative behaviour or a fixed shift.
- Spectrogram-level regression against a reference file.

## 9. Next steps, in order

1. Decide the model. Three candidate designs, in increasing cost:
   a. **Anchor `f0` per voice.** Estimate each voice's spacing from its cleanest
      frames, freeze it, and measure the offset as the deviation of the tracked
      partials from `h*f0_frozen`. This removes the free-parameter escape in 5.1.
      Still ambiguous modulo `f0`, so a seed or an initial guess is required.
   b. **Phase-coherent partial tracker.** Track 3..5 partials with a phase
      vocoder, unwrap each phase, and take the common frequency deviation as the
      offset. Unambiguous apart from index shift, and robust to formants. Hardest
      to get right, most direct measurement.
   c. **Relative de-wobble only.** Do not aim at an absolute target. Track the
      dominant partial, low-pass it over a few seconds, and correct the deviation
      from that local mean. This sidesteps identifiability entirely and is what
      "the pitch should not wave" actually asks for. Cannot restore a constant
      tuning error, only flatten the wobble.
2. Add a test that builds a real detuned voice (`h*f0 + off`), asserts the
   reported offset, and asserts the output partial lands at `h*f0`. Also add the
   two-voice case.
3. Only then tune the smoothing. Cut total lag in 5.4, or state it as a known
   limit.
4. Answer the two-voice question (5.3) with the user before more work. One shift
   cannot serve two offsets.
5. Fix the minor defects in 5.5 and correct the stale README numbers, which
   describe the earlier recording, not `waving.wav`.

## 10. Open questions for the user

1. What should the voices be tuned to? A nominal `f0` per voice, or a known
   carrier, or just "steady, no wobble"?
2. Is the error a constant tuning offset, a slow wobble, or both? The spectrogram
   suggests wobble.
3. Is it acceptable to correct only the currently dominant voice, or must both be
   right at all times? If both, a single global shift is not enough.
4. Is a ~1 s correction lag acceptable, or must the correction track the wobble
   closely?
