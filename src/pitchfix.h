// Harmonic pitch fixer for single-sideband recordings.
//
// The input is a voice whose partials sit at h*f0 + offset instead of h*f0:
// a lower-sideband demodulator tuned to the wrong carrier shifts every
// component by the same amount, so the harmonics are evenly spaced but the
// whole comb is detuned.  The fix is therefore a frequency *translation* by
// -offset, not a resampling: resampling would scale the spacing as well.
//
// The offset is estimated from the signal itself (harmonic comb fit), smoothed
// with a one-pole easing (distance-proportional speed, tau = 0.5 s by default)
// and applied through a 4096-sample overlap-add single-sideband shifter with a
// 1024-sample hop.  The rate is fixed at 48 kHz and all sizes are in samples,
// so the FFT and Hann window keep clean power-of-two shapes.  The delay is one
// window (4096 samples).  Bands pushed past 0 Hz are lost; that is accepted.
#pragma once

#include <complex>
#include <vector>

#include "fft.h"

namespace vi {

// The algorithm is fixed to the rate of this recording.  Sizes are in samples
// so the analysis window, the overlap and the delay stay powers of two; deriving
// them from a time in milliseconds produced odd FFT and Hann lengths.
constexpr int kPitchFixSampleRate = 48000;
constexpr int kPitchFixWindow = 4096;
constexpr int kPitchFixHop = 1024;

struct PitchFixOptions {
  // Largest correction the filter will apply, either way.  Also the search
  // range of the offset estimate.
  double max_shift_hz = 300.0;
  // One-pole easing time constant: the correction closes 63% of the remaining
  // distance to a new target in tau_s, and slows down as it approaches.
  double tau_s = 0.5;
  // Estimate the offset per frame (true) or apply `forced_offset_hz` (false).
  bool estimate = true;
  double forced_offset_hz = 0.0;
  // Seed for the estimate.  The offset is only observable modulo the harmonic
  // spacing, so any alias m*f0 apart aligns the comb; the estimate follows the
  // branch nearest this seed.  has_seed=false starts on the smallest alias.
  bool has_seed = false;
  double seed_offset_hz = 0.0;
  // Harmonic spacing search range.
  double f0_min_hz = 80.0;
  double f0_max_hz = 320.0;
  // Frames scoring below this comb contrast are ignored (unvoiced, silence).
  double min_confidence = 0.20;
  // Estimator window in samples, clamped to the shifter window.
  int analysis_size = 2048;
};

// One measurement, for reporting and tests.
struct PitchFixStats {
  double offset_hz = 0.0;      // smoothed offset currently applied
  double raw_offset_hz = 0.0;  // this frame's measurement before easing
  double f0_hz = 0.0;          // harmonic spacing of this frame
  double confidence = 0.0;     // comb contrast of this frame
  bool updated = false;        // this frame passed the confidence gate
};

class PitchFixer {
 public:
  explicit PitchFixer(const PitchFixOptions& options);
  PitchFixer(const PitchFixer&) = delete;
  PitchFixer& operator=(const PitchFixer&) = delete;

  const PitchFixOptions& options() const { return opt_; }
  static constexpr int sample_rate() { return kPitchFixSampleRate; }
  // Delay from an input sample to the output sample carrying it.
  int latency_samples() const { return latency_; }
  int window_samples() const { return n_shift_; }
  int block_samples() const { return hop_; }

  // Restarts the stream.  The delay line restarts as leading silence.
  void reset();

  // Consumes `count` input samples and writes exactly `count` output samples.
  // Output is delayed by latency_samples(); earlier samples are zero.
  void process(const float* in, int count, float* out);

  // Pushes the delay line through with silence and returns the samples still
  // held inside, up to `capacity`.  A file-sized run that calls this produces
  // count + latency_samples() outputs; skip it to get exactly `count`.
  int finish(float* out, int capacity);

  const PitchFixStats& stats() const { return stats_; }

 private:
  static int analysis_samples_for(const PitchFixOptions& o, int window);
  void analyse(const float* frame, int count);
  void shift_block();
  bool fit_lattice(double* f0, double* delta);
  void update_offset(double measured, double f0, double confidence);

  PitchFixOptions opt_;
  int fs_;
  int n_shift_;   // shifter window
  int hop_;       // n_shift_/4, the block size
  int latency_;   // n_shift_, the delay through the window
  int n_an_;      // estimator window

  Fft fft_shift_;
  Fft fft_analysis_;
  std::vector<float> shift_window_;
  std::vector<float> analysis_window_;
  std::vector<std::complex<float>> spectrum_;
  std::vector<std::complex<float>> analytic_;
  std::vector<float> hilbert_;
  std::vector<float> weighted_;
  std::vector<std::complex<float>> an_spectrum_;
  std::vector<float> an_frame_;

  std::vector<float> hist_;      // last n_shift_ input samples
  std::vector<float> in_fifo_;   // input not yet packed into blocks
  std::vector<float> out_fifo_;  // processed output not yet delivered
  std::vector<float> ola_;       // overlap-add accumulator, n_shift_

  // Estimator workspace.
  std::vector<float> log_mag_;     // trend-removed log magnitude
  std::vector<float> raw_log_mag_;
  std::vector<float> trend_scratch_;
  float median_buffer_[13] = {};
  // Peak list of the current frame, and the robust median of recent fits.
  std::vector<double> peak_hz_;
  std::vector<double> peak_w_;
  std::vector<double> recent_;        // fast history, for alias unwrapping
  std::vector<double> median_scratch_;
  size_t recent_pos_ = 0;
  int recent_filled_ = 0;
  std::vector<double> slow_history_;  // one second, for the applied offset
  size_t slow_pos_ = 0;
  int slow_filled_ = 0;
  std::vector<double> slow_scratch_;
  double slow_offset_ = 0.0;
  bool have_slow_ = false;

  double offset_ = 0.0;
  bool have_offset_ = false;
  double raw_anchor_ = 0.0;   // previous frame's unwrapped measurement
  bool have_raw_ = false;
  double phase_ = 0.0;  // accumulated modulator phase, radians
  PitchFixStats stats_;
};

}  // namespace vi
