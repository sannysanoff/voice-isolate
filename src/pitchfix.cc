#include "pitchfix.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vi {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Periodic Hann: COLA at 75% overlap, and the squared-window sum is 1.5.
void fill_hann(std::vector<float>* w) {
  const int n = static_cast<int>(w->size());
  for (int i = 0; i < n; ++i)
    (*w)[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / n));
}

float median_of(const float* p, int n, float* scratch) {
  const int m = std::min(13, n);
  std::memcpy(scratch, p, sizeof(float) * static_cast<size_t>(m));
  std::nth_element(scratch, scratch + m / 2, scratch + m);
  return scratch[m / 2];
}

double clamp_abs(double v, double limit) {
  return std::max(-limit, std::min(limit, v));
}

}  // namespace

int PitchFixer::analysis_samples_for(const PitchFixOptions& o, int window) {
  int n = std::min(o.analysis_size, window);
  n = std::max(n, std::min(window, 256));
  n &= ~1;
  return std::max(n, 4);
}

PitchFixer::PitchFixer(const PitchFixOptions& options)
    : opt_(options),
      fs_(kPitchFixSampleRate),
      n_shift_(kPitchFixWindow),
      hop_(kPitchFixHop),
      latency_(n_shift_),
      n_an_(analysis_samples_for(options, n_shift_)),
      fft_shift_(n_shift_),
      fft_analysis_(n_an_),
      shift_window_(static_cast<size_t>(n_shift_)),
      analysis_window_(static_cast<size_t>(n_an_)),
      spectrum_(static_cast<size_t>(n_shift_ / 2 + 1)),
      analytic_(static_cast<size_t>(n_shift_ / 2 + 1)),
      hilbert_(static_cast<size_t>(n_shift_)),
      weighted_(static_cast<size_t>(n_shift_)),
      an_spectrum_(static_cast<size_t>(n_an_ / 2 + 1)),
      an_frame_(static_cast<size_t>(n_an_)),
      hist_(static_cast<size_t>(n_shift_), 0.f),
      out_fifo_(static_cast<size_t>(hop_), 0.f),  // one block of prime
      ola_(static_cast<size_t>(n_shift_), 0.f),
      log_mag_(static_cast<size_t>(n_an_ / 2 + 1)),
      raw_log_mag_(static_cast<size_t>(n_an_ / 2 + 1)),
      trend_scratch_(static_cast<size_t>(n_an_ / 2 + 1)) {
  fill_hann(&shift_window_);
  fill_hann(&analysis_window_);
  in_fifo_.reserve(static_cast<size_t>(4 * hop_));
  // A short median of recent measurements rejects ordinary per-frame fit
  // noise.  This fast value also anchors alias unwrapping, so it must follow
  // a real re-lock without delay.
  const int fast_span = std::max(1, static_cast<int>(std::lround(0.15 * fs_ / hop_)));
  recent_.assign(static_cast<size_t>(fast_span), 0.0);
  median_scratch_.assign(static_cast<size_t>(fast_span), 0.0);
  // A second, one-second median drives the applied offset.  With two alternating
  // speakers the comb fit sometimes re-locks to a different harmonic alias for
  // a few frames; a long median drops those short outliers so the correction
  // follows the smooth tuning drift instead of jumping.
  const int slow_span = std::max(1, static_cast<int>(std::lround(0.5 * fs_ / hop_)));
  slow_history_.assign(static_cast<size_t>(slow_span), 0.0);
  slow_scratch_.assign(static_cast<size_t>(slow_span), 0.0);
  // Pre-fill with the anchor: the long median should track the signal from the
  // first block, not sit at zero for a second while its ring fills.
  if (have_offset_) std::fill(slow_history_.begin(), slow_history_.end(), offset_);

  if (!opt_.estimate) {
    offset_ = clamp_abs(opt_.forced_offset_hz, opt_.max_shift_hz);
    have_offset_ = true;
  } else if (opt_.has_seed) {
    offset_ = clamp_abs(opt_.seed_offset_hz, opt_.max_shift_hz);
    have_offset_ = true;
  }
  stats_.offset_hz = offset_;
}

void PitchFixer::reset() {
  std::fill(hist_.begin(), hist_.end(), 0.f);
  std::fill(ola_.begin(), ola_.end(), 0.f);
  in_fifo_.clear();
  out_fifo_.assign(static_cast<size_t>(hop_), 0.f);
  recent_filled_ = 0;
  slow_filled_ = 0;
  slow_pos_ = 0;
  have_slow_ = false;
  recent_pos_ = 0;
  phase_ = 0.0;
  have_raw_ = false;
  raw_anchor_ = 0.0;
  have_offset_ = opt_.has_seed || !opt_.estimate;
  slow_offset_ = offset_;
  have_slow_ = have_offset_;
  offset_ = !opt_.estimate ? clamp_abs(opt_.forced_offset_hz, opt_.max_shift_hz)
           : opt_.has_seed ? clamp_abs(opt_.seed_offset_hz, opt_.max_shift_hz)
                           : 0.0;
  stats_ = PitchFixStats{};
  stats_.offset_hz = offset_;
}

void PitchFixer::process(const float* in, int count, float* out) {
  if (count <= 0) return;
  in_fifo_.insert(in_fifo_.end(), in, in + count);

  size_t consumed = 0;
  while (in_fifo_.size() - consumed >= static_cast<size_t>(hop_)) {
    if (n_shift_ > hop_)
      std::memmove(hist_.data(), hist_.data() + hop_,
                   sizeof(float) * static_cast<size_t>(n_shift_ - hop_));
    std::memcpy(hist_.data() + (n_shift_ - hop_), in_fifo_.data() + consumed,
                sizeof(float) * static_cast<size_t>(hop_));
    consumed += static_cast<size_t>(hop_);
    if (opt_.estimate) analyse(hist_.data() + (n_shift_ - n_an_), n_an_);
    shift_block();
  }
  if (consumed > 0) in_fifo_.erase(in_fifo_.begin(), in_fifo_.begin() + consumed);

  // The queue is primed with one block, and each new block arrives one block
  // before it is needed, so it never runs dry in steady state.
  const int take = std::min(count, static_cast<int>(out_fifo_.size()));
  std::memcpy(out, out_fifo_.data(), sizeof(float) * static_cast<size_t>(take));
  out_fifo_.erase(out_fifo_.begin(), out_fifo_.begin() + take);
  // Only the very end of a stream can be short: the newest latency samples
  // have not been through the window yet.  finish() emits them.
  for (int i = take; i < count; ++i) out[i] = 0.f;
}

int PitchFixer::finish(float* out, int capacity) {
  const int n = std::min(latency_, capacity);
  if (n <= 0) return 0;
  std::vector<float> zeros(static_cast<size_t>(n), 0.f);
  float* sink = out;
  std::vector<float> scratch;
  if (out == nullptr) {
    scratch.resize(static_cast<size_t>(n));
    sink = scratch.data();
  }
  process(zeros.data(), n, sink);
  return n;
}

void PitchFixer::shift_block() {
  const int bins = n_shift_ / 2 + 1;
  for (int i = 0; i < n_shift_; ++i) weighted_[i] = hist_[i] * shift_window_[i];
  fft_shift_.rfft(weighted_.data(), spectrum_.data());

  // H{x} is real with one-sided spectrum -i X; DC and Nyquist are dropped so
  // the analytic signal stays exactly real after taking the real part.
  analytic_[0] = std::complex<float>(0.f, 0.f);
  for (int k = 1; k < bins - 1; ++k)
    analytic_[k] = std::complex<float>(spectrum_[k].imag(), -spectrum_[k].real());
  analytic_[bins - 1] = std::complex<float>(0.f, 0.f);
  fft_shift_.irfft(analytic_.data(), hilbert_.data());

  // y = x cos(theta) + H{x} sin(theta) translates down by the offset; the
  // phase is accumulated so a changing offset does not click at block edges.
  const double dphi = 2.0 * kPi * (opt_.estimate ? slow_offset_ : offset_) / static_cast<double>(fs_);
  double theta = phase_;
  for (int i = 0; i < n_shift_; ++i, theta += dphi) {
    const float c = static_cast<float>(std::cos(theta));
    const float s = static_cast<float>(std::sin(theta));
    ola_[i] += (weighted_[i] * c + hilbert_[i] * s) * shift_window_[i];
  }
  // The next block starts one hop later, not one window later.
  phase_ = std::fmod(phase_ + static_cast<double>(hop_) * dphi, 2.0 * kPi);

  const float cola = 2.f / 3.f;
  for (int i = 0; i < hop_; ++i) out_fifo_.push_back(ola_[i] * cola);
  std::memmove(ola_.data(), ola_.data() + hop_,
               sizeof(float) * static_cast<size_t>(n_shift_ - hop_));
  std::memset(ola_.data() + (n_shift_ - hop_), 0, sizeof(float) * static_cast<size_t>(hop_));
}

void PitchFixer::analyse(const float* frame, int count) {
  const int bins = count / 2 + 1;
  const double df = static_cast<double>(fs_) / count;
  const double f_lo = 110.0;
  const double f_hi = std::min(3600.0, 0.45 * fs_);

  for (int i = 0; i < count; ++i) an_frame_[i] = frame[i] * analysis_window_[i];
  fft_analysis_.rfft(an_frame_.data(), an_spectrum_.data());

  // Work on the spectral envelope, not the raw level: a log spectrum with the
  // 300 Hz trend removed makes the comb contrast independent of the vowel and
  // of the low-frequency tilt, which otherwise biases the search.
  for (int k = 0; k < bins; ++k)
    raw_log_mag_[k] = std::log(std::abs(an_spectrum_[k]) + 1e-6f);
  const int half = 6;  // 13 bins, ~300 Hz at 2048/48 kHz
  for (int k = 0; k < bins; ++k) {
    const int lo = std::max(0, k - half);
    const int hi = std::min(bins - 1, k + half);
    trend_scratch_[k] = median_of(raw_log_mag_.data() + lo, hi - lo + 1, median_buffer_);
  }
  for (int k = 0; k < bins; ++k)
    log_mag_[k] = std::max(-3.f, std::min(6.f, raw_log_mag_[k] - trend_scratch_[k]));

  auto sample = [&](double f) {
    const double x = f / df;
    const int i = static_cast<int>(x);
    if (i < 1 || i + 1 >= bins) return 0.0;
    const double t = x - i;
    return log_mag_[i] * (1.0 - t) + log_mag_[i + 1] * t;
  };
  // Contrast between harmonic positions and their midpoints: the midpoint term
  // is what rejects sub-multiples of the true spacing.
  auto contrast = [&](double f0, double delta) {
    double on = 0.0, mid = 0.0;
    int n_on = 0, n_mid = 0;
    for (int h = 1; h <= 8; ++h) {
      double f = h * f0 + delta;
      if (f >= f_lo && f <= f_hi) { on += sample(f); ++n_on; }
      f = (h + 0.5) * f0 + delta;
      if (f >= f_lo && f <= f_hi) { mid += sample(f); ++n_mid; }
    }
    if (n_on < 4 || n_mid < 3) return -1e9;
    return on / n_on - mid / n_mid;
  };

  const double dmax = opt_.max_shift_hz;
  double best = -1e9, best_f0 = 0.0, best_d = 0.0;
  // A spacing and its offset are degenerate: f0 = 250 with delta = -250 scores
  // the same comb as f0 = 200 with delta = +50.  The search therefore runs over
  // the whole range every frame.  Latching a spacing across frames force-fits
  // the frame after a speaker change onto the previous speaker's grid.
  for (double f0 = opt_.f0_min_hz; f0 <= opt_.f0_max_hz + 1e-9; f0 += 2.0)
    for (double d = -dmax; d <= dmax + 1e-9; d += 4.0) {
      const double s = contrast(f0, d);
      if (s > best) { best = s; best_f0 = f0; best_d = d; }
    }
  for (double f0 = best_f0 - 4.0; f0 <= best_f0 + 4.0 + 1e-9; f0 += 1.0) {
    if (f0 < 0.5 * opt_.f0_min_hz || f0 > 2.0 * opt_.f0_max_hz) continue;
    for (double d = best_d - 4.0; d <= best_d + 4.0 + 1e-9; d += 1.0) {
      const double s = contrast(f0, d);
      if (s > best) { best = s; best_f0 = f0; best_d = d; }
    }
  }

  stats_.f0_hz = best_f0;
  stats_.confidence = best;
  if (std::getenv("PITCHFIX_DEBUG")) std::fprintf(stderr, "[grid] f0=%.1f delta=%.1f score=%.3f\n", best_f0, best_d, best);
  stats_.updated = false;
  if (best < opt_.min_confidence) return;

  // Prominent partials, then a least-squares lattice fit.  The grid above is
  // limited by its 2 Hz step and by interpolation; the fit is exact for clean
  // partials and averages the rest.
  peak_hz_.clear();
  peak_w_.clear();
  for (int k = 2; k < bins - 2; ++k) {
    if (!(log_mag_[k] >= log_mag_[k - 1] && log_mag_[k] > log_mag_[k + 1])) continue;
    if (log_mag_[k] < 0.6f) continue;
    const double f = k * df;
    if (f < f_lo || f > f_hi) continue;
    float valley = log_mag_[k];
    for (int j = std::max(0, k - 5); j < k; ++j) valley = std::min(valley, log_mag_[j]);
    for (int j = k + 1; j <= std::min(bins - 1, k + 5); ++j)
      valley = std::min(valley, log_mag_[j]);
    if (log_mag_[k] - valley < 0.4f) continue;
    // Exact for an isolated Hann peak: the three-bin magnitude interpolation.
    const double a = std::abs(an_spectrum_[k - 1]);
    const double b = std::abs(an_spectrum_[k]);
    const double c = std::abs(an_spectrum_[k + 1]);
    const double den = a + 2.0 * b + c;
    const double frac = den > 0.0 ? 2.0 * (c - a) / den : 0.0;
    peak_hz_.push_back((k + frac) * df);
    peak_w_.push_back(std::min(std::max(static_cast<double>(log_mag_[k]), 0.5), 4.0));
  }

  // The 2 Hz grid cannot tell a spacing of 250 Hz from an offset of 50 Hz on a
  // 200 Hz spacing: both put the peaks on the same comb.  The peak fit can,
  // because it uses the true partial positions.  Re-lattice the peaks with the
  // spacing the frame suggests, so a grid alias cannot survive.
  double f0 = best_f0, delta = best_d;
  if (!fit_lattice(&f0, &delta)) {
    stats_.updated = false;
    return;
  }
  // The peak fit refines the spacing on clean partials, this frame only.  No
  // state carries over: after a speaker change the old spacing is worthless.
  stats_.f0_hz = f0;
  update_offset(delta, f0, best);
}

bool PitchFixer::fit_lattice(double* f0, double* delta) {
  if (peak_hz_.size() < 3) return false;
  int used = 0;
  for (int it = 0; it < 4; ++it) {
    double sw = 0.0, sh = 0.0, sp = 0.0, shh = 0.0, shp = 0.0;
    int n = 0;
    for (size_t i = 0; i < peak_hz_.size(); ++i) {
      const double p = peak_hz_[i];
      const double h = std::round((p - *delta) / *f0);
      if (h < 1.0 || h > 14.0) continue;
      if (std::fabs(p - (h * *f0 + *delta)) > 0.22 * *f0) continue;
      const double w = peak_w_[i];
      sw += w;
      sh += w * h;
      sp += w * p;
      shh += w * h * h;
      shp += w * h * p;
      ++n;
    }
    used = n;
    if (n < 3) return false;
    const double det = shh * sw - sh * sh;
    if (std::fabs(det) < 1e-12) return false;
    const double nf0 = (shp * sw - sh * sp) / det;
    const double nd = (shh * sp - sh * shp) / det;
    // A fit that doubled or halved the spacing latched onto a sub-harmonic.
    if (!(nf0 > 0.5 * *f0 && nf0 < 2.0 * *f0)) return false;
    *f0 = nf0;
    *delta = nd;
  }
  return used >= 3;
}

void PitchFixer::update_offset(double measured, double f0, double confidence) {
  (void)confidence;
  // The measurement is only defined modulo f0: shifting by offset, or by
  // offset - m*f0, puts the comb on an exact harmonic lattice either way.  Stay
  // on the branch nearest the previous *raw* measurement.  Anchoring on the
  // eased output instead would compare this frame against a value that lags by
  // the whole memory (~0.9 s); on a sweeping carrier that is more than f0/2, so
  // the unwrap slips a branch every few frames and the correction silently
  // under-tracks.
  double target = measured;
  if (f0 > 1e-6) {
    const double reference = have_raw_ ? raw_anchor_ : (have_offset_ ? offset_ : 0.0);
    target -= f0 * std::round((target - reference) / f0);
  }

  target = clamp_abs(target, opt_.max_shift_hz);
  stats_.raw_offset_hz = target;

  auto median_of = [](const std::vector<double>& src, int n, std::vector<double>& scratch) {
    std::copy(src.begin(), src.begin() + n, scratch.begin());
    std::nth_element(scratch.begin(), scratch.begin() + n / 2, scratch.begin() + n);
    return scratch[static_cast<size_t>(n / 2)];
  };

  recent_[recent_pos_] = target;
  recent_pos_ = (recent_pos_ + 1) % recent_.size();
  recent_filled_ = std::min<int>(recent_filled_ + 1, static_cast<int>(recent_.size()));
  const double fast = median_of(recent_, recent_filled_, median_scratch_);
  // Anchor the next unwrap on this short median: robust against a single noisy
  // frame, and only 0.15 s behind, which at a real carrier sweep is far less
  // than half a spacing.  Anchoring on one raw frame lets every frame error
  // through and the branch walks away by whole spacings.
  raw_anchor_ = fast;
  have_raw_ = true;

  // The applied correction is the one-pole easing driven by the long median.
  // The anchor that picks the alias branch uses the same value, so the two can
  // never drift apart.
  if (!have_offset_) {
    offset_ = clamp_abs(fast, opt_.max_shift_hz);
    have_offset_ = true;
  }

  slow_history_[slow_pos_] = fast;
  slow_pos_ = (slow_pos_ + 1) % slow_history_.size();
  slow_filled_ = std::min<int>(slow_filled_ + 1, static_cast<int>(slow_history_.size()));
  double slow_target = median_of(slow_history_, slow_filled_, slow_scratch_);
  slow_target = clamp_abs(slow_target, opt_.max_shift_hz);
  if (!have_slow_) { offset_ = target; have_slow_ = true; }
  else {
    // Speed is proportional to the remaining distance: the correction closes
    // 63% of the gap per tau, so it arrives fast and eases in.
    const double alpha = opt_.tau_s > 0.0
        ? 1.0 - std::exp(-static_cast<double>(hop_) / (static_cast<double>(fs_) * opt_.tau_s))
        : 1.0;
    offset_ = clamp_abs(offset_ + alpha * (slow_target - offset_), opt_.max_shift_hz);
  }
  slow_offset_ = offset_;
  stats_.offset_hz = offset_;
  stats_.updated = true;
}

}  // namespace vi
