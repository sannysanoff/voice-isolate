// Tests for the SSB pitch-correction filter: translation accuracy, offset
// tracking, easing, clamping, latency and streaming determinism.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "pitchfix.h"

namespace {

int g_failures = 0;

void expect(bool condition, const char* what) {
  std::printf("  %-58s %s\n", what, condition ? "ok" : "FAIL");
  if (!condition) ++g_failures;
}

constexpr double kPi = 3.14159265358979323846;

// Harmonic voice: partial h at h*f0 + offset, with a gentle 1/h envelope.
std::vector<float> make_comb(int fs, double seconds, double f0, double offset,
                             double fade_at = -1.0, double offset_after = 0.0) {
  const int n = static_cast<int>(seconds * fs);
  std::vector<float> x(static_cast<size_t>(n), 0.f);
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / fs;
    const double off = (fade_at >= 0.0 && t >= fade_at) ? offset_after : offset;
    double v = 0.0;
    for (int h = 1; h <= 8; ++h) {
      const double f = h * f0 + off;
      if (f >= fs * 0.45) break;
      v += std::cos(2.0 * kPi * f * t + 0.7 * h) / h;
    }
    x[static_cast<size_t>(i)] = static_cast<float>(0.2 * v);
  }
  return x;
}

std::vector<float> run(const std::vector<float>& x, const vi::PitchFixOptions& opt,
                       int block, std::vector<double>* trace = nullptr) {
  vi::PitchFixer fixer(opt);
  std::vector<float> out(x.size(), 0.f);
  std::vector<float> chunk(static_cast<size_t>(block));
  for (size_t pos = 0; pos < x.size(); pos += static_cast<size_t>(block)) {
    const int n = static_cast<int>(std::min<size_t>(block, x.size() - pos));
    if (n != block) chunk.resize(static_cast<size_t>(n));
    fixer.process(x.data() + pos, n, chunk.data());
    std::copy(chunk.begin(), chunk.begin() + n, out.begin() + pos);
    if (trace) trace->push_back(fixer.stats().offset_hz);
  }
  return out;
}

// Hann-windowed DFT amplitude of one frequency over [from, to).
double amplitude(const std::vector<float>& x, double f, int fs, size_t from, size_t to) {
  double re = 0.0, im = 0.0, wsum = 0.0;
  const double w = 2.0 * kPi * f / fs;
  for (size_t i = from; i < to; ++i) {
    const double win = 0.5 - 0.5 * std::cos(2.0 * kPi * (i - from) / (to - from));
    re += x[i] * win * std::cos(w * i);
    im += x[i] * win * std::sin(w * i);
    wsum += win;
  }
  return 2.0 * std::sqrt(re * re + im * im) / wsum;
}

// Energy on the h*f0 lattice over energy halfway between, in dB.  A detuned
// comb scores low, an aligned one high; this is what the filter has to fix.
double alignment(const std::vector<float>& x, double f0, int fs, size_t from, size_t to) {
  double on = 0.0, mid = 0.0;
  for (int h = 1; h <= 8; ++h) {
    const double f = h * f0;
    if (f >= 0.45 * fs) break;
    on += amplitude(x, f, fs, from, to) / h;
    mid += amplitude(x, f + 0.5 * f0, fs, from, to) / h;
  }
  return 20.0 * std::log10((on + 1e-9) / (mid + 1e-9));
}

void test_dc_and_latency() {
  std::printf("dc, latency\n");
  const int fs = 48000;
  vi::PitchFixOptions opt;
  opt.estimate = false;
  opt.forced_offset_hz = 0.0;
  vi::PitchFixer fixer(opt);
  const int latency = fixer.latency_samples();
  const int n = fs;  // 1 s
  std::vector<float> in(static_cast<size_t>(n), 0.25f);
  std::vector<float> out(static_cast<size_t>(n), 0.f);
  fixer.process(in.data(), n, out.data());

  expect(latency == 4096, "latency is one 4096-sample window");
  double before = 0.0, after = 0.0;
  for (int i = 0; i < latency; ++i) before = std::max(before, std::fabs(static_cast<double>(out[i])));
  for (int i = latency + 200; i < n - 200; ++i) after = std::max(after, std::fabs(static_cast<double>(out[i]) - 0.25));
  expect(before == 0.0, "output is silent before the latency");
  expect(after < 1e-4, "rest of the DC signal passes at unity gain");
}

void test_translation() {
  std::printf("fixed translation\n");
  const int fs = 48000;
  const double f0 = 200.0, offset = 100.0;
  const std::vector<float> x = make_comb(fs, 2.0, f0, offset);

  vi::PitchFixOptions opt;
  opt.estimate = false;
  opt.forced_offset_hz = offset;
  const std::vector<float> y = run(x, opt, 480);

  const size_t from = static_cast<size_t>(0.6 * fs), to = static_cast<size_t>(1.9 * fs);
  const double at_new = amplitude(y, f0, fs, from, to);
  const double at_old = amplitude(y, f0 + offset, fs, from, to);
  const double at_spacing = amplitude(y, 2.0 * f0, fs, from, to);
  std::printf("    |200| %.4f  |300| %.4f  |400| %.4f\n", at_new, at_old, at_spacing);
  expect(at_new > 0.05, "fundamental moved to h*f0");
  expect(at_old < 0.15 * at_new, "old detuned position is 16 dB down or more");
  // The test comb has a 1/h envelope, so h=2 must land at half the fundamental.
  expect(std::fabs(at_spacing - 0.5 * at_new) < 0.15 * at_new, "spacing unchanged: h*f0 lands on h*f0");
  expect(alignment(y, f0, fs, from, to) > alignment(x, f0, fs, from, to) + 20.0,
         "harmonic alignment improves by more than 20 dB");
}

void test_estimate() {
  std::printf("offset estimation\n");
  const int fs = 48000;
  const double f0 = 190.0, offset = 100.0;
  const std::vector<float> x = make_comb(fs, 2.5, f0, offset);
  vi::PitchFixOptions opt;
  const std::vector<float> y = run(x, opt, 480);

  const size_t from = static_cast<size_t>(1.4 * fs), to = static_cast<size_t>(2.4 * fs);
  // The estimate is defined modulo f0; any alias aligns the comb.  The applied
  // correction should be ~100 Hz from one of the two nearest aliases.
  vi::PitchFixer fixer(opt);
  std::vector<float> tmp(480);
  std::vector<double> trace;
  for (size_t pos = 0; pos < x.size(); pos += 480) {
    const int n = static_cast<int>(std::min<size_t>(480, x.size() - pos));
    fixer.process(x.data() + pos, n, tmp.data());
    trace.push_back(fixer.stats().offset_hz);
  }
  const double est = trace.back();
  const double alias_error = std::fabs(std::fabs(est) - offset);
  const double alias_error2 = std::fabs(std::fabs(est) - std::fabs(offset - f0));
  std::printf("    estimate %.1f Hz (|e|-100 = %.1f, |e|-90 = %.1f)\n", est, alias_error,
              alias_error2);
  expect(std::fabs(est) <= 300.0 + 1e-9, "estimate stays inside max-shift");
  expect(alias_error < 15.0 || alias_error2 < 15.0,
         "estimate locks to the +100 Hz error (up to a f0 alias)");
  expect(alignment(y, f0, fs, from, to) > alignment(x, f0, fs, from, to) + 10.0,
         "estimated correction aligns the comb by more than 10 dB");
  // With an aliased estimate the corrected comb can sit a whole spacing away,
  // so compare the old detuned positions with the harmonic lattice itself.
  const double at_old = amplitude(y, f0 + offset, fs, from, to);
  const double on_lattice = amplitude(y, 2.0 * f0, fs, from, to) +
                            amplitude(y, 3.0 * f0, fs, from, to);
  expect(at_old < 0.25 * on_lattice, "steady state leaves little energy at the old positions");
}

void test_easing() {
  std::printf("easing (0.5 s one-pole)\n");
  const int fs = 48000;
  const double f0 = 200.0;
  // Step by less than half a spacing so the alias branch stays put.
  const std::vector<float> x = make_comb(fs, 3.0, f0, 50.0, 1.0, 130.0);
  vi::PitchFixOptions opt;
  opt.tau_s = 0.5;
  opt.has_seed = true;
  opt.seed_offset_hz = 50.0;
  std::vector<double> trace;
  run(x, opt, 480, &trace);

  const int block = 480;
  auto at = [&](double t) {
    const size_t k = std::min(trace.size() - 1, static_cast<size_t>(t * fs / block));
    return trace[k];
  };
  const double start = at(0.95);
  // The correction is eased through a 0.5 s median, so it lags the raw fit by
  // half the median window.  tau is a time constant, not a crossing time: the
  // value tau after the step sits between start and the 63% level.
  const double mid = at(1.0 + 0.5);     // half a tau after the step
  const double late = at(1.0 + 2.5);    // ~99.3%
  std::printf("    offset t=0.95 %.1f  t=1.5 %.1f  t=3.5 %.1f (target 130)\n", start, mid,
              late);
  double max_step = 0.0;
  for (size_t i = 1; i < trace.size(); ++i)
    max_step = std::max(max_step, std::fabs(trace[i] - trace[i - 1]));
  std::printf("    largest one-block change %.2f Hz\n", max_step);
  expect(std::fabs(start - 50.0) < 12.0, "starts from the first offset");
  expect(mid > start + 5.0 && mid < 50.0 + 0.63 * 80.0 + 8.0,
         "eases up after the step, no faster than tau");
  expect(std::fabs(late - 130.0) < 8.0, "settles on the new offset");
  expect(max_step < 20.0, "no step bigger than 20 Hz per 10 ms block");
}

void test_clamp() {
  std::printf("max-shift clamp\n");
  const int fs = 48000;
  vi::PitchFixOptions opt;
  opt.estimate = false;
  opt.forced_offset_hz = 400.0;
  opt.max_shift_hz = 150.0;
  vi::PitchFixer fixer(opt);
  expect(std::fabs(fixer.stats().offset_hz - 150.0) < 1e-9, "forced offset clamps to max-shift");

  const std::vector<float> x = make_comb(fs, 1.5, 200.0, 400.0);
  const std::vector<float> y = run(x, opt, 480);
  const size_t from = static_cast<size_t>(0.5 * fs), to = static_cast<size_t>(1.4 * fs);
  const double moved = amplitude(y, 600.0 - 150.0, fs, from, to);
  const double stayed = amplitude(y, 600.0, fs, from, to);
  std::printf("    |450| %.4f  |600| %.4f\n", moved, stayed);
  expect(moved > 0.05 && stayed < 0.3 * moved, "correction is exactly max-shift, no more");
}

void test_streaming_determinism() {
  std::printf("streaming determinism\n");
  const int fs = 48000;
  const std::vector<float> x = make_comb(fs, 1.2, 180.0, 90.0);
  vi::PitchFixOptions opt;
  const std::vector<float> a = run(x, opt, 1);
  const std::vector<float> b = run(x, opt, 37);
  const std::vector<float> c = run(x, opt, 480);
  const std::vector<float> d = run(x, opt, 5000);
  bool same = a.size() == b.size() && a.size() == c.size() && a.size() == d.size();
  for (size_t i = 0; same && i < a.size(); ++i)
    same = a[i] == b[i] && a[i] == c[i] && a[i] == d[i];
  expect(same, "block size does not change the output");
}

// Two speakers taking turns, each with its own harmonic spacing and its own
// carrier error, plus an optional shared carrier sweep.  Built by phase
// accumulation: cos(2*pi*f(t)*t) with a moving f(t) gives the wrong
// instantaneous frequency, so the sweep has to be integrated per harmonic.
std::vector<float> make_two_speakers(int fs, double turn_s, int turns, double f0a,
                                     double offa, double f0b, double offb,
                                     double sweep_hz, double sweep_period_s) {
  const int H = 10;
  const int n = static_cast<int>(turn_s * turns * fs);
  std::vector<float> x(static_cast<size_t>(n), 0.f);
  double ph[2][H] = {};
  const double f0v[2] = {f0a, f0b};
  const double ofv[2] = {offa, offb};
  const double ramp = 0.02;
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / fs;
    const double sweep =
        sweep_hz * std::sin(2.0 * kPi * t / sweep_period_s);
    double w[2] = {0.0, 0.0};
    for (int k = 0; k < turns; ++k) {
      const int s = k % 2;
      const double lo = k * turn_s, hi = (k + 1) * turn_s;
      double g = 0.0;
      if (k > 0 && t >= lo - ramp && t < lo + ramp)
        g = 0.5 + 0.5 * (t - lo) / ramp;
      else if (k + 1 < turns && t >= hi - ramp && t < hi + ramp)
        g = 0.5 - 0.5 * (t - hi) / ramp;
      else if (t >= lo + ramp && t <= hi - ramp)
        g = 1.0;
      w[s] = std::max(w[s], g);
    }
    double v = 0.0;
    for (int s = 0; s < 2; ++s) {
      if (w[s] <= 0.0) continue;
      double acc = 0.0;
      for (int h = 1; h <= H; ++h) {
        const double f = h * f0v[s] + ofv[s] + sweep;
        if (f <= 0.0 || f >= 0.45 * fs) continue;
        ph[s][h - 1] += 2.0 * kPi * f / fs;
        acc += std::cos(ph[s][h - 1]) / h;
      }
      v += w[s] * acc;
    }
    x[static_cast<size_t>(i)] = static_cast<float>(0.2 * v);
  }
  return x;
}

// Distance between an applied offset and an expected one, allowing for the
// alias: the offset is only observable modulo the spacing.
double alias_error(double applied, double expected, double f0) {
  double e = std::fabs(applied - expected);
  e = e - f0 * std::round(e / f0);
  return std::fabs(e);
}

void test_two_speakers() {
  std::printf("two speakers, turns 1-2-1-2\n");
  const int fs = 48000;
  const double turn = 2.0;
  const int turns = 4;
  const double f0[2] = {170.0, 115.0};
  const double off[2] = {30.0, 70.0};
  const std::vector<float> x =
      make_two_speakers(fs, turn, turns, f0[0], off[0], f0[1], off[1], 0.0, 11.0);

  vi::PitchFixOptions opt;
  opt.max_shift_hz = 400.0;
  std::vector<double> trace;
  const std::vector<float> y = run(x, opt, 480, &trace);

  // Middle of each turn, past the crossfade and past the easing settle.
  double worst = 0.0;
  for (int k = 0; k < turns; ++k) {
    const int s = k % 2;
    const size_t idx = static_cast<size_t>((k * turn + 1.4) * fs / 480);
    const double applied = trace[std::min(idx, trace.size() - 1)];
    const double e = alias_error(applied, off[s], f0[s]);
    worst = std::max(worst, e);
    std::printf("    turn %d speaker %d: applied %+7.1f  expected %+7.1f  alias err %5.1f\n",
                k, s + 1, applied, off[s], e);

    // The correction must align this speaker's own harmonics, not the other's.
    // The correction eases into a new speaker, so measure the settled part of
    // the turn, not the crossing.
    const size_t from = static_cast<size_t>((k * turn + 1.0) * fs);
    const size_t to = static_cast<size_t>((k * turn + 1.9) * fs);
    const double gain = alignment(y, f0[s], fs, from, to) -
                        alignment(x, f0[s], fs, from, to);
    std::printf("      alignment gain on the %.0f Hz lattice: %+6.1f dB\n", f0[s], gain);
    expect(gain > 12.0, "speaker's own comb is aligned during its turn");
  }
  expect(worst < 15.0, "both speakers' offsets are tracked within 15 Hz");
}

void test_two_speakers_sweep() {
  std::printf("two speakers with a carrier sweep\n");
  const int fs = 48000;
  const double turn = 3.0;
  const int turns = 4;
  const double f0[2] = {170.0, 115.0};
  const double off[2] = {30.0, 70.0};
  const double amp = 50.0, period = 11.0;
  const std::vector<float> x = make_two_speakers(fs, turn, turns, f0[0], off[0], f0[1],
                                                 off[1], amp, period);
  vi::PitchFixOptions opt;
  opt.max_shift_hz = 400.0;
  std::vector<double> trace;
  run(x, opt, 480, &trace);

  double worst = 0.0;
  for (int k = 0; k < turns; ++k) {
    const int s = k % 2;
    const double t = k * turn + 2.2;
    // The correction is a smoothed copy of the error, so it lags by about the
    // memory length.  Compare with the sweep one second earlier.
    const double sweep = amp * std::sin(2.0 * kPi * (t - 1.0) / period);
    const double applied = trace[std::min(static_cast<size_t>(t * fs / 480),
                                          trace.size() - 1)];
    const double e = alias_error(applied, off[s] + sweep, f0[s]);
    worst = std::max(worst, e);
    std::printf("    turn %d: applied %+7.1f  expected %+7.1f  alias err %5.1f\n", k,
                applied, off[s] + sweep, e);
  }
  std::printf("    worst %5.1f Hz after allowing the 1 s memory lag\n", worst);
  expect(worst < 20.0, "sweeping offsets are followed within the memory lag");
}

void test_silence() {
  std::printf("silence\n");
  const int fs = 48000;
  vi::PitchFixOptions opt;
  vi::PitchFixer fixer(opt);
  std::vector<float> in(static_cast<size_t>(fs / 2), 0.f), out(in.size());
  fixer.process(in.data(), static_cast<int>(in.size()), out.data());
  bool clean = true;
  for (float v : out) clean = clean && std::isfinite(v) && std::fabs(v) < 1e-6f;
  expect(clean, "silence produces silence");
  expect(std::isfinite(fixer.stats().offset_hz), "offset stays finite");
}

}  // namespace

int main() {
  test_dc_and_latency();
  test_translation();
  test_estimate();
  test_easing();
  test_clamp();
  test_streaming_determinism();
  test_two_speakers();
  test_two_speakers_sweep();
  test_silence();
  std::printf("\n%s\n", g_failures == 0 ? "all tests passed" : "FAILURES");
  return g_failures == 0 ? 0 : 1;
}
