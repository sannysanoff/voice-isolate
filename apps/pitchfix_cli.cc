// pitchfix-cli: cancel a constant frequency offset in a voice recording, as a
// streaming filter with a fixed lookahead.
//
//   pitchfix-cli in.wav out.wav [options]
//   pitchfix-cli --filter [options] < in.f32 > out.f32
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pitchfix.h"
#include "wav.h"

namespace {

void usage() {
  std::fprintf(stderr,
      "usage: pitchfix-cli <in.wav> <out.wav> [options]\n"
      "       pitchfix-cli --filter [options] < in.f32 > out.f32\n"
      "\n"
      "cancels a constant single-sideband tuning error: every partial of the\n"
      "voice sits at h*f0 + offset, and the filter translates the signal by\n"
      "-offset so the harmonics line up again.\n"
      "\n"
      "Fixed 48 kHz; 4096-sample window, 1024-sample hop, 4096-sample delay.\n"
      "\n"
      "options:\n"
      "  --max-shift HZ       largest correction applied (300)\n"
      "  --tau-ms N           easing time constant (500); speed falls off with distance\n"
      "  --fixed HZ           do not estimate, always cancel this offset\n"
      "  --seed HZ            initial estimate; pins the alias branch (see below)\n"
      "  --f0-min HZ          lowest harmonic spacing searched (80)\n"
      "  --f0-max HZ          highest harmonic spacing searched (320)\n"
      "  --min-confidence X   ignore frames whose comb contrast is below X (0.20)\n"
      "  --block N            samples per call, 480 = 10 ms (480)\n"
      "  --report             print the detected offset once per second\n"
      "  --float32            write float32 WAV instead of PCM16\n"
      "  --filter             raw float32 mono on stdin, filter on stdout\n"
      "  --quiet              no summary\n"
      "\n"
      "The offset is measurable only modulo the harmonic spacing f0, so the\n"
      "estimate can lock onto offset + m*f0.  Every alias aligns the harmonics;\n"
      "using --seed keeps the branch nearest to it and to the previous frames.\n");
}

double parse_number(const std::string& s, bool* ok) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  *ok = end != nullptr && *end == '\0';
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  vi::PitchFixOptions opt;
  std::string input, output;
  bool filter = false, report = false, quiet = false, float32 = false;
  int block = 480;

  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    bool ok = false;
    if (arg == "--filter") { filter = true; continue; }
    if (arg == "--report") { report = true; continue; }
    if (arg == "--quiet") { quiet = true; continue; }
    if (arg == "--float32") { float32 = true; continue; }
    if (arg == "--help" || arg == "-h") { usage(); return 0; }
    else if (arg == "--max-shift") { opt.max_shift_hz = parse_number(next("--max-shift"), &ok); }
    else if (arg == "--tau-ms") { opt.tau_s = parse_number(next("--tau-ms"), &ok) / 1000.0; }
    else if (arg == "--fixed") { opt.forced_offset_hz = parse_number(next("--fixed"), &ok); opt.estimate = false; }
    else if (arg == "--seed") { opt.seed_offset_hz = parse_number(next("--seed"), &ok); opt.has_seed = true; }
    else if (arg == "--f0-min") { opt.f0_min_hz = parse_number(next("--f0-min"), &ok); }
    else if (arg == "--f0-max") { opt.f0_max_hz = parse_number(next("--f0-max"), &ok); }
    else if (arg == "--min-confidence") { opt.min_confidence = parse_number(next("--min-confidence"), &ok); }
    else if (arg == "--block") { block = static_cast<int>(parse_number(next("--block"), &ok)); }
    else if (!arg.empty() && arg[0] == '-') { std::fprintf(stderr, "unknown option %s\n", arg.c_str()); return 2; }
    else { positional.push_back(arg); continue; }
    if (!ok) { std::fprintf(stderr, "bad value for %s\n", arg.c_str()); return 2; }
  }
  if (block < 1) block = 1;
  if (opt.f0_max_hz <= opt.f0_min_hz || opt.max_shift_hz <= 0.0) {
    std::fprintf(stderr, "needs 0 < max-shift and f0-min < f0-max\n");
    return 2;
  }

  std::vector<float> samples;
  const int sample_rate = vi::kPitchFixSampleRate;
  if (!filter) {
    if (positional.size() != 2) { usage(); return 2; }
    input = positional[0];
    output = positional[1];
    const vi::Audio audio = vi::read_wav(input);
    if (audio.samples.empty()) {
      std::fprintf(stderr, "cannot read %s\n", input.c_str());
      return 1;
    }
    if (audio.sample_rate != sample_rate) {
      std::fprintf(stderr, "%s: %d Hz; this filter requires %d Hz\n", input.c_str(),
                   audio.sample_rate, sample_rate);
      return 1;
    }
    samples = audio.samples;
  }

  vi::PitchFixer fixer(opt);

  const int latency = fixer.latency_samples();
  if (!quiet && !filter) {
    std::fprintf(stderr,
        "window %d samples, hop %d, latency %d samples, max shift %.0f Hz, tau %.0f ms\n",
        fixer.window_samples(), fixer.block_samples(), latency,
        opt.max_shift_hz, opt.tau_s * 1000.0);
    std::fprintf(stderr, "output is delayed by the latency and flushed with %d zeros\n",
                 latency);
  }

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> in(static_cast<size_t>(block)), out(static_cast<size_t>(block));
  std::vector<float> file_output;
  if (!filter) file_output.reserve(samples.size());
  double seconds = 0.0, next_report = 0.0;
  double min_offset = 0.0, max_offset = 0.0;
  long frames = 0;

  auto consume = [&](const float* chunk, int count) {
    fixer.process(chunk, count, out.data());
    if (filter) {
      std::fwrite(out.data(), sizeof(float), static_cast<size_t>(count), stdout);
    } else {
      file_output.insert(file_output.end(), out.begin(), out.begin() + count);
    }
    const vi::PitchFixStats& st = fixer.stats();
    if (st.updated) {
      min_offset = frames == 0 ? st.offset_hz : std::min(min_offset, st.offset_hz);
      max_offset = frames == 0 ? st.offset_hz : std::max(max_offset, st.offset_hz);
      ++frames;
    }
    seconds += static_cast<double>(count) / sample_rate;
    if (report && seconds >= next_report) {
      std::fprintf(stderr, "  t=%6.2fs  offset %+7.1f Hz  f0 %5.1f Hz  conf %5.2f%s\n",
                   seconds, st.offset_hz, st.f0_hz, st.confidence,
                   st.updated ? "" : "  (hold)");
      next_report += 1.0;
    }
  };

  if (filter) {
    while (true) {
      const size_t got = std::fread(in.data(), sizeof(float), in.size(), stdin);
      if (got == 0) break;
      consume(in.data(), static_cast<int>(got));
    }
    std::fflush(stdout);
  } else {
    for (size_t pos = 0; pos < samples.size(); pos += static_cast<size_t>(block)) {
      const int n = static_cast<int>(std::min<size_t>(block, samples.size() - pos));
      consume(samples.data() + pos, n);
    }
    // The filter delays by `latency`.  Flush it with silence so nothing is cut
    // off: the file holds leading silence plus the whole processed input.
    std::vector<float> tail(static_cast<size_t>(latency), 0.f);
    const int got = fixer.finish(tail.data(), latency);
    file_output.insert(file_output.end(), tail.begin(), tail.begin() + got);
    vi::write_wav(output, file_output, sample_rate, float32 ? 32 : 16, false);
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(t1 - t0).count();
  if (!quiet) {
    std::fprintf(stderr,
        "%s: %zu samples, %.2f s audio in %.2f s (%.2fx realtime), "
        "offset %.1f Hz (range %+.1f..%+.1f over %ld frames)\n",
        filter ? "filter" : output.c_str(), filter ? 0 : file_output.size(),
        seconds, elapsed, seconds > 0.0 ? seconds / elapsed : 0.0,
        fixer.stats().offset_hz, min_offset, max_offset, frames);
  }
  return 0;
}
