// Numerical and reference tests for the portable voice isolator.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "stream.h"
#include "vecmath.h"

namespace {

int g_failures = 0;

std::vector<float> load_bin(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) return {};
  std::fseek(file, 0, SEEK_END);
  const long bytes = std::ftell(file);
  std::fseek(file, 0, SEEK_SET);
  std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
  if (std::fread(data.data(), sizeof(float), data.size(), file) != data.size())
    data.clear();
  std::fclose(file);
  return data;
}

struct Diff {
  double max_abs = 0.0;
  double rms = 0.0;
  double ref_rms = 0.0;
  double snr() const {
    return ref_rms > 0.0 && rms > 0.0 ? 20.0 * std::log10(ref_rms / rms) : 999.0;
  }
};

Diff compare(const float* a, const float* b, size_t n) {
  Diff d;
  double acc = 0.0, ref = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double diff = static_cast<double>(a[i]) - b[i];
    d.max_abs = std::max(d.max_abs, std::fabs(diff));
    acc += diff * diff;
    ref += static_cast<double>(b[i]) * b[i];
  }
  d.rms = std::sqrt(acc / static_cast<double>(n));
  d.ref_rms = std::sqrt(ref / static_cast<double>(n));
  return d;
}

// A stage passes when it agrees to float32 rounding: the error must sit at
// least 100 dB below the reference, and its peak within 2e-5 of the reference
// peak (large activations carry proportionally larger absolute error).
void check(const char* name, const Diff& d, double max_abs_limit, bool& group_ok) {
  const bool ok = d.snr() >= 100.0 || d.max_abs <= max_abs_limit;
  group_ok = group_ok && ok;
  std::printf("  %-24s max %.3e  rms %.3e  snr %6.1f dB  %s\n", name, d.max_abs,
              d.rms, d.snr(), ok ? "ok" : "FAIL");
}

void expect(bool condition, const char* what) {
  std::printf("  %-52s %s\n", what, condition ? "ok" : "FAIL");
  if (!condition) ++g_failures;
}

// ------------------------------------------------------------------ tests ---

void test_fft() {
  std::printf("fft\n");
  vi::Fft fft(2048);
  std::vector<float> x(2048), y(2048);
  std::vector<std::complex<float>> spec(fft.bins());
  unsigned seed = 12345;
  for (auto& v : x) {
    seed = seed * 1103515245u + 12345u;
    v = static_cast<float>((seed >> 9) & 0xffff) / 32768.0f - 1.0f;
  }
  fft.rfft(x.data(), spec.data());
  fft.irfft(spec.data(), y.data());
  double err = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    err = std::max(err, std::fabs(static_cast<double>(x[i]) - y[i]));
  expect(err < 1e-6, "rfft -> irfft roundtrip");
  std::printf("  roundtrip max error %.3e\n", err);
}

void test_vector_math() {
  std::printf("vector math\n");
  const int n = 1027;
  std::vector<float> x(n), ref(n);
  for (int i = 0; i < n; ++i) x[i] = -8.0f + 16.0f * static_cast<float>(i) / n;
  auto relative = [&](const char* name, double limit) {
    double err = 0.0;
    for (int i = 0; i < n; ++i) {
      const double denom = std::max(1e-6, std::fabs(static_cast<double>(ref[i])));
      err = std::max(err, std::fabs(static_cast<double>(x[i]) - ref[i]) / denom);
    }
    char label[96];
    std::snprintf(label, sizeof(label), "%s (max rel %.1e)", name, err);
    expect(err < limit, label);
  };
  for (int i = 0; i < n; ++i) ref[i] = std::exp(x[i]);
  std::vector<float> y = x;
  vi::v_exp(y.data(), n);
  std::swap(x, y);
  std::swap(x, ref);
  relative("exp", 1e-6);
  std::swap(x, ref);

  for (int i = 0; i < n; ++i) ref[i] = std::tanh(x[i]);
  y = x;
  vi::v_tanh(y.data(), n);
  std::swap(x, y);
  relative("tanh", 1e-6);
  std::swap(x, y);
  std::swap(x, ref);

  for (int i = 0; i < n; ++i)
    ref[i] = 0.5f * x[i] * (1.0f + std::erff(x[i] * 0.70710678f));
  y = x;
  vi::v_gelu(y.data(), n);
  std::swap(x, y);
  relative("gelu", 1e-6);
  std::swap(x, y);
  std::swap(x, ref);

  for (int i = 0; i < n; ++i)
    ref[i] = x[i] / (1.0f + std::exp(-x[i]));
  y = x;
  vi::v_silu(y.data(), n);
  std::swap(x, y);
  relative("silu", 1e-6);
  std::swap(x, y);
  std::swap(x, ref);

  for (int i = 0; i < n; ++i)
    ref[i] = x[i] > 20.0f ? x[i] : std::log1p(std::exp(x[i]));
  y = x;
  vi::v_softplus(y.data(), n);
  std::swap(x, y);
  relative("softplus", 1e-6);
  std::swap(x, y);
  std::swap(x, ref);

  // out[i] = exp(a[i] * scale[i]) over an arbitrary length (tails included)
  const int m = 3071;
  std::vector<float> a(m), out(m), scale(m);
  for (int i = 0; i < m; ++i) {
    a[i] = -4.0f + 8.0f * static_cast<float>(i % 701) / 701.0f;
    scale[i] = -2.0f + 4.0f * static_cast<float>(i % 977) / 977.0f;
  }
  vi::v_mul_exp(out.data(), a.data(), scale.data(), m);
  double err = 0.0;
  for (int i = 0; i < m; ++i) {
    const double want = std::exp(static_cast<double>(a[i]) * scale[i]);
    err = std::max(err, std::fabs(static_cast<double>(out[i]) - want) / want);
  }
  char label[96];
  std::snprintf(label, sizeof(label), "mul_exp (max rel %.1e)", err);
  expect(err < 1e-6, label);
}

// Runs the reference stages dump and compares every intermediate.
void test_reference(const std::string& root) {
  const std::string dir = root + "/ref/stages";
  const std::string weights = root + "/model";
  const std::string input_path = dir + "/input.bin";
  const std::vector<float> input = load_bin(input_path);
  if (input.empty()) {
    std::printf("reference: %s missing, skipped\n", input_path.c_str());
    return;
  }
  std::printf("reference stages (%zu samples)\n", input.size());

  auto model = std::make_shared<vi::Model>(weights);
  // VI_QUANT=1 runs the same stage comparison through the int8 dense layers, to
  // see where the quantisation error accumulates.
  if (std::getenv("VI_QUANT") != nullptr) model->set_quant(true);
  const vi::Config& cfg = model->config();
  const int bands = cfg.bands();
  const int width = cfg.width;
  const int bins = cfg.bins();

  // The stream ends at samples/hop + 1 frames: the clip's own plus the flush
  // padding that turns the tail into whole windows.
  const size_t frames = input.size() / cfg.hop + 1;
  std::vector<std::vector<float>> block_in(cfg.num_blocks),
      block_out(cfg.num_blocks);
  for (int i = 0; i < cfg.num_blocks; ++i) {
    block_in[i].assign(frames * bands * width, 0.0f);
    block_out[i].assign(frames * bands * width, 0.0f);
  }
  std::vector<float> raw(frames * bins * 4, 0.0f);
  std::vector<size_t> seen(cfg.num_blocks, 0);
  size_t raw_seen = 0;
  model->debug_decode = [&](int, int count, const float* values) {
    std::memcpy(raw.data() + raw_seen * bins * 4, values,
                sizeof(float) * static_cast<size_t>(count) * bins * 4);
    raw_seen += static_cast<size_t>(count);
  };

  // Track frame counts through the block hook; block 0's input is the encoder
  // output, which the reference dumps separately.
  model->debug_block = [&](int index, int count, const float* in,
                           const float* out) {
    const size_t offset = seen[static_cast<size_t>(index)] * bands * width;
    for (int t = 0; t < count; ++t)
      for (int b = 0; b < bands; ++b) {
        const size_t from = (static_cast<size_t>(b) * count + t) * width;
        const size_t to = ((static_cast<size_t>(t) * bands + b)) * width;
        std::memcpy(block_in[static_cast<size_t>(index)].data() + offset + to,
                    in + from, sizeof(float) * width);
        std::memcpy(block_out[static_cast<size_t>(index)].data() + offset + to,
                    out + from, sizeof(float) * width);
      }
    seen[static_cast<size_t>(index)] += static_cast<size_t>(count);
  };

  vi::StreamOptions options;
  options.window_frames = 0;
  options.crossfade_frames = 0;
  options.block_frames = 4;
  vi::Stream stream(model, options);
  vi::set_default_pool_threads(1);

  std::vector<float> output;
  constexpr size_t kPiece = 24000;
  for (size_t start = 0; start < input.size(); start += kPiece) {
    const size_t count = std::min(kPiece, input.size() - start);
    const std::vector<float> part = stream.process(input.data() + start, count);
    output.insert(output.end(), part.begin(), part.end());
  }
  const std::vector<float> tail = stream.flush();
  output.insert(output.end(), tail.begin(), tail.end());

  bool ok = true;
  std::printf("  frames %lld, output %zu samples\n",
              static_cast<long long>(stream.frames_processed()), output.size());
  expect(output.size() == input.size(), "output length equals input length");
  expect(stream.latency_samples() == 2048 + 480 * 4, "latency is 3968 samples");

  const std::vector<float> ref_encoded = load_bin(dir + "/encoded.bin");
  if (!ref_encoded.empty())
    check("encoded", compare(block_in[0].data(), ref_encoded.data(),
                             ref_encoded.size()), 2e-5, ok);
  for (int i = 0; i < cfg.num_blocks; ++i) {
    const std::vector<float> in_ref =
        load_bin(dir + "/block" + std::to_string(i) + "_in.bin");
    const std::vector<float> out_ref =
        load_bin(dir + "/block" + std::to_string(i) + "_out.bin");
    char name[64];
    if (!in_ref.empty()) {
      std::snprintf(name, sizeof(name), "block%d in", i);
      check(name, compare(block_in[static_cast<size_t>(i)].data(), in_ref.data(),
                          in_ref.size()), 2e-5, ok);
    }
    if (!out_ref.empty()) {
      std::snprintf(name, sizeof(name), "block%d out", i);
      check(name, compare(block_out[static_cast<size_t>(i)].data(),
                          out_ref.data(), out_ref.size()), 2e-5, ok);
    }
  }
  const std::vector<float> raw_ref = load_bin(dir + "/raw.bin");
  if (!raw_ref.empty())
    check("decoder raw", compare(raw.data(), raw_ref.data(), raw_ref.size()),
          2e-4, ok);
  const std::vector<float> out_ref = load_bin(dir + "/output.bin");
  if (!out_ref.empty())
    check("samples", compare(output.data(), out_ref.data(), out_ref.size()),
          1e-5, ok);
  if (!ok) ++g_failures;
}

// Compares a streamed run against a reference dump directory.
void check_stream(const std::string& root, const std::string& dir,
                  vi::StreamOptions options, const char* name) {
  const std::vector<float> input = load_bin(dir + "/input.bin");
  const std::vector<float> expected = load_bin(dir + "/output.bin");
  if (input.empty() || expected.empty()) {
    std::printf("%s: %s missing, skipped\n", name, dir.c_str());
    return;
  }
  auto model = std::make_shared<vi::Model>(root + "/model");
  vi::Stream stream(model, options);
  std::vector<float> output;
  constexpr size_t kPiece = 24000;
  for (size_t start = 0; start < input.size(); start += kPiece) {
    const size_t count = std::min(kPiece, input.size() - start);
    const std::vector<float> part = stream.process(input.data() + start, count);
    output.insert(output.end(), part.begin(), part.end());
  }
  const std::vector<float> tail = stream.flush();
  output.insert(output.end(), tail.begin(), tail.end());
  bool ok = true;
  std::printf("%s (%.1f s, %zu samples, %lld frames)\n", name,
              static_cast<double>(input.size()) / 48000.0, input.size(),
              static_cast<long long>(stream.frames_processed()));
  expect(output.size() == input.size(), "output length equals input length");
  check("samples", compare(output.data(), expected.data(),
                           std::min(output.size(), expected.size())),
        1e-5, ok);
  if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  test_fft();
  test_vector_math();
  const bool quick = argc > 2 && std::string(argv[2]) == "quick";
  test_reference(root);
  if (quick) {
    std::printf("\n%s\n", g_failures == 0 ? "all tests passed" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
  }
  vi::StreamOptions windowed;
  windowed.window_frames = 600;
  windowed.crossfade_frames = 200;
  windowed.block_frames = 4;
  check_stream(root, root + "/ref/win10", windowed, "windowed 600/200");
  vi::StreamOptions plain;
  plain.window_frames = 0;
  plain.crossfade_frames = 0;
  plain.block_frames = 4;
  check_stream(root, root + "/ref/plain20", plain, "plain, cache compaction");
  std::printf("\n%s\n", g_failures == 0 ? "all tests passed" : "FAILURES");
  return g_failures == 0 ? 0 : 1;
}
