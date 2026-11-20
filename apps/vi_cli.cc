#include "model.h"
#include "stream.h"
#include "wav.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "Usage: %s INPUT.wav OUTPUT.wav [MODEL_DIR]\n", argv[0]);
    return 1;
  }
  try {
    if (const char* threads = std::getenv("VI_THREADS")) {
      const int count = std::stoi(threads);
      if (count < 1 || count > 256) throw std::runtime_error("VI_THREADS must be between 1 and 256");
      vi::set_default_pool_threads(count);
    }
    vi::Model::profile_enabled() = std::getenv("VI_PROFILE") != nullptr;
    const auto started = std::chrono::steady_clock::now();
    const std::string model_dir = argc == 4 ? argv[3] : "model";
    const vi::Audio input = vi::read_wav(argv[1]);
    auto model = std::make_shared<vi::Model>(model_dir);
    if (input.sample_rate != model->config().sample_rate)
      throw std::runtime_error("input sample rate does not match model sample rate");
    vi::StreamOptions options;
    options.block_frames = 4;
    vi::Stream stream(model, options);
    std::vector<float> output;
    output.reserve(input.samples.size());
    constexpr size_t chunk = 24000;
    for (size_t start = 0; start < input.samples.size(); start += chunk) {
      const auto part = stream.process(input.samples.data() + start,
                                       std::min(chunk, input.samples.size() - start));
      output.insert(output.end(), part.begin(), part.end());
    }
    const auto tail = stream.flush();
    output.insert(output.end(), tail.begin(), tail.end());
    if (output.size() != input.samples.size())
      throw std::runtime_error("unexpected output sample count");
    vi::write_wav(argv[2], output, input.sample_rate, 16, true);
    std::printf("Wrote %zu samples to %s (%lld frames)\n", output.size(), argv[2],
                static_cast<long long>(stream.frames_processed()));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("Elapsed %.2f s, audio %.2f s\n", seconds, double(input.samples.size()) / input.sample_rate);
    if (vi::Model::profile_enabled()) {
      const auto& p = vi::Model::profile();
      std::printf("Profile (s): encode %.2f time-attention %.2f band-attention %.2f mamba %.2f decode %.2f; GEMM %.2f scan %.2f\n", p.encode, p.time_attn, p.band_attn, p.mamba, p.decode, p.gemm, p.mamba_scan);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "voice-isolate: %s\n", e.what());
    return 1;
  }
}
