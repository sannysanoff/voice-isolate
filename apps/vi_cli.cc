// voice-isolate-cli: run the streaming voice isolator over a WAV file, or as a
// raw sample-in / sample-out filter on stdin/stdout.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "stream.h"
#include "wav.h"

namespace {

void usage() {
  std::fprintf(stderr,
      "usage: voice-isolate-cli <in.wav> <out.wav> [options]\n"
      "       voice-isolate-cli --filter [options] < in.f32 > out.f32\n"
      "\n"
      "options:\n"
      "  --model DIR          weight directory (default: model)\n"
      "  --block-frames N     frames computed together (default 4)\n"
      "  --window-frames N    time-attention window; 0 = unbounded (default 600)\n"
      "  --crossfade-frames N crossfade before each boundary (default 200)\n"
      "  --cache-frames N     attention lookback (default: checkpoint)\n"
      "  --threads N          worker threads (default: all cores)\n"
      "  --float32            write float32 WAV instead of PCM16\n"
      "  --quant MODE         dense precision: fp32 (default), fp16, bf16, int8, int8g\n"
      "  --fp16-kv            store time-attention KV cache in fp16 (faster, half RAM)\n"
      "  --filter             raw float32 mono on stdin, filter on stdout\n"
      "  --chunk N            filter mode: samples per read (default 480)\n"
      "  --quiet              no timing report\n"
      "  --profile            per-stage timing report\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = "model";
  std::string input, output;
  vi::StreamOptions options;
  int threads = 0;
  bool float32 = false;
  vi::Model::QuantMode quant_mode = vi::Model::QuantMode::Fp32;
  std::string quant_name = "fp32";
  bool fp16_kv = false;
  bool fp16_kv_explicit = false;
  bool filter = false;
  bool quiet = false;
  int chunk = 480;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--model") model_dir = next();
    else if (arg == "--block-frames") options.block_frames = std::stoi(next());
    else if (arg == "--window-frames") options.window_frames = std::stoi(next());
    else if (arg == "--crossfade-frames")
      options.crossfade_frames = std::stoi(next());
    else if (arg == "--cache-frames") options.cache_frames = std::stoi(next());
    else if (arg == "--threads") threads = std::stoi(next());
    else if (arg == "--chunk") chunk = std::stoi(next());
    else if (arg == "--float32") float32 = true;
    else if (arg == "--quant") {
      quant_name = next();
      if (quant_name == "fp32") quant_mode = vi::Model::QuantMode::Fp32;
      else if (quant_name == "fp16") quant_mode = vi::Model::QuantMode::Fp16;
      else if (quant_name == "bf16") quant_mode = vi::Model::QuantMode::Bf16;
      else if (quant_name == "int8") quant_mode = vi::Model::QuantMode::Int8;
      else if (quant_name == "int8g") quant_mode = vi::Model::QuantMode::Int8Group;
      else { usage(); return 2; }
    }
    else if (arg == "--fp16-kv") { fp16_kv = true; fp16_kv_explicit = true; }
    else if (arg == "--filter") filter = true;
    else if (arg == "--quiet") quiet = true;
    else if (arg == "--profile") vi::Model::profile_enabled() = true;
    else if (arg == "-h" || arg == "--help") { usage(); return 0; }
    else if (arg.rfind("--", 0) == 0) { usage(); return 2; }
    else if (input.empty()) input = arg;
    else if (output.empty()) output = arg;
    else { usage(); return 2; }
  }

  if (threads > 0) vi::set_default_pool_threads(threads);

  try {
    auto model = std::make_shared<vi::Model>(model_dir);
    model->set_quant_mode(quant_mode);
    if (fp16_kv_explicit)
      model->set_fp16_kv(fp16_kv);
    else if (quant_mode != vi::Model::QuantMode::Fp32)
      model->set_fp16_kv(true);
    vi::Stream stream(model, options);
    if (!quiet)
      std::fprintf(stderr,
                   "weights %.1f MB, threads %d, %s weights%s%s, look-ahead %d frames, latency "
                   "%d samples (%.1f ms), attention window %d frames, cache "
                   "%d frames\n",
                   static_cast<double>(model->weight_bytes()) / 1e6,
                   vi::default_pool().size(),
                   quant_name.c_str(),
                   vi::sme2_available() ? " (SME2)" : "",
                   model->fp16_kv() ? " + fp16-kv" : "",
                   stream.lookahead_frames(),
                   stream.latency_samples(),
                   1000.0 * stream.latency_samples() / stream.sample_rate(),
                   options.window_frames, stream.attention_capacity());

    if (filter) {
      // A live source: hand over each chunk as it arrives, write what is final.
      std::vector<float> in(static_cast<size_t>(chunk));
      std::vector<unsigned char> out;
      while (true) {
        const size_t got = std::fread(in.data(), sizeof(float), in.size(), stdin);
        if (got == 0) break;
        const std::vector<float> y = stream.process(in.data(), got);
        out.resize(y.size() * sizeof(float));
        if (!y.empty())
          std::fwrite(y.data(), sizeof(float), y.size(), stdout);
      }
      const std::vector<float> tail = stream.flush();
      if (!tail.empty()) std::fwrite(tail.data(), sizeof(float), tail.size(), stdout);
      std::fflush(stdout);
      return 0;
    }

    if (input.empty() || output.empty()) { usage(); return 2; }
    const vi::Audio audio = vi::read_wav(input);
    if (audio.sample_rate != stream.sample_rate()) {
      std::fprintf(stderr, "%s: %d Hz, the model needs %d Hz\n", input.c_str(),
                   audio.sample_rate, stream.sample_rate());
      return 1;
    }

    const auto begin = std::chrono::steady_clock::now();
    std::vector<float> y;
    y.reserve(audio.samples.size());
    constexpr size_t kPiece = 24000;  // 0.5 s, as the reference pushes
    for (size_t start = 0; start < audio.samples.size(); start += kPiece) {
      const size_t count = std::min(kPiece, audio.samples.size() - start);
      const std::vector<float> part = stream.process(audio.samples.data() + start, count);
      y.insert(y.end(), part.begin(), part.end());
    }
    const std::vector<float> tail = stream.flush();
    y.insert(y.end(), tail.begin(), tail.end());
    const auto end = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(end - begin).count();

    y.resize(audio.samples.size());
    vi::write_wav(output, y, audio.sample_rate, float32 ? 32 : 16,
                  /*normalize_peak=*/!float32);
    if (!quiet) {
      const vi::Model::Profile& p = vi::Model::profile();
      const double stage = p.encode + p.time_attn + p.band_attn + p.mamba + p.decode;
      if (vi::Model::profile_enabled())
        std::fprintf(stderr,
                     "  encode %.2f s  time_attn %.2f  band_attn %.2f  mamba %.2f"
                     "  decode %.2f  (stages %.2f of %.2f s = %.0f%%)\n"
                     "    gemm %.2f  ln %.2f  scores %.2f  pv %.2f  rope %.2f"
                     "  transpose %.2f  mconv %.2f  scan %.2f  add %.2f"
                     "  corr %.2f  ola %.2f\n",
                     p.encode, p.time_attn, p.band_attn, p.mamba, p.decode,
                     stage, seconds, 100.0 * stage / seconds, p.gemm, p.ln,
                     p.attn_scores, p.attn_pv, p.rope, p.transpose, p.mamba_conv,
                     p.mamba_scan, p.add, p.corr, p.ola);
      const double audio_seconds =
          static_cast<double>(audio.samples.size()) / audio.sample_rate;
      std::fprintf(stderr, "%s: %.2f s audio in %.2f s (%.2fx realtime)\n",
                   output.c_str(), audio_seconds, seconds,
                   audio_seconds / seconds);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
