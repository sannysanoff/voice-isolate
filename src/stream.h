// A realtime voice-isolation filter: samples in, samples out.
#pragma once

#include <complex>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "model.h"

namespace vi {

struct StreamOptions {
  int block_frames = 8;      // frames computed together; throughput only (x86: ~1.98x realtime int8)
  int window_frames = 0;     // 0 = one continuous lane (sliding KV, no restarts); N = restart lanes every N frames
  int crossfade_frames = 200;  // lane-boundary crossfade, used only when window_frames > 0
  int cache_frames = 400;    // sliding-mode attention lookback; 0 = the checkpoint's window (1200)
};

// Streaming filter. `process` takes mono float samples at the model rate and
// returns the samples that became final; `flush` returns the rest. Latency is
// n_fft + hop * lookahead samples.
class Stream {
 public:
  Stream(std::shared_ptr<Model> model, const StreamOptions& options);

  std::vector<float> process(const float* samples, size_t count);
  std::vector<float> flush();

  int sample_rate() const;
  int latency_samples() const;
  int lookahead_frames() const;
  int64_t frames_processed() const { return frames_; }
  int64_t samples_released() const { return released_; }
  int attention_capacity() const;

 private:
  void advance(int64_t limit);
  void run_frames(int64_t first, int count);
  void analyze(int64_t first, int count);
  void synthesize();
  std::vector<float> release(int64_t upto);
  void trim();
  int64_t next_cut() const;

  std::shared_ptr<Model> model_;
  StreamOptions opt_;
  LaneParams lane_params_;
  Rope rope_;

  Lane lane_a_, lane_b_;
  bool lane_b_started_ = false;
  bool windowed_ = false;
  bool closed_ = false;

  // Audio history, indexed from audio_base_.
  std::vector<float> audio_;
  int64_t audio_base_ = 0;
  // Overlap-add of the synthesized change, indexed from ola_base_.
  std::vector<float> ola_, envelope_;
  int64_t ola_base_ = 0;

  int64_t frames_ = 0;    // analysis frames computed
  int64_t emitted_ = 0;   // frames synthesized
  int64_t released_ = 0;  // output samples returned
  int64_t pending_base_ = 0;

  std::deque<std::vector<float>> raw_q_;                 // [bins][4] per frame
  std::deque<std::vector<std::complex<float>>> spec_q_;  // [bins] per frame

  std::vector<std::complex<float>> spectra_;
  std::vector<float> feat_a_, feat_b_, raw_a_, raw_b_;
  std::vector<float> frame_audio_, delta_audio_;
  std::vector<std::complex<float>> delta_spec_;
};

}  // namespace vi
