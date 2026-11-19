// Shared types and numeric helpers for the voice-isolator port.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vi {

// ---------------------------------------------------------------- tensors ---

// A view into the packed weight blob. `shape` is the checkpoint's shape, in
// torch order (a Linear weight is [out, in]).
struct Tensor {
  const float* data = nullptr;
  std::vector<int> shape;

  int dim(int i) const { return shape[static_cast<size_t>(i)]; }
  size_t size() const {
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    return n;
  }
};

class Weights {
 public:
  // Loads <dir>/weights.bin and <dir>/weights.txt.
  void load(const std::string& dir);

  const Tensor& get(const std::string& name) const;
  // Same, requiring an exact shape; used to fail fast on a mismatched export.
  const Tensor& get(const std::string& name, std::vector<int> shape) const;

  size_t bytes() const { return blob_.size() * sizeof(float); }

 private:
  std::vector<float> blob_;
  std::unordered_map<std::string, Tensor> tensors_;
};

// ---------------------------------------------------------------- config ----

struct Config {
  int n_fft = 2048;
  int hop = 480;
  int sample_rate = 48000;
  int width = 512;
  int heads = 8;
  int mamba_state = 16;
  int mamba_conv = 4;
  int mamba_expand = 2;
  float correction_bound = 0.5f;
  float chunk_seconds = 12.0f;
  int attention_window_frames = 1200;
  int lookahead_frames = 0;
  int lookahead_index = 0;
  int num_blocks = 0;
  std::vector<std::string> block_types;
  std::vector<std::pair<int, int>> edges;  // [start, end) bins per band

  int bins() const { return n_fft / 2 + 1; }
  int bands() const { return static_cast<int>(edges.size()); }
  int inner() const { return mamba_expand * width; }
  int dt_rank() const { return (width + 15) / 16; }
  int head_dim() const { return width / heads; }
};

Config load_config(const std::string& dir);

// -------------------------------------------------------------- math ops ----

inline float gelu(float x) {
  // torch.nn.GELU() default: the exact erf form.
  return 0.5f * x * (1.0f + std::erff(x * 0.70710678118654752440f));
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline float softplus(float x) {
  // torch's softplus threshold is 20 and is a hard switch, not a clamp.
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

// y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * w + b, biases per column.
void layer_norm(const float* x, float* y, int rows, int cols, const float* w,
                const float* b, float eps = 1e-5f);

// --------------------------------------------------------------- threads ----

// A fixed pool running independent indices of a range. Splitting work across
// bands or frames never changes the arithmetic, so results stay deterministic.
class Pool {
 public:
  explicit Pool(int threads);
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  int size() const { return threads_; }
  void parallel_for(int n, const std::function<void(int)>& fn);

 private:
  struct Impl;
  Impl* impl_;
  int threads_;
};

// Process-wide pool (hardware_concurrency unless overridden).
Pool& default_pool();
void set_default_pool_threads(int threads);

}  // namespace vi
