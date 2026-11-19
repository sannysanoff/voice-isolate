// The isolator network: band-split dual-path transformer/Mamba, streaming form.
#pragma once

#include <complex>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "gemm_int8.h"
#include "fft.h"

namespace vi {

// Rotary tables for the time attention: cos/sin per position, head_dim/2 wide.
struct Rope {
  int half = 0;
  std::vector<float> cos, sin;

  void init(int head_dim, int positions);
  const float* cos_at(int pos) const {
    return cos.data() + static_cast<size_t>(pos) * half;
  }
  const float* sin_at(int pos) const {
    return sin.data() + static_cast<size_t>(pos) * half;
  }
};

// One block's time-attention past for one lane: a key lives at the slot that is
// its rotary position, so it is rotated exactly once, when it is written.
struct AttnState {
  std::vector<float> k, v;  // [bands][capacity][width]
  int capacity = 0;
  int end = 0;

  void clear() { end = 0; }
  void release() {
    k.clear();
    v.clear();
    capacity = 0;
    end = 0;
  }
};

// The carried recurrence of one Mamba layer, per band: h is [state][inner].
struct MambaState {
  std::vector<float> conv;  // [bands][inner][conv - 1]
  std::vector<float> ssm;   // [bands][state][inner]
};

// Everything a lane carries from frame to frame.
struct Lane {
  std::vector<AttnState> attn;    // indexed by block
  std::vector<MambaState> mamba;  // indexed by block
  bool started = false;
};

// The bounds a lane runs under.
struct LaneParams {
  int cache_frames = 1200;    // how far back attention may look
  int capacity_limit = 1800;  // buffer slots before compacting
  int window = 1200;          // trained attention window, 0 = unbounded
};

class Model {
 public:
  explicit Model(const std::string& dir);

  const Config& config() const { return cfg_; }
  const Fft& fft() const { return fft_; }
  const std::vector<float>& window() const { return window_; }
  size_t weight_bytes() const { return weights_.bytes(); }
  size_t scratch_bytes() const;

  // int8 weight-only dense layers. Off by default: the fp32 path is the
  // reference-verified one, and the quantised path trades accuracy for speed.
  void set_quant(bool on) { quant_enabled_ = on; }
  bool quant_enabled() const { return quant_enabled_; }

  void init_lane(Lane& lane) const;
  // A lane's start: the recurrence carried, the attention past dropped.
  void clone_lane(const Lane& src, Lane& dst) const;

  // spectra [frames][bins] -> band tokens [bands][frames][width]
  void encode(const std::complex<float>* spectra, int frames, float* feat) const;

  // Band tokens through every block, carrying the lane's state.
  void backbone(float* feat, int frames, Lane& lane, const Rope& rope,
                const LaneParams& lp) const;

  // Band tokens -> decoder output [frames][bins][4]
  void decode(const float* feat, int frames, float* raw) const;

  // The decoder's complex change of one frame's spectrum.
  void correction(const float* raw, const std::complex<float>* spectrum,
                  int bins, std::complex<float>* out) const;

  // Optional stage timings (seconds); enabled by the CLI's --profile.
  struct Profile {
    double encode = 0, time_attn = 0, band_attn = 0, mamba = 0, decode = 0;
    // Sub-stages, to see what inside the three hot passes actually costs.
    double gemm = 0;          // all linear_par calls
    double ln = 0;            // all layer_norm_par calls
    double attn_scores = 0;   // q.k + softmax
    double attn_pv = 0;       // weights x values
    double rope = 0;          // rotary application + cache trim
    double transpose = 0;     // to_frame_major / to_band_major
    double mamba_conv = 0;    // depthwise convolution pass
    double mamba_scan = 0;    // selective scan pass
    double mamba_exp = 0;     // the exp(delta*A) inside the scan
    double add = 0;           // residual adds
    double corr = 0;          // spectrum correction + iSTFT
    double ola = 0;           // overlap-add and release
  };
  static Profile& profile();
  static bool& profile_enabled();

  // Test hooks.
  std::function<void(int, int, const float*, const float*)> debug_block;
  std::function<void(int, int, const float*)> debug_decode;

 private:
  struct AttnWeights {
    const Tensor *n1w, *n1b, *n2w, *n2b;
    const Tensor *qkv_w, *qkv_b, *proj_w, *proj_b;
    const Tensor *f0w, *f0b, *f1w, *f1b;
    const Tensor *bn1w, *bn1b, *bn2w, *bn2b;
    const Tensor *in_w, *in_b, *out_w, *out_b;
    const Tensor *bl1w, *bl1b, *bl2w, *bl2b;
  };

  struct MambaLayer {
    const Tensor *in_w, *conv_w, *conv_b, *x_w, *dt_w, *dt_b, *d, *out_w;
    std::vector<float> a;       // [state][inner] = -exp(A_log)
    std::vector<float> conv_t;  // [taps][inner] depthwise kernel, transposed
  };

  struct MambaWeights {
    MambaLayer forward, backward;
    const Tensor *out_w, *out_b, *norm_w, *norm_b;
    bool causal = false;
  };

  struct BandHead {
    const Tensor *n_w, *n_b, *h1_w, *h1_b, *h3_w, *h3_b;
  };

  struct Block {
    bool transformer = false;
    AttnWeights attn{};
    MambaWeights time_mamba{};
    MambaWeights freq_mamba{};
  };

  // Reused working memory; one Model serves one stream at a time.
  struct Scratch {
    int rows = 0;
    std::vector<float> norm, qkv, attn, ffn, trans, rev, cat, bwd;
    std::vector<float> xz, u2, y, delta, xp;
    std::vector<float> scores;
    std::vector<float> bscores;
    // Band attention's q/k/v transposed to [head][band][hd] per frame, so each
    // head's attention runs on contiguous rows instead of a 3*width stride.
    std::vector<float> bqkv;
    std::vector<float> mamba_ext, mamba_decay, mamba_h;
    // Per-worker scratch for the int8 activations and their row scales.
    std::vector<int8_t> q8;
    std::vector<float> q8s;
  };

  Scratch& scratch(int rows) const;
  // C[m, n] = A[m, k] * W[n, k]^T + bias, columns spread over the pool.
  void linear_par(const float* a, int m, int k, const Tensor& w,
                  const float* bias, float* c) const;
  // The int8 path; false when the layer is too small to be worth quantising.
  bool linear_par_i8(const float* a, int m, int k, int stride, const Tensor& w,
                     const float* bias, float* c) const;
  const Int8Matrix* quantized(const Tensor& w) const;
  mutable std::unordered_map<const Tensor*, Int8Matrix> quant_;
  bool quant_enabled_ = false;
  void linear_par(const float* a, int m, int k, int stride, const Tensor& w,
                  const float* bias, float* c) const;
  // cost_per_element lets the caller say a kernel is expensive (erf, exp).
  void elementwise_par(float* x, int n, void (*fn)(float*, int),
                       int cost_per_element = 1) const;
  void layer_norm_par(const float* x, float* y, int rows, int cols,
                      const float* w, const float* b) const;
  void add_par(float* dst, const float* src, int n) const;
  void load_blocks();
 public:
  // Microbenchmark entry: the same GEMM the transformer runs.
  void linear_par_bench(const float* a, int m, int k, const Tensor& w,
                        const float* bias, float* c) const {
    linear_par(a, m, k, w, bias, c);
  }
 private:

  void time_transformer(const AttnWeights& w, float* feat, int frames,
                        AttnState& state, const Rope& rope,
                        const LaneParams& lp) const;
  void band_transformer(const AttnWeights& w, float* feat, int frames) const;
  // One Mamba layer over `rows` tokens: rows = tasks * len, each task a
  // sequence of `len` tokens. Carries conv/ssm state when given.
  void mamba_layer(const MambaLayer& l, const float* x, int rows, int len,
                   float* out, float* conv_state, float* ssm_state,
                   float* slab_ext, float* slab_decay, float* slab_h) const;

  Config cfg_;
  Weights weights_;
  Fft fft_;
  std::vector<float> window_;
  std::vector<Block> blocks_;
  std::vector<const Tensor*> enc_w_, enc_b_;
  std::vector<const Tensor*> enc_norm_w_, enc_norm_b_;
  const Tensor* band_embedding_ = nullptr;
  const Tensor* lookahead_embedding_ = nullptr;
  std::vector<float> decoder_scale_, decoder_shift_;  // FiLM on the decoder
  std::vector<BandHead> heads_;
  mutable Scratch scratch_;
};

}  // namespace vi
