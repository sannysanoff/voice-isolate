// The isolator network: band-split dual-path transformer/Mamba, streaming form.
#pragma once

#include <complex>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "gemm_int8.h"
#include "gemm_avx2.h"
#include "gemm_sme.h"
#include "fft.h"
#include "metal_backend.h"

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
  std::vector<float> k, v;        // [bands][head][capacity][hd] (fp32 KV cache)
  std::vector<uint16_t> k16, v16; // [bands][head][capacity][hd] (fp16 KV cache)
  std::vector<int8_t> k8, v8;     // [bands][head][capacity][hd] (int8 KV cache)
  std::vector<float> k8s, v8s;    // [bands][heads][capacity] per-slot scales
  int capacity = 0;
  int end = 0;

  void clear() { end = 0; }
  void release() {
    k.clear();
    v.clear();
    k16.clear();
    v16.clear();
    k8.clear();
    v8.clear();
    k8s.clear();
    v8s.clear();
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
  std::shared_ptr<MetalLaneState> metal;
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

  enum class Device {
    Cpu = 0,
    Metal = 1,
  };

  enum class QuantMode {
    Fp32 = 0,
    Fp16 = 1,
    Bf16 = 2,
    Int8 = 3,
    Int8Group = 4,
  };

  // Execution device. Cpu is the default; Metal offloads the backbone to GPU.
  void set_device(Device dev);
  Device device() const { return device_; }
  std::string device_description() const;

  // Precision mode for dense layers. Fp32 is the reference-verified default.
  void set_quant(bool on) {
    set_quant_mode(on ? QuantMode::Int8 : QuantMode::Fp32);
  }
  void set_quant_mode(QuantMode mode);
  QuantMode quant_mode() const { return quant_mode_; }
  bool quant_enabled() const { return quant_mode_ != QuantMode::Fp32; }
  void set_fp16_kv(bool on);
  bool fp16_kv() const { return fp16_kv_; }
  // x86 int8 KV cache (per-token quant, fp32 accumulation). Auto-enabled for
  // --quant int8 on AVX2+FMA; silently off where unsupported.
  void set_int8_kv(bool on);
  bool int8_kv() const { return int8_kv_; }
  void prepack_weights() const;

  void init_lane(Lane& lane) const;
  // A lane's start: the recurrence carried, the attention past dropped.
  void clone_lane(const Lane& src, Lane& dst) const;

  // spectra [frames][bins] -> band tokens [bands][frames][width]
  void encode(const std::complex<float>* spectra, int frames, float* feat) const;

  // Band tokens through every block, carrying the lane's state.
  void backbone(float* feat, int frames, Lane& lane, const Rope& rope,
                const LaneParams& lp) const;
  // Batched dual-lane backbone for crossfade windows (halves weight reads).
  void backbone_pair(float* feat_a, float* feat_b, int frames, Lane& lane_a,
                     Lane& lane_b, const Rope& rope,
                     const LaneParams& lp) const;

  // Band tokens -> decoder output [frames][bins][4]
  void decode(const float* feat, int frames, float* raw) const;
  void decode_pair(const float* feat_a, const float* feat_b, int frames,
                   float* raw_a, float* raw_b) const;

  // The decoder's complex change of one frame's spectrum.
  void correction(const float* raw, const std::complex<float>* spectrum,
                  int bins, std::complex<float>* out) const;

  // Optional stage timings (seconds); enabled by the CLI's --profile.
  // attn_*_w accumulate wall time across all pool workers (atomic), so they
  // are comparable with the single-threaded attn_scores/attn_pv fields.
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
    double elem = 0;          // GELU/SiLU/softplus elementwise passes
    double gemm_flops = 0;    // sum of 2*m*k*n over linear_par calls
  double gemm_fp32_flops = 0;  // ... over the fp32 Eigen fallback path
  double gemm_quant = 0;       // activation quantization phase (int8 path)
    std::atomic<double> attn_qk_w{0};    // batched QK dots (all workers)
    std::atomic<double> attn_soft_w{0};  // softmax + masking (all workers)
    std::atomic<double> attn_pv_w{0};    // batched PV (all workers)
    std::atomic<double> attn_ortho_w{0};  // ortho removal (all workers)
    std::atomic<double> attn_store_w{0};   // rope + KV store per query (all workers)
    std::atomic<double> elem_w{0};         // GELU/SiLU/softplus elementwise (all workers)
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
    std::vector<float> a;              // [state][inner] = -exp(A_log)
    std::vector<float> a_log2e4;       // [inner/4][state][4] = a * log2(e)
    std::vector<uint16_t> a_log2e8_16; // [inner/8][state][8] fp16 = a * log2(e)
    std::vector<float> conv_t;         // [taps][inner] depthwise kernel, transposed
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
    // Scratch for SME2 packed activations.
    std::vector<float> sme_ap_f32;
    std::vector<uint16_t> sme_ap_16;
    std::vector<int8_t> sme_ap_i8;
    std::vector<float> sme_as;
    std::vector<float> pair;
  };

  Scratch& scratch(int rows) const;
  // C[m, n] = A[m, k] * W[n, k]^T + bias, columns spread over the pool.
  void linear_par(const float* a, int m, int k, const Tensor& w,
                  const float* bias, float* c) const;
  // The SME2 path (handles fp32, fp16, bf16, int8, int8g when SME2 is available).
  bool linear_par_sme(const float* a, int m, int k, int stride, const Tensor& w,
                      const float* bias, float* c) const;
  void linear_par_avx2(const float* a, int m, int k, int stride, const Tensor& w,
                       const float* bias, float* c) const;
  // CPU int8 dispatch after SME2 has declined the operation.
  bool linear_par_i8(const float* a, int m, int k, int stride, const Tensor& w,
                     const float* bias, float* c) const;
  const Int8Matrix* quantized(const Tensor& w) const;
  const PackedF32* packed_f32(const Tensor& w) const;
  const PackedF16* packed_f16(const Tensor& w) const;
  const PackedBF16* packed_bf16(const Tensor& w) const;
  const PackedI8* packed_i8(const Tensor& w) const;
  const PackedI8Group* packed_i8g(const Tensor& w) const;
  mutable std::unordered_map<const Tensor*, Int8Matrix> quant_;
  mutable std::unordered_map<const Tensor*, Avx2Weights> avx2_weights_;
  mutable std::unordered_map<const Tensor*, PackedF32> sme_f32_;
  mutable std::unordered_map<const Tensor*, PackedF16> sme_f16_;
  mutable std::unordered_map<const Tensor*, PackedBF16> sme_bf16_;
  mutable std::unordered_map<const Tensor*, PackedI8> sme_i8_;
  mutable std::unordered_map<const Tensor*, PackedI8Group> sme_i8g_;
  QuantMode quant_mode_ = QuantMode::Fp32;
  bool fp16_kv_ = false;
  bool int8_kv_ = false;
  Device device_ = Device::Cpu;
  mutable std::unique_ptr<MetalContext> metal_;
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
  void time_transformer_pair(const AttnWeights& w, float* pair_feat, int frames,
                             AttnState& st_a, AttnState& st_b, const Rope& rope,
                             const LaneParams& lp) const;
  void band_transformer(const AttnWeights& w, float* feat, int frames) const;
  void band_transformer_core(const AttnWeights& w, float* trans,
                             int frames) const;
  // One Mamba layer over `rows` tokens: rows = tasks * len, each task a
  // sequence of `len` tokens. Carries conv/ssm state when given.
  void mamba_layer(const MambaLayer& l, const float* x, int rows, int len,
                   float* out, float* conv_state, float* ssm_state,
                   float* slab_ext, float* slab_decay, float* slab_h) const;
  void mamba_layer_pair(const MambaLayer& l, const float* x, int rows_each,
                        int len, float* out, float* conv_a, float* ssm_a,
                        float* conv_b, float* ssm_b, float* slab_ext) const;

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
