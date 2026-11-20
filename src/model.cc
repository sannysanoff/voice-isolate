#include "model.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cstring>

#include "vecmath.h"

namespace vi {

Model::Profile& Model::profile() {
  static Profile instance;
  return instance;
}
bool& Model::profile_enabled() {
  static bool enabled = false;
  return enabled;
}

namespace {

// Scoped wall-clock accumulator.
struct Timer {
  double* slot;
  std::chrono::steady_clock::time_point begin;
  explicit Timer(double* s) : slot(s), begin(std::chrono::steady_clock::now()) {}
  ~Timer() {
    if (slot != nullptr) {
      *slot += std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            begin)
                   .count();
    }
  }
};

constexpr float kPi = 3.14159265358979323846f;

// With ARM64 ldaxr+wfe barriers (~1.5 us), passes above ~150k ops (including
// LayerNorm on 184x512 = 282k ops) speed up cleanly across cores.
#if defined(__aarch64__)
constexpr int64_t kMinParallelOps = 150'000;
#else
constexpr int64_t kMinParallelOps = 2'000'000;
#endif

// Below this the int8 threshold arithmetic and the extra activation pass cost
// more than the four-fold MAC rate saves; dt_proj and friends stay in fp32.
constexpr int64_t kMinQuantOps = 16'000'000;

inline bool worth_splitting(int64_t ops) { return ops >= kMinParallelOps; }

// How many channel ranges to carve each task into: enough to fill the pool,
// but not so fine that the fixed cost per unit dominates.
int channel_split(int tasks, int inner) {
  const int threads = default_pool().size();
  // Round up: the frequency layers have only `frames` tasks (4), and floor
  // division left them running on 4 of the 6 threads.
  const int wanted = std::max(1, threads / std::max(1, tasks));
  const int largest = std::max(1, inner / 128);
  return std::min(wanted, largest);
}

using RowMajor =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using StridedMap =
    Eigen::Map<const RowMajor, 0, Eigen::OuterStride<Eigen::Dynamic>>;

// C[m, n] = A[m, k] * W[n, k]^T + bias, the torch Linear layout.
inline void linear(const float* a, int m, int k, int stride, const Tensor& w,
                   const float* bias, float* c) {
  const int n = w.dim(0);
  if (m <= 0 || n <= 0) return;
  Eigen::Map<RowMajor> out(c, m, n);
  Eigen::Map<const RowMajor> weights(w.data, n, k);
  if (stride == k) {
    Eigen::Map<const RowMajor> in(a, m, k);
    out.noalias() = in * weights.transpose();
  } else {
    StridedMap in(a, m, k, Eigen::OuterStride<Eigen::Dynamic>(stride));
    out.noalias() = in * weights.transpose();
  }
  if (bias != nullptr) {
    Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> b(bias, n);
    out.rowwise() += b;
  }
}

inline void linear(const float* a, int m, int k, const Tensor& w,
                   const float* bias, float* c) {
  linear(a, m, k, k, w, bias, c);
}

// Rotary rotation of one head: [x1, x2] -> [x1 c - x2 s, x1 s + x2 c].
// head_dim is even and a multiple of the lane width, so it runs fully vectorised.
inline void rope_head(float* x, int head_dim, const float* cos,
                      const float* sin, bool inverse) {
  const int half = head_dim / 2;
  const vx::batch sign(inverse ? -1.0f : 1.0f);
  int i = 0;
  for (; i + vx::lanes <= half; i += vx::lanes) {
    const vx::batch a = loadf(x + i);
    const vx::batch b = loadf(x + i + half);
    const vx::batch c = loadf(cos + i);
    const vx::batch s = loadf(sin + i) * sign;
    storef(x + i, a * c - b * s);
    storef(x + i + half, a * s + b * c);
  }
  const float sgn = inverse ? -1.0f : 1.0f;
  for (; i < half; ++i) {
    const float a = x[i];
    const float b = x[i + half];
    x[i] = a * cos[i] - sgn * b * sin[i];
    x[i + half] = sgn * a * sin[i] + b * cos[i];
  }
}

// Vector helpers for the attention inner loops: contiguous `count`-wide
// vectors, four keys at a time so one load of the query feeds four dots.
inline float dot(const float* a, const float* b, int count) {
  vx::batch acc = vx::batch(0.0f);
  int i = 0;
  for (; i + vx::lanes <= count; i += vx::lanes)
    acc += loadf(a + i) * loadf(b + i);
  float sum = xsimd::reduce_add(acc);
  for (; i < count; ++i) sum += a[i] * b[i];
  return sum;
}

inline void dot4(const float* q, const float* k0, const float* k1,
                 const float* k2, const float* k3, int count, float* out) {
  vx::batch a = vx::batch(0.0f), b = vx::batch(0.0f), c = vx::batch(0.0f),
            d = vx::batch(0.0f);
  int i = 0;
  for (; i + vx::lanes <= count; i += vx::lanes) {
    const vx::batch qv = loadf(q + i);
    a += qv * loadf(k0 + i);
    b += qv * loadf(k1 + i);
    c += qv * loadf(k2 + i);
    d += qv * loadf(k3 + i);
  }
  float s0 = xsimd::reduce_add(a), s1 = xsimd::reduce_add(b),
        s2 = xsimd::reduce_add(c), s3 = xsimd::reduce_add(d);
  for (; i < count; ++i) {
    s0 += q[i] * k0[i];
    s1 += q[i] * k1[i];
    s2 += q[i] * k2[i];
    s3 += q[i] * k3[i];
  }
  out[0] = s0;
  out[1] = s1;
  out[2] = s2;
  out[3] = s3;
}

inline void axpy(float* out, float scale, const float* x, int count) {
  const vx::batch s(scale);
  int i = 0;
  for (; i + vx::lanes <= count; i += vx::lanes)
    storef(out + i, loadf(out + i) + s * loadf(x + i));
  for (; i < count; ++i) out[i] += scale * x[i];
}

// o = sum_j s[j] * inv * values[j], with the values head-major (`count` floats
// per row). The first row is a store, so the destination needs no zeroing.
inline void pv_row(const float* s, float inv, const float* values, int n,
                   int count, float* o) {
  const vx::batch p(s[0] * inv);
  int i = 0;
  for (; i + vx::lanes <= count; i += vx::lanes)
    storef(o + i, p * loadf(values + i));
  for (; i < count; ++i) o[i] = s[0] * inv * values[i];
  for (int j = 1; j < n; ++j) axpy(o, s[j] * inv, values + static_cast<size_t>(j) * count, count);
}

// Scores of one query against `n` keys, keys laid out head-major with `hd`
// floats per key. Leaves exp(score - max) in `s` and returns the sum.
inline float scores_softmax(const float* q, const float* keys, int n, int hd,
                            float scale, float* s) {
  float maxs = -INFINITY;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    dot4(q, keys + static_cast<size_t>(i) * hd,
         keys + static_cast<size_t>(i + 1) * hd,
         keys + static_cast<size_t>(i + 2) * hd,
         keys + static_cast<size_t>(i + 3) * hd, hd, s + i);
    for (int k = 0; k < 4; ++k) {
      s[i + k] *= scale;
      maxs = std::max(maxs, s[i + k]);
    }
  }
  for (; i < n; ++i) {
    const float acc = dot(q, keys + static_cast<size_t>(i) * hd, hd) * scale;
    s[i] = acc;
    maxs = std::max(maxs, acc);
  }
  const vx::batch vmax(maxs);
  (void)vmax;
  for (int j = 0; j < n; ++j) s[j] -= maxs;
  v_exp(s, n);
  (void)vmax;
  return v_sum(s, n);
}
// Grows a lane's attention buffers to hold `slots`, keeping what is written.
// Keys are stored head-major, [band][head][slot][hd], so a growth has to copy
// each head's run separately: the head stride changes with the capacity.
void reserve_slots(AttnState& st, int slots, int limit, int bands, int width,
                   int head_dim) {
  if (slots <= st.capacity) return;
  int capacity = std::max(slots, std::max(2 * st.capacity, 64));
  if (capacity > limit) capacity = limit;
  const int heads = width / head_dim;
  const size_t old_band = static_cast<size_t>(st.capacity) * width;
  const size_t new_band = static_cast<size_t>(capacity) * width;
  std::vector<float> k(static_cast<size_t>(bands) * new_band);
  std::vector<float> v(k.size());
  if (st.end > 0) {
    const size_t run = sizeof(float) * static_cast<size_t>(st.end) * head_dim;
    for (int b = 0; b < bands; ++b) {
      for (int h = 0; h < heads; ++h) {
        const size_t old_at = static_cast<size_t>(b) * old_band +
                              static_cast<size_t>(h) * st.capacity * head_dim;
        const size_t new_at = static_cast<size_t>(b) * new_band +
                              static_cast<size_t>(h) * capacity * head_dim;
        std::memcpy(k.data() + new_at, st.k.data() + old_at, run);
        std::memcpy(v.data() + new_at, st.v.data() + old_at, run);
      }
    }
  }
  st.k.swap(k);
  st.v.swap(v);
  st.capacity = capacity;
}

void reserve_slots_f16(AttnState& st, int slots, int limit, int bands,
                       int width, int head_dim) {
  if (slots <= st.capacity) return;
  int capacity = std::max(slots, std::max(2 * st.capacity, 64));
  if (capacity > limit) capacity = limit;
  const int heads = width / head_dim;
  const size_t old_band = static_cast<size_t>(st.capacity) * width;
  const size_t new_band = static_cast<size_t>(capacity) * width;
  std::vector<uint16_t> k(static_cast<size_t>(bands) * new_band);
  std::vector<uint16_t> v(k.size());
  if (st.end > 0) {
    const size_t run =
        sizeof(uint16_t) * static_cast<size_t>(st.end) * head_dim;
    for (int b = 0; b < bands; ++b) {
      for (int h = 0; h < heads; ++h) {
        const size_t old_at = static_cast<size_t>(b) * old_band +
                              static_cast<size_t>(h) * st.capacity * head_dim;
        const size_t new_at = static_cast<size_t>(b) * new_band +
                              static_cast<size_t>(h) * capacity * head_dim;
        std::memcpy(k.data() + new_at, st.k16.data() + old_at, run);
        std::memcpy(v.data() + new_at, st.v16.data() + old_at, run);
      }
    }
  }
  st.k16.swap(k);
  st.v16.swap(v);
  st.capacity = capacity;
}

// [bands][frames][width] <-> [frames][bands][width]
void to_frame_major(const float* feat, float* out, int bands, int frames,
                    int width) {
  for (int b = 0; b < bands; ++b)
    for (int t = 0; t < frames; ++t)
      std::memcpy(out + (static_cast<size_t>(t) * bands + b) * width,
                  feat + (static_cast<size_t>(b) * frames + t) * width,
                  sizeof(float) * width);
}

void to_band_major(const float* in, float* feat, int bands, int frames,
                   int width) {
  for (int b = 0; b < bands; ++b)
    for (int t = 0; t < frames; ++t)
      std::memcpy(feat + (static_cast<size_t>(b) * frames + t) * width,
                  in + (static_cast<size_t>(t) * bands + b) * width,
                  sizeof(float) * width);
}

}  // namespace

// ------------------------------------------------------------------ rope ----

void Rope::init(int head_dim, int positions) {
  half = head_dim / 2;
  cos.resize(static_cast<size_t>(positions) * half);
  sin.resize(cos.size());
  for (int p = 0; p < positions; ++p) {
    for (int i = 0; i < half; ++i) {
      const float inv = std::pow(10000.0f, -static_cast<float>(i) /
                                                 static_cast<float>(half));
      const float angle = static_cast<float>(p) * inv;
      cos[static_cast<size_t>(p) * half + i] = std::cos(angle);
      sin[static_cast<size_t>(p) * half + i] = std::sin(angle);
    }
  }
}

// ----------------------------------------------------------------- model ----

Model::Model(const std::string& dir) : cfg_(load_config(dir)), fft_(cfg_.n_fft) {
  weights_.load(dir);
  window_.resize(static_cast<size_t>(cfg_.n_fft));
  for (int n = 0; n < cfg_.n_fft; ++n)
    window_[static_cast<size_t>(n)] =
        0.5f * (1.0f - std::cos(2.0f * kPi * static_cast<float>(n) /
                               static_cast<float>(cfg_.n_fft)));
  load_blocks();
  prepack_weights();
}

void Model::set_quant_mode(QuantMode mode) {
  quant_mode_ = mode;
  prepack_weights();
}

void Model::set_fp16_kv(bool on) { fp16_kv_ = on && fp16_neon_available(); }

void Model::prepack_weights() const {
  if (!sme2_available()) return;
  std::vector<const Tensor*> list;
  list.reserve(blocks_.size() * 14);
  for (const Block& b : blocks_) {
    if (b.transformer) {
      const AttnWeights& a = b.attn;
      for (const Tensor* t :
           {a.qkv_w, a.proj_w, a.f0w, a.f1w, a.in_w, a.out_w, a.bl1w, a.bl2w})
        list.push_back(t);
    } else {
      for (const MambaLayer* l : {&b.time_mamba.forward, &b.freq_mamba.forward,
                                  &b.freq_mamba.backward}) {
        for (const Tensor* t : {l->in_w, l->x_w, l->dt_w, l->out_w})
          list.push_back(t);
      }
      list.push_back(b.time_mamba.out_w);
      list.push_back(b.freq_mamba.out_w);
    }
  }
  const int count = static_cast<int>(list.size());
  // Reserve map slots up front so parallel workers write disjoint entries.
  const QuantMode mode = quant_mode_;
  if (mode == QuantMode::Fp32) {
    std::vector<std::pair<const Tensor*, PackedF32*>> work;
    for (const Tensor* t : list) {
      if (sme_f32_.find(t) == sme_f32_.end())
        work.emplace_back(t, &sme_f32_[t]);
    }
    default_pool().parallel_for(static_cast<int>(work.size()), [&](int i) {
      const Tensor* t = work[static_cast<size_t>(i)].first;
      sme_pack_weights_f32(t->data, t->dim(0), t->dim(1),
                           *work[static_cast<size_t>(i)].second);
    });
  } else if (mode == QuantMode::Fp16) {
    std::vector<std::pair<const Tensor*, PackedF16*>> work;
    for (const Tensor* t : list) {
      if (t->dim(1) % 2 == 0 && sme_f16_.find(t) == sme_f16_.end())
        work.emplace_back(t, &sme_f16_[t]);
    }
    default_pool().parallel_for(static_cast<int>(work.size()), [&](int i) {
      const Tensor* t = work[static_cast<size_t>(i)].first;
      sme_pack_weights_f16(t->data, t->dim(0), t->dim(1),
                           *work[static_cast<size_t>(i)].second);
    });
  } else if (mode == QuantMode::Bf16) {
    std::vector<std::pair<const Tensor*, PackedBF16*>> work;
    for (const Tensor* t : list) {
      if (t->dim(1) % 2 == 0 && sme_bf16_.find(t) == sme_bf16_.end())
        work.emplace_back(t, &sme_bf16_[t]);
    }
    default_pool().parallel_for(static_cast<int>(work.size()), [&](int i) {
      const Tensor* t = work[static_cast<size_t>(i)].first;
      sme_pack_weights_bf16(t->data, t->dim(0), t->dim(1),
                            *work[static_cast<size_t>(i)].second);
    });
  } else if (mode == QuantMode::Int8) {
    std::vector<std::pair<const Tensor*, PackedI8*>> work;
    std::vector<std::pair<const Tensor*, PackedF32*>> work_f32;
    for (const Tensor* t : list) {
      if (t->dim(1) % 4 == 0 && sme_i8_.find(t) == sme_i8_.end())
        work.emplace_back(t, &sme_i8_[t]);
      if (sme_f32_.find(t) == sme_f32_.end())
        work_f32.emplace_back(t, &sme_f32_[t]);
    }
    default_pool().parallel_for(static_cast<int>(work.size()), [&](int i) {
      const Tensor* t = work[static_cast<size_t>(i)].first;
      sme_pack_weights_i8(t->data, t->dim(0), t->dim(1),
                          *work[static_cast<size_t>(i)].second);
    });
    default_pool().parallel_for(static_cast<int>(work_f32.size()), [&](int i) {
      const Tensor* t = work_f32[static_cast<size_t>(i)].first;
      sme_pack_weights_f32(t->data, t->dim(0), t->dim(1),
                           *work_f32[static_cast<size_t>(i)].second);
    });
  } else if (mode == QuantMode::Int8Group) {
    std::vector<std::pair<const Tensor*, PackedI8Group*>> work;
    for (const Tensor* t : list) {
      if (t->dim(1) % 4 == 0 && sme_i8g_.find(t) == sme_i8g_.end())
        work.emplace_back(t, &sme_i8g_[t]);
    }
    default_pool().parallel_for(static_cast<int>(work.size()), [&](int i) {
      const Tensor* t = work[static_cast<size_t>(i)].first;
      sme_pack_weights_i8g(t->data, t->dim(0), t->dim(1), 64,
                           *work[static_cast<size_t>(i)].second);
    });
  }
  (void)count;
}

void Model::load_blocks() {
  const int width = cfg_.width;
  const int bands = cfg_.bands();
  const int inner = cfg_.inner();
  const int state = cfg_.mamba_state;
  const int conv = cfg_.mamba_conv;
  const int dt_rank = cfg_.dt_rank();
  const int dt_total = dt_rank + 2 * state;

  for (int b = 0; b < bands; ++b) {
    const std::string pb = "encoders." + std::to_string(b);
    const auto& edge = cfg_.edges[static_cast<size_t>(b)];
    const int w2 = 2 * (edge.second - edge.first);
    enc_norm_w_.push_back(&weights_.get(pb + ".0.weight", {w2}));
    enc_norm_b_.push_back(&weights_.get(pb + ".0.bias", {w2}));
    enc_w_.push_back(&weights_.get(pb + ".1.weight", {width, w2}));
    enc_b_.push_back(&weights_.get(pb + ".1.bias", {width}));

    const std::string ph = "denoise_decoder.heads." + std::to_string(b);
    const int out = 4 * (edge.second - edge.first);
    BandHead head;
    head.n_w = &weights_.get(ph + ".0.weight", {width});
    head.n_b = &weights_.get(ph + ".0.bias", {width});
    head.h1_w = &weights_.get(ph + ".1.weight", {width, width});
    head.h1_b = &weights_.get(ph + ".1.bias", {width});
    head.h3_w = &weights_.get(ph + ".3.weight", {out, width});
    head.h3_b = &weights_.get(ph + ".3.bias", {out});
    heads_.push_back(head);
  }
  // The decoder conditions on the fixed control vector [1, 0]: scale and shift
  // are that row of the conditioning projection plus its bias.
  {
    const Tensor& cond_w =
        weights_.get("denoise_decoder.condition.weight", {2 * width, 2});
    const Tensor& cond_b =
        weights_.get("denoise_decoder.condition.bias", {2 * width});
    decoder_scale_.resize(static_cast<size_t>(width));
    decoder_shift_.resize(static_cast<size_t>(width));
    for (int j = 0; j < width; ++j) {
      decoder_scale_[static_cast<size_t>(j)] =
          cond_w.data[static_cast<size_t>(j) * 2] + cond_b.data[j];
      decoder_shift_[static_cast<size_t>(j)] =
          cond_w.data[static_cast<size_t>(width + j) * 2] + cond_b.data[width + j];
    }
  }
  band_embedding_ = &weights_.get("band_embedding", {bands, width});
  lookahead_embedding_ =
      &weights_.get("lookahead_embedding.weight", {3, width});

  auto load_layer = [&](const std::string& base) {
    MambaLayer l;
    l.in_w = &weights_.get(base + ".in_proj.weight", {2 * inner, width});
    l.conv_w = &weights_.get(base + ".conv1d.weight", {inner, 1, conv});
    l.conv_b = &weights_.get(base + ".conv1d.bias", {inner});
    l.x_w = &weights_.get(base + ".x_proj.weight", {dt_total, inner});
    l.dt_w = &weights_.get(base + ".dt_proj.weight", {inner, dt_rank});
    l.dt_b = &weights_.get(base + ".dt_proj.bias", {inner});
    l.d = &weights_.get(base + ".D", {inner});
    l.out_w = &weights_.get(base + ".out_proj.weight", {width, inner});
    const Tensor& a_log = weights_.get(base + ".A_log", {inner, state});
    // -exp(A_log), laid out [state][inner], [inner/4][state][4] * log2(e),
    // and [inner/8][state][8] fp16 * log2(e) for the fast fp16 scan.
    l.a.resize(static_cast<size_t>(state) * inner);
    l.a_log2e4.resize(static_cast<size_t>(state) * inner);
    std::vector<float> a_log2e8_f32(static_cast<size_t>(state) * inner);
    constexpr float kLog2E = 1.44269504088896341f;
    for (int c = 0; c < inner; ++c)
      for (int n = 0; n < state; ++n) {
        const float val = -std::exp(a_log.data[static_cast<size_t>(c) * state + n]);
        l.a[static_cast<size_t>(n) * inner + c] = val;
        l.a_log2e4[(static_cast<size_t>(c / 4) * state + n) * 4 + (c % 4)] =
            val * kLog2E;
        a_log2e8_f32[(static_cast<size_t>(c / 8) * state + n) * 8 + (c % 8)] =
            val * kLog2E;
      }
    if (fp16_neon_available()) {
      l.a_log2e8_16.resize(a_log2e8_f32.size());
      v_f32_to_f16(a_log2e8_f32.data(), l.a_log2e8_16.data(),
                   static_cast<int>(a_log2e8_f32.size()));
    }
    // conv kernel as [tap][channel], contiguous over channels.
    l.conv_t.resize(static_cast<size_t>(conv) * inner);
    for (int c = 0; c < inner; ++c)
      for (int j = 0; j < conv; ++j)
        l.conv_t[static_cast<size_t>(j) * inner + c] =
            l.conv_w->data[static_cast<size_t>(c) * conv + j];
    return l;
  };

  auto load_output = [&](MambaWeights& m, const std::string& base,
                         const std::vector<int>& shape) {
    m.out_w = &weights_.get(base + ".output_proj.weight", shape);
    m.out_b = &weights_.get(base + ".output_proj.bias", {shape[0]});
    m.norm_w = &weights_.get(base + ".norm.weight", {width});
    m.norm_b = &weights_.get(base + ".norm.bias", {width});
  };

  for (int i = 0; i < cfg_.num_blocks; ++i) {
    const std::string p = "blocks." + std::to_string(i);
    Block block;
    if (cfg_.block_types[static_cast<size_t>(i)] == "transformer") {
      block.transformer = true;
      AttnWeights& a = block.attn;
      a.n1w = &weights_.get(p + ".time_transformer.norm1.weight", {width});
      a.n1b = &weights_.get(p + ".time_transformer.norm1.bias", {width});
      a.n2w = &weights_.get(p + ".time_transformer.norm2.weight", {width});
      a.n2b = &weights_.get(p + ".time_transformer.norm2.bias", {width});
      a.qkv_w = &weights_.get(p + ".time_transformer.attn.qkv.weight",
                              {3 * width, width});
      a.qkv_b =
          &weights_.get(p + ".time_transformer.attn.qkv.bias", {3 * width});
      a.proj_w = &weights_.get(p + ".time_transformer.attn.proj.weight",
                               {width, width});
      a.proj_b = &weights_.get(p + ".time_transformer.attn.proj.bias", {width});
      a.f0w = &weights_.get(p + ".time_transformer.ffn.0.weight",
                            {4 * width, width});
      a.f0b = &weights_.get(p + ".time_transformer.ffn.0.bias", {4 * width});
      a.f1w = &weights_.get(p + ".time_transformer.ffn.3.weight",
                            {width, 4 * width});
      a.f1b = &weights_.get(p + ".time_transformer.ffn.3.bias", {width});
      a.bn1w = &weights_.get(p + ".band_transformer.norm1.weight", {width});
      a.bn1b = &weights_.get(p + ".band_transformer.norm1.bias", {width});
      a.bn2w = &weights_.get(p + ".band_transformer.norm2.weight", {width});
      a.bn2b = &weights_.get(p + ".band_transformer.norm2.bias", {width});
      a.in_w = &weights_.get(p + ".band_transformer.self_attn.in_proj_weight",
                             {3 * width, width});
      a.in_b = &weights_.get(p + ".band_transformer.self_attn.in_proj_bias",
                             {3 * width});
      a.out_w = &weights_.get(p + ".band_transformer.self_attn.out_proj.weight",
                              {width, width});
      a.out_b = &weights_.get(p + ".band_transformer.self_attn.out_proj.bias",
                              {width});
      a.bl1w = &weights_.get(p + ".band_transformer.linear1.weight",
                             {4 * width, width});
      a.bl1b =
          &weights_.get(p + ".band_transformer.linear1.bias", {4 * width});
      a.bl2w = &weights_.get(p + ".band_transformer.linear2.weight",
                             {width, 4 * width});
      a.bl2b = &weights_.get(p + ".band_transformer.linear2.bias", {width});
    } else {
      block.time_mamba.forward = load_layer(p + ".time_mamba.forward_blocks");
      load_output(block.time_mamba, p + ".time_mamba", {width, width});
      block.time_mamba.causal = true;

      block.freq_mamba.forward =
          load_layer(p + ".freq_mamba.forward_blocks");
      block.freq_mamba.backward =
          load_layer(p + ".freq_mamba.backward_blocks");
      load_output(block.freq_mamba, p + ".freq_mamba", {width, 2 * width});
      block.freq_mamba.causal = false;
    }
    blocks_.push_back(block);
  }
}

Model::Scratch& Model::scratch(int rows) const {
  Scratch& s = scratch_;
  if (s.rows >= rows) return s;
  const int width = cfg_.width;
  const int inner = cfg_.inner();
  const int dt_total = cfg_.dt_rank() + 2 * cfg_.mamba_state;
  s.rows = rows;
  s.norm.resize(static_cast<size_t>(rows) * width);
  s.qkv.resize(static_cast<size_t>(rows) * 3 * width);
  s.attn.resize(static_cast<size_t>(rows) * width);
  s.ffn.resize(static_cast<size_t>(rows) * 4 * width);
  s.trans.resize(static_cast<size_t>(rows) * width);
  s.rev.resize(static_cast<size_t>(rows) * width);
  s.bwd.resize(static_cast<size_t>(rows) * width);
  s.cat.resize(static_cast<size_t>(rows) * 2 * width);
  s.xz.resize(static_cast<size_t>(rows) * 2 * inner);
  s.u2.resize(static_cast<size_t>(rows) * inner);
  s.y.resize(static_cast<size_t>(rows) * inner);
  s.delta.resize(static_cast<size_t>(rows) * inner);
  s.xp.resize(static_cast<size_t>(rows) * dt_total);
  return s;
}

const Int8Matrix* Model::quantized(const Tensor& w) const {
  // Keyed by the weight pointer inside the loaded blob, which never moves.
  auto it = quant_.find(&w);
  if (it != quant_.end()) return &it->second;
  Int8Matrix matrix;
  quantize_rows(w.data, w.dim(0), w.dim(1), matrix);
  return &quant_.emplace(&w, std::move(matrix)).first->second;
}

const PackedF32* Model::packed_f32(const Tensor& w) const {
  auto it = sme_f32_.find(&w);
  if (it != sme_f32_.end()) return &it->second;
  PackedF32 p;
  sme_pack_weights_f32(w.data, w.dim(0), w.dim(1), p);
  return &sme_f32_.emplace(&w, std::move(p)).first->second;
}

const PackedF16* Model::packed_f16(const Tensor& w) const {
  auto it = sme_f16_.find(&w);
  if (it != sme_f16_.end()) return &it->second;
  PackedF16 p;
  sme_pack_weights_f16(w.data, w.dim(0), w.dim(1), p);
  return &sme_f16_.emplace(&w, std::move(p)).first->second;
}

const PackedBF16* Model::packed_bf16(const Tensor& w) const {
  auto it = sme_bf16_.find(&w);
  if (it != sme_bf16_.end()) return &it->second;
  PackedBF16 p;
  sme_pack_weights_bf16(w.data, w.dim(0), w.dim(1), p);
  return &sme_bf16_.emplace(&w, std::move(p)).first->second;
}

const PackedI8* Model::packed_i8(const Tensor& w) const {
  auto it = sme_i8_.find(&w);
  if (it != sme_i8_.end()) return &it->second;
  PackedI8 p;
  sme_pack_weights_i8(w.data, w.dim(0), w.dim(1), p);
  return &sme_i8_.emplace(&w, std::move(p)).first->second;
}

const PackedI8Group* Model::packed_i8g(const Tensor& w) const {
  auto it = sme_i8g_.find(&w);
  if (it != sme_i8g_.end()) return &it->second;
  PackedI8Group p;
  sme_pack_weights_i8g(w.data, w.dim(0), w.dim(1), /*group_size=*/64, p);
  return &sme_i8g_.emplace(&w, std::move(p)).first->second;
}

bool Model::linear_par_sme(const float* a, int m, int k, int stride,
                           const Tensor& w, const float* bias,
                           float* c) const {
  if (!sme2_available()) return false;
  const int n = w.dim(0);
  if (m <= 0 || n <= 0 || k <= 0) return true;
  const int64_t ops = 2 * static_cast<int64_t>(m) * k * n;
  const int mb = (m + 15) / 16;
  const size_t ap_elems = static_cast<size_t>(mb) * k * 16;
  Scratch& sc = scratch(m);

  // Choose effective precision: very small layers (e.g. dt_proj with k=32)
  // stay in fp32 for int8 modes to avoid quantization overhead & accuracy loss.
  QuantMode mode = quant_mode_;
  if ((mode == QuantMode::Int8 || mode == QuantMode::Int8Group) &&
      (k % 4 != 0 || ops < kMinQuantOps)) {
    mode = QuantMode::Fp32;
  }
  if ((mode == QuantMode::Fp16 || mode == QuantMode::Bf16) && (k % 2 != 0)) {
    mode = QuantMode::Fp32;
  }

  const int threads = default_pool().size();
  // Apple M4 Max has 2 SME2 units (1 per 5-core P-cluster). Running more than
  // 2 SME threads concurrently causes severe ZA/arbiter contention across cores
  // in the same cluster (measured 1.37 s at 2 chunks vs 2.43 s at 8 chunks).
  const int max_sme_chunks = []() {
    if (const char* env = std::getenv("VI_SME_CHUNKS")) {
      const int v = std::atoi(env);
      if (v >= 1) return v;
    }
    return 2;
  }();
  const int tile_n = (n >= 64) ? 32 : 16;
  const int num_tiles = (n + tile_n - 1) / tile_n;
  const int chunks =
      worth_splitting(ops)
          ? std::min(std::min(threads, max_sme_chunks), std::max(1, num_tiles))
          : 1;

  auto dispatch = [&](auto fn) {
    if (chunks <= 1) {
      fn(0, n);
    } else {
      default_pool().parallel_for(chunks, [&](int idx) {
        const int t0 = static_cast<int>(static_cast<int64_t>(num_tiles) * idx /
                                        chunks);
        const int t1 = static_cast<int>(
            static_cast<int64_t>(num_tiles) * (idx + 1) / chunks);
        const int n0 = std::min(n, t0 * tile_n);
        const int n1 = std::min(n, t1 * tile_n);
        if (n1 > n0) fn(n0, n1);
      });
    }
  };

  if (mode == QuantMode::Fp32) {
    const PackedF32* pw = packed_f32(w);
    if (sc.sme_ap_f32.size() < ap_elems) sc.sme_ap_f32.resize(ap_elems);
    sme_pack_a_f32(a, m, k, stride, sc.sme_ap_f32.data());
    dispatch([&](int n0, int n1) {
      sme_gemm_f32_cols(sc.sme_ap_f32.data(), *pw, bias, c, m, n, n0, n1);
    });
    return true;
  }
  if (mode == QuantMode::Fp16) {
    const PackedF16* pw = packed_f16(w);
    if (sc.sme_ap_16.size() < ap_elems) sc.sme_ap_16.resize(ap_elems);
    sme_pack_a_f16(a, m, k, stride, sc.sme_ap_16.data());
    dispatch([&](int n0, int n1) {
      sme_gemm_f16_cols(sc.sme_ap_16.data(), *pw, bias, c, m, n, n0, n1);
    });
    return true;
  }
  if (mode == QuantMode::Bf16) {
    const PackedBF16* pw = packed_bf16(w);
    if (sc.sme_ap_16.size() < ap_elems) sc.sme_ap_16.resize(ap_elems);
    sme_pack_a_bf16(a, m, k, stride, sc.sme_ap_16.data());
    dispatch([&](int n0, int n1) {
      sme_gemm_bf16_cols(sc.sme_ap_16.data(), *pw, bias, c, m, n, n0, n1);
    });
    return true;
  }
  if (mode == QuantMode::Int8) {
    const PackedI8* pw = packed_i8(w);
    if (sc.sme_ap_i8.size() < ap_elems) sc.sme_ap_i8.resize(ap_elems);
    if (sc.sme_as.size() < static_cast<size_t>(m)) sc.sme_as.resize(m);
    if (threads > 1 && mb >= 4) {
      const int pchunks = std::min(threads, mb);
      default_pool().parallel_for(pchunks, [&](int idx) {
        const int b0 = mb * idx / pchunks;
        const int b1 = mb * (idx + 1) / pchunks;
        for (int b = b0; b < b1; ++b) {
          const int m0 = b * 16;
          const int mr = std::min(16, m - m0);
          v_quant_pack_a_i8_block(
              a, m0, mr, k, stride,
              sc.sme_ap_i8.data() + static_cast<size_t>(b) * k * 16,
              sc.sme_as.data());
        }
      });
    } else {
      for (int b = 0; b < mb; ++b) {
        const int m0 = b * 16;
        const int mr = std::min(16, m - m0);
        v_quant_pack_a_i8_block(
            a, m0, mr, k, stride,
            sc.sme_ap_i8.data() + static_cast<size_t>(b) * k * 16,
            sc.sme_as.data());
      }
    }
    dispatch([&](int n0, int n1) {
      sme_gemm_i8_cols(sc.sme_ap_i8.data(), sc.sme_as.data(), *pw, bias, c, m,
                       n, n0, n1);
    });
    return true;
  }
  // QuantMode::Int8Group
  const PackedI8Group* pw = packed_i8g(w);
  const int ng = (k + pw->group_size - 1) / pw->group_size;
  if (sc.sme_ap_i8.size() < ap_elems) sc.sme_ap_i8.resize(ap_elems);
  if (sc.sme_as.size() < static_cast<size_t>(ng) * m)
    sc.sme_as.resize(static_cast<size_t>(ng) * m);
  sme_quant_pack_a_i8g(a, m, k, stride, pw->group_size, sc.sme_ap_i8.data(),
                       sc.sme_as.data());
  dispatch([&](int n0, int n1) {
    sme_gemm_i8g_cols(sc.sme_ap_i8.data(), sc.sme_as.data(), *pw, bias, c, m, n,
                      n0, n1);
  });
  return true;
}

bool Model::linear_par_i8(const float* a, int m, int k, int stride,
                          const Tensor& w, const float* bias, float* c) const {
  if (quant_mode_ != QuantMode::Int8 || !int8_available()) return false;
  const int n = w.dim(0);
  // sdot consumes 16 int8 per instruction and the kernel tiles four columns.
  if (k % 16 != 0 || n % 4 != 0) return false;
  if (2 * static_cast<int64_t>(m) * k * n < kMinQuantOps) return false;
  const int threads = default_pool().size();
  const int chunks = std::min(threads, std::max(1, n / 64));
  const Int8Matrix* qw = quantized(w);
  Scratch& sc = scratch(m);
  const size_t need_q = static_cast<size_t>(chunks) * m * k;
  if (sc.q8.size() < need_q) sc.q8.resize(need_q);
  if (sc.q8s.size() < static_cast<size_t>(chunks) * m)
    sc.q8s.resize(static_cast<size_t>(chunks) * m);
  // Split on 4-column tiles so every worker covers whole tiles: a boundary
  // that is not a multiple of 4 leaves a scalar tail that dominates its slice.
  const int tiles = n / 4;
  default_pool().parallel_for(chunks, [&](int index) {
    const int j0 = static_cast<int>(4 * static_cast<int64_t>(tiles) * index / chunks);
    const int j1 =
        static_cast<int>(4 * static_cast<int64_t>(tiles) * (index + 1) / chunks);
    if (j1 <= j0) return;
    gemm_int8_columns(a, m, k, stride, *qw, bias, c, j0, j1,
                      sc.q8.data() + static_cast<size_t>(index) * m * k,
                      sc.q8s.data() + static_cast<size_t>(index) * m);
  });
  return true;
}

void Model::linear_par(const float* a, int m, int k, int stride, const Tensor& w,
                       const float* bias, float* c) const {
  Timer timer(profile_enabled() ? &profile().gemm : nullptr);
  if (linear_par_sme(a, m, k, stride, w, bias, c)) return;
  if (linear_par_i8(a, m, k, stride, w, bias, c)) return;
  const int n = w.dim(0);
  const int threads = default_pool().size();
  if (!worth_splitting(2 * static_cast<int64_t>(m) * k * n) || threads == 1) {
    linear(a, m, k, stride, w, bias, c);
    return;
  }
  // Split the output columns: the tokens are few and shared, so splitting rows
  // would make every thread stream the whole weight matrix. Narrow layers
  // (x_proj and dt_proj have n = 64) cannot be split by columns, and there the
  // weight is small enough that a row split is the cheap way to use the pool.
  const int chunks = std::min(threads, std::max(1, n / 64));
  if (chunks <= 1) {
    const int row_chunks = std::min(threads, std::max(1, m / 8));
    if (row_chunks <= 1 || static_cast<int64_t>(n) * k > 1 << 16) {
      linear(a, m, k, stride, w, bias, c);
      return;
    }
    default_pool().parallel_for(row_chunks, [&](int index) {
      const int m0 = static_cast<int>(static_cast<int64_t>(m) * index / row_chunks);
      const int m1 =
          static_cast<int>(static_cast<int64_t>(m) * (index + 1) / row_chunks);
      if (m1 <= m0) return;
      linear(a + static_cast<size_t>(m0) * stride, m1 - m0, k, stride, w, bias,
             c + static_cast<size_t>(m0) * n);
    });
    return;
  }
  StridedMap in(a, m, k, Eigen::OuterStride<Eigen::Dynamic>(stride));
  default_pool().parallel_for(chunks, [&](int index) {
    const int n0 = static_cast<int>(static_cast<int64_t>(n) * index / chunks);
    const int n1 =
        static_cast<int>(static_cast<int64_t>(n) * (index + 1) / chunks);
    if (n1 <= n0) return;
    Eigen::Map<const RowMajor> tile(w.data + static_cast<size_t>(n0) * k,
                                    n1 - n0, k);
    Eigen::Map<RowMajor, 0, Eigen::OuterStride<Eigen::Dynamic>> out(
        c + n0, m, n1 - n0, Eigen::OuterStride<Eigen::Dynamic>(n));
    out.noalias() = in * tile.transpose();
    if (bias != nullptr) {
      Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> b(bias + n0,
                                                                 n1 - n0);
      out.rowwise() += b;
    }
  });
}

void Model::linear_par(const float* a, int m, int k, const Tensor& w,
                       const float* bias, float* c) const {
  linear_par(a, m, k, k, w, bias, c);
}

void Model::layer_norm_par(const float* x, float* y, int rows, int cols,
                           const float* w, const float* b) const {
  Timer timer(profile_enabled() ? &profile().ln : nullptr);
  if (!worth_splitting(3 * static_cast<int64_t>(rows) * cols)) {
    layer_norm(x, y, rows, cols, w, b);
    return;
  }
  const int chunks = std::min(default_pool().size(), std::max(1, rows / 8));
  if (chunks <= 1) {
    layer_norm(x, y, rows, cols, w, b);
    return;
  }
  default_pool().parallel_for(chunks, [&](int index) {
    const int a = static_cast<int>(static_cast<int64_t>(rows) * index / chunks);
    const int c =
        static_cast<int>(static_cast<int64_t>(rows) * (index + 1) / chunks);
    if (c > a)
      layer_norm(x + static_cast<size_t>(a) * cols, y + static_cast<size_t>(a) * cols,
                 c - a, cols, w, b);
  });
}

void Model::add_par(float* dst, const float* src, int n) const {
  Timer timer(profile_enabled() ? &profile().add : nullptr);
  if (!worth_splitting(n)) {
    for (int i = 0; i < n; ++i) dst[i] += src[i];
    return;
  }
  const int chunks = std::min(default_pool().size(), std::max(1, n / 16384));
  if (chunks <= 1) {
    for (int i = 0; i < n; ++i) dst[i] += src[i];
    return;
  }
  default_pool().parallel_for(chunks, [&](int index) {
    const int a = static_cast<int>(static_cast<int64_t>(n) * index / chunks);
    const int b =
        static_cast<int>(static_cast<int64_t>(n) * (index + 1) / chunks);
    for (int i = a; i < b; ++i) dst[i] += src[i];
  });
}

void Model::elementwise_par(float* x, int n, void (*fn)(float*, int),
                            int cost_per_element) const {
  if (!worth_splitting(static_cast<int64_t>(n) * cost_per_element)) {
    fn(x, n);
    return;
  }
  const int chunks = std::min(default_pool().size(), std::max(1, n / 4096));
  if (chunks <= 1) {
    fn(x, n);
    return;
  }
  default_pool().parallel_for(chunks, [&](int index) {
    const int a = static_cast<int>(static_cast<int64_t>(n) * index / chunks);
    const int b =
        static_cast<int>(static_cast<int64_t>(n) * (index + 1) / chunks);
    if (b > a) fn(x + a, b - a);
  });
}

size_t Model::scratch_bytes() const {
  const Scratch& s = scratch_;
  size_t total = 0;
  for (const auto* v : {&s.norm, &s.qkv, &s.attn, &s.ffn, &s.trans, &s.rev,
                        &s.bwd, &s.cat, &s.xz, &s.u2, &s.y, &s.delta, &s.xp,
                        &s.scores, &s.bscores, &s.mamba_ext, &s.mamba_decay,
                        &s.mamba_h})
    total += v->size() * sizeof(float);
  return total;
}

void Model::init_lane(Lane& lane) const {
  lane.attn.assign(static_cast<size_t>(cfg_.num_blocks), AttnState());
  lane.mamba.assign(static_cast<size_t>(cfg_.num_blocks), MambaState());
  const size_t conv = static_cast<size_t>(cfg_.bands()) * cfg_.inner() *
                      (cfg_.mamba_conv - 1);
  const size_t ssm = static_cast<size_t>(cfg_.bands()) * cfg_.mamba_state *
                     cfg_.inner();
  for (int i = 0; i < cfg_.num_blocks; ++i) {
    if (!blocks_[static_cast<size_t>(i)].transformer) {
      lane.mamba[static_cast<size_t>(i)].conv.assign(conv, 0.0f);
      lane.mamba[static_cast<size_t>(i)].ssm.assign(ssm, 0.0f);
    }
  }
  lane.started = true;
}

void Model::clone_lane(const Lane& src, Lane& dst) const {
  dst.attn.resize(src.attn.size());
  for (auto& state : dst.attn) state.release();
  dst.mamba = src.mamba;  // the recurrence carries over
  dst.started = true;
}

// --------------------------------------------------------------- encoder ----

void Model::encode(const std::complex<float>* spectra, int frames,
                   float* feat) const {
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int bins = cfg_.bins();
  const float* comp = reinterpret_cast<const float*>(spectra);
  const float* la =
      lookahead_embedding_->data +
      static_cast<size_t>(cfg_.lookahead_index) * width;

  Timer timer(profile_enabled() ? &profile().encode : nullptr);
  default_pool().parallel_for(bands, [&](int b) {
    const auto& edge = cfg_.edges[static_cast<size_t>(b)];
    const int w2 = 2 * (edge.second - edge.first);
    std::vector<float> x(static_cast<size_t>(frames) * w2);
    std::vector<float> n(x.size());
    for (int t = 0; t < frames; ++t)
      std::memcpy(x.data() + static_cast<size_t>(t) * w2,
                  comp + (static_cast<size_t>(t) * bins + edge.first) * 2,
                  sizeof(float) * w2);
    layer_norm(x.data(), n.data(), frames, w2,
               enc_norm_w_[static_cast<size_t>(b)]->data,
               enc_norm_b_[static_cast<size_t>(b)]->data);
    float* out = feat + static_cast<size_t>(b) * frames * width;
    linear(n.data(), frames, w2, *enc_w_[static_cast<size_t>(b)],
           enc_b_[static_cast<size_t>(b)]->data, out);
    const float* be = band_embedding_->data + static_cast<size_t>(b) * width;
    for (int t = 0; t < frames; ++t) {
      float* row = out + static_cast<size_t>(t) * width;
      for (int j = 0; j < width; ++j) row[j] += be[j] + la[j];
    }
  });
}

// ----------------------------------------------------------- transformer ----

void Model::time_transformer(const AttnWeights& w, float* feat, int frames,
                             AttnState& st, const Rope& rope,
                             const LaneParams& lp) const {
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int heads = cfg_.heads;
  const int hd = cfg_.head_dim();
  const int rows = bands * frames;
  Scratch& sc = scratch(rows);

  Timer timer(profile_enabled() ? &profile().time_attn : nullptr);
  const bool use_f16_kv = fp16_kv_ && (hd % 32 == 0) && (hd <= 128);
  int first = st.end;
  {
    Timer slot(profile_enabled() ? &profile().rope : nullptr);
    if (first + frames > lp.capacity_limit) {
      // Keep the last cache_frames keys, renumbered from zero: each key is
      // rotated back by the distance it moved.
      const int keep = std::min(first, lp.cache_frames);
      const int shift = first - keep;
      const int lane = st.capacity * hd;  // one head's slots, contiguous
      if (shift > 0) {
        const float* cos = rope.cos_at(shift);
        const float* sin = rope.sin_at(shift);
        for (int b = 0; b < bands; ++b) {
          if (use_f16_kv) {
            uint16_t* kb =
                st.k16.data() + static_cast<size_t>(b) * st.capacity * width;
            uint16_t* vb =
                st.v16.data() + static_cast<size_t>(b) * st.capacity * width;
            for (int h = 0; h < heads; ++h) {
              uint16_t* kh = kb + static_cast<size_t>(h) * lane;
              for (int s = 0; s < keep; ++s)
                v_rope_f16_inplace(kh + static_cast<size_t>(s + shift) * hd, hd,
                                   cos, sin, /*inverse=*/true);
              std::memmove(kh, kh + static_cast<size_t>(shift) * hd,
                           sizeof(uint16_t) * keep * hd);
              std::memmove(vb + static_cast<size_t>(h) * lane,
                           vb + static_cast<size_t>(h) * lane + shift * hd,
                           sizeof(uint16_t) * keep * hd);
            }
          } else {
            float* kb =
                st.k.data() + static_cast<size_t>(b) * st.capacity * width;
            float* vb =
                st.v.data() + static_cast<size_t>(b) * st.capacity * width;
            for (int h = 0; h < heads; ++h) {
              float* kh = kb + static_cast<size_t>(h) * lane;
              for (int s = 0; s < keep; ++s)
                rope_head(kh + static_cast<size_t>(s + shift) * hd, hd, cos,
                          sin, /*inverse=*/true);
              std::memmove(kh, kh + static_cast<size_t>(shift) * hd,
                           sizeof(float) * keep * hd);
              std::memmove(vb + static_cast<size_t>(h) * lane,
                           vb + static_cast<size_t>(h) * lane + shift * hd,
                           sizeof(float) * keep * hd);
            }
          }
        }
        st.end = keep;
        first = keep;
      }
    }
  }
  if (use_f16_kv)
    reserve_slots_f16(st, first + frames, lp.capacity_limit, bands, width, hd);
  else
    reserve_slots(st, first + frames, lp.capacity_limit, bands, width, hd);
  if (sc.scores.size() <
      static_cast<size_t>(bands) * 4 * static_cast<size_t>(lp.capacity_limit))
    sc.scores.resize(static_cast<size_t>(bands) * 4 * lp.capacity_limit);

  layer_norm_par(feat, sc.norm.data(), rows, width, w.n1w->data, w.n1b->data);
  linear_par(sc.norm.data(), rows, width, *w.qkv_w, w.qkv_b->data,
             sc.qkv.data());

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const bool pf_single = profile_enabled() && default_pool().size() == 1;
  const int past_lo = std::max(0, first - lp.cache_frames);
  const size_t lane_stride = static_cast<size_t>(st.capacity) * hd;
  const size_t score_stride = static_cast<size_t>(lp.capacity_limit);

  default_pool().parallel_for(bands, [&](int b) {
    float* kbase = use_f16_kv ? nullptr : st.k.data() + static_cast<size_t>(b) * st.capacity * width;
    float* vbase = use_f16_kv ? nullptr : st.v.data() + static_cast<size_t>(b) * st.capacity * width;
    uint16_t* k16base = use_f16_kv ? st.k16.data() + static_cast<size_t>(b) * st.capacity * width : nullptr;
    uint16_t* v16base = use_f16_kv ? st.v16.data() + static_cast<size_t>(b) * st.capacity * width : nullptr;
    float* scores_base = sc.scores.data() + static_cast<size_t>(b) * 4 * score_stride;

    for (int t = 0; t < frames; ++t) {
      const int pos = first + t;
      float* row =
          sc.qkv.data() + static_cast<size_t>(b * frames + t) * 3 * width;
      const float* cos = rope.cos_at(pos);
      const float* sin = rope.sin_at(pos);
      for (int h = 0; h < heads; ++h) {
        float* q = row + h * hd;
        float* k = row + width + h * hd;
        const float* v = row + 2 * width + h * hd;
        rope_head(q, hd, cos, sin, false);
        rope_head(k, hd, cos, sin, false);
        const size_t slot_off =
            static_cast<size_t>(h) * lane_stride + static_cast<size_t>(pos) * hd;
        if (use_f16_kv) {
          v_f32_to_f16(k, k16base + slot_off, hd);
          v_f32_to_f16(v, v16base + slot_off, hd);
        } else {
          std::memcpy(kbase + slot_off, k, sizeof(float) * hd);
          std::memcpy(vbase + slot_off, v, sizeof(float) * hd);
        }
      }
    }

    int t0 = 0;
    for (; t0 + 4 <= frames; t0 += 4) {
      int pos[4], lo[4];
      for (int r = 0; r < 4; ++r) {
        pos[r] = first + t0 + r;
        lo[r] = past_lo;
        if (lp.window > 0) lo[r] = std::max(lo[r], pos[r] - lp.window + 1);
      }
      const int span_lo = lo[0];
      const int n_span = pos[3] - span_lo + 1;
      const size_t first_slot = static_cast<size_t>(span_lo) * hd;

      const float* r0 = sc.qkv.data() + static_cast<size_t>(b * frames + t0 + 0) * 3 * width;
      const float* r1 = sc.qkv.data() + static_cast<size_t>(b * frames + t0 + 1) * 3 * width;
      const float* r2 = sc.qkv.data() + static_cast<size_t>(b * frames + t0 + 2) * 3 * width;
      const float* r3 = sc.qkv.data() + static_cast<size_t>(b * frames + t0 + 3) * 3 * width;
      float* out0 = sc.attn.data() + static_cast<size_t>(b * frames + t0 + 0) * width;
      float* out1 = sc.attn.data() + static_cast<size_t>(b * frames + t0 + 1) * width;
      float* out2 = sc.attn.data() + static_cast<size_t>(b * frames + t0 + 2) * width;
      float* out3 = sc.attn.data() + static_cast<size_t>(b * frames + t0 + 3) * width;
      float* s0 = scores_base + 0 * score_stride;
      float* s1 = scores_base + 1 * score_stride;
      float* s2 = scores_base + 2 * score_stride;
      float* s3 = scores_base + 3 * score_stride;
      float* sptrs[4] = {s0, s1, s2, s3};

      for (int h = 0; h < heads; ++h) {
        const size_t head_slot = static_cast<size_t>(h) * lane_stride + first_slot;
        {
          Timer slot(pf_single ? &profile().attn_scores : nullptr);
          if (use_f16_kv)
            v_attn4_qk_f16(r0 + h * hd, r1 + h * hd, r2 + h * hd, r3 + h * hd,
                           k16base + head_slot, n_span, hd, scale, s0, s1, s2, s3);
          else
            v_attn4_qk(r0 + h * hd, r1 + h * hd, r2 + h * hd, r3 + h * hd,
                       kbase + head_slot, n_span, hd, scale, s0, s1, s2, s3);
          for (int r = 0; r < 4; ++r) {
            const int st_r = lo[r] - span_lo;
            const int en_r = pos[r] - span_lo + 1;
            v_softmax_norm(sptrs[r] + st_r, en_r - st_r);
            for (int z = 0; z < st_r; ++z) sptrs[r][z] = 0.0f;
            for (int z = en_r; z < n_span; ++z) sptrs[r][z] = 0.0f;
          }
        }
        float* o0 = out0 + h * hd;
        float* o1 = out1 + h * hd;
        float* o2 = out2 + h * hd;
        float* o3 = out3 + h * hd;
        {
          Timer slot(pf_single ? &profile().attn_pv : nullptr);
          if (use_f16_kv)
            v_attn4_pv_f16(s0, s1, s2, s3, v16base + head_slot, n_span, hd,
                           o0, o1, o2, o3);
          else
            v_attn4_pv(s0, s1, s2, s3, vbase + head_slot, n_span, hd,
                       o0, o1, o2, o3);
        }
        float* optrs[4] = {o0, o1, o2, o3};
        const float* vptrs[4] = {r0 + 2 * width + h * hd, r1 + 2 * width + h * hd,
                                 r2 + 2 * width + h * hd, r3 + 2 * width + h * hd};
        for (int r = 0; r < 4; ++r) {
          const float dot_v = dot(optrs[r], vptrs[r], hd);
          const float norm_v = dot(vptrs[r], vptrs[r], hd);
          const float coef = dot_v / std::max(norm_v, 1e-6f);
          axpy(optrs[r], -coef, vptrs[r], hd);
        }
      }
    }

    for (int t = t0; t < frames; ++t) {
      const int pos = first + t;
      const float* row =
          sc.qkv.data() + static_cast<size_t>(b * frames + t) * 3 * width;
      float* out =
          sc.attn.data() + static_cast<size_t>(b * frames + t) * width;
      int lo = past_lo;
      if (lp.window > 0) lo = std::max(lo, pos - lp.window + 1);
      const int n = pos - lo + 1;
      const size_t first_slot = static_cast<size_t>(lo) * hd;

      for (int h = 0; h < heads; ++h) {
        const float* q = row + h * hd;
        const size_t head_slot = static_cast<size_t>(h) * lane_stride + first_slot;
        float* s = scores_base;
        float sum;
        {
          Timer slot(pf_single ? &profile().attn_scores : nullptr);
          if (use_f16_kv)
            sum = v_attn1_scores_softmax_f16(q, k16base + head_slot, n, hd, scale, s);
          else
            sum = scores_softmax(q, kbase + head_slot, n, hd, scale, s);
        }
        const float inv = 1.0f / sum;
        float* o = out + h * hd;
        {
          Timer slot(pf_single ? &profile().attn_pv : nullptr);
          if (use_f16_kv)
            v_attn1_pv_f16(s, inv, v16base + head_slot, n, hd, o);
          else
            pv_row(s, inv, vbase + head_slot, n, hd, o);
        }
        const float* v = row + 2 * width + h * hd;
        const float dot_v = dot(o, v, hd);
        const float norm_v = dot(v, v, hd);
        const float coef = dot_v / std::max(norm_v, 1e-6f);
        axpy(o, -coef, v, hd);
      }
    }
  });
  st.end = first + frames;

  linear_par(sc.attn.data(), rows, width, *w.proj_w, w.proj_b->data,
             sc.trans.data());
  add_par(feat, sc.trans.data(), rows * width);

  layer_norm_par(feat, sc.norm.data(), rows, width, w.n2w->data, w.n2b->data);
  linear_par(sc.norm.data(), rows, width, *w.f0w, w.f0b->data,
             sc.ffn.data());
  elementwise_par(sc.ffn.data(), rows * 4 * width, v_gelu, /*cost=*/20);
  linear_par(sc.ffn.data(), rows, 4 * width, *w.f1w, w.f1b->data,
             sc.trans.data());
  add_par(feat, sc.trans.data(), rows * width);
}

void Model::time_transformer_pair(const AttnWeights& w, float* pair_feat,
                                  int frames, AttnState& st_a, AttnState& st_b,
                                  const Rope& rope,
                                  const LaneParams& lp) const {
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int heads = cfg_.heads;
  const int hd = cfg_.head_dim();
  const int rows = bands * frames;
  const int rows2 = 2 * rows;
  Scratch& sc = scratch(rows2);

  Timer timer(profile_enabled() ? &profile().time_attn : nullptr);
  const bool use_f16_kv = fp16_kv_ && (hd % 32 == 0) && (hd <= 128);
  auto prep_state = [&](AttnState& st) -> int {
    int first = st.end;
    if (first + frames > lp.capacity_limit) {
      const int keep = std::min(first, lp.cache_frames);
      const int shift = first - keep;
      const int lane = st.capacity * hd;
      if (shift > 0) {
        const float* cos = rope.cos_at(shift);
        const float* sin = rope.sin_at(shift);
        for (int b = 0; b < bands; ++b) {
          if (use_f16_kv) {
            uint16_t* kb =
                st.k16.data() + static_cast<size_t>(b) * st.capacity * width;
            uint16_t* vb =
                st.v16.data() + static_cast<size_t>(b) * st.capacity * width;
            for (int h = 0; h < heads; ++h) {
              uint16_t* kh = kb + static_cast<size_t>(h) * lane;
              for (int s = 0; s < keep; ++s)
                v_rope_f16_inplace(kh + static_cast<size_t>(s + shift) * hd, hd,
                                   cos, sin, true);
              std::memmove(kh, kh + static_cast<size_t>(shift) * hd,
                           sizeof(uint16_t) * keep * hd);
              std::memmove(vb + static_cast<size_t>(h) * lane,
                           vb + static_cast<size_t>(h) * lane + shift * hd,
                           sizeof(uint16_t) * keep * hd);
            }
          } else {
            float* kb =
                st.k.data() + static_cast<size_t>(b) * st.capacity * width;
            float* vb =
                st.v.data() + static_cast<size_t>(b) * st.capacity * width;
            for (int h = 0; h < heads; ++h) {
              float* kh = kb + static_cast<size_t>(h) * lane;
              for (int s = 0; s < keep; ++s)
                rope_head(kh + static_cast<size_t>(s + shift) * hd, hd, cos,
                          sin, true);
              std::memmove(kh, kh + static_cast<size_t>(shift) * hd,
                           sizeof(float) * keep * hd);
              std::memmove(vb + static_cast<size_t>(h) * lane,
                           vb + static_cast<size_t>(h) * lane + shift * hd,
                           sizeof(float) * keep * hd);
            }
          }
        }
        st.end = keep;
        first = keep;
      }
    }
    if (use_f16_kv)
      reserve_slots_f16(st, first + frames, lp.capacity_limit, bands, width, hd);
    else
      reserve_slots(st, first + frames, lp.capacity_limit, bands, width, hd);
    return first;
  };

  int firsts[2];
  {
    Timer slot(profile_enabled() ? &profile().rope : nullptr);
    firsts[0] = prep_state(st_a);
    firsts[1] = prep_state(st_b);
  }
  const size_t score_stride = static_cast<size_t>(lp.capacity_limit);
  if (sc.scores.size() < static_cast<size_t>(2 * bands) * 4 * score_stride)
    sc.scores.resize(static_cast<size_t>(2 * bands) * 4 * score_stride);

  layer_norm_par(pair_feat, sc.norm.data(), rows2, width, w.n1w->data,
                 w.n1b->data);
  linear_par(sc.norm.data(), rows2, width, *w.qkv_w, w.qkv_b->data,
             sc.qkv.data());

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  AttnState* states[2] = {&st_a, &st_b};

  default_pool().parallel_for(2 * bands, [&](int idx) {
    const int lane_idx = idx / bands;
    const int b = idx % bands;
    AttnState& st = *states[lane_idx];
    const int first = firsts[lane_idx];
    const int past_lo = std::max(0, first - lp.cache_frames);
    const size_t lane_stride = static_cast<size_t>(st.capacity) * hd;
    float* kbase = use_f16_kv ? nullptr : st.k.data() + static_cast<size_t>(b) * st.capacity * width;
    float* vbase = use_f16_kv ? nullptr : st.v.data() + static_cast<size_t>(b) * st.capacity * width;
    uint16_t* k16base = use_f16_kv ? st.k16.data() + static_cast<size_t>(b) * st.capacity * width : nullptr;
    uint16_t* v16base = use_f16_kv ? st.v16.data() + static_cast<size_t>(b) * st.capacity * width : nullptr;
    float* scores_base =
        sc.scores.data() + static_cast<size_t>(idx) * 4 * score_stride;
    const size_t row_off = static_cast<size_t>(lane_idx) * rows;

    for (int t = 0; t < frames; ++t) {
      const int pos = first + t;
      float* row = sc.qkv.data() +
                   (row_off + static_cast<size_t>(b * frames + t)) * 3 * width;
      const float* cos = rope.cos_at(pos);
      const float* sin = rope.sin_at(pos);
      for (int h = 0; h < heads; ++h) {
        float* q = row + h * hd;
        float* k = row + width + h * hd;
        const float* v = row + 2 * width + h * hd;
        rope_head(q, hd, cos, sin, false);
        rope_head(k, hd, cos, sin, false);
        const size_t slot_off =
            static_cast<size_t>(h) * lane_stride + static_cast<size_t>(pos) * hd;
        if (use_f16_kv) {
          v_f32_to_f16(k, k16base + slot_off, hd);
          v_f32_to_f16(v, v16base + slot_off, hd);
        } else {
          std::memcpy(kbase + slot_off, k, sizeof(float) * hd);
          std::memcpy(vbase + slot_off, v, sizeof(float) * hd);
        }
      }
    }

    int t0 = 0;
    for (; t0 + 4 <= frames; t0 += 4) {
      int pos[4], lo[4];
      for (int r = 0; r < 4; ++r) {
        pos[r] = first + t0 + r;
        lo[r] = past_lo;
        if (lp.window > 0) lo[r] = std::max(lo[r], pos[r] - lp.window + 1);
      }
      const int span_lo = lo[0];
      const int n_span = pos[3] - span_lo + 1;
      const size_t first_slot = static_cast<size_t>(span_lo) * hd;

      const float* r0 = sc.qkv.data() + (row_off + static_cast<size_t>(b * frames + t0 + 0)) * 3 * width;
      const float* r1 = sc.qkv.data() + (row_off + static_cast<size_t>(b * frames + t0 + 1)) * 3 * width;
      const float* r2 = sc.qkv.data() + (row_off + static_cast<size_t>(b * frames + t0 + 2)) * 3 * width;
      const float* r3 = sc.qkv.data() + (row_off + static_cast<size_t>(b * frames + t0 + 3)) * 3 * width;
      float* out0 = sc.attn.data() + (row_off + static_cast<size_t>(b * frames + t0 + 0)) * width;
      float* out1 = sc.attn.data() + (row_off + static_cast<size_t>(b * frames + t0 + 1)) * width;
      float* out2 = sc.attn.data() + (row_off + static_cast<size_t>(b * frames + t0 + 2)) * width;
      float* out3 = sc.attn.data() + (row_off + static_cast<size_t>(b * frames + t0 + 3)) * width;
      float* s0 = scores_base + 0 * score_stride;
      float* s1 = scores_base + 1 * score_stride;
      float* s2 = scores_base + 2 * score_stride;
      float* s3 = scores_base + 3 * score_stride;
      float* sptrs[4] = {s0, s1, s2, s3};

      for (int h = 0; h < heads; ++h) {
        const size_t head_slot = static_cast<size_t>(h) * lane_stride + first_slot;
        if (use_f16_kv)
          v_attn4_qk_f16(r0 + h * hd, r1 + h * hd, r2 + h * hd, r3 + h * hd,
                         k16base + head_slot, n_span, hd, scale, s0, s1, s2, s3);
        else
          v_attn4_qk(r0 + h * hd, r1 + h * hd, r2 + h * hd, r3 + h * hd,
                     kbase + head_slot, n_span, hd, scale, s0, s1, s2, s3);
        for (int r = 0; r < 4; ++r) {
          const int st_r = lo[r] - span_lo;
          const int en_r = pos[r] - span_lo + 1;
          v_softmax_norm(sptrs[r] + st_r, en_r - st_r);
          for (int z = 0; z < st_r; ++z) sptrs[r][z] = 0.0f;
          for (int z = en_r; z < n_span; ++z) sptrs[r][z] = 0.0f;
        }
        float* o0 = out0 + h * hd;
        float* o1 = out1 + h * hd;
        float* o2 = out2 + h * hd;
        float* o3 = out3 + h * hd;
        if (use_f16_kv)
          v_attn4_pv_f16(s0, s1, s2, s3, v16base + head_slot, n_span, hd,
                         o0, o1, o2, o3);
        else
          v_attn4_pv(s0, s1, s2, s3, vbase + head_slot, n_span, hd,
                     o0, o1, o2, o3);
        float* optrs[4] = {o0, o1, o2, o3};
        const float* vptrs[4] = {r0 + 2 * width + h * hd, r1 + 2 * width + h * hd,
                                 r2 + 2 * width + h * hd, r3 + 2 * width + h * hd};
        for (int r = 0; r < 4; ++r) {
          const float dot_v = dot(optrs[r], vptrs[r], hd);
          const float norm_v = dot(vptrs[r], vptrs[r], hd);
          const float coef = dot_v / std::max(norm_v, 1e-6f);
          axpy(optrs[r], -coef, vptrs[r], hd);
        }
      }
    }
    for (int t = t0; t < frames; ++t) {
      const int pos = first + t;
      const float* row = sc.qkv.data() +
                         (row_off + static_cast<size_t>(b * frames + t)) * 3 * width;
      float* out = sc.attn.data() +
                   (row_off + static_cast<size_t>(b * frames + t)) * width;
      int lo = past_lo;
      if (lp.window > 0) lo = std::max(lo, pos - lp.window + 1);
      const int n = pos - lo + 1;
      const size_t first_slot = static_cast<size_t>(lo) * hd;
      for (int h = 0; h < heads; ++h) {
        const float* q = row + h * hd;
        const size_t head_slot = static_cast<size_t>(h) * lane_stride + first_slot;
        float* s = scores_base;
        const float sum = use_f16_kv
                              ? v_attn1_scores_softmax_f16(q, k16base + head_slot, n, hd, scale, s)
                              : scores_softmax(q, kbase + head_slot, n, hd, scale, s);
        const float inv = 1.0f / sum;
        float* o = out + h * hd;
        if (use_f16_kv)
          v_attn1_pv_f16(s, inv, v16base + head_slot, n, hd, o);
        else
          pv_row(s, inv, vbase + head_slot, n, hd, o);
        const float* v = row + 2 * width + h * hd;
        const float dot_v = dot(o, v, hd);
        const float norm_v = dot(v, v, hd);
        const float coef = dot_v / std::max(norm_v, 1e-6f);
        axpy(o, -coef, v, hd);
      }
    }
  });
  st_a.end = firsts[0] + frames;
  st_b.end = firsts[1] + frames;

  linear_par(sc.attn.data(), rows2, width, *w.proj_w, w.proj_b->data,
             sc.trans.data());
  add_par(pair_feat, sc.trans.data(), rows2 * width);

  layer_norm_par(pair_feat, sc.norm.data(), rows2, width, w.n2w->data,
                 w.n2b->data);
  linear_par(sc.norm.data(), rows2, width, *w.f0w, w.f0b->data, sc.ffn.data());
  elementwise_par(sc.ffn.data(), rows2 * 4 * width, v_gelu, /*cost=*/20);
  linear_par(sc.ffn.data(), rows2, 4 * width, *w.f1w, w.f1b->data,
             sc.trans.data());
  add_par(pair_feat, sc.trans.data(), rows2 * width);
}

void Model::band_transformer_core(const AttnWeights& w, float* trans,
                                  int frames) const {
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int heads = cfg_.heads;
  const int hd = cfg_.head_dim();
  const int rows = bands * frames;
  const int units = frames * heads;
  Scratch& sc = scratch(rows);
  if (sc.bscores.size() < static_cast<size_t>(bands) * bands * units)
    sc.bscores.resize(static_cast<size_t>(bands) * bands * units);

  layer_norm_par(trans, sc.norm.data(), rows, width, w.bn1w->data,
                 w.bn1b->data);
  linear_par(sc.norm.data(), rows, width, *w.in_w, w.in_b->data, sc.qkv.data());

  const size_t per_unit = static_cast<size_t>(bands) * hd;
  {
    Timer slot(profile_enabled() ? &profile().transpose : nullptr);
    if (sc.bqkv.size() < 3 * per_unit * units) sc.bqkv.resize(3 * per_unit * units);
    float* qall = sc.bqkv.data();
    float* kall = qall + per_unit * units;
    float* vall = kall + per_unit * units;
    const int w3 = 3 * width;
    for (int f = 0; f < frames; ++f) {
      const float* base = sc.qkv.data() + static_cast<size_t>(f) * bands * w3;
      for (int h = 0; h < heads; ++h) {
        float* q = qall + (static_cast<size_t>(f) * heads + h) * per_unit;
        float* k = kall + (static_cast<size_t>(f) * heads + h) * per_unit;
        float* v = vall + (static_cast<size_t>(f) * heads + h) * per_unit;
        for (int b = 0; b < bands; ++b) {
          const float* src = base + static_cast<size_t>(b) * w3 + h * hd;
          std::memcpy(q + static_cast<size_t>(b) * hd, src, sizeof(float) * hd);
          std::memcpy(k + static_cast<size_t>(b) * hd, src + width,
                      sizeof(float) * hd);
          std::memcpy(v + static_cast<size_t>(b) * hd, src + 2 * width,
                      sizeof(float) * hd);
        }
      }
    }
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const float* qall = sc.bqkv.data();
  const float* kall = qall + per_unit * units;
  const float* vall = kall + per_unit * units;
  default_pool().parallel_for(units, [&](int unit) {
    const float* q = qall + static_cast<size_t>(unit) * per_unit;
    const float* k = kall + static_cast<size_t>(unit) * per_unit;
    const float* v = vall + static_cast<size_t>(unit) * per_unit;
    const int f = unit / heads;
    const int h = unit % heads;
    float* scores = sc.bscores.data() + static_cast<size_t>(unit) * bands * bands;
    float* out = sc.attn.data() + static_cast<size_t>(f) * bands * width + h * hd;
    int i = 0;
    for (; i + 4 <= bands; i += 4) {
      float* s0 = scores + static_cast<size_t>(i + 0) * bands;
      float* s1 = scores + static_cast<size_t>(i + 1) * bands;
      float* s2 = scores + static_cast<size_t>(i + 2) * bands;
      float* s3 = scores + static_cast<size_t>(i + 3) * bands;
      v_attn4_qk(q + static_cast<size_t>(i + 0) * hd,
                 q + static_cast<size_t>(i + 1) * hd,
                 q + static_cast<size_t>(i + 2) * hd,
                 q + static_cast<size_t>(i + 3) * hd, k, bands, hd, scale,
                 s0, s1, s2, s3);
      v_softmax_norm(s0, bands);
      v_softmax_norm(s1, bands);
      v_softmax_norm(s2, bands);
      v_softmax_norm(s3, bands);
      v_attn4_pv(s0, s1, s2, s3, v, bands, hd,
                 out + static_cast<size_t>(i + 0) * width,
                 out + static_cast<size_t>(i + 1) * width,
                 out + static_cast<size_t>(i + 2) * width,
                 out + static_cast<size_t>(i + 3) * width);
    }
    for (; i < bands; ++i) {
      float* s = scores + static_cast<size_t>(i) * bands;
      const float sum = scores_softmax(q + static_cast<size_t>(i) * hd, k,
                                       bands, hd, scale, s);
      pv_row(s, 1.0f / sum, v, bands, hd, out + static_cast<size_t>(i) * width);
    }
  });
  linear_par(sc.attn.data(), rows, width, *w.out_w, w.out_b->data,
             sc.norm.data());
  add_par(trans, sc.norm.data(), rows * width);

  layer_norm_par(trans, sc.norm.data(), rows, width, w.bn2w->data,
                 w.bn2b->data);
  linear_par(sc.norm.data(), rows, width, *w.bl1w, w.bl1b->data, sc.ffn.data());
  elementwise_par(sc.ffn.data(), rows * 4 * width, v_gelu, /*cost=*/20);
  linear_par(sc.ffn.data(), rows, 4 * width, *w.bl2w, w.bl2b->data,
             sc.norm.data());
  add_par(trans, sc.norm.data(), rows * width);
}

void Model::band_transformer(const AttnWeights& w, float* feat,
                             int frames) const {
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int rows = bands * frames;
  Scratch& sc = scratch(rows);
  Timer timer(profile_enabled() ? &profile().band_attn : nullptr);
  {
    Timer slot(profile_enabled() ? &profile().transpose : nullptr);
    to_frame_major(feat, sc.trans.data(), bands, frames, width);
  }
  band_transformer_core(w, sc.trans.data(), frames);
  {
    Timer slot(profile_enabled() ? &profile().transpose : nullptr);
    to_band_major(sc.trans.data(), feat, bands, frames, width);
  }
}

// ----------------------------------------------------------------- mamba ----

void Model::mamba_layer(const MambaLayer& l, const float* x, int rows, int len,
                        float* out, float* conv_state, float* ssm_state,
                        float* slab_ext, float* slab_decay,
                        float* slab_h) const {
  const int width = cfg_.width;
  const int inner = cfg_.inner();
  const int state = cfg_.mamba_state;
  const int conv = cfg_.mamba_conv;
  const int taps = conv - 1;
  const int dt_rank = cfg_.dt_rank();
  const int dt_total = dt_rank + 2 * state;
  const int tasks = rows / len;
  Scratch& sc = scratch(rows);
  Timer timer(profile_enabled() ? &profile().mamba : nullptr);

  linear_par(x, rows, width, *l.in_w, nullptr, sc.xz.data());

  const size_t ext_stride = static_cast<size_t>(len + taps) * inner;
  const size_t state_stride = static_cast<size_t>(state) * inner;

  // Channels are independent through the convolution and the scan, so each
  // task is split into 4-channel-aligned ranges.
  const int split = channel_split(tasks, inner);
  const int c_groups = inner / 4;
  {
    Timer slot(profile_enabled() ? &profile().mamba_conv : nullptr);
    default_pool().parallel_for(tasks * split, [&](int unit) {
      const int task = unit / split;
      const int part = unit % split;
      const int c0 =
          4 * static_cast<int>(static_cast<int64_t>(c_groups) * part / split);
      const int c1 = 4 * static_cast<int>(static_cast<int64_t>(c_groups) *
                                          (part + 1) / split);
      if (c1 <= c0) return;
      const int channels = c1 - c0;
      float* ext = slab_ext + static_cast<size_t>(task) * ext_stride;
      if (conv_state != nullptr) {
        // conv_state is laid out [task][taps][inner] for contiguous copies.
        const float* carried =
            conv_state + static_cast<size_t>(task) * taps * inner;
        for (int j = 0; j < taps; ++j)
          std::memcpy(ext + static_cast<size_t>(j) * inner + c0,
                      carried + static_cast<size_t>(j) * inner + c0,
                      sizeof(float) * static_cast<size_t>(channels));
      } else {
        for (int j = 0; j < taps; ++j)
          std::memset(ext + static_cast<size_t>(j) * inner + c0, 0,
                      sizeof(float) * static_cast<size_t>(channels));
      }
      for (int t = 0; t < len; ++t)
        std::memcpy(
            ext + static_cast<size_t>(t + taps) * inner + c0,
            sc.xz.data() + static_cast<size_t>(task * len + t) * 2 * inner + c0,
            sizeof(float) * static_cast<size_t>(channels));
      // Fused 4-tap depthwise conv1d + bias + SiLU in registers.
      if (conv == 4) {
        const float* bptr = l.conv_b->data + c0;
        const float* w0 = l.conv_t.data() + 0 * inner + c0;
        const float* w1 = l.conv_t.data() + 1 * inner + c0;
        const float* w2 = l.conv_t.data() + 2 * inner + c0;
        const float* w3 = l.conv_t.data() + 3 * inner + c0;
        for (int t = 0; t < len; ++t) {
          float* dst =
              sc.u2.data() + static_cast<size_t>(task * len + t) * inner + c0;
          v_conv4_silu(dst, bptr, w0, w1, w2, w3,
                       ext + static_cast<size_t>(t + 0) * inner + c0,
                       ext + static_cast<size_t>(t + 1) * inner + c0,
                       ext + static_cast<size_t>(t + 2) * inner + c0,
                       ext + static_cast<size_t>(t + 3) * inner + c0, channels);
        }
      } else {
        const int vend = c1 - (channels % vx::lanes);
        for (int t = 0; t < len; ++t) {
          float* dst =
              sc.u2.data() + static_cast<size_t>(task * len + t) * inner + c0;
          std::memcpy(dst, l.conv_b->data + c0,
                      sizeof(float) * static_cast<size_t>(channels));
          for (int j = 0; j < conv; ++j) {
            const float* tap =
                l.conv_t.data() + static_cast<size_t>(j) * inner + c0;
            const float* src = ext + static_cast<size_t>(t + j) * inner + c0;
            int c = c0;
            for (; c < vend; c += vx::lanes) {
              const int o = c - c0;
              storef(dst + o, loadf(dst + o) + loadf(tap + o) * loadf(src + o));
            }
            for (; c < c1; ++c) dst[c - c0] += tap[c - c0] * src[c - c0];
          }
          v_silu(dst, channels);
        }
      }
      if (conv_state != nullptr) {
        float* carried = conv_state + static_cast<size_t>(task) * taps * inner;
        for (int j = 0; j < taps; ++j)
          std::memcpy(carried + static_cast<size_t>(j) * inner + c0,
                      ext + static_cast<size_t>(len + j) * inner + c0,
                      sizeof(float) * static_cast<size_t>(channels));
      }
    });
  }

  linear_par(sc.u2.data(), rows, inner, *l.x_w, nullptr, sc.xp.data());
  linear_par(sc.xp.data(), rows, dt_rank, dt_total, *l.dt_w, l.dt_b->data,
             sc.delta.data());
  elementwise_par(sc.delta.data(), rows * inner, v_softplus, /*cost=*/10);

  {
    Timer slot(profile_enabled() ? &profile().mamba_scan : nullptr);
    // Use 8-wide fp16 scan only for stateless freq_mamba (len=46, reset every
    // frame); time_mamba carries state across thousands of frames where fp16
    // rounding of decay near 1.0 causes cumulative drift.
    const bool use_f16_scan =
        quant_mode_ != QuantMode::Fp32 && ssm_state == nullptr &&
        !l.a_log2e8_16.empty() && (inner % 8 == 0);
    const int align_c = use_f16_scan ? 8 : 4;
    const int scan_groups = inner / align_c;
    default_pool().parallel_for(tasks * split, [&](int unit) {
      const int task = unit / split;
      const int part = unit % split;
      const int c0 = align_c * static_cast<int>(
                                   static_cast<int64_t>(scan_groups) * part /
                                   split);
      const int c1 = align_c * static_cast<int>(
                                   static_cast<int64_t>(scan_groups) *
                                   (part + 1) / split);
      if (c1 <= c0) return;
      float* h = ssm_state != nullptr
                     ? ssm_state + static_cast<size_t>(task) * state_stride
                     : nullptr;
      const size_t row0 = static_cast<size_t>(task) * len;
      if (use_f16_scan)
        v_mamba_scan_f16(h, l.a_log2e8_16.data(), l.d->data,
                         sc.u2.data() + row0 * inner,
                         sc.delta.data() + row0 * inner,
                         sc.xz.data() + row0 * 2 * inner,
                         sc.xp.data() + row0 * dt_total,
                         sc.y.data() + row0 * inner,
                         len, inner, state, dt_rank, dt_total, c0, c1);
      else
        v_mamba_scan(h, l.a_log2e4.data(), l.d->data,
                     sc.u2.data() + row0 * inner,
                     sc.delta.data() + row0 * inner,
                     sc.xz.data() + row0 * 2 * inner,
                     sc.xp.data() + row0 * dt_total,
                     sc.y.data() + row0 * inner,
                     len, inner, state, dt_rank, dt_total, c0, c1);
    });
  }

  linear_par(sc.y.data(), rows, inner, *l.out_w, nullptr, out);
}

void Model::mamba_layer_pair(const MambaLayer& l, const float* x, int rows_each,
                             int len, float* out, float* conv_a, float* ssm_a,
                             float* conv_b, float* ssm_b,
                             float* slab_ext) const {
  const int width = cfg_.width;
  const int inner = cfg_.inner();
  const int state = cfg_.mamba_state;
  const int conv = cfg_.mamba_conv;
  const int taps = conv - 1;
  const int dt_rank = cfg_.dt_rank();
  const int dt_total = dt_rank + 2 * state;
  const int tasks_each = rows_each / len;
  const int tasks = 2 * tasks_each;
  const int rows = 2 * rows_each;
  Scratch& sc = scratch(rows);
  Timer timer(profile_enabled() ? &profile().mamba : nullptr);

  linear_par(x, rows, width, *l.in_w, nullptr, sc.xz.data());

  const size_t ext_stride = static_cast<size_t>(len + taps) * inner;
  const size_t state_stride = static_cast<size_t>(state) * inner;
  const int split = channel_split(tasks, inner);
  const int c_groups = inner / 4;
  float* conv_states[2] = {conv_a, conv_b};
  float* ssm_states[2] = {ssm_a, ssm_b};

  {
    Timer slot(profile_enabled() ? &profile().mamba_conv : nullptr);
    default_pool().parallel_for(tasks * split, [&](int unit) {
      const int task = unit / split;
      const int part = unit % split;
      const int lane_idx = task / tasks_each;
      const int local_task = task % tasks_each;
      const int c0 =
          4 * static_cast<int>(static_cast<int64_t>(c_groups) * part / split);
      const int c1 = 4 * static_cast<int>(static_cast<int64_t>(c_groups) *
                                          (part + 1) / split);
      if (c1 <= c0) return;
      const int channels = c1 - c0;
      float* ext = slab_ext + static_cast<size_t>(task) * ext_stride;
      float* cstate = conv_states[lane_idx];
      if (cstate != nullptr) {
        const float* carried =
            cstate + static_cast<size_t>(local_task) * taps * inner;
        for (int j = 0; j < taps; ++j)
          std::memcpy(ext + static_cast<size_t>(j) * inner + c0,
                      carried + static_cast<size_t>(j) * inner + c0,
                      sizeof(float) * static_cast<size_t>(channels));
      } else {
        for (int j = 0; j < taps; ++j)
          std::memset(ext + static_cast<size_t>(j) * inner + c0, 0,
                      sizeof(float) * static_cast<size_t>(channels));
      }
      for (int t = 0; t < len; ++t)
        std::memcpy(
            ext + static_cast<size_t>(t + taps) * inner + c0,
            sc.xz.data() + static_cast<size_t>(task * len + t) * 2 * inner + c0,
            sizeof(float) * static_cast<size_t>(channels));
      if (conv == 4) {
        const float* bptr = l.conv_b->data + c0;
        const float* w0 = l.conv_t.data() + 0 * inner + c0;
        const float* w1 = l.conv_t.data() + 1 * inner + c0;
        const float* w2 = l.conv_t.data() + 2 * inner + c0;
        const float* w3 = l.conv_t.data() + 3 * inner + c0;
        for (int t = 0; t < len; ++t) {
          float* dst =
              sc.u2.data() + static_cast<size_t>(task * len + t) * inner + c0;
          v_conv4_silu(dst, bptr, w0, w1, w2, w3,
                       ext + static_cast<size_t>(t + 0) * inner + c0,
                       ext + static_cast<size_t>(t + 1) * inner + c0,
                       ext + static_cast<size_t>(t + 2) * inner + c0,
                       ext + static_cast<size_t>(t + 3) * inner + c0, channels);
        }
      }
      if (cstate != nullptr) {
        float* carried =
            cstate + static_cast<size_t>(local_task) * taps * inner;
        for (int j = 0; j < taps; ++j)
          std::memcpy(carried + static_cast<size_t>(j) * inner + c0,
                      ext + static_cast<size_t>(len + j) * inner + c0,
                      sizeof(float) * static_cast<size_t>(channels));
      }
    });
  }

  linear_par(sc.u2.data(), rows, inner, *l.x_w, nullptr, sc.xp.data());
  linear_par(sc.xp.data(), rows, dt_rank, dt_total, *l.dt_w, l.dt_b->data,
             sc.delta.data());
  elementwise_par(sc.delta.data(), rows * inner, v_softplus, /*cost=*/10);

  {
    Timer slot(profile_enabled() ? &profile().mamba_scan : nullptr);
    // Use 8-wide fp16 scan only for stateless freq_mamba (len=46, reset every
    // frame); time_mamba carries state across thousands of frames where fp16
    // rounding of decay near 1.0 causes cumulative drift.
    const bool use_f16_scan =
        quant_mode_ != QuantMode::Fp32 && ssm_a == nullptr &&
        ssm_b == nullptr && !l.a_log2e8_16.empty() && (inner % 8 == 0);
    const int align_c = use_f16_scan ? 8 : 4;
    const int scan_groups = inner / align_c;
    default_pool().parallel_for(tasks * split, [&](int unit) {
      const int task = unit / split;
      const int part = unit % split;
      const int lane_idx = task / tasks_each;
      const int local_task = task % tasks_each;
      const int c0 = align_c * static_cast<int>(
                                   static_cast<int64_t>(scan_groups) * part /
                                   split);
      const int c1 = align_c * static_cast<int>(
                                   static_cast<int64_t>(scan_groups) *
                                   (part + 1) / split);
      if (c1 <= c0) return;
      float* sstate = ssm_states[lane_idx];
      float* h = sstate != nullptr
                     ? sstate + static_cast<size_t>(local_task) * state_stride
                     : nullptr;
      const size_t row0 = static_cast<size_t>(task) * len;
      if (use_f16_scan)
        v_mamba_scan_f16(h, l.a_log2e8_16.data(), l.d->data,
                         sc.u2.data() + row0 * inner,
                         sc.delta.data() + row0 * inner,
                         sc.xz.data() + row0 * 2 * inner,
                         sc.xp.data() + row0 * dt_total,
                         sc.y.data() + row0 * inner,
                         len, inner, state, dt_rank, dt_total, c0, c1);
      else
        v_mamba_scan(h, l.a_log2e4.data(), l.d->data,
                     sc.u2.data() + row0 * inner,
                     sc.delta.data() + row0 * inner,
                     sc.xz.data() + row0 * 2 * inner,
                     sc.xp.data() + row0 * dt_total,
                     sc.y.data() + row0 * inner,
                     len, inner, state, dt_rank, dt_total, c0, c1);
    });
  }

  linear_par(sc.y.data(), rows, inner, *l.out_w, nullptr, out);
}

void Model::backbone(float* feat, int frames, Lane& lane, const Rope& rope,
                     const LaneParams& lp) const {
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int inner = cfg_.inner();
  const int state = cfg_.mamba_state;
  const int taps = cfg_.mamba_conv - 1;
  const int rows = bands * frames;
  Scratch& sc = scratch(rows);

  const size_t ext_need = std::max(
      static_cast<size_t>(bands) * (frames + taps) * inner,
      static_cast<size_t>(frames) * (bands + taps) * inner);
  const size_t state_need =
      static_cast<size_t>(std::max(bands, frames)) * state * inner;
  if (sc.mamba_ext.size() < ext_need) sc.mamba_ext.resize(ext_need);
  if (sc.mamba_decay.size() < state_need) sc.mamba_decay.resize(state_need);
  if (sc.mamba_h.size() < state_need) sc.mamba_h.resize(state_need);

  std::vector<float> debug_in;
  if (debug_block) debug_in.resize(static_cast<size_t>(rows) * width);

  for (int i = 0; i < cfg_.num_blocks; ++i) {
    const Block& block = blocks_[static_cast<size_t>(i)];
    if (debug_block)
      std::memcpy(debug_in.data(), feat,
                  sizeof(float) * static_cast<size_t>(rows) * width);
    if (block.transformer) {
      time_transformer(block.attn, feat, frames,
                       lane.attn[static_cast<size_t>(i)], rope, lp);
      band_transformer(block.attn, feat, frames);
    } else {
      MambaState& ms = lane.mamba[static_cast<size_t>(i)];
      // time: causal, carrying state, per band sequences of `frames`.
      // The block is norm(output_proj(mamba + x)) added to x.
      mamba_layer(block.time_mamba.forward, feat, rows, frames,
                  sc.trans.data(), ms.conv.data(), ms.ssm.data(),
                  sc.mamba_ext.data(), sc.mamba_decay.data(),
                  sc.mamba_h.data());
      for (size_t k = 0; k < static_cast<size_t>(rows) * width; ++k)
        sc.trans[k] += feat[k];
      linear_par(sc.trans.data(), rows, width, *block.time_mamba.out_w,
                 block.time_mamba.out_b->data, sc.rev.data());
      layer_norm_par(sc.rev.data(), sc.norm.data(), rows, width,
                     block.time_mamba.norm_w->data,
                     block.time_mamba.norm_b->data);
      add_par(feat, sc.norm.data(), rows * width);

      // frequency: two-sided over the bands of each frame
      to_frame_major(feat, sc.trans.data(), bands, frames, width);
      mamba_layer(block.freq_mamba.forward, sc.trans.data(), rows, bands,
                  sc.attn.data(), nullptr, nullptr, sc.mamba_ext.data(),
                  sc.mamba_decay.data(), sc.mamba_h.data());
      for (int f = 0; f < frames; ++f)
        for (int b = 0; b < bands; ++b)
          std::memcpy(
              sc.rev.data() +
                  static_cast<size_t>(f) * bands * width + b * width,
              sc.trans.data() + (static_cast<size_t>(f) * bands +
                                 (bands - 1 - b)) * width,
              sizeof(float) * width);
      mamba_layer(block.freq_mamba.backward, sc.rev.data(), rows, bands,
                  sc.bwd.data(), nullptr, nullptr, sc.mamba_ext.data(),
                  sc.mamba_decay.data(), sc.mamba_h.data());
      for (int f = 0; f < frames; ++f) {
        for (int b = 0; b < bands; ++b) {
          const size_t row = static_cast<size_t>(f) * bands + b;
          const size_t flipped = static_cast<size_t>(f) * bands + (bands - 1 - b);
          const float* fwd = sc.attn.data() + row * width;
          const float* bwd = sc.bwd.data() + flipped * width;
          const float* tr = sc.trans.data() + row * width;
          float* dst0 = sc.cat.data() + row * 2 * width;
          float* dst1 = dst0 + width;
          for (int col = 0; col < width; ++col) {
            dst0[col] = fwd[col] + tr[col];
            dst1[col] = bwd[col] + tr[col];
          }
        }
      }
      linear_par(sc.cat.data(), rows, 2 * width, *block.freq_mamba.out_w,
                 block.freq_mamba.out_b->data, sc.norm.data());
      layer_norm_par(sc.norm.data(), sc.attn.data(), rows, width,
                     block.freq_mamba.norm_w->data,
                     block.freq_mamba.norm_b->data);
      add_par(sc.trans.data(), sc.attn.data(), rows * width);
      to_band_major(sc.trans.data(), feat, bands, frames, width);
    }
    if (debug_block) debug_block(i, frames, debug_in.data(), feat);
  }
}

void Model::backbone_pair(float* feat_a, float* feat_b, int frames,
                          Lane& lane_a, Lane& lane_b, const Rope& rope,
                          const LaneParams& lp) const {
  if (debug_block) {
    backbone(feat_a, frames, lane_a, rope, lp);
    backbone(feat_b, frames, lane_b, rope, lp);
    return;
  }
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int inner = cfg_.inner();
  const int state = cfg_.mamba_state;
  const int taps = cfg_.mamba_conv - 1;
  const int rows = bands * frames;
  const int rows2 = 2 * rows;
  const int frames2 = 2 * frames;
  Scratch& sc = scratch(rows2);

  const size_t ext_need = std::max(
      static_cast<size_t>(2 * bands) * (frames + taps) * inner,
      static_cast<size_t>(frames2) * (bands + taps) * inner);
  const size_t state_need =
      static_cast<size_t>(std::max(2 * bands, frames2)) * state * inner;
  if (sc.mamba_ext.size() < ext_need) sc.mamba_ext.resize(ext_need);
  if (sc.mamba_decay.size() < state_need) sc.mamba_decay.resize(state_need);
  if (sc.mamba_h.size() < state_need) sc.mamba_h.resize(state_need);
  if (sc.pair.size() < static_cast<size_t>(rows2) * width)
    sc.pair.resize(static_cast<size_t>(rows2) * width);

  const size_t lane_floats = static_cast<size_t>(rows) * width;
  std::memcpy(sc.pair.data(), feat_a, sizeof(float) * lane_floats);
  std::memcpy(sc.pair.data() + lane_floats, feat_b, sizeof(float) * lane_floats);

  for (int i = 0; i < cfg_.num_blocks; ++i) {
    const Block& block = blocks_[static_cast<size_t>(i)];
    if (block.transformer) {
      time_transformer_pair(block.attn, sc.pair.data(), frames,
                            lane_a.attn[static_cast<size_t>(i)],
                            lane_b.attn[static_cast<size_t>(i)], rope, lp);
      {
        Timer timer(profile_enabled() ? &profile().band_attn : nullptr);
        {
          Timer slot(profile_enabled() ? &profile().transpose : nullptr);
          to_frame_major(sc.pair.data(), sc.trans.data(), bands, frames, width);
          to_frame_major(sc.pair.data() + lane_floats,
                         sc.trans.data() + lane_floats, bands, frames, width);
        }
        band_transformer_core(block.attn, sc.trans.data(), frames2);
        {
          Timer slot(profile_enabled() ? &profile().transpose : nullptr);
          to_band_major(sc.trans.data(), sc.pair.data(), bands, frames, width);
          to_band_major(sc.trans.data() + lane_floats,
                        sc.pair.data() + lane_floats, bands, frames, width);
        }
      }
    } else {
      MambaState& ms_a = lane_a.mamba[static_cast<size_t>(i)];
      MambaState& ms_b = lane_b.mamba[static_cast<size_t>(i)];
      mamba_layer_pair(block.time_mamba.forward, sc.pair.data(), rows, frames,
                       sc.trans.data(), ms_a.conv.data(), ms_a.ssm.data(),
                       ms_b.conv.data(), ms_b.ssm.data(), sc.mamba_ext.data());
      add_par(sc.trans.data(), sc.pair.data(), rows2 * width);
      linear_par(sc.trans.data(), rows2, width, *block.time_mamba.out_w,
                 block.time_mamba.out_b->data, sc.rev.data());
      layer_norm_par(sc.rev.data(), sc.norm.data(), rows2, width,
                     block.time_mamba.norm_w->data,
                     block.time_mamba.norm_b->data);
      add_par(sc.pair.data(), sc.norm.data(), rows2 * width);

      // Frequency Mamba is stateless across time, so both lanes batch into
      // 2 * frames independent sequences!
      to_frame_major(sc.pair.data(), sc.trans.data(), bands, frames, width);
      to_frame_major(sc.pair.data() + lane_floats,
                     sc.trans.data() + lane_floats, bands, frames, width);
      mamba_layer(block.freq_mamba.forward, sc.trans.data(), rows2, bands,
                  sc.attn.data(), nullptr, nullptr, sc.mamba_ext.data(),
                  sc.mamba_decay.data(), sc.mamba_h.data());
      for (int f = 0; f < frames2; ++f)
        for (int b = 0; b < bands; ++b)
          std::memcpy(
              sc.rev.data() +
                  static_cast<size_t>(f) * bands * width + b * width,
              sc.trans.data() + (static_cast<size_t>(f) * bands +
                                 (bands - 1 - b)) * width,
              sizeof(float) * width);
      mamba_layer(block.freq_mamba.backward, sc.rev.data(), rows2, bands,
                  sc.bwd.data(), nullptr, nullptr, sc.mamba_ext.data(),
                  sc.mamba_decay.data(), sc.mamba_h.data());
      for (int f = 0; f < frames2; ++f) {
        for (int b = 0; b < bands; ++b) {
          const size_t row = static_cast<size_t>(f) * bands + b;
          const size_t flipped =
              static_cast<size_t>(f) * bands + (bands - 1 - b);
          const float* fwd = sc.attn.data() + row * width;
          const float* bwd = sc.bwd.data() + flipped * width;
          const float* tr = sc.trans.data() + row * width;
          float* dst0 = sc.cat.data() + row * 2 * width;
          float* dst1 = dst0 + width;
          for (int col = 0; col < width; ++col) {
            dst0[col] = fwd[col] + tr[col];
            dst1[col] = bwd[col] + tr[col];
          }
        }
      }
      linear_par(sc.cat.data(), rows2, 2 * width, *block.freq_mamba.out_w,
                 block.freq_mamba.out_b->data, sc.norm.data());
      layer_norm_par(sc.norm.data(), sc.attn.data(), rows2, width,
                     block.freq_mamba.norm_w->data,
                     block.freq_mamba.norm_b->data);
      add_par(sc.trans.data(), sc.attn.data(), rows2 * width);
      to_band_major(sc.trans.data(), sc.pair.data(), bands, frames, width);
      to_band_major(sc.trans.data() + lane_floats,
                    sc.pair.data() + lane_floats, bands, frames, width);
    }
  }
  std::memcpy(feat_a, sc.pair.data(), sizeof(float) * lane_floats);
  std::memcpy(feat_b, sc.pair.data() + lane_floats, sizeof(float) * lane_floats);
}

// --------------------------------------------------------------- decoder ----

void Model::decode(const float* feat, int frames, float* raw) const {
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int bins = cfg_.bins();

  Timer timer(profile_enabled() ? &profile().decode : nullptr);
  default_pool().parallel_for(bands, [&](int b) {
    const auto& edge = cfg_.edges[static_cast<size_t>(b)];
    const int n_bins = edge.second - edge.first;
    const int out_dim = 4 * n_bins;
    const BandHead& head = heads_[static_cast<size_t>(b)];
    // Stack scratch for the common frames <= 4 & width <= 512 case (zero mallocs).
    alignas(64) float sx[4 * 512];
    alignas(64) float sn[4 * 512];
    alignas(64) float sp[4 * 512];
    alignas(64) float so[4 * 512];
    std::vector<float> vx_buf, vn_buf, vp_buf, vo_buf;
    float *xp = sx, *np = sn, *pp = sp, *op = so;
    if (frames > 4 || width > 512 || out_dim > 512) {
      vx_buf.resize(static_cast<size_t>(frames) * width);
      vn_buf.resize(vx_buf.size());
      vp_buf.resize(vx_buf.size());
      vo_buf.resize(static_cast<size_t>(frames) * out_dim);
      xp = vx_buf.data();
      np = vn_buf.data();
      pp = vp_buf.data();
      op = vo_buf.data();
    }
    for (int t = 0; t < frames; ++t) {
      const float* src =
          feat + (static_cast<size_t>(b) * frames + t) * width;
      float* dst = xp + static_cast<size_t>(t) * width;
      for (int j = 0; j < width; ++j)
        dst[j] = src[j] * (1.0f + decoder_scale_[static_cast<size_t>(j)]) +
                 decoder_shift_[static_cast<size_t>(j)];
    }
    layer_norm(xp, np, frames, width, head.n_w->data, head.n_b->data);
    int r = 0;
    for (; r + 4 <= frames; r += 4)
      v_linear_4rows(np + static_cast<size_t>(r) * width, width,
                     head.h1_w->data, width, head.h1_b->data,
                     pp + static_cast<size_t>(r) * width);
    if (r < frames)
      linear(np + static_cast<size_t>(r) * width, frames - r, width,
             *head.h1_w, head.h1_b->data, pp + static_cast<size_t>(r) * width);
    v_gelu(pp, frames * width);
    r = 0;
    for (; r + 4 <= frames; r += 4)
      v_linear_4rows(pp + static_cast<size_t>(r) * width, width,
                     head.h3_w->data, out_dim, head.h3_b->data,
                     op + static_cast<size_t>(r) * out_dim);
    if (r < frames)
      linear(pp + static_cast<size_t>(r) * width, frames - r, width,
             *head.h3_w, head.h3_b->data, op + static_cast<size_t>(r) * out_dim);
    for (int t = 0; t < frames; ++t)
      std::memcpy(raw + (static_cast<size_t>(t) * bins + edge.first) * 4,
                  op + static_cast<size_t>(t) * out_dim,
                  sizeof(float) * static_cast<size_t>(out_dim));
  });
  if (debug_decode) debug_decode(0, frames, raw);
}

void Model::decode_pair(const float* feat_a, const float* feat_b, int frames,
                        float* raw_a, float* raw_b) const {
  if (debug_decode) {
    decode(feat_a, frames, raw_a);
    decode(feat_b, frames, raw_b);
    return;
  }
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int bins = cfg_.bins();
  const int frames2 = 2 * frames;

  Timer timer(profile_enabled() ? &profile().decode : nullptr);
  default_pool().parallel_for(bands, [&](int b) {
    const auto& edge = cfg_.edges[static_cast<size_t>(b)];
    const int n_bins = edge.second - edge.first;
    const int out_dim = 4 * n_bins;
    const BandHead& head = heads_[static_cast<size_t>(b)];
    alignas(64) float sx[8 * 512];
    alignas(64) float sn[8 * 512];
    alignas(64) float sp[8 * 512];
    alignas(64) float so[8 * 512];
    std::vector<float> vx_buf, vn_buf, vp_buf, vo_buf;
    float *xp = sx, *np = sn, *pp = sp, *op = so;
    if (frames2 > 8 || width > 512 || out_dim > 512) {
      vx_buf.resize(static_cast<size_t>(frames2) * width);
      vn_buf.resize(vx_buf.size());
      vp_buf.resize(vx_buf.size());
      vo_buf.resize(static_cast<size_t>(frames2) * out_dim);
      xp = vx_buf.data();
      np = vn_buf.data();
      pp = vp_buf.data();
      op = vo_buf.data();
    }
    const float* feats[2] = {feat_a, feat_b};
    for (int l = 0; l < 2; ++l) {
      for (int t = 0; t < frames; ++t) {
        const float* src =
            feats[l] + (static_cast<size_t>(b) * frames + t) * width;
        float* dst = xp + static_cast<size_t>(l * frames + t) * width;
        for (int j = 0; j < width; ++j)
          dst[j] = src[j] * (1.0f + decoder_scale_[static_cast<size_t>(j)]) +
                   decoder_shift_[static_cast<size_t>(j)];
      }
    }
    layer_norm(xp, np, frames2, width, head.n_w->data, head.n_b->data);
    int r = 0;
    for (; r + 4 <= frames2; r += 4)
      v_linear_4rows(np + static_cast<size_t>(r) * width, width,
                     head.h1_w->data, width, head.h1_b->data,
                     pp + static_cast<size_t>(r) * width);
    if (r < frames2)
      linear(np + static_cast<size_t>(r) * width, frames2 - r, width,
             *head.h1_w, head.h1_b->data, pp + static_cast<size_t>(r) * width);
    v_gelu(pp, frames2 * width);
    r = 0;
    for (; r + 4 <= frames2; r += 4)
      v_linear_4rows(pp + static_cast<size_t>(r) * width, width,
                     head.h3_w->data, out_dim, head.h3_b->data,
                     op + static_cast<size_t>(r) * out_dim);
    if (r < frames2)
      linear(pp + static_cast<size_t>(r) * width, frames2 - r, width,
             *head.h3_w, head.h3_b->data, op + static_cast<size_t>(r) * out_dim);
    for (int t = 0; t < frames; ++t) {
      std::memcpy(raw_a + (static_cast<size_t>(t) * bins + edge.first) * 4,
                  op + static_cast<size_t>(t) * out_dim,
                  sizeof(float) * static_cast<size_t>(out_dim));
      std::memcpy(raw_b + (static_cast<size_t>(t) * bins + edge.first) * 4,
                  op + static_cast<size_t>(frames + t) * out_dim,
                  sizeof(float) * static_cast<size_t>(out_dim));
    }
  });
}

void Model::correction(const float* raw, const std::complex<float>* spectrum,
                       int bins, std::complex<float>* out) const {
  const float bound = cfg_.correction_bound;
  for (int k = 0; k < bins; ++k) {
    const float* r = raw + static_cast<size_t>(k) * 4;
    const float mr = 2.0f * std::tanh(r[0]);
    const float mi = 2.0f * std::tanh(r[1]);
    const float ar = std::tanh(r[2]);
    const float ai = std::tanh(r[3]);
    const float norm = std::max(std::sqrt(ar * ar + ai * ai), 1.0f);
    const std::complex<float>& s = spectrum[k];
    const float magnitude = bound * std::sqrt(s.real() * s.real() +
                                             s.imag() * s.imag());
    const float rr = ar / norm * magnitude;
    const float ri = ai / norm * magnitude;
    out[k] = std::complex<float>(s.real() * mr - s.imag() * mi + rr,
                                 s.real() * mi + s.imag() * mr + ri);
  }
}

}  // namespace vi
