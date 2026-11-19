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

// How many channel ranges to carve each task into: enough to fill the pool,
// but not so fine that the fixed cost per unit dominates.
int channel_split(int tasks, int inner) {
  const int threads = default_pool().size();
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

inline void rope_all(float* x, int width, int head_dim, const float* cos,
                     const float* sin, bool inverse) {
  for (int base = 0; base < width; base += head_dim)
    rope_head(x + base, head_dim, cos, sin, inverse);
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
    // -exp(A_log), laid out [state][inner] for the recurrence.
    l.a.resize(static_cast<size_t>(state) * inner);
    for (int c = 0; c < inner; ++c)
      for (int n = 0; n < state; ++n)
        l.a[static_cast<size_t>(n) * inner + c] =
            -std::exp(a_log.data[static_cast<size_t>(c) * state + n]);
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

void Model::linear_par(const float* a, int m, int k, int stride, const Tensor& w,
                       const float* bias, float* c) const {
  const int n = w.dim(0);
  const int threads = default_pool().size();
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

void Model::elementwise_par(float* x, int n, void (*fn)(float*, int)) const {
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
          float* kb = st.k.data() + static_cast<size_t>(b) * st.capacity * width;
          float* vb = st.v.data() + static_cast<size_t>(b) * st.capacity * width;
          for (int h = 0; h < heads; ++h) {
            float* kh = kb + static_cast<size_t>(h) * lane;
            for (int s = 0; s < keep; ++s)
              rope_head(kh + static_cast<size_t>(s + shift) * hd, hd, cos, sin,
                        /*inverse=*/true);
            std::memmove(kh, kh + static_cast<size_t>(shift) * hd,
                         sizeof(float) * keep * hd);
            std::memmove(vb + static_cast<size_t>(h) * lane,
                         vb + static_cast<size_t>(h) * lane + shift * hd,
                         sizeof(float) * keep * hd);
          }
        }
        st.end = keep;
        first = keep;
      }
    }
  }
  reserve_slots(st, first + frames, lp.capacity_limit, bands, width, hd);
  if (sc.scores.size() <
      static_cast<size_t>(bands) * static_cast<size_t>(lp.capacity_limit))
    sc.scores.resize(static_cast<size_t>(bands) * lp.capacity_limit);

  layer_norm_par(feat, sc.norm.data(), rows, width, w.n1w->data, w.n1b->data);
  linear_par(sc.norm.data(), rows, width, *w.qkv_w, w.qkv_b->data,
             sc.qkv.data());

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const int past_lo = std::max(0, first - lp.cache_frames);
  // Keys and values live head-major: [band][head][slot][hd]. A head then walks
  // its whole window as one contiguous run instead of touching one eighth of
  // every cache line.
  const size_t lane_stride = static_cast<size_t>(st.capacity) * hd;

  default_pool().parallel_for(bands, [&](int b) {
    float* kbase = st.k.data() + static_cast<size_t>(b) * st.capacity * width;
    float* vbase = st.v.data() + static_cast<size_t>(b) * st.capacity * width;
    float* scores = sc.scores.data() + static_cast<size_t>(b) * lp.capacity_limit;
    for (int t = 0; t < frames; ++t) {
      const int pos = first + t;
      float* row =
          sc.qkv.data() + static_cast<size_t>(b * frames + t) * 3 * width;
      float* out =
          sc.attn.data() + static_cast<size_t>(b * frames + t) * width;
      const float* cos = rope.cos_at(pos);
      const float* sin = rope.sin_at(pos);

      // One pass over the heads to rotate, scale and cache this token.
      for (int h = 0; h < heads; ++h) {
        float* q = row + h * hd;
        float* k = row + width + h * hd;
        const float* v = row + 2 * width + h * hd;
        rope_head(q, hd, cos, sin, false);
        rope_head(k, hd, cos, sin, false);
        std::memcpy(kbase + static_cast<size_t>(h) * lane_stride +
                        static_cast<size_t>(pos) * hd,
                    k, sizeof(float) * hd);
        std::memcpy(vbase + static_cast<size_t>(h) * lane_stride +
                        static_cast<size_t>(pos) * hd,
                    v, sizeof(float) * hd);
      }

      int lo = past_lo;
      if (lp.window > 0) lo = std::max(lo, pos - lp.window + 1);
      const int n = pos - lo + 1;
      const size_t first_slot = static_cast<size_t>(lo) * hd;

      for (int h = 0; h < heads; ++h) {
        const float* q = row + h * hd;
        const float* keys =
            kbase + static_cast<size_t>(h) * lane_stride + first_slot;
        float* s = scores;
        const float sum = scores_softmax(q, keys, n, hd, scale, s);
        const float inv = 1.0f / sum;
        float* o = out + h * hd;
        const float* values =
            vbase + static_cast<size_t>(h) * lane_stride + first_slot;
        pv_row(s, inv, values, n, hd, o);
        // Exclusive self attention: drop the component along the token's value.
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
  elementwise_par(sc.ffn.data(), rows * 4 * width, v_gelu);
  linear_par(sc.ffn.data(), rows, 4 * width, *w.f1w, w.f1b->data,
             sc.trans.data());
  add_par(feat, sc.trans.data(), rows * width);
}

void Model::band_transformer(const AttnWeights& w, float* feat,
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

  Timer timer(profile_enabled() ? &profile().band_attn : nullptr);
  {
    // Per frame the bands are one short sequence; flattening (frame, band) rows
    // lets every projection run as one wide GEMM over all frames.
    Timer slot(profile_enabled() ? &profile().transpose : nullptr);
    to_frame_major(feat, sc.trans.data(), bands, frames, width);
  }

  layer_norm_par(sc.trans.data(), sc.norm.data(), rows, width, w.bn1w->data,
                 w.bn1b->data);
  linear_par(sc.norm.data(), rows, width, *w.in_w, w.in_b->data, sc.qkv.data());

  // Attention is 46 tokens of 64 lanes per (frame, head). The qkv rows are
  // 3*width apart in the GEMM output, so transpose once into a head-major
  // layout and every head then reads contiguous 46x64 tiles.
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
    float* out = sc.attn.data() + static_cast<size_t>(f) * bands * width + h * hd;
    float* scores = sc.bscores.data() + static_cast<size_t>(unit) * bands * bands;
    for (int i = 0; i < bands; ++i) {
      float* s = scores + static_cast<size_t>(i) * bands;
      const float sum = scores_softmax(q + static_cast<size_t>(i) * hd, k,
                                       bands, hd, scale, s);
      const float inv = 1.0f / sum;
      float* o = out + static_cast<size_t>(i) * width;
      pv_row(s, inv, v, bands, hd, o);
    }
  });
  // Exclusive self attention is off across bands for this checkpoint.
  linear_par(sc.attn.data(), rows, width, *w.out_w, w.out_b->data,
             sc.norm.data());
  add_par(sc.trans.data(), sc.norm.data(), rows * width);

  layer_norm_par(sc.trans.data(), sc.norm.data(), rows, width, w.bn2w->data,
                 w.bn2b->data);
  linear_par(sc.norm.data(), rows, width, *w.bl1w, w.bl1b->data, sc.ffn.data());
  elementwise_par(sc.ffn.data(), rows * 4 * width, v_gelu);
  linear_par(sc.ffn.data(), rows, 4 * width, *w.bl2w, w.bl2b->data,
             sc.norm.data());
  add_par(sc.trans.data(), sc.norm.data(), rows * width);
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
  // task is split into channel ranges: the frequency layer only has `frames`
  // tasks (4) and would otherwise leave most of the pool idle.
  const int split = channel_split(tasks, inner);
  {
    Timer slot(profile_enabled() ? &profile().mamba_conv : nullptr);
    default_pool().parallel_for(tasks * split, [&](int unit) {
      const int task = unit / split;
      const int part = unit % split;
      const int c0 = static_cast<int>(static_cast<int64_t>(inner) * part / split);
      const int c1 =
          static_cast<int>(static_cast<int64_t>(inner) * (part + 1) / split);
      if (c1 <= c0) return;
      const int channels = c1 - c0;
      float* ext = slab_ext + static_cast<size_t>(task) * ext_stride;
      if (conv_state != nullptr) {
        const float* carried =
            conv_state + static_cast<size_t>(task) * inner * taps;
        for (int c = c0; c < c1; ++c)
          for (int j = 0; j < taps; ++j)
            ext[static_cast<size_t>(j) * inner + c] =
                carried[static_cast<size_t>(c) * taps + j];
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
      // Depthwise convolution, with the SiLU of the result folded in: the pass
      // that used to read and rewrite the whole buffer goes away.
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
      if (conv_state != nullptr) {
        float* carried = conv_state + static_cast<size_t>(task) * inner * taps;
        for (int c = c0; c < c1; ++c)
          for (int j = 0; j < taps; ++j)
            carried[static_cast<size_t>(c) * taps + j] =
                ext[static_cast<size_t>(len + j) * inner + c];
      }
    });
  }

  linear_par(sc.u2.data(), rows, inner, *l.x_w, nullptr, sc.xp.data());
  linear_par(sc.xp.data(), rows, dt_rank, dt_total, *l.dt_w, l.dt_b->data,
             sc.delta.data());
  elementwise_par(sc.delta.data(), rows * inner, v_softplus);

  {
    Timer slot(profile_enabled() ? &profile().mamba_scan : nullptr);
    default_pool().parallel_for(tasks * split, [&](int unit) {
      const int task = unit / split;
      const int part = unit % split;
      const int c0 = static_cast<int>(static_cast<int64_t>(inner) * part / split);
      const int c1 =
          static_cast<int>(static_cast<int64_t>(inner) * (part + 1) / split);
      if (c1 <= c0) return;
      const int channels = c1 - c0;
      float* h = ssm_state != nullptr
                     ? ssm_state + static_cast<size_t>(task) * state_stride
                     : slab_h + static_cast<size_t>(task) * state_stride;
      if (ssm_state == nullptr) {
        for (int n = 0; n < state; ++n)
          std::memset(h + static_cast<size_t>(n) * inner + c0, 0,
                      sizeof(float) * static_cast<size_t>(channels));
      }
      // delta*u is shared by every state row, so it is computed once per step
      // into scratch that used to hold the decay tensor.
      float* du = slab_decay + static_cast<size_t>(task) * state_stride + c0;
      const vx::batch one(1.0f);
      for (int t = 0; t < len; ++t) {
        const size_t row = static_cast<size_t>(task) * len + t;
        const float* u = sc.u2.data() + row * inner;
        const float* d = sc.delta.data() + row * inner;
        const float* bv = sc.xp.data() + row * dt_total + dt_rank;
        const float* cv = bv + state;
        const float* z = sc.xz.data() + row * 2 * inner + inner;
        float* y = sc.y.data() + row * inner;

        int c = 0;
        for (; c + vx::lanes <= channels; c += vx::lanes) {
          storef(du + c, loadf(d + c0 + c) * loadf(u + c0 + c));
          storef(y + c0 + c, vx::batch(0.0f));
        }
        for (; c < channels; ++c) {
          du[c] = d[c0 + c] * u[c0 + c];
          y[c0 + c] = 0.0f;
        }
        for (int n = 0; n < state; ++n)
          v_scan_step(h + static_cast<size_t>(n) * inner + c0,
                      l.a.data() + static_cast<size_t>(n) * inner + c0,
                      d + c0, du, bv[n], cv[n], y + c0, channels);

        // y = (y + D*u) * silu(z)
        c = 0;
        for (; c + vx::lanes <= channels; c += vx::lanes) {
          const vx::batch uv = loadf(u + c0 + c);
          const vx::batch zv = loadf(z + c0 + c);
          const vx::batch silu = zv / (one + xsimd::exp(-zv));
          storef(y + c0 + c,
                 (loadf(y + c0 + c) + loadf(l.d->data + c0 + c) * uv) * silu);
        }
        for (; c < channels; ++c) {
          const int at = c0 + c;
          y[at] = (y[at] + l.d->data[at] * u[at]) * silu_scalar(z[at]);
        }
      }
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
      for (size_t k = 0; k < static_cast<size_t>(rows) * width; ++k) {
        const size_t row = k / width;
        const int band = static_cast<int>(row % bands);
        const size_t flipped = (row - static_cast<size_t>(band)) +
                               static_cast<size_t>(bands - 1 - band);
        const size_t col = k % width;
        sc.cat[row * 2 * width + col] = sc.attn[k] + sc.trans[k];
        sc.cat[row * 2 * width + width + col] =
            sc.bwd[flipped * width + col] + sc.trans[k];
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

// --------------------------------------------------------------- decoder ----

void Model::decode(const float* feat, int frames, float* raw) const {
  const int bands = cfg_.bands();
  const int width = cfg_.width;
  const int bins = cfg_.bins();

  Timer timer(profile_enabled() ? &profile().decode : nullptr);
  default_pool().parallel_for(bands, [&](int b) {
    const auto& edge = cfg_.edges[static_cast<size_t>(b)];
    const int n_bins = edge.second - edge.first;
    const BandHead& head = heads_[static_cast<size_t>(b)];
    std::vector<float> x(static_cast<size_t>(frames) * width);
    std::vector<float> n(x.size());
    std::vector<float> p(x.size());
    std::vector<float> o(static_cast<size_t>(frames) * 4 * n_bins);
    for (int t = 0; t < frames; ++t) {
      const float* src =
          feat + (static_cast<size_t>(b) * frames + t) * width;
      float* dst = x.data() + static_cast<size_t>(t) * width;
      for (int j = 0; j < width; ++j)
        dst[j] = src[j] * (1.0f + decoder_scale_[static_cast<size_t>(j)]) +
                 decoder_shift_[static_cast<size_t>(j)];
    }
    layer_norm(x.data(), n.data(), frames, width, head.n_w->data,
               head.n_b->data);
    linear(n.data(), frames, width, *head.h1_w, head.h1_b->data, p.data());
    v_gelu(p.data(), frames * width);
    linear(p.data(), frames, width, *head.h3_w, head.h3_b->data, o.data());
    for (int t = 0; t < frames; ++t)
      for (int i = 0; i < n_bins; ++i)
        for (int j = 0; j < 4; ++j)
          raw[(static_cast<size_t>(t) * bins + edge.first + i) * 4 + j] =
              o[(static_cast<size_t>(t) * n_bins + i) * 4 + j];
  });
  if (debug_decode) debug_decode(0, frames, raw);
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
