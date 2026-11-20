// Apple Metal 3 + MPS GPU backend for the isolator backbone.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_backend.h"
#include "model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vi {
namespace {

static NSString* const kMetalShaderSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float fast_gelu(float x) {
  float ax = fabs(x);
  float u = ax * 0.70710678118654752440f;
  float t = 1.0f / (1.0f + 0.38f * u);
  float p = 3.196994254e-2f;
  p = fma(p, t, -2.618669679e-1f);
  p = fma(p, t,  5.775581467e-1f);
  p = fma(p, t, -2.990170914e-1f);
  p = fma(p, t,  3.732227110e-1f);
  p = fma(p, t,  1.411143942e-1f);
  p = fma(p, t,  2.231841650e-1f);
  p = fma(p, t,  2.138345334e-1f);
  float erfc_u = exp(-u * u) * t * p;
  float half_x_erfc = 0.5f * x * erfc_u;
  return (x >= 0.0f) ? (x - half_x_erfc) : half_x_erfc;
}

inline float fast_silu(float x) {
  return x / (1.0f + exp(-x));
}

inline float fast_softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -15.0f) return exp(x);
  return log(1.0f + exp(x));
}

// 1 threadgroup (256 threads) per row.
kernel void layer_norm_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device const float* b [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  threadgroup float sh[8];
  threadgroup float sh_mean;
  threadgroup float sh_inv;
  device const float* xr = x + size_t(row) * cols;
  device float* yr = y + size_t(row) * cols;

  float sum = 0.0f;
  for (uint i = tid; i < cols; i += tg_size) sum += xr[i];
  sum = simd_sum(sum);
  if ((tid & 31u) == 0u) sh[tid >> 5] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint n_sg = (tg_size + 31u) >> 5;
  if (tid < 32u) {
    float v = (tid < n_sg) ? sh[tid] : 0.0f;
    v = simd_sum(v);
    if (tid == 0u) sh_mean = v / float(cols);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean = sh_mean;

  float sq = 0.0f;
  for (uint i = tid; i < cols; i += tg_size) {
    float d = xr[i] - mean;
    sq += d * d;
  }
  sq = simd_sum(sq);
  if ((tid & 31u) == 0u) sh[tid >> 5] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32u) {
    float v = (tid < n_sg) ? sh[tid] : 0.0f;
    v = simd_sum(v);
    if (tid == 0u) sh_inv = rsqrt(v / float(cols) + 1e-5f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = sh_inv;

  for (uint i = tid; i < cols; i += tg_size) {
    yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
  }
}

kernel void add_bias_f32(
    device float* x [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= total) return;
  x[i] += b[i % cols];
}

kernel void add_bias_gelu_f32(
    device float* x [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= total) return;
  x[i] = fast_gelu(x[i] + b[i % cols]);
}

kernel void add_bias_residual_f32(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const float* b [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& total [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= total) return;
  dst[i] += src[i] + b[i % cols];
}

kernel void add_vec_f32(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant uint& total [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= total) return;
  dst[i] += src[i];
}

kernel void add_bias_softplus_f32(
    device float* x [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= total) return;
  x[i] = fast_softplus(x[i] + b[i % cols]);
}

kernel void f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& total [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= total) return;
  dst[i] = half(src[i]);
}

kernel void f32_to_f16_strided(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& src_stride [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
  uint total = rows * cols;
  if (idx >= total) return;
  uint r = idx / cols;
  uint c = idx % cols;
  dst[idx] = half(src[size_t(r) * src_stride + c]);
}

// [lanes][bands][frames][width] -> [lanes][frames][bands][width]
kernel void to_frame_major_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& bands [[buffer(2)]],
    constant uint& frames [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& lanes [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  size_t total = size_t(lanes) * bands * frames * width;
  if (idx >= total) return;
  uint col = idx % width;
  uint row = idx / width;
  uint lane_rows = bands * frames;
  uint l = row / lane_rows;
  uint rem = row % lane_rows;
  uint b = rem / frames;
  uint t = rem % frames;
  size_t dst_row = size_t(l) * lane_rows + size_t(t) * bands + b;
  dst[dst_row * width + col] = src[idx];
}

// [lanes][frames][bands][width] -> [lanes][bands][frames][width]
kernel void to_band_major_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& bands [[buffer(2)]],
    constant uint& frames [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& lanes [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  size_t total = size_t(lanes) * bands * frames * width;
  if (idx >= total) return;
  uint col = idx % width;
  uint row = idx / width;
  uint lane_rows = bands * frames;
  uint l = row / lane_rows;
  uint rem = row % lane_rows;
  uint t = rem / bands;
  uint b = rem % bands;
  size_t dst_row = size_t(l) * lane_rows + size_t(b) * frames + t;
  dst[dst_row * width + col] = src[idx];
}

// Reverse bands per frame: rev[f][b] = trans[f][bands - 1 - b]
kernel void reverse_bands_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& frames_total [[buffer(2)]],
    constant uint& bands [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
  size_t total = size_t(frames_total) * bands * width;
  if (idx >= total) return;
  uint col = idx % width;
  uint row = idx / width;
  uint f = row / bands;
  uint b = row % bands;
  size_t src_row = size_t(f) * bands + (bands - 1u - b);
  dst[idx] = src[src_row * width + col];
}

// cat[row, 0..width-1] = fwd[row] + trans[row]
// cat[row, width..2*width-1] = bwd[flipped] + trans[row]
kernel void mamba_freq_cat_f32(
    device const float* fwd [[buffer(0)]],
    device const float* bwd [[buffer(1)]],
    device const float* trans [[buffer(2)]],
    device float* cat [[buffer(3)]],
    constant uint& frames_total [[buffer(4)]],
    constant uint& bands [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  size_t total = size_t(frames_total) * bands * width;
  if (idx >= total) return;
  uint col = idx % width;
  uint row = idx / width;
  uint f = row / bands;
  uint b = row % bands;
  size_t flipped = size_t(f) * bands + (bands - 1u - b);
  float tr = trans[idx];
  size_t out_base = size_t(row) * 2u * width;
  cat[out_base + col] = fwd[idx] + tr;
  cat[out_base + width + col] = bwd[flipped * width + col] + tr;
}

// Compact KV cache in-place (fp32): 1 threadgroup (half = hd/2 threads) per (b, h).
kernel void compact_kv_f32(
    device float* k_cache [[buffer(0)]],
    device float* v_cache [[buffer(1)]],
    device const float* rope_cos [[buffer(2)]],
    device const float* rope_sin [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& hd [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& keep [[buffer(7)]],
    constant uint& shift [[buffer(8)]],
    uint bh [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]]) {
  uint half_hd = hd >> 1;
  if (i >= half_hd) return;
  uint b = bh / heads;
  uint h = bh % heads;
  uint width = heads * hd;
  size_t base = size_t(b) * capacity * width + size_t(h) * capacity * hd;
  float c = rope_cos[size_t(shift) * half_hd + i];
  float s = rope_sin[size_t(shift) * half_hd + i];
  for (uint slot = 0; slot < keep; ++slot) {
    size_t src = base + size_t(slot + shift) * hd;
    size_t dst = base + size_t(slot) * hd;
    float a = k_cache[src + i];
    float bb = k_cache[src + i + half_hd];
    // inverse RoPE: sign = -1
    k_cache[dst + i] = a * c + bb * s;
    k_cache[dst + i + half_hd] = -a * s + bb * c;
    v_cache[dst + i] = v_cache[src + i];
    v_cache[dst + i + half_hd] = v_cache[src + i + half_hd];
  }
}

kernel void compact_kv_f16(
    device half* k_cache [[buffer(0)]],
    device half* v_cache [[buffer(1)]],
    device const float* rope_cos [[buffer(2)]],
    device const float* rope_sin [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& hd [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    constant uint& keep [[buffer(7)]],
    constant uint& shift [[buffer(8)]],
    uint bh [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]]) {
  uint half_hd = hd >> 1;
  if (i >= half_hd) return;
  uint b = bh / heads;
  uint h = bh % heads;
  uint width = heads * hd;
  size_t base = size_t(b) * capacity * width + size_t(h) * capacity * hd;
  float c = rope_cos[size_t(shift) * half_hd + i];
  float s = rope_sin[size_t(shift) * half_hd + i];
  for (uint slot = 0; slot < keep; ++slot) {
    size_t src = base + size_t(slot + shift) * hd;
    size_t dst = base + size_t(slot) * hd;
    float a = float(k_cache[src + i]);
    float bb = float(k_cache[src + i + half_hd]);
    k_cache[dst + i] = half(a * c + bb * s);
    k_cache[dst + i + half_hd] = half(-a * s + bb * c);
    v_cache[dst + i] = v_cache[src + i];
    v_cache[dst + i + half_hd] = v_cache[src + i + half_hd];
  }
}

struct RopeStoreParams {
  uint bands;
  uint frames;
  uint heads;
  uint hd;
  uint first;
  uint capacity;
  uint row_offset;
};

// Applies QKV bias + forward RoPE to Q and K, stores K and V into KV cache.
// Grid: bands * frames * heads * (hd / 2) threads.
kernel void rope_store_kv_f32(
    device float* qkv [[buffer(0)]],
    device const float* qkv_bias [[buffer(1)]],
    device float* k_cache [[buffer(2)]],
    device float* v_cache [[buffer(3)]],
    device const float* rope_cos [[buffer(4)]],
    device const float* rope_sin [[buffer(5)]],
    constant RopeStoreParams& p [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  uint half_hd = p.hd >> 1;
  uint total = p.bands * p.frames * p.heads * half_hd;
  if (idx >= total) return;
  uint i = idx % half_hd;
  uint rem = idx / half_hd;
  uint h = rem % p.heads;
  rem /= p.heads;
  uint t = rem % p.frames;
  uint b = rem / p.frames;

  uint width = p.heads * p.hd;
  uint pos = p.first + t;
  size_t row_idx = size_t(p.row_offset + b * p.frames + t);
  device float* row = qkv + row_idx * 3u * width;

  float c = rope_cos[size_t(pos) * half_hd + i];
  float s = rope_sin[size_t(pos) * half_hd + i];

  uint q_off0 = h * p.hd + i;
  uint q_off1 = q_off0 + half_hd;
  float qa = row[q_off0] + qkv_bias[q_off0];
  float qb = row[q_off1] + qkv_bias[q_off1];
  row[q_off0] = qa * c - qb * s;
  row[q_off1] = qa * s + qb * c;

  uint k_off0 = width + q_off0;
  uint k_off1 = width + q_off1;
  float ka = row[k_off0] + qkv_bias[k_off0];
  float kb = row[k_off1] + qkv_bias[k_off1];
  float rk0 = ka * c - kb * s;
  float rk1 = ka * s + kb * c;
  row[k_off0] = rk0;
  row[k_off1] = rk1;

  uint v_off0 = 2u * width + q_off0;
  uint v_off1 = 2u * width + q_off1;
  float v0 = row[v_off0] + qkv_bias[v_off0];
  float v1 = row[v_off1] + qkv_bias[v_off1];
  row[v_off0] = v0;
  row[v_off1] = v1;

  size_t slot = size_t(b) * p.capacity * width + size_t(h) * p.capacity * p.hd +
                size_t(pos) * p.hd;
  k_cache[slot + i] = rk0;
  k_cache[slot + i + half_hd] = rk1;
  v_cache[slot + i] = v0;
  v_cache[slot + i + half_hd] = v1;
}

kernel void rope_store_kv_f16(
    device float* qkv [[buffer(0)]],
    device const float* qkv_bias [[buffer(1)]],
    device half* k_cache [[buffer(2)]],
    device half* v_cache [[buffer(3)]],
    device const float* rope_cos [[buffer(4)]],
    device const float* rope_sin [[buffer(5)]],
    constant RopeStoreParams& p [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  uint half_hd = p.hd >> 1;
  uint total = p.bands * p.frames * p.heads * half_hd;
  if (idx >= total) return;
  uint i = idx % half_hd;
  uint rem = idx / half_hd;
  uint h = rem % p.heads;
  rem /= p.heads;
  uint t = rem % p.frames;
  uint b = rem / p.frames;

  uint width = p.heads * p.hd;
  uint pos = p.first + t;
  size_t row_idx = size_t(p.row_offset + b * p.frames + t);
  device float* row = qkv + row_idx * 3u * width;

  float c = rope_cos[size_t(pos) * half_hd + i];
  float s = rope_sin[size_t(pos) * half_hd + i];

  uint q_off0 = h * p.hd + i;
  uint q_off1 = q_off0 + half_hd;
  float qa = row[q_off0] + qkv_bias[q_off0];
  float qb = row[q_off1] + qkv_bias[q_off1];
  row[q_off0] = qa * c - qb * s;
  row[q_off1] = qa * s + qb * c;

  uint k_off0 = width + q_off0;
  uint k_off1 = width + q_off1;
  float ka = row[k_off0] + qkv_bias[k_off0];
  float kb = row[k_off1] + qkv_bias[k_off1];
  float rk0 = ka * c - kb * s;
  float rk1 = ka * s + kb * c;
  row[k_off0] = rk0;
  row[k_off1] = rk1;

  uint v_off0 = 2u * width + q_off0;
  uint v_off1 = 2u * width + q_off1;
  float v0 = row[v_off0] + qkv_bias[v_off0];
  float v1 = row[v_off1] + qkv_bias[v_off1];
  row[v_off0] = v0;
  row[v_off1] = v1;

  size_t slot = size_t(b) * p.capacity * width + size_t(h) * p.capacity * p.hd +
                size_t(pos) * p.hd;
  k_cache[slot + i] = half(rk0);
  k_cache[slot + i + half_hd] = half(rk1);
  v_cache[slot + i] = half(v0);
  v_cache[slot + i + half_hd] = half(v1);
}

struct TimeAttnParams {
  uint bands;
  uint frames;
  uint heads;
  uint hd;
  uint first;
  uint capacity;
  uint past_lo;
  uint window;
  uint row_offset;
  float scale;
};

// Register-resident simdgroup FlashAttention for time_transformer (hd = 64 -> 2 elems/thread).
// 1 threadgroup (32 threads) per (b, t, h).
kernel void time_attn_f32(
    device const float* qkv [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* attn [[buffer(3)]],
    constant TimeAttnParams& p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint h = tg % p.heads;
  uint rem = tg / p.heads;
  uint t = rem % p.frames;
  uint b = rem / p.frames;
  uint width = p.heads * p.hd;
  size_t row_idx = size_t(p.row_offset + b * p.frames + t);
  device const float* row = qkv + row_idx * 3u * width;

  uint pos = p.first + t;
  int lo = int(p.past_lo);
  if (p.window > 0u) lo = max(lo, int(pos) - int(p.window) + 1);

  float q0 = row[h * p.hd + tid];
  float q1 = row[h * p.hd + tid + 32u];
  size_t head_base = size_t(b) * p.capacity * width + size_t(h) * p.capacity * p.hd;

  float m = -INFINITY;
  float l = 0.0f;
  float o0 = 0.0f, o1 = 0.0f;
  for (int j = lo; j <= int(pos); ++j) {
    size_t off = head_base + size_t(j) * p.hd;
    float score = simd_sum(q0 * k_cache[off + tid] + q1 * k_cache[off + tid + 32u]) * p.scale;
    float m_new = max(m, score);
    float alpha = (m == -INFINITY) ? 0.0f : exp(m - m_new);
    float w = exp(score - m_new);
    o0 = o0 * alpha + w * v_cache[off + tid];
    o1 = o1 * alpha + w * v_cache[off + tid + 32u];
    l = l * alpha + w;
    m = m_new;
  }
  float inv_l = 1.0f / l;
  o0 *= inv_l;
  o1 *= inv_l;

  float vc0 = row[2u * width + h * p.hd + tid];
  float vc1 = row[2u * width + h * p.hd + tid + 32u];
  float dv = simd_sum(o0 * vc0 + o1 * vc1);
  float nv = simd_sum(vc0 * vc0 + vc1 * vc1);
  float coef = dv / max(nv, 1e-6f);
  o0 -= coef * vc0;
  o1 -= coef * vc1;

  device float* out = attn + row_idx * width + h * p.hd;
  out[tid] = o0;
  out[tid + 32u] = o1;
}

kernel void time_attn_f16kv(
    device const float* qkv [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* attn [[buffer(3)]],
    constant TimeAttnParams& p [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint h = tg % p.heads;
  uint rem = tg / p.heads;
  uint t = rem % p.frames;
  uint b = rem / p.frames;
  uint width = p.heads * p.hd;
  size_t row_idx = size_t(p.row_offset + b * p.frames + t);
  device const float* row = qkv + row_idx * 3u * width;

  uint pos = p.first + t;
  int lo = int(p.past_lo);
  if (p.window > 0u) lo = max(lo, int(pos) - int(p.window) + 1);

  float q0 = row[h * p.hd + tid];
  float q1 = row[h * p.hd + tid + 32u];
  size_t head_base = size_t(b) * p.capacity * width + size_t(h) * p.capacity * p.hd;

  float m = -INFINITY;
  float l = 0.0f;
  float o0 = 0.0f, o1 = 0.0f;
  for (int j = lo; j <= int(pos); ++j) {
    size_t off = head_base + size_t(j) * p.hd;
    float score = simd_sum(q0 * float(k_cache[off + tid]) +
                           q1 * float(k_cache[off + tid + 32u])) * p.scale;
    float m_new = max(m, score);
    float alpha = (m == -INFINITY) ? 0.0f : exp(m - m_new);
    float w = exp(score - m_new);
    o0 = o0 * alpha + w * float(v_cache[off + tid]);
    o1 = o1 * alpha + w * float(v_cache[off + tid + 32u]);
    l = l * alpha + w;
    m = m_new;
  }
  float inv_l = 1.0f / l;
  o0 *= inv_l;
  o1 *= inv_l;

  float vc0 = row[2u * width + h * p.hd + tid];
  float vc1 = row[2u * width + h * p.hd + tid + 32u];
  float dv = simd_sum(o0 * vc0 + o1 * vc1);
  float nv = simd_sum(vc0 * vc0 + vc1 * vc1);
  float coef = dv / max(nv, 1e-6f);
  o0 -= coef * vc0;
  o1 -= coef * vc1;

  device float* out = attn + row_idx * width + h * p.hd;
  out[tid] = o0;
  out[tid + 32u] = o1;
}

struct BandAttnParams {
  uint frames_total;
  uint bands;
  uint heads;
  uint hd;
  float scale;
};

// 1 threadgroup (32 threads) per (f, b_q, h). Reads QKV directly (no bqkv transpose!).
kernel void band_attn_f32(
    device const float* qkv [[buffer(0)]],
    device float* attn [[buffer(1)]],
    constant BandAttnParams& p [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint h = tg % p.heads;
  uint rem = tg / p.heads;
  uint bq = rem % p.bands;
  uint f = rem / p.bands;
  uint width = p.heads * p.hd;
  uint w3 = 3u * width;
  size_t frame_base = size_t(f) * p.bands * w3;
  device const float* qrow = qkv + frame_base + size_t(bq) * w3 + h * p.hd;

  float q0 = qrow[tid];
  float q1 = qrow[tid + 32u];

  float m = -INFINITY;
  float l = 0.0f;
  float o0 = 0.0f, o1 = 0.0f;
  for (uint bk = 0; bk < p.bands; ++bk) {
    device const float* krow = qkv + frame_base + size_t(bk) * w3 + width + h * p.hd;
    device const float* vrow = krow + width;
    float score = simd_sum(q0 * krow[tid] + q1 * krow[tid + 32u]) * p.scale;
    float m_new = max(m, score);
    float alpha = (m == -INFINITY) ? 0.0f : exp(m - m_new);
    float w = exp(score - m_new);
    o0 = o0 * alpha + w * vrow[tid];
    o1 = o1 * alpha + w * vrow[tid + 32u];
    l = l * alpha + w;
    m = m_new;
  }
  float inv_l = 1.0f / l;
  device float* out = attn + (size_t(f) * p.bands + bq) * width + h * p.hd;
  out[tid] = o0 * inv_l;
  out[tid + 32u] = o1 * inv_l;
}

struct MambaConvParams {
  uint tasks;
  uint len;
  uint inner;
  uint has_state;
  uint task_offset;
};

// Fused 4-tap depthwise conv1d + bias + SiLU; reads xz directly without slab_ext!
// 1 thread per (task, channel).
kernel void mamba_conv4_silu_f32(
    device const float* xz [[buffer(0)]],
    device const float* conv_t [[buffer(1)]],
    device const float* conv_b [[buffer(2)]],
    device float* conv_state [[buffer(3)]],
    device float* u2 [[buffer(4)]],
    constant MambaConvParams& p [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  uint total = p.tasks * p.inner;
  if (idx >= total) return;
  uint c = idx % p.inner;
  uint task = idx / p.inner;
  uint global_task = p.task_offset + task;

  float w0 = conv_t[0u * p.inner + c];
  float w1 = conv_t[1u * p.inner + c];
  float w2 = conv_t[2u * p.inner + c];
  float w3 = conv_t[3u * p.inner + c];
  float bias = conv_b[c];

  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
  size_t st_base = size_t(task) * 3u * p.inner + c;
  if (p.has_state != 0u) {
    s0 = conv_state[st_base + 0u * p.inner];
    s1 = conv_state[st_base + 1u * p.inner];
    s2 = conv_state[st_base + 2u * p.inner];
  }

  size_t row0 = size_t(global_task) * p.len;
  for (uint t = 0; t < p.len; ++t) {
    float x = xz[(row0 + t) * 2u * p.inner + c];
    float acc = bias + w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
    u2[(row0 + t) * p.inner + c] = fast_silu(acc);
    s0 = s1;
    s1 = s2;
    s2 = x;
  }

  if (p.has_state != 0u) {
    conv_state[st_base + 0u * p.inner] = s0;
    conv_state[st_base + 1u * p.inner] = s1;
    conv_state[st_base + 2u * p.inner] = s2;
  }
}

struct MambaScanParams {
  uint tasks;
  uint len;
  uint inner;
  uint state;
  uint dt_rank;
  uint dt_total;
  uint has_state;
  uint task_offset;
};

// Register-resident exact-fp32 selective scan (state = 16).
// 1 thread per (task, channel).
kernel void mamba_scan_f32(
    device float* ssm_state [[buffer(0)]],
    device const float* a_neg [[buffer(1)]],
    device const float* d_skip [[buffer(2)]],
    device const float* u [[buffer(3)]],
    device const float* delta [[buffer(4)]],
    device const float* xz [[buffer(5)]],
    device const float* xp [[buffer(6)]],
    device float* y [[buffer(7)]],
    constant MambaScanParams& p [[buffer(8)]],
    uint idx [[thread_position_in_grid]]) {
  uint total = p.tasks * p.inner;
  if (idx >= total) return;
  uint c = idx % p.inner;
  uint task = idx / p.inner;
  uint global_task = p.task_offset + task;

  float h[16];
  float a[16];
  size_t st_base = size_t(task) * p.state * p.inner + c;
  for (uint n = 0; n < 16u; ++n) {
    a[n] = a_neg[size_t(n) * p.inner + c];
    h[n] = (p.has_state != 0u) ? ssm_state[st_base + size_t(n) * p.inner] : 0.0f;
  }
  float d_val = d_skip[c];

  size_t row0 = size_t(global_task) * p.len;
  for (uint t = 0; t < p.len; ++t) {
    size_t r = row0 + t;
    float uc = u[r * p.inner + c];
    float dc = delta[r * p.inner + c];
    float du = dc * uc;
    device const float* bptr = xp + r * p.dt_total + p.dt_rank;
    device const float* cptr = bptr + p.state;
    float acc = d_val * uc;
    for (uint n = 0; n < 16u; ++n) {
      float hn = fma(exp(a[n] * dc), h[n], du * bptr[n]);
      h[n] = hn;
      acc = fma(hn, cptr[n], acc);
    }
    float zc = xz[r * 2u * p.inner + p.inner + c];
    y[r * p.inner + c] = acc * fast_silu(zc);
  }

  if (p.has_state != 0u) {
    for (uint n = 0; n < 16u; ++n) {
      ssm_state[st_base + size_t(n) * p.inner] = h[n];
    }
  }
}
)MSL";

struct MetalAttnState {
  id<MTLBuffer> k = nil;
  id<MTLBuffer> v = nil;
  int capacity = 0;
  int end = 0;
  bool is_f16 = false;
};

struct MetalMambaState {
  id<MTLBuffer> conv = nil;
  id<MTLBuffer> ssm = nil;
};

struct MetalLaneStateImpl : public MetalLaneState {
  std::vector<MetalAttnState> attn;
  std::vector<MetalMambaState> mamba;
};

struct MetalMambaLayerBufs {
  id<MTLBuffer> in_w = nil;
  id<MTLBuffer> conv_t = nil;
  id<MTLBuffer> conv_b = nil;
  id<MTLBuffer> x_w = nil;
  id<MTLBuffer> dt_w = nil;
  id<MTLBuffer> dt_b = nil;
  id<MTLBuffer> d = nil;
  id<MTLBuffer> a = nil;
  id<MTLBuffer> out_w = nil;
};

struct MetalBlockBufs {
  bool transformer = false;
  // Transformer
  id<MTLBuffer> n1w = nil, n1b = nil, n2w = nil, n2b = nil;
  id<MTLBuffer> qkv_w = nil, qkv_b = nil;
  id<MTLBuffer> proj_w = nil, proj_b = nil;
  id<MTLBuffer> f0w = nil, f0b = nil, f1w = nil, f1b = nil;
  id<MTLBuffer> bn1w = nil, bn1b = nil, bn2w = nil, bn2b = nil;
  id<MTLBuffer> in_w = nil, in_b = nil;
  id<MTLBuffer> out_w = nil, out_b = nil;
  id<MTLBuffer> bl1w = nil, bl1b = nil, bl2w = nil, bl2b = nil;
  // Mamba
  MetalMambaLayerBufs time_fwd;
  id<MTLBuffer> tm_out_w = nil, tm_out_b = nil;
  id<MTLBuffer> tm_norm_w = nil, tm_norm_b = nil;
  MetalMambaLayerBufs freq_fwd;
  MetalMambaLayerBufs freq_bwd;
  id<MTLBuffer> fm_out_w = nil, fm_out_b = nil;
  id<MTLBuffer> fm_norm_w = nil, fm_norm_b = nil;
};

struct MetalScratch {
  int rows = 0;
  id<MTLBuffer> feat = nil;
  id<MTLBuffer> norm = nil;
  id<MTLBuffer> qkv = nil;
  id<MTLBuffer> attn = nil;
  id<MTLBuffer> ffn = nil;
  id<MTLBuffer> trans = nil;
  id<MTLBuffer> rev = nil;
  id<MTLBuffer> bwd = nil;
  id<MTLBuffer> cat = nil;
  id<MTLBuffer> xz = nil;
  id<MTLBuffer> u2 = nil;
  id<MTLBuffer> y = nil;
  id<MTLBuffer> delta = nil;
  id<MTLBuffer> xp = nil;
  id<MTLBuffer> dummy = nil;  // 16-byte dummy buffer for uncarried state args
  id<MTLBuffer> act16 = nil;  // f16 activation scratch for fp16 MPS GEMM
};

struct RopeStoreParamsHost {
  uint32_t bands, frames, heads, hd, first, capacity, row_offset;
};
struct TimeAttnParamsHost {
  uint32_t bands, frames, heads, hd, first, capacity, past_lo, window, row_offset;
  float scale;
};
struct BandAttnParamsHost {
  uint32_t frames_total, bands, heads, hd;
  float scale;
};
struct MambaConvParamsHost {
  uint32_t tasks, len, inner, has_state, task_offset;
};
struct MambaScanParamsHost {
  uint32_t tasks, len, inner, state, dt_rank, dt_total, has_state, task_offset;
};

class MetalContextImpl final : public MetalContext {
 public:
  MetalContextImpl(const Config& cfg, const std::vector<MetalBlockDesc>& blocks,
                   bool fp16_w, bool fp16_kv)
      : cfg_(cfg), descs_(blocks), fp16_weights_(fp16_w), fp16_kv_(fp16_kv) {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) {
      throw std::runtime_error("Metal device is unavailable on this machine");
    }
    queue_ = [device_ newCommandQueue];
    if (queue_ == nil) {
      throw std::runtime_error("Failed to create Metal command queue");
    }
    compile_pipelines();
    upload_weights();
    scratch_.dummy =
        [device_ newBufferWithLength:64 options:MTLResourceStorageModeShared];
  }

  std::string device_name() const override {
    return std::string([[device_ name] UTF8String]);
  }
  bool fp16_weights() const override { return fp16_weights_; }
  bool fp16_kv() const override { return fp16_kv_; }

  void set_precision(bool fp16_w, bool fp16_kv) override {
    fp16_kv_ = fp16_kv;
    if (fp16_w != fp16_weights_) {
      fp16_weights_ = fp16_w;
      mps_cache_.clear();
      upload_weights();
    }
  }

  std::shared_ptr<MetalLaneState> init_lane() const override {
    auto st = std::make_shared<MetalLaneStateImpl>();
    st->attn.resize(static_cast<size_t>(cfg_.num_blocks));
    st->mamba.resize(static_cast<size_t>(cfg_.num_blocks));
    const size_t conv_bytes = static_cast<size_t>(cfg_.bands()) *
                              (cfg_.mamba_conv - 1) * cfg_.inner() *
                              sizeof(float);
    const size_t ssm_bytes = static_cast<size_t>(cfg_.bands()) *
                             cfg_.mamba_state * cfg_.inner() * sizeof(float);
    for (int i = 0; i < cfg_.num_blocks; ++i) {
      if (!blocks_[static_cast<size_t>(i)].transformer) {
        st->mamba[static_cast<size_t>(i)].conv =
            [device_ newBufferWithLength:conv_bytes
                                 options:MTLResourceStorageModeShared];
        st->mamba[static_cast<size_t>(i)].ssm =
            [device_ newBufferWithLength:ssm_bytes
                                 options:MTLResourceStorageModeShared];
        std::memset([st->mamba[static_cast<size_t>(i)].conv contents], 0,
                    conv_bytes);
        std::memset([st->mamba[static_cast<size_t>(i)].ssm contents], 0,
                    ssm_bytes);
      }
    }
    return st;
  }

  std::shared_ptr<MetalLaneState> clone_lane(
      const std::shared_ptr<MetalLaneState>& src_base) const override {
    auto dst = std::static_pointer_cast<MetalLaneStateImpl>(init_lane());
    auto src = std::static_pointer_cast<MetalLaneStateImpl>(src_base);
    if (!src) return dst;
    for (int i = 0; i < cfg_.num_blocks; ++i) {
      if (!blocks_[static_cast<size_t>(i)].transformer) {
        id<MTLBuffer> sc = src->mamba[static_cast<size_t>(i)].conv;
        id<MTLBuffer> dc = dst->mamba[static_cast<size_t>(i)].conv;
        id<MTLBuffer> ss = src->mamba[static_cast<size_t>(i)].ssm;
        id<MTLBuffer> ds = dst->mamba[static_cast<size_t>(i)].ssm;
        if (sc != nil && dc != nil)
          std::memcpy([dc contents], [sc contents], [dc length]);
        if (ss != nil && ds != nil)
          std::memcpy([ds contents], [ss contents], [ds length]);
      }
    }
    return dst;
  }

  void backbone(
      float* feat, int frames, std::shared_ptr<MetalLaneState>& lane_base,
      const Rope& rope, const LaneParams& lp,
      const std::function<void(int, int, const float*, const float*)>&
          debug_block) const override {
    @autoreleasepool {
      if (!lane_base) lane_base = init_lane();
      auto lane = std::static_pointer_cast<MetalLaneStateImpl>(lane_base);
      const int bands = cfg_.bands();
      const int width = cfg_.width;
      const int rows = bands * frames;
      ensure_scratch(rows);
      ensure_rope(rope);

      const size_t feat_bytes = static_cast<size_t>(rows) * width * sizeof(float);
      std::memcpy([scratch_.feat contents], feat, feat_bytes);

      std::vector<float> debug_in;
      if (debug_block) debug_in.resize(static_cast<size_t>(rows) * width);

      auto t0 = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> cmd = debug_block ? nil : [queue_ commandBuffer];

      for (int i = 0; i < cfg_.num_blocks; ++i) {
        if (debug_block) {
          std::memcpy(debug_in.data(), [scratch_.feat contents], feat_bytes);
          cmd = [queue_ commandBuffer];
        }
        const MetalBlockBufs& blk = blocks_[static_cast<size_t>(i)];
        if (blk.transformer) {
          encode_time_transformer(cmd, blk, scratch_.feat, frames,
                                  lane->attn[static_cast<size_t>(i)], lp);
          encode_band_transformer(cmd, blk, scratch_.feat, frames, /*lanes=*/1);
        } else {
          encode_mamba_block(cmd, blk, scratch_.feat, frames,
                             lane->mamba[static_cast<size_t>(i)]);
        }
        if (debug_block) {
          [cmd commit];
          [cmd waitUntilCompleted];
          std::memcpy(feat, [scratch_.feat contents], feat_bytes);
          debug_block(i, frames, debug_in.data(), feat);
        }
      }

      if (!debug_block) {
        [cmd commit];
        [cmd waitUntilCompleted];
        std::memcpy(feat, [scratch_.feat contents], feat_bytes);
      }
      if (Model::profile_enabled()) {
        const double dt = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        Model::profile().time_attn += dt;  // total GPU backbone wall time
      }
    }
  }

  void backbone_pair(float* feat_a, float* feat_b, int frames,
                     std::shared_ptr<MetalLaneState>& lane_a_base,
                     std::shared_ptr<MetalLaneState>& lane_b_base,
                     const Rope& rope, const LaneParams& lp) const override {
    @autoreleasepool {
      if (!lane_a_base) lane_a_base = init_lane();
      if (!lane_b_base) lane_b_base = init_lane();
      auto lane_a = std::static_pointer_cast<MetalLaneStateImpl>(lane_a_base);
      auto lane_b = std::static_pointer_cast<MetalLaneStateImpl>(lane_b_base);
      const int bands = cfg_.bands();
      const int width = cfg_.width;
      const int rows = bands * frames;
      const int rows2 = 2 * rows;
      ensure_scratch(rows2);
      ensure_rope(rope);

      const size_t lane_bytes = static_cast<size_t>(rows) * width * sizeof(float);
      float* dst = reinterpret_cast<float*>([scratch_.feat contents]);
      std::memcpy(dst, feat_a, lane_bytes);
      std::memcpy(dst + static_cast<size_t>(rows) * width, feat_b, lane_bytes);

      auto t0 = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> cmd = [queue_ commandBuffer];

      for (int i = 0; i < cfg_.num_blocks; ++i) {
        const MetalBlockBufs& blk = blocks_[static_cast<size_t>(i)];
        if (blk.transformer) {
          encode_time_transformer_pair(cmd, blk, scratch_.feat, frames,
                                       lane_a->attn[static_cast<size_t>(i)],
                                       lane_b->attn[static_cast<size_t>(i)], lp);
          encode_band_transformer(cmd, blk, scratch_.feat, frames, /*lanes=*/2);
        } else {
          encode_mamba_block_pair(cmd, blk, scratch_.feat, frames,
                                  lane_a->mamba[static_cast<size_t>(i)],
                                  lane_b->mamba[static_cast<size_t>(i)]);
        }
      }

      [cmd commit];
      [cmd waitUntilCompleted];
      std::memcpy(feat_a, dst, lane_bytes);
      std::memcpy(feat_b, dst + static_cast<size_t>(rows) * width, lane_bytes);

      if (Model::profile_enabled()) {
        const double dt = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        Model::profile().time_attn += dt;
      }
    }
  }

 private:
  void compile_pipelines() {
    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
      opts.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    id<MTLLibrary> lib = [device_ newLibraryWithSource:kMetalShaderSource
                                               options:opts
                                                 error:&err];
    if (lib == nil) {
      throw std::runtime_error(std::string("Metal shader compile failed: ") +
                               [[err localizedDescription] UTF8String]);
    }
    auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> fn = [lib newFunctionWithName:name];
      if (fn == nil) {
        throw std::runtime_error(std::string("Missing Metal kernel: ") +
                                 [name UTF8String]);
      }
      NSError* perr = nil;
      id<MTLComputePipelineState> pso =
          [device_ newComputePipelineStateWithFunction:fn error:&perr];
      if (pso == nil) {
        throw std::runtime_error(std::string("Metal PSO failed: ") +
                                 [[perr localizedDescription] UTF8String]);
      }
      return pso;
    };
    pso_ln_ = make_pso(@"layer_norm_f32");
    pso_add_bias_ = make_pso(@"add_bias_f32");
    pso_add_bias_gelu_ = make_pso(@"add_bias_gelu_f32");
    pso_add_bias_res_ = make_pso(@"add_bias_residual_f32");
    pso_add_vec_ = make_pso(@"add_vec_f32");
    pso_add_bias_sp_ = make_pso(@"add_bias_softplus_f32");
    pso_f32_to_f16_ = make_pso(@"f32_to_f16");
    pso_f32_to_f16_strided_ = make_pso(@"f32_to_f16_strided");
    pso_to_frame_ = make_pso(@"to_frame_major_f32");
    pso_to_band_ = make_pso(@"to_band_major_f32");
    pso_rev_bands_ = make_pso(@"reverse_bands_f32");
    pso_mamba_cat_ = make_pso(@"mamba_freq_cat_f32");
    pso_compact_f32_ = make_pso(@"compact_kv_f32");
    pso_compact_f16_ = make_pso(@"compact_kv_f16");
    pso_rope_store_f32_ = make_pso(@"rope_store_kv_f32");
    pso_rope_store_f16_ = make_pso(@"rope_store_kv_f16");
    pso_time_attn_f32_ = make_pso(@"time_attn_f32");
    pso_time_attn_f16_ = make_pso(@"time_attn_f16kv");
    pso_band_attn_f32_ = make_pso(@"band_attn_f32");
    pso_mamba_conv_ = make_pso(@"mamba_conv4_silu_f32");
    pso_mamba_scan_ = make_pso(@"mamba_scan_f32");
  }

  id<MTLBuffer> make_f32_buf(const float* data, size_t count) const {
    return [device_ newBufferWithBytes:data
                                length:count * sizeof(float)
                               options:MTLResourceStorageModeShared];
  }

  id<MTLBuffer> make_weight_buf(const float* data, size_t count) const {
    if (!fp16_weights_) return make_f32_buf(data, count);
    id<MTLBuffer> buf =
        [device_ newBufferWithLength:count * sizeof(__fp16)
                             options:MTLResourceStorageModeShared];
    __fp16* dst = reinterpret_cast<__fp16*>([buf contents]);
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<__fp16>(data[i]);
    return buf;
  }

  void upload_weights() {
    const int width = cfg_.width;
    const int inner = cfg_.inner();
    const int state = cfg_.mamba_state;
    const int conv = cfg_.mamba_conv;
    const int dt_rank = cfg_.dt_rank();
    const int dt_total = dt_rank + 2 * state;

    blocks_.clear();
    blocks_.resize(descs_.size());
    auto load_mamba = [&](const MetalMambaLayerDesc& d) {
      MetalMambaLayerBufs b;
      b.in_w = make_weight_buf(d.in_w, static_cast<size_t>(2 * inner) * width);
      b.conv_t = make_f32_buf(d.conv_t, static_cast<size_t>(conv) * inner);
      b.conv_b = make_f32_buf(d.conv_b, static_cast<size_t>(inner));
      b.x_w = make_weight_buf(d.x_w, static_cast<size_t>(dt_total) * inner);
      b.dt_w = make_weight_buf(d.dt_w, static_cast<size_t>(inner) * dt_rank);
      b.dt_b = make_f32_buf(d.dt_b, static_cast<size_t>(inner));
      b.d = make_f32_buf(d.d, static_cast<size_t>(inner));
      b.a = make_f32_buf(d.a, static_cast<size_t>(state) * inner);
      b.out_w = make_weight_buf(d.out_w, static_cast<size_t>(width) * inner);
      return b;
    };

    for (size_t i = 0; i < descs_.size(); ++i) {
      const MetalBlockDesc& d = descs_[i];
      MetalBlockBufs& b = blocks_[i];
      b.transformer = d.transformer;
      if (d.transformer) {
        b.n1w = make_f32_buf(d.n1w, width);
        b.n1b = make_f32_buf(d.n1b, width);
        b.n2w = make_f32_buf(d.n2w, width);
        b.n2b = make_f32_buf(d.n2b, width);
        b.qkv_w = make_weight_buf(d.qkv_w, static_cast<size_t>(3 * width) * width);
        b.qkv_b = make_f32_buf(d.qkv_b, 3 * width);
        b.proj_w = make_weight_buf(d.proj_w, static_cast<size_t>(width) * width);
        b.proj_b = make_f32_buf(d.proj_b, width);
        b.f0w = make_weight_buf(d.f0w, static_cast<size_t>(4 * width) * width);
        b.f0b = make_f32_buf(d.f0b, 4 * width);
        b.f1w = make_weight_buf(d.f1w, static_cast<size_t>(width) * 4 * width);
        b.f1b = make_f32_buf(d.f1b, width);
        b.bn1w = make_f32_buf(d.bn1w, width);
        b.bn1b = make_f32_buf(d.bn1b, width);
        b.bn2w = make_f32_buf(d.bn2w, width);
        b.bn2b = make_f32_buf(d.bn2b, width);
        b.in_w = make_weight_buf(d.in_w, static_cast<size_t>(3 * width) * width);
        b.in_b = make_f32_buf(d.in_b, 3 * width);
        b.out_w = make_weight_buf(d.out_w, static_cast<size_t>(width) * width);
        b.out_b = make_f32_buf(d.out_b, width);
        b.bl1w = make_weight_buf(d.bl1w, static_cast<size_t>(4 * width) * width);
        b.bl1b = make_f32_buf(d.bl1b, 4 * width);
        b.bl2w = make_weight_buf(d.bl2w, static_cast<size_t>(width) * 4 * width);
        b.bl2b = make_f32_buf(d.bl2b, width);
      } else {
        b.time_fwd = load_mamba(d.time_fwd);
        b.tm_out_w = make_weight_buf(d.tm_out_w, static_cast<size_t>(width) * width);
        b.tm_out_b = make_f32_buf(d.tm_out_b, width);
        b.tm_norm_w = make_f32_buf(d.tm_norm_w, width);
        b.tm_norm_b = make_f32_buf(d.tm_norm_b, width);
        b.freq_fwd = load_mamba(d.freq_fwd);
        b.freq_bwd = load_mamba(d.freq_bwd);
        b.fm_out_w = make_weight_buf(d.fm_out_w, static_cast<size_t>(width) * 2 * width);
        b.fm_out_b = make_f32_buf(d.fm_out_b, width);
        b.fm_norm_w = make_f32_buf(d.fm_norm_w, width);
        b.fm_norm_b = make_f32_buf(d.fm_norm_b, width);
      }
    }
  }

  void ensure_scratch(int rows) const {
    if (scratch_.rows >= rows) return;
    const int width = cfg_.width;
    const int inner = cfg_.inner();
    const int dt_total = cfg_.dt_rank() + 2 * cfg_.mamba_state;
    auto alloc = [&](size_t elems) -> id<MTLBuffer> {
      return [device_ newBufferWithLength:elems * sizeof(float)
                                  options:MTLResourceStorageModeShared];
    };
    scratch_.rows = rows;
    scratch_.feat = alloc(static_cast<size_t>(rows) * width);
    scratch_.norm = alloc(static_cast<size_t>(rows) * width);
    scratch_.qkv = alloc(static_cast<size_t>(rows) * 3 * width);
    scratch_.attn = alloc(static_cast<size_t>(rows) * width);
    scratch_.ffn = alloc(static_cast<size_t>(rows) * 4 * width);
    scratch_.trans = alloc(static_cast<size_t>(rows) * width);
    scratch_.rev = alloc(static_cast<size_t>(rows) * width);
    scratch_.bwd = alloc(static_cast<size_t>(rows) * width);
    scratch_.cat = alloc(static_cast<size_t>(rows) * 2 * width);
    scratch_.xz = alloc(static_cast<size_t>(rows) * 2 * inner);
    scratch_.u2 = alloc(static_cast<size_t>(rows) * inner);
    scratch_.y = alloc(static_cast<size_t>(rows) * inner);
    scratch_.delta = alloc(static_cast<size_t>(rows) * inner);
    scratch_.xp = alloc(static_cast<size_t>(rows) * dt_total);
    scratch_.act16 = [device_
        newBufferWithLength:static_cast<size_t>(rows) * 4 * width * sizeof(__fp16)
                    options:MTLResourceStorageModeShared];
  }

  void ensure_rope(const Rope& rope) const {
    if (rope_cos_ != nil && rope_ptr_ == rope.cos.data() &&
        rope_len_ == rope.cos.size()) {
      return;
    }
    rope_cos_ = make_f32_buf(rope.cos.data(), rope.cos.size());
    rope_sin_ = make_f32_buf(rope.sin.data(), rope.sin.size());
    rope_ptr_ = rope.cos.data();
    rope_len_ = rope.cos.size();
  }

  void reserve_kv(MetalAttnState& st, int slots, int limit) const {
    if (st.capacity >= slots && st.is_f16 == fp16_kv_) return;
    int cap = std::max(slots, std::max(2 * st.capacity, 64));
    if (cap > limit) cap = limit;
    const int bands = cfg_.bands();
    const int width = cfg_.width;
    const int heads = cfg_.heads;
    const int hd = cfg_.head_dim();
    const size_t esz = fp16_kv_ ? sizeof(__fp16) : sizeof(float);
    const size_t bytes = static_cast<size_t>(bands) * cap * width * esz;
    id<MTLBuffer> nk =
        [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> nv =
        [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (st.end > 0 && st.k != nil && st.is_f16 == fp16_kv_) {
      const size_t old_band = static_cast<size_t>(st.capacity) * width;
      const size_t new_band = static_cast<size_t>(cap) * width;
      const size_t run = static_cast<size_t>(st.end) * hd * esz;
      const uint8_t* okp = static_cast<const uint8_t*>([st.k contents]);
      const uint8_t* ovp = static_cast<const uint8_t*>([st.v contents]);
      uint8_t* nkp = static_cast<uint8_t*>([nk contents]);
      uint8_t* nvp = static_cast<uint8_t*>([nv contents]);
      for (int b = 0; b < bands; ++b) {
        for (int h = 0; h < heads; ++h) {
          const size_t oa =
              (static_cast<size_t>(b) * old_band +
               static_cast<size_t>(h) * st.capacity * hd) *
              esz;
          const size_t na = (static_cast<size_t>(b) * new_band +
                             static_cast<size_t>(h) * cap * hd) *
                            esz;
          std::memcpy(nkp + na, okp + oa, run);
          std::memcpy(nvp + na, ovp + oa, run);
        }
      }
    }
    st.k = nk;
    st.v = nv;
    st.capacity = cap;
    st.is_f16 = fp16_kv_;
  }

  void dispatch_1d(id<MTLComputeCommandEncoder> enc,
                   id<MTLComputePipelineState> pso, size_t total,
                   uint32_t tg = 256) const {
    [enc setComputePipelineState:pso];
    const NSUInteger groups = (total + tg - 1) / tg;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  }

  void encode_gemm(id<MTLCommandBuffer> cmd, id<MTLBuffer> a, int m, int k,
                   int stride, id<MTLBuffer> w, int n, id<MTLBuffer> c) const {
    const uint64_t key = (static_cast<uint64_t>(m) << 32) |
                         (static_cast<uint64_t>(n) << 16) |
                         static_cast<uint64_t>(k);
    MPSMatrixMultiplication* mm = nil;
    auto it = mps_cache_.find(key);
    if (it != mps_cache_.end()) {
      mm = it->second;
    } else {
      mm = [[MPSMatrixMultiplication alloc] initWithDevice:device_
                                             transposeLeft:NO
                                            transposeRight:YES
                                                resultRows:m
                                             resultColumns:n
                                           interiorColumns:k
                                                     alpha:1.0
                                                      beta:0.0];
      mps_cache_[key] = mm;
    }

    MPSMatrix* matA = nil;
    MPSMatrix* matB = nil;
    if (fp16_weights_) {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      if (stride == k) {
        uint32_t total = static_cast<uint32_t>(m * k);
        [enc setBuffer:a offset:0 atIndex:0];
        [enc setBuffer:scratch_.act16 offset:0 atIndex:1];
        [enc setBytes:&total length:4 atIndex:2];
        dispatch_1d(enc, pso_f32_to_f16_, total);
      } else {
        uint32_t um = m, uk = k, ustr = stride;
        [enc setBuffer:a offset:0 atIndex:0];
        [enc setBuffer:scratch_.act16 offset:0 atIndex:1];
        [enc setBytes:&um length:4 atIndex:2];
        [enc setBytes:&uk length:4 atIndex:3];
        [enc setBytes:&ustr length:4 atIndex:4];
        dispatch_1d(enc, pso_f32_to_f16_strided_, static_cast<size_t>(m) * k);
      }
      [enc endEncoding];

      MPSMatrixDescriptor* dA =
          [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                columns:k
                                               rowBytes:k * sizeof(__fp16)
                                               dataType:MPSDataTypeFloat16];
      MPSMatrixDescriptor* dB =
          [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                columns:k
                                               rowBytes:k * sizeof(__fp16)
                                               dataType:MPSDataTypeFloat16];
      matA = [[MPSMatrix alloc] initWithBuffer:scratch_.act16 descriptor:dA];
      matB = [[MPSMatrix alloc] initWithBuffer:w descriptor:dB];
    } else {
      MPSMatrixDescriptor* dA =
          [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                columns:k
                                               rowBytes:stride * sizeof(float)
                                               dataType:MPSDataTypeFloat32];
      MPSMatrixDescriptor* dB =
          [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                columns:k
                                               rowBytes:k * sizeof(float)
                                               dataType:MPSDataTypeFloat32];
      matA = [[MPSMatrix alloc] initWithBuffer:a descriptor:dA];
      matB = [[MPSMatrix alloc] initWithBuffer:w descriptor:dB];
    }

    MPSMatrixDescriptor* dC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                              columns:n
                                             rowBytes:n * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:c descriptor:dC];
    [mm encodeToCommandBuffer:cmd
                   leftMatrix:matA
                  rightMatrix:matB
                 resultMatrix:matC];
  }

  void encode_ln(id<MTLCommandBuffer> cmd, id<MTLBuffer> x, id<MTLBuffer> y,
                 id<MTLBuffer> w, id<MTLBuffer> b, int rows, int cols) const {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t ucols = static_cast<uint32_t>(cols);
    [enc setComputePipelineState:pso_ln_];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:w offset:0 atIndex:2];
    [enc setBuffer:b offset:0 atIndex:3];
    [enc setBytes:&ucols length:4 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
  }

  void encode_add_bias(id<MTLCommandBuffer> cmd, id<MTLBuffer> x,
                       id<MTLBuffer> b, int rows, int cols) const {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t ucols = cols, utotal = rows * cols;
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBytes:&ucols length:4 atIndex:2];
    [enc setBytes:&utotal length:4 atIndex:3];
    dispatch_1d(enc, pso_add_bias_, utotal);
    [enc endEncoding];
  }

  void encode_add_bias_gelu(id<MTLCommandBuffer> cmd, id<MTLBuffer> x,
                            id<MTLBuffer> b, int rows, int cols) const {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t ucols = cols, utotal = rows * cols;
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBytes:&ucols length:4 atIndex:2];
    [enc setBytes:&utotal length:4 atIndex:3];
    dispatch_1d(enc, pso_add_bias_gelu_, utotal);
    [enc endEncoding];
  }

  void encode_add_bias_res(id<MTLCommandBuffer> cmd, id<MTLBuffer> dst,
                           id<MTLBuffer> src, id<MTLBuffer> b, int rows,
                           int cols) const {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t ucols = cols, utotal = rows * cols;
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:src offset:0 atIndex:1];
    [enc setBuffer:b offset:0 atIndex:2];
    [enc setBytes:&ucols length:4 atIndex:3];
    [enc setBytes:&utotal length:4 atIndex:4];
    dispatch_1d(enc, pso_add_bias_res_, utotal);
    [enc endEncoding];
  }

  void encode_add_vec(id<MTLCommandBuffer> cmd, id<MTLBuffer> dst,
                      id<MTLBuffer> src, int total) const {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint32_t utotal = total;
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:src offset:0 atIndex:1];
    [enc setBytes:&utotal length:4 atIndex:2];
    dispatch_1d(enc, pso_add_vec_, utotal);
    [enc endEncoding];
  }

  int prep_attn_state(id<MTLCommandBuffer> cmd, MetalAttnState& st, int frames,
                      const LaneParams& lp) const {
    int first = st.end;
    if (first + frames > lp.capacity_limit) {
      const int keep = std::min(first, lp.cache_frames);
      const int shift = first - keep;
      if (shift > 0) {
        const uint32_t heads = cfg_.heads;
        const uint32_t hd = cfg_.head_dim();
        const uint32_t cap = st.capacity;
        const uint32_t ukeep = keep, ushift = shift;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:st.is_f16 ? pso_compact_f16_
                                               : pso_compact_f32_];
        [enc setBuffer:st.k offset:0 atIndex:0];
        [enc setBuffer:st.v offset:0 atIndex:1];
        [enc setBuffer:rope_cos_ offset:0 atIndex:2];
        [enc setBuffer:rope_sin_ offset:0 atIndex:3];
        [enc setBytes:&heads length:4 atIndex:4];
        [enc setBytes:&hd length:4 atIndex:5];
        [enc setBytes:&cap length:4 atIndex:6];
        [enc setBytes:&ukeep length:4 atIndex:7];
        [enc setBytes:&ushift length:4 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(cfg_.bands() * heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(hd / 2, 1, 1)];
        [enc endEncoding];
        st.end = keep;
        first = keep;
      }
    }
    reserve_kv(st, first + frames, lp.capacity_limit);
    return first;
  }

  void encode_rope_and_time_attn(id<MTLCommandBuffer> cmd,
                                 const MetalBlockBufs& blk,
                                 MetalAttnState& st, int frames, int first,
                                 int row_offset, const LaneParams& lp) const {
    const int bands = cfg_.bands();
    const int heads = cfg_.heads;
    const int hd = cfg_.head_dim();
    const int past_lo = std::max(0, first - lp.cache_frames);

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    RopeStoreParamsHost rp{static_cast<uint32_t>(bands),
                           static_cast<uint32_t>(frames),
                           static_cast<uint32_t>(heads),
                           static_cast<uint32_t>(hd),
                           static_cast<uint32_t>(first),
                           static_cast<uint32_t>(st.capacity),
                           static_cast<uint32_t>(row_offset)};
    [enc setBuffer:scratch_.qkv offset:0 atIndex:0];
    [enc setBuffer:blk.qkv_b offset:0 atIndex:1];
    [enc setBuffer:st.k offset:0 atIndex:2];
    [enc setBuffer:st.v offset:0 atIndex:3];
    [enc setBuffer:rope_cos_ offset:0 atIndex:4];
    [enc setBuffer:rope_sin_ offset:0 atIndex:5];
    [enc setBytes:&rp length:sizeof(rp) atIndex:6];
    dispatch_1d(enc, st.is_f16 ? pso_rope_store_f16_ : pso_rope_store_f32_,
                static_cast<size_t>(bands) * frames * heads * (hd / 2));

    TimeAttnParamsHost ap{static_cast<uint32_t>(bands),
                          static_cast<uint32_t>(frames),
                          static_cast<uint32_t>(heads),
                          static_cast<uint32_t>(hd),
                          static_cast<uint32_t>(first),
                          static_cast<uint32_t>(st.capacity),
                          static_cast<uint32_t>(past_lo),
                          static_cast<uint32_t>(lp.window),
                          static_cast<uint32_t>(row_offset),
                          1.0f / std::sqrt(static_cast<float>(hd))};
    [enc setComputePipelineState:st.is_f16 ? pso_time_attn_f16_
                                           : pso_time_attn_f32_];
    [enc setBuffer:scratch_.qkv offset:0 atIndex:0];
    [enc setBuffer:st.k offset:0 atIndex:1];
    [enc setBuffer:st.v offset:0 atIndex:2];
    [enc setBuffer:scratch_.attn offset:0 atIndex:3];
    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(bands * frames * heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    st.end = first + frames;
  }

  void encode_time_transformer(id<MTLCommandBuffer> cmd,
                               const MetalBlockBufs& blk, id<MTLBuffer> feat,
                               int frames, MetalAttnState& st,
                               const LaneParams& lp) const {
    const int width = cfg_.width;
    const int rows = cfg_.bands() * frames;
    const int first = prep_attn_state(cmd, st, frames, lp);

    encode_ln(cmd, feat, scratch_.norm, blk.n1w, blk.n1b, rows, width);
    encode_gemm(cmd, scratch_.norm, rows, width, width, blk.qkv_w, 3 * width,
                scratch_.qkv);
    encode_rope_and_time_attn(cmd, blk, st, frames, first, /*row_offset=*/0, lp);
    encode_gemm(cmd, scratch_.attn, rows, width, width, blk.proj_w, width,
                scratch_.trans);
    encode_add_bias_res(cmd, feat, scratch_.trans, blk.proj_b, rows, width);

    encode_ln(cmd, feat, scratch_.norm, blk.n2w, blk.n2b, rows, width);
    encode_gemm(cmd, scratch_.norm, rows, width, width, blk.f0w, 4 * width,
                scratch_.ffn);
    encode_add_bias_gelu(cmd, scratch_.ffn, blk.f0b, rows, 4 * width);
    encode_gemm(cmd, scratch_.ffn, rows, 4 * width, 4 * width, blk.f1w, width,
                scratch_.trans);
    encode_add_bias_res(cmd, feat, scratch_.trans, blk.f1b, rows, width);
  }

  void encode_time_transformer_pair(id<MTLCommandBuffer> cmd,
                                    const MetalBlockBufs& blk,
                                    id<MTLBuffer> pair_feat, int frames,
                                    MetalAttnState& st_a, MetalAttnState& st_b,
                                    const LaneParams& lp) const {
    const int width = cfg_.width;
    const int rows = cfg_.bands() * frames;
    const int rows2 = 2 * rows;
    const int first_a = prep_attn_state(cmd, st_a, frames, lp);
    const int first_b = prep_attn_state(cmd, st_b, frames, lp);

    encode_ln(cmd, pair_feat, scratch_.norm, blk.n1w, blk.n1b, rows2, width);
    encode_gemm(cmd, scratch_.norm, rows2, width, width, blk.qkv_w, 3 * width,
                scratch_.qkv);
    encode_rope_and_time_attn(cmd, blk, st_a, frames, first_a, 0, lp);
    encode_rope_and_time_attn(cmd, blk, st_b, frames, first_b, rows, lp);
    encode_gemm(cmd, scratch_.attn, rows2, width, width, blk.proj_w, width,
                scratch_.trans);
    encode_add_bias_res(cmd, pair_feat, scratch_.trans, blk.proj_b, rows2, width);

    encode_ln(cmd, pair_feat, scratch_.norm, blk.n2w, blk.n2b, rows2, width);
    encode_gemm(cmd, scratch_.norm, rows2, width, width, blk.f0w, 4 * width,
                scratch_.ffn);
    encode_add_bias_gelu(cmd, scratch_.ffn, blk.f0b, rows2, 4 * width);
    encode_gemm(cmd, scratch_.ffn, rows2, 4 * width, 4 * width, blk.f1w, width,
                scratch_.trans);
    encode_add_bias_res(cmd, pair_feat, scratch_.trans, blk.f1b, rows2, width);
  }

  void encode_transpose(id<MTLCommandBuffer> cmd, id<MTLBuffer> src,
                        id<MTLBuffer> dst, int frames, int lanes,
                        bool to_frame) const {
    const uint32_t bands = cfg_.bands(), uframes = frames, width = cfg_.width,
                   ulanes = lanes;
    const size_t total = static_cast<size_t>(lanes) * bands * frames * width;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&bands length:4 atIndex:2];
    [enc setBytes:&uframes length:4 atIndex:3];
    [enc setBytes:&width length:4 atIndex:4];
    [enc setBytes:&ulanes length:4 atIndex:5];
    dispatch_1d(enc, to_frame ? pso_to_frame_ : pso_to_band_, total);
    [enc endEncoding];
  }

  void encode_band_transformer(id<MTLCommandBuffer> cmd,
                               const MetalBlockBufs& blk, id<MTLBuffer> feat,
                               int frames, int lanes) const {
    const int bands = cfg_.bands();
    const int width = cfg_.width;
    const int heads = cfg_.heads;
    const int hd = cfg_.head_dim();
    const int frames_total = frames * lanes;
    const int rows = bands * frames_total;

    encode_transpose(cmd, feat, scratch_.trans, frames, lanes, true);
    encode_ln(cmd, scratch_.trans, scratch_.norm, blk.bn1w, blk.bn1b, rows, width);
    encode_gemm(cmd, scratch_.norm, rows, width, width, blk.in_w, 3 * width,
                scratch_.qkv);
    encode_add_bias(cmd, scratch_.qkv, blk.in_b, rows, 3 * width);

    {
      BandAttnParamsHost bp{static_cast<uint32_t>(frames_total),
                            static_cast<uint32_t>(bands),
                            static_cast<uint32_t>(heads),
                            static_cast<uint32_t>(hd),
                            1.0f / std::sqrt(static_cast<float>(hd))};
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pso_band_attn_f32_];
      [enc setBuffer:scratch_.qkv offset:0 atIndex:0];
      [enc setBuffer:scratch_.attn offset:0 atIndex:1];
      [enc setBytes:&bp length:sizeof(bp) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(frames_total * bands * heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc endEncoding];
    }

    encode_gemm(cmd, scratch_.attn, rows, width, width, blk.out_w, width,
                scratch_.norm);
    encode_add_bias_res(cmd, scratch_.trans, scratch_.norm, blk.out_b, rows, width);

    encode_ln(cmd, scratch_.trans, scratch_.norm, blk.bn2w, blk.bn2b, rows, width);
    encode_gemm(cmd, scratch_.norm, rows, width, width, blk.bl1w, 4 * width,
                scratch_.ffn);
    encode_add_bias_gelu(cmd, scratch_.ffn, blk.bl1b, rows, 4 * width);
    encode_gemm(cmd, scratch_.ffn, rows, 4 * width, 4 * width, blk.bl2w, width,
                scratch_.norm);
    encode_add_bias_res(cmd, scratch_.trans, scratch_.norm, blk.bl2b, rows, width);
    encode_transpose(cmd, scratch_.trans, feat, frames, lanes, false);
  }

  void encode_mamba_layer(id<MTLCommandBuffer> cmd,
                          const MetalMambaLayerBufs& l, id<MTLBuffer> x,
                          int rows, int len, id<MTLBuffer> out,
                          id<MTLBuffer> conv_a, id<MTLBuffer> ssm_a,
                          id<MTLBuffer> conv_b, id<MTLBuffer> ssm_b) const {
    const int width = cfg_.width;
    const int inner = cfg_.inner();
    const int state = cfg_.mamba_state;
    const int dt_rank = cfg_.dt_rank();
    const int dt_total = dt_rank + 2 * state;
    const int tasks_total = rows / len;
    const bool paired = (conv_b != nil);
    const int tasks_each = paired ? (tasks_total / 2) : tasks_total;

    encode_gemm(cmd, x, rows, width, width, l.in_w, 2 * inner, scratch_.xz);

    auto run_conv = [&](id<MTLBuffer> cbuf, uint32_t tasks, uint32_t toff) {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      MambaConvParamsHost cp{tasks, static_cast<uint32_t>(len),
                             static_cast<uint32_t>(inner),
                             cbuf != nil ? 1u : 0u, toff};
      [enc setBuffer:scratch_.xz offset:0 atIndex:0];
      [enc setBuffer:l.conv_t offset:0 atIndex:1];
      [enc setBuffer:l.conv_b offset:0 atIndex:2];
      [enc setBuffer:(cbuf != nil ? cbuf : scratch_.dummy) offset:0 atIndex:3];
      [enc setBuffer:scratch_.u2 offset:0 atIndex:4];
      [enc setBytes:&cp length:sizeof(cp) atIndex:5];
      dispatch_1d(enc, pso_mamba_conv_, static_cast<size_t>(tasks) * inner);
      [enc endEncoding];
    };
    run_conv(conv_a, tasks_each, 0);
    if (paired) run_conv(conv_b, tasks_each, tasks_each);

    encode_gemm(cmd, scratch_.u2, rows, inner, inner, l.x_w, dt_total,
                scratch_.xp);
    encode_gemm(cmd, scratch_.xp, rows, dt_rank, dt_total, l.dt_w, inner,
                scratch_.delta);
    {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      uint32_t ucols = inner, utotal = rows * inner;
      [enc setBuffer:scratch_.delta offset:0 atIndex:0];
      [enc setBuffer:l.dt_b offset:0 atIndex:1];
      [enc setBytes:&ucols length:4 atIndex:2];
      [enc setBytes:&utotal length:4 atIndex:3];
      dispatch_1d(enc, pso_add_bias_sp_, utotal);
      [enc endEncoding];
    }

    auto run_scan = [&](id<MTLBuffer> sbuf, uint32_t tasks, uint32_t toff) {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      MambaScanParamsHost sp{tasks,
                             static_cast<uint32_t>(len),
                             static_cast<uint32_t>(inner),
                             static_cast<uint32_t>(state),
                             static_cast<uint32_t>(dt_rank),
                             static_cast<uint32_t>(dt_total),
                             sbuf != nil ? 1u : 0u,
                             toff};
      [enc setBuffer:(sbuf != nil ? sbuf : scratch_.dummy) offset:0 atIndex:0];
      [enc setBuffer:l.a offset:0 atIndex:1];
      [enc setBuffer:l.d offset:0 atIndex:2];
      [enc setBuffer:scratch_.u2 offset:0 atIndex:3];
      [enc setBuffer:scratch_.delta offset:0 atIndex:4];
      [enc setBuffer:scratch_.xz offset:0 atIndex:5];
      [enc setBuffer:scratch_.xp offset:0 atIndex:6];
      [enc setBuffer:scratch_.y offset:0 atIndex:7];
      [enc setBytes:&sp length:sizeof(sp) atIndex:8];
      dispatch_1d(enc, pso_mamba_scan_, static_cast<size_t>(tasks) * inner);
      [enc endEncoding];
    };
    run_scan(ssm_a, tasks_each, 0);
    if (paired) run_scan(ssm_b, tasks_each, tasks_each);

    encode_gemm(cmd, scratch_.y, rows, inner, inner, l.out_w, width, out);
  }

  void encode_mamba_freq(id<MTLCommandBuffer> cmd, const MetalBlockBufs& blk,
                         id<MTLBuffer> feat, int frames, int lanes) const {
    const int bands = cfg_.bands();
    const int width = cfg_.width;
    const int frames_total = frames * lanes;
    const int rows = bands * frames_total;

    encode_transpose(cmd, feat, scratch_.trans, frames, lanes, true);
    encode_mamba_layer(cmd, blk.freq_fwd, scratch_.trans, rows, bands,
                       scratch_.attn, nil, nil, nil, nil);
    {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      uint32_t uf = frames_total, ub = bands, uw = width;
      [enc setBuffer:scratch_.trans offset:0 atIndex:0];
      [enc setBuffer:scratch_.rev offset:0 atIndex:1];
      [enc setBytes:&uf length:4 atIndex:2];
      [enc setBytes:&ub length:4 atIndex:3];
      [enc setBytes:&uw length:4 atIndex:4];
      dispatch_1d(enc, pso_rev_bands_, static_cast<size_t>(rows) * width);
      [enc endEncoding];
    }
    encode_mamba_layer(cmd, blk.freq_bwd, scratch_.rev, rows, bands,
                       scratch_.bwd, nil, nil, nil, nil);
    {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      uint32_t uf = frames_total, ub = bands, uw = width;
      [enc setBuffer:scratch_.attn offset:0 atIndex:0];
      [enc setBuffer:scratch_.bwd offset:0 atIndex:1];
      [enc setBuffer:scratch_.trans offset:0 atIndex:2];
      [enc setBuffer:scratch_.cat offset:0 atIndex:3];
      [enc setBytes:&uf length:4 atIndex:4];
      [enc setBytes:&ub length:4 atIndex:5];
      [enc setBytes:&uw length:4 atIndex:6];
      dispatch_1d(enc, pso_mamba_cat_, static_cast<size_t>(rows) * width);
      [enc endEncoding];
    }
    encode_gemm(cmd, scratch_.cat, rows, 2 * width, 2 * width, blk.fm_out_w,
                width, scratch_.norm);
    encode_add_bias(cmd, scratch_.norm, blk.fm_out_b, rows, width);
    encode_ln(cmd, scratch_.norm, scratch_.attn, blk.fm_norm_w, blk.fm_norm_b,
              rows, width);
    encode_add_vec(cmd, scratch_.trans, scratch_.attn, rows * width);
    encode_transpose(cmd, scratch_.trans, feat, frames, lanes, false);
  }

  void encode_mamba_block(id<MTLCommandBuffer> cmd, const MetalBlockBufs& blk,
                          id<MTLBuffer> feat, int frames,
                          MetalMambaState& ms) const {
    const int width = cfg_.width;
    const int rows = cfg_.bands() * frames;
    encode_mamba_layer(cmd, blk.time_fwd, feat, rows, frames, scratch_.trans,
                       ms.conv, ms.ssm, nil, nil);
    encode_add_vec(cmd, scratch_.trans, feat, rows * width);
    encode_gemm(cmd, scratch_.trans, rows, width, width, blk.tm_out_w, width,
                scratch_.rev);
    encode_add_bias(cmd, scratch_.rev, blk.tm_out_b, rows, width);
    encode_ln(cmd, scratch_.rev, scratch_.norm, blk.tm_norm_w, blk.tm_norm_b,
              rows, width);
    encode_add_vec(cmd, feat, scratch_.norm, rows * width);
    encode_mamba_freq(cmd, blk, feat, frames, /*lanes=*/1);
  }

  void encode_mamba_block_pair(id<MTLCommandBuffer> cmd,
                               const MetalBlockBufs& blk,
                               id<MTLBuffer> pair_feat, int frames,
                               MetalMambaState& ms_a,
                               MetalMambaState& ms_b) const {
    const int width = cfg_.width;
    const int rows2 = 2 * cfg_.bands() * frames;
    encode_mamba_layer(cmd, blk.time_fwd, pair_feat, rows2, frames,
                       scratch_.trans, ms_a.conv, ms_a.ssm, ms_b.conv, ms_b.ssm);
    encode_add_vec(cmd, scratch_.trans, pair_feat, rows2 * width);
    encode_gemm(cmd, scratch_.trans, rows2, width, width, blk.tm_out_w, width,
                scratch_.rev);
    encode_add_bias(cmd, scratch_.rev, blk.tm_out_b, rows2, width);
    encode_ln(cmd, scratch_.rev, scratch_.norm, blk.tm_norm_w, blk.tm_norm_b,
              rows2, width);
    encode_add_vec(cmd, pair_feat, scratch_.norm, rows2 * width);
    encode_mamba_freq(cmd, blk, pair_feat, frames, /*lanes=*/2);
  }

  Config cfg_;
  std::vector<MetalBlockDesc> descs_;
  bool fp16_weights_ = false;
  bool fp16_kv_ = false;

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  std::vector<MetalBlockBufs> blocks_;
  mutable MetalScratch scratch_;
  mutable id<MTLBuffer> rope_cos_ = nil;
  mutable id<MTLBuffer> rope_sin_ = nil;
  mutable const float* rope_ptr_ = nullptr;
  mutable size_t rope_len_ = 0;
  mutable std::unordered_map<uint64_t, MPSMatrixMultiplication*> mps_cache_;

  id<MTLComputePipelineState> pso_ln_ = nil;
  id<MTLComputePipelineState> pso_add_bias_ = nil;
  id<MTLComputePipelineState> pso_add_bias_gelu_ = nil;
  id<MTLComputePipelineState> pso_add_bias_res_ = nil;
  id<MTLComputePipelineState> pso_add_vec_ = nil;
  id<MTLComputePipelineState> pso_add_bias_sp_ = nil;
  id<MTLComputePipelineState> pso_f32_to_f16_ = nil;
  id<MTLComputePipelineState> pso_f32_to_f16_strided_ = nil;
  id<MTLComputePipelineState> pso_to_frame_ = nil;
  id<MTLComputePipelineState> pso_to_band_ = nil;
  id<MTLComputePipelineState> pso_rev_bands_ = nil;
  id<MTLComputePipelineState> pso_mamba_cat_ = nil;
  id<MTLComputePipelineState> pso_compact_f32_ = nil;
  id<MTLComputePipelineState> pso_compact_f16_ = nil;
  id<MTLComputePipelineState> pso_rope_store_f32_ = nil;
  id<MTLComputePipelineState> pso_rope_store_f16_ = nil;
  id<MTLComputePipelineState> pso_time_attn_f32_ = nil;
  id<MTLComputePipelineState> pso_time_attn_f16_ = nil;
  id<MTLComputePipelineState> pso_band_attn_f32_ = nil;
  id<MTLComputePipelineState> pso_mamba_conv_ = nil;
  id<MTLComputePipelineState> pso_mamba_scan_ = nil;
};

}  // namespace

bool metal_available() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil;
  }
}

std::unique_ptr<MetalContext> MetalContext::create(
    const Config& cfg, const std::vector<MetalBlockDesc>& blocks,
    bool fp16_weights, bool fp16_kv) {
  return std::make_unique<MetalContextImpl>(cfg, blocks, fp16_weights, fp16_kv);
}

}  // namespace vi
