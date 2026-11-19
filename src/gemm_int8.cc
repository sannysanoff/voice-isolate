#include "gemm_int8.h"

#include "vecmath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define VI_HAS_SDOT 1
#else
#define VI_HAS_SDOT 0
#endif

namespace vi {

// The quantisation range is the row's largest magnitude. Capping it at a few
// standard deviations was tried and is much worse: several of these layers are
// sparse (measured peak/rms up to 47) and there the spike *is* the signal, so
// clipping it destroys the layer. Kept as a note because it looks like it should
// work and does not.
inline float row_range(const float* row, int k) { return v_absmax(row, k); }

bool int8_available() { return VI_HAS_SDOT != 0; }

void quantize_rows(const float* w, int n, int k, Int8Matrix& out) {
  out.n = n;
  out.k = k;
  out.q.resize(static_cast<size_t>(n) * k);
  out.scale.resize(static_cast<size_t>(n));
  for (int j = 0; j < n; ++j) {
    const float* row = w + static_cast<size_t>(j) * k;
    const float range = row_range(row, k);
    const float scale = range > 0.0f ? range / 127.0f : 1.0f;
    out.scale[static_cast<size_t>(j)] = scale;
    v_quantize_i8(row, out.q.data() + static_cast<size_t>(j) * k, k,
                  1.0f / scale);
  }
}

void quantize_activations(const float* a, int m, int k, int stride, int8_t* aq,
                          float* scale) {
  for (int i = 0; i < m; ++i) {
    const float* row = a + static_cast<size_t>(i) * stride;
    const float range = row_range(row, k);
    const float s = range > 0.0f ? range / 127.0f : 1.0f;
    scale[i] = s;
    v_quantize_i8(row, aq + static_cast<size_t>(i) * k, k, 1.0f / s);
  }
}

namespace {

#if VI_HAS_SDOT

// 4 output rows x 4 output columns of int32 accumulators. k advances by 16,
// which is what one sdot consumes.
inline void tile4x4(const int8_t* a0, const int8_t* a1, const int8_t* a2,
                    const int8_t* a3, const int8_t* w0, const int8_t* w1,
                    const int8_t* w2, const int8_t* w3, int k, int32_t* sums) {
  int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00;
  int32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  int32x4_t c20 = c00, c21 = c00, c22 = c00, c23 = c00;
  int32x4_t c30 = c00, c31 = c00, c32 = c00, c33 = c00;
  for (int kk = 0; kk < k; kk += 16) {
    const int8x16_t va0 = vld1q_s8(a0 + kk);
    const int8x16_t va1 = vld1q_s8(a1 + kk);
    const int8x16_t va2 = vld1q_s8(a2 + kk);
    const int8x16_t va3 = vld1q_s8(a3 + kk);
    const int8x16_t vw0 = vld1q_s8(w0 + kk);
    c00 = vdotq_s32(c00, va0, vw0);
    c10 = vdotq_s32(c10, va1, vw0);
    c20 = vdotq_s32(c20, va2, vw0);
    c30 = vdotq_s32(c30, va3, vw0);
    const int8x16_t vw1 = vld1q_s8(w1 + kk);
    c01 = vdotq_s32(c01, va0, vw1);
    c11 = vdotq_s32(c11, va1, vw1);
    c21 = vdotq_s32(c21, va2, vw1);
    c31 = vdotq_s32(c31, va3, vw1);
    const int8x16_t vw2 = vld1q_s8(w2 + kk);
    c02 = vdotq_s32(c02, va0, vw2);
    c12 = vdotq_s32(c12, va1, vw2);
    c22 = vdotq_s32(c22, va2, vw2);
    c32 = vdotq_s32(c32, va3, vw2);
    const int8x16_t vw3 = vld1q_s8(w3 + kk);
    c03 = vdotq_s32(c03, va0, vw3);
    c13 = vdotq_s32(c13, va1, vw3);
    c23 = vdotq_s32(c23, va2, vw3);
    c33 = vdotq_s32(c33, va3, vw3);
  }
  int32x4_t t[16] = {c00, c01, c02, c03, c10, c11, c12, c13,
                     c20, c21, c22, c23, c30, c31, c32, c33};
  for (int i = 0; i < 16; ++i) {
    const int32x2_t pair = vadd_s32(vget_low_s32(t[i]), vget_high_s32(t[i]));
    sums[i] = vget_lane_s32(pair, 0) + vget_lane_s32(pair, 1);
  }
}

// Dot product of one activation row against one weight row, k a multiple of 16.
inline int32_t dot_row(const int8_t* a, const int8_t* w, int k) {
  int32x4_t acc = vdupq_n_s32(0);
  for (int kk = 0; kk < k; kk += 16)
    acc = vdotq_s32(acc, vld1q_s8(a + kk), vld1q_s8(w + kk));
  const int32x2_t pair = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
  return vget_lane_s32(pair, 0) + vget_lane_s32(pair, 1);
}

#endif  // VI_HAS_SDOT

}  // namespace

void gemm_int8_columns(const float* a, int m, int k, int stride,
                       const Int8Matrix& w, const float* bias, float* c, int j0,
                       int j1, int8_t* aq, float* as) {
  const int n = w.n;
  quantize_activations(a, m, k, stride, aq, as);

#if VI_HAS_SDOT
  const int mb = m & ~3;
  for (int i = 0; i < mb; i += 4) {
    const int8_t* a0 = aq + static_cast<size_t>(i + 0) * k;
    const int8_t* a1 = aq + static_cast<size_t>(i + 1) * k;
    const int8_t* a2 = aq + static_cast<size_t>(i + 2) * k;
    const int8_t* a3 = aq + static_cast<size_t>(i + 3) * k;
    int j = j0;
    for (; j + 4 <= j1; j += 4) {
      const int8_t* w0 = w.q.data() + static_cast<size_t>(j + 0) * k;
      const int8_t* w1 = w.q.data() + static_cast<size_t>(j + 1) * k;
      const int8_t* w2 = w.q.data() + static_cast<size_t>(j + 2) * k;
      const int8_t* w3 = w.q.data() + static_cast<size_t>(j + 3) * k;
      int32_t s[16];
      tile4x4(a0, a1, a2, a3, w0, w1, w2, w3, k, s);
      for (int r = 0; r < 4; ++r) {
        float* out = c + static_cast<size_t>(i + r) * n + j;
        const float base = as[i + r];
        for (int q = 0; q < 4; ++q) {
          out[q] = static_cast<float>(s[r * 4 + q]) * base *
                       w.scale[static_cast<size_t>(j + q)] +
                   (bias != nullptr ? bias[j + q] : 0.0f);
        }
      }
    }
    for (; j < j1; ++j) {
      const int8_t* wr = w.q.data() + static_cast<size_t>(j) * k;
      const float ws = w.scale[static_cast<size_t>(j)];
      const float b = bias != nullptr ? bias[j] : 0.0f;
      for (int r = 0; r < 4; ++r) {
        const int8_t* ar = aq + static_cast<size_t>(i + r) * k;
        c[static_cast<size_t>(i + r) * n + j] =
            static_cast<float>(dot_row(ar, wr, k)) * as[i + r] * ws + b;
      }
    }
  }
  for (int i = mb; i < m; ++i) {
    const int8_t* ar = aq + static_cast<size_t>(i) * k;
    for (int j = j0; j < j1; ++j) {
      const int8_t* wr = w.q.data() + static_cast<size_t>(j) * k;
      c[static_cast<size_t>(i) * n + j] =
          static_cast<float>(dot_row(ar, wr, k)) * as[i] *
              w.scale[static_cast<size_t>(j)] +
          (bias != nullptr ? bias[j] : 0.0f);
    }
  }
#else
  // Portable fallback: same arithmetic, one scalar dot per output element. The
  // column range has to be honoured, because workers own disjoint slices.
  for (int i = 0; i < m; ++i) {
    const int8_t* ar = aq + static_cast<size_t>(i) * k;
    for (int j = j0; j < j1; ++j) {
      const int8_t* wr = w.q.data() + static_cast<size_t>(j) * k;
      int32_t s = 0;
      for (int t = 0; t < k; ++t) s += ar[t] * wr[t];
      c[static_cast<size_t>(i) * n + j] =
          static_cast<float>(s) * as[i] * w.scale[static_cast<size_t>(j)] +
          (bias != nullptr ? bias[j] : 0.0f);
    }
  }
#endif
}

}  // namespace vi
