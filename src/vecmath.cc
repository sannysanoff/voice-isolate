#include "vecmath.h"

#include <algorithm>
#include <cmath>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define VI_HAS_NEON 1
#else
#define VI_HAS_NEON 0
#endif

namespace vi {
namespace {

inline xsimd::batch<float, vx::arch> load(const float* p) {
  return xsimd::load_unaligned(p);
}
inline void store(float* p, xsimd::batch<float, vx::arch> v) {
  xsimd::store_unaligned(p, v);
}

template <class Fn>
void elementwise(float* x, int n, Fn fn) {
  using batch = vx::batch;
  int i = 0;
  for (; i + vx::lanes <= n; i += vx::lanes)
    store(x + i, fn(load(x + i)));
  for (; i < n; ++i) x[i] = fn(batch(x[i])).get(0);
}

#if VI_HAS_NEON

// Accurate fast exp(x) for fp32 NEON (max rel err ~2.2e-7 on [-87, 88]).
// Clamps underflow to 0 cleanly without denormals.
inline float32x4_t neon_exp_f32x4(float32x4_t x) {
  x = vmaxq_f32(x, vdupq_n_f32(-87.9f));
  x = vminq_f32(x, vdupq_n_f32(88.0f));
  const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t ln2_hi = vdupq_n_f32(0.693359375f);
  const float32x4_t ln2_lo = vdupq_n_f32(-2.12194440e-4f);
  const float32x4_t n = vrndnq_f32(vmulq_f32(x, log2e));
  float32x4_t r = vfmsq_f32(x, n, ln2_hi);
  r = vfmsq_f32(r, n, ln2_lo);
  // Degree-6 Taylor on [-ln2/2, ln2/2] -> ~2.2e-7 max rel error.
  float32x4_t p = vdupq_n_f32(1.38888889e-3f);
  p = vfmaq_f32(vdupq_n_f32(8.33333333e-3f), p, r);
  p = vfmaq_f32(vdupq_n_f32(4.16666667e-2f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.66666667e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(5.00000000e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  const int32x4_t ni = vcvtq_s32_f32(n);
  const int32x4_t pow2 = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(pow2));
}

// Ultra-fast 2^y for y <= 0 (used in Mamba scan with pre-scaled a_log2e = a * log2(e)).
// Degree-6 polynomial on [-0.5, 0.5] (max rel err 2.6e-9, exact 1.0f at y=0 so
// long recurrences have zero systematic drift).
inline float32x4_t neon_exp2_neg_f32x4(float32x4_t y) {
  y = vmaxq_f32(y, vdupq_n_f32(-126.9f));
  const float32x4_t n = vrndnq_f32(y);
  const float32x4_t f = vsubq_f32(y, n);
  float32x4_t p = vdupq_n_f32(1.546973199e-4f);
  p = vfmaq_f32(vdupq_n_f32(1.340043216e-3f), p, f);
  p = vfmaq_f32(vdupq_n_f32(9.618025603e-3f), p, f);
  p = vfmaq_f32(vdupq_n_f32(5.550327214e-2f), p, f);
  p = vfmaq_f32(vdupq_n_f32(2.402265121e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(6.931471806e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
  const int32x4_t pow2 = vcvtq_s32_f32(
      vfmaq_f32(vdupq_n_f32(1065353216.0f), n, vdupq_n_f32(8388608.0f)));
  return vmulq_f32(p, vreinterpretq_f32_s32(pow2));
}

// Fast accurate log(x) for x > 0 (max abs err ~3e-8 on mantissa in [sqrt(0.5), sqrt(2)]).
inline float32x4_t neon_log_f32x4(float32x4_t x) {
  // Split x = m * 2^e with m in [sqrt(0.5), sqrt(2)].
  const int32x4_t xi = vreinterpretq_s32_f32(x);
  // 0x3f3504f3 = sqrt(0.5)
  const int32x4_t offset = vdupq_n_s32(0x3f3504f3);
  const int32x4_t e_int = vshrq_n_s32(vsubq_s32(xi, offset), 23);
  const int32x4_t m_bits =
      vaddq_s32(vandq_s32(vsubq_s32(xi, offset), vdupq_n_s32(0x007fffff)), offset);
  const float32x4_t m = vreinterpretq_f32_s32(m_bits);
  const float32x4_t e = vcvtq_f32_s32(e_int);
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t u = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
  const float32x4_t u2 = vmulq_f32(u, u);
  float32x4_t p = vdupq_n_f32(1.0f / 7.0f);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 5.0f), p, u2);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 3.0f), p, u2);
  p = vfmaq_f32(one, p, u2);
  const float32x4_t log_m = vmulq_f32(vmulq_f32(vdupq_n_f32(2.0f), u), p);
  return vfmaq_f32(log_m, e, vdupq_n_f32(0.6931471805599453f));
}

inline float32x4_t neon_silu_f32x4(float32x4_t x) {
  return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0f), neon_exp_f32x4(vnegq_f32(x))));
}

inline float32x4_t neon_softplus_f32x4(float32x4_t x) {
  const float32x4_t ex = neon_exp_f32x4(x);
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t curved = neon_log_f32x4(vaddq_f32(one, ex));
  // For x < -15, log1p(exp(x)) == exp(x) to float32 precision (avoids 1+ex cancellation).
  const uint32x4_t is_small = vcltq_f32(x, vdupq_n_f32(-15.0f));
  const uint32x4_t is_large = vcgtq_f32(x, vdupq_n_f32(20.0f));
  return vbslq_f32(is_large, x, vbslq_f32(is_small, ex, curved));
}

// Exact-fp32 GELU via zero-cancellation erfc approximation (max rel err ~4.8e-7).
inline float32x4_t neon_gelu_f32x4(float32x4_t x) {
  const float32x4_t ax = vabsq_f32(x);
  const float32x4_t u = vmulq_f32(ax, vdupq_n_f32(0.70710678118654752440f));
  const float32x4_t t =
      vdivq_f32(vdupq_n_f32(1.0f), vfmaq_f32(vdupq_n_f32(1.0f), u, vdupq_n_f32(0.38f)));
  float32x4_t p = vdupq_n_f32(3.196994254e-2f);
  p = vfmaq_f32(vdupq_n_f32(-2.618669679e-1f), p, t);
  p = vfmaq_f32(vdupq_n_f32(5.775581467e-1f), p, t);
  p = vfmaq_f32(vdupq_n_f32(-2.990170914e-1f), p, t);
  p = vfmaq_f32(vdupq_n_f32(3.732227110e-1f), p, t);
  p = vfmaq_f32(vdupq_n_f32(1.411143942e-1f), p, t);
  p = vfmaq_f32(vdupq_n_f32(2.231841650e-1f), p, t);
  p = vfmaq_f32(vdupq_n_f32(2.138345334e-1f), p, t);
  const float32x4_t eu2 = neon_exp_f32x4(vnegq_f32(vmulq_f32(u, u)));
  const float32x4_t erfc_u = vmulq_f32(eu2, vmulq_f32(t, p));
  const float32x4_t half_x_erfc = vmulq_f32(vmulq_f32(vdupq_n_f32(0.5f), x), erfc_u);
  const uint32x4_t ge0 = vcgeq_f32(x, vdupq_n_f32(0.0f));
  return vbslq_f32(ge0, vsubq_f32(x, half_x_erfc), half_x_erfc);
}

#endif  // VI_HAS_NEON

}  // namespace

void v_exp(float* x, int n) {
#if VI_HAS_NEON
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t v0 = vld1q_f32(x + i);
    float32x4_t v1 = vld1q_f32(x + i + 4);
    vst1q_f32(x + i, neon_exp_f32x4(v0));
    vst1q_f32(x + i + 4, neon_exp_f32x4(v1));
  }
  for (; i + 4 <= n; i += 4) vst1q_f32(x + i, neon_exp_f32x4(vld1q_f32(x + i)));
  for (; i < n; ++i) x[i] = std::exp(x[i]);
#else
  elementwise(x, n, [](vx::batch v) { return xsimd::exp(v); });
#endif
}

void v_tanh(float* x, int n) {
  elementwise(x, n, [](vx::batch v) { return xsimd::tanh(v); });
}

void v_gelu(float* x, int n) {
#if VI_HAS_NEON
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t v0 = vld1q_f32(x + i);
    float32x4_t v1 = vld1q_f32(x + i + 4);
    vst1q_f32(x + i, neon_gelu_f32x4(v0));
    vst1q_f32(x + i + 4, neon_gelu_f32x4(v1));
  }
  for (; i + 4 <= n; i += 4) vst1q_f32(x + i, neon_gelu_f32x4(vld1q_f32(x + i)));
  for (; i < n; ++i)
    x[i] = 0.5f * x[i] * (1.0f + std::erff(x[i] * 0.70710678118654752440f));
#else
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  elementwise(x, n, [](vx::batch v) {
    return 0.5f * v * (vx::batch(1.0f) + xsimd::erf(v * vx::batch(kInvSqrt2)));
  });
#endif
}

void v_silu(float* x, int n) {
#if VI_HAS_NEON
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t v0 = vld1q_f32(x + i);
    float32x4_t v1 = vld1q_f32(x + i + 4);
    vst1q_f32(x + i, neon_silu_f32x4(v0));
    vst1q_f32(x + i + 4, neon_silu_f32x4(v1));
  }
  for (; i + 4 <= n; i += 4) vst1q_f32(x + i, neon_silu_f32x4(vld1q_f32(x + i)));
  for (; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
#else
  elementwise(x, n, [](vx::batch v) {
    return v / (vx::batch(1.0f) + xsimd::exp(-v));
  });
#endif
}

void v_softplus(float* x, int n) {
#if VI_HAS_NEON
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t v0 = vld1q_f32(x + i);
    float32x4_t v1 = vld1q_f32(x + i + 4);
    vst1q_f32(x + i, neon_softplus_f32x4(v0));
    vst1q_f32(x + i + 4, neon_softplus_f32x4(v1));
  }
  for (; i + 4 <= n; i += 4)
    vst1q_f32(x + i, neon_softplus_f32x4(vld1q_f32(x + i)));
  for (; i < n; ++i) x[i] = x[i] > 20.0f ? x[i] : std::log1p(std::exp(x[i]));
#else
  elementwise(x, n, [](vx::batch v) {
    const auto linear = v;
    const auto curved = xsimd::log1p(xsimd::exp(v));
    return xsimd::select(v > vx::batch(20.0f), linear, curved);
  });
#endif
}

void v_scale(float* x, int n, float factor) {
  const vx::batch s(factor);
  elementwise(x, n, [s](vx::batch v) { return v * s; });
}

float v_sum(const float* x, int n) {
  vx::batch acc(0.0f);
  int i = 0;
  for (; i + vx::lanes <= n; i += vx::lanes) acc += load(x + i);
  float sum = xsimd::reduce_add(acc);
  for (; i < n; ++i) sum += x[i];
  return sum;
}

float v_absmax(const float* x, int n) {
  vx::batch acc(0.0f);
  int i = 0;
  for (; i + vx::lanes <= n; i += vx::lanes)
    acc = xsimd::max(acc, xsimd::abs(load(x + i)));
  float peak = xsimd::reduce_max(acc);
  for (; i < n; ++i) peak = std::max(peak, std::fabs(x[i]));
  return peak;
}

float v_rms(const float* x, int n) {
  vx::batch acc(0.0f);
  int i = 0;
  for (; i + vx::lanes <= n; i += vx::lanes) {
    const vx::batch v = load(x + i);
    acc += v * v;
  }
  float sum = xsimd::reduce_add(acc);
  for (; i < n; ++i) sum += x[i] * x[i];
  return std::sqrt(sum / static_cast<float>(n));
}

void v_quantize_i8(const float* src, int8_t* dst, int n, float inv_scale) {
#if VI_HAS_NEON
  const float32x4_t vs = vdupq_n_f32(inv_scale);
  const float32x4_t vlo = vdupq_n_f32(-127.0f), vhi = vdupq_n_f32(127.0f);
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    int32x4_t q0 = vcvtaq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src + i + 0), vs), vlo), vhi));
    int32x4_t q1 = vcvtaq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src + i + 4), vs), vlo), vhi));
    int32x4_t q2 = vcvtaq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src + i + 8), vs), vlo), vhi));
    int32x4_t q3 = vcvtaq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src + i + 12), vs), vlo), vhi));
    int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    int8x16_t b = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
    vst1q_s8(dst + i, b);
  }
  for (; i < n; ++i) {
    const float v = src[i] * inv_scale;
    const int q = static_cast<int>(std::lround(v));
    dst[i] = static_cast<int8_t>(std::min(127, std::max(-127, q)));
  }
#else
  const vx::batch scale(inv_scale);
  const vx::batch lo(-127.0f), hi(127.0f);
  int i = 0;
  for (; i + vx::lanes <= n; i += vx::lanes) {
    const vx::batch v = xsimd::min(xsimd::max(load(src + i) * scale, lo), hi);
    const auto rounded = xsimd::round(v);
    alignas(vx::lanes * 4) float lane[vx::lanes];
    store(lane, rounded);
    for (int t = 0; t < vx::lanes; ++t) dst[i + t] = static_cast<int8_t>(lane[t]);
  }
  for (; i < n; ++i) {
    const float v = src[i] * inv_scale;
    const int q = static_cast<int>(std::lround(v));
    dst[i] = static_cast<int8_t>(std::min(127, std::max(-127, q)));
  }
#endif
}

void v_mul_exp(float* out, const float* a, const float* scale, int n) {
#if VI_HAS_NEON
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t v0 = vmulq_f32(vld1q_f32(a + i), vld1q_f32(scale + i));
    float32x4_t v1 = vmulq_f32(vld1q_f32(a + i + 4), vld1q_f32(scale + i + 4));
    vst1q_f32(out + i, neon_exp_f32x4(v0));
    vst1q_f32(out + i + 4, neon_exp_f32x4(v1));
  }
  for (; i + 4 <= n; i += 4)
    vst1q_f32(out + i, neon_exp_f32x4(vmulq_f32(vld1q_f32(a + i), vld1q_f32(scale + i))));
  for (; i < n; ++i) out[i] = std::exp(a[i] * scale[i]);
#else
  const int i_end = n - (n % vx::lanes);
  for (int i = 0; i < i_end; i += vx::lanes)
    store(out + i, xsimd::exp(load(a + i) * load(scale + i)));
  for (int i = i_end; i < n; ++i) out[i] = std::exp(a[i] * scale[i]);
#endif
}

void v_scan_step(float* h, const float* a, const float* delta,
                 const float* du, float bn, float cn, float* y, int n) {
  const vx::batch vbn(bn);
  const vx::batch vcn(cn);
  const int i_end = n - (n % vx::lanes);
  for (int i = 0; i < i_end; i += vx::lanes) {
    const vx::batch hv = load(h + i);
    const vx::batch decay = xsimd::exp(load(a + i) * load(delta + i));
    const vx::batch next = decay * hv + vbn * load(du + i);
    store(h + i, next);
    store(y + i, load(y + i) + vcn * next);
  }
  for (int i = i_end; i < n; ++i) {
    const float next = std::exp(a[i] * delta[i]) * h[i] + du[i] * bn;
    h[i] = next;
    y[i] += next * cn;
  }
}

void v_mamba_scan(float* __restrict h4, const float* __restrict a_log2e4,
                  const float* __restrict d_skip, const float* __restrict u,
                  const float* __restrict delta, const float* __restrict z,
                  const float* __restrict xp, float* __restrict y, int len,
                  int inner, int state, int dt_rank, int dt_total, int c0,
                  int c1) {
#if VI_HAS_NEON
  if (state == 16 && (c0 % 4 == 0) && (c1 % 4 == 0)) {
    const size_t inner_s = static_cast<size_t>(inner);
    const size_t z_stride = 2 * inner_s;
    // Two passes of 8 states keep h[0..7], a[0..7], and all exp2 constants
    // simultaneously in the 32 NEON registers with zero spills.
    for (int c = c0; c < c1; c += 4) {
      const size_t grp = static_cast<size_t>(c / 4) * 16 * 4;
      const float* ap = a_log2e4 + grp;
      float* hp = h4 != nullptr ? (h4 + grp) : nullptr;
      const float32x4_t vD = vld1q_f32(d_skip + c);

      // Pass 1: states 0..7
      float32x4_t h0, h1, h2, h3, h4v, h5, h6, h7;
      if (hp != nullptr) {
        h0 = vld1q_f32(hp + 0);   h1 = vld1q_f32(hp + 4);
        h2 = vld1q_f32(hp + 8);   h3 = vld1q_f32(hp + 12);
        h4v = vld1q_f32(hp + 16); h5 = vld1q_f32(hp + 20);
        h6 = vld1q_f32(hp + 24);  h7 = vld1q_f32(hp + 28);
      } else {
        const float32x4_t z0 = vdupq_n_f32(0.0f);
        h0 = z0; h1 = z0; h2 = z0; h3 = z0;
        h4v = z0; h5 = z0; h6 = z0; h7 = z0;
      }
      const float32x4_t a0 = vld1q_f32(ap + 0), a1 = vld1q_f32(ap + 4);
      const float32x4_t a2 = vld1q_f32(ap + 8), a3 = vld1q_f32(ap + 12);
      const float32x4_t a4 = vld1q_f32(ap + 16), a5 = vld1q_f32(ap + 20);
      const float32x4_t a6 = vld1q_f32(ap + 24), a7 = vld1q_f32(ap + 28);

      const float* ut = u + c;
      const float* dt = delta + c;
      float* yt = y + c;
      const float* bt = xp + dt_rank;

      for (int t = 0; t < len; ++t) {
        const float32x4_t vu = vld1q_f32(ut);
        const float32x4_t vd = vld1q_f32(dt);
        const float32x4_t vdu = vmulq_f32(vd, vu);
        float32x4_t vy = vmulq_f32(vD, vu);
        const float* ct = bt + 16;
        const float32x4_t vb0 = vld1q_f32(bt + 0), vb1 = vld1q_f32(bt + 4);
        const float32x4_t vc0 = vld1q_f32(ct + 0), vc1 = vld1q_f32(ct + 4);
        h0 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(a0, vd)), h0), vdu, vb0, 0);
        vy = vfmaq_laneq_f32(vy, h0, vc0, 0);
        h1 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(a1, vd)), h1), vdu, vb0, 1);
        vy = vfmaq_laneq_f32(vy, h1, vc0, 1);
        h2 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(a2, vd)), h2), vdu, vb0, 2);
        vy = vfmaq_laneq_f32(vy, h2, vc0, 2);
        h3 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(a3, vd)), h3), vdu, vb0, 3);
        vy = vfmaq_laneq_f32(vy, h3, vc0, 3);
        h4v = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(a4, vd)), h4v), vdu, vb1, 0);
        vy = vfmaq_laneq_f32(vy, h4v, vc1, 0);
        h5 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(a5, vd)), h5), vdu, vb1, 1);
        vy = vfmaq_laneq_f32(vy, h5, vc1, 1);
        h6 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(a6, vd)), h6), vdu, vb1, 2);
        vy = vfmaq_laneq_f32(vy, h6, vc1, 2);
        h7 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(a7, vd)), h7), vdu, vb1, 3);
        vy = vfmaq_laneq_f32(vy, h7, vc1, 3);
        vst1q_f32(yt, vy);
        ut += inner_s; dt += inner_s; yt += inner_s; bt += dt_total;
      }
      if (hp != nullptr) {
        vst1q_f32(hp + 0, h0);   vst1q_f32(hp + 4, h1);
        vst1q_f32(hp + 8, h2);   vst1q_f32(hp + 12, h3);
        vst1q_f32(hp + 16, h4v); vst1q_f32(hp + 20, h5);
        vst1q_f32(hp + 24, h6);  vst1q_f32(hp + 28, h7);
      }

      // Pass 2: states 8..15 + fused silu(z)
      if (hp != nullptr) {
        h0 = vld1q_f32(hp + 32);  h1 = vld1q_f32(hp + 36);
        h2 = vld1q_f32(hp + 40);  h3 = vld1q_f32(hp + 44);
        h4v = vld1q_f32(hp + 48); h5 = vld1q_f32(hp + 52);
        h6 = vld1q_f32(hp + 56);  h7 = vld1q_f32(hp + 60);
      } else {
        const float32x4_t z0 = vdupq_n_f32(0.0f);
        h0 = z0; h1 = z0; h2 = z0; h3 = z0;
        h4v = z0; h5 = z0; h6 = z0; h7 = z0;
      }
      const float32x4_t b0 = vld1q_f32(ap + 32), b1 = vld1q_f32(ap + 36);
      const float32x4_t b2 = vld1q_f32(ap + 40), b3 = vld1q_f32(ap + 44);
      const float32x4_t b4 = vld1q_f32(ap + 48), b5 = vld1q_f32(ap + 52);
      const float32x4_t b6 = vld1q_f32(ap + 56), b7 = vld1q_f32(ap + 60);

      ut = u + c; dt = delta + c; yt = y + c; bt = xp + dt_rank;
      const float* zt = z + inner + c;
      for (int t = 0; t < len; ++t) {
        const float32x4_t vu = vld1q_f32(ut);
        const float32x4_t vd = vld1q_f32(dt);
        const float32x4_t vdu = vmulq_f32(vd, vu);
        float32x4_t vy = vld1q_f32(yt);
        const float* ct = bt + 16;
        const float32x4_t vb0 = vld1q_f32(bt + 8), vb1 = vld1q_f32(bt + 12);
        const float32x4_t vc0 = vld1q_f32(ct + 8), vc1 = vld1q_f32(ct + 12);
        h0 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(b0, vd)), h0), vdu, vb0, 0);
        vy = vfmaq_laneq_f32(vy, h0, vc0, 0);
        h1 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(b1, vd)), h1), vdu, vb0, 1);
        vy = vfmaq_laneq_f32(vy, h1, vc0, 1);
        h2 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(b2, vd)), h2), vdu, vb0, 2);
        vy = vfmaq_laneq_f32(vy, h2, vc0, 2);
        h3 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(b3, vd)), h3), vdu, vb0, 3);
        vy = vfmaq_laneq_f32(vy, h3, vc0, 3);
        h4v = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(b4, vd)), h4v), vdu, vb1, 0);
        vy = vfmaq_laneq_f32(vy, h4v, vc1, 0);
        h5 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(b5, vd)), h5), vdu, vb1, 1);
        vy = vfmaq_laneq_f32(vy, h5, vc1, 1);
        h6 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(b6, vd)), h6), vdu, vb1, 2);
        vy = vfmaq_laneq_f32(vy, h6, vc1, 2);
        h7 = vfmaq_laneq_f32(vmulq_f32(neon_exp2_neg_f32x4(vmulq_f32(b7, vd)), h7), vdu, vb1, 3);
        vy = vfmaq_laneq_f32(vy, h7, vc1, 3);
        const float32x4_t vz = vld1q_f32(zt);
        vst1q_f32(yt, vmulq_f32(vy, neon_silu_f32x4(vz)));
        ut += inner_s; dt += inner_s; yt += inner_s; zt += z_stride; bt += dt_total;
      }
      if (hp != nullptr) {
        vst1q_f32(hp + 32, h0);  vst1q_f32(hp + 36, h1);
        vst1q_f32(hp + 40, h2);  vst1q_f32(hp + 44, h3);
        vst1q_f32(hp + 48, h4v); vst1q_f32(hp + 52, h5);
        vst1q_f32(hp + 56, h6);  vst1q_f32(hp + 60, h7);
      }
    }
    return;
  }
#endif
  // Portable fallback for non-NEON or non-16 state.
  constexpr float kLn2 = 0.6931471805599453f;
  std::vector<float> h_local;
  if (h4 == nullptr) h_local.assign(static_cast<size_t>(state) * 4, 0.0f);
  for (int c = c0; c < c1; ++c) {
    const int g = c / 4;
    const int lane = c % 4;
    const float* ap = a_log2e4 + static_cast<size_t>(g) * state * 4 + lane;
    float* hp = h4 != nullptr ? (h4 + static_cast<size_t>(g) * state * 4 + lane)
                              : h_local.data();
    const int hstep = h4 != nullptr ? 4 : 1;
    if (h4 == nullptr) std::fill(h_local.begin(), h_local.end(), 0.0f);
    for (int t = 0; t < len; ++t) {
      const float uc = u[static_cast<size_t>(t) * inner + c];
      const float dc = delta[static_cast<size_t>(t) * inner + c];
      const float du = dc * uc;
      const float* bv = xp + static_cast<size_t>(t) * dt_total + dt_rank;
      const float* cv = bv + state;
      float acc = d_skip[c] * uc;
      for (int n = 0; n < state; ++n) {
        const float a_val = ap[n * 4] * kLn2;
        const float hn = std::exp(a_val * dc) * hp[n * hstep] + du * bv[n];
        hp[n * hstep] = hn;
        acc += hn * cv[n];
      }
      const float zc = z[static_cast<size_t>(t) * 2 * inner + inner + c];
      y[static_cast<size_t>(t) * inner + c] = acc * silu_scalar(zc);
    }
  }
}

void v_mamba_scan_f16(float* __restrict h4,
                      const uint16_t* __restrict a_log2e8_u,
                      const float* __restrict d_skip, const float* __restrict u,
                      const float* __restrict delta, const float* __restrict z,
                      const float* __restrict xp, float* __restrict y, int len,
                      int inner, int state, int dt_rank, int dt_total, int c0,
                      int c1) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  if (state == 16 && (c0 % 8 == 0) && (c1 % 8 == 0)) {
    const __fp16* a16 = reinterpret_cast<const __fp16*>(a_log2e8_u);
    const size_t inner_s = static_cast<size_t>(inner);
    const size_t z_stride = 2 * inner_s;
    const float16x8_t c_min = vdupq_n_f16(-14.9f);
    const float16x8_t c3 = vdupq_n_f16(0.0555041f);
    const float16x8_t c2 = vdupq_n_f16(0.2402265f);
    const float16x8_t c1k = vdupq_n_f16(0.6931472f);
    const float16x8_t c0k = vdupq_n_f16(1.0f);
    const float16x8_t c15s = vdupq_n_f16(15360.0f);
    const float16x8_t c1024 = vdupq_n_f16(1024.0f);
    auto exp2_f16 = [&](float16x8_t yv) __attribute__((always_inline)) {
      yv = vmaxq_f16(yv, c_min);
      const float16x8_t n = vrndnq_f16(yv);
      const float16x8_t f = vsubq_f16(yv, n);
      float16x8_t p = vfmaq_f16(c2, c3, f);
      p = vfmaq_f16(c1k, p, f);
      p = vfmaq_f16(c0k, p, f);
      const int16x8_t pow2 = vcvtq_s16_f16(vfmaq_f16(c15s, n, c1024));
      return vmulq_f16(p, vreinterpretq_f16_s16(pow2));
    };

    for (int c = c0; c < c1; c += 8) {
      const __fp16* ap = a16 + static_cast<size_t>(c / 8) * 16 * 8;
      // h4 is [inner/4][16][4]: two adjacent 4-channel groups for [c..c+3] and [c+4..c+7].
      float* hp_lo = h4 != nullptr ? (h4 + static_cast<size_t>(c / 4) * 16 * 4) : nullptr;
      float* hp_hi = h4 != nullptr ? (hp_lo + 16 * 4) : nullptr;
      const float16x8_t vD = vcombine_f16(vcvt_f16_f32(vld1q_f32(d_skip + c)),
                                          vcvt_f16_f32(vld1q_f32(d_skip + c + 4)));

      // Pass 1: states 0..7
      float16x8_t h0, h1, h2, h3, h4v, h5, h6, h7;
      if (hp_lo != nullptr) {
        h0 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 0)),  vcvt_f16_f32(vld1q_f32(hp_hi + 0)));
        h1 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 4)),  vcvt_f16_f32(vld1q_f32(hp_hi + 4)));
        h2 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 8)),  vcvt_f16_f32(vld1q_f32(hp_hi + 8)));
        h3 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 12)), vcvt_f16_f32(vld1q_f32(hp_hi + 12)));
        h4v = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 16)), vcvt_f16_f32(vld1q_f32(hp_hi + 16)));
        h5 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 20)), vcvt_f16_f32(vld1q_f32(hp_hi + 20)));
        h6 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 24)), vcvt_f16_f32(vld1q_f32(hp_hi + 24)));
        h7 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 28)), vcvt_f16_f32(vld1q_f32(hp_hi + 28)));
      } else {
        const float16x8_t z0 = vdupq_n_f16(0);
        h0 = z0; h1 = z0; h2 = z0; h3 = z0;
        h4v = z0; h5 = z0; h6 = z0; h7 = z0;
      }
      const float16x8_t a0 = vld1q_f16(ap + 0),  a1 = vld1q_f16(ap + 8);
      const float16x8_t a2 = vld1q_f16(ap + 16), a3 = vld1q_f16(ap + 24);
      const float16x8_t a4 = vld1q_f16(ap + 32), a5 = vld1q_f16(ap + 40);
      const float16x8_t a6 = vld1q_f16(ap + 48), a7 = vld1q_f16(ap + 56);
      const float *ut = u + c, *dt = delta + c, *bt = xp + dt_rank;
      float* yt = y + c;
      for (int t = 0; t < len; ++t) {
        const float16x8_t vu = vcombine_f16(vcvt_f16_f32(vld1q_f32(ut)), vcvt_f16_f32(vld1q_f32(ut + 4)));
        const float16x8_t vd = vcombine_f16(vcvt_f16_f32(vld1q_f32(dt)), vcvt_f16_f32(vld1q_f32(dt + 4)));
        const float16x8_t vdu = vmulq_f16(vd, vu);
        float16x8_t vy = vmulq_f16(vD, vu);
        const float* ct = bt + 16;
        const float16x8_t vb = vcombine_f16(vcvt_f16_f32(vld1q_f32(bt)), vcvt_f16_f32(vld1q_f32(bt + 4)));
        const float16x8_t vc = vcombine_f16(vcvt_f16_f32(vld1q_f32(ct)), vcvt_f16_f32(vld1q_f32(ct + 4)));
        h0 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(a0, vd)), h0), vdu, vb, 0); vy = vfmaq_laneq_f16(vy, h0, vc, 0);
        h1 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(a1, vd)), h1), vdu, vb, 1); vy = vfmaq_laneq_f16(vy, h1, vc, 1);
        h2 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(a2, vd)), h2), vdu, vb, 2); vy = vfmaq_laneq_f16(vy, h2, vc, 2);
        h3 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(a3, vd)), h3), vdu, vb, 3); vy = vfmaq_laneq_f16(vy, h3, vc, 3);
        h4v = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(a4, vd)), h4v), vdu, vb, 4); vy = vfmaq_laneq_f16(vy, h4v, vc, 4);
        h5 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(a5, vd)), h5), vdu, vb, 5); vy = vfmaq_laneq_f16(vy, h5, vc, 5);
        h6 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(a6, vd)), h6), vdu, vb, 6); vy = vfmaq_laneq_f16(vy, h6, vc, 6);
        h7 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(a7, vd)), h7), vdu, vb, 7); vy = vfmaq_laneq_f16(vy, h7, vc, 7);
        vst1q_f32(yt, vcvt_f32_f16(vget_low_f16(vy)));
        vst1q_f32(yt + 4, vcvt_f32_f16(vget_high_f16(vy)));
        ut += inner_s; dt += inner_s; yt += inner_s; bt += dt_total;
      }
      if (hp_lo != nullptr) {
        vst1q_f32(hp_lo + 0,  vcvt_f32_f16(vget_low_f16(h0)));  vst1q_f32(hp_hi + 0,  vcvt_f32_f16(vget_high_f16(h0)));
        vst1q_f32(hp_lo + 4,  vcvt_f32_f16(vget_low_f16(h1)));  vst1q_f32(hp_hi + 4,  vcvt_f32_f16(vget_high_f16(h1)));
        vst1q_f32(hp_lo + 8,  vcvt_f32_f16(vget_low_f16(h2)));  vst1q_f32(hp_hi + 8,  vcvt_f32_f16(vget_high_f16(h2)));
        vst1q_f32(hp_lo + 12, vcvt_f32_f16(vget_low_f16(h3)));  vst1q_f32(hp_hi + 12, vcvt_f32_f16(vget_high_f16(h3)));
        vst1q_f32(hp_lo + 16, vcvt_f32_f16(vget_low_f16(h4v))); vst1q_f32(hp_hi + 16, vcvt_f32_f16(vget_high_f16(h4v)));
        vst1q_f32(hp_lo + 20, vcvt_f32_f16(vget_low_f16(h5)));  vst1q_f32(hp_hi + 20, vcvt_f32_f16(vget_high_f16(h5)));
        vst1q_f32(hp_lo + 24, vcvt_f32_f16(vget_low_f16(h6)));  vst1q_f32(hp_hi + 24, vcvt_f32_f16(vget_high_f16(h6)));
        vst1q_f32(hp_lo + 28, vcvt_f32_f16(vget_low_f16(h7)));  vst1q_f32(hp_hi + 28, vcvt_f32_f16(vget_high_f16(h7)));
      }

      // Pass 2: states 8..15 + fp32 silu(z)
      if (hp_lo != nullptr) {
        h0 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 32)), vcvt_f16_f32(vld1q_f32(hp_hi + 32)));
        h1 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 36)), vcvt_f16_f32(vld1q_f32(hp_hi + 36)));
        h2 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 40)), vcvt_f16_f32(vld1q_f32(hp_hi + 40)));
        h3 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 44)), vcvt_f16_f32(vld1q_f32(hp_hi + 44)));
        h4v = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 48)), vcvt_f16_f32(vld1q_f32(hp_hi + 48)));
        h5 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 52)), vcvt_f16_f32(vld1q_f32(hp_hi + 52)));
        h6 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 56)), vcvt_f16_f32(vld1q_f32(hp_hi + 56)));
        h7 = vcombine_f16(vcvt_f16_f32(vld1q_f32(hp_lo + 60)), vcvt_f16_f32(vld1q_f32(hp_hi + 60)));
      } else {
        const float16x8_t z0 = vdupq_n_f16(0);
        h0 = z0; h1 = z0; h2 = z0; h3 = z0;
        h4v = z0; h5 = z0; h6 = z0; h7 = z0;
      }
      const float16x8_t b0 = vld1q_f16(ap + 64),  b1 = vld1q_f16(ap + 72);
      const float16x8_t b2 = vld1q_f16(ap + 80),  b3 = vld1q_f16(ap + 88);
      const float16x8_t b4 = vld1q_f16(ap + 96),  b5 = vld1q_f16(ap + 104);
      const float16x8_t b6 = vld1q_f16(ap + 112), b7 = vld1q_f16(ap + 120);
      ut = u + c; dt = delta + c; yt = y + c; bt = xp + dt_rank;
      const float* zt = z + inner + c;
      for (int t = 0; t < len; ++t) {
        const float16x8_t vu = vcombine_f16(vcvt_f16_f32(vld1q_f32(ut)), vcvt_f16_f32(vld1q_f32(ut + 4)));
        const float16x8_t vd = vcombine_f16(vcvt_f16_f32(vld1q_f32(dt)), vcvt_f16_f32(vld1q_f32(dt + 4)));
        const float16x8_t vdu = vmulq_f16(vd, vu);
        float16x8_t vy = vcombine_f16(vcvt_f16_f32(vld1q_f32(yt)), vcvt_f16_f32(vld1q_f32(yt + 4)));
        const float* ct = bt + 16;
        const float16x8_t vb = vcombine_f16(vcvt_f16_f32(vld1q_f32(bt + 8)), vcvt_f16_f32(vld1q_f32(bt + 12)));
        const float16x8_t vc = vcombine_f16(vcvt_f16_f32(vld1q_f32(ct + 8)), vcvt_f16_f32(vld1q_f32(ct + 12)));
        h0 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(b0, vd)), h0), vdu, vb, 0); vy = vfmaq_laneq_f16(vy, h0, vc, 0);
        h1 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(b1, vd)), h1), vdu, vb, 1); vy = vfmaq_laneq_f16(vy, h1, vc, 1);
        h2 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(b2, vd)), h2), vdu, vb, 2); vy = vfmaq_laneq_f16(vy, h2, vc, 2);
        h3 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(b3, vd)), h3), vdu, vb, 3); vy = vfmaq_laneq_f16(vy, h3, vc, 3);
        h4v = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(b4, vd)), h4v), vdu, vb, 4); vy = vfmaq_laneq_f16(vy, h4v, vc, 4);
        h5 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(b5, vd)), h5), vdu, vb, 5); vy = vfmaq_laneq_f16(vy, h5, vc, 5);
        h6 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(b6, vd)), h6), vdu, vb, 6); vy = vfmaq_laneq_f16(vy, h6, vc, 6);
        h7 = vfmaq_laneq_f16(vmulq_f16(exp2_f16(vmulq_f16(b7, vd)), h7), vdu, vb, 7); vy = vfmaq_laneq_f16(vy, h7, vc, 7);
        const float32x4_t s0 = neon_silu_f32x4(vld1q_f32(zt));
        const float32x4_t s1 = neon_silu_f32x4(vld1q_f32(zt + 4));
        vst1q_f32(yt, vmulq_f32(vcvt_f32_f16(vget_low_f16(vy)), s0));
        vst1q_f32(yt + 4, vmulq_f32(vcvt_f32_f16(vget_high_f16(vy)), s1));
        ut += inner_s; dt += inner_s; yt += inner_s; zt += z_stride; bt += dt_total;
      }
      if (hp_lo != nullptr) {
        vst1q_f32(hp_lo + 32, vcvt_f32_f16(vget_low_f16(h0)));  vst1q_f32(hp_hi + 32, vcvt_f32_f16(vget_high_f16(h0)));
        vst1q_f32(hp_lo + 36, vcvt_f32_f16(vget_low_f16(h1)));  vst1q_f32(hp_hi + 36, vcvt_f32_f16(vget_high_f16(h1)));
        vst1q_f32(hp_lo + 40, vcvt_f32_f16(vget_low_f16(h2)));  vst1q_f32(hp_hi + 40, vcvt_f32_f16(vget_high_f16(h2)));
        vst1q_f32(hp_lo + 44, vcvt_f32_f16(vget_low_f16(h3)));  vst1q_f32(hp_hi + 44, vcvt_f32_f16(vget_high_f16(h3)));
        vst1q_f32(hp_lo + 48, vcvt_f32_f16(vget_low_f16(h4v))); vst1q_f32(hp_hi + 48, vcvt_f32_f16(vget_high_f16(h4v)));
        vst1q_f32(hp_lo + 52, vcvt_f32_f16(vget_low_f16(h5)));  vst1q_f32(hp_hi + 52, vcvt_f32_f16(vget_high_f16(h5)));
        vst1q_f32(hp_lo + 56, vcvt_f32_f16(vget_low_f16(h6)));  vst1q_f32(hp_hi + 56, vcvt_f32_f16(vget_high_f16(h6)));
        vst1q_f32(hp_lo + 60, vcvt_f32_f16(vget_low_f16(h7)));  vst1q_f32(hp_hi + 60, vcvt_f32_f16(vget_high_f16(h7)));
      }
    }
    return;
  }
#endif
  (void)h4; (void)a_log2e8_u; (void)d_skip; (void)u; (void)delta; (void)z;
  (void)xp; (void)y; (void)len; (void)inner; (void)state; (void)dt_rank;
  (void)dt_total; (void)c0; (void)c1;
}

void v_conv4_silu(float* __restrict dst, const float* __restrict bias,
                  const float* __restrict w0, const float* __restrict w1,
                  const float* __restrict w2, const float* __restrict w3,
                  const float* __restrict x0, const float* __restrict x1,
                  const float* __restrict x2, const float* __restrict x3,
                  int n) {
#if VI_HAS_NEON
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t a0 = vld1q_f32(bias + i);
    float32x4_t a1 = vld1q_f32(bias + i + 4);
    a0 = vfmaq_f32(a0, vld1q_f32(w0 + i), vld1q_f32(x0 + i));
    a1 = vfmaq_f32(a1, vld1q_f32(w0 + i + 4), vld1q_f32(x0 + i + 4));
    a0 = vfmaq_f32(a0, vld1q_f32(w1 + i), vld1q_f32(x1 + i));
    a1 = vfmaq_f32(a1, vld1q_f32(w1 + i + 4), vld1q_f32(x1 + i + 4));
    a0 = vfmaq_f32(a0, vld1q_f32(w2 + i), vld1q_f32(x2 + i));
    a1 = vfmaq_f32(a1, vld1q_f32(w2 + i + 4), vld1q_f32(x2 + i + 4));
    a0 = vfmaq_f32(a0, vld1q_f32(w3 + i), vld1q_f32(x3 + i));
    a1 = vfmaq_f32(a1, vld1q_f32(w3 + i + 4), vld1q_f32(x3 + i + 4));
    vst1q_f32(dst + i, neon_silu_f32x4(a0));
    vst1q_f32(dst + i + 4, neon_silu_f32x4(a1));
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t a0 = vld1q_f32(bias + i);
    a0 = vfmaq_f32(a0, vld1q_f32(w0 + i), vld1q_f32(x0 + i));
    a0 = vfmaq_f32(a0, vld1q_f32(w1 + i), vld1q_f32(x1 + i));
    a0 = vfmaq_f32(a0, vld1q_f32(w2 + i), vld1q_f32(x2 + i));
    a0 = vfmaq_f32(a0, vld1q_f32(w3 + i), vld1q_f32(x3 + i));
    vst1q_f32(dst + i, neon_silu_f32x4(a0));
  }
  for (; i < n; ++i) {
    float a = bias[i] + w0[i] * x0[i] + w1[i] * x1[i] + w2[i] * x2[i] +
              w3[i] * x3[i];
    dst[i] = silu_scalar(a);
  }
#else
  for (int i = 0; i < n; ++i) {
    float a = bias[i] + w0[i] * x0[i] + w1[i] * x1[i] + w2[i] * x2[i] +
              w3[i] * x3[i];
    dst[i] = silu_scalar(a);
  }
#endif
}

float v_scale_softmax(float* __restrict s, int n, float scale) {
#if VI_HAS_NEON
  const float32x4_t vscale = vdupq_n_f32(scale);
  float32x4_t vmax = vdupq_n_f32(-INFINITY);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vmulq_f32(vld1q_f32(s + i), vscale);
    vst1q_f32(s + i, v);
    vmax = vmaxq_f32(vmax, v);
  }
  float maxs = vmaxvq_f32(vmax);
  for (; i < n; ++i) {
    s[i] *= scale;
    maxs = std::max(maxs, s[i]);
  }
  const float32x4_t vms = vdupq_n_f32(maxs);
  float32x4_t vsum0 = vdupq_n_f32(0.0f), vsum1 = vdupq_n_f32(0.0f);
  i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t e0 = neon_exp_f32x4(vsubq_f32(vld1q_f32(s + i), vms));
    float32x4_t e1 = neon_exp_f32x4(vsubq_f32(vld1q_f32(s + i + 4), vms));
    vst1q_f32(s + i, e0);
    vst1q_f32(s + i + 4, e1);
    vsum0 = vaddq_f32(vsum0, e0);
    vsum1 = vaddq_f32(vsum1, e1);
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t e0 = neon_exp_f32x4(vsubq_f32(vld1q_f32(s + i), vms));
    vst1q_f32(s + i, e0);
    vsum0 = vaddq_f32(vsum0, e0);
  }
  float sum = vaddvq_f32(vaddq_f32(vsum0, vsum1));
  for (; i < n; ++i) {
    s[i] = std::exp(s[i] - maxs);
    sum += s[i];
  }
  return sum;
#else
  float maxs = -INFINITY;
  for (int i = 0; i < n; ++i) {
    s[i] *= scale;
    maxs = std::max(maxs, s[i]);
  }
  for (int i = 0; i < n; ++i) s[i] -= maxs;
  v_exp(s, n);
  return v_sum(s, n);
#endif
}

void v_softmax_norm(float* __restrict s, int n) {
#if VI_HAS_NEON
  float32x4_t vmax = vdupq_n_f32(-INFINITY);
  int i = 0;
  for (; i + 4 <= n; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(s + i));
  float maxs = vmaxvq_f32(vmax);
  for (; i < n; ++i) maxs = std::max(maxs, s[i]);

  const float32x4_t vms = vdupq_n_f32(maxs);
  float32x4_t vsum0 = vdupq_n_f32(0.0f), vsum1 = vdupq_n_f32(0.0f);
  i = 0;
  for (; i + 8 <= n; i += 8) {
    float32x4_t e0 = neon_exp_f32x4(vsubq_f32(vld1q_f32(s + i), vms));
    float32x4_t e1 = neon_exp_f32x4(vsubq_f32(vld1q_f32(s + i + 4), vms));
    vst1q_f32(s + i, e0);
    vst1q_f32(s + i + 4, e1);
    vsum0 = vaddq_f32(vsum0, e0);
    vsum1 = vaddq_f32(vsum1, e1);
  }
  for (; i + 4 <= n; i += 4) {
    float32x4_t e0 = neon_exp_f32x4(vsubq_f32(vld1q_f32(s + i), vms));
    vst1q_f32(s + i, e0);
    vsum0 = vaddq_f32(vsum0, e0);
  }
  float sum = vaddvq_f32(vaddq_f32(vsum0, vsum1));
  for (; i < n; ++i) {
    s[i] = std::exp(s[i] - maxs);
    sum += s[i];
  }

  const float inv = 1.0f / sum;
  const float32x4_t vinv = vdupq_n_f32(inv);
  i = 0;
  for (; i + 8 <= n; i += 8) {
    vst1q_f32(s + i, vmulq_f32(vld1q_f32(s + i), vinv));
    vst1q_f32(s + i + 4, vmulq_f32(vld1q_f32(s + i + 4), vinv));
  }
  for (; i + 4 <= n; i += 4)
    vst1q_f32(s + i, vmulq_f32(vld1q_f32(s + i), vinv));
  for (; i < n; ++i) s[i] *= inv;
#else
  float maxs = -INFINITY;
  for (int i = 0; i < n; ++i) maxs = std::max(maxs, s[i]);
  for (int i = 0; i < n; ++i) s[i] -= maxs;
  v_exp(s, n);
  const float inv = 1.0f / v_sum(s, n);
  v_scale(s, n, inv);
#endif
}

void v_attn4_qk(const float* __restrict q0, const float* __restrict q1,
                const float* __restrict q2, const float* __restrict q3,
                const float* __restrict keys, int n, int hd, float scale,
                float* __restrict s0, float* __restrict s1,
                float* __restrict s2, float* __restrict s3) {
#if VI_HAS_NEON
  const float32x4_t vscale = vdupq_n_f32(scale);
  int j = 0;
  for (; j + 4 <= n; j += 4) {
    const float* k0 = keys + static_cast<size_t>(j + 0) * hd;
    const float* k1 = keys + static_cast<size_t>(j + 1) * hd;
    const float* k2 = keys + static_cast<size_t>(j + 2) * hd;
    const float* k3 = keys + static_cast<size_t>(j + 3) * hd;
    float32x4_t a00 = vdupq_n_f32(0.0f), a01 = a00, a02 = a00, a03 = a00;
    float32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    float32x4_t a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    float32x4_t a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    int d = 0;
    for (; d + 4 <= hd; d += 4) {
      const float32x4_t vq0 = vld1q_f32(q0 + d);
      const float32x4_t vq1 = vld1q_f32(q1 + d);
      const float32x4_t vq2 = vld1q_f32(q2 + d);
      const float32x4_t vq3 = vld1q_f32(q3 + d);
      const float32x4_t vk0 = vld1q_f32(k0 + d);
      a00 = vfmaq_f32(a00, vq0, vk0);
      a10 = vfmaq_f32(a10, vq1, vk0);
      a20 = vfmaq_f32(a20, vq2, vk0);
      a30 = vfmaq_f32(a30, vq3, vk0);
      const float32x4_t vk1 = vld1q_f32(k1 + d);
      a01 = vfmaq_f32(a01, vq0, vk1);
      a11 = vfmaq_f32(a11, vq1, vk1);
      a21 = vfmaq_f32(a21, vq2, vk1);
      a31 = vfmaq_f32(a31, vq3, vk1);
      const float32x4_t vk2 = vld1q_f32(k2 + d);
      a02 = vfmaq_f32(a02, vq0, vk2);
      a12 = vfmaq_f32(a12, vq1, vk2);
      a22 = vfmaq_f32(a22, vq2, vk2);
      a32 = vfmaq_f32(a32, vq3, vk2);
      const float32x4_t vk3 = vld1q_f32(k3 + d);
      a03 = vfmaq_f32(a03, vq0, vk3);
      a13 = vfmaq_f32(a13, vq1, vk3);
      a23 = vfmaq_f32(a23, vq2, vk3);
      a33 = vfmaq_f32(a33, vq3, vk3);
    }
    vst1q_f32(s0 + j, vmulq_f32(vpaddq_f32(vpaddq_f32(a00, a01), vpaddq_f32(a02, a03)), vscale));
    vst1q_f32(s1 + j, vmulq_f32(vpaddq_f32(vpaddq_f32(a10, a11), vpaddq_f32(a12, a13)), vscale));
    vst1q_f32(s2 + j, vmulq_f32(vpaddq_f32(vpaddq_f32(a20, a21), vpaddq_f32(a22, a23)), vscale));
    vst1q_f32(s3 + j, vmulq_f32(vpaddq_f32(vpaddq_f32(a30, a31), vpaddq_f32(a32, a33)), vscale));
    for (; d < hd; ++d) {
      s0[j + 0] += q0[d] * k0[d] * scale; s0[j + 1] += q0[d] * k1[d] * scale;
      s0[j + 2] += q0[d] * k2[d] * scale; s0[j + 3] += q0[d] * k3[d] * scale;
      s1[j + 0] += q1[d] * k0[d] * scale; s1[j + 1] += q1[d] * k1[d] * scale;
      s1[j + 2] += q1[d] * k2[d] * scale; s1[j + 3] += q1[d] * k3[d] * scale;
      s2[j + 0] += q2[d] * k0[d] * scale; s2[j + 1] += q2[d] * k1[d] * scale;
      s2[j + 2] += q2[d] * k2[d] * scale; s2[j + 3] += q2[d] * k3[d] * scale;
      s3[j + 0] += q3[d] * k0[d] * scale; s3[j + 1] += q3[d] * k1[d] * scale;
      s3[j + 2] += q3[d] * k2[d] * scale; s3[j + 3] += q3[d] * k3[d] * scale;
    }
  }
  for (; j < n; ++j) {
    const float* kj = keys + static_cast<size_t>(j) * hd;
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    int d = 0;
    for (; d + 4 <= hd; d += 4) {
      const float32x4_t vk = vld1q_f32(kj + d);
      a0 = vfmaq_f32(a0, vld1q_f32(q0 + d), vk);
      a1 = vfmaq_f32(a1, vld1q_f32(q1 + d), vk);
      a2 = vfmaq_f32(a2, vld1q_f32(q2 + d), vk);
      a3 = vfmaq_f32(a3, vld1q_f32(q3 + d), vk);
    }
    float r0 = vaddvq_f32(a0), r1 = vaddvq_f32(a1), r2 = vaddvq_f32(a2),
          r3 = vaddvq_f32(a3);
    for (; d < hd; ++d) {
      r0 += q0[d] * kj[d]; r1 += q1[d] * kj[d];
      r2 += q2[d] * kj[d]; r3 += q3[d] * kj[d];
    }
    s0[j] = r0 * scale; s1[j] = r1 * scale;
    s2[j] = r2 * scale; s3[j] = r3 * scale;
  }
#else
  for (int j = 0; j < n; ++j) {
    const float* kj = keys + static_cast<size_t>(j) * hd;
    float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    for (int d = 0; d < hd; ++d) {
      r0 += q0[d] * kj[d]; r1 += q1[d] * kj[d];
      r2 += q2[d] * kj[d]; r3 += q3[d] * kj[d];
    }
    s0[j] = r0 * scale; s1[j] = r1 * scale;
    s2[j] = r2 * scale; s3[j] = r3 * scale;
  }
#endif
}

void v_attn4_pv(const float* __restrict p0, const float* __restrict p1,
                const float* __restrict p2, const float* __restrict p3,
                const float* __restrict values, int n, int hd,
                float* __restrict o0, float* __restrict o1,
                float* __restrict o2, float* __restrict o3) {
#if VI_HAS_NEON
  if (hd % 16 == 0) {
    constexpr int kJTile = 128;
    for (int j0 = 0; j0 < n; j0 += kJTile) {
      const int j1 = std::min(n, j0 + kJTile);
      for (int d = 0; d < hd; d += 16) {
        float32x4_t a00, a01, a02, a03;
        float32x4_t a10, a11, a12, a13;
        float32x4_t a20, a21, a22, a23;
        float32x4_t a30, a31, a32, a33;
        if (j0 == 0) {
          const float32x4_t z = vdupq_n_f32(0.0f);
          a00 = z; a01 = z; a02 = z; a03 = z;
          a10 = z; a11 = z; a12 = z; a13 = z;
          a20 = z; a21 = z; a22 = z; a23 = z;
          a30 = z; a31 = z; a32 = z; a33 = z;
        } else {
          a00 = vld1q_f32(o0 + d + 0);  a01 = vld1q_f32(o0 + d + 4);
          a02 = vld1q_f32(o0 + d + 8);  a03 = vld1q_f32(o0 + d + 12);
          a10 = vld1q_f32(o1 + d + 0);  a11 = vld1q_f32(o1 + d + 4);
          a12 = vld1q_f32(o1 + d + 8);  a13 = vld1q_f32(o1 + d + 12);
          a20 = vld1q_f32(o2 + d + 0);  a21 = vld1q_f32(o2 + d + 4);
          a22 = vld1q_f32(o2 + d + 8);  a23 = vld1q_f32(o2 + d + 12);
          a30 = vld1q_f32(o3 + d + 0);  a31 = vld1q_f32(o3 + d + 4);
          a32 = vld1q_f32(o3 + d + 8);  a33 = vld1q_f32(o3 + d + 12);
        }
        const float* vj = values + static_cast<size_t>(j0) * hd + d;
        int j = j0;
        for (; j + 4 <= j1; j += 4) {
          const float32x4_t wp0 = vld1q_f32(p0 + j);
          const float32x4_t wp1 = vld1q_f32(p1 + j);
          const float32x4_t wp2 = vld1q_f32(p2 + j);
          const float32x4_t wp3 = vld1q_f32(p3 + j);
#define PV_STEP(lane)                                                          \
  do {                                                                         \
    const float32x4_t v0 = vld1q_f32(vj + 0);                                  \
    const float32x4_t v1 = vld1q_f32(vj + 4);                                  \
    const float32x4_t v2 = vld1q_f32(vj + 8);                                  \
    const float32x4_t v3 = vld1q_f32(vj + 12);                                 \
    a00 = vfmaq_laneq_f32(a00, v0, wp0, lane);                                 \
    a01 = vfmaq_laneq_f32(a01, v1, wp0, lane);                                 \
    a02 = vfmaq_laneq_f32(a02, v2, wp0, lane);                                 \
    a03 = vfmaq_laneq_f32(a03, v3, wp0, lane);                                 \
    a10 = vfmaq_laneq_f32(a10, v0, wp1, lane);                                 \
    a11 = vfmaq_laneq_f32(a11, v1, wp1, lane);                                 \
    a12 = vfmaq_laneq_f32(a12, v2, wp1, lane);                                 \
    a13 = vfmaq_laneq_f32(a13, v3, wp1, lane);                                 \
    a20 = vfmaq_laneq_f32(a20, v0, wp2, lane);                                 \
    a21 = vfmaq_laneq_f32(a21, v1, wp2, lane);                                 \
    a22 = vfmaq_laneq_f32(a22, v2, wp2, lane);                                 \
    a23 = vfmaq_laneq_f32(a23, v3, wp2, lane);                                 \
    a30 = vfmaq_laneq_f32(a30, v0, wp3, lane);                                 \
    a31 = vfmaq_laneq_f32(a31, v1, wp3, lane);                                 \
    a32 = vfmaq_laneq_f32(a32, v2, wp3, lane);                                 \
    a33 = vfmaq_laneq_f32(a33, v3, wp3, lane);                                 \
    vj += hd;                                                                  \
  } while (0)
          PV_STEP(0);
          PV_STEP(1);
          PV_STEP(2);
          PV_STEP(3);
#undef PV_STEP
        }
        for (; j < j1; ++j) {
          const float32x4_t v0 = vld1q_f32(vj + 0);
          const float32x4_t v1 = vld1q_f32(vj + 4);
          const float32x4_t v2 = vld1q_f32(vj + 8);
          const float32x4_t v3 = vld1q_f32(vj + 12);
          const float w0 = p0[j], w1 = p1[j], w2 = p2[j], w3 = p3[j];
          a00 = vfmaq_n_f32(a00, v0, w0); a01 = vfmaq_n_f32(a01, v1, w0);
          a02 = vfmaq_n_f32(a02, v2, w0); a03 = vfmaq_n_f32(a03, v3, w0);
          a10 = vfmaq_n_f32(a10, v0, w1); a11 = vfmaq_n_f32(a11, v1, w1);
          a12 = vfmaq_n_f32(a12, v2, w1); a13 = vfmaq_n_f32(a13, v3, w1);
          a20 = vfmaq_n_f32(a20, v0, w2); a21 = vfmaq_n_f32(a21, v1, w2);
          a22 = vfmaq_n_f32(a22, v2, w2); a23 = vfmaq_n_f32(a23, v3, w2);
          a30 = vfmaq_n_f32(a30, v0, w3); a31 = vfmaq_n_f32(a31, v1, w3);
          a32 = vfmaq_n_f32(a32, v2, w3); a33 = vfmaq_n_f32(a33, v3, w3);
          vj += hd;
        }
        vst1q_f32(o0 + d + 0, a00);  vst1q_f32(o0 + d + 4, a01);
        vst1q_f32(o0 + d + 8, a02);  vst1q_f32(o0 + d + 12, a03);
        vst1q_f32(o1 + d + 0, a10);  vst1q_f32(o1 + d + 4, a11);
        vst1q_f32(o1 + d + 8, a12);  vst1q_f32(o1 + d + 12, a13);
        vst1q_f32(o2 + d + 0, a20);  vst1q_f32(o2 + d + 4, a21);
        vst1q_f32(o2 + d + 8, a22);  vst1q_f32(o2 + d + 12, a23);
        vst1q_f32(o3 + d + 0, a30);  vst1q_f32(o3 + d + 4, a31);
        vst1q_f32(o3 + d + 8, a32);  vst1q_f32(o3 + d + 12, a33);
      }
    }
    return;
  }
#endif
  for (int d = 0; d < hd; ++d) { o0[d] = o1[d] = o2[d] = o3[d] = 0.0f; }
  for (int j = 0; j < n; ++j) {
    const float* vj = values + static_cast<size_t>(j) * hd;
    const float w0 = p0[j], w1 = p1[j], w2 = p2[j], w3 = p3[j];
    for (int d = 0; d < hd; ++d) {
      const float v = vj[d];
      o0[d] += w0 * v; o1[d] += w1 * v;
      o2[d] += w2 * v; o3[d] += w3 * v;
    }
  }
}

void v_linear_4rows(const float* __restrict a, int k, const float* __restrict w,
                    int n, const float* __restrict bias, float* __restrict c) {
  const float* a0 = a + 0 * k;
  const float* a1 = a + 1 * k;
  const float* a2 = a + 2 * k;
  const float* a3 = a + 3 * k;
  float* c0 = c + 0 * n;
  float* c1 = c + 1 * n;
  float* c2 = c + 2 * n;
  float* c3 = c + 3 * n;
#if VI_HAS_NEON
  int j = 0;
  for (; j + 4 <= n; j += 4) {
    const float* w0 = w + static_cast<size_t>(j + 0) * k;
    const float* w1 = w + static_cast<size_t>(j + 1) * k;
    const float* w2 = w + static_cast<size_t>(j + 2) * k;
    const float* w3 = w + static_cast<size_t>(j + 3) * k;
    float32x4_t s00 = vdupq_n_f32(0.0f), s01 = s00, s02 = s00, s03 = s00;
    float32x4_t s10 = s00, s11 = s00, s12 = s00, s13 = s00;
    float32x4_t s20 = s00, s21 = s00, s22 = s00, s23 = s00;
    float32x4_t s30 = s00, s31 = s00, s32 = s00, s33 = s00;
    int d = 0;
    for (; d + 4 <= k; d += 4) {
      const float32x4_t va0 = vld1q_f32(a0 + d);
      const float32x4_t va1 = vld1q_f32(a1 + d);
      const float32x4_t va2 = vld1q_f32(a2 + d);
      const float32x4_t va3 = vld1q_f32(a3 + d);
      const float32x4_t vw0 = vld1q_f32(w0 + d);
      s00 = vfmaq_f32(s00, va0, vw0); s10 = vfmaq_f32(s10, va1, vw0);
      s20 = vfmaq_f32(s20, va2, vw0); s30 = vfmaq_f32(s30, va3, vw0);
      const float32x4_t vw1 = vld1q_f32(w1 + d);
      s01 = vfmaq_f32(s01, va0, vw1); s11 = vfmaq_f32(s11, va1, vw1);
      s21 = vfmaq_f32(s21, va2, vw1); s31 = vfmaq_f32(s31, va3, vw1);
      const float32x4_t vw2 = vld1q_f32(w2 + d);
      s02 = vfmaq_f32(s02, va0, vw2); s12 = vfmaq_f32(s12, va1, vw2);
      s22 = vfmaq_f32(s22, va2, vw2); s32 = vfmaq_f32(s32, va3, vw2);
      const float32x4_t vw3 = vld1q_f32(w3 + d);
      s03 = vfmaq_f32(s03, va0, vw3); s13 = vfmaq_f32(s13, va1, vw3);
      s23 = vfmaq_f32(s23, va2, vw3); s33 = vfmaq_f32(s33, va3, vw3);
    }
    const float32x4_t vb = bias ? vld1q_f32(bias + j) : vdupq_n_f32(0.0f);
    vst1q_f32(c0 + j, vaddq_f32(vpaddq_f32(vpaddq_f32(s00, s01), vpaddq_f32(s02, s03)), vb));
    vst1q_f32(c1 + j, vaddq_f32(vpaddq_f32(vpaddq_f32(s10, s11), vpaddq_f32(s12, s13)), vb));
    vst1q_f32(c2 + j, vaddq_f32(vpaddq_f32(vpaddq_f32(s20, s21), vpaddq_f32(s22, s23)), vb));
    vst1q_f32(c3 + j, vaddq_f32(vpaddq_f32(vpaddq_f32(s30, s31), vpaddq_f32(s32, s33)), vb));
    for (; d < k; ++d) {
      c0[j+0] += a0[d]*w0[d]; c0[j+1] += a0[d]*w1[d]; c0[j+2] += a0[d]*w2[d]; c0[j+3] += a0[d]*w3[d];
      c1[j+0] += a1[d]*w0[d]; c1[j+1] += a1[d]*w1[d]; c1[j+2] += a1[d]*w2[d]; c1[j+3] += a1[d]*w3[d];
      c2[j+0] += a2[d]*w0[d]; c2[j+1] += a2[d]*w1[d]; c2[j+2] += a2[d]*w2[d]; c2[j+3] += a2[d]*w3[d];
      c3[j+0] += a3[d]*w0[d]; c3[j+1] += a3[d]*w1[d]; c3[j+2] += a3[d]*w2[d]; c3[j+3] += a3[d]*w3[d];
    }
  }
  for (; j < n; ++j) {
    const float* wj = w + static_cast<size_t>(j) * k;
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    int d = 0;
    for (; d + 4 <= k; d += 4) {
      const float32x4_t vw = vld1q_f32(wj + d);
      s0 = vfmaq_f32(s0, vld1q_f32(a0 + d), vw);
      s1 = vfmaq_f32(s1, vld1q_f32(a1 + d), vw);
      s2 = vfmaq_f32(s2, vld1q_f32(a2 + d), vw);
      s3 = vfmaq_f32(s3, vld1q_f32(a3 + d), vw);
    }
    const float b = bias ? bias[j] : 0.0f;
    float r0 = vaddvq_f32(s0) + b, r1 = vaddvq_f32(s1) + b,
          r2 = vaddvq_f32(s2) + b, r3 = vaddvq_f32(s3) + b;
    for (; d < k; ++d) {
      r0 += a0[d]*wj[d]; r1 += a1[d]*wj[d]; r2 += a2[d]*wj[d]; r3 += a3[d]*wj[d];
    }
    c0[j] = r0; c1[j] = r1; c2[j] = r2; c3[j] = r3;
  }
#else
  for (int j = 0; j < n; ++j) {
    const float* wj = w + static_cast<size_t>(j) * k;
    const float b = bias ? bias[j] : 0.0f;
    float r0 = b, r1 = b, r2 = b, r3 = b;
    for (int d = 0; d < k; ++d) {
      r0 += a0[d]*wj[d]; r1 += a1[d]*wj[d]; r2 += a2[d]*wj[d]; r3 += a3[d]*wj[d];
    }
    c0[j] = r0; c1[j] = r1; c2[j] = r2; c3[j] = r3;
  }
#endif
}

void v_quant_pack_a_i8_block(const float* __restrict a, int m0, int mr, int k,
                             int lda, int8_t* __restrict ap_block,
                             float* __restrict scales) {
  const int k4 = k / 4;
  uint32_t* dst32 = reinterpret_cast<uint32_t*>(ap_block);
  float inv_s[16];
#if VI_HAS_NEON
  for (int r = 0; r < mr; ++r) {
    const float* row = a + static_cast<size_t>(m0 + r) * lda;
    float32x4_t mx0 = vdupq_n_f32(0.0f), mx1 = mx0, mx2 = mx0, mx3 = mx0;
    int kk = 0;
    for (; kk + 16 <= k; kk += 16) {
      mx0 = vmaxq_f32(mx0, vabsq_f32(vld1q_f32(row + kk + 0)));
      mx1 = vmaxq_f32(mx1, vabsq_f32(vld1q_f32(row + kk + 4)));
      mx2 = vmaxq_f32(mx2, vabsq_f32(vld1q_f32(row + kk + 8)));
      mx3 = vmaxq_f32(mx3, vabsq_f32(vld1q_f32(row + kk + 12)));
    }
    float peak = vmaxvq_f32(vmaxq_f32(vmaxq_f32(mx0, mx1), vmaxq_f32(mx2, mx3)));
    for (; kk < k; ++kk) peak = std::max(peak, std::fabs(row[kk]));
    const float s = peak > 0.0f ? peak / 127.0f : 1.0f;
    scales[m0 + r] = s;
    inv_s[r] = 1.0f / s;
  }
  const float32x4_t vlo = vdupq_n_f32(-127.0f), vhi = vdupq_n_f32(127.0f);
  auto quant16 = [&](const float* p, float32x4_t vs) -> uint32x4_t {
    int32x4_t q0 = vcvtaq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(p + 0), vs), vlo), vhi));
    int32x4_t q1 = vcvtaq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(p + 4), vs), vlo), vhi));
    int32x4_t q2 = vcvtaq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(p + 8), vs), vlo), vhi));
    int32x4_t q3 = vcvtaq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(p + 12), vs), vlo), vhi));
    int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    return vreinterpretq_u32_s8(vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
  };
  int r = 0;
  for (; r + 4 <= mr; r += 4) {
    const float* p0 = a + static_cast<size_t>(m0 + r + 0) * lda;
    const float* p1 = a + static_cast<size_t>(m0 + r + 1) * lda;
    const float* p2 = a + static_cast<size_t>(m0 + r + 2) * lda;
    const float* p3 = a + static_cast<size_t>(m0 + r + 3) * lda;
    const float32x4_t vs0 = vdupq_n_f32(inv_s[r + 0]);
    const float32x4_t vs1 = vdupq_n_f32(inv_s[r + 1]);
    const float32x4_t vs2 = vdupq_n_f32(inv_s[r + 2]);
    const float32x4_t vs3 = vdupq_n_f32(inv_s[r + 3]);
    int kk = 0;
    for (; kk + 16 <= k; kk += 16) {
      const uint32x4_t u0 = quant16(p0 + kk, vs0);
      const uint32x4_t u1 = quant16(p1 + kk, vs1);
      const uint32x4_t u2 = quant16(p2 + kk, vs2);
      const uint32x4_t u3 = quant16(p3 + kk, vs3);
      // 4x4 uint32 transpose
      const uint32x4_t a0 = vtrn1q_u32(u0, u1);
      const uint32x4_t a1 = vtrn2q_u32(u0, u1);
      const uint32x4_t a2 = vtrn1q_u32(u2, u3);
      const uint32x4_t a3 = vtrn2q_u32(u2, u3);
      const uint32x4_t t0 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a2)));
      const uint32x4_t t1 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(a1), vreinterpretq_u64_u32(a3)));
      const uint32x4_t t2 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a2)));
      const uint32x4_t t3 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(a1), vreinterpretq_u64_u32(a3)));
      const int kk4 = kk / 4;
      vst1q_u32(dst32 + static_cast<size_t>(kk4 + 0) * 16 + r, t0);
      vst1q_u32(dst32 + static_cast<size_t>(kk4 + 1) * 16 + r, t1);
      vst1q_u32(dst32 + static_cast<size_t>(kk4 + 2) * 16 + r, t2);
      vst1q_u32(dst32 + static_cast<size_t>(kk4 + 3) * 16 + r, t3);
    }
    for (; kk < k; kk += 4) {
      const int kk4 = kk / 4;
      for (int u = 0; u < 4; ++u) {
        int8_t* dst = ap_block + (static_cast<size_t>(kk4) * 16 + r + u) * 4;
        const float* pr = a + static_cast<size_t>(m0 + r + u) * lda + kk;
        for (int t = 0; t < 4; ++t) {
          int q = static_cast<int>(std::lround(pr[t] * inv_s[r + u]));
          dst[t] = static_cast<int8_t>(std::min(127, std::max(-127, q)));
        }
      }
    }
  }
  for (; r < mr; ++r) {
    const float* pr = a + static_cast<size_t>(m0 + r) * lda;
    const float32x4_t vs = vdupq_n_f32(inv_s[r]);
    int kk = 0;
    for (; kk + 16 <= k; kk += 16) {
      const uint32x4_t u = quant16(pr + kk, vs);
      const int kk4 = kk / 4;
      dst32[static_cast<size_t>(kk4 + 0) * 16 + r] = vgetq_lane_u32(u, 0);
      dst32[static_cast<size_t>(kk4 + 1) * 16 + r] = vgetq_lane_u32(u, 1);
      dst32[static_cast<size_t>(kk4 + 2) * 16 + r] = vgetq_lane_u32(u, 2);
      dst32[static_cast<size_t>(kk4 + 3) * 16 + r] = vgetq_lane_u32(u, 3);
    }
    for (; kk < k; kk += 4) {
      int8_t* dst = ap_block + (static_cast<size_t>(kk / 4) * 16 + r) * 4;
      for (int t = 0; t < 4; ++t) {
        int q = static_cast<int>(std::lround(pr[kk + t] * inv_s[r]));
        dst[t] = static_cast<int8_t>(std::min(127, std::max(-127, q)));
      }
    }
  }
  if (mr < 16) {
    for (int kk4 = 0; kk4 < k4; ++kk4)
      for (int rr = mr; rr < 16; ++rr)
        dst32[static_cast<size_t>(kk4) * 16 + rr] = 0;
  }
#else
  for (int r = 0; r < mr; ++r) {
    const float* row = a + static_cast<size_t>(m0 + r) * lda;
    float peak = 0.0f;
    for (int kk = 0; kk < k; ++kk) peak = std::max(peak, std::fabs(row[kk]));
    const float s = peak > 0.0f ? peak / 127.0f : 1.0f;
    scales[m0 + r] = s;
    const float inv = 1.0f / s;
    for (int kk = 0; kk < k; kk += 4) {
      int8_t* dst = ap_block + (static_cast<size_t>(kk / 4) * 16 + r) * 4;
      for (int t = 0; t < 4; ++t) {
        int q = static_cast<int>(std::lround(row[kk + t] * inv));
        dst[t] = static_cast<int8_t>(std::min(127, std::max(-127, q)));
      }
    }
  }
  if (mr < 16) {
    for (int kk4 = 0; kk4 < k4; ++kk4)
      for (int rr = mr; rr < 16; ++rr)
        dst32[static_cast<size_t>(kk4) * 16 + rr] = 0;
  }
#endif
}

void v_pack_a_f32_block(const float* __restrict a, int m0, int mr, int k,
                        int lda, float* __restrict ap_block) {
#if VI_HAS_NEON
  int r = 0;
  for (; r + 4 <= mr; r += 4) {
    const float* p0 = a + static_cast<size_t>(m0 + r + 0) * lda;
    const float* p1 = a + static_cast<size_t>(m0 + r + 1) * lda;
    const float* p2 = a + static_cast<size_t>(m0 + r + 2) * lda;
    const float* p3 = a + static_cast<size_t>(m0 + r + 3) * lda;
    int kk = 0;
    for (; kk + 4 <= k; kk += 4) {
      const float32x4_t u0 = vld1q_f32(p0 + kk);
      const float32x4_t u1 = vld1q_f32(p1 + kk);
      const float32x4_t u2 = vld1q_f32(p2 + kk);
      const float32x4_t u3 = vld1q_f32(p3 + kk);
      const float32x4_t a0 = vtrn1q_f32(u0, u1);
      const float32x4_t a1 = vtrn2q_f32(u0, u1);
      const float32x4_t a2 = vtrn1q_f32(u2, u3);
      const float32x4_t a3 = vtrn2q_f32(u2, u3);
      const float32x4_t t0 = vreinterpretq_f32_f64(
          vtrn1q_f64(vreinterpretq_f64_f32(a0), vreinterpretq_f64_f32(a2)));
      const float32x4_t t1 = vreinterpretq_f32_f64(
          vtrn1q_f64(vreinterpretq_f64_f32(a1), vreinterpretq_f64_f32(a3)));
      const float32x4_t t2 = vreinterpretq_f32_f64(
          vtrn2q_f64(vreinterpretq_f64_f32(a0), vreinterpretq_f64_f32(a2)));
      const float32x4_t t3 = vreinterpretq_f32_f64(
          vtrn2q_f64(vreinterpretq_f64_f32(a1), vreinterpretq_f64_f32(a3)));
      vst1q_f32(ap_block + static_cast<size_t>(kk + 0) * 16 + r, t0);
      vst1q_f32(ap_block + static_cast<size_t>(kk + 1) * 16 + r, t1);
      vst1q_f32(ap_block + static_cast<size_t>(kk + 2) * 16 + r, t2);
      vst1q_f32(ap_block + static_cast<size_t>(kk + 3) * 16 + r, t3);
    }
    for (; kk < k; ++kk) {
      ap_block[static_cast<size_t>(kk) * 16 + r + 0] = p0[kk];
      ap_block[static_cast<size_t>(kk) * 16 + r + 1] = p1[kk];
      ap_block[static_cast<size_t>(kk) * 16 + r + 2] = p2[kk];
      ap_block[static_cast<size_t>(kk) * 16 + r + 3] = p3[kk];
    }
  }
  for (; r < mr; ++r) {
    const float* pr = a + static_cast<size_t>(m0 + r) * lda;
    for (int kk = 0; kk < k; ++kk)
      ap_block[static_cast<size_t>(kk) * 16 + r] = pr[kk];
  }
  if (mr < 16) {
    for (int kk = 0; kk < k; ++kk)
      for (int rr = mr; rr < 16; ++rr)
        ap_block[static_cast<size_t>(kk) * 16 + rr] = 0.0f;
  }
#else
  for (int r = 0; r < mr; ++r) {
    const float* pr = a + static_cast<size_t>(m0 + r) * lda;
    for (int kk = 0; kk < k; ++kk)
      ap_block[static_cast<size_t>(kk) * 16 + r] = pr[kk];
  }
  if (mr < 16) {
    for (int kk = 0; kk < k; ++kk)
      for (int rr = mr; rr < 16; ++rr)
        ap_block[static_cast<size_t>(kk) * 16 + rr] = 0.0f;
  }
#endif
}

void v_pack_a_f16_block(const float* __restrict a, int m0, int mr, int k,
                        int lda, uint16_t* __restrict ap_block) {
  uint32_t* dst32 = reinterpret_cast<uint32_t*>(ap_block);
  const int k2 = k / 2;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  auto cvt8 = [](const float* p) -> uint32x4_t {
    const float16x4_t lo = vcvt_f16_f32(vld1q_f32(p));
    const float16x4_t hi = vcvt_f16_f32(vld1q_f32(p + 4));
    return vreinterpretq_u32_f16(vcombine_f16(lo, hi));
  };
  int r = 0;
  for (; r + 4 <= mr; r += 4) {
    const float* p0 = a + static_cast<size_t>(m0 + r + 0) * lda;
    const float* p1 = a + static_cast<size_t>(m0 + r + 1) * lda;
    const float* p2 = a + static_cast<size_t>(m0 + r + 2) * lda;
    const float* p3 = a + static_cast<size_t>(m0 + r + 3) * lda;
    int kk = 0;
    for (; kk + 8 <= k; kk += 8) {
      const uint32x4_t u0 = cvt8(p0 + kk);
      const uint32x4_t u1 = cvt8(p1 + kk);
      const uint32x4_t u2 = cvt8(p2 + kk);
      const uint32x4_t u3 = cvt8(p3 + kk);
      const uint32x4_t a0 = vtrn1q_u32(u0, u1);
      const uint32x4_t a1 = vtrn2q_u32(u0, u1);
      const uint32x4_t a2 = vtrn1q_u32(u2, u3);
      const uint32x4_t a3 = vtrn2q_u32(u2, u3);
      const uint32x4_t t0 = vreinterpretq_u32_u64(
          vtrn1q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a2)));
      const uint32x4_t t1 = vreinterpretq_u32_u64(
          vtrn1q_u64(vreinterpretq_u64_u32(a1), vreinterpretq_u64_u32(a3)));
      const uint32x4_t t2 = vreinterpretq_u32_u64(
          vtrn2q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a2)));
      const uint32x4_t t3 = vreinterpretq_u32_u64(
          vtrn2q_u64(vreinterpretq_u64_u32(a1), vreinterpretq_u64_u32(a3)));
      const int kk2 = kk / 2;
      vst1q_u32(dst32 + static_cast<size_t>(kk2 + 0) * 16 + r, t0);
      vst1q_u32(dst32 + static_cast<size_t>(kk2 + 1) * 16 + r, t1);
      vst1q_u32(dst32 + static_cast<size_t>(kk2 + 2) * 16 + r, t2);
      vst1q_u32(dst32 + static_cast<size_t>(kk2 + 3) * 16 + r, t3);
    }
    for (; kk < k; kk += 2) {
      const int kk2 = kk / 2;
      __fp16* d = reinterpret_cast<__fp16*>(ap_block);
      d[(static_cast<size_t>(kk2) * 16 + r + 0) * 2 + 0] = static_cast<__fp16>(p0[kk]);
      d[(static_cast<size_t>(kk2) * 16 + r + 0) * 2 + 1] = static_cast<__fp16>(p0[kk + 1]);
      d[(static_cast<size_t>(kk2) * 16 + r + 1) * 2 + 0] = static_cast<__fp16>(p1[kk]);
      d[(static_cast<size_t>(kk2) * 16 + r + 1) * 2 + 1] = static_cast<__fp16>(p1[kk + 1]);
      d[(static_cast<size_t>(kk2) * 16 + r + 2) * 2 + 0] = static_cast<__fp16>(p2[kk]);
      d[(static_cast<size_t>(kk2) * 16 + r + 2) * 2 + 1] = static_cast<__fp16>(p2[kk + 1]);
      d[(static_cast<size_t>(kk2) * 16 + r + 3) * 2 + 0] = static_cast<__fp16>(p3[kk]);
      d[(static_cast<size_t>(kk2) * 16 + r + 3) * 2 + 1] = static_cast<__fp16>(p3[kk + 1]);
    }
  }
  for (; r < mr; ++r) {
    const float* pr = a + static_cast<size_t>(m0 + r) * lda;
    int kk = 0;
    for (; kk + 8 <= k; kk += 8) {
      const uint32x4_t u = cvt8(pr + kk);
      const int kk2 = kk / 2;
      dst32[static_cast<size_t>(kk2 + 0) * 16 + r] = vgetq_lane_u32(u, 0);
      dst32[static_cast<size_t>(kk2 + 1) * 16 + r] = vgetq_lane_u32(u, 1);
      dst32[static_cast<size_t>(kk2 + 2) * 16 + r] = vgetq_lane_u32(u, 2);
      dst32[static_cast<size_t>(kk2 + 3) * 16 + r] = vgetq_lane_u32(u, 3);
    }
    __fp16* d = reinterpret_cast<__fp16*>(ap_block);
    for (; kk < k; kk += 2) {
      const int kk2 = kk / 2;
      d[(static_cast<size_t>(kk2) * 16 + r) * 2 + 0] = static_cast<__fp16>(pr[kk]);
      d[(static_cast<size_t>(kk2) * 16 + r) * 2 + 1] = static_cast<__fp16>(pr[kk + 1]);
    }
  }
  if (mr < 16) {
    for (int kk2 = 0; kk2 < k2; ++kk2)
      for (int rr = mr; rr < 16; ++rr)
        dst32[static_cast<size_t>(kk2) * 16 + rr] = 0;
  }
#else
  (void)a; (void)m0; (void)mr; (void)k; (void)lda; (void)dst32; (void)k2;
#endif
}

void v_pack_a_bf16_block(const float* __restrict a, int m0, int mr, int k,
                         int lda, uint16_t* __restrict ap_block) {
  uint32_t* dst32 = reinterpret_cast<uint32_t*>(ap_block);
  const int k2 = k / 2;
#if VI_HAS_NEON
  const uint32x4_t vbias = vdupq_n_u32(0x7FFFu);
  const uint32x4_t vone = vdupq_n_u32(1u);
  auto cvt8_bf16 = [&](const float* p) -> uint32x4_t {
    const uint32x4_t u0 = vreinterpretq_u32_f32(vld1q_f32(p));
    const uint32x4_t u1 = vreinterpretq_u32_f32(vld1q_f32(p + 4));
    const uint32x4_t r0 = vaddq_u32(u0, vaddq_u32(vbias, vandq_u32(vshrq_n_u32(u0, 16), vone)));
    const uint32x4_t r1 = vaddq_u32(u1, vaddq_u32(vbias, vandq_u32(vshrq_n_u32(u1, 16), vone)));
    return vreinterpretq_u32_u16(vcombine_u16(vshrn_n_u32(r0, 16), vshrn_n_u32(r1, 16)));
  };
  int r = 0;
  for (; r + 4 <= mr; r += 4) {
    const float* p0 = a + static_cast<size_t>(m0 + r + 0) * lda;
    const float* p1 = a + static_cast<size_t>(m0 + r + 1) * lda;
    const float* p2 = a + static_cast<size_t>(m0 + r + 2) * lda;
    const float* p3 = a + static_cast<size_t>(m0 + r + 3) * lda;
    int kk = 0;
    for (; kk + 8 <= k; kk += 8) {
      const uint32x4_t u0 = cvt8_bf16(p0 + kk);
      const uint32x4_t u1 = cvt8_bf16(p1 + kk);
      const uint32x4_t u2 = cvt8_bf16(p2 + kk);
      const uint32x4_t u3 = cvt8_bf16(p3 + kk);
      const uint32x4_t a0 = vtrn1q_u32(u0, u1);
      const uint32x4_t a1 = vtrn2q_u32(u0, u1);
      const uint32x4_t a2 = vtrn1q_u32(u2, u3);
      const uint32x4_t a3 = vtrn2q_u32(u2, u3);
      const uint32x4_t t0 = vreinterpretq_u32_u64(
          vtrn1q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a2)));
      const uint32x4_t t1 = vreinterpretq_u32_u64(
          vtrn1q_u64(vreinterpretq_u64_u32(a1), vreinterpretq_u64_u32(a3)));
      const uint32x4_t t2 = vreinterpretq_u32_u64(
          vtrn2q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a2)));
      const uint32x4_t t3 = vreinterpretq_u32_u64(
          vtrn2q_u64(vreinterpretq_u64_u32(a1), vreinterpretq_u64_u32(a3)));
      const int kk2 = kk / 2;
      vst1q_u32(dst32 + static_cast<size_t>(kk2 + 0) * 16 + r, t0);
      vst1q_u32(dst32 + static_cast<size_t>(kk2 + 1) * 16 + r, t1);
      vst1q_u32(dst32 + static_cast<size_t>(kk2 + 2) * 16 + r, t2);
      vst1q_u32(dst32 + static_cast<size_t>(kk2 + 3) * 16 + r, t3);
    }
    for (; kk < k; kk += 2) {
      const int kk2 = kk / 2;
      const float* ps[4] = {p0, p1, p2, p3};
      for (int u = 0; u < 4; ++u) {
        uint32_t x0, x1;
        std::memcpy(&x0, ps[u] + kk, 4);
        std::memcpy(&x1, ps[u] + kk + 1, 4);
        x0 += 0x7FFFu + ((x0 >> 16) & 1u);
        x1 += 0x7FFFu + ((x1 >> 16) & 1u);
        ap_block[(static_cast<size_t>(kk2) * 16 + r + u) * 2 + 0] = static_cast<uint16_t>(x0 >> 16);
        ap_block[(static_cast<size_t>(kk2) * 16 + r + u) * 2 + 1] = static_cast<uint16_t>(x1 >> 16);
      }
    }
  }
  for (; r < mr; ++r) {
    const float* pr = a + static_cast<size_t>(m0 + r) * lda;
    for (int kk = 0; kk < k; kk += 2) {
      uint32_t x0, x1;
      std::memcpy(&x0, pr + kk, 4);
      std::memcpy(&x1, pr + kk + 1, 4);
      x0 += 0x7FFFu + ((x0 >> 16) & 1u);
      x1 += 0x7FFFu + ((x1 >> 16) & 1u);
      ap_block[(static_cast<size_t>(kk / 2) * 16 + r) * 2 + 0] = static_cast<uint16_t>(x0 >> 16);
      ap_block[(static_cast<size_t>(kk / 2) * 16 + r) * 2 + 1] = static_cast<uint16_t>(x1 >> 16);
    }
  }
  if (mr < 16) {
    for (int kk2 = 0; kk2 < k2; ++kk2)
      for (int rr = mr; rr < 16; ++rr)
        dst32[static_cast<size_t>(kk2) * 16 + rr] = 0;
  }
#else
  (void)a; (void)m0; (void)mr; (void)k; (void)lda; (void)dst32; (void)k2;
#endif
}

bool fp16_neon_available() {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  return true;
#else
  return false;
#endif
}

void v_f32_to_f16(const float* __restrict src, uint16_t* __restrict dst, int n) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  __fp16* d = reinterpret_cast<__fp16*>(dst);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float16x4_t lo = vcvt_f16_f32(vld1q_f32(src + i));
    float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + i + 4));
    vst1q_f16(d + i, vcombine_f16(lo, hi));
  }
  for (; i < n; ++i) d[i] = static_cast<__fp16>(src[i]);
#else
  (void)src; (void)dst; (void)n;
#endif
}

void v_rope_f16_inplace(uint16_t* __restrict x_u, int head_dim,
                        const float* __restrict cos,
                        const float* __restrict sin, bool inverse) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  __fp16* x = reinterpret_cast<__fp16*>(x_u);
  const int half = head_dim / 2;
  const float32x4_t sign = vdupq_n_f32(inverse ? -1.0f : 1.0f);
  int i = 0;
  for (; i + 4 <= half; i += 4) {
    const float32x4_t a = vcvt_f32_f16(vld1_f16(x + i));
    const float32x4_t b = vcvt_f32_f16(vld1_f16(x + i + half));
    const float32x4_t c = vld1q_f32(cos + i);
    const float32x4_t s = vmulq_f32(vld1q_f32(sin + i), sign);
    vst1_f16(x + i, vcvt_f16_f32(vfmsq_f32(vmulq_f32(a, c), b, s)));
    vst1_f16(x + i + half, vcvt_f16_f32(vfmaq_f32(vmulq_f32(b, c), a, s)));
  }
  const float sgn = inverse ? -1.0f : 1.0f;
  for (; i < half; ++i) {
    const float a = static_cast<float>(x[i]);
    const float b = static_cast<float>(x[i + half]);
    x[i] = static_cast<__fp16>(a * cos[i] - sgn * b * sin[i]);
    x[i + half] = static_cast<__fp16>(sgn * a * sin[i] + b * cos[i]);
  }
#else
  (void)x_u; (void)head_dim; (void)cos; (void)sin; (void)inverse;
#endif
}

void v_attn4_qk_f16(const float* __restrict q0, const float* __restrict q1,
                    const float* __restrict q2, const float* __restrict q3,
                    const uint16_t* __restrict keys_u, int n, int hd,
                    float scale, float* __restrict s0, float* __restrict s1,
                    float* __restrict s2, float* __restrict s3) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
  const __fp16* keys = reinterpret_cast<const __fp16*>(keys_u);
  alignas(16) __fp16 q16_0[128], q16_1[128], q16_2[128], q16_3[128];
  const float32x4_t vscale = vdupq_n_f32(scale);
  for (int d = 0; d < hd; d += 8) {
    vst1q_f16(q16_0 + d, vcombine_f16(vcvt_f16_f32(vld1q_f32(q0 + d)),
                                      vcvt_f16_f32(vld1q_f32(q0 + d + 4))));
    vst1q_f16(q16_1 + d, vcombine_f16(vcvt_f16_f32(vld1q_f32(q1 + d)),
                                      vcvt_f16_f32(vld1q_f32(q1 + d + 4))));
    vst1q_f16(q16_2 + d, vcombine_f16(vcvt_f16_f32(vld1q_f32(q2 + d)),
                                      vcvt_f16_f32(vld1q_f32(q2 + d + 4))));
    vst1q_f16(q16_3 + d, vcombine_f16(vcvt_f16_f32(vld1q_f32(q3 + d)),
                                      vcvt_f16_f32(vld1q_f32(q3 + d + 4))));
  }
  int j = 0;
  for (; j + 4 <= n; j += 4) {
    const __fp16* k0 = keys + static_cast<size_t>(j + 0) * hd;
    const __fp16* k1 = keys + static_cast<size_t>(j + 1) * hd;
    const __fp16* k2 = keys + static_cast<size_t>(j + 2) * hd;
    const __fp16* k3 = keys + static_cast<size_t>(j + 3) * hd;
    float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a02 = a00, a03 = a00;
    float32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    float32x4_t a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    float32x4_t a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    for (int d = 0; d < hd; d += 8) {
      const float16x8_t vq0 = vld1q_f16(q16_0 + d);
      const float16x8_t vq1 = vld1q_f16(q16_1 + d);
      const float16x8_t vq2 = vld1q_f16(q16_2 + d);
      const float16x8_t vq3 = vld1q_f16(q16_3 + d);
      const float16x8_t vk0 = vld1q_f16(k0 + d);
      a00 = vfmlalq_low_f16(a00, vq0, vk0); a00 = vfmlalq_high_f16(a00, vq0, vk0);
      a10 = vfmlalq_low_f16(a10, vq1, vk0); a10 = vfmlalq_high_f16(a10, vq1, vk0);
      a20 = vfmlalq_low_f16(a20, vq2, vk0); a20 = vfmlalq_high_f16(a20, vq2, vk0);
      a30 = vfmlalq_low_f16(a30, vq3, vk0); a30 = vfmlalq_high_f16(a30, vq3, vk0);
      const float16x8_t vk1 = vld1q_f16(k1 + d);
      a01 = vfmlalq_low_f16(a01, vq0, vk1); a01 = vfmlalq_high_f16(a01, vq0, vk1);
      a11 = vfmlalq_low_f16(a11, vq1, vk1); a11 = vfmlalq_high_f16(a11, vq1, vk1);
      a21 = vfmlalq_low_f16(a21, vq2, vk1); a21 = vfmlalq_high_f16(a21, vq2, vk1);
      a31 = vfmlalq_low_f16(a31, vq3, vk1); a31 = vfmlalq_high_f16(a31, vq3, vk1);
      const float16x8_t vk2 = vld1q_f16(k2 + d);
      a02 = vfmlalq_low_f16(a02, vq0, vk2); a02 = vfmlalq_high_f16(a02, vq0, vk2);
      a12 = vfmlalq_low_f16(a12, vq1, vk2); a12 = vfmlalq_high_f16(a12, vq1, vk2);
      a22 = vfmlalq_low_f16(a22, vq2, vk2); a22 = vfmlalq_high_f16(a22, vq2, vk2);
      a32 = vfmlalq_low_f16(a32, vq3, vk2); a32 = vfmlalq_high_f16(a32, vq3, vk2);
      const float16x8_t vk3 = vld1q_f16(k3 + d);
      a03 = vfmlalq_low_f16(a03, vq0, vk3); a03 = vfmlalq_high_f16(a03, vq0, vk3);
      a13 = vfmlalq_low_f16(a13, vq1, vk3); a13 = vfmlalq_high_f16(a13, vq1, vk3);
      a23 = vfmlalq_low_f16(a23, vq2, vk3); a23 = vfmlalq_high_f16(a23, vq2, vk3);
      a33 = vfmlalq_low_f16(a33, vq3, vk3); a33 = vfmlalq_high_f16(a33, vq3, vk3);
    }
    vst1q_f32(s0 + j, vmulq_f32(vpaddq_f32(vpaddq_f32(a00, a01), vpaddq_f32(a02, a03)), vscale));
    vst1q_f32(s1 + j, vmulq_f32(vpaddq_f32(vpaddq_f32(a10, a11), vpaddq_f32(a12, a13)), vscale));
    vst1q_f32(s2 + j, vmulq_f32(vpaddq_f32(vpaddq_f32(a20, a21), vpaddq_f32(a22, a23)), vscale));
    vst1q_f32(s3 + j, vmulq_f32(vpaddq_f32(vpaddq_f32(a30, a31), vpaddq_f32(a32, a33)), vscale));
  }
  for (; j < n; ++j) {
    const __fp16* kj = keys + static_cast<size_t>(j) * hd;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int d = 0; d < hd; d += 8) {
      const float16x8_t vk = vld1q_f16(kj + d);
      const float16x8_t vq0 = vld1q_f16(q16_0 + d);
      const float16x8_t vq1 = vld1q_f16(q16_1 + d);
      const float16x8_t vq2 = vld1q_f16(q16_2 + d);
      const float16x8_t vq3 = vld1q_f16(q16_3 + d);
      a0 = vfmlalq_low_f16(a0, vq0, vk); a0 = vfmlalq_high_f16(a0, vq0, vk);
      a1 = vfmlalq_low_f16(a1, vq1, vk); a1 = vfmlalq_high_f16(a1, vq1, vk);
      a2 = vfmlalq_low_f16(a2, vq2, vk); a2 = vfmlalq_high_f16(a2, vq2, vk);
      a3 = vfmlalq_low_f16(a3, vq3, vk); a3 = vfmlalq_high_f16(a3, vq3, vk);
    }
    s0[j] = vaddvq_f32(a0) * scale; s1[j] = vaddvq_f32(a1) * scale;
    s2[j] = vaddvq_f32(a2) * scale; s3[j] = vaddvq_f32(a3) * scale;
  }
#else
  (void)q0; (void)q1; (void)q2; (void)q3; (void)keys_u; (void)n; (void)hd;
  (void)scale; (void)s0; (void)s1; (void)s2; (void)s3;
#endif
}

void v_attn4_pv_f16(const float* __restrict p0, const float* __restrict p1,
                    const float* __restrict p2, const float* __restrict p3,
                    const uint16_t* __restrict values_u, int n, int hd,
                    float* __restrict o0, float* __restrict o1,
                    float* __restrict o2, float* __restrict o3) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  const __fp16* values = reinterpret_cast<const __fp16*>(values_u);
  constexpr int kJTile = 256;
  for (int j0 = 0; j0 < n; j0 += kJTile) {
    const int j1 = std::min(n, j0 + kJTile);
    for (int d = 0; d < hd; d += 16) {
      float32x4_t a00, a01, a02, a03;
      float32x4_t a10, a11, a12, a13;
      float32x4_t a20, a21, a22, a23;
      float32x4_t a30, a31, a32, a33;
      if (j0 == 0) {
        const float32x4_t z = vdupq_n_f32(0.0f);
        a00 = z; a01 = z; a02 = z; a03 = z;
        a10 = z; a11 = z; a12 = z; a13 = z;
        a20 = z; a21 = z; a22 = z; a23 = z;
        a30 = z; a31 = z; a32 = z; a33 = z;
      } else {
        a00 = vld1q_f32(o0 + d + 0);  a01 = vld1q_f32(o0 + d + 4);
        a02 = vld1q_f32(o0 + d + 8);  a03 = vld1q_f32(o0 + d + 12);
        a10 = vld1q_f32(o1 + d + 0);  a11 = vld1q_f32(o1 + d + 4);
        a12 = vld1q_f32(o1 + d + 8);  a13 = vld1q_f32(o1 + d + 12);
        a20 = vld1q_f32(o2 + d + 0);  a21 = vld1q_f32(o2 + d + 4);
        a22 = vld1q_f32(o2 + d + 8);  a23 = vld1q_f32(o2 + d + 12);
        a30 = vld1q_f32(o3 + d + 0);  a31 = vld1q_f32(o3 + d + 4);
        a32 = vld1q_f32(o3 + d + 8);  a33 = vld1q_f32(o3 + d + 12);
      }
      const __fp16* vj = values + static_cast<size_t>(j0) * hd + d;
      int j = j0;
      for (; j + 4 <= j1; j += 4) {
        const float32x4_t wp0 = vld1q_f32(p0 + j);
        const float32x4_t wp1 = vld1q_f32(p1 + j);
        const float32x4_t wp2 = vld1q_f32(p2 + j);
        const float32x4_t wp3 = vld1q_f32(p3 + j);
#define PVF16_STEP(lane)                                                       \
  do {                                                                         \
    const float16x8_t h0 = vld1q_f16(vj + 0);                                  \
    const float16x8_t h1 = vld1q_f16(vj + 8);                                  \
    const float32x4_t v0 = vcvt_f32_f16(vget_low_f16(h0));                     \
    const float32x4_t v1 = vcvt_f32_f16(vget_high_f16(h0));                    \
    const float32x4_t v2 = vcvt_f32_f16(vget_low_f16(h1));                     \
    const float32x4_t v3 = vcvt_f32_f16(vget_high_f16(h1));                    \
    a00 = vfmaq_laneq_f32(a00, v0, wp0, lane);                                 \
    a01 = vfmaq_laneq_f32(a01, v1, wp0, lane);                                 \
    a02 = vfmaq_laneq_f32(a02, v2, wp0, lane);                                 \
    a03 = vfmaq_laneq_f32(a03, v3, wp0, lane);                                 \
    a10 = vfmaq_laneq_f32(a10, v0, wp1, lane);                                 \
    a11 = vfmaq_laneq_f32(a11, v1, wp1, lane);                                 \
    a12 = vfmaq_laneq_f32(a12, v2, wp1, lane);                                 \
    a13 = vfmaq_laneq_f32(a13, v3, wp1, lane);                                 \
    a20 = vfmaq_laneq_f32(a20, v0, wp2, lane);                                 \
    a21 = vfmaq_laneq_f32(a21, v1, wp2, lane);                                 \
    a22 = vfmaq_laneq_f32(a22, v2, wp2, lane);                                 \
    a23 = vfmaq_laneq_f32(a23, v3, wp2, lane);                                 \
    a30 = vfmaq_laneq_f32(a30, v0, wp3, lane);                                 \
    a31 = vfmaq_laneq_f32(a31, v1, wp3, lane);                                 \
    a32 = vfmaq_laneq_f32(a32, v2, wp3, lane);                                 \
    a33 = vfmaq_laneq_f32(a33, v3, wp3, lane);                                 \
    vj += hd;                                                                  \
  } while (0)
        PVF16_STEP(0); PVF16_STEP(1); PVF16_STEP(2); PVF16_STEP(3);
#undef PVF16_STEP
      }
      for (; j < j1; ++j) {
        const float16x8_t h0 = vld1q_f16(vj + 0);
        const float16x8_t h1 = vld1q_f16(vj + 8);
        const float32x4_t v0 = vcvt_f32_f16(vget_low_f16(h0));
        const float32x4_t v1 = vcvt_f32_f16(vget_high_f16(h0));
        const float32x4_t v2 = vcvt_f32_f16(vget_low_f16(h1));
        const float32x4_t v3 = vcvt_f32_f16(vget_high_f16(h1));
        const float w0 = p0[j], w1 = p1[j], w2 = p2[j], w3 = p3[j];
        a00 = vfmaq_n_f32(a00, v0, w0); a01 = vfmaq_n_f32(a01, v1, w0);
        a02 = vfmaq_n_f32(a02, v2, w0); a03 = vfmaq_n_f32(a03, v3, w0);
        a10 = vfmaq_n_f32(a10, v0, w1); a11 = vfmaq_n_f32(a11, v1, w1);
        a12 = vfmaq_n_f32(a12, v2, w1); a13 = vfmaq_n_f32(a13, v3, w1);
        a20 = vfmaq_n_f32(a20, v0, w2); a21 = vfmaq_n_f32(a21, v1, w2);
        a22 = vfmaq_n_f32(a22, v2, w2); a23 = vfmaq_n_f32(a23, v3, w2);
        a30 = vfmaq_n_f32(a30, v0, w3); a31 = vfmaq_n_f32(a31, v1, w3);
        a32 = vfmaq_n_f32(a32, v2, w3); a33 = vfmaq_n_f32(a33, v3, w3);
        vj += hd;
      }
      vst1q_f32(o0 + d + 0, a00);  vst1q_f32(o0 + d + 4, a01);
      vst1q_f32(o0 + d + 8, a02);  vst1q_f32(o0 + d + 12, a03);
      vst1q_f32(o1 + d + 0, a10);  vst1q_f32(o1 + d + 4, a11);
      vst1q_f32(o1 + d + 8, a12);  vst1q_f32(o1 + d + 12, a13);
      vst1q_f32(o2 + d + 0, a20);  vst1q_f32(o2 + d + 4, a21);
      vst1q_f32(o2 + d + 8, a22);  vst1q_f32(o2 + d + 12, a23);
      vst1q_f32(o3 + d + 0, a30);  vst1q_f32(o3 + d + 4, a31);
      vst1q_f32(o3 + d + 8, a32);  vst1q_f32(o3 + d + 12, a33);
    }
  }
#else
  (void)p0; (void)p1; (void)p2; (void)p3; (void)values_u; (void)n; (void)hd;
  (void)o0; (void)o1; (void)o2; (void)o3;
#endif
}

float v_attn1_scores_softmax_f16(const float* __restrict q,
                                 const uint16_t* __restrict keys_u, int n,
                                 int hd, float scale, float* __restrict s) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  const __fp16* keys = reinterpret_cast<const __fp16*>(keys_u);
  alignas(16) __fp16 q16[128];
  const float32x4_t vscale = vdupq_n_f32(scale);
  for (int d = 0; d < hd; d += 8) {
    vst1q_f16(q16 + d, vcombine_f16(vcvt_f16_f32(vmulq_f32(vld1q_f32(q + d), vscale)),
                                    vcvt_f16_f32(vmulq_f32(vld1q_f32(q + d + 4), vscale))));
  }
  for (int j = 0; j < n; ++j) {
    const __fp16* kj = keys + static_cast<size_t>(j) * hd;
    float16x8_t acc = vdupq_n_f16(0);
    for (int d = 0; d < hd; d += 8)
      acc = vfmaq_f16(acc, vld1q_f16(q16 + d), vld1q_f16(kj + d));
    s[j] = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(acc)),
                                vcvt_f32_f16(vget_high_f16(acc))));
  }
  return v_scale_softmax(s, n, 1.0f);
#else
  (void)q; (void)keys_u; (void)n; (void)hd; (void)scale; (void)s;
  return 1.0f;
#endif
}

void v_attn1_pv_f16(const float* __restrict s, float inv,
                    const uint16_t* __restrict values_u, int n, int hd,
                    float* __restrict o) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  const __fp16* values = reinterpret_cast<const __fp16*>(values_u);
  for (int d = 0; d < hd; d += 16) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int j = 0; j < n; ++j) {
      const float w = s[j] * inv;
      const __fp16* vj = values + static_cast<size_t>(j) * hd + d;
      const float16x8_t h0 = vld1q_f16(vj + 0);
      const float16x8_t h1 = vld1q_f16(vj + 8);
      a0 = vfmaq_n_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), w);
      a1 = vfmaq_n_f32(a1, vcvt_f32_f16(vget_high_f16(h0)), w);
      a2 = vfmaq_n_f32(a2, vcvt_f32_f16(vget_low_f16(h1)), w);
      a3 = vfmaq_n_f32(a3, vcvt_f32_f16(vget_high_f16(h1)), w);
    }
    vst1q_f32(o + d + 0, a0);  vst1q_f32(o + d + 4, a1);
    vst1q_f32(o + d + 8, a2);  vst1q_f32(o + d + 12, a3);
  }
#else
  (void)s; (void)inv; (void)values_u; (void)n; (void)hd; (void)o;
#endif
}

}  // namespace vi
