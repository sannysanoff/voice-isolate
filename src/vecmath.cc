#include "vecmath.h"

#include <cmath>

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

}  // namespace

void v_exp(float* x, int n) {
  elementwise(x, n, [](vx::batch v) { return xsimd::exp(v); });
}

void v_tanh(float* x, int n) {
  elementwise(x, n, [](vx::batch v) { return xsimd::tanh(v); });
}

void v_gelu(float* x, int n) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  elementwise(x, n, [](vx::batch v) {
    return 0.5f * v * (vx::batch(1.0f) + xsimd::erf(v * vx::batch(kInvSqrt2)));
  });
}

void v_silu(float* x, int n) {
  elementwise(x, n, [](vx::batch v) {
    return v / (vx::batch(1.0f) + xsimd::exp(-v));
  });
}

void v_softplus(float* x, int n) {
  elementwise(x, n, [](vx::batch v) {
    // log1p(exp(v)), or v itself above 20 (torch's threshold).
    const auto linear = v;
    const auto curved = xsimd::log1p(xsimd::exp(v));
    return xsimd::select(v > vx::batch(20.0f), linear, curved);
  });
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

void v_mul_exp(float* out, const float* a, const float* scale, int n) {
  const int i_end = n - (n % vx::lanes);
  for (int i = 0; i < i_end; i += vx::lanes)
    store(out + i, xsimd::exp(load(a + i) * load(scale + i)));
  for (int i = i_end; i < n; ++i) out[i] = std::exp(a[i] * scale[i]);
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

}  // namespace vi
