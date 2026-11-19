// Vector math over arrays, on top of xsimd's portable SIMD kernels.
#pragma once

#include <cstdint>

#include <xsimd/xsimd.hpp>

namespace vi {
namespace vx {
using arch = xsimd::default_arch;
using batch = xsimd::batch<float, arch>;
constexpr int lanes = batch::size;
}  // namespace vx

// Element-wise, in place.
void v_exp(float* x, int n);
void v_tanh(float* x, int n);
void v_gelu(float* x, int n);      // exact erf form, as torch.nn.GELU
void v_silu(float* x, int n);
void v_softplus(float* x, int n);  // torch's threshold-20 form
void v_scale(float* x, int n, float factor);

// Sum of n contiguous elements.
float v_sum(const float* x, int n);

// Largest magnitude in n contiguous elements.
float v_absmax(const float* x, int n);

// Root mean square of n contiguous elements.
float v_rms(const float* x, int n);

// dst[i] = round(src[i] * inv_scale), saturated to [-127, 127].
void v_quantize_i8(const float* src, int8_t* dst, int n, float inv_scale);

// out[i] = exp(a[i] * scale[i]) over n contiguous elements.
void v_mul_exp(float* out, const float* a, const float* scale, int n);

// One step of the state-space scan, over a run of channels:
//   h[i] = exp(a[i] * delta[i]) * h[i] + du[i] * bn
//   y[i] += h[i] * cn
// The decay is consumed in registers instead of being written out, which is
// what the scan used to spend most of its memory traffic on.
void v_scan_step(float* h, const float* a, const float* delta, const float* du,
                 float bn, float cn, float* y, int n);

inline vx::batch loadf(const float* p) { return xsimd::load_unaligned(p); }
inline void storef(float* p, vx::batch v) { xsimd::store_unaligned(p, v); }

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float silu_scalar(float x) { return x * sigmoid(x); }

}  // namespace vi
