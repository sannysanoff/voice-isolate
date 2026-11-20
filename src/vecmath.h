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

// Full register-resident selective scan over channels [c0, c1) for `len` steps.
// - `a_log2e4`: [inner/4][state][4], pre-multiplied by log2(e) (values <= 0).
// - `h4`: [inner/4][state][4] (carried state, or nullptr for zero-init uncarried).
// - `u`, `delta`, `y`: [len][inner]
// - `z`: [len][2*inner] (z starts at offset `inner` in each row)
// - `xp`: [len][dt_total] (B at offset `dt_rank`, C at offset `dt_rank + state`)
// - `d_skip`: [inner]
void v_mamba_scan(float* h4, const float* a_log2e4, const float* d_skip,
                  const float* u, const float* delta, const float* z,
                  const float* xp, float* y, int len, int inner, int state,
                  int dt_rank, int dt_total, int c0, int c1);

// 8-wide NEON FP16 selective scan (2x faster exp2; used in fp16/int8 modes).
// `a_log2e8_f16`: [inner/8][state][8] fp16. `h4`: [inner/4][state][4] fp32 (or nullptr).
void v_mamba_scan_f16(float* h4, const uint16_t* a_log2e8_f16,
                      const float* d_skip, const float* u, const float* delta,
                      const float* z, const float* xp, float* y, int len,
                      int inner, int state, int dt_rank, int dt_total, int c0,
                      int c1);

// Fused 4-tap depthwise conv1d + bias + SiLU over [c0, c1) channels for one step:
//   dst[c] = silu(bias[c] + sum_{j=0..3} w[j][c] * ext[j][c])
void v_conv4_silu(float* dst, const float* bias, const float* w0,
                  const float* w1, const float* w2, const float* w3,
                  const float* x0, const float* x1, const float* x2,
                  const float* x3, int n);

// Fast vectorized softmax over `n` scores: s[i] = exp(s[i] * scale - max), returns sum.
float v_scale_softmax(float* s, int n, float scale);

// In-place normalized softmax: s[i] = exp(s[i] - max) / sum.
void v_softmax_norm(float* s, int n);

// Batched 4-query attention Q*K^T: s_r[j] = dot(q_r, keys[j]) * scale for r=0..3, j=0..n-1.
void v_attn4_qk(const float* q0, const float* q1, const float* q2,
                const float* q3, const float* keys, int n, int hd, float scale,
                float* s0, float* s1, float* s2, float* s3);

// Batched 4-query attention P*V: o_r[d] = sum_j p_r[j] * values[j*hd + d] for r=0..3.
void v_attn4_pv(const float* p0, const float* p1, const float* p2,
                const float* p3, const float* values, int n, int hd, float* o0,
                float* o1, float* o2, float* o3);

// Direct 4-row linear layer (no weight packing): C[4, n] = A[4, k] * W[n, k]^T + bias.
void v_linear_4rows(const float* a, int k, const float* w, int n,
                    const float* bias, float* c);

// Multi-core friendly NEON pack of one 16-row block [m0, m0+mr) from A[m, k]
// (stride lda) into SME2 panel layouts:
// - f32: ap_block[k][16]
// - f16/bf16: ap_block[k/2][16][2]
// - i8: ap_block[k/4][16][4] + scales[m0..m0+mr-1]
void v_pack_a_f32_block(const float* a, int m0, int mr, int k, int lda,
                        float* ap_block);
void v_pack_a_f16_block(const float* a, int m0, int mr, int k, int lda,
                        uint16_t* ap_block);
void v_pack_a_bf16_block(const float* a, int m0, int mr, int k, int lda,
                         uint16_t* ap_block);
void v_quant_pack_a_i8_block(const float* a, int m0, int mr, int k, int lda,
                             int8_t* ap_block, float* scales);

// FP16 KV-cache helpers (hardware NEON FP16 when available, else software fallback).
bool fp16_neon_available();
void v_f32_to_f16(const float* src, uint16_t* dst, int n);
void v_rope_f16_inplace(uint16_t* x, int head_dim, const float* cos,
                        const float* sin, bool inverse);
void v_attn4_qk_f16(const float* q0, const float* q1, const float* q2,
                    const float* q3, const uint16_t* keys, int n, int hd,
                    float scale, float* s0, float* s1, float* s2, float* s3);
void v_attn4_pv_f16(const float* p0, const float* p1, const float* p2,
                    const float* p3, const uint16_t* values, int n, int hd,
                    float* o0, float* o1, float* o2, float* o3);
float v_attn1_scores_softmax_f16(const float* q, const uint16_t* keys, int n,
                                 int hd, float scale, float* s);
void v_attn1_pv_f16(const float* s, float inv, const uint16_t* values, int n,
                    int hd, float* o);

inline vx::batch loadf(const float* p) { return xsimd::load_unaligned(p); }
inline void storef(float* p, vx::batch v) { xsimd::store_unaligned(p, v); }

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float silu_scalar(float x) { return x * sigmoid(x); }

}  // namespace vi
