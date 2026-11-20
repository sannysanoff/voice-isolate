// ARM SME2 matrix-multiplication kernels (fp32, fp16->fp32, bf16->fp32, int8->int32).
//
// Apple M4 Max provides two 512-bit SME2 units (SVL = 64 bytes = 16 fp32 lanes),
// delivering ~2.8 TFLOP/s in exact fp32 / fp16->fp32 and ~5.3 TOPS/s in int8->int32
// on CPU alone (no Metal). All kernels fall back cleanly when SME2 is absent.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vi {

// True when compiled with SME2 support AND the host CPU advertises FEAT_SME2.
bool sme2_available();

struct PackedF32 {
  // Layout: [ceil(n/16)][k][16], 64-byte aligned.
  std::vector<float> data;
  int n = 0;
  int k = 0;
};

struct PackedF16 {
  // Layout: [ceil(n/16)][k/2][16][2] stored as uint16_t (IEEE fp16).
  std::vector<uint16_t> data;
  int n = 0;
  int k = 0;
};

struct PackedBF16 {
  // Layout: [ceil(n/16)][k/2][16][2] stored as uint16_t (bfloat16).
  std::vector<uint16_t> data;
  int n = 0;
  int k = 0;
};

struct PackedI8 {
  // Layout: [ceil(n/16)][k/4][16][4] int8, with one symmetric scale per output row.
  std::vector<int8_t> data;
  std::vector<float> scale;
  int n = 0;
  int k = 0;
};

struct PackedI8Group {
  // Per-group (along K) symmetric int8 weights:
  // data: [ceil(n/16)][k/4][16][4]
  // scale: [num_groups][n_padded] where num_groups = ceil(k / group_size).
  std::vector<int8_t> data;
  std::vector<float> scale;
  int n = 0;
  int k = 0;
  int group_size = 64;
};

// Offline weight packing (done once per weight tensor).
void sme_pack_weights_f32(const float* w, int n, int k, PackedF32& out);
void sme_pack_weights_f16(const float* w, int n, int k, PackedF16& out);
void sme_pack_weights_bf16(const float* w, int n, int k, PackedBF16& out);
void sme_pack_weights_i8(const float* w, int n, int k, PackedI8& out);
void sme_pack_weights_i8g(const float* w, int n, int k, int group_size,
                          PackedI8Group& out);

// Online activation packing: A[m, k] (row stride `lda`) -> packed scratch.
// Caller must size scratch buffers to at least ((m+15)/16) * k * 16 elements.
void sme_pack_a_f32(const float* a, int m, int k, int lda, float* ap);
void sme_pack_a_f16(const float* a, int m, int k, int lda, uint16_t* ap);
void sme_pack_a_bf16(const float* a, int m, int k, int lda, uint16_t* ap);
void sme_quant_pack_a_i8(const float* a, int m, int k, int lda, int8_t* ap,
                         float* scales);
void sme_quant_pack_a_i8g(const float* a, int m, int k, int lda, int group_size,
                          int8_t* ap, float* scales);

// Column-slice GEMMs: C[m, n0..n1) = A * W[n0..n1)^T + bias[n0..n1).
// n0 must be a multiple of 16 (32 preferred for full 2x2 ZA tiling).
void sme_gemm_f32_cols(const float* ap, const PackedF32& w, const float* bias,
                       float* c, int m, int ldc, int n0, int n1);
void sme_gemm_f16_cols(const uint16_t* ap, const PackedF16& w,
                       const float* bias, float* c, int m, int ldc, int n0,
                       int n1);
void sme_gemm_bf16_cols(const uint16_t* ap, const PackedBF16& w,
                        const float* bias, float* c, int m, int ldc, int n0,
                        int n1);
void sme_gemm_i8_cols(const int8_t* ap, const float* as, const PackedI8& w,
                      const float* bias, float* c, int m, int ldc, int n0,
                      int n1);
void sme_gemm_i8g_cols(const int8_t* ap, const float* as,
                       const PackedI8Group& w, const float* bias, float* c,
                       int m, int ldc, int n0, int n1);

// Direct small GEMM for attention: C[m, n] = A[m, k] * B[n, k]^T (both unpacked).
// `b_trans == true`: B is [n, k] row-major (e.g. Q * K^T).
// `b_trans == false`: B is [k, n] row-major (e.g. P * V).
void sme_gemm_f32_small(const float* a, int lda, const float* b, int ldb,
                        bool b_trans, float* c, int ldc, int m, int n, int k,
                        float* scratch_ap, float* scratch_bp);

}  // namespace vi
