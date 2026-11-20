#include "gemm_sme.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(VI_HAS_SME2)
#include <arm_sme.h>
#endif

namespace vi {

bool sme2_available() {
#if defined(VI_HAS_SME2)
  static const bool available = []() {
#if defined(__APPLE__)
    int val = 0;
    size_t len = sizeof(val);
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &val, &len, nullptr, 0) == 0)
      return val == 1;
    return false;
#else
    return true;
#endif
  }();
  return available;
#else
  return false;
#endif
}

#if defined(VI_HAS_SME2)

namespace {

// ------------------------------------------------------------ packing -------

__arm_locally_streaming __arm_new("za")
void pack_f32_impl(const float* __restrict a, int m, int k, int lda,
                   float* __restrict ap) {
  const int mb_cnt = (m + 15) / 16;
  const svbool_t pg = svptrue_b32();
  for (int mb = 0; mb < mb_cnt; ++mb) {
    const int m0 = mb * 16;
    const int mr = std::min(16, m - m0);
    float* dst = ap + static_cast<size_t>(mb) * k * 16;
    for (int k0 = 0; k0 < k; k0 += 16) {
      const svbool_t pgk = svwhilelt_b32(k0, k);
      svzero_za();
      for (int r = 0; r < mr; ++r)
        svld1_hor_za32(0, r, pgk, a + static_cast<size_t>(m0 + r) * lda + k0);
      const int kr = std::min(16, k - k0);
      for (int c = 0; c < kr; ++c)
        svst1_ver_za32(0, c, pg, dst + static_cast<size_t>(k0 + c) * 16);
    }
  }
}

// Pack B[k, n] (non-transposed, row stride ldb) -> Bp[ceil(n/16)][k][16].
__arm_locally_streaming
void pack_f32_notrans_impl(const float* __restrict b, int k, int n, int ldb,
                           float* __restrict bp) {
  const int nb_cnt = (n + 15) / 16;
  const svbool_t pg = svptrue_b32();
  for (int nb = 0; nb < nb_cnt; ++nb) {
    const int n0 = nb * 16;
    const svbool_t pgn = svwhilelt_b32(n0, n);
    float* dst = bp + static_cast<size_t>(nb) * k * 16;
    for (int kk = 0; kk < k; ++kk) {
      svfloat32_t v = svld1_f32(pgn, b + static_cast<size_t>(kk) * ldb + n0);
      svst1_f32(pg, dst + static_cast<size_t>(kk) * 16, v);
    }
  }
}

__arm_locally_streaming __arm_new("za")
void pack_f32_to_f16_impl(const float* __restrict a, int m, int k, int lda,
                          uint16_t* __restrict ap) {
  const int mb_cnt = (m + 15) / 16;
  const int k2 = k / 2;
  const svbool_t pg32 = svptrue_b32();
  uint32_t* dst32_all = reinterpret_cast<uint32_t*>(ap);
  for (int mb = 0; mb < mb_cnt; ++mb) {
    const int m0 = mb * 16;
    const int mr = std::min(16, m - m0);
    uint32_t* dst = dst32_all + static_cast<size_t>(mb) * k2 * 16;
    for (int k0 = 0; k0 < k; k0 += 32) {
      const svbool_t pg0 = svwhilelt_b32(k0, k);
      const svbool_t pg1 = svwhilelt_b32(k0 + 16, k);
      svzero_za();
      for (int r = 0; r < mr; ++r) {
        const float* row = a + static_cast<size_t>(m0 + r) * lda + k0;
        svfloat32_t f0 = svld1_f32(pg0, row);
        svfloat32_t f1 = svld1_f32(pg1, row + 16);
        svfloat16_t h0 = svcvt_f16_f32_x(pg32, f0);
        svfloat16_t h1 = svcvt_f16_f32_x(pg32, f1);
        svfloat16_t packed = svuzp1_f16(h0, h1);
        svwrite_hor_za32_u32_m(0, r, pg32, svreinterpret_u32_f16(packed));
      }
      const int kr2 = std::min(16, (k - k0) / 2);
      for (int c = 0; c < kr2; ++c)
        svst1_ver_za32(0, c, pg32, dst + static_cast<size_t>(k0 / 2 + c) * 16);
    }
  }
}

__arm_locally_streaming __arm_new("za")
void pack_f32_to_bf16_impl(const float* __restrict a, int m, int k, int lda,
                           uint16_t* __restrict ap) {
  const int mb_cnt = (m + 15) / 16;
  const int k2 = k / 2;
  const svbool_t pg32 = svptrue_b32();
  uint32_t* dst32_all = reinterpret_cast<uint32_t*>(ap);
  for (int mb = 0; mb < mb_cnt; ++mb) {
    const int m0 = mb * 16;
    const int mr = std::min(16, m - m0);
    uint32_t* dst = dst32_all + static_cast<size_t>(mb) * k2 * 16;
    for (int k0 = 0; k0 < k; k0 += 32) {
      const svbool_t pg0 = svwhilelt_b32(k0, k);
      const svbool_t pg1 = svwhilelt_b32(k0 + 16, k);
      svzero_za();
      for (int r = 0; r < mr; ++r) {
        const float* row = a + static_cast<size_t>(m0 + r) * lda + k0;
        svfloat32_t f0 = svld1_f32(pg0, row);
        svfloat32_t f1 = svld1_f32(pg1, row + 16);
        svbfloat16_t h0 = svcvt_bf16_f32_x(pg32, f0);
        svbfloat16_t h1 = svcvt_bf16_f32_x(pg32, f1);
        svbfloat16_t packed = svuzp1_bf16(h0, h1);
        svwrite_hor_za32_u32_m(0, r, pg32, svreinterpret_u32_bf16(packed));
      }
      const int kr2 = std::min(16, (k - k0) / 2);
      for (int c = 0; c < kr2; ++c)
        svst1_ver_za32(0, c, pg32, dst + static_cast<size_t>(k0 / 2 + c) * 16);
    }
  }
}

__arm_locally_streaming __arm_new("za")
void quant_pack_i8_impl(const float* __restrict a, int m, int k, int lda,
                        int8_t* __restrict ap, float* __restrict scales) {
  const int mb_cnt = (m + 15) / 16;
  const int k4 = k / 4;
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vlo = svdup_f32(-127.0f), vhi = svdup_f32(127.0f);
  uint32_t* dst32_all = reinterpret_cast<uint32_t*>(ap);

  for (int mb = 0; mb < mb_cnt; ++mb) {
    const int m0 = mb * 16;
    const int mr = std::min(16, m - m0);
    float inv_s[16];
    for (int r = 0; r < mr; ++r) {
      const float* row = a + static_cast<size_t>(m0 + r) * lda;
      svfloat32_t vmax = svdup_f32(0.0f);
      for (int kk = 0; kk < k; kk += 16) {
        svbool_t pgk = svwhilelt_b32(kk, k);
        svfloat32_t v = svabs_f32_x(pgk, svld1_f32(pgk, row + kk));
        vmax = svmax_f32_m(pgk, vmax, v);
      }
      const float peak = svmaxv_f32(pg32, vmax);
      const float s = peak > 0.0f ? peak / 127.0f : 1.0f;
      scales[m0 + r] = s;
      inv_s[r] = 1.0f / s;
    }
    uint32_t* dst = dst32_all + static_cast<size_t>(mb) * k4 * 16;
    for (int k0 = 0; k0 < k; k0 += 64) {
      svzero_za();
      for (int r = 0; r < mr; ++r) {
        const float* row = a + static_cast<size_t>(m0 + r) * lda + k0;
        const svfloat32_t inv = svdup_f32(inv_s[r]);
        svbool_t pg0 = svwhilelt_b32(k0 + 0, k);
        svfloat32_t v0 = svmin_f32_x(
            pg32, svmax_f32_x(pg32, svmul_f32_z(pg0, svld1_f32(pg0, row), inv), vlo),
            vhi);
        svint32_t i0 = svcvt_s32_f32_x(pg32, svrinta_f32_x(pg32, v0));

        svbool_t pg1 = svwhilelt_b32(k0 + 16, k);
        svfloat32_t v1 = svmin_f32_x(
            pg32,
            svmax_f32_x(pg32, svmul_f32_z(pg1, svld1_f32(pg1, row + 16), inv), vlo),
            vhi);
        svint32_t i1 = svcvt_s32_f32_x(pg32, svrinta_f32_x(pg32, v1));

        svbool_t pg2 = svwhilelt_b32(k0 + 32, k);
        svfloat32_t v2 = svmin_f32_x(
            pg32,
            svmax_f32_x(pg32, svmul_f32_z(pg2, svld1_f32(pg2, row + 32), inv), vlo),
            vhi);
        svint32_t i2 = svcvt_s32_f32_x(pg32, svrinta_f32_x(pg32, v2));

        svbool_t pg3 = svwhilelt_b32(k0 + 48, k);
        svfloat32_t v3 = svmin_f32_x(
            pg32,
            svmax_f32_x(pg32, svmul_f32_z(pg3, svld1_f32(pg3, row + 48), inv), vlo),
            vhi);
        svint32_t i3 = svcvt_s32_f32_x(pg32, svrinta_f32_x(pg32, v3));

        svint16_t s01 =
            svuzp1_s16(svreinterpret_s16_s32(i0), svreinterpret_s16_s32(i1));
        svint16_t s23 =
            svuzp1_s16(svreinterpret_s16_s32(i2), svreinterpret_s16_s32(i3));
        svint8_t b0123 =
            svuzp1_s8(svreinterpret_s8_s16(s01), svreinterpret_s8_s16(s23));
        svwrite_hor_za32_u32_m(0, r, pg32, svreinterpret_u32_s8(b0123));
      }
      const int kr4 = std::min(16, (k - k0) / 4);
      for (int c = 0; c < kr4; ++c)
        svst1_ver_za32(0, c, pg32, dst + static_cast<size_t>(k0 / 4 + c) * 16);
    }
  }
}

// Per-group (along K) symmetric int8 quantization and packing.
// `scales` is [num_groups][m_stride] where m_stride = m_padded (for weights) or m (for A).
__arm_locally_streaming __arm_new("za")
void quant_pack_i8g_impl(const float* __restrict a, int m, int k, int lda,
                         int group_size, int scale_stride,
                         int8_t* __restrict ap, float* __restrict scales) {
  const int mb_cnt = (m + 15) / 16;
  const int k4 = k / 4;
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vlo = svdup_f32(-127.0f), vhi = svdup_f32(127.0f);
  uint32_t* dst32_all = reinterpret_cast<uint32_t*>(ap);

  for (int mb = 0; mb < mb_cnt; ++mb) {
    const int m0 = mb * 16;
    const int mr = std::min(16, m - m0);
    uint32_t* dst = dst32_all + static_cast<size_t>(mb) * k4 * 16;
    for (int g = 0, kg = 0; kg < k; ++g, kg += group_size) {
      const int kend = std::min(k, kg + group_size);
      float inv_s[16];
      for (int r = 0; r < mr; ++r) {
        const float* row = a + static_cast<size_t>(m0 + r) * lda;
        svfloat32_t vmax = svdup_f32(0.0f);
        for (int kk = kg; kk < kend; kk += 16) {
          svbool_t pgk = svwhilelt_b32(kk, kend);
          svfloat32_t v = svabs_f32_x(pgk, svld1_f32(pgk, row + kk));
          vmax = svmax_f32_m(pgk, vmax, v);
        }
        const float peak = svmaxv_f32(pg32, vmax);
        const float s = peak > 0.0f ? peak / 127.0f : 1.0f;
        scales[static_cast<size_t>(g) * scale_stride + m0 + r] = s;
        inv_s[r] = 1.0f / s;
      }
      for (int k0 = kg; k0 < kend; k0 += 64) {
        svzero_za();
        for (int r = 0; r < mr; ++r) {
          const float* row = a + static_cast<size_t>(m0 + r) * lda + k0;
          const svfloat32_t inv = svdup_f32(inv_s[r]);
          svbool_t pg0 = svwhilelt_b32(k0 + 0, kend);
          svfloat32_t v0 = svmin_f32_x(
              pg32,
              svmax_f32_x(pg32, svmul_f32_z(pg0, svld1_f32(pg0, row), inv), vlo),
              vhi);
          svint32_t i0 = svcvt_s32_f32_x(pg32, svrinta_f32_x(pg32, v0));

          svbool_t pg1 = svwhilelt_b32(k0 + 16, kend);
          svfloat32_t v1 = svmin_f32_x(
              pg32,
              svmax_f32_x(pg32, svmul_f32_z(pg1, svld1_f32(pg1, row + 16), inv),
                          vlo),
              vhi);
          svint32_t i1 = svcvt_s32_f32_x(pg32, svrinta_f32_x(pg32, v1));

          svbool_t pg2 = svwhilelt_b32(k0 + 32, kend);
          svfloat32_t v2 = svmin_f32_x(
              pg32,
              svmax_f32_x(pg32, svmul_f32_z(pg2, svld1_f32(pg2, row + 32), inv),
                          vlo),
              vhi);
          svint32_t i2 = svcvt_s32_f32_x(pg32, svrinta_f32_x(pg32, v2));

          svbool_t pg3 = svwhilelt_b32(k0 + 48, kend);
          svfloat32_t v3 = svmin_f32_x(
              pg32,
              svmax_f32_x(pg32, svmul_f32_z(pg3, svld1_f32(pg3, row + 48), inv),
                          vlo),
              vhi);
          svint32_t i3 = svcvt_s32_f32_x(pg32, svrinta_f32_x(pg32, v3));

          svint16_t s01 =
              svuzp1_s16(svreinterpret_s16_s32(i0), svreinterpret_s16_s32(i1));
          svint16_t s23 =
              svuzp1_s16(svreinterpret_s16_s32(i2), svreinterpret_s16_s32(i3));
          svint8_t b0123 =
              svuzp1_s8(svreinterpret_s8_s16(s01), svreinterpret_s8_s16(s23));
          svwrite_hor_za32_u32_m(0, r, pg32, svreinterpret_u32_s8(b0123));
        }
        const int kr4 = std::min(16, (kend - k0) / 4);
        for (int c = 0; c < kr4; ++c)
          svst1_ver_za32(0, c, pg32, dst + static_cast<size_t>(k0 / 4 + c) * 16);
      }
    }
  }
}

// ------------------------------------------------------------ f32 GEMM ------

__arm_locally_streaming __arm_new("za")
void gemm_f32_cols_impl(const float* __restrict ap, const float* __restrict bp,
                        const float* __restrict bias, float* __restrict c,
                        int m, int k, int ldc, int n0, int n1) {
  const svbool_t pg = svptrue_b32();
  int row = 0;
  // Main 32-row strips (2x2 ZA tiles)
  for (; row + 16 < m; row += 32) {
    const int mr0 = 16;
    const int mr1 = std::min(16, m - (row + 16));
    const float* a0 = ap + static_cast<size_t>(row / 16) * k * 16;
    const float* a1 = ap + static_cast<size_t>(row / 16 + 1) * k * 16;
    int n = n0;
    for (; n + 32 <= n1; n += 32) {
      const float* b0 = bp + static_cast<size_t>(n / 16) * k * 16;
      const float* b1 = bp + static_cast<size_t>(n / 16 + 1) * k * 16;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pg, bias + n);
        svfloat32_t vb1 = svld1_f32(pg, bias + n + 16);
        for (int r = 0; r < 16; ++r) {
          svwrite_hor_za32_f32_m(0, r, pg, vb0);
          svwrite_hor_za32_f32_m(1, r, pg, vb1);
          svwrite_hor_za32_f32_m(2, r, pg, vb0);
          svwrite_hor_za32_f32_m(3, r, pg, vb1);
        }
      } else {
        svzero_za();
      }
      int kk = 0;
      for (; kk + 1 < k; kk += 2) {
        svfloat32_t va0 = svld1_f32(pg, a0 + kk * 16);
        svfloat32_t va1 = svld1_f32(pg, a1 + kk * 16);
        svfloat32_t vb0 = svld1_f32(pg, b0 + kk * 16);
        svfloat32_t vb1 = svld1_f32(pg, b1 + kk * 16);
        svmopa_za32_f32_m(0, pg, pg, va0, vb0);
        svmopa_za32_f32_m(1, pg, pg, va0, vb1);
        svmopa_za32_f32_m(2, pg, pg, va1, vb0);
        svmopa_za32_f32_m(3, pg, pg, va1, vb1);

        va0 = svld1_f32(pg, a0 + (kk + 1) * 16);
        va1 = svld1_f32(pg, a1 + (kk + 1) * 16);
        vb0 = svld1_f32(pg, b0 + (kk + 1) * 16);
        vb1 = svld1_f32(pg, b1 + (kk + 1) * 16);
        svmopa_za32_f32_m(0, pg, pg, va0, vb0);
        svmopa_za32_f32_m(1, pg, pg, va0, vb1);
        svmopa_za32_f32_m(2, pg, pg, va1, vb0);
        svmopa_za32_f32_m(3, pg, pg, va1, vb1);
      }
      for (; kk < k; ++kk) {
        svfloat32_t va0 = svld1_f32(pg, a0 + kk * 16);
        svfloat32_t va1 = svld1_f32(pg, a1 + kk * 16);
        svfloat32_t vb0 = svld1_f32(pg, b0 + kk * 16);
        svfloat32_t vb1 = svld1_f32(pg, b1 + kk * 16);
        svmopa_za32_f32_m(0, pg, pg, va0, vb0);
        svmopa_za32_f32_m(1, pg, pg, va0, vb1);
        svmopa_za32_f32_m(2, pg, pg, va1, vb0);
        svmopa_za32_f32_m(3, pg, pg, va1, vb1);
      }
      for (int r = 0; r < mr0; ++r) {
        svst1_hor_za32(0, r, pg, c + static_cast<size_t>(row + r) * ldc + n);
        svst1_hor_za32(1, r, pg,
                       c + static_cast<size_t>(row + r) * ldc + n + 16);
      }
      for (int r = 0; r < mr1; ++r) {
        svst1_hor_za32(2, r, pg,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n);
        svst1_hor_za32(3, r, pg,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n + 16);
      }
    }
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const float* b0 = bp + static_cast<size_t>(n / 16) * k * 16;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pgn, bias + n);
        for (int r = 0; r < 16; ++r) {
          svwrite_hor_za32_f32_m(0, r, pg, vb0);
          svwrite_hor_za32_f32_m(2, r, pg, vb0);
        }
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k; ++kk) {
        svfloat32_t va0 = svld1_f32(pg, a0 + kk * 16);
        svfloat32_t va1 = svld1_f32(pg, a1 + kk * 16);
        svfloat32_t vb0 = svld1_f32(pg, b0 + kk * 16);
        svmopa_za32_f32_m(0, pg, pg, va0, vb0);
        svmopa_za32_f32_m(2, pg, pg, va1, vb0);
      }
      for (int r = 0; r < mr0; ++r)
        svst1_hor_za32(0, r, pgn, c + static_cast<size_t>(row + r) * ldc + n);
      for (int r = 0; r < mr1; ++r)
        svst1_hor_za32(2, r, pgn,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n);
    }
  }
  // Remainder <= 16 rows (also fast path for small M <= 16!): 1x4 ZA tiles (16x64).
  if (row < m) {
    const int mr = m - row;
    const float* a0 = ap + static_cast<size_t>(row / 16) * k * 16;
    int n = n0;
    for (; n + 64 <= n1; n += 64) {
      const float* b0 = bp + static_cast<size_t>(n / 16) * k * 16;
      const float* b1 = bp + static_cast<size_t>(n / 16 + 1) * k * 16;
      const float* b2 = bp + static_cast<size_t>(n / 16 + 2) * k * 16;
      const float* b3 = bp + static_cast<size_t>(n / 16 + 3) * k * 16;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pg, bias + n);
        svfloat32_t vb1 = svld1_f32(pg, bias + n + 16);
        svfloat32_t vb2 = svld1_f32(pg, bias + n + 32);
        svfloat32_t vb3 = svld1_f32(pg, bias + n + 48);
        for (int r = 0; r < mr; ++r) {
          svwrite_hor_za32_f32_m(0, r, pg, vb0);
          svwrite_hor_za32_f32_m(1, r, pg, vb1);
          svwrite_hor_za32_f32_m(2, r, pg, vb2);
          svwrite_hor_za32_f32_m(3, r, pg, vb3);
        }
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k; ++kk) {
        svfloat32_t va = svld1_f32(pg, a0 + kk * 16);
        svmopa_za32_f32_m(0, pg, pg, va, svld1_f32(pg, b0 + kk * 16));
        svmopa_za32_f32_m(1, pg, pg, va, svld1_f32(pg, b1 + kk * 16));
        svmopa_za32_f32_m(2, pg, pg, va, svld1_f32(pg, b2 + kk * 16));
        svmopa_za32_f32_m(3, pg, pg, va, svld1_f32(pg, b3 + kk * 16));
      }
      for (int r = 0; r < mr; ++r) {
        svst1_hor_za32(0, r, pg, c + static_cast<size_t>(row + r) * ldc + n);
        svst1_hor_za32(1, r, pg,
                       c + static_cast<size_t>(row + r) * ldc + n + 16);
        svst1_hor_za32(2, r, pg,
                       c + static_cast<size_t>(row + r) * ldc + n + 32);
        svst1_hor_za32(3, r, pg,
                       c + static_cast<size_t>(row + r) * ldc + n + 48);
      }
    }
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const float* b0 = bp + static_cast<size_t>(n / 16) * k * 16;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pgn, bias + n);
        for (int r = 0; r < mr; ++r) svwrite_hor_za32_f32_m(0, r, pg, vb0);
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k; ++kk) {
        svfloat32_t va = svld1_f32(pg, a0 + kk * 16);
        svmopa_za32_f32_m(0, pg, pg, va, svld1_f32(pg, b0 + kk * 16));
      }
      for (int r = 0; r < mr; ++r)
        svst1_hor_za32(0, r, pgn, c + static_cast<size_t>(row + r) * ldc + n);
    }
  }
}

// ------------------------------------------------------------ f16 GEMM ------

__arm_locally_streaming __arm_new("za")
void gemm_f16_cols_impl(const uint16_t* __restrict ap_u,
                        const uint16_t* __restrict bp_u,
                        const float* __restrict bias, float* __restrict c,
                        int m, int k, int ldc, int n0, int n1) {
  const __fp16* ap = reinterpret_cast<const __fp16*>(ap_u);
  const __fp16* bp = reinterpret_cast<const __fp16*>(bp_u);
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const int k2 = k / 2;
  int row = 0;
  for (; row + 16 < m; row += 32) {
    const int mr0 = 16;
    const int mr1 = std::min(16, m - (row + 16));
    const __fp16* a0 = ap + static_cast<size_t>(row / 16) * k2 * 32;
    const __fp16* a1 = ap + static_cast<size_t>(row / 16 + 1) * k2 * 32;
    int n = n0;
    for (; n + 32 <= n1; n += 32) {
      const __fp16* b0 = bp + static_cast<size_t>(n / 16) * k2 * 32;
      const __fp16* b1 = bp + static_cast<size_t>(n / 16 + 1) * k2 * 32;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pg32, bias + n);
        svfloat32_t vb1 = svld1_f32(pg32, bias + n + 16);
        for (int r = 0; r < 16; ++r) {
          svwrite_hor_za32_f32_m(0, r, pg32, vb0);
          svwrite_hor_za32_f32_m(1, r, pg32, vb1);
          svwrite_hor_za32_f32_m(2, r, pg32, vb0);
          svwrite_hor_za32_f32_m(3, r, pg32, vb1);
        }
      } else {
        svzero_za();
      }
      int kk = 0;
      for (; kk + 1 < k2; kk += 2) {
        svfloat16_t va0 = svld1_f16(pg16, a0 + kk * 32);
        svfloat16_t va1 = svld1_f16(pg16, a1 + kk * 32);
        svfloat16_t vb0 = svld1_f16(pg16, b0 + kk * 32);
        svfloat16_t vb1 = svld1_f16(pg16, b1 + kk * 32);
        svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
        svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
        svmopa_za32_f16_m(2, pg16, pg16, va1, vb0);
        svmopa_za32_f16_m(3, pg16, pg16, va1, vb1);

        va0 = svld1_f16(pg16, a0 + (kk + 1) * 32);
        va1 = svld1_f16(pg16, a1 + (kk + 1) * 32);
        vb0 = svld1_f16(pg16, b0 + (kk + 1) * 32);
        vb1 = svld1_f16(pg16, b1 + (kk + 1) * 32);
        svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
        svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
        svmopa_za32_f16_m(2, pg16, pg16, va1, vb0);
        svmopa_za32_f16_m(3, pg16, pg16, va1, vb1);
      }
      for (; kk < k2; ++kk) {
        svfloat16_t va0 = svld1_f16(pg16, a0 + kk * 32);
        svfloat16_t va1 = svld1_f16(pg16, a1 + kk * 32);
        svfloat16_t vb0 = svld1_f16(pg16, b0 + kk * 32);
        svfloat16_t vb1 = svld1_f16(pg16, b1 + kk * 32);
        svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
        svmopa_za32_f16_m(1, pg16, pg16, va0, vb1);
        svmopa_za32_f16_m(2, pg16, pg16, va1, vb0);
        svmopa_za32_f16_m(3, pg16, pg16, va1, vb1);
      }
      for (int r = 0; r < mr0; ++r) {
        svst1_hor_za32(0, r, pg32, c + static_cast<size_t>(row + r) * ldc + n);
        svst1_hor_za32(1, r, pg32,
                       c + static_cast<size_t>(row + r) * ldc + n + 16);
      }
      for (int r = 0; r < mr1; ++r) {
        svst1_hor_za32(2, r, pg32,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n);
        svst1_hor_za32(3, r, pg32,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n + 16);
      }
    }
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const __fp16* b0 = bp + static_cast<size_t>(n / 16) * k2 * 32;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pgn, bias + n);
        for (int r = 0; r < 16; ++r) {
          svwrite_hor_za32_f32_m(0, r, pg32, vb0);
          svwrite_hor_za32_f32_m(2, r, pg32, vb0);
        }
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k2; ++kk) {
        svfloat16_t va0 = svld1_f16(pg16, a0 + kk * 32);
        svfloat16_t va1 = svld1_f16(pg16, a1 + kk * 32);
        svfloat16_t vb0 = svld1_f16(pg16, b0 + kk * 32);
        svmopa_za32_f16_m(0, pg16, pg16, va0, vb0);
        svmopa_za32_f16_m(2, pg16, pg16, va1, vb0);
      }
      for (int r = 0; r < mr0; ++r)
        svst1_hor_za32(0, r, pgn, c + static_cast<size_t>(row + r) * ldc + n);
      for (int r = 0; r < mr1; ++r)
        svst1_hor_za32(2, r, pgn,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n);
    }
  }
  if (row < m) {
    const int mr = m - row;
    const __fp16* a0 = ap + static_cast<size_t>(row / 16) * k2 * 32;
    int n = n0;
    for (; n + 64 <= n1; n += 64) {
      const __fp16* b0 = bp + static_cast<size_t>(n / 16) * k2 * 32;
      const __fp16* b1 = bp + static_cast<size_t>(n / 16 + 1) * k2 * 32;
      const __fp16* b2 = bp + static_cast<size_t>(n / 16 + 2) * k2 * 32;
      const __fp16* b3 = bp + static_cast<size_t>(n / 16 + 3) * k2 * 32;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pg32, bias + n);
        svfloat32_t vb1 = svld1_f32(pg32, bias + n + 16);
        svfloat32_t vb2 = svld1_f32(pg32, bias + n + 32);
        svfloat32_t vb3 = svld1_f32(pg32, bias + n + 48);
        for (int r = 0; r < mr; ++r) {
          svwrite_hor_za32_f32_m(0, r, pg32, vb0);
          svwrite_hor_za32_f32_m(1, r, pg32, vb1);
          svwrite_hor_za32_f32_m(2, r, pg32, vb2);
          svwrite_hor_za32_f32_m(3, r, pg32, vb3);
        }
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k2; ++kk) {
        svfloat16_t va = svld1_f16(pg16, a0 + kk * 32);
        svmopa_za32_f16_m(0, pg16, pg16, va, svld1_f16(pg16, b0 + kk * 32));
        svmopa_za32_f16_m(1, pg16, pg16, va, svld1_f16(pg16, b1 + kk * 32));
        svmopa_za32_f16_m(2, pg16, pg16, va, svld1_f16(pg16, b2 + kk * 32));
        svmopa_za32_f16_m(3, pg16, pg16, va, svld1_f16(pg16, b3 + kk * 32));
      }
      for (int r = 0; r < mr; ++r) {
        svst1_hor_za32(0, r, pg32, c + static_cast<size_t>(row + r) * ldc + n);
        svst1_hor_za32(1, r, pg32,
                       c + static_cast<size_t>(row + r) * ldc + n + 16);
        svst1_hor_za32(2, r, pg32,
                       c + static_cast<size_t>(row + r) * ldc + n + 32);
        svst1_hor_za32(3, r, pg32,
                       c + static_cast<size_t>(row + r) * ldc + n + 48);
      }
    }
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const __fp16* b0 = bp + static_cast<size_t>(n / 16) * k2 * 32;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pgn, bias + n);
        for (int r = 0; r < mr; ++r) svwrite_hor_za32_f32_m(0, r, pg32, vb0);
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k2; ++kk) {
        svfloat16_t va = svld1_f16(pg16, a0 + kk * 32);
        svmopa_za32_f16_m(0, pg16, pg16, va, svld1_f16(pg16, b0 + kk * 32));
      }
      for (int r = 0; r < mr; ++r)
        svst1_hor_za32(0, r, pgn, c + static_cast<size_t>(row + r) * ldc + n);
    }
  }
}

// ------------------------------------------------------------ bf16 GEMM -----

__arm_locally_streaming __arm_new("za")
void gemm_bf16_cols_impl(const uint16_t* __restrict ap_u,
                         const uint16_t* __restrict bp_u,
                         const float* __restrict bias, float* __restrict c,
                         int m, int k, int ldc, int n0, int n1) {
  const __bf16* ap = reinterpret_cast<const __bf16*>(ap_u);
  const __bf16* bp = reinterpret_cast<const __bf16*>(bp_u);
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const int k2 = k / 2;
  int row = 0;
  for (; row + 16 < m; row += 32) {
    const int mr0 = 16;
    const int mr1 = std::min(16, m - (row + 16));
    const __bf16* a0 = ap + static_cast<size_t>(row / 16) * k2 * 32;
    const __bf16* a1 = ap + static_cast<size_t>(row / 16 + 1) * k2 * 32;
    int n = n0;
    for (; n + 32 <= n1; n += 32) {
      const __bf16* b0 = bp + static_cast<size_t>(n / 16) * k2 * 32;
      const __bf16* b1 = bp + static_cast<size_t>(n / 16 + 1) * k2 * 32;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pg32, bias + n);
        svfloat32_t vb1 = svld1_f32(pg32, bias + n + 16);
        for (int r = 0; r < 16; ++r) {
          svwrite_hor_za32_f32_m(0, r, pg32, vb0);
          svwrite_hor_za32_f32_m(1, r, pg32, vb1);
          svwrite_hor_za32_f32_m(2, r, pg32, vb0);
          svwrite_hor_za32_f32_m(3, r, pg32, vb1);
        }
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k2; ++kk) {
        svbfloat16_t va0 = svld1_bf16(pg16, a0 + kk * 32);
        svbfloat16_t va1 = svld1_bf16(pg16, a1 + kk * 32);
        svbfloat16_t vb0 = svld1_bf16(pg16, b0 + kk * 32);
        svbfloat16_t vb1 = svld1_bf16(pg16, b1 + kk * 32);
        svmopa_za32_bf16_m(0, pg16, pg16, va0, vb0);
        svmopa_za32_bf16_m(1, pg16, pg16, va0, vb1);
        svmopa_za32_bf16_m(2, pg16, pg16, va1, vb0);
        svmopa_za32_bf16_m(3, pg16, pg16, va1, vb1);
      }
      for (int r = 0; r < mr0; ++r) {
        svst1_hor_za32(0, r, pg32, c + static_cast<size_t>(row + r) * ldc + n);
        svst1_hor_za32(1, r, pg32,
                       c + static_cast<size_t>(row + r) * ldc + n + 16);
      }
      for (int r = 0; r < mr1; ++r) {
        svst1_hor_za32(2, r, pg32,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n);
        svst1_hor_za32(3, r, pg32,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n + 16);
      }
    }
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const __bf16* b0 = bp + static_cast<size_t>(n / 16) * k2 * 32;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pgn, bias + n);
        for (int r = 0; r < 16; ++r) {
          svwrite_hor_za32_f32_m(0, r, pg32, vb0);
          svwrite_hor_za32_f32_m(2, r, pg32, vb0);
        }
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k2; ++kk) {
        svbfloat16_t va0 = svld1_bf16(pg16, a0 + kk * 32);
        svbfloat16_t va1 = svld1_bf16(pg16, a1 + kk * 32);
        svbfloat16_t vb0 = svld1_bf16(pg16, b0 + kk * 32);
        svmopa_za32_bf16_m(0, pg16, pg16, va0, vb0);
        svmopa_za32_bf16_m(2, pg16, pg16, va1, vb0);
      }
      for (int r = 0; r < mr0; ++r)
        svst1_hor_za32(0, r, pgn, c + static_cast<size_t>(row + r) * ldc + n);
      for (int r = 0; r < mr1; ++r)
        svst1_hor_za32(2, r, pgn,
                       c + static_cast<size_t>(row + 16 + r) * ldc + n);
    }
  }
  if (row < m) {
    const int mr = m - row;
    const __bf16* a0 = ap + static_cast<size_t>(row / 16) * k2 * 32;
    int n = n0;
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const __bf16* b0 = bp + static_cast<size_t>(n / 16) * k2 * 32;
      if (bias != nullptr) {
        svfloat32_t vb0 = svld1_f32(pgn, bias + n);
        for (int r = 0; r < mr; ++r) svwrite_hor_za32_f32_m(0, r, pg32, vb0);
      } else {
        svzero_za();
      }
      for (int kk = 0; kk < k2; ++kk) {
        svbfloat16_t va = svld1_bf16(pg16, a0 + kk * 32);
        svmopa_za32_bf16_m(0, pg16, pg16, va, svld1_bf16(pg16, b0 + kk * 32));
      }
      for (int r = 0; r < mr; ++r)
        svst1_hor_za32(0, r, pgn, c + static_cast<size_t>(row + r) * ldc + n);
    }
  }
}

// ------------------------------------------------------------ i8 GEMM -------

__arm_locally_streaming __arm_new("za")
void gemm_i8_cols_impl(const int8_t* __restrict ap, const float* __restrict as,
                       const int8_t* __restrict bp, const float* __restrict ws,
                       const float* __restrict bias, float* __restrict c,
                       int m, int k, int ldc, int n0, int n1) {
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const int k4 = k / 4;
  int row = 0;
  for (; row + 16 < m; row += 32) {
    const int mr0 = 16;
    const int mr1 = std::min(16, m - (row + 16));
    const int8_t* a0 = ap + static_cast<size_t>(row / 16) * k4 * 64;
    const int8_t* a1 = ap + static_cast<size_t>(row / 16 + 1) * k4 * 64;
    int n = n0;
    for (; n + 32 <= n1; n += 32) {
      const int8_t* b0 = bp + static_cast<size_t>(n / 16) * k4 * 64;
      const int8_t* b1 = bp + static_cast<size_t>(n / 16 + 1) * k4 * 64;
      svzero_za();
      int kk = 0;
      for (; kk + 1 < k4; kk += 2) {
        svint8_t va0 = svld1_s8(pg8, a0 + kk * 64);
        svint8_t va1 = svld1_s8(pg8, a1 + kk * 64);
        svint8_t vb0 = svld1_s8(pg8, b0 + kk * 64);
        svint8_t vb1 = svld1_s8(pg8, b1 + kk * 64);
        svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
        svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
        svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
        svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);

        va0 = svld1_s8(pg8, a0 + (kk + 1) * 64);
        va1 = svld1_s8(pg8, a1 + (kk + 1) * 64);
        vb0 = svld1_s8(pg8, b0 + (kk + 1) * 64);
        vb1 = svld1_s8(pg8, b1 + (kk + 1) * 64);
        svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
        svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
        svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
        svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);
      }
      for (; kk < k4; ++kk) {
        svint8_t va0 = svld1_s8(pg8, a0 + kk * 64);
        svint8_t va1 = svld1_s8(pg8, a1 + kk * 64);
        svint8_t vb0 = svld1_s8(pg8, b0 + kk * 64);
        svint8_t vb1 = svld1_s8(pg8, b1 + kk * 64);
        svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
        svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
        svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
        svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);
      }
      svfloat32_t vws0 = svld1_f32(pg32, ws + n);
      svfloat32_t vws1 = svld1_f32(pg32, ws + n + 16);
      svfloat32_t vb0 = bias ? svld1_f32(pg32, bias + n) : svdup_f32(0.0f);
      svfloat32_t vb1 = bias ? svld1_f32(pg32, bias + n + 16) : svdup_f32(0.0f);

      for (int r = 0; r < mr0; r += 4) {
        svint32x4_t q0 = svread_hor_za32_s32_vg4(0, r);
        svint32x4_t q1 = svread_hor_za32_s32_vg4(1, r);
        svfloat32_t sc00 = svmul_f32_x(pg32, vws0, svdup_f32(as[row + r + 0]));
        svfloat32_t sc01 = svmul_f32_x(pg32, vws1, svdup_f32(as[row + r + 0]));
        svfloat32_t sc10 = svmul_f32_x(pg32, vws0, svdup_f32(as[row + r + 1]));
        svfloat32_t sc11 = svmul_f32_x(pg32, vws1, svdup_f32(as[row + r + 1]));
        svfloat32_t sc20 = svmul_f32_x(pg32, vws0, svdup_f32(as[row + r + 2]));
        svfloat32_t sc21 = svmul_f32_x(pg32, vws1, svdup_f32(as[row + r + 2]));
        svfloat32_t sc30 = svmul_f32_x(pg32, vws0, svdup_f32(as[row + r + 3]));
        svfloat32_t sc31 = svmul_f32_x(pg32, vws1, svdup_f32(as[row + r + 3]));
        svst1_f32(pg32, c + static_cast<size_t>(row + r + 0) * ldc + n,
                  svmla_f32_x(pg32, vb0, svcvt_f32_s32_x(pg32, svget4_s32(q0, 0)), sc00));
        svst1_f32(pg32, c + static_cast<size_t>(row + r + 0) * ldc + n + 16,
                  svmla_f32_x(pg32, vb1, svcvt_f32_s32_x(pg32, svget4_s32(q1, 0)), sc01));
        svst1_f32(pg32, c + static_cast<size_t>(row + r + 1) * ldc + n,
                  svmla_f32_x(pg32, vb0, svcvt_f32_s32_x(pg32, svget4_s32(q0, 1)), sc10));
        svst1_f32(pg32, c + static_cast<size_t>(row + r + 1) * ldc + n + 16,
                  svmla_f32_x(pg32, vb1, svcvt_f32_s32_x(pg32, svget4_s32(q1, 1)), sc11));
        svst1_f32(pg32, c + static_cast<size_t>(row + r + 2) * ldc + n,
                  svmla_f32_x(pg32, vb0, svcvt_f32_s32_x(pg32, svget4_s32(q0, 2)), sc20));
        svst1_f32(pg32, c + static_cast<size_t>(row + r + 2) * ldc + n + 16,
                  svmla_f32_x(pg32, vb1, svcvt_f32_s32_x(pg32, svget4_s32(q1, 2)), sc21));
        svst1_f32(pg32, c + static_cast<size_t>(row + r + 3) * ldc + n,
                  svmla_f32_x(pg32, vb0, svcvt_f32_s32_x(pg32, svget4_s32(q0, 3)), sc30));
        svst1_f32(pg32, c + static_cast<size_t>(row + r + 3) * ldc + n + 16,
                  svmla_f32_x(pg32, vb1, svcvt_f32_s32_x(pg32, svget4_s32(q1, 3)), sc31));
      }
      for (int r = 0; r < mr1; r += 4) {
        svint32x4_t q0 = svread_hor_za32_s32_vg4(2, r);
        svint32x4_t q1 = svread_hor_za32_s32_vg4(3, r);
        for (int u = 0; u < 4 && r + u < mr1; ++u) {
          svfloat32_t vas = svdup_f32(as[row + 16 + r + u]);
          svint32_t z0 = (u == 0) ? svget4_s32(q0, 0)
                       : (u == 1) ? svget4_s32(q0, 1)
                       : (u == 2) ? svget4_s32(q0, 2)
                                  : svget4_s32(q0, 3);
          svint32_t z1 = (u == 0) ? svget4_s32(q1, 0)
                       : (u == 1) ? svget4_s32(q1, 1)
                       : (u == 2) ? svget4_s32(q1, 2)
                                  : svget4_s32(q1, 3);
          svst1_f32(pg32, c + static_cast<size_t>(row + 16 + r + u) * ldc + n,
                    svmla_f32_x(pg32, vb0, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z0), vas), vws0));
          svst1_f32(pg32, c + static_cast<size_t>(row + 16 + r + u) * ldc + n + 16,
                    svmla_f32_x(pg32, vb1, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z1), vas), vws1));
        }
      }
    }
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const int8_t* b0 = bp + static_cast<size_t>(n / 16) * k4 * 64;
      svzero_za();
      for (int kk = 0; kk < k4; ++kk) {
        svint8_t va0 = svld1_s8(pg8, a0 + kk * 64);
        svint8_t va1 = svld1_s8(pg8, a1 + kk * 64);
        svint8_t vb0 = svld1_s8(pg8, b0 + kk * 64);
        svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
        svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
      }
      svfloat32_t vws0 = svld1_f32(pgn, ws + n);
      svfloat32_t vb0 = bias ? svld1_f32(pgn, bias + n) : svdup_f32(0.0f);
      for (int r = 0; r < mr0; ++r) {
        svfloat32_t vas = svdup_f32(as[row + r]);
        svfloat32_t f0 = svcvt_f32_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, r));
        svst1_f32(pgn, c + static_cast<size_t>(row + r) * ldc + n,
                  svmla_f32_x(pg32, vb0, svmul_f32_x(pg32, f0, vas), vws0));
      }
      for (int r = 0; r < mr1; ++r) {
        svfloat32_t vas = svdup_f32(as[row + 16 + r]);
        svfloat32_t f0 = svcvt_f32_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, r));
        svst1_f32(pgn, c + static_cast<size_t>(row + 16 + r) * ldc + n,
                  svmla_f32_x(pg32, vb0, svmul_f32_x(pg32, f0, vas), vws0));
      }
    }
  }
  if (row < m) {
    const int mr = m - row;
    const int8_t* a0 = ap + static_cast<size_t>(row / 16) * k4 * 64;
    int n = n0;
    for (; n + 64 <= n1; n += 64) {
      const int8_t* b0 = bp + static_cast<size_t>(n / 16) * k4 * 64;
      const int8_t* b1 = bp + static_cast<size_t>(n / 16 + 1) * k4 * 64;
      const int8_t* b2 = bp + static_cast<size_t>(n / 16 + 2) * k4 * 64;
      const int8_t* b3 = bp + static_cast<size_t>(n / 16 + 3) * k4 * 64;
      svzero_za();
      for (int kk = 0; kk < k4; ++kk) {
        svint8_t va = svld1_s8(pg8, a0 + kk * 64);
        svmopa_za32_s8_m(0, pg8, pg8, va, svld1_s8(pg8, b0 + kk * 64));
        svmopa_za32_s8_m(1, pg8, pg8, va, svld1_s8(pg8, b1 + kk * 64));
        svmopa_za32_s8_m(2, pg8, pg8, va, svld1_s8(pg8, b2 + kk * 64));
        svmopa_za32_s8_m(3, pg8, pg8, va, svld1_s8(pg8, b3 + kk * 64));
      }
      svfloat32_t vws0 = svld1_f32(pg32, ws + n);
      svfloat32_t vws1 = svld1_f32(pg32, ws + n + 16);
      svfloat32_t vws2 = svld1_f32(pg32, ws + n + 32);
      svfloat32_t vws3 = svld1_f32(pg32, ws + n + 48);
      svfloat32_t vb0 = bias ? svld1_f32(pg32, bias + n) : svdup_f32(0.0f);
      svfloat32_t vb1 = bias ? svld1_f32(pg32, bias + n + 16) : svdup_f32(0.0f);
      svfloat32_t vb2 = bias ? svld1_f32(pg32, bias + n + 32) : svdup_f32(0.0f);
      svfloat32_t vb3 = bias ? svld1_f32(pg32, bias + n + 48) : svdup_f32(0.0f);
      for (int r = 0; r < mr; r += 4) {
        svint32x4_t q0 = svread_hor_za32_s32_vg4(0, r);
        svint32x4_t q1 = svread_hor_za32_s32_vg4(1, r);
        svint32x4_t q2 = svread_hor_za32_s32_vg4(2, r);
        svint32x4_t q3 = svread_hor_za32_s32_vg4(3, r);
        for (int u = 0; u < 4 && r + u < mr; ++u) {
          svfloat32_t vas = svdup_f32(as[row + r + u]);
          svint32_t z0 = (u == 0) ? svget4_s32(q0, 0) : (u == 1) ? svget4_s32(q0, 1)
                       : (u == 2) ? svget4_s32(q0, 2) : svget4_s32(q0, 3);
          svint32_t z1 = (u == 0) ? svget4_s32(q1, 0) : (u == 1) ? svget4_s32(q1, 1)
                       : (u == 2) ? svget4_s32(q1, 2) : svget4_s32(q1, 3);
          svint32_t z2 = (u == 0) ? svget4_s32(q2, 0) : (u == 1) ? svget4_s32(q2, 1)
                       : (u == 2) ? svget4_s32(q2, 2) : svget4_s32(q2, 3);
          svint32_t z3 = (u == 0) ? svget4_s32(q3, 0) : (u == 1) ? svget4_s32(q3, 1)
                       : (u == 2) ? svget4_s32(q3, 2) : svget4_s32(q3, 3);
          float* out = c + static_cast<size_t>(row + r + u) * ldc + n;
          svst1_f32(pg32, out,
                    svmla_f32_x(pg32, vb0, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z0), vas), vws0));
          svst1_f32(pg32, out + 16,
                    svmla_f32_x(pg32, vb1, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z1), vas), vws1));
          svst1_f32(pg32, out + 32,
                    svmla_f32_x(pg32, vb2, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z2), vas), vws2));
          svst1_f32(pg32, out + 48,
                    svmla_f32_x(pg32, vb3, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z3), vas), vws3));
        }
      }
    }
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const int8_t* b0 = bp + static_cast<size_t>(n / 16) * k4 * 64;
      svzero_za();
      for (int kk = 0; kk < k4; ++kk) {
        svint8_t va = svld1_s8(pg8, a0 + kk * 64);
        svmopa_za32_s8_m(0, pg8, pg8, va, svld1_s8(pg8, b0 + kk * 64));
      }
      svfloat32_t vws0 = svld1_f32(pgn, ws + n);
      svfloat32_t vb0 = bias ? svld1_f32(pgn, bias + n) : svdup_f32(0.0f);
      for (int r = 0; r < mr; ++r) {
        svfloat32_t vas = svdup_f32(as[row + r]);
        svfloat32_t f0 = svcvt_f32_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, r));
        svst1_f32(pgn, c + static_cast<size_t>(row + r) * ldc + n,
                  svmla_f32_x(pg32, vb0, svmul_f32_x(pg32, f0, vas), vws0));
      }
    }
  }
}

// ------------------------------------------------------------ i8g GEMM ------

__arm_locally_streaming __arm_new("za")
void gemm_i8g_cols_impl(const int8_t* __restrict ap, const float* __restrict as,
                        const int8_t* __restrict bp, const float* __restrict ws,
                        const float* __restrict bias, float* __restrict c,
                        int m, int n_padded, int k, int group_size, int ldc,
                        int n0, int n1) {
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const int k4 = k / 4;
  const int g4 = group_size / 4;
  const int num_groups = (k + group_size - 1) / group_size;
  alignas(64) float acc[32 * 32];

  for (int row = 0; row < m; row += 32) {
    const int mr0 = std::min(16, m - row);
    const int mr1 = std::max(0, std::min(16, m - (row + 16)));
    const int8_t* a0 = ap + static_cast<size_t>(row / 16) * k4 * 64;
    const int8_t* a1 =
        mr1 > 0 ? ap + static_cast<size_t>(row / 16 + 1) * k4 * 64 : a0;
    int n = n0;
    for (; n + 32 <= n1; n += 32) {
      const int8_t* b0 = bp + static_cast<size_t>(n / 16) * k4 * 64;
      const int8_t* b1 = bp + static_cast<size_t>(n / 16 + 1) * k4 * 64;
      svfloat32_t vb0 = bias ? svld1_f32(pg32, bias + n) : svdup_f32(0.0f);
      svfloat32_t vb1 = bias ? svld1_f32(pg32, bias + n + 16) : svdup_f32(0.0f);
      for (int r = 0; r < mr0 + mr1; ++r) {
        svst1_f32(pg32, acc + r * 32, vb0);
        svst1_f32(pg32, acc + r * 32 + 16, vb1);
      }
      for (int g = 0; g < num_groups; ++g) {
        const int kk0 = g * g4;
        const int kk1 = std::min(k4, kk0 + g4);
        svzero_za();
        for (int kk = kk0; kk < kk1; ++kk) {
          svint8_t va0 = svld1_s8(pg8, a0 + kk * 64);
          svint8_t va1 = svld1_s8(pg8, a1 + kk * 64);
          svint8_t vb0s = svld1_s8(pg8, b0 + kk * 64);
          svint8_t vb1s = svld1_s8(pg8, b1 + kk * 64);
          svmopa_za32_s8_m(0, pg8, pg8, va0, vb0s);
          svmopa_za32_s8_m(1, pg8, pg8, va0, vb1s);
          if (mr1 > 0) {
            svmopa_za32_s8_m(2, pg8, pg8, va1, vb0s);
            svmopa_za32_s8_m(3, pg8, pg8, va1, vb1s);
          }
        }
        const float* asg = as + static_cast<size_t>(g) * m;
        const float* wsg = ws + static_cast<size_t>(g) * n_padded + n;
        svfloat32_t vws0 = svld1_f32(pg32, wsg);
        svfloat32_t vws1 = svld1_f32(pg32, wsg + 16);
        for (int r = 0; r < mr0; r += 4) {
          svint32x4_t q0 = svread_hor_za32_s32_vg4(0, r);
          svint32x4_t q1 = svread_hor_za32_s32_vg4(1, r);
          for (int u = 0; u < 4 && r + u < mr0; ++u) {
            svfloat32_t vas = svdup_f32(asg[row + r + u]);
            svint32_t z0 = (u == 0) ? svget4_s32(q0, 0)
                         : (u == 1) ? svget4_s32(q0, 1)
                         : (u == 2) ? svget4_s32(q0, 2)
                                    : svget4_s32(q0, 3);
            svint32_t z1 = (u == 0) ? svget4_s32(q1, 0)
                         : (u == 1) ? svget4_s32(q1, 1)
                         : (u == 2) ? svget4_s32(q1, 2)
                                    : svget4_s32(q1, 3);
            svfloat32_t p0 = svld1_f32(pg32, acc + (r + u) * 32);
            svfloat32_t p1 = svld1_f32(pg32, acc + (r + u) * 32 + 16);
            svst1_f32(pg32, acc + (r + u) * 32,
                      svmla_f32_x(pg32, p0, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z0), vas), vws0));
            svst1_f32(pg32, acc + (r + u) * 32 + 16,
                      svmla_f32_x(pg32, p1, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z1), vas), vws1));
          }
        }
        for (int r = 0; r < mr1; r += 4) {
          svint32x4_t q0 = svread_hor_za32_s32_vg4(2, r);
          svint32x4_t q1 = svread_hor_za32_s32_vg4(3, r);
          for (int u = 0; u < 4 && r + u < mr1; ++u) {
            svfloat32_t vas = svdup_f32(asg[row + 16 + r + u]);
            svint32_t z0 = (u == 0) ? svget4_s32(q0, 0)
                         : (u == 1) ? svget4_s32(q0, 1)
                         : (u == 2) ? svget4_s32(q0, 2)
                                    : svget4_s32(q0, 3);
            svint32_t z1 = (u == 0) ? svget4_s32(q1, 0)
                         : (u == 1) ? svget4_s32(q1, 1)
                         : (u == 2) ? svget4_s32(q1, 2)
                                    : svget4_s32(q1, 3);
            svfloat32_t p0 = svld1_f32(pg32, acc + (16 + r + u) * 32);
            svfloat32_t p1 = svld1_f32(pg32, acc + (16 + r + u) * 32 + 16);
            svst1_f32(pg32, acc + (16 + r + u) * 32,
                      svmla_f32_x(pg32, p0, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z0), vas), vws0));
            svst1_f32(pg32, acc + (16 + r + u) * 32 + 16,
                      svmla_f32_x(pg32, p1, svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, z1), vas), vws1));
          }
        }
      }
      for (int r = 0; r < mr0 + mr1; ++r) {
        svst1_f32(pg32, c + static_cast<size_t>(row + r) * ldc + n,
                  svld1_f32(pg32, acc + r * 32));
        svst1_f32(pg32, c + static_cast<size_t>(row + r) * ldc + n + 16,
                  svld1_f32(pg32, acc + r * 32 + 16));
      }
    }
    for (; n < n1; n += 16) {
      const svbool_t pgn = svwhilelt_b32(n, n1);
      const int8_t* b0 = bp + static_cast<size_t>(n / 16) * k4 * 64;
      svfloat32_t vb0 = bias ? svld1_f32(pgn, bias + n) : svdup_f32(0.0f);
      for (int r = 0; r < mr0 + mr1; ++r) svst1_f32(pg32, acc + r * 16, vb0);
      for (int g = 0; g < num_groups; ++g) {
        const int kk0 = g * g4;
        const int kk1 = std::min(k4, kk0 + g4);
        svzero_za();
        for (int kk = kk0; kk < kk1; ++kk) {
          svint8_t va0 = svld1_s8(pg8, a0 + kk * 64);
          svint8_t va1 = svld1_s8(pg8, a1 + kk * 64);
          svint8_t vb0s = svld1_s8(pg8, b0 + kk * 64);
          svmopa_za32_s8_m(0, pg8, pg8, va0, vb0s);
          if (mr1 > 0) svmopa_za32_s8_m(2, pg8, pg8, va1, vb0s);
        }
        const float* asg = as + static_cast<size_t>(g) * m;
        const float* wsg = ws + static_cast<size_t>(g) * n_padded + n;
        svfloat32_t vws0 = svld1_f32(pgn, wsg);
        for (int r = 0; r < mr0; ++r) {
          svfloat32_t vas = svdup_f32(asg[row + r]);
          svfloat32_t f0 = svcvt_f32_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, r));
          svfloat32_t p0 = svld1_f32(pg32, acc + r * 16);
          svst1_f32(pg32, acc + r * 16,
                    svmla_f32_x(pg32, p0, svmul_f32_x(pg32, f0, vas), vws0));
        }
        for (int r = 0; r < mr1; ++r) {
          svfloat32_t vas = svdup_f32(asg[row + 16 + r]);
          svfloat32_t f0 = svcvt_f32_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, r));
          svfloat32_t p0 = svld1_f32(pg32, acc + (16 + r) * 16);
          svst1_f32(pg32, acc + (16 + r) * 16,
                    svmla_f32_x(pg32, p0, svmul_f32_x(pg32, f0, vas), vws0));
        }
      }
      for (int r = 0; r < mr0 + mr1; ++r)
        svst1_f32(pgn, c + static_cast<size_t>(row + r) * ldc + n,
                  svld1_f32(pg32, acc + r * 16));
    }
  }
}

}  // namespace

void sme_pack_weights_f32(const float* w, int n, int k, PackedF32& out) {
  out.n = n;
  out.k = k;
  const int nb = (n + 15) / 16;
  out.data.resize(static_cast<size_t>(nb) * k * 16);
  pack_f32_impl(w, n, k, k, out.data.data());
}

void sme_pack_weights_f16(const float* w, int n, int k, PackedF16& out) {
  out.n = n;
  out.k = k;
  const int nb = (n + 15) / 16;
  out.data.resize(static_cast<size_t>(nb) * k * 16);
  pack_f32_to_f16_impl(w, n, k, k, out.data.data());
}

void sme_pack_weights_bf16(const float* w, int n, int k, PackedBF16& out) {
  out.n = n;
  out.k = k;
  const int nb = (n + 15) / 16;
  out.data.resize(static_cast<size_t>(nb) * k * 16);
  pack_f32_to_bf16_impl(w, n, k, k, out.data.data());
}

void sme_pack_weights_i8(const float* w, int n, int k, PackedI8& out) {
  out.n = n;
  out.k = k;
  const int nb = (n + 15) / 16;
  out.data.resize(static_cast<size_t>(nb) * k * 16);
  out.scale.assign(static_cast<size_t>(nb) * 16, 1.0f);
  quant_pack_i8_impl(w, n, k, k, out.data.data(), out.scale.data());
}

void sme_pack_weights_i8g(const float* w, int n, int k, int group_size,
                          PackedI8Group& out) {
  out.n = n;
  out.k = k;
  out.group_size = group_size;
  const int nb = (n + 15) / 16;
  const int n_padded = nb * 16;
  const int ng = (k + group_size - 1) / group_size;
  out.data.resize(static_cast<size_t>(nb) * k * 16);
  out.scale.assign(static_cast<size_t>(ng) * n_padded, 1.0f);
  quant_pack_i8g_impl(w, n, k, k, group_size, n_padded, out.data.data(),
                      out.scale.data());
}

void sme_pack_a_f32(const float* a, int m, int k, int lda, float* ap) {
  pack_f32_impl(a, m, k, lda, ap);
}

void sme_pack_a_f16(const float* a, int m, int k, int lda, uint16_t* ap) {
  pack_f32_to_f16_impl(a, m, k, lda, ap);
}

void sme_pack_a_bf16(const float* a, int m, int k, int lda, uint16_t* ap) {
  pack_f32_to_bf16_impl(a, m, k, lda, ap);
}

void sme_quant_pack_a_i8(const float* a, int m, int k, int lda, int8_t* ap,
                         float* scales) {
  quant_pack_i8_impl(a, m, k, lda, ap, scales);
}

void sme_quant_pack_a_i8g(const float* a, int m, int k, int lda, int group_size,
                          int8_t* ap, float* scales) {
  quant_pack_i8g_impl(a, m, k, lda, group_size, m, ap, scales);
}

void sme_gemm_f32_cols(const float* ap, const PackedF32& w, const float* bias,
                       float* c, int m, int ldc, int n0, int n1) {
  gemm_f32_cols_impl(ap, w.data.data(), bias, c, m, w.k, ldc, n0, n1);
}

void sme_gemm_f16_cols(const uint16_t* ap, const PackedF16& w,
                       const float* bias, float* c, int m, int ldc, int n0,
                       int n1) {
  gemm_f16_cols_impl(ap, w.data.data(), bias, c, m, w.k, ldc, n0, n1);
}

void sme_gemm_bf16_cols(const uint16_t* ap, const PackedBF16& w,
                        const float* bias, float* c, int m, int ldc, int n0,
                        int n1) {
  gemm_bf16_cols_impl(ap, w.data.data(), bias, c, m, w.k, ldc, n0, n1);
}

void sme_gemm_i8_cols(const int8_t* ap, const float* as, const PackedI8& w,
                      const float* bias, float* c, int m, int ldc, int n0,
                      int n1) {
  gemm_i8_cols_impl(ap, as, w.data.data(), w.scale.data(), bias, c, m, w.k, ldc,
                    n0, n1);
}

void sme_gemm_i8g_cols(const int8_t* ap, const float* as,
                       const PackedI8Group& w, const float* bias, float* c,
                       int m, int ldc, int n0, int n1) {
  const int n_padded = ((w.n + 15) / 16) * 16;
  gemm_i8g_cols_impl(ap, as, w.data.data(), w.scale.data(), bias, c, m,
                     n_padded, w.k, w.group_size, ldc, n0, n1);
}

void sme_gemm_f32_small(const float* a, int lda, const float* b, int ldb,
                        bool b_trans, float* c, int ldc, int m, int n, int k,
                        float* scratch_ap, float* scratch_bp) {
  pack_f32_impl(a, m, k, lda, scratch_ap);
  if (b_trans)
    pack_f32_impl(b, n, k, ldb, scratch_bp);
  else
    pack_f32_notrans_impl(b, k, n, ldb, scratch_bp);
  gemm_f32_cols_impl(scratch_ap, scratch_bp, nullptr, c, m, k, ldc, 0, n);
}

#else  // !VI_HAS_SME2

void sme_pack_weights_f32(const float*, int, int, PackedF32&) {}
void sme_pack_weights_f16(const float*, int, int, PackedF16&) {}
void sme_pack_weights_bf16(const float*, int, int, PackedBF16&) {}
void sme_pack_weights_i8(const float*, int, int, PackedI8&) {}
void sme_pack_weights_i8g(const float*, int, int, int, PackedI8Group&) {}
void sme_pack_a_f32(const float*, int, int, int, float*) {}
void sme_pack_a_f16(const float*, int, int, int, uint16_t*) {}
void sme_pack_a_bf16(const float*, int, int, int, uint16_t*) {}
void sme_quant_pack_a_i8(const float*, int, int, int, int8_t*, float*) {}
void sme_quant_pack_a_i8g(const float*, int, int, int, int, int8_t*, float*) {}
void sme_gemm_f32_cols(const float*, const PackedF32&, const float*, float*,
                       int, int, int, int) {}
void sme_gemm_f16_cols(const uint16_t*, const PackedF16&, const float*, float*,
                       int, int, int, int) {}
void sme_gemm_bf16_cols(const uint16_t*, const PackedBF16&, const float*,
                        float*, int, int, int, int) {}
void sme_gemm_i8_cols(const int8_t*, const float*, const PackedI8&,
                      const float*, float*, int, int, int, int) {}
void sme_gemm_i8g_cols(const int8_t*, const float*, const PackedI8Group&,
                       const float*, float*, int, int, int, int) {}
void sme_gemm_f32_small(const float*, int, const float*, int, bool, float*, int,
                        int, int, int, float*, float*) {}

#endif  // VI_HAS_SME2

}  // namespace vi
