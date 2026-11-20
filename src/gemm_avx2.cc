#include "gemm_avx2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define VI_AVX2 __attribute__((target("avx2")))
#endif

namespace vi {

bool avx2_int8_available() {
#ifdef VI_AVX2
  static const bool available = __builtin_cpu_supports("avx2");
  return available;
#else
  return false;
#endif
}

void pack_avx2_weights(const float* w, int n, int k, Avx2Weights& out) {
  out.k = k;
  out.scales.resize(n);
  out.offsets.resize(n);
  out.panels.resize(static_cast<size_t>(n / 16) * (k / 4));
  for (int j = 0; j < n; ++j) {
    const float* row = w + static_cast<size_t>(j) * k;
    // Bound each same-sign pair as well as each byte. With rounding, the
    // positive and negative pair sums then fit +/-128, so even activation
    // bytes of 255 cannot saturate vpmaddubsw's signed 16-bit result.
    double range = 0;
    for (int t = 0; t < k; t += 2) {
      const double a = row[t], b = row[t + 1];
      range = std::max({range, std::abs(a), std::abs(b), std::abs(a + b)});
    }
    const float scale = range > 0
                            ? std::max(static_cast<float>(range / 127.0),
                                       std::numeric_limits<float>::min())
                            : 1.0f;
    out.scales[j] = scale;
    int sum = 0;
    auto* panel = out.panels.data() + static_cast<size_t>(j / 16) * (k / 4);
    for (int t = 0; t < k; ++t) {
      const int q = static_cast<int>(std::round(row[t] / scale));
      panel[t / 4].bytes[j % 16][t % 4] = static_cast<int8_t>(q);
      sum += q;
    }
    out.offsets[j] = -128 * sum;
  }
#if defined(__linux__)
  // Packed panels stream every GEMM; huge pages cut TLB misses on THP systems.
  if (!out.panels.empty())
    madvise(out.panels.data(), out.panels.size() * sizeof(Avx2Weights::Panel),
            MADV_HUGEPAGE);
#endif
}

#ifdef VI_AVX2
namespace {

VI_AVX2 void quantize_row(const float* a, int k, uint8_t* dst, float& scale) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  __m256 vmax = _mm256_setzero_ps();
  int t = 0;
  for (; t + 8 <= k; t += 8)
    vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(a + t)));
  alignas(32) float maxima[8];
  _mm256_store_ps(maxima, vmax);
  float range = *std::max_element(maxima, maxima + 8);
  for (; t < k; ++t) range = std::max(range, std::abs(a[t]));
  scale = range > 0 ? std::max(range / 127.f, std::numeric_limits<float>::min()) : 1.f;
  const float inv = 1.0f / scale;
  const __m256 factor = _mm256_set1_ps(inv);
  const __m128i offset = _mm_set1_epi8(-128);
  t = 0;
  for (; t + 8 <= k; t += 8) {
    __m256 v = _mm256_mul_ps(_mm256_loadu_ps(a + t), factor);
    __m256i q = _mm256_cvtps_epi32(v);
    __m128i s = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + t),
                     _mm_xor_si128(_mm_packs_epi16(s, s), offset));
  }
  for (; t < k; ++t) dst[t] = static_cast<uint8_t>(128 + std::nearbyint(a[t] * inv));
}

// Each weight load feeds four rows and each activation broadcast feeds
// sixteen outputs. Both halves of the packed cache line are consumed.
template <int Rows, bool WithBias>
VI_AVX2 __attribute__((always_inline)) inline void tile(
    const uint8_t* a, const float* scales, const Avx2Weights& w, int j,
    const float* bias, float* c) {
  const int n = static_cast<int>(w.scales.size());
  const auto* panel = w.panels.data() + static_cast<size_t>(j / 16) * (w.k / 4);
  __m256i lo[Rows], hi[Rows];
  const __m256i offlo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.offsets.data() + j));
  const __m256i offhi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.offsets.data() + j + 8));
  for (int r = 0; r < Rows; ++r) { lo[r] = offlo; hi[r] = offhi; }
  const __m256i ones = _mm256_set1_epi16(1);
  for (int t = 0; t < w.k; t += 4, ++panel) {
    const auto* p = reinterpret_cast<const __m256i*>(panel);
    const __m256i w0 = _mm256_load_si256(p), w1 = _mm256_load_si256(p + 1);
    for (int r = 0; r < Rows; ++r) {
      int32_t bytes;
      std::memcpy(&bytes, a + r * w.k + t, sizeof(bytes));
      const __m256i v = _mm256_set1_epi32(bytes);
      lo[r] = _mm256_add_epi32(lo[r], _mm256_madd_epi16(_mm256_maddubs_epi16(v, w0), ones));
      hi[r] = _mm256_add_epi32(hi[r], _mm256_madd_epi16(_mm256_maddubs_epi16(v, w1), ones));
    }
  }
  for (int r = 0; r < Rows; ++r) {
    const __m256 s = _mm256_set1_ps(scales[r]);
    for (int half = 0; half < 2; ++half) {
      const int col = j + half * 8;
      __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(half ? hi[r] : lo[r]), s);
      v = _mm256_mul_ps(v, _mm256_loadu_ps(w.scales.data() + col));
      if (WithBias) v = _mm256_add_ps(v, _mm256_loadu_ps(bias + col));
      _mm256_storeu_ps(c + static_cast<size_t>(r) * n + col, v);
    }
  }
}
// Specialize the epilogue once instead of testing `bias` inside every tile.
template <bool WithBias>
VI_AVX2 void gemm_tiles_impl(const uint8_t* a, const float* scales, int rows,
                             const Avx2Weights& w, const float* bias, float* c,
                             int tile_begin, int tile_end) {
  const int n = static_cast<int>(w.scales.size());
  for (int j = 16 * tile_begin; j < 16 * tile_end; j += 16) {
    int i = 0;
    for (; i + 4 <= rows; i += 4)
      tile<4, WithBias>(a + static_cast<size_t>(i) * w.k, scales + i, w, j, bias,
                        c + static_cast<size_t>(i) * n);
    for (; i < rows; ++i)
      tile<1, WithBias>(a + static_cast<size_t>(i) * w.k, scales + i, w, j, bias,
                        c + static_cast<size_t>(i) * n);
  }
}
}  // namespace

VI_AVX2
#endif
void quantize_avx2_rows(const float* a, int rows, int k, int stride,
                        uint8_t* scratch, float* scales) {
#ifdef VI_AVX2
  for (int i = 0; i < rows; ++i)
    quantize_row(a + static_cast<size_t>(i) * stride, k,
                 scratch + static_cast<size_t>(i) * k, scales[i]);
#else
  throw std::logic_error("AVX2 kernel unavailable");
#endif
}

#ifdef VI_AVX2
VI_AVX2
#endif
void gemm_avx2_tiles(const uint8_t* a, const float* scales, int rows,
                     const Avx2Weights& w, const float* bias, float* c,
                     int tile_begin, int tile_end) {
#ifdef VI_AVX2
  if (bias) gemm_tiles_impl<true>(a, scales, rows, w, bias, c, tile_begin, tile_end);
  else      gemm_tiles_impl<false>(a, scales, rows, w, bias, c, tile_begin, tile_end);
#else
  throw std::logic_error("AVX2 kernel unavailable");
#endif
}
}  // namespace vi
