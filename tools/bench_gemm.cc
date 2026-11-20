// Single-thread AVX2 int8 GEMM throughput probe.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gemm_avx2.h"

using namespace vi;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  int m = argc > 1 ? atoi(argv[1]) : 184;
  int k = argc > 2 ? atoi(argv[2]) : 512;
  int n = argc > 3 ? atoi(argv[3]) : 1536;
  int iters = argc > 4 ? atoi(argv[4]) : 50;
  printf("avx2_int8_available=%d m=%d k=%d n=%d\n", avx2_int8_available(), m, k,
         n);
  std::vector<float> a((size_t)m * k), w((size_t)n * k), c((size_t)m * n),
      bias(n, 0.1f);
  srand(123);
  for (auto& v : a) v = (rand() / (float)RAND_MAX - 0.5f) * 2;
  for (auto& v : w) v = (rand() / (float)RAND_MAX - 0.5f) * 2;
  Avx2Weights pw;
  pack_avx2_weights(w.data(), n, k, pw);
  std::vector<uint8_t> aq((size_t)m * k);
  std::vector<float> sc(m);
  quantize_avx2_rows(a.data(), m, k, k, aq.data(), sc.data());
  auto t0 = clk::now();
  for (int i = 0; i < iters; ++i)
    gemm_avx2_tiles(aq.data(), sc.data(), m, pw, bias.data(), c.data(), 0,
                    n / 16);
  auto t1 = clk::now();
  double s = std::chrono::duration<double>(t1 - t0).count() / iters;
  double gflops = 2.0 * m * k * n / s / 1e9;
  // checksum to prevent DCE
  double chk = 0;
  for (size_t i = 0; i < c.size(); i += 97) chk += c[i];
  printf("per-iter %.3f ms  %.1f GOPS  chk %.6f\n", s * 1e3, gflops, chk);
  // with quantization included
  t0 = clk::now();
  for (int i = 0; i < iters; ++i) {
    quantize_avx2_rows(a.data(), m, k, k, aq.data(), sc.data());
    gemm_avx2_tiles(aq.data(), sc.data(), m, pw, bias.data(), c.data(), 0,
                    n / 16);
  }
  t1 = clk::now();
  s = std::chrono::duration<double>(t1 - t0).count() / iters;
  printf("with-quant %.3f ms  %.1f GOPS\n", s * 1e3, 2.0 * m * k * n / s / 1e9);
  return 0;
}
