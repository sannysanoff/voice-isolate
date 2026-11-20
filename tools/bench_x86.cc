// Microbench for x86 hot kernels.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "vecmath.h"
#include "model.h"

using namespace vi;
using clk = std::chrono::steady_clock;

static double bench(const char* name, int iters, std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 2; ++i) fn();
  auto t0 = clk::now();
  for (int i = 0; i < iters; ++i) fn();
  auto t1 = clk::now();
  double s = std::chrono::duration<double>(t1 - t0).count();
  printf("%-28s %8.3f ms/iter (%d iters)\n", name, s / iters * 1e3, iters);
  return s / iters;
}

static std::vector<float> q0, q1, q2, q3, keys, vals, s0, s1, s2, s3, o0, o1, o2, o3, buf, buf2;
static std::vector<float> gelu_buf, sp_buf, silu_buf, exp_buf;

int main() {
  const int n = 600, hd = 64;
  q0.assign(hd, 0.1f); q1.assign(hd, 0.2f); q2.assign(hd, -0.15f); q3.assign(hd, 0.05f);
  keys.assign(n * hd, 0.0f); vals.assign(n * hd, 0.0f);
  srand(42);
  for (auto& v : keys) v = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
  for (auto& v : vals) v = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
  for (auto& v : q0) v = (rand() / (float)RAND_MAX - 0.5f);
  for (auto& v : q1) v = (rand() / (float)RAND_MAX - 0.5f);
  for (auto& v : q2) v = (rand() / (float)RAND_MAX - 0.5f);
  for (auto& v : q3) v = (rand() / (float)RAND_MAX - 0.5f);
  s0.assign(n, 0); s1.assign(n, 0); s2.assign(n, 0); s3.assign(n, 0);
  o0.assign(hd, 0); o1.assign(hd, 0); o2.assign(hd, 0); o3.assign(hd, 0);
  buf.assign(n, 0); buf2.assign(n, 0);
  const float scale = 1.0f / std::sqrt((float)hd);

  bench("v_attn4_qk n=600 hd=64", 200, [&] {
    v_attn4_qk(q0.data(), q1.data(), q2.data(), q3.data(), keys.data(), n, hd, scale,
               s0.data(), s1.data(), s2.data(), s3.data());
  });
  // softmax x4
  bench("v_softmax_norm x4 n=600", 200, [&] {
    v_softmax_norm(s0.data(), n); v_softmax_norm(s1.data(), n);
    v_softmax_norm(s2.data(), n); v_softmax_norm(s3.data(), n);
  });
  bench("v_attn4_pv n=600 hd=64", 200, [&] {
    v_attn4_pv(s0.data(), s1.data(), s2.data(), s3.data(), vals.data(), n, hd,
               o0.data(), o1.data(), o2.data(), o3.data());
  });
  bench("v_exp n=600", 2000, [&] {
    for (int i = 0; i < n; ++i) buf[i] = s0[i % n] - 1.0f;
    v_exp(buf.data(), n);
  });
  const int big = 276 * 2048;
  gelu_buf.assign(big, 0.3f);
  for (size_t i = 0; i < gelu_buf.size(); ++i) gelu_buf[i] = (rand() / (float)RAND_MAX - 0.5f) * 4;
  bench("v_gelu n=565K", 10, [&] { v_gelu(gelu_buf.data(), big); });
  const int mrot = 276 * 1024;
  sp_buf.assign(mrot, 0.2f);
  for (size_t i = 0; i < sp_buf.size(); ++i) sp_buf[i] = (rand() / (float)RAND_MAX - 0.5f) * 3;
  bench("v_softplus n=282K", 10, [&] { v_softplus(sp_buf.data(), mrot); });
  silu_buf.assign(mrot, 0.1f);
  for (size_t i = 0; i < silu_buf.size(); ++i) silu_buf[i] = (rand() / (float)RAND_MAX - 0.5f) * 4;
  bench("v_silu n=282K", 10, [&] { v_silu(silu_buf.data(), mrot); });
  exp_buf.assign(600, 0.0f);
  for (int i = 0; i < 600; ++i) exp_buf[i] = -(rand() / (float)RAND_MAX) * 10;
  bench("v_exp n=600 (neg)", 5000, [&] { v_exp(exp_buf.data(), 600); });
  // layer norm
  std::vector<float> lnx(276 * 512, 0.5f), lny(276 * 512), w(512, 1.0f), b(512, 0.0f);
  bench("layer_norm 276x512", 20, [&] { layer_norm(lnx.data(), lny.data(), 276, 512, w.data(), b.data()); });
  return 0;
}
