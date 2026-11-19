// int8 weight-only quantisation of the dense layers.
//
// The dense layers are 76 % of the run time and the model is compute-bound in
// fp32, so the only way forward is more multiply-accumulates per cycle. The
// CPU's int8 dot product (`sdot`) does 16 MACs per instruction against 4 for an
// fp32 FMA, and it also cuts the weight stream to a quarter.
//
// Weights are quantised once, offline, per output channel (symmetric, int8).
// Activations are quantised per row at call time, which keeps the maths honest
// for the layers whose inputs have wildly different magnitudes.
#pragma once

#include <cstdint>
#include <vector>

namespace vi {

// q[j][k] with one scale per output channel.
struct Int8Matrix {
  std::vector<int8_t> q;
  std::vector<float> scale;
  int n = 0;
  int k = 0;
};

// True when the build has a vectorised int8 dot product. Without one the int8
// path is a scalar fallback and is slower than fp32, so callers skip it.
bool int8_available();

// Quantises an [n][k] row-major matrix, one symmetric scale per row.
void quantize_rows(const float* w, int n, int k, Int8Matrix& out);

// C[i][j] = sum_k A[i][k] * W[j][k] * (a_scale[i] * w_scale[j]) + bias[j],
// for the output columns [j0, j1). A is quantised per row into `aq`/`as`, which
// are scratch owned by the caller (m*k int8 and m float per worker).
void gemm_int8_columns(const float* a, int m, int k, int stride,
                       const Int8Matrix& w, const float* bias, float* c, int j0,
                       int j1, int8_t* aq, float* as);

// One row of activations to int8, symmetric about zero.
void quantize_activations(const float* a, int m, int k, int stride, int8_t* aq,
                          float* scale);

}  // namespace vi
