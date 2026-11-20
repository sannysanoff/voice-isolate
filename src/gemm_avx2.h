// Packed unsigned-activation x signed-weight int8 GEMM for x86.
#pragma once

#include <cstdint>
#include <vector>

namespace vi {

struct Avx2Weights {
  struct alignas(64) Panel {
    int8_t bytes[16][4];
  };
  // [column tile][k/4], one cache line per depth step.
  std::vector<Panel> panels;
  std::vector<float> scales;
  std::vector<int32_t> offsets;
  int k = 0;
};

bool avx2_int8_available();
// n must be a multiple of 16, k a multiple of 4 and <= 65536.
void pack_avx2_weights(const float* w, int n, int k, Avx2Weights& out);
// Call only when avx2_int8_available(). Scratch holds rows*k bytes.
void quantize_avx2_rows(const float* a, int rows, int k, int stride,
                        uint8_t* scratch, float* scales);
// Each tile writes 16 columns of the full row-major output.
void gemm_avx2_tiles(const uint8_t* a, const float* scales, int rows,
                     const Avx2Weights& w, const float* bias, float* c,
                     int tile_begin, int tile_end);

}  // namespace vi
