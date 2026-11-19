#include "fft.h"

#include <cstring>

// Keep built plans around: the model transforms 100 windows per second.
#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"

namespace vi {

Fft::Fft(int n) : n_(n), scratch_(static_cast<size_t>(n / 2 + 1)),
                  scratch_real_(static_cast<size_t>(n)) {
  if (n < 4 || (n & 1)) throw std::runtime_error("Fft: even size >= 4 needed");
}

void Fft::rfft(const float* in, std::complex<float>* out) const {
  const pocketfft::shape_t shape{static_cast<size_t>(n_)};
  const pocketfft::stride_t stride_in{sizeof(float)};
  const pocketfft::stride_t stride_out{sizeof(std::complex<float>)};
  pocketfft::r2c<float>(shape, stride_in, stride_out, 0, true, in, out, 1.0f);
}

void Fft::irfft(const std::complex<float>* in, float* out) const {
  const pocketfft::shape_t shape{static_cast<size_t>(n_)};
  const pocketfft::stride_t stride_in{sizeof(std::complex<float>)};
  const pocketfft::stride_t stride_out{sizeof(float)};
  const float scale = 1.0f / static_cast<float>(n_);
  pocketfft::c2r<float>(shape, stride_in, stride_out, 0, false, in, out, scale);
}

}  // namespace vi
