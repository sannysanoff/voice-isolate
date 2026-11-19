// Real FFT of the model's window length, on top of pocketfft.
#pragma once

#include <complex>
#include <vector>

namespace vi {

class Fft {
 public:
  explicit Fft(int n);

  int size() const { return n_; }
  int bins() const { return n_ / 2 + 1; }

  // Unnormalized forward transform, `bins()` complex outputs.
  void rfft(const float* in, std::complex<float>* out) const;
  // Inverse transform scaled by 1/n, so rfft -> irfft is the identity.
  void irfft(const std::complex<float>* in, float* out) const;

 private:
  int n_;
  mutable std::vector<std::complex<float>> scratch_;
  mutable std::vector<float> scratch_real_;
};

}  // namespace vi
