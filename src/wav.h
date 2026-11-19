// Minimal WAV I/O: reads any PCM/float WAV as mono float, writes float or
// 16-bit PCM. No dependencies beyond the standard library.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vi {

struct Audio {
  std::vector<float> samples;  // mono
  int sample_rate = 0;
  int channels = 0;
  int source_bits = 0;
};

// Reads a WAV file, averaging channels like the reference loader does.
Audio read_wav(const std::string& path);

// Writes mono float samples. `bits` 16 scales to PCM; 32 writes float32.
void write_wav(const std::string& path, const std::vector<float>& samples,
               int sample_rate, int bits, bool normalize_peak);

}  // namespace vi
