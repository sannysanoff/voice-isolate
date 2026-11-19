#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace vi {
namespace {

uint32_t read_u32(const unsigned char* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t read_u16(const unsigned char* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
void write_u32(std::vector<unsigned char>& out, uint32_t v) {
  out.push_back(static_cast<unsigned char>(v & 0xff));
  out.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
  out.push_back(static_cast<unsigned char>((v >> 16) & 0xff));
  out.push_back(static_cast<unsigned char>((v >> 24) & 0xff));
}
void write_u16(std::vector<unsigned char>& out, uint16_t v) {
  out.push_back(static_cast<unsigned char>(v & 0xff));
  out.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
}

}  // namespace

Audio read_wav(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) throw std::runtime_error("cannot open " + path);
  std::vector<unsigned char> bytes;
  {
    unsigned char buffer[1 << 16];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0)
      bytes.insert(bytes.end(), buffer, buffer + got);
  }
  std::fclose(file);
  if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
    throw std::runtime_error(path + ": not a RIFF/WAVE file");

  int format = 0, channels = 0, bits = 0, sample_rate = 0;
  const unsigned char* data = nullptr;
  size_t data_size = 0;
  size_t pos = 12;
  while (pos + 8 <= bytes.size()) {
    const unsigned char* chunk = bytes.data() + pos;
    const uint32_t size = read_u32(chunk + 4);
    const size_t body = pos + 8;
    if (std::memcmp(chunk, "fmt ", 4) == 0 && body + 16 <= bytes.size()) {
      format = read_u16(bytes.data() + body);
      channels = read_u16(bytes.data() + body + 2);
      sample_rate = static_cast<int>(read_u32(bytes.data() + body + 4));
      bits = read_u16(bytes.data() + body + 14);
      if (format == 0xFFFE && body + 26 <= bytes.size())
        format = read_u16(bytes.data() + body + 24);
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      data = bytes.data() + body;
      data_size = std::min<size_t>(size, bytes.size() - body);
    }
    pos = body + size + (size & 1);
  }
  if (channels < 1 || data == nullptr || bits == 0)
    throw std::runtime_error(path + ": incomplete WAV header");

  const int width = bits / 8;
  const size_t frames = data_size / static_cast<size_t>(width * channels);
  Audio audio;
  audio.sample_rate = sample_rate;
  audio.channels = channels;
  audio.source_bits = bits;
  audio.samples.resize(frames);
  for (size_t f = 0; f < frames; ++f) {
    double sum = 0.0;
    for (int c = 0; c < channels; ++c) {
      const unsigned char* p =
          data + (f * static_cast<size_t>(channels) + c) * width;
      float v = 0.0f;
      if (format == 3 && bits == 32) {
        float raw;
        std::memcpy(&raw, p, 4);
        v = raw;
      } else if (bits == 8) {
        v = (static_cast<float>(p[0]) - 128.0f) / 128.0f;
      } else if (bits == 16) {
        const int16_t s = static_cast<int16_t>(p[0] | (p[1] << 8));
        v = static_cast<float>(s) / 32768.0f;
      } else if (bits == 24) {
        int32_t s = static_cast<int32_t>(p[0] | (p[1] << 8) | (p[2] << 16));
        if (s & 0x800000) s |= ~0xFFFFFF;
        v = static_cast<float>(s) / 8388608.0f;
      } else if (bits == 32) {
        const int32_t s = static_cast<int32_t>(read_u32(p));
        v = static_cast<float>(static_cast<double>(s) / 2147483648.0);
      } else {
        throw std::runtime_error(path + ": unsupported sample width");
      }
      sum += v;
    }
    audio.samples[f] = static_cast<float>(sum / channels);
  }
  return audio;
}

void write_wav(const std::string& path, const std::vector<float>& samples,
               int sample_rate, int bits, bool normalize_peak) {
  std::vector<float> y = samples;
  if (bits == 16 && normalize_peak) {
    float peak = 0.0f;
    for (float v : y) peak = std::max(peak, std::fabs(v));
    if (peak > 1.0f)
      for (float& v : y) v /= peak;
  }
  const int width = bits / 8;
  std::vector<unsigned char> body;
  body.reserve(y.size() * static_cast<size_t>(width));
  for (float v : y) {
    if (bits == 32) {
      unsigned char tmp[4];
      std::memcpy(tmp, &v, 4);
      body.insert(body.end(), tmp, tmp + 4);
    } else {
      const float clipped = std::max(-1.0f, std::min(1.0f, v));
      const int32_t s = static_cast<int32_t>(std::lrint(clipped * 32767.0f));
      write_u16(body, static_cast<uint16_t>(s & 0xffff));
    }
  }
  std::vector<unsigned char> header;
  header.insert(header.end(), {'R', 'I', 'F', 'F'});
  write_u32(header, static_cast<uint32_t>(36 + body.size()));
  header.insert(header.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  write_u32(header, 16);
  write_u16(header, bits == 32 ? 3 : 1);
  write_u16(header, 1);
  write_u32(header, static_cast<uint32_t>(sample_rate));
  write_u32(header, static_cast<uint32_t>(sample_rate * width));
  write_u16(header, static_cast<uint16_t>(width));
  write_u16(header, static_cast<uint16_t>(bits));
  header.insert(header.end(), {'d', 'a', 't', 'a'});
  write_u32(header, static_cast<uint32_t>(body.size()));

  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) throw std::runtime_error("cannot write " + path);
  std::fwrite(header.data(), 1, header.size(), file);
  std::fwrite(body.data(), 1, body.size(), file);
  std::fclose(file);
}

}  // namespace vi
