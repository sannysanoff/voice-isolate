#include "common.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace vi {
namespace {

std::string join(const std::string& dir, const std::string& name) {
  if (!dir.empty() && dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> parts;
  std::istringstream in(line);
  std::string part;
  while (in >> part) parts.push_back(part);
  return parts;
}

}  // namespace

void Weights::load(const std::string& dir) {
  std::ifstream manifest(join(dir, "weights.txt"));
  if (!manifest) throw std::runtime_error("cannot open " + join(dir, "weights.txt"));
  std::string line;
  struct Entry {
    std::string name;
    std::vector<int> shape;
    size_t offset;
  };
  std::vector<Entry> entries;
  while (std::getline(manifest, line)) {
    if (line.empty()) continue;
    const auto parts = split(line);
    if (parts.size() < 3) throw std::runtime_error("bad manifest line: " + line);
    const int rank = std::stoi(parts[1]);
    Entry entry;
    entry.name = parts[0];
    for (int i = 0; i < rank; ++i)
      entry.shape.push_back(std::stoi(parts[2 + static_cast<size_t>(i)]));
    entry.offset = static_cast<size_t>(
        std::stoll(parts[2 + static_cast<size_t>(rank)]));
    entries.push_back(std::move(entry));
  }

  std::ifstream blob(join(dir, "weights.bin"), std::ios::binary | std::ios::ate);
  if (!blob) throw std::runtime_error("cannot open " + join(dir, "weights.bin"));
  const auto bytes = static_cast<size_t>(blob.tellg());
  blob.seekg(0);
  blob_.resize(bytes / sizeof(float));
  if (!blob.read(reinterpret_cast<char*>(blob_.data()),
                 static_cast<std::streamsize>(blob_.size() * sizeof(float))))
    throw std::runtime_error("short read on " + join(dir, "weights.bin"));
#if defined(__linux__)
  // The blob (232 MB) is streamed every block; on madvise-mode THP systems
  // huge pages cut the per-block TLB misses ~500x. Silent no-op otherwise.
  madvise(blob_.data(), blob_.size() * sizeof(float), MADV_HUGEPAGE);
#endif

  for (const auto& entry : entries) {
    Tensor tensor;
    tensor.shape = entry.shape;
    tensor.data = blob_.data() + entry.offset;
    if (entry.offset + tensor.size() > blob_.size())
      throw std::runtime_error("tensor out of range: " + entry.name);
    tensors_.emplace(entry.name, std::move(tensor));
  }
}

const Tensor& Weights::get(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("missing tensor: " + name);
  return it->second;
}

const Tensor& Weights::get(const std::string& name,
                           std::vector<int> shape) const {
  const Tensor& t = get(name);
  if (t.shape != shape) {
    std::string got, want;
    for (int d : t.shape) got += std::to_string(d) + "x";
    for (int d : shape) want += std::to_string(d) + "x";
    throw std::runtime_error("shape mismatch for " + name + ": got " + got +
                             "want " + want);
  }
  return t;
}

Config load_config(const std::string& dir) {
  std::ifstream in(join(dir, "config.txt"));
  if (!in) throw std::runtime_error("cannot open " + join(dir, "config.txt"));
  Config c;
  std::string key;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto parts = split(line);
    if (parts.size() < 2) continue;
    key = parts[0];
    if (key == "n_fft") c.n_fft = std::stoi(parts[1]);
    else if (key == "hop_length") c.hop = std::stoi(parts[1]);
    else if (key == "sample_rate") c.sample_rate = std::stoi(parts[1]);
    else if (key == "width") c.width = std::stoi(parts[1]);
    else if (key == "num_heads") c.heads = std::stoi(parts[1]);
    else if (key == "mamba_state") c.mamba_state = std::stoi(parts[1]);
    else if (key == "mamba_conv") c.mamba_conv = std::stoi(parts[1]);
    else if (key == "mamba_expand") c.mamba_expand = std::stoi(parts[1]);
    else if (key == "correction_bound")
      c.correction_bound = std::stof(parts[1]);
    else if (key == "chunk_seconds") c.chunk_seconds = std::stof(parts[1]);
    else if (key == "attention_window_frames")
      c.attention_window_frames = std::stoi(parts[1]);
    else if (key == "lookahead_frames") c.lookahead_frames = std::stoi(parts[1]);
    else if (key == "lookahead_index") c.lookahead_index = std::stoi(parts[1]);
    else if (key == "num_blocks") c.num_blocks = std::stoi(parts[1]);
    else if (key == "block_types")
      c.block_types.assign(parts.begin() + 1, parts.end());
    else if (key == "num_bands") {
      const int bands = std::stoi(parts[1]);
      c.edges.reserve(static_cast<size_t>(bands));
    } else if (key == "band_edges") {
      c.edges.clear();
      for (size_t i = 1; i + 1 < parts.size(); i += 2)
        c.edges.emplace_back(std::stoi(parts[i]), std::stoi(parts[i + 1]));
    }
  }
  if (c.num_blocks <= 0 || c.edges.empty() ||
      static_cast<int>(c.block_types.size()) != c.num_blocks)
    throw std::runtime_error("incomplete config.txt in " + dir);
  return c;
}

}  // namespace vi
