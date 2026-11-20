// Optional Apple Metal GPU backend for the isolator backbone.
// Pure C++ header; Objective-C++ / Metal lives strictly inside metal_backend.mm.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common.h"

namespace vi {

struct Rope;
struct LaneParams;

// Read-only descriptor of one backbone block's weights, passed once at init.
struct MetalMambaLayerDesc {
  const float* in_w = nullptr;    // [2*inner, width]
  const float* conv_t = nullptr;  // [conv, inner]
  const float* conv_b = nullptr;  // [inner]
  const float* x_w = nullptr;     // [dt_total, inner]
  const float* dt_w = nullptr;    // [inner, dt_rank]
  const float* dt_b = nullptr;    // [inner]
  const float* d = nullptr;       // [inner]
  const float* a = nullptr;       // [state, inner] = -exp(A_log)
  const float* out_w = nullptr;   // [width, inner]
};

struct MetalBlockDesc {
  bool transformer = false;
  // Transformer weights
  const float *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;
  const float *qkv_w = nullptr, *qkv_b = nullptr;
  const float *proj_w = nullptr, *proj_b = nullptr;
  const float *f0w = nullptr, *f0b = nullptr, *f1w = nullptr, *f1b = nullptr;
  const float *bn1w = nullptr, *bn1b = nullptr, *bn2w = nullptr, *bn2b = nullptr;
  const float *in_w = nullptr, *in_b = nullptr;
  const float *out_w = nullptr, *out_b = nullptr;
  const float *bl1w = nullptr, *bl1b = nullptr, *bl2w = nullptr, *bl2b = nullptr;
  // Mamba weights
  MetalMambaLayerDesc time_fwd;
  const float *tm_out_w = nullptr, *tm_out_b = nullptr;
  const float *tm_norm_w = nullptr, *tm_norm_b = nullptr;
  MetalMambaLayerDesc freq_fwd;
  MetalMambaLayerDesc freq_bwd;
  const float *fm_out_w = nullptr, *fm_out_b = nullptr;
  const float *fm_norm_w = nullptr, *fm_norm_b = nullptr;
};

struct MetalLaneState {
  virtual ~MetalLaneState() = default;
};

bool metal_available();

class MetalContext {
 public:
  static std::unique_ptr<MetalContext> create(
      const Config& cfg, const std::vector<MetalBlockDesc>& blocks,
      bool fp16_weights, bool fp16_kv);

  virtual ~MetalContext() = default;

  virtual std::string device_name() const = 0;
  virtual bool fp16_weights() const = 0;
  virtual bool fp16_kv() const = 0;
  virtual void set_precision(bool fp16_weights, bool fp16_kv) = 0;

  virtual std::shared_ptr<MetalLaneState> init_lane() const = 0;
  virtual std::shared_ptr<MetalLaneState> clone_lane(
      const std::shared_ptr<MetalLaneState>& src) const = 0;

  virtual void backbone(
      float* feat, int frames, std::shared_ptr<MetalLaneState>& lane,
      const Rope& rope, const LaneParams& lp,
      const std::function<void(int, int, const float*, const float*)>&
          debug_block) const = 0;

  virtual void backbone_pair(float* feat_a, float* feat_b, int frames,
                             std::shared_ptr<MetalLaneState>& lane_a,
                             std::shared_ptr<MetalLaneState>& lane_b,
                             const Rope& rope,
                             const LaneParams& lp) const = 0;
};

}  // namespace vi
