#include "stream.h"

#include <algorithm>
#include <cstring>

namespace vi {
namespace {

int64_t floor_div(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

}  // namespace

Stream::Stream(std::shared_ptr<Model> model, const StreamOptions& options)
    : model_(std::move(model)), opt_(options) {
  const Config& cfg = model_->config();
  windowed_ = opt_.window_frames > 0;
  if (windowed_ && !(opt_.crossfade_frames > 0 &&
                     opt_.crossfade_frames < opt_.window_frames))
    throw std::runtime_error(
        "crossfade_frames must be positive and shorter than window_frames");
  if (opt_.block_frames < 1) throw std::runtime_error("block_frames must be >= 1");

  lane_params_.window = cfg.attention_window_frames;
  lane_params_.cache_frames =
      opt_.cache_frames > 0
          ? opt_.cache_frames
          : (cfg.attention_window_frames > 0
                 ? cfg.attention_window_frames
                 : static_cast<int>(std::lround(cfg.chunk_seconds * cfg.sample_rate / cfg.hop)));
  if (!windowed_) {
    lane_params_.capacity_limit =
        lane_params_.cache_frames +
        std::max(lane_params_.cache_frames / 2, opt_.block_frames);
  } else {
    const int lane = opt_.window_frames + opt_.crossfade_frames;
    lane_params_.capacity_limit =
        lane <= lane_params_.cache_frames + opt_.block_frames
            ? lane
            : lane_params_.cache_frames +
                  std::max(lane_params_.cache_frames / 2, opt_.block_frames);
  }

  model_->init_lane(lane_a_);
  rope_.init(cfg.head_dim(), lane_params_.capacity_limit + opt_.block_frames + 1);

  ola_.assign(static_cast<size_t>(cfg.n_fft), 0.0f);
  envelope_.assign(static_cast<size_t>(cfg.n_fft), 0.0f);
  ola_base_ = -cfg.n_fft / 2;

  const size_t block = static_cast<size_t>(opt_.block_frames);
  spectra_.resize(block * static_cast<size_t>(cfg.bins()));
  feat_a_.resize(block * static_cast<size_t>(cfg.bands()) * cfg.width);
  feat_b_.resize(feat_a_.size());
  raw_a_.resize(block * static_cast<size_t>(cfg.bins()) * 4);
  raw_b_.resize(raw_a_.size());
  frame_audio_.resize(static_cast<size_t>(cfg.n_fft));
  delta_spec_.resize(static_cast<size_t>(cfg.bins()));
}

int Stream::sample_rate() const { return model_->config().sample_rate; }

int Stream::latency_samples() const {
  const Config& cfg = model_->config();
  return cfg.n_fft + cfg.hop * cfg.lookahead_frames;
}

int Stream::lookahead_frames() const {
  return model_->config().lookahead_frames;
}

int Stream::attention_capacity() const { return lane_params_.capacity_limit; }

int64_t Stream::next_cut() const {
  if (!windowed_) return frames_ + opt_.block_frames;
  const int64_t start = frames_ - frames_ % opt_.window_frames;
  const int64_t crossfade_start =
      start + opt_.window_frames - opt_.crossfade_frames;
  if (frames_ < crossfade_start) return crossfade_start;
  return start + opt_.window_frames;
}

void Stream::analyze(int64_t first, int count) {
  const Config& cfg = model_->config();
  const int n_fft = cfg.n_fft;
  const int hop = cfg.hop;
  const int64_t total = audio_base_ + static_cast<int64_t>(audio_.size());
  for (int t = 0; t < count; ++t) {
    const int64_t start = (first + t) * hop - n_fft / 2;
    float* seg = frame_audio_.data();
    const int64_t lo = std::max(start, audio_base_);
    const int64_t hi = std::min(start + n_fft, total);
    if (hi > lo) {
      std::memcpy(seg + (lo - start),
                  audio_.data() + (lo - audio_base_),
                  sizeof(float) * static_cast<size_t>(hi - lo));
      if (lo > start)
        std::memset(seg, 0, sizeof(float) * static_cast<size_t>(lo - start));
      if (hi < start + n_fft)
        std::memset(seg + (hi - start), 0,
                    sizeof(float) * static_cast<size_t>(start + n_fft - hi));
    } else {
      std::memset(seg, 0, sizeof(float) * static_cast<size_t>(n_fft));
    }
    for (int i = 0; i < n_fft; ++i) seg[i] *= model_->window()[static_cast<size_t>(i)];
    model_->fft().rfft(seg, spectra_.data() + static_cast<size_t>(t) * cfg.bins());
  }
}

void Stream::run_frames(int64_t first, int count) {
  const Config& cfg = model_->config();
  const size_t frame_bins = static_cast<size_t>(cfg.bins());
  const size_t feat_size = static_cast<size_t>(cfg.bands()) * cfg.width;

  analyze(first, count);

  if (!windowed_) {
    model_->encode(spectra_.data(), count, feat_a_.data());
    model_->backbone(feat_a_.data(), count, lane_a_, rope_, lane_params_);
    model_->decode(feat_a_.data(), count, raw_a_.data());
    for (int t = 0; t < count; ++t) {
      std::vector<float> raw(raw_a_.begin() + static_cast<long>(t * frame_bins * 4),
                             raw_a_.begin() +
                                 static_cast<long>((t + 1) * frame_bins * 4));
      raw_q_.push_back(std::move(raw));
      std::vector<std::complex<float>> spec(
          spectra_.begin() + static_cast<long>(t * frame_bins),
          spectra_.begin() + static_cast<long>((t + 1) * frame_bins));
      spec_q_.push_back(std::move(spec));
    }
    return;
  }

  const int within = static_cast<int>(first % opt_.window_frames);
  const int crossfade_start = opt_.window_frames - opt_.crossfade_frames;
  if (within >= crossfade_start && !lane_b_started_) {
    // A second lane starts at the crossfade with the recurrence cloned and no
    // attention past: it reaches the boundary warmed up.
    model_->clone_lane(lane_a_, lane_b_);
    lane_b_started_ = true;
  }

  if (within < crossfade_start) {
    model_->encode(spectra_.data(), count, feat_a_.data());
    model_->backbone(feat_a_.data(), count, lane_a_, rope_, lane_params_);
    model_->decode(feat_a_.data(), count, raw_a_.data());
  } else {
    model_->encode(spectra_.data(), count, feat_a_.data());
    std::memcpy(feat_b_.data(), feat_a_.data(), sizeof(float) * feat_size * count);
    model_->backbone(feat_a_.data(), count, lane_a_, rope_, lane_params_);
    model_->decode(feat_a_.data(), count, raw_a_.data());
    model_->backbone(feat_b_.data(), count, lane_b_, rope_, lane_params_);
    model_->decode(feat_b_.data(), count, raw_b_.data());
    for (int t = 0; t < count; ++t) {
      const double weight = std::clamp(
          static_cast<double>(within + t - crossfade_start + 1) /
              opt_.crossfade_frames,
          0.0, 1.0);
      const size_t offset = static_cast<size_t>(t) * frame_bins * 4;
      for (size_t i = 0; i < frame_bins * 4; ++i) {
        raw_a_[offset + i] = static_cast<float>(
            raw_a_[offset + i] * (1.0 - weight) + raw_b_[offset + i] * weight);
      }
    }
  }

  for (int t = 0; t < count; ++t) {
    std::vector<float> raw(raw_a_.begin() + static_cast<long>(t * frame_bins * 4),
                           raw_a_.begin() +
                               static_cast<long>((t + 1) * frame_bins * 4));
    raw_q_.push_back(std::move(raw));
    std::vector<std::complex<float>> spec(
        spectra_.begin() + static_cast<long>(t * frame_bins),
        spectra_.begin() + static_cast<long>((t + 1) * frame_bins));
    spec_q_.push_back(std::move(spec));
  }

  if (within + count == opt_.window_frames) {
    // The boundary: the trailing lane carries on, the leading one is dropped.
    std::swap(lane_a_, lane_b_);
    lane_b_ = Lane();
    lane_b_started_ = false;
  }
}

void Stream::synthesize() {
  const Config& cfg = model_->config();
  const int n_fft = cfg.n_fft;
  const int hop = cfg.hop;
  const int64_t index = emitted_;
  const int64_t last = frames_ - 1;
  const int64_t source = std::min(index + cfg.lookahead_frames, last);
  if (source < pending_base_ || index < pending_base_)
    throw std::runtime_error("synthesize: frame no longer pending");
  const std::vector<float>& raw = raw_q_[static_cast<size_t>(source - pending_base_)];
  const std::vector<std::complex<float>>& spec =
      spec_q_[static_cast<size_t>(index - pending_base_)];

  model_->correction(raw.data(), spec.data(), cfg.bins(), delta_spec_.data());
  model_->fft().irfft(delta_spec_.data(), frame_audio_.data());

  // Overlap-add of this frame's window, which spans audio samples
  // [index*hop - n_fft/2, +n_fft). The retained buffer may start inside that
  // span (samples below `released_` are already final), so the write is
  // clipped to what the buffer still holds.
  const int64_t frame_start = index * hop - n_fft / 2;
  const int64_t lo = std::max(frame_start, ola_base_);
  const int64_t hi = frame_start + n_fft;
  const int64_t need = hi - ola_base_;
  if (need > static_cast<int64_t>(ola_.size())) {
    ola_.resize(static_cast<size_t>(need), 0.0f);
    envelope_.resize(static_cast<size_t>(need), 0.0f);
  }
  const float* window = model_->window().data();
  for (int64_t sample = lo; sample < hi; ++sample) {
    const int i = static_cast<int>(sample - frame_start);
    const size_t at = static_cast<size_t>(sample - ola_base_);
    const float w = window[i];
    ola_[at] += frame_audio_[static_cast<size_t>(i)] * w;
    envelope_[at] += w * w;
  }
  ++emitted_;
}

std::vector<float> Stream::release(int64_t upto) {
  const int64_t total = audio_base_ + static_cast<int64_t>(audio_.size());
  const int64_t start = released_;
  const int64_t ola_end = ola_base_ + static_cast<int64_t>(envelope_.size());
  const int64_t stop = std::min(std::min(upto, total), ola_end);
  if (stop <= start) return {};
  std::vector<float> out(static_cast<size_t>(stop - start));
  for (int64_t s = start; s < stop; ++s) {
    const size_t index = static_cast<size_t>(s - ola_base_);
    const float env = std::max(envelope_[index], 1e-11f);
    out[static_cast<size_t>(s - start)] =
        audio_[static_cast<size_t>(s - audio_base_)] + ola_[index] / env;
  }
  released_ = stop;
  return out;
}

void Stream::trim() {
  const Config& cfg = model_->config();
  const int64_t drop = emitted_ - pending_base_;
  for (int64_t i = 0; i < drop; ++i) {
    if (!raw_q_.empty()) raw_q_.pop_front();
    if (!spec_q_.empty()) spec_q_.pop_front();
  }
  pending_base_ = emitted_;

  // Everything below `released_` is final, but the buffer may not reach that
  // far yet (a small push can release before anything is synthesized), so the
  // erase is clipped to what is held.
  const int64_t held = static_cast<int64_t>(ola_.size());
  const int64_t keep_from = std::min(std::max<int64_t>(released_ - ola_base_, 0),
                                     held);
  if (keep_from > 0) {
    ola_.erase(ola_.begin(), ola_.begin() + static_cast<long>(keep_from));
    envelope_.erase(envelope_.begin(),
                    envelope_.begin() + static_cast<long>(keep_from));
    ola_base_ += keep_from;
  }

  const int64_t needed_from =
      std::min(released_, std::max<int64_t>(0, frames_ * cfg.hop - cfg.n_fft / 2));
  const int64_t drop_audio =
      std::min(std::max<int64_t>(needed_from - audio_base_, 0),
               static_cast<int64_t>(audio_.size()));
  if (drop_audio > 0) {
    audio_.erase(audio_.begin(), audio_.begin() + static_cast<long>(drop_audio));
    audio_base_ = needed_from;
  }
}

void Stream::advance(int64_t limit) {
  while (frames_ < limit) {
    int64_t count = std::min<int64_t>(opt_.block_frames, limit - frames_);
    count = std::min<int64_t>(count, next_cut() - frames_);
    if (count < 1) break;
    run_frames(frames_, static_cast<int>(count));
    frames_ += count;
  }
  while (emitted_ + model_->config().lookahead_frames < frames_) synthesize();
  trim();
}

std::vector<float> Stream::process(const float* samples, size_t count) {
  if (closed_) throw std::runtime_error("the stream was flushed");
  audio_.insert(audio_.end(), samples, samples + count);
  const Config& cfg = model_->config();
  const int64_t total = audio_base_ + static_cast<int64_t>(audio_.size());
  const int64_t computable =
      floor_div(total - cfg.n_fft / 2, cfg.hop) + 1;
  advance(std::max<int64_t>(0, computable));
  std::vector<float> out = release(emitted_ * cfg.hop - cfg.n_fft / 2);
  trim();
  return out;
}

std::vector<float> Stream::flush() {
  if (closed_) return {};
  closed_ = true;
  const Config& cfg = model_->config();
  const int64_t total = audio_base_ + static_cast<int64_t>(audio_.size());
  advance(total / cfg.hop + 1);
  while (emitted_ < frames_) synthesize();
  std::vector<float> out = release(total);
  trim();
  return out;
}

}  // namespace vi
