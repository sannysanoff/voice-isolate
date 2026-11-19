#include "common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include "vecmath.h"

namespace vi {

void layer_norm(const float* x, float* y, int rows, int cols, const float* w,
                const float* b, float eps) {
  const float inv_cols = 1.0f / static_cast<float>(cols);
  const int lanes = vx::lanes;
  const int vend = cols - (cols % lanes);
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<size_t>(r) * cols;
    float* yr = y + static_cast<size_t>(r) * cols;
    const float mean = v_sum(xr, cols) * inv_cols;
    const vx::batch vmean(mean);
    vx::batch acc(0.0f);
    int i = 0;
    for (; i < vend; i += lanes) {
      const vx::batch d = loadf(xr + i) - vmean;
      acc += d * d;
    }
    float square = xsimd::reduce_add(acc);
    for (; i < cols; ++i) {
      const float d = xr[i] - mean;
      square += d * d;
    }
    const float inv = 1.0f / std::sqrt(square * inv_cols + eps);
    const vx::batch vinv(inv);
    if (w != nullptr) {
      i = 0;
      for (; i < vend; i += lanes)
        storef(yr + i, (loadf(xr + i) - vmean) * vinv * loadf(w + i) +
                           loadf(b + i));
      for (; i < cols; ++i) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
    } else {
      i = 0;
      for (; i < vend; i += lanes)
        storef(yr + i, (loadf(xr + i) - vmean) * vinv);
      for (; i < cols; ++i) yr[i] = (xr[i] - mean) * inv;
    }
  }
}

namespace {

// A short pause during a spin; keeps the core free for the working threads.
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  asm volatile("yield" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

constexpr int kSpinRounds = 20000;

}  // namespace

// ------------------------------------------------------------------ pool ----

struct Pool::Impl {
  std::mutex mutex;
  std::condition_variable wake;
  std::vector<std::thread> threads;
  const std::function<void(int)>* job = nullptr;
  int count = 0;
  std::atomic<int> next{0};
  std::atomic<int> pending{0};
  std::atomic<uint64_t> generation{0};
  std::atomic<bool> stop{false};
  std::atomic<int> sleepers{0};

  // One worker's body: spin for the next job, then take indices until done.
  void worker() {
#if defined(__APPLE__)
    // Without an elevated QoS class macOS parks these threads on the efficiency
    // cores, and one slow core at each barrier is what caps the scaling.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    uint64_t seen = 0;
    for (;;) {
      int spins = 0;
      while (generation.load(std::memory_order_acquire) == seen) {
        if (stop.load(std::memory_order_relaxed)) return;
        // Spin, then block. A worker that sleeps through a barrier stalls every
        // thread behind it, so the spin window has to cover the barrier gap; the
        // condvar is the fallback for when the pool really is idle. Yielding in
        // between is a syscall and made this much slower, not faster.
        if (spins < kSpinRounds) {
          cpu_relax();
        } else {
          sleepers.fetch_add(1, std::memory_order_seq_cst);
          {
            std::unique_lock<std::mutex> lock(mutex);
            wake.wait_for(lock, std::chrono::microseconds(200), [&] {
              return stop.load() || generation.load() != seen;
            });
          }
          sleepers.fetch_sub(1, std::memory_order_seq_cst);
          spins = 0;
          continue;
        }
        ++spins;
      }
      if (stop.load(std::memory_order_relaxed)) return;
      seen = generation.load(std::memory_order_acquire);
      const std::function<void(int)>* fn = job;
      const int n = count;
      for (;;) {
        const int index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= n) break;
        (*fn)(index);
      }
      pending.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
};

Pool::Pool(int threads) : impl_(new Impl), threads_(threads < 1 ? 1 : threads) {
  impl_->threads.reserve(static_cast<size_t>(threads_ - 1));
  for (int i = 1; i < threads_; ++i)
    impl_->threads.emplace_back([this] { impl_->worker(); });
}

Pool::~Pool() {
  impl_->stop.store(true);
  impl_->generation.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
  }
  impl_->wake.notify_all();
  for (auto& t : impl_->threads) t.join();
  delete impl_;
}

void Pool::parallel_for(int n, const std::function<void(int)>& fn) {
  if (n <= 0) return;
  if (threads_ == 1 || n == 1) {
    for (int i = 0; i < n; ++i) fn(i);
    return;
  }
  impl_->job = &fn;
  impl_->count = n;
  impl_->next.store(0, std::memory_order_relaxed);
  impl_->pending.store(threads_ - 1, std::memory_order_relaxed);
  impl_->generation.fetch_add(1, std::memory_order_release);
  // Only syscall when a worker is asleep: during a stream they spin instead.
  if (impl_->sleepers.load(std::memory_order_seq_cst) > 0)
    impl_->wake.notify_all();

  for (;;) {
    const int index = impl_->next.fetch_add(1, std::memory_order_relaxed);
    if (index >= n) break;
    fn(index);
  }
  // Spin for the workers, then block: the pool is warm during a stream.
  int spins = 0;
  while (impl_->pending.load(std::memory_order_acquire) != 0) {
    if (++spins > kSpinRounds * 16) {
      std::unique_lock<std::mutex> lock(impl_->mutex);
      impl_->wake.wait_for(lock, std::chrono::milliseconds(1));
      spins = 0;
    } else {
      cpu_relax();
    }
  }
}

namespace {

// The threads this workload wants. Measured on an M4 Max (10 performance + 4
// efficiency cores), 6 beats every other count for both the fp32 and the int8
// path — 6.8 s against 7.4 s at 8 and 15.8 s at 10 on a 5 s clip. The cores are
// grouped in clusters and a barrier makes every thread wait for the slowest, so
// spilling past one cluster costs more than the extra cores return.
int preferred_threads() {
  const int cores = static_cast<int>(std::thread::hardware_concurrency());
  if (cores <= 0) return 1;
#if defined(__APPLE__)
  return std::min(cores, 6);
#else
  return std::min(cores, 8);
#endif
}

std::unique_ptr<Pool>& pool_slot() {
  static std::unique_ptr<Pool> slot(new Pool(preferred_threads()));
  return slot;
}
}  // namespace

Pool& default_pool() { return *pool_slot(); }

void set_default_pool_threads(int threads) {
  if (threads == pool_slot()->size()) return;
  pool_slot().reset(new Pool(threads));
}

}  // namespace vi
