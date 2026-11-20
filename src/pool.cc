#include "common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>
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

#if !defined(__aarch64__)
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__arm__)
  asm volatile("yield" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}
#endif

// Low-power userspace hardware wait on an atomic value change (ARM64 ldaxr+wfe).
// Wakes in ~1.5 us on a store to the monitored cache line while consuming near-zero
// CPU time and power (no thermal throttling, no kernel syscalls).
inline void wait_for_change_u64(const std::atomic<uint64_t>& a, uint64_t seen) {
#if defined(__aarch64__)
  uint64_t val;
  asm volatile("ldaxr %0, [%1]" : "=r"(val) : "r"(&a) : "memory");
  if (val == seen) {
    asm volatile("wfe" ::: "memory");
  } else {
    asm volatile("clrex" ::: "memory");
  }
#else
  (void)a;
  (void)seen;
  cpu_relax();
#endif
}

inline void wait_for_nonzero_i32(const std::atomic<int>& a) {
#if defined(__aarch64__)
  int val;
  asm volatile("ldaxr %w0, [%1]" : "=r"(val) : "r"(&a) : "memory");
  if (val != 0) {
    asm volatile("wfe" ::: "memory");
  } else {
    asm volatile("clrex" ::: "memory");
  }
#else
  (void)a;
  cpu_relax();
#endif
}

// Apple Silicon L2 cache lines are 128 bytes; padding hot atomics prevents
// cross-cluster cache-line ping-pong during barriers.
constexpr size_t kCacheLine = 128;

#if defined(__aarch64__)
// Each wfe sleeps until an event (~10-100 us max), so 500 wfe iterations easily
// covers all inter-op gaps without falling into mutex/condvar syscalls.
constexpr int kSpinRounds = 500;
#else
constexpr int kSpinRounds = 20000;
#endif

}  // namespace

#if defined(__APPLE__)
// Hint XNU to place threads with different tags on separate L2 cache domains
// (= the two P-core / SME2 clusters on M4 Max), avoiding intra-cluster SME2
// arbiter collisions during 2-chunk GEMM dispatch.
inline void set_cluster_affinity(int tag) {
  thread_affinity_policy_data_t policy = {tag};
  thread_policy_set(pthread_mach_thread_np(pthread_self()),
                    THREAD_AFFINITY_POLICY,
                    reinterpret_cast<thread_policy_t>(&policy),
                    THREAD_AFFINITY_POLICY_COUNT);
}
#endif

// ------------------------------------------------------------------ pool ----

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
// Pick one logical CPU per physical core from the process's allowed set. On
// SMT x86 machines letting the scheduler place all workers freely can stack two
// of them on one core while another core sits idle; pinning one thread per
// core measured ~3% faster on an 8-core/16-thread Ryzen with eight workers.
// Set VI_NO_AFFINITY=1 to disable.
std::vector<int> one_cpu_per_core(int max) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return {};
  std::vector<int> reps, cores;
  for (int c = 0; c < CPU_SETSIZE && static_cast<int>(reps.size()) < max; ++c) {
    if (!CPU_ISSET(c, &allowed)) continue;
    int core = -1;
    char path[128], text[64];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
    if (FILE* f = std::fopen(path, "r")) {
      if (std::fgets(text, sizeof(text), f)) core = std::atoi(text);
      std::fclose(f);
    }
    if (core >= 0 &&
        std::find(cores.begin(), cores.end(), core) != cores.end())
      continue;
    if (core >= 0) cores.push_back(core);
    reps.push_back(c);
  }
  return reps;
}
#endif

struct Pool::Impl {
  std::mutex mutex;
  std::condition_variable wake;
  std::vector<std::thread> threads;

  // Read-mostly job dispatch fields, isolated on their own cache line.
  alignas(kCacheLine) const std::function<void(int)>* job = nullptr;
  int count = 0;
  bool static_dispatch = false;
  bool stride_dispatch = false;
  int stride_peer = 1;
  std::atomic<uint64_t> generation{0};
  std::atomic<bool> stop{false};

  // Dynamic work-stealing counter, isolated on its own cache line.
  alignas(kCacheLine) std::atomic<int> next{0};

  // Barrier completion counter, isolated on its own cache line.
  alignas(kCacheLine) std::atomic<int> pending{0};

  // Sleep tracking counter, isolated on its own cache line.
  alignas(kCacheLine) std::atomic<int> sleepers{0};

  void worker(int worker_id, int total_threads) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    set_cluster_affinity(worker_id < (total_threads + 1) / 2 ? 1 : 2);
#else
    (void)total_threads;
#endif
    uint64_t seen = 0;
    for (;;) {
      int spins = 0;
      while (generation.load(std::memory_order_acquire) == seen) {
        if (stop.load(std::memory_order_relaxed)) return;
        if (spins < kSpinRounds) {
          wait_for_change_u64(generation, seen);
          ++spins;
        } else {
          sleepers.fetch_add(1, std::memory_order_seq_cst);
          {
            std::unique_lock<std::mutex> lock(mutex);
            wake.wait_for(lock, std::chrono::microseconds(500), [&] {
              return stop.load(std::memory_order_relaxed) ||
                     generation.load(std::memory_order_acquire) != seen;
            });
          }
          sleepers.fetch_sub(1, std::memory_order_seq_cst);
          // On x86 stay parked across timeout checks. Restarting the busy
          // spin here consumed cores even with no audio arriving. Each new
          // job starts its own spin budget at the top of the worker loop.
#if !defined(__x86_64__) && !defined(__i386__)
          spins = 0;
#endif
        }
      }
      if (stop.load(std::memory_order_relaxed)) return;
      seen = generation.load(std::memory_order_acquire);
      const std::function<void(int)>* fn = job;
      const int n = count;
      if (static_dispatch) {
        if (stride_dispatch) {
          // Spread small chunk counts (e.g. 2 SME workers) across worker IDs
          // so they land on different P-core clusters on multi-cluster Apple Silicon.
          if (worker_id == stride_peer) (*fn)(1);
        } else if (worker_id < n) {
          (*fn)(worker_id);
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
      } else {
        for (;;) {
          const int index = next.fetch_add(1, std::memory_order_relaxed);
          if (index >= n) break;
          (*fn)(index);
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
    }
  }
};

Pool::Pool(int threads) : impl_(new Impl), threads_(threads < 1 ? 1 : threads) {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (threads_ > 1) set_cluster_affinity(1);
#endif
  impl_->threads.reserve(static_cast<size_t>(threads_ - 1));
  for (int i = 1; i < threads_; ++i)
    impl_->threads.emplace_back([this, i] { impl_->worker(i, threads_); });
#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
  const char* no_affinity = std::getenv("VI_NO_AFFINITY");
  if (threads_ > 1 && !(no_affinity && std::atoi(no_affinity))) {
    // Process-wide mask before any Pool pinned it. The default pool is built
    // lazily (8 threads, pinned) and `set_default_pool_threads(N)` then
    // replaces it: the replacement's `one_cpu_per_core` would otherwise see
    // only the pinned main-thread mask, fail its size check, skip pinning,
    // and leave every new worker inheriting the single-CPU mask (~serial).
    static const std::vector<int> original = [] {
      cpu_set_t allowed;
      CPU_ZERO(&allowed);
      std::vector<int> out;
      if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return out;
      for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &allowed)) out.push_back(c);
      return out;
    }();
    const std::vector<int> cpus = one_cpu_per_core(threads_);
    cpu_set_t set;
    CPU_ZERO(&set);
    if (static_cast<int>(cpus.size()) == threads_) {
      CPU_SET(cpus[0], &set);
      sched_setaffinity(0, sizeof(set), &set);  // calling (main) thread
      for (int i = 1; i < threads_; ++i) {
        CPU_ZERO(&set);
        CPU_SET(cpus[i], &set);
        pthread_setaffinity_np(impl_->threads[static_cast<size_t>(i - 1)].native_handle(),
                               sizeof(set), &set);
      }
    } else if (!original.empty()) {
      // Pinning not possible (e.g. replacement pool probing a pinned mask):
      // restore the full mask so workers never inherit a single-CPU mask.
      for (int c : original) CPU_SET(c, &set);
      sched_setaffinity(0, sizeof(set), &set);
      for (auto& t : impl_->threads)
        pthread_setaffinity_np(t.native_handle(), sizeof(set), &set);
    }
  }
#endif
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
  const bool use_static = (n <= threads_);
  impl_->job = &fn;
  impl_->count = n;
  impl_->static_dispatch = use_static;
  impl_->stride_dispatch = (use_static && n == 2 && threads_ > 2);
  impl_->stride_peer = threads_ - 1;
  if (!use_static) impl_->next.store(0, std::memory_order_relaxed);
  impl_->pending.store(threads_ - 1, std::memory_order_relaxed);
  impl_->generation.fetch_add(1, std::memory_order_release);
  if (impl_->sleepers.load(std::memory_order_seq_cst) > 0) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->wake.notify_all();
  }

  if (use_static) {
    fn(0);
  } else {
    for (;;) {
      const int index = impl_->next.fetch_add(1, std::memory_order_relaxed);
      if (index >= n) break;
      fn(index);
    }
  }

  while (impl_->pending.load(std::memory_order_acquire) != 0) {
    wait_for_nonzero_i32(impl_->pending);
  }
}

namespace {

int preferred_threads() {
  const int cores = static_cast<int>(std::thread::hardware_concurrency());
  if (cores <= 0) return 1;
#if defined(__APPLE__)
  // Measured on M4 Max (10 P-cores + 4 E-cores): 8 threads spans both P-core
  // clusters (so both SME2 units stay active) and beats 10 threads on both wall
  // clock and user CPU time.
  return std::min(cores, 8);
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
