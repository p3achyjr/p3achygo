#ifndef CORE_SPIN_MUTEX_H_
#define CORE_SPIN_MUTEX_H_

#include <atomic>
#include <mutex>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace core {

// A simple spin lock for very short critical sections (lc0's SpinMutex,
// src/utils/mutex.h): CAS loop with a CPU pause, yielding every 512 spins.
class SpinMutex final {
 public:
  class Lock final {
   public:
    explicit Lock(SpinMutex& m) : lock_(m) {}

   private:
    std::unique_lock<SpinMutex> lock_;
  };

  void lock() {
    int spins = 0;
    while (true) {
      int val = 0;
      if (mutex_.compare_exchange_weak(val, 1, std::memory_order_acq_rel)) {
        break;
      }
      ++spins;
      if (spins % 512 == 0) {
        std::this_thread::yield();
      } else {
        Pause();
      }
    }
  }
  void unlock() { mutex_.store(0, std::memory_order_release); }

 private:
  static inline void Pause() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    asm volatile("");
#endif
  }

  std::atomic<int> mutex_{0};
};

}  // namespace core

#endif
