#ifndef CORE_RELAXED_ATOMIC_H_
#define CORE_RELAXED_ATOMIC_H_

#include <atomic>

namespace core {

/*
 * A value whose loads and stores are relaxed atomics: concurrent readers see
 * some written value (never a torn one) without synchronizing with the writer.
 * On x86-64 these compile to plain loads and stores.
 *
 * Read-modify-write operators (+=, -=, *=, /=, ++, --) are a relaxed load
 * followed by a relaxed store, NOT an atomic RMW: callers must serialize
 * writers (e.g. by holding the owning node's mutex), exactly as with a plain
 * variable.
 */
template <typename T>
class RelaxedAtomic final {
 public:
  constexpr RelaxedAtomic() noexcept : v_(T{}) {}
  // Explicit so `cond ? atomic : value` converts the atomic to T.
  constexpr explicit RelaxedAtomic(T x) noexcept : v_(x) {}
  RelaxedAtomic(const RelaxedAtomic& other) noexcept : v_(other.load()) {}
  RelaxedAtomic& operator=(const RelaxedAtomic& other) noexcept {
    store(other.load());
    return *this;
  }
  RelaxedAtomic& operator=(T x) noexcept {
    store(x);
    return *this;
  }

  operator T() const noexcept { return load(); }
  T load() const noexcept { return v_.load(std::memory_order_relaxed); }
  void store(T x) noexcept { v_.store(x, std::memory_order_relaxed); }

  RelaxedAtomic& operator+=(T d) noexcept {
    store(load() + d);
    return *this;
  }
  RelaxedAtomic& operator-=(T d) noexcept {
    store(load() - d);
    return *this;
  }
  RelaxedAtomic& operator*=(T d) noexcept {
    store(load() * d);
    return *this;
  }
  RelaxedAtomic& operator/=(T d) noexcept {
    store(load() / d);
    return *this;
  }
  RelaxedAtomic& operator++() noexcept { return *this += T(1); }
  RelaxedAtomic& operator--() noexcept { return *this -= T(1); }
  T operator++(int) noexcept {
    const T old = load();
    store(old + T(1));
    return old;
  }
  T operator--(int) noexcept {
    const T old = load();
    store(old - T(1));
    return old;
  }

 private:
  std::atomic<T> v_;
};

}  // namespace core

#endif
