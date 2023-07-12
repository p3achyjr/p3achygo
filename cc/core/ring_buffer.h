#ifndef __CORE_CIRCULAR_BUFFER_H_
#define __CORE_CIRCULAR_BUFFER_H_

#include <array>
#include <cstddef>
#include <memory>
#include <optional>

#include "cc/core/rand.h"

namespace core {

/*
 * A very simple ring buffer class.
 */
template <typename T, size_t N>
class RingBuffer final {
 public:
  RingBuffer() : start_(0), size_(0) {}
  ~RingBuffer() = default;

  void Append(T elem) {
    int idx = (start_ + size_) % N;
    buffer_[idx] = std::make_unique<T>(elem);
    if (size_ == N) {
      start_ = (start_ + 1) % N;
    } else {
      ++size_;
    }
  }

  std::optional<T> PopRandom() {
    if (size_ == 0 || !buffer_[start_]) {
      return std::nullopt;
    }

    int idx = (start_ + RandRange(prng_, 0, size_)) % N;
    std::swap(buffer_[start_], buffer_[idx]);

    T elem = *buffer_[start_];
    buffer_[start_].reset();
    start_ = (start_ + 1) % N;
    --size_;

    return elem;
  }

 private:
  std::array<std::unique_ptr<T>, N> buffer_;
  int start_;
  int size_;
  PRng prng_;
};

}  // namespace core

#endif
