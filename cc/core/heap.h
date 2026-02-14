#ifndef CORE_HEAP_H_
#define CORE_HEAP_H_

#include <algorithm>
#include <limits>
#include <vector>

namespace core {

/*
 * Thin convenience wrapper around STL heaps.
 */
template <typename T, typename Cmp>
class Heap {
 public:
  Heap(Cmp cmp, int max_size = std::numeric_limits<int>::max())
      : cmp_(cmp), max_size_(max_size) {}
  Heap(std::vector<T> data, Cmp cmp,
       int max_size = std::numeric_limits<int>::max())
      : data_(data), cmp_(cmp), max_size_(max_size) {
    std::make_heap(data_.begin(), data_.end(), cmp);
    Evict();
  }

  ~Heap() = default;

  size_t Size() { return data_.size(); }

  T PopHeap() {
    std::pop_heap(data_.begin(), data_.end(), cmp_);
    T back = data_.back();
    data_.pop_back();
    return back;
  }

  void PushHeap(const T& t) {
    data_.push_back(t);
    std::push_heap(data_.begin(), data_.end(), cmp_);
    Evict();
  }

 private:
  void Evict() {
    while (data_.size() > max_size_) {
      PopHeap();
    }
  }
  std::vector<T> data_;
  Cmp cmp_;
  int max_size_;
};

}  // namespace core

#endif  // CORE_HEAP_H_
