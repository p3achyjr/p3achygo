#ifndef __CORE_HEAP_H_
#define __CORE_HEAP_H_

#include <algorithm>
#include <vector>

namespace core {

/*
 * Thin convenience wrapper around STL heaps.
 */
template <typename T, typename Cmp>
class Heap {
 public:
  Heap(Cmp cmp) : cmp_(cmp) {}
  Heap(std::vector<T> data, Cmp cmp) : data_(data), cmp_(cmp) {
    std::make_heap(data_.begin(), data_.end(), cmp);
  }

  ~Heap() = default;

  T& PopHeap() {
    std::pop_heap(data_.begin(), data_.end(), cmp_);
    T back = data_.back();
    data_.pop_back();
    return back;
  }

  void PushHeap(const T& t) {
    data_.push_back(t);
    std::push_heap(data_.begin(), data_.end(), cmp_);
  }

 private:
  std::vector<T> data_;
  Cmp cmp_;
};

}  // namespace core

#endif  // __CORE_HEAP_H_