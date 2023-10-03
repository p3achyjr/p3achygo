#ifndef NN_ENGINE_VAL_DS_H_
#define NN_ENGINE_VAL_DS_H_

#include <array>
#include <optional>

#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/nn/engine/go_features.h"

namespace nn {

/*
 * In-memory dataset of Go examples. Fixed batch.
 *
 * Will read entire batch into memory--not resilient.
 */
class GoDataset final {
 public:
  struct Row {
    GoFeatures features;
    GoLabels labels;
  };

  class Iterator {
   public:
    Iterator(std::vector<Row>* ptr) : ptr_(ptr) {}
    ~Iterator() = default;

    std::vector<Row>& operator*() { return *ptr_; }
    Iterator& operator++() {
      ++ptr_;
      return *this;
    }
    bool operator==(const Iterator& other) const { return ptr_ == other.ptr_; }
    bool operator!=(const Iterator& other) const { return ptr_ != other.ptr_; }

   private:
    std::vector<Row>* ptr_;
  };

  GoDataset(size_t batch_size, std::string ds_path);
  ~GoDataset() = default;

  size_t batch_size() const { return batch_size_; }
  size_t size() const { return batches_.size(); }
  Iterator begin() { return Iterator(batches_.data()); }
  Iterator end() { return Iterator(batches_.data() + size()); }

 private:
  std::vector<std::vector<Row>> batches_;
  int index_;
  size_t batch_size_;
};

}  // namespace nn

#endif
