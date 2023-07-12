#ifndef __NN_SHAPE_H_
#define __NN_SHAPE_H_

#include "tensorflow/core/framework/tensor_shape.h"

// Need this because of linker errors on absl::Span in optimized builds.
namespace nn {

inline ::tensorflow::TensorShape CreateTensorShape(
    std::initializer_list<int64_t> dims) {
  ::tensorflow::TensorShape shape;
  for (const auto& dim : dims) {
    shape.AddDim(dim);
  }

  return shape;
}

}  // namespace nn

#endif
