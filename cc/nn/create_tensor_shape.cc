#include "cc/nn/create_tensor_shape.h"

namespace nn {

::tensorflow::TensorShape CreateTensorShape(
    std::initializer_list<int64_t> dims) {
  ::tensorflow::TensorShape shape;
  for (const auto& dim : dims) {
    shape.AddDim(dim);
  }

  return shape;
}

}  // namespace nn
