#ifndef __NN_SHAPE_H_
#define __NN_SHAPE_H_

#include "tensorflow/core/framework/tensor_shape.h"

// Need this because of linker errors on absl::Span in optimized builds.
namespace nn {

::tensorflow::TensorShape CreateTensorShape(
    std::initializer_list<int64_t> dims);

}  // namespace nn

#endif
