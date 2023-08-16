#ifndef NN_BACKEND_BUILD_TRT_ENGINE_H_
#define NN_BACKEND_BUILD_TRT_ENGINE_H_

#include <NvInfer.h>

#include "cc/nn/engine/model_arch.h"

namespace nn {
namespace trt {

/*
 * Builds and returns engine from a model architecture.
 */
nvinfer1::IHostMemory* BuildEngine(model_arch::Model* model_arch,
                                   size_t batch_size);

}  // namespace trt
}  // namespace nn

#endif
