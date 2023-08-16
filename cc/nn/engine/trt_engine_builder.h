#ifndef NN_ENGINE_BUILD_TRT_ENGINE_H_
#define NN_ENGINE_BUILD_TRT_ENGINE_H_

#include <NvInfer.h>

#include "cc/nn/engine/model_arch.h"

namespace nn {
namespace trt {

/*
 * Builds and returns engine from a model architecture.
 */
nvinfer1::IHostMemory* BuildEngine(model_arch::Model* model_arch,
                                   size_t batch_size);

/*
 * Write serialized engine to disk.
 */
void WriteEngineToDisk(nvinfer1::IHostMemory* serialized_engine,
                       std::string path);

}  // namespace trt
}  // namespace nn

#endif
