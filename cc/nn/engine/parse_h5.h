#ifndef NN_ENGINE_PARSE_H5_H_
#define NN_ENGINE_PARSE_H5_H_

#include <memory>

#include "cc/nn/engine/model_arch.h"

namespace nn {

std::unique_ptr<model_arch::Model> ParseFromH5(std::string path);

}

#endif
