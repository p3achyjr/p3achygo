#ifndef NN_ENGINE_ENGINE_FACTORY_H_
#define NN_ENGINE_ENGINE_FACTORY_H_

#include <memory>

#include "cc/nn/engine/engine.h"

namespace nn {

std::unique_ptr<Engine> CreateEngine(Engine::Kind kind, std::string path,
                                     int batch_size);

}  // namespace nn

#endif
