#ifndef NN_ENGINE_ENGINE_FACTORY_H_
#define NN_ENGINE_ENGINE_FACTORY_H_

#include <memory>

#include "cc/nn/engine/engine.h"

namespace nn {

Engine::Kind KindFromEnginePath(std::string path);
int GetVersionFromModelPath(std::string path);
std::unique_ptr<Engine> CreateEngine(Engine::Kind kind, std::string path,
                                     int batch_size, int version);

}  // namespace nn

#endif
