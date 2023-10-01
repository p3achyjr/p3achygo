#include "cc/nn/engine/engine_factory.h"

#include "absl/log/log.h"
#include "cc/nn/engine/tf_engine.h"
#include "cc/nn/engine/trt_engine.h"

namespace nn {

std::unique_ptr<Engine> CreateEngine(Engine::Kind kind, std::string path,
                                     int batch_size) {
  switch (kind) {
    case Engine::Kind::kTF:
      return TFEngine::Create(path, TFEngine::Kind::kTF, batch_size);
    case Engine::Kind::kTFTrt:
      return TFEngine::Create(path, TFEngine::Kind::kTRT, batch_size);
    case Engine::Kind::kTrt:
      return TrtEngine::Create(path, batch_size);
    default:
      LOG(FATAL) << "Unknown Engine Kind.";
  }
}

}  // namespace nn
