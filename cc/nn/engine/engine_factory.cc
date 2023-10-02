#include "cc/nn/engine/engine_factory.h"

#include <filesystem>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/nn/engine/tf_engine.h"
#include "cc/nn/engine/trt_engine.h"

namespace nn {
namespace {
namespace fs = std::filesystem;
}

Engine::Kind KindFromEnginePath(std::string path) {
  // Attempts to create engine from filepath.
  LOG(INFO) << "FILEPATH: " << path;
  fs::path filepath(path);
  if (fs::is_regular_file(filepath)) {
    LOG(INFO) << "extension: " << filepath.extension() << " correct ? "
              << (filepath.extension() == ".trt");
    if (filepath.extension() != ".trt") return Engine::Kind::kUnknown;
    return Engine::Kind::kTrt;
  }

  // Directory. Get Last Folder.
  auto last_folder = filepath.filename();
  LOG(INFO) << "last_folder: " << last_folder << " correct ? "
            << (last_folder == "_trt");
  if (last_folder == "_trt") {
    return Engine::Kind::kTFTrt;
  }

  return Engine::Kind::kTF;
}

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
