#include "cc/nn/engine/engine_factory.h"

#include <filesystem>
#include <fstream>

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
  fs::path filepath(path);
  if (fs::is_regular_file(filepath)) {
    if (filepath.extension() == ".trt") {
      return Engine::Kind::kTrt;
    } else if (filepath.extension() == ".pb") {
      return Engine::Kind::kTFXla;
    }
    return Engine::Kind::kUnknown;
  }

  // Directory. Get Last Folder.
  auto last_folder = filepath.filename();
  if (last_folder == "_trt") {
    return Engine::Kind::kTFTrt;
  }

  return Engine::Kind::kTF;
}

int GetVersionFromModelPath(std::string path) {
  fs::path filepath(path);
  fs::path parent_dir =
      fs::is_regular_file(filepath) ? filepath.parent_path() : filepath;
  fs::path version_file = parent_dir / "VERSION";

  if (fs::exists(version_file) && fs::is_regular_file(version_file)) {
    std::ifstream ifs(version_file);
    int version;
    if (ifs >> version) {
      return version;
    }
    LOG(WARNING) << "Failed to parse VERSION file at " << version_file
                 << ", defaulting to version 1";
  }

  return 1;
}

std::unique_ptr<Engine> CreateEngine(Engine::Kind kind, std::string path,
                                     int batch_size, int version) {
  switch (kind) {
    case Engine::Kind::kTrt:
      return TrtEngine::Create(path, batch_size, version);
    case Engine::Kind::kTF:
      CHECK(version == 0) << "Unsupported";
      return TFEngine::Create(path, TFEngine::Kind::kTF, batch_size);
    case Engine::Kind::kTFTrt:
      CHECK(version == 0) << "Unsupported";
      return TFEngine::Create(path, TFEngine::Kind::kTRT, batch_size);
    case Engine::Kind::kTFXla:
      CHECK(version == 0) << "Unsupported";
      return TFEngine::Create(path, TFEngine::Kind::kXLA, batch_size);
    default:
      LOG(FATAL) << "Unknown Engine Kind.";
  }
}

}  // namespace nn
