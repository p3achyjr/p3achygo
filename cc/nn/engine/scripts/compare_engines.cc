#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/nn/engine/benchmark_engine.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/engine/go_dataset.h"
#include "cc/nn/engine/tf_engine.h"
#include "cc/nn/engine/trt_engine.h"

using namespace ::nn;

ABSL_FLAG(std::string, ds_path, "", "Path to DS for benchmarks.");
ABSL_FLAG(std::vector<std::string>, paths, {}, "Paths to models.");
ABSL_FLAG(std::vector<std::string>, kinds, {},
          "Kind of each path. 0=TF, 1=TF-TRT, 2=TRT");
ABSL_FLAG(int, batch_size, 48, "Batch Size.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  int batch_size = absl::GetFlag(FLAGS_batch_size);
  std::string ds_path = absl::GetFlag(FLAGS_ds_path);
  std::vector<std::string> model_paths = absl::GetFlag(FLAGS_paths);
  std::vector<std::string> kinds = absl::GetFlag(FLAGS_kinds);
  CHECK(model_paths.size() == kinds.size());

  std::vector<std::unique_ptr<Engine>> engines;
  for (int i = 0; i < model_paths.size(); ++i) {
    int kind = std::atoi(kinds[i].c_str());
    engines.emplace_back(CreateEngine(static_cast<Engine::Kind>(kind + 1),
                                      model_paths[i], batch_size));
  }

  std::unique_ptr<GoDataset> go_ds =
      std::make_unique<GoDataset>(batch_size, ds_path);
  for (std::unique_ptr<Engine>& engine : engines) {
    DefaultStats stats;
    std::string kind_str = [](Engine::Kind kind) {
      switch (kind) {
        case Engine::Kind::kTF:
          return "Tensorflow";
        case Engine::Kind::kTFTrt:
          return "TF-TRT";
        case Engine::Kind::kTrt:
          return "TensorRT";
        default:
          return "?";
      }
    }(engine->kind());

    LOG(INFO) << "Evaluating Engine (" << kind_str << "): " << engine->path();
    Benchmark(engine.get(), go_ds.get(), stats);
    LOG(INFO) << stats.ToString();
  }
}
