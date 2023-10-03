#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cfloat>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/game/board.h"
#include "cc/nn/engine/benchmark_engine.h"
#include "cc/nn/engine/buf_utils.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/go_dataset.h"
#include "cc/nn/engine/go_features.h"
#include "cc/nn/engine/trt_calibrator.h"
#include "cc/nn/engine/trt_engine.h"
#include "cc/nn/engine/trt_logger.h"
#include "cc/nn/engine/trt_names.h"

namespace {
namespace nv = ::nvinfer1;
namespace fs = ::std::filesystem;
using namespace ::nn;

struct BufferHandle {
  size_t size;
  nv::Dims dims;
  void* host_buf;
  void* device_buf;
};

void WriteEngineToDisk(nvinfer1::IHostMemory* serialized_engine,
                       std::string path) {
  std::ofstream file(path, std::ios::binary);

  if (!file) {
    LOG(ERROR) << "Cannot open engine file for writing: " << path << std::endl;
    return;
  }

  file.write(static_cast<char*>(serialized_engine->data()),
             serialized_engine->size());
  file.close();
}

template <size_t N>
int Argmax(const std::array<float, N>& arr) {
  size_t arg_max = 0;
  float max_val = -FLT_MAX;
  for (size_t i = 0; i < N; ++i) {
    if (arr[i] > max_val) {
      max_val = arr[i];
      arg_max = i;
    }
  }

  return static_cast<int>(arg_max);
}
}  // namespace

ABSL_FLAG(std::string, onnx_path, "", "Path to onnx model.");
ABSL_FLAG(std::string, ds_path, "", "Path to calibration ds.");
ABSL_FLAG(std::string, out_dir, "",
          "Directory to store engine. Defaults to same directory as weights.");
ABSL_FLAG(std::string, engine_path, "", "Path to existing engine.");
ABSL_FLAG(std::string, engine_name, "engine.trt", "Name of engine.");
ABSL_FLAG(bool, use_int8, false, "Whether to enable INT8.");
ABSL_FLAG(int, batch_size, 0, "Batch Size.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  int batch_size = absl::GetFlag(FLAGS_batch_size);
  std::string onnx_path = absl::GetFlag(FLAGS_onnx_path);
  std::string engine_path = absl::GetFlag(FLAGS_engine_path);
  std::string out_dir = absl::GetFlag(FLAGS_out_dir);
  std::string ds_path = absl::GetFlag(FLAGS_ds_path);
  if (batch_size == 0) {
    LOG(ERROR) << "Must Specify --batch_size.";
    return 1;
  }

  if (absl::GetFlag(FLAGS_use_int8) && ds_path == "") {
    LOG(ERROR) << "Must Specify Dataset Path.";
    return 1;
  }

  std::unique_ptr<nn::GoDataset> go_ds =
      std::make_unique<nn::GoDataset>(batch_size, ds_path);
  if (engine_path == "") {
    if (onnx_path == "") {
      LOG(ERROR) << "Must Specify ONNX Path.";
      return 1;
    }

    if (out_dir == "") {
      out_dir = fs::path(onnx_path).parent_path();
    }

    engine_path = fs::path(out_dir) / absl::GetFlag(FLAGS_engine_name);

    // 1. Create TensorRT builder, logger, and network
    nv::IBuilder* builder = nv::createInferBuilder(nn::trt::logger());
    nv::INetworkDefinition* network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(
            nv::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // 2. Create ONNX parser
    nvonnxparser::IParser* parser =
        nvonnxparser::createParser(*network, nn::trt::logger());

    // 3. Parse ONNX model
    if (!parser->parseFromFile(
            onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      LOG(ERROR) << "Could not parse ONNX model.";
      return 1;
    }

    for (int i = 0; i < network->getNbInputs(); ++i) {
      nv::ITensor* tensor = network->getInput(i);
      std::cout << "Tensor " << i << ": " << static_cast<int>(tensor->getType())
                << ", " << tensor->getName()
                << ", formats: " << tensor->getAllowedFormats() << std::endl;
    }

    for (int i = 0; i < 25; ++i) {
      nv::ILayer* layer = network->getLayer(i);
      std::cout << "Layer " << i << ": " << static_cast<int>(layer->getType())
                << ", " << layer->getName() << std::endl;
    }

    // Build Optimization Profile.
    nv::IOptimizationProfile* profile = builder->createOptimizationProfile();

    // Min
    profile->setDimensions("board_state", nv::OptProfileSelector::kMIN,
                           nv::Dims4(1, 19, 19, 13));
    profile->setDimensions("game_state", nv::OptProfileSelector::kMIN,
                           nv::Dims2(1, 7));

    // Opt
    profile->setDimensions("board_state", nv::OptProfileSelector::kOPT,
                           nv::Dims4(batch_size, 19, 19, 13));
    profile->setDimensions("game_state", nv::OptProfileSelector::kOPT,
                           nv::Dims2(batch_size, 7));

    // Max
    profile->setDimensions("board_state", nv::OptProfileSelector::kMAX,
                           nv::Dims4(batch_size, 19, 19, 13));
    profile->setDimensions("game_state", nv::OptProfileSelector::kMAX,
                           nv::Dims2(batch_size, 7));

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->addOptimizationProfile(profile);
    std::unique_ptr<nn::trt::Int8Calibrator> calibrator =
        nn::trt::Int8Calibrator::Create(batch_size, go_ds.get(),
                                        "/tmp/int8_cache.trt");
    config->setFlag(nv::BuilderFlag::kFP16);
    if (absl::GetFlag(FLAGS_use_int8) && builder->platformHasFastInt8()) {
      config->setFlag(nv::BuilderFlag::kINT8);
      config->setInt8Calibrator(calibrator.get());
      config->setCalibrationProfile(profile);
    }
    nv::IHostMemory* serialized_engine =
        builder->buildSerializedNetwork(*network, *config);

    WriteEngineToDisk(serialized_engine, engine_path);

    delete config;
    delete network;
    delete builder;
  }

  CHECK(engine_path != "");
  if (ds_path == "") return 0;

  // Read back from file.
  std::unique_ptr<TrtEngine> trt_engine =
      TrtEngine::Create(engine_path, batch_size);

  // Benchmark/verify inference.
  DefaultStats stats;
  Benchmark(trt_engine.get(), go_ds.get(), stats);
  LOG(INFO) << stats.ToString();

  return 0;
}
