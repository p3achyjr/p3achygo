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

#include "NvInferImpl.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/nn/engine/benchmark_engine.h"
#include "cc/nn/engine/buf_utils.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/engine_factory.h"
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

constexpr char kTimingCachePath[] = "/tmp/trt_timing_cache.bin";

struct BufferHandle {
  size_t size;
  nv::Dims dims;
  void* host_buf;
  void* device_buf;
};

static std::vector<char> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  f.seekg(0, std::ios::end);
  const std::streamsize size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  f.read(data.data(), size);
  return data;
}

static void WriteFile(const std::string& path, const void* data, size_t size) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data),
          static_cast<std::streamsize>(size));
}

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
ABSL_FLAG(std::string, engine_name, "",
          "Name of engine. Defaults to <onnx_stem>.trt.");
ABSL_FLAG(bool, use_int8, false, "Whether to enable INT8.");
ABSL_FLAG(int, batch_size, 0, "Batch Size. ");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  int batch_size = absl::GetFlag(FLAGS_batch_size);
  if (batch_size == 0) {
    LOG(ERROR) << "Must Specify --batch_size.";
    return 1;
  }

  std::string onnx_path = absl::GetFlag(FLAGS_onnx_path);
  std::string engine_path = absl::GetFlag(FLAGS_engine_path);
  std::string out_dir = absl::GetFlag(FLAGS_out_dir);
  std::string ds_path = absl::GetFlag(FLAGS_ds_path);
  if (absl::GetFlag(FLAGS_use_int8) && ds_path == "") {
    LOG(ERROR) << "Must Specify Dataset Path.";
    return 1;
  }

  // Extract version from model path
  std::string model_path = onnx_path.empty() ? engine_path : onnx_path;
  int version = nn::GetVersionFromModelPath(model_path);
  LOG(INFO) << "Detected model version: " << version;

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

    std::string engine_name = absl::GetFlag(FLAGS_engine_name);
    if (engine_name.empty()) {
      engine_name = fs::path(onnx_path).stem().string() + ".trt";
    }
    engine_path = fs::path(out_dir) / engine_name;

    // 1. Create TensorRT builder, logger, and network
    nv::IBuilder* builder = nv::createInferBuilder(nn::trt::logger());
    nv::INetworkDefinition* network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(
            nv::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));

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

    // Set static input dimensions
    const int num_planes = version >= 1 ? constants::kNumInputFeaturePlanesV1
                                        : constants::kNumInputFeaturePlanesV0;
    const int num_features = version >= 1
                                 ? constants::kNumInputFeatureScalarsV1
                                 : constants::kNumInputFeatureScalarsV0;
    for (int i = 0; i < network->getNbInputs(); ++i) {
      nv::ITensor* input = network->getInput(i);
      std::string name = input->getName();
      LOG(INFO) << "Input " << i << ": " << static_cast<int>(input->getType())
                << ", " << name << ", formats: " << input->getAllowedFormats();

      if (name == "board_state") {
        input->setDimensions(nv::Dims4(batch_size, 19, 19, num_planes));
      } else if (name == "game_state") {
        input->setDimensions(nv::Dims2(batch_size, num_features));
      }
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    const std::vector<char> timing_cache_blob = ReadFile(kTimingCachePath);
    std::unique_ptr<nvinfer1::ITimingCache> timingCache(
        config->createTimingCache(
            timing_cache_blob.empty() ? nullptr : timing_cache_blob.data(),
            timing_cache_blob.size()));
    config->setMemoryPoolLimit(nv::MemoryPoolType::kWORKSPACE,
                               static_cast<size_t>(20) << 32);
    const bool ok =
        config->setTimingCache(*timingCache, /*ignoreMismatch=*/false);
    LOG(INFO) << "Timing cache set: " << ok
              << ", initial size: " << timing_cache_blob.size();
    nv::IHostMemory* serialized_engine =
        builder->buildSerializedNetwork(*network, *config);

    WriteEngineToDisk(serialized_engine, engine_path);
    if (auto const* new_timing_cache = config->getTimingCache()) {
      std::unique_ptr<nvinfer1::IHostMemory> new_timing_cache_blob(
          new_timing_cache->serialize());
      WriteFile(kTimingCachePath, new_timing_cache_blob->data(),
                new_timing_cache_blob->size());
    }

    delete config;
    delete network;
    delete builder;
  }

  CHECK(engine_path != "");

  // Read back from file.
  std::unique_ptr<TrtEngine> trt_engine =
      TrtEngine::Create(engine_path, batch_size, version);

  // Benchmark/verify inference.
  DefaultStats stats;
  Benchmark(trt_engine.get(), go_ds.get(), stats);
  LOG(INFO) << stats.ToString();

  return 0;
}
