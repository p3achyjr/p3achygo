#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/nn/engine/parse_h5.h"
#include "cc/nn/engine/trt_calibrator.h"
#include "cc/nn/engine/trt_engine_builder.h"
#include "cc/nn/engine/trt_logger.h"
#include "cc/nn/engine/trt_names.h"
#include "cc/nn/engine/validate_model_arch.h"

namespace {
namespace nv = ::nvinfer1;
namespace fs = ::std::filesystem;
using namespace ::nn;

static constexpr char kEngineFilename[] = "engine.trt";

struct BufferHandle {
  size_t size;
  nv::Dims dims;
  void* host_buf;
  void* device_buf;
};
}  // namespace

ABSL_FLAG(std::string, onnx_path, "", "Path to onnx model.");
ABSL_FLAG(std::string, weights_path, "", "Path to model weights.");
ABSL_FLAG(std::string, out_dir, "",
          "Directory to store engine. Defaults to same directory as weights.");
ABSL_FLAG(std::string, engine_path, "", "Path to existing engine.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  int batch_size = 48;
  std::string engine_path = absl::GetFlag(FLAGS_engine_path);
  std::string weights_path = absl::GetFlag(FLAGS_weights_path);
  std::string onnx_path = absl::GetFlag(FLAGS_onnx_path);
  if (engine_path == "") {
    if (absl::GetFlag(FLAGS_out_dir) == "") {
      fs::path path_to_weights(weights_path);
      engine_path = path_to_weights.parent_path() / kEngineFilename;
    } else {
      engine_path = fs::path(absl::GetFlag(FLAGS_out_dir)) / kEngineFilename;
    }

    if (weights_path != "") {
      std::unique_ptr<model_arch::Model> model = ParseFromH5(weights_path);
      model_arch::ValidateModelArch(model.get());
      nv::IHostMemory* serialized_engine =
          nn::trt::BuildEngine(model.get(), batch_size);

      // Write to disk.
      nn::trt::WriteEngineToDisk(serialized_engine, engine_path);
      delete serialized_engine;
    } else if (onnx_path != "") {
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
        std::cout << "Tensor " << i << ": "
                  << static_cast<int>(tensor->getType()) << ", "
                  << tensor->getName()
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
      profile->setDimensions("args_0", nv::OptProfileSelector::kMIN,
                             nv::Dims4(1, 19, 19, 13));
      profile->setDimensions("args_1", nv::OptProfileSelector::kMIN,
                             nv::Dims2(1, 7));

      // Opt
      profile->setDimensions("args_0", nv::OptProfileSelector::kOPT,
                             nv::Dims4(batch_size, 19, 19, 13));
      profile->setDimensions("args_1", nv::OptProfileSelector::kOPT,
                             nv::Dims2(batch_size, 7));

      // Max
      profile->setDimensions("args_0", nv::OptProfileSelector::kMAX,
                             nv::Dims4(batch_size, 19, 19, 13));
      profile->setDimensions("args_1", nv::OptProfileSelector::kMAX,
                             nv::Dims2(batch_size, 7));

      nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
      config->addOptimizationProfile(profile);
      std::unique_ptr<nn::trt::Int8Calibrator> calibrator =
          nn::trt::Int8Calibrator::Create(batch_size,
                                          "/tmp/p3achygo/val.tfrecord.zz",
                                          "/tmp/int8_cache.trt");
      config->setFlag(nv::BuilderFlag::kFP16);
      if (builder->platformHasFastInt8()) {
        config->setFlag(nv::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
        config->setCalibrationProfile(profile);
      }
      nv::IHostMemory* serialized_engine =
          builder->buildSerializedNetwork(*network, *config);

      nn::trt::WriteEngineToDisk(serialized_engine, engine_path);

      delete config;
      delete network;
      delete builder;
    }
  }

  CHECK(engine_path != "");

  // Read back from file.
  std::string engine_data;
  FILE* const fp = fopen(engine_path.c_str(), "r");
  while (!feof(fp)) {
    char buf[4096];
    size_t num_read = fread(buf, 1, 4096, fp);
    engine_data.append(std::string(buf, num_read));
  }
  fclose(fp);

  // Decode engine.
  std::unique_ptr<nv::IRuntime> runtime(
      nv::createInferRuntime(nn::trt::logger()));
  std::unique_ptr<nv::ICudaEngine> engine(runtime->deserializeCudaEngine(
      engine_data.c_str(), engine_data.length()));
  std::unique_ptr<nv::IExecutionContext> exec_context(
      engine->createExecutionContext());

  // Create stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Allocate buffers.
  std::unordered_map<std::string, BufferHandle> buf_map;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    nv::Dims dims = engine->getTensorShape(name);

    CHECK(dims.nbDims > 0);
    CHECK(dims.d[0] == -1);
    dims.d[0] = batch_size;
    size_t num_elems = std::accumulate(&dims.d[1], &dims.d[dims.nbDims],
                                       batch_size, std::multiplies<size_t>());
    size_t num_bytes = num_elems * sizeof(float);
    LOG(INFO) << "Requesting " << num_bytes << " bytes for buffer: " << name;

    BufferHandle buf_handle;
    buf_handle.size = num_bytes;
    buf_handle.dims = dims;
    cudaMallocHost(&buf_handle.host_buf, num_bytes);
    cudaMalloc(&buf_handle.device_buf, num_bytes);
    exec_context->setTensorAddress(name, buf_handle.device_buf);

    buf_map[name] = buf_handle;
  }

  // Configure inputs for execution context.
  BufferHandle& input_planes = buf_map[nn::trt::input::kPlanesName];
  BufferHandle& input_features = buf_map[nn::trt::input::kFeaturesName];
  exec_context->setInputShape(nn::trt::input::kPlanesName, input_planes.dims);
  exec_context->setInputShape(nn::trt::input::kFeaturesName,
                              input_features.dims);

  // Outputs.
  BufferHandle& output_policy = buf_map[nn::trt::output::kPolicyCombinedName];
  BufferHandle& output_zq = buf_map[nn::trt::output::kZqName];
  BufferHandle& output_own = buf_map[nn::trt::output::kOwnershipName];
  BufferHandle& output_score = buf_map[nn::trt::output::kScoreName];

  exec_context->setOptimizationProfileAsync(0, cudaStreamPerThread);
  cudaStreamSynchronize(stream);

  // Benchmark inference.
  float count = 0;
  float avg_us = 0;
  for (int _ = 0; _ < 10000; ++_) {
    auto start = std::chrono::steady_clock::now();
    cudaMemcpyAsync(input_planes.device_buf, input_planes.host_buf,
                    input_planes.size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(input_features.device_buf, input_features.host_buf,
                    input_features.size, cudaMemcpyHostToDevice, stream);
    CHECK(exec_context->enqueueV3(stream));
    // cudaMemcpyAsync(output_policy.host_buf, output_policy.device_buf,
    //                 output_policy.size, cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(output_zq.host_buf, output_zq.device_buf, output_zq.size,
    //                 cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(output_own.host_buf, output_own.device_buf,
    // output_own.size,
    //                 cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(output_score.host_buf, output_score.device_buf,
    //                 output_score.size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    LOG_EVERY_N_SEC(INFO, 2) << "Inference took " << elapsed_us << "us.";
    count++;
    avg_us = avg_us * ((count - 1) / count) + (1 / count) * elapsed_us;
  }
  LOG(INFO) << "Average Time: " << avg_us << "us.";

  // Free buffers.
  for (auto& [name, buf_handle] : buf_map) {
    cudaFreeHost(buf_handle.host_buf);
    cudaFree(buf_handle.device_buf);
  }

  return 0;
}
