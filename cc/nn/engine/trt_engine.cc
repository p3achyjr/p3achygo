#include "cc/nn/engine/trt_engine.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <numeric>
#include <unordered_map>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/nn/engine/buf_utils.h"
#include "cc/nn/engine/trt_logger.h"
#include "cc/nn/engine/trt_names.h"

namespace nn {
namespace {
namespace nv = ::nvinfer1;
using namespace ::nn;

class TrtEngineImpl : public TrtEngine {
 public:
  TrtEngineImpl(std::string path, int batch_size);
  ~TrtEngineImpl();

  Engine::Kind kind() override { return Engine::Kind::kTrt; }
  std::string path() override { return path_; }
  void LoadBatch(int batch_id, const GoFeatures& features) override;
  void RunInference() override;
  void GetBatch(int batch_id, NNInferResult& result) override;
  void GetOwnership(int batch_id,
                    std::array<float, constants::kNumBoardLocs>& own) override;

 private:
  struct BufferHandle {
    size_t size;
    nv::Dims dims;
    void* host_buf;
    void* device_buf;
  };

  std::unordered_map<std::string, BufferHandle> buf_map_;
  std::unique_ptr<nv::IRuntime> runtime_;
  std::unique_ptr<nv::ICudaEngine> engine_;
  std::unique_ptr<nv::IExecutionContext> exec_context_;
  cudaStream_t stream_;

  std::vector<float> scores_;
  const int batch_size_;
  const std::string path_;

  const std::array<int, 4> planes_shape_;
  const std::array<int, 2> feats_shape_;
  const std::array<int, 2> pi_shape_;
  const std::array<int, 2> outcome_shape_;
  const std::array<int, 2> score_shape_;
};

TrtEngineImpl::TrtEngineImpl(std::string path, int batch_size)
    : batch_size_(batch_size),
      path_(path),
      planes_shape_{batch_size_, BOARD_LEN, BOARD_LEN,
                    constants::kNumInputFeaturePlanes},
      feats_shape_{batch_size_, constants::kNumInputFeatureScalars},
      pi_shape_{batch_size_, constants::kMaxMovesPerPosition},
      outcome_shape_{batch_size_, constants::kNumValueLogits},
      score_shape_{batch_size_, constants::kNumScoreLogits} {
  // Read back from file.
  std::string engine_data;
  FILE* const fp = fopen(path.c_str(), "r");
  while (!feof(fp)) {
    char buf[4096];
    size_t num_read = fread(buf, 1, 4096, fp);
    engine_data.append(std::string(buf, num_read));
  }
  fclose(fp);

  runtime_ =
      std::unique_ptr<nv::IRuntime>(nv::createInferRuntime(nn::trt::logger()));
  engine_ = std::unique_ptr<nv::ICudaEngine>(runtime_->deserializeCudaEngine(
      engine_data.c_str(), engine_data.length()));
  exec_context_ =
      std::unique_ptr<nv::IExecutionContext>(engine_->createExecutionContext());

  // Create stream.
  cudaStreamCreate(&stream_);

  // Allocate buffers.
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    nv::Dims dims = engine_->getTensorShape(name);

    CHECK(dims.nbDims > 0);
    if (dims.d[0] == -1) dims.d[0] = batch_size_;
    size_t num_elems = std::accumulate(&dims.d[0], &dims.d[dims.nbDims], 1,
                                       std::multiplies<size_t>());
    size_t num_bytes = num_elems * sizeof(float);

    BufferHandle buf_handle;
    buf_handle.size = num_bytes;
    buf_handle.dims = dims;
    cudaMallocHost(&buf_handle.host_buf, num_bytes);
    cudaMalloc(&buf_handle.device_buf, num_bytes);
    exec_context_->setTensorAddress(name, buf_handle.device_buf);

    buf_map_[name] = buf_handle;
  }

  // Set scores as constant.
  for (int i = 0; i < constants::kNumScoreLogits; ++i) {
    float score =
        0.05f *
        (static_cast<float>(i - constants::kScoreInflectionPoint) + 0.5f);
    static_cast<float*>(buf_map_[nn::trt::input::kScoresName].host_buf)[i] =
        score;
  }

  // Need to do this first according to API.
  exec_context_->setOptimizationProfileAsync(0, cudaStreamPerThread);
  cudaStreamSynchronize(stream_);

  // Configure inputs for execution context.
  BufferHandle input_planes = buf_map_[nn::trt::input::kPlanesName];
  BufferHandle input_features = buf_map_[nn::trt::input::kFeaturesName];
  BufferHandle input_scores = buf_map_[nn::trt::input::kScoresName];
  exec_context_->setInputShape(nn::trt::input::kPlanesName, input_planes.dims);
  exec_context_->setInputShape(nn::trt::input::kFeaturesName,
                               input_features.dims);
  exec_context_->setInputShape(nn::trt::input::kScoresName, input_scores.dims);
}

TrtEngineImpl::~TrtEngineImpl() {
  for (auto& [name, buf_handle] : buf_map_) {
    cudaFreeHost(buf_handle.host_buf);
    cudaFree(buf_handle.device_buf);
  }
}

void TrtEngineImpl::LoadBatch(int batch_id, const GoFeatures& features) {
  float* planes_buf =
      static_cast<float*>(buf_map_[nn::trt::input::kPlanesName].host_buf);
  float* feats_buf =
      static_cast<float*>(buf_map_[nn::trt::input::kFeaturesName].host_buf);
  int planes_slice_size = SliceSize(planes_shape_, 1);
  int feats_slice_size = SliceSize(feats_shape_, 1);

  std::fill(planes_buf + batch_id * planes_slice_size,
            planes_buf + (batch_id + 1) * planes_slice_size, 0);
  std::fill(feats_buf + batch_id * feats_slice_size,
            feats_buf + (batch_id + 1) * feats_slice_size, 0);
  LoadGoFeatures(planes_buf, feats_buf, planes_shape_, feats_shape_, features,
                 batch_id);
}

void TrtEngineImpl::RunInference() {
  // Inputs.
  BufferHandle input_planes = buf_map_[nn::trt::input::kPlanesName];
  BufferHandle input_features = buf_map_[nn::trt::input::kFeaturesName];
  BufferHandle input_scores = buf_map_[nn::trt::input::kScoresName];

  // Outputs.
  BufferHandle output_pi_logits = buf_map_[nn::trt::output::kPiLogitsName];
  BufferHandle output_pi_probs = buf_map_[nn::trt::output::kPiProbsName];
  BufferHandle output_outcome = buf_map_[nn::trt::output::kOutcomeName];
  BufferHandle output_own = buf_map_[nn::trt::output::kOwnershipName];
  BufferHandle output_score = buf_map_[nn::trt::output::kScoreName];

  cudaMemcpyAsync(input_planes.device_buf, input_planes.host_buf,
                  input_planes.size, cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(input_features.device_buf, input_features.host_buf,
                  input_features.size, cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(input_scores.device_buf, input_scores.host_buf,
                  input_scores.size, cudaMemcpyHostToDevice, stream_);
  CHECK(exec_context_->enqueueV3(stream_));
  cudaMemcpyAsync(output_pi_logits.host_buf, output_pi_logits.device_buf,
                  output_pi_logits.size, cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_pi_probs.host_buf, output_pi_probs.device_buf,
                  output_pi_probs.size, cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_outcome.host_buf, output_outcome.device_buf,
                  output_outcome.size, cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_own.host_buf, output_own.device_buf, output_own.size,
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_score.host_buf, output_score.device_buf,
                  output_score.size, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
}

void TrtEngineImpl::GetBatch(int batch_id, NNInferResult& result) {
  float* pi_logits_buf =
      static_cast<float*>(buf_map_[nn::trt::output::kPiLogitsName].host_buf);
  float* pi_probs_buf =
      static_cast<float*>(buf_map_[nn::trt::output::kPiProbsName].host_buf);
  float* outcome_buf =
      static_cast<float*>(buf_map_[nn::trt::output::kOutcomeName].host_buf);
  float* score_buf =
      static_cast<float*>(buf_map_[nn::trt::output::kScoreName].host_buf);

  int pi_slice_size = SliceSize(pi_shape_, 1);
  int outcome_slice_size = SliceSize(outcome_shape_, 1);
  int score_slice_size = SliceSize(score_shape_, 1);

  // std::copy doesn't work here.
  float* pi_logits = pi_logits_buf + pi_slice_size * batch_id;
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    result.move_logits[i] = pi_logits[i];
  }
  float* pi_probs = pi_probs_buf + pi_slice_size * batch_id;
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    result.move_probs[i] = pi_probs[i];
  }
  float* outcome = outcome_buf + outcome_slice_size * batch_id;
  for (int i = 0; i < constants::kNumValueLogits; ++i) {
    result.value_probs[i] = outcome[i];
  }
  float* score = score_buf + score_slice_size * batch_id;
  for (int i = 0; i < constants::kNumScoreLogits; ++i) {
    result.score_probs[i] = score[i];
  }
}

void TrtEngineImpl::GetOwnership(
    int batch_id, std::array<float, constants::kNumBoardLocs>& own) {
  LOG(ERROR) << "Ownership Not Supported on TRT.";
}

}  // namespace

/* static */ std::unique_ptr<TrtEngine> TrtEngine::Create(std::string path,
                                                          int batch_size) {
  return std::make_unique<TrtEngineImpl>(path, batch_size);
}

}  // namespace nn
