#include "cc/nn/engine/trt_calibrator.h"

#include <NvInfer.h>

#include <deque>
#include <fstream>

#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/nn/engine/buf_utils.h"
#include "cc/nn/engine/go_dataset.h"
#include "cc/nn/engine/go_features.h"
#include "cc/nn/engine/trt_names.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature_util.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/record_reader.h"

namespace nn {
namespace trt {
namespace {
using namespace ::game;
using ::nn::GoFeatures;
using ::nn::LoadFeatures;
using ::nn::LoadPlanes;

class Int8CalibratorImpl : public Int8Calibrator {
 public:
  Int8CalibratorImpl(size_t batch_size, GoDataset* go_ds,
                     std::string calib_cache_path, bool use_cache);
  ~Int8CalibratorImpl();
  void initialize();
  int getBatchSize() const noexcept override { return batch_size_; }
  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) noexcept override;
  const void* readCalibrationCache(size_t& length) noexcept override;
  void writeCalibrationCache(const void* cache,
                             size_t length) noexcept override;

 private:
  static constexpr int kNumCalibExamples = 2000;
  size_t batch_size_;
  std::string calib_tfrec_path_;
  int batch_counter_;
  int ex_counter_;
  int num_batches_;
  GoDataset* const go_ds_;
  GoDataset::Iterator go_ds_iterator_;
  std::string calib_cache_path_;
  std::vector<char> calibration_cache_;
  const bool use_cache_;

  size_t nbytes_planes_;
  void* host_input_planes_;
  void* device_input_planes_;

  size_t nbytes_features_;
  void* host_input_features_;
  void* device_input_features_;

  size_t nbytes_scores_;
  std::vector<float> scores_;
  void* device_input_scores_;
};

Int8CalibratorImpl::Int8CalibratorImpl(size_t batch_size, GoDataset* go_ds,
                                       std::string calib_cache_path,
                                       bool use_cache)
    : batch_size_(batch_size),
      batch_counter_(0),
      ex_counter_(0),
      go_ds_(go_ds),
      go_ds_iterator_(go_ds_->begin()),
      calib_cache_path_(calib_cache_path),
      use_cache_(use_cache) {
  CHECK(go_ds->batch_size() == batch_size_);
  initialize();
}

Int8CalibratorImpl::~Int8CalibratorImpl() {
  cudaFree(device_input_planes_);
  cudaFreeHost(host_input_planes_);
  cudaFree(device_input_features_);
  cudaFreeHost(host_input_features_);
  cudaFree(device_input_scores_);
}

void Int8CalibratorImpl::initialize() {
  nbytes_planes_ = sizeof(float) * batch_size_ *
                   constants::kNumInputFeaturePlanes * BOARD_LEN * BOARD_LEN;
  nbytes_features_ =
      sizeof(float) * batch_size_ * constants::kNumInputFeatureScalars;

  nbytes_scores_ = sizeof(float) * constants::kNumScoreLogits;
  cudaMalloc(&device_input_planes_, nbytes_planes_);
  cudaMallocHost(&host_input_planes_, nbytes_planes_);
  cudaMalloc(&device_input_features_, nbytes_features_);
  cudaMallocHost(&host_input_features_, nbytes_features_);
  cudaMalloc(&device_input_scores_, nbytes_scores_);

  scores_.resize(constants::kNumScoreLogits);
  for (int i = 0; i < constants::kNumScoreLogits; ++i) {
    float score =
        0.05f *
        (static_cast<float>(i - constants::kScoreInflectionPoint) + 0.5f);
    scores_[i] = score;
  }
}

bool Int8CalibratorImpl::getBatch(void* bindings[], const char* names[],
                                  int nbBindings) noexcept {
  if (ex_counter_ >= kNumCalibExamples || go_ds_iterator_ == go_ds_->end()) {
    return false;
  }

  std::array<int, 4> planes_shape = {static_cast<int>(batch_size_), BOARD_LEN,
                                     BOARD_LEN,
                                     constants::kNumInputFeaturePlanes};
  std::array<int, 2> feats_shape = {static_cast<int>(batch_size_),
                                    constants::kNumInputFeatureScalars};
  std::vector<GoDataset::Row> batch_examples = *go_ds_iterator_;
  ++go_ds_iterator_;

  for (int i = 0; i < nbBindings; ++i) {
    const char* name = names[i];
    void* host_binding;
    void* device_binding;
    size_t nbytes;

    if (std::string(name) == input::kPlanesName) {
      host_binding = host_input_planes_;
      device_binding = device_input_planes_;
      nbytes = nbytes_planes_;
      memset(host_binding, 0, nbytes);
    } else if (std::string(name) == input::kFeaturesName) {
      host_binding = host_input_features_;
      device_binding = device_input_features_;
      nbytes = nbytes_features_;
      memset(host_binding, 0, nbytes);
    } else if (std::string(name) == input::kScoresName) {
      host_binding = reinterpret_cast<void*>(scores_.data());
      device_binding = device_input_scores_;
      nbytes = nbytes_scores_;
    } else {
      LOG(FATAL) << "Unknown Binding Name: " << name;
    }

    // Fill binding.
    for (int batch_id = 0; batch_id < batch_size_; ++batch_id) {
      const GoFeatures& example = batch_examples[batch_id].features;
      if (std::string(name) == input::kPlanesName) {
        LoadPlanes(static_cast<float*>(host_binding), planes_shape, example,
                   batch_id);
      } else if (std::string(name) == input::kFeaturesName) {
        LoadFeatures(static_cast<float*>(host_binding), feats_shape, example,
                     batch_id);
      }
    }

    // Copy to device.
    cudaMemcpy(device_binding, host_binding, nbytes, cudaMemcpyHostToDevice);
    bindings[i] = device_binding;
  }

  ++batch_counter_;
  ex_counter_ += batch_size_;
  return true;
}

const void* Int8CalibratorImpl::readCalibrationCache(size_t& length) noexcept {
  if (!use_cache_) return nullptr;

  calibration_cache_.clear();
  std::ifstream input(calib_cache_path_, std::ios::binary);
  input >> std::noskipws;
  if (input.good()) {
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
              std::back_inserter(calibration_cache_));
  }
  length = calibration_cache_.size();
  return length ? calibration_cache_.data() : nullptr;
}

void Int8CalibratorImpl::writeCalibrationCache(const void* cache,
                                               size_t length) noexcept {
  if (!use_cache_) return;

  std::ofstream output(calib_cache_path_, std::ios::binary);
  output.write(reinterpret_cast<const char*>(cache), length);
}

}  // namespace

/* static */ std::unique_ptr<Int8Calibrator> Int8Calibrator::Create(
    size_t batch_size, GoDataset* go_ds, std::string calib_cache_path) {
  std::unique_ptr<Int8Calibrator> calibrator =
      std::make_unique<Int8CalibratorImpl>(batch_size, go_ds, calib_cache_path,
                                           false /* use_cache */);

  return calibrator;
}

}  // namespace trt
}  // namespace nn
