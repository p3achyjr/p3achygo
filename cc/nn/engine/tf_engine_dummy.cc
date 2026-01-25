#include "cc/nn/engine/tf_engine.h"

#include "absl/log/log.h"

namespace nn {
namespace {

class TFEngineImpl : public TFEngine {
 public:
  TFEngineImpl(std::string path, Kind kind, int batch_size)
      : path_(path), kind_(kind), batch_size_(batch_size) {
    LOG(WARNING) << "Using dummy TensorFlow engine (TensorFlow support not "
                    "compiled). Path: "
                 << path << ", Kind: " << static_cast<int>(kind)
                 << ", Batch size: " << batch_size;
  }
  ~TFEngineImpl() = default;

  Engine::Kind kind() override {
    switch (kind_) {
      case TFEngine::Kind::kTF:
        return Engine::Kind::kTF;
      case TFEngine::Kind::kTRT:
        return Engine::Kind::kTFTrt;
      case TFEngine::Kind::kXLA:
        return Engine::Kind::kTFXla;
      default:
        return Engine::Kind::kUnknown;
    }
  }

  std::string path() override { return path_; }

  void LoadBatch(int batch_id, const GoFeatures& features) override {
    LOG(INFO) << "Dummy TFEngine::LoadBatch called for batch_id: " << batch_id;
  }

  void RunInference() override {
    LOG(INFO) << "Dummy TFEngine::RunInference called (no-op)";
  }

  void GetBatch(int batch_id, NNInferResult& result) override {
    LOG(INFO) << "Dummy TFEngine::GetBatch called for batch_id: " << batch_id;
    // Initialize with zeros to avoid undefined behavior
    result.move_logits.fill(0.0f);
    result.move_probs.fill(0.0f);
    result.value_probs.fill(0.0f);
    result.score_probs.fill(0.0f);
  }

  void GetOwnership(
      int batch_id,
      std::array<float, constants::kNumBoardLocs>& own) override {
    LOG(INFO) << "Dummy TFEngine::GetOwnership called for batch_id: "
              << batch_id;
    // Initialize with zeros
    own.fill(0.0f);
  }

 private:
  const std::string path_;
  const Kind kind_;
  const int batch_size_;
};

}  // namespace

/* static */ std::unique_ptr<TFEngine> TFEngine::Create(std::string path,
                                                        TFEngine::Kind kind,
                                                        int batch_size) {
  return std::make_unique<TFEngineImpl>(path, kind, batch_size);
}

}  // namespace nn