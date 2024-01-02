#include "cc/nn/engine/trt_engine.h"

namespace nn {
namespace {

class TrtEngineImpl : public TrtEngine {
 public:
  TrtEngineImpl() = default;
  ~TrtEngineImpl() = default;

  Engine::Kind kind() override { return Engine::Kind::kTrt; }
  std::string path() override { return ""; }
  void LoadBatch(int batch_id, const GoFeatures& features) override{};
  void RunInference() override{};
  void GetBatch(int batch_id, NNInferResult& result) override{};
  void GetOwnership(
      int batch_id,
      std::array<float, constants::kNumBoardLocs>& own) override{};
};

}  // namespace

/* static */ std::unique_ptr<TrtEngine> TrtEngine::Create(std::string path,
                                                          int batch_size) {
  return std::make_unique<TrtEngineImpl>();
}

}  // namespace nn
