#include "cc/nn/engine/trt_engine.h"

namespace nn {
namespace {

class TrtEngineImpl : public TrtEngine {
 public:
  TrtEngineImpl() = default;
  ~TrtEngineImpl() = default;

  Engine::Kind kind() override { return Engine::Kind::kTrt; }
  std::string path() override { return ""; }
  void LoadBatch(int batch_id, const GoFeatures& features) override {};
  void RunInference() override {};
  // Uniform policy / neutral value. Real inference is unavailable in CPU-only
  // builds, but leaving `result` uninitialized makes search UB, so emit a valid
  // degenerate distribution instead. Lets sanitizer builds play real games.
  void GetBatch(int batch_id, NNInferResult& result) override {
    result.move_probs.fill(1.0f / constants::kMaxMovesPerPosition);
    result.move_logits.fill(0.0f);
    result.value_probs.fill(1.0f / constants::kNumValueLogits);
    result.score_probs.fill(1.0f / constants::kNumScoreLogits);
    result.opt_move_probs.fill(1.0f / constants::kMaxMovesPerPosition);
    result.err2_outcome = 0.0f;
    result.mcts_value_dist.fill(1.0f / constants::kNumVBuckets);
  };
  void GetOwnership(int batch_id,
                    std::array<float, constants::kNumBoardLocs>& own) override {
    own.fill(0.0f);
  };
};

}  // namespace

/* static */ std::unique_ptr<TrtEngine> TrtEngine::Create(std::string path,
                                                          int batch_size,
                                                          int version) {
  return std::make_unique<TrtEngineImpl>();
}

}  // namespace nn
