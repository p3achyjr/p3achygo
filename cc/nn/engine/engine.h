#ifndef NN_ENGINE_ENGINE_H_
#define NN_ENGINE_ENGINE_H_

#include <memory>

#include "cc/nn/engine/go_features.h"

namespace nn {

struct NNInferResult {
  std::array<float, constants::kMaxMovesPerPosition> move_logits;
  std::array<float, constants::kMaxMovesPerPosition> move_probs;
  std::array<float, constants::kNumValueLogits> value_probs;
  std::array<float, constants::kNumScoreLogits> score_probs;
};

class Engine {
 public:
  enum class Kind : uint8_t {
    kTF = 0,
    kTFTrt = 1,
    kTrt = 2,
  };

  virtual ~Engine() = default;
  virtual Kind kind() = 0;
  virtual std::string path() = 0;
  virtual void LoadBatch(int batch_id, const GoFeatures& features) = 0;
  virtual void RunInference() = 0;
  virtual void GetBatch(int batch_id, NNInferResult& result) = 0;
  virtual void GetOwnership(
      int batch_id,
      std::array<float, constants::kMaxMovesPerPosition>& own) = 0;

 protected:
  Engine() = default;
};
}  // namespace nn

#endif
