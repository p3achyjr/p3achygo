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
    kUnknown = 0,
    kTrt = 1,
    kTF = 2,
    kTFTrt = 3,
    kTFXla = 4,
  };

  virtual ~Engine() = default;
  virtual Kind kind() = 0;
  virtual std::string path() = 0;
  virtual void LoadBatch(int batch_id, const GoFeatures& features) = 0;
  virtual void RunInference() = 0;
  virtual void GetBatch(int batch_id, NNInferResult& result) = 0;
  virtual void GetOwnership(
      int batch_id, std::array<float, constants::kNumBoardLocs>& own) = 0;

 protected:
  Engine() = default;
};

inline std::string KindToString(Engine::Kind kind) {
  if (kind == Engine::Kind::kTrt) {
    return "TensorRT";
  } else if (kind == Engine::Kind::kTF) {
    return "TF";
  } else if (kind == Engine::Kind::kTFTrt) {
    return "TF-TRT";
  } else if (kind == Engine::Kind::kTFXla) {
    return "TF-XLA";
  }

  return "??";
}
}  // namespace nn

#endif
