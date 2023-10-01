#ifndef NN_ENGINE_TRT_ENGINE_H_
#define NN_ENGINE_TRT_ENGINE_H_

#include <memory>

#include "cc/nn/engine/engine.h"

namespace nn {

/*
 * Wrapper around inference for TRT engines converted from a .onnx model.
 */
class TrtEngine : public Engine {
 public:
  virtual ~TrtEngine() = default;

  virtual Engine::Kind kind() override = 0;
  virtual std::string path() override = 0;
  virtual void LoadBatch(int batch_id, const GoFeatures& features) override = 0;
  virtual void RunInference() override = 0;
  virtual void GetBatch(int batch_id, NNInferResult& result) override = 0;
  virtual void GetOwnership(
      int batch_id,
      std::array<float, constants::kMaxMovesPerPosition>& own) override = 0;

  static std::unique_ptr<TrtEngine> Create(std::string path, int batch_size);

 protected:
  TrtEngine() = default;
};

}  // namespace nn

#endif
