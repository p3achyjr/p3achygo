#ifndef NN_ENGINE_TF_ENGINE_H_
#define NN_ENGINE_TF_ENGINE_H_

#include <memory>

#include "cc/nn/engine/engine.h"

namespace nn {

/*
 * Wrapper around inference for Tensorflow engines.
 */
class TFEngine : public Engine {
 public:
  enum class Kind {
    kUnknown = 0,
    kTF = 1,
    kTRT = 2,
  };

  virtual ~TFEngine() = default;

  virtual Engine::Kind kind() override = 0;
  virtual std::string path() override = 0;
  virtual void LoadBatch(int batch_id, const GoFeatures& features) override = 0;
  virtual void RunInference() override = 0;
  virtual void GetBatch(int batch_id, NNInferResult& result) override = 0;
  virtual void GetOwnership(
      int batch_id,
      std::array<float, constants::kNumBoardLocs>& own) override = 0;

  static std::unique_ptr<TFEngine> Create(std::string path, Kind kind,
                                          int batch_size);

 protected:
  TFEngine() = default;
};

}  // namespace nn

#endif
