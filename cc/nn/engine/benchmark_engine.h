#ifndef NN_ENGINE_BENCHMARK_ENGINE_H_
#define NN_ENGINE_BENCHMARK_ENGINE_H_

#include <unordered_map>

#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/go_dataset.h"

namespace nn {

class Stats {
 public:
  virtual ~Stats() = default;
  virtual void Update(const NNInferResult& result, const GoDataset::Row& row,
                      double time) = 0;

 protected:
  Stats() = default;
};

class DefaultStats final : public Stats {
 public:
  DefaultStats() = default;
  ~DefaultStats() = default;
  void Update(const NNInferResult& result, const GoDataset::Row& row,
              double time) override;
  std::string ToString();

 private:
  std::unordered_map<std::string, double> stats_;
};

void Benchmark(Engine* const engine, GoDataset* const go_ds, Stats& stats);

}  // namespace nn

#endif
