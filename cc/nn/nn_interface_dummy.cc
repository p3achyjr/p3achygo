#include <stdlib.h>

#include "cc/game/game.h"
#include "cc/nn/nn_interface.h"

namespace nn {
namespace {
using namespace ::tensorflow;
}  // namespace

NNInterface::Cache::Cache(int num_threads)
    : num_threads_(num_threads), thread_cache_size_(num_threads) {}

void NNInterface::Cache::Insert(int thread_id, const CacheKey& cache_key,
                                const NNInferResult& infer_result) {}

bool NNInterface::Cache::Contains(int thread_id, const CacheKey& cache_key) {
  return false;
}

std::optional<NNInferResult> NNInterface::Cache::Get(
    int thread_id, const CacheKey& cache_key) {
  return std::nullopt;
}

NNInterface::NNInterface(int _num_threads)
    : scope_preprocess_(Scope::NewRootScope()),
      scope_postprocess_(Scope::NewRootScope()),
      num_threads_(0),
      nn_cache_(0) {}

NNInterface::~NNInterface() = default;

absl::Status NNInterface::Initialize(std::string&& _model_path) {
  return absl::OkStatus();
}

NNInferResult NNInterface::LoadAndGetInference(int _thread_id,
                                               const game::Game& _game,
                                               game::Color _color_to_move) {
  return NNInferResult{
      .move_logits{},
      .move_probs{},
      .value_probs{},
      .score_probs{},
  };
}

void NNInterface::RegisterThread(int thread_id) {}

void NNInterface::UnregisterThread(int thread_id) {}

void NNInterface::InferLoop() {}

void NNInterface::Infer() {}

bool NNInterface::ShouldInfer() const { return false; }

}  // namespace nn
