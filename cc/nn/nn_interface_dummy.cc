#include <stdlib.h>

#include "cc/core/cache.h"
#include "cc/nn/nn_interface.h"

namespace nn {
namespace {
using namespace ::tensorflow;
using namespace ::core;
}  // namespace

NNInterface::NNInterface(int _num_threads) : num_threads_(0), timeout_(0) {}

NNInterface::~NNInterface() = default;

absl::Status NNInterface::Initialize(std::string&& _model_path) {
  return absl::OkStatus();
}

NNInferResult NNInterface::LoadAndGetInference(int _thread_id,
                                               const game::Game& _game,
                                               game::Color _color_to_move,
                                               Probability& _probability) {
  return NNInferResult{
      .move_logits{},
      .move_probs{},
      .value_probs{},
      .score_probs{},
  };
}

void NNInterface::RegisterThread(int thread_id) {}

void NNInterface::UnregisterThread(int thread_id) {}

void NNInterface::InitializeCache(size_t cache_size) {}

bool NNInterface::CacheContains(int thread_id, const NNKey& key) {
  return false;
}

std::optional<NNInferResult> NNInterface::CacheGet(int thread_id,
                                                   const NNKey& key) {
  return std::nullopt;
}

void NNInterface::CacheInsert(int thread_id, const NNKey& key,
                              const NNInferResult& result) {}

void NNInterface::InferLoop() {}

void NNInterface::Infer() {}

bool NNInterface::ShouldInfer() const { return false; }

}  // namespace nn
