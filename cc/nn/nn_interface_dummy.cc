#include <stdlib.h>

#include "cc/game/game.h"
#include "cc/nn/nn_interface.h"

namespace nn {
namespace {
using namespace ::tensorflow;
static constexpr int kNumMovesToDraw = 16;
static constexpr int kValuableCounters[] = {16, 17, 35, 41, 57};

bool IsValuableCounter(int counter) {
  for (int v_counter : kValuableCounters) {
    if (v_counter == counter) {
      return true;
    }
  }

  return false;
}
}  // namespace

NNInterface::NNInterface(int _num_threads)
    : scope_cast_input_(Scope::NewRootScope()),
      scope_cast_output_(Scope::NewRootScope()),
      batch_size_(0) {}

NNInterface::~NNInterface() = default;

absl::Status NNInterface::Initialize(std::string&& _model_path) {
  return absl::OkStatus();
}

absl::Status NNInterface::LoadBatch(int _thread_id, const game::Game& _game,
                                    int _color_to_move) {
  return absl::OkStatus();
}

NNInferResult NNInterface::GetInferenceResult(int _thread_id) {
  NNInferResult result{
      .move_logits{},
      .move_probs{},
      .value_probs{},
      .ownership{},
      .score_probs{},
  };

  return result;
}

void NNInterface::RegisterThread(int thread_id) {}

void NNInterface::UnregisterThread(int thread_id) {}

void NNInterface::InferLoop() {}

void NNInterface::Infer() {}

bool NNInterface::ShouldInfer() const { return false; }

}  // namespace nn
