// Degenerate-eval re-evaluation (NNInterface::SetDegenerateReeval): the engine
// plays a script of outputs, one per RunInference, so each test knows which
// call is degenerate and checks which result the interface keeps and how many
// engine calls it made. Covers the synchronous path (LoadAndGetInference) and
// round-free LoadEntry / FetchEntry.

#include <atomic>
#include <vector>

#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/nn_interface.h"
// After the absl headers, so doctest's CHECK is the one in effect.
#include "cc/core/doctest_include.h"

namespace nn {
namespace {

struct Step {
  float logit;             // every move logit
  bool value_bad = false;  // outcome +1 with all mcts-value mass at -1
};

class ScriptEngine final : public Engine {
 public:
  explicit ScriptEngine(std::vector<Step> script) : script_(script) {}

  Kind kind() override { return Kind::kUnknown; }
  std::string path() override { return "script"; }
  void LoadBatch(int, const GoFeatures&) override {}
  void RunInference() override {
    const int i = calls_.fetch_add(1);
    cur_ = script_[std::min<int>(i, script_.size() - 1)];
  }
  void GetBatch(int, NNInferResult& r) override {
    r = NNInferResult{};
    r.move_logits.fill(cur_.logit);
    r.move_probs.fill(1.0f / constants::kMaxMovesPerPosition);
    r.opt_move_probs.fill(1.0f / constants::kMaxMovesPerPosition);
    r.mcts_value_dist.fill(0.0f);
    if (cur_.value_bad) {
      r.value_probs = {0.0f, 1.0f};
      r.mcts_value_dist[0] = 1.0f;
    } else {
      r.value_probs = {0.5f, 0.5f};
      r.mcts_value_dist[constants::kNumVBuckets / 2] = 1.0f;
    }
  }
  void GetOwnership(int,
                    std::array<float, constants::kNumBoardLocs>& own) override {
    own.fill(0.0f);
  }
  int calls() const { return calls_.load(); }

 private:
  std::vector<Step> script_;
  std::atomic<int> calls_{0};
  Step cur_{0.0f};
};

float MaxLogit(const NNInferResult& r) {
  return *std::max_element(r.move_logits.begin(), r.move_logits.end());
}

// Synchronous path, single-thread interface (inference runs inline).
std::pair<float, int> RunSync(std::vector<Step> script, float max_logit,
                              float value_disagree, int retries) {
  auto* engine = new ScriptEngine(script);
  NNInterface nn_interface(1, std::unique_ptr<Engine>(engine));
  nn_interface.SetDegenerateReeval(max_logit, value_disagree, retries);
  core::Probability prob(7);
  game::Game game;
  NNInferResult r = nn_interface.LoadAndGetInference(0, game, BLACK, prob);
  return {MaxLogit(r), engine->calls()};
}

// Round-free LoadEntry / FetchEntry (no cache).
std::pair<float, int> RunRoundFree(std::vector<Step> script, float max_logit,
                                   float value_disagree, int retries) {
  auto* engine = new ScriptEngine(script);
  NNInterface nn_interface(2, /*cache_size=*/0, std::unique_ptr<Engine>(engine),
                           /*engine_batch=*/1);
  nn_interface.SetDegenerateReeval(max_logit, value_disagree, retries);
  core::Probability prob(7);
  game::Game game;
  nn_interface.LoadEntry(0, /*offset=*/0, game, BLACK, prob);
  NNInferResult r = nn_interface.FetchEntry(0, /*offset=*/0, game, BLACK);
  return {MaxLogit(r), engine->calls()};
}

TEST_CASE("off: degenerate eval is returned as is") {
  CHECK(RunSync({{30}}, 0, 0, 2) == std::make_pair(30.0f, 1));
  CHECK(RunRoundFree({{30}}, 0, 0, 2) == std::make_pair(30.0f, 1));
}

TEST_CASE("non-degenerate eval is not retried") {
  CHECK(RunSync({{5}, {1}}, 15, 1.0f, 2) == std::make_pair(5.0f, 1));
  CHECK(RunRoundFree({{5}, {1}}, 15, 1.0f, 2) == std::make_pair(5.0f, 1));
}

TEST_CASE("max-logit gate: first non-degenerate retry is kept") {
  CHECK(RunSync({{30}, {5}, {1}}, 15, 0, 2) == std::make_pair(5.0f, 2));
  CHECK(RunRoundFree({{30}, {30}, {5}}, 15, 0, 2) == std::make_pair(5.0f, 3));
}

TEST_CASE("all retries degenerate: smallest max logit is kept") {
  CHECK(RunSync({{30}, {20}, {25}}, 15, 0, 2) == std::make_pair(20.0f, 3));
  CHECK(RunRoundFree({{30}, {20}, {25}}, 15, 0, 2) == std::make_pair(20.0f, 3));
}

TEST_CASE("value-disagreement gate") {
  // Logits fine, outcome +1 vs mcts value ~-1: flagged, retry is clean.
  CHECK(RunSync({{2, true}, {3, false}}, 15, 1.0f, 2) ==
        std::make_pair(3.0f, 2));
  CHECK(RunRoundFree({{2, true}, {3, false}}, 15, 1.0f, 2) ==
        std::make_pair(3.0f, 2));
  // Value gate off: not flagged.
  CHECK(RunSync({{2, true}, {3, false}}, 15, 0, 2) == std::make_pair(2.0f, 1));
}

}  // namespace
}  // namespace nn
