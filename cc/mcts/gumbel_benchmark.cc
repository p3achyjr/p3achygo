#include "benchmark/benchmark.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/tree.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/go_features.h"
#include "cc/nn/nn_interface.h"

#define DEFINE_BENCHMARK(name, sequence)           \
  static void BM_##name(benchmark::State& state) { \
    for (auto _ : state) {                         \
      state.PauseTiming();                         \
      Board board;                                 \
      state.ResumeTiming();                        \
      for (const Move& move : sequence) {          \
        board.PlayMove(move.loc, move.color);      \
      }                                            \
    }                                              \
  }                                                \
  BENCHMARK(BM_##name);

#define DEFINE_CHECKED_BENCHMARK(name, sequence)             \
  static void BM_Checked##name(benchmark::State& state) {    \
    for (auto _ : state) {                                   \
      state.PauseTiming();                                   \
      Board board;                                           \
      state.ResumeTiming();                                  \
      for (const Move& move : sequence) {                    \
        if (!board.IsValidMove(move.loc, move.color)) break; \
        board.PlayMove(move.loc, move.color);                \
      }                                                      \
    }                                                        \
  }                                                          \
  BENCHMARK(BM_Checked##name);

namespace {

using namespace ::core;
using namespace ::game;
using namespace ::mcts;
using namespace ::nn;

class NullEngine final : public Engine {
 public:
  Kind kind() override { return Kind::kUnknown; }
  std::string path() override { return "null"; }
  void LoadBatch(int, const GoFeatures&) override {}
  void RunInference() override {}
  void GetBatch(int, NNInferResult& result) override {
    result.move_probs.fill(1.0f / constants::kMaxMovesPerPosition);
    result.move_logits.fill(0.0f);
    result.value_probs.fill(1.0f / constants::kNumValueLogits);
    result.score_probs.fill(1.0f / constants::kNumScoreLogits);
  }
  void GetOwnership(int,
                    std::array<float, constants::kNumBoardLocs>& own) override {
    own.fill(0.0f);
  }
};

static void BM_Gumbel(benchmark::State& state) {
  NNInterface nn_interface(1, std::make_unique<NullEngine>());
  Probability probability;
  for (auto _ : state) {
    state.PauseTiming();
    Game game;
    Color color_to_move = BLACK;
    auto node_table = std::make_unique<MctsNodeTable>();
    TreeNode* root_node =
        node_table->GetOrCreate(game.board().hash(), color_to_move, false);
    GumbelEvaluator gumbel_evaluator(&nn_interface, 0);
    state.ResumeTiming();

    GumbelResult gumbel_res = gumbel_evaluator.SearchRoot(
        probability, game, node_table.get(), root_node, color_to_move,
        GumbelSearchParams{state.range(0), state.range(1)});
    Loc move = gumbel_res.mcts_move;
    game.PlayMove(move, color_to_move);
    color_to_move = OppositeColor(color_to_move);
  }
}

BENCHMARK(BM_Gumbel)
    ->Args({1, 1})
    ->Args({8, 2})
    ->Args({32, 4})
    ->Args({48, 8})
    ->Args({64, 4})
    ->Args({128, 4});

}  // namespace

BENCHMARK_MAIN();

#undef DEFINE_BENCHMARK
