#include "benchmark/benchmark.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/tree.h"
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

static void BM_Gumbel(benchmark::State& state) {
  NNInterface nn_interface(1);
  Probability probability;
  for (auto _ : state) {
    state.PauseTiming();
    Game game;
    Color color_to_move = BLACK;
    TreeNode root_node;
    GumbelEvaluator gumbel_evaluator(&nn_interface, 0);
    state.ResumeTiming();

    GumbelResult gumbel_res = gumbel_evaluator.SearchRoot(
        probability, game, &root_node, color_to_move, state.range(0),
        state.range(1));
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
