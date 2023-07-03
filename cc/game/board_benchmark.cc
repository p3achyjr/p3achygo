#include "benchmark/benchmark.h"
#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"

#define DEFINE_BENCHMARK(name, sequence)           \
  static void BM_##name(benchmark::State& state) { \
    for (auto _ : state) {                         \
      PlayGame(sequence);                          \
    }                                              \
  }

namespace {

using namespace ::game;

// . . . . . . .
// . x o x o x o
static constexpr Move kBasicSequence[] = {{BLACK, {1, 1}}, {WHITE, {1, 2}},
                                          {BLACK, {1, 3}}, {WHITE, {1, 4}},
                                          {BLACK, {1, 5}}, {WHITE, {1, 6}}};

// . . . . . . .
// . . x . . . .
// . x o x . . .
// . . x . . . .
static constexpr Move kPonnukiSequence[] = {{BLACK, {1, 2}},
                                            {WHITE, {2, 2}},
                                            {BLACK, {2, 1}},
                                            {BLACK, {3, 2}},
                                            {BLACK, {2, 3}}};

// . . . . . . .
// . o x x x o x
// o o o o o x x
// . o x x x o .
// . . . . . x .
// . . o . . . .
static constexpr Move kSanSanSequence[] = {
    {BLACK, {3, 3}}, {WHITE, {2, 2}}, {BLACK, {3, 2}}, {WHITE, {2, 3}},
    {BLACK, {2, 5}}, {WHITE, {2, 4}}, {BLACK, {3, 4}}, {WHITE, {1, 5}},
    {BLACK, {1, 2}}, {WHITE, {1, 1}}, {BLACK, {1, 4}}, {WHITE, {3, 1}},
    {BLACK, {1, 2}}, {WHITE, {2, 0}}, {BLACK, {1, 3}}, {WHITE, {2, 1}},
    {BLACK, {1, 6}}, {WHITE, {3, 5}}, {BLACK, {2, 6}}, {WHITE, {5, 2}},
    {BLACK, {4, 5}}};

// . . . . . . .
// . x x x x x .
// . x o o o o o
// . x o . . . .
// . x o . . . .
// . x o . . . .
// . . o . . . .
static constexpr Move kBigChainJoinsSequence[] = {
    {BLACK, {1, 2}}, {WHITE, {2, 3}}, {BLACK, {1, 4}}, {WHITE, {2, 5}},
    {BLACK, {2, 1}}, {WHITE, {3, 2}}, {BLACK, {4, 1}}, {WHITE, {5, 2}},
    {BLACK, {1, 3}}, {WHITE, {2, 4}}, {BLACK, {1, 5}}, {WHITE, {2, 6}},
    {BLACK, {3, 1}}, {WHITE, {4, 2}}, {BLACK, {5, 1}}, {WHITE, {6, 2}},
    {BLACK, {1, 1}}, {WHITE, {2, 2}}};

// . . . x o o x
// . . . x o x .
// . . x o o x .
// . . x o x . .
// . . o x . . .
// . . o . . . .
static constexpr Move kLadderSequence[] = {
    {BLACK, {3, 2}}, {WHITE, {4, 2}}, {BLACK, {4, 3}}, {WHITE, {3, 3}},
    {BLACK, {2, 2}}, {WHITE, {5, 2}}, {BLACK, {3, 4}}, {WHITE, {2, 3}},
    {BLACK, {1, 3}}, {WHITE, {2, 4}}, {BLACK, {2, 5}}, {WHITE, {1, 4}},
    {BLACK, {1, 5}}, {WHITE, {0, 4}}, {BLACK, {0, 3}}, {WHITE, {0, 5}},
    {BLACK, {0, 6}}};

template <size_t N>
static inline void PlayGame(const Move (&moves)[N]) {
  Board board;
  for (const Move& move : moves) {
    board.PlayMove(move.loc, move.color);
  }
}

DEFINE_BENCHMARK(BasicSequence, kBasicSequence);
DEFINE_BENCHMARK(PonnukiSequence, kPonnukiSequence);
DEFINE_BENCHMARK(SanSanSequence, kSanSanSequence);
DEFINE_BENCHMARK(BigChainJoinsSequence, kBigChainJoinsSequence);
DEFINE_BENCHMARK(LadderSequence, kLadderSequence);

BENCHMARK(BM_BasicSequence);
BENCHMARK(BM_PonnukiSequence);
BENCHMARK(BM_SanSanSequence);
BENCHMARK(BM_BigChainJoinsSequence);
BENCHMARK(BM_LadderSequence);

}  // namespace

BENCHMARK_MAIN();

#undef DEFINE_BENCHMARK
