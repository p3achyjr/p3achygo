#include "cc/nn/nn_interface_base.h"

#include <algorithm>

#include "cc/game/symmetry.h"

namespace nn {

using namespace ::game;

std::unique_ptr<Waker> MakeWaker(NNInterfaceBase::WakeStrategy strategy) {
  switch (strategy) {
    case NNInterfaceBase::WakeStrategy::kMutex:
      return std::make_unique<MutexWaker>();
    case NNInterfaceBase::WakeStrategy::kGenCounter:
    default:
      return std::make_unique<GenCounterWaker>();
  }
}

NNKey MakeNNKey(const Game& game, Color color_to_move, int num_cache_last_moves,
                bool cache_key_ko) {
  const int num_moves = game.num_moves();
  std::array<Loc, constants::kNumLastMoves> last_moves;
  last_moves.fill(kNoopLoc);
  for (int i = 0; i < num_cache_last_moves; ++i) {
    const int off = num_moves - num_cache_last_moves + i;
    if (off >= 0) {
      last_moves[constants::kNumLastMoves - num_cache_last_moves + i] =
          game.move(off).loc;
    }
  }
  return NNKey{color_to_move, game.board().hash(), last_moves, game.komi(),
               cache_key_ko ? game.board().ko_loc() : kNoopLoc};
}

GoFeatures MakeGoFeatures(const Game& game, Color color_to_move, Symmetry sym) {
  GoFeatures input_features;
  input_features.bsize = BOARD_LEN;
  input_features.color = color_to_move;

  int num_moves = game.num_moves();
  for (int i = 0; i < constants::kNumLastMoves; ++i) {
    int mv_offset = num_moves - constants::kNumLastMoves + i;
    if (mv_offset < 0) {
      input_features.last_moves[i] = kNoopLoc;
      continue;
    } else if (game.move(mv_offset).loc == kPassLoc) {
      input_features.last_moves[i] = kPassLoc;
      continue;
    }
    input_features.last_moves[i] =
        ApplySymmetry(sym, game.move(mv_offset).loc, BOARD_LEN);
  }

  input_features.board = ApplySymmetry(sym, game.board().position(), BOARD_LEN);
  input_features.stones_atari =
      ApplySymmetry(sym, game.board().GetStonesInAtari(), BOARD_LEN);
  input_features.stones_two_liberties =
      ApplySymmetry(sym, game.board().GetStonesWithLiberties(2), BOARD_LEN);
  input_features.stones_three_liberties =
      ApplySymmetry(sym, game.board().GetStonesWithLiberties(3), BOARD_LEN);
  input_features.stones_laddered =
      ApplySymmetry(sym, game.board().GetLadderedStones(), BOARD_LEN);
  input_features.komi = game.komi();
  return input_features;
}

void UnapplySymmetry(Symmetry sym, NNInferResult& infer_result) {
  std::array<float, constants::kNumBoardLocs> grid_logits_sym;
  std::array<float, constants::kNumBoardLocs> grid_probs_sym;
  std::array<float, constants::kNumBoardLocs> grid_probs_opt_sym;
  std::copy(infer_result.move_logits.begin(),
            infer_result.move_logits.begin() + constants::kNumBoardLocs,
            grid_logits_sym.begin());
  std::copy(infer_result.move_probs.begin(),
            infer_result.move_probs.begin() + constants::kNumBoardLocs,
            grid_probs_sym.begin());
  std::copy(infer_result.opt_move_probs.begin(),
            infer_result.opt_move_probs.begin() + constants::kNumBoardLocs,
            grid_probs_opt_sym.begin());
  std::array<float, constants::kNumBoardLocs> grid_logits =
      ApplyInverse(sym, grid_logits_sym, BOARD_LEN);
  std::array<float, constants::kNumBoardLocs> grid_probs =
      ApplyInverse(sym, grid_probs_sym, BOARD_LEN);
  std::array<float, constants::kNumBoardLocs> grid_probs_opt =
      ApplyInverse(sym, grid_probs_opt_sym, BOARD_LEN);
  std::copy(grid_logits.begin(), grid_logits.end(),
            infer_result.move_logits.begin());
  std::copy(grid_probs.begin(), grid_probs.end(),
            infer_result.move_probs.begin());
  std::copy(grid_probs_opt.begin(), grid_probs_opt.end(),
            infer_result.opt_move_probs.begin());
}

}  // namespace nn
