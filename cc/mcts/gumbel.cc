#include <cmath>

#include "cc/game/board.h"
#include "cc/mcts/tree.h"

using ::game::Board;
using ::game::Loc;

namespace mcts {
namespace {

int log2(int x) {
  // x needs to be > 0
  int i = 1;
  while (x >> i > 0) {
    i++;
  }

  return i - 1;
}

}  // namespace

// `n`: total number of simulations.
// `k`: initial number of actions selected.
// `n` must be >= `klogk`.
Loc GumbelSearchRoot(Board* board, TreeNode* node, int n, int k) {
  int num_rounds = log2(k);
  int visits_per_action = n / k * num_rounds;
  
}

}  // namespace mcts
