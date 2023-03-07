#ifndef __MCTS_TREE_H_
#define __MCTS_TREE_H_

#include <cstddef>

#include "cc/game/constants.h"

namespace mcts {

struct NodeStats {
  int N;
  float W;
  float Q;
};

struct TreeNode {
  static constexpr size_t kMaxNumActions = BOARD_LEN * BOARD_LEN + 1;

  TreeNode* children[kMaxNumActions]{};
  NodeStats node_stats[kMaxNumActions];
  float logits[kMaxNumActions];
};

}  // namespace mcts

#endif  // __MCTS_TREE_H_