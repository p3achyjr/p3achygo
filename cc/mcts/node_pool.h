#pragma once

#include "absl/container/flat_hash_map.h"
#include "cc/game/zobrist.h"
#include "cc/mcts/tree.h"

namespace mcts {

class NodePool final {
 public:
  NodePool() = default;
  ~NodePool() = default;

  NodePool(const NodePool&) = delete;

  inline TreeNode* New(game::Zobrist::Hash board_hash) {
    TreeNode* node = new TreeNode(board_hash);
    node_pool_[board_hash] = std::unique_ptr<TreeNode>(node);
    return node;
  }

  inline TreeNode* LookupOrNew(game::Zobrist::Hash board_hash) {
    if (node_pool_.contains(board_hash)) {
      return node_pool_[board_hash].get();
    }

    return New(board_hash);
  }

  // returns number of reaped nodes
  uint32_t Reap(TreeNode* new_root);

 private:
  absl::flat_hash_map<game::Zobrist::Hash, std::unique_ptr<TreeNode>>
      node_pool_;
};

}  // namespace mcts
