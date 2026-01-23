#include "cc/mcts/node_pool.h"

#include "absl/container/flat_hash_set.h"

namespace mcts {

using namespace ::game;

uint32_t NodePool::Reap(TreeNode* new_root) {
  if (new_root == nullptr) {
    return 0;
  }

  absl::flat_hash_set<Zobrist::Hash> reachable;
  std::vector<TreeNode*> worklist = {new_root};
  while (!worklist.empty()) {
    TreeNode* next = worklist.back();
    worklist.pop_back();
    if (reachable.contains(next->board_hash)) {
      continue;
    }

    reachable.insert(next->board_hash);
    for (auto child : next->children) {
      if (child == nullptr) {
        continue;
      }

      worklist.push_back(child);
    }
  }

  return absl::erase_if(node_pool_, [&](const auto& entry) {
    const auto& [board_hash, node] = entry;
    return !reachable.contains(board_hash);
  });
}

}  // namespace mcts
