#include "cc/mcts/node_table.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "cc/game/loc.h"

namespace mcts {

using namespace ::game;

uint32_t MctsNodeTable::Reap(TreeNode* new_root) {
  absl::MutexLock l(&mu_);
  if (new_root == nullptr) {
    return 0;
  }

  absl::flat_hash_set<TreeNode*> reachable;
  std::vector<TreeNode*> worklist = {new_root};

  while (!worklist.empty()) {
    TreeNode* node = worklist.back();
    worklist.pop_back();

    if (reachable.contains(node)) {
      continue;
    }

    reachable.insert(node);
    for (TreeNode* child : node->children) {
      if (child != nullptr) {
        worklist.push_back(child);
      }
    }
  }

  return absl::erase_if(nodes_, [&](const auto& node_ptr) {
    return !reachable.contains(node_ptr.get());
  });
}

uint32_t McgsNodeTable::Reap(TreeNode* new_root) {
  absl::MutexLock l(&mu_);
  if (new_root == nullptr) {
    return 0;
  }

  absl::flat_hash_set<TreeNode*> reachable;
  std::vector<TreeNode*> worklist = {new_root};

  while (!worklist.empty()) {
    TreeNode* node = worklist.back();
    worklist.pop_back();

    if (reachable.contains(node)) {
      continue;
    }

    reachable.insert(node);
    for (TreeNode* child : node->children) {
      if (child != nullptr) {
        worklist.push_back(child);
      }
    }
  }

  return absl::erase_if(table_, [&](const auto& entry) {
    return !reachable.contains(entry.second.get());
  });
}

}  // namespace mcts
