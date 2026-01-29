#pragma once

#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "cc/game/color.h"
#include "cc/game/zobrist.h"
#include "cc/mcts/tree.h"

namespace mcts {

// Abstract base class for node allocation strategies
class NodeTable {
 public:
  virtual ~NodeTable() = default;

  // Get existing node or create a new one
  // color_to_move: The player whose turn it is at this position
  virtual TreeNode* GetOrCreate(game::Zobrist::Hash board_hash,
                                game::Color color_to_move,
                                bool is_terminal) = 0;

  // Reap all nodes not reachable from new_root
  virtual uint32_t Reap(TreeNode* new_root) = 0;

  // Return number of nodes currently allocated
  virtual size_t Size() const = 0;

  // Return whether this is a graph
  virtual bool is_graph() const = 0;
};

// For traditional tree search - always creates fresh nodes, never reuses
class MctsNodeTable final : public NodeTable {
 public:
  MctsNodeTable() = default;
  ~MctsNodeTable() override = default;

  MctsNodeTable(const MctsNodeTable&) = delete;
  MctsNodeTable& operator=(const MctsNodeTable&) = delete;

  // Always creates a new node (color_to_move is ignored, just for interface compat)
  inline TreeNode* GetOrCreate(game::Zobrist::Hash board_hash,
                               game::Color color_to_move,
                               bool is_terminal) override {
    TreeNode* node = new TreeNode(board_hash);
    nodes_.insert(std::unique_ptr<TreeNode>(node));
    return node;
  }

  // Reaps all nodes not reachable from new_root
  uint32_t Reap(TreeNode* new_root) override;

  size_t Size() const override { return nodes_.size(); }

  bool is_graph() const override { return false; }

 private:
  absl::flat_hash_set<std::unique_ptr<TreeNode>> nodes_;
};

// For graph search - reuses nodes with the same (hash, color) pair
class McgsNodeTable final : public NodeTable {
 public:
  McgsNodeTable() = default;
  ~McgsNodeTable() override = default;

  McgsNodeTable(const McgsNodeTable&) = delete;
  McgsNodeTable& operator=(const McgsNodeTable&) = delete;

  // Looks up by (hash, color_to_move), creates only if not found
  // This allows same board position with different player-to-move to have
  // distinct nodes (e.g., after a pass move)
  inline TreeNode* GetOrCreate(game::Zobrist::Hash board_hash,
                               game::Color color_to_move,
                               bool is_terminal) override {
    auto key = std::make_tuple(board_hash, color_to_move, is_terminal);
    auto it = table_.find(key);
    if (it != table_.end()) {
      return it->second.get();
    }

    TreeNode* node = new TreeNode(board_hash);
    table_[key] = std::unique_ptr<TreeNode>(node);
    return node;
  }

  // Reaps all nodes not reachable from new_root
  uint32_t Reap(TreeNode* new_root) override;

  size_t Size() const override { return table_.size(); }

  bool is_graph() const override { return true; }

 private:
  using NodeKey = std::tuple<game::Zobrist::Hash, game::Color, bool>;
  absl::flat_hash_map<NodeKey, std::unique_ptr<TreeNode>> table_;
};

}  // namespace mcts
