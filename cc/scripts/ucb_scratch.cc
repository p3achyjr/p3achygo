#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/core/rand.h"
#include "cc/game/board.h"

ABSL_FLAG(int, num_iterations, 1000, "Number of MCTS iterations per move");

/*
 * Scratch/scripts for MCTS.
 */
namespace mcts {

using ::game::Loc;

static constexpr int kPassMove = 361;
static constexpr float kExploration = 1.0;
static constexpr int kMaxMoveBucketSize = 512;

// Janky segtree-ish implementation.
class MoveBucket final {
 public:
  MoveBucket(const std::vector<int>& elems);
  ~MoveBucket() = default;

  int at(int i);
  int size();
  void RemoveAt(int i);
  void Insert(int x);

 private:
  struct Node {
    int subtree_size;
    int value;
  };

  int size_;
  Node arr_[2 * kMaxMoveBucketSize];
};

struct DfsNode {
  int i;
  int arr_index;
  int subtree_size;
  int pow2_split;
};

MoveBucket::MoveBucket(const std::vector<int>& elems) {
  size_ = elems.size();
  std::vector<DfsNode> dfs_stack{DfsNode{1, 0, size_, kMaxMoveBucketSize / 2}};
  int max_k = 0;
  while (!dfs_stack.empty()) {
    DfsNode dfs_node = dfs_stack.back();
    dfs_stack.pop_back();
    max_k = dfs_node.i > max_k ? dfs_node.i : max_k;

    if (dfs_node.i >= 2 * kMaxMoveBucketSize) {
      continue;
    } else if (dfs_node.i >= kMaxMoveBucketSize) {
      arr_[dfs_node.i] = Node{1, elems[dfs_node.arr_index]};
      continue;
    }

    // internal node
    arr_[dfs_node.i] = Node{dfs_node.subtree_size, 0};
    int left_size = std::min(dfs_node.pow2_split, dfs_node.subtree_size);
    int right_size = std::min(dfs_node.pow2_split,
                              dfs_node.subtree_size - dfs_node.pow2_split);

    if (left_size > 0) {
      dfs_stack.emplace_back(DfsNode{2 * dfs_node.i, dfs_node.arr_index,
                                     left_size, dfs_node.pow2_split / 2});
    }

    if (right_size > 0) {
      dfs_stack.emplace_back(DfsNode{
          2 * dfs_node.i + 1, static_cast<int>(dfs_node.arr_index + left_size),
          right_size, dfs_node.pow2_split / 2});
    }
  }
}

int MoveBucket::at(int i) {
  DCHECK(i < size_);
  int k = 1;
  while (k < kMaxMoveBucketSize) {
    size_t left_size = arr_[2 * k].subtree_size;
    k = i < left_size ? 2 * k : 2 * k + 1;
    i = i < left_size ? i : i - left_size;
  }

  return arr_[k].value;
}

int MoveBucket::size() { return size_; }

void MoveBucket::RemoveAt(int i) {
  DCHECK(i < size_);
  int k = 1;
  while (k < kMaxMoveBucketSize) {
    size_t left_size = arr_[2 * k].subtree_size;
    arr_[k].subtree_size -= 1;
    k = i < left_size ? 2 * k : 2 * k + 1;
    i = i < left_size ? i : i - left_size;
  }

  // should now be at leaf node
  arr_[k].subtree_size -= 1;
  size_ -= 1;
}

void MoveBucket::Insert(int x) {
  // find first available slot
  int k = 1;
  int full_size = kMaxMoveBucketSize;
  while (full_size > 1) {
    size_t left_size = arr_[2 * k].subtree_size;
    arr_[k].subtree_size += 1;
    k = left_size < full_size / 2 ? 2 * k : 2 * k + 1;
    full_size /= 2;
  }

  // should now be at leaf node
  arr_[k].subtree_size += 1;
  arr_[k].value = x;
  size_ += 1;
}

struct RandomDevice {
  typedef uint32_t result_type;
  core::PRng prng_;
  uint32_t operator()() { return prng_.next(); }
  static constexpr uint32_t max() {
    return std::numeric_limits<uint32_t>::max();
  }
  static constexpr uint32_t min() {
    return std::numeric_limits<uint32_t>::min();
  }
};

struct GameState {
  game::Zobrist::Hash board_hash;
  int color_to_move;
};

bool operator==(const GameState& x, const GameState& y) {
  return x.board_hash == y.board_hash && x.color_to_move == y.color_to_move;
}

template <typename H>
H AbslHashValue(H h, const GameState& g) {
  return H::combine(std::move(h), g.board_hash, g.color_to_move);
}

struct UcbTuple {
  float priority;  // calculated via UCB
  int move;
};

bool CompareUcb(const UcbTuple& x, const UcbTuple& y) {
  return x.priority < y.priority;
}

float UcbPriority(float v_i, int N, int n_i) {
  return v_i + kExploration * std::sqrt(std::log(N) / n_i);
}

struct UcbState {
  int N;
  float W;
  float Q;

  std::vector<UcbTuple> actions_explored;
  std::vector<int> actions_unexplored;
};

using Tree = absl::flat_hash_map<GameState, UcbState>;

// returns whether `color_to_move` won the playout.
int UcbRandomPlayout(game::Board& board, Tree& tree, int color_to_move,
                     const std::vector<int>& valid_moves) {
  core::PRng prng;
  int color = color_to_move;
  MoveBucket move_bucket(valid_moves);
  while (!board.IsGameOver() && board.move_count() < 400) {
    int move_index = core::RandRange(prng, 0, move_bucket.size());
    int move = move_bucket.at(move_index);

    if (move == kPassMove) {
      // pass move
      board.MovePass(color);
      color = game::OppositeColor(color);
    } else {
      Loc loc = board.MoveAsLoc(move);
      std::optional<game::MoveInfo> move_info = board.MoveDry(loc, color);
      if (!move_info.has_value()) {
        continue;
      }

      board.Move(loc, color);
      color = game::OppositeColor(color);
      move_bucket.RemoveAt(move_index);
      for (auto& transition : move_info->capture_transitions) {
        move_bucket.Insert(board.LocAsMove(transition.loc));
      }
    }
  }

  int b_score = board.BlackScore();
  int w_score = board.WhiteScore();

  return b_score > w_score ? BLACK : WHITE;
}

int UcbSearch(game::Board& board, Tree& tree, int color_to_move) {
  GameState game_state = GameState{board.hash(), color_to_move};
  if (board.IsGameOver()) {
    int b_score = board.BlackScore();
    int w_score = board.WhiteScore();
    int winner = b_score > w_score ? BLACK : WHITE;

    if (!tree.contains(game_state)) {
      UcbState state =
          UcbState{0, 0, 0.0, std::vector<UcbTuple>(), std::vector<int>()};

      state.N = 1;
      state.W = winner == color_to_move ? 1.0 : -1.0;
      state.Q = state.W;

      tree[std::move(game_state)] = std::move(state);
    } else {
      UcbState& state = tree.at(game_state);
      state.N++;
      state.W += (winner == color_to_move ? 1.0 : -1.0);
      state.Q = state.W / state.N;
    }

    return winner;
  }

  if (!tree.contains(game_state)) {
    // leaf node
    std::vector<int> valid_moves;
    for (auto i = 0; i < board.length(); ++i) {
      for (auto j = 0; j < board.length(); ++j) {
        if (board.IsValidMove(Loc{i, j}, color_to_move)) {
          valid_moves.emplace_back(i * board.length() + j);
        }
      }
    }

    valid_moves.emplace_back(kPassMove);

    int winner = UcbRandomPlayout(board, tree, color_to_move, valid_moves);

    std::shuffle(valid_moves.begin(), valid_moves.end(), RandomDevice());

    UcbState state = UcbState{0, 0, 0.0, std::vector<UcbTuple>(), valid_moves};
    state.N = 1;
    state.W = winner == color_to_move ? 1.0 : -1.0;
    state.Q = state.W;

    tree[std::move(game_state)] = std::move(state);

    return winner;
  } else if (!tree.at(game_state).actions_unexplored.empty()) {
    // boundary of tree. Select random node to expand.
    int move_to_expand = tree.at(game_state).actions_unexplored.back();

    if (move_to_expand == kPassMove) {
      board.MovePass(color_to_move);
    } else {
      Loc move_loc = board.MoveAsLoc(move_to_expand);
      board.Move(move_loc, color_to_move);
    }

    GameState child_game_state =
        GameState{board.hash(), game::OppositeColor(color_to_move)};
    int winner = UcbSearch(board, tree, game::OppositeColor(color_to_move));
    UcbState& state = tree.at(game_state);
    state.N++;
    state.W += (winner == color_to_move ? 1.0 : -1.0);
    state.Q = state.W / state.N;

    const UcbState& child_state = tree.at(child_game_state);
    state.actions_unexplored.pop_back();
    float priority = UcbPriority(child_state.Q, state.N, child_state.N);
    state.actions_explored.push_back(UcbTuple{priority, move_to_expand});
    std::push_heap(state.actions_explored.begin(), state.actions_explored.end(),
                   CompareUcb);

    return winner;
  } else {
    // internal node
    DCHECK(tree.at(game_state).actions_unexplored.size() == 0);
    std::vector<UcbTuple>& actions_explored =
        tree.at(game_state).actions_explored;
    std::pop_heap(actions_explored.begin(), actions_explored.end(), CompareUcb);

    int move = actions_explored.back().move;
    actions_explored.pop_back();
    if (move == kPassMove) {
      board.MovePass(color_to_move);
    } else {
      Loc move_loc = board.MoveAsLoc(move);
      board.Move(move_loc, color_to_move);
    }

    GameState child_game_state =
        GameState{board.hash(), game::OppositeColor(color_to_move)};
    int winner = UcbSearch(board, tree, game::OppositeColor(color_to_move));
    UcbState& state = tree.at(game_state);
    state.N++;
    state.W += (winner == color_to_move ? 1.0 : -1.0);
    state.Q = state.W / state.N;

    const UcbState& child_state = tree.at(child_game_state);
    float priority = UcbPriority(child_state.Q, state.N, child_state.N);
    state.actions_explored.push_back(UcbTuple{priority, move});
    std::push_heap(state.actions_explored.begin(), state.actions_explored.end(),
                   CompareUcb);

    return winner;
  }
}

Loc Ucb(game::Board root_board, Tree& tree, int color_to_move,
        int num_iterations) {
  for (auto i = 0; i < num_iterations; ++i) {
    game::Board board = root_board;  // make a copy
    UcbSearch(board, tree, color_to_move);
  }

  // pick node with highest value
  GameState game_state = GameState{root_board.hash(), color_to_move};
  DCHECK(tree.contains(game_state));
  UcbState& root_state = tree.at(game_state);
  DCHECK(root_state.actions_unexplored.empty());

  LOG(INFO) << "UcbState N:" << root_state.N << ", Q: " << root_state.Q
            << ", W: " << root_state.W
            << ", explored_size: " << root_state.actions_explored.size()
            << ", unexplored_size: " << root_state.actions_unexplored.size();

  std::vector<UcbTuple> actions = root_state.actions_explored;
  float max_value = -1;
  Loc max_move = {9, 9};
  for (auto& action : actions) {
    int move = action.move;
    GameState child_game_state;
    Loc move_loc = root_board.MoveAsLoc(move);
    if (move == kPassMove) {
      child_game_state =
          GameState{root_board.hash(), game::OppositeColor(color_to_move)};
    } else {
      std::optional<game::MoveInfo> move_info =
          root_board.MoveDry(move_loc, color_to_move);
      DCHECK(move_info.has_value());
      if (!move_info.has_value()) {
        continue;
      }

      child_game_state =
          GameState{move_info->new_hash, game::OppositeColor(color_to_move)};
    }

    DCHECK(tree.contains(child_game_state));
    UcbState& child_state = tree.at(child_game_state);

    if (child_state.Q > max_value) {
      max_value = child_state.Q;
      max_move = move_loc;
    }
  }

  return max_move;
}

}  // namespace mcts

int main(int argc, char** argv) {
  game::Zobrist zobrist_table;
  game::Board board(&zobrist_table);
  absl::flat_hash_map<mcts::GameState, mcts::UcbState> tree;

  int color = BLACK;
  int num_iterations = absl::GetFlag(FLAGS_num_iterations);
  while (!board.IsGameOver() && board.move_count() < 400) {
    game::Loc move_loc = mcts::Ucb(board, tree, color, num_iterations);
    board.Move(move_loc, color);
    color = game::OppositeColor(color);

    LOG(INFO) << "Move: " << board.move_count() << "\n";
    LOG(INFO) << "Board:\n" << board;
  }

  return 0;
}
