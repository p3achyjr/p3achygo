#ifndef __GAME_BOARD_H_
#define __GAME_BOARD_H_

#include <cstdint>
#include <iostream>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "cc/game/zobrist.h"

namespace nn {
class NNBoardUtils;
}

namespace game {

using groupid = int;
using color = int8_t;

/*
 * Represents cartesian index into grid.
 */
struct Loc {
  int i;
  int j;

  // Index into a 1D representation of a 2D grid of length `len`.
  int as_index(int len) { return i * len + j; }
};

/*
 * State transition for a single point on the grid.
 */
struct Transition {
  Loc loc;
  int last_piece;
  int current_piece;
};

/*
 * A struct that fully describes the state transitions for a board from a move.
 */
struct MoveInfo {
  Transition stone_transition;
  std::vector<Transition> capture_transitions;

  // internal
  Zobrist::Hash new_hash;
};

/*
 * Scores for both players.
 */
struct Scores {
  float black_score;
  float white_score;
};

template <typename H>
H AbslHashValue(H h, const Loc& loc) {
  return H::combine(std::move(h), loc.i, loc.j);
}

inline bool operator==(const Loc& x, const Loc& y) {
  return x.i == y.i && x.j == y.j;
}

inline std::ostream& operator<<(std::ostream& os, const Loc& loc) {
  return os << "Loc(" << loc.i << ", " << loc.j << ")";
}

inline std::ostream& operator<<(std::ostream& os,
                                const Transition& transition) {
  return os << transition.loc << ", last_piece: " << transition.last_piece
            << ", current_piece: " << transition.current_piece;
}

inline int OppositeColor(int color) { return -color; }

static constexpr Loc kNoopLoc = Loc{-1, -1};
static constexpr Loc kPassLoc = Loc{19, 0};

static constexpr groupid kInvalidGroupId = -1;
static constexpr int kInvalidLiberties = -1;

/*
 * Stack implementation for board coordinates.
 */
class LocStack final {
 public:
  LocStack() = default;
  ~LocStack() = default;

  bool Empty() const { return stack_.empty(); }

  void Push(Loc loc) { stack_.emplace_back(loc); }

  Loc Pop() {
    Loc loc = stack_.back();
    stack_.pop_back();

    return loc;
  }

 private:
  absl::InlinedVector<Loc, constants::kMaxNumBoardLocs> stack_;
};

/*
 * Light visitor class to aid dfs-traversal over board coordinates.
 *
 * Abstracts keeping track of seen coords and dfs stack.
 */
class LocVisitor final {
 public:
  LocVisitor(Loc root_loc) {
    stack_.Push(root_loc);
    seen_[root_loc.i][root_loc.j] = true;
  }
  ~LocVisitor() = default;

  bool Done() const { return stack_.Empty(); }

  Loc Next() {
    DCHECK(!stack_.Empty());
    return stack_.Pop();
  }

  void Visit(Loc loc) {
    if (!seen_[loc.i][loc.j]) {
      seen_[loc.i][loc.j] = true;
      stack_.Push(loc);
    }
  }

 private:
  LocStack stack_;
  bool seen_[BOARD_LEN][BOARD_LEN] = {};
};

/*
 * Tracks groups and liberties throughout game.
 *
 * Internal to `Board` class.
 */
class GroupTracker final {
 public:
  /*
   * Metadata for a single group.
   */
  struct GroupInfo {
    int liberties;
    Loc root;
    int color;
    bool is_valid;
  };

  /*
   * Benson Solver
   *
   * Uses Benson's Algorithm to calculate pass-alive regions.
   */
  class BensonSolver final {
   public:
    using regionid = int;
    struct BensonGroupInfo {
      groupid id;
      Loc root;

      int num_vital_regions = 0;
      absl::flat_hash_set<groupid> adj_regions;
    };

    struct BensonRegionInfo {
      regionid id;
      std::vector<Loc> locs;
      absl::flat_hash_set<groupid> vital_groups;
    };

    using GroupMap = absl::flat_hash_map<groupid, BensonGroupInfo>;
    using RegionMap = absl::flat_hash_map<regionid, BensonRegionInfo>;

    BensonSolver(GroupTracker* group_tracker);
    ~BensonSolver() = default;

    void CalculatePassAliveRegionForColor(int color);
    GroupMap GetGroupMap(int color);
    RegionMap GetRegionMap(int color);

    void PopulateAdjacentRegions(GroupMap& group_map, RegionMap& region_map);
    void PopulateVitalRegions(GroupMap& group_map, RegionMap& region_map);
    void RunBenson(GroupMap& group_map, RegionMap& region_map);

   private:
    GroupTracker* group_tracker_;
  };

  using ExpandedGroup = absl::InlinedVector<Loc, constants::kMaxNumBoardLocs>;
  GroupTracker(int length);
  ~GroupTracker() = default;

  int length() const;
  groupid GroupAt(Loc loc) const;
  groupid NewGroup(Loc loc, int color);
  void AddToGroup(Loc loc, groupid id);

  void Move(Loc loc, int color);
  int LibertiesAt(Loc loc) const;  // returns number of empty neighboring spots.
  int LibertiesForGroup(groupid group_id) const;
  int LibertiesForGroupAt(Loc loc) const;
  ExpandedGroup ExpandGroup(groupid group_id) const;
  void RemoveCaptures(const std::vector<Loc>& captures);

  // coalesces all groups adjacent to `loc` into a single group.
  groupid CoalesceGroups(Loc loc);

  // calculates pass-alive regions according to Benson's algorithm
  // https://senseis.xmp.net/?BensonsAlgorithm
  void CalculatePassAliveRegions();
  void CalculatePassAliveRegionForColor(int color);

  bool IsPassAlive(Loc loc) const;
  bool IsPassAliveForColor(Loc loc, int color) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const GroupTracker& group_tracker);

 private:
  void SetLoc(Loc loc, groupid id);
  groupid EmplaceGroup(GroupInfo group_info);
  void SetGroupInvalid(groupid id);
  bool LocIsEmpty(Loc loc);
  int ColorAt(Loc loc);

  int length_;
  std::array<groupid, BOARD_LEN * BOARD_LEN> groups_;
  std::array<color, BOARD_LEN * BOARD_LEN> pass_alive_;
  absl::InlinedVector<GroupInfo, BOARD_LEN * BOARD_LEN> group_info_map_;
  int next_group_id_ = 0;
  absl::InlinedVector<groupid, BOARD_LEN * BOARD_LEN> available_group_ids_;
};

/*
 * Interface for Go Board.
 */
class Board final {
 public:
  using BoardData = std::array<color, BOARD_LEN * BOARD_LEN>;
  Board();
  Board(int length);
  ~Board() = default;

  int length() const;
  int at(int i, int j) const;
  float komi() const;
  Zobrist::Hash hash() const;
  int move_count() const;

  bool IsValidMove(Loc loc, int color) const;
  bool IsGameOver() const;

  bool MoveBlack(int i, int j);
  bool MoveWhite(int i, int j);
  bool Move(Loc loc, int color);
  bool MovePass(int color);
  std::optional<MoveInfo> MoveDry(Loc loc, int color) const;
  Loc AsLoc(int move) const;
  int AsIndex(Loc loc) const;

  Scores GetScores();

  std::string ToString() const;

  friend std::ostream& operator<<(std::ostream& os, const Board& board);

 private:
  int AtLoc(Loc loc) const;
  void SetLoc(Loc loc, int color);

  bool IsSelfCapture(Loc loc, int color) const;
  bool IsInAtari(Loc loc) const;

  float Score(int color) const;

  std::vector<groupid> GetCapturedGroups(Loc loc, int captured_color) const;
  absl::InlinedVector<Loc, 4> AdjacentOfColor(Loc loc, int color) const;
  Zobrist::Hash RecomputeHash(
      const Transition& move_transition,
      const std::vector<Transition>& capture_transitions) const;

  const Zobrist& zobrist_;
  int length_;
  BoardData board_;
  int move_count_;
  int pass_count_;
  int total_pass_count_;
  float komi_;

  // Hash value of empty board. Keep to avoid recomputing for each new game.
  Zobrist::Hash initial_hash_;
  Zobrist::Hash hash_;
  GroupTracker group_tracker_;
  absl::flat_hash_set<Zobrist::Hash> seen_states_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const GroupTracker& group_tracker) {
  os << "----Groups----\n";
  for (auto i = 0; i < group_tracker.length_; i++) {
    for (auto j = 0; j < group_tracker.length_; j++) {
      if (group_tracker.GroupAt(Loc{i, j}) == kInvalidGroupId) {
        os << ". ";
        continue;
      }
      os << group_tracker.GroupAt(Loc{i, j}) << " ";
    }

    os << "\n";
  }

  os << "----Group Info----\n";
  for (auto i = 0; i < group_tracker.group_info_map_.size(); i++) {
    auto liberties = group_tracker.group_info_map_[i].liberties;
    auto root = group_tracker.group_info_map_[i].root;
    auto color = group_tracker.group_info_map_[i].color;
    auto is_valid = group_tracker.group_info_map_[i].is_valid;
    os << "Group " << i << ": (liberties: " << liberties << ", color: " << color
       << ", root: " << root << ", is_valid: " << is_valid << ")\n";
  }

  os << "----Pass Alive Regions----\n";
  for (auto i = 0; i < group_tracker.length_; i++) {
    for (auto j = 0; j < group_tracker.length_; j++) {
      if (group_tracker.IsPassAliveForColor(Loc{i, j}, BLACK)) {
        os << "o ";
      } else if (group_tracker.IsPassAliveForColor(Loc{i, j}, WHITE)) {
        os << "x ";
      } else {
        os << ". ";
      }
    }

    os << "\n";
  }

  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Board& board) {
  auto is_star_point = [](int i, int j) {
    return (i == 3 || i == 9 || i == 15) && (j == 3 || j == 9 || j == 15);
  };

  for (auto i = 0; i < board.length_; i++) {
    if (i < 10)
      os << i << "  ";
    else
      os << i << " ";
    for (auto j = 0; j < board.length_; j++) {
      if (board.at(i, j) == EMPTY && is_star_point(i, j)) {
        os << "+ ";
      } else if (board.at(i, j) == EMPTY) {
        os << "⋅ ";
      } else if (board.at(i, j) == BLACK) {
        os << "○ ";
      } else if (board.at(i, j) == WHITE) {
        os << "● ";
      }
    }

    os << "\n";
  }

  os << "   "
     << "A B C D E F G H I J K L M N O P Q R S";

  return os;
}

}  // namespace game

#endif  // __GAME_BOARD_H_
