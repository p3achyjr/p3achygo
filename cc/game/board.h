#ifndef __GAME_BOARD_H_
#define __GAME_BOARD_H_

#include <iostream>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "cc/constants/constants.h"
#include "cc/game/zobrist_hash.h"

namespace nn {
class NNBoardUtils;
}

namespace game {

using groupid = int;

struct Loc {
  int i;
  int j;
};

struct Transition {
  Loc loc;
  int last_piece;
  int current_piece;
};

struct MoveInfo {
  Transition stone_transition;
  std::vector<Transition> capture_transitions;

  // internal-only fields
  Zobrist::Hash new_hash;
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

/*
 * Tracks groups and liberties throughout game.
 *
 * Should be internal to `Board` class.
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

  GroupTracker(int length);
  ~GroupTracker() = default;

  int length() const;
  groupid GroupAt(Loc loc) const;
  groupid NewGroup(Loc loc, int color);
  void AddToGroup(Loc loc, groupid id);
  int LibertiesAt(Loc loc) const;  // returns number of empty neighboring spots.
  int LibertiesForGroup(groupid group_id) const;
  int LibertiesForGroupAt(Loc loc) const;
  std::vector<Loc> ExpandGroup(groupid group_id) const;
  void RemoveCaptures(const std::vector<Loc>& captures);

  // coalesces `groups` into a single group.
  // Precondition: all `groups` must be strongly-connected.
  groupid CoalesceGroups(const std::vector<groupid>& groups);

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
  groupid groups_[BOARD_LEN][BOARD_LEN];
  int pass_alive_[BOARD_LEN][BOARD_LEN] = {};
  std::vector<GroupInfo> group_info_map_;
  int next_group_id_ = 0;
  std::vector<groupid> available_group_ids_;
};

/*
 * Interface for Go Board.
 */
class Board final {
 public:
  Board(Zobrist* const zobrist_table);
  Board(Zobrist* const zobrist_table, int length);
  ~Board() = default;

  int length() const;
  int at(int i, int j) const;
  Zobrist::Hash hash() const;
  int move_count() const;

  bool IsValidMove(Loc loc, int color) const;
  bool IsGameOver() const;

  bool MoveBlack(int i, int j);
  bool MoveWhite(int i, int j);
  bool Move(Loc loc, int color);
  bool MovePass(int color);
  std::optional<MoveInfo> MoveDry(Loc loc, int color) const;
  Loc MoveAsLoc(int move) const;
  int LocAsMove(Loc loc) const;

  float BlackScore() const;
  float WhiteScore() const;
  float Score(int color) const;

  std::string ToString() const;

  friend std::ostream& operator<<(std::ostream& os, const Board& board);
  friend class ::nn::NNBoardUtils;

 private:
  int AtLoc(Loc loc) const;
  void SetLoc(Loc loc, int color);

  bool IsSelfCapture(Loc loc, int color) const;

  bool IsInAtari(Loc loc) const;

  absl::InlinedVector<Loc, 4> AdjacentOfColor(Loc loc, int color) const;

  std::vector<groupid> GetCapturedGroups(Loc loc, int captured_color) const;
  Zobrist::Hash RecomputeHash(
      const Transition& move_transition,
      const std::vector<Transition>& capture_transitions) const;

  Zobrist* const zobrist_table_;
  int length_;
  int board_[BOARD_LEN][BOARD_LEN] = {};
  int move_count_ = 0;
  int pass_count_ = 0;
  int total_pass_count_ = 0;
  float komi_ = 7.5;

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
      if (group_tracker.GroupAt(Loc{i, j}) == -1) {
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
      int pa_color = group_tracker.pass_alive_[i][j];
      if (pa_color == EMPTY) {
        os << ". ";
        continue;
      }

      os << pa_color << " ";
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
