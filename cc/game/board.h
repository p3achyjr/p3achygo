#ifndef __GAME_BOARD_H_
#define __GAME_BOARD_H_

#include <iostream>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "cc/core/cache.h"
#include "cc/core/util.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "cc/game/zobrist.h"

namespace nn {
class NNBoardUtils;
}

namespace game {

using groupid = int;
using LocVec = absl::InlinedVector<Loc, constants::kMaxNumBoardLocs>;
using BensonCache =
    core::Cache<Zobrist::Hash, std::array<Color, BOARD_LEN * BOARD_LEN>>;

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
  using Transitions =
      absl::InlinedVector<Transition, constants::kMaxNumBoardLocs>;

  Transition stone_transition;
  Transitions capture_transitions;

  // internal
  Zobrist::Hash new_hash;
};

/*
 * Status of attempted move.
 */
enum class MoveStatus : uint8_t {
  kValid,
  kUnknownColor,
  kOutOfBounds,
  kLocNotEmpty,
  kPassAliveRegion,
  kSelfCapture,
  kRepeatedPosition,
};

/*
 * A wrapper struct around `MoveInfo`.
 */
struct MoveResult {
  MoveStatus status;
  std::optional<MoveInfo> move_info;
};

/*
 * Scores and ownership for both players.
 */
struct Scores {
  float black_score;
  float white_score;
  std::array<Color, BOARD_LEN * BOARD_LEN> ownership;
};

inline std::ostream& operator<<(std::ostream& os,
                                const Transition& transition) {
  return os << transition.loc << ", last_piece: " << transition.last_piece
            << ", current_piece: " << transition.current_piece;
}

inline bool MoveOk(MoveStatus status) { return status == MoveStatus::kValid; }
inline bool MoveOk(MoveResult result) {
  return result.status == MoveStatus::kValid;
}

static constexpr int kInvalidMoveEncoding = -1;

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
  LocVec stack_;
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
    Color color;
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

    void CalculatePassAliveRegionForColor(Color color);
    GroupMap GetGroupMap(Color color);
    RegionMap GetRegionMap(Color color);

    void PopulateAdjacentRegions(GroupMap& group_map, RegionMap& region_map);
    void PopulateVitalRegions(GroupMap& group_map, RegionMap& region_map);
    void RunBenson(GroupMap& group_map, RegionMap& region_map);

   private:
    GroupTracker* group_tracker_;
  };

  GroupTracker();
  ~GroupTracker() = default;

  groupid GroupAt(Loc loc) const;
  groupid NewGroup(Loc loc, Color color);
  void AddToGroup(Loc loc, groupid id);

  void Move(Loc loc, Color color);
  int LibertiesAt(Loc loc) const;  // returns number of empty neighboring spots.
  int LibertiesForGroup(groupid gid) const;
  int LibertiesForGroupAt(Loc loc) const;
  LocVec ExpandGroup(groupid gid) const;
  void RemoveCaptures(const LocVec& captures);

  // coalesces all groups adjacent to `loc` into a single group.
  groupid CoalesceGroups(Loc loc);

  // calculates pass-alive regions according to Benson's algorithm
  // https://senseis.xmp.net/?BensonsAlgorithm
  void CalculatePassAliveRegions(Zobrist::Hash hash);
  void CalculatePassAliveRegionForColor(Color color);

  bool IsPassAlive(Loc loc) const;
  bool IsPassAliveForColor(Loc loc, Color color) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const GroupTracker& group_tracker);

 private:
  void SetLoc(Loc loc, groupid id);
  groupid EmplaceGroup(GroupInfo group_info);
  void SetGroupInvalid(groupid id);
  bool LocIsEmpty(Loc loc);
  int ColorAt(Loc loc);

  std::array<groupid, BOARD_LEN * BOARD_LEN> groups_;
  std::array<Color, BOARD_LEN * BOARD_LEN> pass_alive_;
  absl::InlinedVector<GroupInfo, BOARD_LEN * BOARD_LEN> group_info_map_;
  int next_group_id_ = 0;
  absl::InlinedVector<groupid, BOARD_LEN * BOARD_LEN> available_group_ids_;
};

/*
 * Interface for Go Board.
 */
class Board final {
 public:
  using BoardData = std::array<Color, BOARD_LEN * BOARD_LEN>;
  Board();
  ~Board() = default;

  int at(int i, int j) const;
  float komi() const;
  Zobrist::Hash hash() const;
  int move_count() const;
  const BoardData& position() const;

  bool IsValidMove(Loc loc, Color color) const;
  bool IsGameOver() const;

  MoveStatus PlayBlack(int i, int j);
  MoveStatus PlayWhite(int i, int j);
  MoveStatus PlayMove(Loc loc, Color color);
  MoveStatus Pass(Color color);
  MoveResult PlayMoveDry(Loc loc, Color color) const;

  void CalculatePassAliveRegions();

  Scores GetScores();

  std::string ToString() const;

  friend std::ostream& operator<<(std::ostream& os, const Board& board);

 private:
  int AtLoc(Loc loc) const;
  void SetLoc(Loc loc, Color color);

  bool IsSelfCapture(Loc loc, Color color) const;
  bool IsInAtari(Loc loc) const;

  std::pair<float, std::array<Color, BOARD_LEN * BOARD_LEN>> ScoreAndOwnership(
      Color color) const;

  absl::InlinedVector<groupid, 4> GetCapturedGroups(Loc loc,
                                                    int captured_color) const;
  absl::InlinedVector<Loc, 4> AdjacentOfColor(Loc loc, Color color) const;
  Zobrist::Hash RecomputeHash(
      const Transition& move_transition,
      const MoveInfo::Transitions& capture_transitions) const;

  const Zobrist& zobrist_;
  BoardData board_;
  int move_count_;
  int consecutive_passes_;
  int passes_;
  float komi_;

  Zobrist::Hash hash_;
  GroupTracker group_tracker_;
  absl::flat_hash_set<Zobrist::Hash> seen_states_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const GroupTracker& group_tracker) {
  auto width = [](groupid gid) {
    if (gid == kInvalidGroupId || gid == 0) {
      return 1;
    }

    int width = 0;
    while (gid > 0) {
      gid /= 10;
      ++width;
    }

    return width;
  };

  int col_widths[BOARD_LEN]{};
  for (auto j = 0; j < BOARD_LEN; ++j) {
    for (auto i = 0; i < BOARD_LEN; ++i) {
      auto gid = group_tracker.GroupAt(Loc{i, j});
      col_widths[j] = std::max(col_widths[j], width(gid));
    }
  }

  os << "----Groups----\n";
  for (auto i = 0; i < BOARD_LEN; i++) {
    for (auto j = 0; j < BOARD_LEN; j++) {
      auto gid = group_tracker.GroupAt(Loc{i, j});
      int padding = col_widths[j] - width(gid) + 1;
      if (gid == kInvalidGroupId) {
        os << ".";
      } else {
        os << gid;
      }

      os << std::string(padding, ' ');
    }

    os << "\n";
  }

  os << "----Group Info----\n";
  for (auto i = 0; i < group_tracker.group_info_map_.size(); i++) {
    auto liberties = group_tracker.group_info_map_[i].liberties;
    auto root = group_tracker.group_info_map_[i].root;
    auto color = group_tracker.group_info_map_[i].color;
    auto is_valid = group_tracker.group_info_map_[i].is_valid;
    os << "Group " << i << ": (liberties: " << liberties
       << ", color: " << static_cast<int>(color) << ", root: " << root
       << ", is_valid: " << is_valid << ")\n";
  }

  os << "----Pass Alive Regions----\n";
  for (auto i = 0; i < BOARD_LEN; i++) {
    for (auto j = 0; j < BOARD_LEN; j++) {
      if (group_tracker.IsPassAliveForColor(Loc{i, j}, BLACK)) {
        os << "○ ";
      } else if (group_tracker.IsPassAliveForColor(Loc{i, j}, WHITE)) {
        os << "● ";
      } else {
        os << "⋅ ";
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

  for (auto i = 0; i < BOARD_LEN; i++) {
    if (i < 10)
      os << i << "  ";
    else
      os << i << " ";
    for (auto j = 0; j < BOARD_LEN; j++) {
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
