#include "cc/game/board.h"

#include <memory>
#include <sstream>
#include <stack>

#include "absl/debugging/stacktrace.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/game/zobrist_hash.h"

namespace game {
namespace {

static constexpr groupid kInvalidGroupId = -1;
static constexpr int kInvalidLiberties = -1;

inline unsigned ZobristState(int state) { return state + 1; }

inline std::vector<Loc> Adjacent(Loc loc, int board_length) {
  std::vector<Loc> adjacent_points;

  if (loc.i > 0) {
    adjacent_points.emplace_back(Loc{loc.i - 1, loc.j});
  }

  if (loc.j > 0) {
    adjacent_points.emplace_back(Loc{loc.i, loc.j - 1});
  }

  if (loc.i < board_length - 1) {
    adjacent_points.emplace_back(Loc{loc.i + 1, loc.j});
  }

  if (loc.j < board_length - 1) {
    adjacent_points.emplace_back(Loc{loc.i, loc.j + 1});
  }

  return adjacent_points;
}

template <typename T>
inline bool VecContains(const std::vector<T> vec, T x) {
  return std::find(vec.begin(), vec.end(), x) != vec.end();
}

template <typename T>
inline std::string VecToString(const std::vector<T> vec) {
  std::stringstream ss;
  ss << "<";
  for (auto& x : vec) {
    ss << x << ", ";
  }
  ss << ">";

  // convert the stream buffer into a string
  return ss.str();
}

}  // namespace

GroupTracker::GroupTracker(int length) : length_(length) {
  for (auto i = 0; i < length_; i++) {
    for (auto j = 0; j < length_; j++) {
      groups_[i][j] = kInvalidGroupId;
    }
  }
}

groupid GroupTracker::GroupAt(Loc loc) const { return groups_[loc.i][loc.j]; }

groupid GroupTracker::NewGroup(Loc loc, int color) {
  // precondition: loc is not connected to any other group.
  groupid group_id = next_group_id_;
  int liberties = 0;
  std::vector<groupid> seen_groups;  // can only subtract up to 1 liberty from
                                     // each adjacent group
  for (const Loc& adj_loc : Adjacent(loc, length_)) {
    if (GroupAt(adj_loc) == kInvalidGroupId) {
      liberties++;
    } else if (group_info_map_[GroupAt(adj_loc)].color ==
               OppositeColor(color)) {
      if (VecContains(seen_groups, GroupAt(adj_loc))) continue;
      seen_groups.emplace_back(GroupAt(adj_loc));
      group_info_map_[GroupAt(adj_loc)].liberties--;
    }
  }

  DCHECK(group_info_map_.size() == group_id);
  SetLoc(loc, group_id);
  group_info_map_.emplace_back(GroupInfo{liberties, loc, color, true});
  next_group_id_++;

  return group_id;
}

void GroupTracker::AddToGroup(Loc loc, groupid id) {
  // precondition: loc is strongly connected to groupid `id`.
  std::vector<groupid> adjacent_groups;
  for (const Loc& adj_loc : Adjacent(loc, length_)) {
    if (GroupAt(adj_loc) == kInvalidGroupId) {
      // liberty of stone, and thereby liberty of group it is now connected to.
      bool is_counted = false;
      for (const Loc& adj_loc : Adjacent(adj_loc, length_)) {
        if (adj_loc == loc) {
          continue;
        } else if (GroupAt(adj_loc) == id) {
          // already seen
          is_counted = true;
        }
      }

      if (!is_counted) group_info_map_[id].liberties++;
    } else if (!VecContains(adjacent_groups, GroupAt(adj_loc))) {
      // neighboring group, need to subtract liberty to account for stone.
      adjacent_groups.emplace_back(GroupAt(adj_loc));
    }
  }

  for (const groupid& id : adjacent_groups) {
    group_info_map_[id].liberties--;
  }

  SetLoc(loc, id);
}

int GroupTracker::LibertiesAt(Loc loc) const {
  int l = 0;
  for (const Loc& next_loc : Adjacent(loc, length_)) {
    if (GroupAt(next_loc) == kInvalidGroupId) {
      l++;
    }
  }

  return l;
}

int GroupTracker::LibertiesForGroup(groupid group_id) const {
  if (group_id < 0 || group_id >= group_info_map_.size()) {
    return kInvalidLiberties;
  }

  return group_info_map_[group_id].liberties;
}

int GroupTracker::LibertiesForGroupAt(Loc loc) const {
  return LibertiesForGroup(GroupAt(loc));
}

std::vector<Loc> GroupTracker::ExpandGroup(groupid group_id) const {
  std::vector<Loc> group;

  std::vector<Loc> dfs_stack{group_info_map_[group_id].root};
  while (!dfs_stack.empty()) {
    Loc loc = dfs_stack.back();
    dfs_stack.pop_back();

    if (VecContains(group, loc)) {
      continue;
    }

    DCHECK(GroupAt(loc) == group_id);

    group.emplace_back(loc);
    for (const Loc& loc : Adjacent(loc, length_)) {
      if (GroupAt(loc) == group_id) {
        dfs_stack.emplace_back(loc);
      }
    }
  }

  return group;
}

void GroupTracker::RemoveCaptures(const std::vector<Loc>& captures) {
  for (const Loc& loc : captures) {
    group_info_map_[GroupAt(loc)].is_valid = false;

    std::vector<groupid> adj_groups;
    for (const Loc& adj_loc : Adjacent(loc, length_)) {
      if (GroupAt(adj_loc) == kInvalidGroupId) {
        continue;
      } else if (VecContains(adj_groups, GroupAt(adj_loc))) {
        continue;
      }

      adj_groups.emplace_back(GroupAt(adj_loc));
    }

    // for this new liberty, add to each group once.
    for (const groupid& id : adj_groups) {
      group_info_map_[id].liberties++;
    }

    SetLoc(loc, kInvalidGroupId);
  }
}

groupid GroupTracker::CoalesceGroups(const std::vector<groupid>& groups) {
  if (groups.empty()) {
    // nothing to do.
    return kInvalidGroupId;
  }

  groupid canonical_group_id = groups[0];
  int liberties = LibertiesForGroup(canonical_group_id);
  for (auto i = 1; i < groups.size(); i++) {
    groupid group_id = groups[i];
    Loc group_root = group_info_map_[groups[i]].root;

    // Traverse group, update group ids, and add liberties, avoiding liberties
    // that have already been counted.
    // A liberty has already been counted if it is adjacent to a stone that is
    // equal to the canonical group id. This invariant holds because we are
    // continually adding each stone to the canonical group after processing it.
    std::vector<Loc> dfs_stack{group_root};
    std::vector<Loc> seen;
    while (!dfs_stack.empty()) {
      // invariant: group_id[next_loc] = group_id
      Loc next_loc = dfs_stack.back();

      dfs_stack.pop_back();
      if (VecContains(seen, next_loc)) {
        continue;
      }

      DCHECK(GroupAt(next_loc) == group_id);
      for (const Loc& adj_loc : Adjacent(next_loc, length_)) {
        groupid adj_group_id = GroupAt(adj_loc);
        if (adj_group_id == group_id) {
          // part of contiguous group
          dfs_stack.emplace_back(adj_loc);
        } else if (adj_group_id == kInvalidGroupId) {
          // is a liberty. count this liberty if it has not already been counted
          // (i.e. it is not adjacent to a member of the) canonical group.
          bool is_counted = false;
          for (const Loc& adj_to_lib : Adjacent(adj_loc, length_)) {
            if (GroupAt(adj_to_lib) == canonical_group_id) {
              is_counted = true;
              break;
            }
          }

          if (!is_counted) liberties++;
        }
      }

      SetLoc(next_loc, canonical_group_id);
      seen.emplace_back(next_loc);
    }
  }

  // mark all absorbed groups as invalid.
  for (auto& groupid : groups) {
    if (groupid == canonical_group_id) continue;
    group_info_map_[groupid].is_valid = false;
  }

  group_info_map_[canonical_group_id].liberties = liberties;
  return canonical_group_id;
}

void GroupTracker::SetLoc(Loc loc, groupid id) { groups_[loc.i][loc.j] = id; }

Board::Board(ZobristTable* const zobrist_table)
    : Board::Board(zobrist_table, BOARD_LEN) {}

Board::Board(ZobristTable* const zobrist_table, int length)
    : zobrist_table_(zobrist_table), length_(length), group_tracker_(length) {
  HashKey hash = 0;
  for (auto i = 0; i < length_; i++) {
    for (auto j = 0; j < length_; j++) {
      hash ^= zobrist_table->hash_at(i, j, ZobristState(EMPTY));
    }
  }

  initial_hash_ = hash;
  hash_ = hash;
  seen_states_.insert(hash_);
}

int Board::length() const { return length_; }

int Board::at(int i, int j) const { return AtLoc(Loc{i, j}); }

HashKey Board::hash() const { return hash_; }

bool Board::IsValidMove(Loc loc, int color) const {
  return MoveDry(loc, color).has_value();
}

bool Board::IsGameOver() const { return pass_count_ == 2; }

int Board::move_count() const { return move_count_; }

bool Board::MoveBlack(int i, int j) { return Move(Loc{i, j}, BLACK); }

bool Board::MoveWhite(int i, int j) { return Move(Loc{i, j}, WHITE); }

bool Board::Move(Loc loc, int color) {
  if (loc == kPassLoc) {
    return MovePass(color);
  }

  std::optional<MoveInfo> move_info = MoveDry(loc, color);
  if (!move_info.has_value()) {
    return false;
  }

  // reaching here means the move is valid. Mutable portion begins here.

  // remove captures.
  std::vector<Loc> captured_stones;
  for (const Transition& transition : move_info->capture_transitions) {
    captured_stones.emplace_back(transition.loc);
  }

  group_tracker_.RemoveCaptures(captured_stones);
  for (const Loc& loc : captured_stones) {
    SetLoc(loc, EMPTY);
  }

  std::vector<groupid> adjacent_groups;
  for (const Loc& adj_loc : AdjacentOfColor(loc, color)) {
    if (!VecContains(adjacent_groups, group_tracker_.GroupAt(adj_loc))) {
      adjacent_groups.emplace_back(group_tracker_.GroupAt(adj_loc));
    }
  }

  // play stone.
  SetLoc(loc, color);
  if (adjacent_groups.empty()) {
    // is a single stone. create new group.
    group_tracker_.NewGroup(loc, color);
  } else {
    // is part of another group. add to an arbitrary group and coalesce all
    // groups.
    group_tracker_.AddToGroup(loc, adjacent_groups[0]);
    group_tracker_.CoalesceGroups(adjacent_groups);
  }

  // update tracking variables
  pass_count_ = 0;
  move_count_++;

  hash_ = move_info->new_hash;
  seen_states_.insert(hash_);

  return true;
}

bool Board::MovePass(int color) {
  pass_count_++;
  total_pass_count_++;

  return true;
}

std::optional<MoveInfo> Board::MoveDry(Loc loc, int color) const {
  if (color != BLACK && color != WHITE) {
    DLOG_EVERY_N_SEC(INFO, 5) << "Unknown Color: " << color;
    return std::nullopt;
  } else if (loc.i < 0 || loc.i >= length_ || loc.j < 0 || loc.j >= length_) {
    DLOG_EVERY_N_SEC(INFO, 5)
        << "Out of Bounds. i: " << loc.i << " j: " << loc.j;
    return std::nullopt;
  } else if (AtLoc(loc) != EMPTY) {
    DLOG_EVERY_N_SEC(INFO, 5) << "Board Position Not Empty: " << loc;
    return std::nullopt;
  }

  // check for captures.
  std::vector<groupid> captured_groups =
      GetCapturedGroups(loc, OppositeColor(color));

  // if no captures, check for self-atari.
  if (captured_groups.size() == 0 && IsSelfCapture(loc, color)) {
    DLOG_EVERY_N_SEC(INFO, 5)
        << "Played Self Capture Move at " << loc << " for Color " << color;
    return std::nullopt;
  }

  // reaching here means the move is playable.
  Transition move_transition = Transition{loc, AtLoc(loc), color};
  std::vector<Transition> capture_transitions;

  // resolve captures.
  std::vector<Loc> captured_stones;
  for (const groupid& captured_id : captured_groups) {
    std::vector<Loc> captures = group_tracker_.ExpandGroup(captured_id);
    captured_stones.insert(captured_stones.end(), captures.begin(),
                           captures.end());
  }

  for (const Loc& loc : captured_stones) {
    capture_transitions.emplace_back(Transition{loc, AtLoc(loc), EMPTY});
  }

  // check if we have already seen this board position.
  HashKey hash = RecomputeHash(move_transition, capture_transitions);
  if (seen_states_.contains(hash)) {
    DLOG_EVERY_N_SEC(INFO, 5) << "Already seen this board state. Move: " << loc
                              << " Color: " << color;
    return std::nullopt;
  }

  return MoveInfo{move_transition, capture_transitions, hash};
}

Loc Board::MoveAsLoc(int move) const {
  return Loc{move / length_, move % length_};
}

int Board::LocAsMove(Loc loc) const { return loc.i * length_ + loc.j; }

float Board::BlackScore() const { return Score(BLACK); }

float Board::WhiteScore() const { return Score(WHITE); }

float Board::Score(int color) const {
  bool seen[BOARD_LEN][BOARD_LEN]{};
  int score = 0;
  for (auto i = 0; i < length_; ++i) {
    for (auto j = 0; j < length_; ++j) {
      if (seen[i][j]) {
        continue;
      } else if (at(i, j) == color) {
        score++;
        continue;
      } else if (at(i, j) == OppositeColor(color)) {
        continue;
      }

      // empty location, dfs to find strongly contiguous region.
      std::vector<Loc> dfs_stack{Loc{i, j}};
      int contiguous_region_size = 0;
      while (!dfs_stack.empty()) {
        Loc loc = dfs_stack.back();
        dfs_stack.pop_back();
        if (seen[loc.i][loc.j]) {
          continue;
        } else if (AtLoc(loc) == OppositeColor(color)) {
          contiguous_region_size = 0;
          break;
        } else if (AtLoc(loc) == color) {
          continue;
        }

        // empty location
        seen[loc.i][loc.j] = true;
        contiguous_region_size++;

        auto adjacent = Adjacent(loc, length_);
        dfs_stack.insert(dfs_stack.end(), adjacent.begin(), adjacent.end());
      }

      score += contiguous_region_size;
    }
  }

  return static_cast<float>(score) + (color == WHITE ? komi_ : 0);
}

int Board::AtLoc(Loc loc) const { return board_[loc.i][loc.j]; }

void Board::SetLoc(Loc loc, int color) { board_[loc.i][loc.j] = color; }

bool Board::IsSelfCapture(Loc loc, int color) const {
  bool adjacent_in_atari = true;
  for (Loc& loc : AdjacentOfColor(loc, color)) {
    if (!IsInAtari(loc)) {
      adjacent_in_atari = false;
      break;
    }
  }

  if (!adjacent_in_atari) {
    return false;
  }

  return group_tracker_.LibertiesAt(loc) == 0;
}

bool Board::IsInAtari(Loc loc) const {
  return group_tracker_.LibertiesForGroupAt(loc) == 1;
}

std::vector<groupid> Board::GetCapturedGroups(Loc loc,
                                              int captured_color) const {
  std::vector<groupid> captured_groups;

  for (Loc& next_loc : Adjacent(loc, length_)) {
    groupid id = group_tracker_.GroupAt(next_loc);
    if (AtLoc(next_loc) == captured_color && IsInAtari(next_loc) &&
        !VecContains(captured_groups, id)) {
      // this group is in atari, and we have filled the last liberty.
      captured_groups.emplace_back(group_tracker_.GroupAt(next_loc));
    }
  }

  return captured_groups;
}

std::vector<Loc> Board::AdjacentOfColor(Loc loc, int color) const {
  std::vector<Loc> adjacent_points;

  Loc left = Loc{loc.i - 1, loc.j};
  if (loc.i > 0 && AtLoc(left) == color) {
    adjacent_points.emplace_back(left);
  }

  Loc top = Loc{loc.i, loc.j - 1};
  if (loc.j > 0 && AtLoc(top) == color) {
    adjacent_points.emplace_back(top);
  }

  Loc right = Loc{loc.i + 1, loc.j};
  if (loc.i < length_ - 1 && AtLoc(right) == color) {
    adjacent_points.emplace_back(right);
  }

  Loc bottom = Loc{loc.i, loc.j + 1};
  if (loc.j < length_ - 1 && AtLoc(bottom) == color) {
    adjacent_points.emplace_back(bottom);
  }

  return adjacent_points;
}

HashKey Board::RecomputeHash(
    const Transition& move_transition,
    const std::vector<Transition>& capture_transitions) const {
  HashKey hash = hash_;
  hash ^= zobrist_table_->hash_at(move_transition.loc.i, move_transition.loc.j,
                                  ZobristState(move_transition.last_piece));
  hash ^= zobrist_table_->hash_at(move_transition.loc.i, move_transition.loc.j,
                                  ZobristState(move_transition.current_piece));
  for (auto& capture_transition : capture_transitions) {
    // one hash to "undo" presence of last piece, one to add new piece.
    hash ^= zobrist_table_->hash_at(
        capture_transition.loc.i, capture_transition.loc.j,
        ZobristState(capture_transition.last_piece));
    hash ^= zobrist_table_->hash_at(
        capture_transition.loc.i, capture_transition.loc.j,
        ZobristState(capture_transition.current_piece));
  }

  return hash;
}
}  // namespace game
