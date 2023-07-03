#include "cc/game/board.h"

#include <memory>
#include <sstream>

#include "absl/log/check.h"
#include "cc/core/util.h"

namespace game {
namespace {

using namespace ::core;

using GroupMap = GroupTracker::BensonSolver::GroupMap;
using RegionMap = GroupTracker::BensonSolver::RegionMap;

// 2 ** 18 total size across program.
static constexpr size_t kBensonCacheSize = 262144 / constants::kMaxNumThreads;

// Array containing adjacent vertices for every coordinate.
static const std::array<absl::InlinedVector<Loc, 4>,
                        constants::kMaxNumBoardLocs>
    kAdjacentPoints = []() {
      std::array<absl::InlinedVector<Loc, 4>, constants::kMaxNumBoardLocs> adj;
      for (int idx = 0; idx < constants::kMaxNumBoardLocs; ++idx) {
        Loc loc = Loc{idx / BOARD_LEN, idx % BOARD_LEN};
        if (loc.i > 0) {
          adj[idx].emplace_back(Loc{loc.i - 1, loc.j});
        }

        if (loc.j > 0) {
          adj[idx].emplace_back(Loc{loc.i, loc.j - 1});
        }

        if (loc.i < BOARD_LEN - 1) {
          adj[idx].emplace_back(Loc{loc.i + 1, loc.j});
        }

        if (loc.j < BOARD_LEN - 1) {
          adj[idx].emplace_back(Loc{loc.i, loc.j + 1});
        }
      }

      return adj;
    }();

inline unsigned ZobristState(int state) { return state + 1; }

inline absl::InlinedVector<Loc, 4> Adjacent(Loc loc) {
  return kAdjacentPoints[loc];
}

}  // namespace

GroupTracker::GroupTracker() : pass_alive_(), next_group_id_(0) {
  groups_.fill(kInvalidGroupId);
}

groupid GroupTracker::NewGroup(Loc loc, Color color) {
  // precondition: loc is not connected to any other group.
  int liberties = 0;
  absl::InlinedVector<groupid, 4>
      seen_groups;  // can only subtract up to 1 liberty from
                    // each adjacent group
  for (const Loc& nloc : Adjacent(loc)) {
    if (GroupAt(nloc) == kInvalidGroupId) {
      liberties++;
    } else if (group_info_map_[GroupAt(nloc)].color == OppositeColor(color)) {
      if (InlinedVecContains(seen_groups, GroupAt(nloc))) continue;
      seen_groups.emplace_back(GroupAt(nloc));
      group_info_map_[GroupAt(nloc)].liberties--;
    }
  }

  groupid gid = EmplaceGroup(GroupInfo{liberties, loc, color, true});
  SetLoc(loc, gid);

  return gid;
}

void GroupTracker::AddToGroup(Loc loc, groupid gid) {
  // precondition: loc is strongly connected to groupid `id`.
  absl::InlinedVector<groupid, 4> adjacent_groups;
  for (const Loc& nloc : Adjacent(loc)) {
    if (GroupAt(nloc) == kInvalidGroupId) {
      // liberty of stone, and thereby liberty of group it is now connected to.
      bool is_counted = false;
      for (const Loc& nloc : Adjacent(nloc)) {
        if (nloc == loc) {
          continue;
        } else if (GroupAt(nloc) == gid) {
          // already seen
          is_counted = true;
        }
      }

      if (!is_counted) group_info_map_[gid].liberties++;
    } else if (!InlinedVecContains(adjacent_groups, GroupAt(nloc))) {
      // neighboring group, need to subtract liberty to account for stone.
      adjacent_groups.emplace_back(GroupAt(nloc));
    }
  }

  for (const groupid& ngid : adjacent_groups) {
    group_info_map_[ngid].liberties--;
  }

  SetLoc(loc, gid);
}

void GroupTracker::Move(Loc loc, Color color) {
  absl::InlinedVector<groupid, 4> adjacent_groups;
  for (const Loc& nloc : Adjacent(loc)) {
    if (ColorAt(nloc) != color) {
      continue;
    }

    if (!InlinedVecContains(adjacent_groups, GroupAt(nloc))) {
      adjacent_groups.emplace_back(GroupAt(nloc));
    }
  }

  if (adjacent_groups.empty()) {
    // is a single stone. create new group.
    NewGroup(loc, color);
  } else {
    // is part of another group. add to an arbitrary group and coalesce.
    AddToGroup(loc, adjacent_groups[0]);
    groupid canonical_gid = CoalesceGroups(loc);

    // mark dissolved groups as invalid.
    for (const auto gid : adjacent_groups) {
      if (gid != canonical_gid) {
        SetGroupInvalid(gid);
      }
    }
  }
}

int GroupTracker::LibertiesAt(Loc loc) const {
  int l = 0;
  for (const Loc& nloc : Adjacent(loc)) {
    if (GroupAt(nloc) == kInvalidGroupId) {
      l++;
    }
  }

  return l;
}

LocVec GroupTracker::ExpandGroup(groupid gid) const {
  DCHECK(group_info_map_[gid].is_valid);
  LocVec group;

  LocVisitor visitor(group_info_map_[gid].root);
  while (!visitor.Done()) {
    Loc loc = visitor.Next();
    group.emplace_back(loc);

    DCHECK(GroupAt(loc) == gid);
    for (const Loc& nloc : Adjacent(loc)) {
      if (GroupAt(nloc) == gid) {
        // visitor ensures each node is only visited once, so we do not need to
        // do any additional filtering.
        visitor.Visit(nloc);
      }
    }
  }

  return group;
}

void GroupTracker::RemoveCaptures(const LocVec& captures) {
  for (const Loc& loc : captures) {
    SetGroupInvalid(GroupAt(loc));

    absl::InlinedVector<groupid, 4> adj_groups;
    for (const Loc& nloc : Adjacent(loc)) {
      if (GroupAt(nloc) == kInvalidGroupId) {
        continue;
      } else if (InlinedVecContains(adj_groups, GroupAt(nloc))) {
        continue;
      }

      adj_groups.emplace_back(GroupAt(nloc));
    }

    // for this new liberty, add to each group once.
    for (const groupid& id : adj_groups) {
      group_info_map_[id].liberties++;
    }

    SetLoc(loc, kInvalidGroupId);
  }
}

groupid GroupTracker::CoalesceGroups(Loc loc) {
  groupid canonical_group_id = GroupAt(loc);
  Color color = ColorAt(loc);
  int liberties = 0;

  LocVisitor visitor(loc);
  while (!visitor.Done()) {
    loc = visitor.Next();
    if (ColorAt(loc) == EMPTY) {
      ++liberties;
    } else {
      DCHECK(ColorAt(loc) == color);

      groups_[loc] = canonical_group_id;
      for (const Loc& nloc : Adjacent(loc)) {
        // add all liberties and stones of color.
        // Visitor ensures that each liberty is only counted once.
        if (ColorAt(nloc) == OppositeColor(color)) {
          continue;
        }

        visitor.Visit(nloc);
      }
    }
  }

  group_info_map_[canonical_group_id].liberties = liberties;
  return canonical_group_id;
}

void GroupTracker::CalculatePassAliveRegions(Zobrist::Hash hash) {
  thread_local BensonCache benson_cache(kBensonCacheSize);
  if (benson_cache.Contains(hash)) {
    pass_alive_ = benson_cache.Get(hash).value();
  }

  CalculatePassAliveRegionForColor(BLACK);
  CalculatePassAliveRegionForColor(WHITE);
  benson_cache.Insert(hash, pass_alive_);
}

void GroupTracker::CalculatePassAliveRegionForColor(Color color) {
  BensonSolver benson_solver(this);
  benson_solver.CalculatePassAliveRegionForColor(color);
}

bool GroupTracker::IsPassAlive(Loc loc) const {
  return pass_alive_[loc] != EMPTY;
}

bool GroupTracker::IsPassAliveForColor(Loc loc, Color color) const {
  return pass_alive_[loc] == color;
}

void GroupTracker::SetLoc(Loc loc, groupid id) { groups_[loc] = id; }

groupid GroupTracker::EmplaceGroup(GroupInfo group_info) {
  groupid gid;
  if (!available_group_ids_.empty()) {
    gid = available_group_ids_.back();
    available_group_ids_.pop_back();

    group_info_map_[gid] = group_info;
  } else {
    gid = next_group_id_;
    next_group_id_++;

    group_info_map_.emplace_back(group_info);
  }

  return gid;
}

void GroupTracker::SetGroupInvalid(groupid id) {
  if (group_info_map_[id].is_valid) {
    group_info_map_[id].is_valid = false;
    available_group_ids_.emplace_back(id);
  }
}

GroupTracker::BensonSolver::BensonSolver(GroupTracker* group_tracker)
    : group_tracker_(group_tracker) {}

void GroupTracker::BensonSolver::CalculatePassAliveRegionForColor(Color color) {
  absl::flat_hash_map<groupid, BensonGroupInfo> group_map = GetGroupMap(color);
  absl::flat_hash_map<regionid, BensonRegionInfo> region_map =
      GetRegionMap(color);

  PopulateAdjacentRegions(group_map, region_map);
  PopulateVitalRegions(group_map, region_map);
  RunBenson(group_map, region_map);

  // group_map and region_map are now trimmed to the pass-alive region.
  for (auto& [gid, group_info] : group_map) {
    bool gid_valid = group_tracker_->group_info_map_[gid].is_valid;
    if (!gid_valid) {
      continue;
    }
    LocVec group = group_tracker_->ExpandGroup(gid);
    for (auto& loc : group) {
      group_tracker_->pass_alive_[loc] = color;
    }
  }

  for (auto& [rid, region_info] : region_map) {
    for (auto& loc : region_info.locs) {
      group_tracker_->pass_alive_[loc] = color;
    }
  }
}

GroupMap GroupTracker::BensonSolver::GetGroupMap(Color color) {
  absl::flat_hash_map<groupid, BensonGroupInfo> group_map;

  for (int gid = 0; gid < group_tracker_->group_info_map_.size(); ++gid) {
    const auto& group_info = group_tracker_->group_info_map_[gid];
    if (!group_info.is_valid) {
      continue;
    }

    if (group_info.color == color) {
      group_map[gid] = BensonGroupInfo{
          gid,
          group_info.root,
          0,
      };
    }
  }

  return group_map;
}

RegionMap GroupTracker::BensonSolver::GetRegionMap(Color color) {
  absl::flat_hash_map<regionid, BensonRegionInfo> region_map;
  // Find all small regions and populate them in `region_map`.
  int next_region_id = 1;

  bool seen[BOARD_LEN][BOARD_LEN] = {};
  for (int i = 0; i < BOARD_LEN; ++i) {
    for (int j = 0; j < BOARD_LEN; ++j) {
      if (seen[i][j]) {
        continue;
      }

      if (!group_tracker_->LocIsEmpty(Loc{i, j})) {
        if (group_tracker_->ColorAt(Loc{i, j}) == color) seen[i][j] = true;
        continue;
      }

      // Square is empty. Find if it is small.
      // A region is small iff all empty coords in the region are liberties of
      // an enclosing chain of `color`.
      LocVisitor visitor(Loc{i, j});
      std::vector<Loc> region;
      bool region_is_small = true;
      while (!visitor.Done()) {
        Loc loc = visitor.Next();
        region.emplace_back(loc);
        seen[loc.i][loc.j] = true;

        bool loc_is_liberty = group_tracker_->LocIsEmpty(loc) ? false : true;
        for (const auto& nloc : Adjacent(loc)) {
          if (group_tracker_->ColorAt(nloc) == color) {
            // `loc` is a liberty of a `color` chain. Thus, the region seen so
            // far is still small.
            if (group_tracker_->LocIsEmpty(loc)) loc_is_liberty = true;
            continue;
          }

          // nloc is either EMPTY or `OppositeColor(color)`.
          visitor.Visit(nloc);
        }

        if (!loc_is_liberty) {
          // empty, but does not border a stone of `color`.
          region_is_small = false;
        }
      }

      if (!region_is_small) {
        continue;
      }

      // found a small, `color` enclosed region.
      regionid region_id = next_region_id++;
      region_map[region_id] = BensonRegionInfo{
          region_id,
          std::move(region),
      };
    }
  }

  return region_map;
}

void GroupTracker::BensonSolver::PopulateAdjacentRegions(
    GroupMap& group_map, RegionMap& region_map) {
  for (auto& [region_id, region_info] : region_map) {
    // find empty points
    absl::InlinedVector<Loc, BOARD_LEN * BOARD_LEN> empty_locs;
    for (auto& loc : region_info.locs) {
      if (group_tracker_->LocIsEmpty(loc)) {
        empty_locs.emplace_back(loc);
      }
    }

    for (const auto& loc : empty_locs) {
      // check that set of groups this is adjacent to match vital_groups.
      // Any group in vital_groups that is not found is removed.
      for (const auto& nloc : Adjacent(loc)) {
        groupid gid = group_tracker_->GroupAt(nloc);
        if (!MapContains(group_map, gid)) {
          continue;
        }

        group_map[gid].adj_regions.insert(region_id);
      }
    }
  }
}

void GroupTracker::BensonSolver::PopulateVitalRegions(GroupMap& group_map,
                                                      RegionMap& region_map) {
  // A region is vital to a group if all its empty intersections are liberties
  // of that group.
  for (auto& [region_id, region_info] : region_map) {
    // find empty points
    absl::InlinedVector<Loc, BOARD_LEN * BOARD_LEN> empty_locs;
    for (auto& loc : region_info.locs) {
      if (group_tracker_->LocIsEmpty(loc)) {
        empty_locs.emplace_back(loc);
      }
    }

    // find vital groups
    for (const auto& loc : Adjacent(empty_locs[0])) {
      if (MapContains(group_map, group_tracker_->GroupAt(loc))) {
        region_info.vital_groups.insert(group_tracker_->GroupAt(loc));
      }
    }

    for (const auto& loc : empty_locs) {
      // check that set of groups this is adjacent to match vital_groups.
      // Any group in vital_groups that is not found is removed.
      absl::InlinedVector<groupid, 4> loc_adj_groups;
      for (const auto& nloc : Adjacent(loc)) {
        if (!MapContains(group_map, group_tracker_->GroupAt(nloc))) {
          continue;
        }

        loc_adj_groups.emplace_back(group_tracker_->GroupAt(nloc));
      }

      // remove all groups in `vital_groups` that are not adjacent to this loc.
      absl::InlinedVector<groupid, 32> groups_to_remove;  // arbitrary bound
      for (const auto& gid : region_info.vital_groups) {
        if (!InlinedVecContains(loc_adj_groups, gid)) {
          groups_to_remove.emplace_back(gid);
        }
      }

      // remove all non-vital groups
      for (const auto& gid : groups_to_remove) {
        region_info.vital_groups.erase(gid);
      }
    }

    for (const auto& gid : region_info.vital_groups) {
      group_map[gid].num_vital_regions++;
    }
  }
}

void GroupTracker::BensonSolver::RunBenson(GroupMap& group_map,
                                           RegionMap& region_map) {
  // Do the following until neither step removes any chains/regions:
  // - Remove from `group_map` all groups with less than 2 vital regions.
  // - For all removed groups, remove any adjacent small regions.
  absl::InlinedVector<groupid, BOARD_LEN * BOARD_LEN> groups_to_remove;
  while (true) {
    for (auto& [gid, group_info] : group_map) {
      if (group_info.num_vital_regions < 2) {
        groups_to_remove.emplace_back(gid);
      }
    }

    if (groups_to_remove.empty()) {
      // Did not find any groups to remove. We are done!
      break;
    }

    for (auto& gid : groups_to_remove) {
      for (auto& rid : group_map[gid].adj_regions) {
        // remove all regions this group is touching.
        for (auto& vgid : region_map[rid].vital_groups) {
          group_map[vgid].num_vital_regions--;
        }
        region_map.erase(rid);
      }

      group_map.erase(gid);
    }

    groups_to_remove.clear();
  }
}

Board::Board()
    : zobrist_(Zobrist::get()),
      board_(),
      move_count_(0),
      consecutive_passes_(0),
      passes_(0),
      komi_(7.5),
      group_tracker_() {
  Zobrist::Hash hash = 0;
  for (auto i = 0; i < BOARD_LEN; i++) {
    for (auto j = 0; j < BOARD_LEN; j++) {
      hash ^= zobrist_.hash_at(i, j, ZobristState(EMPTY));
    }
  }

  hash_ = hash;
  seen_states_.insert(hash_);
}

bool Board::IsValidMove(Loc loc, Color color) const {
  if (loc == kPassLoc) {
    return true;
  }

  return MoveOk(PlayMoveDry(loc, color));
}

bool Board::IsGameOver() const { return consecutive_passes_ == 2; }

MoveStatus Board::PlayBlack(int i, int j) { return PlayMove(Loc{i, j}, BLACK); }
MoveStatus Board::PlayWhite(int i, int j) { return PlayMove(Loc{i, j}, WHITE); }

MoveStatus Board::PlayMove(Loc loc, Color color) {
  if (loc == kPassLoc) {
    return Pass(color);
  }

  MoveResult move_result = PlayMoveDry(loc, color);
  if (!MoveOk(move_result)) {
    return move_result.status;
  }

  // reaching here means the move is valid. Mutable portion begins here.
  std::optional<MoveInfo>& move_info = move_result.move_info;

  // remove captures.
  LocVec captured_stones;
  for (const Transition& transition : move_info->capture_transitions) {
    captured_stones.emplace_back(transition.loc);
  }

  group_tracker_.RemoveCaptures(captured_stones);
  for (const Loc& loc : captured_stones) {
    SetLoc(loc, EMPTY);
  }

  // play stone.
  SetLoc(loc, color);
  group_tracker_.Move(loc, color);

  // update tracking variables
  consecutive_passes_ = 0;
  move_count_++;

  hash_ = move_info->new_hash;
  seen_states_.insert(hash_);

  return MoveStatus::kValid;
}

MoveStatus Board::Pass(Color color) {
  consecutive_passes_++;
  passes_++;

  if (!IsGameOver() && passes_ >= constants::kNumPassesBeforeBensons) {
    group_tracker_.CalculatePassAliveRegions(hash_);
  }

  return MoveStatus::kValid;
}

MoveResult Board::PlayMoveDry(Loc loc, Color color) const {
  if (loc == kPassLoc) {
    return MoveResult{MoveStatus::kValid,
                      MoveInfo{Transition{loc, color, color},
                               MoveInfo::Transitions(), hash_}};
  } else if (color != BLACK && color != WHITE) {
    return MoveResult{MoveStatus::kUnknownColor, std::nullopt};
  } else if (loc.i < 0 || loc.i >= BOARD_LEN || loc.j < 0 ||
             loc.j >= BOARD_LEN) {
    return MoveResult{MoveStatus::kOutOfBounds, std::nullopt};
  } else if (AtLoc(loc) != EMPTY) {
    return MoveResult{MoveStatus::kLocNotEmpty, std::nullopt};
  } else if (group_tracker_.IsPassAlive(loc)) {
    return MoveResult{MoveStatus::kPassAliveRegion, std::nullopt};
  }

  // check for captures.
  absl::InlinedVector<groupid, 4> captured_groups =
      GetCapturedGroups(loc, OppositeColor(color));

  // if no captures, check for self-atari.
  if (captured_groups.size() == 0 && IsSelfCapture(loc, color)) {
    return MoveResult{MoveStatus::kSelfCapture, std::nullopt};
  }

  // reaching here means the move is playable.
  Transition move_transition = Transition{loc, AtLoc(loc), color};
  MoveInfo::Transitions capture_transitions;

  // resolve captures.
  absl::InlinedVector<Loc, constants::kMaxNumBoardLocs> captured_stones;
  for (const groupid& captured_id : captured_groups) {
    LocVec captures = group_tracker_.ExpandGroup(captured_id);
    captured_stones.insert(captured_stones.end(), captures.begin(),
                           captures.end());
  }

  for (const Loc& loc : captured_stones) {
    capture_transitions.emplace_back(Transition{loc, AtLoc(loc), EMPTY});
  }

  // check if we have already seen this board position.
  Zobrist::Hash hash = RecomputeHash(move_transition, capture_transitions);
  if (seen_states_.contains(hash)) {
    return MoveResult{MoveStatus::kRepeatedPosition, std::nullopt};
  }

  return MoveResult{MoveStatus::kValid,
                    MoveInfo{move_transition, capture_transitions, hash}};
}

void Board::CalculatePassAliveRegions() {
  group_tracker_.CalculatePassAliveRegions(hash_);
}

Scores Board::GetScores() {
  // (re) calculate PA regions for score accuracy.
  group_tracker_.CalculatePassAliveRegions(hash_);
  std::pair<float, std::array<Color, BOARD_LEN* BOARD_LEN>> bscore_ownership =
      ScoreAndOwnership(BLACK);
  std::pair<float, std::array<Color, BOARD_LEN* BOARD_LEN>> wscore_ownership =
      ScoreAndOwnership(WHITE);

  std::array<Color, BOARD_LEN * BOARD_LEN> ownership;
  for (int i = 0; i < BOARD_LEN; ++i) {
    for (int j = 0; j < BOARD_LEN; ++j) {
      int idx = i * BOARD_LEN + j;
      if (bscore_ownership.second[idx] == BLACK) {
        ownership[idx] = BLACK;
      } else if (wscore_ownership.second[idx] == WHITE) {
        ownership[idx] = WHITE;
      } else {
        ownership[idx] = EMPTY;
      }
    }
  }
  return Scores{bscore_ownership.first, wscore_ownership.first, ownership};
}

std::string Board::ToString() const {
  std::stringstream ss;
  ss << *this;

  return ss.str();
}

bool Board::IsSelfCapture(Loc loc, Color color) const {
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

std::pair<float, std::array<Color, BOARD_LEN * BOARD_LEN>>
Board::ScoreAndOwnership(Color color) const {
  bool counted[BOARD_LEN][BOARD_LEN]{};
  std::array<Color, BOARD_LEN * BOARD_LEN> ownership{};
  int score = 0;
  for (auto i = 0; i < BOARD_LEN; ++i) {
    for (auto j = 0; j < BOARD_LEN; ++j) {
      if (counted[i][j]) {
        continue;
      } else if (at(i, j) == color) {
        bool stone_is_dead =
            group_tracker_.IsPassAliveForColor(Loc{i, j}, OppositeColor(color));
        if (!stone_is_dead) {
          ++score;
          ownership[i * BOARD_LEN + j] = color;
        }

        counted[i][j] = true;
        continue;
      } else if (at(i, j) == OppositeColor(color)) {
        counted[i][j] = true;
        continue;
      }

      // empty, unseen region.
      // only count empty + dead stones in visitor. Self-colored stones are
      // handled in loop.
      LocVisitor visitor(Loc{i, j});
      LocVec region;
      int region_score = 0;
      bool seen_self_color = false;
      bool seen_opp_color = false;
      while (!visitor.Done()) {
        Loc loc = visitor.Next();
        if (AtLoc(loc) == color) {
          seen_self_color = true;
          continue;
        }

        // either empty region or opposite color
        if (AtLoc(loc) == OppositeColor(color)) {
          if (group_tracker_.IsPassAliveForColor(loc, color)) {
            region.emplace_back(loc);
            ++region_score;  // dead stone should not be double-counted.
          } else {
            seen_opp_color = true;
          }
        } else {
          // empty region
          region.emplace_back(loc);
          ++region_score;
        }

        counted[loc.i][loc.j] = true;
        for (const auto& nloc : Adjacent(loc)) {
          visitor.Visit(nloc);
        }
      }

      bool count_region = seen_self_color && !seen_opp_color;
      if (count_region) {
        score += region_score;
        for (const auto& loc : region) {
          ownership[loc.i * BOARD_LEN + loc.j] = color;
        }
      }
    }
  }

  return std::make_pair(
      static_cast<float>(score) + (color == WHITE ? komi_ : 0), ownership);
}

absl::InlinedVector<groupid, 4> Board::GetCapturedGroups(
    Loc loc, int captured_color) const {
  absl::InlinedVector<groupid, 4> captured_groups;

  for (const Loc& nloc : Adjacent(loc)) {
    groupid id = group_tracker_.GroupAt(nloc);
    if (AtLoc(nloc) == captured_color && IsInAtari(nloc) &&
        !InlinedVecContains(captured_groups, id)) {
      // this group is in atari, and we have filled the last liberty.
      captured_groups.emplace_back(group_tracker_.GroupAt(nloc));
    }
  }

  return captured_groups;
}

absl::InlinedVector<Loc, 4> Board::AdjacentOfColor(Loc loc, Color color) const {
  absl::InlinedVector<Loc, 4> adjacent_points;

  Loc left = Loc{loc.i - 1, loc.j};
  if (loc.i > 0 && AtLoc(left) == color) {
    adjacent_points.emplace_back(left);
  }

  Loc top = Loc{loc.i, loc.j - 1};
  if (loc.j > 0 && AtLoc(top) == color) {
    adjacent_points.emplace_back(top);
  }

  Loc right = Loc{loc.i + 1, loc.j};
  if (loc.i < BOARD_LEN - 1 && AtLoc(right) == color) {
    adjacent_points.emplace_back(right);
  }

  Loc bottom = Loc{loc.i, loc.j + 1};
  if (loc.j < BOARD_LEN - 1 && AtLoc(bottom) == color) {
    adjacent_points.emplace_back(bottom);
  }

  return adjacent_points;
}

Zobrist::Hash Board::RecomputeHash(
    const Transition& move_transition,
    const MoveInfo::Transitions& capture_transitions) const {
  Zobrist::Hash hash = hash_;
  hash ^= zobrist_.hash_at(move_transition.loc.i, move_transition.loc.j,
                           ZobristState(move_transition.last_piece));
  hash ^= zobrist_.hash_at(move_transition.loc.i, move_transition.loc.j,
                           ZobristState(move_transition.current_piece));
  for (auto& capture_transition : capture_transitions) {
    // one hash to "undo" presence of last piece, one to add new piece.
    hash ^= zobrist_.hash_at(capture_transition.loc.i, capture_transition.loc.j,
                             ZobristState(capture_transition.last_piece));
    hash ^= zobrist_.hash_at(capture_transition.loc.i, capture_transition.loc.j,
                             ZobristState(capture_transition.current_piece));
  }

  return hash;
}
}  // namespace game
