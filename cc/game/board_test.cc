#include "cc/game/board.h"

#include "absl/container/flat_hash_set.h"
#include "cc/core/doctest_include.h"
#include "cc/game/zobrist_hash.h"

#define IN_UNIT_TEST

namespace game {

TEST_CASE("BoardTest") {
  SUBCASE("NewBoardIsEmpty") {
    Zobrist table_;
    game::Board board(&table_);

    for (unsigned i = 0; i < BOARD_LEN; i++) {
      for (unsigned j = 0; j < BOARD_LEN; j++) {
        CHECK_EQ(board.at(i, j), EMPTY);
      }
    }
  }

  SUBCASE("MovingOnEmptyBoardSetsColor") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(0, 0);

    CHECK_EQ(board.at(0, 0), BLACK);
  }

  SUBCASE("MovingOnOccupiedSpotFails") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(0, 0);

    CHECK_FALSE(board.MoveWhite(0, 0));
    CHECK_EQ(board.at(0, 0), BLACK);
  }

  SUBCASE("BoardStateNotAlreadySeenReturnsTrue") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(0, 0);

    CHECK(board.MoveWhite(1, 1));
  }

  SUBCASE("NoSelfAtari") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(1, 0);
    board.MoveBlack(0, 1);

    CHECK_FALSE(board.MoveWhite(0, 0));
  }

  SUBCASE("NoSelfAtariMultipleStones") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(2, 3);
    board.MoveBlack(2, 4);
    board.MoveBlack(3, 2);
    board.MoveBlack(4, 3);
    board.MoveBlack(4, 4);
    board.MoveBlack(3, 5);

    board.MoveWhite(3, 3);

    CHECK_FALSE(board.MoveWhite(3, 4));
  }

  SUBCASE("BlackCapturesAdjacentWhite") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(0, 1);
    board.MoveWhite(0, 0);

    CHECK(board.MoveBlack(1, 0));
    CHECK(board.at(0, 0) == EMPTY);
  }

  SUBCASE("BlackCapturesWhiteCenter") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(1, 1);
    board.MoveBlack(2, 2);
    board.MoveBlack(3, 1);
    board.MoveWhite(2, 1);

    CHECK(board.at(2, 1) == WHITE);
    CHECK(board.MoveBlack(2, 0));
    CHECK(board.at(2, 1) == EMPTY);
  }

  SUBCASE("BlackCapturesMultipleStones") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveWhite(1, 1);
    board.MoveWhite(2, 1);

    board.MoveBlack(0, 1);
    board.MoveBlack(1, 2);
    board.MoveBlack(2, 2);
    board.MoveBlack(3, 1);
    board.MoveBlack(2, 0);

    CHECK(board.at(1, 1) == WHITE);
    CHECK(board.at(2, 1) == WHITE);

    CHECK(board.MoveBlack(1, 0));

    CHECK(board.at(1, 1) == EMPTY);
    CHECK(board.at(2, 1) == EMPTY);
  }

  SUBCASE("KoCannotRecaptureImmediately") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(2, 1);
    board.MoveWhite(2, 2);
    board.MoveBlack(3, 2);
    board.MoveWhite(3, 3);
    board.MoveBlack(2, 3);
    board.MoveWhite(2, 4);
    board.MoveBlack(1, 2);
    board.MoveWhite(1, 3);
    board.MoveWhite(2, 2);

    CHECK(board.at(2, 3) == EMPTY);
    CHECK_FALSE(board.MoveBlack(2, 3));
  }

  SUBCASE("SendTwoReturnOne") {
    Zobrist table_;
    game::Board board(&table_);

    board.MoveBlack(2, 1);
    board.MoveWhite(0, 1);
    board.MoveBlack(1, 0);
    board.MoveWhite(1, 1);
    board.MoveBlack(3, 0);

    // send two
    board.MoveBlack(0, 0);
    board.MoveWhite(2, 0);

    CHECK(board.at(0, 0) == EMPTY);
    CHECK(board.at(1, 0) == EMPTY);
    CHECK_FALSE(board.MoveBlack(1, 0));
  }
}

TEST_CASE("GroupTrackerTest") {
  GroupTracker group_tracker(BOARD_LEN);

  SUBCASE("NewGroup") {
    groupid gid = group_tracker.NewGroup(Loc{3, 3}, BLACK);

    CHECK(group_tracker.LibertiesForGroup(gid) == 4);
    CHECK(group_tracker.LibertiesForGroupAt(Loc{3, 3}) == 4);
    CHECK(group_tracker.GroupAt(Loc{3, 3}) == gid);
  }

  SUBCASE("AddToGroup") {
    groupid gid = group_tracker.NewGroup(Loc{3, 3}, BLACK);
    group_tracker.AddToGroup(Loc{3, 4}, gid);

    CHECK(group_tracker.LibertiesForGroup(gid) == 6);
    CHECK(group_tracker.LibertiesForGroupAt(Loc{3, 3}) == 6);
    CHECK(group_tracker.GroupAt(Loc{3, 4}) == gid);
  }

  SUBCASE("NoDoubleCountLiberties") {
    groupid gid = group_tracker.NewGroup(Loc{3, 3}, BLACK);
    group_tracker.AddToGroup(Loc{3, 4}, gid);
    group_tracker.AddToGroup(Loc{4, 4}, gid);

    CHECK(group_tracker.LibertiesForGroup(gid) == 7);
    CHECK(group_tracker.LibertiesForGroupAt(Loc{3, 3}) == 7);
    CHECK(group_tracker.GroupAt(Loc{3, 4}) == gid);
  }

  SUBCASE("OppositeStonesRemoveLiberties") {
    groupid gid0 = group_tracker.NewGroup(Loc{3, 3}, BLACK);
    groupid gid1 = group_tracker.NewGroup(Loc{3, 4}, WHITE);

    CHECK(group_tracker.LibertiesForGroup(gid0) == 3);
    CHECK(group_tracker.LibertiesForGroup(gid1) == 3);
  }

  SUBCASE("CoalesceGroups") {
    groupid gid0 = group_tracker.NewGroup(Loc{3, 3}, BLACK);
    groupid gid1 = group_tracker.NewGroup(Loc{3, 5}, BLACK);

    CHECK(group_tracker.LibertiesForGroup(gid0) == 4);
    CHECK(group_tracker.LibertiesForGroup(gid1) == 4);

    group_tracker.AddToGroup(Loc{3, 4}, gid0);
    groupid gid2 = group_tracker.CoalesceGroups({gid0, gid1});

    CHECK(group_tracker.GroupAt(Loc{3, 3}) == gid2);
    CHECK(group_tracker.GroupAt(Loc{3, 4}) == gid2);
    CHECK(group_tracker.GroupAt(Loc{3, 5}) == gid2);
    CHECK(group_tracker.LibertiesForGroup(gid2) == 8);
    CHECK(group_tracker.LibertiesForGroupAt(Loc{3, 3}) ==
          group_tracker.LibertiesForGroupAt(Loc{3, 4}));
    CHECK(group_tracker.LibertiesForGroupAt(Loc{3, 3}) ==
          group_tracker.LibertiesForGroupAt(Loc{3, 5}));
    CHECK(group_tracker.LibertiesForGroupAt(Loc{3, 4}) ==
          group_tracker.LibertiesForGroupAt(Loc{3, 5}));
  }

  SUBCASE("CoalesceMultipleGroups") {
    // groups hug star point.
    groupid gid0 = group_tracker.NewGroup(Loc{2, 3}, BLACK);
    groupid gid1 = group_tracker.NewGroup(Loc{4, 3}, BLACK);
    groupid gid2 = group_tracker.NewGroup(Loc{3, 2}, BLACK);
    groupid gid3 = group_tracker.NewGroup(Loc{3, 4}, BLACK);

    group_tracker.AddToGroup(Loc{1, 3}, gid0);
    CHECK(group_tracker.LibertiesForGroup(gid0) == 6);
    CHECK(group_tracker.LibertiesForGroup(gid1) == 4);
    CHECK(group_tracker.LibertiesForGroup(gid2) == 4);
    CHECK(group_tracker.LibertiesForGroup(gid3) == 4);

    group_tracker.AddToGroup(Loc{3, 3}, gid2);
    groupid final_gid = group_tracker.CoalesceGroups({gid0, gid1, gid2, gid3});
    CHECK(group_tracker.GroupAt(Loc{2, 3}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{4, 3}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{3, 2}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{3, 4}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{1, 3}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{3, 3}) == final_gid);
    CHECK(group_tracker.LibertiesForGroup(final_gid) == 10);
  }

  SUBCASE("CoalesceGroupsOverlappingLiberties") {
    groupid gid0 = group_tracker.NewGroup(Loc{2, 3}, BLACK);
    groupid gid1 = group_tracker.NewGroup(Loc{4, 3}, BLACK);

    group_tracker.AddToGroup(Loc{2, 4}, gid0);
    group_tracker.AddToGroup(Loc{4, 4}, gid1);

    group_tracker.AddToGroup(Loc{3, 3}, gid0);
    groupid final_gid = group_tracker.CoalesceGroups({gid0, gid1});
    CHECK(group_tracker.GroupAt(Loc{2, 4}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{4, 4}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{3, 3}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{2, 3}) == final_gid);
    CHECK(group_tracker.GroupAt(Loc{4, 3}) == final_gid);
    CHECK(group_tracker.LibertiesForGroup(final_gid) == 10);
  }

  SUBCASE("EdgeKo") {
    groupid gid0 = group_tracker.NewGroup(Loc{0, 2}, BLACK);
    group_tracker.AddToGroup(Loc{1, 2}, gid0);
    group_tracker.AddToGroup(Loc{1, 1}, gid0);

    CHECK(group_tracker.LibertiesForGroup(gid0) == 6);

    groupid gid1 = group_tracker.NewGroup(Loc{0, 1}, WHITE);
    CHECK(group_tracker.LibertiesForGroup(gid0) == 5);
  }
}

bool PaRegionsMatch(GroupTracker& group_tracker,
                    absl::flat_hash_set<Loc>& region, int color) {
  for (int i = 0; i < group_tracker.length(); ++i) {
    for (int j = 0; j < group_tracker.length(); ++j) {
      bool loc_pass_alive = group_tracker.IsPassAliveForColor(Loc{i, j}, color);
      if (loc_pass_alive && !region.contains(Loc{i, j})) {
        std::cerr << "<<axlui> Missing loc: " << Loc{i, j};
        return false;
      }

      region.erase(Loc{i, j});
    }
  }

  if (!region.empty()) {
    for (auto& loc : region) {
      std::cerr << "<<axlui>> Loc remaining: " << loc;
    }
    return false;
  }

  return true;
}

TEST_CASE("BensonTest") {
  GroupTracker group_tracker(BOARD_LEN);

  // . o . o
  // o o o o
  SUBCASE("Corner") {
    groupid gid = group_tracker.NewGroup(Loc{0, 1}, BLACK);
    group_tracker.AddToGroup(Loc{0, 3}, gid);
    group_tracker.AddToGroup(Loc{1, 0}, gid);
    group_tracker.AddToGroup(Loc{1, 1}, gid);
    group_tracker.AddToGroup(Loc{1, 2}, gid);
    group_tracker.AddToGroup(Loc{1, 3}, gid);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {{0, 0}, {0, 1}, {0, 2}, {0, 3},
                                          {1, 0}, {1, 1}, {1, 2}, {1, 3}};
    CHECK(PaRegionsMatch(group_tracker, pa_region, BLACK));
  }

  // x . o . o
  // o o o o o
  SUBCASE("WhiteStone") {
    groupid gid = group_tracker.NewGroup(Loc{0, 2}, BLACK);
    group_tracker.AddToGroup(Loc{0, 4}, gid);
    group_tracker.AddToGroup(Loc{1, 0}, gid);
    group_tracker.AddToGroup(Loc{1, 1}, gid);
    group_tracker.AddToGroup(Loc{1, 2}, gid);
    group_tracker.AddToGroup(Loc{1, 3}, gid);
    group_tracker.AddToGroup(Loc{1, 4}, gid);

    groupid gid1 = group_tracker.NewGroup(Loc{0, 0}, WHITE);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {{0, 0}, {0, 1}, {0, 2}, {0, 3},
                                          {0, 4}, {1, 0}, {1, 1}, {1, 2},
                                          {1, 3}, {1, 4}};
    CHECK(PaRegionsMatch(group_tracker, pa_region, BLACK));
  }

  // x x . o . o
  // o o o o o o
  SUBCASE("ManyWhiteStones") {
    groupid gid1 = group_tracker.NewGroup(Loc{0, 0}, WHITE);
    group_tracker.AddToGroup(Loc{0, 1}, gid1);

    groupid gid = group_tracker.NewGroup(Loc{0, 3}, BLACK);
    group_tracker.AddToGroup(Loc{0, 5}, gid);
    group_tracker.AddToGroup(Loc{1, 0}, gid);
    group_tracker.AddToGroup(Loc{1, 1}, gid);
    group_tracker.AddToGroup(Loc{1, 2}, gid);
    group_tracker.AddToGroup(Loc{1, 3}, gid);
    group_tracker.AddToGroup(Loc{1, 4}, gid);
    group_tracker.AddToGroup(Loc{1, 5}, gid);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {{0, 0}, {0, 1}, {0, 2}, {0, 3},
                                          {0, 4}, {0, 5}, {1, 0}, {1, 1},
                                          {1, 2}, {1, 3}, {1, 4}, {1, 5}};
    CHECK(PaRegionsMatch(group_tracker, pa_region, BLACK));
  }

  // . . . . . .
  // . . . . o o
  // . o o o . o
  // . o . . o o
  // . o o o . .
  // . . . . . .
  SUBCASE("NonVitalRegion") {
    groupid gid0 = group_tracker.NewGroup(Loc{1, 4}, BLACK);
    group_tracker.AddToGroup(Loc{1, 5}, gid0);
    group_tracker.AddToGroup(Loc{3, 4}, gid0);
    group_tracker.AddToGroup(Loc{3, 5}, gid0);
    group_tracker.AddToGroup(Loc{2, 5}, gid0);

    groupid gid1 = group_tracker.NewGroup(Loc{2, 1}, BLACK);
    group_tracker.AddToGroup(Loc{2, 1}, gid1);
    group_tracker.AddToGroup(Loc{2, 2}, gid1);
    group_tracker.AddToGroup(Loc{2, 3}, gid1);
    group_tracker.AddToGroup(Loc{3, 1}, gid1);
    group_tracker.AddToGroup(Loc{4, 1}, gid1);
    group_tracker.AddToGroup(Loc{4, 2}, gid1);
    group_tracker.AddToGroup(Loc{4, 3}, gid1);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {};
    CHECK(PaRegionsMatch(group_tracker, pa_region, BLACK));
  }

  // . . . . . .
  // . o o o o o
  // o o . o . o
  // o . x . o o
  // o o . o . .
  // . o o o . .
  SUBCASE("VitalRegion") {
    groupid gid = group_tracker.NewGroup(Loc{1, 4}, BLACK);
    group_tracker.AddToGroup(Loc{1, 1}, gid);
    group_tracker.AddToGroup(Loc{1, 2}, gid);
    group_tracker.AddToGroup(Loc{1, 3}, gid);
    group_tracker.AddToGroup(Loc{1, 4}, gid);
    group_tracker.AddToGroup(Loc{1, 5}, gid);
    group_tracker.AddToGroup(Loc{2, 0}, gid);
    group_tracker.AddToGroup(Loc{2, 1}, gid);
    group_tracker.AddToGroup(Loc{2, 3}, gid);
    group_tracker.AddToGroup(Loc{2, 5}, gid);
    group_tracker.AddToGroup(Loc{3, 0}, gid);
    group_tracker.AddToGroup(Loc{3, 4}, gid);
    group_tracker.AddToGroup(Loc{3, 5}, gid);
    group_tracker.AddToGroup(Loc{4, 0}, gid);
    group_tracker.AddToGroup(Loc{4, 1}, gid);
    group_tracker.AddToGroup(Loc{4, 3}, gid);
    group_tracker.AddToGroup(Loc{5, 1}, gid);
    group_tracker.AddToGroup(Loc{5, 2}, gid);
    group_tracker.AddToGroup(Loc{5, 3}, gid);

    groupid gid0 = group_tracker.NewGroup(Loc{3, 2}, WHITE);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {
        {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 0}, {2, 1}, {2, 2},
        {2, 3}, {2, 4}, {2, 5}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4},
        {3, 5}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {5, 1}, {5, 2}, {5, 3}};
    CHECK(PaRegionsMatch(group_tracker, pa_region, BLACK));
  }

  // . . . . . .
  // . o o o o o
  // o o . o . o
  // o . . . o o
  // o o . o . .
  // . o o o . .
  SUBCASE("NonSmallRegion") {
    groupid gid = group_tracker.NewGroup(Loc{1, 4}, BLACK);
    group_tracker.AddToGroup(Loc{1, 1}, gid);
    group_tracker.AddToGroup(Loc{1, 2}, gid);
    group_tracker.AddToGroup(Loc{1, 3}, gid);
    group_tracker.AddToGroup(Loc{1, 4}, gid);
    group_tracker.AddToGroup(Loc{1, 5}, gid);
    group_tracker.AddToGroup(Loc{2, 0}, gid);
    group_tracker.AddToGroup(Loc{2, 1}, gid);
    group_tracker.AddToGroup(Loc{2, 3}, gid);
    group_tracker.AddToGroup(Loc{2, 5}, gid);
    group_tracker.AddToGroup(Loc{3, 0}, gid);
    group_tracker.AddToGroup(Loc{3, 4}, gid);
    group_tracker.AddToGroup(Loc{3, 5}, gid);
    group_tracker.AddToGroup(Loc{4, 0}, gid);
    group_tracker.AddToGroup(Loc{4, 1}, gid);
    group_tracker.AddToGroup(Loc{4, 3}, gid);
    group_tracker.AddToGroup(Loc{5, 1}, gid);
    group_tracker.AddToGroup(Loc{5, 2}, gid);
    group_tracker.AddToGroup(Loc{5, 3}, gid);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {};
    CHECK(PaRegionsMatch(group_tracker, pa_region, BLACK));
  }

  // . o . o
  // o . o o
  SUBCASE("NonPA") {
    groupid gid = group_tracker.NewGroup(Loc{0, 3}, BLACK);
    group_tracker.AddToGroup(Loc{1, 2}, gid);
    group_tracker.AddToGroup(Loc{1, 3}, gid);

    groupid gid1 = group_tracker.NewGroup(Loc{0, 1}, BLACK);
    groupid gid2 = group_tracker.NewGroup(Loc{1, 0}, BLACK);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {};
    CHECK(PaRegionsMatch(group_tracker, pa_region, BLACK));
  }

  // . . o . o . o o
  // o o o o . o . o
  // . . . . . . o o
  SUBCASE("Senseis") {
    groupid gid0 = group_tracker.NewGroup(Loc{1, 0}, BLACK);
    group_tracker.AddToGroup(Loc{1, 1}, gid0);
    group_tracker.AddToGroup(Loc{1, 2}, gid0);
    group_tracker.AddToGroup(Loc{1, 3}, gid0);
    group_tracker.AddToGroup(Loc{0, 2}, gid0);

    groupid gid1 = group_tracker.NewGroup(Loc{0, 4}, BLACK);
    groupid gid2 = group_tracker.NewGroup(Loc{1, 5}, BLACK);
    groupid gid3 = group_tracker.NewGroup(Loc{0, 6}, BLACK);
    group_tracker.AddToGroup(Loc{0, 7}, gid3);
    group_tracker.AddToGroup(Loc{1, 7}, gid3);
    group_tracker.AddToGroup(Loc{2, 7}, gid3);
    group_tracker.AddToGroup(Loc{2, 6}, gid3);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {1, 0},
        {1, 1}, {1, 2}, {1, 3}, {1, 5}, {1, 6}, {1, 7}, {2, 6}, {2, 7}};
    CHECK(PaRegionsMatch(group_tracker, pa_region, BLACK));
  }

  // . . . . o o .
  // o o o o . o .
  // o o x x o o .
  // o x x . x o .
  // o x . x x o .
  // o o x x o o .
  // . o o o o . .
  SUBCASE("NestedPaRegions") {
    groupid gid0 = group_tracker.NewGroup(Loc{2, 2}, WHITE);
    group_tracker.AddToGroup(Loc{2, 3}, gid0);
    group_tracker.AddToGroup(Loc{3, 1}, gid0);
    group_tracker.AddToGroup(Loc{3, 2}, gid0);
    group_tracker.AddToGroup(Loc{4, 1}, gid0);

    groupid gid1 = group_tracker.NewGroup(Loc{3, 4}, WHITE);
    group_tracker.AddToGroup(Loc{4, 4}, gid1);
    group_tracker.AddToGroup(Loc{4, 3}, gid1);
    group_tracker.AddToGroup(Loc{5, 3}, gid1);
    group_tracker.AddToGroup(Loc{5, 2}, gid1);

    groupid gid2 = group_tracker.NewGroup(Loc{0, 4}, BLACK);
    group_tracker.AddToGroup(Loc{0, 5}, gid2);
    group_tracker.AddToGroup(Loc{1, 0}, gid2);
    group_tracker.AddToGroup(Loc{1, 1}, gid2);
    group_tracker.AddToGroup(Loc{1, 2}, gid2);
    group_tracker.AddToGroup(Loc{1, 3}, gid2);
    group_tracker.AddToGroup(Loc{1, 5}, gid2);
    group_tracker.AddToGroup(Loc{2, 0}, gid2);
    group_tracker.AddToGroup(Loc{2, 1}, gid2);
    group_tracker.AddToGroup(Loc{2, 4}, gid2);
    group_tracker.AddToGroup(Loc{2, 5}, gid2);
    group_tracker.AddToGroup(Loc{3, 0}, gid2);
    group_tracker.AddToGroup(Loc{3, 5}, gid2);
    group_tracker.AddToGroup(Loc{4, 0}, gid2);
    group_tracker.AddToGroup(Loc{4, 5}, gid2);
    group_tracker.AddToGroup(Loc{5, 0}, gid2);
    group_tracker.AddToGroup(Loc{5, 1}, gid2);
    group_tracker.AddToGroup(Loc{5, 4}, gid2);
    group_tracker.AddToGroup(Loc{5, 5}, gid2);
    group_tracker.AddToGroup(Loc{6, 1}, gid2);
    group_tracker.AddToGroup(Loc{6, 2}, gid2);
    group_tracker.AddToGroup(Loc{6, 3}, gid2);
    group_tracker.AddToGroup(Loc{6, 4}, gid2);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);
    group_tracker.CalculatePassAliveRegionForColor(WHITE);

    absl::flat_hash_set<Loc> pa_region_black = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {1, 0},
        {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 0}, {2, 1},
        {2, 4}, {2, 5}, {3, 0}, {3, 5}, {4, 0}, {4, 5}, {5, 0},
        {5, 1}, {5, 4}, {5, 5}, {6, 1}, {6, 2}, {6, 3}, {6, 4}};
    absl::flat_hash_set<Loc> pa_region_white = {{2, 2}, {2, 3}, {3, 1}, {3, 2},
                                                {3, 3}, {3, 4}, {4, 1}, {4, 2},
                                                {4, 3}, {4, 4}, {5, 2}, {5, 3}};
    CHECK(PaRegionsMatch(group_tracker, pa_region_black, BLACK));
    CHECK(PaRegionsMatch(group_tracker, pa_region_white, WHITE));
  }
}  // namespace game

}  // namespace game
