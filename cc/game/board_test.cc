#include "cc/game/board.h"

#include "cc/core/doctest_include.h"
#include "cc/game/zobrist_hash.h"

#define IN_UNIT_TEST

namespace game {

TEST_CASE("BoardTest") {
  SUBCASE("NewBoardIsEmpty") {
    ZobristTable table_;
    game::Board board(&table_);

    for (unsigned i = 0; i < BOARD_LEN; i++) {
      for (unsigned j = 0; j < BOARD_LEN; j++) {
        CHECK_EQ(board.at(i, j), EMPTY);
      }
    }
  }

  SUBCASE("MovingOnEmptyBoardSetsColor") {
    ZobristTable table_;
    game::Board board(&table_);

    board.MoveBlack(0, 0);

    CHECK_EQ(board.at(0, 0), BLACK);
  }

  SUBCASE("MovingOnOccupiedSpotFails") {
    ZobristTable table_;
    game::Board board(&table_);

    board.MoveBlack(0, 0);

    CHECK_FALSE(board.MoveWhite(0, 0));
    CHECK_EQ(board.at(0, 0), BLACK);
  }

  SUBCASE("BoardStateNotAlreadySeenReturnsTrue") {
    ZobristTable table_;
    game::Board board(&table_);

    board.MoveBlack(0, 0);

    CHECK(board.MoveWhite(1, 1));
  }

  SUBCASE("NoSelfAtari") {
    ZobristTable table_;
    game::Board board(&table_);

    board.MoveBlack(1, 0);
    board.MoveBlack(0, 1);

    CHECK_FALSE(board.MoveWhite(0, 0));
  }

  SUBCASE("NoSelfAtariMultipleStones") {
    ZobristTable table_;
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
    ZobristTable table_;
    game::Board board(&table_);

    board.MoveBlack(0, 1);
    board.MoveWhite(0, 0);

    CHECK(board.MoveBlack(1, 0));
    CHECK(board.at(0, 0) == EMPTY);
  }

  SUBCASE("BlackCapturesWhiteCenter") {
    ZobristTable table_;
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
    ZobristTable table_;
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
    ZobristTable table_;
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
    ZobristTable table_;
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

}  // namespace game
