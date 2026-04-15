#include "cc/game/board.h"

#include "absl/container/flat_hash_set.h"
#include "cc/core/doctest_include.h"
#include "cc/game/board_dsl.h"
#include "cc/game/zobrist.h"

namespace game {

game::Board ParseBoard(const std::string& s) {
  game::Board board;
  for (int idx = 0; idx < constants::kNumBoardLocs; ++idx) {
    game::Loc loc = game::AsLoc(idx);
    switch (s[idx]) {
      case 'X':
        board.PlayMove(loc, BLACK);
        break;
      case 'O':
        board.PlayMove(loc, WHITE);
        break;
      case '.':
      case '+':
        continue;
      default:
        std::cerr << "Invalid Character: " << s[idx] << "\n";
        std::abort();
    }
  }

  return board;
}

TEST_CASE("BoardTest") {
  SUBCASE("NewBoardIsEmpty") {
    game::Board board;

    for (unsigned i = 0; i < BOARD_LEN; i++) {
      for (unsigned j = 0; j < BOARD_LEN; j++) {
        CHECK_EQ(board.at(i, j), EMPTY);
      }
    }
  }

  SUBCASE("MovingOnEmptyBoardSetsColor") {
    game::Board board;

    board.PlayBlack(0, 0);

    CHECK_EQ(board.at(0, 0), BLACK);
  }

  SUBCASE("MovingOnOccupiedSpotFails") {
    game::Board board;

    board.PlayBlack(0, 0);

    CHECK_FALSE(MoveOk(board.PlayWhite(0, 0)));
    CHECK_EQ(board.at(0, 0), BLACK);
  }

  SUBCASE("BoardStateNotAlreadySeenReturnsTrue") {
    game::Board board;

    board.PlayBlack(0, 0);

    CHECK(MoveOk(board.PlayWhite(1, 1)));
  }

  SUBCASE("NoSelfAtari") {
    game::Board board;

    board.PlayBlack(1, 0);
    board.PlayBlack(0, 1);

    CHECK_FALSE(MoveOk(board.PlayWhite(0, 0)));
  }

  SUBCASE("NoSelfAtariMultipleStones") {
    game::Board board;

    board.PlayBlack(2, 3);
    board.PlayBlack(2, 4);
    board.PlayBlack(3, 2);
    board.PlayBlack(4, 3);
    board.PlayBlack(4, 4);
    board.PlayBlack(3, 5);

    board.PlayWhite(3, 3);

    CHECK_FALSE(MoveOk(board.PlayWhite(3, 4)));
  }

  SUBCASE("BlackCapturesAdjacentWhite") {
    game::Board board;

    board.PlayBlack(0, 1);
    board.PlayWhite(0, 0);

    CHECK(MoveOk(board.PlayBlack(1, 0)));
    CHECK(board.at(0, 0) == EMPTY);
  }

  SUBCASE("BlackCapturesWhiteCenter") {
    game::Board board;

    board.PlayBlack(1, 1);
    board.PlayBlack(2, 2);
    board.PlayBlack(3, 1);
    board.PlayWhite(2, 1);

    CHECK(board.at(2, 1) == WHITE);
    CHECK(MoveOk(board.PlayBlack(2, 0)));
    CHECK(board.at(2, 1) == EMPTY);
  }

  SUBCASE("BlackCapturesMultipleStones") {
    game::Board board;

    board.PlayWhite(1, 1);
    board.PlayWhite(2, 1);

    board.PlayBlack(0, 1);
    board.PlayBlack(1, 2);
    board.PlayBlack(2, 2);
    board.PlayBlack(3, 1);
    board.PlayBlack(2, 0);

    CHECK(board.at(1, 1) == WHITE);
    CHECK(board.at(2, 1) == WHITE);

    CHECK(MoveOk(board.PlayBlack(1, 0)));

    CHECK(board.at(1, 1) == EMPTY);
    CHECK(board.at(2, 1) == EMPTY);
  }

  SUBCASE("KoCannotRecaptureImmediately") {
    game::Board board;

    board.PlayBlack(2, 1);
    board.PlayWhite(2, 2);
    board.PlayBlack(3, 2);
    board.PlayWhite(3, 3);
    board.PlayBlack(2, 3);
    board.PlayWhite(2, 4);
    board.PlayBlack(1, 2);
    board.PlayWhite(1, 3);
    board.PlayWhite(2, 2);

    CHECK(board.at(2, 3) == EMPTY);
    CHECK_FALSE(MoveOk(board.PlayBlack(2, 3)));
  }

  SUBCASE("SendTwoReturnOne") {
    game::Board board;

    board.PlayBlack(2, 1);
    board.PlayWhite(0, 1);
    board.PlayBlack(1, 0);
    board.PlayWhite(1, 1);
    board.PlayBlack(3, 0);

    // send two
    board.PlayBlack(0, 0);
    board.PlayWhite(2, 0);

    CHECK(board.at(0, 0) == EMPTY);
    CHECK(board.at(1, 0) == EMPTY);
    CHECK_FALSE(MoveOk(board.PlayBlack(1, 0)));
  }

  // . o . o
  // o o o o
  SUBCASE("CannotMoveInPARegion") {
    game::Board board;

    board.Pass(WHITE);
    board.PlayBlack(0, 1);
    board.PlayBlack(0, 3);
    board.PlayBlack(1, 0);
    board.PlayBlack(1, 1);
    board.PlayBlack(1, 2);
    board.Pass(BLACK);
    board.PlayBlack(1, 3);

    CHECK(MoveOk(board.PlayMoveDry(Loc{0, 0}, BLACK)));
    CHECK(MoveOk(board.PlayMoveDry(Loc{0, 2}, BLACK)));

    board.Pass(BLACK);

    CHECK_FALSE(MoveOk(board.PlayBlack(0, 0)));
    CHECK_FALSE(MoveOk(board.PlayBlack(0, 2)));
  }

  // o o .
  // x x .
  // . . .
  SUBCASE("GroupsInAtariCornerBlack") {
    game::Board board;
    board.PlayBlack(0, 0);
    board.PlayWhite(1, 0);
    board.PlayBlack(0, 1);
    board.PlayWhite(1, 1);

    Board::BoardData in_atari = board.GetStonesInAtari();
    CHECK(in_atari[Loc{0, 0}] == BLACK);
    CHECK(in_atari[Loc{0, 1}] == BLACK);
    CHECK(in_atari[Loc{1, 0}] == EMPTY);
    CHECK(in_atari[Loc{1, 1}] == EMPTY);
  }

  // x x .
  // o o .
  // . . .
  SUBCASE("GroupsInAtariCornerWhite") {
    game::Board board;
    board.PlayWhite(0, 0);
    board.PlayBlack(1, 0);
    board.PlayWhite(0, 1);
    board.PlayBlack(1, 1);

    Board::BoardData in_atari = board.GetStonesInAtari();
    CHECK(in_atari[Loc{0, 0}] == WHITE);
    CHECK(in_atari[Loc{0, 1}] == WHITE);
    CHECK(in_atari[Loc{1, 0}] == EMPTY);
    CHECK(in_atari[Loc{1, 1}] == EMPTY);
  }

  // . o o x . x o
  // . x x . . o .
  // . . . x o . .
  // . o x o x . .
  // . o x o o x .
  // . . o x o x .
  // . . o x o x .
  SUBCASE("GroupsInAtariSidesCenterMixed") {
    game::Board board;
    board.PlayBlack(0, 1);
    board.PlayWhite(1, 1);
    board.PlayBlack(0, 2);
    board.PlayWhite(1, 2);
    board.PlayWhite(0, 3);

    board.PlayBlack(0, 6);
    board.PlayWhite(0, 5);
    board.PlayBlack(1, 5);

    board.PlayWhite(2, 3);
    board.PlayWhite(3, 2);
    board.PlayWhite(3, 4);
    board.PlayWhite(4, 2);
    board.PlayWhite(4, 5);
    board.PlayWhite(5, 3);
    board.PlayWhite(5, 5);
    board.PlayWhite(6, 3);
    board.PlayWhite(6, 5);

    board.PlayBlack(2, 4);
    board.PlayBlack(3, 1);
    board.PlayBlack(3, 3);
    board.PlayBlack(4, 1);
    board.PlayBlack(4, 3);
    board.PlayBlack(4, 4);
    board.PlayBlack(5, 2);
    board.PlayBlack(5, 4);
    board.PlayBlack(6, 2);
    board.PlayBlack(6, 4);

    Board::BoardData in_atari = board.GetStonesInAtari();
    CHECK(in_atari[Loc{0, 1}] == BLACK);
    CHECK(in_atari[Loc{0, 2}] == BLACK);
    CHECK(in_atari[Loc{0, 3}] == EMPTY);
    CHECK(in_atari[Loc{0, 5}] == WHITE);
    CHECK(in_atari[Loc{0, 6}] == EMPTY);
    CHECK(in_atari[Loc{1, 1}] == EMPTY);
    CHECK(in_atari[Loc{1, 2}] == EMPTY);
    CHECK(in_atari[Loc{1, 5}] == EMPTY);
    CHECK(in_atari[Loc{2, 3}] == EMPTY);
    CHECK(in_atari[Loc{2, 4}] == EMPTY);
    CHECK(in_atari[Loc{3, 1}] == EMPTY);
    CHECK(in_atari[Loc{3, 2}] == WHITE);
    CHECK(in_atari[Loc{3, 3}] == BLACK);
    CHECK(in_atari[Loc{3, 4}] == WHITE);
    CHECK(in_atari[Loc{4, 1}] == EMPTY);
    CHECK(in_atari[Loc{4, 2}] == WHITE);
    CHECK(in_atari[Loc{4, 3}] == BLACK);
    CHECK(in_atari[Loc{4, 4}] == BLACK);
    CHECK(in_atari[Loc{4, 5}] == EMPTY);
    CHECK(in_atari[Loc{5, 2}] == EMPTY);
    CHECK(in_atari[Loc{5, 3}] == WHITE);
    CHECK(in_atari[Loc{5, 4}] == BLACK);
    CHECK(in_atari[Loc{5, 5}] == EMPTY);
    CHECK(in_atari[Loc{6, 2}] == EMPTY);
    CHECK(in_atari[Loc{6, 3}] == WHITE);
    CHECK(in_atari[Loc{6, 4}] == BLACK);
    CHECK(in_atari[Loc{6, 5}] == EMPTY);
  }

  // . x x o . .
  // . . o . . .
  // . . . . . .
  // . . . x o .
  // . o x x o .
  // . . o o . .
  SUBCASE("GroupsWith2Liberties") {
    game::Board board;
    board.PlayBlack(0, 3);
    board.PlayBlack(1, 2);
    board.PlayWhite(0, 1);
    board.PlayWhite(0, 2);

    board.PlayBlack(3, 4);
    board.PlayBlack(4, 4);
    board.PlayBlack(5, 3);
    board.PlayBlack(5, 2);
    board.PlayBlack(4, 1);

    board.PlayWhite(3, 3);
    board.PlayWhite(4, 3);
    board.PlayWhite(4, 2);

    Board::BoardData two_liberties = board.GetStonesWithLiberties(2);
    CHECK(two_liberties[Loc{0, 1}] == WHITE);
    CHECK(two_liberties[Loc{0, 2}] == WHITE);
    CHECK(two_liberties[Loc{0, 3}] == BLACK);
    CHECK(two_liberties[Loc{1, 2}] == EMPTY);

    CHECK(two_liberties[Loc{3, 3}] == WHITE);
    CHECK(two_liberties[Loc{3, 4}] == EMPTY);
    CHECK(two_liberties[Loc{4, 1}] == EMPTY);
    CHECK(two_liberties[Loc{4, 2}] == WHITE);
    CHECK(two_liberties[Loc{4, 3}] == WHITE);
    CHECK(two_liberties[Loc{4, 4}] == EMPTY);
    CHECK(two_liberties[Loc{5, 2}] == EMPTY);
    CHECK(two_liberties[Loc{5, 3}] == EMPTY);
  }

  // . o o x . .
  // . . . . . .
  // . . . . o .
  // . . o o x .
  // . . x x o .
  // . . . . . .
  SUBCASE("GroupsWith3Liberties") {
    game::Board board;
    board.PlayBlack(0, 1);
    board.PlayBlack(0, 2);
    board.PlayWhite(0, 3);

    board.PlayBlack(2, 4);
    board.PlayBlack(3, 2);
    board.PlayBlack(3, 3);
    board.PlayBlack(4, 4);

    board.PlayWhite(3, 4);
    board.PlayWhite(4, 3);
    board.PlayWhite(4, 2);

    Board::BoardData three_liberties = board.GetStonesWithLiberties(3);
    CHECK(three_liberties[Loc{0, 1}] == BLACK);
    CHECK(three_liberties[Loc{0, 2}] == BLACK);
    CHECK(three_liberties[Loc{0, 3}] == EMPTY);

    CHECK(three_liberties[Loc{2, 4}] == BLACK);
    CHECK(three_liberties[Loc{3, 2}] == BLACK);
    CHECK(three_liberties[Loc{3, 3}] == BLACK);
    CHECK(three_liberties[Loc{4, 4}] == EMPTY);
    CHECK(three_liberties[Loc{3, 4}] == EMPTY);
    CHECK(three_liberties[Loc{4, 3}] == WHITE);
    CHECK(three_liberties[Loc{4, 2}] == WHITE);
  }
}

TEST_CASE("GroupTrackerTest") {
  GroupTracker group_tracker;

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
    groupid gid2 = group_tracker.CoalesceGroups(Loc{3, 4});

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
    groupid final_gid = group_tracker.CoalesceGroups(Loc{3, 3});
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
    groupid final_gid = group_tracker.CoalesceGroups(Loc{3, 3});
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

    group_tracker.NewGroup(Loc{0, 1}, WHITE);
    CHECK(group_tracker.LibertiesForGroup(gid0) == 5);
  }
}

bool PaRegionsMatch(GroupTracker& group_tracker,
                    absl::flat_hash_set<Loc>&& region, int color) {
  for (int i = 0; i < BOARD_LEN; ++i) {
    for (int j = 0; j < BOARD_LEN; ++j) {
      bool loc_pass_alive = group_tracker.IsPassAliveForColor(Loc{i, j}, color);
      if (loc_pass_alive && !region.contains(Loc{i, j})) {
        return false;
      }

      region.erase(Loc{i, j});
    }
  }

  if (!region.empty()) {
    return false;
  }

  return true;
}

TEST_CASE("PassAliveTest") {
  GroupTracker group_tracker;

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
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
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

    group_tracker.NewGroup(Loc{0, 0}, WHITE);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {{0, 0}, {0, 1}, {0, 2}, {0, 3},
                                          {0, 4}, {1, 0}, {1, 1}, {1, 2},
                                          {1, 3}, {1, 4}};
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
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
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
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
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
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

    group_tracker.NewGroup(Loc{3, 2}, WHITE);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {
        {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 0}, {2, 1}, {2, 2},
        {2, 3}, {2, 4}, {2, 5}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4},
        {3, 5}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {5, 1}, {5, 2}, {5, 3}};
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
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
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
  }

  // . o . o
  // o . o o
  SUBCASE("NonPA") {
    groupid gid = group_tracker.NewGroup(Loc{0, 3}, BLACK);
    group_tracker.AddToGroup(Loc{1, 2}, gid);
    group_tracker.AddToGroup(Loc{1, 3}, gid);

    group_tracker.NewGroup(Loc{0, 1}, BLACK);
    group_tracker.NewGroup(Loc{1, 0}, BLACK);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {};
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
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

    group_tracker.NewGroup(Loc{0, 4}, BLACK);
    group_tracker.NewGroup(Loc{1, 5}, BLACK);
    groupid gid3 = group_tracker.NewGroup(Loc{0, 6}, BLACK);
    group_tracker.AddToGroup(Loc{0, 7}, gid3);
    group_tracker.AddToGroup(Loc{1, 7}, gid3);
    group_tracker.AddToGroup(Loc{2, 7}, gid3);
    group_tracker.AddToGroup(Loc{2, 6}, gid3);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {1, 0},
        {1, 1}, {1, 2}, {1, 3}, {1, 5}, {1, 6}, {1, 7}, {2, 6}, {2, 7}};
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
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
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region_black), BLACK));
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region_white), WHITE));
  }

  // . . . o .
  // o o o . .
  // . . . o .
  // o o . o .
  // o . o o .
  // . o . . .
  // o o . . .
  SUBCASE("AlmostPA") {
    groupid gid = group_tracker.NewGroup(Loc{1, 0}, BLACK);
    group_tracker.AddToGroup(Loc{1, 1}, gid);
    group_tracker.AddToGroup(Loc{1, 2}, gid);

    groupid gid1 = group_tracker.NewGroup(Loc{2, 3}, BLACK);
    group_tracker.AddToGroup(Loc{3, 3}, gid1);
    group_tracker.AddToGroup(Loc{4, 3}, gid1);
    group_tracker.AddToGroup(Loc{4, 2}, gid1);

    groupid gid2 = group_tracker.NewGroup(Loc{3, 0}, BLACK);
    group_tracker.AddToGroup(Loc{3, 1}, gid2);
    group_tracker.AddToGroup(Loc{4, 0}, gid2);

    groupid gid3 = group_tracker.NewGroup(Loc{5, 1}, BLACK);
    group_tracker.AddToGroup(Loc{6, 0}, gid3);
    group_tracker.AddToGroup(Loc{6, 1}, gid3);

    group_tracker.NewGroup(Loc{0, 3}, BLACK);

    group_tracker.CalculatePassAliveRegionForColor(BLACK);

    absl::flat_hash_set<Loc> pa_region = {};
    CHECK(PaRegionsMatch(group_tracker, std::move(pa_region), BLACK));
  }

  SUBCASE("CleanupBug") {
    const std::string s =
        "OOOOO.OOOOXXXX.XXXO"
        "XOXO.O.OXXXOX.XXXOO"
        "XOXO..OOOXXOXXOOO.O"
        "XXXO..OXXXXOOO.O.OO"
        "X.XOOOOOOOO.O.OOOO."
        ".XOXXO...O...O..O.O"
        "XOOOX.O.OOOOO.OOXOO"
        "XXXOOO.OOX.OOOOOXXO"
        ".XXOO.O.OOOXXOOXX.X"
        "XOOOXO.O.OOXXXOOXX."
        "XXOXXXOOOXXXXXXXX.X"
        "XOOXOOOXOOOOOXX.OX."
        "XXXXXXXXOXOOX.XOXXX"
        "XOOXXOOOOXXXX.XX..X"
        "XOXX.XXOOX.X.X.XXXX"
        "OOOXXXOOOX..X.X+..."
        "OOOOXXOXXX.XXXXXXX."
        "OOOOOXOOXX...XXXX.X"
        "OO.XXXXOOX.XXX.XXXX";
    auto board = ParseBoard(s);
    CHECK(board.IsAllPassAlive());
  }
}

bool OwnershipRegionsMatch(
    const std::array<Color, BOARD_LEN * BOARD_LEN>& ownership,
    absl::flat_hash_set<Loc>&& bregion, absl::flat_hash_set<Loc>&& wregion) {
  for (int i = 0; i < BOARD_LEN; ++i) {
    for (int j = 0; j < BOARD_LEN; ++j) {
      int idx = i * BOARD_LEN + j;
      if (ownership[idx] == BLACK) {
        if (!bregion.contains(Loc{i, j})) {
          std::cerr << "Black Region does not contain " << Loc{i, j} << "\n";
          return false;
        }

        bregion.erase(Loc{i, j});
      } else if (ownership[idx] == WHITE) {
        if (!wregion.contains(Loc{i, j})) {
          std::cerr << "White Region does not contain " << Loc{i, j} << "\n";
          return false;
        }

        wregion.erase(Loc{i, j});
      } else if (ownership[idx] != EMPTY) {
        std::cerr << "Ownership contains unknown node " << Loc{i, j} << "\n";
        return false;
      }
    }
  }

  if (!bregion.empty() || !wregion.empty()) {
    std::cerr << "Stragglers found";
    return false;
  }

  return true;
}

TEST_CASE("ScoreTest") {
  SUBCASE("Empty") {
    game::Board board;

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 0);
    CHECK(scores.white_score == 7.5);
    CHECK(OwnershipRegionsMatch(scores.ownership, {}, {}));
  }

  // . . . o .
  // o o o x .
  SUBCASE("BlackCorner") {
    game::Board board;

    board.PlayBlack(1, 0);
    board.PlayBlack(1, 1);
    board.PlayBlack(1, 2);
    board.PlayBlack(0, 3);

    board.PlayWhite(1, 3);

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 7);
    CHECK(scores.white_score == 8.5);
    CHECK(OwnershipRegionsMatch(scores.ownership,
                                {Loc{0, 0}, Loc{0, 1}, Loc{0, 2}, Loc{0, 3},
                                 Loc{1, 0}, Loc{1, 1}, Loc{1, 2}},
                                {Loc{1, 3}}));
  }

  // . x . o .
  // o o o x .
  SUBCASE("BlackCornerWhiteStone") {
    game::Board board;

    board.PlayBlack(1, 0);
    board.PlayBlack(1, 1);
    board.PlayBlack(1, 2);
    board.PlayBlack(0, 3);

    board.PlayWhite(1, 3);
    board.PlayWhite(0, 1);

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 4);
    CHECK(scores.white_score == 9.5);
    CHECK(OwnershipRegionsMatch(scores.ownership,
                                {Loc{0, 3}, Loc{1, 0}, Loc{1, 1}, Loc{1, 2}},
                                {Loc{0, 1}, Loc{1, 3}}));
  }

  // . . . o
  // . . . o
  // . . . o
  // o o o x
  SUBCASE("BigBlackCorner") {
    game::Board board;

    board.PlayBlack(3, 0);
    board.PlayBlack(3, 1);
    board.PlayBlack(3, 2);
    board.PlayBlack(0, 3);
    board.PlayBlack(1, 3);
    board.PlayBlack(2, 3);

    board.PlayWhite(3, 3);

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 15);
    CHECK(scores.white_score == 8.5);
    CHECK(OwnershipRegionsMatch(
        scores.ownership,
        {Loc{0, 0}, Loc{0, 1}, Loc{0, 2}, Loc{0, 3}, Loc{1, 0}, Loc{1, 1},
         Loc{1, 2}, Loc{1, 3}, Loc{2, 0}, Loc{2, 1}, Loc{2, 2}, Loc{2, 3},
         Loc{3, 0}, Loc{3, 1}, Loc{3, 2}},
        {Loc{3, 3}}));
  }

  // . . . x
  // . . . x
  // . . . x
  // x x x o
  SUBCASE("BigWhiteCorner") {
    game::Board board;

    board.PlayWhite(3, 0);
    board.PlayWhite(3, 1);
    board.PlayWhite(3, 2);
    board.PlayWhite(0, 3);
    board.PlayWhite(1, 3);
    board.PlayWhite(2, 3);

    board.PlayBlack(3, 3);

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 1);
    CHECK(scores.white_score == 22.5);
    CHECK(OwnershipRegionsMatch(
        scores.ownership, {Loc{3, 3}},
        {Loc{0, 0}, Loc{0, 1}, Loc{0, 2}, Loc{0, 3}, Loc{1, 0}, Loc{1, 1},
         Loc{1, 2}, Loc{1, 3}, Loc{2, 0}, Loc{2, 1}, Loc{2, 2}, Loc{2, 3},
         Loc{3, 0}, Loc{3, 1}, Loc{3, 2}}));
  }

  // . x . x
  // . x . x
  // . x . x
  // x x x o
  SUBCASE("MultipleRegions") {
    game::Board board;

    board.PlayWhite(3, 0);
    board.PlayWhite(3, 1);
    board.PlayWhite(3, 2);
    board.PlayWhite(0, 3);
    board.PlayWhite(1, 3);
    board.PlayWhite(2, 3);

    board.PlayBlack(3, 3);

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 1);
    CHECK(scores.white_score == 22.5);
    CHECK(OwnershipRegionsMatch(
        scores.ownership, {Loc{3, 3}},
        {Loc{0, 0}, Loc{0, 1}, Loc{0, 2}, Loc{0, 3}, Loc{1, 0}, Loc{1, 1},
         Loc{1, 2}, Loc{1, 3}, Loc{2, 0}, Loc{2, 1}, Loc{2, 2}, Loc{2, 3},
         Loc{3, 0}, Loc{3, 1}, Loc{3, 2}}));
  }

  // . . . . . . .
  // . o o o . . .
  // . o . o . . .
  // . o o x x x .
  // . . x . . . x
  // . . . x x x .
  SUBCASE("DifferentColorRegions") {
    game::Board board;

    board.PlayBlack(1, 1);
    board.PlayBlack(1, 2);
    board.PlayBlack(1, 3);
    board.PlayBlack(2, 1);
    board.PlayBlack(2, 3);
    board.PlayBlack(3, 1);
    board.PlayBlack(3, 2);

    board.PlayWhite(3, 3);
    board.PlayWhite(3, 4);
    board.PlayWhite(3, 5);
    board.PlayWhite(4, 2);
    board.PlayWhite(4, 6);
    board.PlayWhite(5, 3);
    board.PlayWhite(5, 4);
    board.PlayWhite(5, 5);

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 8);
    CHECK(scores.white_score == 18.5);
    CHECK(OwnershipRegionsMatch(
        scores.ownership,
        {Loc{1, 1}, Loc{1, 2}, Loc{1, 3}, Loc{2, 1}, Loc{2, 2}, Loc{2, 3},
         Loc{3, 1}, Loc{3, 2}},
        {Loc{3, 3}, Loc{3, 4}, Loc{3, 5}, Loc{4, 2}, Loc{4, 3}, Loc{4, 4},
         Loc{4, 5}, Loc{4, 6}, Loc{5, 3}, Loc{5, 4}, Loc{5, 5}}));
  }

  // . . . . . . .
  // . o o o . . .
  // . o . o . . .
  // . . o . o . .
  // . . o x o . .
  // . . o o o x .
  SUBCASE("PaRegionStones") {
    game::Board board;

    board.PlayBlack(1, 1);
    board.PlayBlack(1, 2);
    board.PlayBlack(1, 3);
    board.PlayBlack(2, 1);
    board.PlayBlack(2, 3);
    board.PlayBlack(3, 2);
    board.PlayBlack(3, 4);
    board.PlayBlack(4, 2);
    board.PlayBlack(4, 4);
    board.PlayBlack(5, 2);
    board.PlayBlack(5, 3);
    board.PlayBlack(5, 4);

    board.PlayWhite(4, 3);
    board.PlayWhite(5, 5);

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 15);
    CHECK(scores.white_score == 8.5);
    CHECK(OwnershipRegionsMatch(
        scores.ownership,
        {Loc{1, 1}, Loc{1, 2}, Loc{1, 3}, Loc{2, 1}, Loc{2, 2}, Loc{2, 3},
         Loc{3, 2}, Loc{3, 3}, Loc{3, 4}, Loc{4, 2}, Loc{4, 3}, Loc{4, 4},
         Loc{5, 2}, Loc{5, 3}, Loc{5, 4}},
        {Loc{5, 5}}));
  }

  // x x x . o . o
  // o o o o . o o
  // . . . o x x o
  // . . . o o o x
  SUBCASE("PaRegionStoneGroup") {
    game::Board board;

    board.PlayBlack(0, 4);
    board.PlayBlack(0, 6);
    board.PlayBlack(1, 0);
    board.PlayBlack(1, 1);
    board.PlayBlack(1, 2);
    board.PlayBlack(1, 3);
    board.PlayBlack(1, 5);
    board.PlayBlack(1, 6);
    board.PlayBlack(2, 3);
    board.PlayBlack(2, 6);
    board.PlayBlack(3, 3);
    board.PlayBlack(3, 4);
    board.PlayBlack(3, 5);

    board.PlayWhite(0, 0);
    board.PlayWhite(0, 1);
    board.PlayWhite(0, 2);
    board.PlayWhite(2, 4);
    board.PlayWhite(2, 5);
    board.PlayWhite(3, 6);

    board.Pass(BLACK);
    board.Pass(WHITE);

    Scores scores = board.GetScores();
    CHECK(scores.black_score == 21);
    CHECK(scores.white_score == 8.5);
    CHECK(OwnershipRegionsMatch(
        scores.ownership,
        {Loc{0, 0}, Loc{0, 1}, Loc{0, 2}, Loc{0, 3}, Loc{0, 4}, Loc{0, 5},
         Loc{0, 6}, Loc{1, 0}, Loc{1, 1}, Loc{1, 2}, Loc{1, 3}, Loc{1, 4},
         Loc{1, 5}, Loc{1, 6}, Loc{2, 3}, Loc{2, 4}, Loc{2, 5}, Loc{2, 6},
         Loc{3, 3}, Loc{3, 4}, Loc{3, 5}},
        {Loc{3, 6}}));
  }
}

#if 0
auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");
#endif
TEST_CASE("BoardDSLTest") {
  using game::BoardToDSL;
  using game::ParseBoardDSL;
  using game::ParseBoardGrid;

  SUBCASE("SimplePattern") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . x . x o . . . . . . . . . . . . .
      . . x x o . o . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    CHECK(board.at(0, 2) == BLACK);
    CHECK(board.at(0, 4) == BLACK);
    CHECK(board.at(1, 2) == BLACK);
    CHECK(board.at(1, 3) == BLACK);
    CHECK(board.at(0, 5) == WHITE);
    CHECK(board.at(1, 4) == WHITE);
    CHECK(board.at(1, 6) == WHITE);
  }

  SUBCASE("#879") {
    // Real game position from p3achygo vs katago
    // Testing P17 (2, 14) and T19 (0, 18) playability
    // Note: Y-coordinates are flipped (row 0 is displayed as row 19)
    auto board = ParseBoardDSL(R"(
      . . . . . . . x x o o o o x . x x x .
      o x x . . . x . x o x x x x x x x o x
      . o x x . x x x o o o o x o . x o o o
      . o o x x o o o x x x o o o x x o x o
      o o x o . o . x . x . x o o o o o x x
      . o x . o x . . . x x . x x x x x x .
      o x x x x o o o o o x x o o x o o o .
      . o x o x x o . x x o o . o x o x o .
      . o o . o o o . o o . o o o o o x . .
      . . . o . . . . . x o . o x o x . x .
      . o . . . . . . . x o o x x o x x . .
      o x x x o . . . . . x x o x x o o x .
      o x . x o . . . o . x o o x . . . . .
      . x x o o . x x o . o x . x . o o . .
      x . x x x x o o x . x o o x . o x . .
      x x o . . o o o x x x o x x o . x . .
      x o o o . o o x x o x x o x o x . . .
      o x o . . . . . x o o o o o x . x . .
      . . . . . . . . . . . . . . . x . . .
    )");

    // P17 is at (2, 14) - should be playable for BLACK but NOT for WHITE
    // (self-capture)
    CHECK(MoveOk(board.PlayMoveDry(Loc{2, 14}, BLACK)));
    CHECK_FALSE(MoveOk(board.PlayMoveDry(Loc{2, 14}, WHITE)));

    // T19 is at (0, 18) - should be playable for both colors
    CHECK(MoveOk(board.PlayMoveDry(Loc{0, 18}, BLACK)));
    CHECK(MoveOk(board.PlayMoveDry(Loc{0, 18}, WHITE)));

    // Check that the chain at (2, 18) has exactly 1 liberty (is in atari)
    Board::BoardData in_atari = board.GetStonesInAtari();
    CHECK(in_atari[Loc{2, 18}] != EMPTY);  // Stone at (2,18) is in atari

    // Check that the chain at (0, 17) has exactly 3 liberties
    Board::BoardData three_libs = board.GetStonesWithLiberties(3);
    CHECK(three_libs[Loc{0, 17}] != EMPTY);  // Stone at (0,17) has 3 liberties

    // Run Benson solver (pass-alive calculation)
    bool all_pass_alive = board.IsAllPassAlive();

    CHECK(MoveOk(board.PlayMoveDry(Loc{2, 14}, BLACK)));
    CHECK_FALSE(MoveOk(board.PlayMoveDry(Loc{2, 14}, WHITE)));

    // T19 is at (0, 18) - should be playable for both colors
    CHECK(MoveOk(board.PlayMoveDry(Loc{0, 18}, BLACK)));
    CHECK(MoveOk(board.PlayMoveDry(Loc{0, 18}, WHITE)));
  }

  SUBCASE("#276") {
    // Real game position from p3achygo vs katago
    // Testing P17 (2, 14) and T19 (0, 18) playability
    // Note: Y-coordinates are flipped (row 0 is displayed as row 19)
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . o . o . . . . . . . . . . . . .
      . . o . . . . . . . o x x x x x x x .
      . . o x . o . . . o x o o x o x o . o
      . . x x o x o . . . x x o o . o o o x
      . x . o o . x x x . x x x o o o . x .
      . o o o x x x . . x x . x o . x o . .
      . . x x o o o x x . . x x o . x o o .
      . . x o x . . x x . x o o . . x o x .
      . . . o x o o x . . x . . o o x x x x
      . . . o x . . x x . x . x x o o x o o
      . . . o x o o . o x o . . x x o o x .
      . . o x . x o o . . . . o o o x o x x
      . . o x . . . . . o o . . o x x o x x
      . . o x . . . . x x . . o x x o x o .
      . o . o x . . . . . . x . x . o x o o
      . . o o x . . . . . . . . . x x x o o
      . . . . . . . . . . . . . . . x x x x
    )");

    CHECK(MoveOk(board.PlayMoveDry(Loc{15, 18}, BLACK)));
    CHECK_FALSE(MoveOk(board.PlayMoveDry(Loc{15, 18}, WHITE)));

    Board::BoardData in_atari = board.GetStonesInAtari();
    CHECK(in_atari[Loc{16, 18}] != EMPTY);

    Board::BoardData two_libs = board.GetStonesWithLiberties(2);
    CHECK(two_libs[Loc{14, 18}] != EMPTY);

    bool all_pass_alive = board.IsAllPassAlive();

    CHECK(MoveOk(board.PlayMoveDry(Loc{15, 18}, BLACK)));
    CHECK_FALSE(MoveOk(board.PlayMoveDry(Loc{15, 18}, WHITE)));
  }

  SUBCASE("#589") {
    auto board = ParseBoardDSL(R"(
      . o o o o x x . x . x x x . . . . . .
      x o x o . o x x x . x o x o o o . . .
      x x x o o o x o o x x o o x . x . . .
      o o o x x x . . . o o o . x . . . . .
      x o o . x . x o o x x o o o . . o . .
      . x o . x o o . . x . x x . . o x . .
      . x x x . . . o . x x . . . . . o . .
      o x . . o . . . . . o o x . . x . . .
      . o . o . . . . . o . . x . . . . . .
      . . . . . . . . . . x . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . o . . . . . . .
      . . x x . . . . . . . x o o x . . . .
      . o o x . . . . . o o x o x x x x . .
      . . x . . . . o o . x o o o x o x . .
      . . o x x . o . o . x x . o o o x . .
      . o x x o o . o x x . . . o x x . . .
      . . . . o x x x . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    CHECK(MoveOk(board.PlayMoveDry(Loc{0, 0}, WHITE)));
    bool all_pass_alive = board.IsAllPassAlive();
    CHECK(MoveOk(board.PlayMoveDry(Loc{0, 0}, WHITE)));
  }

  SUBCASE("#20") {
    auto board = ParseBoardDSL(R"(
      . x . o o x x x x o . . . . . . . . .
      . o o . o o . . o o . . . . . . . . .
      o . o o x x x x o . . . o o o o . . .
      x o o x x o o x o o o o x o x x o . .
      x x o x . o . o o x o x x x . x o o .
      . x x x . o o . o x o o o x . x x o o
      . . . . x x o x x x x o . x x . . x o
      . . . . . . x . . . x o o o x x x . x
      . o o x . . . . . . x o x o x . x . .
      . x x . x . . . . x o o x o o x x x x
      . x o x o . . . . x o . x o . o o o x
      . o o x x . . . . x o . x x o o o . o
      . o . o . x . . x o o x x . x x x o .
      x . o o o x . . x o o o o x . x o . o
      o o x x x o x x x x . x x x o x o o .
      o x x x o o o o o x . x o o o x o . .
      . o x x x o . o x x x x . o x o x . .
      o o x . x x o x o x o x o o x o o . .
      . o o o x x x x o o o o o . x o . . .
    )");

    CHECK(MoveOk(board.PlayMoveDry(Loc{16, 6}, BLACK)));
    CHECK(MoveOk(board.PlayMoveDry(Loc{17, 3}, WHITE)));
    bool all_pass_alive = board.IsAllPassAlive();
    CHECK(MoveOk(board.PlayMoveDry(Loc{16, 6}, BLACK)));
    CHECK(MoveOk(board.PlayMoveDry(Loc{17, 3}, WHITE)));
  }
}

TEST_CASE("LadderTest") {
  using game::BoardToDSL;
  using game::ParseBoardDSL;
  using game::ParseBoardGrid;

  SUBCASE("SimpleLadder") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . x . . . . . . . . . . . . . . . . .
      . x o x . . . . . . . . . . . . . . .
      . . x . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("SimpleLadder took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{16, 2}, BOARD_LEN)] == WHITE);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("SimpleBrokenLadder") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . x . . . . . . . . . . . . . . . . .
      . x o x . . . . . . . . . . . . . . .
      . . x . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("SimpleBrokenLadder took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{16, 2}, BOARD_LEN)] == EMPTY);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("AtariBrokenLadder") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . x . . . . . . . . . . . . . . . . .
      . x o x . . . . . . . . . . . . . . .
      . . x . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;

    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("AtariBrokenLadder took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{16, 2}, BOARD_LEN)] == EMPTY);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("DirectionLadder") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . o . . . . . . . . . . . .
      . . . . o . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . x . . . x x x . . . . . . . . . .
      . x o o x . . . . . . . . . . . . . .
      . . x x . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("DirectionLadder took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{5, 2}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{5, 3}, BOARD_LEN)] == WHITE);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("DoubleLadder") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . o x o . .
      . . . . . . . . . . . . . . . . o . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . x . . . . . . . . . . . . . . . . .
      . x o x . . . . . . . . . . . . . . .
      . . x . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("DoubleLadder took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{16, 2}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{8, 15}, BOARD_LEN)] == BLACK);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("NoLadder") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . o x o . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . x o x . . . . . . . . . . . . . . .
      . . x . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("NoLadder took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{16, 2}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{8, 15}, BOARD_LEN)] == EMPTY);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("EdgeLadder") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . o o . . . . . . . . . .
      . . . . . . o x x . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("EdgeLadder took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{18, 7}, BOARD_LEN)] == BLACK);
    CHECK(laddered_stones[AsIndex(Loc{18, 8}, BOARD_LEN)] == BLACK);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("LadderSenseis1") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . x o x . . . . . . . . . . . .
      . x x x o x . . . . . . . . . . . . .
      . x o o o . . . . + . . . . . + . . .
      . o o . . . . . . . . . . . . . . . .
      . . x . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . o . . . . . .
      . . . . . . . . . . o x x o . . . . .
      . . . + . . . . . + . o o . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("LadderSenseis1 took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{8, 11}, BOARD_LEN)] == BLACK);
    CHECK(laddered_stones[AsIndex(Loc{8, 12}, BOARD_LEN)] == BLACK);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("LadderSenseis3") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . x x . . . . . .
      . . . x . . . . . + x o o x . + . . .
      . . . . . . . . . . . . x . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . o . . . . . . . . . . . . . .
      . . x . x . . . . . . . . . . . . . .
      . . . . x . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . o o . .
      . . . + . . . . . + . . . . x + x x .
      . . . . . . . . . x . . o . . x . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("LadderSenseis3 took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{3, 11}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{3, 12}, BOARD_LEN)] == WHITE);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("LadderOdnihs") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . o . . . . o . . . . . . . . . . .
      . o . o x x o x o . . . . . . . . . .
      . . o x . . x x x o . . o o o o x . .
      . o . x x x x x . + o o x x x o x . .
      . . x . . . . . . . . . . . o x . . .
      . . . . . . . . . . . . x . x x . . .
      . . . . . . . . . . . . . o . . . . .
      . . . . . . . . . . . . . . x x . . .
      . . . . . . . . . . . . . . o o . . .
      . . . + . . . . . + . . . . x o . . .
      . . . . . . . . . . . . . o x o . . .
      . . . . . . . . . . . . . . o . . . .
      . . . . . . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("LadderOdnihs took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{9, 14}, BOARD_LEN)] == BLACK);
    CHECK(laddered_stones[AsIndex(Loc{10, 14}, BOARD_LEN)] == BLACK);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("BrokenLadderOdnihs") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . o . . . . o . . . . . . . . . . .
      . o . o x x o x o . . . . . . . . . .
      . . o x . . x x x o . . o o o o x . .
      . o . x x x x x . + o o x x x o x . .
      . . x . . . . . . . . . . x . x . . .
      . . . . . . . . . . . . x . x x . . .
      . . . . . . . . . . . . . o . . . . .
      . . . . . . . . . . . . . . x x . . .
      . . . . . . . . . . . . . . o o . . .
      . . . + . . . . . + . . . . x o . . .
      . . . . . . . . . . . . . o x o . . .
      . . . . . . . . . . . . . . o . . . .
      . . . . . . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("BrokenLadderOdnihs took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{9, 14}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{10, 14}, BOARD_LEN)] == EMPTY);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("BrokenLadderSenseis1") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . x
      . . . . . . . . . . . . . . . . . . x
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . o
      . . . . . . . . . o o . . . . . . . .
      . . . + . . . . o x x o x . . + . . .
      . . . . . . . . o o x . x x o . . . .
      . . . . . . . . . . o o o o . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("BrokenLadderSenseis1 took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{16, 9}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{16, 10}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{17, 10}, BOARD_LEN)] == EMPTY);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("LadderSenseis2") {
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . x . . . . . .
      . . . . . . . . . . . x o x . . . . .
      . . . . . . . . o . x o . x . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . o o . . . . . . .
      . . . o . . . . . x . x . . . . . . .
      . . o + . . . . . + . x . . . + . . .
      . . . . x . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("LadderSenseis2 took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{7, 12}, BOARD_LEN)] == WHITE);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("NakayamaLadder") {
    auto board = ParseBoardDSL(R"(
      . . . . o . . . . . . . . . . . . . .
      o . . . . . . . . . . . . . . . . . o
      . . . . . . . . . . . . . . . . . . o
      . . . o o x o . . + . . . . . + . . .
      o . . . o x o o o o . o . . . . . o .
      o x x x x x x x x x o . . . . . . . .
      x o o o . x . . . . x . . . . . . . o
      x . o o o x . . . . x o . . . . . . .
      x o o o o x . . . . x . . . . . . . .
      o x x x x x x x x x o . . . . + . . o
      o o . . . x o . . o . . . . . . . . .
      . o . o . x . o . . . . . . . . . . .
      . . . . . . o . . . . . . . . . . . .
      . . . . o . . . . x . . . . . . . . .
      o . . . . x . . . x . . . x o . . . .
      . . . + . x . . . x . . . x . + . . .
      . . . . . x . . . x . . . x . . . . .
      . . . . o . x x x x x x x o . . . o .
      . o . . . x o o o o o o o . o . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("NakayamaLadder took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{18, 6}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{18, 7}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{18, 8}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{18, 9}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{18, 10}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{18, 11}, BOARD_LEN)] == WHITE);
    CHECK(laddered_stones[AsIndex(Loc{18, 12}, BOARD_LEN)] == WHITE);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("BrokenLadderSenseis1") {
    // Test a simple pattern using the DSL
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . x o x . . . . . . . . . . . .
      . x x x o x . . x . . . . . . . . . .
      . x o o o . . . . + . . . . . + . . .
      . o o . . . . . . . . . . . . . . . .
      . . x . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . o . . . . . .
      . . . . . . . . . . o x x o . . . . .
      . . . + . . . . . + . o o . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("BrokenLadderSenseis1 took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{8, 11}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{8, 12}, BOARD_LEN)] == EMPTY);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("NakayamaLadderBroken") {
    auto board = ParseBoardDSL(R"(
      . . . . o . . . . . . . . . . . . . .
      o . . . . . . . . . . . . . . . . . o
      . . . . . . . . . . . . . . . . . . o
      . . . o o x o . . + . . . . . + . . .
      o . . . o x o o o o . o . . . . . o .
      o x x x x x x x x x o . . . . . . . .
      x o o o . x . . . . x . . . . . . o o
      x . o o o x . . . . x o . . . . . . .
      x o o o o x . . . . x . . . . . . . .
      o x x x x x x x x x o . . . . + . . o
      o o . . . x o . . o . . . . . . . . .
      . o . o . x . o . . . . . . . . . . .
      . . . . . . o . . . . . . . . . . . .
      . . . . o . . . . x . . . . . . . . .
      o . . . . x . . . x . . . x o . . . .
      . . . + . x . . . x . . . x . + . . .
      . . . . . x . . . x . . . x . . . . .
      . . . . o . x x x x x x x o . . . o .
      . o . . . x o o o o o o o . o . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("NakayamaLadderBroken took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{18, 6}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{18, 7}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{18, 8}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{18, 9}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{18, 10}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{18, 11}, BOARD_LEN)] == EMPTY);
    CHECK(laddered_stones[AsIndex(Loc{18, 12}, BOARD_LEN)] == EMPTY);
    CHECK_EQ(board, board_copy);
  }

  SUBCASE("BrokenLadderSenseis2") {
    auto board = ParseBoardDSL(R"(
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . x . . . . . .
      . . . . . . . . . . . x o x . . . . .
      . . . . . . . . o . x o . x . . . . .
      . . . + . . . . . + . . . . . + . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . o . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . o o . . . . . . .
      . . . o . . . . . x . x . . . . . . .
      . . o + . . . . . + . x . . . + . . .
      . . . . x . o . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
      . . . . . . . . . . . . . . . . . . .
    )");

    auto board_copy = board;
    auto start = std::chrono::steady_clock::now();
    auto laddered_stones = board.GetLadderedStones();
    auto end = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    MESSAGE("BrokenLadderSenseis2 took " << us << " us\n");
    CHECK(laddered_stones[AsIndex(Loc{7, 12}, BOARD_LEN)] == EMPTY);
    CHECK_EQ(board, board_copy);
  }
}
}  // namespace game