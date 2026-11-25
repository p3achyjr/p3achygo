#include "cc/gtp/analyze.h"

#include <memory>
#include <string>
#include <vector>

#include "cc/constants/constants.h"
#include "cc/gtp/command.h"
#include "cc/mcts/tree.h"
// After absl headers, so CHECK is doctest's.
#include "cc/core/doctest_include.h"

namespace gtp {
namespace {

using mcts::TreeNode;

std::string Loc(int a) { return GtpValueString(game::AsLoc(a)); }

// Root with two visited children:
//   action 60: 3 visits, prior 0.25, child v_outcome -0.2 (mover wins 60%),
//              child score -4.5 (mover +4.5), one grandchild at 100.
//   action 0:  1 visit, prior 0.05, child v_outcome 0.6 (mover wins 20%).
// Action 5 has a prior but no visits.
struct Tree {
  TreeNode root{0};
  TreeNode a60{0}, a0{0}, g100{0};

  Tree() {
    a60.n = 3;
    a60.v_outcome = -0.2f;
    a60.v_outcome_var = 0.0f;
    a60.score = -4.5f;
    a60.children[100] = &g100;
    a60.child_visits[100] = 2;
    g100.n = 2;
    a0.n = 1;
    a0.v_outcome = 0.6f;
    a0.score = 2.0f;
    root.children[60] = &a60;
    root.child_visits[60] = 3;
    root.move_probs[60] = 0.25f;
    root.children[0] = &a0;
    root.child_visits[0] = 1;
    root.move_probs[0] = 0.05f;
    root.move_probs[5] = 0.5f;
    root.n = 5;
  }
};

TEST_CASE("ParseAnalyzeArgs") {
  SUBCASE("empty: defaults") {
    auto a = ParseAnalyzeArgs(false, {});
    REQUIRE(a.ok());
    CHECK(!a->color.has_value());
    CHECK(a->interval_cs == 100);
    CHECK(!a->kata);
  }
  SUBCASE("player and interval") {
    auto a = ParseAnalyzeArgs(true, {"B", "50"});
    REQUIRE(a.ok());
    CHECK(a->color == BLACK);
    CHECK(a->interval_cs == 50);
    CHECK(a->kata);
  }
  SUBCASE("interval only") {
    auto a = ParseAnalyzeArgs(false, {"10"});
    REQUIRE(a.ok());
    CHECK(!a->color.has_value());
    CHECK(a->interval_cs == 10);
  }
  SUBCASE("keywords") {
    auto a = ParseAnalyzeArgs(
        true, {"w", "interval", "20", "maxmoves", "3", "ownership", "true",
               "pvVisits", "true", "minmoves", "2"});
    REQUIRE(a.ok());
    CHECK(a->color == WHITE);
    CHECK(a->interval_cs == 20);
    CHECK(a->max_moves == 3);
    CHECK(a->ownership);
  }
  SUBCASE("rejected") {
    CHECK(!ParseAnalyzeArgs(true, {"10", "avoid", "b", "D4", "1"}).ok());
    CHECK(!ParseAnalyzeArgs(false, {"10", "ownership", "true"}).ok());
    CHECK(!ParseAnalyzeArgs(true, {"10", "maxmoves"}).ok());
    CHECK(!ParseAnalyzeArgs(true, {"0"}).ok());
    CHECK(!ParseAnalyzeArgs(true, {"10", "bogus", "1"}).ok());
  }
}

TEST_CASE("lz-analyze line: integer scaling, visit order, pv") {
  Tree t;
  AnalyzeArgs args;
  CHECK(AnalyzeLine(&t.root, args, {}) ==
        "info move " + Loc(60) + " visits 3 winrate 6000 prior 2500 lcb 6000" +
            " order 0 pv " + Loc(60) + " " + Loc(100) + " info move " + Loc(0) +
            " visits 1 winrate 2000 prior 500 lcb 0 order 1 pv " + Loc(0));
}

TEST_CASE("kata-analyze line: floats, score, ownership") {
  Tree t;
  AnalyzeArgs args;
  args.kata = true;
  args.ownership = true;
  args.max_moves = 1;
  CHECK(AnalyzeLine(&t.root, args, {0.5f, -1.0f}) ==
        "info move " + Loc(60) +
            " visits 3 winrate 0.600000 scoreMean 4.500000 scoreLead 4.500000"
            " prior 0.250000 lcb 0.600000 order 0 pv " +
            Loc(60) + " " + Loc(100) + " ownership 0.500000 -1.000000");
}

TEST_CASE("no visited children: empty line") {
  TreeNode root{0};
  root.move_probs[3] = 1.0f;
  CHECK(AnalyzeLine(&root, AnalyzeArgs{}, {}).empty());
}

}  // namespace
}  // namespace gtp
