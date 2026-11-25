#include "cc/gtp/pass_guard.h"

#include <memory>
#include <vector>

#include "cc/constants/constants.h"
#include "cc/core/doctest_include.h"
#include "cc/game/game.h"
#include "cc/mcts/tree.h"

namespace gtp {
namespace {

using mcts::TreeNode;
constexpr int kPass = constants::kPassMoveEncoding;

// A root whose children (action, visits, prior) each have one visit's worth of
// stats; the rest of the priors are 0.
struct Root {
  struct Kid {
    int a;
    int n;
    float prior;
  };
  TreeNode root{0};
  std::vector<std::unique_ptr<TreeNode>> kids;

  explicit Root(const std::vector<Kid>& ks) {
    int total = 1;
    for (const Kid& k : ks) {
      auto c = std::make_unique<TreeNode>(0);
      c->n = k.n;
      root.children[k.a] = c.get();
      root.child_visits[k.a] = k.n;
      root.move_probs[k.a] = k.prior;
      total += k.n;
      kids.push_back(std::move(c));
    }
    root.n = total;
  }
};

// The CGOS loss shape: a tiny search, one visit per child, pass with the
// highest prior.
Root OneVisitEach() {
  return Root({{kPass, 1, 0.9f}, {0, 1, 0.02f}, {60, 1, 0.05f}});
}

TEST_CASE("one-visit children: the LCB pick is pass, by prior") {
  Root r = OneVisitEach();
  CHECK(mcts::BestLcbAction(&r.root, 0.1f, [](int) { return true; }) == kPass);
}

TEST_CASE("pass that loses on the board count is replaced") {
  // Empty board, komi 7.5: Black's pass loses by 7.5.
  game::Game game(false, game::Board::Scoring::kTrompTaylor);
  Root r = OneVisitEach();
  CHECK(AvoidLosingPass(game, &r.root, BLACK, 0.1f, game::kPassLoc) ==
        game::AsLoc(60));
}

TEST_CASE("pass that wins on the board count is kept") {
  Root r = OneVisitEach();
  SUBCASE("White on an empty board (komi)") {
    game::Game game(false, game::Board::Scoring::kTrompTaylor);
    CHECK(AvoidLosingPass(game, &r.root, WHITE, 0.1f, game::kPassLoc) ==
          game::kPassLoc);
  }
  SUBCASE("Black owning the whole board") {
    game::Game game(false, game::Board::Scoring::kTrompTaylor);
    game.PlayMove(game::Loc{9, 9}, BLACK);
    CHECK(AvoidLosingPass(game, &r.root, BLACK, 0.1f, game::kPassLoc) ==
          game::kPassLoc);
  }
}

TEST_CASE("a lone opponent stone makes the empty region neutral") {
  // Tromp-Taylor counts every stone: with one White stone on the board, the
  // empty region touches both colors and Black's area is its one stone.
  game::Game game(false, game::Board::Scoring::kTrompTaylor);
  game.PlayMove(game::Loc{9, 9}, BLACK);
  game.PlayMove(game::Loc{15, 15}, WHITE);
  Root r = OneVisitEach();
  CHECK(AvoidLosingPass(game, &r.root, BLACK, 0.1f, game::kPassLoc) ==
        game::AsLoc(60));
}

TEST_CASE("non-pass moves are returned unchanged") {
  game::Game game(false, game::Board::Scoring::kTrompTaylor);
  Root r = OneVisitEach();
  CHECK(AvoidLosingPass(game, &r.root, BLACK, 0.1f, game::AsLoc(0)) ==
        game::AsLoc(0));
}

TEST_CASE("only pass searched: falls back to the highest-prior legal move") {
  game::Game game(false, game::Board::Scoring::kTrompTaylor);
  Root r({{kPass, 20, 0.9f}});
  r.root.move_probs[100] = 0.08f;
  r.root.move_probs[200] = 0.02f;
  CHECK(AvoidLosingPass(game, &r.root, BLACK, 0.1f, game::kPassLoc) ==
        game::AsLoc(100));
}

TEST_CASE("an occupied point is never returned") {
  game::Game game(false, game::Board::Scoring::kTrompTaylor);
  game.PlayMove(game::AsLoc(60), WHITE);  // Black to move, behind on count.
  Root r = OneVisitEach();
  CHECK(AvoidLosingPass(game, &r.root, BLACK, 0.1f, game::kPassLoc) ==
        game::AsLoc(0));
}

}  // namespace
}  // namespace gtp
