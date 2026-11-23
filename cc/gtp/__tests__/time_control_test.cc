#include "cc/gtp/time_control.h"

#include <memory>
#include <vector>

#include "cc/core/doctest_include.h"
#include "cc/game/game.h"
#include "cc/mcts/tree.h"

namespace gtp {
namespace {

using mcts::TreeNode;

// An evaluated root whose children (action, visits, v, v_outcome, var) are
// given from the child's own perspective. Root n = 1 + sum of child visits.
struct Root {
  struct Kid {
    int a;
    int n;
    float v;
    float v_outcome;
    float var;
  };
  TreeNode root{0};
  std::vector<std::unique_ptr<TreeNode>> kids;

  explicit Root(const std::vector<Kid>& ks) {
    int total = 1;
    for (const Kid& k : ks) {
      auto c = std::make_unique<TreeNode>(0);
      c->n = k.n;
      c->v = k.v;
      c->v_outcome = k.v_outcome;
      c->v_var = k.var;
      c->v_outcome_var = k.var;
      root.children[k.a] = c.get();
      root.child_visits[k.a] = k.n;
      total += k.n;
      kids.push_back(std::move(c));
    }
    root.n = total;
    root.v_var = 0.01f;
    root.state = mcts::TreeNodeState::kNnEvaluated;
  }
};

// 30 minutes sudden death, 30 minutes left, ply 0, no factors: 450 plies left
// -> 225 of ours; (1800 s - 15 s reserve) / 225.
constexpr int kBaseMs = 7933;

TimeControl SuddenDeath(uint32_t flags, int secs_left = 1800) {
  TimeControl tc(flags);
  tc.SetTimeSettings(1800, 0, 0);
  tc.SetTimeLeft(secs_left, 0, 0);
  return tc;
}

TEST_CASE("disabled until time_settings") {
  TimeControl tc(~0u);
  game::Game game;
  CHECK(!tc.IsEnabled());
  CHECK(tc.ComputeMoveTimeMs(game, nullptr).first == 0);
}

TEST_CASE("byoyomi: use the period minus 1 s") {
  TimeControl tc(~0u);
  game::Game game;
  tc.SetTimeSettings(0, 30, 5);
  tc.SetTimeLeft(0, 30, 5);
  CHECK(tc.ComputeMoveTimeMs(game, nullptr).first == 29000);
}

TEST_CASE("main time: even split over estimated moves left") {
  game::Game game;
  TimeControl tc = SuddenDeath(0);
  auto [ms, md] = tc.ComputeMoveTimeMs(game, nullptr);
  CHECK(ms == kBaseMs);
  CHECK(md.appx_moves_left == 450);
  CHECK(md.middlegame_factor == doctest::Approx(0.7f));
}

TEST_CASE("reserve: inside the last 15 s every move gets the 1 ms floor") {
  game::Game game;
  TimeControl tc = SuddenDeath(~0u, 14);
  CHECK(tc.ComputeMoveTimeMs(game, nullptr).first == 1);
}

TEST_CASE("winning attenuator") {
  game::Game game;
  TimeControl tc = SuddenDeath(TimeControl::kWinningAttenuator);

  SUBCASE("certain win with >= 400 root visits: no time") {
    Root r({{0, 500, 0, -1.0f, 0.0f}});
    auto [ms, md] = tc.ComputeMoveTimeMs(game, &r.root);
    CHECK(md.winning_attenuator == 0.0f);
    CHECK(ms == 1);
  }

  SUBCASE("certain win under 400 root visits: full time") {
    Root r({{0, 398, 0, -1.0f, 0.0f}});
    REQUIRE(r.root.n == 399);
    auto [ms, md] = tc.ComputeMoveTimeMs(game, &r.root);
    CHECK(md.winning_attenuator == 1.0f);
    CHECK(md.pre_search_winrate == doctest::Approx(1.0f));
    CHECK(ms == kBaseMs);
  }

  SUBCASE("LCB halfway between 90% and 98%: 1 - 0.5^2") {
    Root r({{0, 1000, 0, -0.88f, 1e-8f}});
    auto [ms, md] = tc.ComputeMoveTimeMs(game, &r.root);
    CHECK(md.pre_search_winrate == doctest::Approx(0.94f).epsilon(1e-3));
    CHECK(md.winning_attenuator == doctest::Approx(0.75f).epsilon(1e-2));
  }

  SUBCASE("LCB at or below 90%: full time") {
    Root r({{0, 1000, 0, -0.8f, 0.04f}});
    auto [ms, md] = tc.ComputeMoveTimeMs(game, &r.root);
    CHECK(md.winning_attenuator == 1.0f);
    CHECK(ms == kBaseMs);
  }
}

TEST_CASE("obvious move") {
  game::Game game;
  TimeControl tc = SuddenDeath(TimeControl::kObviousMoveFactor);
  // Best move Q = 0.5, second Q = -0.5 with tight variance: z far above 3.
  const std::vector<Root::Kid> obvious = {{0, 80, -0.5f, 0, 1e-4f},
                                          {1, 19, 0.5f, 0, 1e-4f}};
  // Best Q = 0.01, second Q = 0: z below 3.
  const std::vector<Root::Kid> close = {{0, 120, -0.01f, 0, 0.04f},
                                        {1, 99, 0.0f, 0, 0.04f}};
  Root seed(obvious);  // n = 100
  Root big_obvious({{0, 160, -0.5f, 0, 1e-4f}, {1, 39, 0.5f, 0, 1e-4f}});
  Root big_close(close);  // n = 220
  REQUIRE(seed.root.n == 100);
  REQUIRE(big_obvious.root.n == 200);

  // First root seeds the root-n EMA, so no insta-move yet.
  CHECK(tc.ComputeMoveTimeMs(game, &seed.root).second.obv_move_factor == 1.0f);

  SUBCASE("big root and clear best move: insta-move") {
    auto [ms, md] = tc.ComputeMoveTimeMs(game, &big_obvious.root);
    CHECK(md.obv_move_factor == 0.0f);
    CHECK(ms == 1);
  }

  SUBCASE("big root but close moves: full time") {
    CHECK(tc.ComputeMoveTimeMs(game, &big_close.root).second.obv_move_factor ==
          1.0f);
  }

  SUBCASE("small root: full time") {
    CHECK(tc.ComputeMoveTimeMs(game, &seed.root).second.obv_move_factor ==
          1.0f);
  }

  SUBCASE("ResetGame clears the EMA") {
    tc.ResetGame();
    CHECK(
        tc.ComputeMoveTimeMs(game, &big_obvious.root).second.obv_move_factor ==
        1.0f);
  }
}

}  // namespace
}  // namespace gtp
