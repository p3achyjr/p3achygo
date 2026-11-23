#include "cc/mcts/tree.h"

#include <memory>
#include <vector>

#include "cc/core/doctest_include.h"

namespace mcts {
namespace {

// A child at action `a` with `n` visits (edge and node) and the given stats
// from the child's own perspective.
struct Kid {
  int a;
  int n;
  float v;
  float v_outcome;
  float var;
};

// A root with the given children. Root n = 1 + sum of child visits.
struct Tree {
  TreeNode root{0};
  std::vector<std::unique_ptr<TreeNode>> kids;

  explicit Tree(const std::vector<Kid>& ks) {
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
  }
};

const auto kAllValid = [](int) { return true; };

TEST_CASE("OutcomeWinrateLcb") {
  // Parent's winrate for each child: (-v_outcome + 1) / 2.
  Tree t({{0, 1, 0, -0.8f, 0.04f},
          {1, 100, 0, -0.8f, 0.04f},
          {2, 10000, 0, -0.8f, 0.04f},
          {3, 50, 0, -1.0f, 0.0f}});

  // Fewer than 2 visits: no bound.
  CHECK(OutcomeWinrateLcb(&t.root, 0) == 0.0f);
  // Unvisited action.
  CHECK(OutcomeWinrateLcb(&t.root, 4) == 0.0f);

  // Below the 0.9 mean, and tighter with more visits.
  const float lcb100 = OutcomeWinrateLcb(&t.root, 1);
  const float lcb10k = OutcomeWinrateLcb(&t.root, 2);
  CHECK(lcb100 < 0.9f);
  CHECK(lcb100 > 0.85f);
  CHECK(lcb10k > lcb100);
  CHECK(lcb10k < 0.9f);

  // A certain win with no variance is exactly 1.
  CHECK(OutcomeWinrateLcb(&t.root, 3) == doctest::Approx(1.0f));
}

TEST_CASE("BestLcbAction") {
  // Action 1 has the best LCB but few visits; action 2 is also low-visit.
  Tree t({{0, 100, -0.1f, 0, 0.01f},
          {1, 5, -0.5f, 0, 0.01f},
          {2, 8, -0.05f, 0, 0.01f}});
  REQUIRE(Lcb(&t.root, 1) > Lcb(&t.root, 0));

  SUBCASE("no gate picks the best LCB") {
    CHECK(BestLcbAction(&t.root, 0.0f, kAllValid) == 1);
  }

  SUBCASE("gate drops actions below the visit fraction") {
    // min visits = 0.1 * 100 = 10: actions 1 (5) and 2 (8) are gated out.
    CHECK(BestLcbAction(&t.root, 0.1f, kAllValid) == 0);
  }

  SUBCASE("gated-out actions rank by visits, after every gated-in one") {
    // With action 0 invalid, action 2 (8 visits) beats action 1 (5 visits)
    // even though action 1's LCB is higher.
    CHECK(BestLcbAction(&t.root, 0.1f, [](int a) { return a != 0; }) == 2);
  }

  SUBCASE("skips invalid actions") {
    CHECK(BestLcbAction(&t.root, 0.0f, [](int a) { return a != 1; }) != 1);
  }

  SUBCASE("no valid action") {
    CHECK(BestLcbAction(&t.root, 0.1f, [](int) { return false; }) == -1);
  }
}

TEST_CASE("RecomputeNodeStats: visit weighting") {
  Tree t({{0, 30, -0.2f, -0.1f, 0.01f}, {1, 10, 0.4f, 0.3f, 0.01f}});
  t.root.init_util_est = 0.05f;
  t.root.init_outcome_est = 0.02f;
  RecomputeNodeStats(&t.root, /*obs_bias=*/0.0f, /*exponent=*/0.0f);

  // Children enter from the parent's perspective (sign flipped).
  const float n = 41.0f;
  CHECK(t.root.v == doctest::Approx((0.05f + 30 * 0.2f + 10 * -0.4f) / n));
  CHECK(t.root.v_outcome ==
        doctest::Approx((0.02f + 30 * 0.1f + 10 * -0.3f) / n));
  CHECK(t.root.max_child_n == 30);
}

TEST_CASE("RecomputeNodeStats: value weighting") {
  SUBCASE("equal children: weights are rescaled to the visit total") {
    Tree a({{0, 40, -0.2f, -0.2f, 0.01f}, {1, 40, -0.2f, -0.2f, 0.01f}});
    Tree b({{0, 40, -0.2f, -0.2f, 0.01f}, {1, 40, -0.2f, -0.2f, 0.01f}});
    RecomputeNodeStats(&a.root, 0.0f, 0.0f);
    RecomputeNodeStats(&b.root, 0.0f, 0.5f);
    CHECK(b.root.v == doctest::Approx(a.root.v));
    CHECK(b.root.v_outcome == doctest::Approx(a.root.v_outcome));
  }

  SUBCASE("a bad child is downweighted") {
    // Children 0 and 1 are good for the parent (Q = 0.2); child 2 is bad
    // (Q = -0.6). Value weighting moves the parent's value toward the good
    // children.
    const std::vector<Kid> kids = {{0, 50, -0.2f, -0.2f, 0.01f},
                                   {1, 50, -0.2f, -0.2f, 0.01f},
                                   {2, 50, 0.6f, 0.6f, 0.01f}};
    Tree off(kids), on(kids);
    RecomputeNodeStats(&off.root, 0.0f, 0.0f);
    RecomputeNodeStats(&on.root, 0.0f, 1.0f);
    CHECK(on.root.v > off.root.v);
    CHECK(on.root.v_outcome > off.root.v_outcome);
    CHECK(on.root.v < 0.2f);
  }
}

}  // namespace
}  // namespace mcts
