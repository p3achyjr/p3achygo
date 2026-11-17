#include "cc/mcts/tree.h"

#include <memory>
#include <vector>

#include "cc/core/doctest_include.h"
#include "cc/mcts/search_policy.h"

namespace mcts {
namespace {

// Builds a node whose children have the given visit counts and values (child
// perspective, i.e. the parent sees -v). Children own their own stats only.
struct Fixture {
  TreeNode node{0};
  std::vector<std::unique_ptr<TreeNode>> children;

  Fixture(float init_util, const std::vector<std::pair<int, float>>& kids) {
    node.init_util_est = init_util;
    node.init_outcome_est = init_util;
    node.n = 1;
    for (int i = 0; i < kids.size(); ++i) {
      auto [n, v] = kids[i];
      auto child = std::make_unique<TreeNode>(i + 1);
      child->n = n;
      child->v = v;
      child->v_outcome = v;
      child->score = 2 * v;
      node.children[i] = child.get();
      node.child_visits[i] = n;
      node.n += n;
      children.push_back(std::move(child));
    }
  }
};

TEST_CASE("TDist3Cdf") {
  DOCTEST_CHECK(TDist3Cdf(0.0) == doctest::Approx(0.5));
  DOCTEST_CHECK(TDist3Cdf(2.0) == doctest::Approx(0.930337).epsilon(1e-5));
  DOCTEST_CHECK(TDist3Cdf(-2.0) == doctest::Approx(1.0 - 0.930337).epsilon(1e-5));
}

TEST_CASE("Exponent 0 matches default recompute") {
  Fixture a(0.1f, {{30, -0.2f}, {10, 0.3f}, {5, 0.6f}});
  Fixture b(0.1f, {{30, -0.2f}, {10, 0.3f}, {5, 0.6f}});
  RecomputeNodeStats(&a.node);
  RecomputeNodeStats(&b.node, 0.0f, 0.0f);
  DOCTEST_CHECK(a.node.v == b.node.v);
  DOCTEST_CHECK(a.node.v_outcome == b.node.v_outcome);
  DOCTEST_CHECK(a.node.score == b.node.score);
  DOCTEST_CHECK(a.node.v_var == b.node.v_var);
  DOCTEST_CHECK(a.node.v_m3 == b.node.v_m3);
  // Plain visit-weighted mean: (0.1 + 30*0.2 - 10*0.3 - 5*0.6) / 46.
  DOCTEST_CHECK(a.node.v == doctest::Approx((0.1 + 6.0 - 3.0 - 3.0) / 46.0));
}

TEST_CASE("Equal child values are unchanged") {
  Fixture a(0.0f, {{20, 0.1f}, {20, 0.1f}});
  Fixture b(0.0f, {{20, 0.1f}, {20, 0.1f}});
  RecomputeNodeStats(&a.node);
  RecomputeNodeStats(&b.node, 0.0f, 0.25f);
  DOCTEST_CHECK(b.node.v == doctest::Approx(a.node.v));
}

TEST_CASE("Bad children are downweighted") {
  // From the parent's view child 0 is worth +0.2, child 1 is worth -0.5.
  Fixture plain(0.0f, {{40, -0.2f}, {40, 0.5f}});
  Fixture w25(0.0f, {{40, -0.2f}, {40, 0.5f}});
  Fixture w50(0.0f, {{40, -0.2f}, {40, 0.5f}});
  RecomputeNodeStats(&plain.node);
  RecomputeNodeStats(&w25.node, 0.0f, 0.25f);
  RecomputeNodeStats(&w50.node, 0.0f, 0.5f);
  DOCTEST_CHECK(w25.node.v > plain.node.v);
  DOCTEST_CHECK(w50.node.v > w25.node.v);
  DOCTEST_CHECK(w50.node.v < 0.2f);  // still an average, below the best child
  DOCTEST_CHECK(w25.node.v_outcome > plain.node.v_outcome);
  DOCTEST_CHECK(w25.node.score > plain.node.score);
  // Total weight preserved: w / n is still the value.
  DOCTEST_CHECK(w25.node.w / w25.node.n == doctest::Approx(w25.node.v));

  // Hand computation for exponent 0.25: mean = (0.2 - 0.5) / 2 = -0.15,
  // stdev = sqrt(1 / (1.5 * sqrt(40))), z = +-0.35 / stdev.
  const double stdev = std::sqrt(1e-8 + 1.0 / (1.5 * std::sqrt(40.0)));
  const double z = 0.35 / stdev;
  const double w_good = 40 * std::pow(TDist3Cdf(z) + 1e-4, 0.25);
  const double w_bad = 40 * std::pow(TDist3Cdf(-z) + 1e-4, 0.25);
  const double scale = 80.0 / (w_good + w_bad);
  const double expected =
      (0.0 + scale * (w_good * 0.2 + w_bad * -0.5)) / 81.0;
  DOCTEST_CHECK(w25.node.v == doctest::Approx(expected).epsilon(1e-5));
}

TEST_CASE("Leaf weights of 1 match visit weighting") {
  Fixture a(0.1f, {{30, -0.2f}, {10, 0.3f}, {5, 0.6f}});
  Fixture b(0.1f, {{30, -0.2f}, {10, 0.3f}, {5, 0.6f}});
  for (auto& c : b.children) c->wsum = c->n;
  b.node.leaf_weight = 1.0f;
  RecomputeNodeStats(&a.node);
  RecomputeNodeStats(&b.node, 0.0f, 0.0f, /*use_leaf_weights=*/true);
  DOCTEST_CHECK(b.node.v == doctest::Approx(a.node.v));
  DOCTEST_CHECK(b.node.v_outcome == doctest::Approx(a.node.v_outcome));
  DOCTEST_CHECK(b.node.score == doctest::Approx(a.node.score));
  DOCTEST_CHECK(b.node.v_var == doctest::Approx(a.node.v_var));
  DOCTEST_CHECK(b.node.wsum == doctest::Approx(a.node.n));
}

TEST_CASE("Leaf weights weight the average") {
  // Two single-visit children (leaves) with weights 3 and 0.5; own weight 2.
  Fixture f(0.1f, {{1, -0.4f}, {1, 0.2f}});
  f.children[0]->wsum = 3.0f;
  f.children[1]->wsum = 0.5f;
  f.node.leaf_weight = 2.0f;
  RecomputeNodeStats(&f.node, 0.0f, 0.0f, /*use_leaf_weights=*/true);
  const double total = 2.0 + 3.0 + 0.5;
  DOCTEST_CHECK(f.node.wsum == doctest::Approx(total));
  DOCTEST_CHECK(f.node.v == doctest::Approx((2 * 0.1 + 3 * 0.4 + 0.5 * -0.2) / total));
  DOCTEST_CHECK(f.node.v_outcome ==
        doctest::Approx((2 * 0.1 + 3 * 0.4 + 0.5 * -0.2) / total));
  // score = 2 * v for the children in the fixture; own init_score_est = 0.
  DOCTEST_CHECK(f.node.score == doctest::Approx((3 * 0.8 + 0.5 * -0.4) / total));
}

TEST_CASE("ChildEdgeWeight scales subtree weight to edge visits") {
  Fixture f(0.0f, {{4, 0.0f}});
  f.children[0]->wsum = 8.0f;
  f.node.child_visits[0] = 2;  // e.g. a transposition: 2 of 4 visits here
  DOCTEST_CHECK(ChildEdgeWeight(&f.node, 0) == doctest::Approx(4.0f));
}

TEST_CASE("UncertaintyLeafWeight") {
  const PuctParams p = PuctParams::Builder()
                           .set_uncertainty_weight_coeff(0.25f)
                           .set_uncertainty_max_weight(8.0f)
                           .set_uncertainty_weight_scale(2.0f)
                           .build();
  DOCTEST_CHECK(UncertaintyLeafWeight(0.0f, p) == doctest::Approx(4.0f));
  // err = c * (1 - 1/max) gives raw weight 1.
  DOCTEST_CHECK(UncertaintyLeafWeight(0.25f * (1 - 1 / 8.0f), p) ==
        doctest::Approx(0.5f));
}

TEST_CASE("ChildVarShrinkFactors") {
  // Children: 6 visits (var 0.09), 2 visits (var 0.01), plus unvisited slots.
  Fixture f(0.0f, {{6, 0.0f}, {2, 0.0f}});
  f.node.v_var = 0.04f;
  f.children[0]->v_var = 0.09f;
  f.children[1]->v_var = 0.01f;
  const auto factors = ChildVarShrinkFactors(&f.node, 3.0f);
  const double s0 = std::sqrt((3 * 0.04 + 6 * 0.09) / 9.0);
  const double s1 = std::sqrt((3 * 0.04 + 2 * 0.01) / 5.0);
  const double s_unvisited = std::sqrt(0.04);
  const double ref = (6 * s0 + 2 * s1) / 8.0;
  DOCTEST_CHECK(factors[0] == doctest::Approx(s0 / ref));
  DOCTEST_CHECK(factors[1] == doctest::Approx(s1 / ref));
  DOCTEST_CHECK(factors[5] == doctest::Approx(s_unvisited / ref));
  // Visit-weighted mean of the factors is 1.
  DOCTEST_CHECK((6 * factors[0] + 2 * factors[1]) / 8.0 ==
                doctest::Approx(1.0));

  // No visited child: all 1.
  Fixture empty(0.0f, {});
  empty.node.v_var = 0.04f;
  const auto ones = ChildVarShrinkFactors(&empty.node, 3.0f);
  DOCTEST_CHECK(ones[0] == 1.0f);
}

TEST_CASE("ChildSdNormFactors") {
  // Children: 6 visits (var 0.09), 2 visits (var 0.01), 1 visit, plus
  // unvisited slots. S(n) = 0.1 + 0.05 log2 n.
  Fixture f(0.0f, {{6, 0.0f}, {2, 0.0f}, {1, 0.0f}});
  f.children[0]->v_var = 0.09f;
  f.children[1]->v_var = 0.01f;
  f.children[2]->v_var = 0.5f;  // n < 2: ignored, rho = 1
  const std::array<float, 3> poly = {0.1f, 0.05f, 0.0f};
  const auto factors = ChildSdNormFactors(&f.node, 3.0f, poly);
  const double rho0 = 0.3 / (0.1 + 0.05 * std::log2(6.0));
  const double rho1 = 0.1 / (0.1 + 0.05 * 1.0);
  const double h0 = (3 + 6 * rho0) / 9.0, h1 = (3 + 2 * rho1) / 5.0, h2 = 1.0;
  const double ref = (6 * h0 + 2 * h1 + 1 * h2) / 9.0;
  DOCTEST_CHECK(factors[0] == doctest::Approx(h0 / ref));
  DOCTEST_CHECK(factors[1] == doctest::Approx(h1 / ref));
  DOCTEST_CHECK(factors[2] == doctest::Approx(1.0 / ref));
  DOCTEST_CHECK(factors[7] == doctest::Approx(1.0 / ref));  // unvisited
  DOCTEST_CHECK((6 * factors[0] + 2 * factors[1] + factors[2]) / 9.0 ==
                doctest::Approx(1.0));

  // No visited child: all 1.
  Fixture empty(0.0f, {});
  DOCTEST_CHECK(ChildSdNormFactors(&empty.node, 3.0f, poly)[0] == 1.0f);

  // nonroot_only: the root scores match the feature being off; non-root
  // nodes match the feature being on.
  f.node.state = TreeNodeState::kNnEvaluated;
  for (int a = 0; a < 3; ++a) f.node.move_probs[a] = 0.3f;
  for (auto& c : f.children) c->state = TreeNodeState::kNnEvaluated;
  const auto scores = [&](float k, bool nonroot_only, bool is_root) {
    PuctScorer scorer(
        PuctParams::Builder().set_child_sd_norm(k, nonroot_only, poly).build(),
        IdentityQ{}, IdentityN{});
    return scorer.ComputeScores(&f.node, is_root);
  };
  DOCTEST_CHECK(scores(3.0f, true, true) == scores(0.0f, false, true));
  DOCTEST_CHECK(scores(3.0f, true, false) == scores(3.0f, false, false));
  DOCTEST_CHECK(!(scores(3.0f, false, false) == scores(0.0f, false, false)));
}

TEST_CASE("Root logit-space optimism mix") {
  // Unvisited children all get the same FPU value, so score differences are
  // proportional to prior differences.
  TreeNode node{0};
  node.state = TreeNodeState::kNnEvaluated;
  node.n = 1;
  const std::array<float, 3> l = {1.0f, 0.0f, -1.0f};
  const std::array<float, 3> l_opt = {-1.0f, 0.5f, 0.0f};
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    node.move_logits[a] = a < 3 ? l[a] : -30.0f;
  }
  std::array<float, constants::kMaxMovesPerPosition> opt_l;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    opt_l[a] = a < 3 ? l_opt[a] : -30.0f;
  }
  const auto softmax = [](const auto& x) {
    std::array<float, constants::kMaxMovesPerPosition> y;
    double z = 0;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) z += std::exp(x[a]);
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) y[a] = std::exp(x[a]) / z;
    return y;
  };
  node.opt_probs = softmax(opt_l);
  node.move_probs = softmax(node.move_logits);

  const auto scores = [&](float p_opt, float root_w, bool is_root) {
    PuctScorer scorer(PuctParams::Builder()
                          .set_p_opt_weight(p_opt)
                          .set_root_p_opt_logit_weight(root_w)
                          .build(),
                      IdentityQ{}, IdentityN{});
    std::array<float, 3> s;
    for (const auto& [a, v] : scorer.ComputeScores(&node, is_root)) {
      if (a < 3) s[a] = v;
    }
    return s;
  };
  const auto ratio = [](const std::array<float, 3>& s) {
    return (s[0] - s[2]) / (s[1] - s[2]);
  };

  // w = 0.2 matches softmax(0.8 l + 0.2 l_opt).
  std::array<double, 3> p;
  double sum = 0;
  for (int a = 0; a < 3; ++a) sum += p[a] = std::exp(0.8 * l[a] + 0.2 * l_opt[a]);
  for (int a = 0; a < 3; ++a) p[a] /= sum;
  DOCTEST_CHECK(ratio(scores(1.0f, 0.2f, true)) ==
                doctest::Approx((p[0] - p[2]) / (p[1] - p[2])).epsilon(1e-4));
  // w = 1 at the root equals the optimistic policy; w = 0 the main policy.
  DOCTEST_CHECK(ratio(scores(0.0f, 1.0f, true)) ==
                doctest::Approx(ratio(scores(1.0f, -1.0f, true))).epsilon(1e-4));
  DOCTEST_CHECK(ratio(scores(1.0f, 0.0f, true)) ==
                doctest::Approx(ratio(scores(0.0f, -1.0f, true))).epsilon(1e-4));
  // Non-root nodes ignore the root weight.
  DOCTEST_CHECK(scores(1.0f, 0.2f, false) == scores(1.0f, -1.0f, false));
}

}  // namespace
}  // namespace mcts
