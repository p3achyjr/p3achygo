#include "cc/selfplay/move_sel_manager.h"

#include "cc/core/doctest_include.h"

namespace selfplay {
namespace {

// Returns a calibration with realistic percentile values (from gen data).
SelMultCalibration MakeCalibration() {
  SelMultCalibration c;
  // StdPenalty: p01/p50; StdBonus: p80/p99.
  c.v_outcome_stddev_adj["p01"] = 0.02f;
  c.v_outcome_stddev_adj["p50"] = 0.64f;
  c.v_outcome_stddev_adj["p80"] = 1.52f;
  c.v_outcome_stddev_adj["p99"] = 4.96f;
  // NnMctsBonus: p70/p99.
  c.nn_mcts_diff["p70"] = 0.15f;
  c.nn_mcts_diff["p99"] = 0.65f;
  // Simple expected_std: constant 0.16 across all bins.
  for (int n = 0; n <= 200; n += 5) c.expected_std_by_n[n] = 0.16f;
  return c;
}

TEST_CASE("uncalibrated: std_adj=0 => modifier=1") {
  SelMultCalibration empty;
  MoveSelManager mgr(kStddevBonus | kStddevPenalty, empty);
  auto r = mgr.Compute(32, 0.15f, 0.0f, 0.0f, 0.0f, 1.0f);
  CHECK(r.std_adj == 0.0f);
  CHECK(r.std_adj_att == 0.0f);
  CHECK(r.sel_std_bonus == doctest::Approx(1.0f));
  CHECK(r.sel_std_penalty == doctest::Approx(1.0f));
  CHECK(r.modifier == doctest::Approx(1.0f));
}

TEST_CASE("neutral position: std_adj~1 => modifier~1") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kStddevBonus | kStddevPenalty, c);
  // std_dev == expected_std => std_adj = 1.0, which is between p50 and p80.
  auto r = mgr.Compute(128, 0.16f, 0.0f, 0.0f, 0.0f, 1.0f);
  CHECK(r.std_adj == doctest::Approx(1.0f));
  CHECK(r.sel_std_bonus == doctest::Approx(1.0f));    // below p80 lb
  CHECK(r.sel_std_penalty == doctest::Approx(1.0f));  // above p50 ub
  CHECK(r.modifier == doctest::Approx(1.0f));
}

TEST_CASE("high std_adj => bonus active") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kStddevBonus | kStddevPenalty, c);
  // std_dev = 0.48 => std_adj = 3.0, above p80=1.52.
  // At n=128, att = min(1, 0.2 + 0.8*(128/40)^0.54) = 1.0.
  // std_adj_att = 3.0. bonus = 1 + 0.5*(3.0-1.52)/(4.96-1.52) ~ 1.215.
  auto r = mgr.Compute(128, 0.48f, 0.0f, 0.0f, 0.0f, 1.0f);
  CHECK(r.std_adj == doctest::Approx(3.0f));
  CHECK(r.std_adj_att == doctest::Approx(3.0f));
  CHECK(r.sel_std_bonus > 1.0f);
  CHECK(r.sel_std_penalty == doctest::Approx(1.0f));
  CHECK(r.modifier > 1.0f);
}

TEST_CASE("low std_adj => penalty active") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kStddevBonus | kStddevPenalty, c);
  // std_dev = 0.016 => std_adj = 0.1, below p50=0.64.
  auto r = mgr.Compute(128, 0.016f, 0.0f, 0.0f, 0.0f, 1.0f);
  CHECK(r.std_adj < 0.64f);
  CHECK(r.sel_std_penalty < 1.0f);
  CHECK(r.sel_std_bonus == doctest::Approx(1.0f));
  CHECK(r.modifier < 1.0f);
}

TEST_CASE("low n_pre attenuates std_adj toward 1") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kStddevBonus | kStddevPenalty, c);
  // Same std_dev=0.48 (std_adj=3.0) but n_pre=5 vs n_pre=128.
  auto r_low = mgr.Compute(5, 0.48f, 0.0f, 0.0f, 0.0f, 1.0f);
  auto r_high = mgr.Compute(128, 0.48f, 0.0f, 0.0f, 0.0f, 1.0f);
  CHECK(r_low.std_adj_att < r_high.std_adj_att);
  CHECK(r_low.modifier < r_high.modifier);
}

TEST_CASE("decisive position attenuates modifier via sel_q_adjust") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kStddevBonus | kStddevPenalty, c);
  auto r_contested = mgr.Compute(128, 0.48f, 0.0f, 0.0f, 0.0f, 1.0f);
  auto r_decisive = mgr.Compute(128, 0.48f, 0.0f, 0.0f, 0.95f, 1.0f);
  CHECK(r_decisive.sel_q_adjust < r_contested.sel_q_adjust);
  // Bonus is pulled toward 1 in decisive positions.
  CHECK(r_decisive.sel_bonus < r_contested.sel_bonus);
}

TEST_CASE("flags: kStddevBonus only => penalty not applied to modifier") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kStddevBonus, c);
  // std_adj < p50 so penalty would fire if enabled, but flag is off.
  auto r = mgr.Compute(128, 0.016f, 0.0f, 0.0f, 0.0f, 1.0f);
  CHECK(r.sel_std_penalty < 1.0f);                // computed but...
  CHECK(r.sel_penalty == doctest::Approx(1.0f));  // ...not applied
  CHECK(r.modifier == doctest::Approx(1.0f));  // no bonus either (std_adj<p80)
}

TEST_CASE("sel_mult_scale_factor=0 => modifier=1") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kStddevBonus | kStddevPenalty, c);
  auto r = mgr.Compute(128, 0.48f, 0.0f, 0.0f, 0.0f, 0.0f);
  CHECK(r.modifier == doctest::Approx(1.0f));
  CHECK(r.modifier_unscaled > 1.0f);  // unscaled still has the bonus
}

TEST_CASE("bonus is max of active terms, not product") {
  auto c = MakeCalibration();
  c.pre_kld["p70"] = 0.1f;
  c.pre_kld["p95"] = 0.5f;
  // std_dev=0.48 => sel_std_bonus>1; pre_kld=0.3 => sel_kld_bonus>1.
  // With both flags: modifier should use max(std_bonus, kld_bonus), not
  // product.
  MoveSelManager mgr_both(kStddevBonus | kKldBonus, c);
  MoveSelManager mgr_std(kStddevBonus, c);
  MoveSelManager mgr_kld(kKldBonus, c);
  auto r_both = mgr_both.Compute(128, 0.48f, 0.3f, 0.0f, 0.0f, 1.0f);
  auto r_std = mgr_std.Compute(128, 0.48f, 0.3f, 0.0f, 0.0f, 1.0f);
  auto r_kld = mgr_kld.Compute(128, 0.48f, 0.3f, 0.0f, 0.0f, 1.0f);
  // Both signals computed regardless of flags.
  CHECK(r_both.sel_std_bonus > 1.0f);
  CHECK(r_both.sel_kld_bonus > 1.0f);
  // Combined bonus = max, so equal to the larger of the two individual bonuses.
  CHECK(r_both.sel_bonus ==
        doctest::Approx(std::max(r_std.sel_bonus, r_kld.sel_bonus))
            .epsilon(0.001f));
  // And strictly less than the product.
  CHECK(r_both.sel_bonus < r_std.sel_bonus * r_kld.sel_bonus);
}

TEST_CASE("penalty is min of active terms, not product") {
  auto c = MakeCalibration();
  // KldPenalty lb=p05, ub=0.06 (hardcoded). pre_kld=0.005 is in penalty zone.
  c.pre_kld["p05"] = 0.001f;
  // std_dev=0.016 => sel_std_penalty<1; pre_kld=0.005 => sel_kld_penalty<1.
  MoveSelManager mgr_both(kStddevPenalty | kKldPenalty, c);
  MoveSelManager mgr_std(kStddevPenalty, c);
  MoveSelManager mgr_kld(kKldPenalty, c);
  auto r_both = mgr_both.Compute(128, 0.016f, 0.005f, 0.0f, 0.0f, 1.0f);
  auto r_std = mgr_std.Compute(128, 0.016f, 0.005f, 0.0f, 0.0f, 1.0f);
  auto r_kld = mgr_kld.Compute(128, 0.016f, 0.005f, 0.0f, 0.0f, 1.0f);
  CHECK(r_both.sel_std_penalty < 1.0f);
  CHECK(r_both.sel_kld_penalty < 1.0f);
  // Combined penalty = min, so equal to the smaller of the two.
  CHECK(r_both.sel_penalty ==
        doctest::Approx(std::min(r_std.sel_penalty, r_kld.sel_penalty))
            .epsilon(0.001f));
  // And strictly greater than the product.
  CHECK(r_both.sel_penalty > r_std.sel_penalty * r_kld.sel_penalty);
}

TEST_CASE("kNnMctsBonus: high nn_mcts_diff => bonus active") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kNnMctsBonus, c);
  // nn_mcts_diff=0.5 > p70=0.15; bonus = 1 + 0.60*(0.5-0.15)/(0.65-0.15)
  // = 1.42.
  auto r = mgr.Compute(128, 0.0f, 0.0f, 0.5f, 0.0f, 1.0f);
  CHECK(r.sel_nn_mcts_bonus > 1.0f);
  CHECK(r.sel_bonus > 1.0f);
  CHECK(r.modifier > 1.0f);
}

TEST_CASE("kNnMctsBonus: nn_mcts_diff=0 (uninit root) => no bonus") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kNnMctsBonus, c);
  auto r = mgr.Compute(0, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  CHECK(r.sel_nn_mcts_bonus == doctest::Approx(1.0f));
  CHECK(r.modifier == doctest::Approx(1.0f));
}

TEST_CASE("kNnMctsBonus: nn_mcts_diff below lb => no bonus") {
  auto c = MakeCalibration();
  MoveSelManager mgr(kNnMctsBonus, c);
  // nn_mcts_diff=0.05 < p70=0.15 => no bonus.
  auto r = mgr.Compute(128, 0.0f, 0.0f, 0.05f, 0.0f, 1.0f);
  CHECK(r.sel_nn_mcts_bonus == doctest::Approx(1.0f));
  CHECK(r.modifier == doctest::Approx(1.0f));
}

TEST_CASE("kNnMctsBonus is included in max with other bonus signals") {
  auto c = MakeCalibration();
  c.pre_kld["p70"] = 0.1f;
  c.pre_kld["p95"] = 0.5f;
  MoveSelManager mgr(kNnMctsBonus | kKldBonus, c);
  MoveSelManager mgr_nm(kNnMctsBonus, c);
  MoveSelManager mgr_kld(kKldBonus, c);
  // nn_mcts_diff=0.5, pre_kld=0.3 — both signals active.
  auto r_both = mgr.Compute(128, 0.0f, 0.3f, 0.5f, 0.0f, 1.0f);
  auto r_nm = mgr_nm.Compute(128, 0.0f, 0.3f, 0.5f, 0.0f, 1.0f);
  auto r_kld = mgr_kld.Compute(128, 0.0f, 0.3f, 0.5f, 0.0f, 1.0f);
  CHECK(r_both.sel_bonus ==
        doctest::Approx(std::max(r_nm.sel_bonus, r_kld.sel_bonus))
            .epsilon(0.001f));
}

}  // namespace
}  // namespace selfplay
