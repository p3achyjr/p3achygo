#include "cc/eval/player_config.h"

namespace eval {
mcts::ScoreUtilityParams MakeScoreUtilityParams(
    const eval::PlayerSearchConfig& cfg) {
  mcts::ScoreUtilityMode mode = cfg.score_utility_mode == "integral"
                                    ? mcts::ScoreUtilityMode::kIntegral
                                    : mcts::ScoreUtilityMode::kDirect;
  return mcts::ScoreUtilityParams{.score_weight = cfg.score_weight,
                                  .mode = mode};
}

mcts::PuctParams MakePuctParams(const eval::PlayerSearchConfig& cfg) {
  mcts::PuctRootSelectionPolicy policy;
  if (cfg.puct_root_policy == "lcb") {
    policy = mcts::PuctRootSelectionPolicy::kLcb;
  } else if (cfg.puct_root_policy == "visit_count_sample") {
    policy = mcts::PuctRootSelectionPolicy::kVisitCountSample;
  } else if (cfg.puct_root_policy == "visit_count") {
    policy = mcts::PuctRootSelectionPolicy::kVisitCount;
  } else {
    policy = cfg.use_lcb ? mcts::PuctRootSelectionPolicy::kLcb
                         : mcts::PuctRootSelectionPolicy::kVisitCount;
  }
  return mcts::PuctParams::Builder()
      .set_kind(policy)
      .set_c_puct(cfg.c_puct)
      .set_c_puct_visit_scaling(cfg.c_puct_visit_scaling)
      .set_c_puct_v_2(cfg.c_puct_v_2)
      .set_use_puct_v(cfg.use_puct_v)
      .set_enable_var_scaling(cfg.var_scale_cpuct)
      .set_var_scale_prior_visits(cfg.var_scale_prior_visits)
      .set_tau(cfg.tau)
      .set_enable_m3_bonus(cfg.enable_m3_bonus)
      .set_m3_prior_visits(cfg.m3_prior_visits)
      .set_p_opt_weight(cfg.p_opt_weight)
      .build();
}

mcts::Search::Params MakeSearchParams(const eval::PlayerSearchConfig& cfg) {
  // Root selection policy: puct_root_policy takes priority over use_lcb.
  mcts::PuctRootSelectionPolicy root_policy;
  if (cfg.puct_root_policy == "visit_count") {
    root_policy = mcts::PuctRootSelectionPolicy::kVisitCount;
  } else if (cfg.puct_root_policy == "visit_count_sample") {
    root_policy = mcts::PuctRootSelectionPolicy::kVisitCountSample;
  } else if (cfg.puct_root_policy == "lcb" || cfg.puct_root_policy.empty()) {
    // Empty = fall back to legacy use_lcb bool.
    root_policy = (cfg.puct_root_policy == "lcb" || cfg.use_lcb)
                      ? mcts::PuctRootSelectionPolicy::kLcb
                      : mcts::PuctRootSelectionPolicy::kVisitCount;
  } else {
    root_policy = mcts::PuctRootSelectionPolicy::kLcb;  // safe default
  }

  // Q function.
  mcts::QFnKind q_fn_kind;
  if (cfg.q_fn == "identity") {
    q_fn_kind = mcts::QFnKind::kIdentity;
  } else if (cfg.q_fn == "virtual_loss_soft") {
    q_fn_kind = mcts::QFnKind::kVirtualLossSoft;
  } else {
    q_fn_kind = mcts::QFnKind::kVirtualLoss;
  }

  // N function.
  mcts::NFnKind n_fn_kind = (cfg.n_fn == "identity")
                                ? mcts::NFnKind::kIdentity
                                : mcts::NFnKind::kVirtualVisit;

  // Collision policy.
  mcts::CollisionPolicyKind collision_policy;
  if (cfg.collision_policy == "retry") {
    collision_policy = mcts::CollisionPolicyKind::kRetry;
  } else if (cfg.collision_policy == "smart_retry") {
    collision_policy = mcts::CollisionPolicyKind::kSmartRetry;
  } else {
    collision_policy = mcts::CollisionPolicyKind::kAbort;
  }

  // Collision detector.
  mcts::CollisionDetectorKind collision_detector;
  if (cfg.collision_detector == "n_in_flight") {
    collision_detector = mcts::CollisionDetectorKind::kNInFlight;
  } else if (cfg.collision_detector == "level_saturation") {
    collision_detector = mcts::CollisionDetectorKind::kLevelSaturation;
  } else if (cfg.collision_detector == "product") {
    collision_detector = mcts::CollisionDetectorKind::kProduct;
  } else {
    collision_detector = mcts::CollisionDetectorKind::kNoOp;
  }

  mcts::Search::Mode mode = (cfg.search_mode == "batch")
                                ? mcts::Search::Mode::kBatch
                                : mcts::Search::Mode::kConcurrent;

  return mcts::Search::Params{
      .num_threads = cfg.num_threads_per_game,
      .total_visit_budget = cfg.time_ms > 0 ? 1 << 20 : cfg.n,
      .total_visit_time_ms = cfg.time_ms,
      .puct_params = mcts::PuctParams::Builder()
                         .set_kind(root_policy)
                         .set_c_puct(cfg.c_puct)
                         .set_c_puct_visit_scaling(cfg.c_puct_visit_scaling)
                         .set_c_puct_v_2(cfg.c_puct_v_2)
                         .set_use_puct_v(cfg.use_puct_v)
                         .set_enable_var_scaling(cfg.var_scale_cpuct)
                         .set_var_scale_prior_visits(cfg.var_scale_prior_visits)
                         .set_tau(cfg.tau)
                         .set_enable_m3_bonus(cfg.enable_m3_bonus)
                         .set_m3_prior_visits(cfg.m3_prior_visits)
                         .set_p_opt_weight(cfg.p_opt_weight)
                         .set_root_fpu(cfg.root_fpu)
                         .build(),
      .q_fn_kind = q_fn_kind,
      .n_fn_kind = n_fn_kind,
      .descent_policy_kind = (cfg.descent_policy == "bu_uct")
                                 ? mcts::DescentPolicyKind::kBuUct
                                 : mcts::DescentPolicyKind::kDeterministic,
      .collision_policy_kind = collision_policy,
      .collision_detector_kind = collision_detector,
      .vl_delta = cfg.vl_delta,
      .max_collision_retries = cfg.max_collision_retries,
      .max_o_ratio = cfg.max_o_ratio,
      .mode = mode,
      .score_util_params = MakeScoreUtilityParams(cfg),
  };
}
}  // namespace eval