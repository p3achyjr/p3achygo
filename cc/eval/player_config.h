#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/mcts/constants.h"
#include "cc/mcts/search.h"

namespace eval {

/*
 * All per-player search options. Shared by both cur and cand.
 * Fields mirror the corresponding ABSL flags in main.cc (legacy) or are
 * config-file-only (parallel search knobs).
 */
struct PlayerSearchConfig {
  std::string name;

  // Gumbel
  int n = 128;
  int k = 8;
  float noise_scaling = 1.0f;
  bool early_stopping_for_gumbel = false;

  // PUCT (shared between Gumbel's SearchRootPuct and parallel Search)
  bool use_puct = true;
  float c_puct = 1.0f;
  float c_puct_visit_scaling = 0.45f;
  bool var_scale_cpuct = false;
  bool use_puct_v = false;
  float c_puct_v_2 = 3.0f;
  float tau = 1.0f;

  // Root selection policy for parallel Search.
  // "visit_count" | "lcb" | "visit_count_sample"
  // Empty string = derive from the legacy use_lcb field below.
  std::string puct_root_policy = "";

  // Legacy bool kept for backward compat with existing config files.
  // Ignored when puct_root_policy is non-empty.
  bool use_lcb = true;

  // Score utility
  float score_weight = mcts::kDefaultScoreWeight;
  // "direct" | "integral"
  std::string score_utility_mode = "direct";

  // Other
  bool use_mcgs = false;
  bool use_bias_cache = false;
  float bias_cache_alpha = 0.8f;
  float bias_cache_lambda = 0.4f;
  bool enable_m3_bonus = false;
  int var_scale_prior_visits = 0;
  int m3_prior_visits = 20;
  bool enable_v_cat_var_scaling = false;
  int v_cat_var_scale_prior_visits = 10;
  float p_opt_weight = 0.0f;
  float root_fpu = 0.1f;
  // KataGo rootPolicyOptimism (logit-space root mix). < 0 disables.
  float root_p_opt_logit_weight = -1.0f;
  // KataGo valueWeightExponent (Gumbel/PUCT path only). 0 disables.
  float value_weight_exponent = 0.0f;
  // false: start every search from a fresh root (no subtree carried over).
  bool tree_reuse = true;
  // N-aware parent variance scaling (see mcts::PuctParams).
  bool enable_parent_var_scaling = false;
  float parent_var_lambda = 0.85f;
  float parent_var_prior_visits = 2.0f;
  float s_ref_default = 0.4f;
  // Loaded from `s_ref_path`: whitespace-separated stddev per floor(log2 n)
  // bin. Missing bins stay 0 and fall back to s_ref_default.
  std::array<float, mcts::PuctParams::kSRefBins> s_ref{};
  // `s_ref_poly: c0 c1 c2` -> s_ref(n) = c0 + c1 L + c2 L^2, L = log2 n.
  bool use_s_ref_poly = false;
  std::array<float, 3> s_ref_poly{};
  // Uncertainty-weighted backups (see mcts::PuctParams). 0 disables.
  float uncertainty_weight_coeff = 0.0f;
  float uncertainty_max_weight = 8.0f;
  float uncertainty_weight_scale = 1.0f;
  float child_var_shrink_k = 0.0f;
  // See mcts::PuctParams::child_sd_norm_k. `child_sd_poly: c0 c1 c2`.
  float child_sd_norm_k = 0.0f;
  bool child_sd_norm_nonroot_only = false;
  std::array<float, 3> child_sd_poly{};
  int sampling_num_moves = 0;
  float sampling_temperature = 0.0f;

  // --- Config-file-only parallel search knobs (no-op for Gumbel) ---

  // Number of parallel search worker threads per game.
  int num_threads_per_game = 1;

  // Time control (ms). 0 = disabled; use visit_budget instead.
  int time_ms = 0;

  // Enable pondering (search on opponent's time).
  bool enable_pondering = false;

  // TimeControl::Flags bitmask. 0 = all factors disabled (fixed-visit mode).
  // Pass an integer literal or "all" (= ~0u, all flags on).
  uint32_t time_control_flags = 0;

  // Q function: "identity" | "virtual_loss" | "virtual_loss_soft"
  std::string q_fn = "virtual_loss_soft";

  // N function: "identity" | "virtual_visit"
  std::string n_fn = "virtual_visit";

  // Collision policy: "abort" | "retry" | "smart_retry"
  std::string collision_policy = "abort";

  // Collision detector: "noop" | "n_in_flight" | "level_saturation" | "product"
  std::string collision_detector = "noop";

  // Virtual loss magnitude (negative). Applied when q_fn is "virtual_loss" or
  // "virtual_loss_soft".
  float vl_delta = -1.0f;

  // Maximum number of collision retries before aborting. Used by "retry" and
  // "smart_retry" collision policies.
  int max_collision_retries = 4;

  // Search mode: "concurrent" | "batch"
  std::string search_mode = "concurrent";

  // Descent policy: "deterministic" | "bu_uct"
  std::string descent_policy = "deterministic";

  // BuUct descent: maximum allowed ratio of in-flight visits to total visits.
  float max_o_ratio = 1.0f;
};

namespace internal {

inline std::string Trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

inline bool ParseBool(const std::string& val) {
  return val == "true" || val == "1";
}

}  // namespace internal

/*
 * Parses a line-based config file into a PlayerSearchConfig.
 *
 * Format (one field per line):
 *   key: value
 *   # this is a comment
 *   (blank lines are ignored)
 *
 * All PlayerSearchConfig fields are recognised by their exact field name.
 * Unknown keys are silently ignored.
 * The `name` field is never set from the file; callers set it after parsing.
 */
inline PlayerSearchConfig ParsePlayerConfigFile(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Could not open player config file: " + path);
  }

  PlayerSearchConfig cfg;
  std::string line;
  while (std::getline(f, line)) {
    line = internal::Trim(line);
    if (line.empty() || line[0] == '#') continue;

    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;

    const std::string key = internal::Trim(line.substr(0, colon));
    const std::string val = internal::Trim(line.substr(colon + 1));

    // Gumbel
    if (key == "n")
      cfg.n = std::stoi(val);
    else if (key == "k")
      cfg.k = std::stoi(val);
    else if (key == "noise_scaling")
      cfg.noise_scaling = std::stof(val);
    else if (key == "early_stopping_for_gumbel")
      cfg.early_stopping_for_gumbel = internal::ParseBool(val);
    // PUCT (shared)
    else if (key == "use_puct")
      cfg.use_puct = internal::ParseBool(val);
    else if (key == "c_puct")
      cfg.c_puct = std::stof(val);
    else if (key == "c_puct_visit_scaling")
      cfg.c_puct_visit_scaling = std::stof(val);
    else if (key == "var_scale_cpuct")
      cfg.var_scale_cpuct = internal::ParseBool(val);
    else if (key == "use_puct_v")
      cfg.use_puct_v = internal::ParseBool(val);
    else if (key == "c_puct_v_2")
      cfg.c_puct_v_2 = std::stof(val);
    else if (key == "tau")
      cfg.tau = std::stof(val);
    else if (key == "puct_root_policy")
      cfg.puct_root_policy = val;
    else if (key == "use_lcb")
      cfg.use_lcb = internal::ParseBool(val);
    // Score utility
    else if (key == "score_weight")
      cfg.score_weight = std::stof(val);
    else if (key == "score_utility_mode")
      cfg.score_utility_mode = val;
    // Other
    else if (key == "use_mcgs")
      cfg.use_mcgs = internal::ParseBool(val);
    else if (key == "use_bias_cache")
      cfg.use_bias_cache = internal::ParseBool(val);
    else if (key == "bias_cache_alpha")
      cfg.bias_cache_alpha = std::stof(val);
    else if (key == "bias_cache_lambda")
      cfg.bias_cache_lambda = std::stof(val);
    else if (key == "enable_m3_bonus")
      cfg.enable_m3_bonus = internal::ParseBool(val);
    else if (key == "var_scale_prior_visits")
      cfg.var_scale_prior_visits = std::stoi(val);
    else if (key == "m3_prior_visits")
      cfg.m3_prior_visits = std::stoi(val);
    else if (key == "enable_v_cat_var_scaling")
      cfg.enable_v_cat_var_scaling = internal::ParseBool(val);
    else if (key == "v_cat_var_scale_prior_visits")
      cfg.v_cat_var_scale_prior_visits = std::stoi(val);
    else if (key == "p_opt_weight")
      cfg.p_opt_weight = std::stof(val);
    else if (key == "root_fpu")
      cfg.root_fpu = std::stof(val);
    else if (key == "root_p_opt_logit_weight")
      cfg.root_p_opt_logit_weight = std::stof(val);
    else if (key == "value_weight_exponent")
      cfg.value_weight_exponent = std::stof(val);
    else if (key == "tree_reuse")
      cfg.tree_reuse = internal::ParseBool(val);
    else if (key == "enable_parent_var_scaling")
      cfg.enable_parent_var_scaling = internal::ParseBool(val);
    else if (key == "parent_var_lambda")
      cfg.parent_var_lambda = std::stof(val);
    else if (key == "parent_var_prior_visits")
      cfg.parent_var_prior_visits = std::stof(val);
    else if (key == "s_ref_default")
      cfg.s_ref_default = std::stof(val);
    else if (key == "uncertainty_weight_coeff")
      cfg.uncertainty_weight_coeff = std::stof(val);
    else if (key == "uncertainty_max_weight")
      cfg.uncertainty_max_weight = std::stof(val);
    else if (key == "uncertainty_weight_scale")
      cfg.uncertainty_weight_scale = std::stof(val);
    else if (key == "child_var_shrink_k")
      cfg.child_var_shrink_k = std::stof(val);
    else if (key == "child_sd_norm_k")
      cfg.child_sd_norm_k = std::stof(val);
    else if (key == "child_sd_norm_nonroot_only")
      cfg.child_sd_norm_nonroot_only = internal::ParseBool(val);
    else if (key == "child_sd_poly") {
      std::istringstream ss(val);
      if (!(ss >> cfg.child_sd_poly[0] >> cfg.child_sd_poly[1] >>
            cfg.child_sd_poly[2])) {
        throw std::runtime_error("child_sd_poly needs 3 floats: " + val);
      }
    }
    else if (key == "s_ref_path") {
      std::ifstream sf(val);
      if (!sf.is_open()) {
        throw std::runtime_error("Could not open s_ref_path: " + val);
      }
      float s;
      for (int i = 0; i < mcts::PuctParams::kSRefBins && (sf >> s); ++i) {
        cfg.s_ref[i] = s;
      }
    } else if (key == "s_ref_poly") {
      std::istringstream ss(val);
      if (!(ss >> cfg.s_ref_poly[0] >> cfg.s_ref_poly[1] >> cfg.s_ref_poly[2])) {
        throw std::runtime_error("s_ref_poly needs 3 floats: " + val);
      }
      cfg.use_s_ref_poly = true;
    } else if (key == "sampling_temperature")
      cfg.sampling_temperature = std::stof(val);
    else if (key == "sampling_num_moves")
      cfg.sampling_num_moves = std::stoi(val);
    // Parallel search knobs
    else if (key == "num_threads_per_game")
      cfg.num_threads_per_game = std::stoi(val);
    else if (key == "time_ms")
      cfg.time_ms = val == "auto" ? -1 : std::stoi(val);
    else if (key == "enable_pondering")
      cfg.enable_pondering = internal::ParseBool(val);
    else if (key == "time_control_flags")
      cfg.time_control_flags =
          (val == "all") ? ~0u : static_cast<uint32_t>(std::stoul(val));
    else if (key == "q_fn")
      cfg.q_fn = val;
    else if (key == "n_fn")
      cfg.n_fn = val;
    else if (key == "collision_policy")
      cfg.collision_policy = val;
    else if (key == "collision_detector")
      cfg.collision_detector = val;
    else if (key == "vl_delta")
      cfg.vl_delta = std::stof(val);
    else if (key == "max_collision_retries")
      cfg.max_collision_retries = std::stoi(val);
    else if (key == "search_mode")
      cfg.search_mode = val;
    else if (key == "descent_policy")
      cfg.descent_policy = val;
    else if (key == "max_o_ratio")
      cfg.max_o_ratio = std::stof(val);
    // Unknown keys are silently ignored.
  }
  return cfg;
}

/*
 * Formats two PlayerSearchConfig structs side-by-side and returns the string.
 *
 * Example output:
 *   Cur Config                        Cand Config
 *     n: 128                            n: 256
 *     k: 8                              k: 8
 *     ...
 */
inline std::string FormatPlayerConfigs(const PlayerSearchConfig& cur,
                                       const PlayerSearchConfig& cand) {
  using Row = std::tuple<std::string, std::string, std::string>;
  std::vector<Row> rows;

  auto b = [](bool v) -> std::string { return v ? "true" : "false"; };
  auto f = [](float v) -> std::string {
    std::ostringstream o;
    o << v;
    return o.str();
  };
  auto i_ = [](int v) -> std::string { return std::to_string(v); };
  auto s = [](const std::string& v) -> std::string { return v; };

#define ROW(field, fmt) \
  rows.emplace_back(#field, fmt(cur.field), fmt(cand.field))
  ROW(name, s);
  ROW(n, i_);
  ROW(k, i_);
  ROW(noise_scaling, f);
  ROW(early_stopping_for_gumbel, b);
  ROW(use_puct, b);
  ROW(c_puct, f);
  ROW(c_puct_visit_scaling, f);
  ROW(var_scale_cpuct, b);
  ROW(use_puct_v, b);
  ROW(c_puct_v_2, f);
  ROW(tau, f);
  ROW(puct_root_policy, s);
  ROW(use_lcb, b);
  ROW(score_weight, f);
  ROW(score_utility_mode, s);
  ROW(use_mcgs, b);
  ROW(use_bias_cache, b);
  ROW(bias_cache_alpha, f);
  ROW(bias_cache_lambda, f);
  ROW(enable_m3_bonus, b);
  ROW(var_scale_prior_visits, i_);
  ROW(m3_prior_visits, i_);
  ROW(enable_v_cat_var_scaling, b);
  ROW(v_cat_var_scale_prior_visits, i_);
  ROW(p_opt_weight, f);
  ROW(root_fpu, f);
  ROW(root_p_opt_logit_weight, f);
  ROW(value_weight_exponent, f);
  ROW(tree_reuse, b);
  ROW(enable_parent_var_scaling, b);
  ROW(parent_var_lambda, f);
  ROW(parent_var_prior_visits, f);
  ROW(s_ref_default, f);
  ROW(use_s_ref_poly, b);
  auto poly = [&](const std::array<float, 3>& c) {
    return f(c[0]) + " " + f(c[1]) + " " + f(c[2]);
  };
  ROW(s_ref_poly, poly);
  ROW(uncertainty_weight_coeff, f);
  ROW(uncertainty_max_weight, f);
  ROW(uncertainty_weight_scale, f);
  ROW(child_var_shrink_k, f);
  ROW(child_sd_norm_k, f);
  ROW(child_sd_norm_nonroot_only, b);
  ROW(child_sd_poly, poly);
  ROW(sampling_temperature, f);
  ROW(sampling_num_moves, i_);
  ROW(num_threads_per_game, i_);
  ROW(time_ms, i_);
  ROW(enable_pondering, b);
  ROW(time_control_flags, i_);
  ROW(q_fn, s);
  ROW(n_fn, s);
  ROW(collision_policy, s);
  ROW(collision_detector, s);
  ROW(vl_delta, f);
  ROW(max_collision_retries, i_);
  ROW(search_mode, s);
  ROW(descent_policy, s);
  ROW(max_o_ratio, f);
#undef ROW

  // Compute column widths.
  static constexpr size_t kIndent = 2;
  static constexpr size_t kColGap = 4;

  size_t key_w = 0, cur_w = 0;
  for (const auto& [k, c, d] : rows) {
    key_w = std::max(key_w, k.size());
    cur_w = std::max(cur_w, c.size());
  }

  std::ostringstream oss;
  const std::string indent(kIndent, ' ');

  // Header: "Cur Config" aligned over the left column, "Cand Config" over
  // right.
  size_t left_total = kIndent + key_w + 2 + cur_w + kColGap;
  oss << "Cur Config" << std::string(left_total - 10, ' ') << "Cand Config\n";

  for (const auto& [k, c, d] : rows) {
    std::string left = indent + k + ": " + c;
    if (left.size() < left_total)
      left += std::string(left_total - left.size(), ' ');
    oss << left << indent << k << ": " << d << "\n";
  }

  return oss.str();
}

// Parsing utils
mcts::ScoreUtilityParams MakeScoreUtilityParams(
    const eval::PlayerSearchConfig& cfg);

mcts::PuctParams MakePuctParams(const eval::PlayerSearchConfig& cfg);

// Converts a PlayerSearchConfig into Search::Params for parallel search.
// The Gumbel path in PlayEvalGame does not use this; it is here so that
// integrating parallel search is a mechanical substitution.
mcts::Search::Params MakeSearchParams(const eval::PlayerSearchConfig& cfg);

}  // namespace eval
