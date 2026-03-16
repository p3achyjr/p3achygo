#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

#include "cc/mcts/constants.h"

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

  // --- Config-file-only parallel search knobs (no-op for Gumbel) ---

  // Number of parallel search worker threads per game.
  int num_threads_per_game = 1;

  // Time control (ms). 0 = disabled; use visit_budget instead.
  int time_ms = 0;

  // Q function: "identity" | "virtual_loss" | "virtual_loss_soft"
  std::string q_fn = "virtual_loss";

  // N function: "identity" | "virtual_visit"
  std::string n_fn = "virtual_visit";

  // Collision policy: "abort" | "retry" | "smart_retry"
  std::string collision_policy = "abort";

  // Collision detector: "noop" | "n_in_flight" | "level_saturation" | "product"
  std::string collision_detector = "noop";

  // Virtual loss magnitude (negative). Applied when q_fn is "virtual_loss" or
  // "virtual_loss_soft".
  float vl_delta = -1.5f;

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
    // Parallel search knobs
    else if (key == "num_threads_per_game")
      cfg.num_threads_per_game = std::stoi(val);
    else if (key == "time_ms")
      cfg.time_ms = std::stoi(val);
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

}  // namespace eval
