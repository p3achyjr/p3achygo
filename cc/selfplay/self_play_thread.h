#ifndef SELF_PLAY_THREAD_H_
#define SELF_PLAY_THREAD_H_

#include "absl/container/flat_hash_map.h"
#include "cc/nn/nn_interface.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/fork_manager.h"
#include "cc/selfplay/reuse_buffer.h"

namespace selfplay {

struct GumbelParams {
  int n;
  int k;
};

// Per-generation threshold calibration for sel_mult signal components.
// Values are derived from .stats files by the Python RL loop and written to
// a key=value text file passed via --sel_mult_calibration_file.
//
// Each field holds a percentile table keyed by "p01", "p05", ..., "p95",
// "p99". Use get() to look up a value with a hardcoded fallback default.
struct SelMultCalibration {
  absl::flat_hash_map<std::string, float> v_outcome_stddev;
  absl::flat_hash_map<std::string, float> v_outcome_stddev_adj;
  absl::flat_hash_map<std::string, float> pre_kld;
  absl::flat_hash_map<std::string, float> nn_mcts_diff;
  // visit_count_pre bin (rounded down to nearest 5) -> mean v_outcome_stddev.
  absl::flat_hash_map<int, float> expected_std_by_n;

  float get(const absl::flat_hash_map<std::string, float>& m,
            const std::string& pct, float def) const {
    auto it = m.find(pct);
    return it != m.end() ? it->second : def;
  }
};

struct SPConfig {
  const int max_moves;
  const GumbelParams selected_params;
  const GumbelParams default_params;
  const float use_seen_state_prob;
  // Base multiplier for selection probability. If 0, sel_mult is disabled
  // (falls back to 1.0). Otherwise, the full signal-based multiplier is
  // computed and scaled by this value.
  const float sel_mult_base = 0.0f;
  // Scale factor in [0, 1] applied to all sel_mult signal components.
  // 0 = signals have no effect (sel_mult collapses to base); 1 = full effect.
  const float sel_mult_scale_factor = 1.0f;
  // Bias cache parameters. lambda=0 disables the bias cache entirely.
  const float bias_cache_lambda = 0.0f;
  const float bias_cache_alpha = 0.8f;
  // Prior visits for nonroot var-scaling PUCT. -1 disables var scaling.
  const int nonroot_var_scale_prior_visits = 10;
  // Fork manager parameters.
  const ForkManager::Params fork_params = ForkManager::Params{};
  // Per-generation sel_mult threshold calibration.
  const SelMultCalibration calibration = {};
};

void Run(size_t seed, int thread_id, nn::NNInterface* nn_interface,
         recorder::GameRecorder* game_recorder,
         GoExploitReuseBuffer* reuse_buffer, std::string logfile,
         SPConfig config);

void SignalStop();
bool IsRunning();

}  // namespace selfplay

#endif  // __SELF_PLAY_THREAD_H_
