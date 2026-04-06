#ifndef SELF_PLAY_THREAD_H_
#define SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/reuse_buffer.h"

namespace selfplay {

struct GumbelParams {
  int n;
  int k;
};

// Per-generation threshold calibration for the sel_mult signal components.
// Values are derived from .stats files by the Python RL loop and written to
// a key=value text file passed via --sel_mult_calibration_file. Defaults
// match the initial hardcoded thresholds.
struct SelMultCalibration {
  // G1 bonus breakpoints (nn_mcts_diff): linear 1.0→2.0 from p50→p95.
  float g1_p50 = 0.068f;
  float g1_p72_5 = 0.170f;
  float g1_p95 = 0.563f;
  // G2 bonus breakpoints (top12_q_gap_nz, small-gap regime).
  float g2b_p2_5 = 0.0007f;
  float g2b_p25 = 0.0276f;
  // G2 penalty breakpoints (top12_q_gap_nz, large-gap regime).
  // p70/p95 used for dynamic min_prior_gap scaling; p80/p97_5 for magnitude.
  float g2p_p70 = 0.064f;
  float g2p_p80 = 0.1057f;
  float g2p_p92_5 = 0.2445f;
  float g2p_p95 = 0.316f;
  float g2p_p97_5 = 0.4790f;
  // Stddev bonus breakpoints (v_outcome_stddev).
  float std_p70 = 0.185f;
  float std_p95 = 0.393f;
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
  // Per-generation sel_mult threshold calibration.
  const SelMultCalibration calibration = {};
};

void Run(size_t seed, int thread_id, nn::NNInterface* nn_interface,
         recorder::GameRecorder* game_recorder, ReuseBuffer* reuse_buffer,
         std::string logfile, SPConfig config);

void SignalStop();
bool IsRunning();

}  // namespace selfplay

#endif  // __SELF_PLAY_THREAD_H_
