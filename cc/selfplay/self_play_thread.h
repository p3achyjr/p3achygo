#ifndef SELF_PLAY_THREAD_H_
#define SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/fork_manager.h"
#include "cc/selfplay/reuse_buffer.h"

namespace selfplay {

struct GumbelParams {
  int n;
  int k;
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
  // Fraction of moves [0, 1] where the signal-based sel_mult is applied.
  // The remaining fraction falls back to sel_mult = 1.0.
  const float sel_mult_prob = 1.0f;
  // Bias cache parameters. lambda=0 disables the bias cache entirely.
  const float bias_cache_lambda = 0.0f;
  const float bias_cache_alpha = 0.8f;
  // Prior visits for nonroot var-scaling PUCT. -1 disables var scaling.
  const int nonroot_var_scale_prior_visits = 10;
  // Fork manager parameters.
  const ForkManager::Params fork_params = ForkManager::Params{};
};

void Run(size_t seed, int thread_id, nn::NNInterface* nn_interface,
         recorder::GameRecorder* game_recorder,
         GoExploitReuseBuffer* reuse_buffer, std::string logfile,
         SPConfig config);

void SignalStop();
bool IsRunning();

}  // namespace selfplay

#endif  // __SELF_PLAY_THREAD_H_
