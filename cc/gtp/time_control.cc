#include "cc/gtp/time_control.h"

#include <algorithm>

namespace gtp {

void TimeControl::SetTimeSettings(int main_time_secs, int byoyomi_time_secs,
                                  int byoyomi_periods) {
  main_time_secs_ = main_time_secs;
  byoyomi_time_secs_ = byoyomi_time_secs;
  byoyomi_periods_ = byoyomi_periods;

  // Reset clock state for the new game.
  main_time_left_secs_ = main_time_secs;
  byoyomi_time_left_secs_ = byoyomi_time_secs;
  byoyomi_periods_left_ = byoyomi_periods;
  in_byoyomi_ = false;

  enabled_ = (main_time_secs > 0 || byoyomi_time_secs > 0);
}

void TimeControl::SetTimeLeft(int main_time_left_secs,
                              int byoyomi_time_left_secs,
                              int byoyomi_periods_left) {
  main_time_left_secs_ = main_time_left_secs;
  byoyomi_time_left_secs_ = byoyomi_time_left_secs;
  byoyomi_periods_left_ = byoyomi_periods_left;
  in_byoyomi_ = (byoyomi_periods_left > 0);
}

int TimeControl::ComputeMoveTimeMs(const game::Game& game,
                                   const mcts::TreeNode* root) const {
  if (!enabled_) {
    return 0;
  }

  if (in_byoyomi_) {
    // Use a fixed fraction of the byoyomi period.
    return std::max(0, byoyomi_time_left_secs_ * 1000 - 500);
  }

  // Main time: divide remaining time evenly over estimated remaining moves.
  const int appx_moves_left = std::max(300 - game.num_moves(), 5);
  int ms = (main_time_left_secs_ * 1000) / appx_moves_left;
  return std::max(ms, 0);
}

}  // namespace gtp
