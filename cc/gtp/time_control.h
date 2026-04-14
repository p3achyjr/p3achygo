#pragma once

#include "cc/game/game.h"
#include "cc/mcts/tree.h"

namespace gtp {

// Tracks GTP clock state and computes per-move time budgets.
//
// Usage:
//   1. Call SetTimeSettings() once on receipt of the GTP `time_settings`
//      command.
//   2. Call SetTimeLeft() before each move on receipt of `time_left`.
//   3. Call ComputeMoveTimeMs() to get the time budget for the next search.
class TimeControl {
 public:
  enum Flags : uint32_t {
    kWinrateFactor = 1 << 0,
    kObviousMoveFactor = 1 << 1,
    kStddevFactor = 1 << 2,
    kMiddleGameFactor = 1 << 3,
  };
  struct Metadata {
    float base_ms;
    int appx_moves_left;
    int appx_moves_left_by_q;
    float obv_move_factor;
    float stddev_factor;
    float middlegame_factor;
  };
  TimeControl(uint32_t flags) : flags_(flags) {}

  // Called on `time_settings <main_time> <byo_yomi_time> <byo_yomi_stones>`.
  // byo_yomi_periods == 0 means sudden-death (no byoyomi).
  void SetTimeSettings(int main_time_secs, int byoyomi_time_secs,
                       int byoyomi_periods);

  // Called on `time_left <color> <seconds> <stones>`.
  // When in main time, stones == 0 from GTP; pass byoyomi_periods_left = 0.
  // When in byoyomi, seconds is the time left in the current period and
  // byoyomi_periods_left is the remaining stones (= periods for Japanese byo).
  void SetTimeLeft(int main_time_left_secs, int byoyomi_time_left_secs,
                   int byoyomi_periods_left);

  // Returns the recommended search time in milliseconds for the current move.
  // Returns 0 if time control has not been configured (time_settings not yet
  // received), which keeps the engine in its default fixed-visit mode.
  std::pair<int, Metadata> ComputeMoveTimeMs(const game::Game& game,
                                             const mcts::TreeNode* root);

  void Enable(bool enable) { enabled_ = enable; }
  bool IsEnabled() const { return enabled_; }

 private:
  // From time_settings.
  int main_time_secs_ = 0;
  int byoyomi_time_secs_ = 0;
  int byoyomi_periods_ = 0;

  // From the most recent time_left.
  int main_time_left_secs_ = 0;
  int byoyomi_time_left_secs_ = 0;
  int byoyomi_periods_left_ = 0;
  bool in_byoyomi_ = false;

  bool enabled_ = false;
  const uint32_t flags_;

  // track stddev ema
  float stddev_ema_ = 0;
};

}  // namespace gtp
