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
    kWinningAttenuator = 1 << 4,
  };
  struct Metadata {
    float base_ms;
    int appx_moves_left;
    int appx_moves_left_by_q;
    float obv_move_factor;
    float stddev_factor;
    float middlegame_factor;
    float winning_attenuator;
    // What the attenuator actually saw pre-search. -1 means the root was null
    // or unevaluated, in which case the attenuator is skipped.
    float pre_search_winrate;
    // Logging only.
    float root_stddev;     // visit-normalized pre-search root stddev; 0 if n<2.
    float stddev_ema;      // after this move's update.
    float low_root_bonus;  // 1.5 at root n = 0 -> 1.0 at n >= 400.
    int main_time_left_secs;
    // Obvious-move inputs (logging; filled by builds that compute them).
    // z = (Lcb(best) - Q(second)) / sqrt(QVar(second)), best/second by Q.
    float obv_z = 0;
    float root_n_ema = 0;  // EMA of pre-search root n, before this move.
    int obv_a1 = -1, obv_a2 = -1;
    float obv_q1 = 0, obv_lcb1 = 0, obv_q2 = 0, obv_sd2 = 0;
    int obv_n1 = 0, obv_n2 = 0;
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

  // Clears per-game heuristic state (not the clock or time settings).
  // Called on clear_board and time_settings; otherwise the previous game's
  // endgame stddev seeds the next game's EMA.
  void ResetGame() {
    stddev_ema_ = 0;
    stddev_samples_ = 0;
    moves_this_game_ = 0;
    root_n_ema_ = 0;
  }

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

  // track stddev ema. Reset per game in SetTimeSettings.
  float stddev_ema_ = 0;
  int stddev_samples_ = 0;   // qualifying roots seen this game
  int moves_this_game_ = 0;  // ComputeMoveTimeMs calls (main time) this game
  float root_n_ema_ = 0;     // EMA of pre-search root n (v3r-om obvious move)
};

}  // namespace gtp
