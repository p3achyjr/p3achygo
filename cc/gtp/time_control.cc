#include "cc/gtp/time_control.h"

#include <algorithm>

#include "cc/constants/constants.h"
#include "cc/game/move.h"
#include "cc/mcts/tree.h"

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

std::pair<int, TimeControl::Metadata> TimeControl::ComputeMoveTimeMs(
    const game::Game& game, const mcts::TreeNode* root) {
  const auto update_stddev_ema = [&]() {
    if (root == nullptr || root->state == mcts::TreeNodeState::kNew) {
      return;
    }

    const float stddev = std::sqrt(root->v_var);
    if (stddev_ema_ == 0) {
      stddev_ema_ = stddev;
      return;
    }
    stddev_ema_ = 0.75 * stddev_ema_ + 0.25 * stddev;
  };

  if (!enabled_) {
    return {0, {}};
  }

  if (in_byoyomi_) {
    // move at last second.
    update_stddev_ema();
    return {std::max(0, byoyomi_time_left_secs_ * 1000 - 1000), {}};
  }

  constexpr int kGameLengthUpperBound = 450;
  // Never let this go non-positive. It is blended into the estimate below at
  // 0.25 weight, so a negative value *lowers* the estimate and therefore
  // *raises* time per move -- backwards, and precisely in the deep endgame
  // where the clock is tightest. It also divides base_ms.
  constexpr int kMinPliesLeft = 20;  // 10 of our moves; matches the floor below
  const auto appx_moves_left_from_move_num = [&]() {
    return std::max(kGameLengthUpperBound - game.num_moves(), kMinPliesLeft);
  }();
  const auto appx_moves_left_from_q = [&]() {
    if (root == nullptr || root->state == mcts::TreeNodeState::kNew) {
      return appx_moves_left_from_move_num;
    }
    // experimentally derived from eval games.
    const float appx_moves_left =
        std::pow((std::abs(root->v) - 1.2525) / -0.1800, 1 / 0.3386) - 1;

    // a little bump. try not to lose by time ;)
    return static_cast<int>(std::round(appx_moves_left) + 10);
  }();
  const auto obvious_move_factor = [&]() {
    // check what the z-score of the top move's LCB is compared to the 2nd best
    // move.
    if (root == nullptr || root->state == mcts::TreeNodeState::kNew) {
      return 1.0f;
    }
    std::array<int, 2> best_moves = {-1, -1};
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      if (root->child_visits[a] == 0) continue;
      if (best_moves[0] == -1) {
        best_moves[0] = a;
      } else if (Q(root, a) > Q(root, best_moves[0])) {
        best_moves[1] = best_moves[0];
        best_moves[0] = a;
      } else if (best_moves[1] == -1 || Q(root, a) > Q(root, best_moves[1])) {
        best_moves[1] = a;
      }
    }
    if (best_moves[0] == -1 || best_moves[1] == -1) return 1.0f;

    constexpr float kZLower = 1.0f;
    constexpr float kZUpper = 2.0f;
    const auto lcb_best = Lcb(root, best_moves[0]);
    const auto q_second_best = Q(root, best_moves[1]);
    const auto z_score =
        (lcb_best - q_second_best) / std::sqrt(QVar(root, best_moves[1]));
    return 1.0f -
           std::clamp((z_score - kZLower) / (kZUpper - kZLower), 0.0f, 1.0f);
  }();
  const auto stddev_factor = [&]() {
    // range: [0.5, 2.0]
    constexpr float kMinScale = 0.5f;
    constexpr float kMaxScale = 2.0f;

    if (game.num_moves() == 0) {
      return kMinScale;
    }

    if (root == nullptr) {
      // we are off policy. search more.
      return 1.0f + 0.5f * (kMaxScale - 1.0f);
    }

    const float stddev = std::sqrt(root->v_var);
    if (stddev == 0.0f || stddev_ema_ == 0.0f) {
      // how does this happen?
      return 1.0f;
    }

    return std::clamp(stddev / stddev_ema_, kMinScale, kMaxScale);
  }();
  const auto middlegame_factor = [&]() {
    // Linear baseline: 1.0 until move 150, then ramps down to 0.8 at move 399.
    // Gaussian bump centered at move 125 (sigma=50, amplitude=0.2) adds the
    // middlegame peak, giving: start≈1.0, peak≈1.2 at move 125, end≈0.8.
    constexpr float kPeakN = 125.0f;
    constexpr float kSigma = 50.0f;
    constexpr float kAmp = 0.2f;
    constexpr float kRampStart = 150.0f;
    constexpr float kRampEnd = kGameLengthUpperBound - 1.0f;
    constexpr float kFloor = 0.8f;
    const float n = static_cast<float>(game.num_moves());
    const float d = (n - kPeakN) / kSigma;
    const float bump = kAmp * std::exp(-0.5f * d * d);
    const float linear = n < kRampStart
                             ? 1.0f
                             : 1.0f - (1.0f - kFloor) * (n - kRampStart) /
                                          (kRampEnd - kRampStart);
    return linear + bump;
  }();

  // Winning attenuator: once the game is decided, stop spending on it and lean
  // on the ponder tree instead. Guards the adversarial case where a losing
  // opponent declines to resign and drags the game out hundreds of plies -- the
  // engine used to burn its whole clock while up 40+ points and lose on time.
  // 1.0 at <=80% winrate, quadratic down to 0.0 at >=95%.
  const auto winning_attenuator = [&]() {
    constexpr float kLo = 0.80f;
    constexpr float kHi = 0.95f;
    if (root == nullptr || root->state == mcts::TreeNodeState::kNew) {
      return 1.0f;
    }
    // v_outcome is the pure game outcome in [-1, 1]. root->v is utility
    // (outcome + score utility) and can exceed 1, so it is not a winrate.
    const float winrate = (root->v_outcome + 1.0f) / 2.0f;
    const float t = std::clamp((winrate - kLo) / (kHi - kLo), 0.0f, 1.0f);
    return 1.0f - t * t;
  }();

  // Main time: divide remaining time evenly over estimated remaining moves,
  // then scale by all three factors.
  const int interpolated_appx_moves_left =
      flags_ & kWinrateFactor
          ? static_cast<int>(std::round(0.75f * appx_moves_left_from_q +
                                        0.25f * appx_moves_left_from_move_num))
          : appx_moves_left_from_move_num;
  const int appx_moves_left_us =
      std::max(std::round(float(interpolated_appx_moves_left) / 2), 10.0f);

  update_stddev_ema();
  float base_ms =
      (main_time_left_secs_ * 1000.0f) / (appx_moves_left_from_move_num / 2.0f);
  float ms = (main_time_left_secs_ * 1000.0f) / appx_moves_left_us;
  if (flags_ & kObviousMoveFactor) ms *= obvious_move_factor;
  if (flags_ & kStddevFactor) ms *= stddev_factor;
  if (flags_ & kMiddleGameFactor) ms *= middlegame_factor;
  if (flags_ & kWinningAttenuator) ms *= winning_attenuator;
  // The floor is load-bearing: GenMoveCommon passes this straight to
  // Search::Params::total_visit_time_ms, and Run() treats <= 0 as "no timer",
  // falling back to a 1<<20 visit budget. A literal 0 would search forever.
  return {std::max(static_cast<int>(ms), 1),
          Metadata{base_ms, appx_moves_left_from_move_num,
                   appx_moves_left_from_q, obvious_move_factor, stddev_factor,
                   middlegame_factor, winning_attenuator}};
}

}  // namespace gtp
