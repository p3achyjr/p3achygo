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
  ResetGame();  // v3r: per-game reset (also called from ClearBoard)

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
  // v3r-om (2026-11-21, user): insta-move only when the pre-search root is
  // large relative to this game's history AND the best move is clearly ahead:
  // root n >= 1.2 * EMA(pre-search root n) and z >= 3, where
  // z = (Lcb(best) - Q(second)) / sqrt(QVar(second)), best/second by Q.
  // Otherwise full time (no partial cut). EMA (0.75/0.25, seeded by the first
  // root) is read before this move's update; reset per game.
  int obv_a1 = -1, obv_a2 = -1;
  float obv_z = 0;
  const float root_n_ema_before = root_n_ema_;
  const auto obvious_move_factor = [&]() {
    if (root == nullptr || root->state == mcts::TreeNodeState::kNew) {
      return 1.0f;
    }
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      if (root->child_visits[a] == 0) continue;
      if (obv_a1 == -1) {
        obv_a1 = a;
      } else if (Q(root, a) > Q(root, obv_a1)) {
        obv_a2 = obv_a1;
        obv_a1 = a;
      } else if (obv_a2 == -1 || Q(root, a) > Q(root, obv_a2)) {
        obv_a2 = a;
      }
    }
    if (obv_a1 == -1 || obv_a2 == -1) return 1.0f;
    obv_z =
        (Lcb(root, obv_a1) - Q(root, obv_a2)) / std::sqrt(QVar(root, obv_a2));
    constexpr float kRootRatio = 1.2f;
    constexpr float kZInsta = 3.0f;
    const bool big_root =
        root_n_ema_before > 0 && root->n >= kRootRatio * root_n_ema_before;
    return big_root && obv_z >= kZInsta ? 0.0f : 1.0f;
  }();
  {
    const float n = root == nullptr ? 0.0f : static_cast<float>(root->n);
    root_n_ema_ = root_n_ema_ == 0 ? n : 0.75f * root_n_ema_ + 0.25f * n;
  }
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
    // 2026-11-21 (v3r-om-ghd): grow-hold-decay. 0.7 at ply 0 -> 1.5 at ply 100
    // (linear), hold 1.5 until ply 200, linear down to 0.8 at ply 280, then
    // 0.8.
    constexpr float kStart = 0.7f, kPeak = 1.5f, kFloor = 0.8f;
    constexpr float kGrowEnd = 100.0f, kHoldEnd = 200.0f, kDecayEnd = 280.0f;
    const float n = static_cast<float>(game.num_moves());
    if (n < kGrowEnd) return kStart + (kPeak - kStart) * n / kGrowEnd;
    if (n < kHoldEnd) return kPeak;
    if (n < kDecayEnd)
      return kPeak + (kFloor - kPeak) * (n - kHoldEnd) / (kDecayEnd - kHoldEnd);
    return kFloor;
  }();

  // Winning attenuator: once the game is decided, stop spending on it and lean
  // on the ponder tree instead. Guards the adversarial case where a losing
  // opponent declines to resign and drags the game out hundreds of plies -- the
  // engine used to burn its whole clock while up 40+ points and lose on time.
  // Driven by the outcome-winrate LCB of the most-visited root child, and only
  // once the root has > kMinRootVisits visits (a small tree's LCB is not
  // trusted): 1.0 at LCB <= 90%, quadratic down to 0.0 at LCB >= 98%.
  float pre_search_winrate = -1.0f;
  const auto winning_attenuator = [&]() {
    constexpr float kLo = 0.90f;
    constexpr float kHi = 0.98f;
    constexpr int kMinRootVisits = 2000;
    if (root == nullptr || root->state == mcts::TreeNodeState::kNew) {
      return 1.0f;
    }
    int best = -1;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      if (root->child_visits[a] > 0 &&
          (best == -1 || root->child_visits[a] > root->child_visits[best])) {
        best = a;
      }
    }
    if (best == -1) return 1.0f;
    const float lcb = mcts::OutcomeWinrateLcb(root, best);
    pre_search_winrate = lcb;
    if (root->n <= kMinRootVisits) return 1.0f;
    const float t = std::clamp((lcb - kLo) / (kHi - kLo), 0.0f, 1.0f);
    return 1.0f - t * t;
  }();

  // Main time: divide remaining time evenly over estimated remaining moves,
  // then scale by all three factors.
  // Above a 90% root winrate, rely less on the value-based estimate (it
  // assumes the loser resigns soon): its weight ramps linearly 0.75 -> 0 over
  // winrate 0.90 -> 0.98, so a won game against a non-resigning opponent is
  // paced by the move-number estimate and we don't run the clock down.
  const float q_weight = [&]() {
    if (root == nullptr || root->state == mcts::TreeNodeState::kNew) {
      return 0.75f;
    }
    const float winrate = (mcts::VOutcome(root) + 1.0f) / 2.0f;
    const float t = std::clamp((winrate - 0.90f) / 0.08f, 0.0f, 1.0f);
    return 0.75f * (1.0f - t);
  }();
  const int interpolated_appx_moves_left =
      flags_ & kWinrateFactor
          ? static_cast<int>(
                std::round(q_weight * appx_moves_left_from_q +
                           (1.0f - q_weight) * appx_moves_left_from_move_num))
          : appx_moves_left_from_move_num;
  const int appx_moves_left_us =
      std::max(std::round(float(interpolated_appx_moves_left) / 2), 10.0f);

  update_stddev_ema();
  float base_ms =
      (main_time_left_secs_ * 1000.0f) / (appx_moves_left_from_move_num / 2.0f);
  // Hard reserve: never budget the last kReserveMs of the clock, so that once
  // we are down to it every move is near-instant and a lag spike or a long
  // game cannot flag us.
  constexpr float kReserveMs = 15000.0f;
  const float budget_ms =
      std::max(main_time_left_secs_ * 1000.0f - kReserveMs, 0.0f);
  float ms = budget_ms / appx_moves_left_us;
  if (flags_ & kObviousMoveFactor) ms *= obvious_move_factor;
  if (flags_ & kStddevFactor) ms *= stddev_factor;
  if (flags_ & kMiddleGameFactor) ms *= middlegame_factor;
  if (flags_ & kWinningAttenuator) ms *= winning_attenuator;
  // The floor is load-bearing: GenMoveCommon passes this straight to
  // Search::Params::total_visit_time_ms, and Run() treats <= 0 as "no timer",
  // falling back to a 1<<20 visit budget. A literal 0 would search forever.
  return {
      std::max(static_cast<int>(ms), 1),
      Metadata{base_ms,
               appx_moves_left_from_move_num,
               appx_moves_left_from_q,
               obvious_move_factor,
               stddev_factor,
               middlegame_factor,
               winning_attenuator,
               pre_search_winrate,
               /*root_stddev=*/0,
               stddev_ema_,
               /*low_root_bonus=*/1,
               main_time_left_secs_,
               obv_z,
               root_n_ema_before,
               obv_a1,
               obv_a2,
               obv_a1 >= 0 ? Q(root, obv_a1) : 0,
               obv_a1 >= 0 ? Lcb(root, obv_a1) : 0,
               obv_a2 >= 0 ? Q(root, obv_a2) : 0,
               obv_a2 >= 0 ? std::sqrt(QVar(root, obv_a2)) : 0,
               obv_a1 >= 0 ? static_cast<int>(root->child_visits[obv_a1]) : 0,
               obv_a2 >= 0 ? static_cast<int>(root->child_visits[obv_a2]) : 0}};
}

}  // namespace gtp
