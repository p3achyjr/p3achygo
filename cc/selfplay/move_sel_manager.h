#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "cc/selfplay/self_play_thread.h"

namespace selfplay {

enum MoveSelFlags : uint32_t {
  kStddevBonus = 1 << 0,
  kStddevPenalty = 1 << 1,
  kKldBonus = 1 << 2,
  kKldPenalty = 1 << 3,
  kNnMctsBonus = 1 << 4,
};

struct MoveSelResult {
  float modifier;  // effective modifier (sel_mult_scale_factor applied)
  float modifier_unscaled;  // modifier without sel_mult_scale_factor
  float sel_bonus;          // q-adjusted combined bonus
  float sel_penalty;        // q-adjusted combined penalty
  float sel_std_bonus;      // raw stddev bonus (pre q-adjust)
  float sel_std_penalty;    // raw stddev penalty (pre q-adjust)
  float sel_kld_bonus;      // raw kld bonus (pre q-adjust)
  float sel_kld_penalty;    // raw kld penalty (pre q-adjust)
  float sel_nn_mcts_bonus;  // raw |NN-MCTS| bonus (pre q-adjust)
  float sel_q_adjust;
  float std_adj;      // v_outcome_stddev / expected_std(n_pre)
  float std_adj_att;  // std_adj attenuated toward 1 at low n_pre
};

class MoveSelManager {
 public:
  MoveSelManager(uint32_t flags, const SelMultCalibration& calibration)
      : flags_(flags), calibration_(calibration) {}

  MoveSelResult Compute(int n_pre, float std_dev, float pre_kld,
                        float nn_mcts_diff, float q_canonical,
                        float sel_mult_scale_factor) const {
    const float std_adj = StdAdj(n_pre, std_dev);
    const float std_adj_att = StdAdjAtt(n_pre, std_adj);
    const float sel_q_adjust = SelQAdjust(q_canonical);

    // Always compute all signals for logging; flags gate which are applied.
    const float sel_std_bonus = StdBonus(std_adj_att);
    const float sel_std_penalty = StdPenalty(std_adj_att);
    const float sel_kld_bonus = KldBonus(pre_kld);
    const float sel_kld_penalty = KldPenalty(pre_kld);
    const float sel_nn_mcts_bonus = NnMctsBonus(nn_mcts_diff);

    constexpr float kMaxBonus = 2.5f;
    const float raw_bonus =
        std::min(std::max({flags_ & kStddevBonus ? sel_std_bonus : 1.0f,
                           flags_ & kKldBonus ? sel_kld_bonus : 1.0f,
                           flags_ & kNnMctsBonus ? sel_nn_mcts_bonus : 1.0f}),
                 kMaxBonus);
    const float raw_penalty =
        std::min(flags_ & kStddevPenalty ? sel_std_penalty : 1.0f,
                 flags_ & kKldPenalty ? sel_kld_penalty : 1.0f);

    const float sel_bonus = 1.0f + sel_q_adjust * (raw_bonus - 1.0f);
    const float sel_penalty = 1.0f + sel_q_adjust * (raw_penalty - 1.0f);
    const float modifier_unscaled = sel_bonus * sel_penalty;
    const float modifier =
        1.0f + sel_mult_scale_factor * (modifier_unscaled - 1.0f);

    return MoveSelResult{modifier,      modifier_unscaled, sel_bonus,
                         sel_penalty,   sel_std_bonus,     sel_std_penalty,
                         sel_kld_bonus, sel_kld_penalty,   sel_nn_mcts_bonus,
                         sel_q_adjust,  std_adj,           std_adj_att};
  }

 private:
  float StdAdj(int n_pre, float std_dev) const {
    if (std_dev == 0.0f) return 0.0f;
    const auto& m = calibration_.expected_std_by_n;
    if (m.empty()) return 0.0f;
    constexpr int kCap = 200;
    const int query = std::min((n_pre / 5) * 5, kCap);
    struct Neighbor {
      int bin;
      float val;
    };
    std::vector<Neighbor> neighbors;
    neighbors.reserve(m.size());
    for (const auto& [bin, val] : m) {
      if (val > 0.0f) neighbors.push_back({bin, val});
    }
    std::sort(neighbors.begin(), neighbors.end(),
              [&](const Neighbor& a, const Neighbor& b) {
                return std::abs(a.bin - query) < std::abs(b.bin - query);
              });
    const int k = std::min(4, static_cast<int>(neighbors.size()));
    float sum_w = 0.0f, sum_wv = 0.0f;
    for (int i = 0; i < k; ++i) {
      const float dist = std::abs(neighbors[i].bin - query);
      const float w = 1.0f / (dist + 5.0f);
      sum_w += w;
      sum_wv += w * neighbors[i].val;
    }
    const float expected = sum_wv / sum_w;
    return expected > 0.0f ? std_dev / expected : 0.0f;
  }

  // Attenuate std_adj toward 1.0 at low n_pre (noisy estimates).
  // att = min(1.0, 0.2 + 0.8*(n_pre/40)^0.54):
  //   att(0)=0.20, att(10)=0.58, att(20)=0.75, att(32)=0.91, att(40+)=1.0
  float StdAdjAtt(int n_pre, float std_adj) const {
    if (std_adj == 0.0f) return 0.0f;
    const float att =
        std::min(1.0f, 0.2f + 0.8f * std::pow(n_pre / 40.0f, 0.54f));
    return 1.0f + (std_adj - 1.0f) * att;
  }

  // At very won/lost positions signals are naturally weak; attenuate
  // bonus/penalty to avoid artificially penalizing these positions.
  float SelQAdjust(float q_canonical) const {
    const float base =
        1.0f - std::clamp((std::abs(q_canonical) - 0.5f) / 0.4f, 0.0f, 1.0f);
    return std::pow(base, 0.4f);
  }

  float StdBonus(float sa) const {
    if (sa == 0.0f) return 1.0f;
    const float lb =
        calibration_.get(calibration_.v_outcome_stddev_adj, "p80", 1.52f);
    const float ub =
        calibration_.get(calibration_.v_outcome_stddev_adj, "p99", 4.96f);
    if (sa <= lb || ub <= lb) return 1.0f;
    return 1.0f + 0.5f * (sa - lb) / (ub - lb);
  }

  float StdPenalty(float sa) const {
    if (sa == 0.0f) return 1.0f;
    constexpr float kFloor = 0.3f;
    const float lb =
        calibration_.get(calibration_.v_outcome_stddev_adj, "p01", 0.02f);
    const float ub =
        calibration_.get(calibration_.v_outcome_stddev_adj, "p50", 0.64f);
    if (sa >= ub) return 1.0f;
    if (sa <= lb || ub <= lb) return kFloor;
    return 1.0f - (1.0f - kFloor) * (ub - sa) / (ub - lb);
  }

  float KldBonus(float pre_kld) const {
    const float lb = calibration_.get(calibration_.pre_kld, "p70", 0.310f);
    const float ub = calibration_.get(calibration_.pre_kld, "p95", 1.166f);
    if (pre_kld == 0.0f || pre_kld <= lb || ub <= lb) return 1.0f;
    return std::min(1.5f, 1.0f + 0.5f * (pre_kld - lb) / (ub - lb));
  }

  float KldPenalty(float pre_kld) const {
    constexpr float kFloor = 0.3f;
    const float lb = calibration_.get(calibration_.pre_kld, "p05", 0.0001f);
    constexpr float kUb = 0.06f;
    if (pre_kld == 0.0f || pre_kld >= kUb) return 1.0f;
    if (pre_kld <= lb || kUb <= lb) return kFloor;
    return 1.0f - (1.0f - kFloor) * (kUb - pre_kld) / (kUb - lb);
  }

  // Bonus for high |NN-MCTS| disagreement (pre-search).
  // Guard: returns 1.0 if signal is zero (uninitialized root).
  float NnMctsBonus(float nn_mcts_diff) const {
    if (nn_mcts_diff == 0.0f) return 1.0f;
    const float lb =
        calibration_.get(calibration_.nn_mcts_diff, "p70", 0.1463f);
    const float ub =
        calibration_.get(calibration_.nn_mcts_diff, "p99", 0.6500f);
    if (nn_mcts_diff <= lb || ub <= lb) return 1.0f;
    return 1.0f + 0.60f * (nn_mcts_diff - lb) / (ub - lb);
  }

  uint32_t flags_;
  const SelMultCalibration& calibration_;
};

}  // namespace selfplay
