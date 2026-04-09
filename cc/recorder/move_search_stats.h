#pragma once

namespace recorder {

// Per-move search diagnostics collected during self-play.
//
// Aggregated into percentile tables (p0..p100 at 5% increments) and written
// to .stats text files alongside .tfrecord.zz chunks.
// Use MoveSearchStats::Builder to construct instances.
struct MoveSearchStats {
  bool sampled_raw_policy;
  float nn_q;              // NN value estimate before search (init_util_est)
  float mcts_q;            // MCTS Q pre-search (from tree reuse)
  float nn_mcts_diff;      // |nn_q - mcts_q|
  float v_outcome_stddev;  // sqrt(v_outcome_var) from tree reuse
  float prior_entropy;     // Shannon entropy of NN policy distribution H(pi)
  float nn_uncertainty;    // NN value uncertainty (v_err)
  float kld;      // KL divergence: improved policy vs NN prior (post-search)
  float pre_kld;  // KL divergence: improved policy vs NN prior (pre-search,
                  // reused tree)
  float sel_mult_modifier;         // training selection probability multiplier
  float sel_mult_modifier_weight;  // how much this sel_mult_modifier affects
                                   // move selection across the game.
  float visit_count;               // MCTS visit count for this move
  class Builder;
};

class MoveSearchStats::Builder {
 public:
  Builder& sampled_raw_policy(bool b) {
    s_.sampled_raw_policy = b;
    return *this;
  }
  Builder& nn_q(float v) {
    s_.nn_q = v;
    return *this;
  }
  Builder& mcts_q(float v) {
    s_.mcts_q = v;
    return *this;
  }
  Builder& nn_mcts_diff(float v) {
    s_.nn_mcts_diff = v;
    return *this;
  }
  Builder& v_outcome_stddev(float v) {
    s_.v_outcome_stddev = v;
    return *this;
  }
  Builder& prior_entropy(float v) {
    s_.prior_entropy = v;
    return *this;
  }
  Builder& nn_uncertainty(float v) {
    s_.nn_uncertainty = v;
    return *this;
  }
  Builder& kld(float v) {
    s_.kld = v;
    return *this;
  }
  Builder& pre_kld(float v) {
    s_.pre_kld = v;
    return *this;
  }
  Builder& sel_mult_modifier(float v) {
    s_.sel_mult_modifier = v;
    return *this;
  }
  Builder& sel_mult_modifier_weight(float v) {
    s_.sel_mult_modifier_weight = v;
    return *this;
  }
  Builder& visit_count(float v) {
    s_.visit_count = v;
    return *this;
  }
  MoveSearchStats build() { return s_; }

 private:
  MoveSearchStats s_{};
};

}  // namespace recorder
