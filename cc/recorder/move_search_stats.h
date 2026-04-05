#pragma once

namespace recorder {

// Per-move search diagnostics collected during self-play.
//
// Aggregated into percentile tables (p0..p100 at 5% increments) and written
// to .stats text files alongside .tfrecord.zz chunks.
// Use MoveSearchStats::Builder to construct instances.
struct MoveSearchStats {
  float nn_q;              // NN value estimate before search (init_util_est)
  float mcts_q;            // MCTS Q pre-search (from tree reuse)
  float nn_mcts_diff;      // |nn_q - mcts_q|
  float v_outcome_stddev;  // sqrt(v_outcome_var) from tree reuse
  float top12_q_gap;       // Q gap between top-1 and top-2 visited children
  float prior_gap;         // policy prior gap between top-1 and top-2 visited
  float prior_entropy;     // Shannon entropy of NN policy distribution H(pi)
  float nn_uncertainty;    // NN value uncertainty (v_err)
  float q_variance;        // Q variance among visited children
  float kld;               // KL divergence: improved policy vs NN prior
  float sel_mult;          // training selection probability multiplier
  float visit_count;       // MCTS visit count for this move
  class Builder;
};

class MoveSearchStats::Builder {
 public:
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
  Builder& top12_q_gap(float v) {
    s_.top12_q_gap = v;
    return *this;
  }
  Builder& prior_gap(float v) {
    s_.prior_gap = v;
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
  Builder& q_variance(float v) {
    s_.q_variance = v;
    return *this;
  }
  Builder& kld(float v) {
    s_.kld = v;
    return *this;
  }
  Builder& sel_mult(float v) {
    s_.sel_mult = v;
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
