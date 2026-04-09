#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "cc/core/heap.h"
#include "cc/core/probability.h"
#include "cc/core/rand.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/tree.h"
#include "cc/selfplay/fork_kind.h"
#include "cc/selfplay/reuse_buffer.h"

namespace selfplay {

// Manages position diversity for a single self-play game by sampling
// alternative continuations and adding them to the reuse buffer.
//
// Create once per game. For non-fresh games (kGoExploit restarts),
// pass `started_from_forced_search=true`; all methods become no-ops.
//
// Lifecycle:
//   1. Construct — samples a ForkKind and target move number.
//   2. Call MaybeFork() on every move — fires once at the target move.
//   3. Call FinalizeGame() after the game loop (kRegret only).
class ForkManager final {
 public:
  // Conditional probabilities at baseline reuse_prob=0.2.
  static constexpr float kBaseProbs[5] = {0.0f, 0.08f, 0.0f, 0.0f, 0.02f};
  struct Params {
    // Probability of each fork kind. Must sum to 1.
    // Tuned for reuse_chance=0.2 (conditional on being a fresh game).
    float early_fork_prob = kBaseProbs[0];
    float late_fork_prob = kBaseProbs[1];
    float sample_policy_t1_prob = kBaseProbs[2];  // absorbed into uniform.
    float sample_policy_t2_prob = kBaseProbs[3];
    float sample_random_prob = kBaseProbs[4];
    float regret_prob = 0.0f;
    float uniform_prob = 1.0f - (kBaseProbs[0] + kBaseProbs[1] + kBaseProbs[2] +
                                 kBaseProbs[3] + kBaseProbs[4]);
    // Post-fork probabilities (independent; need not sum to 1).
    float force_full_search_prob = 0.25f;
    float double_sample_prob = 0.5f;

    class Builder;

    // Rescales fork probabilities so that the 5 active fork kinds maintain
    // their effective per-game rates across different reuse fractions.
    // Baseline: reuse_prob=0.2. Scales up/down from there.
    static Params ForReuse(float reuse_prob) {
      static constexpr float kBaseReuse = 0.2f;  // 1 - baseline reuse 0.2
      const float scale = reuse_prob == 0 ? 0 : (kBaseReuse / reuse_prob);

      Params p;
      p.early_fork_prob = kBaseProbs[0] * scale;
      p.late_fork_prob = kBaseProbs[1] * scale;
      p.sample_policy_t1_prob = kBaseProbs[2] * scale;
      p.sample_policy_t2_prob = kBaseProbs[3] * scale;
      p.sample_random_prob = kBaseProbs[4] * scale;
      float fork_sum = p.early_fork_prob + p.late_fork_prob +
                       p.sample_policy_t1_prob + p.sample_policy_t2_prob +
                       p.sample_random_prob;

      if (fork_sum >= 1.0f) {
        // Scale down to reserve at least 10% mass for regret/uniform.
        const float scale_down = 0.9f / fork_sum;
        p.early_fork_prob *= scale_down;
        p.late_fork_prob *= scale_down;
        p.sample_policy_t1_prob *= scale_down;
        p.sample_policy_t2_prob *= scale_down;
        p.sample_random_prob *= scale_down;
        fork_sum = 0.9f;
      }
      const float remaining = 1.0f - fork_sum;
      p.regret_prob = 0.0f;
      p.uniform_prob = remaining;
      return p;
    }
  };

  // Per-move data passed to MaybeFork.
  struct MoveData {
    const game::Board& board;  // Board state BEFORE the move was played.
    game::Color color;
    game::Loc move;
    float nn_value;
    float mcts_value;
    bool is_eligible;  // true iff move has MCTS visits (move_n != 0)
  };

  explicit ForkManager(Params params, GoExploitReuseBuffer* reuse_buffer,
                       mcts::GumbelEvaluator* evaluator,
                       core::Probability& prob, bool started_from_forced_search)
      : params_(params),
        reuse_buffer_(reuse_buffer),
        evaluator_(evaluator),
        did_fork_(false),
        started_from_forced_search_(started_from_forced_search) {
    // Samples a move number from a trapezoidal distribution:
    //   [kFlatStart, kFlatEnd) : uniform     (kFlatMass of total probability)
    //   [kFlatEnd,   kMax)     : linearly decaying to 0 at kMax
    const auto sample_trapezoidal = [&]() {
      constexpr int kFlatStart = 10;
      constexpr int kFlatEnd = 100;
      constexpr int kMax = 250;
      constexpr float kFlatMass = 0.6f;
      constexpr float kFlatDensity = kFlatMass / (kFlatEnd - kFlatStart);
      constexpr float kTailStartDensity =
          2.0f * (1.0f - kFlatMass) / (kMax - kFlatEnd);
      constexpr float kTailSlope = kTailStartDensity / (kMax - kFlatEnd);

      const float p = prob.Uniform();
      float cumulative = 0.0f;
      for (int mv = kFlatStart; mv < kMax; ++mv) {
        cumulative += mv < kFlatEnd
                          ? kFlatDensity
                          : kTailStartDensity - kTailSlope * (mv - kFlatEnd);
        if (p <= cumulative) return mv;
      }
      return kMax;
    };

    const float total =
        params.early_fork_prob + params.late_fork_prob +
        params.sample_policy_t1_prob + params.sample_policy_t2_prob +
        params.sample_random_prob + params.regret_prob + params.uniform_prob;
    CHECK(std::abs(total - 1.0f) < 1e-4f)
        << "Fork kind probabilities must sum to 1, got " << total;

    const float p = prob.Uniform();
    float cumulative = params.early_fork_prob;
    if (p < cumulative) {
      kind_ = ForkKind::kEarly;
      fork_mv_num_ = std::round(prob.Exponential() * 9);
    } else if (p < (cumulative += params.late_fork_prob)) {
      kind_ = ForkKind::kLate;
      fork_mv_num_ = sample_trapezoidal();
    } else if (p < (cumulative += params.sample_policy_t1_prob)) {
      kind_ = ForkKind::kSampleT1;
      fork_mv_num_ = sample_trapezoidal();
    } else if (p < (cumulative += params.sample_policy_t2_prob)) {
      kind_ = ForkKind::kSampleT2;
      fork_mv_num_ = sample_trapezoidal();
    } else if (p < (cumulative += params.sample_random_prob)) {
      kind_ = ForkKind::kSampleUniform;
      fork_mv_num_ = sample_trapezoidal();
    } else if (p < (cumulative += params.regret_prob)) {
      kind_ = ForkKind::kRegret;
      fork_mv_num_ = -1;  // unused
    } else {
      kind_ = ForkKind::kUniform;
      fork_mv_num_ = sample_trapezoidal();
    }
  }

  void MaybeFork(const game::Game& game, const MoveData& data,
                 core::Probability& prob) {
    if (started_from_forced_search_) return;
    if (game.IsGameOver()) return;

    if (kind_ == ForkKind::kRegret) {
      // Accumulate per-move data for end-of-game regret computation.
      positions_for_regret_.emplace_back(data.color, data.board, data.move,
                                         data.nn_value, data.mcts_value,
                                         data.is_eligible);
      return;
    }

    if (did_fork_) return;

    const int move_num = game.num_moves();
    if (move_num != fork_mv_num_) return;
    did_fork_ = true;

    using LastMoves = absl::InlinedVector<game::Move, constants::kNumLastMoves>;

    const game::Color color = data.color;
    const game::Color opponent_color = game::OppositeColor(color);
    const game::Board& board = data.board;

    // kUniform: restart from the current position, optionally adjusting komi
    // to make the game score-neutral (70% probability of adjustment).
    if (kind_ == ForkKind::kUniform) {
      game::Board restart_board = board;
      const LastMoves last_moves = BuildLastMoves(game, move_num, {});
      const EvalResult eval = EvalBoard(board, color, last_moves, prob);
      const float komi_delta = ComputeKomiDelta(eval.init_score_est, color);
      const float p_adjust_komi =
          std::atan(std::abs(eval.init_score_est) / 3.0) * M_2_PI;
      if (prob.Uniform() < p_adjust_komi) {
        restart_board.SetKomi(board.komi() + komi_delta);
      }
      reuse_buffer_->Add(InitState{
          restart_board, BuildLastMoves(game, move_num, {}), color, move_num,
          FirstMoveBehavior::kSample, InitState::Kind::kGoExploit, kind_});
      return;
    }

    // kEarly / kLate / kSampleT1 / kSampleT2 / kSampleUniform:
    // Sample one (or two) alternative moves and add the resulting position.
    const int num_candidates =
        (kind_ == ForkKind::kEarly)  ? core::RandRange(prob.prng(), 3, 13)
        : (kind_ == ForkKind::kLate) ? core::RandRange(prob.prng(), 5, 37)
                                     : 0;

    // Returns the indices of all legal moves for `color` on `board`.
    const auto get_legal_moves = [](const game::Board& board,
                                    game::Color color) {
      absl::InlinedVector<int, constants::kMaxMovesPerPosition> legal;
      for (int action = 0; action < constants::kMaxMovesPerPosition; ++action) {
        if (board.IsValidMove(game::AsLoc(action), color))
          legal.push_back(action);
      }
      return legal;
    };

    // Shifts `last_moves` left by one slot and appends `move`.
    const auto shift_last_moves = [](const LastMoves& last_moves,
                                     game::Move move) {
      LastMoves shifted(last_moves.begin() + 1, last_moves.end());
      shifted.push_back(move);
      return shifted;
    };

    // Samples a uniformly random legal move for `color` on `board`.
    const auto sample_uniform_move = [&](const game::Board& board,
                                         game::Color color) -> game::Loc {
      const auto legal = get_legal_moves(board, color);
      if (legal.empty()) return game::kNoopLoc;
      return game::AsLoc(
          legal[core::RandRange(prob.prng(), 0, (int)legal.size())]);
    };

    // Samples a move proportional to policy probabilities (or sqrt of policy
    // if `use_sqrt`). Falls back to uniform if all weights are zero.
    const auto sample_from_policy =
        [&](const game::Board& board, game::Color color,
            const LastMoves& last_moves, bool use_sqrt) -> game::Loc {
      const auto legal = get_legal_moves(board, color);
      if (legal.empty()) return game::kNoopLoc;
      const EvalResult eval = EvalBoard(board, color, last_moves, prob);
      absl::InlinedVector<float, constants::kMaxMovesPerPosition> weights;
      float weight_sum = 0.0f;
      for (int action : legal) {
        const float weight = use_sqrt ? std::sqrt(eval.move_probs[action])
                                      : eval.move_probs[action];
        weights.push_back(weight);
        weight_sum += weight;
      }
      if (weight_sum <= 0.0f) return sample_uniform_move(board, color);
      const float target = prob.Uniform() * weight_sum;
      float cumulative = 0.0f;
      for (int i = 0; i < (int)weights.size(); ++i) {
        cumulative += weights[i];
        if (target <= cumulative) return game::AsLoc(legal[i]);
      }
      return game::AsLoc(legal.back());
    };

    // Evaluates `num_candidates` randomly sampled legal moves and returns
    // the one that minimizes the opponent's utility on the resulting position.
    const auto sample_best_of_n =
        [&](const game::Board& board, game::Color color,
            const LastMoves& last_moves, int num_candidates) -> game::Loc {
      const game::Color opponent = game::OppositeColor(color);
      auto candidates = get_legal_moves(board, color);
      if (candidates.empty()) return game::kNoopLoc;
      const int take = std::min(num_candidates, (int)candidates.size());
      // Partial Fisher-Yates to uniformly select `take` candidates.
      for (int i = 0; i < take; ++i) {
        const int j =
            i + core::RandRange(prob.prng(), 0, (int)candidates.size() - i);
        std::swap(candidates[i], candidates[j]);
      }
      game::Loc best_move = game::kNoopLoc;
      float best_opp_util = std::numeric_limits<float>::max();
      for (int i = 0; i < take; ++i) {
        const game::Loc candidate = game::AsLoc(candidates[i]);
        game::Board candidate_board = board;
        candidate_board.PlayMove(candidate, color);
        const EvalResult eval =
            EvalBoard(candidate_board, opponent,
                      shift_last_moves(last_moves, {color, candidate}), prob);
        if (eval.init_util_est < best_opp_util) {
          best_opp_util = eval.init_util_est;
          best_move = candidate;
        }
      }
      return best_move;
    };

    // Dispatches to the appropriate sampling strategy for this ForkKind.
    const auto sample_alt_move = [&](const game::Board& board,
                                     game::Color color,
                                     const LastMoves& last_moves) -> game::Loc {
      switch (kind_) {
        case ForkKind::kEarly:
        case ForkKind::kLate:
          return sample_best_of_n(board, color, last_moves, num_candidates);
        case ForkKind::kSampleT1:
          return sample_from_policy(board, color, last_moves,
                                    /*use_sqrt=*/false);
        case ForkKind::kSampleT2:
          return sample_from_policy(board, color, last_moves,
                                    /*use_sqrt=*/true);
        case ForkKind::kSampleUniform:
          return sample_uniform_move(board, color);
        default:
          return game::kNoopLoc;
      }
    };

    const LastMoves origin_last_moves = BuildLastMoves(game, move_num, {});

    // Sample the first alternative move at P (the current position).
    const game::Loc alt_move = sample_alt_move(board, color, origin_last_moves);
    if (alt_move == game::kNoopLoc) return;

    game::Board fork_board = board;
    fork_board.PlayMove(alt_move, color);
    const FirstMoveBehavior first_move_behavior =
        (kind_ == ForkKind::kSampleUniform ||
         prob.Uniform() < params_.force_full_search_prob)
            ? FirstMoveBehavior::kForceFullSearch
            : FirstMoveBehavior::kPlay;
    const LastMoves first_fork_last_moves =
        BuildLastMoves(game, move_num, {{color, alt_move}});

    // Optionally double-sample: also pick an opponent move at P'.
    // On success, add P'' (not P'); on failure, fall through to add P'.
    if (prob.Uniform() < params_.double_sample_prob) {
      const game::Loc alt_move2 =
          sample_alt_move(fork_board, opponent_color, first_fork_last_moves);
      if (alt_move2 != game::kNoopLoc) {
        const LastMoves second_fork_last_moves = BuildLastMoves(
            game, move_num, {{color, alt_move}, {opponent_color, alt_move2}});
        fork_board.PlayMove(alt_move2, opponent_color);
        const float adj_komi = ComputeAdjKomi(color, color, board, fork_board,
                                              second_fork_last_moves);
        if (prob.Uniform() < 0.5f) {
          fork_board.SetKomi(adj_komi);
        }
        reuse_buffer_->Add(InitState{
            fork_board,
            BuildLastMoves(game, move_num,
                           {{color, alt_move}, {opponent_color, alt_move2}}),
            color, move_num + 2, first_move_behavior,
            InitState::Kind::kGoExploit, kind_});
        return;
      }
    }

    // Single-sample: add P'.
    const float adj_komi = ComputeAdjKomi(color, opponent_color, board,
                                          fork_board, first_fork_last_moves);
    fork_board.SetKomi(adj_komi);
    reuse_buffer_->Add(InitState{
        fork_board, BuildLastMoves(game, move_num, {{color, alt_move}}),
        opponent_color, move_num + 1, first_move_behavior,
        InitState::Kind::kGoExploit, kind_});
  }

  void FinalizeGame(const game::Game& game, core::Probability& prob) {
    if (kind_ != ForkKind::kRegret) return;
    if (started_from_forced_search_) return;
    if (positions_for_regret_.empty()) return;
    CHECK((int)positions_for_regret_.size() == game.num_moves());

    // EMA decay per step; weight drops to ~5% at kRegretHorizon steps.
    static constexpr float kRegretEmaDecay = 0.94f;
    static constexpr int kRegretHorizon = 50;

    struct RegretCandidate {
      float regret_score;
      int move_num;
      game::Color color;
      game::Board board;
      game::Loc move;
      float mcts_value;
      float game_outcome;
    };
    struct RegretCandidateCmp {
      bool operator()(const RegretCandidate& a,
                      const RegretCandidate& b) const {
        return a.regret_score < b.regret_score;
      }
    };
    core::Heap<RegretCandidate, RegretCandidateCmp> regret_heap(
        RegretCandidateCmp{});

    for (int move_num = 0; move_num < (int)positions_for_regret_.size();
         ++move_num) {
      const auto& [color, board, move, nn_value, mcts_value, is_eligible] =
          positions_for_regret_[move_num];
      if (!is_eligible) continue;

      const float game_outcome = game.result().winner == color ? 1.5f : -1.5f;

      // Weighted average of future Q values from `color`'s perspective.
      float future_value_ema = 0.0f;
      float weight = 1.0f;
      float weight_sum = 0.0f;
      for (int k = 1; k < kRegretHorizon &&
                      move_num + k < (int)positions_for_regret_.size();
           ++k) {
        const auto& [future_color, future_board, future_move, future_nn_value,
                     future_mcts_value, future_eligible] =
            positions_for_regret_[move_num + k];
        weight *= kRegretEmaDecay;
        if (!future_eligible) continue;
        const float future_value_for_color =
            (future_color == color) ? future_mcts_value : -future_mcts_value;
        future_value_ema += weight * future_value_for_color;
        weight_sum += weight;
      }
      if (weight_sum > 0.0f) future_value_ema /= weight_sum;

      const float smoothed_value =
          (mcts_value + future_value_ema * kRegretEmaDecay) /
          (1.0f + kRegretEmaDecay);
      const float nn_miseval = std::abs(nn_value - smoothed_value);
      const float wr_drift = std::abs(mcts_value - future_value_ema);
      const float value_error = std::max(
          smoothed_value - game_outcome - std::abs(game_outcome), 0.0f);
      const float regret_score = nn_miseval * nn_miseval + wr_drift * wr_drift +
                                 value_error * value_error;

      // Attenuate by winrate magnitude and move number.
      const float winrate_weight = [](float value) {
        static constexpr float kMaxV = 0.9f;
        static constexpr float kAnnealStart = 0.5f;
        if (std::abs(value) > kMaxV) return 0.0f;
        if (std::abs(value) <= kAnnealStart) return 1.0f;
        return (kMaxV - std::abs(value)) / (kMaxV - kAnnealStart);
      }(mcts_value);
      const float move_num_weight = [](int abs_move_num) {
        static constexpr int kMinAnnealMove = 100;
        static constexpr int kInterval = 100;
        const float offset = static_cast<float>(
            std::clamp(abs_move_num - kMinAnnealMove, 0, kInterval));
        const float fraction = 1.0f - (offset / kInterval);
        return static_cast<float>(
            std::clamp(std::pow(fraction, 1.2), 0.0, 1.0));
      }(game.init_mv_num() + move_num);

      if (prob.Uniform() >= winrate_weight * move_num_weight) continue;
      regret_heap.PushHeap({regret_score, move_num, color, board, move,
                            mcts_value, game_outcome});
    }

    if (regret_heap.Size() == 0) return;

    const RegretCandidate top = regret_heap.PopHeap();
    const FirstMoveBehavior first_move_behavior =
        prob.Uniform() < params_.force_full_search_prob
            ? FirstMoveBehavior::kForceFullSearch
            : FirstMoveBehavior::kSample;

    absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
    last_moves.reserve(constants::kNumLastMoves);
    for (int offset = constants::kNumLastMoves; offset > 0; --offset) {
      last_moves.emplace_back(game.move(top.move_num - offset));
    }
    reuse_buffer_->Add(InitState{top.board, last_moves, top.color, top.move_num,
                                 first_move_behavior,
                                 InitState::Kind::kGoExploit, kind_});
  }

 private:
  struct EvalResult {
    float init_util_est;
    float init_score_est;
    std::array<float, constants::kMaxMovesPerPosition> move_probs;
  };

  // Returns the komi delta needed to make the fork position score-neutral
  // from `color`'s perspective. `fork_score` is the NN score estimate at
  // the fork in points from `color`'s perspective (includes current komi).
  //
  // Since komi is added to White's score:
  //   Black winning by S → raise komi by S  (delta = +S)
  //   White winning by S → lower komi by S  (delta = -S)
  //
  // Rounded to the nearest integer to keep komi in (k + 0.5) form.
  static float ComputeKomiDelta(float fork_score, game::Color color) {
    return std::round((color == BLACK) ? fork_score : -fork_score);
  }

  // Performs a single NN evaluation at `board` from `color`'s perspective.
  // Creates a temporary MctsNodeTable; returns utility, score, and policy.
  EvalResult EvalBoard(
      const game::Board& board, game::Color color,
      const absl::InlinedVector<game::Move, constants::kNumLastMoves>&
          last_moves,
      core::Probability& prob) {
    auto node_table = std::make_unique<mcts::MctsNodeTable>();
    mcts::TreeNode* root_node =
        node_table->GetOrCreate(board.hash(), color, /*is_terminal=*/false);
    game::Game temp_game(board, last_moves, 0);
    evaluator_->SearchRoot(
        prob, temp_game, node_table.get(), root_node, color,
        mcts::GumbelSearchParams::Builder().set_n(1).set_k(1).build());
    return {root_node->init_util_est, root_node->init_score_est,
            root_node->move_probs};
  }

  // Evaluates `fork_board` using a fresh PRNG (so the main game's PRNG is
  // unaffected). Returns the komi that makes the fork position score-neutral
  // from `orig_color`'s perspective.
  //
  // `fork_color` is the color to move at `fork_board`. Pass `orig_color` for
  // double-sample (P'', same side to move) or the opponent for single-sample
  // (P', opponent to move); the score is negated accordingly.
  float ComputeAdjKomi(
      game::Color orig_color, game::Color fork_color,
      const game::Board& orig_board, const game::Board& fork_board,
      const absl::InlinedVector<game::Move, constants::kNumLastMoves>&
          fork_last_moves) {
    core::Probability eval_prob;
    const EvalResult fork_eval =
        EvalBoard(fork_board, fork_color, fork_last_moves, eval_prob);
    const float fork_score = (orig_color == fork_color)
                                 ? fork_eval.init_score_est
                                 : -fork_eval.init_score_est;
    return orig_board.komi() + ComputeKomiDelta(fork_score, orig_color);
  }

  // Returns the string name of the current ForkKind for logging.
  const char* KindName() const {
    switch (kind_) {
      case ForkKind::kEarly:
        return "kEarly";
      case ForkKind::kLate:
        return "kLate";
      case ForkKind::kSampleT1:
        return "kSampleT1";
      case ForkKind::kSampleT2:
        return "kSampleT2";
      case ForkKind::kSampleUniform:
        return "kSampleUniform";
      case ForkKind::kRegret:
        return "kRegret";
      case ForkKind::kUniform:
        return "kUniform";
    }
  }

  // Builds exactly kNumLastMoves entries: the (kNumLastMoves - |extra_moves|)
  // most-recent game moves ending at `move_num`, followed by `extra_moves`.
  absl::InlinedVector<game::Move, constants::kNumLastMoves> BuildLastMoves(
      const game::Game& game, int move_num,
      std::initializer_list<game::Move> extra_moves) {
    const int n_extra = (int)extra_moves.size();
    absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
    last_moves.reserve(constants::kNumLastMoves);
    for (int offset = constants::kNumLastMoves - n_extra; offset > 0;
         --offset) {
      last_moves.emplace_back(game.move(move_num - offset));
    }
    for (const auto& move : extra_moves) {
      last_moves.emplace_back(move);
    }
    return last_moves;
  }

  const Params params_;
  GoExploitReuseBuffer* reuse_buffer_;
  mcts::GumbelEvaluator* evaluator_;
  ForkKind kind_;
  bool did_fork_;
  bool started_from_forced_search_;
  int fork_mv_num_;

  // Per-move accumulation for kRegret. Fields per entry:
  //   (color, board_before_move, move, nn_value, mcts_value, is_eligible)
  using RegretPositionEntry =
      std::tuple<game::Color, game::Board, game::Loc, float, float, bool>;
  std::vector<RegretPositionEntry> positions_for_regret_;
};

class ForkManager::Params::Builder {
 public:
  Builder() = default;

  Builder& set_early_fork_prob(float v) {
    p_.early_fork_prob = v;
    return *this;
  }
  Builder& set_late_fork_prob(float v) {
    p_.late_fork_prob = v;
    return *this;
  }
  Builder& set_sample_policy_t1_prob(float v) {
    p_.sample_policy_t1_prob = v;
    return *this;
  }
  Builder& set_sample_policy_t2_prob(float v) {
    p_.sample_policy_t2_prob = v;
    return *this;
  }
  Builder& set_sample_random_prob(float v) {
    p_.sample_random_prob = v;
    return *this;
  }
  Builder& set_regret_prob(float v) {
    p_.regret_prob = v;
    return *this;
  }
  Builder& set_uniform_prob(float v) {
    p_.uniform_prob = v;
    return *this;
  }
  Builder& set_force_full_search_prob(float v) {
    p_.force_full_search_prob = v;
    return *this;
  }
  Builder& set_double_sample_prob(float v) {
    p_.double_sample_prob = v;
    return *this;
  }

  Params build() const { return p_; }

 private:
  Params p_;
};

}  // namespace selfplay
