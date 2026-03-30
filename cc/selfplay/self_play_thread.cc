#include "cc/selfplay/self_play_thread.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <tuple>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/file_log_sink.h"
#include "cc/core/heap.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/tree.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/book.h"
#include "cc/selfplay/reuse_buffer.h"

#define LOG_TO_SINK(severity, sink) LOG(severity).ToSinkOnly(&sink)

namespace selfplay {
namespace {
using namespace ::game;
using namespace ::core;
using namespace ::mcts;
using namespace ::nn;
using namespace ::recorder;

// Whether the current thread should log.
static constexpr int kShouldLogShard = 8;

// Probability to add any state to the seen buffer.
static constexpr float kAddSeenStateProb = 0.04f;

// Max Number of beginning moves to sample directly.
static constexpr int kMaxNumRawPolicyMoves = 30;

// Probability of exploration.
static constexpr float kOpeningExploreProb = 1.0f;

// Probability of playing from opening book.
static constexpr float kPlayFromBookProb = 0.0f;

// Probability of handicap game.
static constexpr float kHandicapGame = 0.05f;

// Thresholds at which to compute pass-alive regions.
static constexpr int kComputePAMoveNums[] = {200, 250, 300, 350, 400};

// Probability that game is selected for full tree logging (currently broken).
static constexpr float kLogFullTreeProb = 0.0f;  // .002;

// Probability that we pick a move for training.
static constexpr float kMoveSelectedForTrainingProb = 0.25f;

// Base probability that we over-search a node.
static constexpr float kOverSearchNodeProb = 0.15f;

// Threshold beneath which we are down bad (should resign).
static constexpr float kDownBadThreshold = -0.90f;

// Number of moves at down bad threshold before decreasing visit count.
static constexpr float kNumDownBadMovesThreshold = 5;

// Whether the thread should continue running.
static std::atomic<bool> running = true;

void AddNewInitState(ReuseBuffer* buffer, const Game& game, const Board& board,
                     Color color_to_move, int abs_move_num,
                     float regret = 0.0f) {
  absl::InlinedVector<Move, constants::kMaxGameLen> last_moves;
  for (int off = constants::kNumLastMoves; off > 0; --off) {
    // game.move() handles the kMoveOffset, so abs_move_num - off can be as
    // low as -kMoveOffset (accessing the initial last_moves noops).
    last_moves.emplace_back(game.move(abs_move_num - off));
  }
  CHECK(last_moves.size() == constants::kNumLastMoves);
  buffer->Add(InitState{board, last_moves, color_to_move, abs_move_num},
              regret);
}

// (color, board, move, nn_value, mcts_value, is_eligible)
using PositionsForRegret =
    std::vector<std::tuple<Color, Board, game::Loc, float, float, bool>>;

struct RegretCmp {
  bool operator()(
      const std::tuple<float, int, Color, Board, Loc, float, float>& e0,
      const std::tuple<float, int, Color, Board, Loc, float, float>& e1) const {
    return std::get<0>(e0) < std::get<0>(e1);
  }
};

// Computes per-position regret via forward EMA over Q values, then adds the
// top `max_states` highest-regret positions to the reuse buffer.
void AddRegretGuidedStates(Probability& probability, ReuseBuffer* reuse_buffer,
                           const Game& game,
                           const PositionsForRegret& positions_for_regret,
                           int max_states = 1) {
  // EMA decay per step. Effective weight at step k = kRegretEmaDecay^k;
  // horizon where weight drops to ~5% is log(0.05)/log(kRegretEmaDecay) steps.
  static constexpr float kRegretEmaDecay = 0.94f;  // ~50 step horizon
  static constexpr int kRegretHorizon = 50;
  if (max_states <= 0) {
    return;
  }

  CHECK(positions_for_regret.size() == (size_t)game.num_moves())
      << "positions_for_regret size " << positions_for_regret.size()
      << " != game.num_moves() " << game.num_moves();

  core::Heap<std::tuple<float, int, Color, Board, Loc, float, float>, RegretCmp>
      regret_buffer(RegretCmp{});
  for (int mv_num = 0; mv_num < (int)positions_for_regret.size(); ++mv_num) {
    const auto& [color, board, move, nn_v, v, is_eligible] =
        positions_for_regret[mv_num];
    if (!is_eligible) {
      continue;
    }

    const float z = game.result().winner == color ? 1.5f : -1.5f;

    // Weighted average of future Q values from color's perspective.
    float future_v_ema = 0.0f;
    float weight = 1.0f;
    float weight_sum = 0.0f;
    for (int k = 1;
         k < kRegretHorizon && mv_num + k < (int)positions_for_regret.size();
         ++k) {
      const auto& [ck, bk, mk, nn_vk, vk, ek] =
          positions_for_regret[mv_num + k];
      weight *= kRegretEmaDecay;
      if (!ek) continue;
      const float vk_color = (ck == color) ? vk : -vk;
      future_v_ema += weight * vk_color;
      weight_sum += weight;
    }
    if (weight_sum > 0.0f) future_v_ema /= weight_sum;

    // All Q values including self.
    const float v_ema =
        (v + future_v_ema * kRegretEmaDecay) / (1 + kRegretEmaDecay);
    // Interpolate z. Use constant factor b/c z changes the regret scale.
    const float future_v_ema_with_z = future_v_ema * .8 + z * .2;

    // How different is the game than the NN expected?
    const float nn_v_miseval_score = std::abs(nn_v - v_ema);
    // How much did the winrate drift after this move?
    const float wr_drift_score = std::abs(v - future_v_ema);
    // How different is short-term value from the game result?
    // Only count this term if v_ema is in the opposite direction.
    const float v_miseval_score =
        std::abs(std::max(v_ema - z - std::abs(z), 0.0f));
    const float nn_v_miseval_squared = nn_v_miseval_score * nn_v_miseval_score;
    const float wr_drift_squared = wr_drift_score * wr_drift_score;
    const float v_miseval_squared = v_miseval_score * v_miseval_score;
    const float regret =
        nn_v_miseval_squared + wr_drift_squared + v_miseval_squared;

    // Skip positions that are too won/lost: same attenuation as GoExploit.
    // Linearly attenuate from |v|=0.5 to 0 at |v|=0.9.
    const float winrate_correction = [](const float v) {
      static constexpr float kMaxV = 0.9f;
      static constexpr float kAnnealStart = 0.5f;
      if (std::abs(v) > kMaxV) return 0.0f;
      if (std::abs(v) <= kAnnealStart) return 1.0f;
      return (kMaxV - std::abs(v)) / (kMaxV - kAnnealStart);
    }(v);
    const float mv_num_correction = [](const int mv_num) {
      static constexpr int kMinAnnealMove = 100;
      static constexpr int kInterval = 100;
      const float offset = std::clamp(mv_num - kMinAnnealMove, 0, kInterval);
      const float p = 1.0 - (offset / kInterval);
      return std::clamp(std::pow(p, 1.2), 0.0, 1.0);
    }(game.init_mv_num() + mv_num);
    const float regret_add_prob = mv_num_correction * winrate_correction;
    if (probability.Uniform() >= regret_add_prob) continue;

    regret_buffer.PushHeap({regret, mv_num, color, board, move, v, z});
  }

  // Add the top highest-regret positions to the reuse buffer.
  // mv_num is 0-indexed from game start (same scale as game.num_moves()),
  // so it can be passed directly to AddNewInitState as abs_move_num.
  for (int i = 0; i < (max_states + 4) && regret_buffer.Size() > 0; ++i) {
    const auto [regret, mv_num, color, board, move, v, z] =
        regret_buffer.PopHeap();
    AddNewInitState(reuse_buffer, game, board, color, mv_num, regret);
  }
}

InitState GetInitState(Probability& probability, ReuseBuffer* buffer,
                       float use_seen_state_prob, int num_games_played) {
  const float komi = std::round(7.0f + std::min(probability.Gaussian(), 3.0f)) +
                     (probability.Uniform() < 0.5f ? 0.5f : -0.5f);
  InitState s0 =
      InitState{Board(komi),
                {kNoopMove, kNoopMove, kNoopMove, kNoopMove, kNoopMove},
                BLACK,
                0,
                /*force_full_search=*/false,
                InitState::Kind::kEmpty};
  float p = probability.Uniform();
  if (p <= kPlayFromBookProb) {
    const int index = probability.Uniform() * kOpeningBook.size();
    const int num_moves =
        std::round(probability.Uniform() * kOpeningBook[index].size());
    s0.kind = InitState::Kind::kBook;
    s0.last_moves.clear();
    for (int i = 0; i < constants::kNumLastMoves - num_moves; ++i) {
      s0.last_moves.emplace_back(kNoopMove);
    }

    for (int i = 0; i < num_moves; ++i) {
      const auto loc = kOpeningBook[index][i];
      s0.board.PlayMove(loc, s0.color_to_move);
      s0.last_moves.emplace_back(Move{s0.color_to_move, loc});
      s0.color_to_move = game::OppositeColor(s0.color_to_move);
    }

    return s0;
  } else if (p <= kPlayFromBookProb + use_seen_state_prob) {
    if (num_games_played < 3 && buffer->GetType() == BufferType::kRegret) {
      // give the regret buffer some time to fill up.
      return s0;
    }
    std::optional<InitState> seen_state = buffer->Get();
    if (!seen_state) {
      return s0;
    }

    seen_state->kind = seen_state->force_full_search
                           ? InitState::Kind::kRegret
                           : InitState::Kind::kGoExploit;
    return seen_state.value();
  } else if (p <= kPlayFromBookProb + use_seen_state_prob + kHandicapGame) {
    int handicap = std::floor(probability.Uniform() * 3 + 2);
    float komi = (handicap - 2) * 14 + 20.5;  // katago ;)
    return InitState{Board(handicap, komi),
                     {kNoopMove, kNoopMove, kNoopMove, kNoopMove, kNoopMove},
                     WHITE,
                     0,
                     /*force_full_search=*/false,
                     InitState::Kind::kHandicap};
  }

  return s0;
}

std::string ToString(const InitState::Kind& kind) {
  switch (kind) {
    case InitState::Kind::kEmpty:
      return "Empty";
    case InitState::Kind::kBook:
      return "Book";
    case InitState::Kind::kHandicap:
      return "Handicap";
    case InitState::Kind::kGoExploit:
      return "GoExploit";
    case InitState::Kind::kRegret:
      return "Regret";
    default:
      return "Unknown";
  }
}

std::string ToString(const Color& color) {
  switch (color) {
    case BLACK:
      return "B";
    case WHITE:
      return "W";
    case EMPTY:
      return "E";
    default:
      return "U";
  }
}

}  // namespace

void Run(size_t seed, int thread_id, NNInterface* nn_interface,
         GameRecorder* game_recorder, ReuseBuffer* reuse_buffer,
         std::string logfile, SPConfig config) {
  FileSink sink(logfile.c_str());
  Probability probability(seed);
  int num_games_played = 0;

  // Main loop.
  while (true) {
    // Populate initial state either from seen states or s_0.
    InitState init_state =
        GetInitState(probability, reuse_buffer, config.use_seen_state_prob,
                     num_games_played);
    const bool is_regret_game = init_state.force_full_search;
    const int init_mv_num = init_state.move_num;

    // Game state.
    Game game(init_state.board, init_state.last_moves, init_mv_num);
    Color color_to_move = init_state.color_to_move;

    // Node table for MCTS.
    std::unique_ptr<NodeTable> node_table = std::make_unique<MctsNodeTable>();

    // Search Tree.
    TreeNode* root_node = node_table->GetOrCreate(
        game.board().hash(), color_to_move, /*is_terminal=*/false);

    // Completed Q-values for each timestep.
    std::vector<std::array<float, constants::kMaxMovesPerPosition>> mcts_pis;

    // Root Q values for each timestep (outcome only).
    std::vector<float> root_q_outcomes;

    // Root Score values for each timestep.
    std::vector<float> root_scores;

    // Whether the i'th move is trainable.
    std::vector<uint8_t> move_trainables;

    // The KLD of the i'th move from the prior.
    std::vector<float> klds;

    // Visit counts for each move.
    std::vector<uint32_t> visit_counts;

    // Number of consecutive moves at which we are beneath kDownBadThreshold.
    int num_consecutive_down_bad_moves = 0;

    // Whether to log full MCTS search trees.
    bool const log_mcts_trees = probability.Uniform() < kLogFullTreeProb;

    // Tree roots throughout game (if logging full trees).
    std::vector<TreeNode*> search_roots;

    // Starting positions throughout game.
    PositionsForRegret positions_for_regret;

    // Number of moves for which to sample directly from the policy.
    // Regret games skip raw policy sampling — every move is fully searched.
    // For mid-game positions, the cap decays with a half-life of 40 moves so
    // that late-game restarts do less raw-policy sampling.
    int const max_raw_policy_moves = std::round(
        kMaxNumRawPolicyMoves * std::pow(0.5f, init_state.move_num / 40.0f));
    int const num_moves_raw_policy =
        !is_regret_game && max_raw_policy_moves > 0 &&
                probability.Uniform() < kOpeningExploreProb
            ? RandRange(probability.prng(), 0, max_raw_policy_moves)
            : 0;
    LOG_TO_SINK(INFO, sink) << "\nNEW GAME  Kind=" << ToString(init_state.kind)
                            << "  StartMove=" << init_state.move_num
                            << "  Color=" << ToString(color_to_move)
                            << "  Raw Policy Moves=" << num_moves_raw_policy;

    // Begin Search.
    int search_dur_avg = 0;
    int search_dur_ema = 0;

    // Uncertainty tracking (v_err = MCTS-aggregated NN uncertainty estimate).
    float uncertainty_avg = 0, uncertainty_ema = 0;
    float uncertainty_min = std::numeric_limits<float>::max();
    float uncertainty_max = std::numeric_limits<float>::lowest();

    // |NN_util - MCTS_util| tracking.
    float diff_avg = 0, diff_ema = 0;
    float diff_min = std::numeric_limits<float>::max();
    float diff_max = std::numeric_limits<float>::lowest();

    // Pre-search variance (v_outcome_var from tree reuse) tracking.
    float var_avg = 0, var_ema = 0;
    float var_min = std::numeric_limits<float>::max();
    float var_max = std::numeric_limits<float>::lowest();

    // Over-search multiplier tracking.
    float sel_mult_avg = 0, sel_mult_ema = 0;

    // Pre-search move-level Q stats (top-8 children by visit count).
    float q_var_avg = 0, q_var_ema = 0;
    float q_var_min = std::numeric_limits<float>::max();
    float q_var_max = std::numeric_limits<float>::lowest();
    float top_gap_avg = 0, top_gap_ema = 0;
    float top_gap_min = std::numeric_limits<float>::max();
    float top_gap_max = std::numeric_limits<float>::lowest();
    std::optional<mcts::BiasCache> bias_cache_storage;
    mcts::BiasCache* bias_cache = nullptr;
    if (config.bias_cache_lambda > 0.0f) {
      bias_cache_storage.emplace(config.bias_cache_alpha,
                                 config.bias_cache_lambda);
      bias_cache = &bias_cache_storage.value();
    }
    GumbelEvaluator gumbel_evaluator(nn_interface, thread_id, bias_cache);
    while (IsRunning() && !game.IsGameOver() &&
           game.num_moves() < config.max_moves) {
      auto round_start = std::chrono::steady_clock::now();
      // Choose n, k = 1 if we have not reached `num_moves_raw_policy` number of
      // moves.
      // Choose n, k = kDownBadParams if we are down bad.
      bool const sampling_raw_policy = game.num_moves() < num_moves_raw_policy;
      bool const is_either_down_bad =
          num_consecutive_down_bad_moves >= kNumDownBadMovesThreshold;

      // How down bad our root q is, from [0, 1]. 0 is max, 1 is min (i.e.)
      // down_bad_coeff(-1) = 0, down_bad_coeff(|q| < .9) = 0.
      float const down_bad_coeff = [&root_node]() {
        if (!root_node) {
          return 1.0f;
        }

        float root_v = VOutcome(root_node);
        if (root_v > kDownBadThreshold && root_v < -kDownBadThreshold) {
          // We are not down bad.
          return 1.0f;
        }

        // max distance to min/max v.
        float max_delta = 1.0f - std::abs(kDownBadThreshold);
        // how close we are to min/max v (closer should be penalized more).
        float delta = 1.0f - std::abs(root_v);
        float down_bad_coeff = delta / max_delta;

        return down_bad_coeff;
      }();

      // NN Stats.
      float qz_nn = root_node->init_outcome_est;
      float score_nn = root_node->init_score_est;

      // Pre-search stats (from tree reuse).
      int n_pre = N(root_node);
      float q_pre = V(root_node);
      float qz_pre = VOutcome(root_node);
      float var_pre = n_pre < 3 ? 0.0f : root_node->v_outcome_var;

      // Pre-search move-level Q stats: top-8 children by visit count.
      float pre_q_var = 0.0f, pre_top_gap = 0.0f, pre_prior_gap = 0.0f;
      int pre_k = 0;
      {
        constexpr int kTopK = 8;
        std::array<int, kTopK> top_actions;
        top_actions.fill(-1);
        std::array<int, kTopK> top_visits;
        top_visits.fill(0);
        for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
          const int n = root_node->child_visits[a];
          if (n == 0 || root_node->child(a) == nullptr) continue;
          for (int i = 0; i < kTopK; ++i) {
            if (n > top_visits[i]) {
              for (int j = kTopK - 1; j > i; --j) {
                top_visits[j] = top_visits[j - 1];
                top_actions[j] = top_actions[j - 1];
              }
              top_visits[i] = n;
              top_actions[i] = a;
              break;
            }
          }
        }
        std::array<float, kTopK> qs;
        for (int i = 0; i < kTopK && top_actions[i] >= 0; ++i) {
          qs[pre_k++] = Q(root_node, top_actions[i]);
        }
        if (pre_k >= 2) {
          std::sort(qs.begin(), qs.begin() + pre_k, std::greater<float>());
          pre_top_gap = qs[0] - qs[1];
          float mean = 0;
          for (int i = 0; i < pre_k; ++i) mean += qs[i];
          mean /= pre_k;
          float var = 0;
          for (int i = 0; i < pre_k; ++i) var += (qs[i] - mean) * (qs[i] - mean);
          pre_q_var = var / pre_k;

          // Prior gap: top1 - top2 prior among visited children.
          float p1 = 0.0f, p2 = 0.0f;
          for (int i = 0; i < pre_k; ++i) {
            float p = root_node->move_probs[top_actions[i]];
            if (p >= p1) { p2 = p1; p1 = p; }
            else if (p > p2) { p2 = p; }
          }
          pre_prior_gap = p1 - p2;
        }
      }

      // Selection probability multiplier.
      // Group 1: |NN_util - MCTS_util| pre-search.
      //   No bonus below p50; scales linearly to 2.0x at p95.
      const float sel_g1_mult = [&]() {
        constexpr float kP50 = 0.060f;
        constexpr float kP95 = 0.436f;
        const float diff = std::abs(root_node->init_util_est - q_pre);
        return 1.0f + std::clamp((diff - kP50) / (kP95 - kP50), 0.0f, 1.0f);
      }();

      // Group 2 bonus: top1/2 Q gap is small and NN prior gap >= 0.2.
      //   The NN is confident in a move that MCTS finds equivalent to
      //   alternatives — more search likely to change the decision.
      //   No bonus above p30; scales linearly to 1.5x at p5.
      const float sel_g2_bonus = [&]() {
        if (pre_k < 2 || pre_prior_gap < 0.2f) return 1.0f;
        constexpr float kP5  = 0.0013f;
        constexpr float kP30 = 0.012f;
        return 1.0f + 0.5f * std::clamp((kP30 - pre_top_gap) / (kP30 - kP5),
                                        0.0f, 1.0f);
      }();

      // Group 2 penalty: top1/2 Q gap is large and NN prior gap >= 0.2 —
      //   best move dominates and NN agrees, more search won't change the
      //   decision. Voided when prior gap < 0.2 (NN can't distinguish moves,
      //   large Q gap may be noise). 1.0x below p85; scales to 0.5x at p95.
      const float sel_g2_penalty = [&]() {
        if (pre_k < 2 || pre_prior_gap < 0.2f) return 1.0f;
        constexpr float kP85 = 0.158f;
        constexpr float kP95 = 0.345f;
        return 1.0f - 0.5f * std::clamp(
                                 (pre_top_gap - kP85) / (kP95 - kP85),
                                 0.0f, 1.0f);
      }();

      const bool apply_sel_mult = config.sel_mult_base > 0.0f &&
                                  probability.Uniform() < config.sel_mult_prob;
      const float sel_mult = apply_sel_mult
                                 ? config.sel_mult_base *
                                       std::max(sel_g1_mult, sel_g2_bonus) *
                                       sel_g2_penalty
                                 : 1.0f;

      // Probability of selecting a move for training, scaled by sel_mult.
      // If we are not down bad, this is a base probability of
      // `kMoveSelectedForTrainingProb`. If we are down bad, then we anneal our
      // selection probability by down_bad_coeff^2. So if V(root) = -.95,
      // down_bad_coeff = .5, and our probability is annealed by .5 * .5 = .25.
      float const select_move_prob = [sampling_raw_policy, is_either_down_bad,
                                      down_bad_coeff, sel_mult]() {
        if (sampling_raw_policy) {
          return 0.0f;
        }
        const float base = is_either_down_bad
                               ? down_bad_coeff * down_bad_coeff *
                                     kMoveSelectedForTrainingProb
                               : kMoveSelectedForTrainingProb;
        return base * sel_mult;
      }();

      // Probability of over-searching this move, if selected for training.
      float const over_search_prob = [sampling_raw_policy, is_either_down_bad,
                                      &root_node]() {
        if (sampling_raw_policy || is_either_down_bad) {
          return 0.0f;
        }
        if (root_node == nullptr) {
          return kOverSearchNodeProb;
        }

        constexpr float kBaseStdDev = 0.15;
        const float v = VOutcome(root_node);
        const float std = root_node->v_outcome_var == 0.0
                              ? kBaseStdDev
                              : std::sqrt(root_node->v_outcome_var);
        const float cv = std::clamp((1.0f - std::abs(v)) / 0.8f, 0.0f, 1.0f);
        const float cstd = std / kBaseStdDev;
        return std::min(kOverSearchNodeProb, cv * cstd * kOverSearchNodeProb);
      }();

      // Visit count cap for trainable moves. If we are not down bad, then we
      // return the default. Otherwise, we anneal according to n =
      // (1 - down_bad_coeff) * default_n + down_bad_coeff * default_n. We
      // anneal k linearly according to where n falls on the scale [default_n,
      // training_n].
      GumbelParams const trainable_gumbel_params = [&config, is_either_down_bad,
                                                    down_bad_coeff]() {
        if (!is_either_down_bad) {
          return config.selected_params;
        }

        int n = (1.0f - down_bad_coeff) * config.default_params.n +
                down_bad_coeff * config.selected_params.n;
        int k = config.default_params.k;
        return GumbelParams{n, k};
      }();

      // Force full search on the first move of a regret-game restart.
      bool const force_first_move = is_regret_game && game.num_moves() == 0;
      bool const is_move_selected_for_training =
          force_first_move || probability.Uniform() < select_move_prob;
      bool const is_move_over_search =
          (probability.Uniform() < over_search_prob &&
           is_move_selected_for_training && false);
      bool const early_stopping_enabled = !is_move_over_search && false;
      auto const [gumbel_n, gumbel_k, noise_scaling] = [&]() {
        int gumbel_n, gumbel_k;
        float noise_scaling = 1.0f;
        if (force_first_move) {
          gumbel_n = config.selected_params.n;
          gumbel_k = config.selected_params.k;
        } else if (sampling_raw_policy) {
          gumbel_n = 1;
          gumbel_k = 1;
        } else if (is_move_selected_for_training) {
          gumbel_n = trainable_gumbel_params.n;
          gumbel_k = trainable_gumbel_params.k;
        } else {
          gumbel_n = config.default_params.n;
          gumbel_k = config.default_params.k;
          noise_scaling = 0.0f;
        }

        return std::make_tuple(gumbel_n, gumbel_k, noise_scaling);
      }();

      float const tau = [&game, num_moves_raw_policy]() {
        static constexpr float kMaxTau = 0.8f;
        static constexpr float kMinTau = 0.2f;
        static constexpr int kHalfLife = 19;
        int const num_non_sample_moves =
            game.num_moves() - num_moves_raw_policy;
        float const lambda = std::log(2) / kHalfLife;
        return std::min(
            std::max(kMaxTau * std::exp(-lambda * num_non_sample_moves),
                     kMinTau),
            kMaxTau);
      }();

      // Run and Profile Search.
      auto begin = std::chrono::steady_clock::now();
      GumbelResult gumbel_res =
          is_move_selected_for_training || sampling_raw_policy || gumbel_n < 100
              ? gumbel_evaluator.SearchRoot(
                    probability, game, node_table.get(), root_node,
                    color_to_move,
                    GumbelSearchParams::Builder()
                        .set_n(gumbel_n)
                        .set_k(gumbel_k)
                        .set_noise_scaling(noise_scaling)
                        .set_early_stopping_enabled(early_stopping_enabled)
                        .set_over_search_enabled(is_move_over_search)
                        .set_tau(tau)
                        .set_nonroot_var_scale_prior_visits(
                            config.nonroot_var_scale_prior_visits)
                        .build())
              : gumbel_evaluator.SearchRootPuct(
                    probability, game, node_table.get(), root_node,
                    color_to_move, gumbel_n,
                    PuctParams::Builder()
                        .set_kind(PuctRootSelectionPolicy::kVisitCountSample)
                        .set_c_puct(1.05f)
                        .set_c_puct_visit_scaling(0.28f)
                        .set_c_puct_v_2(0.0f)
                        .set_tau(tau)
                        .build());
      auto end = std::chrono::steady_clock::now();

      // Post Search Statstics.
      int n_post = N(root_node);
      float q_post = V(root_node);
      float root_q_outcome = VOutcome(root_node);
      float root_score = Score(root_node);

      Loc nn_move = gumbel_res.nn_move;
      Loc move = gumbel_res.mcts_move;

      int nn_move_n = NAction(root_node, nn_move);
      float nn_util_est = root_node->init_util_est;
      float nn_move_q = Q(root_node, nn_move);
      float nn_move_qz = QOutcome(root_node, nn_move);

      int move_n = NAction(root_node, move);
      float move_q = Q(root_node, move);
      float move_qz = QOutcome(root_node, move);

      // Bias cache adjustment for this root (read-only, no update).
      float qadj = bias_cache != nullptr ? bias_cache->Fetch(root_node) : 0.0f;

      // Per-move uncertainty and NN/MCTS divergence (pre-search, via tree reuse).
      float node_uncertainty = root_node->v_err;
      float nn_mcts_diff = std::abs(nn_util_est - q_pre);

      // Update tracking data structures.
      mcts_pis.push_back(gumbel_res.pi_improved);
      move_trainables.push_back(is_move_selected_for_training);
      root_q_outcomes.push_back(root_q_outcome);
      root_scores.push_back(root_score);
      klds.push_back(gumbel_res.kld);
      visit_counts.push_back(gumbel_res.visits);
      positions_for_regret.push_back({color_to_move, game.board(), move,
                                      nn_util_est, q_post,
                                      /*is_eligible=*/move_n != 0});
      if (-std::abs(root_q_outcome) < kDownBadThreshold) {
        ++num_consecutive_down_bad_moves;
      } else {
        num_consecutive_down_bad_moves = 0;
      }

      // Play move, and calculate PA regions if we hit a checkpoint.
      game.PlayMove(move, color_to_move);
      if (std::find(std::begin(kComputePAMoveNums),
                    std::end(kComputePAMoveNums),
                    game.num_moves()) != std::end(kComputePAMoveNums)) {
        game.CalculatePassAliveRegions();
      }
      color_to_move = OppositeColor(color_to_move);

      // Advance to next root.
      TreeNode* next_root = root_node->children[move];
      if (!next_root) {
        // this is possible if we draw directly from policy, or if pass is the
        // only legal move found in search.
        next_root = node_table->GetOrCreate(game.board().hash(), color_to_move,
                                            game.IsGameOver());
      }

      // Store root for tree logging if enabled.
      int num_nodes_reaped = 0, reap_time_us = 0;
      if (log_mcts_trees) {
        search_roots.emplace_back(root_node);
      } else {
        auto reap_begin = std::chrono::steady_clock::now();
        num_nodes_reaped = node_table->Reap(next_root);
        auto reap_end = std::chrono::steady_clock::now();
        reap_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           reap_end - reap_begin)
                           .count();
      }
      root_node = next_root;

      // Reap Bias Cache.
      int num_bias_cache_entries_pruned = 0;
      if (bias_cache != nullptr) {
        num_bias_cache_entries_pruned = bias_cache->PruneUnused();
      }

      // Add to seen state buffer.
      float add_init_state_prob = [](float root_q) {
        // Adds states with sharper V values with lower probability. According
        // to these rules:
        // - Any |V| > 0.9 is added with probability 0.
        // - Any V where 0.5 < |V| <= 0.9 is penalized linearly, according to
        // the formula
        //   (0.9 - |V|) / 0.4
        static constexpr float kMaxVToAdd = 0.9;
        static constexpr float kMinVToAnneal = 0.5;

        if (std::abs(root_q) > kMaxVToAdd) {
          return 0.0f;
        } else if (std::abs(root_q) <= kMinVToAnneal) {
          return kAddSeenStateProb;
        }

        float penalty =
            (kMaxVToAdd - std::abs(root_q)) / (kMaxVToAdd - kMinVToAnneal);
        float p = penalty * kAddSeenStateProb;
        return p;
      }(root_q_outcome);

      if (probability.Uniform() < add_init_state_prob &&
          (reuse_buffer->GetType() == BufferType::kGoExploit ||
           reuse_buffer->GetType() == BufferType::kComposite)) {
        AddNewInitState(reuse_buffer, game, game.board(), color_to_move,
                        game.num_moves());
      }

      auto search_dur =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count();
      search_dur_avg = (search_dur_avg * (game.num_moves() - 1) + search_dur) /
                       game.num_moves();
      search_dur_ema = search_dur_ema == 0
                           ? search_dur
                           : (0.75 * search_dur_ema + .25 * search_dur);

      uncertainty_avg =
          (uncertainty_avg * (game.num_moves() - 1) + node_uncertainty) /
          game.num_moves();
      uncertainty_min = std::min(uncertainty_min, node_uncertainty);
      uncertainty_max = std::max(uncertainty_max, node_uncertainty);
      uncertainty_ema =
          uncertainty_ema == 0
              ? node_uncertainty
              : (0.75f * uncertainty_ema + 0.25f * node_uncertainty);

      diff_avg =
          (diff_avg * (game.num_moves() - 1) + nn_mcts_diff) / game.num_moves();
      diff_min = std::min(diff_min, nn_mcts_diff);
      diff_max = std::max(diff_max, nn_mcts_diff);
      diff_ema = diff_ema == 0 ? nn_mcts_diff
                               : (0.75f * diff_ema + 0.25f * nn_mcts_diff);

      var_avg =
          (var_avg * (game.num_moves() - 1) + var_pre) / game.num_moves();
      var_min = std::min(var_min, var_pre);
      var_max = std::max(var_max, var_pre);
      var_ema = var_ema == 0 ? var_pre
                             : (0.75f * var_ema + 0.25f * var_pre);

      if (pre_k >= 2) {
        const int n = game.num_moves();
        q_var_avg = (q_var_avg * (n - 1) + pre_q_var) / n;
        q_var_min = std::min(q_var_min, pre_q_var);
        q_var_max = std::max(q_var_max, pre_q_var);
        q_var_ema = q_var_ema == 0 ? pre_q_var
                                   : (0.75f * q_var_ema + 0.25f * pre_q_var);
        top_gap_avg = (top_gap_avg * (n - 1) + pre_top_gap) / n;
        top_gap_min = std::min(top_gap_min, pre_top_gap);
        top_gap_max = std::max(top_gap_max, pre_top_gap);
        top_gap_ema = top_gap_ema == 0
                          ? pre_top_gap
                          : (0.75f * top_gap_ema + 0.25f * pre_top_gap);
      }

      sel_mult_avg =
          (sel_mult_avg * (game.num_moves() - 1) + sel_mult) /
          game.num_moves();
      sel_mult_ema =
          sel_mult_ema == 0
              ? sel_mult
              : (0.75f * sel_mult_ema + 0.25f * sel_mult);

      auto round_end = std::chrono::steady_clock::now();
      auto round_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
                           round_end - round_start)
                           .count();
      auto mv_to_string = [](const game::Loc& move) {
        return ("ABCDEFGHIJKLMNOPQRS"[move.j]) + std::to_string(move.i);
      };

      if (thread_id % kShouldLogShard == 0) {
        std::string root_color = ToString(OppositeColor(color_to_move));
        std::stringstream s;
        s << "\n----- Move " << game.num_moves() << " | Abs Move "
          << (game.num_moves() + game.init_mv_num()) << " -----\n";
        s << "N=" << gumbel_n << "  K=" << gumbel_k
          << "  Select Move Prob=" << select_move_prob
          << "  Add Init State Prob=" << add_init_state_prob
          << "  Trainable=" << is_move_selected_for_training
          << "  Early Stopping=" << early_stopping_enabled
          << "  Over Search=" << is_move_over_search
          << "  Visits=" << gumbel_res.visits << "  Tau=" << tau << "\n";
        s << "Last 5 Moves: " << game.move(game.num_moves() - 5) << "  "
          << game.move(game.num_moves() - 4) << "  "
          << game.move(game.num_moves() - 3) << "  "
          << game.move(game.num_moves() - 2) << "  "
          << game.move(game.num_moves() - 1) << "\n";
        s << "(" << root_color << ") NN Stats\n  Qz = " << qz_nn
          << "\n  Score = " << score_nn << "\n";
        s << "(" << root_color << ") Pre-Search Stats\n  N = " << n_pre
          << "\n  Q = " << q_pre << "\n  Qz = " << qz_pre << "\n";
        s << "(" << root_color << ") Post-Search Stats\n  N = " << n_post
          << "\n  Q = " << q_post << "\n  Qz = " << root_q_outcome
          << "\n  Qadj = " << qadj << "\n";
        s << "Raw NN Move=" << mv_to_string(nn_move) << "\n  n = " << nn_move_n
          << "\n  q = " << nn_move_q << "\n  qz = " << nn_move_qz << "\n";
        s << "Gumbel Move=" << mv_to_string(move) << "\n  n = " << move_n
          << "\n  q = " << move_q << "\n  qz = " << move_qz << "\n";
        if (!gumbel_res.child_stats.empty()) {
          s << "Considered Moves:\n";
          for (const ChildStats& mv_stats : gumbel_res.child_stats) {
            s << "  " << mv_to_string(mv_stats.move)
              << "  p=" << absl::StrFormat("%.3f", mv_stats.prob)
              << "  p'=" << absl::StrFormat("%.3f", mv_stats.improved_policy)
              << "  n=" << absl::StrFormat("%d", mv_stats.n)
              << "  q=" << absl::StrFormat("%.3f", mv_stats.q)
              << "  qz=" << absl::StrFormat("%.3f", mv_stats.qz)
              << "  score=" << absl::StrFormat("%.3f", mv_stats.score) << "\n";
          }
        }
        s << "(" << ToString(root_node->color_to_move)
          << ") Next Root  N=" << root_node->n << "  Q=" << root_node->v
          << "  Qz=" << root_node->v_outcome
          << "  Score=" << root_node->init_score_est << "\n";
        s << "Uncertainty=" << node_uncertainty << "  Avg=" << uncertainty_avg
          << "  Min=" << uncertainty_min << "  Max=" << uncertainty_max
          << "  EMA=" << uncertainty_ema << "\n";
        s << "|NN-MCTS|=" << nn_mcts_diff << "  Avg=" << diff_avg
          << "  Min=" << diff_min << "  Max=" << diff_max
          << "  EMA=" << diff_ema << "\n";
        s << "Var(pre)=" << var_pre << "  Avg=" << var_avg
          << "  Min=" << var_min << "  Max=" << var_max
          << "  EMA=" << var_ema << "\n";
        s << "MoveQ(k=" << pre_k << ")"
          << "  Var=" << pre_q_var << "  VarAvg=" << q_var_avg
          << "  VarMin=" << q_var_min << "  VarMax=" << q_var_max
          << "  VarEMA=" << q_var_ema << "\n";
        s << "  Top1/2Gap=" << pre_top_gap << "  PriorGap=" << pre_prior_gap
          << "  GapAvg=" << top_gap_avg << "  GapMin=" << top_gap_min
          << "  GapMax=" << top_gap_max << "  GapEMA=" << top_gap_ema << "\n";
        s << "SelMult=" << sel_mult
          << "  G1=" << sel_g1_mult
          << "  G2Bonus=" << sel_g2_bonus
          << "  G2Penalty=" << sel_g2_penalty
          << "  Avg=" << sel_mult_avg
          << "  EMA=" << sel_mult_ema << "\n";
        s << "Board:\n" << game.board() << "\n";
        s << "Nodes Reaped=" << num_nodes_reaped
          << "  Reap Time=" << reap_time_us
          << "us  Bias Cache Reaped=" << num_bias_cache_entries_pruned << "\n";
        s << "Search Took " << search_dur << "ms  Average=" << search_dur_avg
          << "ms  EMA=" << search_dur_ema << "ms  Search (incl. overhead) Took "
          << round_dur << "ms\n";

        LOG_TO_SINK(INFO, sink) << s.str();
      }
    }

    nn_interface->UnregisterThread(thread_id);
    if (!IsRunning()) break;

    game.WriteResult();
    ++num_games_played;

    LOG_TO_SINK(INFO, sink) << "Black Score=" << game.result().bscore
                            << "  White Score=" << game.result().wscore;

    // Find the positions with the most "regret" and add them to the reuse
    // buffer. The rough intuition for "regret" is the positions that led to
    // the biggest loss in winrate. It is implemented by taking an EMA of
    // value loss over a 50-move horizon.
    if (reuse_buffer->GetType() == BufferType::kRegret ||
        reuse_buffer->GetType() == BufferType::kComposite) {
      AddRegretGuidedStates(probability, reuse_buffer, game,
                            positions_for_regret,
                            is_regret_game ? probability.Uniform() < 0.5f : 1);
    }

    auto begin = std::chrono::high_resolution_clock::now();
    game_recorder->RecordGame(thread_id, init_state.board, game, mcts_pis,
                              move_trainables, root_q_outcomes, root_scores,
                              klds, search_roots, visit_counts);
    auto end = std::chrono::high_resolution_clock::now();

    LOG_TO_SINK(INFO, sink)
        << "Recording Game Took "
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
               .count()
        << "us";

    // Threads start off auto-registered, so doing this at the beginning of the
    // loop is incorrect.
    nn_interface->RegisterThread(thread_id);
  }
}

void SignalStop() { running.store(false, std::memory_order_release); }

bool IsRunning() { return running.load(std::memory_order_acquire); }

}  // namespace selfplay

#undef LOG_TO_SINK
