#include "cc/eval/eval.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <sstream>

#include "absl/strings/str_format.h"
#include "cc/constants/constants.h"
#include "cc/core/file_log_sink.h"
#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search.h"

#define LOG_TO_SINK(severity, sink) LOG(severity).ToSinkOnly(&sink)

namespace {
using namespace ::core;
using namespace ::game;
using namespace ::mcts;
using namespace ::nn;

// Threshold under which to immediately resign.
static constexpr float kResignThreshold = -.96f;

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

std::string ToString(const Winner& winner) {
  return winner == Winner::kCur ? "CUR" : "CAND";
}

std::string LocToString(const game::Loc& move) {
  if (move == game::kPassLoc) return "pass";
  if (move == game::kNoopLoc) return "noop";
  return std::string(1, "ABCDEFGHIJKLMNOPQRS"[move.j]) + std::to_string(move.i);
}

// Collect top-k moves by visit count from a tree node.
struct MoveInfo {
  game::Loc move;
  int n;
  float p;
  float p_opt;
  float q;
  float lcb;
  float stddev;
};

game::Loc ActionToLoc(int a) {
  if (a == BOARD_LEN * BOARD_LEN) return game::kPassLoc;
  return game::Loc{static_cast<int8_t>(a / BOARD_LEN),
                   static_cast<int8_t>(a % BOARD_LEN)};
}

std::vector<MoveInfo> TopMoves(const TreeNode* node, int k) {
  if (node == nullptr) {
    return {};
  }

  std::vector<MoveInfo> moves;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (node->child_visits[a] == 0) continue;
    float n_a = NAction(node, a);
    moves.push_back(MoveInfo{
        .move = ActionToLoc(a),
        .n = node->child_visits[a],
        .p = node->move_probs[a],
        .p_opt = node->opt_probs[a],
        .q = Q(node, a),
        .lcb = Lcb(node, a),
        .stddev = n_a >= 2 ? std::sqrt(QVar(node, a)) : 0.0f,
    });
  }
  std::sort(moves.begin(), moves.end(),
            [](const MoveInfo& a, const MoveInfo& b) { return a.n > b.n; });
  if (static_cast<int>(moves.size()) > k) moves.resize(k);
  return moves;
}

mcts::ScoreUtilityParams MakeScoreUtilityParams(
    const eval::PlayerSearchConfig& cfg) {
  mcts::ScoreUtilityMode mode = cfg.score_utility_mode == "integral"
                                    ? mcts::ScoreUtilityMode::kIntegral
                                    : mcts::ScoreUtilityMode::kDirect;
  return mcts::ScoreUtilityParams{.score_weight = cfg.score_weight,
                                  .mode = mode};
}

// Converts a PlayerSearchConfig into Search::Params for parallel search.
// The Gumbel path in PlayEvalGame does not use this; it is here so that
// integrating parallel search is a mechanical substitution.
mcts::Search::Params MakeSearchParams(const eval::PlayerSearchConfig& cfg) {
  // Root selection policy: puct_root_policy takes priority over use_lcb.
  mcts::PuctRootSelectionPolicy root_policy;
  if (cfg.puct_root_policy == "visit_count") {
    root_policy = mcts::PuctRootSelectionPolicy::kVisitCount;
  } else if (cfg.puct_root_policy == "visit_count_sample") {
    root_policy = mcts::PuctRootSelectionPolicy::kVisitCountSample;
  } else if (cfg.puct_root_policy == "lcb" || cfg.puct_root_policy.empty()) {
    // Empty = fall back to legacy use_lcb bool.
    root_policy = (cfg.puct_root_policy == "lcb" || cfg.use_lcb)
                      ? mcts::PuctRootSelectionPolicy::kLcb
                      : mcts::PuctRootSelectionPolicy::kVisitCount;
  } else {
    root_policy = mcts::PuctRootSelectionPolicy::kLcb;  // safe default
  }

  // Q function.
  mcts::QFnKind q_fn_kind;
  if (cfg.q_fn == "identity") {
    q_fn_kind = mcts::QFnKind::kIdentity;
  } else if (cfg.q_fn == "virtual_loss_soft") {
    q_fn_kind = mcts::QFnKind::kVirtualLossSoft;
  } else {
    q_fn_kind = mcts::QFnKind::kVirtualLoss;
  }

  // N function.
  mcts::NFnKind n_fn_kind = (cfg.n_fn == "identity")
                                ? mcts::NFnKind::kIdentity
                                : mcts::NFnKind::kVirtualVisit;

  // Collision policy.
  mcts::CollisionPolicyKind collision_policy;
  if (cfg.collision_policy == "retry") {
    collision_policy = mcts::CollisionPolicyKind::kRetry;
  } else if (cfg.collision_policy == "smart_retry") {
    collision_policy = mcts::CollisionPolicyKind::kSmartRetry;
  } else {
    collision_policy = mcts::CollisionPolicyKind::kAbort;
  }

  // Collision detector.
  mcts::CollisionDetectorKind collision_detector;
  if (cfg.collision_detector == "n_in_flight") {
    collision_detector = mcts::CollisionDetectorKind::kNInFlight;
  } else if (cfg.collision_detector == "level_saturation") {
    collision_detector = mcts::CollisionDetectorKind::kLevelSaturation;
  } else if (cfg.collision_detector == "product") {
    collision_detector = mcts::CollisionDetectorKind::kProduct;
  } else {
    collision_detector = mcts::CollisionDetectorKind::kNoOp;
  }

  mcts::Search::Mode mode = (cfg.search_mode == "batch")
                                ? mcts::Search::Mode::kBatch
                                : mcts::Search::Mode::kConcurrent;

  return mcts::Search::Params{
      .num_threads = cfg.num_threads_per_game,
      .total_visit_budget = cfg.time_ms > 0 ? 1 << 20 : cfg.n,
      .total_visit_time_ms = cfg.time_ms,
      .puct_params =
          mcts::PuctParams::Builder()
              .set_kind(root_policy)
              .set_c_puct(cfg.c_puct)
              .set_c_puct_visit_scaling(cfg.c_puct_visit_scaling)
              .set_c_puct_v_2(cfg.c_puct_v_2)
              .set_use_puct_v(cfg.use_puct_v)
              .set_enable_var_scaling(cfg.var_scale_cpuct)
              .set_var_scale_prior_visits(cfg.var_scale_prior_visits)
              .set_tau(cfg.tau)
              .set_enable_m3_bonus(cfg.enable_m3_bonus)
              .set_m3_prior_visits(cfg.m3_prior_visits)
              .set_p_opt_weight(cfg.p_opt_weight)
              .build(),
      .q_fn_kind = q_fn_kind,
      .n_fn_kind = n_fn_kind,
      .descent_policy_kind = (cfg.descent_policy == "bu_uct")
                                 ? mcts::DescentPolicyKind::kBuUct
                                 : mcts::DescentPolicyKind::kDeterministic,
      .collision_policy_kind = collision_policy,
      .collision_detector_kind = collision_detector,
      .vl_delta = cfg.vl_delta,
      .max_collision_retries = cfg.max_collision_retries,
      .max_o_ratio = cfg.max_o_ratio,
      .mode = mode,
      .score_util_params = MakeScoreUtilityParams(cfg),
  };
}

}  // namespace

// Returns true when a player's config calls for the parallel mcts::Search path
// (either >1 thread per game, or time-control which Gumbel does not support).
static bool UsesParallelSearch(const eval::PlayerSearchConfig& cfg) {
  return cfg.num_threads_per_game > 1 || cfg.time_ms > 0;
}

void PlayEvalGame(size_t seed, int game_id, int total_num_workers,
                  NNInterface* cur_nn, NNInterface* cand_nn,
                  std::string logfile, std::promise<EvalResult> result,
                  recorder::GameRecorder* recorder, EvalConfig config) {
  FileSink sink(logfile.c_str());
  Probability probability(seed);
  auto search_dur_ema = 0;
  bool cur_is_black = game_id % 2 == 0;
  NNInterface* black_nn = cur_is_black ? cur_nn : cand_nn;
  NNInterface* white_nn = cur_is_black ? cand_nn : cur_nn;

  // Per-color config references. black_cfg/white_cfg are stable for the game;
  // active_cfg points to whichever color is to move each ply.
  const eval::PlayerSearchConfig& black_cfg =
      cur_is_black ? config.cur : config.cand;
  const eval::PlayerSearchConfig& white_cfg =
      cur_is_black ? config.cand : config.cur;
  const eval::PlayerSearchConfig& cur_cfg = config.cur;
  const eval::PlayerSearchConfig& cand_cfg = config.cand;

  // Setup eval game.
  Game game;
  std::unique_ptr<NodeTable> node_table_b;
  if (black_cfg.use_mcgs) {
    node_table_b = std::make_unique<McgsNodeTable>();
  } else {
    node_table_b = std::make_unique<MctsNodeTable>();
  }
  std::unique_ptr<NodeTable> node_table_w;
  if (white_cfg.use_mcgs) {
    node_table_w = std::make_unique<McgsNodeTable>();
  } else {
    node_table_w = std::make_unique<MctsNodeTable>();
  }
  // Both trees start at the empty board with BLACK to move
  TreeNode* btree = node_table_b->GetOrCreate(game.board().hash(), BLACK,
                                              /*is_terminal=*/false);
  TreeNode* wtree = node_table_w->GetOrCreate(game.board().hash(), BLACK,
                                              /*is_terminal=*/false);
  Color player_resigned = EMPTY;

  // Track root history for tree logging
  std::vector<TreeNode*> root_history;

  // Determine search path per color.
  const bool black_uses_search = UsesParallelSearch(black_cfg);
  const bool white_uses_search = UsesParallelSearch(white_cfg);

  // task_offset for parallel search: each game owns a contiguous slice of
  // thread IDs [game_id * N, (game_id+1) * N).  For Gumbel (1 thread/game)
  // the NNInterface thread ID equals game_id, encoded via the legacy
  // GumbelEvaluator(nn_interface, game_id) constructor.
  std::optional<BiasCache> bias_cache_b, bias_cache_w;
  if (black_cfg.use_bias_cache)
    bias_cache_b.emplace(black_cfg.bias_cache_alpha,
                         black_cfg.bias_cache_lambda);
  if (white_cfg.use_bias_cache)
    bias_cache_w.emplace(white_cfg.bias_cache_alpha,
                         white_cfg.bias_cache_lambda);

  std::optional<GumbelEvaluator> gumbel_b, gumbel_w;
  std::optional<Search> search_b, search_w;
  if (black_uses_search) {
    search_b.emplace(
        black_nn->MakeSlot(game_id * black_cfg.num_threads_per_game),
        bias_cache_b ? &*bias_cache_b : nullptr);
  } else {
    gumbel_b.emplace(black_nn, game_id, MakeScoreUtilityParams(black_cfg),
                     bias_cache_b ? &*bias_cache_b : nullptr);
  }
  if (white_uses_search) {
    search_w.emplace(
        white_nn->MakeSlot(game_id * white_cfg.num_threads_per_game),
        bias_cache_w ? &*bias_cache_w : nullptr);
  } else {
    gumbel_w.emplace(white_nn, game_id, MakeScoreUtilityParams(white_cfg),
                     bias_cache_w ? &*bias_cache_w : nullptr);
  }

  int cur_total_visits = 0;
  int cand_total_visits = 0;
  float cur_total_time = 0;
  float cand_total_time = 0;
  int cur_turns = 0;
  int cand_turns = 0;

  auto color_to_move = BLACK;
  while (!game.IsGameOver()) {
    TreeNode*& player_tree = color_to_move == BLACK ? btree : wtree;
    TreeNode*& opp_tree = color_to_move == BLACK ? wtree : btree;
    TreeNode* cur_tree = cur_is_black ? btree : wtree;
    TreeNode* cand_tree = cur_is_black ? wtree : btree;
    NodeTable* player_table =
        color_to_move == BLACK ? node_table_b.get() : node_table_w.get();
    NodeTable* opp_table =
        color_to_move == BLACK ? node_table_w.get() : node_table_b.get();
    const bool active_uses_search =
        color_to_move == BLACK ? black_uses_search : white_uses_search;
    const eval::PlayerSearchConfig& active_cfg =
        color_to_move == BLACK ? black_cfg : white_cfg;
    std::optional<BiasCache>& active_bias_cache =
        color_to_move == BLACK ? bias_cache_b : bias_cache_w;

    // NN statistics.
    float cur_q_nn = cur_tree->init_util_est;
    float cur_qz_nn = cur_tree->init_outcome_est;
    float cur_score_est_nn = cur_tree->init_score_est;
    float cand_q_nn = cand_tree->init_util_est;
    float cand_qz_nn = cand_tree->init_outcome_est;
    float cand_score_est_nn = cand_tree->init_score_est;

    // Pre-search statistics.
    float cur_n_pre = N(cur_tree);
    float cur_q_pre = V(cur_tree);
    float cur_q_outcome_pre = VOutcome(cur_tree);
    float cur_score_pre = Score(cur_tree);
    float cand_n_pre = N(cand_tree);
    float cand_q_pre = V(cand_tree);
    float cand_q_outcome_pre = VOutcome(cand_tree);
    float cand_score_pre = Score(cand_tree);
    float cur_var_pre = VVar(cur_tree);
    float cand_var_pre = VVar(cand_tree);

    // Search: dispatch to mcts::Search (parallel/time-control) or Gumbel.
    auto begin = std::chrono::high_resolution_clock::now();
    Loc move;
    int num_aborted = 0, num_collisions = 0;
    if (active_uses_search) {
      Search& s = color_to_move == BLACK ? *search_b : *search_w;
      Search::Result res = s.Run(probability, game, player_table, player_tree,
                                 color_to_move, MakeSearchParams(active_cfg));
      move = res.move;
      num_aborted = res.num_aborted;
      num_collisions = res.num_collisions;
    } else {
      GumbelEvaluator& gumbel = color_to_move == BLACK ? *gumbel_b : *gumbel_w;
      GumbelResult gumbel_res =
          active_cfg.use_puct
              ? gumbel.SearchRootPuct(
                    probability, game, player_table, player_tree, color_to_move,
                    active_cfg.n,
                    PuctParams::Builder()
                        .set_kind(active_cfg.use_lcb
                                      ? PuctRootSelectionPolicy::kLcb
                                      : PuctRootSelectionPolicy::kVisitCount)
                        .set_c_puct(active_cfg.c_puct)
                        .set_c_puct_v_2(active_cfg.c_puct_v_2)
                        .set_use_puct_v(active_cfg.use_puct_v)
                        .set_enable_var_scaling(active_cfg.var_scale_cpuct)
                        .set_var_scale_prior_visits(
                            active_cfg.var_scale_prior_visits)
                        .set_enable_m3_bonus(active_cfg.enable_m3_bonus)
                        .set_m3_prior_visits(active_cfg.m3_prior_visits)
                        .set_p_opt_weight(active_cfg.p_opt_weight)
                        .build())
              : gumbel.SearchRoot(
                    probability, game, player_table, player_tree, color_to_move,
                    mcts::GumbelSearchParams{active_cfg.n, active_cfg.k,
                                             active_cfg.noise_scaling});
      move = gumbel_res.mcts_move;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto search_dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count();
    search_dur_ema = search_dur_ema == 0
                         ? search_dur
                         : (search_dur_ema * 0.9 + search_dur * 0.1);
    if (VOutcome(player_tree) < kResignThreshold) {
      LOG_TO_SINK(INFO, sink) << "Player " << ToString(color_to_move)
                              << " Resigned. Qz: " << VOutcome(player_tree);
      player_resigned = color_to_move;
      break;
    }

    // Post-search statistics.
    float cur_n_post = N(cur_tree);
    float cur_q_post = V(cur_tree);
    float cur_q_nn_adj =
        cur_q_nn -
        (active_bias_cache ? active_bias_cache->Fetch(cur_tree) : 0.0f);
    float cur_q_outcome_post = VOutcome(cur_tree);
    float cur_score_post = Score(cur_tree);
    float cand_n_post = N(cand_tree);
    float cand_q_post = V(cand_tree);
    float cand_q_nn_adj =
        cand_q_nn -
        (active_bias_cache ? active_bias_cache->Fetch(cand_tree) : 0.0f);
    float cand_q_outcome_post = VOutcome(cand_tree);
    float cand_score_post = Score(cand_tree);
    float cur_var_post = VVar(cur_tree);
    float cand_var_post = VVar(cand_tree);
    const int num_visits = cur_n_post > cur_n_pre ? cur_n_post - cur_n_pre
                                                  : cand_n_post - cand_n_pre;
    // Capture top moves BEFORE PlayMove/Reap invalidates the tree nodes.
    auto cur_top_moves = TopMoves(cur_tree, 8);
    auto cand_top_moves = TopMoves(cand_tree, 8);

    const bool active_is_cur = (color_to_move == BLACK) == cur_is_black;
    if (active_is_cur) {
      cur_total_visits += static_cast<int>(cur_n_post - cur_n_pre);
      cur_total_time += search_dur;
      ++cur_turns;
    } else {
      cand_total_visits += static_cast<int>(cand_n_post - cand_n_pre);
      cand_total_time += search_dur;
      ++cand_turns;
    }

    // Commit move changes.
    float move_q = Q(player_tree, move);
    game.PlayMove(move, color_to_move);
    color_to_move = OppositeColor(color_to_move);

    // Advance to next roots.
    TreeNode* next_player = player_tree->children[move];
    TreeNode* next_opp = opp_tree->children[move];
    if (!next_player) {
      // After the move, it's the opponent's turn (color_to_move was already
      // flipped)
      next_player = player_table->GetOrCreate(game.board().hash(),
                                              color_to_move, game.IsGameOver());
    }
    if (!next_opp) {
      // After the move, it's the opponent's turn (color_to_move was already
      // flipped)
      next_opp = opp_table->GetOrCreate(game.board().hash(), color_to_move,
                                        game.IsGameOver());
    }

    // Reap tree.
    int num_nodes_reaped = 0, reap_time_us = 0;
    auto reap_begin = std::chrono::steady_clock::now();
    num_nodes_reaped += player_table->Reap(next_player);
    num_nodes_reaped += opp_table->Reap(next_opp);
    auto reap_end = std::chrono::steady_clock::now();
    reap_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       reap_end - reap_begin)
                       .count();

    // Then reap bias cache.
    uint32_t num_bias_cache_entries_pruned = 0;
    if (active_bias_cache.has_value()) {
      num_bias_cache_entries_pruned = active_bias_cache->PruneUnused();
    }

    player_tree = next_player;
    opp_tree = next_opp;

    // Log.
    [&]() {
      std::string rc = ToString(OppositeColor(color_to_move));
      std::stringstream s;
      s << "\n----- Move " << game.num_moves() << " -----\n";
      s << "N=" << active_cfg.n << "  K=" << active_cfg.k
        << "  PUCT=" << active_cfg.use_puct << "  LCB=" << active_cfg.use_lcb
        << "  cPUCT=" << active_cfg.c_puct
        << "  VarScale=" << active_cfg.var_scale_cpuct
        << "  MCGS=" << active_cfg.use_mcgs
        << "  PUCT-V=" << active_cfg.use_puct_v
        << "  cPUCT_2=" << active_cfg.c_puct_v_2 << "\n";
      s << "Move=" << LocToString(move) << "  q=" << move_q
        << "  Visits=" << num_visits << "  Aborts=" << num_aborted
        << "  Collisions=" << num_collisions << "\n";
      s << "Avg Visits/Turn — Cur="
        << absl::StrFormat(
               "%.0f", cur_turns > 0
                           ? static_cast<float>(cur_total_visits) / cur_turns
                           : 0.0f)
        << " (" << cur_turns << "t)  Cand="
        << absl::StrFormat(
               "%.0f", cand_turns > 0
                           ? static_cast<float>(cand_total_visits) / cand_turns
                           : 0.0f)
        << " (" << cand_turns << "t)\n";
      s << "Avg ms/Turn — Cur="
        << absl::StrFormat("%.0f",
                           cur_turns > 0
                               ? static_cast<float>(cur_total_time) / cur_turns
                               : 0.0f)
        << " (" << cur_turns << "t)  Cand="
        << absl::StrFormat(
               "%.0f", cand_turns > 0
                           ? static_cast<float>(cand_total_time) / cand_turns
                           : 0.0f)
        << " (" << cand_turns << "t)\n";
      s << "Last 5: " << game.move(game.num_moves() - 5) << "  "
        << game.move(game.num_moves() - 4) << "  "
        << game.move(game.num_moves() - 3) << "  "
        << game.move(game.num_moves() - 2) << "  "
        << game.move(game.num_moves() - 1) << "\n";
      s << "Cur=" << (cur_is_black ? "B" : "W")
        << "  Cand=" << (cur_is_black ? "W" : "B")
        << "  ToMove=" << ToString(color_to_move) << " ("
        << (color_to_move == BLACK ? (cur_is_black ? "CUR" : "CAND")
                                   : (cur_is_black ? "CAND" : "CUR"))
        << ")\n";

      // Cur tree stats: NN, pre-search, post-search on one line each.
      s << "(" << rc << ") Cur NN: Q=" << cur_q_nn << "  Qadj=" << cur_q_nn_adj
        << "  Qz=" << cur_qz_nn << "  Score=" << cur_score_est_nn << "\n";
      s << "(" << rc << ") Cur Pre:  N=" << cur_n_pre << "  Q=" << cur_q_pre
        << "  Qz=" << cur_q_outcome_pre << "  Stddev=" << std::sqrt(cur_var_pre)
        << "  Score=" << cur_score_pre << "\n";
      s << "(" << rc << ") Cur Post: N=" << cur_n_post << "  Q=" << cur_q_post
        << "  Qz=" << cur_q_outcome_post
        << "  Stddev=" << std::sqrt(cur_var_post)
        << "  Score=" << cur_score_post << "\n";

      // Cand tree stats.
      s << "(" << rc << ") Cand NN: Q=" << cand_q_nn
        << "  Qadj=" << cand_q_nn_adj << "  Qz=" << cand_qz_nn
        << "  Score=" << cand_score_est_nn << "\n";
      s << "(" << rc << ") Cand Pre:  N=" << cand_n_pre << "  Q=" << cand_q_pre
        << "  Qz=" << cand_q_outcome_pre
        << "  Stddev=" << std::sqrt(cand_var_pre)
        << "  Score=" << cand_score_pre << "\n";
      s << "(" << rc << ") Cand Post: N=" << cand_n_post
        << "  Q=" << cand_q_post << "  Qz=" << cand_q_outcome_post
        << "  Stddev=" << std::sqrt(cand_var_post)
        << "  Score=" << cand_score_post << "\n";

      // Top moves (captured before Reap).
      if (!cur_top_moves.empty()) {
        s << "Cur Top Moves:\n";
        for (const auto& m : cur_top_moves) {
          s << "  " << absl::StrFormat("%-4s", LocToString(m.move))
            << "  n=" << absl::StrFormat("%d", m.n)
            << "  q=" << absl::StrFormat("%.3f", m.q)
            << "  lcb=" << absl::StrFormat("%.3f", m.lcb)
            << "  p=" << absl::StrFormat("%.3f", m.p)
            << "  p'=" << absl::StrFormat("%.3f", m.p_opt)
            << "  std=" << absl::StrFormat("%.3f", m.stddev) << "\n";
        }
      }
      if (!cand_top_moves.empty()) {
        s << "Cand Top Moves:\n";
        for (const auto& m : cand_top_moves) {
          s << "  " << absl::StrFormat("%-4s", LocToString(m.move))
            << "  n=" << absl::StrFormat("%d", m.n)
            << "  q=" << absl::StrFormat("%.3f", m.q)
            << "  lcb=" << absl::StrFormat("%.3f", m.lcb)
            << "  p=" << absl::StrFormat("%.3f", m.p)
            << "  p'=" << absl::StrFormat("%.3f", m.p_opt)
            << "  std=" << absl::StrFormat("%.3f", m.stddev) << "\n";
        }
      }

      s << game.board() << "\n";
      s << "Reaped=" << num_nodes_reaped << "  ReapTime=" << reap_time_us
        << "us  BiasCachePruned=" << num_bias_cache_entries_pruned << "\n";
      s << "Search " << search_dur << "ms  EMA=" << search_dur_ema << "ms\n";

      LOG_TO_SINK(INFO, sink) << s.str();
    }();
  }

  // Unregister from NN interfaces. Parallel search tasks decrement the shared
  // task counter; Gumbel threads decrement the registered-thread count.
  if (UsesParallelSearch(config.cur)) {
    cur_nn->UnregisterSearchTask();
  } else {
    cur_nn->UnregisterThread(game_id);
  }
  if (UsesParallelSearch(config.cand)) {
    cand_nn->UnregisterSearchTask();
  } else {
    cand_nn->UnregisterThread(game_id);
  }
  game.WriteResult();

  if (player_resigned != EMPTY) {
    game.SetWinner(OppositeColor(player_resigned));
    game.SetDidResign(true);
  }

  auto game_result = game.result();
  Winner winner =
      cur_is_black
          ? (game_result.winner == BLACK ? Winner::kCur : Winner::kCand)
          : (game_result.winner == WHITE ? Winner::kCur : Winner::kCand);
  float score_diff = game_result.winner == BLACK
                         ? game_result.bscore - game_result.wscore
                         : game_result.wscore - game_result.bscore;
  LOG_TO_SINK(INFO, sink) << "Winner: " << ToString(winner) << ". Cand is "
                          << (cur_is_black ? "W" : "B") << ", Result: "
                          << (game_result.winner == BLACK ? "B" : "W") << "+"
                          << (player_resigned != EMPTY
                                  ? "R"
                                  : absl::StrFormat("%.1f", score_diff));
  LOG(INFO) << "Winner: " << ToString(winner) << ". Cand is "
            << (cur_is_black ? "W" : "B")
            << ", Result: " << (game_result.winner == BLACK ? "B" : "W") << "+"
            << (player_resigned != EMPTY ? "R"
                                         : absl::StrFormat("%.1f", score_diff))
            << ", Black Score: " << game_result.bscore
            << ", White Score: " << game_result.wscore << " (" << game_id
            << ")";

  const std::string& b_name = black_cfg.name;
  const std::string& w_name = white_cfg.name;
  recorder->RecordEvalGame(game_id, game, b_name, w_name);
  result.set_value(EvalResult{winner, game.num_moves()});
}
