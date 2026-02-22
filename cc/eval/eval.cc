#include "cc/eval/eval.h"

#include <chrono>
#include <sstream>

#include "cc/constants/constants.h"
#include "cc/core/file_log_sink.h"
#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"

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

}  // namespace

void PlayEvalGame(size_t seed, int thread_id, NNInterface* cur_nn,
                  NNInterface* cand_nn, std::string logfile,
                  std::promise<EvalResult> result,
                  recorder::GameRecorder* recorder, EvalConfig config) {
  FileSink sink(logfile.c_str());
  Probability probability(seed);
  auto search_dur_ema = 0;
  bool cur_is_black = thread_id % 2 == 0;
  NNInterface* black_nn = cur_is_black ? cur_nn : cand_nn;
  NNInterface* white_nn = cur_is_black ? cand_nn : cur_nn;

  // Unpack config.
  std::string cur_name = config.cur_name;
  std::string cand_name = config.cand_name;
  int cur_n = config.cur_n;
  int cur_k = config.cur_k;
  int cand_n = config.cand_n;
  int cand_k = config.cand_k;

  // Setup eval game.
  int n_b = cur_is_black ? cur_n : cand_n;
  int k_b = cur_is_black ? cur_k : cand_k;
  int n_w = cur_is_black ? cand_n : cur_n;
  int k_w = cur_is_black ? cand_k : cur_k;
  float noise_scaling_b =
      cur_is_black ? config.cur_noise_scaling : config.cand_noise_scaling;
  float noise_scaling_w =
      cur_is_black ? config.cand_noise_scaling : config.cur_noise_scaling;
  bool use_puct_b = cur_is_black ? config.cur_use_puct : config.cand_use_puct;
  bool use_puct_w = cur_is_black ? config.cand_use_puct : config.cur_use_puct;
  bool use_lcb_b = cur_is_black ? config.cur_use_lcb : config.cand_use_lcb;
  bool use_lcb_w = cur_is_black ? config.cand_use_lcb : config.cur_use_lcb;
  float c_puct_b = cur_is_black ? config.cur_c_puct : config.cand_c_puct;
  float c_puct_w = cur_is_black ? config.cand_c_puct : config.cur_c_puct;
  float var_scale_cpuct_b =
      cur_is_black ? config.cur_var_scale_cpuct : config.cand_var_scale_cpuct;
  float var_scale_cpuct_w =
      cur_is_black ? config.cand_var_scale_cpuct : config.cur_var_scale_cpuct;
  bool use_mcgs_b = cur_is_black ? config.cur_use_mcgs : config.cand_use_mcgs;
  bool use_mcgs_w = cur_is_black ? config.cand_use_mcgs : config.cur_use_mcgs;
  bool use_puct_v_b =
      cur_is_black ? config.cur_use_puct_v : config.cand_use_puct_v;
  bool use_puct_v_w =
      cur_is_black ? config.cand_use_puct_v : config.cur_use_puct_v;
  float c_puct_v_2_b =
      cur_is_black ? config.cur_c_puct_v_2 : config.cand_c_puct_v_2;
  float c_puct_v_2_w =
      cur_is_black ? config.cand_c_puct_v_2 : config.cur_c_puct_v_2;

  Game game;
  std::unique_ptr<NodeTable> node_table_b;
  if (use_mcgs_b) {
    node_table_b = std::make_unique<McgsNodeTable>();
  } else {
    node_table_b = std::make_unique<MctsNodeTable>();
  }
  std::unique_ptr<NodeTable> node_table_w;
  if (use_mcgs_w) {
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

  GumbelEvaluator gumbel_b(black_nn, thread_id);
  GumbelEvaluator gumbel_w(white_nn, thread_id);
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
    GumbelEvaluator& gumbel = color_to_move == BLACK ? gumbel_b : gumbel_w;

    // NN statistics.
    float cur_qz_nn = cur_tree->init_outcome_est;
    float cur_score_est_nn = cur_tree->init_score_est;
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

    // Search.
    int n = color_to_move == BLACK ? n_b : n_w;
    int k = color_to_move == BLACK ? k_b : k_w;
    float noise_scaling =
        color_to_move == BLACK ? noise_scaling_b : noise_scaling_w;
    bool use_puct = color_to_move == BLACK ? use_puct_b : use_puct_w;
    bool use_lcb = color_to_move == BLACK ? use_lcb_b : use_lcb_w;
    float c_puct = color_to_move == BLACK ? c_puct_b : c_puct_w;
    bool var_scale_cpuct =
        color_to_move == BLACK ? var_scale_cpuct_b : var_scale_cpuct_w;
    bool use_mcgs = color_to_move == BLACK ? use_mcgs_b : use_mcgs_w;
    bool use_puct_v = color_to_move == BLACK ? use_puct_v_b : use_puct_v_w;
    float c_puct_v_2 = color_to_move == BLACK ? c_puct_v_2_b : c_puct_v_2_w;
    auto begin = std::chrono::high_resolution_clock::now();
    GumbelResult gumbel_res =
        use_puct
            ? gumbel.SearchRootPuct(
                  probability, game, player_table, player_tree, color_to_move,
                  n,
                  PuctParams{use_lcb ? PuctRootSelectionPolicy::kLcb
                                     : PuctRootSelectionPolicy::kVisitCount,
                             c_puct, 0.45f, c_puct_v_2, use_puct_v,
                             var_scale_cpuct, 1.0f})
            : gumbel.SearchRoot(probability, game, player_table, player_tree,
                                color_to_move,
                                mcts::GumbelSearchParams{n, k, noise_scaling});
    auto end = std::chrono::high_resolution_clock::now();
    if (VOutcome(player_tree) < kResignThreshold) {
      LOG_TO_SINK(INFO, sink) << "Player " << ToString(color_to_move)
                              << " Resigned. Qz: " << VOutcome(player_tree);
      player_resigned = color_to_move;
      break;
    }

    // Post-search statistics.
    float cur_n_post = N(cur_tree);
    float cur_q_post = V(cur_tree);
    float cur_q_outcome_post = VOutcome(cur_tree);
    float cur_score_post = Score(cur_tree);
    float cand_n_post = N(cand_tree);
    float cand_q_post = V(cand_tree);
    float cand_q_outcome_post = VOutcome(cand_tree);
    float cand_score_post = Score(cand_tree);
    float cur_var_post = VVar(cur_tree);
    float cand_var_post = VVar(cand_tree);

    // Commit move changes.
    Loc move = gumbel_res.mcts_move;
    float move_q = Q(player_tree, move);
    game.PlayMove(move, color_to_move);
    color_to_move = OppositeColor(color_to_move);

    // Advance to next roots.
    TreeNode* next_player = player_tree->children[move];
    TreeNode* next_opp = opp_tree->children[move];
    if (!next_player) {
      // After the move, it's the opponent's turn (color_to_move was already flipped)
      next_player = player_table->GetOrCreate(game.board().hash(),
                                              color_to_move, game.IsGameOver());
    }
    if (!next_opp) {
      // After the move, it's the opponent's turn (color_to_move was already flipped)
      next_opp = opp_table->GetOrCreate(game.board().hash(), color_to_move,
                                        game.IsGameOver());
    }

    int num_nodes_reaped = 0, reap_time_us = 0;
    auto reap_begin = std::chrono::steady_clock::now();
    num_nodes_reaped += player_table->Reap(next_player);
    num_nodes_reaped += opp_table->Reap(next_opp);
    auto reap_end = std::chrono::steady_clock::now();
    reap_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       reap_end - reap_begin)
                       .count();

    player_tree = next_player;
    opp_tree = next_opp;

    // Time.
    auto search_dur =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    search_dur_ema = search_dur_ema == 0
                         ? search_dur
                         : (search_dur_ema * 0.9 + search_dur * 0.1);

    // Log.
    [&]() {
      std::string root_color = ToString(OppositeColor(color_to_move));
      std::stringstream s;
      s << "\n----- Move Num: " << game.num_moves() << " -----\n";
      s << "N: " << n << ", K: " << k << ", Noise Scaling: " << noise_scaling
        << ", PUCT: " << use_puct << ", LCB: " << use_lcb
        << ", cPUCT: " << c_puct << ", cPUCT Var Scaling: " << var_scale_cpuct
        << ", MCGS: " << use_mcgs << ", PUCT-V: " << use_puct_v
        << ", cPUCT_2: " << c_puct_v_2 << "\n";
      s << "Gumbel Move: " << move << ", q: " << move_q << "\n";
      s << "Last 5 Moves: " << game.move(game.num_moves() - 5) << ", "
        << game.move(game.num_moves() - 4) << ", "
        << game.move(game.num_moves() - 3) << ", "
        << game.move(game.num_moves() - 2) << ", "
        << game.move(game.num_moves() - 1) << "\n";
      s << "Cur Color: " << (cur_is_black ? ToString(BLACK) : ToString(WHITE))
        << ", Cand Color: "
        << (cur_is_black ? ToString(WHITE) : ToString(BLACK)) << "\n";
      s << "(" << root_color << ") Cur Tree NN Stats:\n  Q_z: " << cur_qz_nn
        << "\n  Score: " << cur_score_est_nn << "\n";
      s << "(" << root_color
        << ") Cur Tree Stats, Pre-Search :\n  N: " << cur_n_pre
        << "\n  Q: " << cur_q_pre << "\n  Q_z: " << cur_q_outcome_pre
        << "\n  Score: " << cur_score_pre << ", Variance: " << cur_var_pre
        << "\n";
      s << "(" << root_color
        << ") Cur Tree Stats, Post-Search :\n  N: " << cur_n_post
        << "\n  Q: " << cur_q_post << "\n  Q_z: " << cur_q_outcome_post
        << "\n  Score: " << cur_score_post << ", Variance: " << cur_var_post
        << "\n";
      s << "(" << root_color << ") Cand Tree NN Stat:\n  Q_z: " << cand_qz_nn
        << "\n  Score: " << cand_score_est_nn << "\n";
      s << "(" << root_color
        << ") Cand Tree Stats, Pre-Search :\n  N: " << cand_n_pre
        << "\n  Q: " << cand_q_pre << "\n  Q_z: " << cand_q_outcome_pre
        << "\n  Score: " << cand_score_pre << ", Variance: " << cand_var_pre
        << "\n";
      s << "(" << root_color
        << ") Cand Tree Stats, Post-Search :\n  N: " << cand_n_post
        << "\n  Q: " << cand_q_post << "\n  Q_z: " << cand_q_outcome_post
        << "\n  Score: " << cand_score_post << ", Variance: " << cand_var_post
        << "\n";
      s << "Player to Move: " << ToString(color_to_move) << ", "
        << (color_to_move == BLACK ? (cur_is_black ? "CUR" : "CAND")
                                   : (cur_is_black ? "CAND" : "CUR"))
        << "\n";
      s << "Board:\n" << game.board() << "\n";
      s << "Nodes Reaped: " << num_nodes_reaped
        << ", Reap Time: " << reap_time_us << "us\n";
      s << "Search Took " << search_dur << "us. Search EMA: " << search_dur_ema
        << "us.\n";

      LOG_TO_SINK(INFO, sink) << s.str();
    }();
  }

  cur_nn->UnregisterThread(thread_id);
  cand_nn->UnregisterThread(thread_id);
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
            << ", White Score: " << game_result.wscore << " (" << thread_id
            << ")";

  const std::string& b_name = cur_is_black ? cur_name : cand_name;
  const std::string& w_name = cur_is_black ? cand_name : cur_name;
  recorder->RecordEvalGame(thread_id, game, b_name, w_name);
  result.set_value(EvalResult{winner, game.num_moves()});
}
