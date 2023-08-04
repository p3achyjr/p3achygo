#include "cc/eval/eval.h"

#include <chrono>
#include <sstream>

#include "cc/constants/constants.h"
#include "cc/core/file_log_sink.h"
#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"

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
  float c_puct_b = cur_is_black ? config.cur_c_puct : config.cand_c_puct;
  float c_puct_w = cur_is_black ? config.cand_c_puct : config.cur_c_puct;

  Game game;
  std::unique_ptr<TreeNode> btree = std::make_unique<TreeNode>();
  std::unique_ptr<TreeNode> wtree = std::make_unique<TreeNode>();
  Color player_resigned = EMPTY;

  GumbelEvaluator gumbel_b(black_nn, thread_id);
  GumbelEvaluator gumbel_w(white_nn, thread_id);
  auto color_to_move = BLACK;
  while (!game.IsGameOver()) {
    std::unique_ptr<TreeNode>& player_tree =
        color_to_move == BLACK ? btree : wtree;
    std::unique_ptr<TreeNode>& opp_tree =
        color_to_move == BLACK ? wtree : btree;
    std::unique_ptr<TreeNode>& cur_tree = cur_is_black ? btree : wtree;
    std::unique_ptr<TreeNode>& cand_tree = cur_is_black ? wtree : btree;
    GumbelEvaluator& gumbel = color_to_move == BLACK ? gumbel_b : gumbel_w;

    // NN statistics.
    float cur_qz_nn = cur_tree->outcome_est;
    float cur_score_est_nn = cur_tree->score_est;
    float cand_qz_nn = cand_tree->outcome_est;
    float cand_score_est_nn = cand_tree->score_est;

    // Pre-search statistics.
    float cur_n_pre = N(cur_tree.get());
    float cur_q_pre = V(cur_tree.get());
    float cur_q_outcome_pre = VOutcome(cur_tree.get());
    float cand_n_pre = N(cand_tree.get());
    float cand_q_pre = V(cand_tree.get());
    float cand_q_outcome_pre = VOutcome(cand_tree.get());

    // Search.
    int n = color_to_move == BLACK ? n_b : n_w;
    int k = color_to_move == BLACK ? k_b : k_w;
    float noise_scaling =
        color_to_move == BLACK ? noise_scaling_b : noise_scaling_w;
    bool use_puct = color_to_move == BLACK ? use_puct_b : use_puct_w;
    float c_puct = color_to_move == BLACK ? c_puct_b : c_puct_w;
    auto begin = std::chrono::high_resolution_clock::now();
    GumbelResult gumbel_res =
        use_puct ? gumbel.SearchRootPuct(probability, game, player_tree.get(),
                                         color_to_move, n, c_puct)
                 : gumbel.SearchRoot(probability, game, player_tree.get(),
                                     color_to_move, n, k, noise_scaling);
    auto end = std::chrono::high_resolution_clock::now();
    if (VOutcome(player_tree.get()) < kResignThreshold) {
      LOG_TO_SINK(INFO, sink)
          << "Player " << ToString(color_to_move)
          << " Resigned. Qz: " << VOutcome(player_tree.get());
      player_resigned = color_to_move;
      break;
    }

    // Post-search statistics.
    float cur_n_post = N(cur_tree.get());
    float cur_q_post = V(cur_tree.get());
    float cur_q_outcome_post = VOutcome(cur_tree.get());
    float cand_n_post = N(cand_tree.get());
    float cand_q_post = V(cand_tree.get());
    float cand_q_outcome_post = VOutcome(cand_tree.get());

    // Commit move changes.
    Loc move = gumbel_res.mcts_move;
    float move_q = Q(player_tree.get(), move);
    game.PlayMove(move, color_to_move);
    color_to_move = OppositeColor(color_to_move);

    player_tree = std::move(player_tree->children[move]);
    opp_tree = std::move(opp_tree->children[move]);
    if (!player_tree) player_tree = std::make_unique<TreeNode>();
    if (!opp_tree) opp_tree = std::make_unique<TreeNode>();

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
        << ", PUCT: " << use_puct << "\n";
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
        << "\n  Q: " << cur_q_pre << "\n  Q_z: " << cur_q_outcome_pre << "\n";
      s << "(" << root_color
        << ") Cur Tree Stats, Post-Search :\n  N: " << cur_n_post
        << "\n  Q: " << cur_q_post << "\n  Q_z: " << cur_q_outcome_post << "\n";
      s << "(" << root_color << ") Cand Tree NN Stat:\n  Q_z: " << cand_qz_nn
        << "\n  Score: " << cand_score_est_nn << "\n";
      s << "(" << root_color
        << ") Cand Tree Stats, Pre-Search :\n  N: " << cand_n_pre
        << "\n  Q: " << cand_q_pre << "\n  Q_z: " << cand_q_outcome_pre << "\n";
      s << "(" << root_color
        << ") Cand Tree Stats, Post-Search :\n  N: " << cand_n_post
        << "\n  Q: " << cand_q_post << "\n  Q_z: " << cand_q_outcome_post
        << "\n";
      s << "Player to Move: " << ToString(color_to_move) << ", "
        << (color_to_move == BLACK ? (cur_is_black ? "CUR" : "CAND")
                                   : (cur_is_black ? "CAND" : "CUR"))
        << "\n";
      s << "Board:\n" << game.board() << "\n";
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
