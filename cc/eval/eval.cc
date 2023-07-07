#include "cc/eval/eval.h"

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

static constexpr int kGumbelN = 160;  // 20, 40 visits.
static constexpr int kGumbelK = 4;

// Threshold under which to immediately resign.
static constexpr float kResignThreshold = -.98;

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
                  std::promise<Winner> result) {
  FileSink sink(logfile.c_str());
  Probability probability(seed);
  bool cur_is_black = thread_id % 2 == 0;
  NNInterface* black_nn = cur_is_black ? cur_nn : cand_nn;
  NNInterface* white_nn = cur_is_black ? cand_nn : cur_nn;

  Game game;
  std::unique_ptr<TreeNode> btree = std::make_unique<TreeNode>();
  std::unique_ptr<TreeNode> wtree = std::make_unique<TreeNode>();
  bool did_resign = false;

  GumbelEvaluator gumbel_b(black_nn, thread_id);
  GumbelEvaluator gumbel_w(white_nn, thread_id);
  auto color_to_move = BLACK;
  while (!game.IsGameOver()) {
    std::unique_ptr<TreeNode>& player_tree =
        color_to_move == BLACK ? btree : wtree;
    std::unique_ptr<TreeNode>& opp_tree =
        color_to_move == BLACK ? wtree : btree;
    GumbelEvaluator& gumbel = color_to_move == BLACK ? gumbel_b : gumbel_w;
    GumbelResult gumbel_res =
        gumbel.SearchRoot(probability, game, player_tree.get(), color_to_move,
                          kGumbelN, kGumbelK);
    if (QOutcome(player_tree.get()) < kResignThreshold) {
      did_resign = true;
      break;
    }

    Loc move = gumbel_res.mcts_move;
    float move_q = QAction(player_tree.get(), move);
    game.PlayMove(move, color_to_move);
    color_to_move = OppositeColor(color_to_move);

    player_tree = std::move(player_tree->children[move]);
    opp_tree = std::move(opp_tree->children[move]);
    if (!player_tree) player_tree = std::make_unique<TreeNode>();
    if (!opp_tree) opp_tree = std::make_unique<TreeNode>();

    LOG_TO_SINK(INFO, sink)
        << "----- Move Num: " << game.num_moves() << " -----";
    LOG_TO_SINK(INFO, sink) << "Gumbel Move: " << move << ", q: " << move_q;
    LOG_TO_SINK(INFO, sink) << "Move Num: " << game.num_moves();
    LOG_TO_SINK(INFO, sink)
        << "Last 5 Moves: " << game.move(game.num_moves() - 5) << ", "
        << game.move(game.num_moves() - 4) << ", "
        << game.move(game.num_moves() - 3) << ", "
        << game.move(game.num_moves() - 2) << ", "
        << game.move(game.num_moves() - 1);
    LOG_TO_SINK(INFO, sink)
        << "Cur Color: " << (cur_is_black ? ToString(BLACK) : ToString(WHITE))
        << ", Cand Color: "
        << (cur_is_black ? ToString(WHITE) : ToString(BLACK));
    LOG_TO_SINK(INFO, sink)
        << "Cur Tree N: " << (cur_is_black ? btree->n : wtree->n);
    LOG_TO_SINK(INFO, sink)
        << "Cur Tree Q: " << (cur_is_black ? btree->q : wtree->q);
    LOG_TO_SINK(INFO, sink)
        << "Cand Tree N: " << (cur_is_black ? wtree->n : btree->n);
    LOG_TO_SINK(INFO, sink)
        << "Cand Tree Q: " << (cur_is_black ? wtree->q : btree->q);
    LOG_TO_SINK(INFO, sink)
        << "Color to Move: " << ToString(player_tree->color_to_move) << ", "
        << (color_to_move == BLACK ? (cur_is_black ? "CUR" : "CAND")
                                   : (cur_is_black ? "CAND" : "CUR"));
    LOG_TO_SINK(INFO, sink) << "Board:\n" << game.board();
  }

  cur_nn->UnregisterThread(thread_id);
  cand_nn->UnregisterThread(thread_id);
  game.WriteResult();

  if (did_resign) {
    game.SetWinner(OppositeColor(color_to_move));
  }

  auto game_result = game.result();
  Winner winner =
      cur_is_black
          ? (game_result.winner == BLACK ? Winner::kCur : Winner::kCand)
          : (game_result.winner == WHITE ? Winner::kCur : Winner::kCand);
  float score_diff = game_result.winner == BLACK
                         ? game_result.bscore - game_result.wscore
                         : game_result.wscore - game_result.bscore;
  LOG(INFO) << "Winner: " << ToString(winner) << ". Cand is "
            << (cur_is_black ? "W" : "B")
            << ", Result: " << (game_result.winner == BLACK ? "B" : "W") << "+"
            << (did_resign ? "R" : absl::StrFormat("%f", score_diff))
            << ", Black Score: " << game_result.bscore
            << ", White Score: " << game_result.wscore;
  result.set_value(winner);
}
