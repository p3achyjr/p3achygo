#include "callbacks.h"

namespace {
using namespace ::core;
using namespace ::game;
using namespace ::mcts;

void dump_entry(BiasCallback::Entry e) {
  const auto color_to_string = [](Color color) {
    if (color == BLACK) {
      return "BLACK";
    } else if (color == WHITE) {
      return "WHITE";
    } else {
      return "?????";
    }
  };
  std::cout << "Color: " << color_to_string(e.color_to_move)
            << "\nLast Moves: ";
  for (const auto move : e.last_five_moves) {
    std::cout << move << " ";
  }
  std::cout << "\nPosition\n";
  std::cout << game::ToString(e.position) << "\n";
  std::cout << "Num Visits: " << e.num_visits << "\n";
  std::cout << "NN Eval: " << e.nn_eval << "\n";
  std::cout << "MCTS Eval: " << e.mcts_eval << "\n";
  std::cout << "NN Outcome: " << e.nn_v << "\n";
  std::cout << "MCTS Outcome: " << e.mcts_v << "\n";
  std::cout << "NN Score: " << e.nn_score << "\n";
  std::cout << "MCTS Score: " << e.mcts_score << "\n";
  std::cout << "Bias: " << e.nn_eval - e.mcts_eval << "\n";
  std::cout << "----------------------\n";
}
}  // namespace

BiasCallback::BiasCallback()
    : game_top_bias_positions(Heap<Entry, Cmp>(Cmp{}, 10)),
      episode_top_bias_positions(Heap<Entry, Cmp>(Cmp{}, 20)) {}

void BiasCallback::OnMove(const Game& game, const Color color_to_move,
                          const TreeNode* root,
                          const GumbelResult& search_result) {
  std::vector<Move> last_moves;
  for (int move_num = game.num_moves() - 5; move_num < game.num_moves();
       ++move_num) {
    if (move_num < 0) continue;
    last_moves.push_back(game.move(move_num));
  }
  Entry entry{
      game.board().position(),
      color_to_move,
      last_moves,
      root->init_util_est,
      root->v,
      root->init_outcome_est,
      root->v_outcome,
      root->init_score_est,
      root->score,
      root->n,
  };
  game_top_bias_positions.PushHeap(entry);
  episode_top_bias_positions.PushHeap(entry);

  // std::cout << "Move Num: " << game.num_moves() << "\n";
  // std::cout << game::ToString(game.board().position()) << "\n\n";
}

void BiasCallback::OnGameEnd(const game::Game& game) {
  const auto result_to_string = [](Game::Result result) {
    std::string winner =
        result.winner == BLACK ? "B" : (result.winner == WHITE ? "W" : "?");
    auto score = result.winner == BLACK ? result.bscore - result.wscore
                                        : result.wscore - result.bscore;
    return winner + "+" + std::to_string(score);
  };
  std::cout << "Game Result: " << result_to_string(game.result()) << "\n";
  std::cout << "Dumping Most Biased Positions for Game\n";
  while (game_top_bias_positions.Size() > 0) {
    Entry e = game_top_bias_positions.PopHeap();
    dump_entry(e);
  }
}

void BiasCallback::OnEpisodeEnd() {
  std::cout << "Dumping Most Biased Positions for Episode\n";
  while (episode_top_bias_positions.Size() > 0) {
    Entry e = episode_top_bias_positions.PopHeap();
    dump_entry(e);
  }
}
