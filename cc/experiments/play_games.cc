#include "cc/experiments/play_games.h"

#include "cc/core/probability.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"

namespace {
using namespace ::game;
using namespace ::core;
using namespace ::mcts;
}  // namespace

void PlayGames(nn::NNInterface* nn_interface, int num_games, int visit_count,
               bool seq_halving, std::vector<Callback*> callbacks) {
  Probability probability;

  for (int i = 0; i < num_games; ++i) {
    Game game;
    Color color_to_move = BLACK;
    std::unique_ptr<NodeTable> node_table = std::make_unique<MctsNodeTable>();
    TreeNode* root =
        node_table->GetOrCreate(game.board().hash(), color_to_move, false);
    GumbelEvaluator gumbel_evaluator(nn_interface, 0);
    while (!game.IsGameOver()) {
      GumbelResult search_result =
          seq_halving
              ? gumbel_evaluator.SearchRoot(
                    probability, game, node_table.get(), root, color_to_move,
                    mcts::GumbelSearchParams{visit_count, 16})
              : gumbel_evaluator.SearchRootPuct(
                    probability, game, node_table.get(), root, color_to_move,
                    visit_count,
                    PuctParams{PuctRootSelectionPolicy::kLcb, 1.0f, 0.45f});
      for (auto& cb : callbacks) {
        cb->OnMove(game, color_to_move, root, search_result);
      }

      // play move.
      const auto move = search_result.mcts_move;
      game.PlayMove(move, color_to_move);
      color_to_move = OppositeColor(color_to_move);
      root = root->children[move];
      if (root == nullptr) {
        root =
            node_table->GetOrCreate(game.board().hash(), color_to_move, false);
      }
      node_table->Reap(root);
    }
    game.WriteResult();
    for (auto& cb : callbacks) {
      cb->OnGameEnd(game);
    }
  }
  for (auto& cb : callbacks) {
    cb->OnEpisodeEnd();
  }
}
