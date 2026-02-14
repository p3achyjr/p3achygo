#pragma once

#include "cc/core/heap.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

struct Callback {
  virtual void OnMove(const game::Game& game, const game::Color color_to_move,
                      const mcts::TreeNode* root,
                      const mcts::GumbelResult& search_result) = 0;
  virtual void OnGameEnd(const game::Game& game) = 0;
  virtual void OnEpisodeEnd() = 0;
};

struct BiasCallback final : public Callback {
  BiasCallback();
  ~BiasCallback() = default;
  void OnMove(const game::Game& game, const game::Color color_to_move,
              const mcts::TreeNode* root,
              const mcts::GumbelResult& search_result) override;
  void OnGameEnd(const game::Game& game) override;
  void OnEpisodeEnd() override;

  struct Entry {
    game::Board::BoardData position;
    game::Color color_to_move;
    std::vector<game::Move> last_five_moves;
    float nn_eval;
    float mcts_eval;
    float nn_v;
    float mcts_v;
    float nn_score;
    float mcts_score;
    int num_visits;
  };
  struct Cmp {
    bool operator()(const Entry& e0, const Entry& e1) {
      const auto e0_bias = std::abs(e0.nn_eval - e0.mcts_eval);
      const auto e1_bias = std::abs(e1.nn_eval - e1.mcts_eval);
      return e0_bias > e1_bias;
    }
  };

  core::Heap<Entry, Cmp> game_top_bias_positions;
  core::Heap<Entry, Cmp> episode_top_bias_positions;
};
