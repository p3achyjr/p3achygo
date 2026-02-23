#pragma once

#include "cc/experiments/callbacks.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

void PlayGames(nn::NNInterface* nn_interface, int num_games, int visit_count,
               bool seq_halving, std::vector<Callback*> callbacks);
