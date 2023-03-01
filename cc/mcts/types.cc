#include "cc/mcts/types.h"

namespace mcts {

GoMctsState::GoMctsState(game::Board board) : {}

bool GoMctsState::IsTerminal() {
  
}

int GoMctsState::Reward() {}

std::vector<GoAction> GoMctsState::ValidActions() {}

GoMctsState& GoMctsState::Transition(GoAction action) {}

}  // namespace mcts