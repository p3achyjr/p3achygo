#ifndef __MCTS_TYPES_H_
#define __MCTS_TYPES_H_

#include <vector>

#include "cc/game/board.h"

namespace mcts {

template <typename ActionT>
class MctsState {
 public:
  virtual bool IsTerminal() = 0;
  virtual int Reward() = 0;
  virtual std::vector<ActionT> ValidActions() = 0;
  virtual MctsState& Transition(ActionT action) = 0;
};

using GoAction = game::Loc;

class GoMctsState final : MctsState<GoAction> {
 public:
  GoMctsState(game::Board board);
  ~GoMctsState() = default;
  // Disable Copy
  GoMctsState(const GoMctsState&) = delete;
  GoMctsState& operator=(const GoMctsState&) = delete;

  bool IsTerminal();
  int Reward();
  std::vector<GoAction> ValidActions();
  GoMctsState& Transition(GoAction action);

 private:
  game::Board board_;
};

}  // namespace mcts

#endif  // __MCTS_TYPES_H_