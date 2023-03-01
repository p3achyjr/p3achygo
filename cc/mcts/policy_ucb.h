#ifndef __MCTS_MCTS_UCB_H_
#define __MCTS_MCTS_UCB_H_

#include "cc/mcts/policy.h"
#include "cc/mcts/types.h"

namespace mcts {

template <typename ActionT>
class MctsPolicyUcb : MctsPolicy {
 public:
  MctsPolicyUcb(MctsState<ActionT>* initial_state);
  void SelectRootAction();
  void SelectNonRootAction();
  void Expand();
  void Backward();
}

}  // namespace mcts

#endif  // __MCTS_MCTS_UCB_H_