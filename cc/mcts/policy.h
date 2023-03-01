#ifndef __MCTS_MCTS_POLICY_H_
#define __MCTS_MCTS_POLICY_H_

namespace mcts {

/*
 * Interface that defines action selection for each step in MCTS.
 *
 * Broadly, MCTS can be divided into the following steps:
 *
 * 1. Select Action at Root (Current Board Position).
 * 2. Select Action at Non-Root.
 * 3. Expand Leaf.
 * 4. Backward (Walk MCTS path backward and update MCTS state).
 */

class MctsPolicy {
 public:
  virtual void SelectRootAction() = 0;
  virtual void SelectNonRootAction() = 0;
  virtual void Expand() = 0;
  virtual void Backward() = 0;
};

}  // namespace mcts

#endif  // __MCTS_MCTS_POLICY_H_