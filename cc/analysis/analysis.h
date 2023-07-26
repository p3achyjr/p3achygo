#ifndef ANALYSIS_ANALYSIS_H_
#define ANALYSIS_ANALYSIS_H_

#include <vector>

#include "cc/game/loc.h"
#include "cc/mcts/tree.h"

namespace analysis {

struct AnalysisSnapshot {
  struct Row {
    game::Loc loc;
    int visits;
    float winrate;
    float prior;
    std::vector<game::Loc> principal_variation;
  };

  std::vector<Row> rows;
};

/*
 * Constructs an analysis snapshot from an MCTS tree node.
 *
 * NOTE: The entire subtree under `node`, including itself, must be guaranteed
 * to _only_ grow during the duration of this function. Normal MCTS should only
 * expand the tree. In practice, this means that if tree reuse is disabled,
 * discarding the tree must be done after this function completes.
 *
 * Not thread safe.
 */
AnalysisSnapshot ConstructAnalysisSnapshot(mcts::TreeNode* const node);

}  // namespace analysis

#endif
