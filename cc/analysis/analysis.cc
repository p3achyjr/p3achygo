#include "cc/analysis/analysis.h"

#include <algorithm>
#include <sstream>

#include "cc/constants/constants.h"
#include "cc/game/loc.h"

namespace analysis {
namespace {
using namespace ::game;
using namespace ::mcts;
}  // namespace

AnalysisSnapshot ConstructAnalysisSnapshot(TreeNode* const node) {
  static constexpr int kMaxPvDepth = 8;
  struct Child {
    Loc loc;
    float prior;
    TreeNode* node;
  };
  auto best_children = [](TreeNode* const node) {
    std::vector<Child> children;
    for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
      std::unique_ptr<TreeNode>& child = node->children[i];
      if (child) {
        children.emplace_back(
            Child{AsLoc(i), node->move_probs[i], child.get()});
      }
    }

    std::sort(children.begin(), children.end(),
              [](Child c0, Child c1) { return N(c0.node) > N(c1.node); });

    return children;
  };

  // "Freezing" the tree seems hard without making some kind of deepcopy, so
  // we will compute principal variations online. The contents of the tree
  // may change, but we can live with that.
  AnalysisSnapshot snapshot;
  std::vector<Child> first_children = best_children(node);
  for (Child& child : first_children) {
    AnalysisSnapshot::Row row;
    row.loc = child.loc;
    row.visits = N(child.node);
    row.winrate = (V(child.node) + 1.0f) / 2.0f;
    row.prior = child.prior;

    std::vector<Child> pv = {child};
    for (int _ = 0; _ < kMaxPvDepth - 1; ++_) {
      std::vector<Child> children = best_children(pv.back().node);
      if (children.empty()) {
        break;
      }

      pv.emplace_back(children[0]);
    }

    std::transform(pv.begin(), pv.end(),
                   std::back_inserter(row.principal_variation),
                   [](Child& child) { return child.loc; });
    snapshot.rows.emplace_back(std::move(row));
  }

  return snapshot;
}
}  // namespace analysis
