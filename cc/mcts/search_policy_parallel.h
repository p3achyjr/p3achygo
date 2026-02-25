#pragma once

#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/tree.h"

namespace mcts {
#if 0
struct GlobalSearchState;
using TopActions = std::array<std::pair<int, float>, 4>;
using PathElem = std::tuple<TreeNode*, game::Loc, TopActions>;
using SearchPath = absl::InlinedVector<PathElem, 128>;

class DescentPolicy {
 public:
  struct Result {
    game::Loc selected_move;
    TopActions top_moves;
  };
  virtual Result Run(const GlobalSearchState& global_search_state,
                     const TreeNode* node, const game::Game& game,
                     game::Color color) = 0;
  virtual ~DescentPolicy() = default;

 protected:
  DescentPolicy() = default;
};

class CollisionPolicy {
 public:
  enum class Action : uint8_t {
    kAbort = 0,
    kRetry = 1,
  };

  struct Result {
    Action action;
    std::optional<SearchPath> retry_path_prefix;
  };
  virtual Result Handle(const GlobalSearchState& global_search_state,
                        const TreeNode* node, const game::Game& game,
                        game::Color color) = 0;
  virtual void Reset() = 0;
  virtual ~CollisionPolicy() = default;

 protected:
  CollisionPolicy() = default;
};

class CollisionDetector {
 public:
  enum class Kind : uint8_t {
    kNone = 0,
    kRecoverable = 1,
    kUnrecoverable = 2,
  };

  virtual Kind IsCollision(const GlobalSearchState& global_search_state,
                           const TreeNode* node, const game::Game& game,
                           game::Color color) = 0;
  virtual ~CollisionDetector() = default;

 protected:
  CollisionDetector() = default;
};
#endif
}  // namespace mcts
