#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <optional>

#include "absl/synchronization/mutex.h"
#include "cc/core/heap.h"
#include "cc/game/loc.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/search_policy_parallel.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

struct GlobalSearchState;
using TopActions = std::array<std::pair<int, float>, 4>;
using PathElem = std::tuple<TreeNode*, game::Loc, TopActions>;
using SearchPath = absl::InlinedVector<PathElem, 128>;

enum class DescentPolicyKind : uint8_t {
  kDeterministic = 0,
};

enum class CollisionPolicyKind : uint8_t {
  kAbort = 0,
  kRetry = 1,
  kSmartRetry = 2,
};

enum class CollisionDetectorKind : uint8_t {
  kNoOp = 0,
  kNInFlight = 1,
  kLevelSaturation = 2,
  kProduct = 3,
};

struct GlobalSearchState {
  absl::Mutex mu;

  // Per-round state; all guarded by mu.
  bool did_signal = false;
  // Flips on every round completion. Workers capture their expected parity
  // before entering Barrier 2 and wait for it to change. Safe because a round
  // can only complete when all num_workers threads participate — so the bit
  // cannot flip twice before any stuck worker observes the first flip.
  bool round_parity = false;
  int descent_remaining = 0;
  int pending = 0;  // leaf evals queued but not yet fetched
  int round_remaining = 0;

  // Immutable after initialization.
  int num_workers = 0;
  int visit_budget = 0;
  std::array<std::atomic<int>, 8> pending_each_level = {};
  int max_pending_each_level = 0;

  void inc_pending_at_level(int level) {
    if (level < 0 || level >= (int)pending_each_level.size()) {
      return;
    }
    pending_each_level[level].fetch_add(1, std::memory_order_release);
  }

  // Lock-free global counters.
  std::atomic<int> total_num_visits = 0;
  std::atomic<int> total_num_aborted = 0;
  std::atomic<int> total_num_collisions = 0;
  std::atomic<bool> should_stop = false;
  bool should_stop_this_round = false;
};

class Search final {
 public:
  struct Params {
    int num_threads;
    int total_visit_budget;
    int total_visit_time_ms;
    PuctParams puct_params;
    QFnKind q_fn_kind;
    NFnKind n_fn_kind;
    DescentPolicyKind descent_policy_kind;
    CollisionPolicyKind collision_policy_kind;
    CollisionDetectorKind collision_detector_kind;
    float vl_delta = -1.0f;
    int max_collision_retries = 4;
  };
  struct Result {
    game::Loc move;
    size_t num_visits;
    size_t num_aborted;
    size_t num_collisions;
    size_t time_ms;
  };
  explicit Search(nn::NNInterface::Slot slot);
  ~Search() = default;
  // Disable Copy and Move.
  Search(Search const&) = delete;
  Search& operator=(Search const&) = delete;
  Search(Search&&) = delete;
  Search& operator=(Search&&) = delete;

  Result Run(core::Probability& probability, game::Game& game,
             NodeTable* node_table, TreeNode* const root,
             game::Color color_to_move, Params params);

 private:
  nn::NNInterface::Slot slot_;
};

/*
 * Various descent/collision policies.
 */
using DescentStep = std::pair<game::Loc, TopActions>;

struct CollisionResult {
  enum class Action : uint8_t {
    kAbort = 0,
    kRetry = 1,
  };
  Action action;
  std::optional<SearchPath> retry_path_prefix;
};

enum class CollisionKind : uint8_t {
  kNone = 0,
  kRecoverable = 1,
  kUnrecoverable = 2,
};

/*
 * Always picks best PUCT action.
 */
template <typename QFn, typename NFn>
class DeterministicDescentPolicy final {
 public:
  DeterministicDescentPolicy(PuctParams puct_params, const QFn& q_fn,
                             const NFn& n_fn)
      : puct_scorer_(puct_params, q_fn, n_fn) {}
  ~DeterministicDescentPolicy() = default;

  inline DescentStep Run(const GlobalSearchState& global_search_state,
                         const TreeNode* node, const game::Game& game,
                         game::Color color) {
    const TopActions top_actions = puct_scorer_.TopScores(node, game, color);
    return {game::AsLoc(top_actions[0].first), top_actions};
  }

 private:
  const PuctScorer<QFn, NFn> puct_scorer_;
};

/*
 * Always aborts collisions.
 */
class AbortCollisionPolicy final {
 public:
  AbortCollisionPolicy() = default;
  ~AbortCollisionPolicy() = default;
  inline CollisionResult Handle(const GlobalSearchState& global_search_state,
                                const SearchPath& search_path) {
    return {CollisionResult::Action::kAbort, std::nullopt};
  }

  inline void Reset() {}
};

/*
 * Retries collisions up to a certain limit.
 */
class RetryCollisionPolicy final {
 public:
  RetryCollisionPolicy(const int max_num_retries)
      : max_num_retries_(max_num_retries){};
  ~RetryCollisionPolicy() = default;
  inline CollisionResult Handle(const GlobalSearchState& global_search_state,
                                const SearchPath& search_path) {
    if (num_retries_ >= max_num_retries_) {
      return {CollisionResult::Action::kAbort, std::nullopt};
    }

    ++num_retries_;
    return {CollisionResult::Action::kRetry, std::nullopt};
  }

  inline void Reset() { num_retries_ = 0; }

 private:
  const int max_num_retries_;
  int num_retries_ = 0;
};

/*
 * Retries by attempting to find the next-best path.
 */
class SmartRetryCollisionPolicy final {
 public:
  SmartRetryCollisionPolicy(const int max_num_retries)
      : max_num_retries_(max_num_retries){};
  ~SmartRetryCollisionPolicy() = default;
  inline CollisionResult Handle(const GlobalSearchState& global_search_state,
                                const SearchPath& search_path) {
    if (num_retries_ >= max_num_retries_) {
      return {CollisionResult::Action::kAbort, std::nullopt};
    }
    ++num_retries_;
    if (search_path.size() <= 1) {
      return {CollisionResult::Action::kRetry, std::nullopt};
    }

    int min_index = -1;
    float min_diff = std::numeric_limits<float>::max();
    for (int i = 0; i < search_path.size(); ++i) {
      const auto& [node, move, top_actions] = search_path[i];
      if (move == game::kNoopLoc || top_actions[0].first < 0 ||
          top_actions[1].first < 0) {
        continue;
      }

      const float diff =
          std::abs(top_actions[0].second - top_actions[1].second);
      if (std::abs(top_actions[0].second - top_actions[1].second) < min_diff) {
        min_index = i;
        min_diff = diff;
      }
    }

    if (min_index == -1) {
      // out of paths to retry.
      return {CollisionResult::Action::kAbort, std::nullopt};
    }

    // fork at min_index.
    const auto& [node, move, top_actions] = search_path[min_index];
    // if we retry again from here, the forked move is the best move.
    const auto new_move = game::AsLoc(top_actions[1].first);
    const TopActions new_top_actions = {top_actions[1],
                                        top_actions[2],
                                        top_actions[3],
                                        {game::kNoopLoc, -10000}};
    SearchPath new_search_path(search_path.begin(),
                               search_path.begin() + min_index + 1);
    new_search_path[min_index] = {node, new_move, new_top_actions};
    return {CollisionResult::Action::kRetry, new_search_path};
  }

  inline void Reset() { num_retries_ = 0; }

 private:
  struct Cmp {
    bool operator()(const std::pair<int, float>& e0,
                    const std::pair<int, float>& e1) {
      return e0.second > e1.second;
    }
  };
  const int max_num_retries_;
  int num_retries_ = 0;
};
/*
 * Various collision detector policies.
 * These fire during descent at any already-evaluated node, before a leaf is
 * reached, and can trigger an early collision.
 *
 * IsCollision(state, n_in_flight_old, level):
 *   n_in_flight_old - value of child->n_in_flight BEFORE this thread's
 *                     increment (i.e., number of threads already there).
 *   level           - depth of the child being entered (path.size() after
 *                     pushing the parent).
 */

// Never detects a collision.
struct NoOpCollisionDetector final {
  inline bool IsCollision(const GlobalSearchState&, int /*n_in_flight_old*/,
                          int /*level*/) const {
    return false;
  }
};

// Collides when this thread would be the nth or more thread at the node.
// n_threshold should be precomputed (default: log2(num_workers)).
struct NInFlightCollisionDetector final {
  int n_threshold;
  inline bool IsCollision(const GlobalSearchState&, int n_in_flight_old,
                          int /*level*/) const {
    return n_in_flight_old + 1 >= n_threshold;
  }
};

// Collides when pending leaves at this level exceed base_threshold * (level+1).
// base_threshold should be precomputed (default: log2(num_workers)).
struct LevelSaturationCollisionDetector final {
  int base_threshold;
  inline bool IsCollision(const GlobalSearchState& state,
                          int /*n_in_flight_old*/, int level) const {
    if (level < 0 || level >= (int)state.pending_each_level.size()) {
      return false;
    }
    return state.pending_each_level[level].load(std::memory_order_relaxed) >=
           base_threshold * (level + 1);
  }
};

// Collides when both NInFlight and LevelSaturation detectors fire.
struct ProductCollisionDetector final {
  NInFlightCollisionDetector n_detector;
  LevelSaturationCollisionDetector level_detector;
  inline bool IsCollision(const GlobalSearchState& state, int n_in_flight_old,
                          int level) const {
    return n_detector.IsCollision(state, n_in_flight_old, level) &&
           level_detector.IsCollision(state, n_in_flight_old, level);
  }
};

}  // namespace mcts
