#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <optional>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/heap.h"
#include "cc/core/probability.h"
#include "cc/game/loc.h"
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/leaf_evaluator.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/search_policy_parallel.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

struct GlobalSearchState;
class ForkEventSink;
using TopActions = std::array<std::pair<int, float>, 4>;
using PathElem = std::tuple<TreeNode*, game::Loc, TopActions>;
using SearchPath = absl::InlinedVector<PathElem, 128>;

enum class DescentPolicyKind : uint8_t {
  kDeterministic = 0,
  kBuUct = 1,
  // Preemptive forking: sample among the top actions from a tempered softmax
  // over their PUCT scores, so threads spread without needing a collision.
  kSampled = 2,
};

enum class CollisionPolicyKind : uint8_t {
  kAbort = 0,
  kRetry = 1,
  kSmartRetry = 2,
  // As kSmartRetry, but aborts instead of forking when the cheapest fork point
  // is the root. Root forks directly perturb the root visit distribution,
  // which is the search's output.
  kSmartRetryNoRoot = 3,
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

  // Optional bias cache
  BiasCache* bias_cache = nullptr;

  void reset() {
    did_signal = false;
    round_parity = false;
    num_workers = 0;
    visit_budget = 0;
    descent_remaining = 0;
    pending = 0;
    round_remaining = 0;
    for (auto& pending : pending_each_level) {
      pending.store(0, std::memory_order_relaxed);
    }
    max_pending_each_level = 0;
    total_num_visits.store(0, std::memory_order_relaxed);
    total_num_aborted.store(0, std::memory_order_relaxed);
    total_num_collisions.store(0, std::memory_order_relaxed);
    should_stop.store(false, std::memory_order_relaxed);
    should_stop_this_round = false;
    bias_cache = nullptr;
  }
};

class Search final {
 public:
  enum class Mode {
    kConcurrent = 0,
    kBatch = 1,
  };
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
    float vl_delta = -1.5f;
    int max_collision_retries = 4;
    float max_o_ratio = 0.8f;
    // Softmax temperature for DescentPolicyKind::kSampled. <= 0 is
    // deterministic (argmax). Smaller = sharper = closer to deterministic.
    float descent_temperature = 0.05f;
    Mode mode = Search::Mode::kConcurrent;
    ScoreUtilityParams score_util_params;
    // Optional; when set, SmartRetryCollisionPolicy records every fork it
    // takes.
    ForkEventSink* fork_sink = nullptr;
  };
  struct Result {
    game::Loc move;
    size_t num_visits;
    size_t num_aborted;
    size_t num_collisions;
    size_t time_ms;
  };
  explicit Search(nn::NNInterface::Slot slot);
  explicit Search(nn::NNInterface::Slot slot, BiasCache* bias_cache);
  ~Search() = default;
  // Disable Copy and Move.
  Search(Search const&) = delete;
  Search& operator=(Search const&) = delete;
  Search(Search&&) = delete;
  Search& operator=(Search&&) = delete;

  Result Run(core::Probability& probability, game::Game& game,
             NodeTable* node_table, TreeNode* const root,
             game::Color color_to_move, Params params);
  void StopSearch() { global_search_state_.should_stop = true; }

 private:
  nn::NNInterface::Slot slot_;
  BiasCache* bias_cache_ = nullptr;
  GlobalSearchState global_search_state_{};
};

/*
 * Instrumentation: one record per fork taken by SmartRetryCollisionPolicy.
 * Lets a caller ask how strongly the network preferred the path that collided,
 * and what the retry went to instead. Off unless Params::fork_sink is set.
 */
// Max path depth for which per-node stats are recorded.
static constexpr int kMaxForkPathRecord = 8;

struct ForkEvent {
  const TreeNode* node;  // node forked at; valid until its NodeTable dies.
  int fork_depth;        // index of the fork point within the colliding path.
  int path_len;          // length of the colliding path.
  int action_best;       // action PUCT wanted (and that we are abandoning).
  int action_chosen;     // action the retry will take instead.
  float puct_best;
  float puct_chosen;
  float p_best;    // network prior on action_best at the fork node.
  float p_chosen;  // network prior on action_chosen.
  int n_best;      // child visits at fork time.
  int n_chosen;

  // Per-node profile of the whole colliding path, for asking why the argmin
  // lands where it does. Entries beyond the path (or past
  // kMaxForkPathRecord) are -1.
  int path_recorded;  // number of valid entries below.
  std::array<float, kMaxForkPathRecord> path_gap;       // top1 - top2 PUCT.
  std::array<int, kMaxForkPathRecord> path_node_n;      // node->n at descent.
  std::array<int, kMaxForkPathRecord> path_n_children;  // visited children.
};

class ForkEventSink final {
 public:
  void Record(const ForkEvent& e) {
    absl::MutexLock l(&mu_);
    events_.push_back(e);
  }
  std::vector<ForkEvent> Drain() {
    absl::MutexLock l(&mu_);
    std::vector<ForkEvent> out;
    out.swap(events_);
    return out;
  }

 private:
  absl::Mutex mu_;
  std::vector<ForkEvent> events_ ABSL_GUARDED_BY(mu_);
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
                         core::Probability& prob, const TreeNode* node,
                         const game::Game& game, game::Color color,
                         bool is_root = false) {
    const TopActions top_actions =
        puct_scorer_.TopScores(node, game, color, is_root);
    return {game::AsLoc(top_actions[0].first), top_actions};
  }

 private:
  const PuctScorer<QFn, NFn> puct_scorer_;
};

/*
 * Define O'(s, a) = O(s, a) / n, where n is the number of rollouts so far.
 * This descent policy prevents us from descending into nodes where O'(s, a) is
 * too large.
 */
template <typename QFn, typename NFn>
class BuUctDescentPolicy final {
 public:
  BuUctDescentPolicy(PuctParams puct_params, const QFn& q_fn, const NFn& n_fn,
                     const float max_o)
      : puct_scorer_(puct_params, q_fn, n_fn), max_o_(max_o) {}
  ~BuUctDescentPolicy() = default;

  inline DescentStep Run(const GlobalSearchState& global_search_state,
                         core::Probability& prob, const TreeNode* node,
                         const game::Game& game, game::Color color,
                         bool is_root = false) {
    PuctScores pucts = puct_scorer_.ComputeScores(node, is_root);
    std::array<std::pair<int, float>, 4> top_scores = {
        {{game::kNoopLoc, -1000},
         {game::kNoopLoc, -1000},
         {game::kNoopLoc, -1000},
         {game::kNoopLoc, -1000}}};
    std::array<std::pair<int, float>, 4> fallback_scores = {
        {{game::kNoopLoc, -1000},
         {game::kNoopLoc, -1000},
         {game::kNoopLoc, -1000},
         {game::kNoopLoc, -1000}}};
    const auto find_ranking = [](const auto& scores, const float score) -> int {
      for (int r = 0; r < static_cast<int>(scores.size()); ++r) {
        if (score >= scores[r].second) {
          return r;
        }
      }
      return static_cast<int>(scores.size());
    };
    const auto sift = [](auto& scores, const int ranking,
                         const std::pair<int, float> e) {
      if (ranking < 0 || ranking >= static_cast<int>(scores.size())) return;
      for (int i = static_cast<int>(scores.size()) - 1; i > ranking; --i) {
        scores[i] = scores[i - 1];
      }
      scores[ranking] = e;
    };

    for (const auto& [a, puct_score] : pucts) {
      if (!game.IsValidMove(game::AsLoc(a), color)) {
        continue;
      }

      int fallback_ranking = find_ranking(fallback_scores, puct_score);
      int ranking = find_ranking(top_scores, puct_score);
      const TreeNode* child = node->children[a].load(std::memory_order_acquire);
      if (child != nullptr) {
        const float n = child->n;
        const float n_in_flight =
            child->n_in_flight.load(std::memory_order_acquire);
        const float sum_n_in_flights =
            child->sum_n_in_flights.load(std::memory_order_acquire);
        const float o = sum_n_in_flights / (n + n_in_flight);
        if (o > max_o_) {
          ranking = static_cast<int>(top_scores.size());
        }
      }

      sift(fallback_scores, fallback_ranking, {a, puct_score});
      sift(top_scores, ranking, {a, puct_score});
    }

    if (game::AsLoc(top_scores[0].first) != game::kNoopLoc) {
      return {game::AsLoc(top_scores[0].first), top_scores};
    }
    return {game::AsLoc(fallback_scores[0].first), fallback_scores};
  }

 private:
  const PuctScorer<QFn, NFn> puct_scorer_;
  const float max_o_;
};

/*
 * Preemptive forking. Instead of always taking the PUCT argmax, samples among
 * the top actions from softmax(puct / T). Threads then spread on their own
 * rather than needing a collision to push them apart.
 *
 * Uses exp rather than a power transform because PUCT scores are signed (the Q
 * term spans roughly [-1.5, 1.5]); squaring is not monotone in the score and
 * would rank -1.0 above +0.5.
 *
 * top_actions is returned in unchanged PUCT rank order, so collision policies
 * that fork on it are unaffected.
 */
template <typename QFn, typename NFn>
class SampledDescentPolicy final {
 public:
  SampledDescentPolicy(PuctParams puct_params, const QFn& q_fn, const NFn& n_fn,
                       const float temperature)
      : puct_scorer_(puct_params, q_fn, n_fn), temperature_(temperature) {}
  ~SampledDescentPolicy() = default;

  inline DescentStep Run(const GlobalSearchState& global_search_state,
                         core::Probability& prob, const TreeNode* node,
                         const game::Game& game, game::Color color,
                         bool is_root = false) {
    const TopActions top_actions =
        puct_scorer_.TopScores(node, game, color, is_root);
    if (temperature_ <= 0.0f || top_actions[0].first == game::kNoopLoc) {
      return {game::AsLoc(top_actions[0].first), top_actions};
    }

    // Softmax over the valid top actions, shifted by the max for stability.
    const float s_max = top_actions[0].second;
    std::array<float, 4> w{};
    float total = 0.0f;
    for (int i = 0; i < static_cast<int>(top_actions.size()); ++i) {
      if (top_actions[i].first < 0) break;
      w[i] = std::exp((top_actions[i].second - s_max) / temperature_);
      total += w[i];
    }
    if (total <= 0.0f) {
      return {game::AsLoc(top_actions[0].first), top_actions};
    }

    float target = prob.Uniform() * total;
    for (int i = 0; i < static_cast<int>(top_actions.size()); ++i) {
      if (top_actions[i].first < 0) break;
      target -= w[i];
      if (target <= 0.0f) {
        return {game::AsLoc(top_actions[i].first), top_actions};
      }
    }
    return {game::AsLoc(top_actions[0].first), top_actions};
  }

 private:
  const PuctScorer<QFn, NFn> puct_scorer_;
  const float temperature_;
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
      : max_num_retries_(max_num_retries) {};
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
  SmartRetryCollisionPolicy(const int max_num_retries,
                            ForkEventSink* fork_sink = nullptr,
                            const bool allow_root_fork = true)
      : max_num_retries_(max_num_retries),
        fork_sink_(fork_sink),
        allow_root_fork_(allow_root_fork) {};
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
      if (move == game::kNoopLoc || top_actions[1].first < 0) {
        continue;
      }

      const float diff =
          std::abs(top_actions[0].second - top_actions[1].second);
      if (diff < min_diff) {
        min_index = i;
        min_diff = diff;
      }
    }

    if (min_index == -1) {
      // out of paths to retry.
      return {CollisionResult::Action::kAbort, std::nullopt};
    }
    if (min_index == 0 && !allow_root_fork_) {
      // Cheapest deviation is at the root; forking there would reallocate a
      // root visit, which is what move selection reads. Drop the visit instead.
      return {CollisionResult::Action::kAbort, std::nullopt};
    }

    // fork at min_index.
    const auto& [node, move, top_actions] = search_path[min_index];
    // if we retry again from here, the forked move is the best move.
    const auto new_move = game::AsLoc(top_actions[1].first);
    if (fork_sink_ != nullptr) {
      const int a_best = top_actions[0].first;
      const int a_chosen = top_actions[1].first;
      std::array<float, kMaxForkPathRecord> path_gap;
      std::array<int, kMaxForkPathRecord> path_node_n;
      std::array<int, kMaxForkPathRecord> path_n_children;
      path_gap.fill(-1.0f);
      path_node_n.fill(-1);
      path_n_children.fill(-1);
      int recorded = 0;
      for (int i = 0;
           i < static_cast<int>(search_path.size()) && i < kMaxForkPathRecord;
           ++i) {
        const auto& [pnode, pmove, ptop] = search_path[i];
        if (pmove == game::kNoopLoc || ptop[1].first < 0) {
          continue;  // collision leaf, or no second action to compare against.
        }
        path_gap[i] = ptop[0].second - ptop[1].second;
        path_node_n[i] = pnode->n;
        int visited = 0;
        for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
          if (pnode->child_visits[a] > 0) ++visited;
        }
        path_n_children[i] = visited;
        recorded = i + 1;
      }
      fork_sink_->Record(ForkEvent{
          .node = node,
          .fork_depth = min_index,
          .path_len = static_cast<int>(search_path.size()),
          .action_best = a_best,
          .action_chosen = a_chosen,
          .puct_best = top_actions[0].second,
          .puct_chosen = top_actions[1].second,
          .p_best = node->move_probs[a_best],
          .p_chosen = node->move_probs[a_chosen],
          .n_best = node->child_visits[a_best],
          .n_chosen = node->child_visits[a_chosen],
          .path_recorded = recorded,
          .path_gap = path_gap,
          .path_node_n = path_node_n,
          .path_n_children = path_n_children,
      });
    }
    // keep top_actions[0] as the top action as |p0 - pk| is what we want, not
    // |p_{k - 1} - pk|. We cannot reselect the top move, so this is correct.
    const TopActions new_top_actions = {top_actions[0],
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
  const int max_num_retries_;
  ForkEventSink* fork_sink_ = nullptr;
  bool allow_root_fork_ = true;
  int num_retries_ = 0;
};

/*
 * Retries by attempting to find the next-best path.
 */
class GlobalSmartRetryCollisionPolicy final {
 public:
  GlobalSmartRetryCollisionPolicy(const int max_num_retries)
      : max_num_retries_(max_num_retries), fork_points_(ForkCmp{}) {};
  ~GlobalSmartRetryCollisionPolicy() = default;
  inline CollisionResult Handle(const GlobalSearchState& global_search_state,
                                const SearchPath& search_path) {
    if (num_retries_ >= max_num_retries_) {
      return {CollisionResult::Action::kAbort, std::nullopt};
    }
    const int retry_index = num_retries_;
    ++num_retries_;
    if (search_path.size() <= 1) {
      return {CollisionResult::Action::kAbort, std::nullopt};
    }

    // first add everything to the priority queue. cannot fork at leaf, so skip.
    search_paths_.push_back(search_path);
    for (int path_index = 0; path_index < search_path.size() - 1;
         ++path_index) {
      const auto& [node, move, top_actions] = search_path[path_index];
      if (stored_fork_points_.contains(node) || move == game::kNoopLoc) {
        continue;
      }

      stored_fork_points_.insert(node);
      for (int puct_index = 1; puct_index < top_actions.size(); ++puct_index) {
        if (top_actions[puct_index].first < 0) {
          continue;
        }

        const float diff =
            std::abs(top_actions[0].second - top_actions[puct_index].second);
        fork_points_.PushHeap({diff, retry_index, path_index, puct_index});
      }
    }

    if (fork_points_.Size() == 0) {
      return {CollisionResult::Action::kAbort, std::nullopt};
    }

    // get the best fork point.
    const auto [_, path_num, path_index, puct_index] = fork_points_.PopHeap();

    // reconstruct path.
    SearchPath path_prefix(search_paths_[path_num].begin(),
                           search_paths_[path_num].begin() + path_index + 1);
    auto& [node, old_move, top_actions] = path_prefix.back();
    path_prefix.back() = {node,
                          game::AsLoc(top_actions[puct_index].first, BOARD_LEN),
                          top_actions};
    return {CollisionResult::Action::kRetry, path_prefix};
  }

  inline void Reset() {
    num_retries_ = 0;
    search_paths_.clear();
    fork_points_.Clear();
    stored_fork_points_.clear();
  }

 private:
  // priority queue of fork points from all previous paths.
  // (puct_diff, path_num, path_index, puct_index)
  using ForkElem = std::tuple<float, int, int, int>;
  struct ForkCmp final {
    bool operator()(const ForkElem& e0, const ForkElem& e1) {
      return std::get<0>(e0) > std::get<0>(e1);
    }
  };
  const int max_num_retries_;
  int num_retries_ = 0;
  absl::InlinedVector<SearchPath, 4> search_paths_;
  core::Heap<ForkElem, ForkCmp> fork_points_;
  absl::flat_hash_set<TreeNode*> stored_fork_points_;
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
