#include "cc/mcts/search.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/mcts/leaf_evaluator.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

#define SPIN_WHILE(cond) \
  while ((cond)) {       \
    _mm_pause();         \
  }

namespace mcts {
namespace {
using namespace ::game;
using namespace ::nn;

// Definitions for topological backprop. Keyed by (depth, node, action,
// is_leaf).
using BackupElem = std::tuple<int, TreeNode*, Loc, bool>;
struct BackupCmp final {
  bool operator()(const BackupElem& e0, const BackupElem& e1) {
    return std::get<0>(e0) < std::get<0>(e1);
  }
};
using BackupPriorityQueue = core::Heap<BackupElem, BackupCmp>;

void AssignBiasCacheEntry(BiasCache* bias_cache, const game::Game& game,
                          TreeNode* node) {
  if (!bias_cache) return;
  std::optional<LocalPattern> local_pattern =
      LocalPattern::FromCurrentPosition(game);
  if (!local_pattern.has_value()) return;
  node->bias_cache_entry = bias_cache->GetOrCreate(*local_pattern);
}

float FetchObsBias(BiasCache* bias_cache, TreeNode* node) {
  return bias_cache != nullptr && node->bias_cache_entry
             ? bias_cache->UpdateAndFetch(node)
             : 0.0f;
}

void FetchLeafEval(GlobalSearchState& global_state,
                   LeafEvaluator* leaf_evaluator, TreeNode* leaf, Game& game,
                   Color color_to_move, Color root_color, float root_score_est,
                   bool needs_nn_eval) {
  if (needs_nn_eval) {
    leaf_evaluator->FetchLeafEval(leaf, game, color_to_move, root_color,
                                  root_score_est);
    absl::MutexLock l(&global_state.mu);
    global_state.pending--;
  }

  if (game.IsGameOver()) {
    game::Scores scores = game.GetScores();
    leaf_evaluator->EvaluateTerminal(scores, leaf, color_to_move, root_color,
                                     root_score_est);
  }
}

void UnsafeGlobalStateReset(GlobalSearchState& global_state) {
  global_state.descent_remaining = global_state.num_workers;
  global_state.did_signal = false;
  for (auto& p : global_state.pending_each_level) {
    p.store(0, std::memory_order_relaxed);
  }
  global_state.round_remaining = global_state.num_workers;
  global_state.round_parity = !global_state.round_parity;
  global_state.should_stop_this_round =
      global_state.should_stop.load(std::memory_order_relaxed);
}

template <typename DescentPolicy, typename CollisionPolicy,
          typename CollisionDetector>
std::pair<SearchPath, std::optional<CollisionResult>> Descend(
    const int worker_id, core::Probability& prob,
    GlobalSearchState& global_state, NNInterface::Slot& slot,
    LeafEvaluator* leaf_evaluator, DescentPolicy& descent_policy,
    CollisionPolicy& collision_policy,
    const CollisionDetector& collision_detector, Game& game,
    NodeTable* node_table, TreeNode* const root, game::Color root_color,
    const float root_score_est, const SearchPath& path_prefix,
    const bool block_on_leaf_eval = true) {
  const auto on_collision = [&](SearchPath& colliding_path)
      -> std::pair<SearchPath, std::optional<CollisionResult>> {
    // undo n_in_flight for every node on the path.
    for (auto& [node, loc, actions] : colliding_path) {
      const float old_n_in_flight =
          node->n_in_flight.fetch_sub(1, std::memory_order_release);
      node->sum_n_in_flights.fetch_sub(old_n_in_flight - 1,
                                       std::memory_order_release);
    }
    return {colliding_path,
            collision_policy.Handle(global_state, colliding_path)};
  };

  CHECK(path_prefix.size() == 0 || std::get<0>(path_prefix[0]) == root);
  Color color_to_move = root_color;
  const int root_n_in_flight =
      root->n_in_flight.fetch_add(1, std::memory_order_release);
  const int root_total_n_in_flight = root->sum_n_in_flights.fetch_add(
      root_n_in_flight, std::memory_order_release);

  SearchPath path = {};
  TreeNode* cur_node = root;
  for (int path_index = 0;; ++path_index) {
    // This loop replays `path` until exhausted. Afterwards, it invokes the
    // descent policy until a leaf is selected.
    const TreeNodeState node_state =
        cur_node->state.load(std::memory_order_relaxed);
    if (node_state != TreeNodeState::kNnEvaluated) {
      path.push_back({cur_node, game::kNoopLoc, {}});
      return on_collision(path);
    }
    int child_n_in_flight = -1;
    int child_sum_in_flights = -1;
    game::Loc next_move = game::kNoopLoc;
    TopActions next_top_actions{};
    TreeNode* child = nullptr;
    {
      absl::MutexLock node_lock(&cur_node->mu);
      auto descent_result =
          path_index < path_prefix.size()
              ? std::make_pair(std::get<1>(path_prefix[path_index]),
                               std::get<2>(path_prefix[path_index]))
              : descent_policy.Run(global_state, cur_node, game, color_to_move,
                                   /*is_root=*/cur_node == root);
      next_move = descent_result.first;
      next_top_actions = descent_result.second;
      CHECK(game.PlayMove(next_move, color_to_move))
          << "Invalid Move " << game::Move{color_to_move, next_move}
          << "  Score=" << next_top_actions[0].second << "\n"
          << game::ToString(game.board().position());
      color_to_move = game::OppositeColor(color_to_move);
      path.push_back({cur_node, next_move, next_top_actions});
      child = cur_node->children[next_move].load(std::memory_order_relaxed);
      if (child != nullptr) {
        child_n_in_flight =
            child->n_in_flight.fetch_add(1, std::memory_order_acq_rel);
        child_sum_in_flights = child->sum_n_in_flights.fetch_add(
            child_n_in_flight, std::memory_order_release);
      } else {
        // leaf case.
        child = node_table->GetOrCreateGuarded(
            game.board().hash(), color_to_move, game.IsGameOver());
        child_n_in_flight =
            child->n_in_flight.fetch_add(1, std::memory_order_acq_rel);
        child_sum_in_flights = child->sum_n_in_flights.fetch_add(
            child_n_in_flight, std::memory_order_release);
        cur_node->children[next_move].store(child, std::memory_order_relaxed);
        if (child->state.load(std::memory_order_relaxed) ==
            TreeNodeState::kNew) {
          if (child_n_in_flight > 0) {
            // Possible in MCGS. some other thread transposed to the same node
            // and won.
            path.push_back({child, game::kNoopLoc, {}});
            return on_collision(path);
          }
          // truly a leaf. In MCGS, it is possible that we fetch a transposed-to
          // node. In that case, we continue descending.
          // No child exists. claim it.
          child->state.store(TreeNodeState::kPending,
                             std::memory_order_release);
          cur_node = child;
          break;
        }
      }
    }

    // Detector-triggered collision.
    if (collision_detector.IsCollision(global_state, child_n_in_flight,
                                       (int)path.size() + 1)) {
      path.push_back({child, game::kNoopLoc, {}});
      return on_collision(path);
    }

    // Continue descent.
    CHECK(child != nullptr);
    cur_node = child;
    const float child_n = child->n;

    // terminal node case.
    if (game.IsGameOver() || child->is_terminal) {
      if (child_n_in_flight == 0) {
        // this thread won. claim this terminal node.
        break;
      } else {
        // two threads hit the same terminal node, and this thread lost.
        path.push_back({child, game::kNoopLoc, {}});
        return on_collision(path);
      }
    }
  }

  // leaf and/or terminal found. the loop above guarantees that we are the only
  // thread pending this node.
  CHECK(cur_node != nullptr);
  path.push_back({cur_node, game::kNoopLoc, {}});
  global_state.inc_pending_at_level(path.size() - 2);
  TreeNode* leaf = cur_node;
  const TreeNodeState leaf_state = leaf->state.load(std::memory_order_relaxed);
  if (game.IsGameOver()) {
    // prevent data race (according to claude)
    leaf->is_terminal = true;
  }

  // Mark all nodes in path as pending update. We must do this update before we
  // cross the barrier.
  for (const auto& [node, move, top_actions] : path) {
    node->is_pending_update.store(true, std::memory_order_release);
  }

  // Only queue NN eval for non-terminal leaves. Terminals are evaluated
  // directly via EvaluateTerminal; MCGS transposed nodes are already evaluated.
  const bool needs_nn_eval =
      (leaf_state != TreeNodeState::kNnEvaluated) && !game.IsGameOver();
  if (needs_nn_eval) {
    leaf_evaluator->QueueEval(prob, game, color_to_move);
  }

  // Signal NN when all workers have finished descending (descent_remaining==0).
  // Track pending NN fetches separately so Barrier 1 knows when all evals are
  // complete (pending==0).
  {
    absl::MutexLock l(&global_state.mu);
    global_state.descent_remaining--;
    if (needs_nn_eval) global_state.pending++;
    if (global_state.descent_remaining == 0 && global_state.pending > 0 &&
        !global_state.did_signal) {
      slot.SignalReadyForInference();
      global_state.did_signal = true;
    }
  }

  if (block_on_leaf_eval) {
    FetchLeafEval(global_state, leaf_evaluator, leaf, game, color_to_move,
                  root_color, root_score_est, needs_nn_eval);
  }

  AssignBiasCacheEntry(global_state.bias_cache, game, leaf);

  return {path, std::nullopt};
}

// Not thread safe. Mirrors BackupStep: always increments n and child_visits,
// decrements n_in_flight, and only calls RecomputeNodeStats (or sets leaf
// values) on the last pop for each node (old n_in_flight == 1). The heap
// processes nodes deepest-first, so all children of a node are guaranteed to
// be fully updated before the node itself is finalized.
void TopologicalBackup(BackupPriorityQueue& backup_pq, BiasCache* bias_cache) {
  while (backup_pq.Size() > 0) {
    auto [_, node, action, is_leaf] = backup_pq.PopHeap();
    node->n += 1;
    if (!is_leaf) {
      node->child_visits[game::AsIndex(action, BOARD_LEN)] += 1;
    }
    const int old_n_in_flight =
        node->n_in_flight.fetch_sub(1, std::memory_order_acq_rel);
    if (old_n_in_flight == 1) {
      if (is_leaf) {
        node->w = node->init_util_est;
        node->w_outcome = node->init_outcome_est;
        node->v = node->init_util_est;
        node->v_outcome = node->init_outcome_est;
      } else {
        const float obs_bias = FetchObsBias(bias_cache, node);
        RecomputeNodeStats(node, obs_bias);
      }
      node->is_pending_update.store(false, std::memory_order_relaxed);
    }
  }
}

void BackupStep(int worker_id, TreeNode* node, game::Loc action, bool is_leaf,
                BiasCache* bias_cache) {
  // While running backup, we want to preserve the following invariant:
  // - When updating the stats for any node, that node's children are fully
  // updated.
  //
  // To ensure this, we can make use of `node->n_in_flight`. This variable
  // tracks the number of workers that have descended through a node. If
  // `n_in_flight` workers descended through a node, then `n_in_flight` workers
  // must also come back through the node. Then, we will create a barrier at
  // each node, waiting for the last worker to arrive before updating the node.
  // Once the node is updated, all threads are free to proceed.
  //
  // This means that once a thread proceeds past any node, that node is fully
  // updated. Therefore, when any thread goes to its parent, its children are
  // fully updated.
  {
    absl::MutexLock l(&node->mu);
    node->n += 1;
    if (!is_leaf) {
      node->child_visits[game::AsIndex(action, BOARD_LEN)] += 1;
    }
  }

  const int old_in_flight =
      node->n_in_flight.fetch_sub(1, std::memory_order_acq_rel);
  DCHECK(old_in_flight > 0);
  if (old_in_flight == 1) {
    // last thread to arrive. responsible for update.
    if (is_leaf) {
      node->w = node->init_util_est;
      node->w_outcome = node->init_outcome_est;
      node->v = node->init_util_est;
      node->v_outcome = node->init_outcome_est;
      node->v_err = node->init_err_est;
    } else {
      const float obs_bias = FetchObsBias(bias_cache, node);
      RecomputeNodeStats(node, obs_bias);
    }
    node->is_pending_update.store(false, std::memory_order_release);
  }
}

void Backup(int worker_id, SearchPath& search_path, BiasCache* bias_cache) {
  // Can ignore the last node in the path because it is either a leaf or
  // terminal, from the perspective of this path.
  for (int i = search_path.size() - 1; i >= 0; --i) {
    auto& [node, action, _] = search_path[i];
    BackupStep(worker_id, node, action, i == (int)search_path.size() - 1,
               bias_cache);
  }
}

template <typename DescentPolicy, typename CollisionPolicy,
          typename CollisionDetector>
void SearchTask(const int worker_id, core::Probability& prob,
                GlobalSearchState& global_state, NNInterface::Slot slot,
                DescentPolicy& descent_policy,
                CollisionPolicy& collision_policy,
                const CollisionDetector& collision_detector, Game& game,
                NodeTable* node_table, TreeNode* const root,
                game::Color color_to_move,
                ScoreUtilityParams score_util_params) {
  LeafEvaluator leaf_evaluator(slot, worker_id, score_util_params);
  const auto should_stop = [&global_state]() {
    if (global_state.should_stop_this_round) {
      return true;
    }

    const int num_visits_so_far =
        global_state.total_num_visits.load(std::memory_order_relaxed);
    if (num_visits_so_far >= global_state.visit_budget) {
      return true;
    }
    return false;
  };
  while (!should_stop()) {
    // Capture parity at round start. We wait for it to flip at Barrier 2.
    // Safe because a round cannot complete until all num_workers threads
    // participate — so the bit cannot flip twice before any stuck worker
    // observes the first flip.
    bool this_parity;
    {
      absl::MutexLock l(&global_state.mu);
      this_parity = global_state.round_parity;
    }

    collision_policy.Reset();
    Game search_game = game;
    auto [search_path, collision_result] = Descend(
        worker_id, prob, global_state, slot, &leaf_evaluator, descent_policy,
        collision_policy, collision_detector, search_game, node_table, root,
        color_to_move, root->init_score_est, {});
    if (collision_result.has_value()) {
      global_state.total_num_collisions.fetch_add(1, std::memory_order_relaxed);
      while (collision_result.has_value()) {
        if (collision_result->action == CollisionResult::Action::kAbort) {
          {
            absl::MutexLock l(&global_state.mu);
            global_state.total_num_aborted.fetch_add(1,
                                                     std::memory_order_relaxed);
            global_state.descent_remaining--;
            if (global_state.descent_remaining == 0 &&
                global_state.pending > 0 && !global_state.did_signal) {
              slot.SignalReadyForInference();
              global_state.did_signal = true;
            }
          }
          break;
        }
        // kRetry: replay the retry path prefix onto a fresh game copy so the
        // game state matches the prefix path before re-descending.
        SearchPath retry_path_prefix =
            collision_result->retry_path_prefix.has_value()
                ? collision_result->retry_path_prefix.value()
                : SearchPath{};
        Game retry_game = game;
        auto [retry_path, retry_collision] =
            Descend(worker_id, prob, global_state, slot, &leaf_evaluator,
                    descent_policy, collision_policy, collision_detector,
                    retry_game, node_table, root, color_to_move,
                    root->init_score_est, retry_path_prefix);
        search_path = retry_path;
        collision_result = retry_collision;
      }
    }

    // Barrier 1: wait until all workers have finished descending and all
    // pending NN fetches are complete. Once this passes, all leaves are
    // evaluated and backprop can begin.
    {
      absl::MutexLock l(&global_state.mu);
      global_state.mu.Await(absl::Condition(
          +[](GlobalSearchState* s) {
            return s->descent_remaining == 0 && s->pending == 0;
          },
          &global_state));
    }

    // Backprop now ready.
    if (!collision_result.has_value()) {
      // We must keep the invariant that all collisions retry until abort before
      // reaching here.
      Backup(worker_id, search_path, global_state.bias_cache);
    }

    // Count this descent as a visit only if we successfully reached a leaf.
    if (!collision_result.has_value()) {
      global_state.total_num_visits.fetch_add(1, std::memory_order_relaxed);
    }

    // Barrier 2: wait for all workers to finish backprop. The last worker
    // resets per-round state and flips round_parity to wake all waiters.
    // Descent now ready. Reset the end-of-round barrier counter before
    // flipping parity so all threads see it zeroed once they exit the wait.
    {
      absl::MutexLock l(&global_state.mu);
      if (--global_state.round_remaining == 0) {
        // Last worker through: reset for next round.
        UnsafeGlobalStateReset(global_state);
      }
      // Wait for round_parity to flip (indicating the round has ended and
      // state has been reset for the next round).
      struct PArg {
        const GlobalSearchState* s;
        bool expected_parity;
      };
      PArg parg{&global_state, this_parity};
      global_state.mu.Await(absl::Condition(
          +[](PArg* a) { return a->s->round_parity != a->expected_parity; },
          &parg));
    }
  }
}

// Launches `num_threads` worker threads, each with its own CollisionPolicy
// instance (so stateful policies are safe) and Probability source.
// DescentPolicy is shared read-only across threads.
template <typename DescentPolicy, typename CollisionPolicy,
          typename CollisionDetector>
void SpawnSearchTasks(GlobalSearchState& global_state, NNInterface::Slot slot,
                      DescentPolicy& descent_policy,
                      CollisionPolicy collision_policy_proto,
                      CollisionDetector collision_detector, Game& game,
                      NodeTable* node_table, TreeNode* root,
                      game::Color color_to_move, int num_threads,
                      ScoreUtilityParams score_util_params) {
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (int worker_id = 0; worker_id < num_threads; ++worker_id) {
    workers.emplace_back([&, worker_id, cp = collision_policy_proto]() mutable {
      core::Probability prob;
      SearchTask(worker_id, prob, global_state, slot, descent_policy, cp,
                 collision_detector, game, node_table, root, color_to_move,
                 score_util_params);
    });
  }
  for (auto& w : workers) w.join();
}

template <typename DescentPolicy, typename CollisionPolicy,
          typename CollisionDetector>
void BatchSearch(GlobalSearchState& global_state, NNInterface::Slot slot,
                 DescentPolicy& _descent_policy,
                 CollisionPolicy _collision_policy_proto,
                 CollisionDetector _collision_detector, Game& game,
                 NodeTable* node_table, TreeNode* root,
                 game::Color color_to_move, int batch_size,
                 PuctParams puct_params) {
  // Currently ignores policies and does a priority-first search.
  // TODO: Assess whether we should use the policies.
  const bool is_graph = node_table->is_graph();
  const auto should_stop = [&]() {
    return global_state.should_stop ||
           global_state.total_num_visits >= global_state.visit_budget;
  };
  core::Probability prob;

  // no-op policy.
  auto n_fn = IdentityN{};
  auto q_fn = IdentityQ{};
  auto descent_policy = DeterministicDescentPolicy(puct_params, q_fn, n_fn);
  auto collision_policy = AbortCollisionPolicy();
  auto collision_detector = NoOpCollisionDetector();

  // aborts
  absl::InlinedVector<bool, 32> did_abort(batch_size, false);
  absl::InlinedVector<Game, 32> worker_games(batch_size);

  // priority queue of fork points from all previous paths.
  // (puct_diff, path_num, path_index, puct_index)
  using ForkElem = std::tuple<float, int, int, int>;
  struct ForkCmp final {
    bool operator()(const ForkElem& e0, const ForkElem& e1) {
      return std::get<0>(e0) > std::get<0>(e1);
    }
  };
  core::Heap<ForkElem, ForkCmp> fork_points(ForkCmp{});
  absl::flat_hash_set<TreeNode*> stored_fork_points;

  // all search paths accumulated in a single round.
  std::vector<SearchPath> search_paths(batch_size);

  // priority queue of backprop elements.
  BackupPriorityQueue backup_pq(BackupCmp{});

  // search loop.
  while (!should_stop()) {
    // descent.
    for (int worker_id = 0; worker_id < batch_size; ++worker_id) {
      LeafEvaluator leaf_evaluator(slot, worker_id);
      Game search_game = game;
      SearchPath search_prefix = [&]() -> SearchPath {
        if (fork_points.Size() == 0) {
          return {};
        }

        const auto [diff, path_num, path_index, puct_index] =
            fork_points.PopHeap();
        SearchPath path_prefix(search_paths[path_num].begin(),
                               search_paths[path_num].begin() + path_index + 1);
        const auto& [fork_node, fork_move, fork_actions] = path_prefix.back();
        const Loc new_move = game::AsLoc(fork_actions[puct_index].first);
        path_prefix.back() = {fork_node, new_move, fork_actions};
        return path_prefix;
      }();
      auto [search_path, collision_result] = Descend(
          worker_id, prob, global_state, slot, &leaf_evaluator, descent_policy,
          collision_policy, collision_detector, search_game, node_table, root,
          color_to_move, root->init_score_est, search_prefix,
          /*block_on_leaf_eval=*/false);
      worker_games[worker_id] = search_game;
      if (collision_result.has_value()) {
        did_abort[worker_id] = true;
        global_state.descent_remaining--;
        global_state.total_num_aborted.fetch_add(1, std::memory_order_relaxed);
        global_state.total_num_collisions.fetch_add(1,
                                                    std::memory_order_relaxed);
        continue;
      }

      search_paths[worker_id] = search_path;

      // add to fork points and backprop nodes.
      for (int path_index = 0; path_index < search_path.size(); ++path_index) {
        const auto& [node, mv, top_actions] = search_path[path_index];
        backup_pq.PushHeap(
            {path_index, node, mv, path_index == search_path.size() - 1});

        // leaf nodes have no valid top_actions; skip fork-point collection.
        if (mv == game::kNoopLoc) continue;

        // do not re-add this node to fork points.
        if (stored_fork_points.contains(node)) {
          continue;
        }

        for (int puct_index = 1; puct_index < top_actions.size();
             ++puct_index) {
          if (top_actions[puct_index].first < 0) {
            continue;
          }
          const float diff =
              std::abs(top_actions[0].second - top_actions[puct_index].second);
          fork_points.PushHeap({diff, worker_id, path_index, puct_index});
        }

        stored_fork_points.insert(node);
      }
    }

    if (!global_state.did_signal && global_state.pending > 0) {
      slot.SignalReadyForInference();
    }

    // one of the descenders should have signalled. Fetch.
    for (int worker_id = 0; worker_id < batch_size; ++worker_id) {
      if (did_abort[worker_id]) {
        continue;
      }
      LeafEvaluator leaf_evaluator(slot, worker_id);
      auto& search_path = search_paths[worker_id];
      auto& game = worker_games[worker_id];
      auto& [leaf, _mv, _actions] = search_path.back();
      const TreeNodeState leaf_state =
          leaf->state.load(std::memory_order_relaxed);
      const bool needs_nn_eval =
          (leaf_state != TreeNodeState::kNnEvaluated) && !game.IsGameOver();
      Color root_color = color_to_move;
      Color leaf_color = search_path.size() % 2 == 1
                             ? root_color
                             : game::OppositeColor(root_color);
      FetchLeafEval(global_state, &leaf_evaluator, leaf, game, leaf_color,
                    root_color, root->init_score_est, needs_nn_eval);
      global_state.total_num_visits.fetch_add(1, std::memory_order_relaxed);
    }

    // topological backprop.
    TopologicalBackup(backup_pq, global_state.bias_cache);

    // reset state.
    fork_points.Clear();
    stored_fork_points.clear();
    backup_pq.Clear();
    for (int worker_id = 0; worker_id < batch_size; ++worker_id) {
      did_abort[worker_id] = false;
      search_paths[worker_id] = {};
      // don't bother clearing games as constructing games is expensive.
    }

    // reset search control
    UnsafeGlobalStateReset(global_state);
  }
}

// Dispatches QFn and NFn independently — O(N_q + N_n) cases rather than
// O(N_q × N_n). The cross-product of instantiations is still produced by the
// compiler, but the dispatch code doesn't grow with it.
template <typename CollisionPolicy, typename CollisionDetector>
void RunWithCollision(GlobalSearchState& global_state, NNInterface::Slot slot,
                      Game& game, NodeTable* node_table, TreeNode* root,
                      game::Color color_to_move,
                      CollisionPolicy collision_policy,
                      CollisionDetector collision_detector,
                      const Search::Params& params) {
  auto with_nfn = [&](auto qfn, auto nfn) {
    if (params.descent_policy_kind == DescentPolicyKind::kBuUct) {
      const float max_o =
          params.max_o_ratio * (float(global_state.num_workers - 1) / 2.0f);
      BuUctDescentPolicy<decltype(qfn), decltype(nfn)> dp(params.puct_params,
                                                          qfn, nfn, max_o);
      if (params.mode == Search::Mode::kConcurrent) {
        SpawnSearchTasks(global_state, slot, dp, collision_policy,
                         collision_detector, game, node_table, root,
                         color_to_move, params.num_threads,
                         params.score_util_params);
      } else {
        BatchSearch(global_state, slot, dp, collision_policy,
                    collision_detector, game, node_table, root, color_to_move,
                    params.num_threads, params.puct_params);
      }
    } else {
      DeterministicDescentPolicy<decltype(qfn), decltype(nfn)> dp(
          params.puct_params, qfn, nfn);
      if (params.mode == Search::Mode::kConcurrent) {
        SpawnSearchTasks(global_state, slot, dp, collision_policy,
                         collision_detector, game, node_table, root,
                         color_to_move, params.num_threads,
                         params.score_util_params);
      } else {
        BatchSearch(global_state, slot, dp, collision_policy,
                    collision_detector, game, node_table, root, color_to_move,
                    params.num_threads, params.puct_params);
      }
    }
  };

  auto with_qfn = [&](auto qfn) {
    switch (params.n_fn_kind) {
      case NFnKind::kIdentity:
        return with_nfn(qfn, IdentityN{});
      case NFnKind::kVirtualVisit:
        return with_nfn(qfn, VirtualVisitN{});
    }
  };

  switch (params.q_fn_kind) {
    case QFnKind::kIdentity:
      return with_qfn(IdentityQ{});
    case QFnKind::kVirtualLoss:
      return with_qfn(VirtualLossQ{params.vl_delta});
    case QFnKind::kVirtualLossSoft:
      return with_qfn(VirtualLossSoftQ{params.vl_delta});
  }
}
template <typename CollisionPolicy>
void RunWithDetector(GlobalSearchState& global_state, NNInterface::Slot slot,
                     Game& game, NodeTable* node_table, TreeNode* root,
                     game::Color color_to_move,
                     CollisionPolicy collision_policy,
                     const Search::Params& params) {
  const int base_threshold =
      std::max(1, (int)std::log2(std::max(params.num_threads, 1)));
  switch (params.collision_detector_kind) {
    case CollisionDetectorKind::kNoOp:
      return RunWithCollision(global_state, slot, game, node_table, root,
                              color_to_move, collision_policy,
                              NoOpCollisionDetector{}, params);
    case CollisionDetectorKind::kNInFlight:
      return RunWithCollision(
          global_state, slot, game, node_table, root, color_to_move,
          collision_policy, NInFlightCollisionDetector{base_threshold}, params);
    case CollisionDetectorKind::kLevelSaturation:
      return RunWithCollision(global_state, slot, game, node_table, root,
                              color_to_move, collision_policy,
                              LevelSaturationCollisionDetector{base_threshold},
                              params);
    case CollisionDetectorKind::kProduct:
      return RunWithCollision(
          global_state, slot, game, node_table, root, color_to_move,
          collision_policy,
          ProductCollisionDetector{
              NInFlightCollisionDetector{base_threshold},
              LevelSaturationCollisionDetector{base_threshold}},
          params);
  }
}

}  // namespace

Search::Search(NNInterface::Slot slot) : Search(slot, nullptr) {}

Search::Search(NNInterface::Slot slot, BiasCache* bias_cache)
    : slot_(slot), bias_cache_(bias_cache) {
  CHECK(slot_.signal_kind() == NNInterface::SignalKind::kExplicit);
}

Search::Result Search::Run(core::Probability& probability, Game& game,
                           NodeTable* node_table, TreeNode* const root,
                           Color color_to_move, Params params) {
  // TODO: centralize.
  const auto best_lcb_move = [&](const TreeNode* node) {
    std::array<std::pair<int, float>, constants::kMaxMovesPerPosition>
        move_lcbs;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      move_lcbs[a] = {a, Lcb(node, a)};
    }

    std::sort(
        move_lcbs.begin(), move_lcbs.end(),
        [](const auto& p0, const auto& p1) { return p1.second < p0.second; });

    for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
      auto [a, _] = move_lcbs[i];
      if (game.IsValidMove(a, color_to_move)) {
        return game::AsLoc(a);
      }
    }

    return game::kPassLoc;
  };

  const auto search_begin = std::chrono::steady_clock::now();
  GlobalSearchState global_search_state{};
  {
    global_search_state.did_signal = false;
    global_search_state.round_parity = false;
    global_search_state.num_workers = params.num_threads;
    global_search_state.visit_budget = params.total_visit_budget;
    global_search_state.descent_remaining = params.num_threads;
    global_search_state.round_remaining = params.num_threads;
    global_search_state.max_pending_each_level = 100000;
    global_search_state.bias_cache = bias_cache_;
  }

  if (game.IsGameOver()) {
    // not a well-defined search.
    return Search::Result{game::kPassLoc, 0, 0, 0};
  }

  // initialize root if not already initialized.
  if (root->state.load(std::memory_order_relaxed) == TreeNodeState::kNew) {
    LeafEvaluator leaf_evaluator(slot_, 0, params.score_util_params);
    leaf_evaluator.QueueEval(probability, game, color_to_move);
    slot_.SignalReadyForInference();
    leaf_evaluator.FetchRootEval(root, game, color_to_move);
    AssignBiasCacheEntry(global_search_state.bias_cache, game, root);
  }

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  std::thread timer([&, f = std::move(done_future)]() mutable {
    if (params.total_visit_time_ms <= 0) {
      return;
    }

    if (f.wait_for(std::chrono::milliseconds(params.total_visit_time_ms)) ==
        std::future_status::timeout) {
      global_search_state.should_stop.store(true, std::memory_order_relaxed);
    }
  });

  switch (params.collision_policy_kind) {
    case CollisionPolicyKind::kAbort:
      RunWithDetector(global_search_state, slot_, game, node_table, root,
                      color_to_move, AbortCollisionPolicy{}, params);
      break;
    case CollisionPolicyKind::kRetry:
      RunWithDetector(
          global_search_state, slot_, game, node_table, root, color_to_move,
          RetryCollisionPolicy(params.max_collision_retries), params);
      break;
    case CollisionPolicyKind::kSmartRetry:
      RunWithDetector(
          global_search_state, slot_, game, node_table, root, color_to_move,
          SmartRetryCollisionPolicy(params.max_collision_retries), params);
      break;
  }
  done_promise.set_value();
  timer.join();
  const auto search_end = std::chrono::steady_clock::now();
  const auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           search_end - search_begin)
                           .count();
  const auto best_move = best_lcb_move(root);
  const auto visit_count =
      global_search_state.total_num_visits.load(std::memory_order_relaxed);
  const auto abort_count =
      global_search_state.total_num_aborted.load(std::memory_order_relaxed);
  const auto collision_count =
      global_search_state.total_num_collisions.load(std::memory_order_relaxed);
  return Search::Result{best_move, size_t(visit_count), size_t(abort_count),
                        size_t(collision_count), size_t(time_ms)};
}
}  // namespace mcts

#undef SPIN_WHILE
