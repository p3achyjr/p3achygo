#include "cc/mcts/search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string>
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

// INST: thread-local timing accumulators, flushed into global_state.inst at
// Barrier 2 (under global mu).
inline int64_t InstNow() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
thread_local int64_t tl_descents = 0, tl_root_wait = 0, tl_root_hold = 0,
                     tl_other_wait = 0, tl_other_hold = 0, tl_table_wait = 0,
                     tl_queue = 0, tl_fetch = 0, tl_backup = 0, tl_b1_wait = 0,
                     tl_b2_wait = 0, tl_root_pre = 0, tl_root_puct = 0,
                     tl_token_wait = 0, tl_seq_reads = 0, tl_seq_overlap = 0;
struct InstHoldTimer {
  int64_t t0;
  int64_t* acc;
  ~InstHoldTimer() { *acc += InstNow() - t0; }
};
void InstRecordSubmit(GlobalSearchState& gs) {  // gs.mu held
  const int64_t t = InstNow();
  gs.inst.submit_max = std::max(gs.inst.submit_max, t);
  gs.inst.submit_sum += t;
  gs.inst.submit_n++;
}
void InstFlushThread(GlobalSearchState& gs) {  // gs.mu held
  gs.inst.descents += tl_descents;
  gs.inst.root_wait += tl_root_wait;
  gs.inst.root_hold += tl_root_hold;
  gs.inst.other_wait += tl_other_wait;
  gs.inst.other_hold += tl_other_hold;
  gs.inst.table_wait += tl_table_wait;
  gs.inst.queue += tl_queue;
  gs.inst.fetch += tl_fetch;
  gs.inst.backup += tl_backup;
  gs.inst.b1_wait += tl_b1_wait;
  gs.inst.b2_wait += tl_b2_wait;
  gs.inst.root_pre += tl_root_pre;
  gs.inst.root_puct += tl_root_puct;
  gs.inst.token_wait += tl_token_wait;
  gs.inst.seq_reads += tl_seq_reads;
  gs.inst.seq_overlap += tl_seq_overlap;
  gs.inst.sr_forks += tl_sr.forks;
  gs.inst.sr_fork_root += tl_sr.fork_root;
  gs.inst.sr_plain += tl_sr.plain;
  gs.inst.sr_depth += tl_sr.depth;
  gs.inst.sr_path_len += tl_sr.path_len;
  gs.inst.sr_rank += tl_sr.rank;
  gs.inst.sr_gap_e6 += tl_sr.gap_e6;
  tl_sr = SrInst{};
  tl_descents = tl_root_wait = tl_root_hold = tl_other_wait = tl_other_hold =
      tl_table_wait = tl_queue = tl_fetch = tl_backup = tl_b1_wait =
          tl_b2_wait = tl_root_pre = tl_root_puct = tl_token_wait =
              tl_seq_reads = tl_seq_overlap = 0;
}
void InstRoundEnd(GlobalSearchState& gs) {  // gs.mu held, last worker
  auto& I = gs.inst;
  const int64_t t = InstNow(), s = I.round_start;
  I.rounds++;
  if (I.submit_n > 0) {
    I.a_submit_mean += I.submit_sum / I.submit_n - s;
    I.a_submit_max += I.submit_max - s;
  }
  const int64_t is = gs.inst_nn != nullptr
                         ? gs.inst_nn->inst_wave_start[gs.inst_wave].load()
                         : nn::g_inst_infer_start.load(),
                ie = gs.inst_nn != nullptr
                         ? gs.inst_nn->inst_wave_end[gs.inst_wave].load()
                         : nn::g_inst_infer_end.load();
  if (is >= s && I.fetch_max > 0) {
    I.nn_rounds++;
    I.a_infer_start += is - s;
    I.a_infer_end += ie - s;
    I.a_fetch_min += I.fetch_min - s;
    I.a_fetch_max += I.fetch_max - s;
  }
  I.a_b1_max += gs.inst_b1_max.load(std::memory_order_relaxed) - s;
  I.a_b2_max += gs.inst_b2_max.load(std::memory_order_relaxed) - s;
  I.a_end += t - s;
  I.round_start = t;
  I.submit_max = I.submit_sum = I.submit_n = 0;
  I.fetch_min = INT64_MAX;
  I.fetch_max = 0;
  gs.inst_b1_max.store(0, std::memory_order_relaxed);
  gs.inst_b2_max.store(0, std::memory_order_relaxed);
}
void InstAtomicMax(std::atomic<int64_t>& a, int64_t v) {
  int64_t cur = a.load(std::memory_order_relaxed);
  while (cur < v &&
         !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
  }
}
// INST: P3_BARRIER=gen selects the generation-counter barriers in SearchTask.
const bool kInstGenBarrier = [] {
  const char* v = std::getenv("P3_BARRIER");
  const bool gen = v != nullptr && std::string(v) == "gen";
  fprintf(stderr, "INST barrier=%s\n", gen ? "gen" : "mutex");
  return gen;
}();
// INST: P3_SPLIT_PUCT=1 scores the n_in_flight-free PUCT terms before taking
// node->mu; =check also recomputes ComputeScores under the lock and CHECKs
// bitwise equality.
const int kInstSplitPuct = [] {
  const char* v = std::getenv("P3_SPLIT_PUCT");
  const int mode = v == nullptr                ? 0
                   : std::string(v) == "1"     ? 1
                   : std::string(v) == "check" ? 2
                                               : 0;
  fprintf(stderr, "INST split_puct=%s\n",
          mode == 0   ? "off"
          : mode == 1 ? "on"
                      : "check");
  return mode;
}();
// INST: P3_ROOT_PICKS=1 computes each round's root moves for all workers in
// one pass (lc0-style) at the barrier-2 last arriver; workers take theirs
// without locking root->mu. =check also verifies the picks (CheckRootPicks).
const int kInstRootPicks = [] {
  const char* v = std::getenv("P3_ROOT_PICKS");
  const int mode = v == nullptr                ? 0
                   : std::string(v) == "1"     ? 1
                   : std::string(v) == "check" ? 2
                                               : 0;
  fprintf(stderr, "INST root_picks=%s\n",
          mode == 0   ? "off"
          : mode == 1 ? "on"
                      : "check");
  return mode;
}();

// INST: P3_SEQCHECK=1 counts PUCT child reads that overlap a pipelined
// backup's recompute of that child (per-node seqlock version, see BackupStep).
const bool kInstSeqCheck = [] {
  const char* v = std::getenv("P3_SEQCHECK");
  const bool on = v != nullptr && std::string(v) == "1";
  fprintf(stderr, "INST seqcheck=%s\n", on ? "on" : "off");
  return on;
}();

// EXP lock-free descent (P3_LOCKFREE=1, KataGo-style): Descend takes no node
// lock. PUCT reads node stats and in-flight counts unlocked (concurrent
// selections at a node may pick the same child: more collisions), a new child
// is published with a CAS on the parent's child pointer (the loser uses the
// winner's node; its own stays in the node table until Reap), and a new leaf
// is claimed with a CAS on its state (kNew -> kPending). Backup is unchanged
// (n / child_visits written under node->mu).
const bool kLockFree = [] {
  const char* v = std::getenv("P3_LOCKFREE");
  const bool on = v != nullptr && std::string(v) == "1";
  fprintf(stderr, "INST lockfree=%s\n", on ? "on" : "off");
  return on;
}();

// EXP pipelined immediate backup (P3_PIPE_IB=1, pipelined search only): a
// descent whose leaf is an NN cache hit backs up at once and descends again,
// until a descent claims a leaf that needs the NN (or a terminal, collision,
// stop, or P3_PIPE_IB_MAX hits this round, default 64); that descent joins the
// round as before.
const bool kPipeIb = [] {
  const char* v = std::getenv("P3_PIPE_IB");
  const bool on = v != nullptr && std::string(v) == "1";
  fprintf(stderr, "INST pipe_ib=%s\n", on ? "on" : "off");
  return on;
}();
const int kPipeIbMax = [] {
  const char* v = std::getenv("P3_PIPE_IB_MAX");
  const int m = v != nullptr ? std::atoi(v) : 64;
  fprintf(stderr, "INST pipe_ib_max=%d\n", m);
  return m;
}();

// INST: computes global_state.root_picks. Caller guarantees no worker is
// touching the tree (barrier-2 last arriver, or before workers spawn) and
// holds global_state.mu.
template <typename DescentPolicy>
void InstComputeRootPicks(GlobalSearchState& gs, DescentPolicy& dp,
                          const Game& game, TreeNode* root,
                          game::Color color_to_move) {
  if constexpr (requires { DescentPolicy::kSupportsSplit; }) {
    if (kInstRootPicks == 0) return;
    const int64_t t0 = InstNow();
    dp.RootPicks(root, game, color_to_move, gs.num_workers, &gs.root_picks);
    gs.inst.picks_ns += InstNow() - t0;
    gs.inst.pick_rounds++;
    if (kInstRootPicks == 2) {
      gs.inst.picks_seq_checked +=
          dp.CheckRootPicks(root, game, color_to_move, gs.root_picks);
    }
    gs.root_picks_valid = true;
  }
}

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
    const int64_t tf0 = InstNow();
    leaf_evaluator->FetchLeafEval(leaf, game, color_to_move, root_color,
                                  root_score_est);
    const int64_t tf1 = InstNow();
    tl_fetch += tf1 - tf0;
    // EXP round-free: no round bookkeeping.
    if (!global_state.round_free) {
      absl::MutexLock l(&global_state.mu);
      global_state.inst.fetch_min = std::min(global_state.inst.fetch_min, tf1);
      global_state.inst.fetch_max = std::max(global_state.inst.fetch_max, tf1);
      global_state.pending--;
    }
  }

  if (game.IsGameOver()) {
    game::Scores scores = game.GetScores();
    leaf_evaluator->EvaluateTerminal(scores, leaf, color_to_move, root_color,
                                     root_score_est);
  }
}

void UnsafeGlobalStateRoundReset(GlobalSearchState& global_state) {
  // INST pipelined search: the visit budget counts both groups. Decided once
  // per round here (the last arriver), so all workers of this group stop on
  // the same round even while the other group keeps adding visits.
  const bool pipe_budget_done =
      global_state.pipe != nullptr &&
      global_state.pipe->total_num_visits.load(std::memory_order_relaxed) >=
          global_state.visit_budget;
  global_state.descent_remaining = global_state.num_workers;
  global_state.did_signal = false;
  for (auto& p : global_state.pending_each_level) {
    p.store(0, std::memory_order_relaxed);
  }
  global_state.round_remaining = global_state.num_workers;
  global_state.round_parity = !global_state.round_parity;
  global_state.should_stop_this_round =
      global_state.should_stop.load(std::memory_order_relaxed) ||
      pipe_budget_done;
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
    const bool block_on_leaf_eval = true, const bool use_root_pick = false,
    const bool allow_ib = false, bool* ib_hit = nullptr) {
  const auto on_collision = [&](SearchPath& colliding_path)
      -> std::pair<SearchPath, std::optional<CollisionResult>> {
    // undo n_in_flight for every node on the path.
    for (auto& [node, loc, actions] : colliding_path) {
      const float old_n_in_flight =
          node->n_in_flight.fetch_sub(1, std::memory_order_release);
      node->n_in_flight_group[global_state.group].fetch_sub(
          1, std::memory_order_release);
      node->sum_n_in_flights.fetch_sub(old_n_in_flight - 1,
                                       std::memory_order_release);
    }
    return {colliding_path,
            collision_policy.Handle(global_state, colliding_path)};
  };

  CHECK(path_prefix.size() == 0 || std::get<0>(path_prefix[0]) == root);
  tl_descents++;
  Color color_to_move = root_color;
  const int root_n_in_flight =
      root->n_in_flight.fetch_add(1, std::memory_order_release);
  root->n_in_flight_group[global_state.group].fetch_add(
      1, std::memory_order_release);
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
      const bool inst_is_root = cur_node == root;
      const bool replay = path_index < path_prefix.size();
      // INST root picks: this worker's root move was chosen at the round start
      // (InstComputeRootPicks); no root scoring and no root->mu here. The child
      // in-flight increment and child creation below use atomics and the
      // guarded node table, as for any concurrent visit.
      const bool take_pick = use_root_pick && path_index == 0 && !replay;
      // INST split PUCT: n_in_flight-free terms are computed before locking.
      constexpr bool kSplitOk = requires { DescentPolicy::kSupportsSplit; };
      const bool split =
          kSplitOk && kInstSplitPuct != 0 && !replay && !take_pick;
      auto* pre = [&]() {
        if constexpr (kSplitOk) {
          thread_local typename DescentPolicy::Precomp tl_pre;
          if (split) {
            const int64_t tp0 = InstNow();
            descent_policy.Precompute(cur_node, inst_is_root, &tl_pre);
            if (inst_is_root) tl_root_pre += InstNow() - tp0;
          }
          return &tl_pre;
        } else {
          return static_cast<void*>(nullptr);
        }
      }();
      const int64_t tw0 = InstNow();
      std::optional<absl::MutexLock> node_lock;
      if (!take_pick && !kLockFree) node_lock.emplace(&cur_node->mu);
      const int64_t tw1 = InstNow();
      (inst_is_root ? tl_root_wait : tl_other_wait) += tw1 - tw0;
      InstHoldTimer inst_hold{tw1,
                              inst_is_root ? &tl_root_hold : &tl_other_hold};
      // INST pipelined search: the other group may have evaluated this node
      // (state kNnEvaluated) without having backed up its visit yet (n == 0).
      // Treat it as in flight, as the round barriers guarantee without
      // pipelining. node->n is written under node->mu, so this read also
      // orders the evaluating thread's writes before ours.
      // EXP round-free: the same holds (no barrier between another worker's
      // evaluation and its backup).
      if ((global_state.pipe != nullptr || global_state.round_free) &&
          !inst_is_root && !replay && cur_node->n == 0) {
        path.push_back({cur_node, game::kNoopLoc, {}});
        return on_collision(path);
      }
      const auto run_policy = [&]() -> DescentStep {
        if constexpr (kSplitOk) {
          if (split) {
            return descent_policy.RunSplit(*pre, cur_node, game, color_to_move,
                                           /*check=*/kInstSplitPuct == 2);
          }
        }
        return descent_policy.Run(global_state, cur_node, game, color_to_move,
                                  /*is_root=*/cur_node == root);
      };
      // INST (P3_SEQCHECK=1): snapshot every child's stats_seq around the
      // PUCT evaluation; a read overlapped a recompute if the version was odd
      // before or changed.
      const auto run_policy_checked = [&]() -> DescentStep {
        if (!kInstSeqCheck) return run_policy();
        std::array<uint32_t, constants::kMaxMovesPerPosition> seq0;
        for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
          const TreeNode* c = cur_node->child(a);
          seq0[a] =
              c == nullptr ? 0 : c->stats_seq.load(std::memory_order_acquire);
        }
        DescentStep r = run_policy();
        for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
          const TreeNode* c = cur_node->child(a);
          if (c == nullptr) continue;
          ++tl_seq_reads;
          const uint32_t s1 = c->stats_seq.load(std::memory_order_acquire);
          if ((seq0[a] & 1u) != 0 || s1 != seq0[a]) ++tl_seq_overlap;
        }
        return r;
      };
      auto descent_result =
          replay      ? std::make_pair(std::get<1>(path_prefix[path_index]),
                                       std::get<2>(path_prefix[path_index]))
          : take_pick ? global_state.root_picks[worker_id]
                      : run_policy_checked();
      if (inst_is_root && !replay) tl_root_puct += InstNow() - tw1;
      next_move = descent_result.first;
      next_top_actions = descent_result.second;
      CHECK(game.PlayMove(next_move, color_to_move))
          << "Invalid Move " << game::Move{color_to_move, next_move}
          << "  Score=" << next_top_actions[0].second << "\n"
          << game::ToString(game.board().position());
      color_to_move = game::OppositeColor(color_to_move);
      path.push_back({cur_node, next_move, next_top_actions});
      // EXP lock-free: acquire pairs with the publishing CAS (the child's
      // fields were written by another thread without our node lock).
      child = cur_node->children[next_move].load(
          kLockFree ? std::memory_order_acquire : std::memory_order_relaxed);
      if (child != nullptr) {
        child_n_in_flight =
            child->n_in_flight.fetch_add(1, std::memory_order_acq_rel);
        child->n_in_flight_group[global_state.group].fetch_add(
            1, std::memory_order_acq_rel);
        child_sum_in_flights = child->sum_n_in_flights.fetch_add(
            child_n_in_flight, std::memory_order_release);
      } else if (kLockFree) {
        // EXP lock-free leaf case: publish the child with a CAS, then claim a
        // new leaf with a CAS on its state.
        const int64_t tt0 = InstNow();
        TreeNode* created = node_table->GetOrCreateGuarded(
            game.board().hash(), color_to_move, game.IsGameOver());
        tl_table_wait += InstNow() - tt0;
        TreeNode* expected = nullptr;
        child = cur_node->children[next_move].compare_exchange_strong(
                    expected, created, std::memory_order_acq_rel,
                    std::memory_order_acquire)
                    ? created
                    : expected;
        child_n_in_flight =
            child->n_in_flight.fetch_add(1, std::memory_order_acq_rel);
        child->n_in_flight_group[global_state.group].fetch_add(
            1, std::memory_order_acq_rel);
        child_sum_in_flights = child->sum_n_in_flights.fetch_add(
            child_n_in_flight, std::memory_order_release);
        TreeNodeState new_state = TreeNodeState::kNew;
        if (child->state.load(std::memory_order_acquire) ==
            TreeNodeState::kNew) {
          if (!child->state.compare_exchange_strong(
                  new_state, TreeNodeState::kPending, std::memory_order_acq_rel,
                  std::memory_order_acquire)) {
            // Another thread claimed it first.
            path.push_back({child, game::kNoopLoc, {}});
            return on_collision(path);
          }
          cur_node = child;
          break;
        }
      } else {
        // leaf case.
        const int64_t tt0 = InstNow();
        child = node_table->GetOrCreateGuarded(
            game.board().hash(), color_to_move, game.IsGameOver());
        tl_table_wait += InstNow() - tt0;
        child_n_in_flight =
            child->n_in_flight.fetch_add(1, std::memory_order_acq_rel);
        child->n_in_flight_group[global_state.group].fetch_add(
            1, std::memory_order_acq_rel);
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
    const int64_t tq0 = InstNow();
    const bool cache_hit = leaf_evaluator->QueueEval(prob, game, color_to_move);
    tl_queue += InstNow() - tq0;
    // EXP pipelined immediate backup: a cache hit is evaluated now and does not
    // join the round (no submit / pending / signal); the caller backs it up.
    if (allow_ib && cache_hit) {
      leaf_evaluator->FetchLeafEval(leaf, game, color_to_move, root_color,
                                    root_score_est);
      AssignBiasCacheEntry(global_state.bias_cache, game, leaf);
      *ib_hit = true;
      return {path, std::nullopt};
    }
  }

  // Signal NN when all workers have finished descending (descent_remaining==0).
  // Track pending NN fetches separately so Barrier 1 knows when all evals are
  // complete (pending==0). EXP round-free: no rounds (queue-mode NNInterface).
  if (!global_state.round_free) {
    absl::MutexLock l(&global_state.mu);
    InstRecordSubmit(global_state);
    global_state.descent_remaining--;
    if (needs_nn_eval) global_state.pending++;
    // INST pipelined search: the group's last submit hands the descent token
    // to the other group.
    if (global_state.descent_remaining == 0 && global_state.pipe != nullptr) {
      global_state.pipe->Pass(1 - global_state.group);
    }
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
    node->n_in_flight_group[0].fetch_sub(1, std::memory_order_acq_rel);
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
                BiasCache* bias_cache, const PuctParams& puct_params,
                const float leaf_q_outcome, const int group,
                const bool pipelined) {
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

  UpdateVCategorial(node, leaf_q_outcome);
  const int old_in_flight =
      node->n_in_flight.fetch_sub(1, std::memory_order_acq_rel);
  DCHECK(old_in_flight > 0);
  // INST pipelined search: the last arriver is per group (the other group may
  // have paths in flight through this node at any time).
  const int old_group_in_flight =
      node->n_in_flight_group[group].fetch_sub(1, std::memory_order_acq_rel);
  DCHECK(old_group_in_flight > 0);
  if ((pipelined ? old_group_in_flight : old_in_flight) == 1) {
    // last thread to arrive. responsible for update.
    // INST pipelined search: recompute under node->mu so the two groups never
    // recompute the same node concurrently; stats_seq brackets the recompute
    // for the P3_SEQCHECK overlap count.
    std::optional<absl::MutexLock> recompute_lock;
    if (pipelined) {
      recompute_lock.emplace(&node->mu);
      node->stats_seq.fetch_add(1, std::memory_order_acq_rel);
    }
    // INST pipelined search: the other group may already have backed up a
    // visit through this leaf's children (it descended through the leaf after
    // its evaluation, see Descend); then the leaf values would drop them.
    const bool leaf_has_children = [&]() {
      if (!pipelined || !is_leaf) return false;
      for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
        if (node->child_visits[a] > 0) return true;
      }
      return false;
    }();
    if (is_leaf && !leaf_has_children) {
      node->w = node->init_util_est;
      node->w_outcome = node->init_outcome_est;
      node->v = node->init_util_est;
      node->v_outcome = node->init_outcome_est;
      node->v_err = node->init_err_est;
    } else {
      const float obs_bias = FetchObsBias(bias_cache, node);
      RecomputeNodeStats(node, obs_bias);
    }
    RecomputeVCategoricalStats(node, puct_params.v_cat_var_scale_prior_visits);
    node->is_pending_update.store(false, std::memory_order_release);
    if (pipelined) node->stats_seq.fetch_add(1, std::memory_order_release);
  }
}

void Backup(int worker_id, SearchPath& search_path, BiasCache* bias_cache,
            const PuctParams puct_params, const int group,
            const bool pipelined) {
  // Can ignore the last node in the path because it is either a leaf or
  // terminal, from the perspective of this path.
  const float leaf_q_outcome = std::get<0>(search_path.back())->v_outcome;
  const auto leaf_parity = (search_path.size() - 1) % 2;
  for (int i = search_path.size() - 1; i >= 0; --i) {
    auto& [node, action, _] = search_path[i];
    const auto parity = i % 2;
    const auto leaf_sign = parity == leaf_parity ? 1 : -1;
    BackupStep(worker_id, node, action, i == (int)search_path.size() - 1,
               bias_cache, puct_params, leaf_sign * leaf_q_outcome, group,
               pipelined);
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
                const ScoreUtilityParams score_util_params,
                const PuctParams puct_params) {
  LeafEvaluator leaf_evaluator(slot, worker_id, score_util_params);
  const auto should_stop = [&global_state]() {
    if (global_state.should_stop_this_round) {
      return true;
    }
    // INST pipelined search: the visit budget is folded into
    // should_stop_this_round by the round reset (shared visit counter).
    if (global_state.pipe != nullptr) {
      return false;
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
    bool this_parity = false;
    if (!kInstGenBarrier) {
      absl::MutexLock l(&global_state.mu);
      this_parity = global_state.round_parity;
    }

    // INST pipelined search: wait for this group's turn to descend.
    if (global_state.pipe != nullptr) {
      const int64_t tk0 = InstNow();
      global_state.pipe->WaitTurn(global_state.group);
      tl_token_wait += InstNow() - tk0;
    }

    collision_policy.Reset();
    Game search_game = game;
    // INST root picks: only the round's first descent uses the precomputed
    // root move; retries score the root under root->mu as before.
    const bool use_root_pick = global_state.root_picks_valid;
    // EXP pipelined immediate backup: cache-hit descents back up here and
    // descend again; the first descent that is not an immediate backup joins
    // the round.
    const auto ib_stop = [&global_state]() {
      return global_state.should_stop.load(std::memory_order_relaxed) ||
             global_state.pipe->total_num_visits.load(
                 std::memory_order_relaxed) >= global_state.visit_budget;
    };
    const bool ib_on = kPipeIb && global_state.pipe != nullptr;
    int ib_count = 0;
    std::pair<SearchPath, std::optional<CollisionResult>> first_descent;
    while (true) {
      const bool allow_ib = ib_on && ib_count < kPipeIbMax && !ib_stop();
      bool ib_hit = false;
      first_descent = Descend(
          worker_id, prob, global_state, slot, &leaf_evaluator, descent_policy,
          collision_policy, collision_detector, search_game, node_table, root,
          color_to_move, root->init_score_est, {}, /*block_on_leaf_eval=*/true,
          use_root_pick && ib_count == 0, allow_ib, &ib_hit);
      if (!ib_hit) break;
      const int64_t tbk0 = InstNow();
      Backup(worker_id, first_descent.first, global_state.bias_cache,
             puct_params, global_state.group, /*pipelined=*/true);
      tl_backup += InstNow() - tbk0;
      global_state.total_num_visits.fetch_add(1, std::memory_order_relaxed);
      global_state.pipe->total_num_visits.fetch_add(1,
                                                    std::memory_order_relaxed);
      global_state.ib_hits.fetch_add(1, std::memory_order_relaxed);
      ++ib_count;
      collision_policy.Reset();
      search_game = game;
    }
    if (ib_on && ib_count >= kPipeIbMax) {
      global_state.ib_cap_triggers.fetch_add(1, std::memory_order_relaxed);
    }
    auto& [search_path, collision_result] = first_descent;
    if (collision_result.has_value()) {
      global_state.total_num_collisions.fetch_add(1, std::memory_order_relaxed);
      while (collision_result.has_value()) {
        if (collision_result->action == CollisionResult::Action::kAbort) {
          {
            absl::MutexLock l(&global_state.mu);
            InstRecordSubmit(global_state);
            global_state.total_num_aborted.fetch_add(1,
                                                     std::memory_order_relaxed);
            global_state.descent_remaining--;
            if (global_state.descent_remaining == 0 &&
                global_state.pending > 0 && !global_state.did_signal) {
              slot.SignalReadyForInference();
              global_state.did_signal = true;
            }
            if (global_state.descent_remaining == 0 &&
                global_state.pipe != nullptr) {
              global_state.pipe->Pass(1 - global_state.group);
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
    if (kInstGenBarrier) {
      // INST: every worker reaches this point exactly once per round, and only
      // after its own submit (descent_remaining--) and, if it queued an eval,
      // its own fetch (pending--). So "all workers arrived" implies
      // descent_remaining == 0 && pending == 0. Those counters are written
      // under mu before arrival and next written by the Barrier 2 reset,
      // which needs this thread's later arrival, so the reads below are
      // ordered by the barrier.
      global_state.gen_b1.ArriveAndWait([&global_state] {
        global_state.inst_b1_release.store(InstNow(),
                                           std::memory_order_relaxed);
      });
      CHECK_EQ(global_state.descent_remaining, 0);
      CHECK_EQ(global_state.pending, 0);
      const int64_t tb1 = InstNow();
      tl_b1_wait +=
          tb1 - global_state.inst_b1_release.load(std::memory_order_relaxed);
      InstAtomicMax(global_state.inst_b1_max, tb1);
    } else {
      absl::MutexLock l(&global_state.mu);
      global_state.mu.Await(absl::Condition(
          +[](GlobalSearchState* s) {
            return s->descent_remaining == 0 && s->pending == 0;
          },
          &global_state));
      // INST: wake latency after the barrier condition became true.
      const int64_t tb1 = InstNow();
      const int64_t cond_t =
          std::max(global_state.inst.fetch_max, global_state.inst.submit_max);
      tl_b1_wait += tb1 - cond_t;
      InstAtomicMax(global_state.inst_b1_max, tb1);
    }

    // Backprop now ready.
    if (!collision_result.has_value()) {
      // We must keep the invariant that all collisions retry until abort before
      // reaching here.
      const int64_t tbk0 = InstNow();
      Backup(worker_id, search_path, global_state.bias_cache, puct_params,
             global_state.group, global_state.pipe != nullptr);
      tl_backup += InstNow() - tbk0;
    }

    // Count this descent as a visit only if we successfully reached a leaf.
    if (!collision_result.has_value()) {
      global_state.total_num_visits.fetch_add(1, std::memory_order_relaxed);
      if (global_state.pipe != nullptr) {
        global_state.pipe->total_num_visits.fetch_add(
            1, std::memory_order_relaxed);
      }
    }

    // Barrier 2: wait for all workers to finish backprop. The last worker
    // resets per-round state and flips round_parity to wake all waiters.
    // Descent now ready. Reset the end-of-round barrier counter before
    // flipping parity so all threads see it zeroed once they exit the wait.
    InstAtomicMax(global_state.inst_b2_max, InstNow());
    if (kInstGenBarrier) {
      // INST: the last arriver resets round state (under mu, as the mutex path
      // does) before releasing; waiters read it after acquiring the release.
      // should_stop_this_round and total_num_visits are therefore identical
      // for every worker at the loop check, so all workers stop on the same
      // round and no barrier is left waiting.
      global_state.gen_b2.ArriveAndWait([&] {
        absl::MutexLock l(&global_state.mu);
        InstRoundEnd(global_state);
        UnsafeGlobalStateRoundReset(global_state);
        InstComputeRootPicks(global_state, descent_policy, game, root,
                             color_to_move);
      });
      tl_b2_wait += InstNow() - global_state.inst.round_start;
    } else {
      absl::MutexLock l(&global_state.mu);
      if (--global_state.round_remaining == 0) {
        // Last worker through: reset for next round.
        InstRoundEnd(global_state);
        UnsafeGlobalStateRoundReset(global_state);
        InstComputeRootPicks(global_state, descent_policy, game, root,
                             color_to_move);
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
      // INST: wake latency after the round reset.
      tl_b2_wait += InstNow() - global_state.inst.round_start;
    }
  }
  // INST pipelined search: this group no longer blocks the other.
  if (global_state.pipe != nullptr) {
    global_state.pipe->Exit(global_state.group);
  }
  // INST: flush this thread's timing accumulators once, at exit.
  absl::MutexLock l(&global_state.mu);
  InstFlushThread(global_state);
}

// EXP round-free backup step: every node on the path is updated and
// recomputed by every backup, under node->mu (KataGo recomputes each node on
// the path per playout under its stats lock). Workers enter and leave nodes at
// any time, so the round "last arriver" rule does not apply. Children may be
// mid-update by other workers (unlocked reads of their stats, as in
// pipelined search).
void BackupStepRoundFree(TreeNode* node, game::Loc action, bool is_leaf,
                         BiasCache* bias_cache, const PuctParams& puct_params,
                         const float leaf_q_outcome) {
  absl::MutexLock l(&node->mu);
  node->n += 1;
  if (!is_leaf) {
    node->child_visits[game::AsIndex(action, BOARD_LEN)] += 1;
  }
  UpdateVCategorial(node, leaf_q_outcome);
  node->n_in_flight.fetch_sub(1, std::memory_order_acq_rel);
  node->n_in_flight_group[0].fetch_sub(1, std::memory_order_acq_rel);
  // The leaf of this path may already have children backed up by other
  // workers (they descended through it after its evaluation).
  bool leaf_has_children = false;
  if (is_leaf) {
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      if (node->child_visits[a] > 0) {
        leaf_has_children = true;
        break;
      }
    }
  }
  if (is_leaf && !leaf_has_children) {
    node->w = node->init_util_est;
    node->w_outcome = node->init_outcome_est;
    node->v = node->init_util_est;
    node->v_outcome = node->init_outcome_est;
    node->v_err = node->init_err_est;
  } else {
    RecomputeNodeStats(node, FetchObsBias(bias_cache, node));
  }
  RecomputeVCategoricalStats(node, puct_params.v_cat_var_scale_prior_visits);
  node->is_pending_update.store(false, std::memory_order_release);
}

void BackupRoundFree(SearchPath& search_path, BiasCache* bias_cache,
                     const PuctParams puct_params) {
  const float leaf_q_outcome = std::get<0>(search_path.back())->v_outcome;
  const auto leaf_parity = (search_path.size() - 1) % 2;
  for (int i = search_path.size() - 1; i >= 0; --i) {
    auto& [node, action, _] = search_path[i];
    const auto leaf_sign = i % 2 == leaf_parity ? 1 : -1;
    BackupStepRoundFree(node, action, i == (int)search_path.size() - 1,
                        bias_cache, puct_params, leaf_sign * leaf_q_outcome);
  }
}

// EXP round-free worker (P3_ROUNDFREE=1): descend (node locks as usual), wait
// for this worker's own NN result, back up, repeat. An aborted descent retries
// at once; after kRfBackoffAfter consecutive aborts the worker sleeps briefly
// (the colliding leaves are waiting on the GPU or on another worker's backup).
template <typename DescentPolicy, typename CollisionPolicy,
          typename CollisionDetector>
void SearchTaskRoundFree(const int worker_id, core::Probability& prob,
                         GlobalSearchState& global_state,
                         NNInterface::Slot slot, DescentPolicy& descent_policy,
                         CollisionPolicy& collision_policy,
                         const CollisionDetector& collision_detector,
                         Game& game, NodeTable* node_table,
                         TreeNode* const root, game::Color color_to_move,
                         const ScoreUtilityParams score_util_params,
                         const PuctParams puct_params) {
  static constexpr int kRfBackoffAfter = 4;
  static constexpr int kRfBackoffUs = 20;
  LeafEvaluator leaf_evaluator(slot, worker_id, score_util_params);
  int consecutive_aborts = 0;
  while (!global_state.should_stop.load(std::memory_order_relaxed) &&
         global_state.total_num_visits.load(std::memory_order_relaxed) <
             global_state.visit_budget) {
    collision_policy.Reset();
    Game search_game = game;
    auto [search_path, collision_result] = Descend(
        worker_id, prob, global_state, slot, &leaf_evaluator, descent_policy,
        collision_policy, collision_detector, search_game, node_table, root,
        color_to_move, root->init_score_est, {}, /*block_on_leaf_eval=*/true);
    if (collision_result.has_value()) {
      global_state.total_num_collisions.fetch_add(1, std::memory_order_relaxed);
      global_state.total_num_aborted.fetch_add(1, std::memory_order_relaxed);
      if (++consecutive_aborts >= kRfBackoffAfter) {
        tl_b1_wait += kRfBackoffUs * 1000;  // INST: backoff time (approx).
        std::this_thread::sleep_for(std::chrono::microseconds(kRfBackoffUs));
      }
      continue;
    }
    consecutive_aborts = 0;
    const int64_t tbk0 = InstNow();
    BackupRoundFree(search_path, global_state.bias_cache, puct_params);
    tl_backup += InstNow() - tbk0;
    global_state.total_num_visits.fetch_add(1, std::memory_order_relaxed);
  }
  absl::MutexLock l(&global_state.mu);
  InstFlushThread(global_state);
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
                      ScoreUtilityParams score_util_params,
                      PuctParams puct_params) {
  // INST root picks for the first round (no worker exists yet).
  {
    absl::MutexLock l(&global_state.mu);
    CHECK_EQ(global_state.num_workers, num_threads);
    InstComputeRootPicks(global_state, descent_policy, game, root,
                         color_to_move);
  }
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (int worker_id = 0; worker_id < num_threads; ++worker_id) {
    workers.emplace_back([&, worker_id, cp = collision_policy_proto]() mutable {
      core::Probability prob;
      if (global_state.round_free) {
        SearchTaskRoundFree(worker_id, prob, global_state, slot, descent_policy,
                            cp, collision_detector, game, node_table, root,
                            color_to_move, score_util_params, puct_params);
        return;
      }
      SearchTask(worker_id, prob, global_state, slot, descent_policy, cp,
                 collision_detector, game, node_table, root, color_to_move,
                 score_util_params, puct_params);
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
    UnsafeGlobalStateRoundReset(global_state);
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
                         params.score_util_params, params.puct_params);
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
                         params.score_util_params, params.puct_params);
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
  GlobalSearchState& global_search_state = global_search_state_;
  // INST pipelined search: two groups of num_threads / 2 workers, each with
  // its own GlobalSearchState and NNInterface slot, taking turns descending.
  const bool pipelined = slot2_.has_value() &&
                         params.mode == Search::Mode::kConcurrent &&
                         params.num_threads > 1;
  CHECK(!(pipelined && slot_.nn()->queue_mode()))
      << "round-free (queue-mode) search cannot be pipelined";
  if (pipelined) {
    CHECK(params.num_threads % 2 == 0 && params.num_threads >= 2)
        << "pipelined search needs an even num_threads >= 2, got "
        << params.num_threads;
    // Root picks assume no other worker touches the tree at the round end.
    CHECK(kInstRootPicks == 0) << "P3_ROOT_PICKS is not supported pipelined";
  }
  const int num_groups = pipelined ? 2 : 1;
  const int group_threads =
      pipelined ? params.num_threads / 2 : params.num_threads;
  GlobalSearchState* const states[2] = {&global_search_state_,
                                        &global_search_state2_};
  const NNInterface::Slot slots[2] = {slot_, pipelined ? *slot2_ : slot_};
  for (int g = 0; g < num_groups; ++g) {
    GlobalSearchState& gs = *states[g];
    gs.did_signal = false;
    gs.round_parity = false;
    gs.num_workers = group_threads;
    gs.visit_budget = params.total_visit_budget;
    gs.descent_remaining = group_threads;
    gs.round_remaining = group_threads;
    // INST: arm gen barriers before workers spawn (thread creation orders
    // these writes before every worker's first arrival).
    gs.gen_b1.Init(group_threads);
    gs.gen_b2.Init(group_threads);
    gs.max_pending_each_level = 100000;
    gs.bias_cache = bias_cache_;
    gs.group = g;
    gs.pipe = pipelined ? &pipe_ : nullptr;
    gs.inst_nn = slots[g].nn();
    gs.inst_wave = slots[g].wave();
    // EXP round-free: a queue-mode interface runs workers without rounds.
    gs.round_free = slots[g].nn()->queue_mode();
  }
  if (pipelined) pipe_.Reset();

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

  // INST: drop inference intervals logged before this search's workers start.
  for (int g = 0; g < num_groups; ++g) slots[g].nn()->InstTakeIntervals();
  // INST: distinct NNInterfaces of this search; reset their cache counters.
  std::vector<NNInterface*> nnc_ifaces;
  for (int g = 0; g < num_groups; ++g) {
    NNInterface* nn = slots[g].nn();
    if (std::find(nnc_ifaces.begin(), nnc_ifaces.end(), nn) ==
        nnc_ifaces.end()) {
      nnc_ifaces.push_back(nn);
      (void)nn->TakeCacheCounters();
    }
  }
  const int64_t inst_search_t0 = InstNow();
  // INST round-free: queue-mode batch counters at search start.
  NNInterface* const q_nn = slot_.nn();
  const int64_t q_b0 = q_nn->inst_q_batches.load(),
                q_r0 = q_nn->inst_q_rows.load(),
                q_in0 = q_nn->inst_q_in_ns.load(),
                q_out0 = q_nn->inst_q_out_ns.load();
  for (int g = 0; g < num_groups; ++g) {
    absl::MutexLock l(&states[g]->mu);
    states[g]->inst.round_start = inst_search_t0;
  }
  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  std::thread timer([&, f = std::move(done_future)]() mutable {
    if (params.total_visit_time_ms <= 0) {
      return;
    }

    if (f.wait_for(std::chrono::milliseconds(params.total_visit_time_ms)) ==
        std::future_status::timeout) {
      for (int g = 0; g < num_groups; ++g) {
        states[g]->should_stop.store(true, std::memory_order_relaxed);
      }
    }
  });

  const auto run_group = [&](GlobalSearchState& gs, NNInterface::Slot slot,
                             const Search::Params& p) {
    switch (p.collision_policy_kind) {
      case CollisionPolicyKind::kAbort:
        RunWithDetector(gs, slot, game, node_table, root, color_to_move,
                        AbortCollisionPolicy{}, p);
        break;
      case CollisionPolicyKind::kRetry:
        RunWithDetector(gs, slot, game, node_table, root, color_to_move,
                        RetryCollisionPolicy(p.max_collision_retries), p);
        break;
      case CollisionPolicyKind::kSmartRetry:
        RunWithDetector(gs, slot, game, node_table, root, color_to_move,
                        SmartRetryCollisionPolicy(p.max_collision_retries), p);
        break;
    }
  };
  if (pipelined) {
    Search::Params group_params = params;
    group_params.num_threads = group_threads;
    std::thread group1(
        [&]() { run_group(*states[1], slots[1], group_params); });
    run_group(*states[0], slots[0], group_params);
    group1.join();
  } else {
    run_group(global_search_state, slot_, params);
  }
  done_promise.set_value();
  timer.join();
  const int64_t inst_search_t1 = InstNow();
  const auto search_end = std::chrono::steady_clock::now();
  const auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           search_end - search_begin)
                           .count();
  const auto best_move = best_lcb_move(root);
  int64_t visit_count = 0, abort_count = 0, collision_count = 0;
  for (int g = 0; g < num_groups; ++g) {
    visit_count += states[g]->total_num_visits.load(std::memory_order_relaxed);
    abort_count += states[g]->total_num_aborted.load(std::memory_order_relaxed);
    collision_count +=
        states[g]->total_num_collisions.load(std::memory_order_relaxed);
  }
  {
    // INST: per-round phase means (ms, relative to round start) and
    // per-descent / per-thread-round means (us). Pipelined: both groups'
    // rounds are pooled.
    GlobalSearchState::Inst I;
    for (int g = 0; g < num_groups; ++g) {
      absl::MutexLock l(&states[g]->mu);
      I.Add(states[g]->inst);
    }
    const double R = std::max<int64_t>(I.rounds, 1),
                 NR = std::max<int64_t>(I.nn_rounds, 1),
                 D = std::max<int64_t>(I.descents, 1),
                 TR = std::max<int64_t>(I.rounds * group_threads, 1);
    fprintf(stderr,
            "PHASES rounds=%ld nn_rounds=%ld descents=%ld threads=%d "
            "submit_mean_ms=%.3f submit_max_ms=%.3f infer_start_ms=%.3f "
            "infer_end_ms=%.3f fetch_min_ms=%.3f fetch_max_ms=%.3f "
            "b1_max_ms=%.3f b2_max_ms=%.3f end_ms=%.3f | per_descent_us "
            "root_wait=%.2f root_hold=%.2f other_wait=%.2f other_hold=%.2f "
            "table=%.2f | per_thread_round_us queue=%.2f fetch=%.2f "
            "backup=%.2f b1_wake=%.2f b2_wake=%.2f | per_descent_us "
            "root_pre=%.2f root_puct=%.2f | picks_us_per_round=%.2f "
            "pick_rounds=%ld picks_seq_checked=%ld\n",
            (long)I.rounds, (long)I.nn_rounds, (long)I.descents,
            params.num_threads, I.a_submit_mean / R / 1e6,
            I.a_submit_max / R / 1e6, I.a_infer_start / NR / 1e6,
            I.a_infer_end / NR / 1e6, I.a_fetch_min / NR / 1e6,
            I.a_fetch_max / NR / 1e6, I.a_b1_max / R / 1e6,
            I.a_b2_max / R / 1e6, I.a_end / R / 1e6, I.root_wait / D / 1e3,
            I.root_hold / D / 1e3, I.other_wait / D / 1e3,
            I.other_hold / D / 1e3, I.table_wait / D / 1e3, I.queue / TR / 1e3,
            I.fetch / TR / 1e3, I.backup / TR / 1e3, I.b1_wait / TR / 1e3,
            I.b2_wait / TR / 1e3, I.root_pre / D / 1e3, I.root_puct / D / 1e3,
            I.picks_ns / std::max<double>(I.pick_rounds, 1) / 1e3,
            (long)I.pick_rounds, (long)I.picks_seq_checked);
    // INST smart retry (all zero for other collision policies).
    {
      const double F = std::max<int64_t>(I.sr_forks, 1);
      fprintf(stderr,
              "SR forks=%ld fork_root=%ld plain=%ld collisions=%ld "
              "aborted=%ld mean_fork_depth=%.3f mean_path_len=%.3f "
              "mean_rank=%.3f mean_gap=%.5f\n",
              (long)I.sr_forks, (long)I.sr_fork_root, (long)I.sr_plain,
              (long)collision_count, (long)abort_count, I.sr_depth / F,
              I.sr_path_len / F, I.sr_rank / F, I.sr_gap_e6 / F / 1e6);
    }

    // INST: inference intervals of every NNInterface used by this search,
    // clipped to the worker window. union = time any engine was running;
    // overlap = sum - union (time two engines ran at once).
    std::vector<std::pair<int64_t, int64_t>> iv;
    for (int g = 0; g < num_groups; ++g) {
      for (auto [a, b] : slots[g].nn()->InstTakeIntervals()) {
        a = std::max(a, inst_search_t0);
        b = std::min(b, inst_search_t1);
        if (b > a) iv.emplace_back(a, b);
      }
    }
    std::sort(iv.begin(), iv.end());
    int64_t gpu_sum = 0, gpu_union = 0, cur_a = 0, cur_b = -1;
    for (const auto& [a, b] : iv) {
      gpu_sum += b - a;
      if (a > cur_b) {
        if (cur_b > cur_a) gpu_union += cur_b - cur_a;
        cur_a = a;
        cur_b = b;
      } else {
        cur_b = std::max(cur_b, b);
      }
    }
    if (cur_b > cur_a) gpu_union += cur_b - cur_a;
    const double wall = std::max<int64_t>(inst_search_t1 - inst_search_t0, 1);
    fprintf(stderr,
            "PIPE pipe=%d groups=%d group_threads=%d rounds_g0=%ld "
            "rounds_g1=%ld token_wait_us_per_thread_round=%.2f "
            "seq_reads=%ld seq_overlap=%ld infers=%zu gpu_union_ms=%.3f "
            "gpu_sum_ms=%.3f gpu_overlap_ms=%.3f search_ms=%.3f "
            "gpu_busy=%.4f\n",
            pipelined ? 1 : 0, num_groups, group_threads,
            (long)states[0]->inst.rounds,
            (long)(pipelined ? states[1]->inst.rounds : 0),
            I.token_wait / TR / 1e3, (long)I.seq_reads, (long)I.seq_overlap,
            iv.size(), gpu_union / 1e6, gpu_sum / 1e6,
            (gpu_sum - gpu_union) / 1e6, wall / 1e6, gpu_union / wall);
    if (q_nn->queue_mode()) {
      const double nb =
          std::max<int64_t>(q_nn->inst_q_batches.load() - q_b0, 1);
      fprintf(stderr,
              "RF batches=%ld rows_per_batch=%.2f copy_in_us_per_batch=%.1f "
              "copy_out_us_per_batch=%.1f aborted=%ld backoff_ms_sum=%.2f "
              "backup_us_per_visit=%.2f\n",
              (long)(q_nn->inst_q_batches.load() - q_b0),
              (q_nn->inst_q_rows.load() - q_r0) / nb,
              (q_nn->inst_q_in_ns.load() - q_in0) / nb / 1e3,
              (q_nn->inst_q_out_ns.load() - q_out0) / nb / 1e3,
              (long)abort_count, I.b1_wait / 1e6,
              I.backup / std::max<double>(visit_count, 1) / 1e3);
    }
  }
  {
    // INST: NN cache lookups / hits during this search (all interfaces).
    NNInterface::CacheCounters cc;
    for (NNInterface* nn : nnc_ifaces) {
      const NNInterface::CacheCounters c = nn->TakeCacheCounters();
      cc.lookups += c.lookups;
      cc.hits += c.hits;
      cc.lock_contended += c.lock_contended;
      cc.lock_spins += c.lock_spins;
    }
    int64_t ib_hits = 0, ib_caps = 0;
    for (int g = 0; g < num_groups; ++g) {
      ib_hits += states[g]->ib_hits.load(std::memory_order_relaxed);
      ib_caps += states[g]->ib_cap_triggers.load(std::memory_order_relaxed);
    }
    fprintf(stderr,
            "NNC lookups=%ld hits=%ld hit_rate=%.4f spin_contended=%ld "
            "spin_spins=%ld ib_hits=%ld ib_cap_triggers=%ld\n",
            (long)cc.lookups, (long)cc.hits,
            cc.hits / std::max<double>(cc.lookups, 1), (long)cc.lock_contended,
            (long)cc.lock_spins, (long)ib_hits, (long)ib_caps);
  }
  // INST (P3_ROOTDIST=1): root value and visit distribution, for accuracy
  // comparisons against a reference search.
  static const bool kInstRootDist = [] {
    const char* v = std::getenv("P3_ROOTDIST");
    return v != nullptr && std::string(v) == "1";
  }();
  if (kInstRootDist) {
    std::string dist;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      if (root->child_visits[a] > 0) {
        dist += " " + std::to_string(a) + ":" +
                std::to_string(root->child_visits[a]);
      }
    }
    fprintf(stderr,
            "ROOTDIST n=%d v=%.6f v_outcome=%.6f score=%.4f best=%d "
            "visits=%ld dist=%s\n",
            root->n, root->v, root->v_outcome, root->score,
            game::AsIndex(best_move, BOARD_LEN), (long)visit_count,
            dist.c_str());
  }
  global_search_state_.reset();
  global_search_state2_.reset();
  return Search::Result{best_move, size_t(visit_count), size_t(abort_count),
                        size_t(collision_count), size_t(time_ms)};
}
}  // namespace mcts

#undef SPIN_WHILE
