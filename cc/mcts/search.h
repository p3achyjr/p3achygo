#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/heap.h"
#include "cc/game/loc.h"
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/leaf_evaluator.h"
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
  kBuUct = 1,
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

// INST: generation-counter barrier (selected with env P3_BARRIER=gen).
//
// Each thread reads `gen_` BEFORE decrementing `remaining_`, so the value it
// waits on is always the pre-release generation (the release needs this
// thread's own decrement). The last arriver runs `on_last`, re-arms
// `remaining_`, then publishes with a release increment of `gen_` and wakes
// everyone. Visibility: all arrivals are acq_rel RMWs on `remaining_` (one
// release sequence), so every thread's pre-arrival writes happen-before the
// last arriver's `on_last`; waiters acquire `gen_`, so `on_last` and all
// pre-arrival writes happen-before every waiter's return. `remaining_` is
// re-armed before the release, and no thread can arrive at the next use until
// it has observed the release, so the next use always starts at n.
class GenBarrier {
 public:
  void Init(int n) {
    n_ = n;
    remaining_.store(n, std::memory_order_relaxed);
  }

  template <typename F>
  void ArriveAndWait(F&& on_last) {
    const uint32_t g = gen_.load(std::memory_order_acquire);
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      on_last();
      remaining_.store(n_, std::memory_order_relaxed);
      gen_.fetch_add(1, std::memory_order_release);
      gen_.notify_all();
      return;
    }
    // atomic::wait may return spuriously; loop until the generation changes.
    while (gen_.load(std::memory_order_acquire) == g) {
      gen_.wait(g, std::memory_order_acquire);
    }
  }

 private:
  int n_ = 0;
  std::atomic<int> remaining_{0};
  std::atomic<uint32_t> gen_{0};
};

// INST pipelined search: state shared by the two thread groups. Each group
// owns a GlobalSearchState (its own round, barriers and NNInterface); the
// groups take turns descending: a group waits for the token at the start of
// each round and hands it to the other group once all its workers have
// submitted (descent_remaining hits 0). A group that has exited no longer
// blocks the other.
struct PipeShared {
  std::atomic<int> token{0};
  std::array<std::atomic<bool>, 2> exited{};
  std::atomic<uint32_t> gen{0};
  // Visits of both groups, for the visit budget (checked once per round by
  // each group's round reset, so all workers of a group stop together).
  std::atomic<int> total_num_visits{0};

  void Reset() {
    token.store(0, std::memory_order_relaxed);
    exited[0].store(false, std::memory_order_relaxed);
    exited[1].store(false, std::memory_order_relaxed);
    total_num_visits.store(0, std::memory_order_relaxed);
  }
  void Notify() {
    gen.fetch_add(1, std::memory_order_release);
    gen.notify_all();
  }
  void Pass(int to) {
    token.store(to, std::memory_order_release);
    Notify();
  }
  void Exit(int g) {
    exited[g].store(true, std::memory_order_release);
    Notify();
  }
  void WaitTurn(int g) {
    while (true) {
      const uint32_t cur = gen.load(std::memory_order_acquire);
      if (token.load(std::memory_order_acquire) == g ||
          exited[1 - g].load(std::memory_order_acquire)) {
        return;
      }
      gen.wait(cur, std::memory_order_acquire);
    }
  }
};

struct GlobalSearchState {
  absl::Mutex mu;

  // INST pipelined search: group index (0 when not pipelined) and the shared
  // pipeline state (nullptr when not pipelined).
  int group = 0;
  PipeShared* pipe = nullptr;
  // INST: NNInterface used by this state's workers (per-interface inference
  // timestamps for PHASES).
  nn::NNInterface* inst_nn = nullptr;
  // INST: this state's wave in inst_nn (EXP shared interface; else 0).
  int inst_wave = 0;
  // EXP round-free search (P3_ROUNDFREE=1 with a queue-mode NNInterface):
  // no rounds or barriers; Descend skips the round submit bookkeeping.
  bool round_free = false;

  // INST: gen-counter barriers (P3_BARRIER=gen), armed in Search::Run.
  GenBarrier gen_b1, gen_b2;
  // INST: per-round timestamps written without mu (CAS max) or by the last
  // arriver of a gen barrier.
  std::atomic<int64_t> inst_b1_max{0}, inst_b2_max{0}, inst_b1_release{0};

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

  // EXP pipelined immediate backup (P3_PIPE_IB=1): cache-hit leaves backed up
  // at descent time, and slots whose hit count reached P3_PIPE_IB_MAX.
  std::atomic<int64_t> ib_hits = 0;
  std::atomic<int64_t> ib_cap_triggers = 0;

  // Optional bias cache
  BiasCache* bias_cache = nullptr;

  // INST lc0-style root picks (P3_ROOT_PICKS). Written by the barrier-2 last
  // arriver (or before workers spawn); read by workers after the barrier.
  bool root_picks_valid = false;
  std::vector<std::pair<game::Loc, TopActions>> root_picks;

  // INST: per-round phase timestamps (steady_clock ns) and accumulators.
  // All guarded by mu.
  struct Inst {
    int64_t round_start = 0, submit_max = 0, submit_sum = 0, submit_n = 0;
    int64_t fetch_min = INT64_MAX, fetch_max = 0;
    // accumulators (ns, summed over rounds)
    int64_t rounds = 0, nn_rounds = 0, a_submit_mean = 0, a_submit_max = 0,
            a_infer_start = 0, a_infer_end = 0, a_fetch_min = 0,
            a_fetch_max = 0, a_b1_max = 0, a_b2_max = 0, a_end = 0;
    // per-thread accumulators flushed each round (ns, summed over threads)
    int64_t descents = 0, root_wait = 0, root_hold = 0, other_wait = 0,
            other_hold = 0, table_wait = 0, queue = 0, fetch = 0, backup = 0,
            b1_wait = 0, b2_wait = 0, root_pre = 0, root_puct = 0;
    // root picks: ns spent computing picks, rounds with picks, picks checked
    // against the original locked path.
    int64_t picks_ns = 0, pick_rounds = 0, picks_seq_checked = 0;
    // pipelined search: token wait (ns, summed over threads); seqlock check
    // (P3_SEQCHECK=1): PUCT child reads and reads that overlapped a recompute.
    int64_t token_wait = 0, seq_reads = 0, seq_overlap = 0;
    // smart retry: forks, forks at the root, plain retries (path too short),
    // and sums of fork depth, colliding path length, forced-move rank and
    // fork score gap (x1e6).
    int64_t sr_forks = 0, sr_fork_root = 0, sr_plain = 0, sr_depth = 0,
            sr_path_len = 0, sr_rank = 0, sr_gap_e6 = 0;

    void Add(const Inst& o) {
      rounds += o.rounds;
      nn_rounds += o.nn_rounds;
      a_submit_mean += o.a_submit_mean;
      a_submit_max += o.a_submit_max;
      a_infer_start += o.a_infer_start;
      a_infer_end += o.a_infer_end;
      a_fetch_min += o.a_fetch_min;
      a_fetch_max += o.a_fetch_max;
      a_b1_max += o.a_b1_max;
      a_b2_max += o.a_b2_max;
      a_end += o.a_end;
      descents += o.descents;
      root_wait += o.root_wait;
      root_hold += o.root_hold;
      other_wait += o.other_wait;
      other_hold += o.other_hold;
      table_wait += o.table_wait;
      queue += o.queue;
      fetch += o.fetch;
      backup += o.backup;
      b1_wait += o.b1_wait;
      b2_wait += o.b2_wait;
      root_pre += o.root_pre;
      root_puct += o.root_puct;
      picks_ns += o.picks_ns;
      pick_rounds += o.pick_rounds;
      picks_seq_checked += o.picks_seq_checked;
      token_wait += o.token_wait;
      seq_reads += o.seq_reads;
      seq_overlap += o.seq_overlap;
      sr_forks += o.sr_forks;
      sr_fork_root += o.sr_fork_root;
      sr_plain += o.sr_plain;
      sr_depth += o.sr_depth;
      sr_path_len += o.sr_path_len;
      sr_rank += o.sr_rank;
      sr_gap_e6 += o.sr_gap_e6;
    }
  } inst;

  void reset() {
    inst = Inst{};
    inst_b1_max.store(0, std::memory_order_relaxed);
    inst_b2_max.store(0, std::memory_order_relaxed);
    inst_b1_release.store(0, std::memory_order_relaxed);
    gen_b1.Init(0);
    gen_b2.Init(0);
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
    ib_hits.store(0, std::memory_order_relaxed);
    ib_cap_triggers.store(0, std::memory_order_relaxed);
    bias_cache = nullptr;
    root_picks_valid = false;
    group = 0;
    pipe = nullptr;
    inst_nn = nullptr;
    inst_wave = 0;
    round_free = false;
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
    Mode mode = Search::Mode::kConcurrent;
    ScoreUtilityParams score_util_params;
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
  void StopSearch() {
    global_search_state_.should_stop = true;
    global_search_state2_.should_stop = true;
  }
  // INST pipelined search: a second NNInterface slot. When set, Run() in
  // concurrent mode splits num_threads into two groups of num_threads / 2,
  // group 0 on `slot_`, group 1 on `slot2`.
  void SetPipelineSlot(nn::NNInterface::Slot slot2) { slot2_.emplace(slot2); }

 private:
  nn::NNInterface::Slot slot_;
  BiasCache* bias_cache_ = nullptr;
  GlobalSearchState global_search_state_{};
  std::optional<nn::NNInterface::Slot> slot2_;
  GlobalSearchState global_search_state2_{};
  PipeShared pipe_{};
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
                         game::Color color, bool is_root = false) {
    const TopActions top_actions =
        puct_scorer_.TopScores(node, game, color, is_root);
    return {game::AsLoc(top_actions[0].first), top_actions};
  }

  // INST split PUCT (see PuctScorer::Precompute).
  using Precomp = typename PuctScorer<QFn, NFn>::Precomp;
  static constexpr bool kSupportsSplit = true;
  inline void Precompute(const TreeNode* node, bool is_root,
                         Precomp* pre) const {
    puct_scorer_.Precompute(node, is_root, pre);
  }
  // node->mu must be held. check: also run ComputeScores and CHECK bitwise
  // equality.
  inline DescentStep RunSplit(const Precomp& pre, const TreeNode* node,
                              const game::Game& game, game::Color color,
                              bool check) const {
    const PuctScores pucts = puct_scorer_.Finalize(pre, node);
    if (check) {
      const PuctScores ref = puct_scorer_.ComputeScores(node, pre.is_root);
      for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
        CHECK(ref[a].first == pucts[a].first &&
              std::memcmp(&ref[a].second, &pucts[a].second, sizeof(float)) == 0)
            << "split PUCT mismatch a=" << a << " ref=" << ref[a].second
            << " split=" << pucts[a].second;
      }
    }
    const TopActions top_actions =
        puct_scorer_.TopScoresFrom(pucts, game, color);
    return {game::AsLoc(top_actions[0].first), top_actions};
  }

  // INST lc0-style root picks. Pick k is the TopScores result with the root
  // in-flight counts of picks 0..k-1 added to the current ones. Must run while
  // no other thread touches the root (barrier last arriver, or before spawn).
  inline void RootPicks(const TreeNode* root, const game::Game& game,
                        game::Color color, int num,
                        std::vector<DescentStep>* out) const {
    thread_local Precomp pre;
    puct_scorer_.Precompute(root, /*is_root=*/true, &pre);
    std::array<int, constants::kMaxMovesPerPosition> cif;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      TreeNode* child = root->child(a);
      cif[a] = child == nullptr
                   ? 0
                   : child->n_in_flight.load(std::memory_order_relaxed);
    }
    std::array<int8_t, constants::kMaxMovesPerPosition> legal;
    legal.fill(-1);
    out->resize(num);
    for (int k = 0; k < num; ++k) {
      const PuctScores pucts = puct_scorer_.FinalizeWith(pre, cif);
      const TopActions top =
          puct_scorer_.TopScoresFrom(pucts, game, color, &legal);
      (*out)[k] = {game::AsLoc(top[0].first), top};
      if (top[0].first >= 0) cif[top[0].first]++;
    }
  }

  // INST check for RootPicks. (1) Memoised legality gives the same TopActions
  // as the uncached path for every pick. (2) While every earlier pick hit an
  // existing child, bump those children's real n_in_flight and CHECK that the
  // original locked path (TopScores -> ComputeScores) returns the same
  // TopActions bitwise; undo the bumps at the end. Returns the number of picks
  // compared against the original path.
  inline int CheckRootPicks(TreeNode* root, const game::Game& game,
                            game::Color color,
                            const std::vector<DescentStep>& picks) const {
    thread_local Precomp pre;
    puct_scorer_.Precompute(root, /*is_root=*/true, &pre);
    std::array<int, constants::kMaxMovesPerPosition> cif;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      TreeNode* child = root->child(a);
      cif[a] = child == nullptr
                   ? 0
                   : child->n_in_flight.load(std::memory_order_relaxed);
    }
    const auto same = [](const TopActions& x, const TopActions& y) {
      for (int i = 0; i < 4; ++i) {
        if (x[i].first != y[i].first ||
            std::memcmp(&x[i].second, &y[i].second, sizeof(float)) != 0) {
          return false;
        }
      }
      return true;
    };
    std::vector<TreeNode*> bumped;
    bool prefix_exists = true;
    int n_seq = 0;
    for (int k = 0; k < static_cast<int>(picks.size()); ++k) {
      const TopActions uncached = puct_scorer_.TopScoresFrom(
          puct_scorer_.FinalizeWith(pre, cif), game, color);
      CHECK(same(uncached, picks[k].second))
          << "root pick memo mismatch k=" << k;
      if (prefix_exists) {
        const TopActions seq =
            puct_scorer_.TopScores(root, game, color, /*is_root=*/true);
        CHECK(same(seq, picks[k].second))
            << "root pick vs sequential mismatch k=" << k;
        ++n_seq;
      }
      const int a = picks[k].second[0].first;
      if (a < 0) continue;
      cif[a]++;
      TreeNode* child = root->child(a);
      if (child == nullptr) {
        prefix_exists = false;
      } else if (prefix_exists) {
        child->n_in_flight.fetch_add(1, std::memory_order_relaxed);
        bumped.push_back(child);
      }
    }
    for (TreeNode* child : bumped) {
      child->n_in_flight.fetch_sub(1, std::memory_order_relaxed);
    }
    return n_seq;
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
                         const TreeNode* node, const game::Game& game,
                         game::Color color, bool is_root = false) {
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

// INST smart retry: thread-local counters, flushed into
// GlobalSearchState::inst with the other INST counters.
struct SrInst {
  int64_t forks = 0, fork_root = 0, plain = 0, depth = 0, path_len = 0,
          rank = 0, gap_e6 = 0;
};
inline thread_local SrInst tl_sr;

/*
 * Retries by attempting to find the next-best path.
 */
class SmartRetryCollisionPolicy final {
 public:
  SmartRetryCollisionPolicy(const int max_num_retries)
      : max_num_retries_(max_num_retries) {};
  ~SmartRetryCollisionPolicy() = default;
  inline CollisionResult Handle(const GlobalSearchState& global_search_state,
                                const SearchPath& search_path) {
    if (num_retries_ >= max_num_retries_) {
      return {CollisionResult::Action::kAbort, std::nullopt};
    }
    ++num_retries_;
    if (search_path.size() <= 1) {
      ++tl_sr.plain;
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

    // fork at min_index.
    const auto& [node, move, top_actions] = search_path[min_index];
    // INST: forced-move rank is 2 at a new fork node, +1 per repeat fork at
    // the same node within this descent (top_actions shift left below).
    sr_rank_ = node == sr_last_node_ ? sr_rank_ + 1 : 2;
    sr_last_node_ = node;
    ++tl_sr.forks;
    tl_sr.fork_root += min_index == 0;
    tl_sr.depth += min_index;
    tl_sr.path_len += search_path.size();
    tl_sr.rank += sr_rank_;
    tl_sr.gap_e6 += static_cast<int64_t>(min_diff * 1e6);
    // if we retry again from here, the forked move is the best move.
    const auto new_move = game::AsLoc(top_actions[1].first);
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

  inline void Reset() {
    num_retries_ = 0;
    sr_last_node_ = nullptr;
    sr_rank_ = 0;
  }

 private:
  const int max_num_retries_;
  int num_retries_ = 0;
  const TreeNode* sr_last_node_ = nullptr;  // INST
  int sr_rank_ = 0;                         // INST
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
