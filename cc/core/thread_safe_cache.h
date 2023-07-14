#ifndef CORE_THREAD_SAFE_CACHE_H_
#define CORE_THREAD_SAFE_CACHE_H_

#include <deque>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

namespace core {

/*
 * Simple thread-safe cache implementation.
 *
 * Eviction Policy is LRU.
 *
 * Type `K` must be hashable via absl::Hash.
 */
template <typename K, typename V>
class ThreadSafeCache final {
 public:
  ThreadSafeCache(int max_size) : max_size_(max_size) {}
  ~ThreadSafeCache() = default;

  bool Contains(const K& key) { return cache_.contains(key); }

  std::optional<V> Get(const K& key) {
    absl::MutexLock l(&mu_);

    auto v_it = cache_.find(K);
    if (v_it == cache_.end()) {
      return std::nullopt;
    }

    return *v_it;
  }

  void Insert(const K& key, const V& val) {
    absl::MutexLock l(&mu_);

    cache_[key] = val;
    lru_queue_.push_back(key);

    if (cache_.size() > max_size_) {
      K lru_key = lru_queue_.front();
      cache_.erase(K);
      lru_queue_.pop_front();
    }
  }

 private:
  const int max_size_;
  std::deque<K> lru_queue_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<K, V> cache_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
}

}  // namespace core

#endif
