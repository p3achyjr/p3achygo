#ifndef CORE_LRU_CACHE_H_
#define CORE_LRU_CACHE_H_

#include <deque>
#include <optional>

#include "absl/container/flat_hash_map.h"

namespace core {

/*
 * Simple LRU Cache Implementation. Not thread-safe.
 *
 * Type `K` must be hashable via absl::Hash.
 */
template <typename K, typename V>
class LRUCache final {
 public:
  LRUCache(int max_size) : max_size_(max_size) {}
  ~LRUCache() = default;

  bool Contains(const K& key) const { return cache_.contains(key); }

  std::optional<V> Get(const K& key) {
    auto v_it = cache_.find(key);
    if (v_it == cache_.end()) {
      return std::nullopt;
    }

    return v_it->second;
  }

  void Insert(const K& key, const V& val) {
    cache_[key] = val;
    lru_queue_.push_back(key);

    if (cache_.size() > max_size_) {
      K lru_key = lru_queue_.front();
      cache_.erase(lru_key);
      lru_queue_.pop_front();
    }
  }

 private:
  const int max_size_;
  std::deque<K> lru_queue_;
  absl::flat_hash_map<K, V> cache_;
};

}  // namespace core

#endif
