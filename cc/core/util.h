#ifndef __CORE_UTIL_H_
#define __CORE_UTIL_H_

#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"

namespace core {

template <typename T>
inline std::string VecToString(const std::vector<T>& vec) {
  std::stringstream ss;
  ss << "<";
  for (auto& x : vec) {
    ss << x << ", ";
  }
  ss << ">";

  // convert the stream buffer into a string
  return ss.str();
}

template <typename T, size_t N>
inline std::string InlinedVecToString(const absl::InlinedVector<T, N>& vec) {
  std::stringstream ss;
  ss << "<";
  for (auto& x : vec) {
    ss << x << ", ";
  }
  ss << ">";

  // convert the stream buffer into a string
  return ss.str();
}

template <typename K>
inline std::string SetToString(const absl::flat_hash_set<K>& set) {
  std::stringstream ss;
  ss << "<";
  for (auto& x : set) {
    ss << x << ", ";
  }
  ss << ">";

  // convert the stream buffer into a string
  return ss.str();
}

template <typename K, typename V>
inline std::string MapToString(const absl::flat_hash_map<K, V>& map) {
  std::stringstream ss;
  ss << "<";
  for (auto& [k, v] : map) {
    ss << "(" << k << ": " << v << "), ";
  }
  ss << ">";

  // convert the stream buffer into a string
  return ss.str();
}

template <typename T>
inline bool VecContains(const std::vector<T> vec, T x) {
  return std::find(vec.begin(), vec.end(), x) != vec.end();
}

template <typename T, size_t N>
inline bool InlinedVecContains(const absl::InlinedVector<T, N> vec, T x) {
  return std::find(vec.begin(), vec.end(), x) != vec.end();
}

template <typename K>
inline bool SetContains(const absl::flat_hash_set<K> set, K x) {
  return set.find(x) != set.end();
}

template <typename K, typename V>
inline bool MapContains(const absl::flat_hash_map<K, V> map, K x) {
  return map.find(x) != map.end();
}

}  // namespace core

#endif
