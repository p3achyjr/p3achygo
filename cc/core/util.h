#ifndef __CC_CORE_UTIL_H_
#define __CC_CORE_UTIL_H_

#include <sstream>
#include <string>
#include <vector>

namespace core {

template <typename T>
inline std::string VecToString(const std::vector<T> vec) {
  std::stringstream ss;
  ss << "<";
  for (auto& x : vec) {
    ss << x << ", ";
  }
  ss << ">";

  // convert the stream buffer into a string
  return ss.str();
}

}  // namespace core

#endif