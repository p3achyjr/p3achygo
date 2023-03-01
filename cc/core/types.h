#ifndef __CORE_TYPES_H_
#define __CORE_TYPES_H_

#include <cstdint>
#include <iostream>

namespace core {

struct UInt128 {
  uint64_t upper;
  uint64_t lower;
};

inline bool operator==(const UInt128& x, const UInt128& y) {
  return x.upper == y.upper && x.lower == y.lower;
}

inline std::ostream& operator<<(std::ostream& os, const UInt128& x) {
  return os << std::hex << x.upper << " " << std::hex << x.lower;
}
}  // namespace core

#endif  // __CORE_TYPES_H_
