#pragma once

#include <cstdint>

namespace selfplay {

enum class ForkKind : uint8_t {
  kEarly = 0,
  kLate = 1,
  kSampleT1 = 2,
  kSampleT2 = 3,
  kSampleUniform = 4,
  kRegret = 5,
  kUniform = 6,
};

}  // namespace selfplay
