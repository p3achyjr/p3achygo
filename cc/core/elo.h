#ifndef CORE_ELO_H_
#define CORE_ELO_H_

#include <cmath>

namespace core {

inline float RelativeElo(float winrate) {
  static constexpr int kEloOffset = 400;

  return kEloOffset * std::log10(winrate / (1.0f - winrate));
}

}  // namespace core

#endif
