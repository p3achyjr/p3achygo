#ifndef __GAME_COLOR_H_
#define __GAME_COLOR_H_

#include <cstdint>

namespace game {
using Color = int8_t;

inline int OppositeColor(Color color) { return -color; }

}  // namespace game

#endif
