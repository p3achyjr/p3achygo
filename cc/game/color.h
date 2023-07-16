#ifndef GAME_COLOR_H_
#define GAME_COLOR_H_

#include <cstdint>
#include <iostream>

namespace game {
using Color = int8_t;

inline Color OppositeColor(Color color) { return -color; }

}  // namespace game

#endif
