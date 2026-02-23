#ifndef SELFPLAY_BOOK_H_
#define SELFPLAY_BOOK_H_

#include <vector>

#include "cc/game/loc.h"

namespace selfplay {
const std::vector<std::vector<game::Loc>> kOpeningBook = {
    {{3, 3}, {15, 15}, {15, 4}, {4, 15}},
    {{3, 3}, {15, 15}, {16, 4}, {4, 15}},
    {{3, 3}, {15, 4}, {15, 16}, {15, 4}},
    {{3, 3}, {15, 4}, {15, 15}, {4, 15}},
    {{3, 3}, {15, 15}, {2, 15}, {15, 15}},
    {{3, 3}, {15, 15}, {2, 15}, {16, 15}}};
}

#endif
