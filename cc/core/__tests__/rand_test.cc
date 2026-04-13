#include "cc/core/rand.h"

#include <stdio.h>

#include "cc/core/doctest_include.h"

namespace core {
namespace {

TEST_CASE("Same Seed Determinism int32") {
  PRng prng0(7), prng1(7);  // arbitrarily chosen seed

  CHECK(prng0.next() == prng1.next());
}

TEST_CASE("Same Seed Determinism int64") {
  PRng prng0(11, 12), prng1(11, 12);  // arbitrarily chosen seed

  CHECK(prng0.next64() == prng1.next64());
}

TEST_CASE("Same Seed Determinism int128") {
  PRng prng0(17, 13, 4, 5), prng1(17, 13, 4, 5);  // arbitrarily chosen seed

  CHECK(prng0.next128() == prng1.next128());
}

TEST_CASE("Same Seed Determinism Sequential") {
  PRng prng0(0, 1, 2, 3), prng1(0, 1, 2, 3);  // arbitrarily chosen seed

  CHECK(prng0.next128() == prng1.next128());
  CHECK(prng0.next128() == prng1.next128());
  CHECK(prng0.next128() == prng1.next128());
  CHECK(prng0.next128() == prng1.next128());
}

TEST_CASE("RandRange") {
  PRng prng(0, 1, 2, 3);

  int res0 = RandRange(prng, 0, 5);
  int res1 = RandRange(prng, -80, -40);
  int res2 = RandRange(prng, 10, 50);
  int res3 = RandRange(prng, -40, 30);

  CHECK(res0 >= 0);
  CHECK(res0 < 5);
  CHECK(res1 >= -80);
  CHECK(res1 < -40);
  CHECK(res2 >= 10);
  CHECK(res2 < 50);
  CHECK(res3 >= -40);
  CHECK(res3 < 30);
}

}  // namespace
}  // namespace core
