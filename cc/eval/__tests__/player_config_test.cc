#include "cc/eval/player_config.h"

#include "cc/core/doctest_include.h"

namespace eval {
namespace {

TEST_CASE("NewPlayouts") {
  PlayerSearchConfig cfg;
  cfg.n = 500;

  SUBCASE("default: n new playouts regardless of the reused tree") {
    CHECK(NewPlayouts(cfg, 0) == 500);
    CHECK(NewPlayouts(cfg, 300) == 500);
    CHECK(NewPlayouts(cfg, 5000) == 500);
  }

  SUBCASE("n_counts_reused_visits: top the root up to n") {
    cfg.n_counts_reused_visits = true;
    // A fresh root gets 1 visit from its own evaluation.
    CHECK(NewPlayouts(cfg, 0) == 499);
    CHECK(NewPlayouts(cfg, 1) == 499);
    CHECK(NewPlayouts(cfg, 300) == 200);
    CHECK(NewPlayouts(cfg, 499) == 1);
    // A reused tree at or above n still searches 1 playout.
    CHECK(NewPlayouts(cfg, 500) == 1);
    CHECK(NewPlayouts(cfg, 5000) == 1);
  }
}

TEST_CASE("SearchKomi") {
  PlayerSearchConfig cfg;

  SUBCASE("default: komi is used as given") {
    CHECK(SearchKomi(cfg, 7.5f) == 7.5f);
    CHECK(SearchKomi(cfg, 6.5f) == 6.5f);
  }

  SUBCASE("territory_komi: territory komi + 1 = area komi") {
    cfg.territory_komi = true;
    CHECK(SearchKomi(cfg, 6.5f) == 7.5f);
    CHECK(SearchKomi(cfg, 0.5f) == 1.5f);
  }
}

}  // namespace
}  // namespace eval
