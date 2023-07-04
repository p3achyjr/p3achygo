#include "cc/core/vmath.h"

#include "cc/core/doctest_include.h"

#define CHECK_FLOAT_VEC_EQ(x, true)            \
  do {                                         \
    for (size_t i = 0; i < x.size(); ++i) {    \
      CHECK(x[i] == doctest::Approx(true[i])); \
    }                                          \
  } while (0)

namespace core {

TEST_CASE("VMathTest") {
  alignas(16) std::array<float, 3> small = {1.0, 1.5, 2.0};
  alignas(16) std::array<float, 6> small_uneven = {1.0, 1.5, 2.0,
                                                   2.5, 3.0, 3.5};
  alignas(16) std::array<float, 8> even = {5.2,  1.3, -4.4, 4.5,
                                           -3.0, 7.2, 2.1,  -4.0};
  alignas(16) std::array<float, 10> uneven = {5.2, 1.3, -4.4, 4.5, -3.0,
                                              7.2, 2.1, -4.0, 9.8, 4.6};

  SUBCASE("Max") {
    CHECK(MaxV(small) == doctest::Approx(2.0));
    CHECK(MaxV(small_uneven) == doctest::Approx(3.5));
    CHECK(MaxV(even) == doctest::Approx(7.2));
    CHECK(MaxV(uneven) == doctest::Approx(9.8));
  }

  SUBCASE("Sum") {
    CHECK(SumV(small) == doctest::Approx(4.5));
    CHECK(SumV(small_uneven) == doctest::Approx(13.5));
    CHECK(SumV(even) == doctest::Approx(8.9));
    CHECK(SumV(uneven) == doctest::Approx(23.3));
  }

  SUBCASE("Softmax") {
    alignas(16) std::array<float, 3> small_softmax = {0.18632372, 0.30719589,
                                                      0.50648039};
    alignas(16) std::array<float, 8> even_softmax = {
        1.11714669e-01, 2.26131844e-03, 7.56629338e-06, 5.54758628e-02,
        3.06828327e-05, 8.25465956e-01, 5.03265673e-03, 1.12875833e-05};
    alignas(16) std::array<float, 10> uneven_softmax = {
        9.17561645e-03, 1.85732016e-04, 6.21452908e-07, 4.55647628e-03,
        2.52011581e-06, 6.77991447e-02, 4.13354202e-04, 9.27098797e-07,
        9.12829923e-01, 5.03568507e-03};
    CHECK_FLOAT_VEC_EQ(SoftmaxV(small), small_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(even), even_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(uneven), uneven_softmax);
  }
}

}  // namespace core
