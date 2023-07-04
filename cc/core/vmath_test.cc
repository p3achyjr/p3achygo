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
  alignas(MM_ALIGN) std::array<float, 3> small = {1.0, 1.5, 2.0};
  alignas(MM_ALIGN) std::array<float, 6> small_uneven = {1.0, 1.5, 2.0,
                                                         2.5, 3.0, 3.5};
  alignas(MM_ALIGN) std::array<float, 8> even = {5.2,  1.3, -4.4, 4.5,
                                                 -3.0, 7.2, 2.1,  -4.0};
  alignas(MM_ALIGN) std::array<float, 10> uneven = {5.2, 1.3, -4.4, 4.5, -3.0,
                                                    7.2, 2.1, -4.0, 9.8, 4.6};
  alignas(MM_ALIGN) std::array<float, 7> all_neg = {
      -149.944, -157.025, -158.732, -158.947, -160.693, -161.818, -161.623};

  SUBCASE("Max") {
    CHECK(MaxV(small) == doctest::Approx(2.0));
    CHECK(MaxV(small_uneven) == doctest::Approx(3.5));
    CHECK(MaxV(even) == doctest::Approx(7.2));
    CHECK(MaxV(uneven) == doctest::Approx(9.8));
    CHECK(MaxV(all_neg) == doctest::Approx(-149.944));
  }

  SUBCASE("Sum") {
    CHECK(SumV(small) == doctest::Approx(4.5));
    CHECK(SumV(small_uneven) == doctest::Approx(13.5));
    CHECK(SumV(even) == doctest::Approx(8.9));
    CHECK(SumV(uneven) == doctest::Approx(23.3));
    CHECK(SumV(all_neg) == doctest::Approx(-1108.782));
  }

  SUBCASE("Softmax") {
    alignas(MM_ALIGN) std::array<float, 3> small_softmax = {
        0.18632372, 0.30719589, 0.50648039};
    alignas(MM_ALIGN) std::array<float, 8> even_softmax = {
        1.11714669e-01, 2.26131844e-03, 7.56629338e-06, 5.54758628e-02,
        3.06828327e-05, 8.25465956e-01, 5.03265673e-03, 1.12875833e-05};
    alignas(MM_ALIGN) std::array<float, 10> uneven_softmax = {
        9.17561645e-03, 1.85732016e-04, 6.21452908e-07, 4.55647628e-03,
        2.52011581e-06, 6.77991447e-02, 4.13354202e-04, 9.27098797e-07,
        9.12829923e-01, 5.03568507e-03};
    alignas(MM_ALIGN) std::array<float, 7> all_neg_softmax = {
        9.9884784e-01, 8.3996827e-04, 1.5237786e-04, 1.2289763e-04,
        2.1442285e-05, 6.9612906e-06, 8.4600651e-06};
    CHECK_FLOAT_VEC_EQ(SoftmaxV(small), small_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(even), even_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(uneven), uneven_softmax);
    CHECK_FLOAT_VEC_EQ(SoftmaxV(all_neg), all_neg_softmax);
  }
}
}  // namespace core
