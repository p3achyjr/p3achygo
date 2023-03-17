package(default_visibility = ["//visibility:public"])

cc_library(
    name = "core",
    deps = [
        ":probability",
        ":rand",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
)

cc_library(
    name = "probability",
    srcs = ["probability.cc"],
    hdrs = ["probability.h"],
    deps = [":rand"],
)

cc_library(
    name = "rand",
    srcs = ["rand.cc"],
    hdrs = ["rand.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
    ],
)

cc_test(
    name = "rand_test",
    srcs = ["rand_test.cc"],
    deps = [
        ":doctest",
        ":rand",
    ],
)

cc_library(
    name = "doctest",
    hdrs = [
        "doctest_include.h",
    ],
    deps = ["@doctest//doctest"],
)
