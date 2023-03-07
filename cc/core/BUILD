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
)

cc_library(
    name = "rand",
    srcs = ["rand.cc"],
    hdrs = ["rand.h"],
    deps = [
        "@absl//absl/numeric:int128",
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
        "doctest.h",
        "doctest_include.h",
    ],
)
