load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "position_cache",
    hdrs = ["position_cache.h"],
    deps = ["//cc/core:lru_cache"],
)

cc_library(
    name = "gumbel",
    srcs = ["gumbel.cc"],
    hdrs = ["gumbel.h"],
    deps = [
        ":leaf_evaluator",
        ":tree",
        "//cc/constants",
        "//cc/core",
        "//cc/game",
        "//cc/nn:nn_interface",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "gumbel_test",
    srcs = ["gumbel_test.cc"],
    deps = [
        ":gumbel",
        "//cc/core",
        "//cc/core:doctest",
    ],
)

cc_test(
    name = "gumbel_benchmark",
    srcs = ["gumbel_benchmark.cc"],
    deps = [
        ":gumbel",
        "//cc/core",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "mcts_constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "leaf_evaluator",
    srcs = select({
        "//cc:test": ["leaf_evaluator_dummy.cc"],
        "//conditions:default": ["leaf_evaluator.cc"],
    }),
    hdrs = ["leaf_evaluator.h"],
    deps = [
        ":mcts_constants",
        ":tree",
        "//cc/game",
        "//cc/nn:nn_interface",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "tree",
    srcs = ["tree.cc"],
    hdrs = ["tree.h"],
    deps = [
        ":mcts_constants",
        "//cc/constants",
        "//cc/game",
    ],
)

cc_library(
    name = "mcts",
    deps = [
        ":gumbel",
        ":tree",
    ],
)
