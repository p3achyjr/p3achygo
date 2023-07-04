load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "game",
    deps = [
        ":board",
        ":game_impl",
        ":hdrs",
        ":symmetry",
        ":zobrist",
        "//cc/constants",
        "//cc/core",
    ],
)

cc_library(
    name = "game_impl",
    srcs = ["game.cc"],
    hdrs = ["game.h"],
    deps = [
        ":board",
        ":hdrs",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "board",
    srcs = ["board.cc"],
    hdrs = ["board.h"],
    deps = [
        ":hdrs",
        ":zobrist",
        "//cc/constants",
        "//cc/core",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
    ],
)

cc_binary(
    name = "board_benchmark",
    srcs = ["board_benchmark.cc"],
    deps = [
        ":board",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "hdrs",
    hdrs = [
        "color.h",
        "loc.h",
        "move.h",
    ],
    deps = [
        "//cc/constants",
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "symmetry",
    srcs = ["symmetry.cc"],
    hdrs = ["symmetry.h"],
    deps = [
        ":hdrs",
        "//cc/core:rand",
    ],
)

cc_library(
    name = "zobrist",
    srcs = ["zobrist.cc"],
    hdrs = ["zobrist.h"],
    deps = [
        "//cc/constants",
        "//cc/core",
        "@com_google_absl//absl/numeric:int128",
    ],
)

cc_test(
    name = "board_test",
    srcs = ["board_test.cc"],
    linkopts = ["-lm"],
    deps = [
        ":board",
        "//cc/core:doctest",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "symmetry_test",
    srcs = ["symmetry_test.cc"],
    deps = [
        ":symmetry",
        "//cc/core:doctest",
    ],
)
