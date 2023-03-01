package(default_visibility = ["//visibility:public"])

cc_library(
    name = "game",
    srcs = [
        "board.cc",
        "zobrist_hash.cc",
    ],
    hdrs = [
        "board.h",
        "constants.h",
        "zobrist_hash.h",
    ],
    deps = [
        "//cc/core:rand",
        "@absl//absl/container:flat_hash_set",
        "@absl//absl/log",
        "@absl//absl/log:check",
        "@absl//absl/numeric:int128",
    ],
)

cc_test(
    name = "board_test",
    srcs = ["board_test.cc"],
    deps = [
        ":game",
        "//cc/core:doctest",
    ],
)
