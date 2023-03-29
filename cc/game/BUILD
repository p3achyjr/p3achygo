package(default_visibility = ["//visibility:public"])

cc_library(
    name = "game",
    srcs = [
        "board.cc",
        "zobrist_hash.cc",
    ],
    hdrs = [
        "board.h",
        "zobrist_hash.h",
    ],
    deps = [
        "//cc/constants",
        "//cc/core",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
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
