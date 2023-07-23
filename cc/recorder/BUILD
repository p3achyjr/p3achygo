package(default_visibility = ["//visibility:public"])

cc_library(
    name = "recorder",
    deps = [
        ":dir",
        ":game_recorder",
        ":sgf_recorder",
        ":tf_recorder",
    ],
)

cc_library(
    name = "dir",
    hdrs = ["dir.h"],
)

cc_library(
    name = "make_tf_example",
    hdrs = ["make_tf_example.h"],
    deps = [
        "//cc/game",
        "//cc/tensorflow",
    ],
)

cc_library(
    name = "game_recorder",
    srcs = ["game_recorder.cc"],
    hdrs = ["game_recorder.h"],
    deps = [
        ":dir",
        ":sgf_recorder",
        ":tf_recorder",
        "//cc/core:filepath",
        "//cc/game",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "sgf_recorder",
    srcs = ["sgf_recorder.cc"],
    hdrs = ["sgf_recorder.h"],
    deps = [
        "//cc/constants",
        "//cc/core:filepath",
        "//cc/data:filename_format",
        "//cc/game",
        "//cc/mcts",
        "//cc/sgf:sgf_serializer",
        "//cc/sgf:sgf_tree",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tf_recorder",
    srcs = ["tf_recorder.cc"],
    hdrs = ["tf_recorder.h"],
    deps = [
        ":make_tf_example",
        "//cc/constants",
        "//cc/core:filepath",
        "//cc/data:filename_format",
        "//cc/game",
        "//cc/tensorflow",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
