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
    name = "recorder_base",
    hdrs = ["recorder.h"],
)

cc_library(
    name = "game_recorder",
    srcs = ["game_recorder.cc"],
    hdrs = ["game_recorder.h"],
    deps = [
        ":dir",
        ":recorder_base",
        ":sgf_recorder",
        ":tf_recorder",
        "//cc/game",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "sgf_recorder",
    srcs = ["sgf_recorder.cc"],
    hdrs = ["sgf_recorder.h"],
    deps = [
        ":recorder_base",
        ":sgf_serializer",
        ":sgf_tree",
        "//cc/constants",
        "//cc/game",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sgf_tree",
    hdrs = ["sgf_tree.h"],
    deps = [
        "//cc/game",
    ],
)

cc_library(
    name = "sgf_visitor",
    hdrs = ["sgf_visitor.h"],
)

cc_library(
    name = "sgf_serializer",
    srcs = ["sgf_serializer.cc"],
    hdrs = ["sgf_serializer.h"],
    deps = [
        ":sgf_tree",
        ":sgf_visitor",
        "//cc/constants",
        "//cc/game",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "sgf_serializer_test",
    srcs = ["sgf_serializer_test.cc"],
    linkopts = ["-lm"],
    deps = [
        ":sgf_serializer",
        "//cc/core:doctest",
    ],
)

cc_library(
    name = "tf_recorder",
    srcs = ["tf_recorder.cc"],
    hdrs = ["tf_recorder.h"],
    deps = [
        ":recorder_base",
        "//cc/constants",
        "//cc/game",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/example:example_protos_cc",
    ],
)
