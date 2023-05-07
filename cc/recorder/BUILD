package(default_visibility = ["//visibility:public"])

cc_library(
    name = "recorder_base",
    hdrs = ["recorder.h"],
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
