package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sgf",
    deps = [
        ":parse_sgf",
        ":sgf_serializer",
        ":sgf_tree",
        ":sgf_visitor",
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
    name = "parse_sgf",
    srcs = ["parse_sgf.cc"],
    hdrs = ["parse_sgf.h"],
    deps = [
        ":sgf_tree",
        ":sgf_visitor",
        "//cc/constants",
        "//cc/core",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "parse_sgf_test_data",
    hdrs = ["parse_sgf_test_data.h"],
)

cc_test(
    name = "parse_sgf_test",
    srcs = ["parse_sgf_test.cc"],
    deps = [
        ":parse_sgf",
        ":parse_sgf_test_data",
        ":sgf_serializer",
        "//cc/core:doctest",
    ],
)
