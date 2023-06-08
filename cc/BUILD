load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

bool_flag(
    name = "is_test",
    build_setting_default = False,
)

bool_flag(
    name = "is_trt",
    build_setting_default = False,
)

config_setting(
    name = "test",
    flag_values = {":is_test": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "trt",
    flag_values = {":is_trt": "true"},
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    linkopts = [
        "-lm",
        "-Wl,-ldl",
    ],
    deps = [
        ":self_play",
        "//cc/nn",
        "//cc/recorder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "self_play",
    srcs = ["self_play_thread.cc"],
    hdrs = ["self_play_thread.h"],
    deps = [
        "//cc/constants",
        "//cc/core",
        "//cc/game",
        "//cc/mcts",
        "//cc/nn",
        "//cc/recorder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)
