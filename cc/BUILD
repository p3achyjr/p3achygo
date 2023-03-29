cc_binary(
    name = "main",
    srcs = ["main.cc"],
    linkopts = ["-lm"],
    deps = [
        ":self_play",
        "//cc/nn",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)
