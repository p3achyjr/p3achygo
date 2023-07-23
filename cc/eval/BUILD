# Eval Binary

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    linkopts = [
        "-lm",
        "-Wl,-ldl",
    ],
    deps = [
        ":eval",
        "//cc/core",
        "//cc/nn",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "eval",
    srcs = ["eval.cc"],
    hdrs = ["eval.h"],
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
