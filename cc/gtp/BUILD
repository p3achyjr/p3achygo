# GTP Protocol Implementation.

cc_library(
    name = "gtp",
    srcs = [
        "client.cc",
        "command.cc",
        "parse.cc",
        "service.cc",
    ],
    hdrs = [
        "client.h",
        "command.h",
        "parse.h",
        "service.h",
    ],
    deps = [
        "//cc/analysis",
        "//cc/core",
        "//cc/mcts",
        "//cc/nn",
        "//cc/nn/engine:engine_factory",
        "//cc/recorder:sgf_recorder",
        "//cc/sgf",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    linkopts = [
        "-lm",
        "-Wl,-ldl",
    ],
    deps = [
        ":gtp",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
