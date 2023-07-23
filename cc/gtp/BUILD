# GTP Protocol Implementation.

cc_library(
    name = "gtp",
    srcs = [
        "client.cc",
        "command.cc",
        "service.cc",
    ],
    hdrs = [
        "client.h",
        "command.h",
        "service.h",
    ],
    deps = [
        "//cc/core",
        "//cc/mcts",
        "//cc/nn",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":gtp",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
