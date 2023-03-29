package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gumbel",
    srcs = ["gumbel.cc"],
    hdrs = ["gumbel.h"],
    deps = [
        ":tree",
        "//cc/constants",
        "//cc/core",
        "//cc/game",
        "//cc/nn:nn_interface",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "tree",
    srcs = ["tree.cc"],
    hdrs = ["tree.h"],
    deps = [
        "//cc/constants",
        "//cc/game",
    ],
)

cc_library(
    name = "mcts",
    deps = [
        ":gumbel",
        ":tree",
    ],
)
