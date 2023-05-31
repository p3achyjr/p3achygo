package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nn_board_utils",
    srcs = ["nn_board_utils.cc"],
    hdrs = ["nn_board_utils.h"],
    deps = [
        "//cc/constants",
        "//cc/game",
        "//cc/tensorflow",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "nn_interface",
    srcs = ["nn_interface.cc"],
    hdrs = ["nn_interface.h"],
    deps = [
        ":nn_board_utils",
        "//cc/constants",
        "//cc/game",
        "//cc/tensorflow",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "nn",
    deps = [
        ":nn_board_utils",
        ":nn_interface",
    ],
)
