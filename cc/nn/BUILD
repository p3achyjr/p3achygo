package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nn",
    deps = [
        ":nn_interface",
    ],
)

cc_library(
    name = "create_tensor_shape",
    hdrs = ["create_tensor_shape.h"],
    deps = ["//cc/tensorflow"],
)

cc_library(
    name = "nn_board_utils",
    srcs = ["nn_board_utils.cc"],
    hdrs = ["nn_board_utils.h"],
    deps = [
        ":create_tensor_shape",
        "//cc/constants",
        "//cc/game",
        "//cc/tensorflow",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "nn_interface",
    srcs = select({
        "//cc:nn_interface_dummy": ["nn_interface_dummy.cc"],
        "//conditions:default": ["nn_interface.cc"],
    }),
    hdrs = ["nn_interface.h"],
    deps = [
        "//cc/constants",
        "//cc/core:cache",
        "//cc/game",
        "//cc/nn/engine:engine_hdr",
        "//cc/nn/engine:go_features",
        "//cc/tensorflow",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ] + select({
        "//cc:nn_interface_dummy": [],
        "//conditions:default": [
            "@com_google_absl//absl/container:inlined_vector",
        ],
    }),
)

cc_test(
    name = "nn_board_utils_test",
    srcs = ["nn_board_utils_test.cc"],
    deps = [
        ":create_tensor_shape",
        ":nn_board_utils",
        "//cc/core:doctest",
        "//cc/tensorflow",
    ],
)
