package(default_visibility = ["//visibility:public"])

cc_library(
    name = "create_tensor_shape",
    srcs = ["create_tensor_shape.cc"],
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
        "//cc:test": ["nn_interface_dummy.cc"],
        "//conditions:default": ["nn_interface.cc"],
    }),
    hdrs = ["nn_interface.h"],
    deps = [
        ":feed_fetch_names",
        "//cc/constants",
        "//cc/game",
        "//cc/tensorflow",
    ] + select({
        "//cc:test": [],
        "//conditions:default": [
            ":create_tensor_shape",
            ":nn_board_utils",
            "@com_google_absl//absl/container:inlined_vector",
            "@com_google_absl//absl/status:statusor",
        ],
    }),
)

cc_library(
    name = "feed_fetch_names",
    srcs = select({
        "//cc:trt": ["feed_fetch_names_trt.cc"],
        "//conditions:default": ["feed_fetch_names.cc"],
    }),
    hdrs = ["feed_fetch_names.h"],
)

cc_library(
    name = "nn",
    deps = [
        ":nn_board_utils",
        ":nn_interface",
    ],
)
