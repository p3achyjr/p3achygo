package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nn_evaluator",
    srcs = ["nn_evaluator.cc"],
    hdrs = ["nn_evaluator.h"],
    deps = [
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
    ],
)

cc_library(
    name = "nn_board_utils",
    srcs = ["nn_board_utils.cc"],
    hdrs = ["nn_board_utils.h"],
    deps = [
        "//cc/game",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/core",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "nn",
    deps = [
        ":nn_board_utils",
        ":nn_evaluator",
    ],
)
