package(default_visibility = ["//visibility:public"])

# Depends on TRT being installed in default system path.
# Also depends on existence of `/usr/local/cuda-12.0/targets/x86_64-linux`.
cc_library(
    name = "trt",
    srcs = [
        "trt_calibrator.cc",
    ],
    hdrs = [
        "trt_calibrator.h",
    ],
    deps = [
        ":go_dataset",
        ":go_features",
        ":trt_engine",
        ":trt_utils",
        "//cc/constants",
        "//cc/game",
        "//cc/tensorflow",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@cuda",
    ],
)

cc_library(
    name = "trt_utils",
    srcs = [
        "trt_logger.cc",
    ],
    hdrs = [
        "trt_logger.h",
        "trt_names.h",
    ],
    deps = ["@cuda"],
)

cc_library(
    name = "go_dataset",
    srcs = ["go_dataset.cc"],
    hdrs = ["go_dataset.h"],
    deps = [
        ":go_features",
        "//cc/constants",
        "//cc/game",
        "//cc/tensorflow",
    ],
)

cc_library(
    name = "go_features",
    srcs = [
        "go_features.cc",
    ],
    hdrs = [
        "buf_utils.h",
        "go_features.h",
    ],
    deps = [
        "//cc/constants",
        "//cc/game",
        "//cc/tensorflow",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "engine_hdr",
    hdrs = ["engine.h"],
    deps = [":go_features"],
)

cc_library(
    name = "engine_factory",
    srcs = ["engine_factory.cc"],
    hdrs = ["engine_factory.h"],
    deps = [
        ":engine_hdr",
        ":tf_engine",
        ":trt_engine",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "tf_engine",
    srcs = ["tf_engine.cc"],
    hdrs = ["tf_engine.h"],
    deps = [
        ":engine_hdr",
        "//cc/game",
        "//cc/tensorflow",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "trt_engine",
    srcs = ["trt_engine.cc"],
    hdrs = ["trt_engine.h"],
    deps = [
        ":engine_hdr",
        ":trt_utils",
        "//cc/game",
        "//cc/tensorflow",
        "@cuda",
    ],
)

cc_library(
    name = "benchmark_engine",
    srcs = ["benchmark_engine.cc"],
    hdrs = ["benchmark_engine.h"],
    deps = [
        ":engine_hdr",
        ":go_dataset",
        "@com_google_absl//absl/log",
    ],
)
