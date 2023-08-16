package(default_visibility = ["//visibility:public"])

cc_library(
    name = "model_arch",
    hdrs = ["model_arch.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_map"],
)

# Depends on H5 being installed on the host machine.
cc_library(
    name = "parse_h5",
    srcs = [
        "parse_h5.cc",
        "validate_model_arch.cc",
    ],
    hdrs = [
        "parse_h5.h",
        "validate_model_arch.h",
    ],
    deps = [
        ":model_arch",
        "//cc/constants",
        "//cc/core",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@hdf5",
    ],
)

cc_test(
    name = "parse_h5_test",
    srcs = ["parse_h5_test.cc"],
    deps = [
        ":parse_h5",
        "//cc/core:doctest",
    ],
)

# Depends on TRT being installed in default system path.
# Also depends on existence of `/usr/local/cuda-12.0/targets/x86_64-linux`.
cc_library(
    name = "trt_engine_builder",
    srcs = [
        "trt_calibrator.cc",
        "trt_engine_builder.cc",
        "trt_logger.cc",
    ],
    hdrs = [
        "trt_calibrator.h",
        "trt_engine_builder.h",
        "trt_logger.h",
        "trt_names.h",
    ],
    deps = [
        ":model_arch",
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
    name = "trt",
    deps = [
        ":parse_h5",
        ":trt_engine_builder",
    ],
)
