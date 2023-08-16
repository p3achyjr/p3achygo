package(default_visibility = ["//visibility:public"])

cc_library(
    name = "model_arch",
    hdrs = ["model_arch.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_map"],
)

cc_library(
    name = "validate_model_arch",
    srcs = ["validate_model_arch.cc"],
    hdrs = ["validate_model_arch.h"],
    deps = [
        ":model_arch",
        "//cc/core",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

# Depends on H5 being installed on the host machine.
cc_library(
    name = "parse_h5",
    srcs = ["parse_h5.cc"],
    hdrs = ["parse_h5.h"],
    deps = [
        ":model_arch",
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
        ":validate_model_arch",
        "//cc/core:doctest",
    ],
)

cc_library(
    name = "trt_names",
    hdrs = ["trt_names.h"],
)

# Depends on TRT being installed in default system path.
cc_library(
    name = "build_trt_engine",
    srcs = ["build_trt_engine.cc"],
    hdrs = [
        "build_trt_engine.h",
    ],
    deps = [
        ":model_arch",
        ":trt_names",
        "//cc/constants",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@nvinfer",
    ],
)

cc_library(
    name = "trt_engine",
    deps = [
        ":build_trt_engine",
        ":parse_h5",
        ":trt_names",
    ],
)
