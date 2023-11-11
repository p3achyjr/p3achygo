package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "build_and_run_trt_engine",
    srcs = ["build_and_run_trt_engine.cc"],
    linkopts = ["-ldl"],
    deps = [
        "//cc/nn/engine:benchmark_engine",
        "//cc/nn/engine:go_features",
        "//cc/nn/engine:trt",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "compare_engines",
    srcs = ["compare_engines.cc"],
    linkopts = ["-ldl"],
    deps = [
        "//cc/nn/engine:benchmark_engine",
        "//cc/nn/engine:engine_factory",
        "//cc/nn/engine:engine_hdr",
        "//cc/nn/engine:go_dataset",
        "//cc/nn/engine:go_features",
        "//cc/nn/engine:tf_engine",
        "//cc/nn/engine:trt_engine",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
