package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "build_and_run_trt_engine",
    srcs = ["build_and_run_trt_engine.cc"],
    deps = [
        "//cc/nn/engine:trt_engine",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
