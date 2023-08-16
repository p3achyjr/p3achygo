package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "build_and_run_trt_engine",
    srcs = ["build_and_run_trt_engine.cc"],
    linkopts = ["-ldl"],
    deps = [
        "//cc/nn/engine:trt",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
