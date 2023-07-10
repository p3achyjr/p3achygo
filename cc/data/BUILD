package(default_visibility = ["//visibility:public"])

cc_library(
    name = "dataset_creator",
    srcs = [
        "coordinator.cc",
        "worker.cc",
    ],
    hdrs = [
        "coordinator.h",
        "worker.h",
    ],
    deps = [
        "//cc/game",
        "//cc/recorder:make_tf_example",
        "//cc/sgf",
        "//cc/shuffler:filename_buffer",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@indicators",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":dataset_creator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
    ],
)
