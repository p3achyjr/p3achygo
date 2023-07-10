# Shuffler runs in its own binary.

cc_library(
    name = "shuffler",
    srcs = [
        "chunk_info.cc",
        "chunk_manager.cc",
        "tf_record_watcher.cc",
    ],
    hdrs = [
        "chunk_info.h",
        "chunk_manager.h",
        "constants.h",
        "tf_record_watcher.h",
    ],
    deps = [
        ":filename_buffer",
        "//cc/core",
        "//cc/tensorflow",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":shuffler",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
    ],
)

cc_library(
    name = "filename_buffer",
    srcs = ["filename_buffer.cc"],
    hdrs = ["filename_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//cc/constants",
        "//cc/core",
        "//cc/tensorflow",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)
