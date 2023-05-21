# Shuffler runs in its own binary.

cc_library(
    name = "shuffler",
    srcs = [
        "chunk_manager.cc",
        "filename_buffer.cc",
        "tf_record_watcher.cc",
    ],
    hdrs = [
        "chunk_manager.h",
        "filename_buffer.h",
        "tf_record_watcher.h",
    ],
    deps = [
        "//cc/core",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [":shuffler"],
)
