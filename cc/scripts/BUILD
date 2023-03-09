cc_binary(
    name = "ucb_scratch",
    srcs = ["ucb_scratch.cc"],
    deps = [
        "//cc/core",
        "//cc/game",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_binary(
    name = "play_model",
    srcs = ["play_model.cc"],
    deps = [
        "//cc/core",
        "//cc/game",
        "//cc/nn",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)
