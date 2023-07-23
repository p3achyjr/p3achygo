load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "core",
    deps = [
        ":cache",
        ":elo",
        ":file_log_sink",
        ":filepath",
        ":lru_cache",
        ":probability",
        ":rand",
        ":ring_buffer",
        ":util",
        ":vmath",
    ],
)

cc_library(
    name = "cache",
    hdrs = ["cache.h"],
    deps = [
        "@com_google_absl//absl/hash",
    ],
)

cc_library(
    name = "elo",
    hdrs = ["elo.h"],
)

cc_library(
    name = "file_log_sink",
    hdrs = ["file_log_sink.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
)

cc_library(
    name = "filepath",
    hdrs = ["filepath.h"],
)

cc_library(
    name = "probability",
    srcs = select({
        "//cc:probability_dummy": ["probability_dummy.cc"],
        "//conditions:default": ["probability.cc"],
    }),
    hdrs = ["probability.h"],
    deps = [":rand"],
)

cc_library(
    name = "rand",
    srcs = ["rand.cc"],
    hdrs = ["rand.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
    ],
)

cc_test(
    name = "rand_test",
    srcs = ["rand_test.cc"],
    deps = [
        ":doctest",
        ":rand",
    ],
)

cc_library(
    name = "ring_buffer",
    hdrs = ["ring_buffer.h"],
    deps = [":rand"],
)

cc_library(
    name = "doctest",
    hdrs = [
        "doctest_include.h",
    ],
    deps = ["@doctest//doctest"],
)

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_set"],
)

cc_library(
    name = "vmath",
    hdrs = ["vmath.h"],
)

cc_test(
    name = "vmath_test",
    srcs = ["vmath_test.cc"],
    deps = [
        ":doctest",
        ":vmath",
    ],
)
