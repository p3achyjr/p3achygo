# Workspace Config.
# We do not pull tensorflow as an external repo--rather, before build time, we
# copy tensorflow shared libraries and includes in //cc/tensorflow, and link them
# into our binary. This reduces build times significantly (from 3+ hours to a few minutes).

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Bazel Skylib rules.
git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "1.4.1",
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

# Doctest
# https://github.com/doctest/doctest.git
git_repository(
    name = "doctest",
    commit = "f25235f4c2f8a5fcf8e888114a90864ef5e4bf56",
    remote = "https://github.com/doctest/doctest.git",
)

# Abseil-CPP
# https://github.com/abseil/abseil-cpp
http_archive(
    name = "com_google_absl",
    sha256 = "3e0b4a1b8edc78026cd3dc4ecfe793ec5794ae692081db459e1b7bb6d9844375",
    strip_prefix = "abseil-cpp-a0f9b465212aea24d3264b82d315b8ee59e8d7a0",
    urls = ["https://github.com/abseil/abseil-cpp/archive/a0f9b465212aea24d3264b82d315b8ee59e8d7a0.zip"],
)

# Google Benchmark
# https://github.com/google/benchmark
http_archive(
    name = "com_github_google_benchmark",
    sha256 = "5f98b44165f3250f1d749b728018318d654f763ea0f4d7ea156e10e6e0cc678a",
    strip_prefix = "benchmark-5e78bedfb07c615edb2b646d1e354980268c1728",
    urls = [
        "https://github.com/google/benchmark/archive/5e78bedfb07c615edb2b646d1e354980268c1728.zip",
    ],
)

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    strip_prefix = "bazel-compile-commands-extractor-d7a28301d812aeafa36469343538dbc025cec196",

    # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/d7a28301d812aeafa36469343538dbc025cec196.tar.gz",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

# Indicators
# https://github.com/p-ranav/indicators
http_archive(
    name = "indicators",
    build_file_content = """
cc_library(
    name = "indicators",
    hdrs = glob(["include/**/*.hpp"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "70da7a693ff7a6a283850ab6d62acf628eea17d386488af8918576d0760aef7b",
    strip_prefix = "indicators-2.3",
    urls = ["https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz"],
)

# Local hdf5. Will be empty if hdf5 is not installed.
new_local_repository(
    name = "hdf5",
    build_file_content = """
cc_library(
    name = "hdf5",
    hdrs = glob(["include/hdf5/serial/*.h"]),
    includes = ["include/hdf5/serial/"],
    srcs = glob(["lib/x86_64-linux-gnu/hdf5/serial/libhdf5*.so"]),
    visibility = ["//visibility:public"],
)
""",
    path = "/usr",
)

# Local CUDA 12.0, with TRT. If not installed, should be empty.
new_local_repository(
    name = "cuda",
    build_file_content = """
cc_library(
    name = "cuda",
    hdrs = glob(["include/x86_64-linux-gnu/*.h",
                 "local/cuda-12.0/targets/x86_64-linux/include/*.h"]),
    srcs = glob(["lib/x86_64-linux-gnu/libnv*.so",
                 "local/cuda-12.0/targets/x86_64-linux/lib/*.so"]),
    includes = ["include/x86_64-linux-gnu/",
                "local/cuda-12.0/targets/x86_64-linux/include"],
    visibility = ["//visibility:public"],
)
""",
    path = "/usr",
)

# Boost Math
# https://github.com/boostorg/math.git
git_repository(
    name = "boost_math",
    build_file_content = """
cc_library(
    name = "boost_math",
    srcs = glob([
        "src/**/*.cpp",
    ]),
    hdrs = glob([
        "include/**/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
    """,
    commit = "44af29a78c85ee89ce37f7f43d532afd05c3d981",
    remote = "https://github.com/boostorg/math.git",
)
