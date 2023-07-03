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
