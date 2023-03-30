load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "doctest",
    commit = "f25235f4c2f8a5fcf8e888114a90864ef5e4bf56",
    remote = "https://github.com/doctest/doctest.git",
)

http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-a0f9b465212aea24d3264b82d315b8ee59e8d7a0",
    urls = ["https://github.com/abseil/abseil-cpp/archive/a0f9b465212aea24d3264b82d315b8ee59e8d7a0.zip"],
)

# tensorflow_http_archive(
#     name = "org_tensorflow",
#     sha256 = "99c732b92b1b37fc243a559e02f9aef5671771e272758aa4aec7f34dc92dac48",
#     version = "2.11.0",
# )

http_archive(
    name = "org_tensorflow",
    sha256 = "99c732b92b1b37fc243a559e02f9aef5671771e272758aa4aec7f34dc92dac48",
    strip_prefix = "tensorflow-2.11.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.11.0.tar.gz",
    ],
)

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")

workspace()

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

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()
