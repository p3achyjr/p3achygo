load(":cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name = "empty")

cc_toolchain_config(name = "darwin_toolchain_config")

cc_toolchain_suite(
    name = "clang_suite",
    toolchains = {
        "darwin": ":darwin_toolchain",
    },
)

cc_toolchain(
    name = "darwin_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":darwin_toolchain_config",
    toolchain_identifier = "k8-toolchain",
)
