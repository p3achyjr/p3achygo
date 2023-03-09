load(":cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name = "empty")

config_setting(
    name = "macos_build",
    values = {"cpu": "darwin"},
)

config_setting(
    name = "linux_build",
    values = {"cpu": "k8"},
)

local_tool_paths = {
    "gcc": "/usr/local/opt/llvm/bin/clang",
    "ld": "/usr/local/opt/llvm/bin/lld",
    "ar": "/usr/local/opt/llvm/bin/llvm-ar",
    "cpp": "/bin/false",
    "gcov": "/usr/local/opt/llvm/bin/llvm-gcov",
    "nm": "/usr/local/opt/llvm/bin/llvm-nm",
    "objdump": "/usr/local/opt/llvm/bin/llvm-objdump",
    "strip": "/usr/local/opt/llvm/bin/llvm-strip",
}

vm_tool_paths = {
    "gcc": "/usr/lib/llvm-12/bin/clang",
    "ld": "/usr/lib/llvm-12/bin/lld",
    "ar": "/usr/lib/llvm-12/bin/llvm-ar",
    "cpp": "/bin/false",
    "gcov": "/usr/lib/llvm-12/bin/llvm-gcov",
    "nm": "/usr/lib/llvm-12/bin/llvm-nm",
    "objdump": "/usr/lib/llvm-12/bin/llvm-objdump",
    "strip": "/usr/lib/llvm-12/bin/llvm-strip",
}

local_include_paths = [
    "/usr/local/opt/llvm/lib/clang/12.0.0/include",
    "/usr/local/Cellar/llvm/12.0.0/lib/clang/12.0.0/include",
    "/usr/local/opt/llvm/include/c++/v1",
    "/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include/",
    "/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/System/Library/Frameworks",
]

vm_include_paths = [
    "/usr/lib/llvm-12/lib/clang/12.0.1/include",
    "/usr/include",
]

cc_toolchain_config(
    name = "darwin_toolchain_config",
    cpu = "darwin",
    include_paths = local_include_paths,
    tool_paths = local_tool_paths,
)

cc_toolchain_config(
    name = "k8_toolchain_config",
    cpu = "k8",
    include_paths = vm_include_paths,
    tool_paths = vm_tool_paths,
)

cc_toolchain_suite(
    name = "clang_suite",
    toolchains = {
        "darwin": ":darwin_toolchain",
        "k8": ":k8_toolchain",
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
    toolchain_identifier = "darwin-toolchain",
)

cc_toolchain(
    name = "k8_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":k8_toolchain_config",
    toolchain_identifier = "k8-toolchain",
)
