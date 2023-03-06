load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "/usr/local/opt/llvm/bin/clang",
        ),
        tool_path(
            name = "ld",
            path = "/usr/local/opt/llvm/bin/lld",
        ),
        tool_path(
            name = "ar",
            path = "/usr/local/opt/llvm/bin/llvm-ar",
        ),
        tool_path(
            name = "cpp",
            path = "/bin/false",
        ),
        tool_path(
            name = "gcov",
            path = "/usr/local/opt/llvm/bin/llvm-gcov",
        ),
        tool_path(
            name = "nm",
            path = "/usr/local/opt/llvm/bin/llvm-nm",
        ),
        tool_path(
            name = "objdump",
            path = "/usr/local/opt/llvm/bin/llvm-objdump",
        ),
        tool_path(
            name = "strip",
            path = "/usr/local/opt/llvm/bin/llvm-strip",
        ),
    ]

    features = [
        # NEW
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-lstdc++",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,  # NEW
        cxx_builtin_include_directories = [
            "/usr/local/opt/llvm/lib/clang/12.0.0/include",
            "/usr/local/Cellar/llvm/12.0.0/lib/clang/12.0.0/include",
            "/usr/local/opt/llvm/include/c++/v1",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include/",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/System/Library/Frameworks",
        ],
        toolchain_identifier = "local",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "darwin",
        target_libc = "unknown",
        compiler = "clang",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
