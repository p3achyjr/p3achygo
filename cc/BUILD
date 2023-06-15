load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

bool_flag(
    name = "is_test",
    build_setting_default = False,
)

bool_flag(
    name = "is_trt",
    build_setting_default = False,
)

config_setting(
    name = "test",
    flag_values = {":is_test": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "trt",
    flag_values = {":is_trt": "true"},
    visibility = ["//visibility:public"],
)
