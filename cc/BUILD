load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

bool_flag(
    name = "is_leaf_evaluator_dummy",
    build_setting_default = False,
)

bool_flag(
    name = "is_nn_interface_dummy",
    build_setting_default = False,
)

bool_flag(
    name = "is_probability_dummy",
    build_setting_default = False,
)

bool_flag(
    name = "is_cpu_only",
    build_setting_default = False,
)

config_setting(
    name = "leaf_evaluator_dummy",
    flag_values = {":is_leaf_evaluator_dummy": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "nn_interface_dummy",
    flag_values = {":is_nn_interface_dummy": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "probability_dummy",
    flag_values = {":is_probability_dummy": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "cpu_only",
    flag_values = {":is_cpu_only": "true"},
    visibility = ["//visibility:public"],
)
