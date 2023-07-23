#### Forked from minigo ####
#### https://github.com/tensorflow/minigo/blob/master/cc/tensorflow/BUILD ####
# load("@org_tensorflow//tensorflow:tensorflow.bzl", "transitive_hdrs")

cc_library(
    name = "tensorflow",
    srcs = glob(["include/**/*.so.2"]) + glob([
        "libtensorflow_cc.so.*",  # not in CPU builds.
    ]) + [
        "libtensorflow_framework.so.2",
    ],
    hdrs = glob(
        ["include/**/*"],
        exclude = [
            "**/*.so.2",
            # "include/absl/**/*",
        ],
    ),
    includes = [
        "include",
        "include/external/nsync/public",
        "include/third_party/eigen3",
    ],
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
)
