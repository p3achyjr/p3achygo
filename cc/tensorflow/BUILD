#### Forked from minigo ####
#### https://github.com/tensorflow/minigo/blob/master/cc/tensorflow/BUILD ####

cc_library(
    name = "tensorflow",
    srcs = [
        "libtensorflow_cc.so",
        "libtensorflow_framework.so",
    ],
    hdrs = glob(["include/tensorflow/**/*"]),
    includes = [
        "include",
        "include/external/nsync/public",
        "include/third_party/eigen3",
    ],
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
)
