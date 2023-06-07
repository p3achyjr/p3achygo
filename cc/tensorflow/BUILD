#### Forked from minigo ####
#### https://github.com/tensorflow/minigo/blob/master/cc/tensorflow/BUILD ####
# load("@org_tensorflow//tensorflow:tensorflow.bzl", "transitive_hdrs")

cc_library(
    name = "tensorflow",
    srcs = [
        "libtensorflow_cc.so.2",
        "libtensorflow_framework.so.2",
    ],
    hdrs = glob(
        ["include/**/*"],
        exclude = [
            "*.so",
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

# sh_binary(
#     name = "build",
#     srcs = ["copy_outputs.sh"],
#     data = [
#         ":tensorflow_hdrs",
#         "@org_tensorflow//tensorflow:tensorflow_cc",
#         "@org_tensorflow//tensorflow:tensorflow_framework",
#         "@org_tensorflow//third_party/eigen3:eigen_third_party_header_files",
#     ],
#     visibility = ["//visibility:public"],
# )

# transitive_hdrs(
#     name = "tensorflow_hdrs",
#     deps = [
#         "@com_google_protobuf//:protobuf_headers",
#         "@eigen_archive//:eigen3",
#         "@org_tensorflow//tensorflow/c:c_api",
#         "@org_tensorflow//tensorflow/c:c_api_experimental",
#         "@org_tensorflow//tensorflow/c/eager:c_api",
#         "@org_tensorflow//tensorflow/cc:headers",
#         "@org_tensorflow//tensorflow/core:tensorflow",
#     ],
# )
