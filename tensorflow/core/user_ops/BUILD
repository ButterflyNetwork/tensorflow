# User ops.

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

package(
    default_visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/lite/delegates/flex:__pkg__"
    ],
    licenses = ["notice"],  # Apache 2.0
)

# One target for all user ops.
cc_library(
    name = "user_ops_op_lib",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/lite/delegates/flex:__pkg__",
    ],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)
