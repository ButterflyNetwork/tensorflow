exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party/mkl:build_defs.bzl",
    "if_mkl",
)
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_openmp_copts",
)
load(
    "@org_tensorflow//third_party/mkl_dnn:build_defs.bzl",
    "if_mkldnn_openmp",
)
load(
    "@org_tensorflow//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
)
load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_WITH_SYCL": "#undef DNNL_WITH_SYCL",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "#undef DNNL_WITH_LEVEL_ZERO",
    "#cmakedefine DNNL_SYCL_CUDA": "#undef DNNL_SYCL_CUDA",
}

_DNNL_RUNTIME_THREADPOOL = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_WITH_SYCL": "#undef DNNL_WITH_SYCL",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "#undef DNNL_WITH_LEVEL_ZERO",
    "#cmakedefine DNNL_SYCL_CUDA": "#undef DNNL_SYCL_CUDA",
}

template_rule(
    name = "dnnl_config_h",
    src = "include/oneapi/dnnl/dnnl_config.h.in",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = select({
        "@org_tensorflow//third_party/mkl_dnn:build_with_mkldnn_openmp": _DNNL_RUNTIME_OMP,
        "//conditions:default": _DNNL_RUNTIME_THREADPOOL,
    }),
)

# Create the file dnnl_version.h with DNNL version numbers.
# Currently, the version numbers are hard coded here. If DNNL is upgraded then
# the version numbers have to be updated manually. The version numbers can be
# obtained from the PROJECT_VERSION settings in CMakeLists.txt. The variable is
# set to "version_major.version_minor.version_patch". The git hash version can
# be set to NA.
# TODO(agramesh1): Automatically get the version numbers from CMakeLists.txt.
template_rule(
    name = "dnnl_version_h",
    src = "include/oneapi/dnnl/dnnl_version.h.in",
    out = "include/oneapi/dnnl/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "2",
        "@DNNL_VERSION_MINOR@": "1",
        "@DNNL_VERSION_PATCH@": "0",
        "@DNNL_VERSION_HASH@": "N/A",
    },
)

_COPTS_LIST = select({
    "@org_tensorflow//tensorflow:windows": [],
    "//conditions:default": ["-fexceptions"],
}) + [
    "-UUSE_MKL",
    "-UUSE_CBLAS",
    "-DDNNL_ENABLE_MAX_CPU_ISA",
] + tf_openmp_copts()

_INCLUDES_LIST = [
    "include",
    "src",
    "src/common",
    "src/cpu",
    "src/cpu/gemm",
    "src/cpu/x64/xbyak",
]

_TEXTUAL_HDRS_LIST = glob([
    "include/**/*",
    "src/common/*.hpp",
    "src/cpu/*.hpp",
    "src/cpu/**/*.hpp",
    "src/cpu/x64/jit_utils/jitprofiling/*.h",
    "src/cpu/x64/xbyak/*.h",
]) + [
    ":dnnl_config_h",
    ":dnnl_version_h",
]

# Large autogen files take too long time to compile with usual optimization
# flags. These files just generate binary kernels and are not the hot spots,
# so we factor them out to lower compiler optimizations in ":dnnl_autogen".
cc_library(
    name = "onednn_autogen",
    srcs = glob(["src/cpu/x64/gemm/**/*_kern_autogen*.cpp"]),
    copts = select({
        "@org_tensorflow//tensorflow:macos": ["-O0"],
        "//conditions:default": ["-O1"],
    }) + ["-U_FORTIFY_SOURCE"] + _COPTS_LIST,
    includes = _INCLUDES_LIST,
    textual_hdrs = _TEXTUAL_HDRS_LIST,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_dnn",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/*.cpp",
            "src/cpu/**/*.cpp",
            "src/cpu/x64/jit_utils/jitprofiling/*.c",
        ],
        exclude = [
            "src/cpu/aarch64/**",
            "src/cpu/x64/gemm/**/*_kern_autogen.cpp",
        ],
    ),
    copts = _COPTS_LIST,
    includes = _INCLUDES_LIST,
    textual_hdrs = _TEXTUAL_HDRS_LIST,
    visibility = ["//visibility:public"],
    deps = [":onednn_autogen"] + if_mkl_ml(
        ["@org_tensorflow//third_party/mkl:intel_binary_blob"],
        [],
    ),
)

cc_library(
    name = "mkl_dnn_aarch64",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/rnn/*.cpp",
        "src/cpu/rnn/*.hpp",
        "src/cpu/matmul/*.cpp",
        "src/cpu/matmul/*.hpp",
        "src/cpu/gemm/**/*",
    ]) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
    ],
    hdrs = glob(["include/*"]),
    copts = [
        "-fexceptions",
        "-UUSE_MKL",
        "-UUSE_CBLAS",
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
    ],
    linkopts = ["-lgomp"],
    visibility = ["//visibility:public"],
)
