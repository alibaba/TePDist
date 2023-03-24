load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("//tensorflow:tensorflow.bzl", "if_nccl")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library_cc",
)
load(
    "//tensorflow/compiler/xla:xla.bzl",
    "xla_py_grpc_library",
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

tf_proto_library_cc(
    name = "xla_service_proto",
    srcs = ["xla_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    protodeps = [
        "//tensorflow/compiler/xla:xla_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc_service_gpu_library",
    srcs = ["grpc_service_gpu.cc"],
    copts = [
        "-g",
    ],
    deps = [
        ":grpc_service",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "@com_google_absl//absl/strings:str_format",
        tf_grpc_cc_dependency(),
    ] + if_cuda_is_configured([
        "//tensorflow/core:cuda",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
    ]) + if_nccl(["@local_config_nccl//:nccl"]),
)

cc_library(
    name = "run_arbitary_hlo_library",
    srcs = ["run_arbitary_hlo.cc"],
    copts = [
        "-g",
    ],
    deps = [
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "@com_google_absl//absl/strings:str_format",
        tf_grpc_cc_dependency(),
    ] + if_cuda_is_configured([
        "//tensorflow/core:cuda",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
    ]) + if_nccl(["@local_config_nccl//:nccl"]),
)

cc_library(
    name = "grpc_stub",
    srcs = ["grpc_stub.cc"],
    hdrs = ["grpc_stub.h"],
    copts = [
        "-g",
    ],
    deps = [
        ":xla_service_proto_cc",
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)

cc_library(
    name = "grpc_service_main_library",
    srcs = ["grpc_service_main.cc"],
    deps = [
        ":grpc_service",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings:str_format",
        tf_grpc_cc_dependency(),
    ],
)

tf_cc_binary(
    name = "grpc_service_main_cpu",
    deps = [
        ":grpc_service_main_library",
        "//tensorflow/compiler/xla/service:cpu_plugin",
    ],
)

tf_cc_binary(
    name = "grpc_service_gpu",
    deps = [
        ":grpc_service_gpu_library",
        "//tensorflow/core:cuda_libdevice_path",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "@local_config_nccl//:nccl",
    ] + if_cuda_is_configured([
        "//tensorflow/core:cuda",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
    ]),
    linkopts = ["-L/usr/local/lib/ -lCbcSolver -lCbc -lOsiCbc -lOsiClp -lCoinUtils"],
)

tf_cc_binary(
    name = "run_arbitary_hlo",
    deps = [
        ":run_arbitary_hlo_library",
        "//tensorflow/core:cuda_libdevice_path",
        "//tensorflow/compiler/xla/service/gpu:gpu_compiler",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "@local_config_nccl//:nccl",
    ] + if_cuda_is_configured([
        "//tensorflow/core:cuda",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
    ]),
    linkopts = ["-L/usr/local/lib/ -lCbcSolver -lCbc -lOsiCbc -lOsiClp -lCoinUtils"],
)

pkg_tar(
    name = "grpc_service_pkg",
    srcs = [":grpc_service_gpu"],
    mode = "0755",
    package_dir = "/usr/bin",
)

tf_cc_test(
    name = "grpc_client_test",
    srcs = ["grpc_client_test.cc"],
    data = [
        "//tensorflow/compiler/xla/rpc:grpc_service_main_cpu",
    ],
    deps = [
        ":grpc_stub",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_util",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings:str_format",
        tf_grpc_cc_dependency(),
    ],
)

cc_library(
    name = "grpc_service",
    srcs = ["grpc_service.cc"],
    hdrs = ["grpc_service.h"],
    copts = [
        "-g",
    ],
    deps = [
        ":xla_service_proto_cc",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/compiler/xla/client:executable_build_options",
        tf_grpc_cc_dependency(),
    ],
)
