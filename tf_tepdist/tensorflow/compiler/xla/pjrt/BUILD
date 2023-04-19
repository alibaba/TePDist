load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "worker_thread",
    srcs = ["worker_thread.cc"],
    hdrs = ["worker_thread.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "event_pool",
    srcs = ["event_pool.cc"],
    hdrs = ["event_pool.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "semaphore",
    srcs = ["semaphore.cc"],
    hdrs = ["semaphore.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":semaphore",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "initializers_test",
    srcs = ["initializers_test.cc"],
    deps = [
        ":initializers",
        ":slice_utils",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "slice_utils_test",
    srcs = ["slice_utils_test.cc"],
    deps = [
        ":slice_utils",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "tracked_device_buffer",
    srcs = ["tracked_device_buffer.cc"],
    hdrs = ["tracked_device_buffer.h"],
    deps = [
        ":event_pool",
        ":local_device_state",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//tensorflow/stream_executor:event",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "tracked_device_buffer_test",
    srcs = ["tracked_device_buffer_test.cc"],
    deps = [
        ":tracked_device_buffer",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor:device_memory",
        "//tensorflow/stream_executor:device_memory_allocator",
    ],
)

cc_library(
    name = "local_device_state",
    srcs = ["local_device_state.cc"],
    hdrs = ["local_device_state.h"],
    deps = [
        ":event_pool",
        ":semaphore",
        ":worker_thread",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "//tensorflow/stream_executor:event",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "execution_coordinator",
    srcs = ["execution_coordinator.cc"],
    hdrs = ["execution_coordinator.h"],
    deps = [
        ":dapple_buffer",
        ":execution_plan",
        "//tensorflow/compiler/xla/service/gpu:only_nccl_utils",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/client", 
        "//tensorflow/compiler/xla/rpc:grpc_stub",
    ],
)

cc_library(
    name = "initializers",
    srcs = ["initializers.cc"],
    hdrs = ["initializers.h"],
    deps = [
        "//tensorflow/core/framework:tensor_types",
        "//tensorflow/core/util:work_sharder",
        "//tensorflow/core/lib/random:fill_philox",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:rng_distribution_config_proto_cc",
    ],
)

cc_library(
    name = "slice_utils",
    srcs = ["slice_utils.cc"],
    hdrs = ["slice_utils.h"],
    deps = [
        "//tensorflow/compiler/xla/service:dist_spec",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "execution_plan",
    srcs = ["execution_plan.cc",
            "task_graph.cc"],
    hdrs = ["execution_plan.h",
            "task_graph.h"],
    deps = [
        ":dev_id_util",
        ":task_scheduler",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/client:local_executable",
    ],
)

cc_library(
    name = "lifetime_tracker",
    srcs = ["lifetime_tracker.cc", "execution_plan.h", "task_graph.h", "task_scheduler.h"],
    hdrs = [
        "lifetime_tracker.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/service:executable",
    ],
)

cc_library(
    name = "task_scheduler",
    srcs = ["task_scheduler.cc", "execution_plan.h", "task_graph.h"],
    hdrs = [
        "task_scheduler.h",
    ],
    deps = [
        ":lifetime_tracker",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:service_env",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
    ],
)

cc_library(
    name = "dapple_buffer_utils",
    srcs = ["dapple_buffer_utils.cc"],
    hdrs = ["dapple_buffer_utils.h"],
    deps = [
        ":dapple_buffer",
        ":pjrt_client",
        ":slice_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
    ],
)

cc_library(
    name = "dapple_buffer",
    srcs = ["dapple_buffer.cc"],
    hdrs = ["dapple_buffer.h"],
    deps = [
        ":pjrt_client",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
    ],
)

cc_library(
    name = "whole_graph_launch_context",
    srcs = ["whole_graph_launch_context.cc"],
    hdrs = ["whole_graph_launch_context.h"],
    deps = [
        ":dapple_buffer",
        "//tensorflow/compiler/xla/service:allocation_tracker",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo",

    ],
)

cc_library(
    name = "task_context",
    srcs = ["task_context.cc"],
    hdrs = ["task_context.h"],
    deps = [
        ":dapple_buffer",
        ":execution_plan",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla:statusor",
    ],
)

cc_library(
    name = "execution_state",
    srcs = ["execution_state.cc"],
    hdrs = ["execution_state.h"],
    deps = [
        ":dapple_buffer",
        ":dapple_buffer_utils",
        ":distributed_checkpoint_utils",
        ":execution_plan",
        ":initializers",
        ":task_context",
        ":variable_specs",
        ":whole_graph_launch_context",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla:statusor",
    ],
)

cc_library(
    name = "distributed_checkpoint_utils",
    srcs = ["distributed_checkpoint_utils.cc"],
    hdrs = ["distributed_checkpoint_utils.h"],
    deps = [
        ":dapple_buffer",
        ":dapple_buffer_utils",
        ":variable_specs",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

cc_library(
    name = "variable_specs",
    srcs = ["variable_specs.cc"],
    hdrs = ["variable_specs.h"],
    deps = [
        ":dapple_buffer",
        ":execution_plan",
        ":slice_utils",
        "//tensorflow/compiler/tf2xla:common",
    ],
)

cc_library(
    name = "nccl_comm",
    hdrs = ["nccl_comm.h"],
    deps = [
        ":nccl_unique_group_key",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/strings:str_format",
    ] + if_cuda(["@local_config_nccl//:nccl"]),
)

cc_library(
    name = "nccl_unique_group_key",
    srcs = ["nccl_unique_group_key.cc"],
    hdrs = ["nccl_unique_group_key.h"],
    deps = [
        "//tensorflow/compiler/xla/service/gpu:only_nccl_utils",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/strings:str_format",
    ] + if_cuda(["@local_config_nccl//:nccl"]),
)

cc_library(
    name = "nccl_context",
    srcs = ["nccl_context.cc"],
    hdrs = ["nccl_context.h"],
    deps = [
        ":dev_id_util",
        ":nccl_comm",
        ":nccl_unique_group_key",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:cluster_and_device_spec",
        "@com_google_absl//absl/container:node_hash_map",
    ] + if_cuda(["@local_config_nccl//:nccl"]),
)

cc_library(
    name = "dev_id_util",
    srcs = ["dev_id_util.cc"],
    hdrs = ["dev_id_util.h"],
    deps = [
        "//tensorflow/core/platform:logging",
    ],
)

cc_library(
    name = "virtual_client",
    srcs = ["virtual_client.cc"],
    hdrs = ["virtual_client.h"],
    visibility = ["//tensorflow/compiler/xla:friends"],
    deps = [
        ":nccl_context",
        ":dapple_buffer",
        ":dapple_buffer_utils",
        ":pjrt_client",
        ":execution_coordinator",
        ":execution_plan",
        ":execution_state",
        ":event_pool",
        ":local_device_state",
        ":task_context",
        ":tracked_device_buffer",
        ":nccl_comm",
        ":nccl_unique_group_key",
        "//tensorflow/compiler/xla/service/gpu:only_nccl_utils",
        ":whole_graph_launch_context",
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/pjrt/distributed:protocol_proto_cc",
        "//tensorflow/compiler/xla/service:allocation_tracker",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:service_env",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/core:allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pjrt_client",
    srcs = ["pjrt_client.cc"],
    hdrs = ["pjrt_client.h"],
    visibility = ["//tensorflow/compiler/xla:friends"],
    deps = [
        ":event_pool",
        ":device",
        ":local_device_state",
        ":tracked_device_buffer",
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client:local_executable",
        "//tensorflow/compiler/xla/pjrt/distributed:protocol_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/core:allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor/host:host_platform_id",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "virtual_device",
    srcs = ["virtual_device.cc"],
    hdrs = ["virtual_device.h"],
    copts = if_cuda(["-DNCCL_ENABLED=1"]),
    deps = [
        ":cpu_device",
        ":nvidia_gpu_device",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:memory_space_assignment",
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/core/common_runtime:bfc_allocator",
        "//tensorflow/stream_executor:tf_allocator_adapter",
        "//tensorflow/core/common_runtime/gpu:gpu_mem_allocator",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device",
    hdrs = ["device.h"],
)

cc_library(
    name = "cpu_device",
    srcs = ["cpu_device.cc"],
    hdrs = ["cpu_device.h", "device.h"],
    deps = [
        ":pjrt_client",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:memory_space_assignment",
        "//tensorflow/compiler/xla/service:platform_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "nvidia_gpu_device",
    srcs = ["nvidia_gpu_device.cc"],
    hdrs = ["nvidia_gpu_device.h"],
    copts = if_cuda(["-DNCCL_ENABLED=1"]),
    deps = [
        ":pjrt_client",
        ":virtual_client",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/common_runtime:bfc_allocator",
        "//tensorflow/core/common_runtime/gpu:gpu_mem_allocator",
        "//tensorflow/stream_executor:tf_allocator_adapter",
    ] + if_cuda(["@local_config_nccl//:nccl"]),
)

tf_cc_test(
    name = "gpu_multistream_test",
    srcs = ["gpu_multistream_test.cc"],
    tags = [
        # TODO(phawkins): figure out TF test infra such that this only runs under GPU.
        "no_oss",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":nvidia_gpu_device",
        ":pjrt_client",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:random",
    ],
)
