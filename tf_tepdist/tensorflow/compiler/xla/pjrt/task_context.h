/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TASK_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TASK_CONTEXT_H_

#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// Task execution context for one specified task. It maintains the input
// and output DAPPLEBuffer pointers. Some task node may maintain the lifetime
// of newly produced DAPPLEBuffer objects.
class TaskContext {
 public:
  explicit TaskContext(TaskNode* task_node);
  // Moveable, but not copyable.
  TaskContext& operator=(const TaskContext&) = delete;
  TaskContext(const TaskContext&) = delete;

  // Initialize outputs buffers for kComputeTask and kRecvTask
  Status InitializeOutputBuffers(
      Device* virtual_device, PjRtClient* gpu_client, se::Platform* platform,
      int global_device_count);

  // Resolve specified output buffer for current task node
  StatusOr<DAPPLEBuffer*> ResolveOutput(int out_idx);

  void CleanupTaskOutputs();

  // Setter and getter for inputs and outputs buffers.
  std::vector<DAPPLEBuffer*>* mutable_input_buffers();
  std::vector<DAPPLEBuffer*>* mutable_output_buffers();
  std::vector<DAPPLEBuffer*>& input_buffers();
  std::vector<DAPPLEBuffer*>& output_buffers();
  
  const TaskNode* task_node() const { return task_node_; }

 private:
  Status InitializeBuffersOnly(int global_device_count);
  Status InitializeBuffersOnGPU(
      Device* virtual_device, PjRtClient* gpu_client, se::Platform* platform,
      int global_device_count);

  TaskNode* task_node_; // Not owned
  std::vector<DAPPLEBuffer*> input_buffers_;
  std::vector<DAPPLEBuffer*> output_buffers_;
  // Outputs for compute and GA node should be preserved because the task graph
  // does execution according to topology sort. Some node may retreive its
  // parent's outputs as inputs.
  std::vector<std::unique_ptr<DAPPLEBuffer>> task_outputs_; 
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_TASK_CONTEXT_H_
