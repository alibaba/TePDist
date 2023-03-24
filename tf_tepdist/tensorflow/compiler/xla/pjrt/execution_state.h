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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_STATE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_STATE_H_

#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/distributed_checkpoint_utils.h"
#include "tensorflow/compiler/xla/pjrt/task_context.h"
#include "tensorflow/compiler/xla/pjrt/variable_specs.h"
#include "tensorflow/compiler/xla/pjrt/whole_graph_launch_context.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/status.h"

#include <unordered_map>

using namespace dist_ckpt_utils;

namespace xla {

typedef std::vector<std::unique_ptr<DAPPLEBuffer>> RecvTaskOutputs;

// Class for maintaining the entire TaskDAG execution state during
// one iteration, which includes:
// 1. To pre-allocate all task output buffers
// 2. To propagate new-produced outputs to the producer's buffers
// 3. To provide universe interface for preparing task inputs pointers
class ExecutionState {
 public:
  explicit ExecutionState(
      TaskDAG* task_graph,  const std::map<int, std::string>* variable_map,
      bool sharding_across_machine, int local_device_count, int worker_rank,
      int worker_count, const int64 ckpt_max_to_keep);

  // Moveable, but not copyable.
  ExecutionState& operator=(const ExecutionState&) = delete;
  ExecutionState(const ExecutionState&) = delete;

  void InitializeDistributedSaver(
      std::vector<DAPPLEBuffer*>& local_variables, std::map<int, int>& arg_var_map,
      PjRtClient* gpu_client);

  // Create and initialize all TaskContexts of current TaskDAG. The
  // ExecutionState keeps lifetimes of all TaskContexts.
  Status InitializeTaskContexts();

  // Interface of pre-allocating all outputs DAPPLEBuffer for each task. Each
  // task outputs buffer are maintained within their TaskContext.
  Status InitializeAllTaskOutputBuffers(
      Device* virtual_device, PjRtClient* gpu_client, se::Platform* platform);
  
  Status CreateRecvTaskBuffers(
      Device* virtual_device, PjRtClient* gpu_client, se::Platform* platform, const BufferInfo& buffer_info);

  // Initialize all variables sequentially.
  void InitializeLocalVariables(
      std::vector<DAPPLEBuffer*>& local_variables,
      std::map<int, int>& arg_var_map,
      std::map<int, string>& init_specs_map,
      PjRtClient* gpu_client);

  void ResolveLocalVariables(
      absl::Span<DAPPLEBuffer* const> argument_handles,
      std::vector<DAPPLEBuffer*> *local_variables,
      std::map<int, int>* arg_var_map);

  // Propagate the new produced `results` to the producer's outputs buffers
  // in its TaskContext. Such kinds of `results` lifetime should be maintained
  // within their corresponding TaskContext class which can be resolved during
  // the execution TaskDAG.
  StatusOr<std::vector<DAPPLEBuffer*>> PropagateOutputs(
      TaskNode* task_node, ScopedShapedBuffer output_buffers,
      absl::Span<DAPPLEBuffer* const>& input_args,
      std::shared_ptr<BufferSequencingEvent>& definition_event,
      PjRtClient* gpu_client, int local_device_id);

  // Universal interface to prepare inputs DAPPLEBuffer pointers for each task.
  void PrepareInputsForTask(TaskNode* task_node);

  // TaskContext getter. All TaskContexts should be created at the beginning of
  // execution.
  TaskContext* get_task_context(int node_id);

  CheckpointUtil* get_saver() { return ckpt_util_.get(); }
  Status SaveCheckpoint();
  Status RestoreFromCheckpoint();

  // Cleanup all tasks' outputs except GPU buffers of RecvTask
  void CleanupCachedTasksOutputsBuffers();

  // Cleanup DAPPLEBuffers triggered by specified task node
  void CleanupPjRtBuffersTriggeredBy(TaskNode* task_node);

  void set_global_step(const int64 global_step);
  const int64 global_step() const { return global_step_; }

 private:
  void PrepareInputsForCommonTask(TaskNode* task_node);
  void PrepareInputsForSendTask(TaskNode* task_node);
  void PrepareInputsForRecvTask(TaskNode* task_node);

  bool sharding_across_machine_;
  int worker_rank_;
  int worker_count_;

  int local_device_count_;
  int global_device_count_;

  int64 global_step_;
  std::unordered_map<int/*node_id*/,
                     std::unique_ptr<TaskContext>> node_task_ctxs_map_;
  std::unique_ptr<CheckpointUtil> ckpt_util_;
  std::unique_ptr<VariableSpecsMgr> var_spec_mgr_;
  const std::map<int, std::string>* variable_map_; // Not owned
  TaskDAG* task_graph_; // not owned
  std::map<std::pair<int/*device id*/,int/*buffer type*/>, std::vector<RecvTaskOutputs>> recv_dapple_buffer_ptr_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_STATE_H_
