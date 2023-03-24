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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_WHOLE_GRAPH_LAUNCH_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_WHOLE_GRAPH_LAUNCH_CONTEXT_H_

#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/allocation_tracker.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

using DAPPLEBufferHandle = int64;

struct SplitRecvArgsRecorder {
  std::vector<GlobalDataHandle> variable_data_handles;
  std::vector<DAPPLEBufferHandle> variable_buf_handles;
  std::unordered_map<int/*g_idx*/, std::pair<bool/*variable*/, int/*local idx*/>> recv_args_storage_map;
  void RecordVariableHandle(
      GlobalDataHandle& data_handle, DAPPLEBufferHandle& buf_handle, int global_idx);
  void RecordSampleIndex(int global_idx);
};

// The class for Managing the lifetimes of trainable variables and final outputs.
// Although the actual PjRtBuffers of trainable variables and final outputs are
// different for each iteration, their DAPPLEBuffers should be maintain through
// the entire training progress as placeholders and should be pre-allocated
// before the first iteraion.
class WholeGraphLaunchContext {
 public:
  explicit WholeGraphLaunchContext(int local_dev_count);
  // Moveable, but not copyable.
  WholeGraphLaunchContext& operator=(const WholeGraphLaunchContext&) = delete;
  WholeGraphLaunchContext(const WholeGraphLaunchContext&) = delete;

  // Register and resolver for trainable variables
  Status RegisterVariable(DAPPLEBufferHandle handle, Literal literal);
  Status RegisterVariable(DAPPLEBufferHandle handle, const Shape& shape);
  StatusOr<DAPPLEBuffer*> Resolve(const DAPPLEBufferHandle handle);
  StatusOr<std::vector<DAPPLEBuffer*>> Resolve(
      const std::vector<DAPPLEBufferHandle>& handles);

  // Resolve sample inputs DAPPLEBuffer. The buffer is created when it is first
  // called and then reused every iteration.
  StatusOr<DAPPLEBuffer*> ResolveSampleInputBuffer(
      int64 idx, const Shape& shape, Device* virtual_device);

  StatusOr<DAPPLEBuffer*> ResolveSampleInputBuffer(int64 idx);

  // Create DAPPLEBuffer for final outputs
  Status InitializePlanOutputsBuffers(HloModule* entry_module);
  StatusOr<DAPPLEBuffer*> ResolveOutput(int64 idx);

  // Release host memory for trainable variables. This function should be called
  // when all variables data have been copied on GPU.
  void CleanupVariablesOnHost();

  const int64 num_vars() const { return num_vars_; }
  void set_num_vars(int num_vars) { num_vars_ = num_vars; }
  void set_worker_count(int worker_count) { worker_count_ = worker_count; }
  void set_sharding_across_machine(bool sharding_across_machine) {
    sharding_across_machine_ = sharding_across_machine;
  }
  bool sharding_across_machine() const {
    return sharding_across_machine_;
  }

  SplitRecvArgsRecorder* mutable_recv_args_recorder() { return &recv_args_recorder_; }
  const SplitRecvArgsRecorder& recv_args_recorder() const { return recv_args_recorder_; }

  void set_allocation_tracker(AllocationTracker* tracker) {
    service_allocation_tracker_ = tracker;
  }

  AllocationTracker* allocation_tracker() const { return service_allocation_tracker_; }

 private:
  Status DoRegisterDAPPLEBuffer(
      DAPPLEBufferHandle handle, std::unique_ptr<DAPPLEBuffer> dapple_buf,
      Literal literal);
  Status DoRegisterDAPPLEBuffer(
      DAPPLEBufferHandle handle, std::unique_ptr<DAPPLEBuffer> dapple_buf);

  int local_dev_count_;
  int worker_count_ = 0;
  bool sharding_across_machine_ = false;
  int num_vars_ = 0;

  SplitRecvArgsRecorder recv_args_recorder_;
  // Owns the trainable variables, inputs and outputs.
  std::unordered_map<int/*idx*/, std::unique_ptr<DAPPLEBuffer>> sample_inputs_;
  std::vector<std::unique_ptr<DAPPLEBuffer>> resource_variables_;
  std::vector<std::unique_ptr<DAPPLEBuffer>> outputs_;
  // Preserve the lifetime of literals for trainable variables, since
  // DAPPLEBuffer's `raw` points to contents of literal but not own it.
  std::vector<Literal> literals_;
  AllocationTracker* service_allocation_tracker_ = nullptr; // Not owned
}; 

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_WHOLE_GRAPH_LAUNCH_CONTEXT_H_
