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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_VIRTUAL_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_VIRTUAL_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/pjrt/tracked_device_buffer.h"
#include "tensorflow/compiler/xla/service/cluster_and_device_spec.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/pjrt/execution_state.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/compiler/xla/pjrt/execution_coordinator.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/task_context.h"
#include "tensorflow/compiler/xla/pjrt/whole_graph_launch_context.h"
#include "tensorflow/compiler/xla/pjrt/nccl_context.h"
#include "third_party/nccl/nccl.h"

namespace xla {

class VirtualClient;
using DAPPLEBufferHandle = int64;

struct MicroTaskSchedule {
  MicroTaskSchedule(std::size_t micro_id, std::size_t stage_id, float finish_time)
  : micro_id_(micro_id)
  , stage_id_(stage_id)
  , finish_time_(finish_time) {
  }
  std::size_t micro_id_;
  std::size_t stage_id_;
  float       finish_time_;
};

class VirtualClient : public std::enable_shared_from_this<VirtualClient> {
 public:
  // `allocator` may null, in which case the platform default allocator is used.
  explicit VirtualClient(std::string platform_name, int host_id,
      std::shared_ptr<PjRtClient> cpu_client,
      std::shared_ptr<PjRtClient> gpu_client);

  void InitializeExecutionState(
      TaskDAG* task_graph, const std::map<int, std::string>* variable_map,
      int worker_rank);

  virtual ~VirtualClient() = default;

  std::unique_ptr<TaskDAG> BuildTaskDAG(std::shared_ptr<DistributedPlan> dist_plan);

  std::unique_ptr<TaskDAG> CompileTaskDAG(
      std::vector<std::pair<HloModule::DefContext*,
                            std::unique_ptr<HloModule>>>& def_hlo_pairs,
      int num_nodes = 1);

  std::unique_ptr<TaskDAG> CompileTaskDAG(
      std::vector<std::pair<HloModule::DefContext*,
                            HloModule*>>& def_hlo_pairs, int num_nodes = 1);

  std::shared_ptr<LocalPlan> BuildLocalPlan(std::shared_ptr<DistributedPlan> dist_plan);

  std::shared_ptr<DistributedPlan> BuildDistributedPlanRPC(
      int num_workers, int worker_rank, std::vector<ComputeTask>& compute_tasks,
      const std::vector<int>& split_nums, const std::vector<bool>& share_dev_flags,
      const std::vector<int>& placement_layout, int stage_split_ordinal);

  std::shared_ptr<DistributedPlan> BuildDistributedPlan(
        std::vector<std::pair<HloModule::DefContext*,
                              std::unique_ptr<HloModule>>>& def_hlo_pairs,
        int num_nodes, int rank);

  StatusOr<std::vector<std::unique_ptr<DAPPLEBuffer>>> ExecutePlan(
      std::vector<std::unique_ptr<Executable>> executables);

  Status CreateAndInitNcclContext(LocalPlan* plan);
  gpu::NcclContext* GetOrCreateNcclContext();

  gpu::NcclContext* GetNcclContext();
  ExecutionState* execution_state();
  WholeGraphLaunchContext* whole_graph_launch_context();

  int host_id() const { return host_id_; }
  const std::string& platform_name() const { return platform_name_; }
  // A dummy interface to make *JAX* happy.
  int device_count() const { return 1; }
  Device* virtual_device() { return virtual_device_.get(); }

  std::shared_ptr<PjRtClient> gpu_client() const { return gpu_client_; }
  std::shared_ptr<PjRtClient> cpu_client() const { return cpu_client_; }

  ExecutionCoordinator& coordinator() { return coord_; }
  std::vector<std::pair<HloModule::DefContext*,
                        std::unique_ptr<HloModule>>>& rpc_def_hlo_map() {
    return rpc_def_hlo_map_;
  }

  const bool warm_up() const { return warm_up_; }

  void set_warm_up(bool warm_up) { warm_up_ = warm_up; }

  std::map<int, bool>& var_arg_map() {
    return var_arg_map_;
  }
  void set_var_arg_map(std::map<int, bool> var_map) {
    var_arg_map_ = std::move(var_map);
  }

  std::map<int, int>& var_gtol_idx_map() {
    return var_gtol_idx_map_;
  }
  std::map<int, int>* mutable_var_gtol_idx_map() {
    return &var_gtol_idx_map_;
  }

  std::map<int, int>& var_local_idx_map() {
    return var_local_idx_map_;
  }
  std::map<int, int>* mutable_var_local_idx_map() {
    return &var_local_idx_map_;
  }

  std::map<int, int>& input_local_idx_map() {
    return input_local_idx_map_;
  }
  std::map<int, int>* mutable_input_local_idx_map() {
    return &input_local_idx_map_;
  }

  int worker_count() const {
    return worker_count_;
  }

  bool sharding_across_machine() const {
    return sharding_across_machine_;
  }
  const int64 get_ckpt_max_to_keep() const { return ckpt_max_to_keep_; }
  void set_ckpt_max_to_keep(const int64 max_to_keep) {
    ckpt_max_to_keep_ = max_to_keep;
  }

  ClusterSpec& cluster_spec() { return cluster_spec_; }

 private:
  std::unordered_set<const HloInstruction*> FindAllCollectiveCommInstructions(
      const TaskDAG* task_graph);
  std::unordered_set<const HloInstruction*> CollectCommInstructions(
      const HloModule* module);
  Status CreateNcclCliqueForCollectiveInstructions(
      const std::unordered_set<const HloInstruction*>& coll_insts,
      std::shared_ptr<CommDevManager> comm_dev_mgr);
  Status CreateNcclCliqueForSendRecvTask(
      const TaskDAG* task_graph,
      std::shared_ptr<CommDevManager> comm_dev_mgr);

  const std::string platform_name_;
  const int host_id_;
  std::shared_ptr<PjRtClient> cpu_client_;
  std::shared_ptr<PjRtClient> gpu_client_;
  std::unique_ptr<Device> virtual_device_;
  int worker_count_ = 0;
  int worker_rank_ = 0;
  bool sharding_across_machine_ = false;

  std::unique_ptr<gpu::NcclContext> nccl_ctx_ = nullptr;
  std::unique_ptr<ExecutionState> exec_state_ = nullptr;
  std::unique_ptr<WholeGraphLaunchContext> whole_graph_launch_context_ = nullptr;

  bool warm_up_ = true;

  int64 ckpt_max_to_keep_ = -1;

  ExecutionCoordinator coord_;
  std::vector<std::pair<HloModule::DefContext*,
                        std::unique_ptr<HloModule>>> rpc_def_hlo_map_;

  std::map<int, bool> var_arg_map_;
  std::map<int, int> var_gtol_idx_map_;
  std::map<int, int> var_local_idx_map_;
  std::map<int, int> input_local_idx_map_;
  
  ClusterSpec cluster_spec_;
};

class DAPPLEExecutable {
 public:
  static StatusOr<std::unique_ptr<DAPPLEExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options,
      VirtualClient* virtual_client);

  explicit DAPPLEExecutable(
      bool parameter_is_tupled_arguments,
      VirtualClient* virtual_client,
      std::vector<std::unique_ptr<LocalExecutable>>& executables);

  explicit DAPPLEExecutable(
      bool parameter_is_tupled_arguments,
      VirtualClient* virtual_client,
      std::shared_ptr<LocalPlan> plan);

  ~DAPPLEExecutable() {
    for (auto it : sharded_result_buffers_) {
      for (auto ptr : it.second) {
        delete ptr;
      }
    }
    sharded_result_buffers_.clear();
  }

  Status InitializeGABuffers(
      std::vector<std::unique_ptr<PjRtBuffer>>& ga_buffers,
      Device* gpu_device, const HloModule& module) const;

  void BuildPlanOutputs(
      TaskNode* merge_task, int output_count,
      std::vector<std::unique_ptr<DAPPLEBuffer>>& plan_outputs);

  void BuildPlanOutputsTF(
      TaskNode* merge_task, int output_count,
      std::vector<DAPPLEBuffer*>& plan_outputs,
      DistributedPlan* dist_plan = nullptr,
      std::map<int, int>* port_map = nullptr);

  StatusOr<std::vector<std::unique_ptr<DAPPLEBuffer>>> ExecutePlan(
      absl::Span<DAPPLEBuffer* const> arguments,
      const ExecuteOptions& options);

  StatusOr<std::vector<DAPPLEBuffer*>> ExecuteRemotePlan(const ExecuteOptions& options);

  StatusOr<std::vector<std::unique_ptr<DAPPLEBuffer>>> ExecuteDistributedPlan(
      absl::Span<DAPPLEBuffer* const> arguments,
      const ExecuteOptions& options);

  StatusOr<std::vector<DAPPLEBuffer*>> ExecuteRPCPlan(
      absl::Span<DAPPLEBuffer* const> arguments,
      const ExecuteOptions& options);

  StatusOr<std::vector<std::unique_ptr<DAPPLEBuffer>>> Execute(
      absl::Span<DAPPLEBuffer* const> arguments,
      const ExecuteOptions& options);

  void DoInputTask(TaskContext* task_ctx, TaskNode* task,
                   int64 local_dev_id, LocalPlan* plan);

  Status DoComputeTask(
      TaskContext* task_ctx, const ExecuteOptions& options, TaskNode* task,
      int64 local_dev_id, LocalPlan* plan);

  int Preprocess(TaskNode* split_task);
  Status ExecuteHostRecvTask(const TaskNode* split_task);

  void DoRPCSend(TaskNode* task, bool xfer_var, 
                 std::map<int, bool>& variable_arg);
  Status DoSendTask(TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
                  LocalPlan* plan, se::Stream* stream);

  void DoHostRecv(TaskContext* task_ctx, TaskNode* task, LocalPlan* plan);

  DAPPLEBuffer* ResolveRPCVariable(int var_idx);

  Status DoRecvTask(TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
                    LocalPlan* plan, se::Stream* stream);

  // Both cross workers and intra-worker
  Status DoGPURecv(TaskContext* task_ctx, TaskNode* task,
                   int64 local_dev_id, LocalPlan* plan,
                   se::Stream* stream);

  Status DoNcclSendRecv(DAPPLEBuffer* dapple_buf, const std::vector<int64>& send_recv_devices,
                        se::Stream* stream, const ncclComm_t nccl_comm, bool is_send,
                        bool need_sync=false);

  void DoARTask(
      TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
      LocalPlan* plan, se::Stream* stream);

  void DoGATask(TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
                LocalPlan* plan, const ExecuteOptions& options);

  void DoGAInitTask(TaskContext* task_ctx, TaskNode* task,
                    int64 local_dev_id, LocalPlan* plan);

  void DoOutputTask(TaskContext* task_ctx, TaskNode* task,
                    int64 local_dev_id, LocalPlan* plan);

  void ExecuteTaskList(
      int64 local_dev_id, int64 device_count, ExecutionState* exec_state,
      const ExecuteOptions& options, LocalPlan* plan);

  void ExecuteDistributed(
      absl::Span<DAPPLEBuffer* const> arguments, int64 dev_id,
      const ExecuteOptions& options, void* nccl_id,
      void* nccl_comms);

  void NcclAllReduce(int device_id, se::Stream* stream,
      const ncclComm_t& nccl_comm,
      std::vector<DAPPLEBuffer*>& dapple_bufs);

  void NcclAllReduce(int device_id, se::Stream* comm_stream,
      const ncclComm_t& nccl_comm, ScopedShapedBuffer* result_buffer) const;

  void PrepareParamsDistributed(
          absl::Span<DAPPLEBuffer* const> arguments,
          int num_gpus, int num_replicas) const;

  static StatusOr<std::unique_ptr<Literal>> ConvertOutputToLiteral(
      std::vector<DAPPLEBuffer*>& dapple_bufs,
      int64 start_idx, int64 range);

  VirtualClient* virtual_client() const { return virtual_client_; }
  std::shared_ptr<LocalPlan> plan() const { return plan_; }
  const bool fake_input() const { return fake_input_; }

  void set_split_nums(const std::vector<int>& split_nums) {
    split_nums_ = split_nums;
  }
  void set_share_dev_flags(const std::vector<bool>& share_dev_flags) {
    share_dev_flags_ = share_dev_flags;
  }
  void set_stage_split_ordinal(int stage_split_ordinal) {
    stage_split_ordinal_ = stage_split_ordinal;
  }

  std::vector<DAPPLEBuffer*>* mutable_input_data_args() {
    return &input_data_ptrs_cached_;
  }

 private:
  Status SetUpDonation(PjRtClient* client, bool tuple_inputs);

  StatusOr<std::vector<DAPPLEBuffer*>> ExecuteTask(
      absl::Span<DAPPLEBuffer* const> arguments,
      const ExecuteOptions& options, int64 local_dev_id, TaskNode* task,
      std::vector<std::unique_ptr<PjRtBuffer>>* ga_buffers = nullptr) const;

  StatusOr<std::vector<DAPPLEBuffer*>> ExecuteRPCTask(
      absl::Span<DAPPLEBuffer* const> arguments,
      const ExecuteOptions& options, int64 local_dev_id, TaskNode* task,
      std::vector<std::unique_ptr<PjRtBuffer>>* ga_buffers = nullptr) const;

  StatusOr<ScopedShapedBuffer> EnqueueExecution(
      absl::Span<DAPPLEBuffer* const> arguments,
      const ExecuteOptions& options, int64 local_dev_id, int replica_id,
      std::vector<std::unique_ptr<PjRtBuffer>>& ga_buffers,
      std::vector<PjRtBuffer::ScopedHold>* device_buffers) const;

  StatusOr<ScopedShapedBuffer> EnqueueExecution(
      absl::Span<DAPPLEBuffer* const> arguments,
      const ExecuteOptions& options, int64 local_dev_id, TaskNode* task,
      std::vector<std::unique_ptr<PjRtBuffer>>* ga_buffers,
      std::vector<PjRtBuffer::ScopedHold>* device_buffers) const;

  PjRtClient* the_client() const {
    return virtual_client_->gpu_client().get();
  }

  // True if the executables were compiled expecting arguments in a single
  // tuple.
  const bool parameter_is_tupled_arguments_;
  VirtualClient* const virtual_client_;

  std::vector<std::unique_ptr<LocalExecutable>> executables_;
  LocalExecutable* entry_exe_ = nullptr;
  LocalExecutable* ga_exe_ = nullptr;

  std::shared_ptr<LocalPlan> plan_;

  absl::flat_hash_set<int> parameters_that_must_be_donated_;
  std::map<int64/*output no*/, 
           std::vector<ScopedShapedBuffer*>> sharded_result_buffers_;
  std::map<int64/*output no*/, int64> sharded_dim_to_slice_;
  bool rpc_task_ = false;
  bool fake_input_ = false;
  std::vector<DAPPLEBuffer*> input_data_ptrs_cached_;

  std::vector<int> split_nums_;
  std::vector<bool> share_dev_flags_;
  int stage_split_ordinal_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_VIRTUAL_CLIENT_H_
