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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_COORDINATOR_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_COORDINATOR_H_

#include <vector>
#include <string>

#include "include/json/json.h"

#include "tensorflow/compiler/xla/pjrt/execution_plan.h"

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/compiler/xla/statusor.h"

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"

namespace xla {

class ExecutionCoordinator {
 public:
  ExecutionCoordinator() = default;

  void extract_worker_info(const Json::Value& worker,
                           int& gpu_num_per_worker);
  void Init();
  void TransferModuleAndDefCtx(std::vector<std::pair<HloModule::DefContext*,
                               std::unique_ptr<HloModule>>>& def_hlo_pairs);
  void InitRemoteNcclComm(
      ncclUniqueId& nccl_id, const absl::Span<const int64> all_participants,
      int worker_id);

  void TransferHostRawData(
      DAPPLEBuffer* dapple_buf, int worker_id, int sample_idx);

  void TransferToServerHost(
      DAPPLEBuffer* dapple_buf, bool variable, int worker_id, int global_idx);

  void TransferVarsAndData(
    std::vector<DAPPLEBuffer*> dapple_bufs, std::vector<bool>& variables,
    std::vector<int32>& global_indices);

  void TransferVariableArgMap(std::map<int, bool>& var_arg_map, int num_vars,
                              std::map<int, int>& port_map);
  void ExecuteRemotePlan(
      std::vector<std::thread>& dist_workers, const ExecuteOptions& options);
  void DispatchPlan(DistributedPlan* plan);

  void DoRemoteSave(int64 max_to_keep, int64 global_step);

  int num_workers() {
    CHECK(workers_.size() > 0);
    return workers_.size();
  }

  int num_dev_per_worker() const {
    CHECK(workers_.size() > 0);
    CHECK(gpu_ids_.size() % workers_.size() == 0);
    return (gpu_ids_.size() / workers_.size());
  }

  std::map<int, ExecutionPlanHandle>& handle() { return handle_map_; }

 private:
  std::vector<std::pair<std::string/*ip*/, int/*port*/>> workers_;   // first is master
  std::vector<int>  gpu_ids_;      // per worker gpu number: gpu_ids_.size() / workers_.size()
  std::vector<std::unique_ptr<grpc::XlaService::Stub>> service_stubs_;
  std::vector<std::unique_ptr<GRPCStub>> stubs_;
  std::map<int/*worker_id*/, ExecutionPlanHandle> handle_map_;
};

} // namespace xla
#endif //TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_COORDINATOR_H_
