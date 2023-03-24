/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/replica_id_thunk.h"

namespace xla {
namespace gpu {

ReplicaOrPartitionIdThunk::ReplicaOrPartitionIdThunk(Kind kind,
    const BufferAllocation::Slice& dest,
    const HloInstruction* instr)
    : Thunk(kind, instr), dest_(dest) {}

Status ReplicaOrPartitionIdThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());

  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);
  const std::vector<int64>* task_id = params.task_id_;
  CHECK(task_id);
  int split_ordinal = hlo_instruction()->split_ordinal();
  std::shared_ptr<CommDevManager> comm_dev_mgr = params.comm_dev_mgr_;
  CHECK(comm_dev_mgr);
  int replica_or_partition_id = comm_dev_mgr->GetCommRank(*task_id, split_ordinal);
  VLOG(2) << "replica_or_partition_id=" << replica_or_partition_id;
  params.stream->ThenMemset32(&dest_addr, replica_or_partition_id, /*size=*/4);
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
