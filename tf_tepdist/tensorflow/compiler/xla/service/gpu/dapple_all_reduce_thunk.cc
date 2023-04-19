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

#include "tensorflow/compiler/xla/service/gpu/dapple_all_reduce_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/nccl/nccl.h"
#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/refcounting_hash_map.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/core/util/env_var.h"

namespace xla {
namespace gpu {

DAPPLEAllReduceThunk::DAPPLEAllReduceThunk(
    std::vector<BufferAllocation::Slice> src_buffers,
    std::vector<BufferAllocation::Slice> dst_buffers,
    std::vector<int64> element_counts,
    int64 num_replicas, std::unique_ptr<KernelThunk> local_sum_thunk,
    const HloInstruction* hlo)
    : Thunk(Thunk::kDAPPLEAllReduce, hlo),
      num_replicas_(num_replicas),
      src_buffers_(std::move(src_buffers)),
      dst_buffers_(std::move(dst_buffers)),
      element_counts_(std::move(element_counts)),
      local_sum_thunk_(std::move(local_sum_thunk)) {
  CHECK_EQ(hlo_instruction()->operand_count(), src_buffers_.size());
  CHECK_EQ(src_buffers_.size(), dst_buffers_.size());
}

Status DAPPLEAllReduceThunk::Initialize(const GpuExecutable& executable,
                               se::StreamExecutor* executor) {
  if (hlo_instruction()->operand_count() < 2) {
    // No local sum kernel necessary
    return Status::OK();
  }

  return local_sum_thunk_->Initialize(executable, executor);
}

// Figures out which devices (named by their replica-ids) are participating in
// the all-reduce subgroup that contains device_ordinal.
Status DAPPLEAllReduceThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto* hlo = Cast<HloDAPPLEAllReduceInstruction>(hlo_instruction());

  auto element_type = hlo->operand_count() == 1 ? hlo->shape().element_type()
                                                : hlo->operand(0)->shape().element_type();
  TF_ASSIGN_OR_RETURN(ncclDataType_t data_type, ToNcclDataType(element_type));

  VLOG(1) << "Starting DAPPLEAllReduceThunk.";
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  auto reduction =
      Cast<HloDAPPLEAllReduceInstruction>(hlo_instruction())->reduction_type();
  ncclRedOp_t computation;
  if (reduction == "Sum") {
    computation = ncclSum;
  } else if (reduction == "Min") {
    computation = ncclMin;
  } else if (reduction == "Max") {
    computation = ncclMax;
  } else if (reduction == "Prod"){
    computation = ncclProd;
  } else {
    CHECK(0 && "Unhandled reduction");
  }

  VLOG(2) << "nccl allreduce type: " << reduction;

  NcclContext* nccl_ctx = params.nccl_ctx;

  const std::vector<int64>* global_devices;
  int rank;
  const std::vector<int64>* task_id = params.task_id_;
  CHECK(task_id);
  int split_ordinal = hlo->split_ordinal();

  std::shared_ptr<CommDevManager> comm_dev_mgr = params.comm_dev_mgr_;
  CHECK(comm_dev_mgr);
  std::shared_ptr<DevGroup> dev_group = comm_dev_mgr->FindDevGroup(*task_id, split_ordinal);
  VLOG(1) << "NCCL ctx " << nccl_ctx << ", split_ordinal: " << split_ordinal;

  global_devices = &dev_group->global_dev_ids_;

  rank = comm_dev_mgr->GetCommRank(*task_id, split_ordinal);

  int64 local_device_ordinal = params.stream->parent()->device_ordinal();

  VLOG(2) << "on this device, rank is " << rank;
  TF_ASSIGN_OR_RETURN(NcclComm* nccl_comm_ptr,
                      nccl_ctx->GetNcclComm(*global_devices, rank));
  ncclComm_t nccl_comm = nccl_comm_ptr->comm();
  CHECK(nccl_comm && "NCCL communicator uninitialized!");
  VLOG(2) << hlo_instruction()->ToString()
          << " local EXE-dev_id:" << local_device_ordinal 
          << " nccl_comm:" << nccl_comm;

  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      params.stream->implementation()->GpuStreamMemberHack());

  ncclGroupStart();
  for (size_t i = 0; i < src_buffers_.size(); ++i) {
    auto element_count = element_counts_[i];
    void* send_buf =
        params.buffer_allocations->GetDeviceAddress(src_buffers_[i]).opaque();
    void* recv_buf =
        params.buffer_allocations->GetDeviceAddress(dst_buffers_[i]).opaque();
    VLOG(2) << absl::StreamFormat(
        "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
        "reduction_type=%s, comm=%p, stream=%p)",
        send_buf, recv_buf, element_count, reduction,
        static_cast<const void*>(&nccl_comm), cu_stream);

    ncclResult_t result = ncclAllReduce(send_buf, recv_buf,
                                        /*count=*/element_count,
                                        /*datatype=*/data_type,
                                        /*op=*/computation,
                                        /*comm=*/nccl_comm,
                                        /*stream=*/*cu_stream);

    TF_RET_CHECK(ncclSuccess == result)
        << "Failed to perform all-reduce: " << ncclGetErrorString(result);
  }
  ncclGroupEnd();
  VLOG(1) << "Done performing all reduce for ordinal: "
          << local_device_ordinal;

  return Status::OK();
}

DAPPLEAllReduceThunk::~DAPPLEAllReduceThunk() {}

}  // namespace gpu
}  // namespace xla
