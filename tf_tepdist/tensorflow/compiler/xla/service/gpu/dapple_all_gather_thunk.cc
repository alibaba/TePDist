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

#include "tensorflow/compiler/xla/service/gpu/dapple_all_gather_thunk.h"

#include "absl/algorithm/container.h"
#include "third_party/nccl/nccl.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/util/env_var.h"

namespace xla {
namespace gpu {

DAPPLEAllGatherThunk::DAPPLEAllGatherThunk(
    std::vector<Buffer> buffers,
    const HloInstruction* dapple_all_gather)
    : Thunk(Thunk::kDAPPLEAllGather, dapple_all_gather),
      buffers_(std::move(buffers)) {
  CHECK_EQ(hlo_instruction()->operand_count(), buffers_.size());
}

/*static*/ bool DAPPLEAllGatherThunk::CanImplement(const HloInstruction* op) {
  CHECK_EQ(op->operands().size(), 1);
  auto shape = op->operand(0)->shape();
  auto ag = DynCast<HloDAPPLEAllGatherInstruction>(op);
  return LayoutUtil::IsDenseArray(shape) &&
         IsTypeSupportedByNccl(shape.element_type()) &&
         LayoutUtil::MinorToMajor(shape).back() == ag->all_gather_dimension();
}

Status DAPPLEAllGatherThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto* hlo = Cast<HloDAPPLEAllGatherInstruction>(hlo_instruction());
  Shape shape = hlo->operand(0)->shape();
  if (!DAPPLEAllGatherThunk::CanImplement(hlo_instruction())) {
    std::string message = absl::StrFormat(
      "DAPPLEAllGatherThunk Umplemented. "
      "Not dense array or major dimension and all_gather_dimension are not equal. "
      "IsDenseArray : %d; all_gather_dimension at : %d; "
      "Major dimension at : %d; , hlo : %s .", LayoutUtil::IsDenseArray(shape),
      hlo->all_gather_dimension(), LayoutUtil::MinorToMajor(shape).back(),
      hlo->ToString());
    return Unimplemented("%s", message);
  }

  TF_ASSIGN_OR_RETURN(ncclDataType_t data_type,
                      ToNcclDataType(hlo->shape().element_type()));

  int device_ordinal = params.stream->parent()->device_ordinal();
  VLOG(1) << "Performing all-gather from device ordinal: " << device_ordinal;

  NcclContext* nccl_ctx = params.nccl_ctx;
  CHECK(nccl_ctx);

  const std::vector<int64>* global_devices;
  int rank;
  const std::vector<int64>* task_id = params.task_id_;
  CHECK(task_id);
  VLOG(2) << "task_id size: " << task_id->size();
  for (int64 one_id : *task_id) {
    VLOG(2) << "one id: " << one_id;
  }
  int split_ordinal = hlo->split_ordinal();
  VLOG(2) << "split_ordinal: " << split_ordinal;

  std::shared_ptr<CommDevManager> comm_dev_mgr = params.comm_dev_mgr_;
  CHECK(comm_dev_mgr);
  std::shared_ptr<DevGroup> dev_group = comm_dev_mgr->FindDevGroup(*task_id, split_ordinal);
  CHECK(dev_group);

  global_devices = &dev_group->global_dev_ids_;

  rank = comm_dev_mgr->GetCommRank(*task_id, split_ordinal);

  int64 local_device_ordinal = params.stream->parent()->device_ordinal();

  VLOG(2) << "on this device, rank is " << rank;
  TF_ASSIGN_OR_RETURN(NcclComm* nccl_comm_ptr,
                      nccl_ctx->GetNcclComm(*global_devices, rank));
  ncclComm_t nccl_comm = nccl_comm_ptr->comm();

  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      params.stream->implementation()->GpuStreamMemberHack());

  CHECK(nccl_comm && "NCCL communicator uninitialized!");
  VLOG(2) << hlo_instruction()->ToString()
         << " EXE-dev_id:" << local_device_ordinal
         << " nccl_comm:" << nccl_comm;

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const Buffer& buffer = buffers_[i];
    const void* send_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
            .opaque();
    void* recv_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
            .opaque();

    VLOG(2) << absl::StreamFormat(
        "Calling ncclAllGather(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, buffer.element_count,
        static_cast<const void*>(nccl_comm), cu_stream);

    XLA_CUDA_RETURN_IF_ERROR(ncclAllGather(send_buffer, recv_buffer,
                                           buffer.element_count, data_type,
					                                 nccl_comm, *cu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(1) << "Done performing all-gather for ordinal: " << local_device_ordinal;
  return Status::OK();
}


DAPPLEAllGatherThunk::~DAPPLEAllGatherThunk() {}


}  // namespace gpu
}  // namespace xla
