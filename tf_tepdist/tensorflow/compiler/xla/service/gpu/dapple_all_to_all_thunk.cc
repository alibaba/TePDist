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

#include "tensorflow/compiler/xla/service/gpu/dapple_all_to_all_thunk.h"

#include "absl/algorithm/container.h"
#include "third_party/nccl/nccl.h"

#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/util/env_var.h"

namespace xla {
namespace gpu {

DAPPLEAllToAllThunk::DAPPLEAllToAllThunk(
    std::vector<DAPPLEAllToAllThunk::Buffer> buffers,
    const HloInstruction* all_to_all)
    : Thunk(Thunk::kDAPPLEAllToAll, all_to_all),
      buffers_(std::move(buffers)) {
  CHECK_EQ(hlo_instruction()->operand_count(), buffers_.size());
}

DAPPLEAllToAllThunk::~DAPPLEAllToAllThunk() {}

/*static*/ bool DAPPLEAllToAllThunk::CanImplement(const HloInstruction* hlo) {
  return absl::c_all_of(hlo->operands(), [hlo](HloInstruction* operand) {
    Shape shape = operand->shape();
    auto* all_to_all = Cast<HloDAPPLEAllToAllInstruction>(hlo);
    return LayoutUtil::IsDenseArray(shape) &&
           (!all_to_all->split_dimension() ||
            LayoutUtil::MinorToMajor(shape).back() == all_to_all->split_dimension());
  });
}

Status DAPPLEAllToAllThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto* hlo = Cast<HloDAPPLEAllToAllInstruction>(hlo_instruction());
  Shape shape = hlo->operand(0)->shape();
  if (!DAPPLEAllToAllThunk::CanImplement(hlo_instruction())) {
    std::string message = absl::StrFormat(
      "DAPPLEAllToAllThunk Umplemented. "
      "Not dense array or major dimension and split_dimension are not equal. "
      "IsDenseArray : %d; split_dimension at : %d; "
      "Major dimension at : %d; hlo : %s .", LayoutUtil::IsDenseArray(shape),
      hlo->split_dimension(), LayoutUtil::MinorToMajor(shape).back(),
      hlo->ToString());
    return Unimplemented("%s", message);
  }
  // AllToAll specifies a split dimension, in which case inputs are split and outputs
  // concatenated in that dimension (here, we only support dimension 0)
  TF_ASSIGN_OR_RETURN(ncclDataType_t data_type,
                      ToNcclDataType(hlo->shape().element_type()));

  VLOG(1) << "Start DAPPLEAllToAllThunk.";
  NcclContext* nccl_ctx = params.nccl_ctx;

  const std::vector<int64>* global_devices;
  int rank;
  const std::vector<int64>* task_id = params.task_id_;
  CHECK(task_id);
  int split_ordinal = hlo->split_ordinal();

  std::shared_ptr<CommDevManager> comm_dev_mgr = params.comm_dev_mgr_;
  CHECK(comm_dev_mgr);
  std::shared_ptr<DevGroup> dev_group = comm_dev_mgr->FindDevGroup(*task_id, split_ordinal);

  global_devices = &dev_group->global_dev_ids_;

  rank = comm_dev_mgr->GetCommRank(*task_id, split_ordinal);

  int64 local_device_ordinal = params.stream->parent()->device_ordinal();

  VLOG(2) << "on this device, rank is " << rank;
  TF_ASSIGN_OR_RETURN(NcclComm* nccl_comm_ptr,
                      nccl_ctx->GetNcclComm(*global_devices, rank));
  ncclComm_t nccl_comm = nccl_comm_ptr->comm();
  VLOG(2) << hlo_instruction()->ToString()
          << " EXE-dev_id:" << local_device_ordinal 
          << " nccl_comm:" << nccl_comm;
  CHECK(nccl_comm && "NCCL communicator uninitialized!");
  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      params.stream->implementation()->GpuStreamMemberHack());

  int num_participants;
  XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(nccl_comm, &num_participants));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const Buffer& buffer = buffers_[i];
    const uint8* send_buffer = static_cast<uint8*>(
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
            .opaque());
    uint8* recv_buffer = static_cast<uint8*>(
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
            .opaque());

    TF_RET_CHECK(buffer.element_count % num_participants == 0)
        << "Buffer was not an exact multiple of the number of participants.";
    size_t chunk_elements = buffer.element_count / num_participants;
    size_t chunk_bytes = chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(
        hlo->shape().element_type());

    for (int rank = 0; rank < num_participants; ++rank) {
      VLOG(2) << absl::StreamFormat(
          "Calling ncclSend(send_buffer=%p, count=%d, peer=%d, "
          "comm=%p, stream=%p)",
          send_buffer + rank * chunk_bytes, chunk_elements, rank,
          static_cast<const void*>(nccl_comm), *cu_stream);
      XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer + rank * chunk_bytes,
                                        chunk_elements, data_type, rank, nccl_comm,
                                        *cu_stream));
      VLOG(2) << absl::StreamFormat(
          "Calling ncclRecv(recv_buffer=%p, count=%d, peer=%d, "
          "comm=%p, stream=%p)",
          recv_buffer + rank * chunk_bytes, chunk_elements, rank,
          static_cast<const void*>(nccl_comm), *cu_stream);
      XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer + rank * chunk_bytes,
                                        chunk_elements, data_type, rank, nccl_comm,
                                        *cu_stream));
    }
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());
  VLOG(1) << "Done performing all to all for ordinal: "
          << local_device_ordinal;

  return Status::OK();
}

} // namespace gpu
} // namespace xla
