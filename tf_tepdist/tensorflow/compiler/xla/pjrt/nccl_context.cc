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

#include "tensorflow/compiler/xla/pjrt/nccl_context.h"

namespace xla {
namespace gpu {

Status NcclContext::MaybeCreateNcclUniqueId(NcclUniqueGroupKey& group_key) {
  if (nccl_grp_id_map_.find(group_key) != nccl_grp_id_map_.end()) {
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(CreateNcclUniqueId(group_key));
  return Status::OK();
}

Status NcclContext::CreateNcclUniqueId(NcclUniqueGroupKey& group_key) {
  std::unique_ptr<ncclUniqueId> nccl_id = absl::make_unique<ncclUniqueId>();
  XLA_CUDA_RETURN_IF_ERROR(ncclGetUniqueId(nccl_id.get()));
  nccl_grp_id_map_.emplace(group_key, std::move(nccl_id));
  return Status::OK();
}

Status NcclContext::RegisterNcclUniqueId(
    NcclUniqueGroupKey& group_key, ncclUniqueId& nccl_id) {
  if (nccl_grp_id_map_.find(group_key) != nccl_grp_id_map_.end()) {
    return InvalidArgument("group key has been created");
  }

  std::unique_ptr<ncclUniqueId> nccl_id_ptr = absl::make_unique<ncclUniqueId>();
  std::memcpy((void *)nccl_id_ptr.get(), (const void*)&nccl_id, sizeof(ncclUniqueId));
  nccl_grp_id_map_.emplace(group_key, std::move(nccl_id_ptr));
  return Status::OK();
}

bool NcclContext::NcclUniqueIdAlreadyCreated(NcclUniqueGroupKey& group_key) {
  return nccl_grp_id_map_.find(group_key) != nccl_grp_id_map_.end();
}

Status NcclContext::MaybeCreateNcclComms(
    NcclUniqueGroupKey& group_key,
    std::vector<int64>& filtered_participants,
    std::shared_ptr<CommDevManager> comm_dev_mgr) {
  std::unordered_map<int64, int64> global_dev_id_rank_map;
  const std::vector<int64>& all_participants = group_key.global_devices();
  for (int r = 0; r < all_participants.size(); ++r) {
    global_dev_id_rank_map[all_participants[r]] = r;
  }

  TF_ASSIGN_OR_RETURN(ncclUniqueId* nccl_id, GetNcclUniqueId(group_key));

  int num_cache = 0;
  for (int i = 0; i < filtered_participants.size(); ++i) {
    int64 rank = global_dev_id_rank_map[filtered_participants[i]];
    NcclCommKey comm_key(group_key, rank);
    num_cache += (nccl_comm_map_.find(comm_key) != nccl_comm_map_.end());
  }

  if (num_cache > 0 && num_cache < filtered_participants.size()) {
    return InvalidArgument("Only parts of participants are created");
  }

  // Cache hit!
  if (num_cache == filtered_participants.size()) return Status::OK();

  std::vector<ncclComm_t> raw_comms(filtered_participants.size(), nullptr);
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (int i = 0; i < filtered_participants.size(); ++i) {
    int64 rank = global_dev_id_rank_map[filtered_participants[i]];
    NcclCommKey comm_key(group_key, rank);

    if (rank >= all_participants.size()) {
      return InvalidArgument("rank should not be >= all_participants size,",
                              "got rank %d, all_participants size %d",
                              rank, all_participants.size());
    }

    int initial_cuda_device;
    XLA_CUDA_RETURN_IF_ERROR(cudaGetDevice(&initial_cuda_device));

    int local_dev_id = comm_dev_mgr->local_dev_id(filtered_participants[i]);
    XLA_CUDA_RETURN_IF_ERROR(cudaSetDevice(local_dev_id));
    VLOG(1) << "local_dev_id = " << local_dev_id
            << ", global_dev_id = " << filtered_participants[i]
            << ", rank = " << rank << ", i = " << i
            << ", initial_cuda_device = " << initial_cuda_device;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommInitRank(&raw_comms[i], all_participants.size(),
                                              *nccl_id, rank));
    XLA_CUDA_RETURN_IF_ERROR(cudaSetDevice(initial_cuda_device));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  for (int i = 0; i < filtered_participants.size(); ++i) {
    int64 rank = global_dev_id_rank_map[filtered_participants[i]];
    NcclCommKey comm_key(group_key, rank);
    NcclComm comm(raw_comms[i]);
    nccl_comm_map_.emplace(comm_key, std::move(comm));
  }
  return Status::OK();
}

StatusOr<ncclUniqueId*> NcclContext::GetNcclUniqueId(
    NcclUniqueGroupKey& group_key) {
  if (nccl_grp_id_map_.find(group_key) == nccl_grp_id_map_.end()) {
    return InvalidArgument("GetNcclUniqueId : Cannot find ncclUniqueId");
  }

  return nccl_grp_id_map_[group_key].get();
}

StatusOr<ncclUniqueId*> NcclContext::GetOrCreateNcclUniqueId(
    NcclUniqueGroupKey& group_key) {
  TF_RETURN_IF_ERROR(MaybeCreateNcclUniqueId(group_key));
  TF_ASSIGN_OR_RETURN(ncclUniqueId* nccl_id, GetNcclUniqueId(group_key));
  return nccl_id; 
}

StatusOr<NcclComm*> NcclContext::GetNcclComm(NcclCommKey& comm_key) {
  absl::node_hash_map<NcclCommKey,
                      NcclComm>::iterator iter = nccl_comm_map_.find(comm_key);
  if (iter == nccl_comm_map_.end()) {
    return InvalidArgument("GetNcclComm : Cannot find ncclComm");
  }

  return &iter->second;
}

StatusOr<NcclComm*> NcclContext::GetNcclComm(
    const absl::Span<const int64> global_devices, int rank) {
  NcclUniqueGroupKey group_key(global_devices);
  TF_ASSIGN_OR_RETURN(ncclUniqueId* nccl_id, GetNcclUniqueId(group_key));
  NcclCommKey comm_key(group_key, rank);
  TF_ASSIGN_OR_RETURN(NcclComm* nccl_comm, GetNcclComm(comm_key));
  return nccl_comm;
}

Status NcclContext::MaybeCreateNcclComms(
    absl::Span<const int64> global_devices, int worker_id,
    std::shared_ptr<CommDevManager> comm_dev_mgr) {
  NcclUniqueGroupKey group_key(global_devices);

  std::vector<int64> filtered;
  for (int64 g_dev : global_devices) {
    if (comm_dev_mgr->worker_id(g_dev) == worker_id) {
      filtered.push_back(g_dev);
    }
  }

  if (filtered.empty()) return Status::OK();

  return MaybeCreateNcclComms(group_key, filtered, comm_dev_mgr);
}

} // namespace gpu
} // namespace xla

