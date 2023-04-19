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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_NCCL_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_NCCL_CONTEXT_H_

#include <string>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "tensorflow/compiler/xla/pjrt/dev_id_util.h"
#include "tensorflow/compiler/xla/pjrt/nccl_comm.h"
#include "tensorflow/compiler/xla/pjrt/nccl_unique_group_key.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "third_party/nccl/nccl.h"

namespace xla {
namespace gpu {


class NcclContext {
 public:
  explicit NcclContext() = default;

  Status MaybeCreateNcclUniqueId(NcclUniqueGroupKey& group_key);

  Status CreateNcclUniqueId(NcclUniqueGroupKey& group_key);

  Status RegisterNcclUniqueId(
      NcclUniqueGroupKey& group_key, ncclUniqueId& nccl_id);

  bool NcclUniqueIdAlreadyCreated(NcclUniqueGroupKey& group_key);

  Status MaybeCreateNcclComms(
      NcclUniqueGroupKey& group_key,
      std::vector<int64>& filtered_participants,
      std::shared_ptr<CommDevManager> comm_dev_mgr);

  Status MaybeCreateNcclComms(
      absl::Span<const int64> global_devices, int worker_id,
      std::shared_ptr<CommDevManager> comm_dev_mgr);

  StatusOr<ncclUniqueId*> GetNcclUniqueId(NcclUniqueGroupKey& group_key);

  StatusOr<ncclUniqueId*> GetOrCreateNcclUniqueId(NcclUniqueGroupKey& group_key);

  StatusOr<NcclComm*> GetNcclComm(NcclCommKey& comm_key);
  
  StatusOr<NcclComm*> GetNcclComm(
      const absl::Span<const int64> global_devices, int rank);

  ~NcclContext() {}

 private:
  absl::node_hash_map<NcclUniqueGroupKey,
                      std::unique_ptr<ncclUniqueId>> nccl_grp_id_map_;

  absl::node_hash_map<NcclCommKey,
                      gpu::NcclComm> nccl_comm_map_;

};

} // namespace gpu
} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_NCCL_CONTEXT_H_
