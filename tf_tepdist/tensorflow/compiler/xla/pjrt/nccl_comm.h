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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_NCCL_COMM_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_NCCL_COMM_H_

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/nccl_unique_group_key.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "absl/types/optional.h"
#include "third_party/nccl/nccl.h"

namespace xla {
namespace gpu {

// RAII class owning a ncclComm_t, ensuring it doesn't leak.
class NcclComm {
 public:
  explicit NcclComm(ncclComm_t comm) : comm_(comm) {}

  // Movable, but not copyable.
  NcclComm(NcclComm&& c) noexcept : comm_(c.comm_) { c.comm_.reset(); }
  NcclComm& operator=(NcclComm&& c) noexcept {
    comm_ = c.comm_;
    c.comm_.reset();
    return *this;
  }
  NcclComm(const NcclComm&) = delete;
  NcclComm& operator=(const NcclComm&) = delete;

  ~NcclComm() {
    if (comm_.has_value() && *comm_ != nullptr) {
      VLOG(0) << absl::StreamFormat("Destroying comm %p", *comm_);
      XLA_CUDA_WARN_IF_ERROR(ncclCommDestroy(*comm_));
    }
  }

  ncclComm_t comm() { return *comm_; }

 private:
  absl::optional<ncclComm_t> comm_;
};

class NcclCommKey {
 public:
  explicit NcclCommKey(NcclUniqueGroupKey& group_key, int rank)
      : group_key_(group_key), rank_(rank) {}

  template <typename H>
  friend H AbslHashValue(H h, const NcclCommKey& k) {
    return H::combine(std::move(h), k.group_key_, k.rank_);
  }
  friend bool operator==(const NcclCommKey& a, const NcclCommKey& b) {
    return a.group_key_ == b.group_key_ && a.rank_ == b.rank_;
  }

  const NcclUniqueGroupKey& nccl_unique_group_key() const {
    return group_key_;
  }

  const int64 rank() const { return rank_; }
 private:
  NcclUniqueGroupKey group_key_;
  int rank_;
};

} // namespace gpu
} // namespace xla

#endif // #ifndef TENSORFLOW_COMPILER_XLA_PJRT_NCCL_COMM_H_
