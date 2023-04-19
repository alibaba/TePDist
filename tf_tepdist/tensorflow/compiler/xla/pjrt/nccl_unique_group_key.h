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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_NCCL_UNIQUE_GROUP_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_NCCL_UNIQUE_GROUP_H_

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "absl/types/optional.h"
#include "absl/strings/str_format.h"
#include "third_party/nccl/nccl.h"

namespace xla {
namespace gpu {

class NcclUniqueGroupKey {
 public:
  explicit NcclUniqueGroupKey(const absl::Span<const int64> global_devices);

  template <typename H>
  friend H AbslHashValue(H h, const NcclUniqueGroupKey& k) {
    return H::combine(std::move(h), k.global_devices_);
  }
  friend bool operator==(const NcclUniqueGroupKey& a, const NcclUniqueGroupKey& b) {
    return a.global_devices_ == b.global_devices_;
  }

  const std::vector<int64>& global_devices() const {
    return global_devices_;
  }

 private:
  std::vector<int64> global_devices_;
};

} // namespace gpu
} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_NCCL_UNIQUE_GROUP_H_
