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

#include "tensorflow/compiler/xla/pjrt/nccl_unique_group_key.h"

#include "tensorflow/compiler/xla/util.h"
#include "absl/algorithm/container.h"

namespace xla {
namespace gpu {

NcclUniqueGroupKey::NcclUniqueGroupKey(const absl::Span<const int64> global_devices)
    : global_devices_(global_devices.begin(), global_devices.end()) {
  //absl::c_sort(global_devices_);
  CHECK(absl::c_adjacent_find(global_devices_) == global_devices_.end())
      << "Duplicate devices are not allowed: "
      << GlobalDevicesToString(global_devices_);
}

} // namespace gpu
} // namespace xla
