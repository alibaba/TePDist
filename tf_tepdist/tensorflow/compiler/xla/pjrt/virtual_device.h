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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_VIRTUAL_DEVICE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_VIRTUAL_DEVICE_H_

#include "tensorflow/compiler/xla/pjrt/device.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"

namespace xla {

class VirtualClient;
class VirtualDevice : public Device {
 public:
  VirtualDevice(int id, std::unique_ptr<LocalDeviceState> local_device_state);
};

StatusOr<std::shared_ptr<VirtualClient>> GetVirtualClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config,
    std::shared_ptr<DistributedRuntimeClient> distributed_client, int node_id);

VirtualClient* CreateVirtualClient(int node_id);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_VIRTUAL_DEVICE_H_
