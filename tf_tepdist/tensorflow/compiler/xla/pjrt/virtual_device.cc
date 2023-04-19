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

#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/nvidia_gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/virtual_client.h"
#include "tensorflow/compiler/xla/pjrt/virtual_device.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"

namespace xla {

static const char kVirtualPlatformName[] = "virtual";

VirtualDevice::VirtualDevice(int id,
                     std::unique_ptr<LocalDeviceState> local_device_state)
    : Device(id, std::move(local_device_state), kVirtualPlatformName,
             /*device_kind=*/kVirtualPlatformName) {}

StatusOr<std::shared_ptr<VirtualClient>> GetVirtualClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config,
    std::shared_ptr<DistributedRuntimeClient> distributed_client, int node_id) {
  auto cpu_client = GetCpuClient(asynchronous).ConsumeValueOrDie();
  /* Bail out when there is no gpu available */
  auto gpu_client = GetNvidiaGpuClient(asynchronous, allocator_config,
      distributed_client, node_id).ConsumeValueOrDie();
  return std::make_shared<VirtualClient>(kVirtualPlatformName, /*host_id=*/0,
                                         cpu_client, gpu_client);
}

VirtualClient* CreateVirtualClient(int node_id=0) {
  auto cpu_client = GetCpuClient(true).ConsumeValueOrDie();
  auto allocator_config = GpuAllocatorConfig();
  allocator_config.kind = GpuAllocatorConfig::Kind::kBFC;
  //allocator_config.preallocate = false;
  // Increase maximum fraction of available memory from default 90% to radical
  // 97% so that we can handle larger models.
  allocator_config.memory_fraction = 0.97;

  /* Bail out when there is no gpu available */
  // Note: we assume that each machine owns same number of available devices.
  auto gpu_client = GetNvidiaGpuClient(true, std::move(allocator_config),
      nullptr, node_id).ConsumeValueOrDie();
  return new VirtualClient(kVirtualPlatformName, /*host_id=*/node_id,
                           cpu_client, gpu_client);
}

}  // namespace xla
