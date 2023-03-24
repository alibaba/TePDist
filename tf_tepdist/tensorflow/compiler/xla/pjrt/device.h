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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DEVICE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DEVICE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/pjrt/tracked_device_buffer.h"

// API notes:
// PjRt stands for "Pretty much Just another RunTime".

namespace xla {

class Device {
 public:
  explicit Device(int id, std::unique_ptr<LocalDeviceState> local_device_state,
                  std::string platform_name, std::string device_kind,
                  int host_id = 0)
      : id_(id),
        local_device_state_(std::move(local_device_state)),
        host_id_(host_id),
        platform_name_(std::move(platform_name)),
        device_kind_(std::move(device_kind)) {}

  explicit Device(int id, std::string platform_name, std::string device_kind,
                  int host_id = 0)
      : id_(id),
        host_id_(host_id),
        platform_name_(std::move(platform_name)),
        device_kind_(std::move(device_kind)) {}

  virtual ~Device() {}

  // The ID of this device. IDs are unique among devices of this type
  // (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
  // hosts' devices.  This is the ID that should be used in a DeviceAssignment.
  int id() const { return id_; }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns nullptr if the device is
  // not local to this host.
  LocalDeviceState* local_device_state() const {
    return local_device_state_.get();
  }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns an error if the device
  // is not local to this host.
  StatusOr<LocalDeviceState*> GetLocalDeviceState() const;

  // The ID of this device's host. This is always 0 on single-host platforms.
  int host_id() const { return host_id_; }

  const std::string& platform_name() const { return platform_name_; }

  // A vendor-dependent string that uniquely identifies the kind of device.
  const std::string& device_kind() const { return device_kind_; }

  virtual std::string DebugString() const;

 private:
  const int id_;
  const std::unique_ptr<LocalDeviceState> local_device_state_;
  const int host_id_;
  const std::string platform_name_;
  const std::string device_kind_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DEVICE_H_
