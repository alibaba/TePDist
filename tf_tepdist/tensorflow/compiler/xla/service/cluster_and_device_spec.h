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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CLUSTER_AND_DEVICE_SPEC_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CLUSTER_AND_DEVICE_SPEC_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "absl/types/span.h"

#include <vector>
#include <string>

namespace xla {

typedef struct GlobalDeviceSpec {
  int64 global_dev_id;
  int64 node_id;
  int64 local_dev_id;
} GlobalDeviceSpec;

class ClusterSpec {
 public:
  explicit ClusterSpec() {}
  explicit ClusterSpec(
      std::vector<std::pair<std::string, int64>>& node_local_dev_count);

  void Initialize(std::vector<std::pair<std::string, int64>>& node_local_dev_count);

  const std::vector<int64>& global_devices() const { return global_devices_; }
  const std::vector<std::string>& nodes() const { return nodes_; }
  const std::vector<std::unordered_set<int64>>& node_to_devices() const {
    return node_to_devices_;
  }
  const std::unordered_map<int64, int64>& device_node_map() const {
    return device_node_map_;
  }

  const int64 num_nodes() const { return nodes_.size(); }

  StatusOr<GlobalDeviceSpec*> GetDeviceSpec(int64 global_device_id);
  StatusOr<std::vector<GlobalDeviceSpec*>> GetDeviceSpecs(
      const absl::Span<const int64> global_device_ids);
  StatusOr<std::vector<int64>> PickGlobalDevicesForNode(
      const absl::Span<const int64> global_device_ids, int node_id);

 private:
  std::vector<int64> global_devices_;
  std::vector<std::string> nodes_;
  std::vector<std::unordered_set<int64>> node_to_devices_;
  std::unordered_map<int64, int64> device_node_map_;
  std::unordered_map<int64, GlobalDeviceSpec> global_id_to_dev_spec_map_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_CLUSTER_AND_DEVICE_SPEC_H_
