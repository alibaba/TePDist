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

#include "tensorflow/compiler/xla/service/cluster_and_device_spec.h"

#include "tensorflow/compiler/xla/util.h"

namespace xla {

ClusterSpec::ClusterSpec(
    std::vector<std::pair<std::string, int64>>& node_local_dev_count) {
  Initialize(node_local_dev_count);
}

void ClusterSpec::Initialize(
    std::vector<std::pair<std::string, int64>>& node_local_dev_count) {
  int global_dev_id = 0;
  for (int n_id = 0; n_id < node_local_dev_count.size(); ++n_id) {
    nodes_.push_back(node_local_dev_count[n_id].first);
    std::unordered_set<int64> node_devices;
    for (int l_dev_id = 0; l_dev_id < node_local_dev_count[n_id].second; ++l_dev_id) {
      global_devices_.push_back(global_dev_id);
      node_devices.insert(global_dev_id);
      device_node_map_[global_dev_id] = n_id;
      GlobalDeviceSpec dev_spec = {global_dev_id, n_id, l_dev_id};
      global_id_to_dev_spec_map_[global_dev_id++] = dev_spec;
    }
    node_to_devices_.push_back(node_devices);
  }
}

StatusOr<GlobalDeviceSpec*> ClusterSpec::GetDeviceSpec(int64 global_device_id) {
  if (global_id_to_dev_spec_map_.find(global_device_id) ==
      global_id_to_dev_spec_map_.end()) {
    return InvalidArgument("Cannot find global_device id %d", global_device_id);
  }

  return &global_id_to_dev_spec_map_[global_device_id];
}

StatusOr<std::vector<GlobalDeviceSpec*>> ClusterSpec::GetDeviceSpecs(
    const absl::Span<const int64> global_device_ids) {
  std::vector<GlobalDeviceSpec*> specs;
  for (int64 gid : global_device_ids) {
    TF_ASSIGN_OR_RETURN(GlobalDeviceSpec* spec, GetDeviceSpec(gid));
    specs.push_back(spec);
  }
  return specs;
}

StatusOr<std::vector<int64>> ClusterSpec::PickGlobalDevicesForNode(
    const absl::Span<const int64> global_device_ids, int node_id) {
  std::vector<int64> filtered_global_devices;
  if (node_id >= nodes_.size()) {
    return InvalidArgument(
        "node_id should be less than %d, but got %d", (nodes_.size(), node_id));
  }
  for (int64 gid : global_device_ids) {
    if (node_to_devices_[node_id].find(gid) == node_to_devices_[node_id].end()) continue;
    filtered_global_devices.push_back(gid);
  }

  return filtered_global_devices;
}

} // namespace xla
