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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CHECKPOINT_UTILS_H
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CHECKPOINT_UTILS_H

#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/compiler/xla/pjrt/variable_specs.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

#include <deque>

namespace dist_ckpt_utils {

using namespace tensorflow;

// Checkpoint utility for saving or restoring model parameters on servers.
class CheckpointUtil {
 public:
  CheckpointUtil(std::string save_path, int64 max_to_keep);

  void Initialize(
      const xla::VariableSpecsMgr* var_spec_mgr, xla::PjRtClient* gpu_client);

  void Save(int64 global_step);
  void Restore(int64 global_step);

 private:
  std::vector<tstring> WriteTensorsToTempFiles(int64 global_step);
  std::string MergeShardedTempFiles(
      std::vector<tstring>& input_prefixes, int64 global_step);

  std::unique_ptr<Tensor> LookupTensor(
      BundleReader* reader, std::string tensor_name,
      TensorShape& sharded_shape, TensorSlice& slice);
  void FlushPrefixQueueToFile();
  void ReadPrefixQueueFromFile();
  std::string save_path_;
  int64 max_to_keep_ = -1;
  std::deque<std::string> ckpt_prefix_queue_;
  xla::PjRtClient* gpu_client_; // Not owned
  const xla::VariableSpecsMgr* var_spec_mgr_; // Not owned
  std::string uuid_str_;

};

} // namespace dist_ckpt_utils

#endif // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CHECKPOINT_UTILS_H
