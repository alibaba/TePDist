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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_VARIABLE_SPECS_H
#define TENSORFLOW_COMPILER_XLA_PJRT_VARIABLE_SPECS_H

#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {

// Sharding information for all variables
struct VariableSpec {
  int top_arg_no = -1;
  Shape arg_shape;
  xla::DAPPLEBuffer* d_buf = nullptr;
  std::unordered_map<int64/*global_dev*/,
                     int64/*local_dev*/> global_local_dev_map;
  std::unordered_map<int64/*global_dev*/,
                     int64/*linear_slice_id*/> global_dev_linear_slice_map;
  std::unordered_map<int64/*global_dev*/,
                     std::vector<std::pair<int64, int64>>> start_offset_pairs_map;
};

// Manager class for extracting VariableSpec
class VariableSpecsMgr {
 public:
  explicit VariableSpecsMgr(
      TaskDAG* local_graph, const std::map<int, std::string>* variable_map,
      bool sharding_across_machine, int local_dev_count, int worker_rank,
      int worker_count)
      : local_graph_(local_graph), variable_map_(variable_map),
        sharding_across_machine_(sharding_across_machine),
        local_dev_count_(local_dev_count), worker_rank_(worker_rank),
        worker_count_(worker_count) {}

  void ExtractLocalVariableSpecs(
      std::vector<DAPPLEBuffer*>& local_variables, std::map<int, int>& arg_var_map);

  const std::unordered_map<std::string, VariableSpec>& name_spec_map() const {
    return name_spec_map_;
  }

 private:
  void ExtractVariableSpecsFromInputTask(
    TaskNode* recv_task, std::vector<DAPPLEBuffer*>& local_variables,
    std::map<int, int>& arg_var_map);

  void ExtractVariableSpecsFromRecvTask(
    TaskNode* recv_task, std::vector<DAPPLEBuffer*>& local_variables,
    std::map<int, int>& arg_var_map);

  TaskDAG* local_graph_; // Not owned
  const std::map<int, std::string>* variable_map_; // Not owned
  bool sharding_across_machine_;
  int local_dev_count_;
  int worker_rank_;
  int worker_count_;
  std::unordered_map<std::string, VariableSpec> name_spec_map_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_VARIABLE_SPECS_H
