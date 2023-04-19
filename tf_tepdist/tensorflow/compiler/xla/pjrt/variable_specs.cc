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

#include "tensorflow/compiler/xla/pjrt/variable_specs.h"

#include "tensorflow/compiler/xla/pjrt/slice_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

void VariableSpecsMgr::ExtractLocalVariableSpecs(
    std::vector<DAPPLEBuffer*>& local_variables, std::map<int, int>& arg_var_map) {
  // Avoid duplicate extraction
  if (name_spec_map_.size()) return;
  for (auto* task : local_graph_->source()->children()) {
    if (task->task_type() == TaskNode::TaskType::kRecv) {
      ExtractVariableSpecsFromRecvTask(task, local_variables, arg_var_map);
    } else if (task->task_type() == TaskNode::TaskType::kInput) {
      ExtractVariableSpecsFromInputTask(task, local_variables, arg_var_map);
    }
  }
}

void VariableSpecsMgr::ExtractVariableSpecsFromRecvTask(
    TaskNode* recv_task, std::vector<DAPPLEBuffer*>& local_variables,
    std::map<int, int>& arg_var_map) {
  for (auto* task : recv_task->children()) {
    if (task->task_type() == TaskNode::TaskType::kInput) {
      ExtractVariableSpecsFromInputTask(task, local_variables, arg_var_map);
    }
  }
}

void VariableSpecsMgr::ExtractVariableSpecsFromInputTask(
    TaskNode* input_task, std::vector<DAPPLEBuffer*>& local_variables,
    std::map<int, int>& arg_var_map) {
  auto* hlo_module = local_graph_->task_module(local_graph_->source());
  auto* def_ctx = input_task->def_ctx();
  auto* entry = hlo_module->entry_computation();
  auto& input_specs = input_task->input_specs();
  std::shared_ptr<CommDevManager> comm_dev_mgr = input_task->comm_dev_mgr();
  CHECK(comm_dev_mgr);
  int64 global_dev_id = comm_dev_mgr->global_dev_id(input_task->split_id());
  int64 local_dev_id = comm_dev_mgr->local_dev_id(global_dev_id);
  std::vector<int> split_nums = comm_dev_mgr->split_nums();
  std::vector<int64> split_ids = input_task->split_id().ids_;

  for (auto& iter : input_specs) {
    auto& node_out_pair = iter.second[0];
    TaskNode* parent = node_out_pair.first;
    if (parent->task_type() != TaskNode::TaskType::kSplit &&
        parent->task_type() != TaskNode::TaskType::kRecv) continue;
    // We should resolve arg_no in kSplit:
    // 1. If parent is kSplit, node_out_pair.second is what we are looking for.
    // 2. If parent is kRecv, node_out_pair.second is the port_map's value. But
    //    the corresponding port_map's key is what we are looking for.
    int arg_no = node_out_pair.second;
    if (parent->task_type() == TaskNode::TaskType::kRecv) {
      auto& port_map = parent->port_map();
      for (auto& p_iter : port_map) {
        if (p_iter.second != node_out_pair.second) continue;
        arg_no = p_iter.first;
        break;
      }
    }

    // If the argument is not a variable or is not a local variable, ignore extraction.
    if (!variable_map_->count(arg_no) || !arg_var_map.count(arg_no)) continue;

    HloInstruction* param = entry->parameter_instruction(arg_no);
    auto& dist_spec = param->dist_spec();
    std::vector<int64> slice_ids(dist_spec.size());
    std::vector<int> slice_nums(dist_spec.size());
    for (int ordinal = 0; ordinal < dist_spec.size(); ++ordinal) {
      auto& dim_spec = dist_spec.get_dim_spec(ordinal);
      if (dim_spec->stride_on_dim()) {
        slice_ids[ordinal] = split_ids[ordinal];
        slice_nums[ordinal] = split_nums[ordinal];
      } else {
        slice_ids[ordinal] = 0;
        slice_nums[ordinal] = 1;
      }
    }

    std::vector<int64> slice_nums_base(slice_nums.size());
    int64 base = 1;
    for (int ordinal = slice_nums.size() - 1; ordinal >=0; --ordinal) {
      slice_nums_base[ordinal] = base;
      base *= slice_nums[ordinal];
    }

    int64 linear_slice_id = 0;
    for (int i = 0; i < slice_ids.size(); ++i) {
      linear_slice_id += slice_ids[i] * slice_nums_base[i];
    }
    auto& tensor_name = variable_map_->at(arg_no);
    if (!name_spec_map_.count(tensor_name)) {
      VariableSpec spec;
      CHECK(arg_var_map[arg_no] < local_variables.size());
      spec.d_buf = local_variables[arg_var_map[arg_no]];
      spec.arg_shape = param->shape();
      spec.top_arg_no = arg_no;
      name_spec_map_[tensor_name] = spec;
    }
    auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
        name_spec_map_[tensor_name].d_buf->on_host_shape(),
        param->dist_spec(), input_task->split_id().ids_);

    name_spec_map_[tensor_name].start_offset_pairs_map[global_dev_id] = start_offset_pairs;
    name_spec_map_[tensor_name].global_local_dev_map[global_dev_id] = local_dev_id;
    name_spec_map_[tensor_name].global_dev_linear_slice_map[global_dev_id] = linear_slice_id;
  }
}

} // namespace xla
