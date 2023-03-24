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

#include "tensorflow/compiler/xla/pjrt/lifetime_tracker.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/core/util/env_var.h"

namespace xla {

void OutputBuffersLifeTimeTracker::Initialize(const TaskDAG* task_graph) {
  // task_graph MUST have been setup input spec (have run SetupInputSpecs)
  // currently OutputBuffersLifeTimeTracker is not reentrant.
  CHECK(buf_ref_counts_pool_.empty());
  CHECK(name_buffer_ref_map_.empty());

  for (auto& task : task_graph->task_nodes()) {
    auto task_type = task->task_type();
    switch (task_type) {
      case TaskNode::TaskType::kOutput:
      case TaskNode::TaskType::kGA:
      case TaskNode::TaskType::kRecv: {
        auto def_ctx = task->def_ctx();
        auto& hlo_module = *task_graph->task_module(task.get());
        auto computation = hlo_module.entry_computation();
        auto output_size = computation->root_instruction()->operand_count();
        for (int64 i = 0; i < output_size; ++i) {
          std::string buf_key = task->name() + "_" + std::to_string(i);
          bool persistant = def_ctx->ag_def_ctx() || def_ctx->ag_slice_def_ctx();
          auto buf_ref_count = CreateBufRefCount(task->node_id(), i, persistant);
          name_buffer_ref_map_[buf_key] = buf_ref_count;
        }
        break;
      }
      default: continue;
    }
  }

  // setup reference count for all tensors
  auto& id_def = task_graph->id_def_map();

  auto parent_def = [&id_def](HloModule::DefContext* def_ctx)
      -> HloModule::DefContext* {
    auto parent_def_id = def_ctx->parent_id();
    CHECK(id_def.count(parent_def_id));
    auto& parent_def_ctx = id_def.at(parent_def_id);
    return parent_def_ctx;
  };

  auto lookup_entry_arg = [parent_def](HloModule::DefContext* def_ctx,
                                       int arg) -> int {
    // This loop must terminate at entry def_ctx
    while (!def_ctx->entry_def_ctx()) {
      //def_ctx = parent_def(def_ctx);
      if (!def_ctx->input_arg_map_.count(arg)) return -1;
      arg = def_ctx->input_arg_map_[arg];
      def_ctx = parent_def(def_ctx);
    }
    return arg;
  };

  auto lookup_parent = [&] (TaskNode* task,
                            int prev_slice_id,
                            HloModule::DefContext* src_def_ctx) -> TaskNode* {
    TaskNode* target = nullptr;
    for (auto parent : task->parents()) {
      if (parent->task_type() == TaskNode::TaskType::kAR) {
        // N.B., In a practical sense, the AR task node in task graph is more
        // like a *fake* node which only performs ncclAllReduce of upstream GA
        // node outputs *in-place*.   Thus we simply skip this node and pass the
        // parent search to the upstream of the AR task.
        parent = parent->parent();
      }
      if (parent->def_ctx() != src_def_ctx) continue;
      auto comm_dev_mgr = parent->comm_dev_mgr();
      CHECK(comm_dev_mgr);
      int slice_id = comm_dev_mgr->AddrToLinearIdxBySplitNums(parent->split_id().ids_);
      if (prev_slice_id != slice_id) continue;
      CHECK(!target);
      target = parent;
    }
    return target;
  };

  for (auto& task : task_graph->task_nodes()) {
    CHECK(task->def_ctx());
    HloModule::DefContext* def_ctx = task->def_ctx();
    auto task_type = task->task_type();
    switch (task_type) {
      case TaskNode::TaskType::kGAInit: {
        /* do nothing */
        break;
      }

      case TaskNode::TaskType::kSplit: {
        break;
      }

      case TaskNode::TaskType::kInput: {
        HloModule* module = task_graph->task_module(task.get());
        for (int arg_no = 0; arg_no < module->entry_computation()->num_parameters();
             ++arg_no) {
          CHECK(def_ctx->input_arg_map_.count(arg_no) +
                def_ctx->input_def_map_.count(arg_no) > 0);

          int parent_arg_no = lookup_entry_arg(def_ctx, arg_no);
          if (parent_arg_no < 0) {
            auto comm_dev_mgr = task->comm_dev_mgr();
            int slice_id = comm_dev_mgr->AddrToLinearIdxBySplitNums(task->split_id().ids_);

            auto src_output_or = def_ctx->get_src_output_from_input_def_map(arg_no, slice_id);
            CHECK(src_output_or.ok());
            HloModule::DefContext::SrcOutput src_output = src_output_or.ValueOrDie();
            int src_def_id = src_output.def_id;
            int src_out_idx = src_output.output_idx;
            int src_slice_id = src_output.prev_slice_id;
            auto iter = id_def.find(src_def_id);
            CHECK(iter != id_def.end());
            auto src_def_ctx = iter->second;
            CHECK(src_def_ctx);
            auto src_task = lookup_parent(task.get(), src_slice_id, src_def_ctx);
            if (!src_task) {
              // src_task is Recv node, thus it has the same slice_id with task
              src_task = lookup_parent(task.get(), slice_id, src_def_ctx);
              CHECK(src_task);
              CHECK(src_task->task_type() == TaskNode::TaskType::kRecv);
              auto& port_map = src_task->port_map();
              CHECK(port_map.find(src_out_idx) != port_map.end());
              this->IncreaseTotalRefCount(src_task, port_map[src_out_idx]);
            } else {
              this->IncreaseTotalRefCount(src_task, src_out_idx);
            }
          }
        }
        break;
      }

      case TaskNode::TaskType::kCompute: {
        break;
      }

      case TaskNode::TaskType::kOutput: {
        break;
      }

      case TaskNode::TaskType::kSend: {
        auto& port_map = task->port_map();
        for (auto& it : port_map) {
          auto arg_no = it.second;
          auto out_idx = it.first;
          if (!task->parent()->def_ctx()->entry_def_ctx()) {
            this->IncreaseTotalRefCount(task->parent(), out_idx);
          }
        }
        break;
      }

      case TaskNode::TaskType::kRecv: {
        break;
      }

      case TaskNode::TaskType::kMerge: {
        for (auto parent : task->parents()) {
          if (parent->task_type() == TaskNode::TaskType::kSend) continue;

          CHECK(parent->task_type() == TaskNode::TaskType::kOutput);
          auto def_ctx = parent->def_ctx();
          for (auto it : def_ctx->output_idx_map_) {
            auto local_out_idx = it.first;
            this->IncreaseTotalRefCount(parent, local_out_idx);
          }// for (output_map)
        }
        break;
      }

      case TaskNode::TaskType::kGA: {
        HloModule* module = task_graph->task_module(task.get());
        for (int arg_no = 0; arg_no < module->entry_computation()->num_parameters();
             ++arg_no) {
          auto comm_dev_mgr = task->comm_dev_mgr();
          int slice_id = task_graph->comm_dev_mgr()->AddrToLinearIdxBySplitNums(task->split_id().ids_);

          auto src_output_or = def_ctx->get_src_output_from_input_def_map(arg_no, slice_id);
          CHECK(src_output_or.ok());
          HloModule::DefContext::SrcOutput src_output = src_output_or.ValueOrDie();
          int src_def_id = src_output.def_id;
          int src_out_idx = src_output.output_idx;
          int src_slice_id = src_output.prev_slice_id;
          auto iter = id_def.find(src_def_id);
          CHECK(iter != id_def.end());
          CHECK(iter->second);
          auto src_task = lookup_parent(task.get(), src_slice_id, iter->second);
          if (src_task->task_type() != TaskNode::TaskType::kGAInit) {
            this->IncreaseTotalRefCount(src_task, src_out_idx);
          }
        }
        break;
      }

      case TaskNode::TaskType::kAR: {
        break;
      }

      default: {
        VLOG(0) << "Unknown task type:" << task->task_type_string();
        CHECK(0);
      }
    }
  }
}

BufferRefCount* OutputBuffersLifeTimeTracker::CreateBufRefCount(
    int node_id, int output_id, bool persistant) {
  buf_ref_counts_pool_.emplace_back(
      std::make_unique<BufferRefCount>(node_id, output_id, 0, persistant));
  return buf_ref_counts_pool_.back().get();
}

void OutputBuffersLifeTimeTracker::IncreaseTotalRefCount(TaskNode* task, int output_id) {
  std::string buf_key = task->name() + "_" + std::to_string(output_id);
  VLOG(2) << "increase ref count for " << buf_key;
  CHECK(name_buffer_ref_map_.count(buf_key));
  auto& buf_ref_count = name_buffer_ref_map_.at(buf_key);
  VLOG(2) << "before increase, total: " << buf_ref_count->total()
          << ", current: " << buf_ref_count->current();
  buf_ref_count->IncreaseTotalRefCount();
}

void OutputBuffersLifeTimeTracker::DecreaseRefCount(TaskNode* task, int output_id) {
  std::string buf_key = task->name() + "_" + std::to_string(output_id);
  VLOG(2) << "decrease ref count for " << buf_key;
  CHECK(name_buffer_ref_map_.count(buf_key));
  auto& buf_ref_count = name_buffer_ref_map_.at(buf_key);
  VLOG(2) << "before decrease, total: " << buf_ref_count->total()
          << ", current: " << buf_ref_count->current();
  buf_ref_count->DecreaseRefCount();
}

bool OutputBuffersLifeTimeTracker::Releasable(TaskNode* task, int output_id) {
  std::string buf_key = task->name() + "_" + std::to_string(output_id);
  CHECK(name_buffer_ref_map_.count(buf_key));
  auto& buf_ref_count = name_buffer_ref_map_.at(buf_key);
  return buf_ref_count->Releasable();
}

void OutputBuffersLifeTimeTracker::ResetBufferRefCounts() {
  for (auto iter : name_buffer_ref_map_) {
    iter.second->Reset();
  }
}

std::vector<std::pair<TaskNode*, int/*output_idx*/>>
OutputBuffersLifeTimeTracker::CollectReleasableTensorsForParent(
    TaskNode* output_task, const TaskNode* input_task) {
  auto& input_specs = input_task->input_specs();
  std::vector<std::pair<TaskNode*, int/*output_idx*/>> tensors_to_release;
  for (auto it : input_specs) {
    int arg_no = it.first;
    auto& specs = it.second;
    for (int i = 0; i < specs.size(); ++i) {
      auto parent_task = specs[i].first;
      if (parent_task->task_type() != TaskNode::TaskType::kOutput) continue;
      // Then, we trace back to the parent computation task node
      auto parent_output_idx = specs[i].second;
      auto parent_compute_task = parent_task->parent();
      CHECK(parent_compute_task->task_type() == TaskNode::TaskType::kCompute);
      
      // We record the gc plan for current task
      this->DecreaseRefCount(parent_task, parent_output_idx); 
      if (this->Releasable(parent_task, parent_output_idx)) {
        VLOG(2) << "to release(parent): " << parent_task->parent()->name() << ", out idx: " << parent_output_idx;
        tensors_to_release.push_back(std::make_pair(
            parent_task->parent(), parent_output_idx));
      }
    }
  }

  return std::move(tensors_to_release);
}

std::vector<std::pair<TaskNode*, int/*output_idx*/>>
OutputBuffersLifeTimeTracker::CollectReleasableTensorsForSelf(
                                                    TaskNode* output_task,
                                                    const TaskDAG* task_graph) {
  auto def_ctx = output_task->def_ctx();
  CHECK(!def_ctx->ag_def_ctx() && !def_ctx->ag_slice_def_ctx());
  auto compute_task = output_task->parent();
  CHECK(compute_task->task_type() == TaskNode::TaskType::kCompute);
  auto& hlo_module = *task_graph->task_module(compute_task);
  auto computation = hlo_module.entry_computation();
  auto output_size = computation->root_instruction()->operand_count();
  std::vector<std::pair<TaskNode*, int/*output_idx*/>> tensors_to_release;
  for (int i = 0; i < output_size; ++i) {
    if (this->Releasable(output_task, i)) {
      VLOG(2) << "to release(self): " << compute_task->name() << ", out idx: " << i;
      tensors_to_release.push_back(std::make_pair(compute_task, i));
    }
  }

  return std::move(tensors_to_release);
}

} // namespace xla

