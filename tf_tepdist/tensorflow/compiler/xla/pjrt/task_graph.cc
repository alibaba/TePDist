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

#include "tensorflow/compiler/xla/pjrt/task_graph.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/core/util/env_var.h"

#include <sstream>
namespace xla {

using std::string;
using std::stringstream;
namespace {

// Construct verbosity node name internally.
std::string make_task_name(std::string type_str, std::string def_ctx_name,
                           const SplitId& split_id, int node_id) {
  // Simplify the name of SEND/RECV/SPLIT/MERGE task node.
  if (type_str == "SPLIT" || type_str == "MERGE") {
    return absl::StrCat(type_str, "_uid", node_id);
  }

  std::string split_str = split_id.HumanReadableStr();

  std::string res;
  if (type_str == "GA" || type_str == "AR") {
    return absl::StrCat(def_ctx_name, split_str, "_uid", node_id);
  } else {
    return absl::StrCat(
        type_str, "_", def_ctx_name, split_str, "_uid", node_id);
  }

  return res;
}

std::unordered_map<int, std::unordered_set<int>> ExtractSharedParamUsageMap(
  HloModule::DefContext* cg_def_ctx) {
  std::unordered_map<int/*parent arg*/,
                     std::unordered_set<int/*stage idx*/>> shared_usage_map;
  for (int i = 0; i < cg_def_ctx->children_.size(); ++i) {
    auto stage_def_ctx = cg_def_ctx->child(i);
    auto computation = stage_def_ctx->module()->entry_computation();
    auto& input_arg_map = stage_def_ctx->input_arg_map_;
    for (auto iter : input_arg_map) {
      shared_usage_map[iter.second].insert(i);
    }
  }

  return shared_usage_map;
}

} // namespace

TaskNode::TaskNode(TaskType type, const std::vector<int64>& addr,
                   const std::vector<bool>& share_dev_flags,
                   int stage_split_ordinal, int node_id,
                   const std::string& def_ctx_name)
    : task_type_(type),
      //micro_id_(-1),  // NOTE(zycao): Need to be further polished.
      split_id_(addr, share_dev_flags, stage_split_ordinal),
      node_id_(node_id),
      device_id_(-1), executable_(nullptr) {
  name_ = make_task_name(TaskNode::TaskTypeString(type), def_ctx_name,
                         split_id_, node_id);
}


TaskNode* TaskDAG::new_task_node(std::string name, int task_type_id,
                                 const SplitId& split_id,
                                 int worker_id,
                                 HloModule::DefContext* def_ctx) {
  TaskNode::TaskType task_type = (TaskNode::TaskType)task_type_id;

  if (task_type == TaskNode::TaskType::kMacro) {
    CHECK(0);
  } else {
    task_nodes_.emplace_back(
        absl::WrapUnique(new TaskNode(task_type, split_id.ids_,
                                      split_id.share_dev_flags_,
                                      split_id.stage_split_ordinal_,
                                      task_nodes_.size(), def_ctx->name())));
  }

  auto node = task_nodes_.back().get();
  node_id_task_[node->node_id()] = node;
  node->set_worker_id(worker_id);
  if (node->task_type() == TaskNode::TaskType::kSplit) {
    CHECK(!source_);
    source_ = node;
  }

  if (node->task_type() == TaskNode::TaskType::kMerge) {
    CHECK(!sink_);
    sink_ = node;
  }
  node->set_executable(nullptr, def_ctx);
  node->set_comm_dev_mgr(comm_dev_mgr_);
  return node;
}

TaskNode*
TaskDAG::brief_clone(const TaskNode* task) {
  CHECK (id_def_map_.count(task->def_id()));

  TaskNode* clone;
  clone = new_task_node(task->name(), int(task->task_type()),
                        task->split_id(), task->worker_id(), task->def_id());
  clone->set_executable(task->exe_or_null(), task->def_ctx());
  clone->set_sched_idx_in_dev(task->sched_idx_in_dev());
  clone->set_device_id(task->device_id());
  clone->set_comm_with_lower_stage(task->comm_with_lower_stage());
  clone->set_across_machine(task->across_machine());
  CHECK(task->input_specs().empty());

  return clone;
}

TaskNode*
TaskDAG::clone(TaskNode* task) {
  CHECK (id_def_map_.count(task->def_id()));

  TaskNode* clone;
  clone = new_task_node(task->name(), int(task->task_type()),
                        task->split_id(), task->worker_id(), task->def_id());
  clone->set_executable(task->exe_or_null(), task->def_ctx());
  clone->set_sched_idx_in_dev(task->sched_idx_in_dev());
  clone->set_device_id(task->device_id());
  clone->set_comm_with_lower_stage(task->comm_with_lower_stage());
  clone->set_across_machine(task->across_machine());
  if (!task->port_map().empty()) {
    *clone->mutable_port_map() = task->port_map();
  }
  if (!task->send_recv_global_devs().empty()) {
    *clone->mutable_send_recv_global_devs() = task->send_recv_global_devs();
  }
  //CHECK(task->input_specs().empty());

  return clone;
}

HloModule* TaskDAG::task_module(const TaskNode* task) const {
  CHECK(exe_plan_);
  return exe_plan_->task_module(task);
}

void TaskDAG::SetupInputSpecs(int worker_count) {
  auto& id_def = id_def_map_;
  auto parent_def = [&id_def](HloModule::DefContext* def_ctx)
      -> HloModule::DefContext* {
    auto parent_def_id = def_ctx->parent_id();
    CHECK(id_def.count(parent_def_id));
    auto parent_def_ctx = id_def[parent_def_id];
    return parent_def_ctx;
  };

  auto lookup_entry_arg = [parent_def](HloModule::DefContext* def_ctx,
                                       int arg) -> int {
    // This loop must terminate at entry def_ctx
    while (!def_ctx->entry_def_ctx()) {
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

  for (auto& task : task_nodes_) {
    auto& mutable_input_specs = *task->mutable_input_specs();
    CHECK(task->def_ctx());
    HloModule::DefContext* def_ctx = task->def_ctx();
    auto task_type = task->task_type();
    switch (task_type) {
      case TaskNode::TaskType::kGAInit: {
        /* do nothing */
        break;
      }

      case TaskNode::TaskType::kSplit: {
        CHECK(def_ctx->entry_def_ctx());
        HloModule* module = this->task_module(task.get());
        for (int arg_no = 0; arg_no < module->entry_computation()->num_parameters();
             ++arg_no) {
          mutable_input_specs[arg_no] = {std::make_pair(nullptr, arg_no)};
        }
        break;
      }

      case TaskNode::TaskType::kInput:
      case TaskNode::TaskType::kGA: {
        HloModule* module = this->task_module(task.get());
        for (int arg_no = 0; arg_no < module->entry_computation()->num_parameters();
            ++arg_no) {
          auto& mutable_pairs = mutable_input_specs[arg_no];

          CHECK(def_ctx->input_arg_map_.count(arg_no) +
                def_ctx->input_def_map_.count(arg_no) > 0);
          
          int parent_arg_no = lookup_entry_arg(def_ctx, arg_no);
          if (parent_arg_no >= 0) {
            auto split_task = this->source();
            if (task->has_parent(split_task)) {
              mutable_pairs.push_back(std::make_pair(split_task,
                                                     parent_arg_no));
            } else {
              TaskNode* entry_recv = nullptr;
              for (auto parent : task->parents()) {
                if (parent->def_ctx()->entry_def_ctx() &&
                    parent->task_type() == TaskNode::TaskType::kRecv) {
                  CHECK(!entry_recv);
                  entry_recv = parent;
                }
              }
              CHECK(entry_recv);

              auto& entry_port_map = entry_recv->port_map();
              CHECK(entry_port_map.count(parent_arg_no));
              mutable_pairs.push_back(std::make_pair(entry_recv,
                                          entry_port_map[parent_arg_no]));
            }
          } else {
            CHECK(def_ctx->input_def_map_.count(arg_no));
            auto comm_dev_mgr = task->comm_dev_mgr();
            int slice_id = comm_dev_mgr->AddrToLinearIdxBySplitNums(task->split_id().ids_);

            auto src_output_or = def_ctx->get_src_output_from_input_def_map(arg_no, slice_id);
            CHECK(src_output_or.ok());
            HloModule::DefContext::SrcOutput src_output = src_output_or.ValueOrDie();
            int src_def_id = src_output.def_id;
            int src_slice_id = src_output.prev_slice_id;
            int src_out_idx = src_output.output_idx;
            CHECK(id_def.find(src_def_id) != id_def.end());
            auto src_def_ctx = id_def[src_def_id];
            CHECK(src_def_ctx);
            auto src_task = lookup_parent(task.get(), src_slice_id, src_def_ctx);
            if (!src_task) {
              // src_task is Recv node, thus it has the same slice_id with task
              src_task = lookup_parent(task.get(), slice_id, src_def_ctx);
              CHECK(src_task);
              CHECK(src_task->task_type() == TaskNode::TaskType::kRecv);
              auto& port_map = src_task->port_map();
              CHECK(port_map.find(src_out_idx) != port_map.end());
              mutable_pairs.push_back(std::make_pair(src_task, port_map[src_out_idx]));
            } else {
              mutable_pairs.push_back(std::make_pair(src_task, src_out_idx));
            }
          }
        }
        break;
      }

      case TaskNode::TaskType::kCompute: {
        HloModule* module = this->task_module(task.get());
        for (int64 i = 0; i < module->entry_computation()->num_parameters(); ++i) {
          mutable_input_specs[i] = {std::make_pair(task->parent(), i)};
        }
        break;
      }

      case TaskNode::TaskType::kOutput: {
        HloModule* module = this->task_module(task.get());
        HloInstruction* root = module->entry_computation()->root_instruction();
        for (int64 i = 0; i < root->operand_count(); ++i) {
          mutable_input_specs[i] = {std::make_pair(task->parent(), i)};
        }
        break;
      }

      case TaskNode::TaskType::kSend: {
        auto& port_map = task->port_map();
        for (auto& it : port_map) {
          auto arg_no = it.second;
          auto out_idx = it.first;
          mutable_input_specs[arg_no] = {std::make_pair(task->parent(), out_idx)};
        }
        break;
      }

      case TaskNode::TaskType::kRecv: {
        auto task_def_ctx = task->def_ctx();
        auto& port_map = task->port_map();
        CHECK(!port_map.empty());
        for (auto& it : port_map) {
          mutable_input_specs[it.second] = {std::make_pair(task->parent(), it.second)};
        }
        break;
      }

      case TaskNode::TaskType::kMerge: {
        std::unordered_set<int> outputs_recorded;
        for (auto parent : task->parents()) {
          if (parent->task_type() == TaskNode::TaskType::kSend) continue;

          CHECK(parent->task_type() == TaskNode::TaskType::kOutput);
          auto def_ctx = parent->def_ctx();
          for (auto it : def_ctx->output_idx_map_) {
            auto local_out_idx = it.first;
            auto global_out_idx = it.second;
            auto parent_def_ctx = def_ctx;
            while (!parent_def_ctx->entry_def_ctx()) {
              parent_def_ctx = parent_def(parent_def_ctx);
              global_out_idx = parent_def_ctx->output_idx_map_[global_out_idx];
            }
            outputs_recorded.insert(global_out_idx);
            auto& mutable_pairs = mutable_input_specs[global_out_idx];
            mutable_pairs.push_back(std::make_pair(parent, local_out_idx));
          }// for (output_map)
        }
        // Ensure that each output index is properly produced.
        //
        // N.B. when gradient_accumulation (GA) tasks are enabled, both
        // compute_gradients (CG) task and gradient_accumulation task contribute to
        // 'outputs_recorded', where the regular outputs (i.e., loss and lr)
        // come from CG task and updated resource variables come from GA task.
        if (worker_count == 1) {
          HloModule* module = this->task_module(task.get());
          HloInstruction* root = module->entry_computation()->root_instruction();
          CHECK_EQ(root->operand_count(), outputs_recorded.size());
        }
        break;
      }

      case TaskNode::TaskType::kAR: {
        HloModule* module = this->task_module(task.get());
        int num_params = module->entry_computation()->num_parameters();
        CHECK(!(num_params & 0x1));
        int64 num_grads = num_params >> 1;
        auto parent = task->parent();
        for (int64 i = 0; i < num_grads; ++i) {
          mutable_input_specs[i] = {std::make_pair(parent, i)};
        }
        break;
      }

      default: {
        VLOG(0) << "Unknown task type:" << task->task_type_string();
        CHECK(0);
      }
    }
  }
}

void TaskDAG::Dump(std::string dag_filename) {
  TaskDAGDumper dag_dumper(*this);
  std::string dot_str = dag_dumper.Dump();

  tensorflow::Env* env = tensorflow::Env::Default();
  Status status = tensorflow::WriteStringToFile(env, dag_filename, dot_str);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write dag graph to " 
               << dag_filename << ": " << status;
  }
}
string TaskDAGDumper::Dump() {
  string head = Header();
  string body = Body();
  string footer = Footer();

  return head+body+footer;
}

string TaskDAGDumper::Header() {
  string header = string("digraph G { \n \
rankdir = TB; \n \
compound = true; \n \
\n \
\n");
  return header;
}

Executable* GetExecutable(TaskNode& nd) {
  return nd.executable();
}

string ModuleName(TaskNode& nd) {
  Executable* exe = GetExecutable(nd);
  if (exe != nullptr) {
    HloModule& mod = exe->module();
    return mod.name();
  } else {
    return string();
  }
}

string NodeName(TaskNode& nd) {
  if (!nd.name().empty()) {
    return nd.name();
  }
  //return nd.name() + "_P" + std::to_string(nd.stage_id());

  stringstream ss;
  ss << nd.task_type_string();
  // NOTE(zycao): micro_id needs to be further polished.
  if (nd.micro_id()>=0) {
    ss << "M" << nd.micro_id();
  }
  if (nd.stage_id()>=0) {
    ss << "T" << nd.stage_id();
  }

  if (nd.worker_id() >= 0) {
    ss << "_W" << nd.worker_id();
  }

  if (nd.device_id()>=0) {
    ss << "_D" << nd.device_id();
  }

  ss << "_" << nd.node_id();

  return ss.str();
}

string TaskDAGDumper::Body() {
  const auto& task_nodes = dag_.task_nodes();
  string body;
  /*
  for (auto& nd : task_nodes) {
    string label = NodeName(*nd) + " [label=\"" + ModuleName(*nd) + "\"];\n";
    body += label;
  }

  body += "\n";
  */

  for (auto& nd : task_nodes) {
    string connect;
    auto& children = nd->children();

    VLOG(2) << "src: " << nd->node_id();
    string src = NodeName(*nd);

    for (auto* child : children) {
      VLOG(2) << "tgt: " << child->node_id();
      connect = src + " -> " + NodeName(*child) + ";\n";
      body += connect;
    }
  }

  return body;
}

string TaskDAGDumper::Footer() {
  return string("}\n");
}

} // namespace xla
