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

#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
//#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/service_env.h"

#include <sstream>
namespace xla {

using std::string;
using std::stringstream;


void LocalPlan::MakeTaskGraphGCPlan() {
  OutputBuffersLifeTimeTracker lifetime_tracker;
  lifetime_tracker.Initialize(task_graph_.get());
  lifetime_tracker.ResetBufferRefCounts();
  for (auto i=0; i<used_device_count(); ++i) {
    auto& schedule_list = task_list(i);
    for (auto& task : schedule_list) {
      auto task_type = task->task_type();
      switch (task_type) {
        case TaskNode::TaskType::kOutput: {
          // We find the corresponding input node
          auto input_task = task->parent()->parent();
          auto parent_to_release = lifetime_tracker.CollectReleasableTensorsForParent(task, input_task);
          for (auto& it : parent_to_release) {
            task->mem_to_release().push_back(std::make_pair(it.first->node_id(), it.second));
          }
          auto def_ctx = task->def_ctx();
          if (!def_ctx->ag_def_ctx() && !def_ctx->ag_slice_def_ctx()) {
            auto self_to_release = lifetime_tracker.CollectReleasableTensorsForSelf(task, task_graph());
            for (auto& it : self_to_release) {
              task->mem_to_release().push_back(std::make_pair(it.first->node_id(), it.second));
            }
          }
          break; 
        }

        case TaskNode::TaskType::kSend:
        case TaskNode::TaskType::kGA: {
          auto parent_to_release = lifetime_tracker.CollectReleasableTensorsForParent(task, task);
          for (auto& it : parent_to_release) {
            task->mem_to_release().push_back(std::make_pair(it.first->node_id(), it.second));
          }
          break;
        }

        default: continue;
      }
    }
  }
}

// Topological execution order
void LocalPlan::BuildTaskSpan(
    std::unordered_map<TaskNode*, int>& task_span) {
  auto source = task_graph_->source();

  task_span[source ] = 0;
  std::deque<TaskNode*> worklist({source});

  while (!worklist.empty()) {
    auto task = worklist.front();
    worklist.pop_front();

    for (auto child : task->children()) {
      if (task_span.count(child)) continue;

      int max_operand_span = -1;
      bool all_allocated = true;
      for (auto parent : child->parents()) {
        if (!task_span.count(parent)) {
          all_allocated = false;
          break;
        }

        if (task_span[parent] > max_operand_span) {
          max_operand_span = task_span[parent];
        }
      } // for
      if (all_allocated) {
        task_span[child] = max_operand_span + 1;
        worklist.push_back(child);
      }
    }
  } // while

  for (auto& task : task_graph_->task_nodes()) {
    CHECK(task_span.count(task.get()));
  }
}

bool LocalPlan::Ready(TaskNode* task, 
                          std::unordered_set<TaskNode*>& sched) {
  bool ready = true;
  for (auto parent : task->parents()) {
    if (!sched.count(parent)) {
      ready = false;
      break;
    }
  }
  return ready;
}

void
LocalPlan::RestoreLocalSchedule() {
  auto* top_mod = dist_plan_->top_module();
  CHECK(top_mod);

  auto& nodes = task_graph_->task_nodes();  // get local graph's nodes
  TaskNode* source = nullptr;
  TaskNode* sink = nullptr;
  CHECK(scheduled_tasks_.size() == 0);
  VLOG(0) << "[RestoreLocalSchedule] local_device_count_: " << local_device_count_;
  scheduled_tasks_.push_back(absl::make_unique<ScheduleInMachine>(local_device_count_));
  auto& per_device_tasks = scheduled_tasks_[0]->per_device_tasks_;
  for (auto& node : nodes) {
    if (node->task_type() == TaskNode::TaskType::kSplit) {
      CHECK(source == nullptr);
      source = node.get();      // only for check may remove it later
      continue;
    } else if (node->task_type() == TaskNode::TaskType::kMerge) {
      CHECK(sink == nullptr);
      sink = node.get();        // only for check may remove it later
      continue;
    }
    auto dev_id = node->device_id();
    VLOG(1) << "[RestoreLocalSchedule] task: " << node->name() << ", dev id: "
            << dev_id << ", wid: " << node->worker_id()
            << ", sched_idx_in_dev: " << node->sched_idx_in_dev();
    if (dev_id >= 0) {
      CHECK(node->sched_idx_in_dev() >= 0);
      CHECK(dev_id < per_device_tasks.size());
      per_device_tasks[dev_id].push_back(node.get());
    }
  }

  for (auto& tasks : per_device_tasks) {
    // sort node by sched_idx_in_dev
    // sched_idx_in_dev is set during whole graph scheduling
    std::sort(tasks.begin(), tasks.end(), 
              [](TaskNode* nd1, TaskNode* nd2) {
      return nd1->sched_idx_in_dev() < nd2->sched_idx_in_dev();
    });
  }
}

void LocalPlan::ShowPerDeviceTaskList() {
  VLOG(0) << "Per Device Task List:";
  CHECK(scheduled_tasks_.size() == 1);
  auto& per_device_tasks = scheduled_tasks_[0]->per_device_tasks_;
  for (int device_id = 0; device_id < per_device_tasks.size(); ++device_id) {
    auto& device_tasks = per_device_tasks[device_id];

    for (auto task : device_tasks) {
      VLOG(0) << "Task->" << task->name()
              << " type:" << task->task_type_string()
              << " def-ctx:" << task->def_ctx()->name()
              << " micro:" << task->micro_id()
              << " stage:" << task->stage_id()
              << " device-id:" << device_id << "\n";
    }
  }
}

void BufferReuseAnalysis::SetReuseInfo(TaskNode* task) {
  // 1. classify recv task, use pair<port_map,def_id> as key 
  auto task_root = task->def_ctx()->module()->entry_computation()->root_instruction();
  auto& port_map = task->port_map();
  CHECK(!port_map.empty());
  std::vector<int> parent_out_index;
  std::vector<Shape> buffer_shape;
  for (auto it : port_map) {
    parent_out_index.push_back(it.first);
    buffer_shape.push_back(task_root->operand(it.first)->shape());
  }
  auto key = std::make_pair(parent_out_index, task->def_id());
  if(task_classify_.find(key) == task_classify_.end()) {
    task_classify_[key].first = recv_type_count_;
    task_classify_[key].second = 1;
    current_buffer_index_[recv_type_count_] = 0;
    recv_type_count_ ++;
  } else {
    task_classify_[key].second ++;
  }

  // 2. record buffer reuse info into recv task
  int64 buffer_cnt = ServiceEnv::group_sched_count();
  auto& split_id = task->split_id();
  std::shared_ptr<CommDevManager> comm_dev_mgr = task->comm_dev_mgr();
  int global_device = comm_dev_mgr->global_dev_id(split_id);
  int local_device = comm_dev_mgr->local_dev_id(split_id);
  int buffer_type = task_classify_[key].first;
  bool buffer_reused = task_classify_[key].second > buffer_cnt;
  int buffer_id = current_buffer_index_[buffer_type];
  // update next buffer id to use of specific buffer type
  // at present, current_buffer_index_ is linear increase, it may change strategy in the future
  current_buffer_index_[buffer_type] = (current_buffer_index_[buffer_type] + 1) % buffer_cnt;
  
  auto buffer_info = BufferInfo(global_device, local_device, buffer_id, buffer_type, buffer_reused, buffer_shape);
  task->set_buffer_info(buffer_info);
}

void BufferReuseAnalysis::InsertEventForSynchronizing(TaskNode* task) {
  // 1. First creat self barrier
  int saved_device = 0;
  int local_device = task->device_id();
  CUDACHECK(cudaGetDevice(&saved_device));
  CUDACHECK(cudaSetDevice(local_device));
  std::shared_ptr<cudaEvent_t> barrier = task->CreateBufferBarrier();
  TaskNode* find_output_task = task;
  // At present, recv-input-compute-output only have one child, so the buffer liveness of recv ends at the end of output 
  while(find_output_task->task_type() != TaskNode::TaskType::kOutput) {
    find_output_task = find_output_task->child();
  }
  find_output_task->set_buffer_barrier(barrier);
  find_output_task->set_buffer_record_barrier(true);

  // 2.then set wait barrier if reuse buffer
  int buffer_type = task->buffer_info().buffer_type();
  int buffer_id = task->buffer_info().buffer_id();
  bool buffer_reused = task->buffer_info().buffer_reused();
  auto event_key = std::make_pair(buffer_type, buffer_id);
  if(buffer_reused) {
    CHECK(pre_event_.find(event_key) != pre_event_.end());
    task->set_buffer_wait_barrier(pre_event_[event_key]);
  }
  pre_event_[event_key] = barrier;
  CUDACHECK(cudaSetDevice(saved_device)); 
}

/**
 * @brief This function is used to record buffer reuse info for recv tasknode
 *        First, classify recv task because different recv may use different buffer size
 *               and record recv buffer reuse info to task, which will use in execution stage
 *        Then, insert cudaevent barrier for synchronizing recv tasknode that use the same buffer
 */
void LocalPlan::RecordBufferReuseInfo() {
  CHECK(ServiceEnv::buffer_save());
  CHECK(scheduled_tasks_.size() == 1);
  for(int dev_id=0; dev_id<scheduled_tasks_[0]->per_device_tasks_.size(); ++dev_id) {
    std::vector<TaskNode*>& device_tasks = scheduled_tasks_[0]->per_device_tasks_[dev_id];
    BufferReuseAnalysis buffer_reuse_per_device = BufferReuseAnalysis();
    
    for(TaskNode* task : device_tasks) {
      if (task->task_type() != TaskNode::TaskType::kRecv) continue;
      // At present, recv task only has one parent and one child
      CHECK(task->parents().size() == 1);
      CHECK(task->children().size() == 1);
      
      // 1. classify recv task and record buffer reuse info to recv task
      buffer_reuse_per_device.SetReuseInfo(task);

      // 2. insert cudaevent barrier
      buffer_reuse_per_device.InsertEventForSynchronizing(task);
    }
  }
}

void DistributedPlan::WorkerAssignment() {
  std::shared_ptr<CommDevManager> comm_dev_mgr = task_graph_->comm_dev_mgr();
  CHECK(comm_dev_mgr);

  int worker_id;
  for (auto& task : task_graph_->task_nodes()) {
    worker_id = comm_dev_mgr->worker_id(task->split_id());
    task->set_worker_id(worker_id); 
    VLOG(2) << "Task:" << task->name()
            << " Worker:" << task->worker_id();
  }
}

void DistributedPlan::DeviceAssignment() {
  std::shared_ptr<CommDevManager> comm_dev_mgr = task_graph_->comm_dev_mgr();
  CHECK(comm_dev_mgr);

  int dev_id;
  for (auto& task : task_graph_->task_nodes()) {
    if (task->task_type() == TaskNode::TaskType::kSplit ||
        task->task_type() == TaskNode::TaskType::kMerge) {
      // Set Split/Merge task with no device_id.
      task->set_device_id(-1);
      continue;
    }

    dev_id = comm_dev_mgr->local_dev_id(task->split_id());
    task->set_device_id(dev_id);
  }
}

void DistributedPlan::SetupDefExeMap(
    std::vector<std::pair<HloModule::DefContext*,
                std::unique_ptr<Executable>>>& def_exe_pairs) {
  for (auto& it : def_exe_pairs) {
    def_exe_map_[it.first] = std::move(it.second);
  }
}

void DistributedPlan::ExeDecoration() {
  auto& tasks = task_graph_->task_nodes();
  int64 tasks_size = tasks.size();
  for (int64 i = 0; i < tasks_size; ++i) {
    auto task = tasks[i].get();
    auto def_ctx = task->def_ctx();
    if (def_exe_map_.count(def_ctx)) {
      task_graph_->add_executable(task, def_exe_map_[def_ctx].get());
    }
  }// for
}

void
ExecutionPlan::ScheduleTasks() {
  int max_micro_id = 0;
  std::vector<int> worker_dev_cnt(worker_count_, 0);
  // determine the used gpu number on each machine
  for (auto& task : task_graph_->task_nodes()) {
    auto wid = task->worker_id();
    CHECK(wid<worker_count_);
    auto dev_id = task->device_id();
    if (wid >= 0) {
      if (dev_id>=worker_dev_cnt[wid]) {
        worker_dev_cnt[wid] = dev_id+1;
      }
    }

    if (task->micro_id() > max_micro_id) {
      max_micro_id = task->micro_id();
    }
  }

  int64 micro_num_limit = max_micro_id + 1;
  std::shared_ptr<CommDevManager> comm_dev_mgr = task_graph_->comm_dev_mgr();
  int num_stages = comm_dev_mgr->num_stages();
  int max_forward_micro_num = num_stages;
  //int max_forward_micro_num = num_stages * 2 - 1;
  if (micro_num_limit > max_forward_micro_num) {
    micro_num_limit = max_forward_micro_num;
  }

  if (ServiceEnv::micro_num_limit() > 0)
    micro_num_limit = ServiceEnv::micro_num_limit();

  VLOG(2) << "forward stage num: " << num_stages;
  VLOG(2) << "micro_num_limit: " << micro_num_limit;
  VLOG(2) << "worker_count_: " << worker_count_;

#if 0
  while (micro_num_limit > 0) {
    VLOG(0) << "micro_num_limit: " << micro_num_limit;
    scheduled_tasks_.clear();
    for (int wid=0; wid<worker_count_; ++wid) {
      scheduled_tasks_.push_back(absl::make_unique<ScheduleInMachine>(worker_dev_cnt[wid]));
    }
    TaskScheduler scheduler(task_graph_.get(), worker_dev_cnt);
    if (scheduler.Schedule(micro_num_limit, scheduled_tasks_) == true) {
      // debug
      VLOG(0) << "cluster schedule result:";
      for (int wid=0; wid<scheduled_tasks_.size(); ++wid) {
        VLOG(0) << std::endl << "wid: " << wid << std::endl
                << scheduled_tasks_[wid]->ToString();
      }
      return;
    }

    VLOG(0) << "OOM happens when micro_num_limit is " << micro_num_limit;
    --micro_num_limit;
  }
#else
  scheduled_tasks_.clear();
  for (int wid=0; wid<worker_count_; ++wid) {
    scheduled_tasks_.push_back(absl::make_unique<ScheduleInMachine>(worker_dev_cnt[wid]));
  }
  int64 sched_cnt = ServiceEnv::group_sched_count();

  TaskScheduler scheduler(task_graph_.get(), worker_count_, worker_dev_cnt, sched_cnt);
  if (scheduler.Schedule(micro_num_limit, scheduled_tasks_) == true) {
    // debug
    VLOG(0) << "cluster schedule result:";
    for (int wid=0; wid<scheduled_tasks_.size(); ++wid) {
      VLOG(0) << std::endl << "wid: " << wid << std::endl
              << scheduled_tasks_[wid]->ToString();
    }
    return;
  } else {
    // estimation is not work
    return;
  }
#endif

  will_be_oom_ = true;
  LOG(ERROR) << "Fail to build execution plan because OOM happens!";
}

void ExecutionPlan::SourceCalibration(bool has_sub_def) {
  // Process kSplit->*** edges
  std::map<int/*worker_id*/, std::unordered_set<TaskNode*>> ext_tasks;
  auto split = task_graph_->source();
  for (auto child : split->children()) {
    if (child->worker_id() == split->worker_id()) continue;
    if (child->task_type() == TaskNode::TaskType::kGAInit) continue;
    auto& tasks = ext_tasks[child->worker_id()];
    tasks.insert(child);
  }

  std::vector<int64> dummy_addr;
  std::vector<bool> dummy_flags;
  for (auto& it : ext_tasks) {
    auto ext_id = it.first;
    auto& tasks = ext_tasks[ext_id];

    // Build kSend task
    auto send = task_graph_->new_task_node(TaskNode::TaskType::kSend,
                                           dummy_addr, dummy_flags, // will be placed on host
                                           -1/*stage_split_ordinal*/, {split});
    send->set_executable(nullptr, split->def_ctx());
    send->set_worker_id(split->worker_id());
    send->set_name(split->name() + "_worker" + std::to_string(ext_id) + "_Send");
    VLOG(2) << "wid: " << ext_id << ", send name: " << send->name();
    auto& port_map = send->port_map();

    std::set<int> ports;
    auto& id_def_map = task_graph_->id_def_map();

    // debug codes:
    //--------------------------------------------------------
    for (auto& id_def : id_def_map) {
      VLOG(2) << "id: " << id_def.first << ", def_ctx: " << id_def.second->name() << ", def_ctx parent id: " << id_def.second->parent_id();
    }
    //--------------------------------------------------------

    for (auto task : tasks) {
      auto def_ctx = task->def_ctx();
      VLOG(2) << "task def ctx id: " << def_ctx->def_id() << ", name: " << def_ctx->name() << ", parent id: " << def_ctx->parent_id();

      HloModule::DefContext* top_level_ctx;
      if (def_ctx->parent_id() >= 0) {
        top_level_ctx = id_def_map.at(def_ctx->parent_id());
      } else {
        top_level_ctx = def_ctx;
      }
      VLOG(2) << "top_level_ctx: " << top_level_ctx->name();

      for (auto& arg_it : def_ctx->input_arg_map_) {
        if (top_level_ctx->input_arg_map_.find(arg_it.second) != top_level_ctx->input_arg_map_.end()) {
          ports.insert(top_level_ctx->input_arg_map_[arg_it.second]);
        }
      }
    }
    int port_id = 0;
    for (auto port : ports) {
      port_map[port] = port_id++;
    }

    // Build kRecv task
    auto recv = task_graph_->new_task_node(TaskNode::TaskType::kRecv,
                                              dummy_addr, dummy_flags, // will be placed on host
                                              -1/*stage_split_ordinal*/, {send});
    recv->set_executable(nullptr, split->def_ctx());
    recv->set_worker_id(ext_id);
    recv->set_name(split->name() + "_worker" + std::to_string(ext_id) + "_Recv");
    // Do not worry too much since port map is small.
    *recv->mutable_port_map() = port_map;
    VLOG(2) << "wid: " << ext_id << ", recv name: " << recv->name();

    // Reconnect kRecv->task
    for (auto task : tasks) {
      VLOG(0) << "[kRecv->task]: kRecv: " << recv->name() << ", task: "
              << task->name() << ", task wid: " << task->worker_id();
      split->remove_child(task);
      task->remove_parent(split);
      recv->add_child(task);
      task->add_parent(recv);
    }
  }
}

void ExecutionPlan::CrossDeviceCalibration() {
  std::vector<TaskNode*> task_nodes_copy;
  task_nodes_copy.reserve(task_graph_->task_nodes().size());
  for (auto& task : task_graph_->task_nodes()) {
    task_nodes_copy.emplace_back(task.get());
  }

  std::shared_ptr<CommDevManager> comm_dev_mgr = task_graph_->comm_dev_mgr();
  CHECK(comm_dev_mgr);

  int dev_num_per_worker = comm_dev_mgr->dev_num_per_worker();
  for (auto parent : task_nodes_copy) {
    if (parent == task_graph_->source()) continue;
    if (parent->task_type() == TaskNode::TaskType::kSend ||
        parent->task_type() == TaskNode::TaskType::kRecv) continue;
    CHECK(parent->def_ctx());
    auto parent_def_ctx = parent->def_ctx();
    // Make a copy to avoid iterating a changing container
    auto children_copy = parent->children();
    for (auto child : children_copy) {
      if (child == task_graph_->sink()) continue;
      if (child->worker_id() == parent->worker_id() &&
          child->device_id() == parent->device_id()) continue;
      CHECK(child->task_type() != TaskNode::TaskType::kRecv);

      std::set<int> ports;
      auto child_def_ctx = child->def_ctx();
      CHECK(child_def_ctx);
      HloComputation* entry = child_def_ctx->module()->entry_computation();
      for (int p = 0; p < entry->num_parameters(); ++p) {
        int64 slice_id = task_graph_->comm_dev_mgr()->AddrToLinearIdxBySplitNums(child->split_id().ids_);
        auto src_output_or = child_def_ctx->get_src_output_from_input_def_map(p, slice_id);
        if (!src_output_or.ok()) continue;
        auto src_output = src_output_or.ValueOrDie();
        if (src_output.def_id == parent_def_ctx->def_id()) {
          ports.insert(src_output.output_idx);
        }
      }

      if (ports.empty()) {
        continue;
      }

      VLOG(1) << "Build Send/Recv: " << parent->name()
              << "->" << child->name();

      int64 send_global_dev_id = comm_dev_mgr->global_dev_id(parent->split_id());
      int64 recv_global_dev_id = comm_dev_mgr->global_dev_id(child->split_id());

      // Build kSend task
      auto send = task_graph_->new_task_node(TaskNode::TaskType::kSend,
                                       parent->split_id().ids_,
                                       parent->split_id().share_dev_flags_,
                                       parent->split_id().stage_split_ordinal_,
                                       {parent});
      send->set_executable(nullptr, parent->def_ctx());
      send->set_worker_id(parent->worker_id());
      send->set_device_id(parent->device_id());
      *send->mutable_send_recv_global_devs() = {send_global_dev_id, recv_global_dev_id};
      VLOG(1) << "[Send] " << parent->name() << ", device_id = " << send->device_id();
      auto& port_map = send->port_map();

      int port_id = 0;
      for (auto port : ports) {
        port_map[port] = port_id++;
      }

      // Build kRecv task
      auto recv = task_graph_->new_task_node(TaskNode::TaskType::kRecv,
                                       child->split_id().ids_,
                                       child->split_id().share_dev_flags_,
                                       child->split_id().stage_split_ordinal_,
                                       {send});
      recv->set_executable(nullptr, parent->def_ctx());
      recv->set_worker_id(child->worker_id());
      // Do not worry too much since port map is small.
      *recv->mutable_port_map() = port_map;

      *recv->mutable_send_recv_global_devs() = {send_global_dev_id, recv_global_dev_id};
      // Reconnect kRecv->task
      parent->remove_child(child);
      child->remove_parent(parent);
      recv->add_child(child);
      child->add_parent(recv);
      recv->set_worker_id(child->worker_id());
      recv->set_device_id(child->device_id());

    } // for (child)
  } // for (task_graph_->task_nodes)
  task_graph_->Dump("stage_dag_with_send_recv.dot");
}

} // namespace xla
