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

#include "tensorflow/compiler/xla/pjrt/task_scheduler.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/service_env.h"

namespace xla {

namespace {

bool InOrder(const TaskNode* first, const TaskNode* second,
             ClusterState& cluster_state) {
  CHECK(first);
  CHECK(second);
  auto* first_def_ctx = first->def_ctx();
  auto* second_def_ctx = second->def_ctx();

  CHECK(first_def_ctx->stage_type_ != HloModule::DefContext::StageType::BOTH) << first->name();
  if (first_def_ctx->stage_type_ == HloModule::DefContext::StageType::NA &&
      second_def_ctx->stage_type_ != HloModule::DefContext::StageType::NA) {
    return true;
  }

  CHECK(second_def_ctx->stage_type_ != HloModule::DefContext::StageType::BOTH) << second->name();
  if (second_def_ctx->stage_type_ == HloModule::DefContext::StageType::NA &&
      first_def_ctx->stage_type_ != HloModule::DefContext::StageType::NA) {
    return false;
  }

  if (first->task_type() == TaskNode::TaskType::kSend &&
      second->task_type() != TaskNode::TaskType::kSend) {
    return true;
  }

  if (second->task_type() == TaskNode::TaskType::kSend &&
      first->task_type() != TaskNode::TaskType::kSend) {
    return false;
  }

  VLOG(2) << "first " << first->name() << ", m:" << first->micro_id()
            << "\nsecond " << second->name() << ", m:" << second->micro_id();
  if (first_def_ctx->stage_type_ == HloModule::DefContext::StageType::FORWARD &&
      second_def_ctx->stage_type_ == HloModule::DefContext::StageType::BACKWARD) {
    VLOG(2) << first->name() << " is placed before " << second->name();
    return true;
  } else if (first_def_ctx->stage_type_ == HloModule::DefContext::StageType::BACKWARD &&
             second_def_ctx->stage_type_ == HloModule::DefContext::StageType::FORWARD) {
    VLOG(2) << second->name() << " is placed before " << first->name();
    return false;
  }

  if (second->micro_id() < first->micro_id()) {
    VLOG(2) << "replace " << first->name() << "(m:" << first->micro_id()
            << ") with " << second->name() << "(m:" << second->micro_id() << ")";
    VLOG(2) << second->name() << " is placed before " << first->name();
    return false;
  } else if (second->micro_id() > first->micro_id()) {
    VLOG(2) << first->name() << " is placed before " << second->name();
    return true;
  }

  const TaskNode* orig_task = first;
  const TaskNode* new_task = second;
  if (orig_task->task_type() == TaskNode::TaskType::kSend) {
    auto* recv_task = orig_task->children().front();
    orig_task = recv_task->children().front();
  }
  if (new_task->task_type() == TaskNode::TaskType::kSend) {
    auto* recv_task = new_task->children().front();
    new_task = recv_task->children().front();
  }
  int rank = cluster_state.FindRank(orig_task);
  int new_rank = cluster_state.FindRank(new_task);

  if (new_rank < rank) {
    VLOG(2) << second->name() << " is placed before " << first->name();
    return false;
  } else if (new_rank > rank) {
    VLOG(2) << first->name() << " is placed before " << second->name();
    return true;
  }

  if (!(first->split_id() < second->split_id())) {
    return false;
  }

  return true;
}

int stage_micro_num_limit(const TaskNode* task, int micro_num_limit) {
  int stage_id = task->stage_id();
  std::shared_ptr<CommDevManager> dev_mngr = task->comm_dev_mgr();
  int num_stages = 1;
  if (dev_mngr) {
    num_stages = dev_mngr->num_stages();
  }
  int physical_stage = 0;
  if (stage_id>=0 && num_stages>1) {
    CHECK(stage_id<num_stages);
    physical_stage = stage_id;
  }
  int diff = 1;
  int stage_limit = micro_num_limit - (physical_stage*diff);

  VLOG(2) << "num_stages: " << num_stages << ", cur_dev_micro_num_limit: " << stage_limit
          << ", physical stage: " << physical_stage << ", " << task->name();

  return stage_limit;
}

bool should_ignore_by_sched_id(const TaskNode* task,
                               const ScheduleIdInfo& sched_id) {
  int mid = task->micro_id();
  int dev_id = task->device_id();
  // NOTE (zycao): To ignore a task not belong to current task group during the
  // group scheduling phase, we follow the two criterians below:
  //   1) the micro_id is not -1 and does not match current task group (GA).
  //   2) the micro_id is -1 and it is not a host task (e.g. micro_id for SPLIT
  //      task is always -1), then only the first task group should schedule it
  //      (SPMD only).
  return (mid >= 0 && mid % sched_id.sched_size_ != sched_id.id_) ||
         (mid < 0 && dev_id >= 0 && sched_id.id_ > 0);
}

void MergeDeviceSchedule(std::vector<std::vector<TaskNode*>*> task_group,
                         std::vector<TaskNode*>& merged_sched) {
  CHECK(merged_sched.empty());
  int max_micro_id = -1;
  for (int i = 0; i < task_group.size(); ++i)
    for (TaskNode* task : *task_group[i])
      if (task->micro_id() > max_micro_id) max_micro_id = task->micro_id();
  VLOG(1) << "max_micro_id: " << max_micro_id;
      
  std::vector<int> anchors(task_group.size(), 0);
  int finished = 0, head = 0, merged_size = 0;
  // NOTE(zycao): in case of micro batch num is smaller then group count.
  for (int i = 0; i < task_group.size(); ++i)
    if (task_group[i]->size() == 0) finished++;
  while (finished < task_group.size()) {
    while (anchors[head] == task_group[head]->size()) head++;
    std::vector<TaskNode*>& head_sched = *task_group[head];
    VLOG(1) << head << "-anchors[" << head << "]: " << anchors[head]
            << ", task: " << head_sched[anchors[head]]->name()
            << ", finished: " << finished;

    if (finished == task_group.size() - 1) {
      while(anchors[head] < head_sched.size()) {
        head_sched[anchors[head]]->set_sched_idx_in_dev(merged_size++);
        merged_sched.emplace_back(head_sched[anchors[head]]);
        anchors[head]++;
      }
      finished++;
      continue;
    }
    int last_micro_id = -1;
    switch(head_sched[anchors[head]]->task_type()) {
      case TaskNode::TaskType::kGAInit: {
        head_sched[anchors[head]]->set_sched_idx_in_dev(merged_size++);
        merged_sched.emplace_back(head_sched[anchors[head]]);
        anchors[head]++;
        break;
      }
      case TaskNode::TaskType::kInput: {
        last_micro_id = head_sched[anchors[head]]->micro_id();
        for (int i = head; i < task_group.size(); ++i) {
          std::vector<TaskNode*>& cur_sched = *task_group[i];
          if (anchors[i] == cur_sched.size()) continue;
          if (i > head && last_micro_id == max_micro_id) continue;
          VLOG(1) << i << "-" << anchors[i] << ": " << cur_sched[anchors[i]]->name();
          CHECK(anchors[i] + 2 < cur_sched.size());
          CHECK(cur_sched[anchors[i]]->task_type() == TaskNode::TaskType::kInput)
            << i << "-" << anchors[i] << ": " << cur_sched[anchors[i]]->name();
          int cur_micro_id = cur_sched[anchors[i]]->micro_id();
          CHECK(i == head || cur_micro_id > last_micro_id);
          cur_sched[anchors[i]]->set_sched_idx_in_dev(merged_size++);
          merged_sched.emplace_back(cur_sched[anchors[i]]);
          anchors[i]++;
          CHECK(cur_sched[anchors[i]]->task_type() == TaskNode::TaskType::kCompute);
          CHECK(cur_sched[anchors[i]]->micro_id() == cur_micro_id);
          cur_sched[anchors[i]]->set_sched_idx_in_dev(merged_size++);
          merged_sched.emplace_back(cur_sched[anchors[i]]);
          anchors[i]++;
          CHECK(cur_sched[anchors[i]]->task_type() == TaskNode::TaskType::kOutput);
          CHECK(cur_sched[anchors[i]]->micro_id() == cur_micro_id);
          cur_sched[anchors[i]]->set_sched_idx_in_dev(merged_size++);
          merged_sched.emplace_back(cur_sched[anchors[i]]);
          anchors[i]++;
          last_micro_id = cur_micro_id;
          if (anchors[i] == cur_sched.size()) finished++;
        }
        break;
      }
      default: {
        int anchor_head = anchors[head];
        last_micro_id = head_sched[anchors[head]]->micro_id();
        for (int i = head; i < task_group.size(); ++i) {
          std::vector<TaskNode*>& cur_sched = *task_group[i];
          if (anchors[i] == cur_sched.size()) continue;
          if (i > head && last_micro_id == max_micro_id) continue;
          VLOG(1) << i << "-" << anchors[i] << ": " << cur_sched[anchors[i]]->name();
          int cur_micro_id = cur_sched[anchors[i]]->micro_id();
          CHECK(cur_sched[anchors[i]]->task_type() == head_sched[anchor_head]->task_type())
            << cur_sched[anchors[i]]->name() << " --> " << head_sched[anchor_head]->name();
          CHECK(i == head || cur_micro_id > last_micro_id)
            << cur_micro_id << " vs " << last_micro_id;
          cur_sched[anchors[i]]->set_sched_idx_in_dev(merged_size++);
          merged_sched.emplace_back(cur_sched[anchors[i]]);
          anchors[i]++;
          last_micro_id = cur_micro_id;
          if (anchors[i] == cur_sched.size()) finished++;
        }
      }
    }
  }
}

void MergeMachineSchedule(std::vector<std::unique_ptr<ScheduleInMachine>*> &sched_group,
                          ScheduleInMachine& merged_sched) {
  CHECK(merged_sched.per_device_tasks_.size() ==
        (*sched_group[0])->per_device_tasks_.size()); // same device num
  for (int i = 1; i < sched_group.size(); ++i) {
    CHECK((*sched_group[0])->per_device_tasks_.size()
          == (*sched_group[i])->per_device_tasks_.size()); // same device num
  }

  for (int j = 0; j < (*sched_group[0])->per_device_tasks_.size(); ++j) {
    std::vector<std::vector<TaskNode*>*> task_group;
    for (int i = 0; i < sched_group.size(); ++i)
      task_group.push_back(&((*sched_group[i])->per_device_tasks_[j]));
    MergeDeviceSchedule(task_group, merged_sched.per_device_tasks_[j]);
  }
}

void MergeClusterSchedule(
      std::vector<std::vector<std::unique_ptr<ScheduleInMachine>>>& sub_schedules,
      std::vector<std::unique_ptr<ScheduleInMachine>>& merged_cluster_sched) {
  CHECK(sub_schedules.size() >= 1);
  CHECK(merged_cluster_sched.size() == sub_schedules[0].size()); // same num of machines
  for (int i = 1; i < sub_schedules.size(); ++i)
    CHECK(sub_schedules[0].size() == sub_schedules[i].size());   // same num of machines
  for (int j = 0; j < sub_schedules[0].size(); ++j) {
    std::vector<std::unique_ptr<ScheduleInMachine>*> sched_group;
    for (int i = 0; i < sub_schedules.size(); ++i)
      sched_group.push_back(&(sub_schedules[i][j]));
    MergeMachineSchedule(sched_group, *merged_cluster_sched[j]);
  }
}


} // namespace

std::string ScheduleInMachine::ToString() const {
  std::string res;
  for (int i=0; i<per_device_tasks_.size(); ++i) {
    res += "  dev: " + std::to_string(i);

    for (auto* nd : per_device_tasks_[i]) {
      res += "\n    " + nd->name();
    }
    res += "\n";
  }

  return res;
}

void DevState::AddReadyNode(TaskNode* task) {
  ready_nodes_.insert(task);
}

void DevState::ScheduleTask(TaskNode* task, float start_time) {
  float runtime_ms = EstimateTime(task);

  ScheduledTaskInfo task_info(task, start_time, start_time+runtime_ms);
  scheduled_tasks_.push_back(task_info);
  ready_nodes_.erase(task);
}

TaskNode* DevState::PopOneTask(const int micro_num_limit,
                               ClusterState& cluster_state) {
  CHECK(!ready_nodes_.empty());

  const TaskNode* prefer_task = nullptr;
  // Make sure INPUT-COMPUTE-OUTPUT should be scheduled as a bundle.
  if (!finished_sequence_.empty()) {
    ScheduledTaskInfo last_task_info = finished_sequence_[finished_sequence_.size()-1];
    TaskNode* last_task = last_task_info.task_;
    TaskNode::TaskType last_task_type = last_task->task_type();
    switch (last_task_type) {
    case TaskNode::TaskType::kInput:
    case TaskNode::TaskType::kCompute:
      prefer_task = last_task->children().front();
      break;
    default:
      break;
    }
  }

  // Looking for ready GA node and set as preferred, to release variable memory.
  if (prefer_task == nullptr) {
    for (TaskNode* task : ready_nodes_) {
      if (task->task_type() == TaskNode::TaskType::kGA) {
        prefer_task = task;
        break;
      }
    }
  }
  // Heuristic: task with small micro id owns higher schedule priority because
  // we hope to execute backward steps as early as possible. This strategy
  // allow release GPU memory earlier.
  // The strategy is naive. It may be changed later.
  std::unordered_set<TaskNode*>::iterator it = ready_nodes_.begin();
  std::unordered_set<TaskNode*>::iterator selected_it = ready_nodes_.end();
  bool skip;
  VLOG(2) << "dev state of wid " << wid_ << ", dev id: " << dev_id_;
  std::string nodes_str;
  for (; it != ready_nodes_.end(); ++it) {
    if ((*it) == prefer_task) {
      selected_it = it;
      break;
    }

    if (should_ignore_by_sched_id(*it, sched_id_)) {
      continue;
    }

    if ((*it)->task_type() == TaskNode::TaskType::kInput) {
      int cur_dev_micro_num_limit = stage_micro_num_limit(*it, micro_num_limit);

      if (active_forward_num_ >= cur_dev_micro_num_limit) {
        auto* def_ctx = (*it)->def_ctx();
        if (def_ctx->stage_type_ == HloModule::DefContext::StageType::FORWARD) {
          VLOG(1) << "skip ready node " << (*it)->name() << " for performance";
          continue;
        }
      }

      // AG should be scheduled at last moment.
      auto def_type = (*it)->def_ctx()->def_type();
      if (ready_nodes_.size() > 1 &&
          (def_type == HloModule::DefContext::DefType::AG_SLICE ||
           def_type == HloModule::DefContext::DefType::AG)) {
        VLOG(1) << "skip ready AG node " << (*it)->name() << ", make it tail";
        continue;
      }
    }

    nodes_str += ", " + (*it)->name();
    skip = false;
    if ((*it)->task_type() == TaskNode::TaskType::kSend) {
      auto& children = (*it)->children();  // recv
      CHECK(children.size()==1);
      CHECK(children.front()->task_type() == TaskNode::TaskType::kRecv);

      // NOTE: a tricky: recv node owns the same def context with send node
      auto* recv_nd = children.front();
      auto& recv_children = recv_nd->children();

      // check if peer is ready to receive data
      if (cluster_state.CanRecv(*recv_nd) == false) {
        VLOG(2) << "Cannot receive: " << (*it)->name() << " on " << recv_nd->name();
        skip = true;
      } else {
        int cur_dev_micro_num_limit = stage_micro_num_limit(recv_children.front(), micro_num_limit);
        if (cluster_state.CanSchedule(*recv_children.front(), cur_dev_micro_num_limit) == false) {
          VLOG(2) << "Cannot schedule: " << (*it)->name() << "'s child " << recv_children.front()->name()
                  << " for limit "<< cur_dev_micro_num_limit << " on " << recv_nd->name();
          // TODO(lansong): should check all users of recv
          skip = true;
        }
      }
    }

    if (!skip) {
      if (selected_it == ready_nodes_.end()) {
        selected_it = it;
      } else if (InOrder(*selected_it, *it, cluster_state) == false) {
        selected_it = it;
      }
    } else {
      VLOG(1) << "skip ready node " << (*it)->name() << " because memory limit";
    }
  }

  if (selected_it == ready_nodes_.end()) {
    VLOG(1) << "schedule nothing";
    return nullptr;
  }

  TaskNode* selected_task = (*selected_it);
  CHECK(selected_task);
  ready_nodes_.erase(selected_it);
  cluster_state.SetUpdated();

  VLOG(2) << "ready nodes: " << nodes_str;
  VLOG(1) << "schedule task: " << selected_task->name();

  return selected_task;
}

bool DevState::IncreaseMemUsage(const TaskNode& on_schedule_task) {
  auto task_type = on_schedule_task.task_type();
  switch (task_type) {
  case TaskNode::TaskType::kRecv: {
    auto& recv_children = on_schedule_task.children();  // recv's children
    // NOTE: a tricky: recv node owns the same def context with send node

    // TODO(lansong): should check all users of recv children
    auto* real_recv_ctx = recv_children.front()->def_ctx();

    if (active_ctx_set_.find(real_recv_ctx->def_id()) == active_ctx_set_.end()) {
      // Increase memory for trainable variable size only at the first time.
      int64 trainable_var_bytes = real_recv_ctx->input_var_bytes();
      IncreaseMemBytes(trainable_var_bytes);
    }

    int64 input_activation_bytes = real_recv_ctx->input_activation_bytes();
    IncreaseMemBytes(input_activation_bytes);
    break;
  }
  case TaskNode::TaskType::kInput: {
    // host to device:

    // TODO(lansong):
    // Currently, there is no send/recv to model host to device channel. We
    // have to model the channel on input task node.
    auto& parents = on_schedule_task.parents();
    for (auto* prev_nd : parents) {
      if (prev_nd->task_type() == TaskNode::TaskType::kSplit) {
        HloModule::DefContext* def_ctx = on_schedule_task.def_ctx();
        if (active_ctx_set_.find(def_ctx->def_id()) == active_ctx_set_.end()) {
          // Increase memory for trainable variable size only at the first time.
          int64 trainable_var_bytes = def_ctx->input_var_bytes();
          IncreaseMemBytes(trainable_var_bytes);
        }

        int64 input_activation_bytes = def_ctx->input_activation_bytes();
        IncreaseMemBytes(input_activation_bytes);
      }
    }

    break;
  }
  case TaskNode::TaskType::kOutput: {
    break;
  }
  case TaskNode::TaskType::kGA: {
    break;
  }
  case TaskNode::TaskType::kCompute: {
    Executable* exe = on_schedule_task.exe_or_null();
    CHECK(exe);
    gpu::GpuExecutable* gpu_exe;
    gpu_exe = dynamic_cast<gpu::GpuExecutable*>(exe);
    CHECK(gpu_exe);
    std::shared_ptr<const BufferAssignment> buf_assignment =
                                            gpu_exe->GetBufferAssignment();

    const BufferAssignment::Stats& stats = buf_assignment->GetStats();
    VLOG(2) << "def: " << on_schedule_task.def_ctx()->name() << ", node: " << on_schedule_task.name();
    VLOG(2) << stats.ToString();

    IncreaseMemBytes(stats.preallocated_temp_allocation_bytes);

    HloModule::DefContext* def_ctx = on_schedule_task.def_ctx();
    IncreaseMemBytes(def_ctx->output_tensor_bytes_wo_alias());

    //VLOG(2) << stats.ToString() << std::endl;
    //VLOG(2) << stats.total_allocation_bytes << std::endl;
    break;
  }
  default:
    break;
  }

  if (mem_bytes_ > mem_size_limit_) {
    VLOG(0) << "Out of memory under this schedule strategy->"
            << "mem_bytes=" << mem_bytes_
            << " mem_size_limit=" << mem_size_limit_;
    return false;
  } else {
    return true;
  }
}

float DevState::EstimateTime(const TaskNode* task) {
  float runtime_ms = 0.0;  // time unit: ms

  // currently fake GPU's computing power
  // Assume V100 computing power: 15 TFLOP/s, i.e. 0.015 TFLOP/ms.
  float gpu_power_ms = 0.015 * 1024.0 * 1024.0 * 1024.0 * 1024.0;

  // pcie gen3(x16) bandwidth: 16GB/s, i.e. 0.016 GB/ms.
  // pcie is bottleneck from host memory to gpu memory
  int64 bandwd = ServiceEnv::pp_bandwidth();

  float bw = bandwd * 1024.0 * 1024.0;  // unit: bytes/ms

  if (task->task_type() == TaskNode::TaskType::kSend) {
    // a trick: Send cost is equal to recv cost
    task = task->children().front();
  }

  switch (task->task_type()) {
  case TaskNode::TaskType::kRecv: {
    // communication task
    // TODO(lansong):
    // The communication model is inaccurate because two assumptions:
    //    1). we assume pcie is always the channel of communication. It is
    //        not true, especially inter-GPU communication.
    //    2). communication may happen between machines, or between GPUs
    //        inside machine, or between host and device. we need a more
    //        accurate model.
    auto& recv_children = task->children();  // recv's children
    // NOTE: a tricky: recv node owns the same def context with send node

    // TODO(lansong): should check all users of recv children
    auto* real_recv_ctx = recv_children.front()->def_ctx();
    int def_id = real_recv_ctx->def_id();
    if (active_ctx_set_.find(def_id) == active_ctx_set_.end()) {
      // transfer trainable variable only at the first time.
      int64 trainable_var_bytes = real_recv_ctx->input_var_bytes();
      runtime_ms += 0;
      //runtime_ms += trainable_var_bytes / bw;

      active_ctx_set_.insert(def_id);
    }
    int64 input_activation_bytes = real_recv_ctx->input_activation_bytes();
    runtime_ms += 0;
    //runtime_ms += input_activation_bytes / bw;
    break;
  }
  case TaskNode::TaskType::kInput: {
    // host to device:

    // TODO(lansong):
    // Currently, there is no send/recv to model host to device channel. We
    // have to model the channel on input task node.
    auto& parents = task->parents();
    for (auto* prev_nd : parents) {
      if (prev_nd->task_type() == TaskNode::TaskType::kSplit) {
        HloModule::DefContext* def_ctx = task->def_ctx();
        if (active_ctx_set_.find(def_ctx->def_id()) == active_ctx_set_.end()) {
          // Increase memory for trainable variable size only at the first time.
          int64 trainable_var_bytes = def_ctx->input_var_bytes();
          runtime_ms += 0;
          //runtime_ms += trainable_var_bytes / bw;
        }

        int64 input_activation_bytes = def_ctx->input_activation_bytes();
        runtime_ms += 0;
        //runtime_ms += input_activation_bytes / bw;
      }
    }
    break;
  }
  case TaskNode::TaskType::kCompute: {
    auto* def_ctx = task->def_ctx();
    if (def_ctx->stage_type_ == HloModule::DefContext::StageType::BACKWARD) {
      runtime_ms = 2;
    } else {
      runtime_ms = 1;
    }
    //runtime_ms = (float)(def_ctx->gflops_) / gpu_power_ms;
    break;
  }
  case TaskNode::TaskType::kGA: {
    auto* def_ctx = task->def_ctx();
    runtime_ms = 0;
    //runtime_ms = (float)(def_ctx->gflops_) / gpu_power_ms;
    break;
  }
  case TaskNode::TaskType::kGAInit: {
    auto* def_ctx = task->def_ctx();
    runtime_ms = 0;
    //runtime_ms = (float)(def_ctx->gflops_) / gpu_power_ms;
    break;
  }
  case TaskNode::TaskType::kOutput: {
    // almost do nothing, cost is zero
    break;
  }
  default:
    break;
  }

  return runtime_ms;
}

int DevState::ScheduleNextTask(const float start_time,
                                const int micro_num_limit,
                                ClusterState& cluster_state) {
  if (!scheduled_tasks_.empty()) {
    // scheduled_tasks_ may be scheduled a RECV task by another device when
    // a SEND task is scheduled on that device.

    ScheduledTaskInfo front_task_info = scheduled_tasks_.front();
    CHECK(front_task_info.start_time_ <= start_time) << "front task: "
        << front_task_info.task_->name() << ", task start: " << front_task_info.start_time_
        << ", start: " << start_time;

    // TODO(lansong): may schedule one more task even it is not empty
    return ScheduleState::SCHED_NOTHING;
  }

  int sched_stat = ScheduleState::SCHED_NOTHING;
  if (!ready_nodes_.empty()) {
    // 1. select one task from ready node set
    TaskNode* new_sched_task = PopOneTask(micro_num_limit, cluster_state);
    if (new_sched_task) {
      VLOG(1) << "wid: " << wid_ << ", dev id: " << dev_id_
              << ", scheduled task: " << new_sched_task->name();
    }

    if (new_sched_task == nullptr) {
      return ScheduleState::SCHED_NOTHING;
    }

    if (//new_sched_task == nullptr ||    // indirectly OOM: execeed micro num limit
        IncreaseMemUsage(*new_sched_task) == false) {  // directly OOM
      //return ScheduleState::OOM;
    }

    if (new_sched_task->task_type() == TaskNode::TaskType::kSend) {
      // Current task is a send, we will schedule the matched recv next
      // iteration.
      auto& children = new_sched_task->children();  // recv
      CHECK(children.size()==1);
      CHECK(children.front()->task_type() == TaskNode::TaskType::kRecv);

      auto* recv_nd = children.front();
      // immediately schedule recv task to its device
      cluster_state.ScheduleTask(recv_nd, start_time);

      // record communication direction flags for send/recv
      // these flags will guide how to set rank for nccl communication primitive
      if (new_sched_task->worker_id() < recv_nd->worker_id()) {
        new_sched_task->set_comm_with_lower_stage(false);
        recv_nd->set_comm_with_lower_stage(true);
        new_sched_task->set_across_machine(true);
        recv_nd->set_across_machine(true);
      } else if (new_sched_task->worker_id() > recv_nd->worker_id()) {
        new_sched_task->set_comm_with_lower_stage(true);
        recv_nd->set_comm_with_lower_stage(false);
        new_sched_task->set_across_machine(true);
        recv_nd->set_across_machine(true);
      } else {
        new_sched_task->set_across_machine(false);
        recv_nd->set_across_machine(false);
        if (new_sched_task->device_id() < recv_nd->device_id()) {
          new_sched_task->set_comm_with_lower_stage(false);
          recv_nd->set_comm_with_lower_stage(true);
        } else if (new_sched_task->device_id() > recv_nd->device_id()) {
          new_sched_task->set_comm_with_lower_stage(true);
          recv_nd->set_comm_with_lower_stage(false);
        } else {
          CHECK(0);
        }
      }
    }

    auto* def_ctx = new_sched_task->def_ctx();
    if (new_sched_task->task_type() == TaskNode::TaskType::kInput) {
      if (def_ctx->stage_type_ == HloModule::DefContext::StageType::FORWARD) {
        ++active_forward_num_;
        VLOG(1) << "scheduling forward input, increase active_forward_num_ to " << active_forward_num_;
      }
    }

    // update next finish time
    float runtime_ms = EstimateTime(new_sched_task);
    float finish_time = start_time + runtime_ms;  // accumulate the estimation time
    VLOG(2) << "wid: " << wid_ << ", dev id: " << dev_id_
            << ", task: " << new_sched_task->name()
            << ", start: " << start_time << ", runtime_ms: " << runtime_ms
            << ", finish_time: " << finish_time;

    VLOG(2) << "new scheduled task " << new_sched_task->name();
    scheduled_tasks_.push_back(ScheduledTaskInfo(new_sched_task, start_time, finish_time));
    cluster_state.SetUpdated();

    sched_stat = ScheduleState::SCHED_SUCC;
  }

  return sched_stat;
}

void DevState::MarkTaskDone(const ScheduledTaskInfo& task_info) {
  // only call it for source node
  finished_set_.insert(task_info.task_);
  finished_sequence_.emplace_back(task_info);
}

void DevState::MarkTaskDone(const ScheduledTaskInfo& task_info,
                            ClusterState& cluster_state,
                            OutputBuffersLifeTimeTracker& lifetime_tracker) {
  TaskNode* task = task_info.task_;
  VLOG(2) << "wid: " << wid_ << ", dev id: " << dev_id_ << ", mark task "
          << task->name() << " done";
  finished_set_.insert(task);
  finished_sequence_.emplace_back(task_info);
  cluster_state.SetUpdated();

  const TaskDAG* task_graph = cluster_state.task_graph();

  // decrease memory user's count
  // refer to LocalPlan::MakeTaskGraphGCPlan
  auto task_type = task->task_type();
  switch (task_type) {
  case TaskNode::TaskType::kOutput: {
    // find the corresponding input node
    auto* input_task = task->parent()->parent();
    std::vector<std::pair<TaskNode*, int/*output_idx*/>> parent_to_release
          = lifetime_tracker.CollectReleasableTensorsForParent(task, input_task);
    cluster_state.DecreaseMemUsage(parent_to_release);

    auto* def_ctx = task->def_ctx();
    if (!def_ctx->ag_def_ctx() && !def_ctx->ag_slice_def_ctx()) {
      std::vector<std::pair<TaskNode*, int/*output_idx*/>> self_to_release
            = lifetime_tracker.CollectReleasableTensorsForSelf(task, task_graph);
      cluster_state.DecreaseMemUsage(self_to_release);
    }

    Executable* exe = task->exe_or_null();
    CHECK(exe);
    gpu::GpuExecutable* gpu_exe;
    gpu_exe = dynamic_cast<gpu::GpuExecutable*>(exe);
    CHECK(gpu_exe);
    std::shared_ptr<const BufferAssignment> buf_assignment =
                                            gpu_exe->GetBufferAssignment();

    const BufferAssignment::Stats& stats = buf_assignment->GetStats();
    VLOG(2) << "def: " << def_ctx->name() << ", node: " << task->name();
    VLOG(2) << stats.ToString();

    int64 temp_types = stats.preallocated_temp_allocation_bytes;
    VLOG(1) << "decrease preallocated_temp_allocation_bytes: " << temp_types;
    VLOG(1) << "decrease tensor for: " << task->name() << ", temp memo";
    DecreaseMemBytes(temp_types);

    // decrease active_forward_num_ if this output owned by a backward stage.
    if (def_ctx->stage_type_ == HloModule::DefContext::StageType::BACKWARD) {
      --active_forward_num_;
      VLOG(1) << "scheduling backward output, decrease active_forward_num_ to " << active_forward_num_;
    }
    break;
  }

  case TaskNode::TaskType::kGA: {
    std::vector<std::pair<TaskNode*, int/*output_idx*/>> parent_to_release
                = lifetime_tracker.CollectReleasableTensorsForParent(task, task);
    cluster_state.DecreaseMemUsage(parent_to_release);
    break;
  }

  default:
    break;
  }

  VLOG(1) << "task: " << task->name();
}

bool DevState::IsFinished(const TaskNode* task) const {
  return finished_set_.find(task) != finished_set_.end();
}

std::vector<TaskNode*> DevState::MarkTaskDoneByTime(float finish_time,
                              ClusterState& cluster_state,
                              OutputBuffersLifeTimeTracker& lifetime_tracker) {
  std::vector<TaskNode*> new_ready_tasks;
  if (scheduled_tasks_.empty()) {
    return new_ready_tasks;
  } else {
    ScheduledTaskInfo first_task_info = scheduled_tasks_.front();
    if (first_task_info.end_time_ > finish_time) {
      VLOG(2) << "don't mark " << first_task_info.task_->name()
              << " done because first task's end_time_ > finish_time";
      return new_ready_tasks;
    }
  }

  ScheduledTaskInfo first_task_info = scheduled_tasks_.front();
  TaskNode* first_task = first_task_info.task_;
  first_task->set_sched_idx_in_dev(finished_sequence_.size());
  MarkTaskDone(first_task_info, cluster_state, lifetime_tracker);

  VLOG(2) << "first task: " << first_task->name();
  for (auto* child : first_task->children()) {
    VLOG(2) << "child: " << child->name();
    if (should_ignore_by_sched_id(child, sched_id_)) {
      continue;
    }
    if (cluster_state.ParentsDone(child)) {
      VLOG(2) << "child: " << child->name() << " is ready at " << finish_time;
      new_ready_tasks.emplace_back(child);
    }
  }

  scheduled_tasks_.pop_front();
  cluster_state.SetUpdated();

  return std::move(new_ready_tasks);
}

void DevState::DecreaseMemUsage(const TaskNode* task, int output_idx) {
  HloModule::DefContext* def_ctx = task->def_ctx();
  CHECK(def_ctx->output_tensor_size_map_.find(output_idx) !=
        def_ctx->output_tensor_size_map_.end());

  VLOG(1) << "decrease tensor for: " << task->name() << ", out idx: " << output_idx;
  DecreaseMemBytes(def_ctx->output_tensor_size_map_[output_idx]);
}

bool DevState::TaskScheduled(TaskNode& task) const {
  CHECK(task.worker_id() == wid_);
  CHECK(task.device_id() == dev_id_);
  bool ready = false;
  if (finished_set_.find(&task) != finished_set_.end()) {
    ready = true;
  } else {
    for (auto& sched_info : scheduled_tasks_) {
      if (&task == sched_info.task_) {
        ready = true;
        break;
      }
    }
  }

  return ready;
}

bool DevState::CanRecv(const ClusterState& cluster_state, TaskNode& recv) const {
  std::set<TaskNode*> brothers;
  auto* child = recv.children().front();
  for (auto* brother : child->parents()) {
    if (brother != &recv &&
        brother->task_type() != TaskNode::TaskType::kSplit) {
      if (!should_ignore_by_sched_id(brother, sched_id_)) {
        brothers.insert(brother);
      }
    }
  }

  for (auto* brother : brothers) {
    if (cluster_state.TaskScheduled(*brother) == false) {
      VLOG(1) << "wid: " << brother->worker_id() << ", dev id: " << brother->device_id()
              << ", task " << brother->name() << " is NOT scheduled yet";

      return false;
    } else {
      VLOG(1) << "wid: " << brother->worker_id() << ", dev id: " << brother->device_id()
              << ", task " << brother->name() << " is scheduled or finished";
    }
  }

  if (!scheduled_tasks_.empty()) {
    return false;
  }

  if (!finished_sequence_.empty()) {
    ScheduledTaskInfo last_task_info = finished_sequence_[finished_sequence_.size()-1];
    CHECK(last_task_info.task_);
    switch (last_task_info.task_->task_type()) {
    case TaskNode::TaskType::kRecv:
    case TaskNode::TaskType::kInput:
    case TaskNode::TaskType::kCompute:
      return false;
    default:
      break;
    }
  }

  return true;
}

bool DevState::CanSchedule(const TaskNode& task, const int micro_num_limit) const {
  if (active_forward_num_ < micro_num_limit) {
    return true;
  }

  HloModule::DefContext* def_ctx = task.def_ctx();
  switch (def_ctx->stage_type_) {
  case HloModule::DefContext::StageType::FORWARD:
    return false;
  case HloModule::DefContext::StageType::NA:
  case HloModule::DefContext::StageType::BACKWARD:
  case HloModule::DefContext::StageType::BOTH:
    return true;
  default:
    return true;
  }
}

void DevState::ExportSchedule(std::vector<TaskNode*>& task_sequence) {
  for (auto& task_info : finished_sequence_) {
    task_sequence.emplace_back(task_info.task_);
  }
}

void DevState::PrintState() {
  VLOG(0) << "  dev id: " << dev_id_;
  VLOG(0) << "    tasks in scheduled_tasks:";
  if (!scheduled_tasks_.empty()) {
    for (auto& task_info : scheduled_tasks_) {
      VLOG(0) << "      start time: " << task_info.start_time_
              << ", end time: " << task_info.end_time_
              << ", task: " << task_info.task_->name()
              << ", type: " << task_info.task_->task_type_string();
    }
  } else {
    VLOG(0) << "      empty";
  }

  if (ready_nodes_.empty()) {
    VLOG(0) << "    ready node set is empty";
  } else {
    VLOG(0) << "    ready node set:";
    for (auto* task : ready_nodes_) {
      VLOG(0) << "      " << task->name();
    }
  }

  VLOG(0) << "    finished sequence:";
  if (finished_sequence_.empty()) {
    VLOG(0) << "      empty";
  } else {
    for (auto& task_info : finished_sequence_) {
      VLOG(0) << "      start: " << task_info.start_time_
              << ", end: " << task_info.end_time_
              << ", " << task_info.task_->name();
    }
  }

  CHECK(finished_set_.size() == finished_sequence_.size());
}

MachineState::MachineState(int wid, int dev_count, const ScheduleIdInfo& sched_id)
: wid_(wid)
, sched_id_(sched_id) {
  for (int dev_id=0; dev_id<dev_count; ++dev_id) {
    dev_states_.push_back(absl::make_unique<DevState>(wid, dev_id, sched_id));
  }

  host_state_ = absl::make_unique<DevState>(wid, -1, sched_id);
}

void MachineState::AddReadyNode(TaskNode* task) {
  int device_id = task->device_id();

  //VLOG(2) << "[AddReadyNode] task: " << task->name() << ", dev id: " << device_id
  //        << ", dev_states_.size: " << dev_states_.size();
  CHECK(device_id<(int)dev_states_.size());
  if (device_id>=0) {
    dev_states_[device_id]->AddReadyNode(task);
  } else {
    // NOTE: entry send's device id is -1.
    host_state_->AddReadyNode(task);
  }
}

void MachineState::ScheduleTask(TaskNode* task, float start_time) {
  int device_id = task->device_id();

  CHECK(device_id<(int)dev_states_.size());
  if (device_id>=0) {
    dev_states_[device_id]->ScheduleTask(task, start_time);
  } else {
    // NOTE: entry send's device id is -1.
    host_state_->ScheduleTask(task, start_time);
  }
}

int MachineState::ScheduleNextTask(const float start_time,
                                   const int micro_num_limit,
                                   ClusterState& cluster_state) {
  int sched_stat = ScheduleState::SCHED_NOTHING;

  sched_stat |= host_state_->ScheduleNextTask(start_time, micro_num_limit,
                                             cluster_state);
  if (sched_stat & ScheduleState::OOM) {
    // only schedule one task at each iteration in order to ensure send/recv
    // will be scheduled together.
    return sched_stat;
  }

  for (auto& dev_stat : dev_states_) {
    sched_stat |= dev_stat->ScheduleNextTask(start_time, micro_num_limit,
                                            cluster_state);
    if (sched_stat & ScheduleState::OOM) {
      return sched_stat;
    }
  }

  return sched_stat;
}

std::vector<TaskNode*> MachineState::MarkTaskDoneByTime(float finish_time,
                              ClusterState& cluster_state,
                              OutputBuffersLifeTimeTracker& lifetime_tracker) {
  std::vector<TaskNode*> new_ready_tasks;
  /*
  VLOG(2) << "host" << std::endl << "  next finish time: " << host_state_->next_finish_time()
            << ", finish time: " << finish_time;
  */
  auto dev_new_ready_tasks = host_state_->MarkTaskDoneByTime(finish_time,
                                                             cluster_state,
                                                             lifetime_tracker);
  new_ready_tasks.insert(new_ready_tasks.end(), dev_new_ready_tasks.begin(),
                         dev_new_ready_tasks.end());
  for (auto& dev_stat : dev_states_) {
    /*
    VLOG(2) << "wid: " << dev_stat->wid() << ", dev_stat: " << dev_stat->dev_id();
    VLOG(2) << "  dev_stat->next_finish_time: " << dev_stat->next_finish_time() << ", finish time: " << finish_time;
    */
    auto dev_new_ready_tasks = dev_stat->MarkTaskDoneByTime(finish_time,
                                                            cluster_state,
                                                            lifetime_tracker);
    new_ready_tasks.insert(new_ready_tasks.end(), dev_new_ready_tasks.begin(),
                           dev_new_ready_tasks.end());
  }

  return std::move(new_ready_tasks);
}

void MachineState::ExportSchedule(ScheduleInMachine& sched_in_mach) {
  // NOTE: currently we don't order host tasks because there is at most 1 host task in graph.
  // only entry send may be placed on host.
  for (auto& dev_stat : dev_states_) {
    CHECK(dev_stat->dev_id() < sched_in_mach.per_device_tasks_.size());
    std::vector<TaskNode*>& task_sequence = sched_in_mach.per_device_tasks_[dev_stat->dev_id()];
    dev_stat->ExportSchedule(task_sequence);
  }
}

void MachineState::MarkTaskDone(const ScheduledTaskInfo& task_info) {
  // only call it for source node
  auto device_id = task_info.task_->device_id();

  CHECK(device_id<(int)dev_states_.size());
  if (device_id>=0) {
    dev_states_[device_id]->MarkTaskDone(task_info);
  } else {
    // NOTE: device id of entry send is -1.
    host_state_->MarkTaskDone(task_info);
  }
}

void MachineState::DecreaseMemUsage(const TaskNode* task, int output_idx) {
  auto device_id = task->device_id();

  CHECK(device_id<(int)dev_states_.size());
  if (device_id>=0) {
    dev_states_[device_id]->DecreaseMemUsage(task, output_idx);
  } else {
    // NOTE: device id of entry send is -1.
    host_state_->DecreaseMemUsage(task, output_idx);
  }
}

bool MachineState::TaskScheduled(TaskNode& task) const {
  auto device_id = task.device_id();
  if (device_id>=0) {
    return dev_states_[device_id]->TaskScheduled(task);
  } else {
    return host_state_->TaskScheduled(task);
  }
}

bool MachineState::CanRecv(const ClusterState& cluster_state, TaskNode& recv) const {
  auto device_id = recv.device_id();
  if (device_id>=0) {
    return dev_states_[device_id]->CanRecv(cluster_state, recv);
  }

  return true;
}

bool MachineState::CanSchedule(const TaskNode& task, const int micro_num_limit) const {
  auto device_id = task.device_id();

  CHECK(device_id<(int)dev_states_.size());
  if (device_id>=0) {
    return dev_states_[device_id]->CanSchedule(task, micro_num_limit);
  } else {
    // NOTE: device_id of entry send and entry recv are all -1.
    return host_state_->CanSchedule(task, micro_num_limit);
  }
}

void MachineState::PrintState() {
  VLOG(0) << "worker id: " << wid_;
  host_state_->PrintState();
  for (auto& dev_stat : dev_states_) {
    dev_stat->PrintState();
    VLOG(0) << std::endl;
  }
}

void ClusterState::Init(const std::vector<int>& worker_dev_cnt) {
  for (int wid=0; wid<worker_dev_cnt.size(); ++wid) {
    mach_states_.push_back(absl::make_unique<MachineState>(wid, worker_dev_cnt[wid], sched_id_));
  }


  // set rank of task
  // rank is a factor of schedule priority
  std::deque<const TaskNode*> ready_tasks;
  std::unordered_set<const TaskNode*> visited;
  auto* source = task_graph_->source();
  ready_tasks.push_back(source);
  visited.insert(source);

  auto is_ready = [this] (const TaskNode* task) -> bool {
    for (auto* parent : task->parents()) {
      if (task_rank_map_.find(parent) == task_rank_map_.end()) {
        return false;
      }
    }

    return true;
  };

  while (!ready_tasks.empty()) {
    auto* task = ready_tasks.front();
    ready_tasks.pop_front();
    int rank = 0;
    for (auto* parent : task->parents()) {
      CHECK(task_rank_map_.find(parent) != task_rank_map_.end());
      if (rank<=task_rank_map_[parent]) {
        rank = task_rank_map_[parent] + 1;
      }
    }
    task_rank_map_[task] = rank;
    VLOG(1) << "rank of task " << task->name() << ": " << rank;

    for (auto* child : task->children()) {
      if (is_ready(child)) {
        CHECK(visited.find(child) == visited.end());
        ready_tasks.emplace_back(child);
        visited.insert(child);
      }
    }
  }
}

void ClusterState::AddReadyNode(TaskNode* task) {
  auto wid = task->worker_id();

  CHECK(wid<mach_states_.size()) << wid << " for task " << task->name();
  CHECK(wid>=0) << wid << " for task " << task->name();;
  mach_states_[wid]->AddReadyNode(task);
}

float ClusterState::FindNextFinishTime() {
  float next_time = FLT_MAX;
  for (auto& mach_stat : mach_states_) {
    float mach_next_time = mach_stat->FindNextFinishTime();
    if (mach_next_time < next_time) {
      next_time = mach_next_time;
    }
  }

  return next_time;
}

void ClusterState::SetGlobalNextFinishTime(float global_next_finish_time) {
  for (auto& mach_stat : mach_states_) {
    mach_stat->SetGlobalNextFinishTime(global_next_finish_time);
  }
}

void ClusterState::ScheduleTask(TaskNode* task, float start_time) {
  auto wid = task->worker_id();

  CHECK(wid<mach_states_.size()) << wid << " for task " << task->name();
  CHECK(wid>=0) << wid << " for task " << task->name();;
  mach_states_[wid]->ScheduleTask(task, start_time);
  SetUpdated();
}

int ClusterState::ScheduleNextTask(const float start_time,
                                             const int micro_num_limit) {
  int sched_stat = ScheduleState::SCHED_NOTHING;
  for (auto& mach_stat : mach_states_) {
    sched_stat |= mach_stat->ScheduleNextTask(start_time, micro_num_limit, *this);
    if (sched_stat & ScheduleState::OOM) {
      return sched_stat;
    }
  }

  return sched_stat;
}

std::vector<TaskNode*> ClusterState::MarkTaskDoneByTime(float finish_time,
                              OutputBuffersLifeTimeTracker& lifetime_tracker) {
  std::vector<TaskNode*> new_ready_tasks;
  for (auto& mach_stat : mach_states_) {
    auto mach_new_ready = mach_stat->MarkTaskDoneByTime(finish_time, *this,
                                                        lifetime_tracker);
    new_ready_tasks.insert(new_ready_tasks.end(), mach_new_ready.begin(),
                           mach_new_ready.end());
  }

  return new_ready_tasks;
}

void ClusterState::RecordReadyNodes(
                  const std::vector<TaskNode*>& new_ready_tasks) {
  if (!new_ready_tasks.empty()) {
    for (auto* task : new_ready_tasks) {
      CHECK(task->worker_id() >= 0);    // for any node other than merge, its wid should not less than 0
      if (task->task_type() != TaskNode::TaskType::kRecv) {
        AddReadyNode(task);
      }
    }
    SetUpdated();
  }
}

bool ClusterState::ParentsDone(const TaskNode* task) const {
  for (auto* parent : task->parents()) {
    auto wid = parent->worker_id();
    auto dev_id = parent->device_id();
    auto* dev_state = GetDevState(wid, dev_id);
    if (should_ignore_by_sched_id(parent, sched_id_)) {
      continue;
    }
    if (!dev_state->IsFinished(parent)) {
      return false;
    }
  }

  return true;
}

void ClusterState::ExportSchedule(
            std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched) {
  for (auto& mach_stat : mach_states_) {
    CHECK(mach_stat->wid() < cluster_sched.size()) << "wid: " << mach_stat->wid()
          << ", cluster_sched.size: " << cluster_sched.size();
    ScheduleInMachine* sched_in_mach = cluster_sched[mach_stat->wid()].get();
    mach_stat->ExportSchedule(*sched_in_mach);
  }
}

void ClusterState::MarkTaskDone(const ScheduledTaskInfo& task_info) {
  // only call it for source node
  auto wid = task_info.task_->worker_id();

  CHECK(wid<mach_states_.size());
  CHECK(wid>=0);
  mach_states_[wid]->MarkTaskDone(task_info);
  SetUpdated();
}

void ClusterState::PrintState() {
  for (auto& mach_stat : mach_states_) {
    mach_stat->PrintState();
    VLOG(0) << std::endl;
  }
}

void ClusterState::DecreaseMemUsage(
  const std::vector<std::pair<TaskNode*, int/*output_idx*/>>& tensors_to_release) {
  for (auto& it : tensors_to_release) {
    auto wid = it.first->worker_id();

    CHECK(wid<mach_states_.size());
    CHECK(wid>=0);
    mach_states_[wid]->DecreaseMemUsage(it.first, it.second);
  }
}

bool ClusterState::TaskScheduled(TaskNode& task) const {
  auto wid = task.worker_id();

  CHECK(wid<mach_states_.size());
  CHECK(wid>=0);
  return mach_states_[wid]->TaskScheduled(task);
}

bool ClusterState::CanRecv(TaskNode& recv) const {
  auto wid = recv.worker_id();

  CHECK(wid<mach_states_.size());
  CHECK(wid>=0);
  return mach_states_[wid]->CanRecv(*this, recv);
}

bool ClusterState::CanSchedule(const TaskNode& task, const int micro_num_limit) const {
  auto wid = task.worker_id();

  CHECK(wid<mach_states_.size());
  CHECK(wid>=0);
  return mach_states_[wid]->CanSchedule(task, micro_num_limit);
}

TaskScheduler::TaskScheduler(TaskDAG* dag, int worker_count,
                             const std::vector<int>& worker_dev_cnt,
                             int sched_cnt)
: task_graph_(dag)
, worker_count_(worker_count)
, worker_dev_cnt_(worker_dev_cnt) {
  for (int i = 0; i < sched_cnt; ++i) {
    cluster_states_.push_back(new ClusterState(dag, i, sched_cnt));
    cluster_states_[i]->Init(worker_dev_cnt);
  }
  CHECK(cluster_states_.size() > 0);
}

// TODO(zycao): refactor the Reorder functions with template.
bool TaskScheduler::ReorderSend(
    std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched) {
  bool reordered = false;
  for (int wid = 0; wid < cluster_sched.size(); ++wid) {
    std::unique_ptr<ScheduleInMachine>& machine_sched = cluster_sched[wid];
    for (std::vector<TaskNode*>& dev_tasks : machine_sched->per_device_tasks_) {
      for (int i = 0; i < dev_tasks.size(); ++i) {
        if (dev_tasks[i]->task_type() != TaskNode::TaskType::kSend) {
          continue;
        }

        TaskNode* send = dev_tasks[i];
        TaskNode* send_parent = send->parents().front();
        for (int j = i - 1; j >= 0; --j) {
          // Send reorder:
          // (1) keep all Sends original order due to stream executing mechanism,
          // (2) keep the order of communicator (NCCL) using for correct logic.
          //     that means Recvs can change order with Sends in same stage.
          // (3) to preserve definite order to avoid cyclic replacing, Sends
          //     could only change order with Recvs with larger micro_ids.
          if (dev_tasks[j]->task_type() != TaskNode::TaskType::kSend &&
              (dev_tasks[j]->task_type() != TaskNode::TaskType::kRecv ||
               (send->def_ctx()->stage_type_ == dev_tasks[j]->def_ctx()->stage_type_ &&
                send->micro_id() < dev_tasks[j]->micro_id())) &&
              dev_tasks[j] != send_parent) {
            int send_sched_idx = send->sched_idx_in_dev();
            send->set_sched_idx_in_dev(dev_tasks[j]->sched_idx_in_dev());
            dev_tasks[j]->set_sched_idx_in_dev(send_sched_idx);
            dev_tasks[j+1] = dev_tasks[j];
            dev_tasks[j] = send;
            reordered = true;
          } else {
            break;
          }
        }
      }
    }
  }
  if (reordered) VLOG(1) << "Send reordered.";
  return reordered;
}

bool TaskScheduler::ReorderGA(
    std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched) {
  bool early_ga = ServiceEnv::early_ga();
  if (!early_ga) return false;

  bool reordered = false;
  for (int wid = 0; wid < cluster_sched.size(); ++wid) {
    std::unique_ptr<ScheduleInMachine>& machine_sched = cluster_sched[wid];
    for (std::vector<TaskNode*>& dev_tasks : machine_sched->per_device_tasks_) {
      for (int i = 0; i < dev_tasks.size(); ++i) {
        if (dev_tasks[i]->task_type() != TaskNode::TaskType::kGA) {
          continue;
        }

        TaskNode* ga = dev_tasks[i];
        TaskNode* ga_parent1 = ga->parents().front();
        TaskNode* ga_parent2 = ga->parents().back();
        for (int j = i - 1; j >= 0; --j) {
          // GA could be moved closed to its micro batch OUTPUT. However, this
          // might not influence the max memory usage although it would release
          // memory earlier.
          if ((ga->micro_id() < dev_tasks[j]->micro_id() ||
              ga->def_ctx()->stage_type_ != dev_tasks[j]->def_ctx()->stage_type_) &&
              dev_tasks[j] != ga_parent1 &&
              dev_tasks[j] != ga_parent2) {
            int ga_sched_idx = ga->sched_idx_in_dev();
            ga->set_sched_idx_in_dev(dev_tasks[j]->sched_idx_in_dev());
            dev_tasks[j]->set_sched_idx_in_dev(ga_sched_idx);
            dev_tasks[j+1] = dev_tasks[j];
            dev_tasks[j] = ga;
            reordered = true;
          } else {
            break;
          }
        }
      }
    }
  }
  if (reordered) VLOG(1) << "GA reordered.";
  return reordered;
}

bool TaskScheduler::Schedule(
              const int micro_num_limit,
              std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched) {

  std::vector<std::vector<std::unique_ptr<ScheduleInMachine>>> sub_schedules(cluster_states_.size());
  for (int i = 0; i < cluster_states_.size(); ++i) {
    for (int wid=0; wid<worker_count_; ++wid) {
      sub_schedules[i].push_back(absl::make_unique<ScheduleInMachine>(worker_dev_cnt_[wid]));
    }
    VLOG(1) << "schedule micro task group " << i;
    bool status = ScheduleImpl(micro_num_limit, *cluster_states_[i], sub_schedules[i]);
    if (!status) {
      return false;
    }
  }
  MergeClusterSchedule(sub_schedules, cluster_sched);

  // Moving Send as early as possible for running through the data pipe.
  // Then GA should be done when each micro batch finished its CG task.
  // At last start Receive earlier for possible more overlapping.
  // 
  // NOTE: Scheduling could be reordred multiple times, since later passes may
  // bring further reorder space for earlier passes.
  bool multi_reorder = ServiceEnv::multi_reorder();

  bool reordered = true;
  int i = 0;
  do {
    reordered = ReorderSend(cluster_sched);
    reordered = ReorderGA(cluster_sched) || reordered;
    if (++i > 1000) { /*times limit to break cyclic*/
      VLOG(1) << "Reorder might fall into cyclic running, break it.";
      break;
    }
  } while (reordered && multi_reorder);

  return true;
}

bool TaskScheduler::ScheduleImpl(
              const int micro_num_limit,
              ClusterState& cluster_state,
              std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched) {
  OutputBuffersLifeTimeTracker lifetime_tracker;
  lifetime_tracker.Initialize(task_graph_);
  lifetime_tracker.ResetBufferRefCounts();
  auto* source = task_graph_->source();
  for (auto* child : source->children()) {
    if (child->parents().size() == 1) {
      if (!should_ignore_by_sched_id(child, cluster_state.sched_id())) {
        // select the first group of ready nodes
        // at the beginning, a node is ready when source is the only parent
        cluster_state.AddReadyNode(child);
      }
    }
  }

  ScheduledTaskInfo src_task_info(source, 0, 0);
  cluster_state.MarkTaskDone(src_task_info);

  bool debug = false;

  if (debug) {
    // debug:
    VLOG(0) << "Initially state:";
    cluster_state.PrintState();
  }

  int sched_stat = ScheduleState::SCHED_NOTHING;
  float start_time = 0.0;
  do {
    cluster_state.ResetUpdated();
    // 1. select a task and schedule it
    if (debug) {
      VLOG(0) << "start_time just before schedule next task: " << start_time << std::endl;
    }
    sched_stat = cluster_state.ScheduleNextTask(start_time, micro_num_limit);

    // debug:
    if (debug) {
      //VLOG(0) << "cluster state just after schedule:";
      //cluster_state_.PrintState();
    }

    if (sched_stat & ScheduleState::OOM) {
      return false;
    }

    // 2. find earliest finish time from running tasks
    start_time = cluster_state.FindNextFinishTime();
    if (debug) {
      VLOG(0) << "global next finish time: " << start_time << std::endl;
    }

    /*
    CHECK(start_time != FLT_MAX);
    cluster_state_.SetGlobalNextFinishTime(start_time);
    */

    if (start_time != FLT_MAX) {
      // 3. move current running task to finish task set and determine next ready tasks
      std::vector<TaskNode*> new_ready_tasks = cluster_state.MarkTaskDoneByTime(
                                                  start_time, lifetime_tracker);

      if (debug) {
        VLOG(0) << "new ready tasks size: " << new_ready_tasks.size();
        for (auto* task : new_ready_tasks) {
          VLOG(0) << "task: " << task->name() << ", type: "
                    << task->task_type_string() << ", worker id: "
                    << task->worker_id() << ", dev id: " << task->device_id();
        }
      }

      // 4. record next group of ready tasks
      cluster_state.RecordReadyNodes(new_ready_tasks);
    } else {
      // all scheduled_tasks_ of devices are empty
      // use max end time of finished_sequence_ to prepare start time for next iteration
      start_time = cluster_state.last_finish_time();
      CHECK(start_time != FLT_MAX);
    }

    /*
    float global_next_finish_time = cluster_state_.FindNextFinishTime();
    if (global_next_finish_time != FLT_MAX) {
      cluster_state_.SetGlobalNextFinishTime(global_next_finish_time);
    }
    */

    // debug:
    if (debug) {
      VLOG(0) << "cluster state:";
      cluster_state.PrintState();
    }
  } while ((!cluster_state.AllFinished()) && cluster_state.updated());

  if (!cluster_state.AllFinished()) {
    CHECK(0) << "failed to schedule task graph!";
  }

  if (ServiceEnv::debug()) {
    VLOG(0) << "final cluster state:";
    cluster_state.PrintState();
  }

  // export schedule result to execution plan
  cluster_state.ExportSchedule(cluster_sched);

  return true;
}

}  // namespace xla

