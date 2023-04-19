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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TASK_SCHEDULER_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TASK_SCHEDULER_H_

#include <memory>
#include <vector>
#include <unordered_set>
#include <float.h>
#include <deque>

#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/pjrt/lifetime_tracker.h"

namespace xla {

class TaskDAG;
class TaskNode;
class ClusterState;

enum ScheduleState {
  SCHED_NOTHING = 1,
  SCHED_SUCC = 2,
  OOM = 4,
};

struct ScheduleIdInfo {
  explicit ScheduleIdInfo(const int id, const int sched_size)
    : id_(id), sched_size_(sched_size) {}

  explicit ScheduleIdInfo(const ScheduleIdInfo& sched_id)
    : id_(sched_id.id_), sched_size_(sched_id.sched_size_) {}

  std::string ToString() const {
    return "sched_id: " + std::to_string(id_) + "/" + std::to_string(sched_size_);
  }

  int id_;
  int sched_size_;
};


struct ScheduleInMachine {
  ScheduleInMachine(int dev_num)
  : per_device_tasks_(dev_num) {
  }

  std::string ToString() const;

  std::vector<std::vector<TaskNode*>> per_device_tasks_;
};

struct ScheduledTaskInfo {
  ScheduledTaskInfo()
  : task_(nullptr)
  , start_time_(0)
  , end_time_(0) {
  }

  ScheduledTaskInfo(TaskNode* task, float start_time, float end_time)
  : task_(task)
  , start_time_(start_time)
  , end_time_(end_time) {
  }

  TaskNode* task_;
  float start_time_;
  float end_time_;
};

// represents GPU scheduling state
class DevState {
 public:
  DevState(int wid, int dev_id, const ScheduleIdInfo& sched_id)
  : wid_(wid)
  , dev_id_(dev_id)
  , sched_id_(sched_id) {
  }
  ~DevState() {}
  TaskNode* PopOneTask(const int micro_num_limit,
                       ClusterState& cluster_state);
  void AddReadyNode(TaskNode* task);
  bool IsFinished(const TaskNode* task) const;
  float next_finish_time() const {
    if (scheduled_tasks_.empty()) {
      return FLT_MAX;
    }

    ScheduledTaskInfo first_task_info = scheduled_tasks_.front();
    return first_task_info.end_time_;
  }
  float last_finish_time() const {
    if (!scheduled_tasks_.empty()) {
      return scheduled_tasks_.back().end_time_;
    }

    if (!finished_sequence_.empty()) {
      return finished_sequence_.back().end_time_;
    }

    return 0;
  }
  int ScheduleNextTask(const float start_time,
                       const int micro_num_limit,
                       ClusterState& cluster_state);
  void ScheduleTask(TaskNode* task, float start_time);
  std::vector<TaskNode*> MarkTaskDoneByTime(float finish_time,
                              ClusterState& cluster_state,
                              OutputBuffersLifeTimeTracker& lifetime_tracker);
  void ExportSchedule(std::vector<TaskNode*>& task_sequence);
  void MarkTaskDone(const ScheduledTaskInfo& task_info);
  void MarkTaskDone(const ScheduledTaskInfo& task_info,
                    ClusterState& cluster_state,
                    OutputBuffersLifeTimeTracker& lifetime_tracker);
  bool IncreaseMemUsage(const TaskNode& on_schedule_task);
  void DecreaseMemUsage(const TaskNode* task, int output_idx);
  bool TaskScheduled(TaskNode& task) const;
  bool CanRecv(const ClusterState& cluster_state, TaskNode& recv) const;
  bool CanSchedule(const TaskNode& task, const int micro_num_limit) const;
  void PrintState();
  int wid() const { return wid_; }
  int dev_id() const { return dev_id_; }
  void IncreaseMemBytes(int64 incr_bytes) {
    //VLOG(0) << "[before IncreaseMemBytes] mem_bytes_: " << mem_bytes_ << ", incr_bytes: " << incr_bytes;
    CHECK(incr_bytes >= 0);
    mem_bytes_ += incr_bytes;
    //VLOG(0) << "[after IncreaseMemBytes] mem_bytes_: " << mem_bytes_;
  }
  void DecreaseMemBytes(int64 decr_bytes) {
    CHECK(decr_bytes >= 0);
    //VLOG(0) << "[before DecreaseMemBytes] mem_bytes_: " << mem_bytes_ << ", decr_bytes: " << decr_bytes;
    mem_bytes_ -= decr_bytes;
    //VLOG(0) << "[after DecreaseMemBytes] mem_bytes_: " << mem_bytes_;
    CHECK(mem_bytes_ >= 0);
  }
  bool AllFinished() const {
    if (ready_nodes_.empty() && scheduled_tasks_.empty()) {
      return true;
    } else {
      return false;
    }
  }
  void SetGlobalNextFinishTime(float global_next_finish_time) {
    global_next_finish_time_ = global_next_finish_time;
  }
 private:
  float EstimateTime(const TaskNode* task);

  int wid_;
  int dev_id_;
  float global_next_finish_time_ = FLT_MAX;
  int active_forward_num_ = 0;

  std::unordered_set<TaskNode*> ready_nodes_;
  std::deque<ScheduledTaskInfo> scheduled_tasks_;      // to be finalized
  std::vector<ScheduledTaskInfo> finished_sequence_;
  std::unordered_set<const TaskNode*> finished_set_;

  std::unordered_set<int> active_ctx_set_;
  int64 mem_bytes_ = 0;

  // TODO(shiqing.fsq): set available GPU memory more flexibly.
  int64 mem_size_limit_ = (int64)256*1024*1024*1024;
  const ScheduleIdInfo sched_id_;
};

// represents all GPU scheduling states on a machine
class MachineState {
 public:
  MachineState(int wid, int dev_count, const ScheduleIdInfo& sched_id);
  ~MachineState() {}
  void AddReadyNode(TaskNode* task);
  DevState* GetDevState(int dev_id) const {
    if (dev_id<0) {
      return host_state_.get();
    } else {
      return dev_states_[dev_id].get();
    }
  }
  float FindNextFinishTime() {
    float next_finish_time = host_state_->next_finish_time();
    for (auto& dev_stat : dev_states_) {
      float dev_next_time = dev_stat->next_finish_time();
      if (dev_next_time < next_finish_time) {
        next_finish_time = dev_next_time;
      }
    }
    return next_finish_time;
  }
  void SetGlobalNextFinishTime(float global_next_finish_time) {
    host_state_->SetGlobalNextFinishTime(global_next_finish_time);
    for (auto& dev_stat : dev_states_) {
      dev_stat->SetGlobalNextFinishTime(global_next_finish_time);
    }
  }

  float last_finish_time() const {
    float last = host_state_->last_finish_time();
    for (auto& dev_stat : dev_states_) {
      float dev_last = dev_stat->last_finish_time();
      if (dev_last > last) {
        last = dev_last;
      }
    }
    return last;
  }

  float last_finish_time(int dev_id) const {
    if (dev_id<0) {
      return host_state_->last_finish_time();
    } else {
      return dev_states_[dev_id]->last_finish_time();
    }
  }

  int ScheduleNextTask(const float start_time,
                                 const int micro_num_limit,
                                 ClusterState& cluster_state);
  void ScheduleTask(TaskNode* task, float start_time);
  std::vector<TaskNode*> MarkTaskDoneByTime(float finish_time,
                              ClusterState& cluster_state,
                              OutputBuffersLifeTimeTracker& lifetime_tracker);
  void ExportSchedule(ScheduleInMachine& sched_in_mach);
  void MarkTaskDone(const ScheduledTaskInfo& task_info);
  void DecreaseMemUsage(const TaskNode* task, int output_idx);
  bool TaskScheduled(TaskNode& task) const;
  bool CanRecv(const ClusterState& cluster_state, TaskNode& recv) const;
  bool CanSchedule(const TaskNode& task, const int micro_num_limit) const;
  void PrintState();
  int wid() const { return wid_; }
  bool AllFinished() const {
    if (host_state_->AllFinished() == false) {
      return false;
    }

    for (auto& dev_stat : dev_states_) {
      if (dev_stat->AllFinished() == false) {
        return false;
      }
    }

    return true;
  }
 private:
  int wid_;
  std::vector<std::unique_ptr<DevState>> dev_states_;
  std::unique_ptr<DevState> host_state_;
  const ScheduleIdInfo sched_id_;
};

// represents scheduling states of all GPU in a cluster
class ClusterState {
 public:
  ClusterState(const TaskDAG* task_graph, int id, int sched_size)
  : task_graph_(task_graph)
  , sched_id_(id, sched_size) {
  }
  ~ClusterState() {}
  int sched_stat() const { return sched_stat_; }
  void set_sched_stat(int sched_stat) {
    sched_stat_ = sched_stat;
  }
  float last_finish_time() const {
    float last = 0;
    for (auto& mach_stat : mach_states_) {
      float mach_last = mach_stat->last_finish_time();
      if (mach_last > last) {
        last = mach_last;
      }
    }
    return last;
  }
  float last_finish_time(int worker_id, int dev_id) const {
    return mach_states_[worker_id]->last_finish_time(dev_id);
  }
  void Init(const std::vector<int>& worker_dev_cnt);
  void AddReadyNode(TaskNode* task);
  int ScheduleNextTask(const float start_time, const int micro_num_limit);
  void ScheduleTask(TaskNode* task, float start_time);
  DevState* GetDevState(int wid, int dev_id) const {
    return mach_states_[wid]->GetDevState(dev_id);
  }
  float FindNextFinishTime();
  void SetGlobalNextFinishTime(float global_next_finish_time);
  std::vector<TaskNode*> MarkTaskDoneByTime(float finish_time,
                              OutputBuffersLifeTimeTracker& lifetime_tracker);
  void RecordReadyNodes(const std::vector<TaskNode*>& tasks_done);
  bool ParentsDone(const TaskNode* task) const;
  void ExportSchedule(std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched);
  void MarkTaskDone(const ScheduledTaskInfo& task_info);
  void DecreaseMemUsage(
    const std::vector<std::pair<TaskNode*, int/*output_idx*/>>& tensors_to_release);
  bool TaskScheduled(TaskNode& task) const;
  bool CanRecv(TaskNode& recv) const;
  bool CanSchedule(const TaskNode& task, const int micro_num_limit) const;
  void PrintState();
  const TaskDAG* task_graph() const { return task_graph_; }
  bool AllFinished() const {
    for (auto& mach_stat : mach_states_) {
      if (mach_stat->AllFinished() == false) {
        return false;
      }
    }
    return true;
  }
  int FindRank(const TaskNode* task) const {
    CHECK(task_rank_map_.find(task) != task_rank_map_.end());
    return task_rank_map_.at(task);
  }
  void ResetUpdated() {
    updated_ = false;
  }
  void SetUpdated() {
    updated_ = true;
  }
  bool updated() const {
    return updated_;
  }
  const ScheduleIdInfo& sched_id() const {
    return sched_id_;
  }
 private:
  float cur_time_ = 0;
  int sched_stat_ = ScheduleState::SCHED_NOTHING;
  std::vector<std::unique_ptr<MachineState>> mach_states_;
  const TaskDAG* task_graph_;  // not owned

  std::unordered_map<const TaskNode*, int/*rank*/> task_rank_map_;
  bool updated_ = false;
  const ScheduleIdInfo sched_id_;
};

class TaskScheduler {
 public:
  TaskScheduler(TaskDAG* dag,
                int worker_count,
                const std::vector<int>& worker_dev_cnt,
                int sched_cnt = 2);
  ~TaskScheduler() {
    for (ClusterState* c : cluster_states_) {
      delete c;
    }
    cluster_states_.clear();
  };
  bool Schedule(const int micro_num_limit,
                std::vector<std::unique_ptr<ScheduleInMachine>>& sched_tasks);
  bool ScheduleImpl(const int micro_num_limit,
                ClusterState& cluster_state,
                std::vector<std::unique_ptr<ScheduleInMachine>>& sched_tasks);

 private:
  bool ReorderSend(std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched);
  bool ReorderGA(std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched);
  bool ReorderRecv(std::vector<std::unique_ptr<ScheduleInMachine>>& cluster_sched);

  TaskDAG* task_graph_;
  int worker_count_;
  std::vector<int> worker_dev_cnt_;
  std::vector<ClusterState*> cluster_states_;
};

}  // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_TASK_SCHEDULER_H_


