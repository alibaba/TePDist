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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_PLAN_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_PLAN_H_

#include <map>
#include <vector>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/pjrt/task_graph.h"
#include "tensorflow/compiler/xla/pjrt/task_scheduler.h"
#include "tensorflow/compiler/xla/pjrt/lifetime_tracker.h"

namespace xla {

class DAPPLEBuffer;
class DAPPLEExecutable;
class PjRtBuffer;
class TaskDAG;
class TaskDAGDumper;
class LocalClient;
class ExecutableBuildOptions;
class ExecutionPlan;

class ExecutionPlan {
 public:
  explicit ExecutionPlan(std::unique_ptr<TaskDAG> task_graph,
                           int num_workers, int worker_rank)
    : task_graph_(std::move(task_graph))
    , worker_count_(num_workers)
    , worker_rank_(worker_rank) {
      task_graph_->set_exe_plan(this);
    }
  virtual ~ExecutionPlan() {
  }

  void ScheduleTasks();

  int worker_count() { return worker_count_; }
  int worker_rank() { return worker_rank_; }

  std::vector<std::unique_ptr<ScheduleInMachine>>& scheduled_tasks() {
    return scheduled_tasks_;
  }

  void SourceCalibration(bool has_sub_def = true);
  void CrossDeviceCalibration();

  void ShowWorkerTaskList(int rank) {
    VLOG(2) << "Worker Task List->" << rank << "\n";
    if (scheduled_tasks_.empty())  // empty for dist plan on worker side
      return;
    std::vector<std::vector<TaskNode*>>& per_device_tasks =
                                  scheduled_tasks_[rank]->per_device_tasks_;
    VLOG(2) << "per_device_tasks.size: " << per_device_tasks.size();
    for (int dev_id = 0; dev_id < per_device_tasks.size(); ++dev_id) {
      std::vector<TaskNode*>& cur_device_tasks = per_device_tasks[dev_id];
      VLOG(2) << "dev id: " << dev_id << ", cur_device_tasks: " << cur_device_tasks.size();
      for (auto task : cur_device_tasks) {
        CHECK(task != nullptr);
        CHECK(task->def_ctx());
        VLOG(0) << " Task type:" << task->task_type_string()
                << " def-ctx:" << task->def_ctx()->name()
                << " task: " << task->name()
                << " replicas per instance:" << task->micro_id()
                << " device-id:" << dev_id
                << " worker-id:" << rank << "\n" << std::flush;
      }
      VLOG(0) << std::endl;
    }
  }

  TaskDAG* task_graph() { return task_graph_.get(); }
  virtual HloModule* task_module(const TaskNode* /*task*/) const {
    return nullptr;
  }
 protected:
  std::unique_ptr<TaskDAG> task_graph_;
  int worker_count_ = 1;
  int worker_rank_ = 0;
  std::vector<std::unique_ptr<ScheduleInMachine>> scheduled_tasks_;

  bool will_be_oom_ = false;


  std::unique_ptr<CommDevManager> comm_dev_mgr_;
};

class DistributedPlan : public ExecutionPlan {
 public:
  explicit DistributedPlan(std::unique_ptr<TaskDAG> task_graph,
                           std::vector<std::pair<HloModule::DefContext*,
                                                 std::unique_ptr<HloModule>>>& def_hlo_pairs,
                           int num_workers, int worker_rank)
    : ExecutionPlan(std::move(task_graph), num_workers, worker_rank)
    , def_hlo_pairs_(std::move(def_hlo_pairs)) {
      for (auto& def_hlo : def_hlo_pairs_) {
        VLOG(2) << "def: " << def_hlo.first->name();
        if (def_hlo.first->entry_def_ctx()) {
          top_def_ctx_ = def_hlo.first;
        }
        def_mod_map_[def_hlo.first] = def_hlo.second.get();
      }
    }

  virtual ~DistributedPlan() {}
  int total_num_send_recv_groups() { return total_num_send_recv_groups_; }

  void SetupDefExeMap(std::vector<std::pair<HloModule::DefContext*,
                      std::unique_ptr<Executable>>>& def_exe_pairs);

  void ExeDecoration();

  std::vector<TaskNode*>& WorkerTaskList(int rank) {
    CHECK(per_worker_tasks_.count(rank));
    return per_worker_tasks_[rank];
  }

  void BuildWorkerTaskList(int rank) {
    auto& tasks = per_worker_tasks_[rank];
    for (auto& task : task_graph_->task_nodes()) {
      auto wid = task->worker_id();
      if (wid != rank) continue;

      tasks.push_back(task.get());
    }
  }

  void WorkerAssignment();
  void DeviceAssignment();

  HloModule* hlo_module(HloModule::DefContext* def) const {
    if (def_mod_map_.find(def) == def_mod_map_.end()) return nullptr;
    return def_mod_map_.at(def);
  }
  virtual HloModule* task_module(const TaskNode* task) const {
    auto def = task->def_ctx();
    return hlo_module(def);
  }
  std::vector<std::pair<HloModule::DefContext*, std::unique_ptr<HloModule>>>& def_hlo_pairs() {
    return def_hlo_pairs_;
  }
  HloModule::DefContext* top_def_ctx() { return top_def_ctx_; }

  HloModule* top_module() const {
    return hlo_module(top_def_ctx_);
  }
 private:
  int total_num_send_recv_groups_ = 0;
  std::map<int, std::vector<TaskNode*>> per_worker_tasks_;
  std::vector<std::pair<HloModule::DefContext*, std::unique_ptr<HloModule>>> def_hlo_pairs_;
  std::unordered_map<HloModule::DefContext*,
                     std::unique_ptr<Executable>> def_exe_map_;
  std::unordered_map<HloModule::DefContext*, HloModule*> def_mod_map_;
  HloModule::DefContext* top_def_ctx_ = nullptr;
};

// Note: The task graph for Sharding/DP/Pipeline execution has special
// structures. We take advantage of this to build execution plans.
class LocalPlan : public ExecutionPlan {
 public:
  explicit LocalPlan(std::unique_ptr<TaskDAG> task_graph, 
                     std::shared_ptr<DistributedPlan> dist_plan,
                     int device_count)
    : ExecutionPlan(std::move(task_graph), 1, 0)
    , local_device_count_(device_count)
    , dist_plan_(dist_plan) {
    }

  virtual ~LocalPlan() {}
  int local_device_count() { return local_device_count_; }

  void MakeTaskGraphGCPlan();
  void ShowPerDeviceTaskList();
  void RecordBufferReuseInfo();

  bool Ready(TaskNode* task, 
             std::unordered_set<TaskNode*>& sched);

  // restore local schedule based on global schedule information
  void RestoreLocalSchedule();

  void BuildTaskSpan(std::unordered_map<TaskNode*, int>& task_span);

  std::vector<TaskNode*>& task_list(int index) {
    CHECK(scheduled_tasks_.size() == 1);
    int task_size = scheduled_tasks_[0]->per_device_tasks_.size();
    CHECK(index < task_size);
    return scheduled_tasks_[0]->per_device_tasks_[index];
  }

  bool has_work(int device_id) {
    CHECK(scheduled_tasks_.size() == 1);
    return device_id < int(scheduled_tasks_[0]->per_device_tasks_.size());
  }

  // Returns the actual number of devices *used*.
  int64 used_device_count() {
    CHECK(scheduled_tasks_.size() == 1);

    int64 count = 0;

    // Confirm that all devices are assigned with tasks.
    for (auto& per_device_tasks : scheduled_tasks_[0]->per_device_tasks_) {
      if (per_device_tasks.size() > 0) {
        ++count;
      }
    }

    return count;
  }

  int task_device(TaskNode* task) {
    if (task_device_map_.count(task)) {
      return task_device_map_[task];
    }

    return -1;
  }

  int shard_devices() { return shard_devices_; }
  int stage_devices() { return stage_devices_; }

  DistributedPlan* distributed_plan() { return dist_plan_.get(); }
  void set_dapple(DAPPLEExecutable* dapple) { dapple_exe_ = dapple; }
  DAPPLEExecutable* dapple_executable() { return dapple_exe_; }

  int worker_count() { return worker_count_; }
  void set_worker_count(int worker_count) { worker_count_ = worker_count; }
  int worker_rank() { return worker_rank_; }
  void set_worker_rank(int worker_rank) { worker_rank_ = worker_rank; }

  virtual HloModule* task_module(const TaskNode* task) const {
    return dist_plan_->task_module(task);
  }

 private:

  int local_device_count_ = 0;
  std::shared_ptr<DistributedPlan> dist_plan_;
  std::unordered_map<TaskNode*, int> task_device_map_;
  int shard_devices_ = 1;
  int stage_devices_ = 1;
  DAPPLEExecutable* dapple_exe_ = nullptr;
  int worker_count_ = 1;
  int worker_rank_ = 0;
};

struct BufferReuseAnalysis {
public:
  BufferReuseAnalysis()
  : recv_type_count_(0)
  , task_classify_()
  , current_buffer_index_()
  , pre_event_() {
  }

  void SetReuseInfo(TaskNode* task);
  void InsertEventForSynchronizing(TaskNode* task);

private:
  int recv_type_count_ = 0;
  // task_classify is used to classify recv tasknode, use pair<port_map,def_id> as key 
  // value is also a pair, pair.first is recv_type, pair.second is the number of this recv_type 
  // generally, different recv type has different buffer size, so pair.first also can represent as buffer type
  std::map<std::pair<std::vector<int>/*port_map*/,int/*def_id*/>, std::pair<int/*recv/buffer type*/,int/*count*/>> task_classify_;
  // current_buffer_index decide next buffer id to use of specific buffer type
  std::map<int/*buffer type*/, int/*current index*/> current_buffer_index_;
  // pre_event is used to insert cudaevent for synchronizing tasknode that use the same buffer id
  // it also can represent as one way to hold buffer liveness 
  std::map<std::pair<int/*buffer type*/,int/*buffer id*/>, std::shared_ptr<cudaEvent_t>> pre_event_;
};

}

#endif // TENSORFLOW_COMPILER_XLA_PJRT_EXECUTION_PLAN_H_

