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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TASK_GRAPH_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TASK_GRAPH_H_

#include <map>
#include <vector>

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/pjrt/task_scheduler.h"
#include "tensorflow/compiler/xla/pjrt/lifetime_tracker.h"
#include "tensorflow/compiler/xla/pjrt/dev_id_util.h"

namespace xla {

class DAPPLEBuffer;
class DAPPLEExecutable;
class PjRtBuffer;
class TaskDAG;
class TaskDAGDumper;
class LocalClient;
class ExecutableBuildOptions;
class ExecutionPlan;

struct BufferInfo {
  public:
    BufferInfo(): global_device_(-1), 
                  local_device_(-1), 
                  buffer_id_(-1), 
                  buffer_type_(-1), 
                  buffer_reused_(false), 
                  buffer_shape_(){}
    BufferInfo( int global_device, 
                int local_device, 
                int buffer_id, 
                int buffer_type, 
                bool buffer_reused, 
                const std::vector<Shape>& buffer_shape):
      global_device_(global_device), 
      local_device_(local_device), 
      buffer_id_(buffer_id), 
      buffer_type_(buffer_type), 
      buffer_reused_(buffer_reused), 
      buffer_shape_(buffer_shape){}
    BufferInfo( const BufferInfo& buffer_info) {
      global_device_ = buffer_info.global_device_;
      local_device_ = buffer_info.local_device_;
      buffer_id_ = buffer_info.buffer_id_;
      buffer_type_ = buffer_info.buffer_type_;
      buffer_reused_ = buffer_info.buffer_reused_;
      buffer_shape_ = buffer_info.buffer_shape_;
    }
    BufferInfo& operator=(const BufferInfo& other) {
      if (this == &other) return *this;
      this->global_device_ = other.global_device_;
      this->local_device_ = other.local_device_;
      this->buffer_id_ = other.buffer_id_;
      this->buffer_type_ = other.buffer_type_;
      this->buffer_reused_ = other.buffer_reused_;
      this->buffer_shape_ = other.buffer_shape_;
      return *this;
    }

    int global_device() const { return global_device_; }
    int local_device() const { return local_device_; }
    int buffer_id() const { return buffer_id_; }
    int buffer_type() const { return buffer_type_; }
    bool buffer_reused() const { return buffer_reused_; }
    std::vector<Shape> buffer_shape() const { return buffer_shape_; }

    void set_global_device(int global_device) { global_device_=global_device; }
    void set_local_device(int local_device) { local_device_=local_device; }
    void set_buffer_id(int buffer_id) { buffer_id_ = buffer_id; }
    void set_buffer_type(int buffer_type) { buffer_type_ = buffer_type; }
    void set_buffer_reused(bool buffer_reused) { buffer_reused_ = buffer_reused; }
    void set_buffer_shape(std::vector<Shape>& buffer_shape) { buffer_shape_ = buffer_shape; }

  private:
    int global_device_ = -1;
    int local_device_ = -1;
    int buffer_id_ = -1;
    int buffer_type_ = -1;
    bool buffer_reused_ = false;
    std::vector<Shape> buffer_shape_;
};

class TaskNode {
 public:
  #define TASK_TYPE_LIST(V)           \
    V(kSplit, "SPLIT")                \
    V(kInput, "INPUT")                \
    V(kCompute, "COMPUTE")            \
    V(kOutput, "OUTPUT")              \
    V(kSend, "SEND")                  \
    V(kRecv, "RECV")                  \
    V(kAR, "AR")                      \
    V(kGAInit, "GAINIT")              \
    V(kGA, "GA")                      \
    V(kMerge, "MERGE")                \
    V(kMacro, "MACRO")

#if 0
    V(kAny, "ANY")
    V(kPort, "PORT")
#endif

  
  enum class TaskType {
  #define DECLARE_ENUM(enum_name, attr_name, ...) enum_name,
    TASK_TYPE_LIST(DECLARE_ENUM)
  #undef DECLARE_ENUM
  };
  
  static std::string TaskTypeString(TaskType type) {
    switch (type) {
  #define CASE_TASK_TYPE_STRING(enum_name, attr_name, ...) \
    case TaskType::enum_name:                              \
      return attr_name;
      TASK_TYPE_LIST(CASE_TASK_TYPE_STRING)
  #undef CASE_TASK_TYPE_STRING
      default: CHECK(0); return "Alibaba";
    }
  }
  #undef TASK_TYPE_LIST

 public:
  ~TaskNode() {
    if (barrier_ && record_barrier_ == false) {
      // send node destroy cuda event because it is the real owner
      cudaEventDestroy(*barrier_);
    }
  }
  TaskType task_type() const { return task_type_; }
  std::string task_type_string() const { return TaskNode::TaskTypeString(task_type_); }
  int node_id() const { return node_id_; }
  int worker_id() const { return worker_id_; }

  int device_id() const { return device_id_; }
  int sched_idx_in_dev() const { return sched_idx_in_dev_; }
  const std::string& name() const { return name_; }

  void set_name(std::string name) { name_ = name; }
  void set_worker_id(int worker_id) {
    CHECK(worker_id>=0);
    worker_id_ = worker_id;
  }
  void set_device_id(int dev_id) { device_id_ = dev_id; }
  void set_executable(Executable* exe) {
    executable_ = exe; 
    CHECK(def_ctx_);
  }
  void set_sched_idx_in_dev(int sched_idx_in_dev) {
    sched_idx_in_dev_ = sched_idx_in_dev;
  }
  void set_executable(Executable* exe,
                      HloModule::DefContext* def_ctx) {
    executable_ = exe;
    def_ctx_ = def_ctx;
  }
  Executable* executable() {
    CHECK(executable_);
    return executable_;
  }
  Executable* exe_or_null() const { return executable_; }
  HloModule::DefContext* def_ctx() const {
    return def_ctx_;
  }
  int def_id() const {
    CHECK(def_ctx_);
    return def_ctx_->def_id();
  }

  bool has_parent(TaskNode* p) {
    for (auto parent : parents_) {
      if (parent == p) return true;
    }
    return false;
  }

  bool has_child(TaskNode* p) {
    for (auto child : children_) {
      if (child == p) return true;
    }
    return false;
  }

  const TaskNode* parent() const {
    CHECK(1 == parents_.size());
    return parents_[0];
  }

  TaskNode* parent() {
    CHECK(1 == parents_.size());
    return parents_[0];
  }

  TaskNode* parent(int index) {
    CHECK(index < int(parents_.size()));
    return parents_[index];
  }

  TaskNode* child() {
    CHECK(1 == children_.size());
    return children_[0];
  }

  void add_child(TaskNode* child) {
    children_.push_back(child);
  }

  void remove_child(TaskNode* child) {
    auto it = std::find(children_.begin(), children_.end(), child);
    CHECK(it != children_.end());
    children_.erase(it);
  }

  void add_parent(TaskNode* parent) {
    parents_.push_back(parent);
  }

  void remove_parent(TaskNode* parent) {
    auto it = std::find(parents_.begin(), parents_.end(), parent);
    CHECK(it != parents_.end());
    parents_.erase(it);
  }

  const std::vector<TaskNode*>& children() const { return children_; }
  const std::vector<TaskNode*>& parents() const { return parents_; }

  std::vector<TaskNode*>& children() { return children_; }
  std::vector<TaskNode*>& parents() { return parents_; }

  std::vector<int64>* mutable_send_recv_global_devs() {
    return &send_recv_global_devs_;
  }

  const std::vector<int64>& send_recv_global_devs() const {
    return send_recv_global_devs_;
  }

  std::map<int/*arg_no*/,
           std::vector<std::pair<TaskNode*, int>>>* mutable_input_specs() {
    return &input_specs_;
  }

  const std::map<int/*arg_no*/,
           std::vector<std::pair<TaskNode*, int>>>& input_specs() const {
    return input_specs_;
  }

  std::map<int/*out_idx*/, int/*arg_no*/>& port_map() {
    return port_map_;
  }

  std::map<int/*out_idx*/, int/*arg_no*/>* mutable_port_map() {
    return &port_map_;
  }

  std::vector<std::pair<int/*node_id*/, int/*output_idx*/>>& mem_to_release() {
    return mem_to_release_;
  }

  bool across_machine() const { return across_machine_; }
  void set_across_machine(bool across_machine) {
    across_machine_ = across_machine;
  }
  bool comm_with_lower_stage() const { return comm_with_lower_stage_; }
  void set_comm_with_lower_stage(bool comm_with_lower_stage) {
    comm_with_lower_stage_ = comm_with_lower_stage;
  }
  const SplitId& split_id() const {
    return split_id_;
  }
  const BufferInfo& buffer_info() const { 
    return buffer_info_; 
  }
  void set_comm_dev_mgr(const std::shared_ptr<CommDevManager>& comm_dev_mgr) {
    comm_dev_mgr_ = comm_dev_mgr;
  }
  void set_buffer_info(const BufferInfo& buffer_info) {
    buffer_info_ = buffer_info;
  }
  int micro_id() const {
    return split_id_.micro_id();
  }
  int stage_id() const {
    return split_id_.stage_id();
  }
  std::shared_ptr<CommDevManager> comm_dev_mgr() const {
    return comm_dev_mgr_;
  }
  std::shared_ptr<cudaEvent_t> CreateBarrier() {
    barrier_ = std::make_shared<cudaEvent_t>();
    cudaEventCreate(barrier_.get());
    return barrier_;
  }
  std::shared_ptr<cudaEvent_t> CreateBufferBarrier() {
      buffer_barrier_ = std::make_shared<cudaEvent_t>();
      cudaEventCreate(buffer_barrier_.get());
      return buffer_barrier_;
  }
  std::shared_ptr<cudaEvent_t> CreateReleaseBarrier() {
    release_barrier_ = std::make_shared<cudaEvent_t>();
    cudaEventCreate(release_barrier_.get());
    return release_barrier_;
  }
  std::shared_ptr<cudaEvent_t> barrier() const {
    return barrier_;
  }
  std::shared_ptr<cudaEvent_t> release_barrier() const {
    return release_barrier_;
  }
  std::shared_ptr<cudaEvent_t> buffer_barrier() const { 
    return buffer_barrier_; 
  }
  std::shared_ptr<cudaEvent_t> buffer_wait_barrier() const { 
    return buffer_wait_barrier_; 
  }
  bool buffer_record_barrier() const { 
    return buffer_record_barrier_; 
  }
  void set_barrier(std::shared_ptr<cudaEvent_t> barrier) {
    barrier_ = barrier;
  }
  void set_record_barrier(bool record) {
    record_barrier_ = record;
  }
  bool record_barrier() const {
    return record_barrier_;
  }
  void set_buffer_barrier(std::shared_ptr<cudaEvent_t> barrier) { 
    buffer_barrier_ = barrier; 
  }
  void set_buffer_wait_barrier(std::shared_ptr<cudaEvent_t> barrier) { 
    buffer_wait_barrier_ = barrier; 
  }
  void set_buffer_record_barrier(bool record) { 
    buffer_record_barrier_ = record; 
  }
 protected:
  friend class TaskDAG;
  explicit TaskNode(TaskType type, const std::vector<int64>& addr,
                    const std::vector<bool>& share_dev_flags,
                    int stage_split_ordinal,
                    int node_id, const std::string& def_ctx_name);

 private:
  TaskType task_type_;
  BufferInfo buffer_info_;
  int node_id_ = -1;    // global unique id
  int worker_id_ = 0;
  int device_id_ = -1;
  int sched_idx_in_dev_ = -1;
  std::string name_;
  Executable* executable_ = nullptr;
  HloModule::DefContext* def_ctx_ = nullptr;

  SplitId split_id_;
  std::shared_ptr<CommDevManager> comm_dev_mgr_;

 private:
  std::map<int/*out_idx*/, int/*arg_no*/> port_map_;

  // For send/recv usage
  std::vector<int64> send_recv_global_devs_;

  std::map<int/*arg_no*/,
           std::vector<std::pair<TaskNode*, int/*out_idx*/>>> input_specs_;

  std::vector<TaskNode*> parents_, children_;
  std::vector<std::pair<int/*release node_id*/, int/*output_idx*/>> mem_to_release_;

  // specific for send/recv node between pipeline stages
  // we'd better derive Send/Recv class from TaskNode
  bool comm_with_lower_stage_ = false;
  bool across_machine_ = false;

  std::shared_ptr<cudaEvent_t> barrier_;
  std::shared_ptr<cudaEvent_t> release_barrier_;
  bool record_barrier_ = false;  // true: record barrier;  false: wait barrier
  std::shared_ptr<cudaEvent_t> buffer_barrier_;
  std::shared_ptr<cudaEvent_t> buffer_wait_barrier_;
  bool buffer_record_barrier_ = false;  // true: record barrier;  false: wait barrier
};

// Note: TaskDAG instances are not arbitrary DAGs, but with
// scheduling/placement constraints for runtime execution.
class TaskDAG {
 public:
  explicit TaskDAG(
      std::vector<std::pair<HloModule::DefContext*, HloModule*>>& def_hlo_pairs,
      const std::vector<int>& split_nums, const std::vector<bool>& share_dev_flags,
      const std::vector<int>& placement_layout, int stage_split_ordinal,
      int num_workers)
    : split_nums_(split_nums)
    , share_dev_flags_(share_dev_flags)
    , placement_layout_(placement_layout)
    , stage_split_ordinal_(stage_split_ordinal) {
    top_def_ctx_ = def_hlo_pairs.back().first;

    for (auto& pair : def_hlo_pairs) {
      auto def = pair.first;
      id_def_map_[def->def_id()] = def;
      //cluster_def_module_map_[def] = pair.second;
      def_module_map_[def] = pair.second;
      //def_module_map_[def] = std::move(pair.second);
    }

    // build comm device manager
    comm_dev_mgr_ = std::make_shared<CommDevManager>(split_nums_,
                                                     share_dev_flags_,
                                                     placement_layout_,
                                                     stage_split_ordinal,
                                                     num_workers);

    // debug
    VLOG(2) << "def_module_map in constructor of TaskDAG: ";
    for (auto& def_mod : def_module_map_) {
      VLOG(2) << "def ctx name: " << def_mod.first->name();
    }

    scope_mode_ = true;
  }

  explicit TaskDAG(HloModule::DefContext* top_def_ctx,
                   const std::vector<int>& split_nums,
                   const std::vector<bool>& share_dev_flags,
                   const std::vector<int>& placement_layout,
                   int stage_split_ordinal,
                   int num_workers)
    : source_(nullptr)
    , sink_(nullptr)
    , top_def_ctx_(top_def_ctx)
    , split_nums_(split_nums)
    , share_dev_flags_(share_dev_flags)
    , placement_layout_(placement_layout)
    , stage_split_ordinal_(stage_split_ordinal) {
    // build comm device manager
    comm_dev_mgr_ = std::make_shared<CommDevManager>(split_nums_,
                                                     share_dev_flags_,
                                                     placement_layout_,
                                                     stage_split_ordinal,
                                                     num_workers);
  }

  explicit TaskDAG() {}

  bool scope_mode() const { return scope_mode_; }

  void add_executable(TaskNode* task, Executable* exe) {
    task->set_executable(exe);
    auto def = task->def_ctx();

    def_exe_ptr_map_[def] = exe;
    exe_def_ptr_map_[exe] = def;
  }

  HloModule::DefContext* top_def_ctx() {
    CHECK(top_def_ctx_);
    return top_def_ctx_; 
  }

  void setup_def_exe(HloModule::DefContext* def_ctx, Executable* exe) {
    local_def_exe_map_[def_ctx] = exe;
    local_exe_def_map_[exe] = def_ctx;
    id_def_map_[def_ctx->def_id()] = def_ctx;
  }

  void setup_id_def(const std::vector<std::unique_ptr<TaskNode>>& tasks) {
    for (auto& task : tasks) {
      auto def_ctx = task->def_ctx();
      CHECK(def_ctx);
      id_def_map_[def_ctx->def_id()] = def_ctx;
    }
  }

  std::unordered_map<HloModule::DefContext*, Executable*>& def_exe_map() {
    return local_def_exe_map_;
  }

  std::unordered_map<Executable*, HloModule::DefContext*>& exe_def_map() {
    return local_exe_def_map_;
  }

  const std::unordered_map<int, HloModule::DefContext*>& id_def_map() const {
    return id_def_map_;
  }

  std::unordered_map<int, HloModule::DefContext*>* mutable_id_def_map() {
    return &id_def_map_;
  }

  void set_exe_plan(ExecutionPlan* exe_plan) {
    exe_plan_ = exe_plan;
  }
  HloModule* task_module(const TaskNode* task) const;

  HloModule* GetModule(HloModule::DefContext* def_ctx) {
    CHECK(def_module_map_.count(def_ctx));
    return def_module_map_[def_ctx];
  }

  bool has_source() { return source_ != nullptr; }
  bool has_sink() { return sink_ != nullptr; }

  const TaskNode* source() const {
    CHECK(source_);
    return source_;
  }

  const TaskNode* sink() const {
    CHECK(sink_);
    return sink_; 
  }

  TaskNode* source() {
    CHECK(source_);
    return source_;
  }

  TaskNode* sink() {
    CHECK(sink_);
    return sink_; 
  }

  TaskNode* add_source() {
    CHECK(!source_);
    task_nodes_.emplace_back(
        absl::WrapUnique(new TaskNode(TaskNode::TaskType::kSplit, {}, {}, -1,
                                      task_nodes_.size(), /*def_ctx_name=*/"")));
    source_ = task_nodes_.back().get();
    return source_;
  }

  TaskNode* add_sink(const std::vector<TaskNode*>& parents) {
    CHECK(!sink_);
   task_nodes_.emplace_back(
        absl::WrapUnique(new TaskNode(TaskNode::TaskType::kMerge, {}, {}, -1,
                                      task_nodes_.size(), /*def_ctx_name=*/"")));
    sink_ = task_nodes_.back().get();
    for (auto parent : parents) {
      sink_->add_parent(parent);
      parent->add_child(sink_);
    }

    return sink_;
  }

  TaskNode* new_task_node(std::string name, int task_type_id,
                             const SplitId& split_id,
                             int worker_id, int def_id) {
    CHECK(id_def_map_.count(def_id));
    auto* def_ctx = id_def_map_.at(def_id);
    auto task = new_task_node(name, task_type_id, split_id, worker_id, def_ctx);
    node_id_task_[task->node_id()] = task;
    return task;
  }

  TaskNode* new_task_node(std::string name, int task_type_id,
                             const SplitId& split_id,
                             int worker_id,
                             HloModule::DefContext* def_ctx);

  TaskNode* brief_clone(const TaskNode* task);
  TaskNode* clone(TaskNode* task);

  TaskNode* new_task_node(TaskNode::TaskType task_type,
                             const std::vector<int64>& addr,
                             const std::vector<bool>& share_dev_flags,
                             int stage_split_ordinal,
                             const std::vector<TaskNode*>& parents,
                             std::string def_ctx_name = "") {
    if (task_type == TaskNode::TaskType::kMacro) {
      CHECK(0);
    } else {
      task_nodes_.emplace_back(
          absl::WrapUnique(new TaskNode(task_type, addr, share_dev_flags,
                                        stage_split_ordinal,
                                        task_nodes_.size(), def_ctx_name)));
    }

    auto node = task_nodes_.back().get();
    node->comm_dev_mgr_ = comm_dev_mgr_;
    for (auto parent : parents) {
      node->add_parent(parent);
      parent->add_child(node);
    }

    return node;
  }

  const std::vector<std::unique_ptr<TaskNode>>& task_nodes() const {
    return task_nodes_; 
  }

  std::unordered_map<TaskNode*, std::vector<TaskNode*>>& spmd_partners() {
    return spmd_partners_;
  }

  std::unordered_set<TaskNode*>& spmd_clones() { return spmd_clones_; }

  void ResolveCGLossOutputs() {
    auto merge = sink();
    auto& merge_input_specs = merge->input_specs();
    for (auto& it : merge_input_specs) {
      auto& input_specs = it.second;
      for (auto& spec : input_specs) {
        auto task = spec.first;
        auto out_idx = spec.second;
        if (task->task_type() == TaskNode::TaskType::kOutput &&
            (task->def_ctx()->cg_def_ctx() ||
             task->def_ctx()->cg_slice_def_ctx())) {
          auto compute = task->parent();
          auto& loss_outputs = cg_loss_outputs_[compute->node_id()];
          VLOG(0) << "ResolveCGLossOutputs->" << compute->name()
                  << ":" << out_idx;
          loss_outputs.insert(out_idx);
        }
      }
    }
  }

  std::unordered_map<int/*node_id*/,
                     std::unordered_set<int>>& cg_loss_outputs() { 
    return cg_loss_outputs_; 
  }

  void PostOrderDFS(TaskNode* task,
                    std::unordered_set<TaskNode*>& visited,
                    int& dfs_post_order) {
    visited.insert(task);
    for (auto parent : task->parents()) {
      if (!visited.count(parent)) {
        PostOrderDFS(parent, visited, dfs_post_order);
      }
    }

    task_post_order_[task] = dfs_post_order;
    post_order_task_[dfs_post_order] = task;
    ++dfs_post_order;
  }

  void BuildDominanceTree() {
    auto merge = sink();
    std::unordered_set<TaskNode*> visited;
    int dfs_post_order = 0;
    PostOrderDFS(merge, visited, dfs_post_order);

    auto intersect = [this](TaskNode* b1, TaskNode* b2) -> TaskNode* {
      auto finger1 = b1;
      auto finger2 = b2;
      while (finger1 != finger2) {
        while (task_post_order_[finger1] < task_post_order_[finger2]) {
          CHECK(imm_dom_.count(finger1));
          finger1 = imm_dom_[finger1];
        }

        while (task_post_order_[finger2] < task_post_order_[finger1]) {
          CHECK(imm_dom_.count(finger2));
          finger2 = imm_dom_[finger2];
        }
      }
      return finger1;
    };

    imm_dom_[merge] = merge;
    bool changed = true;
    while (changed) {
      changed = false;

      for (auto rit = post_order_task_.rbegin();
                rit != post_order_task_.rend(); ++rit) {
        auto b = rit->second;
        if (b == merge) continue;

        decltype(b) new_idom = nullptr;
        for (auto p : b->children()) {
          if (!imm_dom_.count(p)) continue;

          if (!new_idom) {
            new_idom = p;
          } else {
            new_idom = intersect(p, new_idom);
          }
        }

        if (!imm_dom_.count(b) || imm_dom_[b] != new_idom) {
          imm_dom_[b] = new_idom;
          changed = true;
        }
      } // for (rit ...)
    } // while(changed)
  }

  // Test if A dominates B
  bool Dominates(TaskNode* A, TaskNode* B) {
    while (A != B && B != sink()) {
      CHECK(imm_dom_.count(B));
      B = imm_dom_[B];
    }
    return A == B;
  }

  void SetupInputSpecs(int worker_count);
  virtual TaskDAG* top_graph() { return this; }
  virtual HloModule::DefContext* def_ctx() { return top_def_ctx_; }
  const std::string& name() { return name_; }
  void SetName(const std::string& name) { name_ = name; }
  void Dump(std::string dag_filename);

  void set_num_dev_per_worker(int num_dev_per_worker ) {
    num_dev_per_worker_ = num_dev_per_worker;
  }

  int num_dev_per_worker() const { return num_dev_per_worker_; }
  const std::vector<int>& split_nums() const {
    return split_nums_;
  }
  const std::vector<bool>& share_dev_flags() const {
    return share_dev_flags_;
  }
  int stage_split_ordinal() const {
    return stage_split_ordinal_;
  }
  const std::vector<int>& placement_layout() const {
    return placement_layout_;
  }
  std::shared_ptr<CommDevManager> comm_dev_mgr() const {
    return comm_dev_mgr_;
  }
 private:
  std::string name_;
  // this worker's binary info:
  std::unordered_map<HloModule::DefContext*, Executable*> local_def_exe_map_;
  std::unordered_map<Executable*, HloModule::DefContext*> local_exe_def_map_;
  std::unordered_map<int, HloModule::DefContext*> id_def_map_;
  std::unordered_map<HloModule::DefContext*, HloModule*> def_module_map_;

  std::map<int, TaskNode*> node_id_task_;

  // other workers's binary info:
  std::vector<std::unique_ptr<Executable>> remote_executables_;
  std::unordered_map<HloModule::DefContext*, Executable*> remote_def_exe_map_;
  std::unordered_map<Executable*, HloModule::DefContext*> remote_exe_def_map_;

  std::unordered_map<HloModule::DefContext*, Executable*> def_exe_ptr_map_;
  std::unordered_map<Executable*, HloModule::DefContext*> exe_def_ptr_map_;

  bool scope_mode_ = false;

  // Total number of physical stages across all master/slave worker nodes.
  TaskNode *source_ = nullptr;
  TaskNode *sink_ = nullptr;
  HloModule::DefContext* top_def_ctx_ = nullptr;
  std::vector<std::unique_ptr<TaskNode>> task_nodes_;

  // sharding partners
  std::unordered_map<TaskNode*, std::vector<TaskNode*>> spmd_partners_;
  std::unordered_set<TaskNode*> spmd_clones_;

  std::unordered_map<int/*node_id*/, 
                     std::unordered_set<int>> cg_loss_outputs_;

  // Dominance Relation among Task Nodes
  // References:
  // I: A Simple, Fast Dominance Algorithm. Keith D. Cooper, et al.
  // II: Ben Harderkopf, Program Analysis Lecture Notes @ UCSB
  std::map<int, TaskNode*> post_order_task_;
  std::unordered_map<TaskNode*, int> task_post_order_;
  std::unordered_map<TaskNode*, TaskNode*> imm_dom_;

  int num_dev_per_worker_ = 1;
  ExecutionPlan* exe_plan_ = nullptr;

  std::vector<int> split_nums_;
  std::vector<bool> share_dev_flags_;
  int stage_split_ordinal_ = -1;
  std::vector<int> placement_layout_;
  std::shared_ptr<CommDevManager> comm_dev_mgr_;
};

class TaskDAGDumper {
 public:
  TaskDAGDumper(const TaskDAG& dag)
    : dag_(dag) {}
  std::string Dump();

 private:
  std::string Header();
  std::string Body();
  std::string Footer();

  const TaskDAG& dag_;
};

}

#endif // TENSORFLOW_COMPILER_XLA_PJRT_TASK_GRAPH_H_

