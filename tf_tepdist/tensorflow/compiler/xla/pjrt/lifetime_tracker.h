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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_LIFETIME_TRACKER_H
#define TENSORFLOW_COMPILER_XLA_PJRT_LIFETIME_TRACKER_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/default/logging.h"

namespace xla {

class TaskDAG;
class TaskNode;

// BufferRefCount is the reference cpounter of one single DAPPLEBuffer
// which produces by ComputeTask or GA.
class BufferRefCount {
 public:
  BufferRefCount(int node_id, int output_id, int total, bool persistant)
      : node_id_(node_id), output_id_(output_id),
        total_(total), current_(0), persistant_(persistant) {}

  void Reset() { current_ = total_; }
  void IncreaseTotalRefCount() { ++total_; }

  void DecreaseRefCount() {
    CHECK(current_ > 0);
    --current_;
  }

  bool Releasable() { return !current_ && !persistant_; }

  int total() const {
    return total_;
  }

  int current() const {
    return current_;
  }

 private:
  int node_id_;
  int output_id_;
  int total_; // Total reference count at the begining
  int current_; // Counter for execution
  bool persistant_; // A persistant DAPPLEBuffer is never released. 
};

// A life time tracker for all outputs DAPPLEBuffer.
class OutputBuffersLifeTimeTracker {
   public:
    explicit OutputBuffersLifeTimeTracker() {}
    // Only initialize with zero reference count
    void Initialize(const TaskDAG* task_graph);
    void IncreaseTotalRefCount(TaskNode* task, int output_id);
    void DecreaseRefCount(TaskNode* task, int output_id);
    bool Releasable(TaskNode* task, int output_id);
    void ResetBufferRefCounts();
    std::vector<std::pair<TaskNode*, int/*output_idx*/>>
    CollectReleasableTensorsForSelf(TaskNode* output_task,
                                        const TaskDAG* task_graph);
    std::vector<std::pair<TaskNode*, int/*output_idx*/>>
    CollectReleasableTensorsForParent(TaskNode* output_task,
                                          const TaskNode* input_task);

   private:
    BufferRefCount* CreateBufRefCount(int node_id, int output_id, bool persistant);
    std::unordered_map<std::string/*name_id*/, BufferRefCount*> name_buffer_ref_map_;
    std::vector<std::unique_ptr<BufferRefCount>> buf_ref_counts_pool_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_LIFETIME_TRACKER_H

