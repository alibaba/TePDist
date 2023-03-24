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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_AUTO_PARALLEL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_AUTO_PARALLEL_H_

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"
#include "tensorflow/compiler/xla/service/parallel/evaluator.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

struct DeviceSplitPlan {
  template <typename H>
  friend H AbslHashValue(H h, const DeviceSplitPlan& plan) {
    return H::combine(std::move(h), plan.num_stages, plan.spmd_mesh);
  }
  friend bool operator==(const DeviceSplitPlan& a, const DeviceSplitPlan& b) {
    return a.num_stages == b.num_stages &&
           a.spmd_mesh == b.spmd_mesh &&
           a.use_local_accumulation == b.use_local_accumulation;
  }

  bool IsMeshMultipleSplit() {
    int count = 0;
    for (int d : spmd_mesh) {
      count += (d > 1);
    }

    return count > 1;
  }
  int num_stages;
  std::vector<int> spmd_mesh;
  bool use_local_accumulation = true;
};

class AutoParallel : public HloModulePass {
 public:
  explicit AutoParallel(int64 num_worker, int64 local_dev_num);

  ~AutoParallel() override = default;

  absl::string_view name() const override { 
    return "Automatic parallelization strategy"; 
  }

  StatusOr<bool> Run(HloModule* module) override;

  TF_DISALLOW_COPY_AND_ASSIGN(AutoParallel);

 private:
  std::vector<DeviceSplitPlan> GenerateSplitProposals();
  void DeepCopyHloModule(HloModule* src, HloModule* dst);
  void DoSyncFreeDecomposition(HloModule* module);
  void SetupEntryDefContext(HloModule* module);
  void SetupInputOutputAliasMap(HloModule* module);
  StatusOr<bool> RunFastMode(HloModule* module);
  StatusOr<bool> RunExplorationlMode(HloModule* module);
  StatusOr<bool> RunConfiglMode(HloModule* module);

  bool fast_mode_ = false;
  int64 num_stages_;
  int64 num_micro_batches_;
  int64 worker_num_;
  int64 local_dev_num_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_AUTO_PARALLEL_H_
