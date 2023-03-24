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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_EVALUATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_EVALUATOR_H_

#include "tensorflow/compiler/xla/pjrt/dev_id_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

struct StrategySpec {
  bool sync_free_enabled;  // Defaults to false
  bool spmd_enabled;       // Defaults to false
  bool pipeline_enabled;   // Defaults to false
  int num_micro_batches;   // Defaults to 0, no micro batches
  std::vector<int> used_spmd_mesh;
  int num_stages;          // Defaults to 0, no pipeline
  const std::string ToString() const;
};

struct Cost {
  double total_duration = 0.0;
  double gpu_efficiency = 0.0;
  double coll_ratio = 0.0;
  double bubble_ratio = 0.0;

  const std::string ToString() const;
};

class Evaluator {
 public:
  // Assume V100 computing power as default: 15 TFLOP/s, i.e. 0.015 TFLOP/ms.
  explicit Evaluator(
      HloModule* module,
      int64 local_dev_num,
      int64 worker_num,
      double gpu_power_ms = 16492674416.64, // 0.015 * 1024.0 * 1024.0 * 1024.0 * 1024.0
      int64 max_bytes_per_device = 137438953472, // 32GB
      double usage_ratio_ = 0.9,
      double inter_bw = 3355443.2, // 3.125GB/s
      double intra_bw = 322122547.2 // 300GB/s
      );

  Cost Run(HloModule* module, StrategySpec& spec);

 private:

  double EstimateCommTime(
      HloModule::DefContext* def_ctx, HloComputation* comp,
      const std::vector<int>& split_nums);
  int64 EstimateCommBytes(HloComputation* comp);

  int64 local_dev_num_;
  int64 worker_num_;

  const double gpu_power_ms_;
  const int64 max_bytes_per_device_;
  const double inter_bw_;
  const double intra_bw_;
  const double usage_ratio_;
  CommDevManager comm_dev_mgr_;
  HloModule* module_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_EVALUATOR_H_
