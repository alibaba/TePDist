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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_SPLITTING_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_SPLITTING_ANALYSIS_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"
#include "tensorflow/compiler/xla/service/parallel/hlo_strategy_spec.h"
#include "tensorflow/core/platform/macros.h"

#include <map>
#include <unordered_set>

namespace xla {

class SyncFreeSplittingAnalysis {
 public:
  explicit SyncFreeSplittingAnalysis(int64 num_stages, int64 num_micro_batches=0)
      : num_stages_(num_stages), num_micro_batches_(num_micro_batches) {}

  HloInstMap<DimStrategy> SearchForMostSyncFreeInsts(HloModule* module);
  std::vector<HloInstruction*> ExtractSampleInputs(HloModule* module);
  std::vector<std::vector<int>> GenerateProposals(std::vector<HloInstruction*>& instrs);
  bool Validate(HloModule* module, HloInstMap<DimStrategy>& strategy_map);
  const int num_micro_batches() const { return num_micro_batches_; }

  StatusOr<bool> Run(HloModule* module);
 private:
  void RecordStrategyToInsts(HloModule* module,
                             HloInstMap<DimStrategy>& strategy_map);

  int64 num_stages_;
  int64 num_micro_batches_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_SPLITTING_ANALYSIS_H_
