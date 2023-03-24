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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_DECOMPOSITION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_DECOMPOSITION_H_

#include "tensorflow/compiler/xla/pjrt/dev_id_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/parallel/computation_clone_utils.h"
#include "tensorflow/compiler/xla/service/parallel/sync_free_chain.h"

#include "tensorflow/core/platform/macros.h"

namespace xla {

class SyncFreeDecomposition {
 public:
  explicit SyncFreeDecomposition(int share_split_ordinal)
      : share_split_ordinal_(share_split_ordinal) {}
  ~SyncFreeDecomposition() = default;
  StatusOr<bool> Run(HloModule* module, SyncFreeChain& sync_free_chain);
 private:
  std::vector<const HloInstruction*> get_fetches_insts(
      HloComputation* computation, int num_fetches);

  CloneSignature BuildSyncFreeSignature(
      HloComputation* entry, std::vector<const HloInstruction*>& root_ops);

  void SetupSyncFreeDefContext(
      CloneSignature& signature, HloModule* module,
      HloComputation* computation, std::vector<const HloInstruction*>& params,
      std::vector<const HloInstruction*>& outputs);

  HloComputation* BuildSyncFreeComputation(
      HloModule* module, SyncFreeChain& sync_free_chain);

  CloneSignature BuildSignatureForRemains(
      HloComputation* entry, SyncFreeChain& sync_free_chain);

  void SetupDefContextForRemains(
      CloneSignature& signature, HloModule* module, HloComputation* computation,
      const std::unordered_set<const HloInstruction*>& sync_points);

  HloComputation* BuildComputationForRemains(
      HloModule* module, SyncFreeChain& sync_free_chain);

  HloComputation* BuildLocalAccumulationComputation(
      HloModule* module, SyncFreeChain& sync_free_chain);

  void BuildLocalInitComputation(HloModule* module, SyncFreeChain& sync_free_chain);

  std::unique_ptr<CommDevManager> comm_dev_mgr_;
  int share_split_ordinal_ = -1;

};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_DECOMPOSITION_H_
