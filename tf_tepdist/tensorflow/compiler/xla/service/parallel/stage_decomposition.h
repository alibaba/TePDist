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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_STAGE_DECOMPOSITION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_STAGE_DECOMPOSITION_H_

#include "tensorflow/compiler/xla/pjrt/dev_id_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/parallel/computation_clone_utils.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Stores the output mapping from current DefContext to its child DefContext's
typedef struct {
  int def_id;
  int out_idx;
} DefInfo;

typedef struct {
  int produced_stage = -1;
  std::map<int/*stage*/, int/*arg_no*/, std::less<int>> consumed_stage_arg_pair;
} CrossStageInfo;

class StageDecomposition {
 public:
  explicit StageDecomposition()
    : main_computation_(nullptr), local_accum_(nullptr), non_sync_free_(nullptr),
    stage_count_(0) {}
  StatusOr<bool> Run(HloModule* module);

 private:
  bool ResolveComputations(HloModule* module);
  void ResolveTotalStageCount();

  // Build CloneSignature function. This function back traverses from the
  // given root operands to its input direction within parent_computation.
  CloneSignature BuildSignatureForStage(
      std::vector<const HloInstruction*>& root_ops,
      std::unordered_set<const HloInstruction*>* excluded,
      HloComputation* parent_computation,
      std::string& stage_name);

  // Build CloneSignature for main_computation
  CloneSignature BuildStageSignatureForMainComputation(
      int stage_id, std::unordered_set<const HloInstruction*>* prev_stages_insts,
      HloComputation* parent_computation, std::string parent_name);

  // Infer and inherit DefContext members from its parent's
  void SetupStageDefContext(
      HloModule* module, CloneSignature& signature,
      HloModule::DefContext* parent_def_ctx, int stage_id);

  // Decompose main computation into stages
  void BuildStagesForMainComputation(HloModule* module);
  // Decompose local_accumulation computation into stages
  void BuildStagesForLocalAccumulation(HloModule* module);
  // Decompose non_sync_free computation into stages
  void BuildStagesForNonSyncFree(HloModule* module);
  // Decompose computation into stages with respect to the parent_def_ctx
  void BuildStagesForComputation(
      HloModule* module, HloModule::DefContext* parent_def_ctx);
  std::vector<const HloInstruction*> ResolveSortedRootOpsFromInsts(
      const HloInstruction* root, std::vector<const HloInstruction*>& insts);

  void BuildStagesForLocalInit(HloModule* module);
  void InferDefMapFromParent(
      std::vector<const HloInstruction*>& stage_params,
      HloModule::DefContext* stage_def_ctx, HloModule::DefContext* parent_def_ctx,
      int phy_s, HloModule* module);
  void InferDefMapForStage(
      std::vector<const HloInstruction*>& stage_params,
      std::unordered_map<const HloInstruction*, DefInfo>& insts_def_info,
      HloModule::DefContext* parent_def_ctx, int s, HloModule* module);

  HloModule::DefContext::DefType GetLowerLevelType(HloModule::DefContext* def_ctx);

  void InitializeDefMap(HloModule* module);

  void PrintCommunicationInfo(HloModule* module);

  std::map<HloInstruction*, CrossStageInfo> CollectCrossStageInsts(
      HloModule* module, std::unordered_map<int/*def_id*/, int/*stage_id*/>& def_stage_map);
  void CrossStageTransferOptimization(HloModule* module);

  HloComputation* main_computation_;
  HloComputation* local_accum_;
  HloComputation* non_sync_free_;
  std::unordered_map<int, HloModule::DefContext*> id_def_map_;
  std::unordered_map<int/*def_id*/,
                     std::vector<DefInfo>/*all outputs info*/> def_output_info_map_;
  
  std::unique_ptr<CommDevManager> comm_dev_mgr_;
  int stage_count_;
  int split_ordinal_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_STAGE_DECOMPOSITION_H_
