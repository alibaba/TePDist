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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_FAST_SPMD_STRATEGY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_FAST_SPMD_STRATEGY_H_

#include <set>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"
#include "tensorflow/compiler/xla/service/parallel/hlo_strategy_spec.h"
#include "tensorflow/compiler/xla/service/parallel/par_type.h"
#include "tensorflow/core/platform/macros.h"

#include "absl/container/flat_hash_map.h"

namespace xla {

class FastSpmdStrategyBase {
 public:
  struct ReshardEdge {
    explicit ReshardEdge(
        HloInstruction* src, HloInstruction* dst, DimStrategy& dst_inferred_strategy)
        : src_(src), dst_(dst), dst_inferred_strategy_(dst_inferred_strategy) {}
    HloInstruction* src_;
    HloInstruction* dst_;
    DimStrategy dst_inferred_strategy_;
  };
  struct Solution {
    std::vector<ReshardEdge> reshard_edges_;
    HloInstMap<DimStrategy> hlo_strategy_map_;
  };

 public:
  explicit FastSpmdStrategyBase();
  explicit FastSpmdStrategyBase(/*ParallelType ttype,*/ bool phase);
  explicit FastSpmdStrategyBase(/*ParallelType ttype,*/ bool phase, int num_replicas);
  virtual ~FastSpmdStrategyBase() = default;

  void ResolveProposalIndices(int64 idx,
    std::vector<int64>& proposal_sizes, std::vector<int64>& proposal_indices);

  bool JAXSetupVariableMap(HloModule* module);

  bool BackInferFilter(HloInstruction* hlo, Solution* solution);

  bool BackInferCustomCall(HloInstruction* hlo, Solution* solution);

  bool BackInferConv(HloInstruction* hlo, Solution* solution);

  bool BackInferGetTupleElement(HloInstruction* hlo, Solution* solution);

  bool BackInferReduceWindowAndSelectScatter(HloInstruction* hlo, Solution* solution);

  bool BackInferTranspose(HloInstruction* hlo, Solution* solution);

  bool BackInferGather(HloInstruction* hlo, Solution* solution);

  bool BackInferReduce(HloInstruction* hlo, Solution* solution);

  bool BackInferDot(HloInstruction* hlo, Solution* solution);

  bool BackInferReshape(HloInstruction* hlo, Solution* solution);

  bool BackInferSlice(HloInstruction* hlo, Solution* solution);

  bool BackInferConcat(HloInstruction* hlo, Solution* solution);

  bool BackInferIota(HloInstruction* hlo, Solution* solution);

  bool BackInferSelect(HloInstruction* hlo, Solution* solution);

  bool BackInferScatter(HloInstruction* hlo, Solution* solution);

  bool BackInferCrossReplicaSum(HloInstruction* hlo, Solution* solution);

  bool BackInferUnary(HloInstruction* hlo, Solution* solution);

  bool BackInferBinary(HloInstruction* hlo, Solution* solution);

  bool BackInferBcast(HloInstruction* hlo, Solution* solution);

  bool BackInferPad(HloInstruction* hlo, Solution* solution);

  bool BackInference(HloInstruction* hlo, Solution* solution);

  bool ReduceWindowAndSelectScatterInference(
      HloInstruction* hlo, Solution* solution);

  bool GetTupleElementInference(HloInstruction* hlo, Solution* solution);

  bool CustomCallInference(HloInstruction* hlo, Solution* solution);

  bool ConvInference(HloInstruction* hlo, Solution* solution);

  bool TransposeInference(HloInstruction* hlo, Solution* solution);

  bool DotInference(HloInstruction* hlo, Solution* solution);

  bool BitcastConvertInference(HloInstruction* hlo, Solution* solution);

  bool ReshapeInference(HloInstruction* hlo, Solution* solution);

  bool IotaInference(HloInstruction* hlo, Solution* solution);

  bool ShiftInference(HloInstruction* hlo, Solution* solution);

  bool ConcatInference(HloInstruction* hlo, Solution* solution);

  bool SliceInference(HloInstruction* hlo, Solution* solution);

  bool PadInference(HloInstruction* hlo, Solution* solution);

  bool ReduceInference(HloInstruction* hlo, Solution* solution);

  bool CrossReplicaSumInference(HloInstruction* hlo, Solution* solution);

  bool GatherInference(HloInstruction* hlo, Solution* solution);

  bool UnaryInference(HloInstruction* hlo, Solution* solution);

  bool ScatterInference(HloInstruction* hlo, Solution* solution);

  bool SelectInference(HloInstruction* hlo, Solution* solution);

  bool BinInference(HloInstruction* hlo, Solution* solution);

  bool BcastInference(HloInstruction* hlo, Solution* solution);

  bool ConstantInference(HloInstruction* hlo, Solution* solution);

  bool StrategyInference(HloInstruction* instr, Solution* solution);

  bool CheckAndBackInfer(HloComputation* entry,
                         Solution* solution,
                         const std::unordered_set<HloInstruction*> worklist,
                         bool* conflict);
  bool CheckAndInfer(HloComputation* entry,
                     Solution* solution,
                     bool* conflict);
  bool ConstantSliced(HloComputation* entry,
                      Solution* solution);
  bool IncrementalSatisfiable(HloInstruction* seed,
      HloComputation* entry, Solution* solution);
  bool StrategySatisfiable(HloComputation* entry,
                           Solution* solution,
                           const std::unordered_set<HloInstruction*> worklist);
  virtual bool StrategyPlanning(HloModule* module,
                        bool extend_sharding_param, int top_k = 0) = 0;
  virtual StatusOr<bool> Run(HloModule* module);
  StatusOr<bool> InferenceCalibration(HloModule* module);

  Status InferDerivationCache(HloModule* module);

 private:
  // Infer derivation table for fusion function. 
  Status InferFusionDistAttrs(HloInstruction* inst);


  // Based on the deterministic mapping relationship between forward
  // instructions and backward instructions in `metadata.op_name`, for these
  // forward insts with user annotation (through `xla_sharding.split(...)`), we
  // collect the corresponding backward instructions to be shareded in later
  // strategy propagation.
  // These collected instructions are critical as they can usually control the
  // DP and MP cut planes in the backward graph.
  void CollectBwdKeyShardingInsts(HloModule* module);

  bool post_layout_assignment_;
 protected:
  int64 num_replicas_ = 1;

  HloInstSet backward_insts_;
  // The instruction set in the backward graph corresponding to the instructions
  // marked sharded by the user annotation (`xla_sharding.split()`) in the
  // forward graph.
  HloMutableInstSet bwd_sharding_annotation_insts_;

  TF_DISALLOW_COPY_AND_ASSIGN(FastSpmdStrategyBase);
};


class FastSpmdStrategy : public FastSpmdStrategyBase {
 public:
  explicit FastSpmdStrategy();
  explicit FastSpmdStrategy(ParallelType ttype, bool phase);
  explicit FastSpmdStrategy(ParallelType ttype, bool phase, int num_replicas);
  virtual ~FastSpmdStrategy() = default;

 private:
  virtual bool StrategyPlanning(HloModule* module,
                        bool extend_sharding_param, int top_k = 0);
  void ResolveParamList(HloModule* module,
                        std::vector<HloInstruction*>& param_list);
  void ResolveParamList(HloModule* module,
                        std::vector<HloInstruction*>& raw_worklist,
                        std::map<int, HloMutableInstSet>& span_hlo,
                        int window_size);
  void ResolveParamDPList(HloModule* module,
                       std::vector<HloInstruction*>& param_list);
  void ResolveParamShardingList(HloModule* module,
                      std::map<int, HloMutableInstSet>& span_hlo);
  ParallelType ttype_;
};

class AnnotFastSpmdStrategy : public FastSpmdStrategyBase {
 public:
  explicit AnnotFastSpmdStrategy() {};
  explicit AnnotFastSpmdStrategy(/*ParallelType ttype,*/ bool phase);
  explicit AnnotFastSpmdStrategy(/*ParallelType ttype,*/ bool phase, int num_replicas);
  virtual ~AnnotFastSpmdStrategy() = default;

  bool Propagate(
     const HloInstSet& start_insts,
     HloInstMap<SharedDimStrategy>& strategy_map,
     bool disable_gradient_inst);

  // only infer direct users of user annotated tensors
  HloInstSet InitializeInference(
    HloInstMap<SharedDimStrategy>& user_annotated_tensors,
    HloInstMap<SharedDimStrategy>& strategy_map);


  // recursively infer
  HloInstSet ForwardPropagate(
      const HloInstSet& start_insts,
      HloInstMap<SharedDimStrategy>& strategy_map);

  // only infer direct users
  virtual HloInstSet InferUsersFromInst(
          const HloInstruction* inst,
          const DimStrategy& inst_strategy,
          HloInstMap<SharedDimStrategy>& strategy_map);

  // recursively infer
  HloInstSet InferFromInst(
          const HloInstruction* inst,
          HloInstMap<SharedDimStrategy>& strategy_map);

  HloInstSet BackwardPropagate(
      const HloInstSet& start_insts,
      HloInstMap<SharedDimStrategy>& strategy_map,
      bool disable_gradient_inst);

  HloInstSet InferFromUser(const HloInstruction* inst,
                           HloInstMap<SharedDimStrategy>& strategy_map,
                           bool disable_gradient_inst);

  virtual bool StrategyPlanning(HloModule* module,
                        bool extend_sharding_param, int top_k = 0);

  // Update bwd_sharding_annotation_inst with meaningful sharding
  // strategy inferred from operand.
  // Then supplement the last strategy propagation in order to avoid the
  // strategy miss of some instructions who's propagation was previously blocked
  // by instrucitons from bwd_sharding_annotation_inst.
  void UpdateBwdShardingAnnotationInstsStrategy(
    HloInstMap<SharedDimStrategy>& best_strategy_map);

  void FillStrategyForAllInstructions(
    const HloModule* module,
    HloInstMap<SharedDimStrategy>* best_strategy_map);

  void ResetUnDivisibleStrategy(
      HloInstMap<SharedDimStrategy>* best_strategy_map,
      int& glued_inst_num);

  void BestStrategyPostProcess(
      HloInstMap<SharedDimStrategy>* best_strategy_map,
      int& glued_inst_num);

  void RecordStrategyToInsts(HloModule* module,
      const HloInstMap<SharedDimStrategy>& strategy_map);
  void DumpStrategies(const HloModule* module,
      const HloInstMap<SharedDimStrategy>& strategy_map);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AnnotFastSpmdStrategy);
};


}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_FAST_SPMD_STRATEGY_H_
