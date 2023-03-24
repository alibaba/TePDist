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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_UTILS_H_

#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/parallel/hlo_strategy_spec.h"

namespace xla {

class HloModule;

void RewriteCustomCallsAsDots(HloModule* module);
bool is_compute_intensive(const HloInstruction* inst);

HloInstSet FindBackwardInsts(const HloModule* module, bool prefer_backward);
HloInstSet FindForwardInsts(const HloModule* module, bool prefer_backward);

std::vector<const HloInstruction*>
StableSort(const HloInstSet& insts);

bool FromZeroConstant(HloInstruction* instr);

class StrategyUtil {
 public:
  static SharedDimStrategy InferUnary(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferBcast(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferBinary(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferTernary(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferReshape(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferConcat(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferSlice(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static HloInstMap<SharedDimStrategy>
  InferDot(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static HloInstMap<SharedDimStrategy>
  InferConvolution(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferReduce(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferReverse(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferGather(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferPad(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferTranspose(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferCustomCall(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferCustomCollective(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferGetTupleElement(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferSelectAndScatter(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferReduceWindow(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static HloInstMap<SharedDimStrategy>
  InferScatter(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static SharedDimStrategy InferSort(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static HloInstMap<SharedDimStrategy>
  ForwardInfer(
      HloInstruction* inst,
      const DimStrategy& input_strategy,
      int input_idx);

  static std::vector<SharedDimStrategy> GenSplitProposals(
                              const HloInstruction* inst,
                              int split_count,
                              bool save_variable_mem);

  static std::vector<SharedDimStrategy> GenDotProposals(
                              const HloInstruction* inst,
                              int split_count,
                              bool save_variable_mem);

  static std::vector<SharedDimStrategy> GenConvProposals(
                              const HloInstruction* inst,
                              int split_count,
                              bool save_variable_mem);

  static SharedDimStrategy BackInferBcast(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferScatter(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferUnary(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferCrossReplicaSum(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferBinary(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferTernary(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferConcat(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferSlice(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferReshape(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferDot(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferConvolution(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferReduce(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferReverse(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferGather(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferPad(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferTranspose(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferIota(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferCustomCall(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferCustomCollective(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy BackInferGetTupleElement(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);

  static SharedDimStrategy BackInferReduceWindow(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);

  static SharedDimStrategy BackInferSelectAndScatter(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
                              
  static SharedDimStrategy BackInferSort(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
 
  static SharedDimStrategy BackInfer(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static SharedDimStrategy NonVerifyBackInfer(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);

  static bool InferGraph(
      HloComputation* entry,
      HloInstMap<DimStrategy>& strategy_map);

  static bool ForwardInfer(HloComputation* entry,
                    HloInstMap<DimStrategy>& strategy_map,
		            bool* changed);
  static bool BackwardInfer(HloComputation* entry,
                    HloInstMap<DimStrategy>& strategy_map,
		            bool* changed);
  
  static int CountSplitInsts(HloInstMap<DimStrategy>& strategy_map);

  static bool CompatibleWith(
      HloInstMap<DimStrategy>& source_map,
      HloInstMap<DimStrategy>& target_map);

  static std::unordered_set<const HloInstruction*> CollectAllSyncPoints(
      HloModule* module);
 
  // TODO(zycao): delete 'multiple_split_' flag after refactoring infer and
  // transform process.
  static bool multiple_split_;
  static void SetMultipleSplit(bool enable) { multiple_split_ = enable; };

 private:
  static SharedDimStrategy BackInferImpl(
                              const HloInstruction* inst,
                              const DimStrategy& inst_strategy,
                              int input_idx);
  static bool VerifyInferGather(const HloInstruction* inst,
                                const SharedDimStrategy inst_strategy);
  static bool VerifyInferScatter(const HloInstruction* inst,
                                 const SharedDimStrategy inst_strategy);
  static bool VerifyInferConvolution(const HloInstruction* inst,
                                     const SharedDimStrategy inst_strategy);
  static bool VerifyInfer(const HloInstruction* inst,
                          const SharedDimStrategy inst_strategy);
};

class DistUtil {
 public:
  static Shape MakeShape(const Shape& old_shape, std::vector<int64>& dims);
  static Shape MakeNewPrimShape(const Shape& old_shape,
                                const DimDistSpec& dist_spec);
  static Shape MakeNewShape(const Shape& old_shape,
                            const DimDistSpec& dist_spec);
  static DimDistSpec MakeDimDistSpec(const DimStrategy& s);
  static Shape MakeNewShape(const Shape& old_shape,
                            const DimStrategy& s);
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_UTILS_H_
