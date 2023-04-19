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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SPMD_TRANSFORM_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SPMD_TRANSFORM_H_

#include <set>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/parallel/hlo_strategy_spec.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"
#include "tensorflow/core/platform/macros.h"
#include "absl/container/flat_hash_map.h"

namespace xla {

// Necessary data structures for creating custom-collectives
struct CustomCollectiveSpec {
  static CustomCollectiveSpec CreateAllReduceSpec(
      HloInstruction* instr, string reduction_type, int num_replicas);

  static CustomCollectiveSpec CreateReshardSpec(
      HloInstruction* instr, int operand_idx, HloInstruction* op,
      const DimStrategy& from_strategy, const DimStrategy& to_strategy, int num_replicas);

  HloInstruction* instr_;
  string custom_type_;
  int operand_idx_;
  gpu::AllReduceBackendConfig allreduce_config_;
  gpu::ReshardBackendConfig reshard_config_;

  int64 cost_;
};

class SpmdTransform {
 public:
  explicit SpmdTransform() {}
  
  explicit SpmdTransform(int split_ordinal) 
      : split_ordinal_(split_ordinal) {}

  ~SpmdTransform() = default;
 
  StatusOr<bool> Run(HloModule* module);

 private:
  bool ShapeCheck(HloComputation* entry);

  void NewTuple(HloInstruction* instr, HloComputation* entry);
  HloInstruction* NewParameter(HloInstruction* instr, 
                    Shape& new_shape, HloComputation* entry,
                    int dim_to_slice);
  HloInstruction* NewDot(HloInstruction* instr, 
              Shape& new_shape, HloComputation* entry,
              int dim_to_slice);
  HloInstruction* NewConvolution(HloInstruction* instr, 
              Shape& new_shape, HloComputation* entry,
              int dim_to_slice);
  HloInstruction* NewUnary(HloInstruction* instr, 
                Shape& new_shape, HloComputation* entry,
                int dim_to_slice);
  HloInstruction* NewBinary(HloInstruction* instr, 
                 Shape& new_shape, HloComputation* entry,
                 int dim_to_slice);
  HloInstruction* NewRng(HloInstruction* instr, 
              Shape& new_shape, HloComputation* entry,
              int dim_to_slice);
  HloInstruction* NewCrossReplicaSum(HloInstruction* instr, 
                          Shape& new_shape, HloComputation* entry,
                          int dim_to_slice);
  HloInstruction* NewCustomCollective(HloInstruction* instr, 
                          Shape& new_shape, HloComputation* entry,
                          int dim_to_slice);
  HloInstruction* NewCompare(HloInstruction* instr, 
                  Shape& new_shape, HloComputation* entry,
                  int dim_to_slice);
  HloInstruction* NewReduce(HloInstruction* instr, 
                 Shape& new_shape, HloComputation* entry,
                 int64 dim_to_slice);
  HloInstruction* NewReverse(HloInstruction* instr, 
                 Shape& new_shape, HloComputation* entry,
                 int64 dim_to_slice);
  HloInstruction* NewTranspose(HloInstruction* instr, 
                 Shape& new_shape, HloComputation* entry,
                 int dim_to_slice);
  HloInstruction* NewScatter(HloInstruction* instr, 
                  Shape& new_shape, HloComputation* entry,
                  int dim_to_slice);
  HloInstruction* NewConvert(HloInstruction* instr, 
                 Shape& new_shape, HloComputation* entry,
                 int dim_to_slice);
  HloInstruction* NewBitcastConvert(HloInstruction* instr, 
                 Shape& new_shape, HloComputation* entry,
                 int dim_to_slice);
  HloInstruction* NewTenary(HloInstruction* instr, 
                 Shape& new_shape, HloComputation* entry,
                 int dim_to_slice);
  HloInstruction* NewConcat(HloInstruction* instr, 
                 Shape& new_shape, HloComputation* entry,
                 int dim_to_slice);
  HloInstruction* NewReshape(HloInstruction* instr, 
                  Shape& new_shape, HloComputation* entry,
                  int dim_to_slice);
  HloInstruction* NewBroadcast(HloInstruction* instr, 
                  Shape& new_shape, HloComputation* entry,
                  int dim_to_slice);
  HloInstruction* NewIota(HloInstruction* instr, 
               Shape& new_shape, HloComputation* entry,
               int dim_to_slice);
  HloInstruction* NewSlice(HloInstruction* instr, 
                Shape& new_shape, HloComputation* entry,
                int64 slice_dim);
  HloInstruction* NewGather(HloInstruction* instr,
                 Shape& new_shape, HloComputation* entry,
                 int64 slice_dim);
  HloInstruction* NewPad(HloInstruction* instr, 
              Shape& new_shape, HloComputation* entry,
              int dim_to_slice);
  HloInstruction* NewCustomCall(HloInstruction* instr,
                     Shape& new_shape, HloComputation* entry,
                     int dim_to_slice);
  HloInstruction* NewReduceWindow(HloInstruction* instr,
                       Shape& new_shape, HloComputation* entry,
                       int dim_to_slice);
  HloInstruction* NewSelectAndScatter(HloInstruction* instr,
                           Shape& new_shape, HloComputation* entry,
                           int dim_to_slice);
  HloInstruction* NewGTE(HloInstruction* instr,
              Shape& new_shape, HloComputation* entry,
              int dim_to_slice);
  HloInstruction* NewSort(HloInstruction* instr, 
                Shape& new_shape, HloComputation* entry,
                int dim_to_slice);
  HloInstruction* NewDynamicSlice(HloInstruction* instr, 
                Shape& new_shape, HloComputation* entry,
                int dim_to_slice, int64 num_replicas);

  int64 LSTransform(HloComputation* computation);
  void CreateNewInstruction(HloInstruction* instr, HloComputation* entry);
  void AllReducePatternOptimization(HloComputation* computation);
  bool CreateAllCustomCollectives(HloComputation* computation);

  // Create data structures needed for creating all custom-collectives
  HloInstruction* CreateCustomCollective(HloComputation* computation, CustomCollectiveSpec& spec);
  // Initialize HLOStrategy for each instruction
  void InitializeDimStrategyIntoMap(HloComputation* computation);
  void InitializeInferedStrategyIntoMap(HloComputation* computation);
  bool DoTransform(HloModule* module);
  void UpdateSplitInfo(HloModule* module);

  TF_DISALLOW_COPY_AND_ASSIGN(SpmdTransform);

 protected:
  HloMutableInstMap<DimStrategy/*split strategy*/> instr_strategy_map_;
  HloMutableInstMap<std::vector<DimStrategy/*infered split strategy*/>> operand_inferred_strategy_map_;
  std::unordered_map<HloInstruction*, std::unordered_map<string, HloInstruction*> > reshard_map_;
  int split_ordinal_ = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SPMD_TRANSFORM_H_
