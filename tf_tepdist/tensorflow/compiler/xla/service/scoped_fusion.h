#include "absl/container/flat_hash_map.h"
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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SCOPED_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SCOPED_FUSION_H_

#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// HLO pass which performs instruction fusion according to high level
// scoping directives.
class ScopedFusion : public HloModulePass {
 public:
  explicit ScopedFusion() {}

  ~ScopedFusion() override = default;
  absl::string_view name() const override { return "scoped_fusion"; }

  // Run instruction fusion on the given computation. Returns whether the
  // computation was changed (instructions were fused).
  StatusOr<bool> Run(HloModule* module) override;

  void FixFalseDeps();
  void ScopeInference();
  void ResolveInstScopes();
  bool StrictScopeInference(HloInstruction* hlo);
  bool ProducerInference(HloInstruction* hlo);
  bool ProducerConsumerInference(HloInstruction* hlo);
  void DoScopeFusion(std::string& scope);
  bool BuildFusionAndGTEs(
      std::string& scope, HloCloneContext *context,
      std::unordered_set<const HloInstruction*>& cloned,
      std::vector<std::unique_ptr<HloInstruction>>* instructions);
  void BuildTopLevelComputation(std::set<std::string>& scopes);
  void LiftScope(std::vector<HloInstruction*>& scope_insts,
                 std::string& scope, std::string scope_type);

 private:
  // Current HloComputation instance the loop fuser is traversing.
  HloComputation* computation_;
  HloModule* module_;
  // Reachability information for the current computation.
  std::unique_ptr<HloReachabilityMap> reachability_;

  HloMutableInstMap<std::string> inst_scope_;
  HloMutableInstMap<HloInstruction*> gte_map_;
  std::unordered_map<std::string,
                     std::vector<HloInstruction*>> fusion_inputs_map_;
  std::unordered_map<std::string,
                     std::vector<HloInstruction*>> fusion_outputs_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(ScopedFusion);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SCOPED_FUSION_H_
