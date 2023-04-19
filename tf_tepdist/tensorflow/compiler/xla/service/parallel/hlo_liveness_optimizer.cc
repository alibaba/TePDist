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

#include "tensorflow/compiler/xla/service/parallel/hlo_liveness_optimizer.h"

#include <unordered_set>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

StatusOr<bool> HloLivenessOptimizer::Run(HloModule* module) {
  bool changed = false;
  HloComputation* entry = module->entry_computation();
  std::unordered_set<HloInstruction*> arg_users;
  // Only kConvert instructions which consumes kParameter instruction
  // will be processed in current implementation.
  for (HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() != HloOpcode::kConvert) continue;
    if (instr->operand(0)->opcode() == HloOpcode::kParameter &&
        !ShapeUtil::IsScalar(instr->shape())) {
      arg_users.insert(instr);
    }
  }

  for (HloInstruction* instr : arg_users) {
    if (instr->user_count() <= 1) continue;
    CHECK(instr->operand(0)->opcode() == HloOpcode::kParameter);
    for (auto* user : instr->users()) {
      auto new_instr = entry->AddInstruction(
          HloInstruction::CreateConvert(instr->shape(), instr->mutable_operand(0)));
      new_instr->set_allow_eliminate(false);
      new_instr->set_metadata(user->metadata());
      auto op_idx = user->operand_index(instr);
      TF_CHECK_OK(user->ReplaceOperandWith(op_idx, new_instr));
      changed = true;
    }
    TF_CHECK_OK(entry->RemoveInstruction(instr));
  }
  return changed;
}

} // namespace xla