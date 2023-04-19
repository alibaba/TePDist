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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_COMPUTATION_CLONE_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_COMPUTATION_CLONE_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace xla {

struct CloneSignature {
  std::string name;
  std::vector<const HloInstruction*> params;
  std::unordered_set<const HloInstruction*> body;
  std::vector<const HloInstruction*> root_operands;
  HloComputation* orig_computation;
};

void InferDefContextMembers(
    CloneSignature& signature, HloModule::DefContext* parent_def_ctx,
    HloModule::DefContext* def_ctx);

HloComputation* CloneComputation(CloneSignature& signature);

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_COMPUTATION_CLONE_UTILS_H_
