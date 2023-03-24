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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_RESOLVE_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_RESOLVE_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include <map>
#include <vector>

namespace xla {

// Maybe thess functions can be reused by SpmdTransform.

// Resolve gradients for the given computation
// Currrently supported Optimzier:
//  - GradientDescentOptimizer
//  - AdamWeightDecayOptimizer
std::vector<HloInstruction*> ResolveGradients(HloModule* module);
std::vector<HloInstruction*> ResolveGradientsForTF114(HloModule* module);
void GradInstsSizeCheck(HloModule* module, std::vector<HloInstruction*>& grads);
HloInstruction* ResolveJAXGradAdaFactor(HloInstruction* var_view);
std::vector<HloInstruction*> GradientDescentOptimizerGradsResolver(
    HloComputation* entry);
// Resolve compute outputs for the given computation
std::vector<HloInstruction*> ResolveComputeOutputs(
    HloComputation* entry, std::map<int, std::string>& var_map);
// Resolve forward, backward and apply_gradients
void ResolveForwardBackwardAndApplyGradients(
    HloComputation* entry, std::vector<HloInstruction*>& compute_outputs,
    std::vector<HloInstruction*>& grads,
    std::unordered_set<HloInstruction*>* forward,
    std::unordered_set<HloInstruction*>* backward,
    std::unordered_set<HloInstruction*>* apply_gradients);

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_RESOLVE_UTILS_H_
