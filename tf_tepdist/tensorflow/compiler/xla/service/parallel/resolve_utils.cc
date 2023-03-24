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

#include "tensorflow/compiler/xla/service/parallel/resolve_utils.h"

#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/service_env.h"

namespace xla {

std::vector<HloInstruction*> ResolveGradients(HloModule* module) {
  std::vector<HloInstruction*> grads;
  auto entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();

  std::string frontend = ServiceEnv::frontend();
  VLOG(2) << "FRONTEND = " << frontend;
  if (frontend.empty() /* default to be JAX */ ||
      frontend == "JAX") {
    int64 num_gradients = ServiceEnv::num_gradients();
    grads.reserve(num_gradients);
    for (int64 i = 0; i < num_gradients; ++i) {
      auto var_view = root->mutable_operand(i);
      auto grad = ResolveJAXGradAdaFactor(var_view);
      grads.emplace_back(grad);
    }
    return grads;
  } else if (frontend == "TF-1.14") {
    grads = ResolveGradientsForTF114(module);
    return grads;
  }

  CHECK(0 && "TODO: Support Tensorflow or other frontends!"); 
}

HloInstruction* ResolveJAXGradAdaFactor(HloInstruction* var_view) {
  HloInstruction* grad_instr = nullptr;
  std::deque<HloInstruction*> worklist = {var_view};
  std::unordered_set<HloInstruction*> visited;

  while (!worklist.empty()) {
    auto instr = worklist.front();
    worklist.pop_front();
    visited.insert(instr);

    switch (instr->opcode()) {
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
        for (auto operand : instr->operands()) {
          if (operand->opcode() == HloOpcode::kPower ||
              operand->opcode() == HloOpcode::kBroadcast ||
              operand->opcode() == HloOpcode::kParameter) continue;
          if (visited.count(operand)) continue;

          worklist.push_back(operand);
        }
        break;

      case HloOpcode::kDot:
      case HloOpcode::kReduce:
      case HloOpcode::kScatter:
        CHECK(!grad_instr);
        grad_instr = instr;
        break;
      default: VLOG(0) << instr->ToString(); CHECK(0);
    }
  }
  return grad_instr;
}

std::vector<HloInstruction*> ResolveGradientsForTF114(HloModule* module) {
  auto entry = module->entry_computation();
  std::map<int64, HloInstruction*, std::less<int64>> sorted_grads;
  std::vector<HloInstruction*> grads;
  // TODO: support more optimizers when needed.
  // Currently supported optimziers are: Adam, Momentum, GradientDescent.

  // Filter gradients for Adam optimizer.
  //
  // Find the *only* consumer of each grad for *Adam* optimizer according
  // to the implementation of kernel `ResourceApplyAdam` which can be found in
  // `compiler/tf2xla/kernels/training_ops.cc` of TF-1.14.
  for (auto hlo : entry->instructions()) {
    if (hlo->metadata().op_type() == "ResourceApplyAdam" &&
        hlo->opcode() == HloOpcode::kMultiply) {
      auto lhs = hlo->operand(0);
      auto rhs = hlo->operand(1);
      if (lhs == rhs) {
        VLOG(0) << "grad op->" << lhs->ToString();
        CHECK(lhs->metadata().op_type() != "ResourceApplyAdam");
        grads.emplace_back(const_cast<HloInstruction*>(lhs));
      }
    }
  }

  // Filter gradients for AdamWeightDecay optimizer.
  //
  // Find the *only* consumer of each grad for *AdamWeightDecay* optimizer
  // according to the implementation of kernel `ResourceApplyAdamWeightDecay`
  // which can be found in `compiler/tf2xla/kernels/training_ops.cc` of TF-1.14.
  for (auto hlo : entry->instructions()) {
    auto optimizer = "ResourceApplyAdamWeightDecay";
    if (hlo->metadata().op_type() == optimizer &&
        hlo->opcode() == HloOpcode::kSubtract) {
      auto lhs = hlo->operand(0);
      auto rhs = hlo->operand(1);
      if (lhs->opcode() == HloOpcode::kMultiply &&
          rhs->opcode() == HloOpcode::kParameter) {
        if (lhs->metadata().op_type() == optimizer &&
            lhs->operand(0) == lhs->operand(1)) {
          VLOG(0) << "Grad instruction:" << lhs->operand(0)->ToString();
          HloInstruction* grad = const_cast<HloInstruction*>(rhs);
          grads.emplace_back(grad);
        }
      }
    }
  }

  // Filter gradients for Momentum optimizer.
  //
  // Find the *only* consumer of each grad for *Momentum* optimizer according
  // to the implementation of kernel `ResourceApplyMomentum` which can be found
  // in `compiler/tf2xla/kernels/training_ops.cc` of TF-1.14.
  for (auto hlo : entry->instructions()) {
    auto optimizer = "ResourceApplyMomentum";
    if (hlo->metadata().op_type() == optimizer &&
        hlo->opcode() == HloOpcode::kAdd) {
      auto lhs = hlo->operand(0);
      auto rhs = hlo->operand(1);
      if (lhs->opcode() == HloOpcode::kMultiply &&
          lhs->metadata().op_type() == optimizer &&
          rhs->metadata().op_type() != optimizer) {
        VLOG(2) << "Grad instruction:" << rhs->ToString();
        HloInstruction* grad = const_cast<HloInstruction*>(rhs);
        grads.emplace_back(grad);
      }
    }
  }

  // Filter gradients for GradientDescent optimizer.
  //
  // Find the *only* consumer of each grad for *GradientDescent* optimizer
  // according to the implementation of kernel `ResourceApplyGradientDescent`
  // which can be found in `compiler/tf2xla/kernels/training_ops.cc` of TF-1.14.
  for (auto hlo : entry->instructions()) {
    auto optimizer = "ResourceApplyGradientDescent";
    if (hlo->metadata().op_type() == optimizer &&
        hlo->operand_count() > 1 &&
        hlo->operand(0)->opcode() == HloOpcode::kParameter) {
      auto hlo_rhs = hlo->operand(1);
      auto param = hlo->operand(0);
      auto arg_no = DynCast<HloParameterInstruction>(param)->parameter_number();
      HloInstruction* grad = nullptr;
      if (hlo_rhs->opcode() == HloOpcode::kMultiply) {
        auto mul_rhs = hlo_rhs->operand(1);
        CHECK(mul_rhs->metadata().op_type() != optimizer);
        VLOG(2) << "Grad instruction:" << mul_rhs->ToString();
        grad = const_cast<HloInstruction*>(mul_rhs);
      } else {
        // AlgebraicSimplifier case
        grad = const_cast<HloInstruction*>(hlo_rhs);
      }
      sorted_grads[arg_no] = grad;
    }
  }

  for (auto& it : sorted_grads) {
    grads.emplace_back(it.second);
  }

  VLOG(0) << "# of grads instructions: " << grads.size();
  GradInstsSizeCheck(module, grads);
  return grads;
}

// Make sure that we have resolved the right number of gradient ops.
void GradInstsSizeCheck(
    HloModule* module, std::vector<HloInstruction*>& grads) {
  auto* variable_map = module->variable_map();
  CHECK(variable_map->size());
  // variable_count represents the # of trainable variables.
  int expected_var_count = 0;
  for (auto iter : *variable_map) {
    auto var_name = iter.second;
    if (tensorflow::str_util::EndsWith(var_name, "beta1_power") ||
        tensorflow::str_util::EndsWith(var_name, "beta2_power") ||
        tensorflow::str_util::EndsWith(var_name, "global_step") ||
        tensorflow::str_util::EndsWith(var_name, "Adam") ||
        tensorflow::str_util::EndsWith(var_name, "Adam_1") ||
        tensorflow::str_util::EndsWith(var_name, "AdamW") ||
        tensorflow::str_util::EndsWith(var_name, "AdamW_1") ||
        tensorflow::str_util::EndsWith(var_name, "AdamWeightDecayOptimizer") ||
        tensorflow::str_util::EndsWith(var_name, "AdamWeightDecayOptimizer_1") ||
        tensorflow::str_util::EndsWith(var_name, "Momentum")) {
      continue;
    }
    expected_var_count ++;
  }
  CHECK_EQ(grads.size(), expected_var_count)
      << "The number of grad ops resolved by `ResolveGradientsForTF114`"
      << " does not meet expectations.";
}

// Filter gradients for GradientDescent optimizer.
//
// Find the *only* consumer of each grad for *GradientDescent* optimizer
// according to the implementation of kernel `ResourceApplyGradientDescent`
// which can be found in `compiler/tf2xla/kernels/training_ops.cc` of TF-1.14.
std::vector<HloInstruction*> GradientDescentOptimizerGradsResolver(
    HloComputation* entry) {
  auto optimizer = "ResourceApplyGradientDescent";
  std::map<int64, HloInstruction*, std::less<int64>> sorted_grads;
  std::vector<HloInstruction*> grads;
  for (auto hlo : entry->instructions()) {
    if (hlo->metadata().op_type() == optimizer &&
        hlo->operand_count() > 1 &&
        hlo->operand(0)->opcode() == HloOpcode::kParameter) {
      auto hlo_rhs = hlo->operand(1);
      auto param = hlo->operand(0);
      auto arg_no = DynCast<HloParameterInstruction>(param)->parameter_number();
      HloInstruction* grad = nullptr;
      if (hlo_rhs->opcode() == HloOpcode::kMultiply) {
        auto mul_rhs = hlo_rhs->operand(1);
        CHECK(mul_rhs->metadata().op_type() != optimizer);
        VLOG(2) << "Grad instruction:" << mul_rhs->ToString();
        grad = const_cast<HloInstruction*>(mul_rhs);
      } else {
        // AlgebraicSimplifier case
        grad = const_cast<HloInstruction*>(hlo_rhs);
      }
      sorted_grads[arg_no] = grad;
    }
  }

  for (auto& it : sorted_grads) {
    grads.emplace_back(it.second);
  }
  VLOG(2) << "# of grads instructions: " << grads.size();
  return grads;
}

std::vector<HloInstruction*> ResolveComputeOutputs(
    HloComputation* entry, std::map<int, std::string>& var_map) {
  std::vector<HloInstruction*> res;
  auto root = entry->root_instruction();
  for (auto operand : root->operands()) {
    if (operand->metadata().op_name().find("XLA_Retvals") != string::npos &&
        operand->shape().element_type() == F32) {
      VLOG(0) << "loss op: " << operand->ToString();
      res.emplace_back(operand);
    }
  }
  // Check if there exists corner case.
  CHECK_EQ(res.size(), 1);
  return res;
}

std::unordered_set<HloInstruction*> ResolveApplyGradientsInstructions(
    HloComputation* entry,
    std::vector<HloInstruction*>& compute_outputs,
    std::vector<HloInstruction*>& grads) {
  std::unordered_set<HloInstruction*> apply_grads_instrs;
  auto entry_root = entry->root_instruction();
  auto& root_operands = entry_root->operands();

  std::deque<HloInstruction*> worklist(
      root_operands.begin() + compute_outputs.size(),
      root_operands.end());
  std::unordered_set<HloInstruction*> visited, params, body;
  std::unordered_set<HloInstruction*> grads_set(grads.begin(), grads.end());

  while (!worklist.empty()) {
    auto instr = worklist.front();
    worklist.pop_front();
    visited.insert(instr);

    if (instr->opcode() == HloOpcode::kParameter ||
        grads_set.count(instr)) {

      continue;
    } else {
      apply_grads_instrs.insert(instr);
    }

    for (auto operand : instr->operands()) {
      if (visited.count(operand)) continue;

      worklist.push_back(operand);
    }
  }

  return apply_grads_instrs;
}

void ResolveForwardBackwardAndApplyGradients(
    HloComputation* entry, std::vector<HloInstruction*>& compute_outputs,
    std::vector<HloInstruction*>& grads,
    std::unordered_set<HloInstruction*>* forward,
    std::unordered_set<HloInstruction*>* backward,
    std::unordered_set<HloInstruction*>* apply_gradients) {
  auto reachability = HloReachabilityMap::Build(entry);
  auto root = entry->root_instruction();

  // 1. Find forward    
  for (auto instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kParameter) continue;

    for (auto* output : compute_outputs) {
      if (reachability->IsReachable(instr, output)) {
        forward->insert(instr);
      }
    }
  }

  // 2. Find backward
  std::deque<HloInstruction*> worklist(grads.begin(), grads.end());
  std::unordered_set<HloInstruction*> visited(grads.begin(), grads.end());
  while (!worklist.empty()) {
    auto instr = worklist.front();
    worklist.pop_front();
    if (!forward->count(instr)) backward->insert(instr);

    for (auto* op : instr->operands()) {
      if (!visited.count(op) || op->opcode() == HloOpcode::kParameter) {
        worklist.push_back(op);
        visited.insert(op);
      }
    }
  }

  // 3. Find apply gradients
  for (auto instr : entry->instructions()) {
    if (forward->count(instr) || backward->count(instr) ||
        instr->opcode() == HloOpcode::kParameter ||
        instr == entry->root_instruction()) continue;
    apply_gradients->insert(instr);
  }

  return;
}

} // namespace xla
