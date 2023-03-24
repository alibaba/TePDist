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

#include "tensorflow/compiler/xla/service/scoped_fusion.h"

#include <algorithm>
#include <list>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

#define BACKWARD_SCOPE(x) absl::StartsWith(x, "grad###")

namespace xla {

namespace {
bool ParseScope(HloInstruction* hlo, std::string& scope) {
  auto op_name = hlo->metadata().op_name();
  auto sep = op_name.find("@^_^@");
  if (sep != string::npos) {
    scope = op_name.substr(0, sep);
    return true;
  }
  return false;
}

bool MatchScope(std::string scope, HloInstruction* hlo) {
  CHECK(!scope.empty());

  std::string _scope;
  ParseScope(hlo, _scope);
  return scope == _scope;
}

inline std::string CanalizeScope(std::string& scope) {
  if (BACKWARD_SCOPE(scope)) {
    return scope.substr(7);
  } else {
    return scope;
  }
}

bool MatchScope(std::string scope, HloInstruction* hlo,
                bool& forward, bool& backward) {
  forward = false;
  backward = false;

  auto op_name = hlo->metadata().op_name();
  auto sep = op_name.find("@^_^@");
  if (sep != string::npos) {
    auto _scope = op_name.substr(0, sep);
    if (!absl::StartsWith(_scope, "grad###")) {
      forward = (_scope == scope);
    } else {
      _scope = _scope.substr(7);
      backward = (_scope == scope);
    }
  }
  CHECK(!forward || !backward);
  return forward | backward;
}

}

void ScopedFusion::LiftScope(std::vector<HloInstruction*>& scope_insts,
                             std::string& scope, std::string scope_type) {
  std::string scope_sig = scope;
  if (scope_type == "bwd") scope_sig = "grad###" + scope;

  std::unordered_set<HloInstruction*> scope_set(scope_insts.begin(),
                                                scope_insts.end());
  std::vector<HloInstruction*> fusion_inputs, fusion_outputs;
  std::unordered_set<HloInstruction*> visited_inputs, visited_outputs;
  for (auto hlo : scope_insts) {
    for (auto operand : hlo->operands()) {
      if (!scope_set.count(operand)) {
        if (!visited_inputs.count(operand)) {
          fusion_inputs.push_back(operand);
          visited_inputs.insert(operand);
        }
      }
    }

    if (visited_outputs.count(hlo)) {
      continue;
    }

    for (auto user : hlo->users()) {
      if (!scope_set.count(user)) {
        fusion_outputs.push_back(hlo);
        visited_outputs.insert(hlo);
        break;
      }
    }
  }
  CHECK(fusion_inputs.size() == visited_inputs.size());
  CHECK(fusion_outputs.size() == visited_outputs.size());
  fusion_inputs_map_[scope_sig] = fusion_inputs;
  fusion_outputs_map_[scope_sig] = fusion_outputs;

  //std::cout << computation_->ToString() << "\n";
  std::cout << "Scope fusion inputs:" << fusion_inputs.size() << "\n";
  for (auto input : fusion_inputs) std::cout << input->ToString() << "\n";
  std::cout << "Scope fusion outputs:" << fusion_outputs.size() << "\n";
  for (auto output : fusion_outputs) std::cout << output->ToString() << "\n";

  // Task Extraction
  std::unique_ptr<HloCloneContext> context_ptr;
  const string suffix = "SubModule." + scope + "." + scope_type;
  context_ptr = absl::make_unique<HloCloneContext>(
      computation_->parent(), suffix);
  auto context = context_ptr.get();
  std::vector<std::unique_ptr<HloInstruction>> instructions;
  std::unique_ptr<HloInstruction> new_instr;

  // Create Parameters
  int new_param_id = 0;
  for (auto instr : fusion_inputs) {
    if (instr->opcode() == HloOpcode::kConstant) {
      // Do not need a kParameter in this case.
      // CHECK(ShapeUtil::IsScalar(instr->shape()));
      std::vector<HloInstruction*> new_operands/*empty*/;
      new_instr =
        instr->CloneWithNewOperands(instr->shape(), new_operands, context);
    } else {
      new_instr = HloInstruction::CreateParameter(new_param_id,
          instr->shape(), "scoped." + instr->name());
      ++new_param_id;
    }

    instr->SetupDerivedInstruction(new_instr.get());
    new_instr->set_parent(computation_);
    context->MapInstruction(instr, new_instr.get());
    //std::cout << "Cloning Param Into->" << new_instr->ToString() << "\n";
    instructions.push_back(std::move(new_instr));
  }

  // Create Computation Body
  for (auto instr : scope_insts) {
    if (visited_inputs.count(instr)) continue;

    // Collect new operands
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(instr->operand_count());
    for (auto operand : instr->operands()) {
      new_operands.emplace_back(context->GetInstruction(operand));
    }

    // Create the instruction
    new_instr =
      instr->CloneWithNewOperands(instr->shape(), new_operands, context);

    //std::cout << "Cloning Instr Into->" << new_instr->ToString() << "\n";
    instructions.push_back(std::move(new_instr));
  }

  // Now build the root *tuple* instruction
  std::vector<HloInstruction*> root_operands;
  root_operands.reserve(fusion_outputs.size());
  for (auto instr : fusion_outputs) {
    auto output_clone = context->GetInstruction(instr);
    root_operands.emplace_back(output_clone);
  }
  new_instr = HloInstruction::CreateTuple(root_operands);
  auto new_root = new_instr.get();
  //std::cout << "Compute Scoped Task Root->" << new_root->ToString() << "\n";
  instructions.push_back(std::move(new_instr));

  HloComputation::Builder builder(computation_->name() + "." + suffix);
  for (auto& instr : instructions) {
    builder.AddInstruction(std::move(instr));
  }
  auto result = builder.Build(new_root);
  auto scoped_task = result.get();
  context->MapComputation(computation_, scoped_task);
  module_->SetScopedSubModule(scope_sig, scoped_task);
  module_->AddEmbeddedComputation(std::move(result));
}

void ScopedFusion::DoScopeFusion(std::string& target_scope) {
  auto post_order = computation_->MakeInstructionPostOrder();
  std::vector<HloInstruction*> fwd_insts, bwd_insts;
  for (auto inst : post_order) {
    if (inst_scope_.count(inst)) {
      auto hlo = inst;
      auto scope = inst_scope_[inst];
      if (CanalizeScope(scope) == target_scope) {
        if (BACKWARD_SCOPE(scope)) {
          bwd_insts.push_back(hlo);
        } else {
          fwd_insts.push_back(hlo);
        }
      }
    }
  }

  std::cout << "Direct Fwd->Bwd Edges:\n";
  std::unordered_set<HloInstruction*> fwd_set(fwd_insts.begin(), 
                                              fwd_insts.end());
  std::unordered_set<HloInstruction*> bwd_set(bwd_insts.begin(), 
                                              bwd_insts.end());
  for (auto inst : post_order) {
    if (!bwd_set.count(inst)) continue;

    for (auto operand : inst->operands()) {
      if (fwd_set.count(operand)) {
        std::cout << operand->ToString() << "\n";
      }
    }
  }

  std::cout << "Forward Instructions in Scope:" << fwd_insts.size() << "\n";
  LiftScope(fwd_insts, target_scope, "fwd");
  std::cout << "Backward Instructions in Scope:" << bwd_insts.size() << "\n";
  LiftScope(bwd_insts, target_scope, "bwd");
  //std::cout << "Module after extracting the scoped tasks->"
  //        << module_->ToString();
}

bool ScopedFusion::BuildFusionAndGTEs(
    std::string& scope, HloCloneContext *context,
    std::unordered_set<const HloInstruction*>& cloned,
    std::vector<std::unique_ptr<HloInstruction>>* instructions) {
  //std::cout << "BuildFusionAndGTEs->" << scope << "\n";

  auto& fusion_inputs = fusion_inputs_map_.at(scope);
  std::vector<HloInstruction*> new_inputs;
  new_inputs.reserve(fusion_inputs.size());
  for (auto input : fusion_inputs) {
    if (inst_scope_.count(input) && 
        input->opcode() != HloOpcode::kConstant) {
      if (!gte_map_.count(input)) {
        return false;
      }

      new_inputs.emplace_back(gte_map_[input]);
    } else {
      if (!cloned.count(input)) {
        return false;
      }
      new_inputs.emplace_back(context->GetInstruction(input));
    }
  }

  auto& fusion_outputs = fusion_outputs_map_.at(scope);
  std::vector<Shape> sub_shapes;
  sub_shapes.reserve(fusion_outputs.size());
  for (auto instr : fusion_outputs) {
    sub_shapes.emplace_back(instr->shape());
  }
  auto output_shape = ShapeUtil::MakeTupleShape(sub_shapes);

  // Create scoped fusion kSubModuleCall
  auto callee = module_->GetScopedSubModule(scope);
  auto fusion = HloInstruction::CreateFusion(output_shape, 
      HloInstruction::FusionKind::kSubModuleCall, new_inputs, callee);
  auto fusion_ptr = fusion.get();
  instructions->push_back(std::move(fusion));

  int num_outputs = fusion_outputs.size();
  // Perform a shape check
  auto callee_root = callee->root_instruction();
  auto& callee_root_shape = callee_root->shape();
  for (int i = 0; i < num_outputs; ++i) {
    CHECK(ShapeUtil::Equal(sub_shapes[i],
          ShapeUtil::GetSubshape(callee_root_shape, {i})));
  }

  // Create output GTE  
  for (int i = 0; i < num_outputs; ++i) {
    auto old_output = fusion_outputs[i];
    auto gte = HloInstruction::CreateGetTupleElement(
        sub_shapes[i], fusion_ptr, i);
    gte_map_[old_output] = gte.get();
    instructions->push_back(std::move(gte));
  }

  //std::cout << "DONE BuildFusionAndGTEs->" << scope << "\n";
  return true;
}

void ScopedFusion::BuildTopLevelComputation(
    std::set<std::string>& scopes) {
  // Task Extraction
  std::unique_ptr<HloCloneContext> context_ptr;
  const string suffix = "TopLevelModule";
  context_ptr = absl::make_unique<HloCloneContext>(
      computation_->parent(), suffix);
  auto context = context_ptr.get();
  std::vector<std::unique_ptr<HloInstruction>> instructions;
  std::unique_ptr<HloInstruction> new_instr;

  // A HLO clone helper
  auto clone_instr = [&instructions, &new_instr, &context](
                     HloInstruction* inst) {
    // Collect new operands
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(inst->operand_count());
    for (int64 i = 0; i < inst->operand_count(); ++i) {
      auto operand = inst->mutable_operand(i);

      HloInstruction* new_operand = nullptr;
      new_operand = context->GetInstruction(operand);
      new_operands.emplace_back(new_operand);
    }

    // Create the instruction
    new_instr =
      inst->CloneWithNewOperands(inst->shape(), new_operands, context);

    //std::cout << "Cloning Instr Into->" << new_instr->ToString() << "\n";
    context->MapInstruction(inst, new_instr.get());
    instructions.push_back(std::move(new_instr));
  };

  std::unordered_set<const HloInstruction*> cloned;
  // Create Parameters
  int new_param_id = 0;
  int num_parameters = computation_->num_parameters();
  for (auto instr : computation_->parameter_instructions()) {
    new_instr = HloInstruction::CreateParameter(new_param_id,
        instr->shape(), instr->name());
    instr->SetupDerivedInstruction(new_instr.get());
    new_instr->set_parent(computation_);
    context->MapInstruction(instr, new_instr.get());
    ++new_param_id;
    cloned.insert(instr);

    VLOG(2) << "Cloning Into->" << new_instr->ToString();
    instructions.push_back(std::move(new_instr));
  }

  // Create Top Level Computation Body
  auto old_root = computation_->root_instruction();
  auto post_order = computation_->MakeInstructionPostOrder();
  std::set<std::string> resolved;
  // Resolve candidates
  std::vector<HloInstruction*> candidates;
  for (auto inst : post_order) {
    if (inst->opcode() == HloOpcode::kParameter) {
      continue;
    }

    // Duplicate kConstant instructions to break false dependences.
    // We may remove such redundant instructions (if any) later.
    if (inst->opcode() == HloOpcode::kConstant) {
      candidates.push_back(inst);
      continue;
    }

    if (inst_scope_.count(inst)) {
      continue;
    }

    if (inst->opcode() == HloOpcode::kTuple) {
      CHECK(inst == old_root);
      continue;
    }
    candidates.push_back(inst);
  }

  // Clone instructions that can be cloned which are out of any scopes
  for (auto instr : candidates) {
    bool all_cloned = true; 
    for (auto operand : instr->operands()) {
      if (!cloned.count(operand)) {
        all_cloned = false;
        break;
      }
    }

    if (!all_cloned) continue;
    clone_instr(instr);
    cloned.insert(instr);
  }

  // It is possible that scoped computations happen to be producers
  // of the top level computation.
  for (auto scope : scopes) {
    // Ignore return values in this case.
    BuildFusionAndGTEs(scope, context, cloned, &instructions);
  }

  // Clone the rest of instructions
  // We need multiple passes to ensure that all candidates are transformed.
  //std::cout << computation_->ToString() << "\n";
  int prev_round_size = 0;
  while (cloned.size() - num_parameters < candidates.size() &&
         cloned.size() > prev_round_size) {
    prev_round_size = cloned.size();
    for (auto inst : candidates) {
      if (cloned.count(inst)) continue;
      // Collect new operands
      std::vector<HloInstruction*> new_operands;
      new_operands.reserve(inst->operand_count());
      bool can_proceed = true;
      for (int64 i = 0; i < inst->operand_count(); ++i) {
        auto operand = inst->mutable_operand(i);

        HloInstruction* new_operand = nullptr;
        if (!inst_scope_.count(operand) ||
            operand->opcode() == HloOpcode::kConstant) {
          if (!cloned.count(operand)) {
            can_proceed = false;
            break;
          }

          new_operand = context->GetInstruction(operand);
        } else {
          auto scope = inst_scope_.at(operand);
          if (!resolved.count(scope)) {
            if (!BuildFusionAndGTEs(scope, context, cloned, &instructions)) {
              can_proceed = false;
              break;
            }
            resolved.insert(scope);
          }
          CHECK(gte_map_.count(operand));
          new_operand = gte_map_[operand];
        }
        new_operands.emplace_back(new_operand);
      } // for (operands)

      if (can_proceed) {
        // Create the instruction
        new_instr =
          inst->CloneWithNewOperands(inst->shape(), new_operands, context);

        //std::cout << "Cloning Instr Into->" << new_instr->ToString() << "\n";
        context->MapInstruction(inst, new_instr.get());
        instructions.push_back(std::move(new_instr));
        cloned.insert(inst);
      }
    } // for
  } // while
  if (cloned.size() - num_parameters < candidates.size()) {
    std::cout << "FATAL ERROR in BuildTopLevelModule:"
              << "cloned->" << cloned.size()
              << " num_parameters->" << num_parameters
              << " candidates->" << candidates.size() << "\n";
    CHECK(0);
  }

  // Now build the root *tuple* instruction
  std::vector<HloInstruction*> root_operands;
  root_operands.reserve(old_root->operand_count());
  for (auto instr : old_root->operands()) {
    HloInstruction* output_clone = nullptr;
    if (gte_map_.count(instr)) {
      output_clone = gte_map_[instr];
    } else {
      output_clone = context->GetInstruction(instr);
    }
    root_operands.emplace_back(output_clone);
  }

  new_instr = HloInstruction::CreateTuple(root_operands);
  auto new_root = new_instr.get();
  //std::cout << "TopLevel Task Root->" << new_root->ToString() << "\n";
  instructions.push_back(std::move(new_instr));

  HloComputation::Builder builder(computation_->name() + "." + suffix);
  for (auto& instr : instructions) {
    builder.AddInstruction(std::move(instr));
  }

  auto result = builder.Build(new_root);
  auto top_level_task = result.get();
  context->MapComputation(computation_, top_level_task);
  module_->AddEmbeddedComputation(std::move(result));
  //std::cout << module_->ToString() << "\n";
}

// Check producers only
bool ScopedFusion::ProducerInference(HloInstruction* inst) {
  std::string input_scope;
  for (auto input : inst->operands()) {
    if (input->opcode() == HloOpcode::kConstant) {
      continue;
    }

    if (!inst_scope_.count(input)) {
      return false;
    }

    auto _scope = inst_scope_[input];
    if (input_scope.empty()) {
      input_scope = _scope;
    } else if (input_scope != _scope) {
      return false;
    }
  }
  if (input_scope.empty()) {
    return false;
  }

  //std::cout << "ProducerInference->" << inst->name() 
  //          << ":" << input_scope << "\n";
  inst_scope_[inst] = input_scope;
  return true;
}

// Check both producers and consumers
bool ScopedFusion::ProducerConsumerInference(HloInstruction* inst) {
  std::string input_scope;
  std::set<std::string> input_scopes;
  for (auto input : inst->operands()) {
    if (input->opcode() == HloOpcode::kConstant/* &&
        ShapeUtil::IsScalar(input->shape())*/) continue;

    if (inst_scope_.count(input)) {
      auto scope_ = inst_scope_[input];
      if (!input_scopes.count(scope_)) {
        input_scopes.insert(scope_);
      }
    }
  }

  // TODO: Make this rule more general later
  int num_input_scopes = input_scopes.size();
  if (1 == num_input_scopes) {
    input_scope = *input_scopes.begin();
  } else if (2 == num_input_scopes) {
    auto x = *input_scopes.begin();
    auto y = *(++input_scopes.begin());
    if (CanalizeScope(x) == CanalizeScope(y)) {
      if (BACKWARD_SCOPE(x)) {
        input_scope = x;
      } else {
        CHECK(BACKWARD_SCOPE(y));
        input_scope = y;
      }
    } else {
      return false;
    }
  } else {
    return false;
  }

  std::deque<HloInstruction*> worklist;
  worklist.push_back(inst);

  std::unordered_set<HloInstruction*> visited;
  visited.insert(inst);

  while (!worklist.empty()) {
    auto hlo = worklist.front();
    worklist.pop_front();
    visited.insert(hlo);

    for (auto user : hlo->users()) {
      if (visited.count(user)) continue;

      if (inst_scope_.count(user) &&
          input_scope == inst_scope_[user]) {
        inst_scope_[inst] = input_scope;
        return true;
      }
      worklist.push_back(user);
      visited.insert(user);
    }
  }

  return false;
}

bool ScopedFusion::StrictScopeInference(HloInstruction* inst) {
  std::string input_scope;
  for (auto input : inst->operands()) {
    if (inst_scope_.count(input)) {
      if (input_scope.empty()) {
        input_scope = inst_scope_[input];
      } else {
        CHECK(input_scope == inst_scope_[input]);
      }
    }
  }

  if (input_scope.empty()) {
    return false;
  }

  std::deque<HloInstruction*> worklist;
  worklist.push_back(inst);

  std::unordered_set<HloInstruction*> visited;
  visited.insert(inst);

  while (!worklist.empty()) {
    auto hlo = worklist.front();
    worklist.pop_front();
    visited.insert(hlo);

    for (auto user : hlo->users()) {
      if (visited.count(user)) continue;

      if (!inst_scope_.count(user) ||
          input_scope != inst_scope_[user]) {
        return false;
      }
      worklist.push_back(user);
      visited.insert(user);
    } // for (user)
  } // while (worklist)

  inst_scope_[inst] = input_scope;
  return true;
}

void ScopedFusion::ResolveInstScopes() {
  auto post_order = computation_->MakeInstructionPostOrder();
  for (auto inst : post_order) {
    std::string scope;
    if (ParseScope(inst, scope)) {
      inst_scope_[inst] = scope;
    }
  }
}

void ScopedFusion::ScopeInference() {
  auto post_order = computation_->MakeInstructionPostOrder();
  std::unordered_set<HloInstruction*> candidates;
  for (auto inst : post_order) {
    std::string scope;
    if (!inst_scope_.count(inst)) {
      candidates.insert(inst);
    }
  }

  std::unordered_set<HloInstruction*> inferred;
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto inst : candidates) {
      if (inferred.count(inst)) continue;

      if (ProducerInference(inst) ||
          ProducerConsumerInference(inst)) {
        inferred.insert(inst);
        changed = true;
      }
    }
  }
  if (!inst_scope_.empty()) std::cout << "HLO instructions inferred:\n";
  for (auto hlo : inferred) {
    std::cout << inst_scope_[hlo] << "-->" << hlo->ToString() << "\n";
  }
  //if (!inst_scope_.empty()) { 
  //  std::cout << computation_->ToString() << "\n"; CHECK(0); 
  //}
}

void ScopedFusion::FixFalseDeps() {
  std::unordered_set<HloInstruction*> fwd_set, bwd_set;
  for (auto& it : inst_scope_) {
    auto hlo = it.first;
    auto scope = it.second;

    if (BACKWARD_SCOPE(scope)) {
      bwd_set.insert(hlo);
    } else {
      fwd_set.insert(hlo);
    }
  }

  //std::cout << "FWDSET Size:" << fwd_set.size() << "\n";
  //std::cout << "BWDSET Size:" << bwd_set.size() << "\n";
  for (auto bwd_inst : bwd_set) {
    bool hit = false;
    for (auto fwd_inst : fwd_set) {
      if (reachability_->IsReachable(bwd_inst, fwd_inst)) {
        hit = true;
        break;
      }
    }
    if (!hit) continue;

    CHECK(bwd_inst->opcode() == HloOpcode::kConstant &&
          ShapeUtil::IsScalar(bwd_inst->shape()));
    // We may need to do something here ...
  }
}

StatusOr<bool> ScopedFusion::Run(HloModule* module) {
  bool changed = false;
  module_ = module;

  for (auto* computation : module->MakeNonfusionComputationsSorted()) {
    computation_ = computation;
    reachability_ = HloReachabilityMap::Build(computation_);

    ResolveInstScopes();

    FixFalseDeps();

    // Scoping HLO instructions that are "accidentally" missed from 
    // the JAX frontend.
    ScopeInference();

    auto post_order = computation_->MakeInstructionPostOrder();
    std::set<std::string> scopes;
    for (auto inst : post_order) {
      auto op_name = inst->metadata().op_name();
      auto sep = op_name.find("@^_^@");
      if (sep != string::npos) {
        auto scope = op_name.substr(0, sep);
        if (!BACKWARD_SCOPE(scope)) {
          scopes.insert(scope);
        }
      }
    }
    
    if (VLOG_IS_ON(2) && !scopes.empty()) {
      std::cout << "Scopes: " << scopes.size() << "\n";
      for (auto& scope : scopes) std::cout << scope << " ";
      std::cout << "\n";
    }

    for (auto scope : scopes) {
      DoScopeFusion(scope);
    }

    if (!scopes.empty()) {
      BuildTopLevelComputation(scopes);
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
