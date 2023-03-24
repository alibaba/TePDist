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

#include "tensorflow/compiler/xla/service/parallel/computation_clone_utils.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/shape.h"

#include <algorithm>

namespace xla {

HloComputation* CloneComputation(CloneSignature& signature) {
  // Clone Computation
  HloComputation* computation = signature.orig_computation;
  std::unique_ptr<HloCloneContext> context_ptr = 
      absl::make_unique<HloCloneContext>(computation->parent(), signature.name);
  auto context = context_ptr.get();
  std::vector<std::unique_ptr<HloInstruction>> instructions;

  // Create parameters 
  int arg_no = 0;
  int64 num_params = signature.params.size();
  std::unordered_set<const HloInstruction*> params(
      signature.params.begin(), signature.params.end());
  for (int64 p = 0; p < num_params; ++p) {
    auto param = signature.params[p];
    std::unique_ptr<HloInstruction> new_param = HloInstruction::CreateParameter(
        arg_no++, param->shape(), param->name());
    *new_param->mutable_dist_spec() = param->dist_spec();
    param->SetupDerivedInstruction(new_param.get());
    context->MapInstruction(param, new_param.get());
    instructions.push_back(std::move(new_param));
  }

  // Create computation body
  auto post_order = computation->MakeInstructionPostOrder();
  for (auto instr : post_order) {
    if (instr->opcode() == HloOpcode::kParameter) continue;

    if (signature.body.find(instr) != signature.body.end()) {
      std::vector<HloInstruction*> new_operands;
      new_operands.reserve(instr->operand_count());
      for (auto operand : instr->operands()) {
        new_operands.emplace_back(context->GetInstruction(operand));
      }

      // Create the instruction
      std::unique_ptr<HloInstruction> new_instr = instr->CloneWithNewOperands(
          instr->shape(), new_operands, context);
      *new_instr->mutable_dist_spec() = instr->dist_spec();
      instructions.push_back(std::move(new_instr));
    }
  } // for (hlo : post_order)

  // Create output tuple
  std::vector<HloInstruction*> new_root_operands;
  new_root_operands.reserve(signature.root_operands.size());
  for (int i = 0; i < signature.root_operands.size(); ++i) {
    auto instr = signature.root_operands[i];
    auto output_clone = context->GetInstruction(instr);
    new_root_operands.emplace_back(output_clone);
  }
 
  // Build the new computation
  std::unique_ptr<HloInstruction> new_instr = 
      HloInstruction::CreateTuple(new_root_operands);
  auto new_root = new_instr.get();
  instructions.push_back(std::move(new_instr));
  HloComputation::Builder builder(computation->name() + "." + signature.name);
  for (auto& instr : instructions) {
    builder.AddInstruction(std::move(instr));
  }
  auto result = builder.Build(new_root);
  auto computation_clone = result.get();
  computation->parent()->AddEmbeddedComputation(std::move(result));
  VLOG(2) << "Done computation : " << computation_clone->ToString();
  return computation_clone;
}

void InferDefContextMembers(
    CloneSignature& signature, HloModule::DefContext* parent_def_ctx,
    HloModule::DefContext* def_ctx) {
  CHECK(parent_def_ctx && def_ctx);  
  HloComputation* orig_comp = signature.orig_computation;
  std::unordered_map<const HloInstruction*, int/*order*/> params_ids_in_signature;
  std::unordered_map<const HloInstruction*, int/*order*/> root_ops_ids_in_signature;
  for (int i = 0; i < signature.params.size(); ++i) {
    const HloInstruction* param = signature.params[i];
    params_ids_in_signature[param] = i;
  }
  for (int i = 0; i < signature.root_operands.size(); ++i) {
    const HloInstruction* op = signature.root_operands[i];
    root_ops_ids_in_signature[op] = i;
  }
  // 1. Infer input_arg_map, input_dim_to_slice, input_var_size_map and
  // sharded_args
  for (int parent_id = 0; parent_id < orig_comp->num_parameters(); ++parent_id) {
    const HloInstruction* param = orig_comp->parameter_instruction(parent_id);
    if (params_ids_in_signature.find(param) == params_ids_in_signature.end()) continue;
    int i = params_ids_in_signature[param];
    def_ctx->input_arg_map_[i] = parent_id;
    int64 result_bytes = ShapeUtil::ByteSizeOf(param->shape());
    def_ctx->input_var_size_map_[i] = result_bytes;
    if (parent_def_ctx->input_dim_to_slice_.find(parent_id) !=
        parent_def_ctx->input_dim_to_slice_.end()) {
      def_ctx->input_dim_to_slice_[i] =
          parent_def_ctx->input_dim_to_slice_[parent_id];
    }

    if (std::find(parent_def_ctx->sharded_args_.begin(),
                  parent_def_ctx->sharded_args_.end(), parent_id) !=
        parent_def_ctx->sharded_args_.end()) {
      def_ctx->sharded_args_.push_back(i);
    }
  }

  // 2. Infer output_idx_map, output_dim_to_slice
  const HloInstruction* root = orig_comp->root_instruction();
  for (int parent_id = 0; parent_id < root->operand_count(); ++parent_id) {
    const HloInstruction* op = root->operand(parent_id);
    if (root_ops_ids_in_signature.find(op) == root_ops_ids_in_signature.end()) continue;
    int i = root_ops_ids_in_signature[op];
    def_ctx->output_idx_map_[i] = parent_id;

    if (parent_def_ctx->output_dim_to_slice_.find(parent_id) !=
        parent_def_ctx->output_dim_to_slice_.end()) {
      def_ctx->output_dim_to_slice_[i] =
          parent_def_ctx->output_dim_to_slice_[parent_id];
    }
  }

  // 3. Infer output_tensor_size_map_
  for (int i = 0; i < signature.root_operands.size(); ++i ) {
    const HloInstruction* op = signature.root_operands[i];
    int64 result_bytes = ShapeUtil::ByteSizeOf(op->shape());
    def_ctx->output_tensor_size_map_[i] = result_bytes;
  }

  // 4. Infer input_output_alias_map
  std::unordered_map<int, int> parent_output_input_alias_map;
  for (auto& iter : parent_def_ctx->input_output_alias_map_) {
    parent_output_input_alias_map[iter.second] = iter.first;
  }

  for (int i = 0; i < signature.root_operands.size(); ++i) {
    if (def_ctx->output_idx_map_.find(i) == def_ctx->output_idx_map_.end()) continue;
    int output_parent_id = def_ctx->output_idx_map_[i];
    if (parent_output_input_alias_map.find(output_parent_id) ==
        parent_output_input_alias_map.end()) continue;

    int alias_input_parent_id = parent_output_input_alias_map[output_parent_id];
    const HloInstruction* param =
        orig_comp->parameter_instruction(alias_input_parent_id);
    const HloInstruction* root_op =
        orig_comp->root_instruction()->operand(output_parent_id);
    if (params_ids_in_signature.find(param) ==
        params_ids_in_signature.end()) continue;

    if (root_ops_ids_in_signature.find(root_op) ==
        root_ops_ids_in_signature.end()) continue;

    int param_id = params_ids_in_signature[param];
    int output_id = root_ops_ids_in_signature[root_op];
    def_ctx->input_output_alias_map_[param_id] = output_id;
  }
}

} // namespace xla
