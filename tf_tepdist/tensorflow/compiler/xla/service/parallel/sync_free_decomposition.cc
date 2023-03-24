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

#include "tensorflow/compiler/xla/service/parallel/sync_free_decomposition.h"

#include "tensorflow/compiler/xla/service/parallel/resolve_utils.h"

#include <deque>

namespace xla {

std::vector<const HloInstruction*> SyncFreeDecomposition::get_fetches_insts(
    HloComputation* computation, int num_fetches) {
  std::vector<const HloInstruction*> fetches;
  const HloInstruction* root = computation->root_instruction();
  for (int i = 0; i < num_fetches; ++i) {
    const HloInstruction* inst = root->operand(i);
    fetches.push_back(inst);
  }
  return fetches;
}

CloneSignature SyncFreeDecomposition::BuildSyncFreeSignature(
    HloComputation* entry, std::vector<const HloInstruction*>& root_ops) {
  // Resolve instructions
  std::deque<const HloInstruction*> worklist(
      root_ops.begin(), root_ops.end());
  std::unordered_set<const HloInstruction*> visited;
  std::unordered_set<const HloInstruction*> params_set;

  CloneSignature signature;
  while (!worklist.empty()) {
    const HloInstruction* instr = worklist.front();
    worklist.pop_front();
    visited.insert(instr);
    if (instr->opcode() == HloOpcode::kParameter) {
      params_set.insert(instr);
    } else {
      signature.body.insert(instr);
    }

    for (const HloInstruction* op : instr->operands()) {
      if (visited.find(op) != visited.end()) continue;
      worklist.push_back(op);
    }
  }

  signature.params.insert(
      signature.params.end(), params_set.begin(), params_set.end());

  std::sort(signature.params.begin(), signature.params.end(),
            [](const HloInstruction* a, const HloInstruction* b) {
    return a->parameter_number() < b->parameter_number();
  });

  signature.name = "SyncFreeComputation";
  signature.root_operands = root_ops;
  signature.orig_computation = entry;
  return signature;
}

void SyncFreeDecomposition::SetupSyncFreeDefContext(
    CloneSignature& signature, HloModule* module,
    HloComputation* computation, std::vector<const HloInstruction*>& params,
    std::vector<const HloInstruction*>& sync_free_outputs) {
  // Setup DefContext
  // DefContext for ComputeGradients->level=1, index=0
  // Note that *def_ctx* for ComputeGradients does not have a module owner yet.
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  CHECK(top_def_ctx);
  HloModule::DefContext* def_ctx = module->new_def_ctx(
      "CG", HloModule::DefContext::DefType::CG,
      top_def_ctx->def_id()/*parent_id*/);
  top_def_ctx->children().push_back(def_ctx);
  HloComputation* entry = module->entry_computation();
  const HloInstruction* root = entry->root_instruction();

  // Setup output_idx_map, output_dim_to_slice and output_tensor_size_map
  for (int i = 0; i < sync_free_outputs.size(); ++i) {
    const HloInstruction* sync = sync_free_outputs[i];
    int64 result_bytes = ShapeUtil::ByteSizeOf(sync->shape());
    CHECK(def_ctx->output_tensor_size_map_.find(i) ==
          def_ctx->output_tensor_size_map_.end());
    def_ctx->output_tensor_size_map_[i] = result_bytes;

    if (!root->IsUserOf(sync)) continue;
    int64 op_idx = root->operand_index(sync);
    def_ctx->output_idx_map_[i] = op_idx;
    def_ctx->output_dim_to_slice_[i] = top_def_ctx->output_dim_to_slice_[op_idx];
  }

  // Setup input_arg_map, input_dim_to_slice and input_var_size_map
  int64 num_params = entry->num_parameters();
  std::unordered_set<const HloInstruction*> params_set(
      params.begin(), params.end());

  std::map<int, std::string>* vars_map = module->variable_map();
  int arg_no = 0;
  for (int64 p = 0; p < num_params; ++p) {
    const HloInstruction* param = entry->parameter_instruction(p);
    if (params_set.find(param) == params_set.end()) continue;

    if (top_def_ctx->input_dim_to_slice_.find(p) != top_def_ctx->input_dim_to_slice_.end()) {
      def_ctx->input_dim_to_slice_[arg_no] = top_def_ctx->input_dim_to_slice_[p];
    }

    def_ctx->input_arg_map_[arg_no] = p;
    if (vars_map->find(p) != vars_map->end()) {
      // record input tensor size
      int64 result_bytes = ShapeUtil::ByteSizeOf(param->shape());

      // TODO(lansong): tensor size of input activation and trainable vars should
      // be recorded in input_activation_size_map_ and input_var_size_map_
      // respectively.
      CHECK(def_ctx->input_var_size_map_.find(arg_no) ==
            def_ctx->input_var_size_map_.end());
      def_ctx->input_var_size_map_[arg_no] = result_bytes;
      def_ctx->sharded_args_.push_back(arg_no);
    }
    ++arg_no;
  }

  // The instances of all slices should be created
  for (int s = 0; s < module->total_split_num(); ++s) {
    def_ctx->instance_slice_ids_.insert(s);
  }

  module->SetDefCompute(def_ctx, computation);
}

HloComputation* SyncFreeDecomposition::BuildSyncFreeComputation(
    HloModule* module, SyncFreeChain& sync_free_chain) {
  std::unordered_set<const HloInstruction*> sync_points = sync_free_chain.sync_points();
  std::vector<const HloInstruction*> sync_free_outputs;
  for (auto* instr : sync_points) {
    sync_free_outputs.push_back(instr->operand(0));
  }
  HloComputation* entry = module->entry_computation();
  CloneSignature signature = BuildSyncFreeSignature(entry, sync_free_outputs);
  HloComputation* computation_clone = CloneComputation(signature);
  SetupSyncFreeDefContext(
      signature, module, computation_clone, signature.params, signature.root_operands);
  return computation_clone;
}

CloneSignature SyncFreeDecomposition::BuildSignatureForRemains(
    HloComputation* entry, SyncFreeChain& sync_free_chain) {
  std::unordered_map<const HloInstruction*, int/*input order*/> sync_points_order_map;
  std::vector<const HloInstruction*> sync_vecs(
      sync_free_chain.sync_points().begin(), sync_free_chain.sync_points().end());
  for (int i = 0; i < sync_vecs.size(); ++i) {
    sync_points_order_map[sync_vecs[i]] = i;
  }

  const HloInstruction* root = entry->root_instruction();
  std::deque<const HloInstruction*> worklist(
      root->operands().begin(), root->operands().end());
  std::unordered_set<const HloInstruction*> params_set;
  std::unordered_set<const HloInstruction*> other_inputs_set;
  std::unordered_set<const HloInstruction*> visited;
  CloneSignature signature;
  while (!worklist.empty()) {
    const HloInstruction* instr = worklist.front();
    worklist.pop_front();
    visited.insert(instr);

    if (instr->opcode() == HloOpcode::kParameter) {
      params_set.insert(instr);
      continue;
    } else if (sync_free_chain.sync_points().find(instr) !=
               sync_free_chain.sync_points().end()) {
      other_inputs_set.insert(instr);
      continue;
    } else {
      signature.body.insert(instr);
    }

    for (const HloInstruction* op : instr->operands()) {
      if (visited.find(op) != visited.end()) continue;
      worklist.push_back(op);
    }
  }

  signature.params.insert(
      signature.params.end(), params_set.begin(), params_set.end());

  std::sort(signature.params.begin(), signature.params.end(),
            [](const HloInstruction* a, const HloInstruction* b) {
    return a->parameter_number() < b->parameter_number();
  });

  std::vector<const HloInstruction*> other_inputs(
      other_inputs_set.begin(), other_inputs_set.end());

  std::sort(other_inputs.begin(), other_inputs.end(),
            [&](const HloInstruction* a, const HloInstruction* b) {
    return sync_points_order_map[a] < sync_points_order_map[b];
  });

  signature.name = "SyncComputation";
  signature.params.insert(
      signature.params.end(), other_inputs.begin(), other_inputs.end());
  signature.root_operands.assign(
      root->operands().begin(), root->operands().end());
  signature.orig_computation = entry;
  return signature;
}

void SyncFreeDecomposition::SetupDefContextForRemains(
    CloneSignature& signature, HloModule* module, HloComputation* computation,
    const std::unordered_set<const HloInstruction*>& sync_points) {
  // DefContext for ApplyGradients->level=1, index=1
  // Note that *def_ctx* for ApplyGradients does not have a module owner yet.
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  CHECK(top_def_ctx);
  HloModule::DefContext* def_ctx = module->new_def_ctx(
      "AG", HloModule::DefContext::DefType::AG,
      top_def_ctx->def_id()/*parent_id*/);
  top_def_ctx->children().push_back(def_ctx);
  HloModule::DefContext* ga_def_ctx = top_def_ctx->GADefCtx();

  InferDefContextMembers(signature, top_def_ctx, def_ctx);

  auto var_map = module->variable_map();
  int def_arg_start = def_ctx->input_arg_map_.size();
  for (auto& iter : def_ctx->input_arg_map_) {
    int arg_no = iter.first;
    int parent_arg_no = iter.second;
    if (var_map->find(parent_arg_no) != var_map->end()) {
      const HloInstruction* param = computation->parameter_instruction(arg_no);
      def_ctx->input_var_size_map_[arg_no] = ShapeUtil::ByteSizeOf(param->shape());
    }
  }

  // Setup input_def_map
  for (int64 i = 0; i < sync_points.size(); ++i) {
    for (int s = 0; s < module->total_split_num(); ++s) {
      std::vector<int64> addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(s);
      if (addr[share_split_ordinal_] < module->split_nums()[share_split_ordinal_] - 1) continue;
      // Only the last slice of share_split_ordinal_ should be created
      def_ctx->instance_slice_ids_.insert(s);
      def_ctx->add_to_input_def_map(def_arg_start + i, s, s, ga_def_ctx->def_id(), i);
    }

    // record input tensor size (for activation tensors)
    CHECK(def_ctx->input_activation_size_map_.find(def_arg_start + i) ==
          def_ctx->input_activation_size_map_.end());
    CHECK(ga_def_ctx->output_tensor_size_map_.find(i) !=
          ga_def_ctx->output_tensor_size_map_.end());    // ga def context is ready
    def_ctx->input_activation_size_map_[def_arg_start + i] =
        ga_def_ctx->output_tensor_size_map_[i];
    def_ctx->sharded_args_.push_back(def_arg_start + i);
  }

  for (auto& it : def_ctx->output_idx_map_) {
    top_def_ctx->output_idx_global_dev_map_[it.second] = def_ctx->instance_slice_ids_;
  }

  module->SetDefCompute(def_ctx, computation);
}

HloComputation* SyncFreeDecomposition::BuildComputationForRemains(
    HloModule* module, SyncFreeChain& sync_free_chain) {
  HloComputation* entry = module->entry_computation();
  CloneSignature signature = BuildSignatureForRemains(entry, sync_free_chain);
  HloComputation* computation_clone = CloneComputation(signature);
  SetupDefContextForRemains(
      signature, module, computation_clone, sync_free_chain.sync_points());
  return computation_clone;
}

HloComputation* SyncFreeDecomposition::BuildLocalAccumulationComputation(
    HloModule* module, SyncFreeChain& sync_free_chain) {
  // DefContext for GA->level=1, index=2
  // Note that *def_ctx* for GA does not have a module owner yet.
  auto top_def_ctx = module->def_ctx();
  CHECK(top_def_ctx);
  auto def_ctx = module->new_def_ctx(
      "GA", HloModule::DefContext::DefType::GA,
      top_def_ctx->def_id()/*parent_id*/);
  top_def_ctx->children().push_back(def_ctx);
  auto cg_def_ctx = top_def_ctx->ComputeGradientsDefCtx();
  auto ga_init_def_ctx = top_def_ctx->GAInitDefCtx();

  for (int s = 0; s < module->total_split_num(); ++s) {
    def_ctx->instance_slice_ids_.insert(s);
  }

  const std::unordered_set<const HloInstruction*>& sync_points = sync_free_chain.sync_points();
  std::vector<const HloInstruction*> sync_vecs(sync_points.begin(), sync_points.end());

  std::vector<std::unique_ptr<HloInstruction>> instructions;
  int64 num_sync_points = sync_vecs.size();
  std::unordered_map<const HloInstruction*, int/*count*/> sync_points_count_pair;
  std::vector<HloInstruction*> lhs_vecs;
  // Connect to kGAInit inputs
  for (int64 i = 0; i < num_sync_points; ++i) {
    const HloInstruction* sync = sync_vecs[i];
    const Shape& sync_shape = sync->shape();
    const std::string& sync_name = sync->name();
    std::unique_ptr<HloInstruction> new_instr;

    for (int s = 0; s < module->total_split_num(); ++s) {
      std::vector<int64> addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(s);
      if (addr[share_split_ordinal_] > 0) {
        std::vector<int64> prev_ga_addr(addr.begin(), addr.end());
        prev_ga_addr[share_split_ordinal_] -= 1;
        int64 prev_s = comm_dev_mgr_->AddrToLinearIdxBySplitNums(prev_ga_addr);
        def_ctx->add_to_input_def_map(i, s, prev_s, def_ctx->def_id(), i);
      } else {
        def_ctx->add_to_input_def_map(i, s, s, ga_init_def_ctx->def_id(), i);
      }
    }

    def_ctx->input_activation_size_map_[i] = ga_init_def_ctx->output_tensor_size_map_[i];
    def_ctx->input_output_alias_map_[i] = i;

    std::string instr_name = sync_name;
    if (sync_points_count_pair.find(sync) != sync_points_count_pair.end()) {
      instr_name = absl::StrCat(sync_name, "_", sync_points_count_pair[sync]);
      sync_points_count_pair[sync]++;
    } else {
      sync_points_count_pair[sync] = 1;
    }
    new_instr = HloInstruction::CreateParameter(i, sync_shape, instr_name + ".GA.lhs");
    *new_instr->mutable_dist_spec() = sync->dist_spec();
    HloInstruction* lhs = new_instr.get();
    lhs_vecs.push_back(lhs);
    instructions.push_back(std::move(new_instr));
  }

  sync_points_count_pair.clear();
  // Connect to other Computation Outputs
  std::vector<HloInstruction*> rhs_vecs;
  for (int64 i = 0; i < num_sync_points; ++i) {
    const HloInstruction* sync = sync_vecs[i];
    const Shape& sync_shape = sync->shape();
    const std::string& sync_name = sync->name();
    std::unique_ptr<HloInstruction> new_instr;

    for (int s = 0; s < module->total_split_num(); ++s) {
      def_ctx->add_to_input_def_map(i + num_sync_points, s, s, cg_def_ctx->def_id(), i);
    }

    // record input tensor size (for activation tensors)
    CHECK(def_ctx->input_activation_size_map_.find(i + num_sync_points) ==
          def_ctx->input_activation_size_map_.end());
    CHECK(cg_def_ctx->output_tensor_size_map_.find(i) !=
          cg_def_ctx->output_tensor_size_map_.end());    // cg def context is ready
    def_ctx->input_activation_size_map_[i + num_sync_points] =
                    cg_def_ctx->output_tensor_size_map_[i];

    std::string instr_name = sync_name;
    if (sync_points_count_pair.count(sync)) {
      instr_name = absl::StrCat(sync_name, "_", sync_points_count_pair[sync]);
      sync_points_count_pair[sync]++;
    } else {
      sync_points_count_pair[sync] = 1;
    }
    new_instr = HloInstruction::CreateParameter(i + num_sync_points,
        sync_shape, instr_name + ".GA.rhs");
    *new_instr->mutable_dist_spec() = sync->dist_spec();
    HloInstruction* rhs = new_instr.get();
    rhs_vecs.push_back(rhs);
    instructions.push_back(std::move(new_instr));
  }

  std::vector<HloInstruction*> tuple_operands;
  tuple_operands.reserve(num_sync_points);
  for (int64 i = 0; i < num_sync_points; ++i) {
    const HloInstruction* sync = sync_vecs[i];
    const Shape& sync_shape = sync->shape();
    HloInstruction* lhs = lhs_vecs[i];
    HloInstruction* rhs = rhs_vecs[i];

    std::unique_ptr<HloInstruction> new_instr = 
        HloInstruction::CreateBinary(sync_shape, HloOpcode::kAdd, lhs, rhs);
    *new_instr->mutable_dist_spec() = sync->dist_spec();
    HloInstruction* add = new_instr.get();
    instructions.push_back(std::move(new_instr));
    tuple_operands.push_back(add);

    // record output tensor size
    int64 result_bytes = ShapeUtil::ByteSizeOf(add->shape());

    CHECK(def_ctx->output_tensor_size_map_.find(i) ==
          def_ctx->output_tensor_size_map_.end());
    def_ctx->output_tensor_size_map_[i] = result_bytes;
  }

  std::unique_ptr<HloInstruction> new_instr = HloInstruction::CreateTuple(tuple_operands);
  HloInstruction* new_root = new_instr.get();
  instructions.push_back(std::move(new_instr));

  HloComputation::Builder builder("Gradients.Accumulation.GA");
  for (auto& instr : instructions) {
    builder.AddInstruction(std::move(instr));
  }
  std::unique_ptr<HloComputation> result = builder.Build(new_root);
  HloComputation* computation = result.get();
  module->SetDefCompute(def_ctx, computation);
  module->AddEmbeddedComputation(std::move(result));
  return computation;
}

void SyncFreeDecomposition::BuildLocalInitComputation(
    HloModule* module, SyncFreeChain& sync_free_chain) {
  auto top_def_ctx = module->def_ctx();
  auto def_ctx = module->new_def_ctx(
      "GAINIT", HloModule::DefContext::DefType::GAINIT,
      top_def_ctx->def_id()/*parent_id*/);
  top_def_ctx->children().push_back(def_ctx);

  const std::unordered_set<const HloInstruction*>& sync_points = sync_free_chain.sync_points();
  std::vector<const HloInstruction*> sync_vecs(sync_points.begin(), sync_points.end());
  std::vector<HloInstruction*> tuple_operands;
  tuple_operands.reserve(sync_vecs.size());
  HloComputation::Builder builder("Gradients.Accumulation.GAInit");
  for (int64 i = 0; i < sync_vecs.size(); ++i) {
    auto new_instr = HloInstruction::CreateParameter(
        i, sync_vecs[i]->shape(), sync_vecs[i]->name() + ".GA.init");
    *new_instr->mutable_dist_spec() = sync_vecs[i]->dist_spec();
    tuple_operands.push_back(new_instr.get());
    builder.AddInstruction(std::move(new_instr));
  }

  for (int s = 0; s < module->total_split_num(); ++s) {
    std::vector<int64> addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(s);
    if (addr[share_split_ordinal_] > 0) continue;
    def_ctx->instance_slice_ids_.insert(s);
  }

  for (int i = 0; i < tuple_operands.size(); ++i) {
    def_ctx->output_tensor_size_map_[i] = ShapeUtil::ByteSizeOf(tuple_operands[i]->shape());
  }

  std::unique_ptr<HloInstruction> root = HloInstruction::CreateTuple(tuple_operands);
  builder.AddInstruction(std::move(root));
  std::unique_ptr<HloComputation> result = builder.Build(root.get());
  HloComputation* computation = result.get();
  module->SetDefCompute(def_ctx, computation);
  module->AddEmbeddedComputation(std::move(result));
}
  
StatusOr<bool> SyncFreeDecomposition::Run(
    HloModule* module, SyncFreeChain& sync_free_chain) {
  CHECK(share_split_ordinal_ >= 0);
  comm_dev_mgr_ = std::make_unique<CommDevManager>(
      module->split_nums(), module->share_dev_flags(),
      module->placement_layout(), module->stage_split_ordinal(), 1/*num_workers not use*/);
	HloComputation* entry = module->entry_computation();
  HloModule::DefContext* entry_def_ctx = module->def_ctx();
  HloComputation* sync_free_comp = BuildSyncFreeComputation(module, sync_free_chain);
  BuildLocalInitComputation(module, sync_free_chain);
  HloComputation* local_acc_comp = BuildLocalAccumulationComputation(module, sync_free_chain);
  HloComputation* ag_computation = BuildComputationForRemains(module, sync_free_chain);
  return true;
}

} // namespace xla
