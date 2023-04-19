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

#include "tensorflow/compiler/xla/service/parallel/stage_decomposition.h"
#include "tensorflow/compiler/xla/service/service_env.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace xla {

using ::tensorflow::strings::HumanReadableNumBytes;
namespace {

int get_def_child_index(
    HloModule::DefContext* parent_def_ctx, HloModule::DefContext* child_def_ctx) {
  for (int i = 0; i < parent_def_ctx->num_children(); ++i) {
    if (child_def_ctx == parent_def_ctx->child(i)) return i;
  }

  return -1;
}

} // namespace

void StageDecomposition::ResolveTotalStageCount() {
  for (const HloInstruction* inst : main_computation_->instructions()) {
    stage_count_ = std::max(inst->dist_spec().stage() + 1, stage_count_);
  }
}

bool StageDecomposition::ResolveComputations(HloModule* module) {
  HloModule::DefContext* entry_def = module->def_ctx();
  HloModule::DefContext* sync_free_def = entry_def->ComputeGradientsDefCtx();
  HloComputation* sync_free_comp = module->Def2Compute(sync_free_def);
  if (sync_free_comp->instruction_count() > 1) {
    main_computation_ = sync_free_comp;
    HloModule::DefContext* local_comm_def = entry_def->GADefCtx();
    local_accum_ = module->Def2Compute(local_comm_def);
    HloModule::DefContext* remains_def = entry_def->ApplyGradientsDefCtx();
    non_sync_free_ = module->Def2Compute(remains_def);
    return true;
  }

  return false;
}

CloneSignature StageDecomposition::BuildSignatureForStage(
    std::vector<const HloInstruction*>& root_ops,
    std::unordered_set<const HloInstruction*>* excluded,
    HloComputation* parent_computation, std::string& stage_name) {
  std::deque<const HloInstruction*> worklist(root_ops.begin(), root_ops.end());
  std::unordered_set<const HloInstruction*> visited;
  std::unordered_set<const HloInstruction*> other_inputs_set;
  std::unordered_set<const HloInstruction*> params_set;

  CloneSignature signature;
  while (!worklist.empty()) {
    const HloInstruction* instr = worklist.front();
    worklist.pop_front();
    visited.insert(instr);

    if (instr->opcode() == HloOpcode::kParameter) {
      params_set.insert(instr);
    } else if (excluded && excluded->find(instr) != excluded->end()) {
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

  // Sort the params to preserve their original sequence
  std::sort(signature.params.begin(), signature.params.end(),
            [](const HloInstruction* a, const HloInstruction* b) {
    return a->parameter_number() < b->parameter_number();
  });

  std::vector<const HloInstruction*> other_inputs(
      other_inputs_set.begin(), other_inputs_set.end());

  signature.name = stage_name;
  // Append all new inputs back
  signature.params.insert(
      signature.params.end(), other_inputs.begin(), other_inputs.end());
  signature.root_operands.assign(root_ops.begin(), root_ops.end());
  signature.orig_computation = parent_computation;

  return signature;
}

CloneSignature StageDecomposition::BuildStageSignatureForMainComputation(
    int stage_id, std::unordered_set<const HloInstruction*>* prev_stages_insts,
    HloComputation* parent_computation, std::string parent_name) {
  std::vector<const HloInstruction*> root_ops;
  for (const HloInstruction* inst : parent_computation->instructions()) {
    if (inst == parent_computation->root_instruction()) continue;
    if (inst->dist_spec().stage() != stage_id) continue;
    for (const HloInstruction* user : inst->users()) {
      if (user->dist_spec().stage() == stage_id
          && user != parent_computation->root_instruction()) continue;
      root_ops.push_back(inst);
      break;
    }
  }

  std::string stage_name = absl::StrCat(parent_name, "_Stage_", stage_id);
  CloneSignature signature = BuildSignatureForStage(
      root_ops, prev_stages_insts, parent_computation, stage_name);

  prev_stages_insts->insert(signature.params.begin(), signature.params.end());
  prev_stages_insts->insert(signature.body.begin(), signature.body.end());
  prev_stages_insts->insert(
      signature.root_operands.begin(), signature.root_operands.end());

  return signature;
}

void StageDecomposition::SetupStageDefContext(
    HloModule* module, CloneSignature& signature,
    HloModule::DefContext* parent_def_ctx, int s) {
  HloModule::DefContext* def_ctx = parent_def_ctx->child(s);
  InferDefContextMembers(signature, parent_def_ctx, def_ctx);

  HloModule::DefContext* entry_def_ctx = module->def_ctx();
  for (auto& it : def_ctx->output_idx_map_) {
    if (parent_def_ctx->output_idx_map_.find(it.second) ==
        parent_def_ctx->output_idx_map_.end()) continue;
    int entry_out_idx = parent_def_ctx->output_idx_map_[it.second];
    entry_def_ctx->output_idx_global_dev_map_[entry_out_idx] = def_ctx->instance_slice_ids_;
  }

  auto var_map = module->variable_map();
  for (int i = 0; i < signature.params.size(); ++i) {
    if (def_ctx->input_arg_map_.find(i) == def_ctx->input_arg_map_.end()) {
      def_ctx->sharded_args_.push_back(i);
      continue;
    }
    int parent_arg_no = def_ctx->input_arg_map_[i];
    if (parent_def_ctx->input_arg_map_.find(parent_arg_no) ==
        parent_def_ctx->input_arg_map_.end()) continue;
    int entry_arg_no = parent_def_ctx->input_arg_map_[parent_arg_no];
    if (var_map->find(entry_arg_no) != var_map->end()) {
      const HloInstruction* input = signature.params[i];
      def_ctx->input_activation_size_map_[i] = ShapeUtil::ByteSizeOf(input->shape());
      def_ctx->sharded_args_.push_back(i);
    }
  }
}

void StageDecomposition::BuildStagesForMainComputation(HloModule* module) {
  std::unordered_set<const HloInstruction*> prev_stages_insts;
  std::unordered_map<const HloInstruction*, DefInfo> insts_def_info;
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  HloModule::DefContext* sync_free_def_ctx = top_def_ctx->ComputeGradientsDefCtx();

  std::vector<DefInfo>& output_info = def_output_info_map_[sync_free_def_ctx->def_id()];
  HloComputation* sync_free = module->Def2Compute(sync_free_def_ctx);
  output_info.resize(sync_free->root_instruction()->operand_count());
  int count = 0;
  for (int s = 0; s < stage_count_; ++s) {
    CloneSignature sig = BuildStageSignatureForMainComputation(
        s, &prev_stages_insts, main_computation_, "SyncFree");
    HloComputation* stage_comp = CloneComputation(sig);
    
    HloModule::DefContext* stage_def_ctx = module->new_def_ctx(
        sig.name, HloModule::DefContext::DefType::CG_SLICE,
        sync_free_def_ctx->def_id()/*parent_id*/);

    sync_free_def_ctx->children_.push_back(stage_def_ctx);
    id_def_map_[stage_def_ctx->def_id()] = stage_def_ctx;

    stage_def_ctx->stage_type_ = s < stage_count_ / 2 ? 
                                 HloModule::DefContext::StageType::FORWARD :
                                 HloModule::DefContext::StageType::BACKWARD;
    int phy_s = s < (stage_count_ / 2) ? s : stage_count_ - s - 1;
    for (int parent_slice_id : sync_free_def_ctx->instance_slice_ids_) {
      std::vector<int64> addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(parent_slice_id);
      addr[split_ordinal_] = phy_s;
      int slice_id = comm_dev_mgr_->AddrToLinearIdxBySplitNums(addr); 
      stage_def_ctx->instance_slice_ids_.insert(slice_id);
    }

    SetupStageDefContext(module, sig, sync_free_def_ctx, s);
    InferDefMapForStage(sig.params, insts_def_info, sync_free_def_ctx, s, module);

    for (int i = 0; i < sig.root_operands.size(); ++i) {
      const HloInstruction* inst = sig.root_operands[i];
      insts_def_info[inst] = { stage_def_ctx->def_id(), i };
    }

    // Fill DefInfo for main_computation in order to provide enough information
    // for setting up other DefContext
    for (auto& it : stage_def_ctx->output_idx_map_) {
      output_info[it.second] = {stage_def_ctx->def_id(), it.first};
      ++count;
    }

    module->SetDefCompute(stage_def_ctx, stage_comp);
  }
  CHECK(count == sync_free->root_instruction()->operand_count());
}

HloModule::DefContext::DefType StageDecomposition::GetLowerLevelType(
    HloModule::DefContext* def_ctx) {
  switch (def_ctx->def_type()) {
    case HloModule::DefContext::DefType::CG:
      return HloModule::DefContext::DefType::CG_SLICE;
    case HloModule::DefContext::DefType::GA:
      return HloModule::DefContext::DefType::GA_SLICE;
    case HloModule::DefContext::DefType::AG:
      return HloModule::DefContext::DefType::AG_SLICE;
    default: {
      CHECK(0);
      return def_ctx->def_type();
    }
  }
}

std::vector<const HloInstruction*> StageDecomposition::ResolveSortedRootOpsFromInsts(
    const HloInstruction* root, std::vector<const HloInstruction*>& insts) {
  std::unordered_set<const HloInstruction*> root_ops;
  std::unordered_set<const HloInstruction*> visited;
  std::deque<const HloInstruction*> worklist;
  for (const HloInstruction* inst : insts) {
    worklist.push_back(inst);
    while (!worklist.empty()) {
      const HloInstruction* inst = worklist.front();
      worklist.pop_front();
      visited.insert(inst);
      for (const HloInstruction* user : inst->users()) {
        if (visited.find(user) != visited.end()) continue;
        if (root == user) {
          root_ops.insert(inst);
          continue;
        }
        worklist.push_back(user);
      }
    }
  }

  std::vector<const HloInstruction*> sorted_root_ops(
      root_ops.begin(), root_ops.end());
  std::sort(sorted_root_ops.begin(), sorted_root_ops.end(),
            [root] (const HloInstruction* lhs, const HloInstruction* rhs) {
              return root->operand_index(lhs) < root->operand_index(rhs);
            });
  return sorted_root_ops;
}

void StageDecomposition::BuildStagesForComputation(
    HloModule* module, HloModule::DefContext* parent_def_ctx) {
  std::map<int, std::vector<const HloInstruction*>, std::less<int>> stage_insts_map;
  HloComputation* computation = module->Def2Compute(parent_def_ctx);
  // In order to make the role of each DefContext clear, we should distinguish 
  // the `parent` and `prev`.
  
  // * `parent` describes the upper lever DefContext in DefContext tree.
  //    This relationship records in `input_arg_map_` and `output_idx_map_` member
  //
  // * `prev` describldes the sibling DecContext. This relationship records in
  //    `input_def_map_` member
  //
  // In this way, we can describe relationship of current DefContext  as follows:
  //
  //                            input_def_map
  //      parent_prev_def_ctx       <====        parent_def_ctx
  //         /          \                          /         \
  //  prev_def_ctx0   prev_def_ctx1    <====   def_ctx0    def_ctx1
  //
  for (auto& def_map_it : parent_def_ctx->input_def_map_) {
    int arg_no = def_map_it.first;
    auto& src_output_map = def_map_it.second;
    const HloInstruction* param = computation->parameter_instruction(arg_no);
    int stage = param->dist_spec().stage();
    CHECK(stage >= 0);
    stage_insts_map[stage].push_back(computation->parameter_instruction(arg_no));
  }

  std::vector<DefInfo>& output_info = def_output_info_map_[parent_def_ctx->def_id()];
  const HloInstruction* root = computation->root_instruction();
  output_info.resize(root->operand_count());

  // We use new assigned stage id for continuous number
  int assigned_stage = 0;
  std::unordered_set<const HloInstruction*> already_assigned;
  std::map<int, std::vector<const HloInstruction*>> stage_sorted_root_ops;
  std::vector<const HloInstruction*> cross_stage_insts;
  for (auto& iter : stage_insts_map) {
    std::vector<const HloInstruction*> sorted_root_ops = \
        ResolveSortedRootOpsFromInsts(root, iter.second);
    already_assigned.insert(sorted_root_ops.begin(), sorted_root_ops.end());
    if (assigned_stage == stage_insts_map.size() - 1) {
      for (const HloInstruction* inst : root->operands()) {
        if (already_assigned.find(inst) != already_assigned.end()) continue;
        cross_stage_insts.push_back(inst);
      }
    }
    stage_sorted_root_ops[iter.first] = sorted_root_ops;
    ++assigned_stage;
  }

  // We assign cross_stage_insts to all stages
  for (auto& iter : stage_sorted_root_ops) {
    for (const HloInstruction* inst : cross_stage_insts) {
      iter.second.push_back(inst);
    }
  }

  assigned_stage = 0;
  for (auto& iter : stage_sorted_root_ops) {
    int stage = iter.first;
    int phy_s = stage_count_ - 1 - stage;
    std::string stage_name = parent_def_ctx->name() + "_Stage_" + std::to_string(assigned_stage);

    // Now, we build clone signature and clone stage computation
    CloneSignature sig = BuildSignatureForStage(
        iter.second, nullptr, computation, stage_name);
    HloComputation* stage_comp = CloneComputation(sig);

    HloModule::DefContext* stage_def_ctx = module->new_def_ctx(
        sig.name, GetLowerLevelType(parent_def_ctx),
        parent_def_ctx->def_id()/*parent_id*/);

    stage_def_ctx->stage_type_ = HloModule::DefContext::StageType::BACKWARD;
    parent_def_ctx->children_.push_back(stage_def_ctx);
    id_def_map_[stage_def_ctx->def_id()] = stage_def_ctx;

    for (int parent_slice_id : parent_def_ctx->instance_slice_ids_) {
      std::vector<int64> addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(parent_slice_id);
      addr[split_ordinal_] = phy_s;
      int slice_id = comm_dev_mgr_->AddrToLinearIdxBySplitNums(addr); 
      stage_def_ctx->instance_slice_ids_.insert(slice_id);
    }

    SetupStageDefContext(module, sig, parent_def_ctx, assigned_stage);

    // Fill DefInfo for parent computation in order to provide enough information
    // for setting up other DefContext 
    for (auto& it : stage_def_ctx->output_idx_map_) {
      output_info[it.second] = {stage_def_ctx->def_id(), it.first};
    }

    InferDefMapFromParent(sig.params, stage_def_ctx, parent_def_ctx, phy_s, module);
    module->SetDefCompute(stage_def_ctx, stage_comp);
    ++assigned_stage;
  }
}

void StageDecomposition::InferDefMapFromParent(
    std::vector<const HloInstruction*>& stage_params,
    HloModule::DefContext* stage_def_ctx, HloModule::DefContext* parent_def_ctx,
    int phy_s, HloModule* module) {
  // Setup input_def_map for each stage. This implementation is generic version. It
  // infers relationships from its parent without any heuristic
  auto& parent_def_map = parent_def_ctx->input_def_map_;
  for (int i = 0; i < stage_params.size(); ++i) {
    int p_idx_in_parent = stage_params[i]->parameter_number();
    if (parent_def_map.find(p_idx_in_parent) == parent_def_map.end()) continue;
    for (int slice_id : parent_def_ctx->instance_slice_ids_) {
      std::vector<int64> parent_addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(slice_id);
      if (parent_addr[split_ordinal_] != phy_s) continue;
      int parent_arg_idx = stage_def_ctx->input_arg_map_[i];
      auto src_output_or = parent_def_ctx->get_src_output_from_input_def_map(parent_arg_idx, slice_id);
      if (src_output_or.ok()) {
        auto src_output = src_output_or.ValueOrDie();
        int parent_prev_slice_id = src_output.prev_slice_id;
        int parent_prev_def_id = src_output.def_id;
        int parent_out_idx = src_output.output_idx;
        std::vector<DefInfo>& def_output_info = def_output_info_map_[parent_prev_def_id];
        int pre_def_id = def_output_info[parent_out_idx].def_id;
        int out_idx = def_output_info[parent_out_idx].out_idx;
        stage_def_ctx->add_to_input_def_map(
            i, slice_id, parent_prev_slice_id, pre_def_id, out_idx);
      }
    }
  }
}

void StageDecomposition::InferDefMapForStage(
    std::vector<const HloInstruction*>& stage_params,
    std::unordered_map<const HloInstruction*, DefInfo>& insts_def_info,
    HloModule::DefContext* parent_def_ctx, int s, HloModule* module) {
  HloModule::DefContext* stage_def_ctx = parent_def_ctx->child(s);
  int phy_s = s < (stage_count_ / 2) ? s : stage_count_ - s - 1;
  // input_def_map_ cannot be infered from InferDefContextMembers, thus we infer
  // this member independently
  for (int i = 0; i < stage_params.size(); ++i) {
    const HloInstruction* input = stage_params[i];
    if (input->opcode() == HloOpcode::kParameter) continue;
    CHECK(insts_def_info.find(input) != insts_def_info.end());

    int p_out_idx = insts_def_info[input].out_idx;
    int p_def_id = insts_def_info[input].def_id;
    HloModule::DefContext* prev_def_ctx = id_def_map_[p_def_id];
    int prev_s = get_def_child_index(parent_def_ctx, prev_def_ctx);
    CHECK(prev_s >= 0);
    int prev_phy_s = prev_s < (stage_count_ / 2) ? prev_s : stage_count_ - prev_s - 1;

    for (int parent_slice_id : parent_def_ctx->instance_slice_ids_) {
      std::vector<int64> parent_addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(parent_slice_id);
      if (parent_addr[split_ordinal_] != phy_s) continue;
      int stage_slice_id = comm_dev_mgr_->AddrToLinearIdxBySplitNums(parent_addr);
      parent_addr[split_ordinal_] = prev_phy_s;
      int prev_stage_slice_id = comm_dev_mgr_->AddrToLinearIdxBySplitNums(parent_addr);
      stage_def_ctx->add_to_input_def_map(
          i, stage_slice_id, prev_stage_slice_id, p_def_id, p_out_idx);
    }
  }
}

void StageDecomposition::BuildStagesForLocalInit(HloModule* module) {
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  HloModule::DefContext* sync_free_def_ctx = top_def_ctx->ComputeGradientsDefCtx();
  HloModule::DefContext* la_init_def_ctx = top_def_ctx->GAInitDefCtx();
  HloModule::DefContext* la_def_ctx = top_def_ctx->GADefCtx();

  HloComputation* la_comp = module->Def2Compute(la_def_ctx);
  std::map<int, std::vector<const HloInstruction*>, std::less<int>> stage_insts_map;
  for (auto& def_map_it : la_def_ctx->input_def_map_) {
    int arg_no = def_map_it.first;
    auto& src_output_map = def_map_it.second;
    const HloInstruction* param = la_comp->parameter_instruction(arg_no);
    int stage = param->dist_spec().stage();
    CHECK(stage >= 0);
    stage_insts_map[stage].push_back(la_comp->parameter_instruction(arg_no));
  }

  std::vector<DefInfo>& output_info = def_output_info_map_[la_init_def_ctx->def_id()];
  const HloInstruction* root = la_comp->root_instruction();
  output_info.resize(root->operands().size());
  int assigned_stage = 0;
  for (auto& iter : stage_insts_map) {
    std::string name = "Gradients_Accumulation_GAInit_Stage_" + std::to_string(assigned_stage);
    HloModule::DefContext* la_stage_init_def_ctx = module->new_def_ctx(
        name, HloModule::DefContext::DefType::GAINIT_SLICE,
        la_init_def_ctx->def_id()/*parent_id*/);
    la_init_def_ctx->children_.push_back(la_stage_init_def_ctx);
    id_def_map_[la_stage_init_def_ctx->def_id()] = la_stage_init_def_ctx;

    la_stage_init_def_ctx->stage_type_ = HloModule::DefContext::StageType::BACKWARD;
    int stage = iter.first;
    int phy_s = stage_count_ - 1 - stage;
    // Infer stage of each root instruction's operand in computation
    std::vector<const HloInstruction*> sorted_root_ops = \
        ResolveSortedRootOpsFromInsts(root, iter.second);

    std::vector<HloInstruction*> tuple_operands;
    HloComputation::Builder builder(name);
    for (int64 i = 0; i < sorted_root_ops.size(); ++i) {
      auto new_instr = HloInstruction::CreateParameter(
          i, sorted_root_ops[i]->shape(), sorted_root_ops[i]->name() + ".GA.init");
      tuple_operands.push_back(new_instr.get());
      builder.AddInstruction(std::move(new_instr));
      output_info[root->operand_index(sorted_root_ops[i])] = { la_stage_init_def_ctx->def_id(), i };
    }
    
    std::unique_ptr<HloInstruction> new_root = HloInstruction::CreateTuple(tuple_operands);
    builder.AddInstruction(std::move(new_root));
    std::unique_ptr<HloComputation> result = builder.Build(new_root.get());
    HloComputation* computation = result.get();

    module->SetDefCompute(la_stage_init_def_ctx, computation);
    module->AddEmbeddedComputation(std::move(result));

    for (int parent_slice_id : la_init_def_ctx->instance_slice_ids_) {
      std::vector<int64> addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(parent_slice_id);
      addr[split_ordinal_] = phy_s;
      int slice_id = comm_dev_mgr_->AddrToLinearIdxBySplitNums(addr); 
      la_stage_init_def_ctx->instance_slice_ids_.insert(slice_id);
    }
    ++assigned_stage;
  }
}

void StageDecomposition::BuildStagesForLocalAccumulation(HloModule* module) {
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  HloModule::DefContext* local_accum_def_ctx = top_def_ctx->GADefCtx();
  BuildStagesForComputation(module, local_accum_def_ctx);
}

void StageDecomposition::BuildStagesForNonSyncFree(HloModule* module) {
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  HloModule::DefContext* remains_ctx = top_def_ctx->ApplyGradientsDefCtx();
  BuildStagesForComputation(module, remains_ctx);
}

void StageDecomposition::InitializeDefMap(HloModule* module) {
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  HloModule::DefContext* sync_free_def_ctx = top_def_ctx->ComputeGradientsDefCtx();
  HloModule::DefContext* local_accum_def_ctx = top_def_ctx->GADefCtx();
  HloModule::DefContext* local_init_def_ctx = top_def_ctx->GAInitDefCtx();
  HloModule::DefContext* remains_def_ctx = top_def_ctx->ApplyGradientsDefCtx();
  id_def_map_[top_def_ctx->def_id()] = top_def_ctx;
  id_def_map_[sync_free_def_ctx->def_id()] = sync_free_def_ctx;
  id_def_map_[local_accum_def_ctx->def_id()] = local_accum_def_ctx;
  id_def_map_[remains_def_ctx->def_id()] = remains_def_ctx;
  id_def_map_[local_init_def_ctx->def_id()] = local_init_def_ctx;
}

void StageDecomposition::PrintCommunicationInfo(HloModule* module) {
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  HloModule::DefContext* sync_free_def_ctx = top_def_ctx->ComputeGradientsDefCtx();
  std::unordered_map<int/*def_id*/, int/*stage_id*/> def_stage_map;
  std::vector<std::vector<int64>> total_statistics;
  bool debug = ServiceEnv::debug();
  if (debug)
    VLOG(0) << "====================Communication Report========================";
  for (int s = 0; s < sync_free_def_ctx->num_children(); ++s) {
    HloModule::DefContext* stage_def_ctx = sync_free_def_ctx->child(s);
    def_stage_map[stage_def_ctx->def_id()] = s;
    HloComputation* comp = module->Def2Compute(stage_def_ctx);
    std::vector<int64> bytes_from_stage(s);
    if (debug) VLOG(0) << "\tStage " << s << " details";
    for (auto& iter : stage_def_ctx->input_def_map_) {
      int arg_no = iter.first;
      int prev_def_id = -1;
      int output_idx = -1;
      int prev_s = -1;
      for (auto src_iter : iter.second) {
        prev_def_id = src_iter.second.def_id;
        output_idx = src_iter.second.output_idx;
        prev_s = def_stage_map[prev_def_id];
        break;
      }

      CHECK(prev_def_id != -1);
      CHECK(output_idx != -1);
      CHECK(prev_s != -1);

      HloInstruction* param = comp->parameter_instruction(arg_no);
      int64 bytes = ShapeUtil::ByteSizeOf(param->shape());
      bytes_from_stage[prev_s] += bytes;
      if (debug) {
        VLOG(0) << "\t\t Stage " << prev_s << " at output " << output_idx
                << " -----> arg " << arg_no << " : bytes " << HumanReadableNumBytes(bytes);
      }
    }

    total_statistics.push_back(bytes_from_stage);
  }

  VLOG(0) << "=======================Total Statistics=========================";

  int64 total_bytes = 0;
  int64 effective_bytes = 0;
  for (int s = 0; s < total_statistics.size(); ++s) {
    std::vector<int64>& bytes_from_stage = total_statistics[s];
    VLOG(0) << "\t Stage " << s << " info";
    for (int p = 0; p < bytes_from_stage.size(); ++p) {
      if (bytes_from_stage[p] == 0) continue;
      VLOG(0) << "\t\t Stage " << p << " -----> Stage " << s
              << ": total bytes " << HumanReadableNumBytes(bytes_from_stage[p]);
      total_bytes += bytes_from_stage[p];
      if (s + p != sync_free_def_ctx->num_children() - 1) effective_bytes += bytes_from_stage[p];
    }
  }

  VLOG(0) << "Total bytes = " << total_bytes
          << " (" << HumanReadableNumBytes(total_bytes) << ")";
  VLOG(0) << "Effective bytes = " << effective_bytes
          << " (" << HumanReadableNumBytes(effective_bytes) << ")";
}

std::map<HloInstruction*, CrossStageInfo> StageDecomposition::CollectCrossStageInsts(
    HloModule* module, std::unordered_map<int, int>& def_stage_map) {
  std::map<HloInstruction*, CrossStageInfo> cross_stage_insts;
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  HloModule::DefContext* sync_free_def_ctx = top_def_ctx->ComputeGradientsDefCtx();
  
  for (int s = 0; s < sync_free_def_ctx->num_children(); ++s) {
    HloModule::DefContext* stage_def_ctx = sync_free_def_ctx->child(s);
    def_stage_map[stage_def_ctx->def_id()] = s;
    HloComputation* comp = module->Def2Compute(stage_def_ctx);
    for (auto& iter : stage_def_ctx->input_def_map_) {
      int arg_no = iter.first;
      int prev_def_id = -1;
      int output_idx = -1;
      int prev_s = -1;
      for (auto src_iter : iter.second) {
        prev_def_id = src_iter.second.def_id;
        output_idx = src_iter.second.output_idx;
        prev_s = def_stage_map[prev_def_id];
      }
      CHECK(prev_def_id != -1);
      CHECK(output_idx != -1);
      CHECK(prev_s != -1);

      HloModule::DefContext* prev_def_ctx = sync_free_def_ctx->child(prev_s);
      HloComputation* prev_comp = module->Def2Compute(prev_def_ctx);
      HloInstruction* prev_root = prev_comp->root_instruction();
      HloInstruction* src = prev_root->mutable_operand(output_idx);
      if (cross_stage_insts.find(src) == cross_stage_insts.end()) {
        CrossStageInfo cross_info;
        cross_info.produced_stage = prev_s;
        cross_stage_insts[src] = cross_info;
      }

      CrossStageInfo& cross_info = cross_stage_insts.at(src);
      cross_info.consumed_stage_arg_pair[s] = arg_no;
    }
  }

  return cross_stage_insts;
}

void StageDecomposition::CrossStageTransferOptimization(HloModule* module) {
  HloModule::DefContext* top_def_ctx = module->def_ctx();
  HloModule::DefContext* sync_free_def_ctx = top_def_ctx->ComputeGradientsDefCtx();
  std::unordered_map<int/*def_id*/, int/*stage_id*/> def_stage_map;
  for (int s = 0; s < sync_free_def_ctx->num_children(); ++s) {
    HloModule::DefContext* stage_def_ctx = sync_free_def_ctx->child(s);
    def_stage_map[stage_def_ctx->def_id()] = s;
  }

  std::map<HloInstruction*, CrossStageInfo> cross_stage_insts = \
      CollectCrossStageInsts(module, def_stage_map);
  for (auto& it : cross_stage_insts) {
    HloInstruction* src = it.first;
    CrossStageInfo& stage_info = it.second;
    int src_stage = stage_info.produced_stage;
    auto& stage_arg_map = stage_info.consumed_stage_arg_pair;
    if (stage_arg_map.size() == 1) {
      const auto& it = stage_arg_map.begin();
      int dst_stage = it->first;
      if (src_stage == dst_stage - 1 ||
          src_stage + dst_stage == sync_free_def_ctx->num_children() - 1) continue;
    }

    int max_stage = -1;
    for (const auto& stage_arg_it : stage_arg_map) {
      max_stage = std::max(max_stage, stage_arg_it.first);
    }
    HloModule::DefContext* src_stage_def_ctx = sync_free_def_ctx->child(src_stage);
    HloComputation* src_comp = module->Def2Compute(src_stage_def_ctx);
    HloInstruction* src_root = src_comp->root_instruction();

    int param_in_out_idx = src_root->operand_index(src);
    for (int s = src_stage + 1; s <= max_stage; ++s) {
      int phy_s = s < (stage_count_ / 2) ? s : stage_count_ - s - 1;
      int prev_s = s - 1;
      int prev_phy_s = prev_s < (stage_count_ / 2) ? prev_s : stage_count_ - prev_s - 1;
      HloModule::DefContext* stage_def_ctx = sync_free_def_ctx->child(s);
      HloModule::DefContext* prev_stage_def_ctx = sync_free_def_ctx->child(prev_s);
      HloComputation* comp = module->Def2Compute(stage_def_ctx);
      HloComputation* prev_comp = module->Def2Compute(prev_stage_def_ctx);
      std::vector<HloInstruction*> tuple_elements;
      int param_no = -1;
      HloInstruction* param = nullptr;
      if (stage_arg_map.find(s) == stage_arg_map.end()) {
        // pass through
        param_no = comp->num_parameters();
        param = comp->AddParameter(
            HloInstruction::CreateParameter(param_no, src->shape(), src->name()));

        HloInstruction* root = comp->root_instruction();
        for (int op_idx = 0; op_idx < root->operand_count(); ++op_idx) {
          tuple_elements.push_back(root->mutable_operand(op_idx));
        }
        tuple_elements.push_back(param);
      } else {
        // No need to recreate parameters
        param_no = stage_arg_map[s];
        stage_def_ctx->input_def_map_.erase(param_no);
        param = comp->parameter_instruction(param_no); 
        HloInstruction* root = comp->root_instruction();
        for (int op_idx = 0; op_idx < root->operand_count(); ++op_idx) {
          tuple_elements.push_back(root->mutable_operand(op_idx));
        }
        tuple_elements.push_back(param);
      }

      for (int slice_id : stage_def_ctx->instance_slice_ids_) {
        std::vector<int64> addr = comm_dev_mgr_->LinearIdxToAddrBySplitNums(slice_id);
        addr[split_ordinal_] = prev_phy_s;
        int prev_slice_id = comm_dev_mgr_->AddrToLinearIdxBySplitNums(addr);
        stage_def_ctx->add_to_input_def_map(
            param_no, slice_id, prev_slice_id, prev_stage_def_ctx->def_id(), param_in_out_idx);
      }


      VLOG(1) << "src : " << src->name()
              << ", from stage " << prev_s << ", at " << param_in_out_idx
              << " ------> " << s << ", at " << param_no;

      if (s < max_stage) {
        HloInstruction* new_root = comp->AddInstruction(HloInstruction::CreateTuple(tuple_elements));
        comp->ReplaceInstructionNoShapeCheck(comp->root_instruction(), new_root);
        comp->set_root_instruction(new_root);

        param_in_out_idx = tuple_elements.size() - 1;
        stage_def_ctx->sharded_args_.push_back(param_no);
        stage_def_ctx->input_activation_size_map_[param_no] = ShapeUtil::ByteSizeOf(param->shape());
        stage_def_ctx->output_tensor_size_map_[param_in_out_idx] = ShapeUtil::ByteSizeOf(param->shape());
      }
    }
  }
}

StatusOr<bool> StageDecomposition::Run(HloModule* module) {
  if (!ResolveComputations(module)) return true;

  comm_dev_mgr_ = std::make_unique<CommDevManager>(
      module->split_nums(), module->share_dev_flags(),
      module->placement_layout(), module->stage_split_ordinal(), 1/*num_workers not use*/);

  split_ordinal_ = module->stage_split_ordinal();

  InitializeDefMap(module);
  ResolveTotalStageCount();

  if (stage_count_ < 1) return true;

  BuildStagesForMainComputation(module);
  CrossStageTransferOptimization(module);
  PrintCommunicationInfo(module);
  if (local_accum_) {
    BuildStagesForLocalInit(module);
    BuildStagesForLocalAccumulation(module);
  }

  if (non_sync_free_) {
    BuildStagesForNonSyncFree(module);
  }

  VLOG(2) << "[StageDecomposition] " << module->ToString();
  return true;
}

} // namespace xla
