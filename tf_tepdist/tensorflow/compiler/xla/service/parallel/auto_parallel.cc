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

#include "tensorflow/compiler/xla/service/parallel/auto_parallel.h"

#include "absl/container/node_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_graph_sketch.h"
#include "tensorflow/compiler/xla/service/parallel/cost_spmd_strategy.h"
#include "tensorflow/compiler/xla/service/parallel/custom_collective_expander.h"
#include "tensorflow/compiler/xla/service/parallel/fast_spmd_strategy.h"
#include "tensorflow/compiler/xla/service/parallel/spmd_transform.h"
#include "tensorflow/compiler/xla/service/parallel/stage_decomposition.h"
#include "tensorflow/compiler/xla/service/parallel/sync_free_decomposition.h"
#include "tensorflow/compiler/xla/service/parallel/sync_free_splitting_analysis.h"
#include "tensorflow/compiler/xla/service/parallel/utils.h"
#include "tensorflow/compiler/xla/service/service_env.h"

#include <deque>

namespace xla {

AutoParallel::AutoParallel(int64 num_worker, int64 local_dev_num)
  : worker_num_(num_worker)
  , local_dev_num_(local_dev_num) {
  CHECK(num_worker > 0 && local_dev_num > 0);

  num_micro_batches_ = ServiceEnv::num_micro_batches();
  num_stages_ = ServiceEnv::num_stages();

  fast_mode_ = ServiceEnv::rule_mode();
}

void AutoParallel::SetupInputOutputAliasMap(HloModule* module) {
  CHECK(module->def_ctx());
  auto def_ctx = module->def_ctx();
  int var_count = module->variable_map()->size();
  HloComputation* entry = module->entry_computation();
  const HloInstruction* root = entry->root_instruction();
  int input_var_offset = entry->num_parameters() - var_count;
  int output_var_offset = entry->root_instruction()->operand_count() - var_count;
  for (int i = 0; i < var_count; ++i) {
    int p_idx = input_var_offset + i;
    int out_idx = output_var_offset + i;
    const HloInstruction* param = entry->parameter_instruction(p_idx);
    const HloInstruction* updated = root->operand(out_idx);
    if (!ShapeUtil::Equal(param->shape(), updated->shape())) CHECK(0 && "Never reach here");
    def_ctx->input_output_alias_map_[p_idx] = out_idx;
  }
}

void AutoParallel::SetupEntryDefContext(HloModule* module) {
  if (!module->def_ctx()) {
    module->set_def_ctx(module->new_def_ctx(
        "Entry", HloModule::DefContext::DefType::ENTRY, -1/*parent_id*/));
  }
  auto def_ctx = module->def_ctx();
  CHECK(def_ctx);
  def_ctx->module_ = module;

  auto entry = module->entry_computation();
  if (!module->Def2Compute(def_ctx)) {
    module->SetDefCompute(def_ctx, entry);
  }

  auto& var_args = def_ctx->sharded_args_;
  std::unordered_set<int> input_variable_args(var_args.begin(),
                                              var_args.end());
  auto var_map = module->variable_map();  // trainable variables
  for (auto input : entry->parameter_instructions()) {
    int arg_no = input->parameter_number();
    if (var_map->count(arg_no) && !input_variable_args.count(arg_no)) {
      // TODO(shiqing.fsq): sharded_args actually records trainable variables.
      def_ctx->sharded_args_.push_back(arg_no);
    }

    if (def_ctx->input_arg_map_.count(arg_no)) {
      CHECK(def_ctx->input_arg_map_[arg_no] == arg_no);
    } else {
      def_ctx->input_arg_map_[arg_no] = arg_no; // Identity mapping

      if (var_map->find(arg_no) != var_map->end()) {
        // record input tensor size
        int64 result_bytes = ShapeUtil::ByteSizeOf(input->shape());

        // TODO(lansong): tensor size of input activation and trainable vars should
        // be recorded in input_activation_size_map_ and input_var_size_map_
        // respectively.
        CHECK(def_ctx->input_var_size_map_.find(arg_no) ==
              def_ctx->input_var_size_map_.end());
        def_ctx->input_var_size_map_[arg_no] = result_bytes;
      }
    }
  } 

  auto root = entry->root_instruction();
  for (int64 out_idx = 0; out_idx < root->operand_count(); ++out_idx) {
    const HloInstruction* op = root->operand(out_idx);
    if (def_ctx->output_idx_map_.count(out_idx)) {
      CHECK(def_ctx->output_idx_map_[out_idx] == out_idx);
    } else {
      def_ctx->output_idx_map_[out_idx] = out_idx; // Identity mapping
      def_ctx->output_tensor_size_map_[out_idx] = ShapeUtil::ByteSizeOf(op->shape());
      def_ctx->output_idx_global_dev_map_[out_idx].insert(0);
    }
  }

  def_ctx->instance_slice_ids_.insert(0);

  // NOTE & TODO (zycao): We found input output buffer alias optimization might
  // cause minor digital error under some cases, which occured under some fused
  // cases with deep level broadcast and constant fusion with other many insts.
  // To validate strict digital percision for baseline tests, we add an environ
  // switch for this optimization. Maybe we could try to find whether the issue
  // could be basically eliminated later.
  bool disable_alias = ServiceEnv::disable_buffer_alias();
  if (!disable_alias) SetupInputOutputAliasMap(module);
}

std::vector<DeviceSplitPlan> AutoParallel::GenerateSplitProposals() {
  int total_dev_num = local_dev_num_ * worker_num_;

  std::deque<std::pair<int/*dev_num*/, std::vector<int>/*split_devs*/>> queue;
  queue.push_back(std::make_pair(total_dev_num, std::vector<int>()));
  // Currently support no more than three layers decomposition
  for (int i = 0; i < 3; ++i) {
    int size = queue.size();
    for (int l = 0; l < size; ++l) {
      std::pair<int, std::vector<int>> dev_path_pair = queue.front();
      int dev_num = dev_path_pair.first;
      std::vector<int>& path = dev_path_pair.second;
      queue.pop_front();
      for (int d = 1; d <= dev_num; d *= 2) {
        if (dev_num % d) continue;
        if (i > 0 && d * d > dev_num) break;
        std::vector<int> added_path(path.begin(), path.end());
        added_path.push_back(d);
        queue.push_back(std::make_pair(dev_num / d, added_path));
      }
    }
  }

  // Post-processing modifies all decomposition paths, forcing three
  // decompositions to get total_dev_num
  absl::node_hash_set<DeviceSplitPlan> visited;
  std::vector<DeviceSplitPlan> proposals;
  while (!queue.empty()) {
    std::pair<int, std::vector<int>> dev_path_pair = queue.front();
    queue.pop_front();
    int dev_num = dev_path_pair.first;
    std::vector<int>& path = dev_path_pair.second;
    path.back() = path.back() * dev_num;
    DeviceSplitPlan plan;
    plan.num_stages = path.front();
    plan.spmd_mesh.insert(plan.spmd_mesh.end(), path.begin() + 1, path.end());
    // Currently disable multiple mesh split proposals
    if (visited.find(plan) != visited.end() ||
        plan.IsMeshMultipleSplit()) continue;
    visited.insert(plan);
    proposals.push_back(plan);
    if (plan.num_stages == 1) {
      // Push another plan
      plan.use_local_accumulation = false;
      proposals.push_back(plan);
    }
  }

  return proposals;
}

void AutoParallel::DeepCopyHloModule(HloModule* src, HloModule* dst) {
  CHECK(src);
  // Now, we deep copy picked_module to best_module
  dst->Cleanup();
  // Remove all computations
  int computation_count = dst->computation_count();
  for (int i = computation_count - 1; i >= 0; --i) {
    dst->RemoveEmbeddedComputation(dst->mutable_computation(i));
  }

  std::unordered_map<HloComputation*, HloComputation*> replacements;
  for (auto* c : src->computations()) {
    auto* new_comp = dst->DeepCloneComputation(c);
    if (c->IsEntryComputation()) dst->set_entry(new_comp);
    replacements[c] = new_comp;
  }

  dst->StealDefCtx(src, replacements);
  CHECK(dst->def_ctx());
  std::unordered_map<HloComputation*, HloComputation*> replace_map;
  for (int i = 0; i < computation_count; ++i) {
    replace_map[src->mutable_computation(i)] = dst->mutable_computation(i);
  }

  dst->ReplaceComputations(replace_map);
  *dst->mutable_entry_computation_layout() =
      src->compute_computation_layout();
  dst->CopyMetaData(src);
}

void AutoParallel::DoSyncFreeDecomposition(HloModule* module) {
  for (int s = 0;  s < module->share_dev_flags().size(); ++s) {
    if (module->share_dev_flags()[s]) {
      std::unordered_set<const HloInstruction*> sync_points = \
          StrategyUtil::CollectAllSyncPoints(module);
      SyncFreeChain sync_chain(sync_points);
      SyncFreeDecomposition sync_free_decomp(s);
      sync_free_decomp.Run(module, sync_chain);
      break;
    }
  }
}

StatusOr<bool> AutoParallel::RunFastMode(HloModule* module) {
  VLOG(0) << "[old version] fast mode 1";
  // run dp/sharding inference in a single pass
  AnnotFastSpmdStrategy spmd;
  TF_RETURN_IF_ERROR(spmd.Run(module).status());
  SpmdTransform spmd_transform(0);
  TF_RETURN_IF_ERROR(spmd_transform.Run(module).status());
  return true;
}

StatusOr<bool> AutoParallel::RunExplorationlMode(HloModule* module) {
  VLOG(0) << "Run exploration mode";
  std::vector<DeviceSplitPlan> proposals = GenerateSplitProposals();

  std::unique_ptr<HloModule> best_clone_module;
  double min_cost = FLT_MAX;
  StrategySpec best_spec;
  for (int i = 0; i < proposals.size(); ++i) {
    std::unique_ptr<HloModule> clone_module_ptr = module->Clone();
    HloModule* clone_module = clone_module_ptr.get();
    SetupEntryDefContext(clone_module);
    bool use_local_accumulation = proposals[i].use_local_accumulation;
    std::vector<int>& spmd_mesh = proposals[i].spmd_mesh;
    int split_ordinal = 0;

    StrategySpec spec;
    spec.sync_free_enabled = false;
    spec.spmd_enabled = false;
    spec.pipeline_enabled = false;
    spec.num_stages = proposals[i].num_stages;

    // 1. SyncFree Extraction(Enumerate dividend num_batches)
    if (use_local_accumulation) {
      SyncFreeSplittingAnalysis sync_analysis(spec.num_stages);
      StatusOr<bool> change_or = sync_analysis.Run(clone_module);
      spec.sync_free_enabled = change_or.ValueOrDie();
      if (spec.sync_free_enabled) {
        spec.num_micro_batches = sync_analysis.num_micro_batches();
        SpmdTransform spmd_transform(split_ordinal);
        TF_ASSIGN_OR_RETURN(bool status, spmd_transform.Run(clone_module));
        if (!status) continue;
        ++split_ordinal;
      }
    }

    // 2. FastSpmdStrategy Search (Switch on to debug multiple split)
    bool mesh_succ = true;
    for (int d = 0; d < spmd_mesh.size(); ++d) {
      if (spmd_mesh[d] == 1) continue;
      VLOG(0) << "d = " << d << ", spmd_mesh[d] = " << spmd_mesh[d];
      CostSpmdStrategy auto_strategy(split_ordinal, spmd_mesh[d], spec.num_stages,
                                     d==spmd_mesh.size() - 1);
      TF_ASSIGN_OR_RETURN(mesh_succ, auto_strategy.Run(clone_module));
      if (!mesh_succ) break; 
      SpmdTransform spmd_transform(split_ordinal);
      TF_ASSIGN_OR_RETURN(mesh_succ, spmd_transform.Run(clone_module));
      if (!mesh_succ) break;
      spec.spmd_enabled = true;
      spec.used_spmd_mesh.push_back(spmd_mesh[d]);
      ++split_ordinal;
    }

    if (!mesh_succ) continue;

    // 3. Pipeline Search
    if (spec.sync_free_enabled && spec.num_stages > 1) {
      std::unique_ptr<GraphSketch> graph_sketch =
          GraphSketch::BuildFineGrainedSketch(clone_module, spec.num_stages);
      spec.pipeline_enabled = graph_sketch->StagePlan();
    }

    // 4. SyncFree Decomposition
    if (spec.sync_free_enabled) DoSyncFreeDecomposition(clone_module);
 
    // 5. Stage Decomposition
    if (spec.pipeline_enabled) {
      StageDecomposition stage_decomp;
      stage_decomp.Run(clone_module);
    }

    if (local_dev_num_ * worker_num_ != clone_module->total_dev_num()) continue;
    Evaluator evaluator(clone_module, local_dev_num_, worker_num_);
    Cost cost = evaluator.Run(clone_module, spec);
    if (ServiceEnv::debug()) {
      VLOG(0) << "[Candidates] spec: " << spec.ToString() << ", " << cost.ToString();
    }
    if (cost.total_duration < min_cost) {
      min_cost = cost.total_duration;
      best_spec = std::move(spec);
      best_clone_module.swap(clone_module_ptr);
    }
  }

  if (best_clone_module) {
    DeepCopyHloModule(best_clone_module.get(), module);
    VLOG(0) << best_spec.ToString();
  }
  return true;
}

StatusOr<bool> AutoParallel::RunConfiglMode(HloModule* module) {
  VLOG(0) << "Run config mode";
  int total_dev_num = local_dev_num_ * worker_num_;
  std::vector<int> spmd_mesh;
  if (num_stages_ > 1 && !(total_dev_num % num_stages_)) {
    spmd_mesh.push_back(total_dev_num / num_stages_);
  } else {
    spmd_mesh.push_back(total_dev_num);
  }

  StrategySpec spec;
  spec.sync_free_enabled = false;
  spec.spmd_enabled = false;
  spec.pipeline_enabled = false;

  int split_ordinal = 0;
  // 1. Try SyncFree Exploration
  if (num_micro_batches_ > 1) {
    SyncFreeSplittingAnalysis sync_analysis(
        0/*num_stages, no use here*/, num_micro_batches_);
    StatusOr<bool> change_or = sync_analysis.Run(module);
    spec.sync_free_enabled = change_or.ValueOrDie();
    CHECK(spec.sync_free_enabled) << "SyncFree failed!";
    if (spec.sync_free_enabled) {
      spec.num_micro_batches = sync_analysis.num_micro_batches();
      SpmdTransform spmd_transform(split_ordinal);
      TF_ASSIGN_OR_RETURN(bool succ, spmd_transform.Run(module));
      CHECK(succ) << "SyncFree Transformation failed!";
      ++split_ordinal;
    }
  }

  // 2. Try SPMDSearch for current mesh configuration
  bool mesh_succ = true;
  for (int d = 0; d < spmd_mesh.size(); ++d) {
    if (spmd_mesh[d] == 1) continue;
    CostSpmdStrategy auto_strategy(
        split_ordinal, spmd_mesh[d], num_stages_, true/*mem_save*/);
    TF_ASSIGN_OR_RETURN(mesh_succ, auto_strategy.Run(module));
    SpmdTransform spmd_transform(split_ordinal);
    TF_ASSIGN_OR_RETURN(mesh_succ, spmd_transform.Run(module));
    CHECK(mesh_succ) << "SPMD planning or transformation failed!";
    spec.spmd_enabled = true;
    spec.used_spmd_mesh.push_back(spmd_mesh[d]);
    ++split_ordinal;
  }

  // 3. Pipeline Search
  if (spec.sync_free_enabled && num_stages_ > 1) {
    std::unique_ptr<GraphSketch> graph_sketch =
        GraphSketch::BuildFineGrainedSketch(module, num_stages_);
    spec.pipeline_enabled = graph_sketch->StagePlan();
    CHECK(spec.pipeline_enabled) << "Pipeline failed!";
  }

  // 4. SyncFree Decomposition
  if (spec.sync_free_enabled) DoSyncFreeDecomposition(module);

  // 5. Stage Decomposition
  if (spec.pipeline_enabled) {
    spec.num_stages = num_stages_;
    StageDecomposition stage_decomp;
    stage_decomp.Run(module);
  }

  VLOG(0) << spec.ToString();
  return true;
}

StatusOr<bool> AutoParallel::Run(HloModule* module) {
  SetupEntryDefContext(module);
  if (fast_mode_) {
    TF_RETURN_IF_ERROR(RunFastMode(module).status());
  } else if (num_micro_batches_ || num_stages_) {
    TF_RETURN_IF_ERROR(RunConfiglMode(module).status());
  } else {
    TF_RETURN_IF_ERROR(RunExplorationlMode(module).status());
  }

  // Expand custom-collectives
  CustomCollectiveExpander expander;
  TF_RETURN_IF_ERROR(expander.Run(module).status());
  return true;
}

} // namespace xla
