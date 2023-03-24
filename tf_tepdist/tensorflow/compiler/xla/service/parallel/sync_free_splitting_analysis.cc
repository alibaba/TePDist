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

#include "tensorflow/compiler/xla/service/parallel/sync_free_splitting_analysis.h"

#include "tensorflow/compiler/xla/service/parallel/utils.h"
#include "tensorflow/compiler/xla/service/service_env.h"

namespace xla {

namespace {

void DFSVisit(
    std::vector<HloInstruction*> instrs, int index,
    std::vector<int>& path, std::vector<std::vector<int>>& proposals) {
  if (index == instrs.size()) {
    proposals.push_back(path);
    return;
  }

  HloInstruction* p = instrs[index];
  for (int r = 0; r < p->shape().rank(); ++r) {
    path.push_back(r);
    DFSVisit(instrs, index + 1, path, proposals);
    path.pop_back();
  }  
}

} // namespace

std::vector<HloInstruction*> SyncFreeSplittingAnalysis::ExtractSampleInputs(HloModule* module) {
  std::map<int, string>* vars_map = module->variable_map();
  HloComputation* entry = module->entry_computation();
  std::vector<HloInstruction*> sample_inputs;
  for (HloInstruction* param : entry->parameter_instructions()) {
    if (ShapeUtil::IsScalar(param->shape()) ||
        param->parameter_number() >= entry->num_parameters() - vars_map->size()) continue;
    sample_inputs.push_back(param);
  }
  return sample_inputs;
}

std::vector<std::vector<int>> SyncFreeSplittingAnalysis::GenerateProposals(
    std::vector<HloInstruction*>& instrs) {
  std::vector<std::vector<int>> proposals;
  std::vector<int> path;
  DFSVisit(instrs, 0, path, proposals);
  return proposals;
}

HloInstMap<DimStrategy>
SyncFreeSplittingAnalysis::SearchForMostSyncFreeInsts(HloModule* module) {
  VLOG(2) << "[SearchForMostSyncFreeInsts] : " << module->ToString();
  HloInstMap<DimStrategy> best_strategy_map;
  std::vector<HloInstruction*> sample_inputs = ExtractSampleInputs(module);
  if (sample_inputs.empty()) return best_strategy_map;
  std::vector<std::vector<int>> proposals = GenerateProposals(sample_inputs);
  int64 lower_bound = 2;
  int64 upper_bound = INT_MAX;
  if (num_micro_batches_) {
    // Force to use num_micro_batches
    CHECK(num_micro_batches_ >= 2)
        << "num_micro_batches is smaller than 2! "
        << "num_micro_batches = " << num_micro_batches_;
    lower_bound = upper_bound = num_micro_batches_;
  } else {
    lower_bound = std::max(2 * num_stages_ - 1, (int64)2); // DAPPLE experience
    for (HloInstruction* instr : sample_inputs) {
      int64 max_dim_val = instr->shape().dimensions(0);
      for (int d = 0; d < instr->shape().dimensions_size(); ++d) {
        max_dim_val = std::max(instr->shape().dimensions(d), max_dim_val);
      }
      VLOG(1) << "max_dim_val = " << max_dim_val;
      // We find the minmal boundary over all shape dimensions
      upper_bound = std::min(max_dim_val, upper_bound);
    }
  }
  HloComputation* entry = module->entry_computation();
  int total_count = entry->instruction_count();

  // Greedy search for micro_batch_num from lower bound to upper bound
  bool found = false;
  VLOG(0) << "Try micro_batch_num from " << lower_bound << " to " << upper_bound;
  for (int micro_batch_num = lower_bound;
           (micro_batch_num <= upper_bound) && !found; ++micro_batch_num) {
    VLOG(0) << "Try micro_batch_num = " << micro_batch_num;
    HloInstMap<DimStrategy> hlo_strategy_map;
    for (HloInstruction* param : sample_inputs) {
      hlo_strategy_map.insert(
          std::make_pair(param, DimStrategy(param->shape(), -1, micro_batch_num)));
    }

    int max_count = 0;
    for (int p = 0; p < proposals.size(); ++p) {
      bool divisible = true;
      std::vector<int>& proposal = proposals[p];
      HloInstMap<DimStrategy> tmp_strategy_map = hlo_strategy_map;
      VLOG(1) << "Exploring proposal " << p;
      for (int i = 0; i < proposal.size(); ++i) {
        VLOG(2) << "\ti = " << i << ", split at " << proposal[i];
      }

      for (int i = 0; i < proposal.size(); ++i) {
        HloInstruction* param = sample_inputs[i];
        int split_dim = proposal[i];
        tmp_strategy_map[param] = DimStrategy(param->shape(), split_dim, micro_batch_num);
        if (tmp_strategy_map[param].Glue()) {
          divisible = false;
          break;
        }
      }

      if (!divisible) {
        VLOG(1) << "Proposal " << p << " is invalid for micro_batch_num "
                << micro_batch_num << " is not divisible";
        continue;
      }

      if (StrategyUtil::InferGraph(entry, tmp_strategy_map)) {
        int count = StrategyUtil::CountSplitInsts(tmp_strategy_map);
        VLOG(0) << "Proposal " << p << " is valid, count = " << count
                << ", progress = " << 1.0 * count / total_count;
        if (count > max_count) {
	        max_count = count;
          best_strategy_map = tmp_strategy_map;
          num_micro_batches_ = micro_batch_num;
          found = true;
        }
      } else {
        int count = StrategyUtil::CountSplitInsts(tmp_strategy_map);
        VLOG(1) << "Proposal " << p << " is invalid, count = " << count;
      }
    }
  }

  return best_strategy_map;
}

void SyncFreeSplittingAnalysis::RecordStrategyToInsts(HloModule* module,
    HloInstMap<DimStrategy>& strategy_map) {
  int num_replicas = num_micro_batches_;
  auto* entry = module->entry_computation();
  for (auto* instr : entry->MakeInstructionPostOrder()) {
    int64 stride = 0;
    int64 stride_on_dim = 0;
    int num_replicas = 1;
    int partition_dim = -1;
    bool partial = false;
    if (strategy_map.find(instr) != strategy_map.end()) {
      auto& strategy = strategy_map.at(instr);
      stride = strategy.stride_on_elements();
      stride_on_dim = strategy.stride_on_dim();
      partition_dim = strategy.partition_dim();
      num_replicas = strategy.num_replicas();
      partial = strategy.IsPartial();
    }

    std::unique_ptr<DimDistSpec> dim_spec = std::make_unique<DimDistSpec>();

    dim_spec->set_partial(partial);
    dim_spec->set_layout_aware_partition(
        stride, stride_on_dim, partition_dim, num_replicas);

    instr->mutable_dist_spec()->AddDimDistSpec(dim_spec);

    VLOG(2) << "[SyncFreeSplittingAnalysis] inst with strategy: " << instr->ToString();
  }
}

bool SyncFreeSplittingAnalysis::Validate(
    HloModule* module, HloInstMap<DimStrategy>& strategy_map) {
  auto* entry = module->entry_computation();
  std::unordered_set<HloInstruction*> partial_set;
  std::unordered_set<HloInstruction*> descendants;
  std::unordered_set<HloInstruction*> precedants;
  for (auto* instr : entry->MakeInstructionPostOrder()) {
    if (strategy_map.find(instr) != strategy_map.end()) {
      auto& strategy = strategy_map.at(instr);
      if (strategy.IsPartial()) partial_set.insert(instr);
    }
  }

  // 1. Find all precedants
  for (auto* partial_inst : partial_set) {
    std::deque<HloInstruction*> worklist = {partial_inst};
    while (!worklist.empty()) {
      auto* instr = worklist.front();
      worklist.pop_front();
      for (auto* op : instr->operands()) {
        if (precedants.find(op) == precedants.end() &&
            partial_set.find(op) == partial_set.end()) {
          worklist.push_back(op);
          precedants.insert(op);
        }
      }
    }
  }

  // 2. Others are descendants
  for (auto* instr : entry->MakeInstructionPostOrder()) {
    if (precedants.find(instr) == precedants.end()) {
      descendants.insert(instr);
    }
  }

  // 3. Validate if there exists cycle
  for (auto* instr : descendants) {
    for (auto* user : instr->users()) {
      if (precedants.find(user) != precedants.end() ||
          partial_set.find(user) != partial_set.end()) return false;
    }
  }
  return true;
}


StatusOr<bool> SyncFreeSplittingAnalysis::Run(HloModule* module) {
  auto strategy_map = SearchForMostSyncFreeInsts(module);
  if (strategy_map.empty()) return false;

  if (!Validate(module, strategy_map)) {
    VLOG(0) << "[SyncFreeSplittingAnalysis] Validation failed. SyncFree disable";
    return false;
  }
  
  module->record_split_info(num_micro_batches_, true/*share_dev_flag*/);
  RecordStrategyToInsts(module, strategy_map);
  return true;
}


} // namespace xla
