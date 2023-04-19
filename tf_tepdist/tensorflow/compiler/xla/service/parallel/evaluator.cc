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

#include "tensorflow/compiler/xla/service/parallel/evaluator.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/parallel/performance_utils.h"

#include <float.h>

namespace xla {

const string kAllToAllType = "AllToAll";
const string kAllReduceType = "AllReduce";
const string kAllGatherType = "AllGather";

const std::string StrategySpec::ToString() const {
  std::string str = "[Strategy] ";
  if (sync_free_enabled) {
    str += "SyncFree enabled, micro_batch num = " + std::to_string(num_micro_batches);
  }

  if (spmd_enabled) {
    str += ", SPMD enabled, mesh = [";
    for (int d : used_spmd_mesh) {
      str += " " + std::to_string(d);
    }
    str += " ]";
  }

  if (pipeline_enabled) {
    str += ", Pipeline enabled, num_stages = " + std::to_string(num_stages);
  }

  return str;
}

const std::string Cost::ToString() const {
  std::string str = "[Cost] ";
  str += "gpu efficiency = " + std::to_string(gpu_efficiency);
  str += ", collective comm ratio = " + std::to_string(coll_ratio);
  str += ", bubble ratio = " + std::to_string(bubble_ratio);
  str += ", duration = " + std::to_string(total_duration);
  return str;
}

Evaluator::Evaluator(HloModule* module, int64 local_dev_num,
                     int64 worker_num, double gpu_power_ms,
                     int64 max_bytes_per_device, double usage_ratio,
                     double inter_bw, double intra_bw)
  : module_(module), local_dev_num_(local_dev_num),
    worker_num_(worker_num), gpu_power_ms_(gpu_power_ms),
    max_bytes_per_device_(max_bytes_per_device), usage_ratio_(usage_ratio),
    inter_bw_(inter_bw), intra_bw_(intra_bw),
    comm_dev_mgr_(module->split_nums(), module->share_dev_flags(),
                  module->placement_layout(), module->stage_split_ordinal(), worker_num) {}

int64 Evaluator::EstimateCommBytes(HloComputation* comp) {
  int64 total_bytes = 0;
  for (const HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() != HloOpcode::kCustomCollective) continue;
    auto custom_collective_instr = DynCast<HloCustomCollectiveInstruction>(instr);
    auto& collective_type = custom_collective_instr->collective_type();
    int64 comm_bytes = 0;
    if (collective_type == kAllReduceType) {
      comm_bytes = PerfUtils::AllReduceCost(instr);
    } else if (collective_type == kAllGatherType) {
      comm_bytes = PerfUtils::AllGatherCost(instr);
    } else if (collective_type == kAllToAllType) {
      comm_bytes = PerfUtils::AllToAllCost(instr);
    }
    total_bytes += comm_bytes;
  }

  return total_bytes;
}

double Evaluator::EstimateCommTime(
    HloModule::DefContext* def_ctx, HloComputation* comp,
    const std::vector<int>& split_nums) {
  double comm_time = 0;
  for (const HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() != HloOpcode::kCustomCollective) continue;

    bool cross_worker = false;
    int split_ordinal = instr->split_ordinal();
    int num_replicas = split_nums[split_ordinal];
    auto dev_groups_arr = comm_dev_mgr_.FindDevGroupArray(split_ordinal);
    CHECK(dev_groups_arr);
    for (auto& dev_group : dev_groups_arr->dev_groups_) {
      std::unordered_set<int64> assigned_workers;
      for (int64 dev_id : dev_group->global_dev_ids_) {
        assigned_workers.insert(comm_dev_mgr_.worker_id(dev_id));
      }

      if (assigned_workers.size() > 1) {
        cross_worker = true;
      }
    }

    auto custom_collective_instr = DynCast<HloCustomCollectiveInstruction>(instr);
    auto& collective_type = custom_collective_instr->collective_type();
    int64 comm_bytes = 0;
    if (collective_type == kAllReduceType) {
      comm_bytes = PerfUtils::AllReduceCost(instr);
    } else if (collective_type == kAllGatherType) {
      comm_bytes = PerfUtils::AllGatherCost(instr);
    } else if (collective_type == kAllToAllType) {
      comm_bytes = PerfUtils::AllToAllCost(instr);
    }

    comm_time += cross_worker ? comm_bytes / inter_bw_ : comm_bytes / intra_bw_;
  }

  return comm_time;
}

Cost Evaluator::Run(HloModule* module, StrategySpec& spec) {
  Cost cost;
  double total_cost = 0;
  HloModule::DefContext* entry_def_ctx = module->def_ctx();
  if (spec.pipeline_enabled) {
    // 1. Memory constraint
    // 2. Pipeline length Estimation (TODO @siyu.wsy, use Evaluator instead)
    std::vector<double> stage_duration;
    std::vector<double> stage_transfer_bytes(2 * spec.num_stages, 0);
    float pure_total_comp_time = 0.0;
    float pure_total_coll_time = 0.0;
    for (auto& it : module->DefComputation()) {
      if (!it.first->cg_def_ctx()) continue;
      std::unordered_map<int/*def_id*/, int/*stage*/> def_stage_map;
      for (int i = 0; i < it.first->num_children(); ++i) {
        HloModule::DefContext* def_ctx = it.first->child(i);
        def_stage_map[def_ctx->def_id()] = i;
        HloComputation* stage_comp = module->Def2Compute(def_ctx);
        double stage_comp_time = PerfUtils::CalculateFlops(stage_comp) / gpu_power_ms_;
        double coll_comm_time = EstimateCommTime(def_ctx, stage_comp, module->split_nums());
        pure_total_comp_time += stage_comp_time;
        pure_total_coll_time += coll_comm_time;
        stage_duration.push_back(stage_comp_time + coll_comm_time);
        auto& input_def_map = def_ctx->input_def_map_;
        for (auto& def_it : input_def_map) {
          bool cross_stage = false;
          for (auto& src_it : def_it.second) {
            int prev_def_id = src_it.second.def_id;
            if (def_stage_map.find(prev_def_id) == def_stage_map.end()) continue;
            if (def_stage_map[prev_def_id] + i != it.first->num_children() - 1) {
              cross_stage = true;
              break;
            }
          }

          if (!cross_stage) continue;
          int param_no = def_it.first;
          HloInstruction* param = stage_comp->parameter_instruction(param_no);
          stage_transfer_bytes[i] += ShapeUtil::ByteSizeOf(param->shape());
        }

        int64 memory = 0;
        for (auto* p : stage_comp->parameter_instructions()) {
          memory += ShapeUtil::ByteSizeOf(p->shape());
        }

        if (memory >= usage_ratio_ * max_bytes_per_device_) {
          cost.total_duration = FLT_MAX;
          return cost;
        }
      }
    }

    std::vector<double> forward_end_time(spec.num_stages, 0);
    // a. Forward estimation 
    for (int m = 0; m < spec.num_micro_batches; ++m) {
      double prev_stage_end_time = 0;
      for (int s = 0; s < spec.num_stages; ++s) {
        // Use inter bandwidth to estimate cross stage data transfer
        // (TODO@siyu.wsy) Support intra bandwidth estimation
        forward_end_time[s] = \
            std::max(prev_stage_end_time + stage_transfer_bytes[s] / inter_bw_,
                     forward_end_time[s]) + stage_duration[s];
        prev_stage_end_time = forward_end_time[s];
      }
    }
    // b. Backward estimation
    std::vector<double> backward_end_time(spec.num_stages, 0);
    for (int m = 0; m < spec.num_micro_batches; ++m) {
      double prev_stage_end_time = forward_end_time.back();
      for (int s = 0; s < spec.num_stages; ++s) {
        // Use inter bandwidth to estimate cross stage data transfer
        // (TODO@siyu.wsy) Support intra bandwidth estimation
        int logical_stage = s + spec.num_stages;
        backward_end_time[s] = \
            std::max(prev_stage_end_time + stage_transfer_bytes[logical_stage] / inter_bw_,
                     backward_end_time[s]) + stage_duration[logical_stage];
        prev_stage_end_time = backward_end_time[s];
      }
    }
    float pipeline_efficiency = \
        pure_total_comp_time * spec.num_micro_batches / (backward_end_time.back() * spec.num_stages);
    float coll_ratio = \
        pure_total_coll_time * spec.num_micro_batches / (backward_end_time.back() * spec.num_stages);

    cost.total_duration = backward_end_time.back();
    cost.gpu_efficiency = pipeline_efficiency;
    cost.coll_ratio = coll_ratio;
    cost.bubble_ratio = 1 - cost.gpu_efficiency - cost.coll_ratio;
  } else {
    double comp_time = 0;
    double comm_time = 0;
    int64 total_bytes = 0;
    if (spec.sync_free_enabled) {
      HloModule::DefContext* sync_free_def_ctx = entry_def_ctx->ComputeGradientsDefCtx();
      HloComputation* comp = module->Def2Compute(sync_free_def_ctx);
      int64 memory = 0;
      for (auto* p : comp->parameter_instructions()) {
        memory += ShapeUtil::ByteSizeOf(p->shape());
      }

      if (memory >= usage_ratio_ * max_bytes_per_device_) {
        cost.total_duration = FLT_MAX;
        return cost;
      }
 
      comp_time += spec.num_micro_batches * PerfUtils::CalculateFlops(comp) / gpu_power_ms_;
      comm_time += EstimateCommTime(sync_free_def_ctx, comp, module->split_nums());
      total_bytes += EstimateCommBytes(comp);
      // We add a small constant to sync_free strategy to
      // distinguish its quanlity with no sync free strategy
      comp_time += 10;
    } else {
      HloComputation* comp = module->entry_computation();

      int64 memory = 0;
      for (auto* p : comp->parameter_instructions()) {
        memory += ShapeUtil::ByteSizeOf(p->shape());
      }

      if (memory >= usage_ratio_ * max_bytes_per_device_) {
        cost.total_duration = FLT_MAX;
        return cost;
      }
      comp_time += PerfUtils::CalculateFlops(comp) / gpu_power_ms_;
      comm_time += EstimateCommTime(entry_def_ctx, comp, module->split_nums());
      total_bytes += EstimateCommBytes(comp);
    }
    
    total_cost = comp_time + comm_time;
    cost.total_duration = total_cost;
    cost.gpu_efficiency = comp_time / total_cost;
    cost.coll_ratio = 1 - cost.gpu_efficiency;
  }

  return cost;
}

} // namespace xla
