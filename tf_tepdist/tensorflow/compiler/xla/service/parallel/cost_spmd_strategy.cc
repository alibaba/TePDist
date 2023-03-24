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

#include "coin/CbcModel.hpp"
#include "coin/OsiClpSolverInterface.hpp"

#include "tensorflow/compiler/xla/service/parallel/cost_spmd_strategy.h"
#include "tensorflow/compiler/xla/service/parallel/performance_utils.h"
#include "tensorflow/compiler/xla/service/parallel/resolve_utils.h"
#include "tensorflow/compiler/xla/service/parallel/utils.h"
#include "tensorflow/compiler/xla/service/hlo_graph_sketch.h"
#include "tensorflow/compiler/xla/service/hlo_instruction_util.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/service_env.h"
#include "tensorflow/core/platform/numbers.h"

namespace xla {

namespace {

using ::tensorflow::strings::HumanReadableNumBytes;



bool IdenticalConcat(HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kConcatenate);

  bool equal = true;
  int64 num_operand_elems = -1;
  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    auto operand = hlo->mutable_operand(i);
    int64 num_elems = ShapeUtil::ElementsIn(operand->shape());
    if (num_operand_elems > 0 && 
        num_operand_elems != num_elems) {
      equal = false;
      break;
    }
    num_operand_elems = num_elems;
  }
  return equal;
}

const char* const kCudnnConvForwardCallTarget = "__cudnn$convForward";
const char* const kCudnnConvBackwardInputCallTarget =
    "__cudnn$convBackwardInput";
const char* const kCudnnConvBackwardFilterCallTarget =
    "__cudnn$convBackwardFilter";
const char* const kCudnnConvBiasActivationForwardCallTarget =
    "__cudnn$convBiasActivationForward";

bool IsConvolution(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCudnnConvForwardCallTarget ||
         target == kCudnnConvBackwardInputCallTarget ||
         target == kCudnnConvBackwardFilterCallTarget ||
         target == kCudnnConvBiasActivationForwardCallTarget;
}

int64 Cost(const DimStrategy& src, const DimStrategy& usr,
           const Shape& tensor_shape, const int split_num, const bool share_dev,
           const bool tuple_src) {
  int64 cost=0;
  if (share_dev) {
    return cost;
  }

  // TODO(lansong)
  //    communication efficiency should be a factor in cost evaluation
  int64 data_bytes = ShapeUtil::ByteSizeOf(tensor_shape, 8);
  if (src.IsPartial()) {
    // 1. allreduce
    // currently allreduce cost is evaluated by AllReduceCost, we should merge it later
    // not update
  } else if (src.Glue()) {
    if (!usr.Glue()) {
      // 2. dynamic slice
      cost += 10;  // lansong: a trick for dynamic slice cost
    }
  } else if (!usr.Match(src)) {
    float cost_factor = ServiceEnv::cost_factor();
    if (usr.Glue()) {
      // 3. allgather
      cost = PerfUtils::AllGatherCost(data_bytes, split_num);
    } else {
      // 4. all2all
      cost = PerfUtils::AllToAllCost(data_bytes, split_num);
      cost = cost * cost_factor;
    }
  }

  if ((!usr.Match(src)) && tuple_src) {
    //  TODO(lansong)
    //  temporary solution for strategy affinity constraint,
    //  we should move affinity setting to ILP model building
    // 
    //         x     y
    //          \   /
    //          tuple(cone root)
    //            |
    //          -----
    //         |     |
    //        gte0  gte1
    //         |     |
    //
    //  zero cost constraint between tuple and gte
    cost = INT64_MAX;
    VLOG(2) << "tuple src: " << src.ToString() << ", usr: " << usr.ToString() << ", cost: " << cost;
  }

  return cost;
}

int64 DataTransferSize(const HloInstMap<SharedDimStrategy>& strategy_map,
                       const HloInstSet& scope,
                       const int split_num) {
  int64 transfer_size = 0;
  std::set<std::pair<const HloInstruction*, DimStrategy>> visited;
  for (auto& inst_strtg : strategy_map) {
    auto* inst = inst_strtg.first;
    if (strategy_map.find(inst) != strategy_map.end()) {
      if (strategy_map.at(inst)->IsPartial()) {
        // inst self cost
        int64 inst_bytes = ShapeUtil::ByteSizeOf(inst->shape(), 8);
        transfer_size += PerfUtils::AllReduceCost(inst_bytes, split_num);  // ring allreduce cost
      }
    }

    // evaluate operands
    for (int i = 0; i < inst->operand_count(); ++i) {
      const HloInstruction* op = inst->operand(i);
      if (scope.find(op) == scope.end()) {
        continue;
      }

      bool tuple_src = false;
      if (op->shape().IsTuple()) {
        tuple_src = true;
      }

      SharedDimStrategy expected_op_strategy =
          StrategyUtil::BackInfer(inst, *inst_strtg.second, i);

      std::pair<const HloInstruction*, DimStrategy> key(op, *expected_op_strategy);
      if (visited.find(key) != visited.end()) continue;
      if (strategy_map.find(op) == strategy_map.end()) {
        // not consider cost here
      } else {
        // check strategy conflict inside the piece
        int slice_num = expected_op_strategy->NumSlices();
        int64 data_cost = Cost(*strategy_map.at(op), *expected_op_strategy,
                               inst->shape(), slice_num, false, tuple_src);   // TODO(lansong): consider share device later
        transfer_size += data_cost;
      }
      visited.insert(key);
    }
  }

  return transfer_size;
}

struct ConeStrategyConn {
  ConeStrategyConn(int input_cone_id,
                   int cone_id,
                   int input_str_id,
                   int cone_str_id)
  : input_cone_id_(input_cone_id)
  , cone_id_(cone_id)
  , input_str_id_(input_str_id)
  , cone_str_id_(cone_str_id) {
    conn_name_ = "c" + std::to_string(input_cone_id);
    conn_name_ += "to" + std::to_string(cone_id);
    conn_name_ += "_s" + std::to_string(input_str_id);
    conn_name_ += "to" + std::to_string(cone_str_id);
  }
  int input_cone_id_;   // source cone id
  int cone_id_;         // target cone id
  int input_str_id_;    // source strategy id
  int cone_str_id_;     // target strategy id

  std::string conn_name_;
};

};

std::string InstCosts::ToString() const {
  CHECK(inst_);
  std::string res = inst_->name() + ":\n";

  int64 min_cost = INT64_MAX;
  int64 max_cost = 0;
  if (strtg_costs_.size() > 1) {
    res += "*** mult strategies ***\n";
  }
  for (auto& one_cost : strtg_costs_) {
    CHECK(one_cost);
    if (min_cost > one_cost->cost_) {
      min_cost = one_cost->cost_;
    }
    if (max_cost < one_cost->cost_) {
      max_cost = one_cost->cost_;
    }
    res += "(" + std::to_string(one_cost->cost_) + ", \""
           + one_cost->strategy_->ToString() + "\", [";

    for (auto idx : one_cost->op_strtg_idx_) {
      res += std::to_string(idx) + ", ";
    }
    res += "])\n";
  }

  res += "min cost: " + std::to_string(min_cost) + ", max cost: "
         + std::to_string(max_cost) + "\n";

  res += "is cone root: ";
  res += is_cone_root_ ? "true" : "false";

  return std::move(res);
}

void ConeStrategy::BuildExpInStrtgMap(
    const std::unordered_set<const HloInstruction*>& insts) {
  for (auto* inst : insts) {
    for (int i = 0; i < inst->operand_count(); ++i) {
      const HloInstruction* op = inst->operand(i);
      if (insts.find(op) != insts.end()) {
        continue;
      }

      if (strategy_map_.find(inst) != strategy_map_.end()) {
        exp_in_strtgs_[op] = StrategyUtil::BackInfer(inst, *strategy_map_[inst], i);
        // TODO(lansong): only record last one if the same "op" is recorded before
      } else {
        exp_in_strtgs_[op] = std::make_shared<DimStrategy>();
        // TODO(lansong): only record last one if the same "op" is recorded before
      }
    }
  }
}

void ConeStrategy::BuildSelfCost(const InstCone& cone, bool share_dev, int split_num) {
  if (share_dev) {
    self_cost_ = 0;
    return;
  }
  HloInstSet scope(cone.insts_.begin(), cone.insts_.end());
  self_cost_ = DataTransferSize(strategy_map_, scope, split_num);
}

void ConeStrategy::BuildInputCost(
    const InstCone& cone,
    const HloInstMap<SharedDimStrategy>& ctx_strtg_map,
    const std::unordered_map<const HloInstruction*, std::shared_ptr<InstCone>>& cone_map,
    int split_num,
    bool share_dev) {
  BuildExpInStrtgMap(cone.insts_);

  for (auto& exp_in_str : exp_in_strtgs_) {
    const HloInstruction* input_inst = exp_in_str.first;
    SharedDimStrategy& exp_str = exp_in_str.second;

    bool tuple_src = false;
    if (input_inst->shape().IsTuple()) {
      tuple_src = true;
    }

    if (cone_map.find(input_inst) != cone_map.end()) {
      // update inter-cone costs
      std::shared_ptr<InstCone> pred_cone = cone_map.at(input_inst);
      for (auto& pred_str : pred_cone->strategies()) {
        std::pair<int, int> key = std::make_pair(pred_cone->id_, pred_str->id_);
        if (cone_in_costs_.find(key) == cone_in_costs_.end()) {
          cone_in_costs_[key] = 0;
        }

        const SharedDimStrategy& in_str = pred_str->FindStrategy(input_inst);

        int64 data_cost = Cost(*in_str, *exp_str, input_inst->shape(), split_num,
                               share_dev, tuple_src);
        cone_in_costs_[key] += data_cost;
      }
    } else if (ctx_strtg_map.find(input_inst) != ctx_strtg_map.end()) {
      // update cone context costs
      const SharedDimStrategy& in_str = ctx_strtg_map.at(input_inst);

      int64 data_cost = Cost(*in_str, *exp_str, input_inst->shape(), split_num,
                             share_dev, tuple_src);
      ctx_in_cost_ += data_cost;
    } else {
      // input instruction is not splitted
      // no cost
    }
  }
}

void ConeStrategy::BuildOutToCtxCost(
    const InstCone& cone,
    const HloInstMap<SharedDimStrategy>& ctx_strtg_map,
    const std::unordered_map<const HloInstruction*, std::shared_ptr<InstCone>>& cone_map,
    int split_num,
    bool share_dev) {
  ctx_out_cost_ = 0;
  if (share_dev) {
    return;
  }
  auto* output_inst = cone.root_inst_;
  CHECK(strategy_map_.find(output_inst) != strategy_map_.end());
  const SharedDimStrategy& out_str = strategy_map_[output_inst];

  bool tuple_src = false;
  if (output_inst->shape().IsTuple()) {
    tuple_src = true;
  }

  for (auto* user : output_inst->users()) {
    // NOTE(zycao): since tuple instructions (e.g, OUTPUT is always tuple) would
    // be commonly set into context parts with Glue strategy. The cost with its
    // preceding instruction strategies should not be counted. 
    if (user->opcode() == HloOpcode::kTuple) continue;
    if (cone.insts_.find(user) != cone.insts_.end()) {
      // not an outside instruction
      continue;
    }

    if (cone_map.find(user) != cone_map.end()) {
      // feed to another cone, skip
      continue;
    }

    if (ctx_strtg_map.find(user) != ctx_strtg_map.end()) {
      int idx = 0;
      for (; idx < user->operand_count(); ++idx) {
        if (user->operand(idx) == output_inst) {
          break;
        }
      }
      CHECK(idx<user->operand_count());
      SharedDimStrategy exp_str =
                  StrategyUtil::BackInfer(user, *ctx_strtg_map.at(user), idx);

      int64 data_cost = Cost(*out_str, *exp_str, output_inst->shape(), split_num,
                             false, tuple_src);
      VLOG(2) << "build output ctx cost: " << output_inst->name() << ", " << out_str->ToString()
              << " // to " << user->ToString() << ", " << exp_str->ToString();
      ctx_out_cost_ += data_cost;
      VLOG(2) << "ctx out cost: " << ctx_out_cost_ << ", " << data_cost;
    }
  }
}

std::string ConeStrategy::ToString() const {
  std::string res = "  [self cost]: " + std::to_string(self_cost_) + "\n";

  res += "  [strategies]:\n";
  for (auto& inst_strtg : strategy_map_) {
    res += "    " + inst_strtg.first->name() + ":\n      "
           + inst_strtg.second->ToString() + "\n";
  }

  res += "  [expected input strategies]:\n";
  for (auto& exp_in_strtg : exp_in_strtgs_) {
    res += "    " + exp_in_strtg.first->name() + ":\n      "
           + exp_in_strtg.second->ToString() + "\n";
  }

  res += "  [cone input costs]:\n";
  for (auto& cone_in_cost : cone_in_costs_) {
    res += "    input cone id: " + std::to_string(cone_in_cost.first.first);
    res += ", input strategy id: " + std::to_string(cone_in_cost.first.second);
    res += ", input cost: " + std::to_string(cone_in_cost.second);
    res += "\n";
  }

  return std::move(res);
}

std::string InstCone::ToString() const {
  std::string res = "************\ncone id: " + std::to_string(id_)
                    + "\n************\n";
  res += "root:\n  " + root_inst_->name() + "\ninst list:\n";
  for (auto* inst : insts_) {
    res += "  " + inst->name() + "\n";
  }
  res += "\nstrategy num: " + std::to_string(strategies_.size()) + "\n";
  for (int i=0; i<strategies_.size(); ++i) {
    res += "strategy " + std::to_string(i) + ":\n";
    res += strategies_[i]->ToString() + "\n";
  }
  res += "\n";

  return std::move(res);
}

void InstCone::CollectInputInsts() {
  input_insts_.clear();

  for (auto* inst : insts_) {
    for (auto* op : inst->operands()) {
      if (insts_.find(op) == insts_.end()) {
        input_insts_.insert(op);
      }
    }
  }
}

void InstCone::BuildSelfCost(bool share_dev, int split_num) {
  for (auto& cone_str : strategies_) {
    cone_str->BuildSelfCost(*this, share_dev, split_num);
  }
}

void InstCone::BuildInputCost(
    const HloInstMap<SharedDimStrategy>& ctx_strtg_map,
    const std::unordered_map<const HloInstruction*, std::shared_ptr<InstCone>>& cone_map,
    int split_num,
    bool share_dev) {
  CollectInputInsts();

  for (auto& cone_str : strategies_) {
    cone_str->BuildInputCost(*this, ctx_strtg_map, cone_map, split_num, share_dev);
  }
}

void InstCone::BuildOutToCtxCost(
    const HloInstMap<SharedDimStrategy>& ctx_strtg_map,
    const std::unordered_map<const HloInstruction*, std::shared_ptr<InstCone>>& cone_map,
    int split_num,
    bool share_dev) {
  for (auto& cone_str : strategies_) {
    cone_str->BuildOutToCtxCost(*this, ctx_strtg_map, cone_map, split_num, share_dev);
  }
}

void GraphPiece::EvalScore(const HloInstSet& scope, int split_num) {
  if (score_>0.0) {
    // score is evaluated
    return;
  }

  // 1. calculate flop count
  int64 flop_count = 0;
  for (auto& inst_strtg : compute_map_) {
    auto* inst = inst_strtg.first;
    flop_count += PerfUtils::CalculateFlops(inst);
  }

  // 2. evaluate boundary tensor size
  int64 transfer_size = DataTransferSize(inst_map_, scope, split_num);

  // 3. calculate computing/communication ratio
  if (transfer_size == 0) {
    transfer_size = 1;
  }
  score_ = (float)flop_count/(float)transfer_size;
}

bool
GraphPiece::CutBy(std::shared_ptr<GraphPiece> piece,
                  std::vector<std::shared_ptr<GraphPiece>>& new_pieces) {
  int orig_size = inst_map_.size();
  for (auto& inst_strtg : piece->inst_map_) {
    const HloInstruction* inst = inst_strtg.first;
    inst_map_.erase(inst);
    compute_map_.erase(inst);
  }

  if (inst_map_.size() == orig_size) {
    // no instruction removed
    return false;
  }

  score_ = 0.0;
  return true;
}

void
GraphPieces::CutBy(std::shared_ptr<GraphPiece> piece) {
  std::list<std::shared_ptr<GraphPiece>>::iterator it = pieces_.begin();
  std::vector<std::shared_ptr<GraphPiece>> all_new_pieces;
  while (it != pieces_.end()) {
    std::vector<std::shared_ptr<GraphPiece>> new_pieces;
    bool changed = (*it)->CutBy(piece, new_pieces);
    if (changed) {
      all_new_pieces.insert(all_new_pieces.end(),
                             new_pieces.begin(),
                             new_pieces.end());
      if ((*it)->compute_map_.empty()) {
        it = pieces_.erase(it);
      } else {
        ++it;
      }
    } else {
      ++it;
    }
  }

  pieces_.insert(pieces_.end(), all_new_pieces.begin(), all_new_pieces.end());
}

bool GraphStrategy::SetInstStrategy(const HloInstruction* inst,
                                    const SharedDimStrategy& strtg) {
  HloInstMap<SharedDimStrategy>& cur_str_map = strategy_map();
  if (cur_str_map.find(inst) != cur_str_map.end()) {
    return false;
  }

  cur_str_map[inst] = strtg;

  // try to evaluate cost
  if (strtg->Glue()) {
    return true;
  }

  return true;
}

void
GraphStrategy::Finalize() {
  VLOG(1) << "[GraphStrategy::Finalize] entry";
  VLOG(1) << "[GraphStrategy::Finalize] dim strategy map size: " << strategy_map_.size();
  for (auto& inst_dim_str : strategy_map_) {
    const HloInstruction* inst = inst_dim_str.first;
    SharedDimStrategy& dim_str = inst_dim_str.second;

    hlo_strategy_map_[inst].AddDimStrategy(*dim_str);
  }

  VLOG(1) << "[GraphStrategy::Finalize] hlo_strategy_map_ size: " << hlo_strategy_map_.size();
}

std::string
GraphStrategy::ToString() {
  std::string res("graph strategies:");
  for (auto& inst_dim_str : strategy_map_) {
    const HloInstruction* inst = inst_dim_str.first;
    SharedDimStrategy& dim_str = inst_dim_str.second;

    res += "\ninst: " + inst->ToString();
    res += "\nstrategy: " + dim_str->ToString();
  }

  return std::move(res);
}

StatusOr<bool> CostSpmdStrategy::Run(HloModule* module) {
  local_dev_num_ = module->num_dev_per_worker();
  worker_num_ = module->num_worker();
  CHECK(local_dev_num_ * worker_num_ >= 1)
      << "local_dev_num: " << local_dev_num_ << "worker_num: " << worker_num_;

  RewriteCustomCallsAsDots(module);

  if (module->variable_map()->empty()) {
    // Skip this pass when there is no trainable variable.
    LOG(INFO) << "variable map is not setup!";
    return false;
  }

  VLOG(2) << "module before strategy planning: " << module->ToString();
  VLOG(0) << "[recorded in module] worker num: " << worker_num_
          << ", dev num per worker: " << local_dev_num_
          << ", cur_split_num_: " << cur_split_num_;

  if (!StrategyPlanning(module)) {
    VLOG(0) << "[SPMD] strategy planning failed!";
    return false;
  }

  module->record_split_info(cur_split_num_, false);

  return true;
}

HloInstMap<SharedDimStrategy>
CostSpmdStrategy::ExtractUserSplit(const HloModule* module) {
  HloInstMap<SharedDimStrategy> user_annotated_tensors;

  bool ignore_annotation = ServiceEnv::ignore_annotation();
  if (ignore_annotation) {
    // return empty map
    return std::move(user_annotated_tensors);
  }

  // collect all annotations
  VLOG(1) << "start ExtractUserSplit";
  auto entry = module->entry_computation();
  CHECK(entry != nullptr);
  auto post_order = entry->MakeInstructionPostOrder();
  VLOG(2) << "inst with annotation:";
  auto num_replicas = cur_split_num_;
  for (auto* inst : post_order) {
    if (inst->has_sharding()) {
      VLOG(2) << "annotated inst:";
      VLOG(2) << inst->ToString();
      const HloSharding& sharding = inst->sharding();
      VLOG(2) << sharding.ToString();
      auto& tile_assignment = sharding.tile_assignment();
      if (sharding.IsReplicated()) {
        user_annotated_tensors[inst] = DimStrategy::MakeReplicate(num_replicas);
        continue;
      }

      int total_dev_num = tile_assignment.num_elements();
      VLOG(2) << "total_dev_num=" << total_dev_num;
      // Single device case where tensor is not sharded.
      if (sharding.IsTileMaximal()) {
        if (num_replicas != total_dev_num) {
          LOG(INFO) << "Mismatched split number, one is '" << num_replicas
                    << "', another one is '" << total_dev_num
                    << "' in annotations.";
          user_annotated_tensors.clear();
          return std::move(user_annotated_tensors);
        }
        continue;
      }

      for (auto dev : tile_assignment) {
        VLOG(2) << "tile assign: " << dev;
        std::vector<int64> tile_idx_for_dev = sharding.TileIndexForDevice(dev);
        for (auto tile_idx : tile_idx_for_dev) {
          VLOG(2) << "  tile_idx: " << tile_idx;
        }
      }

      int partition_dim = -1;
      for (int d = 0; d < tile_assignment.num_dimensions(); ++d) {
        if (tile_assignment.dim(d) > 1) {
          if (partition_dim>=0) {
            VLOG(2) << "Only one dimension can be splitted, but found at least two.";
            user_annotated_tensors.clear();
            return std::move(user_annotated_tensors);
          } else {
            partition_dim = d;
          }
        }
      }
      VLOG(2) << "partition_dim=" << partition_dim;

      CHECK(partition_dim>=0) << "partition_dim=" << partition_dim;

      if (num_replicas != total_dev_num) {
        VLOG(2) << "Mismatched split number, one is '" << num_replicas
                << "', another one is '" << total_dev_num
                << "' in annotations.";
        user_annotated_tensors.clear();
        return std::move(user_annotated_tensors);
      }

      user_annotated_tensors[inst] =
                    std::make_shared<DimStrategy>(inst->shape(), partition_dim, num_replicas);
    }
  }

  if (user_annotated_tensors.empty()) {
    VLOG(1) << "No split is specified.";
  } else {
    VLOG(1) << "user annotations:";
    for (auto& annotation : user_annotated_tensors) {
      VLOG(1) << "user specified inst: " << annotation.first->ToString();
      VLOG(1) << "user specified strategy: " << annotation.second->ToString();
    }
  }

  return std::move(user_annotated_tensors);
}

void
CostSpmdStrategy::InferUsersFromInst(
    const HloInstruction* inst,
    const DimStrategy& inst_strategy,
    const HloInstSet& stop_inst_set,  // stop instruction will be inferred
    const HloInstSet& inst_scope,
    const HloInstMap<SharedDimStrategy>& expected_strategies,
    GraphStrategy& graph_strategy,
    HloInstSet& infered_insts) {
  // User annotated replicated instruction through xla_sharding.replicate()
  // API, which cannot be modified during later sharding propagation.
  if (inst_strategy.replicated() || inst_strategy.Glue()) {
    return;
  }

  VLOG(2) << "infer users of " << inst->name();

  auto& strategy_map = graph_strategy.strategy_map();
  std::deque<std::pair<const HloInstruction* /*inst*/, SharedDimStrategy>> worklist;
  worklist.push_back(std::make_pair(inst, std::make_shared<DimStrategy>(inst_strategy)));

  while (!worklist.empty()) {
    std::pair<const HloInstruction*, std::shared_ptr<DimStrategy>>& inst_strategy_pair = worklist.front();
    const HloInstruction* inst = inst_strategy_pair.first;
    std::shared_ptr<DimStrategy> strategy = inst_strategy_pair.second;
    worklist.pop_front();
    for (auto* user : inst->users()) {
      if ((!inst_scope.empty()) && inst_scope.find(user) == inst_scope.end()) {
        continue;
      }
      if (user->opcode() == HloOpcode::kTuple) continue;

      // determine user's strategy
      CHECK(!strategy->Glue());

      int input_idx = user->operand_index(inst);
      HloInstMap<SharedDimStrategy> infered_strategy_map =
          StrategyUtil::ForwardInfer(user, *strategy, input_idx);

      for (auto& infered_inst_strategy : infered_strategy_map) {
        const HloInstruction* infered_inst = infered_inst_strategy.first;
        SharedDimStrategy& infered_strategy = infered_inst_strategy.second;
        if (infered_strategy->Glue() || infered_strategy->replicated()) {
          continue;
        }
        HloInstMap<SharedDimStrategy>::const_iterator strategy_it;
        strategy_it = strategy_map.find(infered_inst);
        if (strategy_it == strategy_map.end() || strategy_it->second->Glue()) {
          if (infered_inst == user) {
            // Dot instruction may infer another input from one input
            // drop out the infered input
            
            if (expected_strategies.find(user) != expected_strategies.end()) {
              if (*expected_strategies.at(user) != *infered_strategy) {
                continue;
              }
            }

            graph_strategy.SetInstStrategy(infered_inst, infered_strategy);
            infered_insts.insert(infered_inst);

            if (stop_inst_set.find(user) == stop_inst_set.end()) {
              worklist.push_back(std::make_pair(user, std::make_shared<DimStrategy>(*infered_strategy)));
            } else {
              // user is a stop instruction
              // infer it, but not recursively infer any more
            }
          }
        } else {
          if (!infered_strategy->Match(*strategy_map.at(infered_inst)) && record_conflict_) {
            conflict_proposals_[infered_inst].insert(infered_strategy);
            VLOG(2) << "At least two dimensions are specified to be splitted for instruction";
            VLOG(2) << infered_inst->ToString();
          }
        }
      } // end for (auto& infered_inst_strategy : infered_strategy_map)
    } // end for (auto* user : inst->users())
  }

  return;
}

void
CostSpmdStrategy::InferFromUser(
    const HloInstruction* inst,
    const DimStrategy& inst_strategy,
    const HloInstSet& stop_inst_set,  // stop instruction will NOT be inferred
    const HloInstSet& inst_scope,
    const HloInstMap<SharedDimStrategy>& expected_strategies,
    GraphStrategy& graph_strategy,
    HloInstSet& infered_insts) {
  CHECK(inst);
  VLOG(2) << "beginning of InferFromUser, inst: " << inst->name();
  auto& strategy_map = graph_strategy.strategy_map();
  CHECK(strategy_map.count(inst))
      << inst->name() << " does not exist in current strategy_map!";

  // Short circuit for replicated instruction in backward propagation.
  if (inst_strategy.replicated() || inst_strategy.Glue()) {
    return;
  }

  VLOG(2) << "[Reverse start] inst: " << inst->ToString()
          << " strategy: " << inst_strategy.ToString();

  VLOG(2) << "handle all ops in InferFromUser";

  std::deque<std::pair<const HloInstruction* /*inst*/, std::shared_ptr<DimStrategy>>> worklist;
  worklist.push_back(std::make_pair(inst, std::make_shared<DimStrategy>(inst_strategy)));

  while (!worklist.empty()) {
    std::pair<const HloInstruction*, std::shared_ptr<DimStrategy>>& inst_strategy_pair = worklist.front();
    const HloInstruction* inst = inst_strategy_pair.first;
    std::shared_ptr<DimStrategy> strategy = inst_strategy_pair.second;
    worklist.pop_front();
    for (int64 op_idx = 0; op_idx < inst->operand_count(); ++op_idx) {
      auto* op = inst->operand(op_idx);
      if ((!inst_scope.empty()) && inst_scope.find(op) == inst_scope.end()) {
        continue;
      }
      if (stop_inst_set.find(op) != stop_inst_set.end()) {
        // TODO(lansong): user of annotation instruction may be back inferred
        continue;
      }

      auto infered_strategy = StrategyUtil::BackInfer(inst, *strategy, op_idx);
      if (infered_strategy->Glue() || infered_strategy->replicated()) {
        VLOG(2) << "op strategy is inferred as glue";
        continue;
      }
      HloInstMap<SharedDimStrategy>::const_iterator strategy_it;
      strategy_it = strategy_map.find(op);
      if (strategy_it == strategy_map.end() || strategy_it->second->Glue()) {
        if (expected_strategies.find(op) != expected_strategies.end()) {
          if (*expected_strategies.at(op) != *infered_strategy) {
            VLOG(2) << "op strategy conflict, expected " << expected_strategies.at(op)->ToString()
                    << ", but inferred " << infered_strategy->ToString();
            continue;
          }
        }

        VLOG(2) << "op inst: " << op->ToString()
                << "infered strategy: " << infered_strategy->ToString();
        graph_strategy.SetInstStrategy(op, infered_strategy);
        infered_insts.insert(op);
        worklist.push_back(std::make_pair(op, std::make_shared<DimStrategy>(*infered_strategy)));
      } else {
        if (!infered_strategy->Match(*strategy_map.at(op)) && record_conflict_) {
          conflict_proposals_[op].insert(infered_strategy);
          VLOG(2) << "At least two dimensions are specified to be splitted for instruction";
          VLOG(2) << op->ToString();
        }
      }
    }
  }
  return;
}

void CostSpmdStrategy::CalcInstRank(HloModule* module,
                                    HloInstMap<int>& inst_rank_map,
                                    HloInstMap<int>& compute_rank_map) {
  auto entry = module->entry_computation();
  auto post_order = entry->MakeInstructionPostOrder();

  for (auto* inst : post_order) {
    CHECK(inst_rank_map.find(inst) == inst_rank_map.end());
    int rank = 0;
    int compute_rank = -1;
    for (auto* op : inst->operands()) {
      CHECK(inst_rank_map.find(op) != inst_rank_map.end());
      if (inst_rank_map[op] >= rank) {
        rank = inst_rank_map[op]+1;
      }

      CHECK(compute_rank_map.find(op) != compute_rank_map.end());
      if (is_compute_intensive(inst)) {
        if (compute_rank_map[op] >= compute_rank) {
          compute_rank = compute_rank_map[op]+1;
        }
      } else {
        if (compute_rank_map[op] > compute_rank) {
          compute_rank = compute_rank_map[op];
        }
      }
    }
    inst_rank_map[inst] = rank;
    compute_rank_map[inst] = compute_rank;
    VLOG(1) << "inst: " << inst->name() << ", rank: " << rank
            << ", compute rank: " << compute_rank;
  }

  for (auto& compute_rank : compute_rank_map) {
    if (compute_rank.second<0) {
      compute_rank.second = 0;
    }
  }
}

bool CostSpmdStrategy::InferByRank(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<int>& inst_rank_map,
        GraphStrategy& graph_strategy) {
  HloInstSet to_infer_insts;
  for (auto& start_strategy : starting_strategies) {
    to_infer_insts.insert(start_strategy.first);
  }

  HloInstMap<SharedDimStrategy> dummy_expected_strategies;

  bool changed = false;
  while (!to_infer_insts.empty()) {
    VLOG(1) << "remaining start instruction number: " << to_infer_insts.size();
    int max_infered_rank = -1;
    const HloInstruction* max_start_inst = nullptr;
    GraphStrategy max_graph_strategy(cur_split_num_, cur_share_dev_);
    HloInstSet max_infered_insts;
    HloInstMap<std::vector<const HloInstruction*>> split_root_map;
    for (auto* start_inst : to_infer_insts) {
      CHECK(start_inst);
      CHECK(starting_strategies.find(start_inst) != starting_strategies.end());
      VLOG(2) << "single start inst: " << start_inst->ToString();
      VLOG(2) << "start strategy: " << starting_strategies.at(start_inst)->ToString();
      GraphStrategy try_graph_strategy = graph_strategy;
      HloInstSet infered_insts;
      InferUsersFromInst(start_inst,
                         *starting_strategies.at(start_inst),
                         stop_insts,
                         inst_scope,
                         dummy_expected_strategies,
                         try_graph_strategy,
                         infered_insts);
      int infered_rank = 0;
      for (auto* infered_inst : infered_insts) {
        split_root_map[infered_inst].emplace_back(start_inst);

        if (is_compute_intensive(infered_inst)) {
          CHECK(inst_rank_map.find(infered_inst) != inst_rank_map.end());
          if (inst_rank_map.at(infered_inst)>infered_rank) {
            infered_rank = inst_rank_map.at(infered_inst);
          }
        }
      }

      VLOG(1) << "max_infered_rank: " << max_infered_rank << ", infered_rank: " << infered_rank;

      if (infered_rank > max_infered_rank) {
        max_infered_rank = infered_rank;
        max_graph_strategy = try_graph_strategy;
        //max_strategy_map = try_strategy_map;
        max_infered_insts = infered_insts;
        max_start_inst = start_inst;
      }

      VLOG(1) << infered_insts.size() << " instructions are inferred";
      VLOG(1) << "max_infered_rank: " << max_infered_rank << ", infered_rank: " << infered_rank;

      for (auto* infered_inst : infered_insts) {
        VLOG(2) << "  inst: " << infered_inst->ToString();
        VLOG(2) << "  strategy: " << try_graph_strategy.strategy(infered_inst)->ToString();
      }
    }

    if (!max_start_inst) {
      VLOG(1) << "no more progress in inference";
      break;
    }

    VLOG(1) << "current start inst: " << max_start_inst->name()
            << ", finalize " << max_infered_insts.size() << " instructions";

    if (!max_infered_insts.empty()) {
      graph_strategy = max_graph_strategy;
      changed = true;
    }

    to_infer_insts.erase(max_start_inst);
  }

  return changed;
}

bool CostSpmdStrategy::InferByInstCount(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts, // stop instruction will be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy) {
  HloInstSet to_infer_insts;
  for (auto& start_strategy : starting_strategies) {
    to_infer_insts.insert(start_strategy.first);
  }

  HloInstMap<SharedDimStrategy> dummy_expected_strategies;

  bool changed = false;
  while (!to_infer_insts.empty()) {
    VLOG(1) << "remaining start instruction number: " << to_infer_insts.size();
    int max_infered_count = 0;

    const HloInstruction* max_start_inst = nullptr;
    GraphStrategy max_graph_strategy(cur_split_num_, cur_share_dev_);
    HloInstSet max_infered_insts;
    HloInstMap<std::vector<const HloInstruction*>> split_root_map;
    for (auto* start_inst : to_infer_insts) {
      CHECK(start_inst);
      CHECK(starting_strategies.find(start_inst) != starting_strategies.end());
      VLOG(2) << "single start inst: " << start_inst->ToString();
      VLOG(2) << "start strategy: " << starting_strategies.at(start_inst)->ToString();
      GraphStrategy try_graph_strategy = graph_strategy;
      HloInstSet infered_insts;
      InferUsersFromInst(start_inst,
                         *starting_strategies.at(start_inst),
                         stop_insts,
                         inst_scope,
                         dummy_expected_strategies,
                         try_graph_strategy,
                         infered_insts);
      for (auto* infered_inst : infered_insts) {
        split_root_map[infered_inst].emplace_back(start_inst);
      }

      VLOG(2) << "max_infered_count: " << max_infered_count << ", infered_count: " << infered_insts.size();

      if (max_infered_count < infered_insts.size()) {
        max_infered_count = infered_insts.size();
        max_graph_strategy = try_graph_strategy;
        max_infered_insts = infered_insts;
        max_start_inst = start_inst;
      }

      VLOG(2) << "max_infered_count: " << max_infered_count << ", infered_count: " << infered_insts.size();

      for (auto* infered_inst : infered_insts) {
        VLOG(2) << "  inst: " << infered_inst->ToString();
        VLOG(2) << "  strategy: " << try_graph_strategy.strategy(infered_inst)->ToString();
      }
    }

    if (!max_start_inst) {
      VLOG(1) << "no more progress in inference";
      break;
    }

    VLOG(1) << "current start inst: " << max_start_inst->name()
            << ", finalize " << max_infered_insts.size() << " instructions";

    if (!max_infered_insts.empty()) {
      graph_strategy = max_graph_strategy;
      changed = true;
    }

    to_infer_insts.erase(max_start_inst);
  }

  return changed;
}

bool CostSpmdStrategy::ReverseInferByRank(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<int>& inst_rank_map,
        GraphStrategy& graph_strategy) {
  HloInstSet to_infer_insts;
  for (auto& start_strategy : starting_strategies) {
    to_infer_insts.insert(start_strategy.first);
  }

  HloInstMap<SharedDimStrategy> dummy_expected_strategies;

  bool changed = false;
  while (!to_infer_insts.empty()) {
    VLOG(1) << "remaining start instruction number: " << to_infer_insts.size();
    int min_infered_rank = INT_MAX;
    const HloInstruction* min_start_inst = nullptr;
    GraphStrategy min_graph_strategy(cur_split_num_, cur_share_dev_);
    HloInstSet min_infered_insts;
    HloInstMap<std::vector<const HloInstruction*>> split_root_map;
    for (auto* start_inst : to_infer_insts) {
      CHECK(start_inst);
      CHECK(starting_strategies.find(start_inst) != starting_strategies.end());
      VLOG(2) << "single start inst: " << start_inst->ToString();
      VLOG(2) << "start strategy: " << starting_strategies.at(start_inst)->ToString();
      GraphStrategy try_graph_strategy = graph_strategy;
      HloInstSet infered_insts;

      InferFromUser(start_inst,
                    *starting_strategies.at(start_inst),
                    stop_insts,
                    inst_scope,
                    dummy_expected_strategies,
                    try_graph_strategy,
                    infered_insts);
      int infered_rank = INT_MAX;
      for (auto* infered_inst : infered_insts) {
        split_root_map[infered_inst].emplace_back(start_inst);

        if (is_compute_intensive(infered_inst)) {
          CHECK(inst_rank_map.find(infered_inst) != inst_rank_map.end());
          if (inst_rank_map.at(infered_inst)<infered_rank) {
            infered_rank = inst_rank_map.at(infered_inst);
          }
        }
      }

      VLOG(1) << "min_infered_rank: " << min_infered_rank << ", infered_rank: " << infered_rank;
      if (infered_rank < min_infered_rank) {
        min_infered_rank = infered_rank;
        min_graph_strategy = try_graph_strategy;
        min_infered_insts = infered_insts;
        min_start_inst = start_inst;
      }

      VLOG(1) << "min_infered_rank: " << min_infered_rank << ", infered_rank: " << infered_rank;

      VLOG(1) << infered_insts.size() << " instructions are inferred";
    }

    if (!min_start_inst) {
      VLOG(1) << "no more progress in reverse inference";
      break;
    }

    VLOG(1) << "current start inst: " << min_start_inst->name()
            << ", finalize " << min_infered_insts.size() << " instructions";

    if (!min_infered_insts.empty()) {
      graph_strategy = min_graph_strategy;
      changed = true;
    }

    to_infer_insts.erase(min_start_inst);
  }

  return changed;
}

bool CostSpmdStrategy::ReverseInferByInstCount(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy) {
  HloInstSet to_infer_insts;
  for (auto& start_strategy : starting_strategies) {
    to_infer_insts.insert(start_strategy.first);
  }

  HloInstMap<SharedDimStrategy> dummy_expected_strategies;

  bool changed = false;
  while (!to_infer_insts.empty()) {
    VLOG(1) << "remaining start instruction number: " << to_infer_insts.size();
    int max_infered_count = 0;

    const HloInstruction* min_start_inst = nullptr;

    GraphStrategy min_graph_strategy(cur_split_num_, cur_share_dev_);
    HloInstSet min_infered_insts;
    HloInstMap<std::vector<const HloInstruction*>> split_root_map;
    for (auto* start_inst : to_infer_insts) {
      CHECK(start_inst);
      CHECK(starting_strategies.find(start_inst) != starting_strategies.end());
      VLOG(2) << "single start inst: " << start_inst->ToString();
      VLOG(2) << "start strategy: " << starting_strategies.at(start_inst)->ToString();
      GraphStrategy try_graph_strategy = graph_strategy;
      HloInstSet infered_insts;

      InferFromUser(start_inst,
                    *starting_strategies.at(start_inst),
                    stop_insts,
                    inst_scope,
                    dummy_expected_strategies,
                    try_graph_strategy,
                    infered_insts);
      for (auto* infered_inst : infered_insts) {
        split_root_map[infered_inst].emplace_back(start_inst);
      }

      VLOG(2) << "max_infered_count: " << max_infered_count << ", infered_count: " << infered_insts.size();

      if (max_infered_count < infered_insts.size()) {
        max_infered_count = infered_insts.size();
        min_graph_strategy = try_graph_strategy;
        //min_strategy_map = try_strategy_map;
        min_infered_insts = infered_insts;
        min_start_inst = start_inst;
      }

      VLOG(2) << "max_infered_count: " << max_infered_count << ", infered_count: " << infered_insts.size();

      for (auto* infered_inst : infered_insts) {
        VLOG(2) << "  inst: " << infered_inst->ToString();
        VLOG(2) << "  strategy: " << try_graph_strategy.strategy(infered_inst)->ToString();
      }
    }

    if (!min_start_inst) {
      VLOG(1) << "no more progress in reverse inference";
      break;
    }

    VLOG(1) << "current start inst: " << min_start_inst->name()
            << ", finalize " << min_infered_insts.size() << " instructions";

    if (!min_infered_insts.empty()) {
      graph_strategy = min_graph_strategy;
      changed = true;
    }

    to_infer_insts.erase(min_start_inst);
  }

  return changed;
}

HloInstSet CostSpmdStrategy::InferGreedy(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts, // stop instruction will be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy) {
  HloInstMap<SharedDimStrategy> expected_strategies;
  return InferGreedy(starting_strategies, stop_insts, inst_scope,
                     expected_strategies, graph_strategy);
}

HloInstSet CostSpmdStrategy::InferGreedy(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts, // stop instruction will be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<SharedDimStrategy>& expected_strategies,
        GraphStrategy& graph_strategy) {
  HloInstSet infered_insts;
  for (auto& start_strategy : starting_strategies) {
    VLOG(2) << "single start inst: " << start_strategy.first->ToString();
    VLOG(2) << "start strategy: " << start_strategy.second->ToString();
    InferUsersFromInst(start_strategy.first,
                       *start_strategy.second,
                       stop_insts,
                       inst_scope,
                       expected_strategies,
                       graph_strategy,
                       infered_insts);
  }

  if (!infered_insts.empty()) {
    VLOG(1) << infered_insts.size() << " instructions are inferred";
  }

  return std::move(infered_insts);
}

HloInstSet CostSpmdStrategy::ReverseInferGreedy(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy) {
  HloInstMap<SharedDimStrategy> expected_strategies;
  return ReverseInferGreedy(starting_strategies, stop_insts, inst_scope,
                            expected_strategies, graph_strategy);
}

HloInstSet CostSpmdStrategy::ReverseInferGreedy(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<SharedDimStrategy>& expected_strategies,
        GraphStrategy& graph_strategy) {
  HloInstSet infered_insts;
  for (auto& start_strategy : starting_strategies) {
    VLOG(2) << "single start inst: " << start_strategy.first->ToString();
    VLOG(2) << "start strategy: " << start_strategy.second->ToString();
    InferFromUser(start_strategy.first,
                  *start_strategy.second,
                  stop_insts,
                  inst_scope,
                  expected_strategies,
                  graph_strategy,
                  infered_insts);
  }

  VLOG(2) << "in ReverseInferGreedy";
  if (!infered_insts.empty()) {
    VLOG(1) << infered_insts.size() << " instructions are inferred";
  }

  VLOG(2) << "return from ReverseInferGreedy";
  return std::move(infered_insts);
}

HloInstSet CostSpmdStrategy::PopulateStrategy(
      const HloInstruction* inst,
      const SharedDimStrategy& inst_strategy,
      const HloInstSet& stop_insts,
      const HloInstSet& inst_scope,
      GraphStrategy& graph_strategy) {  // graph_strategy contains the starting inst and inst_strategy
  HloInstMap<SharedDimStrategy> expected_strategies;
  return PopulateStrategy(inst, inst_strategy, stop_insts, inst_scope,
                          expected_strategies, graph_strategy);
}

HloInstSet CostSpmdStrategy::PopulateStrategy(
      const HloInstruction* inst,
      const SharedDimStrategy& inst_strategy,
      const HloInstSet& stop_insts,
      const HloInstSet& inst_scope,
      const HloInstMap<SharedDimStrategy>& expected_strategies,
      GraphStrategy& graph_strategy) {  // graph_strategy contains the starting inst and inst_strategy
  HloInstMap<SharedDimStrategy> startings;
  HloInstMap<SharedDimStrategy> reverse_startings;
  startings[inst] = inst_strategy;
  reverse_startings[inst] = inst_strategy;

  VLOG(1) << "start inst: " << inst->ToString();
  VLOG(1) << "strategy: " << inst_strategy->ToString();

  HloInstSet all_infered;
  HloInstSet infered_insts;
  do {
    VLOG(2) << "startings size: " << startings.size();
    infered_insts = InferGreedy(startings, stop_insts, inst_scope,
                                expected_strategies, graph_strategy);
    startings.clear();

    VLOG(2) << "infer ahead, infer num: " << infered_insts.size();
    for (auto* infered_inst : infered_insts) {
      reverse_startings[infered_inst] = graph_strategy.strategy(infered_inst);
      VLOG(2) << "forward result: inst name: " << infered_inst->name()
              << ", strategy: "
              << graph_strategy.strategy(infered_inst)->ToString();
      all_infered.insert(infered_inst);
    }

    VLOG(2) << "reverse startings size: " << reverse_startings.size();
    if (!reverse_startings.empty()) {
      infered_insts = ReverseInferGreedy(reverse_startings, stop_insts,
                                         inst_scope, expected_strategies,
                                         graph_strategy);

      VLOG(2) << "after calling ReverseInferGreedy";
      reverse_startings.clear();

      VLOG(2) << "infer revert, infer num: " << infered_insts.size();
      for (auto* infered_inst : infered_insts) {
        startings[infered_inst] = graph_strategy.strategy(infered_inst);
        VLOG(2) << "backward result: inst name: " << infered_inst->name()
                << ", strategy: "
                << graph_strategy.strategy(infered_inst)->ToString();
        all_infered.insert(infered_inst);
      }
    }
  } while (!startings.empty());

  VLOG(2) << "return from PopulateStrategy";
  return std::move(all_infered);
}

void
CostSpmdStrategy::InferFromAnnotation(
     const HloInstMap<SharedDimStrategy>& user_annotated_tensors,
     const HloInstMap<int>& inst_rank_map,
     const HloInstSet& inst_scope,
     GraphStrategy& graph_strategy) {
  VLOG(1) << "forward inst set:";
  for (auto* tmp : inst_scope) {
    VLOG(1) << "forward inst name: " << tmp->name();
  }
  VLOG(1) << "end of forward inst set";

  // infer sync-free splitting from user annotation
  HloInstSet user_annotated_set;
  for (auto& annotated_tensor : user_annotated_tensors) {
    user_annotated_set.insert(annotated_tensor.first);
  }

  InferByRank(user_annotated_tensors, user_annotated_set,
             inst_scope, inst_rank_map, graph_strategy);

  user_annotated_insts_.clear();

  auto& strategy_map = graph_strategy.strategy_map();
  for (auto& inst_strategy : strategy_map) {
    bool is_annotated = false;
    const HloInstruction* inst = inst_strategy.first;
    for (auto* op : inst->operands()) {
      if (user_annotated_tensors.find(op) != user_annotated_tensors.end()) {
        is_annotated = true;
        break;
      }
    }
    if (is_annotated) {
      graph_strategy.AddUserAnnotatedInst(inst);
      user_annotated_insts_.insert(inst);
    }
  }

  return;
}

HloInstMap<SharedDimStrategy> CostSpmdStrategy::FindStartingStrategies(
          const HloInstMap<SharedDimStrategy>& strategy_map,
          const HloInstSet& sub_graph_insts) {
  HloInstMap<SharedDimStrategy> starting_strategies;
  for (auto& inst_strategy : strategy_map) {
    for (auto* user : inst_strategy.first->users()) {
      if ((!sub_graph_insts.empty()) &&
          sub_graph_insts.find(user) == sub_graph_insts.end()) {
        continue;
      }
      if (strategy_map.find(user) == strategy_map.end()) {
        starting_strategies[inst_strategy.first] = inst_strategy.second;
        break;
      } else {
        if (strategy_map.at(user)->Glue()) {
          starting_strategies[inst_strategy.first] = inst_strategy.second;
          break;
        }
      }
    }
  }

  return std::move(starting_strategies);
}

HloInstMap<SharedDimStrategy> CostSpmdStrategy::FindBackStartingStrategies(
          const HloInstMap<SharedDimStrategy>& strategy_map,
          const HloInstSet& sub_graph_insts) {
  HloInstMap<SharedDimStrategy> starting_strategies;
  for (auto& inst_strategy : strategy_map) {
    for (auto* op : inst_strategy.first->operands()) {
      if ((!sub_graph_insts.empty()) &&
          sub_graph_insts.find(op) == sub_graph_insts.end()) {
        continue;
      }
      if (strategy_map.find(op) == strategy_map.end()) {
        starting_strategies[inst_strategy.first] = inst_strategy.second;
        break;
      } else {
        if (strategy_map.at(op)->Glue()) {
          starting_strategies[inst_strategy.first] = inst_strategy.second;
          break;
        }
      }
    }
  }

  return std::move(starting_strategies);
}

std::list<const HloInstruction*>
CostSpmdStrategy::CollectUnsplittedComputeInsts(
           const HloModule* module,
           const HloInstMap<SharedDimStrategy>& strategy_map) {
  auto* entry = module->entry_computation();

  std::list<const HloInstruction*> unsplitted_compute_insts;

  for (auto* inst : entry->instructions()) {
    if (!is_compute_intensive(inst)) {
      continue;
    }
    if (strategy_map.find(inst) == strategy_map.end() ||
        strategy_map.at(inst)->Glue()) {
      unsplitted_compute_insts.push_back(inst);
    }
  }

  return std::move(unsplitted_compute_insts);
}

std::list<const HloInstruction*>
CostSpmdStrategy::CollectUnsplittedComputeInsts(
           const HloInstSet& inst_scope,
           const HloInstMap<SharedDimStrategy>& strategy_map) {
  std::list<const HloInstruction*> unsplitted_compute_insts;
  for (auto* inst : inst_scope) {
    if (!is_compute_intensive(inst)) {
      continue;
    }
    if (strategy_map.find(inst) == strategy_map.end() ||
        strategy_map.at(inst)->Glue()) {
      unsplitted_compute_insts.push_back(inst);
    }
  }

  return std::move(unsplitted_compute_insts);
}

HloInstMap<SharedDimStrategy>
CostSpmdStrategy::CollectSplittedComputeInsts(
           const HloInstSet& inst_scope,
           const HloInstMap<SharedDimStrategy>& strategy_map) {
  HloInstMap<SharedDimStrategy> compute_strategies;
  for (auto* inst : inst_scope) {
    if (!is_compute_intensive(inst)) {
      continue;
    }
    if (strategy_map.find(inst) != strategy_map.end() &&
        strategy_map.at(inst)->Glue()) {
      compute_strategies[inst] = strategy_map.at(inst);
    }
  }

  return std::move(compute_strategies);
}

bool CostSpmdStrategy::SplitPlanByMemCost(
           const HloModule* module,
           const HloInstMap<SharedDimStrategy>& strategy_map,
           const HloInstMap<int>& compute_rank_map,
           const int64 mem_limit,
           MemSavePlan& mem_save_plan) {
  auto entry = module->entry_computation();

  int dev_count = cur_split_num_;

  int64 total_mem_cost = 0;
  int64 total_orig_mem_cost = 0;
  int64 mem_bytes;
  for (auto* param : entry->parameter_instructions()) {
    if (strategy_map.find(param) != strategy_map.end()) {
      total_mem_cost += ShapeUtil::ByteSizeOf(param->shape(), 8) /
                        strategy_map.at(param)->NumSlices();  // it's better to use cur_split_num_?
    } else {
      total_mem_cost += ShapeUtil::ByteSizeOf(param->shape(), 8);
    }

    total_orig_mem_cost += ShapeUtil::ByteSizeOf(param->shape(), 8);
  }

  mem_save_plan.total_mem_cost_ = total_mem_cost;
  mem_save_plan.total_orig_mem_cost_ = total_orig_mem_cost;

  VLOG(0) << "total_mem_cost: " << total_mem_cost << ", total_orig_mem_cost: "
          << total_orig_mem_cost;

  if ((total_orig_mem_cost + dev_count - 1)/dev_count > mem_limit) {
    VLOG(0) << "(total_orig_mem_cost + dev_count - 1)/dev_count: "
            << (total_orig_mem_cost + dev_count - 1)/dev_count;
    VLOG(0) << "mem_limit: " << mem_limit;
    return false;
  }

  std::list<const HloInstruction*> unsplitted_compute_insts =
                          CollectUnsplittedComputeInsts(module, strategy_map);

  VLOG(0) << "unsplitted_compute_insts size: " << unsplitted_compute_insts.size();

  auto calc_cost = [](const HloInstruction* inst) -> int64 {
    CHECK(is_compute_intensive(inst));
    int64 cost = 0;
    if (inst->opcode() == HloOpcode::kDot || inst->opcode() == HloOpcode::kConvolution) {
      int i=0;
      for (auto* op : inst->operands()) {
        if (i==1) {
          cost += ShapeUtil::ByteSizeOf(op->shape(), 8);
        }
        ++i;
      }
    } else if (inst->opcode() == HloOpcode::kCustomCall &&
               inst->custom_call_target() == "__cublas$gemm") {
    }

    return cost;
  };

  std::vector<std::pair<const HloInstruction*, int64>> inst_costs;
  int64 cost;
  for (auto* inst : unsplitted_compute_insts) {
    cost = calc_cost(inst);
    inst_costs.emplace_back(std::make_pair(inst, cost));
    VLOG(1) << "inst: " << inst->name() << ", cost: " << cost;
  }

  std::sort(inst_costs.begin(), inst_costs.end(),
            [](const std::pair<const HloInstruction*, int64>& lhs,
               const std::pair<const HloInstruction*, int64>& rhs) {
                 return lhs.second > rhs.second;
               });

  mem_save_plan.expected_mem_cost_ = mem_save_plan.total_mem_cost_;
  for (auto& inst_cost : inst_costs) {
    VLOG(1) << std::endl;
    if (mem_save_plan.expected_mem_cost_ > mem_limit &&
        inst_cost.second > 0) {
      VLOG(1) << "plan to split inst(mem save): " << inst_cost.first->name()
              << ", cost: " << inst_cost.second;
      int64 new_cost = (inst_cost.second + dev_count - 1) / dev_count;
      int64 mem_save = inst_cost.second - new_cost;
      VLOG(1) << "new cost: " << new_cost;
      VLOG(1) << "mem save: " << mem_save;
      mem_save_plan.expected_mem_cost_ -= mem_save;
      VLOG(1) << "expected mem: " << mem_save_plan.expected_mem_cost_;
      mem_save_plan.split_for_mem_save_.insert(inst_cost.first);
    } else {
      mem_save_plan.split_for_compute_.insert(inst_cost.first);
      VLOG(1) << "plan to split inst(compute): " << inst_cost.first->name()
              << ", cost: " << inst_cost.second;
    }
  }

  if (mem_save_plan.expected_mem_cost_ < mem_limit) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<InstCone>
ExtractCone(
            const HloInstruction* cone_root,
            const HloInstSet& inst_scope,
            const std::unordered_set<const HloInstruction*>& exclude_insts) {
  std::shared_ptr<InstCone> cone = std::make_shared<InstCone>(cone_root);
  std::deque<const HloInstruction*> ready_insts;
  std::unordered_set<const HloInstruction*> visited;
  ready_insts.push_back(cone_root);
  visited.insert(cone_root);

  while (!ready_insts.empty()) {
    auto* inst = ready_insts.front();
    ready_insts.pop_front();
    if ((!inst_scope.empty()) && inst_scope.find(inst) == inst_scope.end()) {
      // only search instruction scope
      continue;
    } else if (inst != cone_root &&
               exclude_insts.find(inst) != exclude_insts.end()) {
      // skip exclusion instructions
      continue;
    } else {
      cone->AddInst(inst);

      for (auto* op : inst->operands()) {
        if (visited.find(op) == visited.end()) {
          ready_insts.push_back(op);
          visited.insert(op);
        }
      }
    }
  }

  return cone;
}

std::shared_ptr<InstCone>
CostSpmdStrategy::ExtractReverseCone(
            const HloInstruction* cone_root,
            const HloInstSet& inst_scope,
            const std::unordered_set<const HloInstruction*>& exclude_insts) {
  std::shared_ptr<InstCone> cone = std::make_shared<InstCone>(cone_root);
  std::deque<const HloInstruction*> ready_insts;
  std::unordered_set<const HloInstruction*> visited;
  ready_insts.push_back(cone_root);
  visited.insert(cone_root);
  while (!ready_insts.empty()) {
    auto* inst = ready_insts.front();
    ready_insts.pop_front();
    if ((!inst_scope.empty()) && inst_scope.find(inst) == inst_scope.end()) {
      // only search instruction scope
      continue;
    } else if (inst != cone_root &&
               exclude_insts.find(inst) != exclude_insts.end()) {
      // skip exclusion instructions
      continue;
    } else {
      cone->AddInst(inst);

      for (auto* user : inst->users()) {
        if (visited.find(user) == visited.end()) {
          ready_insts.push_back(user);
          visited.insert(user);
        }
      }
    }
  }

  return cone;
}

HloSubGraph
CostSpmdStrategy::BuildSubGraph(
              const std::vector<const HloInstruction*>& insts,
              const HloInstruction* head,
              const HloInstruction* tail) {
  HloSubGraph sub_graph(head, tail);
  sub_graph.insts_.reserve(insts.size());
  for (auto* inst : insts) {
      sub_graph.insts_.emplace_back(inst);
  }

  return std::move(sub_graph);
}

HloSubGraph
CostSpmdStrategy::ExtractSubGraph(
              const HloInstruction* start,
              const HloInstruction* end,
              const HloInstSet& inst_scope,
              const std::unordered_set<const HloInstruction*>& exclude_insts) {
  HloSubGraph sub_graph(start, end);

  if (end) {
    std::shared_ptr<InstCone> cone = ExtractCone(end, inst_scope, exclude_insts);
    sub_graph.insts_.insert(sub_graph.insts_.end(),
                            cone->insts_.begin(),
                            cone->insts_.end());

    if (start) {
      sub_graph.insts_.emplace_back(start);
    }
    std::reverse(sub_graph.insts_.begin(), sub_graph.insts_.end());
  } else {
    CHECK(start);
    std::shared_ptr<InstCone> cone = ExtractReverseCone(start,
                                                        inst_scope,
                                                        exclude_insts);
    sub_graph.insts_.insert(sub_graph.insts_.end(),
                            cone->insts_.begin(),
                            cone->insts_.end());
  }

  return std::move(sub_graph);
}

std::vector<HloSubGraph> CostSpmdStrategy::FindSubGraphs(
    HloModule* module, const HloInstSet& forward_insts) {
  VLOG(1) << "before BuildGraphSketch";
  std::unique_ptr<GraphSketch> graph_sketch = GraphSketch::BuildGraphSketch(module, 3);
  VLOG(1) << "before FindCriticalInsts";
  std::vector<const HloInstruction*> critical_insts = graph_sketch->FindCriticalInsts();

  std::vector<HloSubGraph> sub_graphs;
  std::unordered_set<const HloInstruction*> exclude_insts;

  VLOG(2) << "inst num: " << forward_insts.size();
  VLOG(2) << "critical_insts size: " << critical_insts.size();
  if (critical_insts.empty()) {
    return std::move(sub_graphs);
  }

  int foward_critical_inst_num = critical_insts.size();
  for (int i=0; i<critical_insts.size(); ++i) {
    VLOG(2) << "critical inst: " << critical_insts[i]->ToString();
    if (forward_insts.find(critical_insts[i]) == forward_insts.end()) {
      foward_critical_inst_num = i;
      break;
    }
  }

  if (foward_critical_inst_num < critical_insts.size()) {
    VLOG(2) << "find " << foward_critical_inst_num
            << " forward critical instructions and "
            << critical_insts.size() - foward_critical_inst_num
            << " backward critical instructions";

    critical_insts.resize(foward_critical_inst_num);
  }

  int64 group_num = ServiceEnv::forward_sub_graph_num();
  int critical_inst_num_per_group = 1;
  if (critical_insts.size() > group_num) {
    critical_inst_num_per_group = critical_insts.size() / group_num;
  }
  VLOG(2) << "forward group num: " << group_num
          << ", critical_inst_num_per_group: " << critical_inst_num_per_group;

  VLOG(2) << "ExtractSubGraph 0";
  int sub_end = critical_inst_num_per_group-1;
  if (sub_graphs.size() == group_num-1) {
    // to build last sub graph
    sub_end = critical_insts.size() - 1;
  }
  HloSubGraph first_sub = ExtractSubGraph(nullptr, critical_insts[sub_end],
                                          forward_insts, exclude_insts);
  exclude_insts.insert(first_sub.insts_.begin(), first_sub.insts_.end());
  first_sub.id_ = sub_graphs.size();
  sub_graphs.emplace_back(std::move(first_sub));
  for (int i=1; i<group_num; ++i) {
    VLOG(2) << "ExtractSubGraph " << i;
    VLOG(2) << "sub_graphs.size: " << sub_graphs.size();
    int pre_sub_end = sub_end;
    sub_end += critical_inst_num_per_group;
    if (sub_end >= critical_insts.size()) {
      sub_end = critical_insts.size() - 1;
    } else {
      if (sub_graphs.size() == group_num-1) {
        // to build last sub graph
        sub_end = critical_insts.size() - 1;
      }
    }

    CHECK(pre_sub_end < sub_end);

    HloSubGraph sub = ExtractSubGraph(critical_insts[pre_sub_end],
                                      critical_insts[sub_end],
                                      forward_insts, exclude_insts);
    exclude_insts.insert(sub.insts_.begin(), sub.insts_.end());
    sub.id_ = sub_graphs.size();
    sub_graphs.emplace_back(std::move(sub));
  }

  VLOG(2) << "sub_graphs.size: " << sub_graphs.size();

  VLOG(2) << "before expand sub graph";
  for (int i=0; i<sub_graphs.size(); ++i) {
    VLOG(2) << "start of sub graph " << sub_graphs[i].id_;
    VLOG(2) << sub_graphs[i].ToString();
    VLOG(2) << "end of sub graph " << i << std::endl;
  }

  // add backward instructions into sub graphs
  ExpandSubGraphs(module, critical_insts, exclude_insts, sub_graphs);

  VLOG(2) << "after expand sub graph";
  for (int i=0; i<sub_graphs.size(); ++i) {
    VLOG(2) << "start of sub graph " << sub_graphs[i].id_;
    VLOG(2) << sub_graphs[i].ToString();
    VLOG(2) << "end of sub graph " << i << std::endl;
  }

  return std::move(sub_graphs);
}

void MapInstToSubGraph(
             const HloModule* module,
             const std::vector<const HloInstruction*>& critical_insts,
             std::unordered_set<const HloInstruction*>& exclude_insts,
             std::vector<HloSubGraph>& sub_graphs) {
  std::set<const HloInstruction*> critical_set;
  critical_set.insert(critical_insts.begin(), critical_insts.end());

  std::map<int/*group id*/, std::set<int/*sub graph id*/>> group_graph_map;
  for (auto& sub : sub_graphs) {
    for (auto* sub_inst : sub.insts_) {
      if (critical_set.find(sub_inst) != critical_set.end()) {
        if (sub_inst == sub.head_ && sub.id_ > 0) {
          // head is owned by previous sub graph,
          // tail is owned by current sub graph
          continue;
        }
      }
      if (sub_inst->metadata().op_group() > 0) {
        int orig_size = group_graph_map[sub_inst->metadata().op_group()].size();
        group_graph_map[sub_inst->metadata().op_group()].insert(sub.id_);
        if (group_graph_map[sub_inst->metadata().op_group()].size() > orig_size) {
          VLOG(2) << "op group " << sub_inst->metadata().op_group() << " cover "
                  << group_graph_map[sub_inst->metadata().op_group()].size() << " sub graphs";
        }
      }
    }
  }

  std::map<int/*group id*/, int/*sub graph id*/> group_unique_graph_map;
  for (auto& group_graph : group_graph_map) {
    CHECK(!group_graph.second.empty());
    group_unique_graph_map[group_graph.first] = *(group_graph.second.begin());

    VLOG(2) << "group: " << group_graph.first << ", sub graph:";
    if (group_graph.second.size() > 1) {
      VLOG(2) << "  more than one sub graph, sub num: " << group_graph.second.size();
    }
    for (int sub_id : group_graph.second) {
      VLOG(2) << "  sub id: " << sub_id;
    }
  }

  const HloComputation* entry = module->entry_computation();
  int missed_cnt = 0;
  for (auto* inst : entry->instructions()) {
    if (exclude_insts.find(inst) != exclude_insts.end()) {
      continue;
    }

    int group_id = inst->metadata().op_group();
    if (inst->metadata().op_group() > 0) {
      if (group_unique_graph_map.find(group_id) != group_unique_graph_map.end()) {
        int sub_graph_id = group_unique_graph_map[group_id];
        CHECK(sub_graph_id>=0 && sub_graph_id<sub_graphs.size()) << sub_graph_id;
        sub_graphs[sub_graph_id].insts_.emplace_back(inst);
        exclude_insts.insert(inst);
      } else {
        VLOG(2) << "miss sub graph: inst " << inst->ToString();
        ++missed_cnt;
      }
    } else {
      VLOG(2) << "miss sub graph: inst " << inst->ToString();
      ++missed_cnt;
    }
  }

  VLOG(2) << "place " << exclude_insts.size() << " instructions in sub graph";
  VLOG(2) << missed_cnt << " instructions are not placed yet";
}

std::unordered_map<const HloInstruction*, int>
GenMinSubIdMap(
             const HloModule* module,
             std::vector<HloSubGraph>& sub_graphs) {
  std::unordered_map<const HloInstruction*, int> min_sub_id_map;
  for (auto& sub : sub_graphs) {
    for (auto* sub_inst : sub.insts_) {
      min_sub_id_map[sub_inst] = sub.id_;
    }
  }

  auto max_user_sub_id = [&min_sub_id_map] (const HloInstruction* inst) -> int {
    int max_sub_id = -1;
    for (auto* user : inst->users()) {
      if (min_sub_id_map.find(user) != min_sub_id_map.end()) {
        int user_sub_id = min_sub_id_map[user];
        if (user_sub_id > max_sub_id) {
          max_sub_id = user_sub_id;
        }
      } else {
        max_sub_id = -1;
        break;
      }
    }

    return max_sub_id;
  };

  const HloComputation* entry = module->entry_computation();
  std::deque<std::pair<const HloInstruction*, int>> ready_insts;
  // initialize ready_insts for search more extra instructions
  for (auto* inst : entry->instructions()) {
    if (min_sub_id_map.find(inst) == min_sub_id_map.end()) {
      continue;
    }
    int inst_sub_id = min_sub_id_map[inst];
    for (auto* inst_op : inst->operands()) {
      if (min_sub_id_map.find(inst_op) != min_sub_id_map.end()) {
        continue;
      }

      int op_sub_id = max_user_sub_id(inst_op);
      if (op_sub_id >= 0) {
        // op is ready to determine min sub graph id
        ready_insts.push_back(std::make_pair(inst_op, op_sub_id));
      }
    }
  }

  // iteratively determine min sub id for each instructions
  while (!ready_insts.empty()) {
    auto inst_sub_id = ready_insts.front();
    const HloInstruction* inst = inst_sub_id.first;
    min_sub_id_map[inst] = inst_sub_id.second;
    VLOG(2) << "min sub id: " << inst_sub_id.second << ", inst: " << inst->ToString();
    ready_insts.pop_front();
    for (auto* inst_op : inst->operands()) {
      if (min_sub_id_map.find(inst_op) != min_sub_id_map.end()) {
        continue;
      }
      int op_sub_id = max_user_sub_id(inst_op);
      if (op_sub_id >= 0) {
        // op is ready to determine min sub graph id
        ready_insts.push_back(std::make_pair(inst_op, op_sub_id));
      }
    }
  }

  for (auto& sub : sub_graphs) {
    for (auto* sub_inst : sub.insts_) {
      min_sub_id_map.erase(sub_inst);
    }
  }

  VLOG(2) << "min_sub_id_map size: " << min_sub_id_map.size();

  return std::move(min_sub_id_map);
}

void CostSpmdStrategy::ExpandSubGraphs(
             const HloModule* module,
             const std::vector<const HloInstruction*>& critical_insts,
             std::unordered_set<const HloInstruction*>& exclude_insts,
             std::vector<HloSubGraph>& sub_graphs) {
  MapInstToSubGraph(module, critical_insts, exclude_insts, sub_graphs);

  const HloComputation* entry = module->entry_computation();
  const HloInstruction* root = entry->root_instruction();

  // place root instruction into sub 0
  CHECK(exclude_insts.find(root) == exclude_insts.end());
  sub_graphs[0].insts_.emplace_back(root);
  exclude_insts.insert(root);

  std::unordered_map<const HloInstruction*, int> min_sub_id_map =
                                      GenMinSubIdMap(module, sub_graphs);

  for (auto& min_sub_id : min_sub_id_map) {
    sub_graphs[min_sub_id.second].insts_.emplace_back(min_sub_id.first);
    exclude_insts.insert(min_sub_id.first);
  }
}

std::vector<SharedDimStrategy>
CostSpmdStrategy::GenSplitProposals(const HloInstruction* inst) {
  CHECK(is_compute_intensive(inst));

  std::vector<SharedDimStrategy> res =
      StrategyUtil::GenSplitProposals(inst, cur_split_num_, false);

  return std::move(res);
}

std::vector<SharedDimStrategy> CostSpmdStrategy::GenSplitProposals(
                  const HloInstruction* inst,
                  const HloInstSet& split_for_mem_save) {
  CHECK(is_compute_intensive(inst));

  bool save_variable_mem = false;
  if (split_for_mem_save.find(inst) != split_for_mem_save.end()) {
    save_variable_mem = true;
  }
  std::vector<SharedDimStrategy> res =
      StrategyUtil::GenSplitProposals(inst, cur_split_num_, save_variable_mem);

  return std::move(res);
}

void
CostSpmdStrategy::GenComputeSplitProposalMap(
      const std::list<const HloInstruction*>& unsplitted_computes,
      const HloInstSet& split_for_mem_save,
      HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>& split_proposals) {
  VLOG(1) << "unsplitted_computes size: " << unsplitted_computes.size();
  if (unsplitted_computes.empty()) {
    VLOG(1) << "empty unsplitted_computes";
  }
  VLOG(1) << "split_for_mem_save size: " << split_for_mem_save.size();
  for (auto* unsplitted : unsplitted_computes) {
    // enumulate possible strategies:
    std::vector<SharedDimStrategy> proposals =
                          GenSplitProposals(unsplitted, split_for_mem_save);
    if (!proposals.empty()) {
      for (auto& str : proposals) {
        split_proposals[unsplitted].insert(str);
      }
    }
  }
}

GraphPieces
CostSpmdStrategy::BuildGraphPieces(
    const HloInstSet& user_annotated_set,
    const HloInstMap<SharedDimStrategy>& init_strategy_map, // after inference by user annotations
    const HloInstMap<SharedDimStrategy>& expected_strategies,
    const HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>& split_proposals,
    const HloInstSet& graph_scope,
    bool shrink) {
  GraphPieces pieces;

  VLOG(1) << "user_annotated_set size: " << user_annotated_set.size();
  VLOG(1) << "init_strategy_map size: " << init_strategy_map.size();
  VLOG(1) << "expected_strategies size: " << expected_strategies.size();
  VLOG(1) << "graph_scope size: " << graph_scope.size();
  record_conflict_ = true;
  for (auto& split_proposal : split_proposals) {
    const HloInstruction* unsplitted = split_proposal.first;

    const std::set<SharedDimStrategy, DimStrategyPtrLess>& proposals = split_proposal.second;
    
    // populate strategy to other instructions from each possible strategy
    for (auto& start_strtg : proposals) {
      VLOG(2) << "[shrink version] start from unsplitted compute: "
              << unsplitted->name() << ", with strategy: "
              << start_strtg->ToString();
      std::shared_ptr<GraphPiece> piece = BuildOnePiece(unsplitted,
                                                        *start_strtg,
                                                        user_annotated_set,
                                                        init_strategy_map,
                                                        expected_strategies,
                                                        graph_scope,
                                                        pieces,
                                                        shrink);
      if (piece.get() != nullptr) {
        pieces.AddPiece(piece);
        VLOG(2) << "[shrink version] one piece generated for " << unsplitted->name()
                << ", strategy: " << start_strtg->ToString();
        CHECK(piece->inst_map_.find(unsplitted) != piece->inst_map_.end())
            << "start inst: " << unsplitted->name();
      } else {
        VLOG(2) << "no piece generated for " << unsplitted->name();
      }
    }
  }

  // We set record_conflict_ false to forbid recording conflicts during the following koop
  record_conflict_ = false;
  for (auto& split_proposal : conflict_proposals_) {
    const HloInstruction* unsplitted = split_proposal.first;
    const std::set<SharedDimStrategy, DimStrategyPtrLess>& proposals = split_proposal.second;
    // populate strategy to other instructions from each possible strategy
    for (auto& start_strtg : proposals) {
      VLOG(2) << "[shrink version] start from unsplitted compute: "
              << unsplitted->name() << ", with strategy: "
              << start_strtg->ToString();
      std::shared_ptr<GraphPiece> piece = BuildOnePiece(unsplitted,
                                                        *start_strtg,
                                                        user_annotated_set,
                                                        init_strategy_map,
                                                        expected_strategies,
                                                        graph_scope,
                                                        pieces,
                                                        shrink);
      if (piece.get() != nullptr) {
        pieces.AddPiece(piece);
        VLOG(2) << "[shrink version] one piece generated for " << unsplitted->name()
                << ", strategy: " << start_strtg->ToString();
        CHECK(piece->inst_map_.find(unsplitted) != piece->inst_map_.end())
            << "start inst: " << unsplitted->name();
      } else {
        VLOG(2) << "no piece generated for " << unsplitted->name();
      }
    }
  }

  conflict_proposals_.clear();

  VLOG(1) << "after conflict, pieces num : " << pieces.pieces_.size();

  return std::move(pieces);
}


GraphPieces
CostSpmdStrategy::BuildGraphPieces(
              const HloInstSet& user_annotated_set,
              const HloInstMap<SharedDimStrategy>& init_strategy_map,
              const HloInstSet& graph_scope,
              const HloInstMap<SharedDimStrategy>& start_strategies) {
  GraphPieces pieces;
  const HloInstMap<SharedDimStrategy> dummy_expected_strategies;

  VLOG(1) << "start_strategies size: " << start_strategies.size();
  CHECK(!start_strategies.empty());

  for (auto& start_strtg : start_strategies) {
    VLOG(2) << "[non-shrink version] start from inst: "
            << start_strtg.first->name() << ", with strategy: "
            << start_strtg.second->ToString();
    std::shared_ptr<GraphPiece> piece = BuildOnePiece(start_strtg.first,
                                                      *start_strtg.second,
                                                      user_annotated_set,
                                                      init_strategy_map,
                                                      dummy_expected_strategies,
                                                      graph_scope,
                                                      pieces,
                                                      false);
    if (piece.get() != nullptr) {
      pieces.AddPiece(piece);
    }
  }

  return std::move(pieces);
}

std::shared_ptr<GraphPiece>
CostSpmdStrategy::BuildOnePiece(
              const HloInstruction* start_inst,
              const DimStrategy& strtg,
              const HloInstSet& user_annotated_set,
              const HloInstMap<SharedDimStrategy>& init_strategy_map, // may contain or not contain start_inst and its strategy
              const HloInstMap<SharedDimStrategy>& expected_strategies,
              const HloInstSet& graph_scope,
              const GraphPieces& pieces,
              bool do_shrink) {
  VLOG(2) << "BuildOnePiece: start inst: " << start_inst->name();
  for (auto& tmp : expected_strategies) {
    VLOG(2) << "expected inst: " << tmp.first->name() << ", strategy: " << tmp.second->ToString();
  }

  SharedDimStrategy strtg_ptr = std::make_shared<DimStrategy>(strtg);

  // 2.1. check if the start instruction with the strategy is
  //      'compatible' with some other compute instructions
  InstStrategy inst_strtg(start_inst, strtg_ptr);
  if (pieces.HasPiece(inst_strtg)) {
    // TODO(lansong): we may miss some strategies
    std::shared_ptr<GraphPiece> piece;  // nullptr
    CHECK(piece.get() == nullptr);
    VLOG(2) << "no piece generated";
    return piece;
  }

  VLOG(2) << "BuildOnePiece 1: start inst: " << start_inst->name() << ", strategy: " << strtg.ToString();

  // 2.2. populate strategy to other instructions
  GraphStrategy try_graph_strategy(cur_split_num_, cur_share_dev_);
  VLOG(2) << "initial strategies:";
  for (auto& inst_strtg : init_strategy_map) {
    VLOG(2) << "inst: " << inst_strtg.first->ToString();
    VLOG(2) << "strategy: " << inst_strtg.second->ToString();
    try_graph_strategy.SetInstStrategy(inst_strtg.first, inst_strtg.second);
  }
  try_graph_strategy.SetInstStrategy(start_inst, strtg_ptr);
  auto& strategy_map = try_graph_strategy.strategy_map();
  HloInstSet infered_insts = PopulateStrategy(start_inst, strtg_ptr,
                                              user_annotated_set,
                                              graph_scope,
                                              expected_strategies,
                                              try_graph_strategy);
  // record the starting instruction
  infered_insts.insert(start_inst);

  CHECK(strategy_map.find(start_inst) != strategy_map.end()) << start_inst->name();

  VLOG(2) << "inferred insts before shrink:";
  for (auto* infered_inst : infered_insts) {
    VLOG(2) << "inferred inst: " << infered_inst->name();
  }

  // 2.3. shrink the inferred instruction set
  if (do_shrink) {
    HloInstSet boundary_insts;
    bool changed;
    do {
      // find boundary instructions other than computing instructions(such as gemm)
      for (auto* infered_inst : infered_insts) {
        if (is_compute_intensive(infered_inst)) {
          continue;
        }
        bool read_by_external_only = true;
        for (auto* user : infered_inst->users()) {
          if (infered_insts.find(user) != infered_insts.end()) {
            read_by_external_only = false;
            break;
          }
        }
        if (read_by_external_only) {
          boundary_insts.insert(infered_inst);
        }

        bool read_external_only = true;
        for (auto* op : infered_inst->operands()) {
          if (infered_insts.find(op) != infered_insts.end()) {
            read_external_only = false;
            break;
          }
        }
        if (read_external_only) {
          boundary_insts.insert(infered_inst);
        }
      }

      changed = false;
      // erase boundary instructions
      for (auto* b_inst : boundary_insts) {
        if (b_inst->opcode() == HloOpcode::kParameter) {
          continue;
        }
        VLOG(2) << "erase boundary inst: " << b_inst->name();
        infered_insts.erase(b_inst);
        changed = true;
      }

      boundary_insts.clear();
    } while (changed);
  }


  int compute_count = 0;
  for (auto* infered_inst : infered_insts) {
    VLOG(2) << "inferred inst: " << infered_inst->name();
    if (is_compute_intensive(infered_inst)) {
      ++compute_count;
    }
  }

  if (infered_insts.empty()) {
    // don't create piece for zero instruction set
    std::shared_ptr<GraphPiece> piece;  // nullptr
    CHECK(piece.get() == nullptr);
    VLOG(2) << "no piece generated";
    return piece;
  }

  VLOG(2) << "BuildOnePiece 2: start inst: " << start_inst->name() << ", strategy: " << strtg.ToString();

  const HloInstMap<SharedDimStrategy>&
  try_strategy_map = try_graph_strategy.strategy_map();

  for (auto* infered_inst : infered_insts) {
    CHECK(try_strategy_map.find(infered_inst) != try_strategy_map.end());
    if (expected_strategies.find(infered_inst) != expected_strategies.end()) {
      if (*try_strategy_map.at(infered_inst) != *expected_strategies.at(infered_inst)) {
        VLOG(2) << "infered_inst: " << infered_inst->name() << ", inferred strategy: "
                << try_strategy_map.at(infered_inst)->ToString();
        VLOG(2) << "expected strategy: " << expected_strategies.at(infered_inst)->ToString();
        VLOG(2) << "no piece generated";
        std::shared_ptr<GraphPiece> piece;
        return piece;
      }
    }
  }

  VLOG(2) << "BuildOnePiece 3: start inst: " << start_inst->name() << ", strategy: " << strtg.ToString();

  CHECK(strategy_map.find(start_inst) != strategy_map.end()) << start_inst->name();

  VLOG(1) << "infered_insts size: " << infered_insts.size();
  // 3. build graph piece
  std::shared_ptr<GraphPiece> piece = std::make_shared<GraphPiece>();
  for (auto* infered_inst : infered_insts) {
    CHECK(try_strategy_map.find(infered_inst) != try_strategy_map.end());
    piece->Insert(infered_inst, try_strategy_map.at(infered_inst));
  }

  piece->start_inst_ = start_inst;
  
  VLOG(1) << "the piece: " << piece->ToString();
  VLOG(1) << "end of BuildOnePiece: start inst: " << start_inst->name() << ", strategy: " << strtg.ToString();

  return piece;
}

void CostSpmdStrategy::InferSubGraph(
    const HloInstSet& user_annotated_set,
    const HloInstMap<SharedDimStrategy>& init_strategy_map,  // strategies infered by user annotations
    const HloInstMap<SharedDimStrategy>& expected_strategies, // head & tail strategies
    const MemSavePlan& mem_save_plan,
    const HloInstMap<SharedDimStrategy>* acc_strtg_map,
    const InstAffinityMap& affinity_map,
    HloSubGraph& sub_graph,
    int opt_level) {
  const HloInstSet sub_graph_insts(sub_graph.insts_.begin(),
                                   sub_graph.insts_.end());
  HloInstMap<SharedDimStrategy> global_strategy_map;
  global_strategy_map = init_strategy_map;
  VLOG(1) << "init global_strategy_map size: " << global_strategy_map.size();
  if (acc_strtg_map) {
    global_strategy_map.insert(acc_strtg_map->begin(), acc_strtg_map->end());
  }
  global_strategy_map.insert(expected_strategies.begin(), expected_strategies.end());
  VLOG(1) << "after merging acc_strtg_map, global_strategy_map size: "
          << global_strategy_map.size();
  InferSubGraphByCone(
      user_annotated_set, expected_strategies, mem_save_plan, sub_graph_insts,
      affinity_map, sub_graph, global_strategy_map, opt_level);

  std::shared_ptr<SubGraphStrategy> sub_g_strtgy =
                                    std::make_shared<SubGraphStrategy>();

  HloInstMap<SharedDimStrategy> strategy_map;
  for (auto& strtg : global_strategy_map) {
    if (sub_graph_insts.find(strtg.first) != sub_graph_insts.end()) {
      strategy_map[strtg.first] = strtg.second;
    }
  }
  if (strategy_map.find(sub_graph.head_) != strategy_map.end()) {
    sub_g_strtgy->head_strategy_ = strategy_map.at(sub_graph.head_);
  }

  if (strategy_map.find(sub_graph.tail_) != strategy_map.end()) {
    sub_g_strtgy->tail_strategy_ = strategy_map.at(sub_graph.tail_);
    sub_graph.tail_strtg_map_[sub_g_strtgy->tail_strategy_].push_back(sub_g_strtgy);
  }

  int64 transfer_size = DataTransferSize(strategy_map, sub_graph_insts,
                                         cur_split_num_);

  sub_g_strtgy->self_cost_ = transfer_size;
  sub_g_strtgy->strategy_map_ = std::move(strategy_map);
  sub_g_strtgy->id_ = sub_graph.graph_strategies_.size();
  sub_g_strtgy->sub_graph_id_ = sub_graph.id_;

  sub_graph.graph_strategies_.push_back(sub_g_strtgy);
}

HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>
CostSpmdStrategy::InferStartingStrtgs(
          const HloInstMap<SharedDimStrategy>& acc_strtg_map,
          const HloInstSet& graph_scope) {
  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>> starting_strategies;
  for (auto& inst_strategy : acc_strtg_map) {
    if (inst_strategy.second->Glue() ||
        inst_strategy.second->IsPartial() ||
        inst_strategy.second->replicated()) {
      continue;
    }

    for (auto* user : inst_strategy.first->users()) {
      if ((!graph_scope.empty()) &&
          graph_scope.find(user) == graph_scope.end()) {
        continue;
      }
      if (acc_strtg_map.find(user) != acc_strtg_map.end()) {
        if (!acc_strtg_map.at(user)->Glue()) {
          starting_strategies[user].insert(acc_strtg_map.at(user));
        }
        continue;
      }

      // user's strategy is not determined and user is inside scope
      int input_idx = user->operand_index(inst_strategy.first);
      HloInstMap<SharedDimStrategy> infered_strtg_map =
            StrategyUtil::ForwardInfer(user, *inst_strategy.second, input_idx);

      for (auto& inst_strtg : infered_strtg_map) {
        const HloInstruction* infered_inst = inst_strtg.first;
        if (infered_inst != user) {
          continue;
        }
        starting_strategies[user].insert(inst_strtg.second);
      }
    }
  }

  return std::move(starting_strategies);
}

void CostSpmdStrategy::FastInferSubGraph(
    const HloInstSet& user_annotated_set,
    const HloInstMap<SharedDimStrategy>& init_strategy_map,  // strategies infered by user annotations
    const HloInstMap<SharedDimStrategy>& expected_strategies, // head & tail strategies
    const MemSavePlan& mem_save_plan,
    const HloInstMap<SharedDimStrategy>* acc_strtg_map,
    const InstAffinityMap& affinity_map,
    HloSubGraph& sub_graph,
    int opt_level) {
  const HloInstSet sub_graph_insts(sub_graph.insts_.begin(),
                                   sub_graph.insts_.end());
  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>> start_inst_strtgs;
  if (acc_strtg_map) {
    start_inst_strtgs = InferStartingStrtgs(*acc_strtg_map, sub_graph_insts);
  }

  // 1. find unsplitted gemm
  std::list<const HloInstruction*> unsplitted_computes =
                  CollectUnsplittedComputeInsts(sub_graph_insts, init_strategy_map);

  std::list<const HloInstruction*>::iterator unsplit_it = unsplitted_computes.begin();
  while (unsplit_it != unsplitted_computes.end()) {
    if (expected_strategies.find(*unsplit_it) != expected_strategies.end()) {
      unsplit_it = unsplitted_computes.erase(unsplit_it);
    } else {
      ++unsplit_it;
    }
  }

  // 2. build split proposals for all compute intensive instructions
  GenComputeSplitProposalMap(unsplitted_computes,
                             mem_save_plan.split_for_mem_save_,
                             start_inst_strtgs);

  for (auto& expected_strg : expected_strategies) {
    if (start_inst_strtgs.find(expected_strg.first) != start_inst_strtgs.end()) {
      start_inst_strtgs[expected_strg.first].clear();
    }

    start_inst_strtgs[expected_strg.first].insert(expected_strg.second);
  }

  // 3. heuristic: select max compute/communication ratio piece step by step
  GraphPieces pieces = BuildGraphPieces(user_annotated_set,
                                        init_strategy_map,
                                        expected_strategies,
                                        start_inst_strtgs,
                                        sub_graph_insts,
                                        true);
  for (auto& expected_strtg : expected_strategies) {
    CHECK(pieces.HasInstruction(expected_strtg.first))
        << expected_strtg.first->name();
  }

  for (auto* unsplitted : unsplitted_computes) {
    CHECK(pieces.HasInstruction(unsplitted)) << unsplitted->name() << " in sub graph: " << sub_graph.id_;
  }

  pieces.EvalScore(sub_graph_insts, cur_split_num_);

  // select pieces to combine the sub graph strategy
  HloInstMap<SharedDimStrategy> strategy_map;
  while (!pieces.is_empty()) {
    std::list<std::shared_ptr<GraphPiece>>::iterator it = pieces.pieces_.begin();
    std::list<std::shared_ptr<GraphPiece>>::iterator it_max = it;
    float max_score = (*it)->score_;
    ++it;
    for (; it != pieces.pieces_.end(); ++it) {
      if (max_score < (*it)->score_) {
        max_score = (*it)->score_;
        it_max = it;
      }
    }

    std::shared_ptr<GraphPiece> max_piece = (*it_max);
    pieces.pieces_.erase(it_max);

    for (auto& inst_strtg : max_piece->inst_map_) {
      CHECK(strategy_map.find(inst_strtg.first) == strategy_map.end());
      strategy_map[inst_strtg.first] = inst_strtg.second;
    }

    VLOG(1) << "piece with max score: " << max_piece->ToString();

    pieces.CutBy(max_piece);
    pieces.EvalScore(sub_graph_insts, cur_split_num_);
  }

  // 3. build cones, infer cones and stitch cones
  VLOG(1) << "sub id: " << sub_graph.id_ << ", str id: " << sub_graph.graph_strategies_.size()
          << ", strategy_map size before InferSubGraphByCone: " << strategy_map.size();
  FastInferSubGraphByCone(user_annotated_set, sub_graph_insts, affinity_map,
                      sub_graph, strategy_map, opt_level);
  VLOG(1) << "sub id: " << sub_graph.id_ << ", str id: " << sub_graph.graph_strategies_.size()
          << ", strategy_map size after InferSubGraphByCone: " << strategy_map.size();

  std::shared_ptr<SubGraphStrategy> sub_g_strtgy =
                                    std::make_shared<SubGraphStrategy>();
  VLOG(1) << "strategy_map size: " << strategy_map.size();

  // debug
  if (sub_graph.head_ &&
      strategy_map.find(sub_graph.head_) == strategy_map.end()) {
    VLOG(1) << "miss strategy for head: " << sub_graph.head_->ToString();
  }
  if (sub_graph.tail_ &&
      strategy_map.find(sub_graph.tail_) == strategy_map.end()) {
    VLOG(1) << "miss strategy for tail: " << sub_graph.tail_->ToString();
  }


  CHECK(strategy_map.find(sub_graph.head_) != strategy_map.end() ||
        strategy_map.find(sub_graph.tail_) != strategy_map.end());
  if (strategy_map.find(sub_graph.head_) != strategy_map.end()) {
    sub_g_strtgy->head_strategy_ = strategy_map.at(sub_graph.head_);
  }

  if (strategy_map.find(sub_graph.tail_) != strategy_map.end()) {
    sub_g_strtgy->tail_strategy_ = strategy_map.at(sub_graph.tail_);
    sub_graph.tail_strtg_map_[sub_g_strtgy->tail_strategy_].push_back(sub_g_strtgy);
  }

  for (auto* unsplitted : unsplitted_computes) {
    if (strategy_map.find(unsplitted) == strategy_map.end()) {
      VLOG(1) << "sub graph: " << sub_graph.id_ << ", strategy map:";
      for (auto& inst_strtg : strategy_map) {
        VLOG(1) << "inst: " << inst_strtg.first->name() << ", strategy: "
                << inst_strtg.second->ToString();
      }
      VLOG(1) << "end of sub graph: " << sub_graph.id_;
    } else {
      VLOG(1) << "sub graph: " << sub_graph.id_ << " contain strategy for " << unsplitted->name();
    }
    CHECK(strategy_map.find(unsplitted) != strategy_map.end())
        << "strategy is missed for " << unsplitted->name() << " in sub graph: "
        << sub_graph.id_;
  }

  int64 transfer_size = DataTransferSize(strategy_map, sub_graph_insts,
                                         cur_split_num_);

  sub_g_strtgy->self_cost_ = transfer_size;
  sub_g_strtgy->strategy_map_ = std::move(strategy_map);
  sub_g_strtgy->id_ = sub_graph.graph_strategies_.size();
  sub_g_strtgy->sub_graph_id_ = sub_graph.id_;

  VLOG(1) << "sub graph: " << sub_g_strtgy->sub_graph_id_
          << ", sub graph str id: " << sub_g_strtgy->id_
          << ", strategy map size: " << sub_g_strtgy->strategy_map_.size();

  sub_graph.graph_strategies_.push_back(sub_g_strtgy);
}

void CostSpmdStrategy::FastInferSubGraphByCone(
          const HloInstSet& user_annotated_set,
          const HloInstSet& graph_scope,  // sub graph instructions
          const InstAffinityMap& affinity_map,
          HloSubGraph& sub_graph,
          HloInstMap<SharedDimStrategy>& strategy_map,
          int opt_level) {
  HloInstMap<SharedDimStrategy> start_strategies;
  start_strategies = FindStartingStrategies(strategy_map, graph_scope);

  HloInstMap<SharedDimStrategy> bound_strategies;
  VLOG(0) << "strategy_map size: " << strategy_map.size();
  bound_strategies = FindBackStartingStrategies(strategy_map, graph_scope);
  bound_strategies.insert(start_strategies.begin(), start_strategies.end());
  // NOTE: strategies in bound_strategies are already inferred

  if (sub_graph.id_ == 1) {
    for (auto& tmp_str : strategy_map) {
      VLOG(2) << "str map: inst: " << tmp_str.first->ToString() << std::endl
              << "strategy: " << tmp_str.second->ToString();
    }

    for (auto* scope_inst : graph_scope) {
      VLOG(2) << "scope inst in post: " << scope_inst->name();
    }

    for (auto* scope_inst : graph_scope) {
      if (strategy_map.find(scope_inst) == strategy_map.end()) {
        VLOG(2) << "no strategy for inst: " << scope_inst->name();
      }
    }

    VLOG(2) << "bound size: " << bound_strategies.size();
    for (auto& b_str : bound_strategies) {
      VLOG(2) << "bound str: " << b_str.first->name() << ", strategy: "
              << b_str.second->ToString();
    }
  }

  if (bound_strategies.empty()) {
    VLOG(0) << "all instruction strategies are determined in current sub graph";
    return;
  }

  GraphPieces pieces = BuildGraphPieces(user_annotated_set,
                                        strategy_map,
                                        graph_scope,
                                        bound_strategies);

  // collect candidate splittings of instruction
  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>
  split_candidates = CollectInstSplits(pieces, strategy_map, graph_scope);

  std::unordered_set<const HloInstruction*> cone_roots =
                          sub_graph.BuildInstCones(graph_scope, opt_level);

  std::string cones_str = sub_graph.ConesToString(); // no cone strategy here
  VLOG(1) << "cones:" << std::endl << cones_str;

  // infer sub graph by dynamic programming algorithm
  HloInstMap<std::unique_ptr<InstCosts>> inst_costs =
      InferInstCosts(cone_roots, split_candidates, graph_scope, strategy_map);

  VLOG(1) << "strategy_map size before StitchInstCones: " << strategy_map.size();
  StitchInstCones(inst_costs, graph_scope, affinity_map, sub_graph, strategy_map);
  VLOG(1) << "strategy_map size after StitchInstCones: " << strategy_map.size();

  bool debug_mode = ServiceEnv::debug();

  if (debug_mode) {
    std::string inst_cost_str = InstCostsToString(inst_costs, sub_graph.id_);
    std::string cone_list_str = sub_graph.ConesToString();  // with cone strategy here
    tensorflow::Env* env = tensorflow::Env::Default();
    string fname = "cone_info." + std::to_string(cur_split_ordinal_) + ".txt";
    Status status = tensorflow::WriteStringToFile(env, fname,
                                                  cone_list_str+inst_cost_str);

    if (!status.ok()) {
      LOG(ERROR) << "Could not write comm info to " << fname << ": " << status;
    }
  }

  for (auto* scope_inst : graph_scope) {
    if (strategy_map.find(scope_inst) == strategy_map.end()) {
      strategy_map[scope_inst] = std::make_shared<DimStrategy>();
      VLOG(1) << "set glue strategy for missed inst: " << scope_inst->name();
    }
  }
}

void CostSpmdStrategy::InferSubGraphByCone(
    const HloInstSet& user_annotated_set,
    const HloInstMap<SharedDimStrategy>& expected_strategies, // head & tail strategies
    const MemSavePlan& mem_save_plan,
    const HloInstSet& graph_scope,  // sub graph instructions
    const InstAffinityMap& affinity_map,
    HloSubGraph& sub_graph,
    HloInstMap<SharedDimStrategy>& strategy_map,  // whole graph strategy map
    int opt_level) {
  HloInstMap<SharedDimStrategy> start_strategies;
  start_strategies = FindStartingStrategies(strategy_map, graph_scope);

  HloInstMap<SharedDimStrategy> bound_strategies;
  bound_strategies = FindBackStartingStrategies(strategy_map, graph_scope);
  bound_strategies.insert(start_strategies.begin(), start_strategies.end());
  // NOTE: strategies in bound_strategies are already inferred

  if (sub_graph.id_ == 1) {
    for (auto& tmp_str : strategy_map) {
      VLOG(2) << "str map: inst: " << tmp_str.first->ToString() << std::endl
              << "strategy: " << tmp_str.second->ToString();
    }

    for (auto* scope_inst : graph_scope) {
      VLOG(2) << "scope inst in post: " << scope_inst->name();
    }

    for (auto* scope_inst : graph_scope) {
      if (strategy_map.find(scope_inst) == strategy_map.end()) {
        VLOG(2) << "no strategy for inst: " << scope_inst->name();
      }
    }

    VLOG(2) << "bound size: " << bound_strategies.size();
    for (auto& b_str : bound_strategies) {
      VLOG(2) << "bound str: " << b_str.first->name() << ", strategy: " << b_str.second->ToString();
    }
  }

  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>> start_inst_strtgs;
  start_inst_strtgs = InferStartingStrtgs(strategy_map, graph_scope);

  std::list<const HloInstruction*> unsplitted_computes =
                          CollectUnsplittedComputeInsts(graph_scope, strategy_map);

  std::list<const HloInstruction*>::iterator unsplit_it = unsplitted_computes.begin();
  while (unsplit_it != unsplitted_computes.end()) {
    if (expected_strategies.find(*unsplit_it) != expected_strategies.end()) {
      unsplit_it = unsplitted_computes.erase(unsplit_it);
    } else {
      ++unsplit_it;
    }
  }

  GenComputeSplitProposalMap(unsplitted_computes,
                             mem_save_plan.split_for_mem_save_,
                             start_inst_strtgs);

  for (auto& expected_strg : expected_strategies) {
    if (start_inst_strtgs.find(expected_strg.first) != start_inst_strtgs.end()) {
      start_inst_strtgs[expected_strg.first].clear();
    }

    start_inst_strtgs[expected_strg.first].insert(expected_strg.second);
  }

  GraphPieces pieces = BuildGraphPieces(user_annotated_set,
                                        strategy_map, // whole graph strategy map
                                        bound_strategies,
                                        start_inst_strtgs,
                                        graph_scope,
                                        false);

  HloInstMap<SharedDimStrategy> aff_strategies;
  for (auto inst_pair : affinity_map.AllAffinities()) {
    VLOG(2) << "affinity first " << inst_pair.first->name();
    VLOG(2) << "shape: " << inst_pair.first->shape().ToString();
    VLOG(2) << "affinity second " << inst_pair.second->name();
    VLOG(2) << "shape: " << inst_pair.second->shape().ToString();
    CHECK(inst_pair.first->shape() == inst_pair.second->shape());
    if (strategy_map.find(inst_pair.first) != strategy_map.end()) {
      aff_strategies[inst_pair.second] = strategy_map.at(inst_pair.first);
    }
  }

  // collect candidate splittings of instruction
  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>
  split_candidates = CollectInstSplits(pieces, strategy_map, graph_scope);

  for (auto& aff_expected : aff_strategies) {
    if (split_candidates.find(aff_expected.first) == split_candidates.end()) {
      continue;
    }

    std::set<SharedDimStrategy, DimStrategyPtrLess>& candidates =
                                          split_candidates[aff_expected.first];
    if (candidates.find(aff_expected.second) != candidates.end()) {
      if (candidates.size() > 1) {
        VLOG(2) << "Remove some strategy candidates for "
                << aff_expected.first->name()
                << ". Only following candidate is remained:\n"
                << aff_expected.second->ToString();
        candidates.clear();
        candidates.insert(aff_expected.second);
      } else {
        // lansong: do nothing. It should be refined later.
      }
    }
  }

  std::unordered_set<const HloInstruction*> cone_roots =
                          sub_graph.BuildInstCones(graph_scope, opt_level);

  std::string cones_str = sub_graph.ConesToString(); // no cone strategy here
  VLOG(1) << "cones:" << std::endl << cones_str;

  // infer sub graph by dynamic programming algorithm
  HloInstMap<std::unique_ptr<InstCosts>> inst_costs =
      InferInstCosts(cone_roots, split_candidates, graph_scope, strategy_map);

  VLOG(1) << "strategy_map size before StitchInstCones: " << strategy_map.size();
  StitchInstCones(inst_costs, graph_scope, affinity_map, sub_graph, strategy_map);
  VLOG(1) << "strategy_map size after StitchInstCones: " << strategy_map.size();

  bool debug_mode = ServiceEnv::debug();
  if (debug_mode) {
    std::string inst_cost_str = InstCostsToString(inst_costs, sub_graph.id_);
    std::string cone_list_str = sub_graph.ConesToString(); // with cone strategy here
    tensorflow::Env* env = tensorflow::Env::Default();
    string fname = "cone_info." + std::to_string(cur_split_ordinal_) + ".sub"
                   + std::to_string(sub_graph.id_) + ".txt";
    Status status = tensorflow::WriteStringToFile(env, fname,
                                                  cone_list_str+inst_cost_str);

    if (!status.ok()) {
      LOG(ERROR) << "Could not write comm info to " << fname << ": " << status;
    }
  }

  for (auto* scope_inst : graph_scope) {
    if (strategy_map.find(scope_inst) == strategy_map.end()) {
      strategy_map[scope_inst] = std::make_shared<DimStrategy>();
      VLOG(1) << "set glue strategy for missed inst: " << scope_inst->name();
    }
  }
}

std::vector<const HloInstruction*>
HloSubGraph::FindConeRoots(
    const std::vector<const HloInstruction*>& sorted_insts,
    const HloInstSet& graph_scope,  // sub graph instructions
    int opt_level) {
  std::vector<const HloInstruction*> all_roots;
  std::unordered_set<const HloInstruction*> root_set;
  for (auto* inst : sorted_insts) {
    VLOG(1) << "sorted inst: " << inst->name();
    if (opt_level>=4) {
      all_roots.emplace_back(inst);
      continue;
    }
    // 1. instruction with mult local users:
    //    it is a root
    int local_user_count = 0;
    for (auto* user : inst->users()) {
      if (graph_scope.find(user) != graph_scope.end()) {
        ++local_user_count;
        if (local_user_count>1 && root_set.find(inst) == root_set.end()) {
          all_roots.emplace_back(inst);
          root_set.insert(inst);
          break;
        }
      }
    }

    // For tail instruction
    if (local_user_count==0 && root_set.find(inst) == root_set.end()) {
      all_roots.emplace_back(inst);
      root_set.insert(inst);
    }

    if (opt_level>=2) {
      // 2. instruction with mult local operands:
      //    itself and its local operands are roots
      if (inst->operand_count()>1) {
        std::vector<const HloInstruction*> local_ops;
        for (auto* op : inst->operands()) {
          if (graph_scope.find(op) != graph_scope.end()) {
            local_ops.emplace_back(op);
          }
        }

        if (local_ops.size() > 1) {
          for (auto* op : local_ops) {
            if (root_set.find(op) == root_set.end()) {
              all_roots.emplace_back(op);
              root_set.insert(op);
            }
          }
          if (root_set.find(inst) == root_set.end()) {
            all_roots.emplace_back(inst);
            root_set.insert(inst);
          }
        }
      }
    }
  }

  return std::move(all_roots);
}

std::string HloSubGraph::ConesToString() const {
  std::string res;
  for (auto& cone : inst_cones_) {
    res += cone->ToString() + "\n";
  }

  res += "\n";

  return std::move(res);
}

std::unordered_set<const HloInstruction*>
HloSubGraph::BuildInstCones(
    const HloInstSet& graph_scope,
    int opt_level) {  // sub graph instructions
  inst_cones_.clear();

  std::vector<const HloInstruction*> sorted_insts = StableSort(graph_scope);
  std::vector<const HloInstruction*> sorted_roots = FindConeRoots(
                                                              sorted_insts,
                                                              graph_scope,
                                                              opt_level);
  std::unordered_set<const HloInstruction*> root_set;
  root_set.insert(sorted_roots.begin(), sorted_roots.end());

  for (auto* root : sorted_roots) {
    VLOG(1) << "root name: " << root->name();
    std::shared_ptr<InstCone> cone = ExtractCone(root, graph_scope, root_set);
    cone->id_ = inst_cones_.size();
    inst_cones_.push_back(cone);
  }

  return std::move(root_set);
}

void
HloSubGraph::ExtractConeStrategy(
    const HloInstMap<std::unique_ptr<InstCosts>>& inst_costs,
    const std::unordered_set<const HloInstruction*>& cone_scope,
    const HloInstruction* inst,
    int strategy_idx,
    HloInstMap<SharedDimStrategy>& cone_strtg_map) {
  if (strategy_idx < 0) {
    return;
  }

  CHECK(inst_costs.find(inst) != inst_costs.end());
  const std::unique_ptr<InstCosts>& costs = inst_costs.at(inst);
  std::unique_ptr<InstCosts::OneStrategyCost>& one_strtg_cost =
                                            costs->get_one_cost(strategy_idx);

  cone_strtg_map[inst] = one_strtg_cost->strategy_;

  for (int op_idx = 0; op_idx < inst->operand_count(); ++op_idx) {
    const HloInstruction* op = inst->operand(op_idx);
    if (cone_scope.find(op) == cone_scope.end()) {
      continue;
    }

    if (cone_strtg_map.find(op) != cone_strtg_map.end()) {
      continue;
    }

    int op_strtg_idx = one_strtg_cost->op_strtg_idx(op_idx);
    VLOG(2) << "inst name: " << inst->name() << ", op idx: "
            << op_idx << ", op_strtg_idx: " << op_strtg_idx;
    ExtractConeStrategy(inst_costs, cone_scope, op, op_strtg_idx, cone_strtg_map);
  }
}

void CostSpmdStrategy::StitchInstCones(
    const HloInstMap<std::unique_ptr<InstCosts>>& inst_costs,
    const HloInstSet& graph_scope,
    const InstAffinityMap& affinity_map,
    HloSubGraph& sub_graph,
    HloInstMap<SharedDimStrategy>& strategy_map) {   // whole graph strategy map
  // finalize instruction cone if its root has only one strategy
  std::unordered_set<const HloInstruction*> finalized_insts;
  std::list<std::shared_ptr<InstCone>> inst_cone_list(
                                            sub_graph.inst_cones_.begin(),
                                            sub_graph.inst_cones_.end());
  std::list<std::shared_ptr<InstCone>>::iterator it = inst_cone_list.begin();

  VLOG(1) << "origin instruction cone num: " << inst_cone_list.size();
  VLOG(1) << "origin strategy_map size: " << strategy_map.size();

  while (it != inst_cone_list.end()) {
    const HloInstruction* cone_root = (*it)->root_inst_;
    CHECK(inst_costs.find(cone_root) != inst_costs.end());
    const std::unique_ptr<InstCosts>& costs = inst_costs.at(cone_root);
    if (!costs->is_mult_strategies()) {
      // single strategy
      HloInstMap<SharedDimStrategy> cone_strtg_map;
      sub_graph.ExtractConeStrategy(inst_costs, (*it)->insts_, cone_root, 0, cone_strtg_map);

      for (auto& inst_strtg : cone_strtg_map) {
        const HloInstruction* inst = inst_strtg.first;
        SharedDimStrategy& strtg = inst_strtg.second;
        if (strategy_map.find(inst) != strategy_map.end()) {
          CHECK(strategy_map.at(inst)->Match(*strtg)) << "inst: " << inst->name()
              << "\nstrategy in map: " << strategy_map.at(inst)->ToString()
              << "\nstrategy in cone: " << strtg->ToString();
        } else {
          VLOG(2) << "add inst: " << inst->name();
          strategy_map[inst] = strtg;
        }
      }
      it = inst_cone_list.erase(it);
      continue;
    } else {
      ++it;
    }
  }

  VLOG(1) << "after finalize single strategy cone, instruction cone num: "
          << inst_cone_list.size();
  VLOG(1) << "after finalize single strategy cone, strategy_map size: "
          << strategy_map.size();

  if (!inst_cone_list.empty()) {
    sub_graph.BuildConeStrtgMngr(strategy_map, inst_costs, inst_cone_list,
                                 cur_split_num_, cur_share_dev_);

    ILPModel ilp_mod;
    sub_graph.BuildILPModel(inst_cone_list, affinity_map, strategy_map, ilp_mod);
    std::string ilp_mod_str = ilp_mod.ExportToString();
    VLOG(1) << "ilp model:";
    VLOG(1) << ilp_mod_str;

    std::unordered_map<int/*cone id*/, int/*str id*/> str_ids = ilp_mod.Solve();
    CHECK(inst_cone_list.size() == str_ids.size())
                << "inst_cone_list.size: " << inst_cone_list.size()
                << ", str_ids.size: " << str_ids.size();
    int count = 0;
    for (auto& cone : inst_cone_list) {
      CHECK(str_ids.find(cone->id_) != str_ids.end());
      int cone_str_id = str_ids[cone->id_];
      const std::unique_ptr<ConeStrategy>& cone_str = cone->strategy(cone_str_id);
      const HloInstMap<SharedDimStrategy>&
      cone_str_map = cone_str->strategy_map_;

      for (auto& inst_strtg : cone_str_map) {
        const HloInstruction* inst = inst_strtg.first;
        const SharedDimStrategy& strtg = inst_strtg.second;
        if (strategy_map.find(inst) != strategy_map.end()) {
          CHECK(strategy_map.at(inst)->Match(*strtg)) << "inst: " << inst->name();
        } else {
          VLOG(1) << "ilp result: sub graph: " << sub_graph.id_ << ", inst: "
                  << inst->name() << ", strategy: " << strtg->ToString();
          ++count;
          strategy_map[inst] = strtg;
        }
      }
    }

    VLOG(0) << "ilp infer " << count << " instructions for sub graph " << sub_graph.id_;
    VLOG(0) << "after ilp, strategy_map size: " << strategy_map.size();
  }
}

void HloSubGraph::BuildConeStrtgMngr(
    const HloInstMap<SharedDimStrategy>& ctx_strategy_map,
    const HloInstMap<std::unique_ptr<InstCosts>>& inst_costs,
    std::list<std::shared_ptr<InstCone>>& inst_cones,
    int split_num,
    bool share_dev) {
  cone_strtg_mngr_ = absl::make_unique<ConeStrategyManager>();
  // 1. extract cone strategy
  VLOG(1) << "multi-strategy cone: " << inst_cones.size();
  for (auto& inst_cone : inst_cones) {
    VLOG(1) << inst_cone->ToString();
    const HloInstruction* root = inst_cone->root_inst_;
    CHECK(inst_costs.find(root) != inst_costs.end());
    const std::unique_ptr<InstCosts>& root_costs = inst_costs.at(root);
    CHECK(root_costs->strtg_costs_.size()>0);
    for (int i=0; i<root_costs->strtg_costs_.size(); ++i) {
      std::unique_ptr<ConeStrategy> cone_strtg = std::make_unique<ConeStrategy>();
      ExtractConeStrategy(inst_costs, inst_cone->insts_, root, i,
                          cone_strtg->strategy_map_);
      inst_cone->AddConeStrategy(cone_strtg);
    }
  }

  // 2. build instruction-cone map
  std::unordered_map<const HloInstruction*, std::shared_ptr<InstCone>>&
  cone_map = cone_strtg_mngr_->cone_map_;

  for (auto& inst_cone : inst_cones) {
    for (auto* inst : inst_cone->insts_) {
      cone_map[inst] = inst_cone;
    }
  }

  // 3. build cone adjacency matrix(use map instead of matrix)
  int src_cone_id, tgt_cone_id;
  for (auto& inst_cone : inst_cones) {
    for (auto* inst : inst_cone->insts_) {
      // 3.1. handle operands
      for (auto* op : inst->operands()) {
        if (inst_cone->insts_.find(op) == inst_cone->insts_.end()) {
          // op is out of current cone
          tgt_cone_id = inst_cone->id_;
          if (cone_map.find(op) != cone_map.end()) {
            src_cone_id = cone_map.at(op)->id_;
          } else {
            src_cone_id = -1;
          }
          cone_strtg_mngr_->AddAdjCones(src_cone_id, tgt_cone_id);
        }
      }

      // 3.2 handle users
      for (auto* user : inst->users()) {
        if (cone_map.find(user) == cone_map.end()) {
          // user is not contained by any cone
          tgt_cone_id = -1;
          src_cone_id = inst_cone->id_;
          cone_strtg_mngr_->AddAdjCones(src_cone_id, tgt_cone_id);
        }
      }
    }
  }

  // 4. build cone-cone tensor transfer cost map
  for (auto& inst_cone : inst_cones) {
    inst_cone->BuildInputCost(ctx_strategy_map, cone_map, split_num, share_dev);
    inst_cone->BuildSelfCost(share_dev, split_num);
    inst_cone->BuildOutToCtxCost(ctx_strategy_map, cone_map, split_num, share_dev);
  }

  // 5. build cost matrix
  std::map<std::pair<int/*pred cone id*/, int/*user cone id*/>,
                     std::vector<int64/*cost*/>/*1D to store matrix*/>&
  inter_cost_map = cone_strtg_mngr_->inter_cost_map_;

  for (auto& usr_cone : inst_cones) {
    int usr_cone_id = usr_cone->id_;
    int usr_str_num = usr_cone->strategy_num();

    for (auto& usr_str : usr_cone->strategies()) {
      int usr_str_id = usr_str->id_;
      for (auto& src_info : usr_str->cone_in_costs_) {
        int src_cone_id = src_info.first.first;
        std::shared_ptr<InstCone> src_cone = this->inst_cones_[src_cone_id];
        int src_str_id = src_info.first.second;
        std::pair<int, int> key = std::make_pair(src_cone_id, usr_cone_id);
        int src_str_num = src_cone->strategy_num();
        if (inter_cost_map.find(key) == inter_cost_map.end()) {
          CHECK(src_str_num>0 && usr_str_num>0)
                << "src_str_num: " << src_str_num
                << ", usr_str_num: " << usr_str_num;
          int matrix_elem_num = src_str_num * usr_str_num;
          std::vector<int64/*cost*/> cost_matrix(matrix_elem_num, 0);
          inter_cost_map[key] = std::move(cost_matrix);
        }
        int64 cost = src_info.second;

        // matrix shape: (usr_str_num, src_str_num)
        int cost_offset = cone_strtg_mngr_->MatrixAddrToOffset(src_str_num,
                                                              src_str_id,
                                                              usr_str_id);
        CHECK(src_str_id < src_str_num && usr_str_id < usr_str_num);
        inter_cost_map[key][cost_offset] = cost;
      }

      // build context cost:
      std::pair<int, int> ctx_key1 = std::make_pair(-1, usr_cone_id);
      std::pair<int, int> ctx_key2 = std::make_pair(usr_cone_id, -1);
      if (inter_cost_map.find(ctx_key1) == inter_cost_map.end()) {
        CHECK(usr_str_num>0) << "usr_str_num: " << usr_str_num;
        std::vector<int64/*cost*/> cost_matrix(usr_str_num, 0);
        inter_cost_map[ctx_key1] = std::move(cost_matrix);
      }

      if (inter_cost_map.find(ctx_key2) == inter_cost_map.end()) {
        CHECK(usr_str_num>0) << "usr_str_num: " << usr_str_num;
        std::vector<int64/*cost*/> cost_matrix(usr_str_num, 0);
        inter_cost_map[ctx_key2] = std::move(cost_matrix);
      }

      CHECK(usr_str_id < usr_str_num);

      inter_cost_map[ctx_key1][usr_str_id] += usr_str->ctx_in_cost_;
      inter_cost_map[ctx_key2][usr_str_id] += usr_str->ctx_out_cost_;
    }
  }

  bool debug_mode = ServiceEnv::debug();
  if (debug_mode) {
    std::string inter_cost_info = DumpInterCost();
    tensorflow::Env* env = tensorflow::Env::Default();
    Status status = tensorflow::WriteStringToFile(env, "inter_cost_info.txt",
                                                  inter_cost_info);
    if (!status.ok()) {
      LOG(ERROR) << "Could not write comm info to inter_cost_info.txt: " << status;
    }
  }
}

std::string HloSubGraph::DumpInterCost() {
  std::map<std::pair<int/*pred cone id*/, int/*user cone id*/>,
                     std::vector<int64/*cost*/>/*1D to store matrix*/>&
  inter_cost_map = cone_strtg_mngr_->inter_cost_map_;

  std::string cost_info;
  cost_info = "  src_str -> usr_str: cost\n";
  for (auto& inter_cost : inter_cost_map) {
    int src_cone_id = inter_cost.first.first;
    int usr_cone_id = inter_cost.first.second;

    std::string src_root_name = "ctx";
    int src_str_num = 1;
    if (src_cone_id >= 0) {
      std::shared_ptr<InstCone> src_cone = this->inst_cones_[src_cone_id];
      src_str_num = src_cone->strategy_num();
      src_root_name = src_cone->root_inst_->name();
    }

    std::string usr_root_name = "ctx";
    int usr_str_num = 1;
    if (usr_cone_id >= 0) {
      std::shared_ptr<InstCone> usr_cone = this->inst_cones_[usr_cone_id];
      usr_str_num = usr_cone->strategy_num();
      usr_root_name = usr_cone->root_inst_->name();
    }

    cost_info += "id(" + std::to_string(src_cone_id) + ", "
                 + std::to_string(usr_cone_id) + "), name(" + src_root_name
                 + ", " + usr_root_name + "):\n";
    CHECK(inter_cost.second.size() == src_str_num*usr_str_num)
          << "elem count of cost matrix: " << inter_cost.second.size()
          << "src_str_num: " << src_str_num
          << "usr_str_num: " << usr_str_num;
    int src_str_id, usr_str_id;
    for (int k=0; k<inter_cost.second.size(); ++k) {
      cone_strtg_mngr_->OffsetToMatrixAddr(src_str_num, k, src_str_id, usr_str_id);
      cost_info += "  " + std::to_string(src_str_id) + " -> "
                   + std::to_string(usr_str_id) + ": "
                   + std::to_string(inter_cost.second[k]) + "\n";
    }
  }

  return std::move(cost_info);
}

#if 0
// TODO(lansong): neighbor vote algorithm may be useful as heuristic
void HloSubGraph::NeighborVote(
    std::list<std::shared_ptr<InstCone>>& inst_cones) {
  // 1. initialize probability for each instruction in multi-strategy cone
  VLOG(1) << "multi-strategy cone info: " << inst_cones.size();
  for (auto& inst_cone : inst_cones) {
    inst_cone->InitializeProb();
    VLOG(1) << inst_cone->ToString();
  }

  // 2. neighbor voting loop
  int iteration_num = 10;
  do {
    // ??????

    --iteration_num;
  } while (iteration_num>0);
}
#endif

std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>>
ILPModel::BuildVarMap() {
  auto extract_var_map = [&](
      std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>>& var_map) {
    var_map.clear();
    // extract var from constraints expr
    for (auto& constraint : constraints_) {
      int row_id = constraint->id_;
      std::shared_ptr<ILPSumExpr> sum_expr = constraint->left_;
      for (auto& prim_expr : sum_expr->operands_) {
        int var_id = prim_expr->var_id();  // column id
        int scale = prim_expr->scale();    // scale
        auto& row_scale_map = var_map[var_id];
        row_scale_map[row_id] = (double)scale;
      }
    }
  };

  std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>> var_map;
  extract_var_map(var_map);
  std::unordered_set<int> invalid_vars;
  for (int i=0; i<(all_str_self_vars_.size()+all_str_edge_vars_.size()); ++i) {
    if (var_map.find(i) == var_map.end()) {
      invalid_vars.insert(i);
    }
  }

  if (!invalid_vars.empty()) {
    // remove invalid var
    VLOG(0) << "some opt var is not referenced by constraints and opt object";

    std::vector<std::shared_ptr<ILPSelfVarExpr>> new_all_str_self_vars;
    std::vector<std::shared_ptr<ILPEdgeVarExpr>> new_all_str_edge_vars;

    for (auto& self_var : all_str_self_vars_) {
      if (invalid_vars.find(self_var->id_) == invalid_vars.end()) {
        self_var->id_ = new_all_str_self_vars.size();
        new_all_str_self_vars.emplace_back(self_var);
      }
    }

    int edge_offset = new_all_str_self_vars.size();
    for (auto& edge_var : all_str_edge_vars_) {
      if (invalid_vars.find(edge_var->id_) == invalid_vars.end()) {
        edge_var->id_ = edge_offset + new_all_str_edge_vars.size();
        new_all_str_edge_vars.emplace_back(edge_var);
      }
    }

    all_str_self_vars_ = std::move(new_all_str_self_vars);
    all_str_edge_vars_ = std::move(new_all_str_edge_vars);
    extract_var_map(var_map);
  }

  return std::move(var_map);
}

std::unordered_map<int/*cone id*/, int/*strategy id*/>
ILPModel::Solve() {
  // solve by Cbc
  int self_var_num = all_str_self_vars_.size();
  for (auto& edge_var : all_str_edge_vars_) {
    edge_var->id_ += self_var_num;
  }

  std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>> var_map = BuildVarMap();

  int row_num = constraints_.size();
  int col_num = var_map.size();
  int coeff_num = 0;
  for (auto& var_info : var_map) {
    coeff_num += var_info.second.size();
  }
  VLOG(0) << "coeff_num: " << coeff_num;
  VLOG(0) << "row num: " << row_num;
  VLOG(0) << "col num: " << col_num;
  CHECK(coeff_num > 0);

  // matrix data - column ordered
  CoinBigIndex* col_start = new CoinBigIndex[col_num+1]; // start
  int* col_length = new int[col_num];   // length
  int* coeff_rows = new int[coeff_num]; // rows
  double* coeffs = new double[coeff_num];  // elements(values)

  double* objective = new double[col_num];
  double* row_lower = new double[row_num];
  double* row_upper = new double[row_num];
  double* col_lower = new double[col_num];
  double* col_upper = new double[col_num];

  int coeff_id = 0;
  // fill coeffs, coeff_rows, col_strart and col_length array
  for (auto& var_info : var_map) {
    int var_id = var_info.first;  // column id
    col_start[var_id] = coeff_id;
    col_length[var_id] = var_info.second.size();

    for (auto& row_scale : var_info.second) {
      coeff_rows[coeff_id] = row_scale.first;
      coeffs[coeff_id] = row_scale.second;
      ++coeff_id;
    }
  }
  col_start[col_num] = coeff_id;

  // initialze objective, col_lower and col_upper array
  for (int i=0; i<col_num; ++i) {
    objective[i] = 0.0;
    col_lower[i] = 0;
    col_upper[i] = 1;
  }

  // fill objective, col_lower, col_upper array
  for (std::shared_ptr<ILPPrimExpr>& prim : opt_obj_->operands_) {
    objective[prim->var_id()] = (double)prim->scale();
  }

  for (int i=0; i<row_num; ++i) {
    std::shared_ptr<ILPEqualIntExpr>& eq = constraints_[i];
    CHECK(eq);
    row_lower[i] = (double)eq->right_;
    row_upper[i] = (double)eq->right_;
  }

  CoinPackedMatrix matrix(true, row_num, col_num, coeff_num, coeffs,
                          coeff_rows, col_start, col_length);
  OsiClpSolverInterface solver;
  // load problem
  solver.loadProblem(matrix, col_lower, col_upper,
                     objective, row_lower, row_upper);

  for (int i=0; i<col_num; ++i) {
    solver.setInteger(i);
  }

  // Solve
  // Currently we encounter a bug in linear relaxations caused infeasible
  // solution space.
  //solver.initialSolve(); // linear relaxation

  //solver.setObjSense(-1.0);

  // Pass data and solver to CbcModel
  CbcModel cbc_model(solver);

  int64 time_limit = ServiceEnv::ilp_time_limit();
  if (time_limit > 0) {
    double time_limit_sec = (double)(time_limit*60);
    cbc_model.setMaximumSeconds(time_limit_sec);
    LOG(INFO) << "Setting ILP_TIME_LIMIT as " << time_limit << " minutes.";
  }

  int64 num_threads = ServiceEnv::ilp_num_threads();
  if (num_threads > 0) {
    cbc_model.setNumberThreads(num_threads);
    LOG(INFO) << "Setting ILP_NUM_THREADS as " << num_threads << ".";
  }

  // reduce printout
  if (!ServiceEnv::debug()) cbc_model.setLogLevel(0);

  //model.solver()->setHintParam(OsiDoReducePrint, true, OsiHintTry);

  // Do complete search
  cbc_model.branchAndBound();

  bool succ = true;
  if (cbc_model.isProvenOptimal()) {
    VLOG(0) << "optimal solution found by ilp solver";
  } else {
    if (cbc_model.isProvenInfeasible()) {
      VLOG(0) << "infeasible ILP model";
      succ = false;
    } else {
      VLOG(0) << "optimal solution NOT found by ilp solver";
    }
  }

  CHECK(succ);

  const double *val = cbc_model.getColSolution();
  VLOG(0) << "Cbc solver finished";

  bool debug = ServiceEnv::debug();
  if (debug) printf("self var solution:");
  std::unordered_map<int/*cone id*/, int/*strategy id*/> cone_str_ids;
  CHECK(all_str_self_vars_.size() <= col_num);
  for (int i=0; i<all_str_self_vars_.size(); ++i) {
    int self_var_val = (int)(val[i]+0.5);
    if (debug) {
      if (i % 32 == 0) printf ("\n[%7x]: ", i);
      printf("%1d ", self_var_val);
    }

    std::shared_ptr<ILPSelfVarExpr>& self_var = all_str_self_vars_[i];
    if (self_var->cone_id_ < 0) {
      // ignore context cone
      continue;
    }
    if (self_var_val == 1) {
      VLOG(1) << "cone_str_ids size: " << cone_str_ids.size();
      VLOG(1) << "cone id: " << self_var->cone_id_ << ", str id: " << self_var->strtg_id_;
      CHECK(cone_str_ids.find(self_var->cone_id_) == cone_str_ids.end());
      cone_str_ids[self_var->cone_id_] = self_var->strtg_id_;
    }
  }
  if (debug) {
    printf("\n");
    fflush(stdout);
  }

  delete [] col_start;
  delete [] col_length;
  delete [] coeff_rows;
  delete [] coeffs;

  delete [] objective;
  delete [] row_lower;
  delete [] row_upper;
  delete [] col_lower;
  delete [] col_upper;

  return std::move(cone_str_ids);
}

std::string ILPModel::ExportToString() const {
  std::string ilp_mod_str("ilp:\n");
  // 1. dump ILP optimization object
  ilp_mod_str += opt_obj_->ModelStr() + "\n";

  // 2. dump ILP constraints
  for (auto& constraint : constraints_) {
    ilp_mod_str += constraint->ModelStr() + "\n";
  }

  return std::move(ilp_mod_str);
}

void HloSubGraph::BuildILPModel(
    std::list<std::shared_ptr<InstCone>>& inst_cones,
    const InstAffinityMap& affinity_map,
    const HloInstMap<SharedDimStrategy>& strategy_map,  // remove it later
    ILPModel& ilp_model) {
  // 1. build ILP variables
  // 1.1. build self variables
  std::unordered_map<int/*cone id*/,
                     std::vector<std::shared_ptr<ILPSelfVarExpr>>>&
  str_self_var_map = ilp_model.str_self_var_map_;

  std::map<std::pair<int/*cone id*/, int/*strategy id*/>,
           std::shared_ptr<ILPSelfVarExpr>>&
  str_id_self_var_map = ilp_model.str_id_self_var_map_;

  for (auto& cone : inst_cones) {
    CHECK(str_self_var_map.find(cone->id_) == str_self_var_map.end());
    for (int str_id = 0; str_id < cone->strategies().size(); ++str_id) {
      CHECK(cone->id_ >= 0);
      CHECK(cone->root_inst_);
      std::shared_ptr<ILPSelfVarExpr> self_var =
                              ilp_model.BuildSelfVar(cone->id_, str_id);

      // --------------------------------------------
      // debug purpose:
      self_var->cone_root_ = cone->root_inst_;
      // --------------------------------------------

      self_var->set_cost(cone->strategy_cost(str_id));
      str_self_var_map[cone->id_].emplace_back(self_var);


      auto key = std::make_pair(cone->id_, str_id);
      CHECK(str_id_self_var_map.find(key) == str_id_self_var_map.end());
      str_id_self_var_map[key] = self_var;
    }
  }

  // context is a special cone, its cone id is -1 and strategy id is 0
  std::shared_ptr<ILPSelfVarExpr> ctx_self_var = ilp_model.BuildSelfVar(-1, 0);
  str_self_var_map[-1].emplace_back(ctx_self_var);
  str_id_self_var_map[std::make_pair(-1, 0)] = ctx_self_var;


  // 1.2. build strategy matching variables(edge variables)
  const std::unordered_map<int/*one cone id*/, std::set<std::pair<int/*another cone id*/, bool/*is src*/>>>&
  adjacency_map = cone_strtg_mngr_->adjacency_map_;

  std::map<std::pair<int/*cone id*/, int/*strategy id*/>,
           std::vector<std::shared_ptr<ILPEdgeVarExpr>>>&
  str_edge_map = ilp_model.str_edge_map_;

  for (auto& one_src_cones : adjacency_map) {
    // ------------------------------------------------------------
    // debug purpose
    const HloInstruction* src_cone_root = nullptr;
    // ------------------------------------------------------------

    int src_cone_id = one_src_cones.first;
    int src_str_num;
    if (src_cone_id >= 0) {
      CHECK(src_cone_id < inst_cones_.size()) << "src_cone_id: " << src_cone_id
                                << ", inst_cones_.size: " << inst_cones_.size();
      std::shared_ptr<InstCone> src_cone = inst_cones_[src_cone_id];
      src_str_num = src_cone->strategies().size();
      // ------------------------------------------------------------
      // debug purpose
      src_cone_root = src_cone->root_inst_;
      // ------------------------------------------------------------
    } else {
      src_str_num = 1; // only 1 strategy id(the id is 0)
    }

    for (auto& peer_cone : one_src_cones.second) {
      if (peer_cone.second) {
        // key is target cone
        continue;
      }

      // key is source cone, build edge variable
      // ------------------------------------------------------------
      // debug purpose
      const HloInstruction* usr_cone_root = nullptr;
      // ------------------------------------------------------------
      int usr_cone_id = peer_cone.first;
      int usr_str_num;
      if (usr_cone_id >= 0) {
        CHECK(usr_cone_id < inst_cones_.size()) << "usr_cone_id: " << usr_cone_id
                                << ", inst_cones_.size: " << inst_cones_.size();
        std::shared_ptr<InstCone> usr_cone = inst_cones_[usr_cone_id];
        usr_str_num = usr_cone->strategies().size();
        // ------------------------------------------------------------
        // debug purpose
        usr_cone_root = usr_cone->root_inst_;
        // ------------------------------------------------------------
      } else {
        usr_str_num = 1; // only 1 strategy id(the id is 0)
      }

      // ------------------------------------------------------------
      // debug purpose
      if (usr_cone_id < 0) {
        VLOG(2) << "usr_cone_id: " << usr_cone_id
                << ", usr_cone root: " << usr_cone_root;
      }
      // ------------------------------------------------------------

      std::pair<int, int> cone_key = std::make_pair(src_cone_id, usr_cone_id);

      std::shared_ptr<ILPSelfVarExpr> src_self_var;
      std::shared_ptr<ILPSelfVarExpr> usr_self_var;
      for (int src_str_id=0; src_str_id<src_str_num; ++src_str_id) {
        for (int usr_str_id=0; usr_str_id<usr_str_num; ++usr_str_id) {
          src_self_var = ilp_model.FindStrSelfVar(src_cone_id, src_str_id);
          usr_self_var = ilp_model.FindStrSelfVar(usr_cone_id, usr_str_id);

          std::shared_ptr<ILPEdgeVarExpr> edge_var;
          edge_var = ilp_model.BuildEdgeVar(src_self_var, usr_self_var);

          // ------------------------------------------------------------
          // debug purpose
          edge_var->src_cone_root_ = src_cone_root;
          edge_var->usr_cone_root_ = usr_cone_root;
          // ------------------------------------------------------------

          int64 edge_cost = cone_strtg_mngr_->FindCost(src_str_num, src_cone_id,
                                                       src_str_id, usr_cone_id,
                                                       usr_str_id);

          // ------------------------------------------------------------
          // debug purpose
          VLOG(2) << "edge: " << edge_var->ModelStr() << std::endl
                  << "  cost: " << edge_cost;
          // ------------------------------------------------------------
          
          edge_var->set_cost(edge_cost);
        }
      }
    }
  }

  // 2. build ILP optimization object
  std::shared_ptr<ILPSumExpr> opt_obj = std::make_shared<ILPSumExpr>();

  // 2.1. intra-cone cost
  for (auto& one_cone_strs : str_self_var_map) {
    if (one_cone_strs.first<0) {
      continue;
    }
    for (auto& one_str_self : one_cone_strs.second) {
      std::shared_ptr<ILPScaledExpr> sc_var =
            std::make_shared<ILPScaledExpr>(one_str_self, one_str_self->cost_);
      opt_obj->operands_.emplace_back(sc_var);
    }
  }

  // 2.2. inter-cone cost
  for (auto& small_major_item : ilp_model.str_edge_map_no_dup_) {
    for (auto& edge_item : small_major_item.second) {
      std::shared_ptr<ILPScaledExpr> sc_var;
      sc_var = std::make_shared<ILPScaledExpr>(edge_item.second,
                                               edge_item.second->cost_);
      opt_obj->operands_.emplace_back(sc_var);
    }
  }

  ilp_model.opt_obj_ = opt_obj;

  // 3. build ILP constraints
  // 3.1. only one strategy is active for each cone:
  //        only one strategy variable is 1, others are all 0 for each cone
  for (auto& str_self_vars : str_self_var_map) {
    std::shared_ptr<ILPSumExpr> self_sum = std::make_shared<ILPSumExpr>();
    for (auto& var : str_self_vars.second) {
      self_sum->operands_.emplace_back(var);
    }
    std::shared_ptr<ILPEqualIntExpr> self_eq =
                            std::make_shared<ILPEqualIntExpr>(self_sum, 1);
    self_eq->id_ = ilp_model.constraints_.size();
    ilp_model.constraints_.emplace_back(self_eq);
  }

  // 3.2. self active strategy should match edge active
  for (auto& id_self_var : str_id_self_var_map) {
    std::shared_ptr<ILPSelfVarExpr> self_var = id_self_var.second;
    CHECK(str_edge_map.find(id_self_var.first) != str_edge_map.end())
          << "cone id: " << id_self_var.first.first << ", str id: "
          << id_self_var.first.second << ", self var: " << self_var->ModelStr()
          << ", edge map size: " << str_edge_map.size();

    std::unordered_map<int/*peer cone id*/, std::set<int>> grouped_edge;

    for (auto& edge_var : str_edge_map[id_self_var.first]) {
      std::shared_ptr<ILPSelfVarExpr> peer_self_var;
      if (edge_var->src_var_.get() == self_var.get()) {
        peer_self_var = edge_var->usr_var_;
      } else {
        peer_self_var = edge_var->src_var_;
      }
      grouped_edge[peer_self_var->cone_id_].insert(edge_var->id_);
    }

    for (auto& one_group : grouped_edge) {
      auto& edge_ids = one_group.second;
      std::shared_ptr<ILPSumExpr> edge_minus_self = std::make_shared<ILPSumExpr>();
      for (auto& edge_id : edge_ids) {
        CHECK(edge_id < ilp_model.all_str_edge_vars_.size());
        std::shared_ptr<ILPEdgeVarExpr>& edge_var = ilp_model.all_str_edge_vars_[edge_id];
        edge_minus_self->operands_.emplace_back(edge_var);
      }

      auto minus_self_var = std::make_shared<ILPScaledExpr>(self_var, -1);
      edge_minus_self->operands_.emplace_back(minus_self_var);
      std::shared_ptr<ILPEqualIntExpr> self_edge_eq =
                      std::make_shared<ILPEqualIntExpr>(edge_minus_self, 0);
      self_edge_eq->id_ = ilp_model.constraints_.size();
      ilp_model.constraints_.emplace_back(self_edge_eq);
    }
  }

  //  TODO(lansong):
  //  set affinity between tuple and gte to suppress the strategy mismatch between them
  // 3.3. strategy affinity constraint
  //         x     y
  //          \   /
  //          tuple(cone root)
  //            |
  //          -----
  //         |     |
  //        gte0  gte1
  //         |     |
  //
  //  zero cost constraint between tuple and gte

  // 3.4. rule based strategy affinity from affinity_map
  std::set<std::pair<int, int>> cone_affinity_pairs;
  auto& cone_map = cone_strtg_mngr_->cone_map_;
  VLOG(1) << "start affinity constraints.";
  for (auto inst_pair : affinity_map.AllAffinities()) {
    if (cone_map.count(inst_pair.second)) {
      if (!cone_map.count(inst_pair.first)) {
        CHECK(strategy_map.find(inst_pair.first) != strategy_map.end());
      }
    }
    if (!cone_map.count(inst_pair.first) ||
        !cone_map.count(inst_pair.second)) continue;
    int src_cone_id = cone_map.at(inst_pair.first)->id_;
    int usr_cone_id = cone_map.at(inst_pair.second)->id_;
    VLOG(2) << "affinity input: " << inst_pair.first->name() << " cone " << src_cone_id
            << ", output: " << inst_pair.second->name() << " cone " << usr_cone_id;
    CHECK(src_cone_id < inst_cones_.size() && usr_cone_id < inst_cones_.size());
    if (src_cone_id == usr_cone_id) continue;

    std::shared_ptr<ILPSumExpr> var_sum = std::make_shared<ILPSumExpr>();
    int num_i = 0;
    for (const std::unique_ptr<ConeStrategy>& cone_strategy : inst_cones_[src_cone_id]->strategies()) {
      SharedDimStrategy s_in = cone_strategy->FindStrategy(inst_pair.first);
      VLOG(1) << "input strategy " << num_i << " for " << inst_pair.first->name() << ": " << s_in->ToString();
      int num_o = 0;
      for (const std::unique_ptr<ConeStrategy>& cone_strategy : inst_cones_[usr_cone_id]->strategies()) {
        SharedDimStrategy s_out = cone_strategy->FindStrategy(inst_pair.second);
        int64 cost = Cost(*s_in, *s_out, inst_pair.second->shape(), 2, false, false);
        VLOG(2) << "  cost " << cost << " for out strategy " << num_o
                << " for " << inst_pair.second->name() << ": " << s_out->ToString();

        if (cost == 0) {
          auto src_self_var = ilp_model.FindStrSelfVar(src_cone_id, num_i);
          auto dst_self_var = ilp_model.FindStrSelfVar(usr_cone_id, num_o);
          auto neg_dst = std::make_shared<ILPScaledExpr>(dst_self_var, -1);
          auto var_eq = std::make_shared<ILPSumExpr>();
          var_eq->operands_.emplace_back(src_self_var);
          var_eq->operands_.emplace_back(neg_dst);
          std::shared_ptr<ILPEqualIntExpr> affinity_var_eq =
                      std::make_shared<ILPEqualIntExpr>(var_eq, 0);
          affinity_var_eq->id_ = ilp_model.constraints_.size();
          ilp_model.constraints_.emplace_back(affinity_var_eq);
          VLOG(2) << "adding affinity var equal constraint: " << affinity_var_eq->ModelStr();

          var_sum->operands_.emplace_back(src_self_var);
          var_sum->operands_.emplace_back(dst_self_var);
        }
        ++num_o;
      }
      ++num_i;
    }
    std::shared_ptr<ILPEqualIntExpr> affinity_sum =
                std::make_shared<ILPEqualIntExpr>(var_sum, 2);
    affinity_sum->id_ = ilp_model.constraints_.size();
    ilp_model.constraints_.emplace_back(affinity_sum);
    VLOG(2) << "adding affinity sum constraint: " << affinity_sum->ModelStr();
  }
  VLOG(0) << "finish ILP model building";
}

std::string CostSpmdStrategy::InstCostsToString(
    const HloInstMap<std::unique_ptr<InstCosts>>& inst_costs,
    int sub_graph_id) {
  std::string costs_str;
  costs_str += "all instruction costs in sub graph "
               + std::to_string(sub_graph_id) + ":\n";
  costs_str += "instruction num: " + std::to_string(inst_costs.size()) + "\n";
  int cone_num = 0;
  int64 total_cost = 0;
  for (auto& inst_cost : inst_costs) {
    costs_str += "inst: " + inst_cost.first->name() + "\n";
    CHECK(inst_cost.second);
    costs_str += inst_cost.second->ToString() + "\n";
    if (inst_cost.second->is_cone_root()) {
      ++cone_num;

      int64 min_cost = INT64_MAX;
      for (auto& one_strtg_cost : inst_cost.second->strtg_costs_) {
        // accumulate cost of cone root
        if (min_cost > one_strtg_cost->cost_) {
          min_cost = one_strtg_cost->cost_;
        }
      }

      total_cost += min_cost;
    }
  }

  costs_str += "cone num: " + std::to_string(cone_num) + "\n";
  costs_str += "internal minimal cost of all cones in sub graph "
               + std::to_string(sub_graph_id)
               + " is " + std::to_string(total_cost) + "\n";

  return std::move(costs_str);
}

HloInstMap<std::unique_ptr<InstCosts>>
CostSpmdStrategy::InferInstCosts(
    const std::unordered_set<const HloInstruction*>& cone_roots,
    const HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>& split_candidates,
    const HloInstSet& graph_scope,  // sub graph instructions
    const HloInstMap<SharedDimStrategy>& strategy_map) {  // whole graph strategy map
  HloInstMap<std::unique_ptr<InstCosts>> inst_costs_map;
  std::deque<const HloInstruction*> ready_insts;
  std::unordered_set<const HloInstruction*> visited;

  auto is_ready = [&graph_scope, &strategy_map, &inst_costs_map]
                                  (const HloInstruction* inst) -> bool {
    bool ready = true;
    for (auto* op : inst->operands()) {
      if (graph_scope.find(op) == graph_scope.end()) {
        continue;
      }
      if (op->opcode() == HloOpcode::kConstant ||
          op->opcode() == HloOpcode::kParameter) {
        // constant is not splitted because it breaks SPMD logic
        // its user don't care if it is splitted
        continue;
      }
      if (strategy_map.find(op) == strategy_map.end() &&
          inst_costs_map.find(op) == inst_costs_map.end()) {
        ready = false;
        break;
      }
    }

    return ready;
  };

  VLOG(1) << "split_candidates size: " << split_candidates.size();
  // 1. initialize ready_insts
  // 1.1. push constant and args first
  for (auto& split_cand : split_candidates) {
    auto* inst = split_cand.first;
    if (!(inst->opcode() == HloOpcode::kConstant ||
          inst->opcode() == HloOpcode::kParameter)) continue;
    VLOG(2) << "split inst: " << inst->name() << ", strategy num: "
            << split_cand.second.size();
    if (visited.find(inst) == visited.end()) {
      ready_insts.push_back(inst);
      visited.insert(inst);
    }
  }
  // 1.2. push other insts.
  for (auto& split_cand : split_candidates) {
    auto* inst = split_cand.first;
    if (inst->opcode() == HloOpcode::kConstant ||
        inst->opcode() == HloOpcode::kParameter) continue;
    VLOG(2) << "split inst: " << inst->name() << ", strategy num: "
            << split_cand.second.size();
    if (is_ready(inst) && visited.find(inst) == visited.end()) {
      ready_insts.push_back(inst);
      visited.insert(inst);
    }
  }

  for (auto* ready_inst : ready_insts) {
    VLOG(2) << "initial ready inst: " << ready_inst->name() << ", strategy num: "
            << split_candidates.at(ready_inst).size();
  }

  // 2. infer by dynamic programming algorithm
  while (!ready_insts.empty()) {
    auto* inst = ready_insts.front();
    ready_insts.pop_front();

    bool is_cone_root = false;
    if (cone_roots.find(inst) != cone_roots.end()) {
      is_cone_root = true;
    }

    std::unique_ptr<InstCosts> inst_costs =
                              std::make_unique<InstCosts>(inst, is_cone_root);
    CHECK(split_candidates.find(inst) != split_candidates.end());
    const std::set<SharedDimStrategy, DimStrategyPtrLess>& inst_split_cands = split_candidates.at(inst);
    int64 inst_bytes = ShapeUtil::ByteSizeOf(inst->shape(), 8);

    for (auto& strtg : inst_split_cands) {
      // build cost for one of splitting candidates
      std::unique_ptr<InstCosts::OneStrategyCost> one_strtg_cost =
                            std::make_unique<InstCosts::OneStrategyCost>(strtg);
      if (strtg->IsPartial()) {
        one_strtg_cost->cost_ += PerfUtils::AllReduceCost(inst_bytes, cur_split_num_);  // ring allreduce cost
      }

      for (int i = 0; i < inst->operand_count(); ++i) {
        // one op cost:
        //    calculate the lowest cost for current operand
        const HloInstruction* op = inst->operand(i);
        // Temporarily hack for get-tuple-elements shape.
        Shape op_shape = op->shape();
        bool tuple_src = false;
        if (op->shape().IsTuple()) {
          tuple_src = true;
        }

        if (inst->opcode() == HloOpcode::kGetTupleElement) {
          int64 tuple_index = DynCast<HloGetTupleElementInstruction>(inst)->tuple_index();
          op_shape = op_shape.tuple_shapes(tuple_index);
        }
        int64 op_bytes = ShapeUtil::ByteSizeOf(op_shape, 8);
 
        // reverse infer each input strategy
        SharedDimStrategy expected_op_strtg =
                                    StrategyUtil::BackInfer(inst, *strtg, i);
        if (inst_costs_map.find(op) != inst_costs_map.end()) {
          // select the lowest cost for current op
          std::unique_ptr<InstCosts>& op_costs = inst_costs_map.at(op);

          if (op_costs->is_cone_root()) {
            one_strtg_cost->op_strtg_idx_.emplace_back(-1);
            one_strtg_cost->cost_ += 0; // NOT update cost
            continue;
          }

          int64 min_cost = INT64_MAX;
          int min_idx = 0;
          // select the op strategy with minimal cost
          std::vector<float> record_costs;
          for (int strtg_idx=0; strtg_idx<op_costs->strtg_costs_.size(); ++strtg_idx) {
            const std::unique_ptr<InstCosts::OneStrategyCost>& op_cost =
                                            op_costs->strtg_costs_[strtg_idx];
            int64 accu_cost = op_cost->cost_;
            VLOG(2) << "op strategy id: " << std::to_string(strtg_idx) << ", strategy: "
                    << op_cost->strategy_->ToString() << ", expected strategy: "
                    << expected_op_strtg->ToString() << ", cost is " << std::to_string(op_bytes);

            int64 data_cost = Cost(*op_cost->strategy_, *expected_op_strtg,
                                   op_shape, cur_split_num_, false, tuple_src);  // TODO(lansong): modify cost calculation for shared device
            accu_cost += data_cost;

            record_costs.push_back(accu_cost);
            if (accu_cost < min_cost) {
              min_cost = accu_cost;
              min_idx = strtg_idx;
            }
          }

          int count = 0;
          for (int i = 0; i < record_costs.size(); ++i) {
            if (min_cost == record_costs[i]) ++count;
          }

          if (count > 1) {
            // multiple strategies op should be regarded as cone root
            op_costs->set_is_cone_root(true);
            one_strtg_cost->op_strtg_idx_.emplace_back(-1);
            one_strtg_cost->cost_ += 0; // NOT update cost
            continue;
          }
          one_strtg_cost->op_strtg_idx_.emplace_back(min_idx);
          one_strtg_cost->cost_ += min_cost;

          continue;
        }

        one_strtg_cost->op_strtg_idx_.emplace_back(-1);
        if (op->opcode() == HloOpcode::kConstant) {
          CHECK(strategy_map.find(op) == strategy_map.end());
          // constant is not splitted because splitting will break SPMD logic
          // no cost
        } else {
          // 1. op may be in another sub graph
          //    1.1. op strategy is determined by another sub graph
          //    1.2. op strategy is not determined yet because op's sub graph is not inferred yet
          // 2. op strategy may be determined by user annotation
          if (strategy_map.find(op) != strategy_map.end()) {
            //    1.1. op strategy is determined
            VLOG(2) << "op " << op->name() << ", strategy: "
                    << strategy_map.at(op)->ToString() << ", expected strategy: "
                    << expected_op_strtg->ToString() << ", cost is "
                    << std::to_string(op_bytes);

            int64 data_cost = Cost(*strategy_map.at(op), *expected_op_strtg,
                                   op_shape, cur_split_num_, false, tuple_src);  // TODO(lansong): modify cost calculation for shared device
            one_strtg_cost->cost_ += data_cost;
          } else {
            //    1.2. op strategy is not determined yet
            // don't update cost
          }
        }
      }

      if (!inst->shape().IsTuple()) {
        // if current instruction is read by an instruction outside of current context,
        // accumulate output cost.
        for (auto * user : inst->users()) {
          if (user->shape().IsTuple()) {
            continue;
          }
          // 1. user may be in another sub graph
          //    1.2. user strategy is not determined yet because user's sub graph is not inferred yet
          if (strategy_map.find(user) != strategy_map.end()) {
            //  user strategy is determined by another sub graph
            //  user strategy may be determined by user annotation

            int idx = 0;
            for (auto* brother : user->operands()) {
              if (brother == inst) {
                break;
              }

              ++idx;
            }
            CHECK(idx < user->operand_count());

            // reverse infer expected strategy for this instruction from user instruction
            SharedDimStrategy expected_strtg =
                   StrategyUtil::BackInfer(user, *(strategy_map.at(user)), idx);
            int64 data_cost = Cost(*strtg, *expected_strtg,
                                   inst->shape(), cur_split_num_, false, false);
            if (data_cost > 0) {
              VLOG(0) << inst->name() << " has " << data_cost << " output cost";
            }
            one_strtg_cost->cost_ += data_cost;
          } else {
            //  user strategy is not determined yet
            // don't update cost
          }
        }
      }

      CHECK(inst->operand_count() == one_strtg_cost->op_strtg_idx_.size());

      inst_costs->strtg_costs_.emplace_back(std::move(one_strtg_cost));
    }

    CHECK(inst_costs_map.find(inst) == inst_costs_map.end());
    inst_costs_map[inst] = std::move(inst_costs);

    for (auto* user : inst->users()) {
      if (split_candidates.find(user) == split_candidates.end()) {
        continue;
      }
      if (is_ready(user) && visited.find(user) == visited.end()) {
        VLOG(2) << "Ready: " << user->name();
        ready_insts.emplace_back(user);
        visited.insert(user);
      }
    }
  }

  return std::move(inst_costs_map);
}

HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>
CostSpmdStrategy::CollectInstSplits(
      const GraphPieces& pieces,
      const HloInstMap<SharedDimStrategy>& strategy_map,
      const HloInstSet& graph_scope) {
  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>> split_candidates;
  for (auto& piece : pieces.pieces_) {
    VLOG(1) << "inst num in piece: " << piece->inst_map_.size();
    for (auto& inst_strtg : piece->inst_map_) {
      split_candidates[inst_strtg.first].insert(inst_strtg.second);
    }
  }

  // Add Glue to all instructions except for computation intensive instruction
  for (auto* inst : graph_scope) {
    if (inst->opcode() == HloOpcode::kDot ||
        inst->opcode() == HloOpcode::kConvolution) continue;
    split_candidates[inst].insert(std::make_shared<DimStrategy>());
  }

  return std::move(split_candidates);
}

void CostSpmdStrategy::Infer(
              const HloInstSet& user_annotated_set,
              const HloInstMap<SharedDimStrategy>& init_strategy_map,
              HloSubGraph& sub_graph) {
  const HloInstSet sub_graph_insts(sub_graph.insts_.begin(),
                                                        sub_graph.insts_.end());
  HloInstMap<SharedDimStrategy> starting_strategies;
  bool changed = false;
  GraphStrategy graph_strategy(cur_split_num_, cur_share_dev_);
  for (auto& inst_strtg : init_strategy_map) {
    graph_strategy.SetInstStrategy(inst_strtg.first, inst_strtg.second);
  }
  auto& strategy_map = graph_strategy.strategy_map();
  VLOG(0) << "init strategy_map size: " << strategy_map.size();
  do {
    // 1. find starting instructions (forward)
    starting_strategies = FindStartingStrategies(strategy_map, sub_graph_insts);

    // 2. sync free inference
    changed = InferByInstCount(starting_strategies,
                               user_annotated_set,
                               sub_graph_insts,
                               graph_strategy);

    if (changed) {
      // 3. find starting instructions (reverse)
      starting_strategies = FindBackStartingStrategies(strategy_map,
                                                       sub_graph_insts);
      // 4. sync free inference
      changed = ReverseInferByInstCount(starting_strategies,
                                        user_annotated_set,
                                        sub_graph_insts,
                                        graph_strategy);
    }
  } while (changed);

  VLOG(0) << "strategy_map size: " << strategy_map.size();
  // 5. find unsplitted gemm
  std::list<const HloInstruction*> unsplitted_insts =
                  CollectUnsplittedComputeInsts(sub_graph_insts, strategy_map);

  while (!unsplitted_insts.empty()) {
    VLOG(1) << "unsplitted compute insts count: " << unsplitted_insts.size();
    // 6. select max splitting starting from a unsplitted computing instruction
    std::list<const HloInstruction*>::iterator max_split_it = unsplitted_insts.end();
    GraphStrategy max_split_graph_strategy(cur_split_num_, cur_share_dev_);
    int max_splitted_count = 0;
    std::list<const HloInstruction*>::iterator it = unsplitted_insts.begin();
    HloInstSet infered_insts;

    for (; it != unsplitted_insts.end(); ++it) {
      VLOG(2) << "current unsplitted inst: " << (*it)->ToString();
      std::vector<SharedDimStrategy> strategies = GenSplitProposals(*it);
      VLOG(2) << "strategies vector size: " << strategies.size();
      int tmp_1 = 0;
      for (auto& strtg : strategies) {
        VLOG(2) << "current strtg id: " << ++tmp_1;
        GraphStrategy try_graph_strategy = graph_strategy;
        try_graph_strategy.SetInstStrategy(*it, strtg);

        infered_insts = PopulateStrategy(*it, strtg, user_annotated_set,
                                         sub_graph_insts, try_graph_strategy);
        VLOG(2) << "after calling PopulateStrategy";
        VLOG(2) << "infered_insts size: " << infered_insts.size();
        if (max_splitted_count < infered_insts.size()+1) {
          VLOG(2) << "update max split strategy";
          max_split_it = it;
          max_split_graph_strategy = try_graph_strategy;
          max_splitted_count = infered_insts.size()+1;
        }
      }
      VLOG(2) << "current strtg id: " << tmp_1;
      VLOG(2) << "after traverse strategies vector";
    }

    // update strategy map
    graph_strategy = max_split_graph_strategy;
    VLOG(1) << "before erase from unsplitted_insts, unsplitted_insts size: " << unsplitted_insts.size();

    CHECK(max_split_it != unsplitted_insts.end());
    unsplitted_insts.erase(max_split_it);
    VLOG(1) << "after erase from unsplitted_insts, unsplitted_insts size: " << unsplitted_insts.size();
  }

  VLOG(1) << "mid of Infer";
  std::shared_ptr<SubGraphStrategy> sub_graph_strategy =
                                    std::make_shared<SubGraphStrategy>();
  VLOG(0) << "strategy_map size: " << strategy_map.size();
  auto& res_strategy_map = graph_strategy.strategy_map();
  VLOG(0) << "res_strategy_map size: " << res_strategy_map.size();
  CHECK(res_strategy_map.find(sub_graph.head_) != res_strategy_map.end() ||
        res_strategy_map.find(sub_graph.tail_) != res_strategy_map.end());
  if (res_strategy_map.find(sub_graph.head_) != res_strategy_map.end()) {
    sub_graph_strategy->head_strategy_ = res_strategy_map.at(sub_graph.head_);
  }

  if (res_strategy_map.find(sub_graph.tail_) != res_strategy_map.end()) {
    sub_graph_strategy->tail_strategy_ = res_strategy_map.at(sub_graph.tail_);
  }
  sub_graph_strategy->strategy_map_ = std::move(res_strategy_map);
  sub_graph_strategy->id_ = sub_graph.graph_strategies_.size();
  sub_graph_strategy->sub_graph_id_ = sub_graph.id_;

  VLOG(0) << "sub graph: " << sub_graph_strategy->sub_graph_id_
          << ", sub graph str id: " << sub_graph_strategy->id_
          << ", strategy map size: " << sub_graph_strategy->strategy_map_.size();

  sub_graph.graph_strategies_.push_back(sub_graph_strategy);
  VLOG(0) << "end of Infer";
}

void CostSpmdStrategy::PlanSubGraph(
    const HloInstMap<SharedDimStrategy>& fixed_strategy_map, // strategies infered by user annotations
    const HloInstSet& user_annotated_set,
    const MemSavePlan& mem_save_plan,
    const HloSubGraph* pre_sub_graph,
    const InstAffinityMap& affinity_map,
    int opt_level,
    HloSubGraph& sub_graph) {
  VLOG(0) << "[PlanSubGraph] plan sub graph " << sub_graph.id_
          << ", inst size: " << sub_graph.insts_.size();

  std::vector<SharedDimStrategy> head_proposals;
  std::vector<SharedDimStrategy> tail_proposals;
  const HloInstruction* head = sub_graph.head_;
  const HloInstruction* tail = sub_graph.tail_;
  if (head) {
    CHECK(is_compute_intensive(head));
    if (fixed_strategy_map.find(head) == fixed_strategy_map.end()) {
      VLOG(0) << "before GenSplitProposals for head";
      head_proposals = GenSplitProposals(head);
      VLOG(0) << "after GenSplitProposals for head";
    } else {
      VLOG(0) << "adding annotated proposals for head";
      head_proposals.push_back(fixed_strategy_map.at(head));
    }
  }

  if (tail) {
    CHECK(is_compute_intensive(tail));
    if (fixed_strategy_map.find(tail) == fixed_strategy_map.end()) {
      VLOG(0) << "before GenSplitProposals for tail";
      tail_proposals = GenSplitProposals(tail);
      VLOG(0) << "after GenSplitProposals for tail";
    } else {
      VLOG(0) << "adding annotated proposals for tail";
      tail_proposals.push_back(fixed_strategy_map.at(tail));
    }
  }

  if (!head_proposals.empty() && !tail_proposals.empty()) {
    for (auto& head_prop : head_proposals) {
      for (auto& tail_prop : tail_proposals) {
        HloInstMap<SharedDimStrategy> expected_strategies;
        CHECK(head);
        CHECK(tail);
        expected_strategies[head] = head_prop;
        expected_strategies[tail] = tail_prop;

        VLOG(2) << std::endl;
        VLOG(2) << std::endl << "#############################################################################"
                << std::endl << "#############################################################################" << std::endl;
        VLOG(2) << std::endl;
        VLOG(2) << "[PlanSubGraph] plan sub graph " << sub_graph.id_
                << std::endl << "head: " << head->name() << " with strategy: " << head_prop->ToString()
                << std::endl << "tail: " << tail->name() << " with strategy: " << tail_prop->ToString();
        // get accumulated strategy map from pre_sub_graph
        const HloInstMap<SharedDimStrategy>* acc_strtg_map = nullptr;

        if (pre_sub_graph) {
          acc_strtg_map = pre_sub_graph->FindAccStrtgMap(head_prop);
        }

        if (acc_strtg_map) {
          VLOG(1) << "acc strategy map size: " << acc_strtg_map->size()
                  << ", head name: " << head->name()
                  << ", head strategy: " << head_prop->ToString();
        } else {
          VLOG(1) << "acc strategy map size: 0"
                  << ", head name: " << head->name()
                  << ", head strategy: " << head_prop->ToString();
        }

        if (opt_level<1) {
          FastInferSubGraph(user_annotated_set, fixed_strategy_map, expected_strategies,
                            mem_save_plan, acc_strtg_map, affinity_map, sub_graph, opt_level);
        } else {
          InferSubGraph(user_annotated_set, fixed_strategy_map, expected_strategies,
                        mem_save_plan, acc_strtg_map, affinity_map, sub_graph, opt_level);
        }
      }
    }
  } else if (!head_proposals.empty()) {
    for (auto& head_prop : head_proposals) {
      HloInstMap<SharedDimStrategy> expected_strategies;
      expected_strategies[head] = head_prop;

      VLOG(2) << std::endl;
      VLOG(2) << std::endl << "#############################################################################"
              << std::endl << "#############################################################################" << std::endl;
      VLOG(2) << std::endl;
      VLOG(2) << "[PlanSubGraph] plan sub graph " << sub_graph.id_
              << std::endl << "head: " << head->name() << " head strategy: " << head_prop->ToString();
      // get accumulated strategy map from pre_sub_graph
      const HloInstMap<SharedDimStrategy>* acc_strtg_map = nullptr;
      
      if (pre_sub_graph) {
        acc_strtg_map = pre_sub_graph->FindAccStrtgMap(head_prop);
      }

      if (opt_level<1) {
        FastInferSubGraph(user_annotated_set, fixed_strategy_map,
                          expected_strategies, mem_save_plan,
                          acc_strtg_map, affinity_map, sub_graph, opt_level);
      } else {
        InferSubGraph(user_annotated_set, fixed_strategy_map,
                      expected_strategies, mem_save_plan,
                      acc_strtg_map, affinity_map, sub_graph, opt_level);
      }
    }
  } else if (!tail_proposals.empty()) {
    for (auto& tail_prop : tail_proposals) {
      HloInstMap<SharedDimStrategy> expected_strategies;
      expected_strategies[tail] = tail_prop;

      VLOG(0) << std::endl;
      VLOG(0) << std::endl << "#############################################################################"
              << std::endl << "#############################################################################" << std::endl;
      VLOG(0) << std::endl;
      VLOG(0) << "[PlanSubGraph] plan sub graph " << sub_graph.id_
              << std::endl << "tail: " << tail->name() << " tail strategy: " << tail_prop->ToString();
      if (opt_level<1) {
        FastInferSubGraph(user_annotated_set, fixed_strategy_map,
                          expected_strategies, mem_save_plan,
                          nullptr, affinity_map, sub_graph, opt_level);
      } else {
        InferSubGraph(user_annotated_set, fixed_strategy_map,
                      expected_strategies, mem_save_plan,
                      nullptr, affinity_map, sub_graph, opt_level);
      }
    }
  } else {
    HloInstMap<SharedDimStrategy> expected_strategies;
    if (opt_level<1) {
      FastInferSubGraph(user_annotated_set, fixed_strategy_map,
                        expected_strategies, mem_save_plan,
                        nullptr, affinity_map, sub_graph, opt_level);
    } else {
      InferSubGraph(user_annotated_set, fixed_strategy_map,
                    expected_strategies, mem_save_plan,
                    nullptr, affinity_map, sub_graph, opt_level);
    }
  }
}

void CostSpmdStrategy::PlanSubGraphs(
        HloModule* module,
        const HloInstMap<SharedDimStrategy>& fixed_strategy_map, // strategies infered by user annotations
        const MemSavePlan& mem_save_plan,
        int opt_level,
        std::vector<HloSubGraph>& sub_graphs) {
  VLOG(1) << "start PlanSubGraphs";
  if (sub_graphs.empty()) {
    return;
  }

  VLOG(1) << "before ExtractUserSplit";
  HloInstMap<SharedDimStrategy> user_annotated_tensors =
                                                  ExtractUserSplit(module);

  HloInstSet user_annotated_set;
  for (auto& annotated_tensor : user_annotated_tensors) {
    user_annotated_set.insert(annotated_tensor.first);
  }

  VLOG(0) << "num of sub graphs: " << sub_graphs.size();

  for (int i=0; i<sub_graphs.size(); ++i) {
    VLOG(1) << "start of sub graph " << sub_graphs[i].id_;
    VLOG(1) << sub_graphs[i].ToString();
    VLOG(1) << "end of sub graph " << i << std::endl;
  }

  InstAffinityMapBuilder builder;
  builder.AddRule<InOutAffinity>();
  bool aux_affinity = false;
  TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
                                  "AUX_AFFINITY", /*default_val=*/false,
                                  &aux_affinity));
  if (aux_affinity) builder.AddRule<VarAuxAffinity>();

  std::unique_ptr<InstAffinityMap> affinity_map = builder.Build(module);

  // combine sub graph strategies with dynamic programming algorithm:
  auto dp_combine_graph = [](HloSubGraph& pre_sub, HloSubGraph& cur_sub) {
    auto& pre_finalized_tail_map = pre_sub.finalized_tail_strtg_map_;
    if (cur_sub.tail_ == nullptr) {
      // multiple tails
      CHECK(pre_sub.tail_ != nullptr) << "pre sub: " << pre_sub.id_;

      std::shared_ptr<SubGraphStrategy> min_cost_pre_str;
      std::shared_ptr<SubGraphStrategy> min_cost_cur_str;
      int64 min_accu_cost = INT64_MAX;
      for (auto& cur_str : cur_sub.graph_strategies_) {
        // current sub:
        //    a specific strategy
        const SharedDimStrategy& cur_head_strategy = cur_str->head_strategy_;

        // try to match current sub's head with previous sub's tail
        CHECK(pre_finalized_tail_map.find(cur_head_strategy) !=
              pre_finalized_tail_map.end());

        std::shared_ptr<SubGraphStrategy>& pre_str =
                                  pre_finalized_tail_map.at(cur_head_strategy);

        int64 accu_cost = pre_str->accu_cost_ + cur_str->self_cost_;

        if (accu_cost < min_accu_cost) {
          VLOG(2) << "[DP] min cost: " << min_accu_cost << " --> " << accu_cost
                  << ", pre_str: " << pre_str->head_strategy_->ToString()
                  << " // " << pre_str->tail_strategy_->ToString()
                  << ", cur_str: " << cur_str->head_strategy_->ToString()
                  << " // " << cur_str->tail_strategy_->ToString();
          min_accu_cost = accu_cost;
          min_cost_pre_str = pre_str;
          min_cost_cur_str = cur_str;
        }
      }

      CHECK(min_cost_cur_str);
      min_cost_cur_str->accu_cost_ = min_accu_cost;
      min_cost_cur_str->pre_sub_strategy_ = min_cost_pre_str;

      //    update acc_strategy_map_
      min_cost_cur_str->acc_strategy_map_ = min_cost_pre_str->acc_strategy_map_;
      min_cost_cur_str->acc_strategy_map_.insert(
                                      min_cost_cur_str->strategy_map_.begin(),
                                      min_cost_cur_str->strategy_map_.end());
      VLOG(1) << "sub graph id: " << cur_sub.id_ << ", acc_strategy_map_ size: "
              << min_cost_cur_str->acc_strategy_map_.size();
      cur_sub.unique_min_cost_strtg_ = min_cost_cur_str;

      return;
    }

    for (auto& cur_tail_strs : cur_sub.tail_strtg_map_) {
      // current sub:
      //    a specific tail strategy
      if (cur_tail_strs.second.empty()) {
        continue;
      }

      std::shared_ptr<SubGraphStrategy> min_cost_pre_str;
      std::shared_ptr<SubGraphStrategy> min_cost_cur_str;
      int64 min_accu_cost = INT64_MAX;
      // traverse different heads under the same tail
      for (std::shared_ptr<SubGraphStrategy>& cur_str : cur_tail_strs.second) {
        // calculate cost
        // current sub:
        //    a head strategy under a specific tail strategy
        const SharedDimStrategy& cur_head_strategy = cur_str->head_strategy_;

        // try to match current sub's head with previous sub's tail
        CHECK(pre_finalized_tail_map.find(cur_head_strategy) !=
              pre_finalized_tail_map.end());

        std::shared_ptr<SubGraphStrategy>& pre_str =
                                  pre_finalized_tail_map.at(cur_head_strategy);

        int64 accu_cost = pre_str->accu_cost_ + cur_str->self_cost_;

        if (accu_cost < min_accu_cost) {
          min_accu_cost = accu_cost;
          min_cost_pre_str = pre_str;
          min_cost_cur_str = cur_str;
        }
      }

      CHECK(min_cost_cur_str);
      min_cost_cur_str->accu_cost_ = min_accu_cost;
      min_cost_cur_str->pre_sub_strategy_ = min_cost_pre_str;

      //    update acc_strategy_map_
      min_cost_cur_str->acc_strategy_map_ = min_cost_pre_str->acc_strategy_map_;
      min_cost_cur_str->acc_strategy_map_.insert(
                                      min_cost_cur_str->strategy_map_.begin(),
                                      min_cost_cur_str->strategy_map_.end());
      VLOG(1) << "sub graph id: " << cur_sub.id_ << ", acc_strategy_map_ size: "
              << min_cost_cur_str->acc_strategy_map_.size();
      cur_sub.finalized_tail_strtg_map_[cur_tail_strs.first] = min_cost_cur_str;
    }

    cur_sub.tail_strtg_map_.clear();
  };

  auto dp_init_first_sub_graph = [](HloSubGraph& first_sub) {
    if (first_sub.tail_ == nullptr) {
      std::shared_ptr<SubGraphStrategy> min_cost_str;
      int64 min_cost = INT64_MAX;
      for (auto& sub_strtg : first_sub.graph_strategies_) {
        // for a specific strategy

        if (sub_strtg->self_cost_ < min_cost) {
          min_cost = sub_strtg->self_cost_;
          min_cost_str = sub_strtg;
        }
      }

      CHECK(min_cost_str);
      min_cost_str->accu_cost_ = min_cost;
      //    update acc_strategy_map_
      min_cost_str->acc_strategy_map_.insert(
                                      min_cost_str->strategy_map_.begin(),
                                      min_cost_str->strategy_map_.end());
      VLOG(1) << "sub graph id: " << first_sub.id_ << ", acc_strategy_map_ size: "
              << min_cost_str->acc_strategy_map_.size();
      first_sub.unique_min_cost_strtg_ = min_cost_str;

      return;
    }

    for (auto& tail_strs : first_sub.tail_strtg_map_) {
      // for a specific tail strategy
      if (tail_strs.second.empty()) {
        continue;
      }

      std::shared_ptr<SubGraphStrategy> min_cost_str;
      int64 min_cost = INT64_MAX;
      for (std::shared_ptr<SubGraphStrategy>& sub_strtg : tail_strs.second) {
        if (sub_strtg->self_cost_ < min_cost) {
          min_cost = sub_strtg->self_cost_;
          min_cost_str = sub_strtg;
        }
      }

      CHECK(min_cost_str);
      min_cost_str->accu_cost_ = min_cost;

      //    update acc_strategy_map_
      min_cost_str->acc_strategy_map_.insert(
                                      min_cost_str->strategy_map_.begin(),
                                      min_cost_str->strategy_map_.end());
      VLOG(1) << "sub graph id: " << first_sub.id_ << ", acc_strategy_map_ size: "
              << min_cost_str->acc_strategy_map_.size();
      first_sub.finalized_tail_strtg_map_[tail_strs.first] = min_cost_str;
    }

    first_sub.tail_strtg_map_.clear();
  };

  VLOG(1) << "plan sub graph 0";
  PlanSubGraph(fixed_strategy_map, user_annotated_set, mem_save_plan,
               nullptr, *affinity_map, opt_level, sub_graphs.front());
  dp_init_first_sub_graph(sub_graphs.front());

  for (int i=1; i<sub_graphs.size(); ++i) {
    VLOG(1) << "plan sub graph " << sub_graphs[i].id_ << " stored in sub graph";
    PlanSubGraph(fixed_strategy_map, user_annotated_set, mem_save_plan,
                 &sub_graphs[i-1], *affinity_map, opt_level, sub_graphs[i]);
    dp_combine_graph(sub_graphs[i-1], sub_graphs[i]);
  }

  VLOG(0) << "finish planning all sub graphs";
}

void
CostSpmdStrategy::FinalizeCurStrMap(std::vector<HloSubGraph>& sub_graphs,
                            GraphStrategy& graph_strategy) {
  HloSubGraph& last_sub_graph = sub_graphs.back();

  std::shared_ptr<SubGraphStrategy> min_cost_str;
  if (last_sub_graph.finalized_tail_strtg_map_.empty()) {
    min_cost_str = last_sub_graph.unique_min_cost_strtg_;
  } else {
    int64 min_cost = INT64_MAX;
    for (auto& sub_strtg : last_sub_graph.finalized_tail_strtg_map_) {
      if (sub_strtg.second->accu_cost_ < min_cost) {
        min_cost_str = sub_strtg.second;
        min_cost = sub_strtg.second->accu_cost_;
      }
    }
  }

  CHECK(min_cost_str);

  std::shared_ptr<SubGraphStrategy> local_str = min_cost_str;
  while (local_str) {
    VLOG(1) << "select sub graph: " << local_str->sub_graph_id_
            << ", sub graph str id: " << local_str->id_
            << ", strategy map size: " << local_str->strategy_map_.size();
    for (auto& inst_strtg : local_str->strategy_map_) {
      graph_strategy.SetInstStrategy(inst_strtg.first, inst_strtg.second);
    }
    local_str = local_str->pre_sub_strategy_;
  }
}

void CostSpmdStrategy::ResetUnDivisibleStrategy(
    HloInstMap<HLOStrategy>* best_strategy_map,
    int& glued_inst_num) {
  for (auto& inst_strategy : *best_strategy_map) {
    auto inst = inst_strategy.first;
    auto& shape = inst->shape();
    for (int i = 0; i < inst_strategy.second.dim_strategies().size(); ++i) {
      if (shape.IsTuple()) continue;
      auto* dim_str = inst_strategy.second.mutable_dim_strategy(i);
      if (dim_str->Glue() || dim_str->IsPartial() || dim_str->replicated() ||
          !(shape.dimensions(dim_str->partition_dim()) % dim_str->num_replicas())) continue;
      *dim_str = DimStrategy();
      glued_inst_num ++;
    }
  }
}

void CostSpmdStrategy::AlignInputOutputStrategy(
    HloModule* module,
    HloInstMap<HLOStrategy>* best_strategy_map,
    int& glued_inst_num) {
  const int var_count = module->variable_map()->size();
  const HloComputation* entry = module->entry_computation();
  int input_var_offset = entry->num_parameters() - var_count;
  int output_var_offset = entry->root_instruction()->operand_count() - var_count;
  const HloInstruction* root = entry->root_instruction();
  for (int i = 0; i < var_count; ++i) {
    int p_idx = input_var_offset + i;
    int out_idx = output_var_offset + i;
    // Alias relationship
    const HloInstruction* param = entry->parameter_instruction(p_idx);
    const HloInstruction* updated = root->operand(out_idx);
    CHECK(best_strategy_map->find(param) != best_strategy_map->end());
    CHECK(best_strategy_map->find(updated) != best_strategy_map->end());
    (*best_strategy_map)[updated] = (*best_strategy_map)[param];
  }
}

void CostSpmdStrategy::BestStrategyPostProcess(
    HloModule* module,
    HloInstMap<HLOStrategy>* best_strategy_map,
    int& glued_inst_num) {
  // 1. Reset all DimStrategy with default for undividend instructions
  ResetUnDivisibleStrategy(best_strategy_map, glued_inst_num);

  // 2. Set all root operands DimStrategy same with its parameter
  AlignInputOutputStrategy(module, best_strategy_map, glued_inst_num);

  // 3. Reset all special instructions with default which are not guaranteed
  // by bi-verification inference.
  for (auto& inst_strategy : *best_strategy_map) {
    auto inst = inst_strategy.first;
    for (int i = 0; i < inst_strategy.second.dim_strategies().size(); ++i) {
      auto* dim_str = inst_strategy.second.mutable_dim_strategy(i);
      switch (inst->opcode()) {
        case HloOpcode::kSlice: {
          auto* op = inst->operand(0);
          int64 input_dim = dim_str->partition_dim();
          const HloSliceInstruction* slice = DynCast<HloSliceInstruction>(inst);
          const std::vector<int64> slice_starts = slice->slice_starts();
          const std::vector<int64> slice_limits = slice->slice_limits();
          const std::vector<int64> slice_strides = slice->slice_strides();
          if (!dim_str->Glue() &&
              slice_limits[input_dim] - slice_starts[input_dim] != op->shape().dimensions(input_dim)) {
            *dim_str = DimStrategy();
            glued_inst_num ++;
          }
          break;
        }

        case HloOpcode::kReverse: {
          auto& reverse_dims = inst->dimensions();
          auto* dim_str = inst_strategy.second.mutable_dim_strategy(i);
          int64 input_dim = dim_str->partition_dim();
          std::unordered_set<int> reverse_dims_set(reverse_dims.begin(), reverse_dims.end());
          if (reverse_dims_set.count(input_dim)) {
            *dim_str = DimStrategy();
            glued_inst_num ++;
          }
          break;
        }

        case HloOpcode::kReduceWindow: {
          const Window& window = inst->window();
          if (!dim_str->Glue()) {
            auto dim_to_slice = dim_str->partition_dim();
            // Get window-reduction dimension index.
            auto w_dim = window.dimensions(dim_to_slice);
            auto w_dim_size = w_dim.size();
            if (w_dim_size > 1) {
              // Split at window-reduction dimension!
              VLOG(2) << "w_dim_size=" << w_dim_size;
              *dim_str = DimStrategy();
              VLOG(2) << "Reset kReduceWindow from sharding of reduced dimension to "
                      << "replication(i.e., Glue strategy): " << inst->ToString();
              glued_inst_num ++;
            }
          }

          auto* op = inst->operand(0);
          for (int j = 0; j < (*best_strategy_map)[op].dim_strategies().size(); ++j) {
            auto* op_dim_str = (*best_strategy_map)[op].mutable_dim_strategy(i);
            if (op_dim_str->Glue()) continue;
            auto op_dim_to_slice = op_dim_str->partition_dim();
            auto op_w_dim = window.dimensions(op_dim_to_slice);
            auto op_w_dim_size = op_w_dim.size();
            if (op_w_dim_size > 1) {
              *op_dim_str = DimStrategy();
              glued_inst_num ++;
            }
          }
          break;
        }

        case HloOpcode::kDot: {
          auto* lhs = inst->operand(0);
          auto* rhs = inst->operand(1);
          auto& lhs_shape = lhs->shape();
          auto& rhs_shape = rhs->shape();
          auto* lhs_str = (*best_strategy_map)[lhs].mutable_dim_strategy(i);
          auto* rhs_str = (*best_strategy_map)[rhs].mutable_dim_strategy(i);
          const DotDimensionNumbers &dim_nums = inst->dot_dimension_numbers();
          int64 lhs_contracting_dim = dim_nums.lhs_contracting_dimensions(0);
          int64 rhs_contracting_dim = dim_nums.rhs_contracting_dimensions(0);
          int num_replicas = dim_str->num_replicas();
          if (dim_str->IsPartial()) {
            if(lhs_shape.dimensions(lhs_contracting_dim) % num_replicas ||
               rhs_shape.dimensions(rhs_contracting_dim) % num_replicas) {
              *dim_str = DimStrategy();
            }
          }
          break;
        }

        default : {}
      }
    }
  }
}

void CostSpmdStrategy::PlanByAnnotations(const HloModule* module,
                  const HloInstMap<int>& inst_rank_map,
                  const HloInstSet& inst_scope,
                  GraphStrategy& graph_strategy) {
  HloInstMap<SharedDimStrategy>
  user_annotated_tensors = ExtractUserSplit(module);

  HloInstSet user_annotated_set;
  for (auto& annotated_tensor : user_annotated_tensors) {
    user_annotated_set.insert(annotated_tensor.first);
  }

  InferFromAnnotation(user_annotated_tensors,
                      inst_rank_map,
                      inst_scope,
                      graph_strategy);
  VLOG(1) << "after InferFromAnnotation";
  VLOG(1) << graph_strategy.ToString();

  HloInstMap<SharedDimStrategy> starting_strategies;
  const HloInstSet sub_graph_insts;  // empty: whole graph search
  bool changed = false;
  do {
    starting_strategies = FindBackStartingStrategies(graph_strategy.strategy_map(), sub_graph_insts);
    changed = ReverseInferByRank(starting_strategies, user_annotated_set,
                           inst_scope, inst_rank_map, graph_strategy);

    if (changed) {
      starting_strategies = FindStartingStrategies(graph_strategy.strategy_map(), sub_graph_insts);
      changed = InferByRank(starting_strategies, user_annotated_set,
                           inst_scope, inst_rank_map, graph_strategy);
    }
  } while (changed);

  // ignore user annotation
  user_annotated_set.clear();
  HloInstSet infered_insts;
  do {
    starting_strategies = FindBackStartingStrategies(graph_strategy.strategy_map(), sub_graph_insts);
    infered_insts = ReverseInferGreedy(starting_strategies, user_annotated_set,
                                       inst_scope, graph_strategy);

    VLOG(1) << "after calling ReverseInferGreedy";
    if (!infered_insts.empty()) {
      starting_strategies = FindStartingStrategies(graph_strategy.strategy_map(), sub_graph_insts);
      infered_insts = InferGreedy(starting_strategies, user_annotated_set,
                                  inst_scope, graph_strategy);
    }
  } while (!infered_insts.empty());
}

HloInstMap<std::pair<const HloInstruction*, const HloInstruction*>>
CostSpmdStrategy::CreateComputeToParamMap(const HloModule* module) {
  HloInstMap<std::pair<const HloInstruction*, const HloInstruction*>> compute_params;

  auto* entry = module->entry_computation();

  HloInstSet compute_insts;
  for (auto* inst : entry->instructions()) {
    if (is_compute_intensive(inst)) {
      compute_insts.insert(inst);
    }
  }

  for (auto* compute_inst : compute_insts) {
    for (auto* op : compute_inst->operands()) {
      if (op->opcode() == HloOpcode::kParameter) {
        if (compute_params.find(compute_inst) == compute_params.end()) {
          compute_params[compute_inst] = std::make_pair(op, nullptr);
        } else {
          CHECK(compute_params[compute_inst].second == nullptr);
          compute_params[compute_inst].second = op;
        }
      }
    }
  }

  for (auto& param : compute_params) {
    if (param.second.second != nullptr) {
      VLOG(1) << "compute inst: " << param.first->name() << ", param1: "
              << param.second.first->name() << ", param2: "
              << param.second.second->name();
    } else {
      VLOG(1) << "compute inst: " << param.first->name() << std::endl << "  param1: "
              << param.second.first->ToString() << ", param2: nullptr";
    }
  }

  return std::move(compute_params);
}

bool CostSpmdStrategy::StrategyPlanning(HloModule* module) {
  VLOG(2) << "before strategy planning, module:\n" << module->ToString();
  HloInstMap<int> inst_rank_map;
  HloInstMap<int> compute_rank_map;
  CalcInstRank(module, inst_rank_map, compute_rank_map);

  HloInstMap<std::pair<const HloInstruction*, const HloInstruction*>>
  compute_params = CreateComputeToParamMap(module);

  HloInstSet forward_insts = FindForwardInsts(module, false);

  GraphStrategy graph_strategy(cur_split_num_, cur_share_dev_);

  // NOTE(zycao): When running under 'N * strategy_infer' + 'N * transform'
  // mode, we needs to temporarily update shapes of all insts to make right
  // decisions during (oridinal >= 1) rounds. Then restore original shapes
  // after strategy infer.

  // Fast path for dims without real splitting.
  if (cur_split_num_ == 1) {
    HloInstMap<SharedDimStrategy>& strtg = graph_strategy.strategy_map();
    SharedDimStrategy glue = std::make_shared<DimStrategy>();
    for (auto inst : module->entry_computation()->MakeInstructionPostOrder()) {
      strtg[inst] = glue;
    }
  } else {
    PlanByAnnotations(module, inst_rank_map, forward_insts, graph_strategy);

    VLOG(1) << "after plan by annotation:";
    VLOG(1) << graph_strategy.ToString();

    int64 var_mem_limit = ServiceEnv::var_mem_limit();  // GB
    int64 opt_level = ServiceEnv::opt_level();

    var_mem_limit *= (int64)1024*1024*1024;  // convert to Bytes
    MemSavePlan mem_save_plan;
    bool succ = SplitPlanByMemCost(module,
                                   graph_strategy.strategy_map(),
                                   compute_rank_map,
                                   var_mem_limit,
                                   mem_save_plan);

    VLOG(0) << "estimated memory cost after splitting: "
            << HumanReadableNumBytes(mem_save_plan.expected_mem_cost_);
    VLOG(0) << "estimated memory cost before splitting: "
            << HumanReadableNumBytes(mem_save_plan.total_mem_cost_);
    VLOG(0) << "estimated orig memory cost: "
            << HumanReadableNumBytes(mem_save_plan.total_orig_mem_cost_);
    VLOG(0) << "memory limit: " << HumanReadableNumBytes(var_mem_limit);

    if (!succ) {
      VLOG(0) << "out of memory!";
      return false;
    }

    if (!mem_save_plan.split_for_mem_save_.empty() ||
        !mem_save_plan.split_for_compute_.empty()) {
      VLOG(1) << "before FindForwardInsts";

      auto* entry = module->entry_computation();
      CHECK(entry);
      std::vector<HloSubGraph> sub_graphs;
      if (opt_level<3) {
        sub_graphs = FindSubGraphs(module, forward_insts);

        if (!sub_graphs.empty()) {
          VLOG(2) << "try to build last sub graph";
          VLOG(2) << "build last sub graph";
          HloInstSet sub_graph_insts;
          for (HloSubGraph& sub : sub_graphs) {
            sub_graph_insts.insert(sub.insts_.begin(), sub.insts_.end());
          }
          const HloComputation* entry = module->entry_computation();
          const HloInstruction* root = entry->root_instruction();
          std::vector<const HloInstruction*> missed_insts;
          for (auto* inst : entry->instructions()) {
            if (sub_graph_insts.find(inst) == sub_graph_insts.end()) {
              if (inst != root) {
                VLOG(2) << "missed inst " << inst->ToString();
                missed_insts.emplace_back(inst);
              }
            }
          }

          VLOG(2) << sub_graph_insts.size() << " instructions are placed in sub graphs";
          VLOG(2) << missed_insts.size() << " instructions are missed";
          if (!missed_insts.empty()) {
            HloSubGraph& first_sub = sub_graphs.front();
            first_sub.insts_.insert(first_sub.insts_.end(),
                                   missed_insts.begin(),
                                   missed_insts.end());
          }
        }
      }

      if (sub_graphs.empty()) {
        // put whole graph into a single graph when
        //   1. no critical op
        //   2. optimize level>=3
        //
        // whole graph strategy exploration by ILP
        std::vector<HloInstruction*> post_order = entry->MakeInstructionPostOrder();
        std::vector<const HloInstruction*> all_insts;
        for (auto* inst : post_order) {
          all_insts.emplace_back(inst);
        }
        HloSubGraph single_sub_graph = BuildSubGraph(all_insts, nullptr, nullptr);
        single_sub_graph.id_ = sub_graphs.size();
        sub_graphs.emplace_back(std::move(single_sub_graph));
      }

      for (auto& sub : sub_graphs) {
        VLOG(2) << "sub graph size: " << sub.insts_.size();
        VLOG(2) << "sub graph:" << std::endl << sub.ToString();
      }

      PlanSubGraphs(module, graph_strategy.strategy_map(),
                    mem_save_plan, opt_level, sub_graphs);
      FinalizeCurStrMap(sub_graphs, graph_strategy);
    } else {
      VLOG(0) << "skip PlanSubGraphs";
    }
  }

  graph_strategy.Finalize();

  int glued_inst_num = 0;
  FillStrategyForAllInstructions(module, graph_strategy.mutable_hlo_strategy_map());
  // We do post process to reset strategy with default for some instructions.
  BestStrategyPostProcess(module, graph_strategy.mutable_hlo_strategy_map(), glued_inst_num);
  RecordStrategyToInsts(module, graph_strategy.hlo_strategy_map());
  DumpStrategies(module, graph_strategy.hlo_strategy_map());

  return true;
}

void CostSpmdStrategy::FillStrategyForAllInstructions(
    const HloModule* module,
    HloInstMap<HLOStrategy>* hlo_strategy_map) {
  auto* entry = module->entry_computation();
  for (const HloInstruction* instr : entry->MakeInstructionPostOrder()) {
    if (hlo_strategy_map->find(instr) != hlo_strategy_map->end()) continue;
    (*hlo_strategy_map)[instr].AddDimStrategy(DimStrategy());
  }
}

void CostSpmdStrategy::RecordStrategyToInsts(HloModule* module,
        const HloInstMap<HLOStrategy>& hlo_strategy_map) {
  auto* entry = module->entry_computation();
  for (auto* instr : entry->MakeInstructionPostOrder()) {
    int64 stride = 0;
    int64 size = 0;
    int64 stride_on_dim = 0;
    int num_replicas = local_dev_num_ * worker_num_;
    int partition_dim = -1;
    bool partial = false;
    if (hlo_strategy_map.find(instr) != hlo_strategy_map.end()) {
      auto& hlo_strategy = hlo_strategy_map.at(instr);
      for (auto& dim_str : hlo_strategy.dim_strategies()) {
        stride = dim_str.stride_on_elements();
        stride_on_dim = dim_str.stride_on_dim();
        partition_dim = dim_str.partition_dim();
        num_replicas = dim_str.num_replicas();
        partial = dim_str.IsPartial();

        std::unique_ptr<DimDistSpec> dim_spec = std::make_unique<DimDistSpec>();
        dim_spec->set_partial(partial);
        dim_spec->set_layout_aware_partition(
            stride, stride_on_dim, partition_dim, num_replicas);

        instr->mutable_dist_spec()->AddDimDistSpec(dim_spec);

        VLOG(1) << instr->ToString();
        VLOG(1) << "  " << dim_str.ToString();
      }
    } else {
      std::unique_ptr<DimDistSpec> dim_spec = std::make_unique<DimDistSpec>();
      dim_spec->set_partial(partial);
      dim_spec->set_layout_aware_partition(
          stride, stride_on_dim, partition_dim, num_replicas);

      instr->mutable_dist_spec()->AddDimDistSpec(dim_spec);

      VLOG(1) << "No strategy: " << instr->ToString();
    }
  }
}


void CostSpmdStrategy::DumpStrategies(
          const HloModule* module,
          const HloInstMap<HLOStrategy>& hlo_strategy_map) {
  auto entry = module->entry_computation();
  auto post_order = entry->MakeInstructionPostOrder();

  int no_strategy_num = 0;
  int with_strategy_num = 0;
  std::string strategies;
  strategies = "\ntotal instruction num: " + std::to_string(post_order.size());
  for (auto* inst : post_order) {
    strategies += "\n" + inst->ToString();
    if (hlo_strategy_map.find(inst) != hlo_strategy_map.end()) {
      strategies += "\n  strtg: " + hlo_strategy_map.at(inst).ToString();
      ++with_strategy_num;
    } else {
      strategies += "\n  no strtg";
      ++no_strategy_num;
    }
  }

  strategies += "\nnum of insts with strategy: " + std::to_string(with_strategy_num);
  strategies += "\nnum of insts without strategy: " + std::to_string(no_strategy_num);

  VLOG(1) << strategies;

  bool debug_mode = ServiceEnv::debug();
  if (debug_mode) {
    tensorflow::Env* env = tensorflow::Env::Default();
    Status status = tensorflow::WriteStringToFile(env, "strategies.txt", strategies);
    if (!status.ok()) {
      LOG(ERROR) << "Could not write strategies to strategies.txt: " << status;
    }
  }
}


}  // namespace xla

