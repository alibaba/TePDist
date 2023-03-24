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

#include "tensorflow/compiler/xla/service/parallel/utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {

void RewriteCustomCallsAsDots(HloModule* module) {
  auto resolve_dims = [&](string backend_config, string signature,
                          std::vector<int64>& dims) {
    auto signature_pos = 
      backend_config.find(signature);
    CHECK(signature_pos != string::npos);
    auto dims_start = backend_config.find("[", signature_pos);
    CHECK(dims_start != string::npos);
    auto dims_end = backend_config.find("]", signature_pos);
    CHECK(dims_end != string::npos);
    auto dims_str = backend_config.substr(dims_start+1, dims_end-dims_start-1);
    std::vector<std::string> parts = tensorflow::str_util::Split(dims_str, ',');
    for (auto part : parts) {
      CHECK(part.front() == '"');
      CHECK(part.back() == '"');
      part.pop_back();
      part.erase(part.begin());
      CHECK(part.find(",") == string::npos);
      dims.push_back(std::stoi(part));
    }
  };

  auto build_dot_dims = [&resolve_dims](string backend_config) 
      -> DotDimensionNumbers {
    DotDimensionNumbers dot_dims;

    std::vector<int64> lhs_contracting_dimensions;
    resolve_dims(backend_config,
                 "lhs_contracting_dimensions",
                 lhs_contracting_dimensions);

    std::vector<int64> rhs_contracting_dimensions;
    resolve_dims(backend_config,
                 "rhs_contracting_dimensions",
                 rhs_contracting_dimensions);

    std::vector<int64> lhs_batch_dimensions;
    resolve_dims(backend_config,
                 "lhs_batch_dimensions",
                 lhs_batch_dimensions);

    std::vector<int64> rhs_batch_dimensions;
    resolve_dims(backend_config,
                 "rhs_batch_dimensions",
                 rhs_batch_dimensions);

    //dot_dims.lhs_contracting_dimensions() = lhs_contracting_dimensions;
    for (int64 i = 0; i < lhs_contracting_dimensions.size(); ++i) {
      dot_dims.add_lhs_contracting_dimensions(lhs_contracting_dimensions[i]);
    }

    //dot_dims.rhs_contracting_dimensions() = rhs_contracting_dimensions;
    for (int64 i = 0; i < rhs_contracting_dimensions.size(); ++i) {
      dot_dims.add_rhs_contracting_dimensions(rhs_contracting_dimensions[i]);
    }

    //dot_dims.lhs_batch_dimensions = lhs_batch_dimensions;
    for (int64 i = 0; i < lhs_batch_dimensions.size(); ++i) {
      dot_dims.add_lhs_batch_dimensions(lhs_batch_dimensions[i]);
    }

    //dot_dims.rhs_batch_dimensions = rhs_batch_dimensions;
    for (int64 i = 0; i < rhs_batch_dimensions.size(); ++i) {
      dot_dims.add_rhs_batch_dimensions(rhs_batch_dimensions[i]);
    }
    return dot_dims;
  };

  auto entry = module->entry_computation();
  auto post_order = entry->MakeInstructionPostOrder();
  for (auto inst : post_order) {
    if (inst->opcode() != HloOpcode::kCustomCall ||
        inst->custom_call_target() != "__cublas$gemm")  continue;

    auto backend_config = inst->raw_backend_config_string();
    auto dot_dims = build_dot_dims(backend_config);
    auto lhs = inst->mutable_operand(0);
    auto rhs = inst->mutable_operand(1);
    auto new_inst =
         entry->AddInstruction(
            HloInstruction::CreateDot(inst->shape(), lhs, rhs, dot_dims,
                PrecisionConfig()));
    CHECK(entry->ReplaceInstructionNoShapeCheck(inst, new_inst).ok());
  }
}

bool
is_compute_intensive(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kDot ||
      inst->opcode() == HloOpcode::kConvolution) {
    return true;
  } else if (inst->opcode() == HloOpcode::kCustomCall &&
             inst->custom_call_target() == "__cublas$gemm") {
    return true;
  }

  return false;
}

HloInstSet FindBackwardInsts(const HloModule* module, bool prefer_backward) {
  const HloComputation* entry = module->entry_computation();
  std::unique_ptr<HloReachabilityMap> reachability = HloReachabilityMap::Build(entry);
  const std::map<int, std::string>* vars_map = module->variable_map();
  const HloModule::DefContext* def_ctx = module->def_ctx();
  CHECK(def_ctx);
  std::vector<const HloInstruction*> fetch_outputs;

  if (def_ctx->input_output_alias_map_.size()) {
    std::map<int, int> output_input_alias_map;
    for (auto& it : def_ctx->input_output_alias_map_) {
      output_input_alias_map[it.second] = it.first;
    }
    for (int i = 0; i < entry->root_instruction()->operand_count(); ++i) {
      if (output_input_alias_map.find(i) != output_input_alias_map.end()) continue;
      const HloInstruction* instr = entry->root_instruction()->operand(i);
      fetch_outputs.push_back(instr);
    }
  } else {
    for (int i = 0; i < entry->root_instruction()->operand_count() - vars_map->size(); ++i) {
      const HloInstruction* instr = entry->root_instruction()->operand(i);
      fetch_outputs.push_back(instr);
    }
  }

  HloInstSet backward_insts;
  auto post_order = entry->MakeInstructionPostOrder();

  for (HloInstruction* instr : entry->MakeInstructionPostOrder()) {
    bool forward = false;
    for (const HloInstruction* output : fetch_outputs) {
      if (reachability->IsReachable(instr, output)) {
        forward = true;
        break;
      }
    }

    if (!forward) {
      backward_insts.insert(instr);
      instr->metadata().set_backward(true);
    } else {
      instr->metadata().set_backward(false);
    }
  }

  if (prefer_backward) {
    HloInstSet back_candidates;
    for (auto* inst : backward_insts) {
      for (auto* op : inst->operands()) {
        if (backward_insts.find(op) == backward_insts.end()) {
          back_candidates.insert(op);
        }
      }
    }

    bool changed = true;
    if (back_candidates.empty()) {
      changed = false;
    }

    while (changed) {
      HloInstSet new_back_insts;
      for (auto* inst : back_candidates) {
        if (is_compute_intensive(inst)) {
          continue;
        }
        bool all_users_are_back = true;
        for (auto* user : inst->users()) {
          if (backward_insts.find(user) == backward_insts.end()) {
            all_users_are_back = false;
            break;
          }
        }

        if (all_users_are_back) {
          new_back_insts.insert(inst);
          backward_insts.insert(inst);
          const_cast<HloInstruction*>(inst)->metadata().set_backward(true);
        }
      }

      if (!new_back_insts.empty()) {
        changed = true;
        for (auto* new_back : new_back_insts) {
          back_candidates.erase(new_back);

          for (auto* op : new_back->operands()) {
            back_candidates.insert(op);
          }
        }
      } else {
        changed = false;
      }
    }
  }

  return std::move(backward_insts);
}

HloInstSet FindForwardInsts(const HloModule* module, bool prefer_backward) {
  HloInstSet backward_insts = FindBackwardInsts(module, prefer_backward);
  const HloComputation* entry = module->entry_computation();
  HloInstSet forward_insts;
  for (auto* inst : entry->instructions()) {
    if (backward_insts.find(inst) == backward_insts.end()) {
      forward_insts.insert(inst);
    }
  }

  return std::move(forward_insts);
}

const string kAllReduceType = "AllReduce";

bool AllowPartialPassThrough(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kReduce ||
      instr->opcode() == HloOpcode::kDot ||
      instr->opcode() == HloOpcode::kConvolution ||
      instr->opcode() == HloOpcode::kScatter;
}

// TODO(zycao): delete 'multiple_split_' flag after refactoring infer and
// transform process.
bool StrategyUtil::multiple_split_ = false;

std::vector<const HloInstruction*>
StableSort(const HloInstSet& insts) {
  std::unordered_set<const HloInstruction*> visited;
  auto is_ready = [&insts, &visited] (const HloInstruction* inst) -> bool {
    bool ready = true;
    for (auto* op : inst->operands()) {
      if (insts.find(op) == insts.end()) {
        continue;
      }
      if (visited.find(op) == visited.end()) {
        ready = false;
        break;
      }
    }

    return ready;
  };

  // 1. find initial ready instructions with stable order
  std::vector<const HloInstruction*> sorted_insts;
  sorted_insts.reserve(insts.size());
  for (auto* inst : insts) {
    if (is_ready(inst)) {
      sorted_insts.emplace_back(inst);
    }
  }

  std::sort(sorted_insts.begin(), sorted_insts.end(),
            [](const HloInstruction* lhs, const HloInstruction* rhs) {
                 return lhs->name() < rhs->name();
               });

  std::deque<const HloInstruction*> ready_insts;
  // 2. initialize ready_insts
  for (auto* inst : sorted_insts) {
    ready_insts.push_back(inst);
    visited.insert(inst);
  }

  // 3. sort instructions by topo order and instruction name
  while (!ready_insts.empty()) {
    auto* inst = ready_insts.front();
    ready_insts.pop_front();

    std::vector<HloInstruction*> sorted_users = inst->users();
    std::sort(sorted_users.begin(), sorted_users.end(),
              [](const HloInstruction* lhs, const HloInstruction* rhs) {
                   return lhs->name() < rhs->name();
                 });
    for (auto* user : sorted_users) {
      if (is_ready(user) && visited.find(user) == visited.end()) {
        if (insts.find(user) != insts.end()) sorted_insts.emplace_back(user);
        ready_insts.emplace_back(user);
        visited.insert(user);
      }
    }
  }

  return std::move(sorted_insts);
}

bool FromZeroConstant(HloInstruction* instr) {
  HloInstruction* curr_instr = instr;
  while (curr_instr->opcode() == HloOpcode::kBroadcast) {
    curr_instr = curr_instr->mutable_operand(0);
  }

  if (curr_instr->opcode() != HloOpcode::kConstant) return false;

  HloConstantInstruction* const_instr = DynCast<HloConstantInstruction>(curr_instr);

  if (curr_instr->shape().element_type() == PrimitiveType::F32) {
    auto& literal = curr_instr->literal();
    auto* data = (float *)literal.untyped_data();

    for (int i = 0 ; i < literal.size_bytes() / sizeof(data); ++i) {
      if (*(data + i) != 0) return false;
    }
  }

  return true;
}

SharedDimStrategy StrategyUtil::InferUnary(
    HloInstruction* inst,
    const DimStrategy& input_strategy, int input_idx) {
  CHECK(input_idx == 0);
  SharedDimStrategy inst_strategy =
                              std::make_shared<DimStrategy>(input_strategy);
  inst_strategy->set_partial(false);

  return inst_strategy;
}

SharedDimStrategy StrategyUtil::InferBinary(
    HloInstruction* inst,
    const DimStrategy& input_strategy, int input_idx) {
  CHECK(input_idx < 2);
  SharedDimStrategy inst_strategy =
                              std::make_shared<DimStrategy>(input_strategy);
  inst_strategy->set_partial(false);

  return inst_strategy;
}

SharedDimStrategy
StrategyUtil::InferBcast(
    HloInstruction* inst,
    const DimStrategy& input_strategy, int input_idx) {
  CHECK(input_idx == 0);
  auto input = inst->mutable_operand(input_idx);
  if (input_strategy.Glue()) {
    return std::make_shared<DimStrategy>();
  }

  auto& input_shape = input->shape();
  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();
  CHECK (input_dim >= 0);

  auto& bcast_dims = inst->dimensions();
  CHECK(input_dim < bcast_dims.size())
      << "input_dim=" << input_dim
      << " bcast_dims.size()=" << bcast_dims.size();
  if (!(lo > 0 && lo <= input_shape.dimensions(input_dim))) return std::make_shared<DimStrategy>();
  // CHECK(lo == input_stride_on_dim && input_dim == input_strategy.partition_dim());
  
  int64 output_dim = bcast_dims[input_dim];
  return std::make_shared<DimStrategy>(
      inst->shape(), output_dim, input_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::InferTernary(
    HloInstruction* inst,
    const DimStrategy& input_strategy, int input_idx) {
  CHECK(input_idx < 3);
  SharedDimStrategy inst_strategy =
                              std::make_shared<DimStrategy>(input_strategy);
  inst_strategy->set_partial(false);

  return inst_strategy;
}

SharedDimStrategy StrategyUtil::InferReshape(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  CHECK(input_idx == 0);
  // TODO(shiqing.fsq): The partition_dim_ can not copied directly.
  // E.g., %reshape.6255 = f32[2,12,1024]{2,1,0} reshape(f32[2,6,2,1024]{3,0,1,2}
  //       %transpose.6254), metadata={op_type="Reshape"}

  if (input_strategy.IsPartial()) {
    return std::make_shared<DimStrategy>();
  }
  // If input and output layout is same, just reset stride_on_dim.
  SharedDimStrategy inst_strategy =
                              std::make_shared<DimStrategy>(input_strategy);
  inst_strategy->ApplyToShape(inst->shape());

  return inst_strategy;
}

SharedDimStrategy StrategyUtil::InferConcat(
    HloInstruction* inst,
    const DimStrategy& input_strategy, int input_idx) {
  auto* op = inst->mutable_operand(input_idx);

  int64 lo = input_strategy.stride_on_dim();
  int64 dim = input_strategy.partition_dim();
  const HloConcatenateInstruction* concat = DynCast<HloConcatenateInstruction>(inst);
  auto& concat_dims = concat->dimensions();
  std::unordered_set<int> concat_dims_set(concat_dims.begin(), concat_dims.end());

  if (concat_dims_set.count(dim)) return std::make_shared<DimStrategy>();

  return std::make_shared<DimStrategy>(inst->shape(), dim, input_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::InferSlice(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  auto* op = inst->mutable_operand(input_idx);

  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();
  const HloSliceInstruction* slice = DynCast<HloSliceInstruction>(inst);
  const std::vector<int64> slice_starts = slice->slice_starts();
  const std::vector<int64> slice_limits = slice->slice_limits();
  const std::vector<int64> slice_strides = slice->slice_strides();

  if (slice_limits[input_dim] - slice_starts[input_dim] == op->shape().dimensions(input_dim) &&
      slice_strides[input_dim] == 1) {
    return std::make_shared<DimStrategy>(inst->shape(), input_dim, input_strategy.num_replicas(), lo);
  }

  return std::make_shared<DimStrategy>();
}

HloInstMap<SharedDimStrategy> StrategyUtil::InferDot(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  const DotDimensionNumbers &dim_nums = inst->dot_dimension_numbers();
  auto hlo_rank = inst->shape().rank();
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
           dim_nums.rhs_batch_dimensions_size());
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size() + 2, inst->shape().rank());
  int64 num_batches = dim_nums.lhs_batch_dimensions_size();
  int64 lhs_contracting_dim = dim_nums.lhs_contracting_dimensions(0);
  int64 rhs_contracting_dim = dim_nums.rhs_contracting_dimensions(0);

  auto from_operand = [&](int op_idx) -> HloInstMap<SharedDimStrategy> {
    HloInstMap<SharedDimStrategy> infer_res;
    auto* op = inst->mutable_operand(op_idx);
    auto* peer = inst->mutable_operand(1 - op_idx);
    if (!input_strategy.Glue()) {
      int64 lo = input_strategy.stride_on_dim();
      int64 dim = input_strategy.partition_dim();
      bool split_on_batch_dim = dim < num_batches ? 1 : 0;

      bool split_on_contract_dim = op_idx == 0 ? \
        (dim == lhs_contracting_dim ? 1 : 0) : \
        (dim == rhs_contracting_dim ? 1 : 0);

      if (split_on_contract_dim) {
        // Case 1. lhs or rhs infers each other directly
        int64 peer_contract_dim = op_idx == 0 ? rhs_contracting_dim : lhs_contracting_dim;
        infer_res[peer] = std::make_shared<DimStrategy>(
            peer->shape(), peer_contract_dim, input_strategy.num_replicas(), lo);
        // lhs_contracting_dim and lo are recorded for backinfer.
        infer_res[inst] = DimStrategy::MakePartial(
            input_strategy.num_replicas(), lhs_contracting_dim, lo);
      } else if (split_on_batch_dim) {
        // Case 2. (split on batch_dim) lhs or rhs infers dot only
        infer_res[inst] = std::make_shared<DimStrategy>(
            inst->shape(), dim, input_strategy.num_replicas(), lo);
        infer_res[peer] = std::make_shared<DimStrategy>();
      } else {
        // Case 3. (split on row for lhs or col for rhs) lhs or rhs infers dot only
        infer_res[inst] = std::make_shared<DimStrategy>(
            inst->shape(), op_idx == 0 ? hlo_rank - 2 : hlo_rank - 1,
            input_strategy.num_replicas(), lo);
        infer_res[peer] = std::make_shared<DimStrategy>();
      }
    } else {
      infer_res[peer] = std::make_shared<DimStrategy>();
      infer_res[inst] = std::make_shared<DimStrategy>();
    }
    return std::move(infer_res);
  };

  return from_operand(input_idx);
}

HloInstMap<SharedDimStrategy> StrategyUtil::InferConvolution(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  const ConvolutionDimensionNumbers& dim_nums = inst->convolution_dimension_numbers();
  CHECK(input_idx==0 || input_idx==1);
  auto* input = inst->mutable_operand(0);
  auto* filter = inst->mutable_operand(1);
  int64 input_batch_dimension = dim_nums.input_batch_dimension();
  int64 input_feature_dimension = dim_nums.input_feature_dimension();
  int64 kernel_input_feature_dimension = dim_nums.kernel_input_feature_dimension();
  int64 kernel_output_feature_dimension = dim_nums.kernel_output_feature_dimension();
  int64 output_batch_dimension = dim_nums.output_batch_dimension();
  int64 output_feature_dimension = dim_nums.output_feature_dimension();
  HloInstMap<SharedDimStrategy> infer_res;
  if (input_strategy.Glue()) {
    if (input_idx == 0) {
      infer_res[filter] = std::make_shared<DimStrategy>();
    } else if (input_idx == 1) {
      infer_res[input] = std::make_shared<DimStrategy>();
    }
    infer_res[inst] = std::make_shared<DimStrategy>();
    return infer_res;
  }

  if (input_idx == 0) {
    // input_idx 0 is always the input feature
    int64 lo = input_strategy.stride_on_dim();
    int64 dim = input_strategy.partition_dim();
    if (dim == input_batch_dimension) {
      infer_res[filter] = std::make_shared<DimStrategy>();
      infer_res[inst] = std::make_shared<DimStrategy>(
          inst->shape(), output_batch_dimension, input_strategy.num_replicas(), lo);
    } else if (dim == input_feature_dimension) {
      // Partial case
      infer_res[filter] = std::make_shared<DimStrategy>(
          filter->shape(), kernel_input_feature_dimension, input_strategy.num_replicas(), lo);
      infer_res[inst] = DimStrategy::MakePartial(
          input_strategy.num_replicas(), input_feature_dimension, lo);
    } else {
      infer_res[filter] = std::make_shared<DimStrategy>();
      infer_res[inst] = std::make_shared<DimStrategy>();
    }
  } else if (input_idx == 1) {
    // input_idx 1 is always the kernel filter
    int64 lo = input_strategy.stride_on_dim();
    int64 dim = input_strategy.partition_dim();
    if (dim == kernel_input_feature_dimension) {
      // Partial case
      infer_res[input] = std::make_shared<DimStrategy>(
        input->shape(), input_feature_dimension, input_strategy.num_replicas(), lo);
      infer_res[inst] = DimStrategy::MakePartial(
         input_strategy.num_replicas(), input_feature_dimension, lo);
    } else if (dim == kernel_output_feature_dimension) {
      infer_res[input] = std::make_shared<DimStrategy>();
      infer_res[inst] = std::make_shared<DimStrategy>(
          inst->shape(), output_feature_dimension, input_strategy.num_replicas(), lo);
    } else {
      infer_res[input] = std::make_shared<DimStrategy>();
      infer_res[inst] = std::make_shared<DimStrategy>();
    }
  }

  return std::move(infer_res);
}

SharedDimStrategy StrategyUtil::InferReduce(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  auto* op = inst->mutable_operand(input_idx);

  int64 lo = input_strategy.stride_on_dim();
  int64 split_dim = input_strategy.partition_dim();
  
  auto& reduce_dims = inst->dimensions();
  std::set<int> reduce_dims_set(reduce_dims.begin(), reduce_dims.end());
  if (reduce_dims_set.count(split_dim)) {
    // split_dim and lo of operand 0 are recorded for backinfer.
    return DimStrategy::MakePartial(
        input_strategy.num_replicas(), split_dim, lo);
  }

  int64 hlo_dim = 0;
  for (int64 r = 0; r < op->shape().rank(); ++r) {
    if (reduce_dims_set.count(r)) continue;

    if (split_dim == r) break;
    ++hlo_dim;
  }
  return std::make_shared<DimStrategy>(inst->shape(), hlo_dim, input_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::InferReverse(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  auto* op = inst->mutable_operand(input_idx);

  int64 lo = input_strategy.stride_on_dim();
  int64 split_dim = input_strategy.partition_dim();
  
  auto& reverse_dims = inst->dimensions();
  std::unordered_set<int> reverse_dims_set(reverse_dims.begin(), reverse_dims.end());
  if (reverse_dims_set.count(split_dim)) return std::make_shared<DimStrategy>();

  return std::make_shared<DimStrategy>(inst->shape(), split_dim, input_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::InferGather(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  auto rank = inst->shape().rank();

  const HloGatherInstruction* gather = DynCast<HloGatherInstruction>(inst);
  auto& gather_dims = gather->gather_dimension_numbers();
  auto& offset_dims = gather_dims.offset_dims();
  auto index_vector_dim = gather_dims.index_vector_dim();
  auto slice_sizes = gather->gather_slice_sizes();
  auto& collapsed_slice_dims = gather_dims.collapsed_slice_dims();

  auto* op = inst->operand(input_idx);
  int64 lo = input_strategy.stride_on_dim();
  int64 op_slice_dim = input_strategy.partition_dim();

  std::set<int64> offset_dims_set(offset_dims.begin(), offset_dims.end());
  std::set<int64> collapsed_dims_set(collapsed_slice_dims.begin(), 
                                     collapsed_slice_dims.end());
  if (input_idx == 0) {
    // Infer from operand
    if (collapsed_dims_set.count(op_slice_dim)) return std::make_shared<DimStrategy>();

    int op_dim = 0;
    int index_count = 0;
    for (int inst_dim = 0; inst_dim < rank; ++inst_dim) {
      if (!offset_dims_set.count(inst_dim)) {
        CHECK(++index_count <= index_vector_dim);
        continue;
      }
      while (collapsed_dims_set.count(op_dim)) ++op_dim;
      if (op_dim == op_slice_dim) {
        if (slice_sizes[op_dim] == op->shape().dimensions(op_dim)) {
            return std::make_shared<DimStrategy>(inst->shape(), inst_dim,
                               input_strategy.num_replicas(), lo);
        } else {
          return std::make_shared<DimStrategy>();
        }
        ++op_dim;
      }
    }
  } else if (input_idx == 1) {
    // Find the dim which corresponds to the partitioned index dim.
    int id_dim = 0;
    for (int inst_dim = 0; inst_dim < rank; ++inst_dim) {
      if (offset_dims_set.count(inst_dim)) continue;
      if (id_dim == op_slice_dim) {
        CHECK(inst->shape().dimensions(inst_dim) ==
              op->shape().dimensions(id_dim));
        return std::make_shared<DimStrategy>(
            inst->shape(), inst_dim, input_strategy.num_replicas(), lo);
      }
      ++id_dim;
    }
  }

  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::InferTranspose(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  auto* input = inst->mutable_operand(input_idx);
  auto& input_shape = input->shape();

  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();

  int64 hlo_dim = -1;
  for (int64 i = 0; i < inst->shape().rank(); ++i) {
    // where do I move?
    if (input_dim == inst->dimensions(i)) {
      hlo_dim = i;
      break;
    }
  }

  return std::make_shared<DimStrategy>(inst->shape(), hlo_dim, input_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::InferCustomCall(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  CHECK(0);
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::InferCustomCollective(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  auto* coll = DynCast<HloCustomCollectiveInstruction>(inst);
  if (coll->collective_type() == kAllReduceType) {

    if (input_strategy.IsPartial()) {
      return std::make_shared<DimStrategy>();
    } else {
      int64 input_dim = input_strategy.partition_dim();
      return std::make_shared<DimStrategy>(inst->shape(), input_dim, input_strategy.num_replicas());
    }
  } else {
    CHECK(0);
  }
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::InferGetTupleElement(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  SharedDimStrategy inst_strategy =
                              std::make_shared<DimStrategy>(input_strategy);
  inst_strategy->set_partial(false);

  return inst_strategy;
}

SharedDimStrategy StrategyUtil::InferSelectAndScatter(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  const Window& window = inst->window();
  int64 input_dim = input_strategy.partition_dim();

  if (input_idx == 2 ||
      window.dimensions(input_dim).size() != 1 ||
      window.dimensions(input_dim).stride() != 1) {
    return std::make_shared<DimStrategy>();
  }

  int64 lo = input_strategy.stride_on_dim();
  return std::make_shared<DimStrategy>(inst->shape(), input_dim, input_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::InferReduceWindow(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  auto* op = inst->mutable_operand(input_idx);
  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();

  const Window& window = inst->window();
  if (window.dimensions(input_dim).size() != 1 ||
      window.dimensions(input_dim).stride() != 1 ) {
    return std::make_shared<DimStrategy>();
  }

  return std::make_shared<DimStrategy>(inst->shape(), input_dim, input_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::InferSort(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  auto* op = inst->mutable_operand(input_idx);
  auto& sort_dims = inst->dimensions();
  std::set<int> sort_dims_set(sort_dims.begin(), sort_dims.end());

  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();
  if (sort_dims_set.count(input_dim)) return std::make_shared<DimStrategy>();
  return std::make_shared<DimStrategy>(inst->shape(), input_dim, input_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::InferPad(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  const HloPadInstruction* pad = DynCast<HloPadInstruction>(inst);
  if (pad->operand(input_idx) == pad->padding_value()) return std::make_shared<DimStrategy>();

  auto* op = inst->mutable_operand(input_idx);
  auto& padding_config = pad->padding_config();

  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();
  if (input_dim < 0) return std::make_shared<DimStrategy>();

  CHECK(input_dim < padding_config.dimensions_size());

  if (padding_config.dimensions(input_dim).edge_padding_low() != 0 ||
      padding_config.dimensions(input_dim).edge_padding_high() != 0 ||
      padding_config.dimensions(input_dim).interior_padding() != 0) {
    return std::make_shared<DimStrategy>();
  }

  return std::make_shared<DimStrategy>(pad->shape(), input_dim, input_strategy.num_replicas(), lo);
}

HloInstMap<SharedDimStrategy> StrategyUtil::InferScatter(
    HloInstruction* inst,
    const DimStrategy& input_strategy,
    int input_idx) {
  HloInstMap<SharedDimStrategy> infer_res;

  auto* operand = inst->mutable_operand(0);
  auto& operand_shape = operand->shape();
  auto* index = inst->mutable_operand(1);
  auto& index_shape = index->shape();
  auto operand_rank = operand_shape.rank();
  auto* updates = inst->mutable_operand(2);
  auto& updates_shape = updates->shape();
  auto updates_rank = updates_shape.rank();
  auto& hlo_shape = inst->shape();
  auto hlo_rank = hlo_shape.rank();

  CHECK(operand_shape == hlo_shape);

  auto& scatter_dims = inst->scatter_dimension_numbers();
  auto& update_window_dims = scatter_dims.update_window_dims();
  auto& inserted_window_dims = scatter_dims.inserted_window_dims();
  auto& scatter_to_operand_dims = scatter_dims.scatter_dims_to_operand_dims();
  int64 index_vector_dim = scatter_dims.index_vector_dim();
  std::set<int64> update_dims_set(
      update_window_dims.begin(), update_window_dims.end());
  std::set<int64> inserted_dims_set(
      inserted_window_dims.begin(), inserted_window_dims.end());
  std::set<int64> to_operand_dims_set(
      scatter_to_operand_dims.begin(), scatter_to_operand_dims.end());

  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();

  // For scatter instruction, operand and updates should keep same strategy.
  // Neccessary check should be done and consistent strategies should be applied
  // to scatter inst, operand and updates.
  if (input_idx == 0) {
    if (to_operand_dims_set.count(input_dim)) {
      infer_res[inst] = std::make_shared<DimStrategy>();
      return infer_res;
    }
    int update_index = 0;
    for (int64 inst_dim = 0; inst_dim < input_dim; ++inst_dim) {
      if (inserted_dims_set.count(inst_dim)) continue;
      ++update_index;
    }
    CHECK(update_index < update_window_dims.size());
    int64 update_dim = update_window_dims[update_index];
    // TODO(zycao): delete 'multiple_split_' flag after refactoring infer and
    // transform process.
    CHECK(multiple_split_ || (hlo_shape.dimensions(input_dim) ==
                              updates_shape.dimensions(update_dim)));
    infer_res[inst] = std::make_shared<DimStrategy>(
        hlo_shape, input_dim, input_strategy.num_replicas(), lo);
    infer_res[updates] = std::make_shared<DimStrategy>(
        updates_shape, update_dim, input_strategy.num_replicas(), lo);
    return infer_res;
  }

  if (input_idx == 1) {
    if (input_dim < index_vector_dim && FromZeroConstant(operand)) {
      infer_res[updates] = std::make_shared<DimStrategy>(
          updates_shape, input_dim, input_strategy.num_replicas());
      infer_res[inst] = DimStrategy::MakePartial(
          input_strategy.num_replicas(), input_dim, lo);
    }
    return infer_res;
  }

  // input_idx == 2
  if (!update_dims_set.count(input_dim)) {
    infer_res[inst] = std::make_shared<DimStrategy>();
    return infer_res;
  }

  if (inserted_dims_set.size() == 1) {
    int major = *inserted_dims_set.begin();
    if (major != input_dim) return std::move(infer_res);
    infer_res[index] = std::make_shared<DimStrategy>(
        index_shape, index_vector_dim - 1, input_strategy.num_replicas());
    infer_res[updates] = std::make_shared<DimStrategy>(
        updates_shape, major, input_strategy.num_replicas());
    infer_res[inst] = DimStrategy::MakePartial(
        input_strategy.num_replicas(), index_vector_dim - 1);
  }

  int update_index = 0;
  for (int inst_dim = 0; inst_dim < hlo_rank; ++inst_dim) {
    if (inserted_dims_set.count(inst_dim)) continue;
    CHECK(update_index < update_window_dims.size());
    int64 update_dim = update_window_dims[update_index];
    if (update_dim == input_dim) {
      if (to_operand_dims_set.count(inst_dim)) {
        infer_res[inst] = std::make_shared<DimStrategy>();
        return infer_res;
      }
      // TODO(zycao): delete 'multiple_split_' flag after refactoring infer and
      // transform process.
      CHECK(multiple_split_ || (hlo_shape.dimensions(inst_dim) ==
                                updates_shape.dimensions(update_dim)));
      infer_res[inst] = std::make_shared<DimStrategy>(
          hlo_shape, inst_dim, input_strategy.num_replicas(), lo);
      infer_res[operand] = std::make_shared<DimStrategy>(
          operand_shape, inst_dim, input_strategy.num_replicas(), lo);
      return infer_res;
    }
    ++update_index;
  }
  return std::move(infer_res);
}

SharedDimStrategy StrategyUtil::BackInferBcast(const HloInstruction* inst,
                                               const DimStrategy& inst_strategy,
                                               int input_idx) {
  auto* input = inst->operand(input_idx);
  auto& input_shape = input->shape();
  if (ShapeUtil::IsScalar(input_shape)) return std::make_shared<DimStrategy>();

  int64 lo = inst_strategy.stride_on_dim();
  int64 dim = inst_strategy.partition_dim();
  auto& bcast_dims = inst->dimensions();
  std::unordered_set<int> bcast_dims_set(bcast_dims.begin(), bcast_dims.end());

  if (!bcast_dims_set.count(dim)) return std::make_shared<DimStrategy>();

  int64 input_dim = -1;
  for (int64 r = 0; r < input_shape.rank(); ++r) {
    if (dim == bcast_dims[r]) {
      CHECK(input_dim == -1);
      input_dim = r;
    }
  }

  return std::make_shared<DimStrategy>(input_shape, input_dim, inst_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::BackInferScatter(const HloInstruction* inst,
                                           const DimStrategy& inst_strategy,
                                           int input_idx) {
  auto* operand = inst->operand(0);
  auto& operand_shape = operand->shape();
  auto operand_rank = operand_shape.rank();
  auto* index = inst->operand(1);
  auto& index_shape = index->shape();
  auto* updates = inst->operand(2);
  auto& updates_shape = updates->shape();
  auto updates_rank = updates_shape.rank();
  auto& hlo_shape = inst->shape();
  auto hlo_rank = hlo_shape.rank();

  CHECK(operand_shape == hlo_shape);

  auto& scatter_dims = inst->scatter_dimension_numbers();
  auto& update_window_dims = scatter_dims.update_window_dims();
  auto& inserted_window_dims = scatter_dims.inserted_window_dims();
  auto& scatter_to_operand_dims = scatter_dims.scatter_dims_to_operand_dims();
  int64 index_vector_dim = scatter_dims.index_vector_dim();
  std::set<int64> update_dims_set(
      update_window_dims.begin(), update_window_dims.end());
  std::set<int64> inserted_dims_set(
      inserted_window_dims.begin(), inserted_window_dims.end());
  std::set<int64> to_operand_dims_set(
      scatter_to_operand_dims.begin(), scatter_to_operand_dims.end());

  int64 lo = inst_strategy.stride_on_dim();
  int64 dim = inst_strategy.partition_dim();

  // For scatter instruction, operand and updates should keep same strategy.
  // Neccessary check should be done and consistent strategies should be applied
  // to scatter inst, operand and updates.
  if (input_idx == 0) {
    if (inst_strategy.IsPartial()) return std::make_shared<DimStrategy>();
    return std::make_shared<DimStrategy>(operand_shape, dim, inst_strategy.num_replicas(), lo);
  } else if (input_idx == 1) {
    if (inst_strategy.IsPartial() && FromZeroConstant(const_cast<HloInstruction*>(operand))) {
      return std::make_shared<DimStrategy>(
          index_shape, index_vector_dim - 1, inst_strategy.num_replicas());
    }
    return std::make_shared<DimStrategy>();
  } else {
    // input_idx == 2
    if (inst_strategy.IsPartial() && FromZeroConstant(const_cast<HloInstruction*>(operand)) &&
        inserted_window_dims.size() == 1) { 
      int major = inserted_window_dims[0];
      return std::make_shared<DimStrategy>(updates_shape, major, inst_strategy.num_replicas());
    }

    if (update_dims_set.find(dim) != update_dims_set.end()) {
      int64 update_dim = dim;
      // TODO(zycao): delete 'multiple_split_' flag after refactoring infer and
      // transform process.
      CHECK(multiple_split_ || (hlo_shape.dimensions(dim) ==
                                updates_shape.dimensions(update_dim)));
      return std::make_shared<DimStrategy>(
          updates_shape, update_dim, inst_strategy.num_replicas(), lo);     
    }
  }

  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInferUnary(const HloInstruction* inst,
                                         const DimStrategy& inst_strategy,
                                         int input_idx) {
  SharedDimStrategy op_strategy = std::make_shared<DimStrategy>(inst_strategy);
  op_strategy->set_partial(false);
  return op_strategy;
}

SharedDimStrategy StrategyUtil::BackInferCrossReplicaSum(const HloInstruction* inst,
                                                   const DimStrategy& inst_strategy,
                                                   int input_idx) {
  CHECK(0);
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInferBinary(const HloInstruction* inst,
                                          const DimStrategy& inst_strategy,
                                          int input_idx) {
  SharedDimStrategy op_strategy = std::make_shared<DimStrategy>(inst_strategy);
  op_strategy->set_partial(false);
  return op_strategy;
}

SharedDimStrategy StrategyUtil::BackInferTernary(const HloInstruction* inst,
                                           const DimStrategy& inst_strategy,
                                           int input_idx) {
  SharedDimStrategy op_strategy = std::make_shared<DimStrategy>(inst_strategy);
  op_strategy->set_partial(false);
  return op_strategy;
}

SharedDimStrategy StrategyUtil::BackInferConcat(const HloInstruction* inst,
                                          const DimStrategy& inst_strategy,
                                          int input_idx) {
  int64 lo = inst_strategy.stride_on_dim();
  int64 dim = inst_strategy.partition_dim();
  const HloConcatenateInstruction* concat = DynCast<HloConcatenateInstruction>(inst);
  auto& concat_dims = concat->dimensions();
  std::unordered_set<int> concat_dims_set(concat_dims.begin(), concat_dims.end());
  if (concat_dims_set.count(dim)) return std::make_shared<DimStrategy>();

  auto* op = inst->operand(input_idx);
  return std::make_shared<DimStrategy>(op->shape(), dim, inst_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::BackInferSlice(const HloInstruction* inst,
                                         const DimStrategy& inst_strategy,
                                         int input_idx) {
  int64 lo = inst_strategy.stride_on_dim();
  int64 dim = inst_strategy.partition_dim();

  const HloSliceInstruction* slice = DynCast<HloSliceInstruction>(inst);
  const std::vector<int64> slice_starts = slice->slice_starts();
  const std::vector<int64> slice_limits = slice->slice_limits();
  const std::vector<int64> slice_strides = slice->slice_strides();

  auto* op = inst->operand(input_idx);

  if ((slice_limits[dim] - slice_starts[dim]) == op->shape().dimensions(dim)
      && slice_strides[dim] == 1) {
    return std::make_shared<DimStrategy>(op->shape(), dim, inst_strategy.num_replicas(), lo);
  }
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInferReshape(const HloInstruction* inst,
                                           const DimStrategy& inst_strategy,
                                           int input_idx) {
  CHECK(input_idx==0);
  int64 input_stride_on_dim;
  SharedDimStrategy input_strategy =
                              std::make_shared<DimStrategy>(inst_strategy);
  input_strategy->ApplyToShape(inst->operand(input_idx)->shape());
  if (input_strategy->partition_dim() != inst_strategy.partition_dim()) {
    VLOG(2) << "[BackInferReshape]-> " << inst->ToString()
            << " inst_strategy->" << inst_strategy.ToString()
            << " update partition_dim->" << "output_dim=" << inst_strategy.partition_dim()
            << " input_dim=" << input_strategy->partition_dim();
  }
  return input_strategy;
}

SharedDimStrategy StrategyUtil::BackInferDot(const HloInstruction* inst,
                                       const DimStrategy& inst_strategy,
                                       int input_idx) {
  CHECK(input_idx==0 || input_idx==1);
  const HloInstruction* input = inst->operand(input_idx);
  auto& input_shape = input->shape();
  int64 input_rank = input_shape.rank();

  const DotDimensionNumbers &dim_nums = inst->dot_dimension_numbers();
  if (inst_strategy.IsPartial()) {
    int64 contracting_dim = input_idx == 0 ? \
        dim_nums.lhs_contracting_dimensions(0) : dim_nums.rhs_contracting_dimensions(0);
    return std::make_shared<DimStrategy>(input_shape, contracting_dim,
                       inst_strategy.num_replicas(),
                       inst_strategy.stride_on_dim());
  }

  int64 num_batches;
  if (input_idx==0) {
    num_batches = dim_nums.lhs_batch_dimensions_size();
  } else {
    num_batches = dim_nums.rhs_batch_dimensions_size();
  }

  auto& inst_shape = inst->shape();
  int64 inst_rank = inst_shape.rank();
  int64 inst_lo = inst_strategy.stride_on_dim();
  int64 inst_dim = inst_strategy.partition_dim();

  if (inst_dim < num_batches) {
    return std::make_shared<DimStrategy>(input_shape, inst_dim, inst_strategy.num_replicas(), inst_lo);
  } else if (inst_dim == inst_rank - 2 && input_idx == 0) {
    // LHS case
    int64 lhs_contracting_dim = dim_nums.lhs_contracting_dimensions(0);
    int64 lhs_dim = -1;
    if (lhs_contracting_dim == input_rank-1) {
      lhs_dim = input_rank-2;
    } else {
      lhs_dim = input_rank-1;
    }
    return std::make_shared<DimStrategy>(input_shape, lhs_dim, inst_strategy.num_replicas(), inst_lo);
  } else if (inst_dim == inst_rank - 1 && input_idx == 1) {
    // RHS case
    int64 rhs_contracting_dim = dim_nums.rhs_contracting_dimensions(0);

    int64 rhs_dim = -1;
    if (rhs_contracting_dim == input_rank-1) {
      rhs_dim = input_rank-2;
    } else {
      rhs_dim = input_rank-1;
    }
    return std::make_shared<DimStrategy>(input_shape, rhs_dim, inst_strategy.num_replicas(), inst_lo);
  }
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInferConvolution(const HloInstruction* inst,
                                                     const DimStrategy& inst_strategy,
                                                     int input_idx) {
  const ConvolutionDimensionNumbers& dim_nums = inst->convolution_dimension_numbers();
  CHECK(input_idx==0 || input_idx==1);
  const HloInstruction* input = inst->operand(0);
  const HloInstruction* filter = inst->operand(1);
  int64 input_batch_dimension = dim_nums.input_batch_dimension();
  int64 input_feature_dimension = dim_nums.input_feature_dimension();
  int64 kernel_input_feature_dimension = dim_nums.kernel_input_feature_dimension();
  int64 kernel_output_feature_dimension = dim_nums.kernel_output_feature_dimension();
  int64 output_batch_dimension = dim_nums.output_batch_dimension();
  int64 output_feature_dimension = dim_nums.output_feature_dimension();

  if (inst_strategy.IsPartial()) {
    int dim = input_idx == 0 ? input_feature_dimension : kernel_input_feature_dimension;
    const HloInstruction* op = input_idx == 0 ? input : filter;
    return std::make_shared<DimStrategy>(op->shape(), dim,
        inst_strategy.num_replicas(), inst_strategy.stride_on_dim());
  }

  int64 inst_lo = inst_strategy.stride_on_dim();
  int64 inst_dim = inst_strategy.partition_dim();
  if (inst_dim == output_batch_dimension) {
    if (input_idx == 0) {
      return std::make_shared<DimStrategy>(
          input->shape(), input_batch_dimension, inst_strategy.num_replicas(), inst_lo);
    } else if (input_idx == 1) {
      return std::make_shared<DimStrategy>();
    }
  } else if (inst_dim == output_feature_dimension) {
    if (input_idx == 0) {
      return std::make_shared<DimStrategy>();
    } else if (input_idx == 1) {
      return std::make_shared<DimStrategy>(
        filter->shape(), kernel_output_feature_dimension, inst_strategy.num_replicas(), inst_lo);
    }
  } else {
    return std::make_shared<DimStrategy>();
  }
}

SharedDimStrategy StrategyUtil::BackInferReduce(const HloInstruction* inst,
                                          const DimStrategy& inst_strategy,
                                          int input_idx) {
  int64 lo = inst_strategy.stride_on_dim();
  int64 dim = inst_strategy.partition_dim();
  auto& reduce_dims = inst->dimensions();
  std::unordered_set<int> reduce_dims_set(reduce_dims.begin(), reduce_dims.end());

  auto& input_shape = inst->operand(input_idx)->shape();
 
  if (ShapeUtil::IsScalar(input_shape)) return std::make_shared<DimStrategy>();
 
  if (inst_strategy.IsPartial()) {
    if (input_idx != 0) return std::make_shared<DimStrategy>();
    CHECK(reduce_dims_set.count(dim));
    return std::make_shared<DimStrategy>(input_shape, dim, inst_strategy.num_replicas(),
                       inst_strategy.stride_on_dim());
  }

  int64 input_dim = 0;
  int64 output_dim = 0;
  for (int64 r = 0; r < input_shape.rank(); ++r) {
    if (reduce_dims_set.count(r)) continue;
    if (output_dim == dim) {
      input_dim = r;
      break;
    }

    ++output_dim;
  }

  return std::make_shared<DimStrategy>(input_shape, input_dim, inst_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::BackInferReverse(const HloInstruction* inst,
                                          const DimStrategy& inst_strategy,
                                          int input_idx) {
  int64 lo = inst_strategy.stride_on_dim();
  int64 split_dim = inst_strategy.partition_dim();
  auto& input_shape = inst->operand(input_idx)->shape();
  auto& reverse_dims = inst->dimensions();
  std::unordered_set<int> reverse_dims_set(reverse_dims.begin(), reverse_dims.end());
 
  if (reverse_dims_set.count(split_dim)) return std::make_shared<DimStrategy>();
  return std::make_shared<DimStrategy>(input_shape, split_dim, inst_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::BackInferGather(const HloInstruction* inst,
                                          const DimStrategy& inst_strategy,
                                          int input_idx) {
  auto& shape = inst->shape();
  int64 rank = shape.rank();
  int64 lo = inst_strategy.stride_on_dim();
  int64 dim = inst_strategy.partition_dim();

  auto* op = inst->operand(input_idx);

  const HloGatherInstruction* gather = DynCast<HloGatherInstruction>(inst);
  auto& gather_dims = gather->gather_dimension_numbers();
  auto& offset_dims = gather_dims.offset_dims();
  auto index_vector_dim = gather_dims.index_vector_dim();
  auto slice_sizes = gather->gather_slice_sizes();
  auto& collapsed_slice_dims = gather_dims.collapsed_slice_dims();
  int64 num_collapsed = collapsed_slice_dims.size();

  std::set<int64> offset_dims_set(offset_dims.begin(), offset_dims.end());
  std::set<int64> collapsed_dims_set(collapsed_slice_dims.begin(), 
                                     collapsed_slice_dims.end());
  // for (auto collapsed : collapsed_slice_dims) {
  //   CHECK(slice_sizes[collapsed] == 1);
  // }

  if (input_idx == 0) {
    if (!offset_dims_set.count(dim)) return std::make_shared<DimStrategy>();

    int op_dim = 0;
    int index_count = 0;
    for (int inst_dim = 0; inst_dim < dim; ++inst_dim) {
      while (collapsed_dims_set.count(op_dim)) ++op_dim;
      if (!offset_dims_set.count(inst_dim)) {
        CHECK(++index_count <= index_vector_dim);
        continue;
      }
      ++op_dim;
    }
    // TODO(zycao): delete 'multiple_split_' flag after refactoring infer and
    // transform process.
    CHECK(multiple_split_ || (slice_sizes[op_dim] ==
                              op->shape().dimensions(op_dim)));
    return std::make_shared<DimStrategy>(op->shape(), op_dim, inst_strategy.num_replicas(), lo);
  } else if (input_idx == 1) {
    if (offset_dims_set.count(dim)) return std::make_shared<DimStrategy>();
    // Find the dim which corresponds to the partitioned index dim.
    int id_dim = 0;
    for (int inst_dim = 0; inst_dim < dim; ++inst_dim) {
      if (offset_dims_set.count(inst_dim)) continue;
      ++id_dim;
    }
    CHECK(inst->shape().dimensions(dim) == op->shape().dimensions(id_dim));
    return std::make_shared<DimStrategy>(op->shape(), id_dim, inst_strategy.num_replicas(), lo);
  }

  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInferPad(const HloInstruction* inst,
                                       const DimStrategy& inst_strategy,
                                       int input_idx) {
  const HloPadInstruction* pad = DynCast<HloPadInstruction>(inst);
  if (pad->operand(input_idx) == pad->padding_value()) return std::make_shared<DimStrategy>();

  auto* op = inst->operand(input_idx);
  auto& padding_config = pad->padding_config();

  int64 lo = inst_strategy.stride_on_dim();
  int64 dim = inst_strategy.partition_dim();
  if (dim < 0) return std::make_shared<DimStrategy>();

  CHECK(dim < padding_config.dimensions_size());

  if (padding_config.dimensions(dim).edge_padding_low() != 0 ||
      padding_config.dimensions(dim).edge_padding_high() != 0 ||
      padding_config.dimensions(dim).interior_padding() != 0) {
    return std::make_shared<DimStrategy>();
  }

  return std::make_shared<DimStrategy>(op->shape(), dim, inst_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::BackInferTranspose(const HloInstruction* inst,
                                             const DimStrategy& inst_strategy,
                                             int input_idx) {
  int64 lo = inst_strategy.stride_on_dim();
  int64 dim = inst_strategy.partition_dim();
  auto input_dim = inst->dimensions(dim);
  return std::make_shared<DimStrategy>(
      inst->operand(input_idx)->shape(),
      input_dim, inst_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::BackInferIota(const HloInstruction* inst,
                                        const DimStrategy& inst_strategy,
                                        int input_idx) {
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInferCustomCall(const HloInstruction* inst,
                                              const DimStrategy& inst_strategy,
                                              int input_idx) {
  CHECK(0);
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInferCustomCollective(const HloInstruction* inst,
                                              const DimStrategy& inst_strategy,
                                              int input_idx) {
  auto* coll = DynCast<HloCustomCollectiveInstruction>(inst);
  if (coll->collective_type() == kAllReduceType) {
    CHECK(!inst_strategy.IsPartial());
    int64 dim = inst_strategy.partition_dim();
    return std::make_shared<DimStrategy>(inst->shape(), dim, inst_strategy.num_replicas());
  } else {
    CHECK(0);
  }
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInferGetTupleElement(const HloInstruction* inst,
                                                   const DimStrategy& inst_strategy,
                                                   int input_idx) {
  SharedDimStrategy op_strategy = std::make_shared<DimStrategy>(inst_strategy);
  op_strategy->set_partial(false);
  return op_strategy;
}

SharedDimStrategy StrategyUtil::BackInferReduceWindow(const HloInstruction* inst,
                                                const DimStrategy& inst_strategy,
                                                int input_idx) {
  auto* op = inst->operand(input_idx);
  int64 lo = inst_strategy.stride_on_dim();
  int64 hlo_dim = inst_strategy.partition_dim();

  const Window& window = inst->window();
  if (ShapeUtil::IsScalar(op->shape())) return std::make_shared<DimStrategy>();

  if (window.dimensions(hlo_dim).size() != 1 ||
      window.dimensions(hlo_dim).stride() != 1)  {
    return std::make_shared<DimStrategy>();
  }

  return std::make_shared<DimStrategy>(op->shape(), hlo_dim, inst_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::BackInferSelectAndScatter(const HloInstruction* inst,
                                                    const DimStrategy& inst_strategy,
                                                    int input_idx) {
  const Window& window = inst->window();
  int64 partition_dim = inst_strategy.partition_dim();

  if (input_idx == 2 ||
      window.dimensions(partition_dim).size() != 1 ||
      window.dimensions(partition_dim).stride() != 1) {
    return std::make_shared<DimStrategy>();
  }

  int64 lo = inst_strategy.stride_on_dim();
  auto* op = inst->operand(input_idx);
  return std::make_shared<DimStrategy>(op->shape(), partition_dim, inst_strategy.num_replicas(), lo);
}

SharedDimStrategy StrategyUtil::BackInferSort(const HloInstruction* inst,
                                        const DimStrategy& inst_strategy,
                                        int input_idx) {
  auto& sort_dims = inst->dimensions();
  std::set<int> sort_dims_set(sort_dims.begin(), sort_dims.end());

  auto* op = inst->operand(input_idx);

  int64 lo = inst_strategy.stride_on_dim();
  int64 input_dim = inst_strategy.partition_dim();
  if (sort_dims_set.count(input_dim)) return std::make_shared<DimStrategy>();
  return std::make_shared<DimStrategy>(op->shape(), input_dim, inst_strategy.num_replicas(), lo);
}
 
HloInstMap<SharedDimStrategy> StrategyUtil::ForwardInfer(
    HloInstruction* inst, const DimStrategy& input_strategy,
    int input_idx) {
  CHECK(input_idx < inst->operand_count());

  HloInstMap<SharedDimStrategy> infer_res;
  std::shared_ptr<DimStrategy> dummy_strategy(new DimStrategy());
  auto* op = inst->mutable_operand(input_idx);
  if (input_strategy.Glue() || input_strategy.IsPartial()) {
    CHECK(!input_strategy.replicated())
        << "Manual annotated instruction should not be modified."
        << " inst : " << inst->ToString()
        << ", input strategy : " << input_strategy.ToString()
        << ", input_idx : " << input_idx;

    infer_res[inst] = dummy_strategy;
    return infer_res;
  }

  switch (inst->opcode()) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kIota:
    case HloOpcode::kRng: {
      break;
    }

    case HloOpcode::kBroadcast: {
      auto strategy = InferBcast(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kAbs:
    case HloOpcode::kAtan2:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh: {
      auto strategy = InferUnary(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kAnd:
    case HloOpcode::kAdd:
    case HloOpcode::kPower:
    case HloOpcode::kDivide:
    case HloOpcode::kCompare:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic: {
      auto strategy = InferBinary(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kSelect: {
      auto strategy = InferTernary(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kConcatenate: {
      auto strategy = InferConcat(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kSlice: {
      auto strategy = InferSlice(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kReshape:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert: {
      auto strategy = InferReshape(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kDot: {
      auto strategy_map = InferDot(inst, input_strategy, input_idx);
      infer_res.insert(strategy_map.begin(), strategy_map.end());
      break;
    }

    case HloOpcode::kConvolution: {
      auto strategy_map = InferConvolution(inst, input_strategy, input_idx);
      infer_res.insert(strategy_map.begin(), strategy_map.end());
      break;
    }

    case HloOpcode::kReduce: {
      auto strategy = InferReduce(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kReverse: {
      auto strategy = InferReverse(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kGather: {
      auto strategy = InferGather(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kPad: {
      auto strategy = InferPad(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kTranspose: {
      auto strategy = InferTranspose(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kCustomCall: {
      auto strategy = InferCustomCall(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kGetTupleElement: {
      auto strategy = InferGetTupleElement(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kSelectAndScatter: {
      auto strategy = InferSelectAndScatter(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kReduceWindow: {
      auto strategy = InferReduceWindow(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kSort: {
      auto strategy = InferSort(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    case HloOpcode::kScatter: {
      auto strategy_map = InferScatter(inst, input_strategy, input_idx);
      infer_res.insert(strategy_map.begin(), strategy_map.end());
      break;
    }

    case HloOpcode::kCustomCollective: {
      auto strategy = InferCustomCollective(inst, input_strategy, input_idx);
      infer_res[inst] = VerifyInfer(inst, strategy) ? strategy : dummy_strategy;
      break;
    }

    default: {
      VLOG(0) << "Unhandled instruction in ForwardInfer->"
              << inst->ToString() << "\n";
      CHECK(0);
    }
  }

  return infer_res;
}

std::vector<SharedDimStrategy> StrategyUtil::GenSplitProposals(
                                        const HloInstruction* inst,
                                        int split_count,
                                        bool save_variable_mem) {
  std::vector<SharedDimStrategy> res;

  switch (inst->opcode()) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kIota:
    case HloOpcode::kRng:
    case HloOpcode::kBroadcast:
    case HloOpcode::kAbs:
    case HloOpcode::kAtan2:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kAnd:
    case HloOpcode::kAdd:
    case HloOpcode::kPower:
    case HloOpcode::kDivide:
    case HloOpcode::kCompare:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kSelect:
    case HloOpcode::kConcatenate:
    case HloOpcode::kSlice:

    case HloOpcode::kReshape:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
      break;
    case HloOpcode::kDot: {
      res = GenDotProposals(inst, split_count, save_variable_mem);
      break;
    }
    case HloOpcode::kConvolution: {
      res = GenConvProposals(inst, split_count, save_variable_mem);
    }

    case HloOpcode::kReduce:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kTranspose:
      break;
    case HloOpcode::kCustomCall: {
      // TODO(lansong):
      //res = GenCustomCallProposals(inst, split_count);
      break;
    }

    case HloOpcode::kGetTupleElement:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSort:
    case HloOpcode::kScatter:
      break;
    default: {
      VLOG(0) << "Unhandled instruction in GenSplitProposals->"
              << inst->ToString();
      CHECK(0);
    }
  }

  return std::move(res);
}

std::vector<SharedDimStrategy>
StrategyUtil::GenDotProposals(const HloInstruction* inst, // kDot instruction
                              int split_count,
                              bool save_variable_mem) {
  std::vector<SharedDimStrategy> res;

  const DotDimensionNumbers &dim_nums = inst->dot_dimension_numbers();
  auto hlo_rank = inst->shape().rank();
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
           dim_nums.rhs_batch_dimensions_size());
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size() + 2, inst->shape().rank());
  int64 num_batches = dim_nums.lhs_batch_dimensions_size();
  int64 lhs_contracting_dim = dim_nums.lhs_contracting_dimensions(0);

  int64 split_dim;
  if (!save_variable_mem) {
    // 1. split on batch dimension
    split_dim = -1;
    for (int64 k=0; k<num_batches; ++k) {
      if (inst->shape().dimensions(k) % split_count == 0) {
        split_dim = k;
        break;
      }
    }
    VLOG(2) << "num_batches: " << num_batches;

    if (split_dim>=0) {
      res.emplace_back(std::make_shared<DimStrategy>(inst->shape(), split_dim, split_count, -1));
    }
  }

  // 2. split on contracting dimension
  if (inst->operand(0)->shape().dimensions(lhs_contracting_dim)
          % split_count == 0) {
    auto partial_str = DimStrategy::MakePartial(
                  split_count, lhs_contracting_dim,
                  inst->operand(0)->shape().dimensions(lhs_contracting_dim));
    res.emplace_back(partial_str);
  }

  // 3. split on row or column
  for (split_dim=num_batches; split_dim<inst->shape().rank(); ++split_dim) {
    if (inst->shape().dimensions(split_dim) % split_count == 0) {
      res.emplace_back(std::make_shared<DimStrategy>(inst->shape(), split_dim, split_count, -1));
    }
  }

  return std::move(res);
}

std::vector<SharedDimStrategy>
StrategyUtil::GenConvProposals(const HloInstruction* inst, // kDot instruction
                              int split_count,
                              bool save_variable_mem) {
  std::vector<SharedDimStrategy> res;
  const ConvolutionDimensionNumbers& dim_nums = inst->convolution_dimension_numbers();

  if (!save_variable_mem) {
    // 1. split on batch dimension
    int64 output_batch_dimension = dim_nums.output_batch_dimension();
    if (inst->shape().dimensions(output_batch_dimension) % split_count == 0) {
      res.emplace_back(
          std::make_shared<DimStrategy>(inst->shape(), output_batch_dimension, split_count, -1));
    }
  }

  // 2. split on input_feature_dimension (Partial)
  int64 input_feature_dimension = dim_nums.input_feature_dimension();
  if (inst->operand(0)->shape().dimensions(input_feature_dimension)
          % split_count == 0) {
    auto partial_str = DimStrategy::MakePartial(
                  split_count, input_feature_dimension,
                  inst->operand(0)->shape().dimensions(input_feature_dimension));
    res.emplace_back(partial_str);
  }

  // 3. split on output feature
  int64 output_feature_dimension = dim_nums.output_feature_dimension();
  if (inst->shape().dimensions(output_feature_dimension) % split_count == 0) {
    res.emplace_back(std::make_shared<DimStrategy>(
        inst->shape(), output_feature_dimension, split_count, -1));
  }

  return std::move(res);
}

bool StrategyUtil::VerifyInferGather(const HloInstruction* inst,
                                     const SharedDimStrategy inst_strategy) {
  VLOG(2) << "VerifyInferGather: " << inst->ToString() << " with " << inst_strategy->ToString();
  if (inst_strategy->Glue()) return true;
  for (int idx = 0; idx < inst->operand_count(); ++idx) {
    SharedDimStrategy op_strategy = BackInferGather(inst, *inst_strategy, idx);
    if (!op_strategy->Glue()) {
      VLOG(2) << "VerifyInferGather: index " << idx
              << " - " << inst->operand(idx)->ToString()
              << " --> got " << op_strategy->ToString();
      return true;
    }
  }
  return false;
}

bool StrategyUtil::VerifyInferScatter(const HloInstruction* inst,
                                      const SharedDimStrategy inst_strategy) {
  VLOG(2) << "VerifyInferScatter: " << inst->ToString()
          << " with " << inst_strategy->ToString();
  if (inst_strategy->Glue()) return true;
  for (int idx = 0; idx < inst->operand_count(); ++idx) {
    SharedDimStrategy op_strategy = BackInferScatter(inst, *inst_strategy, idx);
    if (!op_strategy->Glue()) {
      VLOG(2) << "VerifyInferGather: index " << idx
              << " - " << inst->operand(idx)->ToString()
              << " --> got " << op_strategy->ToString();
      return true;
    }
  }
  return false;
}

bool StrategyUtil::VerifyInferConvolution(const HloInstruction* inst,
                                          const SharedDimStrategy inst_strategy) {
  VLOG(2) << "VerifyInferConvolution: " << inst->ToString()
          << " with " << inst_strategy->ToString();
  if (inst_strategy->Glue()) return true;
  for (int idx = 0; idx < inst->operand_count(); ++idx) {
    SharedDimStrategy op_strategy = BackInferConvolution(inst, *inst_strategy, idx);
    if (!op_strategy->Glue()) {
      VLOG(2) << "VerifyInferConvolution: index " << idx
              << " - " << inst->operand(idx)->ToString()
              << " --> got " << op_strategy->ToString();
      return true;
    }
  }
  return false;
}

bool StrategyUtil::VerifyInfer(const HloInstruction* inst,
                               const SharedDimStrategy inst_strategy) {
  switch (inst->opcode()) {
    case HloOpcode::kGather:
      return VerifyInferGather(inst, inst_strategy);
    case HloOpcode::kScatter:
      return VerifyInferScatter(inst, inst_strategy);
    case HloOpcode::kConvolution:
      return VerifyInferConvolution(inst, inst_strategy);
    default:
      int dim = inst_strategy->partition_dim();
      if (dim < 0 || inst->shape().IsTuple() || inst_strategy->IsPartial()) {
        return true;
      }
      return (inst->shape().dimensions(dim) % inst_strategy->stride_on_dim() == 0) &&
             (inst->shape().dimensions(dim) % inst_strategy->NumSlices() == 0);
  }
}

SharedDimStrategy StrategyUtil::BackInferImpl(const HloInstruction* inst,
                                        const DimStrategy& inst_strategy,
                                        int input_idx) {
  if (inst_strategy.Glue() ||
      (inst_strategy.IsPartial() && !AllowPartialPassThrough(inst))) {
    return std::make_shared<DimStrategy>();
  }

  switch (inst->opcode()) {
    case HloOpcode::kBroadcast: {
      return BackInferBcast(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kScatter: {
      return BackInferScatter(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kAbs:
    case HloOpcode::kAtan2:
    case HloOpcode::kCeil:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCopy:
    case HloOpcode::kTanh: {
      return BackInferUnary(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kDAPPLEAllReduce: {
      return BackInferCrossReplicaSum(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kAnd:
    case HloOpcode::kAdd:
    case HloOpcode::kPower:
    case HloOpcode::kDivide:
    case HloOpcode::kCompare:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic: {
      return BackInferBinary(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kSelect: {
      return BackInferTernary(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kConcatenate: {
      return BackInferConcat(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kSlice: {
      return BackInferSlice(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kReshape:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert: {
      return BackInferReshape(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kDot: {
      return BackInferDot(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kConvolution: {
      return BackInferConvolution(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kReduce: {
      return BackInferReduce(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kReverse: {
      return BackInferReverse(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kGather: {
      return BackInferGather(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kPad: {
      return BackInferPad(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kTranspose: {
      return BackInferTranspose(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kIota: {
      if (inst->metadata().op_type() == "OneHot") {
        // In OneHot op, splitting on iota_dimension is not allowed for
        // maintaining its SPMD semantics.
        return BackInferIota(inst, inst_strategy, input_idx);
      }
      // In general, splitting on iota_dimension destroys SPMD semantics.
      // However, the following cases are execeptional.
      // Iota in GatherV2 is used to generate batch gather dimension.
      // In this case, spliting on iota_dimension is allowed because the
      // operand of gather is also split on batch dimension, which the
      // index range for each replica is also changed.
      // For example:
      //    %iota = [[0,0,0,0], [1,1,1,1]]
      //    %index = [[0,2,1,3],[1,4,1,3]]
      // And we concate them togather
      //    %concatenate = concatenate(%iota, %index, dimmension=0)
      // Thus the %concatenate = [[(0,0),(0,2),(0,1),(0,3)],
      //                          [(1,1),(1,4),(1,1),(1,3)]]
      // Then, we split on 1st dim of %concatenate, we expect to get
      // re-index the site for each index in %concatenate.
      //    1st replica is : %concatenate[(0,0),(0,2),(0,1),(0,3)]
      //    2st replica is : %concatenate[(0,1),(0,4),(0,1),(0,3)]
      // Thus, the splitting on iota_dimension is an expected behaviour.
      // The corresponding backward instruction is also the same.
      break;
    }

    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kRng: {
      break;
    }

    case HloOpcode::kCustomCall: {
      return BackInferCustomCall(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kGetTupleElement: {
      return BackInferGetTupleElement(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kSelectAndScatter: {
      return BackInferSelectAndScatter(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kReduceWindow: {
      return BackInferReduceWindow(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kSort: {
      return BackInferSort(inst, inst_strategy, input_idx);
    }

    case HloOpcode::kCustomCollective: {
      return BackInferCustomCollective(inst, inst_strategy, input_idx);
    }
    
    default: {
      VLOG(0) << "Unhandled instruction in BackInference->"
              << inst->ToString();
      CHECK(0);
    }
  }
  return std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::BackInfer(const HloInstruction* inst,
                                    const DimStrategy& inst_strategy,
                                    int input_idx) {
  SharedDimStrategy infered_strategy = BackInferImpl(inst, inst_strategy, input_idx);
  auto* op = inst->operand(input_idx);
  return VerifyInfer(op, infered_strategy) ? infered_strategy : std::make_shared<DimStrategy>();
}

SharedDimStrategy StrategyUtil::NonVerifyBackInfer(const HloInstruction* inst,
                                             const DimStrategy& inst_strategy,
                                             int input_idx) {
  return BackInferImpl(inst, inst_strategy, input_idx);
}

bool StrategyUtil::ForwardInfer(
    HloComputation* entry, HloInstMap<DimStrategy>& strategy_map,
    bool* changed) {
  *changed = false;
  for (HloInstruction* inst : entry->MakeInstructionPostOrder()) {
    if (inst == entry->root_instruction()) continue;
    for (int op_idx = 0; op_idx < inst->operand_count(); ++op_idx) {
      HloInstruction* op = inst->mutable_operand(op_idx);
      if (strategy_map.find(op) == strategy_map.end() || strategy_map[op].Glue()) continue;
      const DimStrategy& input_strategy = strategy_map[op];
      HloInstMap<SharedDimStrategy>
      res_map = StrategyUtil::ForwardInfer(inst, input_strategy, op_idx);

      for (auto& iter : res_map) {
        if (iter.second->Glue()) continue;
        if (strategy_map.find(iter.first) == strategy_map.end() || strategy_map[iter.first].Glue()) {
	        strategy_map[iter.first] = *iter.second;
	        *changed = true;
	      } else {
	        if (!strategy_map[iter.first].Match(*res_map[iter.first]))  {
	          VLOG(1) << "[Conflict] " << iter.first->ToString()
	                  << ", split at " << strategy_map[iter.first].ToString()
		                << ", and " << res_map[iter.first]->ToString()
		                << ", input_strategy " << input_strategy.ToString();
	          return false;
	        }
	      }
      }
    }
  }

  return true;
}

bool StrategyUtil::BackwardInfer(
    HloComputation* entry, HloInstMap<DimStrategy>& strategy_map,
    bool* changed) {
  *changed = false;
  std::vector<HloInstruction*> instructions = entry->MakeInstructionPostOrder();
  for (std::vector<HloInstruction*>::reverse_iterator r_iter = instructions.rbegin();
       r_iter != instructions.rend(); ++r_iter) {
    HloInstruction* inst = *r_iter;
    if (inst == entry->root_instruction() ||
        strategy_map.find(inst) == strategy_map.end() ||
        strategy_map[inst].Glue()) continue;
    DimStrategy inst_strategy = strategy_map[inst];
    for (int op_idx = 0; op_idx < inst->operand_count(); ++op_idx) {
      HloInstruction* op = inst->mutable_operand(op_idx);
      if (op->opcode() == HloOpcode::kConstant) continue;
      SharedDimStrategy input_strategy =
                          StrategyUtil::BackInfer(inst, inst_strategy, op_idx);
      if (input_strategy->Glue()) continue;
      if (strategy_map.find(op) == strategy_map.end() || strategy_map[op].Glue()) {
	      strategy_map[op] = *input_strategy;
	      *changed = true;
      } else {
        if (!strategy_map[op].Match(*input_strategy)) {
          VLOG(1) << "[Conflict] " << op->ToString()
	                << ", split at " << strategy_map[op].ToString()
		              << ", and " << input_strategy->ToString();
          return false;
	      }
      }
    }
  }

  return true;
}

bool StrategyUtil::InferGraph(
    HloComputation* entry,
    HloInstMap<DimStrategy>& strategy_map) {
  bool ever_changed = false;
  bool changed = false;
  do {
    ever_changed = false;
    do {
      if (!StrategyUtil::ForwardInfer(entry, strategy_map, &changed)) return false;
      ever_changed |= changed;
    } while (changed);

    do {
      if (!StrategyUtil::BackwardInfer(entry, strategy_map, &changed)) return false;
      ever_changed |= changed;
    } while (changed);
  } while (ever_changed);

  // Fill remains with glue strategy
  for (auto* instr : entry->MakeInstructionPostOrder()) {
    if (instr == entry->root_instruction()) continue;
    if (strategy_map.find(instr) == strategy_map.end()) {
      strategy_map[instr] = DimStrategy();
    }
  }

  // Post check
  for (auto* instr : entry->MakeInstructionPostOrder()) {
    if (instr == entry->root_instruction()) continue;
    for (auto* op : instr->operands()) {
      if (strategy_map[instr].Glue() &&
          !strategy_map[op].Glue() && !strategy_map[op].IsPartial()) {
        VLOG(1) << "[CHECK] instr : " << instr->ToString()
                << "\nop : " << op->ToString()
                << "\nop_strategy : " << strategy_map[op].ToString();
        return false;
      }
    }
  }

  return true;
}

int StrategyUtil::CountSplitInsts(HloInstMap<DimStrategy>& strategy_map) {
  int count = 0;
  for (auto& iter : strategy_map) {
    if (!iter.second.Glue()) ++count;
  }

  return count;
}

bool StrategyUtil::CompatibleWith(
    HloInstMap<DimStrategy>& s_map, HloInstMap<DimStrategy>& t_map) {
  bool compatible = true;
  for (auto& iter : s_map) {
    if (t_map.find(iter.first) == t_map.end() && !iter.second.Glue() ||
        t_map.find(iter.first) != t_map.end() && !t_map[iter.first].Match(iter.second)) {
      compatible = false;
      break;
    }
  }

  return compatible;
}

std::unordered_set<const HloInstruction*> StrategyUtil::CollectAllSyncPoints(
    HloModule* module) {
  std::unordered_set<const HloInstruction*> sync_points;
  HloComputation* entry = module->entry_computation();
  const std::vector<bool>& share_dev_flags = module->share_dev_flags();
  for (HloInstruction* inst : entry->instructions()) {
    if (inst->opcode() != HloOpcode::kCustomCollective) continue;
    auto* coll = DynCast<HloCustomCollectiveInstruction>(inst);
    int split_ordinal = coll->split_ordinal();
    if (coll->collective_type() == "AllReduce" && share_dev_flags[split_ordinal]) {
      sync_points.insert(coll);
    }
  }

  VLOG(1) << "[CollectAllSyncPoints] sync_points count = " << sync_points.size();
  for (const HloInstruction* sync : sync_points) {
    VLOG(1) << "\t Sync on " << sync->ToString();
  }
  return sync_points;
}

Shape DistUtil::MakeShape(const Shape& old_shape, std::vector<int64>& dims) {
  absl::Span<const int64> new_dims(dims);
  switch(old_shape.element_type()) {
    case PRED: // bool
    case S8:   // int8
    case S16:  // int16
    case S32:  // int32
    case S64:  // int64
    case U8:   // unsigned int8
    case U16:  // unsigned int16
    case U32:  // unsigned int32
    case F16:  // float
    case F32:  // float
    case F64:  // float
    case U64:  // unsigned int64
      return ShapeUtil::MakeShapeWithLayout(old_shape.element_type(), new_dims,
          LayoutUtil::MinorToMajor(old_shape));
    default: CHECK(0 && "Unsupported type.");
  }
}

Shape DistUtil::MakeNewPrimShape(const Shape& old_shape,
                                 const DimDistSpec& dist_spec) {
  // create non-tuple shape
  CHECK(!old_shape.IsTuple());

  int rank = old_shape.rank();

  std::vector<int64> new_dims;
  int dim_to_slice = dist_spec.partition_dim();
  CHECK(dim_to_slice >= 0 && dim_to_slice < rank);
  int num_replicas = dist_spec.num_splits();
  int64 old_dim_size = old_shape.dimensions(dim_to_slice);
  int64 new_dim_size = old_dim_size / num_replicas;
  new_dims.reserve(rank);
  for (int r = 0; r < rank; ++r) {
    if (r == dim_to_slice && new_dim_size != -1) {
      new_dims.emplace_back(new_dim_size);
    } else {
      new_dims.emplace_back(old_shape.dimensions(r));
    }
  }

  return MakeShape(old_shape, new_dims);
}

Shape DistUtil::MakeNewShape(const Shape& old_shape,
                             const DimDistSpec& dist_spec) {
  if (old_shape.IsTuple()) {
    std::vector<Shape> new_sub_shapes;
    for (auto& old_sub_shape : old_shape.tuple_shapes()) {
      new_sub_shapes.emplace_back(MakeNewShape(old_sub_shape, dist_spec));
    }
    return ShapeUtil::MakeTupleShape(new_sub_shapes);
  } else if (dist_spec.partial() || dist_spec.num_splits() <= 1) {
    return Shape(old_shape);
  } else {
    return MakeNewPrimShape(old_shape, dist_spec);
  }
}

DimDistSpec DistUtil::MakeDimDistSpec(const DimStrategy& s) {
  DimDistSpec spec;
  spec.set_partial(s.IsPartial());
  spec.set_layout_aware_partition(
      s.stride_on_elements(), s.stride_on_dim(), s.partition_dim(), s.NumSlices());
  return spec;
}

Shape DistUtil::MakeNewShape(const Shape& old_shape, const DimStrategy& s) {
  return MakeNewShape(old_shape, MakeDimDistSpec(s));
}

} // namespace xla
