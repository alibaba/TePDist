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

#include "tensorflow/compiler/xla/service/parallel/fast_spmd_strategy.h"
#include "tensorflow/compiler/xla/service/parallel/resolve_utils.h"
#include "tensorflow/compiler/xla/service/parallel/utils.h"
#include "tensorflow/core/util/env_var.h"

namespace xla {

namespace {

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

};

FastSpmdStrategyBase::FastSpmdStrategyBase() {}

FastSpmdStrategyBase::FastSpmdStrategyBase(/*ParallelType ttype,*/ bool post_layout) 
    : /*ttype_(ttype),*/ post_layout_assignment_(post_layout) {}

FastSpmdStrategyBase::FastSpmdStrategyBase(/*ParallelType ttype,*/ bool post_layout,
                          int num_replicas)
    : /*ttype_(ttype),*/ post_layout_assignment_(post_layout),
      num_replicas_(num_replicas) {}

bool FastSpmdStrategyBase::ConstantInference(HloInstruction* hlo, Solution* solution) {
  // Do not split kConstant
  solution->hlo_strategy_map_[hlo] = DimStrategy(hlo->shape(), -1, num_replicas_);
  return true;
}

bool FastSpmdStrategyBase::BackInferScatter(HloInstruction* hlo, Solution* solution) {
  auto& hlo_shape = hlo->shape();
  int64 hlo_rank = hlo_shape.rank();
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 hlo_stride = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();

  auto init = hlo->mutable_operand(0);
  auto& init_strategy = solution->hlo_strategy_map_.at(init);
  if (init_strategy.Glue()) {
    solution->hlo_strategy_map_[init] = hlo_strategy;
    if (!BackInference(init, solution)) return false;
  }

  auto indices = hlo->mutable_operand(1);
  //auto& indices_shape = indices->shape();
  //auto indices_rank = indices_shape.rank();

  auto updates = hlo->mutable_operand(2);
  auto& updates_shape = updates->shape();
  auto updates_rank = updates_shape.rank();
  auto& updates_strategy = solution->hlo_strategy_map_.at(updates);

  auto& scatter_dims = hlo->scatter_dimension_numbers(); 
  auto& update_window_dims = scatter_dims.update_window_dims();
  int64 update_window_size = update_window_dims.size();

  //CHECK(updates_rank == update_window_size + indices_rank -1);
  //CHECK(indices_shape.dimensions(indices_rank-1) == 1); // trailing dimension

  if (hlo_dim < hlo_rank - update_window_size) {
    // No back inference is needed
    return true;
  }

  auto distance = hlo_rank - hlo_dim;
  auto updates_dim = updates_rank - distance;
  CHECK(updates_dim >= 0);
  // Bounds check
  if (hlo_shape.dimensions(hlo_dim) != 
      updates_shape.dimensions(updates_dim)) {
    return false;
  }

  if (!updates_strategy.Glue()) {
    int64 stride = updates_strategy.stride_on_dim();
    int64 dim = updates_strategy.partition_dim();
    return (dim == updates_dim && stride == hlo_stride);
  }

  solution->hlo_strategy_map_[updates] = DimStrategy(
      updates_shape, updates_dim, num_replicas_, hlo_stride);
  return true;
}

bool FastSpmdStrategyBase::BackInferTranspose(HloInstruction* hlo, Solution* solution) {
  auto& hlo_shape = hlo->shape();
  int64 hlo_rank = hlo_shape.rank();
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 hlo_stride = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();
  CHECK(hlo_dim >= 0);

  auto input = hlo->mutable_operand(0);
  auto& input_shape = input->shape();
  int64 input_rank = input_shape.rank();
  CHECK(input_rank == hlo_rank);
  auto input_dim = hlo->dimensions(hlo_dim);
  CHECK(input_dim < input_rank);

  if (solution->hlo_strategy_map_.count(input)) {
    auto& input_strategy = solution->hlo_strategy_map_.at(input);
    if (!input_strategy.Glue()) {
      int64 stride = input_strategy.stride_on_dim();
      int64 dim = input_strategy.partition_dim();
      CHECK(dim == input_dim && stride == hlo_stride);
      return true;
    }
  }

  solution->hlo_strategy_map_[input] = DimStrategy(
      input_shape, input_dim, num_replicas_, hlo_stride);
  return true;
}

bool FastSpmdStrategyBase::BackInferIota(HloInstruction* hlo, Solution* solution) {
  if (solution->hlo_strategy_map_.count(hlo)) {
    auto& iota_strategy = solution->hlo_strategy_map_.at(hlo);
    if (!iota_strategy.Glue()) {
      const HloIotaInstruction* iota = DynCast<HloIotaInstruction>(hlo);
      int64 iota_dimension = iota->iota_dimension();
      auto dim = iota_strategy.partition_dim();
      return dim != iota_dimension;
    }
  }
  return true;
}

bool FastSpmdStrategyBase::BackInferGather(HloInstruction* hlo, Solution* solution) {
  auto& hlo_shape = hlo->shape();
  int64 hlo_rank = hlo_shape.rank();
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 hlo_stride = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();

  auto data = hlo->mutable_operand(0);
  auto& data_shape = data->shape();
  int64 data_rank = data_shape.rank();
  auto& data_strategy = solution->hlo_strategy_map_.at(data);

  auto indices = hlo->mutable_operand(1);
  auto& indices_shape = indices->shape();
  //int64 index_rank = indices_shape.rank();
  auto& indices_strategy = solution->hlo_strategy_map_.at(indices);

  const HloGatherInstruction* gather = DynCast<HloGatherInstruction>(hlo);
  auto& gather_dims = gather->gather_dimension_numbers();
  auto& offset_dims = gather_dims.offset_dims();
  auto slice_sizes = gather->gather_slice_sizes();
  auto& collapsed_slice_dims = gather_dims.collapsed_slice_dims();
  int64 num_collapsed = collapsed_slice_dims.size();
  std::set<int> collapsed_dims_set(collapsed_slice_dims.begin(), 
                                   collapsed_slice_dims.end());
  for (auto collapsed : collapsed_slice_dims) {
    CHECK(slice_sizes[collapsed] == 1);
  }

  int64 offset_left = offset_dims[0];

  int64 num_offsets = hlo_rank - offset_left;
  //auto num_offsets_uncollapsed = num_offsets + num_collapsed;

  if (hlo_dim < offset_left) {
    // gather -> index
    if (!indices_strategy.Glue()) {
      int64 stride = indices_strategy.stride_on_dim();
      int64 dim = indices_strategy.partition_dim();
      return (dim == hlo_dim && stride == hlo_stride);
    } else {
      if (indices->opcode() == HloOpcode::kConstant) {
        // Never slice kContant instruction in TePDist.
        // Later in CustomCollectiveSpec::CreateSuitableReshardSpec,
        // a kDynamicSlice instruction will be inserted between kConstant
        // and kGather, the pattern is as follows:
        //    %constant(replicated)->%dynamicSlice
        //                                        \
        //                       data(replicated)->%gather(sliced)
        return true;
      } else {
        solution->hlo_strategy_map_[indices] = DimStrategy(
            indices_shape, hlo_dim, num_replicas_, hlo_stride);
        if (!BackInference(indices, solution)) return false;
      }
    }
  } else {
    // gather -> data
    auto hlo_slice_dim = hlo_dim - offset_left;
    int64 data_dim = data_rank - num_offsets + hlo_slice_dim;
    if (slice_sizes[num_collapsed + hlo_slice_dim] != 
        data_shape.dimensions(data_dim)) {
      return false;
    }

    if (!data_strategy.Glue()) {
      int64 stride = data_strategy.stride_on_dim();
      int64 dim = data_strategy.partition_dim();
      return (dim == data_dim && stride == hlo_stride);
    } else {
      solution->hlo_strategy_map_[data] = DimStrategy(
          data_shape, data_dim, num_replicas_, hlo_stride);
      if (!BackInference(data, solution)) return false;
    }
  }
  return true;
}

bool FastSpmdStrategyBase::BackInferReduce(HloInstruction* hlo, Solution* solution) {
  // NOTE: output of reduce instruction may be a tuple, the strategy of reduce
  //       is the strategy of the tuple element.
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 hlo_stride = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();
  CHECK(hlo_stride > 0);

  //VLOG(0) << "reduce strategy: stride_on_elements: " << hlo_strategy.ToSting();

  for (int i=0; i<hlo->operand_count(); ++i) {
    auto input = hlo->mutable_operand(i);
    auto& input_shape = input->shape();
    if (ShapeUtil::IsScalar(input_shape))
      continue;

    auto& input_strategy = solution->hlo_strategy_map_.at(input);

    auto& reduce_dims = hlo->dimensions();
    std::set<int> reduce_dims_set(reduce_dims.begin(), reduce_dims.end());
    int64 input_dim = 0;
    int64 output_dim = 0;
    for (int64 r = 0; r < input_shape.rank(); ++r) {
      if (reduce_dims_set.count(r)) continue;
      if (output_dim == hlo_dim) {
        input_dim = r;
        break;
      }

      ++output_dim;
    }

    if (input_strategy.Glue()) {
      solution->hlo_strategy_map_[input] = DimStrategy(
          input_shape, input_dim, num_replicas_, hlo_stride);
      if (!BackInference(input, solution)) return false;
    } else {
      int64 stride = input_strategy.stride_on_dim();
      int64 dim = input_strategy.partition_dim();
      if (dim != input_dim || stride != hlo_stride) {
        return false;
      }
    }
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferDot(HloInstruction* hlo, Solution* solution) {
  auto lhs = hlo->mutable_operand(0);
  auto& lhs_shape = lhs->shape();
  int64 lhs_rank = lhs_shape.rank();
  auto& lhs_strategy = solution->hlo_strategy_map_.at(lhs);
  int64 lhs_lo = lhs_strategy.stride_on_dim();
  int64 lhs_dim = -1;
  if (!lhs_strategy.Glue()) {
    lhs_dim = lhs_strategy.partition_dim();
  }

  auto rhs = hlo->mutable_operand(1);
  auto& rhs_shape = rhs->shape();
  int64 rhs_rank = rhs_shape.rank();
  auto& rhs_strategy = solution->hlo_strategy_map_.at(rhs);
  int64 rhs_lo = rhs_strategy.stride_on_dim();
  int64 rhs_dim = -1;
  if (!rhs_strategy.Glue()) {
    rhs_dim = rhs_strategy.partition_dim();
  }

  auto& hlo_shape = hlo->shape();
  int64 hlo_rank = hlo_shape.rank();
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 hlo_lo = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();

  const DotDimensionNumbers &dim_nums = hlo->dot_dimension_numbers();
  int64 num_batches = dim_nums.lhs_batch_dimensions_size();
  if (hlo_dim < num_batches) {
    if (lhs_strategy.Glue()) {
      solution->hlo_strategy_map_[lhs] = DimStrategy(
          lhs_shape, hlo_dim, num_replicas_, hlo_lo);
      if (!BackInference(lhs, solution)) return false;
    } else if (hlo_dim != lhs_dim || hlo_lo != lhs_lo) {
      return false;
    }

    if (rhs_strategy.Glue()) {
      solution->hlo_strategy_map_[rhs] = DimStrategy(
          rhs_shape, hlo_dim, num_replicas_, hlo_lo);
      if (!BackInference(rhs, solution)) return false;
    } else if (hlo_dim != rhs_dim || hlo_lo != rhs_lo) {
      return false;
    }
  }

  // LHS case
  if (hlo_dim == hlo_rank - 2) {
    int64 lhs_contracting_dim = dim_nums.lhs_contracting_dimensions(0);
    int64 back_infer_lhs_dim = -1;
    if (lhs_contracting_dim == lhs_rank-1) {
      back_infer_lhs_dim = lhs_rank-2;
    } else {
      back_infer_lhs_dim = lhs_rank-1;
    }

    // Check if conflict exists.
    if (back_infer_lhs_dim != lhs_dim) {
      VLOG(0) << "BackInferDot Conflict!"
              << " back_infer_lhs_dim: " << back_infer_lhs_dim
              << " lhs_dim: " << lhs_dim;
      return false;
    }

    solution->hlo_strategy_map_[lhs] = DimStrategy(
        lhs_shape, lhs_dim, num_replicas_, hlo_lo);
    if (!BackInference(lhs, solution)) return false;
  }

  // RHS case
  if (hlo_dim == hlo_rank - 1) {
    int64 rhs_contracting_dim = dim_nums.rhs_contracting_dimensions(0);

    int64 back_infer_rhs_dim = -1;
    if (rhs_contracting_dim == rhs_rank-1) {
      back_infer_rhs_dim = rhs_rank-2;
    } else {
      back_infer_rhs_dim = rhs_rank-1;
    }

    // Check if conflict exists.
    if (back_infer_rhs_dim != rhs_dim) {
      VLOG(0) << "BackInferDot Conflict!"
              << " back_infer_rhs_dim: " << back_infer_rhs_dim
              << " rhs_dim: " << rhs_dim;
      return false;
    }
    solution->hlo_strategy_map_[rhs] = DimStrategy(
        rhs_shape, rhs_dim, num_replicas_, hlo_lo);
    if (!BackInference(rhs, solution)) return false;
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferReshape(HloInstruction* hlo, Solution* solution) {
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);

  // Backinfer lhs
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_[input] = hlo_strategy;
    if (!BackInference(input, solution)) return false;
  } else if (!hlo_strategy.Match(input_strategy)) {
    return false;
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferSlice(HloInstruction* hlo, Solution* solution) {
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 hlo_stride = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();
  CHECK(hlo_stride > 0);

  auto input = hlo->mutable_operand(0);
  auto& input_shape = input->shape();
  auto& input_strategy = solution->hlo_strategy_map_.at(input);

  const HloSliceInstruction* slice = DynCast<HloSliceInstruction>(hlo);
  const std::vector<int64> slice_starts = slice->slice_starts();
  const std::vector<int64> slice_limits = slice->slice_limits();
  const std::vector<int64> slice_strides = slice->slice_strides();

  int64 slice_size =
      (slice_limits[hlo_dim] - slice_starts[hlo_dim]) / slice_strides[hlo_dim];
  if (slice_size == input_shape.dimensions(hlo_dim)) {
    if (input_strategy.Glue()) {
      solution->hlo_strategy_map_[input] = DimStrategy(
          input_shape, hlo_dim, num_replicas_, hlo_stride);
      if (!BackInference(input, solution)) return false;
    } else {
      int64 input_stride = input_strategy.stride_on_dim();
      int64 input_dim = input_strategy.partition_dim();
      if (input_dim != hlo_dim || input_stride != hlo_stride) {
        return false;
      }
    }
  } else {
    if (input_shape.dimensions(hlo_dim) % hlo_strategy.stride_on_elements()) {
      return false;
    } else {
      if (input_strategy.Glue()) {
        solution->hlo_strategy_map_[input] = DimStrategy(
            input_shape, hlo_dim, num_replicas_, hlo_stride);
        if (!BackInference(input, solution)) return false;
      } else {
        auto input_dim = input_strategy.partition_dim();
        if (input_dim != hlo_dim) {
          return false;
        }
      }
    }
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferConcat(HloInstruction* hlo, Solution* solution) {
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 stride_key = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();
  int64 hlo_stride = hlo_strategy.stride_on_elements();

  int64 concat_dim = hlo->dimensions(0);

  int64 operand_count = hlo->operand_count();

  if (hlo_dim == concat_dim) {
    // if (hlo_stride % operand_count) return false;
    int64 operand_stride = -1;
    if (IdenticalConcat(hlo)) {
      operand_stride = hlo_stride / operand_count;
    }

    // if (!IdenticalConcat(hlo)) return false;

    for (int64 i = 0; i < operand_count; ++i) {
      auto operand = hlo->mutable_operand(i);
      auto& operand_shape = operand->shape();
      auto& operand_strategy = solution->hlo_strategy_map_.at(operand);
      if (operand_strategy.Glue()) {
        solution->hlo_strategy_map_[operand] = DimStrategy(
            operand_shape, hlo_dim, num_replicas_, operand_stride);
        if (!BackInference(operand, solution)) return false;
      } else {
        int64 operand_dim = operand_strategy.partition_dim();
        // if (operand_dim != hlo_dim ||
            // operand_strategy.stride_on_elements() != operand_stride) {
        if (operand_dim != hlo_dim) {
          return false;
        }
      }
    }
  } else {
    for (int64 i = 0; i < operand_count; ++i) {
      auto operand = hlo->mutable_operand(i);
      auto& operand_shape = operand->shape();
      auto& operand_strategy = solution->hlo_strategy_map_.at(operand);
      CHECK(stride_key > 0);
      if (operand_strategy.Glue()) {
        solution->hlo_strategy_map_[operand] = DimStrategy(
            operand_shape, hlo_dim, num_replicas_, stride_key);
        if (!BackInference(operand, solution)) return false;
      } else {
        int64 operand_dim = operand_strategy.partition_dim();
        if (operand_dim != hlo_dim) return false;
      }
    } // for
  } // else

  return true;
}

bool FastSpmdStrategyBase::BackInferSelect(HloInstruction* hlo, Solution* solution) {
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);

  auto pred = hlo->mutable_operand(0);
  auto& pred_strategy = solution->hlo_strategy_map_.at(pred);

  auto lhs = hlo->mutable_operand(1);
  auto& lhs_strategy = solution->hlo_strategy_map_.at(lhs);

  auto rhs = hlo->mutable_operand(2);
  auto& rhs_strategy = solution->hlo_strategy_map_.at(rhs);
 
  // Backinfer lhs
  if (lhs_strategy.Glue()) {
    solution->hlo_strategy_map_[lhs] = hlo_strategy;
    if (!BackInference(lhs, solution)) return false;
  } else if (!hlo_strategy.Match(lhs_strategy)) {
    return false;
  }

  // Backinfer rhs
  if (rhs_strategy.Glue()) {
    solution->hlo_strategy_map_[rhs] = hlo_strategy;
    if (!BackInference(rhs, solution)) return false;
  } else if (!hlo_strategy.Match(rhs_strategy)) {
    return false;
  }

  // Backinfer pred
  if (pred_strategy.Glue()) {
    solution->hlo_strategy_map_[pred] = hlo_strategy;
    if (!BackInference(pred, solution)) return false;
  } else if (!hlo_strategy.Match(pred_strategy)) {
    return false;
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferCrossReplicaSum(
    HloInstruction* hlo, Solution* solution) {
  auto lhs = hlo->mutable_operand(0);
  auto& lhs_strategy = solution->hlo_strategy_map_.at(lhs);
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);

  decltype(lhs) rhs = nullptr;
  if (2 == hlo->operand_count()) {
    rhs = hlo->mutable_operand(1);
    CHECK(rhs->opcode() == HloOpcode::kParameter);
    auto& rhs_strategy = solution->hlo_strategy_map_.at(rhs);
    if (!rhs_strategy.Match(lhs_strategy)) {
      CHECK(hlo_strategy.Match(lhs_strategy));
      CHECK(rhs_strategy.Glue());
      solution->hlo_strategy_map_[rhs] = hlo_strategy;
      return true;
    }
  }

  // Backinfer input
  if (lhs_strategy.Glue()) {
    solution->hlo_strategy_map_[lhs] = hlo_strategy;
    if (rhs) solution->hlo_strategy_map_[rhs] = hlo_strategy;
    if (!BackInference(lhs, solution)) return false;
  } else if (!hlo_strategy.Match(lhs_strategy)) {
    return false;
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferUnary(HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);

  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);

  // Backinfer input
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_[input] = hlo_strategy;
    if (!BackInference(input, solution)) return false;
  } else if (!hlo_strategy.Match(input_strategy)) {
    return false;
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferBinary(HloInstruction* hlo, Solution* solution) {
  auto lhs = hlo->mutable_operand(0);
  auto& lhs_strategy = solution->hlo_strategy_map_.at(lhs);

  auto rhs = hlo->mutable_operand(1);
  auto& rhs_strategy = solution->hlo_strategy_map_.at(rhs);

  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);

  // Backinfer lhs
  if (lhs_strategy.Glue()) {
    solution->hlo_strategy_map_[lhs] = hlo_strategy;
    if (!BackInference(lhs, solution)) return false;
  } else if (!hlo_strategy.Match(lhs_strategy)) {
    return false;
  }

  // Backinfer rhs
  if (rhs_strategy.Glue()) {
    solution->hlo_strategy_map_[rhs] = hlo_strategy;
    if (!BackInference(rhs, solution)) return false;
  } else if (!hlo_strategy.Match(rhs_strategy)) {
    return false;
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferBcast(HloInstruction* hlo, Solution* solution) {
  VLOG(2) << "BackInferBcast->" << hlo->ToString();
  auto input = hlo->mutable_operand(0);
  auto& input_shape = input->shape();
  // Most common case
  if (ShapeUtil::IsScalar(input_shape)) {
    solution->hlo_strategy_map_.insert(std::make_pair(input, DimStrategy()));

    // NOTE: Prevent kBroadcast from polluting the mainstream inference path.
    // Example:
    // y[512, 512] = broadcast(x[])
    // w[512, 512] = add(u[512, 512], y[512, 512])
    // ...
    // z[512, 512] = mul(v[512, 512], y[512, 512])
    // Then:
    // w -> {stride = 512x512, size = ?}
    // z -> {stride = 512, size = ?}
    // w and z collide at y!
    // We can easily resolve this by *renaming* broadcast in each occurance.
    solution->hlo_strategy_map_[hlo] = DimStrategy();

    return true;
  }

  auto& input_strategy = solution->hlo_strategy_map_.at(input);

  auto& hlo_shape = hlo->shape();
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 stride_key = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();
  CHECK(hlo_dim >= 0);

  auto& bcast_dims = hlo->dimensions(); 
  std::set<int> bcast_dims_set(bcast_dims.begin(), bcast_dims.end());
  // If it not happened on bcast_dims, the input should have no solution.
  if (!bcast_dims_set.count(hlo_dim)) {
    return input_strategy.Glue();
  }

  CHECK(input_shape.dimensions().size() == bcast_dims.size());
  int64 input_dim = -1;
  for (int64 r = 0; r < input_shape.rank(); ++r) {
    if (hlo_dim == bcast_dims[r]) {
      CHECK(input_dim == -1);
      input_dim = r;
    }
  }

  CHECK(stride_key > 0 && stride_key <= hlo_shape.dimensions(hlo_dim));
  DimStrategy expect_strategy(
      input_shape, input_dim, num_replicas_, stride_key);
  if (!input_strategy.Glue()) {
    return input_strategy.Match(expect_strategy);
  }
  solution->hlo_strategy_map_[input] = expect_strategy;
  return true;
}

bool FastSpmdStrategyBase::BackInferPad(HloInstruction* hlo, Solution* solution) {
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);

  int64 hlo_dim = hlo_strategy.partition_dim();
  DCHECK(hlo_dim >= 0);

  const HloPadInstruction* pad = DynCast<HloPadInstruction>(hlo);
  DCHECK(pad);

  auto config = pad->padding_config();
  auto config_dims_size = config.dimensions_size();
  DCHECK(hlo_dim >= 0 && hlo_dim < config_dims_size);
  auto paddings = config.dimensions(hlo_dim);
  if (paddings.edge_padding_low() != 0 || paddings.edge_padding_high() != 0 ||
      paddings.interior_padding() != 0) {
    return false;
  }

  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);

  // Backinfer input
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_[input] = DimStrategy(
        input->shape(), hlo_dim, num_replicas_);
    if (!BackInference(input, solution)) return false;
  } else {
    if (input_strategy.partition_dim() != hlo_dim) {
      return false;
    }
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferCustomCall(HloInstruction* hlo, Solution* solution) {
  if (IsConvolution(*hlo)) {
    return BackInferConv(hlo, solution);
  }

  return false;
}

bool FastSpmdStrategyBase::BackInferFilter(HloInstruction* hlo, Solution* solution) {
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);

  const Shape* used_hlo_shape;
  auto& hlo_shape = hlo->shape();
  if (hlo_shape.IsTuple()) {
    // TODO: We assume index == 0. It may not be correct.
    used_hlo_shape = &hlo_shape.tuple_shapes(0);
  } else {
    used_hlo_shape = &hlo_shape;
  }

  int64 stride_key = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();
  DCHECK(hlo_dim >= 0);
  CHECK(stride_key > 0 && stride_key <= used_hlo_shape->dimensions(hlo_dim));

  auto& conv_dnums = hlo->convolution_dimension_numbers();
  auto input_feat_dim = conv_dnums.input_feature_dimension();
  auto kernel_input_feat_dim = conv_dnums.kernel_input_feature_dimension();
  auto output_feat_dim = conv_dnums.output_feature_dimension();
  auto kernel_output_feat_dim = conv_dnums.kernel_output_feature_dimension();

  if (hlo_dim == kernel_input_feat_dim) {
    auto input = hlo->mutable_operand(0);
    auto& input_shape = input->shape();
    auto& input_strategy = solution->hlo_strategy_map_.at(input);

    // Backinfer input
    if (input_strategy.Glue()) {
      solution->hlo_strategy_map_[input] = DimStrategy(
          input_shape, input_feat_dim, num_replicas_, stride_key);
      if (!BackInference(input, solution)) return false;
    } else {
      if (input_strategy.partition_dim() != input_feat_dim) {
        return false;
      }
    }

    return true;
  }

  if (hlo_dim == kernel_output_feat_dim) {
    auto output = hlo->mutable_operand(1);
    auto& output_shape = output->shape();
    auto& output_strategy = solution->hlo_strategy_map_.at(output);

    // Backinfer kernel 
    if (output_strategy.Glue()) {
      solution->hlo_strategy_map_[output] = DimStrategy(
          output_shape, output_feat_dim, num_replicas_, stride_key);
      if (!BackInference(output, solution)) return false;
    } else {
      if (output_strategy.partition_dim() != output_feat_dim) {
        return false;
      }
    }

    return true;
  }

  return false;
}

bool FastSpmdStrategyBase::BackInferConv(HloInstruction* hlo, Solution* solution) {
  if (hlo->custom_call_target() == "__cudnn$convBackwardFilter") {
    return BackInferFilter(hlo, solution);
  }

  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);

  const Shape* used_hlo_shape;
  auto& hlo_shape = hlo->shape();
  if (hlo_shape.IsTuple()) {
    // TODO: We assume index == 0. It may not be correct.
    used_hlo_shape = &hlo_shape.tuple_shapes(0);
  } else {
    used_hlo_shape = &hlo_shape;
  }

  int64 stride_key = hlo_strategy.stride_on_dim();
  int64 hlo_dim = hlo_strategy.partition_dim();
  DCHECK(hlo_dim >= 0);
  CHECK(stride_key > 0 && stride_key <= used_hlo_shape->dimensions(hlo_dim));

  auto& conv_dnums = hlo->convolution_dimension_numbers();
  auto input_batch_dim = conv_dnums.input_batch_dimension();
  auto output_batch_dim = conv_dnums.output_batch_dimension();
  auto output_feat_dim = conv_dnums.output_feature_dimension();
  auto kernel_output_feat_dim = conv_dnums.kernel_output_feature_dimension();

  if (hlo_dim == output_batch_dim) {
    auto input = hlo->mutable_operand(0);
    auto& input_shape = input->shape();
    auto& input_strategy = solution->hlo_strategy_map_.at(input);

    // Backinfer input
    if (input_strategy.Glue()) {
      solution->hlo_strategy_map_[input] = DimStrategy(
          input_shape, input_batch_dim, num_replicas_, stride_key);
      if (!BackInference(input, solution)) return false;
    } else {
      if (input_strategy.partition_dim() != input_batch_dim) {
        return false;
      }
    }
    return true;
  }

  if (hlo_dim == output_feat_dim) {
    auto kernel = hlo->mutable_operand(1);
    auto& kernel_shape = kernel->shape();
    auto& kernel_strategy = solution->hlo_strategy_map_.at(kernel);

    // Backinfer kernel 
    if (kernel_strategy.Glue()) {
      solution->hlo_strategy_map_[kernel] = DimStrategy(
          kernel_shape, kernel_output_feat_dim, num_replicas_, stride_key);
      if (!BackInference(kernel, solution)) {
        return false;
      }
    } else {
      if (kernel_strategy.partition_dim() != kernel_output_feat_dim) {
        return false;
      }
    }

    return true;
  }

  // If neither input nor kernel satisfies, there is conflict!
  return false;
}

bool FastSpmdStrategyBase::BackInferGetTupleElement(
    HloInstruction* hlo, Solution* solution) {

  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 hlo_dim = hlo_strategy.partition_dim();
  DCHECK(hlo_dim >= 0);

  for (int i=0; i<hlo->operands().size(); ++i) {
    auto input = hlo->mutable_operand(i);
    auto& input_strategy = solution->hlo_strategy_map_.at(input);

    // Backinfer input
    if (input_strategy.Glue()) {
      solution->hlo_strategy_map_[input] = DimStrategy(hlo_strategy);
      if (!BackInference(input, solution)) return false;
    } else {
      auto& input_shape = input->shape();
      const Shape* used_input_shape;
      if (input_shape.IsTuple()) {
        // all tensors in a tuple own the same shape
        used_input_shape = &ShapeUtil::GetTupleElementShape(input_shape, 0);
      } else {
        used_input_shape = &input_shape;
      }
      if (input_strategy.partition_dim() != hlo_dim) {
        return false;
      }
    }
  }

  return true;
}

bool FastSpmdStrategyBase::BackInferReduceWindowAndSelectScatter(
    HloInstruction* hlo, Solution* solution) {
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  int64 hlo_dim = hlo_strategy.partition_dim();
  DCHECK(hlo_dim >= 0);

  const Window& window = hlo->window();
  if ( window.dimensions(hlo_dim).size() != 1 ||
       window.dimensions(hlo_dim).stride() != 1 ) {
    return false;
  }

  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);

  // Backinfer input
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_[input] = DimStrategy(
        input->shape(), hlo_dim, num_replicas_);
    if (!BackInference(input, solution)) return false;
  } else {
    if (input_strategy.partition_dim() != hlo_dim) {
      return false;
    }
  }

  return true;
}

bool FastSpmdStrategyBase::BackInference(HloInstruction* hlo, Solution* solution) {
  switch (hlo->opcode()) {
    case HloOpcode::kBroadcast: {
      return BackInferBcast(hlo, solution);
    }

    case HloOpcode::kScatter: {
      return BackInferScatter(hlo, solution);
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
    case HloOpcode::kTanh: {
      return BackInferUnary(hlo, solution);
    }

    case HloOpcode::kDAPPLEAllReduce: {
      return BackInferCrossReplicaSum(hlo, solution);
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
      return BackInferBinary(hlo, solution);
    }

    case HloOpcode::kSelect: {
      return BackInferSelect(hlo, solution);
    }

    case HloOpcode::kConcatenate: {
      return BackInferConcat(hlo, solution);
    }

    case HloOpcode::kSlice: {
      return BackInferSlice(hlo, solution);
    }

    case HloOpcode::kReshape:
    case HloOpcode::kBitcastConvert: {
      return BackInferReshape(hlo, solution);
    }

    case HloOpcode::kDot: {
      return BackInferDot(hlo, solution);
    }

    case HloOpcode::kReduce: {
      return BackInferReduce(hlo, solution);
    }

    case HloOpcode::kGather: {
      return BackInferGather(hlo, solution);
    }

    case HloOpcode::kPad: {
      return BackInferPad(hlo, solution);
    }

    case HloOpcode::kTranspose: {
      return BackInferTranspose(hlo, solution);
    }

    case HloOpcode::kIota: {
      if (hlo->metadata().op_type() == "OneHot") {
        // In OneHot op, splitting on iota_dimension is not allowed for
        // maintaining its SPMD semantics.
        return BackInferIota(hlo, solution);
      }
      // In general, splitting on iota_dimension destroys SPMD semantics.
      // However, the following cases are execeptional.
      // Iota in GatherV2 is used to generate batch gather dimension.
      // In this case, spliting on iota_dimension is allowed because the
      // opreand of gather is also split on batch dimension, which the
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

    case HloOpcode::kConstant: {
      VLOG(0) << "[FIXME] backinfer of kConstant->" << hlo->ToString();
      CHECK(solution->hlo_strategy_map_[hlo].Glue());
      break;
    }

    case HloOpcode::kParameter: {
      break;
    }

    case HloOpcode::kCustomCall: {
      return BackInferCustomCall(hlo, solution);
    }

    case HloOpcode::kGetTupleElement: {
      return BackInferGetTupleElement(hlo, solution);
    }

    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kReduceWindow: {
      return BackInferReduceWindowAndSelectScatter(hlo, solution);
    }

    case HloOpcode::kRng:
      // Do not need to back infer kRng operands
      return true;
    
    default: {
      VLOG(0) << "Unhandled instruction in BackInference->"
              << hlo->ToString();
      CHECK(0);
    }
  }
  return true;
}

bool FastSpmdStrategyBase::CrossReplicaSumInference(
    HloInstruction* hlo, Solution* solution) {
  // We assume AUTO_DP performed before SHARDING
  auto lhs = hlo->mutable_operand(0);
  auto& lhs_strategy = solution->hlo_strategy_map_.at(lhs);

  if (2 == hlo->operand_count()) {
    auto rhs = hlo->mutable_operand(1);
    auto& rhs_strategy = solution->hlo_strategy_map_.at(rhs);
    CHECK(rhs_strategy.Glue());
  }

  solution->hlo_strategy_map_[hlo] = lhs_strategy;
  return true;
}

bool FastSpmdStrategyBase::UnaryInference(HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);

  if (!solution->hlo_strategy_map_.count(hlo) ||
      solution->hlo_strategy_map_.at(hlo).Glue()) {
    solution->hlo_strategy_map_[hlo] = input_strategy;
    return true;
  }
  return input_strategy.Match(solution->hlo_strategy_map_.at(hlo));
}

bool FastSpmdStrategyBase::ScatterInference(HloInstruction* hlo, Solution* solution) {
  auto init_val = hlo->mutable_operand(0);
  auto& init_strategy = solution->hlo_strategy_map_.at(init_val);

  auto indices = hlo->mutable_operand(1);
  auto& indices_shape = indices->shape();
  auto indices_rank = indices_shape.rank();
  auto& indices_strategy = solution->hlo_strategy_map_.at(indices);

  auto updates = hlo->mutable_operand(2);
  auto& updates_shape = updates->shape();
  auto updates_rank = updates_shape.rank();
  auto& updates_strategy = solution->hlo_strategy_map_.at(updates);

  auto& hlo_shape = hlo->shape();
  int64 hlo_rank = hlo_shape.rank();

  auto& scatter_dims = hlo->scatter_dimension_numbers(); 
  auto& update_window_dims = scatter_dims.update_window_dims();

  if (!init_strategy.Glue()) {
    int64 init_lo = init_strategy.stride_on_dim();
    int64 init_dim = init_strategy.partition_dim();
    auto right_gap = updates_rank - update_window_dims[0];
    if (init_dim + right_gap < hlo_rank) {
      // TODO: Support embedding row sharding
      return false;
    }

    CHECK(!updates_strategy.Glue());
    updates_strategy.ApplyToShape(hlo_shape);
    int64 updates_lo = updates_strategy.stride_on_dim();
    int64 updates_dim = updates_strategy.partition_dim();
    CHECK(hlo_rank-init_dim == updates_rank-updates_dim);
    CHECK(init_lo == updates_lo);
    solution->hlo_strategy_map_[hlo] = init_strategy;
    return true;
  }

  auto index_vector_dim = scatter_dims.index_vector_dim();
  int64 indices_rank_without_trailing_dim;
  if (indices_rank > index_vector_dim) {
    // have explicit trailing dimension
    CHECK(indices_rank == index_vector_dim+1);
    CHECK(indices_shape.dimensions(indices_rank-1) == 1); // trailing dimension

    indices_rank_without_trailing_dim = indices_rank-1;
  } else {
    indices_rank_without_trailing_dim = indices_rank;
  }

  //CHECK(update_window_dims.size() == 1);
  CHECK(update_window_dims[0] == indices_rank_without_trailing_dim);

  if (!indices_strategy.Glue()) {
    // Check for strategy consistency
    auto indices_dim = indices_strategy.partition_dim();
    if (indices_dim == indices_rank_without_trailing_dim || 
        updates_strategy.Glue()) {
      // 1). indices is splitted at trailing dimension, it is an invalid split
      // 2). indices's split is valid, but update tensor is not splitted
      return false;
    }

    auto updates_dim = updates_strategy.partition_dim();
    if (indices_dim != updates_dim) {
      return false;
    }
  } else if (!updates_strategy.Glue()) {
    int64 lo = updates_strategy.stride_on_dim();
    int64 updates_dim = updates_strategy.partition_dim();

    if (updates_dim >= update_window_dims[0]) {
      // split update window(i.e. split output tensor)
      int64 right_offset = updates_rank - updates_dim;
      int64 hlo_dim = hlo_rank - right_offset;

      // Bounds check
      CHECK(hlo_shape.dimensions(hlo_dim) ==
            updates_shape.dimensions(updates_dim));

      solution->hlo_strategy_map_[hlo] = DimStrategy(
          hlo_shape, hlo_dim, num_replicas_, lo);
    } else {
      // split index tensor
      solution->hlo_strategy_map_[hlo] = DimStrategy();
    }

    return true;
  }

  solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
  return true;
}

bool FastSpmdStrategyBase::SelectInference(HloInstruction* hlo, Solution* solution) {
  int64 input_dim = -1, stride_key = -1;
  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    auto input = hlo->mutable_operand(i);
    auto& input_strategy = solution->hlo_strategy_map_.at(input);
    if (input_strategy.Glue()) continue;

    int64 lo = input_strategy.stride_on_dim();
    int64 d = input_strategy.partition_dim();
    if (input_dim >= 0) {
      if (input_dim != d) {
        // Input strategies are inconsistent with each other!
        return false;
      }
      CHECK(stride_key == lo);
    } else {
      input_dim = d;
      stride_key = lo;
    }
  }

  if (input_dim < 0) {
    solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
  } else {
    CHECK(stride_key > 0);
    solution->hlo_strategy_map_[hlo] = DimStrategy(
        hlo->shape(), input_dim, num_replicas_, stride_key);
  }

  return true;
}

bool FastSpmdStrategyBase::BinInference(HloInstruction* hlo, Solution* solution) {
  auto lhs = hlo->mutable_operand(0);
  CHECK(solution->hlo_strategy_map_.count(lhs));
  auto& lhs_strategy = solution->hlo_strategy_map_.at(lhs);

  auto rhs = hlo->mutable_operand(1);
  CHECK(solution->hlo_strategy_map_.count(rhs));
  auto& rhs_strategy = solution->hlo_strategy_map_.at(rhs);

  if (lhs_strategy.Glue()) {
    solution->hlo_strategy_map_[hlo] = rhs_strategy;

    if (!rhs_strategy.Glue()) {
      // Strategy replacement
      solution->hlo_strategy_map_[lhs] = rhs_strategy;
      return BackInference(lhs, solution);
    }
    return true;
  } 

  if (rhs_strategy.Glue()) {
    solution->hlo_strategy_map_[hlo] = lhs_strategy;

    if (!lhs_strategy.Glue()) {
      // Strategy replacement
      solution->hlo_strategy_map_[rhs] = lhs_strategy;
      return BackInference(rhs, solution);
    }
    return true;
  }

  if (lhs_strategy.Match(rhs_strategy)) {
    solution->hlo_strategy_map_[hlo] = lhs_strategy;
    return true;
  }

  return false;
}

bool FastSpmdStrategyBase::GatherInference(HloInstruction* hlo, Solution* solution) {
  auto data = hlo->mutable_operand(0);
  auto& data_shape = data->shape();
  int64 data_rank = data_shape.rank();
  auto& data_strategy = solution->hlo_strategy_map_.at(data);

  auto indices = hlo->mutable_operand(1);
  auto& index_shape = indices->shape();
  int64 index_rank = index_shape.rank();
  auto& indices_strategy = solution->hlo_strategy_map_.at(indices);

  const HloGatherInstruction* gather = DynCast<HloGatherInstruction>(hlo);
  auto& gather_dims = gather->gather_dimension_numbers();
  auto& offset_dims = gather_dims.offset_dims();
  auto slice_sizes = gather->gather_slice_sizes();
  auto& collapsed_slice_dims = gather_dims.collapsed_slice_dims();
  int64 num_collapsed = collapsed_slice_dims.size();
  std::set<int> collapsed_dims_set(collapsed_slice_dims.begin(), 
                                   collapsed_slice_dims.end());
  for (auto collapsed : collapsed_slice_dims) {
    CHECK(slice_sizes[collapsed] == 1);
  }

  auto& gather_shape = gather->shape();
  int64 gather_rank = gather_shape.rank();
  // Check offset dims
  int64 offset_left = 0;
  {
    std::set<int> offset_dims_set(offset_dims.begin(), offset_dims.end());
    for (int64 h = gather_rank-1; h >= 0; --h) {
      if (offset_dims_set.count(h)) {
        offset_left = h;
      } else {
        for (int64 l = h-1; l >= 0; --l) {
          CHECK(!offset_dims_set.count(l));
        }
        break;
      }
    }
  }

  int64 num_offsets = gather_rank - offset_left;
  auto num_offsets_uncollapsed = num_offsets + num_collapsed;
  //CHECK(offset_left + 1 == index_rank);
  CHECK(num_offsets_uncollapsed == slice_sizes.size());

  // index -> gather
  if (!indices_strategy.Glue()) {
    int64 indices_dim = indices_strategy.partition_dim();
    CHECK(indices_dim >= 0 && indices_dim < offset_left);
    CHECK(solution->hlo_strategy_map_.insert(std::make_pair(
        hlo, DimStrategy(gather_shape, indices_dim, num_replicas_))).second);
    int batch_dims = 0;
    if (indices->opcode() == HloOpcode::kConcatenate) {
      for (auto op : indices->operands()) {
        if (op->opcode() == HloOpcode::kIota) ++batch_dims;
      }

      if (indices_dim < batch_dims
          && (!solution->hlo_strategy_map_.count(data)
              || solution->hlo_strategy_map_.at(data).Glue())) {
        solution->hlo_strategy_map_[data] = DimStrategy(
            data_shape, indices_dim, num_replicas_);
        if (!BackInference(data, solution)) return false;
      } else if (!solution->hlo_strategy_map_.count(data) ||
                 solution->hlo_strategy_map_.at(data).Glue()) {
        // Split on both data and index are not allowed except on batch_dims
        return false;
      }
    }
    return true;
  }

  // data -> gather
  if (!data_strategy.Glue()) {
    int64 data_dim = data_strategy.partition_dim();
    CHECK(data_dim >= 0);

    int uncollapsed = offset_left;
    for (int r = 0; r < num_offsets_uncollapsed; ++r) {
      int64 data_r = data_rank - num_offsets_uncollapsed + r;
      CHECK(slice_sizes[r] <= data_shape.dimensions(data_r));
      if (collapsed_dims_set.count(r)) {
        CHECK(slice_sizes[r] == 1);
        // (TODO) Consider PEARLGather scenario
        if (data_dim == data_r) {
          int batch_dims = 0;
          if (indices->opcode() != HloOpcode::kConcatenate) return false;
          for (auto op : indices->operands()) {
            if (op->opcode() == HloOpcode::kIota) ++batch_dims;
          }
          if (data_dim < batch_dims
              && (!solution->hlo_strategy_map_.count(hlo)
                  || solution->hlo_strategy_map_.at(hlo).Glue())) {
            solution->hlo_strategy_map_[hlo] = DimStrategy(
                gather_shape, data_dim, num_replicas_);
            return true;
          }
          return false;
        }
        continue;
      } else if (slice_sizes[r] == data_shape.dimensions(data_r)) {
        if (data_dim == data_r) {
          if (!solution->hlo_strategy_map_.count(hlo) ||
              solution->hlo_strategy_map_.at(hlo).Glue()) {
            solution->hlo_strategy_map_[hlo] = DimStrategy(
                gather_shape, uncollapsed, num_replicas_);
            return true;
          }
          CHECK(uncollapsed == solution->hlo_strategy_map_.at(hlo).partition_dim());
          return true;
        }
      }
      ++uncollapsed;
    }
  }

  // Set to default
  solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
  return true;
}

bool FastSpmdStrategyBase::ConcatInference(HloInstruction* hlo, Solution* solution) {
  // if (!IdenticalConcat(hlo)) return false;

  int64 concat_dim = hlo->dimensions(0);
  int64 input_dim = -1, input_stride = -1;
  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    auto input = hlo->mutable_operand(i);
    auto& input_strategy = solution->hlo_strategy_map_.at(input);
    if (input_strategy.Glue()) continue;

    int64 lo = input_strategy.stride_on_dim();
    int64 d = input_strategy.partition_dim();
    if (input_dim >= 0) {
      if (input_dim != d) {
        // Input strategies are inconsistent with each other!
        return false;
      }
      CHECK(input_stride == lo);
    } else {
      input_dim = d;
      input_stride = lo;
    }
  }

  if (input_dim < 0) {
    solution->hlo_strategy_map_[hlo] = DimStrategy();
    return true;
  }

  int64 num_operands = hlo->operand_count();
  int64 hlo_dim = input_dim;
  int64 hlo_stride = input_stride;
  if (input_dim == concat_dim) {
    if (IdenticalConcat(hlo)) {
      // Each param of concat instruction has the same number
      // of elements, and the hlo_stride can linearly expand
      // from input to output.
      hlo_stride *= num_operands;
    } else {
      // Each param of concat instruction does not have the same number
      // of elements, and the real stride of output shall be determined
      // according to the real concated dimension of the output when
      // constructing DimStrategy for the output instruction.
      //
      // E.g., %concatenate = f32[8,272,1024]{2,1,0} concatenate(
      //                      f32[8,16,1024]{2,1,0} %arg1,
      //                      f32[8,256,1024]{2,1,0} %arg2), dimensions={1}
      hlo_stride = -1;
    }
  }

  if (!solution->hlo_strategy_map_.count(hlo) ||
      solution->hlo_strategy_map_.at(hlo).Glue()) {
    solution->hlo_strategy_map_[hlo] =
        DimStrategy(hlo->shape(), hlo_dim, num_replicas_, hlo_stride);
    return true;
  }
  auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
  CHECK(hlo_strategy.stride_on_elements() == hlo_stride);

  return true;
}

bool FastSpmdStrategyBase::SliceInference(HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_shape = input->shape();
  auto& input_strategy = solution->hlo_strategy_map_.at(input);
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_[hlo] = DimStrategy();
    return true;
  }

  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();

  const HloSliceInstruction* slice = DynCast<HloSliceInstruction>(hlo);
  const std::vector<int64> slice_starts = slice->slice_starts();
  const std::vector<int64> slice_limits = slice->slice_limits();
  const std::vector<int64> slice_strides = slice->slice_strides();

  auto& hlo_shape = hlo->shape();
  CHECK(input_dim >= 0 && input_dim < input_shape.rank());
  int64 slice_w_at_in_dim = slice_limits[input_dim] - slice_starts[input_dim];
  if (slice_w_at_in_dim == input_shape.dimensions(input_dim) &&
      slice_strides[input_dim] == 1) {
    // input_dim is not slice dimension
    if (!solution->hlo_strategy_map_.count(hlo) ||
        solution->hlo_strategy_map_.at(hlo).Glue()) {
      solution->hlo_strategy_map_[hlo] = DimStrategy(
          hlo_shape, input_dim, num_replicas_, lo);
    }
    return true;
  } else {
    // input_dim is slice dimension
    int64 input_lower_dim_data_size = 1;
    int64 hlo_lower_dim_data_size = 1;
    for (int i = 0; i < input_dim; ++i) {
      input_lower_dim_data_size *= input_shape.dimensions(i);
      hlo_lower_dim_data_size *= hlo_shape.dimensions(i);
    }
    int64 total_size = input_strategy.size_on_elements() * num_replicas_;
    int64 slice_start_addr = slice_starts[input_dim]*input_lower_dim_data_size;
    if (slice_start_addr % total_size != 0) {
      // slice offset is not aligned
      return false;
    }

    int64 slice_limit_addr = slice_limits[input_dim]*input_lower_dim_data_size;
    if (slice_limit_addr % total_size != 0) {
      // slice offset is not aligned
      return false;
    }

    int64 slice_stride = slice_limit_addr - slice_start_addr;
    int64 slice_on_dim = input_strategy.stride_on_dim() /
                          (input_strategy.stride_on_elements() / slice_stride);
    solution->hlo_strategy_map_[hlo] = DimStrategy(slice_stride, slice_on_dim);
    return true;
  }
}

bool FastSpmdStrategyBase::IotaInference(HloInstruction* hlo, Solution* solution) {
  // Set to default. Can only be inferred in backward inference
  solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
  return true;
}

bool FastSpmdStrategyBase::TransposeInference(HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
    return true;
  }

  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();
  CHECK(input_dim >= 0 && lo > 0);

  int64 hlo_dim = -1;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    // where do I move?
    if (input_dim == hlo->dimensions(i)) {
      hlo_dim = i;
      break;
    }
  }
  CHECK(hlo_dim >= 0);

  solution->hlo_strategy_map_[hlo] = DimStrategy(hlo->shape(), hlo_dim,
                                 num_replicas_, lo);
  return true;
}

bool FastSpmdStrategyBase::GetTupleElementInference(
    HloInstruction* hlo, Solution* solution) {
  int64 split_dim = -1, output_stride = -1;

  for (auto* input : hlo->operands()) {
    auto& input_strategy = solution->hlo_strategy_map_.at(input);
    if (input_strategy.Glue()) {
      continue;
    }

    int64 input_stride = input_strategy.stride_on_dim();
    int64 input_dim = input_strategy.partition_dim();
    if (split_dim >= 0) {
      if (split_dim != input_dim) {
        // Input strategies are inconsistent with each other!
        return false;
      }
      CHECK(output_stride == input_stride);
    } else {
      split_dim = input_dim;
      output_stride = input_stride;
    }
  }

  if (split_dim < 0) {
    solution->hlo_strategy_map_[hlo] = DimStrategy();
    return true;
  }

  if (!solution->hlo_strategy_map_.count(hlo) ||
      solution->hlo_strategy_map_.at(hlo).Glue()) {
    solution->hlo_strategy_map_[hlo] = DimStrategy(
        hlo->shape(), split_dim, num_replicas_, output_stride);
  }

  // TODO(lansong): Need back inference for all operands

  return true;
}

bool FastSpmdStrategyBase::ReduceWindowAndSelectScatterInference(
    HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_[hlo] = DimStrategy();
    return true;
  }

  int64 lo = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();

  const Window& window = hlo->window();
  if ( window.dimensions(input_dim).size() != 1 ||
       window.dimensions(input_dim).stride() != 1 ) {
    return false;
  }

  solution->hlo_strategy_map_[hlo] = DimStrategy(
      hlo->shape(), input_dim, num_replicas_, lo);

  return true;
}

bool FastSpmdStrategyBase::CustomCallInference(HloInstruction* hlo, Solution* solution) {
  if (IsConvolution(*hlo)) {
    return ConvInference(hlo, solution);
  }

  return false;
}

bool FastSpmdStrategyBase::ConvInference(HloInstruction* hlo, Solution* solution) {
  VLOG(2) << "ConvInference->" << hlo->ToString() << "\n";

  auto target = hlo->custom_call_target();

  auto& conv_dnums = hlo->convolution_dimension_numbers();
  auto input_batch_dim = conv_dnums.input_batch_dimension();
  auto output_batch_dim = conv_dnums.output_batch_dimension();
  auto input_feat_dim = conv_dnums.input_feature_dimension();
  auto output_feat_dim = conv_dnums.output_feature_dimension();
  auto kernel_input_feat_dim = conv_dnums.kernel_input_feature_dimension();
  auto kernel_output_feat_dim = conv_dnums.kernel_output_feature_dimension();

  auto input = hlo->mutable_operand(0);
  auto& input_shape = input->shape();
  auto& input_strategy = solution->hlo_strategy_map_.at(input);
  VLOG(2)   << "input_strategy->" << input_strategy.ToString();

  auto kernel = hlo->mutable_operand(1);
  auto& kernel_shape = kernel->shape();
  auto& kernel_strategy = solution->hlo_strategy_map_.at(kernel);
  VLOG(2)   << "kernel_strategy->" << kernel_strategy.ToString();

  if (target == "__cudnn$convBackwardFilter") {
    solution->hlo_strategy_map_[hlo] = DimStrategy();
    return true;
  }

  if (input_strategy.Glue() && kernel_strategy.Glue()) {
    solution->hlo_strategy_map_[hlo] = DimStrategy();
    return true;
  }

  auto& hlo_shape = hlo->shape();
  const Shape* used_hlo_shape;
  if (hlo_shape.IsTuple()) {
    used_hlo_shape = &ShapeUtil::GetTupleElementShape(hlo_shape, 0);
  } else {
    used_hlo_shape = &hlo_shape;
  }

  if (kernel_strategy.Glue()) {
    int64 stride_key = input_strategy.stride_on_dim();
    int64 input_dim = input_strategy.partition_dim();
    CHECK (input_dim >= 0);
    CHECK(stride_key > 0 && 
          stride_key <= input_shape.dimensions(input_dim));

    if (input_dim == input_feat_dim) {
      solution->hlo_strategy_map_[kernel] = DimStrategy(
          kernel_shape, kernel_input_feat_dim, num_replicas_, stride_key);
      if (!BackInference(kernel, solution)) {
        return false;
      }

      solution->hlo_strategy_map_[hlo] = DimStrategy();
      return true;
    }

    if (input_dim != input_batch_dim) {
      return false;
    }

    solution->hlo_strategy_map_[hlo] = DimStrategy(
        *used_hlo_shape, output_batch_dim, num_replicas_, stride_key);
    return true;
  }

  if (input_strategy.Glue()) {
    int64 stride_key = kernel_strategy.stride_on_dim();
    int64 kernel_dim = kernel_strategy.partition_dim();
    CHECK (kernel_dim >= 0);
    CHECK(stride_key > 0 && 
          stride_key <= kernel_shape.dimensions(kernel_dim));

    if (kernel_dim == kernel_input_feat_dim) {
      solution->hlo_strategy_map_[input] = DimStrategy(
          input_shape, input_feat_dim, num_replicas_, stride_key);
      if (!BackInference(input, solution)) {
        return false;
      }

      solution->hlo_strategy_map_[hlo] = DimStrategy();
      return true;
    }

    if (kernel_dim != kernel_output_feat_dim) {
      return false;
    }

    solution->hlo_strategy_map_[hlo] = DimStrategy(
        *used_hlo_shape, output_feat_dim, num_replicas_, stride_key);
    return true;
  }

  // !kernel_strategy.Glue() && !input_strategy.Glue()
  int64 input_stride_key = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();
  CHECK (input_dim >= 0);
  CHECK(input_stride_key > 0 && 
        input_stride_key <= input_shape.dimensions(input_dim));

  int64 kernel_stride_key = kernel_strategy.stride_on_dim();
  int64 kernel_dim = kernel_strategy.partition_dim();
  CHECK (kernel_dim >= 0);
  CHECK(kernel_stride_key > 0 && 
        kernel_stride_key <= kernel_shape.dimensions(kernel_dim));

  if (input_dim != input_feat_dim ||
      kernel_dim != kernel_input_feat_dim ||
      input_stride_key != kernel_stride_key) {
    return false;
  }

  solution->hlo_strategy_map_[hlo] = DimStrategy();
  return true;
}

bool FastSpmdStrategyBase::DotInference(HloInstruction* hlo, Solution* solution) {
  auto lhs = hlo->mutable_operand(0);
  auto& lhs_shape = lhs->shape();
  auto lhs_rank = lhs_shape.rank();
  auto& lhs_strategy = solution->hlo_strategy_map_.at(lhs);

  auto rhs = hlo->mutable_operand(1);
  auto& rhs_shape = rhs->shape();
  auto rhs_rank = rhs_shape.rank();
  auto& rhs_strategy = solution->hlo_strategy_map_.at(rhs);

  auto& hlo_shape = hlo->shape();
  int64 hlo_rank = hlo_shape.rank();

  const DotDimensionNumbers &dim_nums = hlo->dot_dimension_numbers();
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
           dim_nums.rhs_batch_dimensions_size());
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size() + 2, hlo_shape.rank());
  int64 num_batches = dim_nums.lhs_batch_dimensions_size();
  int64 lhs_contracting_dim = dim_nums.lhs_contracting_dimensions(0);
  int64 rhs_contracting_dim = dim_nums.rhs_contracting_dimensions(0);

  if (lhs_strategy.Glue() && rhs_strategy.Glue()) {
    solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
    return true;
  }

  if (lhs_strategy.Glue() && !rhs_strategy.Glue()) {
    int64 rhs_lo = rhs_strategy.stride_on_dim();
    int64 rhs_dim = rhs_strategy.partition_dim();
    CHECK(rhs_dim >= 0 && rhs_lo > 0);
    if (rhs_dim < num_batches) {
      solution->hlo_strategy_map_[lhs] = DimStrategy(
          lhs_shape, rhs_dim, num_replicas_, rhs_lo);
      if (!BackInference(lhs, solution)) return false;
      solution->hlo_strategy_map_[hlo] = DimStrategy(
          hlo_shape, rhs_dim, num_replicas_, rhs_lo);
      return true;
    }

    if (rhs_dim == rhs_contracting_dim) {
      solution->hlo_strategy_map_[lhs] = DimStrategy(
          lhs_shape, lhs_contracting_dim, num_replicas_, rhs_lo);
      if (!BackInference(lhs, solution)) return false;
      solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
      return true;
    }

    solution->hlo_strategy_map_[hlo] = 
        DimStrategy(hlo_shape, hlo_rank-1, num_replicas_, rhs_lo);
    return true;
  }

  if (!lhs_strategy.Glue() && rhs_strategy.Glue()) {
    int64 lhs_lo = lhs_strategy.stride_on_dim();
    int64 lhs_dim = lhs_strategy.partition_dim();
    CHECK(lhs_dim >= 0 && lhs_lo > 0);
    if (lhs_dim < num_batches) {
      solution->hlo_strategy_map_[rhs] = DimStrategy(
          rhs_shape, lhs_dim, num_replicas_, lhs_lo);
      if (!BackInference(rhs, solution)) return false;
      solution->hlo_strategy_map_[hlo] = DimStrategy(
          hlo_shape, lhs_dim, num_replicas_, lhs_lo);
      return true;
    }

    if (lhs_dim == lhs_contracting_dim) {
      solution->hlo_strategy_map_[rhs] = DimStrategy(
          rhs_shape, rhs_contracting_dim, num_replicas_, lhs_lo);
      if (!BackInference(rhs, solution)) return false;
      solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
      return true;
    }

    solution->hlo_strategy_map_[hlo] = DimStrategy(
        hlo_shape, hlo_rank-2, num_replicas_, lhs_lo);
    return true;
  }

  // (!lhs_strategy.Glue() && !rhs_strategy.Glue())
  int64 lhs_lo = lhs_strategy.stride_on_dim();
  int64 lhs_dim = lhs_strategy.partition_dim();

  int64 rhs_lo = rhs_strategy.stride_on_dim();
  int64 rhs_dim = rhs_strategy.partition_dim();

  if (lhs_dim < num_batches && rhs_dim < num_batches) {
    CHECK(rhs_dim == lhs_dim && lhs_lo == rhs_lo);
    solution->hlo_strategy_map_[hlo] = DimStrategy(
        hlo_shape, lhs_dim, num_replicas_, lhs_lo);
    return true;
  }

  if (lhs_dim == lhs_contracting_dim && rhs_dim == rhs_contracting_dim) {
    solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
    return true;
  } else {
    if (lhs_dim != lhs_contracting_dim && rhs_dim != rhs_contracting_dim) {
      // Since we do not split multiple output dimensions, 
      // treat this as conflict!
      return false;
    }

    // Example: for %dot.98 as following,
    //    input split: {%reshape.24:0, %reshape.97:1}, which first input split
    //        on the batch dimension, and the second input split on the
    //        contracting dimension.
    //
    //    %dot.98 = f32[10,15,16]{2,1,0} dot(f32[10,15,120]{2,1,0} %reshape.24, f32[10,120,16]{2,1,0} %reshape.97),
    //        lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1},
    //        metadata={op_type="BatchMatMulV2" op_name="ffn/combined_outputs/MatMul"},
    //
    //    then we simply return conflict.
    return false;

    // if (lhs_dim == lhs_contracting_dim) {
    //   CHECK(rhs_dim == rhs_rank-1 || rhs_dim == rhs_rank-2);
    //   (*solution)[hlo] = DimStrategy(hlo_shape, hlo_rank-1, num_replicas_, rhs_lo);
    // } else {
    //   CHECK(lhs_dim == lhs_rank-1 || lhs_dim == lhs_rank-2);
    //   (*solution)[hlo] = DimStrategy(hlo_shape, hlo_rank-2, num_replicas_, lhs_lo);
    // }
    // return true;
  }
}

bool FastSpmdStrategyBase::BitcastConvertInference(
    HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_shape = input->shape();

  auto& hlo_shape = hlo->shape();
  CHECK(ShapeUtil::ByteSizeOfPrimitiveType(input_shape.element_type()) ==
        ShapeUtil::ByteSizeOfPrimitiveType(hlo_shape.element_type()));

  auto& input_strategy = solution->hlo_strategy_map_.at(input);
  solution->hlo_strategy_map_.insert(std::make_pair(hlo, input_strategy));
  return true;
}

bool FastSpmdStrategyBase::ReshapeInference(HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);
  solution->hlo_strategy_map_[hlo] = input_strategy;
  return true;
}

bool FastSpmdStrategyBase::PadInference(HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
    return true;
  }
  int64 input_dim = input_strategy.partition_dim();
  CHECK(input_dim >= 0);

  const HloPadInstruction* pad = DynCast<HloPadInstruction>(hlo);
  CHECK(pad);

  auto config = pad->padding_config();
  auto config_dims_size = config.dimensions_size();
  CHECK(input_dim >= 0 && input_dim < config_dims_size);
  auto paddings = config.dimensions(input_dim);
  if (paddings.edge_padding_low() == 0 && paddings.edge_padding_high() == 0 &&
      paddings.interior_padding() == 0) {
    if (!solution->hlo_strategy_map_.count(hlo) ||
        solution->hlo_strategy_map_.at(hlo).Glue()) {
      solution->hlo_strategy_map_[hlo] = DimStrategy(
          pad->shape(), input_dim, num_replicas_);
      return true;
    }
    
    // Consistency check
    auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
    CHECK(hlo_strategy.partition_dim() == input_dim);
    return true;
  }

  return false;
}

bool FastSpmdStrategyBase::ReduceInference(HloInstruction* hlo, Solution* solution) {
  // NOTE: output of reduce instruction may be a tuple, the strategy of reduce
  //       is the strategy of the tuple element.
  int64 split_dim = -1, output_stride = -1;
  for (auto* input : hlo->operands()) {
    auto& input_strategy = solution->hlo_strategy_map_.at(input);
    if (input_strategy.Glue()) {
      continue;
    }

    auto& input_shape = input->shape();
    int64 input_stride = input_strategy.stride_on_dim();
    int64 input_dim = input_strategy.partition_dim();
    CHECK(input_dim >= 0);
    if (split_dim >= 0) {
      if (split_dim != input_dim) {
        // Input strategies are inconsistent with each other!
        return false;
      }
      CHECK(output_stride == input_stride);
    } else {
      split_dim = input_dim;
      output_stride = input_stride;
    }
  }

  if (split_dim < 0) {
    solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
    return true;
  }

  auto& reduce_dims = hlo->dimensions();
  std::set<int> reduce_dims_set(reduce_dims.begin(), reduce_dims.end());
  if (reduce_dims_set.count(split_dim)) {
    solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
    return true;
  }

  // Mapping split_dim -> hlo_dim
  // NOTE: all non-scalar inputs own same shape
  auto& input_shape = hlo->mutable_operand(0)->shape();
  int64 hlo_dim = 0;
  for (int64 r = 0; r < input_shape.rank(); ++r) {
    if (reduce_dims_set.count(r)) continue;

    if (split_dim == r) break;
    ++hlo_dim;
  }

  solution->hlo_strategy_map_[hlo] = DimStrategy(
      hlo->shape(), hlo_dim, num_replicas_, output_stride);

  return true;
}

bool FastSpmdStrategyBase::BcastInference(HloInstruction* hlo, Solution* solution) {
  auto input = hlo->mutable_operand(0);
  auto& input_strategy = solution->hlo_strategy_map_.at(input);
  if (input_strategy.Glue()) {
    solution->hlo_strategy_map_.insert(std::make_pair(hlo, DimStrategy()));
    return true;
  }

  auto& input_shape = input->shape();
  int64 stride_key = input_strategy.stride_on_dim();
  int64 input_dim = input_strategy.partition_dim();
  CHECK (input_dim >= 0);

  auto& bcast_dims = hlo->dimensions(); 
  CHECK(input_dim < bcast_dims.size());
  CHECK(stride_key > 0 && stride_key <= input_shape.dimensions(input_dim));
  
  int64 output_dim = bcast_dims[input_dim];
  solution->hlo_strategy_map_[hlo] = DimStrategy(
      hlo->shape(), output_dim, num_replicas_, stride_key);
  return true;
}

bool FastSpmdStrategyBase::StrategyInference(HloInstruction* hlo, Solution* solution) {
  switch (hlo->opcode()) {
    case HloOpcode::kConstant: {
      if (!ConstantInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kBroadcast: {
      if (!BcastInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kSelect: {
      if (!SelectInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kScatter: {
      if (!ScatterInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kAbs:
    case HloOpcode::kLog:
    case HloOpcode::kExp:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kTanh:
    case HloOpcode::kSqrt:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kPower:
    case HloOpcode::kNegate:
    case HloOpcode::kConvert:
    case HloOpcode::kIsFinite: {
      if (!UnaryInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kAnd:
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kCompare:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic: {
      if (!BinInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kGather: {
      if (!GatherInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kReduce: {
      if (!ReduceInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kPad: {
      if (!PadInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kReshape: {
      if (!ReshapeInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kIota: {
      if (!IotaInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kConcatenate: {
      if (!ConcatInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kSlice: {
      if (!SliceInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kBitcastConvert: {
      if (!BitcastConvertInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kDot: {
      if (!DotInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kTranspose: {
      if (!TransposeInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kCustomCall: {
      if (!CustomCallInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kGetTupleElement: {
      if (!GetTupleElementInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kReduceWindow: {
      if (!ReduceWindowAndSelectScatterInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kDAPPLEAllReduce: {
      if (!CrossReplicaSumInference(hlo, solution)) {
        return false;
      }
      break;
    }

    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
      solution->hlo_strategy_map_[hlo] = DimStrategy();
      return true;

    default:
      VLOG(0) << "UnHandled instruction in StrategyInference->"
              << hlo->ToString();
      CHECK(0);
  }

  CHECK(solution->hlo_strategy_map_.count(hlo));
  return true;
}

bool FastSpmdStrategyBase::CheckAndBackInfer(HloComputation* entry,
                         Solution* solution,
                         const std::unordered_set<HloInstruction*> worklist,
                         bool* conflict) {
  bool changed = false;
  auto post_order = entry->MakeInstructionPostOrder();
  for (auto hlo : post_order) {
    if (hlo->opcode() == HloOpcode::kTuple ||
        hlo->opcode() == HloOpcode::kConstant ||
        hlo->opcode() == HloOpcode::kBroadcast ||
        hlo->opcode() == HloOpcode::kParameter) continue;
    if (ShapeUtil::IsScalar(hlo->shape())) continue;

    CHECK(solution->hlo_strategy_map_.count(hlo));
    auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
    if (hlo_strategy.Glue()) continue;

    std::vector<int64> input_strides;
    input_strides.reserve(hlo->operand_count());
    for (auto operand : hlo->operands()) {
      input_strides.emplace_back(
          solution->hlo_strategy_map_.at(operand).stride_on_elements());
    }

    if (!BackInference(hlo, solution)) {
      CHECK (conflict);
      *conflict = true;
      break;
    }

    for (int64 i = 0; i < hlo->operand_count(); ++i) {
      auto operand = hlo->mutable_operand(i);
      if (solution->hlo_strategy_map_.at(operand).stride_on_elements() !=
          input_strides[i]) {
        if (worklist.find(operand) != worklist.end()) {
          VLOG(0) << "WARNING: solution already specificed param->"
                  << operand->ToString()
                  << " has been changed, which results in conflicts!";
          CHECK(conflict);
          *conflict = true;
          break;
        } else {
          changed = true;
        }
        break;
      }
    }
    if (*conflict) {
      break;
    }
  }
  return changed;
}

bool FastSpmdStrategyBase::CheckAndInfer(
    HloComputation* entry, Solution* solution, bool* conflict) {
  bool changed = false;
  auto post_order = entry->MakeInstructionPostOrder();
  for (auto hlo : post_order) {
    if (hlo->opcode() == HloOpcode::kTuple) continue;
    if (ShapeUtil::IsScalar(hlo->shape())) continue;

    CHECK(solution->hlo_strategy_map_.count(hlo));
    auto& hlo_strategy = solution->hlo_strategy_map_.at(hlo);
    if (!hlo_strategy.Glue()) continue;

    bool candidate = false;
    for (auto operand : hlo->operands()) {
      CHECK(solution->hlo_strategy_map_.count(operand));
      auto& operand_strategy = solution->hlo_strategy_map_.at(operand);
      if (!operand_strategy.Glue()) {
        candidate = true;
        break;
      }
    }
    if (!candidate) continue;

    switch (hlo->opcode()) {
      case HloOpcode::kGather: {
        if (!GatherInference(hlo, solution)) {
          *conflict = true;
          return changed;
        }
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      case HloOpcode::kPad: {
        CHECK(PadInference(hlo, solution));
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      case HloOpcode::kSlice: {
        if (!SliceInference(hlo, solution)) {
          *conflict = true;
          return changed;
        }
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      case HloOpcode::kOr:
      case HloOpcode::kXor:
      case HloOpcode::kAnd:
      case HloOpcode::kAdd:
      case HloOpcode::kPower:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kCompare:
      case HloOpcode::kMultiply:
      case HloOpcode::kSubtract:
      case HloOpcode::kShiftLeft:
      case HloOpcode::kShiftRightLogical:
      case HloOpcode::kShiftRightArithmetic: {
        if (!BinInference(hlo, solution)) {
          *conflict = true;
          return changed;
        }
        CHECK(!solution->hlo_strategy_map_.at(hlo).Glue());
        changed = true;
        break;
      }

      case HloOpcode::kLog:
      case HloOpcode::kExp:
      case HloOpcode::kTanh:
      case HloOpcode::kSqrt:
      case HloOpcode::kRsqrt:
      case HloOpcode::kNegate:
      case HloOpcode::kConvert:
      case HloOpcode::kIsFinite: {
        CHECK(UnaryInference(hlo, solution));
        CHECK(!solution->hlo_strategy_map_.at(hlo).Glue());
        changed = true;
        break;
      }

      case HloOpcode::kSelect: {
        CHECK(SelectInference(hlo, solution));
        CHECK(!solution->hlo_strategy_map_.at(hlo).Glue());
        changed = true;
        break;
      }

      case HloOpcode::kBroadcast: {
        CHECK(BcastInference(hlo, solution));
        CHECK(!solution->hlo_strategy_map_.at(hlo).Glue());
        changed = true;
        break;
      }

      case HloOpcode::kConcatenate: {
        if (!ConcatInference(hlo, solution)) {
          *conflict = true;
          return changed;
        }
        CHECK(!solution->hlo_strategy_map_.at(hlo).Glue());
        changed = true;
        break;
      }

      case HloOpcode::kTranspose: {
        CHECK(TransposeInference(hlo, solution));
        CHECK(!solution->hlo_strategy_map_.at(hlo).Glue());
        changed = true;
        break;
      }

      case HloOpcode::kReshape:
      case HloOpcode::kBitcastConvert: {
        CHECK(ReshapeInference(hlo, solution));
        CHECK(!solution->hlo_strategy_map_.at(hlo).Glue());
        changed = true;
        break;
      }

      case HloOpcode::kDot: {
        if (!DotInference(hlo, solution)) {
          *conflict = true;
          return false;
        }
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      case HloOpcode::kReduce: {
        CHECK(ReduceInference(hlo, solution));
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      case HloOpcode::kScatter: {
        auto stride = solution->hlo_strategy_map_.at(hlo).stride_on_elements(); 
        CHECK(ScatterInference(hlo, solution));
        changed = (stride !=
                   solution->hlo_strategy_map_.at(hlo).stride_on_elements());
        break;
      }

      case HloOpcode::kCustomCall: {
        if (!CustomCallInference(hlo, solution)) {
          *conflict = true;
          return changed;
        }
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      case HloOpcode::kGetTupleElement: {
        CHECK(GetTupleElementInference(hlo, solution));
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      case HloOpcode::kSelectAndScatter:
      case HloOpcode::kReduceWindow: {
        CHECK(ReduceWindowAndSelectScatterInference(hlo, solution));
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      case HloOpcode::kDAPPLEAllReduce: {
        CHECK(CrossReplicaSumInference(hlo, solution));
        changed = (!solution->hlo_strategy_map_.at(hlo).Glue());
        break;
      }

      default: {
        VLOG(0) << "UNHANDLED->" << hlo->ToString();
        CHECK(0);
      }
    }
  }
  return changed;
}

bool FastSpmdStrategyBase::ConstantSliced(HloComputation* entry, Solution* solution) {
  for (auto hlo : entry->instructions()) {
    if (hlo->opcode() != HloOpcode::kConstant) continue;
    if (!solution->hlo_strategy_map_.count(hlo)) continue;

    if (!solution->hlo_strategy_map_.at(hlo).Glue()) {
      auto& s = solution->hlo_strategy_map_.at(hlo);
      VLOG(0) << "[FIXME] ConstantSliced:" << hlo->ToString()
              << " stride_on_elements=" << s.ToString();
      return true;
    }
  }
  return false;
}

bool FastSpmdStrategyBase::IncrementalSatisfiable(
    HloInstruction* seed, HloComputation* entry, Solution* solution) {
  std::deque<HloInstruction*> worklist;
  HloMutableInstSet visited;

  for (auto user : seed->users()) {
    worklist.push_back(user);
    visited.insert(user);
  }

  while (!worklist.empty()) {
    HloInstruction* hlo = worklist.front();
    worklist.pop_front();

    if (!StrategyInference(hlo, solution)) {
      VLOG(2) << "Inference inconsistency in IncrementalSatisfiable->"
              << hlo->ToString();
      return false;
    }

    for (auto user : seed->users()) {
      if (user->opcode() != HloOpcode::kTuple &&
          !visited.count(user)) {
        worklist.push_back(user);
        visited.insert(user);
      }
    }
  }

  bool changed = true;
  bool conflict = false;
  while (changed) {
    changed = CheckAndInfer(entry, solution, &conflict);
    if (conflict) {
      return false;
    }
  }

  changed = true;
  conflict = false;
  while (changed) {
    changed = CheckAndBackInfer(entry, solution, {seed}, &conflict);
    if (conflict) break;
  }

  return !conflict;
}

// 'worklist': set of instructions whose slicing strategies are already
// specified at the very beginning.  During forward or backward inference, these
// initial strategies cannot be broken, otherwise we assume that a conflict is
// reached.
bool FastSpmdStrategyBase::StrategySatisfiable(
    HloComputation* entry, Solution* solution,
    const std::unordered_set<HloInstruction*> worklist) {
  auto post_order = entry->MakeInstructionPostOrder();
  for (auto inst : post_order) {
    if (inst->opcode() == HloOpcode::kParameter ||
        inst->opcode() == HloOpcode::kTuple) {
      continue;
    }

    if (!StrategyInference(inst, solution)) {
      VLOG(0) << "Strategy not satisfiable";
      return false/*conflict*/;
    }
  }

  bool changed = true;
  bool conflict = false;
  while (changed) {
    changed = CheckAndInfer(entry, solution, &conflict);
    if (conflict) {
      return false;
    }
  }

  changed = true;
  conflict = false;
  while (changed) {
    changed = CheckAndBackInfer(entry, solution, worklist, &conflict);
    if (conflict) break;
  }

  return !conflict;
}


void FastSpmdStrategyBase::ResolveProposalIndices(int64 idx,
    std::vector<int64>& proposal_sizes, std::vector<int64>& proposal_indices) {
  int64 num_proposals = proposal_sizes.size();
  proposal_indices.reserve(num_proposals);

  int64 product = 1;
  for (int64 p = 1; p < num_proposals; ++p) {
    product *= proposal_sizes[p];
  }

  int64 rem = idx;
  for (int64 i = 0; i < num_proposals-1; ++i) {
    int64 div = rem / product;
    proposal_indices.emplace_back(div);

    rem = rem % product;
    product /= proposal_sizes[i+1];
  }
  proposal_indices.emplace_back(rem);
}

StatusOr<bool> FastSpmdStrategyBase::InferenceCalibration(HloModule* module) {
  bool changed = true;

  if (module->variable_map()->empty()) {
    return changed;
  }

  auto entry = module->entry_computation();
  auto post_order = entry->MakeInstructionPostOrder();
  for (auto instr : post_order) {
    if (instr->opcode() == HloOpcode::kCopy) {
      auto input = instr->mutable_operand(0);
      auto& input_spec = input->dist_spec();
      auto input_dim = input_spec.get_dim_spec(0)->partition_dim();
      *instr->mutable_dist_spec() = input_spec;
    }
  }

  return changed;
}

void FastSpmdStrategyBase::CollectBwdKeyShardingInsts(HloModule* module) {
  auto entry = module->entry_computation();
  auto post_order = entry->MakeInstructionPostOrder();
  std::vector<HloInstruction*> sharded_insts;
  for (auto* inst : post_order) {
    if (inst->has_sharding() && !inst->sharding().IsReplicated()) {
      VLOG(2) << "[CollectBwdKeyShardingInsts] FWD->" << inst->ToString();
      sharded_insts.push_back(inst);
    }
  }

  for (auto sharded_inst : sharded_insts) {
    auto fwd_op_name = sharded_inst->metadata().op_name();
    int match_count = 0;
    for (auto* inst : post_order) {
      if (backward_insts_.find(inst) != backward_insts_.end()) {
        match_count ++;
        // Record the backward instructions corresponding to forward user
        // annotated sharding instructions. Currently these recoreded
        // instructions' sharding strategy cannot be decided yet.
        bwd_sharding_annotation_insts_.insert(inst).second;
        VLOG(2) << "[CollectBwdKeyShardingInsts] BWD->" << inst->ToString();
      }
    }
    //if (match_count > 1) {
    //  for(auto* inst : store_inst[sharded_inst]) {
    //      VLOG(0) << "[DEBUG] exceed "
    //              << "fwd inst : " << sharded_inst->ToString()
    //              << "bwd inst : " << inst->ToString();
    //  }
    //  CHECK(0) << "One-to-many mapping from fwd inst to bwd inst.";
    //}
  }
  return;
}

StatusOr<bool> FastSpmdStrategyBase::Run(HloModule* module) {
  RewriteCustomCallsAsDots(module);

  if (module->variable_map()->empty()) {
    // Skip this pass when there is no trainable variable.
    LOG(INFO) << "variable map is not setup!";
    return false;
  }

  backward_insts_ = FindBackwardInsts(module, false);
  CollectBwdKeyShardingInsts(module);
  
  if (!StrategyPlanning(module, false/*extend_sharding_param*/)) {
    return false;
  }

  auto entry = module->entry_computation();
  auto var_map = module->variable_map();
  VLOG(0) << "STRATEGY OUTPUT:";
  for (int64 i = 0; i < entry->parameter_instructions().size(); ++i) {
    auto param = entry->parameter_instruction(i);
    if (ShapeUtil::IsScalar(param->shape())) continue;

    auto dist_spec = param->dist_spec();
    auto& dim_spec = dist_spec.get_dim_spec(0);
    auto stride = dim_spec->stride();
    auto stride_on_dim = dim_spec->stride_on_dim();
    if (stride > 0 && stride_on_dim > 0) {
      // Manual annotated replicated instructions cannot be modified.
      if (param->has_sharding()) {
        CHECK(!param->sharding().IsReplicated())
            << "Manual replicated inst should not be sharding!";
      }
      VLOG(0) << param->ToString();
    }
  }

  return true;
}

FastSpmdStrategy::FastSpmdStrategy()
  : FastSpmdStrategyBase() {}

FastSpmdStrategy::FastSpmdStrategy(ParallelType ttype, bool post_layout) 
  : FastSpmdStrategyBase(post_layout)
  , ttype_(ttype) {}

FastSpmdStrategy::FastSpmdStrategy(ParallelType ttype, bool post_layout,
                          int num_replicas)
  : FastSpmdStrategyBase(post_layout, num_replicas)
  , ttype_(ttype) {}

bool FastSpmdStrategy::StrategyPlanning(HloModule* module, 
                                    bool extend_sharding_param,
                                    int window_size) {
  std::vector<HloInstruction*> raw_worklist;
  std::map<int, HloMutableInstSet> span_hlo;
  if (ttype_ == ParallelType::SPMD) {
    // run dp/sharding inference in a single pass
    ResolveParamList(module, raw_worklist);
  } else {
    ResolveParamList(module, raw_worklist, span_hlo, window_size);
  }

  auto entry = module->entry_computation();
  //HloMutableInstMap<DimStrategy> param_strategy_map;
  Solution m_solution;
  for (auto param : entry->parameter_instructions()) {
    auto& param_shape = param->shape();
    // Initialize to the default strategy
    m_solution.hlo_strategy_map_.insert(
        std::make_pair(param, DimStrategy(param_shape, -1, num_replicas_)));
  }

  // Heuristic: we assume the raw_worklist size is small in both DP/Sharding.
  // TODO: If raw_worklist is large, we may need other pruning techniques to
  // reduce the search space.
  std::vector<std::vector<int64>> param_proposals;
  std::vector<int64> proposal_sizes;
  for (auto param : raw_worklist) {
    auto& param_shape = param->shape();
    int rank = param_shape.rank();
    std::vector<int64> feasible_dims;
    // -1: represents 'do replication'.
    // Replication is always feasible strategy.
    feasible_dims.push_back(-1);

    for (int i = 0; i < rank; ++i) {
      VLOG(0) << "Exploring " << param->ToString() << ":" << i;
      if (param_shape.dimensions(i) % num_replicas_) {
        LOG(INFO) << "[WARNING] dimension size " << param_shape.dimensions(i) <<
                " can't be splitted for " << num_replicas_ << " replicas.";
        continue;
      }

      // TODO: Remove this copy of the map
      //HloMutableInstMap<DimStrategy> solution = param_strategy_map;
      Solution solution = m_solution;

      CHECK(solution.hlo_strategy_map_.count(param));
      solution.hlo_strategy_map_.erase(param);
      solution.hlo_strategy_map_.emplace(param, DimStrategy(param->shape(), i, num_replicas_));

      if (StrategySatisfiable(entry, &solution, {param}) &&
          !ConstantSliced(entry, &solution)) {
        feasible_dims.push_back(i);
        VLOG(0) << "Inst name: " << param->name()
                << ", shape:" << param->shape().ToString()
                << ", feasible dim index:" << i;
      }
    } // for (rank)
    proposal_sizes.push_back(feasible_dims.size());
    param_proposals.push_back(feasible_dims);
  } // for (param)

  int64 proposal_space = 0;
  if (!proposal_sizes.empty()) {
    proposal_space = 1;
    for (auto splitable_dim_num : proposal_sizes) {
      proposal_space *= splitable_dim_num;
    }
  }

  //HloMutableInstMap<DimStrategy> best_solution;
  Solution best_solution;
  int64 max_sliced_input = -1, max_sliced = -1;
  for (int64 idx = 0; idx < proposal_space; ++idx) {
    std::vector<int64> proposal_indices;
    ResolveProposalIndices(idx, proposal_sizes, proposal_indices);

    Solution solution = m_solution;
    int64 num_params = raw_worklist.size();
    CHECK_EQ(num_params, proposal_indices.size());
    int64 sliced_input = 0;
    for (int64 i = 0; i < num_params; ++i) {
      auto param = raw_worklist[i];
      auto r = param_proposals[i][proposal_indices[i]];
      VLOG(0) << param->name() << ":" << r;

      sliced_input += (r != -1);

      DCHECK(solution.hlo_strategy_map_.count(param));
      solution.hlo_strategy_map_.erase(param);
      solution.hlo_strategy_map_.emplace(param, DimStrategy(param->shape(), r, num_replicas_));
    }

    std::unordered_set<HloInstruction*> worklist_set(raw_worklist.begin(),
                                                     raw_worklist.end());
    if (!StrategySatisfiable(entry, &solution, worklist_set) ||
        ConstantSliced(entry, &solution)) {
      VLOG(0) << "Proposal->" << idx << " not satisfiable";
    } else {
      int64 sliced_params = 0;
      for (auto param : entry->parameter_instructions()) {
        auto& param_strategy = solution.hlo_strategy_map_.at(param);
        if (!param_strategy.Glue()) {
          VLOG(0) << "Proposal Param->" << param->name() 
                  << " " << param_strategy.ToString();
          ++sliced_params;
        }
      }
      VLOG(0) << "Proposal->" << idx << " satisfiable->" 
              << sliced_params;

      bool replace_strategy = false;
      if (ttype_ == ParallelType::AUTO_DP) {
        // For DP replica case, we only prefer strategies which generate
        // the most #slices of *inputs* (e.g., features, labels, etc.).
        replace_strategy = sliced_input > max_sliced_input;
      } else {
        // For sharding case, we prefer strategies which generate
        //  (1) the most #slices of entry paramter instrutions, or
        //  (2) with the same #sliced of entry params but with more input
        //      tensors(e.g., features, labels, etc.) sliced.
        replace_strategy = (sliced_params > max_sliced ||
            (sliced_params == max_sliced && sliced_input > max_sliced_input));
      }
      if (replace_strategy) {
        // NOTE: strategy spans all instructions in the module.
        // A copy can be very expensive.
        best_solution = std::move(solution);
        // Update the maximum number of sliced params and input that
        // have been seen so far.
        max_sliced = sliced_params > max_sliced ? sliced_params : max_sliced;
        max_sliced_input =
            sliced_input > max_sliced_input ? sliced_input : max_sliced_input;
        VLOG(2) << "replace_strategy-->"
                << " max_sliced=" << max_sliced
                << " max_sliced_input=" << max_sliced_input
                << " sliced_input=" << sliced_input
                << " sliced_params=" << sliced_params;
      }
    }
  }

  // Push for further kParameter coverage in Sharding case
  if (extend_sharding_param && ttype_ == ParallelType::SHARDING) {
    for (auto rit = span_hlo.rbegin(); rit != span_hlo.rend(); ++rit) {
      auto& params = rit->second;
      for (auto param : params) {
        auto& param_strategy = best_solution.hlo_strategy_map_.at(param);
        if (!param_strategy.Glue()) continue;

        auto& param_shape = param->shape();
        for (int64 r = 0; r < param_shape.rank(); ++r) {
          // Skip splitting this dimension
          if (param_shape.dimensions(r) % num_replicas_) continue;

          VLOG(0) << "Extending Param " << param->ToString() 
                  << "->" << r;
          //HloMutableInstMap<DimStrategy> solution = best_solution;
          Solution solution = best_solution;
          solution.hlo_strategy_map_[param] = DimStrategy(param_shape, r, num_replicas_);
          
          if (IncrementalSatisfiable(param, entry, &solution) &&
              !ConstantSliced(entry, &solution)) {
            VLOG(0) << "Extended Param " << param->name() 
                    << "->" << r;
            best_solution = std::move(solution);
            break;
          }
        }
      }
    }
  }

  // Finally, record the best strategy
  auto post_order = entry->MakeInstructionPostOrder();
  for (auto instr : post_order) {
    int stride = 0;
    int size = 0;
    int stride_on_dim = 0;
    int num_replicas = 1;
    int partition_dim = -1;
    if (best_solution.hlo_strategy_map_.count(instr)) {
      auto& strategy = best_solution.hlo_strategy_map_.at(instr);
      stride = strategy.stride_on_elements();
      stride_on_dim = strategy.stride_on_dim();
      partition_dim = strategy.partition_dim();
      num_replicas = num_replicas_;
    }
    std::unique_ptr<DimDistSpec> dim_spec = std::make_unique<DimDistSpec>();
    dim_spec->set_layout_aware_partition(
        stride, stride_on_dim, partition_dim, num_replicas);

    instr->mutable_dist_spec()->AddDimDistSpec(dim_spec);
  }
  return true;
}

void FastSpmdStrategy::ResolveParamList(HloModule* module,
                          std::vector<HloInstruction*>& param_list) {
  // TODO: resolve param list based on user's annotations
  CHECK(0) << "Fix me...";
  auto entry = module->entry_computation();
  for (auto param : entry->parameter_instructions()) {
    if (ShapeUtil::IsScalar(param->shape())) {
      continue;
    }
    // Add candidate for exploration
    param_list.push_back(param);
  }
}

void FastSpmdStrategy::ResolveParamList(HloModule* module,
                          std::vector<HloInstruction*>& raw_worklist,
                          std::map<int, HloMutableInstSet>& span_hlo,
                          int window_size) {
  auto entry = module->entry_computation();
  CHECK(module->variable_map());
  if (ttype_ == ParallelType::AUTO_DP) {
    ResolveParamDPList(module, raw_worklist);
  } else { // sharding
    ResolveParamShardingList(module, span_hlo);
    // N.B., After calling of `ResolveParamShardingList`, trainable variables
    // and their corresponding auxillary variables are divided into different
    // levels (level: distance to root instruction of Entry HloModule.), and the
    // trainable variables always achieve lower level id compared to those
    // auxillary variables.
    //
    // E.g., suppose for one model with Adam optimizer, we have resource
    // variables as follows:
    //  - trainable variables: {arg4.5: ffn/in_weights, arg5.6: ffn/out_weights}
    //  - Adam auxillary resource vars: {arg12.13: ffn/out_weights/Adam,
    //      arg14.15: ffn/in_weights/Adam}
    //  - Adam_1 auxillary resource vars: {arg13.14:ffn/out_weights/Adam_1,
    //      arg15.16: ffn/in_weights/Adam_1}
    //
    // Then `ResolveParamShardingList` returns span_hlo as follows (in *forward*
    // traversal order):
    //   - level_id: 2, value: {arg4.5, arg5.6}
    //   - level_id: 5, value: {arg12.13, arg14.15}
    //   - level_id: 6, value: {arg13.14, arg15.16}
    //
    // As we all know, the auxillary variables' sharding strategies should be
    // consistent with those corresponding trainable variables.  Thus we prefer
    // to traverse the `span_hlo` map in a positive direction instead of reverse
    // direction.

    // for (auto rit = span_hlo.rbegin(); rit != rspan_hlo.end(); ++rit) {
    for (auto rit = span_hlo.begin(); rit != span_hlo.end(); ++rit) {
      auto& param_set = rit->second;
      raw_worklist.insert(
          raw_worklist.end(), param_set.begin(), param_set.end());

      // Param filter
      //std::vector<HloInstruction*> candidates;
      //for (auto param: param_set) {
      //  if (param->shape().rank() < 2) continue;
      //  candidates.push_back(param);
      //}
      //worklist.insert(worklist.end(), candidates.begin(), candidates.end());
      if (raw_worklist.size() >= window_size) break;
    }
  }
}

void FastSpmdStrategy::ResolveParamDPList(HloModule* module,
                       std::vector<HloInstruction*>& param_list) {
  auto entry = module->entry_computation();
  auto variable_map = module->variable_map();
  for (auto param : entry->parameter_instructions()) {
    auto param_no = param->parameter_number();
    if (variable_map->count(param_no) ||
        ShapeUtil::IsScalar(param->shape())) {
      continue;
    }
    
    // Add candidate for exploration
    param_list.push_back(param);
  }
}

void FastSpmdStrategy::ResolveParamShardingList(HloModule* module,
                      std::map<int, HloMutableInstSet>& span_hlo) {
  auto entry = module->entry_computation();
  auto variable_map = module->variable_map();

  std::vector<HloInstruction*> var_params;
  for (auto* param : entry->parameter_instructions()) {
    auto param_no = param->parameter_number();
    if (!variable_map->count(param_no) ||
        ShapeUtil::IsScalar(param->shape()))
      continue;
    var_params.push_back(param);
  }

  HloMutableInstMap<int64> hlo_span;
  std::deque<HloInstruction*> worklist;
  HloMutableInstSet visited;

  HloInstruction* root = entry->root_instruction();
  hlo_span[root] = 0;
  worklist.push_back(root);
  CHECK(visited.insert(root).second);
  while (!worklist.empty()) {
    HloInstruction* hlo = worklist.front();
    worklist.pop_front();

    for (auto operand : hlo->operands()) {
      if (visited.count(operand)) continue;

      // Make sure that *all* trainable varibles can be traversed and covered by
      // `hlo_span` in this level-traversal.
      if (operand->opcode() == HloOpcode::kParameter) {
        auto param_no = operand->parameter_number();
        if (variable_map->count(param_no) &&
            !ShapeUtil::IsScalar(operand->shape())) {
          VLOG(2) << "LEVEL: " << hlo_span[hlo] + 1
                  << ", " << operand->ToString();
          hlo_span[operand] = hlo_span[hlo] + 1;
          CHECK(visited.insert(operand).second);
          continue;
        }
      }

      bool ready = true;
      int max_span = -1;
      for (auto user : operand->users()) {
        if (user != root && user->user_count() == 0) {
          continue;
        }
        // CHECK(user == root || user->user_count() > 0);

        if (!visited.count(user)) {
          ready = false;
          break;
        } else {
          CHECK(hlo_span.count(user));
          if (hlo_span[user] > max_span) max_span = hlo_span[user];
        }
      } // for (users)

      if (ready) {
        CHECK(max_span >= 0);
        CHECK(visited.insert(operand).second);
        hlo_span[operand] = max_span + 1;
        worklist.push_back(operand);
      }
    } // for (operands)
  } // worklist

  //int64 max_span = -1;
  //for (auto param : var_params) {
  //  CHECK(hlo_span.count(param));
  //  if (hlo_span[param] >= max_span) {
  //    max_span = hlo_span[param];
  //  }    
  //}
  //for (auto param : var_params) {
  //  if (hlo_span[param] == max_span) {
  //    param_list.push_back(param);
  //  }
  //}
  //VLOG(0) << "Root params:";
  //for (auto instr : param_list) {
  //  VLOG(0) << instr->name() << " " << instr->shape().ToString();
  //}

  HloMutableInstSet params_set(var_params.begin(), var_params.end());
  for (auto it : hlo_span) {
    if (params_set.count(it.first)) {
      auto& pset = span_hlo[it.second];
      pset.insert(it.first);
    }
  }
  //VLOG(0) << "ALL->";
  //for (auto it = span_hlo.rbegin(); it != span_hlo.rend(); ++it) {
  //  auto span = it->first;
  //  auto& param_set = it->second;
  //  VLOG(0) << it->first << "->";
  //  for (auto p : param_set) {
  //    VLOG(0) << p->name() << ":" << p->shape().ToString() << " ";
  //  }
  //  VLOG(0) << "\n";
  //}
}

AnnotFastSpmdStrategy::AnnotFastSpmdStrategy(/*ParallelType ttype,*/ bool post_layout) 
    : FastSpmdStrategyBase(/*ttype,*/ post_layout) {}

AnnotFastSpmdStrategy::AnnotFastSpmdStrategy(/*ParallelType ttype,*/ bool post_layout,
                          int num_replicas)
    : FastSpmdStrategyBase(/*ttype,*/ post_layout, num_replicas) {}


bool AnnotFastSpmdStrategy::StrategyPlanning(HloModule* module,
                                     bool extend_sharding_param,
                                     int window_size) {
  HloInstMap<SharedDimStrategy> user_annotated_tensors;

  // collect all annotations
  auto entry = module->entry_computation();
  auto post_order = entry->MakeInstructionPostOrder();
  VLOG(0) << "inst with annotation:";
  num_replicas_ = 0;
  for (auto* inst : post_order) {
    // NOTE(zycao): workaround for sharding annotation happened marked on
    // 'tf.broadcast_to' op, which would be not handled by scope stratey.
    if (inst->opcode() == HloOpcode::kBroadcast) continue;
    if (inst->has_sharding()) {
      VLOG(0) << "annotated inst:";
      VLOG(0) << inst->ToString();
      const HloSharding& sharding = inst->sharding();
      VLOG(0) << sharding.ToString();
      auto& tile_assignment = sharding.tile_assignment();
      if (sharding.IsReplicated()) {
        auto num_replicas = tile_assignment.num_elements();
        user_annotated_tensors[inst] = DimStrategy::MakeReplicate(num_replicas);
        continue;
      }

      int total_dev_num = tile_assignment.num_elements();
      VLOG(0) << "total_dev_num=" << total_dev_num;
      // Single device case where tensor is not sharded.
      if (sharding.IsTileMaximal()) {
        if (num_replicas_ == 0) {
          num_replicas_ = 1;
        } else if (num_replicas_ != total_dev_num) {
          VLOG(0) << "Mismatched split number, one is '" << num_replicas_
                  << "', another one is '" << total_dev_num
                  << "' in annotations.";
          return false;
        }
        continue;
      }

      for (auto dev : tile_assignment) {
        VLOG(0) << "tile assign: " << dev;
        std::vector<int64> tile_idx_for_dev = sharding.TileIndexForDevice(dev);
        for (auto tile_idx : tile_idx_for_dev) {
          VLOG(0) << "  tile_idx: " << tile_idx;
        }
      }

      int partition_dim = -1;
      for (int d = 0; d < tile_assignment.num_dimensions(); ++d) {
        if (tile_assignment.dim(d) > 1) {
          if (partition_dim>=0) {
            VLOG(0) << "Only one dimension can be splitted, but found at least two.";
            return false;
          } else {
            partition_dim = d;
          }
        }
      }
      VLOG(2) << "partition_dim=" << partition_dim;

      CHECK(partition_dim>=0) << "partition_dim=" << partition_dim;

      if (num_replicas_ < 1) {
        num_replicas_ = total_dev_num;
      } else if (num_replicas_ != total_dev_num) {
        VLOG(0) << "Mismatched split number, one is '" << num_replicas_
                << "', another one is '" << total_dev_num
                << "' in annotations.";
        return false;
      }

      user_annotated_tensors[inst] =
                    std::make_shared<DimStrategy>(inst->shape(), partition_dim, num_replicas_);
    }
  }

  if (num_replicas_ == 0) {
    VLOG(0) << "No split is specified.";
    return false;
  }

  for (auto& annotation : user_annotated_tensors) {
    if (annotation.second->replicated()) {
      annotation.second->set_num_replicas(num_replicas_);
    }
  }

  VLOG(0) << "entry shards: " << num_replicas_;
  module->record_split_info(num_replicas_, false);

  for (auto& annotation : user_annotated_tensors) {
    VLOG(2) << "user specified inst: " << annotation.first->ToString();
    VLOG(2) << "user specified strategy: " << annotation.second->ToString();
  }

  // For these to be annotated key backward instructions, firstly we fake
  // their hlo_strategy to be "user manually replciated" to skip the forward
  // derivation and backward derivation afterwards.
  // When the fwd/bwd derivation converge, we then decide the sharding strategy
  // of these `bwd_sharding_annotation_insts_` instructions by calling function
  // `UpdateBwdShardingAnnotationInstsStrategy()`.
  for (auto inst : this->bwd_sharding_annotation_insts_) {
    VLOG(2) << "[BWD DEBUG] fake inst with manual replicated strategy->"
            << inst->ToString();
    user_annotated_tensors[inst] = DimStrategy::MakeReplicate(1/*num_replicas*/);
  }

  auto init_strategy_map = user_annotated_tensors;
  HloInstSet annotated_insts =
      InitializeInference(user_annotated_tensors, init_strategy_map);

  CHECK_EQ(annotated_insts.size() + user_annotated_tensors.size(),
           init_strategy_map.size());

  std::vector<HloInstMap<SharedDimStrategy>> strategy_maps;
  for (auto* annotated_inst : annotated_insts) {
    auto strategy = init_strategy_map[annotated_inst];
    VLOG(2) << "current seed inst: " << annotated_inst->ToString()
            << " strategy: " << strategy->ToString();
    HloInstSet start_insts;
    start_insts.insert(annotated_inst);

    HloInstMap<SharedDimStrategy> strategy_map = init_strategy_map;
    Propagate(start_insts, strategy_map, true);
    strategy_maps.emplace_back(strategy_map);
  }

  HloInstMap<SharedDimStrategy> best_strategy_map;
  for (auto& strategy_map : strategy_maps) {
    if (best_strategy_map.empty()) {
      best_strategy_map = strategy_map;
      continue;
    }

    HloInstSet insts;
    for (auto& inst_strategy : best_strategy_map) {
      insts.insert(inst_strategy.first);
    }

    HloInstSet new_insts;
    for (auto& inst_strategy : strategy_map) {
      new_insts.insert(inst_strategy.first);
    }

    std::vector<const HloInstruction*> intersect_insts;
    std::set_intersection(insts.begin(), insts.end(), new_insts.begin(),
                     new_insts.end(), std::back_inserter(intersect_insts));

    std::vector<const HloInstruction*> conflict_insts;
    for (auto* inst : intersect_insts) {
      if (!best_strategy_map[inst]->Match(*strategy_map[inst])) {
        conflict_insts.push_back(inst);
        VLOG(2) << "conflict inst: " << inst->ToString();
        VLOG(2) << "strategy 1: " << best_strategy_map[inst]->ToString();
        VLOG(2) << "strategy 2: " << strategy_map[inst]->ToString();
      }
    }

    // TODO(shiqing.fsq): Note this will not update instruction's strategy which
    // is already placed in `best_strategy_map`.
    best_strategy_map.insert(strategy_map.begin(), strategy_map.end());

    if (conflict_insts.size()) VLOG(2) << "accepted ambiguous strategies: ";
    for (auto* inst : conflict_insts) {
      VLOG(2) << "inst: " << inst->ToString();
      VLOG(2) << "accepted strategy: " << best_strategy_map[inst]->ToString();
    }
  }

  int missed_inst_num = 0;
  int glued_inst_num = 0;
  for (auto* inst : post_order) {
    HloInstMap<SharedDimStrategy>::iterator it;
    it = best_strategy_map.find(inst);
    if (it == best_strategy_map.end()) {
      ++missed_inst_num;
      VLOG(2) << "miss strategy: " << inst->ToString();
    } else if (it->second->Glue()) {
      ++glued_inst_num;
      VLOG(2) << "glue strategy: " << inst->ToString();
    }
  }

  VLOG(2) << "missed_inst_num: " << missed_inst_num;
  VLOG(2) << "glued_inst_num: " << glued_inst_num;

  // post-processing for strategy inference
  HloInstSet more_infer_starts;
  for (auto& inst_strategy : best_strategy_map) {
    more_infer_starts.insert(inst_strategy.first);
  }
  HloInstSet new_insts;
  new_insts = BackwardPropagate(more_infer_starts, best_strategy_map, false);
  more_infer_starts.insert(new_insts.begin(), new_insts.end());
  Propagate(more_infer_starts, best_strategy_map, false);

  VLOG(2) << "After post-processing";
  missed_inst_num = 0;
  glued_inst_num = 0;
  for (auto* inst : post_order) {
    HloInstMap<SharedDimStrategy>::iterator it;
    it = best_strategy_map.find(inst);
    if (it == best_strategy_map.end()) {
      ++missed_inst_num;
      VLOG(2) << "miss strategy: " << inst->ToString();
      best_strategy_map[inst] = std::make_shared<DimStrategy>();
    } else if (it->second->Glue()) {
      ++glued_inst_num;
      VLOG(2) << "glue strategy: " << inst->ToString();
    }
  }

  for (auto& inst_strategy : best_strategy_map) {
    if (inst_strategy.first->opcode() == HloOpcode::kConstant) {
      inst_strategy.second = std::make_shared<DimStrategy>();
    }
  }

  UpdateBwdShardingAnnotationInstsStrategy(best_strategy_map);

  FillStrategyForAllInstructions(module, &best_strategy_map);
  BestStrategyPostProcess(&best_strategy_map, glued_inst_num);
 
  VLOG(2) << "missed_inst_num: " << missed_inst_num;
  VLOG(2) << "glued_inst_num: " << glued_inst_num;

  RecordStrategyToInsts(module, best_strategy_map);
  DumpStrategies(module, best_strategy_map);

  return true;
}

void AnnotFastSpmdStrategy::FillStrategyForAllInstructions(
    const HloModule* module,
    HloInstMap<SharedDimStrategy>* hlo_strategy_map) {
  auto* entry = module->entry_computation();
  for (const HloInstruction* instr : entry->MakeInstructionPostOrder()) {
    if (hlo_strategy_map->find(instr) != hlo_strategy_map->end()) continue;
    (*hlo_strategy_map)[instr] = std::make_shared<DimStrategy>();
  }
}

void AnnotFastSpmdStrategy::ResetUnDivisibleStrategy(
    HloInstMap<SharedDimStrategy>* best_strategy_map,
    int& glued_inst_num) {
  for (auto& inst_strategy : *best_strategy_map) {
    auto inst = inst_strategy.first;
    auto& shape = inst->shape();
    if (shape.IsTuple()) continue;
    auto& dim_str = inst_strategy.second;
    if (dim_str->Glue() || dim_str->IsPartial() || dim_str->replicated() ||
        !(shape.dimensions(dim_str->partition_dim()) % dim_str->num_replicas())) continue;
    dim_str = std::make_shared<DimStrategy>();
    glued_inst_num ++;
  }
}

void AnnotFastSpmdStrategy::BestStrategyPostProcess(
    HloInstMap<SharedDimStrategy>* best_strategy_map,
    int& glued_inst_num) {
  // 1. Reset all DimStrategy with default for undividend instructions
  ResetUnDivisibleStrategy(best_strategy_map, glued_inst_num);

  // 2. Reset all special instructions with default which are not guaranteed
  // by bi-verification inference. 
  for (auto& inst_strategy : *best_strategy_map) {
    auto inst = inst_strategy.first;
    auto& dim_str = inst_strategy.second;
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
          dim_str = std::make_shared<DimStrategy>();
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
            VLOG(0) << "w_dim_size=" << w_dim_size;
            dim_str = std::make_shared<DimStrategy>();
            VLOG(0) << "Reset kReduceWindow from sharding of reduced dimension to "
                    << "replication(i.e., Glue strategy): " << inst->ToString();
            glued_inst_num ++;
          }
        }

        auto* op = inst->operand(0);
        auto& op_dim_str = (*best_strategy_map)[op];
        if (op_dim_str->Glue()) continue;
        auto op_dim_to_slice = op_dim_str->partition_dim();
        auto op_w_dim = window.dimensions(op_dim_to_slice);
        auto op_w_dim_size = op_w_dim.size();
        if (op_w_dim_size > 1) {
          op_dim_str = std::make_shared<DimStrategy>();
          glued_inst_num ++;
        }
        break;
      }
      default : {}
    }
  }
}

void AnnotFastSpmdStrategy::UpdateBwdShardingAnnotationInstsStrategy(
    HloInstMap<SharedDimStrategy>& best_strategy_map) {
  for (auto inst : this->bwd_sharding_annotation_insts_) {
    CHECK(best_strategy_map.count(inst) &&
          (best_strategy_map[inst]->replicated() ||
           best_strategy_map[inst]->Glue()));
    for (int64 input_idx = 0; input_idx < inst->operand_count(); ++input_idx) {
      auto operand = inst->operand(input_idx);
      CHECK(best_strategy_map.count(operand));
      auto input_strategy = best_strategy_map[operand];
      auto inferred_inst_strategy =
          StrategyUtil::ForwardInfer(inst, *input_strategy, input_idx);
      auto inferred_strategy = inferred_inst_strategy[inst];

      // Skip operand with glue strategy.
      if (inferred_strategy->Glue()) continue;

      // Update bwd_sharding_annotation_inst with meaningful sharding
      // strategy inferred from operand.
      best_strategy_map[inst] = inferred_strategy;
      VLOG(2) << "[LAZY UPDATE] inst->" << inst->ToString()
              << " inferred_strategy->" << inferred_strategy->ToString()
              << " operand index=" << input_idx
              << " from operand->" << operand->ToString();

      HloInstSet start_insts {inst};

      auto cur_strategy_map = best_strategy_map;
      Propagate(start_insts, cur_strategy_map, false);
      for (auto inst_strategy : cur_strategy_map) {
        auto cur_inst = inst_strategy.first;
        CHECK(best_strategy_map.count(cur_inst));
        auto old_strategy = best_strategy_map[cur_inst];
        auto new_strategy = cur_strategy_map[cur_inst];
        if (!old_strategy->Glue()) continue;
        CHECK(!new_strategy->replicated());
        if (new_strategy->Glue()) continue;
        // Update related instruction strategies.
        VLOG(2) << "[LAZY UPDATE] update inst->" << cur_inst->ToString()
                << " with new strategy: "
                << cur_strategy_map[cur_inst]->ToString();
        best_strategy_map[cur_inst] = new_strategy;
      }

      // Already enough if we get the first none-glue strategy from one of
      // its operands.
      break;
    }
  }
}

HloInstSet AnnotFastSpmdStrategy::InitializeInference(
      HloInstMap<SharedDimStrategy>& user_annotated_tensors,
      HloInstMap<SharedDimStrategy>& strategy_map) {
  HloInstSet infered_insts;

  for (auto& user_annotation : user_annotated_tensors) {
    HloInstSet sub_infered_insts = InferUsersFromInst(
        user_annotation.first, *user_annotation.second, strategy_map);
    VLOG(2) << "InferUsersFromInst: from annotated inst: " << user_annotation.first->ToString();
    VLOG(2) << "annotated strategy: " << user_annotation.second->ToString();
    VLOG(2) << "infer following:";
    for (auto* infered_inst : sub_infered_insts) {
      VLOG(2) << "Inst: " << infered_inst->ToString()
              << " strategy: " << strategy_map[infered_inst]->ToString();
    }
    infered_insts.insert(sub_infered_insts.begin(), sub_infered_insts.end());
  }

  return std::move(infered_insts);
}

bool AnnotFastSpmdStrategy::Propagate(
      const HloInstSet& start_insts,
      HloInstMap<SharedDimStrategy>& strategy_map,
      bool disable_gradient_inst) {
  bool changed = false;

  HloInstSet forward_start_insts, backward_start_insts;
  forward_start_insts = start_insts;
  int iteration = 0;
  while (!forward_start_insts.empty()) {
    VLOG(2) << "iteration: " << iteration;
    // 1. Forward inference
    backward_start_insts = ForwardPropagate(forward_start_insts,
                                            strategy_map);
    if (!backward_start_insts.empty()) {
      changed = true;

      // 2. Backward inference
      forward_start_insts = BackwardPropagate(backward_start_insts,
                                              strategy_map,
                                              disable_gradient_inst);
    } else {
      forward_start_insts.clear();
    }

    ++iteration;
  }

  return changed;
}

HloInstSet AnnotFastSpmdStrategy::ForwardPropagate(
    const HloInstSet& start_insts,
    HloInstMap<SharedDimStrategy>& strategy_map) {
  HloInstSet infered_insts;
  VLOG(2) << "*****************\nforward propagate:\n*********************";
  for (auto* start_inst : start_insts) {
    //VLOG(0) << "  forward for inst: " << start_inst->ToString();
    HloInstSet sub_infered_insts = InferFromInst(start_inst, strategy_map);
    infered_insts.insert(sub_infered_insts.begin(), sub_infered_insts.end());
  }

  return std::move(infered_insts);
}

// Infer as much as possible for **determined** propagation.
HloInstSet AnnotFastSpmdStrategy::InferUsersFromInst(
                const HloInstruction* inst,
                const DimStrategy& inst_strategy,
                HloInstMap<SharedDimStrategy>& strategy_map) {
  HloInstSet infered_insts;

  // User annotated replicated instruction through xla_sharding.replicate()
  // API, which cannot be modified during later sharding propagation.
  if (inst_strategy.replicated()) {
    return infered_insts;
  }

  VLOG(0) << "infer users of " << inst->name();

  for (auto* user : inst->users()) {
    if (user->opcode() == HloOpcode::kTuple) continue;
    HloInstMap<SharedDimStrategy>::iterator user_it;
    user_it = strategy_map.find(user);
    if (user_it != strategy_map.end() && !user_it->second->Glue()) {
      continue;
    }

    // determine user's strategy
    CHECK(!inst_strategy.Glue());

    int input_idx = user->operand_index(inst);
    auto infered_strategy_map = StrategyUtil::ForwardInfer(
                                        user, inst_strategy, input_idx);

    for (auto& infered_inst_strategy : infered_strategy_map) {
      const HloInstruction* infered_inst = infered_inst_strategy.first;
      SharedDimStrategy infered_strategy = infered_inst_strategy.second;
      if (infered_strategy->Glue()) {
        continue;
      }
      HloInstMap<SharedDimStrategy>::iterator strategy_it;
      strategy_it = strategy_map.find(infered_inst);
      if (strategy_it == strategy_map.end() || strategy_it->second->Glue()) {
        if (infered_inst == user) {
          // Dot instruction may infer another input from one input
          // drop out the infered input
          strategy_map[infered_inst] = infered_strategy;
          infered_insts.insert(infered_inst);
          CHECK(!user->has_sharding()) << " user: " << user->name();
          // Propgate sharding annotation to users with determined sharding
          // strategy.
          //user->set_sharding(inst->sharding());
          auto recursive_insts =
              InferUsersFromInst(infered_inst, *infered_strategy, strategy_map);
          for (auto inst : recursive_insts) {
            infered_insts.insert(inst);
          }
        }
      } else {
        if (!infered_strategy->Match(*strategy_map[infered_inst])) {
          VLOG(0) << "At least two dimensions are specified to be splitted for instruction";
          VLOG(0) << infered_inst->ToString();
        }
      }
    }
  }
  return std::move(infered_insts);
}

// recursively infer
HloInstSet AnnotFastSpmdStrategy::InferFromInst(
       const HloInstruction* inst,
       HloInstMap<SharedDimStrategy>& strategy_map) {
  HloInstSet infered_insts;
  CHECK(strategy_map.count(inst))
      << inst->name() << " should exist in strategy_map!";
  auto inst_strategy = strategy_map[inst];
  if (inst_strategy->replicated() || inst_strategy->Glue()) {
    return std::move(infered_insts);
  }
  CHECK(!inst_strategy->Glue());

  for (auto* user : inst->users()) {
    if (user->opcode() == HloOpcode::kTuple) continue;
    if (strategy_map.count(user)) {
      auto user_strategy = strategy_map[user];
      if (!user_strategy->Glue()) {
        continue;
      }
    }

    // determine user's strategy
    int input_idx = user->operand_index(inst);
    auto infered_strategy_map = StrategyUtil::ForwardInfer(
                                        user, *inst_strategy, input_idx);

    VLOG(2) << "forward: from inst: " << inst->ToString();
    VLOG(2) << "forward:     strategy: " << inst_strategy->ToString();

    for (auto& infered_inst_strategy : infered_strategy_map) {
      const HloInstruction* infered_inst = infered_inst_strategy.first;
      SharedDimStrategy infered_strategy = infered_inst_strategy.second;
      if (infered_strategy->Glue()) {
        continue;
      }
      HloInstMap<SharedDimStrategy>::iterator strategy_it;
      strategy_it = strategy_map.find(infered_inst);
      auto strategy = strategy_it->second;
      if (strategy_it == strategy_map.end() || strategy->Glue()) {
        strategy_map[infered_inst] = infered_strategy;
        VLOG(2) << "forward:      strategy: " << infered_strategy->ToString();
        infered_insts.insert(infered_inst);
        if (infered_inst == user) {
          // Dot instruction may infer another input from one input
          HloInstSet child_infered_insts = InferFromInst(user, strategy_map);
          infered_insts.insert(child_infered_insts.begin(), child_infered_insts.end());
        }
      } else {
        //CHECK(infered_strategy.Match(strategy_map[infered_inst]));
      }
    }
  }

  return std::move(infered_insts);
}

HloInstSet AnnotFastSpmdStrategy::BackwardPropagate(
                    const HloInstSet& start_insts,
                    HloInstMap<SharedDimStrategy>& strategy_map,
                    bool disable_gradient_inst) {
  HloInstSet infered_insts;
  VLOG(2) << "*********************\nbackward propagate:\n************************";
  for (auto* start_inst : start_insts) {
    //VLOG(0) << "  backward for inst: " << start_inst->ToString();
    HloInstSet sub_infered_insts =
        InferFromUser(start_inst, strategy_map, disable_gradient_inst);
    infered_insts.insert(sub_infered_insts.begin(), sub_infered_insts.end());
  }

  return std::move(infered_insts);
}

HloInstSet AnnotFastSpmdStrategy::InferFromUser(
       const HloInstruction* inst,
       HloInstMap<SharedDimStrategy>& strategy_map,
       bool disable_gradient_inst) {
  HloInstSet infered_insts;

  if (disable_gradient_inst) {
    if (backward_insts_.find(inst) == backward_insts_.end()) {
      return infered_insts;
    }
  }

  CHECK(strategy_map.count(inst))
      << inst->name() << " does not exist in current strategy_map!";
  auto inst_strategy = strategy_map[inst];
  CHECK(!inst_strategy->Glue()) << "inst: " << inst->ToString() << ", " << inst_strategy->ToString();

  // Short circuit for replicated instruction in backward propagation.
  if (inst_strategy->replicated()) {
    return std::move(infered_insts);
  }

  VLOG(2) << "[Forward] inst: " << inst->ToString()
          << " strategy: " << inst_strategy->ToString();

  for (int64 op_idx = 0; op_idx < inst->operand_count(); ++op_idx) {
    auto* op = inst->operand(op_idx);
    HloInstMap<SharedDimStrategy>::iterator op_it;
    op_it = strategy_map.find(op);
    if (op_it != strategy_map.end() && !op_it->second->Glue()) {
      continue;
    }

    auto infered_strategy = StrategyUtil::BackInfer(inst, *inst_strategy, op_idx);
    if (infered_strategy->Glue()) {
      continue;
    }

    VLOG(2) << "[Backward] inst: " << op->ToString()
            << ", infered strategy: " << infered_strategy->ToString();

    strategy_map[op] = infered_strategy;
    infered_insts.insert(op);

    HloInstSet sub_infered_insts =
        InferFromUser(op, strategy_map, disable_gradient_inst);

    infered_insts.insert(sub_infered_insts.begin(), sub_infered_insts.end());
  }

  return std::move(infered_insts);
}

void AnnotFastSpmdStrategy::RecordStrategyToInsts(HloModule* module,
    const HloInstMap<SharedDimStrategy>& strategy_map) {
  auto* entry = module->entry_computation();
  for (auto* instr : entry->MakeInstructionPostOrder()) {
    int64 stride = 0;
    int64 stride_on_dim = 0;
    int num_replicas = num_replicas_;
    int partition_dim = -1;
    bool partial = false;
    if (strategy_map.find(instr) != strategy_map.end()) {
      auto& strategy = strategy_map.at(instr);
      stride = strategy->stride_on_elements();
      stride_on_dim = strategy->stride_on_dim();
      partition_dim = strategy->partition_dim();
      num_replicas = strategy->num_replicas();
      partial = strategy->IsPartial();

      VLOG(0) << instr->ToString();
      VLOG(0) << "  " << strategy->ToString();
    } else {
      VLOG(0) << "No strategy: " << instr->ToString();
    }

    
    std::unique_ptr<DimDistSpec> dim_spec = std::make_unique<DimDistSpec>();
    dim_spec->set_partial(partial);
    dim_spec->set_layout_aware_partition(
        stride, stride_on_dim, partition_dim, num_replicas);

    instr->mutable_dist_spec()->AddDimDistSpec(dim_spec);

    VLOG(0) << "inst with strategy: " << instr->ToString();
  }
}

void AnnotFastSpmdStrategy::DumpStrategies(
    const HloModule* module,
    const HloInstMap<SharedDimStrategy>& strategy_map) {
  auto entry = module->entry_computation();
  auto post_order = entry->MakeInstructionPostOrder();

  int no_strategy_num = 0;
  int with_strategy_num = 0;
  std::string strategies;
  strategies = "\ntotal instruction num: " + std::to_string(post_order.size());
  for (auto* inst : post_order) {
    if (strategy_map.find(inst) != strategy_map.end()) {
      strategies += "\n" + inst->ToString();
      strategies += "\nStrategy: " + strategy_map.at(inst)->ToString();
      ++with_strategy_num;
    } else {
      strategies += "\nNo strategy: " + inst->ToString();
      ++no_strategy_num;
    }
  }

  strategies += "\nnum of insts with strategy: " + std::to_string(with_strategy_num);
  strategies += "\nnum of insts without strategy: " + std::to_string(no_strategy_num);

  VLOG(0) << strategies;

  tensorflow::Env* env = tensorflow::Env::Default();
  Status status = tensorflow::WriteStringToFile(env, "strategies.txt", strategies);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write strategies to strategies.txt: " << status;
  }
}




}  // namespace xla
