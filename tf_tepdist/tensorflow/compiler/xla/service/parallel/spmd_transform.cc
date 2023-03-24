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

#include "tensorflow/compiler/xla/service/parallel/spmd_transform.h"

#include <algorithm>
#include <fstream>
#include <list>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/parallel/performance_utils.h"
#include "tensorflow/compiler/xla/service/parallel/resolve_utils.h"
#include "tensorflow/compiler/xla/service/parallel/utils.h"

namespace xla {

const string kAllToAllType = "AllToAll";
const string kAllReduceType = "AllReduce";
const string kAllGatherType = "AllGather";
const string kDynamicSliceType = "DynamicSlice";
const string kReshardType = "Reshard";
const string kSumStr = "Sum";
const string kMinStr = "Min";
const string kMaxStr = "Max";
const string kProdStr = "Prod";
const string kAndStr = "And";

namespace {

Shape MakeShape(const Shape& old_shape,
                std::vector<int64>& dims) {
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

Shape MakeNewPrimShape(const HloInstruction* inst, int num_replicas,
                       const DimDistSpec& dist_spec) {
  // create non-tuple shape
  CHECK_NE(inst, nullptr);
  auto old_shape = inst->shape();
  CHECK(!old_shape.IsTuple());

  int rank = old_shape.rank();

  std::vector<int64> new_dims;
  int64 dim_to_slice = -1, new_dim_size = -1;
  if (dist_spec.stride() == -1) {
    CHECK(dist_spec.partition_dim() >= 0);
    CHECK(inst->opcode() == HloOpcode::kReduceWindow);

    dim_to_slice = dist_spec.partition_dim();
    const Window& window = inst->window();
    auto w_dim = window.dimensions(dim_to_slice);
    auto p_low = w_dim.padding_low();
    auto p_high = w_dim.padding_high();
    auto size = w_dim.size();
    new_dim_size = inst->operand(0)->shape().dimensions(dim_to_slice) +
        p_low + p_high - size + 1;
  } else {
    DimStrategy inst_strategy(dist_spec);
    dim_to_slice = inst_strategy.partition_dim();
    CHECK(dim_to_slice >= 0 && dim_to_slice < rank);
    int64 old_dim_size = old_shape.dimensions(dim_to_slice);
    CHECK(old_dim_size % num_replicas == 0);
    new_dim_size = old_dim_size / num_replicas;
  }
  new_dims.reserve(rank);
  for (int r = 0; r < rank; ++r) {
    if (r == dim_to_slice) {
      new_dims.emplace_back(new_dim_size);
    } else {
      new_dims.emplace_back(old_shape.dimensions(r));
    }
  }

  return DistUtil::MakeShape(old_shape, new_dims);
}

Shape MakeNewShape(const Shape& old_shape, const int num_replicas,
                   const DimDistSpec& dist_spec,
                   const HloInstruction* inst) {
  if (old_shape.IsTuple()) {
    std::vector<Shape> new_sub_shapes;
    for (auto& old_sub_shape : old_shape.tuple_shapes()) {
      new_sub_shapes.emplace_back(
          MakeNewShape(old_sub_shape, num_replicas, dist_spec, inst));
    }

    return ShapeUtil::MakeTupleShape(new_sub_shapes);
  } else {
    return MakeNewPrimShape(inst, num_replicas, dist_spec);
  }
}

gpu::ReshardBackendConfig CreateReshardBackendConfig(
    const DimStrategy& to_strategy, const DimStrategy& from_strategy,
    int num_replicas) {
  gpu::ReshardBackendConfig backend_config;
  backend_config.set_split_dim(to_strategy.partition_dim());
  backend_config.set_concat_dim(from_strategy.partition_dim());
  backend_config.set_split_stride_on_dim(to_strategy.stride_on_dim());
  backend_config.set_concat_stride_on_dim(from_strategy.stride_on_dim());
  backend_config.set_num_replicas(num_replicas);
  if (from_strategy.Glue() && !to_strategy.Glue()) {
    backend_config.set_reshard_type(kDynamicSliceType);
  } else if (!from_strategy.Glue() && to_strategy.Glue()) {
    backend_config.set_reshard_type(kAllGatherType);
  } else if (!from_strategy.Match(to_strategy)) {
    backend_config.set_reshard_type(kAllToAllType);
  } else {
    CHECK(0 && "Never reach here!");
  }
  return backend_config;
}

gpu::AllReduceBackendConfig CreateAllReduceBackendConfig(
    string reduction_type, int num_replicas) {
  gpu::AllReduceBackendConfig backend_config;
  *backend_config.mutable_reduction_type() = reduction_type;
  backend_config.set_num_replicas(num_replicas);
  return backend_config;
}

std::string GetReductionTypeStr(HloOpcode opcode) {
  switch(opcode) {
    case HloOpcode::kAdd: return kSumStr;
    case HloOpcode::kMinimum: return kMinStr;
    case HloOpcode::kMaximum: return kMaxStr;
    case HloOpcode::kAnd: return kAndStr;
    default: CHECK(0 && "Unhandled reduction type");
  }

  return "";
}

std::string BackTraceReductionType(HloInstruction* instr) {
  bool found = false;
  HloInstruction* curr_instr = instr;
  std::string reduction_type;
  while (!found) {
    switch (curr_instr->opcode()) {
      case HloOpcode::kReshape:
      case HloOpcode::kConvert: {
        curr_instr = curr_instr->mutable_operand(0);
        break;
      }

      case HloOpcode::kReduce: {
        auto* root = DynCast<HloReduceInstruction>(curr_instr)->to_apply()->root_instruction();
        reduction_type = GetReductionTypeStr(root->opcode());
        found = true;
        break;
      }

      case HloOpcode::kScatter: {
        auto* root = DynCast<HloScatterInstruction>(curr_instr)->to_apply()->root_instruction();
        reduction_type = GetReductionTypeStr(root->opcode());
        found = true;
        break;
      }

      case HloOpcode::kAdd:
      case HloOpcode::kDot:
      case HloOpcode::kConvolution: {
        reduction_type = kSumStr;
        found = true;
        break;
      }

      case HloOpcode::kCustomCollective: {
        HloInstruction* op = instr->mutable_operand(0);
        return BackTraceReductionType(op);
      }

      default: {
        VLOG(0) << "curr_instr : " << curr_instr->ToString();
        CHECK(0 && "Never reach here.");
      }
    }
  }

  CHECK(found);
  return reduction_type;
}

}

bool SpmdTransform::ShapeCheck(HloComputation* computation) {
  auto post_order = computation->MakeInstructionPostOrder();
  auto module = computation->parent();
  int split_ordinal = split_ordinal_;
  for (auto hlo : post_order) {
    VLOG(2) << "ShapeCheck->" << hlo->ToString();

    auto hlo_shape = hlo->shape();
    if (hlo_shape.IsTuple()) {
      hlo_shape = ShapeUtil::GetTupleElementShape(hlo_shape, 0);
    }
    // Make sure to use size instead of stride after rewrite
    // (TODO) Revive split_dim by size is not reasonable
    DimStrategy hlo_strategy(*hlo->dist_spec().get_dim_spec(split_ordinal_));
    auto dim_to_slice = hlo_strategy.partition_dim();
    VLOG(2) << "strategy:" << hlo_strategy.ToString();

    auto check_and_verify = [hlo, dim_to_slice, split_ordinal](HloInstruction* input,
                                                const Shape& hlo_shape) {
      auto& input_shape = input->shape();
      DimStrategy input_strategy(*input->dist_spec().get_dim_spec(split_ordinal));
      auto input_dim = input_strategy.partition_dim();
      if (input_dim >= 0) {
        CHECK(input_shape.dimensions(input_dim) ==
              hlo_shape.dimensions(dim_to_slice));
      }
    };
    switch (hlo->opcode()) {
      case HloOpcode::kReplicaId:
      case HloOpcode::kPartitionId:
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kIota:
      case HloOpcode::kRng:
        break;

      case HloOpcode::kSort:
      case HloOpcode::kTuple: {
        auto num_subshapes = hlo_shape.tuple_shapes_size();
        for (int i = 0; i < num_subshapes; ++i) {
          auto operand = hlo->operand(i);
          CHECK(ShapeUtil::Equal(hlo_shape.tuple_shapes()[i], operand->shape()));
        }
        break;
      }

      case HloOpcode::kBroadcast: {
        auto& input_shape = hlo->operand(0)->shape();
        auto& hlo_shape = hlo->shape();
        int64 input_rank = input_shape.rank();
        CHECK(input_rank == hlo->dimensions().size());
        for (int64 i = 0; i < input_rank; ++i) {
          CHECK(hlo_shape.dimensions(hlo->dimensions(i)) ==
                input_shape.dimensions(i));
        }
        break;
      }

      case HloOpcode::kSlice: {
        break;
      }

      case HloOpcode::kPad: {
        if (dim_to_slice >= 0) {
          auto& input_shape = hlo->operand(dim_to_slice)->shape();
          CHECK(input_shape.dimensions(dim_to_slice) ==
                hlo_shape.dimensions(dim_to_slice));
        }
        break;
      }

      case HloOpcode::kGather: {
        if (dim_to_slice >= 0) {
          check_and_verify(hlo->mutable_operand(0), hlo->shape()); // data
        }
        check_and_verify(hlo->mutable_operand(1), hlo->shape()); // indices
        break;
      }

      case HloOpcode::kScatter: {
        auto hlo_rank = hlo_shape.rank();
        auto updates = hlo->mutable_operand(1); // updates
        auto& updates_shape = updates->shape();
        auto& updates_dist_spec = updates->dist_spec();
        DimStrategy update_strategy(*updates_dist_spec.get_dim_spec(split_ordinal_));
        auto updates_dim = update_strategy.partition_dim();
        auto& scatter_dims = hlo->scatter_dimension_numbers(); 
        auto& update_window_dims = scatter_dims.update_window_dims();
        if (updates_dim >= update_window_dims[0]) {
          auto right_delta = updates_shape.rank() - updates_dim;
          CHECK(updates_shape.dimensions(updates_dim) ==
                hlo_shape.dimensions(hlo_rank - right_delta));
        }
        break;
      }

      case HloOpcode::kConcatenate: {
        auto& hlo_shape = hlo->shape();
        int rank = hlo_shape.rank();
        int concat_dim = hlo->dimensions(0);
        int64 total_elements = 0;
        for (auto& operand : hlo->operands()) {
          total_elements += operand->shape().dimensions(concat_dim);
        }
        CHECK(total_elements == hlo_shape.dimensions(concat_dim));        
        for (int i = 0; i < rank; ++i) {
          auto input = hlo->mutable_operand(0);
          auto& input_shape = input->shape();
          if (i == concat_dim) {
            continue;
          }
          CHECK(hlo_shape.dimensions(i) == input_shape.dimensions(i));
        }
        break;
      }

      case HloOpcode::kDot: {
        const DotDimensionNumbers& dnums = hlo->dot_dimension_numbers();
        int num_batch_dims = dnums.lhs_batch_dimensions_size();
        auto& hlo_shape = hlo->shape();
        int dot_dims = hlo_shape.rank();

        auto M = hlo_shape.dimensions(num_batch_dims);
        auto N = hlo_shape.dimensions(num_batch_dims+1);

        auto lhs = hlo->mutable_operand(0);
        auto& lhs_shape = lhs->shape();
        int lhs_contracting_dim = dnums.lhs_contracting_dimensions(0);
        const bool lhs_trans = lhs_contracting_dim == (dot_dims-2);

        auto rhs = hlo->mutable_operand(1);
        auto& rhs_shape = rhs->shape();
        int rhs_contracting_dim = dnums.rhs_contracting_dimensions(0);
        const bool rhs_trans = rhs_contracting_dim == (dot_dims-1);

        for (int i = 0; i < num_batch_dims; ++i) {
          VLOG(1) << "Shape Check instr -> " << hlo->ToString();
          CHECK(lhs_shape.dimensions(i) == rhs_shape.dimensions(i));
        }
        int64 K = 0;
        if (!lhs_trans) {
          CHECK(M == lhs_shape.dimensions(num_batch_dims));
          K = lhs_shape.dimensions(num_batch_dims+1);
        } else {
          K = lhs_shape.dimensions(num_batch_dims);
          CHECK(M == lhs_shape.dimensions(num_batch_dims+1));
        }

        if (!rhs_trans) {
          CHECK(K == rhs_shape.dimensions(num_batch_dims));
          CHECK(N == rhs_shape.dimensions(num_batch_dims+1));
        } else {
          CHECK(N == rhs_shape.dimensions(num_batch_dims));
          CHECK(K == rhs_shape.dimensions(num_batch_dims+1));
        }
        break;
      }

      case HloOpcode::kTranspose: {
        auto& hlo_shape = hlo->shape();
        auto& input_shape = hlo->mutable_operand(0)->shape();
        int hlo_rank = hlo_shape.rank();
        auto hlo_name = hlo->metadata().op_name();
        auto input_name = hlo->operand(0)->metadata().op_name();

        for (int i = 0; i < hlo_rank; ++i) {
          int j = hlo->dimensions(i);
          CHECK(hlo_shape.dimensions(i) == input_shape.dimensions(j));
        }
        break;
      }

      case HloOpcode::kDynamicSlice: {
        auto* d_slice = DynCast<HloDynamicSliceInstruction>(hlo);
        auto& hlo_shape = hlo->shape();
        for (int i = 0; i < hlo_shape.rank(); ++i) {
          CHECK(hlo_shape.dimensions(i) == d_slice->slice_sizes(i));
        }
        break;
      }

      case HloOpcode::kAbs:
      case HloOpcode::kSin:
      case HloOpcode::kCos:
      case HloOpcode::kExp:
      case HloOpcode::kLog:
      case HloOpcode::kCopy:
      case HloOpcode::kSqrt:
      case HloOpcode::kRsqrt:
      case HloOpcode::kPower:
      case HloOpcode::kTanh:
      case HloOpcode::kNegate:
      case HloOpcode::kConvert:
      case HloOpcode::kReshape:
      case HloOpcode::kIsFinite:
      case HloOpcode::kAllReduce:
      case HloOpcode::kDAPPLEAllReduce:
      case HloOpcode::kBitcastConvert: {
        auto input = hlo->mutable_operand(0);
        CHECK(ShapeUtil::ElementsIn(hlo->shape()) ==
              ShapeUtil::ElementsIn(input->shape()));
        break;
      }

      case HloOpcode::kReduce: {
        Shape hlo_shape;
        if (hlo->shape().IsTuple()) {
          // E.g. %reduce.66 = (f32[10,15]{1,0}, s32[10,15]{1,0}) reduce(
          //      f32[10,15,6]{2,1,0} %divide.53, s32[10,15,6]{2,1,0} %iota.56,
          //      f32[] %constant.35, s32[] %constant.55), dimensions={2},
          //      to_apply=%minmax_func.57,
          //      metadata={op_type="ArgMax" op_name="ffn/top2gating/ArgMax"}
          auto& raw_hlo_shape = hlo->shape();
          hlo_shape = raw_hlo_shape.tuple_shapes(0);
          // Check whether that the tuple shapes are all the same.
          for (auto& shape : raw_hlo_shape.tuple_shapes()) {
            CHECK(ShapeUtil::EqualIgnoringElementType(shape, hlo_shape));
          }
        } else {
          hlo_shape = hlo->shape();
          CHECK(hlo_shape.IsArray()) << "Unhandled Shape in kReduce!";
        }

        auto input = hlo->mutable_operand(0);
        auto& input_shape = input->shape();

        auto reduce_dims = hlo->dimensions();
        std::set<int> reduce_dims_set(reduce_dims.begin(), reduce_dims.end());
        int out_idx = 0;
        for (int i = 0; i < input_shape.rank(); ++i) {
          if (reduce_dims_set.count(i)) continue;
          CHECK(hlo_shape.dimensions(out_idx++) == input_shape.dimensions(i));
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
        auto lhs = hlo->mutable_operand(0);
        auto rhs = hlo->mutable_operand(1);
        CHECK (ShapeUtil::ElementsIn(hlo->shape()) ==
              ShapeUtil::ElementsIn(lhs->shape()));
        CHECK (ShapeUtil::ElementsIn(hlo->shape()) ==
              ShapeUtil::ElementsIn(rhs->shape()));
        break;
      }

      case HloOpcode::kSelect: {
        auto pred = hlo->mutable_operand(0);
        auto lhs = hlo->mutable_operand(1);
        auto rhs = hlo->mutable_operand(2);
        CHECK(ShapeUtil::ElementsIn(hlo->shape()) ==
              ShapeUtil::ElementsIn(pred->shape()));
        CHECK(ShapeUtil::ElementsIn(hlo->shape()) ==
              ShapeUtil::ElementsIn(lhs->shape()));
        CHECK(ShapeUtil::ElementsIn(hlo->shape()) ==
              ShapeUtil::ElementsIn(rhs->shape()));
        break;
      }

      case HloOpcode::kReduceWindow:
        // Example with padding:
        // %reduce-window.82 = f32[10,16,6]{2,1,0}
        //   reduce-window(f32[10,15,6]{2,1,0} %select.75, f32[] %constant.29),
        //   window={size=1x15x1 pad=0_0x15_0x0_0}, to_apply=%add_float_.78,
        //   metadata={op_type="Cumsum" op_name="ffn/top2gating/Cumsum"}
        //
        // Ref https://www.tensorflow.org/xla/operation_semantics#reducewindow
        // for more details.
      case HloOpcode::kSelectAndScatter: {
        auto& input_shape = hlo->operand(0)->shape();
        const Window& window = hlo->window();
        for (int64 i = 0; i < input_shape.rank(); ++i) {
          auto w_dim = window.dimensions(i);
          if (w_dim.size() == 1 || w_dim.stride() == 1 ) {
            auto p_low = w_dim.padding_low();
            auto p_high = w_dim.padding_high();
            auto size = w_dim.size();
            CHECK_EQ(input_shape.dimensions(i) + p_low + p_high - size + 1,
                     hlo_shape.dimensions(i));
          }
        }
        break;
      }

      case HloOpcode::kGetTupleElement: {
        auto input = hlo->operand(0);
        auto input_shape = input->shape();
        int64 tuple_index = hlo->tuple_index();
        auto hlo_shape = input_shape;
        if (input_shape.IsTuple()) {
          hlo_shape = ShapeUtil::GetTupleElementShape(input_shape, tuple_index);
          CHECK(ShapeUtil::Equal(hlo_shape, hlo->shape()));
        } else {
          // For cases where input and output share the same shape while with
          // different element types. E.g.,
          //   %get-tuple-element = s32[5,15]{1,0} get-tuple-element(
          //     f32[5,15]{1,0} %reduce.8), index=1, metadata={op_type="ArgMax"}
          CHECK(ShapeUtil::EqualIgnoringElementType(hlo_shape, hlo->shape()));
        }
        break;
      }

      case HloOpcode::kCustomCall: {
        auto& conv_dnums = hlo->convolution_dimension_numbers();
        auto input_batch_dim = conv_dnums.input_batch_dimension();
        auto output_batch_dim = conv_dnums.output_batch_dimension();
        auto lhs = hlo->mutable_operand(0);
        auto rhs = hlo->mutable_operand(1);
        const auto& target = hlo->custom_call_target();
        if (target != "__cudnn$convBackwardFilter") {
          auto& hlo_shape = ShapeUtil::GetTupleElementShape(hlo->shape(), 0);
          CHECK(lhs->shape().dimensions(input_batch_dim) ==
                hlo_shape.dimensions(output_batch_dim));
        } else {
          CHECK(lhs->shape().dimensions(input_batch_dim) ==
                rhs->shape().dimensions(output_batch_dim));
        } 
        break;
      }

      case HloOpcode::kCustomCollective: {
        HloCustomCollectiveInstruction* collective = Cast<HloCustomCollectiveInstruction>(hlo);
        if (collective->collective_type() == kAllToAllType) {
          auto& hlo_shape = hlo->shape();
          auto& input_shape = hlo->mutable_operand(0)->shape();
          int hlo_rank = hlo_shape.rank();
          int input_rank = input_shape.rank();
          CHECK(hlo_rank == input_rank);
          auto config =
              hlo->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
          int64 split_dim = config.split_dim();
          int64 concat_dim = config.concat_dim();
          int64 num_replicas = config.num_replicas();
          for (int i = 0; i < hlo_rank; ++i) {
            if (i == concat_dim) {
              CHECK(hlo_shape.dimensions(i) == num_replicas * input_shape.dimensions(i));
            } else if (i == split_dim) {
              CHECK(input_shape.dimensions(i) == num_replicas * hlo_shape.dimensions(i));
            } else {
              CHECK(input_shape.dimensions(i) == hlo_shape.dimensions(i));
            }
          }
        } else if (collective->collective_type() == kAllGatherType) {
          auto& hlo_shape = hlo->shape();
          auto& input_shape = hlo->mutable_operand(0)->shape();
          int hlo_rank = hlo_shape.rank();
          int input_rank = input_shape.rank();
          CHECK(hlo_rank == input_rank);
          auto config =
              hlo->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
          int64 concat_dim = config.concat_dim();
          int64 num_replicas = config.num_replicas();
          for (int i = 0; i < hlo_rank; ++i) {
            if (i == concat_dim) {
              CHECK(hlo_shape.dimensions(i) == num_replicas * input_shape.dimensions(i));
            } else {
              CHECK(hlo_shape.dimensions(i) == input_shape.dimensions(i));
            }
          }
        } else if (collective->collective_type() == kAllReduceType) {
          auto input = hlo->mutable_operand(0);
          CHECK(ShapeUtil::ElementsIn(hlo->shape()) ==
                ShapeUtil::ElementsIn(input->shape()));
        } else if (collective->collective_type() == kDynamicSliceType) {
          auto& hlo_shape = hlo->shape();
          auto& input_shape = hlo->mutable_operand(0)->shape();
          int hlo_rank = hlo_shape.rank();
          int input_rank = input_shape.rank();
          CHECK(hlo_rank == input_rank);
          auto config =
              hlo->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
          int64 split_dim = config.split_dim();
          int64 num_replicas = config.num_replicas();
          for (int i = 0; i < hlo_rank; ++i) {
            if (i == split_dim) {
             CHECK_EQ(hlo_shape.dimensions(i), (input_shape.dimensions(i) / num_replicas));
            } else {
              CHECK(hlo_shape.dimensions(i) == input_shape.dimensions(i));
            }
          }         
        }
        break;
      }

      case HloOpcode::kDAPPLEAllToAll: {
        // AllToAll must have a single operand when the split dimension is
        // specified.
        auto& hlo_shape = hlo->shape();
        auto& input_shape = hlo->mutable_operand(0)->shape();
        int hlo_rank = hlo_shape.rank();
        int input_rank = input_shape.rank();
        CHECK(hlo_rank == input_rank);
        auto split_dimension =
            Cast<HloDAPPLEAllToAllInstruction>(hlo)->split_dimension();
        for (int i = 0; i < hlo_rank; ++i) {
          CHECK(input_shape.dimensions(i) == hlo_shape.dimensions(i));
        }
        break;
      }

      case HloOpcode::kDAPPLEAllGather: {
        auto& hlo_shape = hlo->shape();
        auto& input_shape = hlo->mutable_operand(0)->shape();
        int hlo_rank = hlo_shape.rank();
        int input_rank = input_shape.rank();
        CHECK(hlo_rank == input_rank);
        auto all_gather_dimension =
            Cast<HloDAPPLEAllGatherInstruction>(hlo)->all_gather_dimension();
        for (int i = 0; i < hlo_rank; ++i) {
          if (i == all_gather_dimension) {
            CHECK_EQ(hlo_shape.dimensions(i) % input_shape.dimensions(i), 0);
          } else {
            CHECK_EQ(hlo_shape.dimensions(i), input_shape.dimensions(i));
          }
        }
        break;
      }

      case HloOpcode::kRngGetAndUpdateState:
        break;
      case HloOpcode::kFusion:
      default: {
        VLOG(0) << "UNHANDLED->" << hlo->ToString();
        CHECK(0);
      }
    } // switch
  }
  return true;
}

HloInstruction* SpmdTransform::NewGather(HloInstruction* instr,
                           Shape& new_shape, HloComputation* entry,
                           int64 slice_dim) {
  auto updates = instr->mutable_operand(0);
  auto indices = instr->mutable_operand(1);

  const HloGatherInstruction* gather = DynCast<HloGatherInstruction>(instr);
  auto indices_are_sorted = gather->indices_are_sorted();
  CHECK(!indices_are_sorted);
  auto gather_dims = gather->gather_dimension_numbers();
  auto offset_dims = gather_dims.offset_dims();
  auto collapsed_slice_dims = gather_dims.collapsed_slice_dims();
  auto start_index_map = gather_dims.start_index_map();
  auto index_vector_dim = gather_dims.index_vector_dim();

  //std::vector<int64> offset_dims_copy(offset_dims);
  int64 num_offset_dims = offset_dims.size();
  std::vector<int64> offset_dims_copy;
  offset_dims_copy.reserve(num_offset_dims);
  for (int64 i = 0; i < num_offset_dims; ++i) {
    offset_dims_copy.emplace_back(offset_dims[i]);
  }

  //std::vector<int64> collapsed_slice_dims_copy;
  int64 num_collapsed = collapsed_slice_dims.size();
  std::vector<int64> collapsed_slice_dims_copy;
  collapsed_slice_dims_copy.reserve(num_collapsed);
  for (int64 i = 0; i < num_collapsed; ++i) {
    collapsed_slice_dims_copy.emplace_back(collapsed_slice_dims[i]);
  }

  // std::vector<int64> start_index_map_copy;
  int64 start_index_map_size = start_index_map.size();
  std::vector<int64> start_index_map_copy;
  start_index_map_copy.reserve(start_index_map_size);
  for (int64 i = 0; i < start_index_map_size; ++i) {
    start_index_map_copy.emplace_back(start_index_map[i]);
  }

  auto new_gather_dims = HloGatherInstruction::MakeGatherDimNumbers(
      offset_dims_copy, collapsed_slice_dims_copy, start_index_map_copy,
      index_vector_dim);

  auto read_only_slice_sizes = gather->gather_slice_sizes();
  std::vector<int64> slice_sizes(read_only_slice_sizes.begin(), 
                                 read_only_slice_sizes.end());
  int64 uncollapsed_index = slice_dim - offset_dims[0];
  auto& dist_spec = instr->dist_spec();
  auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
  if (uncollapsed_index < 0) {
    auto updates_dist_spec = updates->dist_spec();
    DimStrategy update_strategy(*updates_dist_spec.get_dim_spec(split_ordinal_));
    auto updates_dim = update_strategy.partition_dim();
    auto delta = slice_sizes.size() - num_collapsed;
    auto updates_rank = updates->shape().rank();
  } else {
    auto new_dim_size = new_shape.dimensions(slice_dim);
    auto& old_shape = instr->shape();
    int64 old_dim_size = old_shape.dimensions(slice_dim); 
    CHECK(old_dim_size % dim_spec->num_splits() == 0);
    CHECK(old_dim_size / dim_spec->num_splits() == new_dim_size) << instr->ToString();

    slice_sizes[uncollapsed_index + num_collapsed] = new_dim_size;
  }

  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateGather(new_shape, updates, indices, 
         new_gather_dims, slice_sizes, indices_are_sorted));

  new_inst->set_metadata(instr->metadata());
  
  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewSlice(HloInstruction* instr,
                           Shape& new_shape, HloComputation* entry,
                           int64 slice_dim) {
  auto input = instr->mutable_operand(0);
  //auto& input_shape = input->shape();

  const HloSliceInstruction* slice = DynCast<HloSliceInstruction>(instr);
  std::vector<int64> slice_starts = slice->slice_starts();
  std::vector<int64> slice_limits = slice->slice_limits();
  std::vector<int64> slice_strides = slice->slice_strides();
  auto& dist_spec = instr->dist_spec();
  auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
  CHECK(slice_strides[slice_dim] == 1);
  CHECK((slice_limits[slice_dim] - slice_starts[slice_dim]) % dim_spec->num_splits() == 0);
  slice_limits[slice_dim] = (slice_limits[slice_dim] - slice_starts[slice_dim]) / dim_spec->num_splits();

  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateSlice(new_shape, input, slice_starts, 
                                 slice_limits, slice_strides));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewReverse(HloInstruction* instr,
                           Shape& new_shape, HloComputation* entry,
                           int64 dim_to_slice) {
  auto input = instr->mutable_operand(0);
  //auto& input_shape = input->shape();

  const HloReverseInstruction* reverse = DynCast<HloReverseInstruction>(instr);
  auto& reverse_dims = reverse->dimensions();
  auto& dist_spec = instr->dist_spec();
  auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
  CHECK(std::find(reverse_dims.begin(), reverse_dims.end(), dim_to_slice) == reverse_dims.end());

  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateReverse(new_shape, input, reverse_dims));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  /*
  new_inst->mutable_dist_spec()->AddDimDistSpec(
      dim_spec->stride(), dim_spec->stride_on_dim(),
      dim_spec->partition_dim(), dim_spec->num_splits(),
      dim_spec->partial());
  */

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewDynamicSlice(HloInstruction* instr, 
                             Shape& new_shape, HloComputation* entry,
                             int dim_to_slice, int64 num_replicas) {
  auto input = instr->mutable_operand(0);
  const HloSliceInstruction* slice = DynCast<HloSliceInstruction>(instr);
  std::vector<int64> slice_starts = slice->slice_starts();
  std::vector<int64> slice_limits = slice->slice_limits();
  std::vector<int64> slice_strides = slice->slice_strides();

  for (int i = 0; i < slice_strides.size(); ++i) {
    CHECK(slice_strides[i] == 1);
  }

  std::vector<int64> dy_slice_size;
  for (int i = 0; i < slice_limits.size(); ++i) {
    dy_slice_size.push_back(slice_limits[i] - slice_starts[i]);
  }
  dy_slice_size[dim_to_slice] /= num_replicas;

  auto* replica_id = entry->AddInstruction(HloInstruction::CreateReplicaId());
  std::vector<HloInstruction*> start_indices;
  for (int i = 0; i < slice_starts.size(); ++i) {
    auto* start_offset = entry->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(slice_starts[i])));
    if (dim_to_slice == i) {
      // Compute offset according to replica_id
      auto* split_stride = entry->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(dy_slice_size[i])));
      auto* curr_start = entry->AddInstruction(
          HloInstruction::CreateBinary(start_offset->shape(),
          HloOpcode::kMultiply, replica_id, split_stride));
      auto* start_index = entry->AddInstruction(
          HloInstruction::CreateBinary(curr_start->shape(),
          HloOpcode::kAdd, curr_start, start_offset));
      start_indices.push_back(start_index);
    } else {
      start_indices.push_back(start_offset);
    }
  }

  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateDynamicSlice(new_shape, input, start_indices, dy_slice_size));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewIota(HloInstruction* instr, 
                          Shape& new_shape, HloComputation* entry,
                          int slice_dim) {
  const HloIotaInstruction* iota = DynCast<HloIotaInstruction>(instr);
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateIota(new_shape, iota->iota_dimension()));
  new_inst->set_metadata(instr->metadata());
  
  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewBroadcast(HloInstruction* instr, 
                               Shape& new_shape, HloComputation* entry,
                               int dim_to_slice) {
  auto dimensions = instr->dimensions();

  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto input = instr->mutable_operand(0);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateBroadcast(new_shape, input, dimensions));

  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewReshape(HloInstruction* instr, 
                             Shape& new_shape, HloComputation* entry,
                             int dim_to_slice) {
  const HloReshapeInstruction* reshape = DynCast<HloReshapeInstruction>(instr);
  auto inferred_dimension = reshape->inferred_dimension();

  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto input = instr->mutable_operand(0);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateReshapeNoShapeCheck(new_shape, input, inferred_dimension));

  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewConcat(HloInstruction* instr, 
                            Shape& new_shape, HloComputation* entry,
                            int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto operands = instr->operands();
  auto dimension = instr->dimensions(0);

  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateConcatenate(new_shape, operands, dimension));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewTenary(HloInstruction* instr, 
                            Shape& new_shape, HloComputation* entry,
                            int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto pred = instr->mutable_operand(0);
  auto lhs = instr->mutable_operand(1);
  auto rhs = instr->mutable_operand(2);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateTernary(new_shape, instr->opcode(), pred, lhs, rhs));
  new_inst->set_metadata(instr->metadata());
  
  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewScatter(HloInstruction* instr, 
                             Shape& new_shape, HloComputation* entry,
                             int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto operand = instr->mutable_operand(0);
  auto indices = instr->mutable_operand(1);
  auto updates = instr->mutable_operand(2);

  HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(instr);
  auto update_computation = scatter->called_computations().back();
  auto scatter_dim_numbers = scatter->scatter_dimension_numbers();
  auto indices_are_sorted = scatter->indices_are_sorted();
  auto unique_indices = scatter->unique_indices();
  
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateScatter(new_shape, operand, indices, updates, 
                                   update_computation, scatter_dim_numbers,
                                   indices_are_sorted, unique_indices));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewReduce(HloInstruction* instr, 
                            Shape& new_shape, HloComputation* entry, 
                            int64 dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  HloReduceInstruction* reduce = DynCast<HloReduceInstruction>(instr);
  absl::Span<HloInstruction* const> inputs = reduce->inputs();
  absl::Span<HloInstruction* const> init_values = reduce->init_values();

  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateReduce(new_shape, inputs, init_values,
                               /*window=*/instr->dimensions(),
                               /*reduce_computation=*/instr->to_apply()));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  VLOG(1) << "OLD->" << instr->ToString();
  VLOG(1) << "NEW->" << new_inst->ToString();
  return new_inst;
}

HloInstruction* SpmdTransform::NewTranspose(HloInstruction* instr, 
                               Shape& new_shape, HloComputation* entry,
                               int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto input = instr->mutable_operand(0);
  auto dimensions = instr->dimensions();
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateTranspose(new_shape, input, dimensions));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "OLD->" << instr->ToString();
  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewConvert(HloInstruction* instr, 
                             Shape& new_shape, HloComputation* entry,
                             int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto input = instr->mutable_operand(0);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateConvert(new_shape, input));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewBitcastConvert(HloInstruction* instr, 
                           Shape& new_shape, HloComputation* entry,
                           int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto input = instr->mutable_operand(0);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateBitcastConvert(new_shape, input));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewCompare(HloInstruction* instr, 
                             Shape& new_shape, HloComputation* entry,
                             int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto lhs = instr->mutable_operand(0);
  auto rhs = instr->mutable_operand(1);
  ComparisonDirection cmp_dir = 
      Cast<HloCompareInstruction>(instr)->direction();
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateCompare(new_shape, lhs, rhs, cmp_dir));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewDot(HloInstruction* instr, 
                         Shape& new_shape, HloComputation* entry,
                         int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto lhs = instr->mutable_operand(0);
  auto rhs = instr->mutable_operand(1);
  DotDimensionNumbers dnums = instr->dot_dimension_numbers();
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateDot(new_shape, lhs, rhs, dnums,
                               instr->precision_config()));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "OLD->" << instr->ToString();
  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewConvolution(HloInstruction* instr, 
                         Shape& new_shape, HloComputation* entry,
                         int dim_to_slice) {
  auto* conv = DynCast<HloConvolutionInstruction>(instr);
  auto& dim_spec = conv->dist_spec().get_dim_spec(split_ordinal_);
  auto lhs = conv->mutable_operand(0);
  auto rhs = conv->mutable_operand(1);
  auto new_inst = entry->AddInstruction(HloInstruction::CreateConvolve(
      new_shape, lhs, rhs, conv->feature_group_count(), conv->batch_group_count(),
      conv->window(), conv->convolution_dimension_numbers(), conv->precision_config()));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  /*
  new_inst->mutable_dist_spec()->AddDimDistSpec(
      dim_spec->stride(), dim_spec->stride_on_dim(),
      dim_spec->partition_dim(), dim_spec->num_splits(),
      dim_spec->partial());
  */

  VLOG(1) << "OLD->" << instr->ToString();
  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewUnary(HloInstruction* instr, 
                           Shape& new_shape, HloComputation* entry,
                           int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto input = instr->mutable_operand(0);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateUnary(new_shape, instr->opcode(), input));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewSort(HloInstruction* instr,
                           Shape& new_shape, HloComputation* entry,
                           int dim_to_slice) {
  auto* sort = DynCast<HloSortInstruction>(instr);
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto& dimensions = instr->dimensions();
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateSort(new_shape,
                                dimensions[0],
                                instr->operands(),
                                instr->to_apply(),
                                sort->is_stable()));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewRng(HloInstruction* instr, 
                         Shape& new_shape, HloComputation* entry,
                         int dim_to_slice) {
  CHECK(2 == instr->operand_count());
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateRng(new_shape,
         DynCast<HloRngInstruction>(instr)->random_distribution(),
             instr->operands()));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewCrossReplicaSum(HloInstruction* instr, 
                            Shape& new_shape, HloComputation* entry,
                            int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  std::vector<HloInstruction*> operands;
  auto lhs = instr->mutable_operand(0);
  operands.push_back(lhs);

  decltype(lhs) rhs = nullptr;
  if (2 == instr->operand_count()) {
    rhs = instr->mutable_operand(1);
    operands.push_back(rhs);
  }

  auto num_replicas =
      DynCast<HloDAPPLEAllReduceInstruction>(instr)->num_replicas();
  auto sharding =
      DynCast<HloDAPPLEAllReduceInstruction>(instr)->sharding();
  auto reduction_type =
      DynCast<HloDAPPLEAllReduceInstruction>(instr)->reduction_type();
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateDAPPLEAllReduce(new_shape, operands,
                                           num_replicas, sharding,
                                           reduction_type));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewCustomCollective(HloInstruction* instr, 
                            Shape& new_shape, HloComputation* entry,
                            int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  *(instr->mutable_shape()) = new_shape;

  auto& collective_type = DynCast<HloCustomCollectiveInstruction>(instr)->collective_type();
  if (collective_type != kAllReduceType) {
    auto config = instr->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
    int64 split_dim = config.split_dim();
    if (split_dim >= 0 && split_dim == dim_to_slice) {
      config.set_split_stride_on_dim(config.split_stride_on_dim() / dim_spec->num_splits());
    }
    int64 concat_dim = config.concat_dim();
    if (concat_dim >= 0 && concat_dim == dim_to_slice) {
      config.set_concat_stride_on_dim(config.concat_stride_on_dim() / dim_spec->num_splits());
    }
    instr->set_backend_config(config);
  }
  return instr;
}

HloInstruction* SpmdTransform::NewBinary(HloInstruction* instr, 
                            Shape& new_shape, HloComputation* entry,
                            int dim_to_slice) {
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto lhs = instr->mutable_operand(0);
  auto rhs = instr->mutable_operand(1);
  auto new_inst = entry->AddInstruction(
     HloInstruction::CreateBinary(new_shape, instr->opcode(), lhs, rhs));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

void SpmdTransform::NewTuple(HloInstruction* instr, 
                           HloComputation* entry) {
  CHECK(entry->root_instruction() == instr);
  auto new_root = entry->AddInstruction(
     HloInstruction::CreateTuple(instr->operands()));
  new_root->set_metadata(instr->metadata());
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_root).ok());
  CHECK(entry->root_instruction() == new_root);
}

HloInstruction* SpmdTransform::NewParameter(HloInstruction* instr, 
                               Shape& new_shape, HloComputation* entry,
                               int dim_to_slice) {
  VLOG(1) << "Old Shape:" << ShapeUtil::HumanString(instr->shape())
            << " New Shape:" << ShapeUtil::HumanString(new_shape);

  auto instr_name = instr->name();
  auto param_no = instr->parameter_number();
  auto& dim_spec = instr->dist_spec().get_dim_spec(split_ordinal_);
  auto new_param_ptr = HloInstruction::CreateParameter(param_no,
                           new_shape, instr_name+".new");
  new_param_ptr->set_metadata(instr->metadata());
  auto new_param = new_param_ptr.get();

  *(new_param->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_param->ToString();
  auto* new_param_cast = DynCast<HloParameterInstruction>(new_param);
  new_param_cast->set_split_ordinal(split_ordinal_);
  CHECK(entry->ReplaceParameterWithDifferentShape(param_no, std::move(new_param_ptr)).ok());
  return new_param;
}

HloInstruction* SpmdTransform::NewGTE(HloInstruction* instr,
                         Shape& new_shape, HloComputation* entry,
                         int dim_to_slice) {
  auto operand = instr->mutable_operand(0);
  auto index = instr->tuple_index();
  auto new_inst = entry->AddInstruction(
           HloInstruction::CreateGetTupleElement(new_shape,
                                                 operand, index));
  auto& dist_spec = instr->dist_spec();
  auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "OLD->" << instr->ToString();
  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewSelectAndScatter(HloInstruction* instr,
                         Shape& new_shape, HloComputation* entry,
                         int dim_to_slice) {
  auto lhs = instr->mutable_operand(0);
  auto rhs = instr->mutable_operand(1);
  auto init_val = instr->mutable_operand(2);
  auto new_inst = entry->AddInstruction(
         HloInstruction::CreateSelectAndScatter(
                       /*shape=*/new_shape,
                       /*operand=*/lhs,
                       /*select=*/instr->select(),
                       /*window=*/instr->window(),
                       /*source=*/rhs,
                       /*init_value=*/init_val,
                       /*scatter=*/instr->scatter()));
 
  auto& dist_spec = instr->dist_spec();
  auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewReduceWindow(HloInstruction* instr,
                         Shape& new_shape, HloComputation* entry,
                         int dim_to_slice) {
  auto operand = instr->mutable_operand(0);
  auto init_val = instr->mutable_operand(1);
  HloInstruction* new_inst = nullptr;
  auto new_window = instr->window();
  auto* slice_dim = new_window.mutable_dimensions(dim_to_slice);
  if (slice_dim->size() != 1) {
    CHECK(0) << "Here should never be reached as we already use "
             << "AllGather+ReduceWindow+DynamicSlice to process kReduceWindow "
             << "with reduced-dim sliced case for maintain consistency before "
             << "and after instruction split.";
  }
  new_inst = entry->AddInstruction(
       HloInstruction::CreateReduceWindow(
                       /*shape=*/new_shape,
                       /*operand=*/operand,
                       /*init_value=*/init_val,
                       /*window=*/new_window,
                       /*reduce_computation=*/instr->to_apply()));
 
  auto& dist_spec = instr->dist_spec();
  auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewCustomCall(HloInstruction* instr,
                         Shape& new_shape, HloComputation* entry,
                         int dim_to_slice) {
  auto& conv_dnums = instr->convolution_dimension_numbers();
  //auto input_batch_dim = conv_dnums.input_batch_dimension();
  //auto output_batch_dim = conv_dnums.output_batch_dimension();
  //CHECK(dim_to_slice == output_batch_dim);
  
  auto data = instr->mutable_operand(0);
  auto filter = instr->mutable_operand(1);
  auto target = instr->custom_call_target();

  auto config = instr->raw_backend_config_string();

  auto& tuple_shape = instr->shape();
  auto aux_shape = ShapeUtil::GetTupleElementShape(tuple_shape, 1);
  auto new_tuple_shape = ShapeUtil::MakeTupleShape({new_shape, aux_shape});

  auto num_operands = instr->operand_count();
  HloInstruction* new_inst = nullptr;
  if (num_operands == 2) {
    CHECK (target == "__cudnn$convForward" ||
           target == "__cudnn$convBackwardInput" ||
           target == "__cudnn$convBackwardFilter");
    new_inst = entry->AddInstruction(
        HloInstruction::CreateCustomCall(new_tuple_shape,
                                         {data, filter}, target, config));
  } else {
    CHECK(3 == num_operands);
    CHECK(target == "__cudnn$convBiasActivationForward");
    auto bias = instr->mutable_operand(2);  // bias
    new_inst = entry->AddInstruction(
        HloInstruction::CreateCustomCall(new_tuple_shape,
                                         {data, filter, bias},
                                         target, config));
  }
  new_inst->set_window(instr->window());
  new_inst->set_convolution_dimension_numbers(
      instr->convolution_dimension_numbers());
  new_inst->set_feature_group_count(instr->feature_group_count());
 
  auto new_conv_dnums = conv_dnums;
  new_inst->set_convolution_dimension_numbers(conv_dnums);

  auto& dist_spec = instr->dist_spec();
  auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}

HloInstruction* SpmdTransform::NewPad(HloInstruction* instr, 
                         Shape& new_shape, HloComputation* entry,
                         int dim_to_slice) {
  auto& dist_spec = instr->dist_spec();
  auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
  auto input = instr->mutable_operand(0);
  auto padding_value = instr->mutable_operand(1);

  const HloPadInstruction* pad = DynCast<HloPadInstruction>(instr);
  CHECK(pad);

  auto new_inst = entry->AddInstruction(
     HloInstruction::CreatePad(new_shape, input, padding_value, 
                               pad->padding_config()));
  new_inst->set_metadata(instr->metadata());

  *(new_inst->mutable_dist_spec()) = instr->dist_spec();

  VLOG(1) << "NEW->" << new_inst->ToString();
  CHECK(entry->ReplaceInstructionNoShapeCheck(instr, new_inst).ok());
  return new_inst;
}



/*static*/
CustomCollectiveSpec CustomCollectiveSpec::CreateAllReduceSpec(
    HloInstruction* instr, string reduction_type, int num_replicas) {
  CustomCollectiveSpec spec;
  spec.custom_type_ = kAllReduceType;
  spec.instr_ = instr;
  spec.allreduce_config_ = CreateAllReduceBackendConfig(
      reduction_type, num_replicas);

  int64 inst_bytes = ShapeUtil::ByteSizeOf(instr->shape(), 8);
  spec.cost_ = PerfUtils::AllReduceCost(inst_bytes, num_replicas);  // ring allreduce
  return spec;
}

/*static*/
CustomCollectiveSpec CustomCollectiveSpec::CreateReshardSpec(
    HloInstruction* instr, int operand_idx, HloInstruction* op,
    const DimStrategy& from_strategy, const DimStrategy& to_strategy, int num_replicas) {
  CustomCollectiveSpec spec;
  spec.custom_type_ = kReshardType;
  spec.instr_ = instr;
  spec.operand_idx_ = operand_idx;
  spec.reshard_config_ = CreateReshardBackendConfig(to_strategy, from_strategy, num_replicas);

  int64 op_bytes = ShapeUtil::ByteSizeOf(op->shape(), 8);

  if (spec.reshard_config_.reshard_type() == kAllGatherType) {
    spec.cost_ = PerfUtils::AllGatherCost(op_bytes, num_replicas);
  } else if (spec.reshard_config_.reshard_type() == kAllToAllType) {
    spec.cost_ = PerfUtils::AllToAllCost(op_bytes, num_replicas);
  } else if (spec.reshard_config_.reshard_type() == kDynamicSliceType) {
    spec.cost_ = 0;
  } else {
    CHECK(0);
  }
  return spec;
}

int64 SpmdTransform::LSTransform(HloComputation* computation) {
  int64 num_sliced = 0;
  auto post_order = computation->MakeInstructionPostOrder();

  for (auto instr : post_order) {
    if (instr->opcode() == HloOpcode::kTuple) {
      NewTuple(instr, computation);
      continue;
    }

    auto& dist_spec = instr->dist_spec();
    if (dist_spec.is_empty()) {
      continue;
    }
    auto& dim_spec = dist_spec.get_dim_spec(split_ordinal_);
    auto stride = dim_spec->stride();
    if (stride == 0) {
      CHECK(0 == dim_spec->stride_on_dim() || dim_spec->partial());
      continue; // No rewrite is necessary
    }

    CreateNewInstruction(instr, computation);
    ++num_sliced;
  }

  return num_sliced;
}

void SpmdTransform::CreateNewInstruction(HloInstruction* instr,
                                          HloComputation* computation) {
  const Shape* instr_shape = &instr->shape();
  const Shape* primitive_shape = instr_shape;
  if (instr_shape->IsTuple()) {
    primitive_shape = &ShapeUtil::GetTupleElementShape(*instr_shape, 0);
  }
  auto& dist_spec = instr->dist_spec();
  DimStrategy inst_strategy(*dist_spec.get_dim_spec(split_ordinal_));
  int dim_to_slice = inst_strategy.partition_dim();
  CHECK(dim_to_slice >= 0 && dim_to_slice < primitive_shape->rank());
  VLOG(2) << "[CreateNewInstruction] instr -> " << instr->ToString()
          << " dim_to_slice=" << dim_to_slice;
  Shape new_shape = DistUtil::MakeNewShape(
      *instr_shape, *dist_spec.get_dim_spec(split_ordinal_));
  HloInstruction* new_instr = nullptr;

  // Fetch strategies information for old instruction before being replaced
  auto decided_split_strategy = instr_strategy_map_[instr];
  auto inferred_op_strategy_vec = operand_inferred_strategy_map_[instr];
  if (instr->opcode() == HloOpcode::kConstant) {
    // Update split map record: Do not split kConstant.
    instr_strategy_map_[instr] = DimStrategy();
    return;
  }
  instr_strategy_map_.erase(instr);
  operand_inferred_strategy_map_.erase(instr);

  switch (instr->opcode()) {
    case HloOpcode::kParameter: {
      new_instr = NewParameter(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kExp:
    case HloOpcode::kLog:
    case HloOpcode::kSin:
    case HloOpcode::kCos:
    case HloOpcode::kTanh:
    case HloOpcode::kSqrt:
    case HloOpcode::kCopy:
    case HloOpcode::kRsqrt:
    case HloOpcode::kNegate:
    case HloOpcode::kIsFinite: {
      new_instr = NewUnary(instr, new_shape, computation, dim_to_slice);
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
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic: {
      new_instr = NewBinary(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kCompare: {
      new_instr = NewCompare(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kTranspose: {
      new_instr = NewTranspose(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kConvert: {
      new_instr = NewConvert(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kBitcastConvert: {
      new_instr = NewBitcastConvert(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kReshape: {
      new_instr = NewReshape(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kBroadcast: {
      new_instr = NewBroadcast(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kSelect: {
      new_instr = NewTenary(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kConcatenate: {
      new_instr = NewConcat(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kIota: {
      new_instr = NewIota(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kSlice: {
      new_instr = NewSlice(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kGather: {
      new_instr = NewGather(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kScatter: {
      new_instr = NewScatter(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kReduce: {
      new_instr = NewReduce(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kReverse: {
      new_instr = NewReverse(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kDot: {
      new_instr = NewDot(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kConvolution: {
      new_instr = NewConvolution(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kPad: {
      new_instr = NewPad(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kCustomCall: {
      new_instr = NewCustomCall(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kReduceWindow: {
      new_instr = NewReduceWindow(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kSelectAndScatter: {
      new_instr = NewSelectAndScatter(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kGetTupleElement: {
      new_instr = NewGTE(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kDAPPLEAllReduce: {
      new_instr = NewCrossReplicaSum(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kCustomCollective: {
      new_instr = NewCustomCollective(instr, new_shape, computation, dim_to_slice);
      break;
    }

    case HloOpcode::kRng: {
      new_instr = NewRng(instr, new_shape, computation, dim_to_slice);
      break;
    }

    // case HloOpcode::kConstant: {
    //   CHECK(0);
    //   new_instr = NewConstant(instr, new_shape, entry, dim_to_slice);
    //   break;
    // }

    case HloOpcode::kSort: {
      new_instr = NewSort(instr, new_shape, computation, dim_to_slice);
      break;
    }

    default: {
      VLOG(0) << "UNHANDLED->" << instr->ToString();
      CHECK(0);
    }
  }/* switch */
  // Update table to record split information for new instruction
  instr_strategy_map_[new_instr] = decided_split_strategy;
  operand_inferred_strategy_map_[new_instr] = inferred_op_strategy_vec;
}

// Defer the insertion of allreduce while maintaining:
//   a. semantic equivalence and
//   b. not increasing the amount of communication.
//
// And then we may have chances to merge independent kAllReduces of
// lhs and rhs from kAdd to one semantic equivalent kAllReduce, which reduce
// communication volume by 50% for this pattern.
void SpmdTransform::AllReducePatternOptimization(HloComputation* computation) {
  std::vector<HloInstruction*> ar_src_candidates;
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto* inst : computation->MakeInstructionPostOrder()) {
      if (inst->dist_spec().is_empty() ||
          !inst->dist_spec().get_dim_spec(split_ordinal_)->partial()) {
        continue;
      }
      VLOG(2) << "cur ar_src->" << inst->ToString();
      int num_replicas = instr_strategy_map_[inst].num_replicas();
      auto users = inst->users();
      CHECK(users.size() >= 1) << "users.size()=" << users.size()
          << "None use of allreduce source instruction->"
          << inst->ToString();

      if (users.size() != 1) continue;

      auto only_user = users[0];

      bool all_inferred_glue = true;
      for (HloInstruction* user : only_user->users()) {
        int64 op_idx = user->operand_index(only_user);
        all_inferred_glue &= operand_inferred_strategy_map_[user][op_idx].Glue();
      }

      if (!all_inferred_glue) continue;

      switch (only_user->opcode()) {
        case HloOpcode::kConvert:
          // Pattern1: kReduce->kAllreduce->kConvert to
          //           kReduce->kConvert->kAllReduce equivalently.
        case HloOpcode::kReshape: {
          // Pattern2: kReduce->kAllReduce->kConvert->kReshape to
          //           kReduce->kConvert->kReshape->kAllReduce equivalently.
          if (ShapeUtil::Equal(only_user->shape(), inst->shape())) {
            inst->mutable_dist_spec()->get_dim_spec(split_ordinal_)->set_partial(false);
            only_user->mutable_dist_spec()->get_dim_spec(split_ordinal_)->set_partial(true);
            instr_strategy_map_[inst] = DimStrategy();
            auto usr_str = DimStrategy::MakePartial(num_replicas);
            instr_strategy_map_[only_user] = *usr_str;
            changed = true;
            VLOG(2) << "Equivalently move AllReduce point from->" << inst->name()
                    << " to its only consumer->" << only_user->name();
          }
          break;
        }
        case HloOpcode::kScatter: {
          // Pattern3:
          //  kReduce1->kAllReduce1->kScatter1
          //                                  \
          //                                   -->kAdd
          //                                  /
          //  kReduce2->kAllReduce2->kScatter2
          // transform the above to the following pattern:
          //  kReduce1->kScatter1->kAllReduce1*
          //                                  \
          //                                   -->kAdd
          //                                  /
          //  kReduce2->kScatter2->kAllReduce2*
          //
          // Different from the Pattern1 and Pattern2 introduced aboved, this
          // transformation often leads to an increase in the amount of
          // communication for a single allReduce(e.g., kAllReduce1->kAllReduce*)
          // due to the semantics of kScatter. Thus we only perform the
          // conversion when the total communication volume decreases
          // with the subsequent kAdd allreduce communication merge optimization
          // introduced in Pattern4.
          if (ShapeUtil::ElementsIn(only_user->shape()) <=
              ShapeUtil::ElementsIn(inst->shape())*2) {
            inst->mutable_dist_spec()->get_dim_spec(split_ordinal_)->set_partial(false);
            only_user->mutable_dist_spec()->get_dim_spec(split_ordinal_)->set_partial(true);
            instr_strategy_map_[inst] = DimStrategy();
            auto usr_str = DimStrategy::MakePartial(num_replicas);
            instr_strategy_map_[only_user] = *usr_str;

            changed = true;
            VLOG(0) << "[kScatter]Move AllReduce point from->" << inst->name()
                    << " to its only consumer->" << only_user->name();
          }
        }

        case HloOpcode::kAdd: {
          // The above three patterns actually did not save communication
          // volumes, but actually create conditions for Pattern4 optimization.
          //
          // Pattern4:
          // lhs->kAllReduce1                    lhs
          //                \                      \
          //                 ->kAdd optimize to      ->kAdd->kAllReduce
          //                /                      /
          // rhs->kAllReduce2                    rhs
          //
          // This pattern optimization only requires half the number of
          // kAllReduces(from lhs-allreduce and rhs-allreduce to only one
          // kAdd-allreduce) and half the amount of communications while
          // maintaining semantic equivalence.
          auto lhs = only_user->mutable_operand(0);
          auto rhs = only_user->mutable_operand(1);
          if (lhs->dist_spec().get_dim_spec(split_ordinal_)->partial() &&
              rhs->dist_spec().get_dim_spec(split_ordinal_)->partial()) {
            lhs->mutable_dist_spec()->get_dim_spec(split_ordinal_)->set_partial(false);
            rhs->mutable_dist_spec()->get_dim_spec(split_ordinal_)->set_partial(false);
            only_user->mutable_dist_spec()->get_dim_spec(split_ordinal_)->set_partial(true);
            instr_strategy_map_[lhs] = DimStrategy();
            instr_strategy_map_[rhs] = DimStrategy();
            auto usr_str = DimStrategy::MakePartial(num_replicas);
            instr_strategy_map_[only_user] = *usr_str;

            changed = true;
            VLOG(2) << "Equivalently move AllReduce point from lhs->"
                    << lhs->name() << " and rhs->" << rhs->name()
                    << " to their add op->" << only_user->name();
          }
          break;
        }
        default:
          break;
      }
    }
  }
}

bool SpmdTransform::CreateAllCustomCollectives(HloComputation* computation) {
  std::string comm_info;
  std::string all_insts;

  int64 total_cost = 0;
  int num_ar = 0, num_ag = 0, num_aa = 0, num_ds = 0;
  // 1. We first build CustomCollectiveAllReduce for all partial instructions
  for (auto* instr : computation->MakeInstructionPostOrder()) {
    all_insts += instr->ToString() + "\n";
    if (instr_strategy_map_.find(instr) != instr_strategy_map_.end()) {
      all_insts += "  strtg: " + instr_strategy_map_.at(instr).ToString() + "\n";
    }
    if (instr_strategy_map_[instr].IsPartial()) {
      if (instr->metadata().op_name().find("clip_by_norm") != string::npos) {
        // Local gradient clip norm doesn't need to sync `l2sum` across shards.
        // Thus here we clear the `partial` flag for related reduction ops.
        CHECK_EQ(instr->metadata().op_type(), "Sum");
      } else {
        std::string reduction_type = BackTraceReductionType(instr);
        CustomCollectiveSpec spec =
            CustomCollectiveSpec::CreateAllReduceSpec(
                instr, reduction_type, instr_strategy_map_[instr].num_replicas());
        total_cost += spec.cost_;
        HloInstruction* all_reduce = CreateCustomCollective(computation, spec);
        comm_info += "\n[Comm] AllReduce %" + all_reduce->name()
                     + ": " + instr->ToString() + "\n";
        ++num_ar;
        instr_strategy_map_[all_reduce] = DimStrategy();
      } 
    }  
  }

  // 2. Then, we build all resharding CustomCollectives when needed
  bool changed;
  std::set<std::pair<int/*inst id*/, int/*op_idx*/>> edge_visited;
  do {
    changed = false;
    auto post_order = computation->MakeInstructionPostOrder();
    for (auto* instr : post_order) {
      if (instr == computation->root_instruction() &&
        instr->opcode() == HloOpcode::kTuple) {
        continue;
      }

      if (instr->dist_spec().is_empty()) {
        continue;
      }
  
      if (!operand_inferred_strategy_map_.count(instr)) continue;

      auto& inferred_op_strategy_vec = operand_inferred_strategy_map_[instr];
      for (int op_idx = 0; op_idx < instr->operand_count(); ++op_idx) {
        std::pair<int, int> edge = std::make_pair(instr->unique_id(), op_idx);
        if (edge_visited.find(edge) != edge_visited.end()) continue;
        edge_visited.insert(edge);
        auto* op = instr->mutable_operand(op_idx);
        CHECK(instr_strategy_map_.count(op));
        CHECK(op_idx < inferred_op_strategy_vec.size());
        auto from_op_strategy = instr_strategy_map_[op];
        if (op->opcode() == HloOpcode::kConstant) {
          // Make sure that kConstant instruction is always replicated.
          CHECK(from_op_strategy.Glue());
        }
        auto to_op_strategy = inferred_op_strategy_vec[op_idx];

        // Since all AllReduces have been inserted after `partial` instruction,
        // we ignore all `partial` instruction in this loop and regard them as
        // Glue.
        if (from_op_strategy.IsPartial()) from_op_strategy = DimStrategy();
        if (to_op_strategy.IsPartial()) to_op_strategy = DimStrategy();
        if (!from_op_strategy.Match(to_op_strategy)) {
          VLOG(2) << "[BuildAllCustomCollectiveSpecs] instr = " << instr->ToString();
          int num_replicas = from_op_strategy.Glue() || from_op_strategy.replicated() ?
                             to_op_strategy.num_replicas() : from_op_strategy.num_replicas();
          CustomCollectiveSpec spec = 
              CustomCollectiveSpec::CreateReshardSpec(
                  instr, op_idx, op, from_op_strategy, to_op_strategy, num_replicas);
          total_cost += spec.cost_;
          HloInstruction* reshard_coll = CreateCustomCollective(computation, spec);
          if (!reshard_coll) return false;
          comm_info += "\n[Comm] %" + reshard_coll->name()
                       + " " + spec.reshard_config_.reshard_type()
                       + " cost: " + std::to_string(spec.cost_)
                       + " between A and B:\nA: " + op->ToString()
                       + "\n   A strategy: " + from_op_strategy.ToString()
                       + "\nB: " + instr->ToString() + "\n   B strategy: "
                       + to_op_strategy.ToString() + "\n";

          if (spec.reshard_config_.reshard_type() == kAllGatherType) {
            ++num_ag;
          } else if (spec.reshard_config_.reshard_type() == kAllToAllType) {
            ++num_aa;
          } else if (spec.reshard_config_.reshard_type() == kDynamicSliceType) {
            ++num_ds;
          } else {
            CHECK(0);
          }
          DimStrategy dim_strategy(*reshard_coll->dist_spec().get_dim_spec(split_ordinal_));
          instr_strategy_map_[reshard_coll] = dim_strategy;
          changed = true;
        }
      }
    }
  } while (changed);

  VLOG(0) << "total cost: " << total_cost;
  VLOG(0) << "num_ar = " << num_ar
          << ", num_ag = " << num_ag
          << ", num_aa = " << num_aa
          << ", num_ds = " << num_ds;

  comm_info += "\n\ntotal cost: " + std::to_string(total_cost) + "\n";
  comm_info += "num_ar = " + std::to_string(num_ar) + ", num_ag = "
               + std::to_string(num_ag) + ", num_aa = " + std::to_string(num_aa)
               + ", num_ds = " + std::to_string(num_ds) + "\n";

  all_insts += "\n\ncommunication info:\n" + comm_info;
  tensorflow::Env* env = tensorflow::Env::Default();
  string fname = "comm_info." + std::to_string(split_ordinal_) + ".txt";
  Status status = tensorflow::WriteStringToFile(env, fname, all_insts);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write comm info to " << fname <<  " " << status;
  }
  return true;
}

HloInstruction* SpmdTransform::CreateCustomCollective(
    HloComputation* computation, CustomCollectiveSpec& spec) {
  HloInstruction* coll = nullptr;

  CHECK(computation);
  CHECK(spec.instr_);

  if (spec.custom_type_ == kAllReduceType) {
    auto ar = computation->AddInstruction(
        HloInstruction::CreateCustomCollective(spec.instr_->shape(),
            {spec.instr_}, kAllReduceType));
    CHECK(ar);
    auto* all_reduce = DynCast<HloCustomCollectiveInstruction>(ar);
    CHECK(all_reduce);
    all_reduce->set_backend_config(spec.allreduce_config_);
    all_reduce->set_metadata(spec.instr_->metadata());
    all_reduce->set_split_ordinal(split_ordinal_);
    *(all_reduce->mutable_dist_spec()) = DistSpec(spec.instr_->dist_spec().size());
    // Set AllReduce to Glue
    CHECK(all_reduce->mutable_dist_spec());
    std::unique_ptr<DimDistSpec>& ar_spec =
        all_reduce->mutable_dist_spec()->get_dim_spec(split_ordinal_);
    ar_spec->set_partial(false);
    ar_spec->set_layout_aware_partition(0, 0, -1, ar_spec->num_splits());

    CHECK(spec.instr_->ReplaceAllUsesWithDifferentShape(all_reduce).ok());
    coll = all_reduce;
    VLOG(2) << "[AllReduce] instr -> " << all_reduce->ToString();
  } else {
    HloInstruction* src = spec.instr_->mutable_operand(spec.operand_idx_);
    CHECK(src);
    VLOG(2) << "src: " << src->ToString();
    VLOG(2) << "src->shape().dimensions() size: " << src->shape().dimensions().size();
    for (auto d : src->shape().dimensions()) {
      VLOG(2) << "dim value: " << d;
    }
    std::vector<int64> new_dims(
        src->shape().dimensions().begin(), src->shape().dimensions().end());
    std::vector<int64> orig_dims(
        src->shape().dimensions().begin(), src->shape().dimensions().end());

    VLOG(2) << "new dims size: " << new_dims.size();

    int64 split_dim = spec.reshard_config_.split_dim();
    int64 concat_dim = spec.reshard_config_.concat_dim();
    CHECK(spec.reshard_config_.num_replicas()>0) << spec.reshard_config_.num_replicas();
    if (split_dim >= 0) {
      VLOG(2) << "new_dims size: " << new_dims.size();
      VLOG(2) << "split_dim: " << split_dim;
      if (new_dims[split_dim] < spec.reshard_config_.num_replicas()) return nullptr;
      new_dims[split_dim] /= spec.reshard_config_.num_replicas();
    }

    if (concat_dim >= 0) {
      VLOG(2) << "orig_dims size: " << orig_dims.size();
      VLOG(2) << "new_dims size: " << new_dims.size();
      VLOG(2) << "concat_dim: " << concat_dim;
      orig_dims[concat_dim] *= spec.reshard_config_.num_replicas();
      new_dims[concat_dim] *= spec.reshard_config_.num_replicas();
    }
    Shape orig_shape = DistUtil::MakeShape(src->shape(), orig_dims);
    Shape new_shape = DistUtil::MakeShape(src->shape(), new_dims);
    DimStrategy dim_strategy(orig_shape, split_dim, spec.reshard_config_.num_replicas());

    string reshard_str;
    CHECK(spec.reshard_config_.SerializeToString(&reshard_str));
    HloInstruction* rc;
    auto src_map = reshard_map_.find(src);
    if (src_map == reshard_map_.end() ||
        src_map->second.find(reshard_str) == src_map->second.end()) {
      rc = computation->AddInstruction(
          HloInstruction::CreateCustomCollective(
              new_shape, {src}, spec.reshard_config_.reshard_type()));
      auto* reshard_coll = DynCast<HloCustomCollectiveInstruction>(rc);
      CHECK(reshard_coll);
      // NOTE(zycao): Only keep dist spec info on current split ordinal here,
      // the coll would be identified as Glue on other ordinal. So it needs to
      // rebuild the spec info data structure with same size.
      *(reshard_coll->mutable_dist_spec()) = DistSpec(src->dist_spec().size());
      reshard_coll->set_split_ordinal(split_ordinal_);
      std::unique_ptr<DimDistSpec>& rc_spec =
          reshard_coll->mutable_dist_spec()->get_dim_spec(split_ordinal_);
      rc_spec->set_partial(dim_strategy.IsPartial());
      rc_spec->set_layout_aware_partition(
          dim_strategy.stride_on_elements(), dim_strategy.stride_on_dim(),
          dim_strategy.partition_dim(), dim_strategy.num_replicas());
      reshard_coll->set_backend_config(spec.reshard_config_);
      reshard_coll->set_metadata(src->metadata());
      reshard_map_[src][reshard_str] = rc;
    } else {
      rc = reshard_map_[src][reshard_str];
    }
    CHECK(spec.instr_->ReplaceOperandWithDifferentShape(spec.operand_idx_, rc).ok());
    VLOG(2) << "[CustomCollective] Creating CustomCollevtive type "
            << spec.reshard_config_.reshard_type()
            << " instruction : " << rc->ToString();
    coll = rc;
  }

  return coll;
}

void SpmdTransform::InitializeDimStrategyIntoMap(HloComputation* computation) {
  for (auto* instr : computation->instructions()) {
    if (instr == computation->root_instruction() &&
        instr->opcode() == HloOpcode::kTuple) continue;
    auto& dist_spec = instr->dist_spec();
    if (dist_spec.is_empty()) {
      continue;
    }
    DimStrategy dim_strategy(*(dist_spec.get_dim_spec(split_ordinal_)));
    instr_strategy_map_[instr] = dim_strategy;
  }
}

void SpmdTransform::InitializeInferedStrategyIntoMap(HloComputation* computation) {
  auto post_order = computation->MakeInstructionPostOrder();
  for (auto* instr : post_order) {
    // BackInfer to find all operands infered dimension
    if (instr == computation->root_instruction() &&
        instr->opcode() == HloOpcode::kTuple) continue;
    const DimStrategy& dim_inst_strategy = instr_strategy_map_[instr];
    VLOG(2) << "[Initialize] instr = " << instr->ToString()
            << ", inst_strategy : " << dim_inst_strategy.ToString();
    if (dim_inst_strategy.Glue()) {
      operand_inferred_strategy_map_[instr] =
          std::vector<DimStrategy>(instr->operand_count(), DimStrategy());
      continue;
    }
    std::vector<DimStrategy> op_expected_strategy_vec;
    for (int op_idx = 0; op_idx < instr->operand_count(); ++op_idx) {
      auto* op = instr->mutable_operand(op_idx);
      SharedDimStrategy op_expected_strategy =
          StrategyUtil::NonVerifyBackInfer(instr, dim_inst_strategy, op_idx);
      const DimStrategy* final_op_expected_str = op_expected_strategy.get();
      const DimStrategy& op_decided_strategy = instr_strategy_map_[op];
      if (op_expected_strategy->Glue() && !op_decided_strategy.Glue()) {
        // We should do another forward check from the operand given by instr_split_map
        // to instr to detect whether the existing strategies are compatible.
        auto verified_inst_strategies =
            StrategyUtil::ForwardInfer(instr, op_decided_strategy, op_idx);
        auto verified_dim_strategy = verified_inst_strategies[instr];
        VLOG(2) << "[Strategy backword deduction]: "
                << op->ToString() << " <-- from " << instr->ToString()
                << ";  >>> Deducted " << op_expected_strategy->ToString()
                << " <-- from " << verified_dim_strategy->ToString()
                << ";  <<< Recorded " << op_decided_strategy.ToString()
                << " <-- from " << dim_inst_strategy.ToString();
        if (verified_dim_strategy->Match(dim_inst_strategy)) {
          final_op_expected_str = &op_decided_strategy;
        }
      }
      op_expected_strategy_vec.push_back(*final_op_expected_str);
    }
    operand_inferred_strategy_map_[instr] = op_expected_strategy_vec;
  }
}

#if 0
HloInstMap<DimStrategy>
SpmdTransform::ResumeDimStrategyMap(HloComputation* entry) {
  HloInstMap<DimStrategy> strategy_map;
  for (const HloInstruction* inst : entry->instructions()) {
    if (!inst->dist_spec().is_empty()) {
      DimStrategy strategy(*(inst->dist_spec().get_dim_spec(split_ordinal_)));
      strategy_map[inst] = strategy;
    }
  }
  return strategy_map;
}

int64 SpmdTransform::ResolveGlobalNumReplicas(HloModule* module) {
  HloComputation* entry = module->entry_computation();
  int64 num_replicas = 1;
  for (const HloInstruction* inst : entry->instructions()) {
    if (inst->dist_spec().is_empty()) {
      continue;
    }
    int64 stride = inst->dist_spec().get_dim_spec(split_ordinal_)->stride();
    if (stride > 0) {
      num_replicas = inst->dist_spec().get_dim_spec(split_ordinal_)->num_splits();
      break;
    }
  }

  return num_replicas;
}
#endif

bool SpmdTransform::DoTransform(HloModule* module) {
  std::vector<HloComputation*> computations_to_transform = { module->entry_computation() };
  HloModule::DefContext* entry_def_ctx = module->def_ctx();
  for (HloModule::DefContext* child_def : entry_def_ctx->children()) {
    HloComputation* comp = module->Def2Compute(child_def);
    if (!comp) continue;
    computations_to_transform.push_back(comp);
  }

  for (HloComputation* comp : computations_to_transform) {
    instr_strategy_map_.clear();
    operand_inferred_strategy_map_.clear();
    InitializeDimStrategyIntoMap(comp);
    InitializeInferedStrategyIntoMap(comp);
    int64 num_slice = LSTransform(comp);
    //AllReducePatternOptimization(comp);
    if (!CreateAllCustomCollectives(comp)) return false;
  }

  return true;
}

void SpmdTransform::UpdateSplitInfo(HloModule* module) {
  CHECK(module->def_ctx());
  auto* def_ctx = module->def_ctx();
  def_ctx->instance_slice_ids_.clear();
  // The instances of all slices should be created
  for (int s = 0; s < module->total_split_num(); ++s) {
    def_ctx->instance_slice_ids_.insert(s);
  }

  const HloComputation* entry = module->entry_computation();
  const HloInstruction* root = entry->root_instruction();
  for (int idx = 0; idx < root->operand_count(); ++idx) {
    def_ctx->output_idx_global_dev_map_[idx] = def_ctx->instance_slice_ids_;
  }
}

StatusOr<bool> SpmdTransform::Run(HloModule* module) {
  if (!DoTransform(module)) return false;
  UpdateSplitInfo(module);
  // Refresh the entry computation layout
  *module->mutable_entry_computation_layout() =
       module->compute_computation_layout();
  return true;
}



}  // namespace xla
