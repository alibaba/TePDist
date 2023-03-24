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

#include "tensorflow/compiler/xla/service/parallel/custom_collective_expander.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/service_env.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

const string kAllToAllType = "AllToAll";
const string kAllReduceType = "AllReduce";
const string kAllGatherType = "AllGather";
const string kDynamicSliceType = "DynamicSlice";
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

int64 ExpandDimensionsAndGetStrideDim(
    absl::Span<const int64> full_shape_dimension, int64 orig_dim, int64 stride_on_dim,
    int64 num_splits, std::vector<int64>* expanded_dimension) {
  CHECK(expanded_dimension);
  expanded_dimension->clear();
  int64 new_dim = 0;
  for (int64 i = 0; i < full_shape_dimension.size(); ++i) {
    if (i == orig_dim) {
      // Rewrite the split dimension with [dim / (num_splits * size) , num_splits, size]
      CHECK(full_shape_dimension[i] >= stride_on_dim);
      expanded_dimension->push_back(full_shape_dimension[i] / stride_on_dim);
      new_dim = i + 1;
      expanded_dimension->push_back(num_splits);
      expanded_dimension->push_back(stride_on_dim / num_splits);
    } else {
      expanded_dimension->push_back(full_shape_dimension[i]);
    }
  }

  return new_dim;
}

}

bool CustomCollectiveExpander::ExpandInComputation(HloComputation* computation) {
	bool changed = false;
  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kCustomCollective) {
      auto custom_collective_instr =
            DynCast<HloCustomCollectiveInstruction>(instr);
      VLOG(2) << "collective inst: " << custom_collective_instr->ToString();
      auto& collective_type = custom_collective_instr->collective_type();
      if (collective_type == kAllReduceType) {
        ExpandCrossPartitionAllReduce(computation, custom_collective_instr);
      } else if (collective_type == kAllToAllType) {
        ExpandCrossPartitionAllToAll(computation, custom_collective_instr);
      } else if (collective_type == kAllGatherType) {
        ExpandCrossPartitionAllGather(computation, custom_collective_instr);
      } else if (collective_type == kDynamicSliceType) {
        ExpandDynamicSlice(computation, custom_collective_instr);
      } else {
        CHECK(0 && "Unhandled collective type");
      }
      changed = true;
    }
  }

	return changed;
}

void CustomCollectiveExpander::ExpandCrossPartitionAllReduce(
    HloComputation* computation, HloCustomCollectiveInstruction* custom_allreduce) {
  // TODO(shiqing.fsq): how to get the sharding or replicas type?
  auto config =
      custom_allreduce->backend_config<gpu::AllReduceBackendConfig>().ValueOrDie();
  bool sharding = true;
  std::string reduction_type = config.reduction_type();
  bool need_convert = (reduction_type == kAndStr);
  auto* all_reduce_op = custom_allreduce->mutable_operand(0);
  if (need_convert) {
    reduction_type = kProdStr;
    auto new_shape = ShapeUtil::MakeShape(F32, all_reduce_op->shape().dimensions());
    all_reduce_op = computation->AddInstruction(
      HloInstruction::CreateConvert(new_shape, all_reduce_op));

  }
  // TePDist FP16 communication support.
  HloInstruction* last;
  HloDAPPLEAllReduceInstruction* all_reduce;
  bool fp16_comm = ServiceEnv::fp16_comm();
  if (fp16_comm && all_reduce_op->shape().element_type() == F32) {
    auto half_shape =
        ShapeUtil::MakeShape(F16, all_reduce_op->shape().dimensions());
    auto* half_all_reduce_src = computation->AddInstruction(
        HloInstruction::CreateConvert(half_shape, all_reduce_op));

    auto ar = computation->AddInstruction(
        HloInstruction::CreateDAPPLEAllReduce(
            half_all_reduce_src->shape(), {half_all_reduce_src},
            config.num_replicas(), sharding, reduction_type));

    all_reduce = DynCast<HloDAPPLEAllReduceInstruction>(ar);


    all_reduce->set_split_ordinal(custom_allreduce->split_ordinal());

    auto* full_ar_result = computation->AddInstruction(
        HloInstruction::CreateConvert(all_reduce_op->shape(), all_reduce));

    last = full_ar_result;
  } else {
    auto ar = computation->AddInstruction(
        HloInstruction::CreateDAPPLEAllReduce(
            all_reduce_op->shape(), {all_reduce_op},
            config.num_replicas(), sharding, reduction_type));

    all_reduce = DynCast<HloDAPPLEAllReduceInstruction>(ar);

    all_reduce->set_split_ordinal(custom_allreduce->split_ordinal());

    last = all_reduce;
  }
  if (need_convert) {
    auto new_shape = ShapeUtil::MakeShape(PRED, last->shape().dimensions());
    last = computation->AddInstruction(HloInstruction::CreateConvert(new_shape, last));
  }

  custom_allreduce->SetupDerivedInstruction(all_reduce);
  CHECK(custom_allreduce->ReplaceAllUsesWithDifferentShape(last).ok());
  CHECK(computation->RemoveInstructionAndUnusedOperands(custom_allreduce).ok());
  VLOG(2) << "[AllReduce] instr -> " << all_reduce->ToString();
}

void CustomCollectiveExpander::ExpandCrossPartitionAllToAll(
    HloComputation* computation, HloCustomCollectiveInstruction* custom_all2all) {
  auto config =
      custom_all2all->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
  auto split_dimension = config.split_dim();
  auto concat_dimension = config.concat_dim();
  auto split_stride_on_dim = config.split_stride_on_dim();
  auto concat_stride_on_dim = config.concat_stride_on_dim();
  auto split_count = config.num_replicas();

  std::vector<int64> full_shape_dimension;
  auto* src = custom_all2all->mutable_operand(0);
  for (int i = 0; i < src->shape().dimensions_size(); ++i) {
    if (i == concat_dimension) {
      full_shape_dimension.push_back(src->shape().dimensions(i) * split_count);
    } else {
      full_shape_dimension.push_back(src->shape().dimensions(i));
    }
  }

  int64 expand_split_dim = 0;
  int64 expand_concat_dim = 0;
  bool is_split_dim_smaller =
      split_dimension == concat_dimension ? split_stride_on_dim > concat_stride_on_dim :
                                            split_dimension < concat_dimension;
  int64 smaller_dim = concat_dimension;
  int64 smaller_stride = concat_stride_on_dim;
  int64 larger_dim = split_dimension;
  int64 larger_stride = split_stride_on_dim;
  if (is_split_dim_smaller) {
    smaller_dim = split_dimension;
    smaller_stride = split_stride_on_dim;
    larger_dim = concat_dimension;
    larger_stride = concat_stride_on_dim;
  }
  // The dimension information should be exposed clearly according to the stride and
  // num_replicas.
  // For example, when tensor [8, 680, 4096] has split_dimension on 1, stride 680 and
  // num_replicas = 2, we should reshape this tensor to [8, 1, 2, 340, 4096] to expose
  // the authenticate dimension, which is 2 in this example.
  //
  // Actually, resharding is to split on one dimension and to concatenate on another
  // dimension. We need to expand the original split_dimension and concat_dimension
  // separately and get the new two dimensions, respectively. Two cases should be considered.
  // 
  // 1. When split_dimension and concat_dimension are different.
  //                       s     l
  //                       |     |
  //   E.g. If tensor [8, 680, 4096] has smaller_dim = 1 and larger_dim = 2, the larger_dim
  //   will move `two steps` after smaller dimension expanded to 3 sub-dimensions. Then, we
  //   continue to expand the larger_dim 
  //         s     l             s         l
  //         |     |             |         |
  //   ([8, 680, 4096] -> [8, 1, 2, 340, 4096])
  // 2. When split_dimension and concat_dimension are same but has different stride.
  //                      s l
  //                      | |
  //   E.g. If tensor [8, 680, 4096] has smaller_dim = 1 and larger_dim = 1, the larger_dim
  //   will move `one step` after smaller dimension expanded to 3 sub-dimensions. Then, we
  //   continue to expand the larger_dim 
  //        s l                  s   l
  //        | |                  |   |
  //   ([8, 680, 4096] -> [8, 1, 2, 340, 4096])

  std::vector<int64> temp_expanded_dims;
  int new_smaller_dim = ExpandDimensionsAndGetStrideDim(
      full_shape_dimension, smaller_dim, smaller_stride, split_count, &temp_expanded_dims);

  // Move larger_dim because of the expansion offset
  std::vector<int64> expanded_dims;
  int new_larger_dim = ExpandDimensionsAndGetStrideDim(
      temp_expanded_dims, larger_dim + 2, larger_stride, split_count, &expanded_dims);

  
  if (is_split_dim_smaller) {
    expand_split_dim = new_smaller_dim;
    expand_concat_dim = new_larger_dim;
  } else {
    expand_split_dim = new_larger_dim;
    expand_concat_dim = new_smaller_dim;
  }
  
  expanded_dims[expand_concat_dim] /= split_count;

  auto reshape = computation->AddInstruction(HloInstruction::CreateReshapeNoShapeCheck(
      ShapeUtil::MakeShape(src->shape().element_type(), expanded_dims), src));

  // Due to the implementation constraints of DAPPLEAllToAllThunk, we should make
  // the split_dimension at the major dimension (varaing slowest dimension). When
  // we declare DAPPLEAllToAll instruction here, the default minor-to-major shape
  // is used so that the split_dimension is 0.
  std::vector<int64> permutation(expanded_dims.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation[0] = expand_split_dim;
  permutation[expand_split_dim] = 0;

  auto pre_transpose = computation->AddInstruction(
      HloInstruction::CreateTranspose(
          ShapeInference::InferTransposeShape(reshape->shape(), permutation).ValueOrDie(),
          reshape, permutation));  

  auto a2a = computation->AddInstruction(
       HloInstruction::CreateDAPPLEAllToAll(
           pre_transpose->shape(), {pre_transpose}, 0/*expand_split_dim*/));
  auto all_to_all = DynCast<HloDAPPLEAllToAllInstruction>(a2a);
  all_to_all->set_metadata(src->metadata());
  /*
  all_to_all->set_dev_stride(custom_all2all->dev_stride());
  all_to_all->set_split_num(custom_all2all->split_num());
  all_to_all->set_split_nums(custom_all2all->split_nums());
  all_to_all->set_share_dev_flags(custom_all2all->share_dev_flags());
  all_to_all->set_total_split_num(custom_all2all->total_split_num());
  */
  all_to_all->set_split_ordinal(custom_all2all->split_ordinal());

  // Post permutation is needed so that the concat dimension are combined
  std::vector<int64> post_permutation(expanded_dims.size());
  std::iota(post_permutation.begin(), post_permutation.end(), 0);
  // (Note) Due to the expand_split_dim are used in the aboved transpose, the
  // expand_split_dim here stands for the `0` dimension in original instruction.
  // We move the `0` dimension to the original position, and move the
  // expand_concat_dim to the expand_split_dim's position.
  post_permutation[0] = expand_split_dim;
  post_permutation[expand_split_dim] = expand_concat_dim;
  post_permutation[expand_concat_dim] = 0;

  auto transpose = computation->AddInstruction(HloInstruction::CreateTranspose(
      ShapeInference::InferTransposeShape(all_to_all->shape(), post_permutation)
          .ValueOrDie(),
      all_to_all, post_permutation));

  auto new_shape = ShapeInference::InferAllToAllShape(
                       src->shape(), split_dimension, concat_dimension,
                       split_count).ValueOrDie();
  auto result = computation->AddInstruction(
      HloInstruction::CreateReshapeNoShapeCheck(new_shape, transpose));

  CHECK(custom_all2all->ReplaceAllUsesWithDifferentShape(result).ok());
  CHECK(computation->RemoveInstructionAndUnusedOperands(custom_all2all).ok());
  VLOG(2) << "[AllToAll] instr -> " << all_to_all->ToString();
}

void CustomCollectiveExpander::ExpandCrossPartitionAllGather(
    HloComputation* computation, HloCustomCollectiveInstruction* custom_allgather) {
  auto config =
      custom_allgather->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
  int64 concat_dim = config.concat_dim();
  int64 concat_stride_on_dim = config.concat_stride_on_dim();
  int64 num_replicas = config.num_replicas();
  
  const Shape& full_shape = custom_allgather->shape();
  CHECK(!full_shape.IsTuple()) << "Tuple type cannot be hanlded yet.";

  std::vector<int64> expanded_dims;
  int new_concat_dim = ExpandDimensionsAndGetStrideDim(
      full_shape.dimensions(), concat_dim, concat_stride_on_dim,
      num_replicas, &expanded_dims);

  HloInstruction* src = custom_allgather->mutable_operand(0);

  expanded_dims[new_concat_dim] /= num_replicas;
  Shape shard_shape = ShapeUtil::MakeShape(
    src->shape().element_type(), expanded_dims);

  HloInstruction* reshape = computation->AddInstruction(
      HloInstruction::CreateReshapeNoShapeCheck(shard_shape, src));
  std::vector<int64> permutation(expanded_dims.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation[0] = new_concat_dim;
  permutation[new_concat_dim] = 0;

  // Due to the implementation constraints of DAPPLEAllGatherThunk, we should make
  // the split_dimension at the major dimension (varaing slowest dimension). When
  // we declare DAPPLEAllGather instruction here, the default minor-to-major shape
  // is used so that the split_dimension is 0.
  auto pre_transpose = computation->AddInstruction(
      HloInstruction::CreateTranspose(
          ShapeInference::InferTransposeShape(reshape->shape(), permutation).ValueOrDie(),
          reshape, permutation));  

  std::vector<int64> pre_trans_dimensions(
      pre_transpose->shape().dimensions().begin(),
      pre_transpose->shape().dimensions().end());
  
  pre_trans_dimensions[0] *= num_replicas;
  Shape new_full_shape = ShapeUtil::MakeShape(
      src->shape().element_type(), pre_trans_dimensions);
  auto ag = computation->AddInstruction(
      HloInstruction::CreateDAPPLEAllGather(
          new_full_shape, {pre_transpose}, 0/*new_concat_dim*/));

  auto all_gather = DynCast<HloDAPPLEAllGatherInstruction>(ag);
  all_gather->set_split_ordinal(custom_allgather->split_ordinal());
  auto post_transpose = computation->AddInstruction(
      HloInstruction::CreateTranspose(
          ShapeInference::InferTransposeShape(all_gather->shape(), permutation).ValueOrDie(),
          all_gather, permutation));   

  HloInstruction* result = computation->AddInstruction(
      HloInstruction::CreateReshapeNoShapeCheck(custom_allgather->shape(), post_transpose));

  CHECK(computation->ReplaceInstructionNoShapeCheck(custom_allgather, result).ok());
  VLOG(2) << "[AllGather] instr -> " << all_gather->ToString();
}

void CustomCollectiveExpander::ExpandDynamicSlice(
	  HloComputation* computation, HloCustomCollectiveInstruction* custom_slice) {
  auto config =
      custom_slice->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
  int64 split_dim = config.split_dim();
  int64 split_stride_on_dim = config.split_stride_on_dim();
  int64 num_replicas = config.num_replicas();

  HloInstruction* src = custom_slice->mutable_operand(0);
  const Shape& full_shape = src->shape();
  CHECK(!full_shape.IsTuple()) << "Tuple type cannot be hanlded yet.";

  std::vector<int64> expanded_dims;
  int new_split_dim = ExpandDimensionsAndGetStrideDim(
      full_shape.dimensions(), split_dim, split_stride_on_dim,
      num_replicas, &expanded_dims);

  HloInstruction* reshape = computation->AddInstruction(
      HloInstruction::CreateReshapeNoShapeCheck(
          ShapeUtil::MakeShape(src->shape().element_type(), expanded_dims), src));

  std::vector<int64> new_dims(expanded_dims.begin(), expanded_dims.end());
  new_dims[new_split_dim] /= num_replicas;
  Shape shard_shape = MakeShape(reshape->shape(), new_dims);

  // Prepare start offset indices for each shard.
  // N.B., only support split at one dimension.
  std::vector<int32> offset_array(num_replicas);
  for (int64 i = 0; i < num_replicas; ++i) {
    offset_array[i] = i * new_dims[new_split_dim];
  }

  // Make partition offsets.
  std::vector<HloInstruction*> offsets;
  HloInstruction* partition_id =
      computation->AddInstruction(HloInstruction::CreatePartitionId());
  partition_id->set_split_ordinal(custom_slice->split_ordinal());
  for (int64 i = 0; i < shard_shape.rank(); ++i) {
    if (i == new_split_dim) {
      HloInstruction* offset = computation->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32>(offset_array)));
      HloInstruction* index = computation->AddInstruction(
          HloInstruction::CreateDynamicSlice(
              ShapeUtil::MakeShape(S32, {1}), offset, {partition_id}, {1}));
      offsets.push_back(computation->AddInstruction(
          HloInstruction::CreateReshapeNoShapeCheck(ShapeUtil::MakeShape(S32, {}), index)));
    } else {
      offsets.push_back(computation->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32))));
    }
  }
  HloInstruction* slice = computation->AddInstruction(
      HloInstruction::CreateDynamicSlice(
          shard_shape, reshape, offsets, shard_shape.dimensions()));
  HloInstruction* result = computation->AddInstruction(
      HloInstruction::CreateReshapeNoShapeCheck(custom_slice->shape(), slice));

  CHECK(custom_slice->ReplaceAllUsesWithDifferentShape(result).ok());
  CHECK(computation->RemoveInstructionAndUnusedOperands(custom_slice).ok());

  VLOG(2) << "[DynamicSlice] instr -> " << slice->ToString();
}

StatusOr<bool> CustomCollectiveExpander::Run(HloModule* module) {
	bool changed = false;
  for (HloComputation* computation : module->computations()) {
    changed |= ExpandInComputation(computation);
	}

	return changed;
}

} // namespace xla
