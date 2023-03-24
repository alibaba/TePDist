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

#include "tensorflow/compiler/xla/service/parallel/performance_utils.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_instruction_util.h"

namespace xla {

/*static*/
int64 PerfUtils::CalculateFlops(const HloComputation* comp) {
  int64 kFmaFlops = 2;
  int64 flop_count = 0;
  for (auto* instr : comp->instructions()) {
    flop_count += CalculateFlops(instr);
  }

  return flop_count;
}

/*static*/
int64 PerfUtils::CalculateFlops(const HloInstruction* instr) {
  int64 kFmaFlops = 2;
  int64 inst_flops = 0;
  switch (instr->opcode()) {
    case HloOpcode::kDot: {
      const Shape& lhs_shape = instr->operand(0)->shape();
      const Shape& dot_shape = instr->shape();
      const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
      // Count of elements along the reduction dimension
      int64 reduction_width = 1;
      for (auto dim : dnums.lhs_contracting_dimensions()) {
        reduction_width *= lhs_shape.dimensions(dim);
      }
      // Each output element requires reduction_width FMA operations.
      inst_flops = kFmaFlops * ShapeUtil::ElementsIn(dot_shape) * reduction_width;
      break;
    }

    case HloOpcode::kCustomCall: {
      if (instr->custom_call_target() != "__cublas$gemm") break;
      std::vector<int64> lhs_contracting_dims;
      bool succ = CustomCallUtil::ResolveLhsContractDims(instr,
                                                         lhs_contracting_dims);

      if (succ) {
        const Shape& lhs_shape = instr->operand(0)->shape();
        const Shape& gemm_shape = instr->shape();
        // Count of elements along the reduction dimension
        int64 reduction_width = 1;
        for (auto dim : lhs_contracting_dims) {
          reduction_width *= lhs_shape.dimensions(dim);
        }
        // Each output element requires reduction_width FMA operations.
        inst_flops = kFmaFlops * ShapeUtil::ElementsIn(gemm_shape) * reduction_width;
      }
      break;
    }

    case HloOpcode::kConvolution: {
      const Shape& conv_shape = instr->shape();
      const Shape& kernel_shape = instr->operand(1)->shape();
      const ConvolutionDimensionNumbers& dnums = instr->convolution_dimension_numbers();
      int64 spatial_nums = 1;
      for (auto dim : dnums.kernel_spatial_dimensions()) {
        spatial_nums *= kernel_shape.dimensions(dim);
      }
      // Each output element requires reduction_width FMA operations.
      inst_flops = kFmaFlops * ShapeUtil::ElementsIn(conv_shape) * 
                      spatial_nums * dnums.input_feature_dimension();
      break;
    }

    default: break;
  }

  return inst_flops;
}

/*static*/
int64 PerfUtils::AllReduceCost(const HloInstruction* all_reduce) {
  auto config =
      all_reduce->backend_config<gpu::AllReduceBackendConfig>().ValueOrDie();
  int64 data_bytes = ShapeUtil::ByteSizeOf(all_reduce->shape(), 8);
  return AllReduceCost(data_bytes, config.num_replicas());
}

/*static*/
int64 PerfUtils::AllReduceCost(int64 data_bytes, int split_num) {
  if (split_num > 2) {
    return data_bytes*2;
  } else {
    return data_bytes;
  }
}

/*static*/
int64 PerfUtils::AllGatherCost(const HloInstruction* all_gather) {
  auto config =
      all_gather->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
  int64 op_bytes = ShapeUtil::ByteSizeOf(all_gather->operand(0)->shape());
  return AllGatherCost(op_bytes, config.num_replicas());
}

/*static*/
int64 PerfUtils::AllGatherCost(int64 op_bytes, int split_num) {
  return op_bytes - op_bytes / split_num;
}

/*static*/
int64 PerfUtils::AllToAllCost(const HloInstruction* all_to_all) {
  auto config =
      all_to_all->backend_config<gpu::ReshardBackendConfig>().ValueOrDie();
  int64 op_bytes = ShapeUtil::ByteSizeOf(all_to_all->operand(0)->shape());
  return AllToAllCost(op_bytes, config.num_replicas());
}

/*static*/
int64 PerfUtils::AllToAllCost(int64 op_bytes, int split_num) {
  return (op_bytes+split_num-1)/split_num - op_bytes/(split_num*split_num);
}
} // namespace xla
