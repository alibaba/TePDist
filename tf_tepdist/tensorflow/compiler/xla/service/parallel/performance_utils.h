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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_PERFORMANCE_UTILS_H
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_PERFORMANCE_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

class PerfUtils {
 public:
  static int64 CalculateFlops(const HloComputation* comp);
  static int64 CalculateFlops(const HloInstruction* instr);
  static int64 AllReduceCost(const HloInstruction* all_reduce);
  static int64 AllReduceCost(int64 data_bytes, int split_num);
  static int64 AllToAllCost(const HloInstruction* all_to_all);
  static int64 AllToAllCost(int64 data_bytes, int split_num);
  static int64 AllGatherCost(const HloInstruction* all_gather);
  static int64 AllGatherCost(int64 data_bytes, int split_num);
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_PERFORMANCE_UTILS_H_
