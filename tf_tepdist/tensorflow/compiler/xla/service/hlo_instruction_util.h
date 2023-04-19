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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_UTIL_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
class CustomCallUtil {
 public:
  static bool ResolveDims(const HloInstruction* custom_call,
                          const string& signature,
                          std::vector<int64>& dims);

  static bool ResolveLhsContractDims(const HloInstruction* custom_call,
                                     std::vector<int64>& dims);

  static bool ResolveRhsContractDims(const HloInstruction* custom_call,
                                     std::vector<int64>& dims);

  static bool ResolveLhsBatchDims(const HloInstruction* custom_call,
                                  std::vector<int64>& dims);

  static bool ResolveRhsBatchDims(const HloInstruction* custom_call,
                                  std::vector<int64>& dims);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_UTIL_H_

