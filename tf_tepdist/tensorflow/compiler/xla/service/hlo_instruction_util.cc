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

#include "tensorflow/compiler/xla/service/hlo_instruction_util.h"

namespace xla {

/*static*/ bool CustomCallUtil::ResolveDims(const HloInstruction* custom_call,
                                            const string& signature,
                                            std::vector<int64>& dims) {
  auto& backend_config = custom_call->raw_backend_config_string();
  auto signature_pos = backend_config.find(signature);
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
  return true;
}

/*static*/ bool CustomCallUtil::ResolveLhsContractDims(
                                            const HloInstruction* custom_call,
                                            std::vector<int64>& dims) {
  return CustomCallUtil::ResolveDims(custom_call,
                                     "lhs_contracting_dimensions",
                                     dims);
}

/*static*/ bool CustomCallUtil::ResolveRhsContractDims(
                                            const HloInstruction* custom_call,
                                            std::vector<int64>& dims) {
  return CustomCallUtil::ResolveDims(custom_call,
                                     "rhs_contracting_dimensions",
                                     dims);
}

/*static*/ bool CustomCallUtil::ResolveLhsBatchDims(
                                            const HloInstruction* custom_call,
                                            std::vector<int64>& dims) {
  return CustomCallUtil::ResolveDims(custom_call, "lhs_batch_dimensions", dims);
}

/*static*/ bool CustomCallUtil::ResolveRhsBatchDims(
                                            const HloInstruction* custom_call,
                                            std::vector<int64>& dims) {
  return CustomCallUtil::ResolveDims(custom_call, "rhs_batch_dimensions", dims);
}

}  // namespace xla

