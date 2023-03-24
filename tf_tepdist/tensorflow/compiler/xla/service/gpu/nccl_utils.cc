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

#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace gpu {

ncclRedOp_t ToNcclReduction(NcclReductionKind kind) {
  switch (kind) {
    case NcclReductionKind::SUM:
      return ncclSum;
    case NcclReductionKind::PRODUCT:
      return ncclProd;
    case NcclReductionKind::MIN:
      return ncclMin;
    case NcclReductionKind::MAX:
      return ncclMax;
  }
}

StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType element_type) {
  switch (element_type) {
    case S8:
      return ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
      return ncclFloat32;
    case F64:
      return ncclFloat64;
    default:
      return tensorflow::errors::InvalidArgument(absl::StrFormat(
          "Unsupported data type: %s", PrimitiveType_Name(element_type)));
  }
}

std::string GlobalDevicesToString(const absl::Span<const int64> ids) {
  std::vector<int64> values;
  values.reserve(ids.size());
  for (int64 id : ids) {
    values.push_back(id);
  }
  return absl::StrJoin(values, ",");
}

Status ToStatus(ncclResult_t s, const char* file, int64 line,
                const char* expr) {
  if (s == ncclSuccess) {
    return Status::OK();
  }
  return tensorflow::errors::Internal(
      absl::StrFormat("%s:%d: NCCL operation %s failed: %s", file, line, expr,
                      ncclGetErrorString(s)));
}

Status ToStatus(cudaError_t s, const char* file, int64 line, const char* expr) {
  if (s == cudaSuccess) {
    return Status::OK();
  }
  return tensorflow::errors::Internal(
      absl::StrFormat("%s:%d: CUDA operation %s failed: %s", file, line, expr,
                      cudaGetErrorString(s)));
}

bool IsTypeSupportedByNccl(PrimitiveType element_type) {
  switch (element_type) {
    case S8:
    case PRED:
    case U8:
    case S32:
    case U32:
    case S64:
    case U64:
    case F16:
    case F32:
    case F64:
      return true;
    default:
      return false;
  }
}

}  // namespace gpu
}  // namespace xla
