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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "third_party/nccl/nccl.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

enum class NcclReductionKind { SUM, PRODUCT, MIN, MAX };
ncclRedOp_t ToNcclReduction(NcclReductionKind kind);
StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType element_type);

std::string GlobalDevicesToString(const absl::Span<const int64> ids);

// bool IsGlobalNcclConfig();
// bool IsNcclLaunchModeParallel();

// Functions to translate an ncclResult_t/cudaError_t to a Status object.  Used
// by the macros below.
Status ToStatus(ncclResult_t s, const char* file, int64 line, const char* expr);
Status ToStatus(cudaError_t s, const char* file, int64 line, const char* expr);

// Macros to return or warn on CUDA/NCCL errors.  (The same macro works for both
// NCCL and CUDA errors.)
//
// It's tempting to say these macros belong in an XLA header somewhere, but in
// practice we don't do much direct-to-CUDA-API stuff outside of this file.
#define XLA_CUDA_STATUS(expr) \
  xla::gpu::ToStatus(expr, __FILE__, __LINE__, #expr)

#define XLA_CUDA_RETURN_IF_ERROR(expr) \
  do {                                 \
    Status s = XLA_CUDA_STATUS(expr);  \
    if (!s.ok()) {                     \
      return s;                        \
    }                                  \
  } while (0)

#define XLA_CUDA_WARN_IF_ERROR(expr)  \
  do {                                \
    Status s = XLA_CUDA_STATUS(expr); \
    if (!s.ok()) {                    \
      LOG(ERROR) << s.ToString();     \
    }                                 \
  } while (0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Returns if the given data type is supported by NCCL.
// Note: Keep this in sync with ToNcclDataType().
bool IsTypeSupportedByNccl(PrimitiveType element_type);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_