
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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DAPPLE_ALL_TO_ALL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DAPPLE_ALL_TO_ALL_THUNK_H_

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Thunk that performs a NCCL-based All-to-All among CUDA GPU-based replicas.
class DAPPLEAllToAllThunk : public Thunk {
 public:
  struct Buffer {
   int64 element_count;
   BufferAllocation::Slice source_buffer;
   BufferAllocation::Slice destination_buffer;
  };
  DAPPLEAllToAllThunk(
      std::vector<Buffer> buffers, const HloInstruction* all_to_all);

  Status ExecuteOnStream(const ExecuteParams& params) override;
  ~DAPPLEAllToAllThunk() override;

  // Returns whether the given instruction can be lowered to a nccl all-to-all
  // call
  static bool CanImplement(const HloInstruction* crs);
 private:
  const std::vector<Buffer> buffers_;
};

} // namespace gpu
} // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DAPPLE_ALL_TO_ALL_THUNK_H_