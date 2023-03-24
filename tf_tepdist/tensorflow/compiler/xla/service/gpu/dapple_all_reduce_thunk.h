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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DAPPLE_ALL_REDUCE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DAPPLE_ALL_REDUCE_THUNK_H_

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

// Thunk that performs a NCCL-based All-Reduce among CUDA GPU-based replicas.
class DAPPLEAllReduceThunk : public Thunk {
 public:
  DAPPLEAllReduceThunk(std::vector<BufferAllocation::Slice> src_buffers,
                       std::vector<BufferAllocation::Slice> dst_buffers,
                       std::vector<int64> element_counts,
                       int64 num_replicas,
                       std::unique_ptr<KernelThunk> local_sum_thunk,
                       const HloInstruction* dapple_all_reduce);
  ~DAPPLEAllReduceThunk() override;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const int64 num_replicas_;
  std::vector<BufferAllocation::Slice> src_buffers_;
  std::vector<BufferAllocation::Slice> dst_buffers_;
  std::vector<int64> element_counts_;
  std::unique_ptr<KernelThunk> local_sum_thunk_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DAPPLE_ALL_REDUCE_THUNK_H_
