/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/pjrt/nccl_context.h"
#include "tensorflow/compiler/xla/pjrt/dev_id_util.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

class GpuExecutable;

// Thunk acts as the bridge between IrEmitter and GpuExecutable. It stores the
// metadata IrEmitter generates for GpuExecutable to invoke an HloInstruction.
//
// Thunk provides the Initialize and ExecuteOnStream interface for GpuExecutable
// to initialize and execute the invocation respectively. Its subclasses are
// supposed to override these interfaces to launch a generated kernel or call an
// external library function (such as operations in cuBLAS).
//
// This is thread-compatible.
class Thunk {
 public:
  enum Kind {
    kCholesky,
    kCollectivePermute,
    kConditional,
    kConvolution,
    kCopy,
    kCudnnBatchNormBackward,
    kCudnnBatchNormForwardInference,
    kCudnnBatchNormForwardTraining,
    kCustomCall,
    kFft,
    kGemm,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kDAPPLEAllReduce,
    kDAPPLEAllToAll,
    kDAPPLEAllGather,
    kNcclAllReduce,
    kOutfeed,
    kReplicaId,
    kPartitionId,
    kSequential,
    kTriangularSolve,
    kTuple,
    kWhile,
  };

  // The hlo_instruction argument is meant to be the instruction this thunk was
  // generated from, but Thunk never uses this argument other than to save it
  // to Thunk::hlo_instruction, so it can be null.
  explicit Thunk(Kind kind, const HloInstruction* hlo_instruction)
      : kind_(kind), hlo_instruction_(hlo_instruction) {}
  virtual ~Thunk() {}
  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  Kind kind() const { return kind_; }
  const HloInstruction* hlo_instruction() const { return hlo_instruction_; }
  string profile_annotation() const { return profile_annotation_; }

  // Constructs and caches the profile annotation string for this thunk and
  // any child thunks.
  virtual void ComputeAnnotations() {
    const HloInstruction* hlo = hlo_instruction();
    if (hlo) {
      profile_annotation_ =
          absl::StrFormat("Thunk:#hlo_op=%s,hlo_module=%s#", hlo->name(),
                          hlo->GetModule()->name());
    }
  }

  // Prepares the thunk for execution on the given StreamExecutor.
  //
  // This may be called multiple times.  Its main purpose is to give us a chance
  // to do initialization outside of ExecuteOnStream() so that the
  // time spent initializing doesn't count towards our execution profile.
  virtual Status Initialize(const GpuExecutable& /*executable*/,
                            se::StreamExecutor* /*executor*/) {
    return Status::OK();
  }

#if 0
  std::vector<int> GetDevices(GlobalDeviceId gdev_id, const int split_ordinal,
                              const int dev_stride, const int split_num,
                              const int total_split_num) const;
#endif

  // Parameters passed to ExecuteOnStream.  Encapsulated in a struct so that
  // when we add something we don't have to change every subclass of Thunk.
  struct ExecuteParams {
    const BufferAllocations* buffer_allocations;  // never null
    se::Stream* stream;
    RunId run_id;
    HloExecutionProfiler* profiler;                               // never null
    const DeviceAssignment* device_assn;                          // never null
    std::vector<std::function<void()>>* deferred_host_callbacks;  // never null
    const std::vector<GlobalDeviceId>* gpu_global_device_ids;     // may be null
    const NcclUniqueIdCallback* nccl_unique_id_callback;          // may be null
    NcclContext* nccl_ctx;
    const std::vector<int64>* task_id_;
    std::shared_ptr<CommDevManager> comm_dev_mgr_;
  };

  // Execute the kernel for the thunk on the given stream. This method must be
  // called after Initialize and can be called multiple times over Thunk's
  // lifetime.
  //
  // Precondition: Initialize(stream->parent()) has been called.
  virtual Status ExecuteOnStream(const ExecuteParams& params) = 0;

 protected:
  const HloModuleConfig& GetModuleConfig() const {
    return hlo_instruction()->GetModule()->config();
  }

  // Safely copies the given buffer to the GPU, deleting it on the host only
  // after the copy has completed.
  template <typename T>
  void SafeH2DMemcpy(
      se::DeviceMemory<T> dest, std::unique_ptr<T[]> buf, int64 count,
      se::Stream* stream,
      std::vector<std::function<void()>>* deferred_host_callbacks) {
    stream->ThenMemcpy(&dest, buf.get(), count * sizeof(T));
    auto* buf_raw = buf.release();
    deferred_host_callbacks->push_back([buf_raw] { delete[] buf_raw; });
  }

 private:
  Kind kind_;
  const HloInstruction* hlo_instruction_;
  string profile_annotation_;
};

// A sequence of thunks.
using ThunkSequence = std::vector<std::unique_ptr<Thunk>>;

absl::string_view ThunkKindToString(Thunk::Kind);
std::ostream& operator<<(std::ostream& os, Thunk::Kind kind);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
