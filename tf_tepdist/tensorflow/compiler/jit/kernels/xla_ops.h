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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_XLA_OPS_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_XLA_OPS_H_

#include <atomic>

#include <queue>
#include "absl/synchronization/mutex.h"

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"

#include "tensorflow/compiler/xla/literal.h"

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"

#include "tensorflow/core/platform/mutex.h"

#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace tensorflow {

// Holds some information about the platform on which an
// XlaLaunch/_XlaCompile/_XlaRun op must run on.
class XlaPlatformInfo {
 public:
  XlaPlatformInfo() : device_type_("") {}
  XlaPlatformInfo(XlaPlatformInfo&&) = default;
  explicit XlaPlatformInfo(const DeviceType device_type,
                           se::Platform::Id platform_id,
                           const XlaDevice::Metadata* xla_device_metadata,
                           se::DeviceMemoryAllocator* device_allocator)
      : device_type_(device_type),
        platform_id_(platform_id),
        xla_device_metadata_(xla_device_metadata),
        device_allocator_(device_allocator) {}

  XlaPlatformInfo& operator=(XlaPlatformInfo&& other) = default;

  bool UseMultipleStreams() const {
    return xla_device_metadata_ && xla_device_metadata_->UseMultipleStreams();
  }

  // Non-null only when run on an XLA device.
  se::DeviceMemoryAllocator* custom_allocator() const {
    return device_allocator_;
  }

  DeviceType device_type() const { return device_type_; }

  // This is equal to xla_device_metadata()->platform()->id() if
  // xla_device_metadata() is not nullptr.
  se::Platform::Id platform_id() const { return platform_id_; }

  // This may be null if the op this XlaPlatformInfo is for was not placed on an
  // XLA device.
  const XlaDevice::Metadata* xla_device_metadata() const {
    return xla_device_metadata_;
  }
  bool is_on_xla_device() const { return xla_device_metadata() != nullptr; }

 private:
  DeviceType device_type_;
  se::Platform::Id platform_id_;

  // xla_device_metadata_ lives in the tensorflow::DeviceBase in which the
  // XlaLaunch/_XlaCompile/_XlaRun op is placed and thus does not die before the
  // XlaLaunch/_XlaCompile/_XlaRun OpKernel.
  const XlaDevice::Metadata* xla_device_metadata_;

  // If the op associated with this XlaPlatformInfo is placed on an XLA device
  // then device_allocator_ is the xla::Backend's memory allocator.  If the op
  // is placed on a regular CPU or GPU device then device_allocator_ is null.
  se::DeviceMemoryAllocator* device_allocator_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaPlatformInfo);
};

// XlaLocalLaunchBase is almost the same as XlaLocalLaunchOp.
// The only difference is that it does not require arguments to follow
// the "constants, then regular args, then resources" order.
// It takes vectors of constant and resource arguments explicitly.
// It does not have corresponding OpDef because it is never present
// in the GraphDef.
// Currently, it is used by eager runtime. FunctionLibraryRuntime creates
// this kernel when asked to create a kernel for an XLA-compiled function.
//
// `has_ref_vars`: whether the input computation can have reference variables.
// TODO(cheshire): instead derive this information from the input graph.
class XlaLocalLaunchBase : public OpKernel {
 public:
  XlaLocalLaunchBase(OpKernelConstruction* ctx,
                     const std::vector<int>& constants,
                     const std::vector<int>& resources,
                     const NameAttrList& function, bool has_ref_vars);
  XlaLocalLaunchBase(const XlaLocalLaunchBase&) = delete;
  XlaLocalLaunchBase& operator=(const XlaLocalLaunchBase&) = delete;
  ~XlaLocalLaunchBase() override = default;

  void Compute(OpKernelContext* ctx) override;

 protected:
  // Indexes of compile-time constant inputs
  const std::vector<int> constants_;
  // Indexes of resource inputs
  const std::vector<int> resources_;

  const NameAttrList function_;
  const XlaPlatformInfo platform_info_;

  bool has_ref_vars_;
};

// XlaLocalLaunchOp is used to replace a region of the TensorFlow graph
// which will be compiled and executed using XLA.  The XlaLocalLaunchOp is
// responsible for handling interactions with the TensorFlow executor.
// Once all inputs are present, and their shapes are known, the op can
// use a 'XlaCompilationCache' to compile and execute code which is specific
// to the shapes of input Tensors.
// XlaLocalLaunchOp uses xla::LocalClient::Compile() and
// xla::LocalExecutable::Run(), and passes arguments into/out of XLA in device
// memory.
class XlaLocalLaunchOp : public XlaLocalLaunchBase {
 public:
  explicit XlaLocalLaunchOp(OpKernelConstruction* ctx);
  ~XlaLocalLaunchOp() override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaLocalLaunchOp);
};

class XlaCompileOp : public OpKernel {
 public:
  explicit XlaCompileOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  // Indexes of compile-time constant inputs
  const std::vector<int> constants_;
  // Indexes of resource inputs
  const std::vector<int> resources_;

  const NameAttrList function_;

  XlaPlatformInfo platform_info_;

  const bool must_compile_;

  // Whether the graph has TF reference variables.
  const bool has_ref_vars_;

  // cannot_compile_cluster_ is set to true if XLA returns an Unimplemented
  // error when compiling the cluster this _XlaCompile is supposed to compile.
  // If `cannot_compile_cluster_` is true then we avoid compiling this cluster
  // on any future calls to _XlaCompile.
  bool cannot_compile_cluster_ TF_GUARDED_BY(cannot_compile_cluster_mu_) =
      false;

  mutex cannot_compile_cluster_mu_;
};

// NOTE: copied from
// https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/compiler/xla/pjrt/semaphore.h
class Semaphore {
 public:
  explicit Semaphore(int64 capacity);

  // Acquires `amount` units. Blocks until `amount` units are available.
  void Acquire(int64 amount);

  // Returns `amount` units to the semaphore.
  void Release(int64 amount);

 private:
  struct CanAcquireArgs {
    Semaphore* semaphore;
    int64 amount;
  };
  static bool CanAcquire(CanAcquireArgs* args)
      EXCLUSIVE_LOCKS_REQUIRED(args->semaphore->mu_);

  absl::Mutex mu_;
  int64 value_ GUARDED_BY(mu_);
};

class XlaRunOp : public OpKernel {
 public:
  explicit XlaRunOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

  xla::Literal XlaRunOpRPCImpl(
      xla::LocalClient* client,
      xla::Client* rpc_client,
      xla::ExecutionPlanHandle execution_handle,
      std::vector<xla::ShapedBuffer*> var_args,
      bool fetch_and_apply_resource_variables,
      bool trans_from_host);

 private:
  const XlaPlatformInfo platform_info_;

  mutex mu_;
  // Lazy xla output without updated resource variables fetched from RPC server.
  xla::Literal lazy_xla_output_ GUARDED_BY(mu_);
  // Lazy xla output with *updated variables* from server.
  xla::Literal lazy_xla_output_with_resource_vars_ GUARDED_BY(mu_);
  // Semaphore used to limit on the number *active* `ExecutePlan` RPCs.
  // Before starting a new `ExecutePlan` RPC, client try to acquire a permit.
  // If we get one, the RPC can start. If not, client will wait until one is
  // available. When an `ExecutePlan` RPC completes (either success or failure),
  // we return the permit.  Currently the semaphore capacity is set to 2.
  std::unique_ptr<Semaphore> execute_plan_semaphore_;

  int64 client_step_count_ = 0;

  // FIFO queue for storing adjacent iterations' inputs.
  mutex adjacent_iter_inputs_mu_;
  int64 num_parallel_rpc_steps_;
  std::queue<std::vector<xla::Literal>> adjacent_iter_inputs_
      GUARDED_BY(adjacent_iter_inputs_mu_);
      
};

class XlaMergeOp : public OpKernel {
 public:
  explicit XlaMergeOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_XLA_LAUNCH_OP_H_
