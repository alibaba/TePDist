/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include <cstddef>
#include <memory>
#include <string>
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

class ExecuteNodeArgs : public EagerKernelArgs {
 public:
  explicit ExecuteNodeArgs(int count) : EagerKernelArgs(count) {}

  Status Init(EagerContext* ctx,
              const absl::InlinedVector<TensorHandle*, 4>& op_inputs,
              const core::RefCountPtr<KernelAndDevice>& kernel);

  bool HasRemoteOrPackedInputs() const override {
    return has_remote_inputs_ || has_packed_inputs_;
  };

#if !defined(IS_MOBILE_PLATFORM)
  Status GetRemoteArg(const FunctionArgIndex& index,
                      eager::RemoteTensorHandle* val) const override {
    return serialize_remote_handle_(index, val);
  }
#endif  // IS_MOBILE_PLATFORM

 private:
  bool has_remote_inputs_ = false;
  bool has_packed_inputs_ = false;
#if !defined(IS_MOBILE_PLATFORM)
  std::function<Status(const FunctionArgIndex&, eager::RemoteTensorHandle*)>
      serialize_remote_handle_;
#endif  // IS_MOBILE_PLATFORM
};

class ExecuteNode : public EagerNode {
 public:
  ExecuteNode(
      EagerContext* ctx, const absl::InlinedVector<TensorHandle*, 4>& inputs,
      const absl::optional<EagerRemoteFunctionParams>& remote_func_params,
      const core::RefCountPtr<KernelAndDevice>& kernel,
      GraphCollector* graph_collector,
      CancellationManager* cancellation_manager,
      absl::Span<TensorHandle*> retvals)
      : EagerNode(),
        ctx_(ctx),
        inputs_(inputs),
        remote_func_params_(remote_func_params),
        kernel_(kernel),
        graph_collector_(graph_collector),
        cancellation_manager_(cancellation_manager),
        retvals_(retvals) {}

  Status Run() override {
    int i = 0;
    for (TensorHandle* h : inputs_) {
      if (h->RefCountIsOne()) {
        const Device* d = ctx_->CanonicalDevice(kernel_->InputDevice(i));
        Status s = h->Unprotect(d);
        if (!s.ok()) {
          VLOG(1) << "Unable to unprotect tensor: " << s;
        }
      }
      ++i;
    }
    return EagerKernelExecute(ctx_, inputs_, remote_func_params_, kernel_,
                              graph_collector_, cancellation_manager_,
                              retvals_);
  }

  void Abort(Status status) override {}

  std::string DebugString() const override {
    std::string out = "[ExecuteNode]";
    strings::StrAppend(&out, " kernel: ", kernel_->name());
    return out;
  }

 private:
  EagerContext* ctx_;
  const absl::InlinedVector<TensorHandle*, 4>& inputs_;
  const absl::optional<EagerRemoteFunctionParams>& remote_func_params_;
  const core::RefCountPtr<KernelAndDevice>& kernel_;
  GraphCollector* graph_collector_;
  CancellationManager* const cancellation_manager_;
  absl::Span<TensorHandle*> retvals_;
};

class AsyncExecuteNode : public EagerNode {
 public:
  AsyncExecuteNode(
      EagerContext* ctx, const absl::InlinedVector<TensorHandle*, 4>& inputs,
      const absl::optional<EagerRemoteFunctionParams>& remote_func_params,
      core::RefCountPtr<KernelAndDevice> kernel,
      GraphCollector* graph_collector,
      CancellationManager* cancellation_manager,
      absl::Span<TensorHandle*> retvals)
      : EagerNode(),
        ctx_(ctx),
        inputs_(inputs),
        remote_func_params_(remote_func_params),
        kernel_(std::move(kernel)),
        graph_collector_(graph_collector),
        cancellation_manager_(cancellation_manager) {
    // Copy the output handles, since the container for them might get
    // destroyed.
    for (auto handle : retvals) {
      handle->Ref();
      retvals_.push_back(handle);
    }

    // This is required to ensure that the tensor handles stay alive across
    // the execution.
    for (auto handle : inputs_) {
      handle->Ref();
    }
  }

  ~AsyncExecuteNode() override {
    for (auto handle : retvals_) {
      handle->Unref();
    }

    for (auto handle : inputs_) {
      handle->Unref();
    }
  }

  Status Run() override {
    int i = 0;
    for (TensorHandle* h : inputs_) {
      if (h->RefCountIsOne()) {
        const Device* d = ctx_->CanonicalDevice(kernel_->InputDevice(i));
        Status s = h->Unprotect(d);
        if (!s.ok()) {
          VLOG(1) << "Unable to unprotect tensor: " << s;
        }
      }
      ++i;
    }
    const Status status = EagerKernelExecute(
        ctx_, inputs_, remote_func_params_, kernel_, graph_collector_,
        cancellation_manager_, absl::MakeSpan(retvals_));
    if (!status.ok()) {
      Abort(status);
      return status;
    }
    // If status is ok, EagerKernelExecute would have called SetTensor on
    // all the output handles.
    return Status::OK();
  }

  void Abort(Status status) override {
    int i = 0;
    for (auto handle : retvals_) {
      handle->Poison(status, ctx_->CanonicalDevice(kernel_->OutputDevice(i)));
      ++i;
    }
  }

  std::string DebugString() const override {
    std::string out = "[AsyncExecuteNode]";
    strings::StrAppend(&out, " kernel: ", kernel_->name());
    return out;
  }

 private:
  EagerContext* ctx_;
  absl::InlinedVector<TensorHandle*, 4> inputs_;
  const absl::optional<EagerRemoteFunctionParams> remote_func_params_;
  core::RefCountPtr<KernelAndDevice> kernel_;
  GraphCollector* graph_collector_;
  CancellationManager* const cancellation_manager_;
  absl::InlinedVector<TensorHandle*, 2> retvals_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_
