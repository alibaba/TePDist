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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_EXECUTABLE_H_

#include <vector>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/backend.h"

namespace xla {

class LocalExecutable {
 public:
  // Low-level constructor; LocalClient::Compile() is the usual way to create
  // executables.
  LocalExecutable(std::unique_ptr<Executable> executable, Backend* backend,
                  ExecutableBuildOptions build_options);

  // Run the compiled computation with the given arguments and options and
  // return the result.
  StatusOr<ScopedShapedBuffer> Run(
      const absl::Span<const ShapedBuffer* const> arguments,
      ExecutableRunOptions run_options);

  // Similar to Run(), but need not block the host waiting for the computation
  // to complete before returning.
  StatusOr<ScopedShapedBuffer> RunAsync(
      const absl::Span<const ShapedBuffer* const> arguments,
      ExecutableRunOptions run_options);

  // Similar to RunAsync(), but allows for donating argument buffers to the
  // executable.
  StatusOr<ExecutionOutput> RunAsync(
      absl::Span<Shape const* const> argument_host_shapes,
      std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options,
      void* opaque = nullptr);

  // Return the options used to build the executable.
  const ExecutableBuildOptions& build_options() const { return build_options_; }

  // Return the built executable.
  Executable* executable() const { return executable_.get(); }

 private:
  // Validates that the given arguments and options satisfy various constraints
  // of the computation.
  //
  // The given ExecutableRunOptions override any values from TF_XLA_FLAGS
  // environment variable.
  Status ValidateExecutionOptions(const ExecutableRunOptions& run_options,
                                  const Backend& backend);

  // Returns a literal containing the contents of the given ShapedBuffer.
  StatusOr<Literal> LiteralFromShapedBuffer(const ShapedBuffer& shaped_buffer);

  StatusOr<std::pair<ServiceExecutableRunOptions, StreamPool::Ptr>> RunHelper(
      const absl::Span<const Shape* const> argument_shapes,
      ExecutableRunOptions run_options);

  // The ordinal of the device which this executable was compiled for. The
  // executable can run on all equivalent devices (as determined by
  // Backend::devices_equivalent).
  int build_device_ordinal() const { return build_options_.device_ordinal(); }

  // Compiled computation.
  std::unique_ptr<Executable> executable_;

  // Execution backend.
  Backend* backend_ = nullptr;

  // Options used to build the executable.
  const ExecutableBuildOptions build_options_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_EXECUTABLE_H_

