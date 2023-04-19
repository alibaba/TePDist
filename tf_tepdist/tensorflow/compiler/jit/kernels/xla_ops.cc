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

#include "tensorflow/compiler/jit/kernels/xla_ops.h"

#include <chrono>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/util/env_var.h"

// OP_REQUIRES_OK_RETURN is the same as OP_REQUIRES_OK except that
// in error case, it returns RET instead of void.
#define OP_REQUIRES_OK_RETURN(CTX, RET, ...)                \
  do {                                                      \
    ::tensorflow::Status _s(__VA_ARGS__);                   \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      return RET;                                           \
    }                                                       \
  } while (0)

namespace tensorflow {

namespace {

XlaPlatformInfo PlatformInfoFromContext(OpKernelConstruction* ctx) {
  DeviceType device_type = ctx->device_type();
  se::Platform::Id platform_id = nullptr;
  const XlaDevice::Metadata* xla_device_metadata = nullptr;
  se::DeviceMemoryAllocator* custom_allocator = nullptr;

  if (ctx->device_type() == DeviceType(DEVICE_CPU)) {
    platform_id = se::host::kHostPlatformId;
  } else if (ctx->device_type() == DeviceType(DEVICE_GPU)) {
    platform_id = ctx->device()
                      ->tensorflow_gpu_device_info()
                      ->stream->parent()
                      ->platform()
                      ->id();
  } else if (XlaDevice::GetMetadata(ctx, &xla_device_metadata).ok()) {
    // If we are on an XlaDevice, use the underlying XLA platform's allocator
    // directly. We could use the StreamExecutor's allocator which may
    // theoretically be more correct, but XLA returns a nice OOM message in a
    // Status and StreamExecutor does not.
    // 
    // Importantly we can't use ctx->device()->GetAllocator() as the allocator
    // (which xla_allocator above uses) as on an XlaDevice, this is a dummy
    // allocator that returns XlaTensor objects. The XlaCompiler needs a real
    // allocator to allocate real buffers.
    platform_id = xla_device_metadata->platform()->id();
    custom_allocator =
        xla_device_metadata->client()->backend().memory_allocator();
  }

  return XlaPlatformInfo(device_type, platform_id, xla_device_metadata,
                         custom_allocator);
}

// A closure describing how to run a compiled version of a TensorFlow function.
//
// It may seem unusual to stick the resource variable snapshots in this class.
// This is necessary: we need to use the snapshots observed by the compiler as
// the initial values for the resource variables (and cannot snapshot them again
// during execution) because otherwise we risk observing a different snapshot
// with shapes different from what we compiled for.
class XlaExecutableClosure {
 public:
  explicit XlaExecutableClosure(
      xla::LocalClient* client, xla::LocalExecutable* executable,
      const XlaCompiler::CompilationResult* compilation_result,
      std::map<int, OptionalTensor> resource_var_snapshots,
      int num_constant_args,
      xla::ExecutionPlanHandle handle,
      xla::Client* rpc_client=nullptr)
      : client_(client),
        executable_(executable),
        compilation_result_(compilation_result),
        resource_var_snapshots_(std::move(resource_var_snapshots)),
        num_constant_args_(num_constant_args),
        handle_(handle),
        rpc_client_(rpc_client) {}

  XlaExecutableClosure(XlaExecutableClosure&&) = default;
  XlaExecutableClosure& operator=(XlaExecutableClosure&&) = default;

  xla::LocalClient* client() const { return client_; }
  xla::Client* rpc_client() const { return rpc_client_; }
  xla::LocalExecutable* executable() const { return executable_; }
  xla::ExecutionPlanHandle handle() const {return handle_; }
  const XlaCompiler::CompilationResult* compilation_result() const {
    return compilation_result_;
  }
  const std::map<int, OptionalTensor>& resource_var_snapshots() const {
    return resource_var_snapshots_;
  }
  int num_constant_args() const { return num_constant_args_; }

 private:
  xla::LocalClient* client_;
  xla::Client* rpc_client_;
  xla::LocalExecutable* executable_;
  xla::ExecutionPlanHandle handle_;
  const XlaCompiler::CompilationResult* compilation_result_;
  std::map<int, OptionalTensor> resource_var_snapshots_;
  int num_constant_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaExecutableClosure);
};

// This maintains a mapping from a globally unique ID to XlaExecutableClosure
// instances.
class XlaExecutableClosureStore {
 public:
  XlaExecutableClosureStore() : key_counter_(0) {}

  using KeyT = string;

  KeyT Produce(XlaExecutableClosure result) {
    mutex_lock l(mutex_);
    KeyT key = absl::StrCat(key_counter_++);
    bool insert_successful = closures_.emplace(key, std::move(result)).second;
    DCHECK(insert_successful);
    (void)insert_successful;
    return key;
  }

  XlaExecutableClosure Consume(const KeyT& key) {
    mutex_lock l(mutex_);
    auto it = closures_.find(key);
    DCHECK(it != closures_.end());
    XlaExecutableClosure value = std::move(it->second);
    closures_.erase(it);
    return value;
  }

  static XlaExecutableClosureStore* Global() {
    static XlaExecutableClosureStore* instance = new XlaExecutableClosureStore;
    return instance;
  }

 private:
  mutex mutex_;
  int64 key_counter_ TF_GUARDED_BY(mutex_);
  absl::flat_hash_map<KeyT, XlaExecutableClosure> closures_
      TF_GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaExecutableClosureStore);
};

// Return allocator from platform info if non-null, or populate and return a
// pointer to the allocator adapter with allocator from context.
//
// This is necessary because for XLA devices the underlying TF allocator returns
// dummy tensors.
se::DeviceMemoryAllocator* GetAllocator(
    absl::optional<se::TfAllocatorAdapter>* tf_allocator_adapter,
    OpKernelContext* ctx, const XlaPlatformInfo& platform_info) {
  if (platform_info.custom_allocator()) {
    return platform_info.custom_allocator();
  }
  if (!ctx->op_device_context()) {
    // Stream is not set for the host platform.
    se::Platform* platform =
        se::MultiPlatformManager::PlatformWithId(platform_info.platform_id())
            .ValueOrDie();
    tf_allocator_adapter->emplace(ctx->device()->GetAllocator({}), platform);
    return &tf_allocator_adapter->value();
  }
  // platform_info.
  tf_allocator_adapter->emplace(ctx->device()->GetAllocator({}),
                                ctx->op_device_context()->stream());
  return &tf_allocator_adapter->value();
}

}  // namespace

XlaLocalLaunchBase::XlaLocalLaunchBase(OpKernelConstruction* ctx,
                                       const std::vector<int>& constants,
                                       const std::vector<int>& resources,
                                       const NameAttrList& function,
                                       bool has_ref_vars)
    : OpKernel(ctx),
      constants_(constants),
      resources_(resources),
      function_(function),
      platform_info_(PlatformInfoFromContext(ctx)),
      has_ref_vars_(has_ref_vars) {}

static Status BuildCompilationCache(OpKernelContext* ctx,
                                    const XlaPlatformInfo& platform_info,
                                    XlaCompilationCache** cache) {
  if (platform_info.xla_device_metadata()) {
    *cache = new XlaCompilationCache(
        platform_info.xla_device_metadata()->client(),
        platform_info.xla_device_metadata()->jit_device_type());
    return Status::OK();
  }

  auto platform =
      se::MultiPlatformManager::PlatformWithId(platform_info.platform_id());
  if (!platform.ok()) {
    return platform.status();
  }

  xla::StatusOr<xla::Compiler*> compiler_for_platform =
      xla::Compiler::GetForPlatform(platform.ValueOrDie());
  if (!compiler_for_platform.ok()) {
    // In some rare cases (usually in unit tests with very small clusters) we
    // may end up transforming an XLA cluster with at least one GPU operation
    // (which would normally force the cluster to be compiled using XLA:GPU)
    // into an XLA cluster with no GPU operations (i.e. containing only CPU
    // operations).  Such a cluster can fail compilation (in way that
    // MarkForCompilation could not have detected) if the CPU JIT is not linked
    // in.
    //
    // So bail out of _XlaCompile in this case, and let the executor handle the
    // situation for us.
    const Status& status = compiler_for_platform.status();
    if (status.code() == error::NOT_FOUND) {
      return errors::Unimplemented("Could not find compiler for platform ",
                                   platform.ValueOrDie()->Name(), ": ",
                                   status.ToString());
    }
  }

  xla::LocalClientOptions client_options;
  client_options.set_platform(platform.ValueOrDie());
  client_options.set_intra_op_parallelism_threads(
      ctx->device()->tensorflow_cpu_worker_threads()->num_threads);
  auto client = xla::ClientLibrary::GetOrCreateLocalClient(client_options);
  auto rpc_client = xla::ClientLibrary::GetRPCClient(platform.ValueOrDie());
  if (!client.ok()) {
    return client.status();
  }
  const XlaOpRegistry::DeviceRegistration* registration;
  if (!XlaOpRegistry::GetCompilationDevice(platform_info.device_type().type(),
                                           &registration)) {
    return errors::InvalidArgument("No JIT device registered for ",
                                   platform_info.device_type().type());
  }
  *cache = new XlaCompilationCache(
      client.ValueOrDie(), DeviceType(registration->compilation_device_name),
      rpc_client.ValueOrDie());
  return Status::OK();
}

static Status CompileToLocalExecutable(
    OpKernelContext* ctx, const NameAttrList& function, bool has_ref_vars,
    const XlaPlatformInfo& platform_info, absl::Span<const int> resources,
    absl::Span<const int> constants, bool lazy, xla::LocalClient** client,
    std::map<int, OptionalTensor>* variables,
    const XlaCompiler::CompilationResult** kernel,
    xla::LocalExecutable** executable,
    xla::ExecutionPlanHandle* handle,
    xla::Client** rpc_client) {
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  if (!rm) {
    return errors::Internal("No resource manager.");
  }

  XlaCompilationCache* cache;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaCompilationCache>(
      rm->default_container(), "xla_cache", &cache,
      [&](XlaCompilationCache** cache) {
        return BuildCompilationCache(ctx, platform_info, cache);
      }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref cache_ref(cache);

  TF_RETURN_IF_ERROR(SnapshotResourceVariables(ctx, resources, variables));
  *client = static_cast<xla::LocalClient*>(cache->client());
  *rpc_client = static_cast<xla::Client*>(cache->rpc_client());

  absl::optional<se::TfAllocatorAdapter> tf_allocator_adapter;
  XlaCompiler::Options options;
  options.client = *client;
  if (ctx->op_device_context() != nullptr) {
    options.device_ordinal =
        ctx->op_device_context()->stream()->parent()->device_ordinal();
  }
  options.device_type = cache->device_type();
  options.flib_def = ctx->function_library()->GetFunctionLibraryDefinition();
  options.graph_def_version = ctx->function_library()->graph_def_version();
  options.allow_cpu_custom_calls =
      (platform_info.platform_id() == se::host::kHostPlatformId);
  options.device_allocator =
      GetAllocator(&tf_allocator_adapter, ctx, platform_info);
  if (platform_info.xla_device_metadata()) {
    options.shape_representation_fn =
        platform_info.xla_device_metadata()->shape_representation_fn();
  }
  // If reference variables are not present in the graph, we can safely alias
  // passthrough parameters without performing a copy.
  options.alias_passthrough_params =
      !has_ref_vars && !platform_info.is_on_xla_device();

  std::map<int, Tensor> constant_args;
  for (int i : constants) {
    constant_args.insert({i, ctx->input(i)});
  }
  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;
  // Optimization: where possible, have the computation return a naked array
  // rather than a one-element tuple.
  compile_options.always_return_tuple = false;

  std::vector<XlaCompiler::Argument> args;
  TF_RETURN_IF_ERROR(XlaComputationLaunchContext::BuildXlaCompilerArguments(
      constant_args, *variables, ctx, &args));
  return cache->Compile(options, function, args, compile_options,
                        lazy ? XlaCompilationCache::CompileMode::kLazy
                             : XlaCompilationCache::CompileMode::kStrict,
                        kernel, executable, *rpc_client ? handle : nullptr);
}

void XlaLocalLaunchBase::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XlaLocalLaunchOpBase::Compute "
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));

  xla::LocalClient* client;
  const XlaCompiler::CompilationResult* kernel;
  xla::LocalExecutable* executable;
  std::map<int, OptionalTensor> variables;

  {
    Status s = CompileToLocalExecutable(
        ctx, function_, /*has_ref_vars=*/has_ref_vars_, platform_info_,
        resources_, constants_, /*lazy=*/false, &client, &variables, &kernel,
        &executable, /*execution_handle*/nullptr, /*rpc_client*/nullptr);
    OP_REQUIRES_OK(ctx, s);
  }

  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

  VLOG(1) << "Executing XLA Computation...";

  absl::optional<se::TfAllocatorAdapter> tf_allocator_adapter;
  se::DeviceMemoryAllocator* allocator =
      GetAllocator(&tf_allocator_adapter, ctx, platform_info_);
  XlaComputationLaunchContext launch_context(
      client, allocator,
      /*allocate_xla_tensors=*/platform_info_.is_on_xla_device(),
      platform_info_.UseMultipleStreams());
  launch_context.PopulateInputs(ctx, kernel, variables,
                                /*missing_ctx_input_prefix=*/0);

  // Execute the computation.
  VLOG(2) << "Executing computation.";
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(stream);
  run_options.set_allocator(allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(GetXLARandomSeed());
  xla::ThenExecuteFunction then_execute;
  if (ctx->op_device_context()) {
    then_execute = [&](se::Stream* stream, std::function<void()> fn) {
      Status status = ctx->op_device_context()->ThenExecute(
          down_cast<Device*>(ctx->device()), stream, std::move(fn));
      if (!status.ok()) {
        // This should never happen.
        LOG(ERROR) << "ThenExecute failed " << status;
      }
    };
    run_options.set_then_execute_function(&then_execute);
  }
  Env* env = Env::Default();
  auto start_time = env->NowMicros();

  xla::StatusOr<xla::ScopedShapedBuffer> run_result;
  if (!stream || platform_info_.platform_id() == se::host::kHostPlatformId) {
    run_result = executable->Run(launch_context.arguments(), run_options);
  } else {
    run_result = executable->RunAsync(launch_context.arguments(), run_options);
  }
  OP_REQUIRES(ctx, run_result.ok(), run_result.status());

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time: " << elapsed << "us";

  const xla::HloInputOutputAliasConfig& input_output_alias =
      executable->executable()->module().input_output_alias_config();
  OP_REQUIRES_OK(
      ctx, launch_context.PopulateOutputs(
               ctx, kernel, run_result.ConsumeValueOrDie(),
               /*missing_ctx_input_prefix=*/0, input_output_alias, variables));
  VLOG(1) << "Done";
}

namespace {
// Helper static functions to construct parameters for
// XlaLocalLaunchBase constructor from OpKernelConstruction.
std::vector<int> ConstantsVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Tconstants", &constant_types));
  std::vector<int> constants(constant_types.size());
  std::iota(constants.begin(), constants.end(), 0);
  return constants;
}

std::vector<int> ResourcesVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Tconstants", &constant_types));

  DataTypeVector arg_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Targs", &arg_types));

  int num_resources;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Nresources", &num_resources));

  std::vector<int> resources(num_resources);
  std::iota(resources.begin(), resources.end(),
            constant_types.size() + arg_types.size());
  return resources;
}

NameAttrList FunctionAttr(OpKernelConstruction* ctx) {
  const NameAttrList* func;
  OP_REQUIRES_OK_RETURN(ctx, NameAttrList(), ctx->GetAttr("function", &func));
  return *func;
}

bool MustCompileAttr(OpKernelConstruction* ctx) {
  bool must_compile;
  OP_REQUIRES_OK_RETURN(ctx, false,
                        ctx->GetAttr("must_compile", &must_compile));
  return must_compile;
}

bool HasRefVars(OpKernelConstruction* ctx) {
  bool has_ref_vars;
  OP_REQUIRES_OK_RETURN(ctx, false,
                        ctx->GetAttr(kXlaHasReferenceVarsAttr, &has_ref_vars));
  return has_ref_vars;
}

}  // namespace

XlaLocalLaunchOp::XlaLocalLaunchOp(OpKernelConstruction* ctx)
    : XlaLocalLaunchBase(ctx, ConstantsVector(ctx), ResourcesVector(ctx),
                         FunctionAttr(ctx), /*has_ref_vars=*/true) {}

XlaLocalLaunchOp::~XlaLocalLaunchOp() {
  VLOG(1) << "XlaLocalLaunchOp destroyed";
}

XlaCompileOp::XlaCompileOp(OpKernelConstruction* ctx)
    : OpKernel(ctx),
      constants_(ConstantsVector(ctx)),
      resources_(ResourcesVector(ctx)),
      function_(FunctionAttr(ctx)),
      platform_info_(PlatformInfoFromContext(ctx)),
      must_compile_(MustCompileAttr(ctx)),
      has_ref_vars_(HasRefVars(ctx)) {}

void XlaCompileOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaCompileOp " << def().name()
          << (must_compile_ ? "(must-compile)" : "");
  xla::LocalClient* client;
  xla::Client* rpc_client;
  const XlaCompiler::CompilationResult* kernel;
  xla::LocalExecutable* executable;
  xla::ExecutionPlanHandle handle;
  std::map<int, OptionalTensor> variables;

  bool cannot_compile_cluster;
  {
    mutex_lock guard(cannot_compile_cluster_mu_);
    cannot_compile_cluster = cannot_compile_cluster_;
  }

  if (GetXlaOpsCommonFlags().tf_xla_always_defer_compilation ||
      cannot_compile_cluster) {
    executable = nullptr;
  } else {
    Status status = CompileToLocalExecutable(
        ctx, function_, has_ref_vars_, platform_info_, resources_, constants_,
        /*lazy=*/!must_compile_, &client, &variables, &kernel, &executable, 
        &handle, &rpc_client);
    if (must_compile_ || status.code() != error::UNIMPLEMENTED) {
      OP_REQUIRES_OK(ctx, status);
    }

    if (status.code() == error::UNIMPLEMENTED) {
      LOG(WARNING) << "Compilation failed:" << status.ToString()
                   << ".  Falling back to TF function call.";

      BroadcastOptimizationRemark(
          XlaOptimizationRemark::UNIMPLEMENTED_OPERATION, status.ToString())
          .IgnoreError();
      executable = nullptr;
      mutex_lock guard(cannot_compile_cluster_mu_);
      cannot_compile_cluster_ = true;
    }
  }

  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = ctx->device()->GetAllocator(host_alloc_attrs);

  if (!executable && !rpc_client) {
    DCHECK(!must_compile_);
    Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));

    Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
    compilation_successful.scalar<bool>()() = false;
    ctx->set_output(0, Tensor(cpu_allocator, DT_STRING, TensorShape({})));
    ctx->set_output(1, compilation_successful);
    return;
  }

  // Each execution of an XlaCompile op creates a new XlaExecutableClosure, even
  // if it didn't have to compile the cluster because of a compilation-cache
  // hit.  This is because we at least need new snapshots of the resource
  // variables.
  XlaExecutableClosureStore::KeyT key =
      XlaExecutableClosureStore::Global()->Produce(XlaExecutableClosure(
          client, executable, kernel, std::move(variables), constants_.size(),
          handle, rpc_client));

  Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
  compilation_key.flat<tstring>()(0) = key;

  Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
  compilation_successful.flat<bool>()(0) = true;

  ctx->set_output(0, compilation_key);
  ctx->set_output(1, compilation_successful);
}

Semaphore::Semaphore(int64 capacity) : value_(capacity) {
  CHECK_GE(capacity, 0);
}

bool Semaphore::CanAcquire(CanAcquireArgs* args) {
  return args->semaphore->value_ >= args->amount;
}

void Semaphore::Acquire(int64 amount) {
  CHECK_GE(amount, 0);

  CanAcquireArgs args;
  args.semaphore = this;
  args.amount = amount;

  mu_.LockWhen(absl::Condition(&CanAcquire, &args));
  value_ -= amount;
  mu_.Unlock();
}

void Semaphore::Release(int64 amount) {
  CHECK_GE(amount, 0);
  absl::MutexLock lock(&mu_);
  value_ += amount;
}

XlaRunOp::XlaRunOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), platform_info_(PlatformInfoFromContext(ctx)) {
  // Environment variable to control the rpc_communication-computation
  // overlapping window size.
  //
  // E.g., (1) NUM_PARALLEL_RPC_STEPS=1: Pure synchronous mode, i.e. each
  // iteration's RPC calls wait for the server to respond before starting next
  // step training iteration. (2) NUM_PARALLEL_RPC_STEPS=2: Try overlapping the
  // (N+1) step's TransferToServerHost cost with the N-th step's kernel
  // execution time on the server side.
  auto s = ReadInt64FromEnvVar("NUM_PARALLEL_RPC_STEPS", 0,
                               &num_parallel_rpc_steps_);
  CHECK(s.ok() && num_parallel_rpc_steps_ >= 0 && num_parallel_rpc_steps_ <= 4)
      << "`NUM_PARALLEL_RPC_STEPS` means the number of consecutive training "
      << "iterations RPC request can be made, in order to achieve overlapping "
      << "of network transfer of inputs and server-side kernel computations. "
      << "Thus one nonnegative interger should be required.";
  VLOG(0) << "NUM_PARALLEL_RPC_STEPS=" << num_parallel_rpc_steps_;
  execute_plan_semaphore_.reset(new Semaphore(num_parallel_rpc_steps_));
}

void XlaRunOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaRunOp " << def().name();
  Tensor key_tensor = ctx->input(ctx->num_inputs() - 1);
  const XlaExecutableClosureStore::KeyT& key = key_tensor.flat<tstring>()(0);

  // Default to fetch resource variable *every* step, which is inefficient
  // while client side can always get the latest resource var from the server.
  // TODO (zycao): eliminate this part or make it more reasonable.
  int64 fetch_resource_var_steps = -1;
  auto status = ReadInt64FromEnvVar("FETCH_RESOURCE_VAR_STEPS", 100000,
                                    &fetch_resource_var_steps);
  CHECK(status.ok() && fetch_resource_var_steps >= 1)
      << "Invalid FETCH_RESOURCE_VAR_STEPS: " << fetch_resource_var_steps;
  bool fetch_and_apply_resource_variables =
      (client_step_count_ + 1) % fetch_resource_var_steps == 0 ? true : false;
  VLOG(2) << "fetch_and_apply_resource_variables = "
          << fetch_and_apply_resource_variables;

  XlaExecutableClosure closure =
      XlaExecutableClosureStore::Global()->Consume(key);

  absl::optional<se::TfAllocatorAdapter> tf_allocator_adapter;
  se::DeviceMemoryAllocator* allocator =
      GetAllocator(&tf_allocator_adapter, ctx, platform_info_);
  XlaComputationLaunchContext launch_context(
      closure.client(), allocator,
      /*allocate_xla_tensors=*/platform_info_.is_on_xla_device(),
      /*use_multiple_streams=*/platform_info_.UseMultipleStreams());
  launch_context.set_fetch_all_variables(fetch_and_apply_resource_variables);
  VLOG(2) << "is on xla device = " << platform_info_.is_on_xla_device();

  // We're missing the must-be-constant inputs, tell `PopulateInputs`
  // about this.  We don't actually need these inputs because they've
  // already been baked into the compiled kernel.
  {
    tensorflow::profiler::TraceMe hlo_module_activity(
        [&] {
          return absl::StrCat(
              "Populate Inputs (",
              closure.compilation_result()->xla_input_shapes.size(), ")");
        },
        tensorflow::profiler::TraceMeLevel::kInfo);

    launch_context.PopulateInputs(
        ctx, closure.compilation_result(), closure.resource_var_snapshots(),
        /*missing_ctx_input_prefix=*/closure.num_constant_args());
  }

  xla::LocalClient* client = closure.client();
  int num_devices = client->backend().device_count();

  auto rpc_client = closure.rpc_client();
  if (rpc_client) {
    CHECK_EQ(num_devices, 1);
    launch_context.SeparateArguments(
        closure.compilation_result(), closure.resource_var_snapshots());

    std::vector<xla::ShapedBuffer*> input_args = launch_context.input_arguments();
    std::vector<xla::ShapedBuffer*> var_args = launch_context.var_arguments();

    std::vector<xla::Literal> input_literals;
    for (auto *arg : input_args) {
      xla::Literal arg_literal =
          client->ShapedBufferToLiteral(*arg).ValueOrDie();
      input_literals.emplace_back(std::move(arg_literal));
    }
    adjacent_iter_inputs_.emplace(std::move(input_literals));
    auto execution_plan_handle = closure.handle();
    // num_parallel_rpc_steps_ == 0: means that the asynchronous RPC send/recv
    // feature is disabled.
    if (client_step_count_ <=1 || this->num_parallel_rpc_steps_ == 0) {
      //  Synchronous execution.
      
      VLOG(0) << "Start global_step = " << client_step_count_
              << ", thread id = " << std::this_thread::get_id();

      // Take the first two iteration as *warmup* iterations:
      //   - The first iteration fetch resource variables, which could be very
      //     large for big models (e.g., T5), and initialize variable member
      //     `lazy_xla_output_with_resource_vars_`
      //   - The second iteration does not fetch resource variables,
      //     and initialize `lazy_xla_output_`.
      //
      // In practice, we fetch and apply the resource variables as low frequency
      // as possible for performance considerion. Thus the lightweight one (i.e.
      // `lazy_xla_output_`) is consumed more frequently. These above
      // two-step-warmup is exactly designed for this purpose.
      if (client_step_count_ == 0) {
        // Fetch and apply resource variables for the *first* step
        // fetch_and_apply_resource_variables = true;
        fetch_and_apply_resource_variables = false;
      } else if (client_step_count_ == 1) {
        // Do not fetch and apply resource variables for the second step.
        fetch_and_apply_resource_variables = false;
      }
      auto result_literal = XlaRunOpRPCImpl(client,
          rpc_client,
          execution_plan_handle,
          var_args,
          fetch_and_apply_resource_variables,
          launch_context.init_from_client());

      //if (client_step_count_ == 0 || fetch_and_apply_resource_variables) {
      if (fetch_and_apply_resource_variables) {
        lazy_xla_output_with_resource_vars_ = std::move(result_literal);
      } else {
        lazy_xla_output_ = std::move(result_literal);
      }
    } else {
      // Launch rpc_client::ExecutePlan asynchronously.
      execute_plan_semaphore_->Acquire(1);
      auto t = std::thread([this, client, rpc_client, execution_plan_handle,
          var_args, fetch_and_apply_resource_variables, &launch_context]() {
        auto result_literal = XlaRunOpRPCImpl(client,
            rpc_client,
            execution_plan_handle,
            var_args,
            fetch_and_apply_resource_variables,
            launch_context.init_from_client());
        if (fetch_and_apply_resource_variables) {
          mutex_lock lock(mu_);
          lazy_xla_output_with_resource_vars_ = std::move(result_literal);
        } else {
          mutex_lock lock(mu_);
          lazy_xla_output_ = std::move(result_literal);
        }
        // Unlock another thread, if any.
        execute_plan_semaphore_->Release(1);
      });
      VLOG(0) << "Start global_step = " << client_step_count_;
      t.detach();
    }
    client_step_count_ ++;
    xla::Literal result_literal;
    if (fetch_and_apply_resource_variables) {
      // N.B., when the magnitude of the source variables is large,
      // this operation is time-consuming.
      mutex_lock lock(mu_);
      result_literal = lazy_xla_output_with_resource_vars_.Clone();
    } else {
      mutex_lock lock(mu_);
      result_literal = lazy_xla_output_.Clone();
    }
    auto device_assignment =
        client->backend().computation_placer()->AssignDevices(
            num_devices, /*computation_count=*/1).ValueOrDie();
    auto device_ordinal = device_assignment(0, 0);
    auto run_result = client->LiteralToShapedBuffer(
        result_literal, device_ordinal, platform_info_.custom_allocator());

    xla::HloInputOutputAliasConfig input_output_alias;
    OP_REQUIRES_OK(
        ctx, launch_context.PopulateOutputs(
                  ctx, closure.compilation_result(), run_result.ConsumeValueOrDie(),
                  /*missing_ctx_input_prefix=*/closure.num_constant_args(),
                  input_output_alias, closure.resource_var_snapshots(),
                  &launch_context.vars_update_set()));
    return;
  }
}

xla::Literal XlaRunOp::XlaRunOpRPCImpl(
    xla::LocalClient* client, xla::Client* rpc_client,
    xla::ExecutionPlanHandle execution_handle,
    std::vector<xla::ShapedBuffer*> var_args,
    bool fetch_and_apply_resource_variables,
    bool trans_from_host) {
  VLOG(2) << "execution_handle = " << execution_handle.handle(); 
  // 1. Transfer input data to server.
  CHECK(!adjacent_iter_inputs_.empty());
  std::vector<std::unique_ptr<xla::GlobalData>> inputs_global;
  {
    mutex_lock lock(adjacent_iter_inputs_mu_);
    auto& input_literals = adjacent_iter_inputs_.front();
    for (int g_idx = 0; g_idx < input_literals.size(); ++g_idx) {
      auto& arg_literal = input_literals[g_idx];
      auto arg_global =
          rpc_client->TransferToServerHost(
              arg_literal, /*shape_with_layout=*/nullptr,
              /*device_handle=*/nullptr, /*variable=*/false,
              true/*trans_from_host*/, g_idx).ConsumeValueOrDie();
      inputs_global.emplace_back(std::move(arg_global));
    }
    // This calls the removed element's destructor.
    adjacent_iter_inputs_.pop();
  }
  
  std::vector<xla::GlobalData*> args_global_transformed;
  for (auto& arg_global : inputs_global) {
    args_global_transformed.emplace_back(arg_global.get());
  }
  
  // 2. Transfer variables to server or fetch variable handle from cache.
  if (!rpc_client->VarsCacheInRemote()) {
    
    std::vector<xla::Literal> var_literals;
    for (auto *arg : var_args) {
      if (trans_from_host) {
        xla::Literal arg_literal =
            client->ShapedBufferToLiteral(*arg).ValueOrDie();
        var_literals.emplace_back(std::move(arg_literal));
      } else {
        xla::Literal fake_literal;
        var_literals.emplace_back(std::move(fake_literal));
      }
    }
    auto mutable_global_vars = rpc_client->mutable_global_vars();
    for (int i = 0; i < var_literals.size(); ++i) {
      auto& arg_literal = var_literals[i];
      if (arg_literal.shape().IsArray()) {
        int64 size = xla::ShapeUtil::ByteSizeOf(arg_literal.shape());
        CHECK_LT(size, 2147483647)  // max int32 = 2147483647
            << "exceeds 2GB hard limit of google protobuf!";
      }
      auto* arg_shaped_buf = var_args[i];
      auto arg_global =
          rpc_client->TransferToServerHost(
              arg_literal, /*shape_with_layout=*/&(arg_shaped_buf->on_host_shape()),
              /*device_handle=*/nullptr, /*variable=*/true,
              trans_from_host, i + args_global_transformed.size()).ConsumeValueOrDie();
      mutable_global_vars->emplace_back(std::move(arg_global));
    }
    if (trans_from_host) {
      // Breaking all trainable variables to groups, where each group's total
      // tensor size not exceeds the hard limit of 2GB of protobuf.
      rpc_client->VariablesGrouping(&var_literals);
    }
  }

  auto& global_vars = rpc_client->global_vars();

  for (auto& arg_global : global_vars) {
    args_global_transformed.emplace_back(arg_global.get());
  }

  VLOG(2) << "after transfer to literal";

  auto result = rpc_client->ExecutePlan(execution_handle,
                          args_global_transformed,
                          nullptr,
                          fetch_and_apply_resource_variables)
  .ConsumeValueOrDie();

  return std::move(result);
}

XlaMergeOp::XlaMergeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void XlaMergeOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaMergeOp " << def().name();
  int i = 0;
  if (ctx->has_input(i) || ctx->has_input(++i)) {
    ctx->set_output(0, ctx->input(i));
  }
}

REGISTER_KERNEL_BUILDER(Name("XlaLaunch").Device(DEVICE_CPU), XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("XlaLaunch")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("resources"),
                        XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("_XlaCompile").Device(DEVICE_CPU), XlaCompileOp);
REGISTER_KERNEL_BUILDER(Name("_XlaCompile")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("key")
                            .HostMemory("compilation_successful")
                            .HostMemory("resources"),
                        XlaCompileOp);

REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_CPU), XlaRunOp);
REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_GPU).HostMemory("key"),
                        XlaRunOp);

REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_CPU), XlaMergeOp);
REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_GPU), XlaMergeOp);

}  // namespace tensorflow
