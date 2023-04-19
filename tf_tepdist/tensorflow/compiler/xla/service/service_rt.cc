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

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/pjrt/virtual_client.h"
#include "tensorflow/compiler/xla/pjrt/nvidia_gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/virtual_device.h"
#include "tensorflow/compiler/xla/service/service_env.h"

namespace xla {

/*static*/
VirtualClient* Service::GetVirtualClient() {
  static VirtualClient* virtual_client =
      CreateVirtualClient(Service::TASK_INDEX);
  return virtual_client;
}

StatusOr<GlobalDataHandle> Service::RegisteredForVariable(
    const LiteralProto& literal_proto, int global_idx) {
  TF_ASSIGN_OR_RETURN(GlobalDataHandle data_handle,
                      allocation_tracker_.ProduceAndRegsiterDAPPLEBufferHandle());
  
  TF_ASSIGN_OR_RETURN(DAPPLEBufferHandle d_buf_handle,
                      allocation_tracker_.ResolveDAPPLEBufferHandle(data_handle));

  TF_ASSIGN_OR_RETURN(Literal literal,
                      Literal::CreateFromProto(literal_proto));

  auto* virtual_client = Service::GetVirtualClient();
  auto* whole_graph_launch_context = virtual_client->whole_graph_launch_context();
  auto& shape = literal.shape();
  CHECK(!shape.IsTuple());
  auto status = whole_graph_launch_context->RegisterVariable(
      d_buf_handle, std::move(literal));
  if (!status.ok()) {
    return InternalError(
        "Register DAPPLEBuffer error for handle %d", d_buf_handle);
  }
  auto* recv_args_recorder = whole_graph_launch_context->mutable_recv_args_recorder();
  recv_args_recorder->RecordVariableHandle(data_handle, d_buf_handle, global_idx);
  return data_handle;
}

StatusOr<GlobalDataHandle> Service::RegisteredForVariable(
    const ShapeProto& shape_proto, int global_idx) {
  TF_ASSIGN_OR_RETURN(GlobalDataHandle data_handle,
                      allocation_tracker_.ProduceAndRegsiterDAPPLEBufferHandle());
  
  TF_ASSIGN_OR_RETURN(DAPPLEBufferHandle d_buf_handle,
                      allocation_tracker_.ResolveDAPPLEBufferHandle(data_handle));

  auto* virtual_client = Service::GetVirtualClient();
  auto* whole_graph_launch_context = virtual_client->whole_graph_launch_context();
  auto shape = Shape(shape_proto);
  CHECK(!shape.IsTuple());
  auto status = whole_graph_launch_context->RegisterVariable(d_buf_handle, shape);
  if (!status.ok()) {
    return InternalError(
        "Register DAPPLEBuffer error for handle %d", d_buf_handle);
  }
  auto* recv_args_recorder = whole_graph_launch_context->mutable_recv_args_recorder();
  recv_args_recorder->RecordVariableHandle(data_handle, d_buf_handle, global_idx);
  return data_handle;
}

Status Service::DoRemoteRestore(const DoRemoteRestoreRequest* arg,
                                DoRemoteRestoreResponse* result) {
  ckpt_opts_.restore_from_ckpt = true;
  ckpt_opts_.global_step = arg->global_step();
  return Status::OK();
}

Status Service::DoRemoteSave(const DoRemoteSaveRequest* arg,
                             DoRemoteSaveResponse* result) {
  tensorflow::mutex_lock l(do_save_remote_mutex_);
  auto virtual_client = Service::GetVirtualClient();
  ckpt_opts_.global_step = arg->global_step();
  if (virtual_client->warm_up()) {
    // Lazy Save checkpoints
    virtual_client->set_ckpt_max_to_keep(arg->max_to_keep());
    ckpt_opts_.lazy_save_ckpt = true;
  } else {
    auto* execution_state = virtual_client->execution_state();
    CHECK(execution_state);
    execution_state->set_global_step(arg->global_step());
    execution_state->SaveCheckpoint();
  }
  // If request from client, we need to call DoRemoteSave on other workers.
  auto& coord = virtual_client->coordinator();
  if (arg->from_client() && (coord.num_workers() > 1)) {
    coord.DoRemoteSave(arg->max_to_keep(), arg->global_step());
  }
  return Status::OK();
}

void Service::ResetCheckpointOptions() {
  tensorflow::mutex_lock l(do_save_remote_mutex_);
  ckpt_opts_.lazy_save_ckpt = false;
  ckpt_opts_.restore_from_ckpt = false;
}

//StatusOr<std::vector<std::vector<const ShapedBuffer*>>>
StatusOr<std::vector<const ShapedBuffer*>>
Service::ResolveArguments(
    absl::Span<const GlobalDataHandle* const> arguments) const {
  std::vector<const ShapedBuffer*> result;
  //std::vector<std::vector<const ShapedBuffer*>> result;
  result.reserve(arguments.size());

  for (size_t i = 0; i < arguments.size(); ++i) {
    auto buffer_status = allocation_tracker_.Resolve(*arguments[i]);
    if (!buffer_status.ok()) {
      return Status(buffer_status.status().code(),
                    absl::StrCat(buffer_status.status().error_message(), ", ",
                           "failed to resolve allocation for parameter ", i));
    }
    auto buffers = buffer_status.ValueOrDie();  // buffers type: std::vector<const ShapedBuffer*>
    VLOG(2) << "in Service::ResolveArguments: buffers size: " << buffers.size();
    CHECK(buffers.size() == 1);
    result.emplace_back(buffers[0]);
    //result.emplace_back(std::move(buffers));
  }

  return std::move(result);
}

StatusOr<std::vector<DAPPLEBuffer*>>
Service::ResolveDAPPLEBuffers(
    absl::Span<const GlobalDataHandle* const> d_buf_args) {
  std::vector<DAPPLEBuffer*> result;
  result.reserve(d_buf_args.size());

  auto* virtual_client = Service::GetVirtualClient();
  auto* whole_graph_launch_context = virtual_client->whole_graph_launch_context();
  for (size_t i = 0; i < d_buf_args.size(); ++i) {
    auto d_buf_handle_or =
        allocation_tracker_.ResolveDAPPLEBufferHandle(*d_buf_args[i]);
    if (!d_buf_handle_or.ok()) {
      return Status(d_buf_handle_or.status().code(),
                    absl::StrCat(d_buf_handle_or.status().error_message(), ", ",
                           "failed to resolve allocation for DAPPLE parameter ", i));
    }
    auto d_buf_handle = d_buf_handle_or.ValueOrDie();
    auto d_buf_or = whole_graph_launch_context->Resolve(d_buf_handle);
    if (!d_buf_or.ok()) {
      return Status(d_buf_or.status().code(),
                    absl::StrCat(d_buf_or.status().error_message(), ", ",
                           "failed to resolve DAPPLEBuffer for handle ", i));
    }
    auto d_buf = d_buf_or.ValueOrDie();
    result.emplace_back(d_buf);
  }

  return std::move(result);
}

std::shared_ptr<LocalPlan>
Service::BuildDistPlan(
    const XlaComputation& computation,
    const absl::Span<const Shape* const>& argument_layouts,
    const ExecutableBuildOptions& build_options) {
  auto vclient = Service::GetVirtualClient();
  auto& coord = vclient->coordinator();

  auto client = vclient->gpu_client();
  auto status_or_result = client->client()->PreBuildDefModuleForTaskDAG(
       computation, argument_layouts, build_options);
  auto def_hlo_pairs = status_or_result.ConsumeValueOrDie();
  coord.TransferModuleAndDefCtx(def_hlo_pairs);

  auto num_nodes = coord.num_workers()/*include master*/;
  std::shared_ptr<DistributedPlan> dist_plan =
      vclient->BuildDistributedPlan(def_hlo_pairs, num_nodes, 0/*master_rank*/);

  auto& plan_def_hlo_pairs = dist_plan->def_hlo_pairs();
  auto exe_status_or_result =
      client->client()->BuildDefModules(std::move(plan_def_hlo_pairs),
                                              build_options);
  auto def_exe_pairs = exe_status_or_result.ConsumeValueOrDie();
  dist_plan->SetupDefExeMap(def_exe_pairs);
  dist_plan->ExeDecoration();

  // Run cost based scheduling
  dist_plan->ScheduleTasks();

  VLOG(2) << "Launch slave compilation";
  // Async. Slave Compilation
  std::thread dispatch_thread([&dist_plan, &coord]() {
    coord.DispatchPlan(dist_plan.get());
  });

  // build master's local plan
  auto local_plan = vclient->BuildLocalPlan(dist_plan);

  dispatch_thread.join();
 
  return local_plan;
}

Status Service::BuildExecutionPlan(const BuildExecutionPlanRequest* arg,
                                   BuildExecutionPlanResponse* result) {
  VLOG(0) << "running BuildExecutionPlan request";
  // LOG(INFO) << "[service side] request string for HLO module proto: " << std::endl << arg->DebugString() << std::endl;
  if (!arg->has_computation()) {
    return InvalidArgument("computations may not be empty");
  }
  if (!arg->computation().has_host_program_shape()) {
    return InvalidArgument("program shape may not be empty");
  }

  HloModuleConfig config(ProgramShape{arg->computation().host_program_shape()});

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProto(arg->computation(), config));

  auto virtual_client = Service::GetVirtualClient();
  auto& coord = virtual_client->coordinator();
  int64 num_worker = coord.num_workers()/*include master*/;
  module->set_num_worker(num_worker);
  module->set_num_dev_per_worker(coord.num_dev_per_worker());

  // solve backward compatibility issue
  FixModuleForCompat(*module.get());

  TF_RETURN_IF_ERROR(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false)
          .Run(module.get())
          .status());

  VLOG(1) << "[service side] end module after fix " << std::endl;

  const HloModuleProto new_hlo_module_proto = module->ToProto();
  XlaComputation new_xla_comp(new_hlo_module_proto);

  std::vector<const Shape*> argument_shape_ptrs;
  auto entry = module->entry_computation();
  for (auto param : entry->parameter_instructions()) {
    argument_shape_ptrs.emplace_back(&param->shape());
  }

  ExecutableBuildOptions updated_options;

  // TODO(lansong): it need be refined later
  updated_options.set_device_ordinal(0);

  if (coord.num_workers() > 1) {
    VLOG(0) << "before BuildDistPlan";
    auto plan = BuildDistPlan(new_xla_comp, argument_shape_ptrs,
                              updated_options);
    *result->mutable_handle() = plan_cache_.Insert(std::move(plan));
    return Status::OK();
  }

  auto gpu_client = virtual_client->gpu_client();
  VLOG(1) << "before PreBuildDefModuleForTaskDAG";
  auto def_hlo_pairs_or = gpu_client->client()->PreBuildDefModuleForTaskDAG(
       new_xla_comp, argument_shape_ptrs, updated_options);
  auto def_hlo_pairs = def_hlo_pairs_or.ConsumeValueOrDie();

  // only one machine, build single machine plan
  std::shared_ptr<DistributedPlan> dist_plan =
      virtual_client->BuildDistributedPlan(def_hlo_pairs, 1/*num_nodes*/, 0);

  auto& plan_def_hlo_pairs = dist_plan->def_hlo_pairs();
  auto status_or_result =
      gpu_client->client()->BuildDefModules(std::move(plan_def_hlo_pairs),
                                            updated_options);
  auto def_exe_pairs = status_or_result.ConsumeValueOrDie();
  dist_plan->SetupDefExeMap(def_exe_pairs);
  dist_plan->ExeDecoration();
  // Run cost based scheduling
  dist_plan->ScheduleTasks();

  // dist_plan will be owned by local plan of master
  // need not register dist_plan in plan_cache_?

  // device count is num_shards*num_dp_insts
  auto local_plan = virtual_client->BuildLocalPlan(dist_plan);

  local_plan->task_graph()->ResolveCGLossOutputs();

  local_plan->MakeTaskGraphGCPlan();
  if (ServiceEnv::debug())
    local_plan->ShowPerDeviceTaskList();

  *result->mutable_handle() = plan_cache_.Insert(std::move(local_plan));

  VLOG(1) << "successfully completed 'BuildExecutionPlan' request";
  return Status::OK();
}

Status Service::InitRemoteNcclComm(
    const InitRemoteNcclCommRequest* request,
    InitRemoteNcclCommResponse* response) {
  ncclUniqueId nccl_unique_id;
  std::memcpy((void *)&nccl_unique_id,
              (const void *)request->nccl_unique_id().data(), sizeof(ncclUniqueId));
  std::vector<int64> all_participants(
      request->all_participants().begin(), request->all_participants().end());
  auto vclient = Service::GetVirtualClient();
  gpu::NcclContext* nccl_ctx = vclient->GetOrCreateNcclContext();
  gpu::NcclUniqueGroupKey group_key(all_participants);
  if (!nccl_ctx->NcclUniqueIdAlreadyCreated(group_key)) {
    TF_RETURN_IF_ERROR(nccl_ctx->RegisterNcclUniqueId(group_key, nccl_unique_id));
  }
  TF_ASSIGN_OR_RETURN(auto plan,
                      plan_cache_.LookUp(request->handle()));

  DistributedPlan* dist_plan = plan->distributed_plan();
  TaskDAG* task_graph = dist_plan->task_graph();
  std::shared_ptr<CommDevManager> comm_dev_mgr = task_graph->comm_dev_mgr();
  TF_RETURN_IF_ERROR(nccl_ctx->MaybeCreateNcclComms(
      all_participants, vclient->host_id(), comm_dev_mgr));
  response->set_ret_status(1);
  return Status::OK();
}

Status Service::TransferHostRawData(
    const TransferHostRawDataRequest* request,
    TransferHostRawDataResponse* response) {
  TF_ASSIGN_OR_RETURN(auto plan,
                      plan_cache_.LookUp(request->handle()));
  CHECK(plan.get());

  auto& host_raw_data = request->host_raw_data();
  int sample_idx = host_raw_data.sample_idx();
  auto shape = Shape(host_raw_data.shape());
  auto vclient = Service::GetVirtualClient();
  auto* graph_launch_ctx = vclient->whole_graph_launch_context();
  TF_ASSIGN_OR_RETURN(auto* dapple_buf,
                      graph_launch_ctx->ResolveSampleInputBuffer(
                          sample_idx, shape, vclient->virtual_device()));
  dapple_buf->set_on_cpu();
  auto* dapple_raw = dapple_buf->mutable_raw();
  int size = ShapeUtil::ByteSizeOf(shape);
  std::memcpy(dapple_raw, host_raw_data.raw().data(), size);
  auto* recv_args = graph_launch_ctx->mutable_recv_args_recorder();
  recv_args->RecordSampleIndex(sample_idx);
  response->set_ret_status(1);
  return Status::OK();
}

Status Service::TransferVarArgMap(
    const TransferVarArgMapRequest* request,
    TransferVarArgMapResponse* response) {
  auto vclient = Service::GetVirtualClient();
  auto* whole_graph_launch_context = vclient->whole_graph_launch_context();
  whole_graph_launch_context->set_num_vars(request->num_vars());
  //*vclient->mutable_var_gtol_idx_map() = request->var_gtol_idx_map();
  auto& dst_var_gtol_idx_map = vclient->var_gtol_idx_map();
  auto& src_var_gtol_idx_map = request->var_gtol_idx_map();
  for (auto& it : src_var_gtol_idx_map) {
    dst_var_gtol_idx_map[it.first] = it.second;
  }

  //*vclient->mutable_var_local_idx_map() = request->var_local_idx_map();
  auto& dst_var_local_idx_map = vclient->var_local_idx_map();
  auto& src_var_local_idx_map = request->var_local_idx_map();
  for (auto& it : src_var_local_idx_map) {
    dst_var_local_idx_map[it.first] = it.second;
  }

  //*vclient->mutable_input_local_idx_map() = request->input_local_idx_map();
  auto& dst_input_local_idx_map = vclient->input_local_idx_map();
  auto& src_input_local_idx_map = request->input_local_idx_map();
  for (auto& it : src_input_local_idx_map) {
    dst_input_local_idx_map[it.first] = it.second;
  }

  response->set_ret_status(1);
  return Status::OK();
}

Status Service::ExecuteRemotePlan(
    const ExecuteRemotePlanRequest* request,
    ExecuteRemotePlanResponse* response) {
  TF_ASSIGN_OR_RETURN(auto plan,
                      plan_cache_.LookUp(request->handle()));

  ExecuteOptions execute_options;
  execute_options.untuple_result = true;
  execute_options.global_step = request->global_step();
  execute_options.lazy_save_ckpt = request->lazy_save_ckpt();
  execute_options.restore_from_ckpt = request->restore_from_ckpt();
  auto virtual_client = Service::GetVirtualClient();
  auto py_executable = absl::make_unique<DAPPLEExecutable>(
                          /*options.parameter_is_tupled_arguments*/ false,
                          virtual_client, plan);
  // Since all arguments are transferred from master graph by SplitSend and
  // can be resolved by SplitRecv during running local graph, it is
  // unnecessary to pass arguments to ExecuteRemotePlan.
  plan->set_dapple(py_executable.get());
  TF_ASSIGN_OR_RETURN(std::vector<DAPPLEBuffer*> plan_outputs,
                      py_executable->ExecuteRemotePlan(execute_options));
  response->set_ret_status(1);
  return Status::OK();
}

Status Service::DispatchPlan(
    const DispatchPlanRequest* request,
    DispatchPlanResponse* response) {
  // Parse data 
  auto num_workers = request->num_workers();
  auto worker_rank = request->worker_rank();

  auto num_tasks = request->compute_task_size();
  std::vector<ComputeTask> compute_tasks;
  compute_tasks.reserve(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    compute_tasks.emplace_back(request->compute_task(i));
  }

  std::vector<int> split_nums;
  for (int i=0; i<request->split_nums_size(); ++i) {
    split_nums.emplace_back(request->split_nums(i));
  }

  std::vector<bool> share_dev_flags;
  for (int i=0; i<request->share_dev_flags_size(); ++i) {
    share_dev_flags.emplace_back(request->share_dev_flags(i));
  }

  auto stage_split_ordinal = request->stage_split_ordinal();
  std::vector<int> placement_layout;
  for (int i = 0; i < request->placement_layout_size(); ++i) {
    placement_layout.emplace_back(request->placement_layout(i));
  }

  VLOG(0) << "DispatchPlan: split_nums " << request->split_nums_size() << " " << split_nums.size();
  // Reconstruct the plan
  auto vclient = Service::GetVirtualClient();
  auto dist_plan = vclient->BuildDistributedPlanRPC(
      num_workers, worker_rank, compute_tasks, split_nums,
      share_dev_flags, placement_layout, stage_split_ordinal);

  //auto gpu_client = vclient->gpu_client();
  //ExecutableBuildOptions build_options;
  //if (!build_options.device_allocator()) {
  //  build_options.set_device_allocator(gpu_client->allocator());
  //}
  //auto rank = dist_plan->worker_rank();
  //vclient->BuildModules(dist_plan.get(), rank, build_options);
  auto plan = vclient->BuildLocalPlan(dist_plan);

  *response->mutable_handle() = plan_cache_.Insert(std::move(plan));
  return Status::OK();
}

Status Service::TransferModuleAndDefCtx(
    const TransferModuleAndDefCtxRequest* request,
    TransferModuleAndDefCtxResponse* response) {
  auto virtual_client = Service::GetVirtualClient();
  auto& def_hlo_pairs = virtual_client->rpc_def_hlo_map();

  int num_module_ctx = request->module_ctx_size();
  def_hlo_pairs.reserve(num_module_ctx);

  auto debug_options = request->debug_options();

  HloModule* entry_module = nullptr;
  std::map<int/*def_id*/, HloModule::DefContext*> id_def_map;
  std::map<int/*def_id*/, std::vector<int>> id_children_map;
  for (int i = 0; i < num_module_ctx; ++i) {
    auto& module_ctx = request->module_ctx(i);    
    auto& comp = module_ctx.computation();
    HloModuleConfig config(ProgramShape{comp.host_program_shape()});
    config.set_debug_options(debug_options);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        HloModule::CreateFromProto(comp, config));

    auto& module_def_ctx = module_ctx.def_ctx();
    auto name = module_def_ctx.name();
    if (module_def_ctx.def_type() == HloModule::DefContext::DefType::ENTRY) {
      CHECK(!entry_module);
      CHECK(name == "Entry");
      entry_module = module.get();
    } else {
      CHECK(entry_module);
    }

    auto def_ctx = entry_module->create_def_ctx_from_proto(module_def_ctx);
    // children
    auto& children = module_def_ctx.children();
    auto& xs = id_children_map[def_ctx->def_id()];
    xs.assign(children.begin(), children.end());

    def_ctx->module_ = module.get();
    module->set_def_ctx(def_ctx);

    id_def_map[def_ctx->def_id()] = def_ctx;
    def_hlo_pairs.emplace_back(std::make_pair(def_ctx, std::move(module)));
  }

  // Setup parent-child relationship
  for (auto& it : id_children_map) {
    auto def_id = it.first;
    auto& children_ids = it.second;
    auto def_ctx = id_def_map.at(def_id);
    def_ctx->children_.reserve(children_ids.size());
    for (auto child_id : children_ids) {
      auto child_def_ctx = id_def_map.at(child_id);
      def_ctx->children_.emplace_back(child_def_ctx);
    }
  }

  std::reverse(def_hlo_pairs.begin(), def_hlo_pairs.end());
  response->set_ret_status(1);

  return Status::OK();
}

Status Service::ExecutePlan(const ExecutePlanRequest* arg,
                                  ExecutePlanResponse* result) {
  uint64 start_us = tensorflow::Env::Default()->NowMicros();
  VLOG(1) << "running execute plan request";
  if (!arg->has_handle()) {
    return InvalidArgument("execution plan handle should not be empty");
  }

  TF_ASSIGN_OR_RETURN(auto exec_plan,
                      plan_cache_.LookUp(arg->handle()));

  auto* virtual_client = Service::GetVirtualClient();
  auto* whole_graph_launch_context = virtual_client->whole_graph_launch_context();

  std::vector<const GlobalDataHandle*> input_arg_handles;
  std::vector<const GlobalDataHandle*> variable_handles;

  int vars_count = 0;
  for (auto& data_handle : arg->arguments()) {
    if (data_handle.variable()) {
      variable_handles.emplace_back(&data_handle);
      ++vars_count;
    } else {
      input_arg_handles.emplace_back(&data_handle);
    }
  }

  //TF_ASSIGN_OR_RETURN(auto arguments, ResolveArguments(arg->arguments()));
  TF_ASSIGN_OR_RETURN(auto input_args, ResolveArguments(input_arg_handles));
  TF_ASSIGN_OR_RETURN(auto variables, ResolveDAPPLEBuffers(variable_handles));

  // construct argument_handles
  auto py_executable = absl::make_unique<DAPPLEExecutable>(
                          /*options.parameter_is_tupled_arguments*/ false,
                          virtual_client, exec_plan);

  std::vector<DAPPLEBuffer*> argument_handle_ptrs;
  //std::vector<std::unique_ptr<DAPPLEBuffer>> input_argument_handles;
  std::map<int, bool> variable_arg;
  int sample_input_idx = 0;
  for (const ShapedBuffer* shaped_buffer : input_args) {
    auto root = shaped_buffer->root_buffer();
    auto shape = shaped_buffer->on_host_shape();
    CHECK(!shape.IsTuple());
    auto data = root.opaque();
    auto size = root.size();

    auto buf_status =
        whole_graph_launch_context->ResolveSampleInputBuffer(
            sample_input_idx, shape, virtual_client->virtual_device());
    CHECK(buf_status.ok());
    DAPPLEBuffer* buf = buf_status.ConsumeValueOrDie();
    buf->set_on_cpu();
    auto dapple_raw = buf->mutable_raw();
    CHECK(buf->on_host_shape()==shaped_buffer->on_host_shape());
    CHECK(buf->on_device_shape()==shaped_buffer->on_device_shape());
    CHECK(ShapeUtil::ByteSizeOf(buf->on_host_shape())==size);
    std::memcpy(dapple_raw, data, size);

    variable_arg[argument_handle_ptrs.size()] = false;
    argument_handle_ptrs.emplace_back(buf);
    //input_argument_handles.emplace_back(std::move(buf));
    ++sample_input_idx;
  }

  if (py_executable->fake_input()) {
    if (py_executable->mutable_input_data_args()->empty()) {
      for (auto* dapple_buf : argument_handle_ptrs) {
        py_executable->mutable_input_data_args()->emplace_back(dapple_buf);
      }
    } else {
      argument_handle_ptrs.clear();
      for (auto* dapple_buf : *(py_executable->mutable_input_data_args())) {
        argument_handle_ptrs.emplace_back(dapple_buf);
      }
    }
  }

  for (auto* dapple_buffer : variables) {
    variable_arg[argument_handle_ptrs.size()] = true;
    argument_handle_ptrs.emplace_back(dapple_buffer);
  }

  virtual_client->set_var_arg_map(std::move(variable_arg));

  ExecuteOptions execute_options;
  execute_options.untuple_result = true;

  {
    tensorflow::mutex_lock l(do_save_remote_mutex_);
    execute_options.global_step = ckpt_opts_.global_step;
    execute_options.lazy_save_ckpt = ckpt_opts_.lazy_save_ckpt;
    execute_options.restore_from_ckpt = ckpt_opts_.restore_from_ckpt;
  }

  std::vector<DAPPLEBuffer*> dapple_buf;
  {
    tensorflow::mutex_lock l(execute_plan_mutex_);
    if (py_executable->fake_input()) {
      start_us = tensorflow::Env::Default()->NowMicros();
    }
    TF_ASSIGN_OR_RETURN(dapple_buf,
        py_executable->ExecuteRPCPlan(argument_handle_ptrs, execute_options));

    std::vector<DAPPLEBuffer*> outputs;
    for (int64 i = 0; i < dapple_buf.size() - vars_count; ++i) {
      outputs.emplace_back(dapple_buf[i]);
    }

    TF_ASSIGN_OR_RETURN(auto new_variables,
                        ResolveDAPPLEBuffers(variable_handles)); 
    auto task_graph = exec_plan->task_graph();
    TaskNode* split_task = task_graph->source();
    auto& module = *(exec_plan->task_module(split_task));
    auto& fetch_vars_list = module.fetch_vars_list();
    std::unordered_set<int> fetch_vars_set(
        fetch_vars_list.begin(), fetch_vars_list.end());

    int num_args = module.entry_computation()->num_parameters();
    int num_feed_data = num_args - new_variables.size();
    for (int64 i = 0; i < new_variables.size(); ++i) {
      if (!fetch_vars_set.count(i + num_feed_data)) continue;
      VLOG(2) << "[DEBUG] fetching resource variable " << i;
      outputs.emplace_back(new_variables[i]);
    }

    auto literal_or = py_executable->ConvertOutputToLiteral(
        outputs, 0, outputs.size());
    uint64 end_us = tensorflow::Env::Default()->NowMicros();
    float duration_ms = (end_us - start_us) / 1000.0f;
    VLOG(0) << "[ExecutePlan Duration] finish in " << duration_ms << " ms";
    whole_graph_launch_context->CleanupVariablesOnHost();
    CHECK(literal_or.ok());
    auto literal = literal_or.ConsumeValueOrDie();
    *result->mutable_literal() = literal->ToProto();
  }

  ResetCheckpointOptions();

  VLOG(1) << "successfully completed 'ExecutePlan' request";
  return Status::OK();
}

Status Service::FetchResourceVars(const FetchResourceVarsRequest* arg,
                                  FetchResourceVarsResponse* result) {
  VLOG(1) << "running FetchResourceVars request";
  VLOG(1) << "Fetching #" << arg->arguments().size() << " vars.";
  std::vector<const GlobalDataHandle*> variable_handles;
  for (auto& data_handle : arg->arguments()) {
    variable_handles.emplace_back(&data_handle);
  }

  TF_ASSIGN_OR_RETURN(auto new_variables,
                      ResolveDAPPLEBuffers(variable_handles));
  std::vector<DAPPLEBuffer*> outputs;
  for (int64 i = 0; i < new_variables.size(); ++i) {
    outputs.emplace_back(new_variables[i]);
  }
  auto literal_or = DAPPLEExecutable::ConvertOutputToLiteral(
      outputs, 0, outputs.size());
  CHECK(literal_or.ok());
  auto literal = literal_or.ConsumeValueOrDie();
  *result->mutable_literal() = literal->ToProto();
  VLOG(1) << "successfully completed 'FetchResourceVars' request";
  return Status::OK();
}

// move TransferToServerHost from service.cc to service_rt.cc
Status Service::TransferToServerHost(const TransferToServerRequest* arg,
                                 TransferToServerResponse* result) {
  if (arg->variable()) {
    bool trans_from_host = arg->trans_from_host();
    TF_ASSIGN_OR_RETURN(auto data_handle,
                        trans_from_host ? RegisteredForVariable(arg->literal(), arg->global_idx())
                            : RegisteredForVariable(arg->shape_with_layout(), arg->global_idx()));
    *result->mutable_data() = data_handle;
    return Status::OK();
  }
  TF_ASSIGN_OR_RETURN(Literal literal,
                      Literal::CreateFromProto(arg->literal()));
  const Shape& shape = literal.shape();

  TF_ASSIGN_OR_RETURN(auto executor, execute_backend_->stream_executor(0));

  std::vector<ScopedShapedBuffer> replicated_buffers;
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer shaped_buffer,
      execute_backend_->transfer_manager()->AllocateScopedShapedBuffer(
          shape, execute_backend_->memory_allocator(),
          executor->device_ordinal()));
  TF_ASSIGN_OR_RETURN(auto stream, execute_backend_->BorrowStream(executor));
  TF_RETURN_IF_ERROR(
      execute_backend_->transfer_manager()->TransferLiteralToDevice(
          stream.get(), literal, shaped_buffer));
  replicated_buffers.emplace_back(std::move(shaped_buffer));
  TF_ASSIGN_OR_RETURN(*result->mutable_data(),
                      allocation_tracker_.RegisterReplicatedBuffers(
                          std::move(replicated_buffers),
                          absl::StrCat("TransferToServer literal of shape ",
                                 ShapeUtil::HumanString(shape))));

  return Status::OK();
}

}  // namespace xla


