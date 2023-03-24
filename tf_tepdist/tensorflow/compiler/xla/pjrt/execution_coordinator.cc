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
#include <fstream>
#include <string>
#include <regex>
#include "execution_coordinator.h"
#include "tensorflow/compiler/xla/service/service_env.h"

namespace xla {

void ExecutionCoordinator::extract_worker_info(const Json::Value& worker,
                                               int& gpu_num_per_worker) {
  // get ip
  Json::Value ip_json = worker.get("ip", Json::Value::null);
  if (ip_json.isNull() ||
      (!ip_json.isString())) {
    LOG(FATAL) << "ip is missed or invalid in cluster spec!\n";
    return;
  }
  string ip_str = ip_json.asString();

  // get port
  Json::Value port_json = worker.get("port", Json::Value::null);
  if (port_json.isNull()) {
    LOG(FATAL) << "Miss port in cluster spec!\n";
    return;
  }

  std::regex reg("^[0-9]{1,10}$");
  int port;
  if (port_json.isString()) {
    string port_str = port_json.asString();
    if (!std::regex_match(port_str, reg)) {
      LOG(FATAL) << "Invalid port format in cluster spec!\n";
      return;
    }
    port = std::stoi(port_str);
  } else if (port_json.isInt()) {
    port = port_json.asInt();
  } else {
    LOG(FATAL) << "Invalid port format in cluster spec!\n";
    return;
  }

  // record ip, port for workers
  workers_.push_back(std::make_pair(ip_str, port));

  // get gpu ids
  Json::Value gpu_ids_json = worker.get("gpu_ids", Json::Value::null);
  if (gpu_ids_json.isNull()) {
    LOG(FATAL) << "Miss gpu ids!\n";
  }

  if (!gpu_ids_json.isString()) {
    LOG(FATAL) << "gpu ids should be a string!\n";
  }
  string gpu_ids = gpu_ids_json.asString();
  std::vector<std::string> gpu_list =
      tensorflow::str_util::Split(gpu_ids, ',');
  if (gpu_num_per_worker>0 && gpu_num_per_worker != gpu_list.size()) {
    LOG(FATAL) << "gpu number should be equal!";
  } else {
    gpu_num_per_worker = gpu_list.size();
  }

  for (auto& gpu : gpu_list) {
    if (!std::regex_match(gpu, reg)) {
      LOG(FATAL) << "Invalid gpu id in cluster spec file!\n";
      return;
    }
    int gpu_id = std::stoi(gpu);
    gpu_ids_.emplace_back(gpu_id);
  }
}

void ExecutionCoordinator::Init() {
  std::string cluster_spec_file = ServiceEnv::cluster_spec();
  if (cluster_spec_file.empty()) {
    LOG(FATAL) << "Invalid cluster spec specified!\n";
    return;
  }

  Json::Value json;
  Json::Reader reader;
  std::ifstream cluster_spec_fstream(cluster_spec_file);
  if (cluster_spec_fstream.fail()) {
    LOG(FATAL) << "cluster spec file does not exist!\n";
    return;
  }

  if (!reader.parse(cluster_spec_fstream, json)) {
    LOG(FATAL) << "Invalid cluster spec specified!\n";
    return;
  }

  VLOG(0) << "json object: " << json;
  if (!json.isObject()) {
    LOG(FATAL) << "Invalid cluster spec specified!\n";
    return;
  }

  int gpu_num_per_worker = 0;
  // master
  Json::Value master = json["master"];
  if (master.isNull()) {
    LOG(FATAL) << "Miss master in cluster spec file!\n";
    return;
  }

  if (!master.isObject()) {
    LOG(FATAL) << "Invalid master spec in cluster spec file!\n";
    return;
  }

  extract_worker_info(master, gpu_num_per_worker);

  // workers
  Json::Value workers_json = json["workers"];
  if (workers_json.isNull()) {
    LOG(INFO) << "run on single node";
    return;
  }

  if (!workers_json.isArray()) {
    LOG(FATAL) << "Invalid Worker List in cluster spec!\n";
    return;
  }

  for (size_t i = 0; i < workers_json.size(); i++) {
    const Json::Value worker = workers_json.get(i, Json::Value::null);
    if (!worker.isObject()) {
      LOG(FATAL) << "Invalid Worker spec in cluster spec!\n";
      return;
    }

    extract_worker_info(worker, gpu_num_per_worker);
  }

  service_stubs_.reserve(workers_.size() - 1);
  stubs_.reserve(workers_.size() - 1);

  for (int i = 1; i < workers_.size(); ++i) { // ignore the first because it is master
    auto ip = workers_[i].first;
    auto port = workers_[i].second;
    auto channel = ::grpc::CreateChannel(
                     absl::StrFormat("%s:%d", ip, port),
                     ::grpc::InsecureChannelCredentials());
    CHECK(channel->WaitForConnected(gpr_time_add(
        gpr_now(GPR_CLOCK_REALTIME), gpr_time_from_seconds(10, GPR_TIMESPAN))));
    VLOG(0) << "Channel to server->" << ip
            << " is connected on port->" << port;

    service_stubs_.emplace_back(std::move(grpc::XlaService::NewStub(channel)));
    stubs_.emplace_back(std::make_unique<GRPCStub>(
                        service_stubs_.back().get()));
  }
}

void ExecutionCoordinator::ExecuteRemotePlan(
    std::vector<std::thread>& dist_workers, const ExecuteOptions& options) {
  int num_workers = workers_.size() - 1;   // first is master
  for (int i = 0; i < num_workers; ++i) {
    dist_workers.push_back(std::thread([this, i, options]() {
      auto& stub = stubs_[i];
      ExecuteRemotePlanRequest request;
      *request.mutable_handle() = handle_map_[i];
      request.set_global_step(options.global_step);
      request.set_lazy_save_ckpt(options.lazy_save_ckpt);
      request.set_restore_from_ckpt(options.restore_from_ckpt);
      ExecuteRemotePlanResponse response;
      CHECK(stub->ExecuteRemotePlan(&request, &response).ok());
    }));
  }
}

void ExecutionCoordinator::DoRemoteSave(int64 max_to_keep, int64 global_step) {
  for (int i = 0; i < num_workers() - 1; ++i) {
    auto& stub = stubs_[i];
    DoRemoteSaveRequest request;
    request.set_max_to_keep(max_to_keep);
    request.set_global_step(global_step);
    request.set_from_client(false);
    DoRemoteSaveResponse response;
    CHECK(stub->DoRemoteSave(&request, &response).ok()); 
  }
}

void ExecutionCoordinator::TransferVariableArgMap(
    std::map<int, bool>& var_arg_map_in, int num_vars,
    std::map<int, int>& port_map) {
  for (int i = 0; i < num_workers() - 1; ++i) {
    auto& stub = stubs_[i];

    TransferVarArgMapRequest request;
    request.set_num_vars(num_vars);

    auto& var_gtol_idx_map = *request.mutable_var_gtol_idx_map();
    auto& var_local_idx_map = *request.mutable_var_local_idx_map();
    auto& input_local_idx_map = *request.mutable_input_local_idx_map();

    CHECK(port_map.size() <= var_arg_map_in.size());
    int local_input_idx = 0, local_var_idx = 0;
    int buf_index = 0, global_var_idx = 0;
    for (auto it : var_arg_map_in) {
      auto arg_idx = it.first;
      auto is_var = it.second;
      if (port_map.count(arg_idx)) {
        if (is_var) {
          var_gtol_idx_map[global_var_idx] = local_var_idx;
          var_local_idx_map[local_var_idx++] = buf_index;
        } else {
          input_local_idx_map[local_input_idx++] = buf_index;
        }
      }

      global_var_idx += is_var;
      buf_index += port_map.count(arg_idx);
    }
    CHECK(global_var_idx == num_vars);

    TransferVarArgMapResponse response;
    CHECK(stub->TransferVarArgMap(&request, &response).ok());
  }
}

void ExecutionCoordinator::TransferHostRawData(
      DAPPLEBuffer* dapple_buf, int worker_id, int sample_idx) {
  auto& stub = stubs_[worker_id];
  TransferHostRawDataRequest request;
  *request.mutable_handle() = handle_map_[worker_id];

  auto* raw = dapple_buf->raw();
  auto& shape = dapple_buf->on_host_shape();
  int64 shape_bytes = ShapeUtil::ByteSizeOf(shape);

  HostRawData host_raw_data;
  host_raw_data.set_raw(raw, shape_bytes);
  host_raw_data.set_sample_idx(sample_idx);
  *host_raw_data.mutable_shape() = shape.ToProto();
  *request.mutable_host_raw_data() = std::move(host_raw_data);

  VLOG(1) << "TransferHostRawData sample idx : " << sample_idx
          << ", shape = " << ShapeUtil::HumanString(shape);
  TransferHostRawDataResponse response;
  CHECK(stub->TransferHostRawData(&request, &response).ok());
}

void ExecutionCoordinator::TransferToServerHost(
    DAPPLEBuffer* dapple_buf, bool variable, int worker_id, int global_idx) {
  auto& stub = stubs_[worker_id];
  TransferToServerRequest request;
  request.set_variable(variable);
  request.set_global_idx(global_idx);
  if (dapple_buf->raw() != nullptr) {
    auto literal = dapple_buf->ToLiteral().ConsumeValueOrDie();
    *request.mutable_literal() = literal->ToProto();
    request.set_trans_from_host(true);
  } else {
    *request.mutable_shape_with_layout() = dapple_buf->on_host_shape().ToProto();
    request.set_trans_from_host(false);
  }
  TransferToServerResponse response;
  CHECK(stub->TransferToServerHost(&request, &response).ok());
}

void ExecutionCoordinator::TransferVarsAndData(
  std::vector<DAPPLEBuffer*> dapple_bufs, std::vector<bool>& variables,
  std::vector<int>& global_indices) {
  CHECK(dapple_bufs.size() == variables.size());
  for (int i = 0; i < num_workers() - 1; ++i) {
    for (int arg_no = 0; arg_no < dapple_bufs.size(); ++arg_no) {
      auto* dapple_buf = dapple_bufs[arg_no];
      if (!dapple_buf) continue;
      int idx = global_indices[arg_no];
      if (variables[arg_no]) {
        TransferToServerHost(dapple_buf, true/*variable*/, i, idx);
      } else {
        TransferHostRawData(dapple_buf, i, idx);
      }
    }
  }
}

void ExecutionCoordinator::InitRemoteNcclComm(
    ncclUniqueId& nccl_id, const absl::Span<const int64> all_participants,
    int worker_id) {
  VLOG(2) << "InitRemoteNcclComm for worker " << worker_id;
  auto& stub = stubs_[worker_id - 1];
  InitRemoteNcclCommRequest request;
  *request.mutable_handle() = handle_map_[worker_id - 1];
  request.set_nccl_unique_id((char*)&nccl_id, sizeof(ncclUniqueId));
  for (int64 gid : all_participants) {
    request.add_all_participants(gid);
  }
  InitRemoteNcclCommResponse response;
  Status status = stub->InitRemoteNcclComm(&request, &response);
  CHECK(status.ok());
}

// only a few information of task graph is dispatched
void ExecutionCoordinator::DispatchPlan(DistributedPlan* plan) {
  auto task_graph = plan->task_graph();

  // Transfer the master distributed execution plan to each worker
  int num_slave_workers = workers_.size() - 1;  // first is master
  std::unique_ptr<std::thread> threads[num_slave_workers];
  for (int i = 1; i <= num_slave_workers; ++i) {
    threads[i-1].reset(new std::thread([this, i, num_slave_workers, &task_graph]() {
      auto& stub = stubs_[i-1];
      DispatchPlanRequest request;
      request.set_num_workers(num_slave_workers+1/*master*/);
      request.set_worker_rank(i);
      for (auto& task_ref : task_graph->task_nodes()) {
        auto task = task_ref.get();
        ComputeTask compute_task;
        compute_task.set_name(task->name());
        compute_task.set_task_type(int(task->task_type()));
        compute_task.set_node_id(task->node_id());
        for (int one_dim_id : task->split_id().ids_) {
          compute_task.add_split_id(one_dim_id);
        }
        compute_task.set_worker_id(task->worker_id());
        compute_task.set_def_id(task->def_ctx()->def_id());
        compute_task.set_sched_idx_in_dev(task->sched_idx_in_dev());
        compute_task.set_device_id(task->device_id());
        compute_task.set_comm_with_lower_stage(task->comm_with_lower_stage());
        compute_task.set_across_machine(task->across_machine());

        int port_idx = 0;
        auto* mutable_port_map = compute_task.mutable_port_map();
        for (auto& it : task->port_map()) {
          CHECK(it.second == port_idx++);
          (*mutable_port_map)[it.first] = it.second;
        }

        for (int64 g_dev : task->send_recv_global_devs()) {
          compute_task.add_send_recv_global_devs(g_dev);
        }

        for (auto parent : task->parents()) {
          compute_task.add_parents(parent->node_id());
        }
        for (auto child : task->children()) {
          compute_task.add_children(child->node_id());
        }

        *request.add_compute_task() = compute_task;
      }

      for (int split_num : task_graph->split_nums()) {
        request.add_split_nums(split_num);
      }

      for (bool share_dev_flag : task_graph->share_dev_flags()) {
        request.add_share_dev_flags(share_dev_flag);
      }
      request.set_stage_split_ordinal(task_graph->stage_split_ordinal());

      for (int layout : task_graph->placement_layout()) {
        request.add_placement_layout(layout);
      }
      VLOG(2) << "dispatch plan to worker " << i;
      DispatchPlanResponse response;
      CHECK(stub->DispatchPlan(&request, &response).ok());
      auto exe_handle = response.handle();
      VLOG(2) << "ExecutionHandle from worker:" << i 
              << " Handle->" << exe_handle.handle();
      handle_map_[i-1] = std::move(exe_handle);
    }));
  }
  for (int i = 0; i < num_slave_workers; ++i) {
    threads[i]->join();
  }
}

void ExecutionCoordinator::TransferModuleAndDefCtx(
                           std::vector<std::pair<HloModule::DefContext*,
                           std::unique_ptr<HloModule>>>& def_hlo_pairs) {
  DebugOptions entry_debug_options;
  for (auto& it : def_hlo_pairs) {
    auto def = it.first;
    if (def->entry_def_ctx()) {
      entry_debug_options = it.second->config().debug_options();
      break;
    }
  }

  int num_workers = workers_.size() - 1;  // first is master
  for (int i = 0; i < num_workers; ++i) {
    auto& stub = stubs_[i];

    TransferModuleAndDefCtxRequest request;
    for (auto rit = def_hlo_pairs.rbegin();
              rit != def_hlo_pairs.rend(); ++rit) {
      auto def_ctx = rit->first;
      // Prepare transfer of DefContext
      ModuleDefContext module_def_ctx;
      module_def_ctx.set_def_id(def_ctx->def_id());
      module_def_ctx.set_def_type((int)def_ctx->def_type());
      module_def_ctx.set_parent_id(def_ctx->parent_id());
      module_def_ctx.set_name(def_ctx->name());
      
      auto& input_arg_map = *module_def_ctx.mutable_input_arg_map();
      for (auto& it : def_ctx->input_arg_map_) {
        input_arg_map[it.first] = it.second;
      }

      auto* input_def_map = module_def_ctx.mutable_input_def_map();
      for (auto& arg_iter : def_ctx->input_def_map_) {
        SrcOutputMapProto src_output_map_proto;
        for (auto& slice_iter : arg_iter.second) {
          SrcOutputProto src_output_proto;
          HloModule::DefContext::SrcOutput& src_output = slice_iter.second;
          src_output_proto.set_prev_slice_id(src_output.prev_slice_id);
          src_output_proto.set_def_id(src_output.def_id);
          src_output_proto.set_output_idx(src_output.output_idx);
          auto* src_output_map = src_output_map_proto.mutable_src_output_map();
          (*src_output_map)[slice_iter.first] = src_output_proto;
        }
        (*input_def_map)[arg_iter.first] = src_output_map_proto;
      }

      // input_dim_to_slice
      auto& input_dim_to_slice = *module_def_ctx.mutable_input_dim_to_slice();
      for (auto& it : def_ctx->input_dim_to_slice_) {
        input_dim_to_slice[it.first] = it.second;
      }

      // sharded args
      for (auto arg : def_ctx->sharded_args_) {
        module_def_ctx.add_sharded_args(arg);
      }
     
      // output_idx_map
      auto& output_idx_map = *module_def_ctx.mutable_output_idx_map();
      for (auto& it : def_ctx->output_idx_map_) {
        output_idx_map[it.first] = it.second;
      }

      // output_dim_to_slice
      auto& output_dim_to_slice = *module_def_ctx.mutable_output_dim_to_slice();
      for (auto& it : def_ctx->output_dim_to_slice_) {
        output_dim_to_slice[it.first] = it.second;
      }

      auto& output_idx_global_dev_map = *module_def_ctx.mutable_output_idx_global_dev_map();
      for (auto& it : def_ctx->output_idx_global_dev_map_) {
        GlobalSlices slices;
        for (int id : it.second) {
          slices.add_slices(id);
        }
        output_idx_global_dev_map[it.first] = slices;
      }

      // input_output_alias_map
      auto& input_output_alias_map = 
          *module_def_ctx.mutable_input_output_alias_map();
      for (auto& it : def_ctx->input_output_alias_map_) {
        input_output_alias_map[it.first] = it.second;
      }

      // children
      for (auto child : def_ctx->children_) {
        // For pure EntryDefContext without execution of seperation of
        // compute_gradient and apply_gradient, its 'children_' field is not
        // initialized with specific CG/AG/GA/GAInit/AR child DefContext.
        if (child) {
          module_def_ctx.add_children(child->def_id());
        }
      }

      // Prepare Transfer of HloModule
      auto module = rit->second.get();
      VLOG(2) << "Transferring DefCtx->" << def_ctx->name() 
              << " Module->" << module->name();

      const HloModuleProto hlo_module_proto = module->ToProto();
      XlaComputation xla_comp(hlo_module_proto);
      ModuleAndDefCtx module_ctx;
      // TODO: Optimize to alleviate memory copies
      *module_ctx.mutable_computation() = xla_comp.proto();
      *module_ctx.mutable_def_ctx() = std::move(module_def_ctx);
      *request.add_module_ctx() = std::move(module_ctx);
    } // for DefCtx
    *request.mutable_debug_options() = entry_debug_options;

    TransferModuleAndDefCtxResponse response;
    CHECK(stub->TransferModuleAndDefCtx(&request, &response).ok());
  } // for (worker)
}

} // namespace xla
