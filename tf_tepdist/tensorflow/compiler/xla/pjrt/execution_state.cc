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

#include "tensorflow/compiler/xla/pjrt/execution_state.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/pjrt/initializers.h"
#include "tensorflow/compiler/xla/pjrt/dapple_buffer_utils.h"
#include "tensorflow/compiler/xla/service/service_env.h"

namespace xla {
namespace {

void DistInitializationWrapper(
    const DistributionConfig& proto, const Shape& full_shape, const Shape& shard_shape,
    const std::vector<std::pair<int64, int64>>& start_offset_pairs, char* out_tensor) {
  switch (proto.dtype()) {
    case DT_FLOAT: {
      dist_rng::DistributedRandomInitializer::Initialize(
          proto, full_shape, shard_shape, start_offset_pairs, (float *)out_tensor);
      break;
    }

    case DT_INT64: {
      dist_rng::DistributedRandomInitializer::Initialize(
          proto, full_shape, shard_shape, start_offset_pairs, (int64 *)out_tensor);
      break;
    }

    default: CHECK(0 && "Unhandled DataType");
  }
}
} // namespace

ExecutionState::ExecutionState(
    TaskDAG* task_graph,  const std::map<int, std::string>* variable_map,
    bool sharding_across_machine, int local_device_count, int worker_rank,
    int worker_count, const int64 ckpt_max_to_keep)
  : task_graph_(task_graph), variable_map_(variable_map),
    sharding_across_machine_(sharding_across_machine),
    local_device_count_(local_device_count), global_device_count_(0),
    worker_rank_(worker_rank), worker_count_(worker_count) {
  global_device_count_ = sharding_across_machine_ ? \
      local_device_count_ * worker_count_ : local_device_count_;
  var_spec_mgr_ = std::make_unique<VariableSpecsMgr>(
      task_graph, variable_map_, sharding_across_machine_,
      local_device_count, worker_rank, worker_count);
  std::string path = absl::StrCat("./ckpt_", worker_rank, "_of_", worker_count, "/");
  ckpt_util_ = std::make_unique<CheckpointUtil>(path, ckpt_max_to_keep);
}

Status ExecutionState::InitializeTaskContexts() {
  for (auto& task : task_graph_->task_nodes()) {
    node_task_ctxs_map_[task->node_id()] =
        std::make_unique<TaskContext>(task.get());
  }
  return Status::OK();
}

void ExecutionState::InitializeDistributedSaver(
    std::vector<DAPPLEBuffer*>& local_variables, std::map<int, int>& arg_var_map,
    PjRtClient* gpu_client) {
  var_spec_mgr_->ExtractLocalVariableSpecs(local_variables, arg_var_map);
  ckpt_util_->Initialize(var_spec_mgr_.get(), gpu_client);
}

StatusOr<std::vector<DAPPLEBuffer*>> ExecutionState::PropagateOutputs(
    TaskNode* task_node, ScopedShapedBuffer output_buffers,
    absl::Span<DAPPLEBuffer* const>& input_args,
    std::shared_ptr<BufferSequencingEvent>& definition_event,
    PjRtClient* gpu_client, int local_device_id) {
  std::vector<DAPPLEBuffer*> task_outputs;
  auto* hlo_module = task_graph_->task_module(task_node);
  auto* root_instr = hlo_module->entry_computation()->root_instruction();
  if (!root_instr->shape().IsTuple()) {
    return InternalError(
        "root's shape is not tuple for task %s", task_node->name());
  }
  int64 tuple_count = root_instr->operand_count();
  auto task_def_ctx = task_node->def_ctx();
  auto& task_alias_map = task_def_ctx->input_output_alias_map_;

  auto* task_ctx = node_task_ctxs_map_[task_node->node_id()].get();
  task_outputs.resize(tuple_count, nullptr);

  VLOG(1) << "worker_rank_: " << worker_rank_ << ", local_device_count_: " << local_device_count_
          << ", local_device_id: " << local_device_id;
  int global_dev_id;
  if (sharding_across_machine_) {
    global_dev_id = worker_rank_ * local_device_count_ + local_device_id;
  } else {
    global_dev_id = local_device_id;
  }

  std::unordered_set<int> aliased_outputs;
  for (auto& it : task_alias_map) {
    auto arg_idx = it.first;
    auto out_idx = it.second;

    CHECK(arg_idx < input_args.size());
    auto pjrt_buf = input_args[arg_idx]->steal_gpu_buffer(global_dev_id);
    auto dapple_buf = task_ctx->ResolveOutput(out_idx).ConsumeValueOrDie();
    CHECK_EQ(dapple_buf->on_host_shape(), pjrt_buf->on_host_shape());
    dapple_buf->set_gpu_buffer(std::move(pjrt_buf), global_dev_id);
    task_outputs[out_idx] = dapple_buf;
    aliased_outputs.insert(out_idx);
  }

  Device* device = pjrt_utils::LookupDevice(*gpu_client, local_device_id);
  LocalDeviceState* device_state = &gpu_client->device_state(local_device_id);
  for (int64 i = 0; i < tuple_count; ++i) {
    if (aliased_outputs.count(i)) continue;
    auto dapple_buf = task_ctx->ResolveOutput(i).ConsumeValueOrDie();
    ScopedShapedBuffer tuple_buffer = output_buffers.TakeSubTree({i});
    auto pjrt_buf = pjrt_utils::OutputBufferHelper(
        &tuple_buffer, definition_event, gpu_client, device, device_state);
    CHECK_EQ(dapple_buf->on_host_shape(), pjrt_buf->on_host_shape());
    dapple_buf->set_gpu_buffer(std::move(pjrt_buf), global_dev_id);
    task_outputs[i] = dapple_buf;
  }
  return task_outputs;
}

void ExecutionState::PrepareInputsForTask(TaskNode* task_node) {
  switch (task_node->task_type()) {
    case TaskNode::TaskType::kSend: {
      PrepareInputsForSendTask(task_node);
      break;
    }

    case TaskNode::TaskType::kRecv: {
      PrepareInputsForRecvTask(task_node);
      break;
    }

    default: PrepareInputsForCommonTask(task_node);
  }
}

void ExecutionState::PrepareInputsForCommonTask(TaskNode* task_node) {
  CHECK(task_node);
  auto& input_specs = task_node->input_specs();
  int64 input_specs_size = input_specs.size();

  auto* task_ctx = node_task_ctxs_map_[task_node->node_id()].get();
  CHECK(task_ctx);
  auto* input_buffers = task_ctx->mutable_input_buffers();
  CHECK(input_buffers);
 
  VLOG(1) << "input_specs_size: " << input_specs_size;
  for (int64 arg_no = 0; arg_no < input_specs_size; ++arg_no) {
    auto& specs = input_specs.at(arg_no);
    CHECK(specs.size() == 1);
    auto parent = specs[0].first;
    CHECK(parent);

    auto* parent_task_ctx = node_task_ctxs_map_[parent->node_id()].get();
    auto& parent_output_buffers = parent_task_ctx->output_buffers();
    auto out_idx = specs[0].second;
    CHECK(out_idx < parent_output_buffers.size()) << "out_idx: " << out_idx
          << ", parent_output_buffers.size: " << parent_output_buffers.size()
          << "at tasknode " << task_node->name()
          << "and its parent " << task_node->parents().front()->name();
    auto dapple_buf = parent_output_buffers[out_idx];
    CHECK(dapple_buf);
    input_buffers->emplace_back(dapple_buf);
  }
}

void ExecutionState::PrepareInputsForSendTask(TaskNode* task_node) {
  auto* parent_task = task_node->parent();
  auto* parent_task_ctx = node_task_ctxs_map_[parent_task->node_id()].get();

  auto* task_ctx = node_task_ctxs_map_[task_node->node_id()].get();
  auto& parent_output_buffers = parent_task_ctx->output_buffers();
  auto input_buffers = task_ctx->mutable_input_buffers();
  input_buffers->clear();
  auto& port_map = task_node->port_map();
  CHECK(!port_map.empty());
  input_buffers->reserve(port_map.size());
  for (auto& it : port_map) {
    auto out_idx = it.first;
    CHECK(out_idx < parent_output_buffers.size());
    input_buffers->emplace_back(parent_output_buffers[out_idx]);
  }
}

void ExecutionState::PrepareInputsForRecvTask(TaskNode* task_node) {
  if (task_node->def_ctx()->entry_def_ctx()) return;
  auto* task_ctx = node_task_ctxs_map_[task_node->node_id()].get();
  auto* hlo_module = task_graph_->task_module(task_node);
  auto& port_map = task_node->port_map();
  CHECK(!port_map.empty());
  auto* input_buffers = task_ctx->mutable_input_buffers();
  input_buffers->resize(port_map.size());
  bool buffer_save = ServiceEnv::buffer_save();
  
  for (auto& it : port_map) {
    if(! buffer_save) {
      auto* dapple_buf = task_ctx->ResolveOutput(it.second).ConsumeValueOrDie();
      (*input_buffers)[it.second] = dapple_buf;
    } else {
      int device_id = task_node->buffer_info().local_device();
      int buffer_type = task_node->buffer_info().buffer_type();
      int buffer_id = task_node->buffer_info().buffer_id();
      auto key = std::make_pair(device_id, buffer_type);
      CHECK(recv_dapple_buffer_ptr_.find(key) != recv_dapple_buffer_ptr_.end());
      CHECK(recv_dapple_buffer_ptr_[key].size() > buffer_id);
      auto& task_outputs = recv_dapple_buffer_ptr_[key][buffer_id];
      CHECK(it.second >= 0 && it.second < task_outputs.size());
      auto* dapple_buf = task_outputs[it.second].get();
      (*input_buffers)[it.second] = dapple_buf;
    }
  }
}

void ExecutionState::InitializeLocalVariables(
    std::vector<DAPPLEBuffer*>& local_variables, std::map<int, int>& arg_var_map,
    std::map<int, string>& init_specs_map, PjRtClient* gpu_client) {
  var_spec_mgr_->ExtractLocalVariableSpecs(local_variables, arg_var_map);
  auto& name_spec_map = var_spec_mgr_->name_spec_map();
  for (auto& iter : name_spec_map) {
    auto& var_spec = iter.second;
    VLOG(1) << "Initialize arguments for " << var_spec.top_arg_no
            << ", dapple_buffer shape : "
            << ShapeUtil::HumanString(var_spec.d_buf->on_device_shape());
    auto& rng_config = init_specs_map.at(var_spec.top_arg_no);
    std::vector<std::thread> workers;
    DistributionConfig proto;
    proto.ParseFromString(rng_config);
    for (auto& pair : var_spec.start_offset_pairs_map) {
      workers.push_back(std::thread([pair, var_spec, proto, gpu_client]() {
        char* temp_tensor = (char *)malloc(ShapeUtil::ByteSizeOf(var_spec.arg_shape));
        auto& global_local_dev_map = var_spec.global_local_dev_map;
        int64 global_dev_id = pair.first;
        CHECK(global_local_dev_map.find(global_dev_id) != global_local_dev_map.end());
        int64 local_dev_id = global_local_dev_map.at(global_dev_id);
        DistInitializationWrapper(
            proto, var_spec.d_buf->on_host_shape(), var_spec.arg_shape,
            pair.second, temp_tensor);
        Status h2d_stat = DAPPLEBufferUtils::H2D(
            var_spec.d_buf, var_spec.arg_shape, local_dev_id,
            global_dev_id, gpu_client, (char*)temp_tensor);
        CHECK(h2d_stat.ok());
        free(temp_tensor);
      }));
    }
    std::for_each(workers.begin(), workers.end(), [](std::thread &t) { t.join(); });
  }
}

void ExecutionState::ResolveLocalVariables(
    absl::Span<DAPPLEBuffer* const> argument_handles,
    std::vector<DAPPLEBuffer*> *local_variables, std::map<int, int>* arg_var_map) {
  local_variables->clear();
  arg_var_map->clear();
  TaskNode* split_task = task_graph_->source();
  for (auto* task : split_task->children()) {
    for (auto& iter : task->input_specs()) {
      auto& node_arg_pair = iter.second[0];
      int arg_no = node_arg_pair.second;
      if (!variable_map_->count(arg_no) || arg_var_map->count(arg_no)) continue;
      arg_var_map->insert(std::make_pair(arg_no, local_variables->size()));
      local_variables->emplace_back(argument_handles[arg_no]);
    }
  }
}

Status ExecutionState::InitializeAllTaskOutputBuffers(
    Device* virtual_device, PjRtClient* gpu_client, se::Platform* platform) {
  bool buffer_save = ServiceEnv::buffer_save();
  for (auto& iter : node_task_ctxs_map_) {
    auto& task_ctx = iter.second;
    if(buffer_save && task_ctx->task_node()->task_type() == TaskNode::TaskType::kRecv) {
      bool buffer_reused = task_ctx->task_node()->buffer_info().buffer_reused();
      if(! buffer_reused) { // creat new dapple buffer
        auto* def_ctx = task_ctx->task_node()->def_ctx();
        if (def_ctx->entry_def_ctx()) continue; // SplitRecv task is placed on host.
        auto& buffer_info = task_ctx->task_node()->buffer_info();
        CreateRecvTaskBuffers(virtual_device, gpu_client, platform, buffer_info);
      }  // else reused existed buffer
    } else {
      task_ctx->InitializeOutputBuffers(
        virtual_device, gpu_client, platform, global_device_count_);
    } 
  }
  return Status::OK();
}

Status ExecutionState::CreateRecvTaskBuffers(
  Device* virtual_device, PjRtClient* gpu_client, se::Platform* platform, const BufferInfo& buffer_info) {
  int global_device = buffer_info.global_device();
  int device_id = buffer_info.local_device();
  int buffer_type = buffer_info.buffer_type();
  std::vector<Shape> buffer_shape = buffer_info.buffer_shape();
  Device* gpu_device = pjrt_utils::LookupDevice(*gpu_client, device_id);
  auto local_device = gpu_device->local_device_state();
  auto key = std::make_pair(device_id, buffer_type);
  recv_dapple_buffer_ptr_[key].push_back(RecvTaskOutputs());
  auto& task_outputs = recv_dapple_buffer_ptr_[key].back();
  task_outputs.reserve(buffer_shape.size());
  for (auto& shape : buffer_shape) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer> py_buffer,
        pjrt_utils::AllocateDestinationBuffer(
            shape, gpu_device, local_device,
            local_device->host_to_device_stream(), gpu_client));

    PjRtBuffer::ScopedHold device_buffer(
                           py_buffer->GetBufferWithUsageHold());
    CHECK(device_buffer.ok());

    ShapedBuffer buffer = device_buffer->AsShapedBuffer(
        shape, shape, platform);

    std::shared_ptr<BufferSequencingEvent> event =
        device_buffer->definition_events()[0];

    TF_CHECK_OK(pjrt_utils::AddDestinationBufferSynchronization(
        local_device, std::move(device_buffer), event,
        local_device->host_to_device_stream()));
    auto dapple_buf = DAPPLEBuffer::CreateDAPPLEBufferUnique(
                          virtual_device, shape, shape, std::move(py_buffer),
                          global_device_count_, global_device);
    task_outputs.push_back(std::move(dapple_buf));
  }
  return Status::OK();
}

TaskContext* ExecutionState::get_task_context(int node_id) {
  return node_task_ctxs_map_[node_id].get();
}

void ExecutionState::CleanupCachedTasksOutputsBuffers() {
  for (auto& iter : node_task_ctxs_map_) {
    auto& task_ctx = iter.second;
    CHECK(task_ctx->task_node());
    if (task_ctx->task_node()->task_type() == TaskNode::TaskType::kRecv)
      continue;
    task_ctx->CleanupTaskOutputs();
  }
}

void ExecutionState::CleanupPjRtBuffersTriggeredBy(TaskNode* task_node) {
  auto& mem_to_release = task_node->mem_to_release();

  int64 free_pool = 0;
  for (auto& node_out_idx_pair : mem_to_release) {
    auto node_id = node_out_idx_pair.first;
    auto out_idx = node_out_idx_pair.second;
    auto* task_ctx = node_task_ctxs_map_[node_id].get();
    auto* dapple_buffer = task_ctx->ResolveOutput(out_idx).ConsumeValueOrDie();
    dapple_buffer->DeletePjRtBuffers();
    free_pool += ShapeUtil::ByteSizeOf(dapple_buffer->on_host_shape());
  }

  if (free_pool>0) {
    VLOG(1) << task_node->name() << " freed->" << free_pool << "\n";
  }
}

void ExecutionState::set_global_step(const int64 global_step) {
  global_step_ = global_step;
}

Status ExecutionState::SaveCheckpoint() {
  ckpt_util_->Save(global_step_);
  return Status::OK();
}

Status ExecutionState::RestoreFromCheckpoint() {
  ckpt_util_->Restore(global_step_);
  return Status::OK();
}

} // namespace xla
