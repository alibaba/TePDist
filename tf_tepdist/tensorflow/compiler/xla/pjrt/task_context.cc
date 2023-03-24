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

#include "tensorflow/compiler/xla/pjrt/task_context.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {

TaskContext::TaskContext(TaskNode* task_node) : task_node_(task_node) {}

Status TaskContext::InitializeOutputBuffers(
    Device* virtual_device, PjRtClient* gpu_client, se::Platform* platform,
    int global_device_count) {
  switch (task_node_->task_type()) {
    case TaskNode::TaskType::kGAInit:
    case TaskNode::TaskType::kGA:
    case TaskNode::TaskType::kCompute: {
      CHECK(InitializeBuffersOnly(global_device_count).ok());
      break;
    }
    case TaskNode::TaskType::kRecv: {
      CHECK(InitializeBuffersOnGPU(
          virtual_device, gpu_client, platform, global_device_count).ok());
      break;      
    }
    default: break;/*do nothing*/
  }

  return Status::OK();
}

Status TaskContext::InitializeBuffersOnGPU(
    Device* virtual_device, PjRtClient* gpu_client, se::Platform* platform,
    int global_device_count) {
  VLOG(1) << "Initialize DAPPLEBuffer on GPU for Recv task "
          << task_node_->name();
  auto* def_ctx = task_node_->def_ctx();
  auto* hlo_module = def_ctx->module();
  auto task_root = hlo_module->entry_computation()->root_instruction();
  auto& port_map = task_node_->port_map();
  CHECK(!port_map.empty());
  int num_ports = port_map.size();
  CHECK(num_ports <= task_root->operand_count() ||
        num_ports <= hlo_module->entry_computation()->num_parameters());

  // SplitRecv does not need any pre-allocate buffers.
  if (def_ctx->entry_def_ctx()) {
    CHECK_EQ(task_node_->device_id(), -1); // SplitRecv task is placed on host.
    return Status::OK();
  }

  std::shared_ptr<CommDevManager> comm_dev_mgr = task_node_->comm_dev_mgr();
  CHECK(comm_dev_mgr);
  int64 global_dev_id = comm_dev_mgr->global_dev_id(task_node_->split_id());
  int64 local_dev_id = comm_dev_mgr->local_dev_id(global_dev_id);

  Device* gpu_device = pjrt_utils::LookupDevice(*gpu_client, local_dev_id);
  auto local_device = gpu_device->local_device_state();
  for (auto& it : port_map) {
    auto out_idx = it.first;
    const Shape& shape = task_root->operand(out_idx)->shape();
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
                          global_device_count, global_dev_id);
    task_outputs_.push_back(std::move(dapple_buf));
  } // for (port_map)

  return Status::OK();
}

Status TaskContext::InitializeBuffersOnly(int global_device_count) {
  VLOG(2) << "Initialize Output Buffers for " << task_node_->name();
  VLOG(2) << "global_device_count: " << global_device_count;

  auto* hlo_module = task_node_->def_ctx()->module();
  auto entry = hlo_module->entry_computation();
  auto entry_root = entry->root_instruction();
  int64 num_outputs = entry_root->operand_count();

  auto& root_shape = entry_root->shape();
  CHECK(root_shape.IsTuple());

  task_outputs_.reserve(num_outputs);
  for (int64 i = 0; i < num_outputs; ++i) {
    auto dapple_buf = DAPPLEBuffer::CreateForPlaceholder(
        root_shape.tuple_shapes(i), global_device_count).ConsumeValueOrDie();
    task_outputs_.emplace_back(std::move(dapple_buf));
  }

  return Status::OK();
}

void TaskContext::CleanupTaskOutputs() {
  for (auto& dapple_buffer : task_outputs_) {
    dapple_buffer->DeletePjRtBuffers();
  }
}

StatusOr<DAPPLEBuffer*> TaskContext::ResolveOutput(int out_idx) {
  if (out_idx < 0 || out_idx >= int64(task_outputs_.size())) {
    return NotFound("No allocation record for DAPPLEBuf idx: %d task: %s",
                    out_idx, task_node_->name());
  }
  return task_outputs_[out_idx].get();
}

std::vector<DAPPLEBuffer*>& TaskContext::input_buffers() {
  return input_buffers_;
}

std::vector<DAPPLEBuffer*>& TaskContext::output_buffers() {
  return output_buffers_;
}

std::vector<DAPPLEBuffer*>* TaskContext::mutable_input_buffers() {
  return &input_buffers_;
}

std::vector<DAPPLEBuffer*>* TaskContext::mutable_output_buffers() {
  return &output_buffers_;
}

} // namespace xla
