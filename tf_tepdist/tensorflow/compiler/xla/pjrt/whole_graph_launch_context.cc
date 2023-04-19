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

#include "tensorflow/compiler/xla/pjrt/whole_graph_launch_context.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {

void SplitRecvArgsRecorder::RecordVariableHandle(
    GlobalDataHandle& data_handle, DAPPLEBufferHandle& buf_handle, int global_idx) {
  variable_data_handles.emplace_back(data_handle);
  variable_buf_handles.emplace_back(buf_handle);
  VLOG(2) << "RecordVariable : idx = " << global_idx;
  recv_args_storage_map[global_idx] = std::make_pair(true/*variable*/, buf_handle);
}

void SplitRecvArgsRecorder::RecordSampleIndex(int global_idx) {
  VLOG(2) << "RecordSampleIndex : idx = " << global_idx;
  recv_args_storage_map[global_idx] = std::make_pair(false/*variable*/, global_idx);
}

WholeGraphLaunchContext::WholeGraphLaunchContext(int local_dev_count)
  : local_dev_count_(local_dev_count) {}

Status WholeGraphLaunchContext::DoRegisterDAPPLEBuffer(
    DAPPLEBufferHandle handle, std::unique_ptr<DAPPLEBuffer> dapple_buf,
    Literal literal) {
  if (int64(handle) < int64(resource_variables_.size())) {
    return InvalidArgument("Repeated registration for %d", handle);
  }

  CHECK(int64(handle) == int64(resource_variables_.size()));
  resource_variables_.emplace_back(std::move(dapple_buf));
  literals_.emplace_back(std::move(literal));

  return Status::OK();
}

Status WholeGraphLaunchContext::DoRegisterDAPPLEBuffer(
    DAPPLEBufferHandle handle, std::unique_ptr<DAPPLEBuffer> dapple_buf) {
  if (int64(handle) < int64(resource_variables_.size())) {
    return InvalidArgument("Repeated registration for %d", handle);
  }

  CHECK(int64(handle) == int64(resource_variables_.size()));
  resource_variables_.emplace_back(std::move(dapple_buf));

  return Status::OK();
}

Status WholeGraphLaunchContext::RegisterVariable(
    DAPPLEBufferHandle handle, Literal literal) {
  auto raw_data = literal.untyped_data();
  auto& shape = literal.shape();
  CHECK(!shape.IsTuple());
  VLOG(2) << "host buffer local device count: " << local_dev_count_;
  VLOG(2) << "host buffer worker count: " << worker_count_;
  CHECK(worker_count_>0);
  int dev_count;
  if (sharding_across_machine_) {
    dev_count = local_dev_count_ * worker_count_;
  } else {
    dev_count = local_dev_count_;
  }
  auto buf_status = DAPPLEBuffer::FromHostBuffer(
      raw_data, shape, dev_count);
  CHECK(buf_status.ok());
  auto dapple_buf = buf_status.ConsumeValueOrDie();

  DoRegisterDAPPLEBuffer(
      handle, std::move(dapple_buf), std::move(literal));
  return Status::OK();
}

Status WholeGraphLaunchContext::RegisterVariable(
    DAPPLEBufferHandle handle, const Shape& shape) {
  VLOG(2) << "host buffer local device count: " << local_dev_count_;
  VLOG(2) << "host buffer worker count: " << worker_count_;
  CHECK(worker_count_>0);
  int dev_count;
  if (sharding_across_machine_) {
    dev_count = local_dev_count_ * worker_count_;
  } else {
    dev_count = local_dev_count_;
  }
  auto buf_status = DAPPLEBuffer::FromHostBuffer(
      nullptr, shape, dev_count);
  CHECK(buf_status.ok());
  auto dapple_buf = buf_status.ConsumeValueOrDie();
  DoRegisterDAPPLEBuffer(handle, std::move(dapple_buf));
  return Status::OK();
}

StatusOr<DAPPLEBuffer*> WholeGraphLaunchContext::Resolve(
    const DAPPLEBufferHandle handle) {
  if (int64(handle) >= int64(resource_variables_.size())) {
    return NotFound("No allocation record for DAPPLEBuffer handle: %d",
                    handle);
  }

  return resource_variables_[handle].get();
}

StatusOr<std::vector<DAPPLEBuffer*>>
WholeGraphLaunchContext::Resolve(const std::vector<DAPPLEBufferHandle>& handles) {
  std::vector<DAPPLEBuffer*> variables;
  variables.resize(handles.size());
  for (int i = 0; i < handles.size(); ++i) {
    TF_ASSIGN_OR_RETURN(variables[i], Resolve(handles[i]));
  }
  return variables;
}

StatusOr<DAPPLEBuffer*> WholeGraphLaunchContext::ResolveOutput(int64 idx) {
  if (idx >= int64(outputs_.size())) {
    return NotFound("No allocation record for output DAPPLEBuffer idx: %d",
                    idx);
  }
  return outputs_[idx].get();
}

void WholeGraphLaunchContext::CleanupVariablesOnHost() {
  literals_.clear();
}

Status WholeGraphLaunchContext::InitializePlanOutputsBuffers(
    HloModule* entry_module) {
  if (!entry_module) {
    return NotFound("No entry_module_ has been found");
  }

  auto entry = entry_module->entry_computation();
  auto entry_root = entry->root_instruction();
  int64 num_outputs = entry_root->operand_count();
  auto* vars_map = entry_module->variable_map();
  int64 num_fetches = num_outputs - vars_map->size();

  auto& root_shape = entry_root->shape();
  CHECK(root_shape.IsTuple());

  VLOG(2) << "host buffer local device count: " << local_dev_count_;
  VLOG(2) << "host buffer worker count: " << worker_count_;
  CHECK(worker_count_>0);
  int dev_count;
  if (sharding_across_machine_) {
    dev_count = local_dev_count_ * worker_count_;
  } else {
    dev_count = local_dev_count_;
  }
  for (int64 i = 0; i < num_fetches; ++i) {
    auto dapple_buf = DAPPLEBuffer::FromHostBuffer(
        nullptr, root_shape.tuple_shapes(i),
        dev_count).ConsumeValueOrDie();
    outputs_.emplace_back(std::move(dapple_buf));
  }

  return Status::OK();
}

StatusOr<DAPPLEBuffer*>
WholeGraphLaunchContext::ResolveSampleInputBuffer(
    int64 idx, const Shape& shape, Device* virtual_device) {
  if (sample_inputs_.find(idx) == sample_inputs_.end()) {
    // First iteration
    VLOG(2) << "host buffer local device count: " << local_dev_count_;
    VLOG(2) << "host buffer worker count: " << worker_count_;
    CHECK(worker_count_>0);
    int dev_count;
    if (sharding_across_machine_) {
      dev_count = local_dev_count_ * worker_count_;
    } else {
      dev_count = local_dev_count_;
    }
    auto dapple_buf = DAPPLEBuffer::CreateDAPPLEBufferUnique(
        virtual_device, shape, shape, dev_count,
        true/*alloc_raw_internal*/);
    auto dapple = dapple_buf.get();
    sample_inputs_[idx] = std::move(dapple_buf);
    return dapple;
  }

  // Coming iterations
  return sample_inputs_[idx].get();
}

StatusOr<DAPPLEBuffer*>
WholeGraphLaunchContext::ResolveSampleInputBuffer(int64 idx) {
  CHECK (sample_inputs_.find(idx) != sample_inputs_.end());
  return sample_inputs_[idx].get();
}

} // namespace
