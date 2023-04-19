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

#include "tensorflow/compiler/xla/pjrt/dapple_buffer_utils.h"
#include "tensorflow/compiler/xla/pjrt/slice_utils.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

/*static*/
Status DAPPLEBufferUtils::H2D(DAPPLEBuffer* dapple_buf, Shape shape,
    int replica_id, int num_replicas, int local_device_id, int global_device_id,
    int64 stride, PjRtClient* gpu_client, const void* raw) {
  if (dapple_buf->on_host_shape() == shape) {
    // Replication case
    return DAPPLEBufferUtils::H2D(
        dapple_buf, shape, local_device_id, global_device_id,
        gpu_client, raw ? raw : dapple_buf->raw());
  }

  if (dapple_buf->on_host_shape() != dapple_buf->on_device_shape()) {
    return InvalidArgument("on_host_shape must equal on_device_shape in H2D");
  }

  if (shape.IsTuple()) {
    return InvalidArgument("Use FromHostLiteral to transfer a tuple");
  }

  if (!gpu_client) {
    return InvalidArgument("gpu_client should not be nullptr");
  }

  // shape->replica shape
  int64 full_size = ShapeUtil::ByteSizeOf(dapple_buf->on_host_shape());
  int64 replica_size = ShapeUtil::ByteSizeOf(shape);

  CHECK(replica_size > 0);
  char* data = (char *)malloc(replica_size);
  CHECK(data);
  auto data_base = raw ? raw : dapple_buf->raw();
  int64 elem_size = ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int plane = elem_size * stride / num_replicas;
  for (int i = 0; i < full_size / (plane * num_replicas); ++i) {
    int src_offset = i * plane * num_replicas + replica_id * plane;
    int dst_offset = i * plane;
    CHECK(src_offset < full_size);
    CHECK(dst_offset < replica_size);
    const char* src = (const char*)data_base + src_offset;
    std::memcpy(data + dst_offset, src, plane);
  }

  auto gpu_device = gpu_client->local_devices().at(local_device_id);
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      gpu_device->GetLocalDeviceState());

  TransferManager* transfer_manager =
      gpu_client->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer> py_buffer,
      pjrt_utils::AllocateDestinationBuffer(compact_shape, gpu_device, local_device,
                                            local_device->host_to_device_stream(),
                                            gpu_client));

  PjRtBuffer::ScopedHold device_buffer(py_buffer->GetBufferWithUsageHold());
  CHECK(device_buffer.ok());

  // The host to device transfer is performed on a thread pool, mostly because
  // it includes linearization that may be slow. It is OK to capture the
  // py_buffer pointer because the py_buffer can't be deleted until all the
  // usage holds have gone away.
  // TODO(misard) assess if it would be preferable to introduce a heuristic to
  // put the transfer into the calling thread for small literals.
  auto transfer_h2d = [gpu_client, transfer_manager, local_device, replica_size,
                       movable_device_buffer{device_buffer.ToClosure()},
                       data, shape, py_buffer{py_buffer.get()}, compact_shape,
                       on_device_shape{py_buffer->on_device_shape()},
                       buffer_reference{std::move(dapple_buf->py_buf_ref_)}] () {
    PjRtBuffer::ScopedHold device_buffer(movable_device_buffer);
    // This function uses TF_CHECK_OK and ValueOrDie() since we have no way
    // to report failures from a callback. However, the operations here are
    // unlikely to fail and not recoverable even if we were to fail: DMAs to
    // memory that has already been allocated, and a possible Event
    // allocation.

    ShapedBuffer buffer = device_buffer->AsShapedBuffer(
        compact_shape, on_device_shape, gpu_client->client()->platform());

    std::shared_ptr<void> staging_buffer;

    // If applicable on the backend, stage the transfer via host memory
    // allocated via the host_memory_allocator. On GPU, this is pinned
    // memory.
    if (gpu_client->host_memory_allocator()) {
      void* ptr = gpu_client->host_memory_allocator()->AllocateRaw(
          tensorflow::Allocator::kAllocatorAlignment, replica_size);
      staging_buffer = std::shared_ptr<void>(ptr, [gpu_client](void* ptr) {
        gpu_client->host_memory_allocator()->DeallocateRaw(ptr);
      });
      std::memcpy(ptr, data, replica_size);
      BorrowingLiteral literal(static_cast<const char*>(staging_buffer.get()),
                               shape);
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          local_device->host_to_device_stream(), literal, buffer));
    } else {
      BorrowingLiteral literal(static_cast<const char*>(data), shape);
      // Otherwise, just transfer the literal.
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          local_device->host_to_device_stream(), literal, buffer));
    }

    free(data);

    std::shared_ptr<BufferSequencingEvent> event =
        device_buffer->definition_events()[0];
    TF_CHECK_OK(pjrt_utils::AddDestinationBufferSynchronization(
        local_device, std::move(device_buffer), event,
        local_device->host_to_device_stream()));

    local_device->ThenRelease(
        local_device->host_to_device_stream(),
        std::make_pair(buffer_reference, std::move(staging_buffer)));
  };
  gpu_client->h2d_transfer_pool()->Schedule(transfer_h2d);
  dapple_buf->set_gpu_buffer(std::move(py_buffer), global_device_id);
  return Status::OK();
}

/*static*/
Status DAPPLEBufferUtils::H2D(DAPPLEBuffer* dapple_buf, Shape shape,
    int local_device_id, int global_device_id,
    PjRtClient* gpu_client, const void* sharded_tensor) {
  if (dapple_buf->on_host_shape() != dapple_buf->on_device_shape()) {
    return InvalidArgument("on_host_shape must equal on_device_shape in H2D");
  }

  if (shape.IsTuple()) {
    return InvalidArgument("Use FromHostLiteral to transfer a tuple");
  }

  if (!gpu_client) {
    return InvalidArgument("gpu_client should not be nullptr");
  }

  int64 replica_size = ShapeUtil::ByteSizeOf(shape);
  auto gpu_device = gpu_client->local_devices().at(local_device_id);
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      gpu_device->GetLocalDeviceState());

  TransferManager* transfer_manager =
      gpu_client->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer> py_buffer,
      pjrt_utils::AllocateDestinationBuffer(compact_shape, gpu_device, local_device,
                                            local_device->host_to_device_stream(),
                                            gpu_client));

  PjRtBuffer::ScopedHold device_buffer(py_buffer->GetBufferWithUsageHold());
  CHECK(device_buffer.ok());
  ShapedBuffer buffer = device_buffer->AsShapedBuffer(
      compact_shape, py_buffer->on_device_shape(), gpu_client->client()->platform());
  BorrowingLiteral literal(static_cast<const char*>(sharded_tensor), shape);
  TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
      local_device->host_to_device_stream(), literal, buffer));

  std::shared_ptr<BufferSequencingEvent> event =
        device_buffer->definition_events()[0];
  TF_CHECK_OK(pjrt_utils::AddDestinationBufferSynchronization(
      local_device, std::move(device_buffer), event,
      local_device->host_to_device_stream()));

  local_device->ThenRelease(
      local_device->host_to_device_stream(), std::move(dapple_buf->py_buf_ref_));

  dapple_buf->set_gpu_buffer(std::move(py_buffer), global_device_id);
  return Status::OK();
}

/*static*/
Status DAPPLEBufferUtils::H2D(DAPPLEBuffer* dapple_buf, Shape shape,
    int local_device_id, int global_device_id, const DistSpec& dist_spec,
    const std::vector<int64>& split_ids, PjRtClient* gpu_client) {

  if (dapple_buf->on_host_shape() != dapple_buf->on_device_shape()) {
    return InvalidArgument("on_host_shape must equal on_device_shape in H2D");
  }

  if (shape.IsTuple()) {
    return InvalidArgument("Use FromHostLiteral to transfer a tuple");
  }

  if (!gpu_client) {
    return InvalidArgument("gpu_client should not be nullptr");
  }

  int64 elem_size = ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64 replica_size = ShapeUtil::ByteSizeOf(shape);
  void* data = (void *)malloc(replica_size);
  // 2. Copy slice on CPU buffer
  SliceUtils::SliceCopyOnHost(
      dapple_buf->on_host_shape(), dist_spec, split_ids, dapple_buf->raw(), data);

  // 3. Copy CPU buffer to device
  return DAPPLEBufferUtils::H2D(
      dapple_buf, shape, local_device_id, global_device_id,
      gpu_client, data);
}

} // namespace xla
