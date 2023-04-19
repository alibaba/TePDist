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

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"

namespace xla {

DAPPLEBuffer::DAPPLEBuffer(Device* cpu_device, const void* raw,
                           const Shape& on_host_shape,
                           const Shape& on_device_shape,
                           int device_count)
    : virtual_device_(cpu_device), raw_(raw),
      on_host_shape_(on_host_shape),
      on_device_shape_(on_device_shape),
      device_count_(device_count),
      in_cpu_(true) {
  gpu_present_.assign(device_count, false);
  gpu_buffers_.resize(device_count);
}

DAPPLEBuffer::DAPPLEBuffer(Device* cpu_device,
                           const Shape& on_host_shape,
                           const Shape& on_device_shape,
                           int device_count,
                           bool alloc_raw_internal)
    : virtual_device_(cpu_device),
      on_host_shape_(on_host_shape),
      on_device_shape_(on_device_shape),
      device_count_(device_count),
      in_cpu_(true) {
  if (alloc_raw_internal) { 
    internal_raw_buffer_ = absl::make_unique<char[]>(
        ShapeUtil::ByteSizeOf(on_device_shape_));
    raw_ = internal_raw_buffer_.get();
  }
  gpu_present_.assign(device_count, false);
  gpu_buffers_.resize(device_count);
}

DAPPLEBuffer::DAPPLEBuffer(Device* cpu_device, const void* raw,
                           const Shape& on_host_shape,
                           const Shape& on_device_shape,
                           std::shared_ptr<void> py_buf_ref,
                           int device_count)
      : virtual_device_(cpu_device), raw_(raw), 
        on_host_shape_(on_host_shape),
        on_device_shape_(on_device_shape),
        py_buf_ref_(py_buf_ref), 
        device_count_(device_count),
        in_cpu_(true) { 
    gpu_present_.assign(device_count, false);
    gpu_buffers_.resize(device_count);
}

DAPPLEBuffer::DAPPLEBuffer(Device* cpu_device,
                           const Shape& on_host_shape,
                           const Shape& on_device_shape,
                           std::unique_ptr<PjRtBuffer> gpu_buffer,
                           int device_count, int device_id)
    : virtual_device_(cpu_device), raw_(nullptr),
      on_host_shape_(on_host_shape),
      on_device_shape_(on_device_shape),
      device_count_(device_count),
      in_cpu_(false) {
  CHECK(device_id < device_count);
  gpu_present_.assign(device_count, false);
  gpu_buffers_.resize(device_count);
 
  CHECK_EQ(on_host_shape, on_device_shape);
  CHECK_EQ(on_host_shape, gpu_buffer->on_host_shape());

  gpu_present_[device_id] = true;
  gpu_buffers_[device_id] = std::move(gpu_buffer);
}

DAPPLEBuffer::~DAPPLEBuffer() { Delete(); }

DAPPLEBuffer::DAPPLEBuffer(DAPPLEBuffer&& d_buf)
    : virtual_device_(d_buf.virtual_device_),
      raw_(d_buf.raw_),
      on_host_shape_(std::move(d_buf.on_host_shape_)),
      on_device_shape_(std::move(d_buf.on_device_shape_)),
      py_buf_ref_(std::move(d_buf.py_buf_ref_)),
      device_count_(d_buf.device_count_),
      in_cpu_(d_buf.in_cpu_),
      sharded_(d_buf.sharded_),
      gpu_buffers_(std::move(d_buf.gpu_buffers_)),
      gpu_present_(std::move(d_buf.gpu_present_)) {}

DAPPLEBuffer& DAPPLEBuffer::operator=(DAPPLEBuffer&& d_buf) {
  virtual_device_ = d_buf.virtual_device_;
  device_count_ = d_buf.device_count_;
  on_host_shape_ = std::move(d_buf.on_host_shape_);
  on_device_shape_ = std::move(d_buf.on_device_shape_);
  py_buf_ref_ = std::move(d_buf.py_buf_ref_);
  in_cpu_ = d_buf.in_cpu_;
  sharded_ = d_buf.sharded_;
  gpu_present_ = std::move(d_buf.gpu_present_);
  gpu_buffers_ = std::move(d_buf.gpu_buffers_);
  return *this;
}

/* static */
DAPPLEBuffer*
DAPPLEBuffer::CreateDAPPLEBuffer(Device* cpu_device, const void* raw,
                                 const Shape& on_host_shape,
                                 const Shape& on_device_shape,
                                 int device_count)
{
  return new DAPPLEBuffer(cpu_device, raw, on_host_shape,
                          on_device_shape, device_count);
}

/* static */
DAPPLEBuffer*
DAPPLEBuffer::CreateDAPPLEBuffer(Device* cpu_device, const void* raw,
                                 const Shape& on_host_shape,
                                 const Shape& on_device_shape,
                                 std::shared_ptr<void> py_buf_ref,
                                 int device_count)
{
  return new DAPPLEBuffer(cpu_device, raw, on_host_shape,
                          on_device_shape, py_buf_ref, device_count);
}

/* static */
DAPPLEBuffer*
DAPPLEBuffer::CreateDAPPLEBuffer(Device* cpu_device,
                                 const Shape& on_host_shape,
                                 const Shape& on_device_shape,
                                 int device_count,
                                 bool alloc_raw_internal /*= false*/)
{
  return new DAPPLEBuffer(cpu_device, on_host_shape,
                          on_device_shape, device_count, alloc_raw_internal);
}

/* static */
DAPPLEBuffer*
DAPPLEBuffer::CreateDAPPLEBuffer(Device* cpu_device,
                                 const Shape& on_host_shape,
                                 const Shape& on_device_shape,
                                 std::unique_ptr<PjRtBuffer> gpu_buffer,
                                 int device_count, int device_id /*= 0*/)
{
  return new DAPPLEBuffer(cpu_device, on_host_shape, on_device_shape,
                          std::move(gpu_buffer), device_count, device_id);
}

/* static */
std::unique_ptr<DAPPLEBuffer>
DAPPLEBuffer::CreateDAPPLEBufferUnique(Device* cpu_device, const void* raw,
                                 const Shape& on_host_shape,
                                 const Shape& on_device_shape,
                                 std::shared_ptr<void> py_buf_ref,
                                 int device_count)
{
  return std::unique_ptr<DAPPLEBuffer>(new DAPPLEBuffer(
                                            cpu_device, raw, on_host_shape,
                                            on_device_shape, py_buf_ref,
                                            device_count));
}

/* static */
std::unique_ptr<DAPPLEBuffer>
DAPPLEBuffer::CreateDAPPLEBufferUnique(Device* cpu_device,
                                 const Shape& on_host_shape,
                                 const Shape& on_device_shape,
                                 int device_count,
                                 bool alloc_raw_internal /*= false*/)
{
  return std::unique_ptr<DAPPLEBuffer>(new DAPPLEBuffer(
                                                  cpu_device, on_host_shape,
                                                  on_device_shape, device_count,
                                                  alloc_raw_internal));
}

/* static */
std::unique_ptr<DAPPLEBuffer>
DAPPLEBuffer::CreateDAPPLEBufferUnique(Device* cpu_device,
                                 const Shape& on_host_shape,
                                 const Shape& on_device_shape,
                                 std::unique_ptr<PjRtBuffer> gpu_buffer,
                                 int device_count, int device_id /*= 0*/)
{
  return std::unique_ptr<DAPPLEBuffer>(new DAPPLEBuffer(
                                                 cpu_device, on_host_shape,
                                                 on_device_shape,
                                                 std::move(gpu_buffer),
                                                 device_count, device_id));
}

/* static */
StatusOr<std::unique_ptr<DAPPLEBuffer>> DAPPLEBuffer::CreateForPlaceholder(
    const Shape& shape, int device_count) {
  if (shape.IsTuple()) {
    return InvalidArgument("Use FromHostLiteral to transfer a tuple");
  }

  return std::unique_ptr<DAPPLEBuffer>(new DAPPLEBuffer(nullptr, shape,
                                                        shape, device_count));
}

/* static */
StatusOr<std::unique_ptr<DAPPLEBuffer>> DAPPLEBuffer::FromHostBuffer(
    const void* data, const Shape& shape, int device_count) {
  if (shape.IsTuple()) {
    return InvalidArgument("Use FromHostLiteral to transfer a tuple");
  }

  return std::unique_ptr<DAPPLEBuffer>(new DAPPLEBuffer(nullptr, data, shape,
                                                        shape, device_count));
}

void DAPPLEBuffer::DeletePjRtBuffers() {
  int num_replicas = gpu_buffers_.size();
  CHECK(int(gpu_present_.size()) == num_replicas);
  for (int64 i = 0; i < num_replicas; ++i) {
    if (gpu_present_[i]) {
      gpu_buffers_[i]->Delete();
      gpu_present_[i] = false;
    }
  }
}

const int64 DAPPLEBuffer::num_valid_shards() const { 
  return std::count(gpu_present_.begin(), gpu_present_.end(), true);
}

PjRtBuffer* DAPPLEBuffer::first_valid_shard() {
  PjRtBuffer* pjrt_buf = nullptr;
  int num_replicas = gpu_buffers_.size();
  CHECK(int(gpu_present_.size()) == num_replicas);
  for (int64 i = 0; i < num_replicas; ++i) {
    if (gpu_present_[i]) {
      pjrt_buf = gpu_buffers_.at(i).get();
      break;
    }
  }
  return pjrt_buf;
}

PjRtBuffer* DAPPLEBuffer::gpu_buffer() {
  PjRtBuffer* pjrt_buf = first_valid_shard();
  if (!pjrt_buf) return nullptr;
  if (!ShapeUtil::Equal(on_host_shape(), pjrt_buf->on_host_shape())) {
    // Sharding check
    CHECK(1 == num_valid_shards());
  }
  return pjrt_buf;
}

std::unique_ptr<PjRtBuffer> DAPPLEBuffer::steal_gpu_buffer(int device_id) {
  VLOG(2) << "steal device_id: " << device_id << ", gpu_present_.size: " << gpu_present_.size()
          << ", dapple buffer ptr: " << this;
  CHECK(device_id < int(gpu_present_.size()));
  //CHECK(1 == num_valid_shards());
  CHECK(gpu_present_[device_id] == true);
  gpu_present_[device_id] = false;
  return std::move(gpu_buffers_[device_id]);
}

void DAPPLEBuffer::add_gpu_shard(
    std::unique_ptr<PjRtBuffer> shard, int device_id) {
  VLOG(2) << "add device_id: " << device_id << ", gpu_present_.size: " << gpu_present_.size()
          << ", dapple buffer ptr: " << this;
  CHECK(device_id < int(gpu_buffers_.size()));
  CHECK(gpu_buffers_.size() == gpu_present_.size());
  CHECK(!gpu_present_[device_id]);
  CHECK_EQ(on_host_shape(), shard->on_host_shape());
  gpu_buffers_[device_id] = std::move(shard);
  gpu_present_[device_id] = true;
}

PjRtBuffer* DAPPLEBuffer::gpu_buffer(int device_id) {
  CHECK(device_id < int(gpu_present_.size()));
  CHECK(true == gpu_present_[device_id]);
  return gpu_buffers_.at(device_id).get();
}

const bool DAPPLEBuffer::in_gpu(int dev_id) const {
  CHECK(dev_id < int(gpu_present_.size()));
  CHECK(gpu_present_.size() == gpu_buffers_.size());
  return gpu_present_.at(dev_id);
}

void DAPPLEBuffer::set_gpu_buffer(
    std::unique_ptr<PjRtBuffer> buf, int dev_id) {
  VLOG(2) << "set device_id: " << dev_id << ", gpu_present_.size: " << gpu_present_.size()
          << ", dapple buffer ptr: " << this;
  CHECK(dev_id < int(gpu_buffers_.size()));
  //In case of DP training, we have to replace the buffer with a new replica.
  const int64 on_host_bytes = ShapeUtil::ByteSizeOf(on_host_shape());
  const int64 on_device_bytes = ShapeUtil::ByteSizeOf(buf->on_device_shape());
  CHECK_EQ(on_host_bytes % on_device_bytes, 0);
  gpu_buffers_[dev_id] = std::move(buf);
  tensorflow::mutex_lock lock(mu_);
  CHECK(dev_id < int(gpu_present_.size()));
  gpu_present_[dev_id] = true;
}

// This actually updates the underlying host_value's Literal value of PjRtBuffer.
void DAPPLEBuffer::accumulate_gpu_buffer(
    std::unique_ptr<PjRtBuffer> buf_to_accumulate, int global_dev_id) {
  CHECK(global_dev_id < int(gpu_buffers_.size()));
  {
    tensorflow::mutex_lock lock(mu_);
    CHECK(global_dev_id < int(gpu_present_.size()));
    CHECK(gpu_present_[global_dev_id]);
    auto src_gpu_buffer = gpu_buffers_.at(global_dev_id).get();
    auto src_literal = src_gpu_buffer->ToLiteral().ConsumeValueOrDie();
    auto new_literal = buf_to_accumulate->ToLiteral().ConsumeValueOrDie();
    auto shape = src_literal->shape();
    // For simplification, currently we only process scalar type output
    // (e.g., loss) from GA iterations exists on each same device_id.
    CHECK_EQ(ShapeUtil::ElementsIn(shape), 1)
        << "Unhandled none-scaler output to be processed from one device";
    switch(shape.element_type()) {
      case F32: {
        float* ga_iter_accumulated = (float*)(src_literal->untyped_data());
        float* ga_iter_new = (float*)(new_literal->untyped_data());
        // Accumulated *in-place*.
        *ga_iter_accumulated += *ga_iter_new;
        break;
      }
      case S64: {
        // Fetching of global_step of type S64, just do nothing.
        break;
      }
      default:
        // TODO(shiqing.fsq): add more supported cases when needed.
        CHECK(0) << "Unhandled shape type";
    }
  }
  // Release useless memory.
  buf_to_accumulate->Delete();
}

void DAPPLEBuffer::average_gpu_buffer(int dev_id, int local_ga_iterations) {
  CHECK(dev_id < int(gpu_buffers_.size()) && local_ga_iterations > 1);
  {
    tensorflow::mutex_lock lock(mu_);
    CHECK(gpu_present_[dev_id]);
    auto src_gpu_buffer = gpu_buffers_.at(dev_id).get();
    auto src_literal = src_gpu_buffer->ToLiteral().ConsumeValueOrDie();
    auto shape = src_literal->shape();
    // For simplification, currently we only process scalar type output
    // (e.g., loss) from GA iterations exists on each same device_id.
    CHECK_EQ(ShapeUtil::ElementsIn(shape), 1)
        << "Unhandled none-scaler output to be processed from one device";
    switch(shape.element_type()) {
      case F32: {
        float* ga_iter_accumulated = (float*)(src_literal->untyped_data());
        // Averaged *in-place*.
        *ga_iter_accumulated /= local_ga_iterations;
        break;
      }
      case S64: {
        // Fetching of global_step of type S64, just do nothing.
        break;
      }
      default:
        // TODO(shiqing.fsq): add more supported cases when needed.
        CHECK(0) << "Unhandled shape type";
    }

    // Copy cached value of the buffer on the host to `device_buffer_`, so that
    // possible follow-up `NcclBcast` can consume the *up-to-date* value
    // on device.
    CopyHostCacheToDevice(dev_id);
  }
}

void DAPPLEBuffer::CopyHostCacheToDevice(int dev_id) {
  CHECK(dev_id < int(gpu_buffers_.size()) && gpu_present_[dev_id]);

  auto pjrt_buf = gpu_buffers_.at(dev_id).get();
  auto src_literal = pjrt_buf->ToLiteral().ConsumeValueOrDie();
  auto host_cache = src_literal->untyped_data();

  ShapedBuffer buffer = pjrt_buf->AsShapedBuffer().ConsumeValueOrDie();
  auto root_buf = buffer.root_buffer();
  void* gpu_buf_dst = root_buf.opaque();

  cudaMemcpy(gpu_buf_dst, host_cache, root_buf.size(), cudaMemcpyHostToDevice);
}

void* DAPPLEBuffer::mutable_raw() {
  auto raw = internal_raw_buffer_.get();
  CHECK(raw && raw == raw_);
  return raw; 
}

void DAPPLEBuffer::LossAverageAcrossDPInst() {
  if (num_valid_shards() <= 1) return;

  auto& whole_shape = on_device_shape();
  auto& shard_shape = first_valid_shard()->on_device_shape();

  // Sharding case: loss is already averaged. Do nothing.
  if (!ShapeUtil::Equal(whole_shape, shard_shape)) return;

  // Replication case.
  if (ShapeUtil::IsScalarWithElementType(whole_shape, F32)) {
    float loss_sum = 0.0;
    float* first_replica_src = nullptr;
    int first_dev_id = -1;
    for (int64 i = 0; i < gpu_present_.size(); ++i) {
      if (gpu_present_[i]) {
        auto gpu_buffer = gpu_buffers_.at(i).get();
        auto literal = gpu_buffer->ToLiteral().ConsumeValueOrDie();
        float* src = (float*)(literal->untyped_data());
        if (!first_replica_src) {
          first_replica_src = src;
          first_dev_id = i;
        }
        loss_sum += *src;
      }
    }
    float loss_average = loss_sum / num_valid_shards();

    // Copy the averaged loss to the first replica's host buffer
    CHECK(first_replica_src != nullptr && first_dev_id >= 0);
    std::memcpy(first_replica_src, &loss_average, sizeof(float));

    // Copy updated host cache to the first replica's gpu buffer.
    CopyHostCacheToDevice(first_dev_id);
  }
}

StatusOr<std::shared_ptr<Literal>> DAPPLEBuffer::ToLiteral() {
  for (int i = 0; i < gpu_present_.size(); ++i) {
    if (gpu_present_[i]) {
      gpu_buffers_[i]->ClearHostCache();
    }
  }
  PjRtBuffer* first_valid = first_valid_shard();

  if (first_valid == nullptr) {
    const void* dapple_raw = raw();
    // create literal from raw data
    std::shared_ptr<Literal> value = std::make_shared<Literal>(on_host_shape_);
    std::memcpy(value->untyped_data(), dapple_raw, ShapeUtil::ByteSizeOf(on_host_shape_));
    return value;
  }

  CHECK(first_valid != nullptr);
  
  auto& whole_shape = on_device_shape();
  auto& shard_shape = first_valid->on_device_shape();

  if (num_valid_shards() == 1 || ShapeUtil::Equal(whole_shape, shard_shape)) {
    first_valid->ClearHostCache();
    for (int i = 0; i < num_shards(); ++i) {
      if (gpu_present_[i]) return gpu_buffer(i)->ToLiteral();
    }
    return first_valid->ToLiteral();
  }

  // Sharding case
  int64 hi = 1, lo = 1;
  bool broken = false;
  int64 elem_size = 
      ShapeUtil::ByteSizeOfPrimitiveType(whole_shape.element_type());
  for (int r = 0; r < whole_shape.rank(); ++r) {
    if (whole_shape.dimensions(r) != shard_shape.dimensions(r)) {
      broken = true;
    }

    if (!broken) {
      hi *= whole_shape.dimensions(r);
    } else {
      lo *= shard_shape.dimensions(r);
    }
  }

  int64 plane = lo * elem_size;
  std::vector<char*> src_bufs;
  src_bufs.reserve(num_valid_shards());
  for (int64 i = 0; i < num_shards(); ++i) {
    if (gpu_present_[i]) {
      auto gpu_buffer = gpu_buffers_.at(i).get();
      auto literal = gpu_buffer->ToLiteral().ConsumeValueOrDie();
      char* src = (char*)(literal->untyped_data());
      src_bufs.emplace_back(src);
    }
  }

  std::shared_ptr<Literal> literal_ptr = std::make_shared<Literal>(
      on_host_shape_);
  char* dst = (char*)literal_ptr->untyped_data();
  for (int64 h = 0; h < hi; ++h) {
    for (int64 i = 0; i < num_valid_shards(); ++i) {
      std::memcpy(dst, src_bufs[i], plane);
      src_bufs[i] = src_bufs[i] + plane;
      dst += plane;
    }
  }
  return std::move(literal_ptr);
}

PjRtBuffer::ScopedHold DAPPLEBuffer::GetBufferWithHold(int64 id, 
                           PjRtBuffer::ScopedHold::Type type) {
  CHECK(id < int64(gpu_buffers_.size()));
  auto& buf = gpu_buffers_.at(id);
  return buf->GetBufferWithHold(type);
}

void DAPPLEBuffer::AcquireHoldLocked(PjRtBuffer::ScopedHold* hold) {
  hold->Acquire(GetBufferForHoldLocked(hold->type()));
}

StatusOr<std::shared_ptr<TrackedDeviceBuffer>>
DAPPLEBuffer::GetBufferForHoldLocked(PjRtBuffer::ScopedHold::Type type) {
  CHECK(0);
  auto pjrt_buffer = gpu_buffers_.at(0).get();
  return pjrt_buffer->GetBufferForHoldLocked(type);
}

void DAPPLEBuffer::Delete() {
  if (in_cpu_) {
    if (internal_raw_buffer_) {
      internal_raw_buffer_.reset();
    }
  }
  DeletePjRtBuffers();
}

const int64 DAPPLEBuffer::num_shards() const { return gpu_buffers_.size(); }

} // namespace xla
