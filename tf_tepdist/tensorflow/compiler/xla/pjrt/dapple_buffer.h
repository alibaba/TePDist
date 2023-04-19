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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DAPPLE_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DAPPLE_BUFFER_H_

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/pjrt/device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class DAPPLEBuffer {
 public:
  static DAPPLEBuffer* CreateDAPPLEBuffer(
                            Device* cpu_device, const void* raw,
                            const Shape& on_host_shape,
                            const Shape& on_device_shape,
                            int device_count);

  static DAPPLEBuffer* CreateDAPPLEBuffer(
                            Device* cpu_device, const void* raw,
                            const Shape& on_host_shape,
                            const Shape& on_device_shape,
                            std::shared_ptr<void> py_buf_ref,
                            int device_count);
 
  static DAPPLEBuffer* CreateDAPPLEBuffer(
                            Device* cpu_device,
                            const Shape& on_host_shape,
                            const Shape& on_device_shape,
                            int device_count,
                            bool alloc_raw_internal = false);

  static DAPPLEBuffer* CreateDAPPLEBuffer(
                            Device* cpu_device,
                            const Shape& on_host_shape,
                            const Shape& on_device_shape,
                            std::unique_ptr<PjRtBuffer> gpu_buffer,
                            int device_count, int device_id = 0);
 
  static std::unique_ptr<DAPPLEBuffer> CreateDAPPLEBufferUnique(
                            Device* cpu_device, const void* raw,
                            const Shape& on_host_shape,
                            const Shape& on_device_shape,
                            std::shared_ptr<void> py_buf_ref,
                            int device_count);
 
  static std::unique_ptr<DAPPLEBuffer> CreateDAPPLEBufferUnique(
                            Device* cpu_device,
                            const Shape& on_host_shape,
                            const Shape& on_device_shape,
                            int device_count,
                            bool alloc_raw_internal = false);

  static std::unique_ptr<DAPPLEBuffer> CreateDAPPLEBufferUnique(
                            Device* cpu_device,
                            const Shape& on_host_shape,
                            const Shape& on_device_shape,
                            std::unique_ptr<PjRtBuffer> gpu_buffer,
                            int device_count, int device_id = 0);
 
  static StatusOr<std::unique_ptr<DAPPLEBuffer>> CreateForPlaceholder(
      const Shape& shape, int device_count);

  static StatusOr<std::unique_ptr<DAPPLEBuffer>> FromHostBuffer(
      const void* data, const Shape& shape, int device_count);

 private: 
  explicit DAPPLEBuffer(Device* cpu_device, const void* raw,
                        const Shape& on_host_shape,
                        const Shape& on_device_shape,
                        int device_count);
  
  explicit DAPPLEBuffer(Device* cpu_device, const void* raw,
                        const Shape& on_host_shape,
                        const Shape& on_device_shape,
                        std::shared_ptr<void> py_buf_ref,
                        int device_count);
 
  explicit DAPPLEBuffer(Device* cpu_device,
                        const Shape& on_host_shape,
                        const Shape& on_device_shape,
                        int device_count,
                        bool alloc_raw_internal = false);

  explicit DAPPLEBuffer(Device* cpu_device,
                        const Shape& on_host_shape,
                        const Shape& on_device_shape,
                        std::unique_ptr<PjRtBuffer> gpu_buffer,
                        int device_count, int device_id = 0);
 
 public:
  ~DAPPLEBuffer();

  // Moveable, but not copyable.
  DAPPLEBuffer& operator=(const DAPPLEBuffer&) = delete;
  DAPPLEBuffer(const DAPPLEBuffer&) = delete;

  DAPPLEBuffer(DAPPLEBuffer&& d_buf);
  DAPPLEBuffer& operator=(DAPPLEBuffer&& d_buf);

  // Python interface
  const Shape& on_host_shape() const { return on_host_shape_; }
  const Shape& on_device_shape() const { return on_device_shape_; }
  void update_host_shape(const Shape& shape) { on_host_shape_ = shape; }
  void update_device_shape(const Shape& shape) { on_device_shape_ = shape; }
  void set_sharded() { sharded_ = true; }
  Device* device() const { return virtual_device_; }

  StatusOr<std::shared_ptr<Literal>> ToLiteral();

  PjRtBuffer::ScopedHold GetBufferWithHold(
      int64 id, PjRtBuffer::ScopedHold::Type type);

  StatusOr<std::shared_ptr<TrackedDeviceBuffer>>
      GetBufferForHoldLocked(PjRtBuffer::ScopedHold::Type type);
  void AcquireHoldLocked(PjRtBuffer::ScopedHold* hold);

  void DeletePjRtBuffers();
  void Delete();

  const bool in_cpu() const { return in_cpu_; }
  const bool in_gpu(int dev_id) const;
  void set_on_cpu() { in_cpu_ = true; }
  void set_gpu_buffer(std::unique_ptr<PjRtBuffer> buf, int dev_id);

  const int64 num_valid_shards() const;
  const int64 num_shards() const;

  const bool sharded() const { return sharded_; }
  PjRtBuffer* first_valid_shard();

  PjRtBuffer* gpu_buffer();
  PjRtBuffer* gpu_buffer(int device_id);
  std::unique_ptr<PjRtBuffer> steal_gpu_buffer(int device_id);
  void add_gpu_shard(std::unique_ptr<PjRtBuffer> shard, int device_id);

  // Accumulate new buffer's data to original's one. The original gpu_buffers of
  // 'global_dev_id' must already exist.
  void accumulate_gpu_buffer(
      std::unique_ptr<PjRtBuffer> buf_to_accumulate, int global_dev_id);

  // Perform averaging the buffer's content of specific gpu_buffer. The original
  // gpu_buffers of 'dev_id' must already exist.
  // This function must be called after 'accumulate_gpu_buffer' function.
  void average_gpu_buffer(int dev_id, int local_ga_iterations);

  void LossAverageAcrossDPInst();

  void* mutable_raw();
  const void* raw() const { return raw_; }

 private:
  // Copy cached value of the buffer on the host to `device_buffer_`.
  //
  // Applicable scene: when the cached value is already updated while the
  // `device_buffer_` is stale.
  void CopyHostCacheToDevice(int dev_id);

  friend class DAPPLEBufferUtils;
  Device* virtual_device_;
  int device_count_;
  const void* raw_ ;
  Shape on_host_shape_;
  Shape on_device_shape_;
  std::shared_ptr<void> py_buf_ref_;
  std::unique_ptr<char[]> internal_raw_buffer_;
  bool in_cpu_ = false;
  bool sharded_ = false;
  tensorflow::mutex mu_;
  std::vector<std::unique_ptr<PjRtBuffer>> gpu_buffers_ TF_GUARDED_BY(mu_);
  std::vector<bool> gpu_present_ TF_GUARDED_BY(mu_);
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_DAPPLE_BUFFER_H_
