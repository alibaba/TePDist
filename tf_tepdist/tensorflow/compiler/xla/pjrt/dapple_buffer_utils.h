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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DAPPLE_BUFFER_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DAPPLE_BUFFER_UTILS_H_

#include "tensorflow/compiler/xla/pjrt/dapple_buffer.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {

class DAPPLEBufferUtils {
 public:
  static Status H2D(DAPPLEBuffer* dapple_buf, Shape shape, int replica_id,
      int num_replicas, int local_device_id, int global_device_id,
      int64 stride, PjRtClient* gpu_client, const void* raw = nullptr);

  static Status H2D(DAPPLEBuffer* dapple_buf, Shape shape,
      int local_device_id, int global_device_id,
      PjRtClient* gpu_client, const void* sharded_tensor);

  static Status H2D(DAPPLEBuffer* dapple_buf, Shape shape,
    int local_device_id, int global_device_id, const DistSpec& dist_spec,
    const std::vector<int64>& split_ids, PjRtClient* gpu_client);
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_DAPPLE_BUFFER_UTILS_H_
