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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_SLICE_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_SLICE_UTILS_H_

#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"
#include "tensorflow/compiler/xla/shape.h"

#include <iostream>
#include <vector>

namespace xla {

using ::tensorflow::int64;

class SliceUtils {
 public:
  static void MultiDimensionSliceCopy(
      std::vector<int>& dimensions, std::vector<int64>& strides,
      std::vector<int>& num_splits, std::vector<int64>& split_ids,
      std::vector<int64>& dim_base, int idx, int64 elem_bytes,
      int64 src_start, const void* src, int64* dst_offset, void* dst);

  static std::vector<std::pair<int64, int64>> GetSliceStartOffsetOnSrc(
      const Shape& full_shape, const DistSpec& dist_spec,
      const std::vector<int64>& split_ids);

  static void SliceCopyOnHost(
      const Shape& full_shape, const DistSpec& dist_spec,
      const std::vector<int64>& split_ids, const void* src, void* dst);

 private:
  static void SliceVisitor(
      const Shape& full_shape, const DistSpec& dist_spec,
      const std::vector<int64>& split_ids, const void* src, void* dst,
      std::function<void(const void*, void*, int64, int64, int64)> function);

  static void GetExpandedDimsInfo(
      const Shape& full_shape, const DistSpec& dist_spec,
      const std::vector<int64>& split_ids, std::vector<int>* reinterpret_dims,
      std::vector<int64>* reindex_stride_vecs, std::vector<int>* reindex_num_splits_vecs,
      std::vector<int64>* reindex_split_ids, std::vector<int64>* dim_base);

  static void MultiDimensionSliceDFSVisitor(
      const std::vector<int>& dimensions, const std::vector<int64>& strides,
      const std::vector<int>& num_splits, const std::vector<int64>& split_ids,
      const std::vector<int64>& dim_base, const int idx, const int64 elem_bytes,
      const int64 src_start, const void* src, void* dst, int64* dst_offset,
      std::function<void(const void*, void*, int64, int64, int64)> function);

};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_SLICE_UTILS_H_
