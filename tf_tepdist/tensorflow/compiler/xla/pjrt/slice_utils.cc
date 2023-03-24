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

#include "tensorflow/compiler/xla/pjrt/slice_utils.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/default/logging.h"

#include <cstring>

namespace xla {

/*static*/
void SliceUtils::SliceVisitor(
    const Shape& full_shape, const DistSpec& dist_spec,
    const std::vector<int64>& split_ids, const void* src, void* dst,
    std::function<void(const void*, void*, int64, int64, int64)> function) {
  if (dist_spec.is_empty() || ShapeUtil::IsScalar(full_shape)) {
    function(src, dst, 0, 0, ShapeUtil::ByteSizeOf(full_shape));
    return;
  }

  // 1. Reindex original tensor when multiple splits on same dimension
  std::vector<int> reinterpret_dims;
  std::vector<int64> reindex_stride_vecs;
  std::vector<int> reindex_num_splits_vecs;
  std::vector<int64> reindex_split_ids;
  std::vector<int64> dim_base;

  GetExpandedDimsInfo(full_shape, dist_spec, split_ids,
      &reinterpret_dims, &reindex_stride_vecs, &reindex_num_splits_vecs,
      &reindex_split_ids, &dim_base);

  // 2. DFSVisit
  int64 dst_offset = 0;
  MultiDimensionSliceDFSVisitor(
      reinterpret_dims, reindex_stride_vecs, reindex_num_splits_vecs,
      reindex_split_ids, dim_base, 0/*dim_idx*/,
      ShapeUtil::ByteSizeOfPrimitiveType(full_shape.element_type()),
      0/*src_start*/, src, dst, &dst_offset, function);
}

/*static*/
void SliceUtils::SliceCopyOnHost(
    const Shape& full_shape, const DistSpec& dist_spec,
    const std::vector<int64>& split_ids, const void* src, void* dst) {
  auto copy_func = [] (
      const void* src, void* dst, int64 src_offset, int64 dst_offset, int64 size) {
    std::memcpy(dst + dst_offset, src + src_offset, size);
  };

  SliceVisitor(full_shape, dist_spec, split_ids, src, dst, copy_func);
}

/*static*/
std::vector<std::pair<int64, int64>> SliceUtils::GetSliceStartOffsetOnSrc(
    const Shape& full_shape, const DistSpec& dist_spec,
    const std::vector<int64>& split_ids) {
  std::vector<std::pair<int64, int64>> start_offset_pairs;
  int64 elem_bytes = ShapeUtil::ByteSizeOfPrimitiveType(full_shape.element_type());
  auto get_start_offset_func = [&start_offset_pairs, elem_bytes] (
      const void* src, void* dst, int64 src_offset, int64 dst_offset, int64 size) {
    start_offset_pairs.push_back(std::make_pair(src_offset / elem_bytes, size / elem_bytes));
  };

  SliceVisitor(full_shape, dist_spec, split_ids, nullptr, nullptr, get_start_offset_func);
  return start_offset_pairs;
}

/*static*/
void SliceUtils::MultiDimensionSliceCopy(
    std::vector<int>& dimensions, std::vector<int64>& strides,
    std::vector<int>& num_splits, std::vector<int64>& split_ids,
    std::vector<int64>& dim_base, int idx, int64 elem_size,
    int64 src_start, const void* src, int64* dst_offset, void* dst) {
  auto copy_func = [] (
      const void* src, void* dst, int64 src_offset, int64 dst_offset, int64 size) {
    std::memcpy(dst + dst_offset, src + src_offset, size);
  };

  MultiDimensionSliceDFSVisitor(
      dimensions, strides, num_splits, split_ids, dim_base,
      idx, elem_size, src_start, src, dst, dst_offset, copy_func);
}

/*static*/
void SliceUtils::MultiDimensionSliceDFSVisitor(
    const std::vector<int>& dimensions, const std::vector<int64>& strides,
    const std::vector<int>& num_splits, const std::vector<int64>& split_ids,
    const std::vector<int64>& dim_base, const int idx, const int64 elem_size,
    const int64 src_start, const void* src, void* dst, int64* dst_offset,
    std::function<void(const void*, void*, int64, int64, int64)> function) {
  int64 size = strides[idx] / num_splits[idx];
  for (int i = 0; i < dimensions[idx] / strides[idx]; ++i) {
    int64 stride_offset = strides[idx] * i;
    int64 offset = stride_offset + split_ids[idx] * size;
    int64 src_offset = (src_start + offset) * elem_size;
    if (idx == dimensions.size() - 1) {
      function(src, dst, src_offset, *dst_offset, size * elem_size);
      *dst_offset += size * elem_size;
    } else {
      for (int size_offset = 0; size_offset < size; ++size_offset) {
        MultiDimensionSliceDFSVisitor(
            dimensions, strides, num_splits, split_ids, dim_base,
            idx + 1, elem_size, src_start + (offset + size_offset) * dim_base[idx],
            src, dst, dst_offset, function);
      }
    }
  }
}

/*static*/
void SliceUtils::GetExpandedDimsInfo(
    const Shape& full_shape, const DistSpec& dist_spec,
    const std::vector<int64>& split_ids, std::vector<int>* reinterpret_dims,
    std::vector<int64>* reindex_stride_vecs, std::vector<int>* reindex_num_splits_vecs,
    std::vector<int64>* reindex_split_ids, std::vector<int64>* dim_base) {
  std::unordered_map<int/*origin_dim*/, 
                     std::vector<std::pair<int/*split_ordinal*/, DimDistSpec*>>> dim_spec_map;
  for (int i = 0; i < dist_spec.size(); ++i) {
    auto& dim_spec = dist_spec.get_dim_spec(i);
    dim_spec_map[dim_spec->partition_dim()].push_back(std::make_pair(i, dim_spec.get()));
  }
  
  for (int d = 0; d < full_shape.dimensions_size(); ++d) {
    int dim_val = full_shape.dimensions(d);
    if (dim_spec_map.find(d) != dim_spec_map.end()) {
      for (int s = 0; s < dim_spec_map[d].size(); ++s) {
        int split_ordinal = dim_spec_map[d][s].first;
        auto dim_spec = dim_spec_map[d][s].second;
        int stride_on_dim = dim_spec->stride_on_dim();
        int num_splits = 1;
        int split_id = 0;
        if (stride_on_dim > 0) {
          num_splits = dim_spec->num_splits();
          split_id = split_ids[split_ordinal];
        } else {
          // Set stride_on_dim=dim_val and num_splits=1 when glue mode
          stride_on_dim = dim_val;
        }
        int size_on_dim = stride_on_dim / num_splits;
        int added_dim_val = dim_val / size_on_dim;
        int new_stride = stride_on_dim / size_on_dim;
        if (s == dim_spec_map[d].size() - 1) {
          added_dim_val = dim_val;
          new_stride = stride_on_dim;
        }
        reindex_stride_vecs->push_back(new_stride);
        reindex_num_splits_vecs->push_back(num_splits);
        reindex_split_ids->push_back(split_id);
        reinterpret_dims->push_back(added_dim_val);
        dim_val /= added_dim_val;
      }
    } else {
      reindex_split_ids->push_back(0);
      reindex_stride_vecs->push_back(dim_val);
      reindex_num_splits_vecs->push_back(1);
      reinterpret_dims->push_back(dim_val);
    }
  }

  dim_base->resize(reinterpret_dims->size());
  int64 base = 1;
  for (int d = reinterpret_dims->size() - 1; d >= 0; --d) {
    (*dim_base)[d] = base;
    base *= (*reinterpret_dims)[d];
  }
}
} // namespace xla
