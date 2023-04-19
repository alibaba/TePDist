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

#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include <numeric>

namespace xla {

class SliceUtilsTest : public ::testing::Test {};

TEST_F(SliceUtilsTest, DFSMajorDimensionTest) {
  int N = 32;
  std::vector<float> full_data(N);
  std::iota(full_data.begin(), full_data.end(), 0);

  std::vector<int> dimensions = {8, N / 8};
  std::vector<int64> strides;
  std::vector<int64> split_ids;
  std::vector<int> splits_num = {2, 1};
  std::vector<int64> dim_base = {4, 1};
  int64 dst_offset = 0;

  void* data = (void *)malloc(N / 2 * sizeof(float));

  strides = {8, 4};
  split_ids = {0, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  std::vector<float> expect = {0, 1, 2, 3,
                               4, 5, 6, 7,
                               8, 9, 10, 11,
                               12, 13, 14, 15};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  strides = {8, 4};
  split_ids = {1, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  strides = {4, 4};
  split_ids = {0, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {0, 1, 2, 3,
            4, 5, 6, 7,
            16, 17, 18, 19,
            20, 21, 22, 23};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
 
  strides = {4, 4};
  split_ids = {1, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {8, 9, 10, 11,
            12, 13, 14, 15,
            24, 25, 26, 27,
            28, 29, 30, 31};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
}

TEST_F(SliceUtilsTest, DFSMinorDimensionTest) {
  int N = 32;
  std::vector<float> full_data(N);
  std::iota(full_data.begin(), full_data.end(), 0);

  std::vector<int> dimensions = {8, N / 8};
  std::vector<int64> strides;
  std::vector<int64> split_ids;
  std::vector<int> splits_num = {1, 2};
  std::vector<int64> dim_base = {4, 1};
  int64 dst_offset = 0;

  void* data = (void *)malloc(N / 2 * sizeof(float));

  strides = {8, 4};
  split_ids = {0, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  std::vector<float> expect = {0, 1,
                               4, 5,
                               8, 9,
                               12, 13,
                               16, 17,
                               20, 21,
                               24, 25,
                               28, 29};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }


  split_ids = {0, 1};
  strides = {8, 4};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);
 
  expect = {2, 3,
            6, 7,
            10, 11,
            14, 15,
            18, 19,
            22, 23,
            26, 27,
            30, 31};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  strides = {8, 2};
  split_ids = {0, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {0, 2,
            4, 6,
            8, 10,
            12, 14,
            16, 18,
            20, 22,
            24, 26,
            28, 30};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
}

TEST_F(SliceUtilsTest, DFSTwoDimensionTest) {
  int N = 32;
  std::vector<float> full_data(N);
  std::iota(full_data.begin(), full_data.end(), 0);

  std::vector<int> dimensions = {8, N / 8};
  std::vector<int64> strides;
  std::vector<int64> split_ids;
  std::vector<int> splits_num = {2, 2};
  std::vector<int64> dim_base = {4, 1};
  int64 dst_offset = 0;

  void* data = (void *)malloc(N / 4 * sizeof(float));

  strides = {8, 4};
  split_ids = {0, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  std::vector<float> expect = {0, 1,
                               4, 5,
                               8, 9,
                               12, 13};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  strides = {8, 4};
  split_ids = {0, 1};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {2, 3,
            6, 7,
            10, 11,
            14, 15};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  strides = {8, 4};
  split_ids = {1, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {16, 17,
            20, 21,
            24, 25,
            28, 29};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  strides = {8, 4};
  split_ids = {1, 1};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {18, 19,
            22, 23,
            26, 27,
            30, 31};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  strides = {4, 2};
  split_ids = {0, 0};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {0, 2,
            4, 6,
            16, 18,
            20, 22};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  strides = {4, 2};
  split_ids = {1, 1};
  dst_offset = 0;
  SliceUtils::MultiDimensionSliceCopy(
      dimensions, strides, splits_num, split_ids,
      dim_base, 0/*dim_idx*/, 4/*elem_size*/,
      0/*src_start*/, full_data.data(), &dst_offset, data);

  expect = {9, 11,
            13, 15,
            25, 27,
            29, 31};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
}

TEST_F(SliceUtilsTest, SliceCopySingleDimensionTest) {
  int N = 32;
  std::vector<float> full_data(N);
  std::iota(full_data.begin(), full_data.end(), 0);

  DistSpec dist_spec;
  dist_spec.AddDimDistSpec(32/*stride*/, 8/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  dist_spec.AddDimDistSpec(0/*stride*/, 0/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  std::vector<int64> dimensions = {8, N / 8};
  const Shape shape = ShapeUtil::MakeShape(PrimitiveType::F32, dimensions);
  std::vector<int64> split_ids;

  void* data = (void *)malloc(N / 2 * sizeof(float));

  split_ids = {0, 0};
  SliceUtils::SliceCopyOnHost(shape, dist_spec, split_ids, full_data.data(), data);
  std::vector<float> expect = {0, 1, 2, 3,
                               4, 5, 6, 7,
                               8, 9, 10, 11,
                               12, 13, 14, 15};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
  
  split_ids = {1, 0};
  SliceUtils::SliceCopyOnHost(shape, dist_spec, split_ids, full_data.data(), data);
  expect = {16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }

  DistSpec dist_spec_1;
  dist_spec_1.AddDimDistSpec(32/*stride*/, 8/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  split_ids = {0};
  SliceUtils::SliceCopyOnHost(shape, dist_spec_1, split_ids, full_data.data(), data);
  expect = {0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
            12, 13, 14, 15};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
  
  DistSpec dist_spec_2;
  dist_spec_2.AddDimDistSpec(0/*stride*/, 0/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  dist_spec_2.AddDimDistSpec(32/*stride*/, 8/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  split_ids = {0, 1};
  SliceUtils::SliceCopyOnHost(shape, dist_spec_2, split_ids, full_data.data(), data);
  expect = {16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31};

  for (int i = 0; i < N / 2; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
}

TEST_F(SliceUtilsTest, SliceCopySeparateDimensionTest) {
  int N = 32;
  std::vector<float> full_data(N);
  std::iota(full_data.begin(), full_data.end(), 0);

  DistSpec dist_spec;
  dist_spec.AddDimDistSpec(32/*stride*/, 8/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  dist_spec.AddDimDistSpec(4/*stride*/, 4/*stride_on_dim*/, 1/*partition_dim*/, 2/*num_splits*/, false);
  std::vector<int64> dimensions = {8, N / 8};
  const Shape shape = ShapeUtil::MakeShape(PrimitiveType::F32, dimensions);
  std::vector<int64> split_ids;

  void* data = (void *)malloc(N / 4 * sizeof(float));

  split_ids = {0, 0};
  SliceUtils::SliceCopyOnHost(shape, dist_spec, split_ids, full_data.data(), data);
  std::vector<float> expect = {0, 1,
                               4, 5,
                               8, 9,
                               12, 13};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
 
  DistSpec dist_spec_1;
  dist_spec_1.AddDimDistSpec(16/*stride*/, 4/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  dist_spec_1.AddDimDistSpec(2/*stride*/, 2/*stride_on_dim*/, 1/*partition_dim*/, 2/*num_splits*/, false);
 
  split_ids = {1, 1};
  SliceUtils::SliceCopyOnHost(shape, dist_spec_1, split_ids, full_data.data(), data);
  expect = {9, 11,
            13, 15,
            25, 27,
            29, 31};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
}

TEST_F(SliceUtilsTest, SliceCopyDuplicateDimensionTest) {
  int N = 32;
  std::vector<float> full_data(N);
  std::iota(full_data.begin(), full_data.end(), 0);

  DistSpec dist_spec;
  dist_spec.AddDimDistSpec(32/*stride*/, 8/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  dist_spec.AddDimDistSpec(16/*stride*/, 4/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  std::vector<int64> dimensions = {8, N / 8};
  const Shape shape = ShapeUtil::MakeShape(PrimitiveType::F32, dimensions);
  std::vector<int64> split_ids;

  void* data = (void *)malloc(N / 4 * sizeof(float));

  split_ids = {0, 0};
  SliceUtils::SliceCopyOnHost(shape, dist_spec, split_ids, full_data.data(), data);
  std::vector<float> expect = {0, 1, 2, 3,
                               4, 5, 6, 7};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
  
  split_ids = {1, 0};
  SliceUtils::SliceCopyOnHost(shape, dist_spec, split_ids, full_data.data(), data);
  expect = {16, 17, 18, 19,
            20, 21, 22, 23};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
  
  DistSpec dist_spec_1;
  dist_spec_1.AddDimDistSpec(16/*stride*/, 4/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
  dist_spec_1.AddDimDistSpec(8/*stride*/, 2/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);

  split_ids = {0, 0};
  SliceUtils::SliceCopyOnHost(shape, dist_spec_1, split_ids, full_data.data(), data);
  expect = {0, 1, 2, 3, 16, 17, 18, 19};

  for (int i = 0; i < N / 4; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), expect[i]);
  }
}

TEST_F(SliceUtilsTest, SliceCopyNoDistSpec) {
  int N = 32;
  std::vector<float> full_data(N);
  std::iota(full_data.begin(), full_data.end(), 0);

  DistSpec dist_spec;
  std::vector<int64> dimensions = {8, N / 8};
  const Shape shape = ShapeUtil::MakeShape(PrimitiveType::F32, dimensions);
  std::vector<int64> split_ids;

  void* data = (void *)malloc(N * sizeof(float));

  split_ids = {0, 0};
  SliceUtils::SliceCopyOnHost(shape, dist_spec, split_ids, full_data.data(), data);

  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(*((float *)data + i), full_data[i]);
  }
}

} // namespace xla
