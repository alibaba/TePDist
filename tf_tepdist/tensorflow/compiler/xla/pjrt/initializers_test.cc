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

#include "tensorflow/compiler/xla/pjrt/initializers.h"

#include "tensorflow/compiler/xla/pjrt/slice_utils.h"
#include "tensorflow/compiler/xla/rng_distribution_config.pb.h"
#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"
#include "tensorflow/compiler/xla/test.h"

namespace dist_rng {

using namespace tensorflow;
using namespace tensorflow::random;
using namespace xla;

class InitializersTest : public ::testing::Test {
 protected:
  DistributionConfig MakeDistributionConfig() {
    DistributionConfig config;
    config.set_seed(123123);
    config.set_seed2(46);
    config.set_distribution("truncated_normal");
    config.set_mean(0.0);
    config.set_stddev(0.01);
    return config;
  }

  DistributionConfig MakeDistributionConfig2() {
    DistributionConfig config;
    config.set_seed(123123);
    config.set_seed2(123123);
    config.set_distribution("truncated_normal");
    config.set_mean(0.0);
    config.set_stddev(0.02);
    return config;
  }
};

TEST_F(InitializersTest, ReplicationTest)	{
  DistributionConfig proto = MakeDistributionConfig();
  Shape full_shape = ShapeUtil::MakeShape(xla::F32, {4, 16, 8});
  int primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(xla::F32);
  int num_full_elem = ShapeUtil::ByteSizeOf(full_shape) / primitive_size;
  std::unique_ptr<float[]> expect_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(proto, full_shape, expect_data.get());

  Shape shard_shape = ShapeUtil::MakeShape(xla::F32, {4, 16, 8});
  DistSpec dist_spec;
  std::vector<int64> split_ids = {0};
  auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
      full_shape, dist_spec, split_ids);

  std::unique_ptr<float[]> replication_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(
      proto, full_shape, shard_shape, start_offset_pairs, replication_data.get());

  for (int i = 0; i < num_full_elem; ++i) {
    EXPECT_FLOAT_EQ(*(expect_data.get() + i), *(replication_data.get() + i));
  }
}

TEST_F(InitializersTest, ShardOnTopDimTest)	{
  DistributionConfig proto = MakeDistributionConfig();
  Shape full_shape = ShapeUtil::MakeShape(xla::F32, {4, 16, 8});
  int primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(xla::F32);
  int num_full_elem = ShapeUtil::ByteSizeOf(full_shape) / primitive_size;
  std::unique_ptr<float[]> full_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(proto, full_shape, full_data.get());

  Shape shard_shape = ShapeUtil::MakeShape(xla::F32, {1, 16, 8});
  int num_shard_elem = ShapeUtil::ByteSizeOf(shard_shape) / primitive_size;
  int num_replicas = num_full_elem / num_shard_elem;

  for (int i = 0; i < num_replicas; ++i) {
    std::unique_ptr<float[]> shard_data = std::make_unique<float[]>(num_shard_elem);
    DistSpec dist_spec;
    dist_spec.AddDimDistSpec(
        512/*stride*/, 4/*stride_on_dim*/, 0/*partition_dim*/, 4/*num_splits*/, false);
    std::vector<int64> split_ids = {i};
    auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
        full_shape, dist_spec, split_ids);

    DistributedRandomInitializer::Initialize(
        proto, full_shape, shard_shape, start_offset_pairs, shard_data.get());
    for (int s = 0; s < num_shard_elem; ++s) {
      int glob_idx = i * num_shard_elem + s;
      EXPECT_FLOAT_EQ(*(full_data.get() + glob_idx), *(shard_data.get() + s));
    }
  }
}

TEST_F(InitializersTest, ShardOnInnerDimTest)	{
  DistributionConfig proto = MakeDistributionConfig();
  Shape full_shape = ShapeUtil::MakeShape(xla::F32, {4, 16, 8});
  int primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(xla::F32);
  int num_full_elem = ShapeUtil::ByteSizeOf(full_shape) / primitive_size;
  std::unique_ptr<float[]> full_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(proto, full_shape, full_data.get());

  Shape shard_shape = ShapeUtil::MakeShape(xla::F32, {4, 4, 8});
  int num_shard_elem = ShapeUtil::ByteSizeOf(shard_shape) / primitive_size;
  int num_replicas = num_full_elem / num_shard_elem;

  for (int i = 0; i < num_replicas; ++i) {
    std::unique_ptr<float[]> shard_data = std::make_unique<float[]>(num_shard_elem);
    DistSpec dist_spec;
    dist_spec.AddDimDistSpec(
        128/*stride*/, 4/*stride_on_dim*/, 1/*partition_dim*/, 4/*num_splits*/, false);
    std::vector<int64> split_ids = {i};
    auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
        full_shape, dist_spec, split_ids);

    DistributedRandomInitializer::Initialize(
        proto, full_shape, shard_shape, start_offset_pairs, shard_data.get());
    std::unique_ptr<float[]> shard_copy = std::make_unique<float[]>(num_shard_elem);

    SliceUtils::SliceCopyOnHost(
        full_shape, dist_spec, split_ids, full_data.get(), shard_copy.get());
    for (int s = 0; s < num_shard_elem; ++s) {
      EXPECT_FLOAT_EQ(*(shard_copy.get() + s), *(shard_data.get() + s));
    }
  }
}

TEST_F(InitializersTest, SizeLTGroupTest)	{
  DistributionConfig proto = MakeDistributionConfig();
  Shape full_shape = ShapeUtil::MakeShape(xla::F32, {4, 16, 8});
  int primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(xla::F32);
  int num_full_elem = ShapeUtil::ByteSizeOf(full_shape) / primitive_size;
  std::unique_ptr<float[]> full_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(proto, full_shape, full_data.get());

  Shape shard_shape = ShapeUtil::MakeShape(xla::F32, {4, 16, 2});
  int num_shard_elem = ShapeUtil::ByteSizeOf(shard_shape) / primitive_size;
  int num_replicas = num_full_elem / num_shard_elem;

  for (int i = 0; i < num_replicas; ++i) {
    std::unique_ptr<float[]> shard_data = std::make_unique<float[]>(num_shard_elem);
    DistSpec dist_spec;
    dist_spec.AddDimDistSpec(
        8/*stride*/, 8/*stride_on_dim*/, 2/*partition_dim*/, 4/*num_splits*/, false);
    std::vector<int64> split_ids = {i};
    auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
        full_shape, dist_spec, split_ids);

    DistributedRandomInitializer::Initialize(
        proto, full_shape, shard_shape, start_offset_pairs, shard_data.get());

    std::unique_ptr<float[]> shard_copy = std::make_unique<float[]>(num_shard_elem);
    SliceUtils::SliceCopyOnHost(
        full_shape, dist_spec, split_ids, full_data.get(), shard_copy.get());
 
    for (int s = 0; s < num_shard_elem; ++s) {
      EXPECT_FLOAT_EQ(*(shard_copy.get() + s), *(shard_data.get() + s));
    }
  }
}

TEST_F(InitializersTest, PrimeSizeTest)	{
  DistributionConfig proto = MakeDistributionConfig();
  Shape full_shape = ShapeUtil::MakeShape(xla::F32, {4, 16, 9});
  int primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(xla::F32);
  int num_full_elem = ShapeUtil::ByteSizeOf(full_shape) / primitive_size;
  std::unique_ptr<float[]> full_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(proto, full_shape, full_data.get());

  Shape shard_shape = ShapeUtil::MakeShape(xla::F32, {4, 16, 3});
  int num_shard_elem = ShapeUtil::ByteSizeOf(shard_shape) / primitive_size;
  int num_replicas = num_full_elem / num_shard_elem;

  for (int i = 0; i < num_replicas; ++i) {
    DistSpec dist_spec;
    dist_spec.AddDimDistSpec(
        9/*stride*/, 9/*stride_on_dim*/, 2/*partition_dim*/, 3/*num_splits*/, false);
    std::vector<int64> split_ids = {i};
    auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
        full_shape, dist_spec, split_ids);

    std::unique_ptr<float[]> shard_data = std::make_unique<float[]>(num_shard_elem);
    DistributedRandomInitializer::Initialize(
        proto, full_shape, shard_shape, start_offset_pairs, shard_data.get());

    std::unique_ptr<float[]> shard_copy = std::make_unique<float[]>(num_shard_elem);
    SliceUtils::SliceCopyOnHost(
        full_shape, dist_spec, split_ids, full_data.get(), shard_copy.get());

    for (int s = 0; s < num_shard_elem; ++s) {
      EXPECT_FLOAT_EQ(*(shard_copy.get() + s), *(shard_data.get() + s));
    }
  }
}


TEST_F(InitializersTest, MultithreadTest)	{
  DistributionConfig proto = MakeDistributionConfig2();
  Shape full_shape = ShapeUtil::MakeShape(xla::F32, {2, 1024});
  int primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(xla::F32);
  int num_full_elem = ShapeUtil::ByteSizeOf(full_shape) / primitive_size;
  std::unique_ptr<float[]> full_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(proto, full_shape, full_data.get());

  Shape shard_shape = ShapeUtil::MakeShape(xla::F32, {2, 1024});
  int num_shard_elem = ShapeUtil::ByteSizeOf(shard_shape) / primitive_size;
  int num_replicas = num_full_elem / num_shard_elem;

  for (int i = 0; i < num_replicas; ++i) {
    std::unique_ptr<float[]> shard_data = std::make_unique<float[]>(num_shard_elem);
    DistSpec dist_spec;
    dist_spec.AddDimDistSpec(
        0/*stride*/, 0/*stride_on_dim*/, -1/*partition_dim*/, 2/*num_splits*/, false);
    std::vector<int64> split_ids = {i};
    auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
        full_shape, dist_spec, split_ids);

    DistributedRandomInitializer::Initialize(
        proto, full_shape, shard_shape, start_offset_pairs, shard_data.get());

    std::unique_ptr<float[]> shard_copy = std::make_unique<float[]>(num_shard_elem);
    SliceUtils::SliceCopyOnHost(
        full_shape, dist_spec, split_ids, full_data.get(), shard_copy.get());

    for (int s = 0; s < num_shard_elem; ++s) {
      EXPECT_FLOAT_EQ(*(shard_copy.get() + s), *(shard_data.get() + s));
    }
  }
}

TEST_F(InitializersTest, MultithreadAndPrimeSizeTest)	{
  DistributionConfig proto = MakeDistributionConfig2();
  Shape full_shape = ShapeUtil::MakeShape(xla::F32, {2048, 9});
  int primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(xla::F32);
  int num_full_elem = ShapeUtil::ByteSizeOf(full_shape) / primitive_size;
  std::unique_ptr<float[]> full_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(proto, full_shape, full_data.get());

  Shape shard_shape = ShapeUtil::MakeShape(xla::F32, {2048, 3});
  int num_shard_elem = ShapeUtil::ByteSizeOf(shard_shape) / primitive_size;
  int num_replicas = num_full_elem / num_shard_elem;

  for (int i = 0; i < num_replicas; ++i) {
    std::unique_ptr<float[]> shard_data = std::make_unique<float[]>(num_shard_elem);
    DistSpec dist_spec;
    dist_spec.AddDimDistSpec(
        9/*stride*/, 9/*stride_on_dim*/, 1/*partition_dim*/, 3/*num_splits*/, false);
    std::vector<int64> split_ids = {i};
    auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
        full_shape, dist_spec, split_ids);

    DistributedRandomInitializer::Initialize(
        proto, full_shape, shard_shape, start_offset_pairs, shard_data.get());

    std::unique_ptr<float[]> shard_copy = std::make_unique<float[]>(num_shard_elem);
    SliceUtils::SliceCopyOnHost(
        full_shape, dist_spec, split_ids, full_data.get(), shard_copy.get());

    for (int s = 0; s < num_shard_elem; ++s) {
      EXPECT_FLOAT_EQ(*(shard_copy.get() + s), *(shard_data.get() + s));
    }
  }
}

TEST_F(InitializersTest, MultiDimensionAndPrimeSizeTest)	{
  DistributionConfig proto = MakeDistributionConfig2();
  Shape full_shape = ShapeUtil::MakeShape(xla::F32, {2048, 9});
  int primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(xla::F32);
  int num_full_elem = ShapeUtil::ByteSizeOf(full_shape) / primitive_size;
  std::unique_ptr<float[]> full_data = std::make_unique<float[]>(num_full_elem);
  DistributedRandomInitializer::Initialize(proto, full_shape, full_data.get());

  Shape shard_shape = ShapeUtil::MakeShape(xla::F32, {1024, 3});
  int num_shard_elem = ShapeUtil::ByteSizeOf(shard_shape) / primitive_size;
  int num_replicas = num_full_elem / num_shard_elem;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::unique_ptr<float[]> shard_data = std::make_unique<float[]>(num_shard_elem);
      DistSpec dist_spec;
      dist_spec.AddDimDistSpec(
          18432/*stride*/, 2048/*stride_on_dim*/, 0/*partition_dim*/, 2/*num_splits*/, false);
      dist_spec.AddDimDistSpec(
          9/*stride*/, 9/*stride_on_dim*/, 1/*partition_dim*/, 3/*num_splits*/, false);

      std::vector<int64> split_ids = {i, j};
      auto start_offset_pairs = SliceUtils::GetSliceStartOffsetOnSrc(
          full_shape, dist_spec, split_ids);

      DistributedRandomInitializer::Initialize(
          proto, full_shape, shard_shape, start_offset_pairs, shard_data.get());

      std::unique_ptr<float[]> shard_copy = std::make_unique<float[]>(num_shard_elem);
      SliceUtils::SliceCopyOnHost(
          full_shape, dist_spec, split_ids, full_data.get(), shard_copy.get());

      for (int s = 0; s < num_shard_elem; ++s) {
        EXPECT_FLOAT_EQ(*(shard_copy.get() + s), *(shard_data.get() + s));
      }
    }
  }
}
} // namespace dist_rng