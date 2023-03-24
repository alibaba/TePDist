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

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"

namespace dist_rng {

using namespace Eigen;
using namespace tensorflow;
using namespace tensorflow::random;
using namespace xla;

template <typename T>
using MatrixType = Matrix<T, 1, Dynamic>;
template <typename T>
using MapType = Map<MatrixType<T>>;
template <typename T>
using MapTypeConst = Map<const MatrixType<T>>;

const int64 max_parallelism = 32;

namespace {

template <typename T>
void ConstantInitializeWrapper(
    const DistributionConfig& proto, const Shape& shape, void* out_tensor) {
  ConstantInitializer<T> initializer(proto);
  initializer.DoInitialize(shape, (T *)out_tensor);
}

template <typename T>
void ConstantInitializeWrapper(
    const DistributionConfig& proto, const Shape& full_shape, const Shape& shard_shape,
    const std::vector<std::pair<int64, int64>>& start_offset_pairs, void* out_tensor) {
  ConstantInitializer<T> initializer(proto);
  initializer.DoInitialize(full_shape, shard_shape, start_offset_pairs, (T *)out_tensor);
}

void DoConstantInitialization(
    const DistributionConfig& proto, const Shape& shape, void* out_tensor) {
  switch (proto.dtype()) {
    case DT_FLOAT: {
      ConstantInitializeWrapper<float>(proto, shape, out_tensor);
      break;
    }

    case DT_INT64: {
      ConstantInitializeWrapper<int64>(proto, shape, out_tensor);
      break;
    }

    default : { CHECK(0 && "Unhandled DataType"); }
  }
}

void DoConstantInitialization(
    const DistributionConfig& proto, const Shape& full_shape, const Shape& shard_shape,
    const std::vector<std::pair<int64, int64>>& start_offset_pairs, void* out_tensor) {
  switch (proto.dtype()) {
    case DT_FLOAT: {
      ConstantInitializeWrapper<float>(
          proto, full_shape, shard_shape, start_offset_pairs, out_tensor);
      break;
    }

    case DT_INT64: {
      ConstantInitializeWrapper<int64>(
          proto, full_shape, shard_shape, start_offset_pairs, out_tensor);
      break;
    }

    default : { CHECK(0 && "Unhandled DataType"); }
  }
}

} // namespace

namespace functor {

/*static*/
template <class Distribution>
void FillShardPhiloxRandomTask<Distribution, true>::Run(
    PhiloxRandom base_gen, const std::vector<std::pair<int64, int64>>& start_offset_pairs,
    Distribution dist, int64 start_idx, int64 limit_idx, T* data) {
  const int kGroupSize = Distribution::kResultElementCount;
  static const int kGeneratorSkipPerOutputGroup =
      kGroupSize * kReservedSamplesPerOutput / PhiloxRandom::kResultElementCount;
  std::vector<std::pair<int64, int64>> current_shard_start_offset_pairs;
  // 1. Build start_offset_pairs for current shard
  int64 lo = start_idx, hi = limit_idx, src_skip_size = 0, dst_size = 0;
  for (int i = 0; i < start_offset_pairs.size() && lo < hi; ++i) {
    int64 offset_size = start_offset_pairs[i].second;
    if (lo >= src_skip_size + offset_size) {
      src_skip_size += offset_size;
    } else if (dst_size < limit_idx - start_idx) {
      offset_size -= (lo - src_skip_size);
      if (lo + offset_size >= hi) offset_size = hi - lo;
      int64 start_on_src = lo - src_skip_size + start_offset_pairs[i].first;
      current_shard_start_offset_pairs.push_back(std::make_pair(start_on_src, offset_size));
      src_skip_size += offset_size;
      lo = src_skip_size;
      dst_size += offset_size;
    }
  }

  data += start_idx;
  // 2. Generate samples for current shard
  for (std::pair<int64, int64>& p : current_shard_start_offset_pairs) {
    int64 src_start = p.first;
    int64 offset_size = p.second;
    int64 start_group_idx = src_start / kGroupSize;
    int64 idx_in_group = src_start % kGroupSize;
    int64 curr_group_idx = start_group_idx, curr_idx = 0;
    while (curr_idx < offset_size) {
      int64 sample_size = std::min(kGroupSize - idx_in_group, offset_size - curr_idx);
      PhiloxRandom gen = base_gen;
      gen.Skip(curr_group_idx * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);
      auto samples = dist(&single_samples);
      std::copy(&samples[0] + idx_in_group, &samples[0] + idx_in_group + sample_size, data);
      idx_in_group = 0;
      curr_idx += sample_size;
      data += sample_size;
      ++curr_group_idx;
    }
  }
}

/*static*/
template <class Distribution>
void FillShardPhiloxRandomTask<Distribution, false>::Run(
    PhiloxRandom base_gen, const std::vector<std::pair<int64, int64>>& start_offset_pairs,
    Distribution dist, int64 start_idx, int64 limit_idx, T* data) {
  const int kGroupSize = Distribution::kResultElementCount;
  std::vector<std::pair<int64, int64>> current_shard_start_offset_pairs;
  // 1. Build start_offset_pairs for current shard
  int64 lo = start_idx, hi = limit_idx, skip_size = 0, dst_size = 0;
  for (int i = 0; i < start_offset_pairs.size() && lo < hi; ++i) {
    int64 offset_size = start_offset_pairs[i].second;
    if (lo >= skip_size + offset_size) skip_size += offset_size;
    else if (dst_size < limit_idx - start_idx) {
      offset_size -= (lo - skip_size);
      if (lo + offset_size >= hi) offset_size = hi - lo;
      int64 start_on_src = lo - skip_size + start_offset_pairs[i].first;
      current_shard_start_offset_pairs.push_back(std::make_pair(start_on_src, offset_size));
      skip_size += offset_size;
      lo = skip_size;
      dst_size += offset_size;
    }
  }

  data += start_idx;
  // 2. Generate samples for current shard
  for (std::pair<int64, int64>& p : current_shard_start_offset_pairs) {
    int64 src_start = p.first;
    int64 offset_size = p.second;
    int64 start_group_idx = src_start / kGroupSize;
    int64 idx_in_group = src_start % kGroupSize;
    int64 curr_group_idx = start_group_idx, curr_idx = 0;
    while (curr_idx < offset_size) {
      int64 sample_size = std::min(kGroupSize - idx_in_group, offset_size - curr_idx);
      PhiloxRandom gen = base_gen;
      gen.Skip(curr_group_idx);
      auto samples = dist(&gen);
      std::copy(&samples[0] + idx_in_group, &samples[0] + idx_in_group + sample_size, data);
      idx_in_group = 0;
      curr_idx += sample_size;
      data += sample_size;
      ++curr_group_idx;
    }
  }
}

template <class Distribution>
void FillShardPhiloxRandom<Distribution>::operator()(
    random::PhiloxRandom gen, Distribution dist,
    const std::vector<std::pair<int64, int64>>& start_offset_pairs,
    int64 slice_elem_count, T* data) {
  const int kGroupSize = Distribution::kResultElementCount;
  const int kGroupCost =
      tensorflow::random::PhiloxRandom::kResultElementCount *
      (tensorflow::random::PhiloxRandom::kElementCost + Distribution::kElementCost);
  
  
  int num_threads = std::min(max_parallelism, slice_elem_count);

  VLOG(1) << "slice_elem_count = " << slice_elem_count
          << ", num_threads = " << num_threads;

  tensorflow::thread::ThreadPool thread_pool(
      tensorflow::Env::Default(), "py_xla_sampling", num_threads);

  tensorflow::Shard(num_threads, &thread_pool, slice_elem_count, kGroupCost,
      [&gen, data, dist, &start_offset_pairs] (int64 start_idx, int64 limit_idx) {
        FillShardPhiloxRandomTask<
            Distribution,
            Distribution::kVariableSamplesPerOutput>::Run(
                gen, start_offset_pairs, dist, start_idx, limit_idx, data);
      });
}

} // namespace functor

void GuardedPhiloxRandom::Init(int64 seed, int64 seed2) {
  CHECK(!initialized_);
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = tensorflow::random::New64();
    seed2 = tensorflow::random::New64();
  }
  tensorflow::mutex_lock lock(mu_);
  generator_ = tensorflow::random::PhiloxRandom(seed, seed2);
  initialized_ = true;
}

void GuardedPhiloxRandom::Init(tensorflow::random::PhiloxRandom::ResultType counter,
                               tensorflow::random::PhiloxRandom::Key key) {
  CHECK(!initialized_);
  tensorflow::mutex_lock lock(mu_);
  generator_ = tensorflow::random::PhiloxRandom(counter, key);
  initialized_ = true;
}

tensorflow::random::PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64 samples) {
  CHECK(initialized_);
  tensorflow::mutex_lock lock(mu_);
  auto local = generator_;
  generator_.Skip(samples);
  return local;
}

template <class Distribution>
SamplingBasedInitializer<Distribution>::SamplingBasedInitializer(
    const DistributionConfig& proto) : rng_config_proto_(proto) {
  generator_.Init(rng_config_proto_.seed(), rng_config_proto_.seed2());
}

template <class Distribution>
void SamplingBasedInitializer<Distribution>::DoInitialize(
    const Shape& shape, T* out_tensor) {
  CHECK(out_tensor);
  const int64 size = \
      ShapeUtil::ByteSizeOf(shape) / ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  DoPhiloxRandomSampling(size, out_tensor);
  PostProcess(size, out_tensor);
}

template <class Distribution>
void SamplingBasedInitializer<Distribution>::DoInitialize(
    const Shape& full_shape, const Shape& shard_shape,
    const std::vector<std::pair<int64, int64>>& start_offset_pairs, T* out_tensor) {
  CHECK(out_tensor);
  int64 elem_size = ShapeUtil::ByteSizeOfPrimitiveType(full_shape.element_type());
  const int64 total_elem_count = ShapeUtil::ByteSizeOf(full_shape) / elem_size;
  const int64 slice_elem_count = ShapeUtil::ByteSizeOf(shard_shape) / elem_size;

  DoPhiloxRandomSampling(total_elem_count, slice_elem_count, start_offset_pairs, out_tensor);
  PostProcess(slice_elem_count, out_tensor);
}

template <class Distribution>
void SamplingBasedInitializer<Distribution>::DoPhiloxRandomSampling(
    const int64 size, T* out_tensor) {
  functor::FillPhiloxRandom<Distribution>()(
      // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
      // it just here.
      generator_.ReserveRandomOutputs(size, 256),
      out_tensor, size, Distribution());
}

template <class Distribution>
void SamplingBasedInitializer<Distribution>::DoPhiloxRandomSampling(
    const int64 total_elem_count, const int64 slice_elem_count,
    const std::vector<std::pair<int64, int64>>& start_offset_pairs, T* out_tensor) {
  functor::FillShardPhiloxRandom<Distribution>()(
      // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
      // it just here.
      generator_.ReserveRandomOutputs(total_elem_count, 256),
      Distribution(), start_offset_pairs, slice_elem_count, out_tensor);
}

template <class Distribution>
RandomUniformInitializer<Distribution>::RandomUniformInitializer(
    const DistributionConfig& proto) : SamplingBasedInitializer<Distribution>(proto) {}

template <class Distribution>
RandomNormalInitializer<Distribution>::RandomNormalInitializer(
    const DistributionConfig& proto) : SamplingBasedInitializer<Distribution>(proto) {}

template <class Distribution>
TruncatedNormalInitializer<Distribution>::TruncatedNormalInitializer(
    const DistributionConfig& proto) : SamplingBasedInitializer<Distribution>(proto) {}

template <class Distribution>
void RandomUniformInitializer<Distribution>::PostProcess(
    const int64 size, T* out_tensor) {
  MapType<T> flat_out(out_tensor, size);
  float min_v_f = SamplingBasedInitializer<Distribution>::rng_config_proto_.minval();
  float max_v_f = SamplingBasedInitializer<Distribution>::rng_config_proto_.maxval();

  auto min_v = MatrixType<T>::Constant(1, size, min_v_f);
  auto max_v = MatrixType<T>::Constant(1, size, max_v_f);
  flat_out = (max_v - min_v) * flat_out + min_v;
}

template <class Distribution>
void RandomNormalInitializer<Distribution>::PostProcess(
    const int64 size, T* out_tensor) {
  MapType<T> flat_out(out_tensor, size);
  float mean_f = SamplingBasedInitializer<Distribution>::rng_config_proto_.mean();
  float stddev_f = SamplingBasedInitializer<Distribution>::rng_config_proto_.stddev();

  auto mean = MatrixType<T>::Constant(1, size, mean_f);
  auto stddev = MatrixType<T>::Constant(1, size, stddev_f);
  flat_out = stddev * flat_out + mean;
}
 
template <class Distribution>
void TruncatedNormalInitializer<Distribution>::PostProcess(
    const int64 size, T* out_tensor) {
  MapType<T> flat_out(out_tensor, size);
  float mean_f = SamplingBasedInitializer<Distribution>::rng_config_proto_.mean();
  float stddev_f = SamplingBasedInitializer<Distribution>::rng_config_proto_.stddev();

  auto mean = MatrixType<T>::Constant(1, size, mean_f);
  auto stddev = MatrixType<T>::Constant(1, size, stddev_f);
  flat_out = stddev * flat_out + mean;
}

template <typename T>
void ConstantInitializer<T>::DoInitialize(const Shape& shape, T* out_tensor) {
  CHECK(out_tensor);
  float const_val = rng_config_proto_.const_val();
  const int64 size = \
      ShapeUtil::ByteSizeOf(shape) / ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  MapType<T> flat_out(out_tensor, size);
  auto const_val_flat = MatrixType<T>::Constant(1, size, const_val);
  flat_out = const_val_flat;
}

template <typename T>
void ConstantInitializer<T>::DoInitialize(
    const Shape& full_shape, const Shape& shard_shape,
    const std::vector<std::pair<int64, int64>>& start_offset_pairs, T* out_tensor) {
  CHECK(out_tensor);
  float const_val = rng_config_proto_.const_val();
  int elem_size = ShapeUtil::ByteSizeOfPrimitiveType(full_shape.element_type());
  const int64 slice_elem_count = ShapeUtil::ByteSizeOf(shard_shape) / elem_size;
  MapType<T> flat_out(out_tensor, slice_elem_count);
  auto const_val_flat = MatrixType<T>::Constant(1, slice_elem_count, const_val);
  flat_out = const_val_flat;
}

void DistributedRandomInitializer::Initialize(
    const DistributionConfig& proto, const Shape& shape,
    void* out_tensor) {
  auto distribution = proto.distribution();
  VLOG(2) << "distribution = " << distribution;
  if (distribution == "random_uniform") {
    RandomUniformInitializer<
        tensorflow::random::UniformDistribution<
            tensorflow::random::PhiloxRandom, float>> initializer(proto);
    initializer.DoInitialize(shape, (float *)out_tensor);
  } else if (distribution == "random_normal"){
    RandomNormalInitializer<
        tensorflow::random::NormalDistribution<            \
            tensorflow::random::PhiloxRandom, float>> initializer(proto);
    initializer.DoInitialize(shape, (float *)out_tensor);
  } else if (distribution == "truncated_normal" ){
    TruncatedNormalInitializer<
        tensorflow::random::TruncatedNormalDistribution<            \
            tensorflow::random::SingleSampleAdapter<
                tensorflow::random::PhiloxRandom>, float>> initializer(proto);
    initializer.DoInitialize(shape, (float *)out_tensor);
  } else if (distribution == "Constant") {
    DoConstantInitialization(proto, shape, out_tensor);
  } else {
    CHECK(0);
  }
}

void DistributedRandomInitializer::Initialize(
    const DistributionConfig& proto, const Shape& full_shape, const Shape& shard_shape,
    const std::vector<std::pair<int64, int64>>& start_offset_pairs, void* out_tensor) {
  auto distribution = proto.distribution();
  VLOG(2) << "distribution = " << distribution;
  if (distribution == "random_uniform") {
    RandomUniformInitializer<
        tensorflow::random::UniformDistribution<
            tensorflow::random::PhiloxRandom, float>> initializer(proto);
    initializer.DoInitialize(full_shape, shard_shape, start_offset_pairs, (float *)out_tensor);
  } else if (distribution == "random_normal"){
    RandomNormalInitializer<
        tensorflow::random::NormalDistribution<            \
            tensorflow::random::PhiloxRandom, float>> initializer(proto);
    initializer.DoInitialize(full_shape, shard_shape, start_offset_pairs, (float *)out_tensor);
  } else if (distribution == "truncated_normal" ){
    TruncatedNormalInitializer<
        tensorflow::random::TruncatedNormalDistribution<            \
            tensorflow::random::SingleSampleAdapter<
                tensorflow::random::PhiloxRandom>, float>> initializer(proto);
    initializer.DoInitialize(full_shape, shard_shape, start_offset_pairs, (float *)out_tensor);
  } else if (distribution == "Constant") {
    DoConstantInitialization(proto, full_shape, shard_shape, start_offset_pairs, out_tensor);
  } else {
    CHECK(0);
  }
}

#define REGISTER(TYPE)                                                                        \
   template struct functor::FillShardPhiloxRandom<                                                  \
      tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, TYPE>>;       \
  template struct functor::FillShardPhiloxRandom<                                                  \
      tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, TYPE>>;        \
  template struct functor::FillShardPhiloxRandom<                                                  \
      tensorflow::random::TruncatedNormalDistribution<                                        \
          tensorflow::random::SingleSampleAdapter<tensorflow::random::PhiloxRandom>, TYPE>>;  \
  
REGISTER(float);
#undef REGISTER

} // namespace xla
