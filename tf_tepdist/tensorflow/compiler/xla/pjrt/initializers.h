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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_INITIALIZERS_H
#define TENSORFLOW_COMPILER_XLA_PJRT_INITIALIZERS_H

#include "tensorflow/compiler/xla/rng_distribution_config.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/random/fill_philox_random.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"

#include <iostream>
#include <string>
#include <vector>

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace dist_rng {

using namespace tensorflow;
using namespace tensorflow::random;
using namespace xla;
namespace functor {

// The default implementation of the functor, which should never be invoked
// But we still need to provide implementation for now for the linker to work,
// since we do not support all the distributions yet.
template <class Distribution>
struct FillPhiloxRandom {
  typedef typename Distribution::ResultElementType T;
  void operator()(tensorflow::random::PhiloxRandom gen,
                  T* data, int64 size, Distribution dist);
};

template <class Distribution>
struct FillShardPhiloxRandom {
  typedef typename Distribution::ResultElementType T;
  void operator()(tensorflow::random::PhiloxRandom gen, Distribution dist,
                  const std::vector<std::pair<int64, int64>>& start_offset_pairs,
                  int64 slice_elem_count, T* data);
};

// Partial specialization for CPU to fill the entire region with randoms
// It splits the work into several tasks and run them in parallel
template <class Distribution>
void FillPhiloxRandom<Distribution>::operator()(
    tensorflow::random::PhiloxRandom gen,
    typename Distribution::ResultElementType* data,
    int64 size, Distribution dist) {
  const int kGroupSize = Distribution::kResultElementCount;

  int num_threads = 64;
  tensorflow::thread::ThreadPool thread_pool(
      tensorflow::Env::Default(), "py_xla_h2d_transfer", num_threads);
  int64 total_group_count = (size + kGroupSize - 1) / kGroupSize;

  const int kGroupCost =
      tensorflow::random::PhiloxRandom::kResultElementCount *
      (tensorflow::random::PhiloxRandom::kElementCost + Distribution::kElementCost);
  tensorflow::Shard(num_threads, &thread_pool, total_group_count,
        kGroupCost,
        [&gen, data, size, dist](int64 start_group, int64 limit_group) {
          tensorflow::functor::FillPhiloxRandomTask<
              Distribution,
              Distribution::kVariableSamplesPerOutput>::Run(
                  gen, data, size, start_group, limit_group, dist);
        });
}

// A class to fill a specified range of random groups
template <class Distribution, bool VariableSamplesPerOutput>
struct FillShardPhiloxRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Distribution>
struct FillShardPhiloxRandomTask<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  static void Run(
      PhiloxRandom base_gen, const std::vector<std::pair<int64, int64>>& start_offset_pairs,
      Distribution dist, int64 start_idx, int64 limit_idx, T* data);
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Distribution>
struct FillShardPhiloxRandomTask<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  static constexpr int64 kReservedSamplesPerOutput = 256;
  static void Run(
      PhiloxRandom base_gen, const std::vector<std::pair<int64, int64>>& start_offset_pairs,
      Distribution dist, int64 start_idx, int64 limit_idx, T* data);
};

} // namespace functor

class GuardedPhiloxRandom {
 public:
  // Must call Init to finish initialization
  GuardedPhiloxRandom() : initialized_(false) {}

  // Initialize with given seeds.
  void Init(int64 seed, int64 seed2);
  void Init(tensorflow::random::PhiloxRandom::ResultType counter,
            tensorflow::random::PhiloxRandom::Key key);

  // Reserve a certain number of 128-bit samples.
  // This function is thread safe.  The returned generator is valid for the
  // given number of samples, and can be used without a lock.
  tensorflow::random::PhiloxRandom ReserveSamples128(int64 samples);

  // Reserve a certain number of 32-bit samples.
  tensorflow::random::PhiloxRandom ReserveSamples32(int64 samples) {
    return ReserveSamples128((samples + 3) / 4);
  }

  // Reserve enough random samples in the generator for the given output count.
  tensorflow::random::PhiloxRandom ReserveRandomOutputs(int64 output_count,
                                            int multiplier) {
    int64 conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  tensorflow::mutex mu_;
  tensorflow::random::PhiloxRandom generator_ TF_GUARDED_BY(mu_);
  bool initialized_;

  TF_DISALLOW_COPY_AND_ASSIGN(GuardedPhiloxRandom);
};

template <class Distribution>
class SamplingBasedInitializer {
  typedef typename Distribution::ResultElementType T;
 public:
  explicit SamplingBasedInitializer(const DistributionConfig& proto);
  void DoInitialize(const Shape& shape, T* out_tensor);
  void DoInitialize(const Shape& full_shape, const Shape& shard_shape,
                    const std::vector<std::pair<int64, int64>>& start_offset_pairs,
                    T* out_tensor);
 protected:
  void DoPhiloxRandomSampling(const int64 size, T* out_tensor);
  void DoPhiloxRandomSampling(const int64 total_elem_count, const int64 slice_elem_count,
                              const std::vector<std::pair<int64, int64>>& start_offset_pairs,
                              T* out_tensor);
  virtual void PostProcess(const int64 size, T* out_tensor) = 0;
  DistributionConfig rng_config_proto_;
  GuardedPhiloxRandom generator_;
};

template <class Distribution>
class RandomUniformInitializer : public SamplingBasedInitializer<Distribution> {
 public:
  explicit RandomUniformInitializer(const DistributionConfig& proto);
 protected:
  typedef typename Distribution::ResultElementType T;
  virtual void PostProcess(const int64 size, T* out_tensor) override;
};

template <class Distribution>
class RandomNormalInitializer : public SamplingBasedInitializer<Distribution> {
 public:
  explicit RandomNormalInitializer(const DistributionConfig& proto);
 protected:
  typedef typename Distribution::ResultElementType T;
  virtual void PostProcess(const int64 size, T* out_tensor) override;
};

template <class Distribution>
class TruncatedNormalInitializer : public SamplingBasedInitializer<Distribution> {
 public:
  explicit TruncatedNormalInitializer(const DistributionConfig& proto);
 protected:
  typedef typename Distribution::ResultElementType T;
  virtual void PostProcess(const int64 size, T* out_tensor) override;
};

template<typename T>
class ConstantInitializer {
 public:
  explicit ConstantInitializer(const DistributionConfig& proto)
      : rng_config_proto_(proto) {}
  void DoInitialize(const Shape& shape, T* out_tensor);
  void DoInitialize(const Shape& shape, const Shape& shard_shape,
                    const std::vector<std::pair<int64, int64>>& start_offset_pairs,
                    T* out_tensor);

 private:
  DistributionConfig rng_config_proto_; 
};

#define REGISTER(TYPE)                                                                        \
  template struct functor::FillPhiloxRandom<                                                  \
      tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, TYPE>>;       \
  template struct functor::FillPhiloxRandom<                                                  \
      tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, TYPE>>;        \
  template struct functor::FillPhiloxRandom<                                                  \
      tensorflow::random::TruncatedNormalDistribution<                                        \
          tensorflow::random::SingleSampleAdapter<tensorflow::random::PhiloxRandom>, TYPE>>;  \
  
REGISTER(float);
#undef REGISTER

class DistributedRandomInitializer {
 public:
  // The universe API exposed to users to generate random tensors according to
  // initialization specification.
  static void Initialize(
      const DistributionConfig& proto, const Shape& shape, void* out_tensor);

  // Initialize for sharded tensor.
  static void Initialize(
      const DistributionConfig& proto, const Shape& full_shape, const Shape& shard_shape,
      const std::vector<std::pair<int64, int64>>& start_offset_pairs, void* out_tensor);
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_PJRT_LIFETIME_TRACKER_H
