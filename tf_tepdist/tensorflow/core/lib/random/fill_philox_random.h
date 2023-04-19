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

#ifndef TENSORFLOW_CORE_KERNELS_FILL_PHILOX_RANDOM_OP_CPU_H_
#define TENSORFLOW_CORE_KERNELS_FILL_PHILOX_RANDOM_OP_CPU_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace tensorflow {
namespace functor {

using random::PhiloxRandom;
using random::SingleSampleAdapter;

// A class to fill a specified range of random groups
template <class Distribution, bool VariableSamplesPerOutput>
struct FillPhiloxRandomTask;

// Specialization for distribution that takes a fixed number of samples for
// each output.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, false> {
  typedef typename Distribution::ResultElementType T;
  static void Run(random::PhiloxRandom gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    gen.Skip(start_group);
    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    for (int64 index = start_group; index < limit_group_full; ++index) {
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    if (limit_group_full < limit_group) {
      int64 remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist(&gen);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

// Specialization for distribution that takes a variable number of samples for
// each output. This will be slower due to the generality.
template <class Distribution>
struct FillPhiloxRandomTask<Distribution, true> {
  typedef typename Distribution::ResultElementType T;
  static constexpr int64 kReservedSamplesPerOutput = 256;

  static void Run(random::PhiloxRandom base_gen, T* data, int64 size,
                  int64 start_group, int64 limit_group, Distribution dist) {
    const int kGroupSize = Distribution::kResultElementCount;

    static const int kGeneratorSkipPerOutputGroup =
        kGroupSize * kReservedSamplesPerOutput /
        PhiloxRandom::kResultElementCount;

    int64 offset = start_group * kGroupSize;

    // First fill all the full-size groups
    int64 limit_group_full = std::min(limit_group, size / kGroupSize);
    int64 group_index;
    for (group_index = start_group; group_index < limit_group_full;
         ++group_index) {
      // Reset the generator to the beginning of the output group region
      // This is necessary if we want the results to be independent of order
      // of work
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + kGroupSize, data + offset);
      offset += kGroupSize;
    }

    // If there are any remaining elements that need to be filled, process them
    if (limit_group_full < limit_group) {
      PhiloxRandom gen = base_gen;
      gen.Skip(group_index * kGeneratorSkipPerOutputGroup);
      SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

      int64 remaining_size = size - limit_group_full * kGroupSize;
      auto samples = dist(&single_samples);
      std::copy(&samples[0], &samples[0] + remaining_size, data + offset);
    }
  }
};

} // namespace functor
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_FILL_PHILOX_RANDOM_OP_CPU_H_
