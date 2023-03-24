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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTION_PLAN_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTION_PLAN_CACHE_H_

#include <map>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/pjrt/execution_plan.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {

// A cache which stores LocalPlan indexed by execution plan handle and version.
// TODO: Make it to be a template class
class ExecutionPlanCache {
 public:
  ExecutionPlanCache() {}

  ExecutionPlanHandle Insert(std::shared_ptr<LocalPlan> plan);
  ExecutionPlanHandle Insert(std::shared_ptr<DistributedPlan> plan);

  // Lookup the LocalPlan for the specified handle in the cache. Return a
  // shared_ptr to the LocalPlan if it exists in the cache.
  StatusOr<std::shared_ptr<LocalPlan>> LookUp(
      const ExecutionPlanHandle& handle) const;

  StatusOr<std::shared_ptr<DistributedPlan>> DistLookUp(
      const ExecutionPlanHandle& handle) const;

 protected:
  mutable tensorflow::mutex mutex_;

  using CacheKey = int64;

  absl::flat_hash_map<CacheKey, std::shared_ptr<LocalPlan>> cache_
      TF_GUARDED_BY(mutex_);
  absl::flat_hash_map<CacheKey, std::shared_ptr<DistributedPlan>> dist_cache_
      TF_GUARDED_BY(mutex_);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ExecutionPlanCache);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTION_PLAN_CACHE_H_

