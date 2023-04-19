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

#include "tensorflow/compiler/xla/service/execution_plan_cache.h"

#include <utility>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

int64 GetUniquePlanId() {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  static int64 counter = 0;
  tensorflow::mutex_lock loc(mu);
  const int64 id = counter++;
  return id;
}

}  // namespace

ExecutionPlanHandle ExecutionPlanCache::Insert(
    std::shared_ptr<LocalPlan> plan) {
  tensorflow::mutex_lock lock(mutex_);

  CacheKey key = GetUniquePlanId();
  VLOG(2) << "inserting cache key: " << key;
  CHECK_EQ(cache_.count(key), 0);
  cache_.emplace(key, plan);

  ExecutionPlanHandle handle;
  handle.set_handle(key);
  return handle;
}

ExecutionPlanHandle ExecutionPlanCache::Insert(
    std::shared_ptr<DistributedPlan> plan) {
  tensorflow::mutex_lock lock(mutex_);

  CacheKey key = GetUniquePlanId();
  VLOG(2) << "inserting cache key: " << key;
  CHECK_EQ(dist_cache_.count(key), 0);
  dist_cache_.emplace(key, plan);

  ExecutionPlanHandle handle;
  handle.set_handle(key);
  return handle;
}

StatusOr<std::shared_ptr<LocalPlan>> ExecutionPlanCache::LookUp(
    const ExecutionPlanHandle& handle) const {
  tensorflow::mutex_lock lock(mutex_);

  CacheKey key = handle.handle();
  VLOG(2) << "looking up cache key: " << key;
  if (cache_.count(key) == 0) {
    VLOG(2) << "cache key not found: " << key;
    return InvalidArgumentStrCat("can not find execution plan with handle ", key);
  } else {
    auto& result = cache_.at(key);
    VLOG(2) << "hit execution plan";
    return result;
  }
}

StatusOr<std::shared_ptr<DistributedPlan>> ExecutionPlanCache::DistLookUp(
    const ExecutionPlanHandle& handle) const {
  tensorflow::mutex_lock lock(mutex_);

  CacheKey key = handle.handle();
  VLOG(2) << "looking up cache key: " << key;
  if (dist_cache_.count(key) == 0) {
    VLOG(2) << "cache key not found: " << key;
    return InvalidArgumentStrCat("can not find execution plan with handle ", key);
  } else {
    auto& result = dist_cache_.at(key);
    VLOG(2) << "hit execution plan";
    return result;
  }
}

}  // namespace xla
