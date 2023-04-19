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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_CHAIN_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_CHAIN_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/parallel/hlo_strategy_spec.h"

#include <unordered_set>

namespace xla {

class SyncFreeChain {
 public:
  explicit SyncFreeChain() {};
	explicit SyncFreeChain(
      std::unordered_set<const HloInstruction*>& sync_points) : sync_points_(sync_points) {}

	std::unordered_set<const HloInstruction*>* mutable_sync_points() {
	  return &sync_points_;
	}

  const std::unordered_set<const HloInstruction*>& sync_points() const {
    return sync_points_;
  }

  std::unordered_map<const HloInstruction*,
	                   const HloInstruction*>* mutable_sync_coll_map() {
    return &sync_coll_map_;
  }

  const std::unordered_map<const HloInstruction*,
	                         const HloInstruction*>& sync_coll_map() const {
    return sync_coll_map_;
  }

 private:
  std::unordered_set<const HloInstruction*> sync_points_;     // Collect all sync points groups
	std::unordered_map<const HloInstruction*,
	                   const HloInstruction*> sync_coll_map_; // maps from sync points to custom collective
};
} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_SYNC_FREE_CHAIN_H_
