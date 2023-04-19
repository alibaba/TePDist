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

#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/dapple_all_gather_thunk.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "third_party/nccl/nccl.h"

// Tests cross-GPU operations of kDAPPLEAllGather.
//
// This test requires at least four GPUs.

namespace xla {
namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class DAPPLEAllGatherTest : public HloTestBase {
 protected:
  // Setup nccl communicators for all Sharding replicas.
  void InitShardingComms(ncclComm_t* comms, int num_replicas, HloModule* module) {
    // For NCCL 2
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    int saved_device = 0;
    cudaGetDevice(&saved_device);
    ncclGroupStart();
    for (int i = 0; i < num_replicas; ++i) {
      cudaSetDevice(i);
      ncclCommInitRank(&comms[i], num_replicas, nccl_id, i);
    }
    ncclGroupEnd();
    cudaSetDevice(saved_device);

    // Setup sharding comms.
    for (int i = 0; i < num_replicas; ++i) {
      module->set_sharding_comm(i, &comms[i]);
    }
  }

};

XLA_TEST_F(DAPPLEAllGatherTest, AllGather_Dim0) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[1, 2] broadcast(id), dimensions={}
    a0 = u32[1, 2] constant({{10, 15}})
    a1 = u32[1, 2] add(id2, a0)
    allgather = u32[4, 2] dapple-all-gather(a1), dimensions={0}
    ROOT out = u32[8] reshape(allgather)
  }
  )";
  const int64 kNumReplicas = 4;
  auto config = GetModuleConfigForTest(kNumReplicas);
  ncclComm_t comms[kNumReplicas];
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  InitShardingComms(comms, kNumReplicas, module.get());

  VLOG(0) << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  for (const Literal& result : results) {
    LiteralTestUtil::ExpectR1Equal<uint32>({10, 15, 11, 16, 12, 17, 13, 18},
                                           result);
  }
}

XLA_TEST_F(DAPPLEAllGatherTest, AllGather_Dim1) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2, 1] broadcast(id), dimensions={}
    a0 = u32[2, 1] constant({{10}, {15}})
    a1 = u32[2, 1] add(id2, a0)
    allgather = u32[2, 4] dapple-all-gather(a1), dimensions={1}
    ROOT out = u32[8] reshape(allgather)
  }
  )";
  const int64 kNumReplicas = 4;
  auto config = GetModuleConfigForTest(kNumReplicas);
  ncclComm_t comms[kNumReplicas];
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  InitShardingComms(comms, kNumReplicas, module.get());

  VLOG(0) << "HloModule->" << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  for (const Literal& result : results) {
    LiteralTestUtil::ExpectR1Equal<uint32>({10, 11, 12, 13, 15, 16, 17, 18},
                                           result);
  }
}

XLA_TEST_F(DAPPLEAllGatherTest, AllGather_Dim1_2x3) {
  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2, 3]{1,0} broadcast(id), dimensions={}
    allgather = u32[2, 12] dapple-all-gather(id2), dimensions={1}
    ROOT out = u32[24] reshape(allgather)
  }
  )";
  const int64 kNumReplicas = 4;
  auto config = GetModuleConfigForTest(kNumReplicas);
  ncclComm_t comms[kNumReplicas];
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  InitShardingComms(comms, kNumReplicas, module.get());

  VLOG(0) << "HloModule->" << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  for (const Literal& result : results) {
    LiteralTestUtil::ExpectR1Equal<uint32>(
        {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3},
         result);
  }
}

}  // namespace
}  // namespace xla
