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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/dapple_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "third_party/nccl/nccl.h"

// Tests cross-GPU operations of kDAPPLEAllToAll.
//
// This test requires at least four GPUs.

namespace xla {
namespace {

class DAPPLEAllToAllTest : public HloTestBase {
 protected:
  // Setup nccl communicators for all Sharding replicas.
  void InitSPMDComms(ncclComm_t* comms, int num_replicas, HloModule* module) {
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
    module->init_dp_comms(num_replicas);

    // Setup spmd comms.
    for (int i = 0; i < num_replicas; ++i) {
      module->set_dp_comm(num_replicas, i, &comms[i]);
    }
  }
};

// Test for DAPPLEAllToAll with split_dimension=0.
XLA_TEST_F(DAPPLEAllToAllTest, AllToAll_Dim0) {
  const char* const kModuleStr = R"(
  HloModule dapple-all-to-test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[4, 2] broadcast(id), dimensions={}
    a0 = u32[4, 2] constant({{10, 15}, {20, 25}, {30, 35}, {40, 45}})
    a1 = u32[4, 2] add(id2, a0)
    all2all = u32[4, 2] dapple-all-to-all(a1), dimensions={0}
    ROOT out = u32[8] reshape(all2all)
  }
  )";
  const int64 kNumReplicas = 4;
  auto config = GetModuleConfigForTest(kNumReplicas);
  ncclComm_t comms[kNumReplicas];
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  InitSPMDComms(comms, kNumReplicas, module.get());

  VLOG(0) << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32>({10, 15, 11, 16, 12, 17, 13, 18},
                                         results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32>({20, 25, 21, 26, 22, 27, 23, 28},
                                         results[1]);
  LiteralTestUtil::ExpectR1Equal<uint32>({30, 35, 31, 36, 32, 37, 33, 38},
                                         results[2]);
  LiteralTestUtil::ExpectR1Equal<uint32>({40, 45, 41, 46, 42, 47, 43, 48},
                                         results[3]);
}

// Test for DAPPLEAllToAll with split_dimension=1.
XLA_TEST_F(DAPPLEAllToAllTest, AllToAll_Dim1) {
  const char* const kModuleStr = R"(
  HloModule dapple-all-to-test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[4, 2] broadcast(id), dimensions={}
    a0 = u32[4, 2] constant({{10, 15}, {20, 25}, {30, 35}, {40, 45}})
    a1 = u32[4, 2] add(id2, a0)
    all2all = u32[4, 2] dapple-all-to-all(a1), dimensions={1}
    ROOT out = u32[8] reshape(all2all)
  }
  )";
  const int64 kNumReplicas = 4;
  auto config = GetModuleConfigForTest(kNumReplicas);
  ncclComm_t comms[kNumReplicas];
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  InitSPMDComms(comms, kNumReplicas, module.get());

  VLOG(0) << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32>({10, 12, 20, 22, 11, 13, 21, 23},
                                         results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32>({30, 32, 40, 42, 31, 33, 41, 43},
                                         results[1]);
  LiteralTestUtil::ExpectR1Equal<uint32>({15, 17, 25, 27, 16, 18, 26, 28},
                                         results[2]);
  LiteralTestUtil::ExpectR1Equal<uint32>({35, 37, 45, 47, 36, 38, 46, 48},
                                         results[3]);
}

XLA_TEST_F(DAPPLEAllToAllTest, AllToAll_Dim0_2_shards) {
  const char* const kModuleStr = R"(
  HloModule dapple-all-to-test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[4, 2] broadcast(id), dimensions={}
    a0 = u32[4, 2] constant({{10, 15}, {20, 25}, {30, 35}, {40, 45}})
    a1 = u32[4, 2] add(id2, a0)
    all2all = u32[4, 2] dapple-all-to-all(a1), dimensions={0}
    ROOT out = u32[8] reshape(all2all)
  }
  )";
  const int64 kNumReplicas = 2;
  auto config = GetModuleConfigForTest(kNumReplicas);
  ncclComm_t comms[kNumReplicas];
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  InitSPMDComms(comms, kNumReplicas, module.get());

  VLOG(0) << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32>({10, 15, 20, 25, 11, 16, 21, 26},
                                         results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32>({30, 35, 40, 45, 31, 36, 41, 46},
                                         results[1]);
}

XLA_TEST_F(DAPPLEAllToAllTest, AllToAll_Dim0_2_shards_reshape) {
  const char* const kModuleStr = R"(
  HloModule dapple-all-to-test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[4, 2] broadcast(id), dimensions={}
    a0 = u32[4, 2] constant({{10, 15}, {20, 25}, {30, 35}, {40, 45}})
    a1 = u32[4, 2] add(id2, a0)
    %reshape = u32[4,2,1]{2,1,0} reshape(a1)
    all2all = u32[4,2,1] dapple-all-to-all(%reshape), dimensions={0}
    ROOT out = u32[8] reshape(all2all)
  }
  )";
  const int64 kNumReplicas = 2;
  auto config = GetModuleConfigForTest(kNumReplicas);
  ncclComm_t comms[kNumReplicas];
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  InitSPMDComms(comms, kNumReplicas, module.get());

  VLOG(0) << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32>({10, 15, 20, 25, 11, 16, 21, 26},
                                         results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32>({30, 35, 40, 45, 31, 36, 41, 46},
                                         results[1]);
}

// Test for DAPPLEAllToAll from moe_model_example.py
XLA_TEST_F(DAPPLEAllToAllTest, AllToAll_2) {
  const char* const kModuleStr = R"(
  HloModule dapple-all-to-test
  ENTRY %entry_spmd {
    id = u32[] replica-id()
    id2 = u32[4, 2] broadcast(id), dimensions={}
    a0 = u32[4, 2] constant({{10, 15}, {20, 25}, {30, 35}, {40, 45}})
    a1 = u32[4, 2] add(id2, a0)
    %reshape = u32[4,2,1]{2,1,0} reshape(a1)
    %dapple-all-to-all = u32[4,2,1]{2,1,0} dapple-all-to-all(%reshape), dimensions={0}
    %transpose = u32[2,4,1]{2,0,1} transpose(%dapple-all-to-all), dimensions={1,0,2}
    %reshape.1 = u32[8,1]{1,0} reshape(%transpose)
    ROOT out = u32[8] reshape(reshape.1)
  }

  )";
  const int64 kNumReplicas = 2;
  auto config = GetModuleConfigForTest(kNumReplicas);
  ncclComm_t comms[kNumReplicas];
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  InitSPMDComms(comms, kNumReplicas, module.get());

  VLOG(0) << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32>({10, 20, 11, 21, 15, 25, 16, 26},
                                         results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32>({30, 40, 31, 41, 35, 45, 36, 46},
                                         results[1]);
}

} // namespace
} // namespace xla