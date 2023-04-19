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
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "third_party/nccl/nccl.h"

// Tests cross-GPU operations of kDAPPLEAllReduce.
//
// This test requires at least four GPUs.

namespace xla {
namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

// Shorter alias for this function.
absl::flat_hash_set<GlobalDeviceId> OpenNcclChannels() {
  return gpu::NcclAllReduceThunk::DevicesWithOpenNcclChannels();
}

class DAPPLEAllReduceTest : public HloTestBase {
 protected:
  std::unique_ptr<HloModule> GetCrsModule(const std::string reduction_op,
                                          const HloModuleConfig& config) {
    std::string hlo_template;
    if (reduction_op == "add") {
      hlo_template = R"(
        HloModule dapple_all_reduce_test

        ENTRY %test_computation (p: f32[2,2]) -> f32[2,2] {
          %p = f32[2,2]{1,0} parameter(0)
          %p2 = f32[2,2]{1,0} bitcast(f32[2,2]{1,0} %p)
          %crs = f32[2,2]{1,0} dapple-all-reduce(f32[2,2]{1,0} %p2), num_replicas=4, reduction_type="Sum"
          %copy = f32[2,2]{1,0} copy(f32[2,2]{1,0} %crs)
          ROOT %out = f32[2,2]{1,0} bitcast(f32[2,2]{1,0} %copy)
        }

      )";
    } else if (reduction_op == "maximum") {
      hlo_template = R"(
        HloModule dapple_all_reduce_test

        ENTRY %test_computation (p: f32[2,2]) -> f32[2,2] {
          %p = f32[2,2]{1,0} parameter(0)
          %p2 = f32[2,2]{1,0} bitcast(f32[2,2]{1,0} %p)
          %crs = f32[2,2]{1,0} dapple-all-reduce(f32[2,2]{1,0} %p2), num_replicas=4, reduction_type="Max"
          %copy = f32[2,2]{1,0} copy(f32[2,2]{1,0} %crs)
          ROOT %out = f32[2,2]{1,0} bitcast(f32[2,2]{1,0} %copy)
        }

      )";
    } else { CHECK(0); }

    return ParseAndReturnVerifiedModule(hlo_template, config).ValueOrDie();
  }

  // Setup nccl communicators for all DP replicas.
  void InitDPComms(ncclComm_t* comms, int num_replicas, HloModule* module) {
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

    // Setup dp comms.
    for (int i = 0; i < num_replicas; ++i) {
      module->set_dp_comm(num_replicas, i, &comms[i]);
    }
  }

  template <typename LiteralType>
  void TestFourReplicasOneOperand(const std::string reduction_op,
      Literal input_value, Literal expected_value) {
    const int kNumReplicas = 4;
    auto config = GetModuleConfigForTest();
    config.set_replica_count(kNumReplicas);
    auto module = GetCrsModule(reduction_op, config);
    ncclComm_t comms[kNumReplicas];
    InitDPComms(comms, kNumReplicas, module.get());
    VLOG(0) << module->ToString();
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                            ExecuteReplicated(std::move(module), {&input_value},
                                              /*num_replicas=*/kNumReplicas,
                                              /*use_threads=*/true));

    for (int replica_idx = 0; replica_idx < kNumReplicas; replica_idx++) {
      EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
          expected_value, results[replica_idx], ErrorSpec{1e-5, 1e-5}));
    }

    for (int i = 0; i < kNumReplicas; ++i) {
      ncclCommDestroy(comms[i]);
    }
  }

};

XLA_TEST_F(DAPPLEAllReduceTest, AllReduce_sum_float32_2D) {
  EXPECT_THAT(OpenNcclChannels(), IsEmpty());

  TestFourReplicasOneOperand<float>(
      "add",
      /*input_value=*/LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}),
      /*expected_value=*/LiteralUtil::CreateR2<float>({{4, 8}, {12, 16}}));

  TestFourReplicasOneOperand<float>(
      "add",
      /*input_value=*/LiteralUtil::CreateR2<float>({{1.11, 2.22},
                                                    {3.33, 4.44}}),
      /*expected_value=*/LiteralUtil::CreateR2<float>({{4.44, 8.88},
                                                       {13.32, 17.76}}));

  // Channels are closed as expected.
  EXPECT_THAT(OpenNcclChannels(), IsEmpty());
}

XLA_TEST_F(DAPPLEAllReduceTest, AllReduce_max_float32_2D) {
  EXPECT_THAT(OpenNcclChannels(), IsEmpty());

  TestFourReplicasOneOperand<float>(
      "maximum",
      /*input_value=*/LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}),
      /*expected_value=*/LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}}));

  TestFourReplicasOneOperand<float>(
      "maximum",
      /*input_value=*/LiteralUtil::CreateR2<float>({{1.11, 2.22},
                                                    {3.33, 4.44}}),
      /*expected_value=*/LiteralUtil::CreateR2<float>({{1.11, 2.22},
                                                       {3.33, 4.44}}));

  // Channels are closed as expected.
  EXPECT_THAT(OpenNcclChannels(), IsEmpty());
}

}  // namespace
}  // namespace xla
