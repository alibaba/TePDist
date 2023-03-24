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

#include "tensorflow/compiler/xla/service/dapple_all_reduce_combiner.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using absl::nullopt;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;
int64 kMaxCombineCount = 256;

int64 AllReduceCount(const HloModule& module) {
  int64 count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kDAPPLEAllReduce) {
        ++count;
      }
    }
  }
  return count;
}

// Create and add a reduction computation in the given type to the module.
HloComputation* MakeReduction(const HloOpcode type, HloModule* module) {
  HloComputation::Builder sum_builder(HloOpcodeString(type));
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "y"));
  sum_builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {}), type, x, y));
  HloComputation* reduction =
      module->AddEmbeddedComputation(sum_builder.Build());
  return reduction;
}

using DAPPLEAllReduceCombinerTest = HloTestBase;

HloInstruction* DAPPLEMakeCrossReplicaReductions(
    std::vector<int64> sizes_in_kib, /*std::vector<HloComputation*> reductions,*/
    std::vector<HloInstruction*>* inputs, HloComputation::Builder* b,
    int num_replicas, bool sharding) {
  // CHECK_EQ(reductions.size(), sizes_in_kib.size());
  std::vector<HloInstruction*> all_reduces;
  for (int i = 0; i < sizes_in_kib.size(); i++) {
    int64 size_in_kib = sizes_in_kib[i];
    // HloComputation* reduction = reductions[i];
    auto constant = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.3)));
    Shape shape = ShapeUtil::MakeShape(
        F32, {static_cast<int32>(size_in_kib * 1024 / sizeof(float))});
    auto input =
        b->AddInstruction(HloInstruction::CreateBroadcast(shape, constant, {}));
    inputs->push_back(input);
    all_reduces.push_back(b->AddInstruction(
        HloInstruction::CreateDAPPLEAllReduce(
            shape, {input}, num_replicas, sharding)));
  }
  return b->AddInstruction(HloInstruction::CreateTuple(all_reduces));
}

// Tests combination of several AllReduce instructions.
TEST_F(DAPPLEAllReduceCombinerTest, DAPPLECombineAllReduces) {
  auto module = CreateNewVerifiedModule();
  HloComputation* sum = MakeReduction(HloOpcode::kAdd, module.get());

  HloComputation::Builder b(TestName());
  std::vector<HloInstruction*> inputs;
  int num_replicas= 3;
  bool sharding = false;
  auto root = DAPPLEMakeCrossReplicaReductions(
      {1, 2, 10}, &inputs, &b, num_replicas, sharding);
  auto computation = module->AddEntryComputation(b.Build());

  // Run the AllReduce combiner optimization pass.
  DAPPLEAllReduceCombiner combine(10 * 1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllReduceCount(*module), inputs.size());
  VLOG(0) << "before run combiner module:" << module->ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  VLOG(0) << "after run combiner module:" << module->ToString();
  ASSERT_EQ(AllReduceCount(*module), 1);
  EXPECT_TRUE(changed);

  ASSERT_EQ(root, computation->root_instruction());
  ASSERT_EQ(inputs.size(), root->operands().size());

  HloInstruction* combined = nullptr;
  for (int64 i = 0; i < root->operands().size(); ++i) {
    HloInstruction* hlo = root->mutable_operand(i);
    ASSERT_TRUE(hlo->opcode() == HloOpcode::kGetTupleElement);
    EXPECT_EQ(hlo->tuple_index(), i);
    EXPECT_TRUE(ShapeUtil::Equal(inputs[i]->shape(), hlo->shape()));

    if (combined == nullptr) {
      // Verify the combined all reduce instruction.
      combined = hlo->mutable_operand(0);
      ASSERT_TRUE(combined->opcode() == HloOpcode::kDAPPLEAllReduce);
      EXPECT_TRUE(ShapeUtil::Equal(root->shape(), combined->shape()));
      ASSERT_EQ(combined->operands().size(), inputs.size());
    }
    EXPECT_EQ(combined, hlo->operand(0));
    EXPECT_TRUE(ShapeUtil::Equal(inputs[i]->shape(), hlo->shape()));
    EXPECT_EQ(combined->operand(i), inputs[i]);
    EXPECT_EQ(1, inputs[i]->users().size());
  }
  ASSERT_NE(combined, nullptr);
}

// Tests that the combination threshold is respected.
TEST_F(DAPPLEAllReduceCombinerTest, RespectThreshold) {
  auto module = CreateNewVerifiedModule();
  HloComputation* sum = MakeReduction(HloOpcode::kAdd, module.get());

  HloComputation::Builder b(TestName());
  std::vector<HloInstruction*> inputs;
  int num_replicas= 2;
  bool sharding = false;
  DAPPLEMakeCrossReplicaReductions({8, 4}, &inputs, &b, num_replicas, sharding);
  module->AddEntryComputation(b.Build());

  // Run the AllReduce combiner optimization pass with threshold less than
  // the combined size of the all reduce ops so that the combination
  // cannot occur.
  {
    DAPPLEAllReduceCombiner combine((8 + 4) * 1024 - 1, kMaxCombineCount);
    ASSERT_EQ(AllReduceCount(*module), inputs.size());
    TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
    EXPECT_EQ(AllReduceCount(*module), inputs.size());
    EXPECT_FALSE(changed);
  }

  // Run the AllReduce combiner optimization pass again with a slightly
  // higher threshold so that the combination can occur.
  {
    DAPPLEAllReduceCombiner combine((8 + 4) * 1024, kMaxCombineCount);
    ASSERT_EQ(AllReduceCount(*module), inputs.size());
    TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
    EXPECT_EQ(AllReduceCount(*module), 1);
    EXPECT_TRUE(changed);
  }
}

}  // namespace
}  // namespace xla
