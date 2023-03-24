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

// Basic server binary that exposes a xla::Service through a GRPC interface
// on a configurable port.
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/rpc/grpc_service.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include <fstream>

namespace xla {
namespace {

int ReduceRealMain(int argc, char** argv) {
  const char* const hlo_string = R"(
HloModule BadReduce

Sum {
  x.1 = f32[] parameter(0)
  y.1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(x.1, y.1)
}

ENTRY reduce.1 {
  parameter = f32[4,8,8,50176]{3,2,1,0} parameter(0)
  init_value = f32[] constant(0)
  reduce = f32[4,8]{1,0} reduce(parameter, init_value), dimensions={1,3}, to_apply=Sum
  ROOT copy = f32[4,8]{1,0} copy(reduce)
}
)";

  se::Platform* platform = PlatformUtil::GetPlatform("CUDA").ValueOrDie();
  HloRunner runner(platform);
 
  HloModuleConfig config;
  auto module_or = ParseAndReturnUnverifiedModule(hlo_string, config);
  auto module = module_or.ConsumeValueOrDie();

  Shape reduce_output_shape = ShapeUtil::MakeShape(PrimitiveType::F32, {4, 8});
  *reduce_output_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  Shape reduce_input_shape = ShapeUtil::MakeShape(PrimitiveType::F32, {4, 8, 8, 50176});
  *reduce_input_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2, 3});

  //std::ifstream input_file("spmd_ga/bitcast.64_device0_binary", std::ios::binary);
  auto input = Literal::CreateFromShape(reduce_input_shape);
  int elem_cnt = ShapeUtil::ByteSizeOf(reduce_input_shape) / 4;
  //input_file.read((char *)(input.untyped_data()), sizeof(float) * elem_cnt);
  //for (int x = 0; x < 1; ++x) {
  //  for (int y = 0; y < 5; ++y) {
  //    *((float *)(input.untyped_data()) + x * 401408 + y) = 0.1;
  //  }
  //}

  auto constant = LiteralUtil::Zero(reduce_input_shape.element_type());

  VLOG(0) << "Running HLO module on platform " << platform->Name() << "...\n";
  auto result_status = runner.Execute(std::move(module), {&input, &constant}, false);

  TF_QCHECK_OK(result_status.status())
      << "Failed to execute on " << platform->Name() << "\n";

  auto result = result_status.ConsumeValueOrDie();
  for (int i = 0; i < ShapeUtil::ByteSizeOf(reduce_output_shape) / 4; ++i) {
    VLOG(0) << *((float *)(result.untyped_data()) + i);
  }

  return 0;
}

int AddRealMain(int argc, char** argv) {
  const char* const hlo_string = R"(
HloModule Add64

ENTRY entry {
  p0 = f32[64]{0} parameter(0)
  p1 = f32[64]{0} parameter(1)
  add = f32[64]{0} add(f32[64]{0} p0, f32[64]{0} p1)
  ROOT copy = f32[64]{0} copy(f32[64]{0} add)
}
)";

  se::Platform* platform = PlatformUtil::GetPlatform("CUDA").ValueOrDie();
  HloRunner runner(platform);
 
  HloModuleConfig config;
  auto module_or = ParseAndReturnUnverifiedModule(hlo_string, config);
  auto module = module_or.ConsumeValueOrDie();

  Shape add_shape = ShapeUtil::MakeShape(PrimitiveType::F32, {64});
  *add_shape.mutable_layout() = LayoutUtil::MakeLayout({0});

  Shape param_shape = ShapeUtil::MakeShape(PrimitiveType::F32, {64});
  *param_shape.mutable_layout() = LayoutUtil::MakeLayout({0});

  auto p0 = Literal::CreateFromShape(param_shape);
  auto p1 = Literal::CreateFromShape(param_shape);

  VLOG(0) << "Running HLO module on platform " << platform->Name() << "...\n";
  auto result_status = runner.Execute(std::move(module), {&p0, &p1}, false);

  TF_QCHECK_OK(result_status.status())
      << "Failed to execute on " << platform->Name() << "\n";

  auto result = result_status.ConsumeValueOrDie();

  return 0;
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) { return xla::ReduceRealMain(argc, argv); }
