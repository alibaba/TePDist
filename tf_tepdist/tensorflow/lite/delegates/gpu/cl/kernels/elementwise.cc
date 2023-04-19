/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/kernels/elementwise.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"

namespace tflite {
namespace gpu {
namespace cl {

ElementwiseOneInput::ElementwiseOneInput(ElementwiseOneInput&& operation)
    : ElementwiseOperation(std::move(operation)),
      op_type_(operation.op_type_) {}

ElementwiseOneInput& ElementwiseOneInput::operator=(
    ElementwiseOneInput&& operation) {
  if (this != &operation) {
    std::swap(op_type_, operation.op_type_);
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string ElementwiseOneInput::GetCoreCode(
    const LinkingContext& context) const {
  std::string result;
  switch (op_type_) {
    case OperationType::ABS:
      result = "$0 = fabs($0);\n";
      break;
    case OperationType::COS:
      result = "$0 = cos($0);\n";
      break;
    case OperationType::EXP:
      result = "$0 = exp($0);\n";
      break;
    case OperationType::HARD_SWISH:
      result =
          "$0 *= clamp($0 * (FLT)(0.16666667f) + (FLT)(0.5f), (FLT4)(0.0f), "
          "(FLT4)(1.0f));\n";
      break;
    case OperationType::LOG:
      result = "$0 = log($0);\n";
      break;
    case OperationType::RSQRT:
      result = "$0 = (FLT4)(1.0f) / sqrt($0);\n";
      break;
    case OperationType::SIGMOID:
      if (definition_.precision != CalculationsPrecision::F32) {
        result =
            "$0.x = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.x))));\n";
        result +=
            "$0.y = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.y))));\n";
        result +=
            "$0.z = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.z))));\n";
        result +=
            "$0.w = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.w))));\n";
      } else {
        result = "$0 = (FLT4)(1.0f) / ((FLT4)(1.0f) + exp(-($0)));\n";
      }
      break;
    case OperationType::SIN:
      result = "$0 = sin($0);\n";
      break;
    case OperationType::SQRT:
      result = "$0 = sqrt($0);\n";
      break;
    case OperationType::SQUARE:
      result = "$0 *= $0;\n";
      break;
    case OperationType::TANH:
      result = "$0 = tanh($0);\n";
      break;
    default:
      return "Unknown operation type;\n";
  }
  return absl::Substitute(result, context.var_name);
}

ElementwiseOneInput CreateElementwiseOneInput(const OperationDef& definition,
                                              const OperationType& op_type) {
  ElementwiseOneInput operation(definition, op_type);
  operation.SetLinkIndex(0);
  return operation;
}

ElementwiseTwoInput::ElementwiseTwoInput(ElementwiseTwoInput&& operation)
    : ElementwiseOperation(std::move(operation)),
      link_index_(operation.link_index_),
      op_type_(operation.op_type_),
      broadcast_(operation.broadcast_),
      scalar_para_(operation.scalar_para_),
      use_scalar_para_(operation.use_scalar_para_) {}

ElementwiseTwoInput& ElementwiseTwoInput::operator=(
    ElementwiseTwoInput&& operation) {
  if (this != &operation) {
    link_index_ = operation.link_index_;
    op_type_ = operation.op_type_;
    broadcast_ = operation.broadcast_;
    scalar_para_ = operation.scalar_para_;
    use_scalar_para_ = operation.use_scalar_para_;
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void ElementwiseTwoInput::SetLinkIndex(int index) {
  link_index_ = index;
  if (use_scalar_para_) {
    scalar_para_.SetName(absl::StrCat("scalar_para_", index));
  }
}

std::string ElementwiseTwoInput::GetCoreCode(
    const LinkingContext& context) const {
  std::string result;
  std::string second_var;
  if (use_scalar_para_) {
    second_var = absl::StrCat("(FLT)(", scalar_para_.GetName(), ")");
  } else {
    const std::string size_name = "src_size_" + std::to_string(link_index_);
    TensorCodeGenerator src_tensor(
        absl::StrCat("src_data_", link_index_),
        WHSPoint{size_name + ".x", size_name + ".y", size_name + ".z"},
        definition_.src_tensors[1]);
    const std::string x_coord = broadcast_.width ? "0" : context.x_coord;
    const std::string y_coord = broadcast_.height ? "0" : context.y_coord;
    const std::string s_coord = broadcast_.channels ? "0" : context.s_coord;
    second_var = "second_var_" + std::to_string(link_index_);
    result = "  FLT4 " + second_var + " = " +
             src_tensor.ReadWHS(x_coord, y_coord, s_coord) + ";\n";
    if (broadcast_.channels) {
      result += "  " + second_var + ".y = " + second_var + ".x;\n";
      result += "  " + second_var + ".z = " + second_var + ".x;\n";
      result += "  " + second_var + ".w = " + second_var + ".x;\n";
    }
  }
  switch (op_type_) {
    case OperationType::ADD:
      result += "$0 += $1;\n";
      break;
    case OperationType::DIV:
      result += "$0 /= $1;\n";
      break;
    case OperationType::MAXIMUM:
      result += "$0 = max($0, $1);\n";
      break;
    case OperationType::MINIMUM:
      result += "$0 = min($0, $1);\n";
      break;
    case OperationType::MUL:
      result += "$0 *= $1;\n";
      break;
    case OperationType::POW:
      result += "$0 = pow($0, $1);\n";
      break;
    case OperationType::SQUARED_DIFF:
      result += "$0 -= $1;\n";
      result += "$0 *= $0;\n";
      break;
    case OperationType::SUB:
      result += "$0 -= $1;\n";
      break;
    default:
      return "Unknown operation type;\n";
  }
  return absl::Substitute(result, context.var_name, second_var);
}

std::string ElementwiseTwoInput::GetArgsDeclaration() const {
  std::string args;
  if (use_scalar_para_) {
    absl::StrAppend(&args, ",\n    ", scalar_para_.GetDeclaration());
  } else {
    absl::StrAppend(&args, ",\n",
                    GetTensorDeclaration(AccessType::READ,
                                         absl::StrCat("src_data_", link_index_),
                                         definition_.src_tensors[1]));
    absl::StrAppend(&args, ",\n   int4 src_size_", link_index_);
  }
  return args;
}

absl::Status ElementwiseTwoInput::BindArguments(CLKernel* kernel) {
  if (use_scalar_para_) {
    RETURN_IF_ERROR(kernel->SetBytesAuto(scalar_para_));
  } else {
    RETURN_IF_ERROR(kernel->SetMemoryAuto(src_[1]->GetMemoryPtr()));
    RETURN_IF_ERROR(kernel->SetBytesAuto(src_[1]->GetWBatchedHSB()));
  }
  return absl::OkStatus();
}

ElementwiseTwoInput CreateElementwiseTwoInput(
    const CreationContext& creation_context, const OperationDef& definition,
    const OperationType& op_type, const BroadcastSettings& broadcast,
    const ElementwiseAttributes* attr) {
  ElementwiseTwoInput operation(definition, op_type, broadcast);
  if (attr) {
    const float* scalar = absl::get_if<float>(&attr->param);
    if (scalar) {
      const auto scalar_precision = creation_context.device->IsPowerVR()
                                        ? CalculationsPrecision::F32
                                        : definition.precision;
      operation.SetScalarPara(FLT(scalar_precision, *scalar));
    }
  }
  operation.SetLinkIndex(0);
  return operation;
}

ElementwiseTwoInput CreateElementwiseTwoInput(
    const OperationDef& definition, const OperationType& op_type,
    const BroadcastSettings& broadcast) {
  ElementwiseTwoInput operation(definition, op_type, broadcast);
  operation.SetLinkIndex(0);
  return operation;
}

ElementwiseTwoInput CreateElementwiseTwoInput(const OperationDef& definition,
                                              const OperationType& op_type) {
  BroadcastSettings broadcast;
  broadcast.width = false;
  broadcast.height = false;
  broadcast.channels = false;
  ElementwiseTwoInput operation(definition, op_type, broadcast);
  operation.SetLinkIndex(0);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
