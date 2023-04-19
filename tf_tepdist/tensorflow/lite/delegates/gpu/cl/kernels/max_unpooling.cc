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

#include "tensorflow/lite/delegates/gpu/cl/kernels/max_unpooling.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetMaxUnpoolingKernelCode(
    const OperationDef& op_def, const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src("src_data",
                          WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
                          op_def.src_tensors[0]);
  TensorCodeGenerator src_ind(
      "src_data_indices", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[1]);
  TensorCodeGenerator dst("dst_data",
                          WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
                          op_def.dst_tensors[0]);

  const auto address_mode = GetFastestZeroMode(device);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += src.GetDeclaration(AccessType::READ) + ",\n";
  c += src_ind.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,      \n";
  c += "    int4 dst_size,      \n";
  c += "    int2 kernel_size,   \n";
  c += "    int2 padding,       \n";
  c += "    int2 stride         \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X0 = linear_id / dst_size.w;\n";
    c += "  int B = linear_id % dst_size.w;\n";
    c += "  int src_x0 = (X0 + padding.x) / stride.x;\n";
    c += "  int src_x = src_x0 * dst_size.w + B;\n";
  } else {
    c += "  int src_x = (X + padding.x) / stride.x;\n";
  }
  c += "  int src_y = (Y + padding.y) / stride.y;\n";
  c += "  " + src.GetAddressWHS("src_adr", "src_x", "src_y", "Z") + "\n";
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER) {
    c += "  bool outside = src_x < 0 || src_y < 0 ||";
    c += "  src_x >= src_size.x || src_y >= src_size.y;\n";
    c += "  FLT4 src = (FLT4)(0.0f);\n";
    c += "  int4 ind = (int4)(0);\n";
    c += "  if (!outside) {\n";
    c += "    src = " + src.Read("src_adr") + ";\n";
    c += "    ind = convert_int4(" + src_ind.Read("src_adr") + ");\n";
    c += "  }\n";
  } else {
    c += "  FLT4 src = " + src.Read("src_adr", address_mode) + ";\n";
    c += "  int4 ind = convert_int4(" + src_ind.Read("src_adr", address_mode) +
         ");\n";
  }
  if (op_def.IsBatchSupported()) {
    c += "  int t_x = X0 - (src_x0 * stride.x - padding.x);\n";
  } else {
    c += "  int t_x = X - (src_x * stride.x - padding.x);\n";
  }
  c += "  int t_y = Y - (src_y * stride.y - padding.y);\n";
  c += "  int t_index = t_y * kernel_size.x + t_x;\n";
  c += "  FLT4 result;\n";
  const std::string channels[] = {".x", ".y", ".z", ".w"};
  for (int i = 0; i < 4; ++i) {
    const auto& s = channels[i];
    c += "  result" + s + "= t_index == ind" + s + "? src" + s + ": 0.0f;\n";
  }
  c += PostProcess(linked_operations, {"result", "X", "Y", "Z"});
  c += "  " + dst.WriteWHS("result", "X", "Y", "Z");
  c += "}\n";

  return c;
}

std::string GetMaxUnpooling3DKernelCode(
    const OperationDef& op_def, const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src(
      "src_data",
      WHDSPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator src_ind(
      "src_data_indices",
      WHDSPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[1]);
  TensorCodeGenerator dst(
      "dst_data",
      WHDSPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);

  const auto address_mode = GetFastestZeroMode(device);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += src.GetDeclaration(AccessType::READ) + ",\n";
  c += src_ind.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,      \n";
  c += "    int4 dst_size,      \n";
  if (op_def.IsBatchSupported()) {
    c += "    int batch_size,          \n";
  }
  c += "    int4 kernel_size,   \n";
  c += "    int4 padding,       \n";
  c += "    int4 stride         \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int linear_id_z = get_global_id(2);\n";
  c += "  int S = linear_id_z % dst_size.w;\n";
  c += "  int Z = linear_id_z / dst_size.w;\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X0 = linear_id / batch_size;\n";
    c += "  int B = linear_id % batch_size;\n";
    c += "  int src_x0 = (X0 + padding.x) / stride.x;\n";
    c += "  int src_x = src_x0 * batch_size + B;\n";
  } else {
    c += "  int src_x = (X + padding.x) / stride.x;\n";
  }
  c += "  int src_y = (Y + padding.y) / stride.y;\n";
  c += "  int src_z = (Z + padding.z) / stride.z;\n";
  c += "  " + src.GetAddressWHDS("src_adr", "src_x", "src_y", "src_z", "S") +
       "\n";
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER) {
    c += "  bool outside = src_x < 0 || src_y < 0 || src_z < 0 || ";
    c += "  src_x >= src_size.x || src_y >= src_size.y || src_z >= "
         "src_size.z;\n";
    c += "  FLT4 src = (FLT4)(0.0f);\n";
    c += "  int4 ind = (int4)(0);\n";
    c += "  if (!outside) {\n";
    c += "    src = " + src.Read("src_adr") + ";\n";
    c += "    ind = convert_int4(" + src_ind.Read("src_adr") + ");\n";
    c += "  }\n";
  } else {
    c += "  FLT4 src = " + src.Read("src_adr", address_mode) + ";\n";
    c += "  int4 ind = convert_int4(" + src_ind.Read("src_adr", address_mode) +
         ");\n";
  }
  if (op_def.IsBatchSupported()) {
    c += "  int t_x = X0 - (src_x0 * stride.x - padding.x);\n";
  } else {
    c += "  int t_x = X - (src_x * stride.x - padding.x);\n";
  }
  c += "  int t_y = Y - (src_y * stride.y - padding.y);\n";
  c += "  int t_z = Z - (src_z * stride.z - padding.z);\n";
  c += "  int t_index = (t_y * kernel_size.x + t_x) * kernel_size.z + t_z;\n";
  c += "  FLT4 result;\n";
  const std::string channels[] = {".x", ".y", ".z", ".w"};
  for (int i = 0; i < 4; ++i) {
    const auto& s = channels[i];
    c += "  result" + s + " = t_index == ind" + s + " ? src" + s + ": 0.0f;\n";
  }
  c += PostProcess(linked_operations, {"result", "X", "Y", "S"});
  c += "  " + dst.WriteWHDS("result", "X", "Y", "Z", "S");
  c += "}\n";
  return c;
}
}  // namespace

MaxUnpooling::MaxUnpooling(const OperationDef& definition,
                           const MaxUnpooling2DAttributes& attr)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h),
      padding_(attr.padding.appended.w, attr.padding.appended.h),
      kernel_size_(attr.kernel.w, attr.kernel.h) {}

MaxUnpooling::MaxUnpooling(MaxUnpooling&& kernel)
    : GPUOperation(std::move(kernel)),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      kernel_size_(kernel.kernel_size_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

MaxUnpooling& MaxUnpooling::operator=(MaxUnpooling&& kernel) {
  if (this != &kernel) {
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(kernel_size_, kernel.kernel_size_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

absl::Status MaxUnpooling::Compile(const CreationContext& creation_context) {
  const auto code = GetMaxUnpoolingKernelCode(
      definition_, *creation_context.device, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status MaxUnpooling::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[1]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_size_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(padding_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));
  return absl::OkStatus();
}

int3 MaxUnpooling::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status MaxUnpooling::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status MaxUnpooling::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

MaxUnpooling CreateMaxUnpooling(const OperationDef& definition,
                                const MaxUnpooling2DAttributes& attr) {
  return MaxUnpooling(definition, attr);
}

MaxUnpooling3D::MaxUnpooling3D(const OperationDef& definition,
                               const MaxUnpooling3DAttributes& attr)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h, attr.strides.d),
      padding_(attr.padding.appended.w, attr.padding.appended.h,
               attr.padding.appended.d),
      kernel_size_(attr.kernel.w, attr.kernel.h, attr.kernel.d) {}

MaxUnpooling3D::MaxUnpooling3D(MaxUnpooling3D&& kernel)
    : GPUOperation(std::move(kernel)),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      kernel_size_(kernel.kernel_size_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

MaxUnpooling3D& MaxUnpooling3D::operator=(MaxUnpooling3D&& kernel) {
  if (this != &kernel) {
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(kernel_size_, kernel.kernel_size_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

absl::Status MaxUnpooling3D::Compile(const CreationContext& creation_context) {
  const auto code = GetMaxUnpooling3DKernelCode(
      definition_, *creation_context.device, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status MaxUnpooling3D::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[1]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHDS()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHDS()));
  if (definition_.IsBatchSupported()) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->Batch()));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(
      int4(kernel_size_.x, kernel_size_.y, kernel_size_.z, 1)));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int4(padding_.x, padding_.y, padding_.z, 1)));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int4(stride_.x, stride_.y, stride_.z, 1)));
  return absl::OkStatus();
}

int3 MaxUnpooling3D::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices() * dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status MaxUnpooling3D::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status MaxUnpooling3D::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

MaxUnpooling3D CreateMaxUnpooling3D(const OperationDef& definition,
                                    const MaxUnpooling3DAttributes& attr) {
  return MaxUnpooling3D(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
