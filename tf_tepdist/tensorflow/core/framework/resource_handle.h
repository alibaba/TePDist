/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_
#define TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class ResourceHandleProto;

// Class representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run.
//
// This is the native C++ class equivalent of ResourceHandleProto.  They are
// separate so that kernels do not need to depend on protos.
class ResourceHandle {
 public:
  ResourceHandle();
  ResourceHandle(const ResourceHandleProto& proto);
  ~ResourceHandle();

  // Unique name for the device containing the resource.
  const std::string& device() const { return device_; }
  // Names of the devices containing the resource.
  const std::vector<string>& allowed_devices() const {
    return allowed_devices_;
  }
  void set_device(const std::string& device) { device_ = device; }
  void set_allowed_devices(const std::vector<string>& devices) {
    allowed_devices_ = devices;
  }

  // Container in which this resource is placed.
  const std::string& container() const { return container_; }
  void set_container(const std::string& container) { container_ = container; }

  // Unique name of this resource.
  const std::string& name() const { return name_; }
  void set_name(const std::string& name) { name_ = name; }

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code() const { return hash_code_; }
  void set_hash_code(uint64 hash_code) { hash_code_ = hash_code; }

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  const std::string& maybe_type_name() const { return maybe_type_name_; }
  void set_maybe_type_name(const std::string& value) {
    maybe_type_name_ = value;
  }

  // Data types and shapes for the underlying resource.
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes() const {
    return dtypes_and_shapes_;
  }
  void set_dtypes_and_shapes(
      const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes) {
    dtypes_and_shapes_ = dtypes_and_shapes;
  }

  // Conversion to and from ResourceHandleProto
  void AsProto(ResourceHandleProto* proto) const;
  void FromProto(const ResourceHandleProto& proto);

  // Serialization via ResourceHandleProto
  std::string SerializeAsString() const;
  bool ParseFromString(const std::string& s);

  void set_need_fetch(bool need_fetch) { need_fetch_ = need_fetch; }
  const bool need_fetch() const { return need_fetch_; }

  void set_backward(bool backward) { backward_ = backward; }
  const bool backward() const { return backward_; }

  void set_op_group(int op_group) { op_group_ = op_group; }
  const int op_group() const { return op_group_; }

  void set_init_ref_count(int init_ref_count) { init_ref_count_ = init_ref_count; }
  const int init_ref_count() const { return init_ref_count_; }

  std::string DebugString() const;

  // GUID for anonymous resources. Resources with this shared_name will have
  // their shared_name replaced with a GUID at creation time
  static constexpr const char* ANONYMOUS_NAME =
      "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

 public:
  // The default device containing the resource, where the ResourceHandle is
  // initially created.
  std::string device_;
  // A set of devices containing the resource. If empty, the resource only
  // exists on device_. Can be represented in wildcard patterns.
  std::vector<string> allowed_devices_;
  std::string container_;
  std::string name_;
  uint64 hash_code_ = 0;
  std::string maybe_type_name_;
  bool need_fetch_ = false;
  bool backward_ = false;
  int op_group_ = 0;
  int init_ref_count_ = 1;
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes_;
};

// For backwards compatibility for when this was a proto
std::string ProtoDebugString(const ResourceHandle& handle);

// Encodes a list of ResourceHandle protos in the given StringListEncoder.
void EncodeResourceHandleList(const ResourceHandle* p, int64 n,
                              std::unique_ptr<port::StringListEncoder> e);

// Decodes a list of ResourceHandle protos from the given StringListDecoder.
bool DecodeResourceHandleList(std::unique_ptr<port::StringListDecoder> d,
                              ResourceHandle* ps, int64 n);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_
