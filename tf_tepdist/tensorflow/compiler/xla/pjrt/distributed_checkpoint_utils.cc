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

#include "tensorflow/compiler/xla/pjrt/distributed_checkpoint_utils.h"

#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/pjrt/dapple_buffer_utils.h"

#include <random>
#include <sstream>
#include <fstream>

using namespace tensorflow;

namespace dist_ckpt_utils {
namespace {

void MakeTensorShapeAndSlice(
    TensorShape& full_tensor_shape, const xla::Shape& sharded_shape,
    int64 slice_id, TensorShape* sharded_tensor_shape, TensorSlice* slice) {
  int slice_dim = -1;
  for (int d = 0; d < full_tensor_shape.dims(); ++d) {
    if (full_tensor_shape.dim_size(d) == sharded_shape.dimensions(d)) continue;
    slice_dim = d;
    break;
  }

  for (int d = 0; d < full_tensor_shape.dims(); ++d) {
    int start_offset = slice_dim == d ? slice_id * sharded_shape.dimensions(d) : 0;
    slice->set_start(d, start_offset);
    slice->set_length(d, sharded_shape.dimensions(d));
  }

  slice->SliceTensorShape(full_tensor_shape, sharded_tensor_shape);
}

void DeleteFiles(std::vector<std::string>& files) {
  auto* env = Env::Default();
  for (auto& file_name : files) {
    env->DeleteFile(file_name);
  }
}

// Returns whether "slice_spec" is a full slice, with respect to the full shape.
//
// This can happen say, when "slice_spec" is
// "TensorSlice(full_tensor_shape.dims())", or when it is "TensorSlice({{0,
// dim(0)}, ..., {0, dim(N)}})" -- a degenerate case we need to guard against.
bool IsFullSlice(const TensorSlice& slice_spec,
                 const TensorShape& full_tensor_shape) {
  if (slice_spec.IsFull()) {
    return true;
  } else {
    TensorShape sliced_shape;
    slice_spec.SliceTensorShape(full_tensor_shape, &sliced_shape).IgnoreError();
    return sliced_shape == full_tensor_shape;
  }
}

static std::random_device              rd;
static std::mt19937                    gen(rd());
static std::uniform_int_distribution<> dis(0, 15);
static std::uniform_int_distribution<> dis2(8, 11);

std::string generate_uuid_v4() {
    std::stringstream ss;
    int i;
    ss << std::hex;
    for (i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    for (i = 0; i < 4; i++) {
        ss << dis(gen);
    }
    ss << "4";
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << dis2(gen);
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    for (i = 0; i < 12; i++) {
        ss << dis(gen);
    };
    return ss.str();
}

template<typename T>
void memcpy_wrapper(xla::PjRtBuffer* buf_shard, Tensor& sharded_tensor) {
  auto literal = buf_shard->ToLiteral().ConsumeValueOrDie();
  std::memcpy(
	    sharded_tensor.flat<T>().data(), literal->untyped_data(),
	    xla::ShapeUtil::ByteSizeOf(buf_shard->on_device_shape()));
}

void CopyShardToTensor(
    xla::PjRtBuffer* buf_shard, Tensor& sharded_tensor, DataType dtype) {
  switch (dtype) {
    case DT_FLOAT: {
      memcpy_wrapper<float>(buf_shard, sharded_tensor);
      break;
    }

    case DT_INT64: {
      memcpy_wrapper<int64>(buf_shard, sharded_tensor);
      break;
    }

    default: {
      CHECK(0 && "Unhandled DataType");
    }
  }
}

} // namespace

CheckpointUtil::CheckpointUtil(std::string save_path, int64 max_to_keep)
    : save_path_(save_path), max_to_keep_(max_to_keep), uuid_str_(generate_uuid_v4()) {
  auto env = Env::Default();
  if (!env->FileExists(save_path).ok()) {
    CHECK(env->RecursivelyCreateDir(save_path).ok());
  }
}

void CheckpointUtil::Initialize(
    const xla::VariableSpecsMgr* var_spec_mgr, xla::PjRtClient* gpu_client) {
  gpu_client_ = gpu_client;
  var_spec_mgr_ = var_spec_mgr;
}

std::vector<tstring> CheckpointUtil::WriteTensorsToTempFiles(int64 global_step) {
  std::vector<tstring> input_prefixes;
  std::vector<std::unique_ptr<BundleWriter>> writers;
  int local_dev_count = gpu_client_->device_count();
  for (int s = 0; s < local_dev_count; ++s) {
    std::string input_prefix = strings::Printf(
        "%s/model.ckpt-%d_temp_%s/part-%05d-of-%05d", save_path_.c_str(), global_step,
        uuid_str_.c_str(), s, local_dev_count);
    writers.emplace_back(
        std::make_unique<BundleWriter>(Env::Default(), input_prefix));
    input_prefixes.push_back(input_prefix);
  }

  auto& name_spec_map = var_spec_mgr_->name_spec_map();
  for (auto& iter : name_spec_map) {
    auto& name = iter.first;
    auto& var_spec = iter.second;
    auto* d_buf = var_spec.d_buf;
    auto& arg_shape = var_spec.arg_shape;
    auto& global_local_dev_map = var_spec.global_local_dev_map;
    auto& global_dev_linear_slice_map = var_spec.global_dev_linear_slice_map;
    // Avoid duplicate saving
    std::unordered_set<int/*global_dev*/> saved;
    for (auto& pair : var_spec.start_offset_pairs_map) {
      int64 global_dev_id = pair.first;
      CHECK(global_local_dev_map.find(global_dev_id) != global_local_dev_map.end());
      CHECK(global_dev_linear_slice_map.find(global_dev_id) != global_dev_linear_slice_map.end());
      int64 local_dev_id = global_local_dev_map.at(global_dev_id);
      int64 slice_id = global_dev_linear_slice_map.at(global_dev_id);

      if (saved.count(slice_id) &&
          xla::ShapeUtil::Equal(d_buf->on_device_shape(), arg_shape)) continue;
      saved.insert(slice_id);
      auto* buf_shard = d_buf->gpu_buffer(global_dev_id);
      TensorShape full_shape, sharded_shape;
      int64 elem_size = xla::ShapeUtil::ByteSizeOfPrimitiveType(arg_shape.element_type());
      int64 total_elem = xla::ShapeUtil::ByteSizeOf(d_buf->on_device_shape()) / elem_size;
      int64 sharded_elem = xla::ShapeUtil::ByteSizeOf(arg_shape) / elem_size;
      // Flatten
      full_shape.AddDim(total_elem);
      xla::Shape flatten_arg_shape = xla::ShapeUtil::MakeShape(arg_shape.element_type(), {sharded_elem});
      TensorSlice slice(1);
      MakeTensorShapeAndSlice(
          full_shape, flatten_arg_shape, slice_id, &sharded_shape, &slice);
      auto dtype = EncodePrimitiveTypeAsDataType(
          buf_shard->on_device_shape().element_type()).ConsumeValueOrDie();
      Tensor sharded_tensor(dtype, sharded_shape);
      CopyShardToTensor(buf_shard, sharded_tensor, dtype);
      buf_shard->ClearHostCache();
      auto status = writers[local_dev_id]->AddSlice(name, full_shape, slice, sharded_tensor);
      VLOG(2) << status;
      CHECK(status.ok());
    }
  }

  for (auto& writer : writers) {
    auto status = writer->Finish();
    CHECK(status.ok());
  }

  return input_prefixes;
}

std::string CheckpointUtil::MergeShardedTempFiles(
    std::vector<tstring>& input_prefixes, int64 global_step) {
  std::string merged_prefix = absl::StrCat(save_path_, "model.ckpt-", global_step);
  auto env = Env::Default();
  auto s = MergeBundles(env, input_prefixes, merged_prefix);
  CHECK(s.ok());
  const std::string merged_dir(io::Dirname(merged_prefix));
  for (const std::string& input_prefix : input_prefixes) {
    const std::string dirname(io::Dirname(input_prefix));
    if (dirname == merged_dir) continue;
    auto status = env->DeleteDir(dirname);
    // For sharded save, only the first delete will go through and all
    // others will hit NotFound.  Use vlog to be less verbose.
    if (!status.ok()) VLOG(1) << status;
  }
  return merged_prefix;
}

void CheckpointUtil::Save(int64 global_step) {
  auto input_prefixes = WriteTensorsToTempFiles(global_step);
  auto ckpt_prefix = MergeShardedTempFiles(input_prefixes, global_step);
  if (max_to_keep_ > 0) {
    auto env = Env::Default();
    bool exists = false;
    for (auto& prefix : ckpt_prefix_queue_) {
      if (prefix == ckpt_prefix) {
        exists = true;
        break;
      }
    }

    if (exists) return;

    ckpt_prefix_queue_.push_back(ckpt_prefix);
    if (ckpt_prefix_queue_.size() > max_to_keep_) {
      auto pattern = ckpt_prefix_queue_.front();
      ckpt_prefix_queue_.pop_front();
      std::vector<std::string> matching_files;
      auto s = env->GetMatchingPaths(absl::StrCat(pattern, "*"), &matching_files);
      DeleteFiles(matching_files);
    }
    FlushPrefixQueueToFile();
  }
}

std::unique_ptr<Tensor> CheckpointUtil::LookupTensor(
    BundleReader* reader, std::string tensor_name,
    TensorShape& sharded_shape, TensorSlice& slice) {
  TensorShape restored_full_shape;
  DataType original_dtype;
  auto s = reader->LookupDtypeAndShape(
        tensor_name, &original_dtype, &restored_full_shape);
  CHECK(s.ok());

  VLOG(1) << "Restoring tensor " << tensor_name << " : "
          << restored_full_shape.num_elements();
  std::unique_ptr<Tensor> restored_tensor;
  if (IsFullSlice(slice, restored_full_shape)) {
    // Lookup the full tensor.
    restored_tensor = std::make_unique<Tensor>(original_dtype, restored_full_shape);
    s = reader->Lookup(tensor_name, restored_tensor.get());
  } else {
    // Lookup the slice.
    restored_tensor = std::make_unique<Tensor>(original_dtype, sharded_shape);
    s = reader->LookupSlice(tensor_name, slice, restored_tensor.get());
  }
  CHECK(s.ok());
  return restored_tensor;
}

void CheckpointUtil::Restore(int64 global_step) {
  ReadPrefixQueueFromFile();
  std::string prefix_string = absl::StrCat(save_path_, "model.ckpt-", global_step);
  BundleReader default_reader(Env::Default(), prefix_string);
  auto& name_spec_map = var_spec_mgr_->name_spec_map();
  for (auto& iter : name_spec_map) {
    const string& tensor_name = iter.first;
    auto& var_spec = iter.second;
    auto* d_buf = var_spec.d_buf;
    auto& arg_shape = var_spec.arg_shape;
    auto& global_local_dev_map = var_spec.global_local_dev_map;
    auto& global_dev_linear_slice_map = var_spec.global_dev_linear_slice_map;
 
    std::unordered_map<int/*global_slice id*/, std::unique_ptr<Tensor>> slice_tensor_cache;
    for (auto& pair : var_spec.start_offset_pairs_map) {
      int global_dev_id = pair.first;
      CHECK(global_local_dev_map.find(global_dev_id) != global_local_dev_map.end());
      CHECK(global_dev_linear_slice_map.find(global_dev_id) != global_dev_linear_slice_map.end());
      int64 local_dev_id = global_local_dev_map.at(global_dev_id);
      int64 slice_id = global_dev_linear_slice_map.at(global_dev_id);

      TensorShape full_shape, sharded_shape;
      // Flatten
      int64 elem_size = xla::ShapeUtil::ByteSizeOfPrimitiveType(arg_shape.element_type());
      int64 total_elem = xla::ShapeUtil::ByteSizeOf(d_buf->on_device_shape()) / elem_size;
      int64 sharded_elem = xla::ShapeUtil::ByteSizeOf(arg_shape) / elem_size;
      // Flatten
      full_shape.AddDim(total_elem);
      xla::Shape flatten_arg_shape = xla::ShapeUtil::MakeShape(arg_shape.element_type(), {sharded_elem});
 
      TensorSlice slice(1);
      MakeTensorShapeAndSlice(full_shape, flatten_arg_shape, slice_id, &sharded_shape, &slice);
      if (!slice_tensor_cache.count(slice_id)) {
        slice_tensor_cache[slice_id] = std::move(LookupTensor(
            &default_reader, tensor_name, sharded_shape, slice));
      }
      auto* device = gpu_client_->local_devices().at(local_dev_id);
      if (arg_shape.element_type() == xla::S64) {
        xla::DAPPLEBufferUtils::H2D(
            d_buf, arg_shape, local_dev_id, global_dev_id, gpu_client_,
            slice_tensor_cache[slice_id].get()->flat<int64>().data());
      } else {
        xla::DAPPLEBufferUtils::H2D(
            d_buf, arg_shape, local_dev_id, global_dev_id, gpu_client_,
            slice_tensor_cache[slice_id].get()->flat<float>().data());
      }
    }
  }
}

void CheckpointUtil::FlushPrefixQueueToFile() {
  std::string filename = absl::StrCat(save_path_, "checkpoint_server");
  auto env = Env::Default();
  if (env->FileExists(filename).ok()) {
    env->DeleteFile(filename);
  }
  std::ofstream os(filename, std::ios::app);
  for (auto& prefix : ckpt_prefix_queue_) {
    os << prefix << std::endl;
  }
}

void CheckpointUtil::ReadPrefixQueueFromFile() {
  std::string filename = absl::StrCat(save_path_, "checkpoint_server");
  auto env = Env::Default();
  if (env->FileExists(filename).ok()) {
    std::ifstream is(filename, std::ios::in);
    char line[2048];
    while (is.getline(line, sizeof(line))) {
      std::stringstream prefix_stream(line);
      ckpt_prefix_queue_.push_back(prefix_stream.str());
    }
  }
}

} // namespace dist_ckpt_utils
