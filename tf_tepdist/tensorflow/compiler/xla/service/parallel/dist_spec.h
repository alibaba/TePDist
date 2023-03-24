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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_DIST_SPEC_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_DIST_SPEC_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

#include <unordered_map>
#include <set>
#include <string>
#include <vector>

namespace xla {

class HloInstruction;
class DimDistSpecProto;
class DistSpecProto;
 
using SplitMap = std::unordered_map<const HloInstruction*, int>;
using SplitMapSet = std::unordered_map<const HloInstruction*, std::set<int>>;

class DimDistSpec {
 public:
  std::vector<SplitMapSet>* mutable_fused_solutions();

  const std::vector<SplitMapSet>& fused_solutions() const;

  std::vector<SplitMap>* mutable_derivation_table();

  const std::vector<SplitMap>& derivation_table() const;

  void set_auto_dp_partition(const int auto_dp_partition);

  const int auto_dp_partition() const;

  void set_sharding_partition(const int sharding_partition);

  const int sharding_partition() const;

  void set_layout_aware_partition(int64 stride=0, int64 stride_on_dim=0,
                                  int slice_dim=-1, int num_splits=1) {
    stride_ = stride;
    stride_on_dim_ = stride_on_dim;
    if (stride_on_dim > 0) {
      CHECK(partial_ || (num_splits > 1 && stride_ > 0 && slice_dim >= 0
            && stride_on_dim % num_splits == 0 && stride % stride_on_dim == 0))
          << "invalid split, strie=" << stride
          << ", stride_on_dim=" << stride_on_dim
          << ", slice_dim=" << slice_dim
          << ", num_splits=" << num_splits
          << ", partial " << (partial_ ? 1 : 0);
    }
    dim_to_slice_ = slice_dim;
    num_splits_ = num_splits;
  }

  void clear_spmd_spec() {
    stride_ = 0;
    stride_on_dim_ = 0;
    dim_to_slice_ = -1;
    num_splits_ = 1;
    partial_ = false;
  }

  string ToString() const {
    return "stride=" + std::to_string(stride_) +
           ", stride_on_dim=" + std::to_string(stride_on_dim_) +
           ", num_split_=" + std::to_string(num_splits_) +
           ", split_dim=" + std::to_string(dim_to_slice_);
  }

  void operator=(const DimDistSpec& rhs) {
    stride_ = rhs.stride();
    stride_on_dim_ = rhs.stride_on_dim();
    dim_to_slice_ = rhs.partition_dim();
    num_splits_ = rhs.num_splits();
    partial_ = rhs.partial();
  }

  const int64 stride() const { return stride_; }
  const int64 stride_on_dim() const { return stride_on_dim_; }
  const int partition_dim() const { return dim_to_slice_; }
  const int num_splits() const { return num_splits_; }
  const bool partial() const { return partial_; }

  void set_partial(bool partial) { partial_ = partial; }

  void ToProto(DimDistSpecProto&) const;

  static std::unique_ptr<DimDistSpec> CreateFromProto(
      const DimDistSpecProto&);

 private:
  // NOTE: add more fields here, and getter/setters above
  double perf_time_;

  // Derivation cache for auto dp and sharding tasks
  std::vector<SplitMap> derivation_table_;

  std::vector<SplitMapSet> fused_solutions_;

  // auto dp decision
  int auto_dp_partition_ = -1;

  // sharding decision
  int sharding_partition_ = -1;

  // layout aware dp/sharding spec
  int64 stride_ = 0;
  int64 stride_on_dim_ = 0;
  int dim_to_slice_ = -1;
  int num_splits_ = 1;
  bool partial_ = false;
};

class DistSpec {
 public:
  DistSpec()
  : dim_specs_() {
  }
  DistSpec(int size) {
    for (int i = 0; i < size; ++i) {
      dim_specs_.emplace_back(std::make_unique<DimDistSpec>());
    }
  }
  DistSpec(const DistSpec& src) : stage_(src.stage_) {
    for (auto& dim_spec : src.dim_specs_) {
      auto new_dim_spec = std::make_unique<DimDistSpec>();
      *new_dim_spec = *dim_spec;
      dim_specs_.emplace_back(std::move(new_dim_spec));
    }
  }
  DistSpec(DistSpec& src) : stage_(src.stage_) {
    dim_specs_.insert(dim_specs_.end(),
                      std::make_move_iterator(src.dim_specs_.begin()),
                      std::make_move_iterator(src.dim_specs_.end()));
  }
  DistSpec(DistSpec&& src) : stage_(src.stage_) {
    dim_specs_.insert(dim_specs_.end(),
                      std::make_move_iterator(src.dim_specs_.begin()),
                      std::make_move_iterator(src.dim_specs_.end()));
  }
  DistSpec& operator=(DistSpec&& src) {
    dim_specs_.clear();
    dim_specs_.insert(dim_specs_.end(),
                      std::make_move_iterator(src.dim_specs_.begin()),
                      std::make_move_iterator(src.dim_specs_.end()));
    stage_ = src.stage_;
    return *this;
  }
  DistSpec operator=(DistSpec& src) {
    dim_specs_.clear();
    dim_specs_.insert(dim_specs_.end(),
                      std::make_move_iterator(src.dim_specs_.begin()),
                      std::make_move_iterator(src.dim_specs_.end()));
    stage_ = src.stage_;
    return *this;
  }
  DistSpec& operator=(const DistSpec& src) {
    for (auto& dim_spec : src.dim_specs_) {
      auto new_dim_spec = std::make_unique<DimDistSpec>();
      *new_dim_spec = *dim_spec;
      dim_specs_.emplace_back(std::move(new_dim_spec));
    }
    stage_ = src.stage_;
    return *this;
  }
  void AddDimDistSpec(std::unique_ptr<DimDistSpec>& dim_spec) {
    dim_specs_.emplace_back(std::move(dim_spec));
  }

  void AddDimDistSpec(int64 stride, int64 stride_on_dim,
                      int partition_dim, int num_splits, bool partial);

  const std::unique_ptr<DimDistSpec>& get_dim_spec(int idx) const {
    CHECK(idx>=0 && idx<dim_specs_.size()) << "idx: " << idx << ", dim_specs_ size: " << dim_specs_.size();
    return dim_specs_[idx];
  }
  std::unique_ptr<DimDistSpec>& get_dim_spec(int idx) {
    CHECK(idx>=0 && idx<dim_specs_.size()) << "idx: " << idx << ", dim_specs_ size: " << dim_specs_.size();
    return dim_specs_[idx];
  }
  const std::vector<std::unique_ptr<DimDistSpec>>& dim_specs() const {
    return dim_specs_;
  }
  void set_stage(int stage) { stage_ = stage; }
  const int stage() const { return stage_; }

  string ToString() const {
    string res = "";
    if (dim_specs_.empty()) {
      return std::move(res);
    }

    res += "{" + dim_specs_[0]->ToString() + "}";
    for (int i=1; i<dim_specs_.size(); ++i) {
      res += ", {" + dim_specs_[i]->ToString() + "}";
    }

    res += ", stage = " + std::to_string(stage_);
    return std::move(res);
  }
  bool is_empty() const {
    return dim_specs_.empty();
  }

  const size_t size() const { return dim_specs_.size(); }
  void ToProto(DistSpecProto&) const;
  static DistSpec CreateFromProto(const DistSpecProto&);
 private:
  std::vector<std::unique_ptr<DimDistSpec>> dim_specs_;
  int stage_ = -1;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_DIST_SPEC_H_
