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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_HLO_STRATEGY_SPEC_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_HLO_STRATEGY_SPEC_H_

#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {

class DimStrategy;
using SharedDimStrategy = std::shared_ptr<DimStrategy>;
using UniqueDimStrategy = std::unique_ptr<DimStrategy>;

class DimStrategy {
 public:
  /*explicit*/ DimStrategy(const DimStrategy& strategy)
      : replicated_(strategy.replicated_),
        stride_on_elements_(strategy.stride_on_elements_),
        stride_on_dim_(strategy.stride_on_dim_),
        partition_dim_(strategy.partition_dim_),
        num_replicas_(strategy.num_replicas_),
        partial_(strategy.partial_) {}

  explicit DimStrategy(const DimDistSpec& dist_spec)
      : replicated_(false),
        stride_on_elements_(dist_spec.stride()),
        stride_on_dim_(dist_spec.stride_on_dim()),
        partition_dim_(dist_spec.partition_dim()),
        partial_(dist_spec.partial()),
        num_replicas_(dist_spec.num_splits()) {}

  explicit DimStrategy(int64 stride_on_elements = 0, int64 stride_on_dim = 0,
      int partition_dim = -1, bool partial = false,
      int num_replicas = 1, bool replicated = false)
      : replicated_(replicated),
        stride_on_elements_(stride_on_elements),
        stride_on_dim_(stride_on_dim), partition_dim_(partition_dim),
        partial_(partial), num_replicas_(num_replicas) {}

  static SharedDimStrategy MakeReplicate(int num_replicas) {
    return std::make_shared<DimStrategy>(0/*stride*/, 0/*stride_on_dim*/, -1/*partition_dim*/,
                       false/*partial*/, num_replicas/*num_replicas*/,
                       true/*replicated*/);
  }

  static SharedDimStrategy MakePartial(
      int num_shards, int partition_dim = -1, int64 stride_on_dim = 0) {
    return std::make_shared<DimStrategy>(0/*stride*/, stride_on_dim,
                    partition_dim, true/*partial*/, num_shards/*num_replicas*/);
  }

  DimStrategy& operator=(const DimStrategy& strategy) {
    replicated_ = strategy.replicated_;
    stride_on_elements_ = strategy.stride_on_elements_;
    stride_on_dim_ = strategy.stride_on_dim_;
    partition_dim_ = strategy.partition_dim_;
    num_replicas_ = strategy.num_replicas_;
    partial_ = strategy.partial_;
    return *this;
  }

  bool operator==(const DimStrategy& other) const {
    return partial_ == other.partial_ &&
           stride_on_elements_ == other.stride_on_elements_ &&
           stride_on_dim_ == other.stride_on_dim_ &&
           partition_dim_ == other.partition_dim_ &&
           num_replicas_ == other.num_replicas_;
  }
  bool operator!=(const DimStrategy& other) const {
    return partial_ != other.partial_ ||
           stride_on_elements_ != other.stride_on_elements_ ||
           stride_on_dim_ != other.stride_on_dim_ ||
           partition_dim_ != other.partition_dim_ ||
           num_replicas_ != other.num_replicas_;
  }

  bool operator<(const DimStrategy& other) const {
    if (partial_ == false && other.partial_ == true) {
      return true;
    } else if (partial_ == true && other.partial_ == false) {
      return false;
    } else if (stride_on_elements_ < other.stride_on_elements_) {
      return true;
    } else if (stride_on_elements_ > other.stride_on_elements_) {
      return false;
    } else if (partition_dim_ < other.partition_dim_) {
      return true;
    } else if (partition_dim_ > other.partition_dim_) {
      return false;
    } else if (num_replicas_ < other.num_replicas_) {
      return true;
    } else {
      return false;
    }
  }


  // Input raw shape possible types:
  // (1) Array, e.g., f32[10,15]
  // (2) Tuple, e.g., (f32[10,15], s32[10,15])
  // (3) Not handled types
  explicit DimStrategy(const Shape& raw_shape, int r,
                       int num_replicas, int64 stride_on_dim = -1);
  bool Match(const DimStrategy& s) const {
    if (stride_on_elements_ == -1 || s.stride_on_elements_ == -1)
      return partition_dim_ == s.partition_dim_;
    if (Glue() && s.Glue()) return true;
    return stride_on_elements_ == s.stride_on_elements_ && num_replicas_ == s.num_replicas_;
  }

  // Return true if this strategy is not *determined*.
  // Because replicated() == true indicates user annotated strategy which can
  // not be modified during strategy propagation, thus Glue() return false when
  // replicated() == true.
  bool Glue() const {
    return !replicated() && stride_on_elements_ == 0 && stride_on_dim_ == 0 && !partial_;
  }

  bool replicated() const { return replicated_; }
  void set_replicated(bool replicated) { replicated_ = replicated; }

  int64 stride_on_elements() const { return stride_on_elements_; }
  int64 size_on_elements() const {
    return stride_on_elements_ <= 0 ? stride_on_elements_
                                    : stride_on_elements_ / num_replicas_;
  }
  int64 stride_on_dim() const { return stride_on_dim_; }
  int64 size_on_dim() const {
    return stride_on_dim_ <= 0 ? stride_on_dim_
                               : stride_on_dim_ / num_replicas_;
  }
  int num_replicas() const { return num_replicas_; }
  int set_num_replicas(int num_replicas) { num_replicas_ = num_replicas; }
  int NumSlices() const { return (stride_on_dim_ > 0 && !partial_) ? num_replicas_ : 1; }
  int partition_dim() const { return partition_dim_; }

  bool IsPartial() const { return partial_; }
  void set_partial(bool partial) { partial_ = partial; }

  void ApplyToShape(const Shape& raw_shape);
  std::string ToString() const;

 private:
  bool partial_ = false;
  int64 stride_on_elements_ = 0;
  int64 stride_on_dim_ = 0;
  int partition_dim_ = -1; // Used when stride_on_elements_ is ambiguous.
  int num_replicas_ = 1;

  // Set to true only when xla_sharding.replicate() is set. For instruction with
  // replicated_=true, the stride_on_elements_ stride_on_dim_ fileds can only be 0.
  bool replicated_ = false;
};

bool operator==(const SharedDimStrategy& lhs, const SharedDimStrategy& rhs) = delete;
bool operator!=(const SharedDimStrategy& lhs, const SharedDimStrategy& rhs) = delete;
bool operator<(const SharedDimStrategy& lhs, const SharedDimStrategy& rhs) = delete;
bool operator<=(const SharedDimStrategy& lhs, const SharedDimStrategy& rhs) = delete;
bool operator>(const SharedDimStrategy& lhs, const SharedDimStrategy& rhs) = delete;

bool operator==(const UniqueDimStrategy& lhs, const UniqueDimStrategy& rhs) = delete;
bool operator!=(const UniqueDimStrategy& lhs, const UniqueDimStrategy& rhs) = delete;
bool operator<(const UniqueDimStrategy& lhs, const UniqueDimStrategy& rhs) = delete;
bool operator<=(const UniqueDimStrategy& lhs, const UniqueDimStrategy& rhs) = delete;
bool operator>(const UniqueDimStrategy& lhs, const UniqueDimStrategy& rhs) = delete;


struct DimStrategyPtrLess {
  bool operator()(const SharedDimStrategy& lhs,
                  const SharedDimStrategy& rhs) const {
    return *lhs < *rhs;
  }
};

class HLOStrategy {
 public:
  HLOStrategy()
  : dim_strategies_() {
  }

  void AddDimStrategy(const DimStrategy& dim_str) {
    dim_strategies_.emplace_back(dim_str);
  }
  const DimStrategy& dim_strategy(int idx) {
    CHECK(idx < dim_strategies_.size());
    return dim_strategies_[idx];
  }
  
  DimStrategy* mutable_dim_strategy(int idx) {
    CHECK(idx < dim_strategies_.size());
    return &dim_strategies_[idx];
  }
  std::string ToString() const {
    std::string str;
    for (int i=0; i<dim_strategies_.size(); ++i) {
      if (i>0) {
        str += ", " + dim_strategies_[i].ToString();
      } else {
        str += dim_strategies_[i].ToString();
      }
    }
    return std::move(str);
  }
  const std::vector<DimStrategy>& dim_strategies() const {
    return dim_strategies_;
  }
 private:
  std::vector<DimStrategy> dim_strategies_;
};

} // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_HLO_STRATEGY_SPEC_H_

