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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"

namespace xla {

const std::vector<SplitMap>& DimDistSpec::derivation_table() const {
    return derivation_table_;
}

std::vector<SplitMap>* DimDistSpec::mutable_derivation_table() {
    return &derivation_table_;
}

void DimDistSpec::set_auto_dp_partition(const int auto_dp_partition) {
    auto_dp_partition_ = auto_dp_partition;
}

const int DimDistSpec::auto_dp_partition() const {
    return auto_dp_partition_;
}

void DimDistSpec::set_sharding_partition(const int sharding_partition) {
    sharding_partition_ = sharding_partition;
}

const int DimDistSpec::sharding_partition() const {
    return sharding_partition_;
}

const std::vector<SplitMapSet>& DimDistSpec::fused_solutions() const {
    return fused_solutions_;
}

std::vector<SplitMapSet>* DimDistSpec::mutable_fused_solutions() {
    return &fused_solutions_;
}

void DimDistSpec::ToProto(DimDistSpecProto& proto) const {
  proto.set_auto_dp_partition(auto_dp_partition_);
  proto.set_sharding_partition(sharding_partition_);
  proto.set_stride(stride_);
  proto.set_stride_on_dim(stride_on_dim_);
  proto.set_dim_to_slice(dim_to_slice_);
  proto.set_num_splits(num_splits_);
  proto.set_partial(partial_);
}

/*static*/
std::unique_ptr<DimDistSpec> DimDistSpec::CreateFromProto(
    const DimDistSpecProto& proto) {
  auto dim_dist_spec = absl::make_unique<DimDistSpec>();
  dim_dist_spec->set_auto_dp_partition(proto.auto_dp_partition());
  dim_dist_spec->set_sharding_partition(proto.sharding_partition());
  dim_dist_spec->set_partial(proto.partial());
  dim_dist_spec->set_layout_aware_partition(
      proto.stride(), proto.stride_on_dim(),
      proto.dim_to_slice(), proto.num_splits());
  return std::move(dim_dist_spec);
}

void DistSpec::ToProto(DistSpecProto& proto) const {
  for (auto& dim_spec : dim_specs_) {
    DimDistSpecProto dim_proto;
    dim_spec->ToProto(dim_proto);
    proto.add_dim_dist_specs()->Swap(&dim_proto);
  }
  proto.set_stage(stage_);
}

/*static*/
DistSpec DistSpec::CreateFromProto(
    const DistSpecProto& proto) {
  DistSpec dist_spec;
  for (const DimDistSpecProto& dim_dist_spec_proto : proto.dim_dist_specs()) {
    std::unique_ptr<DimDistSpec> dim_spec =
                        DimDistSpec::CreateFromProto(dim_dist_spec_proto);
    CHECK_NE(dim_spec.get(), nullptr);

    dist_spec.AddDimDistSpec(dim_spec);
  }

  return std::move(dist_spec);
}

void DistSpec::AddDimDistSpec(int64 stride, int64 stride_on_dim,
                              int partition_dim, int num_splits, bool partial) {
  std::unique_ptr<DimDistSpec> dim_spec = std::make_unique<DimDistSpec>();
  dim_spec->set_partial(partial);
  dim_spec->set_layout_aware_partition(
      stride, stride_on_dim, partition_dim, num_splits);
  this->AddDimDistSpec(dim_spec);
}

} // namespace
