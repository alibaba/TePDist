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

#include "tensorflow/compiler/xla/service/parallel/hlo_strategy_spec.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "absl/strings/str_cat.h"


namespace xla {

using absl::StrAppend;

namespace {

Shape CheckAndGetShapeArray(const Shape& raw_shape) {
  if (raw_shape.IsTuple()) {
    Shape shape = raw_shape.tuple_shapes(0);
    // Check whether that the tuple shapes are all the same.
    for (auto& tuple_shape : raw_shape.tuple_shapes()) {
      CHECK(ShapeUtil::EqualIgnoringElementType(shape, tuple_shape));
    }
    return shape;
  } else {
    CHECK(raw_shape.IsArray()) << "Unhandled shape type!";
    return raw_shape;
  }
}

}

DimStrategy::DimStrategy(const Shape& raw_shape, int r,
                         int num_replicas, int64 stride_on_dim) {
  num_replicas_ = num_replicas;
  if (ShapeUtil::IsScalar(raw_shape) || r < 0) {
    stride_on_elements_ = 0;
    stride_on_dim_ = 0;
    return;
  }

  Shape shape = CheckAndGetShapeArray(raw_shape);
  int64 rank = shape.rank();
  CHECK(r < rank) << "Illegal partition dim: " << r << ", rank=" << rank;

  stride_on_dim_ = stride_on_dim < 1 ? shape.dimensions(r) : stride_on_dim;
  // %reduce-window.4937 = f32[1,2177,2]{2,1,0} reduce-window(
  //   f32[1,2176,2]{2,1,0} %convert.4932, f32[] %constant.4931),
  //   window={size=1x2176x1 pad=0_0x2176_0x0_0},
  if (stride_on_dim_ % num_replicas != 0) {
    stride_on_elements_ = 0;
    stride_on_dim_ = 0;
    return;
  }

  int64 stride = stride_on_dim_;
  for (int64 i = r+1; i < rank; ++i) {
    stride *= shape.dimensions(i);
  }
  stride_on_elements_ = stride;
  partition_dim_ = r;
}

void DimStrategy::ApplyToShape(const Shape& raw_shape) {
  Shape shape = CheckAndGetShapeArray(raw_shape);

  // Short circuit for some instructions (e.g., kReduceWindow) which only
  // partition_dim_ is recored and stride_on_elements_ is omitted due to
  // computation of stride_on_elements_ is instruction-dependent.
  if (stride_on_elements_ == -1) {
    CHECK(stride_on_dim_ == -1 &&
          partition_dim_ >= 0 && partition_dim_ < raw_shape.rank());
    return;
  }

  if (stride_on_elements_ == 0) {
    CHECK(stride_on_dim_ == 0);
    return;
  }

  int64 rank = shape.rank();
  int64 accum_stride = 1;
  for (int64 r = rank-1; r >= 0; --r) {
    int64 dim_size = shape.dimensions(r);
    accum_stride *= dim_size;
    if (accum_stride >= stride_on_elements_) {
      // The following Check *not always* work.
      // CHECK(accum_stride % stride_on_elements_ == 0);
      if (accum_stride % stride_on_elements_) {
        VLOG(0) << "WARNING: accum_stride=" << accum_stride
                << ", stride_on_elements_=" << stride_on_elements_;
        stride_on_elements_ = 0;
        stride_on_dim_ = 0;
        partition_dim_ = -1;
        num_replicas_ = 1;
        return;
      }
      partition_dim_ = r;
      stride_on_dim_ = dim_size / (accum_stride / stride_on_elements_);
      return;
    }
  }
}

std::string DimStrategy::ToString() const {
  if (stride_on_elements_ != -1 && stride_on_elements_ != 0) {
    CHECK(stride_on_dim_ > 0 && num_replicas_ > 1)
        << "stride_on_elements_=" << stride_on_elements_
        << ",stride_on_dim="<< stride_on_dim_
        << ",partition_dim_=" << partition_dim_
        << ",num_replicas_=" << num_replicas_;
  }
  string result = "";

  StrAppend(&result,
            replicated_ ? "replicated: true, " : "",
            "stride_on_elements:", std::to_string(stride_on_elements_),
            ",stride_on_dim:", std::to_string(stride_on_dim_),
            ",partition_dim:", std::to_string(partition_dim_),
            ",num_replicas:", std::to_string(num_replicas_),
            partial_ ? ",partial:true" : "");

  return result;
}

} // namespace xla

