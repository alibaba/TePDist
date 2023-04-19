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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DEV_ID_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DEV_ID_UTIL_H_

#include <memory>
#include <vector>

#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/default/integral_types.h"

namespace xla {

using ::tensorflow::int64;

std::vector<int64>
linear_idx_to_addr_by_dims(const std::vector<int>& dims,
                   int64 dev_idx) {
  std::vector<int64> addr(dims.size(), 0);

  int i = dims.size() - 1;
  for (; i>=0; --i) {
    int64 size = dims[i];
    CHECK(size>0);
    addr[i] = dev_idx % size;
    dev_idx = dev_idx / size;
  }

  return std::move(addr);
}

std::vector<int64>
linear_idx_to_addr(const std::vector<int64>& bases, int64 dev_idx) {
  std::vector<int64> addr(bases.size(), 0);

  for (int i=0; i<bases.size(); ++i) {
    int64 base = bases[i];
    if (base>0) {
      addr[i] = dev_idx / base;
      dev_idx = dev_idx % base;
    }
  }

  return std::move(addr);
}

int64 addr_to_linear_idx(const std::vector<int64>& bases,
                         const std::vector<int64>& addr) {
  int linear_idx = 0;
  int i = bases.size() - 1;
  for (; i>=0; --i) {
    linear_idx += addr[i]*bases[i];
  }

  return linear_idx;
}

int64 addr_to_linear_idx_by_dims(const std::vector<int>& dims,
                                 const std::vector<int64>& addr) {
  std::vector<int64> dims_base(dims.size());
  int64 base = 1;
  for (int d = dims.size() - 1; d >=0; --d) {
    dims_base[d] = base;
    base *= dims[d];
  }

  return addr_to_linear_idx(dims_base, addr);
}

template<typename T>
std::vector<T> transpose(const std::vector<T>& array,
                           const std::vector<int>& mapping) {
  std::vector<T> res(array.size());
  for (int i = 0; i < array.size(); ++i) {
    res[i] = array[mapping[i]];
  }

  return res;
}

struct SplitId {
  SplitId()
  : ids_()
  , share_dev_flags_()
  , stage_split_ordinal_(-1)
  , micro_id_ordinal_(-1) {
  }
  SplitId(const std::vector<int64>& ids,
          const std::vector<bool>& share_dev_flags,
          int stage_split_ordinal)
  : ids_(ids)
  , share_dev_flags_(share_dev_flags)
  , stage_split_ordinal_(stage_split_ordinal) {
    micro_id_ordinal_ = -1;
    if (ids_.empty()) {
      share_dev_flags_.clear();
      stage_split_ordinal_ = -1;
    }
    for (int i=0; i<share_dev_flags_.size(); ++i) {
      if (share_dev_flags_[i]) {
        micro_id_ordinal_ = i;
        CHECK(stage_split_ordinal_ != micro_id_ordinal_);
        break;
      }
    }
  }

  bool operator<(const SplitId& rhs) const {
    if (ids_.size() < rhs.ids_.size()) {
      return true;
    } else if (ids_.size() > rhs.ids_.size()) {
      return false;
    }

    for (int i=0; i<ids_.size(); ++i) {
      if (ids_[i] < rhs.ids_[i]) {
        return true;
      } else if (ids_[i] > rhs.ids_[i]) {
        return false;
      }
    }

    return true;
  }
  int micro_id() const {
    if (micro_id_ordinal_ < 0) {
      return -1;
    } else {
      return ids_[micro_id_ordinal_];
    }
  }
  int stage_id() const {
    if (stage_split_ordinal_ < 0) {
      return -1;
    } else {
      return ids_[stage_split_ordinal_];
    }
  }

  std::string spmd_str() const {
    std::string spmd_str;
    for (int i=0; i<ids_.size(); ++i) {
      if (i==stage_split_ordinal_ || i==micro_id_ordinal_) {
        continue;
      }
      if (spmd_str.empty()) {
        spmd_str = "_spmd" + std::to_string(ids_[i]);
      } else {
        spmd_str += "_" + std::to_string(ids_[i]);
      }
    }

    return std::move(spmd_str);
  }

  std::string HumanReadableStr() const {
    std::string res;
    int sid = stage_id();
    int mid = micro_id();
    if (sid >= 0) {
      res += "_s" + std::to_string(sid);
    }
    if (mid >= 0) {
      res += "_m" + std::to_string(mid);
    }

    std::string spmd = spmd_str();
    res += spmd;
    return std::move(res);
  }

  std::vector<int64> ids_;
  // move out following 3 members later
  std::vector<bool> share_dev_flags_;
  int stage_split_ordinal_;
  int micro_id_ordinal_;
};

struct DevGroup {
  DevGroup(int group_size)
  : global_dev_ids_(group_size, 0) {
  }
  std::vector<int64> global_dev_ids_;
};

// all comm dev groups of a single split
struct DevGroupArray {
  DevGroupArray(int split_ordinal, int split_num, int group_num)
  : split_ordinal_(split_ordinal)
  , group_id_bases_(split_num)
  , dev_groups_(group_num) {
  }

  std::shared_ptr<DevGroup> FindDevGroup(const std::vector<int64>& split_id) const {
    VLOG(2) << "DevGroup::FindDevGroup: group_id_bases_.size() "
            << group_id_bases_.size()
            << ", split_ordinal_ " << split_ordinal_;
    int64 linear_idx = addr_to_linear_idx(group_id_bases_, split_id);
    return dev_groups_[linear_idx];
  }

  int split_ordinal_;
  std::vector<int64> group_id_bases_;

  // 1D represent multi dimension array of comm groups
  std::vector<std::shared_ptr<DevGroup>> dev_groups_;
};

struct CommDevManager {
  CommDevManager(const std::vector<int>& split_nums,
                 const std::vector<bool>& share_dev_flags,
                 const std::vector<int>& placement_layout,
                 int stage_split_ordinal,
                 int num_workers)
  : dev_group_arrays_(split_nums.size())
  , split_nums_(split_nums)
  , share_dev_flags_(share_dev_flags)
  , dev_id_bases_(placement_layout.size())
  , placement_layout_(placement_layout)
  , mapping_(placement_layout.size())
  , reverse_mapping_(placement_layout.size())
  , stage_split_ordinal_(stage_split_ordinal)
  , num_workers_(num_workers) {
    Build();
  }
  void Build();
  std::shared_ptr<DevGroup> FindDevGroup(const std::vector<int64>& split_id,
                                         int split_ordinal) const {
    auto group_arr = FindDevGroupArray(split_ordinal);
    CHECK(group_arr);
    std::vector<int64> trans_split_id = transpose<int64>(split_id, mapping_);
    return group_arr->FindDevGroup(trans_split_id);
  }

  std::shared_ptr<DevGroupArray> FindDevGroupArray(int split_ordinal) const {
    int trans_split_ordinal = reverse_mapping_[split_ordinal];
    CHECK(trans_split_ordinal < dev_group_arrays_.size());
    return dev_group_arrays_[trans_split_ordinal];
  }

  int64 GetCommRank(const std::vector<int64>& split_id, int split_ordinal) const {
    CHECK(share_dev_flags_[split_ordinal] == false);
    return split_id[split_ordinal];
  }

  int64 global_dev_id(const SplitId& split_id) const {
    if (!split_id.ids_.empty()) {
      int64 linear_idx = addr_to_linear_idx(dev_id_bases_, split_id.ids_);
      return linear_idx;
    } else {
      return 0;
    }
  }

  const std::vector<int>& split_nums() const { return split_nums_; }

  int64 total_dev_count() const {
    return total_dev_count_;
  }
  int64 dev_num_per_worker() const {
    return dev_num_per_worker_;
  }
  int worker_id(int64 gdev_id) const {
    return dev_worker_id_[gdev_id];
  }
  int worker_id(const SplitId& split_id) const {
    int64 gdev_id = global_dev_id(split_id);
    if (gdev_id<0) {
      // not place on GPU
      return 0;
    } else {
      return worker_id(gdev_id);
    }
  }
  int num_workers() const {
    return num_workers_;
  }
  int num_stages() const {
    if (stage_split_ordinal_ < 0) {
      return 1;
    } else {
      CHECK(stage_split_ordinal_ < split_nums_.size());
      return split_nums_[stage_split_ordinal_];
    }
  }

  int64 local_dev_id(int64 gdev_id) const {
    CHECK(dev_num_per_worker_>0);
    return gdev_id % dev_num_per_worker_;
  }
  int64 local_dev_id(const SplitId& split_id) const {
    int64 gdev_id = global_dev_id(split_id);
    return local_dev_id(gdev_id);
  }

  std::vector<int64> LinearIdxToAddr(int64 dev_idx);
  std::vector<int64> LinearIdxToAddrBySplitNums(int64 dev_idx);
  int64 AddrToLinearIdx(const std::vector<int64>& addr);
  int64 AddrToLinearIdxBySplitNums(const std::vector<int64>& addr);

  std::vector<std::shared_ptr<DevGroupArray>> dev_group_arrays_; // size is equal to split_nums_.size
  std::vector<int> split_nums_;
  std::vector<bool> share_dev_flags_;
  std::vector<int64> dev_id_bases_;
  std::vector<int> placement_layout_;
  std::vector<int> dev_worker_id_;  // record worker id for each device
  std::vector<int> reverse_mapping_; // transpose to the given layout
  std::vector<int> mapping_; // transpose to the row major layout
  int stage_split_ordinal_;
  int num_workers_;
  int64 total_dev_count_;
  int64 dev_num_per_worker_;
};

}

#endif // TENSORFLOW_COMPILER_XLA_PJRT_DEV_ID_UTIL_H_

