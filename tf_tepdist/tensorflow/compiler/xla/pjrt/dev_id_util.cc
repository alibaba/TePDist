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

#include "tensorflow/compiler/xla/pjrt/dev_id_util.h"

namespace xla {

void CommDevManager::Build() {
  std::vector<int> trans_split_nums(split_nums_.size());
  std::vector<bool> trans_share_dev_flags(share_dev_flags_.size());
  std::vector<int64> trans_dev_id_bases(split_nums_.size());

  for (int i = 0; i < placement_layout_.size(); ++i) {
    int major = placement_layout_[i];
    int index = split_nums_.size() - major - 1;
    reverse_mapping_[i] = index;
    mapping_[index] = i;
  }

  trans_split_nums = transpose<int>(split_nums_, mapping_);
  trans_share_dev_flags = transpose<bool>(share_dev_flags_, mapping_);
  int64 base=1;
  for (int split_ordinal=trans_split_nums.size()-1; split_ordinal>=0; --split_ordinal) {
    if (trans_share_dev_flags[split_ordinal]) {
      trans_dev_id_bases[split_ordinal] = 0;
    } else {
      trans_dev_id_bases[split_ordinal] = base;
      base *= trans_split_nums[split_ordinal];
    }
  }
  dev_id_bases_ = transpose<int64>(trans_dev_id_bases, reverse_mapping_);
  total_dev_count_ = base;
  for (int split_ordinal=trans_split_nums.size()-1; split_ordinal>=0; --split_ordinal) {
    if (trans_share_dev_flags[split_ordinal]) {
      continue;
    }

    // 1.1. build an array of device groups for one split
    int group_num = total_dev_count_ / trans_split_nums[split_ordinal];
    std::shared_ptr<DevGroupArray> group_arr =
                      std::make_shared<DevGroupArray>(split_ordinal,
                                                      trans_split_nums.size(),
                                                      group_num);

    dev_group_arrays_[split_ordinal] = group_arr;
    // 1.1.1. build group id bases
    int64 group_id_base = 1;
    for (int k=trans_split_nums.size()-1; k>=0; --k) {
      if (k==split_ordinal) {
        // k is split index
        group_arr->group_id_bases_[k] = 0;
      } else if (trans_share_dev_flags[k]) {
        group_arr->group_id_bases_[k] = 0;
      } else {
        group_arr->group_id_bases_[k] = group_id_base;
        group_id_base *= trans_split_nums[k];
      }
    }

    // 1.1.2. build device group
    int64 dev_stride = trans_dev_id_bases[split_ordinal];
    CHECK(dev_stride > 0);
    CHECK(group_num == group_id_base);

    group_arr->dev_groups_.resize(group_num);
    for (int gid=0; gid<group_num; ++gid) {
      std::shared_ptr<DevGroup> group =
                          std::make_shared<DevGroup>(trans_split_nums[split_ordinal]);
      group_arr->dev_groups_[gid] = group;
      std::vector<int64> gaddr = linear_idx_to_addr(
                                          group_arr->group_id_bases_, gid);
      CHECK(gaddr[split_ordinal]==0);

      // trick: use split id bases to reinterpret group address to get
      //        device address
      int64 dev_id = addr_to_linear_idx(trans_dev_id_bases, gaddr);
      for (int k=0; k<trans_split_nums[split_ordinal]; ++k) {
        group->global_dev_ids_[k] = dev_id;
        dev_id += dev_stride;
      }
    }
  }

  // record worker id for each device
  CHECK(total_dev_count_ % num_workers_ == 0) << "total_dev_count_: " << total_dev_count_
      << ", num_workers_: " << num_workers_;
  dev_num_per_worker_ = total_dev_count_ / num_workers_;
  dev_worker_id_.resize(total_dev_count_);
  for (int i=0; i<total_dev_count_; ++i) {
    dev_worker_id_[i] = i / dev_num_per_worker_;
  }
}

std::vector<int64> CommDevManager::LinearIdxToAddr(int64 dev_idx) {
  std::vector<int64> trans_dev_id_bases = transpose(dev_id_bases_, mapping_);
  std::vector<int64> res = linear_idx_to_addr(trans_dev_id_bases, dev_idx);
  return transpose(res, reverse_mapping_);
}

std::vector<int64> CommDevManager::LinearIdxToAddrBySplitNums(int64 dev_idx) {
  std::vector<int> trans_split_nums = transpose(split_nums_, mapping_);
  std::vector<int64> res = linear_idx_to_addr_by_dims(trans_split_nums, dev_idx);
  return transpose(res, reverse_mapping_);
}

int64 CommDevManager::AddrToLinearIdx(const std::vector<int64>& addr) {
  return addr_to_linear_idx(dev_id_bases_, addr);
}

int64 CommDevManager::AddrToLinearIdxBySplitNums(const std::vector<int64>& addr) {
  std::vector<int> trans_split_nums = transpose(split_nums_, mapping_);
  std::vector<int64> trans_addr = transpose(addr, mapping_);
  return addr_to_linear_idx_by_dims(trans_split_nums, trans_addr);
}

} // namespace xla

