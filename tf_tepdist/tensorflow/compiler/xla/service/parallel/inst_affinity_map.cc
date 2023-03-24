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

#include <stack>

#include "tensorflow/compiler/xla/service/parallel/inst_affinity_map.h"

#include "absl/strings/str_cat.h"

namespace xla {

using absl::StrAppend;

namespace {
  std::vector<std::vector<const HloInstruction*>> FindShortestPaths(
      const HloInstruction* from, const HloInstruction* to) {
    if (from == to) return {{from}};

    std::vector<std::vector<const HloInstruction*>> path_a, path_b, path_found;
    std::vector<std::vector<const HloInstruction*>>* pcheck = &path_a;
    std::vector<std::vector<const HloInstruction*>>* pwait = &path_b;

    pcheck->emplace_back(std::vector<const HloInstruction*>({from}));
    int path_length = 1;
    while (path_found.empty() && !pcheck->empty()) {
      for (std::vector<const HloInstruction*>& path : *pcheck) {
        const HloInstruction* tail = path[path_length - 1];
        for (const HloInstruction* user : tail->users()) {
          std::vector<const HloInstruction*> new_path(path);
          new_path.push_back(user);
          if (user == to) {
            VLOG(2) << "Found shortest path from " << from->name()
                    << " to " << to->name() << " at length " << path_length;
            path_found.emplace_back(new_path);
            break;
          }
          pwait->emplace_back(new_path);
        }
      }
      pcheck->clear();
      std::swap(pcheck, pwait);
      ++path_length;
    }
    return path_found;
  }
}

bool InstAffinityMap::CheckAffinity(const HloInstruction* from_inst,
                                    const HloInstruction* to_inst) const {
  if (affinity_map_.count(from_inst) == 0) return false;
  return affinity_map_.at(from_inst).count(to_inst) ? true : false;
}

void InstAffinityMap::AddAffinity(const HloInstruction* from_inst,
                                  const HloInstruction* to_inst) {
  affinity_map_[from_inst].insert(to_inst);
}

const std::set<std::pair<const HloInstruction*, const HloInstruction*>>
InstAffinityMap::AllAffinities(bool ignore_direction) const {
  std::set<std::pair<const HloInstruction*, const HloInstruction*>> res;
  
  for (auto pair : affinity_map_) {
    for (auto to_inst : pair.second) {
      // a->b and b->a affinity would be same if 'ignore_direction' set true.
      if (ignore_direction && res.count({to_inst, pair.first})) continue;
      res.insert({pair.first, to_inst});
    }
  }
  return res;
}

string InstAffinityMap::ToString(const HloInstruction* inst) const {
  if (affinity_map_.empty()) return string("affinity: none");
  string res = "";
  if (inst) {
    StrAppend(&res, "affinity of ", inst->name(), ": ");
    if (affinity_map_.at(inst).empty()) {
      StrAppend(&res, "none");
      return res;
    }
    for (auto to_inst : affinity_map_.at(inst)) {
      StrAppend(&res, to_inst->name(), ", ");
    }
    return res;
  }
  StrAppend(&res, "All inst affinities: \n");
  for (auto pair : affinity_map_) {
    StrAppend(&res, "  ", pair.first->name(), ": ");
    if (pair.second.empty()) {
      StrAppend(&res, "none\n");
    }
    for (auto to_inst : pair.second) {
      StrAppend(&res, to_inst->name(), ", ");
    }
    StrAppend(&res, "\n");
  }
  return res;
}

Status InOutAffinity::Set(const HloModule* module,
                          InstAffinityMap* affinity_map) {
  VLOG(1) <<"[InOutAffinityRule] " << name_;
  const int var_count = module->variable_map()->size();
  const HloComputation* entry = module->entry_computation();
  int input_var_offset = entry->num_parameters() - var_count;
  int output_var_offset = entry->root_instruction()->operand_count() - var_count;
  const HloInstruction* root = entry->root_instruction();
  for (int i = 0; i < var_count; ++i) {
    int p_idx = input_var_offset + i;
    int out_idx = output_var_offset + i;
    // Alias relationship
    const HloInstruction* param = entry->parameter_instruction(p_idx);
    const HloInstruction* updated = root->operand(out_idx);
    affinity_map->AddAffinity(param, updated);
    VLOG(1) << "setting affinity from input " << param->name()
            << " to output " << updated->name();
  }
  return Status::OK();
}

Status VarAuxAffinity::Set(const HloModule* module,
                           InstAffinityMap* affinity_map) {
  VLOG(1) <<"[VarAuxAffinityRule] " << name_;
  const int var_count = module->variable_map()->size();
  const HloComputation* entry = module->entry_computation();
  int input_var_offset = entry->num_parameters() - var_count;
  int output_var_offset = entry->root_instruction()->operand_count() - var_count;
  const HloInstruction* root = entry->root_instruction();
  std::unordered_set<const HloInstruction*> output_set;
  for (int i = 0; i < var_count; ++i) {
    output_set.insert(root->operand(output_var_offset + i));
  }
  for (int i = 0; i < var_count; ++i) {
    int out_idx = output_var_offset + i;
    const HloInstruction* updated = root->operand(out_idx);
    if (updated->user_count() == 1) {
      int p_idx = input_var_offset + i;
      const HloInstruction* param = entry->parameter_instruction(p_idx);
      VLOG(1) << "Variable output end " << updated->ToString()
              << " is updating for " << param->name();
      continue;
    }
    std::stack<const HloInstruction*> affined_insts;
    affined_insts.push(updated);
    while(!affined_insts.empty()) {
      const HloInstruction* cur = affined_insts.top();
      affined_insts.pop();
      VLOG(2) << "Auxiliary path inst check " << cur->ToString();
      for (HloInstruction* user : cur->users()) {
        if (user == root) continue;

        affinity_map->AddAffinity(cur, user);
        VLOG(2) << "setting inst affinity between inst " << cur->name()
                << " and user " << user->name();
        if (output_set.count(user) == 1) {
          VLOG(2) << "Auxiliary path end: " << updated->name()
                  << " output to " << user->name();
          continue;
        }
        affined_insts.push(user);
      }
    }
  }
  return Status::OK();
}

}  // namespace xla
