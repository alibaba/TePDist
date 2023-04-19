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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_INST_AFFINITY_MAP_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_INST_AFFINITY_MAP_H_

#include <string>
#include <unordered_map>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// To store and query instruction affinities, with direction sensitive.
// Bi-direction affinity rules should be controlled in Affinity.Set().
class InstAffinityMap
{
 public:
  bool CheckAffinity(const HloInstruction* from_inst,
                     const HloInstruction* to_inst) const;

  void AddAffinity(const HloInstruction* from_inst,
                   const HloInstruction* to_inst);
  const std::set<std::pair<const HloInstruction*, const HloInstruction*>>
  AllAffinities(bool ignore_direction = false) const;

  string ToString(const HloInstruction* inst = nullptr) const;

 private:
  std::unordered_map<const HloInstruction*, HloInstSet> affinity_map_;
};


// To define instruction affinity rules and set into map.
class AffinityRule
{
 public:
  virtual Status Set(const HloModule* module,
                     InstAffinityMap* affinity_map) = 0;
  virtual const string name() = 0;
};

// Setting input output affinity.
class InOutAffinity : public AffinityRule
{
 public:
  explicit InOutAffinity(string name = "input_output_affinity")
    : name_(name) {};

  Status Set(const HloModule* module, InstAffinityMap* affinity_map) override;
  const string name() override {
    return name_;
  }

  const std::set<std::pair<const HloInstruction*, const HloInstruction*>>
  AllAffinities(bool ignore_direction = false);

 private:
  const string name_;
};

// Setting variable and its auxiliaries affinity.
class VarAuxAffinity : public AffinityRule
{
 public:
  explicit VarAuxAffinity(string name = "var_aux_affinity")
    : name_(name) {};

  Status Set(const HloModule* module, InstAffinityMap* affinity_map) override;
  const string name() override {
    return name_;
  }

  const std::set<std::pair<const HloInstruction*, const HloInstruction*>>
  AllAffinities(bool ignore_direction = false);

 private:
  const string name_;
};

class InstAffinityMapBuilder
{
 public:
  // Add a rule into the builder,
  // the reference of added rule would be returned.
  template <typename T, typename... Args>
  T& AddRule(Args&&... args) {
    auto affinity_rule = new T(std::forward<Args>(args)...);
    VLOG(2) << "Adding affinity rule: " << affinity_rule->name();
    affinity_rules_.push_back(std::unique_ptr<T>(affinity_rule));
    return *affinity_rule;
  }

  // Run all affinity rules on given module, then return the corresponding
  // instruction affinity map. 
  std::unique_ptr<InstAffinityMap> Build(const HloModule* module) {
    InstAffinityMap affinity_map;
    VLOG(1) << "Builing inst affinity map with "
            << affinity_rules_.size() << " rules.";
    for (auto& rule : affinity_rules_) {
      VLOG(2) << "Setting rule " << rule->name();
      rule->Set(module, &affinity_map);
    }
    return std::make_unique<InstAffinityMap>(std::move(affinity_map));
  }

 private:
  std::vector<std::unique_ptr<AffinityRule>> affinity_rules_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_INST_AFFINITY_MAP_H_
