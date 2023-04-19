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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_SKETCH_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_SKETCH_H_

#include <string>
#include <vector>

#include "coin/CbcModel.hpp"
#include "coin/OsiClpSolverInterface.hpp"

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"

namespace xla {

class SketchNode;

struct IlpStageExpr {
  IlpStageExpr() {}
  virtual std::string ModelStr() const = 0;
};

struct IlpStagePrimExpr : public IlpStageExpr {
  IlpStagePrimExpr()
  : IlpStageExpr() {}

  virtual bool has_scale() const = 0;
  virtual int64 scale() const = 0;
  virtual bool is_const() const {
    return false;
  }
  virtual int var_id() const {
    return -1;
  }
};

struct IlpStageConstExpr : public IlpStagePrimExpr {
  IlpStageConstExpr(int val)
  : IlpStagePrimExpr()
  , value_(val) {
  }
  virtual bool has_scale() const {
    return false;
  }
  virtual int64 scale() const {
    return 1;
  }
  virtual bool is_const() const {
    return true;
  }

  virtual std::string ModelStr() const {
    std::string res;
    res = std::to_string(value_);

    return std::move(res);
  }

  int value_;
};

struct IlpVarExpr : public IlpStagePrimExpr {
  IlpVarExpr(const HloInstruction* inst)
  : IlpStagePrimExpr()
  , var_id_(-1)
  , inst_(inst) {
  }
  virtual bool has_scale() const {
    return false;
  }
  virtual int64 scale() const {
    return 1;
  }
  virtual int var_id() const {
    return var_id_;
  }
  virtual bool is_stage_var() const {
    return false;
  }
  const HloInstruction* inst() const {
    return inst_;
  }

  int var_id_;
  const HloInstruction* inst_;
  double lower_bound_ = -COIN_DBL_MAX;
  double upper_bound_ = COIN_DBL_MAX;
};

struct IlpStageVarExpr : public IlpVarExpr {
  IlpStageVarExpr(const HloInstruction* inst)
  : IlpVarExpr(inst) {
    lower_bound_ = 0.0;
  }
  virtual bool is_stage_var() const {
    return true;
  }
  virtual std::string ModelStr() const {
    CHECK(inst_);
    std::string res;
    res = inst_->name();

    return std::move(res);
  }
};

struct IlpGroupStageVarExpr : public IlpVarExpr {
  IlpGroupStageVarExpr(int group_id)
  : IlpVarExpr(nullptr)
  , group_id_(group_id) {
    lower_bound_ = 0.0;
  }
  virtual bool is_stage_var() const {
    return true;
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = "group_" + std::to_string(group_id_);

    return std::move(res);
  }

  int group_id_;
};

struct IlpOneHotStageVarExpr : public IlpVarExpr {
  IlpOneHotStageVarExpr(const HloInstruction* inst, int stage_id)
  : IlpVarExpr(inst)
  , stage_id_(stage_id) {
  }
  virtual std::string ModelStr() const {
    CHECK(inst_);
    std::string res;
    res = inst_->name()+ "_onehot_" +std::to_string(stage_id_);

    return std::move(res);
  }

  int stage_id_;
};

struct IlpAcrossFlagVarExpr : public IlpVarExpr {
  IlpAcrossFlagVarExpr(const HloInstruction* inst, int input_idx)
  : IlpVarExpr(inst)
  , input_idx_(input_idx) {
  }
  virtual std::string ModelStr() const {
    CHECK(inst_);
    std::string res;
    res = inst_->name()+ "_in_" +std::to_string(input_idx_);

    return std::move(res);
  }

  int input_idx_;
};

// auxiliary variable
struct IlpAuxVarExpr : public IlpVarExpr {
  IlpAuxVarExpr(const std::string& pre_name)
  : IlpVarExpr(nullptr)
  , pre_name_(pre_name) {
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = pre_name_ + "_" + std::to_string(var_id_);

    return std::move(res);
  }

  std::string pre_name_;
};

struct IlpScaledExpr : public IlpStagePrimExpr {
  IlpScaledExpr(std::shared_ptr<IlpStagePrimExpr> op, int64 scale)
  : IlpStagePrimExpr()
  , operand_(op)
  , scale_(scale) {
  }
  virtual bool has_scale() const {
    return true;
  }
  virtual int64 scale() const {
    return scale_;
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = std::to_string(scale_) + "*(" + operand_->ModelStr() + ")";
    return std::move(res);
  }
  virtual int var_id() const {
    return operand_->var_id();
  }

  std::shared_ptr<IlpStagePrimExpr> operand_;
  int64 scale_;
};

struct IlpStageSubExpr : public IlpStageExpr {
  IlpStageSubExpr(std::shared_ptr<IlpStageExpr> left,
                  std::shared_ptr<IlpStageExpr> right)
  : IlpStageExpr()
  , left_(left)
  , right_(right)
  , id_(0) {
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = "(" + left_->ModelStr() + ")-(" + right_->ModelStr() + ")";
    return std::move(res);
  }

  std::shared_ptr<IlpStageExpr> left_;
  std::shared_ptr<IlpStageExpr> right_;

  int id_;
};

struct IlpStageSumExpr : public IlpStageExpr {
  IlpStageSumExpr()
  : IlpStageExpr() {
  }

  virtual std::string ModelStr() const {
    std::string res;
    if (operands_.empty()) {
      return res;
    }
    res = "(" + operands_[0]->ModelStr() + ")";
    for (int i=1; i<operands_.size(); ++i) {
      if (i%5 == 0) {
        res += "\n";
      }
      res += "+(" + operands_[i]->ModelStr() + ")";
    }
    return std::move(res);
  }

  std::vector<std::shared_ptr<IlpStagePrimExpr>> operands_;
};

struct IlpStageRelationExpr : public IlpStageExpr {
  IlpStageRelationExpr()
  : IlpStageExpr()
  , id_(-1) {
  }

  int id_;
};

struct IlpStageRelationRHSIntExpr : public IlpStageRelationExpr {
  IlpStageRelationRHSIntExpr(std::shared_ptr<IlpStageSumExpr> left, int64 right)
  : IlpStageRelationExpr()
  , left_(left)
  , right_(right) {
  }

  virtual double get_row_lower() const = 0;
  virtual double get_row_upper() const = 0;

  std::shared_ptr<IlpStageSumExpr> left_;
  int64 right_;
};

struct IlpStageEqIntExpr : public IlpStageRelationRHSIntExpr {
  IlpStageEqIntExpr(std::shared_ptr<IlpStageSumExpr> left, int64 right)
  : IlpStageRelationRHSIntExpr(left, right) {
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = "(" + left_->ModelStr() + ")=" + std::to_string(right_);
    return std::move(res);
  }
  virtual double get_row_lower() const {
    return (double)right_;
  }
  virtual double get_row_upper() const {
    return (double)right_;
  }
};

struct IlpStageLEIntExpr : public IlpStageRelationRHSIntExpr {
  IlpStageLEIntExpr(std::shared_ptr<IlpStageSumExpr> left, int64 right)
  : IlpStageRelationRHSIntExpr(left, right) {
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = "(" + left_->ModelStr() + ")<=" + std::to_string(right_);
    return std::move(res);
  }
  virtual double get_row_lower() const {
    return -COIN_DBL_MAX;
  }
  virtual double get_row_upper() const {
    return (double)right_;
  }
};

struct IlpStageGEIntExpr : public IlpStageRelationRHSIntExpr {
  IlpStageGEIntExpr(std::shared_ptr<IlpStageSumExpr> left, int64 right)
  : IlpStageRelationRHSIntExpr(left, right) {
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = "(" + left_->ModelStr() + ")>=" + std::to_string(right_);
    return std::move(res);
  }
  virtual double get_row_lower() const {
    return (double)right_;
  }
  virtual double get_row_upper() const {
    return COIN_DBL_MAX;
  }
};

struct IlpStageGrIntExpr : public IlpStageRelationRHSIntExpr {
  IlpStageGrIntExpr(std::shared_ptr<IlpStageSumExpr> left, int64 right)
  : IlpStageRelationRHSIntExpr(left, right) {
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = "(" + left_->ModelStr() + ")>" + std::to_string(right_);
    return std::move(res);
  }
  virtual double get_row_lower() const {
    return (double)right_;
  }
  virtual double get_row_upper() const {
    return COIN_DBL_MAX;
  }
};

struct IlpStageGEExpr : public IlpStageRelationExpr {
  IlpStageGEExpr(std::shared_ptr<IlpStageExpr> left,
                 std::shared_ptr<IlpStageExpr> right)
  : IlpStageRelationExpr()
  , left_(left)
  , right_(right) {
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = "(" + left_->ModelStr() + ")>=(" + right_->ModelStr() + ")";
    return std::move(res);
  }

  std::shared_ptr<IlpStageExpr> left_;
  std::shared_ptr<IlpStageExpr> right_;
};

struct HloPtrIntPairLess {
  bool operator()(const std::pair<const HloInstruction*, int>& lhs,
                  const std::pair<const HloInstruction*, int>& rhs) const {
    HloPtrComparator c;
    if (c(lhs.first, rhs.first)) {
      return true;
    } else if (c(rhs.first, lhs.first)) {
      return false;
    } else {
      return lhs.second < rhs.second;
    }
  }
};

template <typename ValueT>
using HloInstInt2ValMap = std::map<std::pair<const HloInstruction*, int/*input idx*/>,
                                   ValueT, HloPtrIntPairLess>;

class IlpStageModel {
 public:
  std::shared_ptr<IlpStageVarExpr> GetOrCreateStageVar(const HloInstruction* inst) {
    if (stage_var_map_.find(inst) != stage_var_map_.end()) {
      VLOG(2) << "found stage var for inst " << inst->name();
      return stage_var_map_.at(inst);
    }
    std::shared_ptr<IlpStageVarExpr> var = std::make_shared<IlpStageVarExpr>(inst);
    VLOG(2) << "build stage var for inst " << inst->name();
    var->var_id_ = all_vars_.size();
    var->lower_bound_ = (double)logic_stage_start_;
    var->upper_bound_ = (double)logic_stage_end_;

    all_vars_.emplace_back(var);
    stage_var_map_[inst] = var;

    return var;
  }

  std::shared_ptr<IlpOneHotStageVarExpr> GetOrCreateStageVar(
                                  const HloInstruction* inst, int stage_id) {
    std::pair<const HloInstruction*, int> key = std::make_pair(inst, stage_id);
    if (one_hot_stage_var_map_.find(key) != one_hot_stage_var_map_.end()) {
      return one_hot_stage_var_map_.at(key);
    }
    std::shared_ptr<IlpOneHotStageVarExpr> var =
                        std::make_shared<IlpOneHotStageVarExpr>(inst, stage_id);
    var->var_id_ = all_vars_.size();
    var->lower_bound_ = 0.0;
    var->upper_bound_ = 1.0;

    all_vars_.emplace_back(var);
    one_hot_stage_var_map_[key] = var;
    return var;
  }

  std::shared_ptr<IlpGroupStageVarExpr> GetOrCreateStageVar(int group_id) {
    if (group_stage_var_map_.find(group_id) != group_stage_var_map_.end()) {
      VLOG(2) << "found stage var for group " << group_id;
      return group_stage_var_map_.at(group_id);
    }
    std::shared_ptr<IlpGroupStageVarExpr> var =
                    std::make_shared<IlpGroupStageVarExpr>(group_id);
    VLOG(2) << "build stage var for group " << group_id;
    var->var_id_ = all_vars_.size();
    var->lower_bound_ = (double)logic_stage_start_;
    var->upper_bound_ = (double)logic_stage_end_;

    all_vars_.emplace_back(var);
    group_stage_var_map_[group_id] = var;

    return var;
  }

  std::shared_ptr<IlpAcrossFlagVarExpr> GetOrCreateAcrossVar(
                                  const HloInstruction* inst, int input_idx) {
    std::pair<const HloInstruction*, int> key = std::make_pair(inst, input_idx);
    if (across_flag_var_map_.find(key) != across_flag_var_map_.end()) {
      return across_flag_var_map_.at(key);
    }
    std::shared_ptr<IlpAcrossFlagVarExpr> var =
                        std::make_shared<IlpAcrossFlagVarExpr>(inst, input_idx);
    var->var_id_ = all_vars_.size();
    var->lower_bound_ = 0.0;
    var->upper_bound_ = 1.0;

    all_vars_.emplace_back(var);
    across_flag_var_map_[key] = var;
    return var;
  }

  std::shared_ptr<IlpStageConstExpr> GetOrCreateConst(int val) {
    if (const_expr_map_.find(val) != const_expr_map_.end()) {
      return const_expr_map_.at(val);
    }
    std::shared_ptr<IlpStageConstExpr> expr =
                                    std::make_shared<IlpStageConstExpr>(val);
    const_expr_map_[val] = expr;
    return expr;
  }

  std::shared_ptr<IlpStageSumExpr> GetOrCreateOptObj() {
    if (!opt_obj_) {
      opt_obj_ = std::make_shared<IlpStageSumExpr>();
    }

    return opt_obj_;
  }

  std::shared_ptr<IlpAuxVarExpr> CreateAuxVar(const std::string& pre_name,
                                              int lower_bound,
                                              int upper_bound) {
    std::shared_ptr<IlpAuxVarExpr> aux_var =
                                    std::make_shared<IlpAuxVarExpr>(pre_name);
    aux_var->var_id_ = all_vars_.size();
    aux_var->lower_bound_ = (double)lower_bound;
    aux_var->upper_bound_ = (double)upper_bound;

    all_vars_.emplace_back(aux_var);

    return aux_var;
  }

  void BuildOpUserConstraint(int op_stage, int user_stage,
                             std::shared_ptr<IlpStagePrimExpr> op_expr,
                             std::shared_ptr<IlpStagePrimExpr> user_expr,
                             bool affine);

  void BuildOpUserCost(int op_idx,
                       const HloInstruction* op,
                       const HloInstruction* user,
                       int op_stage, int user_stage,
                       std::shared_ptr<IlpStagePrimExpr> op_expr,
                       std::shared_ptr<IlpStagePrimExpr> user_expr);

  bool BuildFlopsLimit(
      const HloInstMap<int>& inst_stage_map,
      const HloInstMap<std::pair<int/*asap*/, int/*alap*/>>& inst_stage_range_map,
      const std::unordered_map<const HloInstruction*, int64>& inst_flops_map);

  std::string ExportToString() const;
  bool Solve(std::unordered_map<const HloInstruction*, int/*stage id*/>& inst_stage_map) const;

  void AddConstraint(std::shared_ptr<IlpStageRelationRHSIntExpr> constraint) {
    constraint->id_ = constraints_.size();
    constraints_.emplace_back(constraint);
  }
  void set_logic_stage_start(int logic_stage_start) {
    logic_stage_start_ = logic_stage_start;
  }
  void set_logic_stage_end(int logic_stage_end) {
    logic_stage_end_ = logic_stage_end;
  }
  void set_physical_stage_num(int physical_stage_num) {
    physical_stage_num_ = physical_stage_num;
  }
  void set_per_stage_flops(int64 per_stage_flops) {
    per_stage_flops_ = per_stage_flops;
  }
  void set_unbalanced_ratio(int unbalanced_ratio) {
    unbalanced_ratio_ = unbalanced_ratio;
  }
  const HloInstMap<std::shared_ptr<IlpStageVarExpr>>& stage_var_map() const {
    return stage_var_map_;
  }

  const std::vector<std::shared_ptr<IlpVarExpr>>& all_vars() const {
    return all_vars_;
  }

 private:
  std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>>
  BuildVarMap() const;

  void BuildOpUserCostInScope(int op_idx,
                              const HloInstruction* op,
                              const HloInstruction* user,
                              int op_stage, int user_stage,
                              std::shared_ptr<IlpStagePrimExpr> op_expr,
                              std::shared_ptr<IlpStagePrimExpr> user_expr);

  void BuildOpUserCostOutScope(int start_stage, int end_stage, int op_idx,
                               const HloInstruction* op,
                               const HloInstruction* user,
                               int op_stage,
                               std::shared_ptr<IlpStagePrimExpr> user_expr);

  HloInstMap<std::shared_ptr<IlpStageVarExpr>> stage_var_map_;
  HloInstInt2ValMap<std::shared_ptr<IlpOneHotStageVarExpr>> one_hot_stage_var_map_;
  HloInstInt2ValMap<std::shared_ptr<IlpAcrossFlagVarExpr>> across_flag_var_map_;
  std::map<int, std::shared_ptr<IlpStageConstExpr>> const_expr_map_;
  std::map<int/*group id*/, std::shared_ptr<IlpGroupStageVarExpr>> group_stage_var_map_;

  std::vector<std::shared_ptr<IlpVarExpr>> all_vars_;

  // ILP model:
  // 1. optimization object
  std::shared_ptr<IlpStageSumExpr> opt_obj_;

  // 2. constraints
  std::vector<std::shared_ptr<IlpStageRelationRHSIntExpr>> constraints_;


  int logic_stage_start_;
  int logic_stage_end_;
  int physical_stage_num_ = 0;
  int64 per_stage_flops_ = 0;
  int unbalanced_ratio_ = 0;
};

struct AccumCompInfo {
  void AddAccumCompNodes(const std::set<SketchNode*>& comp_nodes) {
    accum_comp_nodes_.insert(comp_nodes.begin(), comp_nodes.end());
  }

  void EvalAccumFlopCount();
  void CalcAccumRatio(const int64 total_flop_count) {
    CHECK(total_flop_count>0);
    relative_accum_ = (float)accum_flop_count_ / (float)total_flop_count;
  }

  std::string ToString() const;

  std::set<SketchNode*> accum_comp_nodes_; // count compute-sensitive nodes only
  int64 accum_flop_count_ = 0;             // accumulated flop count
  float relative_accum_ = 0.0;
};

struct OpGroupInfo {
 public:
  OpGroupInfo(int id)
  : id_(id)
  , pre_ids_()
  , direct_pre_ids_()
  , mirror_post_ids_(nullptr)
  , mirror_direct_post_ids_(nullptr) {
  }

  void AddInst(const HloInstruction* inst) {
    insts_.insert(inst);
  }
  void AddPreGroupId(int id) {
    pre_ids_.insert(id);
  }
  void AddDirectPreGroupId(int id) {
    direct_pre_ids_.insert(id);
  }


  std::string ToString() const {
    std::string res;
    res += "id: " + std::to_string(id_) + "\ninst list:\n";
    for (auto* inst : insts_) {
      res += "  " + inst->name() + "\n";
    }

    int i = 0;
    res += "\ndirect pre group ids:\n";
    for (auto direct_pre_id : direct_pre_ids_) {
      res += " " + std::to_string(direct_pre_id) + ",";
      ++i;
      if (i%20 == 0) {
        res += "\n";
      }
    }

    res += "\npre group ids:\n";
    i = 0;
    for (auto pre_id : pre_ids_) {
      res += " " + std::to_string(pre_id) + ",";
      ++i;
      if (i%20 == 0) {
        res += "\n";
      }
    }

    if (mirror_direct_post_ids_) {
      res += "\ndirect post group ids:\n";
      i = 0;
      for (auto direct_post_id : *mirror_direct_post_ids_) {
        res += " " + std::to_string(direct_post_id) + ",";
        ++i;
        if (i%20 == 0) {
          res += "\n";
        }
      }
    }

    if (mirror_post_ids_) {
      res += "\npost group ids:\n";
      i = 0;
      for (auto post_id : *mirror_post_ids_) {
        res += " " + std::to_string(post_id) + ",";
        ++i;
        if (i%20 == 0) {
          res += "\n";
        }
      }
    }

    return std::move(res);
  }

  bool InPreGroups(int id) const {
    if (pre_ids_.find(id) != pre_ids_.end()) {
      return true;
    } else {
      return false;
    }
  }

  const std::set<int>& pre_ids() const {
    return pre_ids_;
  }

  const std::set<int>& direct_pre_ids() const {
    return direct_pre_ids_;
  }

  const std::set<int>* mirror_direct_post_ids() const {
    return mirror_direct_post_ids_;
  }

  int id() const {
    return id_;
  }

  void set_mirror_post_ids(const std::set<int>* mirror_post_ids) {
    mirror_post_ids_ = mirror_post_ids;
  }

  void set_mirror_direct_post_ids(const std::set<int>* mirror_direct_post_ids) {
    mirror_direct_post_ids_ = mirror_direct_post_ids;
  }

 private:
  int id_;

  HloInstSet insts_;
  std::set<int> pre_ids_;
  std::set<int> direct_pre_ids_;
  const std::set<int>* mirror_post_ids_;        // mirror
  const std::set<int>* mirror_direct_post_ids_; // mirror
};

class SketchNode {
 public:
  SketchNode(const HloInstruction* core_instr, bool is_backward);
  virtual ~SketchNode() { }
  void AddInput(SketchNode* input) {
    inputs_.insert(input);
    input->AddUserOnly(this);
  }
  void EraseInput(SketchNode* input) {
    inputs_.erase(input);
    input->EraseUserOnly(this);
  }
  /*
  void AddUser(SketchNode* user) {
    user->AddInput(this);
  }
  void EraseUser(SketchNode* user) {
    user->EraseInput(this);
  }
  */
  const std::set<SketchNode*>& inputs() const {
    return inputs_;
  }
  const std::set<SketchNode*>& users() const {
    return users_;
  }
  int64 user_count() const {
    return users_.size();
  }
  const std::string& name() const {
    return core_instr_->name();
  }

  const HloInstruction* core_instr() const {
    return core_instr_;
  }

  void AddInstruction(HloInstruction* instruction) {
    instructions_.insert(instruction);
  }

  HloMutableInstSet& instructions() {
    return instructions_;
  }
  bool not_connected() const {
    return (users_.empty() && inputs_.empty());
  }
  bool is_tiny() const {
    /*
    if (core_instr_->opcode() == HloOpcode::kCustomCall &&
        core_instr_->custom_call_target() == "__cublas$gemm") {
      return false;
    } else if (core_instr_->opcode() == HloOpcode::kDot) {
      return false;
    }
    return true;
    */

    CHECK(flop_count_ >= 0);
    return (flop_count_ == 0);
  }
  void EvalCost(std::unordered_map<const HloInstruction*, int64>& inst_flops_map);
  const std::set<SketchNode*>& pre_nodes() {
    return pre_nodes_;
  }
  const std::set<SketchNode*>& post_nodes() {
    return post_nodes_;
  }
  const std::set<SketchNode*>& pre_comp_sens_nodes() const {
    return pre_comp_sens_nodes_;
  }
  const std::set<SketchNode*>& post_comp_sens_nodes() {
    return post_comp_sens_nodes_;
  }
  void AddPreNode(SketchNode* pre_node) {
    pre_nodes_.insert(pre_node);
  }
  void AddPreNodes(const std::set<SketchNode*>& pre_nodes) {
    pre_nodes_.insert(pre_nodes.begin(), pre_nodes.end());
  }
  void AddPostNode(SketchNode* post_node) {
    post_nodes_.insert(post_node);
  }
  void AddPostNodes(const std::set<SketchNode*>& post_nodes) {
    post_nodes_.insert(post_nodes.begin(), post_nodes.end());
  }
  void AddPreCompSensNode(SketchNode* pre_comp_sens_node) {
    CHECK(!pre_comp_sens_node->is_tiny());
    pre_comp_sens_nodes_.insert(pre_comp_sens_node);
  }
  void AddPreCompSensNodes(const std::set<SketchNode*>& pre_comp_sens_nodes) {
    pre_comp_sens_nodes_.insert(pre_comp_sens_nodes.begin(), pre_comp_sens_nodes.end());
  }
  void AddPostCompSensNode(SketchNode* post_comp_sens_node) {
    CHECK(!post_comp_sens_node->is_tiny());
    post_comp_sens_nodes_.insert(post_comp_sens_node);
  }
  void AddPostCompSensNodes(const std::set<SketchNode*>& post_comp_sens_nodes) {
    post_comp_sens_nodes_.insert(post_comp_sens_nodes.begin(), post_comp_sens_nodes.end());
  }
  int64 pre_node_count() const {
    return pre_nodes_.size();
  }
  int64 post_node_count() const {
    return post_nodes_.size();
  }
  int64 pre_comp_sens_node_count() const {
    return pre_comp_sens_nodes_.size();
  }
  int64 post_comp_sens_node_count() const {
    return post_comp_sens_nodes_.size();
  }
  int64 flop_count() const {
    return flop_count_;
  }
  int64 asap_comp_sens_rank() const {
    return asap_comp_sens_rank_;
  }
  int64 alap_comp_sens_rank() const {
    return alap_comp_sens_rank_;
  }
  void AccuPreFlopCount(int64 flop_count) {
    accum_pre_flop_count_ += flop_count;
  }
  void AccuPostFlopCount(int64 flop_count) {
    accum_post_flop_count_ += flop_count;
  }
  int64 accum_pre_flop_count() const {
    return accum_pre_flop_count_;
  }
  int64 accum_post_flop_count() const {
    return accum_post_flop_count_;
  }
  void set_asap_rank(int64 asap_rank) {
    asap_rank_ = asap_rank;
  }
  void set_alap_rank(int64 alap_rank) {
    alap_rank_ = alap_rank;
  }
  void set_asap_comp_sens_rank(int64 asap_comp_sens_rank) {
    asap_comp_sens_rank_ = asap_comp_sens_rank;
  }
  void set_alap_comp_sens_rank(int64 alap_comp_sens_rank) {
    alap_comp_sens_rank_ = alap_comp_sens_rank;
  }
  float relative_accum_pre() const {
    return relative_accum_pre_;
  }
  float relative_accum_post() const {
    return relative_accum_post_;
  }
  float relative_self() const {
    return relative_self_;
  }
  void CalcRelativeAccuPre(const int64 total_flop_count) {
    relative_accum_pre_ = (float)accum_pre_flop_count_ / (float)total_flop_count;
  }
  void CalcRelativeAccuPost(const int64 total_flop_count) {
    relative_accum_post_ = (float)accum_post_flop_count_ / (float)total_flop_count;
  }
  void CalcRelativeSelf(const int64 total_flop_count) {
    relative_self_ = (float)flop_count_ / (float)total_flop_count;
  }
  int FreedomDegree() const {
    return alap_comp_sens_rank_ - asap_comp_sens_rank_;
  }
  bool is_critical_node() const {
    if (is_tiny()) {
      return false;
    }
    return asap_comp_sens_rank_ == alap_comp_sens_rank_;
  }
  bool is_strict_critical_node() const {
    if (is_tiny()) {
      return false;
    }
    return asap_rank_ == alap_rank_;
  }
  bool is_backward() const {
    return is_backward_;
  }
  int stage_id() const {
    return stage_id_;
  }
  void set_stage_id(int stage_id) {
    stage_id_ = stage_id;
  }
  void set_as_root() {
    is_root_ = true;
    is_backward_ = true;
  }
  SketchNode* mp_user() const {
    return mp_user_;
  }
  void set_mp_user(SketchNode* mp_user) {
    mp_user_ = mp_user;
  }
  int64 mp_acc_post_flops() const {
    return mp_acc_post_flops_;
  }
  void set_mp_acc_post_flops(int64 mp_acc_post_flops) {
    mp_acc_post_flops_ = mp_acc_post_flops;
  }
  void AccMpAccPostFlops(int64 flops) {
    mp_acc_post_flops_ += flops;
  }
  std::shared_ptr<AccumCompInfo> pre_accum_comp_info() const {
    return pre_accum_comp_info_;
  }
  std::shared_ptr<AccumCompInfo> post_accum_comp_info() const {
    return post_accum_comp_info_;
  }
  void set_pre_accum_comp_info(std::shared_ptr<AccumCompInfo> pre_accum_comp_info) {
    pre_accum_comp_info_ = pre_accum_comp_info;
  }
  void set_post_accum_comp_info(std::shared_ptr<AccumCompInfo> post_accum_comp_info) {
    post_accum_comp_info_ = post_accum_comp_info;
  }
  std::string Dump() const {
    std::string ret("node: ");
    ret += name() + "\n";
    for (auto* inst : instructions_) {
      ret += "  " + inst->name() + "\n";
    }
    return ret;
  }
  int asap_stage_id() const {
    return asap_stage_id_;
  }
  int alap_stage_id() const {
    return alap_stage_id_;
  }
  void set_asap_stage_id(int asap_stage_id) {
    asap_stage_id_ = asap_stage_id;
  }
  void set_alap_stage_id(int alap_stage_id) {
    alap_stage_id_ = alap_stage_id;
  }
 private:
  void AddUserOnly(SketchNode* user) {
    users_.insert(user);
  }
  void EraseUserOnly(SketchNode* user) {
    users_.erase(user);
  }
  const HloInstruction* core_instr_;
  std::set<SketchNode*> inputs_;
  std::set<SketchNode*> users_;
  HloMutableInstSet instructions_;

  std::set<SketchNode*> pre_nodes_;
  std::set<SketchNode*> post_nodes_;
  std::set<SketchNode*> pre_comp_sens_nodes_;   // count compute-sensitive nodes only
  std::set<SketchNode*> post_comp_sens_nodes_;  // count compute-sensitive nodes only

  int64 accum_pre_flop_count_ = 0;  // accumulated pre flop count
  int64 accum_post_flop_count_ = 0; // accumulated post flop count
  float relative_accum_pre_ = 0.0;
  float relative_accum_post_ = 0.0;
  float relative_self_ = 0.0;
  int64 flop_count_ = -1;       // flop count of this node
  int64 asap_rank_ = 0;         // as soon as possible scheduling rank
  int64 alap_rank_ = INT64_MAX; // as late as possible scheduling rank
  int64 asap_comp_sens_rank_ = 0;         // as soon as possible scheduling rank (compute-sensitive nodes only)
  int64 alap_comp_sens_rank_ = INT64_MAX; // as late as possible scheduling rank (compute-sensitive nodes only)
  bool is_backward_ = false;
  bool is_root_ = false;
  int stage_id_ = -1;
  static constexpr int64 kFmaFlops = 2;

  SketchNode* mp_user_ = nullptr;     // user in main path
  int64 mp_acc_post_flops_ = 0;  // main path accumulated post flop count

  std::shared_ptr<AccumCompInfo> pre_accum_comp_info_;
  std::shared_ptr<AccumCompInfo> post_accum_comp_info_;

  int asap_stage_id_ = -1;
  int alap_stage_id_ = -1;
};

class SketchStage {
 public:
  SketchStage(int stage_id)
  : stage_id_(stage_id)
  , nodes_()
  , total_flop_count_(0)
  , relative_flop_(0.0) {
  }
  ~SketchStage() {}
  void AddNode(SketchNode* node) {
    nodes_.emplace_back(node);
    node->set_stage_id(stage_id_);
    total_flop_count_ += node->flop_count();
    relative_flop_ += node->relative_self();
  }
  float relative_flop() const {
    return relative_flop_;
  }
  std::string Dump() const {
    std::string ret("stage:\n");
    ret += "  relative flop: " + std::to_string(relative_flop_) + "\n";
    ret += "  nodes list:\n";
    for (auto* node : nodes_) {
      ret += "    " + node->name() + "\n";
    }
    return std::move(ret);
  }
  int stage_id() const {
    return stage_id_;
  }
  const std::vector<SketchNode*>& nodes() const {
    return nodes_;
  }
 private:
  int stage_id_;
  std::vector<SketchNode*> nodes_;
  int64 total_flop_count_;
  float relative_flop_;
};

class InstStage {
 public:
  InstStage(int stage_id)
  : stage_id_(stage_id)
  , insts_()
  , total_flop_count_(0)
  , relative_flop_(0.0) {
  }

  ~InstStage() {}

  void AddInst(const SketchNode* sketch_node) {
    const HloInstruction* inst = sketch_node->core_instr();
    insts_.emplace_back(inst);
    total_flop_count_ += sketch_node->flop_count();
    relative_flop_ += sketch_node->relative_self();
  }
  void AddInst(const HloInstruction* inst) {
    insts_.emplace_back(inst);
    /*
    total_flop_count_ += sketch_node->flop_count();
    relative_flop_ += sketch_node->relative_self();
    */
  }
  float relative_flop() const {
    return relative_flop_;
  }
  std::string Dump() const {
    std::string ret("stage:\n");
    ret += "  relative flop: " + std::to_string(relative_flop_) + "\n";
    ret += "  nodes list:\n";
    for (auto* inst : insts_) {
      ret += "    " + inst->name() + "\n";
    }
    return std::move(ret);
  }
  int stage_id() const {
    return stage_id_;
  }
  const std::vector<const HloInstruction*>& insts() const {
    return insts_;
  }
 private:
  int stage_id_;
  std::vector<const HloInstruction*> insts_;
  int64 total_flop_count_;
  float relative_flop_;
};

class GraphSketch {
 public:
  enum VisitState { kVisiting, kVisited };

  GraphSketch(HloModule* module, int physical_stage_num);
  virtual ~GraphSketch() { }

  static std::unique_ptr<GraphSketch> BuildGraphSketch(
      HloModule* module, int physical_stage_num);

  // debug purpose
  static std::unique_ptr<GraphSketch> BuildFineGrainedSketch(
                                            HloModule* module,
                                            int physical_stage_num,
                                            const HloInstSet* inst_scope=nullptr);

  HloComputation* SketchComputation() { return computation_; }

  bool StagePlan();
  void Optimize();
  void EvalCost();
  std::string Dump(const HloInstMap<int>* inst_stage_map = nullptr);
  int64 node_count() const {
    return nodes_.size();
  }
  void CalcPreComputeCost(const std::vector<SketchNode*>& post_order,
                          bool is_forward);
  void CalcPostComputeCost(const std::vector<SketchNode*>& post_order,
                           bool is_forward);
  int64 comp_sens_node_count() const {
    return comp_sens_nodes_.size();
  }
  const std::vector<std::unique_ptr<SketchNode>>& nodes() const {
    return nodes_;
  }

  std::vector<const HloInstruction*> FindCriticalInsts();
  std::string DumpStages(
    const HloInstMap<std::pair<int/*asap*/, int/*alap*/>>& inst_stage_range_map,
    const std::vector<SketchNode*>* sketch_nodes) const;
 private:
  // methods to create graph sketch
  void AddNode(std::unique_ptr<SketchNode> node);
  void AbsorbSingleUserNodes();
  void ClusterTinyNodes();
  void MergeNodes(SketchNode* src, SketchNode* tgt);
  bool IsFusable(const SketchNode* src, const SketchNode* tgt) const;
  std::vector<SketchNode*> CollectFusableNodes() const;

  void RemoveIsolatedNodes();

  std::vector<SketchNode*> MakePostOrder() const;

  void ComputePostOrder(std::vector<SketchNode*>& post_order, SketchNode* root,
                  absl::flat_hash_map<SketchNode*, VisitState>& visited) const;

  std::unordered_map<HloInstruction*, SketchNode*>& inst_node_map() {
    return inst_node_map_;
  }

  int64 ComputeTransfersBetween(SketchNode* src, SketchNode* dst);

  void set_root(SketchNode* root) {
    root_ = root;
  }


  // methods to build pipeline stages at sketch level
  void DeterminePrePostNodes();
  bool ForwardPlan(int forward_stage_num,
                     const std::vector<SketchNode*>& sketch_nodes,
                     const std::set<std::pair<int, int>>& group_depend,
                     const HloInstSet& inst_scope);
  void ForwardTinyNodePlan(std::vector<SketchStage>& stages);
  bool BackwardPlan(int physical_stage_num,
                    const std::vector<SketchNode*>& sketch_nodes,
                    const HloInstMap<int>& mirror_stage_map,
                    const std::set<std::pair<int, int>>& group_depend,
                    const HloInstSet& inst_scope);
  bool AssignStageSafely(const HloInstSet& inst_scope);
  HloInstMap<std::pair<int/*asap*/, int/*alap*/>>
  InferStageRangeByFlops(int stage_num,
                         int start_stage,
                         bool is_forward,
                         const std::vector<SketchNode*>& sketch_nodes);

  HloInstMap<std::pair<int/*asap*/, int/*alap*/>>
  InferStageRangeByContext(const HloInstSet& inst_scope,
                           int stage_num,
                           int start_stage);
  void FindMainPath();

  // methods to build pipeline stages at instruction level
  void FindNearestStages(
          const HloInstSet& inst_scope,
          const HloInstMap<int>& inst_stage_map,
          std::unordered_map<const HloInstruction*, int>& pre_stage_map,
          std::unordered_map<const HloInstruction*, int>& succ_stage_map);
  bool PlaceInstsToNeighbor(const HloInstSet& inst_scope,
                            int start_stage, int logic_stage_num);
  bool MapUnstagedInsts(int start_stage, const HloInstSet& inst_scope);

  bool DetectCycle();
  void RecordStagePlanIntoInsts();

  void BuildDeterminedByMap(
        const HloInstSet& inst_scope,
        std::unordered_map<const HloInstruction*, const HloInstruction*>& determined_by_user,
        std::unordered_map<const HloInstruction*, const HloInstruction*>& determined_by_op);
  bool BuildIlpStageModel(
        const HloInstSet& inst_scope,
        int start_stage, int end_stage,
        const HloInstMap<int>& inst_stage_map,
        const HloInstMap<std::pair<int/*asap*/, int/*alap*/>>& inst_stage_range_map,
        const HloInstMap<int>& mirror_stage_map,
        const std::map<int, std::shared_ptr<OpGroupInfo>>& graph_groups,
        const std::set<std::pair<int, int>>& group_depend,
        bool is_forward,
        IlpStageModel& ilp_model); // build pipeline ILP model
  bool BuildOpGroupInfo(const HloInstSet& inst_scope,
                        std::map<int, std::shared_ptr<OpGroupInfo>>& op_groups);

  HloComputation* computation_;
  const std::map<int, string>* const variable_map_;
  HloModule* module_; // duplicated information with computation_ and variable_map_

  std::vector<std::unique_ptr<SketchNode>> nodes_;
  std::vector<SketchNode*> comp_sens_nodes_;
  std::unordered_map<HloInstruction*, SketchNode*> inst_node_map_;
  SketchNode* root_;
  int64 total_flop_count_ = 0;
  int64 forward_flop_count_ = 0;
  int64 backward_flop_count_ = 0;
  int64 forward_per_stage_flops_ = 0;
  int64 backward_per_stage_flops_ = 0;
  int unbalanced_ratio_ = 0;
  std::vector<SketchNode*> main_path_nodes_;  // post order critical nodes
  std::vector<SketchNode*> critical_nodes_;  // post order critical nodes

  HloInstMap<int> inst_stage_map_;
  int physical_stage_num_;
  int logic_stage_num_;
  std::unordered_set<const HloInstruction*> backward_insts_;
  std::unordered_set<const HloInstruction*> forward_insts_;

  std::unique_ptr<HloReachabilityMap> reachability_;


  HloInstMap<int> inst_stage_map_v2_;
  HloInstMap<std::pair<int/*asap*/, int/*alap*/>> inst_stage_range_map_; // record stage range for instruction
  std::unordered_map<const HloInstruction*, int64> inst_flops_map_;

  std::map<int, std::shared_ptr<OpGroupInfo>> forward_op_groups_;
  std::map<int, std::shared_ptr<OpGroupInfo>> backward_op_groups_;

  std::map<int/*group*/, int/*physical stage*/> group_stage_map_;
};


}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_SKETCH_H_

