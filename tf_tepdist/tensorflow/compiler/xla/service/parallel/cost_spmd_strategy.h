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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_COST_SPMD_STRATEGY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_COST_SPMD_STRATEGY_H_

#include <set>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/parallel/dist_spec.h"
#include "tensorflow/compiler/xla/service/parallel/hlo_strategy_spec.h"
#include "tensorflow/compiler/xla/service/parallel/inst_affinity_map.h"
#include "tensorflow/core/platform/macros.h"

#include "absl/container/flat_hash_map.h"

namespace xla {

class InstCone;

bool
is_compute_intensive(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kDot ||
      inst->opcode() == HloOpcode::kConvolution) {
    return true;
  } else if (inst->opcode() == HloOpcode::kCustomCall &&
             inst->custom_call_target() == "__cublas$gemm") {
    return true;
  }

  return false;
}

std::shared_ptr<InstCone> ExtractCone(
      const HloInstruction* cone_root,
      const HloInstSet& inst_scope,
      const std::unordered_set<const HloInstruction*>& exclude_insts);

struct InstCosts {
  InstCosts(const HloInstruction* inst, bool is_cone_root)
  : inst_(inst)
  , is_cone_root_(is_cone_root) { // initialize is_cone_root_, it may be changed later.
  }

  struct OneStrategyCost {
    OneStrategyCost(const SharedDimStrategy& strategy)
    : strategy_(strategy)
    , cost_(0)
    , op_strtg_idx_() {
    }
    int op_strtg_idx(int op_idx) const {
      CHECK(op_idx < op_strtg_idx_.size()) << "op idx: " << op_idx
                << ", op_strtg_idx_ size: " << op_strtg_idx_.size();
      return op_strtg_idx_[op_idx];
    }
    SharedDimStrategy strategy_;
    int64             cost_;
    std::vector<int>  op_strtg_idx_;
  };

  std::string ToString() const;

  bool is_cone_root() const {
    return is_cone_root_;
  }

  bool set_is_cone_root(bool is_cone_root) {
    is_cone_root_ = is_cone_root;
  }

  bool is_mult_strategies() const {
    return (strtg_costs_.size() > 1);
  }

  std::unique_ptr<OneStrategyCost>& get_one_cost(int idx) {
    CHECK(idx < strtg_costs_.size()) << "idx: " << idx
                    << ", strtg_costs_ size: " << strtg_costs_.size();
    return strtg_costs_[idx];
  }

  const HloInstruction* inst_;
  bool is_cone_root_ = false;
  std::vector<std::unique_ptr<OneStrategyCost>> strtg_costs_;
};

struct ConeStrategy {
  std::string ToString() const;
  // build input cost, include both inter-cone and context to cone costs
  void BuildExpInStrtgMap(
      const std::unordered_set<const HloInstruction*>& insts);
  void BuildSelfCost(const InstCone& cone, bool share_dev, int split_num);

  void BuildInputCost(
      const InstCone& cone,
      const HloInstMap<SharedDimStrategy>& ctx_strtg_map,
      const std::unordered_map<const HloInstruction*,
                               std::shared_ptr<InstCone>>& cone_map,
      int split_num,
      bool share_dev);

  // build cone to context cost
  void BuildOutToCtxCost(
      const InstCone& cone,
      const HloInstMap<SharedDimStrategy>& ctx_strtg_map,
      const std::unordered_map<const HloInstruction*,
                               std::shared_ptr<InstCone>>& cone_map,
      int split_num,
      bool share_dev);

  const SharedDimStrategy& FindStrategy(const HloInstruction* inst) const {
    CHECK(strategy_map_.find(inst) != strategy_map_.end());
    return strategy_map_.at(inst);
  }

  int id_ = 0;
  HloInstMap<SharedDimStrategy> strategy_map_;

  // cost inside the cone
  int64 self_cost_ = 0;

  // cost between cone and context: not include by cone costs
  int64 ctx_in_cost_ = 0;
  int64 ctx_out_cost_ = 0;

  // costs between cones
  std::map<std::pair<int/*input cone id*/, int/*input strategy id*/>, int64> cone_in_costs_;

  // expected input strategies, key is outside of this cone
  std::unordered_map<const HloInstruction*, SharedDimStrategy> exp_in_strtgs_;

  float total_cost_ = 0.0;  // weighted by neighbors
  float prob_ = 0.0;   // TODO(lansong): probability to apply this strategy
};

struct InstCone {
  InstCone(const HloInstruction* root_inst)
  : id_(0)
  , root_inst_(root_inst)
  , insts_()
  , strategies_() {
  }
  void AddInst(const HloInstruction* inst) {
    insts_.insert(inst);
  }
  std::string ToString() const;
  void AddConeStrategy(std::unique_ptr<ConeStrategy>& cone_strtg) {
    cone_strtg->id_ = strategies_.size();
    strategies_.emplace_back(std::move(cone_strtg));
  }
  void InitializeProb() {
    float init_prob = 1.0/strategies_.size();
    for (auto& one_strtg : strategies_) {
      one_strtg->prob_ = init_prob;
    }
  }
  void CollectInputInsts();
  void BuildSelfCost(bool share_dev, int split_num);

  void BuildInputCost(
      const HloInstMap<SharedDimStrategy>& ctx_strtg_map,
      const std::unordered_map<const HloInstruction*,
                               std::shared_ptr<InstCone>>& cone_map,
      int split_num,
      bool share_dev);

  // build cone to context cost
  void BuildOutToCtxCost(
      const HloInstMap<SharedDimStrategy>& ctx_strtg_map,
      const std::unordered_map<const HloInstruction*,
                               std::shared_ptr<InstCone>>& cone_map,
      int split_num,
      bool share_dev);
  const std::vector<std::unique_ptr<ConeStrategy>>& strategies() const {
    return strategies_;
  }
  const std::unique_ptr<ConeStrategy>& strategy(int str_id) const {
    return strategies_[str_id];
  }
  int strategy_num() const {
    return strategies_.size();
  }
  int64 strategy_cost(int str_id) const {
    CHECK(str_id < strategies_.size());
    return strategies_[str_id]->self_cost_;
  }

  int id_;
  const HloInstruction*                       root_inst_;
  std::unordered_set<const HloInstruction*>   insts_;
  std::unordered_set<const HloInstruction*>   input_insts_; // outside of this cone
 private:
  std::vector<std::unique_ptr<ConeStrategy>>  strategies_;
};

struct SubGraphStrategy {
  int id_ = 0;
  int sub_graph_id_ = 0;
  SharedDimStrategy head_strategy_;
  SharedDimStrategy tail_strategy_;
  HloInstMap<SharedDimStrategy> strategy_map_;
  int64 self_cost_ = 0;
  int64 accu_cost_ = 0;  // accumulated cost
  std::shared_ptr<SubGraphStrategy> pre_sub_strategy_;

  HloInstMap<SharedDimStrategy> acc_strategy_map_;
};

class GraphStrategy {
 public:
  GraphStrategy(int split_num, bool share_dev)
  : strategy_map_()
  , split_num_(split_num)
  , share_dev_(share_dev) {
  }
  GraphStrategy(const GraphStrategy& src)
  : strategy_map_(src.strategy_map_)
  , split_num_(src.split_num_)
  , share_dev_(src.share_dev_) {
  }

  ~GraphStrategy() {}
  GraphStrategy& operator=(const GraphStrategy& src) {
    strategy_map_ = src.strategy_map_;
    split_num_ = src.split_num_;
    share_dev_ = src.share_dev_;
  }
  bool SetInstStrategy(const HloInstruction* inst,
                       const SharedDimStrategy& strtg);

  SharedDimStrategy strategy(const HloInstruction* inst) const {
    const HloInstMap<SharedDimStrategy>&
    cur_str_map = strategy_map();
    if (cur_str_map.find(inst) != cur_str_map.end()) {
      return cur_str_map.at(inst);
    } else {
      return std::make_shared<DimStrategy>();
    }
  }

  HloInstMap<SharedDimStrategy>& strategy_map() {
    return strategy_map_;
  }
  const HloInstMap<SharedDimStrategy>& strategy_map() const {
    return strategy_map_;
  }
  int split_num() const {
    return split_num_;
  }
  bool share_dev() const {
    return share_dev_;
  }
  void AddUserAnnotatedInst(const HloInstruction* inst) {
    user_annotated_insts_.insert(inst);
  }
  void Finalize();
  const HloInstMap<HLOStrategy>& hlo_strategy_map() const {
    return hlo_strategy_map_;
  }
  
  HloInstMap<HLOStrategy>* mutable_hlo_strategy_map() {
    return &hlo_strategy_map_;
  }
  std::string ToString();
 private:
  HloInstMap<SharedDimStrategy> strategy_map_;

  HloInstSet user_annotated_tensors_;
  HloInstSet user_annotated_insts_;

  HloInstMap<HLOStrategy> hlo_strategy_map_;

  int split_num_;  // not used, remove it later
  bool share_dev_; // not used, remove it later
};

struct ConeStrategyManager {
  void AddAdjCones(int src_id, int tgt_id) {
    adjacency_map_[src_id].insert(std::make_pair(tgt_id, false));
    adjacency_map_[tgt_id].insert(std::make_pair(src_id, true));
  }

  int MatrixAddrToOffset(int src_str_count, int src_str_id, int usr_str_id) const {
    return usr_str_id * src_str_count + src_str_id;
  }

  void OffsetToMatrixAddr(int src_str_count, int linear_addr,
                          int& src_str_id, int& usr_str_id) {
    CHECK(src_str_count>=1);
    src_str_id = linear_addr % src_str_count;
    usr_str_id = linear_addr / src_str_count;
  }

  int64 FindCost(int src_str_count, int src_cone_id, int src_str_id,
                 int usr_cone_id, int usr_str_id) {
    std::pair<int, int> key = std::make_pair(src_cone_id, usr_cone_id);
    CHECK(inter_cost_map_.find(key) != inter_cost_map_.end())
          << "src_cone_id: " << src_cone_id << ", usr_cone_id: " << usr_cone_id;
    std::vector<int64/*cost*/>& cost_matrix = inter_cost_map_[key];

    int offset = MatrixAddrToOffset(src_str_count, src_str_id, usr_str_id);
    return cost_matrix[offset];
  }

  // cone id may be -1, that is a virtual cone which means context
  std::unordered_map<int/*cone id*/,
                     std::set<std::pair<int/*cone id*/,
                                        bool/*is src*/>>> adjacency_map_;
  std::map<std::pair<int/*cone id*/, int/*strategy id*/>, int64> intra_cost_map_;
  std::map<std::pair<int/*source cone id*/, int/*user cone id*/>,
                     std::vector<int64/*cost*/>/*1D stores matrix*/> inter_cost_map_;
  std::unordered_map<const HloInstruction*, std::shared_ptr<InstCone>> cone_map_;
};

struct ILPExpr {
  ILPExpr() {}
  virtual std::string ModelStr() const = 0;
};

struct ILPPrimExpr : public ILPExpr {
  ILPPrimExpr()
  : ILPExpr() {}

  virtual bool has_scale() const = 0;
  virtual int64 scale() const = 0;
  virtual int var_id() const = 0;
};

struct ILPVarExpr : public ILPPrimExpr {
  ILPVarExpr()
  : ILPPrimExpr() {}
  virtual bool has_scale() const {
    return false;
  }
  virtual int64 scale() const {
    return 1;
  }
};

struct ILPSelfVarExpr : public ILPVarExpr {
  ILPSelfVarExpr(int cone_id, int strtg_id)
  : ILPVarExpr()
  , cone_id_(cone_id)
  , strtg_id_(strtg_id)
  , id_(0) {
    if (cone_id>=0) {
      name_ = "c" + std::to_string(cone_id);
    } else {
      // context is a special cone, its id is -1
      name_ = "ctx" + std::to_string(-cone_id);
    }

    name_ += "s" + std::to_string(strtg_id);
  }
  void set_cost(int64 cost) {
    cost_ = cost;
  }
  virtual std::string ModelStr() const {
    std::string human_read;
    if (cone_root_) {
      human_read = cone_root_->name() + "_";
    }
    return human_read + name_;
  }
  virtual int var_id() const {
    return id_;
  }

  int cone_id_;
  int strtg_id_;
  std::string name_;
  int64 cost_;

  // ------------------------------------------------
  // debug purpose
  const HloInstruction* cone_root_ = nullptr;
  // ------------------------------------------------
  int id_;   // for CBC solver
};

struct ILPEdgeVarExpr : public ILPVarExpr {
  ILPEdgeVarExpr(std::shared_ptr<ILPSelfVarExpr> src_var,
                 std::shared_ptr<ILPSelfVarExpr> usr_var)
  : ILPVarExpr()
  , src_var_(src_var)
  , usr_var_(usr_var)
  , cost_(0)
  , id_(0) {
  }

  void set_cost(int64 cost) {
    cost_ = cost;
  }
  virtual std::string ModelStr() const {
    std::string res;
    if (src_cone_root_) {
      if (src_var_->cone_root_ == nullptr) {
        VLOG(0) << "src_cone_root: " << src_cone_root_->name();
      }
      
      CHECK(src_var_->cone_root_) << "src cone id: " << src_var_->cone_id_;
      CHECK(src_cone_root_ == src_var_->cone_root_)
            << "src cone root: " << src_cone_root_->name()
            << ", src var: " << src_var_->cone_root_->name();
      //res += src_cone_root_->name() + "_to_";
    } else {
      //res += "ctx_to_";
    }

    if (usr_cone_root_) {
      if (usr_var_->cone_root_ == nullptr) {
        VLOG(0) << "usr_cone_root: " << usr_cone_root_->name();
        VLOG(0) << "this: " << this;
      }
      CHECK(usr_var_->cone_root_) << "usr cone id: " << usr_var_->cone_id_
            << ", usr_cone_root_: " << usr_cone_root_ << ", usr_cone_root name: " << usr_cone_root_->name();
      CHECK(usr_cone_root_ == usr_var_->cone_root_)
            << "usr cone root: " << usr_cone_root_->name()
            << ", usr var: " << usr_var_->cone_root_->name();
      //res += usr_cone_root_->name() + ":";
    } else {
      //res += "ctx:";
    }
    res += src_var_->ModelStr() + "_to_" + usr_var_->ModelStr();
    return std::move(res);
  }
  virtual int var_id() const {
    return id_;
  }
  std::shared_ptr<ILPSelfVarExpr> src_var_;
  std::shared_ptr<ILPSelfVarExpr> usr_var_;
  int64 cost_;

  // ------------------------------------------------
  // debug purpose
  const HloInstruction* src_cone_root_ = nullptr;
  const HloInstruction* usr_cone_root_ = nullptr;
  // ------------------------------------------------
  int id_;  // for CBC solver
};

struct ILPScaledExpr : public ILPPrimExpr {
  ILPScaledExpr(std::shared_ptr<ILPPrimExpr> op, int64 scale)
  : ILPPrimExpr()
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

  std::shared_ptr<ILPPrimExpr> operand_;
  int64 scale_;
};

struct ILPSumExpr : public ILPExpr {
  ILPSumExpr()
  : ILPExpr() {
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

  std::vector<std::shared_ptr<ILPPrimExpr>> operands_;
};

struct ILPRelationExpr : public ILPExpr {
  ILPRelationExpr()
  : ILPExpr() {
  }
};

struct ILPEqualIntExpr : public ILPRelationExpr {
  ILPEqualIntExpr(std::shared_ptr<ILPSumExpr> left, int64 right)
  : ILPRelationExpr()
  , left_(left)
  , right_(right)
  , id_(0) {
  }
  virtual std::string ModelStr() const {
    std::string res;
    res = "(" + left_->ModelStr() + ")=" + std::to_string(right_);
    return std::move(res);
  }

  std::shared_ptr<ILPSumExpr> left_;
  int64 right_;

  int id_;
};

struct ILPModel {
  std::shared_ptr<ILPSelfVarExpr>
  FindStrSelfVar(int cone_id, int str_id) {
    CHECK(str_self_var_map_.find(cone_id) != str_self_var_map_.end());
    std::vector<std::shared_ptr<ILPSelfVarExpr>>& self_vars =
                                              str_self_var_map_[cone_id];

    CHECK(str_id>=0 && str_id < self_vars.size());
    return self_vars[str_id];
  }

  std::string ExportToString() const;
  std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>> BuildVarMap();
  std::unordered_map<int/*cone id*/, int/*strategy id*/> Solve();
  std::shared_ptr<ILPSelfVarExpr> BuildSelfVar(int cone_id, int str_id) {
    std::shared_ptr<ILPSelfVarExpr> self_var =
                        std::make_shared<ILPSelfVarExpr>(cone_id, str_id);
    self_var->id_ = all_str_self_vars_.size();
    all_str_self_vars_.emplace_back(self_var);
    return self_var;
  }

  std::shared_ptr<ILPEdgeVarExpr> BuildEdgeVar(
                           std::shared_ptr<ILPSelfVarExpr> src_self_var,
                           std::shared_ptr<ILPSelfVarExpr> usr_self_var) {
    int src_cone_id = src_self_var->cone_id_;
    int src_str_id = src_self_var->strtg_id_;
    int usr_cone_id = usr_self_var->cone_id_;
    int usr_str_id = usr_self_var->strtg_id_;
    auto src_str_key = std::make_pair(src_cone_id, src_str_id);
    auto usr_str_key = std::make_pair(usr_cone_id, usr_str_id);


    if (str_edge_map_no_dup_.find(src_str_key) != str_edge_map_no_dup_.end()) {
      if (str_edge_map_no_dup_[src_str_key].find(usr_str_key) !=
          str_edge_map_no_dup_[src_str_key].end()) {
        return str_edge_map_no_dup_[src_str_key][usr_str_key];
      }
    }

    std::shared_ptr<ILPEdgeVarExpr> edge_var =
                  std::make_shared<ILPEdgeVarExpr>(src_self_var, usr_self_var);
    edge_var->id_ = all_str_edge_vars_.size();
    all_str_edge_vars_.emplace_back(edge_var);

    str_edge_map_[src_str_key].emplace_back(edge_var);

    str_edge_map_[usr_str_key].emplace_back(edge_var);

    str_edge_map_no_dup_[src_str_key][usr_str_key] = edge_var;

    return edge_var;
  }

  // auxiliary variables for building ILP model
  // 1. self variables
  std::unordered_map<int/*cone id*/,
           std::vector<std::shared_ptr<ILPSelfVarExpr>>> str_self_var_map_;
  std::map<std::pair<int/*cone id*/, int/*strategy id*/>,
           std::shared_ptr<ILPSelfVarExpr>> str_id_self_var_map_;
  std::vector<std::shared_ptr<ILPSelfVarExpr>> all_str_self_vars_;

  // 2. edge variables
  std::map<std::pair<int/*cone id*/, int/*strategy id*/>,
           std::vector<std::shared_ptr<ILPEdgeVarExpr>>> str_edge_map_;
  std::map<std::pair<int/*src cone id*/, int/*src strategy id*/>,
           std::map<std::pair<int/*tgt cone id*/, int/*tgt str id*/>,
                    std::shared_ptr<ILPEdgeVarExpr>>> str_edge_map_no_dup_;
  std::vector<std::shared_ptr<ILPEdgeVarExpr>> all_str_edge_vars_;

  // ILP model:
  // 1. optimization object
  std::shared_ptr<ILPSumExpr> opt_obj_;
  // 2. constraints
  std::vector<std::shared_ptr<ILPEqualIntExpr>> constraints_;
};

struct HloSubGraph {
  HloSubGraph()
  : id_(0)
  , head_(nullptr)
  , tail_(nullptr)
  , insts_() {
  }
  HloSubGraph(const HloInstruction* head, const HloInstruction* tail)
  : id_(0)
  , head_(head)
  , tail_(tail)
  , insts_() {
  }
  HloSubGraph(HloSubGraph&& g)
  : id_(g.id_)
  , head_(g.head_)
  , tail_(g.tail_)
  , insts_(std::move(g.insts_)) {
  }
  HloSubGraph& operator=(HloSubGraph&& g) {
    id_ = g.id_;
    head_ = g.head_;
    tail_ = g.head_;
    insts_ = std::move(g.insts_);
  }
  HloSubGraph(const HloSubGraph& g)
  : id_(g.id_)
  , head_(g.head_)
  , tail_(g.tail_)
  , insts_(g.insts_) {
  }
  HloSubGraph& operator=(const HloSubGraph& g) {
    id_ = g.id_;
    head_ = g.head_;
    tail_ = g.head_;
    insts_ = g.insts_;
  }

  std::string ToString() {
    std::string head = "id: " + std::to_string(id_) + "\n";
    head += "head: ";
    head += head_ ? head_->name() : "null";
    head += "\ntail: ";
    head += tail_ ? tail_->name() : "null";

    std::string result = "\ninst list:\n";

    int mult_input_inst_num = 0;
    int mult_user_inst_num = 0;
    int total_edge_num = 0;
    for (auto* inst : insts_) {
      if (inst->user_count() > 1) {
        ++mult_user_inst_num;
      }
      if (inst->operand_count() > 1) {
        ++mult_input_inst_num;
      }
      total_edge_num += inst->operand_count();
      //result += inst->ToString() + "\n";
      result += inst->name() + "\n";
    }

    head += "\ntotal inst num: " + std::to_string(insts_.size()) + "\n";
    head += "total edge num: " + std::to_string(total_edge_num) + "\n";
    head += "inst num with multiple inputs: " + std::to_string(mult_input_inst_num) + "\n";
    head += "inst num with multiple users: " + std::to_string(mult_user_inst_num) + "\n";

    return head + result;
  }

  std::vector<const HloInstruction*> FindConeRoots(
      const std::vector<const HloInstruction*>& sorted_insts,
      const HloInstSet& graph_scope,  // sub graph instructions
      int opt_level);

  std::unordered_set<const HloInstruction*> BuildInstCones(
                            const HloInstSet& graph_scope,
                            int opt_level);  // sub graph instructions

  std::string ConesToString() const;

  void ExtractConeStrategy(
      const HloInstMap<std::unique_ptr<InstCosts>>& inst_costs,
      const std::unordered_set<const HloInstruction*>& cone_scope,
      const HloInstruction* inst,
      int strategy_idx,
      HloInstMap<SharedDimStrategy>& cone_strtg_map);

  void BuildConeStrtgMngr(
      const HloInstMap<SharedDimStrategy>& ctx_strategy_map,
      const HloInstMap<std::unique_ptr<InstCosts>>& inst_costs,
      std::list<std::shared_ptr<InstCone>>& inst_cones,
      int split_num,
      bool share_dev);

  std::string DumpInterCost();

#if 0
  void NeighborVote(
      std::list<std::shared_ptr<InstCone>>& inst_cones);
#endif

  void BuildILPModel(
      std::list<std::shared_ptr<InstCone>>& inst_cones,
      const InstAffinityMap& affinity_map,
      const HloInstMap<SharedDimStrategy>& strategy_map,
      ILPModel& ilp_model);

  const HloInstMap<SharedDimStrategy>*
  FindAccStrtgMap(const SharedDimStrategy& tail_strtg) const {
    if (finalized_tail_strtg_map_.find(tail_strtg) == finalized_tail_strtg_map_.end()) {
      return nullptr;
    } else {
      std::shared_ptr<SubGraphStrategy> sub_strtg = finalized_tail_strtg_map_.at(tail_strtg);
      CHECK(sub_strtg);
      return &(sub_strtg->acc_strategy_map_);
    }
  }

  int id_ = 0;
  const HloInstruction* head_;
  const HloInstruction* tail_;

  std::vector<const HloInstruction*> insts_;

  std::list<std::shared_ptr<SubGraphStrategy>> graph_strategies_;
  std::map<SharedDimStrategy, std::list<std::shared_ptr<SubGraphStrategy>>, DimStrategyPtrLess> tail_strtg_map_;

  std::map<SharedDimStrategy, std::shared_ptr<SubGraphStrategy>, DimStrategyPtrLess> finalized_tail_strtg_map_; // valid when tail_ is NOT null

  std::shared_ptr<SubGraphStrategy> unique_min_cost_strtg_; // valid when tail_ is null
  std::vector<std::shared_ptr<InstCone>> inst_cones_;

  std::unique_ptr<ConeStrategyManager> cone_strtg_mngr_;
};

struct InstStrategy {
  InstStrategy(const HloInstruction* inst, const SharedDimStrategy& strategy)
  : inst_(inst)
  , strategy_(strategy) {
  }
  bool operator==(const InstStrategy& other) const {
    return inst_ == other.inst_ && *strategy_ == *other.strategy_;
  }
  bool operator<(const InstStrategy& other) const {
    if (inst_ < other.inst_) {
      return true;
    } else if (inst_ > other.inst_) {
      return false;
    } else {
      return (*strategy_ < *other.strategy_);
    }
  }


  const HloInstruction* inst_;
  SharedDimStrategy strategy_;
};

struct GraphPiece {
  GraphPiece()
  : inst_map_()
  , compute_map_()
  , score_(0.0)
  , start_inst_(nullptr) {
  }
  GraphPiece(const GraphPiece& p)
  : inst_map_(p.inst_map_)
  , compute_map_(p.compute_map_)
  , score_(p.score_)
  , start_inst_(p.start_inst_) {
  }
  GraphPiece(GraphPiece&& p)
  : inst_map_(std::move(p.inst_map_))
  , compute_map_(std::move(p.compute_map_))
  , score_(p.score_)
  , start_inst_(p.start_inst_) {
  }

  void Insert(const HloInstruction* inst, const SharedDimStrategy& strtg) {
    inst_map_[inst] = strtg;
    if (is_compute_intensive(inst)) {
      compute_map_[inst] = strtg;
    }
  }

  bool CutBy(std::shared_ptr<GraphPiece> piece,
             std::vector<std::shared_ptr<GraphPiece>>& new_pieces);

  void EvalScore(const HloInstSet& scope, int split_num);

  std::string ToString() {
    std::string res;
    res += "start inst: " + start_inst_->name() + "\n";
    res += "score: " + std::to_string(score_) + "\n";
    res += "compute inst num: " + std::to_string(compute_map_.size()) + "\n";
    res += "compute insts:\n";
    for (auto& inst_strtg : compute_map_) {
      res += inst_strtg.first->ToString() + "\n" + inst_strtg.second->ToString() + "\n";
    }
    res += "inst num: " + std::to_string(inst_map_.size()) + "\n";
    res += "insts:\n";
    for (auto& inst_strtg : inst_map_) {
      res += inst_strtg.first->ToString() + "\n" + inst_strtg.second->ToString() + "\n";
    }

    return std::move(res);
  }
  HloInstMap<SharedDimStrategy> inst_map_;
  HloInstMap<SharedDimStrategy> compute_map_;

  float score_;
  const HloInstruction* start_inst_;

  static constexpr int64 kFmaFlops = 2;
};

struct GraphPieces {
  GraphPieces()
  : pieces_()
  , inst_piece_map_() {
  }
  GraphPieces(const GraphPieces& p)
  : pieces_(p.pieces_)
  , inst_piece_map_(p.inst_piece_map_) {
  }
  GraphPieces(GraphPieces&& p)
  : pieces_(std::move(p.pieces_))
  , inst_piece_map_(std::move(p.inst_piece_map_)) {
  }

  void EvalScore(const HloInstSet& scope, int split_num) {
    for (auto& piece : pieces_) {
      piece->EvalScore(scope, split_num);
    }
  }

  void AddPiece(const std::shared_ptr<GraphPiece> piece) {
    pieces_.push_back(piece);
    for (auto& strtg : piece->inst_map_) {
      InstStrategy inst_str(strtg.first, strtg.second);

      // debug
      if (inst_piece_map_.find(inst_str) != inst_piece_map_.end()) {
        if (strtg.second->IsPartial()) {
          VLOG(2) << "partial conflict";
        } else {
          VLOG(2) << "other conflict";
          VLOG(2) << "inst, strategy pair in two pieces: " << strtg.first->name();
          if (piece->inst_map_.size() < 200) {
            VLOG(2) << "existing piece:";
            VLOG(2) << inst_piece_map_[inst_str]->ToString();

            VLOG(2) << "new piece:";
            VLOG(2) << piece->ToString();
          }
        }
      }
      inst_piece_map_[inst_str] = piece;
    }
  }
  void CutBy(std::shared_ptr<GraphPiece> piece);

  bool HasPiece(const InstStrategy& inst_strtg) const {
    return (inst_piece_map_.find(inst_strtg) != inst_piece_map_.end());
  }
  bool HasInstruction(const HloInstruction* inst) const {
    for (auto& piece : pieces_) {
      if (piece->inst_map_.find(inst) != piece->inst_map_.end()) {
        return true;
      }
    }
    return false;
  }
  bool is_empty() const {
    return pieces_.empty();
  }
  std::shared_ptr<GraphPiece> FindPiece(const InstStrategy& inst_strtg) {
    return inst_piece_map_[inst_strtg];
  }
  void Dump() {
    VLOG(0) << "pieces count: " << pieces_.size();
    for (auto& piece : pieces_) {
      VLOG(0) << "piece instructions and strategies:\n" << piece->ToString();
    }
  }
  std::list<std::shared_ptr<GraphPiece>> pieces_;
  std::map<InstStrategy, std::shared_ptr<GraphPiece>> inst_piece_map_;
};

struct MemSavePlan {
  HloInstSet split_for_mem_save_;
  HloInstSet split_for_compute_;
  int64 total_mem_cost_;
  int64 total_orig_mem_cost_;
  int64 expected_mem_cost_;
};

struct InstMemCost {
  int64 orig_out_cost_;
  int64 out_cost_;
};

class CostSpmdStrategy {
 public:
  explicit CostSpmdStrategy() {};
  explicit CostSpmdStrategy(int split_ordinal, int split_num, int pp_stage_num, bool mem_save)
      : cur_split_ordinal_(split_ordinal)
      , cur_split_num_(split_num)
      , pp_stage_num_(pp_stage_num)
      , mem_save_(mem_save) {
      }
  virtual ~CostSpmdStrategy() = default;

  void CalcInstRank(HloModule* module,
                    HloInstMap<int>& inst_rank_map,
                    HloInstMap<int>& compute_rank_map);

  void
  InferUsersFromInst(
    const HloInstruction* inst,
    const DimStrategy& inst_strategy,
    const HloInstSet& stop_inst_set,   // stop instruction will be inferred
    const HloInstSet& inst_scope,
    const HloInstMap<SharedDimStrategy>& expected_strategies,
    GraphStrategy& graph_strategy,
    HloInstSet& infered_insts);

  void
  InferFromUser(
    const HloInstruction* inst,
    const DimStrategy& inst_strategy,
    const HloInstSet& stop_inst_set,   // stop instruction will NOT be inferred
    const HloInstSet& inst_scope,
    const HloInstMap<SharedDimStrategy>& expected_strategies,
    GraphStrategy& graph_strategy,
    HloInstSet& infered_insts);

  virtual StatusOr<bool> Run(HloModule* module);
  virtual bool StrategyPlanning(HloModule* module);

  HloInstMap<SharedDimStrategy>
  ExtractUserSplit(const HloModule* module);

 private:
  HloInstMap<SharedDimStrategy> FindStartingStrategies( // instructions in the returned map own strategies already
              const HloInstMap<SharedDimStrategy>& strategy_map,
              const HloInstSet& sub_graph_insts);  // whole graph search if it is empty

  HloInstMap<SharedDimStrategy> FindBackStartingStrategies(
              const HloInstMap<SharedDimStrategy>& strategy_map,
              const HloInstSet& sub_graph_insts);  // whole graph search if it is empty

  void FillStrategyForAllInstructions(
      const HloModule* module,
      HloInstMap<HLOStrategy>* best_strategy_map);

  void AlignInputOutputStrategy(
      HloModule* module,
      HloInstMap<HLOStrategy>* best_strategy_map,
      int& glued_inst_num);

  void ResetUnDivisibleStrategy(
      HloInstMap<HLOStrategy>* best_strategy_map,
      int& glued_inst_num);

  void BestStrategyPostProcess(
      HloModule* module,
      HloInstMap<HLOStrategy>* best_strategy_map,
      int& glued_inst_num);

  void InferFromAnnotation(
    const HloInstMap<SharedDimStrategy>& user_annotated_tensors,
    const HloInstMap<int>& inst_rank_map,
    const HloInstSet& inst_scope,
    GraphStrategy& graph_strategy);

  bool InferByRank(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<int>& inst_rank_map,
        GraphStrategy& graph_strategy);

  bool ReverseInferByRank(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<int>& inst_rank_map,
        GraphStrategy& graph_strategy);

  bool InferByInstCount(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy);

  bool ReverseInferByInstCount(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy);

  HloInstSet InferGreedy(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy);

  HloInstSet ReverseInferGreedy(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy);

  HloInstSet InferGreedy(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<SharedDimStrategy>& expected_strategies,
        GraphStrategy& graph_strategy);

  HloInstSet ReverseInferGreedy(
        const HloInstMap<SharedDimStrategy>& starting_strategies,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<SharedDimStrategy>& expected_strategies,
        GraphStrategy& graph_strategy);

  HloInstSet PopulateStrategy(
        const HloInstruction* inst,
        const SharedDimStrategy& inst_strategy,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy);   // graph_strategy contains the starting inst and inst_strategy

  HloInstSet PopulateStrategy(
        const HloInstruction* inst,
        const SharedDimStrategy& inst_strategy,
        const HloInstSet& stop_insts,  // stop instruction will NOT be inferred
        const HloInstSet& inst_scope,
        const HloInstMap<SharedDimStrategy>& expected_strategies,
        GraphStrategy& graph_strategy);   // graph_strategy contains the starting inst and inst_strategy

  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>> InferStartingStrtgs(
        const HloInstMap<SharedDimStrategy>& acc_strtg_map,
        const HloInstSet& graph_scope);

  void Infer(   // may remove it later, FastInferSubGraph and InferSubGraph are newer version with same functionality
        const HloInstSet& user_annotated_set,
        const HloInstMap<SharedDimStrategy>& init_strategy_map,
        HloSubGraph& sub_graph);

  void InferSubGraph(
        const HloInstSet& user_annotated_set,
        const HloInstMap<SharedDimStrategy>& init_strategy_map,  // strategies infered by user annotations
        const HloInstMap<SharedDimStrategy>& expected_strategies,
        const MemSavePlan& mem_save_plan,
        const HloInstMap<SharedDimStrategy>* acc_strtg_map,
        const InstAffinityMap& affinity_map,
        HloSubGraph& sub_graph,
        int opt_level);

  void FastInferSubGraph(
        const HloInstSet& user_annotated_set,
        const HloInstMap<SharedDimStrategy>& init_strategy_map,  // strategies infered by user annotations
        const HloInstMap<SharedDimStrategy>& expected_strategies,
        const MemSavePlan& mem_save_plan,
        const HloInstMap<SharedDimStrategy>* acc_strtg_map,
        const InstAffinityMap& affinity_map,
        HloSubGraph& sub_graph,
        int opt_level);

  void FastInferSubGraphByCone(
        const HloInstSet& user_annotated_set,
        const HloInstSet& graph_scope,
        const InstAffinityMap& affinity_map,
        HloSubGraph& sub_graph,
        HloInstMap<SharedDimStrategy>& strategy_map,
        int opt_level);

  void InferSubGraphByCone(
        const HloInstSet& user_annotated_set,
        const HloInstMap<SharedDimStrategy>& expected_strategies, // head & tail strategies
        const MemSavePlan& mem_save_plan,
        const HloInstSet& graph_scope,
        const InstAffinityMap& affinity_map,
        HloSubGraph& sub_graph,
        HloInstMap<SharedDimStrategy>& strategy_map,
        int opt_level);

  HloInstMap<std::unique_ptr<InstCosts>> InferInstCosts (
      const std::unordered_set<const HloInstruction*>& cone_roots,
      const HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>& split_candidates,
      const HloInstSet& graph_scope,
      const HloInstMap<SharedDimStrategy>& strategy_map);

  void StitchInstCones(
      const HloInstMap<std::unique_ptr<InstCosts>>& inst_costs,
      const HloInstSet& graph_scope,
      const InstAffinityMap& affinity_map,
      HloSubGraph& sub_graph,
      HloInstMap<SharedDimStrategy>& strategy_map);

  std::string InstCostsToString(
      const HloInstMap<std::unique_ptr<InstCosts>>& inst_costs,
      int sub_graph_id);

  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>> CollectInstSplits(
        const GraphPieces& pieces,
        const HloInstMap<SharedDimStrategy>& strategy_map,
        const HloInstSet& graph_scope);

  void GenComputeSplitProposalMap(
        const std::list<const HloInstruction*>& unsplitted_computes,
        const HloInstSet& split_for_mem_save,
        HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>& split_proposals);

  GraphPieces BuildGraphPieces(
        const HloInstSet& user_annotated_set,
        const HloInstMap<SharedDimStrategy>& init_strategy_map,
        const HloInstMap<SharedDimStrategy>& expected_strategies,
        const HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>>& split_proposals,
        const HloInstSet& graph_scope,
        bool shrink);

  GraphPieces BuildGraphPieces(
        const HloInstSet& user_annotated_set,
        const HloInstMap<SharedDimStrategy>& init_strategy_map,
        const HloInstSet& graph_scope,
        const HloInstMap<SharedDimStrategy>& start_strategies);

  std::shared_ptr<GraphPiece> BuildOnePiece(
        const HloInstruction* start_inst,
        const DimStrategy& strtg,
        const HloInstSet& user_annotated_set,
        const HloInstMap<SharedDimStrategy>& init_strategy_map, // may contain or not contain start_inst and its strategy
        const HloInstMap<SharedDimStrategy>& expected_strategies,
          // expected_strategies may be merged into init_strategy_map, after
          // merging, the generated piece will be smaller because it is break
          // by instructions in expected strategies map.

        const HloInstSet& graph_scope,
        const GraphPieces& pieces,
        bool do_shrink);

  std::vector<SharedDimStrategy>
  GenSplitProposals(const HloInstruction* inst);

  std::vector<SharedDimStrategy>
  GenSplitProposals(const HloInstruction* inst,
                    const HloInstSet& split_for_mem_save);

  std::list<const HloInstruction*> CollectUnsplittedComputeInsts(
        const HloModule* module,
        const HloInstMap<SharedDimStrategy>& strategy_map);

  std::list<const HloInstruction*> CollectUnsplittedComputeInsts(
        const HloInstSet& inst_scope,
        const HloInstMap<SharedDimStrategy>& strategy_map);

  HloInstMap<SharedDimStrategy> CollectSplittedComputeInsts(
        const HloInstSet& inst_scope,
        const HloInstMap<SharedDimStrategy>& strategy_map);

  HloInstMap<std::pair<const HloInstruction*, const HloInstruction*>>
  CreateComputeToParamMap(const HloModule* module);

  std::shared_ptr<InstCone> ExtractReverseCone(
        const HloInstruction* cone_root,
        const HloInstSet& inst_scope,
        const std::unordered_set<const HloInstruction*>& exclude_insts);

  HloSubGraph ExtractSubGraph(
        const HloInstruction* start,
        const HloInstruction* end,
        const HloInstSet& inst_scope,
        const std::unordered_set<const HloInstruction*>& exclude_insts);

  HloSubGraph BuildSubGraph(
        const std::vector<const HloInstruction*>& insts,
        const HloInstruction* head,
        const HloInstruction* tail);

  std::vector<HloSubGraph> FindSubGraphs(
        HloModule* module,
        const HloInstSet& forward_insts);

  void ExpandSubGraphs(
        const HloModule* module,
        const std::vector<const HloInstruction*>& critical_insts,
        std::unordered_set<const HloInstruction*>& exclude_insts,
        std::vector<HloSubGraph>& sub_graphs);

  bool SplitPlanByMemCost(
        const HloModule* module,
        const HloInstMap<SharedDimStrategy>& strategy_map,
        const HloInstMap<int>& compute_rank_map,
        const int64 mem_limit,
        MemSavePlan& mem_save_plan);

  void PlanByAnnotations(
        const HloModule* module,
        const HloInstMap<int>& inst_rank_map,
        const HloInstSet& inst_scope,
        GraphStrategy& graph_strategy);

  void PlanSubGraph(
        const HloInstMap<SharedDimStrategy>& fixed_strategy_map, // strategies infered by user annotations
        const HloInstSet& user_annotated_set,
        const MemSavePlan& mem_save_plan,
        const HloSubGraph* pre_sub_graph,
        const InstAffinityMap& affinity_map,
        int opt_level,
        HloSubGraph& sub_graph);
  void PlanSubGraphs(
        HloModule* module,
        const HloInstMap<SharedDimStrategy>& fixed_strategy_map, // strategies infered by user annotations
        const MemSavePlan& mem_save_plan,
        int opt_level,
        std::vector<HloSubGraph>& sub_graphs);

  void FinalizeCurStrMap(std::vector<HloSubGraph>& sub_graphs,
                         GraphStrategy& graph_strategy);

  void RecordStrategyToInsts(HloModule* module,
        const HloInstMap<HLOStrategy>& hlo_strategy_map);

  void DumpStrategies(
        const HloModule* module,
        const HloInstMap<HLOStrategy>& hlo_strategy_map);

 protected:
  int local_dev_num_ = 2;
  int worker_num_ = 1;

  HloInstSet user_annotated_insts_;  // instructions just after user annotated tensors

  // For conflict pieces candidates
  bool record_conflict_ = false;
  HloInstMap<std::set<SharedDimStrategy, DimStrategyPtrLess>> conflict_proposals_;

  int cur_split_ordinal_;
  int cur_split_num_;
  bool cur_share_dev_ = false;
  int pp_stage_num_ = 1;
  bool mem_save_ = false;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PARALLEL_COST_SPMD_STRATEGY_H_
