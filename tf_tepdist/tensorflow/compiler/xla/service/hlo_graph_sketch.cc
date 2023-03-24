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

#include <map>
#include <string>
#include <fstream>


#include "tensorflow/compiler/xla/service/hlo_graph_sketch.h"
#include "tensorflow/compiler/xla/service/hlo_instruction_util.h"
#include "tensorflow/compiler/xla/service/parallel/utils.h"
#include "tensorflow/compiler/xla/service/parallel/inst_affinity_map.h"
#include "tensorflow/compiler/xla/service/parallel/performance_utils.h"
#include "tensorflow/compiler/xla/service/service_env.h"
#include "tensorflow/core/platform/numbers.h"

namespace xla {

using ::tensorflow::strings::HumanReadableNumBytes;

void
AccumCompInfo::EvalAccumFlopCount() {
  accum_flop_count_ = 0;
  for (auto* node : accum_comp_nodes_) {
    accum_flop_count_ += node->flop_count();
  }
}

std::string
AccumCompInfo::ToString() const {
  std::string res;
  res += "      accum_flop_count: " + std::to_string(accum_flop_count_) + ", node list:";
  for (auto* node : accum_comp_nodes_) {
    res += "\n        " + node->name();
  }

  return std::move(res);
}

SketchNode::SketchNode(const HloInstruction* core_instr, bool is_backward)
  : core_instr_(core_instr)
  , is_backward_(is_backward) {
}

void
SketchNode::EvalCost(
    std::unordered_map<const HloInstruction*, int64>& inst_flops_map) {
  // currently only count the flop of gemm(dot) and convolution
  flop_count_ = 0;
  for (auto* instr : instructions_) {
    int64 inst_flops = PerfUtils::CalculateFlops(instr);
    inst_flops_map[instr] = inst_flops;
    flop_count_ += inst_flops;
  }

  VLOG(2) << "node: " << this->name() << ", cost: " << flop_count_;
}

void IlpStageModel::BuildOpUserConstraint(
                            int op_stage, int user_stage,
                            std::shared_ptr<IlpStagePrimExpr> op_expr,
                            std::shared_ptr<IlpStagePrimExpr> user_expr,
                            bool affine) {
  VLOG(2) << "  build op/user constraint expr";
  if (user_stage>=0) {
    // user's stage is determined
    CHECK(op_stage<0 && op_expr);  // op stage is not determined yet
    // build constraint
    //         op_stage <= user_stage
    //    or   op_stage = user_stage
    //    i.e. op_expr <= constant (user stage)
    //    or   op_expr = constant
    std::shared_ptr<IlpStageSumExpr> cond_left = std::make_shared<IlpStageSumExpr>();
    cond_left->operands_.emplace_back(op_expr);
    if (affine) {
      std::shared_ptr<IlpStageEqIntExpr> eq_expr =
                    std::make_shared<IlpStageEqIntExpr>(cond_left, user_stage);
      VLOG(2) << op_expr->ModelStr() << "(op expr) = " << user_stage << "(user stage)";
      this->AddConstraint(eq_expr);
    } else {
      std::shared_ptr<IlpStageLEIntExpr> le_expr =
                    std::make_shared<IlpStageLEIntExpr>(cond_left, user_stage);
      VLOG(2) << op_expr->ModelStr() << "(op expr) <= " << user_stage << "(user stage)";
      this->AddConstraint(le_expr);
    }
  } else {
    // user's stage is not determined yet
    CHECK(user_expr);
    if (op_stage<0) {
      // op's stage is not determined yet
      //         user_stage - op_stage >= 0
      //    or   user_stage - op_stage = 0
      //    i.e. user_expr - op_expr >= 0
      //    or   user_expr - op_expr = 0
      CHECK(op_expr);
      std::shared_ptr<IlpScaledExpr> minus_op_expr =
                              std::make_shared<IlpScaledExpr>(op_expr, -1);
      std::shared_ptr<IlpStageSumExpr> cond_left =
                                        std::make_shared<IlpStageSumExpr>();
      cond_left->operands_.emplace_back(user_expr);
      cond_left->operands_.emplace_back(minus_op_expr);
      if (affine) {
        std::shared_ptr<IlpStageEqIntExpr> eq_expr =
                            std::make_shared<IlpStageEqIntExpr>(cond_left, 0);
        this->AddConstraint(eq_expr);
      } else {
        std::shared_ptr<IlpStageGEIntExpr> ge_expr =
                            std::make_shared<IlpStageGEIntExpr>(cond_left, 0);
        this->AddConstraint(ge_expr);
      }
    } else {
      // op's stage is determined
      //        user_stage >= op_stage
      //   or   user_stage = op_stage
      //   i.e. user_expr >= constant (op stage)
      //   or   user_expr = constant (op stage)
      if (op_stage >= logic_stage_start_ && op_stage <= logic_stage_end_) {
        std::shared_ptr<IlpStageSumExpr> cond_left =
                                          std::make_shared<IlpStageSumExpr>();
        cond_left->operands_.emplace_back(user_expr);
        if (affine) {
          std::shared_ptr<IlpStageEqIntExpr> eq_expr =
                      std::make_shared<IlpStageEqIntExpr>(cond_left, op_stage);
          this->AddConstraint(eq_expr);
        } else {
          std::shared_ptr<IlpStageGEIntExpr> ge_expr =
                      std::make_shared<IlpStageGEIntExpr>(cond_left, op_stage);
          this->AddConstraint(ge_expr);
        }
      } else {
        // Out of scope
        // do nothing because asap and alap constraints imply user_stage >= op_stage
      }
    }
  }
}

void IlpStageModel::BuildOpUserCost(
                            int op_idx, const HloInstruction* op,
                            const HloInstruction* user,
                            int op_stage, int user_stage,
                            std::shared_ptr<IlpStagePrimExpr> op_expr,
                            std::shared_ptr<IlpStagePrimExpr> user_expr) {
  if (op_stage < 0) {
    // 1. op stage is undetermined
    BuildOpUserCostInScope(op_idx, op, user,
                           op_stage, user_stage, op_expr, user_expr);
  } else if (op_stage >= logic_stage_start_) {
    // 2. op stage is determined and in the same stage range which user will be placed
    BuildOpUserCostInScope(op_idx, op, user, op_stage, user_stage,
                           op_expr, user_expr);
  } else {
    // 3. op stage is determined and out of stage range which user will be placed
    //   i.e.  0 <= op_stage < logic_stage_start_
    CHECK(!op_expr);
    CHECK(user_expr && user_stage < 0);
    /*
    int op_mirror_stage = logic_stage_end_ - op_stage;
    BuildOpUserCostInScope(op_idx, op, user, op_mirror_stage, user_stage,
                           op_expr, user_expr);
    */
    BuildOpUserCostOutScope(logic_stage_start_, logic_stage_end_, op_idx, op,
                            user, op_stage, /*user_stage, op_expr,*/ user_expr);
  }
}

void IlpStageModel::BuildOpUserCostInScope(
                            int op_idx, const HloInstruction* op,
                            const HloInstruction* user,
                            int op_stage, int user_stage,
                            std::shared_ptr<IlpStagePrimExpr> op_expr,
                            std::shared_ptr<IlpStagePrimExpr> user_expr) {
  // build across cost and its constraint
  //  i). stage across flag function:
  //    f = 0, when stage(user)-stage(op) = 0
  //    f = 1, when stage(user)-stage(op) >= 1
  //          where stage(user) >= stage(op)
  //  reform above function to following function by big M method:
  //    s.t.  f <= stage(user)-stage(op) <= fM
  //          where stage(user) >= stage(op)
  //          f is 0-1 variable
  //
  //  ii). stage across cost:
  //    cost = f * ByteSizeOf(op)

  // 1. update optimization object
  //    get or create an across flag variable for the input of user
  std::shared_ptr<IlpStageSumExpr> opt_obj = this->GetOrCreateOptObj();
  int64 op_bytes = ShapeUtil::ByteSizeOf(op->shape(), 8);
  std::shared_ptr<IlpAcrossFlagVarExpr> flag_var =  // binary value
                                    this->GetOrCreateAcrossVar(user, op_idx);
  std::shared_ptr<IlpScaledExpr> sc_var =
                      std::make_shared<IlpScaledExpr>(flag_var, op_bytes);
  CHECK(opt_obj);
  opt_obj->operands_.emplace_back(sc_var);

  int M=logic_stage_end_;
  VLOG(2) << "build constraint for f <= stage(user)-stage(op) <= Mf";

  // 2. build constraint for f <= stage(user)-stage(op) <= Mf,
  //    where stage(user) >= stage(op)
  // 2.1. build constraint for f <= stage(user)-stage(op)
  //        i.e. stage(user)-stage(op)-f >= 0
  //
  // 2.2. build constraint for stage(user)-stage(op) <= Mf
  //        i.e. Mf-stage(user)+stage(op) >= 0
  if (user_stage>=0) {
    VLOG(2) << "user's stage is determined";
    // user's stage is determined, build
    // stage(user) is a constant
    // 2.1. f <= stage(user)-stage(op)
    // i.e. stage(op)+f <= stage(user)
    CHECK(op_stage<0 && op_expr);  // op stage is not determined yet
    std::shared_ptr<IlpStageSumExpr> cond_left = std::make_shared<IlpStageSumExpr>();
    cond_left->operands_.emplace_back(op_expr);
    cond_left->operands_.emplace_back(flag_var);
    std::shared_ptr<IlpStageLEIntExpr> le_expr =
                std::make_shared<IlpStageLEIntExpr>(cond_left, user_stage);
    this->AddConstraint(le_expr);

    // 2.2. stage(user)-stage(op) <= Mf
    // i.e. Mf-stage(user)+stage(op) >= 0
    // i.e. stage(op)+Mf >= stage(user)
    std::shared_ptr<IlpScaledExpr> Mf =
                              std::make_shared<IlpScaledExpr>(flag_var, M);
    cond_left = std::make_shared<IlpStageSumExpr>();
    cond_left->operands_.emplace_back(op_expr);
    cond_left->operands_.emplace_back(Mf);
    std::shared_ptr<IlpStageGEIntExpr> ge_expr =
                std::make_shared<IlpStageGEIntExpr>(cond_left, user_stage);
    this->AddConstraint(ge_expr);
    VLOG(2) << "constraint is added: user's stage is determined";
  } else {
    // user's stage is not determined yet
    CHECK(user_expr);
    if (op_stage<0) {
      VLOG(2) << "both user and op stage NOT determined";
      // op's stage is not determined yet, build
      // 2.1. f <= stage(user)-stage(op)
      //    stage(user)-stage(op)-f >= 0
      CHECK(op_expr);
      std::shared_ptr<IlpScaledExpr> minus_op =
                            std::make_shared<IlpScaledExpr>(op_expr, -1);
      std::shared_ptr<IlpScaledExpr> minus_f =
                            std::make_shared<IlpScaledExpr>(flag_var, -1);

      std::shared_ptr<IlpStageSumExpr> cond_left = std::make_shared<IlpStageSumExpr>();
      cond_left->operands_.emplace_back(user_expr);
      cond_left->operands_.emplace_back(minus_op);
      cond_left->operands_.emplace_back(minus_f);
      std::shared_ptr<IlpStageGEIntExpr> ge_expr =
                        std::make_shared<IlpStageGEIntExpr>(cond_left, 0);
      this->AddConstraint(ge_expr);

      // 2.2. stage(user)-stage(op) <= Mf
      // i.e. Mf-stage(user)+stage(op) >= 0
      // i.e. stage(user)-stage(op)-Mf <= 0
      std::shared_ptr<IlpScaledExpr> minus_Mf =
                              std::make_shared<IlpScaledExpr>(flag_var, -M);
      cond_left = std::make_shared<IlpStageSumExpr>();
      cond_left->operands_.emplace_back(user_expr);
      cond_left->operands_.emplace_back(minus_op);
      cond_left->operands_.emplace_back(minus_Mf);
      std::shared_ptr<IlpStageLEIntExpr> le_expr =
                          std::make_shared<IlpStageLEIntExpr>(cond_left, 0);
      this->AddConstraint(le_expr);
      VLOG(2) << "constraint is added: both user and op stage NOT determined";
    } else {
      VLOG(2) << "op stage is determined, but user stage not";
      // op's stage is determined, build
      // stage(op) is a constant
      // 2.1. f <= stage(user)-stage(op)
      // i.e. stage(user)-f >= stage(op)
      std::shared_ptr<IlpScaledExpr> minus_f =
                            std::make_shared<IlpScaledExpr>(flag_var, -1);
      std::shared_ptr<IlpStageSumExpr> cond_left = std::make_shared<IlpStageSumExpr>();
      cond_left->operands_.emplace_back(user_expr);
      cond_left->operands_.emplace_back(minus_f);
      std::shared_ptr<IlpStageGEIntExpr> ge_expr =
                  std::make_shared<IlpStageGEIntExpr>(cond_left, op_stage);
      this->AddConstraint(ge_expr);

      // 2.2. stage(user)-stage(op) <= Mf
      // i.e. Mf-stage(user)+stage(op) >= 0
      // i.e. stage(user)-Mf <= stage(op)
      std::shared_ptr<IlpScaledExpr> minus_Mf =
                              std::make_shared<IlpScaledExpr>(flag_var, -M);
      cond_left = std::make_shared<IlpStageSumExpr>();
      cond_left->operands_.emplace_back(user_expr);
      cond_left->operands_.emplace_back(minus_Mf);
      std::shared_ptr<IlpStageLEIntExpr> le_expr =
                  std::make_shared<IlpStageLEIntExpr>(cond_left, op_stage);
      this->AddConstraint(le_expr);
      VLOG(2) << "constraint is added: op stage is determined, but user stage not";
    }
  }
}

void IlpStageModel::BuildOpUserCostOutScope(
                            int start_stage, int end_stage,
                            int op_idx, const HloInstruction* op,
                            const HloInstruction* user,
                            int op_stage,
                            std::shared_ptr<IlpStagePrimExpr> user_expr) {
  CHECK(start_stage> 0);
  CHECK(user_expr);
  CHECK(0 <= op_stage && op_stage < start_stage);
  // build across cost and its constraint
  // NOTE: stage(op):   a constant with range [0, start_stage-1]
  //       stage(user): a variable with range [start_stage, end_stage]
  //  1.
  //    stage across flag function:
  //    f = 1, when stage(user)+stage(op)-end_stage= 0
  //    f = 0, otherwise
  //          where stage(user) >= stage(op)
  //
  //  1.1. Construct absolute value of stage(user)+stage(op)-end_stage
  //    Introduce 0-1 variable y to represent if d >= 0 or not.
  //    Introduce variable x to represent absolute(d).
  //
  //    Then,
  //    d = stage(user)+stage(op)-end_stage
  //    (y-1)*M <= d < y*M
  //    -d-M*y <= x <= -d+M*y
  //    d-M*(1-y) <= x <= d+M*(1-y)
  //
  //    NOTE:
  //    2*start_stage-1=end_stage
  //    range of d:
  //        [start_stage-end_stage, start_stage-1]
  //        or
  //        [-(start_stage-1), start_stage-1]
  //        or
  //        [1-start_stage, start_stage-1]
  //    range of absolute(d): [0, start_stage-1]
  //    range of x: [0, start_stage-1]
  //
  //    2*absolute(d) <= 2*start_stage-2 < 2*start_stage-1 = end_stage
  //
  //    M value must be larger than or equal to 2*absolute(d).
  //    Let M be end_stage. It is safe enough.
  //  
  //  1.2. Represent across flag variable with absolute value, i.e. x.
  //    s.t.  f <= x <= fM
  //          f is 0-1 variable
  //
  //  1.3. All constraints of an across flag are list below:
  //    s.t.  f <= x <= fM
  //          d = stage(user)+stage(op)-end_stage
  //          (y-1)*M <= d < y*M
  //          -d-M*y <= x <= -d+M*y
  //          d-M*(1-y) <= x <= d+M*(1-y)
  //
  //          where
  //          f and y are 0-1 variables.
  //          Let M be end_stage.
  //          d is integer.
  //          x is non-negative integer. x represents absolute(d).
  //  2. stage across cost:
  //    cost = f * ByteSizeOf(op)

  // 1. update optimization object
  //    get or create an across flag variable for the input of user
  std::shared_ptr<IlpStageSumExpr> opt_obj = this->GetOrCreateOptObj();
  int64 op_bytes = ShapeUtil::ByteSizeOf(op->shape(), 8);
  std::shared_ptr<IlpAcrossFlagVarExpr> flag_var =  // binary value
                                  this->GetOrCreateAcrossVar(user, op_idx);
  std::shared_ptr<IlpScaledExpr> sc_var =
                          std::make_shared<IlpScaledExpr>(flag_var, op_bytes);
  opt_obj->operands_.emplace_back(sc_var);

  int M = end_stage;

  // 2. build constraints
  std::string aux_prefix_d = op->name() + "_" + user->name() + "_d";
  std::string aux_prefix_x = op->name() + "_" + user->name() + "_x";
  std::shared_ptr<IlpAuxVarExpr> d_var = CreateAuxVar(aux_prefix_d,
                                                      1-start_stage,
                                                      start_stage-1);
  std::shared_ptr<IlpAuxVarExpr> x_var = CreateAuxVar(aux_prefix_x,
                                                      0, start_stage-1);
  // 2.1. constraints for f <= x <= fM  (NOTE: x is absolute(d))
  // 2.1.1. f-x <= 0
  std::shared_ptr<IlpStageSumExpr> cond_left = std::make_shared<IlpStageSumExpr>();
  std::shared_ptr<IlpScaledExpr> minus_x =
                              std::make_shared<IlpScaledExpr>(x_var, -1);
  cond_left->operands_.emplace_back(flag_var);
  cond_left->operands_.emplace_back(minus_x);
  std::shared_ptr<IlpStageLEIntExpr> le_expr =
                              std::make_shared<IlpStageLEIntExpr>(cond_left, 0);
  this->AddConstraint(le_expr);

  // 2.1.2. x-Mf <= 0
  cond_left = std::make_shared<IlpStageSumExpr>();
  std::shared_ptr<IlpScaledExpr> minus_Mf =
                              std::make_shared<IlpScaledExpr>(flag_var, -M);
  cond_left->operands_.emplace_back(x_var);
  cond_left->operands_.emplace_back(minus_Mf);
  le_expr = std::make_shared<IlpStageLEIntExpr>(cond_left, 0);
  this->AddConstraint(le_expr);

  // 2.2. constraints for d = stage(user)+stage(op)-end_stage
  //      i.e. stage(user) - d = end_stage - stage(op)
  std::shared_ptr<IlpScaledExpr> minus_d =
                                  std::make_shared<IlpScaledExpr>(d_var, -1);
  cond_left = std::make_shared<IlpStageSumExpr>();
  cond_left->operands_.emplace_back(user_expr);
  cond_left->operands_.emplace_back(minus_d);
  std::shared_ptr<IlpStageEqIntExpr> eq_expr =
          std::make_shared<IlpStageEqIntExpr>(cond_left, end_stage-op_stage);
  this->AddConstraint(eq_expr);

  // 2.3. constraints for (y-1)*M <= d < y*M
  //      i.e. M*y - d <= M
  //           M*y - d > 0
  //
  //      i.e. M*y - d <= M
  //           M*y - d >= 1
  std::string aux_prefix_y = op->name() + "_" + user->name() + "_y";
  std::shared_ptr<IlpAuxVarExpr> y_var = CreateAuxVar(aux_prefix_y, 0, 1); // binary var
  std::shared_ptr<IlpScaledExpr> My = std::make_shared<IlpScaledExpr>(y_var, M);
  cond_left = std::make_shared<IlpStageSumExpr>();
  cond_left->operands_.emplace_back(My);
  cond_left->operands_.emplace_back(minus_d);
  le_expr = std::make_shared<IlpStageLEIntExpr>(cond_left, M);
  this->AddConstraint(le_expr);
  std::shared_ptr<IlpStageGEIntExpr> ge_expr =
                            std::make_shared<IlpStageGEIntExpr>(cond_left, 1);
  this->AddConstraint(ge_expr);

  // 2.4. constraints for -d-M*y <= x <= -d+M*y
  //      i.e. M*y+d+x >= 0
  //           M*y-d-x >= 0
  cond_left = std::make_shared<IlpStageSumExpr>();
  cond_left->operands_.emplace_back(My);
  cond_left->operands_.emplace_back(d_var);
  cond_left->operands_.emplace_back(x_var);
  ge_expr = std::make_shared<IlpStageGEIntExpr>(cond_left, 0);
  this->AddConstraint(ge_expr);

  cond_left = std::make_shared<IlpStageSumExpr>();
  cond_left->operands_.emplace_back(My);
  cond_left->operands_.emplace_back(minus_d);
  cond_left->operands_.emplace_back(minus_x);
  ge_expr = std::make_shared<IlpStageGEIntExpr>(cond_left, 0);
  this->AddConstraint(ge_expr);

  // 2.5. constraints for d-M*(1-y) <= x <= d+M*(1-y)
  //      i.e. My+d-x <= M
  //           My-d+x <= M
  cond_left = std::make_shared<IlpStageSumExpr>();
  cond_left->operands_.emplace_back(My);
  cond_left->operands_.emplace_back(d_var);
  cond_left->operands_.emplace_back(minus_x);
  le_expr = std::make_shared<IlpStageLEIntExpr>(cond_left, M);
  this->AddConstraint(le_expr);

  cond_left = std::make_shared<IlpStageSumExpr>();
  cond_left->operands_.emplace_back(My);
  cond_left->operands_.emplace_back(minus_d);
  cond_left->operands_.emplace_back(x_var);
  le_expr = std::make_shared<IlpStageLEIntExpr>(cond_left, M);
  this->AddConstraint(le_expr);
}

std::string IlpStageModel::ExportToString() const {
  std::string ilp_mod_str("pipeline ilp:\n");
  // 1. dump ILP optimization object
  if (opt_obj_) {
    VLOG(2) << "optimize objective";
    ilp_mod_str += opt_obj_->ModelStr() + "\n";
  }

  VLOG(2) << "constraints size: " << constraints_.size();
  // 2. dump ILP constraints
  int idx=0;
  for (auto& constraint : constraints_) {
    VLOG(2) << "constraint";
    ilp_mod_str += std::to_string(idx++) + ": " + constraint->ModelStr() + "\n";
  }

  return std::move(ilp_mod_str);
}

std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>>
IlpStageModel::BuildVarMap() const {
  auto extract_var_map = [&](
      std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>>& var_map) {
    var_map.clear();
    // extract var from constraints expr
    for (auto& constraint : constraints_) {
      int row_id = constraint->id_;
      VLOG(2) << "constraint id: " << row_id;
      std::shared_ptr<IlpStageSumExpr> sum_expr = constraint->left_;
      for (auto& prim_expr : sum_expr->operands_) {
        int var_id = prim_expr->var_id();  // column id
        int scale = prim_expr->scale();    // scale
        auto& row_scale_map = var_map[var_id];
        row_scale_map[row_id] = (double)scale;
      }
    }
  };

  std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>> var_map;
  extract_var_map(var_map);

  return std::move(var_map);
}

bool
IlpStageModel::Solve(std::unordered_map<const HloInstruction*, int/*stage id*/>& inst_stage_map) const {
  // solve by Cbc
  std::map<int/*var id*/, std::map<int/*row id*/, double/*scale*/>> var_map = BuildVarMap();

  int row_num = constraints_.size();
  int col_num = var_map.size();
  int coeff_num = 0;
  for (auto& var_info : var_map) {
    coeff_num += var_info.second.size();
  }
  CHECK(coeff_num > 0);

  // matrix data - column ordered
  CoinBigIndex* col_start = new CoinBigIndex[col_num+1]; // start
  int* col_length = new int[col_num];   // length
  int* coeff_rows = new int[coeff_num]; // row index of each coeff
  double* coeffs = new double[coeff_num];  // elements(values)

  double* objective = new double[col_num];
  double* row_lower = new double[row_num];
  double* row_upper = new double[row_num];
  double* col_lower = new double[col_num];
  double* col_upper = new double[col_num];

  int coeff_id = 0;
  // fill coeffs, coeff_rows, col_strart and col_length array
  VLOG(0) << "var map size: " << var_map.size();
  for (auto& var_info : var_map) {
    int var_id = var_info.first;  // column id
    VLOG(2) << "var id: " << var_id;
    VLOG(2) << "col length: " << var_info.second.size();
    col_start[var_id] = coeff_id;
    col_length[var_id] = var_info.second.size();

    for (auto& row_scale : var_info.second) {
      coeff_rows[coeff_id] = row_scale.first;
      coeffs[coeff_id] = row_scale.second;
      ++coeff_id;
    }
  }
  col_start[col_num] = coeff_id;

  // initialize objective, col_lower and col_upper array
  for (int i=0; i<col_num; ++i) {
    std::shared_ptr<IlpVarExpr> var_expr = all_vars_[i];
    objective[i] = 0.0;
    col_lower[i] = var_expr->lower_bound_;
    col_upper[i] = var_expr->upper_bound_;
  }

  // fill objective, col_lower, col_upper array
  for (std::shared_ptr<IlpStagePrimExpr>& prim : opt_obj_->operands_) {
    objective[prim->var_id()] = (double)prim->scale();
  }

  // fill row_lower, row_upper
  for (int i=0; i<row_num; ++i) {
    const std::shared_ptr<IlpStageRelationRHSIntExpr>& cond = constraints_[i];
    CHECK(cond);
    row_lower[i] = cond->get_row_lower();
    row_upper[i] = cond->get_row_upper();
  }

  auto dump_var_map = [&var_map] () -> std::string {
    std::string res;

    for (auto& var_info : var_map) {
      res += "\ncol id: " + std::to_string(var_info.first) + ", row num: "
             + std::to_string(var_info.second.size());

      for (auto& row_scale : var_info.second) {
        res += "\n  row id: " + std::to_string(row_scale.first) + ", scale: "
               + std::to_string(row_scale.second);
      }
    }

    return std::move(res);
  };

  auto dump_cbc_model = [&] () -> std::string {
    std::string res;

    res += "\ncoeff num: " + std::to_string(coeff_num);
    res += "\nrow num: " + std::to_string(row_num);
    res += "\ncol num: " + std::to_string(col_num);

    res += "\n\ncoeff info:\nrow_idx: coeff value";
    for (int i=0; i<coeff_num; ++i) {
      res += "\n" + std::to_string(coeff_rows[i]) + ": " + std::to_string(coeffs[i]);
    }

    res +="\n\ncol info:\ncol_idx, col_name: col_start, col_length";
    for (int i=0; i<col_num; ++i) {
      std::shared_ptr<IlpVarExpr> var_expr = all_vars_[i];
      res += "\n" + std::to_string(i) + ", " + var_expr->ModelStr() + ": ";
      res += std::to_string(col_start[i]) + ", " + std::to_string(col_length[i]);
    }
    res += "\nlast col_start: " + std::to_string(col_start[col_num]);

    res +="\n\ncol bound info:\ncol_idx, col_name: lower, upper";
    for (int i=0; i<col_num; ++i) {
      std::shared_ptr<IlpVarExpr> var_expr = all_vars_[i];
      res += "\n" + std::to_string(i) + ", " + var_expr->ModelStr()
             + ": " + std::to_string(col_lower[i]) + ", "
             + std::to_string(col_upper[i]);
    }

    res +="\n\nrow info:\nrow_idx: lower, upper";
    for (int i=0; i<row_num; ++i) {
      res += "\n" + std::to_string(i) + ": " + std::to_string(row_lower[i])
             + ", " + std::to_string(row_upper[i]);
    }

    res +="\n\nobjective info:\ncol_idx, col_name: objective coefficient";
    for (int i=0; i<col_num; ++i) {
      std::shared_ptr<IlpVarExpr> var_expr = all_vars_[i];
      res += "\n" + std::to_string(i) + ", " + var_expr->ModelStr() + ": ";
      res += std::to_string(objective[i]);
    }
    return std::move(res);
  };

  std::string var_map_str = dump_var_map();
  VLOG(2) << "\nvar map info:" << var_map_str;

  std::string cbc_model_str = dump_cbc_model();
  VLOG(2) << "\ncbc model info:\n" << cbc_model_str;

  CoinPackedMatrix matrix(true, row_num, col_num, coeff_num, coeffs,
                          coeff_rows, col_start, col_length);
  OsiClpSolverInterface solver;
  // load problem
  solver.loadProblem(matrix, col_lower, col_upper,
                     objective, row_lower, row_upper);

  for (int i=0; i<col_num; ++i) {
    solver.setInteger(i);
  }

  // Solve
  // Currently we encounter a bug in linear relaxations caused infeasible
  // solution space.
  //solver.initialSolve(); // linear relaxation

  //solver.setObjSense(-1.0);

  // Pass data and solver to CbcModel
  CbcModel cbc_model(solver);

  int64 time_limit = ServiceEnv::ilp_time_limit();
  if (time_limit > 0) {
    double time_limit_sec = (double)(time_limit*60);
    cbc_model.setMaximumSeconds(time_limit_sec);
    LOG(INFO) << "Setting ILP_TIME_LIMIT as " << time_limit << " minutes.";
  }

  int64 num_threads = ServiceEnv::ilp_num_threads();
  if (num_threads > 0) {
    cbc_model.setNumberThreads(num_threads);
    LOG(INFO) << "Setting ILP_NUM_THREADS as " << num_threads << ".";
  }

  // reduce printout
  if (!ServiceEnv::debug()) cbc_model.setLogLevel(0);

  //model.solver()->setHintParam(OsiDoReducePrint, true, OsiHintTry);

  // Do complete search
  cbc_model.branchAndBound();

  bool succ = true;
  if (cbc_model.isProvenOptimal()) {
    VLOG(0) << "pipeline optimal solution found by ilp solver";
  } else {
    if (cbc_model.isProvenInfeasible()) {
      VLOG(0) << "infeasible ILP model";
      succ = false;
    } else {
      VLOG(0) << "pipeline optimal solution NOT found by ilp solver";
    }
  }

  if (succ) {
    const double *val = cbc_model.getColSolution();

    VLOG(2) << "Pipeline solution:";
    for (int i=0; i<col_num; ++i) {
      VLOG(2) << val[i];
    }

    VLOG(2) << "Pipeline stage var solution:";
    CHECK(all_vars_.size() <= col_num);
    for (int i=0; i<all_vars_.size(); ++i) {
      int var_val = (int)(val[i]+0.5);
      VLOG(2) << var_val;

      const std::shared_ptr<IlpVarExpr>& var = all_vars_[i];
      VLOG(2) << "var name: " << var->ModelStr();
      if (var->is_stage_var()) {
        const HloInstruction* inst = var->inst();
        if (inst) {
          VLOG(2) << "inst: " << inst->name() << ", stage: " << var_val;
          CHECK(inst_stage_map.find(inst) == inst_stage_map.end());
          inst_stage_map[inst] = var_val;
        }
      } else {
        VLOG(2) << "Not stage var";
      }
    }
  }

  delete [] col_start;
  delete [] col_length;
  delete [] coeff_rows;
  delete [] coeffs;

  delete [] objective;
  delete [] row_lower;
  delete [] row_upper;
  delete [] col_lower;
  delete [] col_upper;

  return succ;
}

GraphSketch::GraphSketch(HloModule* module, int physical_stage_num)
    : variable_map_(module->variable_map()),
      physical_stage_num_(physical_stage_num),
      logic_stage_num_(2 * physical_stage_num),
      module_(module) {
#if 0
  HloModule::DefContext* entry_def = module->def_ctx();
  CHECK(entry_def);
  HloModule::DefContext* sync_free_def = entry_def->ComputeGradientsDefCtx();
  CHECK(sync_free_def);
  HloComputation* sync_free_comp = module->Def2Compute(sync_free_def);
  CHECK(sync_free_comp);
  computation_ = sync_free_comp->instruction_count() > 1 ? \
      sync_free_comp : module->entry_computation();
  CHECK(computation_);
#else
  computation_ = module->entry_computation();
#endif
  reachability_ = HloReachabilityMap::Build(computation_);
}

/*static*/
std::unique_ptr<GraphSketch>
GraphSketch::BuildGraphSketch(HloModule* module, int physical_stage_num) {
  HloInstSet backward_insts = FindBackwardInsts(module, true);

  std::string back_insts_str;
  for (auto* inst : backward_insts) {
    back_insts_str += inst->ToString() + "\n";
  }
  VLOG(2) << "backward instructions:\n" << back_insts_str;

  std::unique_ptr<GraphSketch> graph_sketch =
      absl::make_unique<GraphSketch>(module, physical_stage_num);
  std::unordered_map<HloInstruction*, SketchNode*>& inst_node_map =
      graph_sketch->inst_node_map();

  HloComputation* computation = graph_sketch->SketchComputation();
  std::vector<HloInstruction*> post_order = computation->MakeInstructionPostOrder();
  bool is_core_inst;
  for (auto* inst : post_order) {
    auto inst_node_it = inst_node_map.find(inst);
    CHECK(inst_node_it == inst_node_map.end());

    if (inst->operand_count() == 0) {
      is_core_inst = true;
    } else if (inst->opcode() == HloOpcode::kCustomCall &&
               inst->custom_call_target() == "__cublas$gemm") {
      is_core_inst = true;
    } else if (inst->opcode() == HloOpcode::kDot||
               inst->opcode() == HloOpcode::kConvolution) {
      is_core_inst = true;
    } else if (inst == computation->root_instruction()) {
      is_core_inst = true;
    } else {
      is_core_inst = false;
    }

    std::set<SketchNode*> prev_sketch_nodes;
    for (auto* operand : inst->operands()) {
      auto inst_node_it = inst_node_map.find(operand);
      CHECK(inst_node_it != inst_node_map.end());
      prev_sketch_nodes.insert(inst_node_it->second);
    }

    std::unique_ptr<SketchNode> sketch_node;
    SketchNode* sketch_node_ptr = nullptr;
    if (is_core_inst ||                  // core instruction
        prev_sketch_nodes.size() >= 2) { // two or more sketch nodes are merged.
      // create a new sketch node
      bool is_backward = false;
      if (backward_insts.find(inst) != backward_insts.end()) {
        is_backward = true;
      }
      sketch_node = absl::make_unique<SketchNode>(inst, is_backward);
      sketch_node_ptr = sketch_node.get();
      inst_node_map.insert(std::make_pair(inst, sketch_node_ptr));
      sketch_node_ptr->AddInstruction(inst);

      for (auto* prev_node : prev_sketch_nodes) {
        // set inputs of sketch node
        sketch_node->AddInput(prev_node);
      }

      graph_sketch->AddNode(std::move(sketch_node));
    } else {
      CHECK(!prev_sketch_nodes.empty());

      // map to the same sketch node as its operands'
      sketch_node_ptr = *prev_sketch_nodes.begin();
      inst_node_map.insert(std::make_pair(inst, sketch_node_ptr));
      sketch_node_ptr->AddInstruction(inst);
    }
  }

  auto* root_inst = computation->root_instruction();
  CHECK(inst_node_map.find(root_inst) != inst_node_map.end());
  auto* root_node = inst_node_map[root_inst];
  root_node->set_as_root();
  graph_sketch->set_root(root_node);

  return std::move(graph_sketch);
}

/*static*/
std::unique_ptr<GraphSketch>
GraphSketch::BuildFineGrainedSketch(HloModule* module,
                                    int physical_stage_num,
                                    const HloInstSet* inst_scope) {
  HloInstSet backward_insts = FindBackwardInsts(module, true);

  std::string back_insts_str;
  for (auto* inst : backward_insts) {
    back_insts_str += inst->ToString() + "\n";
  }
  VLOG(2) << "backward instructions:\n" << back_insts_str;

  std::unique_ptr<GraphSketch> graph_sketch =
      absl::make_unique<GraphSketch>(module, physical_stage_num);
  std::unordered_map<HloInstruction*, SketchNode*>& inst_node_map =
      graph_sketch->inst_node_map();

  HloComputation* computation = graph_sketch->SketchComputation();
  std::vector<HloInstruction*> post_order = computation->MakeInstructionPostOrder();

  for (auto* inst : post_order) {
    if (inst_scope && inst_scope->find(inst) == inst_scope->end()) {
      continue;
    }

    // create a new sketch node
    std::set<SketchNode*> prev_sketch_nodes;
    for (auto* operand : inst->operands()) {
      auto inst_node_it = inst_node_map.find(operand);
      if (inst_node_it != inst_node_map.end()) {
        prev_sketch_nodes.insert(inst_node_it->second);
      }
    }

    bool is_backward = false;
    if (backward_insts.find(inst) != backward_insts.end()) {
      is_backward = true;
    }
    std::unique_ptr<SketchNode> sketch_node = absl::make_unique<SketchNode>(inst, is_backward);
    SketchNode* sketch_node_ptr = sketch_node.get();
    inst_node_map.insert(std::make_pair(inst, sketch_node_ptr));
    sketch_node_ptr->AddInstruction(inst);

    for (auto* prev_node : prev_sketch_nodes) {
      // set inputs of sketch node
      sketch_node->AddInput(prev_node);
    }

    graph_sketch->AddNode(std::move(sketch_node));
  }

  auto* root_inst = computation->root_instruction();
  if (inst_node_map.find(root_inst) != inst_node_map.end()) {
    auto* root_node = inst_node_map[root_inst];
    root_node->set_as_root();
    graph_sketch->set_root(root_node);
  }

  return std::move(graph_sketch);
}

void
GraphSketch::MergeNodes(SketchNode* to_remove, SketchNode* living_nd) {
  VLOG(2) << "Merge node " << to_remove->name() << " to " << living_nd->name();
  HloMutableInstSet& insts = to_remove->instructions();
  for (auto* inst : insts) {
    inst_node_map_[inst] = living_nd;
    living_nd->AddInstruction(inst);
  }
  insts.clear();

  std::set<SketchNode*> obsolete_inputs = to_remove->inputs(); // copy is cheap here
  for (auto* input : obsolete_inputs) {
    to_remove->EraseInput(input);
    if (input != living_nd) {
      living_nd->AddInput(input);
    }
  }

  std::set<SketchNode*> obsolete_users = to_remove->users(); // copy is cheap here
  for (auto* user : obsolete_users) {
    user->EraseInput(to_remove);
    if (user != living_nd) {
      user->AddInput(living_nd);
    }
  }

  CHECK(to_remove->not_connected());
}

void
GraphSketch::Optimize() {
  AbsorbSingleUserNodes();
  //ClusterTinyNodes();
  RemoveIsolatedNodes();

  // debug
  VLOG(2) << "sketch node list:";
  for (auto& node : nodes_) {
    VLOG(2) << node->Dump();
  }
}

void
GraphSketch::AbsorbSingleUserNodes() {
  bool changed;
  int iter = 0;
  do {
    changed = false;
    for (auto& nd : nodes_) {
      std::vector<std::pair<SketchNode*, SketchNode*>> edges_to_cut;
      for (auto* prev_nd : nd->inputs()) {
        if (prev_nd->inputs().empty() &&
            prev_nd->user_count() == 1) {
          // move instruction of prev node into its user's SketchNode
          // then remove prev node
          VLOG(2) << "cut " << prev_nd->name() << " ->" << nd->name();
          edges_to_cut.emplace_back(std::make_pair(prev_nd, nd.get()));
        }
      }

      if (!edges_to_cut.empty()) {
        changed = true;
      }

      for (auto& edge : edges_to_cut) {
        edge.second->EraseInput(edge.first);
        MergeNodes(edge.first, edge.second);
      }
    }
    VLOG(2) << "AbsorbSingleUserNodes's iteration: " << ++iter;
  } while (changed);

  RemoveIsolatedNodes();

  return;
}

std::vector<SketchNode*>
GraphSketch::CollectFusableNodes() const {
  std::vector<SketchNode*> post_order = MakePostOrder();

  std::vector<SketchNode*> result;
  for (auto* node : post_order) {
    if (node->not_connected() ||
        !node->is_tiny()) {
      continue;
    }
    for (auto* user : node->users()) {
      if (user->is_tiny()) {
        // both node and its user are tiny
        // node -> user
        result.emplace_back(node);
        break;
      }
    }
  }

  return std::move(result);
}

bool
GraphSketch::IsFusable(const SketchNode* src, const SketchNode* tgt) const {
  std::unordered_set<SketchNode*> visited;
  std::deque<SketchNode*> check_list;
  for (auto* user : src->users()) {
    if (user != tgt) {
      check_list.push_back(user);
    }
  }

  while (!check_list.empty()) {
    auto* node = check_list.front();
    check_list.pop_front();
    visited.insert(node);

    for (auto* user : node->users()) {
      if (user == tgt) {
        // when this pattern happens, user and tgt can not be fused.
        // user -> other nodes -> tgt
        return false;
      }

      if (visited.find(user) != visited.end()) {
        continue;
      }
      check_list.push_back(user);
    }
  }
  return true;
}

void
GraphSketch::ClusterTinyNodes() {
  bool changed;
  int iter = 0;
  do {
    changed = false;

    std::vector<SketchNode*> fusable_nodes = CollectFusableNodes();
    std::set<SketchNode*> fused_nodes;

    const auto neighbor_fused = [&](SketchNode* node) {
      for (auto* user : node->users()) {
        if (fused_nodes.find(user) != fused_nodes.end()) {
          return true;
        }
      }

      for (auto* input : node->inputs()) {
        if (fused_nodes.find(input) != fused_nodes.end()) {
          return true;
        }
      }
      return false;
    };

    for (auto* fusable_nd : fusable_nodes) {
      for (auto* user : fusable_nd->users()) {
        if (!user->is_tiny()) {
          continue;
        }
        if (neighbor_fused(fusable_nd)) {
          // this node's neighbor is fused with another node already
          continue;
        }
        if (neighbor_fused(user)) {
          // this node's neighbor is fused with another node already
          continue;
        }
        if (IsFusable(fusable_nd, user)) {
          MergeNodes(fusable_nd, user);
          fused_nodes.insert(user);
          fused_nodes.insert(fusable_nd);
          changed = true;
          break;
        }
      }
    }

    VLOG(2) << "ClusterTinyNodes's iteration: " << ++iter;
  } while (changed);
}

void
GraphSketch::RemoveIsolatedNodes() {
  int pos = 0;
  int last_pos = nodes_.size() - 1;
  while (nodes_[last_pos]->not_connected() && last_pos >= 0) {
    --last_pos;  // last_pos may be -1, it is expected.
  }
  while (pos < last_pos) {
    if (nodes_[pos]->not_connected())  {
      // remove current node
      if (pos < last_pos) {
        nodes_[pos] = std::move(nodes_[last_pos]);
        --last_pos;

        while (nodes_[last_pos]->not_connected() && last_pos >= 0) {
          --last_pos;
        }
      }
    }
    ++pos;
  }

  if (last_pos < 0) {
    nodes_.clear();
  } else {
    nodes_.resize(last_pos+1);
  }
}

void
GraphSketch::ComputePostOrder(std::vector<SketchNode*>& post_order,
                  SketchNode* root,
                  absl::flat_hash_map<SketchNode*, VisitState>& visited) const {
  std::vector<SketchNode*> dfs_stack;
  dfs_stack.push_back(root);
  while (!dfs_stack.empty()) {
    const auto current = dfs_stack.back();
    auto it = visited.find(current);
    if (it != visited.end()) {
      if (it->second == kVisited) {
        // Already visited.
        dfs_stack.pop_back();
        continue;
      }
      // Visit this node.
      CHECK_EQ(kVisiting, it->second);
      dfs_stack.pop_back();
      post_order.push_back(current);
      it->second = kVisited;
      continue;
    }

    visited.insert({current, kVisiting});

    const auto& inputs = current->inputs();
    for (auto* input : inputs) {
      dfs_stack.emplace_back(input);
    }
  }
}

std::vector<SketchNode*>
GraphSketch::MakePostOrder() const {
  std::vector<SketchNode*> post_order;
  post_order.reserve(nodes_.size());
  absl::flat_hash_map<SketchNode*, VisitState> visited;
  for (auto& node : nodes_) {
    ComputePostOrder(post_order, node.get(), visited);
  }
  CHECK_EQ(nodes_.size(), post_order.size())
      << "number of sketch nodes does not match post order size";

  return post_order;
}

void
GraphSketch::EvalCost() {
  total_flop_count_ = 0;
  backward_flop_count_ = 0;
  forward_flop_count_ = 0;
  for (auto& node : nodes_) {
    node->EvalCost(inst_flops_map_);
    total_flop_count_ += node->flop_count();
    if (node->is_backward()) {
      backward_flop_count_ += node->flop_count();
    } else {
      forward_flop_count_ += node->flop_count();
    }
  }
  forward_per_stage_flops_ = (forward_flop_count_+physical_stage_num_-1) / physical_stage_num_;
  backward_per_stage_flops_ = (backward_flop_count_+physical_stage_num_-1) / physical_stage_num_;
  int64 unbalanced_ratio = ServiceEnv::unbalanced_ratio();
  if (unbalanced_ratio >= 100 || unbalanced_ratio <= 0) {
    LOG(FATAL) << "UNBALANCED_RATIO is out of range. It should be an integer in range (0, 100).";
  } else {
    unbalanced_ratio_ = (int)unbalanced_ratio;
  }
  VLOG(0) << "total_flop_count_: " << total_flop_count_;
  VLOG(0) << "forward_per_stage_flops_: " << forward_per_stage_flops_;
  VLOG(0) << "backward_per_stage_flops_: " << backward_per_stage_flops_;
  VLOG(0) << "UNBALANCED_RATIO: " << unbalanced_ratio_ << "%";
}

void
GraphSketch::DeterminePrePostNodes() {
  std::vector<SketchNode*> post_order = MakePostOrder();

  // determine pre nodes for each node
  for (auto* node : post_order) {
    const std::set<SketchNode*>& inputs = node->inputs();
    for (auto* input : inputs) {
      // add a direct pre node
      node->AddPreNode(input);
      if (!input->is_tiny()) {
        node->AddPreCompSensNode(input);
      }

      // add indirect pre nodes
      const std::set<SketchNode*>& pre_pre_nodes = input->pre_nodes();
      node->AddPreNodes(pre_pre_nodes);
      const std::set<SketchNode*>& pre_pre_comp_sens_nodes = input->pre_comp_sens_nodes();
      node->AddPreCompSensNodes(pre_pre_comp_sens_nodes);
    }
    // calculate accumulated pre flop count
    for (auto* pre_node : node->pre_nodes()) {
      node->AccuPreFlopCount(pre_node->flop_count());
      node->CalcRelativeAccuPre(total_flop_count_);
    }
  }

  // determine post nodes for each node
  for (auto rit = post_order.rbegin(); rit != post_order.rend(); ++rit) {
    auto* node = *rit;
    int64 max_acc_post_flops = 0;
    SketchNode* mp_user = nullptr;
    const std::set<SketchNode*>& users = node->users();
    for (auto* user : users) {
      // add a direct post node
      node->AddPostNode(user);
      if (!user->is_tiny()) {
        node->AddPostCompSensNode(user);
      }

      // add indirect post nodes
      const std::set<SketchNode*>& post_post_nodes = user->post_nodes();
      node->AddPostNodes(post_post_nodes);
      const std::set<SketchNode*>& post_post_comp_sens_nodes = user->post_comp_sens_nodes();
      node->AddPostCompSensNodes(post_post_comp_sens_nodes);

      if (user->mp_acc_post_flops() > max_acc_post_flops) {
        max_acc_post_flops = user->mp_acc_post_flops();
        mp_user = user;
      }
    }

    // calculate accumulated post flop count
    for (auto* post_node : node->post_nodes()) {
      node->AccuPostFlopCount(post_node->flop_count());
      node->CalcRelativeAccuPost(total_flop_count_);
    }

    node->set_mp_acc_post_flops(max_acc_post_flops);
    node->AccMpAccPostFlops(node->flop_count());
    node->set_mp_user(mp_user);
  }

  VLOG(2) << "comp_sens_node_count: " << comp_sens_node_count();

  // calculate freedom degree for each node
  for (auto* node : post_order) {
    node->set_asap_rank(node->pre_node_count());
    node->set_alap_rank(node_count() - node->post_node_count() - 1);
    node->set_asap_comp_sens_rank(node->pre_comp_sens_node_count());
    if (node->is_tiny()) {
      node->set_alap_comp_sens_rank(comp_sens_node_count() - node->post_comp_sens_node_count());
    } else {
      node->set_alap_comp_sens_rank(comp_sens_node_count() - node->post_comp_sens_node_count() - 1);
    }
    VLOG(2) << node->name() << ", pre node count: " << node->pre_node_count()
            << ", post node count: " << node->post_node_count() << std::endl
            << ", pre comp sens node count: " << node->pre_comp_sens_node_count() << std::endl
            << ", post comp sens node count: " << node->post_comp_sens_node_count() << std::endl
            << ", accumulated pre flop count: " << node->accum_pre_flop_count() << std::endl
            << ", accumulated post flop count: " << node->accum_post_flop_count() << std::endl
            << ", asap comp sens rank: " << node->asap_comp_sens_rank() << std::endl
            << ", alap comp sens rank: " << node->alap_comp_sens_rank() << std::endl
            << ", freedom degree: " << node->FreedomDegree() << std::endl;
  }
}

std::vector<const HloInstruction*>
GraphSketch::FindCriticalInsts() {
  std::vector<const HloInstruction*> critical_insts;
  EvalCost();
  if (total_flop_count_ == 0) {
    return std::move(critical_insts);
  }
  DeterminePrePostNodes();
  FindMainPath();

  for (auto* node : critical_nodes_) {
    const HloInstruction* core_inst = node->core_instr();
    CHECK(is_compute_intensive(core_inst));
    critical_insts.emplace_back(core_inst);
  }

  return std::move(critical_insts);
}

void
GraphSketch::FindMainPath() {
  main_path_nodes_.clear();
  critical_nodes_.clear();

  // find max accumulated path
  SketchNode* max_path_front = nullptr;
  int64 max_acc_flops = 0;
  for (auto& node : nodes_) {
    if (node->inputs().empty()) {
      if (node->mp_acc_post_flops() > max_acc_flops) {
        max_path_front = node.get();
        max_acc_flops = node->mp_acc_post_flops();
      }
    }
  }

  CHECK(max_path_front);
  SketchNode* path_node = max_path_front;
  VLOG(2) << "main path nodes:";
  while (path_node) {
    if (!path_node->is_tiny()) {
      VLOG(2) << "  " << path_node->name();
      main_path_nodes_.emplace_back(path_node);
      if (path_node->FreedomDegree() == 0) {
        critical_nodes_.emplace_back(path_node);
      }
    }
    path_node = path_node->mp_user();
  }
}

bool
GraphSketch::BuildOpGroupInfo(
        const HloInstSet& inst_scope,
        std::map<int, std::shared_ptr<OpGroupInfo>>& op_groups) {
  HloModule::DefContext* def_ctx = module_->def_ctx();

  int var_count = module_->variable_map()->size();
  HloComputation* entry = module_->entry_computation();
  HloInstruction* root = entry->root_instruction();
  int input_var_offset = entry->num_parameters() - var_count;

  VLOG(1) << "alias map size: " << def_ctx->input_output_alias_map_.size();
  for (auto& alias_pair : def_ctx->input_output_alias_map_) {
    int p_idx = alias_pair.first;
    const HloInstruction* param = entry->parameter_instruction(p_idx);
    int out_idx = alias_pair.second;
    HloInstruction* out = root->mutable_operand(out_idx);
    VLOG(2) << "input idx: " << p_idx << ", inst: " << param->name()
            << ", output idx: " << out_idx << ", inst: " << out->name();
    if (param->metadata().op_group() > 0) {
      VLOG(2) << "orig op_group: " << out->metadata().op_group();
      out->metadata().set_op_group(param->metadata().op_group());
      VLOG(2) << "new op_group: " << out->metadata().op_group();
    }
  }

  //std::set<int> id_set;
  for (const HloInstruction* inst : computation_->instructions()) {
    if (inst_scope.find(inst) == inst_scope.end()) {
      continue;
    }
    if (inst->metadata().op_group() > 0) {
      int group_id = inst->metadata().op_group();
      std::shared_ptr<OpGroupInfo> op_info;
      if (op_groups.find(group_id) == op_groups.end()) {
        op_info = std::make_shared<OpGroupInfo>(group_id);
        op_groups[group_id] = op_info;

        //id_set.insert(inst->metadata().op_group());
      } else {
        op_info = op_groups[group_id];
      }

      op_info->AddInst(inst);
    }
  }

  std::unordered_map<const HloInstruction*, std::shared_ptr<std::set<int>>> pre_group_map;
  std::unordered_map<const HloInstruction*, std::shared_ptr<std::set<int>>> direct_pre_group_map;
  std::vector<HloInstruction*> post_order = computation_->MakeInstructionPostOrder();
  for (const HloInstruction* inst : post_order) {
    if (inst_scope.find(inst) == inst_scope.end()) {
      continue;
    }

    std::set<int> pre_groups;
    std::set<int> direct_pre_groups;
    for (auto* op : inst->operands()) {
      if (inst_scope.find(op) == inst_scope.end()) {
        continue;
      }
      std::shared_ptr<std::set<int>> op_pre;
      std::shared_ptr<std::set<int>> op_direct_pre;
      if (pre_group_map.find(op) != pre_group_map.end()) {
        op_pre = pre_group_map[op];
      }

      if (op->metadata().op_group() > 0 &&
          op->metadata().op_group() != inst->metadata().op_group()) {
        // add direct pre group id
        pre_groups.insert(op->metadata().op_group());
        direct_pre_groups.insert(op->metadata().op_group());
      }

      if (op_pre) {
        // add indirect pre group ids
        pre_groups.insert(op_pre->begin(), op_pre->end());
      }

      if (op->metadata().op_group() <= 0) {
        // pass direct pre groups only when op is not owned by a valid group
        if (direct_pre_group_map.find(op) != direct_pre_group_map.end()) {
          op_direct_pre = direct_pre_group_map[op];
          CHECK(op_direct_pre);
          direct_pre_groups.insert(op_direct_pre->begin(), op_direct_pre->end());
        }
      }
    }

    pre_groups.erase(inst->metadata().op_group());
    direct_pre_groups.erase(inst->metadata().op_group());

    if (!pre_groups.empty()) {
      bool refer_op = false;
      for (auto* op : inst->operands()) {
        if (inst_scope.find(op) == inst_scope.end()) {
          continue;
        }
        std::shared_ptr<std::set<int>> op_pre;
        if (pre_group_map.find(op) != pre_group_map.end()) {
          op_pre = pre_group_map[op];
          CHECK(op_pre);
          CHECK(!op_pre->empty());
          if (std::equal(op_pre->begin(), op_pre->end(),
                         pre_groups.begin(), pre_groups.end())) {
            pre_group_map[inst] = op_pre;
            refer_op = true;
            break;
          }
        }
      }

      if (!refer_op) {
        pre_group_map[inst] = std::make_shared<std::set<int>>(std::move(pre_groups));
      }
    }

    if (!direct_pre_groups.empty()) {
      bool refer_op = false;
      for (auto* op : inst->operands()) {
        if (inst_scope.find(op) == inst_scope.end()) {
          continue;
        }
        if (op->metadata().op_group() <= 0) {
          std::shared_ptr<std::set<int>> op_direct_pre;
          if (direct_pre_group_map.find(op) != direct_pre_group_map.end()) {
            op_direct_pre = direct_pre_group_map[op];
            CHECK(op_direct_pre);
            CHECK(!op_direct_pre->empty());
            if (std::equal(op_direct_pre->begin(), op_direct_pre->end(),
                           direct_pre_groups.begin(), direct_pre_groups.end())) {
              direct_pre_group_map[inst] = op_direct_pre;
              refer_op = true;
              break;
            }
          }
        }
      }

      if (!refer_op) {
        direct_pre_group_map[inst] =
                std::make_shared<std::set<int>>(std::move(direct_pre_groups));
      }
    }
  }

  for (const HloInstruction* inst : post_order) {
    if (inst_scope.find(inst) == inst_scope.end()) {
      continue;
    }

    if (inst->metadata().op_group() <= 0) {
      continue;
    }

    if (pre_group_map.find(inst) != pre_group_map.end()) {
      std::shared_ptr<std::set<int>> pre_groups = pre_group_map[inst];
      CHECK(pre_groups);
      CHECK(!pre_groups->empty());
      CHECK(op_groups.find(inst->metadata().op_group()) != op_groups.end());
      std::shared_ptr<OpGroupInfo> inst_group_info = op_groups[inst->metadata().op_group()];
      CHECK(inst_group_info);
      for (int group_id : *pre_groups) {
        inst_group_info->AddPreGroupId(group_id);
      }
    }

    if (direct_pre_group_map.find(inst) != direct_pre_group_map.end()) {
      std::shared_ptr<std::set<int>> direct_pre_groups = direct_pre_group_map[inst];
      CHECK(direct_pre_groups);
      CHECK(!direct_pre_groups->empty());
      CHECK(op_groups.find(inst->metadata().op_group()) != op_groups.end());
      std::shared_ptr<OpGroupInfo> inst_group_info = op_groups[inst->metadata().op_group()];
      CHECK(inst_group_info);
      for (int group_id : *direct_pre_groups) {
        inst_group_info->AddDirectPreGroupId(group_id);
      }
    }
  }

  for (auto& group_info : op_groups) {
    VLOG(2) << "op group:\n" << group_info.second->ToString();
  }

  // check circlely dependency
  for (auto& group_info : op_groups) {
    std::shared_ptr<OpGroupInfo> group = group_info.second;
    for (int pre_id : group->pre_ids()) {
      CHECK(op_groups.find(pre_id) != op_groups.end());
      std::shared_ptr<OpGroupInfo> pre_group = op_groups[pre_id];
      if (pre_group->InPreGroups(group->id())) {
        VLOG(1) << "group circle found between groups " << pre_id << " and " << group->id();
        return false;
      }
    }
  }

  return true;
}

void
MirrorCopyGroupInfo(
          std::map<int, std::shared_ptr<OpGroupInfo>>& forward_groups,
          std::map<int, std::shared_ptr<OpGroupInfo>>& backward_groups) {
  auto copy_mirror_info = [](std::shared_ptr<OpGroupInfo> src,
                             std::shared_ptr<OpGroupInfo> tgt) {
    tgt->set_mirror_direct_post_ids(&src->direct_pre_ids());
    tgt->set_mirror_post_ids(&src->pre_ids());
  };

  for (auto& group_info : forward_groups) {
    int fg_id = group_info.first;
    if (backward_groups.find(fg_id) != backward_groups.end()) {
      std::shared_ptr<OpGroupInfo> back_group_info = backward_groups[fg_id];
      copy_mirror_info(back_group_info, group_info.second);
    }
  }

  for (auto& group_info : backward_groups) {
    int bg_id = group_info.first;
    if (forward_groups.find(bg_id) != forward_groups.end()) {
      std::shared_ptr<OpGroupInfo> forward_group_info = forward_groups[bg_id];
      copy_mirror_info(forward_group_info, group_info.second);
    }
  }
}

std::set<std::pair<int/*src*/, int/*tgt*/>>
CollectGroupDependency(
        const std::map<int, std::shared_ptr<OpGroupInfo>>& group_info_map) {
  std::set<std::pair<int, int>> dependencies;
  for (auto& group_info : group_info_map) {
    auto info = group_info.second;
    for (int pre_id : info->direct_pre_ids()) {
      dependencies.insert(std::make_pair(pre_id, info->id()));
    }
    if (info->mirror_direct_post_ids()) {
      for (int post_id : *info->mirror_direct_post_ids()) {
        if (group_info_map.find(post_id) != group_info_map.end()) {
          dependencies.insert(std::make_pair(info->id(), post_id));
        }
      }
    }
  }

  return std::move(dependencies);
}

bool
GraphSketch::StagePlan() {
  HloInstSet forward_inst_scope = FindForwardInsts(module_, true);
  BuildOpGroupInfo(forward_inst_scope, forward_op_groups_);

  HloInstSet back_inst_scope = FindBackwardInsts(module_, true);
  BuildOpGroupInfo(back_inst_scope, backward_op_groups_);

  MirrorCopyGroupInfo(forward_op_groups_, backward_op_groups_);

  std::set<std::pair<int/*src*/, int/*tgt*/>> forward_group_depend =
                                    CollectGroupDependency(forward_op_groups_);

  std::set<std::pair<int/*src*/, int/*tgt*/>> back_group_depend =
                                    CollectGroupDependency(backward_op_groups_);

  for (auto& depend : back_group_depend) {
    if (forward_op_groups_.find(depend.first) != forward_op_groups_.end() ||
        forward_op_groups_.find(depend.second) != forward_op_groups_.end()) {
      forward_group_depend.insert(std::make_pair(depend.second, depend.first));  // revert first and second
    }
  }

  for (auto& depend : forward_group_depend) {
    if (backward_op_groups_.find(depend.first) != backward_op_groups_.end() ||
        backward_op_groups_.find(depend.second) != backward_op_groups_.end()) {
      back_group_depend.insert(std::make_pair(depend.second, depend.first));  // revert first and second
    }
  }

  auto dump_group_depend = [] (std::set<std::pair<int/*src*/, int/*tgt*/>> group_depend)
                                        -> std::string {
    // debug
    std::string dump_str;
    int i = 0;
    for (auto& depend : group_depend) {
      dump_str += "(" + std::to_string(depend.first) + ", "
                  + std::to_string(depend.second) + "), ";
      ++i;
      if (i%10 == 0) {
        dump_str += "\n";
      }
    }

    return dump_str;
  };

  std::string forward_dump_str = dump_group_depend(forward_group_depend);
  VLOG(2) << "forward group dependencies:\n" << forward_dump_str;

  std::string back_dump_str = dump_group_depend(back_group_depend);
  VLOG(2) << "\nbackward group dependencies:\n" << back_dump_str;

  EvalCost();
 
  std::ofstream of("sketch_raw.dot", std::ios::ate);
  of << Dump() << std::endl;
  of.close();

  std::vector<SketchNode*> post_order = MakePostOrder();

  CalcPreComputeCost(post_order, true);
  CalcPostComputeCost(post_order, true);

  CHECK(physical_stage_num_>=1);

  VLOG(0) << "physical_stage_num: " << physical_stage_num_;
  if (!ForwardPlan(physical_stage_num_, post_order, forward_group_depend,
                     forward_inst_scope)) {
    return false;
  }

  HloInstMap<std::pair<int/*asap*/, int/*alap*/>> dummy_stage_range_map;
  if (ServiceEnv::debug()) {
    VLOG(0) << "[forward] forward stage of pipeline:";
    std::string stage_info = DumpStages(dummy_stage_range_map, nullptr);
    VLOG(0) << stage_info;
  }

  auto map_to_mirror_stages = []
                              (int logic_stage_num,
                               const HloInstMap<int>& inst_stage_map,
                               HloInstMap<int>& mirror_stage_map) {
    for (auto& inst_stage : inst_stage_map) {
      mirror_stage_map[inst_stage.first] = logic_stage_num - inst_stage.second - 1;
    }
  };

  HloInstMap<int> mirror_stage_map;
  map_to_mirror_stages(physical_stage_num_<<1, inst_stage_map_v2_,
                       mirror_stage_map);


  VLOG(2) << "backward node cost";
  CalcPreComputeCost(post_order, false);
  CalcPostComputeCost(post_order, false);

  bool status = BackwardPlan(physical_stage_num_, post_order,
                             mirror_stage_map, back_group_depend, back_inst_scope);

  auto* root_inst = computation_->root_instruction();
  inst_stage_map_v2_[root_inst] = logic_stage_num_ - 1;

  status = status && DetectCycle();
  if (status) {
    module_->record_split_info(physical_stage_num_, false);
    module_->set_stage_split_ordinal(module_->split_nums().size() - 1);
    std::vector<int> layout(module_->placement_layout());
    CHECK(layout.size());
    int major = layout[0];
    layout.erase(layout.begin());
    layout.emplace_back(major);
    module_->set_placement_layout(layout);
    RecordStagePlanIntoInsts();
  }

  return status;
}

std::string
GraphSketch::DumpStages(
    const HloInstMap<std::pair<int/*asap*/, int/*alap*/>>& inst_stage_range_map,
    const std::vector<SketchNode*>* sketch_nodes) const {
  std::string res;
  res += "stage info:";
  if (sketch_nodes) {
    for (auto* node : *sketch_nodes) {
      res += "\n  " + node->name() + ":";
      res += "    (asap: " + std::to_string(node->asap_stage_id())
             + ", alap: " + std::to_string(node->alap_stage_id()) + ")";
      if (node->pre_accum_comp_info()) {
        res += "\n    pre info: " + node->pre_accum_comp_info()->ToString();
      }
      if (node->post_accum_comp_info()) {
        res += "\n    post info: " + node->post_accum_comp_info()->ToString();
      }
    }
  }

  res += "\n\ninstruction stage info:\ninstruction : stage";
  res += "\nnumber: " + std::to_string(inst_stage_map_v2_.size());
  for (auto& inst_stage : inst_stage_map_v2_) {
    res += "\n  " + inst_stage.first->name() + "(group:"
           + std::to_string(inst_stage.first->metadata().op_group())
           + ") : " + std::to_string(inst_stage.second);
  }

  res += "\ninstruction : (asap, alap)";
  res += "\nnumber: " + std::to_string(inst_stage_range_map.size());
  for (auto& inst_rng : inst_stage_range_map) {
    res += "\n  " + inst_rng.first->ToString() + "(group:"
           + std::to_string(inst_rng.first->metadata().op_group()) + ") : ("
           + std::to_string(inst_rng.second.first) + ", "
           + std::to_string(inst_rng.second.second) + ")";
  }

  return res;
}

HloInstMap<std::pair<int/*asap*/, int/*alap*/>>
GraphSketch::InferStageRangeByFlops(
                              int stage_num,
                              int start_stage,
                              bool is_forward,
                              const std::vector<SketchNode*>& sketch_nodes) {
  int64 total_flop_count = 0;
  if (is_forward) {
    total_flop_count = forward_flop_count_;
  } else {
    total_flop_count = backward_flop_count_;
  }

  CHECK(unbalanced_ratio_ > 0 && unbalanced_ratio_ < 100);
  int64 flops_per_stage = (total_flop_count+stage_num-1) / stage_num;
  int64 flop_tolerance = flops_per_stage * unbalanced_ratio_ / 100;  // allow tolerance

  VLOG(1) << "total_flop_count: " << total_flop_count;
  VLOG(1) << "flops_per_stage: " << flops_per_stage;
  VLOG(1) << "flop_tolerance: " << flop_tolerance;

  for (auto* node : sketch_nodes) {
    if ((is_forward && !node->is_backward()) ||
        (!is_forward && node->is_backward())) {
      VLOG(2) << "node name: " << node->name()
              << ", flop_count: " << node->flop_count();
      // 1. determine first possible stage (as soon as possible)
      std::shared_ptr<AccumCompInfo> pre_accum = node->pre_accum_comp_info();
      int64 pre_accum_flops = 0;
      if (pre_accum) {
        pre_accum_flops = pre_accum->accum_flop_count_;
        if (pre_accum_flops>flop_tolerance) {
          pre_accum_flops -= flop_tolerance;
        } else {
          pre_accum_flops = 0;
        }
      }
      pre_accum_flops += (node->flop_count() / 2);
      int asap_stage_id = pre_accum_flops / flops_per_stage;
      VLOG(2) << "node: " << node->name() << ", asap: " << start_stage+asap_stage_id;
      node->set_asap_stage_id(start_stage+asap_stage_id);

      // 2. determine last possible stage (as late as possible)
      std::shared_ptr<AccumCompInfo> post_accum = node->post_accum_comp_info();
      int64 post_accum_flops = 0;
      if (post_accum) {
        post_accum_flops = post_accum->accum_flop_count_;
      }

      int64 post_flops = post_accum_flops + (node->flop_count() >> 1);
      if (post_flops > flop_tolerance) {
        post_flops -= flop_tolerance;
      } else {
        post_flops = 0;
      }
      int post_stage_num = post_flops / flops_per_stage;
      int alap_stage_id = stage_num - post_stage_num - 1;
      VLOG(2) << "node: " << node->name() << ", alap: " << start_stage+alap_stage_id;
      node->set_alap_stage_id(start_stage+alap_stage_id);
    }
  }

  for (auto* node : sketch_nodes) {
    const HloInstruction* core_inst = node->core_instr();
    if (is_compute_intensive(core_inst) &&
        inst_stage_map_v2_.find(core_inst) != inst_stage_map_v2_.end()) {
      int stage = inst_stage_map_v2_.at(core_inst);
      VLOG(2) << "stage is aligned for inst:\n" << core_inst->ToString()
              << "\nstage: " << stage;
      node->set_asap_stage_id(stage);
      node->set_alap_stage_id(stage);
    }
  }

  HloInstMap<std::pair<int/*asap*/, int/*alap*/>> inst_stage_range_map;
  for (auto* node : sketch_nodes) {
    if ((is_forward && node->is_backward()) ||
        (!is_forward && !node->is_backward())) {
      continue;
    }
    if (node->asap_stage_id()>=0 && node->alap_stage_id()>=0) {
      if (node->asap_stage_id() == node->alap_stage_id()) {
        for (auto* inst : node->instructions()) {
          VLOG(2) << "determined inst: " << inst->ToString() << ", stage: " << node->asap_stage_id();
          inst_stage_map_v2_[inst] = node->asap_stage_id();
        }
      } else {
        CHECK(node->asap_stage_id() < node->alap_stage_id()) << "node: "
            << node->name() << ", asap: " << node->asap_stage_id() << ", alap: "
            << node->alap_stage_id();
        for (auto* inst : node->instructions()) {
          VLOG(2) << "undetermined inst: " << inst->ToString();
          inst_stage_range_map[inst] =
                  std::make_pair(node->asap_stage_id(), node->alap_stage_id());
        }
      }
    }
  }

  std::string stage_info = DumpStages(inst_stage_range_map, &sketch_nodes);
  VLOG(2) << stage_info;

  return std::move(inst_stage_range_map);
}

HloInstMap<std::pair<int/*asap*/, int/*alap*/>>
GraphSketch::InferStageRangeByContext(const HloInstSet& inst_scope,
                                      int stage_num, int start_stage) {
  HloInstMap<std::pair<int/*asap*/, int/*alap*/>> inst_stage_range_map;

  bool changed = true;
  while (changed) {
    changed = false;
    inst_stage_range_map.clear();

    std::unordered_map<const HloInstruction*, int> pre_stage_map;
    std::unordered_map<const HloInstruction*, int> succ_stage_map;
    FindNearestStages(inst_scope, inst_stage_map_v2_,
                      pre_stage_map, succ_stage_map);

    int end_stage = start_stage + stage_num - 1;
    for (auto& pre_stage_info : pre_stage_map) {
      int pre_stage = pre_stage_info.second;

      int succ_stage = INT_MAX;
      if (succ_stage_map.find(pre_stage_info.first) != succ_stage_map.end()) {
        succ_stage = succ_stage_map.at(pre_stage_info.first);
      }

      VLOG(2) << "check middle inst: " << pre_stage_info.first->name();
      if (pre_stage == succ_stage) {
        VLOG(2) << "determine middle stage: " << pre_stage;
        CHECK(pre_stage>=0 && pre_stage != INT_MAX);
        inst_stage_map_v2_[pre_stage_info.first] = pre_stage;
        changed = true;
      } else {
        if (pre_stage<0) {
          pre_stage = start_stage;
        }
        if (succ_stage == INT_MAX) {
          succ_stage = end_stage;
        }

        VLOG(2) << "pre stage, succ stage: " << pre_stage << ", " << succ_stage;
        inst_stage_range_map[pre_stage_info.first] = std::make_pair(pre_stage, succ_stage);
      }
    }

    for (auto& succ_stage_info : succ_stage_map) {
      if (pre_stage_map.find(succ_stage_info.first) != pre_stage_map.end()) {
        continue;
      }

      int succ_stage = succ_stage_info.second;

      if (succ_stage != INT_MAX) {
        inst_stage_range_map[succ_stage_info.first] = std::make_pair(start_stage, succ_stage);
      } else {
        inst_stage_range_map[succ_stage_info.first] = std::make_pair(start_stage, end_stage);
      }
    }
  }

  return std::move(inst_stage_range_map);
}

bool
GraphSketch::ForwardPlan(int forward_stage_num,
                           const std::vector<SketchNode*>& sketch_nodes,
                           const std::set<std::pair<int, int>>& group_depend,
                           const HloInstSet& inst_scope) {
  InferStageRangeByFlops(forward_stage_num, 0, true, sketch_nodes);

  MapUnstagedInsts(0, inst_scope);

  HloInstMap<std::pair<int/*asap*/, int/*alap*/>> inst_stage_range_map =
                  InferStageRangeByContext(inst_scope, forward_stage_num, 0);

  VLOG(2) << "initial stage of forward(before ilp)";
  std::string stage_info = DumpStages(inst_stage_range_map, nullptr);
  VLOG(2) << stage_info;

  VLOG(2) << "start to build pipeline ilp model";
  IlpStageModel ilp_mod;
  ilp_mod.set_logic_stage_start(0);
  ilp_mod.set_logic_stage_end(forward_stage_num-1);
  ilp_mod.set_per_stage_flops(forward_per_stage_flops_);
  ilp_mod.set_physical_stage_num(physical_stage_num_);
  ilp_mod.set_unbalanced_ratio(unbalanced_ratio_);

  HloInstMap<int> dummy_mirror_stage_map;
  bool succ = BuildIlpStageModel(inst_scope, 0, forward_stage_num-1,
                                 inst_stage_map_v2_, inst_stage_range_map,
                                 dummy_mirror_stage_map, forward_op_groups_,
                                 group_depend, true, ilp_mod);
  if (!succ) {
    return false;
  }
  std::string ilp_mod_str = ilp_mod.ExportToString();
  VLOG(2) << "pipeline ilp model(forward):";
  VLOG(2) << ilp_mod_str;

  std::unordered_map<const HloInstruction*, int/*stage id*/> solver_res;
  succ = ilp_mod.Solve(solver_res);

  if (!succ) {
    return false;
  }

  bool debug = ServiceEnv::debug();
  if (debug) {
    VLOG(0) << "inst_stage_range_map size: " << inst_stage_range_map.size();
    VLOG(0) << "ILP solver result for pipeline stages:";
    VLOG(0) << "instruction number: " << solver_res.size();
  }
  std::string ilp_solver_res_str="instruction : stage";
  for (auto& inst_stage : solver_res) {
    ilp_solver_res_str += "\n  " + inst_stage.first->name() + " : "
                          + std::to_string(inst_stage.second);
    CHECK(inst_stage_map_v2_.find(inst_stage.first) == inst_stage_map_v2_.end())
          << inst_stage.first->name();
    inst_stage_map_v2_[inst_stage.first] = inst_stage.second;
    inst_stage_range_map.erase(inst_stage.first);
  }

  if (debug) {
    VLOG(0) << ilp_solver_res_str;
    VLOG(0) << "inst_stage_range_map size: " << inst_stage_range_map.size();
  }

  return true;
}

bool
GraphSketch::AssignStageSafely(const HloInstSet& inst_scope) {
  std::unordered_set<const HloInstruction*> mult_local_user_insts;
  for (auto* inst : computation_->instructions()) {
    if (inst->user_count() > 1) {
      int local_user_count = 0;
      for (auto* user : inst->users()) {
        if (inst_scope.find(user) != inst_scope.end()) {
          ++local_user_count;
        }
        if (local_user_count > 1) {
          break;
        }
      }

      if (local_user_count > 1) {
        // mult-user instruction
        mult_local_user_insts.insert(inst);
      }
    }
  }

  auto max_op_stage = [&inst_scope, &mult_local_user_insts, this]
                          (const HloInstruction* inst) -> std::pair<int, bool> {
    if (inst_scope.find(inst) == inst_scope.end()) {
      VLOG(2) << "out of scope: inst: " << inst->name();
      return std::make_pair(INT_MAX, false);
    }

    if (inst_stage_map_v2_.find(inst) != inst_stage_map_v2_.end()) {
      VLOG(2) << "stage assigned: inst: " << inst->name();
      return std::make_pair(INT_MAX, false);
    }

    if (mult_local_user_insts.find(inst) != mult_local_user_insts.end()) {
      // instruction with mult-user
      VLOG(2) << "mult local user: inst: " << inst->name();
      return std::make_pair(INT_MAX, false);
    }

    int max_op_stage_val = -1;
    for (auto* op : inst->operands()) {
      if (inst_stage_map_v2_.find(op) != inst_stage_map_v2_.end()) {
        if (max_op_stage_val < inst_stage_map_v2_[op]) {
          max_op_stage_val = inst_stage_map_v2_[op];
        }
      } else {
        VLOG(2) << "op stage not assigned: inst: " << inst->name();
        return std::make_pair(INT_MAX, false);
      }

      if (mult_local_user_insts.find(op) != mult_local_user_insts.end()) {
        // instruction which op is consumed by mult-user
        return std::make_pair(INT_MAX, false);
      }
    }

    VLOG(2) << "max op stage: " << max_op_stage_val << ", for inst: " << inst->name();
    return std::make_pair(max_op_stage_val, true);
  };

  int orig_staged_num = inst_stage_map_v2_.size();

  // initialize ready queue
  std::deque<const HloInstruction*> ready_insts;
  for (auto* inst : computation_->instructions()) {
    std::pair<int, bool> max_op_stage_val = max_op_stage(inst);
    if (max_op_stage_val.second == false) {
      continue;
    }

    if (inst->operand_count() == 0) {
      // place stage id same with max op
      // because there is no input, max op stage should be -1
      inst_stage_map_v2_[inst] = max_op_stage_val.first;
      VLOG(2) << "safely assign inst: " << inst->name() << " with stage: "
              << max_op_stage_val.first;
      continue;
    }

    int64 inst_bytes = ShapeUtil::ByteSizeOf(inst->shape(), 8);
    int64 in_bytes = 0;
    for (auto* op : inst->operands()) {
      if (inst_stage_map_v2_.find(op) != inst_stage_map_v2_.end()) {
        if (max_op_stage_val.first == inst_stage_map_v2_[op]) {
          in_bytes += ShapeUtil::ByteSizeOf(op->shape(), 8);
        }
      }
    }

    if (in_bytes >= inst_bytes) {
      ready_insts.push_back(inst);

      // place stage id same with max op
      inst_stage_map_v2_[inst] = max_op_stage_val.first;
      VLOG(2) << "safely assign inst: " << inst->name() << " with stage: "
              << max_op_stage_val.first;
    }
  }

  // iteratively assign stages
  while (!ready_insts.empty()) {
    auto* inst = ready_insts.front();
    ready_insts.pop_front();

    std::pair<int, bool> max_op_stage_val = max_op_stage(inst);
    if (max_op_stage_val.second == false) {
      VLOG(2) << "skip inst: " << inst->name();
      continue;
    }

    VLOG(2) << "try assign inst: " << inst->name();
    for (auto* user : inst->users()) {
      int64 inst_bytes = ShapeUtil::ByteSizeOf(user->shape(), 8);
      int64 in_bytes = 0;
      for (auto* op : user->operands()) {
        if (inst_stage_map_v2_.find(op) != inst_stage_map_v2_.end()) {
          if (max_op_stage_val.first == inst_stage_map_v2_[op]) {
            in_bytes += ShapeUtil::ByteSizeOf(op->shape(), 8);
          }
        }
      }

      if (in_bytes >= inst_bytes) {
        ready_insts.push_back(user);

        // place stage id same with max op
        inst_stage_map_v2_[user] = max_op_stage_val.first;
        VLOG(2) << "safely assign inst: " << user->name() << " with stage: "
                << max_op_stage_val.first;
      }
    }
  }

  // overwrite negative stage, replace it with its user's stage
  std::vector<std::pair<const HloInstruction*, int>> starts;
  for (auto* inst : inst_scope) {
    if (inst_stage_map_v2_.find(inst) != inst_stage_map_v2_.end()) {
      if (inst_stage_map_v2_[inst] == -1) {
        int user_stage = INT_MAX;
        for (auto* user : inst->users()) {
          if (inst_stage_map_v2_.find(user) != inst_stage_map_v2_.end()) {
            if (user_stage > inst_stage_map_v2_[user]) {
              user_stage = inst_stage_map_v2_[user];
            }
          }
        }

        VLOG(2) << "try to overwrite negative stage: " << inst->name()
                << ", user stage: " << user_stage;
        if (user_stage != INT_MAX && user_stage >= 0) {
          // overwrite negative stage
          inst_stage_map_v2_[inst] = user_stage;
          VLOG(2) << "[overwrite] safely assign inst: " << inst->name()
                  << " with stage: " << user_stage;
          starts.push_back(std::make_pair(inst, user_stage));
        }
      }
    }
  }

  for (auto& start : starts) {
    // back assign stage value from start
    std::pair<const HloInstruction*, int> cur_stage = start;
    while (!cur_stage.first->operands().empty()) {
      CHECK(cur_stage.first->operand_count() == 1);
      const HloInstruction* next_inst = cur_stage.first->operand(0);
      CHECK(inst_stage_map_v2_.find(next_inst) != inst_stage_map_v2_.end());
      CHECK(inst_stage_map_v2_.at(next_inst) == -1);
      CHECK(cur_stage.second >= 0);
      inst_stage_map_v2_[next_inst] = cur_stage.second;
      VLOG(2) << "[back update negative stage] safely assign inst: "
              << next_inst->name() << " with stage: " << cur_stage.second;
      cur_stage.first = next_inst;
    }
  }

  // remove remaining negative stage
  HloInstMap<int>::iterator it = inst_stage_map_v2_.begin();
  while (it != inst_stage_map_v2_.end()) {
    if (it->second < 0) {
      it = inst_stage_map_v2_.erase(it);
    } else {
      ++it;
    }
  }

  return (orig_staged_num < inst_stage_map_v2_.size());
}

bool
GraphSketch::BackwardPlan(int physical_stage_num,
                            const std::vector<SketchNode*>& sketch_nodes,
                            const HloInstMap<int>& mirror_stage_map,
                            const std::set<std::pair<int, int>>& group_depend,
                            const HloInstSet& inst_scope) {
  VLOG(2) << "module before BackwardPlan: " << module_->ToString();

  int start_stage = physical_stage_num;
  int end_stage = start_stage + physical_stage_num - 1;

  VLOG(2) << "[backward] initial stage before MapUnstagedInsts:";
  HloInstMap<std::pair<int/*asap*/, int/*alap*/>> dummy_range_map;
  std::string stage_info = DumpStages(dummy_range_map, nullptr);
  VLOG(2) << stage_info;

  MapUnstagedInsts(start_stage, inst_scope);

  VLOG(2) << "[backward] stage after MapUnstagedInsts:";
  stage_info = DumpStages(dummy_range_map, nullptr);
  VLOG(2) << stage_info;

  //AssignStageSafely(inst_scope);
  bool changed;
  do {
    changed = PlaceInstsToNeighbor(inst_scope, start_stage, end_stage+1);
    if (changed) {
      changed = MapUnstagedInsts(start_stage, inst_scope);
    }
  } while (changed);

  /*
  while (changed) {
    changed = AssignStageSafely(inst_scope);
    if (changed) {
      changed = PlaceInstsToNeighbor(start_stage, end_stage+1);
    } else {
      changed = false;
    }
  }
  */

  // debug:
  auto dump_unstaged_insts = [physical_stage_num, this] (const std::string& file_name,
                                 const HloInstMap<int>& inst_stage_map) {
    HloInstSet unstaged_inst_scope;
    for (const HloInstruction* inst : computation_->instructions()) {
      if (inst_stage_map.find(inst) == inst_stage_map.end()) {
        unstaged_inst_scope.insert(inst);
      }
    }

    HloInstSet bound_insts;
    for (const HloInstruction* inst : unstaged_inst_scope) {
      for (auto* user : inst->users()) {
        bound_insts.insert(user);
      }
      for (auto* op : inst->operands()) {
        bound_insts.insert(op);
      }
    }

    for (auto* inst : bound_insts) {
      unstaged_inst_scope.insert(inst);
    }

    std::unique_ptr<GraphSketch> unstaged_graph =
      BuildFineGrainedSketch(module_, physical_stage_num, &unstaged_inst_scope);
    std::ofstream of(file_name, std::ios::ate);
    of << unstaged_graph->Dump(&inst_stage_map) << std::endl;
    of.close();
  };

  dump_unstaged_insts("unstaged_inst0.dot", inst_stage_map_v2_);
  // end debug

  VLOG(2) << "[backward] initial stage before infering stage range:";
  stage_info = DumpStages(dummy_range_map, nullptr);
  VLOG(2) << stage_info;

  HloInstMap<std::pair<int/*asap*/, int/*alap*/>>
  inst_stage_range_map = InferStageRangeByContext(inst_scope,
                                                  physical_stage_num,
                                                  start_stage);

  VLOG(2) << "[backward] initial stage after infer stage range:";
  stage_info = DumpStages(inst_stage_range_map, nullptr);
  VLOG(2) << stage_info;

  for (const HloInstruction* inst : computation_->instructions()) {
    if (inst_stage_map_v2_.find(inst) == inst_stage_map_v2_.end() &&
        inst_stage_range_map.find(inst) == inst_stage_range_map.end()) {
      VLOG(2) << "missed range setting: " << inst->name();
      inst_stage_range_map[inst] = std::make_pair(start_stage, end_stage);
    }
  }
  
  // debug:
  dump_unstaged_insts("unstaged_inst1.dot", inst_stage_map_v2_);

  VLOG(2) << "[backward] initial stage of pipeline:";
  stage_info = DumpStages(inst_stage_range_map, nullptr);
  VLOG(2) << stage_info;

  IlpStageModel ilp_mod;
  ilp_mod.set_logic_stage_start(start_stage);
  ilp_mod.set_logic_stage_end(end_stage);
  ilp_mod.set_per_stage_flops(backward_per_stage_flops_);
  ilp_mod.set_physical_stage_num(physical_stage_num_);
  ilp_mod.set_unbalanced_ratio(unbalanced_ratio_);

  bool succ = BuildIlpStageModel(inst_scope, start_stage, end_stage,
                                 inst_stage_map_v2_, inst_stage_range_map,
                                 mirror_stage_map, backward_op_groups_,
                                 group_depend, false, ilp_mod);

  if (!succ) {
    return false;
  }

  std::string ilp_mod_str = ilp_mod.ExportToString();
  VLOG(2) << "pipeline ilp model(backward):";
  VLOG(2) << ilp_mod_str;

  std::unordered_map<const HloInstruction*, int/*stage id*/> solver_res;
  succ = ilp_mod.Solve(solver_res);

  if (!succ) {
    return false;
  }

  bool debug = ServiceEnv::debug();
  if (debug) {
    VLOG(0) << "[backward] inst_stage_range_map size: " << inst_stage_range_map.size();
    VLOG(0) << "[backward] ILP solver result for pipeline stages:";
    VLOG(0) << "[backward] instruction number: " << solver_res.size();
  }
  std::string ilp_solver_res_str="instruction : stage";
  for (auto& inst_stage : solver_res) {
    ilp_solver_res_str += "\n  " + inst_stage.first->name() + " : "
                          + std::to_string(inst_stage.second);
    if (debug) {
      VLOG(0) << "update stage for inst: " << inst_stage.first->name()
              << ", stage: " << inst_stage.second;
    }
    CHECK(inst_stage_map_v2_.find(inst_stage.first) == inst_stage_map_v2_.end())
                      << "inst: " << inst_stage.first->ToString() << "\nstage: "
                      << inst_stage.second;
    inst_stage_map_v2_[inst_stage.first] = inst_stage.second;
    inst_stage_range_map.erase(inst_stage.first);
  }

  if (debug) {
    VLOG(0) << ilp_solver_res_str;
    VLOG(0) << "[backward] inst_stage_range_map size: " << inst_stage_range_map.size();

    VLOG(0) << "[backward] final stage of pipeline:";
    stage_info = DumpStages(inst_stage_range_map, nullptr);
    VLOG(0) << stage_info;
  }

  return true;
}

const HloInstruction*
DeterminedByUser(
      const HloInstruction* inst,
      const HloInstSet& inst_scope,
      std::unordered_map<const HloInstruction*, const HloInstruction*>& determined_by_map) {
  const HloInstruction* domination_user = nullptr;
  int local_user_count = 0;
  for (auto* user : inst->users()) {
    if (inst_scope.find(user) != inst_scope.end()) {
      ++local_user_count;
    }
    if (local_user_count > 1) {
      return nullptr;
    } else {
      domination_user = user;
    }
  }

  //std::vector<const HloInstruction*> ops;
  for (auto* op : inst->operands()) {
    if (inst_scope.find(op) == inst_scope.end()) {
      continue;
    }

    if (determined_by_map.find(op) == determined_by_map.end()) {
      return nullptr;
    }

    //ops.emplace_back(op);
  }

  /*
  for (auto* op : ops) {
    determined_by_map[op] = inst;
  }
  */

  if (!domination_user) {
    return nullptr;
  }

  determined_by_map[inst] = domination_user;
  VLOG(2) << "inst: " << inst->name() << " is dominated by user: "
          << domination_user->name();

  return domination_user;  // return inst's domination user
}

const HloInstruction*
DeterminedByOp(
      const HloInstruction* inst,
      const HloInstSet& inst_scope,
      std::unordered_map<const HloInstruction*, const HloInstruction*>& determined_by_map) {
  const HloInstruction* domination_op = nullptr;
  int local_op_count = 0;
  for (auto* op : inst->operands()) {
    if (inst_scope.find(op) != inst_scope.end()) {
      ++local_op_count;
    }
    if (local_op_count > 1) {
      return nullptr;
    } else {
      domination_op = op;
    }
  }

  //std::vector<const HloInstruction*> users;
  for (auto* user : inst->users()) {
    if (inst_scope.find(user) == inst_scope.end()) {
      continue;
    }

    if (determined_by_map.find(user) == determined_by_map.end()) {
      return nullptr;
    }

    //users.emplace_back(user);
  }

  /*
  for (auto* user : users) {
    determined_by_map[user] = inst;
  }
  */

  if (!domination_op) {
    return nullptr;
  }

  determined_by_map[inst] = domination_op;
  VLOG(2) << "inst: " << inst->name() << " is dominated by op: "
          << domination_op->name();
  return domination_op;  // return inst's domination op
}

void
GraphSketch::BuildDeterminedByMap(
      const HloInstSet& inst_scope,
      std::unordered_map<const HloInstruction*, const HloInstruction*>& determined_by_user,
      std::unordered_map<const HloInstruction*, const HloInstruction*>& determined_by_op) {
  //
  std::deque<const HloInstruction*> user_ready_insts;
  for (auto* inst : computation_->instructions()) {
    if (inst_scope.find(inst) == inst_scope.end()) {
      continue;
    }

    if (inst_stage_map_v2_.find(inst) != inst_stage_map_v2_.end()) {
      continue;
    }

    if (inst->operand_count() == 0) {
      user_ready_insts.push_back(inst);
    }
  }

  while (!user_ready_insts.empty()) {
    auto* inst = user_ready_insts.front();
    user_ready_insts.pop_front();

    const HloInstruction* domination_user =
                      DeterminedByUser(inst, inst_scope, determined_by_user);
    if (domination_user) {
      if (inst_stage_map_v2_.find(domination_user) == inst_stage_map_v2_.end()) {
        user_ready_insts.push_back(domination_user);
      }
    }
  }

  //
  const HloInstruction* root = computation_->root_instruction();
  std::deque<const HloInstruction*> op_ready_insts;
  for (auto* inst : computation_->instructions()) {
    if (inst_scope.find(inst) == inst_scope.end()) {
      continue;
    }

    if (inst_stage_map_v2_.find(inst) != inst_stage_map_v2_.end()) {
      continue;
    }

    if (inst->user_count() == 1) {
      const std::vector<HloInstruction*>& users = inst->users();
      HloInstruction* user = users[0];
      if (user == root) {
        op_ready_insts.push_back(inst);
      }
    }
  }

  while (!op_ready_insts.empty()) {
    auto* inst = op_ready_insts.front();
    op_ready_insts.pop_front();

    const HloInstruction* domination_op =
                      DeterminedByOp(inst, inst_scope, determined_by_op);
    if (domination_op) {
      if (inst_stage_map_v2_.find(domination_op) == inst_stage_map_v2_.end()) {
        op_ready_insts.push_back(domination_op);
      }
    }
  }
}

bool
GraphSketch::BuildIlpStageModel(
    const HloInstSet& inst_scope,
    int start_stage, int end_stage,
    const HloInstMap<int>& inst_stage_map,
    const HloInstMap<std::pair<int/*asap*/, int/*alap*/>>& inst_stage_range_map,
    const HloInstMap<int>& mirror_stage_map,
    const std::map<int, std::shared_ptr<OpGroupInfo>>& graph_groups,
    const std::set<std::pair<int, int>>& group_depend,
    bool is_forward,
    IlpStageModel& ilp_model) {
  std::unordered_map<const HloInstruction*, const HloInstruction*> determined_by_user;
  std::unordered_map<const HloInstruction*, const HloInstruction*> determined_by_op;
  BuildDeterminedByMap(inst_scope, determined_by_user, determined_by_op);
  // 1. build asap/alap constraints
  // 1.1. build asap and alap constraints
  //       a). stage_var >= asap (constant)
  //       b). stage_var <= alap (constant)
  VLOG(0) << "build asap/alap constraints for each boundary instruction";
  for (auto& inst_stage_rng : inst_stage_range_map) {
    std::shared_ptr<IlpStageVarExpr> var_expr =
                            ilp_model.GetOrCreateStageVar(inst_stage_rng.first);
    std::shared_ptr<IlpStageSumExpr> cond_left = std::make_shared<IlpStageSumExpr>();
    cond_left->operands_.emplace_back(var_expr);
    int asap_stage_id = inst_stage_rng.second.first;
    int alap_stage_id = inst_stage_rng.second.second;
    // stage_var >= asap (constant)
    std::shared_ptr<IlpStageGEIntExpr> ge_expr =
                  std::make_shared<IlpStageGEIntExpr>(cond_left, asap_stage_id);
    ilp_model.AddConstraint(ge_expr);

    // stage_var <= alap (constant)
    std::shared_ptr<IlpStageLEIntExpr> le_expr =
                  std::make_shared<IlpStageLEIntExpr>(cond_left, alap_stage_id);
    ilp_model.AddConstraint(le_expr);
  }

  VLOG(2) << "ilp model after adding asap and alap constraints:";
  VLOG(2) << ilp_model.ExportToString();

  // 2. build ">=" and "<=" constraints between producer and user stages
  // 2.1. find candidate boundary instructions(user side) across stages
  VLOG(0) << "build constraints between producer and user stages";
  HloInstSet unstage_users;
  for (auto& inst_stage_rng : inst_stage_range_map) {
    const HloInstruction* inst = inst_stage_rng.first;
    unstage_users.insert(inst);
  }

  for (auto& inst_stage : inst_stage_map) {
    const HloInstruction* inst = inst_stage.first;
    if (inst->user_count() == 0) {
      VLOG(2) << "skip root: " << inst->name();
      continue;
    }
    for (auto* op : inst->operands()) {
      if (inst_stage_range_map.find(op) != inst_stage_range_map.end()) {
        unstage_users.insert(inst);
        break;
      }
    }
  }

  // 2.2. build topology constraints
  for (auto* inst : unstage_users) {
    VLOG(2) << "boundary inst: " << inst->ToString();
    std::shared_ptr<IlpStagePrimExpr> inst_expr;
    int inst_stage = -1;
    if (inst_stage_range_map.find(inst) != inst_stage_range_map.end()) {
      // inst's stage is not determined
      // get or create a stage variable for stage(inst)
      inst_expr = ilp_model.GetOrCreateStageVar(inst);
    } else {
      // inst's stage is determined, but at least one of its operand is not determined
      inst_stage = inst_stage_map.at(inst);
      if (is_compute_intensive(inst)) {
        VLOG(2) << "[boundary user: compute] inst: " << inst->name() << ", inst stage: " << inst_stage;
      }
    }

    for (auto* op : inst->operands()) {
      VLOG(2) << "  op: " << op->ToString();
      if (inst_stage>=0) { // inst's stage is determined
        if (inst_stage_map.find(op) != inst_stage_map.end()) {
          // both inst's stage and current op's stage are determined
          continue;
        }
      }

      std::shared_ptr<IlpStagePrimExpr> op_expr;
      int op_stage = -1;
      if (inst_stage_range_map.find(op) != inst_stage_range_map.end()) {
        // op's stage is not determined
        VLOG(2) << "  op stage not determined";
        op_expr = ilp_model.GetOrCreateStageVar(op);
      } else {
        // op's stage is determined
        VLOG(2) << "  op stage determined";
        CHECK(inst_stage_map.find(op) != inst_stage_map.end());
        op_stage = inst_stage_map.at(op);
        if (is_compute_intensive(op)) {
          VLOG(2) << "[boundary op: compute] op: " << op->name() << ", op stage: " << op_stage;
        }
      }

      bool affine = false;
      if (determined_by_user.find(op) != determined_by_user.end()) {
        if (determined_by_user[op] == inst) {
          affine = true;
        }
      }

      if (!affine) {
        if (determined_by_op.find(inst) != determined_by_op.end()) {
          if (determined_by_op[inst] == op) {
            affine = true;
          }
        }
      }

      if (affine) {
        VLOG(2) << "op and inst are affine, op: " << op->name() << ", inst: "
                << inst->name();
      }
      ilp_model.BuildOpUserConstraint(op_stage, inst_stage, op_expr, inst_expr, affine);
    }
  }

  // 2.3. build across cost and its constraint
  VLOG(0) << "build across cost and its constraint";
  CHECK(end_stage >= start_stage);
  for (auto* inst : unstage_users) {
    VLOG(2) << "user of unstaged inst: " << inst->ToString();
    std::shared_ptr<IlpStagePrimExpr> inst_expr;
    int inst_stage = -1;
    if (inst_stage_range_map.find(inst) != inst_stage_range_map.end()) {
      // inst's stage is not determined
      // get or create a stage variable for stage(inst)
      inst_expr = ilp_model.GetOrCreateStageVar(inst);
    } else {
      // inst's stage is determined, but at least one of its operand is not determined
      CHECK(inst_stage_map.find(inst) != inst_stage_map.end());
      inst_stage = inst_stage_map.at(inst);
    }

    VLOG(2) << "build cost and constraint for op-user pair";
    for (int i = 0; i < inst->operand_count(); ++i) {
      const HloInstruction* op = inst->operand(i);
      if (inst_stage>=0) {
        if (inst_stage_map.find(op) != inst_stage_map.end()) {
          // both inst's stage and current op's stage are determined
          continue;
        }
      }

      VLOG(2) << "inst or op stage is undetermined";
      std::shared_ptr<IlpStagePrimExpr> op_expr;
      int op_stage = -1;
      if (inst_stage_range_map.find(op) != inst_stage_range_map.end()) {
        // inst's stage is not determined and op's stage is not determined
        // get or create a variable for stage(op)
        op_expr = ilp_model.GetOrCreateStageVar(op);
      } else {
        // inst's stage is not determined and op's stage is determined
        CHECK(inst_stage_map.find(op) != inst_stage_map.end());
        op_stage = inst_stage_map.at(op);
      }

      // 2.3.1. update optimization object
      ilp_model.BuildOpUserCost(i, op, inst, op_stage, inst_stage, op_expr,
                                inst_expr);
    }
  }

  /*
  // trick: gradients calculate as soon as possible
  std::shared_ptr<IlpStageSumExpr> opt_obj = ilp_model.GetOrCreateOptObj();
  const std::vector<std::shared_ptr<IlpVarExpr>>& all_vars = ilp_model.all_vars();
  for (auto& var : all_vars) {
    if (!var->is_stage_var()) {
      continue;
    }
    opt_obj->operands_.emplace_back(var);
  }
  */

  // 3. build group dependency constraint
  // 3.1. instruction stage is equal to its group's stage
  for (auto& inst_stage_rng : inst_stage_range_map) {
    int group_id = inst_stage_rng.first->metadata().op_group();
    if (group_id <= 0) {
      continue;
    }

    std::shared_ptr<IlpStageVarExpr> var_expr =
                            ilp_model.GetOrCreateStageVar(inst_stage_rng.first);
    std::shared_ptr<IlpStageSumExpr> cond_left = std::make_shared<IlpStageSumExpr>();
    cond_left->operands_.emplace_back(var_expr);
    std::shared_ptr<IlpStageEqIntExpr> eq_expr;
    if (group_stage_map_.find(group_id) != group_stage_map_.end()) {
      int group_stage = group_stage_map_.at(group_id);
      // stage_var = group_stage (constant)
      eq_expr = std::make_shared<IlpStageEqIntExpr>(cond_left, group_stage);
      ilp_model.AddConstraint(eq_expr);
    } else {
      std::shared_ptr<IlpGroupStageVarExpr> group_var =
                                      ilp_model.GetOrCreateStageVar(group_id);
      std::shared_ptr<IlpScaledExpr> minus_group =
                                std::make_shared<IlpScaledExpr>(group_var, -1);
      cond_left->operands_.emplace_back(minus_group);
      eq_expr = std::make_shared<IlpStageEqIntExpr>(cond_left, 0);
      ilp_model.AddConstraint(eq_expr);
    }
  }

  
  // 3.2. build dependency constraints between groups
  for (auto& depend : group_depend) {
    int src_stage = -1;
    int tgt_stage = -1;
    std::shared_ptr<IlpGroupStageVarExpr> src_g_var, tgt_g_var;
    if (group_stage_map_.find(depend.first) != group_stage_map_.end()) {
      src_stage = group_stage_map_.at(depend.first);
      if (!is_forward) {
        src_stage = logic_stage_num_ - src_stage - 1;
      }
    } else {
      src_g_var = ilp_model.GetOrCreateStageVar(depend.first);
    }

    if (group_stage_map_.find(depend.second) != group_stage_map_.end()) {
      tgt_stage = group_stage_map_.at(depend.second);
      if (!is_forward) {
        tgt_stage = logic_stage_num_ - tgt_stage - 1;
      }
    } else {
      tgt_g_var = ilp_model.GetOrCreateStageVar(depend.second);
    }

    if (src_g_var && tgt_g_var) {
      std::shared_ptr<IlpStageSumExpr> cond_left =
                                          std::make_shared<IlpStageSumExpr>();
      cond_left->operands_.emplace_back(src_g_var);
      std::shared_ptr<IlpScaledExpr> minus_tgt =
                                std::make_shared<IlpScaledExpr>(tgt_g_var, -1);
      cond_left->operands_.emplace_back(minus_tgt);
      std::shared_ptr<IlpStageLEIntExpr> le_expr =
                            std::make_shared<IlpStageLEIntExpr>(cond_left, 0);
      ilp_model.AddConstraint(le_expr);
    } else if (src_g_var) {
      CHECK(tgt_stage >= 0);
      std::shared_ptr<IlpStageSumExpr> cond_left =
                                          std::make_shared<IlpStageSumExpr>();
      cond_left->operands_.emplace_back(src_g_var);
      std::shared_ptr<IlpStageLEIntExpr> le_expr =
                    std::make_shared<IlpStageLEIntExpr>(cond_left, tgt_stage);
      ilp_model.AddConstraint(le_expr);
    } else if (tgt_g_var) {
      CHECK(src_stage >= 0);
      std::shared_ptr<IlpStageSumExpr> cond_left =
                                          std::make_shared<IlpStageSumExpr>();
      cond_left->operands_.emplace_back(tgt_g_var);
      std::shared_ptr<IlpStageGEIntExpr> ge_expr =
                    std::make_shared<IlpStageGEIntExpr>(cond_left, src_stage);
      ilp_model.AddConstraint(ge_expr);
    } else {
      CHECK(src_stage <= tgt_stage);
    }
  }

  // 4. add constraint for each stage flops
  bool succ = true;
  if (is_forward) {
    succ = ilp_model.BuildFlopsLimit(inst_stage_map_v2_,
                                     inst_stage_range_map,
                                     inst_flops_map_);
  }

  VLOG(0) << "finish stage ILP model building";

  return succ;
}

bool IlpStageModel::BuildFlopsLimit(
      const HloInstMap<int>& inst_stage_map,
      const HloInstMap<std::pair<int/*asap*/, int/*alap*/>>& inst_stage_range_map,
      const std::unordered_map<const HloInstruction*, int64>& inst_flops_map) {
  // called by only forward planning
  CHECK(per_stage_flops_ > 0);
  CHECK(physical_stage_num_ > 0);
  int64 per_stage_budget = (int64)(per_stage_flops_ * unbalanced_ratio_ / 100);  // init by tolerance
  per_stage_budget += per_stage_flops_;
  // 1. calculate how many flops are allowed in each stage
  std::vector<int64> left_flop_budgets(physical_stage_num_, per_stage_budget);
  for (auto& inst_stage : inst_stage_map) {
    if (inst_flops_map.find(inst_stage.first) != inst_flops_map.end()) {
      CHECK(inst_stage.second < physical_stage_num_);
      VLOG(2) << "stage id: " << inst_stage.second << ", orig budget: "
              << left_flop_budgets[inst_stage.second];
      left_flop_budgets[inst_stage.second] -= inst_flops_map.at(inst_stage.first);
      VLOG(2) << "inst: " << inst_stage.first->name() << ", flops: "
              << inst_flops_map.at(inst_stage.first);
      VLOG(2) << "new budget: " << left_flop_budgets[inst_stage.second];
    }
  }

  // 2. initialize flops-summary expression for each stage
  std::vector<std::shared_ptr<IlpStageSumExpr>> stage_flop_sums(physical_stage_num_);
  for (int stage = 0; stage < physical_stage_num_; ++stage) {
    if (left_flop_budgets[stage] < 0) {
      VLOG(0) << "Can't find a balanced solution even with tolerance "
              << unbalanced_ratio_ << "%."
              << " Try larger tolerance by set UNBALANCED_RATIO with larger value";
      return false;
    }
    CHECK(left_flop_budgets[stage] >= 0) << "stage id: " << stage
        << ", left budget: " << left_flop_budgets[stage];
    stage_flop_sums[stage] = std::make_shared<IlpStageSumExpr>();
  }

  // 3. build flops-summary for each stage
  for (auto& inst_stage_rng : inst_stage_range_map) {
    int asap_stage_id = inst_stage_rng.second.first;
    int alap_stage_id = inst_stage_rng.second.second;

    if (inst_flops_map.find(inst_stage_rng.first) == inst_flops_map.end() ||
        inst_flops_map.at(inst_stage_rng.first) == 0) {
      continue;
    }

    CHECK(asap_stage_id < alap_stage_id);
    CHECK(asap_stage_id >= 0 && alap_stage_id < physical_stage_num_) << "inst: "
      << inst_stage_rng.first->name() << ", asap_stage_id: " << asap_stage_id
      << ", alap_stage_id: " << alap_stage_id << ", stage num: " << physical_stage_num_;

    int64 inst_flops = inst_flops_map.at(inst_stage_rng.first);

    std::shared_ptr<IlpStageSumExpr> one_hot_sum = std::make_shared<IlpStageSumExpr>();
    std::shared_ptr<IlpStageSumExpr> stage_sum = std::make_shared<IlpStageSumExpr>();
    for (int i=asap_stage_id; i<=alap_stage_id; ++i) {
      // 3.1. build one-hot stage variable
      std::shared_ptr<IlpOneHotStageVarExpr> one_hot_var =
                            this->GetOrCreateStageVar(inst_stage_rng.first, i);
      one_hot_sum->operands_.emplace_back(one_hot_var);

      // 3.2. build one term of a stage variable based on one-hot stage variable
      std::shared_ptr<IlpScaledExpr> st_term =
                              std::make_shared<IlpScaledExpr>(one_hot_var, i);
      stage_sum->operands_.emplace_back(st_term);

      // 3.3. accumulate flops for stage i
      std::shared_ptr<IlpScaledExpr> accum_item =
                      std::make_shared<IlpScaledExpr>(one_hot_var, inst_flops);
      stage_flop_sums[i]->operands_.emplace_back(accum_item);
    }

    // 3.4. satisfy one-hot attribution
    std::shared_ptr<IlpStageEqIntExpr> one_hot_eq =
                            std::make_shared<IlpStageEqIntExpr>(one_hot_sum, 1);
    AddConstraint(one_hot_eq);

    // 3.5. stage varables based on their one-hot stage variables
    std::shared_ptr<IlpStageVarExpr> st_var =
                              this->GetOrCreateStageVar(inst_stage_rng.first);

    // 3.6. build relation expression between two version stage variables
    std::shared_ptr<IlpScaledExpr> minus_stage =
                                  std::make_shared<IlpScaledExpr>(st_var, -1);
    stage_sum->operands_.emplace_back(minus_stage);
    std::shared_ptr<IlpStageEqIntExpr> stage_eq =
                            std::make_shared<IlpStageEqIntExpr>(stage_sum, 0);
    AddConstraint(stage_eq);
  }

  return true;
}

void
GraphSketch::CalcPreComputeCost(const std::vector<SketchNode*>& post_order,
                                bool is_forward) {
  // determine pre nodes for each node
  for (auto* node : post_order) {
    if ((is_forward && node->is_backward()) ||
        (!is_forward && !node->is_backward())) {
      continue;
    }
    const std::set<SketchNode*>& inputs = node->inputs();
    std::shared_ptr<AccumCompInfo> max_num_info;
    std::set<SketchNode*> total_accum_comp;

    VLOG(2) << "sketch node: " << node->name();
    for (auto* input : inputs) {
      if ((is_forward && input->is_backward()) ||
          (!is_forward && !input->is_backward())) {
        continue;
      }
      int input_comp_num = 0;
      if (input->pre_accum_comp_info()) {
        input_comp_num = input->pre_accum_comp_info()->accum_comp_nodes_.size();
      }
      if (!max_num_info) {
        max_num_info = input->pre_accum_comp_info();
      } else if (max_num_info->accum_comp_nodes_.size() < input_comp_num) {
        max_num_info = input->pre_accum_comp_info();
      }

      // add a direct pre node
      if (!input->is_tiny()) {
        total_accum_comp.insert(input);
      }

      // add indirect pre nodes
      std::shared_ptr<AccumCompInfo> pre_pre_accum_comp_info =
                                                input->pre_accum_comp_info();
      if (pre_pre_accum_comp_info) {
        const std::set<SketchNode*>& pre_pre_accum_comp_nodes =
                                  pre_pre_accum_comp_info->accum_comp_nodes_;
        total_accum_comp.insert(pre_pre_accum_comp_nodes.begin(),
                                pre_pre_accum_comp_nodes.end());
      }
    }

    int max_comp_num = 0;
    if (max_num_info) {
      max_comp_num = max_num_info->accum_comp_nodes_.size();
    }
    if (max_comp_num < total_accum_comp.size()) {
      // create a new AccumCompInfo
      VLOG(2) << "build AccumCompInfo";
      node->set_pre_accum_comp_info(std::make_shared<AccumCompInfo>());
      node->pre_accum_comp_info()->accum_comp_nodes_ = std::move(total_accum_comp);
      node->pre_accum_comp_info()->EvalAccumFlopCount();
      node->pre_accum_comp_info()->CalcAccumRatio(total_flop_count_);
      VLOG(2) << "AccumCompInfo done";
    } else {
      node->set_pre_accum_comp_info(max_num_info); // reuse
    }

    if (node->pre_accum_comp_info()) {
      VLOG(2) << "node: " << node->name() << ", pre accum_flop_count_: " << node->pre_accum_comp_info()->accum_flop_count_;
    }
  }
}

void
GraphSketch::CalcPostComputeCost(const std::vector<SketchNode*>& post_order,
                                 bool is_forward) {
  // determine post nodes for each node
  for (auto rit = post_order.rbegin(); rit != post_order.rend(); ++rit) {
    auto* node = *rit;
    if ((is_forward && node->is_backward()) ||
        (!is_forward && !node->is_backward())) {
      continue;
    }
    const std::set<SketchNode*>& users = node->users();
    std::shared_ptr<AccumCompInfo> max_num_info;
    std::set<SketchNode*> total_accum_comp;

    for (auto* user : users) {
      if ((is_forward && user->is_backward()) ||
          (!is_forward && !user->is_backward())) {
        continue;
      }
      int user_comp_num = 0;
      if (user->post_accum_comp_info()) {
        user_comp_num = user->post_accum_comp_info()->accum_comp_nodes_.size();
      }
      if (!max_num_info) {
        max_num_info = user->post_accum_comp_info();
      } else if (max_num_info->accum_comp_nodes_.size() < user_comp_num) {
        max_num_info = user->post_accum_comp_info();
      }

      // add a direct post node
      if (!user->is_tiny()) {
        total_accum_comp.insert(user);
      }

      // add indirect post nodes
      std::shared_ptr<AccumCompInfo> post_post_accum_comp_info =
                                                user->post_accum_comp_info();
      if (post_post_accum_comp_info) {
        const std::set<SketchNode*>& post_post_accum_comp_nodes =
                                  post_post_accum_comp_info->accum_comp_nodes_;
        total_accum_comp.insert(post_post_accum_comp_nodes.begin(),
                                post_post_accum_comp_nodes.end());
      }
    }

    int max_comp_num = 0;
    if (max_num_info) {
      max_comp_num = max_num_info->accum_comp_nodes_.size();
    }
    if (max_comp_num < total_accum_comp.size()) {
      // create a new AccumCompInfo
      VLOG(2) << "build AccumCompInfo";
      node->set_post_accum_comp_info(std::make_shared<AccumCompInfo>());
      node->post_accum_comp_info()->accum_comp_nodes_ = std::move(total_accum_comp);
      node->post_accum_comp_info()->EvalAccumFlopCount();
      node->post_accum_comp_info()->CalcAccumRatio(total_flop_count_);
      VLOG(2) << "AccumCompInfo done";
    } else {
      node->set_post_accum_comp_info(max_num_info); // reuse
    }

    if (node->post_accum_comp_info()) {
      VLOG(2) << "node: " << node->name() << ", post accum_flop_count_: " << node->post_accum_comp_info()->accum_flop_count_;
    }
  }
}

bool GraphSketch::DetectCycle() {
  for (HloInstruction* inst : computation_->instructions()) {
    if (inst == computation_->root_instruction()) continue;
    if (!inst_stage_map_v2_.count(inst)) continue;
    int inst_stage_id = inst_stage_map_v2_[inst];
    for (const HloInstruction* op : inst->operands()) {
      if (!inst_stage_map_v2_.count(inst)) continue;
      int op_stage_id = inst_stage_map_v2_[op];
      if (op_stage_id > inst_stage_id) {
        VLOG(0) << "[CYCLE] inst : " << inst->name() << ", stage_id = " << inst_stage_id
                << ", op : " << op->name() << ", stage_id = " << op_stage_id;
        return false;
      }
    }
  }

  return true;
}

void GraphSketch::RecordStagePlanIntoInsts() {
  for (HloInstruction* inst : computation_->instructions()) {
    if (inst_stage_map_v2_.find(inst) == inst_stage_map_v2_.end()) continue;
    int inst_stage_id = inst_stage_map_v2_[inst];
    inst->mutable_dist_spec()->set_stage(inst_stage_id);
  }
}

void
GraphSketch::ForwardTinyNodePlan(std::vector<SketchStage>& stages) {
  bool changed;
  do {
    changed = false;
    for (auto& node : nodes_) {
      if (node->stage_id()>=0) {
        continue;
      }

      if (node->is_backward()) {
        continue;
      }

      const std::set<SketchNode*>& pre_nodes = node->pre_nodes();
      int pre_max_stage = -1;
      for (auto* pre_node : pre_nodes) {
        if (pre_node->stage_id() < 0) {
          continue;
        }
        if (pre_node->stage_id() > pre_max_stage) {
          pre_max_stage = pre_node->stage_id();
        }
      }

      const std::set<SketchNode*>& post_nodes = node->post_nodes();
      int post_min_stage = INT_MAX;
      for (auto* post_node : post_nodes) {
        if (post_node->stage_id() < 0) {
          continue;
        }
        if (post_node->stage_id() < post_min_stage) {
          post_min_stage = post_node->stage_id();
        }
      }

      VLOG(2) << "tiny node: " << node->name() << ", pre_max_stage: "
              << pre_max_stage << ", post_min_stage: " << post_min_stage;
      CHECK(pre_max_stage <= post_min_stage);

      if (pre_max_stage == post_min_stage) {
        VLOG(2) << "tiny node: " << node->name() << ", stage: " << pre_max_stage;
        CHECK(pre_max_stage>=0 && pre_max_stage != INT_MAX);
        stages[pre_max_stage].AddNode(node.get());
        changed = true;
      } else if (pre_max_stage == -1 && post_min_stage != INT_MAX) {
        VLOG(2) << "tiny node: " << node->name() << ", stage: " << post_min_stage;
        CHECK(post_min_stage>=0 && post_min_stage != INT_MAX);
        stages[post_min_stage].AddNode(node.get());
        changed = true;
      } else if (pre_max_stage >= 0 && post_min_stage == INT_MAX) {
        VLOG(2) << "tiny node: " << node->name() << ", stage: " << pre_max_stage;
        CHECK(pre_max_stage>=0 && pre_max_stage != INT_MAX);
        stages[pre_max_stage].AddNode(node.get());
        changed = true;
      }
    }
  } while (changed);
}

void
GraphSketch::AddNode(std::unique_ptr<SketchNode> node) {
  node->EvalCost(inst_flops_map_);
  if (!node->is_tiny()) {
    comp_sens_nodes_.emplace_back(node.get());
  }
  nodes_.emplace_back(std::move(node));
}

void
GraphSketch::FindNearestStages(
              const HloInstSet& inst_scope,
              const HloInstMap<int>& inst_stage_map,
              std::unordered_map<const HloInstruction*, int>& pre_stage_map,
              std::unordered_map<const HloInstruction*, int>& succ_stage_map) {
  if (inst_stage_map.empty()) {
    return;
  }

  pre_stage_map.clear();
  succ_stage_map.clear();

  std::vector<HloInstruction*> post_order = computation_->MakeInstructionPostOrder();

  // find nearest pre stage id
  for (auto* inst : post_order) {
    if (inst_scope.find(inst) == inst_scope.end()) {
      continue;
    }

    if (inst_stage_map.find(inst) != inst_stage_map.end()) {
      continue;
    }

    int pre_stage_id = -1;
    for (HloInstruction* operand : inst->operands()) {
      if (inst_scope.find(operand) == inst_scope.end()) {
        continue;
      }

      auto op_stage_it = inst_stage_map.find(operand);
      if (op_stage_it != inst_stage_map.end()) {
        if (op_stage_it->second > pre_stage_id) {
          pre_stage_id = op_stage_it->second;
        }
      } else {
        auto op_pre_stage_it = pre_stage_map.find(operand);
        VLOG(2) << "inst: " << inst->name() << ", operand name: " << operand->name();
        CHECK(op_pre_stage_it != pre_stage_map.end());
        if (op_pre_stage_it->second > pre_stage_id) {
          pre_stage_id = op_pre_stage_it->second;
        }
      }
    }

    pre_stage_map[inst] = pre_stage_id;
  }

  // find nearest successive stage id
  for (auto rit = post_order.rbegin(); rit != post_order.rend(); ++rit) {
    auto* inst = *rit;
    if (inst_scope.find(inst) == inst_scope.end()) {
      continue;
    }

    if (inst_stage_map.find(inst) != inst_stage_map.end()) {
      continue;
    }

    int succ_stage_id = INT_MAX;
    HloInstruction* succ_stage_inst = nullptr;
    for (HloInstruction* user : inst->users()) {
      if (inst_scope.find(user) == inst_scope.end()) {
        continue;
      }

      auto user_stage_it = inst_stage_map.find(user);
      if (user_stage_it != inst_stage_map.end()) {
        if (user_stage_it->second < succ_stage_id) {
          succ_stage_id = user_stage_it->second;
          succ_stage_inst = user;
        }
      } else {
        auto user_succ_stage_it = succ_stage_map.find(user);
        CHECK(user_succ_stage_it != succ_stage_map.end());
        if (user_succ_stage_it->second < succ_stage_id) {
          VLOG(2) << "inst: " << inst->name() << ", user name: " << user->name();
          succ_stage_id = user_succ_stage_it->second;
          succ_stage_inst = user;
        }
      }
    }

    VLOG(2) << "inst: " << inst->name() << ", succ stage id: " << succ_stage_id;
    if (succ_stage_inst) {
      VLOG(2) << "succ_stage_inst: " << succ_stage_inst->name();
    }
    succ_stage_map[inst] = succ_stage_id;
  }
}

bool
GraphSketch::PlaceInstsToNeighbor(const HloInstSet& inst_scope,
                                    int start_stage, int logic_stage_num) {
  // both forward and backward call it
  // only place instructions which stages are deterministic
  // find the pattern with:
  // instruction dependency:   inst i -> inst j -> inst k
  // logic stage:                 m      unstaged     m
  // for this pattern, we place instruction j to stage m
  bool ever_changed = false;

  bool changed = true;
  while (changed) {
    changed = false;
    std::unordered_map<const HloInstruction*, int> pre_stage_map;
    std::unordered_map<const HloInstruction*, int> succ_stage_map;
    FindNearestStages(inst_scope, inst_stage_map_v2_,
                      pre_stage_map, succ_stage_map);

    if (pre_stage_map.empty() && succ_stage_map.empty()) {
      break;
    }

    // debug
    VLOG(2) << "in PlaceInstsToNeighbor";
    for (auto& pre_stage : pre_stage_map) {
      auto succ_it = succ_stage_map.find(pre_stage.first);
      if (succ_it != succ_stage_map.end()) {
        VLOG(2) << pre_stage.first->name() << ": pre stage: " << pre_stage.second
                << ", succ stage: " << succ_it->second;
      } else {
        VLOG(2) << pre_stage.first->name() << ": pre stage: " << pre_stage.second
                << ", succ stage: non-define";
      }
    }

    for (const HloInstruction* inst : computation_->instructions()) {
      if (inst_stage_map_v2_.find(inst) != inst_stage_map_v2_.end() ||
          inst == computation_->root_instruction()) {
        continue;
      }
      VLOG(2) << "inst: " << inst->ToString();
      CHECK(pre_stage_map.find(inst) != pre_stage_map.end());
      CHECK(succ_stage_map.find(inst) != succ_stage_map.end());
      int pre_stage_id = pre_stage_map[inst];
      int succ_stage_id = succ_stage_map[inst];
      int stage_id = -1;
      if (pre_stage_id == -1) {
        if (succ_stage_id == INT_MAX) {
          continue;
        }

        if (succ_stage_id == start_stage) {
          stage_id = succ_stage_id;
        }
      } else {
        if (pre_stage_id == succ_stage_id) {
          stage_id = pre_stage_id;
        } else if (pre_stage_id == logic_stage_num - 1) {
          CHECK(succ_stage_id == INT_MAX || (succ_stage_id == logic_stage_num - 1))
              << "inst: " << inst->name() << ", pre_stage_id: " << pre_stage_id
              << ", succ_stage_id: " << succ_stage_id;
          stage_id = pre_stage_id;
        }
      }

      if (stage_id >= 0) {
        inst_stage_map_v2_[inst] = stage_id;
        changed = true;
        ever_changed = true;
      }
    }
  }

  return ever_changed;
}

bool
GraphSketch::MapUnstagedInsts(int start_stage, const HloInstSet& inst_scope) {
  bool ever_changed = false;
  bool changed;
  int logic_stage_num = physical_stage_num_ << 1;

  std::unordered_map<int/*op group id*/, std::set<int/*physical stage id*/>> group_all_stage_map;
  std::vector<const HloInstruction*> unstaged_insts;
  for (const HloInstruction* inst : computation_->instructions()) {
    HloInstMap<int>::iterator it = inst_stage_map_v2_.find(inst);
    if (it != inst_stage_map_v2_.end()) {
      if (inst->metadata().op_group() > 0) {
        if (it->second < physical_stage_num_) {
          group_stage_map_[inst->metadata().op_group()] = it->second;
          group_all_stage_map[inst->metadata().op_group()].insert(it->second);
          VLOG(2) << "src: " << it->first->name() << ", stage: " << it->second
                  << ", group: " << inst->metadata().op_group();
        } else {
          group_stage_map_[inst->metadata().op_group()] =
                                          logic_stage_num - it->second - 1;
          group_all_stage_map[inst->metadata().op_group()].insert(logic_stage_num - it->second - 1);
          VLOG(2) << "src: " << it->first->name() << ", stage: "
                  << logic_stage_num - it->second - 1
                  << ", group: " << inst->metadata().op_group();
        }
      }
    } else {
      if (inst_scope.find(inst) != inst_scope.end()) {
        VLOG(2) << "unstaged inst: " << inst->name();
        unstaged_insts.emplace_back(inst);
      }
    }
  }

  for (auto& group_stage : group_all_stage_map) {
    if (group_stage.second.size() > 1) {
      group_stage_map_.erase(group_stage.first);

      // debug
      std::string stages;
      for (auto& stage : group_stage.second) {
        stages += ", " + std::to_string(stage);
      }
      VLOG(2) << "group: " << group_stage.first << ", stage list: " << stages;
    }
  }

  std::map<int/*op group id*/, int/*stage id*/>::iterator group_it;
  for (auto* inst : unstaged_insts) {
    VLOG(2) << "[MapUnstagedInsts] unstage inst: " << inst->name()
            << ", op group: " << inst->metadata().op_group();
    if (inst->metadata().op_group() == 0) {
      continue;
    }

    group_it = group_stage_map_.find(inst->metadata().op_group());
    if (group_it == group_stage_map_.end()) {
      continue;
    }
    int stage;
    CHECK(group_it->second < physical_stage_num_);
    if (group_it->second < start_stage) {
      stage = logic_stage_num - group_it->second - 1;
    } else {
      stage = group_it->second;
    }
    inst_stage_map_v2_[inst] = stage;
    ever_changed = true;

    VLOG(2) << "map inst to stage " << stage
            << "\ninst: " << inst->ToString();
  }

  return ever_changed;
}

int64 GraphSketch::ComputeTransfersBetween(SketchNode* src, SketchNode* dst) {
  int64 total_transfers = 0;
  CHECK(dst->inputs().count(src));
  std::unordered_set<HloInstruction*> visited_insts;
  for (auto* dst_instr : dst->instructions()) {
    for (auto* operand : dst_instr->operands()) {
      if (!src->instructions().count(operand) ||
          visited_insts.count(operand)) continue;
      visited_insts.insert(operand);
      total_transfers += ShapeUtil::ByteSizeOf(operand->shape(), 8);
    }
  }
  return total_transfers;
}

std::string
GraphSketch::Dump(const HloInstMap<int>* inst_stage_map) {
  // dump header
  std::string result = std::string("digraph G { \n \
rankdir = TB; \n \
compound = true; \n \
\n \
\n");

  std::unordered_map<std::string, std::string> name_id_map;
  for (int i=0; i<nodes_.size(); ++i) {
    std::unique_ptr<SketchNode>& nd = nodes_[i];
    if (nd->not_connected()) {
      continue;
    }
    std::string nd_dot_name = std::to_string(i);
    name_id_map[nd->name()] = nd_dot_name;

    std::string layer_name;
    const std::string& op_name = nd->core_instr()->metadata().op_name();
    std::string layer_key("layer");
    std::string::size_type layer_key_pos = op_name.find(layer_key);
    if (layer_key_pos != op_name.npos) {
      std::string::size_type layer_key_end = op_name.find("/",
                                            layer_key_pos+layer_key.length());
      if (layer_key_end != op_name.npos) {
        layer_name = op_name.substr(layer_key_pos, layer_key_end-layer_key_pos);
      }
    }

    result += nd_dot_name + " [label=<<b>" + nd->name() + "</b>";

    if (nd->core_instr()->opcode() == HloOpcode::kCustomCall||
        nd->core_instr()->opcode() == HloOpcode::kDot ||
        nd->core_instr()->opcode() == HloOpcode::kConvolution) {
      result += "<br/><b>Shape: " + ShapeUtil::HumanString(nd->core_instr()->shape()) + "</b>";
    }
    bool mark_it = false;
    if (inst_stage_map) {
      VLOG(2) << "core instr: " << nd->core_instr()->name()
              << ", inst_stage_map size: " << inst_stage_map->size();
      HloInstMap<int>::const_iterator stage_it = inst_stage_map->find(nd->core_instr());
      if (stage_it != inst_stage_map->end()) {
        mark_it = true;
        result += "<br/><b>Stage: " + std::to_string(stage_it->second) + "</b>";
      }
    }
    result += ">";  // end of label

    result += ", shape=rect";
    result += ", style=\"filled\", fontcolor=\"black\", color=\"#af8eb5\"";
    if (nd->is_critical_node() || mark_it) {
      result += ", fillcolor=\"red\"";
    } else if (nd->is_tiny()) {
      if (nd->stage_id() < 0) {
        result += ", fillcolor=\"yellow\"";
      } else {
        result += ", fillcolor=\"white\"";
      }
    } else {
      result += ", fillcolor=\"#e1bee7\"";
    }

    result += "];\n";
  }

  result += "\n\n";

  // dump body
  std::string connect;

  for (int i=0; i<nodes_.size(); ++i) {
    auto& nd = nodes_[i];
    if (nd->not_connected()) {
      continue;
    }
    CHECK(name_id_map.find(nd->name())!=name_id_map.end());
    const std::string& tgt_name = name_id_map[nd->name()];
    for (SketchNode* nd_input : nd->inputs()) {
      CHECK(name_id_map.find(nd_input->name())!=name_id_map.end());
      const std::string& src_name = name_id_map[nd_input->name()];
      std::string bytes = HumanReadableNumBytes(ComputeTransfersBetween(nd_input, nd.get()));
#if 1
      connect = src_name + " -> " + tgt_name + " [label=\"" + bytes + "\"];\n";
#else
      connect = src_name + " -> " + tgt_name + ";\n";
#endif
      result += connect;
    }
  }

  // dump footer
  result += "}\n";

  return result;
}

}  // namespace xla

