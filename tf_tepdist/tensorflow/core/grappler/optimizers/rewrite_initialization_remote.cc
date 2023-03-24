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

#include "tensorflow/core/grappler/optimizers/rewrite_initialization_remote.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"

#include <unordered_set>

namespace tensorflow {
namespace grappler {

constexpr char kRngDistConfig[] = "_RNG_DIST_CONFIG";
constexpr char kAssignVariableOp[] = "AssignVariableOp";
constexpr char kNeedFetch[] = "_need_fetch";
constexpr char kAssignOp[] = "Assign";
constexpr char kSaveV2Op[] = "SaveV2";
constexpr char kRestoreV2Op[] = "RestoreV2";
constexpr char kNoOp[] = "NoOp";
constexpr char kConst[] = "Const";
constexpr char kDtype[] = "dtype";
constexpr char kValue[] = "value";
constexpr char kShape[] = "shape";
constexpr char kInitRefCount[] = "_init_ref_count";
constexpr char kImplicitShape[] = "_shape";
constexpr char kRemoteRPC[] = "_remote_rpc";
constexpr char kReWriteConst[] = "ReWriteConst";
constexpr char kRegisterInitializationOp[] = "RegisterInitialization";

string RewriteInitializationRemote::OptimizedNodeName(
    const NodeDef& node, StringPiece suffix) const {
  return AddPrefixToNodeName(strings::StrCat(node.name(), suffix),
                             kReWriteConst);
}

bool RewriteInitializationRemote::VariableNeedFetch(const NodeDef& node) {
  if (HasNodeAttr(node, kNeedFetch)) {
    bool need_fetch = false;;
    GetNodeAttr(node, kNeedFetch, &need_fetch);
    return need_fetch;
  }

  return false;
}

Status RewriteInitializationRemote::Optimize(Cluster* /*cluster*/,
                                     const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  NodeMap node_map(optimized_graph);
  // 1. Record variable's shape into each AssignVariableOp to make
  //    shape available for XLA compilation before initialization.
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    auto* node_def = optimized_graph->mutable_node(i);
    if (node_def->op() == kAssignVariableOp) {
      auto* variable = node_map.GetNode(node_def->input(0));
      TensorShapeProto var_shape;
      TF_RETURN_IF_ERROR(GetNodeAttr(*variable, kShape, &var_shape));
      AddNodeAttr(kImplicitShape, var_shape, node_def);
    }
  }

  // 2. Set remote rpc call flag to each SaveV2 and RestoreV2 to make
  //    save or restore happens on servers.
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    auto* node_def = optimized_graph->mutable_node(i);
    if (node_def->op() == kSaveV2Op || node_def->op() == kRestoreV2Op) {
      AddNodeAttr(kRemoteRPC, true, node_def);
    }
  }

  // 3. Begin to rewrite initialization graph.
  std::unordered_set<string> removeable_nodes;
  std::unordered_set<const NodeDef*> assign_resource_ops, assign_ops;
  bool has_rng_attr = false;
  for (int i = 0; i < item.graph.node_size(); i++) {
    auto& node_def = item.graph.node(i);
    if (node_def.op() == kAssignVariableOp) {
      assign_resource_ops.insert(&node_def);
    } else if (node_def.op() == kAssignOp) {
      assign_ops.insert(&node_def);
    }

    if (HasNodeAttr(node_def, kRngDistConfig)) {
      has_rng_attr = true;
    }
  }

  NodeDef* group_init = nullptr;
  if (assign_resource_ops.size()) {
    for (NodeDef* output : node_map.GetOutputs((*assign_resource_ops.begin())->name())) {
      if (!group_init && output->op() == kNoOp) {
        group_init = output;
        break;
      }
    }
  }
  if (!group_init || !has_rng_attr) return Status::OK();

  bool init_graph = true;
  for (auto& input : group_init->input()) {
    auto* input_node = node_map.GetNode(input);
    if (input_node->op() != kAssignVariableOp &&
        input_node->op() != kAssignOp) {
      init_graph = false;
      break;
    }
  }

  init_graph &= ((assign_resource_ops.size() + assign_ops.size()) ==
                  group_init->input_size());
  
  if (!init_graph) return Status::OK();

  std::unordered_set<NodeDef*> register_nodes;
  std::unordered_set<const NodeDef*> dont_removed_assigned;
  for (auto* node_def : assign_resource_ops) {
    auto* rng_def = node_map.GetNode(node_def->input(1));
    auto* variable = node_map.GetNode(node_def->input(0));
    if (!HasNodeAttr(*rng_def, kRngDistConfig)) continue;

    TensorShapeProto var_shape;
    TF_RETURN_IF_ERROR(GetNodeAttr(*variable, kShape, &var_shape));

    auto device = variable->device();
    std::vector<string> rng_terminals = {rng_def->name()};
    std::vector<const NodeDef*> rng_producers;
    ComputeTransitiveFanin(*optimized_graph, rng_terminals, &rng_producers);
    if (VariableNeedFetch(*variable)) {
      dont_removed_assigned.insert(node_def);
      AddNodeAttr(kInitRefCount, 2, variable);
    } else {
      for (auto producer : rng_producers) {
        removeable_nodes.insert(producer->name());
      }
      removeable_nodes.insert(rng_def->name());
      removeable_nodes.insert(node_def->name());
    }
    // Create Constant initialization information node
    string rng_bytes;
    TF_RETURN_IF_ERROR(GetNodeAttr(*rng_def, kRngDistConfig, &rng_bytes));
    NodeDef* const_node = optimized_graph->add_node();
    const string const_name = OptimizedNodeName(*variable, "const");
    TensorShapeProto shape_proto;
    TensorProto value_proto;
    value_proto.add_string_val(rng_bytes);
    value_proto.set_dtype(DataType::DT_STRING);
    *value_proto.mutable_tensor_shape() = shape_proto;
    const_node->set_name(const_name);
    const_node->set_device(device);
    const_node->set_op(kConst);
    AddNodeAttr(kDtype, DataType::DT_STRING, const_node);
    AddNodeAttr(kValue, value_proto, const_node);

    // Create RegisterInitializationOp node
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(*variable, kDtype, &dtype));
    NodeDef* register_node = optimized_graph->add_node();
    const string register_name = OptimizedNodeName(*node_def, "register");
    register_node->set_name(register_name);
    register_node->set_op(kRegisterInitializationOp);
    register_node->set_device(device);
    *register_node->add_input() = variable->name();
    *register_node->add_input() = const_name;
    AddNodeAttr(kDtype, dtype, register_node);
    AddNodeAttr(kShape, var_shape, register_node);
    register_nodes.insert(register_node);
  }

  if (group_init) {
    node_map.RemoveInputs(group_init->name());
    group_init->clear_input();
    // For rewrite register ops
    for (auto* node_def : register_nodes) {
      *group_init->add_input() = AsControlDependency(*node_def);
    }

    // For common assign ops (e.g. global_step assignment)
    for (auto* node_def : assign_ops) {
      *group_init->add_input() = AsControlDependency(*node_def);
    }

    for (auto * node_def : dont_removed_assigned) {
      *group_init->add_input() = AsControlDependency(*node_def);
    }
  }

  for (auto* node_def : assign_resource_ops) {
    if (dont_removed_assigned.count(node_def)) continue;
    node_map.RemoveNode(node_def->name());
  }
  optimized_graph->mutable_node()->erase(
      std::remove_if(
          optimized_graph->mutable_node()->begin(),
          optimized_graph->mutable_node()->end(),
          [removeable_nodes](const NodeDef& node) {
            return removeable_nodes.find(node.name()) != removeable_nodes.end();
          }), optimized_graph->mutable_node()->end());
  return Status::OK();
}

void RewriteInitializationRemote::Feedback(Cluster* /*cluster*/,
                                   const GrapplerItem& /*item*/,
                                   const GraphDef& /*optimized_graph*/,
                                   double /*result*/) {
  // Nothing to do for RewriteInitializationRemote.
}

bool  RewriteInitializationRemote::UsesFunctionLibrary() const {
  // Nothing to do for RewriteInitializationRemote.
  return false;
}

} // namespace grappler
} // namespace tensorflow