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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REWRITE_INITIALIZATION_REMOTE_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REWRITE_INITIALIZATION_REMOTE_OPTIMIZER_H_

#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

constexpr char kRewriteInitializationOptimizer[] = "RewriteInitializationOptimizer";

class RewriteInitializationRemote : public GraphOptimizer {
 public:
  RewriteInitializationRemote() {}
  ~RewriteInitializationRemote() override {}

  string name() const override { return "rewrite_initialization_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;
  
  bool UsesFunctionLibrary() const override;
  
 private:
  string OptimizedNodeName(const NodeDef& node, StringPiece suffix) const;
  bool VariableNeedFetch(const NodeDef& node);
};

} // end namespace grappler
} // end namespace tensorflow

#endif // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REWRITE_INITIALIZATION_REMOTE_OPTIMIZER_H_
