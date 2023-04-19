/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_MODEL_H_

#include <list>
#include <memory>
#include <string>
// TODO(b/114492873): Move this include into core/platform.
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace model {

// A constant that can be used to enable auto-tuning.
constexpr int64 kAutotune = -1;
constexpr char kParallelism[] = "parallelism";
constexpr char kBufferSize[] = "buffer_size";

enum class AutotuneAlgorithm {
  HILL_CLIMB = 0,
  GRADIENT_DESCENT = 1,
};

// Represents thread-safe state that can be shared between an input pipeline and
// the performance model.
struct SharedState {
 public:
  SharedState(int64 value, std::shared_ptr<mutex> mu,
              std::shared_ptr<condition_variable> cond_var)
      : value(value),
        mu(std::move(mu)),
        cond_var(std::move(cond_var)),
        tunable(value == kAutotune) {}

  double value;
  const std::shared_ptr<mutex> mu;
  const std::shared_ptr<condition_variable> cond_var;
  const bool tunable;
};

// Represents a parameter.
struct Parameter {
  Parameter(const string& name, std::shared_ptr<SharedState> state, double min,
            double max)
      : name(name),
        value(state->value),
        min(min),
        max(max),
        state(std::move(state)) {}

  // Human-readable name of the parameter.
  const string name;

  // Identifies the model value of the parameter. This can be different from
  // the actual value (e.g. during optimization search).
  double value;

  // Identifies the minimum value of the parameter.
  const double min;

  // Identifies the maximum value of the parameter.
  const double max;

  // Shared state of the parameter.
  std::shared_ptr<SharedState> state;
};

std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         double min, double max);

// Abstract representation of a TensorFlow input pipeline node. It collects
// information about inputs to this node, processing time spent executing the
// node logic, number of elements produced by the node, various other
// information (e.g. batch size or execution parallelism).
//
// Developers of tf.data transformations are not expected to interact with
// this class directly. Boiler plate code for creating the abstract
// representation of the input pipeline and collecting common information has
// been added to the implementation of `DatasetBase` and `DatasetBaseIterator`
// respectively.
//
// In addition, `DatasetBaseIterator` provides wrappers that can be used for
// transformation-specific information collection. The `SetMetadata` wrapper
// can be used to pass arbitrary metadata to the modeling framework, while the
// `StartWork` and `StopWork` wrappers should be used to correctly account for
// processing time of multi-threaded transformation that yield the CPU; such
// transformations should invoke `StartWork()` when a transformation thread
// starts executing (e.g. when created or woken up) and `StopWork()` when a
// transformation thread stops executing (e.g. when returning or waiting).
class Node {
 public:
  // Arguments for `Node` constructor.
  struct Args {
    int64 id;
    string name;
    std::shared_ptr<Node> output;
  };

  using Factory = std::function<std::shared_ptr<Node>(Args)>;
  using NodeVector = std::vector<std::shared_ptr<Node>>;
  using NodePairList =
      std::list<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>>;

  explicit Node(Args args)
      : id_(args.id),
        name_(std::move(args.name)),
        autotune_(true),
        buffered_bytes_(0),
        buffered_elements_(0),
        bytes_consumed_(0),
        bytes_produced_(0),
        num_elements_(0),
        processing_time_(0),
        record_metrics_(true),
        metrics_(name_),
        output_(args.output.get()) {}

  virtual ~Node() { FlushMetrics(); }

  // Adds an input.
  void add_input(std::shared_ptr<Node> node) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.push_back(node);
  }

  // Increments the aggregate processing time by the given delta.
  void add_processing_time(int64 delta) TF_LOCKS_EXCLUDED(mu_) {
    processing_time_ += delta;
  }

  // Returns an indication whether autotuning is enabled for this node.
  bool autotune() const TF_LOCKS_EXCLUDED(mu_) {
    return autotune_;
  }

  // Returns the number of bytes stored in this node's buffer.
  int64 buffered_bytes() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_bytes_;
  }

  // Returns the number of elements stored in this node's buffer.
  int64 buffered_elements() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_elements_;
  }

  // Returns the number of bytes consumed by the node.
  int64 bytes_consumed() const TF_LOCKS_EXCLUDED(mu_) {
    return bytes_consumed_;
  }

  // Returns the number of bytes produced by the node.
  int64 bytes_produced() const TF_LOCKS_EXCLUDED(mu_) {
    return bytes_produced_;
  }

  // Indicates whether the node has tunable parameters.
  bool has_tunable_parameters() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    for (const auto& pair : parameters_) {
      if (pair.second->state->tunable) return true;
    }
    return false;
  }

  // Returns the unique node ID.
  int64 id() const TF_LOCKS_EXCLUDED(mu_) { return id_; }

  // Returns the node inputs.
  std::list<std::shared_ptr<Node>> inputs() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return inputs_;
  }

  // Returns a longer node name that is guaranteed to be unique.
  string long_name() const { return strings::StrCat(name_, "(id:", id_, ")"); }

  // Returns the node name.
  const string& name() const { return name_; }

  // Returns the number of elements produced by the node.
  int64 num_elements() const TF_LOCKS_EXCLUDED(mu_) {
    return num_elements_;
  }

  // Returns the node output.
  Node* output() const { return output_; }

  // Returns the aggregate processing time.
  int64 processing_time() const TF_LOCKS_EXCLUDED(mu_) {
    return processing_time_;
  }

  // Records that the node consumed the given number of bytes.
  void record_bytes_consumed(int64 num_bytes) { bytes_consumed_ += num_bytes; }

  // Records that the node produced the given number of bytes.
  void record_bytes_produced(int64 num_bytes) { bytes_produced_ += num_bytes; }

  // Records the change in this node's buffer.
  void record_buffer_event(int64 bytes_delta, int64 elements_delta) {
    buffered_bytes_ += bytes_delta;
    buffered_elements_ += elements_delta;
  }

  // Records that the node produced an element.
  void record_element() TF_LOCKS_EXCLUDED(mu_) {
    num_elements_++;
  }

  // Records that a node thread has started executing.
  void record_start(int64 time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    DCHECK_EQ(work_start_, 0);
    work_start_ = time_nanos;
  }

  // Records that a node thread has stopped executing.
  void record_stop(int64 time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    if (work_start_ != 0) {
      processing_time_ += time_nanos - work_start_;
      work_start_ = 0;
    } else {
      VLOG(1) << "Encountered a stop event without a matching start event.";
    }
  }

  // Removes an input.
  void remove_input(std::shared_ptr<Node> input) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.remove(input);
  }

  // Sets the value that determines whether autotuning is enabled for this node.
  void set_autotune(bool autotune) TF_LOCKS_EXCLUDED(mu_) {
    autotune_.store(autotune);
  }

  // Collects tunable parameters in the subtree rooted in this node.
  void CollectTunableParameters(
      absl::flat_hash_map<string, std::shared_ptr<Parameter>>* parameters) const
      TF_LOCKS_EXCLUDED(mu_);

  // Returns a human-readable representation of this node.
  string DebugString() const TF_LOCKS_EXCLUDED(mu_);

  // Flushes the metrics recorded by this node.
  void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element output time for this node and if `gradient` is not
  // `nullptr`, collects the gradient of the output time w.r.t. tunable
  // parameters of the subtree rooted in this node and the last input time.
  double OutputTime(std::vector<double>* input_times,
                    absl::flat_hash_map<string, double>* gradient) const
      TF_LOCKS_EXCLUDED(mu_);

  // Returns a copy of this node, making a deep copy of its inputs and a
  // shallow copy of its tunable parameters.
  //
  // The purpose for this method is to allow the model optimization logic to
  // operate over immutable state while allowing concurrent model updates.
  std::shared_ptr<Node> Snapshot(std::shared_ptr<Node> output) const
      TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element processing time spent in this node.
  double SelfProcessingTime() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the total number of bytes buffered in all nodes in the subtree for
  // which autotuning is enabled.
  double TotalBufferedBytes() const TF_LOCKS_EXCLUDED(mu_);

  // Collects the total buffer limit of all nodes in the subtree for which
  // autotuning is enabled. This number represents the amount of memory that
  // would be used by the subtree nodes if all of their buffers were full.
  double TotalMaximumBufferedBytes() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element CPU time spent in the subtree rooted in this node.
  // If `processing_times` is not `nullptr`, collects the per-element CPU time
  // spent in each node of the subtree.
  double TotalProcessingTime(
      absl::flat_hash_map<string, double>* processing_times)
      TF_LOCKS_EXCLUDED(mu_);

 protected:
  // Used for (incrementally) recording metrics. The class is thread-safe.
  class Metrics {
   public:
    explicit Metrics(const string& name)
        : bytes_consumed_counter_(metrics::GetTFDataBytesConsumedCounter(name)),
          bytes_produced_counter_(metrics::GetTFDataBytesProducedCounter(name)),
          num_elements_counter_(metrics::GetTFDataElementsCounter(name)),
          recorded_bytes_consumed_(0),
          recorded_bytes_produced_(0),
          recorded_num_elements_(0) {}

    // Expects the total number of bytes consumed and records the delta since
    // last invocation.
    void record_bytes_consumed(int64 total_bytes) {
      int64 delta =
          total_bytes - recorded_bytes_consumed_.exchange(total_bytes);
      bytes_consumed_counter_->IncrementBy(delta);
    }

    // Expects the total number of bytes produced and records the delta since
    // last invocation.
    void record_bytes_produced(int64 total_bytes) {
      int64 delta =
          total_bytes - recorded_bytes_produced_.exchange(total_bytes);
      bytes_produced_counter_->IncrementBy(delta);
    }

    // Expects the total number of elements produced and records the delta since
    // last invocation.
    void record_num_elements(int64 total_elements) {
      int64 delta =
          total_elements - recorded_num_elements_.exchange(total_elements);
      num_elements_counter_->IncrementBy(delta);
    }

   private:
    monitoring::CounterCell* const bytes_consumed_counter_;
    monitoring::CounterCell* const bytes_produced_counter_;
    monitoring::CounterCell* const num_elements_counter_;
    std::atomic<int64> recorded_bytes_consumed_;
    std::atomic<int64> recorded_bytes_produced_;
    std::atomic<int64> recorded_num_elements_;
  };

  // Returns the number of inputs.
  int64 num_inputs() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    int64 num_inputs = 0;
    for (auto& input : inputs_) {
      // Inputs for which autotuning is disabled are excluded.
      if (input->autotune()) {
        ++num_inputs;
      }
    }
    return num_inputs;
  }

  // Creates a clone of this node.
  virtual std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the average size of an element buffered in this node.
  double AverageBufferedElementSize() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the sum of per-element output time for the inputs of this node and
  // if `gradient` is not `nullptr`, collects gradients of output times w.r.t.
  // tunable parameters and the last input time.
  double OutputTimeForInputs(std::vector<double>* input_times,
                             absl::flat_hash_map<string, double>* gradient)
      const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the per-element output time for this node and if `gradient` is not
  // `nullptr`, collects the gradient of the output time w.r.t. tunable
  // parameters of the subtree rooted in this node and the last input time.
  virtual double OutputTimeLocked(std::vector<double>* input_times,
                                  absl::flat_hash_map<string, double>* gradient)
      const TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the sum of per-element processing time for the inputs of this node
  // by adding values for input nodes in `total_processing_times`. Processing
  // time for a given input is a weighted combination of a statistic based on
  // history of input processing time and the actual time. This is done to
  // improve accuracy of processing time estimation for newly created inputs.
  //
  // Uniform distribution of per-element processing times across different
  // inputs is assumed.
  double TotalProcessingTimeForInputs(
      const absl::flat_hash_map<string, double>& total_processing_times)
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the per-element processing time spent in this node.
  double SelfProcessingTimeLocked() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Computes the per-element CPU time spent in the subtree rooted in this node
  // and stores it in `total_processing_times`. If `processing_times` is not
  // `nullptr`, collects the per-element CPU time spent in each node of the
  // subtree.
  virtual void TotalProcessingTimeLocked(
      absl::flat_hash_map<string, double>* processing_times,
      absl::flat_hash_map<string, double>* total_processing_times)
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns a vector of nodes of the subtree rooted in this node.
  // The nodes are in the reverse breadth-first search order.
  NodeVector CollectNodes() const;

  // Collect tunable parameters for the node.
  void CollectTunableParametersHelper(
      absl::flat_hash_map<string, std::shared_ptr<Parameter>>* parameters)
      const;

  // Build up debug string for the node and store in the debug strings map.
  void DebugStringHelper(
      absl::flat_hash_map<string, string>* debug_strings) const;

  // Copy the node and add the (input, copy) pairs to the NodePairList.
  std::shared_ptr<Node> SnapshotHelper(std::shared_ptr<Node> clone_base,
                                       NodePairList* node_pairs) const;

  // Compute total buffered bytes for the node and store in the total bytes map.
  void TotalBufferedBytesHelper(
      absl::flat_hash_map<string, double>* total_bytes) const;

  // Compute total maximum buffered bytes for the node and store in the total
  // bytes map.
  void TotalMaximumBufferedBytesHelper(
      absl::flat_hash_map<string, double>* total_bytes) const;

  // Stores the time passed to the last call to `Node::record_start()` on the
  // current thread.
  //
  // NOTE: This thread-local variable is shared between all instances of `Node`
  // on which the same thread calls `record_start()` or `record_stop()`. It
  // relies on the invariant that at most one `Node` can be "active" on a
  // particular thread at any time. Therefore if `n->record_start()` is called
  // on thread `t`, then `n->record_stop()` must be called before another call
  // to `Node::record_start()` (for any node).
  static thread_local int64 work_start_;  // Will be initialized to zero.

  mutable mutex mu_;
  const int64 id_;
  const string name_;

  // Indicates whether the subtree rooted in this node should be included in
  // autotuning. In particular, if this is `false`, then the subtree is excluded
  // from computation of output time and processing time.
  std::atomic<bool> autotune_;
  std::atomic<int64> buffered_bytes_;
  std::atomic<int64> buffered_elements_;
  std::atomic<int64> bytes_consumed_;
  std::atomic<int64> bytes_produced_;
  std::atomic<int64> num_elements_;
  std::atomic<int64> processing_time_;
  std::atomic<bool> record_metrics_;
  Metrics metrics_;
  absl::flat_hash_map<string, std::shared_ptr<Parameter>> parameters_
      TF_GUARDED_BY(mu_);

  // Statistic of inputs processing time history.
  double input_processing_time_sum_ = 0.0L;
  int64 input_processing_time_count_ = 0;

  // Inputs of this node. These can represent an iterator created from the input
  // dataset but also other input iterators (e.g. created by the user-defined
  // functions of `flat_map` or `interleave`).
  std::list<std::shared_ptr<Node>> inputs_ TF_GUARDED_BY(mu_);

  // The reference to the output node is not owned so that deletion of a
  // node results in recursive deletion of the subtree rooted in the node.
  Node* const output_;
};

// InterleaveMany is used to model datasets whose inputs are used to create
// datasets whose elements are then interleaved.
std::shared_ptr<Node> MakeInterleaveManyNode(Node::Args args);

// AsyncInterleaveMany nodes are the asynchronous version of InterleaveMany
// nodes.
std::shared_ptr<Node> MakeAsyncInterleaveManyNode(
    Node::Args args, std::vector<std::shared_ptr<Parameter>> parameters);

// KnownMany nodes model datasets that synchronously consume known number of
// input element per output element.
std::shared_ptr<Node> MakeKnownRatioNode(Node::Args args, double ratio);

// AsyncKnownRatio nodes are the asynchronous version of KnownRate nodes.
std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio,
    std::vector<std::shared_ptr<Parameter>> parameters);

// Source nodes represent data sources.
std::shared_ptr<Node> MakeSourceNode(Node::Args args);

// UnknownMany nodes represent datasets that synchronously consume an
// unknown number of input elements per output.
//
// Unlike KnownRatio nodes which expect the ratio between inputs and outputs is
// specified as a parameter, UnknownRatio estimates the ratio empirically.
std::shared_ptr<Node> MakeUnknownRatioNode(Node::Args args);

// Unknown nodes represent datasets for which we do not have a model. It acts
// as pass-through between inputs and output.
std::shared_ptr<Node> MakeUnknownNode(Node::Args args);

// Abstract representation of a TensorFlow input pipeline that can be used
// for collecting runtime information and optimizing performance. It collects
// runtime information about execution of the input pipeline that is used to
// create a performance model, which is in turn used to identify optimal values
// of tunable parameters.
//
// Developers of tf.data transformations are not expected to interact with this
// class directly. Boiler plate code for creating the abstract representation of
// the input pipeline and collecting runtime information has been added to the
// implementation of `DatasetBase` and `DatasetBaseIterator` respectively.
class Model {
 public:
  // Creates a new model.
  Model() : collect_resource_usage_(false) {}

  // Indicates whether to collect resource usage.
  bool collect_resource_usage() const { return collect_resource_usage_; }

  // Adds a node with the given name and given output. The method returns
  // a pointer to the node but does not transfer ownership.
  void AddNode(Node::Factory factory, const string& name,
               const string& output_name, std::shared_ptr<Node>* out_node)
      TF_LOCKS_EXCLUDED(mu_);

  // Flushes metrics record by the model.
  void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // Uses the given algorithm to perform the autotuning optimization.
  void Optimize(AutotuneAlgorithm algorithm, int64 cpu_budget, int64 ram_budget)
      TF_LOCKS_EXCLUDED(mu_);

  // Removes the given node.
  void RemoveNode(const string& name) TF_LOCKS_EXCLUDED(mu_);

 private:
  // Collects tunable parameters in the tree rooted in the given node, returning
  // a mapping from a (unique) node name to a tunable parameter.
  absl::flat_hash_map<string, std::shared_ptr<Parameter>>
  CollectTunableParameters(std::shared_ptr<Node> node);

  // Collects "essential" parallelism parameters of transformations in the tree
  // rooted in the given node. Which parameters are essential is determined by
  // comparison the processing time spent in the corresponding transformation
  // relative to other transformations. The collected parameters are returned
  // as a mapping from a (unique) node name to a parallelism parameter.
  absl::flat_hash_map<string, std::shared_ptr<Parameter>>
  CollectEssentialParallelism(std::shared_ptr<Node> node);

  // This optimization algorithm starts by setting all tunable parallelism
  // parameters to the minimum value. It then repeatedly identifies the
  // parameter whose increase in parallelism decreases the output time the most.
  // This process is repeated until all parameters reach their maximum values or
  // the projected output time is less than or equal to the processing time
  // needed to produce an element divided by CPU budget.
  void OptimizeHillClimb(int64 cpu_budget, int64 ram_budget);

  // This optimization algorithm starts by setting all tunable parallelism
  // parameters to the minimum value. It then improves current parameters by
  // making a step in the direction opposite to the gradient of `OutputTime` and
  // projecting resulting values on the feasible intervals. Improvement step is
  // repeated until either the output time improvement is smaller than threshold
  // value or the output time is less than the processing time needed to produce
  // an element divided by CPU budget.
  void OptimizeGradientDescent(int64 cpu_budget, int64 ram_budget);

  // Collects the output time and if `gradient` is not `nullptr`, the output
  // time gradient w.r.t. tunable parameters of the subtree rooted in the given
  // node and the last input time.
  double OutputTime(std::shared_ptr<Node> node,
                    absl::flat_hash_map<string, double>* gradient);

  // Collects the processing time for the given node.
  double TotalProcessingTime(std::shared_ptr<Node> node);

  // Collects the total number of bytes buffered in all nodes in the subtree
  // rooted in the given node for which autotuning is enabled.
  double TotalBufferedBytes(std::shared_ptr<Node> node);

  // Collects the total buffer limit of all nodes in the subtree rooted in the
  // given node for which autotuning is enabled. This number represents the
  // amount of memory that would be used by the subtree nodes if all of their
  // buffers were full.
  double TotalMaximumBufferedBytes(std::shared_ptr<Node> node);

  // Used for coordination between different input pipeline threads. Exclusive
  // access is required only when adding or removing nodes. Concurrent access to
  // existing nodes is protected by a node mutex.
  mutex mu_;
  int64 id_counter_ TF_GUARDED_BY(mu_) = 1;
  std::shared_ptr<Node> output_ TF_GUARDED_BY(mu_);
  absl::flat_hash_map<string, std::shared_ptr<Node>> lookup_table_
      TF_GUARDED_BY(mu_);

  // Indicates whether the modeling framework should collect resource usage
  // (e.g. CPU, memory). The logic for collecting this information assumes that
  // the collection is not repeatedly disabled and enabled. As a consequence,
  // the implementation starts collecting resource usage when it encounters a
  // tunable parameter (because the information is used for for tuning the value
  // of the parameter) and never stops.
  std::atomic<bool> collect_resource_usage_;
};

}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
