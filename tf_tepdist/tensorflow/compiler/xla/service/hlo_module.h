/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_

#include <atomic>
#include <list>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/service/dynamic_parameter_binding.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {

// Describes a compilation unit at the HLO level.
//
// HloModule is the top-level unit in the HLO IR.  It corresponds to a whole
// "program".  Running a module, from beginning to end, is the only way to run
// an XLA program.
//
// A module contains one "entry computation"; this HloComputation is like main()
// in a C program.  The result of running the module is the result of running
// this computation.
//
// A module also contains some number of "nested computations".  Each nested
// computation is attached to an HloInstruction within some other computation.
// The meaning of the nested computation depends on the instruction it's
// attached to.
class HloModule {
 public:
  struct DefContext {
   public:
    struct SrcOutput {
      int prev_slice_id = -1;
      int def_id = -1;
      int output_idx = -1;
    };

    using SrcOutputMap = std::unordered_map<int/*slice_id*/, SrcOutput>;

    typedef enum DefType {
      ENTRY,
      CG,
      AG,
      GA,
      GAINIT,
      AR,
      CG_SLICE,
      AG_SLICE,
      GA_SLICE,
      GAINIT_SLICE,
      AR_SLICE,
      ALL_TO_ALL
    } DefType;

    // Explicit constructor
    explicit DefContext(std::string& name, DefType def_type, int parent_id);

    explicit DefContext(
        std::string& name, DefType def_type, int parent_id, HloModule* module);

    enum StageType {
      NA,       // not a stage of pipeline
      FORWARD,
      BACKWARD,
      BOTH,     // the stage contains both forward/backward
    };

    // Handy utilities
    int def_id() const { return def_id_; }
    DefType def_type() const { return def_type_; }

    // *parent* denotes the computation from which I am derived.
    int parent_id() const { return parent_def_id_; }

    const std::string& name() const { return name_; }

    int num_children() { return children_.size(); }

    std::vector<DefContext*>& children() { return children_; }

    DefContext* child(int idx) { 
      CHECK(idx < children_.size());
      return children_[idx]; 
    }

    HloModule* module() {
      CHECK(module_);
      return module_; 
    }

    int num_comp_units() {
      CHECK(def_type_ == DefType::CG);
      return num_children() >> 1;
    }
    DefContext* ComputeGradientsDefCtx() {
      CHECK(def_type_ == DefType::ENTRY);
      CHECK (0 < children_.size());
      return children_[0];
    }

    DefContext* ApplyGradientsDefCtx() {
      CHECK(def_type_ == DefType::ENTRY);
      CHECK (3 < children_.size());
      return children_[3];
    }

    DefContext* GADefCtx() {
      CHECK(def_type_ == DefType::ENTRY);
      CHECK (2 < children_.size());
      return children_[2];
    }

    DefContext* GAInitDefCtx() {
      CHECK(def_type_ == DefType::ENTRY);
      CHECK (1 < children_.size());
      return children_[1];
    }

    DefContext* ARDefCtx() {
      CHECK(def_type_ == DefType::ENTRY);
      CHECK (4 < children_.size());
      return children_[4];
    }

    void insert_child_at(int pos, DefContext* def_ctx) {
      CHECK(pos < children_.size());
      children_[pos] = def_ctx;
    }

    void resize_children(int num) {
      if (children_.size() != num) {
        children_.resize(num);
      }
    }

    bool entry_def_ctx() { return def_type_ == DefType::ENTRY; }

    bool cg_def_ctx() { return def_type_ == DefType::CG; }
    bool cg_slice_def_ctx() { return def_type_ == DefType::CG_SLICE; }
    bool ag_def_ctx() { return def_type_ == DefType::AG; }
    bool ag_slice_def_ctx() { return def_type_ == DefType::AG_SLICE; }
    bool ga_def_ctx() { return def_type_ == DefType::GA; }
    bool ga_slice_def_ctx() { return def_type_ == DefType::GA_SLICE; }
    bool ga_init_def_ctx() { return def_type_ == DefType::GAINIT; }
    bool ga_init_slice_def_ctx() { return def_type_ == DefType::GAINIT_SLICE; }
    bool ar_def_ctx() { return def_type_ == DefType::AR; }
    bool ar_slice_def_ctx() { return def_type_ == DefType::AR_SLICE; }
    bool all_to_all_def_ctx() { return def_type_ == DefType::ALL_TO_ALL; }

    int64 input_activation_bytes() {
      if (input_activation_bytes_ == 0) {
        for (auto it : input_activation_size_map_) {
          input_activation_bytes_ += it.second;
        }
      }
      return input_activation_bytes_;
    }

    int64 input_var_bytes() {
      if (input_var_bytes_ == 0) {
        for (auto it : input_var_size_map_) {
          input_var_bytes_ += it.second;
        }
      }
      return input_var_bytes_;
    }

    int64 output_tensor_bytes() {
      if (output_tensor_bytes_ == 0) {
        for (auto it : output_tensor_size_map_) {
          output_tensor_bytes_ += it.second;
        }
      }
      return output_tensor_bytes_;
    }

    int64 output_tensor_bytes_wo_alias() {
      if (output_tensor_bytes_wo_alias_ == 0) {
        std::unordered_set<int64> alias_outputs;
        for (auto& alias : input_output_alias_map_) {
          alias_outputs.insert(alias.second);
        }

        for (auto it : output_tensor_size_map_) {
          if (alias_outputs.find(it.first) == alias_outputs.end()) {
            // not alias
            output_tensor_bytes_wo_alias_ += it.second;
          }
        }
      }
      return output_tensor_bytes_wo_alias_;
    }

    // wrapper to set input_def_map_. We set def_id into AvailableDefs.general when slice_id == -1
    void add_to_input_def_map(
        int arg_no, int slice_id, int prev_slice_id, int def_id, int out_idx);
    const StatusOr<SrcOutput> get_src_output_from_input_def_map(
        const int arg_no, const int slice_id) const;

    // Data members
    // There are three levels of computations:
    // L1 -> Entry
    // L2 -> {ComputeGradients, ApplyGradients, GradientsAccumulation, GAInit, AR}
    // L3 -> {Stage Computations for ComputeGradients/ApplyGradients/GAInit/AR}
    std::string name_;
    std::map<int/*my arg*/, int/*parent arg*/> input_arg_map_;
    std::map<int/*my arg*/, SrcOutputMap> input_def_map_;
    // slice_id of active instances for currrent DefContext
    std::set<int> instance_slice_ids_;
    std::map<int/*my arg*/, int/*partition dim*/> input_dim_to_slice_;
    // TOOD(shiqing.fsq): Actually this represents variable inputs.
    std::vector<int/*my arg no*/> sharded_args_;

    std::map<int/*my idx*/, int/*parent idx*/> output_idx_map_;
    std::map<int64, std::set<int>> output_idx_global_dev_map_; // Records on entry
    std::map<int/*my idx*/, int/*parti. dim*/> output_dim_to_slice_;

    // Used by input output buffer alias in HLO. This is useful for
    // saving memory usage especially in ApplyGradient computation.
    std::map<int/*input idx*/, int/*output idx*/> input_output_alias_map_;

    std::map<int/*my arg*/, int64/*tensor size*/>  input_activation_size_map_;  // activation tensor size
    std::map<int/*my arg*/, int64/*tensor size*/>  input_var_size_map_;         // trainable variable size
    std::map<int/*my arg*/, int64/*tensor size*/>  output_tensor_size_map_;     // may use vector? need confirm

    HloModule* module_ = nullptr;
    int parent_def_id_ = -1;
    std::vector<DefContext*> children_;
    int64 input_activation_bytes_ = 0;
    int64 input_var_bytes_ = 0;
    int64 output_tensor_bytes_ = 0;
    int64 output_tensor_bytes_wo_alias_ = 0;  // don't count input/output alias bytes
    StageType stage_type_ = StageType::NA;
    DefType def_type_;
    int64 gflops_ = 0;

   private:
    friend class HloModule;
    explicit DefContext(
        std::string name, DefType def_type, int parent_id, int def_id);
    int def_id_ = 0;
  };

 public:
  // Constructor without a versioned computation handle. This constructor should
  // only be used for HloModules used outside of the XLA service (eg
  // tests). The versioned handle is used by the service in the compilation
  // cache. A default configuration is created for this module.
  explicit HloModule(const string& name, HloModuleConfig config);
  virtual ~HloModule() { }

  // Adds an entry computation to the module. A module can only have one entry
  // computation. Returns a pointer to the newly added computation.
  HloComputation* AddEntryComputation(
      std::unique_ptr<HloComputation> computation);

  // Replaces the current entry computation with another computation.
  // The new entry computation must be a computation that is already in the
  // module.
  void ReplaceEntryComputation(HloComputation* entry_computation);

  // Adds an embedded computation to the module.
  HloComputation* AddEmbeddedComputation(
      std::unique_ptr<HloComputation> computation);

  void SetScopedSubModule(std::string& scope, HloComputation* sub_module) {
    CHECK(!scoped_submodules_.count(scope));
    scoped_submodules_[scope] = sub_module;
  }

  HloComputation* GetScopedSubModule(std::string& scope) {
    CHECK(scoped_submodules_.count(scope));
    return scoped_submodules_[scope];
  }

  // Removes an embedded computation.
  Status RemoveEmbeddedComputation(HloComputation* to_remove);

  // Removes unused computations.
  Status RemoveUnusedComputations();

  // Replaces all uses of computations that are keys of 'replacements' with
  // the corresponding values in 'replacements'. Replaces the entry computation,
  // if applicable.
  //
  // This function iterates over all instructions in the module to find
  // computations to replace. We could speed it up by keeping track of users of
  // computations.
  void ReplaceComputations(
      const std::unordered_map<HloComputation*, HloComputation*>& replacements);

  const string& name() const { return name_; }
  void set_name(string name) { name_ = std::move(name); }

  // Returns a deep copy of this module including all computations.
  std::unique_ptr<HloModule> Clone(const string& suffix = "clone") const;
  std::unique_ptr<HloModule> Clone(const HloModuleConfig& config,
                                   const string& suffix = "clone") const;

  // Performs a deep clone of the computation, by recursively cloning all
  // the called computations as well. If the clone context is specified, it
  // will be populated with the cloned object mappings.
  HloComputation* DeepCloneComputation(HloComputation* computation,
                                       HloCloneContext* context = nullptr);

  // Return a pointer to the entry computation of the module.
  HloComputation* entry_computation() const {
    CHECK_NE(nullptr, entry_computation_);
    return entry_computation_;
  }

  void set_entry(HloComputation* computation) {
    entry_computation_ = computation;
  }

  bool has_entry_computation() const { return entry_computation_ != nullptr; }

  // Returns the root instruction shape of entry computation.
  //
  // Precondition: entry_computation_ is not nullptr.
  const Shape& result_shape() const {
    CHECK_NE(nullptr, entry_computation_);
    return entry_computation()->root_instruction()->shape();
  }

  // Creates the ComputationLayout which describes the current status of the HLO
  // module entry computation.
  ComputationLayout compute_computation_layout() const {
    return ComputationLayout(entry_computation()->ComputeProgramShape(),
                             /*ignore_layouts=*/false);
  }

  ComputationLayout* mutable_entry_computation_layout() {
    return config_.mutable_entry_computation_layout();
  }

  const ComputationLayout& entry_computation_layout() const {
    return config_.entry_computation_layout();
  }

  // Generates a hash value of an HLO module. Hash considers
  // information on opcode, shape, operands, and typically a root instruction.
  // This function returns the same hash value for equivalent HLO modules,
  // with respect to HloInstruction::Identical() method.
  uint64 Hash() const;

  // Gets the computations in this module.
  //
  // Returns a view of HloComputation*s, so you can iterate over this in the
  // natural way:
  //
  //   for (HloComputation* c : module->computations()) { ... }
  //
  tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::const_iterator>>
  computations() const {
    return {MakeUnwrappingIterator(computations_.begin()),
            MakeUnwrappingIterator(computations_.end())};
  }
  tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::iterator>>
  computations() {
    return {MakeUnwrappingIterator(computations_.begin()),
            MakeUnwrappingIterator(computations_.end())};
  }

  // Returns the computation in this module that has the name `name`.  Returns
  // null if there is no such computation.
  HloComputation* GetComputationWithName(absl::string_view name);

  // Gets the number of computations in this module.
  int64 computation_count() const { return computations_.size(); }

  // Returns the mutable computation for the given index.
  HloComputation* mutable_computation(int64 idx) {
    CHECK(idx >= 0 && idx < computations_.size());
    return computations_[idx].get();
  }

  // Gets the number of instructions in this module.
  int64 instruction_count() const;

  // Deallocate removed instructions in each computation.
  void Cleanup() {
    for (auto& comp : computations_) {
      comp->Cleanup();
    }
  }

  // Compute and return a post order of all computations in the module. The sort
  // is defined like so: if computation A has an instruction which calls
  // computation B, then A will appear after B in the sort.
  std::vector<HloComputation*> MakeComputationPostOrder() const;

  // Same as MakeComputationPostOrder() but sorting the computations by their
  // contents. The order is longer post order.
  std::vector<HloComputation*> MakeComputationSorted() const;

  // Gets the computations in this module which aren't for fusion nodes.
  //
  // Postcondition: All computations in the returned list have
  // !IsFusionComputation().
  //
  // Note: Callers can and do rely on the return value here being a *snapshot*
  // of the module's non-fusion computations -- that is, it's OK to add or
  // remove computations from a module while iterating over
  // MakeNonfusionComputations().
  std::vector<HloComputation*> MakeNonfusionComputations() const;

  // Same as MakeNonfusionComputations() but sorting computations by content.
  std::vector<HloComputation*> MakeNonfusionComputationsSorted() const;

  const HloModuleConfig& config() const { return config_; }
  void set_config(const HloModuleConfig& config) { config_ = config; }

  // Return a string representation of the module.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  string ToString() const { return ToString(HloPrintOptions()); }
  string ToString(const HloPrintOptions& options) const;

  // Convert an HloModule to or from a proto.
  HloModuleProto ToProto() const;
  static StatusOr<std::unique_ptr<HloModule>> CreateFromProto(
      const HloModuleProto& proto, const HloModuleConfig& module_config,
      bool prohibit_empty_literal = true);

  // Creates and returns an HloModuleConfig with an appropriate program shape
  // for the HLO module in the given proto.
  static StatusOr<HloModuleConfig> CreateModuleConfigFromProto(
      const HloModuleProto& module, const DebugOptions& debug_options,
      const ExecutionOptions* execution_options = nullptr);

  // Creates and returns an HloModuleConfig with an appropriate program shape
  // for the HLO module in the given proto.
  static StatusOr<HloModuleConfig> CreateModuleConfigFromShape(
      const ProgramShape& program_shape, const DebugOptions& debug_options,
      const ExecutionOptions* execution_options = nullptr);

  // Outlines the given expression from the given computation.
  // instructions_to_outline contains the instructions that form the expression.
  //
  // Precondition: instructions in instructions_to_outline are in topological
  // order (root of outlined instructions last). TODO(jingyue): takes a set of
  // instructions and topologically sorts them.
  HloInstruction* OutlineExpressionFromComputation(
      absl::Span<HloInstruction* const> instructions_to_outline,
      const string& outlined_computation_name, HloComputation* computation);

  // Returns a randomly generated uint64.
  uint64 RandomNew64() const;

  // Returns the NameUniquer for uniquing instruction names in this module.
  NameUniquer& instruction_name_uniquer() { return instruction_name_uniquer_; }

  // Assign a new unique dense id for an instruction
  int NewUniqueInstructionId() {
    int result = next_unique_id_;
    next_unique_id_++;
    return result;
  }

  // input_output_alias_config indicates the list of aliased buffers that are
  // expected from the module.
  HloInputOutputAliasConfig& input_output_alias_config() {
    return input_output_alias_config_;
  }
  const HloInputOutputAliasConfig& input_output_alias_config() const {
    return input_output_alias_config_;
  }

  // DynamicParameterBinding holds the list of bindings that indicates which
  // parameter dimensions are dynamic and which parameters represent their
  // runtime value.
  DynamicParameterBinding& dynamic_parameter_binding() {
    return dynamic_parameter_binding_;
  }
  const DynamicParameterBinding& dynamic_parameter_binding() const {
    return dynamic_parameter_binding_;
  }

  // Returns an id that is unique to this module across all modules created over
  // the lifetime of this process.
  int unique_id() const { return unique_id_; }

  // Sets the schedule of the module to the given schedule.
  Status set_schedule(HloSchedule schedule);

  // Clears the schedule of the module.
  void clear_schedule() { schedule_.reset(); }

  // Returns true if the module has a schedule set.
  bool has_schedule() const { return schedule_.has_value(); }

  // Auto-Parallel Start
  // (TODO. refine the variable and symbolic initialization)
  //void set_variable_map(std::map<int, string>* variable_map);
  std::map<int, string>* variable_map();
  const std::map<int, string>* variable_map() const;
  std::map<int, string>* init_specs_map();
  std::vector<int>& fetch_vars_list();
  void set_symbolic_map(std::map<int, std::set<int>>* symbolic_map);
  std::map<int, std::set<int>>* symbolic_map() const;

  std::unique_ptr<HloModule> Clone(std::string prefix, HloComputation*);

  void set_def_ctx(DefContext* ctx) { def_ctx_ = ctx; }
  DefContext* def_ctx() { return def_ctx_; }
  const DefContext* def_ctx() const { return def_ctx_; }
  DefContext* new_def_ctx(
      std::string name, DefContext::DefType def_type, int parent_id) {
    def_ctx_pool_.push_back(
        absl::make_unique<HloModule::DefContext>(name, def_type, parent_id));
    return def_ctx_pool_.back().get();
  }

  const std::vector<std::unique_ptr<HloModule::DefContext>>& all_def_ctx() const {
    return def_ctx_pool_;
  }

  HloModule::DefContext* create_def_ctx_from_proto(
    const ModuleDefContext& module_def_ctx);

  HloComputation* Def2Compute(DefContext* def_ctx) {
    if (def_computation_.count(def_ctx)) {
      return def_computation_[def_ctx];
    }
    return nullptr;
  }

  void SetDefCompute(DefContext* def_ctx, HloComputation* computation) {
    CHECK(!def_computation_.count(def_ctx));
    def_computation_[def_ctx] = computation;
  }
  std::unordered_map<DefContext*, HloComputation*>& DefComputation() {
    return def_computation_;
  }

  DefContext* def_ctx(int def_id);
  void StealDefCtx(HloModule* module,
                   std::unordered_map<HloComputation*, HloComputation*>& replacements);
  void CopyMetaData(HloModule* module);
  // Auto-Parallel End

  // Returns the schedule of the module. CHECK fails if no schedule is set.
  const HloSchedule& schedule() const { return *schedule_; }
  HloSchedule& schedule() { return *schedule_; }

  HloComputation* AddComputationAndUnifyNamesAndIds(
      std::unique_ptr<HloComputation> computation, bool is_entry) {
    computation->ClearUniqueIdInternal();
    for (auto* instruction : computation->instructions()) {
      instruction->ClearUniqueIdInternal();
    }
    return AddComputationInternal(std::move(computation), is_entry,
                                  /*uniquify_identifiers=*/true);
  }

  Status CheckUniqueNamesAndIdsForComputationsAndInstructions() const;

  // Checks if this config has a list of entry parameters' HLO shardings for
  // SPMD.
  bool has_spmd_parameters_shardings() const {
    return spmd_parameters_shardings_.has_value();
  }

  // Getter and setter for the list of entry parameters' HLO shardings for SPMD.
  const std::vector<HloSharding>& spmd_parameters_shardings() const {
    CHECK(spmd_parameters_shardings_.has_value());
    return *spmd_parameters_shardings_;
  }
  void set_spmd_parameters_shardings(
      const std::vector<HloSharding>& shardings) {
    spmd_parameters_shardings_ = shardings;
  }

  // Checks if this config has the entry computation output's HLO sharding for
  // SPMD.
  bool has_spmd_output_sharding() const {
    return spmd_output_sharding_.has_value();
  }

  // Getter and setter for the entry computation output's HLO shardings for
  // SPMD.
  const HloSharding& spmd_output_sharding() const {
    CHECK(spmd_output_sharding_.has_value());
    return *spmd_output_sharding_;
  }
  void set_spmd_output_sharding(const HloSharding& sharding) {
    spmd_output_sharding_ = sharding;
  }

  // Add a program argument to be prefetched across programs.
  void AddCrossProgramPrefetch(int64 parameter, const ShapeIndex& index) {
    cross_program_prefetches_.emplace_back(parameter, index);
  }

  // Get the list of program arguments to be prefetch across programs.
  const absl::Span<const std::pair<int64, ShapeIndex>> CrossProgramPrefetches()
      const {
    return cross_program_prefetches_;
  }

  void record_split_info(int split_num, bool share_dev_flag) {
    split_nums_.emplace_back(split_num);
    share_dev_flags_.emplace_back(share_dev_flag);
    placement_layout_.insert(placement_layout_.begin(), placement_layout_.size());
  }

  void set_placement_layout(std::vector<int>& placement_layout) {
    CHECK_EQ(placement_layout.size(), split_nums_.size());
    placement_layout_ = placement_layout;
  }

  const std::vector<int>& placement_layout() const { return placement_layout_; }

  void set_stage_split_ordinal(int stage_split_ordinal) {
    stage_split_ordinal_ = stage_split_ordinal;
  }

  const int stage_split_ordinal() const { return stage_split_ordinal_; }

  const std::vector<int>& split_nums() const {
    return split_nums_;
  }

  int last_split_num() const {
    CHECK(!split_nums_.empty());
    return split_nums_.back();
  }

  int total_split_num() const {
    int split_num = 1;
    for (auto num : split_nums_) {
      split_num *= num;
    }
    return split_num;
  }

  const std::vector<bool>& share_dev_flags() const {
    return share_dev_flags_;
  }

  int last_share_dev_flag() const {
    CHECK(!share_dev_flags_.empty());
    return share_dev_flags_.back();
  }

  int total_dev_num() const {
    int split_num = 1;
    CHECK(split_nums_.size() == share_dev_flags_.size());

    for (int i=0; i<split_nums_.size(); ++i) {
      if (share_dev_flags_[i] == false) {
        split_num *= split_nums_[i];
      }
    }
    return split_num;
  }

  int64 num_dev_per_worker() const {
    return num_dev_per_worker_;
  }
  void set_num_dev_per_worker(int64 num_dev_per_worker) {
    num_dev_per_worker_ = num_dev_per_worker;
  }
  int64 num_worker() const {
    return num_worker_;
  }
  void set_num_worker(int64 num_worker) {
    num_worker_ = num_worker;
  }
 private:
  HloComputation* AddComputationInternal(
      std::unique_ptr<HloComputation> computation, bool is_entry,
      bool uniquify_identifiers);

  string name_;
  HloModuleConfig config_;
  HloComputation* entry_computation_ = nullptr;
  std::vector<std::unique_ptr<HloComputation>> computations_;

  // Random number generator engine to use when generating random numbers per
  // HloModule compilation.
  // TODO(b/25995601): Replace with better seed setting or dev/random for
  // where we don't need deterministic execution.
  mutable std::mt19937_64 rng_{42};
  mutable tensorflow::mutex rng_mutex_;

  // Unique name generator for computation and instruction names, which are
  // unique per module.
  NameUniquer computation_name_uniquer_{/*separator=*/"."};
  NameUniquer instruction_name_uniquer_{/*separator=*/"."};
  int next_unique_id_ = 0;

  // Used to keep track of the next unique module id that should be assigned.
  static std::atomic<int> next_unique_module_id_;
  // A unique id to label modules with.
  int unique_id_;

  // The HloSchedule of the module. The schedule if it exists contains a
  // sequential order of instructions for each non-fusion computation in the
  // module.
  absl::optional<HloSchedule> schedule_;

  // alias_config indicates the alias information of input/output buffers that
  // are expected from the module.
  HloInputOutputAliasConfig input_output_alias_config_;

  // Bindings for dynamic parameter mapping.
  DynamicParameterBinding dynamic_parameter_binding_;

  // The HLO shardings of the entry computation's parameters for
  // SPMD-partitioned programs.
  absl::optional<std::vector<HloSharding>> spmd_parameters_shardings_;

  // The HLO sharding of the entry computation's output (root) for
  // SPMD-partitioned programs.
  absl::optional<HloSharding> spmd_output_sharding_;

  // Arguments to be prefetched across programs.
  std::vector<std::pair<int64, ShapeIndex>> cross_program_prefetches_;

  std::unordered_map<std::string, HloComputation*> scoped_submodules_;

  // Auto-Parallel Start
  // Trainable variable map
  std::map<int, string> variable_map_;
  std::map<int, string> init_specs_map_;
  std::vector<int> fetch_vars_list_;

  // Symbolic map
  std::map<int, std::set<int>>* symbolic_map_ = nullptr;

  DefContext* def_ctx_ = nullptr;
  std::vector<std::unique_ptr<DefContext>> def_ctx_pool_;
  std::unordered_map<DefContext*, HloComputation*> def_computation_;

  std::vector<int> split_nums_;
  std::vector<bool> share_dev_flags_;
  int stage_split_ordinal_ = -1;
  std::vector<int> placement_layout_;
  int64 num_dev_per_worker_;
  int64 num_worker_;
  // Auto-Parallel End
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
