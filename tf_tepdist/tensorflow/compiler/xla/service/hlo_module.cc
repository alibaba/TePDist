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

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include <iterator>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

static int max_def_id = 0;

HloModule::DefContext::DefContext(
    std::string& name, DefType def_type, int parent_id)
    : name_(name), def_type_(def_type), parent_def_id_(parent_id),
      module_(nullptr), def_id_(max_def_id++) {}

HloModule::DefContext::DefContext(
    std::string& name, DefType def_type, int parent_id, HloModule* module)
    : name_(name), def_type_(def_type), parent_def_id_(parent_id),
      module_(module), def_id_(max_def_id++) {}

HloModule::DefContext::DefContext(
    std::string name, DefType def_type, int parent_id, int def_id)
    : name_(name), def_type_(def_type), parent_def_id_(parent_id),
      module_(nullptr), def_id_(def_id) {}

void HloModule::DefContext::add_to_input_def_map(
    int arg_no, int slice_id, int prev_slice_id, int def_id, int out_idx) {
  if (input_def_map_.find(arg_no) == input_def_map_.end()) {
    HloModule::DefContext::SrcOutputMap src_output_map;
    input_def_map_[arg_no] = src_output_map;
  }

  HloModule::DefContext::SrcOutputMap& src_output_map = input_def_map_.at(arg_no);
  CHECK(src_output_map.find(slice_id) == src_output_map.end());
  HloModule::DefContext::SrcOutput src_output = {prev_slice_id, def_id, out_idx};
  src_output_map[slice_id] = src_output;
}

const StatusOr<HloModule::DefContext::SrcOutput>
HloModule::DefContext::get_src_output_from_input_def_map(
    const int arg_no, const int slice_id) const {
  if (input_def_map_.find(arg_no) == input_def_map_.end()) {
    return InvalidArgument("arg_no ", arg_no, " Does not exist in input_def_map");
  }

  const HloModule::DefContext::SrcOutputMap& src_output_map = input_def_map_.at(arg_no);
  auto iter = src_output_map.find(slice_id);
  CHECK(iter != src_output_map.end());
  return iter->second;
}

HloModule::HloModule(const string& name, HloModuleConfig config)
    : name_(NameUniquer::GetSanitizedName(name)),
      config_(std::move(config)),
      unique_id_(next_unique_module_id_++) {}

HloModule::DefContext* HloModule::create_def_ctx_from_proto(
    const ModuleDefContext& module_def_ctx) {
  def_ctx_pool_.push_back(
      absl::WrapUnique(new HloModule::DefContext(
          module_def_ctx.name(),
          (HloModule::DefContext::DefType)module_def_ctx.def_type(),
          module_def_ctx.parent_id(), module_def_ctx.def_id())));
  auto* def_ctx = def_ctx_pool_.back().get();

  // input arg map
  auto& input_arg_map = module_def_ctx.input_arg_map();
  for (auto& it : input_arg_map) {
    def_ctx->input_arg_map_[it.first] = it.second;
  }
  // input def map
  for (auto& out_iter : module_def_ctx.input_def_map()) {
    int arg_no = out_iter.first;
    auto& src_output_map_proto = out_iter.second;
    for (auto& in_iter : src_output_map_proto.src_output_map()) {
      int slice_id = in_iter.first;
      auto& src_output_proto = in_iter.second;
      def_ctx->add_to_input_def_map(
          arg_no, slice_id, src_output_proto.prev_slice_id(),
          src_output_proto.def_id(), src_output_proto.output_idx());
    }
  }
  // input dim to slice
  auto& input_dim_to_slice = module_def_ctx.input_dim_to_slice();
  for (auto& it : input_dim_to_slice) {
    def_ctx->input_dim_to_slice_[it.first] = it.second;
  }
  // sharded_args
  auto& sharded_args = module_def_ctx.sharded_args();
  def_ctx->sharded_args_.assign(sharded_args.begin(), sharded_args.end());
  // output_idx_map
  auto& output_idx_map = module_def_ctx.output_idx_map();
  for (auto& it : output_idx_map) {
    def_ctx->output_idx_map_[it.first] = it.second;
  }
  // output dim to slice
  auto& output_dim_to_slice = module_def_ctx.output_dim_to_slice();
  for (auto& it : output_dim_to_slice) {
    def_ctx->output_dim_to_slice_[it.first] = it.second;
  }

  auto& output_idx_global_dev_map = module_def_ctx.output_idx_global_dev_map();
  for (auto& it : output_idx_global_dev_map) {
    std::set<int> slice_ids;
    auto& global_slices_proto = it.second;
    for (int id : global_slices_proto.slices()) {
      slice_ids.insert(id);
    }
    def_ctx->output_idx_global_dev_map_[it.first] = slice_ids;
  }

  // input output alias map
  auto& input_output_alias_map = module_def_ctx.input_output_alias_map();
  for (auto& it : input_output_alias_map) {
    def_ctx->input_output_alias_map_[it.first] = it.second;
  }
  return def_ctx;
}

HloModule::DefContext* HloModule::def_ctx(int def_id) {
  for (auto& def_ctx_unique : def_ctx_pool_) {
    if (def_id == def_ctx_unique->def_id()) {
      return def_ctx_unique.get();
    }
  }
  return nullptr;
}

Status HloModule::set_schedule(HloSchedule schedule) {
  TF_RET_CHECK(schedule.module() == this);
  TF_RETURN_IF_ERROR(schedule.Verify());
  schedule_ = std::move(schedule);
  return Status::OK();
}

void HloModule::ReplaceEntryComputation(HloComputation* entry_computation) {
  entry_computation_ = entry_computation;
  config_.SetDefaultComputationLayout(
      entry_computation_->ComputeProgramShape());
  input_output_alias_config_ = HloInputOutputAliasConfig(
      entry_computation_->root_instruction()->shape());
}

HloComputation* HloModule::AddComputationInternal(
    std::unique_ptr<HloComputation> computation, bool is_entry,
    bool uniquify_identifiers) {
  if (is_entry) {
    CHECK_EQ(nullptr, entry_computation_);
    entry_computation_ = computation.get();

    // If the module configuration has no entry layout computation set, create a
    // default one based on the program shape.
    if (!config_.has_entry_computation_layout()) {
      config_.SetDefaultComputationLayout(
          entry_computation_->ComputeProgramShape());
    }
    input_output_alias_config_ = HloInputOutputAliasConfig(
        entry_computation_->root_instruction()->shape());
  }

  if (uniquify_identifiers) {
    computation->UniquifyName(&computation_name_uniquer_);
    for (auto* instruction : computation->instructions()) {
      instruction->UniquifyName(&instruction_name_uniquer_);
    }

    // Pick unique IDs for each instruction.
    for (auto* instruction : computation->instructions()) {
      instruction->SetUniqueId(NewUniqueInstructionId());
    }
    // Set unique id to this computation.
    CHECK_NE(computation->root_instruction()->unique_id(), -1)
        << "Root has no valid id: " << computation->ToString();
    computation->SetUniqueId(computation->root_instruction()->unique_id());
  } else {
    // Don't uniquify the names of the computation or instruction, but we must
    // run the names through the uniquifiers to prevent future name collisions
    // for computations and instructions created later. Also, set the
    // next_unique_id_ to the one greater than the max unique id of any
    // instruction (or the computation) to avoid ID collisions.
    computation_name_uniquer_.GetUniqueName(computation->name());
    for (auto* instruction : computation->instructions()) {
      instruction_name_uniquer_.GetUniqueName(instruction->name());
      next_unique_id_ = std::max(next_unique_id_, instruction->unique_id() + 1);
    }
    if (next_unique_id_ < computation->unique_id() + 1) {
      next_unique_id_ = computation->unique_id() + 1;
    }
  }

  computation->set_parent(this);
  computations_.push_back(std::move(computation));
  return computations_.back().get();
}

HloComputation* HloModule::AddEntryComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true);
}

Status HloModule::RemoveEmbeddedComputation(HloComputation* to_remove) {
  if (has_schedule() && !to_remove->IsFusionComputation()) {
    schedule_->remove_computation(to_remove);
  }

  auto it = absl::c_find_if(
      computations_, [&to_remove](const std::unique_ptr<HloComputation>& comp) {
        return comp.get() == to_remove;
      });
  TF_RET_CHECK(it != computations_.end());
  TF_RET_CHECK(it->get() == to_remove);
  computations_.erase(it);
  return Status::OK();
}

HloComputation* HloModule::AddEmbeddedComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/false,
                                /*uniquify_identifiers=*/true);
}

void HloModule::ReplaceComputations(
    const std::unordered_map<HloComputation*, HloComputation*>& replacements) {
  // Replace all uses of non-canonical computations with their
  // representatives.
  std::vector<std::unique_ptr<HloComputation>> new_computations;
  new_computations.reserve(computations_.size());

  for (std::unique_ptr<HloComputation>& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      switch (instruction->opcode()) {
        case HloOpcode::kAllReduce:
        case HloOpcode::kCall:
        case HloOpcode::kMap:
        case HloOpcode::kReduce:
        case HloOpcode::kReduceWindow:
        case HloOpcode::kScatter:
        case HloOpcode::kSort: {
          HloComputation* new_arg = tensorflow::gtl::FindWithDefault(
              replacements, instruction->to_apply(), nullptr);
          if (new_arg != nullptr) {
            instruction->set_to_apply(new_arg);
          }
          break;
        }
        case HloOpcode::kWhile: {
          HloComputation* new_condition = tensorflow::gtl::FindWithDefault(
              replacements, instruction->while_condition(), nullptr);
          if (new_condition != nullptr) {
            instruction->set_while_condition(new_condition);
          }
          HloComputation* new_body = tensorflow::gtl::FindWithDefault(
              replacements, instruction->while_body(), nullptr);
          if (new_body != nullptr) {
            instruction->set_while_body(new_body);
          }
          break;
        }
        case HloOpcode::kConditional: {
          for (int b = 0; b < instruction->branch_count(); ++b) {
            HloComputation* new_computation = tensorflow::gtl::FindWithDefault(
                replacements, instruction->branch_computation(b), nullptr);
            if (new_computation != nullptr) {
              instruction->set_branch_computation(b, new_computation);
            }
          }
          break;
        }
        case HloOpcode::kSelectAndScatter: {
          HloComputation* new_select = tensorflow::gtl::FindWithDefault(
              replacements, instruction->select(), nullptr);
          if (new_select != nullptr) {
            instruction->set_select(new_select);
          }
          HloComputation* new_scatter = tensorflow::gtl::FindWithDefault(
              replacements, instruction->scatter(), nullptr);
          if (new_scatter != nullptr) {
            instruction->set_scatter(new_scatter);
          }
          break;
        }
        default:
          break;
      }
    }

    if (replacements.find(computation.get()) == replacements.end()) {
      new_computations.push_back(std::move(computation));
    }
  }

  // Replace entry_computation if necessary.
  entry_computation_ = tensorflow::gtl::FindWithDefault(
      replacements, entry_computation_, entry_computation_);

  for (auto& iter : def_computation_) {
    if (replacements.find(iter.second) != replacements.end()) {
      def_computation_[iter.first] = replacements.at(iter.second);
    }
  }

  computations_ = std::move(new_computations);
}

string HloModule::ToString(const HloPrintOptions& options) const {
  std::ostringstream s;
  s << "HloModule " << PrintName(name(), options.print_ids());
  if (has_schedule()) {
    TF_CHECK_OK(schedule().Verify());
    s << ", is_scheduled=true";
  }
  s << "\n\n";
  const auto& computations = options.canonicalize_computations()
                                 ? MakeComputationSorted()
                                 : MakeComputationPostOrder();
  for (const HloComputation* computation : computations) {
    if (!options.print_computation(computation)) {
      continue;
    }
    if (computation == entry_computation()) {
      s << "ENTRY ";
    }
    if (has_schedule() && schedule().is_computation_scheduled(computation)) {
      s << computation->ToString(
               options, schedule().sequence(computation).instructions())
        << "\n\n";
    } else {
      s << computation->ToString(options) << "\n\n";
    }
  }
  return s.str();
}

HloModuleProto HloModule::ToProto() const {
  HloModuleProto proto;
  proto.set_id(unique_id_);
  proto.set_name(name_);
  proto.set_entry_computation_name(entry_computation_->name());
  proto.set_entry_computation_id(entry_computation_->unique_id());
  proto.set_num_dev_per_worker(num_dev_per_worker_);
  proto.set_num_worker(num_worker_);
  for (const HloComputation* computation : MakeComputationPostOrder()) {
    HloComputationProto computation_proto = computation->ToProto();
    proto.add_computations()->Swap(&computation_proto);
  }
  if (has_schedule()) {
    *proto.mutable_schedule() = schedule().ToProto().ValueOrDie();
  }
  *proto.mutable_host_program_shape() =
      entry_computation_layout().ComputeProgramShape().ToProto();
  *proto.mutable_input_output_alias() = input_output_alias_config().ToProto();
  *proto.mutable_dynamic_parameter_binding() =
      dynamic_parameter_binding().ToProto();
  for (const auto& parameter_indices : CrossProgramPrefetches()) {
    const auto& parameter = parameter_indices.first;
    const auto& indices = parameter_indices.second;
    auto* prefetch = proto.mutable_cross_program_prefetches()->Add();
    prefetch->set_parameter(parameter);
    for (auto index : indices) {
      prefetch->add_index(index);
    }
  }
  for (const auto& var : variable_map_) {
    (*proto.mutable_variable_map())[var.first] = var.second;
  }

  for (const auto& init : init_specs_map_) {
    (*proto.mutable_init_specs_map())[init.first] = init.second;
  }

  for (const int var_id : fetch_vars_list_) {
    proto.add_fetch_vars(var_id);
  } 
  return proto;
}

Status HloModule::CheckUniqueNamesAndIdsForComputationsAndInstructions() const {
  absl::flat_hash_set<string> computation_names;
  absl::flat_hash_set<int> computation_ids;
  absl::flat_hash_set<string> instruction_names;
  absl::flat_hash_set<int> instruction_ids;

  for (const HloComputation* computation : computations()) {
    TF_RET_CHECK(!ContainsKey(computation_names, computation->name()))
        << "Computation name is not unique: " << computation->name();
    computation_names.insert(computation->name());

    TF_RET_CHECK(!ContainsKey(computation_ids, computation->unique_id()))
        << "Computation id is not unique: " << computation->unique_id();
    computation_ids.insert(computation->unique_id());

    for (const HloInstruction* instruction : computation->instructions()) {
      TF_RET_CHECK(!ContainsKey(instruction_names, instruction->name()))
          << "Instruction name is not unique: " << instruction->name();
      instruction_names.insert(instruction->name());

      TF_RET_CHECK(!ContainsKey(instruction_ids, instruction->unique_id()))
          << "Instruction id is not unique: " << instruction->unique_id();
      instruction_ids.insert(instruction->unique_id());
    }
  }
  return Status::OK();
}

/* static */
StatusOr<std::unique_ptr<HloModule>> HloModule::CreateFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config,
    bool prohibit_empty_literal) {
  VLOG(2) << "CreateFromProto()";
  XLA_VLOG_LINES(3, proto.DebugString());

  // The ProgramShape in the passed in module config must match the shapes of
  // the entry parameters and root.
  TF_RET_CHECK(proto.has_host_program_shape())
      << "No program shape found in the proto";
  ProgramShape expected_program_shape(proto.host_program_shape());
  TF_RET_CHECK(expected_program_shape.parameters_size() ==
               module_config.entry_computation_layout().parameter_count());
  for (int i = 0; i < expected_program_shape.parameters_size(); ++i) {
    const Shape& parameter_shape =
        module_config.entry_computation_layout().parameter_layout(i).shape();
    TF_RET_CHECK(ShapeUtil::Compatible(expected_program_shape.parameters(i),
                                       parameter_shape))
        << "HloModuleConfig has different shape for parameter " << i
        << " than the HLO module. Expected: "
        << ShapeUtil::HumanStringWithLayout(
               expected_program_shape.parameters(i))
        << ", actual: " << ShapeUtil::HumanStringWithLayout(parameter_shape);
  }
  const Shape& result_shape =
      module_config.entry_computation_layout().result_layout().shape();
  TF_RET_CHECK(
      ShapeUtil::Compatible(expected_program_shape.result(), result_shape))
      << "HloModuleConfig has different result shape than the HLO module. "
         "Expected: "
      << ShapeUtil::HumanStringWithLayout(expected_program_shape.result())
      << ", actual: " << ShapeUtil::HumanStringWithLayout(result_shape);

  absl::flat_hash_map<int64, HloComputation*> computation_map;
  absl::flat_hash_map<HloComputation*, int64> to_proto_id;
  std::vector<std::unique_ptr<HloComputation>> computations;
  HloComputation* entry = nullptr;
  for (const HloComputationProto& computation_proto : proto.computations()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloComputation> computation,
        HloComputation::CreateFromProto(computation_proto, computation_map,
                                        prohibit_empty_literal));
    CHECK_NE(computation.get(), nullptr);
    int64 computation_id = computation_proto.id();
    TF_RET_CHECK(computation_id != -1);
    TF_RET_CHECK(!ContainsKey(computation_map, computation_id));
    computation_map[computation_id] = computation.get();
    to_proto_id[computation.get()] = computation_id;
    if (computation_id == proto.entry_computation_id()) {
      entry = computation.get();
    }
    computations.push_back(std::move(computation));
  }
  TF_RET_CHECK(entry != nullptr);

  auto module = absl::make_unique<HloModule>(proto.name(), module_config);

  // Sort the computations in the proto id's order.
  absl::c_sort(computations, [&](const std::unique_ptr<HloComputation>& a,
                                 const std::unique_ptr<HloComputation>& b) {
    return to_proto_id[a.get()] < to_proto_id[b.get()];
  });

  // Add sorted computations to the module.
  for (auto& computation : computations) {
    bool is_entry = computation.get() == entry;
    // Don't uniquify names because we want names to be stable across
    // serialization and deserialization.
    module->AddComputationInternal(std::move(computation), is_entry,
                                   /*uniquify_identifiers=*/false);
  }
  TF_RET_CHECK(module->entry_computation_ != nullptr);

  TF_ASSIGN_OR_RETURN(
      module->input_output_alias_config_,
      HloInputOutputAliasConfig::CreateFromProto(
          entry->ComputeProgramShape().result(), proto.input_output_alias()));

  // Because we didn't uniquify the names or the ids, double-check that the
  // instruction and computation names and ids are unique from the proto.
  TF_ASSIGN_OR_RETURN(module->dynamic_parameter_binding_,
                      DynamicParameterBinding::CreateFromProto(
                          proto.dynamic_parameter_binding()));

  TF_RETURN_IF_ERROR(
      module->CheckUniqueNamesAndIdsForComputationsAndInstructions());

  if (proto.has_schedule()) {
    TF_ASSIGN_OR_RETURN(
        HloSchedule schedule,
        HloSchedule::CreateFromProto(module.get(), proto.schedule()));
    TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  }

  for (auto prefetch : proto.cross_program_prefetches()) {
    module->AddCrossProgramPrefetch(
        prefetch.parameter(),
        ShapeIndex(prefetch.index().begin(), prefetch.index().end()));
  }

  auto variable_map = module->variable_map();
  for (const auto& var : proto.variable_map()) {
    (*variable_map)[var.first] = var.second;
  }

  auto init_specs_map = module->init_specs_map();
  for (const auto& init : proto.init_specs_map()) {
    (*init_specs_map)[init.first] = init.second;
  }

  auto& fetch_vars_list = module->fetch_vars_list();
  for (const int var_idx : proto.fetch_vars()) {
    fetch_vars_list.push_back(var_idx);
  }

  module->set_num_dev_per_worker(proto.num_dev_per_worker());
  module->set_num_worker(proto.num_worker());

  return std::move(module);
}

/* static */
StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromShape(
    const ProgramShape& program_shape, const DebugOptions& debug_options,
    const ExecutionOptions* execution_options) {
  HloModuleConfig module_config(ProgramShape{program_shape});
  module_config.set_debug_options(debug_options);
  if (execution_options) {
    if (execution_options->num_replicas() > 0) {
      module_config.set_replica_count(execution_options->num_replicas());
    }
    if (execution_options->num_partitions() > 0) {
      module_config.set_num_partitions(execution_options->num_partitions());
    }
    if (execution_options->has_device_assignment()) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<DeviceAssignment> device_assignment,
                          DeviceAssignment::Deserialize(
                              execution_options->device_assignment()));
      module_config.set_static_device_assignment(*device_assignment);
      if (execution_options->num_replicas() > 0) {
        CHECK_EQ(module_config.static_device_assignment().replica_count(),
                 module_config.replica_count());
      }
      if (execution_options->num_partitions() > 0) {
        CHECK_EQ(module_config.static_device_assignment().computation_count(),
                 module_config.num_partitions());
      }
    }
  }

  // The module config is constructed with default layouts regardless of what is
  // passed in via the ProgramShape. Set the layouts to the appropriate values.
  ComputationLayout* entry_layout =
      module_config.mutable_entry_computation_layout();
  for (int64 i = 0; i < entry_layout->parameter_count(); ++i) {
    TF_RETURN_IF_ERROR(
        entry_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            program_shape.parameters(i)));
  }
  TF_RETURN_IF_ERROR(entry_layout->mutable_result_layout()->CopyLayoutFromShape(
      program_shape.result()));
  return module_config;
}

/* static */
StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromProto(
    const HloModuleProto& module, const DebugOptions& debug_options,
    const ExecutionOptions* execution_options) {
  TF_RET_CHECK(module.has_host_program_shape())
      << "No program shape found in the proto";
  ProgramShape program_shape(module.host_program_shape());
  return CreateModuleConfigFromShape(program_shape, debug_options,
                                     execution_options);
}

namespace {
// Returns whether `hlo` is used outside the given subcomputation.
// `instructions_in_subcomputation` is the instruction set of the given
// subcomputation.
bool IsUsedOutsideSubcomputation(const HloInstruction& hlo,
                                 const absl::flat_hash_set<HloInstruction*>&
                                     instructions_in_subcomputation) {
  return absl::c_any_of(hlo.users(), [&](HloInstruction* user) {
    return !instructions_in_subcomputation.contains(user);
  });
}
}  // anonymous namespace

HloInstruction* HloModule::OutlineExpressionFromComputation(
    absl::Span<HloInstruction* const> instructions_to_outline,
    const string& outlined_computation_name, HloComputation* computation) {
  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their counterparts in the new outlined
  // function.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> outlined_instructions;
  // A set that contains all instructions to be outlined.
  absl::flat_hash_set<HloInstruction*> instruction_set_to_outline(
      instructions_to_outline.begin(), instructions_to_outline.end());
  std::vector<HloInstruction*> arguments;
  std::vector<HloInstruction*> outputs;
  int64 parameter_count = 0;
  for (HloInstruction* instruction_to_outline : instructions_to_outline) {
    // Clone the original instruction.
    HloInstruction* outlined_instruction =
        builder.AddInstruction(instruction_to_outline->Clone());

    // Replace its operands to their counterparts in the new function.
    for (int64 operand_num = 0;
         operand_num < outlined_instruction->operand_count(); ++operand_num) {
      HloInstruction* old_operand =
          outlined_instruction->mutable_operand(operand_num);

      HloInstruction** operand_slot = &(outlined_instructions[old_operand]);
      if (*operand_slot == nullptr) {
        // Because instructions_to_outline is in topological order, if
        // old_operand is not in outlined_instructions, old_operand must be an
        // input of the outlined subcomputation and thus should be represented
        // as a parameter in the new function.
        arguments.push_back(old_operand);
        *operand_slot = builder.AddInstruction(HloInstruction::CreateParameter(
            parameter_count, old_operand->shape(), "p"));
        ++parameter_count;
      }
      TF_CHECK_OK(
          outlined_instruction->ReplaceOperandWith(operand_num, *operand_slot));
    }

    // Insert the new instruction into the outlined_instructions map.
    InsertOrDie(&outlined_instructions, instruction_to_outline,
                outlined_instruction);

    // Mark instruction_to_outline an output if it is used outside the
    // subcomputation or is the output of the original computation (i.e. used
    // externally).
    if (instruction_to_outline->user_count() == 0 ||
        IsUsedOutsideSubcomputation(*instruction_to_outline,
                                    instruction_set_to_outline)) {
      outputs.push_back(instruction_to_outline);
    }
  }

  if (outputs.size() != 1) {
    string error_message =
        "The subcomputation to outline has multiple outputs:\n";
    for (HloInstruction* output : outputs) {
      absl::StrAppend(&error_message, output->ToString(), "\n");
    }
    LOG(FATAL) << error_message;
  }
  HloInstruction* output = outputs[0];

  // Creates a call to the nested computation.
  HloComputation* nested_computation = AddEmbeddedComputation(
      builder.Build(FindOrDie(outlined_instructions, output)));
  HloInstruction* call = computation->AddInstruction(HloInstruction::CreateCall(
      output->shape(), arguments, nested_computation));

  VLOG(2) << "Outlining the following instructions";
  for (auto* instruction_to_outline : instructions_to_outline) {
    VLOG(2) << "  " << instruction_to_outline->ToString();
  }
  VLOG(2) << "as a call " << call->ToString();
  VLOG(2) << "to " << nested_computation->ToString();

  TF_CHECK_OK(output->ReplaceAllUsesWith(call));
  for (auto i = instructions_to_outline.rbegin();
       i != instructions_to_outline.rend(); ++i) {
    TF_CHECK_OK(computation->RemoveInstruction(*i));
  }

  return call;
}

int64 HloModule::instruction_count() const {
  int64 n = 0;
  for (const auto& computation : computations_) {
    n += computation->instruction_count();
  }
  return n;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder() const {
  // First determine all root computations by building a set of nonroot
  // computations (computations which are called by an instruction in the
  // module).
  absl::flat_hash_set<HloComputation*> nonroot_computations;
  for (auto& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      for (HloComputation* called_computation :
           instruction->called_computations()) {
        nonroot_computations.insert(called_computation);
      }
    }
  }

  // Keep track of computations which have already been added to the post
  // order. This prevents duplication as an embedded computation may be called
  // from two different root computations.
  absl::flat_hash_set<HloComputation*> added_computations;
  std::vector<HloComputation*> post_order;
  for (auto& computation : computations_) {
    if (!nonroot_computations.contains(computation.get())) {
      for (HloComputation* embedded_computation :
           computation->MakeEmbeddedComputationsList()) {
        if (!added_computations.contains(embedded_computation)) {
          post_order.push_back(embedded_computation);
          added_computations.insert(embedded_computation);
        }
      }
      // Root computations should only be encountered once.
      CHECK(!added_computations.contains(computation.get()));
      post_order.push_back(computation.get());
      added_computations.insert(computation.get());
    }
  }
  if (post_order.size() != computations_.size()) {
    for (HloComputation* computation : post_order) {
      LOG(ERROR) << "Post Order: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    for (auto& computation : computations_) {
      LOG(ERROR) << "Computations: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    LOG(FATAL) << "Mismatch computation count: post_order=" << post_order.size()
               << " computation_count=" << computations_.size();
  }
  return post_order;
}

namespace {
bool CompareComputationsByContent(HloComputation* a, HloComputation* b) {
  if (a->instruction_count() != b->instruction_count()) {
    return a->instruction_count() < b->instruction_count();
  }
  return a->ToString(HloPrintOptions::Fingerprint()) <
         b->ToString(HloPrintOptions::Fingerprint());
}
}  // anonymous namespace

std::vector<HloComputation*> HloModule::MakeComputationSorted() const {
  std::vector<HloComputation*> result;
  result.reserve(computations_.size());
  for (const auto& computation : computations_) {
    result.push_back(computation.get());
  }
  std::sort(result.begin(), result.end(), CompareComputationsByContent);
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputations() const {
  std::vector<HloComputation*> result;
  for (auto* c : computations()) {
    if (c->IsFusionComputation()) {
      continue;
    }
    result.push_back(c);
  }
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputationsSorted()
    const {
  auto result = MakeNonfusionComputations();
  std::sort(result.begin(), result.end(), CompareComputationsByContent);
  return result;
}

std::unique_ptr<HloModule> HloModule::Clone(std::string prefix, 
                                            HloComputation* computation) {
  auto program_shape = computation->ComputeProgramShape();
  auto config = HloModuleConfig(program_shape);
  config.set_debug_options(config_.debug_options());

  auto module = absl::make_unique<HloModule>("HloModule." + prefix, config);

  HloCloneContext context(module.get(), "");
  auto cloned_computation = computation->Clone("", &context);
  module->AddEntryComputation(std::move(cloned_computation));
  return module;
}

std::unique_ptr<HloModule> HloModule::Clone(const string& suffix) const {
  return Clone(config(), suffix);
}

std::unique_ptr<HloModule> HloModule::Clone(const HloModuleConfig& config,
                                            const string& suffix) const {
  VLOG(1) << "Cloning module :" << name_ << " --> " << suffix << "\n";
  auto module = absl::make_unique<HloModule>(
      absl::StrCat(name_, suffix.empty() ? "" : "-", suffix), config);

  HloCloneContext context(module.get(), suffix);
  auto cloned_computation = entry_computation_->Clone(suffix, &context);
  module->AddEntryComputation(std::move(cloned_computation));

  if (has_schedule() && schedule().Verify().ok()) {
    HloSchedule clone_schedule(module.get());
    for (HloComputation* computation : computations()) {
      if (schedule().is_computation_scheduled(computation)) {
        HloInstructionSequence& clone_sequence =
            clone_schedule.GetOrCreateSequence(
                context.GetComputation(computation));
        for (const HloInstruction* instruction :
             schedule().sequence(computation).instructions()) {
          clone_sequence.push_back(context.GetInstruction(instruction));
        }
      }
    }
    TF_CHECK_OK(module->set_schedule(std::move(clone_schedule)));
  }
  for (const auto& parameter_indices : CrossProgramPrefetches()) {
    const auto& parameter = parameter_indices.first;
    const auto& indices = parameter_indices.second;
    module->AddCrossProgramPrefetch(parameter, indices);
  }

  module->init_specs_map_ = init_specs_map_;
  module->variable_map_ = variable_map_;
  module->fetch_vars_list_ = fetch_vars_list_;
  module->num_dev_per_worker_ = num_dev_per_worker_;
  module->num_worker_ = num_worker_;
  return module;
}

void HloModule::StealDefCtx(HloModule* module,
    std::unordered_map<HloComputation*, HloComputation*>& replacements) {
  def_ctx_ = module->def_ctx_;
  def_ctx_->module_ = this;
  def_ctx_pool_ = std::move(module->def_ctx_pool_);
  def_computation_.clear();
  for (auto& it : module->DefComputation()) {
    HloModule::DefContext* def_ctx = it.first;
    CHECK(def_ctx);
    HloComputation* computation = it.second;
    CHECK(replacements.find(computation) != replacements.end());
    def_computation_[def_ctx] = replacements[computation];
  }

}

void HloModule::CopyMetaData(HloModule* module) {
  split_nums_ = module->split_nums_;
  share_dev_flags_ = module->share_dev_flags_;
  stage_split_ordinal_ = module->stage_split_ordinal_;
  placement_layout_ = module->placement_layout_;
  variable_map_ = module->variable_map_;
  init_specs_map_ = module->init_specs_map_;
  fetch_vars_list_ = module->fetch_vars_list_;
  num_dev_per_worker_ = module->num_dev_per_worker_;
  num_worker_ = module->num_worker_;
}

Status HloModule::RemoveUnusedComputations() {
  std::string suffix = "tmp";
  auto module = absl::make_unique<HloModule>(
      absl::StrCat(name_, suffix.empty() ? "" : "-", suffix), config());
  HloCloneContext context(module.get(), suffix);
  entry_computation_->Clone(suffix, &context);
  std::vector<HloComputation*> to_remove;
  for (auto computation : computations()) {
    auto found_computation = context.FindComputation(computation);
    if (found_computation == nullptr) {
      to_remove.push_back(computation);
    }
  }
  for (auto computation : to_remove) {
    TF_RETURN_IF_ERROR(RemoveEmbeddedComputation(computation));
  }
  return Status::OK();
}

HloComputation* HloModule::DeepCloneComputation(HloComputation* computation,
                                                HloCloneContext* context) {
  HloComputation* new_computation;
  if (context != nullptr) {
    if ((new_computation = context->FindComputation(computation)) != nullptr) {
      return new_computation;
    }
    new_computation =
        AddEmbeddedComputation(computation->Clone(context->suffix(), context));
  } else {
    new_computation = AddEmbeddedComputation(computation->Clone(""));
  }
  return new_computation;
}

uint64 HloModule::RandomNew64() const {
  tensorflow::mutex_lock l(rng_mutex_);
  return rng_();
}

HloComputation* HloModule::GetComputationWithName(absl::string_view name) {
  auto computations_in_module = computations();
  auto it = absl::c_find_if(
      computations_in_module,
      [&](HloComputation* computation) { return computation->name() == name; });
  return it == computations_in_module.end() ? nullptr : *it;
}

uint64 HloModule::Hash() const {
  uint64 result = entry_computation_layout().Hash();
  // Use MakeComputationSorted() instead of MakeComputationPostOrder()
  // because naming may affect the order of MakeComputationPostOrder() but not
  // MakeComputationSorted().
  for (auto* computation : MakeComputationSorted()) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      result = tensorflow::Hash64Combine(result, instruction->Hash());
    }
  }
  return result;
}

// Auto-Parallel Start
//void HloModule::set_variable_map(std::map<int, string>* variable_map) {
//  variable_map_ = variable_map;
//}

std::map<int, string>* HloModule::variable_map() {
  return &variable_map_;
}

const std::map<int, string>* HloModule::variable_map() const {
  return &variable_map_;
}

std::map<int, string>* HloModule::init_specs_map() {
  return &init_specs_map_;
}

std::vector<int>& HloModule::fetch_vars_list() {
  return fetch_vars_list_;
}

void HloModule::set_symbolic_map(std::map<int, std::set<int>>* symbolic_map) {
  symbolic_map_ = symbolic_map;
}

std::map<int, std::set<int>>* HloModule::symbolic_map() const {
  return symbolic_map_;
}
// Auto-Parallel End

/* static */ std::atomic<int> HloModule::next_unique_module_id_(0);

}  // namespace xla
