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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_ENV_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_ENV_H_

#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

#define DEFAULT_CONFIG_FILE "config.json"

namespace xla {

// Overloading function for non-existing std::to_string(string)
namespace {
template <typename T>
string to_string(T value) {
  return std::to_string(value);
}

string to_string(string value) { return value; }

string to_string(char* value) { return string(value); }

string to_string(bool value) { return string(value ? "true" : "false"); }
}  // namespace

// When adding a new environ to the list, just add one line like below.
// The 4 keys means keyword, environ var name, basic C++ type and the default
// value. Type should be one of 'float', 'int64', 'bool' and 'string'.
#define SERVICE_ENV_LIST(V)                                         \
  V(debug, "DEBUG", bool, false)                                    \
  V(cluster_spec, "CLUSTER_SPEC", string, "")                       \
  V(rule_mode, "RULE_MODE", bool, false)                            \
  V(ignore_annotation, "IGNORE_ANNOTATION", bool, true)             \
  V(aux_affinity, "AUX_AFFINITY", bool, false)                      \
  V(cost_factor, "COST_FACTOR", float, 1.0)                         \
  V(fp16_comm, "FP16_COMM", bool, false)                            \
  V(num_gradients, "NUM_GRADIENTS", int64, -1)                      \
  V(forward_sub_graph_num, "FORWARD_SUB_GRAPH_NUM", int64, 1)       \
  V(var_mem_limit, "VAR_MEM_LIMIT", int64, 20)                      \
  V(opt_level, "OPT_LEVEL", int64, 3)                               \
  V(unbalanced_ratio, "UNBALANCED_RATIO", int64, 8)                 \
  V(num_micro_batches, "NUM_MICRO_BATCHES", int64, 0)               \
  V(num_stages, "NUM_STAGES", int64, 0)                             \
  V(micro_num_limit, "MICRO_NUM_LIMIT", int64, 0)                   \
  V(group_sched_count, "GROUP_SCHED_COUNT", int64, 2)               \
  V(pp_bandwidth, "PP_BANDWIDTH", int64, 16)                        \
  V(ilp_time_limit, "ILP_TIME_LIMIT", int64, 5)                     \
  V(ilp_num_threads, "ILP_NUM_THREADS", int64, 0)                   \
  V(buffer_save, "BUFFER_SAVE", bool, true)                         \
  V(early_ga, "EARLY_GA", bool, false)                              \
  V(async_recv, "ASYNC_RECV", bool, true)                           \
  V(async_send, "ASYNC_SEND", bool, true)                           \
  V(multi_reorder, "MULTI_REORDER", bool, true)                     \
  V(fake_input, "FAKE_INPUT", bool, false)                          \
  V(disable_buffer_alias, "DISABLE_BUFFER_ALIAS", bool, false)      \
  V(frontend, "FRONTEND", string, "")                               \
  V(dump_llvm_ptx, "DUMP_LLVM_PTX", bool, false)

class ServiceEnv {
  // NOTE(zycao): Only int64, float, bool and string are supposed to support.
  template <typename T>
  class EnvVariable {
   public:
    explicit EnvVariable(const string name, T default_v)
      : name_(name), value_(default_v) {}

    void set(T v) {
      value_ = v;
      been_set_ = true;
    }
    bool been_set() { return been_set_; }
    T value() { return value_; }
    string debug_string() { return name_ + " = " + to_string(value_); }

   private:
    const string name_;
    T value_;
    bool been_set_ = false;
  };

 public:
  static void Init();
#define GET_VALUE(name, tag, type, ...) \
  static type name() { return Default()->name##_.value(); };
  SERVICE_ENV_LIST(GET_VALUE)
#undef GET_VALUE

 private:
  void InitValues();
  void LoadConfigFileSettings(const string config_file);
  void LoadEnvVars();
  void PrintAllEnvs();

  static ServiceEnv* Default();
  static ServiceEnv* env_;
  static tensorflow::mutex mu_;

#define VALUE_CLASS_OBJECT(name, tag, type, v) \
  EnvVariable<type> name##_{#name, v};
  SERVICE_ENV_LIST(VALUE_CLASS_OBJECT)
#undef VALUE_CLASS_OBJECT
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_ENV_H_
