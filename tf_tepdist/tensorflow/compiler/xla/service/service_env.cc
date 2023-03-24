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

#include <fstream>
#include <string>

#include "include/json/json.h"

#include "tensorflow/compiler/xla/service/service_env.h"

#include "tensorflow/core/platform/str_util.h"

namespace xla {

ServiceEnv* ServiceEnv::env_ = nullptr;
tensorflow::mutex ServiceEnv::mu_;

void ServiceEnv::Init() {
  tensorflow::mutex_lock lock(ServiceEnv::mu_);
  if (env_ != nullptr) return;

  env_ = new ServiceEnv();
  env_->InitValues();
  env_->PrintAllEnvs();
}

ServiceEnv* ServiceEnv::Default() {
  if (env_ == nullptr) Init();
  return env_;
}

void ServiceEnv::InitValues() {
  string config_file;
  tensorflow::ReadStringFromEnvVar("CONFIG_FILE", DEFAULT_CONFIG_FILE,
                                   &config_file);

  LoadConfigFileSettings(config_file);
  LoadEnvVars();
}

#define JsonValue_float(v) v.asFloat()
#define JsonValue_int64(v) v.asInt64()
#define JsonValue_bool(v)                                                 \
  (v.isString() ? tensorflow::str_util::Uppercase(v.asString()) == "TRUE" \
                : v.asBool())
#define JsonValue_string(v) v.asString()

void ServiceEnv::LoadConfigFileSettings(const string config_file) {
  Json::Value json;
  Json::Reader reader;
  std::ifstream fstream(config_file);
  if (fstream.fail()) {
    if (config_file == DEFAULT_CONFIG_FILE) {
      LOG(WARNING) << "Default config file \"" << config_file
                   << "\" does not exist, using default values.";
      return;
    }
    LOG(FATAL) << "Config file \"" << config_file << "\" does not exist!\n";
    return;
  }
  VLOG(0) << "Load config file \"" << config_file << "\" ...";

  if (!reader.parse(fstream, json)) {
    LOG(FATAL) << "Config file loading failed, check the json format!\n";
    return;
  }

#define PARSE_JSON_CONFIG(name, tag, type, ...)                             \
  {                                                                         \
    Json::Value v = json.get(#name, Json::Value::null);                     \
    if (!v.isNull()) name##_.set(JsonValue_##type(v));                      \
    VLOG(1) << "Setting " << #name << " to " << to_string(name##_.value()); \
  }
  SERVICE_ENV_LIST(PARSE_JSON_CONFIG)
#undef PARSE_JSON_CONFIG
}

#define ReadValue_float tensorflow::ReadFloatFromEnvVar
#define ReadValue_int64 tensorflow::ReadInt64FromEnvVar
#define ReadValue_bool tensorflow::ReadBoolFromEnvVar
#define ReadValue_string tensorflow::ReadStringFromEnvVar

void ServiceEnv::LoadEnvVars() {
#define READ_VALUE_FROM_ENV(name, tag, type, ...)                          \
  {                                                                        \
    type v;                                                                \
    ReadValue_##type(tag, name##_.value(), &v);                            \
    if (name##_.been_set() && v != name##_.value()) {                      \
      LOG(WARNING) << "Config setting \"" << name##_.debug_string()        \
                   << "\" woule be OVERRIDDEN by environ var \""           \
                   << #tag << " = " << to_string(v) << "\" !";             \
    }                                                                      \
    name##_.set(v);                                                        \
    VLOG(1) << "Setting " << #tag << " to " << to_string(name##_.value()); \
  }
  SERVICE_ENV_LIST(READ_VALUE_FROM_ENV)
#undef READ_VALUE_FROM_ENV
}

void ServiceEnv::PrintAllEnvs() {
  string env_string = "Effective service envs: {";
#define ADD_ENV_STRING(name, tag, type, ...) \
  env_string = env_string + "\n  " + name##_.debug_string();
  SERVICE_ENV_LIST(ADD_ENV_STRING)
#undef ADD_ENV_STRING
  env_string += "\n}  [Effective service envs listed above]";
  VLOG(0) << env_string;
}

}  // namespace xla
