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

#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"

#include "grpcpp/support/status.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

Status CollectDataToResponse(const ProfileRequest& req,
                             ProfilerSession* profiler,
                             ProfileResponse* response) {
  profiler::XSpace xspace;
  TF_RETURN_IF_ERROR(profiler->CollectData(&xspace));
  TF_RETURN_IF_ERROR(
      profiler::ConvertXSpaceToProfileResponse(xspace, req, response));
  return Status::OK();
}

class ProfilerServiceImpl : public grpc::ProfilerService::Service {
 public:
  ::grpc::Status Monitor(::grpc::ServerContext* ctx, const MonitorRequest* req,
                         MonitorResponse* response) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "unimplemented.");
  }

  ::grpc::Status Profile(::grpc::ServerContext* ctx, const ProfileRequest* req,
                         ProfileResponse* response) override {
    VLOG(1) << "Received a profile request: " << req->DebugString();
    std::unique_ptr<ProfilerSession> profiler =
        ProfilerSession::Create(req->opts());
    Status status = profiler->Status();
    if (!status.ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            status.error_message());
    }

    Env* env = Env::Default();
    for (size_t i = 0; i < req->duration_ms(); ++i) {
      env->SleepForMicroseconds(EnvTime::kMillisToMicros);
      if (ctx->IsCancelled()) {
        return ::grpc::Status::CANCELLED;
      }
    }

    status = CollectDataToResponse(*req, profiler.get(), response);
    if (!status.ok()) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            status.error_message());
    }

    return ::grpc::Status::OK;
  }
};

}  // namespace

std::unique_ptr<grpc::ProfilerService::Service> CreateProfilerService() {
  return MakeUnique<ProfilerServiceImpl>();
}

}  // namespace tensorflow
