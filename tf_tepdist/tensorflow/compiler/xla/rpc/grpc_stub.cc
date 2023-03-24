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

#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace xla {

GRPCStub::~GRPCStub() = default;

Status MakeRPC(
    const std::function<::grpc::Status(::grpc::ClientContext*)>& rpc_method) {
  ::grpc::ClientContext context;
  ::grpc::Status s = rpc_method(&context);
  return tensorflow::FromGrpcStatus(s);
}

Status GRPCStub::TransferToClient(const TransferToClientRequest* request,
                                  TransferToClientResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToClient(context, *request, response);
  });
}

Status GRPCStub::TransferToServer(const TransferToServerRequest* request,
                                  TransferToServerResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToServer(context, *request, response);
  });
}

Status GRPCStub::TransferToServerHost(const TransferToServerRequest* request,
                                  TransferToServerResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToServerHost(context, *request, response);
  });
}

Status GRPCStub::DoRemoteSave(const DoRemoteSaveRequest* request,
                              DoRemoteSaveResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->DoRemoteSave(context, *request, response);
  });
}

Status GRPCStub::DoRemoteRestore(const DoRemoteRestoreRequest* request,
                                 DoRemoteRestoreResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->DoRemoteRestore(context, *request, response);
  });
}

Status GRPCStub::TransferToInfeed(const TransferToInfeedRequest* request,
                                  TransferToInfeedResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferToInfeed(context, *request, response);
  });
}

Status GRPCStub::TransferFromOutfeed(const TransferFromOutfeedRequest* request,
                                     TransferFromOutfeedResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferFromOutfeed(context, *request, response);
  });
}

Status GRPCStub::ResetDevice(const ResetDeviceRequest* request,
                             ResetDeviceResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ResetDevice(context, *request, response);
  });
}

Status GRPCStub::ExecutePlan(const ExecutePlanRequest* request,
                         ExecutePlanResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ExecutePlan(context, *request, response);
  });
}

Status GRPCStub::BuildExecutionPlan(const BuildExecutionPlanRequest* request,
                         BuildExecutionPlanResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->BuildExecutionPlan(context, *request, response);
  });
}

Status GRPCStub::TransferModuleAndDefCtx(
    const TransferModuleAndDefCtxRequest* request,
    TransferModuleAndDefCtxResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferModuleAndDefCtx(context, *request, response);
  });
}

Status GRPCStub::InitRemoteNcclComm(
    const InitRemoteNcclCommRequest* request,
    InitRemoteNcclCommResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->InitRemoteNcclComm(context, *request, response);
  });
}

Status GRPCStub::DispatchPlan(
    const DispatchPlanRequest* request,
    DispatchPlanResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->DispatchPlan(context, *request, response);
  });
}

Status GRPCStub::ExecuteRemotePlan(
    const ExecuteRemotePlanRequest* request,
    ExecuteRemotePlanResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ExecuteRemotePlan(context, *request, response);
  });
}

Status GRPCStub::TransferHostRawData(
    const TransferHostRawDataRequest* request,
    TransferHostRawDataResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferHostRawData(context, *request, response);
  });
}

Status GRPCStub::TransferVarArgMap(
    const TransferVarArgMapRequest* request,
    TransferVarArgMapResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->TransferVarArgMap(context, *request, response);
  });
}

Status GRPCStub::FetchResourceVars(const FetchResourceVarsRequest* request,
                                   FetchResourceVarsResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->FetchResourceVars(context, *request, response);
  });
}

Status GRPCStub::Compile(const CompileRequest* request,
                         CompileResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Compile(context, *request, response);
  });
}

Status GRPCStub::Execute(const ExecuteRequest* request,
                         ExecuteResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Execute(context, *request, response);
  });
}

Status GRPCStub::ExecuteGraphParallel(
    const ExecuteGraphParallelRequest* request,
    ExecuteParallelResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ExecuteGraphParallel(context, *request, response);
  });
}

Status TransferModuleAndDefCtx(
      const TransferModuleAndDefCtxRequest* request,
      TransferModuleAndDefCtxResponse* response) {
      //return Unimplemented("Client::TransferModuleAndDefCtx is not supported on client side");
      return Status::OK();
}

Status InitRemoteNcclComm(
      const InitRemoteNcclCommRequest* request,
      InitRemoteNcclCommResponse* response) {
      //return Unimplemented("Client::InitRemoteNcclComm is not supported on client side");
      return Status::OK();
}

Status DispatchPlan(
      const DispatchPlanRequest* request,
      DispatchPlanResponse* response) {
      //return Unimplemented("Client::DispatchPlan is not supported on client side");
      return Status::OK();
}

Status ExecuteRemotePlan(
      const ExecuteRemotePlanRequest* request,
      ExecuteRemotePlanResponse* response) {
      //return Unimplemented("Client::ExecuteRemotePlan is not supported on client side");
      return Status::OK();
}

Status TransferHostRawData(
      const TransferHostRawDataRequest* request,
      TransferHostRawDataResponse* response) {
      //return Unimplemented("Client::TransferHostRawData is not supported on client side");
      return Status::OK();
}

Status TransferVarArgMap(
      const TransferVarArgMapRequest* request,
      TransferVarArgMapResponse* response) {
      //return Unimplemented("Client::TransferVarArgMap is not supported on client side");
      return Status::OK();
}

Status GRPCStub::WaitForExecution(const WaitForExecutionRequest* request,
                                  WaitForExecutionResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->WaitForExecution(context, *request, response);
  });
}

Status GRPCStub::DeconstructTuple(const DeconstructTupleRequest* request,
                                  DeconstructTupleResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->DeconstructTuple(context, *request, response);
  });
}

Status GRPCStub::GetComputationGraphStats(
    const ComputationGraphStatsRequest* request,
    ComputationStatsResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetComputationGraphStats(context, *request, response);
  });
}

Status GRPCStub::GetShape(const GetShapeRequest* request,
                          GetShapeResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetShape(context, *request, response);
  });
}

Status GRPCStub::GetDeviceHandles(const GetDeviceHandlesRequest* request,
                                  GetDeviceHandlesResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->GetDeviceHandles(context, *request, response);
  });
}

Status GRPCStub::CreateChannelHandle(const CreateChannelHandleRequest* request,
                                     CreateChannelHandleResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->CreateChannelHandle(context, *request, response);
  });
}

Status GRPCStub::ComputeConstantGraph(
    const ComputeConstantGraphRequest* request,
    ComputeConstantResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->ComputeConstantGraph(context, *request, response);
  });
}

// Methods used by GlobalData.
Status GRPCStub::Unregister(const UnregisterRequest* request,
                            UnregisterResponse* response) {
  return MakeRPC([this, request, response](::grpc::ClientContext* context) {
    return grpc_stub_->Unregister(context, *request, response);
  });
}

}  // namespace xla
