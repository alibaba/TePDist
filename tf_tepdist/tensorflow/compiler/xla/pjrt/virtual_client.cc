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

#include "tensorflow/compiler/xla/pjrt/virtual_client.h"

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/dapple_buffer_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include "tensorflow/compiler/xla/service/service_env.h"
#include "tensorflow/core/platform/env.h"

#include <limits>

namespace xla {

namespace {

void RecordOutputsUsage(
    std::vector<PjRtBuffer::ScopedHold>& device_buffers,
    LocalDeviceState* buffer_local_device,
    LocalDeviceState* stream_local_device,
    std::shared_ptr<BufferSequencingEvent> event,
    se::Stream* usage_stream) {
  int64 num_bufs = device_buffers.size();
  for (int64 i = 0; i < num_bufs; ++i) {
    auto& b = device_buffers.at(i);
    // prefer_to_retain_reference=false because when using the
    // ComputeSynchronized allocation model we don't need to retain a reference
    // to the device_buffer during execution because by definition the compute
    // stream is synchronized past the execution.
    if (b.type() == PjRtBuffer::ScopedHold::kUsage) {
      pjrt_utils::RecordUsage(std::move(b), buffer_local_device,
                              stream_local_device, event, usage_stream,
                              /*prefer_to_retain_reference=*/false);
    } else {
      CHECK(b.type() == PjRtBuffer::ScopedHold::kDonation);
      b.ConfirmDonation();
    }
  } 
}

// Converts a ScopedShapedBuffer returned from an execution into a
// PjRtBuffer.
std::unique_ptr<PjRtBuffer> OutputBufferHelper(
    ScopedShapedBuffer* result_buffer,
    std::shared_ptr<BufferSequencingEvent> definition_event, PjRtClient* client,
    Device* device, LocalDeviceState* local_device) {
  std::shared_ptr<TrackedDeviceBuffer> out_buffer =
      TrackedDeviceBuffer::FromScopedShapedBuffer(result_buffer,
                                                  {definition_event});
  auto py_buffer = absl::make_unique<PjRtBuffer>(
      result_buffer->on_host_shape(), result_buffer->on_device_shape(),
      std::move(out_buffer), client, device);
  pjrt_utils::RecordUsage(py_buffer->GetBufferWithUsageHold(),
                          local_device, local_device,
                          definition_event, local_device->compute_stream(),
                          /*prefer_to_retain_reference=*/false);
  return py_buffer;
}

StatusOr<Shape> GetShardedShape(const Shape& shape,
                                const OpSharding& sharding) {
  if (sharding.type() == OpSharding::TUPLE) {
    if (!shape.IsTuple()) {
      return InvalidArgument(
          "Got tuple OpSharding (%s) for non-tuple shape (%s)",
          sharding.DebugString(), shape.ToString());
    }
    if (sharding.tuple_shardings_size() != shape.tuple_shapes_size()) {
      return InvalidArgument(
          "Got mismatched OpSharding tuple size (%d) and shape tuple size (%d)."
          " (OpSharding: %s, shape: %s)",
          sharding.tuple_shardings_size(), shape.tuple_shapes_size(),
          sharding.DebugString(), shape.ToString());
    }
    std::vector<Shape> sharded_subshapes;
    for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          Shape sharded_subshape,
          GetShardedShape(shape.tuple_shapes(i), sharding.tuple_shardings(i)));
      sharded_subshapes.emplace_back(std::move(sharded_subshape));
    }
    return ShapeUtil::MakeTupleShape(sharded_subshapes);
  }
  TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                      HloSharding::FromProto(sharding));
  return hlo_sharding.TileShape(shape);
}

StatusOr<Shape> GetShardedShape(const HloInstructionProto& instr) {
  const Shape unsharded_shape(instr.shape());
  Shape sharded_shape;
  if (instr.has_sharding()) {
    TF_ASSIGN_OR_RETURN(sharded_shape,
                        GetShardedShape(unsharded_shape, instr.sharding()));
  } else {
    sharded_shape = unsharded_shape;
  }
  LayoutUtil::ClearLayout(&sharded_shape);
  return sharded_shape;
}

TaskNode* CreateTask(TaskDAG& task_graph, TaskNode::TaskType task_type,
    std::vector<TaskNode*> parents, Executable* task_exe,
    HloModule::DefContext& def_ctx, const std::vector<int64>& addr,
    const std::vector<bool>& share_dev_flags,
    int stage_split_ordinal) {
  auto* task = task_graph.new_task_node(task_type, addr, share_dev_flags,
                                        stage_split_ordinal,
                                        parents, def_ctx.name());
  task->set_executable(task_exe, &def_ctx);
  return task;
}

TaskNode* CreateTaskGroup(TaskDAG& task_graph, std::vector<TaskNode*> parents,
    Executable* task_exe, HloModule::DefContext& def_ctx,
    const std::vector<int64>& addr, const std::vector<bool>& share_dev_flags,
    int stage_split_ordinal) {
  VLOG(2) << "addr: " << addr.empty();
  auto* input = task_graph.new_task_node(TaskNode::TaskType::kInput,
                                         addr, share_dev_flags,
                                         stage_split_ordinal, parents,
                                         def_ctx.name());
  input->set_executable(task_exe, &def_ctx);
  auto* compute = task_graph.new_task_node(TaskNode::TaskType::kCompute,
                                           addr, share_dev_flags,
                                           stage_split_ordinal, {input},
                                           def_ctx.name());
  compute->set_executable(task_exe, &def_ctx);
  auto* output = task_graph.new_task_node(TaskNode::TaskType::kOutput,
                                          addr, share_dev_flags,
                                          stage_split_ordinal, {compute},
                                          def_ctx.name());
  output->set_executable(task_exe, &def_ctx);
  return output;
}

std::unordered_map<int, HloModule::DefContext*> InitializeIdDefMap(
    std::vector<std::pair<HloModule::DefContext*, HloModule*>>& def_hlo_pairs) {
  std::unordered_map<int, HloModule::DefContext*> id_def_map;
  for (auto& pair : def_hlo_pairs) {
    id_def_map[pair.first->def_id()] = pair.first;
  }

  return id_def_map;
}

}

VirtualClient::VirtualClient(std::string platform_name, int host_id,
      std::shared_ptr<PjRtClient> cpu_client,
      std::shared_ptr<PjRtClient> gpu_client)
      : platform_name_(platform_name), host_id_(host_id),
        cpu_client_(cpu_client), gpu_client_(gpu_client) {
  virtual_device_ = absl::make_unique<Device>(0/*id*/,
      "virtual"/*platform_name*/, "virtual"/*device_kind*/);
  whole_graph_launch_context_ =
      absl::make_unique<WholeGraphLaunchContext>(gpu_client_->device_count());
  coord_.Init();
}

void VirtualClient::InitializeExecutionState(
    TaskDAG* task_graph, const std::map<int, std::string>* variable_map,
    int worker_rank) {
  int local_dev_count = gpu_client_->device_count();
  exec_state_ = absl::make_unique<ExecutionState>(
      task_graph, variable_map, sharding_across_machine_, local_dev_count,
      worker_rank, worker_count_, ckpt_max_to_keep_);
  exec_state_->InitializeTaskContexts();
}

gpu::NcclContext* VirtualClient::GetOrCreateNcclContext() {
  if (!nccl_ctx_) {
    nccl_ctx_ = absl::make_unique<gpu::NcclContext>();
  }
  return nccl_ctx_.get();
}

std::unordered_set<const HloInstruction*> VirtualClient::CollectCommInstructions(
    const HloModule* module) {
  std::unordered_set<const HloInstruction*> comm_insts;
  const HloComputation* entry = module->entry_computation();
  for (const HloInstruction* inst : entry->instructions()) {
    if (inst->opcode() != HloOpcode::kDAPPLEAllReduce &&
        inst->opcode() != HloOpcode::kDAPPLEAllToAll &&
        inst->opcode() != HloOpcode::kDAPPLEAllGather) continue;
    comm_insts.insert(inst);
  }

  return comm_insts;
}

std::unordered_set<const HloInstruction*> VirtualClient::FindAllCollectiveCommInstructions(
    const TaskDAG* task_graph) {
  std::unordered_set<const HloInstruction*> comm_insts_set;
  for (const std::unique_ptr<TaskNode>& task_node : task_graph->task_nodes()) {
    switch (task_node->task_type()) {
      case TaskNode::TaskType::kCompute: {
        auto executable = task_node->executable();
        CHECK(executable);
        auto& module = executable->module();
        std::unordered_set<const HloInstruction*> m_comm_insts = \
           CollectCommInstructions(&module);
        comm_insts_set.insert(m_comm_insts.begin(), m_comm_insts.end()); 
        break;
      }

      default: {}
    }
  }

  return comm_insts_set;
}

Status VirtualClient::CreateNcclCliqueForCollectiveInstructions(
    const std::unordered_set<const HloInstruction*>& coll_insts,
    std::shared_ptr<CommDevManager> comm_dev_mgr) {
  std::set<int> split_ordinal_set;
  for (const HloInstruction* inst : coll_insts) {
    auto coll_inst = dynamic_cast<const HloDAPPLECollectiveInstruction*>(inst);
    VLOG(2) << coll_inst;
    CHECK(coll_inst);
    split_ordinal_set.insert(coll_inst->split_ordinal());
  }

  VLOG(2) << "comm_dev_mgr: " << comm_dev_mgr.get();
  CHECK(comm_dev_mgr);

  std::shared_ptr<DevGroupArray> dev_group_array;
  for (int split_ordinal : split_ordinal_set) {
    VLOG(1) << "split_ordinal: " << split_ordinal;
    dev_group_array = comm_dev_mgr->FindDevGroupArray(split_ordinal);
    CHECK(dev_group_array);
    for (const std::shared_ptr<DevGroup>& dev_group : dev_group_array->dev_groups_) {
      VLOG(1) << "dev_group: " << dev_group.get();
      CHECK(dev_group);
      const std::vector<int64>& global_devices = dev_group->global_dev_ids_;

      // 1. Create ncclUniqueId
      gpu::NcclUniqueGroupKey group_key(global_devices);
      TF_ASSIGN_OR_RETURN(ncclUniqueId* nccl_id,
                          nccl_ctx_->GetOrCreateNcclUniqueId(group_key));

      // 2. Create ncclComm on remote workers asynchoronously
      std::unordered_map<int/*worker id*/, std::vector<int64/*dev id*/>> gdev_wid_map;
      for (int64 gdev : global_devices) {
        int wid = comm_dev_mgr->worker_id(gdev);
        gdev_wid_map[wid].emplace_back(gdev);
        VLOG(0) << "global dev id: " << gdev << ", worker id: " << wid;
      }

      std::vector<std::thread> workers;
      for (int nid = 1; nid < comm_dev_mgr->num_workers(); ++nid) {
        VLOG(0) << "init nccl comm for worker " << nid;
        if (gdev_wid_map.find(nid) == gdev_wid_map.end()) {
          continue;
        }
        workers.push_back(std::thread([this, nccl_id, &global_devices, nid]() {
          coord_.InitRemoteNcclComm(*nccl_id, global_devices, nid);
        }));
      }

      if (gdev_wid_map.find(0) != gdev_wid_map.end()) {
        VLOG(0) << "init nccl comm for master";
        TF_RETURN_IF_ERROR(
            nccl_ctx_->MaybeCreateNcclComms(global_devices, 0/*master*/, comm_dev_mgr));
      }

       // Cleanup distributed workers
      std::for_each(workers.begin(), workers.end(), [](std::thread &t) { t.join(); });
    }
  }

  return Status::OK();
}

Status VirtualClient::CreateNcclCliqueForSendRecvTask(
    const TaskDAG* task_graph, std::shared_ptr<CommDevManager> comm_dev_mgr) {
  auto create_send_recv_comm = [this, comm_dev_mgr] (
      const std::vector<int64>& global_devices, const int wid) -> Status {
    gpu::NcclUniqueGroupKey group_key(global_devices);
    TF_ASSIGN_OR_RETURN(ncclUniqueId* nccl_id,
                        nccl_ctx_->GetNcclUniqueId(group_key));

    std::vector<int64> filtered;
    for (int64 g_dev : global_devices) {
      if (comm_dev_mgr->worker_id(g_dev) == wid) {
        filtered.push_back(g_dev);
      }
    }

    if (filtered.empty()) {
      return InternalError("Cannot find local devices in node %d from global_devices %s",
                            wid, gpu::GlobalDevicesToString(global_devices));
    }

    if (wid > 0) {
      coord_.InitRemoteNcclComm(
          *nccl_id, const_cast<std::vector<int64>&>(global_devices), wid);
    } else {
      TF_RETURN_IF_ERROR(
        this->nccl_ctx_->MaybeCreateNcclComms(global_devices, 0/*master*/, comm_dev_mgr));
    }

    return Status::OK();
  };


  // 1. Create All comms for Send/Recv Task
  for (const std::unique_ptr<TaskNode>& task_node : task_graph->task_nodes()) {
    switch (task_node->task_type()) {
      case TaskNode::TaskType::kSend: {
        auto& global_devs = task_node->send_recv_global_devs();

        if (task_node->parent() == task_graph->source()) break;
        TaskNode* recv_node = task_node->child();
        CHECK(recv_node->task_type() == TaskNode::TaskType::kRecv);
        const std::vector<int64>& global_devices = task_node->send_recv_global_devs();
        gpu::NcclUniqueGroupKey group_key(global_devices);
        TF_ASSIGN_OR_RETURN(ncclUniqueId* nccl_id,
                            nccl_ctx_->GetOrCreateNcclUniqueId(group_key));

        int send_wid = task_node->worker_id();
        int recv_wid = recv_node->worker_id();
        if (send_wid == recv_wid) {
          std::thread send_recv_comm_thread(
              create_send_recv_comm, task_node->send_recv_global_devs(), send_wid);
          send_recv_comm_thread.join();
        } else {
          // For send task
          std::thread send_comm_thread(
              create_send_recv_comm, task_node->send_recv_global_devs(), send_wid);
          // For recv task
          std::thread recv_comm_thread(
              create_send_recv_comm, recv_node->send_recv_global_devs(), recv_wid);
          send_comm_thread.join();
          recv_comm_thread.join();
        }
        break;
      }

      default: {}
    }
  }

  // 2. Create All comms for fetching result
  const TaskNode* sink = task_graph->sink();
  auto entry_def_ctx = sink->def_ctx();
  HloModule* module = task_graph->task_module(sink);
  int output_count = module->entry_computation()->root_instruction()->operand_count();
  int num_fetches = output_count - module->variable_map()->size();
  for (int i = 0; i < num_fetches; ++i) {
    bool need_recv = true;
    for (int slice_id : entry_def_ctx->output_idx_global_dev_map_[i]) {
      std::vector<int64> addr = comm_dev_mgr->LinearIdxToAddrBySplitNums(slice_id);
      int g_dev = comm_dev_mgr->global_dev_id(SplitId(addr,
                                                      task_graph->share_dev_flags(),
                                                      task_graph->stage_split_ordinal()));
      int worker_id = comm_dev_mgr->worker_id(g_dev);

      // fetch result exists on master, there is no need to fetch across worker
      if (worker_id == 0) {
        need_recv = false;
        break;
      }
    }

    if (!need_recv) continue;

    int slice_id = *(entry_def_ctx->output_idx_global_dev_map_[i].begin());
    std::vector<int64> addr = comm_dev_mgr->LinearIdxToAddrBySplitNums(slice_id);
    int g_dev = comm_dev_mgr->global_dev_id(SplitId(addr,
                                                    task_graph->share_dev_flags(),
                                                    task_graph->stage_split_ordinal()));
    int worker_id = comm_dev_mgr->worker_id(g_dev);
    std::vector<int64> global_devices = {g_dev, 0};
    gpu::NcclUniqueGroupKey group_key(global_devices);
    TF_ASSIGN_OR_RETURN(ncclUniqueId* nccl_id,
                        nccl_ctx_->GetOrCreateNcclUniqueId(group_key));
    std::thread send_comm_thread(create_send_recv_comm, global_devices, worker_id);
    std::thread recv_comm_thread(create_send_recv_comm, global_devices, 0/*master*/);
    send_comm_thread.join();
    recv_comm_thread.join();
  }

  return Status::OK();
}

Status VirtualClient::CreateAndInitNcclContext(LocalPlan* plan) {
  DistributedPlan* dist_plan = plan->distributed_plan();
  nccl_ctx_ = absl::make_unique<gpu::NcclContext>();

  TaskDAG* task_graph = dist_plan->task_graph();

  // 1. Collect all collective instructions
  std::unordered_set<const HloInstruction*> coll_insts = \
      FindAllCollectiveCommInstructions(task_graph); 

  std::shared_ptr<CommDevManager> comm_dev_mgr = task_graph->comm_dev_mgr();
  // 2. Create all comms for instructions
  TF_RETURN_IF_ERROR(CreateNcclCliqueForCollectiveInstructions(coll_insts, comm_dev_mgr));

  // 3. Create all comms for Send/Recv task
  TF_RETURN_IF_ERROR(CreateNcclCliqueForSendRecvTask(task_graph, comm_dev_mgr));

  return Status::OK();
}

gpu::NcclContext* VirtualClient::GetNcclContext() {
  return nccl_ctx_.get();
}

ExecutionState* VirtualClient::execution_state() {
  return exec_state_.get();
}

WholeGraphLaunchContext* VirtualClient::whole_graph_launch_context() {
  return whole_graph_launch_context_.get();
}

std::size_t LogicStageIdToStageId(std::size_t logic_stage_id, 
                                      std::size_t stage_num) {
  std::size_t mid_stage = (stage_num-1)/2;
  if (logic_stage_id <= mid_stage)
    return logic_stage_id;
  else
    return stage_num-logic_stage_id-1;
}

bool HasForward(const std::size_t logic_stage_id, const std::size_t stage_num) {
  return (logic_stage_id < ((stage_num+1)>>1));
}

bool HasBackward(const std::size_t logic_stage_id, const std::size_t stage_num) {
  return (logic_stage_id >= (stage_num>>1));
}

std::size_t BackwardStart(const std::size_t stage_num) {
  return (stage_num>>1);
}

// create local plan from distributed plan
std::unique_ptr<TaskDAG> VirtualClient::BuildTaskDAG(
                         std::shared_ptr<DistributedPlan> dist_plan) {
  auto dist_graph = dist_plan->task_graph();
  VLOG(0) << "BuildTaskDAG";
  TaskDAG* task_graph;
  std::shared_ptr<CommDevManager> comm_dev_mgr = dist_graph->comm_dev_mgr();
  CHECK(comm_dev_mgr);
  task_graph = new TaskDAG(dist_plan->top_def_ctx(),
                           dist_graph->split_nums(),
                           dist_graph->share_dev_flags(),
                           dist_graph->placement_layout(),
                           dist_graph->stage_split_ordinal(),
                           comm_dev_mgr->num_workers());
  task_graph->set_num_dev_per_worker(comm_dev_mgr->dev_num_per_worker());

  *task_graph->mutable_id_def_map() = dist_graph->id_def_map();
  for (auto& it : dist_graph->def_exe_map()) {
    task_graph->setup_def_exe(it.first, it.second);
  }

  // Build Local Task List
  TaskNode *source = nullptr, *sink = nullptr;
  auto rank = dist_plan->worker_rank();
  auto& tasks = dist_plan->WorkerTaskList(rank);
  std::unordered_map<TaskNode*, TaskNode*> clone_map;
  for (auto& task : tasks) {
    auto task_clone = task_graph->clone(task);
    
    clone_map[task] = task_clone;
    if (task_clone->task_type() == TaskNode::TaskType::kSplit) {
      CHECK(!source);
      source = task_clone;
    }

    if (task_clone->task_type() == TaskNode::TaskType::kMerge) {
      CHECK(!sink);
      sink = task_clone;
    }
  }

  if (!source) {
    source = task_graph->add_source();
    source->set_device_id(-1);
    source->set_executable(nullptr, task_graph->top_def_ctx());
    source->set_comm_dev_mgr(comm_dev_mgr);
  }

  if (!sink) {
    sink = task_graph->add_sink({});
    sink->set_device_id(-1);
    sink->set_executable(nullptr, task_graph->top_def_ctx());
    sink->set_comm_dev_mgr(comm_dev_mgr);
  }

  for (auto task : tasks) {
    for (auto parent : task->parents()) {
      if (!clone_map.count(parent)) continue;

      auto parent_clone = clone_map[parent];
      auto task_clone = clone_map[task];
      parent_clone->add_child(task_clone);
      task_clone->add_parent(parent_clone);
    }
  }

  // Fix source/sink edges
  CHECK(source && sink);
  for (auto& task_ : task_graph->task_nodes()) {
    auto task = task_.get();
    if (task == source || task == sink) continue;

    if (task->parents().empty()) {
      source->add_child(task);
      task->add_parent(source);
    }

    if (task->children().empty()) {
      task->add_child(sink);
      sink->add_parent(task);
    }
  }

  // Fix aux output edges
  auto dist_sink = dist_graph->sink();
  for (auto dist_sink_parent : dist_sink->parents()) {
    if (!clone_map.count(dist_sink_parent)) continue;

    auto task_clone = clone_map[dist_sink_parent];
    if(!sink->has_parent(task_clone)) {
      task_clone->add_child(sink);
      sink->add_parent(task_clone);
    }
  }

  for (auto& task : task_graph->task_nodes()) {
    if (task->task_type() != TaskNode::TaskType::kSend) continue;
    // barrier is owned by send
    auto& split_id = task->split_id();
    int local_device = comm_dev_mgr->local_dev_id(split_id);
    int saved_device = 0;
    CUDACHECK(cudaGetDevice(&saved_device));
    CUDACHECK(cudaSetDevice(local_device));
    std::shared_ptr<cudaEvent_t> barrier = task->CreateBarrier();
    std::shared_ptr<cudaEvent_t> release_barrier = task->CreateReleaseBarrier();
    task->set_record_barrier(false);

    CHECK(task->parents().size() == 1);
    auto* parent = task->parents().front();
    parent->set_barrier(barrier);
    parent->set_record_barrier(true);
    CUDACHECK(cudaSetDevice(saved_device));
  }

  bool async_recv = ServiceEnv::async_recv();
  for (auto& task : task_graph->task_nodes()) {
    if (!async_recv) break;
    if (task->task_type() != TaskNode::TaskType::kRecv) continue;
    // Exclude Split worker receive case.
    if (task->micro_id() == -1 || task->children().size() > 1) continue;
    CHECK(task->children().size() == 1);

    // barrier is create by recv but destroyed by its child.
    auto& split_id = task->split_id();
    int local_device = comm_dev_mgr->local_dev_id(split_id);
    int saved_device = 0;
    CUDACHECK(cudaGetDevice(&saved_device));
    CUDACHECK(cudaSetDevice(local_device));
    std::shared_ptr<cudaEvent_t> barrier = task->CreateBarrier();
    task->set_record_barrier(true);

    auto* child = task->children().front();
    child->set_barrier(barrier);
    child->set_record_barrier(false);

    CUDACHECK(cudaSetDevice(saved_device));
  }

  return std::unique_ptr<TaskDAG>(task_graph);
}

std::unique_ptr<TaskDAG> VirtualClient::CompileTaskDAG(
      std::vector<std::pair<HloModule::DefContext*,
                            std::unique_ptr<HloModule>>>& def_hlo_pairs,
      int num_nodes) {
  std::vector<std::pair<HloModule::DefContext*, HloModule*>> def_hlo_ptr_pairs;
  for (auto& def_hlo : def_hlo_pairs) {
    def_hlo_ptr_pairs.emplace_back(std::make_pair(def_hlo.first, def_hlo.second.get()));
  }

  return std::move(CompileTaskDAG(def_hlo_ptr_pairs, num_nodes));
}

std::unique_ptr<TaskDAG> VirtualClient::CompileTaskDAG(
    std::vector<std::pair<HloModule::DefContext*, HloModule*>>& def_hlo_pairs,
    int num_nodes) {
  auto module = def_hlo_pairs.back().second;
  auto device_count_per_worker = gpu_client()->device_count();
  auto required_dev_num_per_worker = module->total_dev_num() / num_nodes;

  VLOG(0) << "module->total_dev_num(): " << module->total_dev_num() << ", num_nodes: "
          << num_nodes << ", required_dev_num_per_worker: " << required_dev_num_per_worker
          << ", device_count_per_worker: " << device_count_per_worker;
  if (device_count_per_worker < required_dev_num_per_worker) {
    LOG(ERROR) << "device number is less than replica number.";
    return std::unique_ptr<TaskDAG>(nullptr);
  } else {
    device_count_per_worker = required_dev_num_per_worker;
  }


  // In master-slave distributed execution mode, we assume that each worker
  // holds the same number of devices.
  auto total_device_count = device_count_per_worker * num_nodes;


  auto top_def_ctx = def_hlo_pairs.back().first;

  auto num_splits = module->total_split_num();

  if (num_splits % num_nodes != 0) {
    LOG(ERROR) << "Replicas can not be divided equally between workers."
               << " Replicas is " << num_splits
               << ", while worker num is " << num_nodes;
    return std::unique_ptr<TaskDAG>(nullptr);
  }

  VLOG(0) << "CompileTaskDAG";
  auto* task_graph = new TaskDAG(def_hlo_pairs,
                                 module->split_nums(),
                                 module->share_dev_flags(),
                                 module->placement_layout(),
                                 module->stage_split_ordinal(),
                                 num_nodes);

  task_graph->set_num_dev_per_worker(device_count_per_worker);

  std::unordered_map<int, HloModule::DefContext*> id_def_map = InitializeIdDefMap(def_hlo_pairs);
  std::unordered_set<TaskNode*> outputs;

  auto* source = task_graph->add_source();
  source->set_executable(nullptr, top_def_ctx);
  source->set_name("SPLIT_" + std::to_string(source->node_id()));

  std::vector<TaskNode*> parents = {source};

  TaskNode* output = nullptr;
  // 1. Create TaskNode for leaf node in DefContext Hierarchy
  std::map<std::pair<int/*slice_id*/, int/*def_id*/>, TaskNode*> input_task_nodes_container;
  std::map<std::pair<int/*slice_id*/, int/*def_id*/>, TaskNode*> output_task_nodes_container;
  for (auto& pair : def_hlo_pairs) {
    HloModule::DefContext* def_ctx = pair.first;
    if (def_ctx->num_children()) continue;
    for (int64 slice_id : def_ctx->instance_slice_ids_) {
      std::vector<int64> addr = task_graph->comm_dev_mgr()->LinearIdxToAddrBySplitNums(slice_id);
      TaskNode* input = nullptr;
      TaskNode* output = nullptr;
      switch (def_ctx->def_type()) {
        case HloModule::DefContext::DefType::GA :
        case HloModule::DefContext::DefType::GA_SLICE: {
          input = CreateTask(*task_graph, TaskNode::TaskType::kGA, {},
                                nullptr, *def_ctx, addr,
                                module->share_dev_flags(),
                                module->stage_split_ordinal());
          output = input;
          break;
        }

        case HloModule::DefContext::DefType::GAINIT :
        case HloModule::DefContext::DefType::GAINIT_SLICE: {
          input = CreateTask(*task_graph, TaskNode::TaskType::kGAInit, {source},
                                nullptr, *def_ctx, addr,
                                module->share_dev_flags(),
                                module->stage_split_ordinal());
          output = input;
          break;
        }

        default : {
          output = CreateTaskGroup(*task_graph, {}, nullptr, *def_ctx, addr,
                                   module->share_dev_flags(),
                                   module->stage_split_ordinal());
          input = output->parent()->parent();
        }
      }
      input_task_nodes_container[std::make_pair(slice_id, def_ctx->def_id())] = input;
      output_task_nodes_container[std::make_pair(slice_id, def_ctx->def_id())] = output;
    }
  }

  // 2. Stitch TaskNodes according to input_def_map and input_arg_map
  std::set<std::pair<int/*src_id*/, int/*dst_id*/>> visited;
  for (auto& pair : input_task_nodes_container) {
    auto& key = pair.first;
    TaskNode* dst_n = pair.second;
    int slice_id = key.first;
    HloModule::DefContext* def_ctx = id_def_map[key.second];
    HloComputation* entry = def_ctx->module()->entry_computation();
    for (int p = 0; p < entry->num_parameters(); ++p) {
      auto src_output_or = def_ctx->get_src_output_from_input_def_map(p, slice_id);
      if (src_output_or.ok()) {
        auto src_output = src_output_or.ValueOrDie();
        int prev_slice_id = src_output.prev_slice_id;
        int def_id = src_output.def_id;
        std::pair<int, int> prev_key = {prev_slice_id, def_id};
        CHECK(output_task_nodes_container.find(prev_key) != output_task_nodes_container.end());
        TaskNode* src_n = output_task_nodes_container[prev_key];
        std::pair<int, int> edge_key = std::make_pair(src_n->node_id(), dst_n->node_id());
        if (visited.find(edge_key) != visited.end()) continue;
        src_n->add_child(dst_n);
        dst_n->add_parent(src_n);
        visited.insert(edge_key);
      } else if (def_ctx->input_arg_map_.find(p) != def_ctx->input_arg_map_.end()) {
        std::pair<int, int> edge_key = std::make_pair(source->node_id(), dst_n->node_id());
        if (visited.find(edge_key) != visited.end()) continue;
        source->add_child(dst_n);
        dst_n->add_parent(source);
        visited.insert(edge_key);
      }
    }

    // 3. Collect TaskDAG outputs
    HloInstruction* root = entry->root_instruction();
    for (int o = 0; o < root->operands().size(); ++o) {
      int parent_out_idx = o;
      HloModule::DefContext* curr_def_ctx = def_ctx;
      bool found = curr_def_ctx == top_def_ctx;
      while (curr_def_ctx != top_def_ctx) {
        if (curr_def_ctx->output_idx_map_.find(o) != curr_def_ctx->output_idx_map_.end()) {
          parent_out_idx = curr_def_ctx->output_idx_map_[parent_out_idx];
          curr_def_ctx = id_def_map[curr_def_ctx->parent_id()];
          found = true;
        } else {
          found = false;
          break;
        }
      }

      if (found) {
        TaskNode* n = output_task_nodes_container[key];
        outputs.insert(n);
      }
    }
  }

  std::vector<TaskNode*> outputs_vec(outputs.begin(), outputs.end());
  task_graph->add_sink(outputs_vec);
  auto sink = task_graph->sink();
  sink->set_executable(nullptr, top_def_ctx);
  sink->set_name("Merge_" + std::to_string(sink->node_id()));
  task_graph->Dump("dag.dot");
  return std::unique_ptr<TaskDAG>(task_graph);
}

// this function is called by slave worker
// restore distributed plan at worker side
std::shared_ptr<DistributedPlan> VirtualClient::BuildDistributedPlanRPC(
    int num_workers, int worker_rank, std::vector<ComputeTask>& compute_tasks,
    const std::vector<int>& split_nums, const std::vector<bool>& share_dev_flags,
    const std::vector<int>& placement_layout, int stage_split_ordinal) {
  // num_workers: slave worker num + 1(master)
  worker_count_ = num_workers;
  worker_rank_ = worker_rank;
  VLOG(0) << "worker_rank_: " << worker_rank_;
  VLOG(0) << "worker_count_: " << worker_count_;
  CHECK(worker_count_>0);

  whole_graph_launch_context_->set_worker_count(num_workers);

  bool sharding_across_machine = true;
  whole_graph_launch_context_->set_sharding_across_machine(sharding_across_machine);
  sharding_across_machine_ = sharding_across_machine;

  std::vector<std::pair<HloModule::DefContext*, HloModule*>> def_hlo_ptrs;
  for (auto& def_hlo : rpc_def_hlo_map_) {
    def_hlo_ptrs.emplace_back(std::make_pair(def_hlo.first, def_hlo.second.get()));
  }

  std::unique_ptr<TaskDAG> task_graph;
  // TODO(zycao): Pipeline mode needs to be fix. -- Jun 10th, 2022
  task_graph = std::make_unique<TaskDAG>(def_hlo_ptrs,
                                         split_nums, share_dev_flags,
                                         placement_layout, stage_split_ordinal,
                                         num_workers);
  // We need two rounds to build a complete TaskDAG.
  std::unordered_map<int/*node_id*/, TaskNode*> id_task;
  std::sort(compute_tasks.begin(),
            compute_tasks.end(),
            [](const ComputeTask& lhs, const ComputeTask& rhs) {
              return lhs.node_id() < rhs.node_id();
            });
  for (auto& compute_task : compute_tasks) {
    auto name = compute_task.name();
    VLOG(0) << "node name: " << name;
    auto task_type_id = compute_task.task_type();
    //auto stage_id = compute_task.stage_id();
    auto node_id = compute_task.node_id();
    auto worker_id = compute_task.worker_id();
    auto device_id = compute_task.device_id();
    auto def_id = compute_task.def_id();
    auto sched_idx_in_dev = compute_task.sched_idx_in_dev();
    std::vector<int64> addr;
    for (int i=0; i<compute_task.split_id_size(); ++i) {
      addr.emplace_back(compute_task.split_id(i));
    }
    SplitId split_id(addr, share_dev_flags, stage_split_ordinal);
    auto task = task_graph->new_task_node(name, task_type_id, split_id,
                                          worker_id, def_id);
    task->set_sched_idx_in_dev(sched_idx_in_dev);
    task->set_device_id(device_id);
    task->set_comm_with_lower_stage(compute_task.comm_with_lower_stage());
    task->set_across_machine(compute_task.across_machine());
    CHECK(node_id == task->node_id());
    id_task[node_id] = task;

    // Setup port map
    CHECK(task->mutable_port_map());
    auto& port_map = *task->mutable_port_map();
    for (auto& it : compute_task.port_map()) {
      port_map[it.first] = it.second;
    }

    for (int64 g_dev : compute_task.send_recv_global_devs()) {
      CHECK(task->mutable_send_recv_global_devs());
      task->mutable_send_recv_global_devs()->push_back(g_dev);
    }
  }

  // Setup parent/child relationships
  for (auto& compute_task : compute_tasks) {
    int num_parents = compute_task.parents_size();
    CHECK(id_task.count(compute_task.node_id()));
    auto task = id_task[compute_task.node_id()];
    for (int i = 0; i < num_parents; ++i) {
      auto parent_id = compute_task.parents(i);
      CHECK(id_task.count(parent_id));
      auto parent = id_task[parent_id];
      task->add_parent(parent);
    }
  }
  for (auto& compute_task : compute_tasks) {
    int num_children = compute_task.children_size();
    CHECK(id_task.count(compute_task.node_id()));
    auto task = id_task[compute_task.node_id()];
    for (int i = 0; i < num_children; ++i) {
      auto child_id = compute_task.children(i);
      CHECK(id_task.count(child_id));
      auto child = id_task[child_id];
      task->add_child(child);
    }
  }

  auto dist_plan = std::make_shared<DistributedPlan>(
      std::move(task_graph), rpc_def_hlo_map_, num_workers, worker_rank/*worker*/);

  dist_plan->task_graph()->Dump("dist_dag.dot");
  dist_plan->BuildWorkerTaskList(worker_rank);

  // Build HloModule
  ExecutableBuildOptions build_options;
  build_options.set_device_allocator(gpu_client()->allocator());
  if (build_options.device_ordinal() < 0) {
    auto& local_devices = gpu_client()->local_devices();
    build_options.set_device_ordinal(
        local_devices.front()->local_device_state()->device_ordinal());
  }

  auto& def_hlo_pairs = dist_plan->def_hlo_pairs();
  auto status_or_result = 
      gpu_client()->client()->BuildDefModules(std::move(def_hlo_pairs),
                                              build_options);
  auto def_exe_pairs = status_or_result.ConsumeValueOrDie();
  dist_plan->SetupDefExeMap(def_exe_pairs);
  dist_plan->ExeDecoration();
  dist_plan->ShowWorkerTaskList(worker_rank);
  return dist_plan;
}

std::shared_ptr<DistributedPlan> VirtualClient::BuildDistributedPlan(
      std::vector<std::pair<HloModule::DefContext*,
                            std::unique_ptr<HloModule>>>& def_hlo_pairs,
      int num_nodes, int rank) {
  worker_count_ = num_nodes;
  worker_rank_ = rank;
  VLOG(0) << "worker_rank_: " << worker_rank_;
  VLOG(0) << "worker_count_: " << worker_count_;
  VLOG(0) << "num_nodes: " << num_nodes << ", rank: " << rank;
  CHECK(worker_count_>0);

  whole_graph_launch_context_->set_worker_count(num_nodes);

  bool sharding_across_machine = true;
  whole_graph_launch_context_->set_sharding_across_machine(sharding_across_machine);
  sharding_across_machine_ = sharding_across_machine;

  std::unique_ptr<TaskDAG> task_graph = CompileTaskDAG(def_hlo_pairs, num_nodes);

  auto dist_plan = std::make_shared<DistributedPlan>(std::move(task_graph),
              def_hlo_pairs, num_nodes, rank/*master*/);
  VLOG(0) << "Worker Assignment.";
  dist_plan->WorkerAssignment();
  VLOG(0) << "Device Assignment.";
  dist_plan->DeviceAssignment();
  VLOG(0) << "Source Calibration.";
  dist_plan->SourceCalibration();
  VLOG(0) << "CrossDeviceCalibration.";
  dist_plan->CrossDeviceCalibration();
  dist_plan->task_graph()->Dump("worker-dag.dot");

  dist_plan->task_graph()->SetupInputSpecs(dist_plan->worker_count());

  for (auto i=0; i<dist_plan->worker_count(); ++i) {
    VLOG(2) << "build worker task list for worker " << i;
    dist_plan->BuildWorkerTaskList(i);
    dist_plan->ShowWorkerTaskList(i);
  }

  return dist_plan;
}

std::shared_ptr<LocalPlan>
VirtualClient::BuildLocalPlan(std::shared_ptr<DistributedPlan> dist_plan) {
  // mult-worker flow
  // both master and slave call this routine
  std::unique_ptr<TaskDAG> task_graph = BuildTaskDAG(dist_plan);

  auto num_gpu_devices = gpu_client()->device_count();
  int spmd_devices_needed = task_graph->num_dev_per_worker();
  int actually_used_devices = std::min(num_gpu_devices, spmd_devices_needed);
  VLOG(0) << "actually_used_devices: " << actually_used_devices;
  auto local_plan = std::make_shared<LocalPlan>(std::move(task_graph),
                                          dist_plan,
                                          actually_used_devices);
  auto rank = dist_plan->worker_rank();
  local_plan->task_graph()->Dump("local-dag-" + std::to_string(rank) + ".dot");

  local_plan->set_worker_rank(rank);
  local_plan->set_worker_count(dist_plan->worker_count());
  local_plan->RestoreLocalSchedule();
  local_plan->task_graph()->SetupInputSpecs(local_plan->worker_count());
  local_plan->MakeTaskGraphGCPlan();
  local_plan->task_graph()->ResolveCGLossOutputs();
  local_plan->task_graph()->BuildDominanceTree();
  if (ServiceEnv::debug())
    local_plan->ShowPerDeviceTaskList();
  if (ServiceEnv::buffer_save())
    local_plan->RecordBufferReuseInfo();

  return local_plan;
}

DAPPLEExecutable::DAPPLEExecutable(
                  bool parameter_is_tupled_arguments,
                  VirtualClient* virtual_client,
                  std::shared_ptr<LocalPlan> plan)
    : parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      virtual_client_(virtual_client),
      plan_(plan),
      parameters_that_must_be_donated_(absl::flat_hash_set<int>()) {
  fake_input_ = ServiceEnv::fake_input();
}

DAPPLEExecutable::DAPPLEExecutable(
                  bool parameter_is_tupled_arguments,
                  VirtualClient* virtual_client,
                  std::vector<std::unique_ptr<LocalExecutable>>& executables)
    : parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      virtual_client_(virtual_client),
      executables_(std::move(executables)) {
  int64 num_exe = executables_.size();
  if (num_exe <= 2) {
    entry_exe_ = executables_.at(0).get();
    if (num_exe > 1) {
      ga_exe_ = executables_.at(1).get();
    }
  } else {
    entry_exe_ = executables_.back().get();
  }

  auto executable = entry_exe_->executable();
  auto& module = executable->module();
  auto entry = module.entry_computation();
  auto root = entry->root_instruction();
  for (int64 i = 0; i < root->operand_count(); ++i) {
    auto instr = root->mutable_operand(i);
    CHECK(instr->opcode() != HloOpcode::kTuple);
    auto dim_to_slice = instr->dist_spec().get_dim_spec(0)->partition_dim();
    if (dim_to_slice >= 0) {
      CHECK(!sharded_result_buffers_.count(i));
      int64 local_dev_count = virtual_client_->gpu_client()->device_count();
      int64 device_count;
      if (virtual_client_->sharding_across_machine()) {
        device_count = local_dev_count * virtual_client_->worker_count();
      } else {
        device_count = local_dev_count;
      }

      auto& buffers = sharded_result_buffers_[i];
      buffers.resize(device_count);

      sharded_dim_to_slice_[i] = dim_to_slice;
      //VLOG(0) << "Sharded output instr->" << instr->ToString();
    }
  }

  fake_input_ = ServiceEnv::fake_input();
}

// NOTE(lansong):
// It is called from python side in standalone flow, currently
// standalone is not maintained.
///*static*/ StatusOr<std::unique_ptr<DAPPLEExecutable>>
//DAPPLEExecutable::Compile(const XlaComputation& computation,
//                          CompileOptions options,
//                          VirtualClient* virtual_client) {
//  if (options.parameter_is_tupled_arguments) {
//    return InvalidArgument("Disable parameter as tupled arguments in JAX "
//                           "before running this.");
//  }
//
//  auto client = virtual_client->gpu_client();
//  ExecutableBuildOptions& build_options = options.executable_build_options;
//  if (!build_options.device_allocator()) {
//    build_options.set_device_allocator(client->allocator());
//  }
//
//  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
//                      computation.GetProgramShape());
//  if (!options.argument_layouts) {
//    options.argument_layouts = program_shape.parameters();
//    for (Shape& shape : *options.argument_layouts) {
//      LayoutUtil::ClearLayout(&shape);
//    }
//  } else if (options.argument_layouts->size() !=
//             program_shape.parameters_size()) {
//    return InvalidArgument(
//        "CompileOptions specify %d argument layouts, but computation has %d "
//        "arguments",
//        options.argument_layouts->size(), program_shape.parameters_size());
//  }
//  std::vector<const Shape*> argument_layout_pointers;
//  argument_layout_pointers.reserve(options.argument_layouts->size());
//
//  // Assign a default layout based on `sharded_shape` to any array subshapes in
//  // `dst_shape` that are missing layouts.
//  auto assign_layouts = [client](const Shape& sharded_shape, Shape* dst_shape) {
//    return ShapeUtil::ForEachMutableSubshapeWithStatus(
//        dst_shape, [&](Shape* subshape, const ShapeIndex& idx) {
//          if (subshape->IsArray() && !subshape->has_layout()) {
//            CHECK(ShapeUtil::IndexIsValid(sharded_shape, idx));
//            const Shape& sharded_subshape =
//                ShapeUtil::GetSubshape(sharded_shape, idx);
//            LayoutUtil::SetToDefaultLayout(subshape);
//            TF_ASSIGN_OR_RETURN(Shape layout, client->client()
//                                                  ->backend()
//                                                  .transfer_manager()
//                                                  ->ChooseCompactLayoutForShape(
//                                                      sharded_subshape));
//            *subshape->mutable_layout() = layout.layout();
//          }
//          return Status::OK();
//        });
//  };
//  TF_ASSIGN_OR_RETURN(auto sharded_shapes,
//                      GetShardedProgramShapes(computation));
//
//  CHECK_EQ(sharded_shapes.first.size(), options.argument_layouts->size());
//  for (int i = 0; i < options.argument_layouts->size(); ++i) {
//    Shape* layout = &(*options.argument_layouts)[i];
//    argument_layout_pointers.push_back(layout);
//    TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.first[i], layout));
//  }
//
//  Shape result_layout;
//  if (build_options.result_layout()) {
//    result_layout = *build_options.result_layout();
//  } else {
//    result_layout = program_shape.result();
//    LayoutUtil::ClearLayout(&result_layout);
//  }
//  TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.second, &result_layout));
//  build_options.set_result_layout(result_layout);
//
//  if (build_options.device_ordinal() < 0) {
//    auto& local_devices = client->local_devices();
//    build_options.set_device_ordinal(
//        local_devices.front()->local_device_state()->device_ordinal());
//  }
//
//  CHECK(!options.parameter_is_tupled_arguments);
//  auto& coord = virtual_client->coordinator();
//  if (coord.initialized() &&
//      result_layout.IsTuple() && result_layout.tuple_shapes_size() > 1) {
//    auto status_or_result = client->client()->PreBuildDefModuleForTaskDAG(
//         computation, argument_layout_pointers, build_options);
//    auto def_hlo_pairs = status_or_result.ConsumeValueOrDie();
//    coord.TransferModuleAndDefCtx(def_hlo_pairs);
//
//    auto num_nodes = coord.num_workers()/*include master*/;
//    auto num_physical_stages = ReadNumPhysicalStageFromEnv(num_nodes);
//    std::shared_ptr<DistributedPlan> dist_plan =
//        virtual_client->BuildDistributedPlan(def_hlo_pairs,
//           num_nodes, 0/*master_rank*/, num_physical_stages);
//
//    // sharding across workers
//    auto& plan_def_hlo_pairs = dist_plan->def_hlo_pairs();
//    auto exe_status_or_result =
//        client->client()->BuildDefModules(std::move(plan_def_hlo_pairs),
//                                                build_options);
//    auto def_exe_pairs = exe_status_or_result.ConsumeValueOrDie();
//    dist_plan->SetupDefExeMap(def_exe_pairs);
//    dist_plan->ExeDecoration();
//
//    //dist_plan->CompileHloModules(build_options, *client->client());
//
//    // Run cost based scheduling
//    VLOG(0) << "Before dist_plan ScheduleTasks().";
//    dist_plan->ScheduleTasks();
//
//    // Async. Slave Compilation
//    std::thread dispatch_thread([&dist_plan, &coord]() {    
//      coord.DispatchPlan(dist_plan.get());
//    });
//
//    // build master's local plan
//    auto local_plan = virtual_client->BuildLocalPlan(dist_plan);
//    auto py_executable = absl::make_unique<DAPPLEExecutable>(
//        options.parameter_is_tupled_arguments, virtual_client,
//        std::move(local_plan));
//    TaskDAG* task_graph = dist_plan->task_graph();
//    py_executable->set_split_nums(task_graph->split_nums());
//    py_executable->set_share_dev_flags(task_graph->share_dev_flags());
//
//    dispatch_thread.join();
//
//    return py_executable;
//  }
//
//  auto def_hlo_pairs_or = client->client()->PreBuildDefModuleForTaskDAG(
//      computation, argument_layout_pointers, build_options);
//  auto def_hlo_pairs = def_hlo_pairs_or.ConsumeValueOrDie();
//  std::vector<std::pair<HloModule::DefContext*, HloModule*>> def_hlo_ptr_pairs;
//  for (auto& def_hlo : def_hlo_pairs) {
//    def_hlo_ptr_pairs.emplace_back(std::make_pair(def_hlo.first, def_hlo.second.get()));
//  }
//
//  std::shared_ptr<DistributedPlan> dist_plan =
//      virtual_client->BuildDistributedPlan(def_hlo_pairs,
//         1/*worker and master*/, 0/*master_rank*/,
//         1/*num_physical_stages*/);
//
//  // sharding across workers
//  //dist_plan->CompileHloModules(build_options, *client()->client());
//  auto& plan_def_hlo_pairs = dist_plan->def_hlo_pairs();
//  auto status_or_result =
//      client->client()->BuildDefModules(std::move(plan_def_hlo_pairs),
//                                              build_options);
//  auto def_exe_pairs = status_or_result.ConsumeValueOrDie();
//  dist_plan->SetupDefExeMap(def_exe_pairs);
//  dist_plan->ExeDecoration();
//  // def_hlo_pairs is empty now
//  VLOG(0) << "before BuildExecutionPlan";
//  std::shared_ptr<LocalPlan> local_plan =
//      virtual_client->BuildExecutionPlan(dist_plan,
//                  def_hlo_ptr_pairs, build_options, 1/*num_physical_stages*/);
//
//  auto py_executable = absl::make_unique<DAPPLEExecutable>(
//      options.parameter_is_tupled_arguments, virtual_client,
//      std::move(local_plan)/*local_executables*/);
//  TaskDAG* task_graph = dist_plan->task_graph();
//  py_executable->set_split_nums(task_graph->split_nums());
//  py_executable->set_share_dev_flags(task_graph->share_dev_flags());
//
//  // TODO: Enable InputOutputAliasing for GPU later on.
//  //TF_RETURN_IF_ERROR(py_executable->SetUpDonation(
//  //    client.get(), options.parameter_is_tupled_arguments));
//
//  return py_executable;
//}

Status DAPPLEExecutable::DoComputeTask(
    TaskContext* task_ctx, const ExecuteOptions& options,
    TaskNode* task, int64 local_dev_id, LocalPlan* plan) {
  auto executable = task->executable();
  CHECK(executable);
  auto& module = executable->module();
  VLOG(2) << "DoComputeTask->local_dev_id: " << local_dev_id
          << ", micro_id: " << task->micro_id()
          << ", stage_id: " << task->stage_id()
          << ", node_id: " << task->node_id()
          << ", module-def:" << module.name()
          << ", task->name(): " << task->name();

  auto& input_args = task_ctx->input_buffers();

  auto def_ctx = task->def_ctx();

  StatusOr<std::vector<DAPPLEBuffer*>> status_or_result;
  if (rpc_task_) {
    status_or_result = ExecuteRPCTask(input_args, options, local_dev_id, task);
  } else {
    status_or_result = ExecuteTask(input_args, options, local_dev_id, task);
  }
  // Populate output buffers
  *task_ctx->mutable_output_buffers() =
      std::move(status_or_result.ConsumeValueOrDie());

  return Status::OK();
}

void DAPPLEExecutable::DoHostRecv(
    TaskContext* task_ctx, TaskNode* task, LocalPlan* plan) {
  auto* output_buffers = task_ctx->mutable_output_buffers();
  auto* launch_ctx = virtual_client_->whole_graph_launch_context();
  auto* recv_args_recorder = launch_ctx->mutable_recv_args_recorder();
  auto& module = *(plan_->task_module(task));
  auto* variable_map = module.variable_map();
  bool cached_empty = input_data_ptrs_cached_.empty();
  for (auto& iter : task->port_map()) {
    auto idx_in_split_task = iter.first;
    auto idx_in_recv_task = iter.second;
    if (variable_map->count(idx_in_split_task)) {
      // variables case
      CHECK(recv_args_recorder->recv_args_storage_map[idx_in_split_task].first);
      int idx_in_var_buf = recv_args_recorder->recv_args_storage_map[idx_in_split_task].second;
      DAPPLEBufferHandle var_handle = recv_args_recorder->variable_buf_handles[idx_in_var_buf];
      auto var_dapple_buf_or = launch_ctx->Resolve(var_handle);
      CHECK(var_dapple_buf_or.ok());
      auto* var_dapple_buf = var_dapple_buf_or.ValueOrDie();
      output_buffers->emplace_back(var_dapple_buf);
    } else {
      // sample data case
      CHECK(!recv_args_recorder->recv_args_storage_map[idx_in_split_task].first);
      int idx_in_sample_buf = recv_args_recorder->recv_args_storage_map[idx_in_split_task].second;
      auto sample_dapple_buf_or = launch_ctx->ResolveSampleInputBuffer(idx_in_sample_buf);
      CHECK(sample_dapple_buf_or.ok());
      auto* sample_dapple_buf = sample_dapple_buf_or.ValueOrDie();
      output_buffers->emplace_back(sample_dapple_buf);
    }
  }

  if (fake_input_) {
    if (input_data_ptrs_cached_.empty()) {
      for (auto* dapple_buf : *output_buffers) {
        input_data_ptrs_cached_.emplace_back(dapple_buf);
      }
    } else {
      output_buffers->clear();
      for (auto* dapple_buf : input_data_ptrs_cached_) {
        output_buffers->emplace_back(dapple_buf);
      }
    }
  }
}

Status DAPPLEExecutable::DoRecvTask(
    TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
    LocalPlan* plan, se::Stream* stream) {
  auto task_def_ctx = task->def_ctx();
  CHECK(task_def_ctx);
  if (task_def_ctx->entry_def_ctx()) {
    CHECK(0) << "Here should not be reached! DoHostRecv task is already"
             << " performed before ExecuteTaskList() in ExecuteHostRecvTask()";
  }

  auto& parents = task->parents();
  CHECK(parents.size() == 1);
  auto* prev_nd = parents.front();
  CHECK(prev_nd->task_type() == TaskNode::TaskType::kSplit
        || prev_nd->task_type() == TaskNode::TaskType::kSend);
  // GPU receive date from another GPU either on the same worker or
  // on another worker
  return DoGPURecv(task_ctx, task, local_dev_id, plan, stream);
}

// Both cross workers and intra-worker
Status DAPPLEExecutable::DoGPURecv(TaskContext* task_ctx, TaskNode* task,
                                   int64 local_dev_id, LocalPlan* plan,
                                   se::Stream* stream) {
  auto& input_args = task_ctx->input_buffers();
  CHECK(task->task_type() == TaskNode::TaskType::kRecv);

  auto& module = *plan_->task_module(task);
  auto task_root = module.entry_computation()->root_instruction();

  auto& port_map = task->port_map();
  CHECK(!port_map.empty());
  int num_ports = port_map.size();
  CHECK(num_ports <= task_root->operand_count());

  VLOG(1) << "DoGPURecv task name: " << task->name()
          << " local_dev_id: " << local_dev_id
          << " port_map size:" << num_ports 
          << " micro_id: " << task->micro_id()
          << " stage_id: " << task->stage_id()
          << " node_id: " << task->node_id()
          << " module-def:" << module.name();

  const std::vector<int64> global_devices = task->send_recv_global_devs();
  std::map<const GlobalDeviceId, int64> global_dev_id_to_rank;
  for (int i = 0; i < global_devices.size(); ++i) {
    global_dev_id_to_rank.emplace(global_devices[i], i);
  }
  gpu::NcclContext* nccl_ctx = virtual_client_->GetNcclContext();
  TF_ASSIGN_OR_RETURN(gpu::NcclComm* nccl_comm_ptr,
                      nccl_ctx->GetNcclComm(global_devices, 1/*recv_rank*/));
  ncclComm_t nccl_comm = nccl_comm_ptr->comm(); 
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  // Build up output buffers
  auto* output_buffers = task_ctx->mutable_output_buffers();
  output_buffers->clear();
  output_buffers->reserve(port_map.size());
  for (auto& it : port_map) {
    auto dapple_buf = input_args[it.second];
    CHECK(dapple_buf);
    TF_RETURN_IF_ERROR(DoNcclSendRecv(
        dapple_buf, global_devices, stream, nccl_comm, false/*is_send*/));
    output_buffers->emplace_back(dapple_buf);
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  return Status::OK();
}

void DAPPLEExecutable::DoRPCSend(TaskNode* task, bool xfer_var,
                                 std::map<int, bool>& variable_arg) {
  // Build up input buffers
  auto* exec_state = virtual_client()->execution_state();
  auto parent = task->parent();
  auto* task_ctx = exec_state->get_task_context(task->node_id());
  auto* parent_task_ctx = exec_state->get_task_context(parent->node_id());
  auto& parent_output_buffers = parent_task_ctx->output_buffers();
  auto input_args = task_ctx->mutable_input_buffers();
  input_args->clear();
  auto& port_map = task->port_map();
  CHECK(!port_map.empty());
  input_args->resize(port_map.size());
  std::vector<bool> variables;
  std::vector<int> global_indices;
  variables.resize(port_map.size());
  global_indices.resize(port_map.size());
  for (auto& it : port_map) {
    auto arg_idx = it.first;
    if (xfer_var && variable_arg.at(arg_idx)) {
      (*input_args)[it.second] = parent_output_buffers[arg_idx];
      variables[it.second] = true;
      global_indices[it.second] = arg_idx;
    } else if (!variable_arg.at(arg_idx)) {
      (*input_args)[it.second] = parent_output_buffers[arg_idx];
      variables[it.second] = false;
      global_indices[it.second] = arg_idx;
    }
  }

  // Setup output buffers
  auto output_buffers = task_ctx->mutable_output_buffers();
  output_buffers->clear();
  output_buffers->assign(input_args->begin(), input_args->end());

  // Do CPU Host Args Transfer
  auto task_def_ctx = task->def_ctx();
  CHECK (task_def_ctx->entry_def_ctx());
  auto child = task->child();
  CHECK(child->task_type() == TaskNode::TaskType::kMerge);

  auto& coord = virtual_client_->coordinator();
  coord.TransferVarsAndData(*output_buffers, variables, global_indices);
}

Status DAPPLEExecutable::DoSendTask(
    TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
    LocalPlan* plan, se::Stream* stream) {
  const std::vector<int64> global_devices = task->send_recv_global_devs();
  std::map<const GlobalDeviceId, int64> global_dev_id_to_rank;
  for (int i = 0; i < global_devices.size(); ++i) {
    global_dev_id_to_rank.emplace(global_devices[i], i);
  }
  gpu::NcclContext* nccl_ctx = virtual_client_->GetNcclContext();
  TF_ASSIGN_OR_RETURN(gpu::NcclComm* nccl_comm_ptr,
                      nccl_ctx->GetNcclComm(global_devices, 0/*send_rank*/));
  ncclComm_t nccl_comm = nccl_comm_ptr->comm();
  CHECK(plan_.get() == plan);  // add this check to confirm the arg plan is not neccessary.

  auto def_ctx = task->def_ctx();
  VLOG(2) << "DoSendTask local_dev_id:" << local_dev_id
          << " micro_id: " << task->micro_id()
          << " stage_id: " << task->stage_id()
          << " node_id: " << task->node_id()
          << " def-ctx:" << def_ctx->name();

  auto& input_args = task_ctx->input_buffers();

  // Setup output buffers
  auto output_buffers = task_ctx->mutable_output_buffers();
  output_buffers->clear();
  output_buffers->resize(task->port_map().size());

  // Do CPU Host Args Transfer
  auto task_def_ctx = task->def_ctx();
  CHECK (!task_def_ctx->entry_def_ctx());

  // Do Intra worker NCCL Send
  auto* parent_task_ctx =
      virtual_client()->execution_state()->get_task_context(
          task->parent()->node_id());
  CHECK(output_buffers->size() <= parent_task_ctx->output_buffers().size());
  CHECK(task->device_id() >= 0);

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (auto& iter : task->port_map()) {
    int parent_out_idx = iter.first;
    int curr_out_idx = iter.second;
    auto* dapple_buf = input_args[curr_out_idx];
    CHECK(dapple_buf);
    TF_RETURN_IF_ERROR(DoNcclSendRecv(
        dapple_buf, global_devices, stream, nccl_comm, true/*is_send*/));
    output_buffers->emplace_back(dapple_buf);
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  return Status::OK();
}

void DAPPLEExecutable::DoARTask(
    TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
    LocalPlan* plan, se::Stream* stream) {
  // TODO @siyu.wsy and lansong.dls Rethinking whether AR task is necessary
  LOG(ERROR) << "fatal error! Don't support AR task node!";

  #if 0
  auto& module = task->executable()->module();

  VLOG(0) << "DoARTask->local_dev_id:" << local_dev_id
          << " micro_id: " << task->micro_id()
          << " stage_id: " << task->stage_id()
          << " node_id: " << task->node_id()
          << " module-def:" << module.name();


  auto ga_parent = task->parent();
  auto& ga_outputs = task_ctx->input_buffers();

  if (plan->num_stage_groups() > 0) {
    auto& pp_dp_group = plan->pp_dp_group_id();
    auto& pp_dp_self = plan->pp_dp_self_id();
    if (pp_dp_group.count(local_dev_id)) {
      auto rank = pp_dp_self[local_dev_id];
      CHECK(rank < plan->task_graph()->num_dp_insts());
      auto group_id = pp_dp_group[local_dev_id];
      auto& comm = stage_groups[group_id].nccl_comms[rank];
      NcclAllReduce(local_dev_id, stream, comm, ga_outputs);
    }
  } else {
    //int64 ga_outputs_size = ga_outputs.size();
    auto dp_group_id = plan->dp_group_id()[local_dev_id];
    auto dp_self_id = plan->dp_self_id()[local_dev_id];
    auto& nccl_comm = dp_groups[dp_group_id].nccl_comms[dp_self_id];
    NcclAllReduce(local_dev_id, stream, nccl_comm, ga_outputs);
  }

  // Set output buffers
  *task_ctx->mutable_output_buffers() = ga_outputs;
  #endif
}

void DAPPLEExecutable::DoGATask(
    TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
    LocalPlan* plan, const ExecuteOptions& options) {
  CHECK(task->task_type() == TaskNode::TaskType::kGA);

  auto& module = task->executable()->module();
  int64 num_ga_params = module.entry_computation()->num_parameters();
  CHECK(!(num_ga_params & 0x1));

  VLOG(2) << "DoGATask->local_dev_id:" << local_dev_id
          << " micro_id: " << task->micro_id()
          << " stage_id: " << task->stage_id()
          << " node_id: " << task->node_id()
          << " module-def:" << module.name();

  auto ga_parent = task->parent(0);
  CHECK(ga_parent->task_type() == TaskNode::TaskType::kGAInit ||
        ga_parent->task_type() == TaskNode::TaskType::kGA);
  auto& ga_input_buffers = task_ctx->input_buffers();
  CHECK(ga_input_buffers.size() == num_ga_params);

  StatusOr<std::vector<DAPPLEBuffer*>> status_or_result;
  if (rpc_task_) {
    status_or_result = ExecuteRPCTask(ga_input_buffers, options,
                                      local_dev_id, task);
  } else {
    status_or_result = ExecuteTask(ga_input_buffers, options,
                                      local_dev_id, task);
  }
  *task_ctx->mutable_output_buffers() =
      std::move(status_or_result.ConsumeValueOrDie());
}

void DAPPLEExecutable::DoOutputTask(
    TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
    LocalPlan* plan) {
  CHECK(task->task_type() == TaskNode::TaskType::kOutput);

  auto& module = task->executable()->module();
  VLOG(2) << "DoOutputTask->local_dev_id:" << local_dev_id
          << " micro_id: " << task->micro_id()
          << " stage_id: " << task->stage_id()
          << " node_id: " << task->node_id()
          << " module-def:" << module.name();

  auto& input_args = task_ctx->input_buffers();

  auto executable = task->executable();
  CHECK(executable);
  auto entry = module.entry_computation();
  auto root = entry->root_instruction();

  int64 bound = 1;
  if (root->opcode() == HloOpcode::kTuple) {
    bound = root->operand_count();
  }

  if (!input_args.empty()) {
    CHECK(bound == input_args.size());
    *task_ctx->mutable_output_buffers() = input_args;
  }
}

void DAPPLEExecutable::DoGAInitTask(
  TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
  LocalPlan* plan) {
  auto task_graph = plan_->task_graph();
  auto& module = *(plan->task_module(task));
  auto entry = module.entry_computation();

  VLOG(2) << "DoGAInitTask task name: " << task->name()
          << " local_dev_id: " << local_dev_id
          << " micro_id: " << task->micro_id()
          << " stage_id: " << task->stage_id()
          << " node_id: " << task->node_id()
          << " module-def:" << module.name();

  Device* device = pjrt_utils::LookupDevice(*virtual_client_->gpu_client(), local_dev_id);
  std::vector<std::unique_ptr<PjRtBuffer>> mutable_ga_buffers_ptrs;
  CHECK(InitializeGABuffers(mutable_ga_buffers_ptrs, device, module).ok());

  int64 global_dev_id = task->comm_dev_mgr()->global_dev_id(task->split_id());
  auto output_buffers = task_ctx->mutable_output_buffers();
  output_buffers->clear();
  int64 num_ga_buffers = mutable_ga_buffers_ptrs.size();
  output_buffers->reserve(num_ga_buffers);
  for (int64 i = 0; i < num_ga_buffers; ++i) {
    auto& param_shape = entry->parameter_instruction(i)->shape();
    auto pjrt_buf = std::move(mutable_ga_buffers_ptrs.at(i));
    auto dapple_buf = task_ctx->ResolveOutput(i).ConsumeValueOrDie();
    dapple_buf->set_gpu_buffer(std::move(pjrt_buf), global_dev_id);

    output_buffers->emplace_back(dapple_buf);
  }
}

void DAPPLEExecutable::DoInputTask(
    TaskContext* task_ctx, TaskNode* task, int64 local_dev_id,
    LocalPlan* plan) {
  CHECK(task->task_type() == TaskNode::TaskType::kInput);
  auto micro_id = task->micro_id();
  auto task_graph = plan->task_graph();

  auto executable = task->executable();
  CHECK(executable);
  auto& module = executable->module();
  auto task_def_ctx = task->def_ctx();
  CHECK(task_def_ctx);

  std::unordered_set<int> input_vars(task_def_ctx->sharded_args_.begin(),
                                     task_def_ctx->sharded_args_.end());

  auto entry = module.entry_computation();
  auto num_arg_handles = entry->num_parameters();
  auto& input_specs = task->input_specs();
  int64 input_specs_size = input_specs.size();
  CHECK(input_specs.size() == num_arg_handles);

  auto& input_args = task_ctx->input_buffers();

  VLOG(2) << "DoInputTask-> node name:" << task->name()
          << " input_specs_size:" << input_specs_size
          << " worker_id: " << task->worker_id()
          << " local_dev_id:" << local_dev_id
          << " micro_id: " << task->micro_id()
          << " stage_id: " << task->stage_id()
          << " node_id: " << task->node_id()
          << " module-def:" << module.name();

  int local_dev_count = virtual_client()->gpu_client()->device_count();
  int global_device_count = local_dev_count;
  if (virtual_client()->sharding_across_machine()) {
    global_device_count = local_dev_count * virtual_client()->worker_count();
    VLOG(2) << "worker rank: " << plan->worker_rank()
            << ", global dev count: " << global_device_count;
  }

  std::shared_ptr<CommDevManager> comm_dev_mgr = task->comm_dev_mgr();
  CHECK(comm_dev_mgr);
  int64 global_dev_id = comm_dev_mgr->global_dev_id(task->split_id());

  // Perform H2D copy if necessary
  for (int64 i = 0; i < input_specs_size; ++i) {
    HloInstruction* param = entry->parameter_instruction(i);
    const Shape& param_shape = param->shape();
    DAPPLEBuffer* arg = input_args.at(i);
    const Shape& full_shape = arg->on_host_shape();
    bool sample_input = !input_vars.count(i);

    if (!arg->in_gpu(global_dev_id) || sample_input) {
      if (arg->in_cpu()) {

        if (arg->in_gpu(global_dev_id) && fake_input_) continue; 
        CHECK(DAPPLEBufferUtils::H2D(arg, param_shape, local_dev_id,
              global_dev_id, param->dist_spec(), task->split_id().ids_,
              virtual_client()->gpu_client().get()).ok());
      } else {
        CHECK(!sample_input);
        PjRtBuffer* gpu_buf = arg->gpu_buffer();

        // Partition a GPU buffer
        auto literal = gpu_buf->ToLiteral().ConsumeValueOrDie();
        auto raw = literal->untyped_data();

        CHECK(DAPPLEBufferUtils::H2D(arg, param_shape, local_dev_id,
              global_dev_id, param->dist_spec(), task->split_id().ids_,
              virtual_client()->gpu_client().get()).ok());
      }
    }
  }

  // Setup output buffers for the kInput task
  auto* output_buffers = task_ctx->mutable_output_buffers();
  output_buffers->clear();
  output_buffers->assign(input_args.begin(), input_args.end());
}

void DAPPLEExecutable::ExecuteTaskList(
    int64 local_dev_id, int64 device_count_needed, ExecutionState* exec_state,
    const ExecuteOptions& options, LocalPlan* plan) {
  CHECK (plan->has_work(local_dev_id));

  CHECK(plan_.get() == plan);  // add this check to confirm the arg plan is not neccessary.

  auto nccl_ctx = virtual_client_->GetNcclContext();
  VLOG(2) << "nccl_ctx: " << nccl_ctx;

  bool profiling = ServiceEnv::debug();

  auto gpu_client = virtual_client_->gpu_client();
  Device* device = pjrt_utils::LookupDevice(*gpu_client, local_dev_id);
  CHECK_EQ(device->host_id(), gpu_client->host_id());
  int device_ordinal = device->local_device_state()->device_ordinal();
  CHECK(device_ordinal == local_dev_id);
  int saved_device = 0;
  CUDACHECK(cudaGetDevice(&saved_device));
  CUDACHECK(cudaSetDevice(local_dev_id));

  LocalDeviceState* device_state = device->local_device_state();
  se::Stream* stream = device_state->compute_stream();

  auto& task_list = plan->task_list(local_dev_id);
  uint64 start_us, end_us;
  for (auto task : task_list) {
    if (profiling) {
      VLOG(0) << "task name: " << task->name()
              << ", worker id: " << task->worker_id()
              << ", dev id: " << task->device_id()
              << ", local dev id: " << local_dev_id;
    }
    exec_state->PrepareInputsForTask(task);
    auto* task_ctx = exec_state->get_task_context(task->node_id());
    // Clear output buffers
    task_ctx->mutable_output_buffers()->clear();
    auto type = task->task_type();
    if (profiling) {
      start_us = tensorflow::Env::Default()->NowMicros();
    }
    switch (type) {
      case TaskNode::TaskType::kInput: {
        if (task->barrier() && !task->record_barrier()) {
          cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
                              stream->implementation()->GpuStreamMemberHack());
          CUDACHECK(cudaStreamWaitEvent(*cu_stream, *(task->barrier()), 0));
        }
        DoInputTask(task_ctx, task, local_dev_id, plan);
        break;
      }
 
      case TaskNode::TaskType::kGAInit: {
        DoGAInitTask(task_ctx, task, local_dev_id, plan);
        break;
      }

      case TaskNode::TaskType::kGA: {
        DoGATask(task_ctx, task, local_dev_id, plan, options);
        // Try GC.
        exec_state->CleanupPjRtBuffersTriggeredBy(task);
        break;
      }

      case TaskNode::TaskType::kAR: {
        DoARTask(task_ctx, task, local_dev_id, plan, stream);
        break;
      }

      case TaskNode::TaskType::kCompute: {
        CHECK(DoComputeTask(task_ctx, options, task, local_dev_id, plan).ok());
        break;
      }

      case TaskNode::TaskType::kOutput: {
        DoOutputTask(task_ctx, task, local_dev_id, plan);
        
        cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
                              stream->implementation()->GpuStreamMemberHack());
        if (task->barrier()) {
          CHECK(task->record_barrier());
          CUDACHECK(cudaEventRecord(*(task->barrier()), *cu_stream));
        }
        if (task->buffer_barrier()) { //for buffer reused
          CHECK(task->buffer_record_barrier());
          CUDACHECK(cudaEventRecord(*(task->buffer_barrier()), *cu_stream));
        }
        // Try GC.
        exec_state->CleanupPjRtBuffersTriggeredBy(task);
        break;
      }

      case TaskNode::TaskType::kSend: {
        se::Stream* send_stream = device_state->send_stream();
        cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
            send_stream->implementation()->GpuStreamMemberHack());
 
        if (task->barrier()) {
          CUDACHECK(cudaStreamWaitEvent(*cu_stream, *(task->barrier()), 0));
        }
        CHECK(DoSendTask(task_ctx, task, local_dev_id, plan, send_stream).ok());
        CUDACHECK(cudaEventRecord(*(task->release_barrier()), *cu_stream));
        // Temporaly comment this before callback threads impelement
        device_state->ThenExecuteOnCallbackThread(
            send_stream,
            [exec_state, task, device_state, cu_stream]() {
              CUDACHECK(cudaStreamWaitEvent(*cu_stream, *(task->release_barrier()), 0));
              exec_state->CleanupPjRtBuffersTriggeredBy(task);
            });
        break;
      }

      case TaskNode::TaskType::kRecv: {
        if (task->barrier()) { // ASYNC_RECV was set, using recv steram.
          se::Stream* recv_stream = device_state->recv_stream();
          
          cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
              recv_stream->implementation()->GpuStreamMemberHack());
          if(task->buffer_wait_barrier()) { // wait for buffer released by pre recv task that use the same buffer 
            CUDACHECK(cudaStreamWaitEvent(*cu_stream, *(task->buffer_wait_barrier()), 0));
          }
          CHECK(DoRecvTask(task_ctx, task, local_dev_id, plan, recv_stream).ok());
          CHECK(task->record_barrier());
          CUDACHECK(cudaEventRecord(*(task->barrier()), *cu_stream));
        } else {
          // Original sync behavior.
          CHECK(DoRecvTask(task_ctx, task, local_dev_id, plan, stream).ok());
        }
        break;
      }
 
      default: CHECK(0 && "Unknown task type");
    }
    // Clear input buffers
    task_ctx->mutable_input_buffers()->clear();
    if (profiling) {
      end_us = tensorflow::Env::Default()->NowMicros();
      float duration_ms = (end_us - start_us) / 1000.0f;
      VLOG(0) << "\n  " << task->name() << ", worker id: " << task->worker_id()
              << ", dev id: " << task->device_id() << ", local dev id: " << local_dev_id
              << ": duration = " << duration_ms << " ms";
    }
  }

  CUDACHECK(cudaSetDevice(saved_device));
}

StatusOr<std::vector<DAPPLEBuffer*>> DAPPLEExecutable::ExecuteTask(
    absl::Span<DAPPLEBuffer* const> arguments,
    const ExecuteOptions& options, int64 local_dev_id, TaskNode* task,
    std::vector<std::unique_ptr<PjRtBuffer>>* ga_buffers/* = nullptr*/) const {
  auto client = virtual_client_->gpu_client().get();
  auto executable = task->executable();
  auto& module = executable->module();

  std::vector<PjRtBuffer::ScopedHold> device_buffers;
  StatusOr<ScopedShapedBuffer> result_buffer_or_status =
      EnqueueExecution(arguments, options, local_dev_id,
                       task, ga_buffers, &device_buffers);

  if (!result_buffer_or_status.ok()) {
    LOG(ERROR) << "Execution failed: "
               << result_buffer_or_status.status();
    return result_buffer_or_status.status();
  }
  ScopedShapedBuffer result_buffer =
      result_buffer_or_status.ConsumeValueOrDie();

  Device* device = pjrt_utils::LookupDevice(*client, local_dev_id);
  LocalDeviceState* device_state = &client->device_state(local_dev_id);
  se::Stream* stream = device_state->compute_stream();
  StatusOr<EventPool::Handle> event_or =
      device_state->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    pjrt_utils::StallStreamOnError(device_state, stream);
    for (PjRtBuffer::ScopedHold& b : device_buffers) {
      if (b.type() == PjRtBuffer::ScopedHold::kDonation) {
        // Even though there was an error we need to call ConfirmDonation, which
        // renders b invalid, since the computation has been enqueued and b has
        // been donated.
        b.ConfirmDonation();
      }
    }
    return event_or.status();
  }

  auto task_def_ctx = task->def_ctx();
  std::vector<DAPPLEBuffer*> outputs;
  auto definition_event = std::make_shared<BufferSequencingEvent>();
  definition_event->SetSequencingEvent(event_or.ConsumeValueOrDie(), stream);
  int local_dev_count = virtual_client()->gpu_client()->device_count();
  int device_count;
  int dev_id;
  if (virtual_client()->sharding_across_machine()) {
    device_count = local_dev_count * virtual_client()->worker_count();
    dev_id = plan()->worker_rank() * local_dev_count + local_dev_id;
    VLOG(0) << "worker rank: " << plan()->worker_rank() << ", dev id: " << dev_id
            << "dev count: " << device_count;
  } else {
    device_count = local_dev_count;
    dev_id = local_dev_id;
  }

  if (options.untuple_result && result_buffer.on_host_shape().IsTuple()) {
    auto entry = module.entry_computation();
    auto entry_root = entry->root_instruction();
    int64 num_outputs = entry_root->operand_count();
    int64 tuple_count = num_outputs;
    outputs.reserve(tuple_count);
 
    auto& out_dim_to_slice = task_def_ctx->output_dim_to_slice_;
    auto num_splits = module.total_split_num();
    for (int64 i = 0; i < tuple_count; ++i) {
      ScopedShapedBuffer tuple_buffer = result_buffer.TakeSubTree({i});
      auto pjrt_buf = OutputBufferHelper(&tuple_buffer, definition_event,
                                          client, device, device_state);
      if (!out_dim_to_slice.count(i)) {
        DAPPLEBuffer* dapple_buf = DAPPLEBuffer::CreateDAPPLEBuffer(
            virtual_client()->virtual_device(),
            tuple_buffer.on_host_shape(), tuple_buffer.on_device_shape(),
            std::move(pjrt_buf), device_count, dev_id);
        outputs.emplace_back(dapple_buf);
      } else {
        auto dim_to_slice = out_dim_to_slice.at(i);
        auto& shape = tuple_buffer.on_host_shape();
        int64 rank = shape.rank();
        CHECK(dim_to_slice >= 0 && dim_to_slice < rank && num_splits > 1);
        int64 new_dim_size = shape.dimensions(dim_to_slice) * num_splits;

        std::vector<int64> new_dims;
        new_dims.reserve(rank);
        for (int r = 0; r < rank; ++r) {
          if (r == dim_to_slice) {
            new_dims.emplace_back(new_dim_size);
          } else {
            new_dims.emplace_back(shape.dimensions(r));
          }
        }
        CHECK(shape.element_type() == F32);
        auto new_shape = ShapeUtil::MakeShapeWithLayout(
            shape.element_type(), new_dims,
            LayoutUtil::MinorToMajor(shape));
        
        auto dapple_buf = DAPPLEBuffer::CreateDAPPLEBuffer(
            virtual_client()->virtual_device(), new_shape, new_shape,
            std::move(pjrt_buf), device_count, dev_id);
        outputs.emplace_back(dapple_buf);
      }
    }
    CHECK (device_state->allocation_model() != LocalDeviceState::kSynchronous);
  } else {
    auto pjrt_buf = OutputBufferHelper(&result_buffer, definition_event,
                                        client, device, device_state);
    DAPPLEBuffer* dapple_buf = DAPPLEBuffer::CreateDAPPLEBuffer(
        virtual_client()->virtual_device(),
        result_buffer.on_host_shape(), result_buffer.on_device_shape(),
        std::move(pjrt_buf), device_count, dev_id);
    outputs.push_back(dapple_buf);
  }

  RecordOutputsUsage(device_buffers, device_state, device_state,
                     definition_event, stream);

  return outputs;
}

StatusOr<std::vector<DAPPLEBuffer*>> DAPPLEExecutable::ExecuteRPCTask(
    absl::Span<DAPPLEBuffer* const> arguments,
    const ExecuteOptions& options, int64 local_dev_id, TaskNode* task,
    std::vector<std::unique_ptr<PjRtBuffer>>* ga_buffers) const {
  auto executable = task->executable();
  auto client = virtual_client_->gpu_client().get();
  auto& module = executable->module();

  std::vector<PjRtBuffer::ScopedHold> device_buffers;
  StatusOr<ScopedShapedBuffer> result_buffer_or_status =
      EnqueueExecution(arguments, options, local_dev_id,
                       task, ga_buffers, &device_buffers);

  if (!result_buffer_or_status.ok()) {
    LOG(ERROR) << "Execution failed: "
               << result_buffer_or_status.status();
    return result_buffer_or_status.status();
  }
  ScopedShapedBuffer result_buffer =
      result_buffer_or_status.ConsumeValueOrDie();

  LocalDeviceState* device_state = &client->device_state(local_dev_id);
  se::Stream* stream = device_state->compute_stream();
  StatusOr<EventPool::Handle> event_or =
      device_state->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    pjrt_utils::StallStreamOnError(device_state, stream);
    for (PjRtBuffer::ScopedHold& b : device_buffers) {
      if (b.type() == PjRtBuffer::ScopedHold::kDonation) {
        // Even though there was an error we need to call ConfirmDonation, which
        // renders b invalid, since the computation has been enqueued and b has
        // been donated.
        b.ConfirmDonation();
      }
    }
    return event_or.status();
  }

  auto definition_event = std::make_shared<BufferSequencingEvent>();
  definition_event->SetSequencingEvent(event_or.ConsumeValueOrDie(), stream);

  // 1. We first release the ScopeHold of input PjRtBuffers.
  auto task_def_ctx = task->def_ctx();
  //VLOG(0) << "device_buffers->" << device_buffers.size()
  //        << " arguments->" << arguments.size();
  RecordOutputsUsage(device_buffers, device_state, device_state,
                     definition_event, stream);

  CHECK(options.untuple_result && result_buffer.on_host_shape().IsTuple());
  CHECK(device_state->allocation_model() != LocalDeviceState::kSynchronous);

  auto entry = module.entry_computation();
  auto entry_root = entry->root_instruction();
  CHECK(entry_root->shape().IsTuple());
  int64 num_outputs = entry_root->operand_count();
  //int tuple_count = result_buffer.on_host_shape().tuple_shapes_size();
  int64 tuple_count = num_outputs;

  auto* exec_state = virtual_client()->execution_state();
  auto task_outputs = exec_state->PropagateOutputs(
      task, std::move(result_buffer), arguments, definition_event,
      virtual_client()->gpu_client().get(), local_dev_id).ConsumeValueOrDie();
  return task_outputs;
}

StatusOr<ScopedShapedBuffer> DAPPLEExecutable::EnqueueExecution(
    absl::Span<DAPPLEBuffer* const> arguments,
    const ExecuteOptions& options, int64 local_dev_id, TaskNode* task,
    std::vector<std::unique_ptr<PjRtBuffer>>* ga_buffers,
    std::vector<PjRtBuffer::ScopedHold>* device_buffers) const {
  auto executable = task->executable();
  auto* client = virtual_client_->gpu_client().get();
  int local_dev_count = client->device_count();
  Device* device = pjrt_utils::LookupDevice(*client, local_dev_id);
  int device_ordinal = device->local_device_state()->device_ordinal();
  LocalDeviceState* device_state = &client->device_state(device_ordinal);

  absl::flat_hash_set<BufferSequencingEvent*> events;
  std::vector<const Shape*> argument_host_shapes;
  std::vector<ExecutionInput> execution_inputs;
  const absl::flat_hash_set<int>& parameters_that_must_be_donated =
      parameters_that_must_be_donated_;
  int64 num_arg_handles = arguments.size();

  //auto executable = task->executable();
  CHECK(executable);
  auto& module = executable->module();
  auto entry = module.entry_computation();
  auto num_entry_params = entry->num_parameters();
  if (ga_buffers) {
    // GA execution
    CHECK(ga_buffers->size() == num_arg_handles);
    CHECK(num_entry_params == num_arg_handles << 1);
  } else {
    // Normal execution
    CHECK(num_entry_params == num_arg_handles);
  }

  int global_dev_id = task->comm_dev_mgr()->global_dev_id(task->split_id());
  for (int i = 0; i < num_entry_params; ++i) {
    if (i < num_arg_handles) {
      DAPPLEBuffer* handle = arguments[i];

      bool must_donate = parameters_that_must_be_donated.find(i) !=
                         parameters_that_must_be_donated.end();
      CHECK(!must_donate);
      CHECK (!handle->sharded());
      CHECK(handle->in_gpu(global_dev_id));

      device_buffers->emplace_back(handle->GetBufferWithHold(global_dev_id,
          must_donate ? PjRtBuffer::ScopedHold::kDonation
                      : PjRtBuffer::ScopedHold::kUsage));
    } else {
      auto& handle = ga_buffers->at(i-num_arg_handles);
      device_buffers->emplace_back(handle->GetBufferWithHold(
                                   PjRtBuffer::ScopedHold::kUsage));
    }

    PjRtBuffer::ScopedHold& device_buffer = device_buffers->back();
    if (!device_buffer.ok()) {
      return InvalidArgument(
          "Invalid buffer passed to Execute() as argument %d: "
          "%s",
          i, device_buffer.status().ToString());
    }

    GetDeviceBufferEvents(*device_buffer, /*get_usage_events=*/false, &events);
  }
  CHECK (!options.arguments_are_tupled);
  CHECK (!parameter_is_tupled_arguments_ && !options.arguments_are_tupled);

  execution_inputs.reserve(num_entry_params);
  int hold_idx = 0;
  //VLOG(0) << task->def_ctx()->name() << "->" << "BUFFER HOLDS#:" 
  //        << device_buffers->size() << ":" << num_entry_params;
  for (int i = 0; i < num_entry_params; ++i) {
    DAPPLEBuffer* dapple_buf = nullptr;
    if (i < num_arg_handles) {
      dapple_buf = arguments[i];
      execution_inputs.emplace_back(dapple_buf->on_device_shape());
    } else {
      auto& handle = ga_buffers->at(i-num_arg_handles);
      execution_inputs.emplace_back(handle->on_device_shape());
    }

    // Make an ExecutionInput from the device buffer.
    ExecutionInput& execution_input = execution_inputs.back();
    ShapeTree<MaybeOwningDeviceMemory>::iterator input_iterator =
        execution_input.MutableBuffers()->begin();
    ShapeTree<MaybeOwningDeviceMemory>::iterator iterator_end =
        execution_input.MutableBuffers()->end();
    const PjRtBuffer::ScopedHold& device_buffer =
        (*device_buffers)[hold_idx++];
    device_buffer.AddToInput(&input_iterator, iterator_end, &execution_input,
                             client->allocator());
    CHECK(input_iterator == iterator_end);
  }
 
  for (BufferSequencingEvent* event : events) {
    event->WaitForEventOnStream(device_state->compute_stream());
  }

  ExecutableRunOptions run_options;
  run_options.set_stream(device_state->compute_stream());
  run_options.set_host_to_device_stream(device_state->host_to_device_stream());
  run_options.set_allocator(client->allocator());
  run_options.set_intra_op_thread_pool(
      client->client()->backend().eigen_intra_op_thread_pool_device());
  run_options.set_rng_seed(device_state->GetNewPrngSeed());
  client->gpu_run_options()->set_nccl_context(virtual_client()->GetNcclContext());
  run_options.set_gpu_executable_run_options(client->gpu_run_options());

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  auto compute_reservation = std::make_shared<Semaphore::ScopedReservation>(
      device_state->compute_semaphore().ScopedAcquire(1));

  ServiceExecutableRunOptions opt(run_options,
      client->client()->mutable_backend()->StreamBorrower());
  opt.set_task_id(task->split_id().ids_);
  opt.set_comm_dev_mgr(task->comm_dev_mgr());

  StatusOr<ExecutionOutput> result_buffer_or_status =
      executable->ExecuteAsyncOnStreamWrapper(&opt,
          std::move(execution_inputs));

  CHECK (result_buffer_or_status.ok()) << result_buffer_or_status.status().ToString();
  if (!result_buffer_or_status.ok()) {
    return result_buffer_or_status.status();
  }

  if (device_state->allocation_model() == LocalDeviceState::kSynchronous) {
    ExecutionOutput& execution_output = result_buffer_or_status.ValueOrDie();
    // If we used a transient tuple for the arguments we donated its root table
    // buffer. In that case, and/or if we donated any input buffers that were
    // not aliased, the donated buffers are going to be passed back to us via
    // the execution output. We need to ensure they aren't freed until after
    // execution completes. (Currently XLA does not support aliasing tuple
    // tables, so if any donated parameter is a tuple there will be donated but
    // unaliased buffers.)
    std::vector<se::OwningDeviceMemory> donated_memory =
        execution_output.ConsumeToBeReleased();
    absl::InlinedVector<se::DeviceMemoryBase, 3> donated_ptrs;
    donated_ptrs.reserve(donated_memory.size());
    for (se::OwningDeviceMemory& owning : donated_memory) {
      // Release the owning memory so we can pass it to the closure.
      donated_ptrs.push_back(owning.Release());
    }
    device_state->ThenExecuteOnCallbackThread(
        device_state->compute_stream(),
        [references{std::make_tuple(entry_exe_,
                            compute_reservation/*, device_assignment_*/)},
         donated_ptrs{std::move(donated_ptrs)}, allocator{client->allocator()},
         device_ordinal]() {
          for (const auto& ptr : donated_ptrs) {
            TF_CHECK_OK(allocator->Deallocate(device_ordinal, ptr));
          }
        });
  } else {
    // Any donated memory returned by the ExecutionOutput can be immediately
    // freed.
    device_state->ThenRelease(
        device_state->compute_stream(),
        std::make_tuple(entry_exe_, compute_reservation/*,
                        device_assignment_*/));
  }

  return result_buffer_or_status.ConsumeValueOrDie().ConsumeResult();
}

Status DAPPLEExecutable::DoNcclSendRecv(
    DAPPLEBuffer* dapple_buf, const std::vector<int64>& send_recv_devices,
    se::Stream* stream, const ncclComm_t nccl_comm, bool is_send , bool need_sync) {
  const cudaStream_t* cu_stream = reinterpret_cast<const cudaStream_t*>(
      stream->implementation()->GpuStreamMemberHack());
  CHECK(is_send ? dapple_buf->in_gpu(send_recv_devices[0]) : dapple_buf->in_gpu(send_recv_devices[1]));
  auto pjrt_buf = is_send ? \
      dapple_buf->gpu_buffer(send_recv_devices[0]) : dapple_buf->gpu_buffer(send_recv_devices[1]);
  auto device_buffer = pjrt_buf->device_buffer();
  auto& shape = dapple_buf->on_device_shape();
  ncclDataType_t nccl_type = gpu::ToNcclDataType(shape.element_type()).ValueOrDie();

  ShapedBuffer buffer = device_buffer->AsShapedBuffer(
      shape, shape, virtual_client_->gpu_client()->client()->platform());
  auto root_buf = buffer.root_buffer();
  void* data_ptr = root_buf.opaque();
  int64 elem_cnt = ShapeUtil::ByteSizeOf(shape) / ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  if (is_send) {
    XLA_CUDA_RETURN_IF_ERROR(ncclSend(data_ptr, elem_cnt,
                                      nccl_type, 1/*peer rank*/, nccl_comm, *cu_stream));
  } else {
    XLA_CUDA_RETURN_IF_ERROR(ncclRecv(data_ptr, elem_cnt,
                                      nccl_type, 0/*root rank*/, nccl_comm, *cu_stream));
  }

  if (need_sync) {
    CUDACHECK(cudaStreamSynchronize(*cu_stream));
  }

  return Status::OK();
}

void DAPPLEExecutable::NcclAllReduce(int device_id, se::Stream* stream,
    const ncclComm_t& nccl_comm, std::vector<DAPPLEBuffer*>& dapple_bufs) {
  const cudaStream_t* cu_stream = reinterpret_cast<const cudaStream_t*>(
      stream->implementation()->GpuStreamMemberHack());
  for (auto dapple_buf : dapple_bufs) {
    auto pjrt_buf = dapple_buf->gpu_buffer(device_id);
    ShapedBuffer buffer = pjrt_buf->AsShapedBuffer().ConsumeValueOrDie();
    auto root_buf = buffer.root_buffer();
    void* data_ptr = root_buf.opaque();
    int64 data_size = root_buf.size();
    CHECK(!(data_size & 0x3));
    NCCLCHECK(ncclAllReduce(data_ptr, data_ptr, data_size >> 2,
                       ncclFloat, ncclSum, nccl_comm, *cu_stream));
  }
}

void DAPPLEExecutable::NcclAllReduce(int device_id, se::Stream* stream,
    const ncclComm_t& nccl_comm, ScopedShapedBuffer* result_buffer) const {
  const cudaStream_t* cu_stream = reinterpret_cast<const cudaStream_t*>(
      stream->implementation()->GpuStreamMemberHack());

  auto device_shape = result_buffer->on_device_shape();
  CHECK(device_shape.IsTuple());
  int64 tuple_count = device_shape.tuple_shapes_size();
  for (int64 i = 0; i < tuple_count; ++i) {
    ShapedBuffer tuple_buffer = 
        result_buffer->SubShapedBuffer({i}).ConsumeValueOrDie();
    auto root_buf = tuple_buffer.root_buffer();
    void* data_ptr = root_buf.opaque();
    int64 data_size = root_buf.size();
    CHECK(!(data_size & 0x3));
    NCCLCHECK(ncclAllReduce(data_ptr, data_ptr, data_size >> 2,
                       ncclFloat, ncclSum, nccl_comm, *cu_stream));
  }
}

Status DAPPLEExecutable::InitializeGABuffers(
    std::vector<std::unique_ptr<PjRtBuffer>>& ga_buffers,
    Device* gpu_device, const HloModule& module) const {
  auto ga_entry = module.entry_computation();
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      gpu_device->GetLocalDeviceState());

  auto client = virtual_client_->gpu_client();
  int64 num_params = ga_entry->num_parameters();

  ga_buffers.reserve(num_params);
  for (int64 i = 0; i < num_params; ++i) {
    auto instr = ga_entry->parameter_instruction(i);
    auto& shape = instr->shape();
    CHECK(ShapeUtil::Equal(shape, 
        ga_entry->parameter_instruction(i)->shape()));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer> py_buffer,
        pjrt_utils::AllocateDestinationBuffer(shape, gpu_device, local_device,
            local_device->host_to_device_stream(), client.get()));

    PjRtBuffer::ScopedHold device_buffer(py_buffer->GetBufferWithUsageHold());
    CHECK(device_buffer.ok());

    ShapedBuffer buffer = device_buffer->AsShapedBuffer(
        shape, shape, client->client()->platform());

    auto dev_mem = buffer.root_buffer();
    local_device->compute_stream()->ThenMemZero(&dev_mem, dev_mem.size());

    std::shared_ptr<BufferSequencingEvent> event =
        device_buffer->definition_events()[0];
    TF_CHECK_OK(pjrt_utils::AddDestinationBufferSynchronization(
        local_device, std::move(device_buffer), event,
        local_device->host_to_device_stream()));

    ga_buffers.emplace_back(std::move(py_buffer));
  }

  return Status::OK();
}

/* static */
StatusOr<std::unique_ptr<Literal>> DAPPLEExecutable::ConvertOutputToLiteral(
    std::vector<DAPPLEBuffer*>& dapple_bufs,
    int64 start_idx, int64 range) {
  std::vector<Shape> shape_elements;
  for (int64 i = start_idx; i < start_idx + range; ++i) {
    auto* d_buf = dapple_bufs[i];
    auto& whole_shape = d_buf->on_device_shape();
    shape_elements.emplace_back(whole_shape);
  }

  auto root_shape = ShapeUtil::MakeTupleShape(shape_elements);
  auto ret = absl::make_unique<Literal>(root_shape);

  for (int64 i = start_idx; i < start_idx + range; ++i) {
    auto* d_buf = dapple_bufs[i];
    auto literal_or = d_buf->ToLiteral();
    CHECK(literal_or.ok());
    auto literal = literal_or.ConsumeValueOrDie();

    std::memcpy(
        ret->untyped_data({i}), literal->untyped_data(),
        literal->size_bytes());
  }

  return ret;
}

Status DAPPLEExecutable::ExecuteHostRecvTask(const TaskNode* split_task) {
  for (auto child : split_task->children()) {
    if (child->task_type() == TaskNode::TaskType::kRecv) {
      auto task_def_ctx = child->def_ctx();
      if (task_def_ctx->entry_def_ctx()) {
        VLOG(1) << "DoHostRecv for task->" << child->name();
        auto exec_state = virtual_client()->execution_state();
        auto task_ctx = exec_state->get_task_context(child->node_id());
        DoHostRecv(task_ctx, child, plan_.get());
      }
    }
  }
  return Status::OK();
}

StatusOr<std::vector<DAPPLEBuffer*>> DAPPLEExecutable::ExecuteRemotePlan(
    const ExecuteOptions& options) {
  rpc_task_ = true;

  std::vector<DAPPLEBuffer*> plan_outputs;
  auto task_graph = plan_->task_graph();
  TaskNode* split_task = task_graph->source();

  auto gpu_client = virtual_client()->gpu_client();
  int local_dev_count = gpu_client->device_count();

  auto* launch_ctx = virtual_client()->whole_graph_launch_context();
  auto& module = *(plan_->task_module(split_task));
  
  TaskNode* entry_recv = nullptr;
  for (auto expect_recv : split_task->children()) {
    if (expect_recv->task_type() != TaskNode::TaskType::kRecv) continue;
    if (!expect_recv->def_ctx()->entry_def_ctx()) continue;
    CHECK(!entry_recv);
    entry_recv = expect_recv;
  }
  CHECK(entry_recv);

  ExecutionState* execution_state = nullptr;
  if (virtual_client()->warm_up()) {
    auto* vars_map = module.variable_map();
    virtual_client()->InitializeExecutionState(task_graph, vars_map, plan_->worker_rank());
    launch_ctx->set_num_vars(vars_map->size());
    launch_ctx->InitializePlanOutputsBuffers(&module);
    execution_state = virtual_client()->execution_state();
    execution_state->set_global_step(options.global_step);
    execution_state->InitializeAllTaskOutputBuffers(
        virtual_client()->virtual_device(), gpu_client.get(),
        gpu_client->client()->platform());
    bool vars_init_done = true;
    auto& var_buf_handles = launch_ctx->recv_args_recorder().variable_buf_handles;
    auto& recv_args_storage_map = launch_ctx->recv_args_recorder().recv_args_storage_map;
    auto vars_or = launch_ctx->Resolve(var_buf_handles);
    CHECK(vars_or.ok());
    auto& variables = vars_or.ValueOrDie();
    std::map<int, int> arg_var_map;
    for (auto& it : recv_args_storage_map) {
      if (it.second.first) {
        arg_var_map[it.first] = it.second.second;
      }
    }
    execution_state->InitializeDistributedSaver(
        variables, arg_var_map, gpu_client.get());
    for (auto* var : variables) {
      vars_init_done &= (var->raw() != nullptr);
      if (!vars_init_done) break;
    }
    if (!vars_init_done) {
      if (options.restore_from_ckpt) {
        execution_state->RestoreFromCheckpoint();
      } else {
        auto* init_specs_map = module.init_specs_map();
        uint64 start_us = Env::Default()->NowMicros();
        execution_state->InitializeLocalVariables(
            vars_or.ValueOrDie(), arg_var_map, *init_specs_map, gpu_client.get());
        uint64 end_us = Env::Default()->NowMicros();
        float duration_ms = (end_us - start_us) / 1000.0f;
        VLOG(0) << "All initialization duration = " << duration_ms;

      }
    }
    // Lazy save first iterations
    if (options.lazy_save_ckpt) {
      uint64 start_us = Env::Default()->NowMicros();
      execution_state->SaveCheckpoint();
      uint64 end_us = Env::Default()->NowMicros();
      float duration_ms = (end_us - start_us) / 1000.0f;
      VLOG(0) << "SaveCheckpoint duration = " << duration_ms;

    }
    virtual_client_->set_warm_up(false);
  }

  // Reasoning of actual number of devices needed
  auto plan_device_count = plan_->local_device_count();
  CHECK(plan_device_count <= local_dev_count);
  auto device_count_needed = plan_->used_device_count();
  CHECK(device_count_needed == plan_device_count);

  execution_state = virtual_client()->execution_state();
  CHECK(execution_state);
  execution_state->set_global_step(options.global_step);
  auto output_count = Preprocess(split_task);
  plan_outputs.resize(output_count);

  // 1. Execute SplitRecv to prepare training data for local muliple devices'
  //    consumption. Slave nodes receive data from remote master machine (CPU's domain).
  CHECK(ExecuteHostRecvTask(split_task).ok());

  // 2. Execute the per device task lists
  std::vector<std::thread> workers;
  VLOG(2) << "device_count_needed: " << device_count_needed;
  for (int64 local_dev_id = 0; local_dev_id < device_count_needed; ++local_dev_id) {
    VLOG(2) << "local_dev_id: " << local_dev_id;
    workers.push_back(std::thread([this, options, local_dev_id,
                                   device_count_needed,
                                   execution_state]() {
      ExecuteTaskList(local_dev_id, device_count_needed, execution_state,
                      options, plan_.get());
    }));
  }


  std::for_each(workers.begin(), workers.end(),
                [](std::thread &t) { t.join(); });

  // Execute the Merge Task to Produce Outputs
  TaskNode* merge_task = task_graph->sink();
  CHECK(merge_task->task_type() == TaskNode::TaskType::kMerge);

  // Build outputs
  BuildPlanOutputsTF(merge_task, output_count, plan_outputs,
                     plan_->distributed_plan(),
                     entry_recv->mutable_port_map());

  virtual_client_->execution_state()->CleanupCachedTasksOutputsBuffers();

  return std::move(plan_outputs);
}

int DAPPLEExecutable::Preprocess(TaskNode* split_task) {
  auto task_graph = plan_->task_graph();

  // Execute the Split Task
  CHECK(split_task->task_type() == TaskNode::TaskType::kSplit);
  auto entry_def_ctx = split_task->def_ctx();
  CHECK(entry_def_ctx->entry_def_ctx());
  auto& module = *(plan_->task_module(split_task));

  auto entry = module.entry_computation();
  auto entry_root = entry->root_instruction();
  int64 output_count = 1;
  if (entry_root->opcode() == HloOpcode::kTuple) {
    output_count = entry_root->operand_count();
    for (int64 i = 0; i < output_count; ++i) {
      //auto instr = entry_root->mutable_operand(i);
      //auto dim_to_slice = instr->dist_spec().partition_dim();
      int dim_to_slice = -1;
      if (entry_def_ctx->output_dim_to_slice_.count(i)) {
        dim_to_slice = entry_def_ctx->output_dim_to_slice_[i];
      }

      if (dim_to_slice >= 0) {
        //VLOG(0) << "sharded_dim_to_slice->" << i
        //        << ":" << dim_to_slice;
        sharded_dim_to_slice_[i] = dim_to_slice;
      }
    }
  }
  return output_count;
}

StatusOr<std::vector<std::unique_ptr<DAPPLEBuffer>>>
    DAPPLEExecutable::ExecutePlan(
        absl::Span<DAPPLEBuffer* const> arguments,
        const ExecuteOptions& options) {
  std::vector<std::unique_ptr<DAPPLEBuffer>> plan_outputs;
  auto task_graph = plan_->task_graph();
  TaskNode* split_task = task_graph->source();

  // Reasoning of actual number of devices needed
  auto local_dev_count = plan_->local_device_count();
  auto* execution_state = virtual_client()->execution_state();
  CHECK(local_dev_count == virtual_client_->gpu_client()->device_count());
  auto device_count_needed = plan_->used_device_count();
  CHECK(device_count_needed <= local_dev_count);

  CHECK(plan_->distributed_plan());   // both master and slave own a dist plan

  // Set nccl information in each hlo module.
  if (plan_->distributed_plan() &&    // mult worker
      plan_->worker_rank() == 0) {    // master
    virtual_client_->CreateAndInitNcclContext(plan_.get());
  }

  auto output_count = Preprocess(split_task);
  plan_outputs.resize(output_count);

  // Populate task outputs
  auto* split_task_ctx = execution_state->get_task_context(
      split_task->node_id());
  auto output_buffers = split_task_ctx->mutable_output_buffers();
  output_buffers->clear();

  auto& module = *(plan_->task_module(split_task));
  auto entry = module.entry_computation();

  auto entry_def_ctx = split_task->def_ctx();
  int64 num_args = arguments.size();
  for (int64 i = 0; i < num_args; ++i) {
    auto arg = arguments.at(i);
    auto param = entry->parameter_instruction(i);
    int arg_no = param->parameter_number();
    if (!entry_def_ctx->input_dim_to_slice_.count(arg_no)) {
      CHECK(ShapeUtil::Equal(arg->on_host_shape(), param->shape()));
    }

    output_buffers->emplace_back(arg);
  }

  // Do Host Send
  for (auto expect_send : split_task->children()) {
    if (expect_send->task_type() != TaskNode::TaskType::kSend) continue;
    DoRPCSend(expect_send, true/*transfer_variable*/,
              virtual_client_->var_arg_map());
  }

  // Launch Remote Plan
  std::vector<std::thread> remote_workers;
  if (plan_->distributed_plan() &&    // mult worker
      plan_->worker_rank() == 0) {    // master
    auto& coord = virtual_client_->coordinator();
    coord.ExecuteRemotePlan(remote_workers, options);
  }
 
  // Execute the per device task lists
  std::vector<std::thread> workers;
  VLOG(0) << "device_count_needed: " << device_count_needed;
  for (int64 local_dev_id = 0; local_dev_id < device_count_needed; ++local_dev_id) {
    VLOG(0) << "local_dev_id: " << local_dev_id;
    workers.push_back(std::thread([this, options, local_dev_id,
                                   device_count_needed, execution_state]() {
      ExecuteTaskList(local_dev_id, device_count_needed, execution_state,
                      options, plan_.get());
    }));
  }

  std::for_each(workers.begin(), workers.end(),
                [](std::thread &t) { t.join(); });

  // Execute the Merge Task to Produce Outputs
  TaskNode* merge_task = task_graph->sink();
  CHECK(merge_task->task_type() == TaskNode::TaskType::kMerge);

  // Build outputs
  BuildPlanOutputs(merge_task, output_count, plan_outputs);

  // Cleanup distributed workers
  std::for_each(remote_workers.begin(), remote_workers.end(),
                [](std::thread &t) { t.join(); });

  return std::move(plan_outputs);
}

void DAPPLEExecutable::BuildPlanOutputsTF(
    TaskNode* merge_task, int output_count,
    std::vector<DAPPLEBuffer*>& plan_outputs,
    DistributedPlan* dist_plan,
    std::map<int, int>* port_map) {
  auto* whole_graph_launch_context = virtual_client_->whole_graph_launch_context();

  auto num_vars = whole_graph_launch_context->num_vars();
  auto entry_def_ctx = merge_task->def_ctx();
  int num_inputs = entry_def_ctx->input_arg_map_.size() - num_vars;
  int64 num_fetches = output_count - num_vars;
  auto& input_specs = merge_task->input_specs();
  CHECK(output_count >= int(input_specs.size()));
  CHECK(plan_outputs.size() == output_count);

  auto* exec_state = virtual_client()->execution_state();
  for (int64 i = 0; i < output_count; ++i) {
    if (!input_specs.count(i)) continue;

    auto& specs = input_specs.at(i);
    int64 specs_size = specs.size();

    DAPPLEBuffer* dapple_buf = nullptr;
    if (i < num_fetches) {
      dapple_buf = whole_graph_launch_context->ResolveOutput(i).ConsumeValueOrDie();
    } else {
      auto* recv_args_recorder = whole_graph_launch_context->mutable_recv_args_recorder();
      int idx_in_var_buf = recv_args_recorder->recv_args_storage_map[i - num_fetches + num_inputs].second;
      DAPPLEBufferHandle var_handle = recv_args_recorder->variable_buf_handles[idx_in_var_buf];
      dapple_buf = whole_graph_launch_context->Resolve(var_handle).ConsumeValueOrDie();
    }

    for (int64 s = 0; s < specs_size; ++s) {
      auto parent = specs[s].first;
      int global_parent_dev_id = parent->comm_dev_mgr()->global_dev_id(parent->split_id());
      VLOG(1) << "worker rank: " << plan_->worker_rank() << ", local device count: "
              << plan_->local_device_count() << ", global parent dev id: " << global_parent_dev_id;
      auto* parent_task_ctx = exec_state->get_task_context(parent->node_id());
      auto* output_buffers = parent_task_ctx->mutable_output_buffers();
      auto out_idx = specs[s].second;
      CHECK(out_idx < int(output_buffers->size()));
      auto out_dapple_buf = output_buffers->at(out_idx);
      auto pjrt_buf = out_dapple_buf->steal_gpu_buffer(global_parent_dev_id);
      dapple_buf->set_gpu_buffer(std::move(pjrt_buf), global_parent_dev_id);
    } // End of specs_size for-loop

    plan_outputs[i] = dapple_buf;
  }

  if (dist_plan) {
    int my_rank = dist_plan->worker_rank();
    TaskDAG* task_graph = dist_plan->task_graph();
    for (int i = 0; i < num_fetches; ++i) {
      auto dapple_buf = whole_graph_launch_context->ResolveOutput(i).ConsumeValueOrDie();
      if (my_rank > 0) {
        // slave
        // Skip dapple buffer which is not found in input_specs
        if (input_specs.find(i) == input_specs.end()) continue;

        bool need_send = true;

        for (int slice_id : entry_def_ctx->output_idx_global_dev_map_[i]) {
          std::vector<int64> addr = merge_task->comm_dev_mgr()->LinearIdxToAddrBySplitNums(slice_id);
          int g_dev = merge_task->comm_dev_mgr()->global_dev_id(SplitId(addr,
                                                                  task_graph->share_dev_flags(),
                                                                  task_graph->stage_split_ordinal()));
          int worker_id = merge_task->comm_dev_mgr()->worker_id(g_dev);
          if (worker_id == 0) {
            need_send = false;
            break;
          }
        }

        if (!need_send) continue;

        int slice_id = *(entry_def_ctx->output_idx_global_dev_map_[i].begin());
        std::vector<int64> addr = merge_task->comm_dev_mgr()->LinearIdxToAddrBySplitNums(slice_id);
        int g_dev = merge_task->comm_dev_mgr()->global_dev_id(SplitId(addr,
                                                                  task_graph->share_dev_flags(),
                                                                  task_graph->stage_split_ordinal()));
        int worker_id = merge_task->comm_dev_mgr()->worker_id(g_dev);
        if (worker_id != my_rank) {
          continue;
        }

        std::vector<int64> global_devices = {g_dev, 0};
        int local_dev_id = merge_task->comm_dev_mgr()->local_dev_id(g_dev);
        // w0 : g0(0), g1(1)
        // w1 : g0(2), g1(3)

        auto gpu_client = virtual_client_->gpu_client();
        Device* device = pjrt_utils::LookupDevice(*gpu_client, local_dev_id);
        LocalDeviceState* device_state = device->local_device_state();
        se::Stream* stream = device_state->compute_stream();
        gpu::NcclContext* nccl_ctx = virtual_client_->GetNcclContext();
        gpu::NcclComm* nccl_comm_ptr = nccl_ctx->GetNcclComm(global_devices, 0/*send_rank*/).ConsumeValueOrDie();
        ncclComm_t nccl_comm = nccl_comm_ptr->comm();
        int saved_device = 0;
        CUDACHECK(cudaGetDevice(&saved_device));
        CUDACHECK(cudaSetDevice(local_dev_id));
        CHECK(DoNcclSendRecv(
            dapple_buf, global_devices, stream, nccl_comm, true/*is_send*/, true/*need_sync*/).ok());
        CUDACHECK(cudaSetDevice(saved_device));
      } else {
        // master
        // Skip dapple buffer which is found in input_specs
        if (input_specs.find(i) != input_specs.end()) continue;
        // Recv at first valid shard
        bool need_recv = true;

        for (int slice_id : entry_def_ctx->output_idx_global_dev_map_[i]) {
          std::vector<int64> addr = merge_task->comm_dev_mgr()->LinearIdxToAddrBySplitNums(slice_id);
          int g_dev = merge_task->comm_dev_mgr()->global_dev_id(SplitId(addr,
                                                                  task_graph->share_dev_flags(),
                                                                  task_graph->stage_split_ordinal()));
          int worker_id = merge_task->comm_dev_mgr()->worker_id(g_dev);
          if (worker_id == 0) {
            need_recv = false;
            break;
          }
        }

        if (!need_recv) continue;
        int slice_id = *(entry_def_ctx->output_idx_global_dev_map_[i].begin());
        std::vector<int64> addr = merge_task->comm_dev_mgr()->LinearIdxToAddrBySplitNums(slice_id);
        int g_dev = merge_task->comm_dev_mgr()->global_dev_id(SplitId(addr,
                                                                  task_graph->share_dev_flags(),
                                                                  task_graph->stage_split_ordinal()));
        std::vector<int64> global_devices = {g_dev, 0};

        int local_dev_id = merge_task->comm_dev_mgr()->local_dev_id(0);
        auto gpu_client = virtual_client_->gpu_client();
        Device* device = pjrt_utils::LookupDevice(*gpu_client, local_dev_id);
        LocalDeviceState* device_state = device->local_device_state();
        auto shape = dapple_buf->on_device_shape();
        StatusOr<std::unique_ptr<PjRtBuffer>> buffer_or =
            pjrt_utils::AllocateDestinationBuffer(shape, device, device_state,
                device_state->host_to_device_stream(), gpu_client.get());
        CHECK(buffer_or.ok());
        auto py_buffer = buffer_or.ConsumeValueOrDie();
        PjRtBuffer::ScopedHold device_buffer(
                             py_buffer->GetBufferWithUsageHold());
        CHECK(device_buffer.ok());

        std::shared_ptr<BufferSequencingEvent> event =
            device_buffer->definition_events()[0];

        TF_CHECK_OK(pjrt_utils::AddDestinationBufferSynchronization(
            device_state, std::move(device_buffer), event,
            device_state->host_to_device_stream()));

        dapple_buf->set_gpu_buffer(std::move(py_buffer), 0);
        se::Stream* stream = device_state->compute_stream();
        gpu::NcclContext* nccl_ctx = virtual_client_->GetNcclContext();
        gpu::NcclComm* nccl_comm_ptr = nccl_ctx->GetNcclComm(global_devices, 1/*recv_rank*/).ConsumeValueOrDie();
        ncclComm_t nccl_comm = nccl_comm_ptr->comm();
        int saved_device = 0;
        CUDACHECK(cudaGetDevice(&saved_device));
        CUDACHECK(cudaSetDevice(local_dev_id));
        CHECK(DoNcclSendRecv(
            dapple_buf, global_devices, stream, nccl_comm, false/*is_send*/, true/*need_sync*/).ok());
        CUDACHECK(cudaSetDevice(saved_device));
        plan_outputs[i] = dapple_buf;
      }
    }
  }
}

void DAPPLEExecutable::BuildPlanOutputs(
    TaskNode* merge_task, int output_count,
    std::vector<std::unique_ptr<DAPPLEBuffer>>& plan_outputs) {
  auto* exec_state = virtual_client()->execution_state();
  auto& input_specs = merge_task->input_specs();
  CHECK(output_count >= int(input_specs.size()));
  CHECK(plan_outputs.size() == output_count);
  for (int64 i = 0; i < output_count; ++i) {
    if (!input_specs.count(i)) continue;

    auto& specs = input_specs.at(i);
    int64 specs_size = specs.size();

    DAPPLEBuffer* dapple_buf = nullptr;
    for (int64 s = 0; s < specs_size; ++s) {
      auto parent = specs[s].first;
      auto parent_dev_id = parent->device_id();
      CHECK(parent_dev_id >= 0);
      int global_parent_dev_id = plan_->worker_rank()*plan_->local_device_count()
                                 + parent_dev_id;

      VLOG(0) << "worker rank: " << plan_->worker_rank() << ", local device count: "
              << plan_->local_device_count() << ", global parent dev id: " << global_parent_dev_id;
      auto* parent_task_ctx = exec_state->get_task_context(parent->node_id());
      auto output_buffers = parent_task_ctx->mutable_output_buffers();
      auto out_idx = specs[s].second;
      CHECK(out_idx < int(output_buffers->size()));
      auto shard_buf = output_buffers->at(out_idx);

      if (!dapple_buf) {
        dapple_buf = shard_buf;
        continue;
      } else {// if (sharded_dim_to_slice_.count(i))
        // Let's keep all replicas or shards
        auto pjrt_shard = shard_buf->steal_gpu_buffer(global_parent_dev_id);
        dapple_buf->add_gpu_shard(std::move(pjrt_shard), global_parent_dev_id);
      }
      delete shard_buf;
      //output_buffers->erase(output_buffers->begin() + out_idx);
    }
    plan_outputs[i] = std::move(std::unique_ptr<DAPPLEBuffer>(dapple_buf));
  }
}

StatusOr<std::vector<DAPPLEBuffer*>>
    DAPPLEExecutable::ExecuteRPCPlan(
        absl::Span<DAPPLEBuffer* const> arguments,
        const ExecuteOptions& options) {
  rpc_task_ = true;
  auto task_graph = plan_->task_graph();
  auto plan_device_count = plan_->local_device_count();
  auto gpu_client = virtual_client_->gpu_client();
  auto local_dev_count = gpu_client->device_count();
  CHECK(plan_device_count <= local_dev_count);
  auto* whole_graph_launch_context =
      virtual_client_->whole_graph_launch_context();

  auto device_count_needed = plan_->used_device_count();
  VLOG(1) << "local_dev_count: " << local_dev_count;
  CHECK(device_count_needed <= local_dev_count);

  // setup communicator at each iteration
  if (virtual_client()->warm_up() && plan_->worker_rank() == 0) {
    virtual_client_->CreateAndInitNcclContext(plan_.get());
  }

  auto& coord = virtual_client_->coordinator();
  TaskNode* split_task = task_graph->source();
  auto& module = *(plan_->task_module(split_task));
  ExecutionState *execution_state = nullptr;
  if (virtual_client_->warm_up()) {
    auto* vars_map = module.variable_map();
    virtual_client_->InitializeExecutionState(task_graph, vars_map, plan_->worker_rank());
    whole_graph_launch_context->set_num_vars(vars_map->size());
    whole_graph_launch_context->InitializePlanOutputsBuffers(&module);
    execution_state = virtual_client_->execution_state();
    execution_state->InitializeAllTaskOutputBuffers(
        virtual_client_->virtual_device(), gpu_client.get(),
        gpu_client->client()->platform());
  }

  execution_state = virtual_client_->execution_state();
  execution_state->set_global_step(options.global_step);
  CHECK(execution_state);
  auto output_count = Preprocess(split_task);
  std::vector<DAPPLEBuffer*> plan_outputs(output_count, nullptr);

  auto entry_def_ctx = split_task->def_ctx();
  auto entry = module.entry_computation();

  // Populate task outputs
  auto* split_task_ctx = execution_state->get_task_context(split_task->node_id());
  auto* output_buffers = split_task_ctx->mutable_output_buffers();
  output_buffers->clear();

  int64 num_args = arguments.size();
  for (int64 i = 0; i < num_args; ++i) {
    auto arg = arguments.at(i);
    auto param = entry->parameter_instruction(i);
    int arg_no = param->parameter_number();
    output_buffers->emplace_back(arg);
  }

  // Do Host Send
  for (auto expect_send : split_task->children()) {
    if (expect_send->task_type() != TaskNode::TaskType::kSend) {
      continue;
    }
    if (fake_input_ && !virtual_client_->warm_up()) continue;
    DoRPCSend(expect_send, virtual_client_->warm_up(),
              virtual_client_->var_arg_map());
  }

  // Launch Remote Plan
  std::vector<std::thread> remote_workers;
  if (plan_->distributed_plan() &&     // true: mult worker
      plan_->worker_rank() == 0) {     // true: master
    coord.ExecuteRemotePlan(remote_workers, options);
  }

  if (virtual_client_->warm_up()) {
    std::vector<DAPPLEBuffer*> local_variables;
    std::map<int, int> arg_local_vars_map;
    execution_state->ResolveLocalVariables(
        arguments, &local_variables, &arg_local_vars_map);
    execution_state->InitializeDistributedSaver(
        local_variables, arg_local_vars_map, gpu_client.get());
    bool vars_init_done = true;
    auto* vars_map = module.variable_map();
    for (auto& iter : *vars_map) {
      auto arg_no = iter.first;
      auto* var_d_buf = arguments[arg_no];
      vars_init_done &= (var_d_buf->raw() != nullptr);
      if (!vars_init_done) break;
    }

    if (!vars_init_done) {
      if (options.restore_from_ckpt) {
        execution_state->RestoreFromCheckpoint();
      } else {
        auto* init_specs_map = module.init_specs_map();
        uint64 start_us = Env::Default()->NowMicros();
        execution_state->InitializeLocalVariables(
            local_variables, arg_local_vars_map, *init_specs_map, gpu_client.get());
        uint64 end_us = Env::Default()->NowMicros();
        float duration_ms = (end_us - start_us) / 1000.0f;
        VLOG(0) << "All initialization duration = " << duration_ms;
      }
    } 
    // Lazy save first iterations
    if (options.lazy_save_ckpt) {
      uint64 start_us = Env::Default()->NowMicros();
      execution_state->SaveCheckpoint();
      uint64 end_us = Env::Default()->NowMicros();
      float duration_ms = (end_us - start_us) / 1000.0f;
      VLOG(0) << "SaveCheckpoint duration = " << duration_ms;
    }
    virtual_client_->set_warm_up(false);
  }

  // Execute the per device task lists
  std::vector<std::thread> workers;
  VLOG(1) << "device_count_needed: " << device_count_needed;
  for (int64 local_dev_id = 0; local_dev_id < device_count_needed; ++local_dev_id) {
    VLOG(1) << "local_dev_id: " << local_dev_id;
    workers.push_back(std::thread([this, execution_state, options,
                                   local_dev_id, device_count_needed]() {
      ExecuteTaskList(local_dev_id, device_count_needed, execution_state,
                      options, plan_.get());
    }));
  }

  std::for_each(workers.begin(), workers.end(),
                [](std::thread &t) { t.join(); });

  // Execute the Merge Task to Produce Outputs
  TaskNode* merge_task = task_graph->sink();
  CHECK(merge_task->task_type() == TaskNode::TaskType::kMerge);

  // Build RPC plan outputs
  BuildPlanOutputsTF(merge_task, output_count, plan_outputs,
                     plan_->distributed_plan());

  virtual_client()->execution_state()->CleanupCachedTasksOutputsBuffers();

  // Cleanup distributed workers
  std::for_each(remote_workers.begin(), remote_workers.end(),
                [](std::thread &t) { t.join(); });

  return std::move(plan_outputs);
}

StatusOr<std::vector<std::unique_ptr<DAPPLEBuffer>>>
    DAPPLEExecutable::Execute(
        absl::Span<DAPPLEBuffer* const> arguments,
        const ExecuteOptions& options) {
  std::vector<std::unique_ptr<DAPPLEBuffer>> outputs;
  CHECK(0 && "Deprecated Execute Interface!");
  return outputs;
}

Status DAPPLEExecutable::SetUpDonation(PjRtClient* client, bool tuple_inputs) {
  TF_ASSIGN_OR_RETURN(
      absl::flat_hash_set<int> parameters_to_donate,
      client->GetParametersThatMustBeDonated(*entry_exe_, tuple_inputs));
  parameters_that_must_be_donated_ = std::move(parameters_to_donate);
  return Status::OK();
}

}  // namespace xla
