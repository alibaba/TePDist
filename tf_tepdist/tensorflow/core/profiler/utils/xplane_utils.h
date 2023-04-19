/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {

// Returns the plane with the given name or nullptr if not found.
const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name);

// Returns all the planes with a given prefix.
std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix);

// Returns the plane with the given name, create it if necessary.
XPlane* GetOrCreatePlane(XSpace* space, absl::string_view name);

// Returns true if event is nested by parent.
bool IsNested(const tensorflow::profiler::XEvent& event,
              const tensorflow::profiler::XEvent& parent);

void AddOrUpdateIntStat(int64 metadata_id, int64 value,
                        tensorflow::profiler::XEvent* event);

void AddOrUpdateStrStat(int64 metadata_id, absl::string_view value,
                        tensorflow::profiler::XEvent* event);

// Creates an XEvent with int64 stats.
XEventBuilder CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, int64 /*stat_value*/>& stats = {});
XEventBuilder CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    HostEventType event_type, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, int64 /*stat_value*/>& stats);

// Creates an XEvent with string stats.
XEventBuilder CreateXEventWithStringViewMetadataValue(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, absl::string_view /*stat_value*/>&
        stats);

// Creates an XEvent with int64 and string stats.
XEventBuilder CreateXEventWithIntAndStringViewMetadataValue(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, int64 /*stat_value*/>& int_stats,
    const absl::flat_hash_map<StatType, absl::string_view /*stat_value*/>&
        str_stats);

void RemovePlaneWithName(XSpace* space, absl::string_view name);
void RemoveEmptyPlanes(XSpace* space);
void RemoveEmptyLines(XPlane* plane);

// Returns the plane with the given name in the container or null if not found.
XPlane* FindMutablePlaneWithName(XSpace* space, absl::string_view name);

// Returns the plane with the given name in the container. If necessary, adds a
// new plane to the container.
XPlane* FindOrAddMutablePlaneWithName(XSpace* space, absl::string_view name);

// Sorts each XLine's XEvents by offset_ps (ascending) and duration_ps
// (descending) so nested events are sorted from outer to innermost.
void SortXPlane(XPlane* plane);
// Sorts each plane of the XSpace.
void SortXSpace(XSpace* space);

// Normalize timestamps by time-shifting to start_time_ns_ as origin.
void NormalizeTimestamps(XPlane* plane, uint64 start_time_ns);
void NormalizeTimestamps(XSpace* space, uint64 start_time_ns);

// Merge Xplane src_plane into Xplane dst_plane, both plane level stats, lines,
// events and event level stats are merged; If src_plane and dst_plane both have
// the same line, which have different start timestamps, we will normalize the
// events offset timestamp correspondingly.
void MergePlanes(const XPlane& src_plane, XPlane* dst_plane);

// Plane's start timestamp is defined as the minimum of all lines' start
// timestamps. If zero line exists, return 0;
uint64 GetStartTimestampNs(const XPlane& plane);

// Creates a Xevent in the given plane & line for a tf-function.
void CreateTfFunctionCallEvent(XPlaneBuilder* plane_builder,
                               XLineBuilder* line_builder,
                               absl::string_view function_name, int64 offset_ps,
                               int64 duration_ps,
                               absl::string_view execution_mode,
                               int64 tracing_count = -1);
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_
