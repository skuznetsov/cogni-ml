// Opt-in serial native/MPSGraph command chain. MPSGraph may commitAndContinue;
// every subsequent native encoder must use the wrapper's current root.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#define GS_MPS_API_PROFILE_IMPLEMENTATION
#include "mps_api_profile.h"
#include <mach/mach_time.h>
#include <atomic>
#include <cmath>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <type_traits>

static_assert(sizeof(gs_mps_api_profile_stats) == 10 * sizeof(uint64_t),
              "MPS API profile stats FFI layout changed");
static_assert(sizeof(gs_mps_chain_trace_info) == 8 * sizeof(uint64_t),
              "MPS chain trace info FFI layout changed");
static_assert(sizeof(gs_mps_chain_trace_event) == 12 * sizeof(uint64_t),
              "MPS chain trace event FFI layout changed");
static_assert(sizeof(gs_mps_chain_trace_root) == 7 * sizeof(uint64_t),
              "MPS chain trace root FFI layout changed");

static const uint32_t kGsMpsApiProfileMaxGraphDepth = 16;

struct GsMpsApiProfileGraphFrame {
  uint64_t serial;
  uint32_t phase;
  uint64_t baseNestedCount;
  uint64_t baseNestedWallNs;
  uint64_t baseNestedThreadCpuNs;
  uint64_t nestedCount;
  uint64_t nestedWallNs;
  uint64_t nestedThreadCpuNs;
};

struct GsMpsApiProfileData {
  gs_mps_api_profile_stats stats[GS_MPS_API_PROFILE_PHASE_CAPACITY]
                                [GS_MPS_API_PROFILE_OP_COUNT];
  GsMpsApiProfileGraphFrame graphFrames[kGsMpsApiProfileMaxGraphDepth];
  uint64_t activeSerials[kGsMpsApiProfileMaxGraphDepth];
  pthread_t owner;
  void *commandWrapper;
  void *currentRoot;
  uint64_t nextSerial;
  gs_mps_chain_trace_event *traceEvents;
  uint64_t traceOwnerThreadId;
  uint64_t traceTimebaseNumer;
  uint64_t traceTimebaseDenom;
  uint64_t traceStartMachTicks;
  uint64_t traceEventCount;
  uint64_t traceRootCount;
  uint64_t traceLastRootSequence;
  uint32_t phase;
  uint32_t graphDepth;
  uint32_t activeScopeDepth;
  bool enabled;
  bool failed;
  bool tlsAttached;
  bool traceEnabled;
  bool traceFailed;
};
static_assert(std::is_trivial<GsMpsApiProfileData>::value,
              "MPS API profile storage must remain POD for Objective-C ivars");

static thread_local GsMpsApiProfileData *gs_mps_api_profile_current = nullptr;

static bool gs_mps_api_profile_clock_ns(clockid_t clockId, uint64_t *value) {
  if (!value)
    return false;
  struct timespec ts;
  if (clock_gettime(clockId, &ts) != 0 || ts.tv_sec < 0 || ts.tv_nsec < 0 ||
      ts.tv_nsec >= 1000000000L)
    return false;
  const uint64_t seconds = (uint64_t)ts.tv_sec;
  const uint64_t nanos = (uint64_t)ts.tv_nsec;
  if (seconds > (UINT64_MAX - nanos) / 1000000000ULL)
    return false;
  *value = seconds * 1000000000ULL + nanos;
  return true;
}

static bool gs_mps_api_profile_add(uint64_t *target, uint64_t value) {
  if (!target || value > UINT64_MAX - *target)
    return false;
  *target += value;
  return true;
}

static bool gs_mps_api_profile_record(gs_mps_api_profile_stats *stats,
                                      uint64_t wallNs, uint64_t threadCpuNs,
                                      bool nested) {
  if (!stats || stats->count == UINT64_MAX ||
      (nested && stats->nested_count == UINT64_MAX))
    return false;
  gs_mps_api_profile_stats next = *stats;
  if (!gs_mps_api_profile_add(&next.wall_total_ns, wallNs) ||
      !gs_mps_api_profile_add(&next.thread_cpu_total_ns, threadCpuNs) ||
      !gs_mps_api_profile_add(&next.exclusive_wall_total_ns, wallNs) ||
      !gs_mps_api_profile_add(&next.exclusive_thread_cpu_total_ns,
                              threadCpuNs))
    return false;
  ++next.count;
  if (wallNs > next.wall_max_ns)
    next.wall_max_ns = wallNs;
  if (threadCpuNs > next.thread_cpu_max_ns)
    next.thread_cpu_max_ns = threadCpuNs;
  if (nested) {
    if (!gs_mps_api_profile_add(&next.nested_wall_total_ns, wallNs) ||
        !gs_mps_api_profile_add(&next.nested_thread_cpu_total_ns,
                                threadCpuNs))
      return false;
    ++next.nested_count;
  }
  *stats = next;
  return true;
}

static bool gs_mps_api_profile_add_graph_child(
    GsMpsApiProfileData *data, GsMpsApiProfileGraphFrame *frame,
    uint64_t wallNs, uint64_t threadCpuNs) {
  if (!data || !frame || frame->phase >= GS_MPS_API_PROFILE_PHASE_CAPACITY ||
      frame->nestedCount == UINT64_MAX)
    return false;
  GsMpsApiProfileGraphFrame nextFrame = *frame;
  gs_mps_api_profile_stats *stats =
      &data->stats[frame->phase][GS_MPS_API_PROFILE_OP_GRAPH_ENCODE];
  gs_mps_api_profile_stats nextStats = *stats;
  if (!gs_mps_api_profile_add(&nextFrame.nestedWallNs, wallNs) ||
      !gs_mps_api_profile_add(&nextFrame.nestedThreadCpuNs, threadCpuNs) ||
      !gs_mps_api_profile_add(&nextStats.nested_wall_total_ns, wallNs) ||
      !gs_mps_api_profile_add(&nextStats.nested_thread_cpu_total_ns,
                              threadCpuNs) ||
      nextStats.nested_count == UINT64_MAX)
    return false;
  ++nextFrame.nestedCount;
  ++nextStats.nested_count;
  *frame = nextFrame;
  *stats = nextStats;
  return true;
}

class GsMpsApiProfileScopeGuard {
public:
  GsMpsApiProfileScopeGuard(int32_t operation, void *commandContext) {
    gs_mps_api_profile_scope_begin(operation, commandContext, &scope_);
  }
  ~GsMpsApiProfileScopeGuard() {
    if (scope_.active)
      gs_mps_api_profile_scope_end(&scope_);
  }
  uint64_t serial() const { return scope_.serial; }
  uint64_t parentGraphSerial() const { return scope_.parent_graph_serial; }
  uint32_t phase() const { return scope_.phase; }
  uint32_t nestedInGraph() const { return scope_.nested_in_graph; }
  GsMpsApiProfileScopeGuard(const GsMpsApiProfileScopeGuard &) = delete;
  GsMpsApiProfileScopeGuard &operator=(const GsMpsApiProfileScopeGuard &) = delete;

private:
  gs_mps_api_profile_scope scope_{};
};

static void gs_mps_api_profile_fail(GsMpsApiProfileData *data) {
  if (data)
    data->failed = true;
}

static void gs_mps_chain_trace_fail(GsMpsApiProfileData *data) {
  if (!data)
    return;
  data->traceFailed = true;
  data->failed = true;
}

static bool gs_mps_chain_trace_event_valid(
    const GsMpsApiProfileData *data,
    const gs_mps_chain_trace_event *event) {
  const uint32_t validFlags =
      GS_MPS_CHAIN_TRACE_EVENT_NESTED_IN_GRAPH |
      GS_MPS_CHAIN_TRACE_EVENT_BEFORE_ROOT_VALID |
      GS_MPS_CHAIN_TRACE_EVENT_BEFORE_STATUS_AFTER_CALL_VALID |
      GS_MPS_CHAIN_TRACE_EVENT_AFTER_ROOT_VALID;
  if (!data || !event || !data->traceEnabled || data->traceFailed ||
      !data->traceEvents || event->serial == 0 || event->begin_mach_ticks == 0 ||
      event->end_mach_ticks <= event->begin_mach_ticks ||
      event->begin_thread_cpu_ns == 0 ||
      event->end_thread_cpu_ns < event->begin_thread_cpu_ns ||
      event->api_scope_serial == 0 ||
      event->phase >= GS_MPS_API_PROFILE_PHASE_COUNT ||
      event->graph_depth > kGsMpsApiProfileMaxGraphDepth ||
      (event->flags & ~validFlags) != 0 ||
      (event->flags & (GS_MPS_CHAIN_TRACE_EVENT_BEFORE_ROOT_VALID |
                       GS_MPS_CHAIN_TRACE_EVENT_BEFORE_STATUS_AFTER_CALL_VALID |
                       GS_MPS_CHAIN_TRACE_EVENT_AFTER_ROOT_VALID)) !=
          (GS_MPS_CHAIN_TRACE_EVENT_BEFORE_ROOT_VALID |
           GS_MPS_CHAIN_TRACE_EVENT_BEFORE_STATUS_AFTER_CALL_VALID |
           GS_MPS_CHAIN_TRACE_EVENT_AFTER_ROOT_VALID) ||
      event->before_root_sequence == UINT64_MAX ||
      event->after_root_sequence != event->before_root_sequence + 1 ||
      event->after_root_sequence >= data->traceRootCount)
    return false;
  const bool nested =
      (event->flags & GS_MPS_CHAIN_TRACE_EVENT_NESTED_IN_GRAPH) != 0;
  if (nested != (event->parent_graph_serial != 0) ||
      nested != (event->graph_depth != 0) ||
      (nested && event->parent_graph_serial >= event->api_scope_serial) ||
      event->before_root_status >
          (uint32_t)MTLCommandBufferStatusEnqueued ||
      event->before_root_status_after_call <
          (uint32_t)MTLCommandBufferStatusCommitted ||
      event->before_root_status_after_call >
          (uint32_t)MTLCommandBufferStatusCompleted ||
      event->after_root_status >
          (uint32_t)MTLCommandBufferStatusEnqueued)
    return false;
  return true;
}

static bool gs_mps_chain_trace_append(GsMpsApiProfileData *data,
                                      const gs_mps_chain_trace_event *event) {
  if (!data || !event ||
      data->traceEventCount >= GS_MPS_CHAIN_TRACE_EVENT_CAPACITY ||
      event->serial != data->traceEventCount + 1 ||
      event->after_root_sequence != data->traceLastRootSequence ||
      (data->traceEventCount == 0
           ? event->before_root_sequence != 0 ||
                 event->begin_mach_ticks < data->traceStartMachTicks
           : event->before_root_sequence !=
                     data->traceEvents[data->traceEventCount - 1]
                         .after_root_sequence ||
                 event->begin_mach_ticks <
                     data->traceEvents[data->traceEventCount - 1]
                         .end_mach_ticks ||
                 event->begin_thread_cpu_ns <
                     data->traceEvents[data->traceEventCount - 1]
                         .end_thread_cpu_ns ||
                 event->api_scope_serial <=
                     data->traceEvents[data->traceEventCount - 1]
                         .api_scope_serial) ||
      !gs_mps_chain_trace_event_valid(data, event)) {
    gs_mps_chain_trace_fail(data);
    return false;
  }
  data->traceEvents[data->traceEventCount++] = *event;
  return true;
}

extern "C" void gs_mps_api_profile_scope_begin(
    int32_t operation, void *commandContext,
    gs_mps_api_profile_scope *scope) {
  if (!scope)
    return;
  memset(scope, 0, sizeof(*scope));
  GsMpsApiProfileData *data = gs_mps_api_profile_current;
  // Keep the disabled path free of clock reads and all work beyond these
  // ordinary state checks. Optional bridges use weak no-op fallback hooks.
  if (!data || !data->enabled || !data->tlsAttached)
    return;
  void *expectedContext =
      operation == GS_MPS_API_PROFILE_OP_T5_LINEAR_ENCODE
          ? data->currentRoot
          : data->commandWrapper;
  // A second, unprofiled command chain can run on the same thread. Its
  // context must never be charged to this chain's collector.
  if (!commandContext || commandContext != expectedContext)
    return;
  if (!pthread_equal(data->owner, pthread_self()) ||
      gs_mps_api_profile_current != data || data->failed ||
      operation < 0 || operation >= GS_MPS_API_PROFILE_OP_COUNT ||
      data->phase >= GS_MPS_API_PROFILE_PHASE_CAPACITY ||
      data->activeScopeDepth >= kGsMpsApiProfileMaxGraphDepth ||
      data->nextSerial == UINT64_MAX) {
    gs_mps_api_profile_fail(data);
    return;
  }

  if ((operation == GS_MPS_API_PROFILE_OP_GRAPH_ENCODE &&
       (data->graphDepth != 0 || data->activeScopeDepth != 0)) ||
      (operation == GS_MPS_API_PROFILE_OP_T5_LINEAR_ENCODE &&
       (data->graphDepth != 0 || data->activeScopeDepth != 0)) ||
      (operation == GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE &&
       data->graphDepth == 0 && data->activeScopeDepth != 0)) {
    gs_mps_api_profile_fail(data);
    return;
  }

  uint64_t startWall = 0;
  uint64_t startThreadCpu = 0;
  if (!gs_mps_api_profile_clock_ns(CLOCK_MONOTONIC, &startWall) ||
      !gs_mps_api_profile_clock_ns(CLOCK_THREAD_CPUTIME_ID,
                                   &startThreadCpu)) {
    gs_mps_api_profile_fail(data);
    return;
  }

  scope->owner_data = data;
  scope->start_wall_ns = startWall;
  scope->start_thread_cpu_ns = startThreadCpu;
  scope->serial = ++data->nextSerial;
  scope->operation = (uint32_t)operation;
  scope->phase = data->phase;
  scope->clocks_valid = 1;
  if (operation == GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE &&
      data->graphDepth > 0) {
    GsMpsApiProfileGraphFrame *parent =
        &data->graphFrames[data->graphDepth - 1];
    if (parent->phase != data->phase || parent->serial == 0) {
      gs_mps_api_profile_fail(data);
      memset(scope, 0, sizeof(*scope));
      return;
    }
    scope->nested_in_graph = 1;
    scope->parent_graph_serial = parent->serial;
  }

  data->activeSerials[data->activeScopeDepth++] = scope->serial;
  scope->active = 1;
  if (operation == GS_MPS_API_PROFILE_OP_GRAPH_ENCODE) {
    if (data->graphDepth >= kGsMpsApiProfileMaxGraphDepth) {
      --data->activeScopeDepth;
      scope->active = 0;
      gs_mps_api_profile_fail(data);
      return;
    }
    GsMpsApiProfileGraphFrame *frame =
        &data->graphFrames[data->graphDepth++];
    memset(frame, 0, sizeof(*frame));
    frame->serial = scope->serial;
    frame->phase = scope->phase;
    const gs_mps_api_profile_stats &stats =
        data->stats[scope->phase][GS_MPS_API_PROFILE_OP_GRAPH_ENCODE];
    frame->baseNestedCount = stats.nested_count;
    frame->baseNestedWallNs = stats.nested_wall_total_ns;
    frame->baseNestedThreadCpuNs = stats.nested_thread_cpu_total_ns;
  }
}

extern "C" void gs_mps_api_profile_scope_end(
    gs_mps_api_profile_scope *scope) {
  if (!scope || !scope->active)
    return;
  GsMpsApiProfileData *data =
      static_cast<GsMpsApiProfileData *>(scope->owner_data);
  if (!data) {
    scope->active = 0;
    return;
  }
  if (gs_mps_api_profile_current != data || !data->tlsAttached ||
      !pthread_equal(data->owner, pthread_self()) || data->activeScopeDepth == 0 ||
      data->activeSerials[data->activeScopeDepth - 1] != scope->serial ||
      scope->phase >= GS_MPS_API_PROFILE_PHASE_CAPACITY ||
      scope->operation >= GS_MPS_API_PROFILE_OP_COUNT) {
    gs_mps_api_profile_fail(data);
    scope->active = 0;
    return;
  }

  uint64_t endWall = 0;
  uint64_t endThreadCpu = 0;
  bool clocksValid =
      scope->clocks_valid &&
      gs_mps_api_profile_clock_ns(CLOCK_MONOTONIC, &endWall) &&
      gs_mps_api_profile_clock_ns(CLOCK_THREAD_CPUTIME_ID, &endThreadCpu) &&
      endWall >= scope->start_wall_ns &&
      endThreadCpu >= scope->start_thread_cpu_ns;
  uint64_t wallNs = clocksValid ? endWall - scope->start_wall_ns : 0;
  uint64_t threadCpuNs =
      clocksValid ? endThreadCpu - scope->start_thread_cpu_ns : 0;
  if (!clocksValid)
    gs_mps_api_profile_fail(data);

  GsMpsApiProfileGraphFrame *frame = nullptr;
  if (scope->operation == GS_MPS_API_PROFILE_OP_GRAPH_ENCODE) {
    if (data->graphDepth == 0 ||
        data->graphFrames[data->graphDepth - 1].serial != scope->serial ||
        data->graphFrames[data->graphDepth - 1].phase != scope->phase) {
      gs_mps_api_profile_fail(data);
    } else {
      frame = &data->graphFrames[data->graphDepth - 1];
    }
  } else if (scope->nested_in_graph) {
    if (data->graphDepth == 0) {
      gs_mps_api_profile_fail(data);
    } else {
      frame = &data->graphFrames[data->graphDepth - 1];
      if (frame->serial != scope->parent_graph_serial ||
          frame->phase != scope->phase)
        gs_mps_api_profile_fail(data);
    }
  }

  if (clocksValid && !data->failed) {
    gs_mps_api_profile_stats *stats =
        &data->stats[scope->phase][scope->operation];
    if (!gs_mps_api_profile_record(
            stats, wallNs, threadCpuNs,
            scope->operation == GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE &&
                scope->nested_in_graph)) {
      gs_mps_api_profile_fail(data);
    } else if (scope->operation == GS_MPS_API_PROFILE_OP_GRAPH_ENCODE) {
      if (!frame ||
          frame->baseNestedCount > UINT64_MAX - frame->nestedCount ||
          frame->baseNestedWallNs > UINT64_MAX - frame->nestedWallNs ||
          frame->baseNestedThreadCpuNs >
              UINT64_MAX - frame->nestedThreadCpuNs ||
          frame->baseNestedCount + frame->nestedCount !=
              stats->nested_count ||
          frame->baseNestedWallNs + frame->nestedWallNs !=
              stats->nested_wall_total_ns ||
          frame->baseNestedThreadCpuNs + frame->nestedThreadCpuNs !=
              stats->nested_thread_cpu_total_ns ||
          frame->nestedWallNs > wallNs || frame->nestedThreadCpuNs > threadCpuNs ||
          stats->exclusive_wall_total_ns < frame->nestedWallNs ||
          stats->exclusive_thread_cpu_total_ns < frame->nestedThreadCpuNs) {
        gs_mps_api_profile_fail(data);
      } else {
        stats->exclusive_wall_total_ns -= frame->nestedWallNs;
        stats->exclusive_thread_cpu_total_ns -= frame->nestedThreadCpuNs;
      }
    } else if (scope->operation == GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE &&
               scope->nested_in_graph) {
      if (!frame || !gs_mps_api_profile_add_graph_child(
                        data, frame, wallNs, threadCpuNs))
        gs_mps_api_profile_fail(data);
    }
  }

  if (scope->operation == GS_MPS_API_PROFILE_OP_GRAPH_ENCODE) {
    if (data->graphDepth > 0 &&
        data->graphFrames[data->graphDepth - 1].serial == scope->serial) {
      memset(&data->graphFrames[data->graphDepth - 1], 0,
             sizeof(data->graphFrames[0]));
      --data->graphDepth;
    } else {
      gs_mps_api_profile_fail(data);
    }
  }
  --data->activeScopeDepth;
  data->activeSerials[data->activeScopeDepth] = 0;
  scope->active = 0;
}

static void gs_mps_api_profile_probe_burn(volatile uint64_t *value,
                                          uint64_t iterations) {
  uint64_t current = *value;
  for (uint64_t i = 0; i < iterations; ++i)
    current = current * 6364136223846793005ULL + 1442695040888963407ULL;
  *value = current;
}

extern "C" int32_t gs_mps_api_profile_cpu_probe(uint64_t *wallNs,
                                                 uint64_t *threadCpuNs) {
  if (!wallNs || !threadCpuNs)
    return -1;
  *wallNs = 0;
  *threadCpuNs = 0;
  if (gs_mps_api_profile_current != nullptr)
    return -6;

  // The disabled hook must be a state-only no-op and leave the stack empty.
  GsMpsApiProfileData disabled{};
  disabled.commandWrapper = &disabled;
  gs_mps_api_profile_current = &disabled;
  gs_mps_api_profile_scope disabledScope{};
  gs_mps_api_profile_scope_begin(GS_MPS_API_PROFILE_OP_GRAPH_ENCODE,
                                 disabled.commandWrapper, &disabledScope);
  gs_mps_api_profile_scope_end(&disabledScope);
  if (disabledScope.active || disabled.activeScopeDepth != 0 ||
      disabled.failed) {
    gs_mps_api_profile_current = nullptr;
    return -2;
  }

  GsMpsApiProfileData data{};
  data.owner = pthread_self();
  data.phase = GS_MPS_API_PROFILE_PHASE_SELF_ATTENTION;
  data.enabled = true;
  data.tlsAttached = true;
  volatile uint64_t burn = 1;
  uint64_t rootIdentity = 0;
  data.commandWrapper = &data;
  data.currentRoot = &rootIdentity;
  gs_mps_api_profile_current = &data;
  gs_mps_api_profile_scope graph{};
  gs_mps_api_profile_scope commit{};
  gs_mps_api_profile_scope otherChain{};
  gs_mps_api_profile_scope_begin(GS_MPS_API_PROFILE_OP_GRAPH_ENCODE,
                                 &disabled, &otherChain);
  if (otherChain.active || data.failed || data.activeScopeDepth != 0) {
    gs_mps_api_profile_current = nullptr;
    return -3;
  }
  gs_mps_api_profile_scope_begin(GS_MPS_API_PROFILE_OP_GRAPH_ENCODE,
                                 data.commandWrapper, &graph);
  if (!graph.active) {
    gs_mps_api_profile_current = nullptr;
    return -4;
  }
  gs_mps_api_profile_probe_burn(&burn, 100000);
  gs_mps_api_profile_scope_begin(GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE,
                                 data.commandWrapper, &commit);
  if (!commit.active || !commit.nested_in_graph ||
      commit.parent_graph_serial != graph.serial) {
    gs_mps_api_profile_current = nullptr;
    return -5;
  }
  gs_mps_api_profile_probe_burn(&burn, 100000);
  gs_mps_api_profile_scope_end(&commit);
  gs_mps_api_profile_probe_burn(&burn, 100000);
  gs_mps_api_profile_scope_end(&graph);
  gs_mps_api_profile_current = nullptr;

  const gs_mps_api_profile_stats &graphStats =
      data.stats[GS_MPS_API_PROFILE_PHASE_SELF_ATTENTION]
                [GS_MPS_API_PROFILE_OP_GRAPH_ENCODE];
  const gs_mps_api_profile_stats &commitStats =
      data.stats[GS_MPS_API_PROFILE_PHASE_SELF_ATTENTION]
                [GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE];
  if (data.failed || data.graphDepth != 0 || data.activeScopeDepth != 0 ||
      graphStats.count != 1 || commitStats.count != 1 ||
      graphStats.nested_count != 1 || commitStats.nested_count != 1 ||
      graphStats.nested_wall_total_ns != commitStats.wall_total_ns ||
      graphStats.nested_thread_cpu_total_ns !=
          commitStats.thread_cpu_total_ns ||
      graphStats.wall_total_ns <= graphStats.nested_wall_total_ns ||
      graphStats.thread_cpu_total_ns <=
          graphStats.nested_thread_cpu_total_ns ||
      graphStats.exclusive_wall_total_ns !=
          graphStats.wall_total_ns - graphStats.nested_wall_total_ns ||
      graphStats.exclusive_thread_cpu_total_ns !=
          graphStats.thread_cpu_total_ns -
              graphStats.nested_thread_cpu_total_ns ||
      commitStats.exclusive_wall_total_ns != commitStats.wall_total_ns ||
      commitStats.exclusive_thread_cpu_total_ns !=
          commitStats.thread_cpu_total_ns ||
      graphStats.thread_cpu_total_ns == 0 || graphStats.wall_total_ns == 0 ||
      commitStats.thread_cpu_total_ns == 0 || commitStats.wall_total_ns == 0)
    return -7;
  *wallNs = graphStats.wall_total_ns;
  *threadCpuNs = graphStats.thread_cpu_total_ns;
  return 0;
}

extern "C" int32_t gs_mps_chain_api_trace_self_test(
    uint64_t *positiveEvents, uint64_t *rejectedControls) {
  if (!positiveEvents || !rejectedControls)
    return -1;
  *positiveEvents = 0;
  *rejectedControls = 0;
  if (gs_mps_api_profile_current != nullptr)
    return -2;

  gs_mps_chain_trace_event eventStorage[1]{};
  GsMpsApiProfileData data{};
  data.owner = pthread_self();
  data.traceEvents = eventStorage;
  data.traceEnabled = true;
  data.traceRootCount = 2;
  data.traceLastRootSequence = 1;

  gs_mps_chain_trace_event event{};
  event.serial = 1;
  event.begin_mach_ticks = mach_absolute_time();
  if (event.begin_mach_ticks == 0 ||
      !gs_mps_api_profile_clock_ns(CLOCK_THREAD_CPUTIME_ID,
                                   &event.begin_thread_cpu_ns))
    return -3;
  volatile uint64_t burn = 1;
  gs_mps_api_profile_probe_burn(&burn, 100000);
  event.end_mach_ticks = mach_absolute_time();
  if (event.end_mach_ticks == 0 ||
      !gs_mps_api_profile_clock_ns(CLOCK_THREAD_CPUTIME_ID,
                                   &event.end_thread_cpu_ns))
    return -4;
  event.api_scope_serial = 7;
  event.parent_graph_serial = 6;
  event.before_root_sequence = 0;
  event.after_root_sequence = 1;
  event.phase = GS_MPS_API_PROFILE_PHASE_SELF_ATTENTION;
  event.graph_depth = 1;
  event.before_root_status = MTLCommandBufferStatusEnqueued;
  event.before_root_status_after_call = MTLCommandBufferStatusCommitted;
  event.after_root_status = MTLCommandBufferStatusEnqueued;
  event.flags = GS_MPS_CHAIN_TRACE_EVENT_NESTED_IN_GRAPH |
                GS_MPS_CHAIN_TRACE_EVENT_BEFORE_ROOT_VALID |
                GS_MPS_CHAIN_TRACE_EVENT_BEFORE_STATUS_AFTER_CALL_VALID |
                GS_MPS_CHAIN_TRACE_EVENT_AFTER_ROOT_VALID;
  if (!gs_mps_chain_trace_append(&data, &event) ||
      data.traceEventCount != 1 || data.traceFailed)
    return -5;

  uint64_t rejected = 0;
  gs_mps_chain_trace_event malformed = event;
  malformed.end_mach_ticks = malformed.begin_mach_ticks - 1;
  if (gs_mps_chain_trace_event_valid(&data, &malformed))
    return -6;
  ++rejected;
  malformed = event;
  malformed.end_thread_cpu_ns = malformed.begin_thread_cpu_ns - 1;
  if (gs_mps_chain_trace_event_valid(&data, &malformed))
    return -7;
  ++rejected;

  GsMpsApiProfileData overflow{};
  overflow.owner = pthread_self();
  overflow.traceEvents = eventStorage;
  overflow.traceEnabled = true;
  overflow.traceRootCount = 2;
  overflow.traceLastRootSequence = 1;
  overflow.traceEventCount = GS_MPS_CHAIN_TRACE_EVENT_CAPACITY;
  malformed = event;
  malformed.serial = GS_MPS_CHAIN_TRACE_EVENT_CAPACITY + 1;
  if (gs_mps_chain_trace_append(&overflow, &malformed) ||
      !overflow.traceFailed || !overflow.failed)
    return -8;
  ++rejected;

  if (rejected != 3)
    return -9;
  *positiveEvents = data.traceEventCount;
  *rejectedControls = rejected;
  return 0;
}

static NSString *gs_mps_chain_safe_log_text(NSString *value,
                                            NSUInteger maxCharacters) {
  if (!value)
    return @"none";
  NSUInteger count =
      value.length < maxCharacters ? value.length : maxCharacters;
  NSMutableString *safe = [NSMutableString stringWithCapacity:count + 3];
  for (NSUInteger i = 0; i < count; ++i) {
    unichar character = [value characterAtIndex:i];
    if (character < 0x20 || character == 0x7f)
      character = ' ';
    else if (character == '"' || character == '\\')
      character = '_';
    [safe appendFormat:@"%C", character];
  }
  if (value.length > count)
    [safe appendString:@"..."];
  return safe;
}

// A summary has no strong reference to its command root. The tracker keeps a
// root strongly only while its completion callback is pending. Summary storage
// remains O(number of roots), but completed Metal roots can be reclaimed.
@interface CogniMPSCommandRootSummary : NSObject
@property(nonatomic) NSUInteger sequence;
@property(nonatomic) MTLCommandBufferStatus status;
@property(nonatomic) double gpuStartTime;
@property(nonatomic) double gpuEndTime;
@property(nonatomic) BOOL hasError;
@property(nonatomic) NSInteger errorCode;
@property(nonatomic, copy) NSString *errorDomain;
@property(nonatomic, copy) NSString *errorDescription;
@property(nonatomic) BOOL recorded;
@property(nonatomic) BOOL traceCaptured;
@property(nonatomic) uint64_t traceTrackedOrObservedMachTicks;
@property(nonatomic) uint64_t traceCompletionObservedMachTicks;
@property(nonatomic) MTLCommandBufferStatus traceStatusAtTrackOrEnable;
@property(nonatomic) uint32_t traceCompletionSource;
@end
@implementation CogniMPSCommandRootSummary
@end

@interface CogniMPSPendingRoot : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> root;
@end
@implementation CogniMPSPendingRoot
@end

@interface CogniMPSCommandRootTracker : NSObject
{
@private
  std::atomic<bool> _traceEnabled;
}
@property(nonatomic, strong) NSLock *lock;
@property(nonatomic, strong) NSMutableArray<CogniMPSCommandRootSummary *> *summaries;
// Strong pointer-identity keys own roots only until completion is observed.
@property(nonatomic, strong) NSMapTable *pendingRoots;
// Weak pointer-identity membership prevents duplicate handler registration
// without keeping any completed root alive.
@property(nonatomic, strong) NSHashTable *seenRoots;
@property(nonatomic) NSUInteger maxPendingRoots;
@property(nonatomic) NSUInteger callbackRecords;
@property(nonatomic) NSUInteger terminalFallbackRecords;
- (NSUInteger)trackRoot:(id<MTLCommandBuffer>)root;
- (BOOL)enableTraceForRoot:(id<MTLCommandBuffer>)root
           observationTicks:(uint64_t *)ticks;
- (void)disableTraceCapture;
- (void)recordCompletedRoot:(id<MTLCommandBuffer>)root
          terminalFallback:(BOOL)terminalFallback;
- (NSArray<CogniMPSCommandRootSummary *> *)finishPendingRoots;
- (NSArray<CogniMPSCommandRootSummary *> *)summariesSnapshot;
- (CogniMPSCommandRootSummary *)summaryAtSequence:(NSUInteger)sequence;
- (NSUInteger)pendingCount;
- (NSUInteger)summaryCount;
@end

@implementation CogniMPSCommandRootTracker
- (instancetype)init {
  self = [super init];
  if (self) {
    _traceEnabled.store(false, std::memory_order_relaxed);
    self.lock = [NSLock new];
    self.summaries = [NSMutableArray new];
    self.pendingRoots = [[NSMapTable alloc]
        initWithKeyOptions:NSPointerFunctionsStrongMemory |
                      NSPointerFunctionsObjectPointerPersonality
            valueOptions:NSPointerFunctionsStrongMemory
                capacity:1];
    self.seenRoots = [[NSHashTable alloc]
        initWithOptions:NSPointerFunctionsWeakMemory |
                   NSPointerFunctionsObjectPointerPersonality
              capacity:1];
  }
  return self;
}

- (NSUInteger)trackRoot:(id<MTLCommandBuffer>)root {
  if (!root)
    return NSNotFound;

  [self.lock lock];
  if ([self.seenRoots containsObject:root]) {
    [self.lock unlock];
    return NSNotFound;
  }

  [self.seenRoots addObject:root];
  CogniMPSCommandRootSummary *summary = [CogniMPSCommandRootSummary new];
  summary.sequence = self.summaries.count;
  if (_traceEnabled.load(std::memory_order_acquire)) {
    summary.traceCaptured = YES;
    summary.traceTrackedOrObservedMachTicks = mach_absolute_time();
    summary.traceStatusAtTrackOrEnable = root.status;
  }
  [self.summaries addObject:summary];
  [self.pendingRoots setObject:summary forKey:root];
  NSUInteger pendingCount = self.pendingRoots.count;
  if (pendingCount > self.maxPendingRoots)
    self.maxPendingRoots = pendingCount;
  [self.lock unlock];

  // Register before any caller can commit this root. Capture the tracker
  // weakly: a command buffer retains its completion block, and that block
  // must not form a root -> block -> tracker -> pending-root cycle.
  __weak CogniMPSCommandRootTracker *weakTracker = self;
  [root addCompletedHandler:^(id<MTLCommandBuffer> completedRoot) {
    CogniMPSCommandRootTracker *tracker = weakTracker;
    [tracker recordCompletedRoot:completedRoot terminalFallback:NO];
  }];
  return summary.sequence;
}

- (BOOL)enableTraceForRoot:(id<MTLCommandBuffer>)root
           observationTicks:(uint64_t *)ticks {
  if (!root || !ticks)
    return NO;
  *ticks = 0;
  [self.lock lock];
  CogniMPSCommandRootSummary *summary =
      self.summaries.count == 1 ? self.summaries.firstObject : nil;
  if (!summary || summary.recorded || summary.traceCaptured ||
      [self.pendingRoots objectForKey:root] != summary) {
    [self.lock unlock];
    return NO;
  }
  const MTLCommandBufferStatus status = root.status;
  const uint64_t observedTicks = mach_absolute_time();
  if (observedTicks == 0) {
    [self.lock unlock];
    return NO;
  }
  summary.traceCaptured = YES;
  summary.traceTrackedOrObservedMachTicks = observedTicks;
  summary.traceStatusAtTrackOrEnable = status;
  *ticks = observedTicks;
  _traceEnabled.store(true, std::memory_order_release);
  [self.lock unlock];
  return YES;
}

- (void)disableTraceCapture {
  _traceEnabled.store(false, std::memory_order_release);
}

- (void)recordCompletedRoot:(id<MTLCommandBuffer>)root
          terminalFallback:(BOOL)terminalFallback {
  if (!root)
    return;

  // This is sampled at entry to the existing completion callback (or the
  // existing post-wait fallback). The atomic flag keeps the default path free
  // of clock reads and does not add a callback or tracker lock.
  uint64_t traceCompletionTicks = 0;
  if (_traceEnabled.load(std::memory_order_acquire))
    traceCompletionTicks = mach_absolute_time();

  [self.lock lock];
  CogniMPSCommandRootSummary *summary =
      [self.pendingRoots objectForKey:root];
  // Callback and post-wait fallback can race. Removing the strong map entry
  // while holding the same lock makes the first recorder authoritative.
  if (!summary || summary.recorded) {
    [self.lock unlock];
    return;
  }

  summary.status = root.status;
  summary.gpuStartTime = root.GPUStartTime;
  summary.gpuEndTime = root.GPUEndTime;
  if (summary.traceCaptured) {
    summary.traceCompletionObservedMachTicks = traceCompletionTicks;
    summary.traceCompletionSource = terminalFallback
                                        ? GS_MPS_CHAIN_TRACE_COMPLETION_TERMINAL_FALLBACK
                                        : GS_MPS_CHAIN_TRACE_COMPLETION_CALLBACK;
  }
  NSError *error = root.error;
  summary.hasError = error != nil;
  if (error) {
    summary.errorCode = error.code;
    summary.errorDomain = gs_mps_chain_safe_log_text(error.domain, 96);
    summary.errorDescription =
        gs_mps_chain_safe_log_text(error.localizedDescription, 256);
  }
  summary.recorded = YES;
  if (terminalFallback)
    ++self.terminalFallbackRecords;
  else
    ++self.callbackRecords;
  [self.pendingRoots removeObjectForKey:root];
  [self.lock unlock];
}

- (NSArray<CogniMPSCommandRootSummary *> *)finishPendingRoots {
  // Normally completion handlers empty this map before the terminal wait
  // returns. Keep a defensive fallback for handler delivery races. The
  // snapshot adds references only to roots already strongly held by the map;
  // each successful record immediately removes its map ownership.
  [self.lock lock];
  NSMutableArray<CogniMPSPendingRoot *> *pending = [NSMutableArray new];
  NSEnumerator *keys = self.pendingRoots.keyEnumerator;
  id<MTLCommandBuffer> root = nil;
  while ((root = [keys nextObject])) {
    CogniMPSCommandRootSummary *summary =
        [self.pendingRoots objectForKey:root];
    if (!summary)
      continue;
    CogniMPSPendingRoot *entry = [CogniMPSPendingRoot new];
    entry.root = root;
    [pending addObject:entry];
  }
  [self.lock unlock];

  for (CogniMPSPendingRoot *entry in pending)
    [self recordCompletedRoot:entry.root terminalFallback:YES];
  // Called only after the chain's terminal wait, when no further roots can
  // be registered. All pending roots are now recorded and late callbacks are
  // no-ops, so return the existing ordered storage without an O(N) copy.
  [self.lock lock];
  NSArray<CogniMPSCommandRootSummary *> *finishedSummaries = self.summaries;
  [self.lock unlock];
  return finishedSummaries;
}

- (NSArray<CogniMPSCommandRootSummary *> *)summariesSnapshot {
  [self.lock lock];
  NSArray<CogniMPSCommandRootSummary *> *copy = [self.summaries copy];
  [self.lock unlock];
  return copy;
}

- (CogniMPSCommandRootSummary *)summaryAtSequence:(NSUInteger)sequence {
  [self.lock lock];
  CogniMPSCommandRootSummary *summary =
      sequence < self.summaries.count ? self.summaries[sequence] : nil;
  [self.lock unlock];
  return summary;
}

- (NSUInteger)pendingCount {
  [self.lock lock];
  NSUInteger count = self.pendingRoots.count;
  [self.lock unlock];
  return count;
}

- (NSUInteger)summaryCount {
  [self.lock lock];
  NSUInteger count = self.summaries.count;
  [self.lock unlock];
  return count;
}
@end

@interface CogniTrackedMPSCommandBuffer : MPSCommandBuffer
@property(nonatomic, strong) CogniMPSCommandRootTracker *rootTracker;
@property(nonatomic, strong) id<MTLCommandBuffer> lastRoot;
@property(nonatomic, assign) GsMpsApiProfileData *profileData;
@property(nonatomic) NSUInteger lastRootSequence;
- (void)rememberCurrentRoot;
@end

@implementation CogniTrackedMPSCommandBuffer
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
  self = [super initWithCommandBuffer:commandBuffer];
  if (self) {
    self.rootTracker = [CogniMPSCommandRootTracker new];
    [self rememberCurrentRoot];
  }
  return self;
}
- (void)rememberCurrentRoot {
  id<MTLCommandBuffer> root = self.rootCommandBuffer;
  if (root == self.lastRoot)
    return;
  self.lastRoot = root;
  if (self.profileData)
    self.profileData->currentRoot = (__bridge void *)root;
  NSUInteger sequence = [self.rootTracker trackRoot:root];
  self.lastRootSequence = sequence;
  GsMpsApiProfileData *profile = self.profileData;
  if (profile && profile->traceEnabled) {
    if (sequence == NSNotFound || sequence != profile->traceRootCount) {
      gs_mps_chain_trace_fail(profile);
    } else {
      ++profile->traceRootCount;
      profile->traceLastRootSequence = sequence;
    }
  }
}
- (void)commitAndContinue {
  [self rememberCurrentRoot];
  GsMpsApiProfileData *traceProfile = self.profileData;
  const bool traceActive = traceProfile && traceProfile->traceEnabled;
  gs_mps_chain_trace_event traceEvent{};
  id<MTLCommandBuffer> priorRoot = self.lastRoot;
  if (traceActive) {
    if (traceProfile->traceFailed || traceProfile->failed ||
        !pthread_equal(traceProfile->owner, pthread_self()) || !priorRoot ||
        self.lastRootSequence == NSNotFound ||
        self.lastRootSequence >= traceProfile->traceRootCount) {
      gs_mps_chain_trace_fail(traceProfile);
    } else {
      traceEvent.serial = traceProfile->traceEventCount + 1;
      traceEvent.before_root_sequence = self.lastRootSequence;
      traceEvent.before_root_status = (uint32_t)priorRoot.status;
      traceEvent.flags |= GS_MPS_CHAIN_TRACE_EVENT_BEFORE_ROOT_VALID;
    }
  }
  {
    GsMpsApiProfileScopeGuard profileScope(
        GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE, (__bridge void *)self);
    bool traceStartValid = false;
    if (traceActive && !traceProfile->traceFailed) {
      traceEvent.phase = profileScope.phase();
      traceEvent.graph_depth = traceProfile->graphDepth;
      traceEvent.api_scope_serial = profileScope.serial();
      traceEvent.parent_graph_serial = profileScope.parentGraphSerial();
      if (profileScope.nestedInGraph())
        traceEvent.flags |= GS_MPS_CHAIN_TRACE_EVENT_NESTED_IN_GRAPH;
      traceEvent.begin_mach_ticks = mach_absolute_time();
      traceStartValid =
          traceEvent.begin_mach_ticks != 0 &&
          gs_mps_api_profile_clock_ns(CLOCK_THREAD_CPUTIME_ID,
                                      &traceEvent.begin_thread_cpu_ns);
      if (!traceStartValid || traceEvent.api_scope_serial == 0)
        gs_mps_chain_trace_fail(traceProfile);
    }
    [super commitAndContinue];
    if (traceActive && traceStartValid) {
      traceEvent.end_mach_ticks = mach_absolute_time();
      traceEvent.before_root_status_after_call = (uint32_t)priorRoot.status;
      traceEvent.flags |=
          GS_MPS_CHAIN_TRACE_EVENT_BEFORE_STATUS_AFTER_CALL_VALID;
      if (traceEvent.end_mach_ticks == 0 ||
          !gs_mps_api_profile_clock_ns(CLOCK_THREAD_CPUTIME_ID,
                                       &traceEvent.end_thread_cpu_ns))
        gs_mps_chain_trace_fail(traceProfile);
    }
  }
  [self rememberCurrentRoot];
  if (traceActive && !traceProfile->traceFailed) {
    id<MTLCommandBuffer> afterRoot = self.lastRoot;
    if (!afterRoot || self.lastRootSequence == NSNotFound) {
      gs_mps_chain_trace_fail(traceProfile);
    } else {
      traceEvent.after_root_sequence = self.lastRootSequence;
      traceEvent.after_root_status = (uint32_t)afterRoot.status;
      traceEvent.flags |= GS_MPS_CHAIN_TRACE_EVENT_AFTER_ROOT_VALID;
      if (!gs_mps_chain_trace_append(traceProfile, &traceEvent))
        gs_mps_chain_trace_fail(traceProfile);
    }
  }
}
@end

@interface CogniMPSCommandChain : NSObject
{
@public
  GsMpsApiProfileData apiProfile;
}
@property(nonatomic, strong) CogniTrackedMPSCommandBuffer *command;
@property(nonatomic) BOOL finished;
@property(nonatomic) int32_t result;
@property(nonatomic) double gpuSeconds;
@end
@implementation CogniMPSCommandChain
@end

extern "C" int32_t gs_mps_chain_api_profile_enable(void *handle) {
  if (!handle)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  GsMpsApiProfileData *data = &chain->apiProfile;
  if (chain.finished || data->enabled || data->failed ||
      data->activeScopeDepth != 0 || data->graphDepth != 0)
    return -2;
  if (gs_mps_api_profile_current != nullptr)
    return -3;
  if (chain.command.rootTracker.summaryCount != 1)
    return -4;

  uint64_t wallPreflight = 0;
  uint64_t threadCpuPreflight = 0;
  if (!gs_mps_api_profile_clock_ns(CLOCK_MONOTONIC, &wallPreflight) ||
      !gs_mps_api_profile_clock_ns(CLOCK_THREAD_CPUTIME_ID,
                                   &threadCpuPreflight)) {
    data->failed = true;
    return -5;
  }

  data->owner = pthread_self();
  data->phase = GS_MPS_API_PROFILE_PHASE_UNASSIGNED;
  data->commandWrapper = (__bridge void *)chain.command;
  data->currentRoot = (__bridge void *)chain.command.rootCommandBuffer;
  data->enabled = true;
  data->tlsAttached = true;
  chain.command.profileData = data;
  gs_mps_api_profile_current = data;
  return 0;
}

extern "C" int32_t gs_mps_chain_api_profile_set_phase(void *handle,
                                                        int32_t phase) {
  if (!handle || phase < 0 || phase >= GS_MPS_API_PROFILE_PHASE_COUNT)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  GsMpsApiProfileData *data = &chain->apiProfile;
  if (chain.finished || !data->enabled || !data->tlsAttached || data->failed ||
      gs_mps_api_profile_current != data ||
      !pthread_equal(data->owner, pthread_self()))
    return -2;
  if (data->activeScopeDepth != 0 || data->graphDepth != 0)
    return -3;
  data->phase = (uint32_t)phase;
  return 0;
}

extern "C" int32_t gs_mps_chain_api_profile_snapshot(
    void *handle, int32_t phase, int32_t operation,
    gs_mps_api_profile_stats *stats) {
  if (!handle || !stats || phase < 0 || phase >= GS_MPS_API_PROFILE_PHASE_COUNT ||
      operation < 0 || operation >= GS_MPS_API_PROFILE_OP_COUNT)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  GsMpsApiProfileData *data = &chain->apiProfile;
  if (!chain.finished || !data->enabled || data->tlsAttached ||
      data->failed || data->activeScopeDepth != 0 || data->graphDepth != 0)
    return -2;
  *stats = data->stats[phase][operation];
  return 0;
}

extern "C" int32_t gs_mps_chain_api_trace_enable(void *handle) {
  if (!handle)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  GsMpsApiProfileData *data = &chain->apiProfile;
  if (chain.finished || !data->enabled || !data->tlsAttached || data->failed ||
      data->traceEnabled || data->activeScopeDepth != 0 ||
      data->graphDepth != 0 || gs_mps_api_profile_current != data ||
      !pthread_equal(data->owner, pthread_self()))
    return -2;
  if (chain.command.rootTracker.summaryCount != 1 ||
      chain.command.lastRootSequence != 0 ||
      chain.command.rootCommandBuffer == nil)
    return -3;

  mach_timebase_info_data_t timebase{};
  timespec monotonicResolution{};
  uint64_t ownerThreadId = 0;
  uint64_t threadCpuPreflight = 0;
  if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.numer == 0 ||
      timebase.denom == 0 ||
      clock_getres(CLOCK_MONOTONIC, &monotonicResolution) != 0 ||
      monotonicResolution.tv_sec != 0 || monotonicResolution.tv_nsec <= 0 ||
      monotonicResolution.tv_nsec > 1000 ||
      pthread_threadid_np(pthread_self(), &ownerThreadId) != 0 ||
      ownerThreadId == 0 ||
      !gs_mps_api_profile_clock_ns(CLOCK_THREAD_CPUTIME_ID,
                                   &threadCpuPreflight))
    return -4;

  gs_mps_chain_trace_event *events = static_cast<gs_mps_chain_trace_event *>(
      calloc(GS_MPS_CHAIN_TRACE_EVENT_CAPACITY, sizeof(*events)));
  if (!events)
    return -5;

  data->traceEvents = events;
  data->traceOwnerThreadId = ownerThreadId;
  data->traceTimebaseNumer = timebase.numer;
  data->traceTimebaseDenom = timebase.denom;
  data->traceStartMachTicks = 0;
  data->traceRootCount = 1;
  data->traceLastRootSequence = 0;
  uint64_t traceStartTicks = 0;
  if (![chain.command.rootTracker
          enableTraceForRoot:chain.command.rootCommandBuffer
           observationTicks:&traceStartTicks]) {
    free(data->traceEvents);
    data->traceEvents = nullptr;
    data->traceOwnerThreadId = 0;
    data->traceTimebaseNumer = 0;
    data->traceTimebaseDenom = 0;
    data->traceStartMachTicks = 0;
    data->traceRootCount = 0;
    data->traceLastRootSequence = 0;
    return -6;
  }
  data->traceStartMachTicks = traceStartTicks;
  data->traceEnabled = true;
  return 0;
}

extern "C" int32_t gs_mps_chain_api_trace_info(
    void *handle, gs_mps_chain_trace_info *info) {
  if (!handle || !info)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  GsMpsApiProfileData *data = &chain->apiProfile;
  if (!chain.finished || !data->enabled || data->tlsAttached ||
      !data->traceEnabled || data->traceFailed || data->failed ||
      data->activeScopeDepth != 0 || data->graphDepth != 0 ||
      !data->traceEvents || data->traceEventCount >
                                GS_MPS_CHAIN_TRACE_EVENT_CAPACITY ||
      data->traceRootCount == 0 ||
      data->traceRootCount != data->traceEventCount + 1 ||
      data->traceLastRootSequence != data->traceEventCount ||
      data->traceTimebaseNumer == 0 || data->traceTimebaseDenom == 0 ||
      data->traceStartMachTicks == 0)
    return -2;
  const NSUInteger rootCount = chain.command.rootTracker.summaryCount;
  if (rootCount != data->traceRootCount ||
      rootCount > GS_MPS_CHAIN_TRACE_EVENT_CAPACITY + 1)
    return -3;
  info->owner_thread_id = data->traceOwnerThreadId;
  info->mach_timebase_numer = data->traceTimebaseNumer;
  info->mach_timebase_denom = data->traceTimebaseDenom;
  info->trace_start_mach_ticks = data->traceStartMachTicks;
  info->event_capacity = GS_MPS_CHAIN_TRACE_EVENT_CAPACITY;
  info->event_count = data->traceEventCount;
  info->root_count = data->traceRootCount;
  info->failed = data->traceFailed ? 1 : 0;
  return 0;
}

extern "C" int32_t gs_mps_chain_api_trace_event(
    void *handle, uint64_t index, gs_mps_chain_trace_event *event) {
  if (!handle || !event)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  GsMpsApiProfileData *data = &chain->apiProfile;
  if (!chain.finished || !data->traceEnabled || data->traceFailed ||
      data->failed || !data->traceEvents ||
      data->traceEventCount > GS_MPS_CHAIN_TRACE_EVENT_CAPACITY ||
      data->traceRootCount != data->traceEventCount + 1 ||
      data->traceLastRootSequence != data->traceEventCount)
    return -2;
  if (index >= data->traceEventCount)
    return -3;
  *event = data->traceEvents[index];
  if (!gs_mps_chain_trace_event_valid(data, event) ||
      event->serial != index + 1 ||
      (index == 0 ? event->before_root_sequence != 0
                  : data->traceEvents[index - 1].after_root_sequence !=
                        event->before_root_sequence) ||
      (index + 1 < data->traceEventCount
           ? data->traceEvents[index + 1].before_root_sequence !=
                 event->after_root_sequence
           : event->after_root_sequence != data->traceLastRootSequence))
    return -4;
  return 0;
}

extern "C" int32_t gs_mps_chain_api_trace_root(
    void *handle, uint64_t index, gs_mps_chain_trace_root *root) {
  if (!handle || !root)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  GsMpsApiProfileData *data = &chain->apiProfile;
  if (!chain.finished || !data->traceEnabled || data->traceFailed ||
      data->failed || index >= data->traceRootCount ||
      data->traceRootCount != data->traceEventCount + 1 ||
      data->traceLastRootSequence != data->traceEventCount)
    return -2;
  CogniMPSCommandRootSummary *summary =
      [chain.command.rootTracker summaryAtSequence:(NSUInteger)index];
  if (!summary || summary.sequence != (NSUInteger)index ||
      !summary.traceCaptured || !summary.recorded ||
      summary.traceTrackedOrObservedMachTicks == 0 ||
      summary.traceCompletionObservedMachTicks == 0 ||
      summary.traceCompletionObservedMachTicks <
          summary.traceTrackedOrObservedMachTicks ||
      summary.traceCompletionSource <
          GS_MPS_CHAIN_TRACE_COMPLETION_CALLBACK ||
      summary.traceCompletionSource >
          GS_MPS_CHAIN_TRACE_COMPLETION_TERMINAL_FALLBACK ||
      summary.status != MTLCommandBufferStatusCompleted || summary.hasError ||
      !std::isfinite(summary.gpuStartTime) ||
      !std::isfinite(summary.gpuEndTime) || summary.gpuStartTime <= 0.0 ||
      summary.gpuEndTime <= summary.gpuStartTime)
    return -3;

  root->sequence = summary.sequence;
  root->tracked_or_observed_mach_ticks =
      summary.traceTrackedOrObservedMachTicks;
  root->completion_observed_mach_ticks =
      summary.traceCompletionObservedMachTicks;
  const double gpuStartTime = summary.gpuStartTime;
  const double gpuEndTime = summary.gpuEndTime;
  memcpy(&root->gpu_start_time_bits, &gpuStartTime,
         sizeof(root->gpu_start_time_bits));
  memcpy(&root->gpu_end_time_bits, &gpuEndTime,
         sizeof(root->gpu_end_time_bits));
  root->status_at_track_or_enable =
      (uint32_t)summary.traceStatusAtTrackOrEnable;
  root->final_status = (uint32_t)summary.status;
  root->has_error = summary.hasError ? 1 : 0;
  root->completion_source = summary.traceCompletionSource;
  return 0;
}

extern "C" void *gs_mps_chain_create(void *queueHandle) {
  if (!queueHandle)
    return nullptr;
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queueHandle;
    CogniTrackedMPSCommandBuffer *command =
        [[CogniTrackedMPSCommandBuffer alloc]
            initWithCommandBuffer:[queue commandBuffer]];
    if (!command)
      return nullptr;
    CogniMPSCommandChain *chain = [CogniMPSCommandChain new];
    chain.command = command;
    return (__bridge_retained void *)chain;
  }
}

extern "C" void *gs_mps_chain_wrapper(void *handle) {
  if (!handle)
    return nullptr;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  return chain.finished ? nullptr : (__bridge void *)chain.command;
}

extern "C" void *gs_mps_chain_current_root(void *handle) {
  if (!handle)
    return nullptr;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  [chain.command rememberCurrentRoot];
  return chain.finished ? nullptr
                        : (__bridge void *)chain.command.rootCommandBuffer;
}

extern "C" int32_t gs_mps_chain_continue(void *handle) {
  if (!handle)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  if (chain.finished)
    return -1;
  [chain.command commitAndContinue];
  return 0;
}

// Cursor sampling observes the current root identity but does not commit,
// continue, encode, allocate a Metal resource, or wait for GPU work.
extern "C" int32_t gs_mps_chain_root_count(void *handle, int32_t *count) {
  if (!handle || !count)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  if (chain.finished)
    return -2;
  [chain.command rememberCurrentRoot];
  NSUInteger rootCount = chain.command.rootTracker.summaryCount;
  if (rootCount > (NSUInteger)INT32_MAX)
    return -3;
  *count = (int32_t)rootCount;
  return 0;
}

// Timing summaries are deliberately exposed only after the existing terminal
// wait. This is a read of compact CPU metadata, not an additional sync point.
extern "C" int32_t gs_mps_chain_root_timing(void *handle, int32_t sequence,
                                            int32_t *completed,
                                            int32_t *hasError,
                                            double *gpuStart,
                                            double *gpuEnd) {
  if (!handle || sequence < 0 || !completed || !hasError || !gpuStart ||
      !gpuEnd)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  if (!chain.finished)
    return -2;
  CogniMPSCommandRootSummary *summary =
      [chain.command.rootTracker summaryAtSequence:(NSUInteger)sequence];
  if (!summary || summary.sequence != (NSUInteger)sequence)
    return -3;
  if (!summary.recorded)
    return -4;
  *completed = summary.status == MTLCommandBufferStatusCompleted ? 1 : 0;
  *hasError = summary.hasError ? 1 : 0;
  *gpuStart = summary.gpuStartTime;
  *gpuEnd = summary.gpuEndTime;
  return 0;
}

extern "C" int32_t gs_mps_chain_finish_wait(void *handle, double *gpuSeconds,
                                            int32_t *segments) {
  if (!handle || !gpuSeconds || !segments)
    return -1;
  @autoreleasepool {
    CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
    if (!chain.finished) {
      GsMpsApiProfileData *profile = &chain->apiProfile;
      if (profile->enabled &&
          (!profile->tlsAttached || gs_mps_api_profile_current != profile ||
           !pthread_equal(profile->owner, pthread_self())))
        return -4;
      if (profile->enabled &&
          (profile->activeScopeDepth != 0 || profile->graphDepth != 0)) {
        profile->failed = true;
        return -5;
      }
      [chain.command rememberCurrentRoot];
      // Same-queue completion of the last root orders all earlier roots. This
      // is the sole host wait, including when MPSGraph continued internally.
      [chain.command commit];
      [chain.command waitUntilCompleted];

      if (profile->enabled) {
        chain.command.profileData = nullptr;
        if (gs_mps_api_profile_current == profile)
          gs_mps_api_profile_current = nullptr;
        else
          profile->failed = true;
        profile->tlsAttached = false;
      }

      // Completion handlers normally already have compacted all roots to
      // summaries. This fallback is idempotent and executes only after the
      // one terminal wait; it does not add a synchronization boundary.
      NSArray<CogniMPSCommandRootSummary *> *summaries =
          [chain.command.rootTracker finishPendingRoots];
      if (profile->traceEnabled)
        [chain.command.rootTracker disableTraceCapture];
      chain.result = 0;
      double elapsed = 0.0;
      for (CogniMPSCommandRootSummary *summary in summaries) {
        if (summary.status != MTLCommandBufferStatusCompleted ||
            summary.hasError) {
          fprintf(
              stderr,
              "[MPS_CHAIN_ERROR] segment=%lu status=%ld error_domain=\"%s\" "
              "error_code=%ld description=\"%s\" gpu_start=%.9f gpu_end=%.9f\n",
              (unsigned long)summary.sequence, (long)summary.status,
              summary.errorDomain.UTF8String ?: "none",
              (long)summary.errorCode,
              summary.errorDescription.UTF8String ?: "none",
              summary.gpuStartTime, summary.gpuEndTime);
          fflush(stderr);
          chain.result = -2;
        }
        double seconds = summary.gpuEndTime - summary.gpuStartTime;
        if (!std::isfinite(seconds) || seconds < 0.0)
          chain.result = -3;
        else
          elapsed += seconds;
      }
      chain.gpuSeconds = elapsed;
      *segments = (int32_t)summaries.count;
      chain.finished = YES;
    } else {
      *segments = (int32_t)chain.command.rootTracker.summaryCount;
    }
    *gpuSeconds = chain.gpuSeconds;
    return chain.result;
  }
}

// A nonblocking state query used to distinguish a pre-wait rejection from a
// terminal completion error. The latter is still safe for resource release.
extern "C" int32_t gs_mps_chain_drained(void *handle) {
  if (!handle)
    return -1;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  return chain.finished ? 1 : 0;
}

extern "C" int32_t gs_mps_chain_release(void *handle) {
  if (!handle)
    return 0;
  CogniMPSCommandChain *chain = (__bridge CogniMPSCommandChain *)handle;
  if (!chain.finished)
    return -1;
  GsMpsApiProfileData *profile = &chain->apiProfile;
  if (profile->tlsAttached || profile->activeScopeDepth != 0 ||
      profile->graphDepth != 0) {
    profile->failed = true;
    return -2;
  }
  free(profile->traceEvents);
  profile->traceEvents = nullptr;
  chain.command.profileData = nullptr;
  CFRelease(handle);
  return 0;
}

// CPU-only root object for testing the production tracker without committing
// Metal work. All instances can compare equal; tracking must remain pointer-
// identity based. Its completion block intentionally stays attached after it
// fires, exposing any accidental strong tracker capture as a retain cycle.
static uint64_t gs_mps_chain_probe_equality_calls = 0;

@interface CogniMPSChainRootProbeObject : NSObject
@property(nonatomic) NSUInteger identity;
@property(nonatomic) MTLCommandBufferStatus status;
@property(nonatomic, strong) NSError *error;
@property(nonatomic) double GPUStartTime;
@property(nonatomic) double GPUEndTime;
@property(nonatomic, copy) void (^completionHandler)(id<MTLCommandBuffer>);
@property(nonatomic) NSUInteger handlerRegistrations;
- (instancetype)initWithIdentity:(NSUInteger)identity;
- (void)addCompletedHandler:(void (^)(id<MTLCommandBuffer>))handler;
- (void)completeWithStatus:(MTLCommandBufferStatus)status
                     error:(NSError *)error
                  gpuStart:(double)gpuStart
                    gpuEnd:(double)gpuEnd
            deliverCallback:(BOOL)deliverCallback;
- (void)deliverCompletionCallback;
@end

@implementation CogniMPSChainRootProbeObject
- (instancetype)initWithIdentity:(NSUInteger)identity {
  self = [super init];
  if (self)
    self.identity = identity;
  return self;
}
- (BOOL)isEqual:(id)object {
  ++gs_mps_chain_probe_equality_calls;
  return [object isKindOfClass:[CogniMPSChainRootProbeObject class]] &&
         self.identity == ((CogniMPSChainRootProbeObject *)object).identity;
}
- (NSUInteger)hash { return self.identity; }
- (void)addCompletedHandler:(void (^)(id<MTLCommandBuffer>))handler {
  ++self.handlerRegistrations;
  self.completionHandler = handler;
}
- (void)completeWithStatus:(MTLCommandBufferStatus)status
                     error:(NSError *)error
                  gpuStart:(double)gpuStart
                    gpuEnd:(double)gpuEnd
            deliverCallback:(BOOL)deliverCallback {
  self.status = status;
  self.error = error;
  self.GPUStartTime = gpuStart;
  self.GPUEndTime = gpuEnd;
  if (deliverCallback)
    [self deliverCompletionCallback];
}
- (void)deliverCompletionCallback {
  if (self.completionHandler)
    self.completionHandler((id<MTLCommandBuffer>)self);
}
@end

extern "C" int32_t gs_mps_chain_root_tracking_probe(
    int32_t uniqueRoots, int32_t *retainedRoots, uint64_t *equalityCalls) {
  if (uniqueRoots <= 0 || !retainedRoots || !equalityCalls)
    return -1;
  @autoreleasepool {
    CogniMPSCommandRootTracker *tracker = [CogniMPSCommandRootTracker new];
    NSMutableArray<CogniMPSChainRootProbeObject *> *roots =
        [NSMutableArray arrayWithCapacity:(NSUInteger)uniqueRoots];
    gs_mps_chain_probe_equality_calls = 0;
    for (int32_t index = 0; index < uniqueRoots; ++index) {
      CogniMPSChainRootProbeObject *root =
          [[CogniMPSChainRootProbeObject alloc]
              initWithIdentity:(NSUInteger)index];
      [roots addObject:root];
      [tracker trackRoot:(id<MTLCommandBuffer>)root];
      [tracker trackRoot:(id<MTLCommandBuffer>)root];
    }
    CogniMPSChainRootProbeObject *equalClone =
        [[CogniMPSChainRootProbeObject alloc] initWithIdentity:0];
    [roots addObject:equalClone];
    [tracker trackRoot:(id<MTLCommandBuffer>)equalClone];

    *retainedRoots = (int32_t)[tracker summariesSnapshot].count;
    *equalityCalls = gs_mps_chain_probe_equality_calls;
    BOOL oneHandlerPerIdentity = YES;
    for (CogniMPSChainRootProbeObject *root in roots)
      oneHandlerPerIdentity &= root.handlerRegistrations == 1;
    return oneHandlerPerIdentity &&
                   tracker.pendingCount == (NSUInteger)uniqueRoots + 1 &&
                   *retainedRoots == uniqueRoots + 1 && *equalityCalls == 0
               ? 0
               : -2;
  }
}

// Large multi-root CPU discriminator: normal callbacks immediately release
// each completed root, while two terminal-fallback roots cover both callback
// race orders, exact-once metrics/errors, and weak tracker capture.
extern "C" int32_t gs_mps_chain_retention_probe(
    int32_t completedRoots, int32_t *summaryCount,
    int32_t *pendingAfterFinish, int32_t *maxPending,
    int32_t *errorCount, int32_t *callbackRecords,
    int32_t *fallbackRecords, double *gpuSeconds,
    int32_t *releasedCompletedRoots, int32_t *trackerReleased) {
  if (completedRoots <= 0 || !summaryCount || !pendingAfterFinish ||
      !maxPending || !errorCount || !callbackRecords || !fallbackRecords ||
      !gpuSeconds || !releasedCompletedRoots || !trackerReleased)
    return -1;
  @autoreleasepool {
    CogniMPSCommandRootTracker *tracker = [CogniMPSCommandRootTracker new];
    __weak CogniMPSCommandRootTracker *weakTracker = tracker;
    gs_mps_chain_probe_equality_calls = 0;
    int32_t released = 0;
    for (int32_t index = 0; index < completedRoots; ++index) {
      __weak CogniMPSChainRootProbeObject *weakRoot = nil;
      @autoreleasepool {
        CogniMPSChainRootProbeObject *root =
            [[CogniMPSChainRootProbeObject alloc]
                initWithIdentity:(NSUInteger)index];
        weakRoot = root;
        [tracker trackRoot:(id<MTLCommandBuffer>)root];
        NSError *error = nil;
        MTLCommandBufferStatus status = MTLCommandBufferStatusCompleted;
        if (index == completedRoots / 2) {
          status = MTLCommandBufferStatusError;
          error = [NSError errorWithDomain:@"ProbeDomain"
                                      code:17
                                  userInfo:@{
                                    NSLocalizedDescriptionKey : @"probe failure"
                                  }];
        }
        [root completeWithStatus:status
                           error:error
                        gpuStart:(double)index
                          gpuEnd:(double)index + 0.25
                  deliverCallback:YES];
        root = nil;
      }
      // Check after the per-root autorelease pool drains, so temporary
      // Foundation references cannot make a released root look retained.
      if (!weakRoot)
        ++released;
    }

    CogniMPSChainRootProbeObject *fallbackError =
        [[CogniMPSChainRootProbeObject alloc] initWithIdentity:completedRoots];
    [tracker trackRoot:(id<MTLCommandBuffer>)fallbackError];
    [fallbackError completeWithStatus:MTLCommandBufferStatusError
                                 error:[NSError errorWithDomain:@"FallbackDomain"
                                                           code:23
                                                       userInfo:@{
                                                         NSLocalizedDescriptionKey : @"fallback failure"
                                                       }]
                              gpuStart:10.0
                                gpuEnd:11.5
                        deliverCallback:NO];

    CogniMPSChainRootProbeObject *fallbackRace =
        [[CogniMPSChainRootProbeObject alloc]
            initWithIdentity:(NSUInteger)completedRoots + 1];
    [tracker trackRoot:(id<MTLCommandBuffer>)fallbackRace];
    [fallbackRace completeWithStatus:MTLCommandBufferStatusCompleted
                               error:nil
                            gpuStart:12.0
                              gpuEnd:14.0
                      deliverCallback:NO];

    NSArray<CogniMPSCommandRootSummary *> *summaries =
        [tracker finishPendingRoots];
    // Simulate a completion callback arriving after terminal fallback won.
    [fallbackError deliverCompletionCallback];
    [fallbackRace deliverCompletionCallback];

    double elapsed = 0.0;
    int32_t errors = 0;
    for (CogniMPSCommandRootSummary *summary in summaries) {
      elapsed += summary.gpuEndTime - summary.gpuStartTime;
      errors += summary.hasError ? 1 : 0;
    }
    *summaryCount = (int32_t)summaries.count;
    *pendingAfterFinish = (int32_t)tracker.pendingCount;
    *maxPending = (int32_t)tracker.maxPendingRoots;
    *errorCount = errors;
    *callbackRecords = (int32_t)tracker.callbackRecords;
    *fallbackRecords = (int32_t)tracker.terminalFallbackRecords;
    *gpuSeconds = elapsed;
    *releasedCompletedRoots = released;

    tracker = nil;
    *trackerReleased = weakTracker == nil ? 1 : 0;
    return gs_mps_chain_probe_equality_calls == 0 ? 0 : -2;
  }
}
