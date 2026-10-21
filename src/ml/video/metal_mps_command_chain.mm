// Opt-in serial native/MPSGraph command chain. MPSGraph may commitAndContinue;
// every subsequent native encoder must use the wrapper's current root.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#define GS_MPS_API_PROFILE_IMPLEMENTATION
#include "mps_api_profile.h"
#include <cmath>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <type_traits>

static_assert(sizeof(gs_mps_api_profile_stats) == 10 * sizeof(uint64_t),
              "MPS API profile stats FFI layout changed");

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
  uint32_t phase;
  uint32_t graphDepth;
  uint32_t activeScopeDepth;
  bool enabled;
  bool failed;
  bool tlsAttached;
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
  GsMpsApiProfileScopeGuard(const GsMpsApiProfileScopeGuard &) = delete;
  GsMpsApiProfileScopeGuard &operator=(const GsMpsApiProfileScopeGuard &) = delete;

private:
  gs_mps_api_profile_scope scope_{};
};

static void gs_mps_api_profile_fail(GsMpsApiProfileData *data) {
  if (data)
    data->failed = true;
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
@end
@implementation CogniMPSCommandRootSummary
@end

@interface CogniMPSPendingRoot : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> root;
@end
@implementation CogniMPSPendingRoot
@end

@interface CogniMPSCommandRootTracker : NSObject
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
- (void)trackRoot:(id<MTLCommandBuffer>)root;
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

- (void)trackRoot:(id<MTLCommandBuffer>)root {
  if (!root)
    return;

  [self.lock lock];
  if ([self.seenRoots containsObject:root]) {
    [self.lock unlock];
    return;
  }

  [self.seenRoots addObject:root];
  CogniMPSCommandRootSummary *summary = [CogniMPSCommandRootSummary new];
  summary.sequence = self.summaries.count;
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
}

- (void)recordCompletedRoot:(id<MTLCommandBuffer>)root
          terminalFallback:(BOOL)terminalFallback {
  if (!root)
    return;

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
  [self.rootTracker trackRoot:root];
}
- (void)commitAndContinue {
  [self rememberCurrentRoot];
  {
    GsMpsApiProfileScopeGuard profileScope(
        GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE, (__bridge void *)self);
    [super commitAndContinue];
  }
  [self rememberCurrentRoot];
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
