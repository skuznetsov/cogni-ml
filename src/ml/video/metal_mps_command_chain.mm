// Opt-in serial native/MPSGraph command chain. MPSGraph may commitAndContinue;
// every subsequent native encoder must use the wrapper's current root.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <stdio.h>
#include <stdint.h>

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
  [self.rootTracker trackRoot:root];
}
- (void)commitAndContinue {
  [self rememberCurrentRoot];
  [super commitAndContinue];
  [self rememberCurrentRoot];
}
@end

@interface CogniMPSCommandChain : NSObject
@property(nonatomic, strong) CogniTrackedMPSCommandBuffer *command;
@property(nonatomic) BOOL finished;
@property(nonatomic) int32_t result;
@property(nonatomic) double gpuSeconds;
@end
@implementation CogniMPSCommandChain
@end

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
      [chain.command rememberCurrentRoot];
      // Same-queue completion of the last root orders all earlier roots. This
      // is the sole host wait, including when MPSGraph continued internally.
      [chain.command commit];
      [chain.command waitUntilCompleted];

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
