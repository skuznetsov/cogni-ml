// Opt-in serial native/MPSGraph command chain. MPSGraph may commitAndContinue;
// every subsequent native encoder must use the wrapper's current root.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <stdio.h>
#include <stdint.h>

@interface CogniTrackedMPSCommandBuffer : MPSCommandBuffer
@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *segments;
@property(nonatomic, strong) NSHashTable<id<MTLCommandBuffer>> *segmentSet;
@property(nonatomic, strong) id<MTLCommandBuffer> lastRoot;
- (void)rememberCurrentRoot;
@end

// Keep ordered roots for terminal status/timing inspection, but use a
// pointer-identity hash table for membership. The list alone makes repeated
// root checks quadratic as MPSGraph creates more command-buffer segments.
static BOOL gs_mps_chain_track_root(
    NSMutableArray *segments, NSHashTable *segmentSet, id root) {
  if (!root || [segmentSet containsObject:root])
    return NO;
  [segmentSet addObject:root];
  [segments addObject:root];
  return YES;
}

@implementation CogniTrackedMPSCommandBuffer
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
  self = [super initWithCommandBuffer:commandBuffer];
  if (self) {
    self.segments = [NSMutableArray new];
    self.segmentSet = [[NSHashTable alloc]
        initWithOptions:NSPointerFunctionsStrongMemory |
                   NSPointerFunctionsObjectPointerPersonality
              capacity:1];
    [self rememberCurrentRoot];
  }
  return self;
}
- (void)rememberCurrentRoot {
  id<MTLCommandBuffer> root = self.rootCommandBuffer;
  if (root == self.lastRoot)
    return;
  self.lastRoot = root;
  gs_mps_chain_track_root(self.segments, self.segmentSet, root);
}
- (void)commitAndContinue {
  [self rememberCurrentRoot];
  [super commitAndContinue];
  [self rememberCurrentRoot];
}
@end

// CPU-only discriminator for the root-membership implementation. Distinct
// probe roots deliberately compare equal so an equality-based set would lose
// roots and an ordered-array containsObject scan would invoke isEqual O(N^2).
static uint64_t gs_mps_chain_probe_equality_calls = 0;

@interface CogniMPSChainRootProbeObject : NSObject
@property(nonatomic) NSUInteger identity;
- (instancetype)initWithIdentity:(NSUInteger)identity;
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
@end

extern "C" int32_t gs_mps_chain_root_tracking_probe(
    int32_t uniqueRoots, int32_t *retainedRoots, uint64_t *equalityCalls) {
  if (uniqueRoots <= 0 || !retainedRoots || !equalityCalls)
    return -1;
  @autoreleasepool {
    NSMutableArray *segments = [NSMutableArray new];
    NSHashTable *segmentSet = [[NSHashTable alloc]
        initWithOptions:NSPointerFunctionsStrongMemory |
                   NSPointerFunctionsObjectPointerPersonality
              capacity:(NSUInteger)uniqueRoots];
    NSMutableArray<CogniMPSChainRootProbeObject *> *roots =
        [NSMutableArray arrayWithCapacity:(NSUInteger)uniqueRoots];
    gs_mps_chain_probe_equality_calls = 0;
    for (int32_t index = 0; index < uniqueRoots; ++index) {
      CogniMPSChainRootProbeObject *root =
          [[CogniMPSChainRootProbeObject alloc]
              initWithIdentity:(NSUInteger)index];
      [roots addObject:root];
      gs_mps_chain_track_root(segments, segmentSet, root);
    }
    CogniMPSChainRootProbeObject *equalClone =
        [[CogniMPSChainRootProbeObject alloc] initWithIdentity:0];
    gs_mps_chain_track_root(segments, segmentSet, equalClone);
    for (CogniMPSChainRootProbeObject *root in roots)
      gs_mps_chain_track_root(segments, segmentSet, root);
    *retainedRoots = (int32_t)segments.count;
    *equalityCalls = gs_mps_chain_probe_equality_calls;
    return segmentSet.count == segments.count &&
                   segments.count == (NSUInteger)uniqueRoots + 1
               ? 0
               : -2;
  }
}

@interface CogniMPSCommandChain : NSObject
@property(nonatomic, strong) CogniTrackedMPSCommandBuffer *command;
@property(nonatomic) BOOL finished;
@property(nonatomic) int32_t result;
@property(nonatomic) double gpuSeconds;
@end
@implementation CogniMPSCommandChain
@end

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
      chain.result = 0;
      double elapsed = 0.0;
      NSUInteger segmentIndex = 0;
      for (id<MTLCommandBuffer> root in chain.command.segments) {
        MTLCommandBufferStatus status = root.status;
        NSError *error = root.error;
        if (status != MTLCommandBufferStatusCompleted || error) {
          chain.result = -2;
          NSString *domain = gs_mps_chain_safe_log_text(error.domain, 96);
          NSString *description =
              gs_mps_chain_safe_log_text(error.localizedDescription, 256);
          fprintf(
              stderr,
              "[MPS_CHAIN_ERROR] segment=%lu status=%ld error_domain=\"%s\" "
              "error_code=%ld description=\"%s\" gpu_start=%.9f gpu_end=%.9f\n",
              (unsigned long)segmentIndex, (long)status,
              domain.UTF8String ?: "none", (long)(error ? error.code : 0),
              description.UTF8String ?: "none", root.GPUStartTime,
              root.GPUEndTime);
          fflush(stderr);
        }
        double seconds = root.GPUEndTime - root.GPUStartTime;
        if (!std::isfinite(seconds) || seconds < 0.0)
          chain.result = -3;
        else
          elapsed += seconds;
        ++segmentIndex;
      }
      chain.gpuSeconds = elapsed;
      chain.finished = YES;
    }
    *gpuSeconds = chain.gpuSeconds;
    *segments = (int32_t)chain.command.segments.count;
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
