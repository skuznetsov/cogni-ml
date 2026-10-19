// Stage files directly into overlapping no-copy Metal weight views. Neural
// work stays in the existing compute chain; this bridge never waits on a host.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <stdint.h>

@class CogniStageIO;
@interface CogniStageLoad : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, strong) id<MTLIOFileHandle> source;
@property(nonatomic) NSUInteger offset;
@property(nonatomic) NSUInteger bytes;
@property(nonatomic, copy) NSString *path;
@property(nonatomic, strong) NSDictionary *identity;
@end
@implementation CogniStageLoad
@end

@interface CogniStagePlan : NSObject
@property(nonatomic, weak) CogniStageIO *owner;
@property(nonatomic, strong) NSArray<CogniStageLoad *> *loads;
@end
@implementation CogniStagePlan
@end

@interface CogniStageIO : NSObject {
@public
  std::atomic<int32_t> failure;
}
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLIOCommandQueue> queue;
@property(nonatomic, strong) id<MTLSharedEvent> event;
@property(nonatomic, strong) NSMutableArray<id<MTLIOCommandBuffer>> *commands;
@property(nonatomic, strong) NSMutableArray<CogniStagePlan *> *plans;
@property(nonatomic) uint64_t retired;
@property(nonatomic) uint64_t nextRetired;
@property(nonatomic) BOOL open;
@property(nonatomic) BOOL finished;
@end
@implementation CogniStageIO
@end

extern "C" void *gs_stage_io_create(void *deviceHandle) {
  if (!deviceHandle) return nullptr;
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
    MTLIOCommandQueueDescriptor *descriptor = [MTLIOCommandQueueDescriptor new];
    descriptor.type = MTLIOCommandQueueTypeSerial;
    descriptor.maxCommandBufferCount = 32;
    descriptor.maxCommandsInFlight = 1;
    NSError *error = nil;
    CogniStageIO *owner = [CogniStageIO new];
    owner->failure.store(0);
    owner.device = device;
    owner.queue = [device newIOCommandQueueWithDescriptor:descriptor error:&error];
    owner.event = [device newSharedEvent];
    owner.commands = [NSMutableArray new];
    owner.plans = [NSMutableArray new];
    if (!owner.queue || !owner.event || error) return nullptr;
    return (__bridge_retained void *)owner;
  }
}

extern "C" void *gs_stage_io_plan(void *handle, void **buffers,
                                   const char **paths, const int64_t *offsets,
                                   const int64_t *sizes, int32_t count) {
  if (!handle || !buffers || !paths || !offsets || !sizes || count <= 0)
    return nullptr;
  @autoreleasepool {
    CogniStageIO *owner = (__bridge CogniStageIO *)handle;
    if (owner.finished || owner.commands.count) return nullptr;
    NSMutableArray<CogniStageLoad *> *loads = [NSMutableArray new];
    NSMutableDictionary<NSString *, id<MTLIOFileHandle>> *sources = [NSMutableDictionary new];
    for (int32_t i = 0; i < count; ++i) {
      if (!buffers[i] || !paths[i] || offsets[i] < 0 || sizes[i] <= 0)
        return nullptr;
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffers[i];
      if (buffer.device != owner.device || buffer.storageMode != MTLStorageModeShared ||
          !buffer.contents || (uint64_t)sizes[i] > buffer.length)
        return nullptr;
      uintptr_t begin = (uintptr_t)buffer.contents;
      if ((uint64_t)sizes[i] > UINTPTR_MAX - begin) return nullptr;
      for (CogniStageLoad *old in loads) {
        uintptr_t oldBegin = (uintptr_t)old.buffer.contents;
        if (begin < oldBegin + old.bytes && oldBegin < begin + (uintptr_t)sizes[i])
          return nullptr;
      }
      NSString *path = [NSString stringWithUTF8String:paths[i]];
      if (!path || !path.isAbsolutePath) return nullptr;
      NSError *error = nil;
      NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:&error];
      if (!attributes || error || ![attributes[NSFileType] isEqual:NSFileTypeRegular]) return nullptr;
      uint64_t fileBytes = [attributes[NSFileSize] unsignedLongLongValue];
      if ((uint64_t)offsets[i] > fileBytes || (uint64_t)sizes[i] > fileBytes - offsets[i])
        return nullptr;
      id<MTLIOFileHandle> source = sources[path];
      if (!source) {
        source = [owner.device newIOFileHandleWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (!source || error) return nullptr;
        sources[path] = source;
      }
      CogniStageLoad *load = [CogniStageLoad new];
      load.buffer = buffer;
      load.source = source;
      load.offset = (NSUInteger)offsets[i];
      load.bytes = (NSUInteger)sizes[i];
      load.path = path;
      load.identity = @{
        NSFileSystemFileNumber : attributes[NSFileSystemFileNumber],
        NSFileSize : attributes[NSFileSize],
        NSFileModificationDate : attributes[NSFileModificationDate],
      };
      [loads addObject:load];
    }
    CogniStagePlan *plan = [CogniStagePlan new];
    plan.owner = owner;
    plan.loads = loads;
    [owner.plans addObject:plan];
    return (__bridge_retained void *)plan;
  }
}

extern "C" int32_t gs_stage_io_begin(void *handle, void *planHandle, void *rootHandle) {
  if (!handle || !planHandle || !rootHandle) return -1;
  @autoreleasepool {
    CogniStageIO *owner = (__bridge CogniStageIO *)handle;
    CogniStagePlan *plan = (__bridge CogniStagePlan *)planHandle;
    id<MTLCommandBuffer> root = (__bridge id<MTLCommandBuffer>)rootHandle;
    if (owner.finished || owner.open || plan.owner != owner ||
        root.commandQueue.device != owner.device || root.status != MTLCommandBufferStatusNotEnqueued ||
        owner.retired > UINT64_MAX - 2) return -1;
    // Recheck the immutable-file contract before enqueuing any part of a
    // stage. MetalIO may cache a handle's original bytes after truncation.
    // This is an identity guard, not a lock against concurrent file mutation.
    for (CogniStageLoad *load in plan.loads) {
      NSDictionary *attributes = [[NSFileManager defaultManager]
          attributesOfItemAtPath:load.path error:nil];
      for (NSString *key in load.identity)
        if (![attributes[key] isEqual:load.identity[key]]) return -4;
    }
    id<MTLIOCommandBuffer> command = [owner.queue commandBuffer];
    if (!command) return -2;
    uint64_t ready = owner.retired + 1;
    if (owner.retired) [command waitForEvent:owner.event value:owner.retired];
    for (CogniStageLoad *load in plan.loads)
      [command loadBuffer:load.buffer offset:0 size:load.bytes sourceHandle:load.source
          sourceHandleOffset:load.offset];

    // Readiness is published by the I/O completion notification, after status
    // becomes terminal. This is not a host wait or an activation readback. It
    // also wakes the compute chain on I/O failure, so its single terminal wait
    // can drain and reject the entire result instead of deadlocking. All later
    // stages remain ordered by the same compute-retirement dependency.
    __weak CogniStageIO *weakOwner = owner;
    [command addCompletedHandler:^(id<MTLIOCommandBuffer> completed) {
      CogniStageIO *strongOwner = weakOwner;
      if (!strongOwner) return;
      if (completed.status != MTLIOStatusComplete || completed.error)
        strongOwner->failure.store(-3, std::memory_order_release);
      strongOwner.event.signaledValue = ready;
    }];
    [owner.commands addObject:command];
    owner.open = YES;
    owner.nextRetired = ready + 1;
    [root encodeWaitForEvent:owner.event value:ready];
    [command commit];
    return 0;
  }
}

extern "C" int32_t gs_stage_io_retire(void *handle, void *rootHandle) {
  if (!handle || !rootHandle) return -1;
  CogniStageIO *owner = (__bridge CogniStageIO *)handle;
  id<MTLCommandBuffer> root = (__bridge id<MTLCommandBuffer>)rootHandle;
  if (owner.finished || !owner.open || root.commandQueue.device != owner.device ||
      root.status != MTLCommandBufferStatusNotEnqueued) return -1;
  [root encodeSignalEvent:owner.event value:owner.nextRetired];
  owner.retired = owner.nextRetired;
  owner.open = NO;
  return 0;
}

// Call only after the compute chain's existing terminal completion. Every I/O
// completion published readiness before that chain could pass its stage wait.
extern "C" int32_t gs_stage_io_finish(void *handle, int32_t *stages) {
  if (!handle || !stages) return -1;
  CogniStageIO *owner = (__bridge CogniStageIO *)handle;
  if (owner.open) return -1;
  for (id<MTLIOCommandBuffer> command in owner.commands)
    if (command.status == MTLIOStatusPending) return -2;
  owner.finished = YES;
  *stages = (int32_t)owner.commands.count;
  return owner->failure.load(std::memory_order_acquire);
}

extern "C" int32_t gs_stage_io_release_plan(void *handle) {
  if (!handle) return 0;
  CogniStagePlan *plan = (__bridge CogniStagePlan *)handle;
  CogniStageIO *owner = plan.owner;
  if (owner && owner.commands.count && !owner.finished) return -1;
  CFRelease(handle);
  return 0;
}

extern "C" int32_t gs_stage_io_release(void *handle) {
  if (!handle) return 0;
  CogniStageIO *owner = (__bridge CogniStageIO *)handle;
  if (owner.commands.count && !owner.finished) return -1;
  CFRelease(handle);
  return 0;
}
