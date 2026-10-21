#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>

// Probe-only queue-admission ablation. Never linked by the model runtime.
extern "C" void *gs_wan_attention_profile_queue_create(void *deviceHandle,
                                                       uint32_t capacity) {
  if (!deviceHandle || (capacity != 32 && capacity != 64 && capacity != 128))
    return nullptr;
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
    id<MTLCommandQueue> queue =
        [device newCommandQueueWithMaxCommandBufferCount:capacity];
    return queue ? (__bridge_retained void *)queue : nullptr;
  }
}

extern "C" void gs_wan_attention_profile_queue_release(void *queueHandle) {
  if (queueHandle) {
    id queue = (__bridge_transfer id)queueHandle;
    (void)queue;
  }
}
