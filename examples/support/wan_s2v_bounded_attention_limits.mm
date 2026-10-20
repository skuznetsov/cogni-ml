#import <Metal/Metal.h>
#include <cstdint>

// Probe-only capability query on the same device used by the Crystal bridge.
// This creates no device, queue, buffer or command submission.
extern "C" int64_t wan_bounded_max_threadgroup_memory_length(void *handle) {
    if (!handle) return -1;
    id<MTLDevice> device = (__bridge id<MTLDevice>)handle;
    return static_cast<int64_t>(device.maxThreadgroupMemoryLength);
}
