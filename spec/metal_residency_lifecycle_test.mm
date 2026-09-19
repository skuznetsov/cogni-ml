// Explicit, bounded API qualification: one host page, no model or GPU commands.
// This does not test inference integration, residency readiness or performance.
// Build with ARC, C++17, Metal and Foundation; run with --tiny-residency.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#ifdef NDEBUG
#error "This qualification requires assertions enabled"
#endif

int main(int argc, char** argv) {
    if (argc != 2 || std::strcmp(argv[1], "--tiny-residency") != 0) {
        std::fprintf(stderr, "usage: metal_residency_lifecycle_test --tiny-residency\n");
        return 64; // Reject before device creation.
    }
    @autoreleasepool {
        if (@available(macOS 15.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) return 77; // Unavailable is not a pass.
            const size_t bytes = static_cast<size_t>(getpagesize());
            assert(bytes > 0 && bytes <= 65536);
            void* backing = nullptr;
            assert(posix_memalign(&backing, bytes, bytes) == 0);
            std::memset(backing, 0x5a, bytes);
            __weak id<MTLBuffer> weak_buffer = nil;
            __weak id<MTLResidencySet> weak_set = nil;
            @autoreleasepool {
                id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:backing
                    length:bytes options:MTLResourceStorageModeShared deallocator:nil];
                assert(buffer && buffer.contents == backing && buffer.length == bytes);
                weak_buffer = buffer;
                MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
                desc.label = @"cogni-tiny-residency-qualification";
                desc.initialCapacity = 1;
                NSError* error = nil;
                id<MTLResidencySet> set = [device newResidencySetWithDescriptor:desc error:&error];
                if (!set || error) {
                    std::fprintf(stderr, "residency creation unavailable or failed\n");
                    // No residency or command exists; release wrapper before bytes.
                    buffer = nil;
                    std::free(backing);
                    return 77;
                }
                weak_set = set;
                assert(set.allocationCount == 0 && ![set containsAllocation:buffer]);
                [set addAllocation:buffer];
                [set commit];
                assert(set.allocationCount == 1 && [set containsAllocation:buffer]);
                [set requestResidency];
                // Request-only, like the audited llama.cpp startup path.
                // No queue, command, background heartbeat or inference exists.
                [set endResidency];
                [set removeAllAllocations];
                [set commit];
                assert(set.allocationCount == 0 && ![set containsAllocation:buffer]);
                set = nil;
                buffer = nil;
            }
            assert(weak_set == nil && weak_buffer == nil);
            // The nil-deallocator wrapper did not free the caller-owned bytes.
            for (size_t i = 0; i < bytes; ++i) assert(static_cast<unsigned char*>(backing)[i] == 0x5a);
            std::free(backing);
            std::printf("PASS device=%s bytes=%zu allocations=1->0 released=1 gpu_commands=0 queue_attachment=0\n",
                        device.name.UTF8String, bytes);
            return 0;
        }
        return 77;
    }
}
