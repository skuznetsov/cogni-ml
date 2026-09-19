// Bounded request-only residency diagnostic. No model, queue or GPU commands.
// Anonymous no-copy backing is not a certificate for file-backed model mmap.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static constexpr size_t BYTES = 64ULL << 20;
static const char* mode;
static double origin;

static void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(65);
    }
}

static double now() {
    timespec ts;
    require(clock_gettime(CLOCK_MONOTONIC, &ts) == 0, "clock");
    return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}

static void sample(id<MTLDevice> device, int cycle, const char* phase) {
    task_vm_info_data_t vm = {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    require(task_info(mach_task_self(), TASK_VM_INFO,
                     reinterpret_cast<task_info_t>(&vm), &count) == KERN_SUCCESS,
            "task_vm_info");
    require(count >= TASK_VM_INFO_REV1_COUNT, "phys_footprint unavailable");
    std::printf("{\"event\":\"sample\",\"mode\":\"%s\",\"cycle\":%d,"
                "\"phase\":\"%s\",\"elapsed_ms\":%.3f,\"bytes\":%zu,"
                "\"footprint\":%llu,\"resident\":%llu,\"metal_allocated\":%llu}\n",
                mode, cycle, phase, (now() - origin) * 1000, BYTES,
                (unsigned long long)vm.phys_footprint,
                (unsigned long long)vm.resident_size,
                (unsigned long long)device.currentAllocatedSize);
    std::fflush(stdout);
}

// Absolute deadlines preserve the declared observation period across signals.
static void settle(id<MTLDevice> device, int cycle, double start) {
    const double delays[] = {0.25, 1.0, 5.0};
    const char* phases[] = {"after_250ms", "after_1000ms", "after_5000ms"};
    for (int i = 0; i < 3; ++i) {
        while (now() < start + delays[i]) usleep(10000);
        sample(device, cycle, phases[i]);
    }
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "--control") != 0 &&
                      std::strcmp(argv[1], "--request") != 0 &&
                      std::strcmp(argv[1], "--hold") != 0)) {
        std::fprintf(stderr, "usage: metal_residency_footprint_probe --control|--request|--hold\n");
        return 64;
    }
    mode = argv[1] + 2;
    const bool request = std::strcmp(mode, "request") == 0;
    const bool hold = std::strcmp(mode, "hold") == 0;
    origin = now();
    @autoreleasepool {
        if (@available(macOS 15.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) return 77;
            require([device.name isEqualToString:@"Apple M2 Max"], "device outside diagnostic scope");
            std::printf("{\"event\":\"config\",\"mode\":\"%s\",\"bytes\":%zu,"
                        "\"cycles\":%d,\"device\":\"Apple M2 Max\",\"gpu_commands\":0}\n",
                        mode, BYTES, hold ? 1 : 3);
            for (int cycle = 0; cycle < (hold ? 1 : 3); ++cycle) {
                sample(device, cycle, "baseline");
                void* backing = mmap(nullptr, BYTES, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANON, -1, 0);
                require(backing != MAP_FAILED, "mmap");
                const size_t page = getpagesize();
                require(page > 0 && BYTES % page == 0, "page size");
                for (size_t i = 0; i < BYTES; i += page)
                    static_cast<volatile unsigned char*>(backing)[i] = 0x5a;
                sample(device, cycle, "touched");
                __weak id<MTLBuffer> weak_buffer = nil;
                __weak id<MTLResidencySet> weak_set = nil;
                @autoreleasepool {
                    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:backing
                        length:BYTES options:MTLResourceStorageModeShared deallocator:nil];
                    require(buffer && buffer.contents == backing, "no-copy wrapper");
                    weak_buffer = buffer;
                    id<MTLResidencySet> set = nil;
                    if (request) {
                        MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
                        desc.initialCapacity = 1;
                        NSError* error = nil;
                        set = [device newResidencySetWithDescriptor:desc error:&error];
                        require(set != nil && error == nil, "residency set creation");
                        weak_set = set;
                        [set addAllocation:buffer];
                        [set commit];
                        require(set.allocationCount == 1, "allocation registration");
                        [set requestResidency];
                    }
                    sample(device, cycle, "prepared");
                    if (hold) {
                        sample(device, cycle, "retained");
                        settle(device, cycle, now());
                    }
                    if (set) {
                        [set endResidency];
                        [set removeAllAllocations];
                        [set commit];
                        require(set.allocationCount == 0, "allocation removal");
                    }
                    set = nil;
                    buffer = nil;
                }
                require(weak_set == nil && weak_buffer == nil, "object teardown");
                require(static_cast<volatile unsigned char*>(backing)[0] == 0x5a, "backing ownership");
                require(munmap(backing, BYTES) == 0, "munmap");
                sample(device, cycle, "released");
                if (!hold) settle(device, cycle, now());
            }
            std::printf("{\"event\":\"complete\",\"mode\":\"%s\",\"gpu_commands\":0}\n", mode);
            return 0; // Completion is not a reclamation verdict; inspect samples.
        }
        return 77;
    }
}
