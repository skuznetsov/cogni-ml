// Bounded request-only residency diagnostic. No model, queue or GPU commands.
// Small synthetic backing is not a certificate for model-sized mmap safety.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static constexpr size_t BYTES = 64ULL << 20;
static const char* mode;
static double origin;
static char fixture_path[] = "/private/tmp/cogni-residency-file-XXXXXX";
static bool fixture_exists = false;

static void remove_fixture() {
    if (fixture_exists && unlink(fixture_path) != 0) std::perror("fixture unlink");
}

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

// Prepare a linked, fully written file before sampling. Never write through the
// mapping: the GGUF path uses PROT_READ/MAP_PRIVATE/MADV_RANDOM, not dirty COW.
static int prepare_file() {
    int writer = mkstemp(fixture_path);
    require(writer >= 0, "mkstemp");
    fixture_exists = true;
    require(std::atexit(remove_fixture) == 0, "fixture cleanup registration");
    std::fprintf(stderr, "[FIXTURE] path=%s bytes=%zu\n", fixture_path, BYTES);
    unsigned char block[65536];
    std::memset(block, 0x5a, sizeof(block));
    for (size_t offset = 0; offset < BYTES;) {
        ssize_t written = write(writer, block, sizeof(block));
        require(written > 0, "fixture write");
        offset += size_t(written);
        // A short write is a failed diagnostic, not a different file shape.
        require(written == ssize_t(sizeof(block)), "short fixture write");
    }
    require(fsync(writer) == 0 && close(writer) == 0, "fixture sync/close");
    int reader = open(fixture_path, O_RDONLY);
    require(reader >= 0, "fixture readonly reopen");
    return reader;
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
    const char* selected = argc == 2 ? argv[1] : "";
    const bool file_backed = std::strncmp(selected, "--file-", 7) == 0;
    const char* plain = file_backed ? selected + 7 :
                        (std::strncmp(selected, "--", 2) == 0 ? selected + 2 : "");
    if (argc != 2 || (std::strcmp(plain, "control") != 0 &&
                      std::strcmp(plain, "request") != 0 &&
                      std::strcmp(plain, "hold") != 0)) {
        std::fprintf(stderr, "usage: metal_residency_footprint_probe --[file-]control|--[file-]request|--[file-]hold\n");
        return 64;
    }
    mode = argv[1] + 2;
    const bool request = std::strcmp(plain, "request") == 0;
    const bool hold = std::strcmp(plain, "hold") == 0;
    origin = now();
    @autoreleasepool {
        if (@available(macOS 15.0, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) return 77;
            require([device.name isEqualToString:@"Apple M2 Max"], "device outside diagnostic scope");
            const int file_fd = file_backed ? prepare_file() : -1;
            std::printf("{\"event\":\"config\",\"mode\":\"%s\",\"bytes\":%zu,"
                        "\"cycles\":%d,\"device\":\"Apple M2 Max\",\"gpu_commands\":0}\n",
                        mode, BYTES, hold ? 1 : 3);
            for (int cycle = 0; cycle < (hold ? 1 : 3); ++cycle) {
                sample(device, cycle, "baseline");
                void* backing = mmap(nullptr, BYTES,
                    file_backed ? PROT_READ : PROT_READ | PROT_WRITE,
                    file_backed ? MAP_PRIVATE : MAP_PRIVATE | MAP_ANON, file_fd, 0);
                require(backing != MAP_FAILED, "mmap");
                if (file_backed) require(madvise(backing, BYTES, MADV_RANDOM) == 0, "madvise");
                const size_t page = getpagesize();
                require(page > 0 && BYTES % page == 0, "page size");
                unsigned long long sum = 0;
                for (size_t i = 0; i < BYTES; i += page) {
                    if (!file_backed) static_cast<volatile unsigned char*>(backing)[i] = 0x5a;
                    sum += static_cast<volatile unsigned char*>(backing)[i];
                }
                require(sum == (BYTES / page) * 0x5a, "page touch integrity");
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
            if (file_backed) {
                require(close(file_fd) == 0, "fixture reader close");
                require(unlink(fixture_path) == 0, "fixture unlink");
                fixture_exists = false;
            }
            std::printf("{\"event\":\"complete\",\"mode\":\"%s\",\"gpu_commands\":0}\n", mode);
            return 0; // Completion is not a reclamation verdict; inspect samples.
        }
        return 77;
    }
}
