// Bounded process-exit probe for the Metal command watchdog lifecycle.
// Build standalone with bridge.mm included through WATCHDOG_BRIDGE_INCLUDE.
// The normal mode submits one empty command buffer; it does not dispatch a
// kernel, allocate a buffer, or read GPU memory.

#import <dispatch/dispatch.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>

// These globals are deliberately initialized before bridge.mm's globals.
// C++ destroys them after bridge.mm's globals, allowing late-exit mode to
// wake the actual watchdog source during teardown.
static void* g_watchdog_lease = nullptr;
static std::atomic<unsigned> g_completed_late_callbacks{0};
static std::atomic<unsigned> g_concurrent_waiters{0};
static std::atomic<unsigned> g_concurrent_callbacks{0};
static std::atomic<unsigned> g_concurrent_callback_saw_two_records{0};

struct WatchdogLateExitSentinel {
    ~WatchdogLateExitSentinel();
};
static WatchdogLateExitSentinel g_watchdog_late_exit_sentinel;

#ifndef WATCHDOG_BRIDGE_INCLUDE
#define WATCHDOG_BRIDGE_INCLUDE "../src/ml/metal/bridge.mm"
#endif
#include WATCHDOG_BRIDGE_INCLUDE

// The recorded pre-fix bridge has the legacy file-static wait registry. Keep
// compatibility isolated to this probe so one source file can test both the
// baseline and process-lifetime implementation without changing bridge APIs.
#ifndef WATCHDOG_PROBE_HAS_STATE_API
#define WATCHDOG_PROBE_HAS_STATE_API 1
#endif

#ifndef NDEBUG
#define WATCHDOG_REQUIRE(condition, message) \
    do { if (!(condition)) { std::fprintf(stderr, "PROBE FAIL: %s\n", message); return 90; } } while (0)
#else
#error "Compile this probe without NDEBUG so its checks remain active"
#endif

static bool watchdog_wait_set_is_empty() {
#if WATCHDOG_PROBE_HAS_STATE_API
    GSCommandWaitState& state = command_wait_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.records.empty();
#else
    std::lock_guard<std::mutex> lock(gs_command_waits_mutex);
    return gs_command_waits.empty();
#endif
}

WatchdogLateExitSentinel::~WatchdogLateExitSentinel() {
    const char* mode = std::getenv("WATCHDOG_PROBE_MODE");
    if (mode == nullptr || std::strcmp(mode, "late") != 0) return;
    if (g_watchdog_lease == nullptr) {
        std::fprintf(stderr, "PROBE FAIL: late-exit mode has no retained bridge source\n");
        std::fflush(stderr);
        std::_Exit(91);
    }

    dispatch_source_t source = (__bridge dispatch_source_t)g_watchdog_lease;
    // Keep the bridge-owned source and its real callback function, but count
    // completion after that function returns. This deterministically exercises
    // the production lock path after bridge globals have been destroyed.
    dispatch_source_set_event_handler(source, ^{
        command_watchdog_tick();
        g_completed_late_callbacks.fetch_add(1, std::memory_order_release);
    });
    dispatch_source_set_timer(source, dispatch_time(DISPATCH_TIME_NOW, 0),
                              1000000, 0);

    for (int attempt = 0; attempt < 500; ++attempt) {
        if (g_completed_late_callbacks.load(std::memory_order_acquire) > 0) {
            std::fprintf(stderr,
                         "WATCHDOG_LATE_CALLBACK_COMPLETED count=%u after bridge globals teardown\n",
                         g_completed_late_callbacks.load(std::memory_order_relaxed));
            std::fflush(stderr);
            return;
        }
        usleep(1000);
    }
    std::fprintf(stderr, "PROBE FAIL: post-teardown callback did not complete\n");
    std::fflush(stderr);
    std::_Exit(92);
}

@interface WatchdogTimeoutFake : NSObject
@end

@implementation WatchdogTimeoutFake
- (void)commit {
    if (gs_command_watchdog != nil) {
        dispatch_source_set_timer(gs_command_watchdog,
                                  dispatch_time(DISPATCH_TIME_NOW, 0),
                                  1000000, 0);
    }
}
- (void)waitUntilCompleted { usleep(100000); }
@end

#if WATCHDOG_PROBE_HAS_STATE_API
@interface WatchdogConcurrentFake : NSObject
- (void)commit;
- (void)waitUntilCompleted;
@end

@implementation WatchdogConcurrentFake
- (void)commit {}
- (void)waitUntilCompleted {
    unsigned waiting = g_concurrent_waiters.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (waiting == 2) {
        dispatch_source_set_event_handler(gs_command_watchdog, ^{
            GSCommandWaitState& state = command_wait_state();
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                if (state.records.size() == 2) {
                    g_concurrent_callback_saw_two_records.fetch_add(
                        1, std::memory_order_release);
                }
            }
            command_watchdog_tick();
            g_concurrent_callbacks.fetch_add(1, std::memory_order_release);
        });
        dispatch_source_set_timer(gs_command_watchdog,
                                  dispatch_time(DISPATCH_TIME_NOW, 0),
                                  1000000, 0);
    }

    for (int attempt = 0; attempt < 200; ++attempt) {
        if (g_concurrent_waiters.load(std::memory_order_acquire) == 2 &&
            g_concurrent_callbacks.load(std::memory_order_acquire) > 0) {
            usleep(15000);
            return;
        }
        usleep(1000);
    }
}
@end

static void run_concurrent_wait(void* retained_fake) {
    @autoreleasepool {
        id fake = (__bridge_transfer id)retained_fake;
        wait_for_command_completion((id<MTLCommandBuffer>)fake, true);
    }
}
#endif

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s late|normal|disabled|timeout|concurrent\n", argv[0]);
        return 2;
    }
    const char* mode = argv[1];
    if (std::strcmp(mode, "late") == 0 || std::strcmp(mode, "normal") == 0) {
        setenv("COGNI_METAL_COMMAND_TIMEOUT_MS", "180000", 1);
        setenv("WATCHDOG_PROBE_MODE", mode, 1);
    } else if (std::strcmp(mode, "disabled") == 0) {
        setenv("COGNI_METAL_COMMAND_TIMEOUT_MS", "0", 1);
        setenv("WATCHDOG_PROBE_MODE", mode, 1);
    } else if (std::strcmp(mode, "timeout") == 0) {
        setenv("COGNI_METAL_COMMAND_TIMEOUT_MS", "5", 1);
        setenv("WATCHDOG_PROBE_MODE", mode, 1);
    }
#if WATCHDOG_PROBE_HAS_STATE_API
    else if (std::strcmp(mode, "concurrent") == 0) {
        setenv("COGNI_METAL_COMMAND_TIMEOUT_MS", "180000", 1);
        setenv("WATCHDOG_PROBE_MODE", mode, 1);
    }
#endif
    else {
        std::fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }

    if (std::strcmp(mode, "timeout") == 0) {
        WatchdogTimeoutFake* fake = [WatchdogTimeoutFake new];
        wait_for_command_completion((id<MTLCommandBuffer>)fake, true);
        std::fprintf(stderr, "PROBE FAIL: watchdog timeout did not terminate the wait\n");
        return 93;
    }

#if WATCHDOG_PROBE_HAS_STATE_API
    if (std::strcmp(mode, "concurrent") == 0) {
        WatchdogConcurrentFake* first = [WatchdogConcurrentFake new];
        WatchdogConcurrentFake* second = [WatchdogConcurrentFake new];
        void* first_handle = (__bridge_retained void*)first;
        void* second_handle = (__bridge_retained void*)second;
        std::thread first_waiter(run_concurrent_wait, first_handle);
        std::thread second_waiter(run_concurrent_wait, second_handle);
        first_waiter.join();
        second_waiter.join();

        WATCHDOG_REQUIRE(watchdog_wait_set_is_empty(),
                         "concurrent bridge wait registrations remained active");
        WATCHDOG_REQUIRE(g_concurrent_callbacks.load(std::memory_order_acquire) > 0,
                         "watchdog callback did not run during concurrent waits");
        WATCHDOG_REQUIRE(
            g_concurrent_callback_saw_two_records.load(std::memory_order_acquire) > 0,
            "watchdog callback did not observe both concurrent wait registrations");
        std::fprintf(stderr,
                     "CONCURRENT_WAITS_COMPLETED callbacks=%u callback_saw_two_records=%u pending_waits=0\n",
                     g_concurrent_callbacks.load(std::memory_order_relaxed),
                     g_concurrent_callback_saw_two_records.load(std::memory_order_relaxed));
        return 0;
    }
#endif

    int32_t init_status = gs_init_device();
    WATCHDOG_REQUIRE(init_status == 0, "Metal device initialization failed");
    void* command = gs_create_command_buffer();
    WATCHDOG_REQUIRE(command != nullptr, "empty command buffer creation failed");
    int32_t status = gs_commit_and_wait_status(command);
    WATCHDOG_REQUIRE(status == 0, "empty bridge command did not complete successfully");
    WATCHDOG_REQUIRE(watchdog_wait_set_is_empty(), "bridge wait registration remained active after completion");

    if (std::strcmp(mode, "late") == 0 || std::strcmp(mode, "normal") == 0) {
        WATCHDOG_REQUIRE(gs_command_watchdog != nil, "normal watchdog source was not created");
        g_watchdog_lease = (__bridge_retained void*)gs_command_watchdog;
    } else {
        WATCHDOG_REQUIRE(gs_command_watchdog == nil, "disabled watchdog unexpectedly created a source");
    }

    std::fprintf(stderr,
                 "BRIDGE_NOOP_COMPLETED status=0 pending_waits=0 watchdog=%s mode=%s\n",
                 gs_command_watchdog == nil ? "disabled" : "active", mode);
    std::fflush(stderr);
    return 0;
}
