// Native timing qualification without a Metal device or GPU dispatch.
// Build standalone with ARC, C++17, Metal and Foundation; do not link bridge.o.
#include "../src/ml/metal/bridge.mm"
#include <cassert>
#include <cmath>

@interface SubmitProfileFake : NSObject
@property int commits;
@property int waits;
@property useconds_t commitDelay;
@property useconds_t waitDelay;
@property MTLCommandBufferStatus status;
@property CFTimeInterval GPUStartTime;
@property CFTimeInterval GPUEndTime;
@property CFTimeInterval kernelStartTime;
@property CFTimeInterval kernelEndTime;
@property BOOL useMachTimeline;
@property BOOL expectsRegisteredCommit;
- (void)commit;
- (void)waitUntilCompleted;
- (NSError*)error;
@end

@implementation SubmitProfileFake
- (void)commit {
    assert(gs_command_waits.size() == (self.expectsRegisteredCommit ? 1 : 0));
    self.commits += 1;
    if (self.commitDelay) usleep(self.commitDelay);
}
- (void)waitUntilCompleted {
    assert(gs_command_waits.size() == 1);
    assert(self.commits == 1);
    self.waits += 1;
    if (self.waitDelay) usleep(self.waitDelay);
    if (self.useMachTimeline) {
        double now = command_mach_seconds();
        self.GPUStartTime = now - 0.0002;
        self.GPUEndTime = now - 0.0001;
        self.kernelStartTime = now - 0.0004;
        self.kernelEndTime = now - 0.0003;
    }
}
- (NSError*)error { return nil; }
@end

static SubmitProfileFake* fake(useconds_t commit, useconds_t wait) {
    SubmitProfileFake* cmd = [SubmitProfileFake new];
    cmd.commitDelay = commit;
    cmd.waitDelay = wait;
    cmd.status = MTLCommandBufferStatusCompleted;
    cmd.GPUStartTime = 1.0;
    cmd.GPUEndTime = 1.001;
    cmd.kernelStartTime = 10.0;
    cmd.kernelEndTime = 17.5;
    cmd.expectsRegisteredCommit = YES;
    return cmd;
}

int main() {
    @autoreleasepool {
        setenv("COGNI_METAL_COMMAND_TIMEOUT_MS", "180000", 1);
        unsetenv("COGNI_METAL_SUBMIT_PROFILE");
        assert(!command_submit_profile_enabled());
        for (const char* value : {"", "0", "true", "10", "01"}) {
            setenv("COGNI_METAL_SUBMIT_PROFILE", value, 1);
            assert(!command_submit_profile_enabled());
        }
        setenv("COGNI_METAL_SUBMIT_PROFILE", "1", 1);
        assert(command_submit_profile_enabled());

        // Scheduling duration is a pair-local interval, not GPU ordering.
        assert(command_schedule_valid(10.0, 17.5));
        assert(command_schedule_valid(10.0, 10.0)); // Resolution-limited zero duration.
        assert(!command_schedule_valid(0.0, 0.0)); // Unavailable, not zero cost.
        assert(!command_schedule_valid(0.0, 10.0));
        assert(!command_schedule_valid(-1.0, 10.0));
        assert(!command_schedule_valid(10.0, 9.0));
        assert(!command_schedule_valid(NAN, 10.0));
        assert(!command_schedule_valid(10.0, INFINITY));

        // Same-clock timeline bounds, including unavailable or corrupt GPU data.
        assert(command_timeline_valid(10.0, 18.0, 19.0, 19.002));
        assert(command_timeline_valid(10.0, 10.001, 11.0, 19.0));
        assert(!command_timeline_valid(0.0, 18.0, 19.0, 20.0));
        assert(!command_timeline_valid(10.0, 0.0, 19.0, 20.0));
        assert(!command_timeline_valid(10.0, 9.0, 19.0, 20.0));
        assert(!command_timeline_valid(10.0, 18.0, 17.0, 20.0));
        assert(!command_timeline_valid(10.0, 18.0, 21.0, 20.0));
        assert(!command_timeline_valid(10.0, NAN, 19.0, 20.0));
        assert(!command_timeline_valid(10.0, 18.0, INFINITY, 20.0));
        assert(!command_timeline_valid(10.0, 18.0, 19.0, NAN));

        for (bool slow_commit : {true, false}) {
            SubmitProfileFake* cmd = fake(slow_commit ? 20000 : 0, slow_commit ? 0 : 20000);
            GSCommandSubmitProfile profile;
            wait_for_command_completion((id<MTLCommandBuffer>)cmd, true, &profile);
            assert(cmd.commits == 1 && cmd.waits == 1 && gs_command_waits.empty());
            assert(profile.setup_ms >= 0 && profile.retire_ms >= 0);
            assert(profile.commit_ms >= 0 && profile.wait_ms >= 0);
            assert(profile.mach_before_commit > 0);
            assert(profile.mach_after_wait >= profile.mach_before_commit);
            assert((slow_commit ? profile.commit_ms : profile.wait_ms) >= 19.0);
        }
        // The uninstrumented path retains one commit/wait and registration.
        SubmitProfileFake* plain = fake(0, 0);
        wait_for_command_completion((id<MTLCommandBuffer>)plain, true);
        assert(plain.commits == 1 && plain.waits == 1 && gs_command_waits.empty());
        // Already-committed waits must never commit again.
        SubmitProfileFake* already = fake(0, 0);
        already.commits = 1;
        wait_for_command_completion((id<MTLCommandBuffer>)already, false);
        assert(already.commits == 1 && already.waits == 1 && gs_command_waits.empty());

        // Async timeline captures the existing commit and wait boundaries,
        // with exactly one commit and one completion wait.
        SubmitProfileFake* timed = fake(0, 0);
        timed.useMachTimeline = YES;
        timed.expectsRegisteredCommit = NO;
        double before_commit = 0.0, after_commit = 0.0;
        void* timed_handle = (__bridge_retained void*)timed;
        gs_commit_command_buffer_timeline(timed_handle, &before_commit, &after_commit);
        assert(timed.commits == 1 && timed.waits == 0);
        assert(before_commit > 0.0 && after_commit >= before_commit);
        double before_wait = 0.0, after_wait = 0.0;
        double gpu_start = 0.0, gpu_end = 0.0;
        int32_t gpu_valid = 0;
        double kernel_start = 0.0, kernel_end = 0.0;
        int32_t kernel_valid = 0;
        int32_t timeline_status = gs_wait_command_buffer_status_timeline(
            timed_handle, before_commit, &before_wait, &after_wait,
            &gpu_start, &gpu_end, &gpu_valid, &kernel_start, &kernel_end, &kernel_valid);
        assert(timeline_status == 0 && timed.commits == 1 && timed.waits == 1);
        assert(before_wait > 0.0 && after_wait >= before_wait);
        assert(gpu_valid == 1 && gpu_end > gpu_start);
        assert(kernel_valid == 1 && kernel_end > kernel_start);
        assert(gs_command_waits.empty());

        // Missing timestamps are represented as unavailable metrics, not a
        // failed completion or an additional synchronization operation.
        SubmitProfileFake* unavailable = fake(0, 0);
        unavailable.expectsRegisteredCommit = NO;
        unavailable.GPUStartTime = 0.0;
        unavailable.GPUEndTime = 0.0;
        unavailable.kernelStartTime = 0.0;
        unavailable.kernelEndTime = 0.0;
        before_commit = after_commit = before_wait = after_wait = 0.0;
        gpu_start = gpu_end = 0.0;
        gpu_valid = 1;
        void* unavailable_handle = (__bridge_retained void*)unavailable;
        gs_commit_command_buffer_timeline(unavailable_handle, &before_commit, &after_commit);
        timeline_status = gs_wait_command_buffer_status_timeline(
            unavailable_handle, before_commit, &before_wait, &after_wait,
            &gpu_start, &gpu_end, &gpu_valid, &kernel_start, &kernel_end, &kernel_valid);
        assert(timeline_status == 0 && unavailable.commits == 1 && unavailable.waits == 1);
        assert(gpu_valid == 0 && kernel_valid == 0);
        assert(gs_command_waits.empty());

        // Failed completion remains failed, but the timeline path still waits
        // exactly once and marks optional metrics invalid.
        SubmitProfileFake* failed_timeline = fake(0, 0);
        failed_timeline.expectsRegisteredCommit = NO;
        failed_timeline.status = MTLCommandBufferStatusError;
        before_commit = after_commit = before_wait = after_wait = 0.0;
        gpu_valid = 1;
        void* failed_handle = (__bridge_retained void*)failed_timeline;
        gs_commit_command_buffer_timeline(failed_handle, &before_commit, &after_commit);
        timeline_status = gs_wait_command_buffer_status_timeline(
            failed_handle, before_commit, &before_wait, &after_wait,
            &gpu_start, &gpu_end, &gpu_valid, &kernel_start, &kernel_end, &kernel_valid);
        assert(timeline_status == -6 && failed_timeline.commits == 1 && failed_timeline.waits == 1);
        assert(gpu_valid == 0 && kernel_valid == 0);
        assert(gs_command_waits.empty());

        double gpu_seconds = -1.0;
        SubmitProfileFake* good = fake(0, 0);
        assert(gs_commit_and_wait_status_gpu_elapsed((__bridge_retained void*)good, &gpu_seconds) == 0);
        assert(std::abs(gpu_seconds - 0.001) < 1e-9 && good.commits == 1 && good.waits == 1);
        SubmitProfileFake* bad = fake(0, 0);
        bad.status = MTLCommandBufferStatusError;
        assert(gs_commit_and_wait_status_gpu_elapsed((__bridge_retained void*)bad, &gpu_seconds) == -6);
        assert(gpu_seconds == 0 && bad.commits == 1 && bad.waits == 1);
        assert(gs_command_waits.empty());
        // Resource observers never initialize Metal or wait for GPU work.
        uint64_t resources[8] = {};
        double sample_begin = 0, sample_end = 0;
        int32_t resource_valid = gs_drama3_process_resources(resources, &sample_begin, &sample_end);
        assert((resource_valid & 1) != 0 && (resource_valid & 2) != 0);
        assert(sample_begin > 0 && sample_end >= sample_begin);
        assert(gs_drama3_process_resources(nullptr, &sample_begin, &sample_end) == 0);
        assert(gs_device == nil); // No Metal device was requested by this test.
        puts("native submit profile tests PASS (no GPU)");
    }
}
