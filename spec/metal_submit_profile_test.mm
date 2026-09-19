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
- (void)commit;
- (void)waitUntilCompleted;
- (NSError*)error;
@end

@implementation SubmitProfileFake
- (void)commit {
    assert(gs_command_waits.size() == 1);
    self.commits += 1;
    if (self.commitDelay) usleep(self.commitDelay);
}
- (void)waitUntilCompleted {
    assert(gs_command_waits.size() == 1);
    assert(self.commits == 1);
    self.waits += 1;
    if (self.waitDelay) usleep(self.waitDelay);
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

        double gpu_seconds = -1.0;
        SubmitProfileFake* good = fake(0, 0);
        assert(gs_commit_and_wait_status_gpu_elapsed((__bridge_retained void*)good, &gpu_seconds) == 0);
        assert(std::abs(gpu_seconds - 0.001) < 1e-9 && good.commits == 1 && good.waits == 1);
        SubmitProfileFake* bad = fake(0, 0);
        bad.status = MTLCommandBufferStatusError;
        assert(gs_commit_and_wait_status_gpu_elapsed((__bridge_retained void*)bad, &gpu_seconds) == -6);
        assert(gpu_seconds == 0 && bad.commits == 1 && bad.waits == 1);
        assert(gs_command_waits.empty());
        assert(gs_device == nil); // No Metal device was requested by this test.
        puts("native submit profile tests PASS (no GPU)");
    }
}
