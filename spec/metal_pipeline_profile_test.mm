// CPU-only qualification: fake Metal objects exercise the real bridge getters.
// Build standalone with ARC, C++17, Metal and Foundation, without bridge.o.
#include "../src/ml/metal/bridge.mm"
#include <cassert>
#ifdef NDEBUG
#error "This diagnostic test requires assertions"
#endif

@interface PipelineProfileFake : NSObject
@property int libraries;
@property int functions;
@property int pipelines;
@property int failure;
- (id)newLibraryWithSource:(NSString*)source options:(MTLCompileOptions*)options error:(NSError**)error;
- (id)newLibraryWithURL:(NSURL*)url error:(NSError**)error;
- (id)newFunctionWithName:(NSString*)name;
- (id)newComputePipelineStateWithFunction:(id)function error:(NSError**)error;
@end
@implementation PipelineProfileFake
- (id)newLibraryWithSource:(NSString*)source options:(MTLCompileOptions*)options error:(NSError**)error {
    self.libraries += 1; usleep(5000); return self.failure == 1 ? nil : self;
}
- (id)newLibraryWithURL:(NSURL*)url error:(NSError**)error {
    return [self newLibraryWithSource:@"" options:nil error:error];
}
- (id)newFunctionWithName:(NSString*)name {
    self.functions += 1; usleep(5000); return self.failure == 2 ? nil : self;
}
- (id)newComputePipelineStateWithFunction:(id)function error:(NSError**)error {
    self.pipelines += 1; usleep(5000); return self.failure == 3 ? nil : self;
}
@end

static void release_pipeline(void* handle) {
    if (handle) { id object = (__bridge_transfer id)handle; (void)object; }
}

int main() {
    @autoreleasepool {
        unsetenv("COGNI_METAL_PIPELINE_PROFILE");
        assert(!pipeline_profile_enabled());
        for (const char* value : {"", "0", "true", "10", "01"}) {
            setenv("COGNI_METAL_PIPELINE_PROFILE", value, 1);
            assert(!pipeline_profile_enabled());
        }
        PipelineProfileFake* fake = [PipelineProfileFake new];
        gs_device = (id<MTLDevice>)fake; // ensure_device cannot request a real device.
        gs_libraries = [NSMutableDictionary new];
        gs_default_library = (id<MTLLibrary>)fake;
        release_pipeline(create_pipeline_impl("unused", "off"));
        setenv("COGNI_METAL_PIPELINE_PROFILE", "1", 1);
        assert(pipeline_profile_enabled());
        release_pipeline(create_pipeline_impl("unused", "source_ok"));
        release_pipeline(create_pipeline_from_library_impl("/unused.metallib", "file_miss"));
        release_pipeline(create_pipeline_from_library_impl("/unused.metallib", "file_hit"));
        release_pipeline(create_pipeline_from_default_library_impl("default_ok"));
        for (int failure = 1; failure <= 3; ++failure) {
            fake.failure = failure;
            assert(create_pipeline_impl("unused", "source_failed") == nullptr);
        }
        fake.failure = 0;
        // String escaping must not turn a name into multiple log records.
        release_pipeline(create_pipeline_impl("unused", "quoted\"\nname"));
        assert(fake.libraries == 7 && fake.functions == 8 && fake.pipelines == 7);
        assert(gs_command_queue == nil); // No queue or GPU command exists.
        gs_device = nil;
        gs_default_library = nil;
        gs_libraries = nil;
        puts("native pipeline profile tests PASS (fake device, no GPU)");
    }
}
