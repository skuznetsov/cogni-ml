// Probe-only explicit Metal 4 runtime compiler. Does not submit or wait.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" void *gs_wan_tensorops_probe_pipeline(void *device_handle,
                                                const char *source,
                                                const char *function_name) {
  @autoreleasepool {
    if (!device_handle || !source || !function_name) return nullptr;
    if (@available(macOS 26.2, *)) {
      id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
      MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
      options.languageVersion = MTLLanguageVersion4_0;
      options.mathMode = MTLMathModeSafe;
      NSError *error = nil;
      id<MTLLibrary> library = [device newLibraryWithSource:
          [NSString stringWithUTF8String:source] options:options error:&error];
      if (!library) {
        NSLog(@"TensorOps probe library: %@", error.localizedDescription);
        return nullptr;
      }
      id<MTLFunction> function = [library newFunctionWithName:
          [NSString stringWithUTF8String:function_name]];
      if (!function) return nullptr;
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline) {
        NSLog(@"TensorOps probe pipeline: %@", error.localizedDescription);
        return nullptr;
      }
      return (__bridge_retained void *)pipeline;
    }
    return nullptr;
  }
}
