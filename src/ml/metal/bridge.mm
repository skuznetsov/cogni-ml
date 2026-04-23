// Metal FFI bridge for Crystal
// Implements buffer management, device initialization, and kernel dispatch

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Global device and command queue
static id<MTLDevice> gs_device = nil;
static id<MTLCommandQueue> gs_command_queue = nil;
static NSMutableDictionary<NSString*, id<MTLLibrary>>* gs_libraries = nil;
static id<MTLLibrary> gs_default_library = nil;

extern "C" int32_t init_device_impl();

static int32_t ensure_device() {
    if (gs_device != nil) return 0;
    return init_device_impl();
}

// ============================================================================
// Device Management
// ============================================================================

extern "C" int32_t init_device_impl() {
    if (gs_device != nil) {
        return 0; // Already initialized
    }

    gs_device = MTLCreateSystemDefaultDevice();
    if (gs_device == nil) {
        NSLog(@"GS: Failed to create Metal device");
        return -1;
    }

    gs_command_queue = [gs_device newCommandQueue];
    if (gs_command_queue == nil) {
        NSLog(@"GS: Failed to create command queue");
        gs_device = nil;
        return -2;
    }

    gs_libraries = [NSMutableDictionary new];

    // Try to load default library (for pre-compiled kernels)
    gs_default_library = [gs_device newDefaultLibrary];
    if (gs_default_library == nil) {
        NSLog(@"GS: No default Metal library found (will compile from source)");
    }

    NSLog(@"GS: Metal initialized - Device: %@, Unified Memory: %@",
          gs_device.name,
          gs_device.hasUnifiedMemory ? @"YES" : @"NO");

    return 0;
}

extern "C" void* get_device_impl() {
    return (__bridge void*)gs_device;
}

extern "C" void* get_command_queue_impl() {
    return (__bridge void*)gs_command_queue;
}

extern "C" void synchronize_impl() {
    if (gs_command_queue == nil) return;

    id<MTLCommandBuffer> cmd = [gs_command_queue commandBuffer];
    [cmd commit];
    [cmd waitUntilCompleted];
}

extern "C" const char* device_name_impl() {
    if (gs_device == nil) return "Unknown";
    return [gs_device.name UTF8String];
}

extern "C" int32_t max_threads_per_threadgroup_impl() {
    if (gs_device == nil) return 1;
    // Apple Silicon typically supports 1024 threads per threadgroup
    return (int32_t)gs_device.maxThreadsPerThreadgroup.width;
}

extern "C" int64_t recommended_working_set_size_impl() {
    if (gs_device == nil) return 0;
    return (int64_t)gs_device.recommendedMaxWorkingSetSize;
}

extern "C" int32_t has_unified_memory_impl() {
    if (gs_device == nil) return 0;
    return gs_device.hasUnifiedMemory ? 1 : 0;
}

// ============================================================================
// Buffer Management
// ============================================================================

extern "C" void* create_buffer_impl(int64_t size, int32_t storage_mode) {
    if (ensure_device() != 0) return nullptr;

    MTLResourceOptions options;
    switch (storage_mode) {
        case 0: // Shared (default for Apple Silicon)
            options = MTLResourceStorageModeShared;
            break;
        case 1: // Private (GPU only)
            options = MTLResourceStorageModePrivate;
            break;
        case 2: // Managed (macOS with explicit sync)
            options = MTLResourceStorageModeManaged;
            break;
        default:
            options = MTLResourceStorageModeShared;
    }

    id<MTLBuffer> buffer = [gs_device newBufferWithLength:(NSUInteger)size options:options];
    if (buffer == nil) {
        NSLog(@"GS: Failed to allocate buffer of size %lld", size);
        return nullptr;
    }

    return (__bridge_retained void*)buffer;
}

extern "C" void release_buffer_impl(void* handle) {
    if (handle == nullptr) return;
    id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)handle;
    buffer = nil; // ARC will release
}

// Zero-copy buffer: wraps an existing page-aligned memory region (e.g. the
// mmap'd GGUF file) as an MTLBuffer without allocating or copying. Caller
// retains ownership of the underlying memory and must keep it alive at
// least as long as the MTLBuffer.
//
// Apple Silicon requires the base pointer AND size to be multiples of
// vm_page_size (16 KiB on M-series). Returns nullptr on failure.
extern "C" void* create_buffer_no_copy_impl(void* ptr, int64_t size, int32_t storage_mode) {
    if (ensure_device() != 0) return nullptr;
    if (ptr == nullptr || size <= 0) return nullptr;

    MTLResourceOptions options;
    switch (storage_mode) {
        case 0: options = MTLResourceStorageModeShared;  break;
        case 2: options = MTLResourceStorageModeManaged; break;
        // Private storage is incompatible with NoCopy (GPU-only memory).
        default: options = MTLResourceStorageModeShared;
    }

    // Passing a nil deallocator keeps the MTLBuffer as a pure view —
    // Metal will NOT free this memory on release.
    id<MTLBuffer> buffer = [gs_device newBufferWithBytesNoCopy:ptr
                                                       length:(NSUInteger)size
                                                      options:options
                                                  deallocator:nil];
    if (buffer == nil) {
        NSLog(@"GS: newBufferWithBytesNoCopy failed (ptr=%p, size=%lld). "
              "Check page alignment (need multiple of %lu).",
              ptr, size, (unsigned long)getpagesize());
        return nullptr;
    }

    return (__bridge_retained void*)buffer;
}

extern "C" void* buffer_contents_impl(void* handle) {
    if (handle == nullptr) return nullptr;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    return buffer.contents;
}

extern "C" int64_t buffer_size_impl(void* handle) {
    if (handle == nullptr) return 0;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    return (int64_t)buffer.length;
}

extern "C" void buffer_write_impl(void* handle, void* data, int64_t size) {
    if (handle == nullptr || data == nullptr || size <= 0) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;

    // Bounds check: clamp size to buffer length
    int64_t buffer_len = (int64_t)buffer.length;
    if (size > buffer_len) {
        NSLog(@"GS Warning: buffer_write size (%lld) exceeds buffer length (%lld), clamping", size, buffer_len);
        size = buffer_len;
    }

    memcpy(buffer.contents, data, (size_t)size);

    // Sync if managed mode
    if (buffer.storageMode == MTLStorageModeManaged) {
        [buffer didModifyRange:NSMakeRange(0, (NSUInteger)size)];
    }
}

extern "C" int64_t buffer_read_impl(void* handle, void* dest, int64_t size) {
    if (handle == nullptr || dest == nullptr || size <= 0) return 0;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;

    // Bounds check: clamp size to buffer length
    int64_t buffer_len = (int64_t)buffer.length;
    int64_t actual_size = size;
    if (size > buffer_len) {
        NSLog(@"GS Warning: buffer_read size (%lld) exceeds buffer length (%lld), clamping", size, buffer_len);
        actual_size = buffer_len;
    }

    memcpy(dest, buffer.contents, (size_t)actual_size);
    return actual_size;  // Return actual bytes read for caller validation
}

extern "C" void buffer_sync_impl(void* handle) {
    if (handle == nullptr) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    if (buffer.storageMode == MTLStorageModeManaged) {
        // For managed buffers, synchronize after GPU writes
        id<MTLCommandBuffer> cmd = [gs_command_queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit synchronizeResource:buffer];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

extern "C" void buffer_copy_impl(void* src_handle, void* dst_handle, int64_t size) {
    if (src_handle == nullptr || dst_handle == nullptr) return;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)src_handle;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_handle;
    memcpy(dst.contents, src.contents, (size_t)size);

    // Sync if managed mode
    if (dst.storageMode == MTLStorageModeManaged) {
        [dst didModifyRange:NSMakeRange(0, (NSUInteger)size)];
    }
}

extern "C" int32_t buffer_set_purgeable_impl(void* handle, int32_t state) {
    if (handle == nullptr) return 0;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    MTLPurgeableState mtl_state = MTLPurgeableStateKeepCurrent;

    switch (state) {
        case 0: // NonVolatile
            mtl_state = MTLPurgeableStateNonVolatile;
            break;
        case 1: // Volatile
            mtl_state = MTLPurgeableStateVolatile;
            break;
        case 2: // Empty
            mtl_state = MTLPurgeableStateEmpty;
            break;
        default:
            mtl_state = MTLPurgeableStateKeepCurrent;
    }

    MTLPurgeableState result = [buffer setPurgeableState:mtl_state];
    return (int32_t)result;
}

// ============================================================================
// Command Buffer
// ============================================================================

extern "C" void* create_command_buffer_impl() {
    if (ensure_device() != 0) return nullptr;
    id<MTLCommandBuffer> cmd = [gs_command_queue commandBuffer];
    return (__bridge_retained void*)cmd;
}

// Lightweight command buffer (no retain tracking — faster)
extern "C" void* gs_create_command_buffer_fast() {
    if (ensure_device() != 0) return nullptr;
    id<MTLCommandBuffer> cmd = [gs_command_queue commandBufferWithUnretainedReferences];
    return (__bridge_retained void*)cmd;
}

// Enqueue without commit — tells Metal the execution order
extern "C" void gs_enqueue_command_buffer(void* cmd_handle) {
    if (cmd_handle == nullptr) return;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_handle;
    [cmd enqueue];
}

// Commit without waiting (async GPU execution)
extern "C" void gs_commit_command_buffer(void* cmd_handle) {
    if (cmd_handle == nullptr) return;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_handle;
    [cmd commit];
}

// Submit two command buffers: commit first immediately, commit second after callback
// The first buffer starts executing on GPU while second is still being encoded
extern "C" void gs_submit_pipeline(void* cmd1, void* cmd2) {
    if (cmd1) {
        id<MTLCommandBuffer> c1 = (__bridge id<MTLCommandBuffer>)cmd1;
        [c1 commit];
    }
    if (cmd2) {
        id<MTLCommandBuffer> c2 = (__bridge id<MTLCommandBuffer>)cmd2;
        [c2 commit];
    }
}

// Wait for previously committed command buffer
extern "C" void gs_wait_command_buffer(void* cmd_handle) {
    if (cmd_handle == nullptr) return;
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)cmd_handle;
    [cmd waitUntilCompleted];
}

extern "C" void commit_and_wait_impl(void* cmd_handle) {
    if (cmd_handle == nullptr) return;
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)cmd_handle;
    [cmd commit];
    [cmd waitUntilCompleted];
}

extern "C" void commit_impl(void* cmd_handle) {
    if (cmd_handle == nullptr) return;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_handle;
    [cmd commit];
    // Don't transfer - caller may want to wait later
}

// ============================================================================
// Pipeline Compilation
// ============================================================================

extern "C" void* create_pipeline_impl(const char* source, const char* function_name) {
    if (ensure_device() != 0) return nullptr;
    if (source == nullptr || function_name == nullptr) return nullptr;

    NSString* sourceStr = [NSString stringWithUTF8String:source];
    NSString* funcName = [NSString stringWithUTF8String:function_name];

    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    // Use safe math mode to prevent NaN from fast-math optimizations
    if (@available(macOS 14.0, *)) {
        options.mathMode = MTLMathModeSafe;
    } else {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
        #pragma clang diagnostic pop
    }

    id<MTLLibrary> library = [gs_device newLibraryWithSource:sourceStr
                                                    options:options
                                                      error:&error];
    if (library == nil) {
        NSLog(@"GS: Failed to compile shader: %@", error.localizedDescription);
        return nullptr;
    }

    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (function == nil) {
        NSLog(@"GS: Function '%@' not found in compiled library", funcName);
        return nullptr;
    }

    id<MTLComputePipelineState> pipeline = [gs_device newComputePipelineStateWithFunction:function
                                                                                    error:&error];
    if (pipeline == nil) {
        NSLog(@"GS: Failed to create pipeline: %@", error.localizedDescription);
        return nullptr;
    }

    return (__bridge_retained void*)pipeline;
}

extern "C" void* create_pipeline_from_library_impl(const char* library_path, const char* function_name) {
    if (ensure_device() != 0) return nullptr;
    if (library_path == nullptr || function_name == nullptr) return nullptr;

    NSString* path = [NSString stringWithUTF8String:library_path];
    NSString* funcName = [NSString stringWithUTF8String:function_name];

    // Check cache
    id<MTLLibrary> library = gs_libraries[path];
    if (library == nil) {
        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:path];
        library = [gs_device newLibraryWithURL:url error:&error];
        if (library == nil) {
            NSLog(@"GS: Failed to load library from %@: %@", path, error.localizedDescription);
            return nullptr;
        }
        gs_libraries[path] = library;
    }

    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (function == nil) {
        NSLog(@"GS: Function '%@' not found in library", funcName);
        return nullptr;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [gs_device newComputePipelineStateWithFunction:function
                                                                                    error:&error];
    if (pipeline == nil) {
        NSLog(@"GS: Failed to create pipeline: %@", error.localizedDescription);
        return nullptr;
    }

    return (__bridge_retained void*)pipeline;
}

extern "C" void* create_pipeline_from_default_library_impl(const char* function_name) {
    if (ensure_device() != 0) return nullptr;
    if (gs_default_library == nil || function_name == nullptr) return nullptr;

    NSString* funcName = [NSString stringWithUTF8String:function_name];

    id<MTLFunction> function = [gs_default_library newFunctionWithName:funcName];
    if (function == nil) {
        NSLog(@"GS: Function '%@' not found in default library", funcName);
        return nullptr;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [gs_device newComputePipelineStateWithFunction:function
                                                                                    error:&error];
    if (pipeline == nil) {
        NSLog(@"GS: Failed to create pipeline: %@", error.localizedDescription);
        return nullptr;
    }

    return (__bridge_retained void*)pipeline;
}

extern "C" int32_t pipeline_max_threads_impl(void* pipeline_handle) {
    if (pipeline_handle == nullptr) return 1;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
    return (int32_t)pipeline.maxTotalThreadsPerThreadgroup;
}

// ============================================================================
// Compute Encoder
// ============================================================================

extern "C" void* create_compute_encoder_impl(void* cmd_handle) {
    if (cmd_handle == nullptr) return nullptr;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_handle;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    return (__bridge_retained void*)encoder;
}

extern "C" void encoder_set_pipeline_impl(void* encoder_handle, void* pipeline_handle) {
    if (encoder_handle == nullptr || pipeline_handle == nullptr) return;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_handle;
    [encoder setComputePipelineState:pipeline];
}

extern "C" void encoder_set_buffer_impl(void* encoder_handle, void* buffer_handle, int64_t offset, int32_t index) {
    if (encoder_handle == nullptr || buffer_handle == nullptr) return;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_handle;
    [encoder setBuffer:buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index];
}

extern "C" void encoder_set_bytes_impl(void* encoder_handle, void* data, int32_t length, int32_t index) {
    if (encoder_handle == nullptr || data == nullptr) return;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
    [encoder setBytes:data length:(NSUInteger)length atIndex:(NSUInteger)index];
}

extern "C" void encoder_dispatch_threads_impl(
    void* encoder_handle,
    int32_t grid_x, int32_t grid_y, int32_t grid_z,
    int32_t tg_x, int32_t tg_y, int32_t tg_z
) {
    if (encoder_handle == nullptr) return;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_handle;

    MTLSize gridSize = MTLSizeMake((NSUInteger)grid_x, (NSUInteger)grid_y, (NSUInteger)grid_z);
    MTLSize threadgroupSize = MTLSizeMake((NSUInteger)tg_x, (NSUInteger)tg_y, (NSUInteger)tg_z);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

extern "C" void encoder_end_encoding_impl(void* encoder_handle) {
    if (encoder_handle == nullptr) return;
    id<MTLComputeCommandEncoder> encoder = (__bridge_transfer id<MTLComputeCommandEncoder>)encoder_handle;
    [encoder endEncoding];
}

extern "C" void encoder_set_threadgroup_memory_impl(void* encoder_handle, int32_t length, int32_t index) {
    if (encoder_handle == nullptr) return;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
    [encoder setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index];
}

// ============================================================================
// C Symbol Exports (for Crystal FFI)
// ============================================================================

// MetalFFI (buffer management)
extern "C" void* gs_create_buffer(int64_t size, int32_t storage_mode) {
    return create_buffer_impl(size, storage_mode);
}

extern "C" void* gs_create_buffer_no_copy(void* ptr, int64_t size, int32_t storage_mode) {
    return create_buffer_no_copy_impl(ptr, size, storage_mode);
}

extern "C" void gs_release_buffer(void* handle) {
    release_buffer_impl(handle);
}

extern "C" void* gs_buffer_contents(void* handle) {
    return buffer_contents_impl(handle);
}

extern "C" int64_t gs_buffer_size(void* handle) {
    return buffer_size_impl(handle);
}

extern "C" void gs_buffer_write(void* handle, void* data, int64_t size) {
    buffer_write_impl(handle, data, size);
}

extern "C" int64_t gs_buffer_read(void* handle, void* dest, int64_t size) {
    return buffer_read_impl(handle, dest, size);
}

extern "C" void gs_buffer_sync(void* handle) {
    buffer_sync_impl(handle);
}

extern "C" void gs_buffer_copy(void* src_handle, void* dst_handle, int64_t size) {
    buffer_copy_impl(src_handle, dst_handle, size);
}

extern "C" int32_t gs_buffer_set_purgeable(void* handle, int32_t state) {
    return buffer_set_purgeable_impl(handle, state);
}

// MetalDeviceFFI (device management)
extern "C" int32_t gs_init_device() {
    return init_device_impl();
}

extern "C" void* gs_get_device() {
    return get_device_impl();
}

extern "C" void* gs_get_command_queue() {
    return get_command_queue_impl();
}

extern "C" void gs_synchronize() {
    synchronize_impl();
}

extern "C" const char* gs_device_name() {
    return device_name_impl();
}

extern "C" int32_t gs_max_threads_per_threadgroup() {
    return max_threads_per_threadgroup_impl();
}

extern "C" int64_t gs_recommended_working_set_size() {
    return recommended_working_set_size_impl();
}

extern "C" int32_t gs_has_unified_memory() {
    return has_unified_memory_impl();
}

extern "C" void* gs_create_command_buffer() {
    return create_command_buffer_impl();
}

extern "C" void gs_commit_and_wait(void* cmd) {
    commit_and_wait_impl(cmd);
}

extern "C" void gs_commit(void* cmd) {
    commit_impl(cmd);
}

extern "C" void* gs_create_pipeline(const char* source, const char* function_name) {
    return create_pipeline_impl(source, function_name);
}

extern "C" void* gs_create_pipeline_from_library(const char* library_path, const char* function_name) {
    return create_pipeline_from_library_impl(library_path, function_name);
}

extern "C" void* gs_create_pipeline_from_default_library(const char* function_name) {
    return create_pipeline_from_default_library_impl(function_name);
}

extern "C" int32_t gs_pipeline_max_threads(void* pipeline) {
    return pipeline_max_threads_impl(pipeline);
}

// MetalDispatchFFI (compute encoder)
extern "C" void* gs_create_compute_encoder(void* cmd) {
    return create_compute_encoder_impl(cmd);
}

extern "C" void gs_encoder_set_pipeline(void* encoder, void* pipeline) {
    encoder_set_pipeline_impl(encoder, pipeline);
}

extern "C" void gs_encoder_set_buffer(void* encoder, void* buffer, int64_t offset, int32_t index) {
    encoder_set_buffer_impl(encoder, buffer, offset, index);
}

extern "C" void gs_encoder_set_bytes(void* encoder, void* data, int32_t length, int32_t index) {
    encoder_set_bytes_impl(encoder, data, length, index);
}

extern "C" void gs_encoder_dispatch_threads(
    void* encoder,
    int32_t grid_x, int32_t grid_y, int32_t grid_z,
    int32_t tg_x, int32_t tg_y, int32_t tg_z
) {
    encoder_dispatch_threads_impl(encoder, grid_x, grid_y, grid_z, tg_x, tg_y, tg_z);
}

// Create compute encoder with concurrent dispatch type
extern "C" void* gs_create_concurrent_compute_encoder(void* cmd_handle) {
    if (cmd_handle == nullptr) return nullptr;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_handle;
    MTLComputePassDescriptor *desc = [[MTLComputePassDescriptor alloc] init];
    desc.dispatchType = MTLDispatchTypeConcurrent;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoderWithDescriptor:desc];
    return (__bridge_retained void*)encoder;
}

// Memory barrier for concurrent encoder (between dependent dispatches)
extern "C" void gs_encoder_memory_barrier(void* encoder_handle) {
    if (encoder_handle == nullptr) return;
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

extern "C" void gs_encoder_dispatch_threadgroups(
    void* encoder,
    int32_t tg_count_x, int32_t tg_count_y, int32_t tg_count_z,
    int32_t tg_x, int32_t tg_y, int32_t tg_z
) {
    if (encoder == nullptr) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder;
    MTLSize gridSize = MTLSizeMake((NSUInteger)tg_count_x, (NSUInteger)tg_count_y, (NSUInteger)tg_count_z);
    MTLSize tgSize = MTLSizeMake((NSUInteger)tg_x, (NSUInteger)tg_y, (NSUInteger)tg_z);
    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
}

// Indirect dispatch — threadgroup counts come from a GPU buffer
extern "C" void gs_encoder_dispatch_threadgroups_indirect(
    void* encoder,
    void* indirect_buffer,
    int64_t indirect_offset,
    int32_t tg_x, int32_t tg_y, int32_t tg_z
) {
    if (encoder == nullptr || indirect_buffer == nullptr) return;
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)indirect_buffer;
    MTLSize tgSize = MTLSizeMake((NSUInteger)tg_x, (NSUInteger)tg_y, (NSUInteger)tg_z);
    [enc dispatchThreadgroupsWithIndirectBuffer:buf
                           indirectBufferOffset:(NSUInteger)indirect_offset
                          threadsPerThreadgroup:tgSize];
}

extern "C" void gs_encoder_end_encoding(void* encoder) {
    encoder_end_encoding_impl(encoder);
}

extern "C" void gs_encoder_set_threadgroup_memory(void* encoder, int32_t length, int32_t index) {
    encoder_set_threadgroup_memory_impl(encoder, length, index);
}

// ============================================================================
// Blit Encoder (memory operations)
// ============================================================================

extern "C" void* gs_create_blit_encoder(void* cmd_handle) {
    if (cmd_handle == nullptr) return nullptr;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_handle;
    id<MTLBlitCommandEncoder> encoder = [cmd blitCommandEncoder];
    return (__bridge_retained void*)encoder;
}

extern "C" void gs_blit_fill_buffer(void* encoder_handle, void* buffer_handle, uint8_t value, int64_t offset, int64_t length) {
    if (encoder_handle == nullptr || buffer_handle == nullptr) return;
    id<MTLBlitCommandEncoder> encoder = (__bridge id<MTLBlitCommandEncoder>)encoder_handle;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_handle;
    NSRange range = NSMakeRange((NSUInteger)offset, (NSUInteger)length);
    [encoder fillBuffer:buffer range:range value:value];
}

extern "C" void gs_blit_copy_buffer(void* encoder_handle, void* src_handle, int64_t src_offset, void* dst_handle, int64_t dst_offset, int64_t size) {
    if (encoder_handle == nullptr || src_handle == nullptr || dst_handle == nullptr) return;
    id<MTLBlitCommandEncoder> encoder = (__bridge id<MTLBlitCommandEncoder>)encoder_handle;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)src_handle;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_handle;
    [encoder copyFromBuffer:src sourceOffset:(NSUInteger)src_offset toBuffer:dst destinationOffset:(NSUInteger)dst_offset size:(NSUInteger)size];
}

extern "C" void gs_blit_end_encoding(void* encoder_handle) {
    if (encoder_handle == nullptr) return;
    id<MTLBlitCommandEncoder> encoder = (__bridge_transfer id<MTLBlitCommandEncoder>)encoder_handle;
    [encoder endEncoding];
}
