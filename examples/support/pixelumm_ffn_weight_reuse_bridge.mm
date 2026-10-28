// Bounded diagnostic only. This translation unit includes the production
// block implementation once; never link its standalone object as well.
#include "../../src/ml/video/pixelumm_decoder_block_mps_bridge.mm"

@interface PixelUMMFFNWeightReuse : NSObject
@property(nonatomic, strong) PixelUMMDecoderPlan *plan;
@property(nonatomic, strong) NSArray<PixelUMMGraphStep *> *baseline;
@property(nonatomic, strong) NSArray<PixelUMMGraphStep *> *staged;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *stageBuffers;
@property(nonatomic, strong) id<MTLComputePipelineState> expand;
@property(nonatomic, strong) id<MTLComputePipelineState> compare;
@property(nonatomic) uint64_t captureMask;
@end
@implementation PixelUMMFFNWeightReuse
@end

static const char *pixel_ffn_weight_reuse_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct ReuseCount { uint count; uint slot; };
kernel void pixel_reuse_expand(device const ushort *weight [[buffer(0)]],
                               device float *stage [[buffer(1)]],
                               constant ReuseCount &p [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  if (index < p.count) stage[index] = as_type<float>(uint(weight[index]) << 16u);
}
kernel void pixel_reuse_compare(device const ushort *weight [[buffer(0)]],
                                device const float *stage [[buffer(1)]],
                                device atomic_uint *metrics [[buffer(2)]],
                                constant ReuseCount &p [[buffer(3)]],
                                uint index [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const bool active = index < p.count;
  const float expected = active ? as_type<float>(uint(weight[index]) << 16u) : 0.f;
  const float actual = active ? stage[index] : 0.f;
  const uint nonfinite = active && (!isfinite(expected) || !isfinite(actual));
  const uint different = active && as_type<uint>(expected) != as_type<uint>(actual);
  const uint failures = simd_sum(uint(nonfinite || different));
  const uint words = simd_sum(different);
  const uint bad = simd_sum(nonfinite);
  const uint elements = simd_sum(uint(active));
  if (lane == 0u) {
    atomic_fetch_add_explicit(metrics + p.slot * 4u, failures, memory_order_relaxed);
    atomic_fetch_add_explicit(metrics + p.slot * 4u + 1u, words, memory_order_relaxed);
    atomic_fetch_add_explicit(metrics + p.slot * 4u + 2u, bad, memory_order_relaxed);
    atomic_fetch_add_explicit(metrics + p.slot * 4u + 3u, elements, memory_order_relaxed);
  }
}
)METAL";

// Match the production F32 FFN graph's rank, transpose, accumulation and BF16
// output boundary. Only the exact weight expansion moves outside this graph.
static PixelUMMGraphStep *pixel_reuse_linear_step(
    PixelUMMDecoderPlan *plan, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> stageBuffer, id<MTLBuffer> outputBuffer,
    uint32_t inputWidth, uint32_t outputWidth, NSString *name,
    char *error, size_t capacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      inputBuffer, pixel_shape({1, plan.tokens, inputWidth}),
      MPSDataTypeBFloat16, [name stringByAppendingString:@"_input"]);
  MPSGraphTensor *weight = pixel_add_feed(graph, feeds, placeholders, inputData,
      stageBuffer, pixel_shape({outputWidth, inputWidth}),
      MPSDataTypeFloat32, [name stringByAppendingString:@"_weight_f32"]);
  MPSGraphTensor *inputF32 = input ? [graph castTensor:input
      toType:MPSDataTypeFloat32 name:[name stringByAppendingString:@"_input_f32"]] : nil;
  MPSGraphTensor *flat = inputF32 ? [graph reshapeTensor:inputF32
      withShape:pixel_shape({plan.tokens, inputWidth})
      name:[name stringByAppendingString:@"_flat"]] : nil;
  MPSGraphTensor *transpose = weight ? [graph transposeTensor:weight dimension:0
      withDimension:1 name:[name stringByAppendingString:@"_weight_transpose"]] : nil;
  MPSGraphTensor *product = flat && transpose
      ? [graph matrixMultiplicationWithPrimaryTensor:flat secondaryTensor:transpose
          name:[name stringByAppendingString:@"_matmul"]] : nil;
  MPSGraphTensor *bf16 = pixel_cast_bf16(graph, product,
      [name stringByAppendingString:@"_bf16"]);
  NSArray *outputShape = pixel_shape({1, plan.tokens, outputWidth});
  MPSGraphTensor *output = bf16 ? [graph reshapeTensor:bf16 withShape:outputShape
      name:[name stringByAppendingString:@"_output"]] : nil;
  if (!input || !weight || !inputF32 || !flat || !transpose || !product ||
      !bf16 || !output) {
    pixel_set_error(error, capacity, "unable to build staged FFN graph");
    return nil;
  }
  return pixel_compile_step(plan.device, graph, feeds, placeholders, inputData,
      output, outputBuffer, outputShape, name, error, capacity);
}

extern "C" void *gs_pixelumm_ffn_weight_reuse_prepare(
    void *planHandle, void **stageHandles, char *error, size_t capacity) {
  if (error && capacity) error[0] = '\0';
  if (!planHandle || !stageHandles) {
    pixel_set_error(error, capacity, "FFN staging requires a plan and three buffers");
    return nullptr;
  }
  @autoreleasepool {
    @try {
      PixelUMMDecoderPlan *plan = (__bridge PixelUMMDecoderPlan *)planHandle;
      if (!plan.ffnF32 || plan.sharedGateUpPrepared || plan.encodingStarted ||
          plan.dim != 4096 || plan.ffn != 12288) {
        pixel_set_error(error, capacity, "FFN staging requires an idle independent F32 FFN plan");
        return nullptr;
      }
      NSMutableArray<id<MTLBuffer>> *buffers = [NSMutableArray array];
      const uint64_t bytes = uint64_t(plan.dim) * plan.ffn * sizeof(float);
      for (uint32_t i = 0; i < 3; ++i) {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)stageHandles[i];
        if (!pixel_buffer(buffer, plan.device, bytes)) {
          pixel_set_error(error, capacity, "staged FFN buffer has wrong device, storage or extent");
          return nullptr;
        }
        for (id<MTLBuffer> resident in plan.resident) {
          if (!pixel_disjoint(buffer, resident)) {
            pixel_set_error(error, capacity, "staged FFN weights overlap a block binding");
            return nullptr;
          }
        }
        for (id<MTLBuffer> earlier in buffers) {
          if (!pixel_disjoint(buffer, earlier)) {
            pixel_set_error(error, capacity, "staged FFN buffers overlap each other");
            return nullptr;
          }
        }
        [buffers addObject:buffer];
      }
      PixelUMMFFNWeightReuse *reuse = [PixelUMMFFNWeightReuse new];
      reuse.plan = plan;
      reuse.captureMask = plan.captureMask;
      reuse.stageBuffers = [buffers copy];
      reuse.baseline = @[plan.gateProjection, plan.upProjection, plan.downProjection];
      PixelUMMGraphStep *gate = pixel_reuse_linear_step(plan, plan.core[1], buffers[0],
          plan.core[6], plan.dim, plan.ffn, @"gate_projection", error, capacity);
      PixelUMMGraphStep *up = gate ? pixel_reuse_linear_step(plan, plan.core[1], buffers[1],
          plan.core[7], plan.dim, plan.ffn, @"up_projection", error, capacity) : nil;
      PixelUMMGraphStep *down = up ? pixel_reuse_linear_step(plan, plan.core[7], buffers[2],
          plan.core[4], plan.ffn, plan.dim, @"down_projection", error, capacity) : nil;
      if (!gate || !up || !down) return nullptr;
      reuse.staged = @[gate, up, down];
      MTLCompileOptions *options = [MTLCompileOptions new];
      if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
      else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
      }
      NSError *compileError = nil;
      id<MTLLibrary> library = [plan.device newLibraryWithSource:
          [NSString stringWithUTF8String:pixel_ffn_weight_reuse_source]
          options:options error:&compileError];
      id<MTLFunction> expand = [library newFunctionWithName:@"pixel_reuse_expand"];
      id<MTLFunction> compare = [library newFunctionWithName:@"pixel_reuse_compare"];
      reuse.expand = expand ? [plan.device newComputePipelineStateWithFunction:expand
          error:&compileError] : nil;
      reuse.compare = compare ? [plan.device newComputePipelineStateWithFunction:compare
          error:&compileError] : nil;
      if (!reuse.expand || !reuse.compare) {
        pixel_set_ns_error(error, capacity, [NSString stringWithFormat:
            @"FFN staging pipeline compilation failed: %@", compileError.localizedDescription]);
        return nullptr;
      }
      return (__bridge_retained void *)reuse;
    } @catch (NSException *exception) {
      pixel_set_ns_error(error, capacity, exception.reason);
      return nullptr;
    }
  }
}

// The probe owner serializes selection and only calls it between drained
// requests. Both alternatives, their buffers and captures remain resident.
extern "C" int32_t gs_pixelumm_ffn_weight_reuse_select(
    void *handle, int32_t enabled, int32_t captures, char *error, size_t capacity) {
  if (error && capacity) error[0] = '\0';
  if (!handle || (enabled != 0 && enabled != 1) || (captures != 0 && captures != 1)) {
    pixel_set_error(error, capacity, "invalid staged FFN selection");
    return -1;
  }
  PixelUMMFFNWeightReuse *reuse = (__bridge PixelUMMFFNWeightReuse *)handle;
  NSArray *steps = enabled ? reuse.staged : reuse.baseline;
  reuse.plan.gateProjection = steps[0];
  reuse.plan.upProjection = steps[1];
  reuse.plan.downProjection = steps[2];
  reuse.plan.captureMask = captures ? reuse.captureMask
      : (reuse.captureMask & (uint64_t(1) << PixelOutput));
  return 0;
}

// Host-side buffer identity only: no buffer contents, GPU command or wait.
// This checks the live native bindings separately from fixture declarations.
extern "C" int32_t gs_pixelumm_ffn_weight_reuse_bindings(
    void *handle, void **bindings, size_t count) {
  if (!handle || !bindings || count != 10) return -1;
  PixelUMMFFNWeightReuse *reuse = (__bridge PixelUMMFFNWeightReuse *)handle;
  if (reuse.plan.resident.count < 3 || reuse.plan.core.count != kPixelCoreCount ||
      reuse.plan.weights.count != kPixelWeightCount || reuse.stageBuffers.count != 3)
    return -2;
  for (uint32_t i = 0; i < 3; ++i)
    bindings[i] = (__bridge void *)reuse.plan.resident[i];
  bindings[3] = (__bridge void *)reuse.plan.core[8];
  for (uint32_t i = 0; i < 3; ++i) {
    bindings[4u + i] = (__bridge void *)reuse.plan.weights[10u + i];
    bindings[7u + i] = (__bridge void *)reuse.stageBuffers[i];
  }
  return 0;
}

struct PixelReuseCount { uint32_t count; uint32_t slot; };

static int32_t pixel_reuse_stage_or_check(void *handle, void *commandBufferHandle,
    void *metricsHandle, uint32_t slot, bool check, char *error, size_t capacity) {
  if (error && capacity) error[0] = '\0';
  if (!handle || !commandBufferHandle || (check && (!metricsHandle || slot >= 64))) {
    pixel_set_error(error, capacity, "invalid staged FFN command bindings");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMFFNWeightReuse *reuse = (__bridge PixelUMMFFNWeightReuse *)handle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandBufferHandle;
      id<MTLCommandBuffer> root = pixel_current_root(commandBuffer, reuse.plan.device,
          error, capacity);
      id<MTLBuffer> metrics = (__bridge id<MTLBuffer>)metricsHandle;
      if (check && (!metrics || metrics.device != reuse.plan.device ||
          metrics.storageMode != MTLStorageModeShared ||
          metrics.length < uint64_t(slot + 1u) * 4u * sizeof(uint32_t))) {
        pixel_set_error(error, capacity, "staged FFN metric buffer is too small or incompatible");
        return -1;
      }
      if (check) {
        for (id<MTLBuffer> buffer in reuse.plan.resident)
          if (!pixel_disjoint(metrics, buffer)) {
            pixel_set_error(error, capacity, "staged FFN metrics overlap a block binding");
            return -1;
          }
        for (id<MTLBuffer> buffer in reuse.stageBuffers)
          if (!pixel_disjoint(metrics, buffer)) {
            pixel_set_error(error, capacity, "staged FFN metrics overlap weight stage");
            return -1;
          }
      }
      if (!root) return -2;
      id<MTLComputeCommandEncoder> encoder = [root computeCommandEncoder];
      if (!encoder) return -2;
      id<MTLComputePipelineState> pipeline = check ? reuse.compare : reuse.expand;
      [encoder setComputePipelineState:pipeline];
      const uint32_t count = reuse.plan.dim * reuse.plan.ffn;
      PixelReuseCount params{count, slot};
      for (uint32_t i = 0; i < 3; ++i) {
        [encoder setBuffer:reuse.plan.weights[10u + i] offset:0 atIndex:0];
        [encoder setBuffer:reuse.stageBuffers[i] offset:0 atIndex:1];
        if (check) [encoder setBuffer:metrics offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:check ? 3 : 2];
        const NSUInteger threads = std::min<NSUInteger>(256,
            pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreadgroups:MTLSizeMake((uint64_t(count) + threads - 1) / threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      }
      [encoder endEncoding];
      return 0;
    } @catch (NSException *exception) {
      pixel_set_ns_error(error, capacity, exception.reason);
      return -3;
    }
  }
}

extern "C" int32_t gs_pixelumm_ffn_weight_reuse_stage(
    void *handle, void *commandBuffer, char *error, size_t capacity) {
  return pixel_reuse_stage_or_check(handle, commandBuffer, nullptr, 0, false, error, capacity);
}
extern "C" int32_t gs_pixelumm_ffn_weight_reuse_check_stage(
    void *handle, void *commandBuffer, void *metrics, uint32_t slot,
    char *error, size_t capacity) {
  return pixel_reuse_stage_or_check(handle, commandBuffer, metrics, slot, true, error, capacity);
}
extern "C" void gs_pixelumm_ffn_weight_reuse_release(void *handle) {
  if (!handle) return;
  @autoreleasepool {
    PixelUMMFFNWeightReuse *reuse = (__bridge_transfer PixelUMMFFNWeightReuse *)handle;
    reuse.plan.gateProjection = reuse.baseline[0];
    reuse.plan.upProjection = reuse.baseline[1];
    reuse.plan.downProjection = reuse.baseline[2];
    reuse.plan.captureMask = reuse.captureMask;
  }
}
