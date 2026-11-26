// Bounded diagnostic only. This translation unit enables a compile-time
// guarded FFN hook in the included production bridge; do not also link its
// standalone object.
#define COGNI_PIXELUMM_FFN_DIRECT_MPS_DIAGNOSTIC 1
#include "pixelumm_ffn_weight_reuse_bridge.mm"

#import <objc/runtime.h>

static constexpr uint64_t kPixelFFNDirectMPSExtraBytes =
    uint64_t(12288) * uint64_t(4096) * sizeof(float);
static constexpr uint64_t kPixelFFNDirectMPSScoreBytes =
    uint64_t(32) * uint64_t(640) * uint64_t(4320) * sizeof(float);

struct PixelFFNDirectMPSCount { uint32_t count; };
static_assert(sizeof(PixelFFNDirectMPSCount) == sizeof(uint32_t),
              "PixelUMM direct MPS conversion ABI");

static const char *pixel_ffn_direct_mps_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct ConvertCount { uint count; };

kernel void pixel_ffn_direct_bf16_to_f32(
    device const ushort *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant ConvertCount &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < params.count)
    destination[index] = as_type<float>(uint(source[index]) << 16u);
}

inline ushort pixel_ffn_direct_f32_to_bf16_rne(float value) {
  const uint bits = as_type<uint>(value);
  const uint exponent = bits & 0x7f800000u;
  const uint mantissa = bits & 0x007fffffu;
  if (exponent == 0x7f800000u && mantissa != 0u) {
    // Preserve the high payload and sign, and quiet every NaN even when its
    // payload lives only in the discarded low 16 bits.
    return ushort((bits >> 16u) | 0x0040u);
  }
  const uint retainedLsb = (bits >> 16u) & 1u;
  return ushort((bits + 0x7fffu + retainedLsb) >> 16u);
}

kernel void pixel_ffn_direct_f32_to_bf16(
    device const float *source [[buffer(0)]],
    device ushort *destination [[buffer(1)]],
    constant ConvertCount &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < params.count)
    destination[index] = pixel_ffn_direct_f32_to_bf16_rne(source[index]);
}
)METAL";

@interface PixelUMMFFNDirectMSPPool : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLBuffer> convertedWeight;
@property(nonatomic, strong) id<MTLComputePipelineState> bf16ToF32;
@property(nonatomic, strong) id<MTLComputePipelineState> f32ToBf16;
@property(nonatomic, strong) MPSMatrix *hiddenInput;
@property(nonatomic, strong) MPSMatrix *ffnInput;
@property(nonatomic, strong) MPSMatrix *wideWeight;
@property(nonatomic, strong) MPSMatrix *downWeight;
@property(nonatomic, strong) MPSMatrix *ffnOutput;
@property(nonatomic, strong) MPSMatrix *hiddenOutput;
@property(nonatomic, strong) MPSMatrixMultiplication *gateUpMultiply;
@property(nonatomic, strong) MPSMatrixMultiplication *downMultiply;
@property(nonatomic, strong) id<MTLBuffer> inputScratch;
@property(nonatomic, strong) id<MTLBuffer> outputScratch;
@property(nonatomic) uint64_t extraBytes;
@end
@implementation PixelUMMFFNDirectMSPPool
@end

@interface PixelUMMFFNDirectMPSPlanState : NSObject
@property(nonatomic, strong) PixelUMMFFNDirectMSPPool *pool;
@property(nonatomic, strong) id installationToken;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *baselineResident;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *candidateResident;
@property(nonatomic, strong) id<MTLBuffer> gateInputBuffer;
@property(nonatomic) uint32_t projectionMask;
@property(nonatomic) BOOL active;
@property(nonatomic) BOOL selected;
@end
@implementation PixelUMMFFNDirectMPSPlanState
@end

@interface PixelUMMFFNDirectMPSWrapper : NSObject
@property(nonatomic, strong) PixelUMMDecoderPlan *plan;
@property(nonatomic, strong) PixelUMMFFNDirectMSPPool *pool;
@property(nonatomic, strong) PixelUMMFFNDirectMPSPlanState *state;
@end
@implementation PixelUMMFFNDirectMPSWrapper
@end

// The plan retains only a token-free state object which never retains the
// plan. The wrapper owns the plan; no plan -> wrapper -> plan cycle is formed.
static char pixel_ffn_direct_mps_associated_key;

static bool pixelumm_ffn_direct_mps_conflicts_attention(
    PixelUMMDecoderPlan *plan) {
  return plan && (plan.useDirectMPSFFN ||
      objc_getAssociatedObject(plan, &pixel_ffn_direct_mps_associated_key) != nil);
}

static NSMutableSet<NSValue *> *pixel_ffn_direct_mps_handles(void) {
  static NSMutableSet<NSValue *> *handles = [NSMutableSet set];
  return handles;
}

static NSValue *pixel_ffn_direct_mps_handle_key(void *handle) {
  return [NSValue valueWithPointer:handle];
}

static bool pixel_ffn_direct_mps_handle_registered(void *handle) {
  if (!handle) return false;
  NSMutableSet<NSValue *> *handles = pixel_ffn_direct_mps_handles();
  @synchronized(handles) {
    return [handles containsObject:pixel_ffn_direct_mps_handle_key(handle)];
  }
}

static id<MTLComputePipelineState> pixel_ffn_direct_mps_pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *name,
    char *errorText, size_t errorCapacity) {
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name];
  id<MTLComputePipelineState> pipeline = function
      ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
  if (!pipeline) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to prepare direct MPS FFN kernel %@: %@",
         name, error.localizedDescription ?: @"function missing"]);
  }
  return pipeline;
}

static MPSMatrix *pixel_ffn_direct_mps_matrix(
    id<MTLBuffer> buffer, uint32_t rows, uint32_t columns,
    char *errorText, size_t errorCapacity) {
  if (!buffer || !rows || !columns ||
      uint64_t(columns) * sizeof(float) > NSUIntegerMax ||
      uint64_t(rows) * uint64_t(columns) * sizeof(float) > buffer.length) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN matrix extent exceeds its shared scratch buffer");
    return nil;
  }
  const NSUInteger rowBytes = NSUInteger(columns) * sizeof(float);
  if ([MPSMatrixDescriptor rowBytesForColumns:columns
                                      dataType:MPSDataTypeFloat32] != rowBytes) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN requires compact F32 matrix row strides");
    return nil;
  }
  MPSMatrixDescriptor *descriptor =
      [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns
          rowBytes:rowBytes dataType:MPSDataTypeFloat32];
  MPSMatrix *matrix = descriptor
      ? [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor] : nil;
  if (!matrix) {
    pixel_set_error(errorText, errorCapacity,
        "unable to bind a direct MPS FFN matrix to shared scratch");
    return nil;
  }
  return matrix;
}

static bool pixel_ffn_direct_mps_validate_baseline(
    PixelUMMDecoderPlan *plan, char *errorText, size_t errorCapacity) {
  if (!plan || !plan.device || plan.encodingStarted || !plan.ffnF32 ||
      plan.tokens != 5760 || plan.temporal != 10 || plan.height != 18 ||
      plan.width != 32 || plan.tile != 8 || plan.dim != 4096 ||
      plan.ffn != 12288 || plan.heads != 32 || plan.kvHeads != 8 ||
      plan.headDim != 128 || plan.maxQuery != 640 || plan.maxKey != 4320 ||
      plan.tileCount != 12 || plan.sharedGateUpPrepared ||
      plan.useSharedGateUp || plan.sharedGateUpProjection ||
      plan.softmaxPVPrepared || plan.useSoftmaxPV ||
      plan.softmaxPVOutputOnlyPrepared || plan.useSoftmaxPVOutputOnly ||
      plan.scaleSoftmaxPrepared || plan.useScaleSoftmax ||
      plan.softmaxPVCompare || plan.probabilitySnapshot ||
      plan.attendedSnapshot || plan.softmaxPVCheckMetrics ||
      plan->nativeIslands || plan.useNativeIslands ||
      plan.core.count != kPixelCoreCount ||
      plan.weights.count != kPixelWeightCount ||
      plan.scratch.count != kPixelScratchCount ||
      !plan.steps || !plan.attentionGeometries ||
      plan.tileGeometryIndices.count != 12 ||
      plan.attentionGeometries.count == 0 ||
      objc_getAssociatedObject(plan, &pixel_ffn_direct_mps_associated_key)) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN requires an idle baseline F32 J5760 plan without another candidate");
    return false;
  }
  for (PixelUMMAttentionGeometry *geometry in plan.attentionGeometries) {
    if (!geometry || !geometry.qk || !geometry.softmax || !geometry.pv ||
        geometry.softmaxPV || geometry.softmaxPVOutputOnly || geometry.scaleSoftmax ||
        ![plan.steps containsObject:geometry.qk] ||
        ![plan.steps containsObject:geometry.softmax] ||
        ![plan.steps containsObject:geometry.pv]) {
      pixel_set_error(errorText, errorCapacity,
          "direct MPS FFN requires the original attention geometries and steps");
      return false;
    }
  }
  if (!plan.gateProjection || !plan.upProjection || !plan.downProjection ||
      ![plan.steps containsObject:plan.gateProjection] ||
      ![plan.steps containsObject:plan.upProjection] ||
      ![plan.steps containsObject:plan.downProjection]) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN requires the original gate, up and down graph steps");
    return false;
  }
  if (!plan.resident || plan.resident.count < 39u) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN baseline resident bindings are incomplete");
    return false;
  }

  const uint64_t hiddenBytes = uint64_t(5760) * 4096u * sizeof(uint16_t);
  const uint64_t ffnBytes = uint64_t(5760) * 12288u * sizeof(uint16_t);
  const uint64_t weightBytes = uint64_t(12288) * 4096u * sizeof(uint16_t);
  if (!pixel_buffer(plan.core[1], plan.device, hiddenBytes) ||
      !pixel_buffer(plan.core[4], plan.device, hiddenBytes) ||
      !pixel_buffer(plan.core[6], plan.device, ffnBytes) ||
      !pixel_buffer(plan.core[7], plan.device, ffnBytes) ||
      !pixel_buffer(plan.weights[10], plan.device, weightBytes) ||
      !pixel_buffer(plan.weights[11], plan.device, weightBytes) ||
      !pixel_buffer(plan.weights[12], plan.device, weightBytes) ||
      !pixel_buffer(plan.scratch[3], plan.device, kPixelFFNDirectMPSScoreBytes) ||
      !pixel_buffer(plan.scratch[4], plan.device, kPixelFFNDirectMPSScoreBytes)) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN input/output/weight/scratch buffers do not match exact extents");
    return false;
  }

  // Preserve the complete Owner resident prefix and reject any unrecognized
  // binding appended by another diagnostic. No buffer contents are read.
  NSMutableArray<id<MTLBuffer>> *expectedResident = [NSMutableArray array];
  for (NSUInteger index = 0; index < 3; ++index) {
    id<MTLBuffer> buffer = plan.resident[index];
    if (!pixel_buffer(buffer, plan.device, buffer.length)) {
      pixel_set_error(errorText, errorCapacity,
          "direct MPS FFN baseline input/table resident is invalid");
      return false;
    }
    [expectedResident addObject:buffer];
  }
  [expectedResident addObjectsFromArray:plan.weights];
  [expectedResident addObjectsFromArray:plan.core];
  [expectedResident addObjectsFromArray:plan.scratch];
  for (NSUInteger index = 0; index < kPixelCaptureCount; ++index) {
    if (index == PixelOutput) continue;
    id value = plan.captures[index];
    if (value && value != [NSNull null]) [expectedResident addObject:value];
  }
  if (expectedResident.count != plan.resident.count) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN resident list contains unexpected diagnostic resources");
    return false;
  }
  for (NSUInteger index = 0; index < expectedResident.count; ++index)
    if (expectedResident[index] != plan.resident[index]) {
      pixel_set_error(errorText, errorCapacity,
          "direct MPS FFN baseline resident prefix is not in its original order");
      return false;
    }

  id<MTLBuffer> scratchInput = plan.scratch[3];
  id<MTLBuffer> scratchOutput = plan.scratch[4];
  if (!pixel_disjoint(scratchInput, scratchOutput)) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN score and probability scratch buffers overlap");
    return false;
  }
  for (id<MTLBuffer> resident in plan.resident) {
    if (resident == scratchInput || resident == scratchOutput) continue;
    if (!pixel_disjoint(scratchInput, resident) ||
        !pixel_disjoint(scratchOutput, resident)) {
      pixel_set_error(errorText, errorCapacity,
          "direct MPS FFN scratch aliases a live input, weight, output or capture");
      return false;
    }
  }
  return true;
}

static PixelUMMFFNDirectMSPPool *pixel_ffn_direct_mps_create_pool(
    PixelUMMDecoderPlan *plan, char *errorText, size_t errorCapacity) {
  id<MTLBuffer> convertedWeight = [plan.device
      newBufferWithLength:(NSUInteger)kPixelFFNDirectMPSExtraBytes
                  options:MTLResourceStorageModeShared];
  if (!pixel_buffer(convertedWeight, plan.device,
                    kPixelFFNDirectMPSExtraBytes)) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN shared converted-weight allocation failed");
    return nil;
  }
  for (id<MTLBuffer> resident in plan.resident)
    if (!pixel_disjoint(convertedWeight, resident)) {
      pixel_set_error(errorText, errorCapacity,
          "direct MPS FFN converted-weight pool overlaps a baseline binding");
      return nil;
    }

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
      [NSString stringWithUTF8String:pixel_ffn_direct_mps_source]
      options:options error:&compileError];
  if (!library) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"direct MPS FFN conversion source failed: %@",
         compileError.localizedDescription]);
    return nil;
  }
  id<MTLComputePipelineState> bf16ToF32 = pixel_ffn_direct_mps_pipeline(
      plan.device, library, @"pixel_ffn_direct_bf16_to_f32",
      errorText, errorCapacity);
  id<MTLComputePipelineState> f32ToBf16 = bf16ToF32
      ? pixel_ffn_direct_mps_pipeline(plan.device, library,
          @"pixel_ffn_direct_f32_to_bf16", errorText, errorCapacity) : nil;
  if (!bf16ToF32 || !f32ToBf16) return nil;

  PixelUMMFFNDirectMSPPool *pool = [PixelUMMFFNDirectMSPPool new];
  pool.device = plan.device;
  pool.convertedWeight = convertedWeight;
  pool.bf16ToF32 = bf16ToF32;
  pool.f32ToBf16 = f32ToBf16;
  pool.inputScratch = plan.scratch[3];
  pool.outputScratch = plan.scratch[4];
  pool.extraBytes = convertedWeight.length;
  pool.hiddenInput = pixel_ffn_direct_mps_matrix(pool.inputScratch, 5760, 4096,
      errorText, errorCapacity);
  pool.ffnInput = pool.hiddenInput ? pixel_ffn_direct_mps_matrix(
      pool.inputScratch, 5760, 12288, errorText, errorCapacity) : nil;
  pool.wideWeight = pool.ffnInput ? pixel_ffn_direct_mps_matrix(
      pool.convertedWeight, 12288, 4096, errorText, errorCapacity) : nil;
  pool.downWeight = pool.wideWeight ? pixel_ffn_direct_mps_matrix(
      pool.convertedWeight, 4096, 12288, errorText, errorCapacity) : nil;
  pool.ffnOutput = pool.downWeight ? pixel_ffn_direct_mps_matrix(
      pool.outputScratch, 5760, 12288, errorText, errorCapacity) : nil;
  pool.hiddenOutput = pool.ffnOutput ? pixel_ffn_direct_mps_matrix(
      pool.outputScratch, 5760, 4096, errorText, errorCapacity) : nil;
  pool.gateUpMultiply = pool.hiddenOutput
      ? [[MPSMatrixMultiplication alloc] initWithDevice:plan.device
          transposeLeft:NO transposeRight:YES resultRows:5760
          resultColumns:12288 interiorColumns:4096 alpha:1.0 beta:0.0] : nil;
  pool.downMultiply = pool.gateUpMultiply
      ? [[MPSMatrixMultiplication alloc] initWithDevice:plan.device
          transposeLeft:NO transposeRight:YES resultRows:5760
          resultColumns:4096 interiorColumns:12288 alpha:1.0 beta:0.0] : nil;
  pool.gateUpMultiply.options = MPSKernelOptionsNone;
  pool.downMultiply.options = MPSKernelOptionsNone;
  if (!pool.hiddenInput || !pool.ffnInput || !pool.wideWeight ||
      !pool.downWeight || !pool.ffnOutput || !pool.hiddenOutput ||
      !pool.gateUpMultiply || !pool.downMultiply ||
      pool.extraBytes != kPixelFFNDirectMPSExtraBytes) {
    if (!errorText || !errorCapacity || errorText[0] == '\0')
      pixel_set_error(errorText, errorCapacity,
          "direct MPS FFN matrix preparation failed");
    return nil;
  }
  return pool;
}

static bool pixel_ffn_direct_mps_shared_wrapper(
    void *sharedHandle, PixelUMMFFNDirectMPSWrapper **shared,
    char *errorText, size_t errorCapacity) {
  if (!sharedHandle) {
    if (shared) *shared = nil;
    return true;
  }
  if (!pixel_ffn_direct_mps_handle_registered(sharedHandle)) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN shared wrapper handle is foreign or released");
    return false;
  }
  PixelUMMFFNDirectMPSWrapper *wrapper =
      (__bridge PixelUMMFFNDirectMPSWrapper *)sharedHandle;
  if (!wrapper.plan || !wrapper.pool || !wrapper.state.active ||
      wrapper.pool.extraBytes != kPixelFFNDirectMPSExtraBytes) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN shared wrapper is inactive or incomplete");
    return false;
  }
  if (shared) *shared = wrapper;
  return true;
}

extern "C" void *gs_pixelumm_ffn_direct_mps_prepare(
    void *planHandle, void *sharedWrapperHandle,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN preparation requires a native plan");
    return nullptr;
  }
  @autoreleasepool {
    @try {
      PixelUMMDecoderPlan *plan = (__bridge PixelUMMDecoderPlan *)planHandle;
      PixelUMMFFNDirectMPSWrapper *shared = nil;
      if (!pixel_ffn_direct_mps_shared_wrapper(sharedWrapperHandle, &shared,
              errorText, errorCapacity)) return nullptr;
      @synchronized(plan) {
        if (!pixel_ffn_direct_mps_validate_baseline(plan,
                errorText, errorCapacity)) return nullptr;
        PixelUMMFFNDirectMSPPool *pool = nil;
        if (shared) {
          if (shared.plan == plan || shared.pool.device != plan.device ||
              shared.pool.inputScratch != plan.scratch[3] ||
              shared.pool.outputScratch != plan.scratch[4]) {
            pixel_set_error(errorText, errorCapacity,
                "direct MPS FFN plans do not share the admitted device and scratch identity");
            return nullptr;
          }
          pool = shared.pool;
          if (!pixel_disjoint(pool.convertedWeight, plan.scratch[3]) ||
              !pixel_disjoint(pool.convertedWeight, plan.scratch[4])) {
            pixel_set_error(errorText, errorCapacity,
                "direct MPS FFN shared weight pool aliases reusable attention scratch");
            return nullptr;
          }
          for (id<MTLBuffer> resident in plan.resident)
            if (!pixel_disjoint(pool.convertedWeight, resident)) {
              pixel_set_error(errorText, errorCapacity,
                  "direct MPS FFN shared weight pool overlaps another plan binding");
              return nullptr;
            }
        } else {
          pool = pixel_ffn_direct_mps_create_pool(plan, errorText, errorCapacity);
          if (!pool) return nullptr;
        }

        PixelUMMFFNDirectMPSPlanState *state =
            [PixelUMMFFNDirectMPSPlanState new];
        PixelUMMFFNDirectMPSWrapper *wrapper =
            [PixelUMMFFNDirectMPSWrapper new];
        if (!state || !wrapper) {
          pixel_set_error(errorText, errorCapacity,
              "direct MPS FFN wrapper allocation failed");
          return nullptr;
        }
        state.pool = pool;
        state.installationToken = [NSObject new];
        state.baselineResident = plan.resident;
        state.candidateResident = [plan.resident arrayByAddingObject:pool.convertedWeight];
        state.active = YES;
        state.selected = NO;
        state.projectionMask = 0;
        if (!state.installationToken || !state.baselineResident ||
            !state.candidateResident ||
            state.candidateResident.count != state.baselineResident.count + 1u) {
          pixel_set_error(errorText, errorCapacity,
              "direct MPS FFN baseline/candidate resident setup failed");
          return nullptr;
        }
        wrapper.plan = plan;
        wrapper.pool = pool;
        wrapper.state = state;
        objc_setAssociatedObject(plan, &pixel_ffn_direct_mps_associated_key,
            state, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

        void *handle = (__bridge_retained void *)wrapper;
        NSMutableSet<NSValue *> *handles = pixel_ffn_direct_mps_handles();
        @synchronized(handles) {
          [handles addObject:pixel_ffn_direct_mps_handle_key(handle)];
        }
        return handle;
      }
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return nullptr;
    }
  }
}

extern "C" int32_t gs_pixelumm_ffn_direct_mps_select(
    void *handle, int32_t enabled, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!handle || (enabled != 0 && enabled != 1) ||
      !pixel_ffn_direct_mps_handle_registered(handle)) {
    pixel_set_error(errorText, errorCapacity,
        "invalid or foreign direct MPS FFN selector handle/flag");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMFFNDirectMPSWrapper *wrapper =
          (__bridge PixelUMMFFNDirectMPSWrapper *)handle;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      PixelUMMFFNDirectMPSPlanState *state = wrapper.state;
      if (!wrapper.pool || !plan || !state) {
        pixel_set_error(errorText, errorCapacity,
            "direct MPS FFN selector wrapper is incomplete");
        return -2;
      }
      @synchronized(plan) {
        id associated = objc_getAssociatedObject(plan,
            &pixel_ffn_direct_mps_associated_key);
        const bool residentKnown = plan.resident == state.baselineResident ||
                                   plan.resident == state.candidateResident;
        if (!state.active || associated != state || !residentKnown ||
            state.pool != wrapper.pool || wrapper.pool.extraBytes !=
                kPixelFFNDirectMPSExtraBytes ||
            (state.projectionMask != 0u && state.projectionMask != 7u)) {
          pixel_set_error(errorText, errorCapacity,
              "direct MPS FFN selector found stale or partially encoded plan state");
          return -3;
        }
        if (plan.useSharedGateUp || plan.useNativeIslands ||
            plan->nativeIslands || plan.softmaxPVPrepared ||
            plan.softmaxPVOutputOnlyPrepared || plan.scaleSoftmaxPrepared ||
            plan.softmaxPVCompare) {
          pixel_set_error(errorText, errorCapacity,
              "direct MPS FFN selector conflicts with another prepared mode");
          return -4;
        }
        if (enabled) {
          plan.resident = state.candidateResident;
          plan.useDirectMPSFFN = YES;
          state.selected = YES;
          state.projectionMask = 0;
          state.gateInputBuffer = nil;
        } else {
          plan.resident = state.baselineResident;
          plan.useDirectMPSFFN = NO;
          state.selected = NO;
          state.projectionMask = 0;
          state.gateInputBuffer = nil;
        }
        return 0;
      }
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return -5;
    }
  }
}

extern "C" uint64_t gs_pixelumm_ffn_direct_mps_extra_bytes(void *handle) {
  if (!pixel_ffn_direct_mps_handle_registered(handle)) return 0;
  PixelUMMFFNDirectMPSWrapper *wrapper =
      (__bridge PixelUMMFFNDirectMPSWrapper *)handle;
  return wrapper.state.active && wrapper.pool
      ? wrapper.pool.extraBytes : 0;
}

extern "C" int32_t gs_pixelumm_ffn_direct_mps_release(
    void *handle, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!handle) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN release requires a live wrapper");
    return -1;
  }
  if (!pixel_ffn_direct_mps_handle_registered(handle)) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN release received a foreign or released wrapper");
    return -2;
  }
  @autoreleasepool {
    @try {
      PixelUMMFFNDirectMPSWrapper *wrapper =
          (__bridge PixelUMMFFNDirectMPSWrapper *)handle;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      PixelUMMFFNDirectMPSPlanState *state = wrapper.state;
      if (!plan || !state || !wrapper.pool || state.pool != wrapper.pool) {
        pixel_set_error(errorText, errorCapacity,
            "direct MPS FFN release wrapper is incomplete");
        return -3;
      }
      NSMutableSet<NSValue *> *handles = pixel_ffn_direct_mps_handles();
      NSValue *key = pixel_ffn_direct_mps_handle_key(handle);
      @synchronized(plan) {
        id associated = objc_getAssociatedObject(plan,
            &pixel_ffn_direct_mps_associated_key);
        const bool baselineResident = plan.resident == state.baselineResident;
        const bool candidateResident = plan.resident == state.candidateResident;
        if (!state.active || associated != state ||
            (!baselineResident && !candidateResident) ||
            state.selected != candidateResident ||
            plan.useDirectMPSFFN != candidateResident ||
            !state.baselineResident || !state.candidateResident) {
          pixel_set_error(errorText, errorCapacity,
              "direct MPS FFN release found foreign plan state");
          return -4;
        }
        // Keep lock order consistent with prepare (plan, then handle registry).
        // On every validation failure the registered retained handle and plan
        // association remain intact so the caller can repair and retry.
        @synchronized(handles) {
          if (![handles containsObject:key]) {
            pixel_set_error(errorText, errorCapacity,
                "direct MPS FFN release wrapper ceased to be registered");
            return -5;
          }
          plan.resident = state.baselineResident;
          plan.useDirectMPSFFN = NO;
          state.selected = NO;
          state.active = NO;
          state.projectionMask = 0;
          state.gateInputBuffer = nil;
          objc_setAssociatedObject(plan, &pixel_ffn_direct_mps_associated_key,
              nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
          [handles removeObject:key];
        }
      }
      // Consume the retained wrapper only after plan restoration and registry
      // removal have both succeeded.
      PixelUMMFFNDirectMPSWrapper *consumed =
          (__bridge_transfer PixelUMMFFNDirectMPSWrapper *)handle;
      (void)consumed;
      return 0;
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return -5;
    }
  }
}

static bool pixelumm_ffn_direct_mps_encode_projection(
    PixelUMMDecoderPlan *plan, MPSCommandBuffer *commandBuffer,
    PixelUMMFFNDirectProjection projection,
    char *errorText, size_t errorCapacity) {
  if (!plan || !commandBuffer) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN projection has no plan or command chain");
    return false;
  }
  // MPSGraph may commit the current root and call commitAndContinue while
  // encoding earlier stages. Resolve the root at each projection boundary.
  id<MTLCommandBuffer> root = pixel_current_root(commandBuffer, plan.device,
                                                errorText, errorCapacity);
  if (!root) return false;
  PixelUMMFFNDirectMPSPlanState *state = objc_getAssociatedObject(
      plan, &pixel_ffn_direct_mps_associated_key);
  PixelUMMFFNDirectMSPPool *pool = state.pool;
  if (!state || !state.active || !state.selected || !pool ||
      !plan.useDirectMPSFFN || pool.device != plan.device ||
      pool.inputScratch != plan.scratch[3] ||
      pool.outputScratch != plan.scratch[4] ||
      plan.resident != state.candidateResident ||
      state.projectionMask > 7u) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN projection has stale selected resources");
    return false;
  }
  id<MTLBuffer> input = nil;
  id<MTLBuffer> weight = nil;
  id<MTLBuffer> output = nil;
  MPSMatrix *inputMatrix = nil;
  MPSMatrix *weightMatrix = nil;
  MPSMatrix *outputMatrix = nil;
  MPSMatrixMultiplication *multiply = nil;
  uint32_t inputWidth = 0, outputWidth = 0;
  uint32_t expectedMask = 0;
  switch (projection) {
    case PixelUMMFFNDirectGate:
      expectedMask = 0u;
      input = plan.core[1]; weight = plan.weights[10]; output = plan.core[6];
      inputWidth = 4096; outputWidth = 12288;
      inputMatrix = pool.hiddenInput; weightMatrix = pool.wideWeight;
      outputMatrix = pool.ffnOutput; multiply = pool.gateUpMultiply;
      break;
    case PixelUMMFFNDirectUp:
      expectedMask = 1u;
      input = plan.core[1]; weight = plan.weights[11]; output = plan.core[7];
      inputWidth = 4096; outputWidth = 12288;
      inputMatrix = pool.hiddenInput; weightMatrix = pool.wideWeight;
      outputMatrix = pool.ffnOutput; multiply = pool.gateUpMultiply;
      if (state.gateInputBuffer != input) {
        pixel_set_error(errorText, errorCapacity,
            "direct MPS FFN up projection cannot reuse a different gate input panel");
        return false;
      }
      break;
    case PixelUMMFFNDirectDown:
      expectedMask = 3u;
      input = plan.core[7]; weight = plan.weights[12]; output = plan.core[4];
      inputWidth = 12288; outputWidth = 4096;
      inputMatrix = pool.ffnInput; weightMatrix = pool.downWeight;
      outputMatrix = pool.hiddenOutput; multiply = pool.downMultiply;
      break;
    default:
      pixel_set_error(errorText, errorCapacity,
          "direct MPS FFN projection kind is invalid");
      return false;
  }
  if (state.projectionMask != expectedMask || !input || !weight || !output ||
      !inputMatrix || !weightMatrix || !outputMatrix || !multiply ||
      !pixel_buffer(input, plan.device,
          uint64_t(5760) * inputWidth * sizeof(uint16_t)) ||
      !pixel_buffer(weight, plan.device,
          uint64_t(inputWidth) * outputWidth * sizeof(uint16_t)) ||
      !pixel_buffer(output, plan.device,
          uint64_t(5760) * outputWidth * sizeof(uint16_t))) {
    pixel_set_error(errorText, errorCapacity,
        "direct MPS FFN projection buffers do not match the selected geometry");
    return false;
  }

  PixelFFNDirectMPSCount inputCount{5760u * inputWidth};
  PixelFFNDirectMPSCount weightCount{inputWidth * outputWidth};
  PixelFFNDirectMPSCount outputCount{5760u * outputWidth};
  if (projection != PixelUMMFFNDirectUp &&
      !pixel_encode_kernel(commandBuffer, plan.device, pool.bf16ToF32,
          @[input, pool.inputScratch], &inputCount, sizeof(inputCount),
          inputCount.count, false, errorText, errorCapacity)) return false;
  if (!pixel_encode_kernel(commandBuffer, plan.device, pool.bf16ToF32,
          @[weight, pool.convertedWeight], &weightCount, sizeof(weightCount),
          weightCount.count, false, errorText, errorCapacity)) return false;

  [multiply encodeToCommandBuffer:root leftMatrix:inputMatrix
      rightMatrix:weightMatrix resultMatrix:outputMatrix];
  if (!pixel_encode_kernel(commandBuffer, plan.device, pool.f32ToBf16,
          @[pool.outputScratch, output], &outputCount, sizeof(outputCount),
          outputCount.count, false, errorText, errorCapacity)) return false;

  if (projection == PixelUMMFFNDirectGate) {
    state.gateInputBuffer = input;
    state.projectionMask = 1u;
  } else if (projection == PixelUMMFFNDirectUp) {
    state.projectionMask = 3u;
  } else {
    state.projectionMask = 7u;
  }
  return true;
}
