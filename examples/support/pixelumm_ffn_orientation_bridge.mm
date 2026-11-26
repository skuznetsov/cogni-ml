// Bounded diagnostic graph. The included weight-reuse bridge includes the
// production PixelUMM block ABI exactly once; do not link its object too.
#include "pixelumm_ffn_weight_reuse_bridge.mm"

#import <objc/runtime.h>

@interface PixelUMMFFNOrientation : NSObject
@property(nonatomic, strong) PixelUMMDecoderPlan *plan;
@property(nonatomic, strong) NSArray<PixelUMMGraphStep *> *baseline;
@property(nonatomic, strong) NSArray<PixelUMMGraphStep *> *candidate;
@property(nonatomic, strong) id installationToken;
@property(nonatomic) BOOL active;
@end
@implementation PixelUMMFFNOrientation
@end

// The marker detects a second orientation wrapper for the same plan without
// creating a plan -> wrapper -> plan retain cycle.
static char pixel_ffn_orientation_associated_key;

// Compute W @ transpose(X), then transpose the result back to [J, N]. This
// retains BF16 feeds, F32 operands/reduction and the original BF16 output
// boundary while exposing a different MPS matmul orientation.
static PixelUMMGraphStep *pixel_ffn_orientation_step(
    PixelUMMDecoderPlan *plan, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> weightBuffer, id<MTLBuffer> outputBuffer,
    uint32_t inputWidth, uint32_t outputWidth, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *inputShape = pixel_shape({1, plan.tokens, inputWidth});
  NSArray *weightShape = pixel_shape({outputWidth, inputWidth});
  NSArray *outputShape = pixel_shape({1, plan.tokens, outputWidth});

  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      inputBuffer, inputShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_input"]);
  MPSGraphTensor *weight = pixel_add_feed(graph, feeds, placeholders, inputData,
      weightBuffer, weightShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_weight"]);
  MPSGraphTensor *inputF32 = input
      ? [graph castTensor:input toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_input_f32"]] : nil;
  MPSGraphTensor *weightF32 = weight
      ? [graph castTensor:weight toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_weight_f32"]] : nil;
  MPSGraphTensor *flatInput = inputF32
      ? [graph reshapeTensor:inputF32
          withShape:pixel_shape({plan.tokens, inputWidth})
          name:[name stringByAppendingString:@"_flat_input"]] : nil;
  MPSGraphTensor *transposedInput = flatInput
      ? [graph transposeTensor:flatInput dimension:0 withDimension:1
          name:[name stringByAppendingString:@"_input_transpose"]] : nil;
  MPSGraphTensor *product = weightF32 && transposedInput
      ? [graph matrixMultiplicationWithPrimaryTensor:weightF32
          secondaryTensor:transposedInput
          name:[name stringByAppendingString:@"_matmul_weight_by_input_transpose"]] : nil;
  MPSGraphTensor *oriented = product
      ? [graph transposeTensor:product dimension:0 withDimension:1
          name:[name stringByAppendingString:@"_matmul_output_transpose"]] : nil;
  MPSGraphTensor *bf16 = pixel_cast_bf16(graph, oriented,
      [name stringByAppendingString:@"_bf16"]);
  MPSGraphTensor *output = bf16
      ? [graph reshapeTensor:bf16 withShape:outputShape
          name:[name stringByAppendingString:@"_output"]] : nil;

  if (!input || !weight || !inputF32 || !weightF32 || !flatInput ||
      !transposedInput || !product || !oriented || !bf16 || !output) {
    pixel_set_error(errorText, errorCapacity,
        "unable to build transposed-orientation F32 FFN graph");
    return nil;
  }
  return pixel_compile_step(plan.device, graph, feeds, placeholders, inputData,
      output, outputBuffer, outputShape, name, errorText, errorCapacity);
}

static bool pixel_ffn_orientation_has_no_attention_candidate(
    PixelUMMDecoderPlan *plan) {
  if (plan.softmaxPVPrepared || plan.useSoftmaxPV ||
      plan.scaleSoftmaxPrepared || plan.useScaleSoftmax ||
      plan.softmaxPVCompare || plan.probabilitySnapshot ||
      plan.attendedSnapshot || plan.softmaxPVCheckMetrics) return false;
  for (PixelUMMAttentionGeometry *geometry in plan.attentionGeometries)
    if (geometry.softmaxPV || geometry.scaleSoftmax) return false;
  return true;
}

static bool pixel_ffn_orientation_is_baseline(
    PixelUMMDecoderPlan *plan) {
  if (!plan.gateProjection || !plan.upProjection || !plan.downProjection)
    return false;
  NSArray<PixelUMMGraphStep *> *steps = @[
      plan.gateProjection, plan.upProjection, plan.downProjection];
  for (PixelUMMGraphStep *step in steps)
    if (![plan.steps containsObject:step]) return false;
  return true;
}

extern "C" void *gs_pixelumm_ffn_orientation_prepare(
    void *planHandle, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle) {
    pixel_set_error(errorText, errorCapacity,
        "FFN orientation requires a native PixelUMM plan");
    return nullptr;
  }
  @autoreleasepool {
    @try {
      PixelUMMDecoderPlan *plan = (__bridge PixelUMMDecoderPlan *)planHandle;
      @synchronized(plan) {
        if (!plan.ffnF32 || plan.encodingStarted || plan.tokens != 5760 ||
            plan.dim != 4096 || plan.ffn != 12288 ||
            plan.sharedGateUpPrepared || plan.useSharedGateUp ||
            plan.sharedGateUpProjection || plan->nativeIslands ||
            plan.useNativeIslands ||
            !pixel_ffn_orientation_has_no_attention_candidate(plan) ||
            objc_getAssociatedObject(plan,
                &pixel_ffn_orientation_associated_key) ||
            !pixel_ffn_orientation_is_baseline(plan)) {
          pixel_set_error(errorText, errorCapacity,
              "FFN orientation requires an idle baseline F32 T5760 K4096 N12288 plan without other candidates");
          return nullptr;
        }

        const uint64_t hiddenBytes = uint64_t(plan.tokens) * plan.dim * 2u;
        const uint64_t ffnBytes = uint64_t(plan.tokens) * plan.ffn * 2u;
        const uint64_t weightBytes = uint64_t(plan.dim) * plan.ffn * 2u;
        if (plan.core.count != kPixelCoreCount ||
            plan.weights.count != kPixelWeightCount ||
            !pixel_buffer(plan.core[1], plan.device, hiddenBytes) ||
            !pixel_buffer(plan.core[4], plan.device, hiddenBytes) ||
            !pixel_buffer(plan.core[6], plan.device, ffnBytes) ||
            !pixel_buffer(plan.core[7], plan.device, ffnBytes) ||
            !pixel_buffer(plan.weights[10], plan.device, weightBytes) ||
            !pixel_buffer(plan.weights[11], plan.device, weightBytes) ||
            !pixel_buffer(plan.weights[12], plan.device, weightBytes)) {
          pixel_set_error(errorText, errorCapacity,
              "FFN orientation plan buffers do not match BF16 T5760 K4096 N12288 geometry");
          return nullptr;
        }

        NSArray<PixelUMMGraphStep *> *baseline = @[
            plan.gateProjection, plan.upProjection, plan.downProjection];
        PixelUMMGraphStep *gate = pixel_ffn_orientation_step(plan,
            plan.core[1], plan.weights[10], plan.core[6], plan.dim, plan.ffn,
            @"orientation_gate_projection", errorText, errorCapacity);
        PixelUMMGraphStep *up = gate ? pixel_ffn_orientation_step(plan,
            plan.core[1], plan.weights[11], plan.core[7], plan.dim, plan.ffn,
            @"orientation_up_projection", errorText, errorCapacity) : nil;
        PixelUMMGraphStep *down = up ? pixel_ffn_orientation_step(plan,
            plan.core[7], plan.weights[12], plan.core[4], plan.ffn, plan.dim,
            @"orientation_down_projection", errorText, errorCapacity) : nil;
        if (!gate || !up || !down) return nullptr;

        PixelUMMFFNOrientation *wrapper = [PixelUMMFFNOrientation new];
        wrapper.plan = plan;
        wrapper.baseline = baseline;
        wrapper.candidate = @[gate, up, down];
        wrapper.installationToken = [NSObject new];
        wrapper.active = YES;
        objc_setAssociatedObject(plan, &pixel_ffn_orientation_associated_key,
            wrapper.installationToken, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        return (__bridge_retained void *)wrapper;
      }
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return nullptr;
    }
  }
}

// Call only from the serialized idle owner, after begin_request has restored
// the baseline graph selection and the preceding request's terminal wait drained.
extern "C" int32_t gs_pixelumm_ffn_orientation_select(
    void *handle, int32_t enabled, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!handle || (enabled != 0 && enabled != 1)) {
    pixel_set_error(errorText, errorCapacity,
        "invalid FFN orientation selection");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMFFNOrientation *wrapper = (__bridge PixelUMMFFNOrientation *)handle;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      if (!wrapper.active || !plan) {
        pixel_set_error(errorText, errorCapacity,
            "FFN orientation selection requires an active idle wrapper");
        return -2;
      }
      @synchronized(plan) {
        if (!wrapper.active || objc_getAssociatedObject(plan,
                &pixel_ffn_orientation_associated_key) !=
                wrapper.installationToken ||
            (plan.gateProjection != wrapper.baseline[0] &&
             plan.gateProjection != wrapper.candidate[0]) ||
            (plan.upProjection != wrapper.baseline[1] &&
             plan.upProjection != wrapper.candidate[1]) ||
            (plan.downProjection != wrapper.baseline[2] &&
             plan.downProjection != wrapper.candidate[2])) {
          pixel_set_error(errorText, errorCapacity,
              "FFN orientation selection found stale or foreign plan state");
          return -3;
        }
        NSArray<PixelUMMGraphStep *> *steps = enabled
            ? wrapper.candidate : wrapper.baseline;
        plan.gateProjection = steps[0];
        plan.upProjection = steps[1];
        plan.downProjection = steps[2];
        return 0;
      }
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return -4;
    }
  }
}

// Release only after the caller has drained all requests using this plan.
extern "C" void gs_pixelumm_ffn_orientation_release(void *handle) {
  if (!handle) return;
  @autoreleasepool {
    PixelUMMFFNOrientation *wrapper =
        (__bridge_transfer PixelUMMFFNOrientation *)handle;
    PixelUMMDecoderPlan *plan = wrapper.plan;
    if (!plan) return;
    @synchronized(plan) {
      if (objc_getAssociatedObject(plan,
              &pixel_ffn_orientation_associated_key) ==
          wrapper.installationToken) {
        plan.gateProjection = wrapper.baseline[0];
        plan.upProjection = wrapper.baseline[1];
        plan.downProjection = wrapper.baseline[2];
        objc_setAssociatedObject(plan, &pixel_ffn_orientation_associated_key,
            nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
      }
      wrapper.active = NO;
    }
  }
}
