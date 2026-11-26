// Isolated diagnostic graph. The included weight-reuse bridge includes the
// production PixelUMM block ABI exactly once; do not link its object too.
#include "pixelumm_ffn_weight_reuse_bridge.mm"

@interface PixelUMMPackedGateUp : NSObject
@property(nonatomic, strong) PixelUMMDecoderPlan *plan;
@property(nonatomic, strong) PixelUMMGraphStep *packedProjection;
@property(nonatomic, strong) PixelUMMGraphStep *baselineProjection;
@property(nonatomic) BOOL baselinePrepared;
@property(nonatomic) BOOL baselineSelected;
@property(nonatomic) BOOL active;
@end
@implementation PixelUMMPackedGateUp
@end

static PixelUMMGraphStep *pixel_packed_gate_up_step(
    PixelUMMDecoderPlan *plan, char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  const uint32_t tokens = plan.tokens;
  const uint32_t hidden = plan.dim;
  const uint32_t ffn = plan.ffn;

  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      plan.core[1], pixel_shape({1, tokens, hidden}), MPSDataTypeBFloat16,
      @"packed_gate_up_input");
  MPSGraphTensor *gateWeight = pixel_add_feed(graph, feeds, placeholders, inputData,
      plan.weights[10], pixel_shape({ffn, hidden}), MPSDataTypeBFloat16,
      @"packed_gate_weight");
  MPSGraphTensor *upWeight = pixel_add_feed(graph, feeds, placeholders, inputData,
      plan.weights[11], pixel_shape({ffn, hidden}), MPSDataTypeBFloat16,
      @"packed_up_weight");

  MPSGraphTensor *weights = gateWeight && upWeight
      ? [graph concatTensors:@[gateWeight, upWeight] dimension:0
          name:@"packed_gate_up_weight_concat"] : nil;
  MPSGraphTensor *inputF32 = input
      ? [graph castTensor:input toType:MPSDataTypeFloat32 name:@"packed_gate_up_input_f32"] : nil;
  MPSGraphTensor *weightsF32 = weights
      ? [graph castTensor:weights toType:MPSDataTypeFloat32 name:@"packed_gate_up_weights_f32"] : nil;
  MPSGraphTensor *flatInput = inputF32
      ? [graph reshapeTensor:inputF32 withShape:pixel_shape({tokens, hidden})
          name:@"packed_gate_up_input_flat"] : nil;
  MPSGraphTensor *transposedWeights = weightsF32
      ? [graph transposeTensor:weightsF32 dimension:0 withDimension:1
          name:@"packed_gate_up_weights_transpose"] : nil;
  MPSGraphTensor *product = flatInput && transposedWeights
      ? [graph matrixMultiplicationWithPrimaryTensor:flatInput
          secondaryTensor:transposedWeights name:@"packed_gate_up_single_matmul"] : nil;

  // Preserve the production BF16 boundary across the combined projection:
  // cast the full F32 result before splitting it into the original outputs.
  MPSGraphTensor *productBF16 = pixel_cast_bf16(graph, product,
      @"packed_gate_up_product_bf16");
  MPSGraphTensor *product3D = productBF16
      ? [graph reshapeTensor:productBF16
          withShape:pixel_shape({1, tokens, 2u * ffn})
          name:@"packed_gate_up_product_3d"] : nil;
  MPSGraphTensor *gate = product3D
      ? [graph sliceTensor:product3D dimension:2 start:0 length:ffn
          name:@"packed_gate_output"] : nil;
  MPSGraphTensor *up = product3D
      ? [graph sliceTensor:product3D dimension:2 start:ffn length:ffn
          name:@"packed_up_output"] : nil;

  if (!input || !gateWeight || !upWeight || !weights || !inputF32 ||
      !weightsF32 || !flatInput || !transposedWeights || !product ||
      !productBF16 || !product3D || !gate || !up) {
    pixel_set_error(errorText, errorCapacity,
        "unable to build packed gate/up F32 graph with BF16 output boundary");
    return nil;
  }

  NSArray *outputShape = pixel_shape({1, tokens, ffn});
  return pixel_compile_step_outputs(plan.device, graph, feeds, placeholders,
      inputData, @[gate, up], @[plan.core[6], plan.core[7]],
      @[outputShape, outputShape], @"packed_gate_up_projection",
      errorText, errorCapacity);
}

extern "C" void *gs_pixelumm_packed_gate_up_prepare(
    void *planHandle, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle) {
    pixel_set_error(errorText, errorCapacity, "packed gate/up requires a native plan");
    return nullptr;
  }
  @autoreleasepool {
    @try {
      PixelUMMDecoderPlan *plan = (__bridge PixelUMMDecoderPlan *)planHandle;
      if (!plan.ffnF32 || plan.encodingStarted || plan.dim != 4096 ||
          plan.ffn != 12288 || plan.tokens != 5760 ||
          plan.sharedGateUpPrepared || plan.useSharedGateUp ||
          plan.sharedGateUpProjection) {
        pixel_set_error(errorText, errorCapacity,
            "packed gate/up requires an idle baseline F32 plan with T5760 K4096 N12288");
        return nullptr;
      }
      const uint64_t inputBytes = uint64_t(plan.tokens) * plan.dim * 2u;
      const uint64_t weightBytes = uint64_t(plan.ffn) * plan.dim * 2u;
      const uint64_t outputBytes = uint64_t(plan.tokens) * plan.ffn * 2u;
      if (plan.core.count <= 7 || plan.weights.count <= 11 ||
          !pixel_buffer(plan.core[1], plan.device, inputBytes) ||
          !pixel_buffer(plan.weights[10], plan.device, weightBytes) ||
          !pixel_buffer(plan.weights[11], plan.device, weightBytes) ||
          !pixel_buffer(plan.core[6], plan.device, outputBytes) ||
          !pixel_buffer(plan.core[7], plan.device, outputBytes)) {
        pixel_set_error(errorText, errorCapacity,
            "packed gate/up plan buffers do not match BF16 K4096/N12288 geometry");
        return nullptr;
      }

      PixelUMMGraphStep *projection = pixel_packed_gate_up_step(
          plan, errorText, errorCapacity);
      if (!projection) return nullptr;

      PixelUMMPackedGateUp *wrapper = [PixelUMMPackedGateUp new];
      wrapper.plan = plan;
      wrapper.packedProjection = projection;
      @synchronized(plan) {
        // The caller serializes preparation while idle. Recheck before
        // replacing fields in case another diagnostic touched this plan.
        if (plan.encodingStarted || plan.sharedGateUpPrepared ||
            plan.useSharedGateUp || plan.sharedGateUpProjection) {
          pixel_set_error(errorText, errorCapacity,
              "packed gate/up plan changed during idle preparation");
          return nullptr;
        }
        wrapper.baselineProjection = plan.sharedGateUpProjection;
        wrapper.baselinePrepared = plan.sharedGateUpPrepared;
        wrapper.baselineSelected = plan.useSharedGateUp;
        wrapper.active = YES;
        plan.sharedGateUpProjection = projection;
        plan.sharedGateUpPrepared = YES;
        plan.useSharedGateUp = NO;
      }
      return (__bridge_retained void *)wrapper;
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return nullptr;
    }
  }
}

// Select only while the serialized caller is idle, after the previous
// request's terminal wait and after B::Owner has reset the baseline graphs.
extern "C" int32_t gs_pixelumm_packed_gate_up_select(
    void *handle, int32_t enabled, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!handle || (enabled != 0 && enabled != 1)) {
    pixel_set_error(errorText, errorCapacity, "invalid packed gate/up selection");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMPackedGateUp *wrapper = (__bridge PixelUMMPackedGateUp *)handle;
      if (!wrapper.active || !wrapper.plan || !wrapper.packedProjection) {
        pixel_set_error(errorText, errorCapacity,
            "packed gate/up selection requires an active idle wrapper");
        return -2;
      }
      return gs_pixelumm_decoder_block_plan_set_shared_gate_up(
          (__bridge void *)wrapper.plan, enabled, errorText, errorCapacity);
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return -3;
    }
  }
}

extern "C" void gs_pixelumm_packed_gate_up_release(void *handle) {
  if (!handle) return;
  @autoreleasepool {
    PixelUMMPackedGateUp *wrapper =
        (__bridge_transfer PixelUMMPackedGateUp *)handle;
    PixelUMMDecoderPlan *plan = wrapper.plan;
    if (!plan) return;
    @synchronized(plan) {
      plan.sharedGateUpProjection = wrapper.baselineProjection;
      plan.sharedGateUpPrepared = wrapper.baselinePrepared;
      plan.useSharedGateUp = wrapper.baselineSelected;
      wrapper.active = NO;
    }
  }
}
