// Diagnostic-only whole-FFN MPSGraph candidate. The AA bridge includes the
// production bridge and supplies the connected baseline ABI; including it here
// keeps baseline and candidate in one object without duplicate production symbols.

#if __has_include(<Foundation/Foundation.h>) && \
    __has_include(<Metal/Metal.h>) && \
    __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)

#include "pixelumm_ffn_aa_bridge.mm"

static constexpr uint32_t kPixelFFNGraphTokens = 5760u;
static constexpr uint32_t kPixelFFNGraphHidden = 4096u;
static constexpr uint32_t kPixelFFNGraphWidth = 12288u;

@interface PixelUMMFFNGraphProbePlan : NSObject {
 @public
  uint64_t outputOnlyTopologyEncodes;
  uint64_t outputOnlySplitDownEncodes;
  uint64_t capturedTopologyEncodes;
  uint64_t capturedSplitDownEncodes;
  uint64_t referenceSiluEncodes;
}
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *borrowedBuffers;
@property(nonatomic, strong) PixelUMMGraphStep *outputOnly;
@property(nonatomic, strong) PixelUMMGraphStep *captured;
@property(nonatomic, strong) PixelUMMGraphStep *splitDown;
@property(nonatomic, assign) BOOL usesSplitDown;
@property(nonatomic, assign) BOOL stageOutputs;
@property(nonatomic, strong) id<MTLComputePipelineState> silu;
@end

@implementation PixelUMMFFNGraphProbePlan
@end

static bool pixel_ffngraph_sizes(uint64_t *hiddenBytes, uint64_t *wideBytes,
                                 uint64_t *weightBytes) {
  return pixel_product(kPixelFFNGraphTokens,
                       uint64_t(kPixelFFNGraphHidden) * sizeof(uint16_t),
                       hiddenBytes) &&
         pixel_product(kPixelFFNGraphTokens,
                       uint64_t(kPixelFFNGraphWidth) * sizeof(uint16_t),
                       wideBytes) &&
         pixel_product(kPixelFFNGraphWidth,
                       uint64_t(kPixelFFNGraphHidden) * sizeof(uint16_t),
                       weightBytes);
}

static PixelUMMGraphStep *pixel_ffngraph_build(
    id<MTLDevice> device, id<MTLBuffer> normInput,
    id<MTLBuffer> gateWeight, id<MTLBuffer> upWeight,
    id<MTLBuffer> downWeight, id<MTLBuffer> gateOutput,
    id<MTLBuffer> upOutput, id<MTLBuffer> siluOutput,
    id<MTLBuffer> productOutput, id<MTLBuffer> downOutput,
    bool captureStages, bool splitDown, bool stageOutputs,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *inputShape = pixel_shape({1, kPixelFFNGraphTokens,
                                     kPixelFFNGraphHidden});
  NSArray *wideWeightShape = pixel_shape({kPixelFFNGraphWidth,
                                          kPixelFFNGraphHidden});
  NSArray *downWeightShape = pixel_shape({kPixelFFNGraphHidden,
                                          kPixelFFNGraphWidth});
  NSArray *wideShape = pixel_shape({1, kPixelFFNGraphTokens,
                                    kPixelFFNGraphWidth});
  NSArray *hiddenShape = pixel_shape({1, kPixelFFNGraphTokens,
                                      kPixelFFNGraphHidden});

  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      normInput, inputShape, MPSDataTypeBFloat16, @"ffn_graph_input_bf16");
  MPSGraphTensor *gateWeightBf16 = pixel_add_feed(graph, feeds, placeholders,
      inputData, gateWeight, wideWeightShape, MPSDataTypeBFloat16,
      @"ffn_graph_gate_weight_bf16");
  MPSGraphTensor *upWeightBf16 = pixel_add_feed(graph, feeds, placeholders,
      inputData, upWeight, wideWeightShape, MPSDataTypeBFloat16,
      @"ffn_graph_up_weight_bf16");
  MPSGraphTensor *downWeightBf16 = splitDown ? nil : pixel_add_feed(
      graph, feeds, placeholders, inputData, downWeight, downWeightShape,
      MPSDataTypeBFloat16, @"ffn_graph_down_weight_bf16");

  MPSGraphTensor *inputF32 = input
      ? [graph castTensor:input toType:MPSDataTypeFloat32
          name:@"ffn_graph_input_f32"] : nil;
  MPSGraphTensor *gateWeightF32 = gateWeightBf16
      ? [graph castTensor:gateWeightBf16 toType:MPSDataTypeFloat32
          name:@"ffn_graph_gate_weight_f32"] : nil;
  MPSGraphTensor *upWeightF32 = upWeightBf16
      ? [graph castTensor:upWeightBf16 toType:MPSDataTypeFloat32
          name:@"ffn_graph_up_weight_f32"] : nil;
  MPSGraphTensor *downWeightF32 = downWeightBf16
      ? [graph castTensor:downWeightBf16 toType:MPSDataTypeFloat32
          name:@"ffn_graph_down_weight_f32"] : nil;
  MPSGraphTensor *flatInput = inputF32
      ? [graph reshapeTensor:inputF32
          withShape:pixel_shape({kPixelFFNGraphTokens, kPixelFFNGraphHidden})
          name:@"ffn_graph_input_flat_f32"] : nil;

  MPSGraphTensor *gateBf16 = pixel_linear_f32_weight_projection(
      graph, flatInput, gateWeightF32, kPixelFFNGraphTokens,
      kPixelFFNGraphWidth, @"ffn_graph_gate");
  MPSGraphTensor *upBf16 = pixel_linear_f32_weight_projection(
      graph, flatInput, upWeightF32, kPixelFFNGraphTokens,
      kPixelFFNGraphWidth, @"ffn_graph_up");

  // These casts state the intended BF16 production boundaries. The full graph
  // formulation alone does not prove that downstream work consumed a
  // materialized BF16 value; its captures and output are checked. Split-down
  // mode materializes productOutput before the production-helper down graph.
  MPSGraphTensor *gateF32 = gateBf16
      ? [graph castTensor:gateBf16 toType:MPSDataTypeFloat32
          name:@"ffn_graph_gate_rounded_f32"] : nil;
  MPSGraphTensor *upF32 = upBf16
      ? [graph castTensor:upBf16 toType:MPSDataTypeFloat32
          name:@"ffn_graph_up_rounded_f32"] : nil;
  MPSGraphTensor *minusOne = [graph constantWithScalar:-1.0
      dataType:MPSDataTypeFloat32];
  MPSGraphTensor *one = [graph constantWithScalar:1.0
      dataType:MPSDataTypeFloat32];
  MPSGraphTensor *negativeGate = gateF32 && minusOne
      ? [graph multiplicationWithPrimaryTensor:gateF32
          secondaryTensor:minusOne name:@"ffn_graph_negative_gate_f32"] : nil;
  MPSGraphTensor *exponential = negativeGate
      ? [graph exponentWithTensor:negativeGate
          name:@"ffn_graph_exp_negative_gate_f32"] : nil;
  MPSGraphTensor *denominator = one && exponential
      ? [graph additionWithPrimaryTensor:one secondaryTensor:exponential
          name:@"ffn_graph_sigmoid_denominator_f32"] : nil;
  MPSGraphTensor *sigmoid = one && denominator
      ? [graph divisionWithPrimaryTensor:one secondaryTensor:denominator
          name:@"ffn_graph_sigmoid_f32"] : nil;
  MPSGraphTensor *siluF32 = gateF32 && sigmoid
      ? [graph multiplicationWithPrimaryTensor:gateF32 secondaryTensor:sigmoid
          name:@"ffn_graph_silu_f32"] : nil;
  MPSGraphTensor *siluBf16 = siluF32
      ? pixel_cast_bf16(graph, siluF32, @"ffn_graph_silu_bf16") : nil;
  MPSGraphTensor *siluRoundedF32 = siluBf16
      ? [graph castTensor:siluBf16 toType:MPSDataTypeFloat32
          name:@"ffn_graph_silu_rounded_f32"] : nil;
  MPSGraphTensor *productF32 = siluRoundedF32 && upF32
      ? [graph multiplicationWithPrimaryTensor:siluRoundedF32
          secondaryTensor:upF32 name:@"ffn_graph_product_f32"] : nil;
  MPSGraphTensor *productBf16 = productF32
      ? pixel_cast_bf16(graph, productF32, @"ffn_graph_product_bf16") : nil;
  MPSGraphTensor *productRoundedF32 = !splitDown && productBf16
      ? [graph castTensor:productBf16 toType:MPSDataTypeFloat32
          name:@"ffn_graph_product_rounded_f32"] : nil;
  MPSGraphTensor *flatProduct = productRoundedF32
      ? [graph reshapeTensor:productRoundedF32
          withShape:pixel_shape({kPixelFFNGraphTokens, kPixelFFNGraphWidth})
          name:@"ffn_graph_product_flat_f32"] : nil;
  MPSGraphTensor *downWeightTranspose = downWeightF32
      ? [graph transposeTensor:downWeightF32 dimension:0 withDimension:1
          name:@"ffn_graph_down_weight_transpose_f32"] : nil;
  MPSGraphTensor *downF32 = flatProduct && downWeightTranspose
      ? [graph matrixMultiplicationWithPrimaryTensor:flatProduct
          secondaryTensor:downWeightTranspose name:@"ffn_graph_down_matmul_f32"] : nil;
  MPSGraphTensor *downBf16 = downF32
      ? pixel_cast_bf16(graph, downF32, @"ffn_graph_down_bf16") : nil;
  MPSGraphTensor *downOutputTensor = downBf16
      ? [graph reshapeTensor:downBf16 withShape:hiddenShape
          name:@"ffn_graph_down_output_bf16"] : nil;

  if (!input || !gateWeightBf16 || !upWeightBf16 ||
      !inputF32 || !gateWeightF32 || !upWeightF32 ||
      !flatInput || !gateBf16 || !upBf16 || !gateF32 || !upF32 || !minusOne ||
      !one || !negativeGate || !exponential || !denominator || !sigmoid ||
      !siluF32 || !siluBf16 || !siluRoundedF32 || !productF32 ||
      !productBf16 || (!splitDown &&
      (!downWeightBf16 || !downWeightF32 || !productRoundedF32 || !flatProduct ||
       !downWeightTranspose || !downF32 || !downBf16 || !downOutputTensor))) {
    pixel_set_error(errorText, errorCapacity,
                    "unable to construct whole-FFN diagnostic graph");
    return nil;
  }

  if (splitDown && (captureStages || stageOutputs)) {
    return pixel_compile_step_outputs(device, graph, feeds, placeholders,
        inputData, @[gateBf16, upBf16, siluBf16, productBf16],
        @[gateOutput, upOutput, siluOutput, productOutput],
        @[wideShape, wideShape, wideShape, wideShape],
        @"ffn_graph_captured_front", errorText, errorCapacity);
  }
  if (splitDown) {
    return pixel_compile_step(device, graph, feeds, placeholders, inputData,
        productBf16, productOutput, wideShape, @"ffn_graph_output_front",
        errorText, errorCapacity);
  }
  if (captureStages) {
    return pixel_compile_step_outputs(device, graph, feeds, placeholders,
        inputData, @[gateBf16, upBf16, siluBf16, productBf16,
                     downOutputTensor],
        @[gateOutput, upOutput, siluOutput, productOutput, downOutput],
        @[wideShape, wideShape, wideShape, wideShape, hiddenShape],
        @"ffn_graph_captured", errorText, errorCapacity);
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      downOutputTensor, downOutput, hiddenShape, @"ffn_graph_output_only",
      errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_ffngraph_build_split_down(
    id<MTLDevice> device, id<MTLBuffer> productInput,
    id<MTLBuffer> downWeight, id<MTLBuffer> downOutput,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *productShape = pixel_shape({1, kPixelFFNGraphTokens,
                                      kPixelFFNGraphWidth});
  NSArray *weightShape = pixel_shape({kPixelFFNGraphHidden,
                                     kPixelFFNGraphWidth});
  NSArray *outputShape = pixel_shape({1, kPixelFFNGraphTokens,
                                     kPixelFFNGraphHidden});
  MPSGraphTensor *product = pixel_add_feed(graph, feeds, placeholders,
      inputData, productInput, productShape, MPSDataTypeBFloat16,
      @"ffn_split_down_product_bf16");
  MPSGraphTensor *weight = pixel_add_feed(graph, feeds, placeholders,
      inputData, downWeight, weightShape, MPSDataTypeBFloat16,
      @"ffn_split_down_weight_bf16");
  MPSGraphTensor *productF32 = product
      ? [graph castTensor:product toType:MPSDataTypeFloat32
          name:@"ffn_split_down_product_f32"] : nil;
  MPSGraphTensor *weightF32 = weight
      ? [graph castTensor:weight toType:MPSDataTypeFloat32
          name:@"ffn_split_down_weight_f32"] : nil;
  MPSGraphTensor *flatProduct = productF32
      ? [graph reshapeTensor:productF32
          withShape:pixel_shape({kPixelFFNGraphTokens, kPixelFFNGraphWidth})
          name:@"ffn_split_down_product_flat_f32"] : nil;
  MPSGraphTensor *downBf16 = pixel_linear_f32_weight_projection(
      graph, flatProduct, weightF32, kPixelFFNGraphTokens,
      kPixelFFNGraphHidden, @"ffn_split_down_projection");
  if (!product || !weight || !productF32 || !weightF32 || !flatProduct ||
      !downBf16 || ![downBf16.shape isEqualToArray:outputShape]) {
    pixel_set_error(errorText, errorCapacity,
                    "unable to construct production-helper split down graph");
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      downBf16, downOutput, outputShape, @"ffn_graph_split_down",
      errorText, errorCapacity);
}

extern "C" void *gs_pixelumm_ffn_graph_probe_create(
    void *deviceHandle, void *normInputHandle, void *gateWeightHandle,
    void *upWeightHandle, void *downWeightHandle, void *gateOutputHandle,
    void *upOutputHandle, void *siluOutputHandle, void *productOutputHandle,
    void *downOutputHandle, int32_t tokens, int32_t topology,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!deviceHandle || !normInputHandle || !gateWeightHandle ||
      !upWeightHandle || !downWeightHandle || !gateOutputHandle ||
      !upOutputHandle || !siluOutputHandle || !productOutputHandle ||
      (topology < 0 || topology > 2) ||
      !downOutputHandle || tokens != int32_t(kPixelFFNGraphTokens)) {
    pixel_set_error(errorText, errorCapacity,
                    "invalid J5760 whole-FFN graph probe arguments");
    return nullptr;
  }

  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      @try {
        id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
        id<MTLBuffer> normInput = (__bridge id<MTLBuffer>)normInputHandle;
        id<MTLBuffer> gateWeight = (__bridge id<MTLBuffer>)gateWeightHandle;
        id<MTLBuffer> upWeight = (__bridge id<MTLBuffer>)upWeightHandle;
        id<MTLBuffer> downWeight = (__bridge id<MTLBuffer>)downWeightHandle;
        id<MTLBuffer> gateOutput = (__bridge id<MTLBuffer>)gateOutputHandle;
        id<MTLBuffer> upOutput = (__bridge id<MTLBuffer>)upOutputHandle;
        id<MTLBuffer> siluOutput = (__bridge id<MTLBuffer>)siluOutputHandle;
        id<MTLBuffer> productOutput = (__bridge id<MTLBuffer>)productOutputHandle;
        id<MTLBuffer> downOutput = (__bridge id<MTLBuffer>)downOutputHandle;
        uint64_t hiddenBytes = 0;
        uint64_t wideBytes = 0;
        uint64_t weightBytes = 0;
        if (!pixel_ffngraph_sizes(&hiddenBytes, &wideBytes, &weightBytes)) {
          pixel_set_error(errorText, errorCapacity,
                          "J5760 whole-FFN byte extent overflow");
          return nullptr;
        }

        NSArray<id<MTLBuffer>> *borrowed = @[
          normInput, gateWeight, upWeight, downWeight, gateOutput, upOutput,
          siluOutput, productOutput, downOutput,
        ];
        if (!device || !pixel_buffer(normInput, device, hiddenBytes) ||
            !pixel_buffer(gateWeight, device, weightBytes) ||
            !pixel_buffer(upWeight, device, weightBytes) ||
            !pixel_buffer(downWeight, device, weightBytes) ||
            !pixel_buffer(gateOutput, device, wideBytes) ||
            !pixel_buffer(upOutput, device, wideBytes) ||
            !pixel_buffer(siluOutput, device, wideBytes) ||
            !pixel_buffer(productOutput, device, wideBytes) ||
            !pixel_buffer(downOutput, device, hiddenBytes)) {
          pixel_set_error(errorText, errorCapacity,
              "J5760 whole-FFN buffers must have exact BF16 extent, shared storage, and the requested device");
          return nullptr;
        }
        for (NSUInteger i = 0; i < borrowed.count; ++i) {
          for (NSUInteger j = i + 1; j < borrowed.count; ++j) {
            if (!pixel_disjoint(borrowed[i], borrowed[j])) {
              pixel_set_error(errorText, errorCapacity,
                  "J5760 whole-FFN input, weights, and outputs must be pairwise disjoint");
              return nullptr;
            }
          }
        }

        PixelUMMFFNGraphProbePlan *plan = [PixelUMMFFNGraphProbePlan new];
        if (!plan) {
          pixel_set_error(errorText, errorCapacity,
                          "unable to allocate J5760 whole-FFN graph plan");
          return nullptr;
        }
        plan.device = device;
        plan.borrowedBuffers = borrowed;
        plan.usesSplitDown = topology != 0;
        plan.stageOutputs = topology == 2;
        plan.outputOnly = pixel_ffngraph_build(device, normInput, gateWeight,
            upWeight, downWeight, gateOutput, upOutput, siluOutput,
            productOutput, downOutput, false, plan.usesSplitDown,
            plan.stageOutputs,
            errorText, errorCapacity);
        if (!plan.outputOnly) return nullptr;
        plan.captured = pixel_ffngraph_build(device, normInput, gateWeight,
            upWeight, downWeight, gateOutput, upOutput, siluOutput,
            productOutput, downOutput, true, plan.usesSplitDown,
            plan.stageOutputs,
            errorText, errorCapacity);
        if (!plan.captured) return nullptr;
        if (plan.usesSplitDown) {
          plan.splitDown = pixel_ffngraph_build_split_down(
              device, productOutput, downWeight, downOutput,
              errorText, errorCapacity);
          if (!plan.splitDown) return nullptr;
        }

        NSError *metalError = nil;
        NSString *source = [NSString stringWithUTF8String:pixel_metal_source];
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
          options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
          options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        id<MTLLibrary> library = source
            ? [device newLibraryWithSource:source options:options error:&metalError]
            : nil;
        if (!library) {
          pixel_set_ns_error(errorText, errorCapacity,
              [NSString stringWithFormat:@"unable to compile production PixelUMM Metal source: %@",
                  metalError.localizedDescription ?: @"source unavailable"]);
          return nullptr;
        }
        plan.silu = pixel_pipeline(device, library, @"pixelumm_silu",
                                   errorText, errorCapacity);
        if (!plan.silu) return nullptr;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"J5760 whole-FFN graph setup exception %@: %@",
                exception.name, exception.reason]);
        return nullptr;
      }
    }
  }

  pixel_set_error(errorText, errorCapacity,
                  "J5760 FFN MPSGraph path requires macOS 14 or newer");
  return nullptr;
}

static int32_t pixel_ffngraph_encode(
    void *planHandle, void *commandBufferHandle, PixelUMMGraphStep *step,
    const char *label, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle || !commandBufferHandle || !step) {
    pixel_set_error(errorText, errorCapacity,
                    "invalid J5760 whole-FFN graph or command buffer");
    return -1;
  }
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      @try {
        PixelUMMFFNGraphProbePlan *plan =
            (__bridge PixelUMMFFNGraphProbePlan *)planHandle;
        MPSCommandBuffer *commandBuffer =
            (__bridge MPSCommandBuffer *)commandBufferHandle;
        if (!plan.device || !plan.outputOnly || !plan.captured ||
            (plan.usesSplitDown && !plan.splitDown) || !plan.silu ||
            !plan.borrowedBuffers || plan.borrowedBuffers.count != 9 ||
            !pixel_current_root(commandBuffer, plan.device,
                                errorText, errorCapacity)) return -2;
        if (!pixel_encode_graph(commandBuffer, step, errorText, errorCapacity))
          return -3;
        bool capturedPath = false;
        if (step == plan.outputOnly) ++plan->outputOnlyTopologyEncodes;
        else if (step == plan.captured) {
          ++plan->capturedTopologyEncodes;
          capturedPath = true;
        }
        else {
          pixel_set_error(errorText, errorCapacity,
                          "unknown J5760 whole-FFN graph step");
          return -4;
        }
        if (plan.usesSplitDown) {
          if (!pixel_encode_graph(commandBuffer, plan.splitDown,
                                  errorText, errorCapacity)) return -5;
          if (capturedPath) ++plan->capturedSplitDownEncodes;
          else ++plan->outputOnlySplitDownEncodes;
        }
        (void)label;
        return 0;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"J5760 whole-FFN graph encode exception %@: %@",
                exception.name, exception.reason]);
        return -5;
      }
    }
  }
  pixel_set_error(errorText, errorCapacity,
                  "J5760 FFN MPSGraph path requires macOS 14 or newer");
  return -6;
}

extern "C" int32_t gs_pixelumm_ffn_graph_probe_encode_output_only(
    void *planHandle, void *commandBufferHandle,
    char *errorText, size_t errorCapacity) {
  PixelUMMFFNGraphProbePlan *plan = planHandle
      ? (__bridge PixelUMMFFNGraphProbePlan *)planHandle : nil;
  return pixel_ffngraph_encode(planHandle, commandBufferHandle,
      plan.outputOnly, "output-only", errorText, errorCapacity);
}

extern "C" int32_t gs_pixelumm_ffn_graph_probe_encode_captured(
    void *planHandle, void *commandBufferHandle,
    char *errorText, size_t errorCapacity) {
  PixelUMMFFNGraphProbePlan *plan = planHandle
      ? (__bridge PixelUMMFFNGraphProbePlan *)planHandle : nil;
  return pixel_ffngraph_encode(planHandle, commandBufferHandle,
      plan.captured, "captured", errorText, errorCapacity);
}

extern "C" int32_t gs_pixelumm_ffn_graph_probe_derive_reference_silu(
    void *planHandle, void *commandBufferHandle, void *expectedGateHandle,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle || !commandBufferHandle || !expectedGateHandle) {
    pixel_set_error(errorText, errorCapacity,
                    "invalid J5760 whole-FFN SiLU reference arguments");
    return -1;
  }
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      @try {
        PixelUMMFFNGraphProbePlan *plan =
            (__bridge PixelUMMFFNGraphProbePlan *)planHandle;
        MPSCommandBuffer *commandBuffer =
            (__bridge MPSCommandBuffer *)commandBufferHandle;
        id<MTLBuffer> expectedGate =
            (__bridge id<MTLBuffer>)expectedGateHandle;
        uint64_t hiddenBytes = 0;
        uint64_t wideBytes = 0;
        uint64_t weightBytes = 0;
        if (!pixel_ffngraph_sizes(&hiddenBytes, &wideBytes, &weightBytes) ||
            !pixel_buffer(expectedGate, plan.device, wideBytes) ||
            !pixel_current_root(commandBuffer, plan.device,
                                errorText, errorCapacity)) return -2;
        PixelCountParams count{kPixelFFNGraphTokens * kPixelFFNGraphWidth};
        if (!pixel_encode_kernel(commandBuffer, plan.device, plan.silu,
                @[expectedGate], &count, sizeof(count), uint64_t(count.count),
                false, errorText, errorCapacity)) return -3;
        ++plan->referenceSiluEncodes;
        return 0;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"J5760 whole-FFN SiLU reference encode exception %@: %@",
                exception.name, exception.reason]);
        return -4;
      }
    }
  }
  pixel_set_error(errorText, errorCapacity,
                  "J5760 FFN MPSGraph path requires macOS 14 or newer");
  return -5;
}

extern "C" int32_t gs_pixelumm_ffn_graph_probe_stats(
    void *planHandle, uint64_t *counts, int32_t count) {
  if (!planHandle || !counts || count != 5) return -1;
  PixelUMMFFNGraphProbePlan *plan =
      (__bridge PixelUMMFFNGraphProbePlan *)planHandle;
  counts[0] = plan->outputOnlyTopologyEncodes;
  counts[1] = plan->outputOnlySplitDownEncodes;
  counts[2] = plan->capturedTopologyEncodes;
  counts[3] = plan->capturedSplitDownEncodes;
  counts[4] = plan->referenceSiluEncodes;
  return 0;
}

extern "C" void gs_pixelumm_ffn_graph_probe_release(void *planHandle) {
  if (!planHandle) return;
  @autoreleasepool {
    PixelUMMFFNGraphProbePlan *plan =
        (__bridge_transfer PixelUMMFFNGraphProbePlan *)planHandle;
    (void)plan;
  }
}

#else

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static void pixel_ffngraph_stub_error(char *errorText, size_t errorCapacity) {
  if (!errorText || errorCapacity == 0) return;
  const char *message = "J5760 FFN graph probe requires macOS 14 or newer";
  (void)snprintf(errorText, errorCapacity, "%s", message);
}

extern "C" void *gs_pixelumm_ffn_graph_probe_create(
    void *, void *, void *, void *, void *, void *, void *, void *, void *,
    void *, int32_t, int32_t, char *errorText, size_t errorCapacity) {
  pixel_ffngraph_stub_error(errorText, errorCapacity);
  return nullptr;
}
extern "C" int32_t gs_pixelumm_ffn_graph_probe_encode_output_only(
    void *, void *, char *errorText, size_t errorCapacity) {
  pixel_ffngraph_stub_error(errorText, errorCapacity);
  return -1;
}
extern "C" int32_t gs_pixelumm_ffn_graph_probe_encode_captured(
    void *, void *, char *errorText, size_t errorCapacity) {
  pixel_ffngraph_stub_error(errorText, errorCapacity);
  return -1;
}
extern "C" int32_t gs_pixelumm_ffn_graph_probe_derive_reference_silu(
    void *, void *, void *, char *errorText, size_t errorCapacity) {
  pixel_ffngraph_stub_error(errorText, errorCapacity);
  return -1;
}
extern "C" int32_t gs_pixelumm_ffn_graph_probe_stats(
    void *, uint64_t *, int32_t) { return -1; }
extern "C" void gs_pixelumm_ffn_graph_probe_release(void *) {}

#endif
