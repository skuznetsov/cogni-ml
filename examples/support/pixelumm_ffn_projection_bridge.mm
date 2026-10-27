// Diagnostic-only, isolated PixelUMM FFN projection graphs. This translation
// unit includes the production bridge so the projection probe can reuse its
// static MPSGraph construction, validation, and encode helpers. Do not link the
// separately compiled pixelumm_decoder_block_mps_bridge object with this file.
#include "../../src/ml/video/pixelumm_decoder_block_mps_bridge.mm"

#if __has_include(<Foundation/Foundation.h>) && \
    __has_include(<Metal/Metal.h>) && \
    __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)

@interface PixelUMMFFNProjectionPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) PixelUMMGraphStep *step;
@property(nonatomic) uint32_t matmulOutputType;
@end

@implementation PixelUMMFFNProjectionPlan
@end

enum PixelUMMFFNProjectionMode : int32_t {
  PixelUMMFFNProjectionBF16 = 0,
  PixelUMMFFNProjectionF32 = 1,
  PixelUMMFFNProjectionF16RoundedF32 = 2,
};

// MPSDataTypeBFloat16 includes MPSDataTypeAlternateEncodingBit (0x80000000),
// so keep reported graph datatypes unsigned across the C ABI.

static MPSGraphTensor *pixelumm_projection_cast(
    MPSGraph *graph, MPSGraphTensor *tensor, MPSDataType type,
    NSString *name) {
  return graph && tensor
      ? [graph castTensor:tensor toType:type name:name]
      : nil;
}

static PixelUMMGraphStep *pixelumm_projection_graph_step(
    id<MTLDevice> device, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> weightBuffer, id<MTLBuffer> outputBuffer,
    uint32_t tokens, uint32_t inputWidth, uint32_t outputWidth,
    PixelUMMFFNProjectionMode mode, MPSDataType *matmulOutputType,
    char *errorText, size_t errorCapacity) {
  if (!device || !inputBuffer || !weightBuffer || !outputBuffer ||
      !matmulOutputType || mode < PixelUMMFFNProjectionBF16 ||
      mode > PixelUMMFFNProjectionF16RoundedF32) {
    pixel_set_error(errorText, errorCapacity,
                    "invalid PixelUMM FFN projection graph arguments");
    return nil;
  }

  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSString *name = @"ffn_projection";
  NSArray *inputShape = pixel_shape({1, tokens, inputWidth});
  NSArray *weightShape = pixel_shape({outputWidth, inputWidth});
  NSArray *outputShape = pixel_shape({1, tokens, outputWidth});

  MPSGraphTensor *input = pixel_add_feed(
      graph, feeds, placeholders, inputData, inputBuffer, inputShape,
      MPSDataTypeBFloat16, @"ffn_projection_input");
  MPSGraphTensor *weight = pixel_add_feed(
      graph, feeds, placeholders, inputData, weightBuffer, weightShape,
      MPSDataTypeBFloat16, @"ffn_projection_weight");
  MPSGraphTensor *matrixInput = input;
  MPSGraphTensor *matrixWeight = weight;
  if (mode == PixelUMMFFNProjectionF32) {
    matrixInput = pixelumm_projection_cast(
        graph, input, MPSDataTypeFloat32, @"ffn_projection_input_f32");
    matrixWeight = pixelumm_projection_cast(
        graph, weight, MPSDataTypeFloat32, @"ffn_projection_weight_f32");
  } else if (mode == PixelUMMFFNProjectionF16RoundedF32) {
    MPSGraphTensor *inputF16 = pixelumm_projection_cast(
        graph, input, MPSDataTypeFloat16, @"ffn_projection_input_f16");
    MPSGraphTensor *weightF16 = pixelumm_projection_cast(
        graph, weight, MPSDataTypeFloat16, @"ffn_projection_weight_f16");
    matrixInput = pixelumm_projection_cast(
        graph, inputF16, MPSDataTypeFloat32, @"ffn_projection_input_f16_f32");
    matrixWeight = pixelumm_projection_cast(
        graph, weightF16, MPSDataTypeFloat32, @"ffn_projection_weight_f16_f32");
  }

  MPSGraphTensor *flat = matrixInput
      ? [graph reshapeTensor:matrixInput withShape:pixel_shape({tokens, inputWidth})
              name:@"ffn_projection_flat"]
      : nil;
  MPSGraphTensor *transpose = matrixWeight
      ? [graph transposeTensor:matrixWeight dimension:0 withDimension:1
                             name:@"ffn_projection_weight_transpose"]
      : nil;
  MPSGraphTensor *product = flat && transpose
      ? [graph matrixMultiplicationWithPrimaryTensor:flat secondaryTensor:transpose
                                               name:@"ffn_projection_matmul"]
      : nil;
  if (!product) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM FFN projection graph %@", name]);
    return nil;
  }

  // Report the raw matrix node's graph datatype, before any output-boundary
  // cast. In particular, the BF16 control mirrors pixel_linear_step's dynamic
  // cast rule while exposing the type that rule observes on this OS/backend.
  *matmulOutputType = product.dataType;
  MPSGraphTensor *bf16 = product;
  if (bf16.dataType != MPSDataTypeBFloat16)
    bf16 = pixel_cast_bf16(graph, bf16, @"ffn_projection_output_bf16");
  MPSGraphTensor *reshaped = bf16
      ? [graph reshapeTensor:bf16 withShape:outputShape
                       name:@"ffn_projection_output"]
      : nil;
  if (!input || !weight || !matrixInput || !matrixWeight || !flat ||
      !transpose || !bf16 || !reshaped) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to finish PixelUMM FFN projection graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      reshaped, outputBuffer, outputShape, name, errorText, errorCapacity);
}

extern "C" void *gs_pixelumm_projection_create(
    void *deviceHandle, void *inputHandle, void *weightHandle,
    void *outputHandle, int32_t tokens, int32_t inputWidth,
    int32_t outputWidth, int32_t mode, uint32_t *matmulOutputType,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (matmulOutputType) *matmulOutputType = (uint32_t)MPSDataTypeInvalid;
  if (!deviceHandle || !inputHandle || !weightHandle || !outputHandle ||
      !matmulOutputType) {
    pixel_set_error(errorText, errorCapacity,
                    "required PixelUMM FFN projection handle is null");
    return nullptr;
  }
#if __has_include(<Foundation/Foundation.h>) && \
    __has_include(<Metal/Metal.h>) && \
    __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      @try {
        id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
        id<MTLBuffer> weight = (__bridge id<MTLBuffer>)weightHandle;
        id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
        const bool supportedTokens = tokens == 2048 || tokens == 21120;
        const bool supportedWidths =
            (inputWidth == 4096 && outputWidth == 12288) ||
            (inputWidth == 12288 && outputWidth == 4096);
        if (!device || !supportedTokens || !supportedWidths ||
            mode < PixelUMMFFNProjectionBF16 ||
            mode > PixelUMMFFNProjectionF16RoundedF32) {
          pixel_set_error(errorText, errorCapacity,
                          "unsupported PixelUMM FFN projection geometry/mode");
          return nullptr;
        }
        uint64_t inputElements = 0, weightElements = 0, outputElements = 0;
        uint64_t inputBytes = 0, weightBytes = 0, outputBytes = 0;
        if (!pixel_product((uint64_t)tokens, (uint64_t)inputWidth, &inputElements) ||
            !pixel_product((uint64_t)outputWidth, (uint64_t)inputWidth, &weightElements) ||
            !pixel_product((uint64_t)tokens, (uint64_t)outputWidth, &outputElements) ||
            !pixel_product(inputElements, 2u, &inputBytes) ||
            !pixel_product(weightElements, 2u, &weightBytes) ||
            !pixel_product(outputElements, 2u, &outputBytes) ||
            !pixel_buffer(input, device, inputBytes) ||
            !pixel_buffer(weight, device, weightBytes) ||
            !pixel_buffer(output, device, outputBytes) ||
            !pixel_disjoint(input, weight) || !pixel_disjoint(input, output) ||
            !pixel_disjoint(weight, output)) {
          pixel_set_error(errorText, errorCapacity,
              "PixelUMM FFN projection buffers have invalid device, storage, extent, or aliasing");
          return nullptr;
        }
        MPSDataType productType = MPSDataTypeInvalid;
        PixelUMMGraphStep *step = pixelumm_projection_graph_step(
            device, input, weight, output, (uint32_t)tokens,
            (uint32_t)inputWidth, (uint32_t)outputWidth,
            (PixelUMMFFNProjectionMode)mode, &productType,
            errorText, errorCapacity);
        if (!step) return nullptr;
        PixelUMMFFNProjectionPlan *plan = [PixelUMMFFNProjectionPlan new];
        plan.device = device;
        plan.step = step;
        plan.matmulOutputType = (uint32_t)productType;
        if (!plan.device || !plan.step) {
          pixel_set_error(errorText, errorCapacity,
                          "unable to retain PixelUMM FFN projection plan");
          return nullptr;
        }
        *matmulOutputType = plan.matmulOutputType;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"PixelUMM FFN projection preparation exception %@: %@",
             exception.name, exception.reason]);
        return nullptr;
      }
    }
  }
#endif
  pixel_set_error(errorText, errorCapacity,
                  "PixelUMM FFN projection requires macOS 14 or newer");
  return nullptr;
}

extern "C" int32_t gs_pixelumm_projection_encode(
    void *planHandle, void *commandBufferHandle,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle || !commandBufferHandle) {
    pixel_set_error(errorText, errorCapacity,
                    "required PixelUMM FFN projection plan/chain handle is null");
    return -1;
  }
#if __has_include(<Foundation/Foundation.h>) && \
    __has_include(<Metal/Metal.h>) && \
    __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      PixelUMMFFNProjectionPlan *plan = (__bridge PixelUMMFFNProjectionPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandBufferHandle;
      if (!plan.device || !plan.step ||
          plan.matmulOutputType == (uint32_t)MPSDataTypeInvalid) {
        pixel_set_error(errorText, errorCapacity,
                        "incomplete PixelUMM FFN projection plan");
        return -2;
      }
      id<MTLCommandBuffer> root = pixel_current_root(
          commandBuffer, plan.device, errorText, errorCapacity);
      if (!root) return -3;
      @try {
        return pixel_encode_graph(commandBuffer, plan.step,
                                  errorText, errorCapacity) ? 0 : -4;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"PixelUMM FFN projection encode exception %@: %@",
             exception.name, exception.reason]);
        return -5;
      }
    }
  }
#endif
  pixel_set_error(errorText, errorCapacity,
                  "PixelUMM FFN projection encode requires macOS 14 or newer");
  return -6;
}

extern "C" void gs_pixelumm_projection_release(void *planHandle) {
  if (!planHandle) return;
#if __has_include(<Foundation/Foundation.h>) && \
    __has_include(<Metal/Metal.h>) && \
    __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)
  @autoreleasepool {
    PixelUMMFFNProjectionPlan *plan =
        (__bridge_transfer PixelUMMFFNProjectionPlan *)planHandle;
    (void)plan;
  }
#endif
}

#endif
