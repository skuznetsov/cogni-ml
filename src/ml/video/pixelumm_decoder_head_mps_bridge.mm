// Standalone PixelUMM final head. The production decoder bridge is included
// here so its prepared RMS helpers and decoder C ABI share one translation
// unit; callers link this file instead of also linking the block bridge.
#include "pixelumm_decoder_block_mps_bridge.mm"

struct PixelUMMHeadRestoreParams {
  uint32_t tokens;
  uint32_t dim;
};

struct PixelUMMHeadUnpatchParams {
  uint32_t temporal;
  uint32_t height;
  uint32_t width;
  uint32_t outputTokens;
};

static PixelUMMGraphStep *pixel_head_rms_norm_step(
    id<MTLDevice> device, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> exponentBuffer, id<MTLBuffer> preGainBuffer,
    id<MTLBuffer> statisticsBuffer,
    uint32_t tokens, uint32_t dim, float epsilon, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *hiddenShape = pixel_shape({1, tokens, dim});
  NSArray *exponentShape = pixel_shape({1});
  NSArray *statisticsShape = pixel_shape({1, tokens, 1});
  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      inputBuffer, hiddenShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_input_bf16"]);
  MPSGraphTensor *exponent = pixel_add_feed(graph, feeds, placeholders, inputData,
      exponentBuffer, exponentShape, MPSDataTypeFloat32,
      [name stringByAppendingString:@"_power_exponent_f32"]);
  MPSGraphTensor *inputF32 = input
      ? [graph castTensor:input toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_input_f32"]] : nil;
  MPSGraphTensor *power = inputF32 && exponent
      ? [graph powerWithPrimaryTensor:inputF32 secondaryTensor:exponent
          name:[name stringByAppendingString:@"_pow_f32"]] : nil;
  MPSGraphTensor *mean = power
      ? [graph meanOfTensor:power axes:@[@2]
          name:[name stringByAppendingString:@"_mean_square_f32"]] : nil;
  // PyTorch 2.6's BF16 rms_norm composite converts epsilon to scalar_t
  // before adding it to the F32 reduction. Preserve that scalar rounding.
  MPSGraphTensor *epsilonBf16 = [graph constantWithScalar:epsilon
      shape:@[@1] dataType:MPSDataTypeBFloat16];
  MPSGraphTensor *epsilonF32 = epsilonBf16
      ? [graph castTensor:epsilonBf16 toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_epsilon_f32_from_bf16"]] : nil;
  MPSGraphTensor *plus = mean && epsilonF32
      ? [graph additionWithPrimaryTensor:mean secondaryTensor:epsilonF32
          name:[name stringByAppendingString:@"_epsilon_f32"]] : nil;
  MPSGraphTensor *plusRows = plus
      ? [graph reshapeTensor:plus withShape:statisticsShape
          name:[name stringByAppendingString:@"_statistics_rows_f32"]] : nil;
  MPSGraphTensor *inverse = plus
      ? [graph reciprocalSquareRootWithTensor:plus
          name:[name stringByAppendingString:@"_inverse_rms_f32"]] : nil;
  MPSGraphTensor *inverseRows = inverse
      ? [graph reshapeTensor:inverse withShape:statisticsShape
          name:[name stringByAppendingString:@"_inverse_rows_f32"]] : nil;
  MPSGraphTensor *normalized = inputF32 && inverseRows
      ? [graph multiplicationWithPrimaryTensor:inputF32 secondaryTensor:inverseRows
          name:[name stringByAppendingString:@"_normalized_f32"]] : nil;
  MPSGraphTensor *normalizedBf16 = pixel_cast_bf16(graph, normalized,
      [name stringByAppendingString:@"_normalized_bf16"]);
  if (!input || !exponent || !inputF32 || !power || !mean ||
      !epsilonBf16 || !epsilonF32 || !plus || !plusRows || !inverse ||
      !inverseRows || !normalized || !normalizedBf16 ||
      !preGainBuffer || !statisticsBuffer) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM head RMSNorm graph %@", name]);
    return nil;
  }
  NSArray<MPSGraphTensor *> *targets = @[normalizedBf16, plusRows];
  NSArray<id<MTLBuffer>> *outputs = @[preGainBuffer, statisticsBuffer];
  NSArray<NSArray<NSNumber *> *> *shapes = @[hiddenShape, statisticsShape];
  return pixel_compile_step_outputs(device, graph, feeds, placeholders, inputData,
      targets, outputs, shapes, name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_head_linear_step(
    id<MTLDevice> device, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> weightBuffer, id<MTLBuffer> biasBuffer,
    id<MTLBuffer> outputBuffer, uint32_t tokens, uint32_t inputWidth,
    uint32_t outputWidth, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *inputShape = pixel_shape({1, tokens, inputWidth});
  NSArray *weightShape = pixel_shape({outputWidth, inputWidth});
  NSArray *biasShape = pixel_shape({outputWidth});
  NSArray *outputShape = pixel_shape({1, tokens, outputWidth});
  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      inputBuffer, inputShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_input_bf16"]);
  MPSGraphTensor *weight = pixel_add_feed(graph, feeds, placeholders, inputData,
      weightBuffer, weightShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_weight_bf16"]);
  MPSGraphTensor *bias = pixel_add_feed(graph, feeds, placeholders, inputData,
      biasBuffer, biasShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_bias_bf16"]);
  MPSGraphTensor *inputF32 = input
      ? [graph castTensor:input toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_input_f32"]] : nil;
  MPSGraphTensor *weightF32 = weight
      ? [graph castTensor:weight toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_weight_f32"]] : nil;
  MPSGraphTensor *biasF32 = bias
      ? [graph castTensor:bias toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_bias_f32"]] : nil;
  MPSGraphTensor *flat = inputF32
      ? [graph reshapeTensor:inputF32 withShape:pixel_shape({tokens, inputWidth})
          name:[name stringByAppendingString:@"_flat_f32"]] : nil;
  MPSGraphTensor *transpose = weightF32
      ? [graph transposeTensor:weightF32 dimension:0 withDimension:1
          name:[name stringByAppendingString:@"_weight_transpose_f32"]] : nil;
  MPSGraphTensor *product = flat && transpose
      ? [graph matrixMultiplicationWithPrimaryTensor:flat secondaryTensor:transpose
          name:[name stringByAppendingString:@"_matmul_f32"]] : nil;
  MPSGraphTensor *biasRows = biasF32
      ? [graph reshapeTensor:biasF32 withShape:pixel_shape({1, outputWidth})
          name:[name stringByAppendingString:@"_bias_rows_f32"]] : nil;
  MPSGraphTensor *biased = product && biasRows
      ? [graph additionWithPrimaryTensor:product secondaryTensor:biasRows
          name:[name stringByAppendingString:@"_biased_f32"]] : nil;
  MPSGraphTensor *bf16 = pixel_cast_bf16(graph, biased,
      [name stringByAppendingString:@"_bf16"]);
  MPSGraphTensor *reshaped = bf16
      ? [graph reshapeTensor:bf16 withShape:outputShape
          name:[name stringByAppendingString:@"_output"]] : nil;
  if (!input || !weight || !bias || !inputF32 || !weightF32 || !biasF32 ||
      !flat || !transpose || !product || !biasRows || !biased || !bf16 || !reshaped) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM biased head linear graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      reshaped, outputBuffer, outputShape, name, errorText, errorCapacity);
}

static const char *pixel_head_metal_source = R"metal(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct PixelUMMHeadRestoreParams { uint tokens; uint dim; };
struct PixelUMMHeadUnpatchParams {
  uint temporal; uint height; uint width; uint outputTokens;
};

kernel void pixelumm_head_restore_thw(
    device const ushort *tileMajor [[buffer(0)]],
    device const int *originalToTile [[buffer(1)]],
    device ushort *thw [[buffer(2)]],
    constant PixelUMMHeadRestoreParams &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = p.tokens * p.dim;
  if (index >= count) return;
  const uint row = index / p.dim;
  const uint column = index - row * p.dim;
  const int sourceRow = originalToTile[row];
  if (sourceRow < 0 || uint(sourceRow) >= p.tokens) return;
  thw[index] = tileMajor[uint(sourceRow) * p.dim + column];
}

kernel void pixelumm_head_unpatchify(
    device const ushort *linear [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant PixelUMMHeadUnpatchParams &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= p.outputTokens) return;
  const uint outW = p.width * 16u;
  const uint outH = p.height * 16u;
  const uint outT = p.temporal * 4u;
  const uint channel = index / (outT * outH * outW);
  const uint pixel = index - channel * outT * outH * outW;
  const uint tOut = pixel / (outH * outW);
  const uint spatial = pixel - tOut * outH * outW;
  const uint hOut = spatial / outW;
  const uint wOut = spatial - hOut * outW;
  const uint t = tOut / 4u;
  const uint pt = tOut - t * 4u;
  const uint h = hOut / 16u;
  const uint ph = hOut - h * 16u;
  const uint w = wOut / 16u;
  const uint pw = wOut - w * 16u;
  const uint token = (t * p.height + h) * p.width + w;
  const uint feature = ((pt * 16u + ph) * 16u + pw) * 3u + channel;
  output[index] = linear[token * 3072u + feature];
}
)metal";

@interface PixelUMMDecoderHeadPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *borrowed;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *owned;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *restoreBuffers;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *unpatchBuffers;
@property(nonatomic, strong) PixelUMMGraphStep *finalNormPrepare;
@property(nonatomic, strong) PixelUMMGraphStep *finalNormGain;
@property(nonatomic, strong) PixelUMMGraphStep *headNormPrepare;
@property(nonatomic, strong) PixelUMMGraphStep *headNormGain;
@property(nonatomic, strong) PixelUMMGraphStep *linear;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> restore;
@property(nonatomic, strong) id<MTLComputePipelineState> unpatchify;
@property(nonatomic) uint32_t temporal;
@property(nonatomic) uint32_t height;
@property(nonatomic) uint32_t width;
@property(nonatomic) uint32_t tokens;
@property(nonatomic) uint32_t dim;
@property(nonatomic) uint32_t outputTokens;
@end
@implementation PixelUMMDecoderHeadPlan
@end

static bool pixel_head_validate_indices(id<MTLBuffer> indices, uint32_t tokens) {
  if (!indices || !indices.contents || indices.length != uint64_t(tokens) * 4u)
    return false;
  const int32_t *values = (const int32_t *)indices.contents;
  std::vector<uint8_t> seen(tokens, 0);
  for (uint32_t row = 0; row < tokens; ++row) {
    const int32_t source = values[row];
    if (source < 0 || uint32_t(source) >= tokens || seen[uint32_t(source)] != 0)
      return false;
    seen[uint32_t(source)] = 1;
  }
  return true;
}

extern "C" void *gs_pixelumm_decoder_head_plan_create(
    void *deviceHandle, int32_t temporal, int32_t height, int32_t width,
    int32_t tile, int32_t dim, void *inputHandle, void **weightHandles,
    void **ownedHandles, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!deviceHandle || !inputHandle || !weightHandles || !ownedHandles ||
      temporal <= 0 || height <= 0 || width <= 0 || tile != 8 || dim != 4096) {
    pixel_set_error(errorText, errorCapacity, "invalid PixelUMM head plan geometry or binding");
    return nullptr;
  }
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
      uint64_t tokens64 = uint64_t(temporal) * uint64_t(height) * uint64_t(width);
      if (!device || tokens64 == 0 || tokens64 > UINT32_MAX / uint64_t(dim) ||
          tokens64 * 3072u > UINT32_MAX ||
          uint64_t(temporal) * 4u > INT32_MAX ||
          uint64_t(height) * 16u > INT32_MAX ||
          uint64_t(width) * 16u > INT32_MAX) {
        pixel_set_error(errorText, errorCapacity, "PixelUMM head tensor extent overflows native dimensions");
        return nullptr;
      }
      const uint32_t tokens = uint32_t(tokens64);
      const uint32_t outputTokens = tokens * 3072u;
      const uint64_t hiddenBytes = tokens64 * uint64_t(dim) * 2u;
      const uint64_t outputBytes = tokens64 * 3072u * 2u;
      const uint64_t rowBytes = tokens64 * 4u;
      std::vector<id<MTLBuffer>> weights(4);
      const uint64_t weightBytes[4] = {8192u, 8192u, 25165824u, 6144u};
      for (NSUInteger index = 0; index < weights.size(); ++index) {
        weights[index] = (__bridge id<MTLBuffer>)weightHandles[index];
        if (!pixel_buffer(weights[index], device, weightBytes[index])) {
          pixel_set_error(errorText, errorCapacity, "PixelUMM head weight has wrong extent, device or storage");
          return nullptr;
        }
      }
      std::vector<id<MTLBuffer>> owned(11);
      const uint64_t ownedBytes[11] = {
        hiddenBytes, hiddenBytes, hiddenBytes, hiddenBytes, hiddenBytes,
        outputBytes, outputBytes, rowBytes, rowBytes, sizeof(float), rowBytes,
      };
      for (NSUInteger index = 0; index < owned.size(); ++index) {
        owned[index] = (__bridge id<MTLBuffer>)ownedHandles[index];
        if (!pixel_buffer(owned[index], device, ownedBytes[index])) {
          pixel_set_error(errorText, errorCapacity, "PixelUMM head owned buffer has wrong extent, device or storage");
          return nullptr;
        }
      }
      if (!pixel_buffer(input, device, hiddenBytes) ||
          !pixel_head_validate_indices(owned[10], tokens)) {
        pixel_set_error(errorText, errorCapacity, "PixelUMM head input or inverse tile order is invalid");
        return nullptr;
      }
      std::vector<id<MTLBuffer>> borrowed{input};
      borrowed.insert(borrowed.end(), weights.begin(), weights.end());
      for (NSUInteger i = 0; i < borrowed.size(); ++i)
        for (NSUInteger j = i + 1; j < borrowed.size(); ++j)
          if (!pixel_disjoint(borrowed[i], borrowed[j])) {
            pixel_set_error(errorText, errorCapacity, "borrowed PixelUMM head buffers alias");
            return nullptr;
          }
      for (NSUInteger i = 0; i < owned.size(); ++i) {
        for (NSUInteger j = 0; j < borrowed.size(); ++j)
          if (!pixel_disjoint(owned[i], borrowed[j])) {
            pixel_set_error(errorText, errorCapacity, "PixelUMM head output/scratch aliases a borrowed buffer");
            return nullptr;
          }
        for (NSUInteger j = i + 1; j < owned.size(); ++j)
          if (!pixel_disjoint(owned[i], owned[j])) {
            pixel_set_error(errorText, errorCapacity, "PixelUMM head output/scratch buffers alias");
            return nullptr;
          }
      }

      PixelUMMDecoderHeadPlan *plan = [PixelUMMDecoderHeadPlan new];
      plan.device = device;
      plan.borrowed = @[input, weights[0], weights[1], weights[2], weights[3]];
      plan.owned = @[owned[0], owned[1], owned[2], owned[3], owned[4],
                     owned[5], owned[6], owned[7], owned[8], owned[9], owned[10]];
      plan.restoreBuffers = @[owned[1], owned[10], owned[2]];
      plan.unpatchBuffers = @[owned[5], owned[6]];
      plan.temporal = uint32_t(temporal);
      plan.height = uint32_t(height);
      plan.width = uint32_t(width);
      plan.tokens = tokens;
      plan.dim = uint32_t(dim);
      plan.outputTokens = outputTokens;
      plan.finalNormPrepare = pixel_rms_prepare_step(device, input, owned[9],
          owned[0], owned[7], tokens, uint32_t(dim), 1e-6f,
          @"pixel_head_final_norm", errorText, errorCapacity);
      plan.finalNormGain = pixel_rms_gain_step(device, owned[0], weights[0],
          owned[1], tokens, uint32_t(dim), @"pixel_head_final_norm_gain",
          errorText, errorCapacity);
      plan.headNormPrepare = pixel_head_rms_norm_step(device, owned[2], owned[9],
          owned[3], owned[8], tokens, uint32_t(dim), 1e-6f,
          @"pixel_head_norm", errorText, errorCapacity);
      plan.headNormGain = pixel_rms_gain_step(device, owned[3], weights[1],
          owned[4], tokens, uint32_t(dim), @"pixel_head_norm_gain",
          errorText, errorCapacity);
      plan.linear = pixel_head_linear_step(device, owned[4], weights[2],
          weights[3], owned[5], tokens, uint32_t(dim), 3072u,
          @"pixel_head_linear", errorText, errorCapacity);
      if (!plan.finalNormPrepare || !plan.finalNormGain ||
          !plan.headNormPrepare || !plan.headNormGain || !plan.linear)
        return nullptr;

      NSError *error = nil;
      NSString *source = [NSString stringWithUTF8String:pixel_head_metal_source];
      MTLCompileOptions *options = [MTLCompileOptions new];
      plan.library = [device newLibraryWithSource:source options:options error:&error];
      if (!plan.library) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"unable to prepare PixelUMM head Metal library: %@",
             error.localizedDescription ?: @"unknown error"]);
        return nullptr;
      }
      plan.restore = pixel_pipeline(device, plan.library, @"pixelumm_head_restore_thw",
          errorText, errorCapacity);
      plan.unpatchify = pixel_pipeline(device, plan.library, @"pixelumm_head_unpatchify",
          errorText, errorCapacity);
      if (!plan.restore || !plan.unpatchify) return nullptr;
      return (__bridge_retained void *)plan;
    }
  }
  pixel_set_error(errorText, errorCapacity, "PixelUMM head MPS path requires macOS 14 or newer");
  return nullptr;
}

extern "C" int32_t gs_pixelumm_decoder_head_encode(
    void *planHandle, void *commandBufferHandle,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle || !commandBufferHandle) {
    pixel_set_error(errorText, errorCapacity, "required PixelUMM head plan/chain handle is null");
    return -1;
  }
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      PixelUMMDecoderHeadPlan *plan = (__bridge PixelUMMDecoderHeadPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandBufferHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!plan.device || !plan.borrowed || !plan.owned || !root ||
          !plan.restoreBuffers || !plan.unpatchBuffers ||
          root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted) {
        pixel_set_error(errorText, errorCapacity,
            "PixelUMM head MPSCommandChain root is unavailable, committed or on another device");
        return -2;
      }
      if (!pixel_encode_graph(commandBuffer, plan.finalNormPrepare, errorText, errorCapacity) ||
          !pixel_encode_graph(commandBuffer, plan.finalNormGain, errorText, errorCapacity))
        return -3;
      PixelUMMHeadRestoreParams restoreParams{plan.tokens, plan.dim};
      if (!pixel_encode_kernel(commandBuffer, plan.device, plan.restore,
          plan.restoreBuffers, &restoreParams,
          sizeof(restoreParams), uint64_t(plan.tokens) * plan.dim, false,
          errorText, errorCapacity)) return -4;
      if (!pixel_encode_graph(commandBuffer, plan.headNormPrepare, errorText, errorCapacity) ||
          !pixel_encode_graph(commandBuffer, plan.headNormGain, errorText, errorCapacity) ||
          !pixel_encode_graph(commandBuffer, plan.linear, errorText, errorCapacity))
        return -5;
      PixelUMMHeadUnpatchParams unpatchParams{
        plan.temporal, plan.height, plan.width, plan.outputTokens,
      };
      if (!pixel_encode_kernel(commandBuffer, plan.device, plan.unpatchify,
          plan.unpatchBuffers, &unpatchParams, sizeof(unpatchParams),
          plan.outputTokens, false, errorText, errorCapacity)) return -6;
      return 0;
    }
  }
  pixel_set_error(errorText, errorCapacity, "PixelUMM head MPS encode requires macOS 14 or newer");
  return -7;
}

extern "C" void gs_pixelumm_decoder_head_plan_release(void *planHandle) {
  if (!planHandle) return;
  @autoreleasepool {
    PixelUMMDecoderHeadPlan *plan = (__bridge_transfer PixelUMMDecoderHeadPlan *)planHandle;
    (void)plan;
  }
}
