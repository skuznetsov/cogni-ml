// Prepared BF16 MPSGraph Conv3d plans for the pinned DCAE-V MPS path.
// Plans bind only caller-owned resident buffers; graph-private allocations
// remain owned by MPSGraph and are not visible to MetalBuffer accounting.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>

@interface CogniDCAEVConvTemplate : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphTensor *source;
@property(nonatomic, strong) MPSGraphTensor *cache;
@property(nonatomic, strong) MPSGraphTensor *weight;
@property(nonatomic, strong) MPSGraphTensor *bias;
@property(nonatomic, strong) MPSGraphTensor *result;
@property(nonatomic, strong) NSArray<NSNumber *> *sourceShape;
@property(nonatomic, strong) NSArray<NSNumber *> *cacheShape;
@property(nonatomic, strong) NSArray<NSNumber *> *weightShape;
@property(nonatomic, strong) NSArray<NSNumber *> *biasShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@property(nonatomic) int32_t batches;
@property(nonatomic) int32_t inChannels;
@property(nonatomic) int32_t convOutChannels;
@property(nonatomic) int32_t outputChannels;
@property(nonatomic) int32_t frames;
@property(nonatomic) int32_t sourceFrames;
@property(nonatomic) int32_t height;
@property(nonatomic) int32_t width;
@property(nonatomic) int32_t spatialFactor;
@property(nonatomic) int32_t temporalFactor;
@property(nonatomic, strong) NSArray<MPSGraphTensor *> *feedTensors;
@property(nonatomic, strong) NSArray<NSArray<NSNumber *> *> *feedShapes;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic) BOOL fullClip;
@property(nonatomic) BOOL cacheInput;
@property(nonatomic) BOOL useBias;
@property(nonatomic) uint64_t sourceBytes;
@property(nonatomic) uint64_t cacheBytes;
@property(nonatomic) uint64_t weightBytes;
@property(nonatomic) uint64_t biasBytes;
@property(nonatomic) uint64_t outputBytes;
@end

@implementation CogniDCAEVConvTemplate
@end

@interface CogniDCAEVConvPlan : NSObject
@property(nonatomic, strong) CogniDCAEVConvTemplate *templatePlan;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputs;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputs;
@end

@implementation CogniDCAEVConvPlan
@end

static bool dcae_v_mps_mul(uint64_t *value, uint64_t factor) {
  if (factor != 0 && *value > UINT64_MAX / factor)
    return false;
  *value *= factor;
  return true;
}

static uint64_t dcae_v_mps_elements(int32_t batches, int32_t channels,
                                   int32_t frames, int32_t height,
                                   int32_t width) {
  if (batches <= 0 || channels <= 0 || frames <= 0 || height <= 0 || width <= 0)
    return 0;
  uint64_t value = (uint64_t)batches;
  if (!dcae_v_mps_mul(&value, (uint64_t)channels) ||
      !dcae_v_mps_mul(&value, (uint64_t)frames) ||
      !dcae_v_mps_mul(&value, (uint64_t)height) ||
      !dcae_v_mps_mul(&value, (uint64_t)width) ||
      value > UINT64_MAX / 2)
    return 0;
  return value;
}

static bool dcae_v_mps_disjoint(id<MTLBuffer> a, id<MTLBuffer> b) {
  if (!a || !b || a == b || !a.contents || !b.contents)
    return false;
  const uintptr_t aStart = (uintptr_t)a.contents;
  const uintptr_t bStart = (uintptr_t)b.contents;
  if (a.length > UINTPTR_MAX - aStart || b.length > UINTPTR_MAX - bStart)
    return false;
  const uintptr_t aEnd = aStart + (uintptr_t)a.length;
  const uintptr_t bEnd = bStart + (uintptr_t)b.length;
  return aEnd <= bStart || bEnd <= aStart;
}

// Create and compile one reusable graph geometry. `fullClip` means source is
// the complete input clip and this graph contains a static temporal slice.
// Chunk-causal graphs explicitly concatenate left/cache and right/zero frames,
// matching upstream ConvLayer3d before its zero-temporal-padding convolution.
extern "C" void *gs_dcae_v_mps_conv_template_create(
    void *deviceHandle, int32_t batches, int32_t inChannels,
    int32_t convOutChannels, int32_t outputChannels,
    int32_t frames, int32_t sourceFrames,
    int32_t height, int32_t width, int32_t inputFrameOffset,
    int32_t fullClip, int32_t chunkCausal, int32_t cacheInput,
    int32_t useBias, int32_t spatialFactor, int32_t temporalFactor,
    int32_t shortcut) {
  if (!deviceHandle || batches <= 0 || inChannels <= 0 || convOutChannels <= 0 ||
      outputChannels <= 0 || spatialFactor <= 0 || temporalFactor <= 0 ||
      frames <= 0 || sourceFrames < frames || height <= 0 || width <= 0 ||
      inputFrameOffset < 0 || inputFrameOffset > sourceFrames - frames ||
      (fullClip != 0 && fullClip != 1) ||
      (chunkCausal != 0 && chunkCausal != 1) ||
      (cacheInput != 0 && cacheInput != 1) ||
      (useBias != 0 && useBias != 1) || (shortcut != 0 && shortcut != 1) ||
      (cacheInput && !chunkCausal) || height % spatialFactor != 0 ||
      width % spatialFactor != 0 || frames % temporalFactor != 0 ||
      convOutChannels > INT32_MAX / spatialFactor ||
      convOutChannels * spatialFactor > INT32_MAX / spatialFactor ||
      convOutChannels * spatialFactor * spatialFactor > INT32_MAX / temporalFactor ||
      convOutChannels * spatialFactor * spatialFactor * temporalFactor != outputChannels ||
      (shortcut && ((int64_t)inChannels * spatialFactor * spatialFactor * temporalFactor) < outputChannels) ||
      (shortcut && ((int64_t)inChannels * spatialFactor * spatialFactor * temporalFactor) % outputChannels != 0))
    {
      fprintf(stderr, "DCAE-V MPSGraph Conv3d invalid template geometry B=%d Cin=%d Cconv=%d Cout=%d T=%d/%d HW=%dx%d offset=%d full=%d causal=%d cache=%d bias=%d unshuffle=%dx%d shortcut=%d\n",
              batches, inChannels, convOutChannels, outputChannels, frames,
              sourceFrames, height, width, inputFrameOffset, fullClip,
              chunkCausal, cacheInput, useBias, temporalFactor, spatialFactor,
              shortcut);
      return nullptr;
    }

  if (@available(macOS 13.2, *)) {
    @autoreleasepool {
      const int32_t boundFrames = fullClip ? sourceFrames : frames;
      const uint64_t sourceElements = dcae_v_mps_elements(
          batches, inChannels, boundFrames, height, width);
      const uint64_t cacheElements = dcae_v_mps_elements(
          batches, inChannels, 1, height, width);
      const uint64_t weightElements = dcae_v_mps_elements(
          convOutChannels, inChannels, 3, 3, 3);
      const uint64_t outputElements = dcae_v_mps_elements(
          batches, outputChannels, frames / temporalFactor,
          height / spatialFactor, width / spatialFactor);
      if (!sourceElements || !cacheElements || !weightElements || !outputElements) {
        fprintf(stderr, "DCAE-V MPSGraph Conv3d element overflow B=%d Cin=%d Cconv=%d Cout=%d T=%d/%d HW=%dx%d\n",
                batches, inChannels, convOutChannels, outputChannels, frames,
                sourceFrames, height, width);
        return nullptr;
      }
      if (useBias && (uint64_t)convOutChannels > UINT64_MAX / 2) {
        fprintf(stderr, "DCAE-V MPSGraph Conv3d bias extent overflow Cconv=%d\n", convOutChannels);
        return nullptr;
      }

      @try {
        id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
        CogniDCAEVConvTemplate *plan = [CogniDCAEVConvTemplate new];
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        NSArray<NSNumber *> *sourceShape = @[
          @(batches), @(inChannels), @(boundFrames), @(height), @(width)
        ];
        NSArray<NSNumber *> *cacheShape = @[
          @(batches), @(inChannels), @1, @(height), @(width)
        ];
        NSArray<NSNumber *> *weightShape = @[
          @(convOutChannels), @(inChannels), @3, @3, @3
        ];
        NSArray<NSNumber *> *biasShape = @[@1, @(convOutChannels), @1, @1, @1];
        NSArray<NSNumber *> *outputShape = @[
          @(batches), @(outputChannels), @(frames / temporalFactor),
          @(height / spatialFactor), @(width / spatialFactor)
        ];

        MPSGraphTensor *source = [graph placeholderWithShape:sourceShape
                                                     dataType:MPSDataTypeBFloat16
                                                         name:@"dcae_source"];
        MPSGraphTensor *weight = [graph placeholderWithShape:weightShape
                                                     dataType:MPSDataTypeBFloat16
                                                         name:@"dcae_weight"];
        MPSGraphTensor *bias = useBias
            ? [graph placeholderWithShape:biasShape
                                  dataType:MPSDataTypeBFloat16
                                      name:@"dcae_bias"]
            : nil;
        MPSGraphTensor *tile = source;
        if (fullClip) {
          tile = [graph sliceTensor:source dimension:2
                              start:inputFrameOffset length:frames
                               name:@"dcae_tile_slice"];
        }
        if (!tile) {
          fprintf(stderr, "DCAE-V MPSGraph Conv3d input slice failed offset=%d frames=%d sourceFrames=%d\n",
                  inputFrameOffset, frames, sourceFrames);
          return nullptr;
        }

        MPSGraphTensor *cache = nil;
        NSMutableArray<MPSGraphTensor *> *feedTensors = [NSMutableArray array];
        NSMutableArray<NSArray<NSNumber *> *> *feedShapes = [NSMutableArray array];
        [feedTensors addObject:source];
        [feedShapes addObject:sourceShape];
        if (cacheInput) {
          cache = [graph placeholderWithShape:cacheShape
                                      dataType:MPSDataTypeBFloat16
                                          name:@"dcae_cache"];
          [feedTensors addObject:cache];
          [feedShapes addObject:cacheShape];
        }
        [feedTensors addObject:weight];
        [feedShapes addObject:weightShape];
        if (bias) {
          [feedTensors addObject:bias];
          [feedShapes addObject:biasShape];
        }

        MPSGraphTensor *convInput = tile;
        if (chunkCausal) {
          MPSGraphTensor *zero = [graph constantWithScalar:0.0
                                                      shape:cacheShape
                                                   dataType:MPSDataTypeBFloat16];
          MPSGraphTensor *left = cacheInput ? cache : zero;
          convInput = [graph concatTensors:@[left, tile, zero]
                                 dimension:2
                                      name:@"dcae_chunk_causal_pad"];
        }
        if (!convInput) {
          fprintf(stderr, "DCAE-V MPSGraph Conv3d causal concat failed frames=%d cacheInput=%d\n",
                  frames, cacheInput);
          return nullptr;
        }

        const NSUInteger temporalPad = chunkCausal ? 0 : 1;
        MPSGraphConvolution3DOpDescriptor *descriptor =
            [MPSGraphConvolution3DOpDescriptor
                descriptorWithStrideInX:1 strideInY:1 strideInZ:1
                dilationRateInX:1 dilationRateInY:1 dilationRateInZ:1
                groups:1 paddingLeft:1 paddingRight:1
                paddingTop:1 paddingBottom:1
                paddingFront:temporalPad paddingBack:temporalPad
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNCDHW
                weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
        MPSGraphTensor *convolution = descriptor
            ? [graph convolution3DWithSourceTensor:convInput
                                     weightsTensor:weight
                                       descriptor:descriptor
                                             name:@"dcae_conv3d"]
            : nil;
        // PyTorch MPS lowers Conv3d bias as a separate BF16 add graph op.
        MPSGraphTensor *convResult = bias
            ? [graph additionWithPrimaryTensor:convolution
                              secondaryTensor:bias name:@"dcae_conv_bias"]
            : convolution;
        MPSGraphTensor *result = convResult;
        if (spatialFactor != 1 || temporalFactor != 1) {
          const int32_t outputFrames = frames / temporalFactor;
          const int32_t outputHeight = height / spatialFactor;
          const int32_t outputWidth = width / spatialFactor;
          NSArray<NSNumber *> *convUnshuffleShape = @[
            @(batches), @(convOutChannels), @(outputFrames), @(temporalFactor),
            @(outputHeight), @(spatialFactor), @(outputWidth), @(spatialFactor)
          ];
          MPSGraphTensor *convReshaped = convResult
              ? [graph reshapeTensor:convResult withShape:convUnshuffleShape name:nil]
              : nil;
          MPSGraphTensor *convPermuted = convReshaped
              ? [graph transposeTensor:convReshaped
                           permutation:@[@0,@1,@3,@5,@7,@2,@4,@6]
                                   name:nil]
              : nil;
          NSArray<NSNumber *> *unshuffledShape = @[
            @(batches), @(outputChannels), @(outputFrames), @(outputHeight), @(outputWidth)
          ];
          result = convPermuted
              ? [graph reshapeTensor:convPermuted withShape:unshuffledShape
                                name:@"dcae_conv_pixel_unshuffle"]
              : nil;

        }

        if (shortcut && result) {
            const int32_t outputFrames = frames / temporalFactor;
            const int32_t outputHeight = height / spatialFactor;
            const int32_t outputWidth = width / spatialFactor;
            MPSGraphTensor *shortcutInput = tile;
            if (spatialFactor != 1 || temporalFactor != 1) {
            NSArray<NSNumber *> *shortcutReshapeShape = @[
              @(batches), @(inChannels), @(outputFrames), @(temporalFactor),
              @(outputHeight), @(spatialFactor), @(outputWidth), @(spatialFactor)
            ];
            MPSGraphTensor *shortcutReshaped =
                [graph reshapeTensor:tile withShape:shortcutReshapeShape name:nil];
            MPSGraphTensor *shortcutPermuted = shortcutReshaped
                ? [graph transposeTensor:shortcutReshaped
                             permutation:@[@0,@1,@3,@5,@7,@2,@4,@6]
                                     name:nil]
                : nil;
            const int64_t factor = (int64_t)spatialFactor * spatialFactor * temporalFactor;
            const int64_t shortcutChannels = (int64_t)inChannels * factor;
            NSArray<NSNumber *> *shortcutShape = @[
              @(batches), @(shortcutChannels), @(outputFrames), @(outputHeight), @(outputWidth)
            ];
            shortcutInput = shortcutPermuted
                ? [graph reshapeTensor:shortcutPermuted withShape:shortcutShape name:nil]
                : nil;
            }
            const int64_t factor = (int64_t)spatialFactor * spatialFactor * temporalFactor;
            const int64_t shortcutChannels = (int64_t)inChannels * factor;
            const int64_t group = shortcutChannels / outputChannels;
            NSArray<NSNumber *> *groupShape = @[
              @(batches), @(outputChannels), @(group), @(outputFrames),
              @(outputHeight), @(outputWidth)
            ];
            NSArray<NSNumber *> *shortcutOutputShape = @[
              @(batches), @(outputChannels), @(outputFrames), @(outputHeight), @(outputWidth)
            ];
            MPSGraphTensor *shortcutGrouped = shortcutInput
                ? [graph reshapeTensor:shortcutInput withShape:groupShape name:nil]
                : nil;
            MPSGraphTensor *shortcutFloat = shortcutGrouped
                ? [graph castTensor:shortcutGrouped toType:MPSDataTypeFloat32 name:nil]
                : nil;
            MPSGraphTensor *shortcutMeanFloat = shortcutFloat
                ? [graph meanOfTensor:shortcutFloat axes:@[@2]
                                name:@"dcae_shortcut_channel_mean_f32"]
                : nil;
            MPSGraphTensor *shortcutMeanBFloat16 = shortcutMeanFloat
                ? [graph castTensor:shortcutMeanFloat toType:MPSDataTypeBFloat16 name:nil]
                : nil;
            // PyTorch mean(..., keepdim=False) removes the group axis. The
            // reshape fixes the expected NCDHW rank before residual addition.
            MPSGraphTensor *shortcutMean = shortcutMeanBFloat16
                ? [graph reshapeTensor:shortcutMeanBFloat16
                              withShape:shortcutOutputShape
                                    name:@"dcae_shortcut_channel_mean"]
                : nil;
            result = shortcutMean
                ? [graph additionWithPrimaryTensor:result
                                   secondaryTensor:shortcutMean
                                              name:@"dcae_downsample_residual_add"]
                : nil;
        }
        if (!graph || !source || !weight || (useBias && !bias) || !convResult || !result ||
            result.shape == nil || ![result.shape isEqualToArray:outputShape]) {
          fprintf(stderr, "DCAE-V MPSGraph Conv3d output geometry mismatch expected=%s actual=%s full=%d causal=%d factors=%dx%d shortcut=%d\n",
                  outputShape.description.UTF8String ?: "nil",
                  result.shape.description.UTF8String ?: "nil", fullClip,
                  chunkCausal, temporalFactor, spatialFactor, shortcut);
          return nullptr;
        }

        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
            [NSMutableDictionary dictionary];
        for (NSUInteger feedIndex = 0; feedIndex < feedTensors.count; ++feedIndex) {
          MPSGraphTensor *tensor = feedTensors[feedIndex];
          NSArray<NSNumber *> *shape = tensor == source ? sourceShape
              : tensor == cache ? cacheShape
              : tensor == weight ? weightShape : biasShape;
          MPSGraphShapedType *type = [[MPSGraphShapedType alloc]
              initWithShape:shape dataType:MPSDataTypeBFloat16];
          if (!type) {
            fprintf(stderr, "DCAE-V MPSGraph Conv3d feed type allocation failed feed=%lu\n",
                    (unsigned long)feedIndex);
            return nullptr;
          }
          feeds[tensor] = type;
        }
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds targetTensors:@[result]
                     targetOperations:nil compilationDescriptor:nil];
        if (!executable) {
          fprintf(stderr, "DCAE-V MPSGraph Conv3d compile returned nil B=%d Cin=%d Cconv=%d Cout=%d T=%d/%d HW=%dx%d offset=%d full=%d causal=%d cache=%d bias=%d unshuffle=%dx%d shortcut=%d\n",
                  batches, inChannels, convOutChannels, outputChannels, frames,
                  sourceFrames, height, width, inputFrameOffset, fullClip,
                  chunkCausal, cacheInput, useBias, temporalFactor,
                  spatialFactor, shortcut);
          return nullptr;
        }
        if (executable.feedTensors.count != feedTensors.count ||
            feedShapes.count != feedTensors.count) {
          fprintf(stderr, "DCAE-V MPSGraph Conv3d feed count mismatch expected=%lu actual=%lu\n",
                  (unsigned long)feedTensors.count,
                  (unsigned long)executable.feedTensors.count);
          return nullptr;
        }
        NSMutableIndexSet *seenFeeds = [NSMutableIndexSet indexSet];
        for (MPSGraphTensor *feedTensor in executable.feedTensors) {
          NSUInteger feedIndex = [feedTensors indexOfObjectIdenticalTo:feedTensor];
          if (feedIndex == NSNotFound || [seenFeeds containsIndex:feedIndex] ||
              feedTensor.dataType != MPSDataTypeBFloat16 || !feedTensor.shape ||
              ![feedTensor.shape isEqualToArray:feedShapes[feedIndex]]) {
            fprintf(stderr, "DCAE-V MPSGraph Conv3d executable feed identity/shape mismatch index=%lu\n",
                    (unsigned long)feedIndex);
            return nullptr;
          }
          [seenFeeds addIndex:feedIndex];
        }
        if (seenFeeds.count != feedTensors.count) {
          fprintf(stderr, "DCAE-V MPSGraph Conv3d executable feeds do not cover all placeholders\n");
          return nullptr;
        }
        executable.options = MPSGraphOptionsNone;
        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;

        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.source = source;
        plan.cache = cache;
        plan.weight = weight;
        plan.bias = bias;
        plan.result = result;
        plan.sourceShape = sourceShape;
        plan.cacheShape = cacheShape;
        plan.weightShape = weightShape;
        plan.biasShape = biasShape;
        plan.outputShape = outputShape;
        plan.feedTensors = [feedTensors copy];
        plan.feedShapes = [feedShapes copy];
        plan.execution = execution;
        plan.fullClip = fullClip != 0;
        plan.cacheInput = cacheInput != 0;
        plan.useBias = useBias != 0;
        plan.sourceBytes = sourceElements * 2;
        plan.cacheBytes = cacheElements * 2;
        plan.weightBytes = weightElements * 2;
        plan.biasBytes = useBias ? (uint64_t)convOutChannels * 2 : 0;
        plan.outputBytes = outputElements * 2;
        plan.batches = batches;
        plan.inChannels = inChannels;
        plan.convOutChannels = convOutChannels;
        plan.outputChannels = outputChannels;
        plan.frames = frames;
        plan.sourceFrames = sourceFrames;
        plan.height = height;
        plan.width = width;
        plan.spatialFactor = spatialFactor;
        plan.temporalFactor = temporalFactor;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        fprintf(stderr, "DCAE-V MPSGraph Conv3d setup exception: %s\n",
                exception.reason.UTF8String ?: "(no reason)");
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" void *gs_dcae_v_mps_conv_plan_create(
    void *templateHandle, void *inputHandle, void *cacheHandle,
    void *weightHandle, void *biasHandle, void *outputHandle) {
  if (!templateHandle || !inputHandle || !weightHandle || !outputHandle)
    return nullptr;
  @autoreleasepool {
    @try {
      CogniDCAEVConvTemplate *templatePlan =
          (__bridge CogniDCAEVConvTemplate *)templateHandle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
      id<MTLBuffer> cache = cacheHandle ? (__bridge id<MTLBuffer>)cacheHandle : nil;
      id<MTLBuffer> weight = (__bridge id<MTLBuffer>)weightHandle;
      id<MTLBuffer> bias = biasHandle ? (__bridge id<MTLBuffer>)biasHandle : nil;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
      if (!templatePlan || input.device != templatePlan.device ||
          weight.device != templatePlan.device || output.device != templatePlan.device ||
          input.length < templatePlan.sourceBytes ||
          weight.length < templatePlan.weightBytes ||
          output.length < templatePlan.outputBytes ||
          (templatePlan.cacheInput && (!cache || cache.device != templatePlan.device ||
                                       cache.length < templatePlan.cacheBytes)) ||
          (!templatePlan.cacheInput && cache) ||
          (templatePlan.useBias && (!bias || bias.device != templatePlan.device ||
                                    bias.length < templatePlan.biasBytes)) ||
          (!templatePlan.useBias && bias) ||
          !dcae_v_mps_disjoint(input, output) ||
          !dcae_v_mps_disjoint(weight, output) ||
          (cache && (!dcae_v_mps_disjoint(cache, output) ||
                     !dcae_v_mps_disjoint(cache, input) ||
                     !dcae_v_mps_disjoint(cache, weight))) ||
          !dcae_v_mps_disjoint(weight, input) ||
          (bias && (!dcae_v_mps_disjoint(bias, input) ||
                    !dcae_v_mps_disjoint(bias, output) ||
                    !dcae_v_mps_disjoint(bias, weight) ||
                    (cache && !dcae_v_mps_disjoint(bias, cache)))))
        return nullptr;

      MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:input shape:templatePlan.sourceShape
                   dataType:MPSDataTypeBFloat16];
      MPSGraphTensorData *weightData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:weight shape:templatePlan.weightShape
                   dataType:MPSDataTypeBFloat16];
      MPSGraphTensorData *outputData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:output shape:templatePlan.outputShape
                   dataType:MPSDataTypeBFloat16];
      if (!inputData || !weightData || !outputData ||
          templatePlan.feedTensors.count != templatePlan.feedShapes.count ||
          templatePlan.feedTensors.count != templatePlan.executable.feedTensors.count)
        return nullptr;

      NSMutableArray<MPSGraphTensorData *> *logicalInputs = [NSMutableArray array];
      [logicalInputs addObject:inputData];
      if (templatePlan.cacheInput) {
        MPSGraphTensorData *cacheData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:cache shape:templatePlan.cacheShape
                     dataType:MPSDataTypeBFloat16];
        if (!cacheData)
          return nullptr;
        [logicalInputs addObject:cacheData];
      }
      [logicalInputs addObject:weightData];
      if (templatePlan.useBias) {
        MPSGraphTensorData *biasData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bias shape:templatePlan.biasShape
                     dataType:MPSDataTypeBFloat16];
        if (!biasData)
          return nullptr;
        [logicalInputs addObject:biasData];
      }
      if (logicalInputs.count != templatePlan.feedTensors.count)
        return nullptr;

      // Executable feed order is independent of placeholder insertion order.
      // Reorder the fixed TensorData bindings by placeholder identity here,
      // during setup, and reject missing, duplicate, or mismatched feeds.
      NSMutableArray<MPSGraphTensorData *> *orderedInputs = [NSMutableArray array];
      NSMutableIndexSet *seenFeeds = [NSMutableIndexSet indexSet];
      for (MPSGraphTensor *feedTensor in templatePlan.executable.feedTensors) {
        NSUInteger feedIndex =
            [templatePlan.feedTensors indexOfObjectIdenticalTo:feedTensor];
        if (feedIndex == NSNotFound || [seenFeeds containsIndex:feedIndex] ||
            feedTensor.dataType != MPSDataTypeBFloat16 || !feedTensor.shape ||
            ![feedTensor.shape isEqualToArray:templatePlan.feedShapes[feedIndex]])
          return nullptr;
        [seenFeeds addIndex:feedIndex];
        [orderedInputs addObject:logicalInputs[feedIndex]];
      }
      if (orderedInputs.count != templatePlan.executable.feedTensors.count ||
          seenFeeds.count != templatePlan.feedTensors.count)
        return nullptr;

      CogniDCAEVConvPlan *plan = [CogniDCAEVConvPlan new];
      plan.templatePlan = templatePlan;
      plan.inputs = [orderedInputs copy];
      plan.outputs = @[outputData];
      return (__bridge_retained void *)plan;
    } @catch (NSException *exception) {
      (void)exception;
      return nullptr;
    }
  }
}

extern "C" int32_t gs_dcae_v_mps_conv_encode(void *planHandle,
                                              void *commandHandle) {
  if (!planHandle || !commandHandle)
    return -1;
  @autoreleasepool {
    @try {
      CogniDCAEVConvPlan *plan = (__bridge CogniDCAEVConvPlan *)planHandle;
      MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)commandHandle;
      if (!plan || !plan.templatePlan.executable || !command)
        return -2;
      NSArray<MPSGraphTensorData *> *result =
          [plan.templatePlan.executable encodeToCommandBuffer:command
                                                  inputsArray:plan.inputs
                                                 resultsArray:plan.outputs
                                          executionDescriptor:plan.templatePlan.execution];
      return result.count == 1 ? 0 : -3;
    } @catch (NSException *exception) {
      (void)exception;
      return -4;
    }
  }
}

extern "C" void gs_dcae_v_mps_conv_plan_release(void *planHandle) {
  if (planHandle)
    CFBridgingRelease(planHandle);
}

extern "C" void gs_dcae_v_mps_conv_template_release(void *templateHandle) {
  if (templateHandle)
    CFBridgingRelease(templateHandle);
}
