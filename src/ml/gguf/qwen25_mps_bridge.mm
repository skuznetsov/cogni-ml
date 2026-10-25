// Prepared, caller-buffer-bound Qwen2.5-VL text decoder block. The caller owns
// every activation, weight, stage, and scratch MTLBuffer plus the
// MPSCommandChain. Prepared graph executables are encoded asynchronously in
// sequence so BF16 source-operation boundaries round through caller buffers.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

static const NSUInteger kGsQwen25WeightCount = 12;
static const NSUInteger kGsQwen25StageCount = 17;
static const NSUInteger kGsQwen25ScratchCount = 7;
static const uint64_t kGsQwen25BFloat16Bytes = 2;

@interface GsQwen25MPSGraphBuilder : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds;
@property(nonatomic, strong) NSMutableArray<MPSGraphTensor *> *placeholders;
@property(nonatomic, strong) NSMutableArray<MPSGraphTensorData *> *inputData;
@end

@implementation GsQwen25MPSGraphBuilder
@end

@interface GsQwen25MPSGraphStep : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputs;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputs;
@property(nonatomic, copy) NSString *name;
@property(nonatomic) NSUInteger outputCount;
@end

@implementation GsQwen25MPSGraphStep
@end

@interface GsQwen25MPSBlockPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<GsQwen25MPSGraphStep *> *steps;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@end

@implementation GsQwen25MPSBlockPlan
@end

static void gs_qwen25_set_error(char *destination, size_t capacity,
                                const char *message) {
  if (!destination || capacity == 0)
    return;
  if (!message)
    message = "Qwen2.5-VL MPS bridge failed";
  (void)std::snprintf(destination, capacity, "%s", message);
}

static void gs_qwen25_set_ns_error(char *destination, size_t capacity,
                                   NSString *message) {
  if (!destination || capacity == 0)
    return;
  const char *utf8 = message ? message.UTF8String : nullptr;
  gs_qwen25_set_error(destination, capacity, utf8);
}

static bool gs_qwen25_product(const uint64_t *dimensions, size_t count,
                              uint64_t *productOut) {
  if (!dimensions || !productOut)
    return false;
  uint64_t product = 1;
  for (size_t index = 0; index < count; ++index) {
    if (dimensions[index] == 0 ||
        product > (uint64_t)INT32_MAX / dimensions[index])
      return false;
    product *= dimensions[index];
  }
  *productOut = product;
  return true;
}

static bool gs_qwen25_disjoint(id<MTLBuffer> left,
                               id<MTLBuffer> right) {
  if (!left || !right || left == right || !left.contents || !right.contents ||
      left.length == 0 || right.length == 0)
    return false;
  const uintptr_t leftStart = (uintptr_t)left.contents;
  const uintptr_t rightStart = (uintptr_t)right.contents;
  if (left.length > UINTPTR_MAX - leftStart ||
      right.length > UINTPTR_MAX - rightStart)
    return false;
  const uintptr_t leftEnd = leftStart + (uintptr_t)left.length;
  const uintptr_t rightEnd = rightStart + (uintptr_t)right.length;
  return leftEnd <= rightStart || rightEnd <= leftStart;
}

static MPSGraphTensorData *gs_qwen25_tensor_data(
    id<MTLBuffer> buffer, NSArray<NSNumber *> *shape, MPSDataType dataType) {
  if (!buffer || !shape)
    return nil;
  return [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                shape:shape
                                             dataType:dataType];
}

static MPSGraphTensor *gs_qwen25_add_feed(
    MPSGraph *graph, NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds,
    NSMutableArray<MPSGraphTensor *> *placeholders,
    NSMutableArray<MPSGraphTensorData *> *inputData, id<MTLBuffer> buffer,
    NSArray<NSNumber *> *shape, MPSDataType dataType, NSString *name) {
  if (!graph || !feeds || !placeholders || !inputData || !buffer || !shape ||
      !name)
    return nil;
  MPSGraphTensor *tensor = [graph placeholderWithShape:shape
                                               dataType:dataType
                                                   name:name];
  MPSGraphShapedType *feedType =
      [[MPSGraphShapedType alloc] initWithShape:shape dataType:dataType];
  MPSGraphTensorData *data =
      gs_qwen25_tensor_data(buffer, shape, dataType);
  if (!tensor || !feedType || !data)
    return nil;
  feeds[tensor] = feedType;
  [placeholders addObject:tensor];
  [inputData addObject:data];
  return tensor;
}

static MPSGraphTensor *gs_qwen25_cast_bf16(MPSGraph *graph,
                                           MPSGraphTensor *tensor,
                                           NSString *name) {
  if (!graph || !tensor || !name)
    return nil;
  return [graph castTensor:tensor toType:MPSDataTypeBFloat16 name:name];
}

static MPSGraphTensor *gs_qwen25_linear(
    MPSGraph *graph, MPSGraphTensor *input, MPSGraphTensor *weight,
    MPSGraphTensor *bias, uint32_t batches, uint32_t tokens,
    uint32_t inputWidth, uint32_t outputWidth, NSString *name) {
  if (!graph || !input || !weight || inputWidth == 0 || outputWidth == 0 ||
      batches == 0 || tokens == 0 || !name)
    return nil;
  const uint64_t rows64 = (uint64_t)batches * (uint64_t)tokens;
  if (rows64 == 0 || rows64 > (uint64_t)INT32_MAX)
    return nil;
  const NSUInteger rows = (NSUInteger)rows64;
  MPSGraphTensor *flat = [graph reshapeTensor:input
                                    withShape:@[ @(rows), @(inputWidth) ]
                                         name:[name stringByAppendingString:@"_flat"]];
  MPSGraphTensor *weightTranspose =
      [graph transposeTensor:weight
                   dimension:0
               withDimension:1
                        name:[name stringByAppendingString:@"_weight_transpose"]];
  MPSGraphTensor *product =
      [graph matrixMultiplicationWithPrimaryTensor:flat
                                   secondaryTensor:weightTranspose
                                              name:[name stringByAppendingString:@"_matmul"]];
  if (!flat || !weightTranspose || !product)
    return nil;
  if (bias) {
    product = [graph additionWithPrimaryTensor:product
                               secondaryTensor:bias
                                          name:[name stringByAppendingString:@"_bias"]];
    if (!product)
      return nil;
  }
  MPSGraphTensor *bf16 = gs_qwen25_cast_bf16(
      graph, product, [name stringByAppendingString:@"_bf16"]);
  if (!bf16)
    return nil;
  return [graph reshapeTensor:bf16
                    withShape:@[ @(batches), @(tokens), @(outputWidth) ]
                         name:[name stringByAppendingString:@"_batches_tokens"]];
}

static MPSGraphTensor *gs_qwen25_rms_normalize(
    MPSGraph *graph, MPSGraphTensor *inputBf16, uint32_t batches,
    uint32_t tokens, uint32_t dim, float epsilon, NSString *name) {
  if (!graph || !inputBf16 || !name || dim == 0)
    return nil;
  MPSGraphTensor *inputF32 = [graph castTensor:inputBf16
                                       toType:MPSDataTypeFloat32
                                         name:[name stringByAppendingString:@"_input_f32"]];
  MPSGraphTensor *squareF32 = [graph squareWithTensor:inputF32
                                                 name:[name stringByAppendingString:@"_square_f32"]];
  MPSGraphTensor *meanF32 = [graph meanOfTensor:squareF32
                                           axes:@[ @2 ]
                                           name:[name stringByAppendingString:@"_mean_f32"]];
  MPSGraphTensor *epsilonF32 = [graph constantWithScalar:epsilon
                                                    shape:@[ @1 ]
                                                 dataType:MPSDataTypeFloat32];
  MPSGraphTensor *meanPlusEpsilonF32 =
      [graph additionWithPrimaryTensor:meanF32
                        secondaryTensor:epsilonF32
                                   name:[name stringByAppendingString:@"_epsilon_f32"]];
  MPSGraphTensor *inverseRootF32 =
      [graph reciprocalSquareRootWithTensor:meanPlusEpsilonF32
                                       name:[name stringByAppendingString:@"_rsqrt_f32"]];
  MPSGraphTensor *inverseRootExpandedF32 =
      [graph reshapeTensor:inverseRootF32
                 withShape:@[ @(batches), @(tokens), @1 ]
                      name:[name stringByAppendingString:@"_rsqrt_expanded_f32"]];
  MPSGraphTensor *normalizedF32 =
      [graph multiplicationWithPrimaryTensor:inputF32
                             secondaryTensor:inverseRootExpandedF32
                                        name:[name stringByAppendingString:@"_normalized_f32"]];
  MPSGraphTensor *normalizedBf16 =
      gs_qwen25_cast_bf16(graph, normalizedF32,
                          [name stringByAppendingString:@"_normalized_bf16"]);
  if (!inputF32 || !squareF32 || !meanF32 || !epsilonF32 ||
      !meanPlusEpsilonF32 || !inverseRootF32 || !inverseRootExpandedF32 ||
      !normalizedF32 || !normalizedBf16)
    return nil;
  return normalizedBf16;
}

static MPSGraphTensor *gs_qwen25_apply_rms_weight(
    MPSGraph *graph, MPSGraphTensor *normalizedBf16,
    MPSGraphTensor *weightBf16, NSString *name) {
  if (!graph || !normalizedBf16 || !weightBf16 || !name)
    return nil;
  MPSGraphTensor *normalizedF32 = [graph castTensor:normalizedBf16
                                            toType:MPSDataTypeFloat32
                                              name:[name stringByAppendingString:@"_normalized_f32"]];
  MPSGraphTensor *weightF32 = [graph castTensor:weightBf16
                                        toType:MPSDataTypeFloat32
                                          name:[name stringByAppendingString:@"_weight_f32"]];
  MPSGraphTensor *weightedF32 =
      [graph multiplicationWithPrimaryTensor:normalizedF32
                             secondaryTensor:weightF32
                                        name:[name stringByAppendingString:@"_multiply_f32"]];
  MPSGraphTensor *weightedBf16 = gs_qwen25_cast_bf16(
      graph, weightedF32, [name stringByAppendingString:@"_output_bf16"]);
  if (!normalizedF32 || !weightF32 || !weightedF32 || !weightedBf16)
    return nil;
  return weightedBf16;
}

static bool gs_qwen25_rotary_products(
    MPSGraph *graph, MPSGraphTensor *projectedBf16,
    MPSGraphTensor *cosineBf16, MPSGraphTensor *sineBf16,
    uint32_t batches, uint32_t heads, uint32_t tokens, uint32_t headDim,
    NSString *name, MPSGraphTensor **cosProductOut,
    MPSGraphTensor **sinProductOut) {
  if (cosProductOut)
    *cosProductOut = nil;
  if (sinProductOut)
    *sinProductOut = nil;
  if (!graph || !projectedBf16 || !cosineBf16 || !sineBf16 || !name ||
      !cosProductOut || !sinProductOut || batches == 0 || heads == 0 ||
      tokens == 0 || headDim == 0 || (headDim % 2) != 0)
    return false;
  MPSGraphTensor *byTokenHead =
      [graph reshapeTensor:projectedBf16
                 withShape:@[ @(batches), @(tokens), @(heads), @(headDim) ]
                      name:[name stringByAppendingString:@"_bshd_bf16"]];
  MPSGraphTensor *projected = [graph transposeTensor:byTokenHead
                                            dimension:1
                                        withDimension:2
                                                 name:[name stringByAppendingString:@"_bhtd_bf16"]];
  const uint32_t halfDim = headDim / 2;
  MPSGraphTensor *first = [graph sliceTensor:projected
                                   dimension:3
                                       start:0
                                      length:halfDim
                                        name:[name stringByAppendingString:@"_first_half"]];
  MPSGraphTensor *second = [graph sliceTensor:projected
                                    dimension:3
                                        start:halfDim
                                       length:halfDim
                                         name:[name stringByAppendingString:@"_second_half"]];
  MPSGraphTensor *negativeSecond =
      [graph negativeWithTensor:second
                           name:[name stringByAppendingString:@"_negative_second"]];
  MPSGraphTensor *rotated =
      [graph concatTensors:@[ negativeSecond, first ]
                 dimension:3
                      name:[name stringByAppendingString:@"_rotate_half_bf16"]];
  MPSGraphTensor *projectedF32 = [graph castTensor:projected
                                          toType:MPSDataTypeFloat32
                                            name:[name stringByAppendingString:@"_input_f32"]];
  MPSGraphTensor *rotatedF32 = [graph castTensor:rotated
                                        toType:MPSDataTypeFloat32
                                          name:[name stringByAppendingString:@"_rotated_f32"]];
  MPSGraphTensor *cosineF32 = [graph castTensor:cosineBf16
                                        toType:MPSDataTypeFloat32
                                          name:[name stringByAppendingString:@"_cos_f32"]];
  MPSGraphTensor *sineF32 = [graph castTensor:sineBf16
                                       toType:MPSDataTypeFloat32
                                         name:[name stringByAppendingString:@"_sin_f32"]];
  MPSGraphTensor *cosProductF32 =
      [graph multiplicationWithPrimaryTensor:projectedF32
                             secondaryTensor:cosineF32
                                        name:[name stringByAppendingString:@"_cos_product_f32"]];
  MPSGraphTensor *sinProductF32 =
      [graph multiplicationWithPrimaryTensor:rotatedF32
                             secondaryTensor:sineF32
                                        name:[name stringByAppendingString:@"_sin_product_f32"]];
  MPSGraphTensor *cosProduct = gs_qwen25_cast_bf16(
      graph, cosProductF32, [name stringByAppendingString:@"_cos_product_bf16"]);
  MPSGraphTensor *sinProduct = gs_qwen25_cast_bf16(
      graph, sinProductF32, [name stringByAppendingString:@"_sin_product_bf16"]);
  if (!byTokenHead || !projected || !first || !second || !negativeSecond ||
      !rotated || !projectedF32 || !rotatedF32 || !cosineF32 || !sineF32 ||
      !cosProductF32 || !sinProductF32 || !cosProduct || !sinProduct)
    return false;
  *cosProductOut = cosProduct;
  *sinProductOut = sinProduct;
  return true;
}

static MPSGraphTensor *gs_qwen25_add_bf16(
    MPSGraph *graph, MPSGraphTensor *left, MPSGraphTensor *right,
    NSString *name) {
  if (!graph || !left || !right || !name)
    return nil;
  MPSGraphTensor *sum = [graph additionWithPrimaryTensor:left
                                         secondaryTensor:right
                                                    name:name];
  return gs_qwen25_cast_bf16(graph, sum,
                             [name stringByAppendingString:@"_bf16"]);
}

static MPSGraphTensor *gs_qwen25_rope_sum(
    MPSGraph *graph, MPSGraphTensor *cosProduct, MPSGraphTensor *sinProduct,
    NSString *name) {
  return gs_qwen25_add_bf16(graph, cosProduct, sinProduct, name);
}

static GsQwen25MPSGraphStep *gs_qwen25_compile_step(
    id<MTLDevice> device, MPSGraph *graph,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds,
    NSMutableArray<MPSGraphTensor *> *placeholders,
    NSMutableArray<MPSGraphTensorData *> *inputData,
    NSArray<MPSGraphTensor *> *targetTensors,
    NSArray<id<MTLBuffer>> *targetBuffers,
    NSArray<NSArray<NSNumber *> *> *targetShapes, NSString *name,
    char *errorText, size_t errorCapacity) {
  if (!device || !graph || !feeds || !placeholders || !inputData ||
      !targetTensors || !targetBuffers || !targetShapes || !name ||
      targetTensors.count == 0 || targetTensors.count != targetBuffers.count ||
      targetTensors.count != targetShapes.count) {
    gs_qwen25_set_error(errorText, errorCapacity,
                        "invalid prepared MPSGraph step description");
    return nil;
  }
  NSMutableArray<MPSGraphTensorData *> *outputData =
      [NSMutableArray arrayWithCapacity:targetTensors.count];
  for (NSUInteger index = 0; index < targetTensors.count; ++index) {
    MPSGraphTensor *tensor = targetTensors[index];
    NSArray<NSNumber *> *shape = targetShapes[index];
    MPSGraphTensorData *data = nil;
    if (tensor && tensor.dataType == MPSDataTypeBFloat16 &&
        [tensor.shape isEqualToArray:shape])
      data = gs_qwen25_tensor_data(targetBuffers[index], shape,
                                   MPSDataTypeBFloat16);
    if (!data) {
      NSString *message = [NSString stringWithFormat:
          @"%@ output %lu is not a bindable BF16 tensor of the declared shape",
          name, (unsigned long)index];
      gs_qwen25_set_ns_error(errorText, errorCapacity, message);
      return nil;
    }
    [outputData addObject:data];
  }

  MPSGraphExecutable *executable =
      [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                          feeds:(MPSGraphTensorShapedTypeDictionary *)feeds
                  targetTensors:targetTensors
               targetOperations:nil
          compilationDescriptor:nil];
  if (!executable) {
    NSString *message = [NSString stringWithFormat:
        @"MPSGraph could not compile prepared step %@", name];
    gs_qwen25_set_ns_error(errorText, errorCapacity, message);
    return nil;
  }
  executable.options = MPSGraphOptionsNone;
  NSMutableArray<MPSGraphTensorData *> *orderedInputs =
      [NSMutableArray arrayWithCapacity:placeholders.count];
  for (MPSGraphTensor *feedTensor in executable.feedTensors) {
    NSUInteger index = [placeholders indexOfObjectIdenticalTo:feedTensor];
    if (index == NSNotFound || index >= inputData.count) {
      NSString *message = [NSString stringWithFormat:
          @"compiled MPSGraph feed order mismatch in step %@", name];
      gs_qwen25_set_ns_error(errorText, errorCapacity, message);
      return nil;
    }
    [orderedInputs addObject:inputData[index]];
  }
  if (executable.feedTensors.count != placeholders.count ||
      orderedInputs.count != placeholders.count) {
    NSString *message = [NSString stringWithFormat:
        @"compiled MPSGraph feed count mismatch in step %@", name];
    gs_qwen25_set_ns_error(errorText, errorCapacity, message);
    return nil;
  }
  MPSGraphExecutableExecutionDescriptor *execution =
      [MPSGraphExecutableExecutionDescriptor new];
  execution.waitUntilCompleted = NO;
  GsQwen25MPSGraphStep *step = [GsQwen25MPSGraphStep new];
  step.graph = graph;
  step.executable = executable;
  step.execution = execution;
  step.inputs = [orderedInputs copy];
  step.outputs = [outputData copy];
  step.name = name;
  step.outputCount = targetTensors.count;
  if (!execution || !step.graph || !step.executable || !step.inputs ||
      !step.outputs || !step.name) {
    NSString *message = [NSString stringWithFormat:
        @"unable to retain prepared MPSGraph step %@", name];
    gs_qwen25_set_ns_error(errorText, errorCapacity, message);
    return nil;
  }
  return step;
}

static MPSGraph *gs_qwen25_new_graph(char *errorText,
                                     size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  if (!graph) {
    gs_qwen25_set_error(errorText, errorCapacity,
                        "unable to allocate MPSGraph step");
    return nil;
  }
  graph.options = MPSGraphOptionsNone;
  return graph;
}

static MPSGraphTensor *gs_qwen25_add_bf16_feed(
    MPSGraph *graph,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds,
    NSMutableArray<MPSGraphTensor *> *placeholders,
    NSMutableArray<MPSGraphTensorData *> *inputData, id<MTLBuffer> buffer,
    NSArray<NSNumber *> *shape, NSString *name) {
  return gs_qwen25_add_feed(graph, feeds, placeholders, inputData, buffer,
                            shape, MPSDataTypeBFloat16, name);
}

static MPSGraphTensor *gs_qwen25_add_f32_feed(
    MPSGraph *graph,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds,
    NSMutableArray<MPSGraphTensor *> *placeholders,
    NSMutableArray<MPSGraphTensorData *> *inputData, id<MTLBuffer> buffer,
    NSArray<NSNumber *> *shape, NSString *name) {
  return gs_qwen25_add_feed(graph, feeds, placeholders, inputData, buffer,
                            shape, MPSDataTypeFloat32, name);
}

static GsQwen25MPSGraphBuilder *gs_qwen25_new_builder(
    char *errorText, size_t errorCapacity) {
  GsQwen25MPSGraphBuilder *builder = [GsQwen25MPSGraphBuilder new];
  builder.graph = gs_qwen25_new_graph(errorText, errorCapacity);
  builder.feeds = [NSMutableDictionary new];
  builder.placeholders = [NSMutableArray new];
  builder.inputData = [NSMutableArray new];
  if (!builder || !builder.graph || !builder.feeds || !builder.placeholders ||
      !builder.inputData) {
    gs_qwen25_set_error(errorText, errorCapacity,
                        "unable to prepare MPSGraph feed builder");
    return nil;
  }
  return builder;
}

static MPSGraphTensor *gs_qwen25_builder_add_bf16(
    GsQwen25MPSGraphBuilder *builder, id<MTLBuffer> buffer,
    NSArray<NSNumber *> *shape, NSString *name) {
  if (!builder)
    return nil;
  return gs_qwen25_add_bf16_feed(builder.graph, builder.feeds,
                                 builder.placeholders, builder.inputData,
                                 buffer, shape, name);
}

static MPSGraphTensor *gs_qwen25_builder_add_f32(
    GsQwen25MPSGraphBuilder *builder, id<MTLBuffer> buffer,
    NSArray<NSNumber *> *shape, NSString *name) {
  if (!builder)
    return nil;
  return gs_qwen25_add_f32_feed(builder.graph, builder.feeds,
                                builder.placeholders, builder.inputData,
                                buffer, shape, name);
}

static GsQwen25MPSGraphStep *gs_qwen25_builder_compile(
    GsQwen25MPSGraphBuilder *builder, id<MTLDevice> device,
    NSArray<MPSGraphTensor *> *targetTensors,
    NSArray<id<MTLBuffer>> *targetBuffers,
    NSArray<NSArray<NSNumber *> *> *targetShapes, NSString *name,
    char *errorText, size_t errorCapacity) {
  if (!builder) {
    gs_qwen25_set_error(errorText, errorCapacity,
                        "MPSGraph feed builder is unavailable");
    return nil;
  }
  return gs_qwen25_compile_step(
      device, builder.graph, builder.feeds, builder.placeholders,
      builder.inputData, targetTensors, targetBuffers, targetShapes, name,
      errorText, errorCapacity);
}

extern "C" void *gs_qwen25_mps_block_plan_create(
    void *deviceHandle, int32_t batches, int32_t tokens, int32_t dim,
    int32_t ffnDim, int32_t heads, int32_t kvHeads, int32_t headDim,
    float rmsNormEpsilon, void *inputHandle, void *maskHandle,
    void *positionIdsHandle, void *invFreqHandle,
    void *const *weightHandles, void *const *stageHandles,
    void *const *scratchHandles, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity > 0)
    errorText[0] = '\0';
  if (!deviceHandle || !inputHandle || !maskHandle || !positionIdsHandle ||
      !invFreqHandle || !weightHandles || !stageHandles || !scratchHandles) {
    gs_qwen25_set_error(errorText, errorCapacity,
                        "required device, input, mask, RoPE, weight, stage, or scratch handle is null");
    return nullptr;
  }
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      @try {
      if (batches <= 0 || tokens <= 0 || dim <= 0 || ffnDim <= 0 || heads <= 0 ||
          kvHeads <= 0 || headDim <= 0 || (headDim % 2) != 0 ||
          (uint64_t)heads * (uint64_t)headDim != (uint64_t)dim ||
          kvHeads > heads || heads % kvHeads != 0 ||
          !std::isfinite(rmsNormEpsilon) || rmsNormEpsilon <= 0.0f) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "unsupported dimensions: require D=H*HD, even HD, divisible H/KVH, and positive finite RMS epsilon");
        return nullptr;
      }

      const uint64_t batchTokensDims[] = {(uint64_t)batches, (uint64_t)tokens};
      const uint64_t hiddenDims[] = {(uint64_t)batches, (uint64_t)tokens,
                                     (uint64_t)dim};
      const uint64_t kvDims[] = {(uint64_t)batches, (uint64_t)tokens,
                                 (uint64_t)kvHeads, (uint64_t)headDim};
      const uint64_t qDims[] = {(uint64_t)batches, (uint64_t)tokens,
                                (uint64_t)heads, (uint64_t)headDim};
      const uint64_t feedForwardDims[] = {(uint64_t)batches, (uint64_t)tokens,
                                          (uint64_t)ffnDim};
      const uint64_t maskDims[] = {(uint64_t)tokens, (uint64_t)tokens};
      const uint64_t qWeightDims[] = {(uint64_t)dim, (uint64_t)dim};
      const uint64_t kvWeightDims[] = {(uint64_t)kvHeads * (uint64_t)headDim,
                                       (uint64_t)dim};
      const uint64_t gateWeightDims[] = {(uint64_t)ffnDim, (uint64_t)dim};
      const uint64_t downWeightDims[] = {(uint64_t)dim, (uint64_t)ffnDim};
      uint64_t batchTokens = 0, hiddenElements = 0, kvElements = 0;
      uint64_t queryElements = 0, ffnElements = 0, maskElements = 0;
      uint64_t qWeightElements = 0, kvWeightElements = 0;
      uint64_t gateWeightElements = 0, downWeightElements = 0;
      if (!gs_qwen25_product(batchTokensDims, 2, &batchTokens) ||
          !gs_qwen25_product(hiddenDims, 3, &hiddenElements) ||
          !gs_qwen25_product(kvDims, 4, &kvElements) ||
          !gs_qwen25_product(qDims, 4, &queryElements) ||
          !gs_qwen25_product(feedForwardDims, 3, &ffnElements) ||
          !gs_qwen25_product(maskDims, 2, &maskElements) ||
          !gs_qwen25_product(qWeightDims, 2, &qWeightElements) ||
          !gs_qwen25_product(kvWeightDims, 2, &kvWeightElements) ||
          !gs_qwen25_product(gateWeightDims, 2, &gateWeightElements) ||
          !gs_qwen25_product(downWeightDims, 2, &downWeightElements)) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "tensor element count exceeds the supported bound");
        return nullptr;
      }
      (void)queryElements;

      const uint64_t weightElements[kGsQwen25WeightCount] = {
          (uint64_t)dim, qWeightElements, (uint64_t)dim,
          kvWeightElements, (uint64_t)kvHeads * (uint64_t)headDim,
          kvWeightElements, (uint64_t)kvHeads * (uint64_t)headDim,
          qWeightElements, (uint64_t)dim, gateWeightElements,
          gateWeightElements, downWeightElements};
      const uint64_t rotaryElements = (uint64_t)tokens * (uint64_t)headDim;
      const uint64_t stageElements[kGsQwen25StageCount] = {
          hiddenElements, hiddenElements, kvElements, kvElements, queryElements,
          kvElements, hiddenElements, hiddenElements, hiddenElements,
          hiddenElements, ffnElements, ffnElements, ffnElements,
          hiddenElements, hiddenElements, rotaryElements, rotaryElements};
      const uint64_t scratchElements[kGsQwen25ScratchCount] = {
          hiddenElements, queryElements, queryElements, kvElements, kvElements,
          hiddenElements, ffnElements};

      uint64_t inputBytes = 0, maskBytes = 0, positionsBytes = 0;
      uint64_t invFreqBytes = 0;
      if (hiddenElements > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          kvElements > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          maskElements > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          ffnElements > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          (uint64_t)dim > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          qWeightElements > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          kvWeightElements > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          gateWeightElements > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          downWeightElements > UINT64_MAX / kGsQwen25BFloat16Bytes ||
          (uint64_t)tokens > UINT64_MAX / sizeof(float) ||
          (uint64_t)(headDim / 2) > UINT64_MAX / sizeof(float)) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "tensor byte count overflow");
        return nullptr;
      }
      inputBytes = hiddenElements * kGsQwen25BFloat16Bytes;
      maskBytes = maskElements * kGsQwen25BFloat16Bytes;
      positionsBytes = (uint64_t)tokens * sizeof(float);
      invFreqBytes = (uint64_t)(headDim / 2) * sizeof(float);

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
      id<MTLBuffer> mask = (__bridge id<MTLBuffer>)maskHandle;
      id<MTLBuffer> positions = (__bridge id<MTLBuffer>)positionIdsHandle;
      id<MTLBuffer> invFreq = (__bridge id<MTLBuffer>)invFreqHandle;
      NSMutableArray<id<MTLBuffer>> *weights =
          [NSMutableArray arrayWithCapacity:kGsQwen25WeightCount];
      NSMutableArray<id<MTLBuffer>> *stages =
          [NSMutableArray arrayWithCapacity:kGsQwen25StageCount];
      NSMutableArray<id<MTLBuffer>> *scratches =
          [NSMutableArray arrayWithCapacity:kGsQwen25ScratchCount];
      if (!device || !weights || !stages || !scratches) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "unable to construct native buffer bindings");
        return nullptr;
      }
      for (NSUInteger index = 0; index < kGsQwen25WeightCount; ++index) {
        if (!weightHandles[index]) {
          gs_qwen25_set_error(errorText, errorCapacity,
                              "a required BF16 weight handle is null");
          return nullptr;
        }
        [weights addObject:(__bridge id<MTLBuffer>)weightHandles[index]];
      }
      for (NSUInteger index = 0; index < kGsQwen25StageCount; ++index) {
        if (!stageHandles[index]) {
          gs_qwen25_set_error(errorText, errorCapacity,
                              "a required BF16 stage handle is null");
          return nullptr;
        }
        [stages addObject:(__bridge id<MTLBuffer>)stageHandles[index]];
      }
      for (NSUInteger index = 0; index < kGsQwen25ScratchCount; ++index) {
        if (!scratchHandles[index]) {
          gs_qwen25_set_error(errorText, errorCapacity,
                              "a required BF16 scratch handle is null");
          return nullptr;
        }
        [scratches addObject:(__bridge id<MTLBuffer>)scratchHandles[index]];
      }

      NSArray<id<MTLBuffer>> *weightArray = [weights copy];
      NSArray<id<MTLBuffer>> *stageArray = [stages copy];
      NSArray<id<MTLBuffer>> *scratchArray = [scratches copy];
      NSMutableArray<id<MTLBuffer>> *allBuffers = [NSMutableArray new];
      [allBuffers addObject:input];
      [allBuffers addObject:mask];
      [allBuffers addObject:positions];
      [allBuffers addObject:invFreq];
      [allBuffers addObjectsFromArray:weightArray];
      [allBuffers addObjectsFromArray:stageArray];
      [allBuffers addObjectsFromArray:scratchArray];

      const uint64_t fixedLengths[] = {inputBytes, maskBytes, positionsBytes,
                                       invFreqBytes};
      for (NSUInteger index = 0; index < allBuffers.count; ++index) {
        id<MTLBuffer> buffer = allBuffers[index];
        uint64_t requiredBytes = 0;
        if (index < 4) {
          requiredBytes = fixedLengths[index];
        } else if (index < 4 + kGsQwen25WeightCount) {
          const uint64_t elements = weightElements[index - 4];
          requiredBytes = elements * kGsQwen25BFloat16Bytes;
        } else if (index < 4 + kGsQwen25WeightCount +
                               kGsQwen25StageCount) {
          const uint64_t elements =
              stageElements[index - 4 - kGsQwen25WeightCount];
          requiredBytes = elements * kGsQwen25BFloat16Bytes;
        } else {
          const uint64_t elements =
              scratchElements[index - 4 - kGsQwen25WeightCount -
                              kGsQwen25StageCount];
          requiredBytes = elements * kGsQwen25BFloat16Bytes;
        }
        if (!buffer || buffer.device != device ||
            buffer.storageMode != MTLStorageModeShared || !buffer.contents ||
            buffer.length < requiredBytes) {
          gs_qwen25_set_error(errorText, errorCapacity,
                              "all feeds, stages, weights, and scratch must be sufficiently large shared MTLBuffers on the supplied device");
          return nullptr;
        }
      }
      for (NSUInteger left = 0; left < allBuffers.count; ++left) {
        for (NSUInteger right = left + 1; right < allBuffers.count; ++right) {
          if (!gs_qwen25_disjoint(allBuffers[left], allBuffers[right])) {
            gs_qwen25_set_error(errorText, errorCapacity,
                                "feed, weight, stage, and scratch MTLBuffers must be fully disjoint");
            return nullptr;
          }
        }
      }

      NSArray<NSNumber *> *hiddenShape =
          @[ @(batches), @(tokens), @(dim) ];
      NSArray<NSNumber *> *maskShape = @[ @1, @1, @(tokens), @(tokens) ];
      NSArray<NSNumber *> *positionShape = @[ @(tokens) ];
      NSArray<NSNumber *> *invFreqShape = @[ @(headDim / 2) ];
      NSArray<NSNumber *> *normShape = @[ @(dim) ];
      NSArray<NSNumber *> *qWeightShape = @[ @(dim), @(dim) ];
      NSArray<NSNumber *> *qBiasShape = @[ @(dim) ];
      NSArray<NSNumber *> *kvWeightShape =
          @[ @((int64_t)kvHeads * headDim), @(dim) ];
      NSArray<NSNumber *> *kvBiasShape = @[ @((int64_t)kvHeads * headDim) ];
      NSArray<NSNumber *> *oWeightShape = @[ @(dim), @(dim) ];
      NSArray<NSNumber *> *gateWeightShape = @[ @(ffnDim), @(dim) ];
      NSArray<NSNumber *> *downWeightShape = @[ @(dim), @(ffnDim) ];
      NSArray<NSNumber *> *qProjectedShape = @[ @(batches), @(tokens), @(dim) ];
      NSArray<NSNumber *> *kvProjectedShape =
          @[ @(batches), @(tokens), @((int64_t)kvHeads * headDim) ];
      NSArray<NSNumber *> *qRopeShape =
          @[ @(batches), @(heads), @(tokens), @(headDim) ];
      NSArray<NSNumber *> *kRopeShape =
          @[ @(batches), @(kvHeads), @(tokens), @(headDim) ];
      NSArray<NSNumber *> *ffnShape = @[ @(batches), @(tokens), @(ffnDim) ];
      NSArray<NSNumber *> *rotaryShape = @[ @1, @1, @(tokens), @(headDim) ];

      NSArray<NSArray<NSNumber *> *> *stageShapes = @[
        hiddenShape, qProjectedShape, kvProjectedShape, kvProjectedShape,
        qRopeShape, kRopeShape, hiddenShape, hiddenShape, hiddenShape,
        hiddenShape, ffnShape, ffnShape, ffnShape, hiddenShape, hiddenShape,
        rotaryShape, rotaryShape];
      NSArray<NSArray<NSNumber *> *> *scratchShapes = @[
        hiddenShape, qRopeShape, qRopeShape, kRopeShape, kRopeShape,
        hiddenShape, ffnShape];
      if (stageShapes.count != kGsQwen25StageCount ||
          scratchShapes.count != kGsQwen25ScratchCount) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "internal stage or scratch shape count mismatch");
        return nullptr;
      }

      NSMutableArray<GsQwen25MPSGraphStep *> *preparedSteps =
          [NSMutableArray arrayWithCapacity:15];
      if (!preparedSteps) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "unable to allocate prepared graph step list");
        return nullptr;
      }
#define GS_QWEN25_APPEND_STEP(stepExpression)                                  \
      do {                                                                      \
        GsQwen25MPSGraphStep *preparedStep = (stepExpression);                  \
        if (!preparedStep)                                                      \
          return nullptr;                                                       \
        [preparedSteps addObject:preparedStep];                                 \
      } while (0)

      // Materialize the normalized activation before the BF16 gain multiply.
      GsQwen25MPSGraphBuilder *builder =
          gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *normInput = gs_qwen25_builder_add_bf16(
          builder, input, hiddenShape, @"hidden_in_bf16");
      MPSGraphTensor *normalized = gs_qwen25_rms_normalize(
          builder.graph, normInput, (uint32_t)batches, (uint32_t)tokens,
          (uint32_t)dim, rmsNormEpsilon, @"input_layernorm");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ normalized ], @[ scratchArray[0] ],
          @[ hiddenShape ], @"input_norm_normalized", errorText,
          errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *normalizedFeed = gs_qwen25_builder_add_bf16(
          builder, scratchArray[0], hiddenShape, @"input_normalized_bf16");
      MPSGraphTensor *inputNormWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[0], normShape, @"input_layernorm_weight_bf16");
      MPSGraphTensor *inputNorm = gs_qwen25_apply_rms_weight(
          builder.graph, normalizedFeed, inputNormWeight,
          @"input_layernorm_gain");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ inputNorm ], @[ stageArray[0] ],
          @[ hiddenShape ], @"input_norm_gain", errorText, errorCapacity));

      // Q/K/V and trig are independent outputs; consumers read their bound buffers.
      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *normalizedForProjection = gs_qwen25_builder_add_bf16(
          builder, stageArray[0], hiddenShape, @"input_norm_bf16");
      MPSGraphTensor *qWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[1], qWeightShape, @"q_proj_weight_bf16");
      MPSGraphTensor *qBias = gs_qwen25_builder_add_bf16(
          builder, weightArray[2], qBiasShape, @"q_proj_bias_bf16");
      MPSGraphTensor *kWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[3], kvWeightShape, @"k_proj_weight_bf16");
      MPSGraphTensor *kBias = gs_qwen25_builder_add_bf16(
          builder, weightArray[4], kvBiasShape, @"k_proj_bias_bf16");
      MPSGraphTensor *vWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[5], kvWeightShape, @"v_proj_weight_bf16");
      MPSGraphTensor *vBias = gs_qwen25_builder_add_bf16(
          builder, weightArray[6], kvBiasShape, @"v_proj_bias_bf16");
      MPSGraphTensor *positionIds = gs_qwen25_builder_add_f32(
          builder, positions, positionShape, @"text_position_ids_f32");
      MPSGraphTensor *invFreqInput = gs_qwen25_builder_add_f32(
          builder, invFreq, invFreqShape, @"text_inv_freq_f32");
      MPSGraphTensor *qProjection = gs_qwen25_linear(
          builder.graph, normalizedForProjection, qWeight, qBias,
          (uint32_t)batches, (uint32_t)tokens, (uint32_t)dim,
          (uint32_t)dim, @"q_proj");
      const uint32_t kvWidth = (uint32_t)kvHeads * (uint32_t)headDim;
      MPSGraphTensor *kProjection = gs_qwen25_linear(
          builder.graph, normalizedForProjection, kWeight, kBias,
          (uint32_t)batches, (uint32_t)tokens, (uint32_t)dim, kvWidth,
          @"k_proj");
      MPSGraphTensor *vProjection = gs_qwen25_linear(
          builder.graph, normalizedForProjection, vWeight, vBias,
          (uint32_t)batches, (uint32_t)tokens, (uint32_t)dim, kvWidth,
          @"v_proj");
      MPSGraphTensor *positionsMatrix =
          [builder.graph reshapeTensor:positionIds
                             withShape:@[ @(tokens), @1 ]
                                  name:@"text_positions_matrix_f32"];
      MPSGraphTensor *invFreqMatrix =
          [builder.graph reshapeTensor:invFreqInput
                             withShape:@[ @1, @(headDim / 2) ]
                                  name:@"text_inv_freq_matrix_f32"];
      MPSGraphTensor *frequencies =
          [builder.graph matrixMultiplicationWithPrimaryTensor:positionsMatrix
                                               secondaryTensor:invFreqMatrix
                                                          name:@"text_rope_frequency_outer_product_f32"];
      MPSGraphTensor *cosineHalf = [builder.graph cosWithTensor:frequencies
                                                          name:@"text_rope_cosine_half_f32"];
      MPSGraphTensor *sineHalf = [builder.graph sinWithTensor:frequencies
                                                        name:@"text_rope_sine_half_f32"];
      MPSGraphTensor *cosineFull =
          [builder.graph concatTensor:cosineHalf
                           withTensor:cosineHalf
                            dimension:1
                                 name:@"text_rope_cosine_split_half_f32"];
      MPSGraphTensor *sineFull =
          [builder.graph concatTensor:sineHalf
                           withTensor:sineHalf
                            dimension:1
                                 name:@"text_rope_sine_split_half_f32"];
      MPSGraphTensor *cosineFlat = gs_qwen25_cast_bf16(
          builder.graph, cosineFull, @"text_rope_cosine_bf16");
      MPSGraphTensor *sineFlat = gs_qwen25_cast_bf16(
          builder.graph, sineFull, @"text_rope_sine_bf16");
      MPSGraphTensor *cosine = [builder.graph reshapeTensor:cosineFlat
                                                   withShape:rotaryShape
                                                        name:@"text_rope_cosine_broadcast_bf16"];
      MPSGraphTensor *sine = [builder.graph reshapeTensor:sineFlat
                                                  withShape:rotaryShape
                                                       name:@"text_rope_sine_broadcast_bf16"];
      if (!qProjection || !kProjection || !vProjection || !positionsMatrix ||
          !invFreqMatrix || !frequencies || !cosineHalf || !sineHalf ||
          !cosineFull || !sineFull || !cosineFlat || !sineFlat || !cosine ||
          !sine) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "failed to build QKV and text-RoPE graph step");
        return nullptr;
      }
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device,
          @[ qProjection, kProjection, vProjection, cosine, sine ],
          @[ stageArray[1], stageArray[2], stageArray[3], stageArray[15],
             stageArray[16] ],
          @[ qProjectedShape, kvProjectedShape, kvProjectedShape, rotaryShape,
             rotaryShape ],
          @"qkv_and_rotary", errorText, errorCapacity));

      // Each BF16 RoPE multiply is an output buffer consumed by the next graph.
      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *qProjected = gs_qwen25_builder_add_bf16(
          builder, stageArray[1], qProjectedShape, @"q_projection_bf16");
      MPSGraphTensor *kProjected = gs_qwen25_builder_add_bf16(
          builder, stageArray[2], kvProjectedShape, @"k_projection_bf16");
      MPSGraphTensor *cosForRope = gs_qwen25_builder_add_bf16(
          builder, stageArray[15], rotaryShape, @"text_rope_cosine_bf16");
      MPSGraphTensor *sinForRope = gs_qwen25_builder_add_bf16(
          builder, stageArray[16], rotaryShape, @"text_rope_sine_bf16");
      MPSGraphTensor *qCos = nil;
      MPSGraphTensor *qSin = nil;
      MPSGraphTensor *kCos = nil;
      MPSGraphTensor *kSin = nil;
      if (!gs_qwen25_rotary_products(
              builder.graph, qProjected, cosForRope, sinForRope,
              (uint32_t)batches, (uint32_t)heads, (uint32_t)tokens,
              (uint32_t)headDim, @"q_rope", &qCos, &qSin) ||
          !gs_qwen25_rotary_products(
              builder.graph, kProjected, cosForRope, sinForRope,
              (uint32_t)batches, (uint32_t)kvHeads, (uint32_t)tokens,
              (uint32_t)headDim, @"k_rope", &kCos, &kSin)) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "failed to build BF16 RoPE product graph step");
        return nullptr;
      }
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ qCos, qSin, kCos, kSin ],
          @[ scratchArray[1], scratchArray[2], scratchArray[3],
             scratchArray[4] ],
          @[ qRopeShape, qRopeShape, kRopeShape, kRopeShape ],
          @"rope_products", errorText, errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *qCosFeed = gs_qwen25_builder_add_bf16(
          builder, scratchArray[1], qRopeShape, @"q_cos_product_bf16");
      MPSGraphTensor *qSinFeed = gs_qwen25_builder_add_bf16(
          builder, scratchArray[2], qRopeShape, @"q_sin_product_bf16");
      MPSGraphTensor *kCosFeed = gs_qwen25_builder_add_bf16(
          builder, scratchArray[3], kRopeShape, @"k_cos_product_bf16");
      MPSGraphTensor *kSinFeed = gs_qwen25_builder_add_bf16(
          builder, scratchArray[4], kRopeShape, @"k_sin_product_bf16");
      MPSGraphTensor *qRope = gs_qwen25_rope_sum(
          builder.graph, qCosFeed, qSinFeed, @"q_rope_sum");
      MPSGraphTensor *kRope = gs_qwen25_rope_sum(
          builder.graph, kCosFeed, kSinFeed, @"k_rope_sum");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ qRope, kRope ],
          @[ stageArray[4], stageArray[5] ], @[ qRopeShape, kRopeShape ],
          @"rope_sums", errorText, errorCapacity));

      // Q/K/V are read from their stage buffers before F32 SDPA arithmetic.
      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *qAttention = gs_qwen25_builder_add_bf16(
          builder, stageArray[4], qRopeShape, @"q_rope_bf16");
      MPSGraphTensor *kAttention = gs_qwen25_builder_add_bf16(
          builder, stageArray[5], kRopeShape, @"k_rope_bf16");
      MPSGraphTensor *vProjectionFeed = gs_qwen25_builder_add_bf16(
          builder, stageArray[3], kvProjectedShape, @"v_projection_bf16");
      MPSGraphTensor *maskInput = gs_qwen25_builder_add_bf16(
          builder, mask, maskShape, @"additive_mask_bf16");
      const uint32_t repeats = (uint32_t)heads / (uint32_t)kvHeads;
      NSArray<NSNumber *> *expandedKvShape =
          @[ @(batches), @(kvHeads), @1, @(tokens), @(headDim) ];
      NSArray<NSNumber *> *repeatedKvShape =
          @[ @(batches), @(kvHeads), @(repeats), @(tokens), @(headDim) ];
      NSArray<NSNumber *> *queryHeadShape =
          @[ @(batches), @(heads), @(tokens), @(headDim) ];
      MPSGraphTensor *kExpanded =
          [builder.graph reshapeTensor:kAttention
                             withShape:expandedKvShape
                                  name:@"k_gqa_expand_group_axis_bf16"];
      MPSGraphTensor *kRepeated =
          [builder.graph broadcastTensor:kExpanded
                                 toShape:repeatedKvShape
                                    name:@"k_gqa_repeat_each_kv_head_bf16"];
      MPSGraphTensor *kForAttention =
          [builder.graph reshapeTensor:kRepeated
                             withShape:queryHeadShape
                                  name:@"k_gqa_repeat_kv_order_bf16"];
      MPSGraphTensor *vByHead =
          [builder.graph reshapeTensor:vProjectionFeed
                             withShape:@[ @(batches), @(tokens), @(kvHeads),
                                          @(headDim) ]
                                  name:@"v_projection_kv_heads_tokens_bf16"];
      vByHead = [builder.graph transposeTensor:vByHead
                                        dimension:1
                                    withDimension:2
                                             name:@"v_projection_bhkvtd_bf16"];
      MPSGraphTensor *vExpanded =
          [builder.graph reshapeTensor:vByHead
                             withShape:expandedKvShape
                                  name:@"v_gqa_expand_group_axis_bf16"];
      MPSGraphTensor *vRepeated =
          [builder.graph broadcastTensor:vExpanded
                                 toShape:repeatedKvShape
                                    name:@"v_gqa_repeat_each_kv_head_bf16"];
      MPSGraphTensor *vForAttention =
          [builder.graph reshapeTensor:vRepeated
                             withShape:queryHeadShape
                                  name:@"v_gqa_repeat_kv_order_bf16"];
      MPSGraphTensor *kTranspose = [builder.graph transposeTensor:kForAttention
                                                         dimension:2
                                                     withDimension:3
                                                              name:@"k_attention_transpose_bf16"];
      MPSGraphTensor *scores =
          [builder.graph matrixMultiplicationWithPrimaryTensor:qAttention
                                               secondaryTensor:kTranspose
                                                          name:@"sdpa_qk_scores"];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
      if (@available(macOS 15.0, *)) {
        if (scores.dataType == MPSDataTypeFloat32) {
          MPSGraphTensor *mps15Workaround =
              [builder.graph constantWithScalar:1.0e-20f
                                          shape:@[ @1 ]
                                       dataType:MPSDataTypeFloat32];
          scores = [builder.graph additionWithPrimaryTensor:scores
                                            secondaryTensor:mps15Workaround
                                                       name:@"sdpa_qk_macos15_workaround"];
        }
      }
#endif
      MPSGraphTensor *scoresF32 = scores;
      if (scoresF32.dataType != MPSDataTypeFloat32) {
        scoresF32 = [builder.graph castTensor:scoresF32
                                      toType:MPSDataTypeFloat32
                                        name:@"sdpa_qk_scores_f32"];
      }
      MPSGraphTensor *scale = [builder.graph constantWithScalar:
                                   1.0f / std::sqrt((float)headDim)
                                                 shape:@[ @1 ]
                                              dataType:MPSDataTypeFloat32];
      MPSGraphTensor *scaledScoresF32 =
          [builder.graph multiplicationWithPrimaryTensor:scoresF32
                                         secondaryTensor:scale
                                                    name:@"sdpa_scaled_scores_f32"];
      MPSGraphTensor *scaledScores = scaledScoresF32;
      if (scaledScores.dataType != qAttention.dataType) {
        scaledScores = [builder.graph castTensor:scaledScores
                                         toType:qAttention.dataType
                                           name:@"sdpa_scaled_scores_bf16"];
      }
      MPSGraphTensor *maskedScores =
          [builder.graph additionWithPrimaryTensor:scaledScores
                                  secondaryTensor:maskInput
                                             name:@"sdpa_masked_scores_bf16"];
      MPSGraphTensor *probabilities = [builder.graph softMaxWithTensor:maskedScores
                                                                  axis:3
                                                                  name:@"sdpa_softmax_bf16"];
      MPSGraphTensor *attendedHeads =
          [builder.graph matrixMultiplicationWithPrimaryTensor:probabilities
                                               secondaryTensor:vForAttention
                                                          name:@"sdpa_pv_bf16"];
      MPSGraphTensor *attendedTokens = [builder.graph transposeTensor:attendedHeads
                                                             dimension:1
                                                         withDimension:2
                                                                  name:@"sdpa_attended_bshd"];
      MPSGraphTensor *attendedFlat = [builder.graph reshapeTensor:attendedTokens
                                                       withShape:hiddenShape
                                                            name:@"sdpa_attended_flat"];
      MPSGraphTensor *attended = attendedFlat;
      if (attended.dataType != MPSDataTypeBFloat16) {
        attended = gs_qwen25_cast_bf16(
            builder.graph, attended, @"attended_bf16");
      }
      if (!kExpanded || !kRepeated || !kForAttention || !vByHead ||
          !vExpanded || !vRepeated || !vForAttention || !kTranspose ||
          !scores || !scoresF32 || !scale || !scaledScoresF32 ||
          !scaledScores || !maskedScores || !probabilities ||
          !attendedHeads || !attendedTokens || !attendedFlat || !attended) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "failed to build GQA attention graph step");
        return nullptr;
      }
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ attended ], @[ stageArray[6] ],
          @[ hiddenShape ], @"attention", errorText, errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *attendedFeed = gs_qwen25_builder_add_bf16(
          builder, stageArray[6], hiddenShape, @"attended_bf16");
      MPSGraphTensor *oWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[7], oWeightShape, @"o_proj_weight_bf16");
      MPSGraphTensor *oProjection = gs_qwen25_linear(
          builder.graph, attendedFeed, oWeight, nil, (uint32_t)batches,
          (uint32_t)tokens, (uint32_t)dim, (uint32_t)dim, @"o_proj");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ oProjection ], @[ stageArray[7] ],
          @[ hiddenShape ], @"o_proj", errorText, errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *residualInput = gs_qwen25_builder_add_bf16(
          builder, input, hiddenShape, @"hidden_in_bf16");
      MPSGraphTensor *oProjectionFeed = gs_qwen25_builder_add_bf16(
          builder, stageArray[7], hiddenShape, @"o_projection_bf16");
      MPSGraphTensor *attentionResidual = gs_qwen25_add_bf16(
          builder.graph, residualInput, oProjectionFeed,
          @"attention_residual");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ attentionResidual ], @[ stageArray[8] ],
          @[ hiddenShape ], @"attention_residual", errorText,
          errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *residualFeed = gs_qwen25_builder_add_bf16(
          builder, stageArray[8], hiddenShape, @"attention_residual_bf16");
      MPSGraphTensor *postNormalized = gs_qwen25_rms_normalize(
          builder.graph, residualFeed, (uint32_t)batches,
          (uint32_t)tokens, (uint32_t)dim, rmsNormEpsilon,
          @"post_attention_layernorm");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ postNormalized ], @[ scratchArray[5] ],
          @[ hiddenShape ], @"post_norm_normalized", errorText,
          errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *postNormalizedFeed = gs_qwen25_builder_add_bf16(
          builder, scratchArray[5], hiddenShape, @"post_normalized_bf16");
      MPSGraphTensor *postNormWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[8], normShape,
          @"post_attention_layernorm_weight_bf16");
      MPSGraphTensor *postNorm = gs_qwen25_apply_rms_weight(
          builder.graph, postNormalizedFeed, postNormWeight,
          @"post_attention_layernorm_gain");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ postNorm ], @[ stageArray[9] ],
          @[ hiddenShape ], @"post_norm_gain", errorText, errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *postNormFeed = gs_qwen25_builder_add_bf16(
          builder, stageArray[9], hiddenShape, @"post_norm_bf16");
      MPSGraphTensor *gateWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[9], gateWeightShape, @"gate_proj_weight_bf16");
      MPSGraphTensor *upWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[10], gateWeightShape, @"up_proj_weight_bf16");
      MPSGraphTensor *gateProjection = gs_qwen25_linear(
          builder.graph, postNormFeed, gateWeight, nil, (uint32_t)batches,
          (uint32_t)tokens, (uint32_t)dim, (uint32_t)ffnDim, @"gate_proj");
      MPSGraphTensor *upProjection = gs_qwen25_linear(
          builder.graph, postNormFeed, upWeight, nil, (uint32_t)batches,
          (uint32_t)tokens, (uint32_t)dim, (uint32_t)ffnDim, @"up_proj");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ gateProjection, upProjection ],
          @[ stageArray[10], stageArray[11] ], @[ ffnShape, ffnShape ],
          @"gate_up_projections", errorText, errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *gateFeed = gs_qwen25_builder_add_bf16(
          builder, stageArray[10], ffnShape, @"gate_projection_bf16");
      MPSGraphTensor *gateF32 = [builder.graph castTensor:gateFeed
                                                 toType:MPSDataTypeFloat32
                                                   name:@"gate_activation_f32"];
      MPSGraphTensor *gateSigmoid = [builder.graph sigmoidWithTensor:gateF32
                                                               name:@"gate_sigmoid_f32"];
      MPSGraphTensor *siluF32 =
          [builder.graph multiplicationWithPrimaryTensor:gateF32
                                        secondaryTensor:gateSigmoid
                                                   name:@"gate_silu_f32"];
      MPSGraphTensor *silu = gs_qwen25_cast_bf16(
          builder.graph, siluF32, @"gate_silu_bf16");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ silu ], @[ scratchArray[6] ],
          @[ ffnShape ], @"silu_activation", errorText, errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *siluFeed = gs_qwen25_builder_add_bf16(
          builder, scratchArray[6], ffnShape, @"silu_bf16");
      MPSGraphTensor *upFeed = gs_qwen25_builder_add_bf16(
          builder, stageArray[11], ffnShape, @"up_projection_bf16");
      MPSGraphTensor *siluF32Product = [builder.graph castTensor:siluFeed
                                                       toType:MPSDataTypeFloat32
                                                         name:@"silu_product_input_f32"];
      MPSGraphTensor *upF32Product = [builder.graph castTensor:upFeed
                                                     toType:MPSDataTypeFloat32
                                                       name:@"up_product_input_f32"];
      MPSGraphTensor *productF32 =
          [builder.graph multiplicationWithPrimaryTensor:siluF32Product
                                         secondaryTensor:upF32Product
                                                    name:@"silu_times_up_f32"];
      MPSGraphTensor *siluProduct = gs_qwen25_cast_bf16(
          builder.graph, productF32, @"silu_product_bf16");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ siluProduct ], @[ stageArray[12] ],
          @[ ffnShape ], @"silu_up_product", errorText, errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *productFeed = gs_qwen25_builder_add_bf16(
          builder, stageArray[12], ffnShape, @"silu_product_bf16");
      MPSGraphTensor *downWeight = gs_qwen25_builder_add_bf16(
          builder, weightArray[11], downWeightShape,
          @"down_proj_weight_bf16");
      MPSGraphTensor *downProjection = gs_qwen25_linear(
          builder.graph, productFeed, downWeight, nil, (uint32_t)batches,
          (uint32_t)tokens, (uint32_t)ffnDim, (uint32_t)dim, @"down_proj");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ downProjection ], @[ stageArray[13] ],
          @[ hiddenShape ], @"down_proj", errorText, errorCapacity));

      builder = gs_qwen25_new_builder(errorText, errorCapacity);
      MPSGraphTensor *residualForOutput = gs_qwen25_builder_add_bf16(
          builder, stageArray[8], hiddenShape, @"attention_residual_bf16");
      MPSGraphTensor *downForOutput = gs_qwen25_builder_add_bf16(
          builder, stageArray[13], hiddenShape, @"down_projection_bf16");
      MPSGraphTensor *hiddenOut = gs_qwen25_add_bf16(
          builder.graph, residualForOutput, downForOutput, @"hidden_out");
      GS_QWEN25_APPEND_STEP(gs_qwen25_builder_compile(
          builder, device, @[ hiddenOut ], @[ stageArray[14] ],
          @[ hiddenShape ], @"hidden_out", errorText, errorCapacity));
#undef GS_QWEN25_APPEND_STEP

      if (preparedSteps.count != 15) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "internal prepared MPSGraph step count mismatch");
        return nullptr;
      }
      GsQwen25MPSBlockPlan *plan = [GsQwen25MPSBlockPlan new];
      plan.device = device;
      plan.steps = [preparedSteps copy];
      plan.residentBuffers = [allBuffers copy];
      if (!plan.device || !plan.steps || !plan.residentBuffers) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "unable to retain the prepared MPSGraph plan");
        return nullptr;
      }
      return (__bridge_retained void *)plan;
    } @catch (NSException *exception) {
      NSString *message = [NSString stringWithFormat:
          @"Objective-C exception %@: %@", exception.name,
          exception.reason ?: @"(no reason)"];
      gs_qwen25_set_ns_error(errorText, errorCapacity, message);
      return nullptr;
    }
    }
  }
  gs_qwen25_set_error(errorText, errorCapacity,
                      "Qwen2.5-VL BF16 MPSGraph requires macOS 14 or newer");
  return nullptr;
}

extern "C" int32_t gs_qwen25_mps_block_encode(
    void *planHandle, void *commandChainHandle, char *errorText,
    size_t errorCapacity) {
  if (errorText && errorCapacity > 0)
    errorText[0] = '\0';
  if (!planHandle || !commandChainHandle) {
    gs_qwen25_set_error(errorText, errorCapacity,
                        "required plan or MPSCommandChain handle is null");
    return -1;
  }
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      GsQwen25MPSBlockPlan *plan =
          (__bridge GsQwen25MPSBlockPlan *)planHandle;
      MPSCommandBuffer *commandBuffer =
          (__bridge MPSCommandBuffer *)commandChainHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!plan.device || !plan.steps || plan.steps.count != 15 ||
          !plan.residentBuffers || !root ||
          root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted) {
        gs_qwen25_set_error(errorText, errorCapacity,
                            "MPSCommandChain root is unavailable, committed, or on another device");
        return -2;
      }
      @try {
        for (NSUInteger index = 0; index < plan.steps.count; ++index) {
          GsQwen25MPSGraphStep *step = plan.steps[index];
          if (!step.graph || !step.executable || !step.execution ||
              !step.inputs || !step.outputs || step.outputCount == 0 ||
              step.outputs.count != step.outputCount) {
            NSString *message = [NSString stringWithFormat:
                @"prepared MPSGraph step %lu is incomplete",
                (unsigned long)index];
            gs_qwen25_set_ns_error(errorText, errorCapacity, message);
            return -3;
          }
        }
        for (NSUInteger index = 0; index < plan.steps.count; ++index) {
          GsQwen25MPSGraphStep *step = plan.steps[index];
          NSArray<MPSGraphTensorData *> *result =
              [step.executable encodeToCommandBuffer:commandBuffer
                                         inputsArray:step.inputs
                                        resultsArray:step.outputs
                                 executionDescriptor:step.execution];
          if (result.count != step.outputCount) {
            NSString *message = [NSString stringWithFormat:
                @"MPSGraph failed to encode prepared step %@",
                step.name ?: @"(unnamed)"];
            gs_qwen25_set_ns_error(errorText, errorCapacity, message);
            return -4;
          }
        }
        return 0;
      } @catch (NSException *exception) {
        NSString *message = [NSString stringWithFormat:
            @"Objective-C exception %@ during async step encode: %@",
            exception.name, exception.reason ?: @"(no reason)"];
        gs_qwen25_set_ns_error(errorText, errorCapacity, message);
        return -5;
      }
    }
  }
  gs_qwen25_set_error(errorText, errorCapacity,
                      "Qwen2.5-VL BF16 MPSGraph requires macOS 14 or newer");
  return -5;
}

extern "C" void gs_qwen25_mps_block_plan_destroy(void *planHandle) {
  if (planHandle) {
    __strong id retainedPlan = (__bridge_transfer id)planHandle;
    (void)retainedPlan;
  }
}
