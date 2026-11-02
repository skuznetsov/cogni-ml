// Prepared BF16 MPSGraph plans for the released Hy-Nomas DCAE-V encoder.
// Each RMSNorm eager operation writes a BF16 result to caller-owned storage so
// its rounding boundary cannot be elided by fusing a larger graph.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdint.h>
#include <limits.h>
#include <cmath>

@interface DCAEVNormGraphStep : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputs;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputs;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@end

@implementation DCAEVNormGraphStep
@end

@interface DCAEVRMSNormMPSPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<DCAEVNormGraphStep *> *steps;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic, copy) NSArray<NSNumber *> *expectedBufferBytes;
@end

@implementation DCAEVRMSNormMPSPlan
@end

@interface DCAEVSiLUMPSPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) DCAEVNormGraphStep *step;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic) NSUInteger activationBytes;
@end

@implementation DCAEVSiLUMPSPlan
@end

static bool dcae_v_norm_product(const uint64_t *values, size_t count,
                                uint64_t *product) {
  if (!values || !product || count == 0)
    return false;
  uint64_t result = 1;
  for (size_t index = 0; index < count; ++index) {
    if (values[index] == 0 || result > (uint64_t)INT32_MAX / values[index])
      return false;
    result *= values[index];
  }
  *product = result;
  return true;
}

static bool dcae_v_norm_disjoint(id<MTLBuffer> left,
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

static bool dcae_v_norm_buffer_valid(id<MTLBuffer> buffer,
                                     id<MTLDevice> device,
                                     uint64_t expectedBytes,
                                     bool exactLength) {
  return buffer && buffer.device == device &&
         buffer.storageMode == MTLStorageModeShared && buffer.contents &&
         (exactLength ? buffer.length == expectedBytes
                      : buffer.length >= expectedBytes);
}

static bool dcae_v_norm_bindings_disjoint(
    NSArray<id<MTLBuffer>> *buffers) {
  if (!buffers)
    return false;
  for (NSUInteger index = 0; index < buffers.count; ++index) {
    for (NSUInteger previous = 0; previous < index; ++previous) {
      if (!dcae_v_norm_disjoint(buffers[index], buffers[previous]))
        return false;
    }
  }
  return true;
}

static NSArray<NSNumber *> *dcae_v_shape5(const uint64_t shape[5]) {
  return @[ @(shape[0]), @(shape[1]), @(shape[2]), @(shape[3]), @(shape[4]) ];
}

static NSArray<NSNumber *> *dcae_v_shape_bthwc(const uint64_t shape[5]) {
  return @[ @(shape[0]), @(shape[2]), @(shape[3]), @(shape[4]), @(shape[1]) ];
}

static NSArray<NSNumber *> *dcae_v_shape_row(const uint64_t shape[5]) {
  return @[ @(shape[0]), @(shape[2]), @(shape[3]), @(shape[4]), @1 ];
}

// Convert a logical NCTHW tensor to NTHWC with adjacent graph transposes. This
// mirrors RMSNorm3d.forward's `permute(0, 2, 3, 4, 1)` and keeps channel-axis
// reduction and affine broadcasting on the same logical axis as upstream.
static MPSGraphTensor *dcae_v_to_nthwc(MPSGraph *graph,
                                      MPSGraphTensor *tensor,
                                      NSString *prefix) {
  MPSGraphTensor *a = [graph transposeTensor:tensor dimension:1 withDimension:2
                                       name:[prefix stringByAppendingString:@"_t12"]];
  MPSGraphTensor *b = a ? [graph transposeTensor:a dimension:2 withDimension:3
                                           name:[prefix stringByAppendingString:@"_t23"]] : nil;
  return b ? [graph transposeTensor:b dimension:3 withDimension:4
                               name:[prefix stringByAppendingString:@"_t34"]] : nil;
}

static MPSGraphTensor *dcae_v_to_ncthw(MPSGraph *graph,
                                      MPSGraphTensor *tensor,
                                      NSString *prefix) {
  MPSGraphTensor *a = [graph transposeTensor:tensor dimension:3 withDimension:4
                                       name:[prefix stringByAppendingString:@"_t34"]];
  MPSGraphTensor *b = a ? [graph transposeTensor:a dimension:2 withDimension:3
                                           name:[prefix stringByAppendingString:@"_t23"]] : nil;
  return b ? [graph transposeTensor:b dimension:1 withDimension:2
                               name:[prefix stringByAppendingString:@"_t12"]] : nil;
}

static DCAEVNormGraphStep *dcae_v_make_step(
    id<MTLDevice> device, MPSGraph *graph,
    NSArray<MPSGraphTensor *> *placeholders,
    NSArray<NSArray<NSNumber *> *> *inputShapes,
    NSArray<NSNumber *> *inputTypes,
    NSArray<id<MTLBuffer>> *inputBuffers,
    MPSGraphTensor *target, NSArray<NSNumber *> *outputShape,
    MPSDataType outputType, id<MTLBuffer> outputBuffer) {
  if (!device || !graph || placeholders.count == 0 ||
      placeholders.count != inputShapes.count ||
      placeholders.count != inputTypes.count ||
      placeholders.count != inputBuffers.count || !target || !outputBuffer ||
      !target.shape || ![target.shape isEqualToArray:outputShape] ||
      target.dataType != outputType)
    return nil;

  @try {
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
        [NSMutableDictionary new];
    NSMutableArray<MPSGraphTensorData *> *feedData = [NSMutableArray new];
    for (NSUInteger index = 0; index < placeholders.count; ++index) {
      MPSGraphTensor *placeholder = placeholders[index];
      NSArray<NSNumber *> *shape = inputShapes[index];
      MPSDataType type = (MPSDataType)inputTypes[index].intValue;
      feeds[placeholder] = [[MPSGraphShapedType alloc] initWithShape:shape
                                                            dataType:type];
      MPSGraphTensorData *data =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffers[index]
                                                  shape:shape
                                               dataType:type];
      if (!data)
        return nil;
      [feedData addObject:data];
    }

    MPSGraphExecutable *executable =
        [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                            feeds:feeds
                    targetTensors:@[ target ]
                 targetOperations:nil
            compilationDescriptor:nil];
    if (!executable || executable.feedTensors.count != placeholders.count)
      return nil;
    executable.options = MPSGraphOptionsNone;

    NSMutableArray<MPSGraphTensorData *> *orderedInputs = [NSMutableArray new];
    for (MPSGraphTensor *feedTensor in executable.feedTensors) {
      NSUInteger index = [placeholders indexOfObjectIdenticalTo:feedTensor];
      if (index == NSNotFound || index >= feedData.count)
        return nil;
      [orderedInputs addObject:feedData[index]];
    }

    MPSGraphTensorData *outputData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                shape:outputShape
                                             dataType:outputType];
    MPSGraphExecutableExecutionDescriptor *execution =
        [MPSGraphExecutableExecutionDescriptor new];
    execution.waitUntilCompleted = NO;
    if (!outputData || !execution || orderedInputs.count != placeholders.count)
      return nil;

    DCAEVNormGraphStep *step = [DCAEVNormGraphStep new];
    step.graph = graph;
    step.executable = executable;
    step.inputs = orderedInputs;
    step.outputs = @[ outputData ];
    step.execution = execution;
    return step;
  } @catch (NSException *exception) {
    (void)exception;
    return nil;
  }
}

static MPSGraphTensor *dcae_v_placeholder(MPSGraph *graph,
                                         NSArray<NSNumber *> *shape,
                                         MPSDataType type,
                                         NSString *name) {
  return [graph placeholderWithShape:shape dataType:type name:name];
}

static DCAEVNormGraphStep *dcae_v_unary_step(
    id<MTLDevice> device, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> outputBuffer, NSArray<NSNumber *> *shape,
    MPSGraphTensor *(*operation)(MPSGraph *, MPSGraphTensor *, NSString *),
    NSString *name) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  MPSGraphTensor *input = dcae_v_placeholder(graph, shape,
      MPSDataTypeBFloat16, [name stringByAppendingString:@"_input"]);
  MPSGraphTensor *output = input ? operation(graph, input, name) : nil;
  if (!input || !output)
    return nil;
  return dcae_v_make_step(device, graph, @[ input ], @[ shape ],
      @[ @(MPSDataTypeBFloat16) ], @[ inputBuffer ], output, shape,
      MPSDataTypeBFloat16, outputBuffer);
}

static MPSGraphTensor *dcae_v_square_op(MPSGraph *graph, MPSGraphTensor *input,
                                       NSString *name) {
  return [graph squareWithTensor:input name:name];
}

static MPSGraphTensor *dcae_v_sqrt_op(MPSGraph *graph, MPSGraphTensor *input,
                                     NSString *name) {
  return [graph squareRootWithTensor:input name:name];
}

static void *dcae_v_rms_norm_plan_create(
    void *deviceHandle, const int64_t *shapeValues, int32_t rank,
    void *inputHandle, void *fullScratchHandle,
    void *meanAHandle, void *meanBHandle,
    void *weightHandle, void *biasHandle, float epsilon) {
  if (!deviceHandle || !shapeValues || rank != 5 || !inputHandle ||
      !fullScratchHandle || !meanAHandle || !meanBHandle || !weightHandle ||
      !biasHandle || !std::isfinite(epsilon) || epsilon <= 0.0f)
    return nullptr;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t shape[5];
      for (int32_t axis = 0; axis < 5; ++axis) {
        if (shapeValues[axis] <= 0)
          return nullptr;
        shape[axis] = (uint64_t)shapeValues[axis];
      }
      uint64_t elementCount = 0;
      uint64_t rowCount = 0;
      const uint64_t rowDims[4] = {shape[0], shape[2], shape[3], shape[4]};
      if (!dcae_v_norm_product(shape, 5, &elementCount) ||
          !dcae_v_norm_product(rowDims, 4, &rowCount) ||
          elementCount > UINT64_MAX / 2 || rowCount > UINT64_MAX / 2 ||
          shape[1] > UINT64_MAX / 2)
        return nullptr;
      const uint64_t activationBytes = elementCount * 2;
      const uint64_t rowBytes = rowCount * 2;
      const uint64_t channelBytes = shape[1] * 2;

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
      id<MTLBuffer> fullScratch = (__bridge id<MTLBuffer>)fullScratchHandle;
      id<MTLBuffer> meanA = (__bridge id<MTLBuffer>)meanAHandle;
      id<MTLBuffer> meanB = (__bridge id<MTLBuffer>)meanBHandle;
      id<MTLBuffer> weight = (__bridge id<MTLBuffer>)weightHandle;
      id<MTLBuffer> bias = (__bridge id<MTLBuffer>)biasHandle;
      NSArray<id<MTLBuffer>> *buffers = @[
        input, fullScratch, meanA, meanB, weight, bias,
      ];
      const uint64_t expectedBytes[] = {
        activationBytes, activationBytes, rowBytes, rowBytes,
        channelBytes, channelBytes,
      };
      for (NSUInteger index = 0; index < buffers.count; ++index) {
        id<MTLBuffer> buffer = buffers[index];
        const bool exactLength = index >= 4;
        if (!dcae_v_norm_buffer_valid(buffer, device, expectedBytes[index],
                                      exactLength))
          return nullptr;
      }
      if (!dcae_v_norm_bindings_disjoint(buffers))
        return nullptr;

      NSArray<NSNumber *> *ncthw = dcae_v_shape5(shape);
      NSArray<NSNumber *> *nthwc = dcae_v_shape_bthwc(shape);
      NSArray<NSNumber *> *rows = dcae_v_shape_row(shape);
      NSArray<NSNumber *> *channels = @[ @(shape[1]) ];
      NSMutableArray<DCAEVNormGraphStep *> *steps = [NSMutableArray new];

      // torch.square(permuted_x) -> BF16 temporary.
      MPSGraph *squareGraph = [MPSGraph new];
      squareGraph.options = MPSGraphOptionsNone;
      MPSGraphTensor *squareInput = dcae_v_placeholder(squareGraph, ncthw,
          MPSDataTypeBFloat16, @"dcae_rms_square_input_bf16");
      MPSGraphTensor *squareChannelsLast = squareInput
          ? dcae_v_to_nthwc(squareGraph, squareInput, @"dcae_rms_square_input") : nil;
      MPSGraphTensor *squareValue = squareChannelsLast
          ? [squareGraph squareWithTensor:squareChannelsLast
                                     name:@"dcae_rms_square_bf16"] : nil;
      MPSGraphTensor *squareOutput = squareValue
          ? dcae_v_to_ncthw(squareGraph, squareValue, @"dcae_rms_square_output") : nil;
      DCAEVNormGraphStep *squareStep = squareInput && squareChannelsLast &&
          squareValue && squareOutput
          ? dcae_v_make_step(device, squareGraph, @[ squareInput ], @[ ncthw ],
              @[ @(MPSDataTypeBFloat16) ], @[ input ], squareOutput, ncthw,
              MPSDataTypeBFloat16, fullScratch) : nil;
      if (!squareStep)
        return nullptr;
      [steps addObject:squareStep];

      // mean(square_bf16, dim=-1, keepdim=True): PyTorch MPS converts BF16
      // reduction operands to F32 and casts the mean back to BF16.
      MPSGraph *meanGraph = [MPSGraph new];
      meanGraph.options = MPSGraphOptionsNone;
      MPSGraphTensor *meanInput = dcae_v_placeholder(meanGraph, ncthw,
          MPSDataTypeBFloat16, @"dcae_rms_mean_input_bf16");
      MPSGraphTensor *meanF32 = meanInput
          ? [meanGraph castTensor:meanInput toType:MPSDataTypeFloat32
                             name:@"dcae_rms_mean_input_f32"] : nil;
      MPSGraphTensor *meanChannelsLast = meanF32
          ? dcae_v_to_nthwc(meanGraph, meanF32, @"dcae_rms_mean_channels_last") : nil;
      MPSGraphTensor *meanReduced = meanChannelsLast
          ? [meanGraph meanOfTensor:meanChannelsLast axes:@[@4]
                               name:@"dcae_rms_mean_f32_accum"] : nil;
      MPSGraphTensor *meanKeepDims = meanReduced
          ? [meanGraph reshapeTensor:meanReduced withShape:rows
                                name:@"dcae_rms_mean_keepdim"] : nil;
      MPSGraphTensor *meanBf16 = meanKeepDims
          ? [meanGraph castTensor:meanKeepDims toType:MPSDataTypeBFloat16
                             name:@"dcae_rms_mean_output_bf16"] : nil;
      DCAEVNormGraphStep *meanStep = meanInput && meanF32 && meanChannelsLast &&
          meanReduced && meanKeepDims && meanBf16
          ? dcae_v_make_step(device, meanGraph, @[ meanInput ], @[ ncthw ],
              @[ @(MPSDataTypeBFloat16) ], @[ fullScratch ], meanBf16, rows,
              MPSDataTypeBFloat16, meanA) : nil;
      if (!meanStep)
        return nullptr;
      [steps addObject:meanStep];

      // PyTorch wraps the Python float epsilon as F32, promotes the BF16 mean
      // to F32 for this add, then casts the eager result back to BF16.
      MPSGraph *epsilonGraph = [MPSGraph new];
      epsilonGraph.options = MPSGraphOptionsNone;
      MPSGraphTensor *epsilonInput = dcae_v_placeholder(epsilonGraph, rows,
          MPSDataTypeBFloat16, @"dcae_rms_epsilon_input_bf16");
      MPSGraphTensor *epsilonValue = [epsilonGraph constantWithScalar:epsilon
          shape:@[@1] dataType:MPSDataTypeFloat32];
      MPSGraphTensor *epsilonMeanF32 = epsilonInput
          ? [epsilonGraph castTensor:epsilonInput toType:MPSDataTypeFloat32
                                name:@"dcae_rms_epsilon_mean_f32"] : nil;
      MPSGraphTensor *epsilonSumF32 = epsilonMeanF32 && epsilonValue
          ? [epsilonGraph additionWithPrimaryTensor:epsilonMeanF32
                                   secondaryTensor:epsilonValue
                                              name:@"dcae_rms_epsilon_add_f32"] : nil;
      MPSGraphTensor *epsilonOutput = epsilonSumF32
          ? [epsilonGraph castTensor:epsilonSumF32 toType:MPSDataTypeBFloat16
                                name:@"dcae_rms_epsilon_add_bf16"] : nil;
      DCAEVNormGraphStep *epsilonStep = epsilonInput && epsilonValue &&
          epsilonMeanF32 && epsilonSumF32 && epsilonOutput
          ? dcae_v_make_step(device, epsilonGraph, @[ epsilonInput ], @[ rows ],
              @[ @(MPSDataTypeBFloat16) ], @[ meanA ], epsilonOutput, rows,
              MPSDataTypeBFloat16, meanB) : nil;
      if (!epsilonStep)
        return nullptr;
      [steps addObject:epsilonStep];

      // sqrt(epsilon_stabilized_bf16) -> BF16.
      DCAEVNormGraphStep *sqrtStep = dcae_v_unary_step(device, meanB, meanA,
          rows, dcae_v_sqrt_op, @"dcae_rms_sqrt");
      if (!sqrtStep)
        return nullptr;
      [steps addObject:sqrtStep];

      // torch.div(x_nthwc, rms_bf16) -> BF16, restored to NCTHW storage.
      MPSGraph *divideGraph = [MPSGraph new];
      divideGraph.options = MPSGraphOptionsNone;
      MPSGraphTensor *divideInput = dcae_v_placeholder(divideGraph, ncthw,
          MPSDataTypeBFloat16, @"dcae_rms_divide_input_bf16");
      MPSGraphTensor *denominator = dcae_v_placeholder(divideGraph, rows,
          MPSDataTypeBFloat16, @"dcae_rms_divide_denominator_bf16");
      MPSGraphTensor *divideChannelsLast = divideInput
          ? dcae_v_to_nthwc(divideGraph, divideInput, @"dcae_rms_divide_input") : nil;
      MPSGraphTensor *divideValue = divideChannelsLast && denominator
          ? [divideGraph divisionWithPrimaryTensor:divideChannelsLast
                                  secondaryTensor:denominator
                                             name:@"dcae_rms_divide_bf16"] : nil;
      MPSGraphTensor *divideOutput = divideValue
          ? dcae_v_to_ncthw(divideGraph, divideValue, @"dcae_rms_divide_output") : nil;
      DCAEVNormGraphStep *divideStep = divideInput && denominator &&
          divideChannelsLast && divideValue && divideOutput
          ? dcae_v_make_step(device, divideGraph, @[ divideInput, denominator ],
              @[ ncthw, rows ], @[ @(MPSDataTypeBFloat16), @(MPSDataTypeBFloat16) ],
              @[ input, meanA ], divideOutput, ncthw, MPSDataTypeBFloat16,
              fullScratch) : nil;
      if (!divideStep)
        return nullptr;
      [steps addObject:divideStep];

      // x_bf16 * weight_bf16 -> BF16. The original `input` buffer is now dead
      // and becomes this eager operation's output to avoid another full tile.
      MPSGraph *gainGraph = [MPSGraph new];
      gainGraph.options = MPSGraphOptionsNone;
      MPSGraphTensor *gainInput = dcae_v_placeholder(gainGraph, ncthw,
          MPSDataTypeBFloat16, @"dcae_rms_gain_input_bf16");
      MPSGraphTensor *gainWeight = dcae_v_placeholder(gainGraph, channels,
          MPSDataTypeBFloat16, @"dcae_rms_gain_weight_bf16");
      MPSGraphTensor *gainChannelsLast = gainInput
          ? dcae_v_to_nthwc(gainGraph, gainInput, @"dcae_rms_gain_input") : nil;
      MPSGraphTensor *weightChannelsLast = gainWeight
          ? [gainGraph reshapeTensor:gainWeight
                            withShape:@[@1, @1, @1, @1, @(shape[1])]
                                 name:@"dcae_rms_gain_weight_nthwc"] : nil;
      MPSGraphTensor *gainValue = gainChannelsLast && weightChannelsLast
          ? [gainGraph multiplicationWithPrimaryTensor:gainChannelsLast
                                      secondaryTensor:weightChannelsLast
                                                 name:@"dcae_rms_gain_bf16"] : nil;
      MPSGraphTensor *gainOutput = gainValue
          ? dcae_v_to_ncthw(gainGraph, gainValue, @"dcae_rms_gain_output") : nil;
      DCAEVNormGraphStep *gainStep = gainInput && gainWeight && gainChannelsLast &&
          weightChannelsLast && gainValue && gainOutput
          ? dcae_v_make_step(device, gainGraph, @[ gainInput, gainWeight ],
              @[ ncthw, channels ], @[ @(MPSDataTypeBFloat16), @(MPSDataTypeBFloat16) ],
              @[ fullScratch, weight ], gainOutput, ncthw, MPSDataTypeBFloat16,
              input) : nil;
      if (!gainStep)
        return nullptr;
      [steps addObject:gainStep];

      // x_bf16 + bias_bf16 -> BF16. This is the final RMSNorm tensor consumed
      // by the residual add; the caller's shortcut remains unchanged.
      MPSGraph *biasGraph = [MPSGraph new];
      biasGraph.options = MPSGraphOptionsNone;
      MPSGraphTensor *biasInput = dcae_v_placeholder(biasGraph, ncthw,
          MPSDataTypeBFloat16, @"dcae_rms_bias_input_bf16");
      MPSGraphTensor *biasValue = dcae_v_placeholder(biasGraph, channels,
          MPSDataTypeBFloat16, @"dcae_rms_bias_bf16");
      MPSGraphTensor *biasChannelsLast = biasInput
          ? dcae_v_to_nthwc(biasGraph, biasInput, @"dcae_rms_bias_input") : nil;
      MPSGraphTensor *biasChannels = biasValue
          ? [biasGraph reshapeTensor:biasValue
                            withShape:@[@1, @1, @1, @1, @(shape[1])]
                                 name:@"dcae_rms_bias_nthwc"] : nil;
      MPSGraphTensor *biased = biasChannelsLast && biasChannels
          ? [biasGraph additionWithPrimaryTensor:biasChannelsLast
                                secondaryTensor:biasChannels
                                           name:@"dcae_rms_bias_add_bf16"] : nil;
      MPSGraphTensor *biasOutput = biased
          ? dcae_v_to_ncthw(biasGraph, biased, @"dcae_rms_bias_output") : nil;
      DCAEVNormGraphStep *biasStep = biasInput && biasValue && biasChannelsLast &&
          biasChannels && biased && biasOutput
          ? dcae_v_make_step(device, biasGraph, @[ biasInput, biasValue ],
              @[ ncthw, channels ], @[ @(MPSDataTypeBFloat16), @(MPSDataTypeBFloat16) ],
              @[ input, bias ], biasOutput, ncthw, MPSDataTypeBFloat16,
              fullScratch) : nil;
      if (!biasStep)
        return nullptr;
      [steps addObject:biasStep];

      DCAEVRMSNormMPSPlan *plan = [DCAEVRMSNormMPSPlan new];
      plan.device = device;
      plan.steps = steps;
      plan.residentBuffers = buffers;
      plan.expectedBufferBytes = @[
        @(activationBytes), @(activationBytes), @(rowBytes), @(rowBytes),
        @(channelBytes), @(channelBytes),
      ];
      return (__bridge_retained void *)plan;
    }
  }
  return nullptr;
}

static void *dcae_v_silu_plan_create(
    void *deviceHandle, const int64_t *shapeValues, int32_t rank,
    void *inputHandle, void *outputHandle) {
  if (!deviceHandle || !shapeValues || rank != 5 || !inputHandle || !outputHandle)
    return nullptr;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t shape[5];
      for (int32_t axis = 0; axis < 5; ++axis) {
        if (shapeValues[axis] <= 0)
          return nullptr;
        shape[axis] = (uint64_t)shapeValues[axis];
      }
      uint64_t elementCount = 0;
      if (!dcae_v_norm_product(shape, 5, &elementCount) ||
          elementCount > UINT64_MAX / 2)
        return nullptr;
      const uint64_t bytes = elementCount * 2;
      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
      if (!dcae_v_norm_buffer_valid(input, device, bytes, false) ||
          !dcae_v_norm_buffer_valid(output, device, bytes, false) ||
          !dcae_v_norm_disjoint(input, output))
        return nullptr;

      NSArray<NSNumber *> *ncthw = dcae_v_shape5(shape);
      NSArray<NSNumber *> *nthwc = dcae_v_shape_bthwc(shape);
      MPSGraph *graph = [MPSGraph new];
      graph.options = MPSGraphOptionsNone;
      MPSGraphTensor *inputTensor = dcae_v_placeholder(graph, ncthw,
          MPSDataTypeBFloat16, @"dcae_silu_input_bf16");
      MPSGraphTensor *inputNTHWC = inputTensor
          ? dcae_v_to_nthwc(graph, inputTensor, @"dcae_silu_input") : nil;
      MPSGraphTensor *negative = inputNTHWC
          ? [graph negativeWithTensor:inputNTHWC name:@"dcae_silu_negative_bf16"] : nil;
      MPSGraphTensor *exponential = negative
          ? [graph exponentWithTensor:negative name:@"dcae_silu_exp_negative_bf16"] : nil;
      MPSGraphTensor *one = [graph constantWithScalar:1.0
          shape:@[@1] dataType:MPSDataTypeBFloat16];
      MPSGraphTensor *denominator = exponential && one
          ? [graph additionWithPrimaryTensor:exponential secondaryTensor:one
                                       name:@"dcae_silu_one_plus_exp_bf16"] : nil;
      MPSGraphTensor *activation = inputNTHWC && denominator
          ? [graph divisionWithPrimaryTensor:inputNTHWC secondaryTensor:denominator
                                       name:@"dcae_silu_x_over_one_plus_exp_bf16"] : nil;
      MPSGraphTensor *outputTensor = activation
          ? dcae_v_to_ncthw(graph, activation, @"dcae_silu_output") : nil;
      DCAEVNormGraphStep *step = inputTensor && inputNTHWC && negative &&
          exponential && one && denominator && activation && outputTensor
          ? dcae_v_make_step(device, graph, @[ inputTensor ], @[ ncthw ],
              @[ @(MPSDataTypeBFloat16) ], @[ input ], outputTensor, ncthw,
              MPSDataTypeBFloat16, output) : nil;
      if (!step)
        return nullptr;

      DCAEVSiLUMPSPlan *plan = [DCAEVSiLUMPSPlan new];
      plan.device = device;
      plan.step = step;
      plan.residentBuffers = @[ input, output ];
      plan.activationBytes = bytes;
      return (__bridge_retained void *)plan;
    }
  }
  return nullptr;
}

extern "C" void *gs_dcae_v_rms_norm_mps_plan_create(
    void *device, const int64_t *shape, int32_t rank, void *input,
    void *fullScratch, void *meanA, void *meanB, void *weight, void *bias,
    float epsilon) {
  return dcae_v_rms_norm_plan_create(device, shape, rank, input, fullScratch,
                                    meanA, meanB, weight, bias, epsilon);
}

extern "C" int32_t gs_dcae_v_rms_norm_mps_encode(void *planHandle,
                                                  void *commandHandle) {
  if (!planHandle || !commandHandle)
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      DCAEVRMSNormMPSPlan *plan =
          (__bridge DCAEVRMSNormMPSPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted ||
          plan.residentBuffers.count != 6 || plan.expectedBufferBytes.count != 6 ||
          plan.steps.count != 7)
        return -2;
      for (NSUInteger index = 0; index < plan.residentBuffers.count; ++index) {
        const bool exactLength = index >= 4;
        if (!dcae_v_norm_buffer_valid(plan.residentBuffers[index], plan.device,
                plan.expectedBufferBytes[index].unsignedLongLongValue,
                exactLength))
          return -7;
      }
      if (!dcae_v_norm_bindings_disjoint(plan.residentBuffers))
        return -8;
      @try {
        for (DCAEVNormGraphStep *step in plan.steps) {
          if (!step.executable || !step.execution || !step.inputs ||
              !step.outputs || step.outputs.count != 1)
            return -3;
          NSArray<MPSGraphTensorData *> *result =
              [step.executable encodeToCommandBuffer:commandBuffer
                                         inputsArray:step.inputs
                                        resultsArray:step.outputs
                                 executionDescriptor:step.execution];
          if (!result || result.count != 1)
            return -4;
          root = commandBuffer.rootCommandBuffer;
          if (!root || root.commandQueue.device != plan.device ||
              root.status >= MTLCommandBufferStatusCommitted)
            return -5;
        }
        return 0;
      } @catch (NSException *exception) {
        (void)exception;
        return -6;
      }
    }
  }
  return -7;
}

extern "C" void gs_dcae_v_rms_norm_mps_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}

extern "C" void *gs_dcae_v_silu_mps_plan_create(
    void *device, const int64_t *shape, int32_t rank, void *input,
    void *output) {
  return dcae_v_silu_plan_create(device, shape, rank, input, output);
}

extern "C" int32_t gs_dcae_v_silu_mps_encode(void *planHandle,
                                               void *commandHandle) {
  if (!planHandle || !commandHandle)
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      DCAEVSiLUMPSPlan *plan = (__bridge DCAEVSiLUMPSPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted ||
          plan.residentBuffers.count != 2 || !plan.step.executable ||
          !plan.step.execution)
        return -2;
      if (!dcae_v_norm_buffer_valid(plan.residentBuffers[0], plan.device,
                                    plan.activationBytes, false) ||
          !dcae_v_norm_buffer_valid(plan.residentBuffers[1], plan.device,
                                    plan.activationBytes, false) ||
          !dcae_v_norm_bindings_disjoint(plan.residentBuffers))
        return -6;
      @try {
        NSArray<MPSGraphTensorData *> *result =
            [plan.step.executable encodeToCommandBuffer:commandBuffer
                                           inputsArray:plan.step.inputs
                                          resultsArray:plan.step.outputs
                                   executionDescriptor:plan.step.execution];
        if (!result || result.count != 1)
          return -3;
        root = commandBuffer.rootCommandBuffer;
        return root && root.commandQueue.device == plan.device &&
                       root.status < MTLCommandBufferStatusCommitted
                   ? 0 : -4;
      } @catch (NSException *exception) {
        (void)exception;
        return -5;
      }
    }
  }
  return -6;
}

extern "C" void gs_dcae_v_silu_mps_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}
