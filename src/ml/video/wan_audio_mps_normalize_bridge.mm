// Source-shaped MPSGraph waveform normalization for the pinned XLSR MPS path.
// The plan owns graph metadata and retains caller-owned resident buffers; encode
// only appends work to the caller's MPS command buffer.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdint.h>
#include <limits.h>

@interface CogniWanAudioMPSNormalizePlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLBuffer> input;
@property(nonatomic, strong) id<MTLBuffer> output;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphTensor *inputTensor;
@property(nonatomic, strong) MPSGraphTensor *outputTensor;
@property(nonatomic, strong) MPSGraphTensorData *inputData;
@property(nonatomic, strong) MPSGraphTensorData *outputData;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic) NSUInteger inputSamples;
@end

@implementation CogniWanAudioMPSNormalizePlan
@end

static bool wan_audio_normalize_buffer_bytes(int32_t elements, uint64_t *out) {
  if (elements <= 0 || (uint64_t)elements > (uint64_t)INT32_MAX / sizeof(float))
    return false;
  *out = (uint64_t)elements * sizeof(float);
  return true;
}

static bool wan_audio_normalize_disjoint(id<MTLBuffer> a, id<MTLBuffer> b) {
  if (a == b || a.gpuAddress == 0 || b.gpuAddress == 0) return false;
  const uint64_t a0 = a.gpuAddress, b0 = b.gpuAddress;
  const uint64_t a1 = a0 + a.length, b1 = b0 + b.length;
  return a1 >= a0 && b1 >= b0 && (a1 <= b0 || b1 <= a0);
}

extern "C" void *gs_wan_audio_mps_normalize_plan_create(
    void *device_handle, void *input_handle, void *output_handle,
    int32_t input_samples) {
  if (!device_handle || !input_handle || !output_handle || input_samples <= 0)
    return nullptr;

  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t bytes = 0;
      if (!wan_audio_normalize_buffer_bytes(input_samples, &bytes)) return nullptr;
      id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)input_handle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_handle;
      if (input.device != device || output.device != device ||
          input.length < bytes || output.length < bytes ||
          !wan_audio_normalize_disjoint(input, output)) return nullptr;

      @try {
        CogniWanAudioMPSNormalizePlan *plan =
            [CogniWanAudioMPSNormalizePlan new];
        plan.device = device;
        plan.input = input;
        plan.output = output;
        plan.inputSamples = (NSUInteger)input_samples;

        // Match PyTorch 2.8.0 MPS reduction/operator composition:
        // ReduceOps.mm uses meanOfTensor and varianceOfTensor directly for
        // mean/var(unbiased=False); the Python normalizer then adds 1e-7,
        // applies reciprocalSquareRoot, subtracts the mean, and multiplies.
        // Keep these as distinct graph operations in that order.
        plan.graph = [[MPSGraph alloc] init];
        plan.graph.options = MPSGraphOptionsNone;
        MPSShape *shape = @[@(input_samples)];
        plan.inputTensor = [plan.graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"waveform"];
        NSArray<NSNumber *> *axes = @[@0];
        MPSGraphTensor *mean = [plan.graph meanOfTensor:plan.inputTensor
                                                   axes:axes
                                                   name:@"waveform_mean"];
        MPSGraphTensor *variance = [plan.graph varianceOfTensor:plan.inputTensor
                                                           axes:axes
                                                           name:@"waveform_population_variance"];
        MPSGraphTensor *epsilon = [plan.graph constantWithScalar:1.0e-7
                                                            shape:@[@1]
                                                         dataType:MPSDataTypeFloat32];
        MPSGraphTensor *varianceEpsilon = [plan.graph
            additionWithPrimaryTensor:variance secondaryTensor:epsilon
            name:@"variance_plus_epsilon"];
        MPSGraphTensor *inverseStd = [plan.graph
            reciprocalSquareRootWithTensor:varianceEpsilon
                                       name:@"inverse_standard_deviation"];
        MPSGraphTensor *centered = [plan.graph
            subtractionWithPrimaryTensor:plan.inputTensor secondaryTensor:mean
            name:@"centered_waveform"];
        plan.outputTensor = [plan.graph
            multiplicationWithPrimaryTensor:centered secondaryTensor:inverseStd
            name:@"normalized_waveform"];

        MPSGraphShapedType *inputType = [[MPSGraphShapedType alloc]
            initWithShape:shape dataType:MPSDataTypeFloat32];
        if (!inputType || !plan.outputTensor) return nullptr;
        MPSGraphTensorShapedTypeDictionary *feedTypes = @{
          plan.inputTensor : inputType,
        };
        plan.executable = [plan.graph
            compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                        feeds:feedTypes
                targetTensors:@[plan.outputTensor]
             targetOperations:nil
        compilationDescriptor:nil];
        if (!plan.executable) return nullptr;
        plan.executable.options = MPSGraphOptionsNone;

        plan.inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input shape:shape dataType:MPSDataTypeFloat32];
        plan.outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:output shape:shape dataType:MPSDataTypeFloat32];
        if (!plan.inputData || !plan.outputData ||
            plan.executable.feedTensors.count != 1 ||
            plan.executable.targetTensors.count != 1 ||
            plan.executable.targetTensors.firstObject != plan.outputTensor)
          return nullptr;

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedData = @{
          plan.inputTensor : plan.inputData,
        };
        NSMutableArray<MPSGraphTensorData *> *orderedInputs =
            [NSMutableArray arrayWithCapacity:1];
        for (MPSGraphTensor *feedTensor in plan.executable.feedTensors) {
          MPSGraphTensorData *feedValue = feedData[feedTensor];
          if (!feedValue) return nullptr;
          [orderedInputs addObject:feedValue];
        }
        plan.inputArray = [orderedInputs copy];
        plan.outputArray = @[plan.outputData];
        plan.execution = [MPSGraphExecutableExecutionDescriptor new];
        plan.execution.waitUntilCompleted = NO;
        if (!plan.inputArray || !plan.outputArray || !plan.execution)
          return nullptr;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_wan_audio_mps_normalize_encode(
    void *plan_handle, void *command_handle) {
  if (!plan_handle || !command_handle) return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      CogniWanAudioMPSNormalizePlan *plan =
          (__bridge CogniWanAudioMPSNormalizePlan *)plan_handle;
      MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)command_handle;
      id<MTLCommandBuffer> root = command.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted) return -2;
      @try {
        NSArray<MPSGraphTensorData *> *results =
            [plan.executable encodeToCommandBuffer:command
                                       inputsArray:plan.inputArray
                                      resultsArray:plan.outputArray
                               executionDescriptor:plan.execution];
        if (results.count != 1) return -4;
      } @catch (NSException *exception) {
        (void)exception;
        return -5;
      }
      return 0;
    }
  }
  return -6;
}

extern "C" void gs_wan_audio_mps_normalize_plan_release(void *plan_handle) {
  if (plan_handle) {
    id plan = (__bridge_transfer id)plan_handle;
    (void)plan;
  }
}

// Boundary-preserving normalization candidate. PyTorch evaluates the pinned
// Python expression as separate eager tensor operations:
//   (raw - raw.mean()) * rsqrt(raw.var(unbiased=False) + 1e-7)
// Each executable below materializes one source operation to a caller-owned
// MTLBuffer before the next executable consumes it. This prevents MPSGraph
// from fusing arithmetic across source-operation boundaries. All stages append
// to the caller's existing command buffer; the caller owns the sole terminal
// wait and must keep the buffers/plan alive through it.
typedef NS_ENUM(NSUInteger, WanAudioBoundaryOp) {
  WanAudioBoundaryMean,
  WanAudioBoundarySubtract,
  WanAudioBoundaryVariance,
  WanAudioBoundaryAddEpsilon,
  WanAudioBoundaryRsqrt,
  WanAudioBoundaryMultiply,
};

@interface CogniWanAudioBoundaryStage : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensor *> *inputTensors;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputData;
@property(nonatomic, strong) MPSGraphTensorData *outputData;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@end

@implementation CogniWanAudioBoundaryStage
@end

@interface CogniWanAudioSourceBoundaryNormalizePlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *buffers;
@property(nonatomic, strong) NSArray<CogniWanAudioBoundaryStage *> *stages;
@property(nonatomic) NSUInteger inputSamples;
@end

@implementation CogniWanAudioSourceBoundaryNormalizePlan
@end

static CogniWanAudioBoundaryStage *wan_audio_boundary_make_stage(
    id<MTLDevice> device, WanAudioBoundaryOp operation,
    NSArray<id<MTLBuffer>> *inputBuffers,
    NSArray<MPSShape *> *inputShapes, id<MTLBuffer> outputBuffer,
    MPSShape *outputShape) {
  if (!device || !inputBuffers || inputBuffers.count == 0 ||
      inputBuffers.count != inputShapes.count || !outputBuffer ||
      !outputShape)
    return nil;

  CogniWanAudioBoundaryStage *stage = [CogniWanAudioBoundaryStage new];
  stage.graph = [MPSGraph new];
  stage.graph.options = MPSGraphOptionsNone;
  NSMutableArray<MPSGraphTensor *> *inputs =
      [NSMutableArray arrayWithCapacity:inputBuffers.count];
  NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feedTypes =
      [NSMutableDictionary dictionaryWithCapacity:inputBuffers.count];
  NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedData =
      [NSMutableDictionary dictionaryWithCapacity:inputBuffers.count];
  NSMutableArray<MPSGraphTensorData *> *retainedData =
      [NSMutableArray arrayWithCapacity:inputBuffers.count];

  for (NSUInteger i = 0; i < inputBuffers.count; ++i) {
    MPSShape *shape = inputShapes[i];
    id<MTLBuffer> buffer = inputBuffers[i];
    if (!shape || !buffer || buffer.device != device) return nil;
    MPSGraphTensor *tensor = [stage.graph placeholderWithShape:shape
                                                      dataType:MPSDataTypeFloat32
                                                          name:nil];
    MPSGraphShapedType *shapedType = [[MPSGraphShapedType alloc]
        initWithShape:shape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:buffer shape:shape dataType:MPSDataTypeFloat32];
    if (!tensor || !shapedType || !data) return nil;
    [inputs addObject:tensor];
    feedTypes[tensor] = shapedType;
    feedData[tensor] = data;
    [retainedData addObject:data];
  }

  MPSGraphTensor *output = nil;
  switch (operation) {
    case WanAudioBoundaryMean: {
      MPSGraphTensor *reduced = [stage.graph meanOfTensor:inputs[0]
                                                    axes:@[@0]
                                                    name:nil];
      output = [stage.graph reshapeTensor:reduced withShape:outputShape name:nil];
      break;
    }
    case WanAudioBoundarySubtract:
      output = [stage.graph subtractionWithPrimaryTensor:inputs[0]
                                         secondaryTensor:inputs[1]
                                                    name:nil];
      break;
    case WanAudioBoundaryVariance: {
      MPSGraphTensor *reduced = [stage.graph varianceOfTensor:inputs[0]
                                                         axes:@[@0]
                                                         name:nil];
      output = [stage.graph reshapeTensor:reduced withShape:outputShape name:nil];
      break;
    }
    case WanAudioBoundaryAddEpsilon: {
      MPSGraphTensor *epsilon = [stage.graph constantWithScalar:1.0e-7
                                                          shape:outputShape
                                                       dataType:MPSDataTypeFloat32];
      output = [stage.graph additionWithPrimaryTensor:inputs[0]
                                      secondaryTensor:epsilon
                                                 name:nil];
      break;
    }
    case WanAudioBoundaryRsqrt:
      output = [stage.graph reciprocalSquareRootWithTensor:inputs[0] name:nil];
      break;
    case WanAudioBoundaryMultiply:
      output = [stage.graph multiplicationWithPrimaryTensor:inputs[0]
                                           secondaryTensor:inputs[1]
                                                      name:nil];
      break;
  }
  if (!output) return nil;
  if (output.shape.count != outputShape.count) return nil;
  for (NSUInteger i = 0; i < outputShape.count; ++i) {
    if ([output.shape[i] unsignedIntegerValue] !=
        [outputShape[i] unsignedIntegerValue])
      return nil;
  }
  stage.executable = [stage.graph
      compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                  feeds:feedTypes
          targetTensors:@[output]
       targetOperations:nil
  compilationDescriptor:nil];
  if (!stage.executable) return nil;
  stage.executable.options = MPSGraphOptionsNone;
  stage.outputData = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:outputBuffer
                  shape:outputShape
               dataType:MPSDataTypeFloat32];
  if (!stage.outputData || stage.executable.feedTensors.count != inputs.count ||
      stage.executable.targetTensors.count != 1 ||
      stage.executable.targetTensors.firstObject != output)
    return nil;

  NSMutableArray<MPSGraphTensorData *> *orderedInputs =
      [NSMutableArray arrayWithCapacity:inputs.count];
  NSMutableArray<MPSGraphTensor *> *orderedTensors =
      [NSMutableArray arrayWithCapacity:inputs.count];
  for (MPSGraphTensor *tensor in stage.executable.feedTensors) {
    MPSGraphTensorData *data = feedData[tensor];
    if (!data) return nil;
    [orderedInputs addObject:data];
    [orderedTensors addObject:tensor];
  }
  stage.inputTensors = [orderedTensors copy];
  stage.inputData = [retainedData copy];
  stage.inputArray = [orderedInputs copy];
  stage.outputArray = @[stage.outputData];
  stage.execution = [MPSGraphExecutableExecutionDescriptor new];
  stage.execution.waitUntilCompleted = NO;
  return stage.inputArray && stage.outputArray && stage.execution ? stage : nil;
}

static bool wan_audio_boundary_buffer_disjoint(
    NSArray<id<MTLBuffer>> *buffers) {
  for (NSUInteger i = 0; i < buffers.count; ++i) {
    for (NSUInteger j = i + 1; j < buffers.count; ++j) {
      if (!wan_audio_normalize_disjoint(buffers[i], buffers[j])) return false;
    }
  }
  return true;
}

extern "C" void *gs_wan_audio_mps_source_boundary_normalize_plan_create(
    void *device_handle, void *input_handle, void *output_handle,
    void *scalar_a_handle, void *scalar_b_handle, void *centered_handle,
    int32_t input_samples) {
  if (!device_handle || !input_handle || !output_handle || !scalar_a_handle ||
      !scalar_b_handle || !centered_handle || input_samples <= 0)
    return nullptr;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t bytes = 0;
      if (!wan_audio_normalize_buffer_bytes(input_samples, &bytes)) return nullptr;
      id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)input_handle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_handle;
      id<MTLBuffer> scalarA = (__bridge id<MTLBuffer>)scalar_a_handle;
      id<MTLBuffer> scalarB = (__bridge id<MTLBuffer>)scalar_b_handle;
      id<MTLBuffer> centered = (__bridge id<MTLBuffer>)centered_handle;
      NSArray<id<MTLBuffer>> *buffers =
          @[input, output, scalarA, scalarB, centered];
      for (id<MTLBuffer> buffer in buffers) {
        if (!buffer || buffer.device != device) return nullptr;
      }
      if (input.length < bytes || output.length < bytes ||
          centered.length < bytes || scalarA.length < sizeof(float) ||
          scalarB.length < sizeof(float) ||
          !wan_audio_boundary_buffer_disjoint(buffers))
        return nullptr;

      @try {
        CogniWanAudioSourceBoundaryNormalizePlan *plan =
            [CogniWanAudioSourceBoundaryNormalizePlan new];
        plan.device = device;
        plan.buffers = buffers;
        plan.inputSamples = (NSUInteger)input_samples;
        MPSShape *waveShape = @[@(input_samples)];
        MPSShape *scalarShape = @[@1];
        NSArray<MPSShape *> *waveInputShape = @[waveShape];
        NSArray<MPSShape *> *scalarInputShape = @[scalarShape];
        NSArray<MPSShape *> *subInputShapes = @[waveShape, scalarShape];
        NSArray<MPSShape *> *mulInputShapes = @[waveShape, scalarShape];

        // Eager PyTorch source order, with every returned value materialized:
        // mean -> subtract; variance -> add epsilon -> rsqrt; multiply.
        // scalarA is reusable after centered[] has consumed the mean; scalarB
        // then separates the add and rsqrt outputs. Peak extra storage is
        // centered[N] plus two scalars (4*N + 8 bytes), all caller-accounted.
        NSArray<CogniWanAudioBoundaryStage *> *stages = @[
          wan_audio_boundary_make_stage(device, WanAudioBoundaryMean,
                                        @[input], waveInputShape, scalarA,
                                        scalarShape),
          wan_audio_boundary_make_stage(device, WanAudioBoundarySubtract,
                                        @[input, scalarA], subInputShapes,
                                        centered, waveShape),
          wan_audio_boundary_make_stage(device, WanAudioBoundaryVariance,
                                        @[input], waveInputShape, scalarA,
                                        scalarShape),
          wan_audio_boundary_make_stage(device, WanAudioBoundaryAddEpsilon,
                                        @[scalarA], scalarInputShape, scalarB,
                                        scalarShape),
          wan_audio_boundary_make_stage(device, WanAudioBoundaryRsqrt,
                                        @[scalarB], scalarInputShape, scalarA,
                                        scalarShape),
          wan_audio_boundary_make_stage(device, WanAudioBoundaryMultiply,
                                        @[centered, scalarA], mulInputShapes,
                                        output, waveShape),
        ];
        for (CogniWanAudioBoundaryStage *stage in stages) {
          if (!stage) return nullptr;
        }
        plan.stages = stages;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_wan_audio_mps_source_boundary_normalize_encode(
    void *plan_handle, void *command_handle) {
  if (!plan_handle || !command_handle) return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      CogniWanAudioSourceBoundaryNormalizePlan *plan =
          (__bridge CogniWanAudioSourceBoundaryNormalizePlan *)plan_handle;
      MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)command_handle;
      id<MTLCommandBuffer> root = command.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted)
        return -2;
      for (NSUInteger i = 0; i < plan.stages.count; ++i) {
        CogniWanAudioBoundaryStage *stage = plan.stages[i];
        @try {
          NSArray<MPSGraphTensorData *> *results =
              [stage.executable encodeToCommandBuffer:command
                                          inputsArray:stage.inputArray
                                         resultsArray:stage.outputArray
                                  executionDescriptor:stage.execution];
          if (results.count != 1) return -10 - (int32_t)i;
        } @catch (NSException *exception) {
          (void)exception;
          // Earlier stages may already be enqueued. The caller must drain the
          // chain before releasing any plan or scratch on every failure path.
          return -20 - (int32_t)i;
        }
      }
      return 0;
    }
  }
  return -6;
}

extern "C" void gs_wan_audio_mps_source_boundary_normalize_plan_release(
    void *plan_handle) {
  if (plan_handle) {
    id plan = (__bridge_transfer id)plan_handle;
    (void)plan;
  }
}

// Source-shaped exact GELU for the pinned PyTorch MPS implementation. This
// graph deliberately keeps the erf, add, multiply, and final multiply as
// separate MPSGraph operators, and binds only caller-owned resident buffers.
@interface CogniWanAudioMPSGeluPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLBuffer> input;
@property(nonatomic, strong) id<MTLBuffer> output;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphTensor *inputTensor;
@property(nonatomic, strong) MPSGraphTensor *outputTensor;
@property(nonatomic, strong) MPSGraphTensorData *inputData;
@property(nonatomic, strong) MPSGraphTensorData *outputData;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic) NSUInteger elementCount;
@end

@implementation CogniWanAudioMPSGeluPlan
@end

extern "C" void *gs_wan_audio_mps_gelu_plan_create(
    void *device_handle, void *input_handle, void *output_handle,
    int32_t elements) {
  if (!device_handle || !input_handle || !output_handle || elements <= 0)
    return nullptr;

  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t bytes = 0;
      if (!wan_audio_normalize_buffer_bytes(elements, &bytes)) return nullptr;
      id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)input_handle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_handle;
      if (input.device != device || output.device != device ||
          input.length < bytes || output.length < bytes ||
          !wan_audio_normalize_disjoint(input, output)) return nullptr;

      @try {
        CogniWanAudioMPSGeluPlan *plan = [CogniWanAudioMPSGeluPlan new];
        plan.device = device;
        plan.input = input;
        plan.output = output;
        plan.elementCount = (NSUInteger)elements;
        plan.graph = [[MPSGraph alloc] init];
        plan.graph.options = MPSGraphOptionsNone;

        MPSShape *shape = @[@(elements)];
        plan.inputTensor = [plan.graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"gelu_input"];
        MPSGraphTensor *sqrtInverseTwo = [plan.graph constantWithScalar:
            0.707106781186547524400844362104849039
            shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor *one = [plan.graph constantWithScalar:1.0
                                                       shape:@[@1]
                                                    dataType:MPSDataTypeFloat32];
        MPSGraphTensor *half = [plan.graph constantWithScalar:0.5
                                                        shape:@[@1]
                                                     dataType:MPSDataTypeFloat32];
        MPSGraphTensor *scaled = [plan.graph
            multiplicationWithPrimaryTensor:plan.inputTensor
                           secondaryTensor:sqrtInverseTwo
                                      name:@"gelu_scale"];
        MPSGraphTensor *erf = [plan.graph erfWithTensor:scaled name:@"gelu_erf"];
        MPSGraphTensor *shifted = [plan.graph
            additionWithPrimaryTensor:erf secondaryTensor:one name:@"gelu_add_one"];
        MPSGraphTensor *cdf = [plan.graph
            multiplicationWithPrimaryTensor:shifted secondaryTensor:half
                                      name:@"gelu_half_cdf"];
        plan.outputTensor = [plan.graph
            multiplicationWithPrimaryTensor:plan.inputTensor secondaryTensor:cdf
                                      name:@"gelu_output"];

        MPSGraphShapedType *inputType = [[MPSGraphShapedType alloc]
            initWithShape:shape dataType:MPSDataTypeFloat32];
        if (!inputType || !plan.outputTensor) return nullptr;
        MPSGraphTensorShapedTypeDictionary *feedTypes = @{
          plan.inputTensor : inputType,
        };
        plan.executable = [plan.graph
            compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                        feeds:feedTypes
                targetTensors:@[plan.outputTensor]
             targetOperations:nil
        compilationDescriptor:nil];
        if (!plan.executable) return nullptr;
        plan.executable.options = MPSGraphOptionsNone;
        plan.inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input shape:shape dataType:MPSDataTypeFloat32];
        plan.outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:output shape:shape dataType:MPSDataTypeFloat32];
        if (!plan.inputData || !plan.outputData ||
            plan.executable.feedTensors.count != 1 ||
            plan.executable.targetTensors.count != 1 ||
            plan.executable.targetTensors.firstObject != plan.outputTensor)
          return nullptr;

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedData = @{
          plan.inputTensor : plan.inputData,
        };
        NSMutableArray<MPSGraphTensorData *> *orderedInputs =
            [NSMutableArray arrayWithCapacity:1];
        for (MPSGraphTensor *feedTensor in plan.executable.feedTensors) {
          MPSGraphTensorData *feedValue = feedData[feedTensor];
          if (!feedValue) return nullptr;
          [orderedInputs addObject:feedValue];
        }
        plan.inputArray = [orderedInputs copy];
        plan.outputArray = @[plan.outputData];
        plan.execution = [MPSGraphExecutableExecutionDescriptor new];
        plan.execution.waitUntilCompleted = NO;
        if (!plan.inputArray || !plan.outputArray || !plan.execution)
          return nullptr;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_wan_audio_mps_gelu_encode(
    void *plan_handle, void *command_handle) {
  if (!plan_handle || !command_handle) return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      CogniWanAudioMPSGeluPlan *plan =
          (__bridge CogniWanAudioMPSGeluPlan *)plan_handle;
      MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)command_handle;
      id<MTLCommandBuffer> root = command.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted) return -2;
      @try {
        NSArray<MPSGraphTensorData *> *results =
            [plan.executable encodeToCommandBuffer:command
                                       inputsArray:plan.inputArray
                                      resultsArray:plan.outputArray
                               executionDescriptor:plan.execution];
        if (results.count != 1) return -4;
      } @catch (NSException *exception) {
        (void)exception;
        return -5;
      }
      return 0;
    }
  }
  return -6;
}

extern "C" void gs_wan_audio_mps_gelu_plan_release(void *plan_handle) {
  if (plan_handle) {
    id plan = (__bridge_transfer id)plan_handle;
    (void)plan;
  }
}
