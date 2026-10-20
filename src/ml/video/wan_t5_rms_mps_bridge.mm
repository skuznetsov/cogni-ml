// Prepared Wan T5 RMS mean plan plus bounded power diagnostics. Production
// writes rounded F32 squares into caller-owned storage with a native kernel,
// then reduces that buffer with MPSGraph. Encode uses the caller's
// MPSCommandBuffer wrapper and does not submit, wait, access activation bytes on
// the CPU, or allocate wrapped GPU buffers. MPSGraph's private temporary
// allocations are not observable here and remain outside wrapper accounting.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <limits>
#include <stdint.h>

@interface CogniWanT5RMSGraphPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphTensorData *squaredInput;
@property(nonatomic, strong) MPSGraphTensorData *meanOutput;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@end

@implementation CogniWanT5RMSGraphPlan
@end

@interface CogniWanT5PowerGraphPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphTensorData *input;
@property(nonatomic, strong) MPSGraphTensorData *output;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@end

@implementation CogniWanT5PowerGraphPlan
@end

@interface CogniWanT5PowerFeedGraphPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphTensorData *input;
@property(nonatomic, strong) MPSGraphTensorData *exponent;
@property(nonatomic, strong) MPSGraphTensorData *output;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) MPSGraphTensorDataDictionary *feedDictionary;
@property(nonatomic, strong) MPSGraphTensorDataDictionary *resultDictionary;
@property(nonatomic) BOOL encodeGraphDirectly;
@end

@implementation CogniWanT5PowerFeedGraphPlan
@end

static bool wan_t5_rms_disjoint(id<MTLBuffer> a, id<MTLBuffer> b) {
  if (a == b || a.gpuAddress == 0 || b.gpuAddress == 0)
    return false;
  const uint64_t aStart = (uint64_t)a.gpuAddress;
  const uint64_t bStart = (uint64_t)b.gpuAddress;
  const uint64_t aEnd = aStart + (uint64_t)a.length;
  const uint64_t bEnd = bStart + (uint64_t)b.length;
  if (aEnd < aStart || bEnd < bStart)
    return false;
  return aEnd <= bStart || bEnd <= aStart;
}

static bool wan_t5_rms_sizes(int32_t batches, int32_t tokens, int32_t dim,
                             uint64_t *inputBytes, uint64_t *outputBytes) {
  if (batches <= 0 || tokens <= 0 || dim <= 0)
    return false;
  const uint64_t rows = (uint64_t)batches * (uint64_t)tokens;
  if (rows > (uint64_t)INT32_MAX || rows > (uint64_t)INT32_MAX / (uint64_t)dim)
    return false;
  const uint64_t elements = rows * (uint64_t)dim;
  if (elements > UINT64_MAX / sizeof(float) ||
      rows > UINT64_MAX / sizeof(float))
    return false;
  *inputBytes = elements * sizeof(float);
  *outputBytes = rows * sizeof(float);
  return true;
}

extern "C" void *gs_wan_t5_rms_mps_plan_create(void *deviceHandle,
                                               int32_t batches, int32_t tokens,
                                               int32_t dim,
                                               void *squaredBufferHandle,
                                               void *meanBufferHandle) {
  if (!deviceHandle || !squaredBufferHandle || !meanBufferHandle)
    return nullptr;
  if (@available(macOS 12.0, *)) {
    @autoreleasepool {
      uint64_t squaredBytes = 0, meanBytes = 0;
      if (!wan_t5_rms_sizes(batches, tokens, dim, &squaredBytes, &meanBytes))
        return nullptr;

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> squared = (__bridge id<MTLBuffer>)squaredBufferHandle;
      id<MTLBuffer> means = (__bridge id<MTLBuffer>)meanBufferHandle;
      if (squared.device != device || means.device != device ||
          squared.length < squaredBytes || means.length < meanBytes ||
          !wan_t5_rms_disjoint(squared, means))
        return nullptr;

      @try {
        NSArray<NSNumber *> *inputShape = @[ @(batches), @(tokens), @(dim) ];
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *input = [graph placeholderWithShape:inputShape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"squared_input"];
        MPSGraphTensor *mean = [graph meanOfTensor:input
                                              axes:@[ @2 ]
                                              name:@"mean_feature_axis"];
        if (!graph || !input || !mean || mean.shape == nil)
          return nullptr;

        uint64_t outputElements = 1;
        for (NSNumber *axis in mean.shape) {
          const int64_t extent = axis.longLongValue;
          if (extent <= 0 || outputElements > UINT64_MAX / (uint64_t)extent)
            return nullptr;
          outputElements *= (uint64_t)extent;
        }
        if (outputElements != (uint64_t)batches * (uint64_t)tokens)
          return nullptr;

        MPSGraphShapedType *feedType =
            [[MPSGraphShapedType alloc] initWithShape:inputShape
                                             dataType:MPSDataTypeFloat32];
        MPSGraphTensorShapedTypeDictionary *feeds = @{input : feedType};
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:@[ mean ]
                     targetOperations:nil
                compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        // MPSGraphOptionsDefault includes SynchronizeResults. The diagnostic
        // must leave result residency on the supplied MTLBuffer instead.
        executable.options = MPSGraphOptionsNone;

        MPSGraphTensorData *squaredData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:squared
                                                    shape:inputShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *meanData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:means
                                                    shape:mean.shape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        if (!squaredData || !meanData || !execution)
          return nullptr;

        CogniWanT5RMSGraphPlan *plan = [CogniWanT5RMSGraphPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.squaredInput = squaredData;
        plan.meanOutput = meanData;
        plan.inputArray = @[ squaredData ];
        plan.outputArray = @[ meanData ];
        plan.execution = execution;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

// Diagnostic-only reproduction of the upstream `x.pow(2)` scalar operation.
// The input and rounded square are caller-owned resident F32 buffers. The
// power graph is deliberately separate from the mean graph, as in the pinned
// T5LayerNorm expression and PyTorch MPS scalar-power lowering.
extern "C" void *gs_wan_t5_power_mps_plan_create(void *deviceHandle,
                                                 int32_t batches,
                                                 int32_t tokens, int32_t dim,
                                                 void *inputBufferHandle,
                                                 void *squareBufferHandle) {
  if (!deviceHandle || !inputBufferHandle || !squareBufferHandle)
    return nullptr;
  if (@available(macOS 12.0, *)) {
    @autoreleasepool {
      uint64_t inputBytes = 0, outputBytes = 0;
      if (!wan_t5_rms_sizes(batches, tokens, dim, &inputBytes, &outputBytes))
        return nullptr;
      outputBytes = inputBytes;

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputBufferHandle;
      id<MTLBuffer> squareBuffer = (__bridge id<MTLBuffer>)squareBufferHandle;
      if (inputBuffer.device != device || squareBuffer.device != device ||
          inputBuffer.length < inputBytes ||
          squareBuffer.length < outputBytes ||
          !wan_t5_rms_disjoint(inputBuffer, squareBuffer))
        return nullptr;

      @try {
        NSArray<NSNumber *> *shape = @[ @(batches), @(tokens), @(dim) ];
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *input = [graph placeholderWithShape:shape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"power_input"];
        // PyTorch's pow_tensor_scalar MPS path lowers to MPSGraph `power`.
        // Match its rank-one scalar broadcast shape instead of using the
        // dedicated square op, which is a distinct graph primitive.
        MPSGraphTensor *two = [graph constantWithScalar:2.0
                                                  shape:@[ @1 ]
                                               dataType:MPSDataTypeFloat32];
        MPSGraphTensor *square =
            [graph powerWithPrimaryTensor:input
                          secondaryTensor:two
                                     name:@"power_scalar_two"];
        if (!graph || !input || !two || !square || square.shape == nil)
          return nullptr;

        MPSGraphShapedType *feedType =
            [[MPSGraphShapedType alloc] initWithShape:shape
                                             dataType:MPSDataTypeFloat32];
        MPSGraphTensorShapedTypeDictionary *feeds = @{input : feedType};
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:@[ square ]
                     targetOperations:nil
                compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        executable.options = MPSGraphOptionsNone;

        MPSGraphTensorData *inputData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                    shape:shape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *squareData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:squareBuffer
                                                    shape:shape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        if (!inputData || !squareData || !execution)
          return nullptr;

        CogniWanT5PowerGraphPlan *plan = [CogniWanT5PowerGraphPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.input = inputData;
        plan.output = squareData;
        plan.inputArray = @[ inputData ];
        plan.outputArray = @[ squareData ];
        plan.execution = execution;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

// Source-shaped diagnostic for PyTorch's `x.pow(2)` scalar path. The pinned
// MPS implementation represents the wrapped integer scalar as an Int64 tensor
// with shape [1], casts it to the promoted F32 dtype, then applies graph
// `power`. `executionMode` selects executable+default descriptor (0),
// executable+optimization level 0 (1), executable+optimization level 1 (2),
// or direct MPSGraph encode (3). `exponentType` is 0 for an F32 feed or 1 for
// the upstream Int64 feed. All feeds and results are caller-owned MTLBuffers.
extern "C" void *gs_wan_t5_power_feed_mps_plan_create(
    void *deviceHandle, int32_t batches, int32_t tokens, int32_t dim,
    void *inputBufferHandle, void *exponentBufferHandle,
    void *squareBufferHandle, int32_t exponentType, int32_t executionMode) {
  if (!deviceHandle || !inputBufferHandle || !exponentBufferHandle ||
      !squareBufferHandle || (exponentType != 0 && exponentType != 1) ||
      executionMode < 0 || executionMode > 3)
    return nullptr;
  if (@available(macOS 12.0, *)) {
    @autoreleasepool {
      uint64_t inputBytes = 0, outputBytes = 0;
      if (!wan_t5_rms_sizes(batches, tokens, dim, &inputBytes, &outputBytes))
        return nullptr;
      outputBytes = inputBytes;

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputBufferHandle;
      id<MTLBuffer> exponentBuffer =
          (__bridge id<MTLBuffer>)exponentBufferHandle;
      id<MTLBuffer> squareBuffer = (__bridge id<MTLBuffer>)squareBufferHandle;
      const uint64_t exponentBytes =
          exponentType == 1 ? sizeof(int64_t) : sizeof(float);
      if (inputBuffer.device != device || exponentBuffer.device != device ||
          squareBuffer.device != device || inputBuffer.length < inputBytes ||
          exponentBuffer.length < exponentBytes ||
          squareBuffer.length < outputBytes ||
          !wan_t5_rms_disjoint(inputBuffer, exponentBuffer) ||
          !wan_t5_rms_disjoint(inputBuffer, squareBuffer) ||
          !wan_t5_rms_disjoint(exponentBuffer, squareBuffer))
        return nullptr;

      @try {
        NSArray<NSNumber *> *inputShape = @[ @(batches), @(tokens), @(dim) ];
        NSArray<NSNumber *> *scalarShape = @[ @1 ];
        MPSDataType exponentDataType =
            exponentType == 1 ? MPSDataTypeInt64 : MPSDataTypeFloat32;
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *input = [graph placeholderWithShape:inputShape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"power_input"];
        MPSGraphTensor *exponent =
            [graph placeholderWithShape:scalarShape
                               dataType:exponentDataType
                                   name:@"power_exponent"];
        MPSGraphTensor *promotedExponent = exponent;
        if (exponentDataType != MPSDataTypeFloat32) {
          promotedExponent = [graph castTensor:exponent
                                        toType:MPSDataTypeFloat32
                                          name:@"power_exponent_f32"];
        }
        MPSGraphTensor *square =
            [graph powerWithPrimaryTensor:input
                          secondaryTensor:promotedExponent
                                     name:@"power_scalar_feed"];
        if (!graph || !input || !exponent || !promotedExponent || !square ||
            square.shape == nil)
          return nullptr;

        MPSGraphTensorData *inputData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                    shape:inputShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *exponentData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:exponentBuffer
                                                    shape:scalarShape
                                                 dataType:exponentDataType];
        MPSGraphTensorData *squareData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:squareBuffer
                                                    shape:inputShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        if (!inputData || !exponentData || !squareData || !execution)
          return nullptr;

        MPSGraphExecutable *executable = nil;
        if (executionMode != 3) {
          MPSGraphShapedType *inputType =
              [[MPSGraphShapedType alloc] initWithShape:inputShape
                                               dataType:MPSDataTypeFloat32];
          MPSGraphShapedType *exponentTypeShape =
              [[MPSGraphShapedType alloc] initWithShape:scalarShape
                                               dataType:exponentDataType];
          MPSGraphTensorShapedTypeDictionary *feedTypes =
              @{input : inputType, exponent : exponentTypeShape};
          MPSGraphCompilationDescriptor *compilation = nil;
          if (executionMode == 1 || executionMode == 2) {
            compilation = [MPSGraphCompilationDescriptor new];
            if (@available(macOS 12.3, *)) {
              compilation.optimizationLevel = executionMode == 1
                                                  ? MPSGraphOptimizationLevel0
                                                  : MPSGraphOptimizationLevel1;
            } else {
              return nullptr;
            }
            if (@available(macOS 13.0, *)) {
              compilation.waitForCompilationCompletion = YES;
            }
          }
          executable = [graph
                  compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                              feeds:feedTypes
                      targetTensors:@[ square ]
                   targetOperations:nil
              compilationDescriptor:compilation];
          if (!executable)
            return nullptr;
          executable.options = MPSGraphOptionsNone;
        }

        NSMutableArray<MPSGraphTensorData *> *orderedInputs =
            [NSMutableArray arrayWithCapacity:2];
        if (executable) {
          if (executable.feedTensors.count != 2)
            return nullptr;
          for (MPSGraphTensor *feedTensor in executable.feedTensors) {
            if (feedTensor == input)
              [orderedInputs addObject:inputData];
            else if (feedTensor == exponent)
              [orderedInputs addObject:exponentData];
            else
              return nullptr;
          }
        }

        CogniWanT5PowerFeedGraphPlan *plan = [CogniWanT5PowerFeedGraphPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.input = inputData;
        plan.exponent = exponentData;
        plan.output = squareData;
        plan.inputArray = [orderedInputs copy];
        plan.outputArray = @[ squareData ];
        plan.execution = execution;
        plan.feedDictionary = @{input : inputData, exponent : exponentData};
        plan.resultDictionary = @{square : squareData};
        plan.encodeGraphDirectly = executionMode == 3;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_wan_t5_power_feed_mps_encode(void *planHandle,
                                                   void *commandBufferHandle) {
  if (!planHandle || !commandBufferHandle)
    return -1;
  if (@available(macOS 12.0, *)) {
    @autoreleasepool {
      CogniWanT5PowerFeedGraphPlan *plan =
          (__bridge CogniWanT5PowerFeedGraphPlan *)planHandle;
      MPSCommandBuffer *commandBuffer =
          (__bridge MPSCommandBuffer *)commandBufferHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted)
        return -2;
      @try {
        if (plan.encodeGraphDirectly) {
          [plan.graph encodeToCommandBuffer:commandBuffer
                                      feeds:plan.feedDictionary
                           targetOperations:nil
                          resultsDictionary:plan.resultDictionary
                        executionDescriptor:nil];
          return 0;
        }
        NSArray<MPSGraphTensorData *> *result =
            [plan.executable encodeToCommandBuffer:commandBuffer
                                       inputsArray:plan.inputArray
                                      resultsArray:plan.outputArray
                               executionDescriptor:plan.execution];
        return result.count == 1 ? 0 : -3;
      } @catch (NSException *exception) {
        (void)exception;
        return -4;
      }
    }
  }
  return -5;
}

extern "C" void gs_wan_t5_power_feed_mps_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}

extern "C" int32_t gs_wan_t5_power_mps_encode(void *planHandle,
                                              void *commandBufferHandle) {
  if (!planHandle || !commandBufferHandle)
    return -1;
  if (@available(macOS 12.0, *)) {
    @autoreleasepool {
      CogniWanT5PowerGraphPlan *plan =
          (__bridge CogniWanT5PowerGraphPlan *)planHandle;
      MPSCommandBuffer *commandBuffer =
          (__bridge MPSCommandBuffer *)commandBufferHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted)
        return -2;
      @try {
        NSArray<MPSGraphTensorData *> *result =
            [plan.executable encodeToCommandBuffer:commandBuffer
                                       inputsArray:plan.inputArray
                                      resultsArray:plan.outputArray
                               executionDescriptor:plan.execution];
        return result.count == 1 ? 0 : -3;
      } @catch (NSException *exception) {
        (void)exception;
        return -4;
      }
    }
  }
  return -5;
}

extern "C" void gs_wan_t5_power_mps_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}

// commandBufferHandle is a borrowed MPSCommandBuffer wrapper around the
// caller's current MTLCommandBuffer. Reusing the same wrapper for all native
// and graph work is required because MPSGraph may commitAndContinue internally.
extern "C" int32_t gs_wan_t5_rms_mps_encode(void *planHandle,
                                            void *commandBufferHandle) {
  if (!planHandle || !commandBufferHandle)
    return -1;
  if (@available(macOS 12.0, *)) {
    @autoreleasepool {
      CogniWanT5RMSGraphPlan *plan =
          (__bridge CogniWanT5RMSGraphPlan *)planHandle;
      MPSCommandBuffer *commandBuffer =
          (__bridge MPSCommandBuffer *)commandBufferHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted)
        return -2;
      @try {
        NSArray<MPSGraphTensorData *> *result =
            [plan.executable encodeToCommandBuffer:commandBuffer
                                       inputsArray:plan.inputArray
                                      resultsArray:plan.outputArray
                               executionDescriptor:plan.execution];
        return result.count == 1 ? 0 : -3;
      } @catch (NSException *exception) {
        (void)exception;
        return -4;
      }
    }
  }
  return -5;
}

extern "C" void gs_wan_t5_rms_mps_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}
