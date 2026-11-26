// A bounded, fixed-binding BF16 linear diagnostic for Hy-Nomas modulation.
// This is deliberately separate from the production block/backend path.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdint.h>
#include <limits.h>

enum HyNomasBF16LinearVariant : int32_t {
  HyNomasBF16GraphOps = 0,
  HyNomasF32Accumulation = 1,
};

@interface HyNomasBF16LinearPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic, strong) id<MTLComputePipelineState> expandOutputBF16;
@property(nonatomic) int32_t variant;
@end

@implementation HyNomasBF16LinearPlan
@end

static bool hy_nomas_bf16_linear_disjoint(id<MTLBuffer> left,
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

static bool hy_nomas_bf16_linear_size(int32_t rows, int32_t width,
                                      uint64_t elementBytes,
                                      uint64_t *byteCount) {
  if (rows <= 0 || width <= 0 || !byteCount ||
      (uint64_t)rows > UINT64_MAX / (uint64_t)width)
    return false;
  const uint64_t count = (uint64_t)rows * (uint64_t)width;
  if (count > (uint64_t)INT32_MAX || count > UINT64_MAX / elementBytes)
    return false;
  *byteCount = count * elementBytes;
  return true;
}

static void *hy_nomas_bf16_linear_plan_create(
    void *deviceHandle, int32_t rows, int32_t inputWidth,
    int32_t outputWidth, int32_t variant, void *inputHandle,
    void *weightHandle, void *biasHandle, void *outputBF16Handle,
    void *outputF32Handle) {
  if (!deviceHandle || !inputHandle || !weightHandle || !biasHandle ||
      !outputBF16Handle || !outputF32Handle || (variant != HyNomasBF16GraphOps &&
                        variant != HyNomasF32Accumulation))
    return nullptr;

  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t inputBytes = 0;
      uint64_t weightBytes = 0;
      uint64_t biasBytes = 0;
      uint64_t outputBF16Bytes = 0;
      uint64_t outputBytes = 0;
      if (!hy_nomas_bf16_linear_size(rows, inputWidth, sizeof(float),
                                     &inputBytes) ||
          !hy_nomas_bf16_linear_size(outputWidth, inputWidth, 2,
                                     &weightBytes) ||
          !hy_nomas_bf16_linear_size(1, outputWidth, 2, &biasBytes) ||
          !hy_nomas_bf16_linear_size(rows, outputWidth, 2,
                                     &outputBF16Bytes) ||
          !hy_nomas_bf16_linear_size(rows, outputWidth, sizeof(float),
                                     &outputBytes))
        return nullptr;

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
      id<MTLBuffer> weight = (__bridge id<MTLBuffer>)weightHandle;
      id<MTLBuffer> bias = (__bridge id<MTLBuffer>)biasHandle;
      id<MTLBuffer> outputBF16 = (__bridge id<MTLBuffer>)outputBF16Handle;
      id<MTLBuffer> outputF32 = (__bridge id<MTLBuffer>)outputF32Handle;
      NSArray<id<MTLBuffer>> *buffers = @[
        input, weight, bias, outputBF16, outputF32,
      ];
      const uint64_t expectedBytes[] = {
          inputBytes, weightBytes, biasBytes, outputBF16Bytes, outputBytes,
      };
      for (NSUInteger index = 0; index < buffers.count; ++index) {
        id<MTLBuffer> buffer = buffers[index];
        if (!buffer || buffer.device != device ||
            buffer.storageMode != MTLStorageModeShared ||
            buffer.length != expectedBytes[index])
          return nullptr;
        for (NSUInteger previous = 0; previous < index; ++previous) {
          if (!hy_nomas_bf16_linear_disjoint(buffer, buffers[previous]))
            return nullptr;
        }
      }

      NSArray<NSNumber *> *inputShape = @[ @(rows), @(inputWidth) ];
      NSArray<NSNumber *> *weightShape = @[ @(outputWidth), @(inputWidth) ];
      NSArray<NSNumber *> *biasShape = @[ @(outputWidth) ];
      NSArray<NSNumber *> *outputShape = @[ @(rows), @(outputWidth) ];

      @try {
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *inputF32 =
            [graph placeholderWithShape:inputShape
                               dataType:MPSDataTypeFloat32
                                   name:@"hy_bf16_linear_input_f32_storage"];
        MPSGraphTensor *weightBF16 =
            [graph placeholderWithShape:weightShape
                               dataType:MPSDataTypeBFloat16
                                   name:@"hy_bf16_linear_weight_resident"];
        MPSGraphTensor *biasBF16 =
            [graph placeholderWithShape:biasShape
                               dataType:MPSDataTypeBFloat16
                                   name:@"hy_bf16_linear_bias_resident"];

        // Quantize the activation at the native block's linear input boundary.
        MPSGraphTensor *inputBF16 =
            [graph castTensor:inputF32
                       toType:MPSDataTypeBFloat16
                         name:@"hy_bf16_linear_input_boundary"];
        MPSGraphTensor *weightTransposed =
            [graph transposeTensor:weightBF16
                         dimension:0
                    withDimension:1
                             name:@"hy_bf16_linear_weight_transpose"];

        MPSGraphTensor *linear = nil;
        if (variant == HyNomasBF16GraphOps) {
          // Both graph operations consume and produce BF16 tensors. The
          // graph may fuse them internally; this variant names the explicit
          // graph-level dtype boundaries without claiming PyTorch parity.
          MPSGraphTensor *product =
              [graph matrixMultiplicationWithPrimaryTensor:inputBF16
                                           secondaryTensor:weightTransposed
                                                      name:@"hy_bf16_linear_matmul"];
          linear = [graph additionWithPrimaryTensor:product
                                    secondaryTensor:biasBF16
                                               name:@"hy_bf16_linear_bias_add"];
        } else {
          // BF16 operands enter a graph whose accumulation and epilogue are
          // explicitly F32; the completed linear result is rounded once to
          // BF16 before expansion into caller-owned F32 storage.
          MPSGraphTensor *inputAccum =
              [graph castTensor:inputBF16
                         toType:MPSDataTypeFloat32
                           name:@"hy_bf16_linear_input_accum_f32"];
          MPSGraphTensor *weightAccum =
              [graph castTensor:weightBF16
                         toType:MPSDataTypeFloat32
                           name:@"hy_bf16_linear_weight_accum_f32"];
          MPSGraphTensor *biasAccum =
              [graph castTensor:biasBF16
                         toType:MPSDataTypeFloat32
                           name:@"hy_bf16_linear_bias_accum_f32"];
          MPSGraphTensor *weightAccumTransposed =
              [graph transposeTensor:weightAccum
                           dimension:0
                      withDimension:1
                               name:@"hy_bf16_linear_weight_accum_transpose"];
          MPSGraphTensor *productAccum =
              [graph matrixMultiplicationWithPrimaryTensor:inputAccum
                                           secondaryTensor:weightAccumTransposed
                                                      name:@"hy_bf16_linear_matmul_f32_accum"];
          MPSGraphTensor *sumAccum =
              [graph additionWithPrimaryTensor:productAccum
                                secondaryTensor:biasAccum
                                           name:@"hy_bf16_linear_bias_add_f32"];
          linear = [graph castTensor:sumAccum
                              toType:MPSDataTypeBFloat16
                                name:@"hy_bf16_linear_output_boundary"];
        }

        if (!graph || !inputF32 || !weightBF16 || !biasBF16 || !inputBF16 ||
            !weightTransposed || !linear || linear.dataType != MPSDataTypeBFloat16)
          return nullptr;

        NSDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds = @{
          inputF32 : [[MPSGraphShapedType alloc] initWithShape:inputShape
                                                     dataType:MPSDataTypeFloat32],
          weightBF16 : [[MPSGraphShapedType alloc] initWithShape:weightShape
                                                       dataType:MPSDataTypeBFloat16],
          biasBF16 : [[MPSGraphShapedType alloc] initWithShape:biasShape
                                                     dataType:MPSDataTypeBFloat16],
        };
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:@[ linear ]
                     targetOperations:nil
                compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        executable.options = MPSGraphOptionsNone;

        NSArray<MPSGraphTensor *> *placeholders =
            @[ inputF32, weightBF16, biasBF16 ];
        NSArray<NSArray<NSNumber *> *> *placeholderShapes = @[
          inputShape, weightShape, biasShape,
        ];
        NSArray<NSNumber *> *placeholderTypes = @[
          @(MPSDataTypeFloat32), @(MPSDataTypeBFloat16), @(MPSDataTypeBFloat16),
        ];
        NSArray<id<MTLBuffer>> *feedBuffers = @[ input, weight, bias ];
        NSMutableArray<MPSGraphTensorData *> *feedData = [NSMutableArray new];
        for (NSUInteger index = 0; index < placeholders.count; ++index) {
          NSArray<NSNumber *> *shape = placeholderShapes[index];
          MPSGraphTensorData *data =
              [[MPSGraphTensorData alloc] initWithMTLBuffer:feedBuffers[index]
                                                      shape:shape
                                                   dataType:(MPSDataType)placeholderTypes[index].intValue];
          if (!data)
            return nullptr;
          [feedData addObject:data];
        }
        NSMutableArray<MPSGraphTensorData *> *orderedInputs = [NSMutableArray new];
        for (MPSGraphTensor *feedTensor in executable.feedTensors) {
          NSUInteger index = [placeholders indexOfObjectIdenticalTo:feedTensor];
          if (index == NSNotFound || index >= feedData.count)
            return nullptr;
          [orderedInputs addObject:feedData[index]];
        }
        if (orderedInputs.count != placeholders.count)
          return nullptr;

        MPSGraphTensorData *outputData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBF16
                                                    shape:outputShape
                                                 dataType:MPSDataTypeBFloat16];
        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        if (!outputData || !execution)
          return nullptr;

        NSString *expandSource = [NSString stringWithUTF8String:R"metal(
          #include <metal_stdlib>
          using namespace metal;
          kernel void hy_nomas_bf16_linear_expand_output(
              device const ushort *source [[buffer(0)]],
              device float *destination [[buffer(1)]],
              constant uint &count [[buffer(2)]],
              uint index [[thread_position_in_grid]]) {
            if (index < count)
              destination[index] = as_type<float>(uint(source[index]) << 16);
          }
        )metal"];
        NSError *metalError = nil;
        id<MTLLibrary> metalLibrary = [device newLibraryWithSource:expandSource
                                                           options:nil
                                                             error:&metalError];
        id<MTLFunction> expandFunction =
            [metalLibrary newFunctionWithName:@"hy_nomas_bf16_linear_expand_output"];
        id<MTLComputePipelineState> expandPipeline = expandFunction
            ? [device newComputePipelineStateWithFunction:expandFunction
                                                    error:&metalError] : nil;
        if (!expandSource || !metalLibrary || !expandFunction || !expandPipeline)
          return nullptr;

        HyNomasBF16LinearPlan *plan = [HyNomasBF16LinearPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.inputArray = [orderedInputs copy];
        plan.outputArray = @[ outputData ];
        plan.execution = execution;
        plan.residentBuffers = buffers;
        plan.expandOutputBF16 = expandPipeline;
        plan.variant = variant;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" void *gs_hy_nomas_mps_bf16_linear_plan_create(
    void *deviceHandle, int32_t rows, int32_t inputWidth,
    int32_t outputWidth, int32_t variant, void *inputHandle,
    void *weightHandle, void *biasHandle, void *outputBF16Handle,
    void *outputF32Handle) {
  return hy_nomas_bf16_linear_plan_create(deviceHandle, rows, inputWidth,
      outputWidth, variant, inputHandle, weightHandle, biasHandle,
      outputBF16Handle, outputF32Handle);
}

extern "C" int32_t gs_hy_nomas_mps_bf16_linear_encode(
    void *planHandle, void *commandHandle) {
  if (!planHandle || !commandHandle)
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      HyNomasBF16LinearPlan *plan =
          (__bridge HyNomasBF16LinearPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted ||
          plan.residentBuffers.count != 5 || plan.inputArray.count != 3 ||
          plan.outputArray.count != 1)
        return -2;
      @try {
        NSArray<MPSGraphTensorData *> *result =
            [plan.executable encodeToCommandBuffer:commandBuffer
                                       inputsArray:plan.inputArray
                                      resultsArray:plan.outputArray
                               executionDescriptor:plan.execution];
        if (result.count != 1)
          return -3;
        root = commandBuffer.rootCommandBuffer;
        if (!root || root.commandQueue.device != plan.device ||
            root.status >= MTLCommandBufferStatusCommitted)
          return -4;
        id<MTLComputeCommandEncoder> encoder = [root computeCommandEncoder];
        if (!encoder)
          return -5;
        [encoder setComputePipelineState:plan.expandOutputBF16];
        [encoder setBuffer:plan.residentBuffers[3] offset:0 atIndex:0];
        [encoder setBuffer:plan.residentBuffers[4] offset:0 atIndex:1];
        const uint32_t count =
            (uint32_t)(plan.residentBuffers[4].length / sizeof(float));
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        const NSUInteger width = MIN((NSUInteger)256,
            plan.expandOutputBF16.maxTotalThreadsPerThreadgroup);
        if (width == 0) {
          [encoder endEncoding];
          return -6;
        }
        const NSUInteger groups = (NSUInteger)((count + width - 1) / width);
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        return 0;
      } @catch (NSException *exception) {
        (void)exception;
        return -7;
      }
    }
  }
  return -5;
}

extern "C" void gs_hy_nomas_mps_bf16_linear_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}
