// Fixed-buffer canonical Hy-Nomas attention and same-input reduction controls.
// Fixed Arena buffers stay BQHD/BQHK; MSL transposes materialize contiguous
// BHQD/BHQK operands before Graph and restore the expected output layout.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdint.h>
#include <limits.h>
#include <math.h>

@interface HyNomasMPSReductionPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic, strong) id<MTLComputePipelineState> toHeadMajor;
@property(nonatomic, strong) id<MTLComputePipelineState> toStreamMajor;
@property(nonatomic) int32_t mode;
@property(nonatomic) uint32_t batches;
@property(nonatomic) uint32_t queryRows;
@property(nonatomic) uint32_t keyRows;
@property(nonatomic) uint32_t heads;
@property(nonatomic) uint32_t headDim;
@end

@implementation HyNomasMPSReductionPlan
@end

struct HyNomasTransposeParams {
  uint32_t batches;
  uint32_t rows;
  uint32_t heads;
  uint32_t width;
};

static id<MTLDevice> hy_nomas_cached_device = nil;
static id<MTLComputePipelineState> hy_nomas_to_head_major = nil;
static id<MTLComputePipelineState> hy_nomas_to_stream_major = nil;

static bool hy_nomas_mps_product(const uint64_t *values, size_t count,
                                 uint64_t *result) {
  if (!values || !result)
    return false;
  uint64_t product = 1;
  for (size_t index = 0; index < count; ++index) {
    if (values[index] == 0 || product > UINT64_MAX / values[index])
      return false;
    product *= values[index];
  }
  *result = product;
  return true;
}

static bool hy_nomas_mps_disjoint(id<MTLBuffer> left,
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

static bool hy_nomas_mps_transpose_pipelines(
    id<MTLDevice> device, id<MTLComputePipelineState> *toHeadMajor,
    id<MTLComputePipelineState> *toStreamMajor) {
  if (!device || !toHeadMajor || !toStreamMajor)
    return false;
  @synchronized(device) {
    if (hy_nomas_cached_device == device && hy_nomas_to_head_major &&
        hy_nomas_to_stream_major) {
      *toHeadMajor = hy_nomas_to_head_major;
      *toStreamMajor = hy_nomas_to_stream_major;
      return true;
    }

    NSString *source = [NSString stringWithUTF8String:R"metal(
      #include <metal_stdlib>
      using namespace metal;
      struct HyNomasTransposeParams {
        uint batches;
        uint rows;
        uint heads;
        uint width;
      };
      kernel void hy_nomas_to_head_major(
          device const float *source [[buffer(0)]],
          device float *destination [[buffer(1)]],
          constant HyNomasTransposeParams &p [[buffer(2)]],
          uint index [[thread_position_in_grid]]) {
        const uint total = p.batches * p.heads * p.rows * p.width;
        if (index >= total)
          return;
        const uint width = index % p.width;
        const uint row = (index / p.width) % p.rows;
        const uint head = (index / (p.width * p.rows)) % p.heads;
        const uint batch = index / (p.width * p.rows * p.heads);
        const uint sourceIndex =
            ((batch * p.rows + row) * p.heads + head) * p.width + width;
        destination[index] = source[sourceIndex];
      }
      kernel void hy_nomas_to_stream_major(
          device const float *source [[buffer(0)]],
          device float *destination [[buffer(1)]],
          constant HyNomasTransposeParams &p [[buffer(2)]],
          uint index [[thread_position_in_grid]]) {
        const uint total = p.batches * p.rows * p.heads * p.width;
        if (index >= total)
          return;
        const uint width = index % p.width;
        const uint head = (index / p.width) % p.heads;
        const uint row = (index / (p.width * p.heads)) % p.rows;
        const uint batch = index / (p.width * p.heads * p.rows);
        const uint sourceIndex =
            ((batch * p.heads + head) * p.rows + row) * p.width + width;
        destination[index] = source[sourceIndex];
      }
    )metal"];
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                   options:nil
                                                     error:&error];
    if (!library)
      return false;
    id<MTLFunction> toHeadFunction =
        [library newFunctionWithName:@"hy_nomas_to_head_major"];
    id<MTLFunction> toStreamFunction =
        [library newFunctionWithName:@"hy_nomas_to_stream_major"];
    if (!toHeadFunction || !toStreamFunction)
      return false;
    id<MTLComputePipelineState> toHead =
        [device newComputePipelineStateWithFunction:toHeadFunction error:&error];
    if (!toHead)
      return false;
    id<MTLComputePipelineState> toStream =
        [device newComputePipelineStateWithFunction:toStreamFunction error:&error];
    if (!toStream)
      return false;
    hy_nomas_cached_device = device;
    hy_nomas_to_head_major = toHead;
    hy_nomas_to_stream_major = toStream;
    *toHeadMajor = toHead;
    *toStreamMajor = toStream;
    return true;
  }
}

static bool hy_nomas_mps_encode_transpose(
    id<MTLCommandBuffer> root, id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> source, id<MTLBuffer> destination,
    HyNomasTransposeParams params, uint64_t count) {
  if (!root || !pipeline || !source || !destination || count == 0 ||
      count > UINT32_MAX)
    return false;
  id<MTLComputeCommandEncoder> encoder = [root computeCommandEncoder];
  if (!encoder)
    return false;
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:source offset:0 atIndex:0];
  [encoder setBuffer:destination offset:0 atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  NSUInteger width = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
  if (width == 0) {
    [encoder endEncoding];
    return false;
  }
  const NSUInteger groups = (NSUInteger)((count + width - 1) / width);
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
  return true;
}

extern "C" void *gs_hy_nomas_mps_reduction_plan_create(
    void *deviceHandle, int32_t mode, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, float scale,
    void *inputAHandle, void *inputBHandle, void *inputCHandle,
    void *packedAHandle, void *packedBHandle, void *packedCHandle,
    void *packedOutputHandle, void *outputHandle) {
  if (!deviceHandle || !inputAHandle || !packedAHandle ||
      !packedOutputHandle || !outputHandle || mode < 0 || mode > 3 ||
      batches <= 0 || queryRows <= 0 || keyRows <= 0 || heads <= 0 ||
      headDim <= 0 || !std::isfinite(scale) || scale <= 0.0f || scale > 1.0f)
    return nullptr;
  if ((mode == 0 || mode == 2) &&
      (!inputBHandle || !packedBHandle))
    return nullptr;
  if (mode == 3 && (!inputBHandle || !inputCHandle || !packedBHandle ||
                    !packedCHandle))
    return nullptr;
  if (mode == 1 && (inputBHandle || inputCHandle || packedBHandle ||
                    packedCHandle))
    return nullptr;
  if (mode != 3 && inputCHandle)
    return nullptr;

  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      const uint64_t qShapeValues[] = {(uint64_t)batches,
                                       (uint64_t)queryRows,
                                       (uint64_t)heads, (uint64_t)headDim};
      const uint64_t kShapeValues[] = {(uint64_t)batches,
                                       (uint64_t)keyRows,
                                       (uint64_t)heads, (uint64_t)headDim};
      const uint64_t scoreShapeValues[] = {(uint64_t)batches,
                                           (uint64_t)queryRows,
                                           (uint64_t)heads,
                                           (uint64_t)keyRows};
      uint64_t qElements = 0, kElements = 0, scoreElements = 0;
      if (!hy_nomas_mps_product(qShapeValues, 4, &qElements) ||
          !hy_nomas_mps_product(kShapeValues, 4, &kElements) ||
          !hy_nomas_mps_product(scoreShapeValues, 4, &scoreElements) ||
          qElements > (uint64_t)INT32_MAX ||
          kElements > (uint64_t)INT32_MAX ||
          scoreElements > (uint64_t)INT32_MAX ||
          qElements > UINT64_MAX / sizeof(float) ||
          kElements > UINT64_MAX / sizeof(float) ||
          scoreElements > UINT64_MAX / sizeof(float))
        return nullptr;
      const uint64_t qBytes = qElements * sizeof(float);
      const uint64_t kBytes = kElements * sizeof(float);
      const uint64_t scoreBytes = scoreElements * sizeof(float);

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> inputA = (__bridge id<MTLBuffer>)inputAHandle;
      id<MTLBuffer> inputB = inputBHandle
                                 ? (__bridge id<MTLBuffer>)inputBHandle
                                 : nil;
      id<MTLBuffer> inputC = inputCHandle
                                 ? (__bridge id<MTLBuffer>)inputCHandle
                                 : nil;
      id<MTLBuffer> packedA = (__bridge id<MTLBuffer>)packedAHandle;
      id<MTLBuffer> packedB = packedBHandle
                                  ? (__bridge id<MTLBuffer>)packedBHandle
                                  : nil;
      id<MTLBuffer> packedC = packedCHandle
                                  ? (__bridge id<MTLBuffer>)packedCHandle
                                  : nil;
      id<MTLBuffer> packedOutput =
          (__bridge id<MTLBuffer>)packedOutputHandle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;

      uint64_t inputABytes = (mode == 1 || mode == 2) ? scoreBytes : qBytes;
      uint64_t inputBBytes =
          (mode == 0 || mode == 2 || mode == 3) ? kBytes : 0;
      const uint64_t packedABytes = inputABytes;
      const uint64_t packedBBytes = inputBBytes;
      const uint64_t packedCBytes = kBytes;
      const uint64_t graphOutputBytes = (mode <= 1) ? scoreBytes : qBytes;
      const uint64_t finalOutputBytes = graphOutputBytes;

      NSMutableArray<id<MTLBuffer>> *allBuffers = [NSMutableArray new];
      [allBuffers addObject:inputA];
      [allBuffers addObject:packedA];
      if (inputB)
        [allBuffers addObject:inputB];
      if (packedB)
        [allBuffers addObject:packedB];
      if (inputC)
        [allBuffers addObject:inputC];
      if (packedC)
        [allBuffers addObject:packedC];
      [allBuffers addObject:packedOutput];
      [allBuffers addObject:output];
      for (id<MTLBuffer> buffer in allBuffers) {
        if (!buffer || buffer.device != device ||
            buffer.storageMode != MTLStorageModeShared)
          return nullptr;
      }
      if (inputA.length < inputABytes || packedA.length < packedABytes ||
          packedOutput.length < graphOutputBytes ||
          output.length < finalOutputBytes ||
          (inputB && (inputB.length < inputBBytes ||
                      packedB.length < packedBBytes)) ||
          (inputC && (inputC.length < packedCBytes ||
                      packedC.length < packedCBytes)))
        return nullptr;
      for (NSUInteger left = 0; left < allBuffers.count; ++left) {
        for (NSUInteger right = left + 1; right < allBuffers.count; ++right) {
          if (!hy_nomas_mps_disjoint(allBuffers[left], allBuffers[right]))
            return nullptr;
        }
      }

      id<MTLComputePipelineState> toHeadMajor = nil;
      id<MTLComputePipelineState> toStreamMajor = nil;
      if (!hy_nomas_mps_transpose_pipelines(device, &toHeadMajor,
                                           &toStreamMajor))
        return nullptr;

      @try {
        NSArray<NSNumber *> *headQShape =
            @[ @(batches), @(heads), @(queryRows), @(headDim) ];
        NSArray<NSNumber *> *headKShape =
            @[ @(batches), @(heads), @(keyRows), @(headDim) ];
        NSArray<NSNumber *> *headScoreShape =
            @[ @(batches), @(heads), @(queryRows), @(keyRows) ];
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
            [NSMutableDictionary new];
        NSMutableArray<MPSGraphTensor *> *placeholders = [NSMutableArray new];
        NSMutableArray<MPSGraphTensorData *> *inputData = [NSMutableArray new];
        MPSGraphTensor *result = nil;

        if (mode == 0 || mode == 3) {
          MPSGraphTensor *q = [graph placeholderWithShape:headQShape
                                                 dataType:MPSDataTypeFloat32
                                                     name:@"hy_q_bhqd"];
          MPSGraphTensor *k = [graph placeholderWithShape:headKShape
                                                 dataType:MPSDataTypeFloat32
                                                     name:@"hy_k_bhkd"];
          [placeholders addObject:q];
          [placeholders addObject:k];
          [inputData addObject:[[MPSGraphTensorData alloc]
                                   initWithMTLBuffer:packedA
                                              shape:headQShape
                                           dataType:MPSDataTypeFloat32]];
          [inputData addObject:[[MPSGraphTensorData alloc]
                                   initWithMTLBuffer:packedB
                                              shape:headKShape
                                           dataType:MPSDataTypeFloat32]];
          feeds[q] = [[MPSGraphShapedType alloc] initWithShape:headQShape
                                                     dataType:MPSDataTypeFloat32];
          feeds[k] = [[MPSGraphShapedType alloc] initWithShape:headKShape
                                                     dataType:MPSDataTypeFloat32];

          MPSGraphTensor *kTranspose = [graph transposeTensor:k
                                                    dimension:2
                                                withDimension:3
                                                         name:@"hy_k_bhdk"];
          MPSGraphTensor *logits =
              [graph matrixMultiplicationWithPrimaryTensor:q
                                          secondaryTensor:kTranspose
                                                     name:@"hy_qk_bhqk"];
          MPSGraphTensor *softmaxInput = nil;
          if (mode == 3) {
            // PyTorch 2.8 generic MPS SDPA on macOS 15 adds F32 epsilon before
            // scaling; it then softmaxes over K and multiplies by head-major V.
            MPSGraphTensor *epsilon =
                [graph constantWithScalar:1.0e-20f
                                    shape:@[ @1 ]
                                 dataType:MPSDataTypeFloat32];
            MPSGraphTensor *epsilonLogits =
                [graph additionWithPrimaryTensor:logits
                                  secondaryTensor:epsilon
                                             name:@"hy_qk_plus_epsilon_bhqk"];
            MPSGraphTensor *scaleTensor =
                [graph constantWithScalar:scale
                                    shape:@[ @1 ]
                                 dataType:MPSDataTypeFloat32];
            softmaxInput =
                [graph multiplicationWithPrimaryTensor:epsilonLogits
                                        secondaryTensor:scaleTensor
                                                   name:@"hy_scaled_qk_bhqk"];
            MPSGraphTensor *probabilities =
                [graph softMaxWithTensor:softmaxInput
                                    axis:3
                                    name:@"hy_probabilities_bhqk"];
            MPSGraphTensor *v = [graph placeholderWithShape:headKShape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"hy_v_bhkd"];
            [placeholders addObject:v];
            [inputData addObject:[[MPSGraphTensorData alloc]
                                     initWithMTLBuffer:packedC
                                                shape:headKShape
                                             dataType:MPSDataTypeFloat32]];
            feeds[v] = [[MPSGraphShapedType alloc] initWithShape:headKShape
                                                       dataType:MPSDataTypeFloat32];
            result =
                [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                            secondaryTensor:v
                                                       name:@"hy_context_bhqd"];
          } else {
            MPSGraphTensor *scaleTensor =
                [graph constantWithScalar:scale
                                    shape:@[ @1 ]
                                 dataType:MPSDataTypeFloat32];
            MPSGraphTensor *scaledLogits =
                [graph multiplicationWithPrimaryTensor:logits
                                        secondaryTensor:scaleTensor
                                                   name:@"hy_scaled_qk_bhqk"];
            result = scaledLogits;
          }
        } else if (mode == 1) {
          MPSGraphTensor *scores = [graph placeholderWithShape:headScoreShape
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"hy_scores_bhqk"];
          [placeholders addObject:scores];
          [inputData addObject:[[MPSGraphTensorData alloc]
                                   initWithMTLBuffer:packedA
                                              shape:headScoreShape
                                           dataType:MPSDataTypeFloat32]];
          feeds[scores] = [[MPSGraphShapedType alloc]
                             initWithShape:headScoreShape
                                  dataType:MPSDataTypeFloat32];
          result = [graph softMaxWithTensor:scores
                                      axis:3
                                      name:@"hy_probabilities_bhqk"];
        } else {
          MPSGraphTensor *probabilities =
              [graph placeholderWithShape:headScoreShape
                                 dataType:MPSDataTypeFloat32
                                     name:@"hy_probabilities_bhqk"];
          MPSGraphTensor *v = [graph placeholderWithShape:headKShape
                                                 dataType:MPSDataTypeFloat32
                                                     name:@"hy_v_bhkd"];
          [placeholders addObject:probabilities];
          [placeholders addObject:v];
          [inputData addObject:[[MPSGraphTensorData alloc]
                                   initWithMTLBuffer:packedA
                                              shape:headScoreShape
                                           dataType:MPSDataTypeFloat32]];
          [inputData addObject:[[MPSGraphTensorData alloc]
                                   initWithMTLBuffer:packedB
                                              shape:headKShape
                                           dataType:MPSDataTypeFloat32]];
          feeds[probabilities] = [[MPSGraphShapedType alloc]
                                    initWithShape:headScoreShape
                                         dataType:MPSDataTypeFloat32];
          feeds[v] = [[MPSGraphShapedType alloc]
                        initWithShape:headKShape
                             dataType:MPSDataTypeFloat32];
          result =
              [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                          secondaryTensor:v
                                                     name:@"hy_context_bhqd"];
        }

        if (!graph || !result || placeholders.count != inputData.count ||
            feeds.count != placeholders.count)
          return nullptr;
        NSArray<NSNumber *> *headOutputShape =
            (mode <= 1) ? headScoreShape : headQShape;
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:@[ result ]
                     targetOperations:nil
                compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        executable.options = MPSGraphOptionsNone;
        MPSGraphTensorData *outputData =
            [[MPSGraphTensorData alloc]
                initWithMTLBuffer:packedOutput
                            shape:headOutputShape
                         dataType:MPSDataTypeFloat32];
        if (!outputData)
          return nullptr;

        NSMutableArray<MPSGraphTensorData *> *orderedInputs = [NSMutableArray new];
        for (MPSGraphTensor *feedTensor in executable.feedTensors) {
          NSUInteger index = [placeholders indexOfObjectIdenticalTo:feedTensor];
          if (index == NSNotFound || index >= inputData.count)
            return nullptr;
          [orderedInputs addObject:inputData[index]];
        }
        if (orderedInputs.count != placeholders.count)
          return nullptr;

        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        HyNomasMPSReductionPlan *plan = [HyNomasMPSReductionPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.inputArray = [orderedInputs copy];
        plan.outputArray = @[ outputData ];
        plan.execution = execution;
        plan.residentBuffers = [allBuffers copy];
        plan.toHeadMajor = toHeadMajor;
        plan.toStreamMajor = toStreamMajor;
        plan.mode = mode;
        plan.batches = (uint32_t)batches;
        plan.queryRows = (uint32_t)queryRows;
        plan.keyRows = (uint32_t)keyRows;
        plan.heads = (uint32_t)heads;
        plan.headDim = (uint32_t)headDim;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_hy_nomas_mps_reduction_encode(void *planHandle,
                                                     void *commandHandle) {
  if (!planHandle || !commandHandle)
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      HyNomasMPSReductionPlan *plan =
          (__bridge HyNomasMPSReductionPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted)
        return -2;

      id<MTLBuffer> inputA = plan.residentBuffers[0];
      // residentBuffers order is fixed by plan creation and matches the MSL
      // staging cases below; locate the other fixed resources by mode.
      NSUInteger cursor = 1;
      id<MTLBuffer> packedA = plan.residentBuffers[cursor++];
      id<MTLBuffer> inputB = nil;
      id<MTLBuffer> packedB = nil;
      id<MTLBuffer> inputC = nil;
      id<MTLBuffer> packedC = nil;
      if (plan.mode == 0 || plan.mode == 2) {
        inputB = plan.residentBuffers[cursor++];
        packedB = plan.residentBuffers[cursor++];
      } else if (plan.mode == 3) {
        inputB = plan.residentBuffers[cursor++];
        packedB = plan.residentBuffers[cursor++];
        inputC = plan.residentBuffers[cursor++];
        packedC = plan.residentBuffers[cursor++];
      }
      id<MTLBuffer> packedOutput = plan.residentBuffers[cursor++];
      id<MTLBuffer> output = plan.residentBuffers[cursor++];
      if (cursor != plan.residentBuffers.count)
        return -3;

      HyNomasTransposeParams inputAParams = {
          plan.batches,
          plan.mode == 1 || plan.mode == 2 ? plan.queryRows : plan.queryRows,
          plan.heads,
          plan.mode == 1 || plan.mode == 2 ? plan.keyRows : plan.headDim};
      uint64_t inputACount = (uint64_t)plan.batches * inputAParams.rows *
                             plan.heads * inputAParams.width;
      if (!hy_nomas_mps_encode_transpose(root, plan.toHeadMajor, inputA,
                                         packedA, inputAParams, inputACount))
        return -4;
      if (inputB) {
        HyNomasTransposeParams inputBParams = {
            plan.batches,
            plan.mode == 0 || plan.mode == 2 || plan.mode == 3
                ? plan.keyRows
                : plan.queryRows,
            plan.heads,
            plan.mode == 0 || plan.mode == 2 || plan.mode == 3
                ? plan.headDim
                : plan.keyRows};
        uint64_t inputBCount = (uint64_t)plan.batches * inputBParams.rows *
                               plan.heads * inputBParams.width;
        if (!hy_nomas_mps_encode_transpose(root, plan.toHeadMajor, inputB,
                                           packedB, inputBParams, inputBCount))
          return -5;
      }
      if (inputC) {
        HyNomasTransposeParams inputCParams = {plan.batches, plan.keyRows,
                                               plan.heads, plan.headDim};
        uint64_t inputCCount = (uint64_t)plan.batches * plan.keyRows *
                               plan.heads * plan.headDim;
        if (!hy_nomas_mps_encode_transpose(root, plan.toHeadMajor, inputC,
                                           packedC, inputCParams, inputCCount))
          return -6;
      }

      @try {
        NSArray<MPSGraphTensorData *> *result =
            [plan.executable encodeToCommandBuffer:commandBuffer
                                       inputsArray:plan.inputArray
                                      resultsArray:plan.outputArray
                               executionDescriptor:plan.execution];
        if (result.count != 1)
          return -7;
        root = commandBuffer.rootCommandBuffer;
        if (!root || root.commandQueue.device != plan.device ||
            root.status >= MTLCommandBufferStatusCommitted)
          return -8;
        HyNomasTransposeParams outputParams = {
            plan.batches, plan.queryRows, plan.heads,
            plan.mode <= 1 ? plan.keyRows : plan.headDim};
        uint64_t outputCount = (uint64_t)plan.batches * plan.queryRows *
                               plan.heads * outputParams.width;
        if (!hy_nomas_mps_encode_transpose(root, plan.toStreamMajor,
                                           packedOutput, output, outputParams,
                                           outputCount))
          return -9;
        return 0;
      } @catch (NSException *exception) {
        (void)exception;
        return -10;
      }
    }
  }
  return -11;
}

extern "C" void gs_hy_nomas_mps_reduction_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}

// Test-only, pinned PyTorch 2.8 MPSGraph activation reproductions. Keep these
// unary plans independent from attention reductions: SiLU is negative -> exp
// -> add-one -> divide, while tanh GELU follows the exact upstream operation
// sequence with rank-one F32 scalar constants and original input rank/layout.
@interface HyNomasMPSActivationPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic) int32_t activation;
@end

@implementation HyNomasMPSActivationPlan
@end

static void *hy_nomas_mps_activation_plan_create(
    void *deviceHandle, const int64_t *shapeValues, int32_t rank,
    void *inputHandle, void *outputHandle, int32_t activation) {
  if (!deviceHandle || !shapeValues || !inputHandle || !outputHandle ||
      rank < 1 || rank > 8 || activation < 0 || activation > 1)
    return nullptr;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t shapeUnsigned[8];
      for (int32_t axis = 0; axis < rank; ++axis) {
        if (shapeValues[axis] <= 0)
          return nullptr;
        shapeUnsigned[axis] = (uint64_t)shapeValues[axis];
      }
      uint64_t elementCount = 0;
      if (!hy_nomas_mps_product(shapeUnsigned, (size_t)rank, &elementCount) ||
          elementCount > (uint64_t)INT32_MAX ||
          elementCount > UINT64_MAX / sizeof(float))
        return nullptr;
      const uint64_t byteCount = elementCount * sizeof(float);
      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
      if (input.device != device || output.device != device ||
          input.storageMode != MTLStorageModeShared ||
          output.storageMode != MTLStorageModeShared ||
          input.length != byteCount || output.length != byteCount ||
          !hy_nomas_mps_disjoint(input, output))
        return nullptr;

      @try {
        NSMutableArray<NSNumber *> *shape = [NSMutableArray new];
        for (int32_t axis = 0; axis < rank; ++axis) {
          [shape addObject:@(shapeValues[axis])];
        }

        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *x = [graph placeholderWithShape:shape
                                               dataType:MPSDataTypeFloat32
                                                   name:activation == 0 ? @"hy_silu_x" : @"hy_gelu_x"];
        MPSGraphTensor *result = nil;
        if (activation == 0) {
          MPSGraphTensor *one =
              [graph constantWithScalar:1.0
                                  shape:@[ @1 ]
                               dataType:MPSDataTypeFloat32];
          MPSGraphTensor *negative = [graph negativeWithTensor:x
                                                          name:@"hy_silu_negative"];
          MPSGraphTensor *exponential =
              [graph exponentWithTensor:negative name:@"hy_silu_exp_negative_x"];
          MPSGraphTensor *denominator =
              [graph additionWithPrimaryTensor:exponential
                               secondaryTensor:one
                                          name:@"hy_silu_one_plus_exp"];
          result = [graph divisionWithPrimaryTensor:x
                                secondaryTensor:denominator
                                           name:@"hy_silu_x_over_one_plus_exp"];
          if (!negative || !exponential || !one || !denominator)
            return nullptr;
        } else {
          // PyTorch 2.8 Activation.mm's approximate="tanh" graph, preserving
          // operation order and rank-one scalar constants exactly.
          constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
          constexpr float kKappa = 0.044715f;
          MPSGraphTensor *beta = [graph constantWithScalar:kBeta
                                                      shape:@[ @1 ]
                                                   dataType:MPSDataTypeFloat32];
          MPSGraphTensor *kappa = [graph constantWithScalar:kKappa
                                                       shape:@[ @1 ]
                                                    dataType:MPSDataTypeFloat32];
          MPSGraphTensor *one = [graph constantWithScalar:1.0f
                                                    shape:@[ @1 ]
                                                 dataType:MPSDataTypeFloat32];
          MPSGraphTensor *half = [graph constantWithScalar:0.5f
                                                     shape:@[ @1 ]
                                                  dataType:MPSDataTypeFloat32];
          MPSGraphTensor *cube = [graph multiplicationWithPrimaryTensor:x
                                                        secondaryTensor:x
                                                                   name:@"hy_gelu_x_squared"];
          cube = [graph multiplicationWithPrimaryTensor:cube
                                       secondaryTensor:x
                                                  name:@"hy_gelu_x_cubed"];
          MPSGraphTensor *inner = [graph multiplicationWithPrimaryTensor:cube
                                                         secondaryTensor:kappa
                                                                    name:@"hy_gelu_kappa_x_cubed"];
          inner = [graph additionWithPrimaryTensor:inner
                                secondaryTensor:x
                                           name:@"hy_gelu_x_plus_kappa_x_cubed"];
          inner = [graph multiplicationWithPrimaryTensor:inner
                                       secondaryTensor:beta
                                                  name:@"hy_gelu_beta_inner"];
          MPSGraphTensor *tanhValue = [graph tanhWithTensor:inner
                                                       name:@"hy_gelu_tanh"];
          MPSGraphTensor *scaled = [graph additionWithPrimaryTensor:tanhValue
                                                 secondaryTensor:one
                                                            name:@"hy_gelu_one_plus_tanh"];
          scaled = [graph multiplicationWithPrimaryTensor:scaled
                                         secondaryTensor:half
                                                    name:@"hy_gelu_half_one_plus_tanh"];
          result = [graph multiplicationWithPrimaryTensor:scaled
                                          secondaryTensor:x
                                                     name:@"hy_gelu_x_times_cdf"];
          if (!beta || !kappa || !one || !half || !cube || !inner ||
              !tanhValue || !scaled)
            return nullptr;
        }
        if (!graph || !x || !result || result.shape == nil)
          return nullptr;

        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
            [NSMutableDictionary new];
        feeds[x] = [[MPSGraphShapedType alloc] initWithShape:shape
                                                   dataType:MPSDataTypeFloat32];
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:@[ result ]
                     targetOperations:nil
                compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        executable.options = MPSGraphOptionsNone;

        MPSGraphTensorData *inputData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:input
                                                    shape:shape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *outputData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:output
                                                    shape:shape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        if (!inputData || !outputData || !execution ||
            executable.feedTensors.count != 1 ||
            executable.feedTensors[0] != x)
          return nullptr;

        HyNomasMPSActivationPlan *plan = [HyNomasMPSActivationPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.inputArray = @[ inputData ];
        plan.outputArray = @[ outputData ];
        plan.execution = execution;
        plan.residentBuffers = @[ input, output ];
        plan.activation = activation;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" void *gs_hy_nomas_mps_activation_plan_create(
    void *deviceHandle, int32_t activation, const int64_t *shapeValues,
    int32_t rank, void *inputHandle, void *outputHandle) {
  return hy_nomas_mps_activation_plan_create(
      deviceHandle, shapeValues, rank, inputHandle, outputHandle, activation);
}

extern "C" int32_t gs_hy_nomas_mps_activation_encode(void *planHandle,
                                                       void *commandHandle) {
  if (!planHandle || !commandHandle)
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      HyNomasMPSActivationPlan *plan =
          (__bridge HyNomasMPSActivationPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted ||
          plan.residentBuffers.count != 2 ||
          plan.inputArray.count != 1 || plan.outputArray.count != 1)
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

extern "C" void gs_hy_nomas_mps_activation_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}
