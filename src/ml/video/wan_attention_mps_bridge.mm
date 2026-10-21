// Prepared opt-in unmasked MPSGraph attention experiment. The legacy path uses
// caller-owned F32 BTHD buffers; the typed path accepts F32, F16, or BF16 for
// Q/K/V/context. K/V are BTHD by default or caller-prepared BHTD on a
// head-major entrypoint. MPSGraph may allocate private internal workspace, so
// this bridge makes no memory-efficiency claim.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>
#include <stdint.h>
#include <limits>

@interface CogniWanAttentionMPSPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArrayA;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArrayB;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@end

@implementation CogniWanAttentionMPSPlan
@end

static bool wan_attention_mps_product(const uint64_t *values, size_t count,
                                      uint64_t *result) {
  uint64_t product = 1;
  for (size_t i = 0; i < count; ++i) {
    if (values[i] == 0 || product > UINT64_MAX / values[i])
      return false;
    product *= values[i];
  }
  *result = product;
  return true;
}

static bool wan_attention_mps_region(id<MTLBuffer> buffer, uint64_t offset,
                                     uint64_t size, uint64_t *start,
                                     uint64_t *end) {
  if (buffer.gpuAddress == 0 || offset > buffer.length ||
      size > buffer.length - offset)
    return false;
  const uint64_t base = (uint64_t)buffer.gpuAddress;
  if (offset > UINT64_MAX - base)
    return false;
  *start = base + offset;
  if (size > UINT64_MAX - *start)
    return false;
  *end = *start + size;
  return true;
}

static bool wan_attention_mps_ranges_disjoint(id<MTLBuffer> a,
                                               uint64_t aOffset,
                                               id<MTLBuffer> b,
                                               uint64_t bOffset,
                                               uint64_t size) {
  uint64_t aStart, aEnd, bStart, bEnd;
  if (!wan_attention_mps_region(a, aOffset, size, &aStart, &aEnd) ||
      !wan_attention_mps_region(b, bOffset, size, &bStart, &bEnd))
    return false;
  return aEnd <= bStart || bEnd <= aStart;
}

static bool wan_attention_mps_disjoint(id<MTLBuffer> a, id<MTLBuffer> b) {
  uint64_t aStart, aEnd, bStart, bEnd;
  if (a == b || !wan_attention_mps_region(a, 0, a.length, &aStart, &aEnd) ||
      !wan_attention_mps_region(b, 0, b.length, &bStart, &bEnd))
    return false;
  return aEnd <= bStart || bEnd <= aStart;
}

static bool wan_attention_mps_data_type(int32_t dtype, MPSDataType *dataType,
                                        uint64_t *elementBytes) {
  if (!dataType || !elementBytes)
    return false;
  switch (dtype) {
  case 0: // F32
    *dataType = MPSDataTypeFloat32;
    *elementBytes = 4;
    return true;
  case 1: // F16
    *dataType = MPSDataTypeFloat16;
    *elementBytes = 2;
    return true;
  case 2: // BF16
    if (@available(macOS 14.0, *)) {
      *dataType = MPSDataTypeBFloat16;
      *elementBytes = 2;
      return true;
    }
    return false;
  default:
    return false;
  }
}

static void *wan_attention_mps_plan_create_impl(
    void *deviceHandle, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, void *qHandle,
    void *kHandle, void *vHandle, void *outputAHandle,
    void *outputBHandle, MPSDataType dataType, uint64_t elementBytes,
    bool headMajorKV, bool explicitQKSoftmaxPV,
    bool allowPrivateBindings = false) {
  if (!deviceHandle || !qHandle || !kHandle || !vHandle || !outputAHandle ||
      !outputBHandle)
    return nullptr;
  if (explicitQKSoftmaxPV && !headMajorKV)
    return nullptr;
  if (@available(macOS 15.0, *)) {
    if ((dataType != MPSDataTypeFloat32 && dataType != MPSDataTypeFloat16 &&
         dataType != MPSDataTypeBFloat16) ||
        (elementBytes != 2 && elementBytes != 4) ||
        (dataType == MPSDataTypeFloat32) != (elementBytes == 4) ||
        (explicitQKSoftmaxPV && dataType != MPSDataTypeFloat32))
      return nullptr;
    @autoreleasepool {
      if (batches <= 0 || queryRows <= 0 || keyRows <= 0 || heads <= 0 ||
          headDim <= 0)
        return nullptr;

      const uint64_t qShapeValues[] = {(uint64_t)batches,
                                       (uint64_t)queryRows,
                                       (uint64_t)heads, (uint64_t)headDim};
      const uint64_t kvShapeValues[] = {(uint64_t)batches,
                                        (uint64_t)keyRows,
                                        (uint64_t)heads, (uint64_t)headDim};
      uint64_t qElements = 0, kvElements = 0;
      if (!wan_attention_mps_product(qShapeValues, 4, &qElements) ||
          !wan_attention_mps_product(kvShapeValues, 4, &kvElements) ||
          qElements > (uint64_t)INT32_MAX ||
          kvElements > (uint64_t)INT32_MAX ||
          qElements > UINT64_MAX / elementBytes ||
          kvElements > UINT64_MAX / elementBytes)
        return nullptr;
      const uint64_t qBytes = qElements * elementBytes;
      const uint64_t kvBytes = kvElements * elementBytes;

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> qBuffer = (__bridge id<MTLBuffer>)qHandle;
      id<MTLBuffer> kBuffer = (__bridge id<MTLBuffer>)kHandle;
      id<MTLBuffer> vBuffer = (__bridge id<MTLBuffer>)vHandle;
      id<MTLBuffer> outputABuffer = (__bridge id<MTLBuffer>)outputAHandle;
      id<MTLBuffer> outputBBuffer = (__bridge id<MTLBuffer>)outputBHandle;
      NSArray<id<MTLBuffer>> *allBuffers =
          @[ qBuffer, kBuffer, vBuffer, outputABuffer, outputBBuffer ];
      for (id<MTLBuffer> buffer in allBuffers) {
        // Only the same-stream Torch diagnostic opts into private GPU storage.
        // Every public production constructor keeps the original shared-only gate.
        if (buffer.device != device ||
            (buffer.storageMode != MTLStorageModeShared &&
             !(allowPrivateBindings && buffer.storageMode == MTLStorageModePrivate)))
          return nullptr;
      }
      if (qBuffer.length < qBytes || outputABuffer.length < qBytes ||
          outputBBuffer.length < qBytes || kBuffer.length < kvBytes ||
          vBuffer.length < kvBytes)
        return nullptr;
      for (NSUInteger i = 0; i < allBuffers.count; ++i) {
        for (NSUInteger j = i + 1; j < allBuffers.count; ++j) {
          if (!wan_attention_mps_disjoint(allBuffers[i], allBuffers[j]))
            return nullptr;
        }
      }

      @try {
        NSArray<NSNumber *> *qShape =
            @[ @(batches), @(queryRows), @(heads), @(headDim) ];
        NSArray<NSNumber *> *kvShape =
            headMajorKV ? @[ @(batches), @(heads), @(keyRows), @(headDim) ]
                        : @[ @(batches), @(keyRows), @(heads), @(headDim) ];
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *q = [graph placeholderWithShape:qShape
                                               dataType:dataType
                                                   name:@"wan_q_bthd"];
        MPSGraphTensor *k = [graph placeholderWithShape:kvShape
                                               dataType:dataType
                                                   name:headMajorKV ? @"wan_k_bhtd"
                                                                    : @"wan_k_bthd"];
        MPSGraphTensor *v = [graph placeholderWithShape:kvShape
                                               dataType:dataType
                                                   name:headMajorKV ? @"wan_v_bhtd"
                                                                    : @"wan_v_bthd"];
        MPSGraphTensor *qh = [graph transposeTensor:q
                                          dimension:1
                                      withDimension:2
                                               name:@"wan_q_bhtd"];
        MPSGraphTensor *kh = headMajorKV
                                 ? k
                                 : [graph transposeTensor:k
                                                dimension:1
                                            withDimension:2
                                                     name:@"wan_k_bhtd"];
        MPSGraphTensor *vh = headMajorKV
                                 ? v
                                 : [graph transposeTensor:v
                                                dimension:1
                                            withDimension:2
                                                     name:@"wan_v_bhtd"];
        const float scale = 1.0f / sqrtf((float)headDim);
        MPSGraphTensor *contextHeads = nil;
        MPSGraphTensor *keyTranspose = nil;
        MPSGraphTensor *queryKey = nil;
        MPSGraphTensor *queryKeyWithEpsilon = nil;
        MPSGraphTensor *scaleTensor = nil;
        MPSGraphTensor *scaledQueryKey = nil;
        MPSGraphTensor *probabilities = nil;
        if (explicitQKSoftmaxPV) {
          keyTranspose = [graph transposeTensor:k
                                      dimension:2
                                  withDimension:3
                                           name:@"wan_k_transpose_bhdq"];
          queryKey = [graph matrixMultiplicationWithPrimaryTensor:qh
                                                 secondaryTensor:keyTranspose
                                                            name:@"wan_qk_bhqk"];
          // Match PyTorch 2.8.0's MPS attention workaround for the MPSGraph
          // SDPA optimizer/leak behavior: add F32 epsilon before scaling.
          MPSGraphTensor *epsilon = [graph constantWithScalar:1.0e-20f
                                                        shape:@[ @1 ]
                                                     dataType:MPSDataTypeFloat32];
          queryKeyWithEpsilon = [graph additionWithPrimaryTensor:queryKey
                                                 secondaryTensor:epsilon
                                                            name:@"wan_qk_epsilon_bhqk"];
          scaleTensor = [graph constantWithScalar:scale
                                            shape:@[ @1 ]
                                         dataType:MPSDataTypeFloat32];
          scaledQueryKey = [graph multiplicationWithPrimaryTensor:queryKeyWithEpsilon
                                                  secondaryTensor:scaleTensor
                                                             name:@"wan_scaled_qk_bhqk"];
          probabilities = [graph softMaxWithTensor:scaledQueryKey
                                               axis:3
                                               name:@"wan_probabilities_bhqk"];
          contextHeads = [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                                      secondaryTensor:vh
                                                                 name:@"wan_context_bhqd"];
        } else {
          contextHeads =
              [graph scaledDotProductAttentionWithQueryTensor:qh
                                                     keyTensor:kh
                                                   valueTensor:vh
                                                         scale:scale
                                                          name:@"wan_sdpa_bhqd"];
        }
        MPSGraphTensor *context = [graph transposeTensor:contextHeads
                                               dimension:1
                                           withDimension:2
                                                    name:@"wan_context_bthd"];
        if (!graph || !q || !k || !v || !qh || !kh || !vh || !contextHeads ||
            !context || context.dataType != dataType ||
            (explicitQKSoftmaxPV &&
             (!keyTranspose || !queryKey || !queryKeyWithEpsilon ||
              !scaleTensor || !scaledQueryKey || !probabilities)))
          return nullptr;

        MPSGraphTensorShapedTypeDictionary *feeds = @{
          q : [[MPSGraphShapedType alloc] initWithShape:qShape
                                               dataType:dataType],
          k : [[MPSGraphShapedType alloc] initWithShape:kvShape
                                               dataType:dataType],
          v : [[MPSGraphShapedType alloc] initWithShape:kvShape
                                               dataType:dataType],
        };
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:@[ context ]
                     targetOperations:nil
                compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        executable.options = MPSGraphOptionsNone;

        MPSGraphTensorData *qData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:qBuffer
                                                    shape:qShape
                                                 dataType:dataType];
        MPSGraphTensorData *kData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:kBuffer
                                                    shape:kvShape
                                                 dataType:dataType];
        MPSGraphTensorData *vData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:vBuffer
                                                    shape:kvShape
                                                 dataType:dataType];
        MPSGraphTensorData *outputAData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:outputABuffer
                                                    shape:qShape
                                                 dataType:dataType];
        MPSGraphTensorData *outputBData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBBuffer
                                                    shape:qShape
                                                 dataType:dataType];
        if (!qData || !kData || !vData || !outputAData || !outputBData)
          return nullptr;

        NSMutableArray<MPSGraphTensorData *> *orderedInputs = [NSMutableArray new];
        for (MPSGraphTensor *feedTensor in executable.feedTensors) {
          if (feedTensor == q)
            [orderedInputs addObject:qData];
          else if (feedTensor == k)
            [orderedInputs addObject:kData];
          else if (feedTensor == v)
            [orderedInputs addObject:vData];
          else
            return nullptr;
        }
        if (orderedInputs.count != 3)
          return nullptr;

        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        CogniWanAttentionMPSPlan *plan = [CogniWanAttentionMPSPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.residentBuffers = allBuffers;
        plan.inputArray = [orderedInputs copy];
        plan.outputArrayA = @[ outputAData ];
        plan.outputArrayB = @[ outputBData ];
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

// Preserve the original BTHD K/V entrypoint and graph unchanged by default.
extern "C" void *gs_wan_attention_mps_plan_create(
    void *deviceHandle, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, void *qHandle,
    void *kHandle, void *vHandle, void *outputAHandle,
    void *outputBHandle) {
  return wan_attention_mps_plan_create_impl(
      deviceHandle, batches, queryRows, keyRows, heads, headDim, qHandle,
      kHandle, vHandle, outputAHandle, outputBHandle, MPSDataTypeFloat32, 4,
      false, false);
}

// Opt-in typed entrypoint. dtype codes are 0=F32, 1=F16, and 2=BF16; all
// inputs and both output slots use the selected element type.
extern "C" void *gs_wan_attention_mps_plan_create_typed(
    void *deviceHandle, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, void *qHandle,
    void *kHandle, void *vHandle, void *outputAHandle,
    void *outputBHandle, int32_t dtype) {
  MPSDataType dataType = MPSDataTypeInvalid;
  uint64_t elementBytes = 0;
  if (!wan_attention_mps_data_type(dtype, &dataType, &elementBytes))
    return nullptr;
  return wan_attention_mps_plan_create_impl(
      deviceHandle, batches, queryRows, keyRows, heads, headDim, qHandle,
      kHandle, vHandle, outputAHandle, outputBHandle, dataType, elementBytes,
      false, false);
}

// Opt-in path for caller-prepared contiguous BHTD K/V. Q and context remain
// BTHD, so only Q/context graph transposes are retained.
extern "C" void *gs_wan_attention_mps_plan_create_head_major_kv(
    void *deviceHandle, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, void *qHandle,
    void *kHandle, void *vHandle, void *outputAHandle,
    void *outputBHandle) {
  return wan_attention_mps_plan_create_impl(
      deviceHandle, batches, queryRows, keyRows, heads, headDim, qHandle,
      kHandle, vHandle, outputAHandle, outputBHandle, MPSDataTypeFloat32, 4,
      true, false);
}

// Opt-in explicit F32 QK/scale/softmax/PV graph on caller-prepared BHTD K/V.
extern "C" void *gs_wan_attention_mps_plan_create_head_major_kv_explicit(
    void *deviceHandle, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, void *qHandle,
    void *kHandle, void *vHandle, void *outputAHandle,
    void *outputBHandle) {
  return wan_attention_mps_plan_create_impl(
      deviceHandle, batches, queryRows, keyRows, heads, headDim, qHandle,
      kHandle, vHandle, outputAHandle, outputBHandle, MPSDataTypeFloat32, 4,
      true, true);
}

extern "C" int32_t gs_wan_attention_mps_encode(void *planHandle,
                                                 void *commandBufferHandle,
                                                 int32_t outputSlot) {
  if (!planHandle || !commandBufferHandle || (outputSlot != 0 && outputSlot != 1))
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      CogniWanAttentionMPSPlan *plan =
          (__bridge CogniWanAttentionMPSPlan *)planHandle;
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
                                      resultsArray:outputSlot == 0
                                                      ? plan.outputArrayA
                                                      : plan.outputArrayB
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

// Enqueue a native blit copy into the paused chain's current root. This is
// intentionally included in the same chain as Graph work and never allocates a
// MetalBuffer or waits on the host.
extern "C" int32_t gs_wan_attention_mps_copy(void *commandBufferHandle,
                                              void *sourceHandle,
                                              void *destinationHandle,
                                              uint64_t sourceOffset,
                                              uint64_t destinationOffset,
                                              uint64_t byteCount) {
  if (!commandBufferHandle || !sourceHandle || !destinationHandle ||
      byteCount == 0)
    return -1;
  MPSCommandBuffer *commandBuffer =
      (__bridge MPSCommandBuffer *)commandBufferHandle;
  id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
  id<MTLBuffer> source = (__bridge id<MTLBuffer>)sourceHandle;
  id<MTLBuffer> destination = (__bridge id<MTLBuffer>)destinationHandle;
  if (!root || root.status >= MTLCommandBufferStatusCommitted ||
      source.device != root.commandQueue.device ||
      destination.device != root.commandQueue.device ||
      source.storageMode != MTLStorageModeShared ||
      destination.storageMode != MTLStorageModeShared ||
      sourceOffset % sizeof(float) != 0 ||
      destinationOffset % sizeof(float) != 0 ||
      byteCount % sizeof(float) != 0 ||
      !wan_attention_mps_ranges_disjoint(source, sourceOffset, destination,
                                         destinationOffset, byteCount))
    return -2;
  @try {
    id<MTLBlitCommandEncoder> blit = [root blitCommandEncoder];
    if (!blit)
      return -3;
    [blit copyFromBuffer:source
            sourceOffset:(NSUInteger)sourceOffset
                toBuffer:destination
       destinationOffset:(NSUInteger)destinationOffset
                    size:(NSUInteger)byteCount];
    [blit endEncoding];
    return 0;
  } @catch (NSException *exception) {
    (void)exception;
    return -4;
  }
}

extern "C" void gs_wan_attention_mps_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}
