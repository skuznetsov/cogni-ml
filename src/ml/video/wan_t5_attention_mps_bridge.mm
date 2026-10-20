// Prepared same-input Wan T5 attention diagnostic. It compares MPSGraph's
// batched QK^T, additive bias, F32 softmax, and PV against the pinned MPS
// oracle while consuming the exact resident Q/K/V and attention-bias buffers.
// The caller owns all buffers and serial command submission. Graph internals
// may use private temporary allocations that wrapper accounting cannot see.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <limits>
#include <stdint.h>

@interface CogniWanT5AttentionGraphPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@end

@implementation CogniWanT5AttentionGraphPlan
@end

static bool wan_t5_attention_product(const uint64_t *values, size_t count,
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

static bool wan_t5_attention_disjoint(id<MTLBuffer> a, id<MTLBuffer> b) {
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

extern "C" void *gs_wan_t5_attention_mps_plan_create(
    void *deviceHandle, int32_t batches, int32_t tokens, int32_t heads,
    int32_t headDim, void *qHandle, void *kHandle, void *vHandle,
    void *biasHandle, void *logitsHandle, void *scoresHandle,
    void *probabilitiesHandle, void *contextHandle) {
  if (!deviceHandle || !qHandle || !kHandle || !vHandle || !biasHandle ||
      !logitsHandle || !scoresHandle || !probabilitiesHandle || !contextHandle)
    return nullptr;
  if (@available(macOS 12.0, *)) {
    @autoreleasepool {
      if (batches <= 0 || tokens <= 0 || heads <= 0 || headDim <= 0)
        return nullptr;
      const uint64_t qShape[] = {(uint64_t)batches, (uint64_t)tokens,
                                 (uint64_t)heads, (uint64_t)headDim};
      const uint64_t scoreShape[] = {(uint64_t)batches, (uint64_t)heads,
                                     (uint64_t)tokens, (uint64_t)tokens};
      uint64_t qElements = 0, scoreElements = 0;
      if (!wan_t5_attention_product(qShape, 4, &qElements) ||
          !wan_t5_attention_product(scoreShape, 4, &scoreElements) ||
          qElements > (uint64_t)INT32_MAX ||
          scoreElements > (uint64_t)INT32_MAX ||
          qElements > UINT64_MAX / sizeof(float) ||
          scoreElements > UINT64_MAX / sizeof(float))
        return nullptr;
      const uint64_t qBytes = qElements * sizeof(float);
      const uint64_t scoreBytes = scoreElements * sizeof(float);

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> qBuffer = (__bridge id<MTLBuffer>)qHandle;
      id<MTLBuffer> kBuffer = (__bridge id<MTLBuffer>)kHandle;
      id<MTLBuffer> vBuffer = (__bridge id<MTLBuffer>)vHandle;
      id<MTLBuffer> biasBuffer = (__bridge id<MTLBuffer>)biasHandle;
      id<MTLBuffer> logitsBuffer = (__bridge id<MTLBuffer>)logitsHandle;
      id<MTLBuffer> scoresBuffer = (__bridge id<MTLBuffer>)scoresHandle;
      id<MTLBuffer> probabilitiesBuffer =
          (__bridge id<MTLBuffer>)probabilitiesHandle;
      id<MTLBuffer> contextBuffer = (__bridge id<MTLBuffer>)contextHandle;

      NSArray<id<MTLBuffer>> *inputs =
          @[ qBuffer, kBuffer, vBuffer, biasBuffer ];
      NSArray<id<MTLBuffer>> *outputs =
          @[ logitsBuffer, scoresBuffer, probabilitiesBuffer, contextBuffer ];
      for (id<MTLBuffer> buffer in
           [inputs arrayByAddingObjectsFromArray:outputs]) {
        if (buffer.device != device)
          return nullptr;
      }
      if (qBuffer.length < qBytes || kBuffer.length < qBytes ||
          vBuffer.length < qBytes || contextBuffer.length < qBytes ||
          biasBuffer.length < scoreBytes || logitsBuffer.length < scoreBytes ||
          scoresBuffer.length < scoreBytes ||
          probabilitiesBuffer.length < scoreBytes)
        return nullptr;
      NSArray<id<MTLBuffer>> *allBuffers =
          [inputs arrayByAddingObjectsFromArray:outputs];
      for (NSUInteger i = 0; i < allBuffers.count; ++i) {
        for (NSUInteger j = i + 1; j < allBuffers.count; ++j) {
          if (!wan_t5_attention_disjoint(allBuffers[i], allBuffers[j]))
            return nullptr;
        }
      }

      @try {
        NSArray<NSNumber *> *inputShape =
            @[ @(batches), @(tokens), @(heads), @(headDim) ];
        NSArray<NSNumber *> *scoreTensorShape =
            @[ @(batches), @(heads), @(tokens), @(tokens) ];
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *q = [graph placeholderWithShape:inputShape
                                               dataType:MPSDataTypeFloat32
                                                   name:@"wan_q_bthd"];
        MPSGraphTensor *k = [graph placeholderWithShape:inputShape
                                               dataType:MPSDataTypeFloat32
                                                   name:@"wan_k_bthd"];
        MPSGraphTensor *v = [graph placeholderWithShape:inputShape
                                               dataType:MPSDataTypeFloat32
                                                   name:@"wan_v_bthd"];
        MPSGraphTensor *bias =
            [graph placeholderWithShape:scoreTensorShape
                               dataType:MPSDataTypeFloat32
                                   name:@"wan_attention_bias_bhtt"];
        MPSGraphTensor *qh = [graph transposeTensor:q
                                          dimension:1
                                      withDimension:2
                                               name:@"q_bhtd"];
        MPSGraphTensor *kHeads = [graph transposeTensor:k
                                              dimension:1
                                          withDimension:2
                                                   name:@"k_bhtd"];
        MPSGraphTensor *kh = [graph transposeTensor:kHeads
                                          dimension:2
                                      withDimension:3
                                               name:@"k_bhdt"];
        MPSGraphTensor *vh = [graph transposeTensor:v
                                          dimension:1
                                      withDimension:2
                                               name:@"v_bhtd"];
        // torch.einsum lowers each product to rank-three bmm with the
        // batch/head axes collapsed. Preserve that rank for kernel selection.
        NSArray<NSNumber *> *queryBMMShape =
            @[ @(batches * heads), @(tokens), @(headDim) ];
        NSArray<NSNumber *> *keyBMMShape =
            @[ @(batches * heads), @(headDim), @(tokens) ];
        NSArray<NSNumber *> *scoreBMMShape =
            @[ @(batches * heads), @(tokens), @(tokens) ];
        MPSGraphTensor *q3 = [graph reshapeTensor:qh withShape:queryBMMShape name:nil];
        MPSGraphTensor *k3 = [graph reshapeTensor:kh withShape:keyBMMShape name:nil];
        MPSGraphTensor *v3 = [graph reshapeTensor:vh withShape:queryBMMShape name:nil];
        MPSGraphTensor *logits3 =
            [graph matrixMultiplicationWithPrimaryTensor:q3
                                         secondaryTensor:k3
                                                    name:@"qk_logits_bmm"];
        MPSGraphTensor *logits =
            [graph reshapeTensor:logits3 withShape:scoreTensorShape name:nil];
        MPSGraphTensor *scores =
            [graph additionWithPrimaryTensor:logits
                             secondaryTensor:bias
                                        name:@"biased_scores_bhtt"];
        MPSGraphTensor *probabilities =
            [graph softMaxWithTensor:scores axis:3 name:@"softmax_f32_bhtt"];
        MPSGraphTensor *probabilities3 =
            [graph reshapeTensor:probabilities withShape:scoreBMMShape name:nil];
        MPSGraphTensor *context3 =
            [graph matrixMultiplicationWithPrimaryTensor:probabilities3
                                         secondaryTensor:v3
                                                    name:@"pv_context_bmm"];
        MPSGraphTensor *contextHeads =
            [graph reshapeTensor:context3
                       withShape:@[ @(batches), @(heads), @(tokens), @(headDim) ]
                            name:nil];
        MPSGraphTensor *context = [graph transposeTensor:contextHeads
                                               dimension:1
                                           withDimension:2
                                                    name:@"context_bthd"];
        if (!graph || !q || !k || !v || !bias || !qh || !kHeads || !kh || !vh ||
            !logits || !scores || !probabilities || !contextHeads || !context)
          return nullptr;

        MPSGraphTensorShapedTypeDictionary *feeds = @{
          q : [[MPSGraphShapedType alloc] initWithShape:inputShape
                                               dataType:MPSDataTypeFloat32],
          k : [[MPSGraphShapedType alloc] initWithShape:inputShape
                                               dataType:MPSDataTypeFloat32],
          v : [[MPSGraphShapedType alloc] initWithShape:inputShape
                                               dataType:MPSDataTypeFloat32],
          bias : [[MPSGraphShapedType alloc] initWithShape:scoreTensorShape
                                                  dataType:MPSDataTypeFloat32],
        };
        NSArray<MPSGraphTensor *> *targets =
            @[ logits, scores, probabilities, context ];
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:targets
                     targetOperations:nil
                compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        executable.options = MPSGraphOptionsNone;

        MPSGraphTensorData *qData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:qBuffer
                                                    shape:inputShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *kData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:kBuffer
                                                    shape:inputShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *vData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:vBuffer
                                                    shape:inputShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *biasData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:biasBuffer
                                                    shape:scoreTensorShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *logitsData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:logitsBuffer
                                                    shape:scoreTensorShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *scoresData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:scoresBuffer
                                                    shape:scoreTensorShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *probabilitiesData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:probabilitiesBuffer
                                                    shape:scoreTensorShape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *contextData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:contextBuffer
                                                    shape:inputShape
                                                 dataType:MPSDataTypeFloat32];
        if (!qData || !kData || !vData || !biasData || !logitsData ||
            !scoresData || !probabilitiesData || !contextData)
          return nullptr;

        NSMutableArray<MPSGraphTensorData *> *orderedInputs =
            [NSMutableArray new];
        for (MPSGraphTensor *feedTensor in executable.feedTensors) {
          if (feedTensor == q)
            [orderedInputs addObject:qData];
          else if (feedTensor == k)
            [orderedInputs addObject:kData];
          else if (feedTensor == v)
            [orderedInputs addObject:vData];
          else if (feedTensor == bias)
            [orderedInputs addObject:biasData];
          else
            return nullptr;
        }
        if (orderedInputs.count != 4)
          return nullptr;

        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        CogniWanT5AttentionGraphPlan *plan = [CogniWanT5AttentionGraphPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.inputArray = [orderedInputs copy];
        plan.outputArray =
            @[ logitsData, scoresData, probabilitiesData, contextData ];
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

extern "C" int32_t gs_wan_t5_attention_mps_encode(void *planHandle,
                                                  void *commandBufferHandle) {
  if (!planHandle || !commandBufferHandle)
    return -1;
  if (@available(macOS 12.0, *)) {
    @autoreleasepool {
      CogniWanT5AttentionGraphPlan *plan =
          (__bridge CogniWanT5AttentionGraphPlan *)planHandle;
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
        return result.count == 4 ? 0 : -3;
      } @catch (NSException *exception) {
        (void)exception;
        return -4;
      }
    }
  }
  return -5;
}

extern "C" void gs_wan_t5_attention_mps_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}
