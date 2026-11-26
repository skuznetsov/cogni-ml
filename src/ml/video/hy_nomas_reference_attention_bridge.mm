// Diagnostic BF16 SDPA over the physical first-T0 K/V cache. All bindings and
// graph resources are prepared once; encoding adds no wait or host readback.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>
#include <climits>
#include <cstdint>
#include <initializer_list>

@interface HyNomasReferenceAttentionPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputs;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputs;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *buffers;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@end
@implementation HyNomasReferenceAttentionPlan
@end

extern "C" void *gs_hy_nomas_reference_attention_plan_create(
    void *deviceHandle, int32_t tokens, int32_t heads, int32_t headDim,
    void *queryHandle, void *keyHandle, void *valueHandle, void *outputHandle) {
  if (!deviceHandle || !queryHandle || !keyHandle || !valueHandle || !outputHandle ||
      tokens <= 0 || heads <= 0 || headDim <= 0)
    return nullptr;
  uint64_t count = 1;
  for (int32_t dim : {tokens, heads, headDim}) {
    if (count > (uint64_t)INT32_MAX / (uint64_t)dim)
      return nullptr;
    count *= (uint64_t)dim;
  }
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      NSArray<id<MTLBuffer>> *buffers = @[
        (__bridge id<MTLBuffer>)queryHandle, (__bridge id<MTLBuffer>)keyHandle,
        (__bridge id<MTLBuffer>)valueHandle, (__bridge id<MTLBuffer>)outputHandle,
      ];
      for (NSUInteger index = 0; index < buffers.count; ++index) {
        id<MTLBuffer> buffer = buffers[index];
        if (buffer.device != device || buffer.storageMode != MTLStorageModeShared ||
            buffer.length != count * 2 || !buffer.contents)
          return nullptr;
        const uintptr_t start = (uintptr_t)buffer.contents;
        if (buffer.length > UINTPTR_MAX - start)
          return nullptr;
        for (NSUInteger other = 0; other < index; ++other) {
          const uintptr_t right = (uintptr_t)buffers[other].contents;
          if (!(start + buffer.length <= right || right + buffers[other].length <= start))
            return nullptr;
        }
      }
      @try {
        NSArray<NSNumber *> *shape = @[@1, @(tokens), @(heads), @(headDim)];
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        MPSGraphTensor *q = [graph placeholderWithShape:shape dataType:MPSDataTypeBFloat16 name:@"reference_q_bf16"];
        MPSGraphTensor *k = [graph placeholderWithShape:shape dataType:MPSDataTypeBFloat16 name:@"reference_stored_k_bf16"];
        MPSGraphTensor *v = [graph placeholderWithShape:shape dataType:MPSDataTypeBFloat16 name:@"reference_stored_v_bf16"];
        MPSGraphTensor *qh = [graph transposeTensor:q dimension:1 withDimension:2 name:@"reference_q_bhtd"];
        MPSGraphTensor *kh = [graph transposeTensor:k dimension:1 withDimension:2 name:@"reference_k_bhtd"];
        MPSGraphTensor *vh = [graph transposeTensor:v dimension:1 withDimension:2 name:@"reference_v_bhtd"];
        const float scale = 1.0f / std::sqrt((float)headDim);
        // Match the pinned PyTorch 2.6 MPS math branch. Its BF16 QK and
        // scaled-logit boundaries are observable; the fused SDPA node does
        // not preserve them. Only the scale multiplication is performed in F32.
        MPSGraphTensor *kt = [graph transposeTensor:kh dimension:2 withDimension:3 name:@"reference_k_bhdt"];
        MPSGraphTensor *qk = [graph matrixMultiplicationWithPrimaryTensor:qh
            secondaryTensor:kt name:@"reference_qk_bf16"];
        MPSGraphTensor *qkF32 = [graph castTensor:qk toType:MPSDataTypeFloat32 name:@"reference_qk_f32"];
        MPSGraphTensor *scaleTensor = [graph constantWithScalar:scale shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:qkF32
            secondaryTensor:scaleTensor name:@"reference_scaled_qk_f32"];
        MPSGraphTensor *logits = [graph castTensor:scaled toType:MPSDataTypeBFloat16 name:@"reference_logits_bf16"];
        MPSGraphTensor *probabilities = [graph softMaxWithTensor:logits axis:3 name:@"reference_probabilities_bf16"];
        MPSGraphTensor *headsOutput = [graph matrixMultiplicationWithPrimaryTensor:probabilities
            secondaryTensor:vh name:@"reference_context_bf16"];
        MPSGraphTensor *output = [graph transposeTensor:headsOutput dimension:1 withDimension:2 name:@"reference_output_bthd"];
        NSArray<NSNumber *> *headShape = @[@1, @(heads), @(tokens), @(headDim)];
        if (!output || headsOutput.dataType != MPSDataTypeBFloat16 ||
            output.dataType != MPSDataTypeBFloat16 ||
            ![headsOutput.shape isEqualToArray:headShape] ||
            ![output.shape isEqualToArray:shape])
          return nullptr;
        MPSGraphTensorShapedTypeDictionary *feeds = @{
          q : [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeBFloat16],
          k : [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeBFloat16],
          v : [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeBFloat16],
        };
        MPSGraphExecutable *executable = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
            feeds:feeds targetTensors:@[output] targetOperations:nil compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        executable.options = MPSGraphOptionsNone;
        NSArray<MPSGraphTensor *> *placeholders = @[q, k, v];
        NSMutableArray<MPSGraphTensorData *> *inputs = [NSMutableArray new];
        for (MPSGraphTensor *feed in executable.feedTensors) {
          NSUInteger index = [placeholders indexOfObjectIdenticalTo:feed];
          if (index == NSNotFound || index >= 3)
            return nullptr;
          MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffers[index]
              shape:shape dataType:MPSDataTypeBFloat16];
          if (!data)
            return nullptr;
          [inputs addObject:data];
        }
        MPSGraphTensorData *outData = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffers[3]
            shape:shape dataType:MPSDataTypeBFloat16];
        if (inputs.count != 3 || !outData)
          return nullptr;
        HyNomasReferenceAttentionPlan *plan = [HyNomasReferenceAttentionPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.inputs = [inputs copy];
        plan.outputs = @[outData];
        plan.buffers = buffers;
        plan.execution = [MPSGraphExecutableExecutionDescriptor new];
        plan.execution.waitUntilCompleted = NO;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_hy_nomas_reference_attention_encode(void *planHandle, void *commandHandle) {
  if (!planHandle || !commandHandle)
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      HyNomasReferenceAttentionPlan *plan = (__bridge HyNomasReferenceAttentionPlan *)planHandle;
      MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)commandHandle;
      id<MTLCommandBuffer> root = command.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device || root.status >= MTLCommandBufferStatusCommitted)
        return -2;
      @try {
        NSArray<MPSGraphTensorData *> *result = [plan.executable encodeToCommandBuffer:command
            inputsArray:plan.inputs resultsArray:plan.outputs executionDescriptor:plan.execution];
        return result.count == 1 ? 0 : -3;
      } @catch (NSException *exception) {
        (void)exception;
        return -4;
      }
    }
  }
  return -5;
}

extern "C" void gs_hy_nomas_reference_attention_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}
