// Diagnostic-only DCAE Conv3D reduction probe. No production graph changes.
// Multi-target BF16 graph exposes pre-bias and post-bias tensors from one
// executable. The F32 graph casts the identical BF16 causal input/weights up
// before Conv3D and exposes the unrounded convolution result.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <stdint.h>
#include <stdio.h>

@interface DCAEReductionTemplate : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensor *> *feedTensors;
@property(nonatomic, strong) NSArray<MPSGraphTensor *> *logicalFeedTensors;
@property(nonatomic, strong) NSArray<NSArray<NSNumber *> *> *feedShapes;
@property(nonatomic, strong) NSArray<NSNumber *> *feedTypes;
@property(nonatomic, strong) NSArray<MPSGraphTensor *> *targets;
@property(nonatomic, strong) MPSGraphTensor *prebiasTarget;
@property(nonatomic, strong) MPSGraphTensor *postbiasTarget;
@property(nonatomic, strong) NSArray<NSArray<NSNumber *> *> *targetShapes;
@property(nonatomic, strong) NSArray<NSNumber *> *targetTypes;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic) int32_t mode;
@property(nonatomic) BOOL cacheInput;
@property(nonatomic) int32_t height;
@property(nonatomic) int32_t width;
@end
@implementation DCAEReductionTemplate
@end

@interface DCAEReductionPlan : NSObject
@property(nonatomic, strong) DCAEReductionTemplate *templatePlan;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputs;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputs;
@end
@implementation DCAEReductionPlan
@end

static NSArray<NSNumber *> *dcae_reduction_source_shape(int32_t height, int32_t width) {
  return @[@1, @512, @40, @(height), @(width)];
}
static NSArray<NSNumber *> *dcae_reduction_cache_shape(int32_t height, int32_t width) {
  return @[@1, @512, @1, @(height), @(width)];
}
static NSArray<NSNumber *> *dcae_reduction_weight_shape(void) {
  return @[@512, @512, @3, @3, @3];
}
static NSArray<NSNumber *> *dcae_reduction_bias_shape(void) {
  return @[@1, @512, @1, @1, @1];
}
static NSArray<NSNumber *> *dcae_reduction_output_shape(int32_t height, int32_t width) {
  return @[@1, @512, @40, @(height), @(width)];
}

static bool dcae_reduction_disjoint(id<MTLBuffer> a, id<MTLBuffer> b) {
  if (!a || !b || a == b || !a.contents || !b.contents) return false;
  const uintptr_t a0 = (uintptr_t)a.contents, b0 = (uintptr_t)b.contents;
  if (a.length > UINTPTR_MAX - a0 || b.length > UINTPTR_MAX - b0) return false;
  const uintptr_t a1 = a0 + (uintptr_t)a.length, b1 = b0 + (uintptr_t)b.length;
  return a1 <= b0 || b1 <= a0;
}

extern "C" void *gs_dcae_reduction_template_create(void *deviceHandle,
                                                    int32_t mode,
                                                    int32_t cacheInput,
                                                    int32_t height, int32_t width) {
  if (!((height == 3 && width == 5) || (height == 72 && width == 128)) ||
      !deviceHandle || (mode != 0 && mode != 1) ||
      (cacheInput != 0 && cacheInput != 1)) return nullptr;
  if (@available(macOS 13.2, *)) {
    @autoreleasepool {
      @try {
        id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
        MPSGraph *graph = [MPSGraph new];
        graph.options = MPSGraphOptionsNone;
        NSArray<NSNumber *> *sourceShape = dcae_reduction_source_shape(height, width);
        NSArray<NSNumber *> *cacheShape = dcae_reduction_cache_shape(height, width);
        NSArray<NSNumber *> *weightShape = dcae_reduction_weight_shape();
        NSArray<NSNumber *> *biasShape = dcae_reduction_bias_shape();
        NSArray<NSNumber *> *outputShape = dcae_reduction_output_shape(height, width);
        MPSGraphTensor *source = [graph placeholderWithShape:sourceShape
            dataType:MPSDataTypeBFloat16 name:@"boundary_source_bf16"];
        MPSGraphTensor *weight = [graph placeholderWithShape:weightShape
            dataType:MPSDataTypeBFloat16 name:@"boundary_weight_bf16"];
        MPSGraphTensor *bias = [graph placeholderWithShape:biasShape
            dataType:MPSDataTypeBFloat16 name:@"boundary_bias_bf16"];
        MPSGraphTensor *cache = cacheInput
            ? [graph placeholderWithShape:cacheShape dataType:MPSDataTypeBFloat16
                                    name:@"boundary_cache_bf16"] : nil;
        MPSGraphTensor *zero = [graph constantWithScalar:0.0 shape:cacheShape
            dataType:MPSDataTypeBFloat16];
        MPSGraphTensor *left = cacheInput ? cache : zero;
        MPSGraphTensor *joined = (source && left && zero)
            ? [graph concatTensors:@[left, source, zero] dimension:2
                               name:@"boundary_causal_concat_bf16"] : nil;
        MPSGraphTensor *convSource = joined;
        MPSGraphTensor *convWeight = weight;
        MPSDataType outputType = MPSDataTypeBFloat16;
        if (mode == 1) {
          convSource = joined ? [graph castTensor:joined toType:MPSDataTypeFloat32
                                               name:@"boundary_causal_concat_f32"] : nil;
          convWeight = weight ? [graph castTensor:weight toType:MPSDataTypeFloat32
                                               name:@"boundary_weight_f32"] : nil;
          outputType = MPSDataTypeFloat32;
        }
        MPSGraphConvolution3DOpDescriptor *descriptor =
            [MPSGraphConvolution3DOpDescriptor
                descriptorWithStrideInX:1 strideInY:1 strideInZ:1
                dilationRateInX:1 dilationRateInY:1 dilationRateInZ:1
                groups:1 paddingLeft:1 paddingRight:1
                paddingTop:1 paddingBottom:1 paddingFront:0 paddingBack:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNCDHW
                weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
        MPSGraphTensor *convolution = (descriptor && convSource && convWeight)
            ? [graph convolution3DWithSourceTensor:convSource
                                      weightsTensor:convWeight
                                         descriptor:descriptor
                                               name:mode == 0 ? @"boundary_conv3d_bf16" : @"boundary_conv3d_f32"]
            : nil;
        MPSGraphTensor *postBias = mode == 0 && convolution
            ? [graph additionWithPrimaryTensor:convolution secondaryTensor:bias
                                          name:@"boundary_bias_add_bf16"] : nil;
        NSArray<MPSGraphTensor *> *targets = mode == 0
            ? (convolution && postBias ? @[convolution, postBias] : nil)
            : (convolution ? @[convolution] : nil);
        if (!targets || targets.count != (mode == 0 ? 2u : 1u) ||
            ![convolution.shape isEqualToArray:outputShape] ||
            convolution.dataType != outputType ||
            (postBias && (![postBias.shape isEqualToArray:outputShape] ||
                          postBias.dataType != MPSDataTypeBFloat16))) return nullptr;

        NSMutableArray<MPSGraphTensor *> *feedTensors = [NSMutableArray array];
        NSMutableArray<NSArray<NSNumber *> *> *feedShapes = [NSMutableArray array];
        NSMutableArray<NSNumber *> *feedTypes = [NSMutableArray array];
        [feedTensors addObject:source]; [feedShapes addObject:sourceShape];
        [feedTypes addObject:@(MPSDataTypeBFloat16)];
        if (cacheInput) {
          [feedTensors addObject:cache]; [feedShapes addObject:cacheShape];
          [feedTypes addObject:@(MPSDataTypeBFloat16)];
        }
        [feedTensors addObject:weight]; [feedShapes addObject:weightShape];
        [feedTypes addObject:@(MPSDataTypeBFloat16)];
        if (mode == 0) {
          [feedTensors addObject:bias]; [feedShapes addObject:biasShape];
          [feedTypes addObject:@(MPSDataTypeBFloat16)];
        }
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feedMap =
            [NSMutableDictionary dictionary];
        for (NSUInteger i = 0; i < feedTensors.count; ++i) {
          MPSGraphShapedType *type = [[MPSGraphShapedType alloc]
              initWithShape:feedShapes[i] dataType:(MPSDataType)feedTypes[i].intValue];
          if (!type) return nullptr;
          feedMap[feedTensors[i]] = type;
        }
        MPSGraphExecutable *executable = [graph compileWithDevice:
            [MPSGraphDevice deviceWithMTLDevice:device] feeds:feedMap
            targetTensors:targets targetOperations:nil compilationDescriptor:nil];
        if (!executable || executable.feedTensors.count != feedTensors.count ||
            executable.targetTensors.count != targets.count) return nullptr;
        NSMutableIndexSet *seen = [NSMutableIndexSet indexSet];
        NSMutableArray<NSArray<NSNumber *> *> *orderedShapes = [NSMutableArray array];
        NSMutableArray<NSNumber *> *orderedTypes = [NSMutableArray array];
        for (MPSGraphTensor *tensor in executable.feedTensors) {
          NSUInteger index = [feedTensors indexOfObjectIdenticalTo:tensor];
          if (index == NSNotFound || [seen containsIndex:index] ||
              tensor.dataType != (MPSDataType)feedTypes[index].intValue ||
              ![tensor.shape isEqualToArray:feedShapes[index]]) return nullptr;
          [seen addIndex:index];
          [orderedShapes addObject:feedShapes[index]];
          [orderedTypes addObject:feedTypes[index]];
        }
        if (seen.count != feedTensors.count) return nullptr;
        NSMutableIndexSet *seenTargets = [NSMutableIndexSet indexSet];
        NSMutableArray<NSArray<NSNumber *> *> *orderedTargetShapes = [NSMutableArray array];
        NSMutableArray<NSNumber *> *orderedTargetTypes = [NSMutableArray array];
        NSArray<NSArray<NSNumber *> *> *logicalTargetShapes = mode == 0
            ? @[outputShape, outputShape] : @[outputShape];
        NSArray<NSNumber *> *logicalTargetTypes = mode == 0
            ? @[@(MPSDataTypeBFloat16), @(MPSDataTypeBFloat16)]
            : @[@(MPSDataTypeFloat32)];
        for (MPSGraphTensor *tensor in executable.targetTensors) {
          NSUInteger index = [targets indexOfObjectIdenticalTo:tensor];
          if (index == NSNotFound || [seenTargets containsIndex:index] ||
              tensor.dataType != (MPSDataType)logicalTargetTypes[index].intValue ||
              ![tensor.shape isEqualToArray:logicalTargetShapes[index]]) return nullptr;
          [seenTargets addIndex:index];
          [orderedTargetShapes addObject:logicalTargetShapes[index]];
          [orderedTargetTypes addObject:logicalTargetTypes[index]];
        }
        if (seenTargets.count != targets.count) return nullptr;
        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        DCAEReductionTemplate *plan = [DCAEReductionTemplate new];
        plan.device = device; plan.graph = graph; plan.executable = executable;
        plan.feedTensors = [executable.feedTensors copy];
        plan.logicalFeedTensors = [feedTensors copy];
        plan.feedShapes = [orderedShapes copy]; plan.feedTypes = [orderedTypes copy];
        // MPSGraphExecutable documents targetTensors as the output-ordering
        // authority. Keep both metadata and bound result data in that order.
        plan.targets = [executable.targetTensors copy];
        plan.prebiasTarget = convolution;
        plan.postbiasTarget = postBias;
        plan.targetShapes = [orderedTargetShapes copy];
        plan.targetTypes = [orderedTargetTypes copy];
        plan.execution = execution; plan.mode = mode; plan.cacheInput = cacheInput != 0;
        plan.height = height; plan.width = width;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        fprintf(stderr, "DCAE boundary graph setup failed: %s\n",
                exception.reason.UTF8String ?: "unknown exception");
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" void *gs_dcae_reduction_plan_create(void *templateHandle,
    void *inputHandle, void *cacheHandle, void *weightHandle, void *biasHandle,
    void *output0Handle, void *output1Handle) {
  if (!templateHandle || !inputHandle || !weightHandle || !output0Handle) return nullptr;
  @autoreleasepool {
    @try {
      DCAEReductionTemplate *t = (__bridge DCAEReductionTemplate *)templateHandle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
      id<MTLBuffer> cache = cacheHandle ? (__bridge id<MTLBuffer>)cacheHandle : nil;
      id<MTLBuffer> weight = (__bridge id<MTLBuffer>)weightHandle;
      id<MTLBuffer> bias = biasHandle ? (__bridge id<MTLBuffer>)biasHandle : nil;
      id<MTLBuffer> out0 = (__bridge id<MTLBuffer>)output0Handle;
      id<MTLBuffer> out1 = output1Handle ? (__bridge id<MTLBuffer>)output1Handle : nil;
      if (!t) return nullptr;
      const uint64_t values = 1ULL * 512 * 40 * t.height * t.width;
      const uint64_t inputBytes = values * 2;
      const uint64_t cacheBytes = 512ULL * t.height * t.width * 2;
      const uint64_t weightBytes = 512ULL * 512 * 27 * 2;
      const uint64_t biasBytes = 512ULL * 2;
      const uint64_t outBytes = values * (t.mode == 0 ? 2ULL : 4ULL);
      if (!t || input.device != t.device || weight.device != t.device ||
          out0.device != t.device || input.length < inputBytes ||
          weight.length < weightBytes || out0.length < outBytes ||
          (t.cacheInput && (!cache || cache.device != t.device || cache.length < cacheBytes)) ||
          (!t.cacheInput && cache) ||
          (t.mode == 0 && (!bias || bias.device != t.device || bias.length < biasBytes ||
                           !out1 || out1.device != t.device || out1.length < outBytes)) ||
          (t.mode == 1 && (bias || out1))) return nullptr;
      id<MTLBuffer> buffers[6] = {input, cache, weight, bias, out0, out1};
      for (NSUInteger i = 0; i < 6; ++i) {
        if (!buffers[i]) continue;
        for (NSUInteger j = i + 1; j < 6; ++j) {
          if (buffers[j] && !dcae_reduction_disjoint(buffers[i], buffers[j])) return nullptr;
        }
      }
      MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:input shape:dcae_reduction_source_shape(t.height, t.width)
                   dataType:MPSDataTypeBFloat16];
      MPSGraphTensorData *weightData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:weight shape:dcae_reduction_weight_shape()
                   dataType:MPSDataTypeBFloat16];
      MPSGraphTensorData *cacheData = t.cacheInput ? [[MPSGraphTensorData alloc]
          initWithMTLBuffer:cache shape:dcae_reduction_cache_shape(t.height, t.width)
                   dataType:MPSDataTypeBFloat16] : nil;
      MPSGraphTensorData *biasData = t.mode == 0 ? [[MPSGraphTensorData alloc]
          initWithMTLBuffer:bias shape:dcae_reduction_bias_shape()
                   dataType:MPSDataTypeBFloat16] : nil;
      MPSGraphTensorData *out0Data = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:out0 shape:dcae_reduction_output_shape(t.height, t.width)
                   dataType:t.mode == 0 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32];
      MPSGraphTensorData *out1Data = t.mode == 0 ? [[MPSGraphTensorData alloc]
          initWithMTLBuffer:out1 shape:dcae_reduction_output_shape(t.height, t.width)
                   dataType:MPSDataTypeBFloat16] : nil;
      if (!inputData || !weightData || !out0Data ||
          (t.cacheInput && !cacheData) || (t.mode == 0 && (!biasData || !out1Data))) return nullptr;
      NSMutableArray<MPSGraphTensorData *> *logical = [NSMutableArray array];
      NSMutableArray<MPSGraphTensor *> *logicalTensors = [NSMutableArray array];
      [logical addObject:inputData]; [logicalTensors addObject:t.logicalFeedTensors[0]];
      NSUInteger index = 1;
      if (t.cacheInput) { [logical addObject:cacheData]; [logicalTensors addObject:t.logicalFeedTensors[index++]]; }
      [logical addObject:weightData]; [logicalTensors addObject:t.logicalFeedTensors[index++]];
      if (t.mode == 0) { [logical addObject:biasData]; [logicalTensors addObject:t.logicalFeedTensors[index++]]; }
      if (index != t.logicalFeedTensors.count ||
          t.feedTensors.count != t.logicalFeedTensors.count) return nullptr;
      NSMutableArray<MPSGraphTensorData *> *ordered = [NSMutableArray array];
      for (MPSGraphTensor *feed in t.executable.feedTensors) {
        NSUInteger logicalIndex = [logicalTensors indexOfObjectIdenticalTo:feed];
        if (logicalIndex == NSNotFound) return nullptr;
        [ordered addObject:logical[logicalIndex]];
      }
      NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *targetData =
          [NSMutableDictionary dictionary];
      if (!t.prebiasTarget || (t.mode == 0 && !t.postbiasTarget) ||
          (t.mode == 1 && t.postbiasTarget)) return nullptr;
      targetData[t.prebiasTarget] = out0Data;
      if (t.mode == 0) targetData[t.postbiasTarget] = out1Data;
      NSMutableArray<MPSGraphTensorData *> *outputs = [NSMutableArray array];
      for (MPSGraphTensor *target in t.executable.targetTensors) {
        MPSGraphTensorData *data = targetData[target];
        if (!data) return nullptr;
        [outputs addObject:data];
      }
      DCAEReductionPlan *plan = [DCAEReductionPlan new];
      plan.templatePlan = t; plan.inputs = [ordered copy]; plan.outputs = [outputs copy];
      return (__bridge_retained void *)plan;
    } @catch (NSException *exception) {
      (void)exception;
      return nullptr;
    }
  }
}

extern "C" int32_t gs_dcae_reduction_encode(void *planHandle,
                                             void *commandHandle) {
  if (!planHandle || !commandHandle) return -1;
  @autoreleasepool {
    @try {
      DCAEReductionPlan *plan = (__bridge DCAEReductionPlan *)planHandle;
      MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)commandHandle;
      if (!plan || !plan.templatePlan.executable || !command) return -2;
      NSArray<MPSGraphTensorData *> *result =
          [plan.templatePlan.executable encodeToCommandBuffer:command
                                                  inputsArray:plan.inputs
                                                 resultsArray:plan.outputs
                                          executionDescriptor:plan.templatePlan.execution];
      return result.count == plan.outputs.count ? 0 : -3;
    } @catch (NSException *exception) {
      (void)exception;
      return -4;
    }
  }
}

extern "C" void gs_dcae_reduction_plan_release(void *handle) {
  if (handle) CFBridgingRelease(handle);
}
extern "C" void gs_dcae_reduction_template_release(void *handle) {
  if (handle) CFBridgingRelease(handle);
}
