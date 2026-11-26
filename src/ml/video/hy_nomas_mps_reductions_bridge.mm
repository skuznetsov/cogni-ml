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
@property(nonatomic) BOOL captureProbabilities;
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

// A BF16-typed result cast back to F32 can remain unrounded in a compiled
// graph. Make the observable RNE boundary explicit without a CPU read or an
// intermediate external buffer. Preserve infinities and quiet NaNs.
static MPSGraphTensor *hy_nomas_mps_bf16_round_expand(
    MPSGraph *graph, MPSGraphTensor *input) {
  if (!graph || !input || input.dataType != MPSDataTypeFloat32)
    return nil;
  MPSGraphTensor *bits = [graph reinterpretCastTensor:input
                                              toType:MPSDataTypeUInt32
                                                name:@"hy_context_bits"];
  MPSGraphTensor *shift = [graph constantWithScalar:16.0 shape:@[ @1 ]
                                         dataType:MPSDataTypeUInt32];
  MPSGraphTensor *one = [graph constantWithScalar:1.0 shape:@[ @1 ]
                                       dataType:MPSDataTypeUInt32];
  MPSGraphTensor *base = [graph constantWithScalar:0x7fff shape:@[ @1 ]
                                        dataType:MPSDataTypeUInt32];
  MPSGraphTensor *upperMask = [graph constantWithScalar:0xffff0000 shape:@[ @1 ]
                                             dataType:MPSDataTypeUInt32];
  MPSGraphTensor *magnitudeMask = [graph constantWithScalar:0x7fffffff shape:@[ @1 ]
                                                 dataType:MPSDataTypeUInt32];
  MPSGraphTensor *infinity = [graph constantWithScalar:0x7f800000 shape:@[ @1 ]
                                            dataType:MPSDataTypeUInt32];
  MPSGraphTensor *quiet = [graph constantWithScalar:0x00400000 shape:@[ @1 ]
                                         dataType:MPSDataTypeUInt32];
  if (!bits || !shift || !one || !base || !upperMask || !magnitudeMask ||
      !infinity || !quiet || bits.dataType != MPSDataTypeUInt32)
    return nil;
  MPSGraphTensor *high = [graph bitwiseRightShiftWithPrimaryTensor:bits
                                                secondaryTensor:shift
                                                           name:@"hy_context_high_word"];
  MPSGraphTensor *parity = [graph bitwiseANDWithPrimaryTensor:high
                                            secondaryTensor:one
                                                       name:@"hy_context_round_parity"];
  MPSGraphTensor *bias = [graph additionWithPrimaryTensor:base
                                         secondaryTensor:parity
                                                    name:@"hy_context_round_bias"];
  MPSGraphTensor *rounded = [graph additionWithPrimaryTensor:bits
                                            secondaryTensor:bias
                                                       name:@"hy_context_rounded_bits"];
  MPSGraphTensor *masked = [graph bitwiseANDWithPrimaryTensor:rounded
                                            secondaryTensor:upperMask
                                                       name:@"hy_context_upper_bits"];
  MPSGraphTensor *magnitude = [graph bitwiseANDWithPrimaryTensor:bits
                                               secondaryTensor:magnitudeMask
                                                          name:@"hy_context_magnitude"];
  MPSGraphTensor *isNaN = [graph greaterThanWithPrimaryTensor:magnitude
                                            secondaryTensor:infinity
                                                       name:@"hy_context_is_nan"];
  MPSGraphTensor *nanBase = [graph bitwiseANDWithPrimaryTensor:bits
                                             secondaryTensor:upperMask
                                                        name:@"hy_context_nan_base"];
  MPSGraphTensor *nanBits = [graph bitwiseORWithPrimaryTensor:nanBase
                                            secondaryTensor:quiet
                                                       name:@"hy_context_quiet_nan"];
  MPSGraphTensor *expanded = [graph selectWithPredicateTensor:isNaN
                                        truePredicateTensor:nanBits
                                       falsePredicateTensor:masked
                                                        name:@"hy_context_bf16_bits"];
  if (!high || !parity || !bias || !rounded || !masked || !magnitude ||
      !isNaN || !nanBase || !nanBits || !expanded ||
      high.dataType != MPSDataTypeUInt32 || parity.dataType != MPSDataTypeUInt32 ||
      bias.dataType != MPSDataTypeUInt32 || rounded.dataType != MPSDataTypeUInt32 ||
      masked.dataType != MPSDataTypeUInt32 || magnitude.dataType != MPSDataTypeUInt32 ||
      nanBase.dataType != MPSDataTypeUInt32 || nanBits.dataType != MPSDataTypeUInt32 ||
      expanded.dataType != MPSDataTypeUInt32)
    return nil;
  MPSGraphTensor *result = [graph reinterpretCastTensor:expanded
                                                toType:MPSDataTypeFloat32
                                                  name:@"hy_context_bf16_expanded"];
  return result && result.dataType == MPSDataTypeFloat32 ? result : nil;
}

static void *hy_nomas_mps_reduction_plan_create(
    void *deviceHandle, int32_t mode, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, float scale,
    void *inputAHandle, void *inputBHandle, void *inputCHandle,
    void *packedAHandle, void *packedBHandle, void *packedCHandle,
    void *packedOutputHandle, void *outputHandle,
    void *packedProbabilitiesHandle, void *probabilitiesHandle) {
  const bool attention = mode == 3 || mode == 4;
  const bool captureProbabilities = probabilitiesHandle != nullptr;
  if (!deviceHandle || !inputAHandle || !packedAHandle ||
      !packedOutputHandle || !outputHandle || mode < 0 || mode > 4 ||
      batches <= 0 || queryRows <= 0 || keyRows <= 0 || heads <= 0 ||
      headDim <= 0 || !std::isfinite(scale) || scale <= 0.0f || scale > 1.0f)
    return nullptr;
  if ((mode == 0 || mode == 2) &&
      (!inputBHandle || !packedBHandle))
    return nullptr;
  if (attention && (!inputBHandle || !inputCHandle || !packedBHandle ||
                    !packedCHandle))
    return nullptr;
  if (mode == 1 && (inputBHandle || inputCHandle || packedBHandle ||
                    packedCHandle))
    return nullptr;
  if (!attention && inputCHandle)
    return nullptr;
  if ((packedProbabilitiesHandle != nullptr) != captureProbabilities ||
      (captureProbabilities && mode != 4))
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
      id<MTLBuffer> packedProbabilities = packedProbabilitiesHandle
          ? (__bridge id<MTLBuffer>)packedProbabilitiesHandle : nil;
      id<MTLBuffer> probabilitiesOutput = probabilitiesHandle
          ? (__bridge id<MTLBuffer>)probabilitiesHandle : nil;

      uint64_t inputABytes = (mode == 1 || mode == 2) ? scoreBytes : qBytes;
      uint64_t inputBBytes =
          (mode == 0 || mode == 2 || attention) ? kBytes : 0;
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
      if (captureProbabilities) {
        [allBuffers addObject:packedProbabilities];
        [allBuffers addObject:probabilitiesOutput];
      }
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
                      packedC.length < packedCBytes)) ||
          (captureProbabilities &&
              (packedProbabilities.length < scoreBytes ||
               probabilitiesOutput.length < scoreBytes)))
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
        MPSGraphTensor *probabilitiesResult = nil;

        if (mode == 0 || attention) {
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
          if (attention) {
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
            if (mode == 4) {
              // Opt-in refiner boundary control. Round scaled scores, not
              // raw QK: the latter is a refuted boundary on the pinned input.
              softmaxInput = [graph castTensor:softmaxInput
                                        toType:MPSDataTypeBFloat16
                                          name:@"hy_scaled_qk_bf16_bhqk"];
              if (!softmaxInput ||
                  softmaxInput.dataType != MPSDataTypeBFloat16)
                return nullptr;
            }
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
            MPSGraphTensor *pvValues = v;
            if (mode == 4) {
              pvValues = [graph castTensor:v toType:MPSDataTypeBFloat16
                                      name:@"hy_v_bf16_bhkd"];
              if (!probabilities || !pvValues ||
                  probabilities.dataType != MPSDataTypeBFloat16 ||
                  pvValues.dataType != MPSDataTypeBFloat16)
                return nullptr;
              if (captureProbabilities) {
                probabilitiesResult = [graph castTensor:probabilities
                                                  toType:MPSDataTypeFloat32
                                                    name:@"hy_probabilities_f32_bhqk"];
              }
            }
            result =
                [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                            secondaryTensor:pvValues
                                                       name:@"hy_context_bhqd"];
            if (mode == 4) {
              if (!result || result.dataType != MPSDataTypeBFloat16)
                return nullptr;
              result = [graph castTensor:result toType:MPSDataTypeFloat32
                                     name:@"hy_context_f32_bhqd"];
              result = hy_nomas_mps_bf16_round_expand(graph, result);
            }
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

        if (!graph || !result || (captureProbabilities && !probabilitiesResult) ||
            placeholders.count != inputData.count ||
            feeds.count != placeholders.count)
          return nullptr;
        NSArray<NSNumber *> *headOutputShape =
            (mode <= 1) ? headScoreShape : headQShape;
        NSArray<MPSGraphTensor *> *targets = captureProbabilities
            ? @[ result, probabilitiesResult ] : @[ result ];
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:targets
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
        NSMutableArray<MPSGraphTensorData *> *outputs =
            [NSMutableArray arrayWithObject:outputData];
        if (captureProbabilities) {
          MPSGraphTensorData *probabilitiesData =
              [[MPSGraphTensorData alloc] initWithMTLBuffer:packedProbabilities
                                                    shape:headScoreShape
                                                 dataType:MPSDataTypeFloat32];
          if (!probabilitiesData)
            return nullptr;
          [outputs addObject:probabilitiesData];
        }

        NSMutableArray<MPSGraphTensorData *> *orderedInputs = [NSMutableArray new];
        for (MPSGraphTensor *feedTensor in executable.feedTensors) {
          NSUInteger index = [placeholders indexOfObjectIdenticalTo:feedTensor];
          if (index == NSNotFound || index >= inputData.count)
            return nullptr;
          [orderedInputs addObject:inputData[index]];
        }
        if (orderedInputs.count != placeholders.count)
          return nullptr;
        // Preserve the executable's declared target order just as for feeds.
        NSMutableArray<MPSGraphTensorData *> *orderedOutputs = [NSMutableArray new];
        for (MPSGraphTensor *targetTensor in executable.targetTensors) {
          NSUInteger index = [targets indexOfObjectIdenticalTo:targetTensor];
          if (index == NSNotFound || index >= outputs.count)
            return nullptr;
          [orderedOutputs addObject:outputs[index]];
        }
        if (orderedOutputs.count != targets.count)
          return nullptr;

        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        HyNomasMPSReductionPlan *plan = [HyNomasMPSReductionPlan new];
        plan.device = device;
        plan.graph = graph;
        plan.executable = executable;
        plan.inputArray = [orderedInputs copy];
        plan.outputArray = [orderedOutputs copy];
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
        plan.captureProbabilities = captureProbabilities;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

// Existing F32 callers retain the same ABI, mode and graph targets.
extern "C" void *gs_hy_nomas_mps_reduction_plan_create(
    void *deviceHandle, int32_t mode, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, float scale,
    void *inputAHandle, void *inputBHandle, void *inputCHandle,
    void *packedAHandle, void *packedBHandle, void *packedCHandle,
    void *packedOutputHandle, void *outputHandle) {
  return hy_nomas_mps_reduction_plan_create(
      deviceHandle, mode, batches, queryRows, keyRows, heads, headDim, scale,
      inputAHandle, inputBHandle, inputCHandle, packedAHandle, packedBHandle,
      packedCHandle, packedOutputHandle, outputHandle, nullptr, nullptr);
}

// Optional P capture is diagnostic only. Context-only and instrumented plans
// are compared on the same inputs before using that witness to qualify P.
extern "C" void *gs_hy_nomas_mps_bf16_attention_plan_create(
    void *deviceHandle, int32_t batches, int32_t queryRows, int32_t keyRows,
    int32_t heads, int32_t headDim, float scale,
    void *inputAHandle, void *inputBHandle, void *inputCHandle,
    void *packedAHandle, void *packedBHandle, void *packedCHandle,
    void *packedOutputHandle, void *outputHandle,
    void *packedProbabilitiesHandle, void *probabilitiesHandle) {
  return hy_nomas_mps_reduction_plan_create(
      deviceHandle, 4, batches, queryRows, keyRows, heads, headDim, scale,
      inputAHandle, inputBHandle, inputCHandle, packedAHandle, packedBHandle,
      packedCHandle, packedOutputHandle, outputHandle,
      packedProbabilitiesHandle, probabilitiesHandle);
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
      } else if (plan.mode == 3 || plan.mode == 4) {
        inputB = plan.residentBuffers[cursor++];
        packedB = plan.residentBuffers[cursor++];
        inputC = plan.residentBuffers[cursor++];
        packedC = plan.residentBuffers[cursor++];
      }
      id<MTLBuffer> packedOutput = plan.residentBuffers[cursor++];
      id<MTLBuffer> output = plan.residentBuffers[cursor++];
      id<MTLBuffer> packedProbabilities = nil;
      id<MTLBuffer> probabilitiesOutput = nil;
      if (plan.captureProbabilities) {
        packedProbabilities = plan.residentBuffers[cursor++];
        probabilitiesOutput = plan.residentBuffers[cursor++];
      }
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
            plan.mode == 0 || plan.mode == 2 || plan.mode == 3 || plan.mode == 4
                ? plan.keyRows
                : plan.queryRows,
            plan.heads,
            plan.mode == 0 || plan.mode == 2 || plan.mode == 3 || plan.mode == 4
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
        if (result.count != (plan.captureProbabilities ? 2 : 1))
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
        if (plan.captureProbabilities) {
          HyNomasTransposeParams probabilityParams = {
              plan.batches, plan.queryRows, plan.heads, plan.keyRows};
          uint64_t probabilityCount = (uint64_t)plan.batches * plan.queryRows *
                                       plan.heads * plan.keyRows;
          if (!hy_nomas_mps_encode_transpose(root, plan.toStreamMajor,
                                             packedProbabilities,
                                             probabilitiesOutput,
                                             probabilityParams, probabilityCount))
            return -12;
        }
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

// Pinned PyTorch 2.8 MPSGraph activation reproductions. Keep these unary plans
// independent from attention reductions: SiLU is negative -> exp -> add-one
// -> divide, while tanh GELU follows the exact upstream operation sequence
// with rank-one F32 scalar constants on the F32 path. Physical BF16 GELU binds
// BF16 tensors and constants directly. BF16 SiLU mode keeps its nodes and
// scalar constant in BF16 between casts.
@interface HyNomasMPSActivationPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic) int32_t activation;
@property(nonatomic) BOOL bf16Control;
@end

@implementation HyNomasMPSActivationPlan
@end

// Values shared with HyNomasMPSActivation::UnaryPlan::Precision.
enum HyNomasActivationPrecision : int32_t {
  HyNomasActivationF32 = 0,
  HyNomasActivationBF16Graph = 1,
  HyNomasActivationBF16Output = 2,
  HyNomasActivationPhysicalBF16 = 3,
};

static void *hy_nomas_mps_activation_plan_create(
    void *deviceHandle, const int64_t *shapeValues, int32_t rank,
    void *inputHandle, void *outputHandle, int32_t activation,
    int32_t precision) {
  if (!deviceHandle || !shapeValues || !inputHandle || !outputHandle ||
      rank < 1 || rank > 8 || activation < 0 || activation > 1 ||
      precision < HyNomasActivationF32 ||
      precision > HyNomasActivationPhysicalBF16 ||
      (precision == HyNomasActivationPhysicalBF16
           ? activation != 1
           : (precision != HyNomasActivationF32 && activation != 0)))
    return nullptr;
  const bool physicalBF16 = precision == HyNomasActivationPhysicalBF16;
  const MPSDataType ioType = physicalBF16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
  const bool computeBF16 = precision == HyNomasActivationBF16Graph;
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
      const uint64_t byteCount = elementCount * (physicalBF16 ? 2 : 4);
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
        if (!graph)
          return nullptr;
        graph.options = MPSGraphOptionsNone;
        NSString *inputName = activation == 0 ? @"hy_silu_x" : @"hy_gelu_x";
        if (computeBF16)
          inputName = @"hy_silu_x_f32";
        MPSGraphTensor *inputTensor = [graph placeholderWithShape:shape
                                                          dataType:ioType
                                                              name:inputName];
        if (!inputTensor)
          return nullptr;
        MPSGraphTensor *x = inputTensor;
        if (computeBF16) {
          // PyTorch 2.8 silu_out_mps types the placeholder and every SiLU
          // node from self.scalar_type(); the bound buffers here remain F32.
          x = [graph castTensor:inputTensor toType:MPSDataTypeBFloat16
                           name:@"hy_silu_input_bf16"];
        }
        MPSGraphTensor *result = nil;
        if (activation == 0) {
          MPSGraphTensor *one =
              [graph constantWithScalar:1.0
                                  shape:@[ @1 ]
                               dataType:(computeBF16
                                   ? MPSDataTypeBFloat16
                                   : MPSDataTypeFloat32)];
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
          if (!negative || !exponential || !one || !denominator || !result)
            return nullptr;
          if (computeBF16 &&
              (x.dataType != MPSDataTypeBFloat16 ||
               negative.dataType != MPSDataTypeBFloat16 ||
               exponential.dataType != MPSDataTypeBFloat16 ||
               one.dataType != MPSDataTypeBFloat16 ||
               denominator.dataType != MPSDataTypeBFloat16 ||
               result.dataType != MPSDataTypeBFloat16))
            return nullptr;
        } else {
          // PyTorch 2.8 Activation.mm's approximate="tanh" graph, preserving
          // operation order and rank-one scalar constants exactly.
          constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
          constexpr float kKappa = 0.044715f;
          MPSGraphTensor *beta = [graph constantWithScalar:kBeta
                                                      shape:@[ @1 ]
                                                   dataType:ioType];
          MPSGraphTensor *kappa = [graph constantWithScalar:kKappa
                                                       shape:@[ @1 ]
                                                    dataType:ioType];
          MPSGraphTensor *one = [graph constantWithScalar:1.0f
                                                    shape:@[ @1 ]
                                                 dataType:ioType];
          MPSGraphTensor *half = [graph constantWithScalar:0.5f
                                                     shape:@[ @1 ]
                                                  dataType:ioType];
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
        MPSGraphTensor *resultF32 = result;
        if (!physicalBF16 && precision != HyNomasActivationF32) {
          // Produce an explicit BF16 round-to-nearest-even expansion in the
          // existing F32 checkpoint, including when the SiLU nodes are F32.
          if (computeBF16) {
            resultF32 = [graph castTensor:result toType:MPSDataTypeFloat32
                                     name:@"hy_silu_output_f32"];
          }
          if (!resultF32 || resultF32.dataType != MPSDataTypeFloat32)
            return nullptr;

          MPSGraphTensor *resultBits =
              [graph reinterpretCastTensor:resultF32
                                     toType:MPSDataTypeUInt32
                                       name:@"hy_silu_output_bits"];
          MPSGraphTensor *shift16 = [graph constantWithScalar:16.0
                                                         shape:@[ @1 ]
                                                      dataType:MPSDataTypeUInt32];
          MPSGraphTensor *oneU32 = [graph constantWithScalar:1.0
                                                       shape:@[ @1 ]
                                                    dataType:MPSDataTypeUInt32];
          MPSGraphTensor *roundingBase = [graph constantWithScalar:0x7fff
                                                             shape:@[ @1 ]
                                                          dataType:MPSDataTypeUInt32];
          MPSGraphTensor *upperWordMask = [graph constantWithScalar:0xffff0000
                                                              shape:@[ @1 ]
                                                           dataType:MPSDataTypeUInt32];
          MPSGraphTensor *magnitudeMask = [graph constantWithScalar:0x7fffffff
                                                              shape:@[ @1 ]
                                                           dataType:MPSDataTypeUInt32];
          MPSGraphTensor *infinityBits = [graph constantWithScalar:0x7f800000
                                                             shape:@[ @1 ]
                                                          dataType:MPSDataTypeUInt32];
          MPSGraphTensor *quietNaNBit = [graph constantWithScalar:0x00400000
                                                            shape:@[ @1 ]
                                                         dataType:MPSDataTypeUInt32];
          if (!resultBits || !shift16 || !oneU32 || !roundingBase ||
              !upperWordMask || !magnitudeMask || !infinityBits ||
              !quietNaNBit || resultBits.dataType != MPSDataTypeUInt32)
            return nullptr;

          MPSGraphTensor *highWord =
              [graph bitwiseRightShiftWithPrimaryTensor:resultBits
                                        secondaryTensor:shift16
                                                   name:@"hy_silu_bf16_high_word"];
          MPSGraphTensor *roundingParity =
              [graph bitwiseANDWithPrimaryTensor:highWord
                                 secondaryTensor:oneU32
                                            name:@"hy_silu_bf16_rounding_parity"];
          MPSGraphTensor *roundingBias =
              [graph additionWithPrimaryTensor:roundingBase
                               secondaryTensor:roundingParity
                                          name:@"hy_silu_bf16_rounding_bias"];
          MPSGraphTensor *roundedBits =
              [graph additionWithPrimaryTensor:resultBits
                               secondaryTensor:roundingBias
                                          name:@"hy_silu_bf16_rounded_bits"];
          MPSGraphTensor *maskedRoundedBits =
              [graph bitwiseANDWithPrimaryTensor:roundedBits
                                 secondaryTensor:upperWordMask
                                            name:@"hy_silu_bf16_masked_bits"];
          MPSGraphTensor *magnitude =
              [graph bitwiseANDWithPrimaryTensor:resultBits
                                 secondaryTensor:magnitudeMask
                                            name:@"hy_silu_bf16_magnitude"];
          MPSGraphTensor *isNaN =
              [graph greaterThanWithPrimaryTensor:magnitude
                                  secondaryTensor:infinityBits
                                             name:@"hy_silu_bf16_is_nan"];
          MPSGraphTensor *nanBaseBits =
              [graph bitwiseANDWithPrimaryTensor:resultBits
                                 secondaryTensor:upperWordMask
                                            name:@"hy_silu_bf16_nan_base_bits"];
          MPSGraphTensor *quietNaNBits =
              [graph bitwiseORWithPrimaryTensor:nanBaseBits
                                secondaryTensor:quietNaNBit
                                           name:@"hy_silu_bf16_quiet_nan_bits"];
          MPSGraphTensor *bf16ExpandedBits =
              [graph selectWithPredicateTensor:isNaN
                           truePredicateTensor:quietNaNBits
                          falsePredicateTensor:maskedRoundedBits
                                           name:@"hy_silu_bf16_nan_preserving_bits"];
          if (!highWord || !roundingParity || !roundingBias || !roundedBits ||
              !maskedRoundedBits || !magnitude || !isNaN || !nanBaseBits ||
              !quietNaNBits ||
              !bf16ExpandedBits || highWord.dataType != MPSDataTypeUInt32 ||
              roundingParity.dataType != MPSDataTypeUInt32 ||
              roundingBias.dataType != MPSDataTypeUInt32 ||
              roundedBits.dataType != MPSDataTypeUInt32 ||
              maskedRoundedBits.dataType != MPSDataTypeUInt32 ||
              magnitude.dataType != MPSDataTypeUInt32 ||
              nanBaseBits.dataType != MPSDataTypeUInt32 ||
              quietNaNBits.dataType != MPSDataTypeUInt32 ||
              bf16ExpandedBits.dataType != MPSDataTypeUInt32)
            return nullptr;

          resultF32 = [graph reinterpretCastTensor:bf16ExpandedBits
                                             toType:MPSDataTypeFloat32
                                               name:@"hy_silu_bf16_expanded_f32"];
          if (!resultF32 || resultF32.dataType != MPSDataTypeFloat32)
            return nullptr;
        }
        if (!graph || !inputTensor || !x || !result || !resultF32 ||
            resultF32.shape == nil)
          return nullptr;

        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
            [NSMutableDictionary new];
        feeds[inputTensor] = [[MPSGraphShapedType alloc] initWithShape:shape
                                                            dataType:ioType];
        MPSGraphExecutable *executable =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                feeds:feeds
                        targetTensors:@[ resultF32 ]
                     targetOperations:nil
                compilationDescriptor:nil];
        if (!executable)
          return nullptr;
        executable.options = MPSGraphOptionsNone;

        MPSGraphTensorData *inputData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:input
                                                    shape:shape
                                                 dataType:ioType];
        MPSGraphTensorData *outputData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:output
                                                    shape:shape
                                                 dataType:ioType];
        MPSGraphExecutableExecutionDescriptor *execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        if (!inputData || !outputData || !execution ||
            executable.feedTensors.count != 1 ||
            executable.feedTensors[0] != inputTensor)
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
        plan.bf16Control = computeBF16;
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
    void *deviceHandle, int32_t activation, int32_t precision,
    const int64_t *shapeValues, int32_t rank, void *inputHandle,
    void *outputHandle) {
  return hy_nomas_mps_activation_plan_create(
      deviceHandle, shapeValues, rank, inputHandle, outputHandle, activation,
      precision);
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
