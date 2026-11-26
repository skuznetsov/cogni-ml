// Prepared F32 Hy-Nomas attention with compact query chunks and head groups.
// Caller-owned shared buffers hold BTHD inputs/output and reusable staging.
// MPSGraph can allocate opaque internal workspace; this bridge makes no claim
// about those runtime allocations.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <stdint.h>
#include <limits.h>

@interface HyNomasCompactAttentionVariant : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) MPSGraphTensor *queryTensor;
@property(nonatomic, strong) MPSGraphTensor *keyTensor;
@property(nonatomic, strong) MPSGraphTensor *valueTensor;
@property(nonatomic) uint32_t rows;
@property(nonatomic) uint32_t groupHeads;
@end

@implementation HyNomasCompactAttentionVariant
@end

@interface HyNomasCompactAttentionBinding : NSObject
@property(nonatomic, strong) HyNomasCompactAttentionVariant *variant;
@property(nonatomic, strong) NSArray<MPSNDArrayDescriptor *> *descriptors;
@property(nonatomic, strong) NSArray<MPSNDArray *> *arrays;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@end

@implementation HyNomasCompactAttentionBinding
@end

@interface HyNomasCompactAttentionPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic, strong) id<MTLComputePipelineState> packPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> scatterPipeline;
@property(nonatomic, strong) NSArray<HyNomasCompactAttentionVariant *> *variants;
@property(nonatomic, strong) NSArray *bindings;
@property(nonatomic) uint32_t batches;
@property(nonatomic) uint32_t tokens;
@property(nonatomic) uint32_t heads;
@property(nonatomic) uint32_t headDim;
@property(nonatomic) uint32_t queryChunk;
@property(nonatomic) uint32_t headsPerGroup;
@property(nonatomic) uint32_t groupCount;
@property(nonatomic) uint32_t rowVariantCount;
@property(nonatomic) uint32_t groupVariantCount;
@property(nonatomic) uint32_t variantCount;
@property(nonatomic) float scale;
@end

@implementation HyNomasCompactAttentionPlan
@end

struct GsHyNomasCompactAttentionParams {
  uint32_t batches;
  uint32_t tokens;
  uint32_t heads;
  uint32_t headDim;
  uint32_t currentRows;
  uint32_t rowStart;
};

static id<MTLDevice> hy_nomas_compact_cached_device = nil;
static id<MTLComputePipelineState> hy_nomas_compact_pack_pipeline = nil;
static id<MTLComputePipelineState> hy_nomas_compact_gather_pipeline = nil;
static id<MTLComputePipelineState> hy_nomas_compact_scatter_pipeline = nil;

static bool hy_nomas_compact_product(const uint64_t *values, size_t count,
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

static bool hy_nomas_compact_disjoint(id<MTLBuffer> left,
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

static bool hy_nomas_compact_pipelines(
    id<MTLDevice> device, id<MTLComputePipelineState> *packPipeline,
    id<MTLComputePipelineState> *gatherPipeline,
    id<MTLComputePipelineState> *scatterPipeline) {
  if (!device || !packPipeline || !gatherPipeline || !scatterPipeline)
    return false;
  @synchronized([HyNomasCompactAttentionPlan class]) {
    if (hy_nomas_compact_cached_device == device &&
        hy_nomas_compact_pack_pipeline && hy_nomas_compact_gather_pipeline &&
        hy_nomas_compact_scatter_pipeline) {
      *packPipeline = hy_nomas_compact_pack_pipeline;
      *gatherPipeline = hy_nomas_compact_gather_pipeline;
      *scatterPipeline = hy_nomas_compact_scatter_pipeline;
      return true;
    }

    NSString *source = [NSString stringWithUTF8String:R"metal(
      #include <metal_stdlib>
      using namespace metal;
      struct GsHyNomasCompactAttentionParams {
        uint batches;
        uint tokens;
        uint heads;
        uint headDim;
        uint currentRows;
        uint rowStart;
      };
      kernel void gs_hy_nomas_compact_pack_bthd_to_bhtd(
          device const float *source [[buffer(0)]],
          device float *destination [[buffer(1)]],
          constant GsHyNomasCompactAttentionParams &p [[buffer(2)]],
          uint index [[thread_position_in_grid]]) {
        const uint total = p.batches * p.tokens * p.heads * p.headDim;
        if (index >= total)
          return;
        const uint width = index % p.headDim;
        const uint token = (index / p.headDim) % p.tokens;
        const uint head = (index / (p.headDim * p.tokens)) % p.heads;
        const uint batch = index / (p.headDim * p.tokens * p.heads);
        const uint sourceIndex =
            ((batch * p.tokens + token) * p.heads + head) * p.headDim + width;
        destination[index] = source[sourceIndex];
      }
      kernel void gs_hy_nomas_compact_gather_q_bthd_to_bhqd(
          device const float *source [[buffer(0)]],
          device float *destination [[buffer(1)]],
          constant GsHyNomasCompactAttentionParams &p [[buffer(2)]],
          uint index [[thread_position_in_grid]]) {
        const uint total = p.batches * p.heads * p.currentRows * p.headDim;
        if (index >= total)
          return;
        const uint width = index % p.headDim;
        const uint row = (index / p.headDim) % p.currentRows;
        const uint head = (index / (p.headDim * p.currentRows)) % p.heads;
        const uint batch = index / (p.headDim * p.currentRows * p.heads);
        const uint token = p.rowStart + row;
        const uint sourceIndex =
            ((batch * p.tokens + token) * p.heads + head) * p.headDim + width;
        destination[index] = source[sourceIndex];
      }
      kernel void gs_hy_nomas_compact_scatter_bhqd_to_bthd(
          device const float *source [[buffer(0)]],
          device float *destination [[buffer(1)]],
          constant GsHyNomasCompactAttentionParams &p [[buffer(2)]],
          uint index [[thread_position_in_grid]]) {
        const uint total = p.batches * p.heads * p.currentRows * p.headDim;
        if (index >= total)
          return;
        const uint width = index % p.headDim;
        const uint row = (index / p.headDim) % p.currentRows;
        const uint head = (index / (p.headDim * p.currentRows)) % p.heads;
        const uint batch = index / (p.headDim * p.currentRows * p.heads);
        const uint token = p.rowStart + row;
        const uint destinationIndex =
            ((batch * p.tokens + token) * p.heads + head) * p.headDim + width;
        destination[destinationIndex] = source[index];
      }
    )metal"];
    if (!source)
      return false;
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                   options:nil
                                                     error:&error];
    if (!library)
      return false;
    id<MTLFunction> packFunction =
        [library newFunctionWithName:@"gs_hy_nomas_compact_pack_bthd_to_bhtd"];
    id<MTLFunction> gatherFunction =
        [library newFunctionWithName:@"gs_hy_nomas_compact_gather_q_bthd_to_bhqd"];
    id<MTLFunction> scatterFunction =
        [library newFunctionWithName:@"gs_hy_nomas_compact_scatter_bhqd_to_bthd"];
    if (!packFunction || !gatherFunction || !scatterFunction)
      return false;
    id<MTLComputePipelineState> pack =
        [device newComputePipelineStateWithFunction:packFunction error:&error];
    if (!pack)
      return false;
    id<MTLComputePipelineState> gather =
        [device newComputePipelineStateWithFunction:gatherFunction
                                               error:&error];
    if (!gather)
      return false;
    id<MTLComputePipelineState> scatter =
        [device newComputePipelineStateWithFunction:scatterFunction
                                               error:&error];
    if (!scatter)
      return false;
    hy_nomas_compact_cached_device = device;
    hy_nomas_compact_pack_pipeline = pack;
    hy_nomas_compact_gather_pipeline = gather;
    hy_nomas_compact_scatter_pipeline = scatter;
    *packPipeline = pack;
    *gatherPipeline = gather;
    *scatterPipeline = scatter;
    return true;
  }
}

static MPSNDArray *hy_nomas_compact_array(id<MTLBuffer> buffer,
                                         NSArray<NSNumber *> *shape,
                                         uint64_t offsetBytes,
                                         uint64_t viewBytes,
                                         MPSNDArrayDescriptor **descriptorOut) {
  if (!buffer || !shape || !descriptorOut || offsetBytes % sizeof(float) != 0 ||
      offsetBytes > (uint64_t)NSUIntegerMax || offsetBytes > buffer.length ||
      viewBytes > buffer.length - offsetBytes)
    return nil;
  MPSNDArrayDescriptor *descriptor =
      [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                             shape:shape];
  if (!descriptor)
    return nil;
  descriptor.preferPackedRows = YES;
  MPSNDArray *array = [[MPSNDArray alloc] initWithBuffer:buffer
                                                  offset:(NSUInteger)offsetBytes
                                              descriptor:descriptor];
  if (!array || array.userBuffer != buffer)
    return nil;
  *descriptorOut = descriptor;
  return array;
}

static MPSGraphExecutable *hy_nomas_compact_compile_variant(
    id<MTLDevice> device, uint32_t rows, uint32_t groupHeads, uint32_t tokens,
    uint32_t headDim, float scale, MPSGraph **graphOut,
    MPSGraphTensor **queryOut, MPSGraphTensor **keyOut,
    MPSGraphTensor **valueOut) {
  if (!device || !graphOut || !queryOut || !keyOut || !valueOut)
    return nil;
  NSArray<NSNumber *> *queryShape =
      @[ @1, @(groupHeads), @(rows), @(headDim) ];
  NSArray<NSNumber *> *keyShape =
      @[ @1, @(groupHeads), @(tokens), @(headDim) ];
  MPSGraph *graph = [MPSGraph new];
  if (!graph)
    return nil;
  graph.options = MPSGraphOptionsNone;
  MPSGraphTensor *q = [graph placeholderWithShape:queryShape
                                         dataType:MPSDataTypeFloat32
                                             name:@"hy_compact_q_bhqd"];
  MPSGraphTensor *k = [graph placeholderWithShape:keyShape
                                         dataType:MPSDataTypeFloat32
                                             name:@"hy_compact_k_bhkd"];
  MPSGraphTensor *v = [graph placeholderWithShape:keyShape
                                         dataType:MPSDataTypeFloat32
                                             name:@"hy_compact_v_bhkd"];
  if (!q || !k || !v)
    return nil;
  MPSGraphTensor *keyTranspose = [graph transposeTensor:k
                                            dimension:2
                                        withDimension:3
                                                 name:@"hy_compact_k_bhdk"];
  MPSGraphTensor *logits =
      [graph matrixMultiplicationWithPrimaryTensor:q
                                  secondaryTensor:keyTranspose
                                             name:@"hy_compact_qk_bhqk"];
  MPSGraphTensor *epsilon =
      [graph constantWithScalar:1.0e-20f
                          shape:@[ @1 ]
                       dataType:MPSDataTypeFloat32];
  MPSGraphTensor *epsilonLogits =
      [graph additionWithPrimaryTensor:logits
                        secondaryTensor:epsilon
                                   name:@"hy_compact_qk_plus_epsilon_bhqk"];
  MPSGraphTensor *scaleTensor =
      [graph constantWithScalar:scale
                          shape:@[ @1 ]
                       dataType:MPSDataTypeFloat32];
  MPSGraphTensor *scaledLogits =
      [graph multiplicationWithPrimaryTensor:epsilonLogits
                              secondaryTensor:scaleTensor
                                         name:@"hy_compact_scaled_qk_bhqk"];
  MPSGraphTensor *probabilities = [graph softMaxWithTensor:scaledLogits
                                                       axis:3
                                                       name:@"hy_compact_probabilities_bhqk"];
  MPSGraphTensor *result =
      [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                  secondaryTensor:v
                                             name:@"hy_compact_context_bhqd"];
  if (!keyTranspose || !logits || !epsilon || !epsilonLogits || !scaleTensor ||
      !scaledLogits || !probabilities || !result)
    return nil;

  MPSGraphTensorShapedTypeDictionary *feeds = @{
    q : [[MPSGraphShapedType alloc] initWithShape:queryShape
                                        dataType:MPSDataTypeFloat32],
    k : [[MPSGraphShapedType alloc] initWithShape:keyShape
                                        dataType:MPSDataTypeFloat32],
    v : [[MPSGraphShapedType alloc] initWithShape:keyShape
                                        dataType:MPSDataTypeFloat32],
  };
  MPSGraphExecutable *executable =
      [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                          feeds:feeds
                  targetTensors:@[ result ]
               targetOperations:nil
          compilationDescriptor:nil];
  if (!executable)
    return nil;
  executable.options = MPSGraphOptionsNone;
  *graphOut = graph;
  *queryOut = q;
  *keyOut = k;
  *valueOut = v;
  return executable;
}

static HyNomasCompactAttentionBinding *hy_nomas_compact_make_binding(
    id<MTLBuffer> queryStage, id<MTLBuffer> keyStage,
    id<MTLBuffer> valueStage, id<MTLBuffer> outputStage,
    uint32_t tokens, uint32_t heads, uint32_t headDim, uint32_t batch,
    uint32_t headStart, uint32_t rows, uint32_t groupHeads,
    HyNomasCompactAttentionVariant *variant) {
  if (!queryStage || !keyStage || !valueStage || !outputStage || !variant)
    return nil;
  const uint64_t queryOffsetElements =
      ((uint64_t)batch * heads + headStart) * rows * headDim;
  const uint64_t queryViewElements = (uint64_t)groupHeads * rows * headDim;
  const uint64_t kvOffsetElements =
      ((uint64_t)batch * heads + headStart) * tokens * headDim;
  const uint64_t kvViewElements = (uint64_t)groupHeads * tokens * headDim;
  if (queryOffsetElements > UINT64_MAX / sizeof(float) ||
      queryViewElements > UINT64_MAX / sizeof(float) ||
      kvOffsetElements > UINT64_MAX / sizeof(float) ||
      kvViewElements > UINT64_MAX / sizeof(float))
    return nil;
  const uint64_t queryOffsetBytes = queryOffsetElements * sizeof(float);
  const uint64_t queryViewBytes = queryViewElements * sizeof(float);
  const uint64_t kvOffsetBytes = kvOffsetElements * sizeof(float);
  const uint64_t kvViewBytes = kvViewElements * sizeof(float);
  NSArray<NSNumber *> *queryShape = @[ @1, @(groupHeads), @(rows), @(headDim) ];
  NSArray<NSNumber *> *kvShape = @[ @1, @(groupHeads), @(tokens), @(headDim) ];
  MPSNDArrayDescriptor *queryDescriptor = nil;
  MPSNDArrayDescriptor *keyDescriptor = nil;
  MPSNDArrayDescriptor *valueDescriptor = nil;
  MPSNDArrayDescriptor *outputDescriptor = nil;
  MPSNDArray *queryArray = hy_nomas_compact_array(
      queryStage, queryShape, queryOffsetBytes, queryViewBytes,
      &queryDescriptor);
  MPSNDArray *keyArray = hy_nomas_compact_array(
      keyStage, kvShape, kvOffsetBytes, kvViewBytes, &keyDescriptor);
  MPSNDArray *valueArray = hy_nomas_compact_array(
      valueStage, kvShape, kvOffsetBytes, kvViewBytes, &valueDescriptor);
  MPSNDArray *outputArray = hy_nomas_compact_array(
      outputStage, queryShape, queryOffsetBytes, queryViewBytes,
      &outputDescriptor);
  if (!queryArray || !keyArray || !valueArray || !outputArray)
    return nil;

  MPSGraphTensorData *queryData =
      [[MPSGraphTensorData alloc] initWithMPSNDArray:queryArray];
  MPSGraphTensorData *keyData =
      [[MPSGraphTensorData alloc] initWithMPSNDArray:keyArray];
  MPSGraphTensorData *valueData =
      [[MPSGraphTensorData alloc] initWithMPSNDArray:valueArray];
  MPSGraphTensorData *outputData =
      [[MPSGraphTensorData alloc] initWithMPSNDArray:outputArray];
  if (!queryData || !keyData || !valueData || !outputData)
    return nil;

  NSMutableArray<MPSGraphTensorData *> *orderedInputs = [NSMutableArray new];
  for (MPSGraphTensor *feedTensor in variant.executable.feedTensors) {
    if (feedTensor == variant.queryTensor)
      [orderedInputs addObject:queryData];
    else if (feedTensor == variant.keyTensor)
      [orderedInputs addObject:keyData];
    else if (feedTensor == variant.valueTensor)
      [orderedInputs addObject:valueData];
    else
      return nil;
  }
  if (orderedInputs.count != 3)
    return nil;

  HyNomasCompactAttentionBinding *binding =
      [HyNomasCompactAttentionBinding new];
  binding.variant = variant;
  binding.descriptors = @[ queryDescriptor, keyDescriptor, valueDescriptor,
                           outputDescriptor ];
  binding.arrays = @[ queryArray, keyArray, valueArray, outputArray ];
  binding.inputArray = [orderedInputs copy];
  binding.outputArray = @[ outputData ];
  return binding;
}

static bool hy_nomas_compact_encode_kernel(
    MPSCommandBuffer *commandBuffer, id<MTLDevice> device,
    id<MTLComputePipelineState> pipeline, id<MTLBuffer> source,
    id<MTLBuffer> destination, GsHyNomasCompactAttentionParams params,
    uint64_t count) {
  if (!commandBuffer || !device || !pipeline || !source || !destination ||
      count == 0 || count > (uint64_t)INT32_MAX)
    return false;
  id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
  if (!root || root.commandQueue.device != device ||
      root.status >= MTLCommandBufferStatusCommitted)
    return false;
  id<MTLComputeCommandEncoder> encoder = [root computeCommandEncoder];
  if (!encoder)
    return false;
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:source offset:0 atIndex:0];
  [encoder setBuffer:destination offset:0 atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  const NSUInteger width =
      MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
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

extern "C" void *gs_hy_nomas_mps_attention_compact_plan_create(
    void *deviceHandle, int32_t batches, int32_t tokens, int32_t heads,
    int32_t headDim, int32_t queryChunk, int32_t headsPerGroup, float scale,
    void *queryHandle, void *keyHandle, void *valueHandle,
    void *queryStageHandle, void *keyStageHandle, void *valueStageHandle,
    void *outputStageHandle, void *outputHandle) {
  if (!deviceHandle || !queryHandle || !keyHandle || !valueHandle ||
      !queryStageHandle || !keyStageHandle || !valueStageHandle ||
      !outputStageHandle || !outputHandle || batches <= 0 || tokens <= 0 ||
      heads <= 0 || headDim <= 0 || queryChunk <= 0 || queryChunk > tokens ||
      headsPerGroup <= 0 || headsPerGroup > heads || !std::isfinite(scale) ||
      scale <= 0.0f || scale > 1.0f)
    return nullptr;

  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      const uint32_t batchCount = (uint32_t)batches;
      const uint32_t tokenCount = (uint32_t)tokens;
      const uint32_t headCount = (uint32_t)heads;
      const uint32_t width = (uint32_t)headDim;
      const uint32_t chunkRows = (uint32_t)queryChunk;
      const uint32_t groupRows = (uint32_t)headsPerGroup;
      const uint32_t stageRows = MIN(chunkRows, tokenCount);
      const uint64_t canonicalShape[] = {batchCount, tokenCount, headCount,
                                          width};
      const uint64_t queryStageShape[] = {batchCount, headCount, stageRows,
                                          width};
      uint64_t canonicalElements = 0;
      uint64_t queryStageElements = 0;
      if (!hy_nomas_compact_product(canonicalShape, 4, &canonicalElements) ||
          !hy_nomas_compact_product(queryStageShape, 4,
                                    &queryStageElements) ||
          canonicalElements > (uint64_t)INT32_MAX ||
          queryStageElements > (uint64_t)INT32_MAX ||
          canonicalElements > UINT64_MAX / sizeof(float) ||
          queryStageElements > UINT64_MAX / sizeof(float))
        return nullptr;
      const uint64_t canonicalBytes = canonicalElements * sizeof(float);
      const uint64_t queryStageBytes = queryStageElements * sizeof(float);
      if (canonicalBytes > (uint64_t)NSUIntegerMax ||
          queryStageBytes > (uint64_t)NSUIntegerMax)
        return nullptr;

      // Keep every graph at the prepared queryChunk size. The final window can
      // overlap earlier windows and overwrite their outputs because attention
      // rows are independent here; this avoids a distinct short-tail MPS kernel.
      const uint32_t rowVariantCount = 1u;
      const uint32_t groupTail = headCount % groupRows;
      const uint32_t groupVariantCount = groupTail ? 2u : 1u;
      const uint64_t groupCount64 =
          ((uint64_t)headCount + groupRows - 1) / groupRows;
      const uint64_t variantCount64 =
          (uint64_t)rowVariantCount * groupVariantCount;
      const uint64_t bindingCount64 =
          (uint64_t)batchCount * groupCount64 * variantCount64;
      if (groupCount64 == 0 || groupCount64 > UINT32_MAX ||
          variantCount64 > UINT32_MAX || bindingCount64 > (uint64_t)NSUIntegerMax ||
          bindingCount64 > SIZE_MAX / sizeof(id))
        return nullptr;

      const uint32_t rowShapes[1] = {chunkRows};
      const uint32_t groupShapes[2] = {groupRows, groupTail};
      for (uint32_t rowVariant = 0; rowVariant < rowVariantCount;
           ++rowVariant) {
        const uint32_t rows = rowShapes[rowVariant];
        const uint64_t scoreShape[] = {groupRows, rows, tokenCount};
        uint64_t scoreElements = 0;
        if (!hy_nomas_compact_product(scoreShape, 3, &scoreElements) ||
            scoreElements > (uint64_t)INT32_MAX ||
            scoreElements > UINT64_MAX / sizeof(float))
          return nullptr;
      }
      if (groupVariantCount == 2) {
        for (uint32_t rowVariant = 0; rowVariant < rowVariantCount;
             ++rowVariant) {
          const uint32_t rows = rowShapes[rowVariant];
          const uint64_t scoreShape[] = {groupTail, rows, tokenCount};
          uint64_t scoreElements = 0;
          if (!hy_nomas_compact_product(scoreShape, 3, &scoreElements) ||
              scoreElements > (uint64_t)INT32_MAX ||
              scoreElements > UINT64_MAX / sizeof(float))
            return nullptr;
        }
      }

      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> query = (__bridge id<MTLBuffer>)queryHandle;
      id<MTLBuffer> key = (__bridge id<MTLBuffer>)keyHandle;
      id<MTLBuffer> value = (__bridge id<MTLBuffer>)valueHandle;
      id<MTLBuffer> queryStage = (__bridge id<MTLBuffer>)queryStageHandle;
      id<MTLBuffer> keyStage = (__bridge id<MTLBuffer>)keyStageHandle;
      id<MTLBuffer> valueStage = (__bridge id<MTLBuffer>)valueStageHandle;
      id<MTLBuffer> outputStage = (__bridge id<MTLBuffer>)outputStageHandle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
      @try {
        NSArray<id<MTLBuffer>> *allBuffers =
            @[ query, key, value, queryStage, keyStage, valueStage,
               outputStage, output ];
        for (id<MTLBuffer> buffer in allBuffers) {
          if (!device || !buffer || buffer.device != device ||
              buffer.storageMode != MTLStorageModeShared)
            return nullptr;
        }
        if (query.length < canonicalBytes || key.length < canonicalBytes ||
            value.length < canonicalBytes ||
            keyStage.length < canonicalBytes ||
            valueStage.length < canonicalBytes ||
            queryStage.length < queryStageBytes ||
            outputStage.length < queryStageBytes ||
            output.length < canonicalBytes)
          return nullptr;
        for (NSUInteger left = 0; left < allBuffers.count; ++left) {
          for (NSUInteger right = left + 1; right < allBuffers.count; ++right) {
            if (!hy_nomas_compact_disjoint(allBuffers[left], allBuffers[right]))
              return nullptr;
          }
        }

        id<MTLComputePipelineState> packPipeline = nil;
        id<MTLComputePipelineState> gatherPipeline = nil;
        id<MTLComputePipelineState> scatterPipeline = nil;
        if (!hy_nomas_compact_pipelines(device, &packPipeline, &gatherPipeline,
                                        &scatterPipeline))
          return nullptr;

        NSMutableArray<HyNomasCompactAttentionVariant *> *variants =
            [NSMutableArray arrayWithCapacity:(NSUInteger)variantCount64];
        // NSNull reserves the skipped group slots for the other tail shape;
        // keep the collection untyped so slot indexing stays stable.
        NSMutableArray *bindings =
            [NSMutableArray arrayWithCapacity:(NSUInteger)bindingCount64];
        for (uint32_t rowVariant = 0; rowVariant < rowVariantCount;
             ++rowVariant) {
          const uint32_t rows = rowShapes[rowVariant];
          for (uint32_t groupVariant = 0; groupVariant < groupVariantCount;
               ++groupVariant) {
            const uint32_t groupHeads = groupShapes[groupVariant];
            MPSGraph *graph = nil;
            MPSGraphTensor *queryTensor = nil;
            MPSGraphTensor *keyTensor = nil;
            MPSGraphTensor *valueTensor = nil;
            MPSGraphExecutable *executable =
                hy_nomas_compact_compile_variant(
                    device, rows, groupHeads, tokenCount, width, scale, &graph,
                    &queryTensor, &keyTensor, &valueTensor);
            if (!executable || !graph || !queryTensor || !keyTensor ||
                !valueTensor)
              return nullptr;
            MPSGraphExecutableExecutionDescriptor *execution =
                [MPSGraphExecutableExecutionDescriptor new];
            if (!execution)
              return nullptr;
            execution.waitUntilCompleted = NO;
            HyNomasCompactAttentionVariant *variant =
                [HyNomasCompactAttentionVariant new];
            variant.graph = graph;
            variant.executable = executable;
            variant.execution = execution;
            variant.queryTensor = queryTensor;
            variant.keyTensor = keyTensor;
            variant.valueTensor = valueTensor;
            variant.rows = rows;
            variant.groupHeads = groupHeads;
            [variants addObject:variant];

            for (uint32_t batch = 0; batch < batchCount; ++batch) {
              for (uint64_t groupIndex = 0; groupIndex < groupCount64;
                   ++groupIndex) {
                const uint64_t headStart64 = groupIndex * groupRows;
                if (headStart64 >= headCount)
                  return nullptr;
                const uint32_t headStart = (uint32_t)headStart64;
                const uint32_t currentHeads =
                    MIN(groupRows, headCount - headStart);
                if (currentHeads != groupHeads) {
                  [bindings addObject:[NSNull null]];
                  continue;
                }
                HyNomasCompactAttentionBinding *binding =
                    hy_nomas_compact_make_binding(
                        queryStage, keyStage, valueStage, outputStage,
                        tokenCount, headCount, width, batch, headStart, rows,
                        currentHeads, variant);
                if (!binding)
                  return nullptr;
                [bindings addObject:binding];
              }
            }
          }
        }
        if (variants.count != variantCount64 ||
            bindings.count != bindingCount64)
          return nullptr;

        HyNomasCompactAttentionPlan *plan =
            [HyNomasCompactAttentionPlan new];
        plan.device = device;
        plan.residentBuffers = [allBuffers copy];
        plan.packPipeline = packPipeline;
        plan.gatherPipeline = gatherPipeline;
        plan.scatterPipeline = scatterPipeline;
        plan.variants = [variants copy];
        plan.bindings = [bindings copy];
        plan.batches = batchCount;
        plan.tokens = tokenCount;
        plan.heads = headCount;
        plan.headDim = width;
        plan.queryChunk = chunkRows;
        plan.headsPerGroup = groupRows;
        plan.groupCount = (uint32_t)groupCount64;
        plan.rowVariantCount = rowVariantCount;
        plan.groupVariantCount = groupVariantCount;
        plan.variantCount = (uint32_t)variantCount64;
        plan.scale = scale;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_hy_nomas_mps_attention_compact_encode(
    void *planHandle, void *commandHandle) {
  if (!planHandle || !commandHandle)
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      HyNomasCompactAttentionPlan *plan =
          (__bridge HyNomasCompactAttentionPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted)
        return -2;
      @try {
        id<MTLBuffer> query = plan.residentBuffers[0];
        id<MTLBuffer> key = plan.residentBuffers[1];
        id<MTLBuffer> value = plan.residentBuffers[2];
        id<MTLBuffer> queryStage = plan.residentBuffers[3];
        id<MTLBuffer> keyStage = plan.residentBuffers[4];
        id<MTLBuffer> valueStage = plan.residentBuffers[5];
        id<MTLBuffer> outputStage = plan.residentBuffers[6];
        id<MTLBuffer> output = plan.residentBuffers[7];
        GsHyNomasCompactAttentionParams params = {
            plan.batches, plan.tokens, plan.heads, plan.headDim,
            plan.tokens, 0};
        const uint64_t kvElements =
            (uint64_t)plan.batches * plan.tokens * plan.heads * plan.headDim;
        if (!hy_nomas_compact_encode_kernel(commandBuffer, plan.device,
                                            plan.packPipeline, key, keyStage,
                                            params, kvElements))
          return -3;
        if (!hy_nomas_compact_encode_kernel(commandBuffer, plan.device,
                                            plan.packPipeline, value,
                                            valueStage, params, kvElements))
          return -4;

        const uint64_t chunkCount =
            ((uint64_t)plan.tokens + plan.queryChunk - 1) / plan.queryChunk;
        const uint64_t groupsPerVariant =
            (uint64_t)plan.batches * plan.groupCount;
        for (uint64_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
          uint64_t rowStart64 = chunkIndex * plan.queryChunk;
          const uint64_t finalRowStart =
              (uint64_t)plan.tokens - plan.queryChunk;
          if (rowStart64 > finalRowStart)
            rowStart64 = finalRowStart;
          if (rowStart64 >= plan.tokens ||
              rowStart64 + plan.queryChunk > plan.tokens)
            return -5;
          const uint32_t rowStart = (uint32_t)rowStart64;
          const uint32_t currentRows = plan.queryChunk;
          const uint32_t rowVariant = 0u;
          if (rowVariant >= plan.rowVariantCount)
            return -6;
          params.currentRows = currentRows;
          params.rowStart = rowStart;
          const uint64_t qElements =
              (uint64_t)plan.batches * plan.heads * currentRows * plan.headDim;
          if (!hy_nomas_compact_encode_kernel(commandBuffer, plan.device,
                                              plan.gatherPipeline, query,
                                              queryStage, params, qElements))
            return -7;

          for (uint32_t groupIndex = 0; groupIndex < plan.groupCount;
               ++groupIndex) {
            const uint32_t headStart = groupIndex * plan.headsPerGroup;
            if (headStart >= plan.heads)
              return -8;
            const uint32_t currentHeads =
                MIN(plan.headsPerGroup, plan.heads - headStart);
            const uint32_t groupVariant =
                currentHeads == plan.headsPerGroup ? 0u : 1u;
            if (groupVariant >= plan.groupVariantCount)
              return -9;
            const uint32_t variantIndex =
                rowVariant * plan.groupVariantCount + groupVariant;
            if (variantIndex >= plan.variantCount)
              return -10;
            HyNomasCompactAttentionVariant *variant =
                plan.variants[variantIndex];
            for (uint32_t batch = 0; batch < plan.batches; ++batch) {
              const uint64_t bindingIndex =
                  (uint64_t)variantIndex * groupsPerVariant +
                  (uint64_t)batch * plan.groupCount + groupIndex;
              if (bindingIndex >= plan.bindings.count)
                return -11;
              id bindingObject = plan.bindings[(NSUInteger)bindingIndex];
              if (bindingObject == [NSNull null])
                return -11;
              HyNomasCompactAttentionBinding *binding =
                  (HyNomasCompactAttentionBinding *)bindingObject;
              if (binding.variant != variant)
                return -12;
              root = commandBuffer.rootCommandBuffer;
              if (!root || root.commandQueue.device != plan.device ||
                  root.status >= MTLCommandBufferStatusCommitted)
                return -13;
              NSArray<MPSGraphTensorData *> *result =
                  [variant.executable encodeToCommandBuffer:commandBuffer
                                                inputsArray:binding.inputArray
                                               resultsArray:binding.outputArray
                                        executionDescriptor:variant.execution];
              if (result.count != 1)
                return -14;
              root = commandBuffer.rootCommandBuffer;
              if (!root || root.commandQueue.device != plan.device ||
                  root.status >= MTLCommandBufferStatusCommitted)
                return -15;
            }
          }

          root = commandBuffer.rootCommandBuffer;
          if (!root || root.commandQueue.device != plan.device ||
              root.status >= MTLCommandBufferStatusCommitted)
            return -16;
          if (!hy_nomas_compact_encode_kernel(commandBuffer, plan.device,
                                              plan.scatterPipeline, outputStage,
                                              output, params, qElements))
            return -17;
        }
        return 0;
      } @catch (NSException *exception) {
        (void)exception;
        return -18;
      }
    }
  }
  return -19;
}

extern "C" void gs_hy_nomas_mps_attention_compact_plan_release(
    void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}
