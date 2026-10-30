// Compact F32 Hy-Nomas continuation attention. One segment-scoped workspace
// is reused by serial per-layer plans. K/V packs only valid history and current
// rows; each query chunk uses all heads and at most 256 score rows.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <limits.h>

@interface HyNomasHistoryAttentionVariant : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic) uint32_t rows;
@end

@implementation HyNomasHistoryAttentionVariant
@end

@interface HyNomasHistoryAttentionWorkspaceNative : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic, strong) id<MTLComputePipelineState> joinPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> scatterPipeline;
@property(nonatomic, strong) HyNomasHistoryAttentionVariant *fullVariant;
@property(nonatomic, strong) HyNomasHistoryAttentionVariant *tailVariant;
@property(nonatomic, strong) id<MTLBuffer> queryStage;
@property(nonatomic, strong) id<MTLBuffer> keyStage;
@property(nonatomic, strong) id<MTLBuffer> valueStage;
@property(nonatomic, strong) id<MTLBuffer> outputStage;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *historyKeys;
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *historyValues;
@property(nonatomic) uint32_t historyRows;
@property(nonatomic) uint32_t queryRows;
@property(nonatomic) uint32_t currentRows;
@property(nonatomic) uint32_t keyRows;
@property(nonatomic) uint32_t heads;
@property(nonatomic) uint32_t headDim;
@property(nonatomic) uint32_t queryChunk;
@property(nonatomic) float scale;
@property(nonatomic) BOOL historyFrozen;
@end

@implementation HyNomasHistoryAttentionWorkspaceNative
@end

@interface HyNomasHistoryAttentionPlan : NSObject
@property(nonatomic, strong) HyNomasHistoryAttentionWorkspaceNative *workspace;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic, strong) id<MTLBuffer> query;
@property(nonatomic, strong) id<MTLBuffer> currentKey;
@property(nonatomic, strong) id<MTLBuffer> currentValue;
@property(nonatomic, strong) id<MTLBuffer> output;
@end

@implementation HyNomasHistoryAttentionPlan
@end

struct GsHyNomasHistoryAttentionParams {
  uint32_t historyRows;
  uint32_t queryRows;
  uint32_t keyRows;
  uint32_t heads;
  uint32_t headDim;
  uint32_t currentRows;
  uint32_t rowStart;
};

static id<MTLDevice> hy_nomas_history_cached_device = nil;
static id<MTLComputePipelineState> hy_nomas_history_join_pipeline = nil;
static id<MTLComputePipelineState> hy_nomas_history_gather_pipeline = nil;
static id<MTLComputePipelineState> hy_nomas_history_scatter_pipeline = nil;

static bool hy_nomas_history_product(const uint64_t *values, size_t count,
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

static bool hy_nomas_history_disjoint(id<MTLBuffer> left,
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

static bool hy_nomas_history_valid_buffer(id<MTLDevice> device,
                                          id<MTLBuffer> buffer,
                                          uint64_t byteCount) {
  return device && buffer && buffer.device == device &&
         buffer.storageMode == MTLStorageModeShared && buffer.contents &&
         buffer.length >= byteCount;
}

static bool hy_nomas_history_pipelines(
    id<MTLDevice> device, id<MTLComputePipelineState> *joinPipeline,
    id<MTLComputePipelineState> *gatherPipeline,
    id<MTLComputePipelineState> *scatterPipeline) {
  if (!device || !joinPipeline || !gatherPipeline || !scatterPipeline)
    return false;
  @synchronized([HyNomasHistoryAttentionWorkspaceNative class]) {
    if (hy_nomas_history_cached_device == device &&
        hy_nomas_history_join_pipeline && hy_nomas_history_gather_pipeline &&
        hy_nomas_history_scatter_pipeline) {
      *joinPipeline = hy_nomas_history_join_pipeline;
      *gatherPipeline = hy_nomas_history_gather_pipeline;
      *scatterPipeline = hy_nomas_history_scatter_pipeline;
      return true;
    }

    NSString *source = [NSString stringWithUTF8String:R"metal(
      #include <metal_stdlib>
      using namespace metal;
      struct GsHyNomasHistoryAttentionParams {
        uint historyRows;
        uint queryRows;
        uint keyRows;
        uint heads;
        uint headDim;
        uint currentRows;
        uint rowStart;
      };
      kernel void gs_hy_nomas_history_join_bqhd_to_bhkd(
          device const float *history [[buffer(0)]],
          device const float *current [[buffer(1)]],
          device float *joined [[buffer(2)]],
          constant GsHyNomasHistoryAttentionParams &p [[buffer(3)]],
          uint index [[thread_position_in_grid]]) {
        const uint total = p.keyRows * p.heads * p.headDim;
        if (index >= total) return;
        const uint width = index % p.headDim;
        const uint row = (index / p.headDim) % p.keyRows;
        const uint head = index / (p.headDim * p.keyRows);
        const uint sourceRow = row < p.historyRows ? row : row - p.historyRows;
        const uint sourceIndex = (sourceRow * p.heads + head) * p.headDim + width;
        const float value = row < p.historyRows ? history[sourceIndex] : current[sourceIndex];
        joined[index] = value;
      }
      kernel void gs_hy_nomas_history_gather_q_bqhd_to_bhqd(
          device const float *source [[buffer(0)]],
          device float *destination [[buffer(1)]],
          constant GsHyNomasHistoryAttentionParams &p [[buffer(2)]],
          uint index [[thread_position_in_grid]]) {
        const uint total = p.currentRows * p.heads * p.headDim;
        if (index >= total) return;
        const uint width = index % p.headDim;
        const uint row = (index / p.headDim) % p.currentRows;
        const uint head = index / (p.headDim * p.currentRows);
        const uint sourceIndex = ((p.rowStart + row) * p.heads + head) * p.headDim + width;
        destination[index] = source[sourceIndex];
      }
      kernel void gs_hy_nomas_history_scatter_context_bhqd_to_bqhd(
          device const float *source [[buffer(0)]],
          device float *destination [[buffer(1)]],
          constant GsHyNomasHistoryAttentionParams &p [[buffer(2)]],
          uint index [[thread_position_in_grid]]) {
        const uint total = p.currentRows * p.heads * p.headDim;
        if (index >= total) return;
        const uint width = index % p.headDim;
        const uint row = (index / p.headDim) % p.currentRows;
        const uint head = index / (p.headDim * p.currentRows);
        const uint destinationIndex = ((p.rowStart + row) * p.heads + head) * p.headDim + width;
        destination[destinationIndex] = source[index];
      }
    )metal"];
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                   options:nil
                                                     error:&error];
    if (!library || error)
      return false;
    id<MTLFunction> joinFunction =
        [library newFunctionWithName:@"gs_hy_nomas_history_join_bqhd_to_bhkd"];
    id<MTLFunction> gatherFunction =
        [library newFunctionWithName:@"gs_hy_nomas_history_gather_q_bqhd_to_bhqd"];
    id<MTLFunction> scatterFunction =
        [library newFunctionWithName:@"gs_hy_nomas_history_scatter_context_bhqd_to_bqhd"];
    if (!joinFunction || !gatherFunction || !scatterFunction)
      return false;
    id<MTLComputePipelineState> join =
        [device newComputePipelineStateWithFunction:joinFunction error:&error];
    if (!join || error)
      return false;
    id<MTLComputePipelineState> gather =
        [device newComputePipelineStateWithFunction:gatherFunction error:&error];
    if (!gather || error)
      return false;
    id<MTLComputePipelineState> scatter =
        [device newComputePipelineStateWithFunction:scatterFunction error:&error];
    if (!scatter || error)
      return false;
    hy_nomas_history_cached_device = device;
    hy_nomas_history_join_pipeline = join;
    hy_nomas_history_gather_pipeline = gather;
    hy_nomas_history_scatter_pipeline = scatter;
    *joinPipeline = join;
    *gatherPipeline = gather;
    *scatterPipeline = scatter;
    return true;
  }
}

static HyNomasHistoryAttentionVariant *hy_nomas_history_compile_variant(
    id<MTLDevice> device, id<MTLBuffer> queryStage, id<MTLBuffer> keyStage,
    id<MTLBuffer> valueStage, id<MTLBuffer> outputStage, uint32_t heads,
    uint32_t rows, uint32_t keyRows, uint32_t headDim, float scale) {
  @try {
    NSArray<NSNumber *> *queryShape =
        @[ @1, @(heads), @(rows), @(headDim) ];
    NSArray<NSNumber *> *keyShape =
        @[ @1, @(heads), @(keyRows), @(headDim) ];
    MPSGraph *graph = [MPSGraph new];
    graph.options = MPSGraphOptionsNone;
    MPSGraphTensor *query = [graph placeholderWithShape:queryShape
                                               dataType:MPSDataTypeFloat32
                                                   name:@"hy_history_query_bhqd"];
    MPSGraphTensor *key = [graph placeholderWithShape:keyShape
                                             dataType:MPSDataTypeFloat32
                                                 name:@"hy_history_key_bhkd"];
    MPSGraphTensor *value = [graph placeholderWithShape:keyShape
                                               dataType:MPSDataTypeFloat32
                                                   name:@"hy_history_value_bhkd"];
    MPSGraphTensor *keyTranspose = [graph transposeTensor:key
                                                dimension:2
                                            withDimension:3
                                                     name:@"hy_history_key_bhdk"];
    MPSGraphTensor *logits =
        [graph matrixMultiplicationWithPrimaryTensor:query
                                    secondaryTensor:keyTranspose
                                               name:@"hy_history_logits_bhqk"];
    // Match the pinned upstream PyTorch 2.8 generic MPS SDPA equation:
    // add F32 epsilon before 1/sqrt(D), softmax over K, then multiply V.
    MPSGraphTensor *epsilon = [graph constantWithScalar:1.0e-20f
                                                   shape:@[ @1 ]
                                                dataType:MPSDataTypeFloat32];
    MPSGraphTensor *epsilonLogits =
        [graph additionWithPrimaryTensor:logits
                         secondaryTensor:epsilon
                                    name:@"hy_history_logits_plus_epsilon"];
    MPSGraphTensor *scaleTensor = [graph constantWithScalar:scale
                                                       shape:@[ @1 ]
                                                    dataType:MPSDataTypeFloat32];
    MPSGraphTensor *scaledLogits =
        [graph multiplicationWithPrimaryTensor:epsilonLogits
                               secondaryTensor:scaleTensor
                                          name:@"hy_history_scaled_logits"];
    MPSGraphTensor *probabilities =
        [graph softMaxWithTensor:scaledLogits
                            axis:3
                            name:@"hy_history_probabilities"];
    MPSGraphTensor *context =
        [graph matrixMultiplicationWithPrimaryTensor:probabilities
                                    secondaryTensor:value
                                               name:@"hy_history_context_bhqd"];
    if (!graph || !query || !key || !value || !keyTranspose || !logits ||
        !epsilonLogits || !scaledLogits || !probabilities || !context)
      return nil;

    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
        [NSMutableDictionary new];
    NSArray<MPSGraphTensor *> *placeholders = @[ query, key, value ];
    NSArray<MPSGraphTensorData *> *inputData = @[
      [[MPSGraphTensorData alloc] initWithMTLBuffer:queryStage
                                             shape:queryShape
                                          dataType:MPSDataTypeFloat32],
      [[MPSGraphTensorData alloc] initWithMTLBuffer:keyStage
                                             shape:keyShape
                                          dataType:MPSDataTypeFloat32],
      [[MPSGraphTensorData alloc] initWithMTLBuffer:valueStage
                                             shape:keyShape
                                          dataType:MPSDataTypeFloat32],
    ];
    feeds[query] = [[MPSGraphShapedType alloc] initWithShape:queryShape
                                                    dataType:MPSDataTypeFloat32];
    feeds[key] = [[MPSGraphShapedType alloc] initWithShape:keyShape
                                                  dataType:MPSDataTypeFloat32];
    feeds[value] = [[MPSGraphShapedType alloc] initWithShape:keyShape
                                                    dataType:MPSDataTypeFloat32];
    MPSGraphExecutable *executable =
        [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                            feeds:feeds
                    targetTensors:@[ context ]
                 targetOperations:nil
            compilationDescriptor:nil];
    if (!executable)
      return nil;
    executable.options = MPSGraphOptionsNone;
    MPSGraphTensorData *outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputStage
                    shape:queryShape
                 dataType:MPSDataTypeFloat32];
    if (!outputData)
      return nil;
    NSMutableArray<MPSGraphTensorData *> *orderedInputs = [NSMutableArray new];
    for (MPSGraphTensor *feedTensor in executable.feedTensors) {
      NSUInteger index = [placeholders indexOfObjectIdenticalTo:feedTensor];
      if (index == NSNotFound || index >= inputData.count)
        return nil;
      [orderedInputs addObject:inputData[index]];
    }
    if (orderedInputs.count != placeholders.count)
      return nil;
    MPSGraphExecutableExecutionDescriptor *execution =
        [MPSGraphExecutableExecutionDescriptor new];
    execution.waitUntilCompleted = NO;
    HyNomasHistoryAttentionVariant *variant =
        [HyNomasHistoryAttentionVariant new];
    variant.graph = graph;
    variant.executable = executable;
    variant.execution = execution;
    variant.inputArray = [orderedInputs copy];
    variant.outputArray = @[ outputData ];
    variant.rows = rows;
    return variant;
  } @catch (NSException *exception) {
    (void)exception;
    return nil;
  }
}

static bool hy_nomas_history_encode_compute(
    id<MTLCommandBuffer> root, id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> sourceA, id<MTLBuffer> sourceB, id<MTLBuffer> destination,
    const GsHyNomasHistoryAttentionParams &params, uint64_t workItems) {
  if (!root || !pipeline || !destination || workItems == 0 ||
      workItems > (uint64_t)NSUIntegerMax)
    return false;
  id<MTLComputeCommandEncoder> encoder = [root computeCommandEncoder];
  if (!encoder)
    return false;
  [encoder setComputePipelineState:pipeline];
  if (sourceA)
    [encoder setBuffer:sourceA offset:0 atIndex:0];
  if (sourceB)
    [encoder setBuffer:sourceB offset:0 atIndex:1];
  [encoder setBuffer:destination offset:0 atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  NSUInteger width = std::min((NSUInteger)256,
                              pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake((NSUInteger)workItems, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
  return true;
}

static bool hy_nomas_history_encode_query(
    id<MTLCommandBuffer> root, id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> source, id<MTLBuffer> destination,
    const GsHyNomasHistoryAttentionParams &params, uint64_t workItems) {
  if (!root || !pipeline || !source || !destination || workItems == 0 ||
      workItems > (uint64_t)NSUIntegerMax)
    return false;
  id<MTLComputeCommandEncoder> encoder = [root computeCommandEncoder];
  if (!encoder)
    return false;
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:source offset:0 atIndex:0];
  [encoder setBuffer:destination offset:0 atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  NSUInteger width = std::min((NSUInteger)256,
                              pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake((NSUInteger)workItems, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
  return true;
}

extern "C" void *gs_hy_nomas_history_attention_workspace_create(
    void *deviceHandle, int32_t historyRows, int32_t queryRows,
    int32_t currentRows, int32_t heads, int32_t headDim, int32_t queryChunk,
    float scale,
    void *queryStageHandle, void *keyStageHandle, void *valueStageHandle,
    void *outputStageHandle) {
  if (!deviceHandle || !queryStageHandle || !keyStageHandle ||
      !valueStageHandle || !outputStageHandle || historyRows < 0 ||
      queryRows <= 0 || currentRows <= 0 || currentRows > queryRows ||
      heads <= 0 || headDim <= 0 || queryChunk <= 0 ||
      queryChunk != std::min(queryRows, 256) || !std::isfinite(scale) ||
      scale <= 0.0f || scale > 1.0f)
    return nullptr;
  const int64_t keyRows64 = (int64_t)historyRows + (int64_t)currentRows;
  if (keyRows64 <= 0 || keyRows64 > INT32_MAX)
    return nullptr;
  const uint32_t keyRows = (uint32_t)keyRows64;
  const uint64_t queryShape[] = {1, (uint64_t)queryRows, (uint64_t)heads,
                                 (uint64_t)headDim};
  const uint64_t keyShape[] = {1, keyRows, (uint64_t)heads,
                                (uint64_t)headDim};
  const uint64_t tileShape[] = {1, (uint64_t)queryChunk, (uint64_t)heads,
                                (uint64_t)headDim};
  const uint64_t scoreShape[] = {1, (uint64_t)heads, (uint64_t)queryChunk,
                                 keyRows};
  uint64_t queryElements = 0, keyElements = 0, tileElements = 0,
           scoreElements = 0;
  if (!hy_nomas_history_product(queryShape, 4, &queryElements) ||
      !hy_nomas_history_product(keyShape, 4, &keyElements) ||
      !hy_nomas_history_product(tileShape, 4, &tileElements) ||
      !hy_nomas_history_product(scoreShape, 4, &scoreElements) ||
      queryElements > (uint64_t)INT32_MAX || keyElements > (uint64_t)INT32_MAX ||
      tileElements > (uint64_t)INT32_MAX || scoreElements > (uint64_t)INT32_MAX ||
      queryElements > UINT64_MAX / sizeof(float) ||
      keyElements > UINT64_MAX / sizeof(float) ||
      tileElements > UINT64_MAX / sizeof(float))
    return nullptr;

  id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
  id<MTLBuffer> queryStage = (__bridge id<MTLBuffer>)queryStageHandle;
  id<MTLBuffer> keyStage = (__bridge id<MTLBuffer>)keyStageHandle;
  id<MTLBuffer> valueStage = (__bridge id<MTLBuffer>)valueStageHandle;
  id<MTLBuffer> outputStage = (__bridge id<MTLBuffer>)outputStageHandle;
  NSArray<id<MTLBuffer>> *buffers =
      @[ queryStage, keyStage, valueStage, outputStage ];
  const uint64_t requiredBytes[] = {tileElements * sizeof(float),
                                    keyElements * sizeof(float),
                                    keyElements * sizeof(float),
                                    tileElements * sizeof(float)};
  for (NSUInteger index = 0; index < buffers.count; ++index) {
    if (!hy_nomas_history_valid_buffer(device, buffers[index],
                                      requiredBytes[index]))
      return nullptr;
    for (NSUInteger other = index + 1; other < buffers.count; ++other) {
      if (!hy_nomas_history_disjoint(buffers[index], buffers[other]))
        return nullptr;
    }
  }

  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      @try {
        id<MTLComputePipelineState> joinPipeline = nil;
        id<MTLComputePipelineState> gatherPipeline = nil;
        id<MTLComputePipelineState> scatterPipeline = nil;
        if (!hy_nomas_history_pipelines(device, &joinPipeline,
                                        &gatherPipeline, &scatterPipeline))
          return nullptr;
        uint32_t tailRows = (uint32_t)queryRows % (uint32_t)queryChunk;
        HyNomasHistoryAttentionVariant *full =
            hy_nomas_history_compile_variant(device, queryStage, keyStage,
                                              valueStage, outputStage,
                                              (uint32_t)heads,
                                              (uint32_t)queryChunk, keyRows,
                                              (uint32_t)headDim, scale);
        if (!full)
          return nullptr;
        HyNomasHistoryAttentionVariant *tail = nil;
        if (tailRows > 0) {
          tail = hy_nomas_history_compile_variant(
              device, queryStage, keyStage, valueStage, outputStage,
              (uint32_t)heads, tailRows, keyRows, (uint32_t)headDim, scale);
          if (!tail)
            return nullptr;
        }
        HyNomasHistoryAttentionWorkspaceNative *workspace =
            [HyNomasHistoryAttentionWorkspaceNative new];
        workspace.device = device;
        workspace.residentBuffers = buffers;
        workspace.joinPipeline = joinPipeline;
        workspace.gatherPipeline = gatherPipeline;
        workspace.scatterPipeline = scatterPipeline;
        workspace.fullVariant = full;
        workspace.tailVariant = tail;
        workspace.queryStage = queryStage;
        workspace.keyStage = keyStage;
        workspace.valueStage = valueStage;
        workspace.outputStage = outputStage;
        workspace.historyKeys = [NSMutableArray new];
        workspace.historyValues = [NSMutableArray new];
        workspace.historyRows = (uint32_t)historyRows;
        workspace.queryRows = (uint32_t)queryRows;
        workspace.currentRows = (uint32_t)currentRows;
        workspace.keyRows = keyRows;
        workspace.heads = (uint32_t)heads;
        workspace.headDim = (uint32_t)headDim;
        workspace.queryChunk = (uint32_t)queryChunk;
        workspace.scale = scale;
        return (__bridge_retained void *)workspace;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_hy_nomas_history_attention_workspace_register_history(
    void *workspaceHandle, void *keyHandle, void *valueHandle) {
  if (!workspaceHandle || !keyHandle || !valueHandle)
    return -1;
  HyNomasHistoryAttentionWorkspaceNative *workspace =
      (__bridge HyNomasHistoryAttentionWorkspaceNative *)workspaceHandle;
  if (!workspace || workspace.historyFrozen || workspace.historyRows == 0)
    return -2;
  const uint64_t historyShape[] = {1, workspace.historyRows,
                                   workspace.heads, workspace.headDim};
  uint64_t historyElements = 0;
  if (!hy_nomas_history_product(historyShape, 4, &historyElements) ||
      historyElements > UINT64_MAX / sizeof(float))
    return -3;
  const uint64_t historyBytes = historyElements * sizeof(float);
  id<MTLBuffer> key = (__bridge id<MTLBuffer>)keyHandle;
  id<MTLBuffer> value = (__bridge id<MTLBuffer>)valueHandle;
  if (!hy_nomas_history_valid_buffer(workspace.device, key, historyBytes) ||
      !hy_nomas_history_valid_buffer(workspace.device, value, historyBytes) ||
      !hy_nomas_history_disjoint(key, value))
    return -4;
  for (id<MTLBuffer> stage in workspace.residentBuffers) {
    if (!hy_nomas_history_disjoint(key, stage) ||
        !hy_nomas_history_disjoint(value, stage))
      return -5;
  }
  for (NSUInteger index = 0; index < workspace.historyKeys.count; ++index) {
    if (!hy_nomas_history_disjoint(key, workspace.historyKeys[index]) ||
        !hy_nomas_history_disjoint(key, workspace.historyValues[index]) ||
        !hy_nomas_history_disjoint(value, workspace.historyKeys[index]) ||
        !hy_nomas_history_disjoint(value, workspace.historyValues[index]))
      return -6;
  }
  [workspace.historyKeys addObject:key];
  [workspace.historyValues addObject:value];
  return 0;
}

extern "C" int32_t gs_hy_nomas_history_attention_workspace_freeze_history(
    void *workspaceHandle, int32_t expectedSlots) {
  if (!workspaceHandle || expectedSlots < 0)
    return -1;
  HyNomasHistoryAttentionWorkspaceNative *workspace =
      (__bridge HyNomasHistoryAttentionWorkspaceNative *)workspaceHandle;
  if (!workspace || workspace.historyFrozen ||
      workspace.historyKeys.count != (NSUInteger)expectedSlots ||
      workspace.historyValues.count != (NSUInteger)expectedSlots ||
      (workspace.historyRows == 0 && expectedSlots != 0) ||
      (workspace.historyRows > 0 && expectedSlots == 0))
    return -2;
  workspace.historyFrozen = YES;
  return 0;
}

extern "C" void *gs_hy_nomas_history_attention_plan_create(
    void *workspaceHandle, void *queryHandle, void *currentKeyHandle,
    void *currentValueHandle, void *outputHandle) {
  if (!workspaceHandle || !queryHandle || !currentKeyHandle ||
      !currentValueHandle || !outputHandle)
    return nullptr;
  HyNomasHistoryAttentionWorkspaceNative *workspace =
      (__bridge HyNomasHistoryAttentionWorkspaceNative *)workspaceHandle;
  if (!workspace || !workspace.historyFrozen)
    return nullptr;
  const uint64_t queryShape[] = {1, workspace.queryRows, workspace.heads,
                                 workspace.headDim};
  const uint64_t currentShape[] = {1, workspace.currentRows, workspace.heads,
                                   workspace.headDim};
  uint64_t queryElements = 0, currentElements = 0;
  if (!hy_nomas_history_product(queryShape, 4, &queryElements) ||
      !hy_nomas_history_product(currentShape, 4, &currentElements) ||
      queryElements > UINT64_MAX / sizeof(float) ||
      currentElements > UINT64_MAX / sizeof(float))
    return nullptr;
  const uint64_t queryBytes = queryElements * sizeof(float);
  const uint64_t currentBytes = currentElements * sizeof(float);
  id<MTLBuffer> query = (__bridge id<MTLBuffer>)queryHandle;
  id<MTLBuffer> currentKey = (__bridge id<MTLBuffer>)currentKeyHandle;
  id<MTLBuffer> currentValue = (__bridge id<MTLBuffer>)currentValueHandle;
  id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
  NSArray<id<MTLBuffer>> *buffers = @[ query, currentKey, currentValue, output ];
  const uint64_t requiredBytes[] = {queryBytes, currentBytes,
                                    currentBytes, queryBytes};
  for (NSUInteger index = 0; index < buffers.count; ++index) {
    id<MTLBuffer> buffer = buffers[index];
    if (!hy_nomas_history_valid_buffer(workspace.device, buffer,
                                       requiredBytes[index]))
      return nullptr;
    for (NSUInteger other = index + 1; other < buffers.count; ++other) {
      if (!hy_nomas_history_disjoint(buffer, buffers[other]))
        return nullptr;
    }
    for (id<MTLBuffer> stage in workspace.residentBuffers) {
      if (!hy_nomas_history_disjoint(buffer, stage))
        return nullptr;
    }
    for (id<MTLBuffer> cache in workspace.historyKeys) {
      if (!hy_nomas_history_disjoint(buffer, cache))
        return nullptr;
    }
    for (id<MTLBuffer> cache in workspace.historyValues) {
      if (!hy_nomas_history_disjoint(buffer, cache))
        return nullptr;
    }
  }
  HyNomasHistoryAttentionPlan *plan = [HyNomasHistoryAttentionPlan new];
  plan.workspace = workspace;
  plan.residentBuffers = buffers;
  plan.query = query;
  plan.currentKey = currentKey;
  plan.currentValue = currentValue;
  plan.output = output;
  return (__bridge_retained void *)plan;
}

extern "C" int32_t gs_hy_nomas_history_attention_encode(
    void *planHandle, void *commandHandle, int32_t historySlot) {
  if (!planHandle || !commandHandle)
    return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      HyNomasHistoryAttentionPlan *plan =
          (__bridge HyNomasHistoryAttentionPlan *)planHandle;
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!plan || !plan.workspace || !commandBuffer || !root ||
          root.commandQueue.device != plan.workspace.device ||
          root.status >= MTLCommandBufferStatusCommitted)
        return -2;
      HyNomasHistoryAttentionWorkspaceNative *workspace = plan.workspace;
      if (!workspace.historyFrozen ||
          (workspace.historyRows == 0 && historySlot != -1) ||
          (workspace.historyRows > 0 &&
           (historySlot < 0 ||
            (NSUInteger)historySlot >= workspace.historyKeys.count)))
        return -3;
      id<MTLBuffer> historyKey = workspace.historyRows > 0
                                     ? workspace.historyKeys[(NSUInteger)historySlot]
                                     : plan.currentKey;
      id<MTLBuffer> historyValue = workspace.historyRows > 0
                                       ? workspace.historyValues[(NSUInteger)historySlot]
                                       : plan.currentValue;
      GsHyNomasHistoryAttentionParams params = {
          workspace.historyRows, workspace.queryRows, workspace.keyRows,
          workspace.heads, workspace.headDim, workspace.currentRows, 0};
      const uint64_t keyWork = (uint64_t)workspace.keyRows * workspace.heads *
                               workspace.headDim;
      if (!hy_nomas_history_encode_compute(root, workspace.joinPipeline,
                                            historyKey, plan.currentKey,
                                            workspace.keyStage, params, keyWork))
        return -4;
      if (!hy_nomas_history_encode_compute(root, workspace.joinPipeline,
                                            historyValue, plan.currentValue,
                                            workspace.valueStage, params, keyWork))
        return -5;

      for (uint32_t rowStart = 0; rowStart < workspace.queryRows;
           rowStart += workspace.queryChunk) {
        const uint32_t rows = std::min(workspace.queryChunk,
                                       workspace.queryRows - rowStart);
        params.currentRows = rows;
        params.rowStart = rowStart;
        const uint64_t queryWork = (uint64_t)rows * workspace.heads *
                                   workspace.headDim;
        root = commandBuffer.rootCommandBuffer;
        if (!hy_nomas_history_encode_query(root, workspace.gatherPipeline,
                                            plan.query, workspace.queryStage,
                                            params, queryWork))
          return -6;
        HyNomasHistoryAttentionVariant *variant =
            rows == workspace.queryChunk ? workspace.fullVariant
                                         : workspace.tailVariant;
        if (!variant || variant.rows != rows)
          return -7;
        root = commandBuffer.rootCommandBuffer;
        NSArray<MPSGraphTensorData *> *result =
            [variant.executable encodeToCommandBuffer:commandBuffer
                                          inputsArray:variant.inputArray
                                         resultsArray:variant.outputArray
                                  executionDescriptor:variant.execution];
        if (result.count != 1)
          return -8;
        root = commandBuffer.rootCommandBuffer;
        if (!root || root.commandQueue.device != workspace.device ||
            root.status >= MTLCommandBufferStatusCommitted)
          return -9;
        if (!hy_nomas_history_encode_query(root, workspace.scatterPipeline,
                                            workspace.outputStage, plan.output,
                                            params, queryWork))
          return -10;
      }
      return 0;
    }
  }
  return -11;
}

extern "C" void gs_hy_nomas_history_attention_plan_release(void *planHandle) {
  if (planHandle) {
    id plan = (__bridge_transfer id)planHandle;
    (void)plan;
  }
}

extern "C" void gs_hy_nomas_history_attention_workspace_release(
    void *workspaceHandle) {
  if (workspaceHandle) {
    id workspace = (__bridge_transfer id)workspaceHandle;
    (void)workspace;
  }
}
