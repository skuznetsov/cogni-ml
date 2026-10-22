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
#define GS_MPS_API_PROFILE_WEAK_FALLBACK_IMPLEMENTATION
#include "mps_api_profile.h"
#undef GS_MPS_API_PROFILE_WEAK_FALLBACK_IMPLEMENTATION

// This bridge is also linked by targets that omit the optional command-chain
// object. A weak, non-inlined fallback keeps that link self-contained; when
// the command-chain object is present its strong implementation wins.
extern "C" __attribute__((weak, noinline)) void
gs_mps_api_profile_scope_begin(int32_t operation, void *command_context,
                               gs_mps_api_profile_scope *scope) {
  (void)operation;
  (void)command_context;
  if (scope)
    *scope = {};
}

extern "C" __attribute__((weak, noinline)) void
gs_mps_api_profile_scope_end(gs_mps_api_profile_scope *scope) {
  (void)scope;
}

class WanMpsApiProfileScopeGuard {
public:
  WanMpsApiProfileScopeGuard(int32_t operation, void *commandContext) {
    gs_mps_api_profile_scope_begin(operation, commandContext, &scope_);
  }
  ~WanMpsApiProfileScopeGuard() {
    if (scope_.active)
      gs_mps_api_profile_scope_end(&scope_);
  }
  WanMpsApiProfileScopeGuard(const WanMpsApiProfileScopeGuard &) = delete;
  WanMpsApiProfileScopeGuard &operator=(const WanMpsApiProfileScopeGuard &) =
      delete;

private:
  gs_mps_api_profile_scope scope_{};
};

@interface CogniWanAttentionMPSPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *residentBuffers;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArrayA;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArrayB;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, strong) MPSGraphTensor *queryTensor;
@property(nonatomic, strong) MPSGraphTensor *keyTensor;
@property(nonatomic, strong) MPSGraphTensor *valueTensor;
@property(nonatomic, strong) MPSGraphTensorData *queryData;
@property(nonatomic, strong) MPSGraphTensorData *keyData;
@property(nonatomic, strong) MPSGraphTensorData *valueData;
@property(nonatomic, strong) MPSNDArray *keyArray;
@property(nonatomic, strong) MPSNDArray *valueArray;
@property(nonatomic, strong) MPSGraphTensorData *outputDataA;
@property(nonatomic, strong) MPSGraphTensorData *outputDataB;
@property(nonatomic, strong) id<MTLBuffer> queryBuffer;
@property(nonatomic, strong) id<MTLBuffer> keyBuffer;
@property(nonatomic, strong) id<MTLBuffer> valueBuffer;
@property(nonatomic, strong) id<MTLBuffer> outputBufferA;
@property(nonatomic, strong) id<MTLBuffer> outputBufferB;
@property(nonatomic, strong) NSArray<NSNumber *> *queryShape;
@property(nonatomic, strong) NSArray<NSNumber *> *kvShape;
@property(nonatomic) int32_t batches;
@property(nonatomic) int32_t queryRows;
@property(nonatomic) int32_t keyRows;
@property(nonatomic) int32_t heads;
@property(nonatomic) int32_t headDim;
@property(nonatomic) MPSDataType dataType;
@property(nonatomic) uint64_t elementBytes;
@property(nonatomic) BOOL headMajorKV;
@property(nonatomic) BOOL explicitQKSoftmaxPV;
@property(nonatomic) BOOL headMajorKVOffsetView;
@property(nonatomic) uint64_t kvOffsetBytes;
@property(nonatomic, strong) NSMutableSet<NSNumber *> *groupKVOffsets;
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

static MPSNDArray *wan_attention_mps_group_kv_array(
    id<MTLBuffer> buffer, NSArray<NSNumber *> *shape, MPSDataType dataType,
    uint64_t offsetBytes, uint64_t tensorBytes) {
  if (!buffer || !shape || dataType != MPSDataTypeFloat32 ||
      offsetBytes % sizeof(float) != 0 || offsetBytes > (uint64_t)NSUIntegerMax ||
      offsetBytes > buffer.length || tensorBytes > buffer.length - offsetBytes)
    return nil;
  if (@available(macOS 15.0, *)) {
    MPSNDArrayDescriptor *descriptor =
        [MPSNDArrayDescriptor descriptorWithDataType:dataType shape:shape];
    if (!descriptor)
      return nil;
    descriptor.preferPackedRows = YES;
    MPSNDArray *array = [[MPSNDArray alloc] initWithBuffer:buffer
                                                   offset:(NSUInteger)offsetBytes
                                               descriptor:descriptor];
    if (!array || array.userBuffer != buffer)
      return nil;
    return array;
  }
  return nil;
}

static void *wan_attention_mps_plan_create_impl(
    void *deviceHandle, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, void *qHandle,
    void *kHandle, void *vHandle, void *outputAHandle,
    void *outputBHandle, MPSDataType dataType, uint64_t elementBytes,
    bool headMajorKV, bool explicitQKSoftmaxPV,
    bool allowPrivateBindings = false,
    bool headMajorKVOffsetView = false,
    uint64_t headMajorKVOffsetBytes = 0) {
  if (!deviceHandle || !qHandle || !kHandle || !vHandle || !outputAHandle ||
      !outputBHandle)
    return nullptr;
  if (@available(macOS 15.0, *)) {
    if ((dataType != MPSDataTypeFloat32 && dataType != MPSDataTypeFloat16 &&
         dataType != MPSDataTypeBFloat16) ||
        (elementBytes != 2 && elementBytes != 4) ||
        (dataType == MPSDataTypeFloat32) != (elementBytes == 4) ||
        (explicitQKSoftmaxPV && dataType == MPSDataTypeBFloat16) ||
        (headMajorKVOffsetView &&
         (!headMajorKV || explicitQKSoftmaxPV ||
          dataType != MPSDataTypeFloat32 || elementBytes != sizeof(float))) ||
        (!headMajorKVOffsetView && headMajorKVOffsetBytes != 0))
      return nullptr;
    @autoreleasepool {
      if (batches <= 0 || queryRows <= 0 || keyRows <= 0 || heads <= 0 ||
          headDim <= 0 || (headMajorKVOffsetView && batches != 1))
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
          vBuffer.length < kvBytes ||
          (headMajorKVOffsetView &&
           (headMajorKVOffsetBytes % kvBytes != 0 ||
            headMajorKVOffsetBytes > kBuffer.length ||
            kvBytes > kBuffer.length - headMajorKVOffsetBytes ||
            headMajorKVOffsetBytes > vBuffer.length ||
            kvBytes > vBuffer.length - headMajorKVOffsetBytes)))
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
        MPSGraphTensor *softmaxInput = nil;
        MPSGraphTensor *probabilities = nil;
        if (explicitQKSoftmaxPV) {
          keyTranspose = [graph transposeTensor:kh
                                      dimension:2
                                  withDimension:3
                                           name:@"wan_k_transpose_bhdq"];
          queryKey = [graph matrixMultiplicationWithPrimaryTensor:qh
                                                 secondaryTensor:keyTranspose
                                                            name:@"wan_qk_bhqk"];
          if (dataType == MPSDataTypeFloat32) {
            // Preserve the legacy explicit F32 graph. The pinned upstream
            // adds this epsilon only when the QK tensor itself is F32.
            MPSGraphTensor *epsilon = [graph constantWithScalar:1.0e-20f
                                                        shape:@[ @1 ]
                                                     dataType:MPSDataTypeFloat32];
            queryKeyWithEpsilon = [graph additionWithPrimaryTensor:queryKey
                                                 secondaryTensor:epsilon
                                                            name:@"wan_qk_epsilon_bhqk"];
          } else {
            // Match the connected upstream F16 graph, including its observable
            // cast boundaries. Tensor types do not specify accumulator width.
            queryKeyWithEpsilon = [graph castTensor:queryKey
                                           toType:MPSDataTypeFloat32
                                             name:@"wan_qk_f32_bhqk"];
          }
          scaleTensor = [graph constantWithScalar:scale
                                            shape:@[ @1 ]
                                         dataType:MPSDataTypeFloat32];
          scaledQueryKey = [graph multiplicationWithPrimaryTensor:queryKeyWithEpsilon
                                                  secondaryTensor:scaleTensor
                                                             name:@"wan_scaled_qk_bhqk"];
          softmaxInput = dataType == MPSDataTypeFloat32
                             ? scaledQueryKey
                             : [graph castTensor:scaledQueryKey
                                          toType:dataType
                                            name:@"wan_scaled_qk_f16_bhqk"];
          probabilities = [graph softMaxWithTensor:softmaxInput
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
              !scaleTensor || !scaledQueryKey || !softmaxInput || !probabilities)))
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
        MPSGraphTensorData *kData = nil;
        MPSGraphTensorData *vData = nil;
        MPSNDArray *kArray = nil;
        MPSNDArray *vArray = nil;
        if (headMajorKVOffsetView) {
          // Shape order is slowest-to-fastest: [B,H,K,D]. Packed rows ensure
          // each selected head group is a contiguous, zero-copy BHTD range.
          kArray = wan_attention_mps_group_kv_array(
              kBuffer, kvShape, dataType, headMajorKVOffsetBytes, kvBytes);
          vArray = wan_attention_mps_group_kv_array(
              vBuffer, kvShape, dataType, headMajorKVOffsetBytes, kvBytes);
          if (kArray && vArray) {
            kData = [[MPSGraphTensorData alloc] initWithMPSNDArray:kArray];
            vData = [[MPSGraphTensorData alloc] initWithMPSNDArray:vArray];
          }
        } else {
          kData = [[MPSGraphTensorData alloc] initWithMTLBuffer:kBuffer
                                                          shape:kvShape
                                                       dataType:dataType];
          vData = [[MPSGraphTensorData alloc] initWithMTLBuffer:vBuffer
                                                          shape:kvShape
                                                       dataType:dataType];
        }
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
        plan.queryTensor = q;
        plan.keyTensor = k;
        plan.valueTensor = v;
        plan.queryData = qData;
        plan.keyData = kData;
        plan.valueData = vData;
        plan.keyArray = kArray;
        plan.valueArray = vArray;
        plan.outputDataA = outputAData;
        plan.outputDataB = outputBData;
        plan.queryBuffer = qBuffer;
        plan.keyBuffer = kBuffer;
        plan.valueBuffer = vBuffer;
        plan.outputBufferA = outputABuffer;
        plan.outputBufferB = outputBBuffer;
        plan.queryShape = qShape;
        plan.kvShape = kvShape;
        plan.batches = batches;
        plan.queryRows = queryRows;
        plan.keyRows = keyRows;
        plan.heads = heads;
        plan.headDim = headDim;
        plan.dataType = dataType;
        plan.elementBytes = elementBytes;
        plan.headMajorKV = headMajorKV;
        plan.explicitQKSoftmaxPV = explicitQKSoftmaxPV;
        plan.headMajorKVOffsetView = headMajorKVOffsetView;
        plan.kvOffsetBytes = headMajorKVOffsetBytes;
        if (headMajorKVOffsetView)
          plan.groupKVOffsets = [NSMutableSet setWithObject:@(headMajorKVOffsetBytes)];
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

// Opt-in connected graph matching the pinned upstream F32/F16 operation and
// cast sequence. Q/context remain BTHD. headMajorKV selects BTHD (0) or
// caller-prepared contiguous BHTD (1) K/V; preparation belongs to the caller's
// existing command chain. This does not change the opaque SDPA defaults.
extern "C" void *gs_wan_attention_mps_plan_create_connected_typed(
    void *deviceHandle, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t heads, int32_t headDim, void *qHandle,
    void *kHandle, void *vHandle, void *outputAHandle,
    void *outputBHandle, int32_t dtype, int32_t headMajorKV) {
  if ((dtype != 0 && dtype != 1) || (headMajorKV != 0 && headMajorKV != 1))
    return nullptr;
  MPSDataType dataType = MPSDataTypeInvalid;
  uint64_t elementBytes = 0;
  if (!wan_attention_mps_data_type(dtype, &dataType, &elementBytes))
    return nullptr;
  return wan_attention_mps_plan_create_impl(
      deviceHandle, batches, queryRows, keyRows, heads, headDim, qHandle,
      kHandle, vHandle, outputAHandle, outputBHandle, dataType, elementBytes,
      headMajorKV == 1, true);
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

// Opt-in F32 SDPA plan for one contiguous head group in caller-prepared BHTD
// K/V. K/V storage remains caller-owned; the offset NDArray views are checked
// to alias the original MTLBuffers and are retained by the graph inputs.
extern "C" void *gs_wan_attention_mps_plan_create_head_major_kv_group_view(
    void *deviceHandle, int32_t batches, int32_t queryRows,
    int32_t keyRows, int32_t groupHeads, int32_t headDim, void *qHandle,
    void *kHandle, void *vHandle, void *outputAHandle,
    void *outputBHandle, uint64_t kvOffsetBytes) {
  return wan_attention_mps_plan_create_impl(
      deviceHandle, batches, queryRows, keyRows, groupHeads, headDim, qHandle,
      kHandle, vHandle, outputAHandle, outputBHandle, MPSDataTypeFloat32, 4,
      true, false, false, true, kvOffsetBytes);
}

// Clone one prepared group plan without recompiling its graph. The template's
// graph, executable, Q binding, outputs, and execution descriptor are shared;
// only the packed K/V offset views and their corresponding feed entries differ.
extern "C" void *gs_wan_attention_mps_plan_clone_head_major_kv_group_view(
    void *templateHandle, uint64_t kvOffsetBytes) {
  if (!templateHandle)
    return nullptr;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      CogniWanAttentionMPSPlan *source =
          (__bridge CogniWanAttentionMPSPlan *)templateHandle;
      NSMutableSet<NSNumber *> *offsetRegistry = source.groupKVOffsets;
      if (!offsetRegistry || !source.device || !source.graph ||
          !source.executable || !source.execution ||
          !source.headMajorKVOffsetView || !source.headMajorKV ||
          source.explicitQKSoftmaxPV || source.batches != 1 ||
          source.kvOffsetBytes != 0 ||
          source.dataType != MPSDataTypeFloat32 ||
          source.elementBytes != sizeof(float) || source.heads <= 0 ||
          source.queryRows <= 0 || source.keyRows <= 0 || source.headDim <= 0 ||
          !source.queryTensor || !source.keyTensor || !source.valueTensor ||
          !source.queryData || !source.keyData || !source.valueData ||
          !source.keyArray || !source.valueArray || !source.outputDataA ||
          !source.outputDataB || source.inputArray.count != 3 ||
          source.outputArrayA.count != 1 || source.outputArrayB.count != 1 ||
          source.outputArrayA[0] != source.outputDataA ||
          source.outputArrayB[0] != source.outputDataB ||
          source.residentBuffers.count != 5 || !source.queryBuffer ||
          !source.keyBuffer || !source.valueBuffer || !source.outputBufferA ||
          !source.outputBufferB || source.keyArray.userBuffer != source.keyBuffer ||
          source.valueArray.userBuffer != source.valueBuffer)
        return nullptr;

      NSArray<NSNumber *> *expectedQueryShape =
          @[ @1, @(source.queryRows), @(source.heads), @(source.headDim) ];
      NSArray<NSNumber *> *expectedKVShape =
          @[ @1, @(source.heads), @(source.keyRows), @(source.headDim) ];
      if (![source.queryShape isEqualToArray:expectedQueryShape] ||
          ![source.kvShape isEqualToArray:expectedKVShape] ||
          source.residentBuffers[0] != source.queryBuffer ||
          source.residentBuffers[1] != source.keyBuffer ||
          source.residentBuffers[2] != source.valueBuffer ||
          source.residentBuffers[3] != source.outputBufferA ||
          source.residentBuffers[4] != source.outputBufferB)
        return nullptr;
      NSArray<id<MTLBuffer>> *buffers = @[
        source.queryBuffer, source.keyBuffer, source.valueBuffer,
        source.outputBufferA, source.outputBufferB
      ];
      for (id<MTLBuffer> buffer in buffers) {
        if (buffer.device != source.device ||
            buffer.storageMode != MTLStorageModeShared)
          return nullptr;
      }

      const uint64_t kvShapeValues[] = {
          1, (uint64_t)source.heads, (uint64_t)source.keyRows,
          (uint64_t)source.headDim};
      uint64_t kvElements = 0;
      if (!wan_attention_mps_product(kvShapeValues, 4, &kvElements) ||
          kvElements > UINT64_MAX / sizeof(float))
        return nullptr;
      const uint64_t groupBytes = kvElements * sizeof(float);
      if (groupBytes == 0 || kvOffsetBytes == source.kvOffsetBytes ||
          kvOffsetBytes % groupBytes != 0 ||
          source.keyBuffer.length != source.valueBuffer.length ||
          source.keyBuffer.length % groupBytes != 0 ||
          kvOffsetBytes >= source.keyBuffer.length ||
          groupBytes > source.keyBuffer.length - kvOffsetBytes ||
          kvOffsetBytes >= source.valueBuffer.length ||
          groupBytes > source.valueBuffer.length - kvOffsetBytes)
        return nullptr;

      @synchronized(offsetRegistry) {
        NSNumber *offsetKey = @(kvOffsetBytes);
        if (![offsetRegistry containsObject:@0] ||
            [offsetRegistry containsObject:offsetKey])
          return nullptr;
        for (NSNumber *registeredOffset in offsetRegistry) {
          const uint64_t otherOffset = registeredOffset.unsignedLongLongValue;
          if (otherOffset % groupBytes != 0 || otherOffset >= source.keyBuffer.length ||
              groupBytes > source.keyBuffer.length - otherOffset ||
              !(kvOffsetBytes + groupBytes <= otherOffset ||
                otherOffset + groupBytes <= kvOffsetBytes))
            return nullptr;
        }

        @try {
          MPSNDArray *keyArray = wan_attention_mps_group_kv_array(
              source.keyBuffer, source.kvShape, source.dataType, kvOffsetBytes,
              groupBytes);
          MPSNDArray *valueArray = wan_attention_mps_group_kv_array(
              source.valueBuffer, source.kvShape, source.dataType, kvOffsetBytes,
              groupBytes);
          if (!keyArray || !valueArray ||
              keyArray.userBuffer != source.keyBuffer ||
              valueArray.userBuffer != source.valueBuffer)
            return nullptr;
          MPSGraphTensorData *keyData =
              [[MPSGraphTensorData alloc] initWithMPSNDArray:keyArray];
          MPSGraphTensorData *valueData =
              [[MPSGraphTensorData alloc] initWithMPSNDArray:valueArray];
          if (!keyData || !valueData)
            return nullptr;

          NSMutableArray<MPSGraphTensorData *> *orderedInputs =
              [NSMutableArray new];
          for (MPSGraphTensor *feedTensor in source.executable.feedTensors) {
            if (feedTensor == source.queryTensor)
              [orderedInputs addObject:source.queryData];
            else if (feedTensor == source.keyTensor)
              [orderedInputs addObject:keyData];
            else if (feedTensor == source.valueTensor)
              [orderedInputs addObject:valueData];
            else
              return nullptr;
          }
          if (orderedInputs.count != 3)
            return nullptr;

          CogniWanAttentionMPSPlan *clone = [CogniWanAttentionMPSPlan new];
          clone.device = source.device;
          clone.graph = source.graph;
          clone.executable = source.executable;
          clone.residentBuffers = source.residentBuffers;
          clone.inputArray = [orderedInputs copy];
          clone.outputArrayA = source.outputArrayA;
          clone.outputArrayB = source.outputArrayB;
          clone.execution = source.execution;
          clone.queryTensor = source.queryTensor;
          clone.keyTensor = source.keyTensor;
          clone.valueTensor = source.valueTensor;
          clone.queryData = source.queryData;
          clone.keyData = keyData;
          clone.valueData = valueData;
          clone.keyArray = keyArray;
          clone.valueArray = valueArray;
          clone.outputDataA = source.outputDataA;
          clone.outputDataB = source.outputDataB;
          clone.queryBuffer = source.queryBuffer;
          clone.keyBuffer = source.keyBuffer;
          clone.valueBuffer = source.valueBuffer;
          clone.outputBufferA = source.outputBufferA;
          clone.outputBufferB = source.outputBufferB;
          clone.queryShape = source.queryShape;
          clone.kvShape = source.kvShape;
          clone.batches = source.batches;
          clone.queryRows = source.queryRows;
          clone.keyRows = source.keyRows;
          clone.heads = source.heads;
          clone.headDim = source.headDim;
          clone.dataType = source.dataType;
          clone.elementBytes = source.elementBytes;
          clone.headMajorKV = source.headMajorKV;
          clone.explicitQKSoftmaxPV = source.explicitQKSoftmaxPV;
          clone.headMajorKVOffsetView = source.headMajorKVOffsetView;
          clone.kvOffsetBytes = kvOffsetBytes;
          clone.groupKVOffsets = offsetRegistry;
          [offsetRegistry addObject:offsetKey];
          return (__bridge_retained void *)clone;
        } @catch (NSException *exception) {
          (void)exception;
          return nullptr;
        }
      }
    }
  }
  return nullptr;
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
        WanMpsApiProfileScopeGuard profileScope(
            GS_MPS_API_PROFILE_OP_GRAPH_ENCODE,
            (__bridge void *)commandBuffer);
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
