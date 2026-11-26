// Diagnostic-only independent-window B2 attention batching. The compile guard
// is local to this translation unit; the production tile loop remains the
// default in every other bridge object.
#define COGNI_PIXELUMM_WINDOW_BATCH_DIAGNOSTIC 1
#include "../../src/ml/video/pixelumm_decoder_head_mps_bridge.mm"

#import <objc/runtime.h>

#include <map>
#include <utility>

static constexpr uint32_t kPixelWindowBatchWidth = 2u;
static constexpr uint32_t kPixelWindowBatchQ = 384u;
static constexpr uint32_t kPixelWindowBatchK = 3456u;
static constexpr uint32_t kPixelWindowBatchPairs = 30u;
static constexpr uint32_t kPixelWindowBatchMetricsWords = 4u;

enum PixelWindowBatchCheckMode : int32_t {
  PixelWindowBatchCheckOff = 0,
  PixelWindowBatchCheckRawQK = 1,
  PixelWindowBatchCheckPV = 2,
  PixelWindowBatchCheckPositiveRawQK = 3
};

struct PixelWindowBatchGroupParams {
  uint32_t tile0;
  uint32_t tile1;
  uint32_t maxQuery;
  uint32_t maxKey;
  uint32_t queryCount;
  uint32_t keyCount;
};

struct PixelWindowBatchScatterParams {
  uint32_t tile0;
  uint32_t tile1;
  uint32_t maxQuery;
  uint32_t queryCount;
};

struct PixelWindowBatchCopyParams {
  uint32_t count;
  uint32_t destinationOffset;
};

struct PixelWindowBatchCompareParams {
  uint32_t count;
  uint32_t candidateOffset;
  uint32_t referenceOffset;
  uint32_t groupCount;
  uint32_t injectFirst;
  float atol;
  float rtol;
};

static_assert(sizeof(PixelWindowBatchGroupParams) == 24,
              "PixelUMM window-batch group ABI");
static_assert(sizeof(PixelWindowBatchScatterParams) == 16,
              "PixelUMM window-batch scatter ABI");
static_assert(sizeof(PixelWindowBatchCopyParams) == 8,
              "PixelUMM window-batch copy ABI");
static_assert(sizeof(PixelWindowBatchCompareParams) == 28,
              "PixelUMM window-batch comparator ABI");

@interface PixelUMMWindowBatchGeometry : NSObject
@property(nonatomic) uint32_t queryCount;
@property(nonatomic) uint32_t keyCount;
@property(nonatomic, strong) PixelUMMGraphStep *qk;
@property(nonatomic, strong) PixelUMMGraphStep *softmax;
@property(nonatomic, strong) PixelUMMGraphStep *pv;
@property(nonatomic, strong) PixelUMMGraphStep *pvBaseline;
@end
@implementation PixelUMMWindowBatchGeometry
@end

@interface PixelUMMWindowBatchPool : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *panels;
@property(nonatomic, strong) id<MTLBuffer> q;
@property(nonatomic, strong) id<MTLBuffer> k;
@property(nonatomic, strong) id<MTLBuffer> v;
@property(nonatomic, strong) id<MTLBuffer> scores;
@property(nonatomic, strong) id<MTLBuffer> probabilities;
@property(nonatomic, strong) id<MTLBuffer> output;
@property(nonatomic, strong) NSArray<PixelUMMWindowBatchGeometry *> *geometries;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherQ;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherKVPair;
@property(nonatomic, strong) id<MTLComputePipelineState> scatter;
@property(nonatomic, strong) id<MTLComputePipelineState> panelCopyF32;
@property(nonatomic, strong) id<MTLComputePipelineState> compareF32;
@property(nonatomic) uint64_t extraBytes;
@property(nonatomic) uint32_t wrapperCount;
@end
@implementation PixelUMMWindowBatchPool
@end

@interface PixelUMMWindowBatchPair : NSObject
@property(nonatomic) uint32_t tile0;
@property(nonatomic) uint32_t tile1;
@property(nonatomic) uint32_t queryCount;
@property(nonatomic) uint32_t keyCount;
@property(nonatomic) NSUInteger geometryIndex;
@end
@implementation PixelUMMWindowBatchPair
@end

@interface PixelUMMWindowBatch : NSObject
@property(nonatomic, weak) PixelUMMDecoderPlan *plan;
@property(nonatomic, strong) PixelUMMWindowBatchPool *pool;
@property(nonatomic, strong) NSArray<PixelUMMWindowBatchPair *> *pairs;
@property(nonatomic, strong) id<MTLBuffer> metrics;
@property(nonatomic, strong) id installationToken;
@property(nonatomic) BOOL active;
@property(nonatomic) BOOL selected;
@property(nonatomic) int32_t checkMode;
@end
@implementation PixelUMMWindowBatch
@end

static char pixel_window_batch_associated_key;

static const char *pixel_window_batch_metal_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct WindowBatchGroup {
  uint tile0;
  uint tile1;
  uint maxQuery;
  uint maxKey;
  uint queryCount;
  uint keyCount;
};
struct WindowBatchScatter {
  uint tile0;
  uint tile1;
  uint maxQuery;
  uint queryCount;
};
struct WindowBatchCopy { uint count; uint destinationOffset; };
struct WindowBatchCompare {
  uint count;
  uint candidateOffset;
  uint referenceOffset;
  uint groupCount;
  uint injectFirst;
  float atol;
  float rtol;
};

inline float window_batch_bf16_to_float(ushort value) {
  return as_type<float>(uint(value) << 16u);
}
inline ushort window_batch_float_to_bf16(float value) {
  uint bits = as_type<uint>(value);
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u)
    return ushort((bits >> 16u) | 0x40u);
  return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

kernel void pixelumm_window_batch_gather_q(
    device const ushort *source [[buffer(0)]],
    device const int *queryIndices [[buffer(1)]],
    device float *destination [[buffer(2)]],
    constant WindowBatchGroup &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint panel = 32u * p.queryCount * 128u;
  const uint total = 2u * panel;
  if (index >= total) return;
  const uint slot = index / panel;
  const uint local = index - slot * panel;
  const uint head = local / (p.queryCount * 128u);
  const uint row = (local / 128u) % p.queryCount;
  const uint dim = local % 128u;
  const uint tile = slot == 0u ? p.tile0 : p.tile1;
  const uint tableBase = tile * (p.maxQuery + 1u);
  const uint count = uint(queryIndices[tableBase]);
  if (count != p.queryCount || row >= count) return;
  const uint token = uint(queryIndices[tableBase + 1u + row]);
  destination[index] = window_batch_bf16_to_float(
      source[(token * 32u + head) * 128u + dim]);
}

kernel void pixelumm_window_batch_gather_kv_pair(
    device const ushort *keySource [[buffer(0)]],
    device const ushort *valueSource [[buffer(1)]],
    device const int *keyIndices [[buffer(2)]],
    device float *keyDestination [[buffer(3)]],
    device float *valueDestination [[buffer(4)]],
    constant WindowBatchGroup &p [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const uint panel = 32u * p.keyCount * 128u;
  const uint total = 2u * panel;
  if (index >= total) return;
  const uint slot = index / panel;
  const uint local = index - slot * panel;
  const uint head = local / (p.keyCount * 128u);
  const uint row = (local / 128u) % p.keyCount;
  const uint dim = local % 128u;
  const uint tile = slot == 0u ? p.tile0 : p.tile1;
  const uint tableBase = tile * (p.maxKey + 1u);
  const uint count = uint(keyIndices[tableBase]);
  if (count != p.keyCount || row >= count) return;
  const uint token = uint(keyIndices[tableBase + 1u + row]);
  const uint sourceIndex = (token * 8u + head / 4u) * 128u + dim;
  keyDestination[index] = window_batch_bf16_to_float(keySource[sourceIndex]);
  valueDestination[index] = window_batch_bf16_to_float(valueSource[sourceIndex]);
}

kernel void pixelumm_window_batch_scatter(
    device const float *source [[buffer(0)]],
    device const int *queryIndices [[buffer(1)]],
    device ushort *destination [[buffer(2)]],
    constant WindowBatchScatter &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint panel = 32u * p.queryCount * 128u;
  const uint total = 2u * panel;
  if (index >= total) return;
  const uint slot = index / panel;
  const uint local = index - slot * panel;
  const uint head = local / (p.queryCount * 128u);
  const uint row = (local / 128u) % p.queryCount;
  const uint dim = local % 128u;
  const uint tile = slot == 0u ? p.tile0 : p.tile1;
  const uint tableBase = tile * (p.maxQuery + 1u);
  const uint count = uint(queryIndices[tableBase]);
  if (count != p.queryCount || row >= count) return;
  const uint token = uint(queryIndices[tableBase + 1u + row]);
  destination[(token * 32u + head) * 128u + dim] =
      window_batch_float_to_bf16(source[index]);
}

kernel void pixelumm_window_batch_copy_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant WindowBatchCopy &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < p.count)
    destination[p.destinationOffset + index] = source[index];
}

kernel void pixelumm_window_batch_compare_f32(
    device const float *candidate [[buffer(0)]],
    device const float *reference [[buffer(1)]],
    device atomic_uint *metrics [[buffer(2)]],
    constant WindowBatchCompare &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup uint4 partial[128];
  uint4 local = uint4(0u);
  const uint stride = p.groupCount * threads;
  for (uint index = group * threads + lane; index < p.count; index += stride) {
    const float expected = reference[p.referenceOffset + index];
    float actual = candidate[p.candidateOffset + index];
    if (p.injectFirst != 0u && index == 0u)
      actual = expected + max(1.0f, fabs(expected) * 0.001f);
    const bool finite = isfinite(actual) && isfinite(expected);
    const bool finiteMismatch = finite &&
        as_type<uint>(actual) != as_type<uint>(expected);
    const bool failure = !finite ||
        fabs(actual - expected) > p.atol + p.rtol * fabs(expected);
    local += uint4(uint(failure), uint(finiteMismatch), uint(!finite), 1u);
  }
  partial[lane] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = threads / 2u; step > 0u; step >>= 1u) {
    if (lane < step) partial[lane] += partial[lane + step];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) {
    for (uint field = 0u; field < 4u; ++field)
      if (partial[0][field] != 0u)
        atomic_fetch_add_explicit(metrics + field, partial[0][field],
                                  memory_order_relaxed);
  }
}
)METAL";

static PixelUMMGraphStep *pixel_window_batch_qk_step(
    id<MTLDevice> device, id<MTLBuffer> qBuffer, id<MTLBuffer> kBuffer,
    id<MTLBuffer> scoresBuffer, uint32_t queryCount, uint32_t keyCount,
    NSString *name, char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
      [NSMutableDictionary dictionary];
  NSMutableArray<MPSGraphTensor *> *placeholders = [NSMutableArray array];
  NSMutableArray<MPSGraphTensorData *> *inputData = [NSMutableArray array];
  NSArray *qShape = pixel_shape({64u, queryCount, 128u});
  NSArray *kShape = pixel_shape({64u, keyCount, 128u});
  NSArray *scoreShape3 = pixel_shape({64u, queryCount, keyCount});
  NSArray *scoreShape4 = pixel_shape({2u, 32u, queryCount, keyCount});
  MPSGraphTensor *q = pixel_add_feed(graph, feeds, placeholders, inputData,
      qBuffer, qShape, MPSDataTypeFloat32, @"window_batch_q_f32");
  MPSGraphTensor *k = pixel_add_feed(graph, feeds, placeholders, inputData,
      kBuffer, kShape, MPSDataTypeFloat32, @"window_batch_k_f32");
  MPSGraphTensor *kTranspose = k ? [graph transposeTensor:k dimension:1
      withDimension:2 name:@"window_batch_k_transpose_f32"] : nil;
  MPSGraphTensor *scores3 = q && kTranspose
      ? [graph matrixMultiplicationWithPrimaryTensor:q secondaryTensor:kTranspose
          name:@"window_batch_qk_bmm_f32"] : nil;
  MPSGraphTensor *scores4 = scores3
      ? [graph reshapeTensor:scores3 withShape:scoreShape4
          name:@"window_batch_qk_rank4_view"] : nil;
  if (!q || !k || !kTranspose || !scores3 || !scores4 ||
      ![scores3.shape isEqualToArray:scoreShape3]) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build B2 QK graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      scores4, scoresBuffer, scoreShape4, name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_window_batch_softmax_step(
    id<MTLDevice> device, id<MTLBuffer> scoresBuffer,
    id<MTLBuffer> probabilitiesBuffer, uint32_t queryCount,
    uint32_t keyCount, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
      [NSMutableDictionary dictionary];
  NSMutableArray<MPSGraphTensor *> *placeholders = [NSMutableArray array];
  NSMutableArray<MPSGraphTensorData *> *inputData = [NSMutableArray array];
  NSArray *shape = pixel_shape({2u, 32u, queryCount, keyCount});
  MPSGraphTensor *scores = pixel_add_feed(graph, feeds, placeholders, inputData,
      scoresBuffer, shape, MPSDataTypeFloat32,
      @"window_batch_scaled_scores_f32");
  MPSGraphTensor *probabilities = scores
      ? [graph softMaxWithTensor:scores axis:3 name:@"window_batch_softmax_f32"]
      : nil;
  if (!scores || !probabilities) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build B2 softmax graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      probabilities, probabilitiesBuffer, shape, name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_window_batch_pv_step(
    id<MTLDevice> device, id<MTLBuffer> probabilitiesBuffer,
    id<MTLBuffer> vBuffer, id<MTLBuffer> outputBuffer,
    uint32_t queryCount, uint32_t keyCount, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds =
      [NSMutableDictionary dictionary];
  NSMutableArray<MPSGraphTensor *> *placeholders = [NSMutableArray array];
  NSMutableArray<MPSGraphTensorData *> *inputData = [NSMutableArray array];
  NSArray *probabilityShape4 = pixel_shape({2u, 32u, queryCount, keyCount});
  NSArray *probabilityShape3 = pixel_shape({64u, queryCount, keyCount});
  NSArray *vShape = pixel_shape({64u, keyCount, 128u});
  NSArray *outputShape3 = pixel_shape({64u, queryCount, 128u});
  NSArray *outputShape4 = pixel_shape({2u, 32u, queryCount, 128u});
  MPSGraphTensor *probabilities4 = pixel_add_feed(graph, feeds, placeholders,
      inputData, probabilitiesBuffer, probabilityShape4, MPSDataTypeFloat32,
      @"window_batch_probabilities_f32");
  MPSGraphTensor *probabilities3 = probabilities4
      ? [graph reshapeTensor:probabilities4 withShape:probabilityShape3
          name:@"window_batch_probabilities_rank3_view"] : nil;
  MPSGraphTensor *v = pixel_add_feed(graph, feeds, placeholders, inputData,
      vBuffer, vShape, MPSDataTypeFloat32, @"window_batch_v_f32");
  MPSGraphTensor *attended3 = probabilities3 && v
      ? [graph matrixMultiplicationWithPrimaryTensor:probabilities3 secondaryTensor:v
          name:@"window_batch_pv_bmm_f32"] : nil;
  MPSGraphTensor *attended4 = attended3
      ? [graph reshapeTensor:attended3 withShape:outputShape4
          name:@"window_batch_pv_rank4_view"] : nil;
  if (!probabilities4 || !probabilities3 || !v || !attended3 || !attended4 ||
      ![attended3.shape isEqualToArray:outputShape3]) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build B2 PV graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      attended4, outputBuffer, outputShape4, name, errorText, errorCapacity);
}

static bool pixel_window_batch_add(uint64_t left, uint64_t right,
                                   uint64_t *result) {
  if (!result || left > UINT64_MAX - right) return false;
  *result = left + right;
  return true;
}

static PixelUMMWindowBatchGeometry *pixel_window_batch_find_geometry(
    PixelUMMWindowBatchPool *pool, uint32_t queryCount,
    uint32_t keyCount, NSUInteger *indexOut) {
  if (!pool || !pool.geometries) return nil;
  for (NSUInteger index = 0; index < pool.geometries.count; ++index) {
    PixelUMMWindowBatchGeometry *geometry = pool.geometries[index];
    if (geometry.queryCount == queryCount && geometry.keyCount == keyCount) {
      if (indexOut) *indexOut = index;
      return geometry;
    }
  }
  return nil;
}

static PixelUMMWindowBatchPool *pixel_window_batch_create_pool(
    id<MTLDevice> device, char *errorText, size_t errorCapacity) {
  if (!device) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch pool requires a Metal device");
    return nil;
  }
  uint64_t qBytes = 0, kvBytes = 0, scoreBytes = 0, outputBytes = 0;
  if (!pixel_product(2u * 32u * kPixelWindowBatchQ * 128u,
                     sizeof(float), &qBytes) ||
      !pixel_product(2u * 32u * kPixelWindowBatchK * 128u,
                     sizeof(float), &kvBytes) ||
      !pixel_product(2u * 32u * kPixelWindowBatchQ * kPixelWindowBatchK,
                     sizeof(float), &scoreBytes)) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch panel extent overflowed");
    return nil;
  }
  if (scoreBytes > (uint64_t)device.maxBufferLength) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch score panel exceeds the Metal device buffer limit");
    return nil;
  }
  outputBytes = qBytes;
  const uint64_t panelBytes[6] = {
    qBytes, kvBytes, kvBytes, scoreBytes, scoreBytes, outputBytes
  };
  NSMutableArray<id<MTLBuffer>> *panels =
      [NSMutableArray arrayWithCapacity:6];
  uint64_t totalBytes = 0;
  for (NSUInteger index = 0; index < 6; ++index) {
    id<MTLBuffer> buffer = [device newBufferWithLength:(NSUInteger)panelBytes[index]
        options:MTLResourceStorageModeShared];
    if (!pixel_buffer(buffer, device, panelBytes[index])) {
      pixel_set_error(errorText, errorCapacity,
          "unable to allocate an exact shared B2 attention panel");
      return nil;
    }
    for (id<MTLBuffer> prior in panels) {
      if (!pixel_disjoint(prior, buffer)) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch candidate panels alias each other");
        return nil;
      }
    }
    if (!pixel_window_batch_add(totalBytes, (uint64_t)buffer.length,
                                &totalBytes)) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch panel allocation total overflowed");
      return nil;
    }
    [panels addObject:buffer];
  }

  PixelUMMWindowBatchPool *pool = [PixelUMMWindowBatchPool new];
  pool.device = device;
  pool.panels = [panels copy];
  pool.q = panels[0];
  pool.k = panels[1];
  pool.v = panels[2];
  pool.scores = panels[3];
  pool.probabilities = panels[4];
  pool.output = panels[5];
  pool.extraBytes = totalBytes;

  NSError *metalError = nil;
  NSString *source = [NSString stringWithUTF8String:pixel_window_batch_metal_source];
  MTLCompileOptions *options = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeSafe;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  pool.library = [device newLibraryWithSource:source options:options
                                        error:&metalError];
  if (!pool.library) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to compile window-batch Metal kernels: %@",
         metalError.localizedDescription ?: @"unknown Metal compiler error"]);
    return nil;
  }
#define PIXEL_WINDOW_BATCH_PIPELINE(property, functionName) \
  pool.property = pixel_pipeline(device, pool.library, @functionName, \
      errorText, errorCapacity); \
  if (!pool.property) return nil
  PIXEL_WINDOW_BATCH_PIPELINE(gatherQ, "pixelumm_window_batch_gather_q");
  PIXEL_WINDOW_BATCH_PIPELINE(gatherKVPair,
      "pixelumm_window_batch_gather_kv_pair");
  PIXEL_WINDOW_BATCH_PIPELINE(scatter, "pixelumm_window_batch_scatter");
  PIXEL_WINDOW_BATCH_PIPELINE(panelCopyF32,
      "pixelumm_window_batch_copy_f32");
  PIXEL_WINDOW_BATCH_PIPELINE(compareF32, "pixelumm_window_batch_compare_f32");
#undef PIXEL_WINDOW_BATCH_PIPELINE
  if (pool.gatherQ.maxTotalThreadsPerThreadgroup == 0u ||
      pool.gatherKVPair.maxTotalThreadsPerThreadgroup == 0u ||
      pool.scatter.maxTotalThreadsPerThreadgroup == 0u ||
      pool.panelCopyF32.maxTotalThreadsPerThreadgroup == 0u ||
      pool.compareF32.maxTotalThreadsPerThreadgroup < 128u) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch Metal pipelines do not support the required threadgroup widths");
    return nil;
  }

  const uint32_t geometrySizes[3][2] = {
    {384u, 3456u}, {384u, 2880u}, {192u, 2880u}
  };
  NSMutableArray<PixelUMMWindowBatchGeometry *> *geometries =
      [NSMutableArray arrayWithCapacity:3];
  for (NSUInteger index = 0; index < 3; ++index) {
    const uint32_t queryCount = geometrySizes[index][0];
    const uint32_t keyCount = geometrySizes[index][1];
    NSString *baseName = [NSString stringWithFormat:
        @"pixelumm_window_batch_b2_q%u_k%u", queryCount, keyCount];
    PixelUMMWindowBatchGeometry *geometry =
        [PixelUMMWindowBatchGeometry new];
    geometry.queryCount = queryCount;
    geometry.keyCount = keyCount;
    geometry.qk = pixel_window_batch_qk_step(device, pool.q, pool.k,
        pool.scores, queryCount, keyCount,
        [baseName stringByAppendingString:@"_qk"], errorText, errorCapacity);
    geometry.softmax = geometry.qk
        ? pixel_window_batch_softmax_step(device, pool.scores,
            pool.probabilities, queryCount, keyCount,
            [baseName stringByAppendingString:@"_softmax"],
            errorText, errorCapacity) : nil;
    geometry.pv = geometry.softmax
        ? pixel_window_batch_pv_step(device, pool.probabilities, pool.v,
            pool.output, queryCount, keyCount,
            [baseName stringByAppendingString:@"_pv"],
            errorText, errorCapacity) : nil;
    geometry.pvBaseline = geometry.pv
        ? pixel_window_batch_pv_step(device, pool.scores, pool.v,
            pool.output, queryCount, keyCount,
            [baseName stringByAppendingString:@"_pv_baseline_p"],
            errorText, errorCapacity) : nil;
    if (!geometry.qk || !geometry.softmax || !geometry.pv ||
        !geometry.pvBaseline) return nil;
    [geometries addObject:geometry];
  }
  pool.geometries = [geometries copy];
  return pool;
}

static bool pixel_window_batch_validate_plan(
    PixelUMMDecoderPlan *plan, bool requireIdle,
    char *errorText, size_t errorCapacity) {
  if (!plan || !plan.device || (requireIdle && plan.encodingStarted) ||
      plan->nativeIslands || plan.useNativeIslands || plan.useFusedKVGather ||
      plan.temporal != 6u || plan.height != 44u || plan.width != 80u ||
      plan.tokens != 21120u || plan.tile != 8u || plan.dim != 4096u ||
      plan.ffn != 12288u || plan.heads != 32u || plan.kvHeads != 8u ||
      plan.headDim != 128u || plan.maxQuery != 384u ||
      plan.maxKey != 3456u || plan.tileCount != 60u ||
      plan.softmaxPVPrepared || plan.useSoftmaxPV ||
      plan.softmaxPVOutputOnlyPrepared || plan.useSoftmaxPVOutputOnly ||
      plan.scaleSoftmaxPrepared || plan.useScaleSoftmax ||
      plan.softmaxPVCompare || plan.probabilitySnapshot ||
      plan.attendedSnapshot || plan.softmaxPVCheckMetrics ||
      plan.scratch.count != kPixelScratchCount ||
      plan.core.count != kPixelCoreCount ||
      !plan.steps || !plan.resident || plan.tileGeometryIndices.count != 60u ||
      !plan.attentionGeometries || plan.attentionGeometries.count != 3u) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch requires a baseline full-J21120 T6/H44/W80 PixelUMM plan");
    return false;
  }
  const int32_t *queryTable = (const int32_t *)plan.scratch[6].contents;
  const int32_t *keyTable = (const int32_t *)plan.scratch[7].contents;
  if (!pixel_validate_table(plan.scratch[6], 60u, 384u, 21120u) ||
      !pixel_validate_table(plan.scratch[7], 60u, 3456u, 21120u)) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch requires valid full-J query and key layout tables");
    return false;
  }
  std::vector<uint8_t> querySeen(21120u, uint8_t(0));
  uint64_t queryTotal = 0;
  std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> classes;
  for (uint32_t tile = 0; tile < 60u; ++tile) {
    const uint64_t queryBase = uint64_t(tile) * 385u;
    const uint64_t keyBase = uint64_t(tile) * 3457u;
    const uint32_t queryCount = (uint32_t)queryTable[queryBase];
    const uint32_t keyCount = (uint32_t)keyTable[keyBase];
    if (!((queryCount == 384u &&
           (keyCount == 3456u || keyCount == 2880u)) ||
          (queryCount == 192u && keyCount == 2880u))) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch found an unsupported full-J Q/K class");
      return false;
    }
    for (uint32_t row = 0; row < queryCount; ++row) {
      const int32_t token = queryTable[queryBase + 1u + row];
      if (token < 0 || uint32_t(token) >= querySeen.size() ||
          querySeen[(uint32_t)token] != 0u) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch query scatter is not a disjoint full-J partition");
        return false;
      }
      querySeen[(uint32_t)token] = 1u;
      ++queryTotal;
    }
    NSUInteger geometryIndex =
        plan.tileGeometryIndices[tile].unsignedIntegerValue;
    if (geometryIndex >= plan.attentionGeometries.count) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch tile points outside prepared baseline geometries");
      return false;
    }
    PixelUMMAttentionGeometry *baseline = plan.attentionGeometries[geometryIndex];
    if (!baseline || baseline.queryCount != queryCount ||
        baseline.keyCount != keyCount || !baseline.qk ||
        !baseline.softmax || !baseline.pv || baseline.scaleSoftmax ||
        baseline.softmaxPV || baseline.softmaxPVOutputOnly ||
        ![plan.steps containsObject:baseline.qk] ||
        ![plan.steps containsObject:baseline.softmax] ||
        ![plan.steps containsObject:baseline.pv]) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch baseline Q/K geometry or graph-step provenance changed");
      return false;
    }
    classes[std::make_pair(queryCount, keyCount)].push_back(tile);
  }
  const std::pair<uint32_t, uint32_t> classKeys[3] = {
    {384u, 3456u}, {384u, 2880u}, {192u, 2880u}
  };
  const uint32_t classCounts[3] = {40u, 10u, 10u};
  for (uint32_t index = 0; index < 3u; ++index) {
    auto found = classes.find(classKeys[index]);
    if (found == classes.end() || found->second.size() != classCounts[index] ||
        (found->second.size() % 2u) != 0u) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch Q/K class multiplicity differs from full-J layout");
      return false;
    }
  }
  if (classes.size() != 3u) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch full-J layout contains an unrecognized Q/K class");
    return false;
  }
  if (queryTotal != 21120u ||
      std::find(querySeen.begin(), querySeen.end(), uint8_t(0)) !=
          querySeen.end()) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch query scatter does not cover every full-J token once");
    return false;
  }
  return true;
}

static NSArray<PixelUMMWindowBatchPair *> *pixel_window_batch_make_pairs(
    PixelUMMDecoderPlan *plan, PixelUMMWindowBatchPool *pool,
    char *errorText, size_t errorCapacity) {
  const int32_t *queryTable = (const int32_t *)plan.scratch[6].contents;
  const int32_t *keyTable = (const int32_t *)plan.scratch[7].contents;
  std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> classes;
  for (uint32_t tile = 0; tile < plan.tileCount; ++tile) {
    const uint32_t queryCount =
        (uint32_t)queryTable[uint64_t(tile) * 385u];
    const uint32_t keyCount =
        (uint32_t)keyTable[uint64_t(tile) * 3457u];
    classes[std::make_pair(queryCount, keyCount)].push_back(tile);
  }
  NSMutableArray<PixelUMMWindowBatchPair *> *pairs =
      [NSMutableArray arrayWithCapacity:kPixelWindowBatchPairs];
  const std::pair<uint32_t, uint32_t> classKeys[3] = {
    {384u, 3456u}, {384u, 2880u}, {192u, 2880u}
  };
  for (const auto &key : classKeys) {
    auto found = classes.find(key);
    NSUInteger geometryIndex = NSNotFound;
    if (found == classes.end() || (found->second.size() % 2u) != 0u ||
        !pixel_window_batch_find_geometry(pool, key.first, key.second,
                                           &geometryIndex)) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch cannot pair a full-J Q/K class");
      return nil;
    }
    const std::vector<uint32_t> &tiles = found->second;
    for (size_t index = 0; index < tiles.size(); index += 2u) {
      PixelUMMWindowBatchPair *pair = [PixelUMMWindowBatchPair new];
      pair.tile0 = tiles[index];
      pair.tile1 = tiles[index + 1u];
      pair.queryCount = key.first;
      pair.keyCount = key.second;
      pair.geometryIndex = geometryIndex;
      [pairs addObject:pair];
    }
  }
  if (pairs.count != kPixelWindowBatchPairs) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch did not produce exactly 30 paired windows");
    return nil;
  }
  return [pairs copy];
}

static bool pixel_window_batch_expected_metric_count(
    NSArray<PixelUMMWindowBatchPair *> *pairs, int32_t mode,
    uint32_t *countOut) {
  if (!pairs || pairs.count != kPixelWindowBatchPairs || !countOut ||
      mode < PixelWindowBatchCheckRawQK ||
      mode > PixelWindowBatchCheckPositiveRawQK) return false;
  uint64_t total = 0;
  const NSUInteger limit = mode == PixelWindowBatchCheckPositiveRawQK
      ? 1u : pairs.count;
  for (NSUInteger index = 0; index < limit; ++index) {
    PixelUMMWindowBatchPair *pair = pairs[index];
    const uint64_t elements = mode == PixelWindowBatchCheckPV
        ? uint64_t(2u) * 32u * pair.queryCount * 128u
        : uint64_t(2u) * 32u * pair.queryCount * pair.keyCount;
    if (!pixel_window_batch_add(total, elements, &total) ||
        total > UINT32_MAX) return false;
  }
  if (total == 0u) return false;
  *countOut = (uint32_t)total;
  return true;
}

static bool pixel_window_batch_disjoint_from_plan(
    PixelUMMWindowBatchPool *pool, PixelUMMDecoderPlan *plan,
    id<MTLBuffer> metrics, char *errorText, size_t errorCapacity) {
  for (id<MTLBuffer> candidate in pool.panels) {
    for (id<MTLBuffer> resident in plan.resident) {
      if (!pixel_disjoint(candidate, resident)) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch candidate panel aliases a plan resident buffer");
        return false;
      }
    }
    if (!pixel_disjoint(candidate, metrics)) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch metrics alias a candidate panel");
      return false;
    }
  }
  for (id<MTLBuffer> resident in plan.resident) {
    if (!pixel_disjoint(metrics, resident)) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch metrics alias a plan resident buffer");
      return false;
    }
  }
  return true;
}

static bool pixel_window_batch_validate_poison_panels(
    PixelUMMWindowBatchPool *pool, PixelUMMDecoderPlan *plan,
    id<MTLBuffer> metrics, char *errorText, size_t errorCapacity) {
  if (!pool || !plan || !pool.panels || pool.panels.count != 6u ||
      pool.device != plan.device || !pool.q || !pool.k || !pool.v ||
      !pool.scores || !pool.probabilities || !pool.output) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch poison found incomplete shared panel state");
    return false;
  }
  uint64_t qBytes = 0, kvBytes = 0, scoreBytes = 0;
  if (!pixel_product(2u * 32u * kPixelWindowBatchQ * 128u,
                     sizeof(float), &qBytes) ||
      !pixel_product(2u * 32u * kPixelWindowBatchK * 128u,
                     sizeof(float), &kvBytes) ||
      !pixel_product(2u * 32u * kPixelWindowBatchQ * kPixelWindowBatchK,
                     sizeof(float), &scoreBytes)) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch poison panel extent overflowed");
    return false;
  }
  id<MTLBuffer> expected[6] = {
    pool.q, pool.k, pool.v, pool.scores, pool.probabilities, pool.output
  };
  const uint64_t expectedBytes[6] = {
    qBytes, kvBytes, kvBytes, scoreBytes, scoreBytes, qBytes
  };
  uint64_t totalBytes = 0;
  for (NSUInteger index = 0; index < 6u; ++index) {
    id<MTLBuffer> panel = pool.panels[index];
    if (!panel || panel != expected[index] ||
        !pixel_buffer(panel, plan.device, expectedBytes[index]) ||
        !pixel_window_batch_add(totalBytes, (uint64_t)panel.length,
                                &totalBytes)) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch poison requires six exact shared and accessible panels");
      return false;
    }
    for (NSUInteger prior = 0; prior < index; ++prior) {
      if (!pixel_disjoint(pool.panels[prior], panel)) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch poison candidate panels overlap");
        return false;
      }
    }
  }
  if (totalBytes != pool.extraBytes ||
      !pixel_window_batch_disjoint_from_plan(pool, plan, metrics,
          errorText, errorCapacity)) {
    if (!errorText || !errorCapacity || errorText[0] == '\0')
      pixel_set_error(errorText, errorCapacity,
          "window-batch poison byte accounting changed");
    return false;
  }
  return true;
}

static PixelUMMAttentionGeometry *pixel_window_batch_baseline_geometry(
    PixelUMMDecoderPlan *plan, uint32_t tile) {
  if (!plan || tile >= plan.tileGeometryIndices.count) return nil;
  NSUInteger index = plan.tileGeometryIndices[tile].unsignedIntegerValue;
  if (index >= plan.attentionGeometries.count) return nil;
  return plan.attentionGeometries[index];
}

static bool pixel_window_batch_encode_gather(
    PixelUMMWindowBatch *wrapper, MPSCommandBuffer *commandBuffer,
    PixelUMMWindowBatchPair *pair, bool gatherQ, bool gatherKV,
    char *errorText, size_t errorCapacity) {
  PixelUMMDecoderPlan *plan = wrapper.plan;
  PixelUMMWindowBatchPool *pool = wrapper.pool;
  if (!plan || !pool || !pair) return false;
  PixelWindowBatchGroupParams params{
    pair.tile0, pair.tile1, plan.maxQuery, plan.maxKey,
    pair.queryCount, pair.keyCount
  };
  uint64_t panelCount = 0;
  if (gatherQ) {
    if (!pixel_product(2u * 32u * pair.queryCount * 128u, 1u,
                       &panelCount) ||
        !pixel_encode_kernel(commandBuffer, plan.device, pool.gatherQ,
            @[plan.core[1], plan.scratch[6], pool.q], &params,
            sizeof(params), panelCount, false, errorText, errorCapacity))
      return false;
  }
  if (gatherKV) {
    if (!pixel_product(2u * 32u * pair.keyCount * 128u, 1u,
                       &panelCount) ||
        !pixel_encode_kernel(commandBuffer, plan.device, pool.gatherKVPair,
            @[plan.core[2], plan.core[3], plan.scratch[7], pool.k, pool.v],
            &params, sizeof(params), panelCount, false,
            errorText, errorCapacity)) return false;
  }
  return true;
}

static bool pixel_window_batch_encode_baseline_qk(
    PixelUMMWindowBatch *wrapper, MPSCommandBuffer *commandBuffer,
    uint32_t tile, uint32_t queryCount, uint32_t keyCount,
    bool includeValue, char *errorText, size_t errorCapacity) {
  PixelUMMDecoderPlan *plan = wrapper.plan;
  if (!plan || tile >= plan.tileCount) return false;
  PixelUMMAttentionGeometry *geometry =
      pixel_window_batch_baseline_geometry(plan, tile);
  if (!geometry || geometry.queryCount != queryCount ||
      geometry.keyCount != keyCount) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch baseline tile geometry changed during encode");
    return false;
  }
  PixelGatherQParams qParams{tile, plan.maxQuery, queryCount};
  PixelGatherKVParams kvParams{tile, plan.maxKey, keyCount};
  const uint64_t panelCount = uint64_t(plan.heads) * queryCount * plan.headDim;
  const uint64_t kvPanelCount = uint64_t(plan.heads) * keyCount * plan.headDim;
  if (!pixel_encode_kernel(commandBuffer, plan.device, plan.gatherQ,
          @[plan.core[1], plan.scratch[6], plan.scratch[0]], &qParams,
          sizeof(qParams), panelCount, false, errorText, errorCapacity) ||
      !pixel_encode_kernel(commandBuffer, plan.device, plan.gatherKV,
          @[plan.core[2], plan.scratch[7], plan.scratch[1]], &kvParams,
          sizeof(kvParams), kvPanelCount, false, errorText, errorCapacity))
    return false;
  if (includeValue &&
      !pixel_encode_kernel(commandBuffer, plan.device, plan.gatherKV,
          @[plan.core[3], plan.scratch[7], plan.scratch[2]], &kvParams,
          sizeof(kvParams), kvPanelCount, false, errorText, errorCapacity))
    return false;
  return pixel_encode_graph(commandBuffer, geometry.qk,
                            errorText, errorCapacity);
}

static bool pixel_window_batch_encode_compare(
    PixelUMMWindowBatch *wrapper, MPSCommandBuffer *commandBuffer,
    id<MTLBuffer> candidate, id<MTLBuffer> reference,
    uint64_t candidateOffset, uint64_t referenceOffset, uint64_t count,
    bool injectFirst, char *errorText, size_t errorCapacity) {
  if (!wrapper || !wrapper.pool || !wrapper.metrics || !candidate ||
      !reference || count == 0 || count > UINT32_MAX ||
      candidateOffset > UINT32_MAX || referenceOffset > UINT32_MAX) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch comparator received an unsupported extent");
    return false;
  }
  const uint32_t count32 = (uint32_t)count;
  const uint32_t groupCount = (count32 + (128u * 1024u) - 1u) /
      (128u * 1024u);
  PixelWindowBatchCompareParams params{
    count32, (uint32_t)candidateOffset, (uint32_t)referenceOffset,
    std::max(1u, groupCount), uint32_t(injectFirst), 2.0e-5f, 2.0e-5f
  };
  return pixel_encode_kernel(commandBuffer, wrapper.plan.device,
      wrapper.pool.compareF32, @[candidate, reference, wrapper.metrics],
      &params, sizeof(params), params.groupCount, true,
      errorText, errorCapacity);
}

static bool pixel_window_batch_encode_clear_metrics(
    PixelUMMWindowBatch *wrapper, MPSCommandBuffer *commandBuffer,
    char *errorText, size_t errorCapacity) {
  id<MTLCommandBuffer> root = pixel_current_root(commandBuffer,
      wrapper.plan.device, errorText, errorCapacity);
  if (!root || !wrapper.metrics ||
      wrapper.metrics.length != kPixelWindowBatchMetricsWords * sizeof(uint32_t))
    return false;
  id<MTLBlitCommandEncoder> blit = [root blitCommandEncoder];
  if (!blit) {
    pixel_set_error(errorText, errorCapacity,
        "unable to open Metal blit encoder for window-batch metrics reset");
    return false;
  }
  [blit fillBuffer:wrapper.metrics
             range:NSMakeRange(0, kPixelWindowBatchMetricsWords * sizeof(uint32_t))
             value:0u];
  [blit endEncoding];
  return true;
}

static bool pixel_window_batch_encode_pack_f32(
    PixelUMMWindowBatch *wrapper, MPSCommandBuffer *commandBuffer,
    id<MTLBuffer> source, id<MTLBuffer> destination,
    uint64_t destinationOffset, uint64_t count,
    char *errorText, size_t errorCapacity) {
  if (!wrapper || !wrapper.pool || !source || !destination || count == 0 ||
      count > UINT32_MAX || destinationOffset > UINT32_MAX ||
      destinationOffset + count > destination.length / sizeof(float)) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch F32 panel pack extent is invalid");
    return false;
  }
  PixelWindowBatchCopyParams params{
    (uint32_t)count, (uint32_t)destinationOffset
  };
  return pixel_encode_kernel(commandBuffer, wrapper.plan.device,
      wrapper.pool.panelCopyF32, @[source, destination], &params, sizeof(params),
      count, false, errorText, errorCapacity);
}

static bool pixel_window_batch_encode_raw_qk_check(
    PixelUMMWindowBatch *wrapper, MPSCommandBuffer *commandBuffer,
    PixelUMMWindowBatchPair *pair, bool positiveControl,
    char *errorText, size_t errorCapacity) {
  PixelUMMWindowBatchGeometry *candidateGeometry =
      wrapper.pool.geometries[pair.geometryIndex];
  if (!pixel_window_batch_encode_gather(wrapper, commandBuffer, pair,
          true, true, errorText, errorCapacity) ||
      !pixel_encode_graph(commandBuffer, candidateGeometry.qk,
                          errorText, errorCapacity)) return false;
  const uint64_t scoreCount = uint64_t(wrapper.plan.heads) *
      pair.queryCount * pair.keyCount;
  for (uint32_t slot = 0; slot < 2u; ++slot) {
    const uint32_t tile = slot == 0u ? pair.tile0 : pair.tile1;
    if (!pixel_window_batch_encode_baseline_qk(wrapper, commandBuffer,
            tile, pair.queryCount, pair.keyCount, false,
            errorText, errorCapacity) ||
        !pixel_window_batch_encode_compare(wrapper, commandBuffer,
            wrapper.pool.scores, wrapper.plan.scratch[3],
            uint64_t(slot) * scoreCount, 0u, scoreCount,
            positiveControl && slot == 0u,
            errorText, errorCapacity)) return false;
  }
  return true;
}

static bool pixel_window_batch_encode_pv_check(
    PixelUMMWindowBatch *wrapper, MPSCommandBuffer *commandBuffer,
    PixelUMMWindowBatchPair *pair,
    char *errorText, size_t errorCapacity) {
  PixelUMMDecoderPlan *plan = wrapper.plan;
  PixelUMMWindowBatchGeometry *candidateGeometry =
      wrapper.pool.geometries[pair.geometryIndex];
  if (!pixel_window_batch_encode_gather(wrapper, commandBuffer, pair,
          false, true, errorText, errorCapacity)) return false;
  const uint64_t scoreCount = uint64_t(plan.heads) *
      pair.queryCount * pair.keyCount;
  const uint64_t panelCount = uint64_t(plan.heads) *
      pair.queryCount * plan.headDim;
  for (uint32_t slot = 0; slot < 2u; ++slot) {
    const uint32_t tile = slot == 0u ? pair.tile0 : pair.tile1;
    if (!pixel_window_batch_encode_baseline_qk(wrapper, commandBuffer,
            tile, pair.queryCount, pair.keyCount, true,
            errorText, errorCapacity)) return false;
    PixelScaleParams scaleParams{(uint32_t)scoreCount,
        1.0f / sqrtf(128.0f)};
    PixelUMMAttentionGeometry *baseline =
        pixel_window_batch_baseline_geometry(plan, tile);
    if (!baseline ||
        !pixel_encode_kernel(commandBuffer, plan.device, plan.scaleScores,
            @[plan.scratch[3]], &scaleParams, sizeof(scaleParams),
            scoreCount, false, errorText, errorCapacity) ||
        !pixel_encode_graph(commandBuffer, baseline.softmax,
                            errorText, errorCapacity) ||
        !pixel_encode_graph(commandBuffer, baseline.pv,
                            errorText, errorCapacity) ||
        !pixel_window_batch_encode_pack_f32(wrapper, commandBuffer,
            plan.scratch[4], wrapper.pool.scores,
            uint64_t(slot) * scoreCount, scoreCount,
            errorText, errorCapacity) ||
        !pixel_window_batch_encode_pack_f32(wrapper, commandBuffer,
            plan.scratch[5], wrapper.pool.probabilities,
            uint64_t(slot) * panelCount, panelCount,
            errorText, errorCapacity)) return false;
  }
  if (!pixel_encode_graph(commandBuffer, candidateGeometry.pvBaseline,
                          errorText, errorCapacity)) return false;
  for (uint32_t slot = 0; slot < 2u; ++slot) {
    if (!pixel_window_batch_encode_compare(wrapper, commandBuffer,
            wrapper.pool.output, wrapper.pool.probabilities,
            uint64_t(slot) * panelCount, uint64_t(slot) * panelCount,
            panelCount, false, errorText, errorCapacity)) return false;
  }
  return true;
}

static bool pixel_window_batch_encode_candidate_attention(
    PixelUMMWindowBatch *wrapper, MPSCommandBuffer *commandBuffer,
    char *errorText, size_t errorCapacity) {
  PixelUMMDecoderPlan *plan = wrapper.plan;
  for (PixelUMMWindowBatchPair *pair in wrapper.pairs) {
    PixelUMMWindowBatchGeometry *geometry =
        wrapper.pool.geometries[pair.geometryIndex];
    if (!pixel_window_batch_encode_gather(wrapper, commandBuffer, pair,
            true, true, errorText, errorCapacity) ||
        !pixel_encode_graph(commandBuffer, geometry.qk,
                            errorText, errorCapacity)) return false;
    const uint64_t scoreCount = uint64_t(2u) * plan.heads *
        pair.queryCount * pair.keyCount;
    PixelScaleParams scaleParams{(uint32_t)scoreCount,
        1.0f / sqrtf(128.0f)};
    if (!pixel_encode_kernel(commandBuffer, plan.device, plan.scaleScores,
            @[wrapper.pool.scores], &scaleParams, sizeof(scaleParams),
            scoreCount, false, errorText, errorCapacity) ||
        !pixel_encode_graph(commandBuffer, geometry.softmax,
                            errorText, errorCapacity) ||
        !pixel_encode_graph(commandBuffer, geometry.pv,
                            errorText, errorCapacity)) return false;
    PixelWindowBatchScatterParams scatterParams{
      pair.tile0, pair.tile1, plan.maxQuery, pair.queryCount
    };
    const uint64_t panelCount = uint64_t(2u) * plan.heads *
        pair.queryCount * plan.headDim;
    if (!pixel_encode_kernel(commandBuffer, plan.device, wrapper.pool.scatter,
            @[wrapper.pool.output, plan.scratch[6], plan.core[4]],
            &scatterParams, sizeof(scatterParams), panelCount, false,
            errorText, errorCapacity)) return false;
  }
  return true;
}

static int32_t pixelumm_window_batch_encode_at_attention(
    PixelUMMDecoderPlan *plan, MPSCommandBuffer *commandBuffer,
    char *errorText, size_t errorCapacity) {
  if (!plan || !commandBuffer) return 0;
  PixelUMMWindowBatch *wrapper =
      objc_getAssociatedObject(plan, &pixel_window_batch_associated_key);
  if (!wrapper) return 0;
  if (!wrapper.active || wrapper.plan != plan || !wrapper.pool ||
      !wrapper.pairs || wrapper.pairs.count != kPixelWindowBatchPairs ||
      !wrapper.metrics || wrapper.pool.device != plan.device) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch hook found stale or incomplete plan state");
    return -1;
  }
  if (plan.useNativeIslands || plan.useFusedKVGather ||
      plan.softmaxPVPrepared || plan.useSoftmaxPV ||
      plan.softmaxPVOutputOnlyPrepared || plan.useSoftmaxPVOutputOnly ||
      plan.scaleSoftmaxPrepared || plan.useScaleSoftmax ||
      plan.softmaxPVCompare) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch cannot encode with another attention diagnostic selected");
    return -1;
  }
  if (wrapper.selected) {
    if (wrapper.checkMode != PixelWindowBatchCheckOff) {
      pixel_set_error(errorText, errorCapacity,
          "window-batch connected inference cannot run with a stage check enabled");
      return -1;
    }
    return pixel_window_batch_encode_candidate_attention(
        wrapper, commandBuffer, errorText, errorCapacity) ? 1 : -1;
  }
  if (wrapper.checkMode == PixelWindowBatchCheckOff) return 0;
  if (!pixel_window_batch_encode_clear_metrics(
          wrapper, commandBuffer, errorText, errorCapacity)) return -1;
  if (wrapper.checkMode == PixelWindowBatchCheckRawQK ||
      wrapper.checkMode == PixelWindowBatchCheckPositiveRawQK) {
    const bool positive =
        wrapper.checkMode == PixelWindowBatchCheckPositiveRawQK;
    const NSUInteger pairCount = positive ? 1u : wrapper.pairs.count;
    for (NSUInteger index = 0; index < pairCount; ++index) {
      if (!pixel_window_batch_encode_raw_qk_check(wrapper, commandBuffer,
              wrapper.pairs[index], positive, errorText, errorCapacity))
        return -1;
    }
    return 0;
  }
  if (wrapper.checkMode == PixelWindowBatchCheckPV) {
    for (PixelUMMWindowBatchPair *pair in wrapper.pairs) {
      if (!pixel_window_batch_encode_pv_check(wrapper, commandBuffer,
              pair, errorText, errorCapacity)) return -1;
    }
    return 0;
  }
  pixel_set_error(errorText, errorCapacity,
      "window-batch hook received an invalid stage check mode");
  return -1;
}

extern "C" void *gs_pixelumm_window_batch_prepare(
    void *planHandle, void *sharedWrapperHandle,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch preparation requires a native PixelUMM plan");
    return nullptr;
  }
  @autoreleasepool {
    @try {
      PixelUMMDecoderPlan *plan = (__bridge PixelUMMDecoderPlan *)planHandle;
      @synchronized(plan) {
        if (objc_getAssociatedObject(plan,
                &pixel_window_batch_associated_key)) {
          pixel_set_error(errorText, errorCapacity,
              "PixelUMM plan already has a window-batch wrapper");
          return nullptr;
        }
      }
      if (!pixel_window_batch_validate_plan(plan, true,
              errorText, errorCapacity)) return nullptr;

      PixelUMMWindowBatchPool *pool = nil;
      if (sharedWrapperHandle) {
        PixelUMMWindowBatch *shared =
            (__bridge PixelUMMWindowBatch *)sharedWrapperHandle;
        PixelUMMDecoderPlan *sharedPlan = shared.plan;
        if (!shared.active || !shared.pool || !sharedPlan ||
            shared.selected || shared.checkMode != PixelWindowBatchCheckOff) {
          pixel_set_error(errorText, errorCapacity,
              "window-batch shared wrapper is stale or selected");
          return nullptr;
        }
        pool = shared.pool;
        @synchronized(pool) {
          @synchronized(sharedPlan) {
            if (!shared.active || shared.pool != pool ||
                objc_getAssociatedObject(sharedPlan,
                    &pixel_window_batch_associated_key) != shared ||
                pool.device != plan.device || pool.wrapperCount == 0u ||
                pool.wrapperCount >= 36u) {
              pixel_set_error(errorText, errorCapacity,
                  "window-batch shared pool is stale, mixed-device or full");
              return nullptr;
            }
          }
        }
      } else {
        pool = pixel_window_batch_create_pool(plan.device,
            errorText, errorCapacity);
        if (!pool) return nullptr;
      }

      NSArray<PixelUMMWindowBatchPair *> *pairs =
          pixel_window_batch_make_pairs(plan, pool,
              errorText, errorCapacity);
      if (!pairs) return nullptr;
      id<MTLBuffer> metrics = [plan.device newBufferWithLength:
          kPixelWindowBatchMetricsWords * sizeof(uint32_t)
          options:MTLResourceStorageModeShared];
      if (!pixel_buffer(metrics, plan.device,
              kPixelWindowBatchMetricsWords * sizeof(uint32_t))) {
        pixel_set_error(errorText, errorCapacity,
            "unable to allocate window-batch metric storage");
        return nullptr;
      }
      std::memset(metrics.contents, 0,
          kPixelWindowBatchMetricsWords * sizeof(uint32_t));
      if (!pixel_window_batch_disjoint_from_plan(pool, plan, metrics,
              errorText, errorCapacity)) return nullptr;

      PixelUMMWindowBatch *wrapper = [PixelUMMWindowBatch new];
      wrapper.plan = plan;
      wrapper.pool = pool;
      wrapper.pairs = pairs;
      wrapper.metrics = metrics;
      wrapper.active = NO;
      wrapper.selected = NO;
      wrapper.checkMode = PixelWindowBatchCheckOff;

      @synchronized(pool) {
        @synchronized(plan) {
          if (plan.encodingStarted ||
              objc_getAssociatedObject(plan,
                  &pixel_window_batch_associated_key) ||
              pool.wrapperCount >= 36u ||
              !pixel_window_batch_validate_plan(plan, true,
                  errorText, errorCapacity)) {
            if (!errorText || !errorCapacity || errorText[0] == '\0')
              pixel_set_error(errorText, errorCapacity,
                  "window-batch prepare lost its idle-plan precondition");
            return nullptr;
          }
          wrapper.active = YES;
          ++pool.wrapperCount;
          objc_setAssociatedObject(plan, &pixel_window_batch_associated_key,
              wrapper, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        }
      }
      return (__bridge_retained void *)wrapper;
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"window-batch prepare exception %@: %@",
           exception.name, exception.reason]);
      return nullptr;
    }
  }
}

extern "C" int32_t gs_pixelumm_window_batch_select(
    void *wrapperHandle, int32_t enabled,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!wrapperHandle || (enabled != 0 && enabled != 1)) {
    pixel_set_error(errorText, errorCapacity,
        "invalid window-batch candidate selection");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMWindowBatch *wrapper =
          (__bridge PixelUMMWindowBatch *)wrapperHandle;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      PixelUMMWindowBatchPool *pool = wrapper.pool;
      if (!wrapper.active || !plan || !pool || !wrapper.pairs ||
          (enabled != 0 && wrapper.checkMode != PixelWindowBatchCheckOff)) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch selection requires an active idle wrapper and check mode off");
        return -2;
      }
      @synchronized(pool) {
        @synchronized(plan) {
          if (!wrapper.active || wrapper.pool != pool ||
              objc_getAssociatedObject(plan,
                  &pixel_window_batch_associated_key) != wrapper ||
              pool.device != plan.device ||
              !pixel_window_batch_validate_plan(plan, false,
                  errorText, errorCapacity)) {
            if (!errorText || !errorCapacity || errorText[0] == '\0')
              pixel_set_error(errorText, errorCapacity,
                  "window-batch selection found stale or foreign plan state");
            return -3;
          }
          wrapper.selected = enabled != 0;
          return 0;
        }
      }
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"window-batch selection exception %@: %@",
           exception.name, exception.reason]);
      return -4;
    }
  }
}

extern "C" int32_t gs_pixelumm_window_batch_set_check_mode(
    void *wrapperHandle, int32_t mode,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!wrapperHandle || mode < PixelWindowBatchCheckOff ||
      mode > PixelWindowBatchCheckPositiveRawQK) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch check mode must be 0, 1, 2 or 3");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMWindowBatch *wrapper =
          (__bridge PixelUMMWindowBatch *)wrapperHandle;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      PixelUMMWindowBatchPool *pool = wrapper.pool;
      if (!wrapper.active || !plan || !pool ||
          (mode != PixelWindowBatchCheckOff && wrapper.selected)) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch checks require an active unselected plan");
        return -2;
      }
      uint32_t expectedMetricCount = 0;
      if (mode != PixelWindowBatchCheckOff &&
          !pixel_window_batch_expected_metric_count(
              wrapper.pairs, mode, &expectedMetricCount)) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch stage metric extent is empty or exceeds UInt32");
        return -2;
      }
      @synchronized(pool) {
        @synchronized(plan) {
          if (!wrapper.active || wrapper.pool != pool ||
              objc_getAssociatedObject(plan,
                  &pixel_window_batch_associated_key) != wrapper ||
              pool.device != plan.device ||
              !pixel_window_batch_validate_plan(plan, false,
                  errorText, errorCapacity)) {
            if (!errorText || !errorCapacity || errorText[0] == '\0')
              pixel_set_error(errorText, errorCapacity,
                  "window-batch check mode found stale or foreign plan state");
            return -3;
          }
          wrapper.checkMode = mode;
          return 0;
        }
      }
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"window-batch check-mode exception %@: %@",
           exception.name, exception.reason]);
      return -4;
    }
  }
}

extern "C" void *gs_pixelumm_window_batch_metrics_buffer(void *wrapperHandle) {
  if (!wrapperHandle) return nullptr;
  @autoreleasepool {
    PixelUMMWindowBatch *wrapper =
        (__bridge PixelUMMWindowBatch *)wrapperHandle;
    if (!wrapper.active || !wrapper.metrics) return nullptr;
    return (__bridge void *)wrapper.metrics;
  }
}

extern "C" uint64_t gs_pixelumm_window_batch_metrics_bytes(
    void *wrapperHandle) {
  if (!wrapperHandle) return 0;
  @autoreleasepool {
    PixelUMMWindowBatch *wrapper =
        (__bridge PixelUMMWindowBatch *)wrapperHandle;
    return wrapper.active && wrapper.metrics
        ? (uint64_t)wrapper.metrics.length : 0u;
  }
}

// Test setup only: the serialized Crystal owner must establish that no
// command using this shared pool is in flight before calling this host fill.
// All six panels are validated first so a failure cannot leave a partial poison.
extern "C" int32_t gs_pixelumm_window_batch_poison(
    void *wrapperHandle, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!wrapperHandle) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch poison requires a live wrapper");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMWindowBatch *wrapper =
          (__bridge PixelUMMWindowBatch *)wrapperHandle;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      PixelUMMWindowBatchPool *pool = wrapper.pool;
      if (!wrapper.active || !plan || !pool || !wrapper.metrics ||
          (wrapper.selected && wrapper.checkMode != PixelWindowBatchCheckOff) ||
          wrapper.checkMode < PixelWindowBatchCheckOff ||
          wrapper.checkMode > PixelWindowBatchCheckPositiveRawQK) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch poison requires an active consistently selected wrapper");
        return -2;
      }
      @synchronized(pool) {
        @synchronized(plan) {
          if (!wrapper.active || wrapper.plan != plan || wrapper.pool != pool ||
              objc_getAssociatedObject(plan,
                  &pixel_window_batch_associated_key) != wrapper ||
              pool.wrapperCount == 0u || pool.device != plan.device ||
              !pixel_window_batch_validate_plan(plan, false,
                  errorText, errorCapacity) ||
              !pixel_window_batch_validate_poison_panels(pool, plan,
                  wrapper.metrics, errorText, errorCapacity)) {
            if (!errorText || !errorCapacity || errorText[0] == '\0')
              pixel_set_error(errorText, errorCapacity,
                  "window-batch poison found stale, foreign, or busy plan state");
            return -3;
          }
          for (id<MTLBuffer> panel in pool.panels)
            std::memset(panel.contents, 0xff, panel.length);
          return 0;
        }
      }
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"window-batch poison exception %@: %@",
           exception.name, exception.reason]);
      return -4;
    }
  }
}

extern "C" uint64_t gs_pixelumm_window_batch_extra_bytes(
    void *wrapperHandle) {
  if (!wrapperHandle) return 0;
  @autoreleasepool {
    PixelUMMWindowBatch *wrapper =
        (__bridge PixelUMMWindowBatch *)wrapperHandle;
    return wrapper.active && wrapper.pool ? wrapper.pool.extraBytes : 0u;
  }
}

extern "C" int32_t gs_pixelumm_window_batch_copy_layout(
    void *wrapperHandle, uint32_t *values, uint64_t valueCount) {
  if (!wrapperHandle || !values) return -1;
  @autoreleasepool {
    PixelUMMWindowBatch *wrapper =
        (__bridge PixelUMMWindowBatch *)wrapperHandle;
    if (!wrapper.active || !wrapper.pairs ||
        wrapper.pairs.count != kPixelWindowBatchPairs ||
        valueCount < uint64_t(kPixelWindowBatchPairs) * 4u) return -2;
    for (NSUInteger index = 0; index < wrapper.pairs.count; ++index) {
      PixelUMMWindowBatchPair *pair = wrapper.pairs[index];
      values[index * 4u] = pair.tile0;
      values[index * 4u + 1u] = pair.tile1;
      values[index * 4u + 2u] = pair.queryCount;
      values[index * 4u + 3u] = pair.keyCount;
    }
    return 0;
  }
}

extern "C" int32_t gs_pixelumm_window_batch_release(
    void *wrapperHandle, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!wrapperHandle) {
    pixel_set_error(errorText, errorCapacity,
        "window-batch release requires a live wrapper");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMWindowBatch *wrapper =
          (__bridge PixelUMMWindowBatch *)wrapperHandle;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      PixelUMMWindowBatchPool *pool = wrapper.pool;
      if (!wrapper.active || !plan || !pool) {
        pixel_set_error(errorText, errorCapacity,
            "window-batch release requires an active wrapper");
        return -2;
      }
      @synchronized(pool) {
        @synchronized(plan) {
          if (objc_getAssociatedObject(plan,
                  &pixel_window_batch_associated_key) != wrapper ||
              pool.wrapperCount == 0u) {
            pixel_set_error(errorText, errorCapacity,
                "window-batch release found stale or foreign plan state");
            return -3;
          }
          objc_setAssociatedObject(plan, &pixel_window_batch_associated_key,
              nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
          wrapper.selected = NO;
          wrapper.checkMode = PixelWindowBatchCheckOff;
          wrapper.active = NO;
          --pool.wrapperCount;
        }
      }
      (void)CFBridgingRelease(wrapperHandle);
      return 0;
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"window-batch release exception %@: %@",
           exception.name, exception.reason]);
      return -4;
    }
  }
}
