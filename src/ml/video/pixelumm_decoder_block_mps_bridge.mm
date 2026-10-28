// Prepared, diagnostic PixelUMM decoder layer. The caller owns all Metal
// buffers and the MPSCommandChain; this bridge prepares all graphs/pipelines
// before encode and only records work against the chain's current root.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <time.h>
#include <vector>
#include "mps_api_profile.h"

extern "C" int32_t gs_mps_chain_root_count(void *, int32_t *);

// CPU-only metadata, supplied in a fixed caller-owned array before encoding.
// No counter sampling, command submission or GPU synchronization is added.
struct PixelUMMPhaseMarker {
  uint64_t wall_ns;
  uint64_t thread_cpu_ns;
  int32_t root_count;
  int32_t phase;
};
static constexpr uint32_t kPixelPhaseMarkerCount = 11;

class PixelUMMPhaseRecorder {
public:
  PixelUMMPhaseRecorder(void *chain, PixelUMMPhaseMarker *markers,
                       uint32_t capacity) : chain_(chain), markers_(markers),
                                            capacity_(capacity) {}
  bool mark(int32_t phase, int32_t apiPhase) {
    if (!markers_) return true;
    if (!chain_ || capacity_ != kPixelPhaseMarkerCount ||
        phase != int32_t(cursor_) || cursor_ >= capacity_ ||
        gs_mps_chain_api_profile_set_phase(chain_, apiPhase) != 0) return false;
    int32_t roots = 0;
    timespec wall{}, cpu{};
    if (gs_mps_chain_root_count(chain_, &roots) != 0 || roots <= 0 ||
        clock_gettime(CLOCK_MONOTONIC, &wall) != 0 ||
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu) != 0) return false;
    const uint64_t wallNS = uint64_t(wall.tv_sec) * 1000000000u + wall.tv_nsec;
    const uint64_t cpuNS = uint64_t(cpu.tv_sec) * 1000000000u + cpu.tv_nsec;
    if (cursor_ == 0) { baseWall_ = wallNS; baseCPU_ = cpuNS; }
    if (wallNS < baseWall_ || cpuNS < baseCPU_) return false;
    markers_[cursor_++] = {wallNS - baseWall_, cpuNS - baseCPU_, roots, phase};
    return true;
  }
private:
  void *chain_;
  PixelUMMPhaseMarker *markers_;
  uint32_t capacity_, cursor_ = 0;
  uint64_t baseWall_ = 0, baseCPU_ = 0;
};

class PixelUMMGraphAPIScope {
public:
  explicit PixelUMMGraphAPIScope(MPSCommandBuffer *commandBuffer) {
    if (gs_mps_api_profile_scope_begin)
      gs_mps_api_profile_scope_begin(GS_MPS_API_PROFILE_OP_GRAPH_ENCODE,
                                    (__bridge void *)commandBuffer, &scope_);
  }
  ~PixelUMMGraphAPIScope() {
    if (scope_.active && gs_mps_api_profile_scope_end)
      gs_mps_api_profile_scope_end(&scope_);
  }
private:
  gs_mps_api_profile_scope scope_{};
};

static const NSUInteger kPixelWeightCount = 13;
static const NSUInteger kPixelCoreCount = 9;
static const NSUInteger kPixelCaptureCount = 26;
static const NSUInteger kPixelScratchCount = 14;
static const uint32_t kPixelHeadsPerKV = 4;
static const uint32_t kPixelHeadDim = 128;

enum PixelCapture : NSUInteger {
  PixelInputNorm = 0, PixelQProj, PixelKProj, PixelVProj, PixelQNorm,
  PixelKNorm, PixelQRope, PixelKRope, PixelAttention, PixelOProj,
  PixelAttnResidual, PixelPostNorm, PixelGateProj, PixelUpProj,
  PixelSiluGate, PixelProduct, PixelDownProj, PixelOutput,
  PixelInputNormStatistics, PixelInputNormPreGain, PixelQReady, PixelKReady,
  PixelQNormStatistics, PixelKNormStatistics, PixelQNormInverse, PixelKNormInverse
};

@interface PixelUMMGraphStep : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic, copy) NSArray<MPSGraphTensorData *> *inputs;
@property(nonatomic, copy) NSArray<MPSGraphTensorData *> *outputs;
@property(nonatomic, copy) NSString *name;
@end
@implementation PixelUMMGraphStep
@end

@interface PixelUMMAttentionGeometry : NSObject
@property(nonatomic) uint32_t queryCount;
@property(nonatomic) uint32_t keyCount;
@property(nonatomic, strong) PixelUMMGraphStep *qk;
@property(nonatomic, strong) PixelUMMGraphStep *softmax;
@property(nonatomic, strong) PixelUMMGraphStep *pv;
@end
@implementation PixelUMMAttentionGeometry
@end

@interface PixelUMMDecoderPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *weights;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *core;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *captures;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *scratch;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *resident;
@property(nonatomic, strong) NSArray<PixelUMMGraphStep *> *steps;
@property(nonatomic, strong) NSArray<PixelUMMAttentionGeometry *> *attentionGeometries;
@property(nonatomic, copy) NSArray<NSNumber *> *tileGeometryIndices;
@property(nonatomic, strong) PixelUMMGraphStep *inputNorm;
@property(nonatomic, strong) PixelUMMGraphStep *inputNormGain;
@property(nonatomic, strong) PixelUMMGraphStep *qProjection;
@property(nonatomic, strong) PixelUMMGraphStep *kProjection;
@property(nonatomic, strong) PixelUMMGraphStep *vProjection;
@property(nonatomic, strong) PixelUMMGraphStep *qNormStatistics;
@property(nonatomic, strong) PixelUMMGraphStep *kNormStatistics;
@property(nonatomic, strong) PixelUMMGraphStep *qNormInverse;
@property(nonatomic, strong) PixelUMMGraphStep *kNormInverse;
@property(nonatomic, strong) PixelUMMGraphStep *oProjection;
@property(nonatomic, strong) PixelUMMGraphStep *postNorm;
@property(nonatomic, strong) PixelUMMGraphStep *postNormGain;
@property(nonatomic, strong) PixelUMMGraphStep *gateProjection;
@property(nonatomic, strong) PixelUMMGraphStep *upProjection;
@property(nonatomic, strong) PixelUMMGraphStep *sharedGateUpProjection;
@property(nonatomic, strong) PixelUMMGraphStep *downProjection;
@property(nonatomic, strong) id<MTLLibrary> metalLibrary;
@property(nonatomic, strong) id<MTLComputePipelineState> qkPrepare;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherQ;
@property(nonatomic, strong) id<MTLComputePipelineState> gatherKV;
@property(nonatomic, strong) id<MTLComputePipelineState> scaleScores;
@property(nonatomic, strong) id<MTLComputePipelineState> scatterAttention;
@property(nonatomic, strong) id<MTLComputePipelineState> residual;
@property(nonatomic, strong) id<MTLComputePipelineState> silu;
@property(nonatomic, strong) id<MTLComputePipelineState> product;
@property(nonatomic) uint32_t temporal;
@property(nonatomic) uint32_t height;
@property(nonatomic) uint32_t width;
@property(nonatomic) uint32_t tokens;
@property(nonatomic) uint32_t tile;
@property(nonatomic) uint32_t dim;
@property(nonatomic) uint32_t ffn;
@property(nonatomic) uint32_t heads;
@property(nonatomic) uint32_t kvHeads;
@property(nonatomic) uint32_t headDim;
@property(nonatomic) uint32_t maxQuery;
@property(nonatomic) uint32_t maxKey;
@property(nonatomic) uint32_t tileCount;
@property(nonatomic) BOOL ffnF32;
@property(nonatomic) BOOL sharedGateUpPrepared;
@property(nonatomic) BOOL useSharedGateUp;
@property(nonatomic) uint64_t captureMask;
@end
@implementation PixelUMMDecoderPlan
@end

struct PixelQKParams {
  uint32_t tokens;
  uint32_t heads;
  uint32_t captureNorm;
  uint32_t captureRope;
};
struct PixelGatherQParams {
  uint32_t tile;
  uint32_t maxQuery;
  uint32_t queryCount;
};
struct PixelGatherKVParams {
  uint32_t tile;
  uint32_t maxKey;
  uint32_t keyCount;
};
struct PixelScaleParams { uint32_t count; float scale; };
struct PixelScatterParams {
  uint32_t tile;
  uint32_t maxQuery;
  uint32_t queryCount;
};
struct PixelCountParams { uint32_t count; };

static void pixel_set_error(char *destination, size_t capacity,
                            const char *message) {
  if (!destination || capacity == 0)
    return;
  if (!message)
    message = "PixelUMM MPS bridge failed";
  (void)std::snprintf(destination, capacity, "%s", message);
}

static void pixel_set_ns_error(char *destination, size_t capacity,
                               NSString *message) {
  pixel_set_error(destination, capacity,
                  message ? message.UTF8String : nullptr);
}

static bool pixel_product(uint64_t a, uint64_t b, uint64_t *out) {
  if (!out || a == 0 || b == 0 || a > UINT64_MAX / b)
    return false;
  *out = a * b;
  return true;
}

static bool pixel_disjoint(id<MTLBuffer> left, id<MTLBuffer> right) {
  if (!left || !right || left == right || !left.contents || !right.contents ||
      left.length == 0 || right.length == 0)
    return false;
  uintptr_t a = (uintptr_t)left.contents;
  uintptr_t b = (uintptr_t)right.contents;
  if (left.length > UINTPTR_MAX - a || right.length > UINTPTR_MAX - b)
    return false;
  uintptr_t ae = a + (uintptr_t)left.length;
  uintptr_t be = b + (uintptr_t)right.length;
  return ae <= b || be <= a;
}

static NSArray<NSNumber *> *pixel_shape(std::initializer_list<uint32_t> dims) {
  NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:dims.size()];
  for (uint32_t dim : dims)
    [result addObject:@(dim)];
  return result;
}

static MPSGraphTensor *pixel_add_feed(
    MPSGraph *graph,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds,
    NSMutableArray<MPSGraphTensor *> *placeholders,
    NSMutableArray<MPSGraphTensorData *> *inputData, id<MTLBuffer> buffer,
    NSArray<NSNumber *> *shape, MPSDataType type, NSString *name) {
  if (!graph || !feeds || !placeholders || !inputData || !buffer || !shape ||
      !name)
    return nil;
  MPSGraphTensor *tensor = [graph placeholderWithShape:shape dataType:type name:name];
  MPSGraphShapedType *feedType =
      [[MPSGraphShapedType alloc] initWithShape:shape dataType:type];
  MPSGraphTensorData *data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:type];
  if (!tensor || !feedType || !data)
    return nil;
  feeds[tensor] = feedType;
  [placeholders addObject:tensor];
  [inputData addObject:data];
  return tensor;
}

static MPSGraphTensor *pixel_cast_bf16(MPSGraph *graph,
                                      MPSGraphTensor *tensor,
                                      NSString *name) {
  return graph && tensor
      ? [graph castTensor:tensor toType:MPSDataTypeBFloat16 name:name]
      : nil;
}

static PixelUMMGraphStep *pixel_compile_step_outputs(
    id<MTLDevice> device, MPSGraph *graph,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds,
    NSMutableArray<MPSGraphTensor *> *placeholders,
    NSMutableArray<MPSGraphTensorData *> *inputData,
    NSArray<MPSGraphTensor *> *targets,
    NSArray<id<MTLBuffer>> *outputBuffers,
    NSArray<NSArray<NSNumber *> *> *outputShapes, NSString *name,
    char *errorText, size_t errorCapacity) {
  if (!device || !graph || !feeds || !placeholders || !inputData || !targets ||
      !outputBuffers || !outputShapes || !name || targets.count == 0 ||
      targets.count != outputBuffers.count || targets.count != outputShapes.count) {
    pixel_set_error(errorText, errorCapacity, "invalid PixelUMM prepared graph step");
    return nil;
  }
  NSMutableArray<MPSGraphTensorData *> *outputData =
      [NSMutableArray arrayWithCapacity:targets.count];
  for (NSUInteger index = 0; index < targets.count; ++index) {
    MPSGraphTensor *target = targets[index];
    id<MTLBuffer> outputBuffer = outputBuffers[index];
    NSArray<NSNumber *> *outputShape = outputShapes[index];
    if (!target || !outputBuffer || !outputShape) {
      pixel_set_error(errorText, errorCapacity, "missing PixelUMM graph output binding");
      return nil;
    }
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer shape:outputShape dataType:target.dataType];
    if (!data || ![target.shape isEqualToArray:outputShape]) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"%@ output %lu shape/type cannot bind", name,
           (unsigned long)index]);
      return nil;
    }
    [outputData addObject:data];
  }
  MPSGraphExecutable *executable = [graph compileWithDevice:
      [MPSGraphDevice deviceWithMTLDevice:device]
                                    feeds:(MPSGraphTensorShapedTypeDictionary *)feeds
                            targetTensors:targets
                         targetOperations:nil
                    compilationDescriptor:nil];
  if (!executable) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to compile PixelUMM graph %@", name]);
    return nil;
  }
  executable.options = MPSGraphOptionsNone;
  NSMutableArray<MPSGraphTensorData *> *ordered =
      [NSMutableArray arrayWithCapacity:placeholders.count];
  for (MPSGraphTensor *feedTensor in executable.feedTensors) {
    NSUInteger index = [placeholders indexOfObjectIdenticalTo:feedTensor];
    if (index == NSNotFound || index >= inputData.count) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"feed order mismatch in %@", name]);
      return nil;
    }
    [ordered addObject:inputData[index]];
  }
  if (ordered.count != placeholders.count ||
      executable.feedTensors.count != placeholders.count) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"feed count mismatch in %@", name]);
    return nil;
  }
  MPSGraphExecutableExecutionDescriptor *execution =
      [MPSGraphExecutableExecutionDescriptor new];
  execution.waitUntilCompleted = NO;
  PixelUMMGraphStep *step = [PixelUMMGraphStep new];
  step.graph = graph;
  step.executable = executable;
  step.execution = execution;
  step.inputs = [ordered copy];
  step.outputs = [outputData copy];
  step.name = name;
  if (!step.graph || !step.executable || !step.execution ||
      !step.inputs || !step.outputs || !step.name) {
    pixel_set_error(errorText, errorCapacity, "unable to retain PixelUMM graph step");
    return nil;
  }
  return step;
}

static PixelUMMGraphStep *pixel_compile_step(
    id<MTLDevice> device, MPSGraph *graph,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feeds,
    NSMutableArray<MPSGraphTensor *> *placeholders,
    NSMutableArray<MPSGraphTensorData *> *inputData,
    MPSGraphTensor *target, id<MTLBuffer> outputBuffer,
    NSArray<NSNumber *> *outputShape, NSString *name,
    char *errorText, size_t errorCapacity) {
  return pixel_compile_step_outputs(device, graph, feeds, placeholders, inputData,
      @[target], @[outputBuffer], @[outputShape], name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_rms_prepare_step(
    id<MTLDevice> device, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> exponentBuffer, id<MTLBuffer> preGainBuffer,
    id<MTLBuffer> statisticsBuffer, uint32_t tokens, uint32_t dim,
    float epsilon, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *hiddenShape = pixel_shape({1, tokens, dim});
  NSArray *exponentShape = pixel_shape({1});
  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      inputBuffer, hiddenShape, MPSDataTypeBFloat16, [name stringByAppendingString:@"_input"]);
  MPSGraphTensor *exponent = pixel_add_feed(graph, feeds, placeholders, inputData,
      exponentBuffer, exponentShape, MPSDataTypeFloat32, [name stringByAppendingString:@"_power_exponent"]);
  MPSGraphTensor *inputF32 = input ? [graph castTensor:input toType:MPSDataTypeFloat32
      name:[name stringByAppendingString:@"_input_f32"]] : nil;
  MPSGraphTensor *power = inputF32 && exponent
      ? [graph powerWithPrimaryTensor:inputF32 secondaryTensor:exponent
                                  name:[name stringByAppendingString:@"_pow_f32"]] : nil;
  MPSGraphTensor *mean = power ? [graph meanOfTensor:power axes:@[@2]
      name:[name stringByAppendingString:@"_mean_f32"]] : nil;
  MPSGraphTensor *epsilonTensor = [graph constantWithScalar:epsilon shape:@[@1]
      dataType:MPSDataTypeFloat32];
  MPSGraphTensor *plus = mean ? [graph additionWithPrimaryTensor:mean
      secondaryTensor:epsilonTensor name:[name stringByAppendingString:@"_eps_f32"]] : nil;
  NSArray *statisticsShape = pixel_shape({1, tokens, 1});
  MPSGraphTensor *plusRows = plus ? [graph reshapeTensor:plus
      withShape:statisticsShape name:[name stringByAppendingString:@"_eps_rows_f32"]] : nil;
  MPSGraphTensor *inverse = plus ? [graph reciprocalSquareRootWithTensor:plus
      name:[name stringByAppendingString:@"_rsqrt_f32"]] : nil;
  MPSGraphTensor *inverseRows = inverse ? [graph reshapeTensor:inverse
      withShape:pixel_shape({1, tokens, 1}) name:[name stringByAppendingString:@"_rsqrt_rows"]] : nil;
  MPSGraphTensor *normalized = inputF32 && inverseRows
      ? [graph multiplicationWithPrimaryTensor:inputF32 secondaryTensor:inverseRows
          name:[name stringByAppendingString:@"_normalized_f32"]] : nil;
  MPSGraphTensor *normalizedBf16 = pixel_cast_bf16(graph, normalized,
      [name stringByAppendingString:@"_normalized_bf16"]);
  if (!input || !exponent || !power || !mean || !epsilonTensor ||
      !plus || !plusRows || !statisticsBuffer || !inverse || !inverseRows ||
      !normalized || !normalizedBf16) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM RMS graph %@", name]);
    return nil;
  }
  NSMutableArray<MPSGraphTensor *> *targets = [NSMutableArray arrayWithObject:normalizedBf16];
  [targets addObject:plusRows];
  NSMutableArray<id<MTLBuffer>> *outputs = [NSMutableArray arrayWithObjects:
      preGainBuffer, statisticsBuffer, nil];
  NSMutableArray<NSArray<NSNumber *> *> *shapes = [NSMutableArray arrayWithObjects:
      hiddenShape, statisticsShape, nil];
  return pixel_compile_step_outputs(device, graph, feeds, placeholders, inputData,
      targets, outputs, shapes, [name stringByAppendingString:@"_pre_gain"],
      errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_qk_statistics_step(
    id<MTLDevice> device, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> exponentBuffer, id<MTLBuffer> statisticsBuffer,
    uint32_t tokens, uint32_t heads, float epsilon, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *inputShape = pixel_shape({1, tokens, heads, 128});
  NSArray *exponentShape = pixel_shape({1});
  NSArray *statisticsShape = pixel_shape({1, tokens, heads, 2});
  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      inputBuffer, inputShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_input_bf16"]);
  MPSGraphTensor *exponent = pixel_add_feed(graph, feeds, placeholders, inputData,
      exponentBuffer, exponentShape, MPSDataTypeFloat32,
      [name stringByAppendingString:@"_power_exponent_f32"]);
  MPSGraphTensor *inputF32 = input
      ? [graph castTensor:input toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_input_f32"]] : nil;
  MPSGraphTensor *temporal = inputF32
      ? [graph sliceTensor:inputF32 dimension:3 start:0 length:64
          name:[name stringByAppendingString:@"_temporal_f32"]] : nil;
  MPSGraphTensor *spatial = inputF32
      ? [graph sliceTensor:inputF32 dimension:3 start:64 length:64
          name:[name stringByAppendingString:@"_spatial_f32"]] : nil;
  MPSGraphTensor *temporalPower = temporal && exponent
      ? [graph powerWithPrimaryTensor:temporal secondaryTensor:exponent
          name:[name stringByAppendingString:@"_temporal_pow_fed_f32"]] : nil;
  MPSGraphTensor *spatialPower = spatial && exponent
      ? [graph powerWithPrimaryTensor:spatial secondaryTensor:exponent
          name:[name stringByAppendingString:@"_spatial_pow_fed_f32"]] : nil;
  MPSGraphTensor *temporalMean = temporalPower
      ? [graph meanOfTensor:temporalPower axes:@[@3]
          name:[name stringByAppendingString:@"_temporal_mean_f32"]] : nil;
  MPSGraphTensor *spatialMean = spatialPower
      ? [graph meanOfTensor:spatialPower axes:@[@3]
          name:[name stringByAppendingString:@"_spatial_mean_f32"]] : nil;
  MPSGraphTensor *epsilonF32 = [graph constantWithScalar:epsilon
      shape:exponentShape dataType:MPSDataTypeFloat32];
  MPSGraphTensor *temporalPlus = temporalMean && epsilonF32
      ? [graph additionWithPrimaryTensor:temporalMean secondaryTensor:epsilonF32
          name:[name stringByAppendingString:@"_temporal_plus_epsilon_f32"]] : nil;
  MPSGraphTensor *spatialPlus = spatialMean && epsilonF32
      ? [graph additionWithPrimaryTensor:spatialMean secondaryTensor:epsilonF32
          name:[name stringByAppendingString:@"_spatial_plus_epsilon_f32"]] : nil;
  NSArray *halfShape = pixel_shape({1, tokens, heads, 1});
  MPSGraphTensor *temporalRows = temporalPlus
      ? [graph reshapeTensor:temporalPlus withShape:halfShape
          name:[name stringByAppendingString:@"_temporal_rows_f32"]] : nil;
  MPSGraphTensor *spatialRows = spatialPlus
      ? [graph reshapeTensor:spatialPlus withShape:halfShape
          name:[name stringByAppendingString:@"_spatial_rows_f32"]] : nil;
  MPSGraphTensor *statistics = temporalRows && spatialRows
      ? [graph concatTensors:@[temporalRows, spatialRows] dimension:3
          name:[name stringByAppendingString:@"_statistics_f32"]] : nil;
  if (!input || !exponent || !inputF32 || !temporal || !spatial ||
      !temporalPower || !spatialPower || !temporalMean || !spatialMean ||
      !epsilonF32 || !temporalPlus || !spatialPlus || !temporalRows ||
      !spatialRows || !statistics || !statisticsBuffer) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM QK statistics graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      statistics, statisticsBuffer, statisticsShape, name,
      errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_qk_inverse_step(
    id<MTLDevice> device, id<MTLBuffer> statisticsBuffer,
    id<MTLBuffer> inverseBuffer, uint32_t tokens, uint32_t heads,
    NSString *name, char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *shape = pixel_shape({1, tokens, heads, 2});
  MPSGraphTensor *statistics = pixel_add_feed(graph, feeds, placeholders,
      inputData, statisticsBuffer, shape, MPSDataTypeFloat32,
      [name stringByAppendingString:@"_statistics_f32"]);
  // PyTorch 2.6's MPS rsqrt path uses this legacy reverse-square-root op.
  // Preserve the operator identity here because the direct Metal rsqrt path
  // produces different BF16 Q/K inputs at rounding boundaries.
  MPSGraphTensor *inverse = statistics
      ? [graph reverseSquareRootWithTensor:statistics
          name:[name stringByAppendingString:@"_inverse_f32"]] : nil;
  if (!statistics || !inverse || !inverseBuffer) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM QK inverse-root graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      inverse, inverseBuffer, shape, name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_rms_gain_step(
    id<MTLDevice> device, id<MTLBuffer> preGainBuffer,
    id<MTLBuffer> gainBuffer, id<MTLBuffer> outputBuffer,
    uint32_t tokens, uint32_t dim, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *hiddenShape = pixel_shape({1, tokens, dim});
  NSArray *gainShape = pixel_shape({dim});
  MPSGraphTensor *normalizedBf16 = pixel_add_feed(graph, feeds, placeholders,
      inputData, preGainBuffer, hiddenShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_normalized_bf16"]);
  MPSGraphTensor *gainBf16 = pixel_add_feed(graph, feeds, placeholders,
      inputData, gainBuffer, gainShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_gain_bf16"]);
  MPSGraphTensor *normalizedF32 = normalizedBf16
      ? [graph castTensor:normalizedBf16 toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_normalized_f32"]] : nil;
  MPSGraphTensor *gainF32 = gainBf16
      ? [graph castTensor:gainBf16 toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_gain_f32"]] : nil;
  MPSGraphTensor *gainedF32 = normalizedF32 && gainF32
      ? [graph multiplicationWithPrimaryTensor:normalizedF32 secondaryTensor:gainF32
          name:[name stringByAppendingString:@"_gained_f32"]] : nil;
  MPSGraphTensor *gainedBf16 = pixel_cast_bf16(graph, gainedF32,
      [name stringByAppendingString:@"_gained_bf16"]);
  if (!normalizedBf16 || !gainBf16 || !normalizedF32 || !gainF32 ||
      !gainedF32 || !gainedBf16) {
    pixel_set_error(errorText, errorCapacity, "unable to build PixelUMM RMS gain graph");
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      gainedBf16, outputBuffer, hiddenShape, name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_linear_step(
    id<MTLDevice> device, id<MTLBuffer> inputBuffer, id<MTLBuffer> weightBuffer,
    id<MTLBuffer> outputBuffer, uint32_t tokens, uint32_t inputWidth,
    uint32_t outputWidth, NSString *name,
    char *errorText, size_t errorCapacity, bool castOperandsToF32) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *inputShape = pixel_shape({1, tokens, inputWidth});
  NSArray *weightShape = pixel_shape({outputWidth, inputWidth});
  NSArray *outputShape = pixel_shape({1, tokens, outputWidth});
  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      inputBuffer, inputShape, MPSDataTypeBFloat16, [name stringByAppendingString:@"_input"]);
  MPSGraphTensor *weight = pixel_add_feed(graph, feeds, placeholders, inputData,
      weightBuffer, weightShape, MPSDataTypeBFloat16, [name stringByAppendingString:@"_weight"]);
  MPSGraphTensor *matmulInput = castOperandsToF32 && input
      ? [graph castTensor:input toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_input_f32"]] : input;
  MPSGraphTensor *matmulWeight = castOperandsToF32 && weight
      ? [graph castTensor:weight toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_weight_f32"]] : weight;
  MPSGraphTensor *flat = matmulInput ? [graph reshapeTensor:matmulInput
      withShape:pixel_shape({tokens, inputWidth}) name:[name stringByAppendingString:@"_flat"]] : nil;
  MPSGraphTensor *transpose = matmulWeight ? [graph transposeTensor:matmulWeight dimension:0
      withDimension:1 name:[name stringByAppendingString:@"_weight_transpose"]] : nil;
  MPSGraphTensor *product = flat && transpose
      ? [graph matrixMultiplicationWithPrimaryTensor:flat secondaryTensor:transpose
          name:[name stringByAppendingString:@"_matmul"]] : nil;
  MPSGraphTensor *bf16 = product;
  if (bf16 && bf16.dataType != MPSDataTypeBFloat16)
    bf16 = pixel_cast_bf16(graph, bf16, [name stringByAppendingString:@"_bf16"]);
  MPSGraphTensor *reshaped = bf16 ? [graph reshapeTensor:bf16 withShape:outputShape
      name:[name stringByAppendingString:@"_output"]] : nil;
  if (!input || !weight || !matmulInput || !matmulWeight || !flat ||
      !transpose || !product || !bf16 || !reshaped) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM linear graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      reshaped, outputBuffer, outputShape, name, errorText, errorCapacity);
}

static MPSGraphTensor *pixel_linear_f32_weight_projection(
    MPSGraph *graph, MPSGraphTensor *flatInput, MPSGraphTensor *weightF32,
    uint32_t tokens, uint32_t outputWidth,
    NSString *name) {
  MPSGraphTensor *transpose = weightF32
      ? [graph transposeTensor:weightF32 dimension:0 withDimension:1
          name:[name stringByAppendingString:@"_weight_transpose"]] : nil;
  MPSGraphTensor *product = flatInput && transpose
      ? [graph matrixMultiplicationWithPrimaryTensor:flatInput secondaryTensor:transpose
          name:[name stringByAppendingString:@"_matmul"]] : nil;
  MPSGraphTensor *bf16 = product;
  if (bf16 && bf16.dataType != MPSDataTypeBFloat16)
    bf16 = pixel_cast_bf16(graph, bf16,
        [name stringByAppendingString:@"_bf16"]);
  return bf16
      ? [graph reshapeTensor:bf16 withShape:pixel_shape({1, tokens, outputWidth})
          name:[name stringByAppendingString:@"_output"]] : nil;
}

static PixelUMMGraphStep *pixel_linear_shared_input_pair_step(
    id<MTLDevice> device, id<MTLBuffer> inputBuffer,
    id<MTLBuffer> gateWeightBuffer, id<MTLBuffer> upWeightBuffer,
    id<MTLBuffer> gateOutputBuffer, id<MTLBuffer> upOutputBuffer,
    uint32_t tokens, uint32_t inputWidth, uint32_t outputWidth,
    NSString *name, char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *inputShape = pixel_shape({1, tokens, inputWidth});
  NSArray *weightShape = pixel_shape({outputWidth, inputWidth});
  NSArray *outputShape = pixel_shape({1, tokens, outputWidth});
  MPSGraphTensor *input = pixel_add_feed(graph, feeds, placeholders, inputData,
      inputBuffer, inputShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_input"]);
  MPSGraphTensor *gateWeight = pixel_add_feed(graph, feeds, placeholders, inputData,
      gateWeightBuffer, weightShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_gate_weight"]);
  MPSGraphTensor *upWeight = pixel_add_feed(graph, feeds, placeholders, inputData,
      upWeightBuffer, weightShape, MPSDataTypeBFloat16,
      [name stringByAppendingString:@"_up_weight"]);
  MPSGraphTensor *inputF32 = input
      ? [graph castTensor:input toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_input_f32"]] : nil;
  MPSGraphTensor *gateWeightF32 = gateWeight
      ? [graph castTensor:gateWeight toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_gate_weight_f32"]] : nil;
  MPSGraphTensor *upWeightF32 = upWeight
      ? [graph castTensor:upWeight toType:MPSDataTypeFloat32
          name:[name stringByAppendingString:@"_up_weight_f32"]] : nil;
  MPSGraphTensor *flatInput = inputF32
      ? [graph reshapeTensor:inputF32 withShape:pixel_shape({tokens, inputWidth})
          name:[name stringByAppendingString:@"_flat"]] : nil;
  MPSGraphTensor *gateOutput = pixel_linear_f32_weight_projection(
      graph, flatInput, gateWeightF32, tokens, outputWidth,
      [name stringByAppendingString:@"_gate"]);
  MPSGraphTensor *upOutput = pixel_linear_f32_weight_projection(
      graph, flatInput, upWeightF32, tokens, outputWidth,
      [name stringByAppendingString:@"_up"]);
  if (!input || !gateWeight || !upWeight || !inputF32 || !gateWeightF32 ||
      !upWeightF32 || !flatInput ||
      !gateOutput || !upOutput) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build shared-input PixelUMM graph %@", name]);
    return nil;
  }
  return pixel_compile_step_outputs(device, graph, feeds, placeholders, inputData,
      @[gateOutput, upOutput], @[gateOutputBuffer, upOutputBuffer],
      @[outputShape, outputShape], name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_attention_qk_step(
    id<MTLDevice> device, id<MTLBuffer> qBuffer, id<MTLBuffer> kBuffer,
    id<MTLBuffer> scoresBuffer, uint32_t queryCount, uint32_t keyCount,
    NSString *name, char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  // PyTorch 2.6 _matmul_impl flattens the [1, 32] batch prefix and calls
  // bmm with rank-3 tensors, then views the result back as [1, 32, Q, K].
  // Keep that rank boundary for MPSGraph's matmul and restore rank 4 for the
  // eager softmax path; the reshape does not allocate another panel.
  NSArray *qShape = pixel_shape({32, queryCount, 128});
  NSArray *kShape = pixel_shape({32, keyCount, 128});
  NSArray *scoreShape3 = pixel_shape({32, queryCount, keyCount});
  NSArray *scoreShape4 = pixel_shape({1, 32, queryCount, keyCount});
  MPSGraphTensor *q = pixel_add_feed(graph, feeds, placeholders, inputData,
      qBuffer, qShape, MPSDataTypeFloat32, @"local_q_f32");
  MPSGraphTensor *k = pixel_add_feed(graph, feeds, placeholders, inputData,
      kBuffer, kShape, MPSDataTypeFloat32, @"local_k_f32");
  MPSGraphTensor *kTranspose = k ? [graph transposeTensor:k dimension:1
      withDimension:2 name:@"local_k_transpose_f32"] : nil;
  MPSGraphTensor *scores3 = q && kTranspose
      ? [graph matrixMultiplicationWithPrimaryTensor:q secondaryTensor:kTranspose
          name:@"local_qk_scores_bmm_f32"] : nil;
  MPSGraphTensor *scores4 = scores3
      ? [graph reshapeTensor:scores3 withShape:scoreShape4
          name:@"local_qk_scores_rank4_view"] : nil;
  if (!q || !k || !kTranspose || !scores3 || !scores4 ||
      ![scores3.shape isEqualToArray:scoreShape3]) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM QK graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      scores4, scoresBuffer, scoreShape4, name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_attention_softmax_step(
    id<MTLDevice> device, id<MTLBuffer> scoresBuffer,
    id<MTLBuffer> probabilitiesBuffer, uint32_t queryCount,
    uint32_t keyCount, NSString *name, char *errorText,
    size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  NSArray *shape = pixel_shape({1, 32, queryCount, keyCount});
  MPSGraphTensor *scores = pixel_add_feed(graph, feeds, placeholders, inputData,
      scoresBuffer, shape, MPSDataTypeFloat32, @"local_scaled_scores_f32");
  MPSGraphTensor *probabilities = scores
      ? [graph softMaxWithTensor:scores axis:3 name:@"local_softmax_f32"] : nil;
  if (!scores || !probabilities) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM softmax graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      probabilities, probabilitiesBuffer, shape, name, errorText, errorCapacity);
}

static PixelUMMGraphStep *pixel_attention_pv_step(
    id<MTLDevice> device, id<MTLBuffer> probabilitiesBuffer,
    id<MTLBuffer> vBuffer, id<MTLBuffer> outputBuffer,
    uint32_t queryCount, uint32_t keyCount, NSString *name,
    char *errorText, size_t errorCapacity) {
  MPSGraph *graph = [MPSGraph new];
  graph.options = MPSGraphOptionsNone;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  NSMutableArray *placeholders = [NSMutableArray array];
  NSMutableArray *inputData = [NSMutableArray array];
  // Eager softmax consumes and returns rank 4. PyTorch then folds the first
  // two dimensions for the rank-3 PV bmm; this view preserves the same bytes.
  NSArray *probabilityShape4 = pixel_shape({1, 32, queryCount, keyCount});
  NSArray *probabilityShape3 = pixel_shape({32, queryCount, keyCount});
  NSArray *vShape = pixel_shape({32, keyCount, 128});
  NSArray *outputShape3 = pixel_shape({32, queryCount, 128});
  NSArray *outputShape4 = pixel_shape({1, 32, queryCount, 128});
  MPSGraphTensor *probabilities4 = pixel_add_feed(graph, feeds, placeholders,
      inputData, probabilitiesBuffer, probabilityShape4, MPSDataTypeFloat32,
      @"local_probabilities_f32");
  MPSGraphTensor *probabilities3 = probabilities4
      ? [graph reshapeTensor:probabilities4 withShape:probabilityShape3
          name:@"local_probabilities_rank3_view"] : nil;
  MPSGraphTensor *v = pixel_add_feed(graph, feeds, placeholders, inputData,
      vBuffer, vShape, MPSDataTypeFloat32, @"local_v_f32");
  MPSGraphTensor *attended3 = probabilities3 && v
      ? [graph matrixMultiplicationWithPrimaryTensor:probabilities3 secondaryTensor:v
          name:@"local_pv_bmm_f32"] : nil;
  MPSGraphTensor *attended4 = attended3
      ? [graph reshapeTensor:attended3 withShape:outputShape4
          name:@"local_pv_rank4_view"] : nil;
  if (!probabilities4 || !probabilities3 || !v || !attended3 || !attended4 ||
      ![attended3.shape isEqualToArray:outputShape3]) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to build PixelUMM PV graph %@", name]);
    return nil;
  }
  return pixel_compile_step(device, graph, feeds, placeholders, inputData,
      attended4, outputBuffer, outputShape4, name, errorText, errorCapacity);
}

static const char *pixel_metal_source = R"metal(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct PixelQKParams {
  uint tokens;
  uint heads;
  uint captureNorm;
  uint captureRope;
};
struct PixelGatherQParams { uint tile; uint maxQuery; uint queryCount; };
struct PixelGatherKVParams { uint tile; uint maxKey; uint keyCount; };
struct PixelScaleParams { uint count; float scale; };
struct PixelScatterParams { uint tile; uint maxQuery; uint queryCount; };
struct PixelCountParams { uint count; };

inline float pixel_bf16_to_float(ushort value) {
  return as_type<float>(uint(value) << 16);
}
inline ushort pixel_float_to_bf16(float value) {
  uint bits = as_type<uint>(value);
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u)
    return ushort((bits >> 16) | 0x40u);
  return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

kernel void pixelumm_qk_prepare(
    device ushort *values [[buffer(0)]],
    device const ushort *gainT [[buffer(1)]],
    device const ushort *gainHW [[buffer(2)]],
    device const float *cosine [[buffer(3)]],
    device const float *sine [[buffer(4)]],
    device float *normCapture [[buffer(5)]],
    device float *ropeCapture [[buffer(6)]],
    device const float *inverseRoots [[buffer(7)]],
    constant PixelQKParams &p [[buffer(8)]],
    uint lane [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint simdLane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]) {
  const uint row = group / p.heads;
  const uint head = group % p.heads;
  const uint base = (row * p.heads + head) * 128u;
  threadgroup float normalized[128];
  threadgroup float inverse[2];
  const float x = pixel_bf16_to_float(values[base + lane]);
  if (simdGroup == 0u && simdLane == 0u) {
    const uint inverseBase = (row * p.heads + head) * 2u;
    inverse[0] = inverseRoots[inverseBase];
    inverse[1] = inverseRoots[inverseBase + 1u];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool temporalPart = lane < 64u;
  const uint normIndex = temporalPart ? lane : lane - 64u;
  const float gain = pixel_bf16_to_float(temporalPart ? gainT[normIndex] : gainHW[normIndex]);
  const float n = x * inverse[temporalPart ? 0 : 1] * gain;
  normalized[lane] = n;
  if (p.captureNorm != 0u)
    normCapture[base + lane] = n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sectionStart = 0u;
  uint sectionWidth = 64u;
  if (lane >= 64u && lane < 96u) { sectionStart = 64u; sectionWidth = 32u; }
  else if (lane >= 96u) { sectionStart = 96u; sectionWidth = 32u; }
  const uint offset = lane - sectionStart;
  const uint partner = offset < sectionWidth / 2u
      ? sectionStart + offset + sectionWidth / 2u
      : sectionStart + offset - sectionWidth / 2u;
  const float rotated = offset < sectionWidth / 2u ? -normalized[partner] : normalized[partner];
  const uint position = row * 128u + lane;
  const float rope = n * cosine[position] + rotated * sine[position];
  if (p.captureRope != 0u)
    ropeCapture[base + lane] = rope;
  values[base + lane] = pixel_float_to_bf16(rope);
}

kernel void pixelumm_gather_q(
    device const ushort *source [[buffer(0)]],
    device const int *queryIndices [[buffer(1)]],
    device float *destination [[buffer(2)]],
    constant PixelGatherQParams &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint total = 32u * p.queryCount * 128u;
  if (index >= total) return;
  const uint head = index / (p.queryCount * 128u);
  const uint row = (index / 128u) % p.queryCount;
  const uint dim = index % 128u;
  const uint tableBase = p.tile * (p.maxQuery + 1u);
  const uint count = uint(queryIndices[tableBase]);
  if (count != p.queryCount || row >= count) return;
  const uint token = uint(queryIndices[tableBase + 1u + row]);
  destination[index] = pixel_bf16_to_float(source[(token * 32u + head) * 128u + dim]);
}

kernel void pixelumm_gather_kv(
    device const ushort *source [[buffer(0)]],
    device const int *keyIndices [[buffer(1)]],
    device float *destination [[buffer(2)]],
    constant PixelGatherKVParams &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint total = 32u * p.keyCount * 128u;
  if (index >= total) return;
  const uint head = index / (p.keyCount * 128u);
  const uint row = (index / 128u) % p.keyCount;
  const uint dim = index % 128u;
  const uint tableBase = p.tile * (p.maxKey + 1u);
  const uint count = uint(keyIndices[tableBase]);
  if (count != p.keyCount || row >= count) return;
  const uint token = uint(keyIndices[tableBase + 1u + row]);
  destination[index] = pixel_bf16_to_float(source[(token * 8u + head / 4u) * 128u + dim]);
}

kernel void pixelumm_scale_scores(
    device float *scores [[buffer(0)]],
    constant PixelScaleParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= p.count) return;
  scores[index] = scores[index] * p.scale;
}

kernel void pixelumm_scatter_attention(
    device const float *source [[buffer(0)]],
    device const int *queryIndices [[buffer(1)]],
    device ushort *destination [[buffer(2)]],
    constant PixelScatterParams &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint total = 32u * p.queryCount * 128u;
  if (index >= total) return;
  const uint head = index / (p.queryCount * 128u);
  const uint row = (index / 128u) % p.queryCount;
  const uint dim = index % 128u;
  const uint tableBase = p.tile * (p.maxQuery + 1u);
  const uint count = uint(queryIndices[tableBase]);
  if (count != p.queryCount || row >= count) return;
  const uint token = uint(queryIndices[tableBase + 1u + row]);
  destination[(token * 32u + head) * 128u + dim] = pixel_float_to_bf16(source[index]);
}

kernel void pixelumm_residual(
    device const ushort *left [[buffer(0)]],
    device const ushort *right [[buffer(1)]],
    device ushort *output [[buffer(2)]],
    constant PixelCountParams &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= p.count) return;
  output[index] = pixel_float_to_bf16(pixel_bf16_to_float(left[index]) +
                                      pixel_bf16_to_float(right[index]));
}

kernel void pixelumm_silu(
    device ushort *values [[buffer(0)]],
    constant PixelCountParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= p.count) return;
  const float x = pixel_bf16_to_float(values[index]);
  const float sigmoid = 1.0f / (1.0f + exp(-x));
  values[index] = pixel_float_to_bf16(x * sigmoid);
}

kernel void pixelumm_product(
    device const ushort *left [[buffer(0)]],
    device ushort *rightAndOutput [[buffer(1)]],
    constant PixelCountParams &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= p.count) return;
  rightAndOutput[index] = pixel_float_to_bf16(pixel_bf16_to_float(left[index]) *
                                             pixel_bf16_to_float(rightAndOutput[index]));
}
)metal";

static id<MTLComputePipelineState> pixel_pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *name,
    char *errorText, size_t errorCapacity) {
  NSError *error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name];
  id<MTLComputePipelineState> pipeline = function
      ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
  if (!pipeline) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"unable to prepare Metal kernel %@: %@", name,
         error.localizedDescription ?: @"function missing"]);
    return nil;
  }
  return pipeline;
}

static bool pixel_buffer(id<MTLBuffer> buffer, id<MTLDevice> device,
                         uint64_t bytes) {
  return buffer && buffer.device == device && buffer.storageMode == MTLStorageModeShared &&
         buffer.contents && buffer.length == bytes;
}

static bool pixel_capture_enabled(PixelUMMDecoderPlan *plan, PixelCapture stage) {
  return (plan.captureMask & (uint64_t(1) << (NSUInteger)stage)) != 0;
}

static bool pixel_validate_table(id<MTLBuffer> table, uint32_t rows,
                                 uint32_t width, uint32_t tokens) {
  if (!table || !table.contents) return false;
  const int32_t *values = (const int32_t *)table.contents;
  const uint64_t stride = uint64_t(width) + 1u;
  for (uint32_t row = 0; row < rows; ++row) {
    const uint64_t base = uint64_t(row) * stride;
    const int32_t count = values[base];
    if (count <= 0 || uint32_t(count) > width) return false;
    for (int32_t i = 0; i < count; ++i)
      if (values[base + 1u + uint32_t(i)] < 0 ||
          uint32_t(values[base + 1u + uint32_t(i)]) >= tokens)
        return false;
  }
  return true;
}

static PixelUMMGraphStep *pixel_append_linear(
    PixelUMMDecoderPlan *plan, id<MTLBuffer> input, id<MTLBuffer> weight,
    id<MTLBuffer> output, uint32_t tokens, uint32_t inputWidth,
    uint32_t outputWidth, NSString *name, char *errorText, size_t errorCapacity,
    bool castOperandsToF32 = false) {
  PixelUMMGraphStep *step = pixel_linear_step(plan.device, input, weight, output,
      tokens, inputWidth, outputWidth, name, errorText, errorCapacity,
      castOperandsToF32);
  return step;
}

static bool pixel_encode_graph(MPSCommandBuffer *commandBuffer,
                               PixelUMMGraphStep *step,
                               char *errorText, size_t errorCapacity) {
#if defined(COGNI_PIXELUMM_GRAPH_RESOURCE_PROBE)
  return pixelumm_connected_resource_scope(commandBuffer, step, [&]() -> bool {
#endif
  if (!commandBuffer || !step || !step.executable || !step.execution ||
      !step.inputs || !step.outputs) {
    pixel_set_error(errorText, errorCapacity, "incomplete prepared PixelUMM graph step");
    return false;
  }
  NSArray<MPSGraphTensorData *> *result;
  {
    PixelUMMGraphAPIScope scope(commandBuffer);
    result = [step.executable encodeToCommandBuffer:commandBuffer
        inputsArray:step.inputs resultsArray:step.outputs
        executionDescriptor:step.execution];
  }
  if (!result || result.count != step.outputs.count) {
    pixel_set_ns_error(errorText, errorCapacity,
        [NSString stringWithFormat:@"failed encoding PixelUMM graph %@", step.name]);
    return false;
  }
  return true;
#if defined(COGNI_PIXELUMM_GRAPH_RESOURCE_PROBE)
  });
#endif
}

static id<MTLCommandBuffer> pixel_current_root(MPSCommandBuffer *commandBuffer,
                                              id<MTLDevice> device,
                                              char *errorText,
                                              size_t errorCapacity) {
  id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
  if (!root || root.commandQueue.device != device ||
      root.status >= MTLCommandBufferStatusCommitted) {
    pixel_set_error(errorText, errorCapacity,
                    "PixelUMM current MPSCommandChain root is unavailable");
    return nil;
  }
  return root;
}

static bool pixel_encode_kernel(MPSCommandBuffer *commandBuffer,
                                id<MTLDevice> device,
                                id<MTLComputePipelineState> pipeline,
                                NSArray<id<MTLBuffer>> *buffers,
                                const void *params, NSUInteger paramBytes,
                                uint64_t threadCount, bool threadgroups128,
                                char *errorText, size_t errorCapacity) {
  id<MTLCommandBuffer> root = pixel_current_root(commandBuffer, device,
                                                  errorText, errorCapacity);
  if (!root || !pipeline || !buffers || threadCount == 0) return false;
  id<MTLComputeCommandEncoder> encoder = [root computeCommandEncoder];
  if (!encoder) {
    pixel_set_error(errorText, errorCapacity, "unable to open PixelUMM Metal encoder");
    return false;
  }
  [encoder setComputePipelineState:pipeline];
  for (NSUInteger index = 0; index < buffers.count; ++index)
    [encoder setBuffer:buffers[index] offset:0 atIndex:index];
  [encoder setBytes:params length:paramBytes atIndex:buffers.count];
  if (threadgroups128) {
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)threadCount, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  } else {
    NSUInteger width = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)threadCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }
  [encoder endEncoding];
  return true;
}

static bool pixel_capture(MPSCommandBuffer *commandBuffer,
                          PixelUMMDecoderPlan *plan, PixelCapture stage,
                          id<MTLBuffer> source, uint64_t bytes,
                          char *errorText, size_t errorCapacity) {
  if (!pixel_capture_enabled(plan, stage)) return true;
  id<MTLBuffer> destination = plan.captures[(NSUInteger)stage];
  if (source && destination && source == destination && stage == PixelOutput)
    return true;
  if (!source || !destination || source == destination || destination.length != bytes) {
    pixel_set_error(errorText, errorCapacity, "invalid PixelUMM capture destination");
    return false;
  }
  id<MTLCommandBuffer> root = pixel_current_root(commandBuffer, plan.device,
                                                  errorText, errorCapacity);
  id<MTLBlitCommandEncoder> blit = root ? [root blitCommandEncoder] : nil;
  if (!blit) {
    pixel_set_error(errorText, errorCapacity, "unable to open PixelUMM capture blit encoder");
    return false;
  }
  [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:bytes];
  [blit endEncoding];
  return true;
}

static bool pixel_run_qk_prepare(MPSCommandBuffer *commandBuffer,
                                 PixelUMMDecoderPlan *plan, id<MTLBuffer> values,
                                 uint32_t heads, id<MTLBuffer> gainT,
                                 id<MTLBuffer> gainHW, id<MTLBuffer> normCapture,
                                 id<MTLBuffer> ropeCapture,
                                 id<MTLBuffer> inverseRoots,
                                 char *errorText, size_t errorCapacity) {
  PixelQKParams params{plan.tokens, heads,
      normCapture ? 1u : 0u, ropeCapture ? 1u : 0u};
  id<MTLCommandBuffer> root = pixel_current_root(commandBuffer, plan.device,
                                                  errorText, errorCapacity);
  if (!root || !plan.qkPrepare) return false;
  id<MTLComputeCommandEncoder> encoder = [root computeCommandEncoder];
  if (!encoder) return false;
  [encoder setComputePipelineState:plan.qkPrepare];
  [encoder setBuffer:values offset:0 atIndex:0];
  [encoder setBuffer:gainT offset:0 atIndex:1];
  [encoder setBuffer:gainHW offset:0 atIndex:2];
  [encoder setBuffer:plan.resident[1] offset:0 atIndex:3];
  [encoder setBuffer:plan.resident[2] offset:0 atIndex:4];
  [encoder setBuffer:(normCapture ?: plan.resident[1]) offset:0 atIndex:5];
  [encoder setBuffer:(ropeCapture ?: plan.resident[1]) offset:0 atIndex:6];
  [encoder setBuffer:inverseRoots offset:0 atIndex:7];
  [encoder setBytes:&params length:sizeof(params) atIndex:8];
  [encoder dispatchThreadgroups:MTLSizeMake(uint64_t(plan.tokens) * heads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder endEncoding];
  return true;
}

static bool pixel_copy_capture(MPSCommandBuffer *commandBuffer,
                               PixelUMMDecoderPlan *plan, PixelCapture stage,
                               id<MTLBuffer> source, uint64_t bytes,
                               char *errorText, size_t errorCapacity) {
  return pixel_capture(commandBuffer, plan, stage, source, bytes,
                       errorText, errorCapacity);
}

static void *pixelumm_decoder_block_plan_create(
    void *deviceHandle, int32_t temporal, int32_t height, int32_t width,
    int32_t tile, int32_t dim, int32_t ffn, int32_t heads,
    int32_t kvHeads, int32_t headDim, float epsilon, void *inputHandle,
    void *cosineHandle, void *sineHandle, void **weightHandles,
    void **coreHandles, void **captureHandles, uint64_t captureMask,
    void **scratchHandles, int32_t maxQuery, int32_t maxKey,
    int32_t tileCount, int32_t ffnF32, int32_t prepareSharedGateUp,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (ffnF32 != 0 && ffnF32 != 1) {
    pixel_set_error(errorText, errorCapacity,
                    "PixelUMM FFN F32 mode must be 0 or 1");
    return nullptr;
  }
  if (prepareSharedGateUp != 0 && prepareSharedGateUp != 1) {
    pixel_set_error(errorText, errorCapacity,
                    "PixelUMM shared gate/up preparation flag must be 0 or 1");
    return nullptr;
  }
  if (prepareSharedGateUp && !ffnF32) {
    pixel_set_error(errorText, errorCapacity,
                    "PixelUMM shared gate/up graph requires F32 FFN mode");
    return nullptr;
  }
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      @try {
        id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)inputHandle;
        id<MTLBuffer> cosine = (__bridge id<MTLBuffer>)cosineHandle;
        id<MTLBuffer> sine = (__bridge id<MTLBuffer>)sineHandle;
        if (!device || !input || !cosine || !sine || !weightHandles ||
            !coreHandles || !captureHandles || !scratchHandles ||
            temporal <= 0 || height <= 0 || width <= 0 || tile != 8 ||
            dim != 4096 || ffn != 12288 || heads != 32 || kvHeads != 8 ||
            headDim != 128 || epsilon != 1.0e-6f || maxQuery <= 0 ||
            maxKey <= 0 || tileCount <= 0 || captureMask >> kPixelCaptureCount) {
          pixel_set_error(errorText, errorCapacity,
                          "unsupported or incomplete PixelUMM plan configuration");
          return nullptr;
        }
        uint64_t tokens64 = uint64_t(temporal) * uint64_t(height) * uint64_t(width);
        if (tokens64 == 0 || tokens64 > INT32_MAX / uint64_t(ffn)) {
          pixel_set_error(errorText, errorCapacity, "PixelUMM token extent overflow");
          return nullptr;
        }
        uint32_t tokens = (uint32_t)tokens64;
        if (tileCount == 1 && tokens > 2048u) {
          pixel_set_error(errorText, errorCapacity,
                          "PixelUMM full-attention fallback exceeds 2048 tokens");
          return nullptr;
        }
        if (uint32_t(maxQuery) > tokens || uint32_t(maxKey) > tokens ||
            uint64_t(heads) * uint64_t(maxQuery) * uint64_t(maxKey) > INT32_MAX) {
          pixel_set_error(errorText, errorCapacity,
                          "PixelUMM exact attention extent exceeds prepared MPS limits");
          return nullptr;
        }
        uint64_t dBytes = tokens64 * uint64_t(dim) * 2u;
        uint64_t kvBytes = tokens64 * uint64_t(kvHeads) * uint64_t(headDim) * 2u;
        uint64_t ffBytes = tokens64 * uint64_t(ffn) * 2u;
        uint64_t cosBytes = tokens64 * uint64_t(headDim) * 4u;
        if (!pixel_buffer(input, device, dBytes) || !pixel_buffer(cosine, device, cosBytes) ||
            !pixel_buffer(sine, device, cosBytes)) {
          pixel_set_error(errorText, errorCapacity, "PixelUMM input/table buffers have wrong extent or storage");
          return nullptr;
        }
        const uint64_t weightBytes[kPixelWeightCount] = {
          uint64_t(dim) * 2u, uint64_t(dim) * uint64_t(dim) * 2u,
          uint64_t(kvHeads) * headDim * uint64_t(dim) * 2u,
          uint64_t(kvHeads) * headDim * uint64_t(dim) * 2u,
          uint64_t(dim) * uint64_t(dim) * 2u, 128u, 128u, 128u, 128u,
          uint64_t(dim) * 2u, uint64_t(ffn) * uint64_t(dim) * 2u,
          uint64_t(ffn) * uint64_t(dim) * 2u,
          uint64_t(dim) * uint64_t(ffn) * 2u};
        NSMutableArray<id<MTLBuffer>> *weights = [NSMutableArray arrayWithCapacity:kPixelWeightCount];
        for (NSUInteger i = 0; i < kPixelWeightCount; ++i) {
          id<MTLBuffer> buffer = weightHandles[i]
              ? (__bridge id<MTLBuffer>)weightHandles[i] : nil;
          if (!pixel_buffer(buffer, device, weightBytes[i])) {
            pixel_set_error(errorText, errorCapacity, "PixelUMM weight buffer has wrong extent or storage");
            return nullptr;
          }
          [weights addObject:buffer];
        }
        const uint64_t coreBytes[kPixelCoreCount] = {
          dBytes, dBytes, kvBytes, kvBytes, dBytes, dBytes,
          ffBytes, ffBytes, dBytes};
        NSMutableArray<id<MTLBuffer>> *core = [NSMutableArray arrayWithCapacity:kPixelCoreCount];
        for (NSUInteger i = 0; i < kPixelCoreCount; ++i) {
          id<MTLBuffer> buffer = coreHandles[i]
              ? (__bridge id<MTLBuffer>)coreHandles[i] : nil;
          if (!pixel_buffer(buffer, device, coreBytes[i])) {
            pixel_set_error(errorText, errorCapacity, "PixelUMM workspace buffer has wrong extent or storage");
            return nullptr;
          }
          [core addObject:buffer];
        }
        const uint64_t qBytes = tokens64 * uint64_t(heads) * uint64_t(headDim) * 2u;
        const uint64_t kBytes = kvBytes;
        const uint64_t capBytes[kPixelCaptureCount] = {
          dBytes, qBytes, kBytes, kBytes, qBytes * 2u, kBytes * 2u,
          qBytes * 2u, kBytes * 2u, dBytes, dBytes, dBytes, dBytes,
          ffBytes, ffBytes, ffBytes, ffBytes, dBytes, dBytes,
          tokens64 * 4u, dBytes, dBytes, kBytes,
          tokens64 * uint64_t(heads) * 2u * 4u,
          tokens64 * uint64_t(kvHeads) * 2u * 4u,
          tokens64 * uint64_t(heads) * 2u * 4u,
          tokens64 * uint64_t(kvHeads) * 2u * 4u};
        NSMutableArray *captures = [NSMutableArray arrayWithCapacity:kPixelCaptureCount];
        for (NSUInteger i = 0; i < kPixelCaptureCount; ++i) {
          id<MTLBuffer> buffer = captureHandles[i]
              ? (__bridge id<MTLBuffer>)captureHandles[i] : nil;
          bool enabled = (captureMask & (uint64_t(1) << i)) != 0;
          if ((enabled && (!buffer || buffer.device != device || buffer.storageMode != MTLStorageModeShared ||
                           buffer.length != capBytes[i] || !buffer.contents)) ||
              (!enabled && buffer)) {
            if (i == PixelOutput && enabled && buffer == core[8]) {
              [captures addObject:buffer];
              continue;
            }
            pixel_set_error(errorText, errorCapacity, "PixelUMM capture handle/mask mismatch");
            return nullptr;
          }
          [captures addObject:buffer ?: (id)[NSNull null]];
        }
        // Output capture is the core output buffer, never an extra allocation.
        if ((captureMask & (uint64_t(1) << PixelOutput)) && captures[PixelOutput] != core[8]) {
          pixel_set_error(errorText, errorCapacity, "PixelUMM output capture must alias final workspace");
          return nullptr;
        }
        NSMutableArray<id<MTLBuffer>> *scratch = [NSMutableArray arrayWithCapacity:kPixelScratchCount];
        for (NSUInteger i = 0; i < kPixelScratchCount; ++i) {
          if (!scratchHandles[i]) {
            pixel_set_error(errorText, errorCapacity, "PixelUMM attention/RMS scratch is missing");
            return nullptr;
          }
          [scratch addObject:(__bridge id<MTLBuffer>)scratchHandles[i]];
        }
        const uint64_t scratchBytes[kPixelScratchCount] = {
          uint64_t(heads) * maxQuery * headDim * 4u,
          uint64_t(heads) * maxKey * headDim * 4u,
          uint64_t(heads) * maxKey * headDim * 4u,
          uint64_t(heads) * maxQuery * maxKey * 4u,
          uint64_t(heads) * maxQuery * maxKey * 4u,
          uint64_t(heads) * maxQuery * headDim * 4u,
          uint64_t(tileCount) * (uint64_t(maxQuery) + 1u) * 4u,
          uint64_t(tileCount) * (uint64_t(maxKey) + 1u) * 4u,
          4u, tokens64 * 4u,
          tokens64 * uint64_t(heads) * 2u * 4u,
          tokens64 * uint64_t(kvHeads) * 2u * 4u,
          tokens64 * uint64_t(heads) * 2u * 4u,
          tokens64 * uint64_t(kvHeads) * 2u * 4u};
        for (NSUInteger i = 0; i < kPixelScratchCount; ++i)
          if (!pixel_buffer(scratch[i], device, scratchBytes[i])) {
            pixel_set_error(errorText, errorCapacity, "PixelUMM scratch extent/storage mismatch");
            return nullptr;
          }
        if (!pixel_validate_table(scratch[6], tileCount, maxQuery, tokens) ||
            !pixel_validate_table(scratch[7], tileCount, maxKey, tokens)) {
          pixel_set_error(errorText, errorCapacity, "PixelUMM tile table has an invalid count/index");
          return nullptr;
        }
        std::vector<id<MTLBuffer>> allBorrowed;
        allBorrowed.push_back(input); allBorrowed.push_back(cosine); allBorrowed.push_back(sine);
        for (id<MTLBuffer> buffer in weights) allBorrowed.push_back(buffer);
        for (size_t i = 0; i < allBorrowed.size(); ++i)
          for (size_t j = i + 1; j < allBorrowed.size(); ++j)
            if (!pixel_disjoint(allBorrowed[i], allBorrowed[j])) {
              pixel_set_error(errorText, errorCapacity, "borrowed PixelUMM inputs alias");
              return nullptr;
            }
        std::vector<id<MTLBuffer>> writable;
        for (id<MTLBuffer> buffer in core) writable.push_back(buffer);
        for (NSUInteger i = 0; i < kPixelScratchCount; ++i) writable.push_back(scratch[i]);
        for (NSUInteger i = 0; i < kPixelCaptureCount; ++i) {
          if (!(captureMask & (uint64_t(1) << i)) || i == PixelOutput) continue;
          writable.push_back((id<MTLBuffer>)captures[i]);
        }
        for (id<MTLBuffer> output : writable)
          for (id<MTLBuffer> borrowed : allBorrowed)
            if (!pixel_disjoint(output, borrowed)) {
              pixel_set_error(errorText, errorCapacity, "PixelUMM writable buffers alias borrowed inputs");
              return nullptr;
            }
        for (size_t i = 0; i < writable.size(); ++i)
          for (size_t j = i + 1; j < writable.size(); ++j)
            if (!pixel_disjoint(writable[i], writable[j])) {
              pixel_set_error(errorText, errorCapacity, "PixelUMM owned workspaces/captures alias");
              return nullptr;
            }

        PixelUMMDecoderPlan *plan = [PixelUMMDecoderPlan new];
        plan.device = device;
        plan.weights = [weights copy];
        plan.core = [core copy];
        plan.captures = [captures copy];
        plan.scratch = [scratch copy];
        plan.temporal = (uint32_t)temporal; plan.height = (uint32_t)height;
        plan.width = (uint32_t)width; plan.tokens = tokens; plan.tile = (uint32_t)tile;
        plan.dim = (uint32_t)dim; plan.ffn = (uint32_t)ffn;
        plan.heads = (uint32_t)heads; plan.kvHeads = (uint32_t)kvHeads;
        plan.headDim = (uint32_t)headDim; plan.maxQuery = (uint32_t)maxQuery;
        plan.maxKey = (uint32_t)maxKey; plan.tileCount = (uint32_t)tileCount;
        plan.ffnF32 = ffnF32 != 0;
        plan.sharedGateUpPrepared = prepareSharedGateUp != 0;
        plan.useSharedGateUp = NO;
        plan.captureMask = captureMask;
        NSMutableArray *allResident = [NSMutableArray array];
        [allResident addObject:input]; [allResident addObject:cosine]; [allResident addObject:sine];
        [allResident addObjectsFromArray:weights]; [allResident addObjectsFromArray:core];
        [allResident addObjectsFromArray:scratch];
        for (NSUInteger i = 0; i < kPixelCaptureCount; ++i)
          if ((captureMask & (uint64_t(1) << i)) && i != PixelOutput)
            [allResident addObject:captures[i]];
        plan.resident = [allResident copy];

        NSError *metalError = nil;
        NSString *source = [NSString stringWithUTF8String:pixel_metal_source];
        MTLCompileOptions *metalOptions = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
          metalOptions.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
          metalOptions.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        plan.metalLibrary = [device newLibraryWithSource:source options:metalOptions
                                                   error:&metalError];
        if (!plan.metalLibrary) {
          pixel_set_ns_error(errorText, errorCapacity,
              [NSString stringWithFormat:@"unable to compile PixelUMM Metal source: %@", metalError.localizedDescription]);
          return nullptr;
        }
#define PIXEL_PIPELINE(property, function) \
        plan.property = pixel_pipeline(device, plan.metalLibrary, @function, errorText, errorCapacity); \
        if (!plan.property) return nullptr
        PIXEL_PIPELINE(qkPrepare, "pixelumm_qk_prepare");
        PIXEL_PIPELINE(gatherQ, "pixelumm_gather_q");
        PIXEL_PIPELINE(gatherKV, "pixelumm_gather_kv");
        PIXEL_PIPELINE(scaleScores, "pixelumm_scale_scores");
        PIXEL_PIPELINE(scatterAttention, "pixelumm_scatter_attention");
        PIXEL_PIPELINE(residual, "pixelumm_residual");
        PIXEL_PIPELINE(silu, "pixelumm_silu");
        PIXEL_PIPELINE(product, "pixelumm_product");
#undef PIXEL_PIPELINE
        NSMutableArray<PixelUMMGraphStep *> *steps = [NSMutableArray array];
        NSMutableArray<PixelUMMAttentionGeometry *> *attentionGeometries =
            [NSMutableArray array];
        NSMutableArray<NSNumber *> *tileGeometryIndices =
            [NSMutableArray arrayWithCapacity:(NSUInteger)tileCount];
        const int32_t *queryTable = (const int32_t *)scratch[6].contents;
        const int32_t *keyTable = (const int32_t *)scratch[7].contents;
        for (uint32_t tileIndex = 0; tileIndex < uint32_t(tileCount); ++tileIndex) {
          const uint64_t queryBase = uint64_t(tileIndex) * (uint64_t(maxQuery) + 1u);
          const uint64_t keyBase = uint64_t(tileIndex) * (uint64_t(maxKey) + 1u);
          uint32_t queryCount = (uint32_t)queryTable[queryBase];
          uint32_t keyCount = (uint32_t)keyTable[keyBase];
          NSUInteger geometryIndex = NSNotFound;
          for (NSUInteger index = 0; index < attentionGeometries.count; ++index) {
            PixelUMMAttentionGeometry *candidate = attentionGeometries[index];
            if (candidate.queryCount == queryCount && candidate.keyCount == keyCount) {
              geometryIndex = index;
              break;
            }
          }
          if (geometryIndex == NSNotFound) {
            PixelUMMAttentionGeometry *geometry = [PixelUMMAttentionGeometry new];
            geometry.queryCount = queryCount;
            geometry.keyCount = keyCount;
            NSString *qkName = [NSString stringWithFormat:@"attention_qk_q%u_k%u",
                queryCount, keyCount];
            NSString *softmaxName = [NSString stringWithFormat:@"attention_softmax_q%u_k%u",
                queryCount, keyCount];
            NSString *pvName = [NSString stringWithFormat:@"attention_pv_q%u_k%u",
                queryCount, keyCount];
            geometry.qk = pixel_attention_qk_step(device, scratch[0], scratch[1],
                scratch[3], queryCount, keyCount, qkName,
                errorText, errorCapacity);
            geometry.softmax = pixel_attention_softmax_step(device, scratch[3],
                scratch[4], queryCount, keyCount, softmaxName,
                errorText, errorCapacity);
            geometry.pv = pixel_attention_pv_step(device, scratch[4], scratch[2],
                scratch[5], queryCount, keyCount, pvName,
                errorText, errorCapacity);
            if (!geometry.qk || !geometry.softmax || !geometry.pv) return nullptr;
            [steps addObjectsFromArray:@[geometry.qk, geometry.softmax, geometry.pv]];
            [attentionGeometries addObject:geometry];
            geometryIndex = attentionGeometries.count - 1;
          }
          [tileGeometryIndices addObject:@(geometryIndex)];
        }
        plan.attentionGeometries = [attentionGeometries copy];
        plan.tileGeometryIndices = [tileGeometryIndices copy];
        // core[1] is dead until Q projection, so it temporarily materializes
        // normalized BF16 before the separately encoded learned-gain graph.
        // scratch[9] stores residual RMS statistics; scratch[10:11] store
        // prepared Q/K split-RMS statistics and scratch[12:13] their inverse
        // roots. scratch[6:7] are immutable query/key index tables.
        plan.inputNorm = pixel_rms_prepare_step(device, input, scratch[8], core[1],
            scratch[9], tokens,
            (uint32_t)dim, epsilon, @"input_norm", errorText, errorCapacity);
        if (!plan.inputNorm) return nullptr;
        [steps addObject:plan.inputNorm];
        plan.inputNormGain = pixel_rms_gain_step(device, core[1], weights[0], core[0],
            tokens, (uint32_t)dim, @"input_norm_gain", errorText, errorCapacity);
        if (!plan.inputNormGain) return nullptr;
        [steps addObject:plan.inputNormGain];
        plan.qProjection = pixel_append_linear(plan, core[0], weights[1], core[1],
            tokens, (uint32_t)dim, (uint32_t)dim, @"q_projection", errorText, errorCapacity);
        plan.kProjection = pixel_append_linear(plan, core[0], weights[2], core[2],
            tokens, (uint32_t)dim, (uint32_t)(kvHeads * headDim), @"k_projection", errorText, errorCapacity);
        plan.vProjection = pixel_append_linear(plan, core[0], weights[3], core[3],
            tokens, (uint32_t)dim, (uint32_t)(kvHeads * headDim), @"v_projection", errorText, errorCapacity);
        plan.qNormStatistics = pixel_qk_statistics_step(device, core[1], scratch[8],
            scratch[10], tokens, (uint32_t)heads, epsilon, @"q_norm_statistics",
            errorText, errorCapacity);
        plan.kNormStatistics = pixel_qk_statistics_step(device, core[2], scratch[8],
            scratch[11], tokens, (uint32_t)kvHeads, epsilon, @"k_norm_statistics",
            errorText, errorCapacity);
        plan.qNormInverse = pixel_qk_inverse_step(device, scratch[10], scratch[12],
            tokens, (uint32_t)heads, @"q_norm_inverse", errorText, errorCapacity);
        plan.kNormInverse = pixel_qk_inverse_step(device, scratch[11], scratch[13],
            tokens, (uint32_t)kvHeads, @"k_norm_inverse", errorText, errorCapacity);
        if (!plan.qProjection || !plan.kProjection || !plan.vProjection ||
            !plan.qNormStatistics || !plan.kNormStatistics ||
            !plan.qNormInverse || !plan.kNormInverse) return nullptr;
        plan.oProjection = pixel_append_linear(plan, core[4], weights[4], core[1],
            tokens, (uint32_t)dim, (uint32_t)dim, @"o_projection", errorText, errorCapacity);
        // core[0] is no longer needed after Q/K/V projection and can hold the
        // post-attention normalized BF16 handoff without another workspace.
        plan.postNorm = pixel_rms_prepare_step(device, core[5], scratch[8], core[0],
            scratch[9], tokens, (uint32_t)dim, epsilon, @"post_attn_norm",
            errorText, errorCapacity);
        plan.postNormGain = pixel_rms_gain_step(device, core[0], weights[9], core[1],
            tokens, (uint32_t)dim, @"post_attn_norm_gain", errorText, errorCapacity);
        plan.gateProjection = pixel_append_linear(plan, core[1], weights[10], core[6],
            tokens, (uint32_t)dim, (uint32_t)ffn, @"gate_projection", errorText, errorCapacity,
            plan.ffnF32);
        plan.upProjection = pixel_append_linear(plan, core[1], weights[11], core[7],
            tokens, (uint32_t)dim, (uint32_t)ffn, @"up_projection", errorText, errorCapacity,
            plan.ffnF32);
        // Keep the shared-input experiment opt-in: the baseline plan retains
        // only its existing independent gate/up executables and preparation cost.
        if (plan.sharedGateUpPrepared) {
          plan.sharedGateUpProjection = pixel_linear_shared_input_pair_step(
              device, core[1], weights[10], weights[11], core[6], core[7],
              tokens, (uint32_t)dim, (uint32_t)ffn, @"shared_gate_up_projection",
              errorText, errorCapacity);
        }
        plan.downProjection = pixel_append_linear(plan, core[7], weights[12], core[4],
            tokens, (uint32_t)ffn, (uint32_t)dim, @"down_projection", errorText, errorCapacity,
            plan.ffnF32);
        if (!plan.oProjection || !plan.postNorm || !plan.postNormGain || !plan.gateProjection ||
            !plan.upProjection || !plan.downProjection) return nullptr;
        if (plan.sharedGateUpPrepared && !plan.sharedGateUpProjection) return nullptr;
        [steps addObjectsFromArray:@[plan.qNormStatistics, plan.kNormStatistics,
            plan.qNormInverse, plan.kNormInverse,
            plan.oProjection, plan.postNorm, plan.postNormGain,
            plan.gateProjection, plan.upProjection, plan.downProjection]];
        if (plan.sharedGateUpProjection)
          [steps addObject:plan.sharedGateUpProjection];
        plan.steps = [steps copy];
        if (!plan.steps || plan.resident.count == 0) {
          pixel_set_error(errorText, errorCapacity, "unable to retain PixelUMM prepared plan");
          return nullptr;
        }
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"PixelUMM preparation exception %@: %@",
             exception.name, exception.reason]);
        return nullptr;
      }
    }
  }
  pixel_set_error(errorText, errorCapacity, "PixelUMM MPS path requires macOS 14 or newer");
  return nullptr;
}

// Keep the original ABI and its BF16-only behavior for existing callers.
extern "C" void *gs_pixelumm_decoder_block_plan_create(
    void *deviceHandle, int32_t temporal, int32_t height, int32_t width,
    int32_t tile, int32_t dim, int32_t ffn, int32_t heads,
    int32_t kvHeads, int32_t headDim, float epsilon, void *inputHandle,
    void *cosineHandle, void *sineHandle, void **weightHandles,
    void **coreHandles, void **captureHandles, uint64_t captureMask,
    void **scratchHandles, int32_t maxQuery, int32_t maxKey,
    int32_t tileCount, char *errorText, size_t errorCapacity) {
  return pixelumm_decoder_block_plan_create(deviceHandle, temporal, height,
      width, tile, dim, ffn, heads, kvHeads, headDim, epsilon, inputHandle,
      cosineHandle, sineHandle, weightHandles, coreHandles, captureHandles,
      captureMask, scratchHandles, maxQuery, maxKey, tileCount, 0, 0, errorText,
      errorCapacity);
}

extern "C" void *gs_pixelumm_decoder_block_plan_create_with_ffn_f32(
    void *deviceHandle, int32_t temporal, int32_t height, int32_t width,
    int32_t tile, int32_t dim, int32_t ffn, int32_t heads,
    int32_t kvHeads, int32_t headDim, float epsilon, void *inputHandle,
    void *cosineHandle, void *sineHandle, void **weightHandles,
    void **coreHandles, void **captureHandles, uint64_t captureMask,
    void **scratchHandles, int32_t maxQuery, int32_t maxKey,
    int32_t tileCount, int32_t ffnF32, char *errorText, size_t errorCapacity) {
  return pixelumm_decoder_block_plan_create(deviceHandle, temporal, height,
      width, tile, dim, ffn, heads, kvHeads, headDim, epsilon, inputHandle,
      cosineHandle, sineHandle, weightHandles, coreHandles, captureHandles,
      captureMask, scratchHandles, maxQuery, maxKey, tileCount, ffnF32, 0,
      errorText, errorCapacity);
}

extern "C" void *gs_pixelumm_decoder_block_plan_create_with_ffn_f32_shared_gate_up(
    void *deviceHandle, int32_t temporal, int32_t height, int32_t width,
    int32_t tile, int32_t dim, int32_t ffn, int32_t heads,
    int32_t kvHeads, int32_t headDim, float epsilon, void *inputHandle,
    void *cosineHandle, void *sineHandle, void **weightHandles,
    void **coreHandles, void **captureHandles, uint64_t captureMask,
    void **scratchHandles, int32_t maxQuery, int32_t maxKey,
    int32_t tileCount, int32_t ffnF32, int32_t prepareSharedGateUp,
    char *errorText, size_t errorCapacity) {
  return pixelumm_decoder_block_plan_create(deviceHandle, temporal, height,
      width, tile, dim, ffn, heads, kvHeads, headDim, epsilon, inputHandle,
      cosineHandle, sineHandle, weightHandles, coreHandles, captureHandles,
      captureMask, scratchHandles, maxQuery, maxKey, tileCount, ffnF32,
      prepareSharedGateUp, errorText, errorCapacity);
}

// Mode changes are host-side plan selection only. Callers must select only
// while the session is idle, after prior command-chain work has completed.
extern "C" int32_t gs_pixelumm_decoder_block_plan_set_shared_gate_up(
    void *planHandle, int32_t enabled, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle || (enabled != 0 && enabled != 1)) {
    pixel_set_error(errorText, errorCapacity,
        "PixelUMM shared gate/up selection requires a plan and a 0/1 value");
    return -1;
  }
  PixelUMMDecoderPlan *plan = (__bridge PixelUMMDecoderPlan *)planHandle;
  @synchronized(plan) {
    if (enabled && (!plan.ffnF32 || !plan.sharedGateUpPrepared ||
                    !plan.sharedGateUpProjection)) {
      pixel_set_error(errorText, errorCapacity,
          "PixelUMM shared gate/up graph was not prepared for this F32 plan");
      return -2;
    }
    plan.useSharedGateUp = enabled != 0;
  }
  return 0;
}

static int32_t pixel_encode_block(
    void *planHandle, void *commandChainHandle, char *errorText,
    size_t errorCapacity, void *profileChain,
    PixelUMMPhaseMarker *markers, uint32_t markerCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle || !commandChainHandle ||
      (markers && (!profileChain || markerCapacity != kPixelPhaseMarkerCount))) {
    pixel_set_error(errorText, errorCapacity, "required PixelUMM plan/chain handle is null");
    return -1;
  }
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      PixelUMMDecoderPlan *plan = (__bridge PixelUMMDecoderPlan *)planHandle;
      BOOL useSharedGateUp = NO;
      @synchronized(plan) {
        useSharedGateUp = plan.useSharedGateUp;
      }
      MPSCommandBuffer *commandBuffer = (__bridge MPSCommandBuffer *)commandChainHandle;
      id<MTLCommandBuffer> root = commandBuffer.rootCommandBuffer;
      if (!plan.device || !plan.steps || !plan.resident || !root ||
          root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted) {
        pixel_set_error(errorText, errorCapacity,
                        "PixelUMM MPSCommandChain root is unavailable, committed or on another device");
        return -2;
      }
      PixelUMMPhaseRecorder profile(profileChain, markers, markerCapacity);
      @try {
        id<MTLBuffer> input = plan.resident[0];
        const uint64_t dBytes = uint64_t(plan.tokens) * plan.dim * 2u;
        const uint64_t kvBytes = uint64_t(plan.tokens) * plan.kvHeads * plan.headDim * 2u;
        const uint64_t ffBytes = uint64_t(plan.tokens) * plan.ffn * 2u;
        const uint64_t qStatisticsBytes = uint64_t(plan.tokens) * plan.heads * 2u * 4u;
        const uint64_t kStatisticsBytes = uint64_t(plan.tokens) * plan.kvHeads * 2u * 4u;
        if (!profile.mark(0, GS_MPS_API_PROFILE_PHASE_SELF_PREPARE)) return -18;
        if (!pixel_encode_graph(commandBuffer, plan.inputNorm, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelInputNormStatistics,
                plan.scratch[9], uint64_t(plan.tokens) * 4u, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelInputNormPreGain,
                plan.core[1], dBytes, errorText, errorCapacity) ||
            !pixel_encode_graph(commandBuffer, plan.inputNormGain, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelInputNorm, plan.core[0], dBytes, errorText, errorCapacity)) return -3;
        if (!profile.mark(1, GS_MPS_API_PROFILE_PHASE_SELF_PREPARE)) return -18;
        if (!pixel_encode_graph(commandBuffer, plan.qProjection, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelQProj, plan.core[1], dBytes, errorText, errorCapacity)) return -4;
        if (!pixel_encode_graph(commandBuffer, plan.kProjection, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelKProj, plan.core[2], kvBytes, errorText, errorCapacity)) return -5;
        if (!pixel_encode_graph(commandBuffer, plan.vProjection, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelVProj, plan.core[3], kvBytes, errorText, errorCapacity)) return -6;
        if (!profile.mark(2, GS_MPS_API_PROFILE_PHASE_SELF_PREPARE)) return -18;
        if (!pixel_encode_graph(commandBuffer, plan.qNormStatistics, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelQNormStatistics,
                plan.scratch[10], qStatisticsBytes, errorText, errorCapacity) ||
            !pixel_encode_graph(commandBuffer, plan.kNormStatistics, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelKNormStatistics,
                plan.scratch[11], kStatisticsBytes, errorText, errorCapacity)) return -7;
        if (!pixel_encode_graph(commandBuffer, plan.qNormInverse, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelQNormInverse,
                plan.scratch[12], qStatisticsBytes, errorText, errorCapacity) ||
            !pixel_encode_graph(commandBuffer, plan.kNormInverse, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelKNormInverse,
                plan.scratch[13], kStatisticsBytes, errorText, errorCapacity)) return -7;
        id<MTLBuffer> qNorm = pixel_capture_enabled(plan, PixelQNorm) ? plan.captures[PixelQNorm] : nil;
        id<MTLBuffer> qRope = pixel_capture_enabled(plan, PixelQRope) ? plan.captures[PixelQRope] : nil;
        id<MTLBuffer> kNorm = pixel_capture_enabled(plan, PixelKNorm) ? plan.captures[PixelKNorm] : nil;
        id<MTLBuffer> kRope = pixel_capture_enabled(plan, PixelKRope) ? plan.captures[PixelKRope] : nil;
        if (!pixel_run_qk_prepare(commandBuffer, plan, plan.core[1], plan.heads,
              plan.weights[5], plan.weights[6], qNorm, qRope,
              plan.scratch[12], errorText, errorCapacity) ||
            !pixel_run_qk_prepare(commandBuffer, plan, plan.core[2], plan.kvHeads,
              plan.weights[7], plan.weights[8], kNorm, kRope,
              plan.scratch[13], errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelQReady, plan.core[1], dBytes, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelKReady, plan.core[2], kvBytes, errorText, errorCapacity)) return -7;
        if (!profile.mark(3, GS_MPS_API_PROFILE_PHASE_SELF_ATTENTION)) return -18;
        id<MTLBuffer> queryIndices = plan.scratch[6];
        id<MTLBuffer> keyIndices = plan.scratch[7];
        const int32_t *queryTable = (const int32_t *)queryIndices.contents;
        const int32_t *keyTable = (const int32_t *)keyIndices.contents;
        for (uint32_t tileIndex = 0; tileIndex < plan.tileCount; ++tileIndex) {
          const uint64_t queryBase = uint64_t(tileIndex) * (uint64_t(plan.maxQuery) + 1u);
          const uint64_t keyBase = uint64_t(tileIndex) * (uint64_t(plan.maxKey) + 1u);
          uint32_t queryCount = (uint32_t)queryTable[queryBase];
          uint32_t keyCount = (uint32_t)keyTable[keyBase];
          PixelGatherQParams qParams{tileIndex, plan.maxQuery, queryCount};
          PixelGatherKVParams kvParams{tileIndex, plan.maxKey, keyCount};
          if (tileIndex >= plan.tileGeometryIndices.count) {
            pixel_set_error(errorText, errorCapacity, "PixelUMM attention geometry index is missing");
            return -8;
          }
          NSUInteger geometryIndex = plan.tileGeometryIndices[tileIndex].unsignedIntegerValue;
          if (geometryIndex >= plan.attentionGeometries.count) {
            pixel_set_error(errorText, errorCapacity, "PixelUMM attention geometry is outside the prepared plan");
            return -8;
          }
          PixelUMMAttentionGeometry *geometry = plan.attentionGeometries[geometryIndex];
          uint64_t panelCount = uint64_t(plan.heads) * queryCount * plan.headDim;
          uint64_t scoreCount = uint64_t(plan.heads) * queryCount * keyCount;
          uint64_t kvPanelCount = uint64_t(plan.heads) * keyCount * plan.headDim;
          if (geometry.queryCount != queryCount || geometry.keyCount != keyCount ||
              !pixel_encode_kernel(commandBuffer, plan.device, plan.gatherQ,
                  @[plan.core[1], queryIndices, plan.scratch[0]], &qParams,
                  sizeof(qParams), panelCount, false, errorText, errorCapacity) ||
              !pixel_encode_kernel(commandBuffer, plan.device, plan.gatherKV,
                  @[plan.core[2], keyIndices, plan.scratch[1]], &kvParams,
                  sizeof(kvParams), kvPanelCount, false,
                  errorText, errorCapacity) ||
              !pixel_encode_kernel(commandBuffer, plan.device, plan.gatherKV,
                  @[plan.core[3], keyIndices, plan.scratch[2]], &kvParams,
                  sizeof(kvParams), kvPanelCount, false,
                  errorText, errorCapacity) ||
              !pixel_encode_graph(commandBuffer, geometry.qk, errorText, errorCapacity)) return -9;
          PixelScaleParams scaleParams{(uint32_t)scoreCount, 1.0f / sqrtf(128.0f)};
          if (!pixel_encode_kernel(commandBuffer, plan.device, plan.scaleScores,
                @[plan.scratch[3]], &scaleParams, sizeof(scaleParams), scoreCount,
                false, errorText, errorCapacity) ||
              !pixel_encode_graph(commandBuffer, geometry.softmax, errorText, errorCapacity) ||
              !pixel_encode_graph(commandBuffer, geometry.pv, errorText, errorCapacity)) return -10;
          PixelScatterParams scatterParams{tileIndex, plan.maxQuery, queryCount};
          if (!pixel_encode_kernel(commandBuffer, plan.device, plan.scatterAttention,
                @[plan.scratch[5], queryIndices, plan.core[4]],
                &scatterParams, sizeof(scatterParams), panelCount,
                false, errorText, errorCapacity)) return -11;
        }
        if (!pixel_copy_capture(commandBuffer, plan, PixelAttention, plan.core[4], dBytes, errorText, errorCapacity)) return -11;
        if (!profile.mark(4, GS_MPS_API_PROFILE_PHASE_SELF_PROJECTION_RESIDUAL)) return -18;
        if (!pixel_encode_graph(commandBuffer, plan.oProjection, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelOProj, plan.core[1], dBytes, errorText, errorCapacity)) return -11;
        if (!profile.mark(5, GS_MPS_API_PROFILE_PHASE_SELF_PROJECTION_RESIDUAL)) return -18;
        PixelCountParams hiddenCount{plan.tokens * plan.dim};
        if (!pixel_encode_kernel(commandBuffer, plan.device, plan.residual,
              @[input, plan.core[1], plan.core[5]], &hiddenCount,
              sizeof(PixelCountParams), uint64_t(plan.tokens) * plan.dim,
              false, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelAttnResidual, plan.core[5], dBytes, errorText, errorCapacity) ||
            !pixel_encode_graph(commandBuffer, plan.postNorm, errorText, errorCapacity) ||
            !pixel_encode_graph(commandBuffer, plan.postNormGain, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelPostNorm, plan.core[1], dBytes, errorText, errorCapacity)) return -12;
        if (!profile.mark(6, GS_MPS_API_PROFILE_PHASE_FFN)) return -18;
        if (useSharedGateUp) {
          if (!plan.sharedGateUpPrepared || !plan.sharedGateUpProjection ||
              !pixel_encode_graph(commandBuffer, plan.sharedGateUpProjection,
                  errorText, errorCapacity) ||
              !pixel_copy_capture(commandBuffer, plan, PixelGateProj,
                  plan.core[6], ffBytes, errorText, errorCapacity) ||
              !pixel_copy_capture(commandBuffer, plan, PixelUpProj,
                  plan.core[7], ffBytes, errorText, errorCapacity)) return -13;
        } else if (!pixel_encode_graph(commandBuffer, plan.gateProjection, errorText, errorCapacity) ||
                   !pixel_copy_capture(commandBuffer, plan, PixelGateProj, plan.core[6], ffBytes, errorText, errorCapacity) ||
                   !pixel_encode_graph(commandBuffer, plan.upProjection, errorText, errorCapacity) ||
                   !pixel_copy_capture(commandBuffer, plan, PixelUpProj, plan.core[7], ffBytes, errorText, errorCapacity)) return -13;
        if (!profile.mark(7, GS_MPS_API_PROFILE_PHASE_FFN)) return -18;
        PixelCountParams ffnCount{plan.tokens * plan.ffn};
        if (!pixel_encode_kernel(commandBuffer, plan.device, plan.silu,
              @[plan.core[6]], &ffnCount,
              sizeof(PixelCountParams), uint64_t(plan.tokens) * plan.ffn,
              false, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelSiluGate, plan.core[6], ffBytes, errorText, errorCapacity) ||
            !pixel_encode_kernel(commandBuffer, plan.device, plan.product,
              @[plan.core[6], plan.core[7]], &ffnCount,
              sizeof(PixelCountParams), uint64_t(plan.tokens) * plan.ffn,
              false, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelProduct, plan.core[7], ffBytes, errorText, errorCapacity)) return -14;
        if (!profile.mark(8, GS_MPS_API_PROFILE_PHASE_FFN)) return -18;
        if (!pixel_encode_graph(commandBuffer, plan.downProjection, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelDownProj, plan.core[4], dBytes, errorText, errorCapacity)) return -14;
        if (!profile.mark(9, GS_MPS_API_PROFILE_PHASE_FFN)) return -18;
        if (!pixel_encode_kernel(commandBuffer, plan.device, plan.residual,
              @[plan.core[5], plan.core[4], plan.core[8]], &hiddenCount,
              sizeof(PixelCountParams), uint64_t(plan.tokens) * plan.dim,
              false, errorText, errorCapacity) ||
            !pixel_copy_capture(commandBuffer, plan, PixelOutput, plan.core[8], dBytes, errorText, errorCapacity)) return -15;
        if (!profile.mark(10, GS_MPS_API_PROFILE_PHASE_UNASSIGNED)) return -18;
        return 0;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"PixelUMM encode exception %@: %@", exception.name, exception.reason]);
        return -16;
      }
    }
  }
  pixel_set_error(errorText, errorCapacity, "PixelUMM MPS encode requires macOS 14 or newer");
  return -17;
}

extern "C" int32_t gs_pixelumm_decoder_block_encode(
    void *plan, void *commandBuffer, char *error, size_t capacity) {
  return pixel_encode_block(plan, commandBuffer, error, capacity,
                            nullptr, nullptr, 0);
}

extern "C" int32_t gs_pixelumm_decoder_block_encode_profiled(
    void *plan, void *commandBuffer, void *profileChain,
    PixelUMMPhaseMarker *markers, uint32_t markerCapacity,
    char *error, size_t capacity) {
  if (!profileChain || !markers || markerCapacity != kPixelPhaseMarkerCount)
    return -18;
  return pixel_encode_block(plan, commandBuffer, error, capacity,
                            profileChain, markers, markerCapacity);
}

extern "C" void gs_pixelumm_decoder_block_plan_release(void *planHandle) {
  if (!planHandle) return;
  @autoreleasepool {
    PixelUMMDecoderPlan *plan = (__bridge_transfer PixelUMMDecoderPlan *)planHandle;
    (void)plan;
  }
}
