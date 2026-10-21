// Diagnostic-only interop: borrow zero-offset contiguous Torch MPS storage.
// All encoding shares Torch's serial queue/stream; no host GPU wait or copy.
// The production opaque bridge is compiled verbatim into this local probe.
#include "../../src/ml/video/wan_attention_mps_bridge.mm"
#include <ATen/mps/MPSStream.h>
#include <torch/mps.h>
#include <cstdio>

@interface WanF16StagePlan : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *buffers;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputs;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputs;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic) void *opaque;
@end
@implementation WanF16StagePlan
- (void)dealloc {
  if (_opaque) gs_wan_attention_mps_plan_release(_opaque);
}
@end

static constexpr uint64_t Q = 128, K = 36608, H = 40, D = 128;
static constexpr uint64_t QN = Q * H * D, KN = K * H * D, SN = H * Q * K;

static bool stage_buffer(id<MTLBuffer> buffer, uint64_t bytes,
                         id<MTLDevice> device) {
  return buffer && buffer.device == device &&
         (buffer.storageMode == MTLStorageModeShared ||
          buffer.storageMode == MTLStorageModePrivate) && buffer.length >= bytes;
}

// Guard the production boundary: the diagnostic opt-in must not admit private
// buffers through the existing public typed constructor.
extern "C" int32_t wan_stage_legacy_private_rejected(void **inputs, void **outputs) {
  if (!inputs || !outputs) return -1;
  for (int i=0; i<3; ++i) {
    if (!inputs[i] || ((__bridge id<MTLBuffer>)inputs[i]).storageMode != MTLStorageModePrivate)
      return -2;
  }
  if (!outputs[0] || !outputs[1]) return -1;
  auto q = (__bridge id<MTLBuffer>)inputs[0];
  void *plan = gs_wan_attention_mps_plan_create_typed((__bridge void *)q.device,
      1,Q,K,H,D,inputs[0],inputs[1],inputs[2],outputs[0],outputs[1],1);
  if (!plan) return 0;
  gs_wan_attention_mps_plan_release(plan);
  return -3;
}

// Inputs are BTHD Q/K/V for prefixes, BHQK scores for isolated scale/softmax,
// and BHQK P plus BTHD V for isolated PV. Output order is explicit, not the
// compiler's target order. The caller retains the Tensor allocator ownership.
extern "C" void *wan_stage_create(int32_t kind, void **inputHandles,
                                  void **outputHandles) {
  if (!inputHandles || !outputHandles || kind < 0 || kind > 12 || kind == 7)
    return nullptr;
  @autoreleasepool {
    @try {
      const int inputCount = kind < 8 || kind == 12 ? 3 : (kind == 11 ? 2 : 1);
      const int outputCount = kind == 0 || kind == 6 ? 2 : 1;
      NSMutableArray<id<MTLBuffer>> *buffers = [NSMutableArray new];
      NSMutableArray<id<MTLBuffer>> *ins = [NSMutableArray new];
      NSMutableArray<id<MTLBuffer>> *outs = [NSMutableArray new];
      id<MTLDevice> device = nil;
      for (int i = 0; i < inputCount + outputCount; ++i) {
        void *handle = i < inputCount ? inputHandles[i] : outputHandles[i - inputCount];
        if (!handle) return nullptr;
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
        if (!device) device = buffer.device;
        uint64_t bytes;
        if (i < inputCount) {
          bytes = kind >= 8 && kind <= 11 ? (i == 0 ? SN : KN) * 2
                                        : (i == 0 ? QN : KN) * 2;
        } else {
          bytes = (kind == 0 || kind == 5 || kind == 11 || kind == 12 ||
                   (kind == 6 && i == inputCount + 1)) ? QN * 2
                  : SN * ((kind == 2 || kind == 8) ? 4 : 2);
        }
        if (!stage_buffer(buffer, bytes, device)) return nullptr;
        for (id<MTLBuffer> other in buffers)
          if (!wan_attention_mps_disjoint(buffer, other)) return nullptr;
        [buffers addObject:buffer];
        if (i < inputCount) [ins addObject:buffer]; else [outs addObject:buffer];
      }
      WanF16StagePlan *plan = [WanF16StagePlan new];
      plan.buffers = buffers;
      if (kind == 0) {
        plan.opaque = wan_attention_mps_plan_create_impl(
            (__bridge void *)device, 1, Q, K, H, D, inputHandles[0],
            inputHandles[1], inputHandles[2], outputHandles[0], outputHandles[1],
            MPSDataTypeFloat16, 2, false, false, true);
        return plan.opaque ? (__bridge_retained void *)plan : nullptr;
      }
      MPSGraph *g = [MPSGraph new];
      g.options = MPSGraphOptionsNone;
      NSArray *bthq = @[@1, @(Q), @(H), @(D)];
      NSArray *bthk = @[@1, @(K), @(H), @(D)];
      NSArray *bhqk = @[@1, @(H), @(Q), @(K)];
      NSArray *bhqd = @[@1, @(H), @(Q), @(D)];
      NSMutableArray<MPSGraphTensor *> *feeds = [NSMutableArray new];
      NSMutableArray<MPSGraphTensor *> *targets = [NSMutableArray new];
      MPSGraphTensor *qk = nil, *scaled = nil, *rounded = nil, *prob = nil, *ctx = nil;
      if (kind >= 8 && kind <= 11) {
        auto x = [g placeholderWithShape:bhqk dataType:MPSDataTypeFloat16 name:nil];
        [feeds addObject:x];
        if (kind == 8 || kind == 9) {
          auto x32 = [g castTensor:x toType:MPSDataTypeFloat32 name:nil];
          auto scale = [g constantWithScalar:(float)(1.0 / std::sqrt((double)D))
                                      shape:@[@1] dataType:MPSDataTypeFloat32];
          scaled = [g multiplicationWithPrimaryTensor:x32 secondaryTensor:scale name:nil];
          rounded = [g castTensor:scaled toType:MPSDataTypeFloat16 name:nil];
          [targets addObject:kind == 8 ? scaled : rounded];
        } else if (kind == 10) {
          [targets addObject:[g softMaxWithTensor:x axis:3 name:nil]];
        } else {
          auto v = [g placeholderWithShape:bthk dataType:MPSDataTypeFloat16 name:nil];
          [feeds addObject:v];
          auto vh = [g transposeTensor:v dimension:1 withDimension:2 name:nil];
          [targets addObject:[g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:vh name:nil]];
        }
      } else {
        auto q = [g placeholderWithShape:bthq dataType:MPSDataTypeFloat16 name:nil];
        auto k = [g placeholderWithShape:bthk dataType:MPSDataTypeFloat16 name:nil];
        auto v = [g placeholderWithShape:bthk dataType:MPSDataTypeFloat16 name:nil];
        [feeds addObjectsFromArray:@[q, k, v]];
        auto qh = [g transposeTensor:q dimension:1 withDimension:2 name:nil];
        auto kh = [g transposeTensor:k dimension:1 withDimension:2 name:nil];
        auto vh = [g transposeTensor:v dimension:1 withDimension:2 name:nil];
        if (kind == 12) {
          qh = [g castTensor:qh toType:MPSDataTypeFloat32 name:nil];
          kh = [g castTensor:kh toType:MPSDataTypeFloat32 name:nil];
          vh = [g castTensor:vh toType:MPSDataTypeFloat32 name:nil];
        }
        auto kt = [g transposeTensor:kh dimension:2 withDimension:3 name:nil];
        qk = [g matrixMultiplicationWithPrimaryTensor:qh secondaryTensor:kt name:nil];
        auto scores = qk;
        if (qk.dataType == MPSDataTypeFloat32) {
          auto eps = [g constantWithScalar:1e-20f shape:@[@1] dataType:MPSDataTypeFloat32];
          scores = [g additionWithPrimaryTensor:scores secondaryTensor:eps name:nil];
        } else {
          scores = [g castTensor:scores toType:MPSDataTypeFloat32 name:nil];
        }
        auto scale = [g constantWithScalar:(float)(1.0 / std::sqrt((double)D))
                                    shape:@[@1] dataType:MPSDataTypeFloat32];
        scaled = [g multiplicationWithPrimaryTensor:scores secondaryTensor:scale name:nil];
        rounded = kind == 12 ? scaled : [g castTensor:scaled toType:MPSDataTypeFloat16 name:nil];
        prob = [g softMaxWithTensor:rounded axis:3 name:nil];
        ctx = [g matrixMultiplicationWithPrimaryTensor:prob secondaryTensor:vh name:nil];
        if (kind == 12) ctx = [g castTensor:ctx toType:MPSDataTypeFloat16 name:nil];
        if (kind == 1) [targets addObject:qk];
        if (kind == 2) [targets addObject:scaled];
        if (kind == 3) [targets addObject:rounded];
        if (kind == 4) [targets addObject:prob];
        if (kind == 6) [targets addObject:prob];
        if (kind == 5 || kind == 6 || kind == 12) [targets addObject:ctx];
      }
      if (targets.count != (NSUInteger)outputCount) return nullptr;
      NSMutableDictionary *feedTypes = [NSMutableDictionary new];
      NSMutableDictionary *feedData = [NSMutableDictionary new];
      // Prune unused prefix inputs; compiling an unused feed is not portable.
      const int usedInputs = kind == 1 || kind == 2 || kind == 3 || kind == 4 ? 2 : inputCount;
      for (int i = 0; i < usedInputs; ++i) {
        MPSGraphTensor *t = feeds[i];
        feedTypes[t] = [[MPSGraphShapedType alloc] initWithShape:t.shape dataType:t.dataType];
        feedData[t] = [[MPSGraphTensorData alloc] initWithMTLBuffer:ins[i] shape:t.shape dataType:t.dataType];
      }
      auto ex = [g compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:feedTypes targetTensors:targets targetOperations:nil
               compilationDescriptor:nil];
      if (!ex) return nullptr;
      ex.options = MPSGraphOptionsNone;
      NSMutableArray *orderedInputs = [NSMutableArray new];
      for (MPSGraphTensor *t in ex.feedTensors) {
        if (!feedData[t]) return nullptr;
        [orderedInputs addObject:feedData[t]];
      }
      NSMutableDictionary *targetData = [NSMutableDictionary new];
      for (int i = 0; i < outputCount; ++i) {
        MPSGraphTensor *t = targets[i];
        const auto expected = kind == 2 || kind == 8 ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
        if (t.dataType != expected || ![t.shape isEqualToArray:
            ((kind == 5 || kind == 11 || kind == 12 || (kind == 6 && i == 1)) ? bhqd : bhqk)])
          return nullptr;
        targetData[t] = [[MPSGraphTensorData alloc] initWithMTLBuffer:outs[i] shape:t.shape dataType:t.dataType];
      }
      NSMutableArray *orderedOutputs = [NSMutableArray new];
      for (MPSGraphTensor *t in ex.targetTensors) {
        if (!targetData[t]) return nullptr;
        [orderedOutputs addObject:targetData[t]];
      }
      if (orderedOutputs.count != targets.count) return nullptr;
      plan.graph = g; plan.executable = ex;
      plan.inputs = orderedInputs; plan.outputs = orderedOutputs;
      plan.execution = [MPSGraphExecutableExecutionDescriptor new];
      plan.execution.waitUntilCompleted = NO;
      return (__bridge_retained void *)plan;
    } @catch (NSException *e) {
      std::fprintf(stderr, "stage create exception: %s\n", e.reason.UTF8String);
      return nullptr;
    }
  }
}

extern "C" int32_t wan_stage_encode(void *handle) {
  if (!handle) return -1;
  auto plan = (__bridge WanF16StagePlan *)handle;
  __block int32_t result = -2;
  dispatch_sync(torch::mps::get_dispatch_queue(), ^{
    @autoreleasepool {
      auto stream = at::mps::getCurrentMPSStream();
      stream->endKernelCoalescing();
      auto cb = stream->commandBuffer();
      if (plan.buffers[0].device != stream->device()) { result = -5; return; }
      if (plan.opaque) {
        result = gs_wan_attention_mps_encode(plan.opaque, (__bridge void *)cb, 0);
      } else {
        @try {
          auto out = [plan.executable encodeToCommandBuffer:cb inputsArray:plan.inputs
                     resultsArray:plan.outputs executionDescriptor:plan.execution];
          result = out.count == plan.outputs.count ? 0 : -3;
        } @catch (NSException *e) {
          std::fprintf(stderr, "stage encode exception: %s\n", e.reason.UTF8String);
          result = -4;
        }
      }
    }
  });
  return result;
}

extern "C" void wan_stage_release(void *handle) {
  if (handle) { (void)(__bridge_transfer WanF16StagePlan *)handle; }
}

static const char *metricSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint n, atype, btype, layout, slot, control; };
float read_value(device const uchar *p, uint i, uint dtype) {
  return dtype == 0 ? ((device const float *)p)[i] : float(((device const half *)p)[i]);
}
kernel void stage_fill(device half *a [[buffer(0)]],
                       constant uint &n [[buffer(1)]],
                       constant uint &salt [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
  if (i>=n) return;
  uint x=i+salt*0x9e3779b9u;
  x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16;
  a[i]=half(float(int(x&1023u)-512)/256.0f);
}
kernel void stage_metrics(device const uchar *a [[buffer(0)]],
                          device const uchar *b [[buffer(1)]],
                          device atomic_uint *m [[buffer(2)]],
                          constant Params &p [[buffer(3)]],
                          uint tid [[thread_position_in_grid]],
                          uint width [[threads_per_grid]]) {
  uint changed=0, bad=0, nf=0, ma=0, mr=0;
  for (uint i=tid; i<p.n; i+=width) {
    uint ai=i;
    if (p.layout) {
      uint d=i%128, q=(i/128)%128, h=(i/(128*128))%40;
      ai=(q*40+h)*128+d;
    }
    float y=read_value(b,i,p.btype);
    float x=p.control==1 ? y+1.0f : (p.control==2 ? NAN : read_value(a,ai,p.atype));
    bool finite=isfinite(x)&&isfinite(y);
    float err=finite ? abs(x-y) : INFINITY;
    float tol=2e-5f+2e-5f*abs(y);
    changed+=uint(x!=y); bad+=uint(!finite||err>tol); nf+=uint(!finite);
    ma=max(ma,as_type<uint>(err)); mr=max(mr,as_type<uint>(err/tol));
  }
  changed=simd_sum(changed); bad=simd_sum(bad); nf=simd_sum(nf);
  ma=simd_max(ma); mr=simd_max(mr);
  if ((tid&31)==0) {
    uint off=p.slot*5;
    atomic_fetch_add_explicit(m+off,changed,memory_order_relaxed);
    atomic_fetch_add_explicit(m+off+1,bad,memory_order_relaxed);
    atomic_fetch_add_explicit(m+off+2,nf,memory_order_relaxed);
    atomic_fetch_max_explicit(m+off+3,ma,memory_order_relaxed);
    atomic_fetch_max_explicit(m+off+4,mr,memory_order_relaxed);
  }
}
)METAL";

extern "C" int32_t wan_stage_compare(void *aHandle, void *bHandle, void *mHandle,
    uint64_t n, int32_t atype, int32_t btype, int32_t layout, int32_t slot, int32_t control) {
  if (!aHandle || !bHandle || !mHandle || n == 0 || n > SN || atype < 0 || atype > 1 ||
      btype < 0 || btype > 1 || layout < 0 || layout > 1 || (layout && n != QN) ||
      slot < 0 || slot >= 128 || control < 0 || control > 2) return -1;
  auto a = (__bridge id<MTLBuffer>)aHandle;
  auto b = (__bridge id<MTLBuffer>)bHandle;
  auto m = (__bridge id<MTLBuffer>)mHandle;
  if (!stage_buffer(a, n*(atype == 0 ? 4 : 2), a.device) ||
      !stage_buffer(b, n*(btype == 0 ? 4 : 2), a.device) ||
      !stage_buffer(m, (slot+1)*5*4, a.device) ||
      !wan_attention_mps_disjoint(a,m) || !wan_attention_mps_disjoint(b,m)) return -2;
  static id<MTLComputePipelineState> pipeline;
  if (!pipeline) {
    NSError *error = nil;
    auto options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    auto library = [a.device newLibraryWithSource:[NSString stringWithUTF8String:metricSource] options:options error:&error];
    if (!library) { std::fprintf(stderr, "%s\n", error.description.UTF8String); return -3; }
    pipeline = [a.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"stage_metrics"] error:&error];
    if (!pipeline || pipeline.threadExecutionWidth != 32) return -4;
  }
  if (pipeline.device != a.device) return -5;
  const struct { uint32_t n, atype, btype, layout, slot, control; } params = {
      (uint32_t)n,(uint32_t)atype,(uint32_t)btype,
      (uint32_t)layout,(uint32_t)slot,(uint32_t)control};
  __block int32_t result = 0;
  dispatch_sync(torch::mps::get_dispatch_queue(), ^{
    auto stream = at::mps::getCurrentMPSStream();
    if (stream->device() != a.device) { result = -6; return; }
    auto encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a offset:0 atIndex:0];
    [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBuffer:m offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(32768,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    // End coalescing so a following Torch graph cannot reorder a reused slot.
    stream->endKernelCoalescing();
  });
  return result;
}

extern "C" int32_t wan_stage_fill(void *handle, uint64_t n, int32_t salt) {
  if (!handle || (n != QN && n != KN) || salt < 1 || salt > 4) return -1;
  auto buffer = (__bridge id<MTLBuffer>)handle;
  if (!stage_buffer(buffer,n*2,buffer.device)) return -2;
  static id<MTLComputePipelineState> pipeline;
  if (!pipeline) {
    NSError *error = nil;
    auto options = [MTLCompileOptions new]; options.mathMode = MTLMathModeSafe;
    auto library = [buffer.device newLibraryWithSource:[NSString stringWithUTF8String:metricSource]
                                             options:options error:&error];
    if (!library) return -3;
    pipeline = [buffer.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"stage_fill"] error:&error];
    if (!pipeline) return -4;
  }
  if (pipeline.device != buffer.device) return -5;
  const uint32_t count = (uint32_t)n, inputSalt = (uint32_t)salt;
  __block int32_t result = 0;
  dispatch_sync(torch::mps::get_dispatch_queue(), ^{
    auto stream = at::mps::getCurrentMPSStream();
    if (stream->device() != buffer.device) { result = -6; return; }
    auto encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&count length:sizeof(count) atIndex:1];
    [encoder setBytes:&inputSalt length:sizeof(inputSalt) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    stream->endKernelCoalescing();
  });
  return result;
}
