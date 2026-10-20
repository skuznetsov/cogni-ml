// Optional Wan UMT5 rank-three F32 FC2 backend. The default PyTorch 2.8 MPS
// linear path on macOS 15+ uses this MPSNDArray operation. No command creation,
// submission, wait, CPU activation access or GPU-buffer allocation occurs here.
// MPS implementation-private resource allocation is outside this guarantee.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdint.h>
#include <limits>

@interface CogniWanT5FC2Plan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) MPSNDArrayMatrixMultiplication *kernel;
@property(nonatomic, strong) MPSNDArray *input;
@property(nonatomic, strong) MPSNDArray *weight;
@property(nonatomic, strong) MPSNDArray *output;
@property(nonatomic, strong) NSArray<MPSNDArray *> *sources;
@end
@implementation CogniWanT5FC2Plan
@end

static bool wan_t5_disjoint(id<MTLBuffer> a, id<MTLBuffer> b) {
  if (a == b || a.gpuAddress == 0 || b.gpuAddress == 0) return false;
  uint64_t sa = a.gpuAddress, sb = b.gpuAddress;
  uint64_t ea = sa + a.length, eb = sb + b.length;
  return ea >= sa && eb >= sb && (ea <= sb || eb <= sa);
}

static bool wan_t5_extent(int32_t a, int32_t b, int32_t c, uint64_t *bytes) {
  if (a <= 0 || b <= 0 || c <= 0) return false;
  uint64_t n = (uint64_t)a * (uint64_t)b;
  // The caller's Metal dispatch/storage contract uses signed 32-bit counts.
  if (n > INT32_MAX || n * (uint64_t)c > INT32_MAX) return false;
  *bytes = n * (uint64_t)c * sizeof(float);
  return true;
}

extern "C" void *gs_wan_t5_mps_fc2_plan_create(
    void *device_handle, int32_t batches, int32_t tokens, int32_t din,
    int32_t dout, void *input_handle, void *weight_handle, void *output_handle) {
  if (!device_handle || !input_handle || !weight_handle || !output_handle)
    return nullptr;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t ib = 0, wb = 0, ob = 0;
      if (!wan_t5_extent(batches, tokens, din, &ib) ||
          !wan_t5_extent(1, dout, din, &wb) ||
          !wan_t5_extent(batches, tokens, dout, &ob)) return nullptr;
      id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
      id<MTLBuffer> x = (__bridge id<MTLBuffer>)input_handle;
      id<MTLBuffer> w = (__bridge id<MTLBuffer>)weight_handle;
      id<MTLBuffer> y = (__bridge id<MTLBuffer>)output_handle;
      if (x.device != device || w.device != device || y.device != device ||
          x.length < ib || w.length < wb || y.length < ob ||
          !wan_t5_disjoint(x, w) || !wan_t5_disjoint(x, y) ||
          !wan_t5_disjoint(w, y)) return nullptr;
      @try {
        MPSNDArrayDescriptor *xd = [MPSNDArrayDescriptor
            descriptorWithDataType:MPSDataTypeFloat32
            shape:@[@(batches), @(tokens), @(din)]];
        MPSNDArrayDescriptor *wd = [MPSNDArrayDescriptor
            descriptorWithDataType:MPSDataTypeFloat32 shape:@[@(dout), @(din)]];
        MPSNDArrayDescriptor *yd = [MPSNDArrayDescriptor
            descriptorWithDataType:MPSDataTypeFloat32
            shape:@[@(batches), @(tokens), @(dout)]];
        xd.preferPackedRows = YES;
        wd.preferPackedRows = YES;
        yd.preferPackedRows = YES;
        [wd transposeDimension:0 withDimension:1];
        CogniWanT5FC2Plan *plan = [CogniWanT5FC2Plan new];
        plan.device = device;
        plan.input = [[MPSNDArray alloc] initWithBuffer:x offset:0 descriptor:xd];
        plan.weight = [[MPSNDArray alloc] initWithBuffer:w offset:0 descriptor:wd];
        plan.output = [[MPSNDArray alloc] initWithBuffer:y offset:0 descriptor:yd];
        plan.kernel = [[MPSNDArrayMatrixMultiplication alloc]
            initWithDevice:device sourceCount:2];
        if (!plan.input || !plan.weight || !plan.output || !plan.kernel)
          return nullptr;
        plan.sources = @[plan.input, plan.weight];
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_wan_t5_mps_fc2_encode(
    void *plan_handle, void *command_handle, void *encoder_handle) {
  if (!plan_handle || !command_handle || !encoder_handle) return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      CogniWanT5FC2Plan *plan = (__bridge CogniWanT5FC2Plan *)plan_handle;
      id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)command_handle;
      id<MTLComputeCommandEncoder> encoder =
          (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
      MTLCommandBufferStatus before = command.status;
      if (before >= MTLCommandBufferStatusCommitted ||
          command.commandQueue.device != plan.device ||
          encoder.device != plan.device) return -2;
      @try {
        [plan.kernel encodeToCommandEncoder:encoder commandBuffer:command
            sourceArrays:plan.sources destinationArray:plan.output];
      } @catch (NSException *exception) {
        (void)exception;
        return -3;
      }
      return command.status == before ? 0 : -4;
    }
  }
  return -5;
}

extern "C" void gs_wan_t5_mps_fc2_plan_release(void *plan_handle) {
  if (plan_handle) {
    id plan = (__bridge_transfer id)plan_handle;
    (void)plan;
  }
}
