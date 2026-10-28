// Diagnostic-only isolated PixelUMM FFN graph plans. This translation unit
// includes the production bridge to reuse its exact MPSGraph builders and
// async graph encoding. Do not also link pixelumm_decoder_block_mps_bridge.o.
#include "../../src/ml/video/pixelumm_decoder_block_mps_bridge.mm"

#if __has_include(<Foundation/Foundation.h>) && \
    __has_include(<Metal/Metal.h>) && \
    __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)

@interface PixelUMMFFNIsolatedPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) PixelUMMGraphStep *step;
@property(nonatomic) int32_t kind;
@end

@implementation PixelUMMFFNIsolatedPlan
@end

enum PixelUMMFFNIsolatedKind : int32_t {
  PixelUMMFFNIsolatedSeparateGate = 0,
  PixelUMMFFNIsolatedSeparateUp = 1,
  PixelUMMFFNIsolatedSharedGateUp = 2,
  PixelUMMFFNIsolatedDown = 3,
};

extern "C" void *gs_pixelumm_ffn_isolated_step_create(
    void *deviceHandle, void *normInputHandle, void *productInputHandle,
    void *gateWeightHandle, void *upWeightHandle, void *downWeightHandle,
    void *gateOutputHandle, void *upOutputHandle, void *downOutputHandle,
    int32_t tokens, int32_t kind, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!deviceHandle || !normInputHandle || !productInputHandle ||
      !gateWeightHandle || !upWeightHandle || !downWeightHandle ||
      !gateOutputHandle || !upOutputHandle || !downOutputHandle ||
      (tokens != 2048 && tokens != 21120) ||
      kind < PixelUMMFFNIsolatedSeparateGate ||
      kind > PixelUMMFFNIsolatedDown) {
    pixel_set_error(errorText, errorCapacity,
                    "invalid isolated PixelUMM FFN graph arguments");
    return nullptr;
  }

  @autoreleasepool {
    @try {
      id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
      id<MTLBuffer> normInput = (__bridge id<MTLBuffer>)normInputHandle;
      id<MTLBuffer> productInput = (__bridge id<MTLBuffer>)productInputHandle;
      id<MTLBuffer> gateWeight = (__bridge id<MTLBuffer>)gateWeightHandle;
      id<MTLBuffer> upWeight = (__bridge id<MTLBuffer>)upWeightHandle;
      id<MTLBuffer> downWeight = (__bridge id<MTLBuffer>)downWeightHandle;
      id<MTLBuffer> gateOutput = (__bridge id<MTLBuffer>)gateOutputHandle;
      id<MTLBuffer> upOutput = (__bridge id<MTLBuffer>)upOutputHandle;
      id<MTLBuffer> downOutput = (__bridge id<MTLBuffer>)downOutputHandle;

      uint64_t normBytes = 0;
      uint64_t productBytes = 0;
      uint64_t weightBytes = 0;
      uint64_t outputBytes = 0;
      uint64_t downBytes = 0;
      if (!pixel_product((uint64_t)tokens, 4096u * 2u, &normBytes) ||
          !pixel_product((uint64_t)tokens, 12288u * 2u, &productBytes) ||
          !pixel_product(4096u, 12288u * 2u, &weightBytes) ||
          !pixel_product((uint64_t)tokens, 12288u * 2u, &outputBytes) ||
          !pixel_product((uint64_t)tokens, 4096u * 2u, &downBytes)) {
        pixel_set_error(errorText, errorCapacity,
                        "isolated PixelUMM FFN byte extent overflow");
        return nullptr;
      }
      if (!pixel_buffer(normInput, device, normBytes) ||
          !pixel_buffer(productInput, device, productBytes) ||
          !pixel_buffer(gateWeight, device, weightBytes) ||
          !pixel_buffer(upWeight, device, weightBytes) ||
          !pixel_buffer(downWeight, device, weightBytes) ||
          !pixel_buffer(gateOutput, device, outputBytes) ||
          !pixel_buffer(upOutput, device, outputBytes) ||
          !pixel_buffer(downOutput, device, downBytes)) {
        pixel_set_error(errorText, errorCapacity,
                        "isolated PixelUMM FFN buffers have wrong device, storage or extent");
        return nullptr;
      }

      NSArray<id<MTLBuffer>> *borrowed = @[
        normInput, productInput, gateWeight, upWeight, downWeight,
        gateOutput, upOutput, downOutput,
      ];
      for (NSUInteger i = 0; i < borrowed.count; ++i) {
        for (NSUInteger j = i + 1; j < borrowed.count; ++j) {
          if (!pixel_disjoint(borrowed[i], borrowed[j])) {
            pixel_set_error(errorText, errorCapacity,
                            "isolated PixelUMM FFN buffers must be pairwise disjoint");
            return nullptr;
          }
        }
      }

      PixelUMMGraphStep *step = nil;
      switch (kind) {
        case PixelUMMFFNIsolatedSeparateGate:
          step = pixel_linear_step(device, normInput, gateWeight, gateOutput,
              (uint32_t)tokens, 4096u, 12288u, @"gate_projection",
              errorText, errorCapacity, true);
          break;
        case PixelUMMFFNIsolatedSeparateUp:
          step = pixel_linear_step(device, normInput, upWeight, upOutput,
              (uint32_t)tokens, 4096u, 12288u, @"up_projection",
              errorText, errorCapacity, true);
          break;
        case PixelUMMFFNIsolatedSharedGateUp:
          step = pixel_linear_shared_input_pair_step(device, normInput,
              gateWeight, upWeight, gateOutput, upOutput, (uint32_t)tokens,
              4096u, 12288u, @"shared_gate_up_projection",
              errorText, errorCapacity);
          break;
        case PixelUMMFFNIsolatedDown:
          step = pixel_linear_step(device, productInput, downWeight, downOutput,
              (uint32_t)tokens, 12288u, 4096u, @"down_projection",
              errorText, errorCapacity, true);
          break;
      }
      if (!step) return nullptr;

      PixelUMMFFNIsolatedPlan *plan = [PixelUMMFFNIsolatedPlan new];
      plan.device = device;
      plan.step = step;
      plan.kind = kind;
      if (!plan.device || !plan.step) {
        pixel_set_error(errorText, errorCapacity,
                        "unable to retain isolated PixelUMM FFN graph plan");
        return nullptr;
      }
      return (__bridge_retained void *)plan;
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"isolated PixelUMM FFN preparation exception %@: %@",
           exception.name, exception.reason]);
      return nullptr;
    }
  }
}

extern "C" int32_t gs_pixelumm_ffn_isolated_step_encode(
    void *planHandle, void *commandBufferHandle,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle || !commandBufferHandle) {
    pixel_set_error(errorText, errorCapacity,
                    "isolated PixelUMM FFN plan/chain handle is null");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMFFNIsolatedPlan *plan =
          (__bridge PixelUMMFFNIsolatedPlan *)planHandle;
      MPSCommandBuffer *commandBuffer =
          (__bridge MPSCommandBuffer *)commandBufferHandle;
      if (!plan.device || !plan.step) {
        pixel_set_error(errorText, errorCapacity,
                        "incomplete isolated PixelUMM FFN plan");
        return -2;
      }
      if (!pixel_current_root(commandBuffer, plan.device, errorText, errorCapacity))
        return -3;
      return pixel_encode_graph(commandBuffer, plan.step, errorText, errorCapacity)
          ? 0 : -4;
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity,
          [NSString stringWithFormat:@"isolated PixelUMM FFN encode exception %@: %@",
           exception.name, exception.reason]);
      return -5;
    }
  }
}

extern "C" void gs_pixelumm_ffn_isolated_step_release(void *planHandle) {
  if (!planHandle) return;
  @autoreleasepool {
    PixelUMMFFNIsolatedPlan *plan =
        (__bridge_transfer PixelUMMFFNIsolatedPlan *)planHandle;
    (void)plan;
  }
}

#else
extern "C" void *gs_pixelumm_ffn_isolated_step_create(
    void *, void *, void *, void *, void *, void *, void *, void *, void *,
    int32_t, int32_t, char *errorText, size_t errorCapacity) {
  pixel_set_error(errorText, errorCapacity,
                  "isolated PixelUMM FFN graph requires macOS 14 or newer");
  return nullptr;
}
extern "C" int32_t gs_pixelumm_ffn_isolated_step_encode(
    void *, void *, char *errorText, size_t errorCapacity) {
  pixel_set_error(errorText, errorCapacity,
                  "isolated PixelUMM FFN graph requires macOS 14 or newer");
  return -6;
}
extern "C" void gs_pixelumm_ffn_isolated_step_release(void *) {}
#endif
