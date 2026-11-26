// Diagnostic-only J5760 baseline FFN chain. This translation unit includes
// the production decoder bridge once to reuse its exact F32-compute MPSGraph
// builders, BF16 output boundary, graph encoder, kernels, and API profiling.
// Do not also link pixelumm_decoder_block_mps_bridge.o with this file.

#if __has_include(<Foundation/Foundation.h>) && \
    __has_include(<Metal/Metal.h>) && \
    __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)

#include "../../src/ml/video/pixelumm_decoder_block_mps_bridge.mm"

static constexpr uint32_t kPixelFFNAATokens = 5760u;
static constexpr uint32_t kPixelFFNAADim = 4096u;
static constexpr uint32_t kPixelFFNAAWidth = 12288u;
static constexpr uint32_t kPixelFFNAACount = 7u;

enum PixelUMMFFNAACounter : uint32_t {
  PixelFFNAAEncodeCalls = 0,
  PixelFFNAAGateGraphEncodes = 1,
  PixelFFNAAUpGraphEncodes = 2,
  PixelFFNAADownGraphEncodes = 3,
  PixelFFNAASiluDispatches = 4,
  PixelFFNAAProductDispatches = 5,
  PixelFFNAARawCaptureCopies = 6,
};

@interface PixelUMMFFNAAPlan : NSObject {
 @public
  uint64_t counters[kPixelFFNAACount];
}
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *borrowedBuffers;
@property(nonatomic, strong) PixelUMMGraphStep *gate;
@property(nonatomic, strong) PixelUMMGraphStep *up;
@property(nonatomic, strong) PixelUMMGraphStep *down;
@property(nonatomic, strong) id<MTLComputePipelineState> silu;
@property(nonatomic, strong) id<MTLComputePipelineState> product;
@end

@implementation PixelUMMFFNAAPlan
@end

static bool pixel_ffn_aa_checked_sizes(uint32_t tokens,
                                       uint64_t *hiddenBytes,
                                       uint64_t *wideBytes,
                                       uint64_t *weightBytes) {
  return pixel_product(tokens, uint64_t(kPixelFFNAADim) * sizeof(uint16_t),
                       hiddenBytes) &&
         pixel_product(tokens, uint64_t(kPixelFFNAAWidth) * sizeof(uint16_t),
                       wideBytes) &&
         pixel_product(kPixelFFNAAWidth,
                       uint64_t(kPixelFFNAADim) * sizeof(uint16_t),
                       weightBytes);
}

static bool pixel_ffn_aa_copy_raw(MPSCommandBuffer *commandBuffer,
                                  PixelUMMFFNAAPlan *plan,
                                  id<MTLBuffer> source,
                                  id<MTLBuffer> destination,
                                  uint64_t bytes,
                                  char *errorText, size_t errorCapacity) {
  id<MTLCommandBuffer> root = pixel_current_root(
      commandBuffer, plan.device, errorText, errorCapacity);
  id<MTLBlitCommandEncoder> blit = root ? [root blitCommandEncoder] : nil;
  if (!blit) {
    pixel_set_error(errorText, errorCapacity,
                    "unable to open J5760 FFN raw-capture blit encoder");
    return false;
  }
  [blit copyFromBuffer:source sourceOffset:0
              toBuffer:destination destinationOffset:0 size:(NSUInteger)bytes];
  [blit endEncoding];
  ++plan->counters[PixelFFNAARawCaptureCopies];
  return true;
}

extern "C" void *gs_pixelumm_ffn_aa_create(
    void *deviceHandle,
    void *normInputHandle,
    void *gateWeightHandle,
    void *upWeightHandle,
    void *downWeightHandle,
    void *gateWorkHandle,
    void *upWorkHandle,
    void *downOutputHandle,
    void *gateCaptureHandle,
    void *upCaptureHandle,
    int32_t tokens,
    char *errorText,
    size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!deviceHandle || !normInputHandle || !gateWeightHandle ||
      !upWeightHandle || !downWeightHandle || !gateWorkHandle ||
      !upWorkHandle || !downOutputHandle || !gateCaptureHandle ||
      !upCaptureHandle || tokens != int32_t(kPixelFFNAATokens)) {
    pixel_set_error(errorText, errorCapacity,
                    "invalid J5760 baseline FFN bridge arguments");
    return nullptr;
  }

  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      @try {
        id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
        id<MTLBuffer> normInput = (__bridge id<MTLBuffer>)normInputHandle;
        id<MTLBuffer> gateWeight = (__bridge id<MTLBuffer>)gateWeightHandle;
        id<MTLBuffer> upWeight = (__bridge id<MTLBuffer>)upWeightHandle;
        id<MTLBuffer> downWeight = (__bridge id<MTLBuffer>)downWeightHandle;
        id<MTLBuffer> gateWork = (__bridge id<MTLBuffer>)gateWorkHandle;
        id<MTLBuffer> upWork = (__bridge id<MTLBuffer>)upWorkHandle;
        id<MTLBuffer> downOutput = (__bridge id<MTLBuffer>)downOutputHandle;
        id<MTLBuffer> gateCapture = (__bridge id<MTLBuffer>)gateCaptureHandle;
        id<MTLBuffer> upCapture = (__bridge id<MTLBuffer>)upCaptureHandle;

        uint64_t hiddenBytes = 0;
        uint64_t wideBytes = 0;
        uint64_t weightBytes = 0;
        if (!pixel_ffn_aa_checked_sizes(uint32_t(tokens), &hiddenBytes,
                                        &wideBytes, &weightBytes)) {
          pixel_set_error(errorText, errorCapacity,
                          "J5760 FFN byte extent overflow");
          return nullptr;
        }

        // Validate every borrowed resource before constructing any graph or
        // pipeline. All tensors use exact compact BF16 shared-buffer extents.
        if (!device ||
            !pixel_buffer(normInput, device, hiddenBytes) ||
            !pixel_buffer(gateWeight, device, weightBytes) ||
            !pixel_buffer(upWeight, device, weightBytes) ||
            !pixel_buffer(downWeight, device, weightBytes) ||
            !pixel_buffer(gateWork, device, wideBytes) ||
            !pixel_buffer(upWork, device, wideBytes) ||
            !pixel_buffer(downOutput, device, hiddenBytes) ||
            !pixel_buffer(gateCapture, device, wideBytes) ||
            !pixel_buffer(upCapture, device, wideBytes)) {
          pixel_set_error(errorText, errorCapacity,
              "J5760 FFN buffers must have exact BF16 extent, shared storage, and the requested device");
          return nullptr;
        }

        NSArray<id<MTLBuffer>> *borrowed = @[
          normInput, gateWeight, upWeight, downWeight, gateWork,
          upWork, downOutput, gateCapture, upCapture,
        ];
        for (NSUInteger i = 0; i < borrowed.count; ++i) {
          for (NSUInteger j = i + 1; j < borrowed.count; ++j) {
            if (!pixel_disjoint(borrowed[i], borrowed[j])) {
              pixel_set_error(errorText, errorCapacity,
                  "J5760 FFN input, weights, work, output, and capture buffers must be pairwise disjoint");
              return nullptr;
            }
          }
        }

        PixelUMMFFNAAPlan *plan = [PixelUMMFFNAAPlan new];
        if (!plan) {
          pixel_set_error(errorText, errorCapacity,
                          "unable to allocate J5760 FFN plan");
          return nullptr;
        }
        plan.device = device;
        plan.borrowedBuffers = borrowed;

        // These are the production F32-compute FFN graphs. Keep their BF16
        // bindings and output casts exactly as in the decoder implementation.
        plan.gate = pixel_linear_step(device, normInput, gateWeight, gateWork,
            kPixelFFNAATokens, kPixelFFNAADim, kPixelFFNAAWidth,
            @"ffn_aa_gate", errorText, errorCapacity, true);
        if (!plan.gate) return nullptr;
        plan.up = pixel_linear_step(device, normInput, upWeight, upWork,
            kPixelFFNAATokens, kPixelFFNAADim, kPixelFFNAAWidth,
            @"ffn_aa_up", errorText, errorCapacity, true);
        if (!plan.up) return nullptr;
        plan.down = pixel_linear_step(device, upWork, downWeight, downOutput,
            kPixelFFNAATokens, kPixelFFNAAWidth, kPixelFFNAADim,
            @"ffn_aa_down", errorText, errorCapacity, true);
        if (!plan.down) return nullptr;

        NSError *metalError = nil;
        NSString *source = [NSString stringWithUTF8String:pixel_metal_source];
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
          options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
          options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        id<MTLLibrary> library = source
            ? [device newLibraryWithSource:source options:options error:&metalError]
            : nil;
        if (!library) {
          pixel_set_ns_error(errorText, errorCapacity,
              [NSString stringWithFormat:@"unable to compile production PixelUMM Metal source: %@",
                  metalError.localizedDescription ?: @"source unavailable"]);
          return nullptr;
        }
        plan.silu = pixel_pipeline(device, library, @"pixelumm_silu",
                                   errorText, errorCapacity);
        if (!plan.silu) return nullptr;
        plan.product = pixel_pipeline(device, library, @"pixelumm_product",
                                      errorText, errorCapacity);
        if (!plan.product) return nullptr;

        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"J5760 FFN plan preparation exception %@: %@",
                exception.name, exception.reason]);
        return nullptr;
      }
    }
  }

  pixel_set_error(errorText, errorCapacity,
                  "J5760 FFN MPSGraph path requires macOS 14 or newer");
  return nullptr;
}

extern "C" int32_t gs_pixelumm_ffn_aa_encode(
    void *planHandle, void *commandBufferHandle, int32_t captureRaw,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle || !commandBufferHandle ||
      (captureRaw != 0 && captureRaw != 1)) {
    pixel_set_error(errorText, errorCapacity,
                    "invalid J5760 FFN plan, command buffer, or capture flag");
    return -1;
  }

  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      @try {
        PixelUMMFFNAAPlan *plan = (__bridge PixelUMMFFNAAPlan *)planHandle;
        MPSCommandBuffer *commandBuffer =
            (__bridge MPSCommandBuffer *)commandBufferHandle;
        if (!plan.device || !plan.gate || !plan.up || !plan.down ||
            !plan.silu || !plan.product || !plan.borrowedBuffers ||
            plan.borrowedBuffers.count != 9) {
          pixel_set_error(errorText, errorCapacity,
                          "incomplete J5760 FFN plan");
          return -2;
        }
        if (!pixel_current_root(commandBuffer, plan.device,
                                errorText, errorCapacity)) return -3;

        id<MTLBuffer> gateWork = plan.borrowedBuffers[4];
        id<MTLBuffer> upWork = plan.borrowedBuffers[5];
        id<MTLBuffer> gateCapture = plan.borrowedBuffers[7];
        id<MTLBuffer> upCapture = plan.borrowedBuffers[8];
        const uint64_t wideBytes = uint64_t(kPixelFFNAATokens) *
                                   kPixelFFNAAWidth * sizeof(uint16_t);
        PixelCountParams count{kPixelFFNAATokens * kPixelFFNAAWidth};

        if (!pixel_encode_graph(commandBuffer, plan.gate,
                                errorText, errorCapacity)) return -4;
        ++plan->counters[PixelFFNAAGateGraphEncodes];
        if (captureRaw && !pixel_ffn_aa_copy_raw(commandBuffer, plan,
                gateWork, gateCapture, wideBytes, errorText, errorCapacity))
          return -5;

        if (!pixel_encode_graph(commandBuffer, plan.up,
                                errorText, errorCapacity)) return -6;
        ++plan->counters[PixelFFNAAUpGraphEncodes];
        if (captureRaw && !pixel_ffn_aa_copy_raw(commandBuffer, plan,
                upWork, upCapture, wideBytes, errorText, errorCapacity))
          return -7;

        if (!pixel_encode_kernel(commandBuffer, plan.device, plan.silu,
                @[gateWork], &count, sizeof(count), uint64_t(count.count), false,
                errorText, errorCapacity)) return -8;
        ++plan->counters[PixelFFNAASiluDispatches];

        if (!pixel_encode_kernel(commandBuffer, plan.device, plan.product,
                @[gateWork, upWork], &count, sizeof(count),
                uint64_t(count.count), false,
                errorText, errorCapacity)) return -9;
        ++plan->counters[PixelFFNAAProductDispatches];

        if (!pixel_encode_graph(commandBuffer, plan.down,
                                errorText, errorCapacity)) return -10;
        ++plan->counters[PixelFFNAADownGraphEncodes];
        ++plan->counters[PixelFFNAAEncodeCalls];
        return 0;
      } @catch (NSException *exception) {
        pixel_set_ns_error(errorText, errorCapacity,
            [NSString stringWithFormat:@"J5760 FFN encode exception %@: %@",
                exception.name, exception.reason]);
        return -11;
      }
    }
  }

  pixel_set_error(errorText, errorCapacity,
                  "J5760 FFN MPSGraph path requires macOS 14 or newer");
  return -12;
}

// Call after the caller's terminal wait. This read-only snapshot lets the
// harness prove that no projection or pointwise operation was skipped.
extern "C" int32_t gs_pixelumm_ffn_aa_stats(
    void *planHandle, uint64_t *counts, int32_t count) {
  if (!planHandle || !counts || count != int32_t(kPixelFFNAACount)) return -1;
  PixelUMMFFNAAPlan *plan = (__bridge PixelUMMFFNAAPlan *)planHandle;
  for (uint32_t index = 0; index < kPixelFFNAACount; ++index)
    counts[index] = plan->counters[index];
  return 0;
}

extern "C" void gs_pixelumm_ffn_aa_release(void *planHandle) {
  if (!planHandle) return;
  @autoreleasepool {
    PixelUMMFFNAAPlan *plan =
        (__bridge_transfer PixelUMMFFNAAPlan *)planHandle;
    (void)plan;
  }
}

#else

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static void pixel_ffn_aa_stub_error(char *errorText, size_t errorCapacity) {
  if (!errorText || errorCapacity == 0) return;
  const char *message = "J5760 FFN bridge requires macOS 14 or newer";
  (void)snprintf(errorText, errorCapacity, "%s", message);
}

extern "C" void *gs_pixelumm_ffn_aa_create(
    void *, void *, void *, void *, void *, void *, void *, void *, void *,
    void *, int32_t, char *errorText, size_t errorCapacity) {
  pixel_ffn_aa_stub_error(errorText, errorCapacity);
  return nullptr;
}

extern "C" int32_t gs_pixelumm_ffn_aa_encode(
    void *, void *, int32_t, char *errorText, size_t errorCapacity) {
  pixel_ffn_aa_stub_error(errorText, errorCapacity);
  return -1;
}

extern "C" int32_t gs_pixelumm_ffn_aa_stats(
    void *, uint64_t *, int32_t) {
  return -1;
}

extern "C" void gs_pixelumm_ffn_aa_release(void *) {}

#endif
