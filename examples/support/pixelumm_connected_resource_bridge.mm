// Diagnostic-only resource snapshots around the production PixelUMM graph
// encode body. This translation unit includes the production bridge once; do
// not also link pixelumm_decoder_block_mps_bridge.o into this executable.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <cstring>
#include <libproc.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static constexpr uint32_t kPixelUMMConnectedResourceCapacity = 4096;

enum PixelUMMConnectedResourceGraph : uint32_t {
  PixelUMMResourceUnknown = 0,
  PixelUMMResourceInputNorm = 1,
  PixelUMMResourceInputNormGain = 2,
  PixelUMMResourceQProjection = 3,
  PixelUMMResourceKProjection = 4,
  PixelUMMResourceVProjection = 5,
  PixelUMMResourceQNormStatistics = 6,
  PixelUMMResourceKNormStatistics = 7,
  PixelUMMResourceQNormInverse = 8,
  PixelUMMResourceKNormInverse = 9,
  PixelUMMResourceAttentionQK = 10,
  PixelUMMResourceAttentionSoftmax = 11,
  PixelUMMResourceAttentionPV = 12,
  PixelUMMResourceOProjection = 13,
  PixelUMMResourcePostAttnNorm = 14,
  PixelUMMResourcePostAttnNormGain = 15,
  PixelUMMResourceGateProjection = 16,
  PixelUMMResourceUpProjection = 17,
  PixelUMMResourceSharedGateUpProjection = 18,
  PixelUMMResourceDownProjection = 19,
};

enum PixelUMMConnectedResourcePhase : uint32_t {
  PixelUMMResourceBeforeEncode = 0,
  PixelUMMResourceAfterEncode = 1,
  PixelUMMResourceAfterPoolDrain = 2,
  PixelUMMResourceAfterScope = 3,
};

struct PixelUMMConnectedResourceEvent {
  uint64_t monotonic_ns;
  uint64_t device_allocated_bytes;
  uint64_t process_phys_footprint_bytes;
  uint64_t process_rss_bytes;
  uint32_t graph_id;
  uint32_t graph_index;
  uint32_t phase;
  uint32_t pool_mode;
};

struct PixelUMMConnectedResourceInfo {
  uint32_t event_count;
  uint32_t overflow;
  uint32_t sample_failed;
  uint32_t unknown_graphs;
  uint32_t graph_calls;
};

static_assert(sizeof(PixelUMMConnectedResourceEvent) == 48,
              "PixelUMM connected resource event ABI changed");
static_assert(sizeof(PixelUMMConnectedResourceInfo) == 20,
              "PixelUMM connected resource summary ABI changed");

static PixelUMMConnectedResourceEvent
    gPixelUMMConnectedResourceEvents[kPixelUMMConnectedResourceCapacity];
static uint32_t gPixelUMMConnectedResourceEventCount = 0;
static uint32_t gPixelUMMConnectedResourceOverflow = 0;
static uint32_t gPixelUMMConnectedResourceSampleFailed = 0;
static uint32_t gPixelUMMConnectedResourceUnknownGraphs = 0;
static uint32_t gPixelUMMConnectedResourceGraphCalls = 0;
static uint32_t gPixelUMMConnectedResourcePoolMode = 0;
static bool gPixelUMMConnectedResourceActive = false;
static id<MTLDevice> gPixelUMMConnectedResourceDevice = nil;

static uint32_t pixelumm_connected_resource_graph_id(NSString *name) {
  if (!name) return PixelUMMResourceUnknown;
  if ([name isEqualToString:@"input_norm_pre_gain"]) return PixelUMMResourceInputNorm;
  if ([name isEqualToString:@"input_norm_gain"]) return PixelUMMResourceInputNormGain;
  if ([name isEqualToString:@"q_projection"]) return PixelUMMResourceQProjection;
  if ([name isEqualToString:@"k_projection"]) return PixelUMMResourceKProjection;
  if ([name isEqualToString:@"v_projection"]) return PixelUMMResourceVProjection;
  if ([name isEqualToString:@"q_norm_statistics"]) return PixelUMMResourceQNormStatistics;
  if ([name isEqualToString:@"k_norm_statistics"]) return PixelUMMResourceKNormStatistics;
  if ([name isEqualToString:@"q_norm_inverse"]) return PixelUMMResourceQNormInverse;
  if ([name isEqualToString:@"k_norm_inverse"]) return PixelUMMResourceKNormInverse;
  if ([name hasPrefix:@"attention_qk_q"]) return PixelUMMResourceAttentionQK;
  if ([name hasPrefix:@"attention_softmax_q"]) return PixelUMMResourceAttentionSoftmax;
  if ([name hasPrefix:@"attention_pv_q"]) return PixelUMMResourceAttentionPV;
  if ([name isEqualToString:@"o_projection"]) return PixelUMMResourceOProjection;
  if ([name isEqualToString:@"post_attn_norm_pre_gain"]) return PixelUMMResourcePostAttnNorm;
  if ([name isEqualToString:@"post_attn_norm_gain"]) return PixelUMMResourcePostAttnNormGain;
  if ([name isEqualToString:@"gate_projection"]) return PixelUMMResourceGateProjection;
  if ([name isEqualToString:@"up_projection"]) return PixelUMMResourceUpProjection;
  if ([name isEqualToString:@"shared_gate_up_projection"])
    return PixelUMMResourceSharedGateUpProjection;
  if ([name isEqualToString:@"down_projection"])
    return PixelUMMResourceDownProjection;
  return PixelUMMResourceUnknown;
}

static void pixelumm_connected_resource_sample(uint32_t graphId,
                                               uint32_t graphIndex,
                                               uint32_t phase) {
  if (!gPixelUMMConnectedResourceActive) return;
  if (gPixelUMMConnectedResourceEventCount >=
      kPixelUMMConnectedResourceCapacity) {
    gPixelUMMConnectedResourceOverflow = 1;
    return;
  }

  PixelUMMConnectedResourceEvent event{};
  timespec now{};
  rusage_info_v2 usage{};
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0 ||
      now.tv_nsec < 0 || !gPixelUMMConnectedResourceDevice ||
      proc_pid_rusage(getpid(), RUSAGE_INFO_V2,
                      reinterpret_cast<rusage_info_t *>(&usage)) != 0) {
    gPixelUMMConnectedResourceSampleFailed = 1;
  } else {
    event.monotonic_ns = uint64_t(now.tv_sec) * 1000000000u +
                         uint64_t(now.tv_nsec);
    event.device_allocated_bytes =
        uint64_t(gPixelUMMConnectedResourceDevice.currentAllocatedSize);
    event.process_phys_footprint_bytes = usage.ri_phys_footprint;
    event.process_rss_bytes = usage.ri_resident_size;
  }
  event.graph_id = graphId;
  event.graph_index = graphIndex;
  event.phase = phase;
  event.pool_mode = gPixelUMMConnectedResourcePoolMode;
  gPixelUMMConnectedResourceEvents[gPixelUMMConnectedResourceEventCount++] =
      event;
}

// This dependent template is declared before the production bridge include so
// the macro-only hook can wrap the exact existing graph body. Pool-on records
// after-encode before the nested pool drains, then records the post-drain
// sample. Pool-off never emits a synthetic drain event.
template <typename Step, typename Encode>
static bool pixelumm_connected_resource_scope(MPSCommandBuffer *commandBuffer,
                                              Step *step,
                                              Encode &&encode) {
  if (!gPixelUMMConnectedResourceActive)
    return encode();

  const uint32_t graphId =
      pixelumm_connected_resource_graph_id(step ? step.name : nil);
  const uint32_t graphIndex = gPixelUMMConnectedResourceGraphCalls++;
  if (graphId == PixelUMMResourceUnknown)
    ++gPixelUMMConnectedResourceUnknownGraphs;
  pixelumm_connected_resource_sample(graphId, graphIndex,
                                     PixelUMMResourceBeforeEncode);

  bool encoded = false;
  if (gPixelUMMConnectedResourcePoolMode != 0) {
    @autoreleasepool {
      encoded = encode();
      pixelumm_connected_resource_sample(graphId, graphIndex,
                                         PixelUMMResourceAfterEncode);
    }
    pixelumm_connected_resource_sample(graphId, graphIndex,
                                       PixelUMMResourceAfterPoolDrain);
  } else {
    encoded = encode();
    pixelumm_connected_resource_sample(graphId, graphIndex,
                                       PixelUMMResourceAfterEncode);
    pixelumm_connected_resource_sample(graphId, graphIndex,
                                       PixelUMMResourceAfterScope);
  }
  return encoded;
}

extern "C" int32_t gs_pixelumm_connected_resource_begin(void *deviceHandle,
                                                        int32_t poolMode) {
  if (!deviceHandle || (poolMode != 0 && poolMode != 1) ||
      gPixelUMMConnectedResourceActive)
    return -1;
  gPixelUMMConnectedResourceEventCount = 0;
  gPixelUMMConnectedResourceOverflow = 0;
  gPixelUMMConnectedResourceSampleFailed = 0;
  gPixelUMMConnectedResourceUnknownGraphs = 0;
  gPixelUMMConnectedResourceGraphCalls = 0;
  gPixelUMMConnectedResourcePoolMode = uint32_t(poolMode);
  memset(gPixelUMMConnectedResourceEvents, 0,
         sizeof(gPixelUMMConnectedResourceEvents));
  gPixelUMMConnectedResourceDevice =
      (__bridge id<MTLDevice>)deviceHandle;
  gPixelUMMConnectedResourceActive = true;
  return 0;
}

extern "C" int32_t gs_pixelumm_connected_resource_end(
    PixelUMMConnectedResourceInfo *info) {
  if (!info || !gPixelUMMConnectedResourceActive) return -1;
  *info = {gPixelUMMConnectedResourceEventCount,
           gPixelUMMConnectedResourceOverflow,
           gPixelUMMConnectedResourceSampleFailed,
           gPixelUMMConnectedResourceUnknownGraphs,
           gPixelUMMConnectedResourceGraphCalls};
  gPixelUMMConnectedResourceActive = false;
  gPixelUMMConnectedResourceDevice = nil;
  return 0;
}

extern "C" int32_t gs_pixelumm_connected_resource_event(
    uint32_t index, PixelUMMConnectedResourceEvent *event) {
  if (!event || index >= gPixelUMMConnectedResourceEventCount ||
      index >= kPixelUMMConnectedResourceCapacity)
    return -1;
  *event = gPixelUMMConnectedResourceEvents[index];
  return 0;
}

#define COGNI_PIXELUMM_GRAPH_RESOURCE_PROBE 1
#include "../../src/ml/video/pixelumm_decoder_block_mps_bridge.mm"
