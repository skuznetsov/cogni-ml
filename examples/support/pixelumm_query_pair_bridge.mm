// Bounded diagnostic graph. The included weight-reuse bridge includes the
// production PixelUMM block ABI exactly once; do not link its object too.
#include "pixelumm_ffn_weight_reuse_bridge.mm"

#import <objc/runtime.h>

@interface PixelUMMQueryPairPool : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *baselineScratch;
// Slot order: candidate scratch[0], [3], [4], [5], [6], [7].
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *extraBuffers;
@property(nonatomic, strong) NSArray<PixelUMMAttentionGeometry *> *geometries;
@property(nonatomic, copy) NSArray<NSNumber *> *tileGeometryIndices;
@property(nonatomic) uint64_t extraBytes;
@property(nonatomic) uint32_t wrapperCount;
@end
@implementation PixelUMMQueryPairPool
@end

@interface PixelUMMQueryPair : NSObject
@property(nonatomic, strong) PixelUMMDecoderPlan *plan;
@property(nonatomic, strong) PixelUMMQueryPairPool *pool;
@property(nonatomic, strong) id installationToken;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *baselineScratch;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *candidateScratch;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *baselineResident;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *candidateResident;
@property(nonatomic, strong) NSArray<PixelUMMAttentionGeometry *> *baselineGeometries;
@property(nonatomic, strong) NSArray<NSNumber *> *baselineTileGeometryIndices;
@property(nonatomic) uint32_t baselineMaxQuery;
@property(nonatomic) uint32_t baselineMaxKey;
@property(nonatomic) uint32_t baselineTileCount;
@property(nonatomic) BOOL selected;
@property(nonatomic) BOOL active;
@end
@implementation PixelUMMQueryPair
@end

// The plan retains only this token, so the wrapper may retain the plan without
// forming a cycle. Each plan can own at most one diagnostic wrapper.
static char pixel_query_pair_associated_key;

static bool pixel_query_pair_add(uint64_t left, uint64_t right, uint64_t *out) {
  if (!out || left > UINT64_MAX - right) return false;
  *out = left + right;
  return true;
}

static bool pixel_query_pair_exact_bytes(uint64_t *qBytes, uint64_t *scoreBytes,
                                         uint64_t *tableBytes, uint64_t *total) {
  uint64_t qOne = 0, scoreOne = 0, qTableOne = 0, kTableOne = 0;
  uint64_t twoQ = 0, twoScores = 0, twoTables = 0;
  if (!pixel_product(32u * 1280u, 128u * sizeof(float), &qOne) ||
      !pixel_product(32u * 1280u, 4320u * sizeof(float), &scoreOne) ||
      !pixel_product(6u, (1280u + 1u) * sizeof(int32_t), &qTableOne) ||
      !pixel_product(6u, (4320u + 1u) * sizeof(int32_t), &kTableOne) ||
      !pixel_product(2u, qOne, &twoQ) ||
      !pixel_product(2u, scoreOne, &twoScores) ||
      !pixel_query_pair_add(qTableOne, kTableOne, &twoTables)) return false;
  uint64_t sum = 0;
  if (!pixel_query_pair_add(twoQ, twoScores, &sum) ||
      !pixel_query_pair_add(sum, twoTables, &sum))
    return false;
  if (qBytes) *qBytes = twoQ;
  if (scoreBytes) *scoreBytes = twoScores;
  if (tableBytes) *tableBytes = twoTables;
  if (total) *total = sum;
  return true;
}

static bool pixel_query_pair_validate_plan(PixelUMMDecoderPlan *plan,
                                           char *errorText,
                                           size_t errorCapacity) {
  if (!plan || !plan.device || plan.encodingStarted || plan->nativeIslands ||
      plan.useNativeIslands || !plan.ffnF32 || plan.tokens != 5760 ||
      plan.temporal != 10 || plan.height != 18 || plan.width != 32 ||
      plan.tile != 8 || plan.dim != 4096 || plan.ffn != 12288 ||
      plan.heads != 32 || plan.kvHeads != 8 || plan.headDim != 128 ||
      plan.maxQuery != 640 || plan.maxKey != 4320 || plan.tileCount != 12 ||
      plan.sharedGateUpPrepared || plan.useSharedGateUp ||
      plan.sharedGateUpProjection || plan.softmaxPVPrepared || plan.useSoftmaxPV ||
      plan.softmaxPVOutputOnlyPrepared || plan.useSoftmaxPVOutputOnly ||
      plan.scaleSoftmaxPrepared || plan.useScaleSoftmax ||
      plan.softmaxPVCompare || plan.probabilitySnapshot || plan.attendedSnapshot ||
      plan.softmaxPVCheckMetrics || plan.core.count != kPixelCoreCount ||
      plan.weights.count != kPixelWeightCount ||
      plan.scratch.count != kPixelScratchCount ||
      !plan.steps || !plan.attentionGeometries ||
      plan.tileGeometryIndices.count != 12 || plan.attentionGeometries.count == 0) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair requires an idle baseline F32 J5760 T10x18x32 plan without other attention/FFN candidates");
    return false;
  }
  for (PixelUMMAttentionGeometry *geometry in plan.attentionGeometries) {
    if (!geometry || !geometry.qk || !geometry.softmax || !geometry.pv ||
        geometry.softmaxPV || geometry.softmaxPVOutputOnly || geometry.scaleSoftmax ||
        ![plan.steps containsObject:geometry.qk] ||
        ![plan.steps containsObject:geometry.softmax] ||
        ![plan.steps containsObject:geometry.pv]) {
      pixel_set_error(errorText, errorCapacity,
          "query-pair requires the original baseline attention geometries and steps");
      return false;
    }
  }
  if (!plan.gateProjection || !plan.upProjection || !plan.downProjection ||
      ![plan.steps containsObject:plan.gateProjection] ||
      ![plan.steps containsObject:plan.upProjection] ||
      ![plan.steps containsObject:plan.downProjection]) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair requires original F32 gate, up and down projection steps");
    return false;
  }

  uint64_t qBytes = 0, kvBytes = 0, scoreBytes = 0;
  uint64_t tableQBytes = 0, tableKBytes = 0, dBytes = 0, ffBytes = 0;
  if (!pixel_product(32u * 640u, 128u * sizeof(float), &qBytes) ||
      !pixel_product(32u * 4320u, 128u * sizeof(float), &kvBytes) ||
      !pixel_product(32u * 640u, 4320u * sizeof(float), &scoreBytes) ||
      !pixel_product(12u, (640u + 1u) * sizeof(int32_t), &tableQBytes) ||
      !pixel_product(12u, (4320u + 1u) * sizeof(int32_t), &tableKBytes) ||
      !pixel_product(5760u * 4096u, sizeof(uint16_t), &dBytes) ||
      !pixel_product(5760u * 12288u, sizeof(uint16_t), &ffBytes)) {
    pixel_set_error(errorText, errorCapacity, "query-pair baseline byte extent overflowed");
    return false;
  }
  const uint64_t expected[kPixelScratchCount] = {
    qBytes, kvBytes, kvBytes, scoreBytes, scoreBytes, qBytes,
    tableQBytes, tableKBytes, 4u, 5760u * sizeof(float),
    5760u * 32u * 2u * sizeof(float), 5760u * 8u * 2u * sizeof(float),
    5760u * 32u * 2u * sizeof(float), 5760u * 8u * 2u * sizeof(float)};
  for (NSUInteger index = 0; index < kPixelScratchCount; ++index) {
    if (!pixel_buffer(plan.scratch[index], plan.device, expected[index])) {
      pixel_set_error(errorText, errorCapacity,
          "query-pair baseline scratch has incompatible device, storage or extent");
      return false;
    }
  }
  if (!pixel_buffer(plan.core[1], plan.device, dBytes) ||
      !pixel_buffer(plan.core[2], plan.device, 5760u * 8u * 128u * sizeof(uint16_t)) ||
      !pixel_buffer(plan.core[3], plan.device, 5760u * 8u * 128u * sizeof(uint16_t)) ||
      !pixel_buffer(plan.core[4], plan.device, dBytes) ||
      !pixel_buffer(plan.core[6], plan.device, ffBytes) ||
      !pixel_buffer(plan.core[7], plan.device, ffBytes) ||
      !pixel_validate_table(plan.scratch[6], 12, 640, 5760) ||
      !pixel_validate_table(plan.scratch[7], 12, 4320, 5760)) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair baseline buffers or tile metadata do not match the admitted geometry");
    return false;
  }
  return true;
}

static bool pixel_query_pair_build_tables(PixelUMMDecoderPlan *plan,
                                           id<MTLBuffer> queryOutput,
                                           id<MTLBuffer> keyOutput,
                                           char *errorText,
                                           size_t errorCapacity) {
  if (!pixel_buffer(queryOutput, plan.device,
          6u * (1280u + 1u) * sizeof(int32_t)) ||
      !pixel_buffer(keyOutput, plan.device,
          6u * (4320u + 1u) * sizeof(int32_t))) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair candidate metadata buffers have incompatible extents");
    return false;
  }
  const int32_t *sourceQ = (const int32_t *)plan.scratch[6].contents;
  const int32_t *sourceK = (const int32_t *)plan.scratch[7].contents;
  int32_t *targetQ = (int32_t *)queryOutput.contents;
  int32_t *targetK = (int32_t *)keyOutput.contents;
  memset(targetQ, 0, (size_t)queryOutput.length);
  memset(targetK, 0, (size_t)keyOutput.length);

  std::vector<uint8_t> seen(plan.tokens, uint8_t(0));
  uint64_t queryTotal = 0;
  for (uint32_t tile = 0; tile < 12; ++tile) {
    const uint64_t qBase = uint64_t(tile) * 641u;
    const uint64_t kBase = uint64_t(tile) * 4321u;
    const int32_t qCount = sourceQ[qBase];
    const int32_t kCount = sourceK[kBase];
    if (qCount <= 0 || qCount > 640 || kCount <= 0 || kCount > 4320) {
      pixel_set_error(errorText, errorCapacity,
          "query-pair source metadata contains an invalid row count");
      return false;
    }
    queryTotal += uint32_t(qCount);
    for (int32_t offset = 0; offset < qCount; ++offset) {
      const int32_t token = sourceQ[qBase + 1u + uint32_t(offset)];
      if (token < 0 || uint32_t(token) >= plan.tokens || seen[uint32_t(token)]) {
        pixel_set_error(errorText, errorCapacity,
            "query-pair source queries do not partition tokens exactly once");
        return false;
      }
      seen[uint32_t(token)] = 1;
    }
    for (int32_t offset = 0; offset < kCount; ++offset) {
      const int32_t token = sourceK[kBase + 1u + uint32_t(offset)];
      if (token < 0 || uint32_t(token) >= plan.tokens) {
        pixel_set_error(errorText, errorCapacity,
            "query-pair source context contains an invalid token");
        return false;
      }
    }
  }
  if (queryTotal != plan.tokens ||
      std::find(seen.begin(), seen.end(), uint8_t(0)) != seen.end()) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair source queries do not cover every token exactly once");
    return false;
  }

  for (uint32_t pair = 0; pair < 6; ++pair) {
    const uint32_t firstTile = pair * 2u;
    const uint64_t qFirst = uint64_t(firstTile) * 641u;
    const uint64_t qSecond = uint64_t(firstTile + 1u) * 641u;
    const uint64_t kFirst = uint64_t(firstTile) * 4321u;
    const uint64_t kSecond = uint64_t(firstTile + 1u) * 4321u;
    const int32_t firstQCount = sourceQ[qFirst];
    const int32_t secondQCount = sourceQ[qSecond];
    const int32_t keyCount = sourceK[kFirst];
    if (sourceK[kSecond] != keyCount || keyCount <= 0 || keyCount > 4320) {
      pixel_set_error(errorText, errorCapacity,
          "adjacent query tiles do not have identical ordered key contexts");
      return false;
    }
    for (int32_t offset = 0; offset < keyCount; ++offset)
      if (sourceK[kFirst + 1u + uint32_t(offset)] !=
          sourceK[kSecond + 1u + uint32_t(offset)]) {
        pixel_set_error(errorText, errorCapacity,
            "adjacent query tiles do not have identical ordered key contexts");
        return false;
      }
    const int32_t queryCount = firstQCount + secondQCount;
    if (firstQCount <= 0 || secondQCount <= 0 || queryCount > 1280) {
      pixel_set_error(errorText, errorCapacity,
          "adjacent query tiles exceed the admitted pair query extent");
      return false;
    }
    const uint64_t targetQBase = uint64_t(pair) * 1281u;
    const uint64_t targetKBase = uint64_t(pair) * 4321u;
    targetQ[targetQBase] = queryCount;
    targetK[targetKBase] = keyCount;
    memcpy(targetQ + targetQBase + 1u, sourceQ + qFirst + 1u,
        size_t(firstQCount) * sizeof(int32_t));
    memcpy(targetQ + targetQBase + 1u + uint32_t(firstQCount),
        sourceQ + qSecond + 1u, size_t(secondQCount) * sizeof(int32_t));
    memcpy(targetK + targetKBase + 1u, sourceK + kFirst + 1u,
        size_t(keyCount) * sizeof(int32_t));
  }
  return true;
}

static PixelUMMQueryPairPool *pixel_query_pair_create_pool(
    PixelUMMDecoderPlan *plan, char *errorText, size_t errorCapacity) {
  uint64_t qBytes = 0, scoreBytes = 0, tableBytes = 0, totalBytes = 0;
  if (!pixel_query_pair_exact_bytes(&qBytes, &scoreBytes, &tableBytes,
                                    &totalBytes)) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair candidate byte extent overflowed");
    return nil;
  }
  const uint64_t lengths[6] = {
    qBytes / 2u, scoreBytes / 2u, scoreBytes / 2u,
    qBytes / 2u, 6u * (1280u + 1u) * sizeof(int32_t),
    6u * (4320u + 1u) * sizeof(int32_t)};
  NSMutableArray<id<MTLBuffer>> *buffers =
      [NSMutableArray arrayWithCapacity:6];
  for (NSUInteger index = 0; index < 6; ++index) {
    id<MTLBuffer> buffer = [plan.device newBufferWithLength:(NSUInteger)lengths[index]
        options:MTLResourceStorageModeShared];
    if (!pixel_buffer(buffer, plan.device, lengths[index])) {
      pixel_set_error(errorText, errorCapacity,
          "query-pair native scratch allocation failed or returned an incompatible buffer");
      return nil;
    }
    for (id<MTLBuffer> resident in plan.resident) {
      if (!pixel_disjoint(buffer, resident)) {
        pixel_set_error(errorText, errorCapacity,
            "query-pair native scratch overlaps an existing plan binding");
        return nil;
      }
    }
    for (id<MTLBuffer> earlier in buffers) {
      if (!pixel_disjoint(buffer, earlier)) {
        pixel_set_error(errorText, errorCapacity,
            "query-pair native scratch allocations overlap");
        return nil;
      }
    }
    [buffers addObject:buffer];
  }
  uint64_t measuredTotal = 0;
  for (id<MTLBuffer> buffer in buffers)
    if (!pixel_query_pair_add(measuredTotal, buffer.length, &measuredTotal)) {
      pixel_set_error(errorText, errorCapacity,
          "query-pair native scratch byte total overflowed");
      return nil;
    }
  if (measuredTotal != totalBytes || qBytes != lengths[0] + lengths[3] ||
      scoreBytes != lengths[1] + lengths[2] ||
      tableBytes != lengths[4] + lengths[5]) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair native scratch byte accounting differs from buffer lengths");
    return nil;
  }
  if (!pixel_query_pair_build_tables(plan, buffers[4], buffers[5],
                                     errorText, errorCapacity)) return nil;

  NSMutableArray<id<MTLBuffer>> *candidateScratch =
      [plan.scratch mutableCopy];
  const NSUInteger slotMap[6] = {0, 3, 4, 5, 6, 7};
  for (NSUInteger index = 0; index < 6; ++index)
    candidateScratch[slotMap[index]] = buffers[index];

  NSMutableArray<PixelUMMAttentionGeometry *> *geometries =
      [NSMutableArray arrayWithCapacity:2];
  NSMutableArray<NSNumber *> *indices = [NSMutableArray arrayWithCapacity:6];
  const int32_t *queryTable = (const int32_t *)buffers[4].contents;
  const int32_t *keyTable = (const int32_t *)buffers[5].contents;
  for (uint32_t pair = 0; pair < 6; ++pair) {
    const uint32_t queryCount = uint32_t(queryTable[uint64_t(pair) * 1281u]);
    const uint32_t keyCount = uint32_t(keyTable[uint64_t(pair) * 4321u]);
    NSUInteger geometryIndex = NSNotFound;
    for (NSUInteger index = 0; index < geometries.count; ++index) {
      PixelUMMAttentionGeometry *geometry = geometries[index];
      if (geometry.queryCount == queryCount && geometry.keyCount == keyCount) {
        geometryIndex = index;
        break;
      }
    }
    if (geometryIndex == NSNotFound) {
      PixelUMMAttentionGeometry *geometry = [PixelUMMAttentionGeometry new];
      geometry.queryCount = queryCount;
      geometry.keyCount = keyCount;
      NSString *qkName = [NSString stringWithFormat:
          @"query_pair_qk_q%u_k%u", queryCount, keyCount];
      NSString *softmaxName = [NSString stringWithFormat:
          @"query_pair_softmax_q%u_k%u", queryCount, keyCount];
      NSString *pvName = [NSString stringWithFormat:
          @"query_pair_pv_q%u_k%u", queryCount, keyCount];
      geometry.qk = pixel_attention_qk_step(plan.device, candidateScratch[0],
          candidateScratch[1], candidateScratch[3], queryCount, keyCount,
          qkName, errorText, errorCapacity);
      geometry.softmax = geometry.qk ? pixel_attention_softmax_step(plan.device,
          candidateScratch[3], candidateScratch[4], queryCount, keyCount,
          softmaxName, errorText, errorCapacity) : nil;
      geometry.pv = geometry.softmax ? pixel_attention_pv_step(plan.device,
          candidateScratch[4], candidateScratch[2], candidateScratch[5],
          queryCount, keyCount, pvName, errorText, errorCapacity) : nil;
      if (!geometry.qk || !geometry.softmax || !geometry.pv) return nil;
      [geometries addObject:geometry];
      geometryIndex = geometries.count - 1u;
    }
    [indices addObject:@(geometryIndex)];
  }
  if (geometries.count == 0 || geometries.count > 2) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair precompiled unexpected attention geometry count");
    return nil;
  }
  PixelUMMQueryPairPool *pool = [PixelUMMQueryPairPool new];
  pool.device = plan.device;
  pool.baselineScratch = plan.scratch;
  pool.extraBuffers = [buffers copy];
  pool.geometries = [geometries copy];
  pool.tileGeometryIndices = [indices copy];
  pool.extraBytes = measuredTotal;
  return pool;
}

static bool pixel_query_pair_state_matches(PixelUMMQueryPair *wrapper,
                                           BOOL candidate) {
  PixelUMMDecoderPlan *plan = wrapper.plan;
  if (!plan) return false;
  return plan.scratch == (candidate ? wrapper.candidateScratch : wrapper.baselineScratch) &&
      plan.resident == (candidate ? wrapper.candidateResident : wrapper.baselineResident) &&
      plan.attentionGeometries == (candidate ? wrapper.pool.geometries : wrapper.baselineGeometries) &&
      plan.tileGeometryIndices == (candidate ? wrapper.pool.tileGeometryIndices : wrapper.baselineTileGeometryIndices) &&
      plan.maxQuery == (candidate ? 1280u : wrapper.baselineMaxQuery) &&
      plan.maxKey == wrapper.baselineMaxKey &&
      plan.tileCount == (candidate ? 6u : wrapper.baselineTileCount);
}

extern "C" void *gs_pixelumm_query_pair_prepare(
    void *planHandle, void *sharedWrapperHandle,
    char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!planHandle) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair preparation requires a native PixelUMM plan");
    return nullptr;
  }
  @autoreleasepool {
    @try {
      PixelUMMDecoderPlan *plan = (__bridge PixelUMMDecoderPlan *)planHandle;
      @synchronized(plan) {
        if (objc_getAssociatedObject(plan, &pixel_query_pair_associated_key)) {
          pixel_set_error(errorText, errorCapacity,
              "query-pair plan already has a diagnostic wrapper");
          return nullptr;
        }
      }
      if (!pixel_query_pair_validate_plan(plan, errorText, errorCapacity))
        return nullptr;

      PixelUMMQueryPairPool *pool = nil;
      if (sharedWrapperHandle) {
        PixelUMMQueryPair *shared =
            (__bridge PixelUMMQueryPair *)sharedWrapperHandle;
        if (!shared.active || !shared.pool || shared.selected || !shared.plan ||
            objc_getAssociatedObject(shared.plan,
                &pixel_query_pair_associated_key) != shared.installationToken) {
          pixel_set_error(errorText, errorCapacity,
              "query-pair shared wrapper is stale, selected or released");
          return nullptr;
        }
        pool = shared.pool;
        if (pool.device != plan.device ||
            pool.baselineScratch.count != plan.scratch.count) {
          pixel_set_error(errorText, errorCapacity,
              "query-pair plan uses a mixed device or scratch pool");
          return nullptr;
        }
        for (NSUInteger index = 0; index < kPixelScratchCount; ++index)
          if (pool.baselineScratch[index] != plan.scratch[index]) {
            pixel_set_error(errorText, errorCapacity,
                "query-pair plan uses different shared baseline scratch buffers");
            return nullptr;
          }
      } else {
        pool = pixel_query_pair_create_pool(plan, errorText, errorCapacity);
        if (!pool) return nullptr;
      }

      NSMutableArray<id<MTLBuffer>> *candidateScratch =
          [plan.scratch mutableCopy];
      const NSUInteger slotMap[6] = {0, 3, 4, 5, 6, 7};
      for (NSUInteger index = 0; index < 6; ++index)
        candidateScratch[slotMap[index]] = pool.extraBuffers[index];
      for (id<MTLBuffer> extra in pool.extraBuffers)
        for (id<MTLBuffer> resident in plan.resident)
          if (!pixel_disjoint(extra, resident)) {
            pixel_set_error(errorText, errorCapacity,
                "query-pair shared native scratch overlaps this plan binding");
            return nullptr;
          }

      PixelUMMQueryPair *wrapper = [PixelUMMQueryPair new];
      wrapper.plan = plan;
      wrapper.pool = pool;
      wrapper.installationToken = [NSObject new];
      wrapper.baselineScratch = plan.scratch;
      wrapper.candidateScratch = [candidateScratch copy];
      wrapper.baselineResident = plan.resident;
      wrapper.candidateResident =
          [plan.resident arrayByAddingObjectsFromArray:pool.extraBuffers];
      wrapper.baselineGeometries = plan.attentionGeometries;
      wrapper.baselineTileGeometryIndices = plan.tileGeometryIndices;
      wrapper.baselineMaxQuery = plan.maxQuery;
      wrapper.baselineMaxKey = plan.maxKey;
      wrapper.baselineTileCount = plan.tileCount;
      wrapper.selected = NO;
      wrapper.active = YES;

      @synchronized(pool) {
        @synchronized(plan) {
          if (pool.wrapperCount >= 4 ||
              objc_getAssociatedObject(plan, &pixel_query_pair_associated_key) ||
              !pixel_query_pair_state_matches(wrapper, NO)) {
            pixel_set_error(errorText, errorCapacity,
                "query-pair prepare lost its idle-plan or four-wrapper pool precondition");
            return nullptr;
          }
          objc_setAssociatedObject(plan, &pixel_query_pair_associated_key,
              wrapper.installationToken, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
          ++pool.wrapperCount;
        }
      }
      return (__bridge_retained void *)wrapper;
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return nullptr;
    }
  }
}

// Call only from the serialized idle owner after its prior command buffer
// drained. No GPU resource or graph is created or read by this selector.
extern "C" int32_t gs_pixelumm_query_pair_select(
    void *handle, int32_t enabled, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!handle || (enabled != 0 && enabled != 1)) {
    pixel_set_error(errorText, errorCapacity,
        "invalid query-pair selection");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMQueryPair *wrapper = (__bridge PixelUMMQueryPair *)handle;
      PixelUMMQueryPairPool *pool = wrapper.pool;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      if (!wrapper.active || !pool || !plan) {
        pixel_set_error(errorText, errorCapacity,
            "query-pair selection requires an active plan wrapper");
        return -2;
      }
      @synchronized(pool) {
        @synchronized(plan) {
          if (!wrapper.active || objc_getAssociatedObject(plan,
                  &pixel_query_pair_associated_key) != wrapper.installationToken ||
              !pixel_query_pair_state_matches(wrapper, wrapper.selected)) {
            pixel_set_error(errorText, errorCapacity,
                "query-pair selection found stale or foreign plan state");
            return -3;
          }
          BOOL candidate = enabled != 0;
          plan.scratch = candidate ? wrapper.candidateScratch : wrapper.baselineScratch;
          plan.resident = candidate ? wrapper.candidateResident : wrapper.baselineResident;
          plan.attentionGeometries = candidate ? pool.geometries : wrapper.baselineGeometries;
          plan.tileGeometryIndices = candidate ? pool.tileGeometryIndices
                                                : wrapper.baselineTileGeometryIndices;
          plan.maxQuery = candidate ? 1280u : wrapper.baselineMaxQuery;
          plan.maxKey = wrapper.baselineMaxKey;
          plan.tileCount = candidate ? 6u : wrapper.baselineTileCount;
          wrapper.selected = candidate;
          return 0;
        }
      }
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return -4;
    }
  }
}

extern "C" uint64_t gs_pixelumm_query_pair_extra_bytes(void *handle) {
  if (!handle) return 0;
  @autoreleasepool {
    PixelUMMQueryPair *wrapper = (__bridge PixelUMMQueryPair *)handle;
    return wrapper.active && wrapper.pool ? wrapper.pool.extraBytes : 0;
  }
}

// Release only after every request using the pool has drained. Restoring the
// exact baseline references precedes removal of the per-plan marker.
extern "C" int32_t gs_pixelumm_query_pair_release(
    void *handle, char *errorText, size_t errorCapacity) {
  if (errorText && errorCapacity) errorText[0] = '\0';
  if (!handle) {
    pixel_set_error(errorText, errorCapacity,
        "query-pair release requires a live wrapper");
    return -1;
  }
  @autoreleasepool {
    @try {
      PixelUMMQueryPair *wrapper = (__bridge PixelUMMQueryPair *)handle;
      PixelUMMQueryPairPool *pool = wrapper.pool;
      PixelUMMDecoderPlan *plan = wrapper.plan;
      if (!wrapper.active || !pool || !plan) {
        pixel_set_error(errorText, errorCapacity,
            "query-pair release requires an active plan wrapper");
        return -2;
      }
      @synchronized(pool) {
        @synchronized(plan) {
          if (objc_getAssociatedObject(plan,
                  &pixel_query_pair_associated_key) != wrapper.installationToken ||
              !pixel_query_pair_state_matches(wrapper, wrapper.selected) ||
              pool.wrapperCount == 0) {
            pixel_set_error(errorText, errorCapacity,
                "query-pair release found stale or foreign plan state");
            return -3;
          }
          plan.scratch = wrapper.baselineScratch;
          plan.resident = wrapper.baselineResident;
          plan.attentionGeometries = wrapper.baselineGeometries;
          plan.tileGeometryIndices = wrapper.baselineTileGeometryIndices;
          plan.maxQuery = wrapper.baselineMaxQuery;
          plan.maxKey = wrapper.baselineMaxKey;
          plan.tileCount = wrapper.baselineTileCount;
          objc_setAssociatedObject(plan, &pixel_query_pair_associated_key,
              nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
          wrapper.selected = NO;
          wrapper.active = NO;
          --pool.wrapperCount;
        }
      }
      (void)CFBridgingRelease(handle);
      return 0;
    } @catch (NSException *exception) {
      pixel_set_ns_error(errorText, errorCapacity, exception.reason);
      return -4;
    }
  }
}
