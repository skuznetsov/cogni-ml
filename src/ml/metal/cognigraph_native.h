#pragma once

// Prepared native backend for CogniGraph's shared dependency compiler. All
// recipes and waves are immutable before encoding. This backend borrows a
// current Metal root; it never submits, waits, reads tensors or creates buffers.
#import <Metal/Metal.h>
#include "cognigraph_schedule.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace cogni {

struct NativeGraphBinding {
  id<MTLBuffer> __strong buffer;
  uint32_t index;
  uint64_t offset;
  int64_t length;
  int32_t access;
};

struct NativeGraphOp {
  id<MTLComputePipelineState> __strong pipeline = nil;
  std::vector<NativeGraphBinding> buffers;
  std::vector<uint8_t> parameters;
  uint32_t parameterIndex = 0;
  MTLSize grid = MTLSizeMake(0, 1, 1);
  MTLSize threads = MTLSizeMake(0, 1, 1);
  bool threadgroups = false;
};

class PreparedNativeGraph {
public:
  bool prepare(id<MTLDevice> device, std::vector<NativeGraphOp> ops,
               GsCogniGraphCompile compiler) {
    if (prepared_ || !device || !compiler || ops.empty() || ops.size() > 256)
      return false;
    std::vector<GsCogniGraphAccess> accesses;
    std::vector<uint32_t> offsets{0};
    for (const auto &op : ops) {
      if (!op.pipeline || op.pipeline.device != device || op.buffers.empty() ||
          op.parameters.empty() || op.parameters.size() > 4096 ||
          op.parameterIndex >= 31 || !op.grid.width || !op.grid.height ||
          !op.grid.depth || !op.threads.width || !op.threads.height ||
          !op.threads.depth || op.threads.width * op.threads.height *
              op.threads.depth > op.pipeline.maxTotalThreadsPerThreadgroup)
        return false;
      for (const auto &binding : op.buffers) {
        if (!binding.buffer || binding.buffer.device != device ||
            binding.index >= 31 || binding.index == op.parameterIndex ||
            binding.offset >= binding.buffer.length || binding.length <= 0 ||
            uint64_t(binding.length) > binding.buffer.length - binding.offset ||
            binding.access < 0 || binding.access > 2)
          return false;
        accesses.push_back({uint64_t((uintptr_t)(__bridge void *)binding.buffer),
            int64_t(binding.offset), binding.length, binding.access, -1});
      }
      offsets.push_back(uint32_t(accesses.size()));
    }
    if (accesses.size() > 2048) return false;
    std::vector<int32_t> waveOf(ops.size(), -1);
    uint32_t waveCount = 0;
    if (compiler(accesses.data(), offsets.data(), uint32_t(ops.size()),
          uint32_t(accesses.size()), waveOf.data(), &waveCount) != 0 ||
        !waveCount || waveCount > ops.size()) return false;
    std::vector<std::vector<uint32_t>> waves(waveCount);
    for (uint32_t index = 0; index < ops.size(); ++index) {
      if (waveOf[index] < 0 || uint32_t(waveOf[index]) >= waveCount) return false;
      waves[waveOf[index]].push_back(index);
    }
    uint32_t maxWidth = 0;
    for (const auto &wave : waves) {
      if (wave.empty()) return false;
      maxWidth = std::max(maxWidth, uint32_t(wave.size()));
    }
    device_ = device;
    ops_ = std::move(ops);
    waves_ = std::move(waves);
    maxWidth_ = maxWidth;
    prepared_ = true;
    return true;
  }

  bool encode(id<MTLCommandBuffer> root) const {
    if (!prepared_ || !root || root.commandQueue.device != device_ ||
        root.status >= MTLCommandBufferStatusCommitted) return false;
    id<MTLComputeCommandEncoder> encoder =
        [root computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    if (!encoder) return false;
    @try {
      for (size_t wi = 0; wi < waves_.size(); ++wi) {
        for (uint32_t index : waves_[wi]) {
          const auto &op = ops_[index];
          [encoder setComputePipelineState:op.pipeline];
          for (const auto &binding : op.buffers)
            [encoder setBuffer:binding.buffer offset:binding.offset atIndex:binding.index];
          [encoder setBytes:op.parameters.data() length:op.parameters.size()
                   atIndex:op.parameterIndex];
          if (op.threadgroups)
            [encoder dispatchThreadgroups:op.grid threadsPerThreadgroup:op.threads];
          else
            [encoder dispatchThreads:op.grid threadsPerThreadgroup:op.threads];
        }
        if (wi + 1 < waves_.size())
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    } @finally {
      [encoder endEncoding];
    }
    return true;
  }

  uint32_t opCount() const { return uint32_t(ops_.size()); }
  uint32_t waveCount() const { return uint32_t(waves_.size()); }
  uint32_t barrierCount() const { return waveCount() ? waveCount() - 1 : 0; }
  uint32_t maxWidth() const { return maxWidth_; }

private:
  id<MTLDevice> __strong device_ = nil;
  std::vector<NativeGraphOp> ops_;
  std::vector<std::vector<uint32_t>> waves_;
  uint32_t maxWidth_ = 0;
  bool prepared_ = false;
};

} // namespace cogni
