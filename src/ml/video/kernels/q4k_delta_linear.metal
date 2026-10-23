#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#pragma clang fp contract(off)

constant uint Q4KL_BLOCK_ELEMENTS = 256u;
constant uint Q4KL_BLOCK_BYTES = 144u;
constant uint Q4KL_TM = 64u;
constant uint Q4KL_TN = 32u;
constant uint Q4KL_TK = 32u;
constant uint Q4KL_PATCH_SLOTS = 8u;
constant uint Q4KL_ERROR = 1u;

static inline ushort q4kl_load_u16_le(device const uchar *src) {
  return ushort(src[0]) | (ushort(src[1]) << 8u);
}

static inline float q4kl_load_f16_le(device const uchar *src) {
  return float(as_type<half>(q4kl_load_u16_le(src)));
}

static inline uchar2 q4kl_get_scale_min(uint index, device const uchar *scales) {
  if (index < 4u)
    return uchar2(scales[index] & 63u, scales[index + 4u] & 63u);
  return uchar2(
      (scales[index + 4u] & 0x0fu) | ((scales[index - 4u] >> 6u) << 4u),
      (scales[index + 4u] >> 4u) | ((scales[index] >> 6u) << 4u));
}

// Keep the canonical Q4_K arithmetic sequence from q4k_delta_weights.metal.
static inline float q4kl_decode_base(device const uchar *block, uint within) {
  const float d = q4kl_load_f16_le(block);
  const float dmin = q4kl_load_f16_le(block + 2u);
  device const uchar *scales = block + 4u;
  device const uchar *qs = block + 16u;

  const uint group = within / 64u;
  const uint rem = within - group * 64u;
  const uint scale_index = group * 2u + (rem >= 32u ? 1u : 0u);
  const uchar2 scale_min = q4kl_get_scale_min(scale_index, scales);
  const uchar packed_q = qs[group * 32u + (rem & 31u)];
  const uint q = rem < 32u ? packed_q & 0x0fu : packed_q >> 4u;

  const float d_scale = d * float(scale_min.x);
  const float dmin_min = dmin * float(scale_min.y);
  return d_scale * float(q) - dmin_min;
}

static inline void q4kl_set_error(device atomic_uint *status) {
  atomic_fetch_or_explicit(status, Q4KL_ERROR, memory_order_relaxed);
}

// Shared coefficient semantics for the fused GEMM and the diagnostic probe:
// the canonical base decode and half-rounded delta are independently rounded
// to F32 before the final addition. Invalid corrections invalidate the whole
// block; an invalid base or sum invalidates only this coefficient.
static inline float q4kl_decode_coefficient(
    device const uchar *block, uint within, float delta,
    bool corrected_format, bool malformed, device atomic_uint *status) {
  if (malformed)
    return 0.0f;

  const float base = q4kl_decode_base(block, within);
  if (!isfinite(base)) {
    q4kl_set_error(status);
    return 0.0f;
  }

  // The legacy corrected path adds an F32 zero for unmatched indices too;
  // preserving that operation retains its signed-zero behavior. Patch-count
  // zero is a distinct base-only mode and skips the addition.
  const float value = corrected_format ? base + delta : base;
  if (!isfinite(value)) {
    q4kl_set_error(status);
    return 0.0f;
  }
  return value;
}

// One lane owns each output channel's metadata. This runs once per 256-K block
// and output tile, then the staged records serve all eight K=32 panels.
static inline void q4kl_stage_corrections(
    device const uchar *sidecar, device atomic_uint *status,
    uint channel, uint blocks_per_channel, uint block_in_channel,
    uint patches, uint local_channel, threadgroup uint *indices,
    threadgroup float *deltas, threadgroup uint *malformed) {
  const uint slot_base = local_channel * Q4KL_PATCH_SLOTS;
  for (uint rank = 0u; rank < Q4KL_PATCH_SLOTS; ++rank) {
    indices[slot_base + rank] = 0xffffffffu;
    deltas[slot_base + rank] = 0.0f;
  }
  malformed[local_channel] = 0u;
  if (patches == 0u)
    return;

  const uint flat_block = channel * blocks_per_channel + block_in_channel;
  device const uchar *records = sidecar + flat_block * patches * 3u;
  bool invalid = false;
  for (uint rank = 0u; rank < patches; ++rank) {
    const uint offset = rank * 3u;
    const uint index = uint(records[offset]);
    const float delta = q4kl_load_f16_le(records + offset + 1u);
    indices[slot_base + rank] = index;
    deltas[slot_base + rank] = delta;
    if (!isfinite(delta))
      invalid = true;
    for (uint prior = 0u; prior < rank; ++prior) {
      if (indices[slot_base + prior] == index)
        invalid = true;
    }
  }
  if (invalid) {
    malformed[local_channel] = 1u;
    q4kl_set_error(status);
  }
}

template <bool Q4K>
static inline void q4kl_run_gemm(
    device const float *input, device const float *dense_weight,
    device const uchar *payload, device const uchar *sidecar,
    device const float *bias, device float *output,
    device atomic_uint *status, uint rows, uint in_dim, uint out_dim,
    uint patches, uint2 group, uint tid, uint simdgroup,
    uint threads, threadgroup float *a_panel,
    threadgroup float *b_panel, threadgroup uint *patch_indices,
    threadgroup float *patch_deltas, threadgroup uint *block_malformed) {
  if (threads != 128u)
    return;

  const uint row_begin = group.y * Q4KL_TM;
  const uint col_begin = group.x * Q4KL_TN;
  constexpr uint ROW_BLOCKS = Q4KL_TM / 16u;
  constexpr uint COL_BLOCKS = Q4KL_TN / 16u;
  simdgroup_float8x8 accum[ROW_BLOCKS * COL_BLOCKS];
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < ROW_BLOCKS * COL_BLOCKS; ++i) {
    accum[i].thread_elements()[0] = 0.0f;
    accum[i].thread_elements()[1] = 0.0f;
  }

  const uint sg_row = (simdgroup & 1u) * (Q4KL_TM / 2u);
  const uint sg_col = (simdgroup >> 1u) * (Q4KL_TN / 2u);
  const uint blocks_per_channel = in_dim / Q4KL_BLOCK_ELEMENTS;
  for (uint k0 = 0u; k0 < in_dim; k0 += Q4KL_TK) {
    if (Q4K && (k0 % Q4KL_BLOCK_ELEMENTS) == 0u) {
      for (uint local_n = tid; local_n < Q4KL_TN; local_n += 128u) {
        const uint channel = col_begin + local_n;
        if (channel < out_dim) {
          q4kl_stage_corrections(sidecar, status, channel, blocks_per_channel,
              k0 / Q4KL_BLOCK_ELEMENTS, patches, local_n, patch_indices,
              patch_deltas, block_malformed);
        } else {
          const uint slot_base = local_n * Q4KL_PATCH_SLOTS;
          for (uint rank = 0u; rank < Q4KL_PATCH_SLOTS; ++rank) {
            patch_indices[slot_base + rank] = 0xffffffffu;
            patch_deltas[slot_base + rank] = 0.0f;
          }
          block_malformed[local_n] = 0u;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Read contiguous K values from row-major [N,K] weights, then transpose
    // into the SIMD group's logical B[K,N] panel. This keeps global reads
    // coalesced for both dense F32 values and Q4_K's packed K stream.
    for (uint i = tid; i < Q4KL_TN * Q4KL_TK; i += 128u) {
      const uint local_n = i / Q4KL_TK;
      const uint local_k = i % Q4KL_TK;
      const uint channel = col_begin + local_n;
      const uint k = k0 + local_k;
      float value = 0.0f;
      if (channel < out_dim) {
        if (Q4K) {
          const uint block_in_channel = k / Q4KL_BLOCK_ELEMENTS;
          const uint flat_block = channel * blocks_per_channel + block_in_channel;
          const uint slot_base = local_n * Q4KL_PATCH_SLOTS;
          float delta = 0.0f;
          for (uint rank = 0u; rank < patches; ++rank) {
            if (patch_indices[slot_base + rank] == (k % Q4KL_BLOCK_ELEMENTS)) {
              delta = patch_deltas[slot_base + rank];
              break;
            }
          }
          value = q4kl_decode_coefficient(payload + flat_block * Q4KL_BLOCK_BYTES,
              k % Q4KL_BLOCK_ELEMENTS, delta, patches != 0u,
              block_malformed[local_n] != 0u, status);
        } else {
          value = dense_weight[channel * in_dim + k];
        }
      }
      b_panel[local_k * Q4KL_TN + local_n] = value;
    }

    for (uint i = tid; i < Q4KL_TM * Q4KL_TK; i += 128u) {
      const uint local_m = i / Q4KL_TK;
      const uint local_k = i % Q4KL_TK;
      const uint m = row_begin + local_m;
      const uint k = k0 + local_k;
      a_panel[i] = m < rows ? input[m * in_dim + k] : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint kfrag = 0u; kfrag < Q4KL_TK; kfrag += 8u) {
      #pragma clang loop unroll(full)
      for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
        #pragma clang loop unroll(full)
        for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
          simdgroup_float8x8 a_fragment;
          simdgroup_float8x8 b_fragment;
          simdgroup_load(a_fragment,
              a_panel + (sg_row + rb * 8u) * Q4KL_TK + kfrag, Q4KL_TK);
          simdgroup_load(b_fragment,
              b_panel + kfrag * Q4KL_TN + sg_col + cb * 8u, Q4KL_TN);
          const uint acc_index = rb * COL_BLOCKS + cb;
          simdgroup_multiply_accumulate(
              accum[acc_index], a_fragment, b_fragment, accum[acc_index]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma clang loop unroll(full)
  for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
    #pragma clang loop unroll(full)
    for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
      const uint acc_index = rb * COL_BLOCKS + cb;
      simdgroup_store(accum[acc_index],
          a_panel + (sg_row + rb * 8u) * Q4KL_TN + sg_col + cb * 8u,
          Q4KL_TN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < Q4KL_TM * Q4KL_TN; i += 128u) {
    const uint m = row_begin + i / Q4KL_TN;
    const uint n = col_begin + i % Q4KL_TN;
    if (m < rows && n < out_dim)
      output[m * out_dim + n] = a_panel[i] + bias[n];
  }
}

kernel void video_q4k_delta_linear_f32(
    device const float *input [[buffer(0)]],
    device const uchar *payload [[buffer(1)]],
    device const uchar *sidecar [[buffer(2)]],
    device const float *bias [[buffer(3)]],
    device float *output [[buffer(4)]],
    device atomic_uint *status [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &in_dim [[buffer(7)]],
    constant uint &out_dim [[buffer(8)]],
    constant uint &patches [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float a_panel[Q4KL_TM * Q4KL_TK];
  threadgroup float b_panel[Q4KL_TK * Q4KL_TN];
  threadgroup uint patch_indices[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_TN];
  q4kl_run_gemm<true>(input, nullptr, payload, sidecar, bias, output, status,
      rows, in_dim, out_dim, patches, group, tid, simdgroup,
      threads.x * threads.y, a_panel, b_panel, patch_indices, patch_deltas,
      block_malformed);
}

kernel void video_q4k_delta_dense_linear_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &in_dim [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float a_panel[Q4KL_TM * Q4KL_TK];
  threadgroup float b_panel[Q4KL_TK * Q4KL_TN];
  threadgroup uint patch_indices[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_TN];
  q4kl_run_gemm<false>(input, weight, nullptr, nullptr, bias, output, nullptr,
      rows, in_dim, out_dim, 0u, group, tid, simdgroup,
      threads.x * threads.y, a_panel, b_panel, patch_indices, patch_deltas,
      block_malformed);
}

kernel void video_q4k_delta_decode_probe(
    device const uchar *payload [[buffer(0)]],
    device const uchar *sidecar [[buffer(1)]],
    device float *destination [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &patches [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;

  const uint block_index = gid / Q4KL_BLOCK_ELEMENTS;
  const uint within = gid % Q4KL_BLOCK_ELEMENTS;
  uint indices[Q4KL_PATCH_SLOTS];
  float deltas[Q4KL_PATCH_SLOTS];
  bool malformed = false;
  float delta = 0.0f;

  if (patches != 0u) {
    device const uchar *records = sidecar + block_index * patches * 3u;
    for (uint rank = 0u; rank < patches; ++rank) {
      const uint offset = rank * 3u;
      indices[rank] = uint(records[offset]);
      deltas[rank] = q4kl_load_f16_le(records + offset + 1u);
      if (!isfinite(deltas[rank]))
        malformed = true;
      for (uint prior = 0u; prior < rank; ++prior) {
        if (indices[prior] == indices[rank])
          malformed = true;
      }
      if (indices[rank] == within) {
        delta = deltas[rank];
      }
    }
    if (malformed && within == 0u)
      q4kl_set_error(status);
  }

  const float value = q4kl_decode_coefficient(
      payload + block_index * Q4KL_BLOCK_BYTES, within, delta, patches != 0u,
      malformed, status);
  destination[gid] = value;
}
