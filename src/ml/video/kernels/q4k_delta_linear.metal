#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#pragma clang fp contract(off)

constant uint Q4KL_BLOCK_ELEMENTS = 256u;
constant uint Q4KL_BLOCK_BYTES = 144u;
constant uint Q4KL_TM = 64u;
constant uint Q4KL_TN = 32u;
constant uint Q4KL_TK = 32u;
constant uint Q4KL_CANDIDATE_TM = 128u;
constant uint Q4KL_CANDIDATE_TN = 32u;
constant uint Q4KL_CANDIDATE_TK = 32u;
constant uint Q4KL_BT_K64_CANDIDATE_TK = 64u;
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

// The direct Q4_K path needs to populate a simdgroup matrix fragment from
// packed values without assuming a lane-to-element formula. Seed this shared
// 8x8 tile with logical coordinates, load it through Metal's matrix API, and
// use the fragment's own two elements per thread as the coordinates. This is
// qualified for the compiled backend by the coordinate round-trip probe; it
// is not a portable fixed lane-map promise.
static inline void q4kl_load_coordinate_fragment(
    thread simdgroup_float8x8 &matrix_fragment, threadgroup float *coordinates) {
  simdgroup_load(matrix_fragment, coordinates, 8u);
}

// The direct core currently reloads these 64 logical coordinates for each
// packed B fragment at each K=8 step. This keeps the first implementation
// independent of a fixed lane map; capturing the mapping once per threadgroup
// is a separate optimization that needs its own equivalence check.
static inline void q4kl_load_q4_fragment(
    thread simdgroup_float8x8 &matrix_fragment,
    threadgroup float *coordinates,
    device const uchar *payload, device atomic_uint *status,
    uint in_dim, uint col_begin, uint sg_col, uint cb,
    uint k0, uint kfrag, uint patches,
    threadgroup uint *patch_indices, threadgroup float *patch_deltas,
    threadgroup uint *block_malformed) {
  q4kl_load_coordinate_fragment(matrix_fragment, coordinates);
  const uint blocks_per_channel = in_dim / Q4KL_BLOCK_ELEMENTS;
  #pragma clang loop unroll(full)
  for (uint element = 0u; element < 2u; ++element) {
    const uint coordinate = uint(matrix_fragment.thread_elements()[element]);
    const uint local_k = coordinate >> 3u;
    const uint local_n = coordinate & 7u;
    const uint tile_n = sg_col + cb * 8u + local_n;
    const uint channel = col_begin + tile_n;
    const uint k = k0 + kfrag + local_k;
    const uint block_in_channel = k / Q4KL_BLOCK_ELEMENTS;
    const uint within = k % Q4KL_BLOCK_ELEMENTS;
    const uint flat_block = channel * blocks_per_channel + block_in_channel;
    float delta = 0.0f;
    for (uint rank = 0u; rank < patches; ++rank) {
      const uint slot = tile_n * Q4KL_PATCH_SLOTS + rank;
      if (patch_indices[slot] == within) {
        delta = patch_deltas[slot];
        break;
      }
    }
    const bool malformed = patches != 0u && block_malformed[tile_n] != 0u;
    matrix_fragment.thread_elements()[element] = q4kl_decode_coefficient(
        payload + flat_block * Q4KL_BLOCK_BYTES, within, delta,
        patches != 0u, malformed, status);
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

// Opt-in fragment-reuse schedule. It keeps the same 2x2 SIMD-group mapping
// and F32 8x8 multiply-accumulate sequence as the legacy kernel. B fragments
// are loaded once per ascending K fragment; one A fragment at a time is
// reused across both column blocks to keep fragment register lifetimes short.
// TRANSPOSED_B stores each staged [N,K] row contiguously and loads it as B[K,N].
// Existing variants keep WIDE_K=false; TK64 changes only the panel grouping.
template <bool Q4K, bool WIDE_M, bool WIDE_K, bool TRANSPOSED_B>
static inline void q4kl_run_gemm_reuse(
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

  constexpr uint TM = WIDE_M ? Q4KL_CANDIDATE_TM : Q4KL_TM;
  constexpr uint TK = WIDE_K ? Q4KL_BT_K64_CANDIDATE_TK : Q4KL_CANDIDATE_TK;
  static_assert(TM * TK >= TM * Q4KL_CANDIDATE_TN,
      "A panel must hold the output tile used by the F32 epilogue");
  const uint row_begin = group.y * TM;
  const uint col_begin = group.x * Q4KL_CANDIDATE_TN;
  constexpr uint ROW_BLOCKS = TM / 16u;
  constexpr uint COL_BLOCKS = Q4KL_CANDIDATE_TN / 16u;
  simdgroup_float8x8 accum[ROW_BLOCKS * COL_BLOCKS];
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < ROW_BLOCKS * COL_BLOCKS; ++i) {
    accum[i].thread_elements()[0] = 0.0f;
    accum[i].thread_elements()[1] = 0.0f;
  }

  const uint sg_row = (simdgroup & 1u) * (TM / 2u);
  const uint sg_col = (simdgroup >> 1u) * (Q4KL_CANDIDATE_TN / 2u);
  const uint blocks_per_channel = in_dim / Q4KL_BLOCK_ELEMENTS;
  simdgroup_float8x8 a_fragment;
  simdgroup_float8x8 b_fragments[COL_BLOCKS];
  for (uint k0 = 0u; k0 < in_dim; k0 += TK) {
    if (Q4K && (k0 % Q4KL_BLOCK_ELEMENTS) == 0u) {
      for (uint local_n = tid; local_n < Q4KL_CANDIDATE_TN; local_n += 128u) {
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

    // Read each tile once. TRANSPOSED_B keeps [N,K] rows contiguous for its
    // transposed SIMD-group load; other variants store [K,N].
    for (uint i = tid; i < Q4KL_CANDIDATE_TN * TK; i += 128u) {
      const uint local_n = i / TK;
      const uint local_k = i % TK;
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
      const uint b_index = TRANSPOSED_B
          ? local_n * TK + local_k
          : local_k * Q4KL_CANDIDATE_TN + local_n;
      b_panel[b_index] = value;
    }

    for (uint i = tid; i < TM * TK; i += 128u) {
      const uint local_m = i / TK;
      const uint local_k = i % TK;
      const uint m = row_begin + local_m;
      const uint k = k0 + local_k;
      a_panel[i] = m < rows ? input[m * in_dim + k] : 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint kfrag = 0u; kfrag < TK; kfrag += 8u) {
      #pragma clang loop unroll(full)
      for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
        if (TRANSPOSED_B) {
          simdgroup_load(b_fragments[cb],
              b_panel + (sg_col + cb * 8u) * TK + kfrag,
              TK, 0, true);
        } else {
          simdgroup_load(b_fragments[cb],
              b_panel + kfrag * Q4KL_CANDIDATE_TN + sg_col + cb * 8u,
              Q4KL_CANDIDATE_TN);
        }
      }
      #pragma clang loop unroll(full)
      for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
        simdgroup_load(a_fragment,
            a_panel + (sg_row + rb * 8u) * TK + kfrag,
            TK);
        #pragma clang loop unroll(full)
        for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
          const uint acc_index = rb * COL_BLOCKS + cb;
          simdgroup_multiply_accumulate(accum[acc_index], a_fragment,
              b_fragments[cb], accum[acc_index]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // The A panel has at least TM*TN elements, so its prefix can hold the output
  // tile for the F32 bias epilogue without extra shared memory.
  #pragma clang loop unroll(full)
  for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
    #pragma clang loop unroll(full)
    for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
      const uint acc_index = rb * COL_BLOCKS + cb;
      simdgroup_store(accum[acc_index],
          a_panel + (sg_row + rb * 8u) * Q4KL_CANDIDATE_TN + sg_col + cb * 8u,
          Q4KL_CANDIDATE_TN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < TM * Q4KL_CANDIDATE_TN; i += 128u) {
    const uint m = row_begin + i / Q4KL_CANDIDATE_TN;
    const uint n = col_begin + i % Q4KL_CANDIDATE_TN;
    if (m < rows && n < out_dim)
      output[m * out_dim + n] = a_panel[i] + bias[n];
  }
}

// Dense-only full-tile discriminator. The direct grid contains complete
// 64x32 output tiles, and K is validated as a multiple of 256 by the host, so
// every 8x8 matrix load stays in bounds. Input and row-major [N,K] weights
// are loaded directly from device memory; only the output tile uses
// threadgroup storage for the bias epilogue. This removes operand panels and
// their per-K threadgroup barriers while keeping the B64/32 F32 MAC order.
static inline void q4kl_run_gemm_device_direct(
    device const float *input, device const float *weight,
    device const float *bias, device float *output,
    uint rows, uint in_dim, uint out_dim, uint2 group, uint tid,
    uint simdgroup, uint threads, threadgroup float *output_panel) {
  if (threads != 128u)
    return;

  const uint row_begin = group.y * Q4KL_TM;
  const uint col_begin = group.x * Q4KL_TN;
  if (row_begin + Q4KL_TM > rows || col_begin + Q4KL_TN > out_dim)
    return;
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
  simdgroup_float8x8 a_fragment;
  simdgroup_float8x8 b_fragments[COL_BLOCKS];
  for (uint k0 = 0u; k0 < in_dim; k0 += Q4KL_TK) {
    #pragma clang loop unroll(full)
    for (uint kfrag = 0u; kfrag < Q4KL_TK; kfrag += 8u) {
      #pragma clang loop unroll(full)
      for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
        const uint n = col_begin + sg_col + cb * 8u;
        simdgroup_load(b_fragments[cb],
            weight + n * in_dim + k0 + kfrag, in_dim, 0, true);
      }
      #pragma clang loop unroll(full)
      for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
        const uint m = row_begin + sg_row + rb * 8u;
        simdgroup_load(a_fragment,
            input + m * in_dim + k0 + kfrag, in_dim);
        #pragma clang loop unroll(full)
        for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
          const uint acc_index = rb * COL_BLOCKS + cb;
          simdgroup_multiply_accumulate(accum[acc_index], a_fragment,
              b_fragments[cb], accum[acc_index]);
        }
      }
    }
  }

  #pragma clang loop unroll(full)
  for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
    #pragma clang loop unroll(full)
    for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
      const uint acc_index = rb * COL_BLOCKS + cb;
      simdgroup_store(accum[acc_index],
          output_panel + (sg_row + rb * 8u) * Q4KL_TN + sg_col + cb * 8u,
          Q4KL_TN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < Q4KL_TM * Q4KL_TN; i += 128u) {
    const uint m = row_begin + i / Q4KL_TN;
    const uint n = col_begin + i % Q4KL_TN;
    output[m * out_dim + n] = output_panel[i] + bias[n];
  }
}

// Packed direct core: each full 64x32 output tile streams A from device memory
// and decodes B into simdgroup fragments using the probed coordinate mapping.
// Correction records are staged once per 256-K Q4_K block and reused across
// its 32 ascending K=8 matrix operations. Output scratch initially holds
// coordinates, then becomes the bias epilogue tile after the final MAC.
static inline void q4kl_run_gemm_device_q4k_direct(
    device const float *input, device const uchar *payload,
    device const uchar *sidecar, device const float *bias,
    device float *output, device atomic_uint *status,
    uint rows, uint in_dim, uint out_dim, uint patches, uint2 group,
    uint tid, uint simdgroup, uint threads, threadgroup float *output_panel,
    threadgroup uint *patch_indices, threadgroup float *patch_deltas,
    threadgroup uint *block_malformed) {
  if (threads != 128u)
    return;

  const uint row_begin = group.y * Q4KL_TM;
  const uint col_begin = group.x * Q4KL_TN;
  if (row_begin + Q4KL_TM > rows || col_begin + Q4KL_TN > out_dim)
    return;

  for (uint i = tid; i < 64u; i += 128u)
    output_panel[i] = float(i);
  threadgroup_barrier(mem_flags::mem_threadgroup);

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
  simdgroup_float8x8 a_fragment;
  simdgroup_float8x8 b_fragments[COL_BLOCKS];
  const uint blocks_per_channel = in_dim / Q4KL_BLOCK_ELEMENTS;
  for (uint k0 = 0u; k0 < in_dim; k0 += Q4KL_TK) {
    if (patches != 0u && (k0 % Q4KL_BLOCK_ELEMENTS) == 0u) {
      if (k0 != 0u)
        threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint local_n = tid; local_n < Q4KL_TN; local_n += 128u) {
        q4kl_stage_corrections(sidecar, status, col_begin + local_n,
            blocks_per_channel, k0 / Q4KL_BLOCK_ELEMENTS, patches, local_n,
            patch_indices, patch_deltas, block_malformed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint kfrag = 0u; kfrag < Q4KL_TK; kfrag += 8u) {
      #pragma clang loop unroll(full)
      for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
        q4kl_load_q4_fragment(b_fragments[cb], output_panel, payload, status,
            in_dim, col_begin, sg_col, cb, k0, kfrag, patches,
            patch_indices, patch_deltas, block_malformed);
      }
      #pragma clang loop unroll(full)
      for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
        const uint m = row_begin + sg_row + rb * 8u;
        simdgroup_load(a_fragment,
            input + m * in_dim + k0 + kfrag, in_dim);
        #pragma clang loop unroll(full)
        for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
          const uint acc_index = rb * COL_BLOCKS + cb;
          simdgroup_multiply_accumulate(accum[acc_index], a_fragment,
              b_fragments[cb], accum[acc_index]);
        }
      }
    }
  }

  // Prevent any simdgroup from reading coordinate scratch while another one
  // starts storing the final output tile into the same memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
    #pragma clang loop unroll(full)
    for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
      const uint acc_index = rb * COL_BLOCKS + cb;
      simdgroup_store(accum[acc_index],
          output_panel + (sg_row + rb * 8u) * Q4KL_TN + sg_col + cb * 8u,
          Q4KL_TN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < Q4KL_TM * Q4KL_TN; i += 128u) {
    const uint m = row_begin + i / Q4KL_TN;
    const uint n = col_begin + i % Q4KL_TN;
    output[m * out_dim + n] = output_panel[i] + bias[n];
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

kernel void video_q4k_delta_linear_f32_candidate_b128n32_sg4_tk32(
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
  threadgroup float a_panel[Q4KL_CANDIDATE_TM * Q4KL_CANDIDATE_TK];
  threadgroup float b_panel[Q4KL_CANDIDATE_TK * Q4KL_CANDIDATE_TN];
  threadgroup uint patch_indices[Q4KL_CANDIDATE_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_CANDIDATE_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_CANDIDATE_TN];
  q4kl_run_gemm_reuse<true, true, false, false>(input, nullptr,
      payload, sidecar, bias, output,
      status, rows, in_dim, out_dim, patches, group, tid, simdgroup,
      threads.x * threads.y, a_panel, b_panel, patch_indices, patch_deltas,
      block_malformed);
}

kernel void video_q4k_delta_dense_linear_f32_candidate_b128n32_sg4_tk32(
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
  threadgroup float a_panel[Q4KL_CANDIDATE_TM * Q4KL_CANDIDATE_TK];
  threadgroup float b_panel[Q4KL_CANDIDATE_TK * Q4KL_CANDIDATE_TN];
  threadgroup uint patch_indices[Q4KL_CANDIDATE_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_CANDIDATE_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_CANDIDATE_TN];
  q4kl_run_gemm_reuse<false, true, false, false>(input, weight, nullptr,
      nullptr, bias, output,
      nullptr, rows, in_dim, out_dim, 0u, group, tid, simdgroup,
      threads.x * threads.y, a_panel, b_panel, patch_indices, patch_deltas,
      block_malformed);
}

kernel void video_q4k_delta_linear_f32_candidate_b64n32_sg4_tk32_bt(
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
  q4kl_run_gemm_reuse<true, false, false, true>(input, nullptr, payload, sidecar,
      bias, output, status, rows, in_dim, out_dim, patches, group, tid,
      simdgroup, threads.x * threads.y, a_panel, b_panel, patch_indices,
      patch_deltas, block_malformed);
}

kernel void video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_tk32_bt(
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
  q4kl_run_gemm_reuse<false, false, false, true>(input, weight, nullptr, nullptr,
      bias, output, nullptr, rows, in_dim, out_dim, 0u, group, tid, simdgroup,
      threads.x * threads.y, a_panel, b_panel, patch_indices, patch_deltas,
      block_malformed);
}

kernel void video_q4k_delta_linear_f32_candidate_b64n32_sg4_tk64_bt(
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
  threadgroup float a_panel[Q4KL_TM * Q4KL_BT_K64_CANDIDATE_TK];
  threadgroup float b_panel[Q4KL_CANDIDATE_TN * Q4KL_BT_K64_CANDIDATE_TK];
  threadgroup uint patch_indices[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_TN];
  q4kl_run_gemm_reuse<true, false, true, true>(input, nullptr, payload, sidecar,
      bias, output, status, rows, in_dim, out_dim, patches, group, tid,
      simdgroup, threads.x * threads.y, a_panel, b_panel, patch_indices,
      patch_deltas, block_malformed);
}

kernel void video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_tk64_bt(
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
  threadgroup float a_panel[Q4KL_TM * Q4KL_BT_K64_CANDIDATE_TK];
  threadgroup float b_panel[Q4KL_CANDIDATE_TN * Q4KL_BT_K64_CANDIDATE_TK];
  threadgroup uint patch_indices[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_TN];
  q4kl_run_gemm_reuse<false, false, true, true>(input, weight, nullptr, nullptr,
      bias, output, nullptr, rows, in_dim, out_dim, 0u, group, tid, simdgroup,
      threads.x * threads.y, a_panel, b_panel, patch_indices, patch_deltas,
      block_malformed);
}

kernel void video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_device_f32(
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
  threadgroup float output_panel[Q4KL_TM * Q4KL_TN];
  q4kl_run_gemm_device_direct(input, weight, bias, output, rows, in_dim,
      out_dim, group, tid, simdgroup, threads.x * threads.y, output_panel);
}

// The direct kernel only covers complete tiles. This staging pipeline handles
// the disjoint right and bottom fringes by offsetting its local group origin.
kernel void video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_device_f32_tail(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &in_dim [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &group_base_x [[buffer(7)]],
    constant uint &group_base_y [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float a_panel[Q4KL_TM * Q4KL_TK];
  threadgroup float b_panel[Q4KL_TK * Q4KL_TN];
  threadgroup uint patch_indices[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_TN];
  const uint2 tile_group = group + uint2(group_base_x, group_base_y);
  q4kl_run_gemm_reuse<false, false, false, true>(input, weight, nullptr, nullptr,
      bias, output, nullptr, rows, in_dim, out_dim, 0u, tile_group, tid,
      simdgroup, threads.x * threads.y, a_panel, b_panel, patch_indices,
      patch_deltas, block_malformed);
}

kernel void video_q4k_delta_linear_f32_candidate_b64n32_sg4_device_q4k(
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
  threadgroup float output_panel[Q4KL_TM * Q4KL_TN];
  threadgroup uint patch_indices[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_TN];
  q4kl_run_gemm_device_q4k_direct(input, payload, sidecar, bias, output,
      status, rows, in_dim, out_dim, patches, group, tid, simdgroup,
      threads.x * threads.y, output_panel, patch_indices, patch_deltas,
      block_malformed);
}

// The complete-tile path falls back to the established staged Q4_K loader for
// right and bottom fringes. Group bases keep these rectangles disjoint.
kernel void video_q4k_delta_linear_f32_candidate_b64n32_sg4_device_q4k_tail(
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
    constant uint &group_base_x [[buffer(10)]],
    constant uint &group_base_y [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float a_panel[Q4KL_TM * Q4KL_TK];
  threadgroup float b_panel[Q4KL_TN * Q4KL_TK];
  threadgroup uint patch_indices[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_TN];
  const uint2 tile_group = group + uint2(group_base_x, group_base_y);
  q4kl_run_gemm_reuse<true, false, false, true>(input, nullptr, payload,
      sidecar, bias, output, status, rows, in_dim, out_dim, patches,
      tile_group, tid, simdgroup, threads.x * threads.y, a_panel, b_panel,
      patch_indices, patch_deltas, block_malformed);
}

// This public pipeline qualification test round-trips an 8x8 coordinate tile
// through the same matrix-load helper as the direct packed core.
kernel void video_q4k_delta_coordinate_mapping_probe(
    device float *destination [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float coordinates[64];
  threadgroup float mapped[256];
  for (uint i = tid; i < 64u; i += 128u)
    coordinates[i] = float(i);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 matrix_fragment;
  q4kl_load_coordinate_fragment(matrix_fragment, coordinates);
  #pragma clang loop unroll(full)
  for (uint element = 0u; element < 2u; ++element) {
    const uint coordinate = uint(matrix_fragment.thread_elements()[element]);
    matrix_fragment.thread_elements()[element] =
        float((coordinate >> 3u) * 100u + (coordinate & 7u));
  }
  simdgroup_store(matrix_fragment, mapped + simdgroup * 64u, 8u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 256u; i += 128u)
    destination[i] = mapped[i];
}

// Materialize two decoded 8x8 Q4_K fragments: K=248..255 and K=256..263,
// for both 16-channel halves and all four SIMD groups. Each SIMD group writes
// disjoint slices for both 8-channel subtiles. The same coordinate-derived
// fragment helper is used by the production direct core and the mapping probe.
kernel void video_q4k_delta_q4_fragment_probe(
    device const uchar *payload [[buffer(0)]],
    device const uchar *sidecar [[buffer(1)]],
    device float *destination [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant uint &patches [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint IN_DIM = 512u;
  constexpr uint OUT_DIM = 32u;
  threadgroup float coordinates[64];
  threadgroup float materialized[512];
  threadgroup uint patch_indices[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup float patch_deltas[Q4KL_TN * Q4KL_PATCH_SLOTS];
  threadgroup uint block_malformed[Q4KL_TN];
  for (uint i = tid; i < 64u; i += 128u)
    coordinates[i] = float(i);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (patches != 0u) {
    for (uint local_n = tid; local_n < OUT_DIM; local_n += 128u) {
      q4kl_stage_corrections(sidecar, status, local_n,
          IN_DIM / Q4KL_BLOCK_ELEMENTS, group.y, patches, local_n,
          patch_indices, patch_deltas, block_malformed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_float8x8 matrix_fragment;
  const uint sg_col = (simdgroup >> 1u) * 16u;
  const uint kfrag = group.y == 0u ? 248u : 0u;
  #pragma clang loop unroll(full)
  for (uint cb = 0u; cb < 2u; ++cb) {
    q4kl_load_q4_fragment(matrix_fragment, coordinates, payload, status,
        IN_DIM, 0u, sg_col, cb,
        group.y * Q4KL_BLOCK_ELEMENTS, kfrag, patches,
        patch_indices, patch_deltas, block_malformed);
    simdgroup_store(matrix_fragment,
        materialized + simdgroup * 128u + cb * 64u, 8u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 512u; i += 128u)
    destination[group.y * 512u + i] = materialized[i];
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
