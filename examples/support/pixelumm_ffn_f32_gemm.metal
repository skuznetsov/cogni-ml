#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Diagnostic-only dense PixelUMM FFN projection. It reads the resident BF16
// operands directly, expands each BF16 bit pattern to its exact F32 value in
// threadgroup panels, accumulates in F32 in ascending K32/K8 order, and rounds
// the final result to BF16. This does not select or alter a production path.
#pragma clang fp contract(off)

constant uint PIXELUMM_FFN_TM = 64u;
constant uint PIXELUMM_FFN_TN = 32u;
constant uint PIXELUMM_FFN_TK = 32u;
constant uint PIXELUMM_FFN_THREADS = 128u;
constant uint PIXELUMM_FFN_ROW_BLOCKS = PIXELUMM_FFN_TM / 16u;
constant uint PIXELUMM_FFN_COL_BLOCKS = PIXELUMM_FFN_TN / 16u;

static inline float pixelumm_f32_bf16_to_float(ushort bits) {
  return as_type<float>(uint(bits) << 16);
}

// Keep the same round-to-nearest-even conversion and quiet-NaN handling as
// pixelumm_ffn_dense.metal and the PixelUMM production bridge.
static inline ushort pixelumm_f32_float_to_bf16(float value) {
  uint bits = as_type<uint>(value);
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u)
    return ushort((bits >> 16) | 0x40u);
  return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

kernel void pixelumm_ffn_f32_b64x32(
    device const ushort *weight_bf16 [[buffer(0)]],
    device const ushort *input_bf16 [[buffer(1)]],
    device ushort *output_bf16 [[buffer(2)]],
    constant uint &K [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &M [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  if (M == 0u || K == 0u || N == 0u ||
      M % PIXELUMM_FFN_TM != 0u || K % PIXELUMM_FFN_TK != 0u ||
      N % PIXELUMM_FFN_TN != 0u || tid >= PIXELUMM_FFN_THREADS)
    return;

  threadgroup float a_panel[PIXELUMM_FFN_TM * PIXELUMM_FFN_TK];
  threadgroup float b_panel[PIXELUMM_FFN_TK * PIXELUMM_FFN_TN];
  simdgroup_float8x8 accum[PIXELUMM_FFN_ROW_BLOCKS * PIXELUMM_FFN_COL_BLOCKS];
  for (uint i = 0u; i < PIXELUMM_FFN_ROW_BLOCKS * PIXELUMM_FFN_COL_BLOCKS; ++i)
    accum[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

  const uint m_base = group.y * PIXELUMM_FFN_TM;
  const uint n_base = group.x * PIXELUMM_FFN_TN;
  const uint sg_row = (simdgroup & 1u) * (PIXELUMM_FFN_TM / 2u);
  const uint sg_col = (simdgroup >> 1u) * (PIXELUMM_FFN_TN / 2u);

  for (uint k0 = 0u; k0 < K; k0 += PIXELUMM_FFN_TK) {
    // Weight rows are contiguous in K. Load contiguous K spans per output
    // channel, then transpose into the K x N panel consumed by simdgroup MMA.
    for (uint i = tid; i < PIXELUMM_FFN_TK * PIXELUMM_FFN_TN;
         i += PIXELUMM_FFN_THREADS) {
      const uint local_n = i / PIXELUMM_FFN_TK;
      const uint local_k = i % PIXELUMM_FFN_TK;
      const uint n = n_base + local_n;
      const uint k = k0 + local_k;
      b_panel[local_k * PIXELUMM_FFN_TN + local_n] =
          pixelumm_f32_bf16_to_float(weight_bf16[ulong(n) * K + k]);
    }
    for (uint i = tid; i < PIXELUMM_FFN_TM * PIXELUMM_FFN_TK;
         i += PIXELUMM_FFN_THREADS) {
      const uint local_m = i / PIXELUMM_FFN_TK;
      const uint local_k = i % PIXELUMM_FFN_TK;
      const uint m = m_base + local_m;
      const uint k = k0 + local_k;
      a_panel[i] = pixelumm_f32_bf16_to_float(input_bf16[ulong(m) * K + k]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kfrag = 0u; kfrag < PIXELUMM_FFN_TK; kfrag += 8u) {
      for (uint rb = 0u; rb < PIXELUMM_FFN_ROW_BLOCKS; ++rb) {
        simdgroup_float8x8 a_fragment;
        simdgroup_load(a_fragment,
            a_panel + (sg_row + rb * 8u) * PIXELUMM_FFN_TK + kfrag,
            PIXELUMM_FFN_TK);
        for (uint cb = 0u; cb < PIXELUMM_FFN_COL_BLOCKS; ++cb) {
          simdgroup_float8x8 b_fragment;
          simdgroup_load(b_fragment,
              b_panel + kfrag * PIXELUMM_FFN_TN + sg_col + cb * 8u,
              PIXELUMM_FFN_TN);
          const uint acc_index = rb * PIXELUMM_FFN_COL_BLOCKS + cb;
          simdgroup_multiply_accumulate(accum[acc_index], a_fragment,
                                        b_fragment, accum[acc_index]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // K reduction is complete; reuse A's 8 KiB panel as the 64x32 F32 output
  // tile, avoiding a global F32 intermediate or extra dispatch.
  for (uint rb = 0u; rb < PIXELUMM_FFN_ROW_BLOCKS; ++rb) {
    for (uint cb = 0u; cb < PIXELUMM_FFN_COL_BLOCKS; ++cb) {
      const uint acc_index = rb * PIXELUMM_FFN_COL_BLOCKS + cb;
      simdgroup_store(accum[acc_index],
          a_panel + (sg_row + rb * 8u) * PIXELUMM_FFN_TN + sg_col + cb * 8u,
          PIXELUMM_FFN_TN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < PIXELUMM_FFN_TM * PIXELUMM_FFN_TN;
       i += PIXELUMM_FFN_THREADS) {
    const uint m = m_base + i / PIXELUMM_FFN_TN;
    const uint n = n_base + i % PIXELUMM_FFN_TN;
    output_bf16[ulong(m) * N + n] = pixelumm_f32_float_to_bf16(a_panel[i]);
  }
}

// Diagnostic-only variant of the same dense PixelUMM FFN projection. It keeps
// the baseline tile, BF16 expansion, F32 accumulation order and BF16 epilogue,
// while loading each K8 B fragment once and reusing it across row blocks.
kernel void pixelumm_ffn_f32_b64x32_reuse_b(
    device const ushort *weight_bf16 [[buffer(0)]],
    device const ushort *input_bf16 [[buffer(1)]],
    device ushort *output_bf16 [[buffer(2)]],
    constant uint &K [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &M [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  if (M == 0u || K == 0u || N == 0u ||
      M % PIXELUMM_FFN_TM != 0u || K % PIXELUMM_FFN_TK != 0u ||
      N % PIXELUMM_FFN_TN != 0u || tid >= PIXELUMM_FFN_THREADS)
    return;

  threadgroup float a_panel[PIXELUMM_FFN_TM * PIXELUMM_FFN_TK];
  threadgroup float b_panel[PIXELUMM_FFN_TK * PIXELUMM_FFN_TN];
  simdgroup_float8x8 accum[PIXELUMM_FFN_ROW_BLOCKS * PIXELUMM_FFN_COL_BLOCKS];
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < PIXELUMM_FFN_ROW_BLOCKS * PIXELUMM_FFN_COL_BLOCKS; ++i)
    accum[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

  const uint m_base = group.y * PIXELUMM_FFN_TM;
  const uint n_base = group.x * PIXELUMM_FFN_TN;
  const uint sg_row = (simdgroup & 1u) * (PIXELUMM_FFN_TM / 2u);
  const uint sg_col = (simdgroup >> 1u) * (PIXELUMM_FFN_TN / 2u);

  for (uint k0 = 0u; k0 < K; k0 += PIXELUMM_FFN_TK) {
    // Preserve the baseline's coalesced BF16 loads and exact F32 expansion.
    for (uint i = tid; i < PIXELUMM_FFN_TK * PIXELUMM_FFN_TN;
         i += PIXELUMM_FFN_THREADS) {
      const uint local_n = i / PIXELUMM_FFN_TK;
      const uint local_k = i % PIXELUMM_FFN_TK;
      const uint n = n_base + local_n;
      const uint k = k0 + local_k;
      b_panel[local_k * PIXELUMM_FFN_TN + local_n] =
          pixelumm_f32_bf16_to_float(weight_bf16[ulong(n) * K + k]);
    }
    for (uint i = tid; i < PIXELUMM_FFN_TM * PIXELUMM_FFN_TK;
         i += PIXELUMM_FFN_THREADS) {
      const uint local_m = i / PIXELUMM_FFN_TK;
      const uint local_k = i % PIXELUMM_FFN_TK;
      const uint m = m_base + local_m;
      const uint k = k0 + local_k;
      a_panel[i] = pixelumm_f32_bf16_to_float(input_bf16[ulong(m) * K + k]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (uint kfrag = 0u; kfrag < PIXELUMM_FFN_TK; kfrag += 8u) {
      simdgroup_float8x8 b_fragments[PIXELUMM_FFN_COL_BLOCKS];
      #pragma clang loop unroll(full)
      for (uint cb = 0u; cb < PIXELUMM_FFN_COL_BLOCKS; ++cb) {
        simdgroup_load(b_fragments[cb],
            b_panel + kfrag * PIXELUMM_FFN_TN + sg_col + cb * 8u,
            PIXELUMM_FFN_TN);
      }
      #pragma clang loop unroll(full)
      for (uint rb = 0u; rb < PIXELUMM_FFN_ROW_BLOCKS; ++rb) {
        simdgroup_float8x8 a_fragment;
        simdgroup_load(a_fragment,
            a_panel + (sg_row + rb * 8u) * PIXELUMM_FFN_TK + kfrag,
            PIXELUMM_FFN_TK);
        #pragma clang loop unroll(full)
        for (uint cb = 0u; cb < PIXELUMM_FFN_COL_BLOCKS; ++cb) {
          const uint acc_index = rb * PIXELUMM_FFN_COL_BLOCKS + cb;
          simdgroup_multiply_accumulate(accum[acc_index], a_fragment,
                                        b_fragments[cb], accum[acc_index]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Reuse A's panel for the output tile, exactly as in the baseline kernel.
  #pragma clang loop unroll(full)
  for (uint rb = 0u; rb < PIXELUMM_FFN_ROW_BLOCKS; ++rb) {
    #pragma clang loop unroll(full)
    for (uint cb = 0u; cb < PIXELUMM_FFN_COL_BLOCKS; ++cb) {
      const uint acc_index = rb * PIXELUMM_FFN_COL_BLOCKS + cb;
      simdgroup_store(accum[acc_index],
          a_panel + (sg_row + rb * 8u) * PIXELUMM_FFN_TN + sg_col + cb * 8u,
          PIXELUMM_FFN_TN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < PIXELUMM_FFN_TM * PIXELUMM_FFN_TN;
       i += PIXELUMM_FFN_THREADS) {
    const uint m = m_base + i / PIXELUMM_FFN_TN;
    const uint n = n_base + i % PIXELUMM_FFN_TN;
    output_bf16[ulong(m) * N + n] = pixelumm_f32_float_to_bf16(a_panel[i]);
  }
}
