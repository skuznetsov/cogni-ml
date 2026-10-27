#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Diagnostic-only PixelUMM FFN dense candidate. The GEMM schedule and the
// 64x32 weight / 32x64 activation staging layout derive from
// src/ml/gguf/kernels/gemm_q4k.metal:1124-1270
// (simd_mm_q4k_h16_b64). This keeps its eight-simdgroup B64 tile, ascending
// K32 panels, four K8 MMA steps per panel, and F32 accumulators. Only the
// operand loader and final store change: native BF16 values are converted to
// half in threadgroup memory and F32 results are rounded directly to BF16.
// This is an F16-operand precision experiment, not a precision-preserving
// BF16 implementation. Use pixelumm_ffn_f16_roundtrip and the BF16 output
// comparator before drawing any accuracy conclusion.
//
// Matmul contract: weight[N,K], input[M,K], output[M,N], row-major BF16;
// K is 4096 or 12288, N is 12288 or 4096, and M is 2048 or 21120. All
// dimensions are multiples of the fixed B64/K32 tile. Dispatch groups are
// (M/64,N/64,1), with (256,1,1) threads and 16384 bytes at threadgroup
// memory index 0. No split-K, global F16 staging, or host synchronization.

static inline float pixelumm_ffn_bf16_to_float(ushort value) {
  return as_type<float>(uint(value) << 16);
}

// Match PixelUMM's existing round-to-nearest-even BF16 cast, including quiet
// NaN handling (pixel_float_to_bf16 in pixelumm_decoder_block_mps_bridge.mm).
static inline ushort pixelumm_ffn_float_to_bf16(float value) {
  uint bits = as_type<uint>(value);
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u)
    return ushort((bits >> 16) | 0x40u);
  return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Stage the same B64 operand layout as simd_mm_q4k_h16_b64. A is the
// [K,N] half panel; B is the [K,M] half panel. Every dispatched tile is full.
static inline void pixelumm_ffn_stage_b64(
    device const ushort *weight_bf16, device const ushort *input_bf16,
    threadgroup half *sa, threadgroup half *sb, uint K, uint n_base,
    uint m_base, uint panel, uint tid) {
  const uint tidw = tid & 127u;
  if (tid < 128u) {
    const uint n = n_base + tidw / 2u;
    const uint il0 = tidw % 2u;
    for (uint i = 0u; i < 16u; ++i) {
      const uint sx = 2u * il0 + i / 8u;
      const uint sy = (tidw / 2u) / 8u;
      const uint lx = (tidw / 2u) % 8u;
      const uint ly = i % 8u;
      const uint k = panel * 32u + 16u * il0 + i;
      sa[64u * (8u * sx + sy) + 8u * ly + lx] =
          half(pixelumm_ffn_bf16_to_float(weight_bf16[ulong(n) * K + k]));
    }
  }

  const uint row_group = tid / 4u;
  const uint k_group = tid % 4u;
  const uint k_start = panel * 32u + 8u * k_group;
  const uint m = m_base + row_group;
  threadgroup half *dst = sb + 64u * (8u * k_group + row_group / 8u) +
                          8u * (row_group % 8u);
  for (uint i = 0u; i < 8u; ++i)
    dst[i] = half(pixelumm_ffn_bf16_to_float(
        input_bf16[ulong(m) * K + k_start + i]));
}

kernel void pixelumm_ffn_bf16_b64(
    device const ushort *weight_bf16 [[buffer(0)]],
    device const ushort *input_bf16 [[buffer(1)]],
    device ushort *output_bf16 [[buffer(2)]],
    constant uint &K [[buffer(3)]], constant uint &N [[buffer(4)]],
    constant uint &M [[buffer(5)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  if (K == 0u || (K & 31u) != 0u || (N & 63u) != 0u || (M & 63u) != 0u)
    return;

  constexpr uint TILE_BYTES = 4096u;
  constexpr uint TILE_ELEMENTS = 64u * 64u;
  threadgroup half *sa_buf[2] = {
      (threadgroup half *)shmem,
      (threadgroup half *)(shmem + 2u * TILE_BYTES)};
  threadgroup half *sb_buf[2] = {
      (threadgroup half *)(shmem + TILE_BYTES),
      (threadgroup half *)(shmem + 3u * TILE_BYTES)};

  const uint m_base = group.x * 64u;
  const uint n_base = group.y * 64u;
  threadgroup half *sa = sa_buf[0];
  threadgroup half *sb = sb_buf[0];
  pixelumm_ffn_stage_b64(weight_bf16, input_bf16, sa, sb, K, n_base,
                         m_base, 0u, uint(tid));
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_half8x8 ma[4];
  simdgroup_half8x8 mb[2];
  simdgroup_float8x8 mc[8];
  for (uint i = 0u; i < 8u; ++i)
    mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

  const uint n_iter = K / 32u;
  for (uint iter = 0u; iter < n_iter; ++iter) {
    const uint cur = iter & 1u;
    const uint nxt = 1u - cur;
    threadgroup const half *lsma = sa_buf[cur] + 4u * 64u * (uint(sgitg) & 1u);
    threadgroup const half *lsmb = sb_buf[cur] + 2u * 64u * (uint(sgitg) / 2u);

    for (uint ik = 0u; ik < 4u; ++ik) {
      simdgroup_barrier(mem_flags::mem_none);
      for (uint i = 0u; i < 4u; ++i)
        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
      simdgroup_barrier(mem_flags::mem_none);
      for (uint i = 0u; i < 2u; ++i)
        simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
      simdgroup_barrier(mem_flags::mem_none);
      for (uint i = 0u; i < 8u; ++i)
        simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i % 4u], mc[i]);
      lsma += 8u * 64u;
      lsmb += 8u * 64u;
    }

    if (iter + 1u < n_iter) {
      pixelumm_ffn_stage_b64(weight_bf16, input_bf16, sa_buf[nxt],
                             sb_buf[nxt], K, n_base, m_base, iter + 1u,
                             uint(tid));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // The two K panels are dead after the reduction. Reuse their 16 KiB for one
  // full F32 output tile so BF16 rounding writes directly to the caller's
  // existing buffer without a global F32 scratch allocation.
  threadgroup float *temp = (threadgroup float *)shmem;
  threadgroup float *sg_out =
      temp + 32u * (uint(sgitg) & 1u) +
      16u * (uint(sgitg) / 2u) * 64u;
  for (uint i = 0u; i < 8u; ++i)
    simdgroup_store(mc[i], sg_out + 8u * (i % 4u) +
                                 8u * 64u * (i / 4u),
                    64, 0, false);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint index = uint(tid); index < TILE_ELEMENTS; index += 256u) {
    const uint m = index / 64u;
    const uint n = index % 64u;
    const float value = temp[index];
    output_bf16[ulong(m_base + m) * N + n_base + n] =
        pixelumm_ffn_float_to_bf16(value);
  }
}

// Expand saved small-fixture rows cyclically on GPU. Source and destination
// are row-major BF16 arrays. Dispatch ceil(rows*width/256) groups of 256
// threads. This creates diagnostic inputs/expected outputs only; it does not
// claim that repeated rows reproduce a full-generation activation history.
kernel void pixelumm_ffn_repeat_bf16(
    device const ushort *src [[buffer(0)]], device ushort *dst [[buffer(1)]],
    constant uint &sourceRows [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong total = ulong(rows) * ulong(width);
  if (sourceRows == 0u || width == 0u || ulong(gid) >= total)
    return;
  const uint row = gid / width;
  const uint col = gid % width;
  dst[gid] = src[ulong(row % sourceRows) * width + col];
}

// Inspect whether a stored BF16 tensor survives BF16->F16->F32->BF16.
// stats[0..2] are mismatch, finite nonzero -> zero, finite -> infinity.
// The caller zeros stats before dispatch. Use dispatchThreadgroups with
// ceil(count/256) full groups of 256 threads; three atomic additions per group
// bound contention even when many operands mismatch.
kernel void pixelumm_ffn_f16_roundtrip(
    device const ushort *values [[buffer(0)]],
    device atomic_uint *stats [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]]) {
  threadgroup uint local[3][256];
  uint mismatch = 0u;
  uint nonzero_to_zero = 0u;
  uint finite_to_inf = 0u;
  if (gid < count) {
    const ushort source_bits = values[gid];
    const float source = pixelumm_ffn_bf16_to_float(source_bits);
    const float via_f16 = float(half(source));
    const ushort roundtrip = pixelumm_ffn_float_to_bf16(via_f16);
    mismatch = uint(roundtrip != source_bits);
    nonzero_to_zero = uint(isfinite(source) && source != 0.0f && via_f16 == 0.0f);
    finite_to_inf = uint(isfinite(source) && isinf(via_f16));
  }
  local[0][tid] = mismatch;
  local[1][tid] = nonzero_to_zero;
  local[2][tid] = finite_to_inf;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) {
      local[0][tid] += local[0][tid + stride];
      local[1][tid] += local[1][tid + stride];
      local[2][tid] += local[2][tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    for (uint i = 0u; i < 3u; ++i)
      atomic_fetch_add_explicit(stats + i, local[i][0], memory_order_relaxed);
  }
}

// Compare a BF16 output against a saved small-fixture BF16 reference repeated
// cyclically by flat element index. For row-major outputs, expectedCount must
// be a whole number of rows (normally 2048*width). A zero expectedCount
// contributes numeric failures rather than causing modulo by zero or an
// out-of-bounds read. stats[0..3] are numeric failures, BF16 word differences,
// elements with either side nonfinite, and elements seen. `inject=1` replaces
// only decoded actual element 0 with a quiet NaN for the numeric check. The
// actual word comparison happens first, so injection never fabricates a word
// difference. Run injection with a separate zeroed stats buffer.
//
// Use dispatchThreadgroups with ceil(count/256) full groups of 256 threads;
// tail lanes contribute zero. Exact-count dispatchThreads may create a partial
// final group and is not valid for this fixed-width reduction. The per-group
// reduction keeps the full-J diagnostic to four device atomics per group.
kernel void pixelumm_ffn_compare_bf16(
    device const ushort *actual [[buffer(0)]],
    device const ushort *expected [[buffer(1)]],
    device atomic_uint *stats [[buffer(2)]], constant uint &count [[buffer(3)]],
    constant uint &expectedCount [[buffer(4)]],
    constant uint &inject [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]]) {
  threadgroup uint local[4][256];
  uint numeric_failures = 0u;
  uint word_differences = 0u;
  uint nonfinite_values = 0u;
  uint elements_seen = 0u;
  if (gid < count) {
    elements_seen = 1u;
    if (expectedCount == 0u) {
      numeric_failures = 1u;
    } else {
      const ushort actual_bits = actual[gid];
      const ushort expected_bits = expected[gid % expectedCount];
      word_differences = uint(actual_bits != expected_bits);

      float a = pixelumm_ffn_bf16_to_float(actual_bits);
      const float e = pixelumm_ffn_bf16_to_float(expected_bits);
      if (inject != 0u && gid == 0u)
        a = as_type<float>(0x7fc00000u);

      const bool finite_a = isfinite(a);
      const bool finite_e = isfinite(e);
      nonfinite_values = uint(!finite_a || !finite_e);
      bool within_tolerance = false;
      if (finite_a && finite_e) {
        within_tolerance = abs(a - e) <= 2.0e-5f + 2.0e-5f * abs(e);
      }
      numeric_failures = uint(!within_tolerance);
    }
  }

  local[0][tid] = numeric_failures;
  local[1][tid] = word_differences;
  local[2][tid] = nonfinite_values;
  local[3][tid] = elements_seen;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) {
      local[0][tid] += local[0][tid + stride];
      local[1][tid] += local[1][tid + stride];
      local[2][tid] += local[2][tid + stride];
      local[3][tid] += local[3][tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    for (uint i = 0u; i < 4u; ++i)
      atomic_fetch_add_explicit(stats + i, local[i][0], memory_order_relaxed);
  }
}
