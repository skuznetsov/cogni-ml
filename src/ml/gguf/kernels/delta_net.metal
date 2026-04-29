// DeltaNet / GatedDeltaRule step for Qwen 3.5 / 3.6 recurrent layers.
//
// Formula (from qwen35_cpu.cr forward_recurrent_layer step 12):
//   for each v-head h (0 .. h_v - 1):
//     k_head = h % h_k
//     ghead  = g[h]           (already = softplus(...) * ssm_a[h])
//     bhead  = beta[h]        (already sigmoid'd)
//
//     state[h] *= ghead                                   // [s_v, s_k] decay
//     sk[d2]    = Σ_d1  state[h, d2, d1] * K[k_head, d1]
//     delt[d2]  = bhead * (V[h, d2] - sk[d2])
//     state[h, d2, d1] += K[k_head, d1] * delt[d2]        // outer product add
//     out[h, d2] = Σ_d1  state[h, d2, d1] * (Q[k_head, d1] * scale)
//
// Layout (f32):
//   state  [h_v, s_v, s_v]  state[h * s_v * s_v + d2 * s_v + d1]   (d1 contiguous)
//   q,k    [h_k, s_k]
//   v,out  [h_v, s_v]
//   g,beta [h_v]
//
// Dispatch: (h_v, 1, 1) threadgroups × 32 threads (one simdgroup per head).
//
// Assumes s_k == s_v (verified for Qwen 3.5 9B: both = 128) and s_v % 4 == 0
// so we can issue float4 loads/stores on [h, d2, :] rows (s_v-stride 128 * 4B
// = 512B, always 16-byte aligned).

#include <metal_stdlib>
using namespace metal;

constant ushort DN_SG = 32;   // threads per simdgroup (and per threadgroup)

// Project Q/K heads into fixed low-rank bases.
//
// Layout:
//   q/k    [h_k, s]
//   basis  [h_k, rank, s]
//   c/qbar [h_k, rank]
//
// Dispatch grid: (rank, h_k, 2), where z=0 writes c=K*B and z=1 writes
// qbar=Q*B. This is intentionally simple; s is 128 and rank <= 64 in the
// current experiments.
kernel void lowrank_project_coeffs(
    device const float* q_conv [[buffer(0)]],
    device const float* k_conv [[buffer(1)]],
    device const float* basis  [[buffer(2)]],
    device       float* c      [[buffer(3)]],
    device       float* qbar   [[buffer(4)]],
    constant     uint&  h_k    [[buffer(5)]],
    constant     uint&  s      [[buffer(6)]],
    constant     uint&  rank   [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]])
{
    const uint j = tid.x;
    const uint h = tid.y;
    const uint which = tid.z;
    if (h >= h_k || j >= rank || which > 1) return;

    device const float* src = (which == 0 ? k_conv : q_conv) + h * s;
    device const float* b = basis + (h * rank + j) * s;
    float acc = 0.0f;
    for (uint d = 0; d < s; ++d) {
        acc += src[d] * b[d];
    }

    if (which == 0) {
        c[h * rank + j] = acc;
    } else {
        qbar[h * rank + j] = acc;
    }
}

// Token-major batch version of lowrank_project_coeffs.
//
// Layout:
//   q/k    [n_tokens, h_k, s]
//   basis  [h_k, rank, s]
//   c/qbar [n_tokens, h_k, rank]
//
// Dispatch grid: (rank, h_k, n_tokens * 2), where z/2 is token and z&1
// selects K coefficients (`c`) or Q coefficients (`qbar`).
kernel void lowrank_project_coeffs_chunk(
    device const float* q_conv [[buffer(0)]],
    device const float* k_conv [[buffer(1)]],
    device const float* basis  [[buffer(2)]],
    device       float* c      [[buffer(3)]],
    device       float* qbar   [[buffer(4)]],
    constant     uint&  h_k    [[buffer(5)]],
    constant     uint&  s      [[buffer(6)]],
    constant     uint&  rank   [[buffer(7)]],
    constant     uint&  n_tokens [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]])
{
    const uint j = tid.x;
    const uint h = tid.y;
    const uint token = tid.z >> 1;
    const uint which = tid.z & 1;
    if (h >= h_k || j >= rank || token >= n_tokens) return;

    device const float* src = (which == 0 ? k_conv : q_conv) + (token * h_k + h) * s;
    device const float* b = basis + (h * rank + j) * s;
    float acc = 0.0f;
    for (uint d = 0; d < s; ++d) {
        acc += src[d] * b[d];
    }

    const uint dst = (token * h_k + h) * rank + j;
    if (which == 0) {
        c[dst] = acc;
    } else {
        qbar[dst] = acc;
    }
}

// Project a full DeltaNet recurrent state into the fixed low-rank basis used by
// the draft branch.
//
// Layout:
//   full_state [h_v, s, s]
//   basis      [h_k, rank, s]
//   out        [h_v, s, rank]
//
// Dispatch grid: (rank, s, h_v). For each value head and state row, compute
// dot(full_state[h, row, :], basis[h % h_k, j, :]).
kernel void lowrank_project_state(
    device const float* full_state [[buffer(0)]],
    device const float* basis      [[buffer(1)]],
    device       float* out        [[buffer(2)]],
    constant     uint&  h_k        [[buffer(3)]],
    constant     uint&  h_v        [[buffer(4)]],
    constant     uint&  s          [[buffer(5)]],
    constant     uint&  rank       [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]])
{
    const uint j = tid.x;
    const uint row = tid.y;
    const uint h = tid.z;
    if (j >= rank || row >= s || h >= h_v) return;

    const uint basis_h = h % h_k;
    device const float* src = full_state + (h * s + row) * s;
    device const float* b = basis + (basis_h * rank + j) * s;
    float acc = 0.0f;
    for (uint d = 0; d < s; ++d) {
        acc += src[d] * b[d];
    }
    out[(h * s + row) * rank + j] = acc;
}

// Predict FFN PCA coefficients directly from the post-attention FFN input.
//
// Layout:
//   x             [hidden_dim]
//   x_mean        [hidden_dim]
//   c_mean        [rank]
//   coeff_weights [rank, hidden_dim]
//   coeffs        [rank]
//
// Dispatch grid: rank coefficients. This intentionally keeps one coefficient
// per thread for a correctness/cost microprobe; rank is small (16/32).
kernel void ffn_pca_updown_coeffs(
    device const float* x             [[buffer(0)]],
    device const float* x_mean        [[buffer(1)]],
    device const float* c_mean        [[buffer(2)]],
    device const float* coeff_weights [[buffer(3)]],
    device       float* coeffs        [[buffer(4)]],
    constant     uint&  hidden_dim    [[buffer(5)]],
    constant     uint&  rank          [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint j = tid;
    if (j >= rank) return;

    device const float* w = coeff_weights + j * hidden_dim;
    float acc = c_mean[j];
    for (uint d = 0; d < hidden_dim; ++d) {
        acc += (x[d] - x_mean[d]) * w[d];
    }
    coeffs[j] = acc;
}

// Apply precomputed WdB vectors to PCA coefficients.
//
// Layout:
//   coeffs     [rank]
//   down_basis [rank, hidden_dim]
//   out        [hidden_dim]
//
// Dispatch grid: hidden dimensions.
kernel void ffn_pca_updown_out(
    device const float* coeffs     [[buffer(0)]],
    device const float* down_basis [[buffer(1)]],
    device       float* out        [[buffer(2)]],
    constant     uint&  hidden_dim [[buffer(3)]],
    constant     uint&  rank       [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    const uint d = tid;
    if (d >= hidden_dim) return;

    float acc = 0.0f;
    for (uint j = 0; j < rank; ++j) {
        acc += coeffs[j] * down_basis[j * hidden_dim + d];
    }
    out[d] = acc;
}

// Single-dispatch version for decode-time one-row FFN PCA up/down.
// One threadgroup computes all rank coefficients into threadgroup memory, then
// writes all hidden output dimensions. This avoids the underfilled coefficient
// dispatch and the second command-encoder/kernel boundary above.
kernel void ffn_pca_updown_fused(
    device const float* x             [[buffer(0)]],
    device const float* x_mean        [[buffer(1)]],
    device const float* c_mean        [[buffer(2)]],
    device const float* coeff_weights [[buffer(3)]],
    device const float* down_basis    [[buffer(4)]],
    device       float* out           [[buffer(5)]],
    constant     uint&  hidden_dim    [[buffer(6)]],
    constant     uint&  rank          [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float coeffs[64];

    if (tid < rank && tid < 64) {
        device const float* w = coeff_weights + tid * hidden_dim;
        float acc = c_mean[tid];
        for (uint d = 0; d < hidden_dim; ++d) {
            acc += (x[d] - x_mean[d]) * w[d];
        }
        coeffs[tid] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < hidden_dim; d += 256) {
        float acc = 0.0f;
        for (uint j = 0; j < rank && j < 64; ++j) {
            acc += coeffs[j] * down_basis[j * hidden_dim + d];
        }
        out[d] = acc;
    }
}

// Token-major batch version of ffn_pca_updown_fused.
//
// Layout:
//   x   [n_tokens, hidden_dim]
//   out [n_tokens, hidden_dim]
//
// Dispatch grid: one threadgroup per token.
kernel void ffn_pca_updown_fused_rows(
    device const float* x             [[buffer(0)]],
    device const float* x_mean        [[buffer(1)]],
    device const float* c_mean        [[buffer(2)]],
    device const float* coeff_weights [[buffer(3)]],
    device const float* down_basis    [[buffer(4)]],
    device       float* out           [[buffer(5)]],
    constant     uint&  hidden_dim    [[buffer(6)]],
    constant     uint&  rank          [[buffer(7)]],
    constant     uint&  n_tokens      [[buffer(8)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (token >= n_tokens) return;

    threadgroup float partials[64 * 8];
    threadgroup float coeffs[64];
    device const float* row = x + token * hidden_dim;

    // The original row kernel assigned one thread to each coefficient and
    // made it scan the whole hidden vector. Split each coefficient dot product
    // across several lanes so single-token decode does not serialize on rank
    // threads while preserving the one-dispatch fused path.
    uint lanes = (rank <= 32) ? 8 : 4;
    uint coeff = tid / lanes;
    uint lane = tid - coeff * lanes;
    if (coeff < rank && coeff < 64 && lane < lanes) {
        device const float* w = coeff_weights + coeff * hidden_dim;
        float acc = 0.0f;
        for (uint d = lane; d < hidden_dim; d += lanes) {
            acc += (row[d] - x_mean[d]) * w[d];
        }
        partials[coeff * 8 + lane] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (coeff < rank && coeff < 64 && lane == 0) {
        float acc = c_mean[coeff];
        for (uint l = 0; l < lanes; ++l) {
            acc += partials[coeff * 8 + l];
        }
        coeffs[coeff] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_row = out + token * hidden_dim;
    for (uint d = tid; d < hidden_dim; d += 256) {
        float acc = 0.0f;
        for (uint j = 0; j < rank && j < 64; ++j) {
            acc += coeffs[j] * down_basis[j * hidden_dim + d];
        }
        out_row[d] = acc;
    }
}

// Projected-K low-rank DeltaNet step.
//
// This is the core self-spec draft state update after Q/K have already been
// projected into the per-k-head fixed basis:
//
//   M[h,row,:] *= g[h]
//   sk          = dot(M[h,row,:], c[k_head,:])
//   delta       = beta[h] * (V[h,row] - sk)
//   M[h,row,:] += delta * c[k_head,:]
//   out[h,row]  = dot(M[h,row,:], qbar[k_head,:]) * scale
//
// Layout:
//   m_state [h_v, s, rank]
//   c/qbar  [h_k, rank]
//   v/out   [h_v, s]
//
// Dispatch: (s, h_v, 1) threadgroups x 1 thread. This intentionally starts
// as a correctness kernel; rank is small (32/48/64) and the next pass can
// split each row across a simdgroup if profiling shows this is the bottleneck.
kernel void lowrank_delta_step(
    device       float* m_state [[buffer(0)]],
    device const float* c       [[buffer(1)]],
    device const float* qbar    [[buffer(2)]],
    device const float* v_conv  [[buffer(3)]],
    device const float* g       [[buffer(4)]],
    device const float* beta    [[buffer(5)]],
    device       float* out     [[buffer(6)]],
    constant     uint&  h_k     [[buffer(7)]],
    constant     uint&  h_v     [[buffer(8)]],
    constant     uint&  s       [[buffer(9)]],
    constant     uint&  rank    [[buffer(10)]],
    constant     float& scale   [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]])
{
    const uint row = tid.x;
    const uint h = tid.y;
    if (row >= s || h >= h_v) return;

    const uint k_head = h % h_k;
    device float* m = m_state + (h * s + row) * rank;
    device const float* ch = c + k_head * rank;
    device const float* qh = qbar + k_head * rank;

    const float gh = g[h];
    const float bh = beta[h];

    float sk = 0.0f;
    for (uint j = 0; j < rank; ++j) {
        const float mv = m[j] * gh;
        m[j] = mv;
        sk += mv * ch[j];
    }

    const float delt = bh * (v_conv[h * s + row] - sk);
    float acc = 0.0f;
    for (uint j = 0; j < rank; ++j) {
        const float mv = m[j] + delt * ch[j];
        m[j] = mv;
        acc += mv * qh[j];
    }

    out[h * s + row] = acc * scale;
}

// Token-major low-rank recurrent scan. One threadgroup owns one [h,row] state
// row and scans tokens serially, while rank dot/update work is distributed
// across the threadgroup.
kernel void lowrank_delta_chunk_step_parallel(
    device       float* m_state [[buffer(0)]],
    device const float* c       [[buffer(1)]],
    device const float* qbar    [[buffer(2)]],
    device const float* v_conv  [[buffer(3)]],
    device const float* g       [[buffer(4)]],
    device const float* beta    [[buffer(5)]],
    device       float* out     [[buffer(6)]],
    constant     uint&  h_k     [[buffer(7)]],
    constant     uint&  h_v     [[buffer(8)]],
    constant     uint&  s       [[buffer(9)]],
    constant     uint&  rank    [[buffer(10)]],
    constant     uint&  n_tokens [[buffer(11)]],
    constant     float& scale   [[buffer(12)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]])
{
    const uint row = tgpig.x;
    const uint h = tgpig.y;
    if (row >= s || h >= h_v) return;

    const uint k_head = h % h_k;
    device float* m = m_state + (h * s + row) * rank;
    threadgroup float scratch[128];

    for (uint t = 0; t < n_tokens; ++t) {
        device const float* ch = c + (t * h_k + k_head) * rank;
        device const float* qh = qbar + (t * h_k + k_head) * rank;
        const float gh = g[t * h_v + h];
        const float bh = beta[t * h_v + h];

        float mv = 0.0f;
        float part = 0.0f;
        if (tiitg < rank) {
            mv = m[tiitg] * gh;
            m[tiitg] = mv;
            part = mv * ch[tiitg];
        }
        scratch[tiitg] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 64; stride > 0; stride >>= 1) {
            if (tiitg < stride && tiitg + stride < rank) {
                scratch[tiitg] += scratch[tiitg + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float delt = bh * (v_conv[(t * h_v + h) * s + row] - scratch[0]);
        part = 0.0f;
        if (tiitg < rank) {
            mv = m[tiitg] + delt * ch[tiitg];
            m[tiitg] = mv;
            part = mv * qh[tiitg];
        }
        scratch[tiitg] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 64; stride > 0; stride >>= 1) {
            if (tiitg < stride && tiitg + stride < rank) {
                scratch[tiitg] += scratch[tiitg + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiitg == 0) {
            out[(t * h_v + h) * s + row] = scratch[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void delta_net_step(
    device       float* state  [[buffer(0)]],
    device const float* q_conv [[buffer(1)]],
    device const float* k_conv [[buffer(2)]],
    device const float* v_conv [[buffer(3)]],
    device const float* g      [[buffer(4)]],
    device const float* beta   [[buffer(5)]],
    device       float* out    [[buffer(6)]],
    constant     uint&  h_k    [[buffer(7)]],
    constant     uint&  h_v    [[buffer(8)]],
    constant     uint&  s      [[buffer(9)]],
    constant     float& scale  [[buffer(10)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h      = tgpig;
    if (h >= h_v) return;

    const uint k_head = h % h_k;
    const uint st_b   = h * s * s;

    const float ghead = g[h];
    const float bhead = beta[h];

    device const float* K = k_conv + k_head * s;
    device const float* Q = q_conv + k_head * s;
    device const float* V = v_conv + h      * s;
    device       float* O = out    + h      * s;

    // 1) Decay: state *= ghead. Stripe s*s elements across DN_SG threads.
    for (uint i = tiisg; i < s * s; i += DN_SG) {
        state[st_b + i] *= ghead;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) sk[d2] = Σ_d1 state[h, d2, d1] * K[d1]. Use threadgroup scratch.
    threadgroup float sk[128];   // s_v upper bound for Qwen35 9B

    for (uint d2 = tiisg; d2 < s; d2 += DN_SG) {
        device const float* row = state + st_b + d2 * s;
        float acc = 0.0f;
        for (uint d1 = 0; d1 < s; d1 += 4) {
            float4 rv = *((device const float4*)(row + d1));
            float4 kv = *((device const float4*)(K   + d1));
            acc += rv.x * kv.x + rv.y * kv.y + rv.z * kv.z + rv.w * kv.w;
        }
        sk[d2] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) State update: state[h, d2, :] += K[:] * delt[d2]
    //    where delt[d2] = bhead * (V[d2] - sk[d2]).
    for (uint d2 = tiisg; d2 < s; d2 += DN_SG) {
        float delt = bhead * (V[d2] - sk[d2]);
        device float* row = state + st_b + d2 * s;
        for (uint d1 = 0; d1 < s; d1 += 4) {
            float4 rv = *((device       float4*)(row + d1));
            float4 kv = *((device const float4*)(K   + d1));
            rv.x += kv.x * delt;
            rv.y += kv.y * delt;
            rv.z += kv.z * delt;
            rv.w += kv.w * delt;
            *((device float4*)(row + d1)) = rv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) out[d2] = Σ_d1 state[h, d2, d1] * Q[d1]  (× scale)
    for (uint d2 = tiisg; d2 < s; d2 += DN_SG) {
        device const float* row = state + st_b + d2 * s;
        float acc = 0.0f;
        for (uint d1 = 0; d1 < s; d1 += 4) {
            float4 rv = *((device const float4*)(row + d1));
            float4 qv = *((device const float4*)(Q   + d1));
            acc += rv.x * qv.x + rv.y * qv.y + rv.z * qv.z + rv.w * qv.w;
        }
        O[d2] = acc * scale;
    }
}

// Four-simdgroup variant of DeltaNet step. Arithmetic is identical to
// `delta_net_step`; only the intra-head work partition changes.
//
// Dispatch: (h_v, 1, 1) threadgroups × 128 threads (4 simdgroups per head).
kernel void delta_net_step_128(
    device       float* state  [[buffer(0)]],
    device const float* q_conv [[buffer(1)]],
    device const float* k_conv [[buffer(2)]],
    device const float* v_conv [[buffer(3)]],
    device const float* g      [[buffer(4)]],
    device const float* beta   [[buffer(5)]],
    device       float* out    [[buffer(6)]],
    constant     uint&  h_k    [[buffer(7)]],
    constant     uint&  h_v    [[buffer(8)]],
    constant     uint&  s      [[buffer(9)]],
    constant     float& scale  [[buffer(10)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint h      = tgpig;
    if (h >= h_v) return;

    const ushort lane = tiitg & 31;
    const uint k_head = h % h_k;
    const uint st_b   = h * s * s;

    const float ghead = g[h];
    const float bhead = beta[h];

    device const float* K = k_conv + k_head * s;
    device const float* Q = q_conv + k_head * s;
    device const float* V = v_conv + h      * s;
    device       float* O = out    + h      * s;

    // 1) Decay: state *= ghead. Stripe s*s elements across all 128 threads.
    for (uint i = tiitg; i < s * s; i += 128) {
        state[st_b + i] *= ghead;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) sk[d2] = Σ_d1 state[h, d2, d1] * K[d1].
    threadgroup float sk[128];
    for (uint d2 = sgitg; d2 < s; d2 += 4) {
        device const float* row = state + st_b + d2 * s;
        float acc = 0.0f;
        for (uint d1 = lane * 4; d1 < s; d1 += 128) {
            float4 rv = *((device const float4*)(row + d1));
            float4 kv = *((device const float4*)(K   + d1));
            acc += rv.x * kv.x + rv.y * kv.y + rv.z * kv.z + rv.w * kv.w;
        }
        acc = simd_sum(acc);
        if (lane == 0) sk[d2] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) State update.
    for (uint d2 = sgitg; d2 < s; d2 += 4) {
        const float delt = bhead * (V[d2] - sk[d2]);
        device float* row = state + st_b + d2 * s;
        for (uint d1 = lane * 4; d1 < s; d1 += 128) {
            float4 rv = *((device       float4*)(row + d1));
            float4 kv = *((device const float4*)(K   + d1));
            rv += kv * delt;
            *((device float4*)(row + d1)) = rv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) Output projection against Q.
    for (uint d2 = sgitg; d2 < s; d2 += 4) {
        device const float* row = state + st_b + d2 * s;
        float acc = 0.0f;
        for (uint d1 = lane * 4; d1 < s; d1 += 128) {
            float4 rv = *((device const float4*)(row + d1));
            float4 qv = *((device const float4*)(Q   + d1));
            acc += rv.x * qv.x + rv.y * qv.y + rv.z * qv.z + rv.w * qv.w;
        }
        acc = simd_sum(acc);
        if (lane == 0) O[d2] = acc * scale;
    }
}

// Fused four-simdgroup variant. Instead of materializing the decayed state in
// a separate pass, it applies the decay in the sk and update/output passes:
//
//   sk       = dot(old_state * g, K)
//   new_row  = old_row * g + K * beta * (V - sk)
//   out      = dot(new_row, Q) * scale
//
// This removes one full state write pass and one full state read pass per
// recurrent head while preserving the arithmetic semantics of
// `delta_net_step_128`.
kernel void delta_net_step_128_fused(
    device       float* state  [[buffer(0)]],
    device const float* q_conv [[buffer(1)]],
    device const float* k_conv [[buffer(2)]],
    device const float* v_conv [[buffer(3)]],
    device const float* g      [[buffer(4)]],
    device const float* beta   [[buffer(5)]],
    device       float* out    [[buffer(6)]],
    constant     uint&  h_k    [[buffer(7)]],
    constant     uint&  h_v    [[buffer(8)]],
    constant     uint&  s      [[buffer(9)]],
    constant     float& scale  [[buffer(10)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint h = tgpig;
    if (h >= h_v) return;

    const ushort lane = tiitg & 31;
    const uint k_head = h % h_k;
    const uint st_b   = h * s * s;

    const float ghead = g[h];
    const float bhead = beta[h];

    device const float* K = k_conv + k_head * s;
    device const float* Q = q_conv + k_head * s;
    device const float* V = v_conv + h      * s;
    device       float* O = out    + h      * s;

    for (uint d2 = sgitg; d2 < s; d2 += 4) {
        device float* row = state + st_b + d2 * s;

        float sk_acc = 0.0f;
        for (uint d1 = lane * 4; d1 < s; d1 += 128) {
            float4 rv = *((device const float4*)(row + d1));
            float4 kv = *((device const float4*)(K   + d1));
            sk_acc += rv.x * kv.x + rv.y * kv.y + rv.z * kv.z + rv.w * kv.w;
        }
        const float sk = simd_sum(sk_acc) * ghead;
        const float delt = bhead * (V[d2] - sk);

        float out_acc = 0.0f;
        for (uint d1 = lane * 4; d1 < s; d1 += 128) {
            float4 rv = *((device const float4*)(row + d1));
            float4 kv = *((device const float4*)(K   + d1));
            float4 qv = *((device const float4*)(Q   + d1));
            rv = rv * ghead + kv * delt;
            *((device float4*)(row + d1)) = rv;
            out_acc += rv.x * qv.x + rv.y * qv.y + rv.z * qv.z + rv.w * qv.w;
        }

        const float ov = simd_sum(out_acc);
        if (lane == 0) O[d2] = ov * scale;
    }
}

// Fused DeltaNet step plus Qwen recurrent post-processing:
//   y = DeltaNet(q, k, v, state, g, beta)
//   y = RMSNorm(y, ssm_norm) * silu(z)
//
// One threadgroup owns one v-head. This removes the separate
// `delta_net_post_norm_gate` dispatch and avoids a global read/write round trip
// for the intermediate DeltaNet output.
kernel void delta_net_step_128_fused_post(
    device       float* state    [[buffer(0)]],
    device const float* q_conv   [[buffer(1)]],
    device const float* k_conv   [[buffer(2)]],
    device const float* v_conv   [[buffer(3)]],
    device const float* g        [[buffer(4)]],
    device const float* beta     [[buffer(5)]],
    device       float* out      [[buffer(6)]],
    constant     uint&  h_k      [[buffer(7)]],
    constant     uint&  h_v      [[buffer(8)]],
    constant     uint&  s        [[buffer(9)]],
    constant     float& scale    [[buffer(10)]],
    device const float* z        [[buffer(11)]],
    device const float* ssm_norm [[buffer(12)]],
    constant     float& eps      [[buffer(13)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint h = tgpig;
    if (h >= h_v) return;

    const ushort lane = tiitg & 31;
    const uint k_head = h % h_k;
    const uint st_b   = h * s * s;

    const float ghead = g[h];
    const float bhead = beta[h];

    device const float* K = k_conv + k_head * s;
    device const float* Q = q_conv + k_head * s;
    device const float* V = v_conv + h      * s;
    device const float* Z = z      + h      * s;
    device       float* O = out    + h      * s;

    threadgroup float y_tmp[128];
    threadgroup float ss_part[4];
    threadgroup float inv_rms;

    float local_ss = 0.0f;
    for (uint d2 = sgitg; d2 < s; d2 += 4) {
        device float* row = state + st_b + d2 * s;

        float sk_acc = 0.0f;
        for (uint d1 = lane * 4; d1 < s; d1 += 128) {
            float4 rv = *((device const float4*)(row + d1));
            float4 kv = *((device const float4*)(K   + d1));
            sk_acc += rv.x * kv.x + rv.y * kv.y + rv.z * kv.z + rv.w * kv.w;
        }
        const float sk = simd_sum(sk_acc) * ghead;
        const float delt = bhead * (V[d2] - sk);

        float out_acc = 0.0f;
        for (uint d1 = lane * 4; d1 < s; d1 += 128) {
            float4 rv = *((device const float4*)(row + d1));
            float4 kv = *((device const float4*)(K   + d1));
            float4 qv = *((device const float4*)(Q   + d1));
            rv = rv * ghead + kv * delt;
            *((device float4*)(row + d1)) = rv;
            out_acc += rv.x * qv.x + rv.y * qv.y + rv.z * qv.z + rv.w * qv.w;
        }

        const float yv = simd_sum(out_acc) * scale;
        if (lane == 0) {
            y_tmp[d2] = yv;
            local_ss += yv * yv;
        }
    }

    if (lane == 0) ss_part[sgitg] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiitg == 0) {
        inv_rms = rsqrt((ss_part[0] + ss_part[1] + ss_part[2] + ss_part[3]) / float(s) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tiitg; d < s; d += 128) {
        const float zv = Z[d];
        const float sig = 1.0f / (1.0f + exp(-zv));
        O[d] = y_tmp[d] * inv_rms * ssm_norm[d] * (zv * sig);
    }
}

// Multi-token DeltaNet scan for prefill chunks.
//
// This is the same recurrence as `delta_net_step_128_fused`, but it keeps the
// scan over `n_tokens` inside one Metal dispatch. It mirrors llama.cpp's
// `GGML_OP_GATED_DELTA_NET` strategy: the recurrence is still serial over
// tokens, but Q/K/V/g/b are consumed as a chunk and recurrent state is kept
// inside one layer/head dispatch instead of paying one dispatch per token.
//
// Layout:
//   state  [h_v, s, s]
//   q,k    [n_tokens, h_k, s]
//   v,out  [n_tokens, h_v, s]
//   g,beta [n_tokens, h_v]
kernel void delta_net_chunk_128_fused(
    device       float* state    [[buffer(0)]],
    device const float* q_conv   [[buffer(1)]],
    device const float* k_conv   [[buffer(2)]],
    device const float* v_conv   [[buffer(3)]],
    device const float* g        [[buffer(4)]],
    device const float* beta     [[buffer(5)]],
    device       float* out      [[buffer(6)]],
    constant     uint&  h_k      [[buffer(7)]],
    constant     uint&  h_v      [[buffer(8)]],
    constant     uint&  s        [[buffer(9)]],
    constant     float& scale    [[buffer(10)]],
    constant     uint&  n_tokens [[buffer(11)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint h = tgpig;
    if (h >= h_v) return;

    const ushort lane = tiitg & 31;
    const uint k_head = h % h_k;
    const uint st_b   = h * s * s;

    for (uint t = 0; t < n_tokens; ++t) {
        const float ghead = g[t * h_v + h];
        const float bhead = beta[t * h_v + h];

        device const float* K = k_conv + (t * h_k + k_head) * s;
        device const float* Q = q_conv + (t * h_k + k_head) * s;
        device const float* V = v_conv + (t * h_v + h) * s;
        device       float* O = out    + (t * h_v + h) * s;

        for (uint d2 = sgitg; d2 < s; d2 += 4) {
            device float* row = state + st_b + d2 * s;

            float sk_acc = 0.0f;
            for (uint d1 = lane * 4; d1 < s; d1 += 128) {
                float4 rv = *((device const float4*)(row + d1));
                float4 kv = *((device const float4*)(K   + d1));
                sk_acc += rv.x * kv.x + rv.y * kv.y + rv.z * kv.z + rv.w * kv.w;
            }
            const float sk = simd_sum(sk_acc) * ghead;
            const float delt = bhead * (V[d2] - sk);

            float out_acc = 0.0f;
            for (uint d1 = lane * 4; d1 < s; d1 += 128) {
                float4 rv = *((device const float4*)(row + d1));
                float4 kv = *((device const float4*)(K   + d1));
                float4 qv = *((device const float4*)(Q   + d1));
                rv = rv * ghead + kv * delt;
                *((device float4*)(row + d1)) = rv;
                out_acc += rv.x * qv.x + rv.y * qv.y + rv.z * qv.z + rv.w * qv.w;
            }

            const float ov = simd_sum(out_acc);
            if (lane == 0) O[d2] = ov * scale;
        }
    }
}

// Row-resident variant for s=128 prefill chunks.
//
// One threadgroup owns four state rows for one v-head. Each simdgroup keeps one
// row stripe in registers across the whole token scan, so state is read once and
// written once per chunk instead of once per token. The arithmetic order matches
// `delta_net_chunk_128_fused`: sk is computed from the old row, then multiplied
// by ghead, then the decayed row receives K * delta.
kernel void delta_net_chunk_128_rowwise(
    device       float* state    [[buffer(0)]],
    device const float* q_conv   [[buffer(1)]],
    device const float* k_conv   [[buffer(2)]],
    device const float* v_conv   [[buffer(3)]],
    device const float* g        [[buffer(4)]],
    device const float* beta     [[buffer(5)]],
    device       float* out      [[buffer(6)]],
    constant     uint&  h_k      [[buffer(7)]],
    constant     uint&  h_v      [[buffer(8)]],
    constant     uint&  s        [[buffer(9)]],
    constant     float& scale    [[buffer(10)]],
    constant     uint&  n_tokens [[buffer(11)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint d2 = tgpig.x * 4 + sgitg;
    const uint h  = tgpig.y;
    if (h >= h_v || d2 >= s) return;

    const uint k_head = h % h_k;
    const uint st_b   = h * s * s;
    const uint d1     = tiisg * 4;

    device float* row = state + st_b + d2 * s;
    float4 rs = *((device const float4*)(row + d1));

    for (uint t = 0; t < n_tokens; ++t) {
        const float ghead = g[t * h_v + h];
        const float bhead = beta[t * h_v + h];

        device const float* K = k_conv + (t * h_k + k_head) * s;
        device const float* Q = q_conv + (t * h_k + k_head) * s;
        device const float* V = v_conv + (t * h_v + h) * s;
        device       float* O = out    + (t * h_v + h) * s;

        const float4 kv = *((device const float4*)(K + d1));
        const float4 qv = *((device const float4*)(Q + d1));

        float sk_acc = rs.x * kv.x + rs.y * kv.y + rs.z * kv.z + rs.w * kv.w;
        const float sk = simd_sum(sk_acc) * ghead;
        const float delt = bhead * (V[d2] - sk);

        rs = rs * ghead + kv * delt;

        float out_acc = rs.x * qv.x + rs.y * qv.y + rs.z * qv.z + rs.w * qv.w;
        const float ov = simd_sum(out_acc);
        if (tiisg == 0) O[d2] = ov * scale;
    }

    *((device float4*)(row + d1)) = rs;
}

// In-place recurrent post-processing for Qwen35 after DeltaNet:
//   y[h, d] = RMSNorm(y[h, :], ssm_norm) * silu(z[h, d])
//
// One threadgroup per v-head, one simdgroup per threadgroup.
kernel void delta_net_post_norm_gate(
    device       float* y        [[buffer(0)]],
    device const float* z        [[buffer(1)]],
    device const float* ssm_norm [[buffer(2)]],
    constant     uint&  h_v      [[buffer(3)]],
    constant     uint&  s        [[buffer(4)]],
    constant     float& eps      [[buffer(5)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig;
    if (h >= h_v) return;

    device float*       Y = y + h * s;
    device const float* Z = z + h * s;

    float ss = 0.0f;
    for (uint d = tiisg; d < s; d += DN_SG) {
        float v = Y[d];
        ss += v * v;
    }

    const float sum = simd_sum(ss);
    const float inv_rms = rsqrt(sum / float(s) + eps);

    for (uint d = tiisg; d < s; d += DN_SG) {
        const float zv = Z[d];
        const float sig = 1.0f / (1.0f + exp(-zv));
        Y[d] = Y[d] * inv_rms * ssm_norm[d] * (zv * sig);
    }
}

// Chunked in-place recurrent post-processing:
//   y[t, h, d] = RMSNorm(y[t, h, :], ssm_norm) * silu(z[t, h, d])
kernel void delta_net_post_norm_gate_chunk(
    device       float* y        [[buffer(0)]],
    device const float* z        [[buffer(1)]],
    device const float* ssm_norm [[buffer(2)]],
    constant     uint&  h_v      [[buffer(3)]],
    constant     uint&  s        [[buffer(4)]],
    constant     float& eps      [[buffer(5)]],
    constant     uint&  n_tokens [[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= h_v || t >= n_tokens) return;

    device float*       Y = y + (t * h_v + h) * s;
    device const float* Z = z + (t * h_v + h) * s;

    float ss = 0.0f;
    for (uint d = tiisg; d < s; d += DN_SG) {
        float v = Y[d];
        ss += v * v;
    }

    const float sum = simd_sum(ss);
    const float inv_rms = rsqrt(sum / float(s) + eps);

    for (uint d = tiisg; d < s; d += DN_SG) {
        const float zv = Z[d];
        const float sig = 1.0f / (1.0f + exp(-zv));
        Y[d] = Y[d] * inv_rms * ssm_norm[d] * (zv * sig);
    }
}
