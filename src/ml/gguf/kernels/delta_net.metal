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
