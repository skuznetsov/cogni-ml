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
