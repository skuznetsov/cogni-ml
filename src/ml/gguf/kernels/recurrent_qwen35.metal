#include <metal_stdlib>
using namespace metal;

kernel void qwen35_recurrent_ab(
    device const float* alpha      [[buffer(0)]],
    device       float* beta       [[buffer(1)]],
    device const float* ssm_dt_bias[[buffer(2)]],
    device const float* ssm_a      [[buffer(3)]],
    device       float* ghead      [[buffer(4)]],
    constant     uint&  h_v        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= h_v) return;

    float b = beta[gid];
    beta[gid] = 1.0f / (1.0f + exp(-b));

    float xi = alpha[gid] + ssm_dt_bias[gid];
    float sp = xi > 20.0f ? xi : log(1.0f + exp(xi));
    ghead[gid] = exp(sp * ssm_a[gid]);
}

kernel void qwen35_recurrent_ab_chunk(
    device const float* alpha      [[buffer(0)]],
    device       float* beta       [[buffer(1)]],
    device const float* ssm_dt_bias[[buffer(2)]],
    device const float* ssm_a      [[buffer(3)]],
    device       float* ghead      [[buffer(4)]],
    constant     uint&  h_v        [[buffer(5)]],
    constant     uint&  n_tokens   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = h_v * n_tokens;
    if (gid >= total) return;

    const uint h = gid % h_v;

    float b = beta[gid];
    beta[gid] = 1.0f / (1.0f + exp(-b));

    float xi = alpha[gid] + ssm_dt_bias[h];
    float sp = xi > 20.0f ? xi : log(1.0f + exp(xi));
    ghead[gid] = exp(sp * ssm_a[h]);
}

// qkv_dim = 2*h_k*s + h_v*s
// conv_state layout: [conv_k - 1, qkv_dim] time-major
kernel void qwen35_recurrent_conv(
    device const float* conv_state [[buffer(0)]],
    device const float* qkv_mixed  [[buffer(1)]],
    device const float* conv1d     [[buffer(2)]],
    device       float* q_out      [[buffer(3)]],
    device       float* k_out      [[buffer(4)]],
    device       float* v_out      [[buffer(5)]],
    constant     uint&  h_k        [[buffer(6)]],
    constant     uint&  h_v        [[buffer(7)]],
    constant     uint&  s          [[buffer(8)]],
    constant     uint&  conv_k     [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    const uint qkv_dim = 2 * h_k * s + h_v * s;
    if (gid >= qkv_dim) return;

    float acc = 0.0f;
    const uint w_base = gid * conv_k;
    for (uint t = 0; t + 1 < conv_k; t++) {
        acc += conv_state[t * qkv_dim + gid] * conv1d[w_base + t];
    }
    acc += qkv_mixed[gid] * conv1d[w_base + (conv_k - 1)];

    const float sig = 1.0f / (1.0f + exp(-acc));
    const float val = acc * sig;

    const uint q_dim = h_k * s;
    const uint k_dim = h_k * s;
    if (gid < q_dim) {
        q_out[gid] = val;
    } else if (gid < q_dim + k_dim) {
        k_out[gid - q_dim] = val;
    } else {
        v_out[gid - q_dim - k_dim] = val;
    }
}

kernel void qwen35_recurrent_shift(
    device       float* conv_state [[buffer(0)]],
    device const float* qkv_mixed  [[buffer(1)]],
    constant     uint&  qkv_dim    [[buffer(2)]],
    constant     uint&  conv_k     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= qkv_dim) return;

    for (uint t = 0; t + 2 < conv_k; t++) {
        conv_state[t * qkv_dim + gid] = conv_state[(t + 1) * qkv_dim + gid];
    }
    conv_state[(conv_k - 2) * qkv_dim + gid] = qkv_mixed[gid];
}

// Exact fused form of qwen35_recurrent_conv + qwen35_recurrent_shift.
// Each channel owns one independent conv_state column, so computing the
// convolution from old values and then shifting that same column is safe.
kernel void qwen35_recurrent_conv_shift(
    device       float* conv_state [[buffer(0)]],
    device const float* qkv_mixed  [[buffer(1)]],
    device const float* conv1d     [[buffer(2)]],
    device       float* q_out      [[buffer(3)]],
    device       float* k_out      [[buffer(4)]],
    device       float* v_out      [[buffer(5)]],
    constant     uint&  h_k        [[buffer(6)]],
    constant     uint&  h_v        [[buffer(7)]],
    constant     uint&  s          [[buffer(8)]],
    constant     uint&  conv_k     [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    const uint qkv_dim = 2 * h_k * s + h_v * s;
    if (gid >= qkv_dim) return;

    float acc = 0.0f;
    float cur = 0.0f;
    const uint w_base = gid * conv_k;
    for (uint t = 0; t + 1 < conv_k; t++) {
        cur = conv_state[t * qkv_dim + gid];
        acc += cur * conv1d[w_base + t];
        if (t > 0) {
            conv_state[(t - 1) * qkv_dim + gid] = cur;
        }
    }

    const float mixed = qkv_mixed[gid];
    acc += mixed * conv1d[w_base + (conv_k - 1)];
    if (conv_k >= 2) {
        conv_state[(conv_k - 2) * qkv_dim + gid] = mixed;
    }

    const float sig = 1.0f / (1.0f + exp(-acc));
    const float val = acc * sig;

    const uint q_dim = h_k * s;
    const uint k_dim = h_k * s;
    if (gid < q_dim) {
        q_out[gid] = val;
    } else if (gid < q_dim + k_dim) {
        k_out[gid - q_dim] = val;
    } else {
        v_out[gid - q_dim - k_dim] = val;
    }
}

// Exact multi-token form of qwen35_recurrent_conv_shift.
//
// qkv_mixed is token-major [n_tokens, qkv_dim]. Each channel owns one
// independent convolution state column, so one thread can scan all tokens for
// that channel and commit the final conv_state without cross-thread ordering.
kernel void qwen35_recurrent_conv_shift_chunk(
    device       float* conv_state [[buffer(0)]],
    device const float* qkv_mixed  [[buffer(1)]],
    device const float* conv1d     [[buffer(2)]],
    device       float* q_out      [[buffer(3)]],
    device       float* k_out      [[buffer(4)]],
    device       float* v_out      [[buffer(5)]],
    constant     uint&  h_k        [[buffer(6)]],
    constant     uint&  h_v        [[buffer(7)]],
    constant     uint&  s          [[buffer(8)]],
    constant     uint&  conv_k     [[buffer(9)]],
    constant     uint&  n_tokens   [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    const uint qkv_dim = 2 * h_k * s + h_v * s;
    if (gid >= qkv_dim) return;

    const uint q_dim = h_k * s;
    const uint k_dim = h_k * s;
    const uint w_base = gid * conv_k;

    for (uint tok = 0; tok < n_tokens; ++tok) {
        float acc = 0.0f;
        for (uint t = 0; t + 1 < conv_k; ++t) {
            acc += conv_state[t * qkv_dim + gid] * conv1d[w_base + t];
        }

        const float mixed = qkv_mixed[tok * qkv_dim + gid];
        acc += mixed * conv1d[w_base + (conv_k - 1)];

        if (conv_k >= 2) {
            for (uint t = 0; t + 2 < conv_k; ++t) {
                conv_state[t * qkv_dim + gid] = conv_state[(t + 1) * qkv_dim + gid];
            }
            conv_state[(conv_k - 2) * qkv_dim + gid] = mixed;
        }

        const float sig = 1.0f / (1.0f + exp(-acc));
        const float val = acc * sig;

        if (gid < q_dim) {
            q_out[tok * q_dim + gid] = val;
        } else if (gid < q_dim + k_dim) {
            k_out[tok * k_dim + gid - q_dim] = val;
        } else {
            v_out[tok * (h_v * s) + gid - q_dim - k_dim] = val;
        }
    }
}

// Token-parallel exact form of qwen35_recurrent_conv_shift_chunk. It reads the
// initial conv_state for all tokens, writes token-major Q/K/V in parallel, then
// a companion kernel commits the final conv_state.
kernel void qwen35_recurrent_conv_shift_chunk_token_parallel(
    device const float* conv_state [[buffer(0)]],
    device const float* qkv_mixed  [[buffer(1)]],
    device const float* conv1d     [[buffer(2)]],
    device       float* q_out      [[buffer(3)]],
    device       float* k_out      [[buffer(4)]],
    device       float* v_out      [[buffer(5)]],
    constant     uint&  h_k        [[buffer(6)]],
    constant     uint&  h_v        [[buffer(7)]],
    constant     uint&  s          [[buffer(8)]],
    constant     uint&  conv_k     [[buffer(9)]],
    constant     uint&  n_tokens   [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    const uint qkv_dim = 2 * h_k * s + h_v * s;
    const uint total = n_tokens * qkv_dim;
    if (gid >= total) return;

    const uint tok = gid / qkv_dim;
    const uint d = gid - tok * qkv_dim;
    const uint w_base = d * conv_k;
    const int history = int(conv_k) - 1;

    float acc = 0.0f;
    for (uint kt = 0; kt < conv_k; ++kt) {
        const int src_pos = int(tok) + int(kt) - history;
        float x;
        if (src_pos < 0) {
            x = conv_state[(src_pos + history) * int(qkv_dim) + int(d)];
        } else {
            x = qkv_mixed[uint(src_pos) * qkv_dim + d];
        }
        acc += x * conv1d[w_base + kt];
    }

    const float sig = 1.0f / (1.0f + exp(-acc));
    const float val = acc * sig;

    const uint q_dim = h_k * s;
    const uint k_dim = h_k * s;
    if (d < q_dim) {
        q_out[tok * q_dim + d] = val;
    } else if (d < q_dim + k_dim) {
        k_out[tok * k_dim + d - q_dim] = val;
    } else {
        v_out[tok * h_v * s + d - q_dim - k_dim] = val;
    }
}

kernel void qwen35_recurrent_conv_shift_chunk_token_parallel_state(
    device       float* conv_state [[buffer(0)]],
    device const float* qkv_mixed  [[buffer(1)]],
    constant     uint&  qkv_dim    [[buffer(2)]],
    constant     uint&  conv_k     [[buffer(3)]],
    constant     uint&  n_tokens   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint state_rows = conv_k - 1;
    const uint total = state_rows * qkv_dim;
    if (gid >= total) return;

    const uint row = gid / qkv_dim;
    const uint d = gid - row * qkv_dim;
    const int src_pos = int(n_tokens) + int(row) - int(state_rows);
    if (src_pos < 0) {
        conv_state[row * qkv_dim + d] = conv_state[uint(src_pos + int(state_rows)) * qkv_dim + d];
    } else {
        conv_state[row * qkv_dim + d] = qkv_mixed[uint(src_pos) * qkv_dim + d];
    }
}

// Same as qwen35_recurrent_conv_shift_chunk, but also copies conv_state after
// `checkpoint_index` has been applied. This is the exact recurrent-boundary
// checkpoint needed by branch-guard verifier fusion; callers still own whether
// the copied checkpoint is worth paying for.
kernel void qwen35_recurrent_conv_shift_chunk_checkpoint(
    device       float* conv_state      [[buffer(0)]],
    device const float* qkv_mixed       [[buffer(1)]],
    device const float* conv1d          [[buffer(2)]],
    device       float* q_out           [[buffer(3)]],
    device       float* k_out           [[buffer(4)]],
    device       float* v_out           [[buffer(5)]],
    constant     uint&  h_k             [[buffer(6)]],
    constant     uint&  h_v             [[buffer(7)]],
    constant     uint&  s               [[buffer(8)]],
    constant     uint&  conv_k          [[buffer(9)]],
    constant     uint&  n_tokens        [[buffer(10)]],
    device       float* checkpoint      [[buffer(11)]],
    constant     uint&  checkpoint_index [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    const uint qkv_dim = 2 * h_k * s + h_v * s;
    if (gid >= qkv_dim) return;

    const uint q_dim = h_k * s;
    const uint k_dim = h_k * s;
    const uint w_base = gid * conv_k;

    for (uint tok = 0; tok < n_tokens; ++tok) {
        float acc = 0.0f;
        for (uint t = 0; t + 1 < conv_k; ++t) {
            acc += conv_state[t * qkv_dim + gid] * conv1d[w_base + t];
        }

        const float mixed = qkv_mixed[tok * qkv_dim + gid];
        acc += mixed * conv1d[w_base + (conv_k - 1)];

        if (conv_k >= 2) {
            for (uint t = 0; t + 2 < conv_k; ++t) {
                conv_state[t * qkv_dim + gid] = conv_state[(t + 1) * qkv_dim + gid];
            }
            conv_state[(conv_k - 2) * qkv_dim + gid] = mixed;
        }

        if (tok == checkpoint_index && conv_k >= 2) {
            for (uint t = 0; t + 1 < conv_k; ++t) {
                checkpoint[t * qkv_dim + gid] = conv_state[t * qkv_dim + gid];
            }
        }

        const float sig = 1.0f / (1.0f + exp(-acc));
        const float val = acc * sig;

        if (gid < q_dim) {
            q_out[tok * q_dim + gid] = val;
        } else if (gid < q_dim + k_dim) {
            k_out[tok * k_dim + gid - q_dim] = val;
        } else {
            v_out[tok * (h_v * s) + gid - q_dim - k_dim] = val;
        }
    }
}

// Same as qwen35_recurrent_conv_shift_chunk, but qkv_mixed is already
// rounded to F16 by the upstream quantized GEMM. The old path expanded that
// F16 output to F32 and immediately read it here, so reading half directly
// preserves values while avoiding the expansion buffer.
kernel void qwen35_recurrent_conv_shift_chunk_h16(
    device       float* conv_state [[buffer(0)]],
    device const half*  qkv_mixed  [[buffer(1)]],
    device const float* conv1d     [[buffer(2)]],
    device       float* q_out      [[buffer(3)]],
    device       float* k_out      [[buffer(4)]],
    device       float* v_out      [[buffer(5)]],
    constant     uint&  h_k        [[buffer(6)]],
    constant     uint&  h_v        [[buffer(7)]],
    constant     uint&  s          [[buffer(8)]],
    constant     uint&  conv_k     [[buffer(9)]],
    constant     uint&  n_tokens   [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    const uint qkv_dim = 2 * h_k * s + h_v * s;
    if (gid >= qkv_dim) return;

    const uint q_dim = h_k * s;
    const uint k_dim = h_k * s;
    const uint w_base = gid * conv_k;

    for (uint tok = 0; tok < n_tokens; ++tok) {
        float acc = 0.0f;
        for (uint t = 0; t + 1 < conv_k; ++t) {
            acc += conv_state[t * qkv_dim + gid] * conv1d[w_base + t];
        }

        const float mixed = float(qkv_mixed[tok * qkv_dim + gid]);
        acc += mixed * conv1d[w_base + (conv_k - 1)];

        if (conv_k >= 2) {
            for (uint t = 0; t + 2 < conv_k; ++t) {
                conv_state[t * qkv_dim + gid] = conv_state[(t + 1) * qkv_dim + gid];
            }
            conv_state[(conv_k - 2) * qkv_dim + gid] = mixed;
        }

        const float sig = 1.0f / (1.0f + exp(-acc));
        const float val = acc * sig;

        if (gid < q_dim) {
            q_out[tok * q_dim + gid] = val;
        } else if (gid < q_dim + k_dim) {
            k_out[tok * k_dim + gid - q_dim] = val;
        } else {
            v_out[tok * (h_v * s) + gid - q_dim - k_dim] = val;
        }
    }
}

kernel void qwen35_l2_heads(
    device       float* x      [[buffer(0)]],
    constant     uint&  s      [[buffer(1)]],
    constant     float& eps    [[buffer(2)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig;
    device float* head = x + h * s;

    float ss = 0.0f;
    for (uint d = tiisg; d < s; d += 32) {
        float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum + eps);

    for (uint d = tiisg; d < s; d += 32) {
        head[d] *= inv;
    }
}

kernel void qwen35_l2_heads_chunk(
    device       float* x        [[buffer(0)]],
    constant     uint&  n_heads  [[buffer(1)]],
    constant     uint&  s        [[buffer(2)]],
    constant     float& eps      [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_heads) return;

    device float* head = x + (t * n_heads + h) * s;

    float ss = 0.0f;
    for (uint d = tiisg; d < s; d += 32) {
        float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum + eps);

    for (uint d = tiisg; d < s; d += 32) {
        head[d] *= inv;
    }
}
