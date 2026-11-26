#include <metal_stdlib>
using namespace metal;

inline int q8_packed_load(device const uint* q, uint index) {
    uint word = q[index >> 2];
    uint byte = (word >> ((index & 3) * 8)) & 0xffu;
    int raw = int(byte);
    return raw > 127 ? raw - 256 : raw;
}

kernel void ffn_pca_updown_fused_rows_q8(
    device const float* x              [[buffer(0)]],
    device const float* x_mean         [[buffer(1)]],
    device const float* c_mean         [[buffer(2)]],
    device const uint*  coeff_weights  [[buffer(3)]],
    device const float* coeff_scales   [[buffer(4)]],
    device const uint*  down_basis     [[buffer(5)]],
    device const float* down_scales    [[buffer(6)]],
    device       float* out            [[buffer(7)]],
    constant     uint&  hidden_dim     [[buffer(8)]],
    constant     uint&  rank           [[buffer(9)]],
    constant     uint&  n_tokens       [[buffer(10)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (token >= n_tokens) return;

    threadgroup float partials[64 * 8];
    threadgroup float coeffs[64];
    device const float* row = x + token * hidden_dim;

    uint lanes = (rank <= 32) ? 8 : 4;
    uint coeff = tid / lanes;
    uint lane = tid - coeff * lanes;
    if (coeff < rank && coeff < 64 && lane < lanes) {
        float scale = coeff_scales[coeff];
        float acc = 0.0f;
        uint row_base = coeff * hidden_dim;
        for (uint d = lane; d < hidden_dim; d += lanes) {
            acc += (row[d] - x_mean[d]) * (float(q8_packed_load(coeff_weights, row_base + d)) * scale);
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
            acc += coeffs[j] * (float(q8_packed_load(down_basis, j * hidden_dim + d)) * down_scales[j]);
        }
        out_row[d] = acc;
    }
}
