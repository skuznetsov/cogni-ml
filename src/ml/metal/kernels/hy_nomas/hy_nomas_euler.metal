#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

static inline ushort hy_nomas_euler_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return ushort((bits >> 16u) | 0x0040u);
    }
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static inline float hy_nomas_euler_round_bf16_f32(float value) {
    return as_type<float>((uint)hy_nomas_euler_round_bf16(value) << 16u);
}

kernel void hy_nomas_euler_round_initial_latent_bf16(
    device float *latent [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < count) latent[tid] = hy_nomas_euler_round_bf16_f32(latent[tid]);
}

kernel void hy_nomas_euler_copy_timestep(
    const device float *schedule [[buffer(0)]],
    device float *timestep [[buffer(1)]],
    constant uint &step [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) timestep[0] = schedule[step * 2u];
}

kernel void hy_nomas_euler_update_latent(
    device float *latent [[buffer(0)]],
    const device float *velocity [[buffer(1)]],
    const device float *schedule [[buffer(2)]],
    constant uint &step [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
    const float model_velocity = hy_nomas_euler_round_bf16_f32(velocity[tid]);
    const float delta = model_velocity * schedule[step * 2u + 1u];
    const float updated = latent[tid] + delta;
    latent[tid] = hy_nomas_euler_round_bf16_f32(updated);
}
