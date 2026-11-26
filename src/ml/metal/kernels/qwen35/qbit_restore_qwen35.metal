#include <metal_stdlib>
using namespace metal;

constant uint QWEN_QBIT_P7_POSITIVE_CENTROID_BITS[64] = {
    0x3c2137cbu, 0x3cf1dbd9u, 0x3d499a25u, 0x3d8d2d69u,
    0x3db59c1cu, 0x3dde1d58u, 0x3e035aabu, 0x3e17b432u,
    0x3e2c1d6fu, 0x3e40989fu, 0x3e55280bu, 0x3e69ce0cu,
    0x3e7e8d0eu, 0x3e89b3c9u, 0x3e943016u, 0x3e9ebcc7u,
    0x3ea95b42u, 0x3eb40cfcu, 0x3ebed37eu, 0x3ec9b061u,
    0x3ed4a557u, 0x3edfb429u, 0x3eeadeb8u, 0x3ef62707u,
    0x3f00c79cu, 0x3f068ccau, 0x3f0c6445u, 0x3f124f5cu,
    0x3f184f7bu, 0x3f1e662au, 0x3f249511u, 0x3f2ade02u,
    0x3f3142fbu, 0x3f37c629u, 0x3f3e69fau, 0x3f453121u,
    0x3f4c1ea5u, 0x3f5335f4u, 0x3f5a7afcu, 0x3f61f246u,
    0x3f69a12bu, 0x3f718e08u, 0x3f79c099u, 0x3f812130u,
    0x3f858f9fu, 0x3f8a331cu, 0x3f8f1532u, 0x3f94420au,
    0x3f99c91eu, 0x3f9fbe05u, 0x3fa6394fu, 0x3fad597au,
    0x3fb543ffu, 0x3fbe2689u, 0x3fc83887u, 0x3fd3bd81u,
    0x3fe10908u, 0x3ff085beu, 0x4001615bu, 0x400c4727u,
    0x40199826u, 0x402a7bfdu, 0x40416dd8u, 0x4064c7a3u,
};

kernel void qwen35_qbit_p7_decode_f32(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& block_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }

    const uint block = gid / block_size;
    const uint within = gid - block * block_size;
    const uint plane_bytes = block_size / 8;
    const uint block_stride = 8 + 7 * plane_bytes;
    const uint base = block * block_stride;

    uint mean_bits = ((uint)src[base]) |
                     (((uint)src[base + 1]) << 8) |
                     (((uint)src[base + 2]) << 16) |
                     (((uint)src[base + 3]) << 24);
    uint sigma_bits = ((uint)src[base + 4]) |
                      (((uint)src[base + 5]) << 8) |
                      (((uint)src[base + 6]) << 16) |
                      (((uint)src[base + 7]) << 24);

    const uint byte_offset = plane_bytes - 1 - within / 8;
    const uchar bit_mask = (uchar)(1u << (within & 7u));
    uint raw_code = 0;
    for (uint plane = 0; plane < 7; ++plane) {
        const uint plane_offset = base + 8 + plane * plane_bytes;
        if ((src[plane_offset + byte_offset] & bit_mask) != 0) {
            raw_code |= 1u << (7u - plane);
        }
    }

    const float mean = as_type<float>(mean_bits);
    const float sigma = as_type<float>(sigma_bits);
    const uint prefix = raw_code >> 1;
    float centroid;
    if (prefix < 64) {
        centroid = as_type<float>(QWEN_QBIT_P7_POSITIVE_CENTROID_BITS[prefix]);
    } else {
        centroid = -as_type<float>(QWEN_QBIT_P7_POSITIVE_CENTROID_BITS[127u - prefix]);
    }
    dst[gid] = mean + sigma * centroid;
}

kernel void qwen35_qbit_p7_native_decode_f32(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& block_size [[buffer(3)]],
    constant uint& row_start [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& mean_offset [[buffer(6)]],
    constant uint& sigma_offset [[buffer(7)]],
    constant uint& codes_offset [[buffer(8)]],
    uint group [[thread_position_in_grid]]) {
    const uint groups_per_tile = block_size / 8;
    const uint tile = group / groups_per_tile;
    const uint group_in_tile = group - tile * groups_per_tile;
    const uint output_base = tile * block_size + group_in_tile * 8;
    if (output_base >= count) {
        return;
    }

    const uint row = row_start + tile;
    const uint mean_base = mean_offset + row * 4;
    const uint sigma_base = sigma_offset + row * 4;
    const uint mean_bits = ((uint)src[mean_base]) |
                           (((uint)src[mean_base + 1]) << 8) |
                           (((uint)src[mean_base + 2]) << 16) |
                           (((uint)src[mean_base + 3]) << 24);
    const uint sigma_bits = ((uint)src[sigma_base]) |
                            (((uint)src[sigma_base + 1]) << 8) |
                            (((uint)src[sigma_base + 2]) << 16) |
                            (((uint)src[sigma_base + 3]) << 24);

    const uint plane_bytes = block_size / 8;
    const uint plane_stride = row_count * plane_bytes;
    const uint byte_offset = plane_bytes - 1 - group_in_tile;
    uchar plane_values[7];
    for (uint plane = 0; plane < 7; ++plane) {
        plane_values[plane] = src[codes_offset + plane * plane_stride + row * plane_bytes + byte_offset];
    }

    const float mean = as_type<float>(mean_bits);
    const float sigma = as_type<float>(sigma_bits);
    for (uint lane = 0; lane < 8 && output_base + lane < count; ++lane) {
        const uchar bit_mask = (uchar)(1u << lane);
        uint raw_code = 0;
        for (uint plane = 0; plane < 7; ++plane) {
            if ((plane_values[plane] & bit_mask) != 0) {
                raw_code |= 1u << (7u - plane);
            }
        }

        const uint prefix = raw_code >> 1;
        float centroid;
        if (prefix < 64) {
            centroid = as_type<float>(QWEN_QBIT_P7_POSITIVE_CENTROID_BITS[prefix]);
        } else {
            centroid = -as_type<float>(QWEN_QBIT_P7_POSITIVE_CENTROID_BITS[127u - prefix]);
        }
        dst[output_base + lane] = mean + sigma * centroid;
    }
}
