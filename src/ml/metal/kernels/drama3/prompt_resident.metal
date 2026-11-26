#include <metal_stdlib>
using namespace metal;

// Reference encoder output is codebook-major [books, frames]. Each thread
// validates one ID and writes its part of one row, keeping codes device-side.
kernel void dr3_prompt_reference_rows(
    device const float* codes [[buffer(0)]],
    device float* rows [[buffer(1)]],
    device atomic_uint* validity [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    uint frames = uint(p[0]);
    uint books = uint(p[1]);
    uint width = uint(p[2]);
    uint row_start = uint(p[3]);
    uint semantic_begin = uint(p[4]);
    uint codebook_size = uint(p[5]);
    if (id >= frames * books) return;

    uint book = id / frames;
    uint frame = id % frames;
    float code = codes[id];
    bool valid = isfinite(code) && floor(code) == code && code >= 0.0f && code < float(codebook_size);
    if (!valid) atomic_store_explicit(validity, 0x7fc00000u, memory_order_relaxed);
    float safe_code = valid ? code : 0.0f;
    uint output = (row_start + frame) * width;
    rows[output + book + 1] = safe_code;
    if (book == 0) rows[output] = float(semantic_begin) + safe_code;
}

// Upstream's repetition window is the final sixteen prompt main IDs. Gather
// them from the completed device prompt, including audio reference rows.
kernel void dr3_prompt_main_window(
    device const float* rows [[buffer(0)]],
    device float* window [[buffer(1)]],
    constant ulong* p [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    uint prompt_rows = uint(p[0]);
    uint width = uint(p[1]);
    uint window_rows = uint(p[2]);
    if (id < window_rows) window[id] = rows[(prompt_rows - window_rows + id) * width];
}
