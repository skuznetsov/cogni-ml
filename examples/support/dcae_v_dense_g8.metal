// Appended to dcae_v_k_partials.metal for the sampled dense probe.
constant uint G8 = 8u, KernelTaps = 27u, Outside = 0xffffffffu;

kernel void dense_g8_samples(device const ushort* joined,
                             device const ushort* weights,
                             device const uint* indices,
                             device float* output,
                             uint sample [[thread_position_in_grid]]) {
  const uint coord = indices[sample];
  const uint oc = coord / (T * H * W);
  const uint t = coord / (H * W) % T;
  const uint source_stride = (T + 2u) * H * W;
  const int out_y = int(coord / W % H);
  const int out_x = int(coord % W);
  uint position[KernelTaps];
  for (uint tap = 0u; tap < KernelTaps; ++tap) {
    const int y = out_y + int(tap / 3u % 3u) - 1;
    const int x = out_x + int(tap % 3u) - 1;
    position[tap] = (y >= 0 && y < int(H) && x >= 0 && x < int(W))
        ? ((t + tap / 9u) * H + uint(y)) * W + uint(x)
        : Outside;
  }

  float acc = 0.0f;
  const device ushort* row = weights + oc * K;
  #pragma clang loop unroll(disable)
  for (uint channel = 0u; channel < C; channel += G8) {
    #pragma clang loop unroll(disable)
    for (uint tap = 0u; tap < KernelTaps; ++tap) {
      if (position[tap] == Outside) continue;
      const uint src = channel * source_stride + position[tap];
      const uint weight = channel * KernelTaps + tap;
      #pragma clang loop unroll(full)
      for (uint lane = 0u; lane < G8; ++lane) {
        acc = metal::fma(from_bf16(joined[src + lane * source_stride]),
                         from_bf16(row[weight + lane * KernelTaps]), acc);
      }
    }
  }
  output[sample] = acc;
}
