#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#pragma clang fp contract(off)
#ifndef WTP_TM
#define WTP_TM 32
#endif
#ifndef WTP_TN
#define WTP_TN 32
#endif

// Probe only. The right operand has the exact same cooperative layout that a
// future QBit loader must populate. No global dense staging or relaxed precision.
kernel void wan_tensorops_dense_f32(
    device float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint3 &shape [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]]) {
  constexpr int TM = WTP_TM, TN = WTP_TN, TK = 32;
  constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  const uint M = shape.x, K = shape.y, N = shape.z;
  const uint row = group.y * TM, col = group.x * TN;
  auto a = tensor(input, dextents<int, 2>{int(K), int(M)}, array<int, 2>{1, int(K)});
  auto dst = tensor(output, dextents<int, 2>{int(N), int(M)}, array<int, 2>{1, int(N)});
  auto b = op.get_right_input_cooperative_tensor<float, float, float>();
  auto ap = a.slice(0, int(row));
  auto accum = op.get_destination_cooperative_tensor<decltype(ap), decltype(b), float>();
  // The factory and run() type-check the right-input layout. The separate
  // is_compatible_as_right_input API tests a *destination* tensor for reuse;
  // it cannot be applied to an input tensor returned by this factory.
  #pragma clang loop unroll(full)
  for (uint i = 0; i < accum.get_capacity(); ++i)
    if (accum.is_valid_element(i)) accum[i] = 0.0f;
  for (uint k = 0; k < K; k += TK) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < b.get_capacity(); ++i) {
      if (b.is_valid_element(i)) {
        auto c = b.get_multidimensional_index(i);
        uint n = col + uint(c[0]), kk = k + uint(c[1]);
        b[i] = n < N && kk < K ? weight[ulong(n) * K + kk] : 0.0f;
      }
    }
    auto panel = a.slice(int(k), int(row));
    op.run(panel, b, accum);
  }
  #pragma clang loop unroll(full)
  for (uint i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      auto c = accum.get_multidimensional_index(i);
      uint n = col + uint(c[0]);
      if (n < N) accum[i] = accum[i] + bias[n];
    }
  }
  auto out = dst.slice(int(col), int(row));
  accum.store(out);
}
