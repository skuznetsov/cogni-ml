# Native F32/D128 long-key attention probe

`wan_steel_f32_d128.metal` is a byte-identical copy of the measured external
`steel_f32_d128_candidate-v8.metal` shader. It is opt-in probe source, compiled
by `ComputePipeline.new` at setup. The measured path uses native Metal and does
not load the MLX runtime. Engine defaults are unchanged.

- SHA-256: `b3e27825b3723d80789e6b73acecff0a2a4e66380691d18dce827a3cf3d8664f`.
- Size: 125,484 bytes. Preserve the bytes when refreshing historical evidence.
- Upstream: [MLX STEEL](https://github.com/ml-explore/mlx/tree/c6abf0dbb4fc141fad739b32ae486c06728e49e5/mlx/backend/metal/kernels/steel/attn).
- Upstream commit: `c6abf0dbb4fc141fad739b32ae486c06728e49e5`.
- License: MIT; see [MLX-LICENSE.txt](MLX-LICENSE.txt). Apple copyright notices
  remain in the flattened source. Local adaptations retain that license.

The source inlines eighteen MLX headers and its attention translation unit
because the native compiler accepts a complete source string without a project
include search path. It specializes F32, D128, BQ32/BK32, WM4/WN2, with masks,
sinks and optional outputs disabled. B=1, four-head scheduling and the adapter's
2,048-key continuation are the qualified configuration. Each row keeps all
keys in ascending order; unnormalized output and row maximum/sum live in
preallocated GPU buffers between ordered roots. Normalize only the final shard.

The v8 adaptation adds F32 Kahan compensation when merging BK32 PV contributions
within each shard. Scale the correction with the same online-softmax frame as
the output; fold it before spilling. QK, exponentials and the denominator remain
unchanged. The two additional output fragments cost 32 F32 scalars per thread;
there are no additional buffers, dispatches, allocations or host waits. This
does not promise exact real arithmetic or compensate every inner MMA operation.

The first two shader comments describe its initial preparation state. They are
preserved for byte identity and are superseded by the dated validation below.
Historical runs read the external file, not this repository path; copying its
bytes does not constitute another timing run.

On 2026-10-22, M2 Max / 64 GiB, retained upstream checkpoints and full-key
Q8192/Q257 controls passed the unchanged finite
`2e-5 + 2e-5*abs(reference)` gate. On real block-0 BF16 weights with F32
synthetic activations, two fresh full-block processes gave paired geometric time
reductions of 17.79% and 18.02% versus MPS G4/C128. Every warmup and measured
comparison passed all 187,432,960 values. Host latency drift was substantial;
these are matched relative results, not a stable absolute latency claim.

See [the performance frontier](../../../docs/wan-performance-frontier.md#compensated-native-long-key-attention-2026-10-22)
for measurements, guards, scope and refresh instructions. Evidence is external
under `/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/steel-f32-20261022/`.
Future invocations must pass this repository file explicitly and record/check
the compile-input SHA. A new shader, geometry, dtype, resource/queue policy,
device, SDK or driver requires fresh numerical and matched timing gates.
