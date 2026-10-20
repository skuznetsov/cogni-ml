# Prepared MPS projection experiment

The optional prepared MPS path expands resident BF16 weights into one reusable
F32 GPU buffer, performs F32 matrix multiplication and adds the typed bias on
GPU. Activations remain F32. Plans and buffers are prepared before encoding and
must remain live until the caller's terminal completion boundary. This is a
projection experiment, not a completed Wan block or video benchmark.

## Numerical and lifetime checks

`spec/video_linear_mps_projection_spec.cr` uses the pinned PyTorch 2.8.0 MPS
`torch.nn.functional.linear` fixture produced by
`spec/support/video_linear_tiled_reference.py`. CPU neural fallback is disabled.
It selects all four weight/bias F32/BF16 combinations with F32 activations at
three shapes: `[19,19,13]`, `[3,5120,5]` and `[3,13824,5]` (rows, input width,
output width). Every value must be finite and satisfy the existing
`2e-5 + 2e-5*abs(reference)` gate. Scalar and tiled controls use the same inputs.
Nonzero typed biases are checked separately from the cost experiment.

One fixed plan and weight stage also execute BF16 weights A, distinct weights B,
then A again in one command. The saved A results must agree exactly, B must
differ, and no explicit `MetalBuffer` may be allocated while encoding. Reads
occur after the sole terminal wait. These checks do not count MPS-private
temporary storage or host objects.

## Measured scope

On Apple M2 Max, four alternating resident-weight pairs at input width 5,120
and output width 13,824 produced these median GPU intervals:

| Rows | MPS, ms | Tiled, ms | Tiled / MPS |
| --- | ---: | ---: | ---: |
| 64 | 2.674 | 18.383 | 6.87 |
| 512 | 7.687 | 139.172 | 18.11 |
| 2,048 | 31.188 | 422.577 | 13.55 |

The MPS interval includes BF16-to-F32 weight expansion, projection and a GPU
row-bias add. The tiled interval includes its own bias operation. Synthetic
cost biases are zero; upstream nonzero-bias parity is established by the
separate fixture. Each cost output is checked for finiteness and agreement
between native routes. Each invocation completed two examples with zero
failures, errors or pending examples. No cache purge was performed.

A separate full-shape probe at `[36608,5120,13824]` completed two GPU intervals
of 493.14 and 491.10 ms. Those intervals include weight expansion, MPS projection
and a finite scan of all 506,068,992 output values; they exclude bias addition.
Four selected binary-exact synthetic results also match their analytical sums.
Explicit buffers total 3,198,681,092 bytes. This establishes completion of that
isolated dispatch, not full-output upstream parity or the cost of a Wan block.

Evidence is external under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/`:
`video-linear-mps-final-r{64,512,2048}-native.log` and
`mps-full-projection-native.log`. Runs were serial under `scripts/run_safe.sh`
with a 15% free-memory floor. The complete projection spec can be reproduced
with the prepared video bridge objects, `VIDEO_LINEAR_TILED_MPS_FIXTURE` set
to the pinned fixture, `VIDEO_LINEAR_MPS_COST_PROBE=1`,
`VIDEO_LINEAR_MPS_COST_ROWS` set to a table row, and
`VIDEO_LINEAR_MPS_COST_REPEATS=4`.

The fixture covers model input widths but small output widths. The wide cost
probe is a native-versus-native synthetic control. Selecting this route for
Wan invalidates the previous stack comparison as a bound: refresh exact-prefix
intermediates and record any finite drift before experimental generation.
No full-video speed or quality conclusion follows from these intervals.
