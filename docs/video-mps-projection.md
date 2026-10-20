# Prepared MPS projection experiment

The optional prepared MPS path expands resident BF16 weights into one reusable
F32 GPU buffer, performs F32 matrix multiplication and adds the typed bias on
GPU. Activations remain F32. Plans and buffers are prepared before encoding and
must remain live until the caller's terminal completion boundary. This is a
projection experiment with bounded block integration, not a video benchmark.

## Numerical and lifetime checks

`spec/video_linear_mps_projection_spec.cr` uses the pinned PyTorch 2.8.0 MPS
`torch.nn.functional.linear` fixture produced by
`spec/support/video_linear_tiled_reference.py`. CPU neural fallback is disabled.
It selects all four weight/bias F32/BF16 combinations with F32 activations at
three shapes: `[19,19,13]`, `[3,5120,5]` and `[3,13824,5]` (rows, input width,
output width). Every value must be finite and satisfy the existing
`2e-5 + 2e-5*abs(reference)` gate. Scalar and tiled controls use the same inputs.
The spec calls the production `MetalLinearMPS.project` typed-bias kernel;
nonzero typed biases are checked separately from the cost experiment.

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
| 64 | 2.621 | 17.753 | 6.77 |
| 512 | 7.741 | 136.606 | 17.65 |
| 2,048 | 26.488 | 537.399 | 20.29 |

The MPS interval includes BF16-to-F32 weight expansion, projection and a GPU
typed-bias add. The tiled interval includes its own bias operation. Synthetic
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
`video-linear-mps-production-r{64,512,2048}-native.log` and
`mps-full-projection-native.log`. Runs were serial under `scripts/run_safe.sh`
with a 15% free-memory floor. The complete projection spec can be reproduced
with the prepared video bridge objects, `VIDEO_LINEAR_TILED_MPS_FIXTURE` set
to the pinned fixture, `VIDEO_LINEAR_MPS_COST_PROBE=1`,
`VIDEO_LINEAR_MPS_COST_ROWS` set to a table row, and
`VIDEO_LINEAR_MPS_COST_REPEATS=4`.

The fixture covers model input widths but small output widths. The wide cost
probe is a native-versus-native synthetic control. The optional
`WanS2VMPSLinearWorkspace` prepares ten block and five audio-injector plans,
sharing one 283,115,520-byte F32 weight stage at official widths. Normal and
drop-motion workspaces may share this stage when encoded serially. Fixed plan
bindings and nonoverlap checks reject changed scratch identities; callers must
retain all plans, buffers and views through terminal completion. Release
methods document that completion precondition rather than mechanically proving
it. Scalar projection remains the default and rollback.

The official small-sequence block/injector fixture passes all six examples
with both F32 and BF16 resident parameters using prepared MPS projections,
tiled attention and asynchronous continuations. The tiled-attention 40-layer
exact-prefix comparison captures all 104 points but still fails the unchanged
strict gate: 23 values out of 31,948,800, maximum absolute drift 2.059937e-4.
See `docs/video-numerical-drift.md`.
The later optional query-chunked attention route has its own fresh stack receipt
and bounded completion evidence in `docs/video-attention-mps.md`.

The full-spatial block originally failed Metal completion with `Impacting
Interactivity`. A controlled full-attention discriminator at Q=K=36,608,
H=40, D=128 failed with 4,096 query-tile/head groups per command root and
completed with 320. The latter checks all 187,432,960 synthetic output values
against their exact analytical result: zero violations, 572 segments and one
terminal wait. Its measured summed GPU interval is 29.853 s. The chain overload
now defaults to 320 groups; query reductions and arithmetic are unchanged.

With this boundary, one real BF16 block 0 plus audio injector 0 at full
sequence length completes in a measured 36.952 s summed GPU interval, with
1,148 segments, one terminal host wait, no activation readback, and finite
outputs checked on GPU. The explicit estimate including a 1 GiB reserve is
16,298,549,764 bytes. Its procedural inputs are not a denoising trajectory or
an upstream full-shape fixture. Receipts are `attention-full-g{4096,320}-native.log`
and `dit-full-shape-mps-g320-probe.log` in the external evidence directory.
No full-video speed or quality conclusion follows from these intervals.
