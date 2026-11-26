# Opt-in F32 tiled video projection

`MetalLinearTiled` adapts the existing DR3 8-by-8 SIMD-group matrix kernel.
Inputs, weights and bias independently use F32 or BF16 storage. BF16 values
are expanded on GPU; matrix operands, accumulation and output remain F32.
The scalar `MetalOps.linear` implementation remains the default.

Prepare the pipeline before encoding and retain every buffer until the
caller's terminal completion. Aligned row ranges preserve absolute row
indices and each row's ascending K traversal. `linear_bounded` continues
asynchronously between row ranges on a caller-owned `MPSCommandChain`;
callers must refresh borrowed encoders after it returns. It adds no host
wait, activation readback or `MetalBuffer` creation during encode. Metal
service objects and implementation-private allocations are not counted.

## Executed evidence

On Apple M2 Max, `spec/video_linear_tiled_spec.cr` passes five examples with
zero failures, errors or pending cases. This covers all eight independent
storage triples against pinned PyTorch 2.8 MPS `functional.linear` at three
geometries: tails at 19-by-19-by-13 and cancellation-heavy reductions with
K=5,120 and K=13,824. The finite per-value gate remains
`abs(actual - expected) <= 2e-5 + 2e-5 * abs(expected)`.

Nonuniform row ranges and three continued command roots are bitwise equal
to the corresponding unsplit native kernel. The continued case observes
zero encode-time waits, one terminal wait and zero encode-time buffer
allocations. Invalid ranges, storage sizes and an otherwise valid input/
output alias are rejected before dispatch.

The bounded cost probe uses 64 rows, K=5,120, N=13,824, a resident
141,557,760-byte synthetic BF16 weight matrix and F32 input/bias/output.
Four samples alternate tiled/scalar and scalar/tiled order, retaining
buffers and weights. GPU medians are 18.274 ms tiled and 135.890 ms scalar,
a local ratio of 7.436. The measurement includes no explicit cache purge;
the first tiled sample is slower than the following samples. This is a
synthetic projection result, not full spatial throughput or video timing.

Evidence is under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/`:
`video-linear-tiled-mps.json`, `video-linear-tiled-final-build.log` and
`video-linear-tiled-final-native.log`. Reproduce with that fixture through
`VIDEO_LINEAR_TILED_MPS_FIXTURE`, `VIDEO_LINEAR_TILED_COST_PROBE=1`,
`VIDEO_LINEAR_TILED_COST_ROWS=64` and
`VIDEO_LINEAR_TILED_COST_REPEATS=4`, linking the existing Metal bridge and
MPS command-chain objects. Run serially with the 15% system-memory floor.

The scoped adversary verdict is ROBUST for tested storage, row coverage,
tails and serial lifetime assumptions. Full spatial watchdog behavior,
real-checkpoint stack arithmetic and full request speed remain unqualified.
Selecting this backend changes the reduction route: refresh intermediate
comparisons before treating an earlier scalar-route drift ledger as
applicable. Rollback uses the unchanged scalar path.
