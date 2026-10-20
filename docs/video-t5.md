# Native Wan UMT5 conditioning

The native UMT5 encoder consumes original BF16 checkpoint storage and performs
F32 neural arithmetic on GPU. Setup prepares resident scratch, MPSNDArray
projection plans, MPSGraph attention and RMS-mean plans. GPU conversion reuses
one F32 weight workspace rather than retaining a second F32 model.

The qualified arithmetic route uses source-order GELU, an explicitly rounded
square/mean RMS path and rank-three batched attention matching the pinned
PyTorch 2.8 MPS execution. Native encoding pauses and resumes the caller's
command chain around MPSGraph work and refreshes borrowed encoders. FC2
encodes on the currently open native command/encoder pair. Callers retain
plans, arenas, input buffers and weights until the one terminal wait.
Early explicit release or dropping a plan is outside this ownership contract;
the API does not enforce an in-flight lifetime guard.

## Executed upstream checks

The production gate executes all 24 layers with 512 context tokens, channel
width 4,096, FFN width 10,240 and 64 heads. All 39 selected intermediate
points, including the complete masked output, pass the unchanged finite gate
`abs(actual - expected) <= 2e-5 + 2e-5 * abs(expected)`.
The spec observes one terminal wait and zero encode-time `MetalBuffer`
allocations. MPS implementation-private allocations are outside that counter.

`spec/wan_t5_full_context_spec.cr` passes one example, zero failures, errors
or pending cases. The final small regression in `spec/wan_t5_spec.cr` passes
three examples, including repeated resident execution, long-distance relative
position buckets, shape/input guards and rejection of partial tracing without
a destination map. Partial trace capture otherwise requires explicit known
destinations including both embedding and output.

Evidence lives under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/`:
`t5-full-context-native.log`, `t5-full-context.json` and
`t5-final-regression-native.log`. The small regression uses `t5-graph-long.json`
with `WAN_T5_MPS_ATTENTION=1` and `WAN_T5_SOURCE_GELU=1`. The full gate uses
`WAN_T5_FULL_CONTEXT_FIXTURE=t5-full-context.json` with its hash-checked
checkpoint and trace sidecar. Link all three T5 bridge objects and the existing
Metal and MPS command-chain bridges. Run serially with the 15% memory floor.

The scoped adversary verdict is ROBUST for this serial retained-plan route;
premature release and unmeasured private workspace remain limitations.
Evidence decays with source, checkpoint, arithmetic-route or plan-ownership
changes. Re-run the corresponding gate before reusing its parity claim.
These encoder checks establish neither complete-video quality nor throughput.
