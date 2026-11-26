# Prepared query-chunked Wan attention

`WanAttentionMPSChunkWorkspace` is an opt-in B=1 self-attention route. It keeps
full F32 Q/K/V/output resident, copies at most 1,024 query rows into fixed GPU
scratch, runs a prepared MPSGraph SDPA executable, and copies valid result rows
back on the same native command chain. The final short chunk is padded on GPU.
K/V remain global: chunks do not restrict the attention window.

An optional final pair of `head_major_k` / `head_major_v` buffers enables
head-major K/V reuse. One fused GPU dispatch converts current BTHD K/V to BHTD
at the start of **every** workspace `encode`; all query chunks then consume
those prepared buffers. The pair is caller-owned, shared F32, full-size and
pairwise disjoint from every original/chunk buffer. At K=36,608, H=40, D=128
it adds 1,499,463,680 bytes to `scratch_bytes`. This reuses a layout within a
call, never stale activations across blocks or diffusion steps. Omitting both
buffers retains the original graph path. Direct `transpose_kv!` callers must
serialize access and retain all four bindings through terminal completion.

The final `explicit_qk_softmax_pv: true` argument selects a prepared F32
QK/scale/softmax/PV graph instead of the SDPA graph operator. It requires the
head-major pair and follows the pinned PyTorch MPS math sequence, including its
`1e-20f` addition before scaling. The default is false. Both graph variants
use the same ownership, scratch and terminal-completion rules. Workspaces that
share a writable prepared K/V pair must be serialized by the caller through
terminal completion; their separate in-flight guards do not enforce mutual
exclusion. The new mode's qualification and score-workspace admission are
tracked in [wan-self-attention-benchmark.md](wan-self-attention-benchmark.md).

The caller chooses 1..1,024 query rows; existing callers retain 512 by default.
Three caller-owned chunk buffers cost 31,457,280 bytes at C=5,120 and 512 rows,
or 62,914,560 bytes at 1,024 rows. Admission must account for the chosen size.
MPSGraph may allocate private workspace; explicit buffer counts and current
allocation snapshots do not establish a private-memory peak. There is no CPU
activation copy or intermediate host wait. All views and plans must outlive
terminal completion. A prepared workspace may serve a fresh request only after
the preceding chain completes. Tiled attention remains the default and rollback.

Block and transformer entrypoints validate the chain, dimensions and exact
scratch bindings before dispatch. The transformer's guard regression uses empty
weights so earlier accidental phase encoding would fail with a different error.

## Executed checks

The unchanged numerical gate is `2e-5 + 2e-5*abs(reference)`, with finite operands.
`spec/video_attention_mps_spec.cr` compares six complete pinned PyTorch 2.8.0 MPS
SDPA cases, including K=36,608, near ties, sharp logits, B=2 and single-key
attention. Its chunked Q=777 and Q=1,289 / K=36,608 cases exercise 512- and
1,024-row chunks. They use different query rows in successive chunks, cover the
short tail, initialize output to NaN, and compare every
result to the correspondingly indexed oracle. GPU A/B/A input mutation checks
and a later completed-chain reuse reject stale prepared bindings. Alias, offset,
alignment, overflow and lifetime guards are also exercised.

The earlier head-major cohort completed five examples without failures, errors
or pending cases. Those checks cover exact B=2 head-major indexing, separately changed
K and V in one chain, restored inputs and reuse after a completed chain, plus
alias/extent and in-flight reuse/release guards. The official BF16 block fixture
also passes six regression examples at its pinned upstream intermediate gate.
The earlier shared-scratch transformer/long-key
regression completed eight examples without failures, errors or pending cases.
Each attention request has
one terminal wait and zero explicit MetalBuffer allocations during encoding.
The explicit-graph cohort subsequently passes seven focused examples and the
same six official BF16 regression examples. Full-shape synthetic attention is
bit-identical to its pinned PyTorch oracle, but the real-weight full-block A/B
warmup rejects the graph substitution at 5,085 values under the unchanged gate.
No measured block pair or speedup is admitted. The explicit graph remains
experimental with its default false; details and source bindings are in the
benchmark document above.
The actual 40-layer BF16 stack comparison stays numerically rejected; its fresh
query-chunked receipt is recorded in `docs/video-numerical-drift.md` for the
user-authorized experiment, rather than promoted to strict parity.

A separate full-shape probe executes all 72 query chunks at B=1,
Q=K=36,608, H=40, D=128. GPU-filled zero Q/K and binary-exact row-constant V
produce a known analytical result. All 187,432,960 output values pass the GPU
scan exactly, including the last chunk. It completes 13 command segments and
one terminal wait with zero explicit encode allocations. Explicit buffers use
3,036,241,920 bytes; current device allocation moves from 3,036,659,712 to
3,047,129,088 bytes. Its summed command GPU interval is 11.961 seconds; this is
not a wall-clock video latency or a matched end-to-end speedup. The control does
not establish full-shape upstream parity with actual model activations.

## Evidence and refresh

External evidence remains under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/`:
`attention-mps-chunk-native.log`, `wan-transformer-attention-chunk-native.log`,
`attention-full-mps-chunk-native.log`, their build logs, and the full-shape probe
source `attention-full-mps-chunk-probe.cr`. All GPU runs were serialized under
`scripts/run_safe.sh` with a configured 15% `memory_pressure -Q` threshold.
That telemetry plus RSS does not bound footprint or swap. After the later memory
incident the old video process was terminated with explicit user authorization;
heavy runs are suspended pending stronger memory admission.

Build the focused spec with `make spec_wan`'s prepared bridge objects; set
`WAN_ATTENTION_MPS_FIXTURE` to the pinned `attention-tiled-qualified/manifest.json`.
The stack comparison selects `WAN_S2V_MPS_CHUNK_ATTENTION=1` in addition to
prepared projections. Source, device/runtime, geometry, checkpoint, fixed buffer
identity or arithmetic changes invalidate the corresponding evidence and require
fresh checks. Visible quality and complete cold/repeat video timing remain open.

The nontrivial full-geometry upstream comparison and isolated attention timing
are described in [wan-self-attention-benchmark.md](wan-self-attention-benchmark.md).
Its `head-kv-*-v1.log` receipts and `head-kv-qualified-summary-v1.json` live in
the sibling `evidence/wan-self-attention/` directory. Full attention outputs pass
the pinned upstream gate and whole-block candidate outputs match the native
control bit for bit. The head-major layout candidate wins only two of three
whole-block pairs (medians 18.227773 to 18.091101 seconds), failing performance
admission while adding 1.396 GiB of explicit buffers. It remains optional.
