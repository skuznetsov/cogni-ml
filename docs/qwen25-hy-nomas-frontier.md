# Qwen2.5-VL text decoder for Hy-Nomas

Status: the serial four-token native BF16 block slice passes its declared
qualification, 2026-10-24. The full text encoder and video pipeline remain
in progress. This adapter is opt-in.

## Contract and source

Hy-Nomas source revision `911cea6458a89c2663513ff7099561ec190dd491`
selects Qwen/Qwen2.5-VL-7B-Instruct with BF16 precision, right padding,
attention masks, 1280 text tokens, `hidden_state_skip_layer=2` and
`apply_final_norm=false`. The Hugging Face hidden-state tuple contains the
embedding output followed by decoder-layer outputs, with the last output
normalized by the stack's final RMSNorm. For 28 layers,
`hidden_states[-3]` is therefore the output of block 25. Only blocks 0--25
are needed; neither the last two blocks nor the final RMSNorm, vision tower
or language-model head contribute to that selected state.

This slice consumes resident BF16 embeddings, additive causal/padding mask,
F32 positions and rotary inverse frequencies, and resident BF16 decoder
weights. It reuses CogniQwen's Metal buffers and the engine's serial MPS
command chain. Qwen3.5's hybrid forward pass and weights are not substitutes
for Qwen2.5-VL's biased Q/K/V, split-half rotary embedding and grouped-query
causal attention. Quantized CogniQwen kernels remain a later qualification.

All neural operations execute on GPU. Preparation may upload source bytes,
mask values and positional constants. Encoding must neither allocate named
Metal buffers nor read activations nor wait for GPU completion. Distinct
blocks are linked through resident output/input buffers in one command
chain. The caller owns the sole terminal wait and diagnostic readback.
Prepared plans and their bound buffers remain retained until that wait.
Serial attempts to reuse a plan on another in-flight chain and premature
release fail closed. Concurrent host calls are unsupported; there is no lock.
Borrowed bindings must not be overwritten or released in a way that races
scheduled consumers before the terminal drain. An ordered earlier GPU producer
on the same chain may populate the next block's input. Generic MetalBuffer
write/release APIs do not enforce this caller precondition. The adapter is currently
a diagnostic plan that retains every declared stage, with a per-plan byte
budget. It does not admit the aggregate memory of a full prefix or account for
MPS-internal allocations. At 1280 tokens, 26 diagnostic plans would retain
about 5.6 GiB of stage buffers and 2.1 GiB of BF16 boundary scratch before
weights and MPS-internal scratch. A production prefix needs reusable scratch
and aggregate admission first.

## Qualification and falsifiers

The unchanged finite gate is `abs(error) <= 2e-5 + 2e-5*abs(reference)`.
Compare input normalization, biased Q/K/V, rotary outputs, attention output,
output projection, both residuals, post-attention normalization and all FFN
stages. The oracle must execute the official Transformers decoder on MPS;
locally copied equations cannot establish upstream parity. Record exact
source hashes, runtime versions, dtype, mask and attention implementation.
BF16 storage alone does not prove equivalence of reduction order, fused
operations or SDPA. A mismatch must stop promotion and identify the first
divergent stage; never widen tolerances to make a path pass.

Use one-layer and two-layer fixtures with different successor weights, a
nontrivial position offset, right padding, and a mutation that proves the
comparator rejects an incorrect value. A small synthetic model proves only the
tested geometry and data. One four-token block at 3584/18944/28Q/4KV/128-head
geometry now passes too. Pretrained weights, longer sequences, the full
26-block prefix, tokenization, prompt templates, cropping and integration with
the DiT remain separate qualification boundaries. The admitted masks are right
padded and every query row has at least one allowed key. The finite BF16 0/min
native bias is qualified only for that class; arbitrary and fully masked rows
are not qualified.

CAUTION: numerical/lifetime changes are isolated to new opt-in files. The
strongest failure modes are crossed BF16 rounding boundaries, incorrect GQA
ordering, accidentally using one block's weights twice, and early freeing of
in-flight resources. Guards validate geometry, exact bindings, storage,
device, lengths and disjointness before encoding. Rollback is removal of this
opt-in adapter and its focused build target; existing model paths stay intact.

DoD: run the Python oracle tests and guarded native specs using
`examples/support/wan_memory_guard.py --memory-accounting file-cache`, a
bounded process footprint, bounded swap growth, and the unchanged 15% reserve.
Expected signals: every declared intermediate passes the finite gate,
invalid/alias bindings fail before encoding, no named-buffer allocation is
observed during encoding, and wait count changes from zero to one only at
the terminal drain. Record tested scope and residual mismatches below.
No full generation, Studio or full-model allocation is admitted by this slice.
Temporary artifacts and compiler caches live under
`/opt/mldata_ssd/cogni-ml/qwen25-hy-nomas-20261024`.

## Rejected numerical route

The first one-graph candidate failed the unchanged gate on a tiny official
MPS BF16 fixture: input RMSNorm and biased Q/K/V matched exactly, but Q RoPE
failed 16/96 elements (maximum absolute error 0.0078125). Both rotary cos/sin
outputs matched exactly, rejecting trigonometric values as that first cause.
A two-block fixture with non-unit RMS gains diverged earlier, at input RMSNorm
(23/96 elements, maximum 0.0078125). Explicit F32--BF16--F32 cast round trips
inside the same graph did not change either result. Such casts are not an
empirically valid precision fence in this tested compilation. Evidence:
`logs/native-roundtrip.out` and `logs/native-roundtrip.jsonl` under the artifact
root. Materializing source BF16 operation boundaries in caller-owned buffers
between asynchronous executables eliminated the RMSNorm and RoPE mismatches.
The next first divergence was attention (29/96 elements, maximum 0.00390625),
recorded in `logs/native-materialized.out`. An all-F32 QK/softmax/PV graph does
not reproduce the general BF16 MPS SDPA path. Matching its typed graph sequence
closed that mismatch without a host wait or wider tolerance. Both padded test
geometries dispatch to PyTorch's general MPSGraph SDPA, not its vector shader.
The resulting block encodes 15 prepared executables, retaining seven BF16
boundary scratch buffers as part of its admitted workspace.

## Observed qualification

The official oracle uses PyTorch 2.8.0 and Transformers 4.57.3 on MPS BF16,
with CPU fallback disabled. Hy-Nomas declares Transformers 4.57.1; the producer
records the mismatch and hashes the installed source. Its selected Qwen2.5-VL,
RoPE, activation, mask and SDPA math is byte-equivalent to the pinned 4.57.1
source; no equivalence for unrelated library code is claimed. Geometry comes
from Qwen/Qwen2.5-VL-7B-Instruct revision
`eb6ba55b40ada64977147b52e50a4145ee89ac4f`. These are official constructor
weights, not pretrained checkpoint weights. Every fixture uses four tokens,
mask `[1,1,1,0]`, and position offset 7.

| Fixture | Layers | Dimensions D/FF/Q/KV/HD | RMS gains | BF16 stage comparisons |
| --- | ---: | --- | --- | ---: |
| `tiny-1layer.json` | 1 | 24/48/2/1/12 | unit | 17 |
| `tiny-2layer.json` | 2, distinct weights | 24/48/2/1/12 | unit | 34 |
| `tiny-2layer-nonunit.json` | 2, distinct weights | 24/48/2/1/12 | nonunit | 34 |
| `real7b-s4.json` | 1 | 3584/18944/28/4/128 | unit | 17 |

All 102 stage comparisons pass bit exactly (`max_abs=0`) at the unchanged
finite gate. Each request reports one terminal wait and zero tracked named
MetalBuffer allocations during encoding. The two-block tests pass the exact
resident output object to the successor without CPU readback. Invalid bindings,
overlap, workspace budget, early release and sequential second-chain reuse are
exercised; the comparator rejects a one-ULP mutation and nonfinite values.
Python fixture/sidecar tests pass 22 examples. Crystal CPU configuration,
sidecar and comparator checks pass four examples. Native compilation passes
`-Wall -Wextra -Werror`. Correlated MiniQuad review: ROBUST for this bounded,
serial diagnostic scope, not for arbitrary inputs or production prefix use.

The guarded real-geometry oracle peaks at 2,565,818,240 sampled process-group
bytes; the native Makefile run peaks at 829,557,304 bytes. Minimum cache-aware
headroom is 23,983,620,096 and 26,133,790,720 bytes respectively; memory-pressure
free percentages stay at least 71 and 75. Swap growth is zero. These are sampled
observations, not opaque MPS peak bounds. The external 3 GiB process-group cap,
15% reserve, 64 MiB swap-growth stop and 600-second deadline remain intact.
No speedup or full-generation throughput claim follows from these tests.

Evidence under the artifact root: `logs/native-sdpa-general-v2.*` (one-layer and
nonunit two-layer gate), `logs/native-real7b.*` (Makefile target: six examples,
including unit two-layer and real geometry), and `logs/real7b-s4-oracle.*`.
Source, fixture, bridge-object and runtime pins are indexed in
`logs/final-receipt-v1.json`. The Crystal consumer treats environment-supplied
fixtures as trusted input; it does not authenticate their upstream provenance.
Verify their recorded hashes before using a rerun as an oracle qualification.
The producer pins and validates source/runtime identity and sidecar contents.
The final fresh rebuild, `logs/native-final-clean-v2.*`, passes all eight
Crystal examples and all four fixtures together. The earlier clean-run attempt
used a mistyped fixture directory and failed file lookup; it is retained as a
harness error, not numerical evidence. Crystal's Makefile spec runner deletes
its temporary executable after exit, so the receipt pins its compiled bridge
objects, source and command rather than claiming a retained executable hash.

Reproduce only under the guard with SSD-backed TMPDIR/CRYSTAL_CACHE_DIR:

```sh
python3 -B spec/qwen25_hy_nomas_reference_spec.py
python3 -B examples/support/wan_memory_guard.py \
  --memory-accounting file-cache --max-footprint-mib 3072 \
  --max-swap-growth-mib 64 --timeout-sec 600 --log FRESH.jsonl -- \
  env COGNI_QWEN25_REFERENCES=TRUSTED_FIXTURE_PATHS \
      COGNI_QWEN25_REPORT_DIR=EXTERNAL_REPORT_DIRECTORY \
  make spec_qwen25 BUILD_DIR=EXTERNAL_BUILD_DIRECTORY
```

`COGNI_QWEN25_REFERENCES` is a colon-separated list of the recorded fixture
paths. Oracle production uses the pinned interpreter and exact arguments in
the receipt; the real-geometry BF16 sidecar is 466,875,424 bytes including
weights, source inputs and expected stages. No fixture or generated binary is
added to Git. The next discriminator is a bounded pretrained block chain,
followed by the 26-block memory plan and Hy-Nomas prompt/cropping bridge.

Evidence decays when upstream source, source dtype, attention policy,
geometry, compiler, OS/MPS runtime, weights or buffer lifecycle changes;
refresh the corresponding oracle and guarded native tests before promotion.
