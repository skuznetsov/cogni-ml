# Qwen3.5 State Indirection Audit

Date: 2026-05-27

This note maps the TokenSpeed state-slot idea onto the current native Qwen3.5/3.6 runtime. The goal is not to copy TokenSpeed's CUDA/Blackwell implementation, but to reuse the invariant: move state ownership by indices when possible, not by copying recurrent tensors after every speculative verifier pass.

## External Anchor

The PyTorch/TokenSpeed Qwen3.5 post describes three transferable patterns:

- Prefix-cache nodes own clean recurrent/GDN checkpoints, not just KV page IDs.
- Speculative verifier state recovery writes candidate states into dedicated physical rows, then updates a lightweight current-state index after accepted length is known.
- Host/device syncs and scalar readbacks are treated as critical-path bugs; scheduler metadata and index arithmetic should stay fused or device-side when possible.

The benchmark numbers are not comparable to this local M2 Max/Metal runtime. The state lifecycle invariant is the useful part.

## Current Local State Model

`Qwen35CPU::State` owns one `LayerState` per layer. Full-attention layers have KV buffers, recurrent layers have convolution and SSM buffers.

Current branch copy primitive:

```text
copy_state_metal_used!(dst, src, used_tokens)
  full-attn: copy live KV rows only
  recurrent: copy full conv_state + full ssm_state
```

This is correct and bounded for KV, but recurrent state is copied in full for every branch snapshot/restore. That matches our old exactness requirements, but it is the opposite of TokenSpeed's accepted-state index move.

## Hot Copy/Replay Sites

### 1. Product MTP reject path

File: `bin/qwen35_generate.cr`

Current shape:

```text
backup_state <- copy_state_metal_used!(state)
verifier mutates state over verify_tokens
if reject:
  state <- copy_state_metal_used!(backup_state)
  replay accepted prefix + correction to rebuild boundary
```

This is the highest-value state-indirection candidate because it sits in product MTP and is exactly the pattern TokenSpeed targets.

Boundary risk: this path must preserve exact greedy parity and KV/SSM state at the correction boundary. We cannot switch to index movement unless full-attention KV and recurrent state row ownership are both coherent at the same accepted boundary.

### 2. Sidecar wall harness checkpoint/replay path

File: `bin/qwen35_mtp_sidecar_probe.cr`

Current options:

```text
rec_checkpoint_replay: restore recurrent checkpoint after row 0
rec_rollback_log: compact DeltaNet row-1 rollback, explicit research-only
fallback replay: restore backup then replay accepted prefix + correction
```

This harness is the right sandbox for index indirection because it already has parity gates and records `backup_ms`, `replay_ms`, `verifier_ms`, and `fallback_ms`.

Boundary risk: `rec_rollback_log` previously failed JSON parity when composed with top2-rescue continuation. Any new index-row path needs the same incompatibility guards until proved across adversary prompts.

### 3. Recurrent prefill checkpoint helpers

File: `src/ml/gguf/qwen35_cpu.cr`

Current helpers:

```text
prefill_tokens_hidden_top1s_recurrent_checkpoint
prefill_tokens_hidden_top1s_recurrent_rollback_log
prefill_tokens_top1s_recurrent_checkpoint
```

These can capture or restore recurrent checkpoints while doing known-span verifier work. They are useful correctness surfaces, but not yet an index-indirection implementation: checkpoint buffers are still physical copies.

### 4. Active cursor / prompt cache

Files: `src/ml/gguf/qwen35_resident_session.cr`, `src/ml/gguf/qwen35_serving_route.cr`, `src/ml/gguf/qwen35_prompt_cache.cr`

These are already close to the TokenSpeed prefix-cache framing: validated state artifacts and active cursors are the high-level equivalent of clean reusable slots. The gap is lifecycle and same-process product routing, not tensor math.

## Candidate Design: State Slot Arena

Introduce an opt-in arena concept for recurrent state first:

```text
StateSlotArena
  slot[request, layer, slot_id].conv
  slot[request, layer, slot_id].ssm
  current_slot[request, layer]
```

For a verifier chunk of length K:

```text
input slot = current_slot
for verifier row i:
  kernel reads slot[input_i]
  kernel writes slot[draft_i]
after verification:
  current_slot = draft[accepted_boundary]
```

No recurrent tensor copy is needed for accepted-state recovery. Recovery becomes one index update.

## Why This Is Not Just a Refactor

The current Metal recurrent kernels take concrete `conv_bufs` and `ssm_bufs`. They mutate those buffers in place. To make slot indirection real, kernels need either:

1. multiple per-layer state buffers and CPU chooses which buffer pointers are passed per verifier row, or
2. a larger state arena plus GPU-side slot-index parameters, so the kernel computes offsets internally.

Option 1 is easier but may still require multiple command encodes. Option 2 matches TokenSpeed better but is a kernel/API change.

## LTP/WBA Framing

Window:
- speculative verifier/reject path where accepted boundary is known only after exact verification.

Transport:
- recurrent state ownership across candidate rows.

Legal move:
- write each candidate row's state into a distinct slot and update canonical slot index after accepted length is known.

Boundary safety:
- exact greedy output parity, full-attention KV boundary consistency, recurrent conv/SSM clean-slot invariant, no aliasing between mutable working slot and published checkpoint slot.

Potential:
```text
Phi = (state_copy_bytes, replay_tokens, verifier_passes, host_syncs, wall_ms)
```
A legal move must reduce `state_copy_bytes` or `replay_tokens` without increasing parity failures or verifier passes.

Dual frame:
- current `copy_state_metal_used!` + checkpoint replay remains the exact fallback.

## Recommended Next Experiment

Do not start with product `qwen35_generate`. Start in `bin/qwen35_mtp_sidecar_probe.cr`.

1. Add a default-off `--mtp-spec-wall-state-slot-probe` mode.
2. Restrict to `stage=2`, `top2_rescue_continue=false`, `rec_rollback_log=false`.
3. Allocate two or three recurrent checkpoint states per verifier pass.
4. Preserve the old backup path for full-attention KV initially.
5. On first-row reject, select the row-0 checkpoint state instead of restoring backup and replaying token 0.
6. Measure: parity, `backup_ms`, `replay_tokens`, `replay_ms`, `wall_ms`.

This is not yet true GPU-side index indirection, but it is the smallest proof that accepted-boundary state selection can replace replay. If it wins and preserves parity, move from multiple `State` objects to a real recurrent slot arena.

## Non-Candidates For Now

- More isolated Q4/Q6 B2 GEMV kernels: already refuted as live verifier wins.
- Static late-layer skip: layer-stability atlas refutes broad default.
- Token-window Hadamard overlap tuning: exact overlap is redundant with n-gram; partial overlap needs a different session/template corridor.

## Current Decision

Next implementation should be a sidecar harness experiment for accepted-boundary recurrent state selection. Product path changes wait until the harness proves parity and positive wall contribution.
