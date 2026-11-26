# Qwen-Image 2.1 CUDA generation frontier

Status: IN_PROGRESS (2026-10-17). The verified starting point is commit
`b146f03a`: one resident block on the RTX 5060 Ti, including all 16 operator
checkpoints. Full image generation and its latency are not yet measured.

## Admitted next slice

Build an opt-in, batch-one text-to-image DiT runner over the existing GGUF
weights. All transformer layers, input/time/output projections and FlowMatch
updates must consume device buffers on one stream. Allocate weights and scratch
at setup and reuse them across layers and denoising steps. Prompt conditioning
and final image encoding are explicit request boundaries; an upstream CUDA VAE
may consume the final device latents. A cached-conditioning benchmark must be
identified as such and cannot stand for full text-to-PNG latency.

Keep the existing Metal/package defaults. Reject CPU activation round trips,
per-layer/per-step completion waits, unsafe in-flight scratch reuse, and claims
of complete CUDA generation from a one-block or CPU-only check. Cached prompt
inputs are allowed for an explicitly scoped DiT measurement; they do not verify
a CUDA text encoder. Native CUDA VAE implementation is outside this slice.

## Discriminating checks

Risk tier: CAUTION. The strongest failures are incorrect target/text timestep
selection, scratch aliasing across layers, accumulated numerical drift, and a
benchmark omitting expensive stages. Before a real image timing is promoted:

1. Execute a tiny official-transformer comparison, multiple Euler steps, and a
   changed-input scratch-reuse guard. Tiny max absolute error remains <=2e-5.
2. Compare real operator checkpoints with the pinned Diffusers revision and
   identical dequantized GGUF weights. Retain the existing strict real-block
   bound (<1e-2 at each checkpoint; output cosine >0.999999). Check late blocks
   on identical inputs; do not widen bounds to excuse accumulation.
3. Run all 32 layers and a complete fixed-seed denoising trajectory. Record
   trajectory error separately from teacher-forced operator parity; any failed
   existing gate blocks a verified claim.
4. Generate and inspect an actual image. Report resolution, steps, seed,
   precision, conditioning/VAE path, cold setup and warm denoise timing. Measure
   on the GPU with only existing terminal boundaries, excluding diagnostic
   checkpoints from the timed route. Repeat a warm run to expose first-run cost.
5. Execute existing CUDA block/projection graph replay regressions and attack
   the new interface's size, alias, and lifetime guards.

DoD commands will be the new native generation probe plus its pinned official
oracle, followed by a real text/image request on the authorized CUDA host.
Expected signal: passing unchanged numerical gates, finite final latents and a
valid PNG, with measured stage and request timings. Source compilation and
CPU tests alone leave this frontier IN_PROGRESS.

Rollback: remove the opt-in CUDA generation runner and probe. No default switch
is admitted. Weights, source revision, arithmetic, stream/lifetime handling,
driver or fixture changes invalidate the corresponding numerical/timing result.

## Current external boundary

On 2026-10-17 the previously authorized SSH endpoint returned
`websocket: bad handshake` twice before a remote command could run. This blocks
fresh CUDA execution and timing until the host is reachable. Local preparation
and CPU-side contract tests can continue.

## Continuation capsule (user frontier switch, 2026-10-17)

The user confirmed the NVIDIA host is offline after a hackathon and explicitly
returned to DRAMA-3. Stop CUDA implementation and execution here. No production
source edits or new test files were made; this draft is the only untracked file.
The verified one-block commit remains `b146f03a` on
`codex/qwen-image21-cuda-block`.

Resume at shared block scratch/device modulation support, then a full streamed
transformer/FlowMatch runner and a tiny multiblock/multistep official oracle.
Use one reusable checkpoint workspace and hidden ping-pong buffers rather than
32 private workspaces. A proposed runner would accept external conditioning and
initial latent device pointers and expose its final latent pointer. This API is
not implemented. For a future Torch/VAE bridge, use the caller's current CUDA
context and stream; `Context.create` currently creates a separate context and
must not be used to consume Torch pointers. Q8 projection currently launches
once per token, so its image-scale launch cost is an open performance risk.

Next falsifier: unchanged tiny <=2e-5 full-transformer/multistep GPU gate plus
changed-input scratch reuse, followed by real same-input operator comparisons.
No full-model accuracy, image, or timing result exists. Existing authority is
limited to the authorized development CUDA host, reversible opt-in work and
benchmarks; refresh host access and memory state when it returns. Never infer
full-pipeline correctness or speed from the six-token one-block fixture.
