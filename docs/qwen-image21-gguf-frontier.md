# Qwen-Image 2.1 GGUF Frontier

Status: active implementation frontier (2026-09-22)

## Goal

Admit Qwen-Image 2.1 weights into the native Metal engine without treating a
file name such as `Q4` as evidence of its actual tensor policy.

## Admitted now

- Parse GGUF `qwen_image` metadata without mapping the tensor payload.
- Support GGML BF16 tensor type 30 in byte sizing and CPU dequantization.
- Inventory actual per-tensor quantization types and storage sizes.
- Recover and validate ComfyUI `comfy.gguf.orig_shape.<tensor>` metadata.
- Validate the exact 32-block Qwen-Image 2.1 DiT tensor inventory.
- Report whether every tensor type is readable by the current Cogni-ML GGUF
  dequantization layer.
- Execute a parameterized one-block CPU reference with the upstream 2.1
  equations: affine-less LayerNorm, per-head Q/K RMSNorm, three-axis RoPE,
  block-causal attention, shared tanh modulation gates, and fused SwiGLU.
- Multiply BF16 GGUF matrices directly in the CPU reference path.
- Check the complete tiny block against an independent PyTorch oracle.
- Reject truncated GGUFs before exposing any mmap-backed tensor slice.
- Recover mathematical projection dimensions from Comfy `orig_shape` metadata
  while preserving ordinary GGUF `[in, out]` dimension semantics.
- Load all top-level and 32 transformer-block weights as zero-copy mmap slices,
  with exact per-projection shape validation and explicit mmap lifetime.
- Reuse the Qwen 3.5 Q8/Q5/Q6 Metal projection kernels inside the exact block
  reference. On two real tokens through block 0, the CPU/Metal boundary produced
  `max_abs=1.5258789e-5` and cosine `0.999999999999972` across 8192 outputs.
- Execute the exact batch-one outer transformer contract: zero-centered text
  RMSNorm and projection, VLM image-slot expansion, condition/target latent
  substitution, shape-derived image block ids, centered three-axis positions,
  padding validity, sinusoidal timestep embedding, causal `t=0` prefix
  modulation, all transformer blocks, adaptive output norm, and output head.
- Check that complete outer forward against an independent tiny PyTorch oracle,
  including the appended target placeholder slots used by the official
  pipeline. Adjacent image slots are explicitly kept as separate attention
  blocks, and prefix output is timestep-independent under causal conditioning.
- Execute the complete outer transformer with all 32 real mixed-quant blocks
  for a minimum valid `2x2` target. The measured hybrid run used 192 Metal block
  projections, produced finite output, and completed in approximately `3.7`
  seconds through the original projection-only route; BF16 top-level
  projections and outer orchestration still used the CPU fallback.
- Keep a complete DiT block sequence Metal-resident behind one outer-transformer
  backend call. Affine-less LayerNorm and modulation, per-head Q/K RMSNorm,
  three-axis RoPE, segmented block-causal attention, SwiGLU, residual updates,
  and all six mixed-quant projections per block are encoded into one command
  buffer with shared scratch and ping-pong hidden buffers.
- Execute all 32 real mixed-quant blocks with one command buffer, zero
  intermediate readbacks, 192 projection dispatches, and one final hidden-state
  readback. Against the prior hybrid reference on the minimum valid `2x2`
  target, the outer output produced `max_abs=2.771616e-6` and cosine
  `0.9999999999998054`. A real single-block check produced
  `max_abs=0.00012588501` and cosine `0.999999999999791`.
- Cache the causal prefix K/V tensors for every DiT layer in persistent Metal
  buffers after the first transformer evaluation. Later FlowMatch evaluations
  recompute only the contiguous target suffix while assembling full attention
  K/V on-device and retaining one command buffer, zero intermediate readbacks,
  and one final readback. A real 32-layer check with an unchanged text prefix,
  changed target latents, and changed timestep recomputed four of five tokens
  and matched a complete hybrid reference with `max_abs=5.401671e-6` and cosine
  `0.9999999999988506`.
- Invalidate the prefix cache fail-closed when its token boundary, prefix hidden
  state, prefix modulation, prefix positions, prefix image ids, prefix key
  validity, layer objects, or block configuration changes. A tiny GPU check
  verifies both the cache-hit path and rebuild after a changed prefix value.
- Reproduce the model's configured deterministic FlowMatch Euler schedule:
  linear input sigmas, exponential resolution shift over the exact
  `256..8192` sequence-length range, terminal stretching to `0.02`, and Euler
  updates. A two-step model-backed loop executed 384 Metal projections across
  two complete 32-block evaluations and produced finite changed latents.

## Pinned bootstrap artifact

- Repository: `realrebelai/Qwen-Image-2.1_GGUFs`
- Revision: `8d393b750593a72ee040fe7d40611f479ccee679`
- File: `Qwen-Image-2.1-Q4.gguf`
- Bytes: `5959127264`
- SHA-256: `51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`
- Actual tensor policy: `BF16:8,F32:65,Q8_0:96,Q6_K:64,Q5_K:32`

The repository README describes this policy as its Qwen-specific mixed Q4
release even though its recommended long `Q4_K_M-HQv3` label does not match the
current short filename. The loader trusts the inspected tensor directory, not
either label.

## Not admitted by this slice

- A production-scale, end-to-end resident Metal pipeline, text encoders, VAE,
  or decoded image generation. The admitted DiT block stack is resident, but
  sequence construction, timestep/text/top-level BF16 projections, final norm
  and output projection still cross the CPU boundary once per transformer
  evaluation.
- A representative-token performance claim for the current exact attention
  kernel. The admitted kernel is correctness-first and remains quadratic in
  token count; the minimum `2x2` target check is not a throughput benchmark.
- A compressed or bounded-memory prefix cache. The admitted implementation
  stores per-layer prefix K/V as F32 Metal buffers and therefore trades memory
  for repeated-step projection savings.
- A claim that a readable GGUF has acceptable image quality.
- A custom weight format derived from the resident-KV adaptive QBit codec.
- Trusting repository or file labels (`Q4`, `dynamic`, `HQ`) over tensor data.

## Guard and next transition

The next implementation transition is a batch-capable top-level BF16 Metal
projection route so sequence construction can stop crossing the CPU boundary
around the resident block stack. Representative-token profiling must then
separate projection, attention, cache-copy, and launch costs before changing
the correctness-first attention kernel. Text encoding, sampling, and VAE decode
remain separate frontiers.

The model-backed checks are:

```bash
QWEN_IMAGE21_GGUF=/path/to/Qwen-Image-2.1-Q4.gguf \
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  crystal spec spec/qwen_image21_flow_match_spec.cr \
    spec/qwen_image21_transformer_spec.cr \
    spec/qwen_image21_weights_spec.cr spec/qwen_image21_metal_spec.cr \
    spec/qwen_image21_resident_metal_spec.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"
```

## Falsifiers

- A real candidate contains an unimplemented tensor type.
- A `comfy.gguf.orig_shape.*` entry changes the tensor element count.
- Any required top-level or per-block tensor is absent, duplicated, or appears
  under an incompatible architecture.
- A full candidate is shorter than the last tensor extent declared in its
  directory.
- Any real block projection lacks a strict Metal route or exceeds the declared
  CPU/Metal tolerance.
- A resident stack uses more than one command buffer or performs an
  intermediate host readback for one outer-transformer evaluation.
- Prefix caching changes a prefix hidden state, key, or value when only the
  target latents and FlowMatch timestep change.
- A changed prefix input or block configuration reuses the prior prefix cache
  instead of rebuilding it.
- A later image-quality corpus shows that the selected mixed quantization
  policy is worse than its declared baseline.
