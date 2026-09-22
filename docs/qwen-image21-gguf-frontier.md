# Qwen-Image 2.1 GGUF Frontier

Status: active implementation frontier (2026-09-21)

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
- Execute all 32 real mixed-quant blocks plus the outer head on Metal for a
  minimum valid `2x2` target. The measured run used 192 Metal block projections,
  produced finite output, and completed in approximately `3.7` seconds; BF16 top-level
  projections still used the CPU fallback.
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

- A production-scale, fully resident Metal DiT, text encoders, VAE, or decoded
  image generation. The admitted route moves quantized projection matmuls to
  Metal; block orchestration, attention, elementwise work, and top-level BF16
  projections remain on the CPU.
- A claim that a readable GGUF has acceptable image quality.
- A custom weight format derived from the resident-KV adaptive QBit codec.
- Trusting repository or file labels (`Q4`, `dynamic`, `HQ`) over tensor data.

## Guard and next transition

The next implementation transition is a Metal-native block with resident
intermediates and segmented block-causal attention, followed by prefix KV
caching across denoising steps. Top-level BF16 projections need a batch-capable
route before the full DiT can avoid CPU fallback. Text encoding, sampling, and
VAE decode remain separate frontiers.

The model-backed checks are:

```bash
QWEN_IMAGE21_GGUF=/path/to/Qwen-Image-2.1-Q4.gguf \
  crystal spec spec/qwen_image21_flow_match_spec.cr \
    spec/qwen_image21_transformer_spec.cr \
    spec/qwen_image21_weights_spec.cr spec/qwen_image21_metal_spec.cr \
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
- A later image-quality corpus shows that the selected mixed quantization
  policy is worse than its declared baseline.
