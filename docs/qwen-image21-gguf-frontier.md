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

## Not admitted by this slice

- DiT execution, text encoders, VAE, scheduler, or image generation.
- A claim that a readable GGUF has acceptable image quality.
- A custom weight format derived from the resident-KV adaptive QBit codec.
- Trusting repository or file labels (`Q4`, `dynamic`, `HQ`) over tensor data.

## Guard and next transition

The next implementation transition is a one-block CPU/Metal parity harness for
the Qwen-Image 2.1 transformer. It is legal only after an actual candidate GGUF
passes this inventory check.

## Falsifiers

- A real candidate contains an unimplemented tensor type.
- A `comfy.gguf.orig_shape.*` entry changes the tensor element count.
- Any required top-level or per-block tensor is absent, duplicated, or appears
  under an incompatible architecture.
- A later image-quality corpus shows that the selected mixed quantization
  policy is worse than its declared baseline.
