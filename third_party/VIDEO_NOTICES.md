# Native video source and license notices

The native video modules are opt-in. External checkpoints retain their own terms.

## Hy-Nomas

The Hy-Nomas core block and oracle adapter are based on Tencent-Hunyuan/Hy-Nomas
at `911cea6458a89c2663513ff7099561ec190dd491`.
Upstream: <https://github.com/Tencent-Hunyuan/Hy-Nomas>.
Copyright (C) 2026 Tencent. All rights reserved.

Changes: native Crystal/Metal execution, explicit resident-buffer contracts,
strict checkpoint bindings and a source-pinned diagnostic adapter. The included
core Apache-2.0 terms are in
[licenses/hy-nomas-Apache-2.0.txt](licenses/hy-nomas-Apache-2.0.txt).
The [upstream license bundle](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/911cea6458a89c2663513ff7099561ec190dd491/LICENSE)
also lists separately licensed dependencies; the core license is not a blanket
license for that dependency bundle or for separately supplied encoders/decoders.
No model weights or community ComfyUI implementation are included.

- `src/ml/video/hy_nomas_block.cr`
- `src/ml/video/kernels/hy_nomas_block.metal`
- `spec/hy_nomas_block_spec.cr`
- `spec/support/hy_nomas_reference.py`

## PyTorch MPS LayerNorm compatibility path

`src/ml/video/kernels/hy_nomas_mps_norm.metal` is an opt-in adaptation of the
non-affine F32 `layer_norm_single_row` kernel from PyTorch
`aten/src/ATen/native/mps/kernels/LayerNorm.metal` at tag `v2.8.0`.
Upstream: <https://github.com/pytorch/pytorch/tree/v2.8.0>.
The upstream copyright notices and BSD-3-Clause terms are retained in
[licenses/pytorch-v2.8.0-BSD-3-Clause.txt](licenses/pytorch-v2.8.0-BSD-3-Clause.txt).
This path is selected only by the explicit Hy-Nomas MPS block backend, for its
two non-affine block LayerNorm operations. The native default encoder remains
on its existing normalization path. `spec/support/hy_nomas_norm_controls.cr`
aliases the same prepared kernel for same-input diagnostics.

The SiLU and tanh-GELU graphs in
`src/ml/video/hy_nomas_mps_reductions_bridge.mm` reproduce the operation order
in PyTorch `aten/src/ATen/native/mps/operations/Activation.mm` at tag `v2.8.0`.
The same PyTorch notices and BSD-3-Clause license above apply. The graphs are
used by same-input controls and the explicit Hy-Nomas MPS block backend.

## PyTorch MPS vector attention compatibility probe

`src/ml/video/kernels/hy_nomas_vector_attention.metal` adapts `sdpa_vector`
from PyTorch `aten/src/ATen/native/mps/kernels/Attention.metal` at tag `v2.8.0`,
commit `ba56102387ef21a3b04b357e5b183d48f0afefc7`. Its source SHA-256 is
`aaab42c45caaa222c62013af8dcac896ffa2abd889dbb9427c7648be6448c0cd`.
The upstream file credits MLX's `scaled_dot_product_attention.metal`.
Retain the PyTorch notices and
[BSD-3-Clause license](licenses/pytorch-v2.8.0-BSD-3-Clause.txt), and Apple's
[MLX MIT license](licenses/mlx-MIT.txt) when redistributing this adaptation.

Changes: specialize D=V=128, equal head counts and no mask, use fixed F32
BQHD buffers with explicit BF16 input/output boundaries, and address a compact
query interval. The source's 32-way online softmax and SIMD reduction order are
preserved. The probe is default-off; it does not expose probabilities or change
the production refiner. No PyTorch runtime is used by the native kernel.

## Shared Wan video components

The Wan text encoder and video export helpers retain their upstream source
lineage from [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) at
`1ea34ff48f87168174e12956e200b1d908b1c5ff`.
Copyright 2024-2025 The Alibaba Wan Team Authors.

The [Apache License 2.0](licenses/wan2.2-Apache-2.0.txt) accompanies these
adaptations. The shared components include `src/ml/video/wan_t5.cr` and
`examples/wan_s2v_export.py`.

Shared independent GPU primitives in `src/ml/video/metal_ops.cr` and
`src/ml/video/kernels/common.metal` retain the repository's MIT license.
No upstream weights are bundled. External weights and separately supplied text,
audio, or image encoders retain their own applicable licenses.
