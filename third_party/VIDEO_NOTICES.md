# Native video source and license notices

The native video modules are opt-in. Their presence does not implement a
complete generation pipeline or grant rights to external checkpoints.

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

## Wan 2.2 S2V

The Wan-specific implementation and oracle adapter are based on the architecture
and source of Wan-Video/Wan2.2 at
`1ea34ff48f87168174e12956e200b1d908b1c5ff`.
Upstream: <https://github.com/Wan-Video/Wan2.2>.
Copyright 2024-2025 The Alibaba Wan Team Authors.

Changes: native Crystal buffer contracts, Metal kernels, and diagnostic adapters
for reduced deterministic block fixtures. The upstream Apache License 2.0 is
included in [licenses/wan2.2-Apache-2.0.txt](licenses/wan2.2-Apache-2.0.txt).
Retain these notices when redistributing the adapted Wan-specific files.

The AdaLayerNorm oracle also extracts the exact class from
`huggingface/diffusers@v0.31.0/src/diffusers/models/normalization.py`.
Copyright 2024 HuggingFace Inc.; Apache-2.0. The same Apache-2.0 license text
above applies to this dependency. Its source hash is checked by the oracle.

- `src/ml/video/wan_s2v_block.cr`
- `src/ml/video/kernels/wan_s2v.metal`
- `spec/wan_s2v_block_spec.cr`
- `spec/support/wan_s2v_reference.py`

## LTX 2.5

The following LTX-specific files are adaptations of Lightricks/LTX-2 at
`e98f6d927d83c92ef2be6aec76c8440b19306e39` and are covered by the **LTX-2.x
Community License Agreement dated August 11, 2026**, rather than the general
repository MIT license:

- `src/ml/video/ltx2_block.cr`
- `src/ml/video/kernels/ltx2.metal`
- `spec/ltx2_block_spec.cr`
- `spec/support/ltx2_reference.py`

Upstream: <https://github.com/Lightricks/LTX-2>.
Licensor: Lightricks Ltd.
Changes: native Crystal/Metal block execution and a deterministic upstream
diagnostic adapter. Modified files identify these changes in their headers.

The complete agreement, including Attachment A, is included in
[licenses/ltx-2.x-community.txt](licenses/ltx-2.x-community.txt).
It incorporates the [LTX acceptable-use policy](https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf).
The March 30, 2026 policy retrieved on October 19, 2026 is included in
[licenses/ltx-acceptable-use-policy.txt](licenses/ltx-acceptable-use-policy.txt).
The agreement makes the policy in effect at use time applicable; refresh this
snapshot before redistribution or deployment. It also restricts using the
products or outputs to develop or improve competing products or services.
The revenue threshold alone is not a complete permission test for deployment.
Redistribution or service deployment must retain the agreement, notices, and
applicable use restrictions. In particular, the agreement defines derivatives
broadly, including derivative architectures and inference-enabling code; a
native port does not remove its licensing conditions.

Shared independent GPU primitives in `src/ml/video/metal_ops.cr` and
`src/ml/video/kernels/common.metal` retain the repository's MIT license.
No upstream weights are bundled. External weights and separately supplied text,
audio, or image encoders retain their own applicable licenses.
