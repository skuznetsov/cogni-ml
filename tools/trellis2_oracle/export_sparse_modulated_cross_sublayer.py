#!/usr/bin/env python3
"""Export a source-bound CPU/F32 oracle for the first MSA seam.

The fixture executes the real pinned ``ModulatedSparseTransformerCrossBlock``
``_forward`` and stops at the input to ``norm2``.  Only the sparse full-
attention backend is replaced: the replacement is an explicit, stable,
block-diagonal CPU reference over the upstream ``SparseTensor.layout``.
No production weights, accelerators, checkpointing, cross-attention, MLP, or
``norm2`` arithmetic are executed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SYSTEM_PIN = "Darwin"
MACHINE_PIN = "arm64"
TORCH_CPU_CAPABILITY_PIN = "DEFAULT"
SCHEMA = "cogni-ml/trellis2/sparse-modulated-cross-sublayer-oracle/v1"

SOURCE_PINS = {
    "sparse_basic": (
        "trellis2/modules/sparse/basic.py",
        "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
    ),
    "sparse_config": (
        "trellis2/modules/sparse/config.py",
        "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b",
    ),
    "sparse_attention_modules": (
        "trellis2/modules/sparse/attention/modules.py",
        "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf",
    ),
    "sparse_full_attention": (
        "trellis2/modules/sparse/attention/full_attn.py",
        "bee0c32089f060c8136292f41a2cc7a952a8679a8b6d113d14c772f2c681e520",
    ),
    "sparse_rope": (
        "trellis2/modules/sparse/attention/rope.py",
        "0525164901c3f1c885e961b747c856b654a4d6840882f34827677fb017dcec00",
    ),
    "modulated_cross_block": (
        "trellis2/modules/sparse/transformer/modulated.py",
        "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2",
    ),
    "norm": (
        "trellis2/modules/norm.py",
        "f89c40abf3356f7b06fc85f0498cd77eefb677a0e0d370a43d14a735f4c40172",
    ),
    "structured_latent_flow": (
        "trellis2/models/structured_latent_flow.py",
        "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021",
    ),
}

CONFIG_PINS = {
    "shape_512": (
        "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json",
        "6989e77f8b5ff4eb524522649e7708bee56526544f5d059f55760fcc5567d388",
    ),
    "shape_1024": (
        "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json",
        "310f9588a6d3ebc7c036b1bb5be79e96343ff232cc9c5627e0d590f101949da0",
    ),
    "texture_512": (
        "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
        "a344cef8feca45a4efc2201c53e772ebd77b97f9aff1b1e91328576ab6f3e1c6",
    ),
    "texture_1024": (
        "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json",
        "df727c8b2bcd6fc592e4feb0489ddec57c73f2f4fdb5b4028ded8648d6d37057",
    ),
}

BATCH_SIZE = 3
CHANNELS = 16
NUM_HEADS = 2
HEAD_DIM = CHANNELS // NUM_HEADS
SPATIAL_SHAPE = [3, 3, 3]
SEQUENCE_LENGTHS = [2, 0, 2]
COORDINATES = np.asarray(
    [
        [0, 0, 1, 2],
        [0, 2, 0, 1],
        [2, 1, 2, 0],
        [2, 2, 1, 1],
    ],
    dtype=np.int32,
)
EPSILON = 1.0e-6


class Norm2Boundary(RuntimeError):
    """Internal control flow used to stop before norm2 arithmetic."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str = "<f4") -> str:
    return hashlib.sha256(values.astype(dtype, copy=False).tobytes(order="C")).hexdigest()


def f32_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def patterned(shape: tuple[int, ...], start: float, step: float, jitter: float) -> np.ndarray:
    """Build an asymmetric, deterministic F32 pattern without RNG state."""

    count = int(np.prod(shape))
    index_i = np.arange(count, dtype=np.int32)
    index_f = index_i.astype(np.float32)
    residue = (index_i % np.int32(11)).astype(np.float32) - np.float32(5.0)
    output = (
        np.float32(start)
        + index_f * np.float32(step)
        + residue * np.float32(jitter)
    )
    return output.astype(np.float32, copy=False).reshape(shape)


def stage_payload(value: torch.Tensor | np.ndarray) -> dict[str, object]:
    array = f32_array(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    )
    return {
        "shape": list(array.shape),
        "values": array.reshape(-1).tolist(),
        "f32le_sha256": little_endian_sha(array),
    }


def parameter_payload(value: torch.Tensor | np.ndarray) -> dict[str, object]:
    return stage_payload(value)


def require_f32_cpu_contiguous(name: str, value: torch.Tensor) -> None:
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise AssertionError(f"{name} must be CPU/F32, got {value.device}/{value.dtype}")
    if not value.is_contiguous():
        raise AssertionError(f"{name} must be contiguous")
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} must be finite")


def require_pins(trellis_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    expected_versions = {
        "python": PYTHON_PIN,
        "torch": TORCH_PIN,
        "numpy": NUMPY_PIN,
    }
    if versions != expected_versions:
        raise RuntimeError(
            f"pinned package drift: expected {expected_versions}, got {versions}"
        )

    runtime = {
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }
    expected_runtime = {
        "system": SYSTEM_PIN,
        "machine": MACHINE_PIN,
        "torch_cpu_capability": TORCH_CPU_CAPABILITY_PIN,
    }
    if runtime != expected_runtime:
        raise RuntimeError(
            f"pinned CPU runtime drift: expected {expected_runtime}, got {runtime}"
        )

    sources: dict[str, Path] = {}
    for name, (relative_path, expected_hash) in SOURCE_PINS.items():
        source = trellis_root / relative_path
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_hash}, got {actual_hash}"
            )
        sources[name] = source

    configs: dict[str, Path] = {}
    for name, (relative_path, expected_hash) in CONFIG_PINS.items():
        config = trellis_root / relative_path
        actual_hash = sha256(config)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned {name} config drift: expected {expected_hash}, got {actual_hash}"
            )
        configs[name] = config

    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(trellis_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to verify pinned TRELLIS.2 commit") from error
    if actual_commit != TRELLIS_PIN:
        raise RuntimeError(
            f"pinned TRELLIS.2 commit drift: expected {TRELLIS_PIN}, got {actual_commit}"
        )
    return sources, configs


def load_upstream(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    sparse_module = importlib.import_module("trellis2.modules.sparse")
    modulated = importlib.import_module("trellis2.modules.sparse.transformer.modulated")
    attention_modules = importlib.import_module(
        "trellis2.modules.sparse.attention.modules"
    )
    return sparse_config, sparse_module, modulated, attention_modules


class CaptureNorm1(nn.Module):
    """Delegate exactly to upstream norm1 while retaining its two tensors."""

    def __init__(self, delegate: nn.Module) -> None:
        super().__init__()
        self.delegate = delegate
        self.input_features: torch.Tensor | None = None
        self.output_features: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.input_features = features.detach().clone()
        output = self.delegate(features)
        self.output_features = output.detach().clone()
        return output


class CaptureSelfAttention(nn.Module):
    """Delegate exactly to upstream self-attention and capture its seam."""

    def __init__(self, delegate: nn.Module) -> None:
        super().__init__()
        self.delegate = delegate
        self.input_features: torch.Tensor | None = None
        self.output_features: torch.Tensor | None = None

    def forward(self, value):
        self.input_features = value.feats.detach().clone()
        output = self.delegate(value)
        self.output_features = output.feats.detach().clone()
        return output


class CaptureNorm2Boundary(nn.Module):
    """Capture norm2's input and raise before invoking any norm2 arithmetic."""

    def __init__(self) -> None:
        super().__init__()
        self.input_features: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.input_features = features.detach().clone()
        raise Norm2Boundary


def explicit_block_diagonal_reference(qkv, sparse_module):
    """Stable CPU full attention over each non-empty sparse batch segment."""

    if not isinstance(qkv, sparse_module.SparseTensor):
        raise AssertionError(f"backend expected SparseTensor, got {type(qkv)!r}")
    if qkv.feats.ndim != 4 or tuple(qkv.feats.shape[1:]) != (3, NUM_HEADS, HEAD_DIM):
        raise AssertionError(f"unexpected packed qkv shape: {tuple(qkv.feats.shape)}")
    require_f32_cpu_contiguous("packed qkv", qkv.feats)

    output = torch.empty(
        (qkv.feats.shape[0], NUM_HEADS, HEAD_DIM), dtype=torch.float32, device="cpu"
    )
    for segment in qkv.layout:
        if segment.start == segment.stop:
            continue
        packed = qkv.feats[segment]
        query, key, value = packed.unbind(dim=1)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value).transpose(0, 1)
        output[segment] = attended
    return qkv.replace(output)


def production_configuration(config_paths: dict[str, Path]) -> list[dict[str, object]]:
    configurations = []
    for name, config_path in config_paths.items():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        denoiser = payload["models"]["denoiser"]
        args = denoiser["args"]
        expected = {
            "name": "ElasticSLatFlowModel",
            "num_blocks": 30,
            "share_mod": True,
            "num_heads": 12,
            "pe_mode": "rope",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
        }
        actual = {
            "name": denoiser["name"],
            "num_blocks": args.get("num_blocks"),
            "share_mod": args.get("share_mod"),
            "num_heads": args.get("num_heads"),
            "pe_mode": args.get("pe_mode"),
            "qk_rms_norm": args.get("qk_rms_norm"),
            "qk_rms_norm_cross": args.get("qk_rms_norm_cross"),
        }
        if actual != expected:
            raise AssertionError(f"{name} production config drift: expected {expected}, got {actual}")
        configurations.append(
            {
                "name": name,
                "path": CONFIG_PINS[name][0],
                "sha256": sha256(config_path),
                "model": actual["name"],
                "num_blocks": actual["num_blocks"],
                "share_mod": actual["share_mod"],
                "num_heads": actual["num_heads"],
                "pe_mode": actual["pe_mode"],
                "qk_rms_norm": actual["qk_rms_norm"],
                "qk_rms_norm_cross": actual["qk_rms_norm_cross"],
                "block_attention_mode": "full",
                "block_qkv_bias_default": True,
                "model_use_checkpoint_default": False,
            }
        )
    return configurations


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.manual_seed(20260803)
    sources, config_paths = require_pins(trellis_root)
    _, sparse_module, modulated, attention_modules = load_upstream(trellis_root)

    coordinate_array = COORDINATES.copy()
    residual_array = patterned((4, CHANNELS), -0.72, 0.083, 0.017)
    residual = sparse_module.SparseTensor(
        torch.from_numpy(residual_array.copy()),
        torch.from_numpy(coordinate_array.copy()),
        shape=torch.Size([BATCH_SIZE, CHANNELS]),
    )
    require_f32_cpu_contiguous("residual input", residual.feats)
    if residual.coords.dtype != torch.int32:
        raise AssertionError(f"coordinates must be I32, got {residual.coords.dtype}")
    if not np.array_equal(residual.coords.cpu().numpy(), coordinate_array):
        raise AssertionError("coordinate order drift")
    actual_lengths = [item.stop - item.start for item in residual.layout]
    if actual_lengths != SEQUENCE_LENGTHS:
        raise AssertionError(
            f"pinned sparse layout drift: expected {SEQUENCE_LENGTHS}, got {actual_lengths}"
        )
    batch_map = residual.batch_boardcast_map.detach().cpu().numpy().astype(np.int32)
    expected_batch_map = np.asarray([0, 0, 2, 2], dtype=np.int32)
    if not np.array_equal(batch_map, expected_batch_map):
        raise AssertionError(
            f"pinned batch broadcast drift: expected {expected_batch_map.tolist()}, got {batch_map.tolist()}"
        )
    if residual.coords is not residual.replace(residual.feats).coords:
        raise AssertionError("SparseTensor.replace did not preserve coordinate identity")

    block = modulated.ModulatedSparseTransformerCrossBlock(
        CHANNELS,
        CHANNELS,
        NUM_HEADS,
        attn_mode="full",
        use_rope=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        share_mod=True,
    ).cpu().eval()

    # Explicit synthetic parameters are the only self-attention weights used.
    qkv_weight = patterned((3 * CHANNELS, CHANNELS), -0.14, 0.0017, 0.006)
    qkv_bias = patterned((3 * CHANNELS,), -0.07, 0.009, 0.013)
    q_gamma = patterned((NUM_HEADS, HEAD_DIM), 0.73, 0.027, 0.009)
    k_gamma = patterned((NUM_HEADS, HEAD_DIM), 1.11, -0.019, 0.011)
    to_out_weight = patterned((CHANNELS, CHANNELS), 0.11, -0.00073, 0.004)
    to_out_bias = patterned((CHANNELS,), -0.09, 0.0067, 0.003)
    modulation = patterned((6 * CHANNELS,), -0.12, 0.0023, 0.008)
    # Keep the effective gate exact while retaining asymmetric modulation in
    # the other five chunks; this makes the required zero/negative/>1 cases
    # survive F32 cancellation through ``self.modulation + mod``.
    modulation[2 * CHANNELS : 3 * CHANNELS] = np.float32(0.0)

    gate = np.asarray(
        [
            [0.0, -0.75, 1.25, 0.5, -1.5, 0.25, 1.75, -0.125, 0.875, -0.375, 0.0, 1.1, -0.9, 0.3, 1.4, -0.2],
            [-0.5, 0.0, 1.5, -1.25, 0.75, -0.25, 2.0, -0.75, 0.4, 1.2, -1.1, 0.0, 0.6, -0.4, 1.35, -0.05],
            [1.1, -0.6, 0.0, 1.8, -0.8, 0.2, 1.05, -1.4, 0.7, -0.1, 1.6, -0.3, 0.0, 0.9, -0.7, 1.3],
        ],
        dtype=np.float32,
    )
    scale = patterned((BATCH_SIZE, CHANNELS), -0.41, 0.031, 0.014)
    shift = patterned((BATCH_SIZE, CHANNELS), 0.19, -0.023, 0.011)
    combined = np.concatenate(
        [shift, scale, gate, patterned((BATCH_SIZE, 3 * CHANNELS), -0.08, 0.013, 0.007)],
        axis=1,
    )
    mod_array = (combined - modulation.reshape(1, -1)).astype(np.float32)

    with torch.no_grad():
        block.modulation.copy_(torch.from_numpy(modulation.copy()))
        block.self_attn.to_qkv.weight.copy_(torch.from_numpy(qkv_weight.copy()))
        block.self_attn.to_qkv.bias.copy_(torch.from_numpy(qkv_bias.copy()))
        block.self_attn.q_rms_norm.gamma.copy_(torch.from_numpy(q_gamma.copy()))
        block.self_attn.k_rms_norm.gamma.copy_(torch.from_numpy(k_gamma.copy()))
        block.self_attn.to_out.weight.copy_(torch.from_numpy(to_out_weight.copy()))
        block.self_attn.to_out.bias.copy_(torch.from_numpy(to_out_bias.copy()))
    block.requires_grad_(False)
    if any(parameter.requires_grad for parameter in block.parameters()):
        raise AssertionError("block parameters were not frozen")

    if not block.share_mod or block.use_checkpoint:
        raise AssertionError("fixture block configuration drift")
    self_attn = block.self_attn
    if (
        self_attn.attn_mode != "full"
        or not self_attn.use_rope
        or not self_attn.qk_rms_norm
        or self_attn.to_qkv.bias is None
        or self_attn.num_heads != NUM_HEADS
        or self_attn.head_dim != HEAD_DIM
    ):
        raise AssertionError("self-attention production options drift")

    # Replace only the sparse backend.  This function is an independent
    # block-diagonal CPU reference and does not call pinned full_attn.py.
    backend_capture: dict[str, torch.Tensor] = {}

    def patched_backend(qkv):
        if backend_capture:
            raise AssertionError("self-attention backend called more than once")
        if qkv.coords is not residual.coords:
            raise AssertionError("backend changed coordinate-map identity")
        if [item.stop - item.start for item in qkv.layout] != SEQUENCE_LENGTHS:
            raise AssertionError("backend changed sparse layout")
        backend_capture["packed_qkv"] = qkv.feats.detach().clone()
        output = explicit_block_diagonal_reference(qkv, sparse_module)
        if output.coords is not residual.coords:
            raise AssertionError("reference changed coordinate-map identity")
        backend_capture["attention_pre_output"] = output.feats.detach().clone()
        return output

    original_backend = attention_modules.sparse_scaled_dot_product_attention
    attention_modules.sparse_scaled_dot_product_attention = patched_backend
    try:
        norm1_capture = CaptureNorm1(block.norm1)
        self_attention_capture = CaptureSelfAttention(block.self_attn)
        norm2_capture = CaptureNorm2Boundary()
        block.norm1 = norm1_capture
        block.self_attn = self_attention_capture
        block.norm2 = norm2_capture

        modulation_input = torch.from_numpy(mod_array.copy())
        context = torch.zeros((1, CHANNELS), dtype=torch.float32)
        require_f32_cpu_contiguous("modulation input", modulation_input)
        require_f32_cpu_contiguous("context", context)
        before_residual = residual.feats.detach().clone()
        before_coords = residual.coords.detach().clone()
        before_modulation = modulation_input.detach().clone()
        before_context = context.detach().clone()
        before_parameters = {
            name: parameter.detach().clone() for name, parameter in block.named_parameters()
        }

        try:
            with torch.inference_mode():
                block._forward(residual, modulation_input, context)
        except Norm2Boundary:
            pass
        else:
            raise AssertionError("real block._forward did not reach norm2 boundary")
    finally:
        attention_modules.sparse_scaled_dot_product_attention = original_backend

    if norm1_capture.input_features is None or norm1_capture.output_features is None:
        raise AssertionError("norm1 capture is incomplete")
    if self_attention_capture.input_features is None or self_attention_capture.output_features is None:
        raise AssertionError("self-attention capture is incomplete")
    if norm2_capture.input_features is None:
        raise AssertionError("norm2 boundary capture is incomplete")
    if "packed_qkv" not in backend_capture or "attention_pre_output" not in backend_capture:
        raise AssertionError("attention backend capture is incomplete")

    if not torch.equal(norm1_capture.input_features, before_residual):
        raise AssertionError("norm1 input drift")
    if not torch.equal(residual.feats, before_residual):
        raise AssertionError("residual input was mutated")
    if not torch.equal(residual.coords, before_coords):
        raise AssertionError("coordinate input was mutated")
    if not torch.equal(modulation_input, before_modulation):
        raise AssertionError("modulation input was mutated")
    if not torch.equal(context, before_context):
        raise AssertionError("context input was mutated")
    for name, parameter in block.named_parameters():
        if not torch.equal(parameter, before_parameters[name]):
            raise AssertionError(f"parameter mutated during execution: {name}")

    batch_indices = torch.from_numpy(batch_map.astype(np.int64, copy=False))
    combined_tensor = (block.modulation.detach() + modulation_input).type(modulation_input.dtype)
    shift_tensor, scale_tensor, gate_tensor, _, _, _ = combined_tensor.chunk(6, dim=1)
    shift_tensor = shift_tensor.contiguous()
    scale_tensor = scale_tensor.contiguous()
    gate_tensor = gate_tensor.contiguous()
    if not torch.equal(gate_tensor, torch.from_numpy(gate)):
        raise AssertionError("gate modulation drift")
    if not ((gate_tensor == 0).any() and (gate_tensor < 0).any() and (gate_tensor > 1).any()):
        raise AssertionError("gate must include zero, negative, and >1 values")

    normalized = norm1_capture.output_features
    adaptive = self_attention_capture.input_features
    attention_output = self_attention_capture.output_features
    packed_qkv = backend_capture["packed_qkv"]
    attention_pre_output = backend_capture["attention_pre_output"]
    after_self = norm2_capture.input_features
    expected_adaptive = normalized * (1 + scale_tensor[batch_indices]) + shift_tensor[batch_indices]
    if not torch.equal(adaptive, expected_adaptive):
        raise AssertionError("upstream adaptive affine formula drift")

    qkv_sparse = residual.replace(packed_qkv)
    expected_pre_output = explicit_block_diagonal_reference(qkv_sparse, sparse_module).feats
    # The second call above is only an independent value check; the production
    # backend remains patched and no upstream backend is selected.
    if not torch.equal(attention_pre_output, expected_pre_output):
        raise AssertionError("block-diagonal reference output drift")

    expected_gated_sparse = residual.replace(attention_output) * gate_tensor
    expected_after_sparse = residual + expected_gated_sparse
    if expected_gated_sparse.coords is not residual.coords:
        raise AssertionError("gated attention changed coordinate-map identity")
    if expected_after_sparse.coords is not residual.coords:
        raise AssertionError("residual add changed coordinate-map identity")
    gated_attention = expected_gated_sparse.feats
    if not torch.equal(after_self, expected_after_sparse.feats):
        raise AssertionError("upstream gated residual formula drift")
    expected_attention_output = self_attention_capture.delegate.to_out(
        attention_pre_output.reshape(attention_pre_output.shape[0], -1)
    )
    if not torch.equal(attention_output, expected_attention_output):
        raise AssertionError("upstream output projection drift")

    stages = {
        "residual_input": residual.feats.detach(),
        "normalized_features": normalized,
        "adaptive_affine": adaptive,
        "packed_qkv_at_backend": packed_qkv,
        "attention_pre_output": attention_pre_output,
        "attention_output": attention_output,
        "gated_attention": gated_attention,
        "after_self": after_self,
    }
    expected_shapes = {
        "residual_input": (4, CHANNELS),
        "normalized_features": (4, CHANNELS),
        "adaptive_affine": (4, CHANNELS),
        "packed_qkv_at_backend": (4, 3, NUM_HEADS, HEAD_DIM),
        "attention_pre_output": (4, NUM_HEADS, HEAD_DIM),
        "attention_output": (4, CHANNELS),
        "gated_attention": (4, CHANNELS),
        "after_self": (4, CHANNELS),
    }
    for name, value in stages.items():
        require_f32_cpu_contiguous(name, value)
        if tuple(value.shape) != expected_shapes[name]:
            raise AssertionError(
                f"{name} shape drift: expected {expected_shapes[name]}, got {tuple(value.shape)}"
            )

    parameter_values = {
        "epsilon": EPSILON,
        "scale_msa": scale_tensor,
        "shift_msa": shift_tensor,
        "gate_msa": gate_tensor,
        "qkv_weight": self_attention_capture.delegate.to_qkv.weight.detach(),
        "qkv_bias": self_attention_capture.delegate.to_qkv.bias.detach(),
        "q_gamma": self_attention_capture.delegate.q_rms_norm.gamma.detach(),
        "k_gamma": self_attention_capture.delegate.k_rms_norm.gamma.detach(),
        "to_out_weight": self_attention_capture.delegate.to_out.weight.detach(),
        "to_out_bias": self_attention_capture.delegate.to_out.bias.detach(),
    }
    for name, value in parameter_values.items():
        if isinstance(value, torch.Tensor):
            require_f32_cpu_contiguous(name, value)

    production = production_configuration(config_paths)
    stage_contract = {
        name: {
            "formula": formula,
            "boundary": boundary,
        }
        for name, formula, boundary in (
            ("residual_input", "x.feats", "before _forward"),
            ("normalized_features", "LayerNorm32(x.feats)", "norm1 output"),
            ("adaptive_affine", "normalized * (1 + scale[batch]) + shift[batch]", "self_attn input"),
            ("packed_qkv_at_backend", "RoPE(QK(RMS(to_qkv(adaptive)))) plus V", "patched backend input"),
            ("attention_pre_output", "block_diag_softmax(Q K^T / sqrt(D)) V", "before to_out"),
            ("attention_output", "Linear(C,C, bias)(attention_pre_output)", "self_attn output"),
            ("gated_attention", "attention_output * gate[batch]", "before residual add"),
            ("after_self", "residual_input + gated_attention", "before norm2"),
        )
    }

    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "sources": {
                name: {"path": relative, "sha256": sha256(sources[name])}
                for name, (relative, _) in SOURCE_PINS.items()
            },
            "generator": "tools/trellis2_oracle/export_sparse_modulated_cross_sublayer.py",
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "system": SYSTEM_PIN,
            "machine": MACHINE_PIN,
            "torch_cpu_capability": TORCH_CPU_CAPABILITY_PIN,
            "network": "none",
            "device": "cpu",
            "dtype": "float32",
            "weights": "synthetic patterned F32",
            "sparse_backend": "none",
            "upstream_class_executed": "ModulatedSparseTransformerCrossBlock._forward",
            "upstream_forward_executed": True,
            "upstream_attention_backend_executed": False,
            "independent_reference_executed": True,
            "checkpoint_path_executed": False,
        },
        "production_configuration": {
            "model": "ElasticSLatFlowModel",
            "block": "ModulatedSparseTransformerCrossBlock",
            "block_attention_mode": "full",
            "block_qkv_bias_default": True,
            "qk_rms_norm": True,
            "use_rope": True,
            "share_mod": True,
            "model_use_checkpoint_default": False,
            "configs": production,
        },
        "contract": {
            "owner": "ModulatedSparseTransformerCrossBlock._forward",
            "order": [
                "combined_modulation",
                "norm1",
                "adaptive_affine",
                "self_attn",
                "gate_msa",
                "residual_add",
                "norm2_boundary",
            ],
            "boundary": "before norm2",
            "boundary_detail": "norm2 input capture before norm2 arithmetic",
            "norm1_delegate": "pinned LayerNorm32",
            "self_attn_delegate": "pinned SparseMultiHeadAttention",
            "attention_reference": "explicit stable CPU block-diagonal reference",
            "sequence_lengths": SEQUENCE_LENGTHS,
            "batch_broadcast_map": batch_map.tolist(),
            "coordinate_object_reused": True,
            "coordinate_map_identity_preserved": True,
            "input_and_parameters_unchanged": True,
            "input_and_parameter_storage_unchanged": True,
            "finite_f32_contiguous_layout_checked": True,
            "rejected_scope": [
                "cross attention",
                "MLP",
                "norm2 arithmetic",
                "sparse backend parity",
                "checkpoint execution",
                "production weights",
                "GPU or Metal",
            ],
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "channels": CHANNELS,
            "heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "spatial_shape": SPATIAL_SHAPE,
            "sequence_lengths": SEQUENCE_LENGTHS,
            "coordinates": coordinate_array.tolist(),
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "mod": stage_payload(modulation_input),
            "context": stage_payload(context),
        },
        "parameters": {
            name: value if name == "epsilon" else parameter_payload(value)
            for name, value in parameter_values.items()
        },
        "stages": {name: stage_payload(value) for name, value in stages.items()},
        "stage_contract": stage_contract,
        "stage_f32le_sha256": {
            name: little_endian_sha(value.detach().cpu().numpy())
            for name, value in stages.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.trellis_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} (torch={TORCH_PIN}, device=cpu)")


if __name__ == "__main__":
    main()
