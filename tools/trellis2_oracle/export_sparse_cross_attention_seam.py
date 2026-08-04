#!/usr/bin/env python3
"""Export a source-bound CPU/F32 oracle for the sparse cross-attention seam.

The fixture executes the pinned ``ModulatedSparseTransformerCrossBlock``
``_forward`` through ``norm2 -> cross_attn -> x + h`` and stops immediately
before ``norm3`` arithmetic.  The cross backend capture also exports the real
upstream Q/K RMS-normalized tensors and the unchanged V view at the boundary
before score arithmetic.  The only substituted operation is the sparse
attention backend: an explicit CPU/F32 reference handles the real upstream
backend call shapes (block-diagonal self attention and ragged-query/dense-
context cross attention).  No production wrapper, weights, GPU/Metal path,
norm3, MLP, checkpoint, or full-block execution is included.
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
SCHEMA = "cogni-ml/trellis2/sparse-cross-attention-seam-oracle/v1"
LOCAL_DENSE_CROSS_ATTENTION_PATH = Path("src/ml/three_d/trellis2/dense_block.cr")
LOCAL_DENSE_CROSS_ATTENTION_SHA256 = (
    "6901223b87c6ec4ee39113c7684191b18c68a04edb64bc53ff800a407d82a7ce"
)

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
    "image_feature_extractor": (
        "trellis2/modules/image_feature_extractor.py",
        "12530b23e8b6a2cc6b87d8cd01922c7b0085199a365b731f19dd0e7ef4919150",
    ),
    "trellis2_image_to_3d": (
        "trellis2/pipelines/trellis2_image_to_3d.py",
        "e2addfca672354284b23d1541a8f49228d5a727d49220fcb8512cca2cdd38ce9",
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
CONTEXT_CHANNELS = 5
CONTEXT_LENGTH = 3
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


class Norm3Boundary(RuntimeError):
    """Internal control flow used to stop before norm3 arithmetic."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str = "<f4") -> str:
    return hashlib.sha256(values.astype(dtype, copy=False).tobytes(order="C")).hexdigest()


def f32_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def patterned(shape: tuple[int, ...], start: float, step: float, jitter: float) -> np.ndarray:
    """Build an asymmetric deterministic F32 pattern without RNG state."""

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


def require_f32_cpu_contiguous(name: str, value: torch.Tensor) -> None:
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise AssertionError(f"{name} must be CPU/F32, got {value.device}/{value.dtype}")
    if not value.is_contiguous():
        raise AssertionError(f"{name} must be contiguous")
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} must be finite")


def require_f32_cpu(name: str, value: torch.Tensor) -> None:
    """Validate backend views without forcing a hidden contiguous copy."""

    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise AssertionError(f"{name} must be CPU/F32, got {value.device}/{value.dtype}")
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
    return sparse_module, modulated, attention_modules


class CaptureNorm2(nn.Module):
    """Delegate to pinned affine LayerNorm32 and retain both seam tensors."""

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
    """Delegate to pinned self attention while retaining the residual seam."""

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


class CaptureCrossAttention(nn.Module):
    """Delegate to pinned cross attention and retain Q/context/output."""

    def __init__(self, delegate: nn.Module) -> None:
        super().__init__()
        self.delegate = delegate
        self.input_sparse = None
        self.context = None
        self.output_sparse = None
        self.output_features: torch.Tensor | None = None

    def forward(self, value, context):
        self.input_sparse = value
        self.context = context
        output = self.delegate(value, context)
        self.output_sparse = output
        self.output_features = output.feats.detach().clone()
        return output


class CaptureNorm3SparseBoundary(nn.Module):
    """Sparse wrapper whose tensor argument is captured before norm3."""

    def __init__(self) -> None:
        super().__init__()
        self.input_features: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.input_features = features.detach().clone()
        raise Norm3Boundary


def explicit_attention_reference(
    args: tuple[object, ...], sparse_module, backend_records: list[dict[str, object]]
):
    """Reference for both real upstream backend call shapes, CPU/F32 only."""

    if len(args) == 1:
        qkv = args[0]
        if not isinstance(qkv, sparse_module.SparseTensor):
            raise AssertionError(f"self backend expected SparseTensor, got {type(qkv)!r}")
        if qkv.feats.ndim != 4 or tuple(qkv.feats.shape[1:]) != (3, NUM_HEADS, HEAD_DIM):
            raise AssertionError(f"unexpected packed qkv shape: {tuple(qkv.feats.shape)}")
        require_f32_cpu_contiguous("packed qkv", qkv.feats)
        output = torch.zeros(
            (qkv.feats.shape[0], NUM_HEADS, HEAD_DIM), dtype=torch.float32, device="cpu"
        )
        for segment in qkv.layout:
            if segment.start == segment.stop:
                continue
            query, key, value = qkv.feats[segment].unbind(dim=1)
            scores = torch.einsum("qhd,khd->hqk", query, key) / math.sqrt(HEAD_DIM)
            probabilities = torch.softmax(scores, dim=-1)
            output[segment] = torch.einsum("hqk,khd->qhd", probabilities, value)
        output = output.contiguous()
        backend_records.append(
            {
                "kind": "self",
                "input": qkv.feats.detach().clone(),
                "output": output.detach().clone(),
                "empty_query_batch_skipped": qkv.layout[1].start == qkv.layout[1].stop,
            }
        )
        return qkv.replace(output)

    if len(args) == 3:
        query, key, value = args
        if not isinstance(query, sparse_module.SparseTensor):
            raise AssertionError(f"cross backend expected SparseTensor query, got {type(query)!r}")
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise AssertionError("cross backend expected dense tensor K/V")
        if key.ndim != 4 or value.ndim != 4 or tuple(key.shape) != tuple(value.shape):
            raise AssertionError("cross backend K/V shape mismatch")
        if tuple(key.shape[1:]) != (CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM):
            raise AssertionError(f"unexpected dense K/V shape: {tuple(key.shape)}")
        require_f32_cpu("cross query", query.feats)
        require_f32_cpu("dense cross key", key)
        require_f32_cpu("dense cross value", value)
        output = torch.zeros(
            (query.feats.shape[0], NUM_HEADS, HEAD_DIM), dtype=torch.float32, device="cpu"
        )
        for batch_index, segment in enumerate(query.layout):
            if segment.start == segment.stop:
                continue
            q_rows = query.feats[segment]
            scores = torch.einsum("qhd,lhd->hql", q_rows, key[batch_index]) / math.sqrt(HEAD_DIM)
            probabilities = torch.softmax(scores, dim=-1)
            output[segment] = torch.einsum(
                "hql,lhd->qhd", probabilities, value[batch_index]
            )
        output = output.contiguous()
        backend_records.append(
            {
                "kind": "cross",
                "query": query.feats.detach().clone(),
                "key": key.detach().clone(),
                "value": value.detach().clone(),
                "output": output.detach().clone(),
                "empty_query_batch_skipped": query.layout[1].start == query.layout[1].stop,
            }
        )
        return query.replace(output)

    raise AssertionError(f"unexpected sparse attention backend arity: {len(args)}")


def production_configuration(config_paths: dict[str, Path]) -> list[dict[str, object]]:
    configurations = []
    for name, config_path in config_paths.items():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        denoiser = payload["models"]["denoiser"]
        args = denoiser["args"]
        expected = {
            "name": "ElasticSLatFlowModel",
            "model_channels": 1536,
            "cond_channels": 1024,
            "num_blocks": 30,
            "share_mod": True,
            "num_heads": 12,
            "pe_mode": "rope",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
        }
        actual = {
            "name": denoiser["name"],
            "model_channels": args.get("model_channels"),
            "cond_channels": args.get("cond_channels"),
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
                "variant": name,
                "path": CONFIG_PINS[name][0],
                "sha256": sha256(config_path),
                "model": actual["name"],
                "model_channels": actual["model_channels"],
                "cond_channels": actual["cond_channels"],
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
    sources, config_paths = require_pins(trellis_root)
    local_dense_source = Path(__file__).resolve().parents[2] / LOCAL_DENSE_CROSS_ATTENTION_PATH
    local_dense_digest = sha256(local_dense_source)
    if local_dense_digest != LOCAL_DENSE_CROSS_ATTENTION_SHA256:
        raise AssertionError(
            "local dense cross-attention source drift: "
            f"expected {LOCAL_DENSE_CROSS_ATTENTION_SHA256}, got {local_dense_digest}"
        )
    sparse_module, modulated, attention_modules = load_upstream(trellis_root)

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
        CONTEXT_CHANNELS,
        NUM_HEADS,
        attn_mode="full",
        use_rope=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        share_mod=True,
    ).cpu().eval()

    # Synthetic patterned parameters make the complete arithmetic reproducible
    # while preserving production ownership and unequal context/channel widths.
    base_modulation = patterned((6 * CHANNELS,), -0.12, 0.0023, 0.008)
    # Keep effective gate values exact after the upstream F32 add.
    base_modulation[2 * CHANNELS : 3 * CHANNELS] = np.float32(0.0)
    norm2_weight = patterned((CHANNELS,), 0.81, 0.013, 0.004)
    norm2_bias = patterned((CHANNELS,), -0.17, 0.009, 0.003)
    to_q_weight = patterned((CHANNELS, CHANNELS), -0.14, 0.0017, 0.006)
    to_q_bias = patterned((CHANNELS,), -0.07, 0.009, 0.013)
    to_kv_weight = patterned((2 * CHANNELS, CONTEXT_CHANNELS), 0.07, -0.0021, 0.005)
    to_kv_bias = patterned((2 * CHANNELS,), 0.04, 0.006, 0.009)
    q_gamma = patterned((NUM_HEADS, HEAD_DIM), 0.73, 0.027, 0.009)
    k_gamma = patterned((NUM_HEADS, HEAD_DIM), 1.11, -0.019, 0.011)
    to_out_weight = patterned((CHANNELS, CHANNELS), 0.11, -0.00073, 0.004)
    to_out_bias = patterned((CHANNELS,), -0.09, 0.0067, 0.003)

    gate = np.asarray(
        [
            [0.0, -0.75, 1.25, 0.5, -1.5, 0.25, 1.75, -0.125, 0.875, -0.375, 0.0, 1.1, -0.9, 0.3, 1.4, -0.2],
            [-0.5, 0.0, 1.5, -1.25, 0.75, -0.25, 2.0, -0.75, 0.4, 1.2, -1.1, 0.0, 0.6, -0.4, 1.35, -0.05],
            [1.1, -0.6, 0.0, 1.8, -0.8, 0.2, 1.05, -1.4, 0.7, -0.1, 1.6, -0.3, 0.0, 0.9, -0.7, 1.3],
        ],
        dtype=np.float32,
    )
    combined = patterned((BATCH_SIZE, 6 * CHANNELS), -0.33, 0.017, 0.011)
    combined[:, 2 * CHANNELS : 3 * CHANNELS] = gate
    mod_array = (combined - base_modulation.reshape(1, -1)).astype(np.float32)
    context_array = patterned(
        (BATCH_SIZE, CONTEXT_LENGTH, CONTEXT_CHANNELS), -0.27, 0.023, 0.007
    )

    with torch.no_grad():
        block.modulation.copy_(torch.from_numpy(base_modulation.copy()))
        block.norm2.weight.copy_(torch.from_numpy(norm2_weight.copy()))
        block.norm2.bias.copy_(torch.from_numpy(norm2_bias.copy()))
        block.self_attn.to_qkv.weight.copy_(torch.from_numpy(patterned((3 * CHANNELS, CHANNELS), -0.14, 0.0017, 0.006)))
        block.self_attn.to_qkv.bias.copy_(torch.from_numpy(patterned((3 * CHANNELS,), -0.07, 0.009, 0.013)))
        block.self_attn.q_rms_norm.gamma.copy_(torch.from_numpy(patterned((NUM_HEADS, HEAD_DIM), 0.73, 0.027, 0.009)))
        block.self_attn.k_rms_norm.gamma.copy_(torch.from_numpy(patterned((NUM_HEADS, HEAD_DIM), 1.11, -0.019, 0.011)))
        block.self_attn.to_out.weight.copy_(torch.from_numpy(patterned((CHANNELS, CHANNELS), 0.11, -0.00073, 0.004)))
        block.self_attn.to_out.bias.copy_(torch.from_numpy(patterned((CHANNELS,), -0.09, 0.0067, 0.003)))
        block.cross_attn.to_q.weight.copy_(torch.from_numpy(to_q_weight.copy()))
        block.cross_attn.to_q.bias.copy_(torch.from_numpy(to_q_bias.copy()))
        block.cross_attn.to_kv.weight.copy_(torch.from_numpy(to_kv_weight.copy()))
        block.cross_attn.to_kv.bias.copy_(torch.from_numpy(to_kv_bias.copy()))
        block.cross_attn.q_rms_norm.gamma.copy_(torch.from_numpy(q_gamma.copy()))
        block.cross_attn.k_rms_norm.gamma.copy_(torch.from_numpy(k_gamma.copy()))
        block.cross_attn.to_out.weight.copy_(torch.from_numpy(to_out_weight.copy()))
        block.cross_attn.to_out.bias.copy_(torch.from_numpy(to_out_bias.copy()))
    block.requires_grad_(False)
    if any(parameter.requires_grad for parameter in block.parameters()):
        raise AssertionError("block parameters were not frozen")
    if not block.share_mod or block.use_checkpoint:
        raise AssertionError("fixture block configuration drift")
    cross_delegate = block.cross_attn
    if (
        cross_delegate.attn_mode != "full"
        or cross_delegate._type != "cross"
        or cross_delegate.use_rope
        or not cross_delegate.qk_rms_norm
        or cross_delegate.to_q.bias is None
        or cross_delegate.to_kv.bias is None
        or cross_delegate.num_heads != NUM_HEADS
        or cross_delegate.ctx_channels != CONTEXT_CHANNELS
    ):
        raise AssertionError("cross-attention production options drift")
    if block.norm2.elementwise_affine is not True or block.norm2.eps != EPSILON:
        raise AssertionError("norm2 affine parameters or epsilon drift")

    # Capture wrappers delegate all arithmetic except norm3, which is a hard
    # boundary.  The upstream backend itself is never called.
    backend_records: list[dict[str, object]] = []
    original_backend = attention_modules.sparse_scaled_dot_product_attention

    def patched_backend(*args, **kwargs):
        if kwargs:
            raise AssertionError("unexpected keyword arguments to backend")
        if len(backend_records) >= 2:
            raise AssertionError("attention backend called more than twice")
        result = explicit_attention_reference(args, sparse_module, backend_records)
        if result.coords is not args[0].coords:
            raise AssertionError("reference changed coordinate-map identity")
        return result

    norm2_capture = CaptureNorm2(block.norm2)
    self_capture = CaptureSelfAttention(block.self_attn)
    cross_capture = CaptureCrossAttention(block.cross_attn)
    norm3_capture = CaptureNorm3SparseBoundary()
    block.norm2 = norm2_capture
    block.self_attn = self_capture
    block.cross_attn = cross_capture
    block.norm3 = norm3_capture

    modulation_input = torch.from_numpy(mod_array.copy())
    context = torch.from_numpy(context_array.copy())
    require_f32_cpu_contiguous("modulation input", modulation_input)
    require_f32_cpu_contiguous("dense context", context)
    before_residual = residual.feats.detach().clone()
    before_coords = residual.coords.detach().clone()
    before_modulation = modulation_input.detach().clone()
    before_context = context.detach().clone()
    before_parameters = {
        name: parameter.detach().clone() for name, parameter in block.named_parameters()
    }

    reached_boundary = False
    attention_modules.sparse_scaled_dot_product_attention = patched_backend
    try:
        try:
            with torch.inference_mode():
                block._forward(residual, modulation_input, context)
        except Norm3Boundary:
            reached_boundary = True
    finally:
        attention_modules.sparse_scaled_dot_product_attention = original_backend
    if not reached_boundary:
        raise AssertionError("real block._forward did not reach norm3 boundary")

    if len(backend_records) != 2 or [record["kind"] for record in backend_records] != ["self", "cross"]:
        raise AssertionError("expected exactly self and cross backend calls")
    if self_capture.input_features is None or self_capture.output_features is None:
        raise AssertionError("self-attention capture is incomplete")
    if norm2_capture.input_features is None or norm2_capture.output_features is None:
        raise AssertionError("norm2 capture is incomplete")
    if (
        cross_capture.input_sparse is None
        or cross_capture.output_sparse is None
        or cross_capture.output_features is None
    ):
        raise AssertionError("cross-attention capture is incomplete")
    if cross_capture.context is not context:
        raise AssertionError("cross attention did not receive the original dense context")
    if cross_capture.input_sparse.coords is not residual.coords:
        raise AssertionError("cross-attention query changed coordinate-map identity")
    if cross_capture.output_sparse.coords is not residual.coords:
        raise AssertionError("cross-attention output changed coordinate-map identity")
    if norm3_capture.input_features is None:
        raise AssertionError("norm3 boundary capture is incomplete")

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
    if not torch.equal(gate_tensor, torch.from_numpy(gate)):
        raise AssertionError("gate modulation drift")
    if not ((gate_tensor == 0).any() and (gate_tensor < 0).any() and (gate_tensor > 1).any()):
        raise AssertionError("gate must include zero, negative, and >1 values")

    after_self = norm2_capture.input_features
    cross_output = cross_capture.output_features
    after_cross = norm3_capture.input_features
    self_output = self_capture.output_features
    expected_after_self = residual.feats + self_output * gate_tensor[batch_indices]
    if not torch.equal(after_self, expected_after_self):
        raise AssertionError("upstream first residual formula drift")
    if not torch.equal(after_cross, after_self + cross_output):
        raise AssertionError("upstream cross residual formula drift")
    if cross_capture.output_sparse.coords is not residual.coords:
        raise AssertionError("norm3 boundary carrier changed coordinate-map identity")

    cross_backend = next(record for record in backend_records if record["kind"] == "cross")
    cross_pre_output = cross_backend["output"]
    projected_cross = cross_delegate.to_out(cross_pre_output.reshape(cross_pre_output.shape[0], -1))
    if not torch.equal(cross_output, projected_cross):
        raise AssertionError("upstream cross output projection drift")
    query_projection = cross_delegate.to_q(norm2_capture.output_features)
    context_kv_projection = cross_delegate.to_kv(context)
    require_f32_cpu_contiguous("cross query projection", query_projection)
    require_f32_cpu_contiguous("cross context K/V projection", context_kv_projection)
    if tuple(query_projection.shape) != (4, CHANNELS):
        raise AssertionError(
            f"cross query projection shape drift: {tuple(query_projection.shape)}"
        )
    if tuple(context_kv_projection.shape) != (
        BATCH_SIZE,
        CONTEXT_LENGTH,
        2 * CHANNELS,
    ):
        raise AssertionError(
            "cross context K/V projection shape drift: "
            f"{tuple(context_kv_projection.shape)}"
        )

    normalized_query = cross_backend["query"]
    normalized_key = cross_backend["key"]
    preserved_value = cross_backend["value"]
    query_heads = query_projection.reshape(4, NUM_HEADS, HEAD_DIM)
    key_heads, value_heads = context_kv_projection.reshape(
        BATCH_SIZE,
        CONTEXT_LENGTH,
        2,
        NUM_HEADS,
        HEAD_DIM,
    ).unbind(dim=2)
    expected_query = (
        torch.nn.functional.normalize(query_heads.float(), dim=-1, eps=1e-12)
        * cross_delegate.q_rms_norm.gamma
        * math.sqrt(HEAD_DIM)
    )
    expected_key = (
        torch.nn.functional.normalize(key_heads.float(), dim=-1, eps=1e-12)
        * cross_delegate.k_rms_norm.gamma
        * math.sqrt(HEAD_DIM)
    )
    if not torch.equal(normalized_query, expected_query):
        raise AssertionError("upstream cross Q RMS normalization drift")
    if not torch.equal(normalized_key, expected_key):
        raise AssertionError("upstream cross K RMS normalization drift")
    if not torch.equal(preserved_value, value_heads):
        raise AssertionError("upstream cross normalization changed V")
    if not bool(backend_records[0]["empty_query_batch_skipped"] and backend_records[1]["empty_query_batch_skipped"]):
        raise AssertionError("empty query batch was not skipped by the reference")

    # Every stage is a flat sparse F32 tensor; the empty batch contributes zero
    # rows and coordinate object identity remains the upstream object throughout.
    for name, value in {
        "residual_input": residual.feats,
        "after_self": after_self,
        "norm2_output": norm2_capture.output_features,
        "cross_attention_output": cross_output,
        "after_cross": after_cross,
    }.items():
        require_f32_cpu_contiguous(name, value)
        if tuple(value.shape) != (4, CHANNELS):
            raise AssertionError(f"{name} shape drift: {tuple(value.shape)}")

    stages = {
        "residual_input": residual.feats.detach(),
        "after_self": after_self.detach(),
        "norm2_output": norm2_capture.output_features.detach(),
        "cross_attention_output": cross_output.detach(),
        "after_cross": after_cross.detach(),
    }
    parameters = {
        "epsilon": EPSILON,
        "norm2_weight": norm2_capture.delegate.weight.detach(),
        "norm2_bias": norm2_capture.delegate.bias.detach(),
        "to_q_weight": cross_delegate.to_q.weight.detach(),
        "to_q_bias": cross_delegate.to_q.bias.detach(),
        "to_kv_weight": cross_delegate.to_kv.weight.detach(),
        "to_kv_bias": cross_delegate.to_kv.bias.detach(),
        "q_gamma": cross_delegate.q_rms_norm.gamma.detach(),
        "k_gamma": cross_delegate.k_rms_norm.gamma.detach(),
        "to_out_weight": cross_delegate.to_out.weight.detach(),
        "to_out_bias": cross_delegate.to_out.bias.detach(),
    }
    for name, value in parameters.items():
        if isinstance(value, torch.Tensor):
            require_f32_cpu_contiguous(name, value)

    production = production_configuration(config_paths)
    failure_precedence = [
        "runtime/source/config pins before execution",
        "CPU/F32/contiguous sparse input and dense context checks before _forward",
        "norm2 arithmetic before cross-attention",
        "cross-attention output before the second residual add",
        "norm3 input capture and boundary stop before norm3 arithmetic",
    ]
    stage_contract = {
        "residual_input": {"formula": "x.feats", "boundary": "before _forward"},
        "after_self": {"formula": "x + self_attn(x_norm1_affine) * gate_msa[batch]", "boundary": "before norm2"},
        "norm2_output": {"formula": "LayerNorm32(affine)(after_self)", "boundary": "cross_attn input"},
        "cross_attention_output": {"formula": "Linear(C,C,bias)(reference(Q,K,V))", "boundary": "before second residual add"},
        "after_cross": {"formula": "after_self + cross_attention_output", "boundary": "before norm3"},
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
            "generator": "tools/trellis2_oracle/export_sparse_cross_attention_seam.py",
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
            "upstream_sparse_attention_backend": "not executed",
            "upstream_class_executed": "ModulatedSparseTransformerCrossBlock._forward",
            "upstream_forward_executed": True,
            "upstream_attention_backend_executed": False,
            "independent_reference_executed": True,
            "checkpoint_path_executed": False,
        },
        "production_configuration": {
            "model": "ElasticSLatFlowModel",
            "block": "ModulatedSparseTransformerCrossBlock",
            "channels": 1536,
            "context_channels": 1024,
            "num_heads": 12,
            "share_mod": True,
            "qk_rms_norm_cross": True,
            "block_attention_mode": "full",
            "block_qkv_bias_default": True,
            "model_use_checkpoint_default": False,
            "context": {
                "standard_pipeline_representation": "dense [B,N,1024]",
                "model_api_alternate_representation": "VarLenTensor [B,*,1024]",
                "exact_token_count": "unresolved-gated-config",
            },
            "configs": production,
        },
        "contract": {
            "owner": "ModulatedSparseTransformerCrossBlock._forward",
            "order": [
                "combined_modulation",
                "norm1",
                "self_attn",
                "first_residual_add",
                "norm2",
                "cross_attn",
                "second_residual_add",
                "norm3_boundary",
            ],
            "boundary": "before norm3",
            "boundary_detail": "norm3 input capture before norm3 arithmetic",
            "norm2_delegate": "pinned affine LayerNorm32",
            "norm2_epsilon": EPSILON,
            "cross_attn_delegate": "pinned SparseMultiHeadAttention",
            "attention_reference": "explicit stable CPU ragged-query/dense-context reference",
            "query_sequence_lengths": SEQUENCE_LENGTHS,
            "dense_context_lengths": [CONTEXT_LENGTH] * BATCH_SIZE,
            "batch_broadcast_map": batch_map.tolist(),
            "context_representation": "dense [B,L,ctxC]",
            "context_channels": CONTEXT_CHANNELS,
            "parameter_ownership": {
                "norm2_weight": "block.norm2.weight",
                "norm2_bias": "block.norm2.bias",
                "to_q_weight": "block.cross_attn.to_q.weight",
                "to_q_bias": "block.cross_attn.to_q.bias",
                "to_kv_weight": "block.cross_attn.to_kv.weight",
                "to_kv_bias": "block.cross_attn.to_kv.bias",
                "q_gamma": "block.cross_attn.q_rms_norm.gamma",
                "k_gamma": "block.cross_attn.k_rms_norm.gamma",
                "to_out_weight": "block.cross_attn.to_out.weight",
                "to_out_bias": "block.cross_attn.to_out.bias",
            },
            "coordinate_object_reused": True,
            "coordinate_map_identity_preserved": True,
            "empty_query_batch_emits_no_rows": True,
            "input_and_parameters_unchanged": True,
            "failure_precedence": failure_precedence,
            "input_finite_f32_contiguous_checked": True,
            "backend_kv_views_may_be_strided": True,
            "rejected_scope": [
                "production wrapper",
                "norm3 arithmetic",
                "MLP",
                "sparse backend parity",
                "checkpoint execution",
                "production weights",
                "GPU or Metal",
            ],
        },
        "failure_precedence": failure_precedence,
        "compatibility": {
            "dense_trellis_cross_attention": {
                "source": {
                    "path": str(LOCAL_DENSE_CROSS_ATTENTION_PATH),
                    "sha256": local_dense_digest,
                },
                "verdict": "arithmetic-oracle-only",
                "accepts_dense_uniform_overlap": True,
                "represents_per_batch_query_lengths": False,
                "preserves_flat_sparse_layout": False,
                "requires_full_dense_buffers": True,
                "production_sparse_carrier": False,
            },
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "channels": CHANNELS,
            "context_channels": CONTEXT_CHANNELS,
            "context_length": CONTEXT_LENGTH,
            "heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "spatial_shape": SPATIAL_SHAPE,
            "query_sequence_lengths": SEQUENCE_LENGTHS,
            "coordinates": coordinate_array.tolist(),
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "mod": stage_payload(modulation_input),
            "context": stage_payload(context),
        },
        "parameters": {
            name: value if name == "epsilon" else stage_payload(value)
            for name, value in parameters.items()
        },
        "cross_attention_projections": {
            "query": stage_payload(query_projection),
            "context_kv": stage_payload(context_kv_projection),
        },
        "cross_attention_qk_rms_norm": {
            "query": stage_payload(normalized_query),
            "key": stage_payload(normalized_key),
            "value": stage_payload(preserved_value),
        },
        "cross_attention_projection_contract": {
            "query_input": "stages.norm2_output flat sparse [N,C]",
            "context_input": "input.context dense [B,L,Cctx]",
            "query_formula": "block.cross_attn.to_q(query)",
            "context_kv_formula": "block.cross_attn.to_kv(context)",
            "query_layout": "[N,C] exact sparse row order",
            "context_kv_layout": "[B,L,2C] dense batch-major",
            "boundary": "before head reshape and Q/K RMS normalization",
            "upstream_linear_modules_executed": True,
            "input_and_parameters_unchanged": True,
        },
        "cross_attention_qk_rms_norm_contract": {
            "query_input": "cross_attention_projections.query reshaped by view to [N,H,D]",
            "key_value_input": "cross_attention_projections.context_kv reshaped by view to [B,L,2,H,D]",
            "query_formula": "F.normalize(q.float(), dim=-1) * q_gamma * sqrt(D)",
            "key_formula": "F.normalize(k.float(), dim=-1) * k_gamma * sqrt(D)",
            "epsilon": 1e-12,
            "gamma_layout": "[H,D]",
            "query_layout": "[N,H,D] exact sparse row order",
            "key_value_layout": "[B,L,H,D] dense batch-major",
            "boundary": "after Q/K RMS normalization and before score arithmetic",
            "upstream_qk_rms_normalizers_executed": True,
            "value_preserved_exactly": True,
            "coordinate_map_identity_preserved": True,
            "empty_query_batch_emits_no_rows": True,
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
        json.dumps(fixture, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} (torch={TORCH_PIN}, device=cpu)")


if __name__ == "__main__":
    main()
