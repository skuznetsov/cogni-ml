#!/usr/bin/env python3
"""Export the pinned TRELLIS.2 packed sparse self-attention 3D RoPE seam.

The oracle executes the exact upstream SparseRotaryPositionEmbedder on tiny
synthetic CPU/F32 QKV data. It does not execute attention, model weights, a
sparse convolution backend, GPU, or Metal code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SPARSE_ROPE_SHA256 = (
    "0525164901c3f1c885e961b747c856b654a4d6840882f34827677fb017dcec00"
)
SPARSE_ATTENTION_MODULES_SHA256 = (
    "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf"
)
SPARSE_BASIC_SHA256 = (
    "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
)
SPARSE_CONFIG_SHA256 = (
    "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b"
)
STRUCTURED_FLOW_SHA256 = (
    "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021"
)
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-self-attention-rope-oracle/v1"

COORDINATES = [
    [0, 2, 0, 1],
    [0, 0, 3, 2],
    [2, 1, 2, 0],
    [2, 2, 0, 1],
]
BATCH_SIZE = 3
SPATIAL_SHAPE = [3, 4, 3]
NUM_HEADS = 2
HEAD_DIM = 16
ROPE_FREQ = [1.0, 10000.0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(
        values.astype(dtype, copy=False).tobytes(order="C")
    ).hexdigest()


def require_pins(trellis_root: Path) -> dict[str, Path]:
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    expected = {
        "python": PYTHON_PIN,
        "torch": TORCH_PIN,
        "numpy": NUMPY_PIN,
    }
    if versions != expected:
        raise RuntimeError(f"pinned package drift: expected {expected}, got {versions}")

    sources = {
        "sparse_rope": trellis_root / "trellis2/modules/sparse/attention/rope.py",
        "sparse_attention_modules": (
            trellis_root / "trellis2/modules/sparse/attention/modules.py"
        ),
        "sparse_basic": trellis_root / "trellis2/modules/sparse/basic.py",
        "sparse_config": trellis_root / "trellis2/modules/sparse/config.py",
        "structured_flow": (
            trellis_root / "trellis2/models/structured_latent_flow.py"
        ),
    }
    expected_hashes = {
        "sparse_rope": SPARSE_ROPE_SHA256,
        "sparse_attention_modules": SPARSE_ATTENTION_MODULES_SHA256,
        "sparse_basic": SPARSE_BASIC_SHA256,
        "sparse_config": SPARSE_CONFIG_SHA256,
        "structured_flow": STRUCTURED_FLOW_SHA256,
    }
    for name, source in sources.items():
        actual = sha256(source)
        if actual != expected_hashes[name]:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_hashes[name]}, "
                f"got {actual}"
            )
    return sources


def load_upstream(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    sparse_module = importlib.import_module("trellis2.modules.sparse")
    attention_module = importlib.import_module(
        "trellis2.modules.sparse.attention.modules"
    )
    return sparse_module, attention_module.SparseMultiHeadAttention


def input_values() -> np.ndarray:
    size = len(COORDINATES) * 3 * NUM_HEADS * HEAD_DIM
    linear = np.arange(size, dtype=np.float32)
    values = ((linear % np.float32(41.0)) - np.float32(20.0)) / np.float32(7.0)
    return values.reshape(len(COORDINATES), 3, NUM_HEADS, HEAD_DIM)


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    sparse_module, attention_type = load_upstream(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    qkv_array = input_values()
    sparse_qkv = sparse_module.SparseTensor(
        torch.from_numpy(qkv_array.copy()),
        torch.from_numpy(coordinate_array.copy()),
        shape=torch.Size([BATCH_SIZE, 3, NUM_HEADS, HEAD_DIM]),
    )
    attention = attention_type(
        channels=NUM_HEADS * HEAD_DIM,
        num_heads=NUM_HEADS,
        type="self",
        qkv_bias=True,
        use_rope=True,
        rope_freq=tuple(ROPE_FREQ),
        qk_rms_norm=False,
    ).cpu().eval()

    input_before = sparse_qkv.feats.detach().clone()
    with torch.inference_mode():
        q, k, v = sparse_qkv.unbind(dim=-3)
        q, k = attention.rope(q, k)
        output = sparse_qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))

    if output.coords is not sparse_qkv.coords:
        raise AssertionError("pinned upstream coordinate-object drift")
    if not torch.equal(sparse_qkv.feats, input_before):
        raise AssertionError("pinned upstream mutated the packed QKV input")
    if not torch.equal(output.feats[:, 2], input_before[:, 2]):
        raise AssertionError("pinned upstream changed V while applying RoPE")
    if not torch.equal(output.feats[..., 12:], input_before[..., 12:]):
        raise AssertionError("pinned upstream changed identity-padded RoPE pairs")

    output_array = output.feats.detach().cpu().numpy()
    source_paths = {
        "sparse_rope": "trellis2/modules/sparse/attention/rope.py",
        "sparse_attention_modules": "trellis2/modules/sparse/attention/modules.py",
        "sparse_basic": "trellis2/modules/sparse/basic.py",
        "sparse_config": "trellis2/modules/sparse/config.py",
        "structured_flow": "trellis2/models/structured_latent_flow.py",
    }
    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "sources": {
                name: {"path": source_paths[name], "sha256": sha256(path)}
                for name, path in sources.items()
            },
            "generator": (
                "tools/trellis2_oracle/export_sparse_self_attention_rope.py"
            ),
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "synthetic",
            "sparse_backend": "none",
            "upstream_sparse_rotary_executed": True,
        },
        "contract": {
            "consumer": "SparseMultiHeadAttention.forward self path",
            "upstream_calls": [
                "q, k, v = qkv.unbind(dim=-3)",
                "q, k = rope(q, k)",
                "qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))",
            ],
            "coordinate_columns": ["batch", "x", "y", "z"],
            "phase_coordinate_columns": ["x", "y", "z"],
            "pair_layout": "axis-major frequencies followed by identity padding",
            "phase_shared_across_heads": True,
            "value_component_unchanged": True,
            "input_unchanged": True,
            "not_attention_backend_parity": True,
            "local_mode": "bounded graphless CPU/F32 inference",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "rope_freq": ROPE_FREQ,
            "qkv_features": qkv_array.tolist(),
            "qkv_features_f32le_sha256": little_endian_sha(qkv_array, "<f4"),
        },
        "output": {
            "qkv_features": output_array.tolist(),
            "qkv_features_f32le_sha256": little_endian_sha(output_array, "<f4"),
            "coordinate_object_reused": True,
            "identity_tail_unchanged": True,
            "value_component_unchanged": True,
            "input_unchanged": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--trellis-root", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.trellis_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
