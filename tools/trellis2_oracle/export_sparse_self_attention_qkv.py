#!/usr/bin/env python3
"""Export the pinned TRELLIS.2 sparse self-attention QKV projection.

The oracle executes ``SparseMultiHeadAttention._linear`` followed by
``_fused_pre`` with a tiny asymmetric CPU/F32 input.  It does not execute
normalization, RoPE, attention, caches, model weights, GPU, or Metal code.
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
SPARSE_ATTENTION_MODULES_SHA256 = (
    "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf"
)
SPARSE_BASIC_SHA256 = (
    "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
)
SPARSE_MODULATED_SHA256 = (
    "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2"
)
STRUCTURED_FLOW_SHA256 = (
    "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021"
)
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-self-attention-qkv-oracle/v1"

COORDINATES = [
    [0, 2, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 2, 0],
]
BATCH_SIZE = 2
SPATIAL_SHAPE = [3, 3, 2]
FEATURES = [
    [1.0, -2.0, 0.5, 3.0],
    [-1.5, 0.25, 2.0, -0.75],
    [0.125, 4.0, -3.0, 1.5],
]
CHANNELS = 4
NUM_HEADS = 2
WEIGHT = [
    [((row * 7 + col * 3) % 19 - 9) * 0.03125 for col in range(CHANNELS)]
    for row in range(3 * CHANNELS)
]
BIAS = [(index - 5) * 0.0175 for index in range(3 * CHANNELS)]


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
        "sparse_attention_modules": (
            trellis_root / "trellis2/modules/sparse/attention/modules.py"
        ),
        "sparse_basic": trellis_root / "trellis2/modules/sparse/basic.py",
        "sparse_modulated": (
            trellis_root / "trellis2/modules/sparse/transformer/modulated.py"
        ),
        "structured_flow": (
            trellis_root / "trellis2/models/structured_latent_flow.py"
        ),
    }
    expected_hashes = {
        "sparse_attention_modules": SPARSE_ATTENTION_MODULES_SHA256,
        "sparse_basic": SPARSE_BASIC_SHA256,
        "sparse_modulated": SPARSE_MODULATED_SHA256,
        "structured_flow": STRUCTURED_FLOW_SHA256,
    }
    for name, source in sources.items():
        actual_hash = sha256(source)
        if actual_hash != expected_hashes[name]:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_hashes[name]}, "
                f"got {actual_hash}"
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


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    sparse_module, attention_type = load_upstream(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    feature_array = np.asarray(FEATURES, dtype=np.float32)
    weight_array = np.asarray(WEIGHT, dtype=np.float32)
    bias_array = np.asarray(BIAS, dtype=np.float32)

    coordinates = torch.from_numpy(coordinate_array.copy())
    sparse_input = sparse_module.SparseTensor(
        torch.from_numpy(feature_array.copy()),
        coordinates,
        shape=torch.Size([BATCH_SIZE, CHANNELS]),
    )
    attention = attention_type(
        channels=CHANNELS,
        num_heads=NUM_HEADS,
        type="self",
        qkv_bias=True,
        use_rope=False,
        qk_rms_norm=False,
    ).cpu().eval()
    with torch.no_grad():
        attention.to_qkv.weight.copy_(torch.from_numpy(weight_array.copy()))
        attention.to_qkv.bias.copy_(torch.from_numpy(bias_array.copy()))

    with torch.inference_mode():
        projected = attention._linear(attention.to_qkv, sparse_input)
        qkv = attention._fused_pre(projected, num_fused=3)

    projected_array = projected.feats.detach().cpu().numpy()
    qkv_array = qkv.feats.detach().cpu().numpy()
    expected_shape = (
        feature_array.shape[0],
        3,
        NUM_HEADS,
        CHANNELS // NUM_HEADS,
    )
    if qkv_array.shape != expected_shape:
        raise AssertionError(
            f"pinned upstream QKV shape drift: expected {expected_shape}, "
            f"got {qkv_array.shape}"
        )
    if not np.array_equal(qkv_array, projected_array.reshape(expected_shape)):
        raise AssertionError("pinned upstream QKV reshape value drift")
    if qkv.coords is not sparse_input.coords:
        raise AssertionError("pinned upstream QKV coordinate-object drift")
    if qkv.feats.untyped_storage().data_ptr() != projected.feats.untyped_storage().data_ptr():
        raise AssertionError("pinned upstream QKV reshape materialized new storage")

    source_paths = {
        "sparse_attention_modules": "trellis2/modules/sparse/attention/modules.py",
        "sparse_basic": "trellis2/modules/sparse/basic.py",
        "sparse_modulated": "trellis2/modules/sparse/transformer/modulated.py",
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
            "generator": "tools/trellis2_oracle/export_sparse_self_attention_qkv.py",
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "synthetic",
            "sparse_backend": "none",
        },
        "contract": {
            "consumer": "ModulatedSparseTransformerCrossBlock.self_attn",
            "upstream_calls": [
                "SparseMultiHeadAttention._linear(to_qkv, x)",
                "SparseMultiHeadAttention._fused_pre(qkv, num_fused=3)",
            ],
            "projected_feature_shape": [feature_array.shape[0], 3 * CHANNELS],
            "logical_feature_shape": list(expected_shape),
            "sparse_shape": [
                BATCH_SIZE,
                3,
                NUM_HEADS,
                CHANNELS // NUM_HEADS,
            ],
            "coordinate_object_reused": True,
            "reshape_reuses_projected_storage": True,
            "local_mode": "bounded graphless frozen-parameter CPU inference",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "channels": CHANNELS,
            "features": feature_array.tolist(),
            "features_f32le_sha256": little_endian_sha(feature_array, "<f4"),
        },
        "attention": {
            "num_heads": NUM_HEADS,
            "head_dim": CHANNELS // NUM_HEADS,
            "qkv_weight": weight_array.tolist(),
            "qkv_weight_f32le_sha256": little_endian_sha(weight_array, "<f4"),
            "qkv_bias": bias_array.tolist(),
            "qkv_bias_f32le_sha256": little_endian_sha(bias_array, "<f4"),
        },
        "output": {
            "projected_features": projected_array.tolist(),
            "logical_qkv_features": qkv_array.tolist(),
            "projected_features_f32le_sha256": little_endian_sha(
                projected_array, "<f4"
            ),
            "coordinate_object_reused": True,
            "reshape_reuses_projected_storage": True,
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
