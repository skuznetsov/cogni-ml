#!/usr/bin/env python3
"""Export the pinned TRELLIS.2 sparse self-attention Q/K normalizer.

The oracle executes the exact Q/K RMS-normalization fragment between packed
self-attention projection and attention. It uses tiny synthetic CPU/F32 data;
it does not execute attention, model weights, GPU, or Metal code.
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
SPARSE_CONFIG_SHA256 = (
    "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b"
)
STRUCTURED_FLOW_SHA256 = (
    "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021"
)
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-self-attention-qk-rms-norm-oracle/v1"
EPSILON = 1.0e-12

COORDINATES = [
    [0, 2, 0, 1],
    [0, 0, 1, 0],
    [2, 1, 2, 0],
]
BATCH_SIZE = 3
SPATIAL_SHAPE = [3, 3, 2]
NUM_HEADS = 2
HEAD_DIM = 3
QKV_FEATURES = [
    [
        [[3.0, 4.0, 0.0], [-1.0, 2.0, -2.0]],
        [[0.5, -1.0, 2.0], [0.0, 0.0, 0.0]],
        [[101.0, 102.0, 103.0], [104.0, 105.0, 106.0]],
    ],
    [
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        [[-3.0, 4.0, 12.0], [2.0, -5.0, 1.0]],
        [[201.0, 202.0, 203.0], [204.0, 205.0, 206.0]],
    ],
    [
        [[0.0001, -0.0002, 0.0003], [-4.0, 0.5, 2.0]],
        [[7.0, -8.0, 9.0], [-1.0, -1.0, 2.0]],
        [[301.0, 302.0, 303.0], [304.0, 305.0, 306.0]],
    ],
]
Q_GAMMA = [[0.5, 1.0, 1.5], [-0.75, 0.25, 2.0]]
K_GAMMA = [[1.2, -0.5, 0.75], [0.4, 1.1, -1.3]]


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
        "sparse_config": trellis_root / "trellis2/modules/sparse/config.py",
        "structured_flow": (
            trellis_root / "trellis2/models/structured_latent_flow.py"
        ),
    }
    expected_hashes = {
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


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    sparse_module, attention_type = load_upstream(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    qkv_array = np.asarray(QKV_FEATURES, dtype=np.float32)
    q_gamma_array = np.asarray(Q_GAMMA, dtype=np.float32)
    k_gamma_array = np.asarray(K_GAMMA, dtype=np.float32)
    coordinates = torch.from_numpy(coordinate_array.copy())
    sparse_qkv = sparse_module.SparseTensor(
        torch.from_numpy(qkv_array.copy()),
        coordinates,
        shape=torch.Size([BATCH_SIZE, 3, NUM_HEADS, HEAD_DIM]),
    )
    attention = attention_type(
        channels=NUM_HEADS * HEAD_DIM,
        num_heads=NUM_HEADS,
        type="self",
        qkv_bias=True,
        use_rope=False,
        qk_rms_norm=True,
    ).cpu().eval()
    with torch.no_grad():
        attention.q_rms_norm.gamma.copy_(torch.from_numpy(q_gamma_array.copy()))
        attention.k_rms_norm.gamma.copy_(torch.from_numpy(k_gamma_array.copy()))

    input_before = sparse_qkv.feats.detach().clone()
    with torch.inference_mode():
        q, k, v = sparse_qkv.unbind(dim=-3)
        q = attention.q_rms_norm(q)
        k = attention.k_rms_norm(k)
        output = sparse_qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))

    if output.coords is not sparse_qkv.coords:
        raise AssertionError("pinned upstream coordinate-object drift")
    if not torch.equal(sparse_qkv.feats, input_before):
        raise AssertionError("pinned upstream mutated the packed QKV input")
    if not torch.equal(output.feats[:, 2], input_before[:, 2]):
        raise AssertionError("pinned upstream changed V while normalizing Q/K")
    if not torch.equal(output.feats[0, 1, 1], torch.zeros(HEAD_DIM)):
        raise AssertionError("pinned upstream zero K vector no longer stays zero")
    if not torch.equal(output.feats[1, 0, 0], torch.zeros(HEAD_DIM)):
        raise AssertionError("pinned upstream zero Q vector no longer stays zero")

    output_array = output.feats.detach().cpu().numpy()
    source_paths = {
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
                "tools/trellis2_oracle/"
                "export_sparse_self_attention_qk_rms_norm.py"
            ),
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "synthetic",
            "sparse_backend": "none",
            "upstream_sparse_qk_normalizer_executed": True,
        },
        "contract": {
            "consumer": "SparseMultiHeadAttention.forward self/full path",
            "upstream_calls": [
                "q, k, v = qkv.unbind(dim=-3)",
                "q = q_rms_norm(q)",
                "k = k_rms_norm(k)",
                "qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))",
            ],
            "formula": "F.normalize(x.float(), dim=-1) * gamma * sqrt(head_dim)",
            "normalize_epsilon": EPSILON,
            "logical_feature_shape": list(qkv_array.shape),
            "coordinate_object_reused": True,
            "value_component_unchanged": True,
            "input_unchanged": True,
            "not_attention_backend_parity": True,
            "local_mode": "bounded graphless frozen-parameter CPU inference",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_features": qkv_array.tolist(),
            "qkv_features_f32le_sha256": little_endian_sha(qkv_array, "<f4"),
            "q_gamma": q_gamma_array.tolist(),
            "q_gamma_f32le_sha256": little_endian_sha(q_gamma_array, "<f4"),
            "k_gamma": k_gamma_array.tolist(),
            "k_gamma_f32le_sha256": little_endian_sha(k_gamma_array, "<f4"),
        },
        "output": {
            "qkv_features": output_array.tolist(),
            "qkv_features_f32le_sha256": little_endian_sha(output_array, "<f4"),
            "coordinate_object_reused": True,
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
