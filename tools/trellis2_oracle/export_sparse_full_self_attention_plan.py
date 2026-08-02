#!/usr/bin/env python3
"""Export a source-bound CPU reference for TRELLIS.2 sparse full attention.

TRELLIS.2 sparse full attention delegates execution to xFormers or Flash
Attention and has no CPU backend.  This oracle therefore binds the sparse
layout contract to the pinned upstream sources, then evaluates each non-empty
batch with the pinned dense naive implementation and an independent stable
softmax formula.  It is deliberately not sparse-backend parity evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-full-self-attention-plan-oracle/v1"

SOURCE_PINS = {
    "sparse_full_attention": (
        "trellis2/modules/sparse/attention/full_attn.py",
        "bee0c32089f060c8136292f41a2cc7a952a8679a8b6d113d14c772f2c681e520",
    ),
    "sparse_attention_modules": (
        "trellis2/modules/sparse/attention/modules.py",
        "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf",
    ),
    "sparse_basic": (
        "trellis2/modules/sparse/basic.py",
        "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
    ),
    "sparse_config": (
        "trellis2/modules/sparse/config.py",
        "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b",
    ),
    "dense_full_attention": (
        "trellis2/modules/attention/full_attn.py",
        "64c43354780dcbc3dcf7612ac5e53d6e21c2081234ea63cd329a77f4185dadfc",
    ),
}

BATCH_SIZE = 4
SEQUENCE_LENGTHS = [2, 0, 3, 1]
SPATIAL_SHAPE = [6, 1, 1]
NUM_HEADS = 2
HEAD_DIM = 2
COORDINATES = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [2, 4, 0, 0],
    [2, 2, 0, 0],
    [2, 3, 0, 0],
    [3, 5, 0, 0],
]

# [T, 3, H, D], with deliberately different value ranges in each batch.
QKV_FEATURES = [
    [[[1.0, -0.5], [0.25, 0.75]], [[0.5, 1.0], [-0.25, 0.5]], [[1.0, 2.0], [3.0, 4.0]]],
    [[[-0.5, 1.5], [1.0, -1.0]], [[1.25, -0.75], [0.5, 0.25]], [[5.0, 6.0], [7.0, 8.0]]],
    [[[0.25, 0.5], [-0.5, 1.0]], [[1.0, 0.0], [0.75, -0.25]], [[11.0, 12.0], [13.0, 14.0]]],
    [[[1.25, -1.0], [0.5, 0.25]], [[-0.5, 1.5], [1.0, 0.5]], [[15.0, 16.0], [17.0, 18.0]]],
    [[[-0.75, 0.25], [1.5, -0.5]], [[0.25, 0.75], [-1.0, 1.25]], [[19.0, 20.0], [21.0, 22.0]]],
    [[[2.0, -1.0], [0.5, 1.5]], [[-0.5, 0.25], [1.0, -0.75]], [[101.0, 102.0], [103.0, 104.0]]],
]


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

    sources = {}
    for name, (relative_path, expected_hash) in SOURCE_PINS.items():
        source = trellis_root / relative_path
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        sources[name] = source
    return sources


def load_upstream(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    sparse_module = importlib.import_module("trellis2.modules.sparse")
    dense_attention = importlib.import_module("trellis2.modules.attention.full_attn")
    return sparse_module, dense_attention._naive_sdpa


def explicit_stable_attention(qkv: torch.Tensor) -> torch.Tensor:
    q, k, v = qkv.unbind(dim=1)
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    weights = shifted.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return torch.matmul(weights, v).permute(1, 0, 2)


def evaluate_batches(qkv: torch.Tensor, layout, dense_naive) -> torch.Tensor:
    output = torch.empty((qkv.shape[0], NUM_HEADS, HEAD_DIM), dtype=torch.float32)
    for batch_slice in layout:
        batch_qkv = qkv[batch_slice]
        if batch_qkv.shape[0] == 0:
            continue
        manual = explicit_stable_attention(batch_qkv)
        q, k, v = batch_qkv.unbind(dim=1)
        dense = dense_naive(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0]
        torch.testing.assert_close(manual, dense, rtol=1e-6, atol=1e-6)
        output[batch_slice] = dense
    return output


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    sparse_module, dense_naive = load_upstream(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    qkv_array = np.asarray(QKV_FEATURES, dtype=np.float32)
    coordinates = torch.from_numpy(coordinate_array.copy())
    qkv = torch.from_numpy(qkv_array.copy())
    sparse_qkv = sparse_module.SparseTensor(
        qkv,
        coordinates,
        shape=torch.Size([BATCH_SIZE, 3, NUM_HEADS, HEAD_DIM]),
    )
    if not torch.equal(sparse_qkv.coords, coordinates):
        raise AssertionError("pinned sparse tensor changed caller row order")
    layout = sparse_qkv.layout
    actual_lengths = [batch_slice.stop - batch_slice.start for batch_slice in layout]
    if actual_lengths != SEQUENCE_LENGTHS:
        raise AssertionError(
            f"pinned sparse layout drift: expected {SEQUENCE_LENGTHS}, got {actual_lengths}"
        )

    with torch.inference_mode():
        output = evaluate_batches(qkv, layout, dense_naive)
        stable_output = torch.cat(
            [explicit_stable_attention(qkv[batch_slice]) for batch_slice in layout if batch_slice.stop > batch_slice.start],
            dim=0,
        )
        torch.testing.assert_close(output, stable_output, rtol=1e-6, atol=1e-6)

        mutated = qkv.clone()
        mutated[layout[2]] *= 1000.0
        mutated_output = evaluate_batches(mutated, layout, dense_naive)
        torch.testing.assert_close(output[layout[0]], mutated_output[layout[0]], rtol=0.0, atol=0.0)
        torch.testing.assert_close(output[layout[3]], mutated_output[layout[3]], rtol=0.0, atol=0.0)
        torch.testing.assert_close(output[layout[3]], qkv[layout[3], 2], rtol=0.0, atol=0.0)

    output_array = output.numpy()
    if not np.isfinite(output_array).all():
        raise AssertionError("full-attention reference produced a non-finite value")

    score_elements = NUM_HEADS * sum(length * length for length in SEQUENCE_LENGTHS)
    output_elements = len(COORDINATES) * NUM_HEADS * HEAD_DIM
    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "sources": {
                name: {"path": SOURCE_PINS[name][0], "sha256": sha256(path)}
                for name, path in sources.items()
            },
            "generator": "tools/trellis2_oracle/export_sparse_full_self_attention_plan.py",
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "synthetic",
            "sparse_backend": "none",
            "upstream_sparse_backend_executed": False,
            "upstream_dense_reference_executed": True,
        },
        "contract": {
            "reference_kind": "source-bound explicit block-diagonal CPU reference",
            "not_backend_parity": True,
            "sequence_lengths": SEQUENCE_LENGTHS,
            "score_elements": score_elements,
            "score_bytes_f32": score_elements * 4,
            "output_elements": output_elements,
            "output_bytes_f32": output_elements * 4,
            "attention_mac_elements": score_elements * HEAD_DIM * 2,
            "batch_layout_source": "SparseTensor.layout",
            "formula_source": "modules/attention/full_attn.py::_naive_sdpa",
            "row_order_preserved_checked": True,
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "qkv_features": qkv_array.tolist(),
            "qkv_features_f32le_sha256": little_endian_sha(qkv_array, "<f4"),
        },
        "attention": {
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "scale": 1.0 / math.sqrt(HEAD_DIM),
        },
        "output": {
            "features": output_array.tolist(),
            "features_f32le_sha256": little_endian_sha(output_array, "<f4"),
            "batch_isolation_checked": True,
            "stable_softmax_checked": True,
            "singleton_equals_value_checked": True,
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
