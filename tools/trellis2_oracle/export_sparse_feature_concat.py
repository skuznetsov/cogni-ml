#!/usr/bin/env python3
"""Export a weight-free oracle for TRELLIS.2 sparse feature concat.

The successful case executes the pinned two-input ``sparse_cat(..., dim=-1)``
path on CPU with the sparse convolution backend disabled. A second probe records
that pinned upstream accepts mismatched coordinates; that permissiveness is an
observed non-guarantee and is deliberately not copied by the local contract.
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
SPARSE_BASIC_SHA256 = (
    "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
)
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-feature-concat-oracle/v1"

COORDINATES = [
    [0, 2, 0, 1],
    [0, 0, 1, 0],
    [2, 1, 2, 0],
    [2, 0, 0, 3],
    [2, 3, 1, 2],
    [2, 2, 2, 1],
]
MISMATCHED_COORDINATES = [
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [2, 1, 2, 0],
    [2, 0, 0, 3],
    [2, 3, 1, 2],
    [2, 2, 2, 1],
]
BATCH_SIZE = 3
SPATIAL_SHAPE = [4, 3, 4]
LEFT_CHANNELS = 2
RIGHT_CHANNELS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(values.astype(dtype, copy=False).tobytes(order="C")).hexdigest()


def sentinel_features(channel_count: int, base: int) -> np.ndarray:
    return np.asarray(
        [
            [float(base + row_index * 100 + channel) for channel in range(channel_count)]
            for row_index in range(len(COORDINATES))
        ],
        dtype=np.float32,
    )


def require_pins(trellis_root: Path) -> Path:
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

    sparse_basic = trellis_root / "trellis2/modules/sparse/basic.py"
    actual_hash = sha256(sparse_basic)
    if actual_hash != SPARSE_BASIC_SHA256:
        raise RuntimeError(
            "pinned sparse source drift: "
            f"expected {SPARSE_BASIC_SHA256}, got {actual_hash}"
        )
    return sparse_basic


def load_sparse_module(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    return importlib.import_module("trellis2.modules.sparse.basic")


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sparse_basic_source = require_pins(trellis_root)
    sparse_module = load_sparse_module(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    left_array = sentinel_features(LEFT_CHANNELS, 1_000)
    right_array = sentinel_features(RIGHT_CHANNELS, 10_000)
    expected = np.concatenate([left_array, right_array], axis=-1)

    coordinates = torch.from_numpy(coordinate_array.copy())
    left = sparse_module.SparseTensor(
        torch.from_numpy(left_array.copy()),
        coordinates,
        shape=torch.Size([BATCH_SIZE, LEFT_CHANNELS]),
    )
    right = sparse_module.SparseTensor(
        torch.from_numpy(right_array.copy()),
        coordinates,
        shape=torch.Size([BATCH_SIZE, RIGHT_CHANNELS]),
    )
    output = sparse_module.sparse_cat([left, right], dim=-1)
    output_array = output.feats.detach().cpu().numpy()
    if not np.array_equal(output_array, expected):
        raise AssertionError("pinned upstream feature concat value drift")
    if not torch.equal(output.coords, left.coords):
        raise AssertionError("pinned upstream feature concat coordinate drift")

    mismatched = sparse_module.SparseTensor(
        torch.from_numpy(right_array.copy()),
        torch.from_numpy(np.asarray(MISMATCHED_COORDINATES, dtype=np.int32)),
        shape=torch.Size([BATCH_SIZE, RIGHT_CHANNELS]),
    )
    permissive_output = sparse_module.sparse_cat([left, mismatched], dim=-1)
    if not np.array_equal(permissive_output.feats.detach().cpu().numpy(), expected):
        raise AssertionError("pinned upstream mismatched-coordinate value drift")
    if not torch.equal(permissive_output.coords, left.coords):
        raise AssertionError("pinned upstream stopped retaining first coordinates")

    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "source": (
                "https://github.com/microsoft/TRELLIS.2/blob/"
                f"{TRELLIS_PIN}/trellis2/modules/sparse/basic.py"
            ),
            "source_sha256": sha256(sparse_basic_source),
            "generator": "tools/trellis2_oracle/export_sparse_feature_concat.py",
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "none",
            "sparse_backend": "none",
        },
        "contract": {
            "arity": 2,
            "axis": "feature_channels",
            "upstream_call": "sparse_cat([left, right], dim=-1)",
            "local_coordinate_policy": "exact same immutable CoordinateMap3D object",
            "upstream_coordinate_policy": "no equality check observed",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "left_channels": LEFT_CHANNELS,
            "left_features": left_array.tolist(),
            "left_features_f32le_sha256": little_endian_sha(left_array, "<f4"),
            "right_channels": RIGHT_CHANNELS,
            "right_features": right_array.tolist(),
            "right_features_f32le_sha256": little_endian_sha(right_array, "<f4"),
        },
        "output": {
            "channels": LEFT_CHANNELS + RIGHT_CHANNELS,
            "features": output_array.tolist(),
            "features_f32le_sha256": little_endian_sha(output_array, "<f4"),
            "coordinates_equal_first_input": True,
        },
        "upstream_non_guarantee": {
            "mismatched_coordinates": MISMATCHED_COORDINATES,
            "mismatched_coordinate_concat_succeeded": True,
            "output_coordinates_equal_first_input": True,
            "local_policy": "reject before allocating output features",
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
