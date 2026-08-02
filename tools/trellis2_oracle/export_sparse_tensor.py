#!/usr/bin/env python3
"""Export a weight-free structural oracle for TRELLIS.2 SparseTensor.

The oracle executes only the pinned Python container with the sparse convolution
backend disabled. It records the source-observed batch/layout metadata and an
independent tuple lookup. Dense conversion is deliberately recorded as a pinned
failure, not used as reference behavior.
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
SCHEMA = "cogni-ml/trellis2/sparse-tensor-structural-oracle/v1"

COORDINATES = [
    [0, 2, 0, 1],
    [0, 0, 1, 0],
    [2, 1, 2, 0],
    [2, 0, 0, 3],
    [2, 3, 1, 2],
    [2, 2, 2, 1],
]
BATCH_SIZE = 3
CHANNELS = 2
SPATIAL_SHAPE = [4, 3, 4]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(values.astype(dtype, copy=False).tobytes(order="C")).hexdigest()


def sentinel_feature(row: list[int], channel: int) -> float:
    batch, x, y, z = row
    return float(100_000 * batch + 10_000 * x + 100 * y + 10 * z + channel)


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


def load_sparse_tensor(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    sparse_basic = importlib.import_module("trellis2.modules.sparse.basic")
    return sparse_basic.SparseTensor


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sparse_basic = require_pins(trellis_root)
    sparse_tensor_type = load_sparse_tensor(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    feature_array = np.asarray(
        [
            [sentinel_feature(row, channel) for channel in range(CHANNELS)]
            for row in COORDINATES
        ],
        dtype=np.float32,
    )
    coordinates = torch.from_numpy(coordinate_array.copy())
    features = torch.from_numpy(feature_array.copy())
    sparse = sparse_tensor_type(
        features,
        coordinates,
        shape=torch.Size([BATCH_SIZE, CHANNELS]),
    )

    layout = [[part.start, part.stop] for part in sparse.layout]
    expected_layout = [[0, 2], [2, 2], [2, 6]]
    if layout != expected_layout:
        raise AssertionError(f"upstream layout drift: {layout}")

    dense_failure: dict[str, str]
    try:
        sparse.to_dense()
    except Exception as error:  # The exact failure is part of this pinned probe.
        dense_failure = {
            "type": type(error).__name__,
            "message": str(error),
        }
    else:
        raise AssertionError("pinned upstream SparseTensor.to_dense unexpectedly succeeded")
    if dense_failure["type"] != "TypeError" or "list" not in dense_failure["message"] or "tuple" not in dense_failure["message"]:
        raise AssertionError(f"unexpected upstream to_dense failure: {dense_failure}")

    tuple_index = {tuple(row): index for index, row in enumerate(COORDINATES)}
    tuple_queries = [
        [2, 3, 1, 2],
        [0, 0, 1, 0],
        [2, 0, 0, 3],
        [1, 0, 0, 0],
    ]
    tuple_lookup = [
        {"coordinate": coordinate, "row": tuple_index.get(tuple(coordinate))}
        for coordinate in tuple_queries
    ]
    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "source": (
                "https://github.com/microsoft/TRELLIS.2/blob/"
                f"{TRELLIS_PIN}/trellis2/modules/sparse/basic.py"
            ),
            "source_sha256": sha256(sparse_basic),
            "generator": "tools/trellis2_oracle/export_sparse_tensor.py",
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "none",
            "sparse_backend": "none",
        },
        "contract": {
            "coordinate_columns": ["batch", "x", "y", "z"],
            "input_order": "preserved within nondecreasing batch groups",
            "duplicates": "local CoordinateMap3D rejects",
            "explicit_extents": True,
            "dense_materialization": "not admitted",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "channels": CHANNELS,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "features": feature_array.tolist(),
            "features_f32le_sha256": little_endian_sha(feature_array, "<f4"),
        },
        "upstream": {
            "shape": list(sparse.shape),
            "layout": layout,
            "occupied_spatial_shape": list(sparse.spatial_shape),
            "sequence_lengths": sparse.seqlen.tolist(),
            "cumulative_sequence_lengths": sparse.cum_seqlen.tolist(),
            "batch_broadcast_map": sparse.batch_boardcast_map.tolist(),
            "to_dense_failure": dense_failure,
        },
        "independent_tuple_lookup_queries": tuple_lookup,
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
