#!/usr/bin/env python3
"""Export a synthetic CPU oracle for pinned TRELLIS.2 SparseLinear.

The fixture executes the upstream SparseLinear wrapper itself with the sparse
convolution backend disabled.  Parameters are tiny, asymmetric, and synthetic;
no model weights, network access, GPU, or Metal runtime are involved.
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
SPARSE_LINEAR_SHA256 = (
    "733264d356556108bfbbf31ba757b4d29adde2d126f22a362288848c4c023ce2"
)
STRUCTURED_FLOW_SHA256 = (
    "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021"
)
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-linear-oracle/v1"

COORDINATES = [
    [0, 2, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 2, 0],
]
BATCH_SIZE = 2
SPATIAL_SHAPE = [3, 3, 2]
FEATURES = [
    [1.0, 2.0],
    [3.0, -4.0],
    [0.5, 1.5],
]
WEIGHT = [
    [0.25, -0.5],
    [1.5, 2.0],
    [-2.0, 0.75],
]
BIAS = [0.1, -1.0, 2.5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(values.astype(dtype, copy=False).tobytes(order="C")).hexdigest()


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
        "sparse_basic": trellis_root / "trellis2/modules/sparse/basic.py",
        "sparse_linear": trellis_root / "trellis2/modules/sparse/linear.py",
        "structured_flow": trellis_root / "trellis2/models/structured_latent_flow.py",
    }
    expected_hashes = {
        "sparse_basic": SPARSE_BASIC_SHA256,
        "sparse_linear": SPARSE_LINEAR_SHA256,
        "structured_flow": STRUCTURED_FLOW_SHA256,
    }
    for name, source in sources.items():
        actual_hash = sha256(source)
        expected_hash = expected_hashes[name]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_hash}, got {actual_hash}"
            )
    return sources


def load_sparse_module(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    return importlib.import_module("trellis2.modules.sparse")


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    sparse_module = load_sparse_module(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    feature_array = np.asarray(FEATURES, dtype=np.float32)
    weight_array = np.asarray(WEIGHT, dtype=np.float32)
    bias_array = np.asarray(BIAS, dtype=np.float32)

    coordinates = torch.from_numpy(coordinate_array.copy())
    sparse_input = sparse_module.SparseTensor(
        torch.from_numpy(feature_array.copy()),
        coordinates,
        shape=torch.Size([BATCH_SIZE, feature_array.shape[1]]),
    )
    layer = sparse_module.SparseLinear(
        feature_array.shape[1], weight_array.shape[0], bias=True
    ).cpu().eval()
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight_array.copy()))
        layer.bias.copy_(torch.from_numpy(bias_array.copy()))
    with torch.inference_mode():
        output = layer(sparse_input)

    output_array = output.feats.detach().cpu().numpy()
    reference = feature_array @ weight_array.T + bias_array
    if not np.array_equal(output_array, reference):
        raise AssertionError("pinned upstream SparseLinear value drift")
    if output.coords is not sparse_input.coords:
        raise AssertionError("pinned upstream SparseLinear coordinate-object drift")

    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "sources": {
                "sparse_basic": {
                    "path": "trellis2/modules/sparse/basic.py",
                    "sha256": sha256(sources["sparse_basic"]),
                },
                "sparse_linear": {
                    "path": "trellis2/modules/sparse/linear.py",
                    "sha256": sha256(sources["sparse_linear"]),
                },
                "structured_flow": {
                    "path": "trellis2/models/structured_latent_flow.py",
                    "sha256": sha256(sources["structured_flow"]),
                },
            },
            "generator": "tools/trellis2_oracle/export_sparse_linear.py",
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "synthetic",
            "sparse_backend": "none",
        },
        "contract": {
            "upstream_call": "SparseLinear(input)",
            "upstream_formula": "input.feats @ weight.T + bias",
            "upstream_coordinate_object_reused": True,
            "local_mode": "graphless frozen-parameter CPU inference",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "channels": int(feature_array.shape[1]),
            "features": feature_array.tolist(),
            "features_f32le_sha256": little_endian_sha(feature_array, "<f4"),
        },
        "linear": {
            "out_channels": int(weight_array.shape[0]),
            "weight": weight_array.tolist(),
            "weight_f32le_sha256": little_endian_sha(weight_array, "<f4"),
            "bias": bias_array.tolist(),
            "bias_f32le_sha256": little_endian_sha(bias_array, "<f4"),
        },
        "output": {
            "features": output_array.tolist(),
            "features_f32le_sha256": little_endian_sha(output_array, "<f4"),
            "coordinate_object_reused": True,
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
