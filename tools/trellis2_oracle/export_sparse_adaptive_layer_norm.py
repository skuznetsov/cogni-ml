#!/usr/bin/env python3
"""Export a synthetic CPU oracle for one pinned TRELLIS.2 adaLN seam.

The fixture executes the exact upstream sequence used before sparse attention:
non-affine LayerNorm32 followed by batch-broadcast adaptive scale and shift.
Inputs are tiny, asymmetric, and synthetic; no model weights, network access,
GPU, or Metal runtime are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SPARSE_BASIC_SHA256 = (
    "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
)
MODULATED_BLOCK_SHA256 = (
    "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2"
)
NORM_SHA256 = (
    "f89c40abf3356f7b06fc85f0498cd77eefb677a0e0d370a43d14a735f4c40172"
)
STRUCTURED_FLOW_SHA256 = (
    "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021"
)
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SYSTEM_PIN = "Darwin"
MACHINE_PIN = "arm64"
TORCH_CPU_CAPABILITY_PIN = "DEFAULT"
SCHEMA = "cogni-ml/trellis2/sparse-adaptive-layer-norm-oracle/v1"
EPSILON = 1.0e-6
PYTORCH_CPU_VECTOR_WIDTH = 4
PYTORCH_MOMENTS_CHUNK_SIZE = 16
PYTORCH_LAYER_NORM_SOURCE = {
    "url": (
        "https://github.com/pytorch/pytorch/blob/v2.9.0/"
        "aten/src/ATen/native/cpu/layer_norm_kernel.cpp"
    ),
    "sha256": "14c765e7e931ea313bf0ef42b0c9099d7e762c8f3d33ee408e96807435756711",
}
PYTORCH_MOMENTS_SOURCE = {
    "url": (
        "https://github.com/pytorch/pytorch/blob/v2.9.0/"
        "aten/src/ATen/native/cpu/moments_utils.h"
    ),
    "sha256": "9b421e0b16cdf9f64c4a3e0a97201c71c58f4123522370c8cc79aae0d2f885a1",
}

COORDINATES = [
    [0, 2, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 2, 1],
    [2, 1, 2, 0],
    [2, 2, 1, 1],
]
BATCH_SIZE = 3
SPATIAL_SHAPE = [3, 3, 2]
FEATURES = [
    [0.25, -1.5, 3.75, 8.0],
    [-4.25, 2.5, 0.125, 1.75],
    [262369456.0, 262369472.0, 262369488.0, 262369504.0],
    [12.0, -3.0, 0.5, 7.25],
    [0.001, -0.002, 0.009, -0.004],
]
SCALE = [
    [-0.25, 0.5, -1.25, 0.125],
    [9.0, 19.0, 29.0, 39.0],
    [0.75, -0.5, 0.25, -1.5],
]
SHIFT = [
    [0.1, -0.2, 1.0, 2.5],
    [101.0, 201.0, 301.0, 401.0],
    [-3.0, 4.0, -5.0, 6.0],
]
NUMERICAL_CHANNEL_CASES = [1, 3, 4, 5, 65, 129, 193, 256]


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

    sources = {
        "sparse_basic": trellis_root / "trellis2/modules/sparse/basic.py",
        "modulated_block": (
            trellis_root / "trellis2/modules/sparse/transformer/modulated.py"
        ),
        "norm": trellis_root / "trellis2/modules/norm.py",
        "structured_flow": trellis_root / "trellis2/models/structured_latent_flow.py",
    }
    expected_hashes = {
        "sparse_basic": SPARSE_BASIC_SHA256,
        "modulated_block": MODULATED_BLOCK_SHA256,
        "norm": NORM_SHA256,
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


def load_upstream(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    sparse_module = importlib.import_module("trellis2.modules.sparse")
    norm_module = importlib.import_module("trellis2.modules.norm")
    return sparse_module, norm_module


def build_numerical_cases(norm_module) -> list[dict[str, object]]:
    cases = []
    for channels in NUMERICAL_CHANNEL_CASES:
        indices = np.arange(channels, dtype=np.float32)
        if channels in (5, 129, 193):
            features = np.asarray(
                262369456.0 + (indices.astype(np.int32) % 7) * 16.0,
                dtype=np.float32,
            )
        else:
            features = np.asarray(
                ((indices.astype(np.int32) * 37) % 101 - 50) * 0.03125,
                dtype=np.float32,
            )
        norm = norm_module.LayerNorm32(
            channels, elementwise_affine=False, eps=EPSILON
        ).cpu().eval()
        with torch.inference_mode():
            output = norm(torch.from_numpy(features.copy()).reshape(1, channels))
        output_array = output.detach().cpu().numpy().reshape(channels)
        cases.append(
            {
                "channels": channels,
                "features": features.tolist(),
                "features_f32le_sha256": little_endian_sha(features, "<f4"),
                "output": output_array.tolist(),
                "output_f32le_sha256": little_endian_sha(output_array, "<f4"),
            }
        )
    return cases


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    sparse_module, norm_module = load_upstream(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    feature_array = np.asarray(FEATURES, dtype=np.float32)
    scale_array = np.asarray(SCALE, dtype=np.float32)
    shift_array = np.asarray(SHIFT, dtype=np.float32)

    sparse_input = sparse_module.SparseTensor(
        torch.from_numpy(feature_array.copy()),
        torch.from_numpy(coordinate_array.copy()),
        shape=torch.Size([BATCH_SIZE, feature_array.shape[1]]),
    )
    scale = torch.from_numpy(scale_array.copy())
    shift = torch.from_numpy(shift_array.copy())
    norm = norm_module.LayerNorm32(
        feature_array.shape[1], elementwise_affine=False, eps=EPSILON
    ).cpu().eval()

    with torch.inference_mode():
        normalized = norm(sparse_input.feats)
        normalized_sparse = sparse_input.replace(normalized)
        scaled_sparse = normalized_sparse * (1 + scale)
        output = scaled_sparse + shift

    batch_map = sparse_input.batch_boardcast_map.detach().cpu()
    expected = normalized * (1 + scale[batch_map]) + shift[batch_map]
    if not torch.equal(output.feats, expected):
        raise AssertionError("pinned upstream adaptive layer norm value drift")
    for value in (normalized_sparse, scaled_sparse, output):
        if value.coords is not sparse_input.coords:
            raise AssertionError("pinned upstream coordinate-object drift")

    normalized_array = normalized.detach().cpu().numpy()
    scaled_array = scaled_sparse.feats.detach().cpu().numpy()
    output_array = output.feats.detach().cpu().numpy()
    batch_map_array = batch_map.numpy().astype(np.int32, copy=False)

    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "sources": {
                name: {
                    "path": str(source.relative_to(trellis_root)),
                    "sha256": sha256(source),
                }
                for name, source in sources.items()
            },
            "generator": "tools/trellis2_oracle/export_sparse_adaptive_layer_norm.py",
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "system": SYSTEM_PIN,
            "machine": MACHINE_PIN,
            "torch_cpu_capability": TORCH_CPU_CAPABILITY_PIN,
            "pytorch_cpu_sources": {
                "layer_norm_kernel": PYTORCH_LAYER_NORM_SOURCE,
                "moments_utils": PYTORCH_MOMENTS_SOURCE,
            },
            "network": "none",
            "device": "cpu",
            "weights": "synthetic",
            "sparse_backend": "none",
        },
        "contract": {
            "owner": "ModulatedSparseTransformerCrossBlock.norm1",
            "upstream_calls": [
                "x.replace(LayerNorm32(x.feats))",
                "h * (1 + scale) + shift",
            ],
            "layer_norm_elementwise_affine": False,
            "epsilon": EPSILON,
            "variance": "population",
            "moments_algorithm": "PyTorch F32 RowwiseMoments Welford cascade",
            "cpu_vector_width": PYTORCH_CPU_VECTOR_WIDTH,
            "moments_chunk_size": PYTORCH_MOMENTS_CHUNK_SIZE,
            "adaptive_parameter_shape": "[batch_size, channels]",
            "upstream_coordinate_object_reused": True,
            "local_mode": "graphless fused CPU inference",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "batch_broadcast_map": batch_map_array.tolist(),
            "batch_broadcast_map_i32le_sha256": little_endian_sha(
                batch_map_array, "<i4"
            ),
            "channels": int(feature_array.shape[1]),
            "features": feature_array.tolist(),
            "features_f32le_sha256": little_endian_sha(feature_array, "<f4"),
        },
        "adaptive": {
            "scale": scale_array.tolist(),
            "scale_f32le_sha256": little_endian_sha(scale_array, "<f4"),
            "shift": shift_array.tolist(),
            "shift_f32le_sha256": little_endian_sha(shift_array, "<f4"),
        },
        "intermediate": {
            "normalized_features": normalized_array.tolist(),
            "normalized_features_f32le_sha256": little_endian_sha(
                normalized_array, "<f4"
            ),
            "scaled_features": scaled_array.tolist(),
            "scaled_features_f32le_sha256": little_endian_sha(scaled_array, "<f4"),
        },
        "output": {
            "features": output_array.tolist(),
            "features_f32le_sha256": little_endian_sha(output_array, "<f4"),
            "coordinate_object_reused": True,
        },
        "numerical_cases": build_numerical_cases(norm_module),
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
