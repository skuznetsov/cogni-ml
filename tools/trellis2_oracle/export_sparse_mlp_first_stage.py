#!/usr/bin/env python3
"""Export a source-pinned CPU oracle for TRELLIS.2 final MLP residual.

The oracle executes the upstream ``SparseFeedForwardNet`` with the sparse
backend disabled and captures both the output immediately after its first
``SparseLinear`` and tanh-approximate ``SparseGELU`` and the complete output
after its biased second ``SparseLinear``, then executes the source-owned
``gate_mlp`` broadcast and final residual. Inputs and parameters are tiny
synthetic float32 values; no model weights, network, GPU, or sparse runtime are
admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-mlp-oracle/v3"
GENERATOR_REPO_PATH = "tools/trellis2_oracle/export_sparse_mlp_first_stage.py"
MLP_RATIO = 5.3334

COORDINATES = [
    [0, 2, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 2, 0],
    [1, 0, 0, 1],
]
BATCH_SIZE = 3
SPATIAL_SHAPE = [3, 3, 2]
FEATURES = [
    [-1.25, 0.5, 2.0],
    [0.25, -0.75, 1.5],
    [-2.0, 1.0, 0.125],
    [0.75, 1.25, -1.5],
]
RESIDUAL_FEATURES = [
    [0.125, -0.5, 1.75],
    [-1.0, 0.25, 0.5],
    [2.0, -1.5, 0.75],
    [0.375, 1.25, -2.0],
]
WEIGHT = [
    [0.03125 * (row - 7), -0.0625 * (row + 1), 0.015625 * (2 * row - 5)]
    for row in range(16)
]
BIAS = [0.0625 * (row - 5) for row in range(16)]
TAIL_WEIGHT = [
    [0.0078125 * ((row + 2 * col) % 9 - 4) for row in range(16)]
    for col in range(3)
]
TAIL_BIAS = [-0.03125, 0.0, 0.046875]
GATE_MLP = [
    [0.5, -1.0, 2.0],
    [3.0, 0.25, -0.75],
    [-1.5, 2.5, 0.125],
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(values.astype(dtype, copy=False).tobytes(order="C")).hexdigest()


def scalar_f32_tail_reference(
    hidden: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Compute the biased H->C tail with scalar float32 accumulation.

    This deliberately independent reference avoids using a matrix primitive
    for the final projection.  It is only a consistency check for the tiny
    fixture, not a replacement for the pinned upstream execution.
    """
    point_count, hidden_channels = hidden.shape
    output_channels = weight.shape[0]
    output = np.empty((point_count, output_channels), dtype=np.float32)
    for row in range(point_count):
        for output_channel in range(output_channels):
            total = np.float32(0.0)
            for hidden_channel in range(hidden_channels):
                total = np.float32(
                    total
                    + np.float32(
                        hidden[row, hidden_channel]
                        * weight[output_channel, hidden_channel]
                    )
                )
            output[row, output_channel] = np.float32(total + bias[output_channel])
    return output


def scalar_f32_gate_residual_reference(
    residual: np.ndarray,
    mlp_output: np.ndarray,
    gate_mlp: np.ndarray,
    batch_broadcast_map: np.ndarray,
) -> np.ndarray:
    """Compute ``x + h * gate_mlp`` with scalar float32 operations.

    The batch map mirrors TRELLIS.2 ``SparseTensor.__elemwise__``: a [B, C]
    tensor is expanded to the sparse row order using ``batch_boardcast_map``.
    This independent loop is a consistency check for the source-pinned sparse
    broadcast and residual seam, not a replacement for its execution.
    """
    point_count, channels = residual.shape
    output = np.empty((point_count, channels), dtype=np.float32)
    for row in range(point_count):
        batch = int(batch_broadcast_map[row])
        for channel in range(channels):
            scaled = np.float32(
                np.float32(mlp_output[row, channel])
                * np.float32(gate_mlp[batch, channel])
            )
            output[row, channel] = np.float32(
                np.float32(residual[row, channel]) + scaled
            )
    return output


def require_pins(trellis_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    actual_commit = subprocess.run(
        ["git", "-C", str(trellis_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != TRELLIS_PIN:
        raise RuntimeError(
            f"pinned TRELLIS.2 revision drift: expected {TRELLIS_PIN}, got {actual_commit}"
        )
    dirty_paths = subprocess.run(
        ["git", "-C", str(trellis_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty_paths:
        raise RuntimeError("pinned TRELLIS.2 checkout must be clean")

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    expected = {"python": PYTHON_PIN, "torch": TORCH_PIN, "numpy": NUMPY_PIN}
    if versions != expected:
        raise RuntimeError(f"pinned package drift: expected {expected}, got {versions}")

    sources = {
        "basic": trellis_root / "trellis2/modules/sparse/basic.py",
        "config": trellis_root / "trellis2/modules/sparse/config.py",
        "blocks": trellis_root / "trellis2/modules/sparse/transformer/blocks.py",
        "nonlinearity": trellis_root / "trellis2/modules/sparse/nonlinearity.py",
        "linear": trellis_root / "trellis2/modules/sparse/linear.py",
        "modulated": trellis_root / "trellis2/modules/sparse/transformer/modulated.py",
    }
    expected_sources = {
        "basic": "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
        "config": "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b",
        "blocks": "622e7c5374976c053fb96151c706356b44244e241afab180e0fbdde5da6770d9",
        "nonlinearity": "bc0b7f1f57be9682d0cbbf024a55a3783a80d70c5d0e7f4bede84d13e10430bf",
        "linear": "733264d356556108bfbbf31ba757b4d29adde2d126f22a362288848c4c023ce2",
        "modulated": "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2",
    }
    for name, source in sources.items():
        actual = sha256(source)
        if actual != expected_sources[name]:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_sources[name]}, got {actual}"
            )

    configs = {
        "img2shape_512": trellis_root / "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json",
        "img2shape_ft1024": trellis_root / "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json",
        "imgshape2tex_512": trellis_root / "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
        "imgshape2tex_ft1024": trellis_root / "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json",
    }
    expected_configs = {
        "img2shape_512": "6989e77f8b5ff4eb524522649e7708bee56526544f5d059f55760fcc5567d388",
        "img2shape_ft1024": "310f9588a6d3ebc7c036b1bb5be79e96343ff232cc9c5627e0d590f101949da0",
        "imgshape2tex_512": "a344cef8feca45a4efc2201c53e772ebd77b97f9aff1b1e91328576ab6f3e1c6",
        "imgshape2tex_ft1024": "df727c8b2bcd6fc592e4feb0489ddec57c73f2f4fdb5b4028ded8648d6d37057",
    }
    for name, config in configs.items():
        actual = sha256(config)
        if actual != expected_configs[name]:
            raise RuntimeError(
                f"pinned {name} config drift: expected {expected_configs[name]}, got {actual}"
            )
        payload = json.loads(config.read_text(encoding="utf-8"))
        model_channels = payload["models"]["denoiser"]["args"]["model_channels"]
        mlp_ratio = payload["models"]["denoiser"]["args"]["mlp_ratio"]
        if model_channels != 1536 or mlp_ratio != MLP_RATIO:
            raise RuntimeError(
                f"unexpected {name} production MLP contract: channels={model_channels}, ratio={mlp_ratio}"
            )
    return sources, configs


def load_sparse_feed_forward_net(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    blocks = importlib.import_module("trellis2.modules.sparse.transformer.blocks")
    return blocks.SparseFeedForwardNet, importlib.import_module("trellis2.modules.sparse")


def build_fixture(trellis_root: Path, generator_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources, configs = require_pins(trellis_root)
    sparse_feed_forward_net, sparse_module = load_sparse_feed_forward_net(trellis_root)

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    feature_array = np.asarray(FEATURES, dtype=np.float32)
    residual_array = np.asarray(RESIDUAL_FEATURES, dtype=np.float32)
    weight_array = np.asarray(WEIGHT, dtype=np.float32)
    bias_array = np.asarray(BIAS, dtype=np.float32)
    tail_weight_array = np.asarray(TAIL_WEIGHT, dtype=np.float32)
    tail_bias_array = np.asarray(TAIL_BIAS, dtype=np.float32)
    gate_array = np.asarray(GATE_MLP, dtype=np.float32)

    sparse_input = sparse_module.SparseTensor(
        torch.from_numpy(feature_array.copy()),
        torch.from_numpy(coordinate_array.copy()),
        shape=torch.Size([BATCH_SIZE, feature_array.shape[1]]),
    )
    if residual_array.shape != feature_array.shape:
        raise AssertionError("residual fixture shape must match MLP input")
    if gate_array.shape != (BATCH_SIZE, feature_array.shape[1]):
        raise AssertionError("gate fixture shape must be [B, C]")
    net = sparse_feed_forward_net(feature_array.shape[1], mlp_ratio=MLP_RATIO).cpu().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        net.mlp[0].weight.copy_(torch.from_numpy(weight_array.copy()))
        net.mlp[0].bias.copy_(torch.from_numpy(bias_array.copy()))
        net.mlp[2].weight.copy_(torch.from_numpy(tail_weight_array.copy()))
        net.mlp[2].bias.copy_(torch.from_numpy(tail_bias_array.copy()))

    first_stage: dict[str, np.ndarray] = {}

    def capture_first_stage(_module, _args, output):
        first_stage["features"] = output.feats.detach().cpu().numpy().copy()

    handle = net.mlp[1].register_forward_hook(capture_first_stage)
    with torch.inference_mode():
        full_output = net(sparse_input)
    handle.remove()
    if "features" not in first_stage:
        raise AssertionError("SparseGELU forward hook did not capture the first stage")
    output_array = first_stage["features"]
    tail_output_array = full_output.feats.detach().cpu().numpy().copy()
    if output_array.shape != (feature_array.shape[0], weight_array.shape[0]):
        raise AssertionError(f"unexpected first-stage output shape: {output_array.shape}")
    if not np.isfinite(output_array).all() or not np.isfinite(tail_output_array).all():
        raise AssertionError("upstream first-stage or tail output is not finite")
    if full_output.coords is not sparse_input.coords:
        raise AssertionError("upstream SparseFeedForwardNet coordinate-object drift")
    scalar_tail = scalar_f32_tail_reference(
        output_array,
        tail_weight_array,
        tail_bias_array,
    )
    if not np.allclose(tail_output_array, scalar_tail, rtol=0.0, atol=5e-5):
        raise AssertionError("upstream tail output disagrees with scalar F32 reference")

    # Execute the exact source-pinned ``h = h * gate_mlp; x = x + h`` seam.
    # SparseTensor expands [B, C] using its coordinate-derived broadcast map;
    # retaining the resulting map in the fixture makes empty batches explicit.
    residual_sparse = sparse_module.SparseTensor(
        torch.from_numpy(residual_array.copy()),
        sparse_input.coords,
        shape=torch.Size([BATCH_SIZE, feature_array.shape[1]]),
    )
    gate_tensor = torch.from_numpy(gate_array.copy())
    if residual_sparse.coords is not sparse_input.coords:
        raise AssertionError("upstream residual input coordinate-object drift")
    if gate_tensor.device.type != "cpu" or gate_tensor.dtype != torch.float32:
        raise AssertionError("gate fixture must be CPU float32")
    if not gate_tensor.is_contiguous():
        raise AssertionError("gate fixture must be contiguous")
    gated_sparse = full_output * gate_tensor
    final_sparse = residual_sparse + gated_sparse
    if gated_sparse.coords is not sparse_input.coords:
        raise AssertionError("upstream gate multiplication coordinate-object drift")
    if final_sparse.coords is not residual_sparse.coords:
        raise AssertionError("upstream final residual coordinate-object drift")
    final_output_array = final_sparse.feats.detach().cpu().numpy().copy()
    batch_broadcast_map = sparse_input.batch_boardcast_map.detach().cpu().numpy().copy()
    scalar_final = scalar_f32_gate_residual_reference(
        residual_array,
        tail_output_array,
        gate_array,
        batch_broadcast_map,
    )
    if not np.isfinite(final_output_array).all():
        raise AssertionError("upstream gated residual output is not finite")
    if not np.allclose(final_output_array, scalar_final, rtol=0.0, atol=5e-5):
        raise AssertionError(
            "upstream gated residual output disagrees with scalar F32 reference"
        )
    if not np.array_equal(batch_broadcast_map, np.asarray([0, 0, 1, 1], dtype=np.int64)):
        raise AssertionError(
            f"unexpected coordinate batch broadcast map: {batch_broadcast_map.tolist()}"
        )

    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "sources": {
                name: {"path": str(path.relative_to(trellis_root)), "sha256": sha256(path)}
                for name, path in sources.items()
            },
            "production_slat_configs": {
                name: {"path": str(path.relative_to(trellis_root)), "sha256": sha256(path)}
                for name, path in configs.items()
            },
            "generator": GENERATOR_REPO_PATH,
            "generator_sha256": sha256(generator_path),
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "dtype": "float32",
            "weights": "synthetic",
            "sparse_backend": "none",
        },
        "contract": {
            "upstream_call": "SparseFeedForwardNet.forward -> mlp[0] -> mlp[1] -> mlp[2]",
            "activation": "SparseGELU(approximate=\"tanh\")",
            "hidden_rule": "int(C * 5.3334)",
            "production_model_channels": 1536,
            "production_mlp_ratio": MLP_RATIO,
            "first_stage_hook": "mlp[1] output",
            "tail_projection_executed": True,
            "tail_contract": "biased frozen Linear(H,C)",
            "full_output": "SparseFeedForwardNet output after mlp[2]",
            "modulated_final": "x = residual + gate_mlp[batch] * SparseFeedForwardNet(mlp_input)",
            "gate_broadcast": "SparseTensor.__elemwise__ [B,C] -> batch_boardcast_map -> [N,C]",
            "batch_broadcast_map": batch_broadcast_map.tolist(),
            "gate_shape": [BATCH_SIZE, int(feature_array.shape[1])],
            "total_work_rule": "2 * N * C * H",
            "local_mode": "graphless frozen-parameter CPU inference",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "point_count": int(feature_array.shape[0]),
            "channels": int(feature_array.shape[1]),
            "max_feature_bytes": 1_048_576,
            "features": feature_array.tolist(),
            "features_f32le_sha256": little_endian_sha(feature_array, "<f4"),
        },
        "residual_input": {
            "features": residual_array.tolist(),
            "features_f32le_sha256": little_endian_sha(residual_array, "<f4"),
            "coordinate_object_reused": True,
        },
        "linear": {
            "out_channels": int(weight_array.shape[0]),
            "weight": weight_array.tolist(),
            "weight_f32le_sha256": little_endian_sha(weight_array, "<f4"),
            "bias": bias_array.tolist(),
            "bias_f32le_sha256": little_endian_sha(bias_array, "<f4"),
        },
        "tail_linear": {
            "out_channels": int(tail_weight_array.shape[0]),
            "weight": tail_weight_array.tolist(),
            "weight_f32le_sha256": little_endian_sha(tail_weight_array, "<f4"),
            "bias": tail_bias_array.tolist(),
            "bias_f32le_sha256": little_endian_sha(tail_bias_array, "<f4"),
        },
        "output": {
            "features": output_array.tolist(),
            "features_f32le_sha256": little_endian_sha(output_array, "<f4"),
            "coordinate_object_reused": True,
        },
        "full_output": {
            "features": tail_output_array.tolist(),
            "features_f32le_sha256": little_endian_sha(tail_output_array, "<f4"),
            "coordinate_object_reused": True,
        },
        "gate_mlp": {
            "shape": [BATCH_SIZE, int(feature_array.shape[1])],
            "features": gate_array.tolist(),
            "features_f32le_sha256": little_endian_sha(gate_array, "<f4"),
            "device": "cpu",
            "dtype": "float32",
            "contiguous": True,
        },
        "final_output": {
            "features": final_output_array.tolist(),
            "features_f32le_sha256": little_endian_sha(final_output_array, "<f4"),
            "coordinate_object_reused": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--trellis-root", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.trellis_root, Path(__file__))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
