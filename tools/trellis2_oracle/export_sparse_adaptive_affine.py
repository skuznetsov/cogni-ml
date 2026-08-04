#!/usr/bin/env python3
"""Export the pinned sparse post-norm3 adaptive-affine CPU/F32 seam.

The oracle executes the real upstream modulated cross-block through the MLP
input boundary. Earlier attention modules are replaced with deterministic zero
updates, while the real norm3 and SparseTensor batch broadcast remain active.
No sparse backend, checkpoint, model weights, GPU, or Metal runtime is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import export_sparse_norm3 as norm3_oracle


SCHEMA = "cogni-ml/trellis2/sparse-adaptive-affine-oracle/v1"
NORM3_SCHEMA = "cogni-ml/trellis2/sparse-norm3-oracle/v1"
NORM3_FIXTURE_SHA256 = (
    "84cf93c7afa0e791439c1c41fb3d2afe0e131aff8640912cd26b94ea4773c35e"
)
NORM3_GENERATOR_SHA256 = (
    "1cf99b1f900414908934eaa9f2e882bf67c987a8360c06b381eeb24db05283df"
)


class IdentityDense(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class ZeroSparseUpdate(torch.nn.Module):
    def forward(self, value, *unused):
        return value.replace(torch.zeros_like(value.feats))


class CaptureMLPInput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = None

    def forward(self, value):
        self.value = value
        return value.replace(torch.zeros_like(value.feats))


def load_norm3_fixture(path: Path) -> dict[str, object]:
    if norm3_oracle.cross_seam.sha256(path) != NORM3_FIXTURE_SHA256:
        raise RuntimeError("pinned sparse norm3 fixture drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema"] != NORM3_SCHEMA:
        raise RuntimeError("pinned sparse norm3 schema drift")
    if payload["provenance"]["commit"] != norm3_oracle.cross_seam.TRELLIS_PIN:
        raise RuntimeError("pinned sparse norm3 commit drift")
    if payload["contract"]["boundary"] != (
        "after norm3 and before scale_mlp/shift_mlp"
    ):
        raise RuntimeError("pinned sparse norm3 boundary drift")
    output = payload["output"]["norm3"]
    values = np.asarray(output["values"], dtype=np.float32).reshape(output["shape"])
    if norm3_oracle.cross_seam.little_endian_sha(values) != output["f32le_sha256"]:
        raise RuntimeError("pinned sparse norm3 payload drift")
    return payload


def adaptive_parameters(batch_size: int, channels: int):
    channel = np.arange(channels, dtype=np.int32)[None, :]
    batch = np.arange(batch_size, dtype=np.int32)[:, None]
    scale = ((((batch + 1) * (channel + 3)) % 11) - 5).astype(np.float32)
    scale *= np.float32(0.125)
    shift = ((((batch + 2) * (channel + 5)) % 13) - 6).astype(np.float32)
    shift *= np.float32(0.25)
    scale[1, :] = np.float32(123.25)
    shift[1, :] = np.float32(-456.5)
    return scale, shift


def scalar_coordinate_reference(
    normalized: torch.Tensor,
    coordinates: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
) -> torch.Tensor:
    """Compute the affine result without SparseTensor broadcast helpers."""
    result = np.empty(normalized.shape, dtype=np.float32)
    normalized_array = normalized.detach().cpu().numpy()
    for row in range(normalized_array.shape[0]):
        batch_index = int(coordinates[row, 0])
        for channel in range(normalized_array.shape[1]):
            factor = np.float32(np.float32(1.0) + scale[batch_index, channel])
            scaled = np.float32(normalized_array[row, channel] * factor)
            result[row, channel] = np.float32(
                scaled + shift[batch_index, channel]
            )
    return torch.from_numpy(result)


def build_fixture(trellis_root: Path, norm3_fixture_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources, _ = norm3_oracle.cross_seam.require_pins(trellis_root)
    if norm3_oracle.cross_seam.sha256(Path(norm3_oracle.__file__)) != (
        NORM3_GENERATOR_SHA256
    ):
        raise RuntimeError("pinned sparse norm3 generator drift")
    norm3_fixture = load_norm3_fixture(norm3_fixture_path)
    sparse_module, modulated, _ = norm3_oracle.cross_seam.load_upstream(
        trellis_root
    )

    input_payload = norm3_fixture["input"]
    batch_size = int(input_payload["batch_size"])
    channels = int(input_payload["channels"])
    coordinates = np.asarray(input_payload["coordinates"], dtype=np.int32)
    after_cross_payload = input_payload["after_cross"]
    after_cross_array = np.asarray(
        after_cross_payload["values"], dtype=np.float32
    ).reshape(after_cross_payload["shape"])
    expected_norm3_payload = norm3_fixture["output"]["norm3"]
    expected_norm3 = torch.from_numpy(
        np.asarray(expected_norm3_payload["values"], dtype=np.float32).reshape(
            expected_norm3_payload["shape"]
        )
    )
    scale_array, shift_array = adaptive_parameters(batch_size, channels)
    scale = torch.from_numpy(scale_array.copy())
    shift = torch.from_numpy(shift_array.copy())

    block = modulated.ModulatedSparseTransformerCrossBlock(
        channels=channels,
        ctx_channels=norm3_oracle.cross_seam.CONTEXT_CHANNELS,
        num_heads=norm3_oracle.cross_seam.NUM_HEADS,
        attn_mode="full",
        use_rope=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        share_mod=True,
    ).cpu().eval()
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        block.modulation.zero_()
    block.norm1 = IdentityDense()
    block.self_attn = ZeroSparseUpdate()
    block.norm2 = IdentityDense()
    block.cross_attn = ZeroSparseUpdate()
    capture = CaptureMLPInput()
    block.mlp = capture

    sparse_input = sparse_module.SparseTensor(
        torch.from_numpy(after_cross_array.copy()),
        torch.from_numpy(coordinates.copy()),
        shape=torch.Size([batch_size, channels]),
    )
    modulation = torch.zeros(
        (batch_size, 6 * channels), dtype=torch.float32, device="cpu"
    )
    modulation[:, 3 * channels : 4 * channels] = shift
    modulation[:, 4 * channels : 5 * channels] = scale
    context = torch.zeros(
        (batch_size, 1, norm3_oracle.cross_seam.CONTEXT_CHANNELS),
        dtype=torch.float32,
        device="cpu",
    )
    before_features = sparse_input.feats.detach().clone()
    before_coordinates = sparse_input.coords.detach().clone()
    before_modulation = modulation.detach().clone()
    with torch.inference_mode():
        block._forward(sparse_input, modulation, context)

    captured = capture.value
    if captured is None:
        raise AssertionError("upstream MLP input boundary was not reached")
    actual_norm3 = block.norm3(sparse_input.feats)
    if not torch.equal(actual_norm3, expected_norm3):
        raise AssertionError("real block norm3 drifted from the pinned source fixture")
    normalized_sparse = sparse_input.replace(expected_norm3)
    direct = normalized_sparse * (1 + scale) + shift
    if not torch.equal(captured.feats, direct.feats):
        raise AssertionError("captured upstream adaptive-affine value drift")
    scalar_reference = scalar_coordinate_reference(
        expected_norm3,
        coordinates,
        scale_array,
        shift_array,
    )
    if not torch.equal(captured.feats, scalar_reference):
        raise AssertionError("captured value drifted from scalar coordinate reference")
    if captured.coords is not sparse_input.coords or direct.coords is not sparse_input.coords:
        raise AssertionError("adaptive-affine changed coordinate object identity")
    if not torch.equal(sparse_input.feats, before_features):
        raise AssertionError("adaptive-affine block capture mutated input features")
    if not torch.equal(sparse_input.coords, before_coordinates):
        raise AssertionError("adaptive-affine block capture mutated coordinates")
    if not torch.equal(modulation, before_modulation):
        raise AssertionError("adaptive-affine block capture mutated modulation")

    batch_map = sparse_input.batch_boardcast_map.detach().cpu().to(torch.int32)
    expected_map = torch.tensor([0, 0, 2, 2], dtype=torch.int32)
    if not torch.equal(batch_map, expected_map):
        raise AssertionError("asymmetric empty-middle batch map drift")

    source_pins = {
        name: {
            "path": str(path.relative_to(trellis_root)),
            "sha256": norm3_oracle.cross_seam.sha256(path),
        }
        for name, path in sources.items()
        if name in ("sparse_basic", "modulated_cross_block", "norm")
    }
    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": norm3_oracle.cross_seam.TRELLIS_PIN,
            "sources": source_pins,
            "generator": "tools/trellis2_oracle/export_sparse_adaptive_affine.py",
            "generator_sha256": norm3_oracle.cross_seam.sha256(Path(__file__)),
            "python_version": norm3_oracle.cross_seam.PYTHON_PIN,
            "torch_version": norm3_oracle.cross_seam.TORCH_PIN,
            "numpy_version": norm3_oracle.cross_seam.NUMPY_PIN,
            "system": norm3_oracle.cross_seam.SYSTEM_PIN,
            "machine": norm3_oracle.cross_seam.MACHINE_PIN,
            "torch_cpu_capability": norm3_oracle.cross_seam.TORCH_CPU_CAPABILITY_PIN,
            "network": "none",
            "device": "cpu",
            "dtype": "float32",
            "weights": "synthetic modulation; zero block modulation",
            "sparse_backend": "none",
            "upstream_method_executed": "ModulatedSparseTransformerCrossBlock._forward",
        },
        "source_fixture": {
            "path": "spec/fixtures/trellis2/sparse_norm3_cpu_v1.json",
            "sha256": NORM3_FIXTURE_SHA256,
            "generator": "tools/trellis2_oracle/export_sparse_norm3.py",
            "generator_sha256": NORM3_GENERATOR_SHA256,
            "stage": "norm3",
            "stage_f32le_sha256": expected_norm3_payload["f32le_sha256"],
        },
        "contract": {
            "owner": "ModulatedSparseTransformerCrossBlock._forward",
            "source_expression": "h = h * (1 + scale_mlp) + shift_mlp",
            "modulation_chunk_order": [
                "shift_msa",
                "scale_msa",
                "gate_msa",
                "shift_mlp",
                "scale_mlp",
                "gate_mlp",
            ],
            "boundary": "after adaptive MLP affine and before MLP",
            "adaptive_parameter_shape": "[batch_size, channels]",
            "batch_mapping": "SparseTensor.batch_boardcast_map",
            "input_and_modulation_unchanged": True,
            "coordinate_object_reused": True,
            "rejected_scope": [
                "MLP",
                "gate_mlp",
                "final residual",
                "checkpoint execution",
                "production weights",
                "GPU or Metal",
            ],
        },
        "input": {
            "batch_size": batch_size,
            "channels": channels,
            "spatial_shape": input_payload["spatial_shape"],
            "query_sequence_lengths": input_payload["query_sequence_lengths"],
            "coordinates": coordinates.tolist(),
            "coordinates_i32le_sha256": norm3_oracle.cross_seam.little_endian_sha(
                coordinates, "<i4"
            ),
            "batch_broadcast_map": batch_map.numpy().tolist(),
            "batch_broadcast_map_i32le_sha256": norm3_oracle.cross_seam.little_endian_sha(
                batch_map.numpy(), "<i4"
            ),
            "norm3": norm3_oracle.cross_seam.stage_payload(expected_norm3),
        },
        "adaptive": {
            "shift_mlp": norm3_oracle.cross_seam.stage_payload(shift),
            "scale_mlp": norm3_oracle.cross_seam.stage_payload(scale),
            "empty_batch_index": 1,
        },
        "output": {
            "mlp_input": norm3_oracle.cross_seam.stage_payload(captured.feats)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis-root", required=True, type=Path)
    parser.add_argument("--norm3-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.trellis_root, args.norm3_fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} (torch={norm3_oracle.cross_seam.TORCH_PIN}, device=cpu)")


if __name__ == "__main__":
    main()
