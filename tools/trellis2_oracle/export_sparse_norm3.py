#!/usr/bin/env python3
"""Export the pinned sparse transformer norm3 CPU/F32 seam.

This deliberately consumes the checked-in cross-attention seam at its
``after_cross`` boundary, executes the real upstream non-affine ``norm3``, and
stops before adaptive MLP modulation.  No sparse backend or model weights are
executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import export_sparse_cross_attention_seam as cross_seam


SCHEMA = "cogni-ml/trellis2/sparse-norm3-oracle/v1"
CROSS_SCHEMA = "cogni-ml/trellis2/sparse-cross-attention-seam-oracle/v1"
CROSS_FIXTURE_SHA256 = (
    "04cc1f953acbb14dec9ca723e71d951dbeea155efb27678784c666dee3b7553a"
)
CROSS_GENERATOR_SHA256 = (
    "a7cb916d8d45c81ea4b79da9fdbe060f9b2266af8d9c5367dfeba066b736e574"
)
EPSILON = 1.0e-6


def load_cross_fixture(path: Path) -> dict[str, object]:
    if cross_seam.sha256(path) != CROSS_FIXTURE_SHA256:
        raise RuntimeError("pinned sparse cross-attention fixture drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema"] != CROSS_SCHEMA:
        raise RuntimeError("pinned sparse cross-attention schema drift")
    if payload["provenance"]["commit"] != cross_seam.TRELLIS_PIN:
        raise RuntimeError("pinned sparse cross-attention commit drift")
    if payload["contract"]["boundary"] != "before norm3":
        raise RuntimeError("sparse cross-attention boundary drift")
    if payload["provenance"]["generator"] != (
        "tools/trellis2_oracle/export_sparse_cross_attention_seam.py"
    ):
        raise RuntimeError("sparse cross-attention generator path drift")
    stage = payload["stages"]["after_cross"]
    values = np.asarray(stage["values"], dtype=np.float32).reshape(stage["shape"])
    if cross_seam.little_endian_sha(values) != stage["f32le_sha256"]:
        raise RuntimeError("sparse after_cross stage digest drift")
    if stage["f32le_sha256"] != payload["stage_f32le_sha256"]["after_cross"]:
        raise RuntimeError("sparse after_cross duplicate digest drift")
    return payload


def build_fixture(
    trellis_root: Path,
    cross_fixture_path: Path,
) -> dict[str, object]:
    sources, _ = cross_seam.require_pins(trellis_root)
    if cross_seam.sha256(Path(cross_seam.__file__)) != CROSS_GENERATOR_SHA256:
        raise RuntimeError("pinned sparse cross-attention generator drift")
    cross_fixture = load_cross_fixture(cross_fixture_path)
    sparse_module, modulated, _ = cross_seam.load_upstream(trellis_root)

    block = modulated.ModulatedSparseTransformerCrossBlock(
        channels=cross_seam.CHANNELS,
        ctx_channels=cross_seam.CONTEXT_CHANNELS,
        num_heads=cross_seam.NUM_HEADS,
        attn_mode="full",
        use_rope=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        share_mod=True,
    ).cpu().eval()
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    if block.norm3.elementwise_affine is not False:
        raise AssertionError("norm3 unexpectedly has affine parameters")
    if block.norm3.weight is not None or block.norm3.bias is not None:
        raise AssertionError("non-affine norm3 unexpectedly owns parameters")
    if block.norm3.eps != EPSILON:
        raise AssertionError("norm3 epsilon drift")

    source_stage = cross_fixture["stages"]["after_cross"]
    source_array = np.asarray(source_stage["values"], dtype=np.float32).reshape(
        source_stage["shape"]
    )
    after_cross = torch.from_numpy(source_array.copy())
    coordinates = np.asarray(
        cross_fixture["input"]["coordinates"], dtype=np.int32
    )
    sparse_input = sparse_module.SparseTensor(
        after_cross,
        torch.from_numpy(coordinates.copy()),
        shape=torch.Size([cross_seam.BATCH_SIZE, cross_seam.CHANNELS]),
    )
    cross_seam.require_f32_cpu_contiguous("after_cross", sparse_input.feats)
    before = sparse_input.feats.detach().clone()
    before_coordinates = sparse_input.coords.detach().clone()
    with torch.inference_mode():
        sparse_output = sparse_input.replace(block.norm3(sparse_input.feats))
    output = sparse_output.feats
    cross_seam.require_f32_cpu_contiguous("norm3 output", output)
    if not torch.equal(sparse_input.feats, before):
        raise AssertionError("norm3 mutated after_cross")
    if not torch.equal(sparse_input.coords, before_coordinates):
        raise AssertionError("norm3 seam mutated coordinates")
    if sparse_output.coords is not sparse_input.coords:
        raise AssertionError("SparseTensor.replace changed coordinate identity")
    if output.data_ptr() == sparse_input.feats.data_ptr():
        raise AssertionError("norm3 unexpectedly aliased its input")
    reference = torch.nn.functional.layer_norm(
        sparse_input.feats.float(),
        (cross_seam.CHANNELS,),
        weight=None,
        bias=None,
        eps=EPSILON,
    ).type(after_cross.dtype)
    if not torch.equal(output, reference):
        raise AssertionError("LayerNorm32 delegate and functional reference differ")

    input_stage = cross_seam.stage_payload(sparse_input.feats)
    output_stage = cross_seam.stage_payload(output)
    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": cross_seam.TRELLIS_PIN,
            "sources": {
                "modulated_cross_block": {
                    "path": cross_seam.SOURCE_PINS["modulated_cross_block"][0],
                    "sha256": cross_seam.sha256(sources["modulated_cross_block"]),
                },
                "norm": {
                    "path": cross_seam.SOURCE_PINS["norm"][0],
                    "sha256": cross_seam.sha256(sources["norm"]),
                },
            },
            "generator": "tools/trellis2_oracle/export_sparse_norm3.py",
            "python_version": cross_seam.PYTHON_PIN,
            "torch_version": cross_seam.TORCH_PIN,
            "numpy_version": cross_seam.NUMPY_PIN,
            "system": cross_seam.SYSTEM_PIN,
            "machine": cross_seam.MACHINE_PIN,
            "torch_cpu_capability": cross_seam.TORCH_CPU_CAPABILITY_PIN,
            "network": "none",
            "device": "cpu",
            "dtype": "float32",
            "weights": "none for norm3",
            "upstream_class_executed": "ModulatedSparseTransformerCrossBlock.norm3",
            "checkpoint_path_executed": False,
        },
        "source_fixture": {
            "path": "spec/fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json",
            "sha256": CROSS_FIXTURE_SHA256,
            "generator": "tools/trellis2_oracle/export_sparse_cross_attention_seam.py",
            "generator_sha256": CROSS_GENERATOR_SHA256,
            "stage": "after_cross",
            "stage_f32le_sha256": source_stage["f32le_sha256"],
        },
        "production_configuration": {
            "model": "ElasticSLatFlowModel",
            "block": "ModulatedSparseTransformerCrossBlock",
            "channels": 1536,
        },
        "contract": {
            "owner": "ModulatedSparseTransformerCrossBlock.norm3",
            "formula": "x.replace(LayerNorm32(elementwise_affine=False, eps=1e-6)(after_cross))",
            "boundary": "after norm3 and before scale_mlp/shift_mlp",
            "layer_norm_elementwise_affine": False,
            "epsilon": EPSILON,
            "variance": "population",
            "input_and_coordinates_unchanged": True,
            "coordinate_map_identity_preserved": True,
            "rejected_scope": [
                "adaptive MLP scale/shift",
                "MLP",
                "sparse backend",
                "checkpoint execution",
                "production weights",
                "GPU or Metal",
            ],
        },
        "input": {
            "batch_size": cross_fixture["input"]["batch_size"],
            "channels": cross_fixture["input"]["channels"],
            "spatial_shape": cross_fixture["input"]["spatial_shape"],
            "query_sequence_lengths": cross_fixture["input"]["query_sequence_lengths"],
            "coordinates": coordinates.tolist(),
            "coordinates_i32le_sha256": cross_seam.little_endian_sha(
                coordinates, "<i4"
            ),
            "after_cross": input_stage,
        },
        "output": {"norm3": output_stage},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis-root", required=True, type=Path)
    parser.add_argument("--cross-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.trellis_root, args.cross_fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} (torch={cross_seam.TORCH_PIN}, device=cpu)")


if __name__ == "__main__":
    main()
