#!/usr/bin/env python3
"""Export pinned immediate consumers of TRELLIS.2 sparse self-attention.

This inventory oracle deliberately stops each real upstream `_forward` method
at the normalization immediately following the first self-attention residual.
It proves that the two plain blocks consume the admitted attention output as
`x + h`, while the two modulated blocks first apply a per-batch/per-channel
`gate_msa` and then add the original residual. Attention, cross-attention, MLP,
checkpoint execution, production weights, and accelerator backends are not run.
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
import torch.nn as nn


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-attention-consumers-oracle/v1"

SOURCE_PINS = {
    "sparse_transformer_blocks": (
        "trellis2/modules/sparse/transformer/blocks.py",
        "622e7c5374976c053fb96151c706356b44244e241afab180e0fbdde5da6770d9",
    ),
    "modulated_transformer_blocks": (
        "trellis2/modules/sparse/transformer/modulated.py",
        "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2",
    ),
    "sparse_tensor_arithmetic": (
        "trellis2/modules/sparse/basic.py",
        "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
    ),
    "structured_latent_flow": (
        "trellis2/models/structured_latent_flow.py",
        "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021",
    ),
}

CONFIG_PINS = {
    "shape_512": (
        "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json",
        "6989e77f8b5ff4eb524522649e7708bee56526544f5d059f55760fcc5567d388",
    ),
    "shape_1024": (
        "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json",
        "310f9588a6d3ebc7c036b1bb5be79e96343ff232cc9c5627e0d590f101949da0",
    ),
    "texture_512": (
        "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
        "a344cef8feca45a4efc2201c53e772ebd77b97f9aff1b1e91328576ab6f3e1c6",
    ),
    "texture_1024": (
        "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json",
        "df727c8b2bcd6fc592e4feb0489ddec57c73f2f4fdb5b4028ded8648d6d37057",
    ),
}

BATCH_SIZE = 3
CHANNELS = 4
NUM_HEADS = 2
SPATIAL_SHAPE = [3, 3, 3]
SEQUENCE_LENGTHS = [2, 0, 2]
COORDINATES = np.asarray(
    [
        [0, 0, 1, 2],
        [0, 2, 0, 1],
        [2, 1, 2, 0],
        [2, 2, 1, 1],
    ],
    dtype=np.int32,
)
RESIDUAL_INPUT = np.asarray(
    [
        [1.0, -2.0, 0.5, 4.0],
        [-1.0, 3.0, 2.0, -0.5],
        [5.0, -4.0, 1.5, 0.25],
        [-3.0, 2.5, -1.0, 6.0],
    ],
    dtype=np.float32,
)
ATTENTION_OUTPUT = np.asarray(
    [
        [0.5, 2.0, -4.0, 1.0],
        [-2.0, 0.25, 3.0, -1.0],
        [1.5, -2.0, 0.5, 4.0],
        [-0.75, 3.0, -1.5, 2.0],
    ],
    dtype=np.float32,
)
TARGET_GATE = np.asarray(
    [
        [0.0, 1.0, -0.5, 1.25],
        [2.0, -2.0, 0.75, 0.25],
        [-1.0, 0.5, 2.0, 0.0],
    ],
    dtype=np.float32,
)


class BoundaryReached(RuntimeError):
    """Internal control flow used to stop after the first residual."""


class CaptureNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.features = features.detach().clone()
        raise BoundaryReached


class FixedSparseAttention(nn.Module):
    def __init__(self, features: torch.Tensor) -> None:
        super().__init__()
        self._features = features
        self.input_features: torch.Tensor | None = None

    def forward(self, value):
        self.input_features = value.feats.detach().clone()
        return value.replace(self._features.detach().clone())


class FixedModulation(nn.Module):
    def __init__(self, modulation: torch.Tensor) -> None:
        super().__init__()
        self._modulation = modulation

    def forward(self, _value: torch.Tensor) -> torch.Tensor:
        return self._modulation.detach().clone()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(
        values.astype(dtype, copy=False).tobytes(order="C")
    ).hexdigest()


def require_pins(trellis_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
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

    configs = {}
    for name, (relative_path, expected_hash) in CONFIG_PINS.items():
        config = trellis_root / relative_path
        actual_hash = sha256(config)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned {name} config drift: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        configs[name] = config

    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(trellis_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to verify pinned TRELLIS.2 commit") from error
    if actual_commit != TRELLIS_PIN:
        raise RuntimeError(
            f"pinned TRELLIS.2 commit drift: expected {TRELLIS_PIN}, "
            f"got {actual_commit}"
        )
    return sources, configs


def load_upstream(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    sparse_module = importlib.import_module("trellis2.modules.sparse")
    blocks = importlib.import_module("trellis2.modules.sparse.transformer.blocks")
    modulated = importlib.import_module(
        "trellis2.modules.sparse.transformer.modulated"
    )
    return sparse_module, blocks, modulated


def capture_first_residual(
    block: nn.Module,
    args: tuple[object, ...],
    attention_name: str,
    attention_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block.eval()
    block.requires_grad_(False)
    block.norm1 = nn.Identity()
    fixed_attention = FixedSparseAttention(attention_features)
    setattr(block, attention_name, fixed_attention)
    capture = CaptureNorm()
    block.norm2 = capture
    try:
        with torch.inference_mode():
            block._forward(*args)
    except BoundaryReached:
        pass
    else:
        raise AssertionError("upstream consumer did not reach the first residual boundary")
    if capture.features is None or fixed_attention.input_features is None:
        raise AssertionError("upstream consumer boundary capture is incomplete")
    return capture.features, fixed_attention.input_features


def source_inventory() -> list[dict[str, object]]:
    return [
        {
            "class": "SparseTransformerBlock",
            "source": "trellis2/modules/sparse/transformer/blocks.py",
            "forward_lines": [63, 70],
            "consumer_lines": [65, 66],
            "consumer": "plain_residual",
            "formula": "x + attention_output",
            "production_consumer": False,
        },
        {
            "class": "SparseTransformerCrossBlock",
            "source": "trellis2/modules/sparse/transformer/blocks.py",
            "forward_lines": [129, 139],
            "consumer_lines": [131, 132],
            "consumer": "plain_residual",
            "formula": "x + attention_output",
            "production_consumer": False,
        },
        {
            "class": "ModulatedSparseTransformerBlock",
            "source": "trellis2/modules/sparse/transformer/modulated.py",
            "forward_lines": [57, 72],
            "consumer_lines": [64, 66],
            "consumer": "modulated_gate_plus_residual",
            "formula": "x + attention_output * gate_msa[batch]",
            "production_consumer": False,
        },
        {
            "class": "ModulatedSparseTransformerCrossBlock",
            "source": "trellis2/modules/sparse/transformer/modulated.py",
            "forward_lines": [142, 160],
            "consumer_lines": [149, 151],
            "consumer": "modulated_gate_plus_residual",
            "formula": "x + attention_output * gate_msa[batch]",
            "production_consumer": True,
        },
    ]


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources, config_paths = require_pins(trellis_root)
    sparse_module, blocks, modulated = load_upstream(trellis_root)

    coordinates = torch.from_numpy(COORDINATES.copy())
    residual = sparse_module.SparseTensor(
        torch.from_numpy(RESIDUAL_INPUT.copy()),
        coordinates,
        shape=torch.Size([BATCH_SIZE, CHANNELS]),
    )
    update = sparse_module.SparseTensor(
        torch.from_numpy(ATTENTION_OUTPUT.copy()),
        coordinates,
        shape=torch.Size([BATCH_SIZE, CHANNELS]),
    )
    actual_lengths = [item.stop - item.start for item in residual.layout]
    if actual_lengths != SEQUENCE_LENGTHS:
        raise AssertionError(
            f"pinned sparse layout drift: expected {SEQUENCE_LENGTHS}, got {actual_lengths}"
        )
    batch_map = residual.batch_boardcast_map.detach().cpu().numpy().astype(np.int32)
    expected_batch_map = np.asarray([0, 0, 2, 2], dtype=np.int32)
    if not np.array_equal(batch_map, expected_batch_map):
        raise AssertionError(
            f"pinned batch broadcast drift: expected {expected_batch_map}, got {batch_map}"
        )

    before_residual = residual.feats.detach().clone()
    before_update = update.feats.detach().clone()
    direct_plain = residual + update
    if direct_plain.coords is not residual.coords:
        raise AssertionError("plain residual changed the coordinate object")

    plain_runs = []
    plain_specs = [
        (
            "SparseTransformerBlock",
            blocks.SparseTransformerBlock(
                CHANNELS, NUM_HEADS, attn_mode="full", use_checkpoint=False
            ),
            (residual,),
            "attn",
        ),
        (
            "SparseTransformerCrossBlock",
            blocks.SparseTransformerCrossBlock(
                CHANNELS,
                CHANNELS,
                NUM_HEADS,
                attn_mode="full",
                use_checkpoint=False,
            ),
            (residual, torch.zeros((1, CHANNELS), dtype=torch.float32)),
            "self_attn",
        ),
    ]
    for class_name, block, args, attention_name in plain_specs:
        captured, attention_input = capture_first_residual(
            block,
            args,
            attention_name,
            torch.from_numpy(ATTENTION_OUTPUT.copy()),
        )
        if not torch.equal(attention_input, before_residual):
            raise AssertionError(f"{class_name} identity pre-norm input drift")
        if not torch.equal(captured, direct_plain.feats):
            raise AssertionError(f"{class_name} plain residual formula drift")
        output = captured.detach().cpu().numpy()
        plain_runs.append(
            {
                "class": class_name,
                "checkpoint": False,
                "output_f32le_sha256": little_endian_sha(output, "<f4"),
            }
        )

    block_modulation = (
        (np.arange(6 * CHANNELS, dtype=np.float32) % np.float32(9.0))
        - np.float32(4.0)
    ) / np.float32(8.0)
    shared_model_modulation = np.zeros(
        (BATCH_SIZE, 6 * CHANNELS), dtype=np.float32
    )
    shared_model_modulation[:, 2 * CHANNELS : 3 * CHANNELS] = (
        TARGET_GATE - block_modulation[2 * CHANNELS : 3 * CHANNELS]
    )
    total_modulation = shared_model_modulation + block_modulation
    gate_msa = total_modulation[:, 2 * CHANNELS : 3 * CHANNELS]
    if not np.array_equal(gate_msa, TARGET_GATE):
        raise AssertionError("shared modulation failed to construct target gate")

    gate_tensor = torch.from_numpy(gate_msa.copy())
    direct_gated = update * gate_tensor
    direct_modulated = residual + direct_gated
    if direct_gated.coords is not update.coords:
        raise AssertionError("gate multiplication changed the coordinate object")
    if direct_modulated.coords is not residual.coords:
        raise AssertionError("modulated residual changed the coordinate object")

    modulated_runs = []
    conditioned_reference = None
    modulated_specs = [
        (
            "ModulatedSparseTransformerBlock",
            modulated.ModulatedSparseTransformerBlock,
            "attn",
            False,
        ),
        (
            "ModulatedSparseTransformerCrossBlock",
            modulated.ModulatedSparseTransformerCrossBlock,
            "self_attn",
            False,
        ),
        (
            "ModulatedSparseTransformerBlock",
            modulated.ModulatedSparseTransformerBlock,
            "attn",
            True,
        ),
        (
            "ModulatedSparseTransformerCrossBlock",
            modulated.ModulatedSparseTransformerCrossBlock,
            "self_attn",
            True,
        ),
    ]
    for class_name, block_class, attention_name, share_mod in modulated_specs:
        if "Cross" in class_name:
            block = block_class(
                CHANNELS,
                CHANNELS,
                NUM_HEADS,
                attn_mode="full",
                use_checkpoint=False,
                share_mod=share_mod,
            )
        else:
            block = block_class(
                CHANNELS,
                NUM_HEADS,
                attn_mode="full",
                use_checkpoint=False,
                share_mod=share_mod,
            )

        if share_mod:
            with torch.no_grad():
                block.modulation.copy_(torch.from_numpy(block_modulation.copy()))
            modulation_input = torch.from_numpy(shared_model_modulation.copy())
        else:
            block.adaLN_modulation = FixedModulation(
                torch.from_numpy(total_modulation.copy())
            )
            modulation_input = torch.zeros(
                (BATCH_SIZE, CHANNELS), dtype=torch.float32
            )

        args = [residual, modulation_input]
        if "Cross" in class_name:
            args.append(torch.zeros((1, CHANNELS), dtype=torch.float32))
        captured, attention_input = capture_first_residual(
            block,
            tuple(args),
            attention_name,
            torch.from_numpy(ATTENTION_OUTPUT.copy()),
        )
        if not torch.equal(captured, direct_modulated.feats):
            raise AssertionError(
                f"{class_name} share_mod={share_mod} gate-plus-residual drift"
            )
        if conditioned_reference is None:
            conditioned_reference = attention_input
        elif not torch.equal(attention_input, conditioned_reference):
            raise AssertionError("share_mod changed consumer-side modulation semantics")
        output = captured.detach().cpu().numpy()
        modulated_runs.append(
            {
                "class": class_name,
                "share_mod": share_mod,
                "checkpoint": False,
                "output_f32le_sha256": little_endian_sha(output, "<f4"),
            }
        )

    if torch.equal(direct_plain.feats, direct_modulated.feats):
        raise AssertionError("plain and modulated consumer fixtures must differ")
    if not torch.equal(residual.feats, before_residual):
        raise AssertionError("upstream consumer execution mutated residual input")
    if not torch.equal(update.feats, before_update):
        raise AssertionError("upstream consumer execution mutated attention output")

    configs = []
    for name, config_path in config_paths.items():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        denoiser = payload["models"]["denoiser"]
        args = denoiser["args"]
        if denoiser["name"] != "ElasticSLatFlowModel":
            raise AssertionError(f"{name} configured model drift")
        if args.get("share_mod") is not True:
            raise AssertionError(f"{name} share_mod drift")
        if args.get("num_blocks") != 30:
            raise AssertionError(f"{name} block-count drift")
        configs.append(
            {
                "name": name,
                "path": CONFIG_PINS[name][0],
                "sha256": sha256(config_path),
                "model": denoiser["name"],
                "num_blocks": args["num_blocks"],
                "share_mod": args["share_mod"],
            }
        )

    stages = {
        "residual_input": RESIDUAL_INPUT,
        "attention_output": ATTENTION_OUTPUT,
        "gate_msa": gate_msa,
        "plain_residual": direct_plain.feats.detach().cpu().numpy(),
        "modulated_gate_plus_residual": direct_modulated.feats.detach()
        .cpu()
        .numpy(),
    }
    if not all(np.isfinite(stage).all() for stage in stages.values()):
        raise AssertionError("consumer oracle produced a non-finite value")

    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "sources": {
                name: {"path": SOURCE_PINS[name][0], "sha256": sha256(path)}
                for name, path in sources.items()
            },
            "generator": (
                "tools/trellis2_oracle/export_sparse_attention_consumers.py"
            ),
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "synthetic",
            "upstream_consumer_methods_executed": True,
            "upstream_attention_backend_executed": False,
            "upstream_checkpoint_path_executed": False,
        },
        "inventory": source_inventory(),
        "production_configuration": {
            "model": "ElasticSLatFlowModel",
            "base_model": "SLatFlowModel",
            "block": "ModulatedSparseTransformerCrossBlock",
            "attention_mode": "full",
            "share_mod": True,
            "config_count": len(configs),
            "configs": configs,
        },
        "contract": {
            "channels": CHANNELS,
            "num_heads": NUM_HEADS,
            "sequence_lengths": SEQUENCE_LENGTHS,
            "batch_broadcast_map": batch_map.tolist(),
            "plain_formula": "x + attention_output",
            "modulated_formula": "x + attention_output * gate_msa[batch]",
            "share_mod_changes_gate_source_only": True,
            "input_unchanged": True,
            "direct_sparse_arithmetic_reuses_coordinate_object": True,
            "consumer_boundary": "immediately after self-attention output",
            "not_attention_backend_parity": True,
            "not_transformer_block_parity": True,
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES.tolist(),
            "coordinates_i32le_sha256": little_endian_sha(COORDINATES, "<i4"),
        },
        "modulation": {
            "shared_block_modulation": block_modulation.tolist(),
            "shared_model_modulation": shared_model_modulation.tolist(),
            "total_modulation": total_modulation.tolist(),
        },
        "stages": {
            name: array.tolist()
            for name, array in stages.items()
        },
        "stage_f32le_sha256": {
            name: little_endian_sha(array, "<f4")
            for name, array in stages.items()
        },
        "executions": {
            "plain": plain_runs,
            "modulated": modulated_runs,
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
