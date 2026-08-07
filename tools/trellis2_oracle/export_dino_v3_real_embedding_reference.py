#!/usr/bin/env python3
"""Export a source-backed real-weight DINOv3 embedding-boundary reference.

This probe intentionally stops at the embedding boundary: patch projection,
CLS/register prefix concatenation, dynamic patch coordinates, and 2D RoPE
cos/sin.  It reads only the four certified embedding roles from a pinned
F32 safetensors checkpoint.  The current local Transformers runtime does not
provide DINOv3, so the reference uses ``torch.nn.functional.conv2d`` plus the
same source-pinned formulas rather than claiming to execute Transformers 5.8.1.

Outputs are little-endian F32 files so a Crystal probe can compare every value
without relying on a shared implementation or a sparse hand-picked sample.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open


SCHEMA = "cogni-ml/trellis2/dino-v3-real-embedding-reference/v1"
TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MODEL_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
CHECKPOINT_BYTES = 1_212_559_808
CHECKPOINT_SHA256 = (
    "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
)
TRANSFORMERS_MODELING_SHA256 = (
    "6073b7665eea50fb2260d86984011e6af8ed68cbd37ae57a6095e5e90e0eea34"
)
TRANSFORMERS_CONFIG_SHA256 = (
    "9a13d3c9ea8020aaed7057db28a7ffe0ffd9bb094a6178261bcc45ed04e9bbfc"
)

EDGE = 512
PATCH_SIZE = 16
CHANNELS = 3
HIDDEN_SIZE = 1024
HEADS = 16
HEAD_DIM = HIDDEN_SIZE // HEADS
REGISTER_TOKENS = 4
ROPE_THETA = 100.0

# The Crystal scalar F32 accumulation and the PyTorch CPU kernel are expected
# to differ by a small, measurable amount.  The Crystal consumer also pins an
# upper bound so a generated metadata file cannot silently widen acceptance.
# The observed Crystal-vs-PyTorch maximum is 1.43e-6 on this host; keep a
# roughly 7x margin without accepting a broad numerical drift.
ABSOLUTE_TOLERANCE = 1.0e-5
RELATIVE_TOLERANCE = 1.0e-5

ROLE_NAMES = {
    "embedding.patch_weight": "embeddings.patch_embeddings.weight",
    "embedding.patch_bias": "embeddings.patch_embeddings.bias",
    "embedding.cls_token": "embeddings.cls_token",
    "embedding.register_tokens": "embeddings.register_tokens",
}


def f32le_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )


def f32le_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(f32le_bytes(tensor)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_input(edge: int) -> torch.Tensor:
    """Return deterministic NCHW F32 data with both signs and no zero case."""

    channels = torch.arange(CHANNELS, dtype=torch.int64)[None, :, None, None]
    rows = torch.arange(edge, dtype=torch.int64)[None, None, :, None]
    columns = torch.arange(edge, dtype=torch.int64)[None, None, None, :]
    values = ((17 * channels + 3 * rows + 5 * columns) % 257) - 128
    return values.to(torch.float32) / 128.0


def patch_coordinates(patch_edge: int) -> torch.Tensor:
    """Match get_patches_center_coordinates in the pinned DINOv3 source."""

    indices = torch.arange(patch_edge, dtype=torch.float32)
    centers = ((indices + 0.5) / float(patch_edge)) * 2.0 - 1.0
    rows, columns = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack((rows.reshape(-1), columns.reshape(-1)), dim=1).contiguous()


def rope(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build source-equivalent [patches, head_dim] 2D RoPE tensors."""

    frequency_count = HEAD_DIM // 4
    half_dim = HEAD_DIM // 2
    exponent_step = 4.0 / float(HEAD_DIM)
    exponents = torch.arange(frequency_count, dtype=torch.float64) * exponent_step
    inverse_frequencies = (ROPE_THETA**exponents).reciprocal().to(torch.float32)

    dimensions = torch.arange(HEAD_DIM, dtype=torch.int64)
    half_indices = dimensions % half_dim
    axes = half_indices // frequency_count
    frequencies = half_indices % frequency_count
    selected_coordinates = coordinates[:, axes]
    selected_frequencies = inverse_frequencies[frequencies]
    two_pi = torch.tensor(2.0 * np.pi, dtype=torch.float32)
    angles = two_pi * selected_coordinates * selected_frequencies
    return torch.cos(angles).contiguous(), torch.sin(angles).contiguous()


def load_roles(checkpoint: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        roles = {
            role: handle.get_tensor(name).detach().to(torch.float32).contiguous()
            for role, name in ROLE_NAMES.items()
        }

    expected = {
        "embedding.patch_weight": ((HIDDEN_SIZE, CHANNELS, PATCH_SIZE, PATCH_SIZE), torch.float32),
        "embedding.patch_bias": ((HIDDEN_SIZE,), torch.float32),
        "embedding.cls_token": ((1, 1, HIDDEN_SIZE), torch.float32),
        "embedding.register_tokens": ((1, REGISTER_TOKENS, HIDDEN_SIZE), torch.float32),
    }
    for role, (shape, dtype) in expected.items():
        tensor = roles[role]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise RuntimeError(
                f"{role} has shape/dtype {tuple(tensor.shape)}/{tensor.dtype}, "
                f"expected {shape}/{dtype}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{role} contains non-finite values")
    return roles


def export(checkpoint: Path, output: Path) -> None:
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise RuntimeError(
            f"checkpoint byte length {checkpoint.stat().st_size} != {CHECKPOINT_BYTES}"
        )
    checkpoint_sha256 = file_sha256(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint sha256 {checkpoint_sha256} != pinned {CHECKPOINT_SHA256}"
        )

    # Avoid backend/thread drift in the independent CPU reference.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    roles = load_roles(checkpoint)
    image = recipe_input(EDGE)
    patch_edge = EDGE // PATCH_SIZE

    with torch.no_grad():
        patches = functional.conv2d(
            image,
            roles["embedding.patch_weight"],
            roles["embedding.patch_bias"],
            stride=PATCH_SIZE,
        ).flatten(2).transpose(1, 2).contiguous()
        prefix = torch.cat(
            (
                roles["embedding.cls_token"],
                roles["embedding.register_tokens"],
                patches,
            ),
            dim=1,
        ).contiguous()
        coordinates = patch_coordinates(patch_edge)
        rope_cos, rope_sin = rope(coordinates)

    outputs = {
        "patches": patches,
        "embeddings": prefix,
        "coordinates": coordinates,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for name, tensor in outputs.items():
        filename = f"{name}.f32le"
        payload = f32le_bytes(tensor)
        (output / filename).write_bytes(payload)
        output_files[name] = {
            "file": filename,
            "shape": list(tensor.shape),
            "byte_length": len(payload),
            "f32le_sha256": hashlib.sha256(payload).hexdigest(),
        }

    parameter_bytes = b"".join(
        f32le_bytes(roles[role])
        for role in (
            "embedding.patch_weight",
            "embedding.patch_bias",
            "embedding.cls_token",
            "embedding.register_tokens",
        )
    )
    role_metadata = {
        role: {
            "tensor_name": tensor_name,
            "dtype": "F32",
            "shape": list(roles[role].shape),
            "f32le_sha256": f32le_sha256(roles[role]),
        }
        for role, tensor_name in ROLE_NAMES.items()
    }

    metadata = {
        "schema": SCHEMA,
        "reference": {
            "kind": "source-backed-independent-boundary",
            "scope": "real F32 embedding roles at 512px; no transformer blocks",
            "backend": "torch.nn.functional.conv2d plus explicit F32 prefix/coordinates/RoPE",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "transformers_version": None,
            "transformers_dinov3_runtime_available": False,
            "trellis_revision": TRELLIS_PIN,
            "transformers_modeling_sha256": TRANSFORMERS_MODELING_SHA256,
            "transformers_config_sha256": TRANSFORMERS_CONFIG_SHA256,
        },
        "model": {
            "name": MODEL,
            "revision": MODEL_REVISION,
            "edge": EDGE,
            "patch_size": PATCH_SIZE,
            "num_channels": CHANNELS,
            "hidden_size": HIDDEN_SIZE,
            "num_attention_heads": HEADS,
            "num_register_tokens": REGISTER_TOKENS,
            "rope_theta": ROPE_THETA,
        },
        "checkpoint": {
            "basename": checkpoint.name,
            "byte_length": CHECKPOINT_BYTES,
            "sha256": checkpoint_sha256,
        },
        "roles": role_metadata,
        "parameter_order": list(ROLE_NAMES.keys()),
        "parameter_f32le_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        "input": {
            "shape": list(image.shape),
            "recipe": "x[0,c,y,x]=(((17*c+3*y+5*x)%257)-128)/128",
            "f32le_sha256": f32le_sha256(image),
            "minimum": float(image.min().item()),
            "maximum": float(image.max().item()),
        },
        "outputs": output_files,
        "tolerance": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": RELATIVE_TOLERANCE,
            "comparison": "abs(actual-reference) <= absolute + relative*abs(reference)",
        },
    }
    (output / "reference.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
