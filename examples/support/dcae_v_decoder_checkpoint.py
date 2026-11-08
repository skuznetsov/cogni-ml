#!/usr/bin/env python3
"""Safely extract the pinned Hy-Nomas DCAE-V decoder into BF16 safetensors.

The source is a PyTorch ``.pt`` checkpoint. Loading is restricted to
``weights_only`` and memory-mapped CPU tensors. The converter retains the exact
``decoder.*`` names and BF16 values so native ``WeightBinding.source`` fields
can address the sidecar without key rewriting or numeric conversion.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import dcae_v_checkpoint as base


SCHEMA = "hy-nomas-dcae-v-decoder-checkpoint-v1"
VARIANT = base.VARIANT
PINNED_SOURCE_SHA256 = base.PINNED_SOURCE_SHA256
PINNED_SOURCE_BYTES = base.PINNED_SOURCE_BYTES
PINNED_SOURCE_REVISION = base.PINNED_SOURCE_REVISION
PINNED_UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
PINNED_UPSTREAM_MODULE_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
PINNED_UPSTREAM_MODULE_SHA256 = (
    "38163ebead9a6bb643d27843022c1757c45c6b8638c334a476aee5b373edf8c1"
)
EXPECTED_TENSOR_COUNT = 171
EXPECTED_BF16_BYTES = 2_443_960_856
EXPECTED_ENCODER_TENSOR_COUNT = 169
TILE_FRAMES = 40
TILE_LATENTS = 10
MAX_OUTPUT_BYTES = base.MAX_OUTPUT_BYTES
MAX_MANIFEST_BYTES = base.MAX_MANIFEST_BYTES
DEFAULT_UPSTREAM_MODULE = base.DEFAULT_UPSTREAM_MODULE
REPO_ROOT = Path(__file__).resolve().parents[2]

CheckpointError = base.CheckpointError


def _identity(path: Path) -> tuple[int, int, int, int]:
    return base._identity(path)


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    return base._sha256_file(path, chunk_bytes=chunk_bytes)


def _shape_product(shape: list[int] | tuple[int, ...]) -> int:
    return base._shape_product(shape)


def upstream_decoder_shapes(
    module_path: Path = DEFAULT_UPSTREAM_MODULE,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Build the exact pinned decoder state schema on ``meta``."""
    module_path = Path(module_path).expanduser()
    identity = _identity(module_path)
    source_sha256 = _sha256_file(module_path)
    if source_sha256 != PINNED_UPSTREAM_MODULE_SHA256:
        raise CheckpointError(
            "upstream DCAE-V module SHA-256 differs from the pinned source"
        )
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError(
            f"PyTorch is required to inspect the DCAE-V schema: {exc}"
        ) from None

    upstream = base._load_module(module_path)
    config = upstream.dc_ae_v_f32t4c32_config()
    if (
        config.latent_channels != 32
        or config.decode_temporal_tile_size != TILE_FRAMES
        or config.decode_temporal_tile_latent_size != TILE_LATENTS
        or config.decode_temporal_tile_overlap_factor != 0
        or config.use_feature_cache is not True
        or config.num_pad_frames != 0
        or config.decoder.latent_channels != 32
    ):
        raise CheckpointError(
            "upstream DCAE-V config does not match the pinned f32t4c32 decoder contract"
        )

    with torch.device("meta"):
        decoder = upstream.DCAEVDecoder(config.decoder)
    shapes: dict[str, list[int]] = {}
    for name, tensor in decoder.state_dict().items():
        canonical = name if name.startswith("decoder.") else f"decoder.{name}"
        if canonical in shapes:
            raise CheckpointError(f"duplicate upstream decoder state name: {canonical}")
        shapes[canonical] = [int(dimension) for dimension in tensor.shape]

    if _identity(module_path) != identity or _sha256_file(module_path) != source_sha256:
        raise CheckpointError(
            "upstream DCAE-V source changed while its schema was read"
        )
    if len(shapes) != EXPECTED_TENSOR_COUNT:
        raise CheckpointError(
            f"pinned decoder tensor count mismatch: {len(shapes)} != {EXPECTED_TENSOR_COUNT}"
        )
    payload_bytes = sum(_shape_product(shape) * 2 for shape in shapes.values())
    if payload_bytes != EXPECTED_BF16_BYTES:
        raise CheckpointError(
            f"pinned decoder BF16 byte count mismatch: {payload_bytes} != {EXPECTED_BF16_BYTES}"
        )

    return shapes, {
        "module_path": str(module_path.resolve()),
        "module_sha256": source_sha256,
        "checkout_revision": PINNED_UPSTREAM_REVISION,
        "module_history_revision": PINNED_UPSTREAM_MODULE_REVISION,
        "config": {
            "variant": VARIANT,
            "latent_channels": 32,
            "spatial_compression": 32,
            "temporal_compression": 4,
            "decode_temporal_tile_size": TILE_FRAMES,
            "decode_temporal_tile_latent_size": TILE_LATENTS,
            "decode_temporal_tile_overlap_factor": 0,
            "use_feature_cache": True,
        },
    }


def validate_decoder_state(
    state: Mapping[str, Any],
    expected_shapes: Mapping[str, list[int]],
    torch: Any,
) -> dict[str, Any]:
    """Check exact decoder names/shapes/BF16 and full-source ownership counts."""
    if not isinstance(state, Mapping):
        raise CheckpointError("selected checkpoint state is not a mapping")
    expected_names = set(expected_shapes)
    actual_names: set[str] = set()
    decoder_tensors: dict[str, Any] = {}
    encoder_count = 0
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise CheckpointError(
                "checkpoint state contains a non-string or empty tensor name"
            )
        if not isinstance(tensor, torch.Tensor):
            raise CheckpointError(f"checkpoint entry is not a tensor: {name}")
        if name.startswith("decoder."):
            actual_names.add(name)
            if name not in expected_names:
                raise CheckpointError(f"unexpected DCAE-V decoder tensor name: {name}")
            expected_shape = expected_shapes[name]
            if list(tensor.shape) != list(expected_shape):
                raise CheckpointError(
                    f"decoder shape mismatch for {name}: {list(tensor.shape)} != {list(expected_shape)}"
                )
            if tensor.dtype is not torch.bfloat16:
                raise CheckpointError(
                    f"decoder tensor is not BF16: {name} ({tensor.dtype})"
                )
            if tensor.device.type != "cpu":
                raise CheckpointError(f"decoder tensor is not on CPU: {name}")
            if not tensor.is_contiguous():
                raise CheckpointError(f"decoder tensor is not contiguous: {name}")
            decoder_tensors[name] = tensor
        elif name.startswith("encoder."):
            encoder_count += 1
        else:
            raise CheckpointError(f"unknown DCAE-V checkpoint tensor ownership: {name}")

    missing = sorted(expected_names - actual_names)
    if missing:
        raise CheckpointError(f"missing decoder tensors: {', '.join(missing[:4])}")
    if len(decoder_tensors) != EXPECTED_TENSOR_COUNT:
        raise CheckpointError(
            f"decoder tensor count mismatch: {len(decoder_tensors)} != {EXPECTED_TENSOR_COUNT}"
        )
    if encoder_count != EXPECTED_ENCODER_TENSOR_COUNT:
        raise CheckpointError(
            f"encoder tensor count mismatch: {encoder_count} != {EXPECTED_ENCODER_TENSOR_COUNT}"
        )
    payload_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in decoder_tensors.values()
    )
    if payload_bytes != EXPECTED_BF16_BYTES or payload_bytes > MAX_OUTPUT_BYTES:
        raise CheckpointError(
            f"decoder payload size is not admitted: {payload_bytes} bytes"
        )
    return decoder_tensors


def _validate_output_path(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise CheckpointError(
            f"{label} must be an absolute path outside the repository"
        )
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise CheckpointError(f"{label} parent must be a directory")
    resolved = parent / path.name
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved
    raise CheckpointError(f"{label} must be outside the repository")


def convert_checkpoint(
    source_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    *,
    expected_source_sha256: str,
    expected_source_bytes: int,
    source_revision: str,
    upstream_module_path: Path,
    expected_upstream_sha256: str,
    upstream_revision: str,
    source_repository: str = "Tencent-Hunyuan/Hy-Nomas",
) -> dict[str, Any]:
    """Extract only original decoder BF16 weights and write an authenticated sidecar."""
    source_path = Path(source_path).expanduser()
    output_path = Path(output_path).expanduser()
    manifest_path = (
        Path(manifest_path).expanduser()
        if manifest_path
        else Path(str(output_path) + ".manifest.json")
    )

    if not base._valid_sha256(expected_source_sha256):
        raise CheckpointError(
            "caller must provide a lowercase 64-character source SHA-256"
        )
    if expected_source_sha256 != PINNED_SOURCE_SHA256:
        raise CheckpointError(
            "checkpoint SHA-256 differs from the pinned Hy-Nomas source"
        )
    if (
        type(expected_source_bytes) is not int
        or expected_source_bytes != PINNED_SOURCE_BYTES
    ):
        raise CheckpointError(
            "checkpoint byte count differs from the pinned Hy-Nomas source"
        )
    if source_revision != PINNED_SOURCE_REVISION:
        raise CheckpointError(
            "checkpoint revision differs from the pinned Hy-Nomas source"
        )
    if not base._valid_sha256(expected_upstream_sha256):
        raise CheckpointError(
            "caller must provide a lowercase 64-character upstream SHA-256"
        )
    if expected_upstream_sha256 != PINNED_UPSTREAM_MODULE_SHA256:
        raise CheckpointError(
            "upstream DCAE-V module SHA-256 differs from the pinned source"
        )
    if upstream_revision != PINNED_UPSTREAM_REVISION:
        raise CheckpointError(
            "upstream DCAE-V checkout revision differs from the pinned source"
        )

    source_identity = _identity(source_path)
    if source_identity[2] != expected_source_bytes:
        raise CheckpointError(
            f"checkpoint size mismatch: {source_identity[2]} != {expected_source_bytes}"
        )
    observed_source_sha256 = _sha256_file(source_path)
    if observed_source_sha256 != expected_source_sha256:
        raise CheckpointError(
            "checkpoint SHA-256 differs from the caller-pinned source"
        )
    if _identity(source_path) != source_identity:
        raise CheckpointError("checkpoint changed while its source digest was computed")

    upstream_module_path = Path(upstream_module_path).expanduser()
    upstream_identity = _identity(upstream_module_path)
    observed_upstream_sha256 = _sha256_file(upstream_module_path)
    if observed_upstream_sha256 != expected_upstream_sha256:
        raise CheckpointError(
            "upstream DCAE-V module SHA-256 differs from the caller-pinned source"
        )
    if _identity(upstream_module_path) != upstream_identity:
        raise CheckpointError(
            "upstream DCAE-V source changed while its digest was computed"
        )

    output_path = _validate_output_path(output_path, "output weights")
    manifest_path = _validate_output_path(manifest_path, "output manifest")
    if output_path.resolve(strict=False) == manifest_path.resolve(strict=False):
        raise CheckpointError("output weights and manifest paths must be distinct")
    if output_path == source_path.resolve(
        strict=True
    ) or manifest_path == source_path.resolve(strict=True):
        raise CheckpointError("output paths alias the source checkpoint")
    if output_path == upstream_module_path.resolve(
        strict=True
    ) or manifest_path == upstream_module_path.resolve(strict=True):
        raise CheckpointError("output paths alias the upstream DCAE-V module")
    for path in (output_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise CheckpointError(f"refusing to overwrite an existing output: {path}")

    expected_shapes, upstream_metadata = upstream_decoder_shapes(upstream_module_path)

    try:
        import torch
    except ImportError as exc:
        raise CheckpointError(
            f"PyTorch is required for checkpoint conversion: {exc}"
        ) from None
    try:
        checkpoint_object = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise CheckpointError(
            f"safe memory-mapped weights_only=True PyTorch checkpoint load failed: {exc}"
        ) from None
    state, state_dict_selection = base._select_state_dict(checkpoint_object)
    decoder_tensors = validate_decoder_state(state, expected_shapes, torch)
    del checkpoint_object, state
    if _identity(source_path) != source_identity:
        raise CheckpointError(
            "checkpoint changed while its mapped tensors were inspected"
        )
    if (
        _identity(upstream_module_path) != upstream_identity
        or _sha256_file(upstream_module_path) != observed_upstream_sha256
    ):
        raise CheckpointError(
            "upstream DCAE-V source changed while its schema was inspected"
        )

    try:
        from safetensors import __version__ as safetensors_version
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise CheckpointError(
            f"safetensors is required for checkpoint conversion: {exc}"
        ) from None

    weights_fd, weights_temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent
    )
    os.close(weights_fd)
    weights_temp = Path(weights_temp_name)
    manifest_temp: Path | None = None
    committed_weights = False
    try:
        metadata = {
            "format": "pt",
            "dcae_v_schema": SCHEMA,
            "dcae_v_variant": VARIANT,
            "source_sha256": observed_source_sha256,
            "source_revision": source_revision,
            "upstream_revision": upstream_revision,
            "upstream_module_sha256": observed_upstream_sha256,
            "decoder_tensor_count": str(len(decoder_tensors)),
            "decoder_payload_bytes": str(EXPECTED_BF16_BYTES),
        }
        # Source tensors are already contiguous CPU BF16. This writes their
        # original values under unchanged `decoder.*` names.
        save_file(decoder_tensors, str(weights_temp), metadata=metadata)
        with weights_temp.open("rb") as stream:
            os.fsync(stream.fileno())
        artifact_sha256 = _sha256_file(weights_temp)
        artifact_bytes = _identity(weights_temp)[2]
        if artifact_bytes <= 0 or artifact_bytes > MAX_OUTPUT_BYTES:
            raise CheckpointError(
                f"generated safetensors size is outside its bound: {artifact_bytes}"
            )
        with safe_open(str(weights_temp), framework="pt", device="cpu") as archive:
            observed_names = set(archive.keys())
            if observed_names != set(expected_shapes):
                raise CheckpointError(
                    "generated safetensors tensor names differ from the decoder schema"
                )
            tensor_records: dict[str, dict[str, Any]] = {}
            for name in sorted(observed_names):
                tensor = archive.get_tensor(name)
                if tensor.dtype is not torch.bfloat16 or list(tensor.shape) != list(
                    expected_shapes[name]
                ):
                    raise CheckpointError(
                        f"generated safetensors tensor does not match schema: {name}"
                    )
                tensor_records[name] = {"shape": list(tensor.shape), "dtype": "BF16"}

        manifest = {
            "schema": SCHEMA,
            "source": {
                "repository": source_repository,
                "revision": source_revision,
                "filename": source_path.name,
                "bytes": source_identity[2],
                "sha256": observed_source_sha256,
                "state_dict_selection": state_dict_selection,
                "load_policy": "torch.load(weights_only=True,mmap=True,map_location=cpu)",
            },
            "upstream": upstream_metadata,
            "conversion": {
                "action": "copy decoder BF16 tensors without value conversion",
                "encoder_included": False,
                "decoder_included": True,
                "source_name_prefix": "decoder.",
                "output_name_prefix": "decoder.",
                "tensor_count": len(tensor_records),
                "payload_bytes": EXPECTED_BF16_BYTES,
                "safetensors_version": safetensors_version,
            },
            "output": {
                "filename": output_path.name,
                "bytes": artifact_bytes,
                "sha256": artifact_sha256,
                "tensors": tensor_records,
            },
        }
        encoded_manifest = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(encoded_manifest) > MAX_MANIFEST_BYTES:
            raise CheckpointError("generated manifest exceeds its size bound")
        manifest_temp = base._write_temp_bytes(
            manifest_path.parent, f".{manifest_path.name}.", encoded_manifest
        )

        if output_path.exists() or manifest_path.exists():
            raise CheckpointError(
                "output path appeared during conversion; refusing replacement"
            )
        os.replace(weights_temp, output_path)
        committed_weights = True
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
        return manifest
    except BaseException:
        if committed_weights:
            output_path.unlink(missing_ok=True)
        raise
    finally:
        weights_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="pinned original DCAE-V .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new decoder-only BF16 safetensors path",
    )
    parser.add_argument(
        "--manifest", type=Path, help="manifest path (default: OUTPUT.manifest.json)"
    )
    parser.add_argument(
        "--source-sha256", required=True, help="pinned full source SHA-256"
    )
    parser.add_argument(
        "--source-bytes",
        required=True,
        type=int,
        help="pinned full source file byte count",
    )
    parser.add_argument(
        "--source-revision", required=True, help="pinned Hy-Nomas checkpoint revision"
    )
    parser.add_argument(
        "--upstream-module",
        required=True,
        type=Path,
        help="pinned upstream DCAE-V Python module",
    )
    parser.add_argument(
        "--upstream-sha256", required=True, help="pinned upstream module SHA-256"
    )
    parser.add_argument(
        "--upstream-revision", required=True, help="pinned upstream checkout revision"
    )
    args = parser.parse_args(argv)
    try:
        manifest = convert_checkpoint(
            args.source,
            args.output,
            args.manifest,
            expected_source_sha256=args.source_sha256,
            expected_source_bytes=args.source_bytes,
            source_revision=args.source_revision,
            upstream_module_path=args.upstream_module,
            expected_upstream_sha256=args.upstream_sha256,
            upstream_revision=args.upstream_revision,
        )
    except (CheckpointError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "output": manifest["output"],
                "manifest": str(
                    args.manifest or Path(str(args.output) + ".manifest.json")
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
