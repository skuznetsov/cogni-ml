#!/usr/bin/env python3
"""Safely extract the pinned Hy-Nomas DCAE-V encoder into BF16 safetensors.

The source is a PyTorch ``.pt`` checkpoint. Loading is restricted to PyTorch's
``weights_only`` unpickler and memory mapped; the decoder is omitted. Encoder
names and shapes are kept exactly as stored (``encoder.*``), so Crystal's
``WeightBinding.source`` can address them directly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from collections.abc import Mapping
from typing import Any


SCHEMA = "hy-nomas-dcae-v-encoder-checkpoint-v1"
VARIANT = "dc-ae-v-1.0-f32t4c32-bf16"
PINNED_SOURCE_SHA256 = "ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"
PINNED_SOURCE_BYTES = 4_277_389_995
PINNED_SOURCE_REVISION = "bb6a8ca451bfe84947f0ea1c6d51957f983ffffe"
EXPECTED_TENSOR_COUNT = 169
EXPECTED_BF16_BYTES = 1_833_310_016
MAX_SOURCE_BYTES = 8 * 1024**3
MAX_OUTPUT_BYTES = 4 * 1024**3
MAX_MANIFEST_BYTES = 2 * 1024**2
DEFAULT_UPSTREAM_MODULE = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/"
    "hy-nomas-20261024/hymm/models/vae/autoencoder_dcae_v.py"
)


class CheckpointError(ValueError):
    """A source checkpoint or generated encoder artifact violates its contract."""


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identity(path: Path) -> tuple[int, int, int, int]:
    if path.is_symlink():
        raise CheckpointError(f"symlink is not admitted: {path}")
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CheckpointError(f"cannot stat {path}: {exc}") from None
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointError(f"expected a regular file: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path) -> Any:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise CheckpointError(f"upstream DCAE-V module is not a file: {path}")
    module_name = "_hy_nomas_pinned_autoencoder_dcae_v"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CheckpointError(f"cannot import upstream DCAE-V module: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and postponed annotations expect the module to be registered.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise CheckpointError(f"cannot import upstream DCAE-V module: {exc}") from None
    return module


def upstream_encoder_shapes(module_path: Path = DEFAULT_UPSTREAM_MODULE) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Build the exact encoder state schema on ``meta`` without weight allocation."""
    module_path = Path(module_path).expanduser()
    identity = _identity(module_path)
    source_sha256 = _sha256_file(module_path)
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError(f"PyTorch is required to inspect the DCAE-V schema: {exc}") from None

    upstream = _load_module(module_path)
    config = upstream.dc_ae_v_f32t4c32_config()
    if (
        config.latent_channels != 32
        or config.encode_temporal_tile_size != 40
        or config.encode_temporal_tile_latent_size != 10
        or config.encoder.latent_channels != 32
    ):
        raise CheckpointError("upstream DCAE-V config does not match the pinned f32t4c32 contract")
    with torch.device("meta"):
        encoder = upstream.DCAEVEncoder(config.encoder)
    shapes: dict[str, list[int]] = {}
    for name, tensor in encoder.state_dict().items():
        canonical = name if name.startswith("encoder.") else f"encoder.{name}"
        if canonical in shapes:
            raise CheckpointError(f"duplicate upstream encoder state name: {canonical}")
        shapes[canonical] = [int(dimension) for dimension in tensor.shape]
    if _identity(module_path) != identity or _sha256_file(module_path) != source_sha256:
        raise CheckpointError("upstream DCAE-V source changed while its schema was read")
    if len(shapes) != EXPECTED_TENSOR_COUNT:
        raise CheckpointError(
            f"pinned encoder tensor count mismatch: {len(shapes)} != {EXPECTED_TENSOR_COUNT}"
        )
    nbytes = sum(_shape_product(shape) * 2 for shape in shapes.values())
    if nbytes != EXPECTED_BF16_BYTES:
        raise CheckpointError(
            f"pinned encoder BF16 byte count mismatch: {nbytes} != {EXPECTED_BF16_BYTES}"
        )
    return shapes, {
        "module_path": str(module_path.resolve()),
        "module_sha256": source_sha256,
        "config": {
            "variant": VARIANT,
            "latent_channels": 32,
            "spatial_compression": 32,
            "temporal_compression": 4,
            "encode_temporal_tile_size": 40,
            "encode_temporal_tile_latent_size": 10,
        },
    }


def _shape_product(shape: list[int] | tuple[int, ...]) -> int:
    if not isinstance(shape, (list, tuple)) or not shape:
        raise CheckpointError("tensor shape must be a nonempty sequence")
    product = 1
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise CheckpointError("tensor shape dimensions must be positive integers")
        product *= dimension
        if product > MAX_SOURCE_BYTES:
            raise CheckpointError("tensor shape exceeds the checkpoint size bound")
    return product


def _select_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(checkpoint, Mapping):
        raise CheckpointError("PyTorch checkpoint root must be a mapping")
    if "ema_model_state_dict" in checkpoint:
        state = checkpoint["ema_model_state_dict"]
        selected = "ema_model_state_dict"
        if isinstance(state, Mapping) and len(state) == 1:
            nested_name, nested = next(iter(state.items()))
            if isinstance(nested, Mapping):
                state = nested
                selected += f".{nested_name}"
    else:
        state = checkpoint.get("model_state_dict", checkpoint)
        selected = "model_state_dict" if "model_state_dict" in checkpoint else "root"
    if not isinstance(state, Mapping):
        raise CheckpointError(f"selected checkpoint state {selected} is not a mapping")
    return state, selected


def validate_encoder_state(
    state: Mapping[str, Any], expected_shapes: Mapping[str, list[int]], torch: Any
) -> dict[str, Any]:
    """Check exact encoder names/shapes/BF16 and reject unknown state ownership."""
    expected_names = set(expected_shapes)
    actual_names: set[str] = set()
    encoder_tensors: dict[str, Any] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("checkpoint state contains a non-string or empty tensor name")
        if not isinstance(tensor, torch.Tensor):
            raise CheckpointError(f"checkpoint entry is not a tensor: {name}")
        if name.startswith("encoder."):
            actual_names.add(name)
            if name not in expected_names:
                raise CheckpointError(f"unexpected DCAE-V encoder tensor name: {name}")
            expected_shape = expected_shapes[name]
            if list(tensor.shape) != list(expected_shape):
                raise CheckpointError(
                    f"encoder shape mismatch for {name}: {list(tensor.shape)} != {list(expected_shape)}"
                )
            if tensor.dtype is not torch.bfloat16:
                raise CheckpointError(f"encoder tensor is not BF16: {name} ({tensor.dtype})")
            if tensor.device.type != "cpu":
                raise CheckpointError(f"encoder tensor is not on CPU: {name}")
            if not tensor.is_contiguous():
                raise CheckpointError(f"encoder tensor is not contiguous: {name}")
            encoder_tensors[name] = tensor
        elif name.startswith("decoder."):
            continue
        else:
            raise CheckpointError(f"unknown DCAE-V checkpoint tensor ownership: {name}")
    missing = sorted(expected_names - actual_names)
    if missing:
        raise CheckpointError(f"missing encoder tensors: {', '.join(missing[:4])}")
    if len(encoder_tensors) != EXPECTED_TENSOR_COUNT:
        raise CheckpointError(
            f"encoder tensor count mismatch: {len(encoder_tensors)} != {EXPECTED_TENSOR_COUNT}"
        )
    payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in encoder_tensors.values())
    if payload_bytes != EXPECTED_BF16_BYTES or payload_bytes > MAX_OUTPUT_BYTES:
        raise CheckpointError(f"encoder payload size is not admitted: {payload_bytes} bytes")
    return encoder_tensors


def _write_temp_bytes(parent: Path, prefix: str, payload: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".partial", dir=parent)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def convert_checkpoint(
    source_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    *,
    expected_source_sha256: str,
    expected_source_bytes: int,
    source_revision: str,
    upstream_module_path: Path = DEFAULT_UPSTREAM_MODULE,
    expected_shapes: Mapping[str, list[int]] | None = None,
    source_repository: str = "Tencent-Hunyuan/Hy-Nomas",
) -> dict[str, Any]:
    """Extract only encoder BF16 weights and write an authenticated sidecar."""
    source_path = Path(source_path).expanduser()
    output_path = Path(output_path).expanduser()
    manifest_path = Path(manifest_path).expanduser() if manifest_path else Path(str(output_path) + ".manifest.json")
    if not _valid_sha256(expected_source_sha256):
        raise CheckpointError("caller must provide a lowercase 64-character source SHA-256")
    if expected_source_sha256 != PINNED_SOURCE_SHA256:
        raise CheckpointError("checkpoint SHA-256 differs from the pinned Hy-Nomas source")
    if type(expected_source_bytes) is not int or not (0 < expected_source_bytes <= MAX_SOURCE_BYTES):
        raise CheckpointError("caller must provide a positive bounded source byte count")
    if expected_source_bytes != PINNED_SOURCE_BYTES:
        raise CheckpointError("checkpoint byte count differs from the pinned Hy-Nomas source")
    if source_revision != PINNED_SOURCE_REVISION:
        raise CheckpointError("checkpoint revision differs from the pinned Hy-Nomas source")
    source_identity = _identity(source_path)
    if source_identity[2] != expected_source_bytes:
        raise CheckpointError(
            f"checkpoint size mismatch: {source_identity[2]} != {expected_source_bytes}"
        )
    if source_identity[2] > MAX_SOURCE_BYTES:
        raise CheckpointError("checkpoint exceeds the source size bound")
    observed_source_sha256 = _sha256_file(source_path)
    if observed_source_sha256 != expected_source_sha256:
        raise CheckpointError("checkpoint SHA-256 differs from the caller-pinned source")
    if _identity(source_path) != source_identity:
        raise CheckpointError("checkpoint changed while its source digest was computed")

    output_parent = output_path.parent.resolve(strict=True)
    manifest_parent = manifest_path.parent.resolve(strict=True)
    if not output_parent.is_dir() or not manifest_parent.is_dir():
        raise CheckpointError("output and manifest parents must be directories")
    output_path = output_parent / output_path.name
    manifest_path = manifest_parent / manifest_path.name
    if output_path.resolve(strict=False) == manifest_path.resolve(strict=False):
        raise CheckpointError("output weights and manifest paths must be distinct")
    for path in (output_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise CheckpointError(f"refusing to overwrite an existing output: {path}")
    if output_path.resolve(strict=False) == source_path.resolve(strict=True):
        raise CheckpointError("output path aliases the source checkpoint")

    try:
        import torch
    except ImportError as exc:
        raise CheckpointError(f"PyTorch is required for checkpoint conversion: {exc}") from None
    upstream_metadata: dict[str, Any] = {}
    if expected_shapes is None:
        expected_shapes, upstream_metadata = upstream_encoder_shapes(upstream_module_path)
    else:
        expected_shapes = {name: list(shape) for name, shape in expected_shapes.items()}
        upstream_path = Path(upstream_module_path).expanduser()
        upstream_metadata = {
            "module_path": str(upstream_path),
            "module_sha256": _sha256_file(upstream_path),
            "config": {"variant": VARIANT},
        }

    try:
        checkpoint = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise CheckpointError(f"safe memory-mapped PyTorch checkpoint load failed: {exc}") from None
    state, state_dict_selection = _select_state_dict(checkpoint)
    encoder_tensors = validate_encoder_state(state, expected_shapes, torch)
    del checkpoint, state
    if _identity(source_path) != source_identity:
        raise CheckpointError("checkpoint changed while its mapped tensors were inspected")

    try:
        from safetensors import __version__ as safetensors_version
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise CheckpointError(f"safetensors is required for checkpoint conversion: {exc}") from None

    weights_fd, weights_temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".partial", dir=output_parent
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
            "encoder_tensor_count": str(len(encoder_tensors)),
            "encoder_payload_bytes": str(EXPECTED_BF16_BYTES),
        }
        # No conversion is performed: contiguous CPU BF16 source tensors are
        # written with their original `encoder.*` names.
        save_file(encoder_tensors, str(weights_temp), metadata=metadata)
        with weights_temp.open("rb") as stream:
            os.fsync(stream.fileno())
        artifact_sha256 = _sha256_file(weights_temp)
        artifact_bytes = _identity(weights_temp)[2]
        if artifact_bytes <= 0 or artifact_bytes > MAX_OUTPUT_BYTES:
            raise CheckpointError(f"generated safetensors size is outside its bound: {artifact_bytes}")
        with safe_open(str(weights_temp), framework="pt", device="cpu") as archive:
            observed_names = set(archive.keys())
            if observed_names != set(expected_shapes):
                raise CheckpointError("generated safetensors tensor names differ from the encoder schema")
            tensor_records: dict[str, dict[str, Any]] = {}
            for name in sorted(observed_names):
                tensor = archive.get_tensor(name)
                if tensor.dtype is not torch.bfloat16 or list(tensor.shape) != list(expected_shapes[name]):
                    raise CheckpointError(f"generated safetensors tensor does not match schema: {name}")
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
                "action": "copy encoder BF16 tensors without value conversion",
                "decoder_included": False,
                "source_name_prefix": "encoder.",
                "output_name_prefix": "encoder.",
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
        manifest_temp = _write_temp_bytes(manifest_parent, f".{manifest_path.name}.", encoded_manifest)

        if output_path.exists() or manifest_path.exists():
            raise CheckpointError("output path appeared during conversion; refusing replacement")
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
    parser.add_argument("--source", required=True, type=Path, help="original pinned DCAE-V .pt checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="new encoder-only BF16 safetensors path")
    parser.add_argument("--manifest", type=Path, help="manifest path (default: OUTPUT.manifest.json)")
    parser.add_argument("--source-sha256", required=True, help="caller-pinned full source SHA-256")
    parser.add_argument("--source-bytes", required=True, type=int, help="caller-pinned full source file byte count")
    parser.add_argument("--source-revision", required=True, help="pinned Tencent-Hunyuan/Hy-Nomas revision")
    parser.add_argument("--upstream-module", type=Path, default=DEFAULT_UPSTREAM_MODULE)
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
        )
    except (CheckpointError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "schema": manifest["schema"],
        "output": manifest["output"],
        "manifest": str(args.manifest or Path(str(args.output) + ".manifest.json")),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
