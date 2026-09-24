#!/usr/bin/env python3
"""Create a checksummed Qwen-Image 2.1 bundle with native retained embeddings.

This is a diagnostic bundle transform only. It replaces the retained
``encoder_hidden_states`` tensor in an existing CPU-BF16 conditioning bundle
with BF16 values from a pinned-metadata, checksummed native sidecar, expanded to the
bundle's required float32-le representation. Masks and seeded latents remain
byte-identical. It does not run the DiT, scheduler, or VAE.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


BUNDLE_SCHEMA = "qwen-image21-conditioning"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "qwen_image21_conditioning.json"
BUNDLE_PAYLOAD_NAME = "qwen_image21_conditioning.bin"
MAX_IMAGE_SIDE = 4096
BUNDLE_TENSOR_NAMES = (
    "encoder_hidden_states",
    "encoder_hidden_states_mask",
    "encoder_img_mask",
    "initial_target_latents",
)
NATIVE_SCHEMA = "qwen3vl-retained-embeddings"
NATIVE_SCHEMA_VERSION = 1
NATIVE_DROP_IDX = 14
CONTEXT_DIM = 4096
NATIVE_SHAPE = [10, CONTEXT_DIM]
NATIVE_DTYPE = "bfloat16-le"
PINNED_REFERENCE_FIXTURE_SHA256 = "3edcd7bf7964237d649a43c35cd82f6d6bd7b15835fddec2b1fe42f3a89b1e07"
# SHA of the BF16 pre_final_rmsnorm_embeddings tensor inside that fixture,
# not the fixture payload SHA or the conditioning bundle's F32 payload SHA.
PINNED_REFERENCE_EMBEDDINGS_SHA256 = "4de67b800ab96fb9d71b5eecf0b6c20c61758450bd98b1e8d97e9707fdf8b388"
PINNED_PROMPT = "red cube"
PINNED_MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value, raw


def _safe_sibling(directory: Path, raw_name: Any, description: str) -> Path:
    if (
        not isinstance(raw_name, str)
        or not raw_name
        or raw_name in {".", ".."}
        or "/" in raw_name
        or "\\" in raw_name
        or PurePosixPath(raw_name).name != raw_name
    ):
        raise ValueError(f"{description} payload_file must be a sibling basename")
    path = directory / raw_name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} payload is missing or is not a regular file")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_baseline(bundle_dir: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    if not bundle_dir.is_dir():
        raise ValueError(f"baseline bundle directory does not exist: {bundle_dir}")
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"baseline manifest is missing or is not a regular file: {manifest_path}")
    manifest, _ = _read_json(manifest_path, "baseline conditioning manifest")
    schema_version = manifest.get("schema_version")
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or not _is_int(schema_version)
        or schema_version != BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("baseline is not a supported qwen-image21-conditioning bundle")

    model = manifest.get("model")
    if (
        not isinstance(model, dict)
        or model.get("repo") != "Qwen/Qwen-Image-2.1"
        or not isinstance(model.get("revision"), str)
        or not REVISION_RE.fullmatch(model["revision"])
        or model["revision"] != PINNED_MODEL_REVISION
    ):
        raise ValueError("baseline model must identify a pinned Qwen-Image 2.1 revision")
    prompt = manifest.get("prompt")
    if prompt != PINNED_PROMPT:
        raise ValueError(f"baseline prompt must match the pinned native fixture prompt {PINNED_PROMPT!r}")

    payload_path = _safe_sibling(bundle_dir, manifest.get("payload_file"), "baseline")
    if payload_path.name != BUNDLE_PAYLOAD_NAME:
        raise ValueError(f"baseline payload_file must name {BUNDLE_PAYLOAD_NAME!r}")
    try:
        payload = payload_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read baseline conditioning payload: {exc}") from exc
    expected_sha = manifest.get("payload_sha256")
    if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
        raise ValueError("baseline manifest has an invalid payload SHA-256")
    if manifest.get("payload_nbytes") != len(payload):
        raise ValueError("baseline payload length does not match its manifest")
    if _sha256(payload) != expected_sha:
        raise ValueError("baseline payload SHA-256 does not match its manifest")

    image = manifest.get("image")
    if not isinstance(image, dict):
        raise ValueError("baseline manifest is missing image metadata")
    width, height = image.get("width"), image.get("height")
    if (
        not _is_int(width)
        or not _is_int(height)
        or width <= 0
        or height <= 0
        or width > MAX_IMAGE_SIDE
        or height > MAX_IMAGE_SIDE
        or width % 32
        or height % 32
        or image.get("vae_scale_factor") != 16
    ):
        raise ValueError("baseline image dimensions or VAE scale factor are invalid")
    latent_height, latent_width = height // 16, width // 16
    if (
        image.get("latent_height") != latent_height
        or image.get("latent_width") != latent_width
        or image.get("img_shapes") != [[1, latent_height, latent_width]]
    ):
        raise ValueError("baseline latent geometry metadata does not match image dimensions")
    noise = manifest.get("noise")
    runtime = manifest.get("runtime")
    if (
        not isinstance(noise, dict)
        or noise.get("source_dtype") != "bfloat16"
        or noise.get("generator_device") != "cpu"
        or not isinstance(runtime, dict)
        or runtime.get("device") != "cpu"
    ):
        raise ValueError("baseline must declare CPU BF16 conditioning and CPU noise generation")
    latent_shape = [latent_height * latent_width, 64]
    shapes_and_types = {
        "encoder_hidden_states": ("float32-le", NATIVE_SHAPE),
        "encoder_hidden_states_mask": ("uint8", [NATIVE_SHAPE[0]]),
        "encoder_img_mask": ("uint8", [NATIVE_SHAPE[0]]),
        "initial_target_latents": ("float32-le", latent_shape),
    }
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(BUNDLE_TENSOR_NAMES):
        raise ValueError("baseline tensor descriptors do not match the conditioning bundle layout")

    expected_offset = 0
    descriptors: dict[str, dict[str, Any]] = {}
    for name in BUNDLE_TENSOR_NAMES:
        descriptor = tensors.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"baseline tensor descriptor is invalid: {name}")
        expected_dtype, expected_shape = shapes_and_types[name]
        shape = descriptor.get("shape")
        if (
            descriptor.get("dtype") != expected_dtype
            or shape != expected_shape
            or not isinstance(shape, list)
            or any(not _is_int(dim) or dim <= 0 for dim in shape)
        ):
            raise ValueError(f"baseline tensor descriptor has the wrong shape or dtype: {name}")
        expected_nbytes = int(np.prod(expected_shape, dtype=np.int64)) * (1 if expected_dtype == "uint8" else 4)
        if (
            not _is_int(descriptor.get("offset_bytes"))
            or descriptor["offset_bytes"] != expected_offset
            or not _is_int(descriptor.get("nbytes"))
            or descriptor["nbytes"] != expected_nbytes
        ):
            raise ValueError(f"baseline tensor descriptor has an invalid offset or length: {name}")
        descriptors[name] = descriptor
        expected_offset += expected_nbytes
    if expected_offset != len(payload):
        raise ValueError("baseline tensor layout does not exactly cover its payload")

    hidden_desc = descriptors["encoder_hidden_states"]
    hidden_start = hidden_desc["offset_bytes"]
    hidden_end = hidden_start + hidden_desc["nbytes"]
    hidden_values = np.frombuffer(payload[hidden_start:hidden_end], dtype="<f4")
    if not bool(np.isfinite(hidden_values).all()):
        raise ValueError("baseline encoder_hidden_states must contain only finite values")
    hidden_bits = hidden_values.view("<u4")
    if bool((hidden_bits & np.uint32(0xFFFF)).any()):
        raise ValueError("baseline embeddings are not exact BF16 values expanded to float32")
    baseline_bf16 = (hidden_bits >> np.uint32(16)).astype("<u2")
    if _sha256(baseline_bf16.tobytes()) != PINNED_REFERENCE_EMBEDDINGS_SHA256:
        raise ValueError("baseline embeddings do not match the pinned official reference embeddings")
    masks: dict[str, np.ndarray] = {}
    for name in ("encoder_hidden_states_mask", "encoder_img_mask"):
        desc = descriptors[name]
        mask = np.frombuffer(
            payload[desc["offset_bytes"] : desc["offset_bytes"] + desc["nbytes"]],
            dtype=np.uint8,
        )
        if not bool(np.logical_or(mask == 0, mask == 1).all()):
            raise ValueError(f"baseline {name} values must be 0 or 1")
        masks[name] = mask
    if not bool(masks["encoder_hidden_states_mask"].all()):
        raise ValueError("pinned retained conditioning must mark all ten text tokens valid")
    if bool(masks["encoder_img_mask"].any()):
        raise ValueError("baseline text-to-image conditioning contains image placeholders")
    latent_desc = descriptors["initial_target_latents"]
    latent_values = np.frombuffer(
        payload[
            latent_desc["offset_bytes"] : latent_desc["offset_bytes"] + latent_desc["nbytes"]
        ],
        dtype="<f4",
    )
    if not bool(np.isfinite(latent_values).all()):
        raise ValueError("baseline initial_target_latents must contain only finite values")
    return manifest, payload, descriptors


def _validate_native_sidecar(
    manifest_path: Path,
    native_manifest: dict[str, Any],
    native_manifest_bytes: bytes,
    baseline: dict[str, Any],
) -> tuple[bytes, str]:
    schema_version = native_manifest.get("schema_version")
    if (
        native_manifest.get("schema") != NATIVE_SCHEMA
        or not _is_int(schema_version)
        or schema_version != NATIVE_SCHEMA_VERSION
    ):
        raise ValueError("native manifest schema/version is unsupported")
    for name in ("prompt", "model_revision"):
        expected = baseline["prompt"] if name == "prompt" else baseline["model"]["revision"]
        if native_manifest.get(name) != expected:
            raise ValueError(f"native manifest {name} does not match the baseline bundle")
    if native_manifest.get("fixture_payload_sha256") != PINNED_REFERENCE_FIXTURE_SHA256:
        raise ValueError("native manifest fixture_payload_sha256 does not match the pinned text-reference fixture")
    drop_idx = native_manifest.get("drop_idx")
    if not _is_int(drop_idx) or drop_idx != NATIVE_DROP_IDX:
        raise ValueError(f"native manifest drop_idx must be {NATIVE_DROP_IDX}")
    if native_manifest.get("shape") != NATIVE_SHAPE:
        raise ValueError(f"native manifest shape must be {NATIVE_SHAPE}")
    if native_manifest.get("dtype") != NATIVE_DTYPE:
        raise ValueError(f"native manifest dtype must be {NATIVE_DTYPE!r}")
    expected_nbytes = NATIVE_SHAPE[0] * NATIVE_SHAPE[1] * 2
    if not _is_int(native_manifest.get("nbytes")) or native_manifest["nbytes"] != expected_nbytes:
        raise ValueError(f"native manifest nbytes must be {expected_nbytes}")
    expected_sha = native_manifest.get("sha256")
    if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
        raise ValueError("native manifest has an invalid sidecar SHA-256")

    sidecar_path = _safe_sibling(manifest_path.parent, native_manifest.get("payload_file"), "native")
    try:
        sidecar = sidecar_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read native BF16 sidecar: {exc}") from exc
    if len(sidecar) != expected_nbytes:
        raise ValueError("native BF16 sidecar length does not match its manifest")
    if _sha256(sidecar) != expected_sha:
        raise ValueError("native BF16 sidecar SHA-256 does not match its manifest")

    # BF16 little-endian is the high 16 bits of each little-endian Float32 word.
    words = np.frombuffer(sidecar, dtype="<u2")
    expanded_bits = words.astype("<u4") << np.uint32(16)
    values = expanded_bits.view("<f4")
    if values.size != NATIVE_SHAPE[0] * NATIVE_SHAPE[1] or not bool(np.isfinite(values).all()):
        raise ValueError("native BF16 sidecar must expand to finite retained embeddings")
    return np.ascontiguousarray(values, dtype="<f4").tobytes(order="C"), _sha256(native_manifest_bytes)


def create_ab_bundle(
    baseline_bundle_dir: str | Path,
    native_manifest_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Clone a validated baseline bundle, replacing only its embedding bytes."""
    baseline_dir = Path(baseline_bundle_dir).expanduser().resolve(strict=True)
    native_path = Path(native_manifest_path).expanduser().resolve(strict=True)
    baseline_manifest, baseline_payload, descriptors = _validate_baseline(baseline_dir)
    native_manifest, native_manifest_bytes = _read_json(native_path, "native embedding manifest")
    native_hidden_bytes, native_manifest_sha256 = _validate_native_sidecar(
        native_path, native_manifest, native_manifest_bytes, baseline_manifest
    )

    output_requested = Path(output_dir).expanduser()
    try:
        output_parent = output_requested.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"output parent directory does not exist: {output_requested.parent}") from exc
    if not output_parent.is_dir():
        raise ValueError(f"output parent is not a directory: {output_parent}")
    target_dir = output_parent / output_requested.name
    if os.path.lexists(target_dir):
        raise FileExistsError(f"output directory already exists; refusing to overwrite: {target_dir}")
    if target_dir == baseline_dir or baseline_dir in target_dir.parents:
        raise ValueError("output directory must not be inside the baseline bundle")

    output_payload = bytearray(baseline_payload)
    hidden_desc = descriptors["encoder_hidden_states"]
    start = hidden_desc["offset_bytes"]
    end = start + hidden_desc["nbytes"]
    if len(native_hidden_bytes) != end - start:
        raise ValueError("native embedding byte length differs from the baseline tensor layout")
    output_payload[start:end] = native_hidden_bytes
    output_payload_bytes = bytes(output_payload)

    output_manifest = copy.deepcopy(baseline_manifest)
    output_manifest["payload_sha256"] = _sha256(output_payload_bytes)
    output_manifest["payload_nbytes"] = len(output_payload_bytes)
    output_manifest["conditioning_ab"] = {
        "kind": "native-retained-embeddings",
        "baseline_payload_sha256": baseline_manifest["payload_sha256"],
        "native_reference_fixture_payload_sha256": native_manifest["fixture_payload_sha256"],
        "official_reference_embeddings_sha256": PINNED_REFERENCE_EMBEDDINGS_SHA256,
        "native_manifest_sha256": native_manifest_sha256,
        "native_sidecar_sha256": native_manifest["sha256"],
        "model_revision": native_manifest["model_revision"],
        "drop_idx": NATIVE_DROP_IDX,
        "shape": list(NATIVE_SHAPE),
        "source_dtype": NATIVE_DTYPE,
        "target_dtype": "float32-le",
        "replaced_tensor": "encoder_hidden_states",
        "preserved_tensors": [
            "encoder_hidden_states_mask",
            "encoder_img_mask",
            "initial_target_latents",
        ],
    }

    # Create the destination directory exclusively, then publish payload first
    # and manifest last. Existing output paths and files are never overwritten.
    # If an I/O failure interrupts publication, retain the newly created partial
    # directory for inspection rather than deleting paths that could have been
    # concurrently replaced by another process.
    target_dir.mkdir()
    payload_path = target_dir / BUNDLE_PAYLOAD_NAME
    manifest_path = target_dir / BUNDLE_MANIFEST_NAME
    with payload_path.open("xb") as stream:
        stream.write(output_payload_bytes)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(output_manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", type=Path, required=True, help="validated baseline conditioning bundle directory")
    parser.add_argument("--native-manifest", type=Path, required=True, help="retained-embedding metadata JSON; payload_file resolves as its sibling")
    parser.add_argument("--output-dir", type=Path, required=True, help="new, non-existing directory for the A/B conditioning bundle")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = create_ab_bundle(args.baseline_bundle, args.native_manifest, args.output_dir)
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"qwen_image21_conditioning_ab: {exc}", file=sys.stderr)
        return 2
    manifest, _ = _read_json(result, "written A/B conditioning manifest")
    print(f"conditioning_ab_manifest={result}")
    print(f"conditioning_ab_payload_sha256={manifest['payload_sha256']}")
    print("validation=payload hash/layout, pinned official BF16 embedding SHA, declared fixture identity, native checksums and finite values; masks and initial latents preserved byte-for-byte")
    print("scope=conditioning bundle only; no DiT/VAE inference or image-quality claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
