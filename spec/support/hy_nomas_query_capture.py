#!/usr/bin/env python3
"""Capture authenticated first-frame post-RoPE Q/K from one pinned T0 block-0 pass.

The source tensors are produced from the admitted *native preblock* bundle at its
recorded initialization sigma. This does not claim clean/zero-noise inputs or
full upstream T0 preprocessing parity. No model code runs without ``--execute``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SOURCE_CACHE_PATH = REPO / "spec/support/hy_nomas_reference_cache.py"
SOURCE_BLOCK_HELPER_PATH = REPO / "spec/support/hy_nomas_reference.py"
QUERY_REFERENCE_PATH = REPO / "spec/support/hy_nomas_query_correspondence_reference.py"

DEFAULT_INPUT_MANIFEST = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-connected-t0-20261104.sydTYCdC/enabled-preblock.json"
)
DEFAULT_INPUT_MANIFEST_SHA256 = "d4988156b3d49e9458ecd1f5f3067609f314f1fef772c19a88f9fa94998b4151"
DEFAULT_INPUT_PAYLOAD_SHA256 = "44b527535f2e3835b7168055b49d387987a37bcd43ddea97879cacf9ee03a257"
ORIGINAL_REFERENCE_LATENT_SHA256 = "9e21e8b9e4cc49d355081023346309daa09c52ac6eedafa00a1de577ff17cf60"
TIMELINE_MANIFEST_SHA256 = "26be294251f4e25814b97e8438edffcd7c44c80325ce0d28d91d0fa7d41f5f43"

CAPTURE_KIND = "pinned_source_block0_t0_reference_cache_initialization"
CAPTURE_SCOPE = (
    "post-RoPE image Q/K from pinned upstream block 0 consuming authenticated native "
    "preblock inputs at the recorded T0 reference-cache initialization sigma; not a "
    "clean/zero-noise reference and not a complete upstream T0 handoff"
)
CAPTURE_BOUNDARY = (
    "img.q_rope and img.k_rope returned by the pinned block-0 apply_rotary_emb call; "
    "same tensors passed to the first cache append, before attention"
)
NOISE_SCOPE = (
    "This is the saved native preblock initialization state at the manifest's recorded "
    "sigma/timestep. The capture does not assert zero-noise latents or classify the "
    "selected frame as noise-free."
)
FRAME_ROWS = 144
IMAGE_SHAPE = (1, 4320, 16, 128)
SELECTED_SHAPE = (144, 16, 128)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class QueryCaptureError(ValueError):
    """A source capture, lineage, or output failed closed validation."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QueryCaptureError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, limit: int = 4 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise QueryCaptureError(f"expected a regular non-symlink JSON file: {path}")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise QueryCaptureError(f"JSON file exceeds size limit: {path}")
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QueryCaptureError(f"invalid JSON: {path}") from error
    if type(doc) is not dict:
        raise QueryCaptureError(f"expected a JSON object: {path}")
    return doc, raw


def _canonical_input(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise QueryCaptureError(f"{label} must be an absolute regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise QueryCaptureError(f"cannot resolve {label}") from error


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise QueryCaptureError(f"{label} must be an absolute non-symlink directory")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise QueryCaptureError(f"cannot resolve {label}") from error


def _fresh_output(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise QueryCaptureError(f"{label} must be a fresh absolute path")
    if not path.parent.is_dir():
        raise QueryCaptureError(f"{label} parent directory must exist")
    try:
        return path.parent.resolve(strict=True) / path.name
    except (OSError, RuntimeError) as error:
        raise QueryCaptureError(f"cannot resolve {label} parent") from error


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise QueryCaptureError(f"cannot load source helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_original_bundle(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
    cache_module: Any,
) -> tuple[dict[str, Any], dict[str, Any], Path, bytes]:
    manifest = _canonical_input(manifest_path, "original preblock manifest")
    document, raw_manifest = cache_module._read_json(manifest)
    manifest_sha256 = _sha256_bytes(raw_manifest)
    if manifest_sha256 != expected_manifest_sha256:
        raise QueryCaptureError("original preblock manifest digest differs from pinned input")
    if set(document) != cache_module.MANIFEST_KEYS:
        raise QueryCaptureError("original preblock manifest keyset differs from source contract")
    if (document["schema"] != cache_module.SCHEMA
            or document["schema_version"] != cache_module.SCHEMA_VERSION
            or document["upstream_revision"] != cache_module.UPSTREAM_REVISION
            or document["scope"] != cache_module.SCOPE
            or not cache_module._matches_exact_json(document["preprocessing"], cache_module.PREPROCESSING)
            or not cache_module._matches_exact_json(document["cache_writer"], cache_module.CACHE_WRITER)):
        raise QueryCaptureError("original preblock schema, scope, preprocessing, or T0 writer mismatch")
    if (document["reference_latent_artifact_sha256"] != ORIGINAL_REFERENCE_LATENT_SHA256
            or document["timeline_manifest_sha256"] != TIMELINE_MANIFEST_SHA256):
        raise QueryCaptureError("original preblock latent/timeline pin differs from the admitted source input")
    payload_name = document["payload"]
    expected_name = manifest.with_suffix(".tensors.bin").name
    if payload_name != expected_name or Path(payload_name).name != payload_name:
        raise QueryCaptureError("original preblock payload is not its sibling .tensors.bin")
    payload = _canonical_input(manifest.parent / payload_name, "original preblock payload")
    if (payload.stat().st_size != document["payload_bytes"]
            or document["payload_sha256"] != expected_payload_sha256
            or _sha256_file(payload) != expected_payload_sha256):
        raise QueryCaptureError("original preblock payload size or digest differs from pinned input")
    expected = cache_module._expected_tensors(document["valid_text_tokens"])
    if tuple(document["tensors"]) != tuple(item[0] for item in expected):
        raise QueryCaptureError("original preblock tensor inventory/order mismatch")
    offset = 0
    latent_bytes = None
    for name, dtype, shape in expected:
        record = document["tensors"][name]
        extent = cache_module._expected_nbytes(dtype, shape)
        if (type(record) is not dict or set(record) != cache_module.TENSOR_RECORD_KEYS
                or record["dtype"] != dtype or record["shape"] != list(shape)
                or type(record["offset"]) is not int or record["offset"] != offset
                or type(record["nbytes"]) is not int or record["nbytes"] != extent
                or not cache_module._is_digest(record["sha256"])
                or cache_module._hash_range(payload, offset, extent) != record["sha256"]):
            raise QueryCaptureError(f"original preblock range failed authentication: {name}")
        if name == "reference.latent_bf16":
            with payload.open("rb") as stream:
                stream.seek(offset)
                latent_bytes = stream.read(extent)
            if (latent_bytes is None or len(latent_bytes) != extent
                    or _sha256_bytes(latent_bytes) != record["sha256"]):
                raise QueryCaptureError("embedded reference latent range is truncated or changed")
        offset += extent
    if offset != payload.stat().st_size or latent_bytes is None:
        raise QueryCaptureError("original preblock ranges do not cover the payload")
    return document, {
        "manifest_path": str(manifest), "manifest_sha256": manifest_sha256,
        "payload_path": str(payload), "payload_sha256": expected_payload_sha256,
        "reference_latent_tensor_sha256": document["tensors"]["reference.latent_bf16"]["sha256"],
    }, payload, latent_bytes


def _safetensors_bytes(latent_bytes: bytes) -> bytes:
    if len(latent_bytes) != math.prod((1, 32, 30, 9, 16)) * 2:
        raise QueryCaptureError("embedded reference latent has an unexpected byte extent")
    header = {
        "latent": {
            "dtype": "BF16", "shape": [1, 32, 30, 9, 16],
            "data_offsets": [0, len(latent_bytes)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return len(encoded).to_bytes(8, "little") + encoded + latent_bytes


def prepare_derived_bundle(
    original_manifest_path: str | Path,
    derived_directory: str | Path,
    *,
    expected_manifest_sha256: str = DEFAULT_INPUT_MANIFEST_SHA256,
    expected_payload_sha256: str = DEFAULT_INPUT_PAYLOAD_SHA256,
) -> dict[str, Any]:
    """Make a new strict-validator bundle using the exact embedded BF16 latent bytes.

    The original bundle and its missing external latent artifact are never edited
    or represented as recovered. The new safetensors file is explicitly derived
    from the authenticated embedded range; the copied bundle differs only in its
    sibling payload basename and the new artifact digest.
    """
    cache = _load_module(SOURCE_CACHE_PATH, "hy_nomas_reference_cache_for_query_capture")
    source_dir = Path(derived_directory).expanduser()
    if not source_dir.is_absolute() or source_dir.exists() or source_dir.is_symlink():
        raise QueryCaptureError("derived bundle directory must be a fresh absolute path")
    if not source_dir.parent.is_dir():
        raise QueryCaptureError("derived bundle parent directory must exist")
    source_dir = source_dir.parent.resolve(strict=True) / source_dir.name
    source_dir.mkdir()
    original_doc, original_binding, original_payload, latent_bytes = _validate_original_bundle(
        original_manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_payload_sha256=expected_payload_sha256,
        cache_module=cache,
    )

    artifact_path = source_dir / "reference-latent-reconstructed.safetensors"
    derived_manifest = source_dir / "derived-preblock.json"
    derived_payload = source_dir / "derived-preblock.tensors.bin"
    artifact_raw = _safetensors_bytes(latent_bytes)
    artifact_sha256 = _sha256_bytes(artifact_raw)
    if artifact_sha256 != ORIGINAL_REFERENCE_LATENT_SHA256:
        raise QueryCaptureError(
            "reconstructed Safetensors envelope does not reproduce the original pinned artifact digest"
        )
    with artifact_path.open("xb") as stream:
        stream.write(artifact_raw)
    shutil.copyfile(original_payload, derived_payload)
    derived_doc = json.loads(json.dumps(original_doc))
    derived_doc["payload"] = derived_payload.name
    derived_doc["reference_latent_artifact_sha256"] = artifact_sha256
    with derived_manifest.open("x", encoding="utf-8") as stream:
        json.dump(derived_doc, stream, indent=2)
        stream.write("\n")

    manifest_sha256 = _sha256_file(derived_manifest)
    payload_sha256 = _sha256_file(derived_payload)
    artifact_file_sha256 = _sha256_file(artifact_path)
    if (payload_sha256 != expected_payload_sha256 or artifact_file_sha256 != artifact_sha256
            or payload_sha256 != original_binding["payload_sha256"]):
        raise QueryCaptureError("derived bundle bytes differ from original authenticated payload/artifact")
    derived_validated = cache.validate_reference_cache_manifest(
        derived_manifest,
        reference_latent_path=artifact_path,
        expected_reference_latent_artifact_sha256=artifact_sha256,
        expected_timeline_manifest_sha256=TIMELINE_MANIFEST_SHA256,
        expected_manifest_sha256=manifest_sha256,
    )
    if _sha256_file(original_payload) != expected_payload_sha256:
        raise QueryCaptureError("original source payload changed while deriving the bundle")
    return {
        "original_bundle": original_binding,
        "derived_bundle": {
            "manifest_path": str(derived_manifest), "manifest_sha256": manifest_sha256,
            "payload_path": str(derived_payload), "payload_sha256": payload_sha256,
            "payload_bytes": derived_payload.stat().st_size,
        },
        "reconstructed_reference_latent": {
            "path": str(artifact_path), "sha256": artifact_sha256,
            "source_payload_tensor": "reference.latent_bf16",
            "source_tensor_sha256": original_binding["reference_latent_tensor_sha256"],
            "method": "new safetensors envelope around exact authenticated embedded BF16 bytes",
            "original_external_artifact_sha256": ORIGINAL_REFERENCE_LATENT_SHA256,
            "original_external_artifact_available": False,
        },
        "strict_validator": {
            "schema": derived_validated["schema"], "manifest_sha256": derived_validated["manifest_sha256"],
            "payload_sha256": derived_validated["payload_sha256"],
            "reference_latent_tensor_sha256": derived_validated["reference_latent_tensor_sha256"],
            "scope": derived_validated["scope"],
        },
        "t0_state": original_doc["cache_writer"],
        "preprocessing": original_doc["preprocessing"],
    }


def _safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    artifact = _canonical_input(path, "source tensor artifact")
    size = artifact.stat().st_size
    with artifact.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise QueryCaptureError("source safetensors header is truncated")
        header_size = int.from_bytes(prefix, "little")
        if header_size < 2 or header_size > 16 * 1024 * 1024 or 8 + header_size > size:
            raise QueryCaptureError("source safetensors header extent is invalid")
        raw = stream.read(header_size)
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QueryCaptureError("source safetensors header is invalid") from error
    if type(header) is not dict:
        raise QueryCaptureError("source safetensors header is not an object")
    return header, 8 + header_size, size


def _read_safetensors_range(path: Path, name: str, *, expected_shape: tuple[int, ...],
                            expected_dtype: str, expected_sha256: str) -> tuple[bytes, dict[str, Any]]:
    header, data_start, total_size = _safetensors_header(path)
    record = header.get(name)
    offsets = record.get("data_offsets") if type(record) is dict else None
    if (type(record) is not dict or record.get("dtype") != expected_dtype
            or record.get("shape") != list(expected_shape)
            or type(offsets) is not list or len(offsets) != 2
            or any(type(value) is not int for value in offsets)):
        raise QueryCaptureError(f"source tensor record is invalid: {name}")
    start, stop = offsets
    expected_bytes = math.prod(expected_shape) * 2
    if start < 0 or stop - start != expected_bytes or data_start + stop > total_size:
        raise QueryCaptureError(f"source tensor range extent is invalid: {name}")
    with path.open("rb") as stream:
        stream.seek(data_start + start)
        raw = stream.read(expected_bytes)
    if len(raw) != expected_bytes or _sha256_bytes(raw) != expected_sha256:
        raise QueryCaptureError(f"source tensor payload digest mismatch: {name}")
    return raw, record


def _expected_source_tensor_shapes(valid_text_tokens: Any) -> dict[str, tuple[int, ...]]:
    if type(valid_text_tokens) is not int or not 1 <= valid_text_tokens <= 1280:
        raise QueryCaptureError("source receipt has an invalid valid_text_tokens extent")
    image = (1, 4320, 16, 128)
    text = (1, valid_text_tokens, 16, 128)
    return {
        "img.q_projection": image,
        "img.k_projection": image,
        "img.v_projection": image,
        "img.q_qknorm": image,
        "img.k_qknorm": image,
        "img.q_rope": image,
        "img.k_rope": image,
        "txt.q_projection": text,
        "txt.k_projection": text,
        "txt.v_projection": text,
        "txt.q_qknorm": text,
        "txt.k_qknorm": text,
        "cache.k_bf16": (1, 4320 + valid_text_tokens, 16, 128),
        "cache.v_bf16": (1, 4320 + valid_text_tokens, 16, 128),
    }


def _validate_source_tensor_inventory(
    records: Any,
    header: Any,
    *,
    tensor_count: Any,
    expected_shapes: dict[str, tuple[int, ...]],
    data_start: int,
    total_size: int,
) -> None:
    """Validate receipt and Safetensors inventory without relying on JSON key order."""
    expected_names = set(expected_shapes)
    if (type(records) is not dict or set(records) != expected_names
            or len(records) != len(expected_names)
            or type(tensor_count) is not int or tensor_count != len(expected_names)):
        raise QueryCaptureError("source tensor inventory differs from the pinned block-0 oracle")
    if (type(header) is not dict or set(header) != expected_names | {"__metadata__"}
            or type(data_start) is not int or type(total_size) is not int
            or data_start < 0 or total_size < data_start):
        raise QueryCaptureError("source Safetensors inventory differs from the pinned block-0 oracle")

    extents: list[tuple[int, int]] = []
    for name, shape in expected_shapes.items():
        record = records[name]
        expected_nbytes = math.prod(shape) * 2
        if (type(record) is not dict or set(record) != {"dtype", "shape", "nbytes", "sha256"}
                or record["dtype"] != "BF16" or record["shape"] != list(shape)
                or any(type(value) is not int for value in record["shape"])
                or type(record["nbytes"]) is not int or record["nbytes"] != expected_nbytes
                or type(record["sha256"]) is not str or not _SHA256.fullmatch(record["sha256"])):
            raise QueryCaptureError(f"source tensor receipt record is malformed: {name}")

        header_record = header[name]
        offsets = header_record.get("data_offsets") if type(header_record) is dict else None
        if (type(header_record) is not dict
                or set(header_record) != {"dtype", "shape", "data_offsets"}
                or header_record["dtype"] != record["dtype"]
                or header_record["shape"] != record["shape"]
                or type(offsets) is not list or len(offsets) != 2
                or any(type(value) is not int for value in offsets)):
            raise QueryCaptureError(f"source Safetensors header record differs from receipt: {name}")
        start, stop = offsets
        if (start < 0 or stop - start != expected_nbytes
                or stop > total_size - data_start):
            raise QueryCaptureError(f"source Safetensors tensor extent is invalid: {name}")
        extents.append((start, stop))

    cursor = 0
    for start, stop in sorted(extents):
        if start != cursor:
            raise QueryCaptureError("source Safetensors tensor ranges overlap or leave gaps")
        cursor = stop
    if cursor != total_size - data_start:
        raise QueryCaptureError("source Safetensors tensor ranges do not cover the data section")


def _expand_bf16(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2")
    expanded = (words.astype("<u4") << 16).view("<f4").reshape(shape).copy()
    if not np.all(np.isfinite(expanded)):
        raise QueryCaptureError("source BF16 Q/K contains non-finite values")
    return np.ascontiguousarray(expanded, dtype="<f4")


def _extract_source_qk(receipt_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = _load_module(SOURCE_CACHE_PATH, "hy_nomas_reference_cache_for_query_capture")
    receipt_path = _canonical_input(receipt_path, "source receipt")
    receipt, _ = _read_json(receipt_path)
    artifact_binding = receipt.get("tensor_artifact")
    weight_binding = receipt.get("weight_source")
    input_binding = receipt.get("input_bundle")
    if (receipt.get("schema") != cache.ORACLE_SCHEMA
            or receipt.get("operation") != "upstream_block0_first_chunk_kv_cache_append"
            or receipt.get("upstream_revision") != cache.UPSTREAM_REVISION
            or type(artifact_binding) is not dict or type(weight_binding) is not dict
            or type(input_binding) is not dict):
        raise QueryCaptureError("source receipt schema, operation, or pinned revision mismatch")
    if receipt.get("scope") != (
            "Pinned upstream MMDoubleStreamBlock.forward_with_cache on the exact native "
            "preblock input tensors; captures block-0 K/V at first-chunk cache append. "
            "Does not claim parity for upstream timeline, noise, patch, or text-refiner preprocessing."):
        raise QueryCaptureError("source receipt does not state the expected bounded native-preblock scope")
    if (receipt.get("precision") != {
            "mode": "bf16", "block_parameters": "BF16",
            "block_inputs": "BF16 cast from authenticated F32 preblock bundle",
            "autocast": "MPS BF16", "cache_storage": "BF16"}
            or receipt.get("cache_append") != {
                "cache_valid_len": 0, "update_cache": True, "cache_txt": True,
                "attention_executed": False,
                "cache_tensor_order": "image tokens followed by valid text prefix",
                "cache_tokens": 4320 + receipt.get("input_bundle", {}).get("valid_text_tokens", -999)}):
        raise QueryCaptureError("source receipt precision or pre-attention cache boundary mismatch")
    if (weight_binding.get("sha256") != cache.BLOCK_WEIGHTS_SHA256
            or weight_binding.get("parameter_dtype") != "BF16"
            or weight_binding.get("source_dtype") != "BF16"
            or weight_binding.get("tensor_prefix") != "double_blocks.0."):
        raise QueryCaptureError("source receipt weight identity or BF16 interpretation mismatch")
    weights_path = _canonical_input(weight_binding.get("path", ""), "source block-0 weights")
    if _sha256_file(weights_path) != cache.BLOCK_WEIGHTS_SHA256:
        raise QueryCaptureError("source block-0 weight file differs from the pinned digest")
    artifact_value = artifact_binding.get("path", "")
    if type(artifact_value) is not str:
        raise QueryCaptureError("source tensor artifact path is malformed")
    artifact_candidate = Path(artifact_value)
    if artifact_candidate.is_absolute():
        artifact_path = _canonical_input(artifact_candidate, "source tensor artifact")
    elif (artifact_candidate.name == artifact_value and artifact_value not in (".", "..")):
        # The pinned source writer stores a sibling basename in the on-disk
        # receipt, then returns the canonical absolute path only in memory.
        artifact_path = _canonical_input(receipt_path.parent / artifact_value, "source tensor artifact")
    else:
        raise QueryCaptureError("source receipt artifact path must be absolute or a sibling basename")
    artifact_sha256 = _sha256_file(artifact_path)
    if artifact_sha256 != artifact_binding.get("sha256"):
        raise QueryCaptureError("source tensor artifact digest differs from its receipt")
    records = artifact_binding.get("tensors")
    expected_shapes = _expected_source_tensor_shapes(input_binding.get("valid_text_tokens"))
    if tuple(expected_shapes) != cache.ORACLE_KEYS:
        raise QueryCaptureError("capture helper tensor shapes differ from the pinned source oracle")
    header, data_start, total_size = _safetensors_header(artifact_path)
    _validate_source_tensor_inventory(
        records, header,
        tensor_count=artifact_binding.get("tensor_count"),
        expected_shapes=expected_shapes,
        data_start=data_start,
        total_size=total_size,
    )
    metadata = header.get("__metadata__")
    if (type(metadata) is not dict
            or set(metadata) != {"format", "upstream_revision", "weight_sha256", "cache_dtype"}
            or metadata.get("format") != cache.ORACLE_SCHEMA
            or metadata.get("upstream_revision") != cache.UPSTREAM_REVISION
            or metadata.get("weight_sha256") != cache.BLOCK_WEIGHTS_SHA256
            or metadata.get("cache_dtype") != "BF16"):
        raise QueryCaptureError("source safetensors metadata differs from pinned block-0 receipt")
    q_record = records.get("img.q_rope")
    k_record = records.get("img.k_rope")
    if (type(q_record) is not dict or type(k_record) is not dict
            or q_record.get("dtype") != "BF16" or k_record.get("dtype") != "BF16"
            or q_record.get("shape") != list(IMAGE_SHAPE)
            or k_record.get("shape") != list(IMAGE_SHAPE)
            or q_record.get("nbytes") != math.prod(IMAGE_SHAPE) * 2
            or k_record.get("nbytes") != math.prod(IMAGE_SHAPE) * 2):
        raise QueryCaptureError("source Q/K capture dtype, shape, or extent mismatch")
    q_raw, _ = _read_safetensors_range(
        artifact_path, "img.q_rope", expected_shape=IMAGE_SHAPE,
        expected_dtype="BF16", expected_sha256=q_record["sha256"],
    )
    k_raw, _ = _read_safetensors_range(
        artifact_path, "img.k_rope", expected_shape=IMAGE_SHAPE,
        expected_dtype="BF16", expected_sha256=k_record["sha256"],
    )
    row_bytes = math.prod(SELECTED_SHAPE) * 2
    selected_q_raw, selected_k_raw = q_raw[:row_bytes], k_raw[:row_bytes]
    query = _expand_bf16(selected_q_raw, SELECTED_SHAPE)
    key = _expand_bf16(selected_k_raw, SELECTED_SHAPE)
    return query, key, {
        "receipt_path": str(receipt_path), "receipt_sha256": _sha256_file(receipt_path),
        "artifact_path": str(artifact_path), "artifact_sha256": artifact_sha256,
        "artifact_tensor_count": artifact_binding["tensor_count"],
        "query_record": q_record, "key_record": k_record,
        "query_selected_bf16_sha256": _sha256_bytes(selected_q_raw),
        "key_selected_bf16_sha256": _sha256_bytes(selected_k_raw),
        "weights_path": str(weights_path), "weights_sha256": cache.BLOCK_WEIGHTS_SHA256,
        "source_revision": receipt["upstream_revision"],
        "source_root": receipt.get("upstream_source_root"),
        "source_file_sha256": receipt.get("upstream_source_file_sha256"),
        "source_dependency_sha256": receipt.get("upstream_dependency_sha256"),
        "inference_dtype_source_sha256": receipt.get("inference_dtype_source_sha256"),
        "input_bundle": input_binding,
        "precision": receipt["precision"], "scope": receipt["scope"],
        "cache_append": receipt["cache_append"],
    }


def _verify_bundle_lineage(provenance: dict[str, Any]) -> dict[str, Any]:
    cache = _load_module(SOURCE_CACHE_PATH, "hy_nomas_reference_cache_for_query_capture")
    lineage = provenance.get("input_lineage")
    if type(lineage) is not dict:
        raise QueryCaptureError("published input lacks authenticated preblock lineage")
    original = lineage.get("original_bundle")
    derived = lineage.get("derived_bundle")
    reconstructed = lineage.get("reconstructed_reference_latent")
    if any(type(value) is not dict for value in (original, derived, reconstructed)):
        raise QueryCaptureError("published input preblock lineage record is incomplete")
    original_manifest = _canonical_input(original.get("manifest_path", ""), "original preblock manifest")
    original_doc, original_raw = cache._read_json(original_manifest)
    original_sha = _sha256_bytes(original_raw)
    if (original_sha != DEFAULT_INPUT_MANIFEST_SHA256
            or original.get("manifest_sha256") != original_sha
            or original_doc.get("reference_latent_artifact_sha256") != ORIGINAL_REFERENCE_LATENT_SHA256
            or original_doc.get("timeline_manifest_sha256") != TIMELINE_MANIFEST_SHA256):
        raise QueryCaptureError("original authenticated T0 input identity mismatch")
    original_payload = _canonical_input(original_manifest.parent / original_doc["payload"], "original preblock payload")
    if (original_payload != _canonical_input(original.get("payload_path", ""), "bound original preblock payload")
            or _sha256_file(original_payload) != DEFAULT_INPUT_PAYLOAD_SHA256
            or original.get("payload_sha256") != DEFAULT_INPUT_PAYLOAD_SHA256):
        raise QueryCaptureError("original preblock payload lineage mismatch")

    derived_manifest = _canonical_input(derived.get("manifest_path", ""), "derived preblock manifest")
    derived_doc, derived_raw = cache._read_json(derived_manifest)
    derived_sha = _sha256_bytes(derived_raw)
    if derived.get("manifest_sha256") != derived_sha:
        raise QueryCaptureError("derived preblock manifest digest mismatch")
    derived_payload = _canonical_input(derived_manifest.parent / derived_doc.get("payload", ""),
                                       "derived preblock payload")
    if (derived_payload != _canonical_input(derived.get("payload_path", ""), "bound derived preblock payload")
            or _sha256_file(derived_payload) != DEFAULT_INPUT_PAYLOAD_SHA256
            or derived.get("payload_sha256") != DEFAULT_INPUT_PAYLOAD_SHA256):
        raise QueryCaptureError("derived preblock payload lineage mismatch")
    expected_derived = json.loads(json.dumps(original_doc))
    expected_derived["payload"] = derived_payload.name
    if (derived_doc.get("payload") != derived_payload.name
            or not isinstance(derived_doc.get("reference_latent_artifact_sha256"), str)):
        raise QueryCaptureError("derived preblock manifest has malformed artifact or payload binding")
    derived_artifact_sha = derived_doc["reference_latent_artifact_sha256"]
    if not _SHA256.fullmatch(derived_artifact_sha):
        raise QueryCaptureError("derived reference latent artifact digest is malformed")
    if derived_artifact_sha != ORIGINAL_REFERENCE_LATENT_SHA256:
        raise QueryCaptureError("derived latent artifact does not reproduce the original pinned digest")
    expected_derived["reference_latent_artifact_sha256"] = derived_artifact_sha
    if derived_doc != expected_derived:
        raise QueryCaptureError("derived preblock manifest changes fields beyond payload/artifact identity")
    artifact_path = _canonical_input(reconstructed.get("path", ""), "reconstructed latent artifact")
    artifact_sha = _sha256_file(artifact_path)
    if (artifact_sha != derived_artifact_sha or reconstructed.get("sha256") != artifact_sha
            or reconstructed.get("method") != "new safetensors envelope around exact authenticated embedded BF16 bytes"
            or reconstructed.get("original_external_artifact_sha256") != ORIGINAL_REFERENCE_LATENT_SHA256
            or reconstructed.get("original_external_artifact_available") is not False):
        raise QueryCaptureError("derived reference latent artifact provenance mismatch")
    header, data_start, total_size = _safetensors_header(artifact_path)
    latent = header.get("latent")
    offsets = latent.get("data_offsets") if type(latent) is dict else None
    if (type(latent) is not dict or latent.get("dtype") != "BF16"
            or latent.get("shape") != [1, 32, 30, 9, 16]
            or type(offsets) is not list or len(offsets) != 2
            or offsets[0] != 0 or offsets[1] - offsets[0] != 276480
            or data_start + offsets[1] != total_size):
        raise QueryCaptureError("reconstructed reference latent safetensors tensor is invalid")
    with artifact_path.open("rb") as stream:
        stream.seek(data_start)
        latent_raw = stream.read(276480)
    source_record = original_doc["tensors"]["reference.latent_bf16"]
    with original_payload.open("rb") as stream:
        stream.seek(source_record["offset"])
        original_latent = stream.read(source_record["nbytes"])
    if (len(latent_raw) != 276480 or latent_raw != original_latent
            or _sha256_bytes(latent_raw) != source_record["sha256"]
            or reconstructed.get("source_tensor_sha256") != source_record["sha256"]
            or reconstructed.get("source_payload_tensor") != "reference.latent_bf16"):
        raise QueryCaptureError("reconstructed artifact does not contain exact original embedded BF16 bytes")
    validated = cache.validate_reference_cache_manifest(
        derived_manifest,
        reference_latent_path=artifact_path,
        expected_reference_latent_artifact_sha256=artifact_sha,
        expected_timeline_manifest_sha256=TIMELINE_MANIFEST_SHA256,
        expected_manifest_sha256=derived_sha,
    )
    if (validated["payload_sha256"] != DEFAULT_INPUT_PAYLOAD_SHA256
            or validated["reference_latent_tensor_sha256"] != source_record["sha256"]
            or not cache._matches_exact_json(original_doc["cache_writer"], cache.CACHE_WRITER)
            or not cache._matches_exact_json(original_doc["preprocessing"], cache.PREPROCESSING)):
        raise QueryCaptureError("strict source validator or exact T0 writer contract failed")
    return {
        "original_manifest_sha256": original_sha,
        "original_payload_sha256": DEFAULT_INPUT_PAYLOAD_SHA256,
        "derived_manifest_sha256": derived_sha,
        "derived_payload_sha256": DEFAULT_INPUT_PAYLOAD_SHA256,
        "derived_latent_artifact_sha256": artifact_sha,
        "embedded_latent_sha256": source_record["sha256"],
        "cache_writer": original_doc["cache_writer"],
        "preprocessing": original_doc["preprocessing"],
        "scope": original_doc["scope"],
    }


def verify_published_input(manifest: str | Path, trusted_sha256: str) -> dict[str, Any]:
    """Authenticate the input report back to source Q/K bytes and exact T0 lineage.

    This is stronger than checking the free-form provenance labels: the published
    arrays must equal rows 0:144 of the authenticated source receipt's post-RoPE
    ``img.q_rope``/``img.k_rope`` BF16 tensors.
    """
    query_reference = _load_module(QUERY_REFERENCE_PATH, "hy_nomas_query_correspondence_for_capture")
    report = _canonical_input(manifest, "query-capture input manifest")
    admitted = query_reference.load_input(report, trusted_sha256)
    document = admitted["document"]
    provenance = document.get("provenance")
    if (type(provenance) is not dict or provenance.get("kind") != CAPTURE_KIND
            or provenance.get("scope") != CAPTURE_SCOPE
            or provenance.get("noise_scope") != NOISE_SCOPE
            or provenance.get("boundary") != CAPTURE_BOUNDARY):
        raise QueryCaptureError("query input is not the admitted source T0 block-0 boundary")
    bundle_facts = _verify_bundle_lineage(provenance)
    source = provenance.get("source_oracle")
    capture = provenance.get("selected_capture")
    if type(source) is not dict or type(capture) is not dict:
        raise QueryCaptureError("query input lacks source oracle or selected Q/K binding")
    receipt_path = _canonical_input(source.get("receipt_path", ""), "source block receipt")
    if _sha256_file(receipt_path) != source.get("receipt_sha256"):
        raise QueryCaptureError("source block receipt SHA-256 mismatch")
    query, key, source_facts = _extract_source_qk(receipt_path)
    if (source_facts["artifact_path"] != source.get("artifact_path")
            or source_facts["artifact_sha256"] != source.get("artifact_sha256")
            or source_facts["weights_sha256"] != source.get("weights_sha256")
            or source_facts["weights_path"] != source.get("weights_path")
            or source_facts["source_revision"] != source.get("upstream_revision")
            or source_facts["source_file_sha256"] != source.get("upstream_source_file_sha256")
            or source_facts["source_dependency_sha256"] != source.get("upstream_dependency_sha256")
            or source_facts["inference_dtype_source_sha256"] != source.get("inference_dtype_source_sha256")
            or source_facts["scope"] != source.get("scope")):
        raise QueryCaptureError("source receipt/artifact identities differ from input provenance")
    if (source.get("runner_path") != str(SOURCE_CACHE_PATH.resolve())
            or source.get("runner_sha256") != _sha256_file(SOURCE_CACHE_PATH)
            or source.get("source_helper_path") != str(SOURCE_BLOCK_HELPER_PATH.resolve())
            or source.get("source_helper_sha256") != _sha256_file(SOURCE_BLOCK_HELPER_PATH)
            or source.get("publisher_path") != str(QUERY_REFERENCE_PATH.resolve())
            or source.get("publisher_sha256") != _sha256_file(QUERY_REFERENCE_PATH)
            or source.get("capture_helper_path") != str(Path(__file__).resolve())
            or source.get("capture_helper_sha256") != _sha256_file(Path(__file__).resolve())):
        raise QueryCaptureError("source runner, publisher, or capture-helper bytes differ from recorded scripts")
    cache = _load_module(SOURCE_CACHE_PATH, "hy_nomas_reference_cache_for_query_capture")
    upstream_root = _canonical_directory(source.get("upstream_root", ""), "pinned upstream source root")
    try:
        _upstream, upstream_hashes, upstream_revision = cache._verify_upstream_root(upstream_root, cache._load_support_oracle())
    except Exception as error:
        raise QueryCaptureError("pinned upstream source tree no longer authenticates") from error
    if (upstream_revision != source_facts["source_revision"]
            or upstream_hashes != source_facts["source_dependency_sha256"] | source_facts["inference_dtype_source_sha256"]):
        raise QueryCaptureError("upstream source revision or file hashes differ from source receipt")
    if (source_facts["input_bundle"].get("manifest_sha256")
            != provenance["input_lineage"]["derived_bundle"]["manifest_sha256"]
            or source_facts["input_bundle"].get("payload_sha256")
            != provenance["input_lineage"]["derived_bundle"]["payload_sha256"]
            or source_facts["input_bundle"].get("timeline_manifest_sha256") != TIMELINE_MANIFEST_SHA256
            or source_facts["input_bundle"].get("reference_latent_artifact_sha256")
            != bundle_facts["derived_latent_artifact_sha256"]):
        raise QueryCaptureError("source receipt is not bound to the admitted derived T0 preblock")
    if capture != {
            "query_tensor": "img.q_rope", "key_tensor": "img.k_rope",
            "source_dtype": "BF16", "source_shape": list(IMAGE_SHAPE),
            "source_row_range": [0, FRAME_ROWS], "selected_shape": list(SELECTED_SHAPE),
            "query_source_range_sha256": source_facts["query_selected_bf16_sha256"],
            "key_source_range_sha256": source_facts["key_selected_bf16_sha256"],
            "query_range_sha256": _sha256_bytes(query.astype("<f4", copy=False).tobytes()),
            "key_range_sha256": _sha256_bytes(key.astype("<f4", copy=False).tobytes()),
            "frame_index": 0, "frame_tokens": FRAME_ROWS,
    }:
        raise QueryCaptureError("selected Q/K source tensor or row boundary metadata mismatch")
    if (admitted["query"].tobytes() != query.tobytes()
            or admitted["key"].tobytes() != key.tobytes()):
        raise QueryCaptureError("published Q/K arrays differ from authenticated source tensor ranges")
    return {
        "passed": True,
        "input_manifest_path": str(report), "input_manifest_sha256": trusted_sha256,
        "query_range_sha256": capture["query_range_sha256"],
        "key_range_sha256": capture["key_range_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "source_artifact_sha256": source["artifact_sha256"],
        "source_tensor_names": ["img.q_rope", "img.k_rope"],
        "source_rows": [0, FRAME_ROWS],
        "boundary": CAPTURE_BOUNDARY,
        "scope": CAPTURE_SCOPE,
        "noise_scope": NOISE_SCOPE,
        "bundle_lineage": bundle_facts,
        "upstream_revision": source_facts["source_revision"],
        "weights_sha256": source_facts["weights_sha256"],
    }


def _make_capture_provenance(bundle: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": CAPTURE_KIND,
        "scope": CAPTURE_SCOPE,
        "noise_scope": NOISE_SCOPE,
        "boundary": CAPTURE_BOUNDARY,
        "input_lineage": bundle,
        "source_oracle": {
            "receipt_path": source["receipt_path"], "receipt_sha256": source["receipt_sha256"],
            "artifact_path": source["artifact_path"], "artifact_sha256": source["artifact_sha256"],
            "weights_path": source["weights_path"], "weights_sha256": source["weights_sha256"],
            "upstream_revision": source["source_revision"], "upstream_root": source["source_root"],
            "upstream_source_file_sha256": source["source_file_sha256"],
            "upstream_dependency_sha256": source["source_dependency_sha256"],
            "inference_dtype_source_sha256": source["inference_dtype_source_sha256"],
            "precision": source["precision"], "scope": source["scope"],
            "cache_append": source["cache_append"],
            "runner_path": str(SOURCE_CACHE_PATH.resolve()),
            "runner_sha256": _sha256_file(SOURCE_CACHE_PATH),
            "publisher_path": str(QUERY_REFERENCE_PATH.resolve()),
            "publisher_sha256": _sha256_file(QUERY_REFERENCE_PATH),
            "capture_helper_path": str(Path(__file__).resolve()),
            "capture_helper_sha256": _sha256_file(Path(__file__).resolve()),
            "source_helper_path": str(SOURCE_BLOCK_HELPER_PATH.resolve()),
            "source_helper_sha256": _sha256_file(SOURCE_BLOCK_HELPER_PATH),
        },
        "selected_capture": {
            "query_tensor": "img.q_rope", "key_tensor": "img.k_rope",
            "source_dtype": "BF16", "source_shape": list(IMAGE_SHAPE),
            "source_row_range": [0, FRAME_ROWS], "selected_shape": list(SELECTED_SHAPE),
            "query_source_range_sha256": source["query_selected_bf16_sha256"],
            "key_source_range_sha256": source["key_selected_bf16_sha256"],
            "query_range_sha256": _sha256_bytes(source["query"].astype("<f4", copy=False).tobytes()),
            "key_range_sha256": _sha256_bytes(source["key"].astype("<f4", copy=False).tobytes()),
            "frame_index": 0, "frame_tokens": FRAME_ROWS,
        },
    }


def run_capture(
    *,
    input_manifest: str | Path,
    input_manifest_sha256: str,
    input_payload_sha256: str,
    derived_directory: str | Path,
    source_tensors: str | Path,
    source_receipt: str | Path,
    report: str | Path,
    weights_path: str | Path,
    weights_sha256: str,
    upstream_root: str | Path,
) -> tuple[Path, str]:
    """Run the admitted one-block MPS source oracle and publish frame-0 Q/K."""
    cache = _load_module(SOURCE_CACHE_PATH, "hy_nomas_reference_cache_for_query_capture")
    query_reference = _load_module(QUERY_REFERENCE_PATH, "hy_nomas_query_correspondence_for_capture")
    if (input_manifest_sha256 != DEFAULT_INPUT_MANIFEST_SHA256
            or input_payload_sha256 != DEFAULT_INPUT_PAYLOAD_SHA256
            or weights_sha256 != cache.BLOCK_WEIGHTS_SHA256):
        raise QueryCaptureError("caller pins differ from this narrowly admitted T0 capture")
    tensors_path = _fresh_output(source_tensors, "source tensor output")
    receipt_path = _fresh_output(source_receipt, "source receipt output")
    report_path = _fresh_output(report, "published input report")
    if len({tensors_path, receipt_path, report_path, report_path.with_suffix(".f32le")}) != 4:
        raise QueryCaptureError("all source and published outputs must be distinct")
    if tensors_path.parent != receipt_path.parent:
        raise QueryCaptureError("source tensors and receipt must share a directory")

    bundle = prepare_derived_bundle(
        input_manifest, derived_directory,
        expected_manifest_sha256=input_manifest_sha256,
        expected_payload_sha256=input_payload_sha256,
    )
    derived = bundle["derived_bundle"]
    artifact = bundle["reconstructed_reference_latent"]
    source_receipt_doc = cache.run_reference_block0_cache_oracle(
        derived["manifest_path"],
        reference_latent_path=artifact["path"],
        expected_reference_latent_artifact_sha256=artifact["sha256"],
        expected_timeline_manifest_sha256=TIMELINE_MANIFEST_SHA256,
        expected_manifest_sha256=derived["manifest_sha256"],
        weights_path=weights_path,
        expected_weights_sha256=weights_sha256,
        upstream_root=upstream_root,
        output_safetensors=tensors_path,
        receipt_path=receipt_path,
        precision="bf16",
    )
    query, key, source_facts = _extract_source_qk(receipt_path)
    if (source_facts["artifact_sha256"] != source_receipt_doc["tensor_artifact"]["sha256"]
            or source_facts["receipt_sha256"] != _sha256_file(receipt_path)):
        raise QueryCaptureError("source oracle changed its published artifact/receipt identity")
    bundle["source_receipt_input_bundle"] = source_facts["input_bundle"]
    bundle["source_artifact"] = {
        "path": source_facts["artifact_path"], "sha256": source_facts["artifact_sha256"],
    }
    source_provenance = {
        **source_facts,
        "query": query,
        "key": key,
    }
    provenance = _make_capture_provenance(bundle, source_provenance)
    published, digest = query_reference.publish_input(report_path, query, key, provenance)
    verification = verify_published_input(published, digest)
    return published, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="required explicit authorization to run the single MPS block-0 source pass")
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--input-manifest-sha256", default=DEFAULT_INPUT_MANIFEST_SHA256)
    parser.add_argument("--input-payload-sha256", default=DEFAULT_INPUT_PAYLOAD_SHA256)
    parser.add_argument("--derived-directory", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--source-tensors", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("--execute is required; this command performs one MPS source block-0 forward")
    cache = _load_module(SOURCE_CACHE_PATH, "hy_nomas_reference_cache_for_query_capture")
    weights = args.weights or cache.BLOCK_WEIGHTS_DEFAULT
    weights_sha = args.weights_sha256 or cache.BLOCK_WEIGHTS_SHA256
    upstream = args.upstream_root or cache.UPSTREAM_ROOT_DEFAULT
    report, digest = run_capture(
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
        input_payload_sha256=args.input_payload_sha256,
        derived_directory=args.derived_directory,
        source_tensors=args.source_tensors,
        source_receipt=args.source_receipt,
        report=args.report,
        weights_path=weights,
        weights_sha256=weights_sha,
        upstream_root=upstream,
    )
    print(json.dumps({
        "manifest": str(report), "manifest_sha256": digest,
        "verification": verify_published_input(report, digest),
        "source_run": "one pinned BF16 MPS block-0 pass; stops before attention",
        "scope": CAPTURE_SCOPE,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
