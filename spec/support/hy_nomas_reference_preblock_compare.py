#!/usr/bin/env python3
"""Compare authenticated native and source T0 tensors immediately before block 0."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

try:
    from . import hy_nomas_reference_cache as native_cache
except ImportError:  # direct import by repository test modules
    import hy_nomas_reference_cache as native_cache


SCHEMA = "cogni_ml.hy_nomas_reference_preblock_compare/v1"
SOURCE_SCHEMA = "cogni_ml.hy_nomas_reference_preblock_source_trace/v1"
SOURCE_VERSION = 1
SOURCE_SCOPES = {
    "source-bf16": (
        "actual source-BF16 T0 preprocessing captured immediately before block 0; "
        "no block forward or K/V write"
    ),
    "source-f32-precision-control": (
        "actual source-F32 precision-control T0 preprocessing captured immediately before block 0; "
        "no block forward or K/V write"
    ),
}
TENSOR_ORDER = (
    "reference.latent_bf16",
    "cache.noisy_latent_f32",
    "block.img_f32",
    "block.txt_valid_f32",
    "block.vec_f32",
    "block.vec_txt_f32",
    "block.rope_cos_f32",
    "block.rope_sin_f32",
    "cache.timestep_f32",
)
_SOURCE_FIELDS = {
    "structured_control_schema", "conditioning_manifest_sha256",
    "conditioning_payload_sha256", "t0_qwen_manifest_sha256",
    "t0_qwen_payload_sha256", "t0_qwen_tensor_sha256",
    "t1_qwen_manifest_sha256", "t1_qwen_payload_sha256",
    "t1_qwen_tensor_sha256", "action_profile_sha256",
    "clean_t0_latent_bank_sha256", "cache_index_sha256",
    "global_shard_sha256", "upstream_revision", "upstream_source_sha256",
}
_EXECUTION_FIELDS = {
    "mode", "device", "t0_valid_text_tokens", "dit_text_rows",
    "block_loader_calls", "loaded_dit_blocks", "block_weight_tensor_loads",
    "block_forward_calls", "first_block_boundary_calls", "terminal_cpu_exports",
    "cache_kv_written", "observed_stage_order", "writer_call",
    "source_dtype_by_tensor", "export_dtype_by_tensor", "semantic_notes",
}
_PRELUDE_F32_CONTROL = {
    "module_weights": "exact pinned BF16 checkpoint values widened to F32",
    "refiner_attention_adapter": "F32 installed before the pinned refiner binds its imported attention symbol",
    "img_in_input": "actual source BF16 T0 writer latent widened to F32 at the module prehook",
    "txt_in_input": "actual authenticated source BF16 T0 Qwen tensor widened to F32 at the module prehook",
    "time_in_input": "unchanged source F32 writer timestep",
    "writer_blend": "unchanged source BF16 clean/noise blend and BF16 cache-epsilon input",
    "block_execution": "sentinel stops before block 0; no K/V write",
}
_BF16_SEMANTIC_NOTES = [
    "cache.noisy_latent_f32 is the actual BF16-rounded source writer latent widened to F32 for file export",
    "the other block inputs are the exact tensors handed to source block 0, widened only where required by the native fixture format",
    "this trace proves the preblock boundary only; it does not execute DiT block 0 or claim cache K/V parity",
    "CPU admission still authenticates the pinned 54-shard weight index and identities; the trace loader instantiates none of those block tensors",
]
_F32_CONTROL_SEMANTIC_NOTES = [
    _BF16_SEMANTIC_NOTES[0],
    "the F32 precision-control prelude outputs are the exact tensors handed to source block 0 after widening the actual BF16 writer latent and Qwen input",
    *_BF16_SEMANTIC_NOTES[2:],
]
_SOURCE_SEMANTIC_NOTES = {
    "source-bf16": _BF16_SEMANTIC_NOTES,
    "source-f32-precision-control": _F32_CONTROL_SEMANTIC_NOTES,
}
_TENSOR_RECORD_FIELDS = {"dtype", "shape", "offset", "nbytes", "sha256"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
ATOL = 2e-5
RTOL = 2e-5
METRIC_KEYS = ("count", "finite_count", "max_abs", "mae", "rms", "mismatch_count")
DIAGNOSTIC_VIEW_NAMES = (
    "cache.noisy_latent_f32", "block.img_f32", "block.txt_valid_f32",
    "block.vec_f32", "block.vec_txt_f32",
)
_EXPORT_TYPES = {
    name: ("BF16" if name == "reference.latent_bf16" else "F32")
    for name in TENSOR_ORDER
}
_BF16_SOURCE_TYPES = {
    "reference.latent_bf16": "BF16",
    "cache.noisy_latent_f32": "BF16",
    "block.img_f32": "BF16",
    "block.txt_valid_f32": "BF16",
    "block.vec_f32": "BF16",
    "block.vec_txt_f32": "BF16",
    "block.rope_cos_f32": "F32",
    "block.rope_sin_f32": "F32",
    "cache.timestep_f32": "F32",
}
_F32_CONTROL_SOURCE_TYPES = {
    **_BF16_SOURCE_TYPES,
    "block.img_f32": "F32",
    "block.txt_valid_f32": "F32",
    "block.vec_f32": "F32",
    "block.vec_txt_f32": "F32",
}
_SOURCE_DTYPE_MAPS = {
    "source-bf16": _BF16_SOURCE_TYPES,
    "source-f32-precision-control": _F32_CONTROL_SOURCE_TYPES,
}
_SOURCE_TOP_FIELDS = {
    "schema", "schema_version", "scope", "source", "execution",
    "valid_text_tokens", "payload", "payload_bytes", "payload_sha256",
    "tensor_order", "tensors",
}
_WRITER_FIELDS = {
    "kind", "segment", "step", "condition", "valid_text_tokens",
    "cache_valid_len", "update_cache", "cache_txt", "rope_bank", "timestep",
}


class PreblockCompareError(ValueError):
    """A preblock manifest, payload, or comparison contract was rejected."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _read_unique_json(path: Path, what: str, *, limit: int = 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise PreblockCompareError(f"{what} must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise PreblockCompareError(f"{what} exceeds the metadata size limit")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PreblockCompareError(f"duplicate {what} JSON field: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreblockCompareError(f"{what} JSON is invalid") from error
    if type(document) is not dict:
        raise PreblockCompareError(f"{what} must be a JSON object")
    return document, raw


def _canonical_file(value: str | Path, what: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PreblockCompareError(f"{what} must be an absolute regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PreblockCompareError(f"cannot resolve {what}") from error


def _expected_source_shapes(valid_text_tokens: int) -> dict[str, tuple[int, ...]]:
    if type(valid_text_tokens) is not int or not 1 <= valid_text_tokens <= 1280:
        raise PreblockCompareError("source valid-text token count is outside [1, 1280]")
    return {
        "reference.latent_bf16": (1, 32, 30, 9, 16),
        "cache.noisy_latent_f32": (1, 32, 30, 9, 16),
        "block.img_f32": (1, 4320, 2048),
        "block.txt_valid_f32": (1, valid_text_tokens, 2048),
        "block.vec_f32": (1, 2048),
        "block.vec_txt_f32": (1, 2048),
        "block.rope_cos_f32": (4320, 128),
        "block.rope_sin_f32": (4320, 128),
        "cache.timestep_f32": (1,),
    }


def _validate_source_precision_profile(execution: Mapping[str, Any]) -> dict[str, str]:
    """Validate the consumed dtype map and semantics for the declared source mode."""
    mode = execution.get("mode")
    source_dtypes = _SOURCE_DTYPE_MAPS.get(mode)
    if source_dtypes is None:
        raise PreblockCompareError("source precision mode is not admitted")
    if execution.get("source_dtype_by_tensor") != source_dtypes:
        raise PreblockCompareError("source consumed dtypes differ from its declared precision mode")
    if execution.get("semantic_notes") != _SOURCE_SEMANTIC_NOTES[mode]:
        raise PreblockCompareError("source semantic notes differ from the declared precision mode")
    if (mode == "source-f32-precision-control"
            and execution.get("prelude_precision_control") != _PRELUDE_F32_CONTROL):
        raise PreblockCompareError("source F32 prelude precision-control semantics are not authenticated")
    return source_dtypes


def output_path(value: str | Path) -> Path:
    """Admit a fresh absolute JSON destination without creating it."""
    path = Path(value).expanduser()
    if (not path.is_absolute() or path.suffix.lower() != ".json"
            or not path.parent.is_dir() or path.is_symlink()
            or os.path.lexists(path)):
        raise PreblockCompareError(
            "--output must be a fresh absolute .json path in an existing directory"
        )
    return path


def _expected_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    count = math.prod(shape)
    return count * (2 if dtype == "BF16" else 4)


def _validate_source_trace(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_timeline_manifest_sha256: str,
    native_validation: Mapping[str, Any],
    native_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the source trace and bind every overlapping native identity."""
    if not _is_sha256(expected_manifest_sha256):
        raise PreblockCompareError("expected source manifest SHA-256 is malformed")
    if not _is_sha256(expected_timeline_manifest_sha256):
        raise PreblockCompareError("expected timeline manifest SHA-256 is malformed")
    manifest = _canonical_file(manifest_path, "source preblock manifest")
    document, raw_manifest = _read_unique_json(manifest, "source preblock manifest")
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise PreblockCompareError("source manifest SHA-256 differs from caller authorization")
    if set(document) != _SOURCE_TOP_FIELDS:
        raise PreblockCompareError("source preblock manifest fields differ from its pinned schema")
    if (document.get("schema") != SOURCE_SCHEMA
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != SOURCE_VERSION):
        raise PreblockCompareError("source preblock schema, version, or scope mismatch")

    source = document.get("source")
    if type(source) is not dict or set(source) != _SOURCE_FIELDS:
        raise PreblockCompareError("source identity fields differ from the preblock schema")
    if source.get("structured_control_schema") != "cogni_ml.hy_nomas_structured_reference_control/v1":
        raise PreblockCompareError("source structured-control schema mismatch")
    sha_fields = _SOURCE_FIELDS - {"structured_control_schema", "upstream_revision", "upstream_source_sha256"}
    if any(not _is_sha256(source.get(field)) for field in sha_fields):
        raise PreblockCompareError("source identity contains a malformed SHA-256")
    if not isinstance(source.get("upstream_revision"), str) or not _GIT_SHA1.fullmatch(source["upstream_revision"]):
        raise PreblockCompareError("source upstream revision is malformed")
    upstream_hashes = source.get("upstream_source_sha256")
    if (type(upstream_hashes) is not dict or not upstream_hashes
            or any(not isinstance(name, str) or not name
                   or not _is_sha256(digest) for name, digest in upstream_hashes.items())):
        raise PreblockCompareError("source upstream source inventory is malformed")

    execution = document.get("execution")
    if type(execution) is not dict or not _EXECUTION_FIELDS.issubset(execution):
        raise PreblockCompareError("source preblock execution fields differ from its pinned schema")
    mode = execution.get("mode")
    source_dtypes = _validate_source_precision_profile(execution)
    if document.get("scope") != SOURCE_SCOPES[mode]:
        raise PreblockCompareError("source preblock scope differs from its declared precision mode")
    expected_execution_fields = _EXECUTION_FIELDS | (
        {"prelude_precision_control"}
        if mode == "source-f32-precision-control" else set()
    )
    if set(execution) != expected_execution_fields:
        raise PreblockCompareError("source preblock execution fields differ from its precision-mode schema")
    if execution.get("device") != "mps":
        raise PreblockCompareError("source preblock was not captured on MPS")
    valid_tokens = document.get("valid_text_tokens")
    shapes = _expected_source_shapes(valid_tokens)
    if (type(execution.get("t0_valid_text_tokens")) is not int
            or execution["t0_valid_text_tokens"] != valid_tokens):
        raise PreblockCompareError("source execution token count differs from its tensor geometry")
    text_rows = execution.get("dit_text_rows")
    if (type(text_rows) is not int or not valid_tokens <= text_rows <= 1280
            or text_rows % 64 != 0):
        raise PreblockCompareError("source DIT text rows do not admit the observed text prefix")
    expected_counters = {
        "block_loader_calls": 1,
        "loaded_dit_blocks": 0,
        "block_weight_tensor_loads": 0,
        "block_forward_calls": 0,
        "first_block_boundary_calls": 1,
        "terminal_cpu_exports": 1,
        "cache_kv_written": False,
        "observed_stage_order": ["time_in", "txt_in", "img_in", "block0"],
    }
    for field, expected in expected_counters.items():
        if type(execution.get(field)) is not type(expected) or execution[field] != expected:
            raise PreblockCompareError(f"source execution does not prove the admitted block-0 boundary: {field}")
    if execution.get("export_dtype_by_tensor") != _EXPORT_TYPES:
        raise PreblockCompareError("source export dtypes differ from the nine-tensor contract")

    writer = execution.get("writer_call")
    if type(writer) is not dict or set(writer) != _WRITER_FIELDS:
        raise PreblockCompareError("source first-block writer call is malformed")
    expected_writer = {
        "kind": "writer", "segment": 0, "step": None, "condition": "t0_caption",
        "valid_text_tokens": valid_tokens, "cache_valid_len": 0,
        "update_cache": True, "cache_txt": True, "rope_bank": 0,
    }
    for field, expected in expected_writer.items():
        if type(writer.get(field)) is not type(expected) or writer[field] != expected:
            raise PreblockCompareError(f"source writer call differs from the T0 cache contract: {field}")
    if type(writer.get("timestep")) is not float or writer["timestep"] != 206.897:
        raise PreblockCompareError("source writer timestep differs from the T0 initialization forward")

    if source["conditioning_manifest_sha256"] != expected_timeline_manifest_sha256:
        raise PreblockCompareError("source conditioning timeline differs from caller authorization")
    if (native_validation.get("timeline_manifest_sha256") != expected_timeline_manifest_sha256
            or native_validation.get("valid_text_tokens") != valid_tokens):
        raise PreblockCompareError("source timeline or text extent differs from authenticated native input")

    native_preprocessing = native_document.get("preprocessing") if native_document is not None else None
    native_writer = native_document.get("cache_writer") if native_document is not None else None
    native_tensors = native_document.get("tensors") if native_document is not None else None
    if (type(native_preprocessing) is not dict
            or source["upstream_revision"] != native_preprocessing.get("timeline_upstream_revision")):
        raise PreblockCompareError("source timeline revision differs from authenticated native preprocessing")
    if (type(native_writer) is not dict
            or native_writer.get("segment_index") != writer["segment"]
            or native_writer.get("denoising_steps") != 0
            or native_writer.get("timestep") != writer["timestep"]
            or native_writer.get("technical_label") != "T0"):
        raise PreblockCompareError("source and native cache writers do not identify the same T0 forward")

    if (document.get("tensor_order") != list(TENSOR_ORDER)
            or type(document.get("tensors")) is not dict
            or tuple(document["tensors"]) != TENSOR_ORDER):
        raise PreblockCompareError("source tensor inventory or order differs from the nine-tensor contract")
    records = document["tensors"]
    if native_tensors is not None and (type(native_tensors) is not dict or tuple(native_tensors) != TENSOR_ORDER):
        raise PreblockCompareError("native tensor inventory differs from the source preblock contract")
    offset = 0
    for name in TENSOR_ORDER:
        record = records[name]
        dtype = _EXPORT_TYPES[name]
        nbytes = _expected_bytes(dtype, shapes[name])
        if (type(record) is not dict or set(record) != _TENSOR_RECORD_FIELDS
                or type(record.get("dtype")) is not str or record["dtype"] != dtype
                or type(record.get("shape")) is not list or record["shape"] != list(shapes[name])
                or any(type(value) is not int for value in record["shape"])
                or type(record.get("offset")) is not int or record["offset"] != offset
                or type(record.get("nbytes")) is not int or record["nbytes"] != nbytes
                or not _is_sha256(record.get("sha256"))):
            raise PreblockCompareError(f"source tensor record is invalid: {name}")
        if native_tensors is not None:
            native_record = native_tensors[name]
            if (native_record.get("dtype") != dtype
                    or native_record.get("shape") != list(shapes[name])):
                raise PreblockCompareError(f"source/native tensor geometry or export dtype differs: {name}")
        offset += nbytes
    if type(document.get("payload_bytes")) is not int or document["payload_bytes"] != offset:
        raise PreblockCompareError("source payload byte extent differs from the nine-tensor inventory")
    if not _is_sha256(document.get("payload_sha256")):
        raise PreblockCompareError("source payload SHA-256 is malformed")
    expected_payload_name = f"{manifest.stem}.tensors.bin"
    if document.get("payload") != expected_payload_name or Path(str(document.get("payload"))).name != expected_payload_name:
        raise PreblockCompareError("source payload must be the manifest's sibling .tensors.bin file")
    payload_path = _canonical_file(manifest.parent / expected_payload_name, "source preblock payload")
    if payload_path.stat().st_size != offset:
        raise PreblockCompareError("source payload has the wrong byte extent")
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != document["payload_sha256"]:
        raise PreblockCompareError("source payload SHA-256 mismatch")
    for name in TENSOR_ORDER:
        record = records[name]
        start = record["offset"]
        end = start + record["nbytes"]
        if hashlib.sha256(memoryview(payload)[start:end]).hexdigest() != record["sha256"]:
            raise PreblockCompareError(f"source tensor payload digest mismatch: {name}")

    source_latent_sha = records["reference.latent_bf16"]["sha256"]
    if (source_latent_sha != native_validation.get("reference_latent_tensor_sha256")
            or (native_tensors is not None
                and source_latent_sha != native_tensors["reference.latent_bf16"].get("sha256"))):
        raise PreblockCompareError("source clean T0 tensor differs from authenticated native reference bytes")

    return {
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "payload_path": str(payload_path),
        "payload_sha256": document["payload_sha256"],
        "payload_bytes": len(payload),
        "payload_bytes_data": payload,
        "document": document,
        "precision_mode": mode,
        "source_identities": dict(source),
        "valid_text_tokens": valid_tokens,
        "tensor_order": list(TENSOR_ORDER),
        "cross_bindings": {
            "conditioning_timeline_sha256": "matches caller pin and native timeline manifest",
            "timeline_upstream_revision": "matches native preprocessing timeline revision",
            "clean_reference_latent_tensor_sha256": "matches native payload and authenticated safetensors tensor",
            "valid_text_tokens": "matches native cache input manifest",
            "source_only_qwen_action_latent_bank_cache_index_global_and_source_hashes": (
                "pinned by the authenticated source manifest SHA-256; absent from the native cache input schema"
            ),
        },
    }


def validate_comparison_inputs(
    *, native_manifest_path: str | Path, native_manifest_sha256: str,
    source_manifest_path: str | Path, source_manifest_sha256: str,
    reference_latent_path: str | Path, reference_latent_sha256: str,
    expected_timeline_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate both manifests, all file hashes, and shared timeline/latent identities."""
    try:
        native_validation = native_cache.validate_reference_cache_manifest(
            native_manifest_path,
            reference_latent_path=reference_latent_path,
            expected_reference_latent_artifact_sha256=reference_latent_sha256,
            expected_timeline_manifest_sha256=expected_timeline_manifest_sha256,
            expected_manifest_sha256=native_manifest_sha256,
        )
    except native_cache.ReferenceCacheError as error:
        raise PreblockCompareError(f"native cache manifest rejected: {error}") from error
    native_manifest = _canonical_file(native_manifest_path, "native preblock manifest")
    native_document, native_raw = _read_unique_json(native_manifest, "native preblock manifest")
    native_document_sha = hashlib.sha256(native_raw).hexdigest()
    if native_document_sha != native_validation.get("manifest_sha256"):
        raise PreblockCompareError("native manifest changed after its validator completed")
    source_trace = _validate_source_trace(
        source_manifest_path,
        expected_manifest_sha256=source_manifest_sha256,
        expected_timeline_manifest_sha256=expected_timeline_manifest_sha256,
        native_validation=native_validation,
        native_document=native_document,
    )
    native_payload_path = _canonical_file(native_validation["payload_path"], "native preblock payload")
    native_payload = native_payload_path.read_bytes()
    if (len(native_payload) != native_validation.get("payload_bytes")
            or hashlib.sha256(native_payload).hexdigest() != native_validation.get("payload_sha256")):
        raise PreblockCompareError("native payload changed after its validator completed")
    return {
        "native_validation": native_validation,
        "native_document": native_document,
        "native_payload_bytes": native_payload,
        "native_payload_path": str(native_payload_path),
        "source_trace": source_trace,
    }


def _metric_stat_tensors(
    torch: Any, actual: Any, expected: Any, *, atol: float, rtol: float,
) -> dict[str, Any]:
    """Build same-device float32 metric scalars without reading them back."""
    if tuple(actual.shape) != tuple(expected.shape) or actual.device != expected.device:
        raise PreblockCompareError("metric operands have different shape or device")
    actual = actual.to(dtype=torch.float32)
    expected = expected.to(dtype=torch.float32)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    difference = torch.where(finite, actual - expected, torch.zeros_like(actual))
    absolute = torch.abs(difference)
    finite_f32 = finite.to(dtype=torch.float32)
    finite_count = finite_f32.sum(dtype=torch.float32)
    denominator = torch.clamp(finite_count, min=1.0)
    tolerance = atol + rtol * torch.abs(expected)
    mismatch = (~finite) | (absolute > tolerance)
    return {
        "count": torch.full((), float(actual.numel()), dtype=torch.float32, device=actual.device),
        "finite_count": finite_count,
        "max_abs": absolute.max(),
        "mae": absolute.sum(dtype=torch.float32) / denominator,
        "rms": torch.sqrt((difference * difference).sum(dtype=torch.float32) / denominator),
        "mismatch_count": mismatch.to(dtype=torch.float32).sum(dtype=torch.float32),
    }


def _load_tensor(torch: Any, np: Any, payload: bytes, record: Mapping[str, Any], device: Any) -> Any:
    start = record["offset"]
    view = memoryview(payload)[start:start + record["nbytes"]]
    if record["dtype"] == "BF16":
        host = np.frombuffer(view, dtype="<u2").copy()
        tensor = torch.from_numpy(host).view(torch.bfloat16).to(device=device)
        return tensor.reshape(record["shape"]).to(dtype=torch.float32)
    host = np.frombuffer(view, dtype="<f4").copy()
    return torch.from_numpy(host).to(device=device).reshape(record["shape"])


def _stats_record(values: list[float], *, atol: float, rtol: float) -> dict[str, Any]:
    count, finite_count, max_abs, mae, rms, mismatch_count = values
    count_i = int(round(count))
    finite_i = int(round(finite_count))
    mismatch_i = int(round(mismatch_count))
    return {
        "count": count_i,
        "finite_count": finite_i,
        "max_abs": max_abs,
        "mae": mae,
        "rms": rms,
        "mismatch_count": mismatch_i,
        "passed": count_i == finite_i and mismatch_i == 0,
        "atol": atol,
        "rtol": rtol,
    }


def _run_mps_comparison(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Compare raw tensors and diagnostic BF16-rounded views with one final readback."""
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise PreblockCompareError("NumPy and PyTorch are required for the MPS comparison") from error
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise PreblockCompareError("the comparator requires an available MPS device")
    device = torch.device("mps")
    native_doc = inputs["native_document"]
    source_doc = inputs["source_trace"]["document"]
    native_payload = inputs["native_payload_bytes"]
    source_payload = inputs["source_trace"]["payload_bytes_data"]
    native_records = native_doc["tensors"]
    source_records = source_doc["tensors"]
    queued: list[tuple[str, str, dict[str, Any]]] = []

    for name in TENSOR_ORDER:
        native_tensor = _load_tensor(torch, np, native_payload, native_records[name], device)
        source_tensor = _load_tensor(torch, np, source_payload, source_records[name], device)
        queued.append(("raw", name, _metric_stat_tensors(
            torch, native_tensor, source_tensor, atol=ATOL, rtol=RTOL
        )))
        if name in DIAGNOSTIC_VIEW_NAMES:
            native_rounded = native_tensor.to(dtype=torch.bfloat16).to(dtype=torch.float32)
            queued.append(("native_bf16_rounding_view", name, _metric_stat_tensors(
                torch, native_rounded, source_tensor, atol=ATOL, rtol=RTOL
            )))
        del native_tensor, source_tensor

    # These probes use the same GPU metric path and catch a broken allclose gate.
    control = torch.arange(16, dtype=torch.float32, device=device)
    perturb = torch.full((16,), 0.01, dtype=torch.float32, device=device)
    queued.append(("sensitivity_control", "a_vs_a", _metric_stat_tensors(
        torch, control, control, atol=ATOL, rtol=RTOL
    )))
    queued.append(("sensitivity_control", "deliberate_perturbation", _metric_stat_tensors(
        torch, control + perturb, control, atol=ATOL, rtol=RTOL
    )))

    # All comparison and control reductions are queued above; this is the only
    # device-to-host readback in the comparator.
    flat = [stats[key] for _kind, _name, stats in queued for key in METRIC_KEYS]
    scalar_values = torch.stack(flat).detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    stride = len(METRIC_KEYS)
    for index, (kind, name, _stats) in enumerate(queued):
        values = scalar_values[index * stride:(index + 1) * stride]
        rows.append({"kind": kind, "tensor": name,
                     **_stats_record(values, atol=ATOL, rtol=RTOL)})

    raw_rows = [row for row in rows if row["kind"] == "raw"]
    rounded_rows = [row for row in rows if row["kind"] == "native_bf16_rounding_view"]
    controls = {row["tensor"]: row for row in rows if row["kind"] == "sensitivity_control"}
    if (len(raw_rows) != len(TENSOR_ORDER)
            or tuple(row["tensor"] for row in raw_rows) != TENSOR_ORDER
            or tuple(row["tensor"] for row in rounded_rows) != DIAGNOSTIC_VIEW_NAMES):
        raise PreblockCompareError("comparison queue lost a required tensor or diagnostic view")
    control_passed = controls["a_vs_a"]["passed"] and not controls["deliberate_perturbation"]["passed"]
    if not control_passed:
        raise PreblockCompareError("GPU A/A or deliberate-perturbation sensitivity control failed")
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "numeric_gate_passed": all(row["passed"] for row in raw_rows),
        "raw_tensors": raw_rows,
        "native_bf16_rounded_views": {
            "scope": (
                "diagnostic rounding view of native F32 activation exports; it is not a claim "
                "that all source F32 arithmetic boundaries are emulated"
            ),
            "tensors": rounded_rows,
        },
        "sensitivity_controls": {
            "passed": control_passed,
            "a_vs_a": controls["a_vs_a"],
            "deliberate_perturbation": controls["deliberate_perturbation"],
        },
    }


def _publish_fresh(path: Path, document: Mapping[str, Any]) -> str:
    encoded = (json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--reference-latent", type=Path, required=True)
    parser.add_argument("--reference-latent-sha256", required=True)
    parser.add_argument("--expected-timeline-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        destination = output_path(args.output)
        inputs = validate_comparison_inputs(
            native_manifest_path=args.native_manifest,
            native_manifest_sha256=args.native_manifest_sha256,
            source_manifest_path=args.source_manifest,
            source_manifest_sha256=args.source_manifest_sha256,
            reference_latent_path=args.reference_latent,
            reference_latent_sha256=args.reference_latent_sha256,
            expected_timeline_manifest_sha256=args.expected_timeline_sha256,
        )
        comparison = _run_mps_comparison(inputs)
        source = inputs["source_trace"]
        native = inputs["native_validation"]
        result = {
            "schema": SCHEMA,
            "status": "comparison_complete",
            "scope": "source/native T0 preblock tensors only; no block forward, K/V comparison, or T1 claim",
            "source": {
                "manifest_sha256": source["manifest_sha256"],
                "payload_sha256": source["payload_sha256"],
                "precision_mode": source["precision_mode"],
                "source_identities": source["source_identities"],
                "cross_bindings": source["cross_bindings"],
            },
            "native": {
                "manifest_sha256": native["manifest_sha256"],
                "payload_sha256": native["payload_sha256"],
                "reference_latent_artifact_sha256": native["reference_latent_artifact_sha256"],
                "reference_latent_tensor_sha256": native["reference_latent_tensor_sha256"],
                "timeline_manifest_sha256": native["timeline_manifest_sha256"],
                "upstream_revision": native["upstream_revision"],
                "timeline_upstream_revision": native["preprocessing"]["timeline_upstream_revision"],
                "valid_text_tokens": native["valid_text_tokens"],
            },
            **comparison,
        }
        output_sha = _publish_fresh(destination, result)
    except (PreblockCompareError, native_cache.ReferenceCacheError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected", "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "schema": SCHEMA,
        "status": result["status"],
        "output": str(destination),
        "output_sha256": output_sha,
        "numeric_gate_passed": result["numeric_gate_passed"],
        "precision_mode": result["source"]["precision_mode"],
        "tensor_count": len(result["raw_tensors"]),
        "sensitivity_controls_passed": result["sensitivity_controls"]["passed"],
    }, sort_keys=True))
    # A completed numerical comparison is a successful command even when the
    # unchanged-tolerance gate reports a mismatch.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
