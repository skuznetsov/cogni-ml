#!/usr/bin/env python3
"""Independent CPU qualification for the frozen Hy-Nomas soft-route probe.

This checker verifies pinned manifests and tensor bytes, exact zero/A-A
controls, source/native parity, protected rows, and predeclared regional
velocity sensitivities. It does not inspect rendered video or verify external
process-memory guard receipts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SOURCE_SCHEMA = "hy-nomas-soft-route-reference-v1"
NATIVE_SCHEMA = "hy-nomas-soft-route-native-v1"
PARENT_SCHEMA = "hy-nomas-dit-route-factorial-reference-v1"
PARENT_SHA256 = "efcf746fd334218d25b0a922647f10962c6d5102144f7d80cff67bb327a40c89"
PARENT_PAYLOAD_SHA256 = "2f2c634a97787df22be094efafb778924a545827192c6c50597a63df0578a24a"
BIAS_SCHEMA = "hy-nomas-soft-route-bias-v1"
BIAS_SHA256 = "81dd6c8d9ba01bd8824cba8c0c48f238f9180c2cb22a3f18f7c2cb2d5331235e"
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"

CASE_ORDER = (
    "zero", "A1_low", "A1_high", "A2_low", "A2_high", "A1_sham", "A2_sham", "zero_repeat",
)
POSITIVE_CASES = CASE_ORDER[1:-1]
CHECKPOINTS = ("velocity", "block0.image", "block0.text_valid", "block53.image", "block53.text_valid")
ATOL = 2.0e-5
RTOL = 2.0e-5
VIDEO_ROWS = 4320
QUERY_ROWS = 4544
HISTORY_ROWS = 4583
KEY_ROWS = 9127
HIDDEN = 2048
BLOCKS = 54
T1_TEXT_ROWS = 224
MAX_NAMED_BYTES = 21 * 1024**3
HASH_CHUNK_BYTES = 8 * 1024 * 1024
ARRAY_CHUNK_ELEMENTS = 1_000_000
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

BASE_GEOMETRY = {
    "video_rows": VIDEO_ROWS,
    "t0_valid_text_rows": 263,
    "t1_valid_text_rows": T1_TEXT_ROWS,
    "text_storage_rows": 320,
    "physical_cache_rows": 4640,
    "history_rows": HISTORY_ROWS,
    "query_rows": QUERY_ROWS,
    "key_rows": KEY_ROWS,
    "hidden": HIDDEN,
    "heads": 16,
    "head_dim": 128,
    "blocks": BLOCKS,
    "attention_layout": "compact",
}
SOFT_GEOMETRY = {
    **BASE_GEOMETRY,
    "bias_rows": VIDEO_ROWS,
    "bias_text_suffix_rows": QUERY_ROWS - VIDEO_ROWS,
    "action_key_spans": {
        "A1": [HISTORY_ROWS + VIDEO_ROWS + 49, HISTORY_ROWS + VIDEO_ROWS + 132],
        "A2": [HISTORY_ROWS + VIDEO_ROWS + 132, HISTORY_ROWS + VIDEO_ROWS + 224],
    },
}
OUTPUT_SHAPES = {
    "velocity": (1, 32, 30, 9, 16),
    "block0.image": (1, VIDEO_ROWS, HIDDEN),
    "block0.text_valid": (1, T1_TEXT_ROWS, HIDDEN),
    "block53.image": (1, VIDEO_ROWS, HIDDEN),
    "block53.text_valid": (1, T1_TEXT_ROWS, HIDDEN),
}
PARENT_CASES = ("M11", "M00", "M10", "M01", "M11_repeat")
PARENT_TENSOR_NAMES = tuple(f"{case}.{checkpoint}" for case in PARENT_CASES for checkpoint in CHECKPOINTS)
TENSOR_NAMES = tuple(f"{case}.{checkpoint}" for case in CASE_ORDER for checkpoint in CHECKPOINTS)

ACTION_WINDOWS = {
    "A1": {"frames": (1, 10), "part_hw": ((0, 4), (4, 8)), "actor_hw": ((0, 9), (1, 8))},
    "A2": {"frames": (12, 28), "part_hw": ((4, 6), (11, 13)), "actor_hw": ((2, 8), (10, 15))},
}


class QualificationError(RuntimeError):
    """Fail-closed manifest, payload, parity, or numerical gate error."""


@dataclass(frozen=True)
class Artifact:
    kind: str
    manifest_path: Path
    manifest_sha256: str
    document: dict[str, Any]
    payload_path: Path
    arrays: Mapping[str, np.memmap]


def _fail_if(condition: bool, message: str) -> None:
    if condition:
        raise QualificationError(message)


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise QualificationError(f"{label} must be lowercase SHA-256")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, nbytes: int) -> str:
    if type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes < 0:
        raise QualificationError("tensor range bounds are malformed")
    if offset + nbytes > path.stat().st_size:
        raise QualificationError(f"tensor range escapes payload: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = nbytes
        while remaining:
            block = stream.read(min(remaining, HASH_CHUNK_BYTES))
            if not block:
                raise QualificationError(f"truncated payload range: {path.name}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _canonical_regular(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise QualificationError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise QualificationError(f"{label} is missing: {candidate}") from None
    if not resolved.is_file():
        raise QualificationError(f"{label} is not a regular file: {candidate}")
    return resolved


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise QualificationError(f"non-standard JSON numeric constant: {value}")


def _strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int and int/float coercions."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_strict_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_strict_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def _same_resolved_directory(left: Any, right: Any, label: str) -> None:
    """Allow mount/symlink aliases while requiring the same existing directory."""
    if (not isinstance(left, str) or not left or not Path(left).is_absolute()
            or not isinstance(right, str) or not right or not Path(right).is_absolute()):
        raise QualificationError(f"{label} path binding is malformed")
    try:
        left_path = Path(left).resolve(strict=True)
        right_path = Path(right).resolve(strict=True)
        if not left_path.is_dir() or not right_path.is_dir() or not os.path.samefile(left_path, right_path):
            raise QualificationError(f"{label} does not resolve to the parent source directory")
    except OSError:
        raise QualificationError(f"{label} path is missing or inaccessible") from None


def _read_pinned_json(path: str | Path, expected_sha256: str, label: str) -> tuple[Path, str, dict[str, Any]]:
    expected = _digest(expected_sha256, f"trusted {label} SHA-256")
    canonical = _canonical_regular(path, label)
    if canonical.stat().st_size > MAX_MANIFEST_BYTES:
        raise QualificationError(f"{label} manifest exceeds the bounded size limit")
    actual = sha256_file(canonical)
    if actual != expected:
        raise QualificationError(f"{label} manifest digest mismatch")
    try:
        parsed = json.loads(
            canonical.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except QualificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise QualificationError(f"invalid {label} JSON") from None
    if type(parsed) is not dict:
        raise QualificationError(f"{label} root must be an object")
    return canonical, actual, parsed


def _shape_size(shape: Sequence[int], label: str) -> int:
    if not isinstance(shape, (tuple, list)) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise QualificationError(f"{label} has an invalid tensor shape")
    return math.prod(shape)


def validate_bf16_expansion(array: np.ndarray, label: str, chunk_elements: int = ARRAY_CHUNK_ELEMENTS) -> None:
    """Require finite F32 values whose low 16 bits are zero (lossless BF16 expansion)."""
    if array.dtype != np.dtype("<f4") and array.dtype != np.dtype("float32"):
        raise QualificationError(f"{label} is not little-endian F32")
    flat = array.reshape(-1)
    for start in range(0, flat.size, chunk_elements):
        piece = np.asarray(flat[start : start + chunk_elements])
        if not np.isfinite(piece).all():
            raise QualificationError(f"{label} contains non-finite values")
        bits = piece.view(np.uint32)
        if np.any(bits & np.uint32(0xFFFF)):
            raise QualificationError(f"{label} is not a lossless BF16 expansion")


def validate_tensor_payload(
    payload_path: str | Path,
    records: Any,
    names: Sequence[str],
    shapes: Mapping[str, Sequence[int]],
    *,
    record_fields: Sequence[str],
    label: str,
) -> dict[str, np.memmap]:
    """Validate exact tensor inventory, contiguous extents, hashes and BF16 bytes.

    This lower-level function is parameterized so tests can attack the same
    range/precision checks with small synthetic payloads.
    """
    payload = _canonical_regular(payload_path, f"{label} payload")
    if type(records) is not dict or set(records) != set(names):
        raise QualificationError(f"{label} tensor inventory mismatch")
    mapped: dict[str, np.memmap] = {}
    cursor = 0
    expected_record_fields = set(record_fields)
    for name in names:
        record = records[name]
        if type(record) is not dict or set(record) != expected_record_fields:
            raise QualificationError(f"{label} tensor record field set mismatch: {name}")
        shape = tuple(shapes[name])
        nbytes = _shape_size(shape, name) * 4
        record_shape = record.get("shape")
        if (type(record_shape) is not list or any(type(dim) is not int for dim in record_shape)
                or record_shape != list(shape)):
            raise QualificationError(f"{label} tensor shape mismatch: {name}")
        if record.get("dtype") != "F32" or record.get("source_dtype") != "BF16":
            raise QualificationError(f"{label} tensor dtype mismatch: {name}")
        if type(record.get("offset_bytes")) is not int or record["offset_bytes"] != cursor:
            raise QualificationError(f"{label} tensor offset is not contiguous: {name}")
        if type(record.get("nbytes")) is not int or record["nbytes"] != nbytes:
            raise QualificationError(f"{label} tensor byte length mismatch: {name}")
        if "checkpoint_boundary" in expected_record_fields:
            expected_boundary = name.split(".", 1)[1]
            if record.get("checkpoint_boundary") != expected_boundary:
                raise QualificationError(f"{label} checkpoint boundary mismatch: {name}")
        record_sha = _digest(record.get("sha256"), f"{label} tensor {name} SHA-256")
        if _sha256_range(payload, cursor, nbytes) != record_sha:
            raise QualificationError(f"{label} tensor range digest mismatch: {name}")
        tensor = np.memmap(payload, dtype="<f4", mode="r", offset=cursor, shape=shape, order="C")
        validate_bf16_expansion(tensor, f"{label}.{name}")
        mapped[name] = tensor
        cursor += nbytes
    if cursor != payload.stat().st_size:
        raise QualificationError(f"{label} tensor ranges do not exactly cover payload")
    return mapped


def compare_arrays(expected: Any, actual: Any, *, atol: float = ATOL, rtol: float = RTOL,
                   chunk_elements: int = ARRAY_CHUNK_ELEMENTS) -> dict[str, Any]:
    """Apply the fixed elementwise rule |x-y| <= atol + rtol*|x|."""
    lhs = np.asarray(expected)
    rhs = np.asarray(actual)
    if lhs.shape != rhs.shape or not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise QualificationError("parity comparison shape/nonfinite mismatch")
    if atol != ATOL or rtol != RTOL:
        raise QualificationError("parity tolerance drift; this probe is pinned to 2e-5/2e-5")
    a = lhs.reshape(-1)
    b = rhs.reshape(-1)
    count = a.size
    outside = 0
    exact_bits = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    for start in range(0, count, chunk_elements):
        left = np.asarray(a[start : start + chunk_elements], dtype=np.float64)
        right = np.asarray(b[start : start + chunk_elements], dtype=np.float64)
        delta = np.abs(left - right)
        outside += int(np.count_nonzero(delta > ATOL + RTOL * np.abs(left)))
        exact_bits += int(np.count_nonzero(
            np.asarray(a[start : start + chunk_elements], dtype=np.float32).view(np.uint32)
            != np.asarray(b[start : start + chunk_elements], dtype=np.float32).view(np.uint32)
        ))
        if delta.size:
            max_abs = max(max_abs, float(delta.max()))
            sum_abs += float(delta.sum(dtype=np.float64))
            sum_sq += float(np.dot(delta, delta))
    return {
        "elements": int(count),
        "outside_atol_rtol": outside,
        "exact_bit_differences": exact_bits,
        "max_abs": max_abs,
        "mean_abs": sum_abs / count if count else 0.0,
        "rms_abs": math.sqrt(sum_sq / count) if count else 0.0,
        "atol": ATOL,
        "rtol": RTOL,
    }


def _expected_bias_hashes(document: dict[str, Any]) -> dict[str, str]:
    values = document.get("bias_f32le_sha256")
    if type(values) is not dict or set(values) != set(CASE_ORDER):
        raise QualificationError("bias digest inventory mismatch")
    return {case: _digest(values[case], f"bias {case} digest") for case in CASE_ORDER}


def _load_bias(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    canonical, _, document = _read_pinned_json(path, BIAS_SHA256, "bias")
    if (document.get("schema") != BIAS_SCHEMA
            or not _strict_json_equal(document.get("case_order"), list(CASE_ORDER))
            or not _strict_json_equal(document.get("geometry"), {"video_rows": VIDEO_ROWS, "query_rows": QUERY_ROWS})):
        raise QualificationError("pinned bias schema, case order, or geometry mismatch")
    image = document.get("reference_image")
    if (type(image) is not dict or set(image) != {"path", "sha256"}
            or not isinstance(image.get("path"), str) or not image["path"]):
        raise QualificationError("bias reference image binding malformed")
    _digest(image.get("sha256"), "bias reference image SHA-256")
    image_path = _canonical_regular(image["path"], "bias reference image")
    if sha256_file(image_path) != image["sha256"]:
        raise QualificationError("pinned bias reference image digest mismatch")
    maps = document.get("maps")
    if type(maps) is not dict or maps.get("support") != "accepted M11; no edges added; video queries only":
        raise QualificationError("bias map support contract mismatch")
    raw = document.get("biases")
    if type(raw) is not dict or set(raw) != set(CASE_ORDER):
        raise QualificationError("bias vector inventory mismatch")
    hashes = _expected_bias_hashes(document)
    vectors: dict[str, np.ndarray] = {}
    for case in CASE_ORDER:
        values = raw[case]
        if type(values) is not list or len(values) != QUERY_ROWS:
            raise QualificationError(f"bias vector length mismatch: {case}")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise QualificationError(f"bias vector contains invalid numbers: {case}")
        vector = np.asarray(values, dtype="<f4")
        if not np.all(vector.astype(np.float64) == np.asarray(values, dtype=np.float64)):
            raise QualificationError(f"bias vector contains non-exact F32 values: {case}")
        validate_bf16_expansion(vector, f"bias.{case}")
        if np.any(vector < 0.0) or np.any(vector > 2.0):
            raise QualificationError(f"bias vector outside nonnegative bounded range: {case}")
        if case in ("zero", "zero_repeat") and np.any(vector.view(np.uint32) != 0):
            raise QualificationError(f"bias {case} is not positive-zero throughout")
        raw_bytes = vector.tobytes(order="C")
        if hashlib.sha256(raw_bytes).hexdigest() != hashes[case]:
            raise QualificationError(f"bias vector range digest mismatch: {case}")
        vectors[case] = vector
    _validate_bias_support(vectors)
    for action in ("A1", "A2"):
        low = vectors[f"{action}_low"]
        high = vectors[f"{action}_high"]
        sham = vectors[f"{action}_sham"]
        if not np.all(low <= high):
            raise QualificationError(f"bias dose order failed: {action}")
        for frame in range(30):
            start = frame * 144
            if not np.array_equal(np.sort(high[start : start + 144]), np.sort(sham[start : start + 144])):
                raise QualificationError(f"bias sham frame histogram changed: {action}:{frame}")
    return canonical, document, vectors


def _parent_roi_contains(q: int, action: int) -> bool:
    frame, rem = divmod(q, 144)
    h, w = divmod(rem, 16)
    if action == 1:
        return 1 <= frame < 10 and 0 <= h < 9 and 1 <= w < 8
    return 12 <= frame < 28 and 2 <= h < 8 and 10 <= w < 15


def _action_query_label(q: int) -> int:
    frame = q // 144
    if 1 <= frame < 10:
        return 1
    if 12 <= frame < 28:
        return 2
    return 0


def _validate_bias_support(vectors: Mapping[str, np.ndarray]) -> None:
    if set(vectors) != set(CASE_ORDER):
        raise QualificationError("bias support check has wrong case inventory")
    for case in POSITIVE_CASES:
        action = 1 if case.startswith("A1") else 2
        indexes = np.flatnonzero(vectors[case])
        if indexes.size == 0:
            raise QualificationError(f"positive bias is empty: {case}")
        for q_value in indexes:
            q = int(q_value)
            if q >= VIDEO_ROWS or _action_query_label(q) != action or not _parent_roi_contains(q, action):
                raise QualificationError(f"bias escaped its accepted M11 action support: {case}:{q}")
        if np.any(vectors[case][VIDEO_ROWS:] != 0):
            raise QualificationError(f"bias entered padded/action-text query suffix: {case}")


def _without_path_fields(value: Any) -> Any:
    if type(value) is dict:
        return {
            key: _without_path_fields(item)
            for key, item in value.items()
            if not (isinstance(key, str) and (key == "path" or key.endswith("_path")))
        }
    if type(value) is list:
        return [_without_path_fields(item) for item in value]
    return value


def _expected_source_runtime() -> dict[str, Any]:
    return {
        "attention_calls": 54 * 9,
        "attention_query_chunks": 54 * 9 * 18,
        "t0_attention_calls": 54,
        "t1_attention_calls": 54 * 8,
        "explicit_terminal_synchronizations": 1,
        "intermediate_readbacks": 0,
        "scheduler_calls": 0,
        "decoder_calls": 0,
        "refiner_calls": 0,
        "weights_staged": 54,
        "arms_per_layer": 9,
        "one_layer_cache": True,
        "autocast": "mps BF16 full block and final head",
    }


def _source_expected_precision(parent: dict[str, Any]) -> dict[str, Any]:
    precision = dict(parent["precision"])
    precision.update({
        "boolean_zero_cases": ["zero", "zero_repeat"],
        "positive_mask_dtype": "BF16 additive MPS SDPA mask; 0 on allowed support, -inf off support, authored BF16 bias on target action-text keys",
        "hard_support": "pinned accepted M11 Boolean mask; no edges added or removed",
        "bias_operation": "F32 manifest values exactly BF16-expanded; one BF16 logit addition before softmax",
    })
    return precision


def _validate_parent_document(parent: dict[str, Any], payload_path: Path) -> dict[str, np.memmap]:
    if (parent.get("schema") != PARENT_SCHEMA or type(parent.get("schema_version")) is not int
            or parent.get("schema_version") != 1 or parent.get("upstream_revision") != UPSTREAM_REVISION
            or not _strict_json_equal(parent.get("geometry"), BASE_GEOMETRY)):
        raise QualificationError("pinned direct-route parent schema/geometry/revision mismatch")
    if parent.get("case_factors", {}).get("M11") != {
            "action1_video_to_text": True, "action2_video_to_text": True}:
        raise QualificationError("pinned parent M11 factor definition mismatch")
    payload = parent.get("payload")
    if (type(payload) is not dict or payload.get("basename") != payload_path.name
            or payload.get("bytes") != payload_path.stat().st_size
            or payload.get("sha256") != PARENT_PAYLOAD_SHA256
            or payload.get("encoding") != "f32le_bf16_expansion"):
        raise QualificationError("pinned direct-route parent payload binding mismatch")
    if sha256_file(payload_path) != PARENT_PAYLOAD_SHA256:
        raise QualificationError("pinned direct-route parent payload digest mismatch")
    return validate_tensor_payload(
        payload_path,
        parent.get("tensors"),
        PARENT_TENSOR_NAMES,
        {name: OUTPUT_SHAPES[name.split(".", 1)[1]] for name in PARENT_TENSOR_NAMES},
        record_fields=("dtype", "source_dtype", "shape", "offset_bytes", "nbytes", "sha256", "checkpoint_boundary"),
        label="parent",
    )


def _load_parent(path: str | Path) -> tuple[Path, str, dict[str, Any], Path, dict[str, np.memmap]]:
    parent_path, parent_sha, document = _read_pinned_json(path, PARENT_SHA256, "accepted direct-route parent")
    payload = document.get("payload")
    if type(payload) is not dict or not isinstance(payload.get("basename"), str):
        raise QualificationError("parent payload descriptor malformed")
    basename = payload["basename"]
    if Path(basename).name != basename or basename in ("", ".", ".."):
        raise QualificationError("unsafe parent payload basename")
    payload_path = _canonical_regular(parent_path.parent / basename, "accepted parent tensor payload")
    arrays = _validate_parent_document(document, payload_path)
    return parent_path, parent_sha, document, payload_path, arrays


def _validate_source_bindings(document: dict[str, Any], parent: dict[str, Any], bias: dict[str, Any]) -> None:
    required = {
        "schema", "schema_version", "upstream_revision", "parent_reference", "bias_manifest", "source",
        "source_inputs", "geometry", "precision", "execution", "payload", "tensors",
    }
    if set(document) != required:
        raise QualificationError("source manifest key set mismatch")
    if (document.get("schema") != SOURCE_SCHEMA or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1
            or document.get("upstream_revision") != UPSTREAM_REVISION):
        raise QualificationError("source manifest schema/revision mismatch")
    parent_binding = document["parent_reference"]
    if (type(parent_binding) is not dict
            or set(parent_binding) != {"path", "sha256", "schema", "payload_path", "payload_basename", "payload_bytes", "payload_sha256"}
            or parent_binding.get("sha256") != PARENT_SHA256
            or parent_binding.get("schema") != PARENT_SCHEMA
            or parent_binding.get("payload_basename") != parent["payload"]["basename"]
            or parent_binding.get("payload_bytes") != parent["payload"]["bytes"]
            or parent_binding.get("payload_sha256") != PARENT_PAYLOAD_SHA256):
        raise QualificationError("source parent binding mismatch")
    if any(not isinstance(parent_binding.get(key), str) or not parent_binding[key]
           for key in ("path", "payload_path")):
        raise QualificationError("source parent binding path malformed")
    bias_binding = document["bias_manifest"]
    expected_bias_binding = {
        "path": bias_binding.get("path") if type(bias_binding) is dict else None,
        "sha256": BIAS_SHA256,
        "schema": BIAS_SCHEMA,
        "case_order": list(CASE_ORDER),
        "reference_image": bias["reference_image"],
        "maps": bias["maps"],
        "bias_f32le_sha256": bias["bias_f32le_sha256"],
    }
    if (type(bias_binding) is not dict or set(bias_binding) != set(expected_bias_binding)
            or bias_binding.get("sha256") != BIAS_SHA256
            or bias_binding.get("schema") != BIAS_SCHEMA
            or bias_binding.get("case_order") != list(CASE_ORDER)
            or bias_binding.get("reference_image") != expected_bias_binding["reference_image"]
            or bias_binding.get("maps") != expected_bias_binding["maps"]
            or bias_binding.get("bias_f32le_sha256") != expected_bias_binding["bias_f32le_sha256"]
            or not isinstance(bias_binding.get("path"), str) or not bias_binding["path"]):
        raise QualificationError("source bias binding mismatch")
    if (_without_path_fields(document.get("source")) != _without_path_fields(parent.get("source"))
            or _without_path_fields(document.get("source_inputs")) != _without_path_fields(parent.get("source_inputs"))):
        raise QualificationError("source weights or saved-input digest lineage drifted from parent")
    source_info = document.get("source")
    parent_source_info = parent.get("source")
    if type(source_info) is not dict or type(parent_source_info) is not dict:
        raise QualificationError("source lineage document malformed")
    _same_resolved_directory(source_info.get("upstream_path"), parent_source_info.get("upstream_path"), "upstream")
    if not _strict_json_equal(document.get("geometry"), SOFT_GEOMETRY):
        raise QualificationError("source soft-route geometry mismatch")
    if not _strict_json_equal(document.get("precision"), _source_expected_precision(parent)):
        raise QualificationError("source precision/support contract mismatch")
    execution = document.get("execution")
    expected_execution_keys = {
        "case_order", "scope", "soft_route_support", "runtime_receipt", "parent_replay", "accepted_roi_replay",
        "source_aa", "intervention_effects", "block0_text_valid_equal_all_cases", "zero_repeat_bitwise_exact",
    }
    if (type(execution) is not dict or set(execution) != expected_execution_keys
            or execution.get("case_order") != list(CASE_ORDER)
            or execution.get("scope") != "saved-state direct DiT only; no encoder/refiner/Euler/DCAE"
            or execution.get("soft_route_support") != "accepted M11; per-video-query additive bias to matching action-text keys only"
            or execution.get("runtime_receipt") != _expected_source_runtime()
            or execution.get("block0_text_valid_equal_all_cases") is not True
            or execution.get("zero_repeat_bitwise_exact") is not True):
        raise QualificationError("source execution contract/receipt mismatch")
    for gate_name in ("parent_replay", "accepted_roi_replay"):
        gate = execution.get(gate_name)
        if gate_name == "parent_replay":
            if type(gate) is not dict or set(gate) != {"zero"} or type(gate["zero"]) is not dict or set(gate["zero"]) != set(CHECKPOINTS):
                raise QualificationError(f"source {gate_name} checkpoint inventory mismatch")
            gate = gate["zero"]
        elif type(gate) is not dict or set(gate) != set(CHECKPOINTS):
            raise QualificationError(f"source {gate_name} checkpoint inventory mismatch")
        for checkpoint in CHECKPOINTS:
            record = gate[checkpoint]
            if (type(record) is not dict or record.get("passed") is not True
                    or record.get("all_finite") is not True or record.get("outside_atol_rtol") != 0
                    or record.get("atol") != ATOL or record.get("rtol") != RTOL):
                raise QualificationError(f"source {gate_name} recorded comparator failed: {checkpoint}")
    aa = execution.get("source_aa")
    if type(aa) is not dict or set(aa) != set(CHECKPOINTS):
        raise QualificationError("source A/A checkpoint inventory mismatch")
    for checkpoint in CHECKPOINTS:
        record = aa[checkpoint]
        if (type(record) is not dict or record.get("passed") is not True
                or record.get("all_finite") is not True or record.get("outside_atol_rtol") != 0
                or record.get("max_abs") != 0 or record.get("exact_bit_differences", 0) != 0
                or record.get("atol") != ATOL or record.get("rtol") != RTOL):
            raise QualificationError(f"source A/A record is not exact: {checkpoint}")
    effects = execution.get("intervention_effects")
    if type(effects) is not dict or set(effects) != set(POSITIVE_CASES):
        raise QualificationError("source intervention effect inventory mismatch")
    for case in POSITIVE_CASES:
        if type(effects[case]) is not dict or set(effects[case]) != set(CHECKPOINTS):
            raise QualificationError(f"source intervention effect checkpoint inventory mismatch: {case}")


def _validate_native_bindings(document: dict[str, Any], bias_sha256: str) -> None:
    required = {
        "schema", "parent_reference", "parent_sha256", "bias_manifest", "bias_sha256", "case_order", "geometry",
        "tolerances", "resource_accounting", "execution", "payload", "tensors", "exact_parent_and_AA",
        "passed_native_invariants", "interpretation_scope",
    }
    if set(document) != required:
        raise QualificationError("native manifest key set mismatch")
    if (document.get("schema") != NATIVE_SCHEMA or document.get("parent_sha256") != PARENT_SHA256
            or document.get("bias_sha256") != bias_sha256 or document.get("case_order") != list(CASE_ORDER)
            or not _strict_json_equal(document.get("geometry"), BASE_GEOMETRY)):
        raise QualificationError("native schema/parent/bias/case/geometry binding mismatch")
    if not isinstance(document.get("parent_reference"), str) or not isinstance(document.get("bias_manifest"), str):
        raise QualificationError("native manifest path binding malformed")
    if document.get("tolerances") != {"atol": ATOL, "rtol": RTOL}:
        raise QualificationError("native parity tolerance drift")
    if document.get("passed_native_invariants") is not True:
        raise QualificationError("native producer reported a failed invariant")
    if document.get("interpretation_scope") != "Frozen-input numerical enrichment only; source parity and visual acting require separate checks.":
        raise QualificationError("native interpretation scope mismatch")
    execution = document.get("execution")
    expected_execution_keys = {
        "blocks", "command_chains", "terminal_waits", "named_allocations_during_encode",
        "activation_readbacks_before_terminal_wait", "postwait_checkpoint_readbacks", "forward_seconds",
        "scheduler_updates", "decoder_calls",
    }
    if type(execution) is not dict or set(execution) != expected_execution_keys:
        raise QualificationError("native execution inventory mismatch")
    expected_counters = {
        "blocks": BLOCKS,
        "command_chains": 1,
        "terminal_waits": 1,
        "named_allocations_during_encode": 0,
        "activation_readbacks_before_terminal_wait": 0,
        "postwait_checkpoint_readbacks": len(TENSOR_NAMES),
        "scheduler_updates": 0,
        "decoder_calls": 0,
    }
    for key, value in expected_counters.items():
        if type(execution.get(key)) is not int or execution[key] != value:
            raise QualificationError(f"native execution counter mismatch: {key}")
    seconds = execution.get("forward_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(float(seconds)) or seconds <= 0:
        raise QualificationError("native forward time is malformed")
    memory = document.get("resource_accounting")
    parts = {
        "resident_block_weights", "block_weight_stage", "shared_block_scratch", "source_bf16_workspace",
        "current_inputs", "state_banks", "input_constants", "snapshots", "single_layer_kv",
        "attention_workspaces", "final_weights", "final_workspace",
    }
    expected_memory_keys = parts | {"aggregate_named_bytes", "setup_transient_attention_bytes", "peak_named_bytes"}
    if type(memory) is not dict or set(memory) != expected_memory_keys:
        raise QualificationError("native named-memory accounting inventory mismatch")
    if any(type(memory.get(key)) is not int or memory[key] < 0 for key in expected_memory_keys):
        raise QualificationError("native named-memory byte count is malformed")
    aggregate = sum(memory[key] for key in parts)
    if (memory["aggregate_named_bytes"] != aggregate
            or memory["peak_named_bytes"] != aggregate + memory["setup_transient_attention_bytes"]
            or memory["peak_named_bytes"] > MAX_NAMED_BYTES):
        raise QualificationError("native named-memory accounting exceeds or misstates the 21 GiB bound")
    claimed = document.get("exact_parent_and_AA")
    expected_claims = {f"{kind}.{checkpoint}" for kind in ("parent", "AA") for checkpoint in CHECKPOINTS}
    if type(claimed) is not dict or set(claimed) != expected_claims:
        raise QualificationError("native parent/A-A claim inventory mismatch")
    for name, record in claimed.items():
        if (type(record) is not dict or set(record) != {"failures", "exact_bit_differences", "max_abs"}
                or type(record.get("failures")) is not int or record["failures"] != 0
                or type(record.get("exact_bit_differences")) is not int or record["exact_bit_differences"] != 0
                or type(record.get("max_abs")) not in (int, float) or record["max_abs"] != 0):
            raise QualificationError(f"native reported parent/A-A claim failed: {name}")


def _load_artifact(path: str | Path, expected_sha256: str, kind: str, parent: dict[str, Any],
                   bias: dict[str, Any], bias_sha256: str) -> Artifact:
    schema = SOURCE_SCHEMA if kind == "source" else NATIVE_SCHEMA
    canonical, actual_sha, document = _read_pinned_json(path, expected_sha256, f"{kind} report")
    if document.get("schema") != schema:
        raise QualificationError(f"{kind} report schema mismatch")
    if kind == "source":
        _validate_source_bindings(document, parent, bias)
        record_fields = ("dtype", "source_dtype", "shape", "offset_bytes", "nbytes", "sha256", "checkpoint_boundary")
        encoding = "f32le_bf16_expansion"
    else:
        _validate_native_bindings(document, bias_sha256)
        record_fields = ("shape", "dtype", "source_dtype", "offset_bytes", "nbytes", "sha256")
        encoding = "little-endian F32 lossless BF16 expansion"
    payload = document.get("payload")
    if (type(payload) is not dict or set(payload) != {"basename", "bytes", "sha256", "encoding"}
            or type(payload.get("basename")) is not str or Path(payload["basename"]).name != payload["basename"]
            or payload["basename"] in ("", ".", "..") or type(payload.get("bytes")) is not int
            or payload.get("encoding") != encoding):
        raise QualificationError(f"{kind} payload binding malformed")
    payload_sha = _digest(payload.get("sha256"), f"{kind} payload SHA-256")
    payload_path = _canonical_regular(canonical.parent / payload["basename"], f"{kind} tensor payload")
    if payload_path.parent != canonical.parent:
        raise QualificationError(f"{kind} payload escaped manifest directory")
    if payload_path.stat().st_size != payload["bytes"] or sha256_file(payload_path) != payload_sha:
        raise QualificationError(f"{kind} payload size/digest mismatch")
    arrays = validate_tensor_payload(
        payload_path,
        document.get("tensors"),
        TENSOR_NAMES,
        {name: OUTPUT_SHAPES[name.split(".", 1)[1]] for name in TENSOR_NAMES},
        record_fields=record_fields,
        label=kind,
    )
    # Recheck manifest/payload identities after the potentially long tensor scan.
    if sha256_file(canonical) != actual_sha or sha256_file(payload_path) != payload_sha:
        raise QualificationError(f"{kind} report or payload changed during qualification")
    return Artifact(kind, canonical, actual_sha, document, payload_path, arrays)


def _exact_bits_equal(left: np.ndarray, right: np.ndarray, label: str) -> bool:
    if left.shape != right.shape:
        raise QualificationError(f"exact comparison shape mismatch: {label}")
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def _require_exact_parent_and_aa(artifact: Artifact, parent_arrays: Mapping[str, np.memmap]) -> dict[str, Any]:
    results: dict[str, Any] = {"parent_zero": {}, "parent_zero_repeat": {}, "zero_AA": {}}
    for checkpoint in CHECKPOINTS:
        expected = parent_arrays[f"M11.{checkpoint}"]
        zero = artifact.arrays[f"zero.{checkpoint}"]
        repeat = artifact.arrays[f"zero_repeat.{checkpoint}"]
        results["parent_zero"][checkpoint] = _exact_bits_equal(expected, zero, f"{artifact.kind}.zero.{checkpoint}")
        results["parent_zero_repeat"][checkpoint] = _exact_bits_equal(expected, repeat, f"{artifact.kind}.zero_repeat.{checkpoint}")
        results["zero_AA"][checkpoint] = _exact_bits_equal(zero, repeat, f"{artifact.kind}.zero_AA.{checkpoint}")
    if not all(value for group in results.values() for value in group.values()):
        raise QualificationError(f"{artifact.kind} zero/parent/A-A is not exact at all five checkpoints")
    return results


def _check_protected_outputs(artifact: Artifact, vectors: Mapping[str, np.ndarray]) -> dict[str, Any]:
    block0_image: dict[str, Any] = {}
    zero_image = artifact.arrays["zero.block0.image"]
    zero_text = artifact.arrays["zero.block0.text_valid"]
    for case in CASE_ORDER:
        image = artifact.arrays[f"{case}.block0.image"]
        bias = vectors[case][:VIDEO_ROWS]
        rows = np.flatnonzero(bias == 0.0)
        same = _exact_bits_equal(image[:, rows, :], zero_image[:, rows, :], f"{artifact.kind}.{case}.protected-image")
        if not same:
            raise QualificationError(f"{artifact.kind} block0 image changed on a zero-bias row: {case}")
        block0_image[case] = {"protected_zero_bias_rows": int(rows.size), "exact": True}
        text = artifact.arrays[f"{case}.block0.text_valid"]
        if not _exact_bits_equal(text, zero_text, f"{artifact.kind}.{case}.block0-text"):
            raise QualificationError(f"{artifact.kind} block0 text changed under video-query bias: {case}")
    return {"block0_image_zero_bias_rows": block0_image, "block0_text_all_cases_exact": True}


def _region_mask(height: int, width: int, h_bounds: tuple[int, int], w_bounds: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[h_bounds[0] : h_bounds[1], w_bounds[0] : w_bounds[1]] = True
    return mask


def _region_stats(values: np.ndarray, baseline: np.ndarray, frames: tuple[int, int], mask: np.ndarray) -> dict[str, Any]:
    if values.ndim != 5 or baseline.shape != values.shape or mask.shape != values.shape[-2:]:
        raise QualificationError("velocity region inputs have incompatible shapes")
    start, stop = frames
    if start < 0 or stop <= start or stop > values.shape[2]:
        raise QualificationError("velocity region frame interval is invalid")
    actual_part = np.asarray(values[0, :, start:stop, :, :], dtype=np.float64)
    base_part = np.asarray(baseline[0, :, start:stop, :, :], dtype=np.float64)
    selected = np.broadcast_to(mask, actual_part.shape)
    actual_values = actual_part[selected]
    baseline_values = base_part[selected]
    delta = actual_values - baseline_values
    if not np.isfinite(delta).all() or not np.isfinite(baseline_values).all():
        raise QualificationError("non-finite value in regional velocity metric")
    inc_rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    inc_max = float(np.max(np.abs(delta), initial=0.0))
    inc_mean = float(np.mean(np.abs(delta))) if delta.size else 0.0
    baseline_rms = float(np.sqrt(np.mean(baseline_values * baseline_values))) if baseline_values.size else 0.0
    parity_allowance = ATOL + RTOL * np.abs(baseline_values)
    parity_allowance_rms = (
        float(np.sqrt(np.mean(parity_allowance * parity_allowance)))
        if parity_allowance.size else 0.0
    )
    return {
        "frames": [start, stop],
        "spatial_cells": int(np.count_nonzero(mask)),
        "elements": int(delta.size),
        "increment_rms": inc_rms,
        "increment_max_abs": inc_max,
        "increment_mean_abs": inc_mean,
        "increment_nonzero_elements": int(np.count_nonzero(delta)),
        "baseline_rms": baseline_rms,
        "baseline_normalized_rms": inc_rms / baseline_rms if baseline_rms > 0.0 else None,
        "parity_allowance_rms": parity_allowance_rms,
    }


def region_metrics(values: np.ndarray, baseline: np.ndarray, action: str) -> dict[str, Any]:
    """Measure fixed part, actor, other-actor and outside regions in one time window."""
    if action not in ACTION_WINDOWS:
        raise QualificationError(f"unknown action region: {action}")
    if tuple(values.shape) != OUTPUT_SHAPES["velocity"] or tuple(baseline.shape) != OUTPUT_SHAPES["velocity"]:
        raise QualificationError("regional velocity metric expects the fixed 1x32x30x9x16 tensor")
    spec = ACTION_WINDOWS[action]
    other = "A2" if action == "A1" else "A1"
    frame_start, frame_stop = spec["frames"]
    height, width = values.shape[-2:]
    part = _region_mask(height, width, *spec["part_hw"])
    actor = _region_mask(height, width, *spec["actor_hw"])
    other_actor = _region_mask(height, width, *ACTION_WINDOWS[other]["actor_hw"])
    other_part = _region_mask(height, width, *ACTION_WINDOWS[other]["part_hw"])
    union = _region_mask(height, width, *ACTION_WINDOWS["A1"]["actor_hw"]) | _region_mask(
        height, width, *ACTION_WINDOWS["A2"]["actor_hw"]
    )
    return {
        "part_proxy": _region_stats(values, baseline, (frame_start, frame_stop), part),
        "actor_roi": _region_stats(values, baseline, (frame_start, frame_stop), actor),
        "actor_complement": _region_stats(values, baseline, (frame_start, frame_stop), actor & ~part),
        "other_actor_roi": _region_stats(values, baseline, (frame_start, frame_stop), other_actor),
        "other_part_proxy": _region_stats(values, baseline, (frame_start, frame_stop), other_part),
        "outside_actor_union": _region_stats(values, baseline, (frame_start, frame_stop), ~union),
    }


def _localization_ratio(regions: Mapping[str, Any]) -> float | None:
    target = regions["part_proxy"]["increment_rms"]
    complement = regions["actor_complement"]["increment_rms"]
    if not math.isfinite(target) or not math.isfinite(complement) or complement <= 0.0:
        return None
    ratio = target / complement
    return ratio if math.isfinite(ratio) else None


def action_gate(action: str, measurements: Mapping[str, Mapping[str, Any]], *,
                artifact: str = "source") -> dict[str, Any]:
    """Gate parity-resolved effect, dose response and high-dose localization."""
    if action not in ACTION_WINDOWS:
        raise QualificationError(f"unknown action gate: {action}")
    if artifact not in ("source", "native"):
        raise QualificationError(f"unknown action artifact: {artifact}")
    low_case, high_case, sham_case = f"{action}_low", f"{action}_high", f"{action}_sham"
    try:
        low = measurements[low_case][action]
        high = measurements[high_case][action]
        sham = measurements[sham_case][action]
    except KeyError:
        raise QualificationError(f"missing action measurement arm: {action}") from None
    low_target = low["part_proxy"]["increment_rms"]
    high_target = high["part_proxy"]["increment_rms"]
    sham_target = sham["part_proxy"]["increment_rms"]
    threshold = high["part_proxy"]["parity_allowance_rms"]
    high_ratio = _localization_ratio(high)
    sham_ratio = _localization_ratio(sham)
    checks = {
        "targeted_high_change_nonzero": high_target > 0.0,
        "high_target_effect_exceeds_parity_allowance": high_target > threshold,
        "dose_response_low_not_greater_than_high": low_target <= high_target,
        "high_target_rms_exceeds_sham": high_target > sham_target,
        "target_and_complement_denominators_defined": high_ratio is not None and sham_ratio is not None,
        "high_target_to_complement_ratio_exceeds_sham": (
            high_ratio is not None and sham_ratio is not None and high_ratio > sham_ratio
        ),
    }
    return {
        "action": action,
        "artifact": artifact,
        "low_target_rms": low_target,
        "high_target_rms": high_target,
        "sham_target_rms": sham_target,
        "target_change_threshold_rms": threshold,
        "target_change_threshold_rationale": (
            "RMS of the fixed elementwise comparison allowances ATOL + RTOL*abs(own zero baseline) over "
            "the target part region, computed in float64. The high-dose target increment must strictly exceed "
            "this parity-scale floor. This establishes numerical sensitivity only, not visible acting or "
            "statistical significance."
        ),
        "high_target_to_complement_ratio": high_ratio,
        "sham_target_to_complement_ratio": sham_ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _parity_report(source: Artifact, native: Artifact) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in TENSOR_NAMES:
        results[name] = compare_arrays(source.arrays[name], native.arrays[name])
    failures = {name: row for name, row in results.items() if row["outside_atol_rtol"] != 0}
    return {"checkpoint_count": len(results), "all_within_tolerance": not failures,
            "atol": ATOL, "rtol": RTOL, "results": results, "failures": failures}


def _build_measurements(source: Artifact) -> dict[str, Any]:
    baseline = source.arrays["zero.velocity"]
    result: dict[str, Any] = {}
    for case in POSITIVE_CASES:
        values = source.arrays[f"{case}.velocity"]
        result[case] = {action: region_metrics(values, baseline, action) for action in ("A1", "A2")}
    return result


def _build_action_gate_reports(measurements_by_artifact: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Retain the v1 source gate names and add independent native gate results."""
    if set(measurements_by_artifact) != {"source", "native"}:
        raise QualificationError("action measurements must include source and native artifacts")
    gates: dict[str, Any] = {}
    for artifact in ("source", "native"):
        for action in ("A1", "A2"):
            gate = action_gate(action, measurements_by_artifact[artifact], artifact=artifact)
            key = (f"{action}_soft_dose_localization" if artifact == "source"
                   else f"{action}_native_soft_dose_localization")
            gates[key] = gate
    return gates


def _failed_numerical_gates(gates: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for artifact in ("source", "native"):
        for action in ("A1", "A2"):
            name = (f"{action}_soft_dose_localization" if artifact == "source"
                    else f"{action}_native_soft_dose_localization")
            gate = gates[name]
            if not gate["passed"]:
                failures.append({
                    "name": name,
                    "failed_checks": [check for check, passed in gate["checks"].items() if not passed],
                })
    return failures


def _check_artifact_bindings(source: Artifact, native: Artifact, parent_doc: dict[str, Any], bias_doc: dict[str, Any]) -> None:
    if source.document["parent_reference"]["sha256"] != PARENT_SHA256:
        raise QualificationError("source artifact no longer binds pinned parent")
    if native.document["parent_sha256"] != PARENT_SHA256:
        raise QualificationError("native artifact no longer binds pinned parent")
    if source.document["bias_manifest"]["sha256"] != BIAS_SHA256 or native.document["bias_sha256"] != BIAS_SHA256:
        raise QualificationError("source/native artifacts do not bind pinned bias manifest")
    if parent_doc["payload"]["sha256"] != PARENT_PAYLOAD_SHA256 or bias_doc["schema"] != BIAS_SCHEMA:
        raise QualificationError("pinned reference inputs changed")


def qualify(*, source_path: str | Path, source_sha256: str, native_path: str | Path,
            native_sha256: str, parent_path: str | Path, bias_path: str | Path) -> dict[str, Any]:
    """Load and independently qualify one source/native pair."""
    parent_manifest, parent_sha, parent_doc, parent_payload, parent_arrays = _load_parent(parent_path)
    bias_manifest, bias_doc, vectors = _load_bias(bias_path)
    source = _load_artifact(source_path, source_sha256, "source", parent_doc, bias_doc, BIAS_SHA256)
    native = _load_artifact(native_path, native_sha256, "native", parent_doc, bias_doc, BIAS_SHA256)
    _check_artifact_bindings(source, native, parent_doc, bias_doc)
    source_replay = _require_exact_parent_and_aa(source, parent_arrays)
    native_replay = _require_exact_parent_and_aa(native, parent_arrays)
    source_protected = _check_protected_outputs(source, vectors)
    native_protected = _check_protected_outputs(native, vectors)
    parity = _parity_report(source, native)
    if not parity["all_within_tolerance"]:
        raise QualificationError(f"native/source parity failed at {len(parity['failures'])} of 40 checkpoints")
    source_measurements = _build_measurements(source)
    native_measurements = _build_measurements(native)
    gates = _build_action_gate_reports({"source": source_measurements, "native": native_measurements})
    failed_gates = _failed_numerical_gates(gates)
    internal_checks_passed = not failed_gates
    # These are report-side claims, not substitutes for the raw-byte checks above.
    # External process-footprint, cache-floor and swap guard receipts are supplied
    # and checked by the orchestration layer, outside this tensor qualifier.
    return {
        "schema": "hy-nomas-soft-route-qualification-v1",
        "source": {"path": str(source.manifest_path), "sha256": source.manifest_sha256},
        "native": {"path": str(native.manifest_path), "sha256": native.manifest_sha256},
        "parent": {"path": str(parent_manifest), "sha256": parent_sha,
                   "payload_sha256": sha256_file(parent_payload)},
        "bias": {"path": str(bias_manifest), "sha256": BIAS_SHA256},
        "gates": {
            "raw_parent_zero_and_AA_exact_both_artifacts": True,
            "protected_block0_rows_and_text": True,
            "all_40_source_native_tensors_within_tolerance": True,
            "named_memory_plan_under_21_GiB": True,
            "source_runtime_receipt_and_native_encode_counters": True,
            "external_process_memory_swap_receipts": "required separately; not inspected here",
            "A1_soft_dose_localization": gates["A1_soft_dose_localization"],
            "A1_native_soft_dose_localization": gates["A1_native_soft_dose_localization"],
            "A2_soft_dose_localization": gates["A2_soft_dose_localization"],
            "A2_native_soft_dose_localization": gates["A2_native_soft_dose_localization"],
        },
        "qualification_status": "passed" if internal_checks_passed else "failed_numerical_gate",
        "qualified": internal_checks_passed,
        "failed_gates": failed_gates,
        "source_exact_replay": source_replay,
        "native_exact_replay": native_replay,
        "source_protected_outputs": source_protected,
        "native_protected_outputs": native_protected,
        "source_native_parity": parity,
        "velocity_increment_metrics": {
            "reference": "each artifact; each arm minus that artifact's own zero, BF16 values expanded to F32",
            "normalization": "increment RMS divided by that artifact's zero RMS in the same fixed part/actor region; null when baseline RMS is zero",
            "measurement_artifacts": {"measurements": "source", "native_measurements": "native"},
            "part_boxes": {
                "A1_head_proxy": {"frames": [1, 10], "height": [0, 4], "width": [4, 8]},
                "A2_glove_chest_proxy": {"frames": [12, 28], "height": [4, 6], "width": [11, 13]},
            },
            "actor_rois": {
                "A1": {"height": [0, 9], "width": [1, 8]},
                "A2": {"height": [2, 8], "width": [10, 15]},
            },
            "effect_threshold": {
                "formula": "RMS(ATOL + RTOL * abs(own zero baseline)) over target-region elements",
                "atol": ATOL,
                "rtol": RTOL,
                "arithmetic": "float64",
            },
            "measurements": source_measurements,
            "native_measurements": native_measurements,
            "interpretation_limit": "Resolved numerical locality/sensitivity only; not visible acting, perceptual significance, or statistical significance. These fixed boxes are authored proxies, not segmentation, pose or contact labels.",
        },
        "qualification_scope": "Frozen-input saved-state DiT source/native numerical sensitivity and locality only; this does not establish visible acting or statistical significance. External footprint receipts are separate.",
        "internal_checks_passed": internal_checks_passed,
    }


def _write_report(path: str | Path, report: dict[str, Any]) -> None:
    candidate = Path(path).expanduser()
    parent = candidate.parent.resolve(strict=True)
    if candidate.is_symlink() or os.path.lexists(candidate):
        raise QualificationError(f"refusing to replace report: {candidate}")
    target = parent / candidate.name
    if not target.is_absolute():
        raise QualificationError("qualification report path must resolve absolutely")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--parent", default="/opt/mldata_ssd/cogni-ml/hy-nomas-direct-routes-20261105.klzuhg0r/source-qualified.json", type=Path)
    parser.add_argument("--bias", default="/opt/mldata_ssd/cogni-ml/hy-nomas-soft-routes-20261105.j0wgy7yl/bias-v2.json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = qualify(
            source_path=args.source, source_sha256=args.source_sha256,
            native_path=args.native, native_sha256=args.native_sha256,
            parent_path=args.parent, bias_path=args.bias,
        )
        encoded = json.dumps(report, sort_keys=True, allow_nan=False)
        if args.report:
            _write_report(args.report, report)
            print(json.dumps({
                "report": str(args.report.resolve()),
                "qualification_status": report["qualification_status"],
                "internal_checks_passed": report["internal_checks_passed"],
                "failed_gates": report["failed_gates"],
            }, sort_keys=True))
        else:
            print(encoded)
        return 0 if report["internal_checks_passed"] else 2
    except (QualificationError, OSError, ValueError, TypeError) as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
