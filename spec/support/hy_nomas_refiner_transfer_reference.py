#!/usr/bin/env python3
"""Paired source-BF16 Hy-Nomas text-refiner reference, limited to V224.

The runner authenticates two saved Qwen crops, checks the unchanged
global/Action1 rows before padding normalization, then runs A0/A1/B through
only the pinned timestep embedder and two-block text refiner. Source execution
is MPS-only and uses the already-qualified MPS SDPA adapter. CPU admission,
payload validation, and comparison helpers never import torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

SCHEMA = "hy-nomas-refiner-transfer-reference-v1"
SCHEMA_VERSION = 1
INPUT_SCHEMA = "hy-nomas-refiner-transfer-input-v1"
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
VALID_ROWS = 224
PREFIX_END = 132
ACTION2_START = 132
ACTION2_END = 224
TOKEN_COUNT = 1280
HIDDEN_SIZE = 3584
REFINER_DIM = 2048
HEADS = 16
HEAD_DIM = 128
BLOCKS = 2
STATE_INDEX = 26
CROP_START = 108
RAW_SEQUENCE_LENGTH = 1388
QWEN_SOURCE_SHAPE = (27, 1, RAW_SEQUENCE_LENGTH, HIDDEN_SIZE)
QWEN_SHAPE = (1, TOKEN_COUNT, HIDDEN_SIZE)
INPUT_BYTES = math.prod(QWEN_SHAPE) * 2
QWEN_SOURCE_BYTES = math.prod(QWEN_SOURCE_SHAPE) * 2
ATOL = 2e-5
RTOL = 2e-5
CASE_ORDER = ("A0", "A1", "B")
CHECKPOINTS = ("masked_mean", "c", "vec", "block0.output", "block1.output")
GEOMETRY = {
    "batch_size": 1,
    "valid_rows": VALID_ROWS,
    "pad_rows": TOKEN_COUNT - VALID_ROWS,
    "hidden_size": HIDDEN_SIZE,
    "refiner_dim": REFINER_DIM,
    "heads": HEADS,
    "head_dim": HEAD_DIM,
    "blocks": BLOCKS,
    "qwen_shape": list(QWEN_SHAPE),
}
ACTION_SPANS = {
    "global": [0, 49],
    "action1": [49, 132],
    "action2": [132, 224],
    "padding": [224, 1280],
}
_MASK_BYTES = bytes([1]) * VALID_ROWS + bytes(TOKEN_COUNT - VALID_ROWS)
MASK_RECORD = {
    "dtype": "BOOL",
    "shape": [1, TOKEN_COUNT],
    "valid_true_prefix": VALID_ROWS,
    "pad_false_suffix": TOKEN_COUNT - VALID_ROWS,
    "sha256": hashlib.sha256(_MASK_BYTES).hexdigest(),
    "encoding": "one uint8 byte per bool; true=1,false=0",
}
NORMALIZATION = {
    "prefix_validation": "raw_A_B_prefix_bitmatch_before_normalization",
    "padding": "zero_bf16_rows_after_validation",
    "valid_rows": VALID_ROWS,
    "pad_start": VALID_ROWS,
    "pad_stop": TOKEN_COUNT,
}
_CONTROL_NAMES = (
    "prelude_input_bf16_control",
    "prelude_linear_bf16_control",
    "prelude_silu_bf16_control",
    "prelude_second_linear_bf16_control",
    "prelude_add_bf16_control",
    "block_boundaries_bf16_control",
    "qkv_bf16_control",
    "source_attention_control",
    "source_projection_residual_control",
    "source_norm2_control",
    "source_fc1_control",
    "source_mlp_silu_control",
    "source_fc2_residual_control",
    "source_mean_autocast",
)
PRECISION = {
    "resident_weights": "BF16",
    "compute": "F32",
    "attention": "F32Control",
    "source_bf16_autocast": True,
    "controls": {name: True for name in _CONTROL_NAMES},
    "atol": ATOL,
    "rtol": RTOL,
}
EXECUTION = {
    "timestep": 1000.0,
    "case_order": list(CASE_ORDER),
    "scope": "refiner only; no full DiT",
}
SOURCE_PRECISION = {
    "resident_weights": "BF16 converted from authenticated F32 subset",
    "autocast_device": "mps",
    "autocast_dtype": "BF16",
    "attention_adapter": "qualified hy_nomas_real_attention_reference._install_refiner_attention",
    "attention_adapter_mode": "bf16",
    "query_chunk_tokens": 256,
    "masked_mean_checkpoint": "F32 source masked mean cast to BF16 at c_embedder Linear input",
    "c_checkpoint": "forward_pre_hook of individual_token_refiner input[1]",
    "vec_checkpoint": "time_in output",
    "block_output_rows": "valid prefix [0,224) only",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_HASH_CHUNK = 8 * 1024 * 1024
_REFERENCE_KEYS = {
    "schema", "schema_version", "upstream_revision", "source", "geometry",
    "timestep", "mask", "action_spans", "inputs", "normalization",
    "precision", "execution", "payload", "tensors",
}
_INPUT_CASE_KEYS = {
    "manifest_path", "manifest_sha256", "payload_path", "payload_bytes",
    "payload_sha256", "tensor_name", "offset_bytes", "nbytes",
    "source_tensor_sha256", "source_shape", "source_dtype",
    "normalized_tensor_sha256",
}

from hy_nomas_text_conditioning_reference import PINNED_SOURCE_SHA256


class ReferenceError(RuntimeError):
    """Fail-closed source, input, output, or comparator error."""


def _json_no_duplicates(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReferenceError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except ReferenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError(f"invalid {label} JSON") from None
    if type(value) is not dict:
        raise ReferenceError(f"{label} root must be an object")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ReferenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(payload: bytes, expected: str, label: str) -> str:
    _digest(expected, f"{label} SHA-256")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ReferenceError(f"{label} digest mismatch")
    return actual


def _regular_file(path_arg: str | Path, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ReferenceError(f"{label} does not exist: {path}") from None
    if path.is_symlink() or not resolved.is_file():
        raise ReferenceError(f"{label} must be a canonical regular file: {path}")
    return resolved


def _read_json(path: Path, label: str, *, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ReferenceError(f"{label} exceeds the bounded manifest size")
    return _json_no_duplicates(path.read_bytes(), label)


def _same_int_list(actual: Any, expected: list[int], label: str) -> None:
    if (type(actual) is not list or len(actual) != len(expected)
            or any(type(value) is not int for value in actual)
            or actual != expected):
        raise ReferenceError(f"{label} geometry mismatch")


def _bf16_finite(payload: bytes, label: str) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for BF16 CPU admission: {error}") from error
    if len(payload) % 2:
        raise ReferenceError(f"{label} BF16 byte extent is not word aligned")
    words = np.frombuffer(payload, dtype="<u2")
    if bool(np.any((words & np.uint16(0x7F80)) == np.uint16(0x7F80))):
        raise ReferenceError(f"{label} contains nonfinite BF16 values")


def prepare_pair(input_a: bytes, input_b: bytes) -> dict[str, bytes]:
    """Validate raw Qwen pair, then return the 0-padded canonical A0/A1/B bytes."""
    if len(input_a) != INPUT_BYTES or len(input_b) != INPUT_BYTES:
        raise ReferenceError(f"input tensor extent must be exactly {INPUT_BYTES} bytes")
    _bf16_finite(input_a, "A")
    _bf16_finite(input_b, "B")
    row_bytes = HIDDEN_SIZE * 2
    prefix_stop = PREFIX_END * row_bytes
    if input_a[:prefix_stop] != input_b[:prefix_stop]:
        raise ReferenceError("raw A/B unchanged prefix [0,132) differs")
    action2_start = ACTION2_START * row_bytes
    action2_stop = ACTION2_END * row_bytes
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for pair admission: {error}") from error
    changed_bytes = int(np.count_nonzero(
        np.frombuffer(input_a[action2_start:action2_stop], dtype=np.uint8)
        != np.frombuffer(input_b[action2_start:action2_stop], dtype=np.uint8)
    ))
    if changed_bytes == 0:
        raise ReferenceError("Action2 intervention is byte-identical to A")
    valid_bytes = VALID_ROWS * row_bytes
    padding = bytes(INPUT_BYTES - valid_bytes)
    normalized_a = input_a[:valid_bytes] + padding
    normalized_b = input_b[:valid_bytes] + padding
    return {"A0": normalized_a, "A1": normalized_a, "B": normalized_b}


def compare_checkpoint(
    name: str, expected: Any, actual: Any, *, atol: float = ATOL, rtol: float = RTOL,
) -> dict[str, Any]:
    """Return finite-only allclose statistics; tolerance failures are reported."""
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for checkpoint comparison: {error}") from error
    lhs = np.asarray(expected, dtype=np.float32)
    rhs = np.asarray(actual, dtype=np.float32)
    if lhs.shape != rhs.shape:
        raise ReferenceError(f"{name} checkpoint shape mismatch: {lhs.shape} != {rhs.shape}")
    if not bool(np.isfinite(lhs).all()) or not bool(np.isfinite(rhs).all()):
        raise ReferenceError(f"{name} checkpoint contains nonfinite values")
    delta = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    limit = float(atol) + float(rtol) * np.abs(lhs.astype(np.float64))
    failures = int(np.count_nonzero(delta > limit))
    return {
        "checkpoint": name,
        "elements": int(lhs.size),
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "outside_atol_rtol": failures,
        "all_finite": True,
        "passed": failures == 0,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def assert_checkpoint_close(name: str, expected: Any, actual: Any,
                            *, atol: float = ATOL, rtol: float = RTOL) -> dict[str, Any]:
    result = compare_checkpoint(name, expected, actual, atol=atol, rtol=rtol)
    if not result["passed"]:
        raise ReferenceError(f"{name} checkpoint has values outside atol/rtol")
    return result


def _output_shape(case_name: str, checkpoint: str) -> list[int]:
    if checkpoint == "masked_mean":
        return [1, HIDDEN_SIZE]
    if checkpoint in ("c", "vec"):
        return [1, REFINER_DIM]
    if checkpoint in ("block0.output", "block1.output"):
        return [1, VALID_ROWS, REFINER_DIM]
    raise ReferenceError(f"unknown output checkpoint: {case_name}.{checkpoint}")


def _output_nbytes(shape: list[int]) -> int:
    return math.prod(shape) * 4


def _output_elements(checkpoint: str) -> int:
    return math.prod(_output_shape("A0", checkpoint))


def _validate_source(source: Any) -> None:
    expected = {
        "upstream_path", "revision", "source_files", "text_weights",
        "source_checkpoint", "module_bindings",
    }
    if type(source) is not dict or set(source) != expected:
        raise ReferenceError("source binding inventory mismatch")
    if source.get("revision") != PINNED_REVISION:
        raise ReferenceError("source revision mismatch")
    if (type(source.get("upstream_path")) is not str
            or not Path(source["upstream_path"]).is_absolute()):
        raise ReferenceError("upstream source path must be absolute")
    source_files = source.get("source_files")
    if source_files != PINNED_SOURCE_SHA256:
        raise ReferenceError("pinned upstream source file hashes mismatch")
    weights = source.get("text_weights")
    weight_keys = {
        "path", "sha256", "bytes", "manifest_path", "manifest_sha256",
        "tensor_count", "stored_dtype", "loaded_dtype", "conversion",
    }
    if type(weights) is not dict or set(weights) != weight_keys:
        raise ReferenceError("source text-weight inventory mismatch")
    _digest(weights.get("sha256"), "text weights SHA-256")
    _digest(weights.get("manifest_sha256"), "text weights manifest SHA-256")
    if (type(weights.get("bytes")) is not int or weights["bytes"] <= 0
            or weights.get("tensor_count") != 42
            or weights.get("stored_dtype") != "F32"
            or weights.get("loaded_dtype") != "BF16"
            or weights.get("conversion") != "F32_to_BF16_RNE"):
        raise ReferenceError("source text-weight precision binding mismatch")
    if (type(weights.get("path")) is not str
            or type(weights.get("manifest_path")) is not str):
        raise ReferenceError("source weight paths must be strings")
    checkpoint = source.get("source_checkpoint")
    checkpoint_expected = {
        "repository": "tencent/Hy-Nomas",
        "checkpoint_path": "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
        "checkpoint_lfs_sha256": "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
        "checkpoint_lfs_sha256_status": "declared by pinned HF API; whole file not downloaded or verified",
        "source_snapshot_revision": "bac719a80c8d02f91c485fca804bb8221e149b70",
    }
    if checkpoint != checkpoint_expected:
        raise ReferenceError("source checkpoint provenance mismatch")
    modules = source.get("module_bindings")
    if modules != {
        "refiner": "hymm.models.text_encoder.token_refiner.SingleTokenRefinerPerBlock",
        "time_embedder": "hymm.models.dit.modules.embed_layers.TimestepEmbedder",
    }:
        raise ReferenceError("source module bindings mismatch")


def validate_reference_document(document: Mapping[str, Any]) -> None:
    """Validate the exact source reference contract without importing torch."""
    if type(document) is not dict or set(document) != _REFERENCE_KEYS:
        raise ReferenceError("reference document keys mismatch")
    if (document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != SCHEMA_VERSION
            or document.get("upstream_revision") != PINNED_REVISION):
        raise ReferenceError("reference schema or pinned revision mismatch")
    _validate_source(document.get("source"))
    if document.get("geometry") != GEOMETRY:
        raise ReferenceError("reference geometry mismatch")
    if document.get("timestep") != 1000.0:
        raise ReferenceError("reference timestep mismatch")
    if document.get("mask") != MASK_RECORD:
        raise ReferenceError("reference mask mismatch")
    if document.get("action_spans") != ACTION_SPANS:
        raise ReferenceError("reference action spans mismatch")
    if document.get("normalization") != NORMALIZATION:
        raise ReferenceError("reference normalization contract mismatch")
    precision = document.get("precision")
    if type(precision) is not dict or precision != PRECISION:
        raise ReferenceError("reference precision contract mismatch")
    if type(document.get("execution")) is not dict:
        raise ReferenceError("reference execution contract malformed")
    execution = document["execution"]
    if (execution.get("timestep") != 1000.0
            or execution.get("case_order") != list(CASE_ORDER)
            or execution.get("scope") != "refiner only; no full DiT"
            or execution.get("source_precision") != SOURCE_PRECISION
            or execution.get("source_attention_query_chunks") != 2):
        raise ReferenceError("reference execution scope/adapter mismatch")
    floor = execution.get("source_a_a_floor")
    if type(floor) is not dict or set(floor) != set(CHECKPOINTS):
        raise ReferenceError("source A/A floor checkpoint inventory mismatch")
    floor_keys = {
        "checkpoint", "elements", "max_abs", "mean_abs", "rms_abs",
        "outside_atol_rtol", "all_finite", "passed", "atol", "rtol",
    }
    for checkpoint in CHECKPOINTS:
        result = floor[checkpoint]
        if type(result) is not dict or set(result) != floor_keys:
            raise ReferenceError(f"source A/A floor record mismatch: {checkpoint}")
        numeric_stats = ("max_abs", "mean_abs", "rms_abs")
        if any(type(result.get(key)) not in (int, float)
               or not math.isfinite(float(result[key]))
               or result[key] < 0 for key in numeric_stats):
            raise ReferenceError(f"source A/A floor statistics malformed: {checkpoint}")
        if (result.get("checkpoint") != f"A0/A1.{checkpoint}"
                or type(result.get("elements")) is not int
                or result["elements"] != _output_elements(checkpoint)
                or type(result.get("outside_atol_rtol")) is not int
                or result["outside_atol_rtol"] != 0
                or result.get("all_finite") is not True
                or result.get("passed") is not True
                or result.get("atol") != ATOL
                or result.get("rtol") != RTOL):
            raise ReferenceError(f"source A/A floor failed tolerance or shape: {checkpoint}")

    inputs = document.get("inputs")
    if type(inputs) is not dict or set(inputs) != {"cases", "prefix_checks"}:
        raise ReferenceError("reference input inventory mismatch")
    cases = inputs.get("cases")
    if type(cases) is not dict or set(cases) != set(CASE_ORDER):
        raise ReferenceError("reference input case inventory mismatch")
    for name in CASE_ORDER:
        case = cases[name]
        if type(case) is not dict or set(case) != _INPUT_CASE_KEYS:
            raise ReferenceError(f"reference input case record mismatch: {name}")
        for key in ("manifest_sha256", "payload_sha256", "source_tensor_sha256",
                    "normalized_tensor_sha256"):
            _digest(case.get(key), f"{name} {key}")
        for key in ("manifest_path", "payload_path"):
            value = case.get(key)
            if (type(value) is not str or value in ("", ".", "..")
                    or Path(value).name != value):
                raise ReferenceError(f"{name} input path must be a sibling basename")
        if (type(case.get("payload_bytes")) is not int
                or case["payload_bytes"] != INPUT_BYTES
                or case.get("tensor_name") != "input.qwen_bf16"
                or type(case.get("offset_bytes")) is not int
                or case["offset_bytes"] != 0
                or type(case.get("nbytes")) is not int or case["nbytes"] != INPUT_BYTES
                or case.get("source_shape") != list(QWEN_SHAPE)
                or case.get("source_dtype") != "BF16"):
            raise ReferenceError(f"reference input case geometry mismatch: {name}")
        if case["normalized_tensor_sha256"] != case["payload_sha256"]:
            raise ReferenceError(f"reference normalized input payload digest binding mismatch: {name}")
    a0, a1, case_b = (cases[name] for name in CASE_ORDER)
    for key in _INPUT_CASE_KEYS:
        if a0[key] != a1[key]:
            raise ReferenceError(f"A/A input binding differs: {key}")
    if a0["normalized_tensor_sha256"] == case_b["normalized_tensor_sha256"]:
        raise ReferenceError("normalized Action2 intervention disappeared")
    prefix = inputs.get("prefix_checks")
    prefix_expected = {
        "validated_before_padding_normalization": True,
        "unchanged_rows": [0, PREFIX_END],
        "changed_rows": [ACTION2_START, ACTION2_END],
    }
    if (type(prefix) is not dict
            or any(prefix.get(key) != value for key, value in prefix_expected.items())
            or type(prefix.get("action2_changed_bytes")) is not int
            or prefix["action2_changed_bytes"] <= 0):
        raise ReferenceError("reference prefix/intervention admission mismatch")
    _digest(prefix.get("global_action1_sha256"), "unchanged-prefix SHA-256")
    if prefix.get("A0_A1_same_normalized_qwen_sha256") != a0["normalized_tensor_sha256"]:
        raise ReferenceError("A/A normalized input SHA binding mismatch")

    payload = document.get("payload")
    if (type(payload) is not dict
            or set(payload) != {"basename", "bytes", "sha256", "encoding"}
            or type(payload.get("basename")) is not str
            or Path(payload["basename"]).name != payload["basename"]
            or payload.get("encoding") != "f32le_bf16_expansion"
            or type(payload.get("bytes")) is not int or payload["bytes"] <= 0):
        raise ReferenceError("reference output payload metadata mismatch")
    _digest(payload.get("sha256"), "reference output payload SHA-256")
    tensors = document.get("tensors")
    expected_names = {f"{case}.{checkpoint}" for case in CASE_ORDER for checkpoint in CHECKPOINTS}
    if type(tensors) is not dict or set(tensors) != expected_names:
        raise ReferenceError("reference output tensor inventory mismatch")
    cursor = 0
    for case in CASE_ORDER:
        for checkpoint in CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            record = tensors[name]
            shape = _output_shape(case, checkpoint)
            nbytes = _output_nbytes(shape)
            if (type(record) is not dict
                    or set(record) != {"dtype", "source_dtype", "shape", "offset_bytes",
                                       "nbytes", "sha256", "checkpoint_boundary"}
                    or record.get("dtype") != "F32"
                    or record.get("source_dtype") != "BF16"
                    or record.get("shape") != shape
                    or type(record.get("offset_bytes")) is not int
                    or record["offset_bytes"] != cursor
                    or type(record.get("nbytes")) is not int or record["nbytes"] != nbytes
                    or record.get("checkpoint_boundary") != checkpoint):
                raise ReferenceError(f"reference tensor layout mismatch: {name}")
            _digest(record.get("sha256"), f"{name} SHA-256")
            cursor += nbytes
    if payload["bytes"] != cursor:
        raise ReferenceError("reference output payload extent mismatch")


def validate_reference_payload(document: Mapping[str, Any], payload_path: str | Path) -> None:
    """Authenticate packed outputs and all checkpoint ranges using CPU only."""
    validate_reference_document(document)
    path = _regular_file(payload_path, "reference output payload")
    output = document["payload"]
    if path.name != output["basename"]:
        raise ReferenceError("reference output payload basename mismatch")
    if path.stat().st_size != output["bytes"] or sha256_file(path) != output["sha256"]:
        raise ReferenceError("reference output payload digest/extent mismatch")
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for output payload checks: {error}") from error
    with path.open("rb") as stream:
        for name, record in document["tensors"].items():
            offset = record["offset_bytes"]
            stream.seek(offset)
            data = stream.read(record["nbytes"])
            if len(data) != record["nbytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ReferenceError(f"output checkpoint range digest mismatch: {name}")
            values = np.frombuffer(data, dtype="<f4")
            bits = values.view("<u4")
            if (not bool(np.isfinite(values).all())
                    or bool(np.any(bits & np.uint32(0xFFFF)) )):
                raise ReferenceError(f"output checkpoint is not finite BF16-expanded F32: {name}")


def _sha256_range(path: Path, offset: int, size: int) -> str:
    if offset < 0 or size <= 0 or offset + size > path.stat().st_size:
        raise ReferenceError("source tensor range is outside its sidecar")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            chunk = stream.read(min(_HASH_CHUNK, remaining))
            if not chunk:
                raise ReferenceError("truncated source tensor range")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _verify_qwen_manifest(manifest_path: Path, expected_sha256: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    _digest(expected_sha256, "source Qwen manifest SHA-256")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ReferenceError("source Qwen manifest digest mismatch")
    document = _json_no_duplicates(raw, "source Qwen manifest")
    prompt = document.get("hy_nomas_prompt")
    capture = document.get("capture")
    hidden = capture.get("hidden_states") if type(capture) is dict else None
    sidecars = document.get("sidecars")
    sidecar = sidecars.get("bf16") if type(sidecars) is dict else None
    expected_shape = list(QWEN_SOURCE_SHAPE)
    if (document.get("schema") != "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference"
            or document.get("schema_version") != 1
            or type(prompt) is not dict
            or prompt.get("upstream_revision") != PINNED_REVISION
            or prompt.get("crop_start") != CROP_START
            or prompt.get("raw_length") != RAW_SEQUENCE_LENGTH
            or type(capture) is not dict
            or capture.get("captured_state_count") != STATE_INDEX + 1
            or type(hidden) is not dict
            or hidden.get("dtype") != "bfloat16"
            or hidden.get("shape") != expected_shape
            or type(hidden.get("storage")) is not dict
            or hidden["storage"].get("nbytes") != QWEN_SOURCE_BYTES
            or hidden["storage"].get("offset_bytes") != 0
            or type(sidecar) is not dict
            or sidecar.get("encoding") != "bfloat16-le"
            or sidecar.get("nbytes") != QWEN_SOURCE_BYTES):
        raise ReferenceError("source Qwen manifest geometry/schema mismatch")
    storage_sha = hidden["storage"].get("sha256")
    if storage_sha is not None and storage_sha != sidecar.get("sha256"):
        raise ReferenceError("source Qwen storage/sidecar digest binding mismatch")
    checkpoint = document.get("checkpoint")
    if (type(checkpoint) is not dict or checkpoint.get("repo_id") != "Qwen/Qwen2.5-VL-7B-Instruct"
            or checkpoint.get("revision") != "eb6ba55b40ada64977147b52e50a4145ee89ac4f"):
        raise ReferenceError("source Qwen model checkpoint binding mismatch")
    mask = prompt.get("cropped_attention_mask")
    if mask != [1] * VALID_ROWS + [0] * (TOKEN_COUNT - VALID_ROWS):
        raise ReferenceError("source Qwen cropped mask does not admit exact V224 prefix")
    sidecar_name = sidecar.get("path")
    if (type(sidecar_name) is not str or not sidecar_name
            or Path(sidecar_name).name != sidecar_name):
        raise ReferenceError("source Qwen sidecar must be a sibling basename")
    sidecar_path = _regular_file(manifest_path.parent / sidecar_name, "source Qwen sidecar")
    if (sidecar_path.stat().st_size != QWEN_SOURCE_BYTES
            or sha256_file(sidecar_path) != sidecar.get("sha256")):
        raise ReferenceError("source Qwen sidecar digest/extent mismatch")
    _digest(sidecar.get("sha256"), "source Qwen sidecar SHA-256")
    return document, sidecar_path, sidecar


def _extract_source_qwen_crop(sidecar: Path) -> bytes:
    row_bytes = HIDDEN_SIZE * 2
    offset = (STATE_INDEX * RAW_SEQUENCE_LENGTH + CROP_START) * row_bytes
    size = TOKEN_COUNT * row_bytes
    with sidecar.open("rb") as stream:
        stream.seek(offset)
        result = stream.read(size)
    if len(result) != size:
        raise ReferenceError("source Qwen selected state/crop is truncated")
    _bf16_finite(result, "source Qwen selected crop")
    return result


def _authenticate_generation_binding(origin: Mapping[str, Any], raw_crop: bytes) -> dict[str, Any] | None:
    manifest_arg = origin.get("source_generation_manifest")
    manifest_sha = origin.get("source_generation_manifest_sha256")
    if manifest_arg is None and manifest_sha is None:
        return None
    if type(manifest_arg) is not str:
        raise ReferenceError("accepted generation manifest path is malformed")
    manifest_path = _regular_file(manifest_arg, "accepted generation manifest")
    _digest(manifest_sha, "accepted generation manifest SHA-256")
    manifest_raw = manifest_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha:
        raise ReferenceError("accepted generation manifest digest mismatch")
    try:
        import hy_nomas_generation_inputs as generation_inputs
    except ImportError as error:
        raise ReferenceError(f"generation input validator unavailable: {error}") from error
    document = _json_no_duplicates(manifest_raw, "accepted generation manifest")
    payload_path = _regular_file(
        manifest_path.parent / document.get("payload", ""),
        "accepted generation payload",
    )
    generation_inputs.validate_document(document, payload_bytes=payload_path.stat().st_size)
    if (payload_path.stat().st_size != document["payload_bytes"]
            or sha256_file(payload_path) != document["payload_sha256"]):
        raise ReferenceError("accepted generation payload digest/extent mismatch")
    record = document["tensors"].get("input.qwen_bf16")
    if (type(record) is not dict or record.get("dtype") != "BF16"
            or record.get("shape") != list(QWEN_SHAPE)
            or record.get("nbytes") != INPUT_BYTES):
        raise ReferenceError("accepted generation Qwen tensor geometry mismatch")
    if _sha256_range(payload_path, record["offset"], record["nbytes"]) != record["sha256"]:
        raise ReferenceError("accepted generation Qwen tensor range digest mismatch")
    with payload_path.open("rb") as stream:
        stream.seek(record["offset"])
        generation_crop = stream.read(record["nbytes"])
    if generation_crop != raw_crop or record["sha256"] != origin.get("raw_qwen_tensor_sha256"):
        raise ReferenceError("accepted generation Qwen bytes differ from source-Qwen crop")
    if (origin.get("source_payload_sha256") != document["payload_sha256"]):
        raise ReferenceError("accepted generation payload digest binding mismatch")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "payload_basename": document["payload"],
        "payload_bytes": document["payload_bytes"],
        "payload_sha256": document["payload_sha256"],
        "tensor_name": "input.qwen_bf16",
        "tensor_dtype": record["dtype"],
        "tensor_shape": record["shape"],
        "tensor_offset": record["offset"],
        "tensor_nbytes": record["nbytes"],
        "tensor_sha256": record["sha256"],
    }


def authenticate_case_input(input_arg: str | Path, origin_arg: str | Path,
                            label: str) -> dict[str, Any]:
    """Bind a normalized compact input to its full Qwen sidecar and source range."""
    input_path = _regular_file(input_arg, f"{label} normalized BF16 input")
    origin_path = _regular_file(origin_arg, f"{label} source origin manifest")
    origin = _read_json(origin_path, f"{label} source origin manifest")
    if origin.get("schema") != "hy-nomas-refiner-transfer-input-origin-v1":
        raise ReferenceError(f"{label} input origin schema mismatch")
    origin_input_arg = origin.get("path")
    if type(origin_input_arg) is not str or _regular_file(
            origin_input_arg, f"{label} origin-bound input") != input_path:
        raise ReferenceError(f"{label} source origin path binding mismatch")
    if origin.get("crop") != {
        "state": STATE_INDEX, "start": CROP_START, "tokens": TOKEN_COUNT,
        "hidden": HIDDEN_SIZE,
    }:
        raise ReferenceError(f"{label} source crop geometry mismatch")
    if (origin.get("normalization") != "zero masked padding rows [224,1280); preserve all valid source BF16 words"
            or type(origin.get("pad_was_already_zero")) is not bool
            or origin.get("nbytes") != INPUT_BYTES):
        raise ReferenceError(f"{label} source normalization record mismatch")
    _digest(origin.get("sha256"), f"{label} normalized input SHA-256")
    if (input_path.stat().st_size != INPUT_BYTES
            or sha256_file(input_path) != origin["sha256"]):
        raise ReferenceError(f"{label} normalized input digest/extent mismatch")
    _digest(origin.get("raw_qwen_tensor_sha256"), f"{label} raw Qwen crop SHA-256")
    qwen_manifest_arg = origin.get("source_qwen_manifest")
    qwen_manifest_sha = origin.get("source_qwen_manifest_sha256")
    if (type(qwen_manifest_arg) is not str
            or not Path(qwen_manifest_arg).is_absolute()):
        raise ReferenceError(f"{label} source Qwen manifest path missing")
    qwen_manifest_path = _regular_file(qwen_manifest_arg, f"{label} source Qwen manifest")
    qwen_doc, qwen_sidecar, qwen_sidecar_record = _verify_qwen_manifest(
        qwen_manifest_path, qwen_manifest_sha,
    )
    if origin.get("source_qwen_sidecar_sha256") != qwen_sidecar_record["sha256"]:
        raise ReferenceError(f"{label} origin does not bind the full source Qwen sidecar")
    raw_crop = _extract_source_qwen_crop(qwen_sidecar)
    raw_crop_sha = hashlib.sha256(raw_crop).hexdigest()
    if raw_crop_sha != origin["raw_qwen_tensor_sha256"]:
        raise ReferenceError(f"{label} origin does not match the selected source Qwen crop")
    row_bytes = HIDDEN_SIZE * 2
    normalized = input_path.read_bytes()
    valid_bytes = VALID_ROWS * row_bytes
    if normalized[:valid_bytes] != raw_crop[:valid_bytes]:
        raise ReferenceError(f"{label} normalized valid rows differ from source Qwen")
    if normalized[valid_bytes:] != bytes(INPUT_BYTES - valid_bytes):
        raise ReferenceError(f"{label} masked padding rows are not normalized to zero")
    prefix_sha = hashlib.sha256(normalized[:PREFIX_END * row_bytes]).hexdigest()
    if prefix_sha != origin.get("global_action1_sha256"):
        raise ReferenceError(f"{label} unchanged G+A1 source-prefix digest mismatch")
    _bf16_finite(normalized, label)
    generation_binding = _authenticate_generation_binding(origin, raw_crop)
    if label == "A" and generation_binding is None:
        raise ReferenceError("accepted A input requires generation provenance binding")
    if label == "B" and generation_binding is not None:
        raise ReferenceError("semantic B must be source-Qwen-only, without a generation bundle")
    return {
        "label": label,
        "input_path": input_path,
        "origin_path": origin_path,
        "origin_sha256": sha256_file(origin_path),
        "normalized_bytes": normalized,
        "normalized_sha256": origin["sha256"],
        "prefix_sha256": prefix_sha,
        "raw_qwen_crop": raw_crop,
        "raw_qwen_crop_sha256": raw_crop_sha,
        # Preserve the absolute spelling authenticated by the origin receipt;
        # the reader resolves it to a canonical file before opening.
        "source_qwen_manifest_path": Path(qwen_manifest_arg),
        "source_qwen_manifest_sha256": qwen_manifest_sha,
        "source_qwen_manifest_schema": qwen_doc["schema"],
        "source_qwen_payload_path": qwen_sidecar,
        "source_qwen_payload_bytes": qwen_sidecar_record["nbytes"],
        "source_qwen_payload_sha256": qwen_sidecar_record["sha256"],
        "generation_input_binding": generation_binding,
    }


def _input_case_document(case: Mapping[str, Any], payload_name: str) -> dict[str, Any]:
    normalized_sha = case["normalized_sha256"]
    document: dict[str, Any] = {
        "schema": INPUT_SCHEMA,
        "schema_version": 1,
        "upstream_revision": PINNED_REVISION,
        "source_qwen_manifest": {
            "path": str(case["source_qwen_manifest_path"]),
            "sha256": case["source_qwen_manifest_sha256"],
            "schema": case["source_qwen_manifest_schema"],
            "source_revision": "eb6ba55b40ada64977147b52e50a4145ee89ac4f",
        },
        "source_qwen_payload": {
            "basename": case["source_qwen_payload_path"].name,
            "bytes": case["source_qwen_payload_bytes"],
            "sha256": case["source_qwen_payload_sha256"],
            "dtype": "BF16",
            "shape": list(QWEN_SOURCE_SHAPE),
            "offset_bytes": 0,
        },
        "source_qwen_range": {
            "tensor_name": "capture.hidden_states",
            "dtype": "BF16",
            "shape": list(QWEN_SOURCE_SHAPE),
            "offset_bytes": 0,
            "nbytes": case["source_qwen_payload_bytes"],
            "sha256": case["source_qwen_payload_sha256"],
            "state_index": STATE_INDEX,
            "crop_start": CROP_START,
            "crop_stop": RAW_SEQUENCE_LENGTH,
            "selected_shape": list(QWEN_SHAPE),
            "selected_nbytes": INPUT_BYTES,
            "selected_sha256": case["raw_qwen_crop_sha256"],
        },
        "source_qwen": {
            "name": "input.qwen_bf16",
            "dtype": "BF16",
            "shape": list(QWEN_SHAPE),
            "offset": 0,
            "nbytes": INPUT_BYTES,
            "sha256": case["raw_qwen_crop_sha256"],
        },
        "origin_manifest": {
            "path": str(case["origin_path"]),
            "sha256": case["origin_sha256"],
        },
        "normalization": {
            "valid_prefix_rows": VALID_ROWS,
            "pad_start": VALID_ROWS,
            "pad_stop": TOKEN_COUNT,
            "pad_operation": "zero_after_prefix_validation",
            "normalized_sha256": normalized_sha,
        },
        "payload": {
            "basename": payload_name,
            "bytes": INPUT_BYTES,
            "sha256": normalized_sha,
        },
        "tensors": {
            "input.qwen_bf16": {
                "dtype": "BF16",
                "shape": list(QWEN_SHAPE),
                "offset": 0,
                "nbytes": INPUT_BYTES,
                "sha256": normalized_sha,
            },
        },
    }
    if case["generation_input_binding"] is not None:
        document["generation_input_binding"] = case["generation_input_binding"]
    return document


def _validate_case_manifest(document: Mapping[str, Any]) -> None:
    required = {
        "schema", "schema_version", "upstream_revision", "source_qwen_manifest",
        "source_qwen_payload", "source_qwen_range", "source_qwen", "origin_manifest",
        "normalization", "payload", "tensors",
    }
    optional = {"generation_input_binding"}
    if (type(document) is not dict or not required.issubset(document)
            or set(document) - required - optional):
        raise ReferenceError("case input manifest keys mismatch")
    if (document.get("schema") != INPUT_SCHEMA or document.get("schema_version") != 1
            or document.get("upstream_revision") != PINNED_REVISION):
        raise ReferenceError("case input manifest schema/revision mismatch")
    source_manifest = document.get("source_qwen_manifest")
    if (type(source_manifest) is not dict
            or set(source_manifest) != {"path", "sha256", "schema", "source_revision"}
            or source_manifest.get("source_revision") != "eb6ba55b40ada64977147b52e50a4145ee89ac4f"
            or type(source_manifest.get("path")) is not str
            or not Path(source_manifest["path"]).is_absolute()
            or source_manifest.get("schema") != "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference"):
        raise ReferenceError("case source-Qwen manifest binding malformed")
    _digest(source_manifest.get("sha256"), "case source-Qwen manifest SHA-256")
    source_payload = document.get("source_qwen_payload")
    if (type(source_payload) is not dict
            or set(source_payload) != {"basename", "bytes", "sha256", "dtype", "shape", "offset_bytes"}
            or source_payload.get("dtype") != "BF16"
            or source_payload.get("shape") != list(QWEN_SOURCE_SHAPE)
            or source_payload.get("offset_bytes") != 0
            or source_payload.get("bytes") != QWEN_SOURCE_BYTES
            or Path(source_payload.get("basename", "")).name != source_payload.get("basename")):
        raise ReferenceError("case source-Qwen payload metadata mismatch")
    _digest(source_payload.get("sha256"), "case source-Qwen payload SHA-256")
    qrange = document.get("source_qwen_range")
    if (type(qrange) is not dict or qrange.get("tensor_name") != "capture.hidden_states"
            or qrange.get("dtype") != "BF16" or qrange.get("shape") != list(QWEN_SOURCE_SHAPE)
            or qrange.get("offset_bytes") != 0
            or qrange.get("nbytes") != QWEN_SOURCE_BYTES or qrange.get("state_index") != STATE_INDEX
            or qrange.get("crop_start") != CROP_START or qrange.get("crop_stop") != RAW_SEQUENCE_LENGTH
            or qrange.get("selected_shape") != list(QWEN_SHAPE)
            or qrange.get("selected_nbytes") != INPUT_BYTES):
        raise ReferenceError("case source-Qwen range metadata mismatch")
    if qrange.get("sha256") != source_payload.get("sha256"):
        raise ReferenceError("case source-Qwen whole-sidecar range digest binding mismatch")
    _digest(qrange.get("selected_sha256"), "case source-Qwen crop SHA-256")
    qwen = document.get("source_qwen")
    if (type(qwen) is not dict
            or qwen != {"name": "input.qwen_bf16", "dtype": "BF16", "shape": list(QWEN_SHAPE),
                        "offset": 0, "nbytes": INPUT_BYTES, "sha256": qrange.get("selected_sha256")}):
        raise ReferenceError("case raw source-Qwen tensor metadata mismatch")
    origin = document.get("origin_manifest")
    if (type(origin) is not dict or set(origin) != {"path", "sha256"}
            or type(origin.get("path")) is not str):
        raise ReferenceError("case origin manifest binding malformed")
    _digest(origin.get("sha256"), "case origin manifest SHA-256")
    generation_binding = document.get("generation_input_binding")
    if generation_binding is not None:
        expected_generation_keys = {
            "manifest_path", "manifest_sha256", "payload_basename", "payload_bytes",
            "payload_sha256", "tensor_name", "tensor_dtype", "tensor_shape",
            "tensor_offset", "tensor_nbytes", "tensor_sha256",
        }
        if type(generation_binding) is not dict or set(generation_binding) != expected_generation_keys:
            raise ReferenceError("accepted generation input binding inventory mismatch")
        if (type(generation_binding.get("manifest_path")) is not str
                or not Path(generation_binding["manifest_path"]).is_absolute()
                or generation_binding.get("tensor_name") != "input.qwen_bf16"
                or generation_binding.get("tensor_dtype") != "BF16"
                or generation_binding.get("tensor_shape") != list(QWEN_SHAPE)
                or type(generation_binding.get("tensor_offset")) is not int
                or generation_binding["tensor_offset"] < 0
                or generation_binding.get("tensor_nbytes") != INPUT_BYTES
                or type(generation_binding.get("payload_bytes")) is not int
                or generation_binding["payload_bytes"] <= INPUT_BYTES
                or generation_binding["tensor_offset"] + INPUT_BYTES > generation_binding["payload_bytes"]
                or Path(generation_binding.get("payload_basename", "")).name != generation_binding.get("payload_basename")):
            raise ReferenceError("accepted generation input binding geometry mismatch")
        for key in ("manifest_sha256", "payload_sha256", "tensor_sha256"):
            _digest(generation_binding.get(key), f"accepted generation {key}")
        if generation_binding["tensor_sha256"] != qrange.get("selected_sha256"):
            raise ReferenceError("accepted generation tensor does not match selected source-Qwen crop")
    normalization = document.get("normalization")
    if (type(normalization) is not dict
            or normalization != {
                "valid_prefix_rows": VALID_ROWS, "pad_start": VALID_ROWS,
                "pad_stop": TOKEN_COUNT, "pad_operation": "zero_after_prefix_validation",
                "normalized_sha256": document.get("payload", {}).get("sha256"),
            }):
        raise ReferenceError("case input normalization metadata mismatch")
    payload = document.get("payload")
    if (type(payload) is not dict or set(payload) != {"basename", "bytes", "sha256"}
            or type(payload.get("basename")) is not str
            or Path(payload["basename"]).name != payload["basename"]
            or payload.get("bytes") != INPUT_BYTES):
        raise ReferenceError("case normalized input payload metadata mismatch")
    _digest(payload.get("sha256"), "case normalized Qwen SHA-256")
    tensors = document.get("tensors")
    if type(tensors) is not dict or set(tensors) != {"input.qwen_bf16"}:
        raise ReferenceError("case input tensor inventory mismatch")
    tensor = tensors["input.qwen_bf16"]
    if (type(tensor) is not dict
            or tensor != {"dtype": "BF16", "shape": list(QWEN_SHAPE), "offset": 0,
                          "nbytes": INPUT_BYTES, "sha256": payload["sha256"]}):
        raise ReferenceError("case normalized tensor record mismatch")


def _validate_weights(weights_path: Path, manifest_path: Path) -> dict[str, Any]:
    from hy_nomas_text_conditioning_reference import TEXT_SOURCE, _verify_text_manifest

    weights_sha = sha256_file(weights_path)
    manifest_sha = sha256_file(manifest_path)
    manifest = _read_json(manifest_path, "text subset manifest")
    _verify_text_manifest(manifest, weights_path, weights_sha)
    subset = manifest["subset"]
    source = manifest["source"]
    return {
        "path": str(weights_path),
        "sha256": weights_sha,
        "bytes": weights_path.stat().st_size,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "tensor_count": subset["tensor_count"],
        "stored_dtype": subset["dtype"],
        "loaded_dtype": "BF16",
        "conversion": "F32_to_BF16_RNE",
        "source": source,
        "names": subset["tensor_names"],
        "repository": TEXT_SOURCE["repository"],
    }


def _verify_pinned_source(upstream: Path) -> dict[str, str]:
    from hy_nomas_text_conditioning_reference import PINNED_SOURCE_SHA256
    result = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = upstream / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ReferenceError(f"pinned source file hash mismatch: {relative}")
        result[relative] = expected
    try:
        import subprocess
        revision = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise ReferenceError("could not read pinned upstream git revision") from None
    if revision != PINNED_REVISION:
        raise ReferenceError(f"pinned upstream revision mismatch: {revision}")
    return result


def _module_from_pinned_source(torch: Any, upstream: Path, weights_path: Path,
                               weights_manifest: dict[str, Any], device: Any) -> tuple[dict[str, Any], Any, Any]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required for the source text subset") from error
    import importlib
    import hy_nomas_bf16_dit_quality_control as single
    import hy_nomas_reference as block_reference
    import hy_nomas_real_attention_reference as attention_reference

    dtype = torch.bfloat16
    helpers = block_reference._install_upstream_modules(upstream)
    state: dict[str, Any] = {
        "mode": "bf16",
        "valid_text_tokens": VALID_ROWS,
        "attention_block": 0,
        "attention_query_chunks": 0,
        "compare": single._NoopCompare(),
    }
    _, adapter = attention_reference._install_refiner_attention(torch, state)
    single._bind_refiner_attention_adapter(upstream, adapter, state)
    token_module = importlib.import_module("hymm.models.text_encoder.token_refiner")
    embed_layers = importlib.import_module("hymm.models.dit.modules.embed_layers")
    meta = torch.device("meta")
    modules = {
        "txt_in": token_module.SingleTokenRefinerPerBlock(
            in_channels=HIDDEN_SIZE,
            hidden_size=REFINER_DIM,
            heads_num=HEADS,
            depth=BLOCKS,
            mlp_width_ratio=4.0,
            mlp_drop_rate=0.0,
            act_type="silu",
            qk_norm=False,
            qkv_bias=True,
            attn_mode="torch",
            seg_pos_emb_num=0,
            dtype=dtype,
            device=meta,
        ),
        "time_in": embed_layers.TimestepEmbedder(
            REFINER_DIM, torch.nn.SiLU, dtype=dtype, device=meta,
        ),
    }
    expected_names = {
        f"{prefix}.{name}" for prefix, module in modules.items()
        for name in module.state_dict()
    }
    if len(expected_names) != 42 or set(weights_manifest["names"]) != expected_names:
        raise ReferenceError("source refiner weight subset name inventory mismatch")
    with safe_open(str(weights_path), framework="pt", device="cpu") as safetensor:
        if set(safetensor.keys()) != expected_names:
            raise ReferenceError("source refiner safetensors key inventory mismatch")
        for prefix, module in modules.items():
            state_dict = {}
            for relative, target in module.state_dict().items():
                name = f"{prefix}.{relative}"
                value = safetensor.get_tensor(name)
                if (value.dtype != torch.float32 or tuple(value.shape) != tuple(target.shape)
                        or not bool(torch.isfinite(value).all().item())):
                    raise ReferenceError(f"source refiner weight dtype/shape/finite mismatch: {name}")
                state_dict[relative] = value.to(dtype=dtype).contiguous()
            try:
                module.load_state_dict(state_dict, strict=True, assign=True)
            except (RuntimeError, TypeError) as error:
                raise ReferenceError(f"could not load source refiner module {prefix}: {error}") from None
            module.to(device=device, dtype=dtype).eval()
            del state_dict
    single._bind_refiner_attention_adapter(upstream, adapter, state, modules["txt_in"])
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("source refiner reference is MPS-only; CPU fallback is forbidden")
    return modules, state, adapter


def _prepare_source_inputs(torch: Any, np: Any, cases: Mapping[str, bytes]) -> dict[str, Any]:
    """Upload both saved inputs and shared controls once before the case loop."""
    def upload(payload: bytes, label: str) -> Any:
        words = np.frombuffer(payload, dtype="<u2").copy().reshape(QWEN_SHAPE)
        return torch.from_numpy(words).view(torch.bfloat16).to(device="mps")

    source_a = upload(cases["A0"], "A")
    source_b = upload(cases["B"], "B")
    slot = torch.empty(QWEN_SHAPE, dtype=torch.bfloat16, device="mps")
    mask = torch.zeros((1, TOKEN_COUNT), dtype=torch.bool, device="mps")
    mask[:, :VALID_ROWS] = True
    timestep = torch.tensor([1000.0], dtype=torch.float32, device="mps")
    return {"source": {"A0": source_a, "A1": source_a, "B": source_b},
            "slot": slot, "mask": mask, "timestep": timestep}


def _capture_source_cases(torch: Any, modules: Mapping[str, Any], state: dict[str, Any],
                          inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Run A0/A1/B while retaining all BF16 checkpoints on MPS until one export."""
    captures: dict[str, dict[str, Any]] = {case: {} for case in CASE_ORDER}
    active_case = {"name": ""}
    hooks = []

    def save_output(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            label = active_case["name"]
            if not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16:
                raise ReferenceError(f"source checkpoint is not BF16: {label}.{name}")
            value = output
            if name in ("block0.output", "block1.output"):
                value = value[:, :VALID_ROWS]
            captures[label][name] = value.detach().contiguous().clone()
        return hook

    def save_input(name: str, index: int = 0, bf16_cast: bool = False):
        def hook(_module: Any, inputs_tuple: tuple[Any, ...]) -> None:
            label = active_case["name"]
            if len(inputs_tuple) <= index or not isinstance(inputs_tuple[index], torch.Tensor):
                raise ReferenceError(f"source checkpoint input missing: {label}.{name}")
            value = inputs_tuple[index]
            if bf16_cast:
                if value.dtype != torch.float32:
                    raise ReferenceError(
                        f"source masked-mean input must be observed as F32 before BF16 cast: {label}.{name}"
                    )
                value = value.to(torch.bfloat16)
            if value.dtype != torch.bfloat16:
                raise ReferenceError(f"source checkpoint is not BF16: {label}.{name}")
            captures[label][name] = value.detach().contiguous().clone()
        return hook

    refiner = modules["txt_in"]
    hooks.append(refiner.c_embedder.register_forward_pre_hook(
        save_input("masked_mean", bf16_cast=True),
    ))
    hooks.append(refiner.individual_token_refiner.register_forward_pre_hook(
        save_input("c", index=1),
    ))
    hooks.append(modules["time_in"].register_forward_hook(save_output("vec")))
    for index, block in enumerate(refiner.individual_token_refiner.blocks):
        hooks.append(block.register_forward_hook(save_output(f"block{index}.output")))
    state["attention_block"] = 0
    state["attention_query_chunks"] = 0
    try:
        with torch.inference_mode():
            for case_name in CASE_ORDER:
                active_case["name"] = case_name
                state["attention_block"] = 0
                state["attention_query_chunks"] = 0
                inputs["slot"].copy_(inputs["source"][case_name])
                with torch.autocast("mps", dtype=torch.bfloat16):
                    vector = modules["time_in"](inputs["timestep"])
                    if vector.dtype != torch.bfloat16:
                        raise ReferenceError(f"source timestep vector is not BF16: {case_name}")
                    result = refiner(inputs["slot"], inputs["timestep"], inputs["mask"])
                if not isinstance(result, torch.Tensor) or result.dtype != torch.bfloat16:
                    raise ReferenceError(f"source refiner output is not BF16: {case_name}")
                if state.get("attention_block") != BLOCKS:
                    raise ReferenceError(f"source refiner did not execute exactly two attention blocks: {case_name}")
                if state.get("attention_query_chunks") != BLOCKS:
                    raise ReferenceError(f"source V224 adapter did not use one query chunk per block: {case_name}")
                if set(captures[case_name]) != set(CHECKPOINTS):
                    raise ReferenceError(
                        f"source checkpoint inventory mismatch for {case_name}: "
                        f"{sorted(captures[case_name])}"
                    )
    finally:
        for hook in hooks:
            hook.remove()
        active_case["name"] = ""
    return captures


def _export_source_captures(torch: Any, np: Any,
                            captures: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cast/pack on MPS, synchronize once, and copy one packed tensor to CPU."""
    ordered_keys = [(case, checkpoint) for case in CASE_ORDER for checkpoint in CHECKPOINTS]
    expanded = []
    shapes = {}
    cursor = 0
    ranges = {}
    for case, checkpoint in ordered_keys:
        tensor = captures[case][checkpoint]
        if tensor.dtype != torch.bfloat16:
            raise ReferenceError(f"source checkpoint dtype changed before export: {case}.{checkpoint}")
        shape = _output_shape(case, checkpoint)
        if list(tensor.shape) != shape:
            raise ReferenceError(f"source checkpoint shape changed before export: {case}.{checkpoint}")
        value = tensor.to(dtype=torch.float32).reshape(-1)
        expanded.append(value)
        shapes[(case, checkpoint)] = shape
        ranges[(case, checkpoint)] = (cursor, cursor + value.numel())
        cursor += value.numel()
    packed_device = torch.cat(expanded, dim=0).contiguous()
    torch.mps.synchronize()
    packed = packed_device.to(device="cpu").numpy()
    if packed.dtype != np.float32 or packed.size != cursor or not bool(np.isfinite(packed).all()):
        raise ReferenceError("packed source checkpoint export is malformed or nonfinite")
    result: dict[str, dict[str, Any]] = {case: {} for case in CASE_ORDER}
    for case, checkpoint in ordered_keys:
        start, stop = ranges[(case, checkpoint)]
        values = np.array(packed[start:stop].reshape(shapes[(case, checkpoint)]),
                          dtype="<f4", order="C", copy=True)
        result[case][checkpoint] = values
    return result


def _write_reference_payload(
    cases: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    chunks: list[bytes] = []
    tensors: dict[str, Any] = {}
    cursor = 0
    for case_name in CASE_ORDER:
        for checkpoint in CHECKPOINTS:
            name = f"{case_name}.{checkpoint}"
            values = cases[case_name][checkpoint]
            shape = list(values.shape)
            expected_shape = _output_shape(case_name, checkpoint)
            if shape != expected_shape:
                raise ReferenceError(f"source checkpoint shape mismatch: {name}")
            raw = values.astype("<f4", copy=False).tobytes(order="C")
            if len(raw) != _output_nbytes(shape):
                raise ReferenceError(f"source checkpoint byte extent mismatch: {name}")
            tensors[name] = {
                "dtype": "F32",
                "source_dtype": "BF16",
                "shape": shape,
                "offset_bytes": cursor,
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint_boundary": checkpoint,
            }
            chunks.append(raw)
            cursor += len(raw)
    payload = b"".join(chunks)
    if len(payload) != cursor:
        raise ReferenceError("source output packing extent mismatch")
    return tensors, payload


def _case_reference_record(case: Mapping[str, Any], manifest_path: Path,
                           payload_path: Path, case_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_raw = json.dumps(case_manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    payload_sha = case["normalized_sha256"]
    _validate_case_manifest(case_manifest)
    return {
        "manifest_path": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "payload_path": payload_path.name,
        "payload_bytes": INPUT_BYTES,
        "payload_sha256": payload_sha,
        "tensor_name": "input.qwen_bf16",
        "offset_bytes": 0,
        "nbytes": INPUT_BYTES,
        "source_tensor_sha256": case["raw_qwen_crop_sha256"],
        "source_shape": list(QWEN_SHAPE),
        "source_dtype": "BF16",
        "normalized_tensor_sha256": payload_sha,
    }


def run_reference(
    *, upstream_arg: str | Path,
    weights_arg: str | Path,
    weights_manifest_arg: str | Path,
    input_a_arg: str | Path,
    origin_a_arg: str | Path,
    input_b_arg: str | Path,
    origin_b_arg: str | Path,
    output_manifest_arg: str | Path,
) -> Path:
    """Run exact A0/A1/B source refiner pass on MPS and atomically publish it."""
    upstream = Path(upstream_arg).expanduser().resolve(strict=True)
    if not upstream.is_dir() or upstream.is_symlink():
        raise ReferenceError("upstream checkout must be a canonical directory")
    weights_path = _regular_file(weights_arg, "source text weights")
    weights_manifest_path = _regular_file(weights_manifest_arg, "source text weights manifest")
    output_manifest = Path(output_manifest_arg).expanduser()
    if not output_manifest.is_absolute():
        output_manifest = Path.cwd() / output_manifest
    output_manifest = output_manifest.parent.resolve(strict=True) / output_manifest.name
    output_payload = output_manifest.with_name(output_manifest.stem + ".source.f32le")
    if output_manifest.exists() or output_payload.exists():
        raise ReferenceError("refusing to overwrite an existing source refiner reference")
    case_manifest_paths = {
        "A": output_manifest.with_name(output_manifest.stem + ".input-a.json"),
        "B": output_manifest.with_name(output_manifest.stem + ".input-b.json"),
    }
    if any(path.exists() for path in case_manifest_paths.values()):
        raise ReferenceError("refusing to overwrite an existing source Qwen case manifest")

    # All source provenance and counterfactual gates run before importing torch.
    source_hashes = _verify_pinned_source(upstream)
    weights = _validate_weights(weights_path, weights_manifest_path)
    source_checkpoint = {
        key: weights["source"][key]
        for key in ("repository", "checkpoint_path", "checkpoint_lfs_sha256",
                    "checkpoint_lfs_sha256_status", "source_git_commit")
    }
    source_checkpoint["source_snapshot_revision"] = weights["source"]["revision"]
    if source_checkpoint["source_git_commit"] != PINNED_REVISION:
        raise ReferenceError("text subset source commit does not match pinned Hy-Nomas")

    input_a = authenticate_case_input(input_a_arg, origin_a_arg, "A")
    input_b = authenticate_case_input(input_b_arg, origin_b_arg, "B")
    prepared = prepare_pair(input_a["normalized_bytes"], input_b["normalized_bytes"])
    for name, case in (("A0", input_a), ("A1", input_a), ("B", input_b)):
        if prepared[name] != case["normalized_bytes"]:
            raise ReferenceError(f"{name} normalized tensor differs from validated source rows")
    if input_a["source_qwen_manifest_sha256"] == input_b["source_qwen_manifest_sha256"]:
        raise ReferenceError("semantic B case must have a distinct source Qwen manifest")

    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ReferenceError(f"source MPS runtime dependency missing: {error}") from error
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("source refiner reference is MPS-only; CPU fallback is forbidden")
    modules, state, adapter = _module_from_pinned_source(
        torch, upstream, weights_path, weights, torch.device("mps"),
    )
    source_inputs = _prepare_source_inputs(torch, np, prepared)
    gpu_captures = _capture_source_cases(torch, modules, state, source_inputs)
    source_cases = _export_source_captures(torch, np, gpu_captures)
    aa_stats = {}
    for checkpoint in CHECKPOINTS:
        aa_stats[checkpoint] = compare_checkpoint(
            f"A0/A1.{checkpoint}", source_cases["A0"][checkpoint],
            source_cases["A1"][checkpoint],
        )
        if not aa_stats[checkpoint]["passed"]:
            raise ReferenceError(f"A/A source floor is not stable: {checkpoint}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    cases_for_manifest = {"A0": input_a, "A1": input_a, "B": input_b}
    case_docs = {
        "A": _input_case_document(input_a, input_a["input_path"].name),
        "B": _input_case_document(input_b, input_b["input_path"].name),
    }
    tensors, output_payload_bytes = _write_reference_payload(source_cases)
    source = {
        "upstream_path": str(upstream),
        "revision": PINNED_REVISION,
        "source_files": source_hashes,
        "text_weights": {key: weights[key] for key in (
            "path", "sha256", "bytes", "manifest_path", "manifest_sha256",
            "tensor_count", "stored_dtype", "loaded_dtype", "conversion",
        )},
        "source_checkpoint": {
            "repository": source_checkpoint["repository"],
            "checkpoint_path": source_checkpoint["checkpoint_path"],
            "checkpoint_lfs_sha256": source_checkpoint["checkpoint_lfs_sha256"],
            "checkpoint_lfs_sha256_status": source_checkpoint["checkpoint_lfs_sha256_status"],
            "source_snapshot_revision": source_checkpoint["source_snapshot_revision"],
        },
        "module_bindings": {
            "refiner": "hymm.models.text_encoder.token_refiner.SingleTokenRefinerPerBlock",
            "time_embedder": "hymm.models.dit.modules.embed_layers.TimestepEmbedder",
        },
    }
    input_cases = {
        "A0": _case_reference_record(input_a, case_manifest_paths["A"], input_a["input_path"], case_docs["A"]),
        "A1": _case_reference_record(input_a, case_manifest_paths["A"], input_a["input_path"], case_docs["A"]),
        "B": _case_reference_record(input_b, case_manifest_paths["B"], input_b["input_path"], case_docs["B"]),
    }
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "upstream_revision": PINNED_REVISION,
        "source": source,
        "geometry": GEOMETRY,
        "timestep": 1000.0,
        "mask": MASK_RECORD,
        "action_spans": ACTION_SPANS,
        "inputs": {
            "cases": input_cases,
            "prefix_checks": {
                "validated_before_padding_normalization": True,
                "unchanged_rows": [0, PREFIX_END],
                "changed_rows": [ACTION2_START, ACTION2_END],
                "action2_changed_bytes": sum(
                    a != b for a, b in zip(
                        input_a["raw_qwen_crop"][ACTION2_START * HIDDEN_SIZE * 2:ACTION2_END * HIDDEN_SIZE * 2],
                        input_b["raw_qwen_crop"][ACTION2_START * HIDDEN_SIZE * 2:ACTION2_END * HIDDEN_SIZE * 2],
                    )
                ),
                "global_action1_sha256": input_a["prefix_sha256"],
                "A0_A1_same_normalized_qwen_sha256": input_a["normalized_sha256"],
            },
        },
        "normalization": NORMALIZATION,
        "precision": PRECISION,
        "execution": {
            **EXECUTION,
            "source_precision": SOURCE_PRECISION,
            "source_attention_query_chunks": state["attention_query_chunks"],
            "source_a_a_floor": aa_stats,
        },
        "payload": {
            "basename": output_payload.name,
            "bytes": len(output_payload_bytes),
            "sha256": hashlib.sha256(output_payload_bytes).hexdigest(),
            "encoding": "f32le_bf16_expansion",
        },
        "tensors": tensors,
    }
    validate_reference_document(document)
    manifest_raw = json.dumps(document, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    # Stage and validate the complete output set under a temporary directory
    # on the same filesystem. The source payload uses its final basename there
    # so validation exercises the exact reader path before publication.
    case_manifest_raw = {
        key: json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        for key, value in case_docs.items()
    }
    created: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".hy-nomas-refiner-transfer-",
                                     dir=output_manifest.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged_payload = temp_dir / output_payload.name
        with staged_payload.open("xb") as stream:
            stream.write(output_payload_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        validate_reference_payload(document, staged_payload)
        staged_paths = {
            "payload": staged_payload,
            "A": temp_dir / case_manifest_paths["A"].name,
            "B": temp_dir / case_manifest_paths["B"].name,
            "reference": temp_dir / output_manifest.name,
        }
        for key, raw in (("A", case_manifest_raw["A"]),
                         ("B", case_manifest_raw["B"]),
                         ("reference", manifest_raw)):
            with staged_paths[key].open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        destinations = {
            "payload": output_payload,
            "A": case_manifest_paths["A"],
            "B": case_manifest_paths["B"],
            "reference": output_manifest,
        }
        try:
            for key in ("payload", "A", "B", "reference"):
                os.replace(staged_paths[key], destinations[key])
                created.append(destinations[key])
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
    return output_manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--weights-manifest", required=True, type=Path)
    parser.add_argument("--input-a", required=True, type=Path)
    parser.add_argument("--origin-a", required=True, type=Path)
    parser.add_argument("--input-b", required=True, type=Path)
    parser.add_argument("--origin-b", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_reference(
            upstream_arg=args.upstream,
            weights_arg=args.weights,
            weights_manifest_arg=args.weights_manifest,
            input_a_arg=args.input_a,
            origin_a_arg=args.origin_a,
            input_b_arg=args.input_b,
            origin_b_arg=args.origin_b,
            output_manifest_arg=args.output_manifest,
        )
    except (ReferenceError, OSError) as error:
        print(f"refiner-transfer reference rejected: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
