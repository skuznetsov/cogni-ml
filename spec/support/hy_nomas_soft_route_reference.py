#!/usr/bin/env python3
"""CPU admission and pinned-source reference for bounded soft-route biases.

This experiment holds the accepted M11 hard support fixed and adds authored
BF16 logit priors only to matching action-text keys for video queries. The
priors are static image-derived tents, not tracks or segmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_dit_route_factorial_reference as base  # noqa: E402

SCHEMA = "hy-nomas-soft-route-reference-v1"
SCHEMA_VERSION = 1
DIAGNOSTIC_SCHEMA = "hy-nomas-soft-route-rejected-diagnostic-v1"
DIAGNOSTIC_SCHEMA_VERSION = 1
BIAS_SCHEMA = "hy-nomas-soft-route-bias-v1"
BIAS_SHA256 = "81dd6c8d9ba01bd8824cba8c0c48f238f9180c2cb22a3f18f7c2cb2d5331235e"
PARENT_SHA256 = "efcf746fd334218d25b0a922647f10962c6d5102144f7d80cff67bb327a40c89"
DEFAULT_BIAS_MANIFEST = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-soft-routes-20261105.j0wgy7yl/bias-v2.json")
DEFAULT_PARENT_REFERENCE = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-direct-routes-20261105.klzuhg0r/source-qualified.json")
CASE_ORDER = ("zero", "A1_low", "A1_high", "A2_low", "A2_high", "A1_sham", "A2_sham", "zero_repeat")
ZERO_CASES = ("zero", "zero_repeat")
POSITIVE_CASES = tuple(case for case in CASE_ORDER if case not in ZERO_CASES)
VIDEO_ROWS = base.VIDEO_ROWS
QUERY_ROWS = base.QUERY_ROWS
KEY_ROWS = base.KEY_ROWS
CHUNK = 256
ACTION_SPANS = {
    "A1": (base.HISTORY_ROWS + VIDEO_ROWS + 49, base.HISTORY_ROWS + VIDEO_ROWS + 132),
    "A2": (base.HISTORY_ROWS + VIDEO_ROWS + 132, base.HISTORY_ROWS + VIDEO_ROWS + 224),
}
TENSOR_NAMES = {f"{case}.{checkpoint}" for case in CASE_ORDER for checkpoint in base.CHECKPOINTS}


class SoftRouteError(RuntimeError):
    """Fail-closed bias, parent, source, or output validation error."""


class ExactReplayRejected(SoftRouteError):
    """Numerical gates completed, but at least one exact baseline replay failed."""

    def __init__(self, gates: Mapping[str, Any], failures: list[dict[str, Any]]) -> None:
        self.gates = dict(gates)
        self.failures = list(failures)
        first = self.failures[0]
        control = first["control"]
        label = "pinned parent" if control == "parent_replay" else control.replace("_", " ")
        super().__init__(f"{label} {first['checkpoint']} fails its exact baseline replay gate")


def _bf16(value: float) -> float:
    """Round an F32 value to BF16 with round-to-nearest-even, return as F32."""
    packed = struct.pack("<f", value)
    bits, = struct.unpack("<I", packed)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _expected_maps() -> dict[str, Any]:
    return {
        "kind": "authored static separable tent, not measured tracks",
        "grid": [30, 9, 16],
        "cell_centers_xy": "(32*(w+0.5),32*(h+0.5)) in pinned 512x288 image",
        "A1": {"center_hw": [1.5, 5.5], "frames": [1, 10], "sham_shift_hw": [3, 0],
               "radius_cells": [2, 2], "peak_normalizer": 0.5625},
        "A2": {"center_hw": [4.5, 11.5], "frames": [12, 28], "sham_shift_hw": [-2, 0],
               "radius_cells": [1.5, 2], "peak_normalizer": 0.5},
        "doses": {"low": math.log(1.5), "high": math.log(2)},
        "rounding": "F32 product then BF16 nearest-even, lossless F32 expansion",
        "support": "accepted M11; no edges added; video queries only",
    }


def _prior(action: str, sham: bool) -> list[float]:
    spec = _expected_maps()[action]
    center_h, center_w = spec["center_hw"]
    radius_h, radius_w = spec["radius_cells"]
    start, stop = spec["frames"]
    shift_h, shift_w = spec["sham_shift_hw"] if sham else (0, 0)
    normalizer = spec["peak_normalizer"]
    values = [0.0] * QUERY_ROWS
    for frame in range(start, stop):
        for h in range(9):
            for w in range(16):
                spatial = (
                    max(0.0, 1.0 - abs(h - shift_h - center_h) / radius_h)
                    * max(0.0, 1.0 - abs(w - shift_w - center_w) / radius_w)
                    / normalizer
                )
                values[frame * 144 + h * 16 + w] = spatial
    return values


def expected_bias_vectors() -> dict[str, list[float]]:
    """Regenerate the manifest values independently from its serialized arrays."""
    result: dict[str, list[float]] = {}
    for case in CASE_ORDER:
        if case in ZERO_CASES:
            result[case] = [0.0] * QUERY_ROWS
            continue
        action = case[:2]
        dose = math.log(1.5) if case.endswith("low") else math.log(2.0)
        prior = _prior(action, case.endswith("sham"))
        result[case] = [_bf16(dose * item) for item in prior]
    return result


def _vector_sha256(values: list[float]) -> str:
    try:
        packed = struct.pack(f"<{QUERY_ROWS}f", *values)
    except (OverflowError, struct.error) as error:
        raise SoftRouteError(f"bias vector cannot be encoded as F32: {error}") from None
    return hashlib.sha256(packed).hexdigest()


def validate_bias_document(document: Any, *, verify_reference_image: bool = True) -> dict[str, list[float]]:
    """Validate exact map semantics, F32/BF16 encoding, support, and per-case hashes."""
    keys = {"schema", "geometry", "case_order", "reference_image", "maps", "biases", "bias_f32le_sha256"}
    if type(document) is not dict or set(document) != keys:
        raise SoftRouteError("bias manifest keyset mismatch")
    if (document.get("schema") != BIAS_SCHEMA
            or document.get("geometry") != {"video_rows": VIDEO_ROWS, "query_rows": QUERY_ROWS}
            or document.get("case_order") != list(CASE_ORDER)
            or document.get("maps") != _expected_maps()):
        raise SoftRouteError("bias manifest schema, geometry, case order, or map parameters mismatch")
    image = document.get("reference_image")
    if type(image) is not dict or set(image) != {"path", "sha256"}:
        raise SoftRouteError("reference image binding malformed")
    base._absolute_path(image.get("path"), "bias reference image path")
    base._digest(image.get("sha256"), "bias reference image SHA-256")
    if verify_reference_image:
        image_path = base._canonical_file(image["path"], "bias reference image")
        if base.sha256_file(image_path) != image["sha256"]:
            raise SoftRouteError("bias reference image digest mismatch")

    raw_vectors = document.get("biases")
    claimed_hashes = document.get("bias_f32le_sha256")
    if type(raw_vectors) is not dict or set(raw_vectors) != set(CASE_ORDER):
        raise SoftRouteError("bias case inventory mismatch")
    if type(claimed_hashes) is not dict or set(claimed_hashes) != set(CASE_ORDER):
        raise SoftRouteError("bias hash inventory mismatch")
    expected = expected_bias_vectors()
    vectors: dict[str, list[float]] = {}
    for case in CASE_ORDER:
        value = raw_vectors[case]
        if type(value) is not list or len(value) != QUERY_ROWS:
            raise SoftRouteError(f"bias query vector length mismatch: {case}")
        checked: list[float] = []
        for item in value:
            if type(item) not in (int, float) or not math.isfinite(item) or item < 0:
                raise SoftRouteError(f"bias value must be finite and nonnegative: {case}")
            f32 = _f32(float(item))
            if f32 != item or _bf16(f32) != f32:
                raise SoftRouteError(f"bias value is not an exact BF16-expanded F32 value: {case}")
            checked.append(f32)
        if checked != expected[case]:
            raise SoftRouteError(f"bias values differ from authored map/dose recipe: {case}")
        actual_sha = _vector_sha256(checked)
        if claimed_hashes[case] != actual_sha:
            raise SoftRouteError(f"bias vector range digest mismatch: {case}")
        vectors[case] = checked

    for action in ("A1", "A2"):
        if sorted(vectors[f"{action}_high"]) != sorted(vectors[f"{action}_sham"]):
            raise SoftRouteError(f"sham bias does not preserve the authored value histogram: {action}")
        if sum(vectors[f"{action}_high"]) != sum(vectors[f"{action}_sham"]):
            raise SoftRouteError(f"sham bias does not preserve total mass: {action}")
    validate_bias_support(vectors)
    return vectors


def authenticate_bias_manifest(path: str | Path, expected_sha256: str = BIAS_SHA256) -> dict[str, Any]:
    if expected_sha256 != BIAS_SHA256:
        raise SoftRouteError("bias manifest SHA does not match the pinned bias-v2 digest")
    try:
        manifest_path = base._require_file_digest(path, BIAS_SHA256, "soft-route bias manifest")
        document = base._read_json(manifest_path, "soft-route bias manifest")
        vectors = validate_bias_document(document)
    except base.ReferenceError as error:
        raise SoftRouteError(str(error)) from None
    return {"path": str(manifest_path), "sha256": BIAS_SHA256, "document": document, "vectors": vectors}


def validate_bias_support(vectors: Mapping[str, list[float]], source_m11_chunks: Any = None) -> dict[str, Any]:
    """Prove every positive row lands on its action's accepted M11 text edges."""
    if set(vectors) != set(CASE_ORDER):
        raise SoftRouteError("bias support check requires the exact eight cases")
    query_labels, key_labels = base.query_key_labels()
    nonzero: dict[str, int] = {}
    for case in POSITIVE_CASES:
        action = case[:2]
        action_index = 1 if action == "A1" else 2
        key_start, key_stop = ACTION_SPANS[action]
        span_start, span_stop = ((49, 132) if action == "A1" else (132, 224))
        if key_labels[key_start:key_stop] != [action_index] * (span_stop - span_start):
            raise SoftRouteError(f"action key labels do not match bias target: {case}")
        indexes = [index for index, value in enumerate(vectors[case]) if value != 0]
        if not indexes:
            raise SoftRouteError(f"positive bias case is empty: {case}")
        for query in indexes:
            if (query >= VIDEO_ROWS or query_labels[query] != action_index
                    or not base._roi_contains(query, action_index - 1)):
                raise SoftRouteError(f"bias query escapes action/time/ROI support: {case}:{query}")
            if source_m11_chunks is not None:
                chunk_index, row_index = divmod(query, CHUNK)
                try:
                    row = source_m11_chunks[chunk_index][row_index, key_start:key_stop]
                    allowed = bool(row.all().item())
                except (IndexError, AttributeError, TypeError, ValueError):
                    raise SoftRouteError(f"live M11 mask geometry invalid at {case}:{query}") from None
                if not allowed:
                    raise SoftRouteError(f"bias query targets a masked M11 text edge: {case}:{query}")
        nonzero[case] = len(indexes)
    return {"nonzero_query_rows": nonzero, "target_key_spans": ACTION_SPANS.copy(),
            "text_suffix_rows_zero": all(
                all(value == 0 for value in vectors[case][VIDEO_ROWS:]) for case in CASE_ORDER
            )}


def authenticate_parent_reference(path: str | Path, expected_sha256: str = PARENT_SHA256) -> dict[str, Any]:
    if expected_sha256 != PARENT_SHA256:
        raise SoftRouteError("parent reference SHA does not match the pinned factorial result")
    try:
        manifest_path = base._require_file_digest(path, PARENT_SHA256, "direct-route parent reference")
        document = base._read_json(manifest_path, "direct-route parent reference")
        base.validate_reference_document(document)
        payload_name = document["payload"]["basename"]
        payload_path = base._require_file_digest(
            manifest_path.parent / payload_name, document["payload"]["sha256"], "direct-route parent payload",
        )
    except base.ReferenceError as error:
        raise SoftRouteError(str(error)) from None
    return {"path": str(manifest_path), "sha256": PARENT_SHA256, "document": document,
            "payload_path": str(payload_path)}


def validate_capture_map(captures: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise SoftRouteError(f"NumPy is required for source captures: {error}") from error
    if not isinstance(captures, Mapping) or set(captures) != TENSOR_NAMES:
        raise SoftRouteError("soft-route source checkpoint inventory mismatch")
    checked: dict[str, Any] = {}
    for case in CASE_ORDER:
        for checkpoint in base.CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            value = np.asarray(captures[name])
            if (list(value.shape) != base.OUTPUT_SHAPES[checkpoint]
                    or value.dtype != np.dtype("float32") or not np.isfinite(value).all()):
                raise SoftRouteError(f"soft-route source checkpoint shape/dtype/nonfinite mismatch: {name}")
            checked[name] = value
    reference_text = checked["zero.block0.text_valid"]
    if any(not np.array_equal(reference_text, checked[f"{case}.block0.text_valid"]) for case in CASE_ORDER[1:]):
        raise SoftRouteError("video-query bias altered block0 text-query outputs")
    if any(not np.array_equal(checked[f"zero.{checkpoint}"], checked[f"zero_repeat.{checkpoint}"])
           for checkpoint in base.CHECKPOINTS):
        raise SoftRouteError("zero/zero_repeat source A/A is not bitwise exact")
    return checked


def validate_runtime_receipt(receipt: Any) -> dict[str, Any]:
    expected = {
        "attention_calls": 54 * 9,
        "attention_query_chunks": 54 * 9 * 18,
        "t0_attention_calls": 54,
        "t1_attention_calls": 54 * len(CASE_ORDER),
        "explicit_terminal_synchronizations": 1,
        "intermediate_readbacks": 0,
        "scheduler_calls": 0,
        "decoder_calls": 0,
        "refiner_calls": 0,
        "weights_staged": 54,
        "arms_per_layer": 1 + len(CASE_ORDER),
        "one_layer_cache": True,
        "autocast": "mps BF16 full block and final head",
    }
    if type(receipt) is not dict or receipt != expected:
        raise SoftRouteError("soft-route source runtime receipt mismatch")
    return dict(receipt)


def _comparison_gates(parent: Mapping[str, Any], admitted: Mapping[str, Any], captures: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    parent_doc = parent["document"]
    parent_payload = Path(parent["payload_path"])
    parent_gates: dict[str, Any] = {}
    accepted_gates: dict[str, Any] = {}
    aa_gates: dict[str, Any] = {}
    effects: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for checkpoint in base.CHECKPOINTS:
        record = parent_doc["tensors"][f"M11.{checkpoint}"]
        lhs = np.memmap(parent_payload, dtype="<f4", mode="r", offset=record["offset_bytes"],
                        shape=tuple(base.OUTPUT_SHAPES[checkpoint]))
        rhs = captures[f"zero.{checkpoint}"]
        stats = base.compare_checkpoint(checkpoint, lhs, rhs)
        parent_gates[checkpoint] = stats
        if not _exact_control_matches(lhs, rhs, stats):
            failures.append({"control": "parent_replay", "checkpoint": checkpoint,
                             "reason": "not bitwise identical to pinned parent",
                             "bitwise_equal": False, "stats": stats})
        del lhs
        aa_gates[checkpoint] = base.compare_checkpoint(
            checkpoint, captures[f"zero.{checkpoint}"], captures[f"zero_repeat.{checkpoint}"]
        )
        if not _exact_control_matches(
                captures[f"zero.{checkpoint}"], captures[f"zero_repeat.{checkpoint}"], aa_gates[checkpoint]):
            failures.append({"control": "source_aa", "checkpoint": checkpoint,
                             "reason": "zero and zero_repeat are not bitwise identical",
                             "bitwise_equal": False, "stats": aa_gates[checkpoint]})

    # Recheck the accepted ROI binding directly; the parent factorial helper
    # expects both its Boolean arms, which this soft-route child does not have.
    roi_payload = Path(admitted["roi_payload_path"])
    for checkpoint in base.CHECKPOINTS:
        record = admitted["parent_checkpoints"][checkpoint]
        roi_values = np.memmap(
            roi_payload, dtype="<f4", mode="r", offset=record["offset"],
            shape=tuple(base.OUTPUT_SHAPES[checkpoint]),
        )
        rhs = captures[f"zero.{checkpoint}"]
        stats = base.compare_checkpoint(checkpoint, roi_values, rhs)
        accepted_gates[checkpoint] = stats
        if not _exact_control_matches(roi_values, rhs, stats):
            failures.append({"control": "accepted_roi_replay", "checkpoint": checkpoint,
                             "reason": "not bitwise identical to accepted ROI checkpoint",
                             "bitwise_equal": False, "stats": stats})
        del roi_values

    for case in POSITIVE_CASES:
        effects[case] = {
            checkpoint: base.compare_checkpoint(
                f"{case}-vs-zero:{checkpoint}", captures[f"zero.{checkpoint}"], captures[f"{case}.{checkpoint}"]
            )
            for checkpoint in base.CHECKPOINTS
        }
    gates = {
        "parent_replay": {"zero": parent_gates},
        "accepted_roi_replay": accepted_gates,
        "source_aa": aa_gates,
        "intervention_effects": effects,
    }
    if failures:
        gates["comparison_failures"] = failures
        raise ExactReplayRejected(gates, failures)
    return gates


def _exact_control_matches(expected: Any, actual: Any, stats: Mapping[str, Any]) -> bool:
    """Return whether a tolerance-passing checkpoint is also bitwise identical."""
    import numpy as np

    if not stats.get("passed") or stats.get("max_abs") != 0:
        return False
    lhs = np.ascontiguousarray(expected)
    rhs = np.ascontiguousarray(actual)
    return (
        lhs.dtype == np.dtype("float32") and rhs.dtype == np.dtype("float32")
        and lhs.shape == rhs.shape
        and np.array_equal(lhs.view(np.uint32), rhs.view(np.uint32))
    )


def _assert_exact_control(label: str, expected: Any, actual: Any, stats: Mapping[str, Any]) -> None:
    """Require exact F32 bit-pattern replay for baseline controls, beyond tolerance."""
    if not _exact_control_matches(expected, actual, stats):
        raise SoftRouteError(f"{label} fails its exact baseline replay gate")


def _geometry_document() -> dict[str, Any]:
    return {
        **{
            "video_rows": VIDEO_ROWS, "t0_valid_text_rows": base.T0_VALID_TEXT_ROWS,
            "t1_valid_text_rows": base.T1_VALID_TEXT_ROWS, "text_storage_rows": base.TEXT_STORAGE_ROWS,
            "physical_cache_rows": base.PHYSICAL_CACHE_ROWS, "history_rows": base.HISTORY_ROWS,
            "query_rows": QUERY_ROWS, "key_rows": KEY_ROWS, "hidden": base.HIDDEN,
            "heads": base.HEADS, "head_dim": base.HEAD_DIM, "blocks": base.BLOCKS,
            "attention_layout": "compact",
        },
        "bias_rows": VIDEO_ROWS,
        "bias_text_suffix_rows": QUERY_ROWS - VIDEO_ROWS,
        "action_key_spans": ACTION_SPANS,
    }


def _bias_binding(authenticated: Mapping[str, Any]) -> dict[str, Any]:
    document = authenticated["document"]
    return {
        "path": authenticated["path"], "sha256": authenticated["sha256"],
        "schema": BIAS_SCHEMA, "case_order": list(CASE_ORDER),
        "reference_image": dict(document["reference_image"]),
        "maps": document["maps"], "bias_f32le_sha256": document["bias_f32le_sha256"],
    }


def _parent_binding(parent: Mapping[str, Any]) -> dict[str, Any]:
    payload = parent["document"]["payload"]
    return {
        "path": parent["path"], "sha256": parent["sha256"],
        "schema": base.SCHEMA,
        "payload_path": parent["payload_path"],
        "payload_basename": payload["basename"], "payload_bytes": payload["bytes"],
        "payload_sha256": payload["sha256"],
    }


def _render_document(*, parent: Mapping[str, Any], bias: Mapping[str, Any], source: Mapping[str, Any],
                     admitted: Mapping[str, Any], gates: Mapping[str, Any], receipt: Mapping[str, Any],
                     tensor_records: Mapping[str, Any], payload_basename: str, payload_bytes: int,
                     payload_sha256: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "upstream_revision": base.PINNED_REVISION,
        "parent_reference": _parent_binding(parent),
        "bias_manifest": _bias_binding(bias),
        "source": dict(source),
        "source_inputs": base.source_inputs_document(admitted),
        "geometry": _geometry_document(),
        "precision": {
            **base.PRECISION_CONTRACT,
            "boolean_zero_cases": list(ZERO_CASES),
            "positive_mask_dtype": "BF16 additive MPS SDPA mask; 0 on allowed support, -inf off support, authored BF16 bias on target action-text keys",
            "hard_support": "pinned accepted M11 Boolean mask; no edges added or removed",
            "bias_operation": "F32 manifest values exactly BF16-expanded; one BF16 logit addition before softmax",
        },
        "execution": {
            "case_order": list(CASE_ORDER),
            "scope": "saved-state direct DiT only; no encoder/refiner/Euler/DCAE",
            "soft_route_support": "accepted M11; per-video-query additive bias to matching action-text keys only",
            "runtime_receipt": dict(receipt),
            "parent_replay": gates["parent_replay"],
            "accepted_roi_replay": gates["accepted_roi_replay"],
            "source_aa": gates["source_aa"],
            "intervention_effects": gates["intervention_effects"],
            "block0_text_valid_equal_all_cases": True,
            "zero_repeat_bitwise_exact": True,
        },
        "payload": {"basename": payload_basename, "bytes": payload_bytes,
                    "sha256": payload_sha256, "encoding": "f32le_bf16_expansion"},
        "tensors": dict(tensor_records),
    }


def validate_reference_document(document: Any) -> None:
    required = {"schema", "schema_version", "upstream_revision", "parent_reference", "bias_manifest",
                "source", "source_inputs", "geometry", "precision", "execution", "payload", "tensors"}
    if type(document) is not dict or set(document) != required:
        raise SoftRouteError("soft-route reference document keyset mismatch")
    if (document["schema"] != SCHEMA or document["schema_version"] != SCHEMA_VERSION
            or document["upstream_revision"] != base.PINNED_REVISION):
        raise SoftRouteError("soft-route reference schema/source revision mismatch")
    parent = document["parent_reference"]
    if (type(parent) is not dict or set(parent) != {"path", "sha256", "schema", "payload_path",
                                                   "payload_basename", "payload_bytes", "payload_sha256"}
            or parent.get("schema") != base.SCHEMA or parent.get("sha256") != PARENT_SHA256
            or Path(parent.get("payload_basename", "")).name != parent.get("payload_basename")
            or type(parent.get("payload_bytes")) is not int or parent["payload_bytes"] <= 0):
        raise SoftRouteError("soft-route parent reference binding mismatch")
    for name in ("path", "payload_path"):
        base._absolute_path(parent.get(name), f"parent {name}")
    base._digest(parent.get("payload_sha256"), "parent payload SHA-256")
    bias = document["bias_manifest"]
    bias_keys = {"path", "sha256", "schema", "case_order", "reference_image", "maps", "bias_f32le_sha256"}
    if (type(bias) is not dict or set(bias) != bias_keys or bias.get("sha256") != BIAS_SHA256
            or bias.get("schema") != BIAS_SCHEMA or bias.get("case_order") != list(CASE_ORDER)
            or bias.get("maps") != _expected_maps()):
        raise SoftRouteError("soft-route bias reference binding mismatch")
    base._absolute_path(bias.get("path"), "bias manifest path")
    base._digest(bias.get("sha256"), "bias manifest SHA-256")
    base._absolute_path(bias.get("reference_image", {}).get("path"), "bias reference image path")
    base._digest(bias.get("reference_image", {}).get("sha256"), "bias reference image SHA-256")
    hashes = bias.get("bias_f32le_sha256")
    if type(hashes) is not dict or set(hashes) != set(CASE_ORDER):
        raise SoftRouteError("soft-route bias vector digest inventory mismatch")
    for case in CASE_ORDER:
        base._digest(hashes[case], f"bias {case} digest")
    try:
        base._validate_source(document["source"])
        base._validate_source_inputs(document["source_inputs"])
    except base.ReferenceError as error:
        raise SoftRouteError(str(error)) from None
    if document["geometry"] != _geometry_document():
        raise SoftRouteError("soft-route source geometry mismatch")
    expected_precision = {
        **base.PRECISION_CONTRACT,
        "boolean_zero_cases": list(ZERO_CASES),
        "positive_mask_dtype": "BF16 additive MPS SDPA mask; 0 on allowed support, -inf off support, authored BF16 bias on target action-text keys",
        "hard_support": "pinned accepted M11 Boolean mask; no edges added or removed",
        "bias_operation": "F32 manifest values exactly BF16-expanded; one BF16 logit addition before softmax",
    }
    if document["precision"] != expected_precision:
        raise SoftRouteError("soft-route precision contract mismatch")
    execution = document["execution"]
    exec_keys = {"case_order", "scope", "soft_route_support", "runtime_receipt", "parent_replay",
                 "accepted_roi_replay", "source_aa", "intervention_effects",
                 "block0_text_valid_equal_all_cases", "zero_repeat_bitwise_exact"}
    if (type(execution) is not dict or set(execution) != exec_keys
            or execution.get("case_order") != list(CASE_ORDER)
            or execution.get("scope") != "saved-state direct DiT only; no encoder/refiner/Euler/DCAE"
            or execution.get("soft_route_support") != "accepted M11; per-video-query additive bias to matching action-text keys only"
            or execution.get("block0_text_valid_equal_all_cases") is not True
            or execution.get("zero_repeat_bitwise_exact") is not True):
        raise SoftRouteError("soft-route execution contract mismatch")
    validate_runtime_receipt(execution.get("runtime_receipt"))
    for gate_name in ("parent_replay", "accepted_roi_replay"):
        gates = execution.get(gate_name)
        if gate_name == "parent_replay":
            if (type(gates) is not dict or set(gates) != {"zero"}
                    or type(gates["zero"]) is not dict or set(gates["zero"]) != set(base.CHECKPOINTS)):
                raise SoftRouteError(f"{gate_name} checkpoint inventory mismatch")
            gates = gates["zero"]
        elif type(gates) is not dict or set(gates) != set(base.CHECKPOINTS):
            raise SoftRouteError(f"{gate_name} checkpoint inventory mismatch")
        for checkpoint in base.CHECKPOINTS:
            base._validate_stats(gates[checkpoint], checkpoint, checkpoint)
            if not gates[checkpoint]["passed"] or gates[checkpoint]["max_abs"] != 0:
                raise SoftRouteError(f"{gate_name} requires exact baseline replay: {checkpoint}")
    aa = execution.get("source_aa")
    if type(aa) is not dict or set(aa) != set(base.CHECKPOINTS):
        raise SoftRouteError("soft-route source A/A checkpoint inventory mismatch")
    for checkpoint in base.CHECKPOINTS:
        base._validate_stats(aa[checkpoint], checkpoint, checkpoint)
        if not aa[checkpoint]["passed"] or aa[checkpoint]["max_abs"] != 0:
            raise SoftRouteError("soft-route source A/A must be bitwise exact")
    effects = execution.get("intervention_effects")
    if type(effects) is not dict or set(effects) != set(POSITIVE_CASES):
        raise SoftRouteError("soft-route effect inventory mismatch")
    for case in POSITIVE_CASES:
        if type(effects[case]) is not dict or set(effects[case]) != set(base.CHECKPOINTS):
            raise SoftRouteError(f"soft-route effect checkpoint inventory mismatch: {case}")
        for checkpoint in base.CHECKPOINTS:
            stats = effects[case][checkpoint]
            if (type(stats) is not dict or stats.get("checkpoint") != f"{case}-vs-zero:{checkpoint}"
                    or stats.get("elements") != math.prod(base.OUTPUT_SHAPES[checkpoint])
                    or stats.get("all_finite") is not True):
                raise SoftRouteError(f"soft-route effect record malformed: {case}.{checkpoint}")
    payload = document["payload"]
    if (type(payload) is not dict or set(payload) != {"basename", "bytes", "sha256", "encoding"}
            or type(payload.get("basename")) is not str or Path(payload["basename"]).name != payload["basename"]
            or type(payload.get("bytes")) is not int or payload["bytes"] <= 0
            or payload.get("encoding") != "f32le_bf16_expansion"):
        raise SoftRouteError("soft-route output payload descriptor mismatch")
    base._digest(payload.get("sha256"), "soft-route output payload SHA-256")
    tensors = document["tensors"]
    if type(tensors) is not dict or set(tensors) != TENSOR_NAMES:
        raise SoftRouteError("soft-route output tensor inventory mismatch")
    cursor = 0
    for case in CASE_ORDER:
        for checkpoint in base.CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            record = tensors[name]
            nbytes = math.prod(base.OUTPUT_SHAPES[checkpoint]) * 4
            if (type(record) is not dict or set(record) != {
                    "dtype", "source_dtype", "shape", "offset_bytes", "nbytes", "sha256", "checkpoint_boundary"}
                    or record.get("dtype") != "F32" or record.get("source_dtype") != "BF16"
                    or record.get("shape") != base.OUTPUT_SHAPES[checkpoint]
                    or record.get("offset_bytes") != cursor or record.get("nbytes") != nbytes
                    or record.get("checkpoint_boundary") != checkpoint):
                raise SoftRouteError(f"soft-route tensor record mismatch: {name}")
            base._digest(record.get("sha256"), f"soft-route tensor {name} SHA-256")
            cursor += nbytes
    if payload["bytes"] != cursor:
        raise SoftRouteError("soft-route tensor payload extent mismatch")


def _new_output_path(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    parent = expanded.parent.resolve(strict=True)
    if not parent.is_dir():
        raise SoftRouteError(f"{label} parent is not a directory")
    final = parent / expanded.name
    if final.is_symlink() or os.path.lexists(final):
        raise SoftRouteError(f"{label} must be a new file: {final}")
    return final


def _validate_diagnostic_stats(value: Any, label: str, checkpoint: str) -> None:
    expected_elements = math.prod(base.OUTPUT_SHAPES[checkpoint])
    if (type(value) is not dict
            or set(value) != {"checkpoint", "elements", "max_abs", "mean_abs", "rms_abs",
                              "outside_atol_rtol", "all_finite", "passed", "atol", "rtol"}
            or value.get("checkpoint") != label or value.get("elements") != expected_elements
            or type(value.get("outside_atol_rtol")) is not int
            or not 0 <= value["outside_atol_rtol"] <= expected_elements
            or value.get("all_finite") is not True
            or value.get("passed") is not (value["outside_atol_rtol"] == 0)
            or value.get("atol") != base.ATOL or value.get("rtol") != base.RTOL
            or any(type(value.get(key)) not in (int, float)
                   or not math.isfinite(float(value[key])) or value[key] < 0
                   for key in ("max_abs", "mean_abs", "rms_abs"))):
        raise SoftRouteError(f"rejected diagnostic comparison record malformed: {label}")


def validate_diagnostic_document(document: Any) -> None:
    """Validate a rejected diagnostic without admitting it as a reference."""
    required = {"schema", "schema_version", "status", "upstream_revision", "parent_reference",
                "bias_manifest", "source", "source_inputs", "geometry", "precision", "execution",
                "payload", "tensors", "runtime_environment", "rejection"}
    if type(document) is not dict or set(document) != required:
        raise SoftRouteError("rejected diagnostic document keyset mismatch")
    if (document.get("schema") != DIAGNOSTIC_SCHEMA
            or document.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION
            or document.get("status") != "rejected"
            or document.get("upstream_revision") != base.PINNED_REVISION):
        raise SoftRouteError("rejected diagnostic schema/status/source revision mismatch")
    parent = document["parent_reference"]
    if (type(parent) is not dict or set(parent) != {"path", "sha256", "schema", "payload_path",
                                                   "payload_basename", "payload_bytes", "payload_sha256"}
            or parent.get("sha256") != PARENT_SHA256 or parent.get("schema") != base.SCHEMA
            or Path(parent.get("payload_basename", "")).name != parent.get("payload_basename")
            or type(parent.get("payload_bytes")) is not int or parent["payload_bytes"] <= 0):
        raise SoftRouteError("rejected diagnostic provenance binding malformed")
    for name in ("path", "payload_path"):
        base._absolute_path(parent.get(name), f"diagnostic parent {name}")
    base._digest(parent.get("payload_sha256"), "diagnostic parent payload SHA-256")
    bias = document["bias_manifest"]
    if (type(bias) is not dict
            or set(bias) != {"path", "sha256", "schema", "case_order", "reference_image", "maps",
                             "bias_f32le_sha256"}
            or bias.get("sha256") != BIAS_SHA256 or bias.get("schema") != BIAS_SCHEMA
            or bias.get("case_order") != list(CASE_ORDER) or bias.get("maps") != _expected_maps()):
        raise SoftRouteError("rejected diagnostic bias binding malformed")
    base._absolute_path(bias.get("path"), "diagnostic bias manifest path")
    base._digest(bias.get("sha256"), "diagnostic bias manifest SHA-256")
    image = bias.get("reference_image")
    if type(image) is not dict or set(image) != {"path", "sha256"}:
        raise SoftRouteError("rejected diagnostic image binding malformed")
    base._absolute_path(image.get("path"), "diagnostic reference image path")
    base._digest(image.get("sha256"), "diagnostic reference image SHA-256")
    hashes = bias.get("bias_f32le_sha256")
    if type(hashes) is not dict or set(hashes) != set(CASE_ORDER):
        raise SoftRouteError("rejected diagnostic bias digest inventory malformed")
    for case in CASE_ORDER:
        base._digest(hashes[case], f"diagnostic {case} bias digest")
    try:
        base._validate_source(document["source"])
        base._validate_source_inputs(document["source_inputs"])
    except base.ReferenceError as error:
        raise SoftRouteError(str(error)) from None
    if document["geometry"] != _geometry_document():
        raise SoftRouteError("rejected diagnostic geometry mismatch")
    expected_precision = {
        **base.PRECISION_CONTRACT,
        "boolean_zero_cases": list(ZERO_CASES),
        "positive_mask_dtype": "BF16 additive MPS SDPA mask; 0 on allowed support, -inf off support, authored BF16 bias on target action-text keys",
        "hard_support": "pinned accepted M11 Boolean mask; no edges added or removed",
        "bias_operation": "F32 manifest values exactly BF16-expanded; one BF16 logit addition before softmax",
    }
    if document["precision"] != expected_precision:
        raise SoftRouteError("rejected diagnostic precision contract mismatch")
    environment = document["runtime_environment"]
    if (type(environment) is not dict
            or set(environment) != {"python_version", "torch_version", "mps_available"}
            or type(environment.get("python_version")) is not str or not environment["python_version"]
            or type(environment.get("torch_version")) is not str or not environment["torch_version"]
            or environment.get("mps_available") is not True):
        raise SoftRouteError("rejected diagnostic runtime environment malformed")
    execution = document["execution"]
    exec_keys = {"case_order", "scope", "soft_route_support", "runtime_receipt", "parent_replay",
                 "accepted_roi_replay", "source_aa", "intervention_effects",
                 "block0_text_valid_equal_all_cases", "zero_repeat_bitwise_exact", "comparison_failures"}
    if (type(execution) is not dict or set(execution) != exec_keys
            or execution.get("case_order") != list(CASE_ORDER)
            or execution.get("block0_text_valid_equal_all_cases") is not True
            or execution.get("zero_repeat_bitwise_exact") is not True):
        raise SoftRouteError("rejected diagnostic execution record malformed")
    validate_runtime_receipt(execution.get("runtime_receipt"))
    expected_gate_sets = {
        "parent_replay": ("zero", set(base.CHECKPOINTS)),
        "accepted_roi_replay": (None, set(base.CHECKPOINTS)),
        "source_aa": (None, set(base.CHECKPOINTS)),
    }
    for gate_name, (case_name, expected_checkpoints) in expected_gate_sets.items():
        gates = execution.get(gate_name)
        if case_name is None:
            if type(gates) is not dict or set(gates) != expected_checkpoints:
                raise SoftRouteError(f"rejected diagnostic {gate_name} inventory mismatch")
            entries = gates
        else:
            if type(gates) is not dict or set(gates) != {case_name} or type(gates[case_name]) is not dict \
                    or set(gates[case_name]) != expected_checkpoints:
                raise SoftRouteError(f"rejected diagnostic {gate_name} inventory mismatch")
            entries = gates[case_name]
        for checkpoint in base.CHECKPOINTS:
            _validate_diagnostic_stats(entries[checkpoint], checkpoint, checkpoint)
    effects = execution.get("intervention_effects")
    if type(effects) is not dict or set(effects) != set(POSITIVE_CASES):
        raise SoftRouteError("rejected diagnostic effect inventory mismatch")
    for case in POSITIVE_CASES:
        if type(effects[case]) is not dict or set(effects[case]) != set(base.CHECKPOINTS):
            raise SoftRouteError(f"rejected diagnostic effect checkpoint inventory mismatch: {case}")
        for checkpoint in base.CHECKPOINTS:
            _validate_diagnostic_stats(effects[case][checkpoint], f"{case}-vs-zero:{checkpoint}", checkpoint)
    failures = execution.get("comparison_failures")
    if type(failures) is not list or not failures:
        raise SoftRouteError("rejected diagnostic must contain comparison failures")
    gate_lookup = {
        "parent_replay": execution["parent_replay"]["zero"],
        "accepted_roi_replay": execution["accepted_roi_replay"],
        "source_aa": execution["source_aa"],
    }
    for failure in failures:
        if (type(failure) is not dict or set(failure) != {"control", "checkpoint", "reason", "bitwise_equal", "stats"}
                or failure.get("control") not in gate_lookup
                or failure.get("checkpoint") not in base.CHECKPOINTS
                or type(failure.get("reason")) is not str or not failure["reason"]
                or failure.get("bitwise_equal") is not False):
            raise SoftRouteError("rejected diagnostic failure record malformed")
        expected_stats = gate_lookup[failure["control"]][failure["checkpoint"]]
        if failure["stats"] != expected_stats:
            raise SoftRouteError("rejected diagnostic failure stats do not match comparison gates")
    rejection = document["rejection"]
    if (type(rejection) is not dict or set(rejection) != {"kind", "failures"}
            or rejection.get("kind") != "exact_baseline_replay_mismatch"
            or rejection.get("failures") != failures):
        raise SoftRouteError("rejected diagnostic reason/failure binding malformed")
    payload = document["payload"]
    if (type(payload) is not dict or set(payload) != {"basename", "bytes", "sha256", "encoding"}
            or type(payload.get("basename")) is not str or Path(payload["basename"]).name != payload["basename"]
            or type(payload.get("bytes")) is not int or payload["bytes"] <= 0
            or payload.get("encoding") != "f32le_bf16_expansion"):
        raise SoftRouteError("rejected diagnostic payload descriptor malformed")
    base._digest(payload.get("sha256"), "rejected diagnostic payload SHA-256")
    tensors = document["tensors"]
    if type(tensors) is not dict or set(tensors) != TENSOR_NAMES:
        raise SoftRouteError("rejected diagnostic tensor inventory mismatch")
    cursor = 0
    for case in CASE_ORDER:
        for checkpoint in base.CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            record = tensors[name]
            nbytes = math.prod(base.OUTPUT_SHAPES[checkpoint]) * 4
            if (type(record) is not dict or set(record) != {
                    "dtype", "source_dtype", "shape", "offset_bytes", "nbytes", "sha256", "checkpoint_boundary"}
                    or record.get("dtype") != "F32" or record.get("source_dtype") != "BF16"
                    or record.get("shape") != base.OUTPUT_SHAPES[checkpoint]
                    or record.get("offset_bytes") != cursor or record.get("nbytes") != nbytes
                    or record.get("checkpoint_boundary") != checkpoint):
                raise SoftRouteError(f"rejected diagnostic tensor record mismatch: {name}")
            base._digest(record.get("sha256"), f"rejected diagnostic tensor {name} SHA-256")
            cursor += nbytes
    if payload["bytes"] != cursor:
        raise SoftRouteError("rejected diagnostic payload extent mismatch")


def _publish_capture_artifact(
    manifest_path: Path, payload_path: Path, checked: Mapping[str, Any],
    build_document: Callable[[Mapping[str, Any], int, str], dict[str, Any]],
    validate_document: Callable[[Any], None],
) -> tuple[Path, Path, dict[str, Any]]:
    import numpy as np

    payload_temp: Path | None = None
    manifest_temp: Path | None = None
    published_payload = False
    try:
        digest = hashlib.sha256()
        records: dict[str, Any] = {}
        cursor = 0
        fd, temp_name = tempfile.mkstemp(prefix=f".{payload_path.name}.", suffix=".tmp", dir=payload_path.parent)
        payload_temp = Path(temp_name)
        with os.fdopen(fd, "wb") as stream:
            for case in CASE_ORDER:
                for checkpoint in base.CHECKPOINTS:
                    name = f"{case}.{checkpoint}"
                    raw = np.asarray(checked[name]).astype("<f4", copy=False).tobytes(order="C")
                    if len(raw) != math.prod(base.OUTPUT_SHAPES[checkpoint]) * 4:
                        raise SoftRouteError(f"soft-route checkpoint byte count mismatch: {name}")
                    records[name] = {
                        "dtype": "F32", "source_dtype": "BF16", "shape": base.OUTPUT_SHAPES[checkpoint],
                        "offset_bytes": cursor, "nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                        "checkpoint_boundary": checkpoint,
                    }
                    stream.write(raw)
                    digest.update(raw)
                    cursor += len(raw)
            stream.flush()
            os.fsync(stream.fileno())
        document = build_document(records, cursor, digest.hexdigest())
        validate_document(document)
        fd, temp_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent)
        manifest_temp = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(payload_temp, payload_path)
        published_payload = True
        os.unlink(payload_temp)
        payload_temp = None
        os.link(manifest_temp, manifest_path)
        os.unlink(manifest_temp)
        manifest_temp = None
        return manifest_path, payload_path, document
    except Exception:
        if published_payload:
            try:
                os.unlink(payload_path)
            except OSError:
                pass
        raise
    finally:
        for temp in (payload_temp, manifest_temp):
            if temp is not None:
                try:
                    os.unlink(temp)
                except OSError:
                    pass


def publish_reference(output_manifest: str | Path, *, parent: Mapping[str, Any], bias: Mapping[str, Any],
                      source: Mapping[str, Any], admitted: Mapping[str, Any], captures: Mapping[str, Any],
                      receipt: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    checked = validate_capture_map(captures)
    gates = _comparison_gates(parent, admitted, checked)
    manifest_path = _new_output_path(Path(output_manifest), "soft-route output manifest")
    payload_path = _new_output_path(
        manifest_path.with_name(f"{manifest_path.stem}.tensors.f32le"), "soft-route output payload",
    )
    receipt = validate_runtime_receipt(dict(receipt))
    return _publish_capture_artifact(
        manifest_path, payload_path, checked,
        lambda records, size, sha: _render_document(
            parent=parent, bias=bias, source=source, admitted=admitted, gates=gates, receipt=receipt,
            tensor_records=records, payload_basename=payload_path.name, payload_bytes=size,
            payload_sha256=sha,
        ),
        validate_reference_document,
    )


def _runtime_environment(torch: Any) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def publish_rejected_diagnostic(
    output_manifest: str | Path, *, parent: Mapping[str, Any], bias: Mapping[str, Any],
    source: Mapping[str, Any], admitted: Mapping[str, Any], captures: Mapping[str, Any],
    receipt: Mapping[str, Any], rejection: ExactReplayRejected, runtime_environment: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Publish terminal CPU captures only after a complete numerical gate rejection."""
    if not isinstance(rejection, ExactReplayRejected):
        raise SoftRouteError("diagnostic publication requires an exact replay rejection")
    checked = validate_capture_map(captures)
    manifest_path = _new_output_path(Path(output_manifest), "soft-route diagnostic manifest")
    payload_path = _new_output_path(
        manifest_path.with_name(f"{manifest_path.stem}.tensors.f32le"), "soft-route diagnostic payload",
    )
    receipt = validate_runtime_receipt(dict(receipt))
    environment = dict(runtime_environment)

    def build_document(records: Mapping[str, Any], size: int, sha: str) -> dict[str, Any]:
        document = _render_document(
            parent=parent, bias=bias, source=source, admitted=admitted, gates=rejection.gates,
            receipt=receipt, tensor_records=records, payload_basename=payload_path.name,
            payload_bytes=size, payload_sha256=sha,
        )
        document["schema"] = DIAGNOSTIC_SCHEMA
        document["schema_version"] = DIAGNOSTIC_SCHEMA_VERSION
        document["status"] = "rejected"
        document["runtime_environment"] = environment
        document["execution"]["comparison_failures"] = rejection.failures
        document["rejection"] = {
            "kind": "exact_baseline_replay_mismatch", "failures": rejection.failures,
        }
        return document

    return _publish_capture_artifact(
        manifest_path, payload_path, checked, build_document, validate_diagnostic_document,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-manifest", type=Path, default=DEFAULT_BIAS_MANIFEST)
    parser.add_argument("--bias-sha256", default=BIAS_SHA256)
    parser.add_argument("--parent-reference", type=Path, default=DEFAULT_PARENT_REFERENCE)
    parser.add_argument("--parent-sha256", default=PARENT_SHA256)
    parser.add_argument("--frozen-inputs", type=Path, default=base.DEFAULT_FROZEN_INPUTS)
    parser.add_argument("--roi-manifest", type=Path, default=base.DEFAULT_ROI_MANIFEST)
    parser.add_argument("--continuation-manifest", type=Path, default=base.DEFAULT_CONTINUATION_MANIFEST)
    parser.add_argument("--cache-index", type=Path, default=base.DEFAULT_CACHE_INDEX)
    parser.add_argument("--upstream", type=Path, default=base.DEFAULT_UPSTREAM)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument(
        "--diagnostic-manifest", type=Path,
        help="optional rejected-only capture artifact when an exact baseline replay gate fails",
    )
    parser.add_argument("--admit-only", action="store_true", help="authenticate parent, inputs, and bias without importing torch")
    parser.add_argument("--validate-source-mask", action="store_true", help="authenticate live M11 mask on CPU")
    parser.add_argument("--mps-mask-preflight", action="store_true", help="run only the small MPS additive-mask control")
    return parser


def _run_source(args: argparse.Namespace, admitted: Mapping[str, Any], parent: Mapping[str, Any],
                bias: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any], dict[str, Any], str]:
    if args.output_manifest is None:
        raise SoftRouteError("--output-manifest is required for source execution")
    manifest_path = _new_output_path(args.output_manifest, "soft-route output manifest")
    payload_path = _new_output_path(
        manifest_path.with_name(f"{manifest_path.stem}.tensors.f32le"), "soft-route output payload",
    )
    diagnostic_path: Path | None = None
    diagnostic_payload_path: Path | None = None
    if args.diagnostic_manifest is not None:
        diagnostic_path = _new_output_path(args.diagnostic_manifest, "soft-route diagnostic manifest")
        diagnostic_payload_path = _new_output_path(
            diagnostic_path.with_name(f"{diagnostic_path.stem}.tensors.f32le"),
            "soft-route diagnostic payload",
        )
        planned_paths = {manifest_path, payload_path, diagnostic_path, diagnostic_payload_path}
        if len(planned_paths) != 4:
            raise SoftRouteError("accepted and diagnostic output paths must be distinct")
    try:
        import torch
    except ImportError as error:
        raise SoftRouteError(f"PyTorch is required for pinned source execution: {error}") from error
    if not torch.backends.mps.is_available():
        raise SoftRouteError("pinned soft-route reference requires an available MPS device")
    try:
        masks, mask_receipt = base.build_pinned_source_masks(torch, args.upstream)
        validate_bias_support(bias["vectors"], masks["M11"])
        if (mask_receipt.get("query_rows") != QUERY_ROWS or mask_receipt.get("key_rows") != KEY_ROWS
                or mask_receipt.get("chunks_per_case") != math.ceil(QUERY_ROWS / CHUNK)):
            raise SoftRouteError("pinned source mask geometry receipt mismatch")
        model = base.authenticate_model_sources(args.cache_index, args.upstream)
        runtime_admitted = dict(admitted)
        runtime_admitted["model"] = model
        runtime = __import__("hy_nomas_dit_route_factorial_runtime")
        import numpy as np
        runtime_biases = {
            case: np.asarray(values, dtype=np.float32)
            for case, values in bias["vectors"].items()
        }
        captures, receipt = runtime.run(
            runtime_admitted, args.cache_index, args.upstream, masks,
            cases=CASE_ORDER, biases=runtime_biases,
        )
        receipt = validate_runtime_receipt(receipt)
        source = base.source_document(model)
        capture_map = validate_capture_map(captures)
        try:
            manifest_path, payload_path, document = publish_reference(
                manifest_path, parent=parent, bias=bias, source=source, admitted=admitted,
                captures=capture_map, receipt=receipt,
            )
            return manifest_path, payload_path, document, receipt, "accepted"
        except ExactReplayRejected as rejection:
            if diagnostic_path is None:
                raise
            diagnostic_path, diagnostic_payload_path, diagnostic = publish_rejected_diagnostic(
                diagnostic_path, parent=parent, bias=bias, source=source, admitted=admitted,
                captures=capture_map, receipt=receipt, rejection=rejection,
                runtime_environment=_runtime_environment(torch),
            )
            return diagnostic_path, diagnostic_payload_path, diagnostic, receipt, "rejected_diagnostic"
    except (base.ReferenceError, RuntimeError) as error:
        if isinstance(error, SoftRouteError):
            raise
        raise SoftRouteError(f"pinned source soft-route execution failed: {type(error).__name__}: {error}") from None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.diagnostic_manifest is not None and args.output_manifest is None:
            raise SoftRouteError("--diagnostic-manifest requires --output-manifest")
        if args.diagnostic_manifest is not None and (args.admit_only or args.validate_source_mask
                                                     or args.mps_mask_preflight):
            raise SoftRouteError("--diagnostic-manifest is only available for source execution")
        if args.mps_mask_preflight:
            if args.output_manifest or args.diagnostic_manifest or args.admit_only or args.validate_source_mask:
                raise SoftRouteError("--mps-mask-preflight cannot be combined with other run modes")
            runtime = __import__("hy_nomas_dit_route_factorial_runtime")
            receipt = runtime.run_additive_mask_preflight()
            print(json.dumps({"schema": SCHEMA, "status": "mps_mask_preflight_passed", "receipt": receipt}, sort_keys=True))
            return 0
        bias = authenticate_bias_manifest(args.bias_manifest, args.bias_sha256)
        parent = authenticate_parent_reference(args.parent_reference, args.parent_sha256)
        admitted = base.authenticate_inputs(args.frozen_inputs, args.roi_manifest, args.continuation_manifest)
        validate_bias_support(bias["vectors"])
        if args.admit_only:
            if args.validate_source_mask or args.output_manifest or args.diagnostic_manifest:
                raise SoftRouteError("--admit-only cannot be combined with another execution option")
            print(json.dumps({
                "schema": SCHEMA, "status": "admitted_cpu_only", "model_launched": False,
                "mps_device_used": False, "bias_sha256": bias["sha256"],
                "parent_sha256": parent["sha256"], "case_order": list(CASE_ORDER),
                "geometry": {"video_rows": VIDEO_ROWS, "query_rows": QUERY_ROWS, "key_rows": KEY_ROWS},
                "bias_support": validate_bias_support(bias["vectors"]),
            }, sort_keys=True))
            return 0
        if args.validate_source_mask:
            if args.output_manifest or args.diagnostic_manifest:
                raise SoftRouteError("--validate-source-mask cannot be combined with output manifests")
            import torch
            masks, receipt = base.build_pinned_source_masks(torch, args.upstream)
            support = validate_bias_support(bias["vectors"], masks["M11"])
            if any(chunk.device.type != "cpu" for chunk in masks["M11"]):
                raise SoftRouteError("source M11 mask validation escaped CPU")
            print(json.dumps({"schema": SCHEMA, "status": "m11_support_validated_cpu_only",
                              "mps_device_used": False, "mask_receipt": receipt,
                              "bias_support": support}, sort_keys=True))
            return 0
        manifest, payload, document, receipt, source_status = _run_source(args, admitted, parent, bias)
        if source_status == "rejected_diagnostic":
            print(json.dumps({
                "schema": DIAGNOSTIC_SCHEMA, "status": "rejected_diagnostic_published",
                "reference_qualified": False, "model_launched": True, "mps_device_used": True,
                "manifest": str(manifest), "manifest_sha256": base.sha256_file(manifest),
                "payload": str(payload), "payload_sha256": document["payload"]["sha256"],
                "checkpoint_count": len(document["tensors"]), "runtime_receipt": receipt,
                "comparison_failures": document["rejection"]["failures"],
            }, sort_keys=True))
            return 2
        print(json.dumps({
            "schema": SCHEMA, "status": "source_reference_completed", "model_launched": True,
            "mps_device_used": True, "manifest": str(manifest), "manifest_sha256": base.sha256_file(manifest),
            "payload": str(payload), "payload_sha256": document["payload"]["sha256"],
            "checkpoint_count": len(document["tensors"]), "runtime_receipt": receipt,
        }, sort_keys=True))
        return 0
    except (SoftRouteError, base.ReferenceError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected",
                          "error": f"{type(error).__name__}: {error}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
