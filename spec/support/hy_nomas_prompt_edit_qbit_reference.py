#!/usr/bin/env python3
"""Authenticate the compact-checkpoint prompt-edit continuation matrix.

This CPU-only checker revalidates the prior exact prompt matrix, then checks a
fresh QBit A/B/C matrix. It authenticates both HNQC containers, their embedded
QBit manifests, every row header and compact payload extent, state lineage,
receipts, and F32 captures. It does not run the model, decode video, or assess
visual quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any

import hy_nomas_compact_resume_reference as compact
import hy_nomas_prompt_edit_resume_reference as prompt
import hy_nomas_sequence_resume_reference as resume
import hy_nomas_cache_qbit_compare as qbit_compare


INDEX_SCHEMA = "cogni_ml.hy_nomas_prompt_edit_qbit_resume_matrix/v1"
REPORT_SCHEMA = "cogni_ml.hy_nomas_prompt_edit_qbit_resume_reference/v1"
QBIT_CONTRACT_SCHEMA = "hy-nomas-cache-qbit-checkpoint-v1"
QBIT_MANIFEST_SCHEMA = "hy-nomas-cache-qbit-checkpoint-manifest-v1"
QBIT_RECEIPT_SCHEMA = prompt.EDIT_RECEIPT_SCHEMA
BASE_RECEIPT_SCHEMA = prompt.BASE_RECEIPT_SCHEMA
HNRS_SCHEMA = prompt.HNRS_SCHEMA
HNRS_MAGIC = prompt.HNRS_MAGIC
HNRS_MAX_HEADER_BYTES = prompt.HNRS_MAX_HEADER_BYTES
HNRS_PAYLOAD_OFFSET = prompt.HNRS_PAYLOAD_OFFSET
HEADER_BYTES = 16
ROW_VALUES = 128
ROW_SLOT_BYTES = 512
TIERS = {4: ("p4", 64), 5: ("p5", 80), 8: ("p8", 128), 32: ("f32", 512)}
SEGMENTS = prompt.SEGMENTS
TARGET_SEGMENT = prompt.TARGET_SEGMENT
STEPS_PER_SEGMENT = prompt.STEPS_PER_SEGMENT
SEGMENT_LATENTS = prompt.SEGMENT_LATENTS
TIMELINE_SHAPE = prompt.TIMELINE_SHAPE
STEP_SHAPE = prompt.STEP_SHAPE
TIMELINE_BYTES = prompt.TIMELINE_BYTES
STEP_BYTES = prompt.STEP_BYTES
ATOL = RTOL = 2e-5
ARM_INTERVALS = {
    "parent_qbit_a": (0, 1),
    "qbit_control": (1, 3),
    "qbit_restart_chain": ((1, 2), (2, 3)),
}
IDENTITY_FIELDS = (
    "inputs", "inputs_sha256", "payload_sha256", "conditioning_sha256",
    "model_index_sha256", "upstream_revision", "segments",
    "steps_per_segment", "timeline_shape", "step_shape",
)
METRIC_FIELDS = prompt.METRIC_FIELDS
PINNED_SOURCE_PATHS = (
    "Makefile",
    "examples/hy_nomas_compact_sequence_probe.cr",
    "examples/support/hy_nomas_sequence.cr",
    "src/ml/video/hy_nomas_segment_edit.cr",
    "src/ml/video/hy_nomas_continuation_history.cr",
    "examples/hy_nomas_prompt_history_probe.cr",
    "examples/support/hy_nomas_prompt_history_probe.cr",
    "spec/support/hy_nomas_prompt_history_reference.py",
    "examples/support/wan_memory_guard.py",
    "examples/support/hy_nomas_cache_qbit.cr",
    "examples/support/hy_nomas_cache_qbit.metal",
    "examples/support/hy_nomas_cache_qbit_checkpoint.cr",
    "src/ml/gguf/qwen_qbit_gaussian_codec.cr",
)
TEXT_PROFILE = [557, 585, 557]
CACHE_PROFILE = [True, True, True]
BASE_TEXT_PROFILE = [557, 557, 557]
BASE_CACHE_PROFILE = [True, False, False]


class PromptEditQBitReferenceError(compact.CompactResumeReferenceError):
    """A matrix, QBit checkpoint, receipt, or capture violates its contract."""


def _fail(message: str) -> None:
    raise PromptEditQBitReferenceError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _qbit_identity_json() -> str:
    """Mirror Crystal's ordered QBit checkpoint identity JSON."""
    identity = {
        "schema": QBIT_CONTRACT_SCHEMA,
        "policy": "hy-nomas-cache-qbit-row-error-v1",
        "row_values": ROW_VALUES,
        "header_bytes_per_row": HEADER_BYTES,
        "gpu_payload_slot_bytes_per_row": ROW_SLOT_BYTES,
        "disk_layout": "strided-gpu-512-compact-disk-v1",
        "p4_max_error_over_sigma": 0.50,
        "p4_rms_error_over_sigma": 0.12,
        "p5_max_error_over_sigma": 0.25,
        "p5_rms_error_over_sigma": 0.06,
        "p8_max_error_over_sigma": 0.10,
        "p8_rms_error_over_sigma": 0.02,
        "escape": "exact-f32",
        "raw_centers": "ML::GGUF::QwenQBitGaussianCodec::POSITIVE_LEVELS",
        "reconstruction": "ML::GGUF::QwenQBitGaussianCodec::reconstruct_raw_code",
        "codebook_sha256": qbit_compare.CODEBOOK_SHA256,
    }
    return _canonical_json(identity)


def _history_rows_from_contract(
    contract: dict[str, Any], boundary: int,
    segment_text_tokens: list[int], segment_cache_text: list[bool],
) -> int:
    video_tokens = compact._int(contract.get("video_tokens"), "base contract video_tokens", minimum=1)
    if len(segment_text_tokens) != SEGMENTS or len(segment_cache_text) != SEGMENTS:
        _fail("QBit segment text profile is malformed")
    return sum(
        video_tokens + (segment_text_tokens[i] if segment_cache_text[i] else 0)
        for i in range(boundary)
    )


def _expected_inventory_from_contract(
    contract: dict[str, Any], boundary: int,
    segment_text_tokens: list[int], segment_cache_text: list[bool],
) -> dict[str, int]:
    layers = compact._int(contract.get("layers"), "base contract layers", minimum=1)
    heads = compact._int(contract.get("heads"), "base contract heads", minimum=1)
    head_dim = compact._int(contract.get("head_dim"), "base contract head_dim", minimum=1)
    video_tokens = compact._int(contract.get("video_tokens"), "base contract video_tokens", minimum=1)
    history_rows = _history_rows_from_contract(
        contract, boundary, segment_text_tokens, segment_cache_text
    )
    exact_bytes = history_rows * heads * head_dim * 4
    if exact_bytes % ROW_SLOT_BYTES:
        _fail("QBit cache inventory is not divisible into 128-value rows")
    inventory = {"timeline": TIMELINE_BYTES}
    for layer in range(layers):
        inventory[f"history.layer{layer}.key"] = exact_bytes
        inventory[f"history.layer{layer}.value"] = exact_bytes
    return inventory


def _parse_stats(document: Any, expected_inventory: dict[str, int], label: str) -> dict[str, Any]:
    """Validate native QBit stats against expected inventory and byte formulas."""
    if type(document) is not dict or set(document) != {
        "records", "logical_packed_bytes", "exact_bytes", "tier_counts", "invalid_rows"
    }:
        _fail(f"{label} QBit stats fields are unsupported")
    if compact._int(document.get("invalid_rows"), f"{label} invalid_rows") != 0:
        _fail(f"{label} QBit stats contain invalid rows")
    expected_names = set(expected_inventory) - {"timeline"}
    raw_records = document.get("records")
    if type(raw_records) is not list or len(raw_records) != len(expected_names):
        _fail(f"{label} QBit stats record count does not match the expected inventory")
    records: dict[str, dict[str, Any]] = {}
    all_tiers = {name: 0 for name in ("p4", "p5", "p8", "f32")}
    logical_total = exact_total = 0
    for raw in raw_records:
        if type(raw) is not dict or set(raw) != {
            "name", "count", "rows", "logical_packed_bytes", "exact_bytes",
            "payload_bytes", "tier_counts", "invalid_rows",
        }:
            _fail(f"{label} QBit record stats fields are unsupported")
        name = raw.get("name")
        if type(name) is not str or name not in expected_names or name in records:
            _fail(f"{label} QBit record stats name is unexpected or duplicated")
        exact_bytes = expected_inventory[name]
        rows = exact_bytes // ROW_SLOT_BYTES
        counts = raw.get("tier_counts")
        if type(counts) is not dict or set(counts) != set(all_tiers):
            _fail(f"{label} QBit record tier counts are malformed")
        counts = {key: compact._int(counts[key], f"{label} {name} {key}") for key in all_tiers}
        payload_bytes = sum(counts[tier_name] * payload_size for tier_name, payload_size in TIERS.values())
        packed = rows * HEADER_BYTES + payload_bytes
        if (
            compact._int(raw.get("count"), f"{label} {name} count") != rows * ROW_VALUES
            or compact._int(raw.get("rows"), f"{label} {name} rows") != rows
            or compact._int(raw.get("exact_bytes"), f"{label} {name} exact_bytes") != exact_bytes
            or compact._int(raw.get("payload_bytes"), f"{label} {name} payload_bytes") != payload_bytes
            or compact._int(raw.get("logical_packed_bytes"), f"{label} {name} logical_packed_bytes") != packed
            or compact._int(raw.get("invalid_rows"), f"{label} {name} invalid_rows") != 0
            or sum(counts.values()) != rows
        ):
            _fail(f"{label} QBit record byte/tier accounting is inconsistent for {name}")
        record = dict(raw)
        record["tier_counts"] = counts
        records[name] = record
        logical_total += packed
        exact_total += exact_bytes
        for tier in all_tiers:
            all_tiers[tier] += counts[tier]
    aggregate = document.get("tier_counts")
    if type(aggregate) is not dict or set(aggregate) != set(all_tiers):
        _fail(f"{label} QBit aggregate tier counts are malformed")
    aggregate = {key: compact._int(aggregate[key], f"{label} aggregate {key}") for key in all_tiers}
    if (
        compact._int(document.get("logical_packed_bytes"), f"{label} logical_packed_bytes") != logical_total
        or compact._int(document.get("exact_bytes"), f"{label} exact_bytes") != exact_total
        or aggregate != all_tiers
    ):
        _fail(f"{label} QBit aggregate byte/tier accounting is inconsistent")
    return {
        "records": records,
        "logical_packed_bytes": logical_total,
        "exact_bytes": exact_total,
        "tier_counts": all_tiers,
        "invalid_rows": 0,
    }


def _validate_row_record(
    data: bytes, *, expected_exact_bytes: int, label: str,
) -> dict[str, Any]:
    if expected_exact_bytes % ROW_SLOT_BYTES:
        _fail(f"{label} expected cache bytes are not whole rows")
    rows = expected_exact_bytes // ROW_SLOT_BYTES
    header_bytes = rows * HEADER_BYTES
    if len(data) < header_bytes:
        _fail(f"{label} record is shorter than its row headers")
    cursor = 0
    payload_sizes: list[int] = []
    counts = {"p4": 0, "p5": 0, "p8": 0, "f32": 0}
    for row in range(rows):
        mean, sigma, payload_offset, tier_code = struct.unpack_from("<ffII", data, row * HEADER_BYTES)
        if not math.isfinite(mean) or not math.isfinite(sigma) or sigma < 0:
            _fail(f"{label} row moments are nonfinite or negative")
        if tier_code not in TIERS:
            _fail(f"{label} row tier is unsupported")
        tier_name, payload_bytes = TIERS[tier_code]
        if tier_code not in (4, 32) and sigma <= 0:
            _fail(f"{label} quantized row sigma must be positive")
        if payload_offset != cursor:
            _fail(f"{label} payload offsets are not canonical")
        cursor += payload_bytes
        payload_sizes.append(payload_bytes)
        counts[tier_name] += 1
    if header_bytes + cursor != len(data):
        _fail(f"{label} compact payload coverage differs from its row headers")
    payload_offset = header_bytes
    for tier_code, size in zip(
        (struct.unpack_from("<I", data, row * HEADER_BYTES + 12)[0] for row in range(rows)),
        payload_sizes,
    ):
        if tier_code == 32:
            for (value,) in struct.iter_unpack("<f", data[payload_offset:payload_offset + size]):
                if not math.isfinite(value):
                    _fail(f"{label} exact-f32 escape has a nonfinite value")
        payload_offset += size
    return {
        "rows": rows,
        "exact_bytes": expected_exact_bytes,
        "payload_bytes": cursor,
        "logical_packed_bytes": header_bytes + cursor,
        "tier_counts": counts,
    }


def _stable_stat(path: Path) -> tuple[int, int, int, int]:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        _fail("QBit checkpoint must be a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _parse_qbit_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_file_bytes: int,
    expected_payload_bytes: int,
    expected_base_contract: str,
    expected_inventory: dict[str, int],
    expected_timeline_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Authenticate HNRS + QBit container, all records, stats and row headers."""
    before = _stable_stat(path)
    if before[2] != expected_file_bytes:
        _fail(f"{label} QBit checkpoint size differs from its receipt")
    if path.is_symlink():
        _fail(f"{label} QBit checkpoint cannot be a symlink")
    outer_hash = hashlib.sha256()
    try:
        stream = path.open("rb")
    except OSError as error:
        raise PromptEditQBitReferenceError(f"cannot open {label} QBit checkpoint") from error
    try:
        prefix = stream.read(HNRS_PAYLOAD_OFFSET)
        if len(prefix) != HNRS_PAYLOAD_OFFSET:
            _fail(f"{label} HNRS prefix is truncated")
        outer_hash.update(prefix)
        if prefix[:len(HNRS_MAGIC)] != HNRS_MAGIC:
            _fail(f"{label} HNRS magic mismatch")
        header_length = struct.unpack_from("<I", prefix, len(HNRS_MAGIC))[0]
        if not 0 < header_length <= HNRS_MAX_HEADER_BYTES:
            _fail(f"{label} HNRS header length is invalid")
        header_start = len(HNRS_MAGIC) + 4
        header_stop = header_start + header_length
        if any(prefix[header_stop:]):
            _fail(f"{label} HNRS header padding is not zero")
        try:
            header = json.loads(
                prefix[header_start:header_stop].decode("utf-8"),
                object_pairs_hook=resume._unique_object,
                parse_constant=resume._reject_json_constant,
            )
        except Exception as error:
            raise PromptEditQBitReferenceError(f"{label} HNRS header JSON is malformed") from error
        if type(header) is not dict or set(header) != {"schema", "schema_version", "contract", "records"}:
            _fail(f"{label} HNRS header fields are unsupported")
        if header.get("schema") != HNRS_SCHEMA or type(header.get("schema_version")) is not int or header["schema_version"] != 1:
            _fail(f"{label} HNRS schema is unsupported")
        raw_records = header.get("records")
        if type(raw_records) is not list or not raw_records:
            _fail(f"{label} HNRS record inventory is empty")
        names: list[str] = []
        cursor = 0
        records: dict[str, dict[str, Any]] = {}
        for record in raw_records:
            if type(record) is not dict or set(record) != {"name", "nbytes", "offset", "sha256"}:
                _fail(f"{label} HNRS record fields are malformed")
            name = record.get("name")
            nbytes = record.get("nbytes")
            offset = record.get("offset")
            if type(name) is not str or not name or type(nbytes) is not int or nbytes < 0 or type(offset) is not int or offset != cursor:
                _fail(f"{label} HNRS record name or extent is malformed")
            compact._sha(record.get("sha256"), f"{label} HNRS record SHA")
            records[name] = dict(record)
            names.append(name)
            cursor += nbytes
        if names != sorted(names) or len(names) != len(set(names)):
            _fail(f"{label} HNRS records are not unique canonical order")
        if cursor != expected_payload_bytes or HNRS_PAYLOAD_OFFSET + cursor != expected_file_bytes:
            _fail(f"{label} HNRS record extents differ from receipt accounting")
        expected_names = set(expected_inventory) | {"codec.manifest"}
        if set(records) != expected_names:
            _fail(f"{label} HNRS record inventory does not match expected cache inventory")
        for name, exact_bytes in expected_inventory.items():
            if name == "timeline" and records[name]["nbytes"] != exact_bytes:
                _fail(f"{label} timeline HNRS extent differs from its expected F32 data")
            if name != "timeline":
                rows = exact_bytes // ROW_SLOT_BYTES
                if records[name]["nbytes"] < rows * HEADER_BYTES or records[name]["nbytes"] > rows * (HEADER_BYTES + ROW_SLOT_BYTES):
                    _fail(f"{label} QBit record extent is outside its row inventory bounds")

        try:
            wrapper = json.loads(
                header.get("contract"), object_pairs_hook=resume._unique_object,
                parse_constant=resume._reject_json_constant,
            )
        except Exception as error:
            raise PromptEditQBitReferenceError(f"{label} QBit contract JSON is malformed") from error
        if type(wrapper) is not dict or set(wrapper) != {"schema", "base_contract", "identity_json"}:
            _fail(f"{label} QBit contract fields are unsupported")
        if wrapper.get("schema") != QBIT_CONTRACT_SCHEMA:
            _fail(f"{label} QBit checkpoint contract schema mismatch")
        if wrapper.get("base_contract") != expected_base_contract:
            _fail(f"{label} QBit checkpoint is bound to a stale or different base boundary")
        if wrapper.get("identity_json") != _qbit_identity_json():
            _fail(f"{label} QBit checkpoint codec identity mismatch")
        try:
            base_contract_document = json.loads(
                wrapper["base_contract"], object_pairs_hook=resume._unique_object,
                parse_constant=resume._reject_json_constant,
            )
        except Exception as error:
            raise PromptEditQBitReferenceError(f"{label} base boundary contract JSON is malformed") from error
        if type(base_contract_document) is not dict:
            _fail(f"{label} base boundary contract must be an object")

        observed_stats: dict[str, Any] = {}
        manifest_raw: bytes | None = None
        timeline_digest: str | None = None
        for name in names:
            record = records[name]
            remaining = record["nbytes"]
            digest = hashlib.sha256()
            chunks: list[bytes] | None = [] if name == "codec.manifest" else None
            record_data: bytearray | None = bytearray() if name.startswith("history.") else None
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    _fail(f"{label} HNRS record {name} is truncated")
                remaining -= len(chunk)
                outer_hash.update(chunk)
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
                if record_data is not None:
                    record_data.extend(chunk)
            actual_digest = digest.hexdigest()
            if actual_digest != record["sha256"]:
                _fail(f"{label} HNRS record SHA mismatch for {name}")
            if name == "timeline":
                timeline_digest = actual_digest
            elif name == "codec.manifest":
                manifest_raw = b"".join(chunks or [])
                if len(manifest_raw) > 64 * 1024:
                    _fail(f"{label} QBit codec manifest exceeds its bound")
            else:
                expected_exact = expected_inventory[name]
                observed_stats[name] = _validate_row_record(
                    bytes(record_data or b""), expected_exact_bytes=expected_exact,
                    label=f"{label} {name}",
                )
        if stream.read(1):
            _fail(f"{label} HNRS file has trailing bytes")
    finally:
        stream.close()
    after = _stable_stat(path)
    if before != after:
        _fail(f"{label} QBit checkpoint changed during authentication")
    if outer_hash.hexdigest() != expected_sha256:
        _fail(f"{label} QBit checkpoint file SHA mismatch")
    if timeline_digest != expected_timeline_sha256:
        _fail(f"{label} QBit checkpoint timeline differs from its authenticated capture")
    if manifest_raw is None:
        _fail(f"{label} QBit checkpoint codec manifest is missing")
    try:
        manifest = json.loads(
            manifest_raw.decode("utf-8"), object_pairs_hook=resume._unique_object,
            parse_constant=resume._reject_json_constant,
        )
    except Exception as error:
        raise PromptEditQBitReferenceError(f"{label} QBit codec manifest JSON is malformed") from error
    if type(manifest) is not dict or set(manifest) != {"schema", "identity_json", "timeline_bytes", "stats_json"}:
        _fail(f"{label} QBit codec manifest fields are unsupported")
    if manifest.get("schema") != QBIT_MANIFEST_SCHEMA or manifest.get("identity_json") != _qbit_identity_json():
        _fail(f"{label} QBit codec manifest identity mismatch")
    if compact._int(manifest.get("timeline_bytes"), f"{label} codec timeline_bytes") != expected_inventory["timeline"]:
        _fail(f"{label} QBit codec timeline extent mismatch")
    stats_json = manifest.get("stats_json")
    if type(stats_json) is not str:
        _fail(f"{label} QBit stats_json must be a string")
    try:
        stats_doc = json.loads(
            stats_json, object_pairs_hook=resume._unique_object,
            parse_constant=resume._reject_json_constant,
        )
    except Exception as error:
        raise PromptEditQBitReferenceError(f"{label} QBit stats JSON is malformed") from error
    stats = _parse_stats(stats_doc, expected_inventory, label)
    if set(observed_stats) != set(stats["records"]):
        _fail(f"{label} observed cache rows do not cover the QBit stats inventory")
    for name, observed in observed_stats.items():
        record_stats = stats["records"][name]
        if any(record_stats[key] != observed[key] for key in (
            "rows", "exact_bytes", "payload_bytes", "logical_packed_bytes", "tier_counts"
        )):
            _fail(f"{label} QBit row headers disagree with stats for {name}")
    return {
        "sha256": expected_sha256,
        "file_bytes": expected_file_bytes,
        "payload_bytes": expected_payload_bytes,
        "base_contract": wrapper["base_contract"],
        "base_contract_document": base_contract_document,
        "base_contract_sha256": _sha_bytes(wrapper["base_contract"].encode("utf-8")),
        "codec_identity_sha256": _sha_bytes(wrapper["identity_json"].encode("utf-8")),
        "timeline_sha256": timeline_digest,
        "record_count": len(records),
        "cache_records": len(observed_stats),
        "codec_stats": stats,
        "records": records,
    }


def _stats_from_codec_report(
    report: Any, expected_inventory: dict[str, int], label: str, *, kind: str,
) -> dict[str, Any]:
    if type(report) is not dict:
        _fail(f"{label} codec report must be an object")
    if compact._int(report.get("invalid_rows"), f"{label} invalid_rows") != 0:
        _fail(f"{label} codec report contains invalid rows")
    if kind == "restore":
        return _parse_stats(report, expected_inventory, label)

    if kind == "roundtrip":
        if report.get("schema") != "hy-nomas-cache-qbit-report-v1":
            _fail(f"{label} roundtrip report schema mismatch")
        policy = report.get("policy")
        expected_policy = {
            "name": qbit_compare.POLICY["name"],
            "semantics": "experimental quality selector; thresholds are not upstream exactness tolerances",
            "row_values": ROW_VALUES,
            "header_bytes_per_row": HEADER_BYTES,
            "p4_max_error_over_sigma": 0.50,
            "p4_rms_error_over_sigma": 0.12,
            "p5_max_error_over_sigma": 0.25,
            "p5_rms_error_over_sigma": 0.06,
            "p8_max_error_over_sigma": 0.10,
            "p8_rms_error_over_sigma": 0.02,
            "escape": "exact-f32",
        }
        if policy != expected_policy:
            _fail(f"{label} roundtrip selector policy mismatch")
        if report.get("codebook") != {
            "raw_centers": "ML::GGUF::QwenQBitGaussianCodec::POSITIVE_LEVELS",
            "reconstruction": "ML::GGUF::QwenQBitGaussianCodec::reconstruct_raw_code",
            "sha256": qbit_compare.CODEBOOK_SHA256,
        }:
            _fail(f"{label} roundtrip codebook identity mismatch")
        compact._int(report.get("resident_bytes"), f"{label} resident_bytes", minimum=1)
        compact._int(report.get("scratch_capacity_bytes"), f"{label} scratch_capacity_bytes")
        stats_doc = {key: report.get(key) for key in (
            "records", "logical_packed_bytes", "exact_bytes", "tier_counts", "invalid_rows"
        )}
        # The live roundtrip report omits payload_bytes in per-record stats;
        # reconstruct it from its packed byte count before applying the shared
        # exact inventory/accounting validator.
        records = stats_doc.get("records")
        if type(records) is not list:
            _fail(f"{label} roundtrip record stats are missing")
        normalized_records = []
        for record in records:
            if type(record) is not dict:
                _fail(f"{label} roundtrip record stats are malformed")
            adjusted = dict(record)
            rows = compact._int(adjusted.get("rows"), f"{label} roundtrip rows")
            packed = compact._int(adjusted.get("logical_packed_bytes"), f"{label} roundtrip packed bytes")
            adjusted["payload_bytes"] = packed - rows * HEADER_BYTES
            normalized_records.append(adjusted)
        stats_doc["records"] = normalized_records
        return _parse_stats(stats_doc, expected_inventory, label)

    if kind != "pack":
        _fail(f"{label} codec report kind is unsupported")
    if report.get("schema") != QBIT_CONTRACT_SCHEMA or report.get("identity_json") != _qbit_identity_json():
        _fail(f"{label} checkpoint pack report identity mismatch")
    if report.get("layout") != "strided-gpu-512-compact-disk-v1":
        _fail(f"{label} checkpoint pack layout mismatch")
    if report.get("disk_bytes_known") is not True:
        _fail(f"{label} checkpoint pack report does not know its disk extent")
    compact._int(report.get("disk_bytes"), f"{label} disk_bytes", minimum=1)
    compact._int(report.get("resident_bytes"), f"{label} resident_bytes", minimum=1)
    if compact._int(report.get("payload_scratch_bytes"), f"{label} payload_scratch_bytes") != 0:
        _fail(f"{label} checkpoint pack must not retain payload scratch")
    stats_json = report.get("statistics_json")
    if type(stats_json) is not str:
        _fail(f"{label} checkpoint pack statistics_json must be a string")
    try:
        stats_doc = json.loads(
            stats_json, object_pairs_hook=resume._unique_object,
            parse_constant=resume._reject_json_constant,
        )
    except Exception as error:
        raise PromptEditQBitReferenceError(f"{label} checkpoint pack stats JSON is malformed") from error
    stats = _parse_stats(stats_doc, expected_inventory, label)
    for key in ("records", "logical_packed_bytes", "exact_bytes", "tier_counts", "invalid_rows"):
        if report.get(key) != stats_doc.get(key):
            _fail(f"{label} checkpoint pack duplicate stats field {key} disagrees")
    if compact._int(report.get("logical_packed_kv_bytes"), f"{label} logical_packed_kv_bytes") != stats["logical_packed_bytes"]:
        _fail(f"{label} checkpoint pack logical byte count mismatch")
    return stats


def _validate_receipt(
    receipt: dict[str, Any], arm: str, interval: tuple[int, int],
    exact_report: dict[str, Any], prompt_identity: dict[str, Any],
    prompt_inputs: dict[str, Any], base_prompt_sha256: str,
) -> dict[str, Any]:
    label = f"{arm} receipt"
    start, stop = interval
    edited = arm != "parent_qbit_a"
    schema = QBIT_RECEIPT_SCHEMA if edited else BASE_RECEIPT_SCHEMA
    if receipt.get("schema") != schema:
        _fail(f"{label} schema does not match its prompt-edit treatment")
    if receipt.get("mode") != ("save" if start == 0 else "resume"):
        _fail(f"{label} mode does not match its interval")
    for key, expected in (
        ("segments", SEGMENTS), ("start_segment", start),
        ("stop_after_segments", stop), ("resume_next_segment", start),
        ("steps_per_segment", STEPS_PER_SEGMENT),
    ):
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            _fail(f"{label} {key} does not match its interval")
    compact._int(receipt.get("pid"), f"{label} pid", minimum=1)
    for key in (
        "executable_sha256", "inputs_sha256", "payload_sha256",
        "conditioning_sha256", "model_index_sha256",
    ):
        compact._sha(receipt.get(key), f"{label} {key}")
    if type(receipt.get("inputs")) is not str or not receipt["inputs"]:
        _fail(f"{label} inputs must name its original continuation manifest")
    if type(receipt.get("upstream_revision")) is not str or not receipt["upstream_revision"]:
        _fail(f"{label} upstream_revision must be non-empty")

    expected_tokens = TEXT_PROFILE if edited else BASE_TEXT_PROFILE
    expected_cache = CACHE_PROFILE if edited else BASE_CACHE_PROFILE
    if receipt.get("segment_text_tokens") != expected_tokens:
        _fail(f"{label} segment_text_tokens do not implement the authenticated prompt schedule")
    if receipt.get("segment_cache_text") != expected_cache:
        _fail(f"{label} segment_cache_text does not implement the authenticated text-turn schedule")
    if receipt.get("valid_text_tokens") != exact_report["source_bank"]["original_valid_text_tokens"]:
        _fail(f"{label} valid_text_tokens must remain bound to the original continuation")
    if edited:
        if receipt.get("prompt_edit") != prompt_identity:
            _fail(f"{label} prompt_edit identity differs from the authenticated exact reference")
        if receipt.get("prompt_inputs") != prompt_inputs["manifest_path"]:
            _fail(f"{label} prompt_inputs path differs from the authenticated exact reference")
        if receipt.get("prompt_inputs_sha256") != prompt_inputs["manifest_sha256"]:
            _fail(f"{label} prompt_inputs_sha256 differs from the authenticated exact reference")
        if prompt_identity["prompt_sha256"] == base_prompt_sha256:
            _fail("QBit prompt edit repeats the original prompt")
        if receipt.get("invalidation") != prompt.PROMPT_INVALIDATION:
            _fail(f"{label} invalidation does not cover B and every descendant")
    elif (
        receipt.get("prompt_edit") is not None
        or receipt.get("prompt_inputs") != ""
        or receipt.get("prompt_inputs_sha256") != ""
        or receipt.get("invalidation") is not None
    ):
        _fail(f"{label} parent A cannot claim a prompt edit")
    if receipt.get("noise_edit") is not None or receipt.get("noise_scope") != prompt.UNCHANGED_NOISE_SCOPE:
        _fail(f"{label} must preserve the authenticated latent and epsilon banks")

    for key, expected in (("timeline_shape", TIMELINE_SHAPE), ("step_shape", STEP_SHAPE)):
        value = receipt.get(key)
        if type(value) is not list or any(type(dim) is not int for dim in value) or value != expected:
            _fail(f"{label} {key} differs from the authenticated sequence geometry")
    if receipt.get("state_format") != "qbit":
        _fail(f"{label} state_format must remain QBit")

    expected_roundtrips: list[int]
    if arm == "qbit_control":
        expected_mode, expected_roundtrips, expected_codec = "qbit", [2], {"restore", "roundtrip2"}
        expected_pack_boundary = 0
    elif arm == "parent_qbit_a":
        expected_mode, expected_roundtrips, expected_codec = "exact", [], {"pack"}
        expected_pack_boundary = 1
    elif start == 1:
        expected_mode, expected_roundtrips, expected_codec = "exact", [], {"restore", "pack"}
        expected_pack_boundary = 2
    else:
        expected_mode, expected_roundtrips, expected_codec = "exact", [], {"restore"}
        expected_pack_boundary = 0
    if receipt.get("cache_roundtrip") != expected_mode:
        _fail(f"{label} cache_roundtrip is incorrect for its interval")
    if receipt.get("qbit_roundtrip_boundaries") != expected_roundtrips:
        _fail(f"{label} QBit roundtrip boundaries do not match the admitted treatment")
    if type(receipt.get("compact_pack_boundary")) is not int or receipt["compact_pack_boundary"] != expected_pack_boundary:
        _fail(f"{label} compact pack boundary does not match its interval")
    codec = receipt.get("codec")
    if type(codec) is not dict or set(codec) != expected_codec:
        _fail(f"{label} codec reports do not match its pack/restore/roundtrip operations")

    base_contracts = receipt.get("boundary_contract_sha256")
    if type(base_contracts) is not dict or set(base_contracts) != {"1", "2"}:
        _fail(f"{label} boundary contract map must contain boundaries 1 and 2")
    for boundary, digest in base_contracts.items():
        compact._sha(digest, f"{label} boundary contract {boundary}")
    if start == 0:
        if receipt.get("contract_sha256") != "" or receipt.get("state_input_sha256") != "":
            _fail(f"{label} start-zero interval cannot consume a parent checkpoint")
    else:
        if receipt.get("contract_sha256") != base_contracts[str(start)]:
            _fail(f"{label} start contract digest does not match its boundary map")
        compact._sha(receipt.get("state_input_sha256"), f"{label} state_input_sha256")

    metrics = receipt.get("metrics")
    if type(metrics) is not dict or set(metrics) != METRIC_FIELDS:
        _fail(f"{label} metrics must contain exactly the five instrumented counters")
    for key, value in metrics.items():
        compact._int(value, f"{label} metric {key}")
    for key in ("encode_allocations", "encode_readbacks", "preterminal_waits"):
        if metrics[key] != 0:
            _fail(f"{label} metric {key} must be zero")
    if metrics["terminal_waits"] != 1 or metrics["command_segments"] <= 0:
        _fail(f"{label} must report positive command work and one terminal drain")

    checkpoint_file_bytes = compact._int(receipt.get("checkpoint_file_bytes"), f"{label} checkpoint_file_bytes")
    checkpoint_payload_bytes = compact._int(receipt.get("checkpoint_payload_bytes"), f"{label} checkpoint_payload_bytes")
    compact._int(receipt.get("resident_f32_kv_bytes"), f"{label} resident_f32_kv_bytes", minimum=1)
    logical_packed = compact._int(receipt.get("logical_packed_kv_bytes"), f"{label} logical_packed_kv_bytes")
    if stop < SEGMENTS:
        # The receipt payload count is the uncompressed logical inventory
        # (timeline + exact F32 cache extents). HNQC file bytes include the
        # fixed HNRS prefix plus compacted cache rows, so the two values are
        # intentionally different for QBit checkpoints.
        if checkpoint_file_bytes <= HNRS_PAYLOAD_OFFSET or checkpoint_payload_bytes <= TIMELINE_BYTES:
            _fail(f"{label} checkpoint accounting omits its HNRS records")
    elif checkpoint_file_bytes != 0 or checkpoint_payload_bytes != 0:
        _fail(f"{label} terminal interval cannot claim an output checkpoint")
    if expected_pack_boundary:
        if logical_packed <= 0:
            _fail(f"{label} QBit state pack must report compact cache bytes")
    elif logical_packed != 0:
        _fail(f"{label} interval cannot claim output compact cache bytes")
    if receipt.get("decoder") is not False or receipt.get("full_generation") is not False or receipt.get("full_upstream_video_parity") is not False:
        _fail(f"{label} must remain a DiT-only interval probe")

    source = exact_report["source_bank"]
    for key, expected in (
        ("inputs", source["manifest_path"]),
        ("inputs_sha256", source["manifest_sha256"]),
        ("payload_sha256", source["payload_sha256"]),
        ("conditioning_sha256", exact_report["identity"]["conditioning_sha256"]),
        ("model_index_sha256", exact_report["identity"]["model_index_sha256"]),
        ("upstream_revision", exact_report["identity"]["upstream_revision"]),
    ):
        if receipt.get(key) != expected:
            _fail(f"{label} {key} differs from the authenticated exact prompt input identity")
    return {
        "expected_codec_keys": sorted(expected_codec),
        "checkpoint_file_bytes": checkpoint_file_bytes,
        "checkpoint_payload_bytes": checkpoint_payload_bytes,
        "logical_packed_kv_bytes": logical_packed,
        "boundary_contract_sha256": base_contracts,
    }


def _validate_logical_checkpoint_inventory(
    receipt_check: dict[str, Any], expected_inventory: dict[str, int], label: str,
) -> int:
    expected_bytes = sum(expected_inventory.values())
    if receipt_check.get("checkpoint_payload_bytes") != expected_bytes:
        _fail(f"{label} receipt logical checkpoint payload differs from the expected KV inventory")
    return expected_bytes


def _load_qbit_run(
    arm: str, entry: Any, expected_interval: tuple[int, int],
    exact_report: dict[str, Any], prompt_identity: dict[str, Any],
    prompt_inputs: dict[str, Any], base_prompt_sha256: str,
    expected_cache_contract: dict[str, Any] | None = None,
    expected_output_base_contract: str | None = None,
) -> dict[str, Any]:
    if type(entry) is not dict or set(entry) != {"receipt_path", "interval"}:
        _fail(f"{arm} run entry must contain only receipt_path and interval")
    receipt_path = compact._absolute_file(entry.get("receipt_path"), f"{arm} receipt_path")
    if receipt_path.name != "receipt.json":
        _fail(f"{arm} receipt_path must name receipt.json")
    directory = resume._directory(receipt_path.parent, f"{arm} output directory")
    if receipt_path.parent != directory:
        _fail(f"{arm} receipt must live in its canonical output directory")
    raw_interval = entry.get("interval")
    if type(raw_interval) is not list or len(raw_interval) != 2 or any(type(v) is not int for v in raw_interval):
        _fail(f"{arm} interval must be a two-integer list")
    interval = (raw_interval[0], raw_interval[1])
    if interval != expected_interval:
        _fail(f"{arm} interval must be {list(expected_interval)}")
    receipt = resume._read_json(receipt_path, label=f"{arm} receipt")
    receipt_check = _validate_receipt(
        receipt, arm, interval, exact_report, prompt_identity, prompt_inputs,
        base_prompt_sha256,
    )
    start, stop = interval
    diagnostics = receipt.get("diagnostics")
    expected_artifacts = compact._artifact_names(start, stop, SEGMENTS, "qbit")
    if type(diagnostics) is not dict or set(diagnostics) != expected_artifacts:
        _fail(f"{arm} diagnostics do not name exactly its interval captures")
    timeline, timeline_meta = resume._read_artifact(
        directory, diagnostics, "timeline", "timeline.f32le", TIMELINE_BYTES,
        arm, retain_payload=True,
    )
    resume._finite(timeline, f"{arm} timeline")
    steps: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for segment in range(start, stop):
        for step in range(1, STEPS_PER_SEGMENT + 1):
            key = f"segment{segment}-step{step}"
            payload, metadata = resume._read_artifact(
                directory, diagnostics, key, f"{key}.f32le", STEP_BYTES,
                arm, retain_payload=True,
            )
            resume._finite(payload, f"{arm} {key}")
            steps[key] = (payload, metadata)
        final_step = steps[f"segment{segment}-step{STEPS_PER_SEGMENT}"][0]
        timeline_segment = compact._time_slice(
            timeline, TIMELINE_SHAPE,
            segment * SEGMENT_LATENTS, (segment + 1) * SEGMENT_LATENTS,
        )
        if timeline_segment != final_step:
            _fail(f"{arm} timeline does not contain final segment {segment} capture")

    state = None
    if stop < SEGMENTS:
        _, state_meta = resume._read_artifact(
            directory, diagnostics, "state", "boundary.hnqc",
            receipt_check["checkpoint_file_bytes"], arm, retain_payload=False,
        )
        base_contract = expected_cache_contract
        if base_contract is None or expected_output_base_contract is None:
            _fail(f"{arm} checkpoint expected contract is unavailable")
        expected_inventory = _expected_inventory_from_contract(
            base_contract, stop, receipt["segment_text_tokens"],
            receipt["segment_cache_text"],
        )
        _validate_logical_checkpoint_inventory(
            receipt_check, expected_inventory, arm
        )
        state = _parse_qbit_checkpoint(
            Path(state_meta["path"]),
            expected_sha256=diagnostics["state_sha256"],
            expected_file_bytes=receipt_check["checkpoint_file_bytes"],
            expected_payload_bytes=receipt_check["checkpoint_file_bytes"] - HNRS_PAYLOAD_OFFSET,
            expected_base_contract=expected_output_base_contract,
            expected_inventory=expected_inventory,
            expected_timeline_sha256=timeline_meta["sha256"],
            label=arm,
        )
        state["path"] = state_meta["path"]
        if stop == 1 and arm != "parent_qbit_a":
            _fail(f"{arm} cannot save an after-A state")
        if stop == 2 and arm != "qbit_restart_chain":
            _fail(f"{arm} cannot save an after-B state")
        pack_expected = expected_inventory
        pack_stats = _stats_from_codec_report(
            receipt["codec"]["pack"], pack_expected,
            f"{arm} pack", kind="pack",
        )
        if pack_stats != state["codec_stats"]:
            _fail(f"{arm} pack report differs from its actual checkpoint manifest")
        if receipt["codec"]["pack"].get("disk_bytes") != state["file_bytes"]:
            _fail(f"{arm} pack report file size differs from its checkpoint")
        if receipt_check["logical_packed_kv_bytes"] != state["codec_stats"]["logical_packed_bytes"]:
            _fail(f"{arm} receipt compact byte count differs from its checkpoint")

    expected_codec_inventories = _codec_inventories_for_run(
        arm, receipt, expected_cache_contract
    )
    codec_stats: dict[str, dict[str, Any]] = {}
    for name, inventory in expected_codec_inventories.items():
        kind = "restore" if name == "restore" else "roundtrip" if name.startswith("roundtrip") else "pack"
        report_stats = _stats_from_codec_report(
            receipt["codec"][name], inventory, f"{arm} {name}", kind=kind
        )
        codec_stats[name] = report_stats
    return {
        "arm": arm,
        "interval": interval,
        "directory": directory,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "receipt_check": receipt_check,
        "timeline": timeline,
        "timeline_meta": timeline_meta,
        "steps": steps,
        "state": state,
        "codec_stats": codec_stats,
    }


def _codec_inventories_for_run(
    arm: str, receipt: dict[str, Any], base_contract: dict[str, Any] | None,
) -> dict[str, dict[str, int]]:
    if base_contract is None:
        _fail(f"{arm} codec inventory needs the authenticated QBit base contract")
    result: dict[str, dict[str, int]] = {}
    start, stop = receipt["start_segment"], receipt["stop_after_segments"]
    if start > 0:
        # Every QBit interval resumes from a compact checkpoint. The only
        # allowed input boundaries are A for B runs and edited B for C.
        input_boundary = 1 if start == 1 else 2
        profile_tokens = BASE_TEXT_PROFILE if input_boundary == 1 else TEXT_PROFILE
        profile_cache = BASE_CACHE_PROFILE if input_boundary == 1 else CACHE_PROFILE
        result["restore"] = _expected_inventory_from_contract(
            base_contract, input_boundary, profile_tokens, profile_cache
        )
    if arm == "qbit_control":
        result["roundtrip2"] = _expected_inventory_from_contract(
            base_contract, 2, TEXT_PROFILE, CACHE_PROFILE
        )
    return result


def _load_source_pins(index_path: Path, exact_index_sha256: str) -> dict[str, Any]:
    run_root = index_path.parent.parent
    pins_path = run_root / "source-pins.json"
    pins = resume._read_json(pins_path, label="QBit source pins")
    expected_keys = {
        "head", "executable_sha256", "sources", "exact_index_sha256", "runner_sha256"
    }
    if set(pins) != expected_keys:
        _fail("QBit source pins fields are unsupported")
    head = pins.get("head")
    if type(head) is not str or len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        _fail("QBit source pins HEAD must be a lowercase Git object ID")
    executable_sha = compact._sha(pins.get("executable_sha256"), "source pin executable_sha256")
    runner_sha = compact._sha(pins.get("runner_sha256"), "source pin runner_sha256")
    if pins.get("exact_index_sha256") != exact_index_sha256:
        _fail("QBit source pins do not bind the referenced prior exact prompt index")
    sources = pins.get("sources")
    if type(sources) is not dict or set(sources) != set(PINNED_SOURCE_PATHS):
        _fail("QBit source pin inventory is incomplete or unexpected")
    repository_root = Path(__file__).resolve().parents[2]
    for relative in PINNED_SOURCE_PATHS:
        expected = compact._sha(sources[relative], f"source pin {relative}")
        path = repository_root / relative
        try:
            resolved = path.resolve(strict=True)
            if resolved != path or path.is_symlink() or not path.is_file():
                _fail(f"QBit pinned source path is not a regular in-repository file: {relative}")
            actual = _sha_bytes(path.read_bytes())
        except OSError as error:
            raise PromptEditQBitReferenceError(f"cannot read pinned QBit source: {relative}") from error
        if actual != expected:
            _fail(f"QBit pinned source digest changed: {relative}")
    runner_path = run_root / "run_qbit_matrix.py"
    if runner_path.is_symlink() or not runner_path.is_file() or _sha_bytes(runner_path.read_bytes()) != runner_sha:
        _fail("QBit matrix runner digest differs from its source pin")
    executable_path = run_root / "build" / "hy_nomas_compact_sequence_probe"
    if executable_path.is_symlink() or not executable_path.is_file():
        _fail("pinned QBit executable is missing or not a regular file")
    try:
        actual_executable_sha = _sha_bytes(executable_path.read_bytes())
    except OSError as error:
        raise PromptEditQBitReferenceError("cannot hash the pinned QBit executable") from error
    if actual_executable_sha != executable_sha:
        _fail("built QBit executable bytes differ from source-pins.json")
    return {
        "path": str(pins_path),
        "head": head,
        "executable_sha256": executable_sha,
        "runner_sha256": runner_sha,
        "source_count": len(sources),
        "source_inventory_verified": True,
        "executable_file_verified": True,
    }


def _prompt_boundary_contract(base_contract_sha256: str, identity: dict[str, Any]) -> str:
    return _canonical_json({
        "schema": prompt.EDIT_BOUNDARY_SCHEMA,
        "base_contract_sha256": base_contract_sha256,
        "prompt_edit": identity,
    })


def _reconstruct_qbit_base_contract(
    exact_report: dict[str, Any], qbit_parent_receipt: dict[str, Any],
) -> tuple[dict[str, Any], str, str, str]:
    try:
        exact_doc = exact_report["state_chains"]["parent_a"]["state"]["contract_document"]
    except (KeyError, TypeError) as error:
        raise PromptEditQBitReferenceError("exact prompt reference omits its authenticated parent A contract") from error
    if type(exact_doc) is not dict or tuple(exact_doc) != prompt.BASE_CONTRACT_FIELD_ORDER:
        _fail("exact prompt parent A base contract field order is unsupported")
    base_doc = dict(exact_doc)
    base_doc["executable_sha256"] = compact._sha(
        qbit_parent_receipt.get("executable_sha256"), "QBit parent executable_sha256"
    )
    try:
        prompt._validate_base_contract_document(
            base_doc,
            qbit_parent_receipt,
            next_segment=1,
            original_valid_text_tokens=exact_report["source_bank"]["original_valid_text_tokens"],
            expected_implementation_sha256=prompt._current_implementation_sha256(),
        )
    except PromptEditQBitReferenceError:
        raise
    except Exception as error:
        raise PromptEditQBitReferenceError(f"QBit parent A base contract rejected: {error}") from error
    base1_raw = _canonical_json(base_doc)
    base1_sha = _sha_bytes(base1_raw.encode("utf-8"))
    base2_doc = dict(base_doc)
    base2_doc["next_segment"] = 2
    base2_raw = _canonical_json(base2_doc)
    base2_sha = _sha_bytes(base2_raw.encode("utf-8"))
    return base_doc, base1_raw, base1_sha, base2_sha


def _matrix_entry_and_receipt(arms: dict[str, Any], arm: str, interval: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    document = arms.get(arm)
    if type(document) is not dict or set(document) != {"runs"} or type(document["runs"]) is not list:
        _fail(f"{arm} must contain exactly a runs list")
    expected_intervals = (interval,) if type(interval[0]) is int else interval
    if len(document["runs"]) != len(expected_intervals):
        _fail(f"{arm} run count does not match its required intervals")
    first_entry = document["runs"][0]
    if type(first_entry) is not dict or set(first_entry) != {"receipt_path", "interval"}:
        _fail(f"{arm} first run entry fields are unsupported")
    raw_interval = first_entry["interval"]
    if type(raw_interval) is not list or raw_interval != list(expected_intervals[0]):
        _fail(f"{arm} first run interval is incorrect")
    receipt_path = compact._absolute_file(first_entry["receipt_path"], f"{arm} first receipt_path")
    if receipt_path.name != "receipt.json":
        _fail(f"{arm} receipt_path must name receipt.json")
    return first_entry, resume._read_json(receipt_path, label=f"{arm} receipt")


def _validate_receipt_contract_maps(
    arms: dict[str, list[dict[str, Any]]], base_contracts: dict[str, str],
    edited_contracts: dict[str, str],
) -> None:
    for arm_runs in arms.values():
        for run in arm_runs:
            edited = run["arm"] != "parent_qbit_a"
            expected = edited_contracts if edited else base_contracts
            if run["receipt"]["boundary_contract_sha256"] != expected:
                _fail(f"{run['arm']} receipt boundary contracts do not match reconstructed QBit contracts")


def _validate_identities(
    arms: dict[str, list[dict[str, Any]]], exact_report: dict[str, Any],
    source_pins: dict[str, Any],
) -> dict[str, Any]:
    reference = arms["parent_qbit_a"][0]["receipt"]
    old_identity = exact_report["identity"]
    if reference["executable_sha256"] != source_pins["executable_sha256"]:
        _fail("fresh QBit parent executable differs from its pinned built binary")
    for run_list in arms.values():
        for run in run_list:
            receipt = run["receipt"]
            for key in IDENTITY_FIELDS:
                if receipt.get(key) != reference.get(key):
                    _fail(f"all QBit runs must share identity field {key}")
            if receipt.get("executable_sha256") != reference["executable_sha256"]:
                _fail("all QBit intervals must use the same fresh executable")
            if receipt.get("conditioning_sha256") != old_identity["conditioning_sha256"]:
                _fail("QBit runs differ from the prior exact conditioning bundle")
            if receipt.get("model_index_sha256") != old_identity["model_index_sha256"]:
                _fail("QBit runs differ from the prior exact model index")
            if receipt.get("upstream_revision") != old_identity["upstream_revision"]:
                _fail("QBit runs differ from the prior exact upstream revision")
    if reference["executable_sha256"] == old_identity["executable_sha256"]:
        same_binary_as_exact = True
    else:
        same_binary_as_exact = False
    return {
        "qbit_executable_sha256": reference["executable_sha256"],
        "prior_exact_executable_sha256": old_identity["executable_sha256"],
        "same_binary_as_prior_exact": same_binary_as_exact,
        "all_qbit_runs_share_fresh_executable": True,
        "inputs": reference["inputs"],
        "inputs_sha256": reference["inputs_sha256"],
        "payload_sha256": reference["payload_sha256"],
        "conditioning_sha256": reference["conditioning_sha256"],
        "model_index_sha256": reference["model_index_sha256"],
        "upstream_revision": reference["upstream_revision"],
        "distinct_processes": True,
        "pids": {
            arm: [run["receipt"]["pid"] for run in runs]
            for arm, runs in arms.items()
        },
    }


def _validate_state_lineage(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    parent = arms["parent_qbit_a"][0]
    control = arms["qbit_control"][0]
    split_b, split_c = arms["qbit_restart_chain"]
    parent_state = parent["state"]
    split_b_state = split_b["state"]
    if parent_state is None or split_b_state is None:
        _fail("QBit parent A and edited B must each save a compact checkpoint")
    if parent_state["base_contract_sha256"] != parent["receipt"]["boundary_contract_sha256"]["1"]:
        _fail("QBit parent A checkpoint base contract is not its after-A boundary")
    for run in (control, split_b):
        if run["receipt"]["state_input_sha256"] != parent_state["sha256"]:
            _fail("QBit B interval does not consume the fresh same-matrix after-A checkpoint")
        if run["receipt"]["contract_sha256"] != parent_state["base_contract_sha256"]:
            _fail("QBit B interval contract does not match the after-A checkpoint base contract")
    if split_b_state["base_contract_sha256"] != split_b["receipt"]["boundary_contract_sha256"]["2"]:
        _fail("edited QBit B checkpoint does not bind the edited after-B contract")
    if split_c["receipt"]["state_input_sha256"] != split_b_state["sha256"]:
        _fail("QBit C restart does not consume the actual edited B checkpoint")
    if split_c["receipt"]["contract_sha256"] != split_b_state["base_contract_sha256"]:
        _fail("QBit C restart contract does not match the edited after-B checkpoint")
    parent_codec_stats = parent_state["codec_stats"]
    edited_codec_stats = split_b_state["codec_stats"]
    for run in (control, split_b):
        if run["codec_stats"]["restore"] != parent_codec_stats:
            _fail(f"{run['arm']} restore report does not match the actual after-A checkpoint stats")
    if split_c["codec_stats"]["restore"] != edited_codec_stats:
        _fail("qbit_restart_chain C restore stats do not match the actual edited B checkpoint")
    return {
        "parent_after_a": {
            "state_sha256": parent_state["sha256"],
            "base_contract_sha256": parent_state["base_contract_sha256"],
        },
        "edited_after_b": {
            "state_sha256": split_b_state["sha256"],
            "base_contract_sha256": split_b_state["base_contract_sha256"],
        },
        "qbit_control_restores_parent_a": True,
        "split_b_restores_parent_a": True,
        "split_c_restores_edited_b": True,
    }


def _compare_qbit_restart(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    control = arms["qbit_control"][0]
    restart_by_segment = {
        segment: run
        for run in arms["qbit_restart_chain"]
        for segment in range(*run["interval"])
    }
    comparisons: dict[str, Any] = {}
    all_steps_bitwise = True
    all_steps_within = True
    all_segment_timeline_bitwise = True
    all_segment_timeline_within = True
    for segment in (1, 2):
        comparison, _changed = prompt._compare_segment(
            control, restart_by_segment[segment], segment,
            "QBit edited uninterrupted vs compact restart",
        )
        comparisons[str(segment)] = comparison
        step_results = list(comparison["steps"].values())
        all_steps_bitwise &= all(item["bitwise_equal"] for item in step_results)
        all_steps_within &= all(item["within_tolerance"] for item in step_results)
        all_segment_timeline_bitwise &= comparison["timeline"]["bitwise_equal"]
        all_segment_timeline_within &= comparison["timeline"]["within_tolerance"]
    full_timeline = resume._compare_f32(
        control["timeline"], arms["qbit_restart_chain"][-1]["timeline"],
        "QBit edited uninterrupted vs compact restart full timeline",
    )
    if not all_steps_within or not all_segment_timeline_within or not full_timeline["within_tolerance"]:
        _fail("QBit uninterrupted vs split outputs exceed the 2e-5 tolerance")
    return {
        "within_tolerance": True,
        "step_capture_count": 2 * STEPS_PER_SEGMENT,
        "segment_timeline_count": 2,
        "full_timeline_count": 1,
        "compared_f32_elements": (
            2 * STEPS_PER_SEGMENT * (STEP_BYTES // 4)
            + 2 * (SEGMENT_LATENTS * TIMELINE_SHAPE[1] * TIMELINE_SHAPE[3] * TIMELINE_SHAPE[4])
            + (TIMELINE_BYTES // 4)
        ),
        "all_steps_within_tolerance": all_steps_within,
        "all_steps_bitwise_equal": all_steps_bitwise,
        "all_segment_timelines_within_tolerance": all_segment_timeline_within,
        "all_segment_timelines_bitwise_equal": all_segment_timeline_bitwise,
        "full_timeline": full_timeline,
        "segments": comparisons,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
    }


def _compare_qbit_prompt_effect(
    arms: dict[str, list[dict[str, Any]]], exact_report: dict[str, Any],
) -> dict[str, Any]:
    """Require observed B/C differences from the authenticated exact original control.

    This is a nonvacuity check across the prior exact matrix and the lossy QBit
    matrix. It does not isolate QBit conditioning causality or assert quality.
    """
    try:
        original_meta = exact_report["artifacts"]["original_control"][0]
    except (KeyError, TypeError, IndexError) as error:
        raise PromptEditQBitReferenceError(
            "prior exact prompt reference omits its original control artifacts"
        ) from error
    timeline_meta = original_meta["timeline"]
    timeline_path = compact._absolute_file(timeline_meta.get("path"), "prior exact original timeline")
    original_timeline, _ = compact._read_hashed_file(
        timeline_path,
        compact._sha(timeline_meta.get("sha256"), "prior exact original timeline SHA"),
        "prior exact original timeline",
        expected_bytes=TIMELINE_BYTES,
        retain_payload=True,
    )
    original_steps: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for segment in (1, 2):
        for step in range(1, STEPS_PER_SEGMENT + 1):
            key = f"segment{segment}-step{step}"
            metadata = original_meta["steps"][key]
            step_path = compact._absolute_file(metadata.get("path"), f"prior exact original {key}")
            payload, _ = compact._read_hashed_file(
                step_path,
                compact._sha(metadata.get("sha256"), f"prior exact original {key} SHA"),
                f"prior exact original {key}",
                expected_bytes=STEP_BYTES,
                retain_payload=True,
            )
            original_steps[key] = (payload, metadata)
    original = {"timeline": original_timeline, "steps": original_steps}
    control = arms["qbit_control"][0]
    by_segment: dict[str, Any] = {}
    changed: dict[str, bool] = {}
    for segment in (1, 2):
        comparison, segment_changed = prompt._compare_segment(
            original, control, segment,
            "QBit edited output vs prior exact original control",
        )
        by_segment[str(segment)] = comparison
        changed[str(segment)] = segment_changed
    if not changed["1"]:
        _fail("QBit B output is indistinguishable from the prior exact original control")
    if not changed["2"]:
        _fail("QBit C output is indistinguishable from the prior exact original control")
    return {
        "b_differs_from_prior_exact_original": changed["1"],
        "c_differs_from_prior_exact_original": changed["2"],
        "segments": by_segment,
        "scope": (
            "observed QBit B/C differences from the authenticated prior exact original control; "
            "not causal isolation of QBit prompt conditioning and not a quality claim"
        ),
    }


def _baseline_exact_timeline(exact_report: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    try:
        metadata = exact_report["artifacts"]["parent_a"][0]["timeline"]
    except (KeyError, TypeError, IndexError) as error:
        raise PromptEditQBitReferenceError("exact prompt matrix omits its parent A timeline artifact") from error
    path = compact._absolute_file(metadata.get("path"), "prior exact parent A timeline")
    expected_bytes = compact._int(metadata.get("bytes"), "prior exact parent A timeline bytes")
    payload, total = compact._read_hashed_file(
        path,
        compact._sha(metadata.get("sha256"), "prior exact parent A timeline SHA"),
        "prior exact parent A timeline",
        expected_bytes=expected_bytes,
        retain_payload=True,
    )
    if total != TIMELINE_BYTES or len(payload) != TIMELINE_BYTES:
        _fail("prior exact parent A timeline has the wrong geometry")
    resume._finite(payload, "prior exact parent A timeline")
    return payload, metadata


def _validate_latent_a_and_future(
    arms: dict[str, list[dict[str, Any]]], exact_timeline: bytes,
) -> dict[str, Any]:
    qbit_parent = arms["parent_qbit_a"][0]
    exact_a = compact._time_slice(exact_timeline, TIMELINE_SHAPE, 0, SEGMENT_LATENTS)
    qbit_a = compact._time_slice(qbit_parent["timeline"], TIMELINE_SHAPE, 0, SEGMENT_LATENTS)
    if qbit_a != exact_a:
        _fail("QBit parent A latent bytes differ from the prior exact prompt reference")
    future_checks: dict[str, Any] = {}
    for run_list in arms.values():
        for run in run_list:
            actual_a = compact._time_slice(run["timeline"], TIMELINE_SHAPE, 0, SEGMENT_LATENTS)
            if actual_a != exact_a:
                _fail(f"{run['arm']} changed the immutable A latent")
            stop = run["interval"][1]
            suffix_start = stop * SEGMENT_LATENTS
            exact_future = compact._time_slice(
                exact_timeline, TIMELINE_SHAPE, suffix_start, TIMELINE_SHAPE[2]
            )
            actual_future = compact._time_slice(
                run["timeline"], TIMELINE_SHAPE, suffix_start, TIMELINE_SHAPE[2],
            )
            if actual_future != exact_future:
                _fail(f"{run['arm']} changed untouched future latent/noise bytes after its stop boundary")
            key = f"{run['arm']}:{run['interval'][0]}-{stop}"
            future_checks[key] = {
                "start_frame": suffix_start,
                "frames": TIMELINE_SHAPE[2] - suffix_start,
                "bytes": len(exact_future),
                "bitwise_equal_to_parent_source": True,
            }
    return {
        "a_latent_bitwise_equal_to_prior_exact": True,
        "a_prefix_bytes": len(exact_a),
        "untouched_suffix_checks": future_checks,
    }


def _lossy_exact_diagnostics(
    arms: dict[str, list[dict[str, Any]]], exact_report: dict[str, Any],
) -> dict[str, Any]:
    exact_control = exact_report["artifacts"]["edited_control"][0]
    baseline_timeline, _ = compact._read_hashed_file(
        Path(exact_control["timeline"]["path"]), exact_control["timeline"]["sha256"],
        "prior exact edited-control timeline", expected_bytes=TIMELINE_BYTES,
        retain_payload=True,
    )
    qbit_control = arms["qbit_control"][0]
    timeline = resume._compare_f32(
        baseline_timeline, qbit_control["timeline"],
        "QBit edited control vs prior exact edited control diagnostic",
    )
    per_step: dict[str, Any] = {}
    for segment in (1, 2):
        for step in range(1, STEPS_PER_SEGMENT + 1):
            key = f"segment{segment}-step{step}"
            meta = exact_control["steps"][key]
            path = Path(meta["path"])
            payload, _ = compact._read_hashed_file(
                path, meta["sha256"], f"prior exact edited {key}",
                expected_bytes=STEP_BYTES, retain_payload=True,
            )
            result = resume._compare_f32(
                payload, qbit_control["steps"][key][0],
                f"QBit edited control vs prior exact {key} diagnostic",
            )
            per_step[key] = result
    return {
        "scope": "QBit-lossy output differences from the prior exact edited control are diagnostic only",
        "timeline": timeline,
        "steps": per_step,
        "quality_gate": False,
    }


def compare_index(index_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Authenticate the exact reference and compare the QBit prompt-edit matrix."""
    path = compact._absolute_file(str(Path(index_path).absolute()), "QBit matrix index")
    index = resume._read_json(path, label="QBit matrix index")
    if type(index) is not dict or set(index) != {"schema", "exact_prompt_index", "arms"} or index.get("schema") != INDEX_SCHEMA:
        _fail("QBit matrix index schema or key inventory is unsupported")
    exact_record = index.get("exact_prompt_index")
    if type(exact_record) is not dict or set(exact_record) != {"path", "sha256"}:
        _fail("exact_prompt_index must contain only path and sha256")
    exact_path = compact._absolute_file(exact_record.get("path"), "exact prompt index")
    exact_sha = compact._sha(exact_record.get("sha256"), "exact prompt index SHA")
    _, exact_size = compact._read_hashed_file(
        exact_path, exact_sha, "exact prompt index", retain_payload=False,
    )
    del exact_size
    exact_report = prompt.compare_index(exact_path)
    if exact_report.get("passed") is not True or exact_report.get("schema") != prompt.REPORT_SCHEMA:
        _fail("prior exact prompt matrix did not pass its strict reference comparison")
    compact._read_hashed_file(
        exact_path, exact_sha, "exact prompt index after strict comparison", retain_payload=False,
    )
    prompt_identity = exact_report.get("prompt_edit")
    prompt_inputs = exact_report.get("prompt_inputs")
    if type(prompt_identity) is not dict or type(prompt_inputs) is not dict:
        _fail("prior exact prompt report omits authenticated conditioning identity")
    source = exact_report.get("source_bank")
    base_prompt_sha = source.get("original_prompt_sha256") if type(source) is dict else None
    if type(base_prompt_sha) is not str:
        _fail("prior exact prompt report omits the original prompt digest")
    source_pins = _load_source_pins(path, exact_sha)

    raw_arms = index.get("arms")
    if type(raw_arms) is not dict or set(raw_arms) != set(ARM_INTERVALS):
        _fail("QBit matrix arms must contain exactly parent_qbit_a, qbit_control, and qbit_restart_chain")
    parent_entry, parent_receipt = _matrix_entry_and_receipt(
        raw_arms, "parent_qbit_a", (0, 1)
    )
    _validate_receipt(
        parent_receipt, "parent_qbit_a", (0, 1), exact_report,
        prompt_identity, prompt_inputs, base_prompt_sha,
    )
    qbit_base_doc, qbit_base1_raw, qbit_base1_sha, qbit_base2_sha = _reconstruct_qbit_base_contract(
        exact_report, parent_receipt
    )
    if parent_receipt["executable_sha256"] != source_pins["executable_sha256"]:
        _fail("QBit parent receipt executable differs from its pinned binary")
    base_contracts = {"1": qbit_base1_sha, "2": qbit_base2_sha}
    edited_contracts = {
        "1": qbit_base1_sha,
        "2": prompt._canonical_prompt_edit_contract_sha(qbit_base2_sha, prompt_identity),
    }
    edited_base2_raw = _prompt_boundary_contract(qbit_base2_sha, prompt_identity)
    if _sha_bytes(edited_base2_raw.encode("utf-8")) != edited_contracts["2"]:
        _fail("internal edited after-B contract reconstruction mismatch")

    arms: dict[str, list[dict[str, Any]]] = {}
    expected_intervals = {
        "parent_qbit_a": ((0, 1),),
        "qbit_control": ((1, 3),),
        "qbit_restart_chain": ((1, 2), (2, 3)),
    }
    directories: set[Path] = set()
    pids: set[int] = set()
    for arm, intervals in expected_intervals.items():
        document = raw_arms[arm]
        if type(document) is not dict or set(document) != {"runs"} or type(document["runs"]) is not list or len(document["runs"]) != len(intervals):
            _fail(f"{arm} run list does not match its required intervals")
        runs = []
        for entry, interval in zip(document["runs"], intervals):
            output_contract = qbit_base1_raw if arm == "parent_qbit_a" else edited_base2_raw if interval == (1, 2) else None
            run = _load_qbit_run(
                arm, entry, interval, exact_report, prompt_identity,
                prompt_inputs, base_prompt_sha, qbit_base_doc,
                expected_output_base_contract=output_contract,
            )
            if run["directory"] in directories:
                _fail("each QBit interval must use a distinct output directory")
            directories.add(run["directory"])
            pid = run["receipt"]["pid"]
            if pid in pids:
                _fail("each QBit interval must use a distinct process ID")
            pids.add(pid)
            runs.append(run)
        arms[arm] = runs

    _validate_receipt_contract_maps(arms, base_contracts, edited_contracts)
    identity = _validate_identities(arms, exact_report, source_pins)
    state_lineage = _validate_state_lineage(arms)
    exact_timeline, exact_timeline_meta = _baseline_exact_timeline(exact_report)
    a_and_future = _validate_latent_a_and_future(arms, exact_timeline)
    restart_comparison = _compare_qbit_restart(arms)
    prompt_effect_comparison = _compare_qbit_prompt_effect(arms, exact_report)
    lossy_diagnostics = _lossy_exact_diagnostics(arms, exact_report)

    checkpoints = {}
    for arm, runs in arms.items():
        for run in runs:
            state = run["state"]
            if state is not None:
                checkpoints[f"{arm}:{run['interval'][1]}"] = {
                    "path": state["path"],
                    "file_bytes": state["file_bytes"],
                    "physical_record_payload_bytes": state["payload_bytes"],
                    "logical_inventory_bytes": run["receipt"]["checkpoint_payload_bytes"],
                    "logical_packed_kv_bytes": state["codec_stats"]["logical_packed_bytes"],
                    "exact_f32_cache_bytes": state["codec_stats"]["exact_bytes"],
                    "cache_record_count": state["cache_records"],
                    "tier_counts": state["codec_stats"]["tier_counts"],
                    "codec_identity_sha256": state["codec_identity_sha256"],
                }
    return {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "scope": (
            "QBit prompt-edit restart comparison with authenticated prior exact prompt reference; "
            "validates fresh QBit executable/source pins, real HNRS/HNQC records, codec manifests, "
            "row headers, cache inventories, state contracts and F32 captures. It does not decode "
            "approximate payloads into a quality claim or run the model."
        ),
        "prior_exact_reference": {
            "path": str(exact_path),
            "sha256": exact_sha,
            "schema": exact_report["schema"],
            "strict_matrix_passed": True,
            "executable_sha256": exact_report["identity"]["executable_sha256"],
            "prompt_effect": exact_report["edit_effect"],
            "timeline_path": exact_timeline_meta["path"],
            "timeline_sha256": exact_timeline_meta["sha256"],
        },
        "source_pins": source_pins,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "identity": identity,
        "contracts": {
            "qbit_base": base_contracts,
            "qbit_edited": edited_contracts,
            "qbit_base_reconstructed_from_parent_after_a": True,
            "after_a_executable_bound_by_parent_checkpoint": True,
            "after_b_binds_authenticated_prompt_edit": True,
        },
        "state_lineage": state_lineage,
        "a_prefix_and_future": a_and_future,
        "restart_parity": restart_comparison,
        "qbit_vs_prior_exact_original": prompt_effect_comparison,
        "lossy_vs_prior_exact": lossy_diagnostics,
        "intervals": {
            arm: [list(run["interval"]) for run in runs]
            for arm, runs in arms.items()
        },
        "metrics": {
            arm: [run["receipt"]["metrics"] for run in runs]
            for arm, runs in arms.items()
        },
        "treatments": {
            arm: [
                {
                    "state_format": run["receipt"]["state_format"],
                    "cache_roundtrip": run["receipt"]["cache_roundtrip"],
                    "qbit_roundtrip_boundaries": run["receipt"]["qbit_roundtrip_boundaries"],
                    "compact_pack_boundary": run["receipt"]["compact_pack_boundary"],
                    "codec_operations": sorted(run["receipt"]["codec"]),
                }
                for run in runs
            ]
            for arm, runs in arms.items()
        },
        "checkpoints": checkpoints,
        "text_profiles": exact_report["text_profiles"],
        "checkpoint_validation": (
            "Authenticates whole HNRSB01 file digest and canonical record digests; validates QBit "
            "contract identity, expected history inventory from the per-segment cache schedule, "
            "codec manifest/stat totals, every row tier/moment/offset/payload extent, finite exact-f32 "
            "escapes, and checkpoint timeline linkage. Approximate packed codes are not re-decoded."
        ),
    }


def main() -> None:
    parser_cli = argparse.ArgumentParser(description=__doc__)
    parser_cli.add_argument("index", help="QBit prompt-edit matrix-index.json")
    parser_cli.add_argument("--output", required=True, help="write a fresh validation report")
    args = parser_cli.parse_args()
    report = compare_index(args.index)
    resume._write_fresh_json(Path(args.output).absolute(), report)
    print(json.dumps({"passed": True, "report": args.output}))


if __name__ == "__main__":
    main()
