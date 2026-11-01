#!/usr/bin/env python3
"""Authenticate repeated exact/QBit Hy-Nomas continuation receipts.

This is a CPU-only DiT checkpoint and continuation comparator. It does not run
the model, decode a video, or establish upstream video parity or visual quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any

import hy_nomas_sequence_resume_reference as resume


INDEX_SCHEMA = "cogni_ml.hy_nomas_compact_resume_matrix/v1"
RECEIPT_SCHEMA = "cogni_ml.hy_nomas_compact_sequence_probe/v1"
REPORT_SCHEMA = "cogni_ml.hy_nomas_compact_resume_reference/v1"
INPUTS_SCHEMA = "cogni_ml.hy_nomas_continuation_inputs/v1"
ATOL = RTOL = 2e-5
CHANNELS = 32
SEGMENT_LATENTS = 30
HEIGHT = 9
WIDTH = 16
STEPS_PER_SEGMENT = 4
F32_BYTES = 4
ARM_NAMES = (
    "exact_control",
    "exact_restart_chain",
    "qbit_control",
    "compact_restart_chain",
)
IDENTITY_FIELDS = (
    "executable_sha256",
    "inputs",
    "inputs_sha256",
    "payload_sha256",
    "conditioning_sha256",
    "model_index_sha256",
    "upstream_revision",
    "segments",
    "steps_per_segment",
    "valid_text_tokens",
    "timeline_shape",
    "step_shape",
    "boundary_contract_sha256",
)
METRIC_FIELDS = {
    "encode_allocations",
    "encode_readbacks",
    "preterminal_waits",
    "command_segments",
    "terminal_waits",
}


class CompactResumeReferenceError(resume.ResumeReferenceError):
    """A matrix index, receipt, or artifact fails the compact continuation contract."""


def _fail(message: str) -> None:
    raise CompactResumeReferenceError(message)


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: Any, label: str, *, empty: bool = False) -> str:
    if empty and value == "":
        return ""
    if type(value) is not str or resume.SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _absolute_file(value: Any, label: str) -> Path:
    if type(value) is not str or not value:
        _fail(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        _fail(f"{label} must be an absolute regular non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CompactResumeReferenceError(f"cannot resolve {label}") from error
    return resolved


def _read_hashed_file(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_bytes: int | None = None,
    retain_payload: bool = True,
    retain_range: tuple[int, int] | None = None,
) -> tuple[bytes, int]:
    """Read a stable, non-symlink file in bounded chunks and authenticate it."""
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} must be a regular non-symlink file")
        if expected_bytes is not None and before.st_size != expected_bytes:
            _fail(f"{label} extent does not match its metadata")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or resume._stat_identity(opened) != resume._stat_identity(before):
                _fail(f"{label} changed before it was opened")
            digest = hashlib.sha256()
            if retain_payload and retain_range is not None:
                _fail(f"{label} cannot retain a full payload and a selected range together")
            chunks: list[bytes] | None = [] if retain_payload or retain_range is not None else None
            total = 0
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while True:
                    chunk = stream.read(resume.ARTIFACT_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    chunk_start = total
                    total += len(chunk)
                    digest.update(chunk)
                    if chunks is not None and retain_range is None:
                        chunks.append(chunk)
                    elif chunks is not None:
                        range_start, range_bytes = retain_range
                        range_stop = range_start + range_bytes
                        overlap_start = max(range_start, chunk_start)
                        overlap_stop = min(range_stop, total)
                        if overlap_start < overlap_stop:
                            chunks.append(chunk[overlap_start - chunk_start:overlap_stop - chunk_start])
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.stat(path, follow_symlinks=False)
    except CompactResumeReferenceError:
        raise
    except OSError as error:
        raise CompactResumeReferenceError(f"cannot read {label}") from error
    if (
        resume._stat_identity(before) != resume._stat_identity(after_open)
        or resume._stat_identity(before) != resume._stat_identity(after_path)
        or total != before.st_size
    ):
        _fail(f"{label} changed while being read")
    if expected_bytes is not None and total != expected_bytes:
        _fail(f"{label} extent does not match its metadata")
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha256:
        _fail(f"{label} SHA-256 does not match its authenticated metadata")
    return (b"" if chunks is None else b"".join(chunks)), total


def _load_source_bank(index: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    source = index.get("source_bank")
    expected_keys = {
        "manifest_path",
        "manifest_sha256",
        "payload_path",
        "payload_sha256",
        "record_offset",
        "nbytes",
        "shape",
    }
    if type(source) is not dict or set(source) != expected_keys:
        _fail("source_bank must name the authenticated manifest, payload, and latent record")
    manifest_path = _absolute_file(source["manifest_path"], "source bank manifest")
    manifest_sha = _sha(source["manifest_sha256"], "source bank manifest_sha256")
    manifest = resume._read_json(manifest_path, label="source bank manifest")
    _, manifest_size = _read_hashed_file(
        manifest_path, manifest_sha, "source bank manifest", retain_payload=True
    )
    if manifest.get("schema") != INPUTS_SCHEMA:
        _fail("source bank manifest schema is unsupported")
    if type(manifest.get("payload")) is not str or not manifest["payload"]:
        _fail("source bank manifest payload must be a filename")

    payload_path = _absolute_file(source["payload_path"], "source bank payload")
    payload_sha = _sha(source["payload_sha256"], "source bank payload_sha256")
    if payload_path.parent != manifest_path.parent or payload_path.name != manifest["payload"]:
        _fail("source bank payload path does not match its manifest")
    payload_bytes = _int(manifest.get("payload_bytes"), "manifest payload_bytes", minimum=1)
    if type(manifest.get("payload_sha256")) is not str or manifest["payload_sha256"] != payload_sha:
        _fail("source bank payload SHA-256 differs from the manifest")
    record = manifest.get("tensors", {}).get("input.latent_bank") if type(manifest.get("tensors")) is dict else None
    if type(record) is not dict:
        _fail("source bank manifest lacks input.latent_bank")
    if record.get("dtype") != "F32":
        _fail("source bank latent bank must use F32")
    offset = _int(source["record_offset"], "source_bank record_offset")
    nbytes = _int(source["nbytes"], "source_bank nbytes", minimum=1)
    shape_value = source["shape"]
    if type(shape_value) is not list or len(shape_value) != 5:
        _fail("source_bank shape must be a rank-five list")
    shape = [_int(dim, "source_bank shape extent", minimum=1) for dim in shape_value]
    record_shape = record.get("shape")
    if (
        type(record_shape) is not list
        or any(type(dim) is not int for dim in record_shape)
        or record_shape != shape
    ):
        _fail("source bank record shape does not match its manifest")
    if (
        _int(record.get("offset"), "latent bank manifest offset") != offset
        or _int(record.get("nbytes"), "latent bank manifest nbytes", minimum=1) != nbytes
    ):
        _fail("source bank record metadata does not match its manifest")
    if math.prod(shape) * F32_BYTES != nbytes:
        _fail("source bank F32 extent does not match its shape")
    if offset > payload_bytes or nbytes > payload_bytes - offset:
        _fail("source bank latent record exceeds the authenticated payload")
    record_sha = _sha(record.get("sha256"), "latent bank manifest sha256")
    bank, payload_size = _read_hashed_file(
        payload_path,
        payload_sha,
        "source bank payload",
        expected_bytes=payload_bytes,
        retain_payload=False,
        retain_range=(offset, nbytes),
    )
    if payload_size != payload_bytes or len(bank) != nbytes:
        _fail("source bank payload or latent record extent differs from its manifest")
    if hashlib.sha256(bank).hexdigest() != record_sha:
        _fail("source bank latent record SHA-256 does not match its manifest")
    resume._finite(bank, "authenticated source latent bank")
    metadata = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_bytes": manifest_size,
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "payload_bytes": payload_size,
        "record_offset": offset,
        "record_bytes": nbytes,
        "record_sha256": record_sha,
        "shape": shape,
    }
    return bank, metadata


def _time_slice(payload: bytes, shape: list[int], start: int, stop: int) -> bytes:
    channels, frames, height, width = shape[1], shape[2], shape[3], shape[4]
    channel_stride = frames * height * width * F32_BYTES
    frame_stride = height * width * F32_BYTES
    if len(payload) != math.prod(shape) * F32_BYTES or not 0 <= start <= stop <= frames:
        _fail("timeline time slice does not match its F32 geometry")
    return b"".join(
        payload[channel * channel_stride + start * frame_stride:
                channel * channel_stride + stop * frame_stride]
        for channel in range(channels)
    )


def _artifact_names(start: int, stop: int, segments: int, state_format: str) -> set[str]:
    names = {"timeline", "timeline_sha256"}
    for segment in range(start, stop):
        for step in range(1, STEPS_PER_SEGMENT + 1):
            key = f"segment{segment}-step{step}"
            names.update((key, f"{key}_sha256"))
    if stop < segments:
        names.update(("state", "state_sha256"))
    return names


def _validate_codec(receipt: dict[str, Any], arm: str, start: int, stop: int, segments: int) -> None:
    expected_roundtrips = list(range(1, segments)) if arm == "qbit_control" else []
    actual_roundtrips = receipt.get("qbit_roundtrip_boundaries")
    if (
        type(actual_roundtrips) is not list
        or any(type(boundary) is not int for boundary in actual_roundtrips)
        or actual_roundtrips != expected_roundtrips
    ):
        _fail(f"{arm} QBit roundtrip boundary evidence is missing or unexpected")

    expected_codec: set[str]
    if arm == "qbit_control":
        expected_codec = {f"roundtrip{boundary}" for boundary in range(1, segments)}
    elif arm == "compact_restart_chain":
        expected_codec = set()
        if start > 0:
            expected_codec.add("restore")
        if stop < segments:
            expected_codec.add("pack")
    else:
        expected_codec = set()
    codec = receipt.get("codec")
    if type(codec) is not dict or set(codec) != expected_codec:
        _fail(f"{arm} codec reports do not match its required roundtrip/pack/restore treatment")

    for name, report in codec.items():
        if type(report) is not dict:
            _fail(f"{arm} codec {name} report must be an object")
        if _int(report.get("invalid_rows"), f"{arm} codec {name} invalid_rows") != 0:
            _fail(f"{arm} codec {name} contains invalid rows")
        logical = _int(
            report.get("logical_packed_bytes"),
            f"{arm} codec {name} logical_packed_bytes",
            minimum=1,
        )
        exact = _int(report.get("exact_bytes"), f"{arm} codec {name} exact_bytes", minimum=1)
        if logical >= exact:
            _fail(f"{arm} codec {name} does not report compact bytes below its F32 extent")
        if name == "pack":
            packed = _int(
                report.get("logical_packed_kv_bytes"),
                f"{arm} codec pack logical_packed_kv_bytes",
                minimum=1,
            )
            if packed != receipt.get("logical_packed_kv_bytes"):
                _fail("compact pack byte accounting differs from its receipt")


def _validate_receipt(
    receipt: dict[str, Any],
    arm: str,
    interval: tuple[int, int],
    segments: int,
    steps_per_segment: int,
    timeline_shape: list[int],
) -> None:
    label = f"{arm} receipt"
    start, stop = interval
    expected_mode = "resume" if start > 0 else "control" if stop == segments else "save"
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("mode") != expected_mode:
        _fail(f"{label} schema or interval mode is invalid")
    if _int(receipt.get("segments"), f"{label} segments", minimum=1) != segments:
        _fail(f"{label} segments differs from the index")
    if _int(receipt.get("start_segment"), f"{label} start_segment") != start:
        _fail(f"{label} start_segment differs from its index interval")
    if _int(receipt.get("stop_after_segments"), f"{label} stop_after_segments") != stop:
        _fail(f"{label} stop interval differs from its index interval")
    if _int(receipt.get("resume_next_segment"), f"{label} resume_next_segment") != start:
        _fail(f"{label} resume_next_segment must identify the interval start")
    if _int(receipt.get("steps_per_segment"), f"{label} steps_per_segment", minimum=1) != steps_per_segment:
        _fail(f"{label} steps_per_segment differs from the index")
    if receipt.get("state_format") != ("qbit" if arm == "compact_restart_chain" else "exact"):
        _fail(f"{label} checkpoint state_format is wrong for its arm")
    expected_roundtrip = "qbit" if arm == "qbit_control" else "exact"
    if receipt.get("cache_roundtrip") != expected_roundtrip:
        _fail(f"{label} cache_roundtrip is wrong for its arm")

    pid = _int(receipt.get("pid"), f"{label} pid", minimum=1)
    del pid
    _sha(receipt.get("executable_sha256"), f"{label} executable_sha256")
    _sha(receipt.get("inputs_sha256"), f"{label} inputs_sha256")
    _sha(receipt.get("payload_sha256"), f"{label} payload_sha256")
    _sha(receipt.get("conditioning_sha256"), f"{label} conditioning_sha256")
    _sha(receipt.get("model_index_sha256"), f"{label} model_index_sha256")
    if type(receipt.get("inputs")) is not str or not receipt["inputs"]:
        _fail(f"{label} inputs must name its authenticated input manifest")
    if type(receipt.get("upstream_revision")) is not str or not receipt["upstream_revision"]:
        _fail(f"{label} upstream_revision must be non-empty")
    if type(receipt.get("valid_text_tokens")) is not int or receipt["valid_text_tokens"] <= 0:
        _fail(f"{label} valid_text_tokens must be positive")
    receipt_shape = receipt.get("timeline_shape")
    if (
        type(receipt_shape) is not list
        or any(type(dim) is not int for dim in receipt_shape)
        or receipt_shape != timeline_shape
    ):
        _fail(f"{label} timeline shape differs from the source bank")
    step_shape = receipt.get("step_shape")
    if (
        type(step_shape) is not list
        or any(type(dim) is not int for dim in step_shape)
        or step_shape != [1, CHANNELS, SEGMENT_LATENTS, HEIGHT, WIDTH]
    ):
        _fail(f"{label} step capture shape is invalid")
    if receipt.get("decoder") is not False or receipt.get("full_generation") is not False:
        _fail(f"{label} must remain a DiT-only interval probe")
    if receipt.get("full_upstream_video_parity") is not False:
        _fail(f"{label} cannot claim full upstream video parity")

    boundary_map = receipt.get("boundary_contract_sha256")
    expected_boundaries = {str(boundary) for boundary in range(1, segments)}
    if type(boundary_map) is not dict or set(boundary_map) != expected_boundaries:
        _fail(f"{label} boundary contract map is incomplete")
    for boundary, digest in boundary_map.items():
        _sha(digest, f"{label} boundary contract {boundary}")
    contract = receipt.get("contract_sha256")
    if start == 0:
        if contract != "":
            _fail(f"{label} start-zero contract_sha256 must be empty")
    elif _sha(contract, f"{label} contract_sha256") != boundary_map[str(start)]:
        _fail(f"{label} contract SHA does not match its start boundary")
    state_input_sha = receipt.get("state_input_sha256")
    if start == 0:
        if state_input_sha != "":
            _fail(f"{label} start-zero interval cannot consume a state")
    else:
        _sha(state_input_sha, f"{label} state_input_sha256")

    metrics = receipt.get("metrics")
    if type(metrics) is not dict or set(metrics) != METRIC_FIELDS:
        _fail(f"{label} metrics must contain the five instrumented counters")
    for key, value in metrics.items():
        _int(value, f"{label} metric {key}")
    for key in ("encode_allocations", "encode_readbacks", "preterminal_waits"):
        if metrics[key] != 0:
            _fail(f"{label} metric {key} must be zero")
    if metrics["terminal_waits"] != 1:
        _fail(f"{label} must have exactly one terminal wait")

    checkpoint_file_bytes = _int(receipt.get("checkpoint_file_bytes"), f"{label} checkpoint_file_bytes")
    logical_packed = _int(receipt.get("logical_packed_kv_bytes"), f"{label} logical_packed_kv_bytes")
    resident_f32 = _int(receipt.get("resident_f32_kv_bytes"), f"{label} resident_f32_kv_bytes", minimum=1)
    payload_bytes = _int(receipt.get("checkpoint_payload_bytes"), f"{label} checkpoint_payload_bytes")
    if stop < segments:
        if payload_bytes <= math.prod(timeline_shape) * F32_BYTES:
            _fail(f"{label} checkpoint payload bytes omit its latent timeline or cache records")
        if checkpoint_file_bytes <= 0:
            _fail(f"{label} saved checkpoint file extent must be positive")
    elif checkpoint_file_bytes != 0 or payload_bytes != 0:
        _fail(f"{label} terminal interval cannot claim a saved checkpoint")
    if arm == "compact_restart_chain" and stop < segments:
        if logical_packed <= 0 or logical_packed >= resident_f32:
            _fail(f"{label} logical packed K/V bytes must be below resident F32 K/V bytes")
    elif logical_packed != 0:
        _fail(f"{label} has unexpected logical packed K/V byte accounting")

    expected_pack_boundary = stop if arm == "compact_restart_chain" and stop < segments else 0
    if _int(receipt.get("compact_pack_boundary"), f"{label} compact_pack_boundary") != expected_pack_boundary:
        _fail(f"{label} compact pack boundary evidence is incorrect")
    _validate_codec(receipt, arm, start, stop, segments)


def _load_run(
    arm: str,
    entry: dict[str, Any],
    segments: int,
    steps_per_segment: int,
    timeline_shape: list[int],
) -> dict[str, Any]:
    if type(entry) is not dict or set(entry) != {"receipt_path", "interval"}:
        _fail(f"{arm} run entry must contain only receipt_path and interval")
    receipt_path = _absolute_file(entry["receipt_path"], f"{arm} receipt_path")
    if receipt_path.name != "receipt.json":
        _fail(f"{arm} receipt_path must name receipt.json")
    directory = resume._directory(receipt_path.parent, f"{arm} output directory")
    if receipt_path.parent != directory:
        _fail(f"{arm} receipt_path must use its canonical output directory")
    raw_interval = entry["interval"]
    if type(raw_interval) is not list or len(raw_interval) != 2:
        _fail(f"{arm} interval must be a two-integer [start, stop] list")
    start = _int(raw_interval[0], f"{arm} interval start")
    stop = _int(raw_interval[1], f"{arm} interval stop", minimum=1)
    if not 0 <= start < stop <= segments:
        _fail(f"{arm} interval is outside the sequence bounds")
    interval = (start, stop)
    receipt = resume._read_json(receipt_path, label=f"{arm} receipt")
    _validate_receipt(receipt, arm, interval, segments, steps_per_segment, timeline_shape)

    diagnostics = receipt.get("diagnostics")
    if type(diagnostics) is not dict or set(diagnostics) != _artifact_names(start, stop, segments, receipt["state_format"]):
        _fail(f"{arm} diagnostics do not name exactly the interval artifacts")
    timeline_bytes = math.prod(timeline_shape) * F32_BYTES
    timeline, timeline_meta = resume._read_artifact(
        directory,
        diagnostics,
        "timeline",
        "timeline.f32le",
        timeline_bytes,
        arm,
        retain_payload=True,
    )
    resume._finite(timeline, f"{arm} timeline")

    steps: dict[str, tuple[bytes, dict[str, Any]]] = {}
    step_bytes = math.prod([1, CHANNELS, SEGMENT_LATENTS, HEIGHT, WIDTH]) * F32_BYTES
    for segment in range(start, stop):
        for step in range(1, steps_per_segment + 1):
            key = f"segment{segment}-step{step}"
            payload, metadata = resume._read_artifact(
                directory,
                diagnostics,
                key,
                f"{key}.f32le",
                step_bytes,
                arm,
                retain_payload=True,
            )
            resume._finite(payload, f"{arm} {key}")
            steps[key] = (payload, metadata)
        final_step = steps[f"segment{segment}-step{steps_per_segment}"][0]
        start_frame = segment * SEGMENT_LATENTS
        timeline_segment = _time_slice(
            timeline, timeline_shape, start_frame, start_frame + SEGMENT_LATENTS
        )
        if timeline_segment != final_step:
            _fail(f"{arm} timeline does not contain its final {segment} Euler step capture")

    state = None
    if stop < segments:
        state_format = receipt["state_format"]
        filename = "boundary.hnqc" if state_format == "qbit" else "boundary.hnrs"
        try:
            _, state_meta = resume._read_artifact(
                directory,
                diagnostics,
                "state",
                filename,
                receipt["checkpoint_file_bytes"],
                arm,
                retain_payload=False,
            )
        except resume.ResumeReferenceError as error:
            raise CompactResumeReferenceError(
                f"{arm} checkpoint file byte extent or digest mismatch: {error}"
            ) from error
        state = {
            **state_meta,
            "sha256": diagnostics["state_sha256"],
            "file_bytes": receipt["checkpoint_file_bytes"],
            "logical_packed_kv_bytes": receipt["logical_packed_kv_bytes"],
            "resident_f32_kv_bytes": receipt["resident_f32_kv_bytes"],
            "checkpoint_payload_bytes": receipt["checkpoint_payload_bytes"],
        }
    return {
        "arm": arm,
        "interval": interval,
        "directory": directory,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "timeline": timeline,
        "timeline_meta": timeline_meta,
        "steps": steps,
        "state": state,
    }


def _expected_intervals(arm: str, segments: int) -> list[tuple[int, int]]:
    if arm in ("exact_control", "qbit_control"):
        return [(0, segments)]
    return [(segment, segment + 1) for segment in range(segments)]


def _load_arms(index: dict[str, Any], segments: int, steps: int, shape: list[int]) -> dict[str, list[dict[str, Any]]]:
    raw_arms = index.get("arms")
    if type(raw_arms) is not dict or set(raw_arms) != set(ARM_NAMES):
        _fail("index arms must contain exactly the four admitted exact/QBit paths")
    arms: dict[str, list[dict[str, Any]]] = {}
    all_directories: set[Path] = set()
    all_pids: set[int] = set()
    for arm in ARM_NAMES:
        arm_doc = raw_arms[arm]
        if type(arm_doc) is not dict or set(arm_doc) != {"runs"} or type(arm_doc["runs"]) is not list:
            _fail(f"{arm} must contain a runs list")
        raw_runs = arm_doc["runs"]
        expected = _expected_intervals(arm, segments)
        if len(raw_runs) != len(expected):
            _fail(f"{arm} run count does not cover the required contiguous intervals")
        loaded: list[dict[str, Any]] = []
        for entry, interval in zip(raw_runs, expected):
            run = _load_run(arm, entry, segments, steps, shape)
            if run["interval"] != interval:
                _fail(f"{arm} intervals must be contiguous and match the declared run order")
            if run["directory"] in all_directories:
                _fail("every interval must come from a distinct fresh process output directory")
            all_directories.add(run["directory"])
            pid = run["receipt"]["pid"]
            if pid in all_pids:
                _fail("all intervals must use distinct process IDs and fresh processes")
            all_pids.add(pid)
            loaded.append(run)
        arms[arm] = loaded
    return arms


def _validate_identities(arms: dict[str, list[dict[str, Any]]], source: dict[str, Any]) -> dict[str, Any]:
    reference = arms["exact_control"][0]["receipt"]
    resident_sizes: set[int] = set()
    for arm_runs in arms.values():
        for run in arm_runs:
            receipt = run["receipt"]
            resident_sizes.add(receipt["resident_f32_kv_bytes"])
            for key in IDENTITY_FIELDS:
                if receipt.get(key) != reference.get(key):
                    _fail(f"all arms must share authenticated identity field {key}")
            receipt_manifest = _absolute_file(receipt["inputs"], "receipt input manifest")
            if receipt_manifest != Path(source["manifest_path"]):
                _fail("receipt input manifest does not match the source-bank manifest")
            if receipt["inputs_sha256"] != source["manifest_sha256"]:
                _fail("receipt inputs_sha256 does not match the source-bank manifest")
            if receipt["payload_sha256"] != source["payload_sha256"]:
                _fail("receipt payload_sha256 does not match the source-bank payload")
    if len(resident_sizes) != 1:
        _fail("all arms must report the same resident F32 K/V storage geometry")
    return {
        "executable_sha256": reference["executable_sha256"],
        "inputs": reference["inputs"],
        "inputs_sha256": reference["inputs_sha256"],
        "payload_sha256": reference["payload_sha256"],
        "conditioning_sha256": reference["conditioning_sha256"],
        "model_index_sha256": reference["model_index_sha256"],
        "upstream_revision": reference["upstream_revision"],
        "boundary_contract_sha256": reference["boundary_contract_sha256"],
        "distinct_processes": True,
        "pids": {
            arm: [run["receipt"]["pid"] for run in runs]
            for arm, runs in arms.items()
        },
    }


def _validate_state_chains(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    chain_report: dict[str, Any] = {}
    for arm in ("exact_restart_chain", "compact_restart_chain"):
        runs = arms[arm]
        for index, run in enumerate(runs):
            start, stop = run["interval"]
            receipt = run["receipt"]
            if index == 0:
                if receipt["state_input_sha256"] != "":
                    _fail(f"{arm} first interval cannot consume a checkpoint")
            else:
                parent = runs[index - 1]
                parent_state = parent["state"]
                if parent_state is None or receipt["state_input_sha256"] != parent_state["sha256"]:
                    _fail(f"{arm} parent state SHA does not match the prior authenticated checkpoint chain")
                if start != parent["interval"][1]:
                    _fail(f"{arm} resume interval does not start at its checkpoint boundary")
                shape = receipt["timeline_shape"]
                prefix_stop = parent["interval"][1] * SEGMENT_LATENTS
                if _time_slice(parent["timeline"], shape, 0, prefix_stop) != _time_slice(
                    run["timeline"], shape, 0, prefix_stop
                ):
                    _fail(f"{arm} restart timeline changed its already-generated prefix")
            if stop < receipt["segments"] and run["state"] is None:
                _fail(f"{arm} interval omitted its output checkpoint")
        chain_report[arm] = [
            {
                "interval": list(run["interval"]),
                "state_input_sha256": run["receipt"]["state_input_sha256"],
                "state": run["state"],
            }
            for run in runs
        ]
    return chain_report


def _validate_source_suffixes(
    arms: dict[str, list[dict[str, Any]]],
    source_bank: bytes,
    shape: list[int],
) -> None:
    for arm, runs in arms.items():
        for run in runs:
            stop = run["interval"][1]
            first_unexecuted_frame = stop * SEGMENT_LATENTS
            if _time_slice(run["timeline"], shape, first_unexecuted_frame, shape[2]) != _time_slice(
                source_bank, shape, first_unexecuted_frame, shape[2]
            ):
                _fail(f"{arm} future/unexecuted source bank suffix was mutated")


def _compare_step_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    segment: int,
    steps: int,
    label: str,
) -> tuple[dict[str, Any], bool]:
    step_reports: dict[str, Any] = {}
    changed = False
    for step in range(1, steps + 1):
        key = f"segment{segment}-step{step}"
        left_bytes = left["steps"][key][0]
        right_bytes = right["steps"][key][0]
        result = resume._compare_f32(left_bytes, right_bytes, f"{label} {key}")
        step_reports[str(step)] = result
        changed = changed or not result["bitwise_equal"]
    return step_reports, changed


def _compare_equivalent_arms(
    control: dict[str, Any],
    chain: list[dict[str, Any]],
    segments: int,
    steps: int,
    label: str,
) -> dict[str, Any]:
    per_segment: dict[str, Any] = {}
    for segment in range(segments):
        run = chain[segment]
        capture_reports, _ = _compare_step_pair(control, run, segment, steps, label)
        if any(not result["bitwise_equal"] for result in capture_reports.values()):
            _fail(f"{label} restart step captures are not bitwise identical for segment {segment}")
        per_segment[str(segment)] = capture_reports
    final_timeline = chain[-1]["timeline"]
    timeline_equal = control["timeline"] == final_timeline
    if not timeline_equal:
        _fail(f"{label} control and fresh restart chain timelines are not bitwise identical")
    return {
        "all_steps_bitwise_equal": True,
        "timeline_bitwise_equal": True,
        "steps_by_segment": per_segment,
        "timeline_bytes": len(final_timeline),
    }


def _lossy_comparison(
    exact_control: dict[str, Any],
    qbit_control: dict[str, Any],
    exact_chain: list[dict[str, Any]],
    compact_chain: list[dict[str, Any]],
    segments: int,
    steps: int,
) -> dict[str, Any]:
    control_segments: dict[str, Any] = {}
    restart_segments: dict[str, Any] = {}
    a_segment: dict[str, Any] = {}
    control_bc_diff = False
    restart_bc_diff = False
    for segment in range(segments):
        exact_control_steps, control_step_changed = _compare_step_pair(
            exact_control, qbit_control, segment, steps, "QBit control vs exact"
        )
        exact_restart_steps, restart_step_changed = _compare_step_pair(
            exact_chain[segment], compact_chain[segment], segment, steps,
            "compact restart vs exact restart",
        )
        control_timeline = resume._compare_f32(
            _time_slice(exact_control["timeline"], exact_control["receipt"]["timeline_shape"],
                        segment * SEGMENT_LATENTS, (segment + 1) * SEGMENT_LATENTS),
            _time_slice(qbit_control["timeline"], qbit_control["receipt"]["timeline_shape"],
                        segment * SEGMENT_LATENTS, (segment + 1) * SEGMENT_LATENTS),
            f"QBit control timeline segment {segment}",
        )
        restart_timeline = resume._compare_f32(
            _time_slice(exact_chain[segment]["timeline"], exact_chain[segment]["receipt"]["timeline_shape"],
                        segment * SEGMENT_LATENTS, (segment + 1) * SEGMENT_LATENTS),
            _time_slice(compact_chain[segment]["timeline"], compact_chain[segment]["receipt"]["timeline_shape"],
                        segment * SEGMENT_LATENTS, (segment + 1) * SEGMENT_LATENTS),
            f"compact restart timeline segment {segment}",
        )
        comparison = {
            "steps": exact_control_steps,
            "timeline": control_timeline,
        }
        restart_comparison = {
            "steps": exact_restart_steps,
            "timeline": restart_timeline,
        }
        if segment == 0:
            a_segment = {"control": comparison, "restart": restart_comparison}
        else:
            control_segments[str(segment)] = comparison
            restart_segments[str(segment)] = restart_comparison
        if segment == 0:
            if control_step_changed or restart_step_changed or not control_timeline["bitwise_equal"] or not restart_timeline["bitwise_equal"]:
                _fail("QBit treatment changed segment A before the first checkpoint boundary")
        elif segment in (1, 2):
            control_bc_diff = control_bc_diff or control_step_changed or not control_timeline["bitwise_equal"]
            restart_bc_diff = restart_bc_diff or restart_step_changed or not restart_timeline["bitwise_equal"]

    if not control_bc_diff or not restart_bc_diff:
        _fail("QBit treatment produced no numerical difference from exact across B/C")
    return {
        "segments": control_segments,
        "restart_segments": restart_segments,
        "a_segment": a_segment,
        "treatment_changes_b_or_c": True,
        "scope": "QBit versus exact numerical difference across continuation segments B/C; no quality claim",
    }


def _checkpoint_accounting(
    exact_chain: list[dict[str, Any]],
    compact_chain: list[dict[str, Any]],
) -> dict[str, Any]:
    boundaries: dict[str, Any] = {}
    for boundary, (exact, compact) in enumerate(zip(exact_chain[:-1], compact_chain[:-1]), start=1):
        exact_state = exact["state"]
        compact_state = compact["state"]
        if exact_state is None or compact_state is None:
            _fail(f"boundary {boundary} lacks a comparable exact and QBit checkpoint")
        if compact_state["file_bytes"] >= exact_state["file_bytes"]:
            _fail(f"QBit checkpoint file is not smaller than exact at boundary {boundary}")
        if compact_state["logical_packed_kv_bytes"] >= compact_state["resident_f32_kv_bytes"]:
            _fail(f"QBit logical packed K/V does not reduce resident F32 bytes at boundary {boundary}")
        if compact_state["checkpoint_payload_bytes"] != exact_state["checkpoint_payload_bytes"]:
            _fail(f"exact/QBit checkpoint logical payload inventory differs at boundary {boundary}")
        boundaries[str(boundary)] = {
            "exact_state_sha256": exact_state["sha256"],
            "qbit_state_sha256": compact_state["sha256"],
            "exact_checkpoint_file_bytes": exact_state["file_bytes"],
            "qbit_checkpoint_file_bytes": compact_state["file_bytes"],
            "qbit_logical_packed_kv_bytes": compact_state["logical_packed_kv_bytes"],
            "resident_f32_kv_bytes": compact_state["resident_f32_kv_bytes"],
            "logical_checkpoint_payload_bytes": compact_state["checkpoint_payload_bytes"],
            "qbit_file_bytes_less_than_exact": True,
            "packed_and_file_bytes_are_distinct_fields": True,
        }
    return boundaries


def compare_index(index_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Authenticate a four-arm matrix and return its repeated-resume report."""
    path = _absolute_file(str(Path(index_path).absolute()), "matrix index")
    index = resume._read_json(path, label="matrix index")
    expected_index_keys = {"schema", "segments", "steps_per_segment", "arms", "source_bank"}
    if set(index) != expected_index_keys or index.get("schema") != INDEX_SCHEMA:
        _fail("matrix index schema or key inventory is unsupported")
    segments = _int(index.get("segments"), "index segments", minimum=3)
    if segments > 6:
        _fail("index segments exceeds the admitted six-segment sequence")
    steps = _int(index.get("steps_per_segment"), "index steps_per_segment", minimum=1)
    if steps != STEPS_PER_SEGMENT:
        _fail("index must include all four Euler steps per segment")

    source_bank, source_metadata = _load_source_bank(index)
    shape = source_metadata["shape"]
    if shape[:2] != [1, CHANNELS] or shape[3:] != [HEIGHT, WIDTH]:
        _fail("source bank geometry does not match the admitted DiT latent shape")
    if shape[2] < segments * SEGMENT_LATENTS or shape[2] % SEGMENT_LATENTS != 0:
        _fail("source bank temporal extent cannot cover the selected segments")
    if shape != index["source_bank"]["shape"]:
        _fail("source bank shape differs from the index declaration")

    arms = _load_arms(index, segments, steps, shape)
    identity = _validate_identities(arms, source_metadata)
    _validate_source_suffixes(arms, source_bank, shape)
    state_chains = _validate_state_chains(arms)

    exact_control = arms["exact_control"][0]
    exact_chain = arms["exact_restart_chain"]
    qbit_control = arms["qbit_control"][0]
    compact_chain = arms["compact_restart_chain"]
    exact_restart = _compare_equivalent_arms(
        exact_control, exact_chain, segments, steps, "exact control vs exact restart"
    )
    compact_restart = _compare_equivalent_arms(
        qbit_control, compact_chain, segments, steps, "QBit control vs compact restart"
    )
    lossy = _lossy_comparison(
        exact_control, qbit_control, exact_chain, compact_chain, segments, steps
    )
    checkpoint_bytes = _checkpoint_accounting(exact_chain, compact_chain)

    segment_zero_timeline = _time_slice(exact_control["timeline"], shape, 0, SEGMENT_LATENTS)
    a_prefix = {
        "shape": [1, CHANNELS, SEGMENT_LATENTS, HEIGHT, WIDTH],
        "source_bank_suffix_untouched_for_every_interval": True,
        "segment_a_bitwise_unchanged_by_qbit": True,
        "segment_a_timeline_bytes": len(segment_zero_timeline),
    }
    artifacts = {
        arm: [
            {
                "interval": list(run["interval"]),
                "receipt_path": str(run["receipt_path"]),
                "timeline": run["timeline_meta"],
                "steps": {
                    key: metadata
                    for key, (_, metadata) in sorted(run["steps"].items())
                },
                "state": run["state"],
            }
            for run in runs
        ]
        for arm, runs in arms.items()
    }
    return {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "scope": f"{segments}-segment native DiT continuation; no decoder, full upstream parity, or visual quality claim",
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "identity": identity,
        "source_bank": source_metadata,
        "intervals": {
            arm: [list(run["interval"]) for run in runs]
            for arm, runs in arms.items()
        },
        "metrics": {
            arm: [run["receipt"]["metrics"] for run in runs]
            for arm, runs in arms.items()
        },
        "state_chains": state_chains,
        "exact_restart": exact_restart,
        "compact_restart": compact_restart,
        "lossy_vs_exact": lossy,
        "checkpoint_accounting_by_boundary": checkpoint_bytes,
        "a_prefix_and_future": a_prefix,
        "artifacts": artifacts,
        "finite": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Authenticated four-arm matrix JSON")
    parser.add_argument("--output", required=True, help="Fresh JSON report path")
    args = parser.parse_args(argv)
    try:
        report = compare_index(args.index)
        output = Path(args.output).absolute()
        resume._write_fresh_json(output, report)
    except (OSError, ValueError) as error:
        print(f"compact resume reference rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": report["passed"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
