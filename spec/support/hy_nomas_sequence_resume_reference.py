#!/usr/bin/env python3
"""Authenticate and compare CPU-readable receipts from the bounded native
Hy-Nomas sequence save/restart qualification.

This checks native continuation parity only. It does not execute the model,
load an upstream module, or claim full-video generation parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any


RECEIPT_SCHEMA = "cogni_ml.hy_nomas_sequence_resume_probe/v1"
REPORT_SCHEMA = "cogni_ml.hy_nomas_sequence_resume_reference/v1"
ATOL = RTOL = 2e-5
TIMELINE_SHAPE = (1, 32, 180, 9, 16)
STEP_SHAPE = (1, 32, 30, 9, 16)
F32_BYTES = 4
TIMELINE_BYTES = math.prod(TIMELINE_SHAPE) * F32_BYTES
STEP_BYTES = math.prod(STEP_SHAPE) * F32_BYTES
STATE_PAYLOAD_OFFSET = 1_048_588
STATE_FILENAME = "after-a.hnrs"
ARTIFACT_READ_CHUNK_BYTES = 1024 * 1024
STEP_NAMES = tuple(f"b-step{i}" for i in range(1, 5))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ResumeReferenceError(ValueError):
    """A receipt or artifact does not satisfy the admitted probe contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResumeReferenceError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ResumeReferenceError(f"invalid JSON numeric constant: {value}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ResumeReferenceError(f"{label} must be a regular non-symlink file")
    try:
        before = path.stat(follow_symlinks=False)
        if before.st_size > 1024 * 1024:
            raise ResumeReferenceError(f"{label} exceeds the 1 MiB metadata limit")
        raw = path.read_bytes()
        after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ResumeReferenceError(f"cannot read {label}") from error
    if _stat_identity(before) != _stat_identity(after) or len(raw) != after.st_size:
        raise ResumeReferenceError(f"{label} changed while being read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResumeReferenceError(f"cannot parse {label} JSON") from error
    if type(value) is not dict:
        raise ResumeReferenceError(f"{label} must be a JSON object")
    return value


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _required_string(document: dict[str, Any], key: str, label: str) -> str:
    value = document.get(key)
    if type(value) is not str or not value:
        raise ResumeReferenceError(f"{label} {key} must be a non-empty string")
    return value


def _required_sha(document: dict[str, Any], key: str, label: str) -> str:
    value = _required_string(document, key, label)
    if SHA256_RE.fullmatch(value) is None:
        raise ResumeReferenceError(f"{label} {key} must be lowercase SHA-256")
    return value


def _validate_receipt(document: dict[str, Any], mode: str) -> None:
    label = f"{mode} receipt"
    if document.get("schema") != RECEIPT_SCHEMA:
        raise ResumeReferenceError(f"{label} schema is unsupported")
    if document.get("mode") != mode:
        raise ResumeReferenceError(f"{label} mode must be {mode}")
    if type(document.get("pid")) is not int or document["pid"] <= 0:
        raise ResumeReferenceError(f"{label} pid must be a positive integer")
    _required_string(document, "inputs", label)
    _required_sha(document, "inputs_sha256", label)
    _required_sha(document, "contract_sha256", label)
    if type(document.get("state_input_sha256")) is not str:
        raise ResumeReferenceError(f"{label} state_input_sha256 must be a string")
    if type(document.get("upstream_revision")) is not str or not document["upstream_revision"]:
        raise ResumeReferenceError(f"{label} upstream_revision must be a non-empty string")
    if type(document.get("checkpoint_payload_bytes")) is not int or document["checkpoint_payload_bytes"] <= 0:
        raise ResumeReferenceError(f"{label} checkpoint_payload_bytes must be a positive integer")

    for key, expected in (
        ("segments", 2),
        ("resume_next_segment", 1),
        ("steps_per_segment", 4),
    ):
        if type(document.get(key)) is not int or document[key] != expected:
            raise ResumeReferenceError(f"{label} {key} must be {expected}")
    for key, expected in (
        ("timeline_shape", list(TIMELINE_SHAPE)),
        ("step_shape", list(STEP_SHAPE)),
    ):
        value = document.get(key)
        if type(value) is not list or any(type(dim) is not int for dim in value) or value != expected:
            raise ResumeReferenceError(f"{label} {key} does not match the admitted geometry")

    expected_capture_bytes = 0 if mode == "save" else 4 * STEP_BYTES
    if type(document.get("capture_bytes")) is not int or document["capture_bytes"] != expected_capture_bytes:
        raise ResumeReferenceError(f"{label} capture_bytes does not match the probe mode")
    for key in ("dit_peak_named_bytes", "peak_named_bytes"):
        if type(document.get(key)) is not int or document[key] <= 0:
            raise ResumeReferenceError(f"{label} {key} must be a positive integer")
    if document["peak_named_bytes"] != document["dit_peak_named_bytes"] + expected_capture_bytes:
        raise ResumeReferenceError(f"{label} peak_named_bytes does not match DiT plus capture buffers")

    for key in ("decoder", "full_generation"):
        if type(document.get(key)) is not bool or document[key] is not False:
            raise ResumeReferenceError(f"{label} {key} must be false")

    metrics = document.get("metrics")
    if type(metrics) is not dict:
        raise ResumeReferenceError(f"{label} metrics must be an object")
    if set(metrics) != {
        "encode_allocations", "encode_readbacks", "preterminal_waits",
        "command_segments", "terminal_waits",
    }:
        raise ResumeReferenceError(f"{label} metrics must name exactly the five instrumented counters")
    if any(type(value) is not int or value < 0 for value in metrics.values()):
        raise ResumeReferenceError(f"{label} metrics must be non-negative integers")
    for key in ("encode_allocations", "encode_readbacks", "preterminal_waits"):
        if type(metrics.get(key)) is not int or metrics[key] != 0:
            raise ResumeReferenceError(f"{label} metric {key} must be zero")
    if type(metrics.get("terminal_waits")) is not int or metrics["terminal_waits"] != 1:
        raise ResumeReferenceError(f"{label} metric terminal_waits must be one")

    if mode in ("control", "save") and document["state_input_sha256"] != "":
        raise ResumeReferenceError(f"{label} must not claim an input state")
    if mode == "resume" and SHA256_RE.fullmatch(document["state_input_sha256"]) is None:
        raise ResumeReferenceError("resume receipt state_input_sha256 must be lowercase SHA-256")


def _directory(path: str | os.PathLike[str], label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ResumeReferenceError(f"{label} must be an existing non-symlink directory")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ResumeReferenceError(f"cannot resolve {label}") from error
    return resolved


def _read_artifact(
    directory: Path,
    diagnostics: dict[str, Any],
    name: str,
    filename: str,
    expected_bytes: int,
    label: str,
    *,
    retain_payload: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    path_value = diagnostics.get(name)
    sha_value = diagnostics.get(f"{name}_sha256")
    if type(path_value) is not str or not path_value:
        raise ResumeReferenceError(f"{label} diagnostics missing {name} path")
    if type(sha_value) is not str or SHA256_RE.fullmatch(sha_value) is None:
        raise ResumeReferenceError(f"{label} diagnostics {name}_sha256 is malformed")
    raw_path = Path(path_value)
    if not raw_path.is_absolute() or raw_path.name != filename:
        raise ResumeReferenceError(f"{label} {name} path must be the absolute {filename} sibling")
    try:
        parent = raw_path.parent.resolve(strict=True)
        if parent != directory:
            raise ResumeReferenceError(f"{label} {name} must be beside receipt.json")
        before = os.stat(raw_path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ResumeReferenceError(f"{label} {name} must be a regular non-symlink file")
        if before.st_size != expected_bytes:
            raise ResumeReferenceError(
                f"{label} {name} has {before.st_size} bytes; expected {expected_bytes}"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(raw_path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _stat_identity(opened) != _stat_identity(before):
                raise ResumeReferenceError(f"{label} {name} changed before it was opened")
            digest = hashlib.sha256()
            chunks: list[bytes] | None = [] if retain_payload else None
            bytes_read = 0
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while bytes_read <= expected_bytes:
                    read_size = min(ARTIFACT_READ_CHUNK_BYTES, expected_bytes + 1 - bytes_read)
                    chunk = stream.read(read_size)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    digest.update(chunk)
                    if chunks is not None:
                        chunks.append(chunk)
                    if bytes_read > expected_bytes:
                        break
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.stat(raw_path, follow_symlinks=False)
    except ResumeReferenceError:
        raise
    except OSError as error:
        raise ResumeReferenceError(f"cannot read {label} {name} artifact") from error
    if (
        _stat_identity(before) != _stat_identity(after_open)
        or _stat_identity(before) != _stat_identity(after_path)
        or bytes_read != expected_bytes
    ):
        raise ResumeReferenceError(f"{label} {name} changed while being read")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != sha_value:
        raise ResumeReferenceError(f"{label} {name} SHA-256 does not match its receipt")
    payload = b"" if chunks is None else b"".join(chunks)
    return payload, {"path": str(raw_path), "bytes": expected_bytes, "sha256": actual_sha256}


def _finite(payload: bytes, label: str) -> None:
    for (value,) in struct.iter_unpack("<f", payload):
        if not math.isfinite(value):
            raise ResumeReferenceError(f"{label} contains a non-finite F32 value")


def _compare_f32(left: bytes, right: bytes, label: str) -> dict[str, Any]:
    if len(left) != len(right) or len(left) % F32_BYTES:
        raise ResumeReferenceError(f"{label} byte extents do not match F32 geometry")
    max_abs = 0.0
    max_rel = 0.0
    within = True
    count = len(left) // F32_BYTES
    for (reference,), (actual,) in zip(struct.iter_unpack("<f", left), struct.iter_unpack("<f", right)):
        if not math.isfinite(reference) or not math.isfinite(actual):
            raise ResumeReferenceError(f"{label} contains a non-finite F32 value")
        error = abs(actual - reference)
        max_abs = max(max_abs, error)
        max_rel = max(max_rel, error / max(abs(reference), 1e-30))
        if error > ATOL + RTOL * abs(reference):
            within = False
    return {
        "elements": count,
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "bitwise_equal": left == right,
        "finite": True,
        "within_tolerance": within,
        "tolerance": f"abs(actual-reference) <= {ATOL:g} + {RTOL:g}*abs(reference)",
    }


def _a_prefix(payload: bytes) -> bytes:
    if len(payload) != TIMELINE_BYTES:
        raise ResumeReferenceError("timeline byte extent does not match the admitted geometry")
    channel_stride = TIMELINE_SHAPE[2] * TIMELINE_SHAPE[3] * TIMELINE_SHAPE[4] * F32_BYTES
    prefix_stride = 30 * TIMELINE_SHAPE[3] * TIMELINE_SHAPE[4] * F32_BYTES
    return b"".join(
        payload[channel * channel_stride : channel * channel_stride + prefix_stride]
        for channel in range(TIMELINE_SHAPE[1])
    )


def compare_runs(
    control_dir: str | os.PathLike[str],
    save_dir: str | os.PathLike[str],
    resume_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Authenticate three native receipts and report bounded restart parity."""
    directories = {
        "control": _directory(control_dir, "control directory"),
        "save": _directory(save_dir, "save directory"),
        "resume": _directory(resume_dir, "resume directory"),
    }
    if len(set(directories.values())) != 3:
        raise ResumeReferenceError("control, save, and resume directories must be distinct")

    receipts: dict[str, dict[str, Any]] = {}
    for mode, directory in directories.items():
        receipt = _read_json(directory / "receipt.json", label=f"{mode} receipt")
        _validate_receipt(receipt, mode)
        receipts[mode] = receipt

    pids = {mode: receipt["pid"] for mode, receipt in receipts.items()}
    if len(set(pids.values())) != 3:
        raise ResumeReferenceError("control, save, and resume must have distinct process IDs")

    for key in ("inputs", "inputs_sha256", "contract_sha256", "upstream_revision"):
        values = {receipt[key] for receipt in receipts.values()}
        if len(values) != 1:
            raise ResumeReferenceError(f"all modes must have identical {key}")
    if receipts["resume"]["state_input_sha256"] != receipts["save"]["diagnostics"].get("state_sha256"):
        raise ResumeReferenceError("resume input state SHA-256 does not match the saved boundary")

    artifacts: dict[str, dict[str, tuple[bytes, dict[str, Any]]]] = {}
    for mode, receipt in receipts.items():
        diagnostics = receipt.get("diagnostics")
        if type(diagnostics) is not dict:
            raise ResumeReferenceError(f"{mode} receipt diagnostics must be an object")
        names = ["timeline", "timeline_sha256"]
        if mode == "save":
            names.extend(("state", "state_sha256"))
        else:
            for step_name in STEP_NAMES:
                names.extend((step_name, f"{step_name}_sha256"))
        if set(diagnostics) != set(names):
            raise ResumeReferenceError(f"{mode} diagnostics must name exactly the required artifacts")

        mode_artifacts: dict[str, tuple[bytes, dict[str, Any]]] = {}
        payload, metadata = _read_artifact(
            directories[mode], diagnostics, "timeline", "timeline.f32le", TIMELINE_BYTES, mode
        )
        _finite(payload, f"{mode} timeline")
        mode_artifacts["timeline"] = (payload, metadata)
        if mode == "save":
            payload_bytes = receipt.get("checkpoint_payload_bytes")
            if type(payload_bytes) is not int or payload_bytes <= 0:
                raise ResumeReferenceError("save receipt checkpoint_payload_bytes must be positive")
            state_bytes = STATE_PAYLOAD_OFFSET + payload_bytes
            payload, metadata = _read_artifact(
                directories[mode], diagnostics, "state", STATE_FILENAME, state_bytes, mode,
                retain_payload=False,
            )
            mode_artifacts["state"] = (payload, metadata)
        else:
            for step_name in STEP_NAMES:
                payload, metadata = _read_artifact(
                    directories[mode], diagnostics, step_name, f"{step_name}.f32le", STEP_BYTES, mode
                )
                _finite(payload, f"{mode} {step_name}")
                mode_artifacts[step_name] = (payload, metadata)
        artifacts[mode] = mode_artifacts

    payload_sizes = {receipts[mode]["checkpoint_payload_bytes"] for mode in receipts}
    if len(payload_sizes) != 1 or type(receipts["save"]["checkpoint_payload_bytes"]) is not int:
        raise ResumeReferenceError("all modes must report the same checkpoint payload size")

    prefixes = {mode: _a_prefix(artifacts[mode]["timeline"][0]) for mode in receipts}
    a_prefix = {
        "shape": [1, 32, 30, 9, 16],
        "elements": 1 * 32 * 30 * 9 * 16,
        "control_save_bitwise_equal": prefixes["control"] == prefixes["save"],
        "control_resume_bitwise_equal": prefixes["control"] == prefixes["resume"],
        "save_resume_bitwise_equal": prefixes["save"] == prefixes["resume"],
    }
    if not a_prefix["control_save_bitwise_equal"] or not a_prefix["control_resume_bitwise_equal"]:
        raise ResumeReferenceError("A's first 30 timeline frames are not bitwise identical")

    timeline_parity = _compare_f32(
        artifacts["control"]["timeline"][0], artifacts["resume"]["timeline"][0], "timeline"
    )
    if not timeline_parity["within_tolerance"]:
        raise ResumeReferenceError("control/resume timeline values exceed tolerance")

    step_parity: dict[str, Any] = {}
    for step_name in STEP_NAMES:
        result = _compare_f32(
            artifacts["control"][step_name][0], artifacts["resume"][step_name][0], step_name
        )
        if not result["within_tolerance"]:
            raise ResumeReferenceError(f"control/resume {step_name} values exceed tolerance")
        step_parity[step_name] = result

    report = {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "scope": "native two-segment DiT continuation; no decoder or full generation",
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "identity": {
            "pids": pids,
            "distinct_processes": True,
            "inputs": receipts["control"]["inputs"],
            "inputs_sha256": receipts["control"]["inputs_sha256"],
            "contract_sha256": receipts["control"]["contract_sha256"],
            "upstream_revision": receipts["control"]["upstream_revision"],
            "saved_state_sha256": receipts["save"]["diagnostics"]["state_sha256"],
            "resume_state_input_sha256": receipts["resume"]["state_input_sha256"],
        },
        "metrics": {
            mode: receipts[mode]["metrics"]
            for mode in ("control", "save", "resume")
        },
        "artifacts": {
            mode: {name: metadata for name, (_, metadata) in artifacts[mode].items()}
            for mode in ("control", "save", "resume")
        },
        "finite": True,
        "a_prefix": a_prefix,
        "timeline_control_resume": timeline_parity,
        "b_steps_control_resume": step_parity,
    }
    return report


def _write_fresh_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ResumeReferenceError("output report path must be fresh")
    parent = path.parent.resolve(strict=True)
    if path.name in {"", ".", ".."}:
        raise ResumeReferenceError("output report path must name a file")
    destination = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as error:
        raise ResumeReferenceError("cannot create fresh output report") from error
    try:
        encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True, help="Native control output directory")
    parser.add_argument("--save-dir", required=True, help="Native A-boundary save output directory")
    parser.add_argument("--resume-dir", required=True, help="Fresh-process resume output directory")
    parser.add_argument("--output", required=True, help="Fresh JSON report path")
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.control_dir, args.save_dir, args.resume_dir)
        output = Path(args.output).absolute()
        _write_fresh_json(output, report)
    except (OSError, ResumeReferenceError) as error:
        print(f"resume reference rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": report["passed"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
