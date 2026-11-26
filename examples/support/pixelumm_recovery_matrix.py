#!/usr/bin/env python3
"""Launch and validate the bounded PixelUMM A/A recovery-order experiment.

Each cell is a fresh guarded process with one warmup and two identical saved-
latent decoder requests. The fixed cell order is short-long, then long-short.
This is descriptive timing evidence only; it is not full generation, a DiT
measurement, or a GPU-frequency measurement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
GUARD = ROOT / "examples" / "support" / "wan_memory_guard.py"
MATRIX_SCRIPT = ROOT / "examples" / "support" / "pixelumm_after_dit_matrix.py"
MIB = 1024 * 1024
GIB = 1024 * MIB
OUTPUT_BYTES = 1 * 3 * 120 * 416 * 768 * 4
CELL_ORDERS = ("short-long", "long-short")
RECOVERY_SECONDS = 120
RECOVERY_WAIT_CHUNK_SECONDS = 30
SCHEMA = "cogni_ml.pixelumm_recovery_probe/v1"
RECOVERY_CLOCK = "CLOCK_MONOTONIC_RAW(4)"
PINNED = {
    "input_sha256": "dcbe51749742701a2ee1a6d09a10e2f20e663f3ff54f646d801c743e17c7a104",
    "latent_sha256": "ea79dc6896f3e956035acd0caa69b1730554d5459d80893989023493de4b5bba",
    "noise_sha256": "166ea19d2812c0b4dedfedefd4f101b2abe8b4ec48364fb63705f781e1828547",
    "oracle_report_sha256": "a91ad030b8cd682f4ca6c00ebfad41c4a3ea56a9bcff41155318f6fb12d2ea7e",
}
OUTPUT_SHA256 = "5ed0974dfd3b28d940c370fa8faef6a23b288d86adb03c6810758ae16c55cc9f"
GUARD_LIMITS = {
    "max_footprint_mib": 26_624,
    "max_swap_growth_mib": 64,
    "timeout_sec": 1_200,
    "memory_accounting": "file-cache",
    "runtime_floor_pct": 10,
    "runtime_floor_pct_exact": "10",
}
SOURCE_FILES = (
    "Makefile",
    "examples/pixelumm_after_dit_probe.cr",
    "examples/pixelumm_recovery_probe.cr",
    "examples/support/pixelumm_saved_latent.cr",
    "examples/support/pixelumm_recovery_host.mm",
    "examples/support/pixelumm_recovery_timing.cr",
    "examples/support/pixelumm_after_dit_matrix.py",
    "examples/support/pixelumm_recovery_matrix.py",
    "examples/support/wan_memory_guard.py",
)
OPTIONAL_SOURCE_FILES = ("examples/support/pixelumm_power_observer.py",)


def _load_matrix_utilities():
    """Load the existing pure hash, guard, output-root, and JSON helpers."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("pixelumm_after_dit_matrix", MATRIX_SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load existing PixelUMM matrix helpers: {MATRIX_SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


MATRIX = _load_matrix_utilities()
ValidationError = MATRIX.ValidationError
_require = MATRIX._require
_is_sha256 = MATRIX._is_sha256
_positive_int = MATRIX._positive_int
_finite_number = MATRIX._finite_number
_read_json = MATRIX._read_json
_write_new_json = MATRIX._write_new_json
_append_jsonl = MATRIX._append_jsonl
_sha256_file = MATRIX.sha256_file
_sha256_bytes = MATRIX.sha256_bytes
_utc_now = MATRIX._utc_now
_binary_identity = MATRIX._binary_identity
_output_root_candidate = MATRIX._output_root_candidate


def _source_identity() -> dict[str, Any]:
    paths = {ROOT / relative for relative in SOURCE_FILES}
    paths.update(ROOT / relative for relative in OPTIONAL_SOURCE_FILES
                 if (ROOT / relative).is_file())
    for pattern in MATRIX.CELL_SOURCE_GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    files: dict[str, str] = {}
    for path in sorted(paths):
        _require(path.is_file() and not path.is_symlink(),
                 f"source file is missing or symlinked: {path}")
        files[str(path.relative_to(ROOT))] = _sha256_file(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT, check=True,
                            capture_output=True, text=True, timeout=10).stdout
    return {
        "git_head": head,
        "working_tree_status_sha256": _sha256_bytes(status.encode("utf-8")),
        "files": files,
        "manifest_sha256": _sha256_bytes(encoded),
        "matrix_script_sha256": files["examples/support/pixelumm_after_dit_matrix.py"],
        "recovery_script_sha256": files["examples/support/pixelumm_recovery_matrix.py"],
        "probe_source_sha256": files["examples/pixelumm_recovery_probe.cr"],
        "fixture_source_sha256": files["examples/support/pixelumm_saved_latent.cr"],
        "host_observer_sha256": files["examples/support/pixelumm_recovery_host.mm"],
        "recovery_timing_sha256": files["examples/support/pixelumm_recovery_timing.cr"],
        "guard_source_sha256": files["examples/support/wan_memory_guard.py"],
    }


def _thermal_state() -> dict[str, str | None]:
    return MATRIX._thermal_state()


def _cell_schedule() -> list[dict[str, Any]]:
    return [
        {"cell_index": index, "cell_name": f"cell-{index:02d}-{order}",
         "recovery_order": order, "mode": order}
        for index, order in enumerate(CELL_ORDERS, 1)
    ]


def _command(binary: dict[str, Any], recovery_order: str, output_dir: Path,
             guard_log: Path) -> list[str]:
    return [
        sys.executable, "-B", str(GUARD),
        "--memory-accounting=file-cache",
        f"--max-footprint-mib={GUARD_LIMITS['max_footprint_mib']}",
        f"--max-swap-growth-mib={GUARD_LIMITS['max_swap_growth_mib']}",
        f"--timeout-sec={GUARD_LIMITS['timeout_sec']}",
        "--runtime-floor-pct=10",
        "--log=" + str(guard_log), "--",
        binary["path"], "--recovery-order", recovery_order, "--output-dir", str(output_dir),
    ]


def _make_launch_manifest(binary: dict[str, Any], output_root: Path) -> dict[str, Any]:
    source = _source_identity()
    return {
        "schema": "cogni_ml.pixelumm_recovery_matrix_launch/v1",
        "created_utc": _utc_now(),
        "repository_root": str(ROOT),
        "output_root": str(output_root),
        "binary": binary,
        "source": source,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "macos_version": platform.mac_ver()[0],
            "thermal_state": _thermal_state(),
        },
        "schedule": _cell_schedule(),
        "pre_cell_recovery_seconds": RECOVERY_SECONDS,
        "recovery_wait_chunk_max_seconds": RECOVERY_WAIT_CHUNK_SECONDS,
        "guard_limits": GUARD_LIMITS,
        "scope": "saved-latent PixelUMM decoder recovery-order A/A; no DiT or full generation",
        "claim_limit": "descriptive within-cell short/long timing only; thermal state is not GPU frequency or power",
    }


def dry_run_plan(binary_path: str, output_root_path: str) -> dict[str, Any]:
    binary = _binary_identity(binary_path)
    output_root = _output_root_candidate(output_root_path)
    launch = _make_launch_manifest(binary, output_root)
    cells = []
    for item in launch["schedule"]:
        cell_dir = output_root / item["cell_name"]
        output_dir = cell_dir / "output"
        guard_log = cell_dir / "guard.jsonl"
        cells.append({
            **item,
            "output_dir": str(output_dir),
            "guard_command": _command(binary, item["recovery_order"], output_dir, guard_log),
            "pre_cell_recovery_seconds": RECOVERY_SECONDS,
        })
    return {"dry_run": True, "launch": launch, "cells": cells}


def _wait_for_recovery(log_path: Path, item: dict[str, Any],
                        seconds: int = RECOVERY_SECONDS, *,
                        sleep_fn: Callable[[float], None] | None = None,
                        monotonic_fn: Callable[[], float] | None = None) -> float:
    sleeper = sleep_fn or time.sleep
    monotonic = monotonic_fn or time.monotonic
    _require(isinstance(seconds, int) and not isinstance(seconds, bool) and seconds > 0,
             "pre-cell recovery duration must be a positive integer")
    start = monotonic()
    _append_jsonl(log_path, {
        "event": "recovery_wait_started", "utc": _utc_now(),
        "cell_index": item["cell_index"], "cell_name": item["cell_name"],
        "recovery_order": item["recovery_order"], "requested_seconds": seconds,
    })
    while True:
        elapsed = monotonic() - start
        remaining = seconds - elapsed
        if remaining <= 0:
            break
        chunk = min(float(RECOVERY_WAIT_CHUNK_SECONDS), remaining)
        chunk_start = monotonic()
        sleeper(chunk)
        chunk_elapsed = monotonic() - chunk_start
        _require(math.isfinite(chunk_elapsed) and chunk_elapsed > 0.0,
                 "pre-cell recovery wait did not advance the monotonic clock")
        elapsed = monotonic() - start
        _append_jsonl(log_path, {
            "event": "recovery_wait_progress", "utc": _utc_now(),
            "cell_index": item["cell_index"], "cell_name": item["cell_name"],
            "recovery_order": item["recovery_order"],
            "chunk_requested_seconds": chunk,
            "chunk_elapsed_seconds": chunk_elapsed,
            "elapsed_seconds": elapsed,
            "remaining_seconds": max(0.0, seconds - elapsed),
        })
    actual = monotonic() - start
    _require(actual >= seconds, "pre-cell recovery ended before its requested duration")
    _append_jsonl(log_path, {
        "event": "recovery_wait_completed", "utc": _utc_now(),
        "cell_index": item["cell_index"], "cell_name": item["cell_name"],
        "recovery_order": item["recovery_order"],
        "requested_seconds": seconds, "actual_seconds": actual,
    })
    return actual


def _write_guard_identity(guard_log: Path, item: dict[str, Any],
                          launch: dict[str, Any]) -> None:
    _require(guard_log.is_file(), f"guard did not create its log: {guard_log}")
    _append_jsonl(guard_log, {
        "event": "matrix_cell_identity", "utc": _utc_now(),
        "cell_index": item["cell_index"], "cell_name": item["cell_name"],
        "mode": item["mode"], "binary_path": launch["binary"]["path"],
        "binary_sha256": launch["binary"]["sha256"],
        "source_manifest_sha256": launch["source"]["manifest_sha256"],
        "matrix_script_sha256": launch["source"]["matrix_script_sha256"],
        "probe_source_sha256": launch["source"]["probe_source_sha256"],
    })


def _check_live_identities(launch: dict[str, Any]) -> None:
    _require(_binary_identity(launch["binary"]["path"]) == launch["binary"],
             "binary identity changed during the recovery matrix")
    _require(_source_identity() == launch["source"],
             "source identity changed during the recovery matrix")


def _run_cell(output_root: Path, item: dict[str, Any],
              launch: dict[str, Any]) -> dict[str, Any]:
    _check_live_identities(launch)
    cell_dir = output_root / item["cell_name"]
    cell_dir.mkdir(mode=0o700, exist_ok=False)
    output_dir = cell_dir / "output"
    output_dir.mkdir(mode=0o700, exist_ok=False)
    wait_log = cell_dir / "recovery_wait.jsonl"
    MATRIX._write_exclusive_log(wait_log, "")
    recovery_seconds = _wait_for_recovery(wait_log, item)
    _check_live_identities(launch)

    guard_log = cell_dir / "guard.jsonl"
    stdout_path = cell_dir / "stdout.log"
    stderr_path = cell_dir / "stderr.log"
    command = _command(launch["binary"], item["recovery_order"], output_dir, guard_log)
    intent = {
        "schema": "cogni_ml.pixelumm_recovery_matrix_cell_intent/v1",
        **item,
        "command": command,
        "binary": launch["binary"],
        "source_manifest_sha256": launch["source"]["manifest_sha256"],
        "guard_limits": GUARD_LIMITS,
        "output_dir": str(output_dir),
        "recovery_wait_sha256": _sha256_file(wait_log),
        "pre_cell_recovery_seconds": recovery_seconds,
        "thermal_before_cell": _thermal_state(),
        "created_utc": _utc_now(),
    }
    _write_new_json(cell_dir / "intent.json", intent)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.monotonic()
    returncode: int | None = None
    process_id: int | None = None
    launch_error: str | None = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            os.fchmod(stdout.fileno(), 0o600)
            os.fchmod(stderr.fileno(), 0o600)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            process_id = process.pid
            returncode = process.wait()
    except (OSError, subprocess.SubprocessError) as error:
        launch_error = f"{type(error).__name__}: {error}"
    elapsed = time.monotonic() - start
    if guard_log.is_file():
        _write_guard_identity(guard_log, item, launch)
    completion = {
        "schema": "cogni_ml.pixelumm_recovery_matrix_cell_completion/v1",
        "cell_index": item["cell_index"], "cell_name": item["cell_name"],
        "recovery_order": item["recovery_order"], "mode": item["mode"],
        "process_id": process_id, "returncode": returncode, "launch_error": launch_error,
        "elapsed_seconds": elapsed, "finished_utc": _utc_now(),
        "binary_sha256": launch["binary"]["sha256"],
        "source_manifest_sha256": launch["source"]["manifest_sha256"],
        "pre_cell_recovery_seconds": recovery_seconds,
        "recovery_wait_sha256": _sha256_file(wait_log),
        "guard_sha256": _sha256_file(guard_log) if guard_log.is_file() else None,
        "stdout_sha256": _sha256_file(stdout_path) if stdout_path.is_file() else None,
        "stderr_sha256": _sha256_file(stderr_path) if stderr_path.is_file() else None,
        "receipt_sha256": _sha256_file(output_dir / "receipt.json")
        if (output_dir / "receipt.json").is_file() else None,
        "thermal_after_cell": _thermal_state(),
    }
    _write_new_json(cell_dir / "completion.json", completion)
    _check_live_identities(launch)
    _require(returncode == 0 and launch_error is None,
             f"{item['cell_name']} failed under the memory guard (returncode={returncode}, error={launch_error})")
    return completion


def launch_matrix(binary_path: str, output_root_path: str) -> dict[str, Any]:
    plan = dry_run_plan(binary_path, output_root_path)
    output_root = Path(plan["launch"]["output_root"])
    output_root.mkdir(mode=0o700, exist_ok=False)
    launch = plan["launch"]
    events_path = output_root / "events.jsonl"
    completions = []
    try:
        _write_new_json(output_root / "launch.json", launch)
        MATRIX._write_exclusive_log(events_path, "")
        _append_jsonl(events_path, {
            "event": "recovery_matrix_started", "utc": _utc_now(),
            "schedule": CELL_ORDERS, "binary_sha256": launch["binary"]["sha256"],
            "source_manifest_sha256": launch["source"]["manifest_sha256"],
        })
        for item in launch["schedule"]:
            completion = _run_cell(output_root, item, launch)
            completions.append(completion)
            _append_jsonl(events_path, {
                "event": "recovery_matrix_cell_completed", "utc": _utc_now(),
                "cell_index": item["cell_index"], "cell_name": item["cell_name"],
                "recovery_order": item["recovery_order"],
                "process_id": completion["process_id"],
            })
        report = analyze_matrix(output_root)
        _write_new_json(output_root / "analysis.json", report)
        _write_new_json(output_root / "completion.json", {
            "schema": "cogni_ml.pixelumm_recovery_matrix_completion/v1",
            "passed": True, "cell_completions": completions,
            "analysis_sha256": _sha256_file(output_root / "analysis.json"),
            "events_sha256": _sha256_file(events_path), "finished_utc": _utc_now(),
        })
        return report
    except BaseException as error:
        failure = {
            "schema": "cogni_ml.pixelumm_recovery_matrix_failure/v1",
            "passed": False, "completed_cells": len(completions),
            "error": f"{type(error).__name__}: {error}", "finished_utc": _utc_now(),
        }
        try:
            _append_jsonl(events_path, {
                "event": "recovery_matrix_failed", "utc": _utc_now(),
                "completed_cells": len(completions),
                "error": f"{type(error).__name__}: {error}",
            })
        except Exception:
            pass
        try:
            if events_path.is_file():
                failure["events_sha256"] = _sha256_file(events_path)
            _write_new_json(output_root / "failure.json", failure)
        except Exception:
            pass
        raise


def _validate_source_manifest(source: Any) -> None:
    _require(isinstance(source, dict) and isinstance(source.get("files"), dict),
             "launch source identity is incomplete")
    files = source["files"]
    _require(all(isinstance(path, str) and _is_sha256(digest) for path, digest in files.items()),
             "launch source file digest is malformed")
    required = set(SOURCE_FILES)
    _require(required <= set(files), "launch source manifest omits a required source")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _require(_sha256_bytes(encoded) == source.get("manifest_sha256"),
             "launch source manifest digest mismatch")
    for path, field in (
        ("examples/support/pixelumm_after_dit_matrix.py", "matrix_script_sha256"),
        ("examples/support/pixelumm_recovery_matrix.py", "recovery_script_sha256"),
        ("examples/pixelumm_recovery_probe.cr", "probe_source_sha256"),
        ("examples/support/pixelumm_saved_latent.cr", "fixture_source_sha256"),
        ("examples/support/pixelumm_recovery_host.mm", "host_observer_sha256"),
        ("examples/support/pixelumm_recovery_timing.cr", "recovery_timing_sha256"),
        ("examples/support/wan_memory_guard.py", "guard_source_sha256"),
    ):
        _require(source.get(field) == files.get(path),
                 f"launch source file digest entries are inconsistent: {field}")
    optional_observer = "examples/support/pixelumm_power_observer.py"
    if optional_observer in files:
        _require(_is_sha256(files[optional_observer]),
                 "optional power observer source digest is malformed")


def _validate_wait_log(path: Path, item: dict[str, Any], expected_seconds: int,
                       expected_actual: Any) -> list[dict[str, Any]]:
    _require(path.is_file(), f"{item['cell_name']} pre-cell recovery log is missing")
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{item['cell_name']} pre-cell recovery log is invalid JSONL") from error
    _require(events and all(isinstance(event, dict) for event in events),
             f"{item['cell_name']} pre-cell recovery log is empty or malformed")
    _require(events[0].get("event") == "recovery_wait_started" and
             events[-1].get("event") == "recovery_wait_completed",
             f"{item['cell_name']} pre-cell recovery log is incomplete")
    progress = [event for event in events if event.get("event") == "recovery_wait_progress"]
    _require(len(events) == len(progress) + 2 and progress,
             f"{item['cell_name']} pre-cell recovery log has unexpected events")
    for event in events:
        for field, value in {
            "cell_index": item["cell_index"], "cell_name": item["cell_name"],
            "recovery_order": item["recovery_order"],
        }.items():
            _require(event.get(field) == value,
                     f"{item['cell_name']} pre-cell recovery identity mismatch: {field}")
    _require(events[0].get("requested_seconds") == expected_seconds and
             events[-1].get("requested_seconds") == expected_seconds,
             f"{item['cell_name']} pre-cell recovery duration request mismatch")
    elapsed_sum = 0.0
    prior_elapsed = 0.0
    for event in progress:
        requested = _finite_number(event.get("chunk_requested_seconds"),
                                   f"{item['cell_name']}.wait.chunk_requested_seconds")
        actual = _finite_number(event.get("chunk_elapsed_seconds"),
                                f"{item['cell_name']}.wait.chunk_elapsed_seconds")
        elapsed = _finite_number(event.get("elapsed_seconds"),
                                 f"{item['cell_name']}.wait.elapsed_seconds")
        _require(requested <= RECOVERY_WAIT_CHUNK_SECONDS + 1e-6,
                 f"{item['cell_name']} recovery wait chunk exceeded 30 seconds")
        _require(elapsed > prior_elapsed,
                 f"{item['cell_name']} recovery wait progress is nonmonotonic")
        elapsed_sum += actual
        prior_elapsed = elapsed
    completed_actual = _finite_number(events[-1].get("actual_seconds"),
                                      f"{item['cell_name']}.wait.actual_seconds")
    declared_actual = _finite_number(expected_actual,
                                     f"{item['cell_name']}.pre_cell_recovery_seconds")
    _require(completed_actual >= expected_seconds and
             declared_actual >= expected_seconds,
             f"{item['cell_name']} pre-cell recovery was shorter than {expected_seconds} seconds")
    _require(math.isclose(completed_actual, elapsed_sum, rel_tol=0.0, abs_tol=0.05),
             f"{item['cell_name']} recovery wait chunk total disagrees with completion")
    _require(math.isclose(declared_actual, completed_actual, rel_tol=0.0, abs_tol=0.05),
             f"{item['cell_name']} recovery wait intent disagrees with its log")
    return events


def _validate_passes(passes: Any, recovery_order: str,
                     cell_name: str) -> dict[str, dict[str, Any]]:
    _require(recovery_order in CELL_ORDERS, f"{cell_name} recovery order is invalid")
    _require(isinstance(passes, list) and len(passes) == 3,
             f"{cell_name} must contain exactly warmup, short, and long pass receipts")
    expected_labels = ["warmup", *recovery_order.split("-")]
    labels = [item.get("label") if isinstance(item, dict) else None for item in passes]
    _require(labels == expected_labels,
             f"{cell_name} pass order must be {expected_labels}")
    by_label: dict[str, dict[str, Any]] = {}
    prior_finished_ms = 0
    prior_monotonic_finished_ns = 0
    for output_index, item in enumerate(passes):
        label = expected_labels[output_index]
        expected_recovery = 0 if label == "warmup" else (1 if label == "short" else 120)
        requested = item.get("requested_recovery_seconds")
        _require(isinstance(requested, int) and not isinstance(requested, bool) and
                 requested == expected_recovery,
                 f"{cell_name}.{label}.requested_recovery_seconds mismatch")
        observed = _finite_number(item.get("observed_recovery_seconds"),
                                  f"{cell_name}.{label}.observed_recovery_seconds", positive=False)
        if label == "warmup":
            _require(observed == 0.0, f"{cell_name}.warmup observed recovery must be zero")
        else:
            _require(observed >= expected_recovery,
                     f"{cell_name}.{label} observed recovery is shorter than requested")
            maximum_recovery = 5.0 if label == "short" else 125.0
            _require(observed <= maximum_recovery,
                     f"{cell_name}.{label} observed recovery exceeded {maximum_recovery:g} seconds")
        for field in ("wall_seconds", "encode_seconds", "terminal_wait_seconds", "gpu_seconds"):
            _finite_number(item.get(field), f"{cell_name}.{label}.{field}")
        wall = float(item["wall_seconds"])
        encode = float(item["encode_seconds"])
        wait = float(item["terminal_wait_seconds"])
        _require(wall + 1e-6 >= encode + wait,
                 f"{cell_name}.{label} wall time is shorter than encode plus terminal wait")
        segments = _positive_int(item.get("segments"), f"{cell_name}.{label}.segments")
        _require(_positive_int(item.get("waits"), f"{cell_name}.{label}.waits") == 1,
                 f"{cell_name}.{label} must have exactly one terminal wait")
        for field in ("encode_allocations", "encode_readbacks", "preterminal_waits"):
            value = item.get(field)
            _require(isinstance(value, int) and not isinstance(value, bool) and value == 0,
                     f"{cell_name}.{label}.{field} must be zero")
        for field in ("thermal_before", "thermal_after"):
            value = item.get(field)
            _require(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3,
                     f"{cell_name}.{label}.{field} must be an integer in [0, 3]")
        index = item.get("output_index")
        _require(isinstance(index, int) and not isinstance(index, bool) and index == output_index,
                 f"{cell_name}.{label}.output_index must be {output_index}")
        started_ms = _positive_int(item.get("started_unix_ms"), f"{cell_name}.{label}.started_unix_ms")
        finished_ms = _positive_int(item.get("finished_unix_ms"), f"{cell_name}.{label}.finished_unix_ms")
        _require(finished_ms > started_ms and started_ms >= prior_finished_ms,
                 f"{cell_name}.{label} UTC intervals are not monotonic/nonoverlapping")
        prior_finished_ms = finished_ms
        monotonic_start_ns = _positive_int(item.get("started_monotonic_ns"),
                                           f"{cell_name}.{label}.started_monotonic_ns")
        monotonic_finish_ns = _positive_int(item.get("finished_monotonic_ns"),
                                            f"{cell_name}.{label}.finished_monotonic_ns")
        _require(monotonic_finish_ns > monotonic_start_ns and
                 monotonic_start_ns >= prior_monotonic_finished_ns,
                 f"{cell_name}.{label} monotonic intervals are not increasing/nonoverlapping")
        measured_wall = (monotonic_finish_ns - monotonic_start_ns) / 1_000_000_000
        _require(abs(wall - measured_wall) <= 0.010,
                 f"{cell_name}.{label} wall_seconds disagrees with its monotonic interval")
        if label != "warmup":
            recovery_interval = (monotonic_start_ns - prior_monotonic_finished_ns) / 1_000_000_000
            _require(abs(observed - recovery_interval) <= 1e-6,
                     f"{cell_name}.{label} observed recovery disagrees with the monotonic interval")
        prior_monotonic_finished_ns = monotonic_finish_ns
        by_label[label] = {**item, "segments": segments}
    return by_label


def _validate_outputs(outputs: Any, output_dir: Path, expected_labels: tuple[str, ...] | list[str],
                      cell_name: str, final_pass_finished_ms: int) -> None:
    _require(isinstance(outputs, list) and len(outputs) == 3,
             f"{cell_name} output records must contain exactly three snapshots")
    labels = [item.get("label") if isinstance(item, dict) else None for item in outputs]
    _require(labels == list(expected_labels),
             f"{cell_name} output labels/order must match the pass order")
    seen: set[Path] = set()
    for index, output in enumerate(outputs):
        label = expected_labels[index]
        _require(isinstance(output, dict), f"{cell_name}.{label} output record must be an object")
        output_index = output.get("output_index")
        _require(isinstance(output_index, int) and not isinstance(output_index, bool) and output_index == index,
                 f"{cell_name}.{label}.output_index must be {index}")
        _require(output.get("sha256") == OUTPUT_SHA256,
                 f"{cell_name}.{label} output digest differs from the pinned upstream oracle")
        _require(output.get("upstream_bitwise_equal") is True,
                 f"{cell_name}.{label} output is not declared bitwise equal to the pinned oracle")
        exported = _positive_int(output.get("exported_unix_ms"), f"{cell_name}.{label}.exported_unix_ms")
        _require(exported >= final_pass_finished_ms,
                 f"{cell_name}.{label} was exported before all passes completed")
        raw_path = output.get("path")
        _require(isinstance(raw_path, str) and Path(raw_path).is_absolute(),
                 f"{cell_name}.{label} output path must be absolute")
        path = Path(raw_path)
        _require(path.is_file() and not path.is_symlink(),
                 f"{cell_name}.{label} output file is missing or symlinked")
        canonical = path.resolve(strict=True)
        _require(canonical.parent == output_dir.resolve(strict=True),
                 f"{cell_name}.{label} output path escaped its cell directory")
        _require(canonical.stat().st_size == OUTPUT_BYTES,
                 f"{cell_name}.{label} output byte length is incorrect")
        _require(_sha256_file(canonical) == OUTPUT_SHA256,
                 f"{cell_name}.{label} output file digest mismatch")
        _require(canonical not in seen, f"{cell_name} output paths must be distinct")
        seen.add(canonical)


def _validate_receipt(receipt: dict[str, Any], recovery_order: str,
                      output_dir: Path, cell_name: str) -> dict[str, Any]:
    _require(receipt.get("schema") == SCHEMA, f"{cell_name} recovery receipt schema mismatch")
    _require(receipt.get("recovery_clock") == RECOVERY_CLOCK,
             f"{cell_name} recovery_clock must be {RECOVERY_CLOCK}")
    _require(receipt.get("recovery_order") == recovery_order,
             f"{cell_name} recovery order mismatch")
    for field, expected in PINNED.items():
        _require(receipt.get(field) == expected,
                 f"{cell_name} recovery receipt {field} mismatch")
    _require(receipt.get("oracle_display_sha256") == OUTPUT_SHA256,
             f"{cell_name} upstream display identity mismatch")
    _require(_is_sha256(receipt.get("decoder_cache_sha256")),
             f"{cell_name} decoder cache digest is invalid")
    _require(receipt.get("geometry") == "768x416" and receipt.get("frames") == 120,
             f"{cell_name} geometry mismatch")
    _require(receipt.get("raw_capture") is False and receipt.get("dit_constructed") is False,
             f"{cell_name} must disable raw capture and must not construct DiT")
    _require(receipt.get("atol") == 2e-5 and receipt.get("rtol") == 2e-5,
             f"{cell_name} tolerance changed from 2e-5/2e-5")
    budget = _positive_int(receipt.get("named_budget_bytes"), f"{cell_name}.named_budget_bytes")
    _require(budget == 24 * GIB, f"{cell_name} named budget must remain 24 GiB")
    decoder_named = _positive_int(receipt.get("decoder_named_bytes"),
                                  f"{cell_name}.decoder_named_bytes")
    _require(decoder_named <= budget,
             f"{cell_name} decoder named residency exceeds the 24 GiB budget")
    owner_ids = receipt.get("reused_owner_ids")
    _require(isinstance(owner_ids, list) and len(owner_ids) == 42 and
             all(isinstance(owner, int) and not isinstance(owner, bool) and owner > 0
                 for owner in owner_ids) and len(set(owner_ids)) == len(owner_ids),
             f"{cell_name} decoder owner identity set is incomplete or duplicated")
    timings = receipt.get("timings")
    _require(isinstance(timings, dict), f"{cell_name} timings must be an object")
    decoder_setup = _finite_number(timings.get("decoder_setup_seconds"),
                                   f"{cell_name}.decoder_setup_seconds")
    passes = _validate_passes(receipt.get("passes"), recovery_order, cell_name)
    ordered_labels = ("warmup", *recovery_order.split("-"))
    final_finished = max(passes[label]["finished_unix_ms"] for label in ordered_labels)
    _validate_outputs(receipt.get("outputs"), output_dir, ordered_labels, cell_name, final_finished)
    return {
        "recovery_order": recovery_order,
        "recovery_clock": receipt["recovery_clock"],
        "decoder_cache_sha256": receipt["decoder_cache_sha256"],
        "geometry": receipt["geometry"],
        "frames": receipt["frames"],
        "raw_capture": receipt["raw_capture"],
        "atol": receipt["atol"],
        "rtol": receipt["rtol"],
        "named_budget_bytes": budget,
        "decoder_named_bytes": decoder_named,
        "reused_owner_count": len(owner_ids),
        "decoder_setup_seconds": decoder_setup,
        "passes": passes,
        "outputs": receipt["outputs"],
    }


def _validate_cell(output_root: Path, item: dict[str, Any],
                   launch: dict[str, Any]) -> dict[str, Any]:
    cell_dir = output_root / item["cell_name"]
    intent = _read_json(cell_dir / "intent.json", f"{item['cell_name']} intent")
    completion = _read_json(cell_dir / "completion.json", f"{item['cell_name']} completion")
    _require(intent.get("schema") == "cogni_ml.pixelumm_recovery_matrix_cell_intent/v1",
             f"{item['cell_name']} intent schema mismatch")
    _require(completion.get("schema") == "cogni_ml.pixelumm_recovery_matrix_cell_completion/v1",
             f"{item['cell_name']} completion schema mismatch")
    for document, label in ((intent, "intent"), (completion, "completion")):
        _require(document.get("cell_index") == item["cell_index"] and
                 document.get("cell_name") == item["cell_name"] and
                 document.get("recovery_order") == item["recovery_order"] and
                 document.get("mode") == item["mode"],
                 f"{item['cell_name']} {label} identity mismatch")
    _require(completion.get("launch_error") is None and completion.get("returncode") == 0,
             f"{item['cell_name']} process failed")
    process_id = _positive_int(completion.get("process_id"), f"{item['cell_name']}.process_id")
    _require(intent.get("binary") == launch["binary"] and
             completion.get("binary_sha256") == launch["binary"]["sha256"] and
             intent.get("source_manifest_sha256") == launch["source"]["manifest_sha256"] and
             completion.get("source_manifest_sha256") == launch["source"]["manifest_sha256"],
             f"{item['cell_name']} source or binary identity mismatch")
    _require(completion.get("pre_cell_recovery_seconds") == intent.get("pre_cell_recovery_seconds"),
             f"{item['cell_name']} recovery wait intent/completion mismatch")
    wait_log = cell_dir / "recovery_wait.jsonl"
    _require(completion.get("recovery_wait_sha256") == _sha256_file(wait_log),
             f"{item['cell_name']} recovery wait log digest mismatch")
    _validate_wait_log(wait_log, item, RECOVERY_SECONDS, intent.get("pre_cell_recovery_seconds"))
    for name, expected in (("stdout.log", completion.get("stdout_sha256")),
                           ("stderr.log", completion.get("stderr_sha256"))):
        path = cell_dir / name
        _require(path.is_file() and _is_sha256(expected) and _sha256_file(path) == expected,
                 f"{item['cell_name']} {name} is missing or changed")
    receipt_path = cell_dir / "output" / "receipt.json"
    _require(receipt_path.is_file() and completion.get("receipt_sha256") == _sha256_file(receipt_path),
             f"{item['cell_name']} receipt is missing or changed")
    MATRIX._load_guard(cell_dir / "guard.jsonl", launch, item, completion)
    receipt = _read_json(receipt_path, f"{item['cell_name']} recovery receipt")
    result = _validate_receipt(receipt, item["recovery_order"], cell_dir / "output", item["cell_name"])
    result.update({
        "cell_index": item["cell_index"], "cell_name": item["cell_name"],
        "mode": item["mode"], "process_id": process_id,
        "pre_cell_recovery_seconds": intent["pre_cell_recovery_seconds"],
        "thermal_before_cell": intent.get("thermal_before_cell"),
        "thermal_after_cell": completion.get("thermal_after_cell"),
    })
    return result


def _validate_shared_identity(cells: list[dict[str, Any]]) -> None:
    fields = (
        "input_sha256", "latent_sha256", "noise_sha256", "oracle_report_sha256",
        "oracle_display_sha256", "decoder_cache_sha256", "geometry", "frames",
        "raw_capture", "atol", "rtol", "named_budget_bytes", "recovery_clock",
    )
    for field in fields:
        values = [cell["receipt"][field] for cell in cells]
        _require(all(value == values[0] for value in values),
                 f"matrix cells disagree on shared receipt field {field}")


def _compare_cell(pass_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    short = pass_map["short"]
    long = pass_map["long"]
    return {
        "short_over_long": {
            "wall_seconds": short["wall_seconds"] / long["wall_seconds"],
            "gpu_seconds": short["gpu_seconds"] / long["gpu_seconds"],
        },
        "short": {"wall_seconds": short["wall_seconds"], "gpu_seconds": short["gpu_seconds"]},
        "long": {"wall_seconds": long["wall_seconds"], "gpu_seconds": long["gpu_seconds"]},
        "thermal_levels": {
            "warmup": [pass_map["warmup"]["thermal_before"], pass_map["warmup"]["thermal_after"]],
            "short": [short["thermal_before"], short["thermal_after"]],
            "long": [long["thermal_before"], long["thermal_after"]],
        },
    }


def analyze_matrix(output_root: Path | str) -> dict[str, Any]:
    root = Path(output_root).resolve(strict=True)
    launch = _read_json(root / "launch.json", "recovery matrix launch manifest")
    _require(launch.get("schema") == "cogni_ml.pixelumm_recovery_matrix_launch/v1",
             "recovery matrix launch schema mismatch")
    _require(Path(launch.get("output_root", "")).resolve(strict=True) == root,
             "recovery matrix launch root identity mismatch")
    _require(launch.get("pre_cell_recovery_seconds") == RECOVERY_SECONDS and
             launch.get("recovery_wait_chunk_max_seconds") == RECOVERY_WAIT_CHUNK_SECONDS,
             "recovery matrix pre-cell recovery policy mismatch")
    _require(launch.get("guard_limits") == GUARD_LIMITS,
             "recovery matrix launch guard limits mismatch")
    _require(launch.get("schedule") == _cell_schedule(),
             "recovery matrix must use fixed short-long then long-short order")
    source = launch.get("source")
    binary = launch.get("binary")
    _validate_source_manifest(source)
    _require(isinstance(binary, dict) and _is_sha256(binary.get("sha256")) and
             isinstance(binary.get("path"), str) and Path(binary["path"]).is_absolute(),
             "recovery matrix launch binary identity is incomplete")
    cells = []
    for item in launch["schedule"]:
        cell = _validate_cell(root, item, launch)
        receipt = _read_json(root / item["cell_name"] / "output" / "receipt.json", "recovery receipt")
        cell["receipt"] = receipt
        cells.append(cell)
    _require(len({cell["process_id"] for cell in cells}) == 2,
             "recovery matrix cells must come from two fresh processes")
    _validate_shared_identity(cells)
    segments = {entry["segments"] for cell in cells for entry in cell["passes"].values()}
    _require(len(segments) == 1,
             "recovery matrix segment count differs across passes or cells")
    for cell in cells:
        cell["comparison"] = _compare_cell(cell["passes"])
        del cell["receipt"]
    return {
        "schema": "cogni_ml.pixelumm_recovery_matrix_analysis/v1",
        "passed": True,
        "claim_scope": "descriptive within-cell short versus long recovery timing on one saved latent/noise identity",
        "claim_limit": "not a full-generation or DiT result, not a causal speedup claim; thermal levels are categorical context, not GPU frequency or power",
        "recovery_order": [item["recovery_order"] for item in launch["schedule"]],
        "shared_identity": {
            field: cells[0][field]
            for field in ("geometry", "frames", "raw_capture", "atol", "rtol", "named_budget_bytes",
                          "decoder_cache_sha256", "recovery_clock")
        },
        "segments_per_pass": next(iter(segments)),
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", help="absolute executable built from pixelumm_recovery_probe")
    parser.add_argument("--output-root", required=True,
                        help="fresh subtree below /opt/mldata_ssd for this experiment")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", "--dry-run", dest="plan_only", action="store_true",
                      help="print the exact guarded plan without creating files or launching a process")
    mode.add_argument("--analyze-only", action="store_true",
                      help="validate and summarize an existing recovery matrix")
    args = parser.parse_args(argv)
    if args.analyze_only:
        if args.binary:
            parser.error("--binary is not used with --analyze-only")
        report = analyze_matrix(args.output_root)
    else:
        if not args.binary:
            parser.error("--binary is required unless --analyze-only is selected")
        if args.plan_only:
            report = dry_run_plan(args.binary, args.output_root)
        else:
            report = launch_matrix(args.binary, args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, subprocess.SubprocessError) as error:
        print(f"pixelumm_recovery_matrix: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
