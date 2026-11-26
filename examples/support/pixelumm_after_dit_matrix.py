#!/usr/bin/env python3
"""Run and analyze a guarded four-cell PixelUMM saved-latent lifecycle matrix.

The cells are fresh processes in fixed ABBA order (standalone, after-DiT,
after-DiT, standalone). This reports an operational prelude contrast only; it
does not isolate DiT compute from every process/runtime cache effect.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
GUARD = ROOT / "examples" / "support" / "wan_memory_guard.py"
MIB = 1024 * 1024
GIB = 1024 * MIB
OUTPUT_BYTES = 1 * 3 * 120 * 416 * 768 * 4
ORDER = ("standalone", "after-dit", "after-dit", "standalone")
OUTPUT_SHA256 = "5ed0974dfd3b28d940c370fa8faef6a23b288d86adb03c6810758ae16c55cc9f"
PINNED = {
    "input_sha256": "dcbe51749742701a2ee1a6d09a10e2f20e663f3ff54f646d801c743e17c7a104",
    "latent_sha256": "ea79dc6896f3e956035acd0caa69b1730554d5459d80893989023493de4b5bba",
    "noise_sha256": "166ea19d2812c0b4dedfedefd4f101b2abe8b4ec48364fb63705f781e1828547",
    "oracle_report_sha256": "a91ad030b8cd682f4ca6c00ebfad41c4a3ea56a9bcff41155318f6fb12d2ea7e",
}
GUARD_LIMITS = {
    "max_footprint_mib": 26_624,
    "max_swap_growth_mib": 64,
    "timeout_sec": 1_200,
    "memory_accounting": "file-cache",
    "runtime_floor_pct": 10,
    "runtime_floor_pct_exact": "10",
}
CELL_SOURCE_GLOBS = (
    "src/ml/**/*.cr",
    "src/ml/**/*.mm",
    "src/ml/**/*.metal",
    "examples/support/hy_nomas*.cr",
    "examples/support/hy_nomas*.metal",
    "examples/support/pixelumm*.cr",
    "examples/support/pixelumm*.metal",
    "examples/support/qwen25*.cr",
)
SOURCE_FILES = (
    "Makefile",
    "examples/pixelumm_after_dit_probe.cr",
    "examples/support/pixelumm_after_dit_matrix.py",
    "examples/support/wan_memory_guard.py",
)


class ValidationError(ValueError):
    """Raised when a matrix artifact is absent, inconsistent, or unsafe."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{description} is missing or invalid JSON: {path}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive_int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, field: str, *, positive: bool = True) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{field} must be numeric")
    number = float(value)
    _require(math.isfinite(number), f"{field} must be finite")
    if positive:
        _require(number > 0.0, f"{field} must be positive")
    else:
        _require(number >= 0.0, f"{field} must be nonnegative")
    return number


def _source_identity() -> dict[str, Any]:
    paths = {ROOT / relative for relative in SOURCE_FILES}
    for pattern in CELL_SOURCE_GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    files: dict[str, str] = {}
    for path in sorted(paths):
        _require(path.is_file() and not path.is_symlink(), f"source file is missing or symlinked: {path}")
        files[str(path.relative_to(ROOT))] = sha256_file(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT, check=True,
                            capture_output=True, text=True, timeout=10).stdout
    return {
        "git_head": head,
        "working_tree_status_sha256": sha256_bytes(status.encode("utf-8")),
        "files": files,
        "manifest_sha256": sha256_bytes(encoded),
        "matrix_script_sha256": files["examples/support/pixelumm_after_dit_matrix.py"],
        "probe_source_sha256": files["examples/pixelumm_after_dit_probe.cr"],
        "guard_source_sha256": files["examples/support/wan_memory_guard.py"],
    }


def _binary_identity(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path)
    _require(path.is_absolute(), "--binary must be an absolute path")
    _require(path.is_file() and not path.is_symlink(), "--binary must be an existing regular non-symlink file")
    canonical = path.resolve(strict=True)
    stat = canonical.stat()
    _require(os.access(canonical, os.X_OK), "--binary must be executable")
    return {
        "path": str(canonical),
        "sha256": sha256_file(canonical),
        "size_bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mode": stat.st_mode & 0o777,
    }


def _output_root_candidate(raw_path: str) -> Path:
    path = Path(raw_path)
    base = Path("/opt/mldata_ssd")
    _require(path.is_absolute(), "--output-root must be an absolute path")
    _require(".." not in path.parts, "--output-root may not contain parent traversal")
    _require(base.is_dir(), "/opt/mldata_ssd must exist and be mounted")
    base_real = base.resolve(strict=True)
    _require(path != base, "--output-root must be a new subtree below /opt/mldata_ssd")
    parent = path.parent.resolve(strict=True)
    _require(parent.is_dir() and parent.is_relative_to(base_real),
             "--output-root parent must already exist below /opt/mldata_ssd")
    current = base
    for part in path.relative_to(base).parts[:-1]:
        current = current / part
        _require(not current.is_symlink(), "--output-root may not traverse symlinked directories")
    canonical = parent / path.name
    _require(not canonical.exists() and not canonical.is_symlink(),
             "--output-root must be fresh and must not already exist")
    return canonical


def _thermal_state() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    sysctl = Path("/usr/sbin/sysctl")
    for key in ("machdep.xcpm.cpu_thermal_level", "machdep.xcpm.gpu_thermal_level"):
        value: str | None = None
        if sysctl.is_file():
            try:
                probe = subprocess.run([str(sysctl), "-n", key], capture_output=True,
                                       text=True, check=False, timeout=1)
                if probe.returncode == 0:
                    value = probe.stdout.strip() or None
            except (OSError, subprocess.SubprocessError):
                pass
        result[key] = value
    return result


def _command(binary: dict[str, Any], mode: str, output_dir: Path, guard_log: Path) -> list[str]:
    return [
        sys.executable, "-B", str(GUARD),
        "--memory-accounting=file-cache",
        f"--max-footprint-mib={GUARD_LIMITS['max_footprint_mib']}",
        f"--max-swap-growth-mib={GUARD_LIMITS['max_swap_growth_mib']}",
        f"--timeout-sec={GUARD_LIMITS['timeout_sec']}",
        "--runtime-floor-pct=10",
        "--log=" + str(guard_log), "--",
        binary["path"], "--mode", mode, "--output-dir", str(output_dir),
    ]


def _make_launch_manifest(binary: dict[str, Any], output_root: Path,
                          pause_seconds: int) -> dict[str, Any]:
    source = _source_identity()
    return {
        "schema": "cogni_ml.pixelumm_after_dit_matrix_launch/v1",
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
        "schedule": [
            {"cell_index": index, "cell_name": f"cell-{index + 1:02d}-{mode}", "mode": mode}
            for index, mode in enumerate(ORDER)
        ],
        "pause_seconds_between_cells": pause_seconds,
        "guard_limits": GUARD_LIMITS,
        "scope": "saved-latent PixelUMM lifecycle timing; no full video generation",
        "claim_limit": "descriptive operational prelude contrast only; no isolated DiT-compute causality or p-values",
    }


def dry_run_plan(binary_path: str, output_root: str, pause_seconds: int) -> dict[str, Any]:
    _require(isinstance(pause_seconds, int) and not isinstance(pause_seconds, bool) and 0 <= pause_seconds <= 60,
             "pause seconds must be an integer in [0, 60]")
    binary = _binary_identity(binary_path)
    root = _output_root_candidate(output_root)
    manifest = _make_launch_manifest(binary, root, pause_seconds)
    cells = []
    for item in manifest["schedule"]:
        cell = root / item["cell_name"]
        output = cell / "output"
        guard_log = cell / "guard.jsonl"
        cells.append({
            **item,
            "output_dir": str(output),
            "guard_command": _command(binary, item["mode"], output, guard_log),
        })
    return {"dry_run": True, "launch": manifest, "cells": cells}


def _cell_dir(output_root: Path, index: int, mode: str) -> Path:
    return output_root / f"cell-{index + 1:02d}-{mode}"


def _write_log_identity(guard_log: Path, cell: dict[str, Any],
                        launch: dict[str, Any]) -> None:
    _require(guard_log.is_file(), f"guard did not create its log: {guard_log}")
    _append_jsonl(guard_log, {
        "event": "matrix_cell_identity",
        "utc": _utc_now(),
        "cell_index": cell["cell_index"],
        "cell_name": cell["cell_name"],
        "mode": cell["mode"],
        "binary_path": launch["binary"]["path"],
        "binary_sha256": launch["binary"]["sha256"],
        "source_manifest_sha256": launch["source"]["manifest_sha256"],
        "matrix_script_sha256": launch["source"]["matrix_script_sha256"],
        "probe_source_sha256": launch["source"]["probe_source_sha256"],
    })


def _write_exclusive_log(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def _check_live_identities(launch: dict[str, Any]) -> None:
    _require(_binary_identity(launch["binary"]["path"]) == launch["binary"],
             "binary identity changed during the matrix")
    _require(_source_identity() == launch["source"],
             "source identity changed during the matrix")


def _run_cell(output_root: Path, item: dict[str, Any], launch: dict[str, Any]) -> dict[str, Any]:
    _check_live_identities(launch)
    cell_dir = output_root / item["cell_name"]
    cell_dir.mkdir(mode=0o700, exist_ok=False)
    output_dir = cell_dir / "output"
    output_dir.mkdir(mode=0o700, exist_ok=False)
    guard_log = cell_dir / "guard.jsonl"
    stdout_path = cell_dir / "stdout.log"
    stderr_path = cell_dir / "stderr.log"
    command = _command(launch["binary"], item["mode"], output_dir, guard_log)
    intent = {
        "schema": "cogni_ml.pixelumm_after_dit_matrix_cell_intent/v1",
        **item,
        "command": command,
        "binary": launch["binary"],
        "source_manifest_sha256": launch["source"]["manifest_sha256"],
        "guard_limits": GUARD_LIMITS,
        "output_dir": str(output_dir),
        "thermal_before": _thermal_state(),
        "created_utc": _utc_now(),
    }
    _write_new_json(cell_dir / "intent.json", intent)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.monotonic()
    returncode: int | None = None
    launch_error: str | None = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            os.fchmod(stdout.fileno(), 0o600)
            os.fchmod(stderr.fileno(), 0o600)
            returncode = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                        stderr=stderr, check=False).returncode
    except (OSError, subprocess.SubprocessError) as error:
        launch_error = f"{type(error).__name__}: {error}"
    elapsed_seconds = time.monotonic() - start
    if guard_log.is_file():
        _write_log_identity(guard_log, item, launch)
    completion = {
        "schema": "cogni_ml.pixelumm_after_dit_matrix_cell_completion/v1",
        "cell_index": item["cell_index"],
        "cell_name": item["cell_name"],
        "mode": item["mode"],
        "returncode": returncode,
        "launch_error": launch_error,
        "elapsed_seconds": elapsed_seconds,
        "finished_utc": _utc_now(),
        "binary_sha256": launch["binary"]["sha256"],
        "source_manifest_sha256": launch["source"]["manifest_sha256"],
        "thermal_after": _thermal_state(),
        "guard_sha256": sha256_file(guard_log) if guard_log.is_file() else None,
        "stdout_sha256": sha256_file(stdout_path) if stdout_path.is_file() else None,
        "stderr_sha256": sha256_file(stderr_path) if stderr_path.is_file() else None,
        "receipt_sha256": sha256_file(output_dir / "receipt.json")
        if (output_dir / "receipt.json").is_file() else None,
    }
    _write_new_json(cell_dir / "completion.json", completion)
    _check_live_identities(launch)
    _require(returncode == 0 and launch_error is None,
             f"{item['cell_name']} failed under the memory guard (returncode={returncode}, error={launch_error})")
    return completion


def launch_matrix(binary_path: str, output_root_path: str, pause_seconds: int) -> dict[str, Any]:
    plan = dry_run_plan(binary_path, output_root_path, pause_seconds)
    output_root = Path(plan["launch"]["output_root"])
    output_root.mkdir(mode=0o700, exist_ok=False)
    launch = plan["launch"]
    _write_new_json(output_root / "launch.json", launch)
    completions = []
    try:
        for index, item in enumerate(launch["schedule"]):
            completion = _run_cell(output_root, item, launch)
            completions.append(completion)
            if index + 1 < len(launch["schedule"]):
                time.sleep(pause_seconds)
        report = analyze_matrix(output_root)
        _write_new_json(output_root / "analysis.json", report)
        _write_new_json(output_root / "completion.json", {
            "schema": "cogni_ml.pixelumm_after_dit_matrix_completion/v1",
            "passed": True,
            "cell_completions": completions,
            "analysis_sha256": sha256_file(output_root / "analysis.json"),
            "finished_utc": _utc_now(),
        })
        return report
    except Exception as error:
        _write_new_json(output_root / "failure.json", {
            "schema": "cogni_ml.pixelumm_after_dit_matrix_failure/v1",
            "passed": False,
            "completed_cells": len(completions),
            "error": f"{type(error).__name__}: {error}",
            "finished_utc": _utc_now(),
        })
        raise


def _load_guard(path: Path, launch: dict[str, Any], item: dict[str, Any],
                completion: dict[str, Any]) -> list[dict[str, Any]]:
    _require(path.is_file(), f"{item['cell_name']} guard log is missing")
    _require(completion.get("guard_sha256") == sha256_file(path),
             f"{item['cell_name']} guard log digest mismatch")
    events: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            record = json.loads(line)
            _require(isinstance(record, dict), f"guard event {line_number} must be an object")
            events.append(record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{item['cell_name']} guard log is invalid JSONL") from error
    _require(events, f"{item['cell_name']} guard log is empty")
    limits = [event for event in events if event.get("event") == "limits"]
    _require(len(limits) == 1, f"{item['cell_name']} must have exactly one guard limits record")
    actual = limits[0]
    expected = {
        "runtime_floor_pct": 10,
        "runtime_floor_pct_exact": "10",
        "max_footprint_bytes": GUARD_LIMITS["max_footprint_mib"] * MIB,
        "max_swap_growth_bytes": GUARD_LIMITS["max_swap_growth_mib"] * MIB,
        "timeout_sec": GUARD_LIMITS["timeout_sec"],
        "memory_accounting": GUARD_LIMITS["memory_accounting"],
    }
    for field, value in expected.items():
        _require(actual.get(field) == value, f"{item['cell_name']} guard limit mismatch: {field}")
    failed_events = {"blocked", "launch_failed", "stopping", "cleanup_failed", "cleanup_incomplete"}
    _require(not any(event.get("event") in failed_events for event in events),
             f"{item['cell_name']} guard recorded a block, stop, or cleanup failure")
    starts = [index for index, event in enumerate(events) if event.get("event") == "started"]
    finishes = [index for index, event in enumerate(events) if event.get("event") == "finished"]
    sample_indexes = [index for index, event in enumerate(events) if event.get("event") == "sample"]
    _require(len(starts) == 1 and len(finishes) == 1 and len(sample_indexes) >= 2,
             f"{item['cell_name']} guard log lacks complete start, samples, or finish")
    _require(sample_indexes[0] < starts[0] < sample_indexes[-1] < finishes[0],
             f"{item['cell_name']} guard event order is inconsistent")
    _require(finishes[0] < len(events) - 1 and events[-1].get("event") == "matrix_cell_identity",
             f"{item['cell_name']} identity must be appended after the completed guard log")
    identity_events = [event for event in events if event.get("event") == "matrix_cell_identity"]
    _require(len(identity_events) == 1, f"{item['cell_name']} must record one matrix identity in its guard log")
    identity = identity_events[0]
    for field, value in {
        "cell_index": item["cell_index"],
        "cell_name": item["cell_name"],
        "mode": item["mode"],
        "binary_path": launch["binary"]["path"],
        "binary_sha256": launch["binary"]["sha256"],
        "source_manifest_sha256": launch["source"]["manifest_sha256"],
        "matrix_script_sha256": launch["source"]["matrix_script_sha256"],
        "probe_source_sha256": launch["source"]["probe_source_sha256"],
    }.items():
        _require(identity.get(field) == value,
                 f"{item['cell_name']} guard identity mismatch: {field}")

    samples = [events[index] for index in sample_indexes]
    initial_swap = samples[0].get("swap_used_bytes")
    _require(isinstance(initial_swap, int) and not isinstance(initial_swap, bool) and initial_swap >= 0,
             f"{item['cell_name']} initial swap sample is invalid")
    for sample_index, sample in enumerate(samples):
        pressure = sample.get("pressure_free_pct")
        physical = sample.get("physical_bytes")
        headroom = sample.get("headroom_bytes")
        footprint = sample.get("group_footprint_bytes")
        swap = sample.get("swap_used_bytes")
        for field, value in (("pressure_free_pct", pressure), ("physical_bytes", physical),
                             ("headroom_bytes", headroom), ("group_footprint_bytes", footprint),
                             ("swap_used_bytes", swap)):
            _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                     f"{item['cell_name']} sample {sample_index} has invalid {field}")
        _require(pressure >= GUARD_LIMITS["runtime_floor_pct"],
                 f"{item['cell_name']} guard pressure fell below 10%")
        floor_bytes = (physical + 9) // 10
        required_headroom = floor_bytes
        if sample_index == 0:
            required_headroom += GUARD_LIMITS["max_footprint_mib"] * MIB
        _require(headroom >= required_headroom,
                 f"{item['cell_name']} guard file-cache headroom breached the reserve/cap")
        _require(footprint <= GUARD_LIMITS["max_footprint_mib"] * MIB,
                 f"{item['cell_name']} process footprint exceeded 26624 MiB")
        _require(swap - initial_swap <= GUARD_LIMITS["max_swap_growth_mib"] * MIB,
                 f"{item['cell_name']} swap growth exceeded 64 MiB")
    finished = events[finishes[0]]
    _require(finished.get("returncode") == 0 and completion.get("returncode") == 0,
             f"{item['cell_name']} did not finish successfully")
    return events


def _validate_passes(receipt: dict[str, Any], cell_name: str) -> dict[str, dict[str, Any]]:
    passes = receipt.get("passes")
    _require(isinstance(passes, list) and len(passes) == 2,
             f"{cell_name} must contain exactly cold and warm pass receipts")
    _require([item.get("label") if isinstance(item, dict) else None for item in passes] == ["cold", "warm"],
             f"{cell_name} pass order must be cold then warm")
    by_label: dict[str, dict[str, Any]] = {}
    for item in passes:
        label = item["label"]
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
        by_label[label] = {**item, "segments": segments}
    return by_label


def _validate_outputs(receipt: dict[str, Any], output_dir: Path, cell_name: str) -> None:
    outputs = receipt.get("outputs")
    _require(isinstance(outputs, dict) and set(outputs) == {"cold", "warm"},
             f"{cell_name} output records must be exactly cold and warm")
    seen: set[Path] = set()
    for label in ("cold", "warm"):
        output = outputs[label]
        _require(isinstance(output, dict), f"{cell_name}.{label} output record must be an object")
        _require(output.get("sha256") == OUTPUT_SHA256,
                 f"{cell_name}.{label} output digest differs from the pinned upstream oracle")
        _require(output.get("upstream_bitwise_equal") is True,
                 f"{cell_name}.{label} output is not declared bitwise equal to the pinned oracle")
        raw_path = output.get("path")
        _require(isinstance(raw_path, str) and Path(raw_path).is_absolute(),
                 f"{cell_name}.{label} output path must be absolute")
        path = Path(raw_path)
        _require(path.is_file() and not path.is_symlink(), f"{cell_name}.{label} output file is missing or symlinked")
        canonical = path.resolve(strict=True)
        _require(canonical.parent == output_dir.resolve(strict=True),
                 f"{cell_name}.{label} output path escaped its cell directory")
        _require(canonical.stat().st_size == OUTPUT_BYTES,
                 f"{cell_name}.{label} output byte length is incorrect")
        _require(sha256_file(canonical) == OUTPUT_SHA256,
                 f"{cell_name}.{label} output file digest mismatch")
        _require(canonical not in seen, f"{cell_name} cold and warm outputs must be separate files")
        seen.add(canonical)


def _validate_cell(output_root: Path, item: dict[str, Any], launch: dict[str, Any]) -> dict[str, Any]:
    cell_dir = output_root / item["cell_name"]
    intent = _read_json(cell_dir / "intent.json", f"{item['cell_name']} intent")
    completion = _read_json(cell_dir / "completion.json", f"{item['cell_name']} completion")
    _require(intent.get("schema") == "cogni_ml.pixelumm_after_dit_matrix_cell_intent/v1",
             f"{item['cell_name']} intent schema mismatch")
    _require(completion.get("schema") == "cogni_ml.pixelumm_after_dit_matrix_cell_completion/v1",
             f"{item['cell_name']} completion schema mismatch")
    for document, label in ((intent, "intent"), (completion, "completion")):
        _require(document.get("cell_index") == item["cell_index"] and
                 document.get("cell_name") == item["cell_name"] and
                 document.get("mode") == item["mode"],
                 f"{item['cell_name']} {label} identity mismatch")
    _require(completion.get("launch_error") is None and completion.get("returncode") == 0,
             f"{item['cell_name']} process failed")
    _require(intent.get("binary") == launch["binary"] and
             completion.get("binary_sha256") == launch["binary"]["sha256"] and
             intent.get("source_manifest_sha256") == launch["source"]["manifest_sha256"] and
             completion.get("source_manifest_sha256") == launch["source"]["manifest_sha256"],
             f"{item['cell_name']} source or binary identity mismatch")
    for name, expected in (("stdout.log", completion.get("stdout_sha256")),
                           ("stderr.log", completion.get("stderr_sha256"))):
        path = cell_dir / name
        _require(path.is_file() and _is_sha256(expected) and sha256_file(path) == expected,
                 f"{item['cell_name']} {name} is missing or changed")
    receipt_path = cell_dir / "output" / "receipt.json"
    _require(receipt_path.is_file() and completion.get("receipt_sha256") == sha256_file(receipt_path),
             f"{item['cell_name']} receipt is missing or changed")
    _load_guard(cell_dir / "guard.jsonl", launch, item, completion)
    receipt = _read_json(receipt_path, f"{item['cell_name']} probe receipt")
    _require(receipt.get("schema") == "cogni_ml.pixelumm_after_dit_probe/v1",
             f"{item['cell_name']} probe receipt schema mismatch")
    _require(receipt.get("mode") == item["mode"], f"{item['cell_name']} probe mode mismatch")
    for field, expected in PINNED.items():
        _require(receipt.get(field) == expected,
                 f"{item['cell_name']} probe receipt {field} mismatch")
    _require(receipt.get("oracle_display_sha256") == OUTPUT_SHA256,
             f"{item['cell_name']} oracle display identity mismatch")
    _require(receipt.get("geometry") == "768x416" and receipt.get("frames") == 120,
             f"{item['cell_name']} geometry mismatch")
    _require(receipt.get("raw_capture") is False,
             f"{item['cell_name']} raw decoder capture must be disabled")
    _require(receipt.get("atol") == 2e-5 and receipt.get("rtol") == 2e-5,
             f"{item['cell_name']} tolerance changed from 2e-5/2e-5")
    budget = _positive_int(receipt.get("named_budget_bytes"), f"{item['cell_name']}.named_budget_bytes")
    _require(budget == 24 * GIB, f"{item['cell_name']} named budget must remain 24 GiB")
    dit_named = _positive_int(receipt.get("dit_named_bytes"), f"{item['cell_name']}.dit_named_bytes")
    decoder_named = _positive_int(receipt.get("decoder_named_bytes"), f"{item['cell_name']}.decoder_named_bytes")
    _require(max(dit_named, decoder_named) <= budget,
             f"{item['cell_name']} named phase residency exceeds the 24 GiB budget")
    owner_ids = receipt.get("reused_owner_ids")
    _require(isinstance(owner_ids, list) and len(owner_ids) == 42 and
             all(isinstance(owner, int) and not isinstance(owner, bool) and owner > 0 for owner in owner_ids) and
             len(set(owner_ids)) == len(owner_ids),
             f"{item['cell_name']} decoder owner identity set is incomplete or duplicated")
    timing = _validate_passes(receipt, item["cell_name"])
    _validate_outputs(receipt, cell_dir / "output", item["cell_name"])
    dit_timing = receipt.get("timings")
    _require(isinstance(dit_timing, dict), f"{item['cell_name']} timings must be an object")
    if item["mode"] == "after-dit":
        _finite_number(dit_timing.get("dit_wall_seconds"), f"{item['cell_name']}.dit_wall_seconds")
        _finite_number(dit_timing.get("dit_gpu_seconds"), f"{item['cell_name']}.dit_gpu_seconds")
    else:
        _require("dit_wall_seconds" not in dit_timing and "dit_gpu_seconds" not in dit_timing,
                 f"{item['cell_name']} standalone cell unexpectedly ran a DiT prelude")
    return {
        "cell_index": item["cell_index"],
        "cell_name": item["cell_name"],
        "mode": item["mode"],
        "dit_named_bytes": dit_named,
        "decoder_named_bytes": decoder_named,
        "named_budget_bytes": budget,
        "reused_owner_count": len(owner_ids),
        "passes": timing,
    }


def _ratio_samples(cells: list[dict[str, Any]], label: str, metric: str) -> dict[str, Any]:
    standalone = [cell["passes"][label][metric] for cell in cells if cell["mode"] == "standalone"]
    after_dit = [cell["passes"][label][metric] for cell in cells if cell["mode"] == "after-dit"]
    _require(len(standalone) == 2 and len(after_dit) == 2,
             f"{label} {metric} comparison requires two cells per mode")
    standalone_mean = math.fsum(standalone) / len(standalone)
    after_mean = math.fsum(after_dit) / len(after_dit)
    _require(standalone_mean > 0.0 and after_mean > 0.0,
             f"{label} {metric} comparison contains a nonpositive mean")
    pairs = [
        {"after_dit_cell": 2, "standalone_cell": 1,
         "after_dit_over_standalone": cells[1]["passes"][label][metric] /
         cells[0]["passes"][label][metric]},
        {"after_dit_cell": 3, "standalone_cell": 4,
         "after_dit_over_standalone": cells[2]["passes"][label][metric] /
         cells[3]["passes"][label][metric]},
    ]
    return {
        "standalone_samples": standalone,
        "after_dit_samples": after_dit,
        "standalone_mean": standalone_mean,
        "after_dit_mean": after_mean,
        "after_dit_over_standalone": after_mean / standalone_mean,
        "adjacent_pair_ratios": pairs,
    }


def analyze_matrix(output_root: Path | str) -> dict[str, Any]:
    root = Path(output_root).resolve(strict=True)
    launch = _read_json(root / "launch.json", "matrix launch manifest")
    _require(launch.get("schema") == "cogni_ml.pixelumm_after_dit_matrix_launch/v1",
             "matrix launch schema mismatch")
    _require(Path(launch.get("output_root", "")).resolve(strict=True) == root,
             "matrix launch root identity mismatch")
    _require(launch.get("pause_seconds_between_cells") in range(0, 61),
             "matrix inter-cell pause is outside [0, 60]")
    _require(launch.get("guard_limits") == GUARD_LIMITS, "matrix launch guard limits mismatch")
    _require(launch.get("schedule") == [
        {"cell_index": index, "cell_name": f"cell-{index + 1:02d}-{mode}", "mode": mode}
        for index, mode in enumerate(ORDER)
    ], "matrix must use the fixed standalone/after-DiT ABBA schedule")
    source = launch.get("source")
    binary = launch.get("binary")
    _require(isinstance(source, dict) and isinstance(source.get("files"), dict),
             "launch source identity is incomplete")
    _require(isinstance(binary, dict) and _is_sha256(binary.get("sha256")) and
             Path(binary.get("path", "")).is_absolute(), "launch binary identity is incomplete")
    encoded_sources = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    _require(sha256_bytes(encoded_sources) == source.get("manifest_sha256"),
             "launch source manifest digest mismatch")
    _require(source["files"].get("examples/support/pixelumm_after_dit_matrix.py") == source.get("matrix_script_sha256") and
             source["files"].get("examples/pixelumm_after_dit_probe.cr") == source.get("probe_source_sha256") and
             source["files"].get("examples/support/wan_memory_guard.py") == source.get("guard_source_sha256"),
             "launch source file digest entries are inconsistent")
    cell_items = launch["schedule"]
    cells = [_validate_cell(root, item, launch) for item in cell_items]
    for field in ("input_sha256", "latent_sha256", "noise_sha256", "geometry", "frames",
                  "raw_capture", "atol", "rtol", "oracle_display_sha256", "decoder_cache_sha256",
                  "cinematic_lineage_sha256", "dit_named_bytes", "decoder_named_bytes", "named_budget_bytes"):
        receipt_values = []
        for item in cell_items:
            receipt = _read_json(root / item["cell_name"] / "output" / "receipt.json", "probe receipt")
            receipt_values.append(receipt.get(field))
        _require(all(value == receipt_values[0] for value in receipt_values),
                 f"matrix cells disagree on shared receipt field {field}")
    comparisons = {
        label: {
            metric: _ratio_samples(cells, label, metric)
            for metric in ("wall_seconds", "gpu_seconds")
        }
        for label in ("cold", "warm")
    }
    return {
        "schema": "cogni_ml.pixelumm_after_dit_matrix_analysis/v1",
        "passed": True,
        "claim_scope": "descriptive operational standalone versus after-DiT-prelude contrast on one saved latent/noise identity",
        "claim_limit": "does not isolate DiT compute from process/runtime cache and ordering effects; no p-values or full-generation claim",
        "order": [item["mode"] for item in cell_items],
        "shared_identity": {
            "input_sha256": PINNED["input_sha256"],
            "latent_sha256": PINNED["latent_sha256"],
            "noise_sha256": PINNED["noise_sha256"],
            "geometry": "768x416",
            "frames": 120,
            "raw_capture": False,
            "output_sha256": OUTPUT_SHA256,
        },
        "cells": cells,
        "ratios": comparisons,
    }


def _pause_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pause seconds must be an integer") from error
    if not 0 <= seconds <= 60:
        raise argparse.ArgumentTypeError("pause seconds must be in [0, 60]")
    return seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", help="absolute executable built from the PixelUMM after-DiT probe")
    parser.add_argument("--output-root", required=True,
                        help="fresh new subtree below /opt/mldata_ssd for this run")
    parser.add_argument("--pause-seconds", type=_pause_seconds, default=30,
                        help="bounded pause between fresh-process cells (default: 30; max: 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact ABBA plan without creating files or launching a process")
    parser.add_argument("--analyze-only", action="store_true",
                        help="validate and summarize an existing matrix without launching a process")
    args = parser.parse_args(argv)
    if args.dry_run and args.analyze_only:
        parser.error("--dry-run and --analyze-only are mutually exclusive")
    if args.analyze_only:
        if args.binary:
            parser.error("--binary is not used with --analyze-only")
        report = analyze_matrix(args.output_root)
    else:
        if not args.binary:
            parser.error("--binary is required unless --analyze-only is selected")
        if args.dry_run:
            report = dry_run_plan(args.binary, args.output_root, args.pause_seconds)
        else:
            report = launch_matrix(args.binary, args.output_root, args.pause_seconds)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, subprocess.SubprocessError) as error:
        print(f"pixelumm_after_dit_matrix: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
