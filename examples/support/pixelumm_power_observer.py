#!/usr/bin/env python3
"""Collect low-rate Apple GPU telemetry outside the PixelUMM Metal process.

The observer never opens a Metal device or waits on a command buffer. Its
powermetrics text parser preserves missing measurements as null. Output is
restricted to a fresh directory below /opt/mldata_ssd.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "cogni_ml.pixelumm_power_observer/v1"
SAMPLE_INTERVAL_MS = 2_000
MAX_SECONDS = 3_600
OUTPUT_BASE = Path("/opt/mldata_ssd")
RAW_NAME = "powermetrics.txt"
SAMPLES_NAME = "samples.jsonl"
SUMMARY_NAME = "summary.json"

SAMPLE_HEADER = re.compile(
    r"^\*\*\* Sampled system activity \((?P<timestamp>[^)\r\n]+)\) "
    r"\((?P<elapsed>[0-9]+(?:\.[0-9]+)?)ms elapsed\) \*\*\*$",
    re.MULTILINE,
)
FREQUENCY_RE = re.compile(
    r"^\s*GPU HW active frequency:\s*([0-9]+(?:\.[0-9]+)?)\s+MHz\s*$",
    re.MULTILINE,
)
RESIDENCY_RE = re.compile(
    r"^\s*GPU HW active residency:\s*([0-9]+(?:\.[0-9]+)?)%",
    re.MULTILINE,
)
POWER_RE = re.compile(
    r"^\s*GPU Power:\s*([0-9]+(?:\.[0-9]+)?)\s+mW\s*$",
    re.MULTILINE,
)
THERMAL_RE = re.compile(r"^\s*Current pressure level:\s*(\S.*?)\s*$", re.MULTILINE)
FREQUENCY_STATE_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s+MHz:\s*([0-9]+(?:\.[0-9]+)?)%"
)
REQUIRED_FIELDS = (
    "timestamp_utc",
    "elapsed_ms",
    "gpu_active_frequency_mhz",
    "gpu_active_residency_pct",
    "gpu_power_mw",
    "thermal_pressure",
)

_TERMINATION_SIGNAL: int | None = None
STOP_POLL_SECONDS = 0.5
STOP_COMMAND_TIMEOUT_SECONDS = 5
PGREP_PATH = "/usr/bin/pgrep"
STOP_ESCALATION = (
    (signal.SIGINT, 10),
    (signal.SIGTERM, 5),
    (signal.SIGKILL, 5),
)


class ObserverError(RuntimeError):
    """Raised for an invalid or unsafe observer request."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _iso_utc(value: datetime, *, timespec: str = "microseconds") -> str:
    return value.astimezone(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.strptime(value.strip(), "%a %b %d %H:%M:%S %Y %z")
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _number_matches(pattern: re.Pattern[str], block: str) -> list[float]:
    values: list[float] = []
    for match in pattern.findall(block):
        value = float(match)
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return values


def _single_reported_value(values: Sequence[float]) -> float | None:
    if not values:
        return None
    first = values[0]
    if all(math.isclose(value, first, rel_tol=0.0, abs_tol=1e-9) for value in values[1:]):
        return first
    return None


def _parse_sample(
    timestamp_header: str,
    timestamp_local: str,
    elapsed_ms: float,
    block: str,
    index: int,
) -> dict[str, Any]:
    timestamp = _parse_timestamp(timestamp_local)
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        elapsed_ms = float("nan")

    frequency = _single_reported_value(_number_matches(FREQUENCY_RE, block))
    residency = _single_reported_value(_number_matches(RESIDENCY_RE, block))
    power = _single_reported_value(_number_matches(POWER_RE, block))
    thermal_values = list(dict.fromkeys(value.strip() for value in THERMAL_RE.findall(block) if value.strip()))
    thermal = thermal_values[0] if len(thermal_values) == 1 else None

    if residency is not None and residency > 100:
        residency = None
    if math.isnan(elapsed_ms):
        elapsed_value: float | None = None
    else:
        elapsed_value = elapsed_ms

    if timestamp is None or elapsed_value is None:
        timestamp_utc = None
        timestamp_unix_ms = None
        window_start_utc = None
        window_start_unix_ms = None
        window_end_unix_ms = None
    else:
        timestamp_utc = _iso_utc(timestamp, timespec="seconds")
        timestamp_unix_ms = int(timestamp.timestamp() * 1_000)
        window_start = timestamp - timedelta(milliseconds=elapsed_value)
        window_start_utc = _iso_utc(window_start)
        window_start_unix_ms = round(timestamp_unix_ms - elapsed_value)
        window_end_unix_ms = timestamp_unix_ms

    frequency_states: list[dict[str, float]] | None = None
    residency_line = RESIDENCY_RE.search(block)
    if residency_line:
        line_end = block.find("\n", residency_line.start())
        full_line = block[residency_line.start():line_end if line_end >= 0 else len(block)]
        states = [
            {"frequency_mhz": float(freq), "residency_pct": float(percent)}
            for freq, percent in FREQUENCY_STATE_RE.findall(full_line)
        ]
        frequency_states = states or None

    sample: dict[str, Any] = {
        "sample_index": index,
        "timestamp_header": timestamp_header,
        "timestamp_local": timestamp_local,
        "timestamp_utc": timestamp_utc,
        "timestamp_unix_ms": timestamp_unix_ms,
        "timestamp_resolution_ms": 1_000,
        "timestamp_is_window_end_assumption": True,
        "elapsed_ms": elapsed_value,
        "sample_window_start_utc": window_start_utc,
        "sample_window_start_unix_ms": window_start_unix_ms,
        "sample_window_end_utc": timestamp_utc,
        "sample_window_end_unix_ms": window_end_unix_ms,
        "gpu_active_frequency_mhz": frequency,
        "gpu_active_residency_pct": residency,
        "gpu_frequency_states_mhz": frequency_states,
        "gpu_power_mw": power,
        "thermal_pressure": thermal,
    }
    missing = [field for field in REQUIRED_FIELDS if sample[field] is None]
    sample["missing_fields"] = missing
    sample["valid"] = not missing
    return sample


def parse_powermetrics_text(output: str | bytes) -> list[dict[str, Any]]:
    """Parse sample blocks from `powermetrics -f text` without filling gaps."""

    if isinstance(output, bytes):
        text = output.decode("utf-8", errors="replace")
    else:
        text = output
    headers = list(SAMPLE_HEADER.finditer(text))
    samples: list[dict[str, Any]] = []
    for index, header in enumerate(headers):
        block_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[header.end():block_end]
        timestamp_local = header.group("timestamp").strip()
        elapsed_ms = float(header.group("elapsed"))
        samples.append(_parse_sample(
            header.group(0), timestamp_local, elapsed_ms, block, index
        ))
    return samples


def _numeric_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    expected_sample_count: int,
    interrupted: bool = False,
) -> dict[str, Any]:
    if isinstance(expected_sample_count, bool) or expected_sample_count < 1:
        raise ValueError("expected_sample_count must be a positive integer")
    parsed = list(samples)
    valid = [sample for sample in parsed if sample.get("valid") is True]
    field_missing = Counter(
        field for sample in parsed for field in sample.get("missing_fields", [])
    )
    fraction = len(valid) / len(parsed) if parsed else None
    duration_complete = not interrupted and len(parsed) == expected_sample_count
    all_valid = bool(parsed) and len(valid) == len(parsed)
    return {
        "sample_count": len(parsed),
        "expected_sample_count": expected_sample_count,
        "valid_sample_count": len(valid),
        "valid_sample_fraction": fraction,
        "all_observed_samples_valid": all_valid,
        "duration_complete": duration_complete,
        "interrupted": interrupted,
        "missing_field_counts": {key: field_missing[key] for key in sorted(field_missing)},
        "valid_duration_seconds": sum(float(sample["elapsed_ms"]) for sample in valid) / 1_000,
        "gpu_active_frequency_mhz": _numeric_summary([
            float(sample["gpu_active_frequency_mhz"]) for sample in valid
        ]),
        "gpu_active_residency_pct": _numeric_summary([
            float(sample["gpu_active_residency_pct"]) for sample in valid
        ]),
        "gpu_power_mw": _numeric_summary([
            float(sample["gpu_power_mw"]) for sample in valid
        ]),
        "thermal_pressure_counts": dict(sorted(Counter(
            str(sample["thermal_pressure"]) for sample in valid
        ).items())),
    }


def summarize_utc_ranges(
    samples: Sequence[Mapping[str, Any]],
    ranges: Mapping[str, tuple[int, int] | Mapping[str, int]],
) -> dict[str, Any]:
    """Summarize pass/recovery windows, excluding samples near each boundary.

    Range bounds are Unix milliseconds, inclusive. A sample is eligible only
    when its complete measurement window is inside the range and it does not
    intersect a +/- one-sample-interval exclusion band around either boundary.
    """

    result: dict[str, Any] = {}
    for label, bounds in ranges.items():
        if isinstance(bounds, Mapping):
            start_ms = bounds.get("start_unix_ms")
            end_ms = bounds.get("end_unix_ms")
        else:
            start_ms, end_ms = bounds
        if (isinstance(start_ms, bool) or not isinstance(start_ms, int) or
                isinstance(end_ms, bool) or not isinstance(end_ms, int) or start_ms >= end_ms):
            raise ValueError(f"invalid Unix-ms range for {label!r}")

        overlapping = []
        eligible = []
        boundary_excluded = 0
        for sample in samples:
            window_start = sample.get("sample_window_start_unix_ms")
            window_end = sample.get("sample_window_end_unix_ms")
            interval = sample.get("elapsed_ms")
            if (not isinstance(window_start, int) or isinstance(window_start, bool) or
                    not isinstance(window_end, int) or isinstance(window_end, bool) or
                    not isinstance(interval, (int, float)) or not math.isfinite(float(interval))):
                continue
            if window_end < start_ms or window_start > end_ms:
                continue
            overlapping.append(sample)
            if window_start < start_ms or window_end > end_ms:
                boundary_excluded += 1
                continue
            exclusion_ms = float(interval)
            near_start = window_end >= start_ms - exclusion_ms and window_start <= start_ms + exclusion_ms
            near_end = window_end >= end_ms - exclusion_ms and window_start <= end_ms + exclusion_ms
            if near_start or near_end:
                boundary_excluded += 1
                continue
            eligible.append(sample)

        valid = [sample for sample in eligible if sample.get("valid") is True]
        result[label] = {
            "start_unix_ms": start_ms,
            "end_unix_ms": end_ms,
            "overlapping_samples": len(overlapping),
            "boundary_excluded_samples": boundary_excluded,
            "eligible_samples": len(eligible),
            "valid_samples": len(valid),
            "valid_sample_fraction": len(valid) / len(eligible) if eligible else None,
            "gpu_active_frequency_mhz": _numeric_summary([
                float(sample["gpu_active_frequency_mhz"]) for sample in valid
            ]),
            "gpu_active_residency_pct": _numeric_summary([
                float(sample["gpu_active_residency_pct"]) for sample in valid
            ]),
            "gpu_power_mw": _numeric_summary([
                float(sample["gpu_power_mw"]) for sample in valid
            ]),
            "thermal_pressure_counts": dict(sorted(Counter(
                str(sample["thermal_pressure"]) for sample in valid
            ).items())),
        }
    return result


def expected_sample_count(seconds: int) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 2 <= seconds <= MAX_SECONDS:
        raise ValueError(f"seconds must be an integer in [2, {MAX_SECONDS}]")
    return math.ceil(seconds * 1_000 / SAMPLE_INTERVAL_MS)


def build_command(seconds: int, *, sudo_path: str = "/usr/bin/sudo",
                  powermetrics_path: str = "/usr/bin/powermetrics") -> list[str]:
    count = expected_sample_count(seconds)
    return [
        sudo_path, "-n", powermetrics_path,
        "-i", str(SAMPLE_INTERVAL_MS), "-n", str(count),
        "-s", "gpu_power,thermal", "-f", "text",
    ]


def validate_output_dir(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise ObserverError("--output-dir must be an absolute path without parent traversal")
    if not OUTPUT_BASE.is_dir():
        raise ObserverError(f"{OUTPUT_BASE} must resolve to a mounted directory")
    base = OUTPUT_BASE.resolve(strict=True)
    try:
        if base.stat().st_dev == Path("/opt").stat().st_dev:
            raise ObserverError(f"{OUTPUT_BASE} must be on a separate mounted volume")
    except OSError as error:
        raise ObserverError(f"cannot verify the external volume for {OUTPUT_BASE}") from error
    try:
        relative = path.relative_to(OUTPUT_BASE)
    except ValueError as error:
        raise ObserverError(f"--output-dir must be below {OUTPUT_BASE}") from error
    if not relative.parts:
        raise ObserverError("--output-dir must be a fresh child below the output base")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or not parent.is_relative_to(base):
        raise ObserverError(f"--output-dir parent must already exist below {OUTPUT_BASE}")
    current = OUTPUT_BASE
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ObserverError("--output-dir may not traverse symlinked directories")
    if path.exists() or path.is_symlink():
        raise ObserverError("--output-dir must be fresh")
    return path


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    _write_new(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def _signal_child(signum: int, _frame: Any) -> None:
    global _TERMINATION_SIGNAL
    if _TERMINATION_SIGNAL is None:
        _TERMINATION_SIGNAL = signum


def _signal_name(signum: int) -> str:
    names = {
        signal.SIGINT: "INT",
        signal.SIGTERM: "TERM",
        signal.SIGKILL: "KILL",
    }
    try:
        return names[signum]
    except KeyError as error:
        raise ValueError(f"unsupported sampler stop signal: {signum}") from error


def _send_privileged_group_signal(
    sudo_path: str,
    process_group_id: int,
    signum: int,
) -> dict[str, Any]:
    if isinstance(process_group_id, bool) or process_group_id <= 0:
        raise ValueError("sampler process group ID must be positive")
    name = _signal_name(signum)
    command = [sudo_path, "-n", "/bin/kill", f"-{name}", "--", f"-{process_group_id}"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=STOP_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        return {
            "signal": name,
            "command": command,
            "returncode": completed.returncode,
            "sent": completed.returncode == 0,
            "stderr_excerpt": _safe_excerpt(completed.stderr or b""),
            "error": None,
        }
    except (OSError, subprocess.SubprocessError) as error:
        stderr = getattr(error, "stderr", None) or b""
        return {
            "signal": name,
            "command": command,
            "returncode": None,
            "sent": False,
            "stderr_excerpt": _safe_excerpt(stderr),
            "error": f"{type(error).__name__}: {error}",
        }


def _communicate_until_exit(
    child: subprocess.Popen[bytes], timeout_seconds: float
) -> tuple[bool, bytes]:
    try:
        _stdout, stderr = child.communicate(timeout=max(0.0, timeout_seconds))
        return True, stderr or b""
    except subprocess.TimeoutExpired as error:
        return False, error.stderr or b""


def _check_privileged_group_presence(
    sudo_path: str,
    process_group_id: int,
) -> dict[str, Any]:
    """Distinguish an empty owned process group from a failed presence check."""
    if isinstance(process_group_id, bool) or process_group_id <= 0:
        raise ValueError("sampler process group ID must be positive")
    command = [sudo_path, "-n", PGREP_PATH, "-g", str(process_group_id)]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=STOP_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        stderr = completed.stderr or b""
        if completed.returncode == 0:
            state = "present"
        elif completed.returncode == 1 and not stderr:
            # BSD pgrep uses 1 only when no process matched. A sudo denial also
            # commonly exits 1, but emits stderr and must remain unknown.
            state = "absent"
        else:
            state = "unknown"
        return {
            "state": state,
            "process_group_id": process_group_id,
            "command": command,
            "returncode": completed.returncode,
            "stderr_excerpt": _safe_excerpt(stderr),
            "error": None,
        }
    except (OSError, subprocess.SubprocessError) as error:
        stderr = getattr(error, "stderr", None) or b""
        return {
            "state": "unknown",
            "process_group_id": process_group_id,
            "command": command,
            "returncode": None,
            "stderr_excerpt": _safe_excerpt(stderr),
            "error": f"{type(error).__name__}: {error}",
        }


def _stop_sampler_group(
    child: subprocess.Popen[bytes],
    process_group_id: int,
    sudo_path: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    group_checks: list[dict[str, Any]] = []
    stderr = b""

    initial_presence = _check_privileged_group_presence(sudo_path, process_group_id)
    group_checks.append(initial_presence)
    if initial_presence["state"] == "absent":
        stopped, latest_stderr = _communicate_until_exit(child, 0)
        stderr = latest_stderr or stderr
        return {
            "stopped": stopped,
            "attempts": attempts,
            "group_checks": group_checks,
            "stderr": stderr,
        }

    for signum, grace_seconds in STOP_ESCALATION:
        attempt = _send_privileged_group_signal(sudo_path, process_group_id, signum)
        attempts.append(attempt)
        stopped, latest_stderr = _communicate_until_exit(child, grace_seconds)
        stderr = latest_stderr or stderr
        if stopped:
            presence = _check_privileged_group_presence(sudo_path, process_group_id)
            group_checks.append(presence)
            if presence["state"] == "absent":
                return {
                    "stopped": True,
                    "attempts": attempts,
                    "group_checks": group_checks,
                    "stderr": stderr,
                }
    return {
        "stopped": False,
        "attempts": attempts,
        "group_checks": group_checks,
        "stderr": stderr,
    }


def _safe_excerpt(raw: bytes, limit: int = 500) -> str | None:
    value = raw.decode("utf-8", errors="replace").strip()
    return value[:limit] if value else None


def collect(
    output_dir: str | Path,
    seconds: int,
    *,
    on_started: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise ObserverError("powermetrics observer is supported only on macOS")
    count = expected_sample_count(seconds)
    output_root = validate_output_dir(output_dir)
    sudo_path = shutil.which("sudo") or "/usr/bin/sudo"
    powermetrics_path = shutil.which("powermetrics") or "/usr/bin/powermetrics"
    if not Path(sudo_path).is_file() or not Path(powermetrics_path).is_file():
        raise ObserverError("sudo and powermetrics executables must be present")
    command = build_command(seconds, sudo_path=sudo_path, powermetrics_path=powermetrics_path)
    output_root.mkdir(mode=0o700, exist_ok=False)

    started_utc = _utc_now()
    started_monotonic_ns = time.monotonic_ns()
    stderr = b""
    returncode: int | None = None
    timed_out = False
    launch_error: str | None = None
    child_process: subprocess.Popen[bytes] | None = None
    process_group_id: int | None = None
    stop_result: dict[str, Any] = {
        "stopped": False,
        "attempts": [],
        "group_checks": [],
        "stderr": b"",
    }
    raw_path = output_root / RAW_NAME
    raw_descriptor = os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(raw_descriptor, "wb") as raw_stream:
        try:
            child_process = subprocess.Popen(
                command,
                stdout=raw_stream,
                stderr=subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
            )
            process_group_id = child_process.pid
            if on_started is not None:
                try:
                    on_started({
                        "event": "started",
                        "observer_pid": os.getpid(),
                        "sampler_process_group_id": process_group_id,
                        "output_dir": str(output_root),
                        "sample_interval_ms": SAMPLE_INTERVAL_MS,
                        "planned_sample_count": count,
                        "command": command,
                    })
                except OSError:
                    pass
            deadline = time.monotonic() + seconds + 20
            while True:
                if _TERMINATION_SIGNAL is not None:
                    complete, latest_stderr = _communicate_until_exit(child_process, 0)
                    stderr = latest_stderr or stderr
                    if complete:
                        stop_result = _stop_sampler_group(
                            child_process, process_group_id, sudo_path
                        )
                    else:
                        stop_result = _stop_sampler_group(child_process, process_group_id, sudo_path)
                    stderr = stop_result["stderr"] or stderr
                    break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    stop_result = _stop_sampler_group(
                        child_process, process_group_id, sudo_path
                    )
                    stderr = stop_result["stderr"] or stderr
                    break

                complete, latest_stderr = _communicate_until_exit(
                    child_process, min(STOP_POLL_SECONDS, remaining)
                )
                stderr = latest_stderr or stderr
                if complete:
                    stop_result = _stop_sampler_group(
                        child_process, process_group_id, sudo_path
                    )
                    stderr = stop_result["stderr"] or stderr
                    break
            returncode = child_process.returncode
        except (OSError, subprocess.SubprocessError) as error:
            launch_error = f"{type(error).__name__}: {error}"
            if child_process is not None and process_group_id is not None:
                stop_result = _stop_sampler_group(child_process, process_group_id, sudo_path)
                stderr = stop_result["stderr"] or stderr
                returncode = child_process.returncode
        finally:
            raw_stream.flush()
            os.fsync(raw_stream.fileno())

    finished_monotonic_ns = time.monotonic_ns()
    finished_utc = _utc_now()
    raw_capture = raw_path.read_bytes()
    samples = parse_powermetrics_text(raw_capture)
    interrupted = _TERMINATION_SIGNAL is not None
    sampler_stopped = child_process is None or stop_result["stopped"]
    signal_exit_codes = {
        -signal.SIGINT, 128 + signal.SIGINT,
        -signal.SIGTERM, 128 + signal.SIGTERM,
        -signal.SIGKILL, 128 + signal.SIGKILL,
    }
    sampler_terminated_by_signal = returncode in signal_exit_codes
    interrupted = interrupted or sampler_terminated_by_signal
    coverage = summarize_samples(samples, expected_sample_count=count, interrupted=interrupted)
    observed_valid = coverage["all_observed_samples_valid"] and coverage["sample_count"] > 0
    process_ok = returncode == 0 or (interrupted and sampler_terminated_by_signal)
    if not sampler_stopped:
        status = "stop_failed"
    elif observed_valid and process_ok and not timed_out and launch_error is None:
        status = "complete" if coverage["duration_complete"] else "stopped_early"
    elif samples:
        status = "partial"
    else:
        status = "failed"

    samples_bytes = b"".join(
        (json.dumps(sample, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        for sample in samples
    )
    samples_path = output_root / SAMPLES_NAME
    _write_new(samples_path, samples_bytes)
    summary = {
        "schema": SCHEMA,
        "status": status,
        "collector": {
            "observer_pid": os.getpid(),
            "powermetrics_executable": powermetrics_path,
            "sudo_executable": sudo_path,
            "command": command,
            "sampler_process_group_id": process_group_id,
            "sample_interval_ms": SAMPLE_INTERVAL_MS,
            "requested_seconds": seconds,
            "planned_sample_count": count,
            "actual_returncode": returncode,
            "interrupted_by_signal": _TERMINATION_SIGNAL,
            "sampler_terminated_by_signal": sampler_terminated_by_signal,
            "timed_out": timed_out,
            "sampler_stopped": sampler_stopped,
            "stop_attempts": stop_result["attempts"],
            "sampler_group_checks": stop_result["group_checks"],
            "launch_error": launch_error,
            "stderr_excerpt": _safe_excerpt(stderr),
            "format": "text",
            "format_rationale": "This host's one-sample plist has GPU frequency/residency and thermal_pressure but no direct power field; text reports GPU Power in mW.",
        },
        "clock": {
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "started_monotonic_ns": started_monotonic_ns,
            "finished_monotonic_ns": finished_monotonic_ns,
            "monotonic_elapsed_ns": finished_monotonic_ns - started_monotonic_ns,
            "sample_timestamp_timezone": "parsed from the numeric offset printed in each sample header, then normalized to UTC",
            "sample_timestamp_resolution_ms": 1_000,
            "sample_timestamp_semantics": "treated as the sample-window end; the exact header is preserved, but its timestamp is printed at whole-second precision and may be truncated or rounded",
            "sample_window_start": "sample timestamp minus the printed elapsed_ms",
            "alignment_caveat": "Per-sample wall timestamps can align with pass started_unix_ms/finished_unix_ms, but are not an exact mapping to the probe's monotonic clock. Exclude samples within one sample interval of pass/recovery boundaries.",
        },
        "raw_capture": {
            "file": RAW_NAME,
            "sha256": hashlib.sha256(raw_capture).hexdigest(),
            "size_bytes": len(raw_capture),
            "finalized": sampler_stopped,
        },
        "parsed_samples": {
            "file": SAMPLES_NAME,
            "sha256": hashlib.sha256(samples_bytes).hexdigest(),
            "size_bytes": len(samples_bytes),
            "coverage": coverage,
        },
        "utc_range_summary_api": "summarize_utc_ranges(samples, {name: (start_unix_ms, end_unix_ms)})",
    }
    _write_json_new(output_root / SUMMARY_NAME, summary)
    return summary


def _seconds_arg(value: str) -> int:
    try:
        seconds = int(value)
        expected_sample_count(seconds)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return seconds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory below /opt/mldata_ssd; all artifacts are written there")
    parser.add_argument("--seconds", type=_seconds_arg, required=True,
                        help=f"bounded capture duration, 2..{MAX_SECONDS}; SIGTERM also stops cleanly")
    args = parser.parse_args(argv)

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, _signal_child)
    global _TERMINATION_SIGNAL
    _TERMINATION_SIGNAL = None
    try:
        summary = collect(
            args.output_dir,
            args.seconds,
            on_started=lambda event: print(
                json.dumps(event, sort_keys=True), flush=True
            ),
        )
    except (ObserverError, OSError, ValueError) as error:
        print(f"pixelumm_power_observer: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    print(json.dumps({
        "event": "finished",
        "observer_pid": os.getpid(),
        "status": summary["status"],
        "output_dir": str(Path(args.output_dir)),
        "summary": str(Path(args.output_dir) / SUMMARY_NAME),
        "raw_capture": str(Path(args.output_dir) / RAW_NAME),
        "samples": str(Path(args.output_dir) / SAMPLES_NAME),
        "sample_count": summary["parsed_samples"]["coverage"]["sample_count"],
        "valid_sample_fraction": summary["parsed_samples"]["coverage"]["valid_sample_fraction"],
    }, sort_keys=True))
    coverage = summary["parsed_samples"]["coverage"]
    return 0 if summary["status"] in ("complete", "stopped_early") and coverage["all_observed_samples_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
