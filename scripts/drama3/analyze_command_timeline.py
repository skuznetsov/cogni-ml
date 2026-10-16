#!/usr/bin/env python3
"""Analyze DRAMA-3 native command timing JSON lines without running the model.

Input is a selected stderr stream. Native command indices continue across warm
requests in a worker; phase ranges use those original indices. Duplicate indices
are rejected instead of guessing boundaries between separate workers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence


EVENT_NAME = "drama3_gpu_command"
FORMAT_NAME = "drama3-command-timeline-analysis-v1"
GPU_BOUND_TOLERANCE_SECONDS = 0.001
PARTITION_TOLERANCE_SECONDS = 1e-9


class TimelineError(ValueError):
    """Input contains an ambiguous or internally inconsistent timeline."""


def _finite_number(row: dict, key: str, index: int) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimelineError(f"command {index}: {key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TimelineError(f"command {index}: {key} must be a finite number")
    return result


def _phase_ranges(phases: Sequence[tuple[str, int, int]]) -> list[dict]:
    normalized = []
    names = set()
    for name, start, end in phases:
        if not name or name == "unassigned":
            raise TimelineError("phase names must be nonempty and cannot be 'unassigned'")
        if name in names:
            raise TimelineError(f"duplicate phase name {name!r}")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise TimelineError(f"phase {name!r}: start index must be a nonnegative integer")
        if isinstance(end, bool) or not isinstance(end, int) or end < start:
            raise TimelineError(f"phase {name!r}: end index must be an integer >= start index")
        names.add(name)
        normalized.append({"name": name, "start_index": start, "end_index": end})
    ordered = sorted(normalized, key=lambda item: (item["start_index"], item["end_index"]))
    for previous, current in zip(ordered, ordered[1:]):
        if current["start_index"] <= previous["end_index"]:
            raise TimelineError(
                f"phase ranges overlap: {previous['name']!r} and {current['name']!r}"
            )
    return normalized


def _phase_for(index: int, phases: Sequence[dict]) -> str:
    for phase in phases:
        if phase["start_index"] <= index <= phase["end_index"]:
            return phase["name"]
    return "unassigned"


def _parse_event(row: dict, phases: Sequence[dict]) -> dict:
    raw_index = row.get("index")
    if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
        raise TimelineError("command index must be a nonnegative integer")
    index = raw_index

    status = row.get("completion_status")
    if isinstance(status, bool) or not isinstance(status, int):
        raise TimelineError(f"command {index}: completion_status must be an integer")
    if status != 0:
        raise TimelineError(f"command {index}: completion_status is {status}, expected success (0)")

    host_names = (
        "host_before_commit_s",
        "host_after_commit_s",
        "host_before_wait_s",
        "host_after_wait_s",
    )
    host = {name: _finite_number(row, name, index) for name in host_names}
    if host["host_before_commit_s"] <= 0:
        raise TimelineError(f"command {index}: host_before_commit_s must be positive")
    if not (
        host["host_before_commit_s"]
        <= host["host_after_commit_s"]
        <= host["host_before_wait_s"]
        <= host["host_after_wait_s"]
    ):
        raise TimelineError(f"command {index}: host timestamp order is invalid")

    gpu_valid_flag = row.get("gpu_timing_valid", None)
    if gpu_valid_flag is not None and not isinstance(gpu_valid_flag, bool):
        raise TimelineError(f"command {index}: gpu_timing_valid must be boolean or null")

    gpu_bounds_missing = any(row.get(name) is None for name in ("gpu_start_s", "gpu_end_s"))
    gpu_start = gpu_end = None
    if gpu_valid_flag is True:
        gpu_start = _finite_number(row, "gpu_start_s", index)
        gpu_end = _finite_number(row, "gpu_end_s", index)
        before = host["host_before_commit_s"]
        after = host["host_after_wait_s"]
        if not (
            gpu_start > 0
            and gpu_end > gpu_start
            and gpu_start >= before - GPU_BOUND_TOLERANCE_SECONDS
            and gpu_end <= after + GPU_BOUND_TOLERANCE_SECONDS
        ):
            raise TimelineError(f"command {index}: GPU timing bounds are invalid")
        reported_gpu_seconds = _finite_number(row, "gpu_seconds", index)
        if reported_gpu_seconds <= 0 or not math.isclose(
            reported_gpu_seconds,
            gpu_end - gpu_start,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise TimelineError(f"command {index}: gpu_seconds disagrees with GPU timestamp bounds")

    commit_duration = host["host_after_commit_s"] - host["host_before_commit_s"]
    wait_duration = host["host_after_wait_s"] - host["host_before_wait_s"]
    gpu_duration = None
    wait_partition = None
    unpartitioned_wait = wait_duration
    commit_to_start = None
    if gpu_valid_flag is True:
        # Keep the signed raw bounds: timestamp validity permits 1ms uncertainty.
        gpu_duration = gpu_end - gpu_start
        commit_to_start = {
            "lower": gpu_start - host["host_after_commit_s"],
            "upper": gpu_start - host["host_before_commit_s"],
        }
        wait_start = host["host_before_wait_s"]
        wait_end = host["host_after_wait_s"]
        pre_gpu = max(0.0, min(wait_end, gpu_start) - wait_start)
        gpu_overlap = max(0.0, min(wait_end, gpu_end) - max(wait_start, gpu_start))
        post_gpu = max(0.0, wait_end - max(wait_start, gpu_end))
        partitioned = pre_gpu + gpu_overlap + post_gpu
        if not math.isclose(
            partitioned,
            wait_duration,
            rel_tol=PARTITION_TOLERANCE_SECONDS,
            abs_tol=PARTITION_TOLERANCE_SECONDS,
        ):
            raise TimelineError(f"command {index}: wait/GPU interval partition does not sum to wait")
        wait_partition = {
            "pre_gpu": pre_gpu,
            "gpu_overlap": gpu_overlap,
            "post_gpu": post_gpu,
        }
        unpartitioned_wait = 0.0

    return {
        "index": index,
        "phase": _phase_for(index, phases),
        "completion_status": status,
        "commit_duration_seconds": commit_duration,
        "commit_to_gpu_start_bounds_seconds": commit_to_start,
        "wait_duration_seconds": wait_duration,
        "gpu_execution_seconds": gpu_duration,
        "wait_partition_seconds": wait_partition,
        "unpartitioned_wait_seconds": unpartitioned_wait,
        "gpu_timing_valid": gpu_valid_flag is True,
        "gpu_timing_missing": gpu_valid_flag is None,
        "gpu_bounds_missing": gpu_bounds_missing,
    }


def _bound_summary(commands: Sequence[dict]) -> dict:
    bounds = [row["commit_to_gpu_start_bounds_seconds"] for row in commands if row["commit_to_gpu_start_bounds_seconds"] is not None]
    result = {"valid_command_count": len(bounds)}
    for key in ("lower", "upper"):
        values = [item[key] for item in bounds]
        result[key + "_seconds"] = None if not values else {
            "min": min(values),
            "mean": math.fsum(values) / len(values),
            "max": max(values),
        }
    return result


def _aggregate(commands: Sequence[dict]) -> dict:
    valid = [row for row in commands if row["gpu_timing_valid"]]
    invalid_count = sum(not row["gpu_timing_valid"] and not row["gpu_timing_missing"] for row in commands)
    missing_count = sum(row["gpu_timing_missing"] for row in commands)
    partition = {
        "pre_gpu": math.fsum(row["wait_partition_seconds"]["pre_gpu"] for row in valid),
        "gpu_overlap": math.fsum(row["wait_partition_seconds"]["gpu_overlap"] for row in valid),
        "post_gpu": math.fsum(row["wait_partition_seconds"]["post_gpu"] for row in valid),
        "partitioned_wait": math.fsum(row["wait_duration_seconds"] for row in valid),
        "unpartitioned_wait": math.fsum(row["unpartitioned_wait_seconds"] for row in commands),
    }
    return {
        "command_count": len(commands),
        "gpu_timing_valid_commands": len(valid),
        "gpu_timing_invalid_commands": invalid_count,
        "gpu_timing_missing_commands": missing_count,
        "gpu_bounds_missing_commands": sum(row["gpu_bounds_missing"] for row in commands),
        "commit_duration_seconds": math.fsum(row["commit_duration_seconds"] for row in commands),
        "commit_to_gpu_start_bounds": _bound_summary(commands),
        "wait_duration_seconds": math.fsum(row["wait_duration_seconds"] for row in commands),
        "wait_partition_seconds": partition,
        "gpu_execution_seconds_sum": math.fsum(row["gpu_execution_seconds"] for row in valid),
    }


def analyze_events(
    events: Iterable[dict],
    phases: Sequence[tuple[str, int, int]] = (),
    *,
    input_line_count: int | None = None,
    ignored_line_count: int = 0,
) -> dict:
    """Validate and analyze event objects from one selected request."""
    normalized_phases = _phase_ranges(phases)
    parsed = []
    seen = set()
    for row in events:
        if not isinstance(row, dict) or row.get("event") != EVENT_NAME:
            raise TimelineError("analyze_events accepts only drama3_gpu_command objects")
        item = _parse_event(row, normalized_phases)
        if item["index"] in seen:
            raise TimelineError(f"duplicate command index {item['index']}; select one request per input")
        seen.add(item["index"])
        parsed.append(item)
    if not parsed:
        raise TimelineError("no command timeline events found in the selected request")
    parsed.sort(key=lambda row: row["index"])

    phase_names = [phase["name"] for phase in normalized_phases]
    if "unassigned" not in phase_names:
        phase_names.append("unassigned")
    phase_totals = {
        name: _aggregate([row for row in parsed if row["phase"] == name])
        for name in phase_names
    }
    counts = _aggregate(parsed)
    return {
        "format": FORMAT_NAME,
        "status": "ok",
        "input": {
            "request_scope": "one selected request; command indices must be unique",
            "input_line_count": input_line_count,
            "ignored_line_count": ignored_line_count,
            "command_event_count": len(parsed),
        },
        "phase_ranges_inclusive": normalized_phases,
        "counts": {
            "command_events": len(parsed),
            "gpu_timing_valid_commands": counts["gpu_timing_valid_commands"],
            "gpu_timing_invalid_commands": counts["gpu_timing_invalid_commands"],
            "gpu_timing_missing_commands": counts["gpu_timing_missing_commands"],
            "gpu_bounds_missing_commands": counts["gpu_bounds_missing_commands"],
        },
        "interpretation": {
            "commit_to_gpu_start_bounds": "interval bounds from host commit boundaries; not pure queue time",
            "wait_partition": "intersection of the existing host wait interval with GPU execution; post-GPU tail is not proof of CPU scheduling delay",
            "gpu_execution_seconds_sum": "command durations can overlap across in-flight commands and are not request wall time",
            "request_wall_time": "not measured by this analyzer",
        },
        "totals": counts,
        "phases": phase_totals,
        "commands": parsed,
    }


def parse_stderr(lines: Iterable[str], phases: Sequence[tuple[str, int, int]] = ()) -> dict:
    events = []
    line_count = 0
    ignored = 0
    for line_number, line in enumerate(lines, start=1):
        line_count += 1
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if EVENT_NAME in line:
                raise TimelineError(f"line {line_number}: malformed command timeline JSON") from exc
            ignored += 1
            continue
        if isinstance(value, dict) and value.get("event") == EVENT_NAME:
            events.append(value)
        else:
            ignored += 1
    return analyze_events(events, phases, input_line_count=line_count, ignored_line_count=ignored)


def _parse_phase(value: str) -> tuple[str, int, int]:
    try:
        name, bounds = value.split("=", 1)
        start_text, end_text = bounds.split(":", 1)
        if not start_text.isdecimal() or not end_text.isdecimal():
            raise ValueError
        return name, int(start_text), int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("phase must be NAME=START:END with inclusive nonnegative indices") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze DRAMA-3 command timing JSON lines for one selected request. "
            "Duplicate command indices fail; extract one request before analysis."
        ),
        epilog=(
            "Example: analyze_command_timeline.py --input worker.stderr "
            "--phase prefill=0:15 --phase decode=16:120"
        ),
    )
    parser.add_argument("--input", default="-", help="stderr file to read, or '-' for stdin (default)")
    parser.add_argument(
        "--phase",
        action="append",
        type=_parse_phase,
        default=[],
        metavar="NAME=START:END",
        help="explicit inclusive command-index phase range; repeatable and non-overlapping",
    )
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            report = parse_stderr(sys.stdin, args.phase)
        else:
            with Path(args.input).open("r", encoding="utf-8") as source:
                report = parse_stderr(source, args.phase)
    except (OSError, TimelineError) as exc:
        print(json.dumps({"format": FORMAT_NAME, "status": "invalid", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, allow_nan=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
