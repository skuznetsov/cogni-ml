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
FORMAT_NAME = "drama3-command-timeline-analysis-v2"
GPU_BOUND_TOLERANCE_SECONDS = 0.001
PARTITION_TOLERANCE_SECONDS = 1e-9
RESOURCE_WINDOWS = (
    ("encoding_to_before_commit", "encoding_begin", "before_commit"),
    ("before_commit_to_after_commit", "before_commit", "after_commit"),
    ("after_commit_to_after_wait", "after_commit", "after_wait"),
)
RESOURCE_COUNTERS = (
    "user_ns",
    "system_ns",
    "pageins",
    "disk_read_bytes",
    "faults",
    "cow_faults",
)
TASK_EVENT_COUNTERS = ("pageins", "faults", "cow_faults")
RESOURCE_GAUGES = ("resident_bytes", "footprint_bytes")


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


def _kernel_schedule(row: dict, index: int) -> tuple[bool, float | None]:
    flag = row.get("kernel_timing_valid")
    if flag is not None and not isinstance(flag, bool):
        raise TimelineError(f"command {index}: kernel_timing_valid must be boolean or null")

    start_value = row.get("kernel_start_s")
    end_value = row.get("kernel_end_s")
    if flag is not True:
        if start_value is not None or end_value is not None:
            if flag is False:
                raise TimelineError(f"command {index}: unavailable kernel timing has timestamp values")
            raise TimelineError(f"command {index}: kernel timestamps are present without a valid flag")
        return False, None

    start = _finite_number(row, "kernel_start_s", index)
    end = _finite_number(row, "kernel_end_s", index)
    if start <= 0 or end < start:
        raise TimelineError(f"command {index}: kernel scheduling timestamp pair is invalid")
    return True, end - start


def _snapshot_timing(snapshot: object, label: str, index: int) -> tuple[float, float] | str:
    if not isinstance(snapshot, dict):
        return f"snapshot_missing:{label}"
    try:
        begin = _finite_number(snapshot, "sample_begin_s", index)
        end = _finite_number(snapshot, "sample_end_s", index)
    except TimelineError:
        return f"invalid_sample_timing:{label}"
    if begin <= 0 or end < begin:
        return f"invalid_sample_timing:{label}"
    return begin, end


def _resource_value(snapshot: dict, name: str) -> tuple[int | None, str | None]:
    value = snapshot.get(name)
    if value is None:
        return None, "unavailable"
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, "invalid_value"
    return value, None


def _counter_delta(before: dict, after: dict, name: str) -> dict:
    old, old_error = _resource_value(before, name)
    new, new_error = _resource_value(after, name)
    if old_error or new_error:
        return {"delta": None, "status": old_error or new_error}
    if new < old:
        return {"delta": None, "status": "counter_reset"}
    return {"delta": new - old, "status": "valid"}


def _gauge_change(before: dict, after: dict, name: str) -> dict:
    old, old_error = _resource_value(before, name)
    new, new_error = _resource_value(after, name)
    if old_error or new_error:
        return {"change": None, "status": old_error or new_error}
    return {"change": new - old, "status": "valid"}


def _resource_window(
    snapshots: object,
    before_name: str,
    after_name: str,
    index: int,
) -> dict:
    result = {
        "from": before_name,
        "to": after_name,
        "valid": False,
        "invalid_reason": None,
        "sample_window_seconds": None,
        "counters": {},
        "gauges": {},
    }
    if not isinstance(snapshots, dict):
        reason = "resource_object_invalid"
        before = after = {}
        before_timing = after_timing = reason
    else:
        before = snapshots.get(before_name)
        after = snapshots.get(after_name)
        before_timing = _snapshot_timing(before, before_name, index)
        after_timing = _snapshot_timing(after, after_name, index)

    if isinstance(before_timing, str):
        reason = before_timing
    elif isinstance(after_timing, str):
        reason = after_timing
    elif after_timing[0] < before_timing[1]:
        reason = "sample_intervals_overlap_or_are_out_of_order"
    else:
        reason = None

    if reason is not None:
        result["invalid_reason"] = reason
        result["counters"] = {
            field: {"delta": None, "status": "invalid_window"} for field in RESOURCE_COUNTERS
        }
        result["gauges"] = {
            field: {"change": None, "status": "invalid_window"} for field in RESOURCE_GAUGES
        }
        return result

    result["valid"] = True
    result["sample_window_seconds"] = {
        "lower": after_timing[0] - before_timing[1],
        "upper": after_timing[1] - before_timing[0],
    }
    task_events_unavailable = any(
        isinstance(snapshot.get(field), int)
        and not isinstance(snapshot.get(field), bool)
        and snapshot[field] < 0
        for snapshot in (before, after)
        for field in TASK_EVENT_COUNTERS
    )
    result["counters"] = {
        field: (
            {"delta": None, "status": "unavailable"}
            if task_events_unavailable and field in TASK_EVENT_COUNTERS
            else _counter_delta(before, after, field)
        )
        for field in RESOURCE_COUNTERS
    }
    result["gauges"] = {
        field: _gauge_change(before, after, field) for field in RESOURCE_GAUGES
    }
    return result


def _resource_windows(value: object, index: int) -> dict | None:
    if value is None:
        return None
    return {
        name: _resource_window(value, before, after, index)
        for name, before, after in RESOURCE_WINDOWS
    }


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

    driver_scheduling_valid, driver_scheduling_seconds = _kernel_schedule(row, index)
    resource_windows = _resource_windows(row.get("resources"), index)

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
        "driver_scheduling_seconds": driver_scheduling_seconds,
        "driver_scheduling_valid": driver_scheduling_valid,
        "driver_scheduling_missing": not driver_scheduling_valid,
        "resource_snapshots_missing": "resources" not in row or row.get("resources") is None,
        "resource_windows": resource_windows,
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


def _distribution(values: Sequence[int | float]) -> dict | None:
    if not values:
        return None
    return {"min": min(values), "mean": math.fsum(values) / len(values), "max": max(values)}


def _resource_window_summary(commands: Sequence[dict]) -> dict:
    result = {}
    for name, _, _ in RESOURCE_WINDOWS:
        windows = [
            row["resource_windows"][name]
            for row in commands
            if row["resource_windows"] is not None
        ]
        valid_windows = [window for window in windows if window["valid"]]
        timing = {
            "scope": "nonadditive process windows; in-flight commands can overlap",
            "window_count": len(windows),
            "valid_window_count": len(valid_windows),
            "invalid_window_count": len(windows) - len(valid_windows),
            "sample_window_seconds": None,
        }
        if valid_windows:
            timing["sample_window_seconds"] = {
                key: _distribution([window["sample_window_seconds"][key] for window in valid_windows])
                for key in ("lower", "upper")
            }

        counters = {}
        for field in RESOURCE_COUNTERS:
            values = [window["counters"][field] for window in windows]
            valid_values = [value["delta"] for value in values if value["status"] == "valid"]
            counters[field] = {
                "delta_distribution": _distribution(valid_values),
                "valid_count": len(valid_values),
                "unavailable_count": sum(value["status"] == "unavailable" for value in values),
                "counter_reset_count": sum(value["status"] == "counter_reset" for value in values),
                "invalid_value_count": sum(value["status"] == "invalid_value" for value in values),
                "invalid_window_count": sum(value["status"] == "invalid_window" for value in values),
            }

        gauges = {}
        for field in RESOURCE_GAUGES:
            values = [window["gauges"][field] for window in windows]
            valid_values = [value["change"] for value in values if value["status"] == "valid"]
            gauges[field] = {
                "change_distribution": _distribution(valid_values),
                "valid_count": len(valid_values),
                "unavailable_count": sum(value["status"] == "unavailable" for value in values),
                "invalid_value_count": sum(value["status"] == "invalid_value" for value in values),
                "invalid_window_count": sum(value["status"] == "invalid_window" for value in values),
            }
        result[name] = {**timing, "counters": counters, "gauges": gauges}
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
    driver_timing = [row["driver_scheduling_seconds"] for row in commands if row["driver_scheduling_valid"]]
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
        "driver_scheduling_valid_commands": len(driver_timing),
        "driver_scheduling_missing_commands": sum(row["driver_scheduling_missing"] for row in commands),
        "driver_scheduling_seconds_sum": math.fsum(driver_timing),
        "resource_snapshot_missing_commands": sum(row["resource_snapshots_missing"] for row in commands),
        "resource_counter_windows": _resource_window_summary(commands),
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
            "driver_scheduling_valid_commands": counts["driver_scheduling_valid_commands"],
            "driver_scheduling_missing_commands": counts["driver_scheduling_missing_commands"],
            "resource_snapshot_missing_commands": counts["resource_snapshot_missing_commands"],
        },
        "interpretation": {
            "commit_to_gpu_start_bounds": "interval bounds from host commit boundaries; not pure queue time",
            "wait_partition": "intersection of the existing host wait interval with GPU execution; post-GPU tail is not proof of CPU scheduling delay",
            "gpu_execution_seconds_sum": "command durations can overlap across in-flight commands and are not request wall time",
            "driver_scheduling_seconds": "kernelEndTime minus kernelStartTime is CPU-side driver scheduling duration, not GPU execution or one shader's duration",
            "driver_scheduling_seconds_sum": "scheduling intervals can overlap across in-flight commands; their sum is not request wall time or total process CPU time",
            "resource_counter_windows": "process deltas bracket named snapshot intervals and include all process activity; windows of in-flight commands can overlap, so distributions are nonadditive; use request boundary snapshots for request totals",
            "task_event_counters": "a negative TASK_EVENTS_INFO value makes pageins, faults, and cow_faults unavailable for that window",
            "pageins_and_disk_reads": "counter changes do not prove page-in caused a delay and are not converted into time",
            "cross_clock_timing": "no kernel scheduling to GPU start subtraction is reported because Apple documents the GPU clock as mach-relative but does not state the kernel timestamp epoch",
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
