#!/usr/bin/env python3
"""Analyze completed-command DRAMA-3 prefill timings without causal claims."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


COMPONENTS = ("moe", "backbone")
PRIME_COUNT = 2
ITERATIONS_PER_COMPONENT = 10
EVENT_NAMES = {
    "prefill_benchmark_begin",
    "prefill_benchmark_end",
    "drama3_gpu_command",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: object, message: str, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, message)
    return int(value)


def finite_number(value: object, message: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), message)
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(message) from exc
    require(math.isfinite(result), message)
    return result


def finite_positive(value: object, message: str) -> float:
    result = finite_number(value, message)
    require(result > 0.0, message)
    return result


def _expected_observations(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    require(benchmark.get("persistent_context") is True,
            "persistent context is required for qualified warm timings")
    chunk = integer(benchmark.get("chunk"), "invalid batch chunk", 2)
    require(chunk <= 32, "batch chunk exceeds benchmark bound")
    rows = integer(benchmark.get("rows"), "invalid row count", 1)
    schedule = (1, chunk, 1, chunk, chunk, 1, chunk, 1, 1, chunk)
    raw = benchmark.get("observations")
    require(isinstance(raw, list), "missing benchmark observations")
    require(len(raw) == len(COMPONENTS) * ITERATIONS_PER_COMPONENT,
            "incomplete benchmark observation schedule")

    expected = []
    for component_index, component in enumerate(COMPONENTS):
        for iteration, scheduled_chunk in enumerate(schedule):
            observation_index = component_index * ITERATIONS_PER_COMPONENT + iteration
            observation = raw[observation_index]
            require(isinstance(observation, dict), "invalid benchmark observation")
            actual_component = observation.get("component")
            actual_iteration = integer(observation.get("iteration"), "invalid observation iteration")
            actual_chunk = integer(observation.get("chunk"), "invalid observation chunk", 1)
            key = (actual_component, actual_iteration, actual_chunk)
            require(key == (component, iteration, scheduled_chunk),
                    "changed ABBA/BAAB schedule")
            seconds = finite_positive(observation.get("seconds"), "invalid benchmark wall time")

            graph = observation.get("graph")
            require(isinstance(graph, dict), "missing command completion counters")
            submitted = integer(graph.get("submitted"), "invalid submitted command count", 1)
            completed = integer(graph.get("completed"), "invalid completed command count")
            waits = integer(graph.get("completion_waits"), "invalid completion wait count")
            pending = integer(graph.get("pending"), "invalid pending command count")
            require(submitted == completed == waits, "completion counters do not match submissions")
            require(pending == 0, "pending commands remain after benchmark end")
            expected_commands = (rows + scheduled_chunk - 1) // scheduled_chunk
            require(submitted == expected_commands,
                    "command count does not match measured row/chunk geometry")

            readbacks = observation.get("readbacks")
            require(isinstance(readbacks, dict), "missing readbacks counters")
            for name, value in readbacks.items():
                count = integer(value, f"invalid readbacks counter: {name}")
                require(count == 0, "benchmark performed readbacks")

            expected.append({
                "component": component,
                "iteration": iteration,
                "chunk": scheduled_chunk,
                "seconds": seconds,
                "command_count": submitted,
                "rows": rows,
            })
    return expected


def _recognized_events(log_text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    events = []
    for number, line in enumerate(log_text.splitlines(), 1):
        start = line.find("{")
        if start < 0:
            if any(name in line for name in EVENT_NAMES):
                raise ValueError(f"invalid JSON event at log line {number}")
            continue
        try:
            value, _ = decoder.raw_decode(line[start:])
        except json.JSONDecodeError as exc:
            if any(name in line for name in EVENT_NAMES):
                raise ValueError(f"invalid JSON event at log line {number}") from exc
            continue
        if isinstance(value, dict) and value.get("event") in EVENT_NAMES:
            events.append(value)
    return events


def _command_timing(event: dict[str, Any]) -> dict[str, float | None]:
    seconds = finite_positive(event.get("gpu_seconds"), "invalid GPU time")
    if "completion_status" in event:
        status = integer(event.get("completion_status"), "invalid command completion status", 0)
        require(status == 0, "command completion status is not zero")

    rich_fields = {"gpu_timing_valid", "gpu_start_s", "gpu_end_s"}
    has_rich_timing = any(field in event for field in rich_fields)
    if not has_rich_timing:
        return {"gpu_seconds": seconds, "gpu_start_s": None, "gpu_end_s": None}

    require(event.get("completion_status") == 0, "rich GPU timing lacks successful completion status")
    require(event.get("gpu_timing_valid") is True, "invalid GPU time interval")
    start = finite_number(event.get("gpu_start_s"), "invalid GPU time interval")
    end = finite_number(event.get("gpu_end_s"), "invalid GPU time interval")
    require(end > start, "invalid GPU time interval")
    require(math.isclose(end - start, seconds, rel_tol=1e-9, abs_tol=1e-12),
            "GPU duration disagrees with completed interval")
    return {"gpu_seconds": seconds, "gpu_start_s": start, "gpu_end_s": end}


def _measurement_scopes(log_text: str, expected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    scopes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    expected_index = 0
    unscoped_commands = 0
    globally_seen_indices: set[int] = set()
    last_scoped_index: int | None = None

    for event in _recognized_events(log_text):
        kind = event.get("event")
        if kind == "prefill_benchmark_begin":
            require(current is None, "nested measured scope")
            require(expected_index < len(expected), "unexpected extra measured scope")
            key = _event_identity(event)
            wanted = expected[expected_index]
            require(key == (wanted["component"], wanted["iteration"], wanted["chunk"]),
                    "scope does not match benchmark observation")
            current = {**wanted, "commands": []}
        elif kind == "drama3_gpu_command":
            if current is None:
                # Preparation and teardown commands are deliberately outside the measured scope.
                unscoped_commands += 1
                continue
            index = integer(event.get("index"), "invalid GPU command index")
            require(index not in globally_seen_indices, "duplicate scoped GPU command index")
            require(last_scoped_index is None or index > last_scoped_index,
                    "out-of-order scoped GPU command index")
            globally_seen_indices.add(index)
            last_scoped_index = index
            current["commands"].append({"index": index, **_command_timing(event)})
        elif kind == "prefill_benchmark_end":
            require(current is not None, "end marker without begin")
            key = _event_identity(event)
            expected_key = (current["component"], current["iteration"], current["chunk"])
            require(key == expected_key, "measurement end marker does not match begin")
            marker_seconds = finite_positive(event.get("seconds"), "invalid marker wall time")
            require(math.isclose(marker_seconds, current["seconds"], rel_tol=1e-9, abs_tol=1e-12),
                    "marker wall time does not match benchmark observation")
            commands = current["commands"]
            require(len(commands) == current["command_count"],
                    "GPU command count does not match submissions")
            require(commands, "measured scope has no completed GPU commands")
            try:
                current["gpu_interval_sum_seconds"] = math.fsum(
                    command["gpu_seconds"] for command in commands)
            except OverflowError as exc:
                raise ValueError("invalid completed GPU interval sum") from exc
            require(math.isfinite(current["gpu_interval_sum_seconds"]) and
                    current["gpu_interval_sum_seconds"] > 0,
                    "invalid completed GPU interval sum")
            starts = [command["gpu_start_s"] for command in commands]
            ends = [command["gpu_end_s"] for command in commands]
            has_absolute_times = [value is not None for value in starts + ends]
            require(all(has_absolute_times) or not any(has_absolute_times),
                    "incomplete absolute GPU timing coverage")
            if all(has_absolute_times):
                current["gpu_span_seconds"] = max(ends) - min(starts)
                require(math.isfinite(current["gpu_span_seconds"]) and current["gpu_span_seconds"] > 0,
                        "invalid GPU span")
            else:
                current["gpu_span_seconds"] = None
            scopes.append(current)
            current = None
            expected_index += 1

    require(current is None, "unterminated measured scope")
    require(expected_index == len(expected), "missing measured benchmark scope")
    return scopes, unscoped_commands


def _event_identity(event: dict[str, Any]) -> tuple[object, int, int]:
    component = event.get("component")
    require(component in COMPONENTS, "invalid measured component")
    iteration = integer(event.get("iteration"), "invalid measured iteration")
    chunk = integer(event.get("chunk"), "invalid measured chunk", 1)
    return component, iteration, chunk


def _distribution(values: list[float]) -> dict[str, float | list[float]]:
    require(bool(values), "cannot summarize empty timing sample")
    low, high = min(values), max(values)
    return {"median": statistics.median(values), "min": low, "max": high,
            "range": [low, high]}


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spans = [row["gpu_span_seconds"] for row in rows]
    require(all(value is None for value in spans) or all(value is not None for value in spans),
            "inconsistent GPU span availability")
    try:
        gpu_sums = [math.fsum(command["gpu_seconds"] for command in row["commands"])
                    for row in rows]
    except OverflowError as exc:
        raise ValueError("invalid completed GPU interval sum") from exc
    require(all(math.isfinite(value) and value > 0 for value in gpu_sums),
            "invalid completed GPU interval sum")
    summary: dict[str, Any] = {
        "samples": len(rows),
        "wall_seconds": _distribution([row["seconds"] for row in rows]),
        "completed_gpu_interval_sum_seconds": _distribution(gpu_sums),
    }
    summary["gpu_span_seconds"] = _distribution(spans) if spans and spans[0] is not None else None
    return summary


def _component_report(scopes: list[dict[str, Any]], component: str) -> dict[str, Any]:
    measured = [scope for scope in scopes
                if scope["component"] == component and scope["iteration"] >= PRIME_COUNT]
    require(len(measured) == 8, "missing measured ABBA/BAAB samples")
    arms = {
        "scalar": [row for row in measured if row["chunk"] == 1],
        "batched": [row for row in measured if row["chunk"] > 1],
    }
    require(all(len(rows) == 4 for rows in arms.values()), "unbalanced measured arms")

    block_specs = (("ABBA", (2, 3, 4, 5)), ("BAAB", (6, 7, 8, 9)))
    blocks = []
    for pattern, iterations in block_specs:
        block_rows = [row for row in measured if row["iteration"] in iterations]
        actual = ["scalar" if row["chunk"] == 1 else "batched"
                  for row in sorted(block_rows, key=lambda row: row["iteration"])]
        expected_order = ["scalar", "batched", "batched", "scalar"] if pattern == "ABBA" else [
            "batched", "scalar", "scalar", "batched"]
        require(actual == expected_order, "changed ABBA/BAAB schedule")
        blocks.append({
            "pattern": pattern,
            "iterations": list(iterations),
            "order": actual,
            "arms": {arm: _arm_summary([row for row in block_rows if
                                        (row["chunk"] == 1) == (arm == "scalar")])
                     for arm in ("scalar", "batched")},
        })

    scalar, batched = _arm_summary(arms["scalar"]), _arm_summary(arms["batched"])
    wall_speedup = scalar["wall_seconds"]["median"] / batched["wall_seconds"]["median"]
    gpu_speedup = (scalar["completed_gpu_interval_sum_seconds"]["median"] /
                   batched["completed_gpu_interval_sum_seconds"]["median"])
    require(math.isfinite(wall_speedup) and math.isfinite(gpu_speedup), "invalid measured speedup ratio")
    return {
        "patterns": ["ABBA", "BAAB"],
        "blocks": blocks,
        "arms": {"scalar": scalar, "batched": batched},
        "samples": [{
            "iteration": row["iteration"],
            "pattern": "ABBA" if row["iteration"] <= 5 else "BAAB",
            "arm": "scalar" if row["chunk"] == 1 else "batched",
            "chunk": row["chunk"],
            "command_count": row["command_count"],
            "wall_seconds": row["seconds"],
            "completed_gpu_interval_sum_seconds": row["gpu_interval_sum_seconds"],
            "gpu_span_seconds": row["gpu_span_seconds"],
        } for row in measured],
        "speedup": {
            "wall_scalar_over_batched": wall_speedup,
            "completed_gpu_interval_scalar_over_batched": gpu_speedup,
        },
    }


def qualify(benchmark: dict[str, Any], log_text: str) -> dict[str, Any]:
    require(isinstance(benchmark, dict), "benchmark JSON must be an object")
    expected = _expected_observations(benchmark)
    scopes, unscoped_commands = _measurement_scopes(log_text, expected)
    span_modes = {scope["gpu_span_seconds"] is not None for scope in scopes}
    require(len(span_modes) == 1, "inconsistent absolute GPU timing mode")
    components = {component: _component_report(scopes, component) for component in COMPONENTS}
    timed_commands = sum(len(scope["commands"]) for scope in scopes)
    return {
        "format": "drama3-prefill-benchmark-analysis-v1",
        "qualified": True,
        "scope": "Descriptive same-process prefill timings from measured scopes; no quiet-host or causal speedup claim.",
        "quiet_host_causal_claim": False,
        "persistent_context": True,
        "rows": benchmark["rows"],
        "chunk": benchmark["chunk"],
        "input_sha256": benchmark.get("input_sha256"),
        "gpu_timing_mode": "absolute_completed_intervals" if True in span_modes else "duration_only_checked_completion",
        "excluded_primes_per_component": PRIME_COUNT,
        "timed_gpu_commands": timed_commands,
        "unscoped_gpu_commands_ignored": unscoped_commands,
        "gpu_timing_evidence": (
            "Rich timeline records require completion_status=0 and a valid interval; "
            "duration-only depth-0 records come from the completion-checked GPU-wait path."),
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path,
                        help="benchmark-chunkN-depthM.json emitted by the producer")
    parser.add_argument("--log", required=True, type=Path,
                        help="combined run log containing measurement markers and GPU commands")
    parser.add_argument("--output", type=Path,
                        help="optional new JSON report path (must not already exist)")
    args = parser.parse_args()
    require(not args.output or not args.output.exists(), "output already exists")
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    report = qualify(benchmark, args.log.read_text(encoding="utf-8"))
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
