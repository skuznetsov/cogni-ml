#!/usr/bin/env python3
"""Fail-closed qualification for paired DRAMA-3 dense projection timings."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


PROJECTIONS = ("qkv", "z", "b", "a")
SHAPES = {"qkv": (8192, 2048), "z": (4096, 2048),
          "b": (32, 2048), "a": (32, 2048)}


def events(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON event at {path}:{number}") from exc
        require(isinstance(value, dict), f"non-object JSON event at {path}:{number}")
        rows.append(value)
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: object, message: str, minimum: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and
            (minimum is None or value >= minimum), message)
    return int(value)


def finite_positive(value: object, message: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value > 0, message)
    return float(value)


def qualify(stdout: list[dict], stderr: list[dict]) -> dict:
    summaries = [row for row in stdout if row.get("event") == "dense_benchmark_summary"]
    require(len(summaries) == 1, "missing or duplicate successful summary")
    summary = summaries[0]
    invocation_id = summary.get("invocation_id")
    require(isinstance(invocation_id, str) and bool(invocation_id), "missing invocation identity")
    invocations = [[row.get("id") for row in stream
                    if row.get("event") == "dense_benchmark_invocation"]
                   for stream in (stdout, stderr)]
    require(invocations == [[invocation_id], [invocation_id]],
            "missing or mismatched invocation identity")

    projections = summary.get("projections")
    require(projections == list(PROJECTIONS), "unexpected projection list")
    shapes = summary.get("shapes")
    require(isinstance(shapes, dict) and set(shapes) == set(PROJECTIONS) and
            all(isinstance(shapes[name], list) and
                all(isinstance(dimension, int) and not isinstance(dimension, bool)
                    for dimension in shapes[name]) and tuple(shapes[name]) == SHAPES[name]
                for name in PROJECTIONS),
            "unexpected projection shapes")
    pairs, warmups = summary.get("pairs_per_projection"), summary.get("warmups_per_projection")
    integer(pairs, "invalid repeat count", 1)
    integer(warmups, "invalid repeat count", 1)
    require(pairs <= 15 and warmups == 2, "invalid repeat count")
    repeats = summary.get("repeats")
    require(isinstance(repeats, dict) and set(repeats) == set(PROJECTIONS), "missing repeat geometry")
    for name in PROJECTIONS:
        integer(repeats.get(name), "invalid repeat geometry", 1)
    require(repeats["qkv"] == repeats["z"] == summary.get("large_repeats") and
            repeats["b"] == repeats["a"] == summary.get("tiny_repeats"),
            "repeat geometry does not match CLI settings")
    integer(summary.get("large_repeats"), "invalid large repeat count", 1)
    integer(summary.get("tiny_repeats"), "invalid tiny repeat count", 1)
    require(summary["large_repeats"] <= 8 and summary["tiny_repeats"] <= 4096,
            "repeat count exceeds benchmark bound")
    require(summary.get("weight_dtypes") == {name: "BF16" for name in PROJECTIONS},
            "expected dense BF16 checkpoint weights")
    require(summary.get("candidate_mode") in ("tree", "simd"), "invalid candidate mode")
    require(isinstance(summary.get("model"), str) and bool(summary["model"]), "missing model path")
    require(isinstance(summary.get("input_id"), str) and bool(summary["input_id"]), "missing input identity")
    require(isinstance(summary.get("input_source"), str) and bool(summary["input_source"]), "missing input source")
    integer(summary.get("input_row"), "invalid input row", 0)

    schedule = []
    for projection in PROJECTIONS:
        schedule.extend((projection, True, index) for index in range(warmups))
        schedule.extend((projection, False, index) for index in range(pairs))
    total_pairs = len(schedule)
    total_commands = total_pairs * 2

    gpu = {}
    for row in stderr:
        if row.get("event") != "drama3_gpu_command":
            continue
        index = integer(row.get("index"), "invalid GPU timeline index", 0)
        require(index not in gpu, "duplicate GPU timeline index")
        gpu[index] = finite_positive(row.get("gpu_seconds"), "invalid GPU timestamp")

    group_rows = [row for row in stdout if row.get("event") == "dense_benchmark_arm"]
    grouped = {}
    consumed_indices = set()
    for row in group_rows:
        key = (row.get("projection"), row.get("warmup"), row.get("pair"))
        require(isinstance(row.get("warmup"), bool), "invalid benchmark warmup flag")
        integer(row.get("pair"), "invalid benchmark pair index", 0)
        arms = grouped.setdefault(key, {})
        arm = row.get("arm")
        require(arm in ("baseline", "candidate") and arm not in arms,
                "duplicate or invalid benchmark arm")
        index = row.get("command_index")
        integer(index, "invalid benchmark command index", 0)
        require(index in gpu and index not in consumed_indices,
                "missing or reused GPU timing")
        projection = row.get("projection")
        require(projection in PROJECTIONS, "unknown projection")
        shape = row.get("shape")
        require(isinstance(shape, list) and all(isinstance(dimension, int) and not isinstance(dimension, bool)
                                                for dimension in shape) and tuple(shape) == SHAPES[projection],
                "incorrect projection shape")
        integer(row.get("repetitions"), "incorrect projection repeat count", 1)
        require(row["repetitions"] == repeats[projection], "incorrect projection repeat count")
        require(row.get("weight_dtype") == "BF16", "incorrect projection weight dtype")
        require(row.get("input_id") == summary["input_id"], "arm used a different input fixture")
        consumed_indices.add(index)
        arms[arm] = {**row, "gpu_seconds": gpu[index]}
    require(consumed_indices == set(gpu), "unassigned GPU command")
    require(set(gpu) == set(range(total_commands)), "non-contiguous GPU timeline")
    require(set(grouped) == set(schedule), "incomplete projection schedule")

    pair_rows = [row for row in stdout if row.get("event") == "dense_benchmark_pair"]
    seen = set()
    projection_samples = {name: [] for name in PROJECTIONS}
    pair_position = {key: position for position, key in enumerate(schedule)}
    verified_elements = 0
    expected_terminal_bytes = 0
    for row in pair_rows:
        key = (row.get("projection"), row.get("warmup"), row.get("pair"))
        require(isinstance(row.get("warmup"), bool), "invalid pair warmup flag")
        integer(row.get("pair"), "invalid pair index", 0)
        require(key in pair_position and key not in seen, "invalid or duplicate projection pair")
        seen.add(key)
        projection = key[0]
        arms = grouped[key]
        require(set(arms) == {"baseline", "candidate"}, "incomplete projection pair")
        ordered = sorted(arms, key=lambda name: arms[name]["command_index"])
        expected_order = ["baseline", "candidate"] if key[2] % 2 == 0 else ["candidate", "baseline"]
        require(ordered == row.get("order") == expected_order, "incorrect AB/BA order")
        position = pair_position[key]
        require(sorted(arms[name]["command_index"] for name in arms) ==
                [position * 2, position * 2 + 1], "incorrect command order")
        for field in ("waits_before_drain", "readbacks_before_drain", "submissions", "final_drain_waits",
                      "verified_elements", "terminal_readback_bytes"):
            integer(row.get(field), f"invalid paired counter: {field}", 0)
        require(row.get("waits_before_drain") == 0 and row.get("readbacks_before_drain") == 0,
                "intermediate synchronization or readback")
        require(row.get("submissions") == 2 and row.get("final_drain_waits") == 2,
                "incorrect paired completion boundary")
        require(row.get("exact_bitwise") is True, "bitwise output comparison omitted")
        expected_elements = repeats[projection] * SHAPES[projection][0]
        require(row.get("verified_elements") == expected_elements,
                "incorrect verified output count")
        expected_bytes = 2 * expected_elements * 4
        require(row.get("terminal_readback_bytes") == expected_bytes,
                "unexpected post-drain terminal readback")
        verified_elements += expected_elements
        expected_terminal_bytes += expected_bytes
        finite_positive(row.get("wall_seconds"), "invalid pair wall time")
        graph = row.get("graph")
        completed = (position + 1) * 2
        graph_fields = ("depth", "submitted", "completed", "pending", "max_pending", "completion_waits", "retained_bytes")
        require(isinstance(graph, dict) and all(field in graph for field in graph_fields), "missing paired graph counters")
        for field in graph_fields:
            integer(graph[field], f"invalid paired graph counter: {field}", 0)
        require(graph.get("depth") == 2 and graph.get("max_pending") == 2 and
                graph.get("submitted") == completed and graph.get("completed") == completed and
                graph.get("completion_waits") == completed and graph.get("pending") == 0 and
                graph.get("retained_bytes") == 0,
                "graph did not drain exactly once per command")
        if not key[1]:
            baseline = arms["baseline"]["gpu_seconds"]
            candidate = arms["candidate"]["gpu_seconds"]
            projection_samples[projection].append({
                "pair": key[2], "order": ordered,
                "baseline_gpu_seconds": baseline,
                "candidate_gpu_seconds": candidate,
                "candidate_speedup": baseline / candidate,
                "wall_seconds": row["wall_seconds"],
            })
    require(seen == set(schedule), "missing pair verification")

    final = summary.get("graph")
    graph_fields = ("depth", "submitted", "completed", "pending", "max_pending", "completion_waits", "retained_bytes")
    require(isinstance(final, dict) and all(field in final for field in graph_fields), "missing final graph counters")
    for field in graph_fields:
        integer(final[field], f"invalid final graph counter: {field}", 0)
    require(final.get("depth") == 2 and
            final.get("submitted") == final.get("completed") == final.get("completion_waits") == total_commands and
            final.get("max_pending") == 2 and final.get("pending") == 0 and
            final.get("retained_bytes") == 0, "unexpected final graph counters")
    readbacks = summary.get("readbacks")
    require(isinstance(readbacks, dict) and set(readbacks) == {"control", "diagnostic", "terminal"},
            "missing final readback counters")
    for name, value in readbacks.items():
        integer(value, f"invalid final readback counter: {name}", 0)
    require(readbacks == {"control": 0, "diagnostic": 0, "terminal": expected_terminal_bytes},
            "unexpected final readback counters")
    require(summary.get("verified_elements") == verified_elements,
            "unexpected final verified output count")

    costs = {}
    for projection, rows in projection_samples.items():
        require(len(rows) == pairs, "missing measured projection pairs")
        costs[projection] = {"pairs": len(rows), "samples": rows}
        for arm in ("baseline", "candidate"):
            values = [row[f"{arm}_gpu_seconds"] for row in rows]
            costs[projection][f"{arm}_gpu_seconds"] = {
                "median": statistics.median(values), "min": min(values), "max": max(values)}
        ratios = [row["candidate_speedup"] for row in rows]
        costs[projection]["candidate_speedup"] = {
            "median": statistics.median(ratios), "min": min(ratios), "max": max(ratios)}
    return {
        "format": "drama3-dense-cost-v1", "qualified": True,
        "invocation_id": invocation_id,
        "scope": "Paired GPU time for dense BF16 projection kernels with F32 accumulation; exactness is bitwise against the existing tree kernel.",
        "model": summary["model"], "input_id": summary["input_id"],
        "candidate_mode": summary["candidate_mode"],
        "pairs_per_projection": pairs, "warmups_per_projection": warmups,
        "repeats": repeats, "verified_elements": verified_elements,
        "timed_command_buffers": total_commands,
        "no_wait_or_readback_before_pair_drain": True,
        "projections": costs, "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    report = qualify(events(args.stdout), events(args.stderr))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Qualified {report['timed_command_buffers']} commands / {report['verified_elements']} verified elements")


if __name__ == "__main__":
    main()
