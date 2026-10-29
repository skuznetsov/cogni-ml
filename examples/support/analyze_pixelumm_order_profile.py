#!/usr/bin/env python3
"""CPU-only validator and phase-localization summary for two PixelUMM profile runs.

The analyzer checks the report's source-level schedule, saved BF16 capture
comparisons, one-wait boundary, native root intervals, block phase markers,
MPS API accounting, continuation trace, and existing Mach-time memory events.
It does not validate external runner receipts or make a speedup claim. Trace
parent checks are syntactic because the exported trace omits some graph scopes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPORT = "cogni_ml.pixelumm_native_combined_attention_profile/v1"
PROFILE = "cogni_ml.pixelumm_decoder_phase_profile/v1"
SUMMARY = "cogni_ml.pixelumm_order_analysis/v1"
BLOCKS = 36
TILES_PER_BLOCK = 60
GRID = [6, 44, 80]
TOKENS = math.prod(GRID)
HIDDEN = 4096
FFN = 12288
ATOL = 2e-5
RTOL = 2e-5
MARKERS = ("input_norm", "qkv_projection", "qk_norm_rope", "attention",
           "output_projection", "residual_post_norm", "ffn_gate_up",
           "ffn_activation", "ffn_down", "final_residual", "complete")
PHASES = MARKERS[:-1]
API_PHASES = {
    0: "unassigned",
    1: "self_prepare",
    2: "self_attention",
    3: "self_projection_residual",
    4: "cross_attention_projection_residual",
    5: "ffn",
}
PIXELUMM_API_PHASES = {
    0: "unassigned_head",
    1: "self_prepare",
    2: "local_attention",
    3: "projection_residual",
    4: "unused_cross_attention",
    5: "ffn",
}
OPS = ("graph_encode", "commit_and_continue", "t5_linear_encode")
API_FIELDS = ("count", "wall_total_ns", "wall_max_ns", "thread_cpu_total_ns",
              "thread_cpu_max_ns", "nested_count", "nested_wall_total_ns",
              "nested_thread_cpu_total_ns", "exclusive_wall_total_ns",
              "exclusive_thread_cpu_total_ns")
TRACE_INFO = ("owner_thread_id", "mach_timebase_numer", "mach_timebase_denom",
              "trace_start_mach_ticks", "event_capacity", "event_count", "root_count", "failed")
TRACE_EVENT = ("serial", "begin_mach_ticks", "end_mach_ticks", "begin_thread_cpu_ns",
               "end_thread_cpu_ns", "api_scope_serial", "parent_graph_serial",
               "before_root_sequence", "after_root_sequence", "phase", "graph_depth",
               "before_root_status", "before_root_status_after_call", "after_root_status", "flags")
TRACE_ROOT = ("sequence", "tracked_or_observed_mach_ticks", "completion_observed_mach_ticks",
              "gpu_start_time_bits", "gpu_end_time_bits", "status_at_track_or_enable",
              "final_status", "has_error", "completion_source", "gpu_start_seconds",
              "gpu_end_seconds")
ROOT_CLOCK_BOUND_SECONDS = 2e-6
AGGREGATE_WALL_ROUNDING_NS_PER_CALL = 2000.0


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def obj(value: object, where: str) -> dict[str, Any]:
    need(isinstance(value, dict), f"{where}: expected object")
    return value  # type: ignore[return-value]


def arr(value: object, where: str) -> list[Any]:
    need(isinstance(value, list), f"{where}: expected array")
    return value  # type: ignore[return-value]


def exact_keys(value: dict[str, Any], keys: tuple[str, ...] | set[str], where: str) -> None:
    need(set(value) == set(keys), f"{where}: missing or unexpected fields")


def uint(value: object, where: str, minimum: int = 0) -> int:
    need(isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= (1 << 64) - 1,
         f"{where}: expected unsigned integer")
    return int(value)


def finite(value: object, where: str, minimum: float = 0.0) -> float:
    need(isinstance(value, (int, float)) and not isinstance(value, bool), f"{where}: expected number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{where}: invalid number") from exc
    need(math.isfinite(result) and result >= minimum, f"{where}: non-finite or out of range")
    return result


def boolean(value: object, where: str) -> bool:
    need(isinstance(value, bool), f"{where}: expected boolean")
    return value


def digest(value: object, where: str) -> str:
    need(isinstance(value, str) and len(value) == 64 and
         all(char in "0123456789abcdef" for char in value), f"{where}: invalid SHA-256")
    return value


def near(actual: float, expected: float, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    return math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol)


def interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "min": None, "max": None}
    return {"count": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def _expected_capture_order() -> list[str]:
    blocks = [name for i in range(BLOCKS) for name in
              ((f"block{i}.attention",) if i < 34 else
               (f"block{i}.attention", f"block{i}.output"))]
    return blocks + [f"head.{name}" for name in
                     ("final_norm_tile_major", "restored_thw", "head_norm_thw", "linear_thw", "output_thw")]


def _capture_elements(name: str) -> int:
    if name == "head.output_thw" or name == "head.linear_thw":
        return TOKENS * 3072
    return TOKENS * HIDDEN


def validate_comparisons(raw: object, request_index: int) -> list[dict[str, Any]]:
    comparisons = arr(raw, f"request {request_index} comparisons")
    expected_order = _expected_capture_order()
    need(len(comparisons) == 43, f"request {request_index}: expected all 43 saved captures")
    result = []
    for slot, value in enumerate(comparisons):
        row = obj(value, f"request {request_index} comparison {slot}")
        exact_keys(row, ("stage", "elements", "failures", "different_words", "max_abs", "finite",
                         "actual_sha256", "first_failure_index", "first_expected", "first_actual"),
                   f"request {request_index} comparison {slot}")
        need(row["stage"] == expected_order[slot], f"request {request_index}: capture order changed at {slot}")
        need(uint(row["elements"], "capture elements") == _capture_elements(expected_order[slot]),
             f"request {request_index}: capture element extent changed at {slot}")
        need(uint(row["failures"], "capture failures") == 0 and boolean(row["finite"], "capture finite"),
             f"request {request_index}: saved-capture comparison failed at {expected_order[slot]}")
        different = uint(row["different_words"], "capture different_words")
        need(different <= row["elements"], f"request {request_index}: different_words exceeds element count")
        finite(row["max_abs"], "capture max_abs")
        digest(row["actual_sha256"], "capture actual digest")
        need(row["first_failure_index"] is None and row["first_expected"] is None and row["first_actual"] is None,
             f"request {request_index}: passing comparison carries a failure witness")
        result.append({"stage": row["stage"], "elements": row["elements"],
                       "different_words": different, "max_abs": float(row["max_abs"])})
    return result


def _root_intervals(root_rows: list[Any], where: str, *, profile: bool) -> tuple[list[tuple[float, float]], list[dict[str, Any]]]:
    intervals: list[tuple[float, float]] = []
    roots: list[dict[str, Any]] = []
    expected_fields = (("sequence", "completed", "has_error", "gpu_start_seconds", "gpu_end_seconds")
                       if not profile else
                       ("sequence", "completed", "has_error", "gpu_start_seconds", "gpu_end_seconds",
                        "duration_seconds", "candidate_phases"))
    for index, raw in enumerate(root_rows):
        row = obj(raw, f"{where} root {index}")
        exact_keys(row, expected_fields, f"{where} root {index}")
        need(uint(row["sequence"], "root sequence") == index and row["completed"] is True and row["has_error"] is False,
             f"{where}: root {index} incomplete or reordered")
        start = finite(row["gpu_start_seconds"], "root start", minimum=0.0)
        end = finite(row["gpu_end_seconds"], "root end", minimum=0.0)
        need(start > 0.0 and end > start, f"{where}: invalid root interval {index}")
        duration = end - start
        if profile:
            need(near(finite(row["duration_seconds"], "root duration"), duration, 2e-9, 1e-9),
                 f"{where}: root duration mismatch at {index}")
            candidates = arr(row["candidate_phases"], "root candidate phases")
            need(all(isinstance(label, str) for label in candidates), f"{where}: malformed root candidates")
            roots.append({"sequence": index, "start_seconds": start, "end_seconds": end,
                          "duration_seconds": duration, "candidate_phases": candidates})
        else:
            roots.append({"sequence": index, "start_seconds": start, "end_seconds": end,
                          "duration_seconds": duration})
        intervals.append((start, end))
    return intervals, roots


def _root_metrics(intervals: list[tuple[float, float]]) -> dict[str, float]:
    need(bool(intervals), "root timing list is empty")
    duration_sum = sum(end - start for start, end in intervals)
    union = interval_union(intervals)
    envelope = max(end for _, end in intervals) - min(start for start, _ in intervals)
    return {"root_duration_seconds": duration_sum, "union_seconds": union,
            "envelope_seconds": envelope, "overlap_seconds": max(0.0, duration_sum - union),
            "gap_seconds": max(0.0, envelope - union)}


def _validate_report_root_profile(raw: object, request_index: int, expected_segments: int) -> tuple[list[tuple[float, float]], list[dict[str, Any]], dict[str, float]]:
    profile = obj(raw, f"request {request_index} root_profile")
    exact_keys(profile, ("root_count", "root_duration_seconds", "envelope_seconds", "union_seconds",
                         "overlap_seconds", "roots"), f"request {request_index} root_profile")
    count = uint(profile["root_count"], "root_profile root_count", minimum=1)
    need(count == expected_segments, f"request {request_index}: command segment/root count mismatch")
    intervals, roots = _root_intervals(arr(profile["roots"], "root_profile roots"),
                                       f"request {request_index} root_profile", profile=False)
    need(len(intervals) == count, f"request {request_index}: root_profile interval count mismatch")
    metrics = _root_metrics(intervals)
    for field in ("root_duration_seconds", "envelope_seconds", "union_seconds", "overlap_seconds"):
        need(near(finite(profile[field], f"root_profile {field}"), metrics[field], 2e-8, 1e-8),
             f"request {request_index}: root_profile {field} does not match reported intervals")
    return intervals, roots, metrics


def validate_trace(raw: object, request_index: int, intervals: list[tuple[float, float]]) -> dict[str, Any]:
    trace = obj(raw, f"request {request_index} continuation_trace")
    exact_keys(trace, ("info", "events", "roots"), f"request {request_index} continuation_trace")
    info = obj(trace["info"], "continuation trace info")
    exact_keys(info, TRACE_INFO, "continuation trace info")
    values = {key: uint(info[key], f"continuation trace {key}") for key in TRACE_INFO}
    need(values["owner_thread_id"] > 0 and values["mach_timebase_numer"] > 0 and
         values["mach_timebase_denom"] > 0 and values["trace_start_mach_ticks"] > 0 and
         values["event_capacity"] == 8192 and values["failed"] == 0,
         f"request {request_index}: invalid continuation trace clock/capacity/failure state")
    events = arr(trace["events"], "continuation trace events")
    roots_raw = arr(trace["roots"], "continuation trace roots")
    need(values["event_count"] == len(events) == len(roots_raw) - 1 and
         values["root_count"] == len(roots_raw) == len(intervals) and
         0 < values["event_count"] <= values["event_capacity"],
         f"request {request_index}: continuation event/root counts do not reconcile")

    def host_seconds(ticks: int) -> float:
        return ticks * values["mach_timebase_numer"] / values["mach_timebase_denom"] / 1e9

    roots: list[dict[str, Any]] = []
    for index, raw_root in enumerate(roots_raw):
        root = obj(raw_root, f"trace root {index}")
        exact_keys(root, TRACE_ROOT, f"trace root {index}")
        ints = {key: uint(root[key], f"trace root {index}.{key}") for key in TRACE_ROOT
                if key not in ("gpu_start_seconds", "gpu_end_seconds")}
        start = finite(root["gpu_start_seconds"], "trace root start", minimum=0.0)
        end = finite(root["gpu_end_seconds"], "trace root end", minimum=0.0)
        try:
            bits_start = struct.unpack("=d", struct.pack("=Q", ints["gpu_start_time_bits"]))[0]
            bits_end = struct.unpack("=d", struct.pack("=Q", ints["gpu_end_time_bits"]))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError(f"trace root {index}: invalid GPU timestamp bits") from exc
        need(ints["sequence"] == index and ints["final_status"] == 4 and ints["has_error"] == 0 and
             ints["completion_source"] in (1, 2) and ints["status_at_track_or_enable"] <= 1 and
             ints["tracked_or_observed_mach_ticks"] >= values["trace_start_mach_ticks"] and
             ints["completion_observed_mach_ticks"] >= ints["tracked_or_observed_mach_ticks"] and
             end > start > 0 and math.isfinite(bits_start) and math.isfinite(bits_end) and
             near(bits_start, start, 1e-12, 0.0) and near(bits_end, end, 1e-12, 0.0) and
             host_seconds(ints["tracked_or_observed_mach_ticks"]) <= start + ROOT_CLOCK_BOUND_SECONDS and
             end <= host_seconds(ints["completion_observed_mach_ticks"]) + ROOT_CLOCK_BOUND_SECONDS,
             f"request {request_index}: invalid trace root time/status at {index}")
        need(near(start, intervals[index][0], ROOT_CLOCK_BOUND_SECONDS, 0.0) and
             near(end, intervals[index][1], ROOT_CLOCK_BOUND_SECONDS, 0.0),
             f"request {request_index}: trace/root_profile interval differs at {index}")
        roots.append({**ints, "start_seconds": start, "end_seconds": end,
                      "duration_seconds": end - start})

    previous_end = values["trace_start_mach_ticks"]
    previous_scope = 0
    phase_counts: Counter[int] = Counter()
    parsed_events = []
    for index, raw_event in enumerate(events):
        event = obj(raw_event, f"trace event {index}")
        exact_keys(event, TRACE_EVENT, f"trace event {index}")
        parsed = {key: uint(event[key], f"trace event {index}.{key}") for key in TRACE_EVENT}
        nested = parsed["graph_depth"] > 0
        need(parsed["serial"] == index + 1 and parsed["before_root_sequence"] == index and
             parsed["after_root_sequence"] == index + 1 and parsed["api_scope_serial"] > previous_scope and
             parsed["begin_mach_ticks"] >= previous_end and parsed["end_mach_ticks"] > parsed["begin_mach_ticks"] and
             parsed["begin_thread_cpu_ns"] > 0 and parsed["end_thread_cpu_ns"] >= parsed["begin_thread_cpu_ns"] and
             parsed["phase"] in API_PHASES and
             parsed["graph_depth"] <= 16 and parsed["flags"] == (15 if nested else 14) and
             ((0 < parsed["parent_graph_serial"] < parsed["api_scope_serial"]) if nested
              else parsed["parent_graph_serial"] == 0) and
             parsed["before_root_status"] <= 1 and 2 <= parsed["before_root_status_after_call"] <= 4 and
             parsed["after_root_status"] <= 1 and
             roots[index]["tracked_or_observed_mach_ticks"] <= parsed["begin_mach_ticks"] and
             roots[index + 1]["tracked_or_observed_mach_ticks"] >= parsed["end_mach_ticks"] and
             roots[index]["start_seconds"] + ROOT_CLOCK_BOUND_SECONDS >= host_seconds(parsed["begin_mach_ticks"]),
             f"request {request_index}: invalid continuation event {index}")
        parsed["wall_seconds"] = host_seconds(parsed["end_mach_ticks"]) - host_seconds(parsed["begin_mach_ticks"])
        parsed["thread_cpu_seconds"] = (parsed["end_thread_cpu_ns"] - parsed["begin_thread_cpu_ns"]) / 1e9
        phase_counts[parsed["phase"]] += 1
        parsed_events.append(parsed)
        previous_end = parsed["end_mach_ticks"]
        previous_scope = parsed["api_scope_serial"]
    return {"info": values, "events": parsed_events, "roots": roots,
            "phase_counts": phase_counts, "host_seconds": host_seconds}


def _validate_api_operation_stats(stats: dict[str, int], operation: str, request_index: int) -> None:
    if stats["count"] == 0:
        need(all(value == 0 for value in stats.values()),
             f"request {request_index}: empty API operation has times")
    else:
        need(0 < stats["wall_max_ns"] <= stats["wall_total_ns"] and
             0 < stats["thread_cpu_max_ns"] <= stats["thread_cpu_total_ns"] and
             stats["wall_max_ns"] * stats["count"] >= stats["wall_total_ns"] and
             stats["thread_cpu_max_ns"] * stats["count"] >= stats["thread_cpu_total_ns"] and
             (operation != "commit_and_continue" or stats["nested_count"] <= stats["count"]) and
             stats["nested_wall_total_ns"] <= stats["wall_total_ns"] and
             stats["nested_thread_cpu_total_ns"] <= stats["thread_cpu_total_ns"] and
             ((stats["nested_count"] > 0) == (stats["nested_wall_total_ns"] > 0) ==
              (stats["nested_thread_cpu_total_ns"] > 0)),
             f"request {request_index}: inconsistent {operation} API totals")
    if operation == "graph_encode":
        need(stats["exclusive_wall_total_ns"] == stats["wall_total_ns"] - stats["nested_wall_total_ns"] and
             stats["exclusive_thread_cpu_total_ns"] == stats["thread_cpu_total_ns"] - stats["nested_thread_cpu_total_ns"],
             f"request {request_index}: graph inclusive/exclusive accounting differs")
    elif operation == "commit_and_continue":
        need(stats["exclusive_wall_total_ns"] == stats["wall_total_ns"] and
             stats["exclusive_thread_cpu_total_ns"] == stats["thread_cpu_total_ns"],
             f"request {request_index}: commit inclusive accounting differs")
    else:
        need(stats["nested_count"] == stats["nested_wall_total_ns"] == stats["nested_thread_cpu_total_ns"] == 0 and
             stats["exclusive_wall_total_ns"] == stats["wall_total_ns"] and
             stats["exclusive_thread_cpu_total_ns"] == stats["thread_cpu_total_ns"],
             f"request {request_index}: unexpected linear API nesting")


def _validate_api(profile: dict[str, Any], trace: dict[str, Any], request_index: int) -> dict[int, dict[str, dict[str, int]]]:
    api = obj(profile.get("api_profile"), f"request {request_index}: missing API profile")
    exact_keys(api, ("source_schema", "phases"), "API profile")
    need(api["source_schema"] == "MPSAPIProfile WAN phase IDs", "API profile source schema changed")
    raw_phases = arr(api["phases"], "API phases")
    need(len(raw_phases) == 6, f"request {request_index}: expected six API phase records")
    indexed: dict[int, dict[str, dict[str, int]]] = {}
    for raw_phase in raw_phases:
        phase = obj(raw_phase, "API phase")
        exact_keys(phase, ("pixelumm_phase", "source_phase_id", "source_phase_name", "operations"), "API phase")
        phase_id = uint(phase["source_phase_id"], "API phase id")
        need(phase_id in API_PHASES and phase_id not in indexed and
             phase["source_phase_name"] == API_PHASES[phase_id] and
             phase["pixelumm_phase"] == PIXELUMM_API_PHASES[phase_id], "API phase identity/name mismatch")
        operations = obj(phase["operations"], "API operations")
        exact_keys(operations, OPS, "API operations")
        parsed_ops: dict[str, dict[str, int]] = {}
        for operation in OPS:
            raw_stats = obj(operations[operation], f"API {operation}")
            exact_keys(raw_stats, API_FIELDS, f"API {operation}")
            stats = {key: uint(raw_stats[key], f"API {operation}.{key}") for key in API_FIELDS}
            _validate_api_operation_stats(stats, operation, request_index)
            parsed_ops[operation] = stats
        graph, commit = parsed_ops["graph_encode"], parsed_ops["commit_and_continue"]
        need(tuple(graph[key] for key in ("nested_count", "nested_wall_total_ns", "nested_thread_cpu_total_ns")) ==
             tuple(commit[key] for key in ("nested_count", "nested_wall_total_ns", "nested_thread_cpu_total_ns")),
             f"request {request_index}: graph/continuation nested totals disagree")
        indexed[phase_id] = parsed_ops

    need(set(indexed) == set(API_PHASES), f"request {request_index}: API phase matrix is incomplete")
    expected_graphs = {0: 5, 1: 9 * BLOCKS, 2: 3 * TILES_PER_BLOCK * BLOCKS,
                       3: 3 * BLOCKS, 4: 0, 5: 3 * BLOCKS}
    need({phase_id: indexed[phase_id]["graph_encode"]["count"] for phase_id in API_PHASES} == expected_graphs,
         f"request {request_index}: Graph call counts do not match 36x60 separate-gate/up geometry")
    need(sum(expected_graphs.values()) == 7025 and
         all(indexed[phase_id]["t5_linear_encode"]["count"] == 0 for phase_id in API_PHASES),
         f"request {request_index}: unexpected Graph total or T5 linear operation")
    for phase_id in API_PHASES:
        calls = [event for event in trace["events"] if event["phase"] == phase_id]
        commit = indexed[phase_id]["commit_and_continue"]
        need(commit["count"] == len(calls) and
             commit["nested_count"] == sum(event["graph_depth"] > 0 for event in calls),
             f"request {request_index}: continuation count/nesting differs from trace phase {phase_id}")
        wall_ns = sum(event["wall_seconds"] * 1e9 for event in calls)
        cpu_ns = sum(event["end_thread_cpu_ns"] - event["begin_thread_cpu_ns"] for event in calls)
        need(wall_ns <= commit["wall_total_ns"] + len(calls) * AGGREGATE_WALL_ROUNDING_NS_PER_CALL and
             cpu_ns <= commit["thread_cpu_total_ns"],
             f"request {request_index}: raw continuation spans exceed aggregate API phase {phase_id}")
    return indexed


def _validate_phase_profile(raw: object, request_index: int, enabled: bool,
                            root_intervals: list[tuple[float, float]], trace: dict[str, Any]) -> dict[str, Any]:
    profile = obj(raw, f"request {request_index} phase_profile")
    exact_keys(profile, ("schema", "block_count", "ffn_graph_topology", "fused_kv_gather", "scale_softmax",
                         "blocks", "aggregate_phase_wall_seconds", "aggregate_phase_thread_cpu_seconds",
                         "head_root_range", "root_candidates", "phase_candidate_root_duration_sum_seconds",
                         "phase_definite_root_duration_sum_seconds", "phase_root_duration_sum_semantics",
                         "root_profile", "roots", "api_profile"), f"request {request_index} phase_profile")
    need(profile["schema"] == PROFILE and uint(profile["block_count"], "profile block_count") == BLOCKS and
         profile["ffn_graph_topology"] == "separate_gate_up" and
         boolean(profile["fused_kv_gather"], "fused_kv_gather") is enabled and
         boolean(profile["scale_softmax"], "scale_softmax") is enabled,
         f"request {request_index}: profile schema/topology differs from scheduled mode")
    need(profile["phase_root_duration_sum_semantics"] ==
         "Candidate root-duration sums are non-additive: shared roots may appear in multiple phase candidates and root GPU intervals may overlap; they are not per-phase GPU work.",
         f"request {request_index}: root candidate semantics missing")

    block_rows = arr(profile["blocks"], "phase profile blocks")
    need(len(block_rows) == BLOCKS, f"request {request_index}: phase profile needs all 36 blocks")
    expected_wall = {phase: 0.0 for phase in PHASES}
    expected_cpu = {phase: 0.0 for phase in PHASES}
    expected_candidates: list[list[str]] = [[] for _ in root_intervals]
    expected_per_block: list[dict[str, Any]] = []
    previous_end_count = 1
    for slot, raw_block in enumerate(block_rows):
        block = obj(raw_block, f"phase block {slot}")
        exact_keys(block, ("slot", "attention_tile_count", "markers", "phase_wall_seconds", "phase_thread_cpu_seconds"),
                   f"phase block {slot}")
        need(uint(block["slot"], "phase block slot") == slot and
             uint(block["attention_tile_count"], "phase tile count") == TILES_PER_BLOCK,
             f"request {request_index}: block/tile topology changed at block {slot}")
        marks_raw = arr(block["markers"], f"phase block {slot} markers")
        need(len(marks_raw) == len(MARKERS), f"request {request_index}: block {slot} must have 11 markers")
        marks = []
        last_wall = last_cpu = 0.0
        last_root = previous_end_count
        for mark_index, raw_mark in enumerate(marks_raw):
            marker = obj(raw_mark, f"phase block {slot} marker {mark_index}")
            exact_keys(marker, ("name", "elapsed_seconds", "thread_cpu_seconds", "root_count"), "phase marker")
            wall = finite(marker["elapsed_seconds"], "marker wall")
            cpu = finite(marker["thread_cpu_seconds"], "marker thread cpu")
            cursor = uint(marker["root_count"], "marker root count", minimum=1)
            need(marker["name"] == MARKERS[mark_index] and wall >= last_wall and cpu >= last_cpu and
                 last_root <= cursor <= len(root_intervals),
                 f"request {request_index}: marker order/clock/root cursor invalid at block {slot} marker {mark_index}")
            if mark_index == 0:
                need(wall == 0.0 and cpu == 0.0 and cursor == previous_end_count,
                     f"request {request_index}: block {slot} does not restart at zero/adjacent root cursor")
            marks.append({"name": marker["name"], "elapsed_seconds": wall,
                          "thread_cpu_seconds": cpu, "root_count": cursor})
            last_wall, last_cpu, last_root = wall, cpu, cursor
        actual_wall = obj(block["phase_wall_seconds"], "block phase wall")
        actual_cpu = obj(block["phase_thread_cpu_seconds"], "block phase cpu")
        exact_keys(actual_wall, PHASES, f"block {slot} phase wall")
        exact_keys(actual_cpu, PHASES, f"block {slot} phase cpu")
        block_wall: dict[str, float] = {}
        block_cpu: dict[str, float] = {}
        for phase_index, phase in enumerate(PHASES):
            start, end = marks[phase_index], marks[phase_index + 1]
            wall_delta = end["elapsed_seconds"] - start["elapsed_seconds"]
            cpu_delta = end["thread_cpu_seconds"] - start["thread_cpu_seconds"]
            need(near(finite(actual_wall[phase], "reported block wall"), wall_delta) and
                 near(finite(actual_cpu[phase], "reported block cpu"), cpu_delta),
                 f"request {request_index}: reported per-block {phase} elapsed does not match markers")
            block_wall[phase] = wall_delta
            block_cpu[phase] = cpu_delta
            expected_wall[phase] += wall_delta
            expected_cpu[phase] += cpu_delta
            label = f"block_{slot}/{phase}"
            for root_index in range(start["root_count"] - 1, end["root_count"]):
                expected_candidates[root_index].append(label)
        previous_end_count = marks[-1]["root_count"]
        expected_per_block.append({"slot": slot, "attention_tile_count": TILES_PER_BLOCK,
                                   "phase_wall_seconds": block_wall, "phase_thread_cpu_seconds": block_cpu})

    head = obj(profile["head_root_range"], "head root range")
    exact_keys(head, ("start_root_count", "end_root_count", "candidate_root_indices_inclusive"), "head root range")
    head_start = uint(head["start_root_count"], "head root start", minimum=1)
    head_end = uint(head["end_root_count"], "head root end", minimum=1)
    need(head_start == previous_end_count and head_end == len(root_intervals) and head_end >= head_start and
         head["candidate_root_indices_inclusive"] == [head_start - 1, head_end - 1],
         f"request {request_index}: head root range does not continue block 35 through the terminal root")
    for root_index in range(head_start - 1, head_end):
        expected_candidates[root_index].append("head")
    need(all(expected_candidates), f"request {request_index}: phase profile leaves a command root unattributed")

    for field, expected in (("aggregate_phase_wall_seconds", expected_wall),
                            ("aggregate_phase_thread_cpu_seconds", expected_cpu)):
        values = obj(profile[field], f"profile {field}")
        exact_keys(values, PHASES, field)
        for phase, want in expected.items():
            need(near(finite(values[phase], f"{field}.{phase}"), want),
                 f"request {request_index}: aggregate {field}.{phase} disagrees with 36 marker pairs")

    candidate_maps: dict[str, float] = {}
    definite_maps: dict[str, float] = {}
    labels = [f"block_{slot}/{phase}" for slot in range(BLOCKS) for phase in PHASES] + ["head"]
    candidate_maps = dict.fromkeys(labels, 0.0)
    definite_maps = dict.fromkeys(labels, 0.0)
    roots_raw = arr(profile["roots"], "phase profile roots")
    need(len(roots_raw) == len(root_intervals), f"request {request_index}: phase root count mismatch")
    roots = []
    for index, raw_root in enumerate(roots_raw):
        root = obj(raw_root, f"phase root {index}")
        exact_keys(root, ("sequence", "completed", "has_error", "gpu_start_seconds", "gpu_end_seconds",
                          "duration_seconds", "candidate_phases"), f"phase root {index}")
        need(uint(root["sequence"], "phase root sequence") == index and root["completed"] is True and
             root["has_error"] is False, f"request {request_index}: phase root incomplete at {index}")
        start = finite(root["gpu_start_seconds"], "phase root start", minimum=0.0)
        end = finite(root["gpu_end_seconds"], "phase root end", minimum=0.0)
        duration = end - start
        candidates = arr(root["candidate_phases"], "phase root candidates")
        need(near(start, root_intervals[index][0], ROOT_CLOCK_BOUND_SECONDS, 0.0) and
             near(end, root_intervals[index][1], ROOT_CLOCK_BOUND_SECONDS, 0.0) and duration > 0.0 and
             near(finite(root["duration_seconds"], "phase root duration"), duration, 2e-9, 1e-9) and
             candidates == expected_candidates[index],
             f"request {request_index}: actual root interval/candidates differ at root {index}")
        for label in candidates:
            need(label in candidate_maps, f"request {request_index}: unknown phase root label")
            candidate_maps[label] += duration
        if len(candidates) == 1:
            definite_maps[candidates[0]] += duration
        roots.append({"sequence": index, "start_seconds": start, "end_seconds": end,
                      "duration_seconds": duration, "candidate_phases": candidates})

    need(arr(profile["root_candidates"], "profile root candidates") == expected_candidates,
         f"request {request_index}: top-level phase candidate table disagrees with marker/root intervals")
    for field, expected in (("phase_candidate_root_duration_sum_seconds", candidate_maps),
                            ("phase_definite_root_duration_sum_seconds", definite_maps)):
        values = obj(profile[field], field)
        exact_keys(values, set(labels), field)
        for label, want in expected.items():
            need(near(finite(values[label], f"{field}.{label}"), want, 2e-8, 1e-8),
                 f"request {request_index}: {field} differs at {label}")

    profile_root = obj(profile["root_profile"], "nested profile root metrics")
    exact_keys(profile_root, ("root_count", "root_duration_seconds", "envelope_seconds", "union_seconds",
                              "overlap_seconds", "phase_union_seconds"), "nested profile root metrics")
    root_metrics = _root_metrics(root_intervals)
    need(uint(profile_root["root_count"], "nested root count") == len(root_intervals),
         f"request {request_index}: nested profile root count mismatch")
    for field in ("root_duration_seconds", "envelope_seconds", "union_seconds", "overlap_seconds"):
        need(near(finite(profile_root[field], f"nested root {field}"), root_metrics[field], 2e-8, 1e-8),
             f"request {request_index}: nested profile {field} mismatch")
    phase_union = obj(profile_root["phase_union_seconds"], "nested phase union")
    exact_keys(phase_union, ("whole_chain",), "nested phase union")
    need(near(finite(phase_union["whole_chain"], "whole chain union"), root_metrics["union_seconds"], 2e-8, 1e-8),
         f"request {request_index}: nested whole-chain union mismatch")
    api = _validate_api(profile, trace, request_index)
    need(profile["phase_root_duration_sum_semantics"] ==
         "Candidate root-duration sums are non-additive: shared roots may appear in multiple phase candidates and root GPU intervals may overlap; they are not per-phase GPU work.",
         f"request {request_index}: phase root semantic warning changed")
    return {"phase_wall_seconds": expected_wall, "phase_thread_cpu_seconds": expected_cpu,
            "per_block": expected_per_block, "roots": roots, "root_metrics": root_metrics,
            "candidate_root_duration_sums": candidate_maps,
            "definite_root_duration_sums": definite_maps, "api": api}


def _validate_request(row: object, index: int, expected: dict[str, Any]) -> dict[str, Any]:
    request = obj(row, f"request {index}")
    need(uint(request.get("request"), "request index") == index and request.get("kind") == expected["kind"],
         f"request {index}: index or schedule kind changed")
    for key in ("profiling", "combined_attention", "shared_scratch_poisoned_before_request"):
        value = boolean(request.get(key), f"request {index} {key}")
        schedule_key = {"profiling": "profile",
                        "shared_scratch_poisoned_before_request": "poisoned"}.get(key, key)
        expected_value = expected[schedule_key]
        need(value is expected_value, f"request {index}: {key} differs from profile schedule")
    for field, expected_count in (("terminal_wait_count", 1), ("named_allocations_during_encode", 0),
                                  ("activation_readbacks_before_terminal_wait", 0),
                                  ("diagnostic_readbacks_after_terminal_wait", 43),
                                  ("fused_kv_gather_block_count", 36 if expected["combined_attention"] else 0),
                                  ("scale_softmax_block_count", 36 if expected["combined_attention"] else 0),
                                  ("scale_softmax_prepared_block_count", 36),
                                  ("shared_gate_up_block_count", 0),
                                  ("shared_gate_up_prepared_block_count", 0)):
        need(uint(request.get(field), f"request {index} {field}") == expected_count,
             f"request {index}: {field} must equal {expected_count}")
    need(request.get("head_input_is_block35_output") is True and request.get("passed") is True,
         f"request {index}: full chain edge or numerical request failed")
    need(request.get("early_attention_change_rejected") is (index == 0),
         f"request {index}: in-flight attention selector test changed")
    need(request.get("head_before_blocks_rejected") is (index == 0),
         f"request {index}: head-before-block guard test changed")
    need(request.get("early_mode_change_rejected") is False,
         f"request {index}: unrelated FFN mode rejection changed")
    need(request.get("unsupported_profile_rejected_without_submission") is (index == 0),
         f"request {index}: empty-chain profile rejection test changed")
    for field in ("early_release_rejected", "early_read_rejected", "early_reuse_rejected", "early_poison_rejected"):
        need(request.get(field) is (index == 0), f"request {index}: lifecycle guard {field} changed")
    need(request.get("early_profile_rejected") is expected["profile"],
         f"request {index}: pre-wait profile access guard changed")
    for field in ("blocks_encode_seconds", "head_encode_seconds", "encode_seconds",
                  "observed_encode_span_seconds", "terminal_wait_seconds", "total_seconds", "gpu_elapsed_seconds"):
        finite(request.get(field), f"request {index} {field}")
    need(near(float(request["encode_seconds"]), float(request["blocks_encode_seconds"]) +
              float(request["head_encode_seconds"]), 1e-9, 1e-8),
         f"request {index}: encode_seconds does not equal block plus head encode")
    need(float(request["terminal_wait_seconds"]) <= float(request["total_seconds"]) + 1e-6,
         f"request {index}: terminal wait exceeds measured execution envelope")
    input_hash = digest(request.get("input_sha256"), f"request {index} input")
    comparisons = validate_comparisons(request.get("comparisons"), index)
    block_encode_spans = arr(request.get("per_block_encode_host_seconds"),
                             f"request {index} per-block encode spans")
    need(len(block_encode_spans) == BLOCKS,
         f"request {index}: expected 36 per-block encode host spans")
    block_encode_spans = [finite(value, f"request {index} block encode span")
                          for value in block_encode_spans]
    segment_count = uint(request.get("command_segments"), f"request {index} command segments", minimum=1)
    root_intervals, root_rows, root_metrics = _validate_report_root_profile(
        request.get("root_profile"), index, segment_count)
    need(near(float(request["gpu_elapsed_seconds"]), root_metrics["root_duration_seconds"], 2e-6, 1e-8),
         f"request {index}: gpu_elapsed_seconds differs from actual root durations")

    if expected["profile"]:
        trace = validate_trace(request.get("continuation_trace"), index, root_intervals)
        phase = _validate_phase_profile(request.get("phase_profile"), index,
                                        expected["combined_attention"], root_intervals, trace)
        need(uint(request["command_segments"], "profiled command segments") == trace["info"]["root_count"],
             f"request {index}: profile trace/root count differs from command segments")
        return {"index": index, "kind": request["kind"], "combined_attention": expected["combined_attention"],
                "profiling": True, "input_sha256": input_hash, "comparisons": comparisons,
                "per_block_encode_host_seconds": block_encode_spans,
                "timing": {field: float(request[field]) for field in ("blocks_encode_seconds", "head_encode_seconds",
                    "encode_seconds", "observed_encode_span_seconds", "terminal_wait_seconds", "total_seconds",
                    "gpu_elapsed_seconds")}, "command_segments": segment_count,
                "root_metrics": root_metrics, "phase": phase, "trace": trace}
    need(request.get("phase_profile") is None and request.get("continuation_trace") is None,
         f"request {index}: profile-off request contains detailed profile data")
    return {"index": index, "kind": request["kind"], "combined_attention": expected["combined_attention"],
            "profiling": False, "input_sha256": input_hash, "comparisons": comparisons,
            "per_block_encode_host_seconds": block_encode_spans,
            "timing": {field: float(request[field]) for field in ("blocks_encode_seconds", "head_encode_seconds",
                "encode_seconds", "observed_encode_span_seconds", "terminal_wait_seconds", "total_seconds",
                "gpu_elapsed_seconds")}, "command_segments": segment_count,
            "root_metrics": root_metrics}


def _expected_schedule(reverse: bool) -> list[dict[str, Any]]:
    modes = [True, False, False, True, True, False, False, True]
    if reverse:
        modes = [not mode for mode in modes]
    kinds = ["a_warm_off_0", "a_warm_off_1", "a_profile_0", "a_profile_1",
             "a_profile_2", "a_profile_3", "a_late_off_0", "a_late_off_1"]
    profiles = [False, False, True, True, True, True, False, False]
    return [{"kind": kinds[i], "combined_attention": modes[i], "profile": profiles[i],
             "poisoned": i < 2} for i in range(8)]


def validate_memory_events(raw: object, request_count: int = 8) -> dict[str, Any]:
    rows = arr(raw, "memory_phases")
    need(len(rows) > request_count * 6, "memory phase telemetry is incomplete")
    parsed: list[dict[str, Any]] = []
    previous_guard = previous_monotonic = -1
    for index, raw_event in enumerate(rows):
        event = obj(raw_event, f"memory event {index}")
        need(event.get("schema") == "cogni_ml.pixelumm_memory_phase/v1" and
             event.get("sequence") == index + 1 and event.get("timestamp_clock") == "CLOCK_MONOTONIC" and
             event.get("guard_timestamp_clock") == "mach_absolute_time" and isinstance(event.get("phase"), str),
             f"memory event {index}: schema/clock/sequence mismatch")
        monotonic = uint(event.get("monotonic_ns"), "memory monotonic_ns", minimum=1)
        guard = uint(event.get("guard_monotonic_ns"), "memory guard_monotonic_ns", minimum=1)
        need(monotonic >= previous_monotonic and guard >= previous_guard,
             f"memory event {index}: recorded clock moved backwards")
        previous_monotonic, previous_guard = monotonic, guard
        req_value = event.get("request")
        if req_value is not None:
            req_value = uint(req_value, "memory event request")
            need(req_value < request_count, f"memory event {index}: request index out of range")
        parsed.append({"sequence": index + 1, "phase": event["phase"], "request": req_value,
                       "guard_monotonic_ns": guard, "monotonic_ns": monotonic})

    by_request: dict[int, dict[str, int]] = {i: {} for i in range(request_count)}
    for event in parsed:
        if event["request"] is None:
            continue
        phase = event["phase"]
        if phase in {"request_before_encode", "request_after_terminal_wait", "request_after_profile_extraction",
                     "request_after_diagnostic_comparisons", "after_chain_release", "before_post_diagnostic_gc",
                     "after_post_diagnostic_gc"}:
            need(phase not in by_request[event["request"]],
                 f"request {event['request']}: duplicate memory milestone {phase}")
            by_request[event["request"]][phase] = event["guard_monotonic_ns"]
    milestone_names = ("request_before_encode", "request_after_terminal_wait", "request_after_profile_extraction",
                       "request_after_diagnostic_comparisons", "after_chain_release", "before_post_diagnostic_gc",
                       "after_post_diagnostic_gc")
    for index, milestones in by_request.items():
        need(set(milestones) == set(milestone_names), f"request {index}: memory milestone set incomplete")
        values = [milestones[name] for name in milestone_names]
        need(values == sorted(values), f"request {index}: memory milestones are out of order")

    spans: list[dict[str, Any]] = []
    for index, milestones in by_request.items():
        def gap(start_name: str, end_name: str) -> float:
            return (milestones[end_name] - milestones[start_name]) / 1e9
        spans.append({"request": index,
            "wait_to_profile_extraction_seconds": gap("request_after_terminal_wait", "request_after_profile_extraction"),
            "profile_extraction_to_comparison_seconds": gap("request_after_profile_extraction", "request_after_diagnostic_comparisons"),
            "comparison_to_chain_release_seconds": gap("request_after_diagnostic_comparisons", "after_chain_release"),
            "chain_release_to_gc_start_seconds": gap("after_chain_release", "before_post_diagnostic_gc"),
            "post_gc_seconds": gap("before_post_diagnostic_gc", "after_post_diagnostic_gc")})
    cross_request: list[dict[str, Any]] = []
    for index in range(request_count - 1):
        start = by_request[index]["after_post_diagnostic_gc"]
        end = by_request[index + 1]["request_before_encode"]
        need(end >= start, "inter-request memory-event timestamps moved backwards")
        cross_request.append({"after_request": index, "before_request": index + 1,
                              "post_gc_to_next_encode_seconds": (end - start) / 1e9})

    # Stage-level event gaps separate fixture/readback setup from the
    # comparator's CPU loop and digest work without adding instrumentation.
    gap_rows = []
    for left, right in zip(parsed, parsed[1:]):
        delta = (right["guard_monotonic_ns"] - left["guard_monotonic_ns"]) / 1e9
        if left["request"] == right["request"] and left["request"] is not None:
            if left["phase"] in ("compare_buffers_ready", "integrated_head_compare_buffers_ready") and \
               right["phase"] in ("compare_stage_done", "integrated_head_compare_stage_done"):
                gap_rows.append({"request": left["request"], "from": left["phase"], "to": right["phase"],
                                 "seconds": delta, "kind": "post_readback_compare_and_digest"})
            elif left["phase"] in ("before_compare_buffers", "before_integrated_head_compare_buffers") and \
                 right["phase"] in ("compare_buffers_ready", "integrated_head_compare_buffers_ready"):
                gap_rows.append({"request": left["request"], "from": left["phase"], "to": right["phase"],
                                 "seconds": delta, "kind": "fixture_read_and_activation_readback"})
    for index in range(request_count):
        own = [row for row in gap_rows if row["request"] == index]
        need(sum(row["kind"] == "post_readback_compare_and_digest" for row in own) == 43 and
             sum(row["kind"] == "fixture_read_and_activation_readback" for row in own) == 43,
             f"request {index}: expected 43 measured compare/readback event gaps")
    sorted_gaps = sorted(gap_rows, key=lambda row: row["seconds"], reverse=True)
    all_adjacent = [{"from_sequence": left["sequence"], "to_sequence": right["sequence"],
                     "from_phase": left["phase"], "to_phase": right["phase"],
                     "request": right["request"],
                     "guard_gap_seconds": (right["guard_monotonic_ns"] - left["guard_monotonic_ns"]) / 1e9}
                    for left, right in zip(parsed, parsed[1:])]
    largest_adjacent = sorted(all_adjacent, key=lambda row: row["guard_gap_seconds"], reverse=True)[:20]
    return {"event_count": len(parsed), "clock": "mach_absolute_time converted to nanoseconds by the existing recorder",
            "request_spans": spans, "post_gc_to_next_encode": cross_request,
            "comparison_stage_gap_counts": {"readback_setup": 43, "compare_and_digest": 43},
            "comparison_stage_gap_seconds": {
                "fixture_read_and_activation_readback": distribution([r["seconds"] for r in gap_rows if r["kind"] == "fixture_read_and_activation_readback"]),
                "post_readback_compare_and_digest": distribution([r["seconds"] for r in gap_rows if r["kind"] == "post_readback_compare_and_digest"]),
            },
            "largest_adjacent_guard_gaps": largest_adjacent,
            "interpretation": "These are elapsed spans between existing memory-recorder events, outside the request timer after terminal wait; readback and CPU comparator work are not isolated more finely."}


def analyze_report(data_raw: object, report_sha256: str, path: str) -> dict[str, Any]:
    data = obj(data_raw, "report")
    need(data.get("schema") == REPORT and data.get("generation") is False and
         data.get("scope") == "native_36_blocks_plus_global_head_in_one_gpu_chain" and
         data.get("profile_combined_attention") is True and data.get("compare_combined_attention") is False and
         data.get("profile_variance") is False and data.get("profile_shared_gate_up") is False and
         data.get("compare_shared_gate_up") is False and data.get("correctness_only") is False,
         "report schema/scope/mode flags are not the bounded combined-attention profile")
    reverse = boolean(data.get("reverse_combined_attention"), "reverse_combined_attention")
    need(float(data.get("atol")) == ATOL and float(data.get("rtol")) == RTOL,
         "saved-capture comparator tolerance is not 2e-5/2e-5")
    need(data.get("grid") == GRID and data.get("block_indices") == list(range(BLOCKS)) and
         data.get("output_shape") == [1, 3, 24, 704, 1280],
         "report geometry differs from the full 36-block J=21120 fixture")
    need(data.get("passed") is True and data.get("failure") is None and
         data.get("cleanup_quarantined") is False and uint(data.get("repeats"), "repeats") == 1 and
         uint(data.get("expected_request_count"), "expected request count") == 8,
         "report did not complete the exact eight-request diagnostic")
    need(data.get("duplicate_head_input_bytes") == 0 and data.get("shared_block_workspace") is True and
         data.get("persistent_block_output_banks") == 2 and data.get("head_scratch_aliases_block_workspace") is False,
         "report head/block buffer topology changed")
    for field in ("reference_sha256", "alternate_reference_sha256", "head_reference_sha256",
                  "head_alternate_reference_sha256", "bf16_cache_sha256"):
        digest(data.get(field), f"report {field}")
    for field in ("named_bytes", "block_named_bytes", "head_owned_named_bytes", "head_weight_named_bytes", "named_budget_bytes"):
        uint(data.get(field), f"report {field}")
    need(data["named_bytes"] == data["block_named_bytes"] + data["head_owned_named_bytes"] + data["head_weight_named_bytes"] and
         data["named_bytes"] <= data["named_budget_bytes"] <= 24 * 1024**3,
         "report named-byte inventory exceeds its budget or does not add up")

    capture_order = _expected_capture_order()
    need(data.get("compared_capture_order") == capture_order,
         "report did not declare the complete ordered 43-capture comparison")
    schedule = _expected_schedule(reverse)
    schedule_rows = arr(data.get("combined_attention_schedule"), "combined_attention_schedule")
    need(len(schedule_rows) == 8, "expected eight combined-attention schedule entries")
    for i, (raw_arm, expected) in enumerate(zip(schedule_rows, schedule)):
        arm = obj(raw_arm, f"schedule arm {i}")
        exact_keys(arm, ("kind", "combined_attention", "alternate", "poisoned", "profiling"), f"schedule arm {i}")
        need(arm["kind"] == expected["kind"] and arm["combined_attention"] is expected["combined_attention"] and
             arm["alternate"] is False and arm["poisoned"] is expected["poisoned"] and
             arm["profiling"] is expected["profile"], f"schedule arm {i} does not match the admitted normal/reverse profile schedule")
    shared_schedule = arr(data.get("shared_gate_up_schedule"), "shared gate/up schedule")
    need(len(shared_schedule) == 8, "shared gate/up schedule is incomplete")
    for i, raw_arm in enumerate(shared_schedule):
        arm = obj(raw_arm, f"shared gate/up schedule arm {i}")
        exact_keys(arm, ("kind", "shared_gate_up", "profiling"), f"shared gate/up schedule arm {i}")
        need(arm["kind"] == schedule[i]["kind"] and arm["shared_gate_up"] is False and
             arm["profiling"] is schedule[i]["profile"], f"schedule arm {i} unexpectedly changes FFN mode")

    requests_raw = arr(data.get("requests"), "requests")
    need(len(requests_raw) == 8, "expected eight request rows")
    requests = [_validate_request(row, i, schedule[i]) for i, row in enumerate(requests_raw)]
    inputs = {request["input_sha256"] for request in requests}
    need(len(inputs) == 1, "profile schedule must hold the same input A across all eight requests")
    for mode in (False, True):
        counts = {row["command_segments"] for row in requests if row["combined_attention"] is mode}
        need(len(counts) == 1, f"root/segment topology changed between requests within mode {mode}")
    profiled = [request for request in requests if request["profiling"]]
    need(len(profiled) == 4, "expected four profiled crossover requests")
    memory = validate_memory_events(data.get("memory_phases"))
    return {"path": path, "sha256": report_sha256, "reverse": reverse,
            "mode_order": [row["combined_attention"] for row in requests],
            "same_input_sha256": next(iter(inputs)), "requests": requests,
            "profiled_request_count": len(profiled), "memory_event_gaps": memory,
            "fixture_pins": {key: data[key] for key in ("reference_sha256", "alternate_reference_sha256",
                "head_reference_sha256", "head_alternate_reference_sha256", "bf16_cache_sha256")},
            "capture_order": capture_order}


def _continuation_summary(profiled_requests: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase: dict[int, list[dict[str, Any]]] = {phase: [] for phase in API_PHASES}
    all_events = []
    for request in profiled_requests:
        trace = request["trace"]
        for event in trace["events"]:
            begin = trace["host_seconds"](event["begin_mach_ticks"])
            end = trace["host_seconds"](event["end_mach_ticks"])
            earlier = trace["roots"][:event["before_root_sequence"]]
            intersections = [(max(begin, root["start_seconds"]), min(end, root["end_seconds"]))
                             for root in earlier if root["start_seconds"] < end and root["end_seconds"] > begin]
            intersections = [(left, right) for left, right in intersections if right > left]
            overlap = interval_union(intersections)
            completed_during = sum(begin <= trace["host_seconds"](root["completion_observed_mach_ticks"]) <= end
                                   for root in earlier)
            row = {"request": request["index"], "mode_combined_attention": request["combined_attention"],
                   "event_serial": event["serial"], "phase_id": event["phase"],
                   "phase": API_PHASES[event["phase"]], "graph_depth": event["graph_depth"],
                   "wall_seconds": end - begin, "thread_cpu_seconds": event["thread_cpu_seconds"],
                   "earlier_root_overlap_seconds": overlap,
                   "earlier_root_overlap_fraction_of_call": overlap / (end - begin),
                   "earlier_roots_overlapping": len(intersections),
                   "earlier_root_completions_observed_during_call": completed_during,
                   "completion_observation_note": "Completion callback/status observation, not exact GPU completion time."}
            by_phase[event["phase"]].append(row)
            all_events.append(row)
    phases = {}
    for phase_id, label in API_PHASES.items():
        rows = by_phase[phase_id]
        wall = sum(row["wall_seconds"] for row in rows)
        overlap = sum(row["earlier_root_overlap_seconds"] for row in rows)
        need(overlap <= wall + 1e-9, f"phase {label}: earlier-root overlap exceeds continuation wall time")
        phases[label] = {"event_count": len(rows), "total_wall_seconds": wall,
                         "total_thread_cpu_seconds": sum(row["thread_cpu_seconds"] for row in rows),
                         "total_earlier_root_overlap_seconds": overlap,
                         "wall_weighted_earlier_root_overlap_fraction": overlap / wall if wall else None,
                         "per_call_wall_seconds": distribution([row["wall_seconds"] for row in rows]),
                         "per_call_thread_cpu_seconds": distribution([row["thread_cpu_seconds"] for row in rows]),
                         "per_call_earlier_root_overlap_fraction": distribution([row["earlier_root_overlap_fraction_of_call"] for row in rows]),
                         "callback_observations_during_calls": sum(row["earlier_root_completions_observed_during_call"] for row in rows)}
    return {"event_count": len(all_events), "by_phase": phases,
            "largest_earlier_root_overlaps": sorted(all_events,
                key=lambda row: row["earlier_root_overlap_seconds"], reverse=True)[:20],
            "parent_graph_validation_scope": "Syntactic parent/depth/flag checks only. The exported trace does not contain every graph API scope, so this is not a complete parent-forest validation.",
            "backlog_semantics": "For event before_root_sequence=i, overlap uses only roots with sequence < i; root i being submitted is excluded. GPU timestamps are compared to Mach-host intervals under the trace's existing 2 microsecond clock bound. Callback observations are not exact GPU completion instants."}


def _profile_summary(run: dict[str, Any]) -> dict[str, Any]:
    profiled = [request for request in run["requests"] if request["profiling"]]
    phase_distributions: dict[bool, dict[str, Any]] = {}
    for mode in (False, True):
        selected = [request for request in profiled if request["combined_attention"] is mode]
        need(len(selected) == 2, "each run must profile two requests per mode")
        phase_distributions[mode] = {
            "count": len(selected),
            "host_timers": {key: distribution([request["timing"][key] for request in selected])
                            for key in ("blocks_encode_seconds", "head_encode_seconds", "terminal_wait_seconds", "total_seconds")},
            "phase_wall_seconds": {phase: distribution([request["phase"]["phase_wall_seconds"][phase] for request in selected])
                                   for phase in PHASES},
            "phase_thread_cpu_seconds": {phase: distribution([request["phase"]["phase_thread_cpu_seconds"][phase] for request in selected])
                                          for phase in PHASES},
            "whole_chain_roots": {key: distribution([request["phase"]["root_metrics"][key] for request in selected])
                                   for key in ("root_duration_seconds", "union_seconds", "envelope_seconds", "overlap_seconds", "gap_seconds")},
            "api_profile_inclusive_and_exclusive_ns": _aggregate_api(selected),
        }
    # Preserve per-block wall/CPU vectors for each profiled request; the source
    # aggregate is recomputed from these exact 36x11 marker arrays above.
    per_request = []
    for request in profiled:
        per_request.append({"request": request["index"], "kind": request["kind"],
            "combined_attention": request["combined_attention"], "timing": request["timing"],
            "recomputed_aggregate_phase_wall_seconds": request["phase"]["phase_wall_seconds"],
            "recomputed_aggregate_phase_thread_cpu_seconds": request["phase"]["phase_thread_cpu_seconds"],
            "per_block_phase_markers": request["phase"]["per_block"],
            "root_interval_metrics": request["phase"]["root_metrics"],
            "candidate_root_duration_sums_nonadditive": request["phase"]["candidate_root_duration_sums"],
            "definite_root_duration_sums_nonadditive": request["phase"]["definite_root_duration_sums"]})
    return {"profiled_requests": per_request, "by_mode": {
                "baseline": phase_distributions[False], "combined": phase_distributions[True]},
            "continuation_trace": _continuation_summary(profiled),
            "interpretation": "All phase/API/continuation timings are observer-on localization measurements. Profile-off warm and tail controls reveal drift and do not calibrate profiling overhead; these are not speed or kernel-exclusive timing claims."}


def _aggregate_api(requests: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for request in requests:
        for phase_id, operations in request["phase"]["api"].items():
            for operation, stats in operations.items():
                for field, value in stats.items():
                    if field in ("wall_max_ns", "thread_cpu_max_ns"):
                        totals[phase_id][operation][field] = max(totals[phase_id][operation][field], value)
                    else:
                        totals[phase_id][operation][field] += value
    result = {}
    for phase_id, phase_name in API_PHASES.items():
        result[phase_name] = {}
        for operation in OPS:
            result[phase_name][operation] = dict(totals[phase_id][operation])
    return result


def analyze_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    need(len(reports) == 2, "supply exactly two reports: one normal and one reverse order")
    by_direction = {bool(report["reverse"]): report for report in reports}
    need(set(by_direction) == {False, True}, "reports must contain one normal and one reverse schedule")
    normal, reverse = by_direction[False], by_direction[True]
    need(normal["fixture_pins"] == reverse["fixture_pins"] and
         normal["same_input_sha256"] == reverse["same_input_sha256"],
         "normal/reverse reports do not use the same authenticated fixtures and input A")
    run_summaries = []
    for run in (normal, reverse):
        run_summaries.append({"path": run["path"], "sha256": run["sha256"],
            "order": "reverse" if run["reverse"] else "normal", "same_input_sha256": run["same_input_sha256"],
            "requests": [{"request": row["index"], "kind": row["kind"],
                          "combined_attention": row["combined_attention"], "profiling": row["profiling"],
                          "timing": row["timing"], "command_segments": row["command_segments"],
                          "per_block_encode_host_seconds": row["per_block_encode_host_seconds"],
                          "root_interval_metrics": row["root_metrics"],
                          "different_words": [capture["different_words"] for capture in row["comparisons"]]}
                         for row in run["requests"]],
            "memory_event_gaps": run["memory_event_gaps"],
            "phase_localization": _profile_summary(run)})
    return {"schema": SUMMARY, "input_report_schema": REPORT,
            "scope": "Two same-input normal/reverse eight-request diagnostic runs; no generation.",
            "numerical_scope": {"atol": ATOL, "rtol": RTOL, "captures_per_request": 43,
                "all_captures_finite_and_within_tolerance": True,
                "different_bf16_words_preserved": True,
                "not_claimed": ["bitwise parity", "internal F32 softmax/probability parity", "generation speed or production performance"]},
            "lifecycle_scope": {"requests_per_run": 8, "terminal_waits_per_request": 1,
                "post_wait_capture_readbacks_per_request": 43, "preterminal_activation_readbacks_per_request": 0,
                "named_encode_allocations_per_request": 0, "full36_plans_selected_each_request": 36,
                "continuation_events_checked_after_existing_terminal_wait": True},
            "fixture_pins": normal["fixture_pins"], "runs": run_summaries,
            "analysis_limit": "Nested per-phase root candidate sums and inclusive MPS Graph/continuation spans are not additive GPU work. Memory-event gaps are existing observer timestamps, not instrumented subphase timers. Continuation parent/depth checks are syntactic, not a complete graph-scope parent forest, because the exported trace omits some graph scopes."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path,
                        help="profile report path; repeat once for normal and once for reverse")
    parser.add_argument("--output", type=Path,
                        help="fresh summary file (exclusive create); defaults to stdout")
    args = parser.parse_args(argv)
    try:
        reports = []
        for path in args.report:
            raw = path.read_bytes()
            reports.append(analyze_report(json.loads(raw), hashlib.sha256(raw).hexdigest(), str(path)))
        summary = analyze_reports(reports)
        encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(encoded)
        else:
            sys.stdout.write(encoded)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"pixelumm order analysis rejected report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
