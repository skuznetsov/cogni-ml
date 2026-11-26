#!/usr/bin/env python3
"""Validate and summarize the fixed J5760 FFN A/A continuation-trace sweep.

This CPU-only analyzer checks the four-arm OFF/ON/ON/OFF report set, numerical
controls, the one-wait and allocation boundaries, API-profile counts, trace
root isolation, and the native continuation events. It reuses the shared
PixelUMM trace validator and earlier-root interval-union calculation. Timing
arm differences are descriptive observer-on observations, not causal GPU
attribution or an optimization claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any


REPORT = "cogni_ml.pixelumm_ffn_aa_probe/v1"
SUMMARY = "cogni_ml.pixelumm_ffn_aa_trace_analysis/v1"
SCOPE = "native connected FFN A/A under fixed GPU-comparator cadence; no full generation"
TOKENS = 5760
HIDDEN = 4096
FFN = 12288
WARMUPS = 2
TIMED_SAMPLES = 16
TOTAL_SAMPLES = WARMUPS + TIMED_SAMPLES
METRIC_FIELDS = ("failures", "different_words", "nonfinite", "elements")
API_FIELDS = ("count", "wall_total_ns", "wall_max_ns", "thread_cpu_total_ns",
              "thread_cpu_max_ns", "nested_count", "nested_wall_total_ns",
              "nested_thread_cpu_total_ns", "exclusive_wall_total_ns",
              "exclusive_thread_cpu_total_ns")
PHASES = {0: "unassigned", 1: "self_prepare", 2: "self_attention",
          3: "self_projection_residual", 4: "cross_attention_projection_residual", 5: "ffn"}
OPERATIONS = ("graph_encode", "commit_and_continue", "t5_linear_encode")
EXPECTED_OPERATION_COUNTS = [18, 18, 18, 18, 18, 18, 2]
EXPECTED_NAMED_BYTES = {
    "input.norm": TOKENS * HIDDEN * 2,
    "weight.gate": HIDDEN * FFN * 2,
    "weight.up": HIDDEN * FFN * 2,
    "weight.down": HIDDEN * FFN * 2,
    "expected.gate": TOKENS * FFN * 2,
    "expected.up": TOKENS * FFN * 2,
    "expected.product": TOKENS * FFN * 2,
    "expected.down": TOKENS * HIDDEN * 2,
    "work.gate": TOKENS * FFN * 2,
    "work.up": TOKENS * FFN * 2,
    "work.down": TOKENS * HIDDEN * 2,
    "capture.gate": TOKENS * FFN * 2,
    "capture.up": TOKENS * FFN * 2,
    "canonical.down": TOKENS * HIDDEN * 2,
    "metrics": 57 * 4 * 4,
}
FIXTURE_HASHES = ("metadata_sha256", "payload_sha256", "weight_cache_metadata_sha256",
                  "weight_cache_payload_sha256", "normalized_input", "upstream_model_sha256")


def _load_order_analyzer() -> Any:
    path = Path(__file__).with_name("analyze_pixelumm_order_profile.py")
    spec = importlib.util.spec_from_file_location("pixelumm_order_profile_trace_reused", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared PixelUMM trace validator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORDER = _load_order_analyzer()


def _stats(raw: object, where: str) -> dict[str, int]:
    value = ORDER.obj(raw, where)
    ORDER.exact_keys(value, API_FIELDS, where)
    return {key: ORDER.uint(value[key], f"{where}.{key}") for key in API_FIELDS}


def _validate_api_profile(raw: object, report_index: int) -> dict[str, Any]:
    api = ORDER.obj(raw, f"report {report_index} api_profile")
    ORDER.exact_keys(api, ("phase_count", "phases"), f"report {report_index} api_profile")
    ORDER.need(ORDER.uint(api["phase_count"], "api_profile.phase_count") == 6,
               f"report {report_index}: expected six API phases")
    phase_rows = ORDER.arr(api["phases"], "api_profile.phases")
    ORDER.need(len(phase_rows) == 6, f"report {report_index}: API phase matrix is incomplete")
    indexed: dict[int, dict[str, dict[str, int]]] = {}
    for row_index, raw_phase in enumerate(phase_rows):
        phase = ORDER.obj(raw_phase, f"API phase {row_index}")
        ORDER.exact_keys(phase, ("phase_id", "phase", "operations"), f"API phase {row_index}")
        phase_id = ORDER.uint(phase["phase_id"], "API phase id")
        ORDER.need(phase_id in PHASES and phase_id not in indexed and phase["phase"] == PHASES[phase_id],
                   f"report {report_index}: API phase id/name is unknown or duplicated")
        operations = ORDER.obj(phase["operations"], f"API phase {phase_id} operations")
        ORDER.exact_keys(operations, OPERATIONS, f"API phase {phase_id} operations")
        parsed = {}
        for operation in OPERATIONS:
            stats = _stats(operations[operation], f"API phase {phase_id}.{operation}")
            ORDER._validate_api_operation_stats(stats, operation, report_index)
            parsed[operation] = stats
        graph, commit = parsed["graph_encode"], parsed["commit_and_continue"]
        ORDER.need(tuple(graph[field] for field in ("nested_count", "nested_wall_total_ns",
                                                   "nested_thread_cpu_total_ns")) ==
                   tuple(commit[field] for field in ("nested_count", "nested_wall_total_ns",
                                                     "nested_thread_cpu_total_ns")),
                   f"report {report_index}: graph/continuation nested totals disagree in phase {phase_id}")
        indexed[phase_id] = parsed
    ORDER.need(set(indexed) == set(PHASES), f"report {report_index}: API phase IDs are incomplete")

    graph_counts = {phase: indexed[phase]["graph_encode"]["count"] for phase in PHASES}
    commit_counts = {phase: indexed[phase]["commit_and_continue"]["count"] for phase in PHASES}
    linear_counts = {phase: indexed[phase]["t5_linear_encode"]["count"] for phase in PHASES}
    ORDER.need(graph_counts == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 54} and
               commit_counts == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 161} and
               all(count == 0 for count in linear_counts.values()),
               f"report {report_index}: API counts are not 54 graphs, 161 commits, zero linears, all FFN")
    ffn_commit = indexed[5]["commit_and_continue"]
    ORDER.need(ffn_commit["nested_count"] == 125 and ffn_commit["count"] - ffn_commit["nested_count"] == 36,
               f"report {report_index}: continuation nesting is not 125 nested / 36 nonnested")
    return {"phase_count": 6, "graph_counts": graph_counts, "commit_counts": commit_counts,
            "linear_counts": linear_counts, "ffn_commit_stats": ffn_commit,
            "ffn_graph_stats": indexed[5]["graph_encode"],
            "_phase_operations": indexed}


def _reconcile_trace_api(trace: dict[str, Any], api: dict[str, Any], report_index: int) -> None:
    """Cross-check per-phase trace spans against the independently sampled API aggregates."""
    for phase_id in PHASES:
        events = [event for event in trace["events"] if event["phase"] == phase_id]
        stats = api["_phase_operations"][phase_id]["commit_and_continue"]
        nested_count = sum(event["graph_depth"] > 0 for event in events)
        wall_ns = sum(event["wall_seconds"] * 1e9 for event in events)
        thread_cpu_ns = sum(event["end_thread_cpu_ns"] - event["begin_thread_cpu_ns"]
                             for event in events)
        ORDER.need(stats["count"] == len(events) and stats["nested_count"] == nested_count,
                   f"report {report_index}: API continuation count/nesting differs from raw trace phase {phase_id}")
        ORDER.need(wall_ns <= stats["wall_total_ns"] + len(events) * ORDER.AGGREGATE_WALL_ROUNDING_NS_PER_CALL and
                   thread_cpu_ns <= stats["thread_cpu_total_ns"],
                   f"report {report_index}: raw continuation wall/thread CPU exceeds API aggregate phase {phase_id}")


def _validate_metrics(raw: object, report_index: int) -> None:
    metrics = ORDER.arr(raw, f"report {report_index} numerical_metrics")
    ORDER.need(len(metrics) == 57, f"report {report_index}: expected all 57 comparator slots")
    parsed = []
    for index, raw_metric in enumerate(metrics):
        metric = ORDER.obj(raw_metric, f"metric {index}")
        ORDER.exact_keys(metric, METRIC_FIELDS, f"metric {index}")
        row = {key: ORDER.uint(metric[key], f"metric {index}.{key}") for key in METRIC_FIELDS}
        ORDER.need(all(value <= (1 << 32) - 1 for value in row.values()),
                   f"report {report_index}: metric {index} exceeds UInt32")
        parsed.append(row)
    for sample in range(TOTAL_SAMPLES):
        for offset in range(3):
            row = parsed[sample * 3 + offset]
            extent = TOKENS * (FFN if offset == 0 else HIDDEN)
            ORDER.need(row["elements"] == extent and row["failures"] == 0 and
                       row["nonfinite"] == 0 and row["different_words"] <= extent,
                       f"report {report_index}: normal comparator slot {sample * 3 + offset} failed")
            if offset == 2:
                ORDER.need(row["different_words"] == 0,
                           f"report {report_index}: canonical down words are not bitwise equal")
    for index in (54, 55):
        row = parsed[index]
        ORDER.need(row == {"failures": 0, "different_words": row["different_words"],
                           "nonfinite": 0, "elements": TOKENS * FFN} and
                   row["different_words"] <= row["elements"],
                   f"report {report_index}: raw gate/up comparator slot {index} failed")
    ORDER.need(parsed[56] == {"failures": 1, "different_words": 0, "nonfinite": 1,
                              "elements": TOKENS * HIDDEN},
               f"report {report_index}: positive NaN comparator control is missing or malformed")


def _summary(values: list[float]) -> dict[str, Any]:
    ORDER.need(len(values) == TIMED_SAMPLES and all(math.isfinite(value) and value > 0 for value in values),
               "timed root-union vector must contain 16 positive finite values")
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    lower, upper = math.floor(position), math.ceil(position)
    p90 = ordered[lower] * (1.0 - (position - lower)) + ordered[upper] * (position - lower)
    first4 = statistics.median(values[:4])
    last4 = statistics.median(values[-4:])
    return {"schema": "cogni_ml.pixelumm_ffn_aa_baseline/v1",
            "scope": "J5760_connected_baseline_aa", "sample_count": 16,
            "sample_seconds": values, "median_seconds": center,
            "min_seconds": min(values), "max_seconds": max(values),
            "p90_seconds": p90, "mad_seconds": mad,
            "mad_over_median": mad / center,
            "first4_median_seconds": first4, "last4_median_seconds": last4,
            "last4_over_first4_median": last4 / first4,
            "adjacent_pair_later_over_earlier": [values[index + 1] / values[index]
                                                  for index in range(0, 16, 2)]}


def _validate_report(raw: object, report_index: int, expected_trace: bool,
                     report_sha256: str, path: str) -> dict[str, Any]:
    data = ORDER.obj(raw, f"report {report_index}")
    ORDER.need(data.get("schema") == REPORT and data.get("scope") == SCOPE,
               f"report {report_index}: schema or scope changed")
    ORDER.need(data.get("geometry") == {"tokens": TOKENS, "hidden": HIDDEN,
                                         "ffn": FFN, "grid": [10, 18, 32]},
               f"report {report_index}: geometry differs from J5760")
    fixture = ORDER.obj(data.get("fixture"), f"report {report_index} fixture")
    for field in FIXTURE_HASHES:
        ORDER.digest(fixture.get(field), f"fixture.{field}")
    ORDER.need(fixture.get("input_window") == "w0" and fixture.get("block") == 0 and
               fixture.get("input_provenance") == "saved_initial_noise_conditioning_upstream_stem" and
               fixture.get("model_inference") is False,
               f"report {report_index}: fixture input provenance/model-use changed")
    ORDER.need(isinstance(fixture.get("path"), str) and os.path.isabs(fixture["path"]) and
               os.path.normpath(fixture["path"]) == fixture["path"],
               f"report {report_index}: fixture path is not absolute/canonical")
    comparison = ORDER.obj(data.get("comparison"), "comparison")
    ORDER.exact_keys(comparison, ("atol", "rtol"), "comparison")
    ORDER.need(ORDER.finite(comparison["atol"], "comparison.atol") == 2e-5 and
               ORDER.finite(comparison["rtol"], "comparison.rtol") == 2e-5,
               f"report {report_index}: comparator tolerance changed")

    protocol = ORDER.obj(data.get("protocol"), f"report {report_index} protocol")
    ORDER.need(ORDER.uint(protocol.get("warmups"), "warmups") == WARMUPS and
               ORDER.uint(protocol.get("timed_samples"), "timed_samples") == TIMED_SAMPLES and
               protocol.get("identical_plan_and_buffers") is True and
               ORDER.uint(protocol.get("normal_checks_per_sample"), "normal checks") == 3 and
               ORDER.uint(protocol.get("raw_gate_up_checks_after_loop"), "raw checks") == 2 and
               ORDER.uint(protocol.get("positive_nan_controls"), "NaN controls") == 1 and
               protocol.get("cpu_diagnostics_between_samples") is False and
               protocol.get("cpu_readback_between_samples") is False and
               protocol.get("gpu_checks_between_samples") is True and
               protocol.get("no_candidate_or_promotion") is True and
               protocol.get("continuation_trace_enabled") is expected_trace,
               f"report {report_index}: fixed protocol or trace-arm flag changed")
    ORDER.need(protocol.get("timed_path") ==
               "separate gate/up F32-compute MPSGraph -> BF16 SiLU -> BF16 product -> F32-compute down MPSGraph" and
               protocol.get("correctness_capture_copies") == "raw gate/up in sample 0 warmup only",
               f"report {report_index}: FFN production path/capture schedule changed")

    execution = ORDER.obj(data.get("execution"), f"report {report_index} execution")
    ORDER.need(ORDER.uint(execution.get("encoding_named_allocation_delta"), "encoding allocations") == 0 and
               ORDER.uint(execution.get("encoding_readbacks"), "encoding readbacks") == 0 and
               ORDER.uint(execution.get("total_readbacks"), "total readbacks") == 1 and
               ORDER.uint(execution.get("wait_count"), "wait count") == 1 and
               ORDER.uint(execution.get("total_root_count"), "total root count") == 162 and
               execution.get("operation_counts") == EXPECTED_OPERATION_COUNTS,
               f"report {report_index}: one-wait, no-allocation/readback, root, or operation-count contract failed")
    resources = ORDER.obj(execution.get("resources"), "named resources")
    ORDER.need(resources == EXPECTED_NAMED_BYTES and
               ORDER.uint(execution.get("named_bytes"), "named bytes") == sum(EXPECTED_NAMED_BYTES.values()) and
               sum(resources.values()) <= 2 * 1024**3,
               f"report {report_index}: named resource inventory/cap changed")
    for field in ("encode_loop_wall_seconds", "terminal_wait_wall_seconds"):
        ORDER.need(ORDER.finite(execution.get(field), field) > 0,
                   f"report {report_index}: invalid {field}")

    ORDER.need(data.get("numerical_pass") is True and data.get("numerical_failure") is None and
               data.get("timing_isolation_pass") is True and data.get("pass") is True,
               f"report {report_index}: runner reported a failure")
    _validate_metrics(data.get("numerical_metrics"), report_index)
    api = _validate_api_profile(data.get("api_profile"), report_index)

    raw_roots = ORDER.arr(data.get("root_timings"), "root_timings")
    ORDER.need(len(raw_roots) == 162, f"report {report_index}: expected 162 roots")
    intervals: list[tuple[float, float]] = []
    for sequence, raw_root in enumerate(raw_roots):
        root = ORDER.obj(raw_root, f"root timing {sequence}")
        ORDER.exact_keys(root, ("sequence", "gpu_start_seconds", "gpu_end_seconds", "completed", "has_error"),
                         f"root timing {sequence}")
        start = ORDER.finite(root["gpu_start_seconds"], "root start")
        end = ORDER.finite(root["gpu_end_seconds"], "root end")
        ORDER.need(ORDER.uint(root["sequence"], "root sequence") == sequence and
                   root["completed"] is True and root["has_error"] is False and
                   start > 0 and end > start,
                   f"report {report_index}: root {sequence} is reordered, failed, or non-finite")
        intervals.append((start, end))

    samples_raw = ORDER.arr(data.get("samples"), "samples")
    ORDER.need(len(samples_raw) == TOTAL_SAMPLES, f"report {report_index}: expected 18 scheduled samples")
    samples = []
    used_boundary_values: set[int] = set()
    total_internal = 0
    for index, raw_sample in enumerate(samples_raw):
        sample = ORDER.obj(raw_sample, f"sample {index}")
        ORDER.exact_keys(sample, ("index", "warmup", "capture_raw", "mode", "first_root", "end_root",
                                  "encode_wall_seconds", "gpu_root_union_seconds",
                                  "outside_root_overlap_seconds", "isolated_root_timing"), f"sample {index}")
        first = ORDER.uint(sample["first_root"], "sample first_root")
        end_root = ORDER.uint(sample["end_root"], "sample end_root")
        expected_first = 1 if index == 0 else 9 * index
        expected_end = 8 if index == 0 else 9 * index + 8
        ORDER.need(ORDER.uint(sample["index"], "sample index") == index and
                   sample["warmup"] is (index < WARMUPS) and
                   sample["capture_raw"] is (index == 0) and sample["mode"] == "baseline" and
                   first == expected_first and end_root == expected_end and
                   end_root < len(intervals),
                   f"report {report_index}: sample {index} schedule/root range changed")
        ORDER.finite(sample["encode_wall_seconds"], "sample encode wall")
        selected = intervals[first:end_root]
        gpu_union = ORDER.interval_union(selected)
        intersections = []
        for a, b in selected:
            for c, d in intervals[:first] + intervals[end_root:]:
                left, right = max(a, c), min(b, d)
                if right > left:
                    intersections.append((left, right))
        outside_overlap = ORDER.interval_union(intersections)
        ORDER.need(ORDER.near(ORDER.finite(sample["gpu_root_union_seconds"], "sample GPU union"),
                              gpu_union, 2e-8, 1e-8) and
                   ORDER.near(ORDER.finite(sample["outside_root_overlap_seconds"], "sample outside overlap"),
                              outside_overlap, 2e-8, 1e-8) and
                   sample["isolated_root_timing"] is (outside_overlap == 0.0) and
                   sample["isolated_root_timing"] is True,
                   f"report {report_index}: sample {index} root union/isolation is false or stale")
        start_boundary, end_boundary = first - 1, end_root - 1
        ORDER.need(start_boundary not in used_boundary_values and end_boundary not in used_boundary_values,
                   f"report {report_index}: duplicate boundary event mapping")
        used_boundary_values.update((start_boundary, end_boundary))
        total_internal += end_root - first - 1
        samples.append({**sample, "gpu_root_union_seconds": gpu_union,
                        "outside_root_overlap_seconds": outside_overlap})
    ORDER.need(total_internal + len(used_boundary_values) == 161 and
               total_internal == 125 and len(used_boundary_values) == 36,
               f"report {report_index}: sample topology must classify 125 internal and 36 boundary commits")

    timed_values = [float(sample["gpu_root_union_seconds"]) for sample in samples[WARMUPS:]]
    calculated_summary = _summary(timed_values)
    raw_summary = ORDER.obj(data.get("summary"), "summary")
    ORDER.exact_keys(raw_summary, tuple(calculated_summary), "summary")
    for field, expected in calculated_summary.items():
        if field in ("schema", "scope", "sample_count"):
            ORDER.need(raw_summary[field] == expected, f"report {report_index}: summary.{field} changed")
        elif field == "sample_seconds" or field == "adjacent_pair_later_over_earlier":
            values = ORDER.arr(raw_summary[field], f"summary.{field}")
            ORDER.need(len(values) == len(expected) and all(
                ORDER.near(ORDER.finite(actual, f"summary.{field}"), target, 2e-8, 1e-8)
                for actual, target in zip(values, expected)),
                f"report {report_index}: summary.{field} differs from sample timings")
        else:
            ORDER.need(ORDER.near(ORDER.finite(raw_summary[field], f"summary.{field}"),
                                  expected, 2e-8, 1e-8),
                       f"report {report_index}: summary.{field} differs from sample timings")

    raw_trace = data.get("continuation_trace")
    trace = None
    if expected_trace:
        ORDER.need(raw_trace is not None, f"report {report_index}: enabled trace is missing")
        trace = ORDER.validate_trace(raw_trace, report_index, intervals)
        ORDER.need(trace["info"]["event_count"] == 161 and trace["info"]["root_count"] == 162 and
                   dict(trace["phase_counts"]) == {5: 161} and
                   sum(event["graph_depth"] > 0 for event in trace["events"]) == 125 and
                   sum(event["graph_depth"] == 0 for event in trace["events"]) == 36,
                   f"report {report_index}: continuation trace count, phase, or nesting differs")
        # Every current-root sequence is classified exactly once as an internal
        # encode continuation or one of the two explicit async boundaries.
        event_sequences = [event["before_root_sequence"] for event in trace["events"]]
        ORDER.need(event_sequences == list(range(161)),
                   f"report {report_index}: continuation root sequence has a gap or duplicate")
        event_by_root = {event["before_root_sequence"]: event for event in trace["events"]}
        for sample in samples:
            first, end_root = sample["first_root"], sample["end_root"]
            internal = [event_by_root[sequence] for sequence in range(first, end_root - 1)]
            boundaries = [event_by_root[first - 1], event_by_root[end_root - 1]]
            ORDER.need(all(event["graph_depth"] > 0 for event in internal) and
                       all(event["graph_depth"] == 0 for event in boundaries),
                       f"report {report_index}: internal continuation or async-boundary nesting changed at sample {sample['index']}")
        _reconcile_trace_api(trace, api, report_index)
    else:
        ORDER.need(raw_trace is None, f"report {report_index}: trace-off arm carries a continuation trace")

    api = {key: value for key, value in api.items() if key != "_phase_operations"}

    return {"index": report_index, "path": path, "sha256": report_sha256,
            "trace_enabled": expected_trace, "fixture": fixture,
            "fixture_identity": {key: fixture[key] for key in FIXTURE_HASHES},
            "source_sha256": data.get("source_sha256"), "api": api,
            "samples": samples, "timed_summary": calculated_summary,
            "root_intervals": intervals, "trace": trace,
            "raw_report": data}


def analyze_report(raw: object, report_sha256: str = "0" * 64,
                   path: str = "<memory>", report_index: int = 0,
                   expected_trace: bool = False) -> dict[str, Any]:
    ORDER.digest(report_sha256, "report SHA-256")
    return _validate_report(raw, report_index, expected_trace, report_sha256, path)


def _event_overlap(trace: dict[str, Any], event: dict[str, Any], report_index: int) -> dict[str, Any]:
    one_event_trace = {**trace, "events": [event]}
    summary = ORDER._continuation_summary([{"trace": one_event_trace, "index": report_index,
                                            "combined_attention": False}])
    return summary["largest_earlier_root_overlaps"][0]


def _sample_commit_summary(run: dict[str, Any]) -> list[dict[str, Any]]:
    trace = run["trace"]
    events = trace["events"]
    output = []
    claimed_events: set[int] = set()
    for sample in run["samples"]:
        first, end_root = sample["first_root"], sample["end_root"]
        internal = [event for event in events if first <= event["before_root_sequence"] < end_root - 1]
        start_boundary = [event for event in events if event["before_root_sequence"] == first - 1]
        end_boundary = [event for event in events if event["before_root_sequence"] == end_root - 1]
        ORDER.need(len(start_boundary) == len(end_boundary) == 1,
                   f"report {run['index']} sample {sample['index']}: expected one start and one end boundary")
        for event in internal + start_boundary + end_boundary:
            ORDER.need(event["serial"] not in claimed_events,
                       f"report {run['index']}: event assigned to multiple samples")
            claimed_events.add(event["serial"])
        grouped: dict[int, list[dict[str, Any]]] = {}
        for event in internal:
            overlap = _event_overlap(trace, event, run["index"])
            grouped.setdefault(event["before_root_sequence"], []).append(overlap)

        def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
            wall = sum(row["wall_seconds"] for row in rows)
            overlap = sum(row["earlier_root_overlap_seconds"] for row in rows)
            return {"call_count": len(rows), "wall_seconds": wall,
                    "thread_cpu_seconds": sum(row["thread_cpu_seconds"] for row in rows),
                    "earlier_root_overlap_seconds": overlap,
                    "wall_weighted_earlier_root_overlap_fraction": overlap / wall if wall else None,
                    "nested_call_count": sum(row["graph_depth"] > 0 for row in rows)}

        per_root = []
        for current_root, rows in sorted(grouped.items()):
            per_root.append({"current_root_sequence": current_root, **totals(rows),
                             "events": [{"serial": row["event_serial"],
                                         "wall_seconds": row["wall_seconds"],
                                         "thread_cpu_seconds": row["thread_cpu_seconds"],
                                         "earlier_root_overlap_seconds": row["earlier_root_overlap_seconds"],
                                         "earlier_roots_overlapping": row["earlier_roots_overlapping"],
                                         "graph_depth": row["graph_depth"]} for row in rows]})
        boundary_rows = []
        for label, event in (("start", start_boundary[0]), ("end", end_boundary[0])):
            overlap = _event_overlap(trace, event, run["index"])
            boundary_rows.append({"boundary": label, "before_root_sequence": event["before_root_sequence"],
                                  "event_serial": event["serial"], "wall_seconds": overlap["wall_seconds"],
                                  "thread_cpu_seconds": overlap["thread_cpu_seconds"],
                                  "earlier_root_overlap_seconds": overlap["earlier_root_overlap_seconds"]})
        output.append({"sample": sample["index"], "first_root": first, "end_root": end_root,
                       "internal_commit_count": len(internal),
                       "internal_commit_totals": totals([_event_overlap(trace, event, run["index"])
                                                          for event in internal]),
                       "internal_commits_by_current_root_sequence": per_root,
                       "explicit_boundary_commits_excluded_from_internal_totals": {
                           "count": 2, "wall_seconds": sum(row["wall_seconds"] for row in boundary_rows),
                           "thread_cpu_seconds": sum(row["thread_cpu_seconds"] for row in boundary_rows),
                           "events": boundary_rows},
                       "classification_note": "Internal events use first_root <= before_root_sequence < end_root - 1. Explicit start/end boundaries use first_root - 1 and end_root - 1 and are reported separately."})
    ORDER.need(len(claimed_events) == len(events) == 161,
               f"report {run['index']}: not every trace event was assigned to exactly one sample")
    ORDER.need(sum(row["internal_commit_count"] for row in output) == 125 and
               sum(row["explicit_boundary_commits_excluded_from_internal_totals"]["count"] for row in output) == 36,
               f"report {run['index']}: internal/boundary event totals do not reconcile")
    return output


def analyze_reports(reports: list[object], paths: list[str] | None = None,
                    report_hashes: list[str] | None = None) -> dict[str, Any]:
    ORDER.need(len(reports) == 4, "expected four reports in OFF/ON/ON/OFF order")
    paths = paths or [f"<memory:{index}>" for index in range(4)]
    report_hashes = report_hashes or ["0" * 64 for _ in range(4)]
    ORDER.need(len(paths) == len(report_hashes) == 4, "report path/hash vectors must have four entries")
    expected_flags = [False, True, True, False]
    runs = [analyze_report(raw, report_hashes[index], paths[index], index, expected_flags[index])
            for index, raw in enumerate(reports)]
    pins = runs[0]["fixture_identity"]
    sources = runs[0]["source_sha256"]
    ORDER.need(all(run["fixture_identity"] == pins for run in runs),
               "four arms do not share identical fixture/cache/input hashes")
    ORDER.need(isinstance(sources, dict) and bool(sources) and
               all(isinstance(path, str) and path and ORDER.digest(value, f"source hash {path}")
                   for path, value in sources.items()) and
               all(run["source_sha256"] == sources for run in runs),
               "four arms do not share identical valid recorded source hash entries")

    timed_medians = [run["timed_summary"]["median_seconds"] for run in runs]
    off_values = [value for index, run in enumerate(runs) if not run["trace_enabled"]
                  for value in run["timed_summary"]["sample_seconds"]]
    on_values = [value for run in runs if run["trace_enabled"]
                 for value in run["timed_summary"]["sample_seconds"]]
    off_median, on_median = statistics.median(off_values), statistics.median(on_values)

    trace_runs = [run for run in runs if run["trace"] is not None]
    trace_aggregate = ORDER._continuation_summary([{"trace": run["trace"], "index": run["index"],
                                                     "combined_attention": False}
                                                    for run in trace_runs])
    per_run_trace = []
    for run in trace_runs:
        full_summary = ORDER._continuation_summary([{"trace": run["trace"], "index": run["index"],
                                                      "combined_attention": False}])
        per_run_trace.append({"report_index": run["index"], "path": run["path"],
                              "summary": full_summary,
                              "sample_commit_totals": _sample_commit_summary(run),
                              "api_profile": run["api"]})

    arm_rows = [{"index": run["index"], "path": run["path"], "sha256": run["sha256"],
                 "continuation_trace_enabled": run["trace_enabled"],
                 "timed_sample_count": TIMED_SAMPLES,
                 "timed_gpu_root_union_summary": run["timed_summary"],
                 "api_profile": run["api"]} for run in runs]
    return {"schema": SUMMARY, "input_report_schema": REPORT,
            "scope": "Four sequential same-process OFF/ON/ON/OFF trace-observer arms; fixed J5760 baseline only.",
            "shared_fixture_identity": pins, "shared_source_sha256": sources,
            "arms": arm_rows,
            "observer_effect_descriptive_only": {
                "trace_off_sample_count": len(off_values), "trace_on_sample_count": len(on_values),
                "trace_off_median_seconds": off_median, "trace_on_median_seconds": on_median,
                "trace_on_over_off_median": on_median / off_median,
                "per_arm_medians_seconds": timed_medians,
                "interpretation": "Descriptive arm contrast within this one sequential sweep. It does not identify trace overhead causally or partition GPU execution time."},
            "native_trace_scope": {"traced_report_count": len(trace_runs),
                "event_count_across_traced_arms": trace_aggregate["event_count"],
                "root_overlap_summary_all_events": trace_aggregate,
                "per_traced_arm": per_run_trace,
                "interpretation": "Earlier-root intersections use GPU root timestamps against Mach-host continuation intervals and exclude the current root being committed. Totals sum per-call overlaps, so the same root interval may contribute to multiple nested calls; this is not a union across events or ALU-active time. Callback observations are not exact GPU completion instants. Aggregating both traced arms repeats observations from separate runs; they are not one root timeline."},
            "validated_invariants": {"samples_per_arm": TOTAL_SAMPLES,
                "timed_samples_per_arm": TIMED_SAMPLES, "roots_per_arm": 162,
                "api_graph_encodes_per_arm": 54, "api_continuations_per_arm": 161,
                "api_linears_per_arm": 0, "internal_continuations_per_traced_arm": 125,
                "explicit_boundary_continuations_per_traced_arm": 36,
                "one_terminal_wait_per_arm": True, "encoding_named_allocations_per_arm": 0,
                "encoding_readbacks_per_arm": 0, "postwait_readbacks_per_arm": 1,
                "positive_nan_control_per_arm": True, "root_isolation_recomputed": True},
            "analysis_limit": "The analyzer validates capture counters and native timing/accounting structure. Earlier-root overlap is a clock-domain comparison, not an ALU-active interval or proof of hardware wait attribution; A/A arm differences are descriptive and do not establish optimization headroom."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path,
                        help="one A/A sweep arm report; repeat in OFF/ON/ON/OFF order")
    parser.add_argument("--output", type=Path, help="fresh summary file (exclusive create); defaults to stdout")
    args = parser.parse_args(argv)
    try:
        ORDER.need(len(args.report) == 4, "pass exactly four reports in OFF/ON/ON/OFF order")
        reports, paths, hashes = [], [], []
        for path in args.report:
            raw = path.read_bytes()
            reports.append(json.loads(raw))
            paths.append(str(path))
            hashes.append(hashlib.sha256(raw).hexdigest())
        summary = analyze_reports(reports, paths, hashes)
        encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(encoded)
        else:
            sys.stdout.write(encoded)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"pixelumm FFN A/A trace analysis rejected report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
