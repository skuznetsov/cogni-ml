#!/usr/bin/env python3
"""Validate and summarize a PixelUMM drift probe report (CPU only)."""
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

REPORT = "cogni_ml.pixelumm_drift_probe/v1"
PROFILE = "cogni_ml.pixelumm_drift_profile/v1"
SUMMARY = "cogni_ml.pixelumm_drift_analysis/v1"
GUARD_SWAP_CAP = 64 * 1024 * 1024
GUARD_MAX_FOOTPRINT = 24 * 1024 * 1024 * 1024
GUARD_MAX_TIMEOUT = 900
CAPTURES = ("post_attn_norm", "gate_proj", "up_proj", "product", "down_proj", "output")
PLANS = ("window0/block0", "window0/block1", "window1/block0", "window1/block1")
MARKERS = ("input_norm", "qkv_projection", "qk_norm_rope", "attention", "output_projection",
           "residual_post_norm", "ffn_gate_up", "ffn_activation", "ffn_down", "final_residual", "complete")
PHASES = MARKERS[:-1]
PHASE_LABELS = tuple(f"{plan}/{phase}" for plan in PLANS for phase in PHASES)
API_PHASES = {0: "unassigned", 1: "self_prepare", 2: "self_attention", 3: "self_projection_residual",
              4: "cross_attention_projection_residual", 5: "ffn"}
OPS = ("graph_encode", "commit_and_continue", "t5_linear_encode")
TIMED_PATTERN = (False, True, True, False, True, False, False, True)
HOST_PHASES = ("admission_poison", "encode", "terminal_wait", "profile_extraction", "metrics_release", "critical", "total")
HOST_BOUNDS = {"admission_poison": (1, 2), "encode": (2, 3), "terminal_wait": (3, 4),
               "profile_extraction": (4, 5), "metrics_release": (5, 6), "critical": (1, 4), "total": (0, 6)}
METRIC_FIELDS = ("failures", "different_words", "nonfinite", "elements")
POINT_FIELDS = ("time_ns", "allocated_bytes", "gc_no", "heap_bytes", "free_bytes", "bytes_since_gc",
                "bytes_before_gc", "collection_count", "collection_total_ns", "collection_max_ns",
                "rendezvous_pause_count", "rendezvous_pause_total_ns", "rendezvous_pause_max_ns",
                "stopped_interval_count", "stopped_interval_total_ns", "stopped_interval_max_ns", "unmatched_event_count")
POINT_MONOTONIC = ("time_ns", "allocated_bytes", "gc_no", "collection_count", "collection_total_ns", "collection_max_ns",
                   "rendezvous_pause_count", "rendezvous_pause_total_ns", "rendezvous_pause_max_ns",
                   "stopped_interval_count", "stopped_interval_total_ns", "stopped_interval_max_ns")
GC_FIELDS = ("seconds", "allocated_bytes", "collections", "gc_no_delta", "collection_seconds", "rendezvous_pauses",
             "rendezvous_pause_seconds", "stopped_intervals", "stopped_seconds")
GC_INTS = {"allocated_bytes", "collections", "gc_no_delta", "rendezvous_pauses", "stopped_intervals"}
TRACE_INFO = ("owner_thread_id", "mach_timebase_numer", "mach_timebase_denom", "trace_start_mach_ticks",
              "event_capacity", "event_count", "root_count", "failed")
TRACE_EVENT = ("serial", "begin_mach_ticks", "end_mach_ticks", "begin_thread_cpu_ns", "end_thread_cpu_ns",
               "api_scope_serial", "parent_graph_serial", "before_root_sequence", "after_root_sequence", "phase",
               "graph_depth", "before_root_status", "before_root_status_after_call", "after_root_status", "flags")
TRACE_ROOT = ("sequence", "tracked_or_observed_mach_ticks", "completion_observed_mach_ticks", "gpu_start_time_bits",
              "gpu_end_time_bits", "status_at_track_or_enable", "final_status", "has_error", "completion_source",
              "gpu_start_seconds", "gpu_end_seconds")
API_FIELDS = ("count", "wall_total_ns", "wall_max_ns", "thread_cpu_total_ns", "thread_cpu_max_ns", "nested_count",
              "nested_wall_total_ns", "nested_thread_cpu_total_ns", "exclusive_wall_total_ns", "exclusive_thread_cpu_total_ns")
PROFILE_SEMANTICS = "Candidate sums can count a root for multiple phases. Definite sums use roots with one marker candidate but are still root GPU durations, not exclusive phase work; root intervals may overlap."
GPU_INTERPRETATION = "Marker phases identify CPU-side encode spans. Candidate roots can straddle phases and GPU roots can overlap; no per-phase value is exclusive GPU work."


def need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def mapping(v: object, why: str) -> dict[str, Any]:
    need(isinstance(v, dict), why)
    return v  # type: ignore[return-value]


def sequence(v: object, why: str) -> list[Any]:
    need(isinstance(v, list), why)
    return v  # type: ignore[return-value]


def uint(v: object, why: str, minimum: int = 0) -> int:
    need(isinstance(v, int) and not isinstance(v, bool) and minimum <= v <= (1 << 64) - 1, why)
    return int(v)


def uint32(v: object, why: str, minimum: int = 0) -> int:
    need(isinstance(v, int) and not isinstance(v, bool) and minimum <= v <= (1 << 32) - 1, why)
    return int(v)


def finite(v: object, why: str, minimum: float = 0.0) -> float:
    need(isinstance(v, (int, float)) and not isinstance(v, bool), why)
    try:
        x = float(v)
    except (OverflowError, ValueError) as exc:
        raise ValueError(why) from exc
    need(math.isfinite(x) and x >= minimum, why)
    return x


def near(a: float, b: float, atol: float = 2e-9, rtol: float = 1e-9) -> bool:
    return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def exact_keys(v: dict[str, Any], keys: tuple[str, ...] | set[str], why: str) -> None:
    need(set(v) == set(keys), f"{why}: missing or unexpected fields")


def metric(v: object, where: str) -> dict[str, int]:
    raw = mapping(v, where)
    exact_keys(raw, METRIC_FIELDS, where)
    return {k: uint32(raw[k], f"{where}.{k}") for k in METRIC_FIELDS}


def validate_metrics(req: dict[str, Any], i: int, kind: str) -> None:
    raw = sequence(req.get("numerical_metrics"), f"request {i}: missing numerical_metrics")
    need(len(raw) == 29, f"request {i}: expected 29 metric slots")
    ms = [metric(v, f"request {i} metric {n}") for n, v in enumerate(raw)]
    zero = dict.fromkeys(METRIC_FIELDS, 0)
    need(ms[:4] == [zero] * 4, f"request {i}: inactive metric slots 0..3 are nonzero")
    for plan in range(4):
        for si, stage in enumerate(CAPTURES):
            slot = 4 + plan * 6 + si
            m = ms[slot]
            expected = 5760 * (12288 if stage in ("gate_proj", "up_proj", "product") else 4096)
            if kind == "correctness" or stage == "output":
                need(m["elements"] == expected and m["failures"] == 0 and m["nonfinite"] == 0,
                     f"request {i}: numerical comparison failed in slot {slot}")
                # different_words is retained as evidence; tolerated differences are not failures.
            else:
                need(m == zero, f"request {i}: output-only request touched intermediate slot {slot}")
    control = ms[28]
    if i == 0:
        need(control == {"failures": 1, "different_words": 0, "nonfinite": 1, "elements": 5760 * 4096},
             "request 0: positive NaN control did not match expected single failure")
    else:
        need(control["elements"] == control["failures"] == control["nonfinite"] == 0,
             f"request {i}: positive-control slot is not clear")
    capture = req.get("capture_metrics")
    if kind == "correctness":
        values = sequence(capture, f"request {i}: missing capture_metrics")
        need(len(values) == 29 and [metric(v, f"request {i} captured metric") for v in values] == ms,
             f"request {i}: capture_metrics differ from full readback")
    else:
        need(capture is None, f"request {i}: unexpected capture_metrics")


def validate_gc(req: dict[str, Any], i: int) -> tuple[list[dict[str, int]], dict[str, Any]]:
    raw = sequence(req.get("raw_points"), f"request {i}: missing raw_points")
    need(len(raw) == 7, f"request {i}: expected seven GC points")
    points = []
    for j, value in enumerate(raw):
        p = mapping(value, f"request {i} point {j}")
        exact_keys(p, POINT_FIELDS, f"request {i} point {j}")
        q = {k: uint(p[k], f"request {i} point {j}.{k}") for k in POINT_FIELDS}
        need(q["unmatched_event_count"] == 0, f"request {i}: unmatched GC observer event")
        if points:
            need(all(q[k] >= points[-1][k] for k in POINT_MONOTONIC), f"request {i}: GC counter moved backwards")
        points.append(q)
    a, b = points[0], points[-1]
    expected = {"seconds": (b["time_ns"] - a["time_ns"]) / 1e9,
                "allocated_bytes": b["allocated_bytes"] - a["allocated_bytes"],
                "collections": b["collection_count"] - a["collection_count"],
                "gc_no_delta": b["gc_no"] - a["gc_no"],
                "collection_seconds": (b["collection_total_ns"] - a["collection_total_ns"]) / 1e9,
                "rendezvous_pauses": b["rendezvous_pause_count"] - a["rendezvous_pause_count"],
                "rendezvous_pause_seconds": (b["rendezvous_pause_total_ns"] - a["rendezvous_pause_total_ns"]) / 1e9,
                "stopped_intervals": b["stopped_interval_count"] - a["stopped_interval_count"],
                "stopped_seconds": (b["stopped_interval_total_ns"] - a["stopped_interval_total_ns"]) / 1e9}
    delta = mapping(req.get("gc"), f"request {i}: missing GC delta")
    exact_keys(delta, GC_FIELDS, f"request {i} GC delta")
    parsed: dict[str, int | float] = {}
    for k in GC_FIELDS:
        if k in GC_INTS:
            parsed[k] = uint(delta[k], f"request {i} gc.{k}")
            need(parsed[k] == expected[k], f"request {i}: gc.{k} differs from raw points")
        else:
            parsed[k] = finite(delta[k], f"request {i} gc.{k}")
            need(near(float(parsed[k]), float(expected[k]), 1e-12), f"request {i}: gc.{k} differs from raw points")
    return points, parsed


def validate_host(req: dict[str, Any], i: int) -> tuple[list[dict[str, int]], dict[str, Any]]:
    raw = sequence(req.get("host_points"), f"request {i}: missing host_points")
    need(len(raw) == 7, f"request {i}: expected seven host points")
    points = []
    for j, value in enumerate(raw):
        p = mapping(value, f"request {i} host point {j}")
        exact_keys(p, ("wall_ns", "thread_cpu_ns", "thermal_state"), f"request {i} host point {j}")
        q = {k: uint(p[k], f"request {i} host point {j}.{k}") for k in ("wall_ns", "thread_cpu_ns", "thermal_state")}
        need(q["wall_ns"] > 0 and q["thread_cpu_ns"] > 0 and q["thermal_state"] <= 3,
             f"request {i}: invalid host clock/advisory thermal value")
        if points:
            need(q["wall_ns"] >= points[-1]["wall_ns"] and q["thread_cpu_ns"] >= points[-1]["thread_cpu_ns"],
                 f"request {i}: host clock moved backwards")
        points.append(q)
    raw_phases = mapping(req.get("phases"), f"request {i}: missing phases")
    exact_keys(raw_phases, HOST_PHASES, f"request {i} phases")
    phases: dict[str, Any] = {}
    for name, (a, b) in HOST_BOUNDS.items():
        p = mapping(raw_phases[name], f"request {i} phase {name}")
        exact_keys(p, ("wall_seconds", "thread_cpu_seconds"), f"request {i} phase {name}")
        wall = finite(p["wall_seconds"], f"request {i} phase {name} wall")
        cpu = finite(p["thread_cpu_seconds"], f"request {i} phase {name} CPU")
        need(near(wall, (points[b]["wall_ns"] - points[a]["wall_ns"]) / 1e9, 1e-12) and
             near(cpu, (points[b]["thread_cpu_ns"] - points[a]["thread_cpu_ns"]) / 1e9, 1e-12),
             f"request {i}: phase {name} disagrees with host boundary points")
        phases[name] = {"wall_seconds": wall, "thread_cpu_seconds": cpu}
    return points, phases


def interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for a, b in ordered[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + end - start


def validate_profile(profile: dict[str, Any], index: int) -> dict[str, Any]:
    need(profile.get("schema") == PROFILE, f"request {index}: wrong profile schema")
    need(profile.get("root_candidate_duration_semantics") == PROFILE_SEMANTICS and
         profile.get("gpu_phase_interpretation") == GPU_INTERPRETATION,
         f"request {index}: profile semantics missing/changed")
    roots_raw = sequence(profile.get("roots"), f"request {index}: missing profile roots")
    need(len(roots_raw) > 1, f"request {index}: profile has too few roots")
    plans = sequence(profile.get("markers_by_plan"), f"request {index}: missing marker plans")
    need(len(plans) == 4, f"request {index}: expected four marker plans")
    candidates: list[list[str]] = [[] for _ in roots_raw]
    calculated_wall: dict[str, float] = {}
    calculated_cpu: dict[str, float] = {}
    parsed_plans = []
    cursor = 1
    for pi, item in enumerate(plans):
        plan = mapping(item, f"request {index}: malformed plan {pi}")
        need(plan.get("plan") == PLANS[pi], f"request {index}: marker plan order changed")
        marks = sequence(plan.get("markers"), f"request {index}: missing plan markers")
        need(len(marks) == len(MARKERS), f"request {index}: incomplete marker plan")
        parsed = []
        prev_wall = prev_cpu = 0.0
        prev_root = cursor
        for mi, raw in enumerate(marks):
            m = mapping(raw, f"request {index}: malformed marker")
            exact_keys(m, ("name", "elapsed_seconds", "thread_cpu_seconds", "root_count"), "marker")
            need(m.get("name") == MARKERS[mi], f"request {index}: marker names/order changed")
            wall = finite(m["elapsed_seconds"], "invalid marker wall")
            cpu = finite(m["thread_cpu_seconds"], "invalid marker CPU")
            root = uint(m["root_count"], "invalid marker root_count", 1)
            need(root <= (1 << 31) - 1, f"request {index}: marker root_count exceeds Int32")
            need(wall >= prev_wall and cpu >= prev_cpu and prev_root <= root <= len(roots_raw),
                 f"request {index}: marker clock/root cursor moved backwards")
            if mi == 0:
                need(wall == cpu == 0.0 and root == cursor, f"request {index}: plan start does not align")
            parsed.append((m["name"], wall, cpu, root))
            prev_wall, prev_cpu, prev_root = wall, cpu, root
        for a, b in zip(parsed, parsed[1:]):
            label = f"{PLANS[pi]}/{a[0]}"
            calculated_wall[label] = b[1] - a[1]
            calculated_cpu[label] = b[2] - a[2]
            for ri in range(a[3] - 1, b[3]):
                candidates[ri].append(label)
        cursor = parsed[-1][3]
        parsed_plans.append(parsed)
    need(cursor == len(roots_raw), f"request {index}: last marker does not cover final root")
    candidates[0].append("request_prelude")
    for ri in range(parsed_plans[-1][-1][3] - 1, len(roots_raw)):
        candidates[ri].append("request_final_comparators")
    need(all(candidates), f"request {index}: profile contains unattributed root")

    for key, calculated in (("phase_wall_seconds", calculated_wall), ("phase_thread_cpu_seconds", calculated_cpu)):
        values = mapping(profile.get(key), f"request {index}: missing {key}")
        need(set(values) == set(PHASE_LABELS), f"request {index}: incomplete {key}")
        for label, expect in calculated.items():
            need(near(finite(values[label], f"{key}.{label}"), expect), f"request {index}: {key} mismatch at {label}")

    intervals = []
    candidate_sums = dict.fromkeys(PHASE_LABELS, 0.0)
    definite_sums = dict.fromkeys(PHASE_LABELS, 0.0)
    root_sum = 0.0
    trace = mapping(profile.get("continuation_trace"), f"request {index}: missing continuation trace")
    trace_roots = sequence(trace.get("roots"), f"request {index}: missing trace roots")
    need(len(trace_roots) == len(roots_raw), f"request {index}: profile/trace root counts differ")
    for ri, raw in enumerate(roots_raw):
        r = mapping(raw, f"request {index}: malformed profile root {ri}")
        exact_keys(r, ("sequence", "start_seconds", "end_seconds", "candidates"), "profile root")
        need(uint(r["sequence"], "invalid root sequence") == ri, f"request {index}: root order changed")
        start = finite(r["start_seconds"], "invalid root start")
        end = finite(r["end_seconds"], "invalid root end")
        need(end > start, f"request {index}: nonpositive root duration")
        labels = sequence(r["candidates"], "invalid root candidate labels")
        need(labels == candidates[ri], f"request {index}: root {ri} candidate labels disagree with markers")
        tr = mapping(trace_roots[ri], "invalid trace root")
        need("gpu_start_seconds" in tr and "gpu_end_seconds" in tr and
             near(start, finite(tr["gpu_start_seconds"], "invalid trace root start"), 2e-6, 0.0) and
             near(end, finite(tr["gpu_end_seconds"], "invalid trace root end"), 2e-6, 0.0),
             f"request {index}: root {ri} disagrees with continuation trace")
        intervals.append((start, end))
        duration = end - start
        root_sum += duration
        phase_labels = [label for label in labels if label in candidate_sums]
        for label in phase_labels:
            candidate_sums[label] += duration
        if len(phase_labels) == 1:
            definite_sums[phase_labels[0]] += duration
    for field, expected in (("phase_candidate_root_duration_sum_seconds", candidate_sums),
                            ("phase_definite_root_duration_sum_seconds", definite_sums)):
        values = mapping(profile.get(field), f"request {index}: missing {field}")
        need(set(values) == set(PHASE_LABELS), f"request {index}: incomplete {field}")
        for label, want in expected.items():
            need(near(finite(values[label], f"{field}.{label}"), want), f"request {index}: {field} mismatch")

    union = interval_union(intervals)
    envelope = max(b for _, b in intervals) - min(a for a, _ in intervals)
    rootp = mapping(profile.get("root_profile"), f"request {index}: missing root profile")
    exact_keys(rootp, ("root_count", "root_duration_seconds", "envelope_seconds", "union_seconds",
                       "overlap_seconds", "gap_seconds", "phase_union_seconds"), "root_profile")
    need(uint(rootp["root_count"], "invalid root count") == len(intervals), f"request {index}: root count mismatch")
    for name, expected in (("root_duration_seconds", root_sum), ("envelope_seconds", envelope),
                           ("union_seconds", union), ("overlap_seconds", max(0.0, root_sum - union)),
                           ("gap_seconds", max(0.0, envelope - union))):
        need(near(finite(rootp[name], f"root_profile.{name}"), expected), f"request {index}: root {name} mismatch")
    phase_union = mapping(rootp["phase_union_seconds"], "missing root phase union")
    need(set(phase_union) == {"whole_chain"} and near(finite(phase_union["whole_chain"], "invalid whole-chain union"), union),
         f"request {index}: whole-chain interval union mismatch")

    parsed_trace = validate_trace(trace, index, intervals)
    api = validate_api(profile, parsed_trace, index)
    return {"intervals": intervals, "root_duration_seconds": root_sum, "union_seconds": union,
            "envelope_seconds": envelope, "overlap_seconds": max(0.0, root_sum - union),
            "gap_seconds": max(0.0, envelope - union), "trace": parsed_trace, "api": api,
            "profile": profile}


def validate_trace(trace: dict[str, Any], index: int, intervals: list[tuple[float, float]]) -> dict[str, Any]:
    info = mapping(trace.get("info"), f"request {index}: missing trace info")
    exact_keys(info, TRACE_INFO, "trace info")
    inf = {k: uint(info[k], f"request {index}: trace info {k}") for k in TRACE_INFO}
    need(inf["owner_thread_id"] > 0 and inf["mach_timebase_numer"] > 0 and inf["mach_timebase_denom"] > 0 and
         inf["trace_start_mach_ticks"] > 0 and inf["event_capacity"] == 8192 and inf["failed"] == 0,
         f"request {index}: invalid trace clock/capacity/failure metadata")
    events = sequence(trace.get("events"), f"request {index}: missing trace events")
    roots = sequence(trace.get("roots"), f"request {index}: missing trace roots")
    need(inf["event_count"] == len(events) == len(roots) - 1 and inf["root_count"] == len(roots) == len(intervals) and
         0 < inf["event_count"] <= inf["event_capacity"], f"request {index}: trace counts do not reconcile")
    def host_seconds(t: int) -> float:
        return t * inf["mach_timebase_numer"] / inf["mach_timebase_denom"] / 1e9
    out_roots = []
    for i, raw in enumerate(roots):
        r = mapping(raw, f"request {index}: malformed trace root {i}")
        exact_keys(r, TRACE_ROOT, f"trace root {i}")
        v = {k: uint(r[k], f"trace root {i}.{k}") for k in TRACE_ROOT if k not in ("gpu_start_seconds", "gpu_end_seconds")}
        for key in ("status_at_track_or_enable", "final_status", "has_error", "completion_source"):
            uint32(r[key], f"trace root {i}.{key}")
        start = finite(r["gpu_start_seconds"], "invalid trace root start", 0.0)
        end = finite(r["gpu_end_seconds"], "invalid trace root end", 0.0)
        ds = struct.unpack("=d", struct.pack("=Q", v["gpu_start_time_bits"]))[0]
        de = struct.unpack("=d", struct.pack("=Q", v["gpu_end_time_bits"]))[0]
        need(v["sequence"] == i and v["final_status"] == 4 and v["has_error"] == 0 and
             v["completion_source"] in (1, 2) and v["status_at_track_or_enable"] <= 1 and
             v["tracked_or_observed_mach_ticks"] >= inf["trace_start_mach_ticks"] and
             v["completion_observed_mach_ticks"] >= v["tracked_or_observed_mach_ticks"] and end > start > 0 and
             math.isfinite(ds) and math.isfinite(de) and near(ds, start, 1e-12, 0.0) and near(de, end, 1e-12, 0.0) and
             host_seconds(v["tracked_or_observed_mach_ticks"]) <= start + 2e-6 and
             end <= host_seconds(v["completion_observed_mach_ticks"]) + 2e-6,
             f"request {index}: invalid trace root {i} status/time bounds")
        out_roots.append({**v, "start_seconds": start, "end_seconds": end})
    last_end = inf["trace_start_mach_ticks"]
    last_scope = 0
    phase_counts: Counter[int] = Counter()
    out_events = []
    for i, raw in enumerate(events):
        e0 = mapping(raw, f"request {index}: malformed event {i}")
        exact_keys(e0, TRACE_EVENT, f"trace event {i}")
        e = {k: uint(e0[k], f"trace event {i}.{k}") for k in TRACE_EVENT}
        for key in ("phase", "graph_depth", "before_root_status", "before_root_status_after_call",
                    "after_root_status", "flags"):
            uint32(e0[key], f"trace event {i}.{key}")
        nested = e["graph_depth"] > 0
        need(e["serial"] == i + 1 and e["before_root_sequence"] == i and e["after_root_sequence"] == i + 1 and
             e["api_scope_serial"] > last_scope and e["begin_mach_ticks"] >= last_end and
             e["end_mach_ticks"] > e["begin_mach_ticks"] and e["begin_thread_cpu_ns"] > 0 and
             e["end_thread_cpu_ns"] >= e["begin_thread_cpu_ns"] and e["phase"] in API_PHASES and
             e["graph_depth"] <= 16 and e["flags"] == (15 if nested else 14) and
             ((0 < e["parent_graph_serial"] < e["api_scope_serial"]) if nested else e["parent_graph_serial"] == 0) and
             e["before_root_status"] <= 1 and 2 <= e["before_root_status_after_call"] <= 4 and
             e["after_root_status"] <= 1 and out_roots[i]["tracked_or_observed_mach_ticks"] <= e["begin_mach_ticks"] and
             out_roots[i + 1]["tracked_or_observed_mach_ticks"] >= e["end_mach_ticks"] and
             out_roots[i]["start_seconds"] + 2e-6 >= host_seconds(e["begin_mach_ticks"]),
             f"request {index}: invalid continuation event {i}")
        last_end, last_scope = e["end_mach_ticks"], e["api_scope_serial"]
        phase_counts[e["phase"]] += 1
        out_events.append(e)
    return {"info": inf, "events": out_events, "roots": out_roots,
            "phase_counts": phase_counts, "host_seconds": host_seconds}


def validate_api(profile: dict[str, Any], trace: dict[str, Any], index: int) -> dict[int, dict[str, dict[str, int]]]:
    api = mapping(profile.get("api_profile"), f"request {index}: missing API profile")
    need(api.get("source_schema") == "MPSAPIProfile WAN phase IDs", f"request {index}: API source schema changed")
    phases = sequence(api.get("phases"), f"request {index}: missing API phases")
    need(len(phases) == 6, f"request {index}: expected six API phases")
    indexed: dict[int, dict[str, dict[str, int]]] = {}
    for raw in phases:
        p = mapping(raw, "malformed API phase")
        pid = uint(p.get("source_phase_id"), "invalid API phase ID")
        need(pid in API_PHASES and pid not in indexed and p.get("source_phase_name") == API_PHASES[pid],
             f"request {index}: API phase ID/name mismatch")
        ops = mapping(p.get("operations"), "missing API operations")
        need(set(ops) == set(OPS), f"request {index}: incomplete API operations")
        parsed = {}
        for op in OPS:
            raw_stats = mapping(ops[op], f"invalid {op} stats")
            exact_keys(raw_stats, API_FIELDS, op)
            s = {k: uint(raw_stats[k], f"{op}.{k}") for k in API_FIELDS}
            if s["count"] == 0:
                need(all(v == 0 for v in s.values()), f"request {index}: empty {op} has nonzero fields")
            else:
                need(0 < s["wall_max_ns"] <= s["wall_total_ns"] and
                     0 < s["thread_cpu_max_ns"] <= s["thread_cpu_total_ns"] and
                     (op == "graph_encode" or s["nested_count"] <= s["count"]) and
                     s["nested_wall_total_ns"] <= s["wall_total_ns"] and
                     s["nested_thread_cpu_total_ns"] <= s["thread_cpu_total_ns"] and
                     ((s["nested_count"] > 0) == (s["nested_wall_total_ns"] > 0) ==
                      (s["nested_thread_cpu_total_ns"] > 0)), f"request {index}: inconsistent {op} totals")
            if op == "graph_encode":
                need(s["exclusive_wall_total_ns"] == s["wall_total_ns"] - s["nested_wall_total_ns"] and
                     s["exclusive_thread_cpu_total_ns"] == s["thread_cpu_total_ns"] - s["nested_thread_cpu_total_ns"],
                     f"request {index}: graph exclusive totals inconsistent")
            elif op == "commit_and_continue":
                need(s["exclusive_wall_total_ns"] == s["wall_total_ns"] and
                     s["exclusive_thread_cpu_total_ns"] == s["thread_cpu_total_ns"],
                     f"request {index}: commit totals inconsistent")
            else:
                need(s["nested_count"] == s["nested_wall_total_ns"] == s["nested_thread_cpu_total_ns"] == 0 and
                     s["exclusive_wall_total_ns"] == s["wall_total_ns"] and
                     s["exclusive_thread_cpu_total_ns"] == s["thread_cpu_total_ns"],
                     f"request {index}: unexpected linear nesting")
            parsed[op] = s
        g, c = parsed["graph_encode"], parsed["commit_and_continue"]
        need(tuple(g[k] for k in ("nested_count", "nested_wall_total_ns", "nested_thread_cpu_total_ns")) ==
             tuple(c[k] for k in ("nested_count", "nested_wall_total_ns", "nested_thread_cpu_total_ns")),
             f"request {index}: graph/commit nested totals differ")
        indexed[pid] = parsed
    need(set(indexed) == set(API_PHASES), f"request {index}: missing API phase")
    gc = {p: indexed[p]["graph_encode"]["count"] for p in API_PHASES}
    need(gc[0] == 0 and gc[1] == 36 and gc[2] > 0 and gc[2] % 12 == 0 and gc[3] == 12 and gc[4] == 0 and
         gc[5] == 12 and sum(gc.values()) == 60 + gc[2] and
         all(indexed[p]["t5_linear_encode"]["count"] == 0 for p in API_PHASES),
         f"request {index}: API Graph/linear counts disagree with four plans")
    for pid in API_PHASES:
        calls = [e for e in trace["events"] if e["phase"] == pid]
        commit = indexed[pid]["commit_and_continue"]
        need(commit["count"] == len(calls) and commit["nested_count"] == sum(e["graph_depth"] > 0 for e in calls),
             f"request {index}: commit count/nesting differs from continuation events")
        wall_ns = sum((e["end_mach_ticks"] - e["begin_mach_ticks"]) * trace["info"]["mach_timebase_numer"] /
                      trace["info"]["mach_timebase_denom"] for e in calls)
        cpu_ns = sum(e["end_thread_cpu_ns"] - e["begin_thread_cpu_ns"] for e in calls)
        need(wall_ns <= commit["wall_total_ns"] + len(calls) * 2000 and cpu_ns <= commit["thread_cpu_total_ns"],
             f"request {index}: continuation events exceed aggregate API timing")
    return indexed


def validate_request(req: dict[str, Any], i: int, kind: str, enabled: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    need(uint(req.get("index"), f"request {i}: invalid index") == i, f"request order/index mismatch at {i}")
    need(req.get("kind") == kind and req.get("profile_enabled") is enabled, f"request {i}: schedule changed")
    for key, value in (("named_buffer_allocations", 0), ("terminal_wait_count", 1), ("preterminal_readbacks", 0),
                       ("comparator_readbacks_after_wait", 1), ("comparator_readback_bytes", 464)):
        need(uint(req.get(key), f"request {i}: invalid {key}") == value, f"request {i}: {key} must equal {value}")
    for key in ("body_encode_seconds", "comparator_encode_seconds", "gpu_elapsed_seconds"):
        finite(req.get(key), f"request {i}: invalid {key}")
    host_points, phases = validate_host(req, i)
    need(float(req["body_encode_seconds"]) + float(req["comparator_encode_seconds"]) <=
         phases["encode"]["wall_seconds"] + 0.002, f"request {i}: encode timers exceed host encode span")
    validate_metrics(req, i, kind)
    gc_points, gc_delta = validate_gc(req, i)
    root_count = uint(req.get("root_count"), f"request {i}: invalid root_count", 1)
    if enabled:
        raw_profile = mapping(req.get("profile"), f"request {i}: missing enabled profile")
        parsed = validate_profile(raw_profile, i)
        need(root_count == len(parsed["intervals"]), f"request {i}: root_count differs from profile")
        need(near(float(req["gpu_elapsed_seconds"]), parsed["root_duration_seconds"], 2e-8),
             f"request {i}: GPU elapsed value differs from root-duration sum")
    else:
        need(req.get("profile") is None, f"request {i}: disabled profile contains data")
        parsed = None
    return {**req, "phases": phases, "gc": gc_delta, "host_points": host_points,
            "raw_points": gc_points, "root_count": root_count}, parsed


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "min": None, "max": None}
    return {"count": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def paired_analysis(timed: list[dict[str, Any]], metrics: list[tuple[str, Any]]) -> dict[str, Any]:
    pair_rows = []
    changes: dict[str, list[float]] = defaultdict(list)
    for pi in range(len(timed) // 2):
        a, b = timed[pi * 2:pi * 2 + 2]
        need(a["profile_enabled"] is not b["profile_enabled"], f"timed adjacent pair {pi} is not on/off")
        on, off = (a, b) if a["profile_enabled"] else (b, a)
        row = {"pair": pi, "on_index": on["index"], "off_index": off["index"], "metrics": {}}
        for name, fn in metrics:
            on_value, off_value = float(fn(on)), float(fn(off))
            change = (on_value / off_value - 1.0) * 100.0 if off_value else 0.0
            row["metrics"][name] = {"on": on_value, "off": off_value, "on_vs_off_percent": change}
            changes[name].append(change)
        pair_rows.append(row)
    return {"pair_count": len(pair_rows), "pairs": pair_rows,
            "metrics": {name: {"median_on_vs_off_percent": statistics.median(values),
                               "wins_on_lower": sum(x < 0 for x in values), "ties": sum(x == 0 for x in values),
                               "wins_off_lower": sum(x > 0 for x in values), "range_percent": [min(values), max(values)]}
                        for name, values in changes.items()}}


def drift_analysis(timed: list[dict[str, Any]], metrics: list[tuple[str, Any]]) -> dict[str, Any]:
    bins = []
    for start in range(0, len(timed), 8):
        rows = timed[start:start + 8]
        arms = {}
        for enabled, label in ((False, "profile_off"), (True, "profile_on")):
            arm = [r for r in rows if r["profile_enabled"] is enabled]
            arms[label] = {"count": len(arm), "metrics": {name: distribution([float(fn(r)) for r in arm])
                                                              for name, fn in metrics}}
        bins.append({"timed_ordinal_range": [start, start + len(rows) - 1], "arms": arms})
    result: dict[str, Any] = {"bin_size_requests": 8, "bins": bins, "early_late": {}, "peak_from_first_bin": {}}
    early, late = timed[:16], timed[-16:]
    for enabled, label in ((False, "profile_off"), (True, "profile_on")):
        ea = [r for r in early if r["profile_enabled"] is enabled]
        la = [r for r in late if r["profile_enabled"] is enabled]
        result["early_late"][label] = {}
        result["peak_from_first_bin"][label] = {}
        for name, fn in metrics:
            em = statistics.median(float(fn(r)) for r in ea)
            lm = statistics.median(float(fn(r)) for r in la)
            result["early_late"][label][name] = {"early_count": len(ea), "late_count": len(la),
                "early_median": em, "late_median": lm, "late_vs_early_percent": (lm / em - 1) * 100 if em else 0.0}
            ref = bins[0]["arms"][label]["metrics"][name]["median"]
            peaks = [(j, b["arms"][label]["metrics"][name]["median"]) for j, b in enumerate(bins)]
            j, peak = max(peaks, key=lambda x: abs((x[1] / ref - 1) * 100 if ref else 0.0))
            result["peak_from_first_bin"][label][name] = {"reference_median": ref, "peak_bin": j,
                "peak_range": bins[j]["timed_ordinal_range"], "peak_median": peak,
                "peak_vs_first_percent": (peak / ref - 1) * 100 if ref else 0.0}
    return result


def continuation_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wall = sum(row["wall_seconds"] for row in rows)
    overlap = sum(row["earlier_root_overlap_seconds"] for row in rows)
    need(overlap <= wall + 1e-9, "earlier-root overlap exceeds continuation wall time")
    return {"event_count": len(rows), "total_wall_seconds": wall,
            "total_thread_cpu_seconds": sum(row["thread_cpu_seconds"] for row in rows),
            "total_earlier_root_overlap_seconds": overlap,
            "wall_weighted_earlier_root_overlap_fraction": overlap / wall if wall > 0 else None}


def profile_aggregates(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    phase_values = {label: {key: [] for key in ("wall_seconds", "thread_cpu_seconds",
                      "candidate_root_duration_sum_seconds", "definite_root_duration_sum_seconds")}
                    for label in PHASE_LABELS}
    root_values = {key: [] for key in ("root_duration_seconds", "union_seconds", "envelope_seconds", "overlap_seconds", "gap_seconds")}
    api_totals: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    continuations = []
    for req, parsed in rows:
        p = parsed["profile"]
        for source, target in (("phase_wall_seconds", "wall_seconds"),
                               ("phase_thread_cpu_seconds", "thread_cpu_seconds"),
                               ("phase_candidate_root_duration_sum_seconds", "candidate_root_duration_sum_seconds"),
                               ("phase_definite_root_duration_sum_seconds", "definite_root_duration_sum_seconds")):
            for label, value in p[source].items():
                phase_values[label][target].append(float(value))
        for key in root_values:
            root_values[key].append(float(parsed[key]))
        for pid, operations in parsed["api"].items():
            for op, stats in operations.items():
                for field, value in stats.items():
                    api_totals[API_PHASES[pid]][op][field] += value
        trace = parsed["trace"]
        hs = trace["host_seconds"]
        roots = trace["roots"]
        for e in trace["events"]:
            begin, end = hs(e["begin_mach_ticks"]), hs(e["end_mach_ticks"])
            prior = roots[:e["before_root_sequence"]]  # current submitted root is intentionally excluded
            intersections = [(max(begin, r["start_seconds"]), min(end, r["end_seconds"]))
                             for r in prior if r["start_seconds"] < end and r["end_seconds"] > begin]
            intersections = [(a, b) for a, b in intersections if b > a]
            overlap = interval_union(intersections)
            continuations.append({"request_index": req["index"], "event_serial": e["serial"],
                "source_phase": API_PHASES[e["phase"]], "graph_depth": e["graph_depth"],
                "wall_seconds": end - begin,
                "thread_cpu_seconds": (e["end_thread_cpu_ns"] - e["begin_thread_cpu_ns"]) / 1e9,
                "earlier_root_overlap_seconds": overlap,
                "earlier_root_overlap_fraction_of_call": overlap / (end - begin),
                "earlier_roots_overlapping": len(intersections),
                "earlier_root_completions_observed_during_call": sum(
                    begin <= hs(r["completion_observed_mach_ticks"]) <= end for r in prior),
                "completion_count_interpretation": "callback/status observation during call, not GPU completion instant"})
    api_summary = {phase: {op: dict(api_totals[phase][op]) for op in OPS} for phase in API_PHASES.values()}
    by_phase = {}
    for phase in API_PHASES.values():
        subset = [r for r in continuations if r["source_phase"] == phase]
        by_phase[phase] = {**continuation_totals(subset),
                           "wall_seconds": distribution([r["wall_seconds"] for r in subset]),
                           "thread_cpu_seconds": distribution([r["thread_cpu_seconds"] for r in subset]),
                           "earlier_root_overlap_fraction": distribution([r["earlier_root_overlap_fraction_of_call"] for r in subset])}
    return {"profiled_request_count": len(rows),
        "phase_host_and_root_candidate_metrics": {label: {k: distribution(v) for k, v in vals.items()}
                                                   for label, vals in phase_values.items()},
        "whole_chain_root_interval_metrics": {k: distribution(v) for k, v in root_values.items()},
        "root_sum_semantics": "Whole-chain root-duration sum, interval union, envelope, overlap, and gap are recomputed from command-root intervals. Per-phase candidate/definite root sums can overlap or duplicate and are not additive GPU work.",
        "api_profile_aggregate_ns": api_summary,
        "api_semantics": "graph_encode totals include nested commit calls; graph exclusive totals subtract nested time. commit_and_continue inclusive and nested values are shown separately. Do not add inclusive graph and commit totals.",
        "continuations": {"event_count": len(continuations), "by_phase": by_phase, "events": continuations,
            "backlog_semantics": "For event before_root_sequence=i, overlap uses only roots with sequence < i. Root i being submitted is excluded. Completion observations are not exact GPU completion instants."}}


def validate_guard(events: object, digest: str | None = None) -> dict[str, Any]:
    records = sequence(events, "guard log must be a JSONL record array")
    need(len(records) >= 5, "guard log is missing limits, start, samples, or finish")
    rows = [mapping(value, f"guard event {i}") for i, value in enumerate(records)]
    need(rows[0].get("event") == "limits" and rows[-1].get("event") == "finished",
         "guard log must begin with limits and end with finished")
    limits = rows[0]
    need(limits.get("memory_accounting") == "file-cache" and
         limits.get("headroom_formula") == "printed free + file-backed (includes speculative)" and
         limits.get("runtime_floor_pct_exact") == "10" and
         finite(limits.get("runtime_floor_pct"), "guard runtime floor") == 10.0,
         "guard memory policy differs from the 10 percent file-cache policy")
    footprint_cap = uint(limits.get("max_footprint_bytes"), "guard footprint cap", minimum=1)
    swap_cap = uint(limits.get("max_swap_growth_bytes"), "guard swap cap")
    timeout = uint(limits.get("timeout_sec"), "guard timeout", minimum=1)
    need(footprint_cap <= GUARD_MAX_FOOTPRINT and swap_cap == GUARD_SWAP_CAP and timeout <= GUARD_MAX_TIMEOUT,
         "guard resource caps exceed the permitted probe limits")
    need(limits.get("startup_rule") == "memory_pressure -Q >= 10% and selected headroom >= 10% physical + footprint budget",
         "guard startup rule differs from the 10 percent floor plus footprint budget")
    samples = [row for row in rows if row.get("event") == "sample"]
    starts = [row for row in rows if row.get("event") == "started"]
    finishes = [row for row in rows if row.get("event") == "finished"]
    need(len(samples) >= 2 and len(starts) == 1 and len(finishes) == 1 and
         all(row.get("event") in {"limits", "sample", "started", "finished"} for row in rows),
         "guard lifecycle has missing samples, duplicate process events, or a failure event")
    need(rows[1].get("event") == "sample" and rows[2].get("event") == "started" and
         rows[-2].get("event") == "sample", "guard startup/runtime sampling lifecycle is incomplete")
    process_id = uint(starts[0].get("pid"), "guard process id", minimum=1)
    need(uint(starts[0].get("pgid"), "guard process group", minimum=1) == process_id,
         "guard process group differs from started process")
    need(uint(finishes[0].get("returncode"), "guard return code") == 0,
         "guarded probe did not exit successfully")

    previous_time = finite(limits.get("monotonic"), "guard limits time")
    for i, row in enumerate(rows[1:], start=1):
        now = finite(row.get("monotonic"), f"guard event {i} time")
        need(now > previous_time, "guard event times are not strictly increasing")
        previous_time = now
    initial_swap: int | None = None
    physical_bytes: set[int] = set()
    peaks: list[int] = []
    headrooms: list[int] = []
    pressures: list[float] = []
    swaps: list[int] = []
    for i, row in enumerate(samples):
        label = f"guard sample {i}"
        physical = uint(row.get("physical_bytes"), f"{label} physical bytes", minimum=1)
        headroom = uint(row.get("headroom_bytes"), f"{label} headroom")
        pressure = finite(row.get("pressure_free_pct"), f"{label} pressure", minimum=10.0)
        free_spec = uint(row.get("free_spec_bytes"), f"{label} free/speculative bytes")
        file_backed = uint(row.get("file_backed_bytes"), f"{label} file-backed bytes")
        speculative = uint(row.get("speculative_bytes"), f"{label} speculative bytes")
        footprint = uint(row.get("group_footprint_bytes"), f"{label} footprint")
        swap = uint(row.get("swap_used_bytes"), f"{label} swap")
        uint(row.get("group_live_processes"), f"{label} process count")
        need(pressure <= 100 and headroom >= math.ceil(physical * 0.10) and headroom <= physical,
             f"{label} crossed the 10 percent memory floor")
        need(free_spec <= physical and speculative <= free_spec and file_backed >= speculative and
             headroom == free_spec - speculative + file_backed,
             f"{label} file-cache headroom fields are inconsistent")
        need(footprint <= footprint_cap, f"{label} exceeded the process footprint cap")
        physical_bytes.add(physical)
        peaks.append(footprint)
        headrooms.append(headroom)
        pressures.append(pressure)
        swaps.append(swap)
        if initial_swap is None:
            initial_swap = swap
            need(headroom >= math.ceil(physical * 0.10) + footprint_cap,
                 "guard startup sample lacks floor plus footprint headroom")
        need(swap - initial_swap <= swap_cap, f"{label} exceeded the swap-growth cap")
    need(len(physical_bytes) == 1, "guard physical-memory size changed during the run")
    return {"sha256": digest, "sample_count": len(samples), "runtime_floor_pct": 10,
        "memory_accounting": "file-cache", "footprint_cap_bytes": footprint_cap,
        "swap_growth_cap_bytes": swap_cap, "timeout_sec": timeout,
        "peak_group_footprint_bytes": max(peaks), "minimum_sampled_headroom_bytes": min(headrooms),
        "minimum_pressure_free_pct": min(pressures), "maximum_swap_growth_bytes": max(0, max(swaps) - swaps[0]),
        "physical_bytes": next(iter(physical_bytes)),
        "limitation": limits.get("limitation"),
        "file_cache_limitation": limits.get("file_cache_limitation")}


def analyze(report: object, digest: str | None = None) -> dict[str, Any]:
    data = mapping(report, "report must be an object")
    need(data.get("schema") == REPORT, "wrong report schema")
    comparison = mapping(data.get("comparison"), "missing comparison")
    need(type(comparison.get("atol")) in (int, float) and comparison["atol"] == 2e-5 and
         type(comparison.get("rtol")) in (int, float) and comparison["rtol"] == 2e-5 and
         comparison.get("capture_order") == list(CAPTURES), "comparison settings are not exact")
    flags = mapping(data.get("validation"), "missing validation flags")
    need(flags == {"all_numerical_metrics_passed": True, "positive_nan_control_passed": True,
                   "one_terminal_wait_and_packed_readback_per_request": True,
                   "zero_named_encode_allocations_and_preterminal_readbacks": True,
                   "full_generation_executed": False, "production_promoted": False},
         "global validation flags are incomplete or failed")
    schedule = mapping(data.get("schedule"), "missing schedule")
    expected_counts = {"correctness_requests": 2, "warmup_requests": 4, "timed_requests": 96, "native_requests": 102}
    for key, value in expected_counts.items():
        need(uint(schedule.get(key), f"invalid schedule {key}") == value, f"schedule {key} must equal {value}")
    requests = sequence(schedule.get("requests"), "missing requests")
    need(len(requests) == 102, "expected exactly 102 requests")
    checked = []
    profiled = []
    all_gc = Counter()
    thermal = Counter()
    last_point = None
    root_counts = set()
    for i, raw in enumerate(requests):
        req = mapping(raw, f"malformed request {i}")
        if i < 2:
            kind, enabled = "correctness", bool(i)
        elif i < 6:
            kind, enabled = "warmup", (False, True, True, False)[i - 2]
        else:
            kind, enabled = "timed", TIMED_PATTERN[(i - 6) % 8]
        row, parsed = validate_request(req, i, kind, enabled)
        root_counts.add(row["root_count"])
        for p in row["raw_points"]:
            if last_point is not None:
                need(all(p[k] >= last_point[k] for k in POINT_MONOTONIC),
                     f"request {i}: process observer counter moved backwards")
            last_point = p
        for key, value in row["gc"].items():
            all_gc[key] += value
        for p in row["host_points"]:
            thermal[p["thermal_state"]] += 1
        checked.append(row)
        if parsed is not None:
            profiled.append((row, parsed))
    need(len(root_counts) == 1 and len(profiled) == 51, "request root topology or profile-on count changed")
    root_count = next(iter(root_counts))
    need(all(len(p["intervals"]) == root_count for _, p in profiled), "profile/nonprofile root count mismatch")
    loop = mapping(data.get("observed_loop"), "missing observer envelope")
    exact_keys(loop, GC_FIELDS, "observed_loop")
    for k in GC_FIELDS:
        if k in GC_INTS:
            uint(loop[k], f"invalid observed_loop.{k}")
        else:
            finite(loop[k], f"invalid observed_loop.{k}")
    setup = finite(data.get("setup_seconds"), "invalid setup_seconds")

    timed = checked[6:]
    metric_functions = [
        ("body_encode_seconds", lambda r: r["body_encode_seconds"]),
        ("comparator_encode_seconds", lambda r: r["comparator_encode_seconds"]),
        ("encode_host_wall_seconds", lambda r: r["phases"]["encode"]["wall_seconds"]),
        ("encode_host_cpu_seconds", lambda r: r["phases"]["encode"]["thread_cpu_seconds"]),
        ("terminal_wait_wall_seconds", lambda r: r["phases"]["terminal_wait"]["wall_seconds"]),
        ("terminal_wait_cpu_seconds", lambda r: r["phases"]["terminal_wait"]["thread_cpu_seconds"]),
        ("gpu_root_duration_sum_seconds", lambda r: r["gpu_elapsed_seconds"]),
        ("total_host_wall_seconds", lambda r: r["phases"]["total"]["wall_seconds"]),
        ("total_host_cpu_seconds", lambda r: r["phases"]["total"]["thread_cpu_seconds"]),
    ]
    arms = {}
    for enabled, arm in ((False, "profile_off"), (True, "profile_on")):
        rows = [r for r in timed if r["profile_enabled"] is enabled]
        arms[arm] = {name: distribution([float(fn(r)) for r in rows]) for name, fn in metric_functions}
        arms[arm]["host_phase_wall_and_cpu"] = {phase: {
            "wall_seconds": distribution([r["phases"][phase]["wall_seconds"] for r in rows]),
            "thread_cpu_seconds": distribution([r["phases"][phase]["thread_cpu_seconds"] for r in rows])}
            for phase in HOST_PHASES}
    return {"schema": SUMMARY, "report_schema": REPORT, "report_sha256": digest,
        "request_counts": {"correctness": 2, "warmup": 4, "timed": 96, "total": 102,
                           "profiled": len(profiled), "root_count_per_request": root_count},
        "comparison": {"atol": 2e-5, "rtol": 2e-5, "capture_order": list(CAPTURES),
                       "numerical_metrics_validated_on_every_request": True, "different_words_preserved": True},
        "different_words_by_request": [{"request_index": r["index"], "kind": r["kind"],
            "profile_enabled": r["profile_enabled"],
            "slots": [m["different_words"] for m in r["numerical_metrics"]]} for r in checked],
        "fixed_boundary": {"terminal_waits_per_request": 1, "readbacks_after_wait_per_request": 1,
                           "readbacks_before_wait_per_request": 0, "named_buffer_allocations_per_request": 0,
                           "comparator_readback_bytes": 464},
        "setup_seconds": setup, "arms": arms,
        "paired_adjacent_on_off": paired_analysis(timed, metric_functions),
        "chronological_drift": drift_analysis(timed, metric_functions),
        "profiled_requests": profile_aggregates(profiled),
        "observer": {"global_observed_loop": loop, "sum_of_request_deltas": dict(all_gc),
            "thermal_state_samples": {str(k): v for k, v in sorted(thermal.items())},
            "thermal_interpretation": "NSProcessInfo advisory state only; not GPU frequency or causal thermal evidence.",
            "request_windows_note": "Request deltas are windows within the loop envelope; the global envelope also includes gaps and should not be inferred by adding overlapping or non-partitioning windows."},
        "interpretation_limits": ["gpu_elapsed_seconds is a root-duration sum. Interval union measures time covered by any root; envelope spans the first root start through the last root end and includes gaps. None is pure kernel-active time.",
            "Per-phase root candidate sums, root-duration sums, and graph/commit inclusive API totals are not additive GPU work.",
            "Continuation backlog overlap excludes the root being submitted; completion observations are not exact GPU completion instants.",
            "The probe covers two standalone decoder blocks, not full-generation or production behavior."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--guard", type=Path, help="optional guarded-run JSONL log to validate and summarize")
    parser.add_argument("--output", type=Path, help="fresh summary path; existing files are never replaced")
    args = parser.parse_args(argv)
    try:
        raw = args.report.read_bytes()
        summary = analyze(json.loads(raw), hashlib.sha256(raw).hexdigest())
        if args.guard:
            guard_raw = args.guard.read_bytes()
            guard_events = [json.loads(line) for line in guard_raw.splitlines() if line.strip()]
            summary["guard"] = validate_guard(guard_events, hashlib.sha256(guard_raw).hexdigest())
        encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(encoded)
        else:
            sys.stdout.write(encoded)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"pixelumm drift analysis rejected report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
