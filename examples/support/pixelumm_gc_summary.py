#!/usr/bin/env python3
"""Validate and summarize a PixelUMM GC probe report without running it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "cogni_ml.pixelumm_gc_probe/v1"
SUMMARY_SCHEMA = "cogni_ml.pixelumm_gc_summary/v1"
ATOL = 2e-5
RTOL = 2e-5
CAPTURE_ORDER = ("post_attn_norm", "gate_proj", "up_proj", "product", "down_proj", "output")
POLICIES = ("normal", "scoped_disabled")
TIMED_POLICY_PATTERN = ("normal", "scoped_disabled", "scoped_disabled", "normal",
                        "scoped_disabled", "normal", "normal", "scoped_disabled")
TIMED_TRIALS = 64
CORRECTNESS_TRIALS = 2
WARMUP_TRIALS = 4
TRIAL_COUNT = CORRECTNESS_TRIALS + WARMUP_TRIALS + TIMED_TRIALS
PHASES = (
    "policy_enter", "admission_poison", "encode_body_and_comparators",
    "terminal_wait", "policy_exit", "metrics_release", "critical", "total",
)
DELTA_FIELDS = (
    "seconds", "allocated_bytes", "collections", "gc_no_delta", "collection_seconds",
    "rendezvous_pauses", "rendezvous_pause_seconds", "stopped_intervals", "stopped_seconds",
)
INTEGER_DELTA_FIELDS = {
    "allocated_bytes", "collections", "gc_no_delta", "rendezvous_pauses", "stopped_intervals",
}
POINT_FIELDS = (
    "time_ns", "allocated_bytes", "gc_no", "heap_bytes", "free_bytes", "bytes_since_gc",
    "bytes_before_gc", "collection_count", "collection_total_ns", "collection_max_ns",
    "rendezvous_pause_count", "rendezvous_pause_total_ns", "rendezvous_pause_max_ns",
    "stopped_interval_count", "stopped_interval_total_ns", "stopped_interval_max_ns",
    "unmatched_event_count",
)
MONOTONIC_POINT_FIELDS = (
    "time_ns", "allocated_bytes", "gc_no", "collection_count", "collection_total_ns",
    "collection_max_ns", "rendezvous_pause_count", "rendezvous_pause_total_ns",
    "rendezvous_pause_max_ns", "stopped_interval_count", "stopped_interval_total_ns",
    "stopped_interval_max_ns",
)
METRIC_FIELDS = ("failures", "different_words", "nonfinite", "elements")
METRIC_SLOTS = 29
METRIC_BYTES = METRIC_SLOTS * 4 * 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def object_value(value: object, message: str) -> dict[str, Any]:
    require(isinstance(value, dict), message)
    return value


def integer(value: object, message: str, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, message)
    return int(value)


def finite_number(value: object, message: str, minimum: float = 0.0) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), message)
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(message) from exc
    require(math.isfinite(result) and result >= minimum, message)
    return result


def _point(value: object, message: str) -> dict[str, int]:
    raw = object_value(value, message)
    require(set(raw) == set(POINT_FIELDS), f"{message}: incomplete or unexpected point fields")
    point = {name: integer(raw[name], f"{message}: invalid {name}") for name in POINT_FIELDS}
    require(point["unmatched_event_count"] == 0, f"{message}: observer has unmatched events")
    return point


def _check_point_order(previous: dict[str, int] | None, current: dict[str, int], message: str) -> None:
    if previous is None:
        return
    for name in MONOTONIC_POINT_FIELDS:
        require(current[name] >= previous[name], f"{message}: {name} moved backwards")


def _point_delta(first: dict[str, int], last: dict[str, int]) -> dict[str, int | float]:
    require(last["time_ns"] >= first["time_ns"], "point time moved backwards")
    return {
        "seconds": (last["time_ns"] - first["time_ns"]) / 1e9,
        "allocated_bytes": last["allocated_bytes"] - first["allocated_bytes"],
        "collections": last["collection_count"] - first["collection_count"],
        "gc_no_delta": last["gc_no"] - first["gc_no"],
        "collection_seconds": (last["collection_total_ns"] - first["collection_total_ns"]) / 1e9,
        "rendezvous_pauses": last["rendezvous_pause_count"] - first["rendezvous_pause_count"],
        "rendezvous_pause_seconds":
            (last["rendezvous_pause_total_ns"] - first["rendezvous_pause_total_ns"]) / 1e9,
        "stopped_intervals": last["stopped_interval_count"] - first["stopped_interval_count"],
        "stopped_seconds":
            (last["stopped_interval_total_ns"] - first["stopped_interval_total_ns"]) / 1e9,
    }


def _delta_values(value: object, message: str) -> dict[str, int | float]:
    raw = object_value(value, message)
    require(set(raw) == set(DELTA_FIELDS), f"{message}: incomplete or unexpected delta fields")
    actual: dict[str, int | float] = {}
    for name in DELTA_FIELDS:
        if name in INTEGER_DELTA_FIELDS:
            actual[name] = integer(raw[name], f"{message}: invalid {name}")
        else:
            actual[name] = finite_number(raw[name], f"{message}: invalid {name}")
    return actual


def _check_delta(value: object, expected: dict[str, int | float], message: str) -> dict[str, int | float]:
    actual = _delta_values(value, message)
    for name in DELTA_FIELDS:
        if name in INTEGER_DELTA_FIELDS:
            require(actual[name] == expected[name], f"{message}: {name} disagrees with raw points")
        else:
            require(math.isclose(float(actual[name]), float(expected[name]), rel_tol=1e-9, abs_tol=1e-12),
                    f"{message}: {name} disagrees with raw points")
    return actual


def _require_at_least(actual: int | float, lower_bound: int | float, message: str) -> None:
    if isinstance(actual, int) and isinstance(lower_bound, int):
        require(actual >= lower_bound, message)
    else:
        require(float(actual) >= float(lower_bound) or
                math.isclose(float(actual), float(lower_bound), rel_tol=1e-9, abs_tol=1e-12),
                message)


def _metric_array(value: object, trial_index: int) -> list[dict[str, Any]]:
    require(isinstance(value, list) and len(value) == METRIC_SLOTS,
            f"correctness trial {trial_index} must contain exactly 29 capture metrics")
    metrics = []
    for slot, raw_metric in enumerate(value):
        metric = object_value(raw_metric, f"correctness trial {trial_index} metric {slot} is malformed")
        require(set(metric) == set(METRIC_FIELDS),
                f"correctness trial {trial_index} metric {slot} has incomplete fields")
        parsed = {name: integer(metric[name], f"invalid metric {name} at slot {slot}")
                  for name in METRIC_FIELDS}
        require(parsed["failures"] <= parsed["elements"] and
                parsed["different_words"] <= parsed["elements"] and
                parsed["nonfinite"] <= parsed["elements"],
                f"capture metric counters exceed elements at slot {slot}")
        if slot < 28:
            require(parsed["failures"] == 0 and parsed["nonfinite"] == 0,
                    f"correctness trial {trial_index} numerical metric {slot} failed")
            if slot < 4:
                require(parsed == {"failures": 0, "different_words": 0,
                                   "nonfinite": 0, "elements": 0},
                        f"correctness trial {trial_index} unused identity slot {slot} is nonzero")
            else:
                require(parsed["elements"] > 0,
                        f"correctness trial {trial_index} capture metric {slot} has no elements")
        metrics.append(parsed)

    control = metrics[28]
    if trial_index == 0:
        require(control["failures"] == 1 and control["nonfinite"] == 1 and
                control["different_words"] == 0 and control["elements"] > 0,
                "correctness trial 0 did not record the exact NaN positive control")
    else:
        require(control == {"failures": 0, "different_words": 0,
                           "nonfinite": 0, "elements": 0},
                "correctness trial 1 positive-control slot is not clear")
    return metrics


def _metric_records(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for slot, metric in enumerate(metrics):
        record: dict[str, Any] = {"slot": slot, **metric}
        if slot < 4:
            record.update({"role": "unused_stage_identity", "plan_slot": slot})
        elif slot < 28:
            offset = slot - 4
            record.update({"role": "capture", "plan_slot": offset // len(CAPTURE_ORDER),
                           "stage": CAPTURE_ORDER[offset % len(CAPTURE_ORDER)]})
        else:
            record["role"] = "nan_positive_control"
        records.append(record)
    return records


def _request(value: object, *, expected_index: int, expected_policy: str,
             expected_kind: str, label: str, point_sink: list[dict[str, int]]) -> dict[str, Any]:
    request = object_value(value, f"missing {label} request")
    require(integer(request.get("index"), f"invalid {label} request index") == expected_index,
            f"{label} request index changed")
    require(request.get("policy") == expected_policy, f"{label} request policy changed")
    require(request.get("kind") == expected_kind, f"{label} request kind changed")

    invariants = {
        "named_buffer_allocations": 0,
        "terminal_wait_count": 1,
        "preterminal_readbacks": 0,
        "comparator_readbacks_after_wait": 1,
        "comparator_readback_bytes": METRIC_BYTES,
    }
    for field, expected in invariants.items():
        require(integer(request.get(field), f"invalid {label} {field}") == expected,
                f"{label} violates {field} == {expected}")
    for field in ("body_encode_seconds", "comparator_encode_seconds", "gpu_elapsed_seconds"):
        finite_number(request.get(field), f"invalid {label} {field}")

    raw_points = request.get("raw_points")
    require(isinstance(raw_points, list) and len(raw_points) == 7,
            f"{label} request must contain seven raw points")
    points = []
    for point_index, raw_point in enumerate(raw_points):
        point = _point(raw_point, f"{label} point {point_index}")
        _check_point_order(point_sink[-1] if point_sink else None, point, f"{label} point {point_index}")
        point_sink.append(point)
        points.append(point)

    phase_map = {
        "policy_enter": (0, 1), "admission_poison": (1, 2),
        "encode_body_and_comparators": (2, 3), "terminal_wait": (3, 4),
        "policy_exit": (4, 5), "metrics_release": (5, 6),
        "critical": (1, 4), "total": (0, 6),
    }
    phases = object_value(request.get("phases"), f"missing {label} phases")
    require(set(phases) == set(PHASES), f"{label} has incomplete or unexpected phases")
    parsed_phases = {}
    for name, (start, end) in phase_map.items():
        parsed_phases[name] = _check_delta(
            phases[name], _point_delta(points[start], points[end]), f"{label} phase {name}")
    if expected_policy == "scoped_disabled":
        require(parsed_phases["critical"]["collections"] == 0 and
                parsed_phases["critical"]["gc_no_delta"] == 0,
                f"{label} collected while scoped GC was disabled")

    return {"request": request, "points": points, "phases": parsed_phases}


def _sum_phase(left: dict[str, int | float], right: dict[str, int | float]) -> dict[str, int | float]:
    return {field: left[field] + right[field] for field in DELTA_FIELDS}


def _validated(report: object) -> dict[str, Any]:
    root = object_value(report, "report root must be an object")
    require(root.get("schema") == REPORT_SCHEMA, "unsupported PixelUMM GC report schema")

    comparison = object_value(root.get("comparison"), "missing comparison metadata")
    for field, expected in (("atol", ATOL), ("rtol", RTOL)):
        actual = finite_number(comparison.get(field), f"invalid comparison {field}")
        require(actual == expected, f"comparison {field} must be exactly {expected}")
    require(comparison.get("capture_order") == list(CAPTURE_ORDER), "capture order changed")

    observation = object_value(root.get("observation"), "missing observation metadata")
    require(observation.get("normal_tail_after_every_primary") is True,
            "report does not guarantee a normal-GC tail after every primary")
    require(observation.get("forced_collections_or_artificial_pressure_in_gpu_run") is False,
            "report indicates forced collection or artificial GC pressure")

    validation = object_value(root.get("validation"), "missing probe validation")
    expected_validation = {
        "all_numerical_metrics_passed": True,
        "positive_nan_control_passed": True,
        "one_terminal_wait_and_packed_readback_per_request": True,
        "zero_named_encode_allocations_and_preterminal_readbacks": True,
        "gc_enabled_after_every_scope": True,
        "observer_events_paired": True,
        "full_generation_executed": False,
        "production_promoted": False,
    }
    for name, expected in expected_validation.items():
        require(validation.get(name) is expected, f"probe validation {name} is not {expected}")

    schedule = object_value(root.get("schedule"), "missing trial schedule")
    require(integer(schedule.get("correctness_trials"), "invalid correctness trial count") == CORRECTNESS_TRIALS,
            "correctness trial count changed")
    require(integer(schedule.get("warmup_trials"), "invalid warmup trial count") == WARMUP_TRIALS,
            "warmup trial count changed")
    require(integer(schedule.get("timed_trials"), "invalid timed trial count") == TIMED_TRIALS,
            "timed trial count changed")
    require(integer(schedule.get("native_requests"), "invalid native request count") == TRIAL_COUNT * 2,
            "native request count does not match the fixed schedule")
    require(schedule.get("timed_order") ==
            "ABBA/BAAB repeated eight times; A=normal B=scoped_disabled",
            "timed ABBA/BAAB schedule description changed")
    trials = schedule.get("trials")
    require(isinstance(trials, list) and len(trials) == TRIAL_COUNT,
            "incomplete fixed GC trial schedule")

    point_sink: list[dict[str, int]] = []
    initial = _point(root.get("initial_point"), "invalid initial observer point")
    point_sink.append(initial)
    parsed_trials = []
    correctness_metrics = []
    for index, raw_trial in enumerate(trials):
        trial = object_value(raw_trial, f"invalid trial {index}")
        require(integer(trial.get("index"), f"invalid trial {index} index") == index,
                "trial indices are not consecutive")
        if index < CORRECTNESS_TRIALS:
            expected_kind = "correctness"
            expected_policy = POLICIES[index]
        elif index < CORRECTNESS_TRIALS + WARMUP_TRIALS:
            expected_kind = "warmup"
            expected_policy = ("normal", "scoped_disabled", "scoped_disabled", "normal")[index - CORRECTNESS_TRIALS]
        else:
            expected_kind = "timed"
            expected_policy = TIMED_POLICY_PATTERN[(index - CORRECTNESS_TRIALS - WARMUP_TRIALS) % 8]
        require(trial.get("kind") == expected_kind and trial.get("policy") == expected_policy,
                f"trial {index} changed the fixed kind or policy schedule")

        primary = _request(trial.get("primary"), expected_index=index * 2,
                           expected_policy=expected_policy, expected_kind=expected_kind,
                           label=f"trial {index} primary", point_sink=point_sink)
        tail = _request(trial.get("tail"), expected_index=index * 2 + 1,
                        expected_policy="normal", expected_kind="tail",
                        label=f"trial {index} normal tail", point_sink=point_sink)
        inclusive = _delta_values(trial.get("inclusive"), f"trial {index} inclusive window")

        if expected_kind == "correctness":
            request = primary["request"]
            metrics = _metric_array(request.get("capture_metrics"), index)
            correctness_metrics.append({"trial_index": index, "request_index": index * 2,
                                        "metrics": _metric_records(metrics)})
        else:
            require(primary["request"].get("capture_metrics") is None,
                    f"non-correctness trial {index} unexpectedly contains capture metrics")

        primary_total = primary["phases"]["total"]
        tail_total = tail["phases"]["total"]
        for field in DELTA_FIELDS:
            _require_at_least(inclusive[field], primary_total[field] + tail_total[field],
                              f"trial {index} inclusive {field} is smaller than disjoint primary and tail totals")
        phase_sums = {
            name: _sum_phase(primary["phases"][name], tail["phases"][name])
            for name in PHASES
        }
        primary_timings = {
            "body_encode_seconds": primary["request"]["body_encode_seconds"],
            "comparator_encode_seconds": primary["request"]["comparator_encode_seconds"],
            "gpu_chain_span_seconds": primary["request"]["gpu_elapsed_seconds"],
        }
        inclusive_timings = {
            "body_encode_seconds": primary_timings["body_encode_seconds"] +
                                   tail["request"]["body_encode_seconds"],
            "comparator_encode_seconds": primary_timings["comparator_encode_seconds"] +
                                         tail["request"]["comparator_encode_seconds"],
            "gpu_chain_span_seconds": primary_timings["gpu_chain_span_seconds"] +
                                      tail["request"]["gpu_elapsed_seconds"],
        }
        parsed_trials.append({
            "index": index, "kind": expected_kind, "policy": expected_policy,
            "primary": {
                "wall_seconds": primary_total["seconds"],
                "host_allocated_bytes": primary_total["allocated_bytes"],
                "events": primary_total,
                "phases": primary["phases"],
                "component_timings": primary_timings,
            },
            "inclusive": {
                "wall_seconds": inclusive["seconds"],
                "host_allocated_bytes": inclusive["allocated_bytes"],
                "events": inclusive,
                "phases": phase_sums,
                "component_timings": inclusive_timings,
            },
        })

    final = _point(root.get("final_point"), "invalid final observer point")
    _check_point_order(point_sink[-1], final, "final observer point")
    point_sink.append(final)
    loop = _check_delta(root.get("observed_loop"), _point_delta(initial, final), "all-loop observer window")
    for field in DELTA_FIELDS:
        inclusive_sum = sum((item["inclusive"]["events"][field] for item in parsed_trials), 0)
        _require_at_least(loop[field], inclusive_sum,
                          f"all-loop {field} is smaller than the sum of trial inclusive windows")

    require(len(correctness_metrics) == CORRECTNESS_TRIALS,
            "incomplete correctness capture metrics")
    timed = parsed_trials[CORRECTNESS_TRIALS + WARMUP_TRIALS:]
    require(len(timed) == TIMED_TRIALS and
            sum(item["policy"] == "normal" for item in timed) == TIMED_TRIALS // 2 and
            sum(item["policy"] == "scoped_disabled" for item in timed) == TIMED_TRIALS // 2,
            "timed arm sample count is not balanced")
    return {"trials": parsed_trials, "timed": timed, "correctness_metrics": correctness_metrics,
            "initial_point": initial, "final_point": final, "all_loop": loop}


def _stats(values: list[int | float]) -> dict[str, int | float]:
    require(bool(values), "cannot summarize an empty sample")
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        total: int | float = sum(int(value) for value in values)
    else:
        total = math.fsum(float(value) for value in values)
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "sum": total}


def _event_summary(values: list[dict[str, int | float]]) -> dict[str, Any]:
    return {
        "boehm_collection_count": _stats([item["collections"] for item in values]),
        "gc_no_delta": _stats([item["gc_no_delta"] for item in values]),
        "collection_envelope_seconds": _stats([item["collection_seconds"] for item in values]),
        "rendezvous_pause_count": _stats([item["rendezvous_pauses"] for item in values]),
        "rendezvous_pause_seconds": _stats([item["rendezvous_pause_seconds"] for item in values]),
        "fully_stopped_interval_count": _stats([item["stopped_intervals"] for item in values]),
        "fully_stopped_interval_seconds": _stats([item["stopped_seconds"] for item in values]),
    }


def _measurement_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    events = [sample["events"] for sample in samples]
    phase_summary = {}
    for phase in PHASES:
        phase_values = [sample["phases"][phase] for sample in samples]
        phase_summary[phase] = {
            "wall_seconds": _stats([value["seconds"] for value in phase_values]),
            "host_allocated_bytes": _stats([value["allocated_bytes"] for value in phase_values]),
            "gc_observations": _event_summary(phase_values),
        }
    collection_count = sum(int(value["collections"]) for value in events)
    return {
        "samples": len(samples),
        "wall_seconds": _stats([sample["wall_seconds"] for sample in samples]),
        "host_allocated_bytes": _stats([sample["host_allocated_bytes"] for sample in samples]),
        "component_timings": {
            name: _stats([sample["component_timings"][name] for sample in samples])
            for name in ("body_encode_seconds", "comparator_encode_seconds", "gpu_chain_span_seconds")
        },
        "component_timing_note": (
            "inclusive values sum primary and normal-tail requests; gpu_chain_span_seconds is the full "
            "GPU command-chain span, not pure kernel active time"
        ),
        "gc_observations": _event_summary(events),
        "pause_measurement": {
            "collection_count": collection_count,
            "diagnostic": collection_count > 0,
            "status": "collections_observed" if collection_count else "non_diagnostic_zero_collections",
        },
        "phase_timings": phase_summary,
        "phase_note": "critical and total are rolled-up windows; phase values overlap and must not be added",
    }


def _paired_wall(timed: list[dict[str, Any]], measurement: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(measurement in ("primary", "inclusive"), "unknown paired wall measurement")
    pairs = []
    for pair_index in range(TIMED_TRIALS // 2):
        left, right = timed[2 * pair_index:2 * pair_index + 2]
        require(left["policy"] != right["policy"], "consecutive timed pair does not contain both arms")
        by_policy = {left["policy"]: left, right["policy"]: right}
        normal = by_policy["normal"][measurement]["wall_seconds"]
        disabled = by_policy["scoped_disabled"][measurement]["wall_seconds"]
        require(normal > 0, "normal comparison wall time must be positive for paired percentages")
        delta = (normal - disabled) * 100.0 / normal
        order = "AB" if left["policy"] == "normal" else "BA"
        pairs.append({"pair_index": pair_index, "trial_indices": [left["index"], right["index"]],
                      "order": order, "normal_wall_seconds": normal,
                      "scoped_disabled_wall_seconds": disabled,
                      "normal_minus_scoped_disabled_percent": delta})
    deltas = [pair["normal_minus_scoped_disabled_percent"] for pair in pairs]
    wins = sum(value > 0 for value in deltas)
    ties = sum(value == 0 for value in deltas)
    summary = {
        "pairs": len(pairs),
        "measurement": "primary-only" if measurement == "primary" else "primary plus its normal-GC tail",
        "percent_formula": "(normal wall - scoped_disabled wall) / normal wall * 100; positive favors scoped_disabled",
        "median_normal_minus_scoped_disabled_percent": statistics.median(deltas),
        "range_percent": {"min": min(deltas), "max": max(deltas)},
        "scoped_disabled_wins": wins,
        "ties": ties,
        "normal_wins": len(deltas) - wins - ties,
    }
    return summary, pairs


def _drift(timed: list[dict[str, Any]], pair_sets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    halves = {"early": timed[:TIMED_TRIALS // 2], "late": timed[TIMED_TRIALS // 2:]}
    result: dict[str, Any] = {
        "descriptive_only": True,
        "method": "first and last 32 timed trials, each with 16 samples per arm",
        "arms": {},
        "paired_wall_delta_percent": {},
        "component_timing_drift": {"arms": {}, "descriptive_only": True},
    }
    for policy, name in (("normal", "normal"), ("scoped_disabled", "scoped_disabled")):
        result["arms"][name] = {}
        result["component_timing_drift"]["arms"][name] = {}
        for metric, label in (("primary", "primary_only"),
                              ("inclusive", "inclusive_with_normal_tail")):
            early = [row[metric]["wall_seconds"] for row in halves["early"] if row["policy"] == policy]
            late = [row[metric]["wall_seconds"] for row in halves["late"] if row["policy"] == policy]
            require(len(early) == len(late) == 16, "early/late drift split is not arm-balanced")
            early_median = statistics.median(early)
            late_median = statistics.median(late)
            require(early_median > 0, "early process-drift wall time must be positive")
            result["arms"][name][label] = {
                "early_wall_seconds": _stats(early),
                "late_wall_seconds": _stats(late),
                "late_minus_early_percent": (late_median - early_median) * 100.0 / early_median,
            }
            result["component_timing_drift"]["arms"][name][label] = {}
            for component in ("body_encode_seconds", "comparator_encode_seconds", "gpu_chain_span_seconds"):
                early_values = [row[metric]["component_timings"][component]
                                for row in halves["early"] if row["policy"] == policy]
                late_values = [row[metric]["component_timings"][component]
                               for row in halves["late"] if row["policy"] == policy]
                early_median = statistics.median(early_values)
                late_median = statistics.median(late_values)
                drift = ((late_median - early_median) * 100.0 / early_median
                         if early_median > 0 else None)
                result["component_timing_drift"]["arms"][name][label][component] = {
                    "early_median_seconds": early_median,
                    "late_median_seconds": late_median,
                    "late_minus_early_percent": drift,
                    "status": "computed" if drift is not None else "unavailable_early_median_zero",
                }
    for measurement, pairs in pair_sets.items():
        result["paired_wall_delta_percent"][measurement] = {}
        for label, pair_slice in (("early", pairs[:16]), ("late", pairs[16:])):
            result["paired_wall_delta_percent"][measurement][label] = _stats(
                [pair["normal_minus_scoped_disabled_percent"] for pair in pair_slice])
    return result


def summarize(report: object, report_sha256: str | None = None) -> dict[str, Any]:
    """Validate a complete v1 report and return descriptive CPU-only summaries."""
    checked = _validated(report)
    timed = checked["timed"]
    by_policy = {
        policy: [row for row in timed if row["policy"] == policy]
        for policy in POLICIES
    }
    require(all(len(rows) == TIMED_TRIALS // 2 for rows in by_policy.values()),
            "timed arm sample count is incomplete")

    arms = {}
    for policy in POLICIES:
        primary = [row["primary"] for row in by_policy[policy]]
        inclusive = [row["inclusive"] for row in by_policy[policy]]
        arms[policy] = {
            "timed_primary": _measurement_summary(primary),
            "timed_inclusive_with_normal_tail": _measurement_summary(inclusive),
        }
    primary_pairs, primary_pair_rows = _paired_wall(timed, "primary")
    inclusive_pairs, inclusive_pair_rows = _paired_wall(timed, "inclusive")
    pair_sets = {"primary_only": primary_pair_rows,
                 "inclusive_with_normal_tail": inclusive_pair_rows}
    output: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "report_schema": REPORT_SCHEMA,
        "report_sha256": report_sha256,
        "scope": "CPU-only analysis of a completed bounded GC probe report; no probe execution",
        "fixed_schedule": {
            "timed_trials": TIMED_TRIALS,
            "samples_per_arm": TIMED_TRIALS // 2,
            "order": "ABBA/BAAB repeated eight times; A=normal B=scoped_disabled",
            "paired_consecutive_trials": 32,
        },
        "arms": arms,
        "paired_wall": {
            "primary_only": primary_pairs,
            "inclusive_with_normal_tail": inclusive_pairs,
        },
        "process_drift": _drift(timed, pair_sets),
        "correctness_capture_metrics": {
            "metrics_per_correctness_primary": METRIC_SLOTS,
            "stage_capture_slots": {"first": 4, "last": 27, "plan_slots": 4,
                                     "capture_order": list(CAPTURE_ORDER)},
            "positive_control_slot": 28,
            "correctness_primaries": checked["correctness_metrics"],
        },
        "observer": {
            "all_loop": {
                "wall_seconds": checked["all_loop"]["seconds"],
                "host_allocated_bytes": checked["all_loop"]["allocated_bytes"],
                "gc_observations": {
                    "boehm_collection_count": checked["all_loop"]["collections"],
                    "gc_no_delta": checked["all_loop"]["gc_no_delta"],
                    "collection_envelope_seconds": checked["all_loop"]["collection_seconds"],
                    "rendezvous_pause_count": checked["all_loop"]["rendezvous_pauses"],
                    "rendezvous_pause_seconds": checked["all_loop"]["rendezvous_pause_seconds"],
                    "fully_stopped_interval_count": checked["all_loop"]["stopped_intervals"],
                    "fully_stopped_interval_seconds": checked["all_loop"]["stopped_seconds"],
                },
            },
            "final_point_maxima_ns": {
                "collection_envelope": checked["final_point"]["collection_max_ns"],
                "rendezvous_pause": checked["final_point"]["rendezvous_pause_max_ns"],
                "fully_stopped_interval": checked["final_point"]["stopped_interval_max_ns"],
            },
            "maxima_note": "final Point maxima are cumulative observer maxima; do not subtract them for phase maxima",
            "overlap_note": "collection envelopes, rendezvous-inclusive pauses, and fully-stopped intervals overlap; do not sum their durations",
        },
        "pause_diagnostics": {
            "normal_primary": arms["normal"]["timed_primary"]["pause_measurement"],
            "normal_primary_with_normal_tail":
                arms["normal"]["timed_inclusive_with_normal_tail"]["pause_measurement"],
        },
        "interpretation": "timing and GC observations are descriptive; process drift is not a causal adjustment or qualification",
    }
    return output


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _write_fresh(path: Path, content: str, source: Path) -> None:
    try:
        if path.resolve(strict=False) == source.resolve(strict=True):
            raise ValueError("summary output must not overwrite the input report")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ValueError(f"summary output already exists: {path}") from exc
    except OSError as exc:
        raise ValueError(f"cannot create fresh summary output {path}: {exc}") from exc
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="completed pixelumm_gc_probe/v1 JSON report")
    parser.add_argument("--output", type=Path, help="write summary JSON to a fresh path")
    args = parser.parse_args(argv)
    try:
        raw = args.report.read_bytes()
        require(bool(raw), "input report is empty")
        report = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
        digest = hashlib.sha256(raw).hexdigest()
        summary = summarize(report, digest)
        rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            _write_fresh(args.output, rendered, args.report)
            sys.stdout.write(f"summary={args.output} report_sha256={digest}\n")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
