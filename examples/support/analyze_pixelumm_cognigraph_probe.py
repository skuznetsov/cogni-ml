#!/usr/bin/env python3
"""Validate and summarize a guarded 40-request PixelUMM CogniGraph probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any

REPORT_SCHEMA = "cogni_ml.pixelumm_cognigraph_probe/v1"
SUMMARY_SCHEMA = "cogni_ml.pixelumm_cognigraph_probe_summary/v1"
MIB = 1024 * 1024
FOOTPRINT_CAP = 16_384 * MIB
SWAP_GROWTH_CAP = 64 * MIB
EXPECTED_UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
EXPECTED_MODEL_SHA256 = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
EXPECTED_WEIGHT_CACHE_SHA256 = "74ee5e84f2b63af0189ffd7be0e4c758b8dfa9ff925e3c9cb389d207ce546dba"
EXPECTED_WEIGHT_CACHE_PAYLOAD_SHA256 = "152ba021b52d9546e4c8821b17ab11f67f46486d521b500de3766f14dd2217ca"
PAIR_PATTERN = [False, True, True, False, True, False, False, True]
METRIC_NAMES = ("full_request_wall_seconds", "body_encode_seconds",
                "host_thread_cpu_seconds", "gpu_root_duration_sum_seconds")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: object, label: str) -> dict[str, Any]:
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def uint(value: object, label: str, minimum: int = 0) -> int:
    need(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def ten_percent_floor(physical_bytes: int) -> int:
    return (physical_bytes + 9) // 10


def finite(value: object, label: str, *, positive: bool = False) -> float:
    need(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    number = float(value)
    need(not positive or number > 0.0, f"{label} must be positive")
    return number


def expected_requests() -> list[tuple[bool, str, bool, bool, bool]]:
    result: list[tuple[bool, str, bool, bool, bool]] = []
    for i, enabled in enumerate((False, True, False, True)):
        result.append((enabled, "correctness", i < 2, i == 0, False))
    for enabled in (False, True, True, False):
        result.append((enabled, "warmup", False, False, False))
    for index in range(32):
        result.append((PAIR_PATTERN[index % 8], "timed", False, False, True))
    return result


def validate_metrics(raw: object, index: int, capture: bool, positive_control: bool) -> None:
    metrics = raw
    need(isinstance(metrics, list) and len(metrics) == 29,
         f"request {index}: expected the 29-slot packed comparator result")
    rows = [mapping(value, f"request {index} metric {slot}") for slot, value in enumerate(metrics)]
    for slot, row in enumerate(rows):
        for field in ("failures", "different_words", "nonfinite", "elements"):
            uint(row.get(field), f"request {index} metric {slot}.{field}")
    # Slots 4..27 are four plans by six stages. Output is the final stage.
    capture_slots = [4 + plan * 6 + stage for plan in range(4) for stage in range(6)]
    output_slots = [4 + plan * 6 + 5 for plan in range(4)]
    compared = capture_slots if capture else output_slots
    for slot in compared:
        row = rows[slot]
        need(row["failures"] == 0 and row["nonfinite"] == 0,
             f"request {index}: numerical comparator failed at slot {slot}")
        need(row["elements"] > 0, f"request {index}: comparator slot {slot} was empty")
    for slot in set(capture_slots) - set(compared):
        need(rows[slot] == {"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 0},
             f"request {index}: an unrequested capture comparator ran at slot {slot}")
    need(rows[0:4] == [{"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 0}] * 4,
         f"request {index}: FFN staging metric slots must remain clear")
    control = rows[28]
    if positive_control:
        need(control["failures"] == 1 and control["different_words"] == 0 and
             control["nonfinite"] == 1 and control["elements"] > 0,
             "request 0: positive NaN control was not exact")
    else:
        need(control == {"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 0},
             f"request {index}: unexpected positive-control data")


def validate_report(value: object, digest: str | None = None) -> dict[str, Any]:
    data = mapping(value, "probe report")
    need(data.get("schema") == REPORT_SCHEMA, "wrong CogniGraph report schema")
    need(data.get("probe_completed") is True and data.get("production_promoted") is False,
         "report does not describe a completed diagnostic-only probe")
    probe_process_id = uint(data.get("probe_process_id"), "probe process id", minimum=1)
    need(data.get("full_generation_executed") is False,
         "PixelUMM full generation must remain false")
    comparison = mapping(data.get("numerical_comparison"), "missing numerical comparison")
    need(type(comparison.get("atol")) in (int, float) and comparison["atol"] == 2e-5 and
         type(comparison.get("rtol")) in (int, float) and comparison["rtol"] == 2e-5,
         "numerical tolerance changed from 2e-5")
    pins = mapping(data.get("input_pins"), "missing input/model pins")
    reference_digest = pins.get("reference_manifest_sha256")
    reference_payload_digest = pins.get("reference_payload_sha256")
    need(all(isinstance(value, str) and len(value) == 64 and
             all(char in "0123456789abcdef" for char in value)
             for value in (reference_digest, reference_payload_digest)) and
         pins.get("weight_cache_manifest_sha256") == EXPECTED_WEIGHT_CACHE_SHA256 and
         pins.get("weight_cache_payload_sha256") == EXPECTED_WEIGHT_CACHE_PAYLOAD_SHA256 and
         pins.get("upstream_model_sha256") == EXPECTED_MODEL_SHA256 and
         pins.get("upstream_revision") == EXPECTED_UPSTREAM_REVISION,
         "input/reference/cache/model pins differ from the reused PixelUMM fixture")
    need(data.get("request_count") == 40, "request count must be exactly 40")
    capture_contract = mapping(data.get("capture_contract"), "missing capture contract")
    need(capture_contract.get("request_capture_mode_changes_native_plan_capture_mask") is True and
         capture_contract.get("output_only_requests_retain_only_output_capture") is True and
         capture_contract.get("encoder_counts_are_prepared_plan_model_counters") is True,
         "capture or modeled encoder-count contract is not explicit")
    requests = data.get("requests")
    need(isinstance(requests, list) and len(requests) == 40, "report must contain exactly 40 requests")
    expected = expected_requests()
    prior_mode = False
    parsed: list[dict[str, Any]] = []
    for index, (raw, spec) in enumerate(zip(requests, expected)):
        row = mapping(raw, f"request {index}")
        enabled, kind, capture, positive, timed = spec
        need(row.get("index") == index and row.get("kind") == kind and
             row.get("native_islands_enabled") is enabled and
             row.get("capture_mode") is capture and
             row.get("positive_nan_control") is positive and
             row.get("timing_eligible") is timed,
             f"request {index}: schedule metadata differs from the fixed protocol")
        # The first request explicitly initializes the native selector even
        # though baseline is its default; later requests set it only on change.
        changed = index == 0 or enabled != prior_mode
        need(row.get("native_mode_toggled_before_request") is changed,
             f"request {index}: mode toggle was not recorded at its between-request boundary")
        prior_mode = enabled
        need(uint(row.get("terminal_wait_count"), f"request {index} wait count") == 1 and
             uint(row.get("named_buffer_allocations"), f"request {index} named allocations") == 0 and
             uint(row.get("preterminal_readbacks"), f"request {index} preterminal readbacks") == 0 and
             uint(row.get("comparator_readbacks_after_wait"), f"request {index} comparator readbacks") == 1,
             f"request {index}: request crossed the fixed wait/readback/allocation boundary")
        validate_metrics(row.get("numerical_metrics"), index, capture, positive)
        for metric in ("full_request_wall_seconds", "host_clock_wall_seconds",
                       "host_thread_cpu_seconds", "body_encode_seconds",
                       "comparator_encode_seconds", "terminal_wait_wall_seconds",
                       "native_gpu_elapsed_seconds", "gpu_root_duration_sum_seconds",
                       "gpu_root_union_seconds"):
            finite(row.get(metric), f"request {index}.{metric}", positive=True)
        finite(row.get("gpu_root_overlap_seconds"), f"request {index}.gpu_root_overlap_seconds")
        root_count = uint(row.get("root_count"), f"request {index} root count", minimum=1)
        need(root_count <= 128, f"request {index}: root count exceeds the fixed bounded chain")
        # The public MPS aggregate and recomputed interval sum must agree within
        # timestamp/summation roundoff; root sums may overlap in GPU time.
        gpu = float(row["native_gpu_elapsed_seconds"])
        root_sum = float(row["gpu_root_duration_sum_seconds"])
        need(abs(gpu - root_sum) <= max(1e-8, 1e-6 * max(gpu, root_sum)),
             f"request {index}: native GPU aggregate differs from root interval sum")
        need(float(row["gpu_root_union_seconds"]) <= root_sum + 1e-8 and
             float(row["gpu_root_overlap_seconds"]) >= 0.0,
             f"request {index}: GPU root interval accounting is inconsistent")
        parsed.append(row)

    stats_rows = data.get("native_island_stats")
    need(isinstance(stats_rows, list) and len(stats_rows) == 4,
         "native island stats must cover exactly four plans")
    stats = [mapping(row, f"native plan stats {i}") for i, row in enumerate(stats_rows)]
    keys = ("graphs", "ops", "waves", "barriers", "maxwidth", "native_dispatches_perblock",
            "baseline_encoders_perblock", "candidate_encoders_perblock_no_captures")
    for i, row in enumerate(stats):
        for key in keys:
            uint(row.get(key), f"native plan stats {i}.{key}")
    need(all(row == stats[0] for row in stats), "four identical plan stat records differ")
    need(stats[0]["graphs"] > 0 and stats[0]["ops"] > 0 and stats[0]["waves"] > 0 and
         stats[0]["maxwidth"] > 0 and stats[0]["native_dispatches_perblock"] > 0 and
         stats[0]["baseline_encoders_perblock"] > stats[0]["candidate_encoders_perblock_no_captures"] > 0,
         "native graph statistics or modeled encoder counts are invalid")
    need("model counters" in str(data.get("native_island_stats_interpretation", "")),
         "encoder counts must be labeled as model counters, not measured API calls")
    resources = mapping(data.get("named_resources"), "missing named-resource inventory")
    resource_bytes = mapping(resources.get("resource_bytes"), "missing named-resource byte inventory")
    input_bindings = resources.get("plan_input_binding_sha256")
    need(uint(resources.get("native_plan_count"), "native plan count") == 4 and
         resources.get("native_plan_handles_distinct") is True and
         resources.get("capture_slots_private") is True and
         resources.get("f32_stage_buffers_shared") is True and
         uint(resources.get("named_bytes"), "named byte total", minimum=1) ==
         sum(uint(value, f"named resource {name}") for name, value in resource_bytes.items()) and
         isinstance(input_bindings, list) and len(input_bindings) == 2 and
         all(isinstance(value, str) and len(value) == 64 and
             all(char in "0123456789abcdef" for char in value) for value in input_bindings) and
         len(set(input_bindings)) == 2,
         "named-resource inventory does not describe four live, distinct plans and shared buffers")
    need(data.get("summary") == {
        "correctness_gates_passed": True, "candidate_requests": 20, "baseline_requests": 20,
        "timed_candidate_requests": 16, "timed_baseline_requests": 16},
        "request summary counts differ from the balanced schedule")
    negative = mapping(data.get("native_api_negative_controls"), "missing native API negative controls")
    for field in ("unprepared_enable_status", "duplicate_prepare_status", "invalid_enabled_status",
                  "late_prepare_after_drained_request_status"):
        status = negative.get(field)
        need(type(status) is int and status != 0,
             f"native API negative control {field} unexpectedly succeeded")
    guard_decl = mapping(data.get("resource_guard"), "missing declared resource guard")
    need(guard_decl.get("max_process_footprint_mib") == 16_384 and
         guard_decl.get("minimum_free_memory_percent") == 10 and
         guard_decl.get("maximum_swap_growth_mib") == 64 and
         guard_decl.get("enforced_by_external_memory_guard") is True,
         "report does not declare the required 16 GiB / 10 percent / 64 MiB guard")

    timed = parsed[8:]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for offset in range(0, len(timed), 2):
        left, right = timed[offset:offset + 2]
        need(left["native_islands_enabled"] is not right["native_islands_enabled"],
             "each timed adjacent pair must contain one baseline and one candidate")
        pairs.append((left, right))
    paired: dict[str, dict[str, float]] = {}
    for metric in METRIC_NAMES:
        relatives: list[float] = []
        for left, right in pairs:
            baseline = left if left["native_islands_enabled"] is False else right
            candidate = right if baseline is left else left
            b = finite(baseline[metric], f"paired baseline {metric}", positive=True)
            c = finite(candidate[metric], f"paired candidate {metric}", positive=True)
            relatives.append((c / b) - 1.0)
        paired[metric] = {
            "pair_count": len(relatives),
            "candidate_over_baseline_mean_relative": statistics.fmean(relatives),
            "candidate_over_baseline_median_relative": statistics.median(relatives),
            "candidate_over_baseline_median_relative_percent": statistics.median(relatives) * 100.0,
        }
    root_totals = {}
    for enabled, label in ((False, "baseline"), (True, "candidate")):
        rows = [row for row in timed if row["native_islands_enabled"] is enabled]
        counts = [uint(row["root_count"], f"{label} root count", minimum=1) for row in rows]
        root_totals[label] = {"requests": len(rows), "total_roots": sum(counts),
                              "mean_roots_per_request": statistics.fmean(counts),
                              "min_roots_per_request": min(counts),
                              "max_roots_per_request": max(counts)}
    return {
        "schema": SUMMARY_SCHEMA,
        "report_sha256": digest,
        "probe_process_id": probe_process_id,
        "request_counts": {"correctness": 4, "warmup": 4, "timed": 32, "total": 40,
                           "timed_candidate": 16, "timed_baseline": 16},
        "comparison": {"atol": 2e-5, "rtol": 2e-5,
                       "all_per_request_gpu_comparators_passed": True,
                       "positive_nan_control_passed": True},
        "input_pins": pins,
        "timed_pairing": "16 adjacent AB/BA pairs in four repeated ABBA+BAAB cycles",
        "paired_relative_metrics": paired,
        "timed_root_counts": root_totals,
        "native_island_stats": stats[0],
        "encoder_count_interpretation": "prepared-plan model counters; no API profiling was enabled",
        "gpu_interval_interpretation": "root duration sums can overlap and are not exclusive GPU busy time",
        "setup_seconds": finite(data.get("setup_seconds"), "setup_seconds"),
        "report_digest": digest,
    }


def validate_guard(events: object, expected_process_id: int,
                   digest: str | None = None) -> dict[str, Any]:
    need(isinstance(events, list) and len(events) >= 5, "guard log must contain limits, start, samples and finish")
    rows = [mapping(row, f"guard event {i}") for i, row in enumerate(events)]
    need(rows[0].get("event") == "limits" and rows[-1].get("event") == "finished",
         "guard log must begin with limits and end with finished")
    limits = rows[0]
    need(limits.get("memory_accounting") == "file-cache" and
         limits.get("headroom_formula") == "printed free + file-backed (includes speculative)" and
         limits.get("runtime_floor_pct_exact") == "10" and
         finite(limits.get("runtime_floor_pct"), "guard runtime floor") == 10.0 and
         uint(limits.get("max_footprint_bytes"), "guard footprint cap") == FOOTPRINT_CAP and
         uint(limits.get("max_swap_growth_bytes"), "guard swap cap") == SWAP_GROWTH_CAP and
         uint(limits.get("timeout_sec"), "guard timeout", minimum=1) > 0 and
         limits.get("startup_rule") ==
         "memory_pressure -Q >= 10% and selected headroom >= 10% physical + footprint budget",
         "guard limits differ from 16 GiB, 10 percent and 64 MiB")
    samples = [row for row in rows if row.get("event") == "sample"]
    starts = [row for row in rows if row.get("event") == "started"]
    finishes = [row for row in rows if row.get("event") == "finished"]
    need(len(samples) >= 3 and len(starts) == 1 and len(finishes) == 1 and
         all(row.get("event") in {"limits", "sample", "started", "finished"} for row in rows) and
         rows[1].get("event") == "sample" and rows[2].get("event") == "started" and
         rows[-2].get("event") == "sample",
         "guard lifecycle has missing startup/runtime samples, duplicate process events, or a failure event")
    pid = uint(starts[0].get("pid"), "guard process id", minimum=1)
    need(pid == uint(expected_process_id, "report probe process id", minimum=1) and
         uint(starts[0].get("pgid"), "guard process group", minimum=1) == pid and
         uint(finishes[0].get("returncode"), "guard return code") == 0,
         "guard process does not match the reported probe, has a mismatched process group, or failed")
    first_monotonic = finite(limits.get("monotonic"), "guard limits time")
    previous_monotonic = first_monotonic
    for index, row in enumerate(rows[1:], start=1):
        now = finite(row.get("monotonic"), f"guard event {index} time")
        need(now > previous_monotonic, "guard event times are not strictly increasing")
        previous_monotonic = now
    initial_swap = uint(samples[0].get("swap_used_bytes"), "initial swap")
    min_pressure = 100.0
    peak_footprint = 0
    max_swap_growth = 0
    physical_sizes: set[int] = set()
    headrooms: list[int] = []
    for index, row in enumerate(samples):
        physical = uint(row.get("physical_bytes"), f"guard sample {index} physical bytes", minimum=1)
        headroom = uint(row.get("headroom_bytes"), f"guard sample {index} headroom")
        pressure = finite(row.get("pressure_free_pct"), f"guard sample {index} pressure")
        free_spec = uint(row.get("free_spec_bytes"), f"guard sample {index} free/speculative bytes")
        file_backed = uint(row.get("file_backed_bytes"), f"guard sample {index} file-backed bytes")
        speculative = uint(row.get("speculative_bytes"), f"guard sample {index} speculative bytes")
        footprint = uint(row.get("group_footprint_bytes"), f"guard sample {index} footprint")
        swap = uint(row.get("swap_used_bytes"), f"guard sample {index} swap")
        live_processes = uint(row.get("group_live_processes"), f"guard sample {index} live process count")
        need(10.0 <= pressure <= 100.0 and
             headroom >= ten_percent_floor(physical) and headroom <= physical and
             free_spec <= physical and speculative <= free_spec and
             file_backed >= speculative and headroom == free_spec - speculative + file_backed and
             footprint <= FOOTPRINT_CAP and
             swap - initial_swap <= SWAP_GROWTH_CAP,
             f"guard sample {index} crossed a required resource bound")
        need((index != 0 or live_processes == 0) and
             (index == 0 or index == len(samples) - 1 or live_processes > 0) and
             (index != len(samples) - 1 or live_processes == 0),
             "guard startup, in-run, or final samples disagree with the process lifecycle")
        physical_sizes.add(physical)
        headrooms.append(headroom)
        min_pressure = min(min_pressure, pressure)
        peak_footprint = max(peak_footprint, footprint)
        max_swap_growth = max(max_swap_growth, swap - initial_swap)
        if index == 0:
            need(headroom >= ten_percent_floor(physical) + FOOTPRINT_CAP,
                 "guard startup sample lacks 10 percent floor plus footprint headroom")
    need(len(physical_sizes) == 1,
         "guard physical-memory size changed during the run")
    return {"sha256": digest, "probe_process_id": pid, "sample_count": len(samples),
            "minimum_free_memory_percent": min_pressure,
            "memory_accounting": "file-cache",
            "minimum_sampled_headroom_bytes": min(headrooms),
            "physical_bytes": next(iter(physical_sizes)),
            "peak_group_footprint_bytes": peak_footprint,
            "maximum_swap_growth_bytes": max_swap_growth,
            "footprint_cap_bytes": FOOTPRINT_CAP, "swap_growth_cap_bytes": SWAP_GROWTH_CAP}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--guard", required=True, type=Path,
                        help="required memory-guard JSONL log from the measured run")
    parser.add_argument("--output", type=Path,
                        help="fresh summary file; existing paths are never replaced")
    args = parser.parse_args(argv)
    try:
        report_raw = args.report.read_bytes()
        summary = validate_report(json.loads(report_raw), hashlib.sha256(report_raw).hexdigest())
        guard_raw = args.guard.read_bytes()
        events = [json.loads(line) for line in guard_raw.splitlines() if line.strip()]
        summary["guard"] = validate_guard(events, summary["probe_process_id"],
                                          hashlib.sha256(guard_raw).hexdigest())
        encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(encoded)
        else:
            sys.stdout.write(encoded)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"PixelUMM CogniGraph analysis rejected report: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
