#!/usr/bin/env python3
"""CPU-only validator and descriptive summary for paired PixelUMM A/A profile runs.

This validates the fixed-baseline, same-input six-request schedule, existing
saved-output checks and lifecycle guards, delayed-encode spacing telemetry,
profile traces, and memory-recorder milestones. The output describes observed
timings and drift; it does not attribute causes or claim acceleration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


REPORT = "cogni_ml.pixelumm_native_aa_profile/v1"
SUMMARY = "cogni_ml.pixelumm_aa_profile_analysis/v1"
REQUEST_KINDS = ("a_aa_warm_off", "a_aa_0", "a_aa_1", "a_aa_2", "a_aa_3", "a_aa_late_off")
EXPECTED_COMMAND_SEGMENTS = 4579
PAUSE_NS = 50_000_000_000
ALLOWED_LATENESS_NS = 250_000_000
MEMORY_CLOCK_CAPTURE_TOLERANCE_NS = 5_000_000
I64_MAX = (1 << 63) - 1


def _load_order_analyzer() -> Any:
    path = Path(__file__).with_name("analyze_pixelumm_order_profile.py")
    spec = importlib.util.spec_from_file_location("pixelumm_order_profile_reused", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared PixelUMM order validator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORDER = _load_order_analyzer()


def _expected_schedule(reverse: bool) -> list[dict[str, Any]]:
    profiles = [False, True, False, False, True, False] if reverse else [False, False, True, True, False, False]
    return [{"kind": REQUEST_KINDS[index], "combined_attention": False,
             "alternate": False, "poisoned": index == 0, "profile": profiles[index]}
            for index in range(len(REQUEST_KINDS))]


def _int64(value: object, where: str) -> int:
    ORDER.need(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= I64_MAX,
               f"{where}: expected nonnegative Int64")
    return int(value)


def _validate_spacing(raw: object, request_index: int, previous_terminal_end_ns: int | None,
                      previous_gc_ns: int | None, memory_before_encode_ns: int,
                      terminal_wait_end_ns: int, memory_after_wait_ns: int) -> dict[str, Any]:
    spacing = ORDER.obj(raw, f"request {request_index} aa_spacing")
    ORDER.exact_keys(spacing, ("clock", "previous_terminal_end_ns", "diagnostic_ready_ns",
                               "target_encode_start_ns", "before_encode_ns", "requested_gap_ns",
                               "requested_sleep_ns", "actual_gap_ns", "lateness_ns",
                               "allowed_lateness_ns"), f"request {request_index} aa_spacing")
    ORDER.need(spacing["clock"] == "mach_absolute_time_nanoseconds",
               f"request {request_index}: A/A spacing clock changed")
    diagnostic_ready = _int64(spacing["diagnostic_ready_ns"], "diagnostic_ready_ns")
    before_encode = _int64(spacing["before_encode_ns"], "before_encode_ns")
    requested_gap = _int64(spacing["requested_gap_ns"], "requested_gap_ns")
    requested_sleep = _int64(spacing["requested_sleep_ns"], "requested_sleep_ns")
    allowed_lateness = _int64(spacing["allowed_lateness_ns"], "allowed_lateness_ns")
    terminal_wait_end = _int64(terminal_wait_end_ns, "terminal_wait_end_mach_ns")
    memory_before = _int64(memory_before_encode_ns, "memory request_before_encode timestamp")
    memory_after_wait = _int64(memory_after_wait_ns, "memory request_after_terminal_wait timestamp")

    ORDER.need(requested_gap == PAUSE_NS and allowed_lateness == ALLOWED_LATENESS_NS,
               f"request {request_index}: requested pause or lateness bound changed")
    ORDER.need(diagnostic_ready <= before_encode < terminal_wait_end,
               f"request {request_index}: diagnostic/encode/terminal timestamp order is invalid")
    ORDER.need(0 <= memory_before - before_encode <= MEMORY_CLOCK_CAPTURE_TOLERANCE_NS and
               0 <= memory_after_wait - terminal_wait_end <= MEMORY_CLOCK_CAPTURE_TOLERANCE_NS,
               f"request {request_index}: spacing timestamps disagree with memory milestones")

    if request_index == 0:
        ORDER.need(spacing["previous_terminal_end_ns"] is None and
                   spacing["target_encode_start_ns"] is None and
                   spacing["actual_gap_ns"] is None and spacing["lateness_ns"] is None and
                   requested_sleep == 0,
                   "request 0: initial A/A spacing must not claim a preceding wait or sleep")
        ORDER.need(previous_terminal_end_ns is None and previous_gc_ns is None,
                   "request 0: unexpected prior-request timestamps")
        return {"request": 0, "previous_terminal_end_ns": None,
                "diagnostic_ready_ns": diagnostic_ready, "target_encode_start_ns": None,
                "before_encode_ns": before_encode, "requested_gap_ns": requested_gap,
                "requested_sleep_ns": 0, "actual_gap_ns": None, "lateness_ns": None,
                "allowed_lateness_ns": allowed_lateness,
                "initial_ready_to_encode_seconds": (before_encode - diagnostic_ready) / 1e9}

    previous_terminal = _int64(spacing["previous_terminal_end_ns"], "previous_terminal_end_ns")
    target = _int64(spacing["target_encode_start_ns"], "target_encode_start_ns")
    actual_gap = _int64(spacing["actual_gap_ns"], "actual_gap_ns")
    lateness = _int64(spacing["lateness_ns"], "lateness_ns")
    ORDER.need(previous_terminal == previous_terminal_end_ns,
               f"request {request_index}: prior terminal wait timestamp does not link to previous request")
    ORDER.need(previous_gc_ns is not None and diagnostic_ready >= previous_gc_ns,
               f"request {request_index}: diagnostic readiness predates previous post-diagnostic GC")
    ORDER.need(target == previous_terminal + requested_gap,
               f"request {request_index}: encode target is not prior terminal end plus 50 seconds")
    ORDER.need(requested_sleep == max(0, target - diagnostic_ready),
               f"request {request_index}: requested sleep does not follow diagnostic readiness")
    ORDER.need(actual_gap == before_encode - previous_terminal and
               lateness == before_encode - target and 0 <= lateness <= allowed_lateness,
               f"request {request_index}: actual gap or lateness is invalid")
    ORDER.need(diagnostic_ready >= previous_terminal,
               f"request {request_index}: diagnostic readiness predates prior terminal wait")
    return {"request": request_index, "previous_terminal_end_ns": previous_terminal,
            "diagnostic_ready_ns": diagnostic_ready, "target_encode_start_ns": target,
            "before_encode_ns": before_encode, "requested_gap_ns": requested_gap,
            "requested_sleep_ns": requested_sleep, "actual_gap_ns": actual_gap,
            "actual_gap_seconds": actual_gap / 1e9, "lateness_ns": lateness,
            "lateness_seconds": lateness / 1e9, "allowed_lateness_ns": allowed_lateness,
            "diagnostic_ready_to_encode_seconds": (before_encode - diagnostic_ready) / 1e9}


def _memory_request_timestamps(raw: object, request_count: int) -> dict[int, dict[str, int]]:
    rows = ORDER.arr(raw, "memory_phases")
    wanted = {"request_before_encode", "request_after_terminal_wait", "after_post_diagnostic_gc"}
    result: dict[int, dict[str, int]] = {index: {} for index in range(request_count)}
    for index, value in enumerate(rows):
        event = ORDER.obj(value, f"memory event {index}")
        request = event.get("request")
        phase = event.get("phase")
        if request in result and phase in wanted:
            result[request][phase] = _int64(event.get("guard_monotonic_ns"),
                                             f"memory event {index} guard_monotonic_ns")
    return result


def _validate_report(data_raw: object, report_sha256: str, path: str,
                     expected_segments: int) -> dict[str, Any]:
    data = ORDER.obj(data_raw, "report")
    ORDER.need(data.get("schema") == REPORT and data.get("generation") is False and
               data.get("scope") == "native_36_blocks_plus_global_head_in_one_gpu_chain" and
               data.get("profile_aa") is True and data.get("profile_combined_attention") is False and
               data.get("compare_combined_attention") is False and data.get("profile_variance") is False and
               data.get("profile_shared_gate_up") is False and data.get("compare_shared_gate_up") is False and
               data.get("reverse_combined_attention") is False and data.get("correctness_only") is False,
               "report schema/scope/mode flags are not the fixed-baseline PixelUMM A/A profile")
    reverse = ORDER.boolean(data.get("reverse_aa_profile"), "reverse_aa_profile")
    ORDER.need(ORDER.finite(data.get("atol"), "atol") == ORDER.ATOL and
               ORDER.finite(data.get("rtol"), "rtol") == ORDER.RTOL,
               "saved-capture comparator tolerance is not 2e-5/2e-5")
    ORDER.need(data.get("grid") == ORDER.GRID and data.get("block_indices") == list(range(ORDER.BLOCKS)) and
               data.get("output_shape") == [1, 3, 24, 704, 1280],
               "report geometry differs from the full 36-block J=21120 fixture")
    ORDER.need(ORDER.finite(data.get("aa_pause_seconds"), "aa_pause_seconds") == 50.0 and
               ORDER.finite(data.get("aa_allowed_lateness_seconds"), "aa_allowed_lateness_seconds") == 0.25,
               "report A/A pause or lateness allowance changed")
    ORDER.need(data.get("passed") is True and data.get("failure") is None and
               data.get("cleanup_quarantined") is False and data.get("spacing_abort_before_submission", False) is False and
               ORDER.uint(data.get("repeats"), "repeats") == 1 and
               ORDER.uint(data.get("expected_request_count"), "expected request count") == 6,
               "report is incomplete, failed, or did not finish the exact six-request diagnostic")
    ORDER.need(data.get("duplicate_head_input_bytes") == 0 and data.get("shared_block_workspace") is True and
               data.get("persistent_block_output_banks") == 2 and
               data.get("head_scratch_aliases_block_workspace") is False,
               "report head/block buffer topology changed")
    for field in ("reference_sha256", "alternate_reference_sha256", "head_reference_sha256",
                  "head_alternate_reference_sha256", "bf16_cache_sha256"):
        ORDER.digest(data.get(field), f"report {field}")
    for field in ("named_bytes", "block_named_bytes", "head_owned_named_bytes", "head_weight_named_bytes",
                  "named_budget_bytes"):
        ORDER.uint(data.get(field), f"report {field}")
    ORDER.need(data["named_bytes"] == data["block_named_bytes"] + data["head_owned_named_bytes"] +
               data["head_weight_named_bytes"] and
               data["named_bytes"] <= data["named_budget_bytes"] <= 24 * 1024**3,
               "report named-byte inventory exceeds its budget or does not add up")
    capture_order = ORDER._expected_capture_order()
    ORDER.need(data.get("compared_capture_order") == capture_order,
               "report did not declare the complete ordered 43-capture comparison")

    schedule = _expected_schedule(reverse)
    aa_rows = ORDER.arr(data.get("aa_schedule"), "aa_schedule")
    combined_rows = ORDER.arr(data.get("combined_attention_schedule"), "combined_attention_schedule")
    shared_rows = ORDER.arr(data.get("shared_gate_up_schedule"), "shared_gate_up_schedule")
    ORDER.need(len(aa_rows) == len(combined_rows) == len(shared_rows) == 6,
               "A/A, combined-attention, and shared-gate schedules must each contain six arms")
    for index, expected in enumerate(schedule):
        aa_arm = ORDER.obj(aa_rows[index], f"A/A schedule arm {index}")
        ORDER.exact_keys(aa_arm, ("kind", "combined_attention", "alternate", "poisoned", "profiling"),
                         f"A/A schedule arm {index}")
        ORDER.need(aa_arm["kind"] == expected["kind"] and
                   aa_arm["combined_attention"] is False and aa_arm["alternate"] is False and
                   aa_arm["poisoned"] is expected["poisoned"] and
                   aa_arm["profiling"] is expected["profile"],
                   f"A/A schedule arm {index} differs from fixed normal/reverse profile placement")
        combined = ORDER.obj(combined_rows[index], f"combined-attention schedule arm {index}")
        ORDER.exact_keys(combined, ("kind", "combined_attention", "alternate", "poisoned", "profiling"),
                         f"combined-attention schedule arm {index}")
        ORDER.need(combined["kind"] == expected["kind"] and
                   combined["combined_attention"] is False and combined["alternate"] is False and
                   combined["poisoned"] is expected["poisoned"] and
                   combined["profiling"] is expected["profile"] and combined == aa_arm,
                   f"combined-attention schedule arm {index} disagrees with the A/A schedule")
        shared = ORDER.obj(shared_rows[index], f"shared gate/up schedule arm {index}")
        ORDER.exact_keys(shared, ("kind", "shared_gate_up", "profiling"),
                         f"shared gate/up schedule arm {index}")
        ORDER.need(shared["kind"] == expected["kind"] and shared["shared_gate_up"] is False and
                   shared["profiling"] is expected["profile"],
                   f"shared gate/up schedule arm {index} unexpectedly changes FFN mode")

    request_rows = ORDER.arr(data.get("requests"), "requests")
    ORDER.need(len(request_rows) == len(schedule), "expected six request rows")
    # Reject segment-count drift before parsing the expensive root/profile arrays.
    for index, raw_request in enumerate(request_rows):
        request = ORDER.obj(raw_request, f"request {index}")
        count = ORDER.uint(request.get("command_segments"), f"request {index} command segments", minimum=1)
        ORDER.need(count == expected_segments,
                   f"request {index}: expected exactly {expected_segments} command segments/roots")

    memory = ORDER.validate_memory_events(data.get("memory_phases"), request_count=6)
    memory_times = _memory_request_timestamps(data.get("memory_phases"), request_count=6)
    for index, timestamps in memory_times.items():
        ORDER.need(set(timestamps) == {"request_before_encode", "request_after_terminal_wait",
                                      "after_post_diagnostic_gc"},
                   f"request {index}: required spacing memory milestones are incomplete")

    requests = []
    request_inputs: set[str] = set()
    spacings = []
    terminal_ends: list[int] = []
    for index, (raw_request, expected) in enumerate(zip(request_rows, schedule)):
        request = ORDER.obj(raw_request, f"request {index}")
        parsed = ORDER._validate_request(raw_request, index, {
            "kind": expected["kind"], "combined_attention": False,
            "profile": expected["profile"], "poisoned": expected["poisoned"],
        })
        request_inputs.add(parsed["input_sha256"])
        terminal_end = _int64(request.get("terminal_wait_end_mach_ns"),
                              f"request {index} terminal_wait_end_mach_ns")
        previous_terminal = terminal_ends[index - 1] if index else None
        previous_gc = memory_times[index - 1]["after_post_diagnostic_gc"] if index else None
        spacings.append(_validate_spacing(request.get("aa_spacing"), index, previous_terminal, previous_gc,
                                           memory_times[index]["request_before_encode"], terminal_end,
                                           memory_times[index]["request_after_terminal_wait"]))
        terminal_ends.append(terminal_end)
        ORDER.need(parsed["command_segments"] == expected_segments,
                   f"request {index}: root/segment topology changed across A/A arms")
        parsed.update({"profiling": expected["profile"], "combined_attention": False,
                       "timing": parsed["timing"], "terminal_wait_end_mach_ns": terminal_end,
                       "aa_spacing": spacings[-1]})
        requests.append(parsed)
    ORDER.need(len(request_inputs) == 1,
               "A/A profile schedule must hold the same input A across all six requests")

    # The terminal wait is timestamped directly after the measured wait; confirm
    # every next-arm link used the actual prior request terminal-end value.
    for index in range(1, len(requests)):
        ORDER.need(spacings[index]["previous_terminal_end_ns"] == terminal_ends[index - 1],
                   f"request {index}: inter-request terminal-end link changed")

    profiled = [request for request in requests if request["profiling"]]
    ORDER.need(len(profiled) == 2, "expected exactly two profiled baseline requests")
    return {"path": path, "sha256": report_sha256, "reverse": reverse,
            "same_input_sha256": next(iter(request_inputs)), "fixture_pins": {
                key: data[key] for key in ("reference_sha256", "alternate_reference_sha256",
                    "head_reference_sha256", "head_alternate_reference_sha256", "bf16_cache_sha256")},
            "requests": requests, "profiled_request_count": len(profiled),
            "memory_event_gaps": memory, "spacings": spacings,
            "capture_order": capture_order, "expected_command_segments": expected_segments}


def analyze_report(data_raw: object, report_sha256: str, path: str) -> dict[str, Any]:
    return _validate_report(data_raw, report_sha256, path, EXPECTED_COMMAND_SEGMENTS)


def _profile_summary(requests: list[dict[str, Any]], excluded: set[int]) -> dict[str, Any]:
    all_profiled = [request for request in requests if request["profiling"]]
    profiled = [request for request in all_profiled if request["index"] not in excluded]
    phase_wall = {phase: ORDER.distribution([request["phase"]["phase_wall_seconds"][phase]
                                             for request in profiled]) for phase in ORDER.PHASES}
    phase_cpu = {phase: ORDER.distribution([request["phase"]["phase_thread_cpu_seconds"][phase]
                                            for request in profiled]) for phase in ORDER.PHASES}
    per_request = []
    for request in all_profiled:
        per_request.append({"request": request["index"], "kind": request["kind"],
            "externally_captured": request["index"] in excluded,
            "timing": request["timing"], "phase_wall_seconds": request["phase"]["phase_wall_seconds"],
            "phase_thread_cpu_seconds": request["phase"]["phase_thread_cpu_seconds"],
            "per_block_phase_markers": request["phase"]["per_block"],
            "root_interval_metrics": request["root_metrics"],
            "candidate_root_duration_sums_nonadditive": request["phase"]["candidate_root_duration_sums"],
            "definite_root_duration_sums_nonadditive": request["phase"]["definite_root_duration_sums"]})
    return {"profiled_requests": [row for row in per_request if not row["externally_captured"]],
            "externally_captured_requests": [row for row in per_request if row["externally_captured"]],
            "phase_wall_seconds_across_profiled_requests": phase_wall,
            "phase_thread_cpu_seconds_across_profiled_requests": phase_cpu,
            "continuation_trace": ORDER._continuation_summary(profiled),
            "interpretation": "Declared external-capture arms are retained separately and excluded from distributions and continuation aggregates. Other profile measurements remain observer-on descriptions, not profiler calibration; root sums are non-additive and callback observations are not exact GPU completion times."}


def _timing_drift(requests: list[dict[str, Any]], excluded: set[int]) -> dict[str, Any]:
    fields = ("blocks_encode_seconds", "head_encode_seconds", "encode_seconds",
              "observed_encode_span_seconds", "terminal_wait_seconds", "total_seconds",
              "gpu_elapsed_seconds")
    by_profile = {}
    for enabled in (False, True):
        selected = [request for request in requests if request["profiling"] is enabled and
                    request["index"] not in excluded]
        by_profile["profile_on" if enabled else "profile_off"] = {
            "request_indices": [request["index"] for request in selected],
            "timings_seconds": {field: ORDER.distribution([request["timing"][field] for request in selected])
                                for field in fields}}
    deltas = []
    for previous, current in zip(requests, requests[1:]):
        if previous["index"] in excluded or current["index"] in excluded:
            continue
        deltas.append({"from_request": previous["index"], "to_request": current["index"],
                       "profile_status": [previous["profiling"], current["profiling"]],
                       "timing_delta_seconds": {field: current["timing"][field] - previous["timing"][field]
                                                 for field in fields}})
    return {"excluded_external_capture_request_indices": sorted(excluded),
            "timing_by_profile_selector": by_profile,
            "adjacent_request_timing_deltas_seconds": deltas,
            "start_to_start_seconds": ORDER.distribution([
                (current["aa_spacing"]["before_encode_ns"] - previous["aa_spacing"]["before_encode_ns"]) / 1e9
                for previous, current in zip(requests, requests[1:])]),
            "per_request_start_to_first_seconds": [
                {"request": row["index"], "seconds": (row["aa_spacing"]["before_encode_ns"] -
                    requests[0]["aa_spacing"]["before_encode_ns"]) / 1e9} for row in requests],
            "interpretation": "Declared capture arms and adjacent deltas touching them are excluded from timing comparisons; all spacing milestones remain recorded. Other repetitions are descriptive, not causal controls for capture carryover. Request timing covers encode and terminal wait; comparisons and cleanup follow it."}


def analyze_reports(reports: list[dict[str, Any]],
                    external_captures: list[str] | None = None) -> dict[str, Any]:
    ORDER.need(len(reports) == 2, "supply exactly two reports: one normal and one reverse profile placement")
    by_direction = {bool(report["reverse"]): report for report in reports}
    ORDER.need(set(by_direction) == {False, True}, "reports must contain one normal and one reverse A/A profile schedule")
    normal, reverse = by_direction[False], by_direction[True]
    ORDER.need(normal["fixture_pins"] == reverse["fixture_pins"] and
               normal["same_input_sha256"] == reverse["same_input_sha256"],
               "normal/reverse reports do not use the same fixtures and input A")
    ORDER.need(normal["expected_command_segments"] == reverse["expected_command_segments"],
               "normal/reverse command segment counts differ")
    exclusions: dict[bool, set[int]] = {False: set(), True: set()}
    for annotation in external_captures or []:
        parts = annotation.split(":")
        ORDER.need(len(parts) == 2 and parts[0] in ("normal", "reverse") and
                   parts[1] in tuple(str(index) for index in range(6)),
                   "external capture must be normal:N or reverse:N with request N in 0..5")
        direction, index = parts[0] == "reverse", int(parts[1])
        ORDER.need(index not in exclusions[direction], "duplicate external capture annotation")
        exclusions[direction].add(index)
    run_summaries = []
    for run in (normal, reverse):
        requests = run["requests"]
        excluded = exclusions[run["reverse"]]
        request_summaries = []
        for request in requests:
            request_summaries.append({"request": request["index"], "kind": request["kind"],
                "profiling": request["profiling"], "combined_attention": False,
                "externally_captured": request["index"] in excluded,
                "timing": request["timing"], "command_segments": request["command_segments"],
                "per_block_encode_host_seconds": request["per_block_encode_host_seconds"],
                "root_interval_metrics": request["root_metrics"],
                "captures": {"count": len(request["comparisons"]),
                             "all_finite_within_upstream_tolerance": True,
                             "different_words": [capture["different_words"] for capture in request["comparisons"]],
                             "max_abs": [capture["max_abs"] for capture in request["comparisons"]]},
                "aa_spacing": request["aa_spacing"],
                "terminal_wait_end_mach_ns": request["terminal_wait_end_mach_ns"]})
        run_summaries.append({"path": run["path"], "sha256": run["sha256"],
            "order": "reverse_profile_placement" if run["reverse"] else "normal_profile_placement",
            "same_input_sha256": run["same_input_sha256"], "requests": request_summaries,
            "memory_event_gaps": run["memory_event_gaps"],
            "profile_localization": _profile_summary(requests, excluded),
            "descriptive_drift": _timing_drift(requests, excluded)})
    return {"schema": SUMMARY, "input_report_schema": REPORT,
            "scope": "Two same-input, fixed-baseline, six-request A/A profiles with reversed profile placement; no generation.",
            "numerical_scope": {"atol": ORDER.ATOL, "rtol": ORDER.RTOL,
                "captures_per_request": 43, "all_captures_finite_and_within_tolerance": True,
                "different_bf16_words_preserved": True,
                "not_claimed": ["bitwise parity", "internal F32 softmax/probability parity",
                                "generation speed", "causal profiler effect", "acceleration"]},
            "lifecycle_scope": {"requests_per_run": 6, "terminal_waits_per_request": 1,
                "post_wait_capture_readbacks_per_request": 43,
                "preterminal_activation_readbacks_per_request": 0,
                "named_encode_allocations_per_request": 0,
                "command_segments_per_request": normal["expected_command_segments"],
                "continuation_events_checked_after_existing_terminal_wait": True},
            "pause_scope": {"requested_gap_seconds": PAUSE_NS / 1e9,
                "allowed_lateness_seconds": ALLOWED_LATENESS_NS / 1e9,
                "anchor": "prior terminal-wait return to the next pre-recorder, pre-encode milestone",
                "precision_limit": "The memory recorder and CPU timer setup run after this milestone and before encode_blocks; the timestamp is not the exact first GPU encode call.",
                "actual_intervals_and_lateness_recorded_per_request": True,
                "last_request_has_no_following_pause": True},
            "fixture_pins": normal["fixture_pins"], "runs": run_summaries,
            "external_capture_scope": {
                "declarations": external_captures or [],
                "authenticated_by_this_analyzer": False,
                "note": "Caller-supplied exclusions require a separately checked capture receipt. Undeclared capture status is unknown; later arms may retain capture effects. No causal observer calibration is claimed."},
            "analysis_limit": "The summary reports observed within-run timing drift and profile localization only. The paired runs do not establish causal attribution or acceleration. Nested candidate-root sums and continuation wall spans are not additive GPU work; callback observations are not exact GPU completion times."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path,
                        help="A/A profile report; repeat once for normal and once for reverse profile placement")
    parser.add_argument("--output", type=Path,
                        help="fresh summary file (exclusive create); defaults to stdout")
    parser.add_argument("--external-capture", action="append", default=[],
                        help="exclude a receipt-verified captured arm from timing aggregates, e.g. reverse:1; repeat as needed")
    args = parser.parse_args(argv)
    try:
        reports = []
        for path in args.report:
            raw = path.read_bytes()
            reports.append(analyze_report(json.loads(raw), hashlib.sha256(raw).hexdigest(), str(path)))
        summary = analyze_reports(reports, external_captures=args.external_capture)
        encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(encoded)
        else:
            sys.stdout.write(encoded)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"pixelumm A/A analysis rejected report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
