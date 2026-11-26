#!/usr/bin/env python3
"""Qualify reused-buffer FFN pairs with external GPU frequency samples.

Command-buffer root spans include stalls. They are not per-shader service
times, and this isolated fixture cannot establish full-block acceleration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


_SPEC = importlib.util.spec_from_file_location(
    "pixelumm_frequency_profile_reused", Path(__file__).with_name("pixelumm_frequency_profile.py"))
assert _SPEC and _SPEC.loader
F = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(F)

SCHEMA = "cogni_ml.pixelumm_ffn_stage_timing_analysis/v1"
REPORT_SCHEMA = "cogni_ml.pixelumm_ffn_stage_timing_probe/v1"
LAUNCH_SCHEMA = "cogni_ml.pixelumm_ffn_stage_timing_launch/v1"
SCHEDULE = [("A_warm", "A", True), ("B_warm", "B", True),
            ("A_1", "A", False), ("B_1", "B", False),
            ("B_2", "B", False), ("A_2", "A", False),
            ("B_3", "B", False), ("A_3", "A", False),
            ("A_4", "A", False), ("B_4", "B", False)]
# Each tuple is (baseline, candidate); both temporal orders are mandatory.
PAIRS = [(2, 3), (5, 4), (7, 6), (8, 9)]
MINIMUM_MEDIAN_GAIN = 0.03
PRODUCT_ELEMENTS = 5760 * 12288
DOWN_ELEMENTS = 5760 * 4096
GRAPH_COUNTS = {"output_only_topology_graph_encodes": 640,
                "output_only_split_down_graph_encodes": 640,
                "captured_topology_graph_encodes": 0,
                "captured_split_down_graph_encodes": 0,
                "reference_silu_kernel_encodes": 0}


def _clock(value: object) -> dict[str, int]:
    anchor, _ = F._validate_launch({
        "schema": F.LAUNCH_SCHEMA, "clock_anchor": value,
        "frequency_gate": {"max_mean_relative_difference": 0.02,
                           "minimum_interior_samples": 5,
                           "all_observed_samples_valid": True,
                           "minimum_mean_active_residency_pct": 99.0}})
    return anchor


def _execution(report: Mapping[str, Any]) -> None:
    protocol = F._object(report.get("protocol"), "report.protocol")
    for key, expected in (("tokens", 5760), ("hidden", 4096), ("ffn", 12288),
                          ("block", 0), ("repeat_count_per_sample", 128), ("terminal_waits", 1)):
        F._need(F._integer(protocol.get(key), f"protocol.{key}") == expected,
                f"FFN protocol {key} differs")
    F._need(protocol.get("window") == "w0" and protocol.get("candidate_topology") == "stage_outputs" and
            protocol.get("api_phase_roots_enabled") is False and protocol.get("no_generation") is True,
            "FFN topology or fixture selection differs")
    parity = F._object(report.get("parity"), "report.parity")
    F._need(parity.get("atol") == 2e-5 and parity.get("rtol") == 2e-5,
            "FFN tolerance differs from the unchanged upstream gate")
    execution = F._object(report.get("execution"), "report.execution")
    for key, expected in (("encoding_named_allocation_delta", 0), ("preterminal_readbacks", 0),
                          ("total_readbacks_after_wait", 1), ("wait_count", 1),
                          ("maximum_named_bytes", 2 * 1024**3)):
        F._need(F._integer(execution.get(key), f"execution.{key}") == expected,
                f"FFN execution {key} differs")
    named = F._integer(execution.get("named_bytes"), "named bytes", minimum=1)
    resources = F._object(execution.get("named_resources"), "named resources")
    F._need(named <= 2 * 1024**3 and sum(F._integer(size, "resource bytes", minimum=1)
                                        for size in resources.values()) == named,
            "named buffer inventory differs or exceeds its 2 GiB budget")
    baseline = F._array(execution.get("baseline_operation_counts"), "baseline counts")
    F._need([F._integer(count, "baseline count") for count in baseline] == [641] * 6 + [2],
            "FFN baseline operation counts differ")
    counts = F._object(execution.get("candidate_graph_operation_counts"), "candidate counts")
    F._need({key: F._integer(value, f"candidate count {key}") for key, value in counts.items()} == GRAPH_COUNTS,
            "FFN candidate operation counts differ")
    F._need(execution.get("metric_failures") == [] and execution.get("execution_failures") == [],
            "FFN execution retains reported failures")
    F._need(execution.get("api_profile_pass") is True, "FFN API profile gate did not pass")
    phases = F._array(report.get("api_profile"), "report.api_profile")
    F._need([phase.get("phase_id") for phase in phases] == list(range(6)),
            "FFN API phase inventory differs")
    for phase in phases:
        operations = F._object(phase.get("operations"), "API phase operations")
        F._need(set(operations) == {"graph_encode", "commit_and_continue", "t5_linear_encode"},
                "FFN API operation inventory differs")
        for operation, stats_raw in operations.items():
            stats = F._object(stats_raw, "API operation statistics")
            count = F._integer(stats.get("count"), "API call count")
            if phase["phase_id"] == 5 and operation == "graph_encode":
                F._need(count == 641 * 3 + 640 * 2, "FFN graph count differs from actual workload")
            elif phase["phase_id"] == 5 and operation == "commit_and_continue":
                F._need(count > 0, "FFN continuation count is empty")
            else:
                F._need(count == 0, "API calls escaped the FFN phase or used another linear path")
            for kind in ("wall", "thread_cpu"):
                total = F._integer(stats.get(f"{kind}_total_ns"), "API inclusive duration")
                nested = F._integer(stats.get(f"nested_{kind}_total_ns"), "API nested duration")
                exclusive = F._integer(stats.get(f"exclusive_{kind}_total_ns"), "API exclusive duration")
                # mps_api_profile.h: Graph subtracts child continuations;
                # continuation nested totals instead describe the subset
                # called under Graph, so its own exclusive equals inclusive.
                F._need(total == (nested + exclusive if operation == "graph_encode" else exclusive) and
                        nested <= total, "API inclusive/nested/exclusive durations disagree")
                F._need(count > 0 or total == nested == exclusive == 0,
                        "empty API operation retains nonzero durations")
        for key in ("nested_count", "nested_wall_total_ns", "nested_thread_cpu_total_ns"):
            F._need(operations["graph_encode"][key] == operations["commit_and_continue"][key],
                    "Graph child continuation totals differ from continuation subset totals")
    metrics = F._array(execution.get("metrics"), "execution.metrics")
    F._need(len(metrics) == 57, "FFN metric inventory must have all 57 slots")
    for slot, metric_raw in enumerate(metrics):
        metric = F._object(metric_raw, f"metric {slot}")
        expected_elements = (PRODUCT_ELEMENTS if slot % 2 == 0 else DOWN_ELEMENTS) if slot < 20 else (
            0 if slot < 54 else PRODUCT_ELEMENTS if slot < 56 else DOWN_ELEMENTS)
        F._need(F._integer(metric.get("slot"), "metric slot") == slot and
                F._integer(metric.get("elements"), "metric elements") == expected_elements,
                f"FFN metric {slot} coverage differs")
        F._need(F._integer(metric.get("failures"), "metric failures") == (1 if slot == 56 else 0) and
                F._integer(metric.get("nonfinite"), "metric nonfinite") == (1 if slot == 56 else 0),
                f"FFN metric {slot} failed or lost the positive control")
        different = F._integer(metric.get("different_words"), "metric different words")
        F._need(different <= expected_elements and (slot not in range(20, 54) and slot != 56 or different == 0),
                f"FFN metric {slot} has invalid inactive/control word differences")


def _report(report_raw: object, anchor: Mapping[str, int]
            ) -> tuple[list[dict[str, Any]], dict[int, tuple[int, int]]]:
    report = F._object(report_raw, "report")
    F._finite_tree(report, "report")
    F._need(report.get("schema") == REPORT_SCHEMA, "FFN timing report schema mismatch")
    for key in ("pass", "correctness_pass", "execution_pass", "timing_isolation_pass"):
        F._need(report.get(key) is True, f"FFN report {key} did not pass")
    for key in ("production_promoted", "performance_promoted", "full_generation_executed"):
        F._need(report.get(key) is False, f"FFN report has an unexpected {key} claim")
    _execution(report)
    clocks = F._object(report.get("clock_anchors"), "report.clock_anchors")
    before = _clock(clocks.get("before_encode"))
    after = _clock(clocks.get("after_terminal_wait"))
    F._need(anchor["midpoint_ns"] <= before["midpoint_ns"] < after["midpoint_ns"],
            "native before/after anchors do not follow the launch anchor")
    for clock in (before, after):
        residual = clock["unix_ns"] - (anchor["unix_ns"] + clock["midpoint_ns"] - anchor["midpoint_ns"])
        F._need(abs(residual) <= 100_000_000, "native clock anchor drift exceeds 100 ms")

    roots = F._array(report.get("gpu_roots"), "report.gpu_roots")
    F._need(bool(roots), "report has no completed GPU roots")
    intervals: list[tuple[float, float]] = []
    for index, root_raw in enumerate(roots):
        root = F._object(root_raw, f"root {index}")
        F._need(F._integer(root.get("sequence"), "root sequence") == index and
                root.get("completed") is True and root.get("has_error") is False,
                f"root {index} has incomplete status or noncontiguous sequence")
        start = F._finite(root.get("gpu_start_seconds"), "GPU root start", minimum=0.000001)
        end = F._finite(root.get("gpu_end_seconds"), "GPU root end", minimum=0.000001)
        F._need(start < end, f"root {index} GPU interval is empty or reversed")
        F._need(before["monotonic_before_ns"] - 1_000_000 <= start * 1e9 and
                end * 1e9 <= after["monotonic_after_ns"] + 1_000_000,
                f"root {index} is outside the native Mach-clock envelope")
        intervals.append((start, end))

    rows = F._array(report.get("schedule"), "report.schedule")
    F._need(len(rows) == len(SCHEDULE), "FFN timing schedule count differs")
    output: list[dict[str, Any]] = []
    ranges: dict[int, tuple[int, int]] = {}
    previous_end = 0
    for index, (raw, expected) in enumerate(zip(rows, SCHEDULE)):
        row = F._object(raw, f"schedule {index}")
        F._need((row.get("label"), row.get("mode"), row.get("warmup")) == expected and
                type(row.get("warmup")) is bool and
                F._integer(row.get("repeats"), "repeats") == 128,
                f"schedule {index} differs from the fixed A/B workload")
        first = F._integer(row.get("first_root"), "first_root")
        end = F._integer(row.get("end_root"), "end_root")
        F._need(previous_end <= first < end <= len(roots), "timed root ranges overlap or are invalid")
        previous_end = end
        selected = intervals[first:end]
        union = F._union_length(selected)
        F._need(math.isclose(F._finite(row.get("gpu_root_union_seconds"), "reported GPU union"),
                             union, rel_tol=1e-10, abs_tol=1e-7), "reported GPU root union differs")
        outside = intervals[:first] + intervals[end:]
        overlap = F._union_length([(max(a, c), min(b, d)) for a, b in selected for c, d in outside
                                   if min(b, d) > max(a, c)])
        F._need(overlap == 0.0 and row.get("outside_root_overlap_seconds") == 0.0,
                "timed FFN roots overlap work outside their range")
        start_s, end_s = min(a for a, _ in selected), max(b for _, b in selected)
        for key, observed in (("first_gpu_start_seconds", start_s), ("last_gpu_end_seconds", end_s)):
            F._need(math.isclose(F._finite(row.get(key), key), observed,
                                 rel_tol=0.0, abs_tol=1e-7), f"{key} differs from raw GPU roots")
        F._need(F._integer(row.get("gpu_root_count"), "gpu_root_count") == len(selected),
                "reported timed root count differs")
        wall = F._finite(row.get("encode_wall_seconds"), "encode_wall_seconds", minimum=0.000001)
        ranges[index] = (round(start_s * 1e9), round(end_s * 1e9))
        output.append({"sample": index, "label": expected[0], "mode": expected[1],
                       "warmup": expected[2], "repeats": 128,
                       "gpu_root_union_seconds": union, "gpu_seconds_per_ffn": union / 128,
                       "gpu_root_span_seconds": end_s - start_s, "encode_wall_seconds": wall,
                       "gpu_root_count": len(selected)})
    return output, ranges


def analyze_documents(report_raw: object, launch_raw: object, summary_raw: object,
                      samples: Sequence[Mapping[str, Any]], *, samples_bytes: bytes | None = None
                      ) -> dict[str, Any]:
    launch = F._object(launch_raw, "launch")
    F._need(launch.get("schema") == LAUNCH_SCHEMA, "FFN timing launch schema mismatch")
    anchor, gate = F._validate_launch(dict(launch, schema=F.LAUNCH_SCHEMA))
    rows, ranges = _report(report_raw, anchor)
    observer = F._validate_observer(summary_raw, samples, ranges, anchor, samples_bytes)
    exclusion_ms = (anchor["half_bracket_uncertainty_ns"] + 999_999) // 1_000_000 + 1
    utc_ranges = {str(index): (F._map_monotonic_to_unix_ms(start, anchor) + exclusion_ms,
                              F._map_monotonic_to_unix_ms(end, anchor) - exclusion_ms)
                  for index, (start, end) in ranges.items()}
    power = F._OBSERVER.summarize_utc_ranges(samples, utc_ranges)
    for index, row in enumerate(rows):
        measured = power[str(index)]
        count = F._integer(measured.get("eligible_samples"), "eligible samples")
        valid = F._integer(measured.get("valid_samples"), "valid samples")
        enough = count >= gate["minimum_interior_samples"] and valid == count and measured.get("valid_sample_fraction") == 1.0
        frequency = (measured.get("gpu_active_frequency_mhz") or {}).get("mean")
        active = (measured.get("gpu_active_residency_pct") or {}).get("mean")
        row.update({"interior_samples": count, "mean_frequency_mhz": frequency,
                    "mean_active_residency_pct": active, "sample_coverage_pass": enough,
                    "active_residency_pass": active is not None and active >= gate["minimum_mean_active_residency_pct"]})
    pairs = []
    for a_index, b_index in PAIRS:
        a, b = rows[a_index], rows[b_index]
        clocks_present = all(row["mean_frequency_mhz"] is not None and row["mean_frequency_mhz"] > 0
                             for row in (a, b))
        relative = abs(b["mean_frequency_mhz"] - a["mean_frequency_mhz"]) / a["mean_frequency_mhz"] if clocks_present else None
        qualified = (clocks_present and relative <= gate["max_mean_relative_difference"] and
                     all(row["sample_coverage_pass"] and row["active_residency_pass"] for row in (a, b)))
        gain = 1 - b["gpu_seconds_per_ffn"] / a["gpu_seconds_per_ffn"]
        pairs.append({"baseline_sample": a_index, "candidate_sample": b_index,
                      "order": "AB" if a_index < b_index else "BA",
                      "relative_mean_frequency_difference": relative,
                      "frequency_qualified": bool(qualified),
                      "observed_gpu_root_union_gain": gain,
                      "qualified_gpu_root_union_gain": gain if qualified else None})
    frequency_pass = all(row["frequency_qualified"] for row in pairs)
    median_gain = median(row["observed_gpu_root_union_gain"] for row in pairs)
    worth_full_block = frequency_pass and median_gain >= MINIMUM_MEDIAN_GAIN and all(
        row["observed_gpu_root_union_gain"] > 0 for row in pairs)
    return {"schema": SCHEMA, "samples": rows, "pairs": pairs,
            "frequency_gate": gate, "all_pairs_frequency_qualified": frequency_pass,
            "median_observed_gpu_root_union_gain": median_gain,
            "qualified_median_gpu_root_union_gain": median_gain if frequency_pass else None,
            "full_block_trial_admission": {"minimum_median_gain": MINIMUM_MEDIAN_GAIN,
                                            "every_pair_must_improve": True,
                                            "passed": worth_full_block},
            "observer_provenance": observer, "clock_anchor": anchor,
            "production_promoted": False, "full_block_speedup_established": False,
            "gpu_interval_caveat": "command-buffer interval bounds include stalls; not per-shader service time",
            "frequency_normalized_timing": False}


def validate_artifacts(report_path: Path, launch_path: Path, completion_path: Path
                       ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bind local execution receipts; these are provenance checks, not signatures."""
    report, launch, completion = (F._load_json(path) for path in (report_path, launch_path, completion_path))
    F._need(type(completion.get("returncode")) is int and completion["returncode"] == 0 and
            completion.get("failure") is None and completion.get("sources_unchanged") is True and
            type(completion.get("observer_returncode")) is int and completion["observer_returncode"] == 0,
            "execution completion receipt does not pass")
    artifacts = F._object(completion.get("artifact_sha256"), "completion artifact hashes")
    for path in (report_path, launch_path):
        F._need(artifacts.get(path.name) == hashlib.sha256(path.read_bytes()).hexdigest(),
                f"completion digest differs for {path.name}")
    binary = Path(launch["native_argv"][0])
    F._need(launch.get("binary_sha256") == hashlib.sha256(binary.read_bytes()).hexdigest(),
            "launched binary digest differs")
    parity = F._object(report.get("parity"), "report.parity")
    path = Path(parity["path"])
    F._need(path.is_absolute() and path.is_file(), "parity report path is unavailable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pins = F._object(launch.get("fixture_pins"), "launch.fixture_pins")
    F._need(parity.get("sha256") == digest == pins.get(str(path)), "stage parity report digest differs")
    document = F._load_json(path)
    F._need(document.get("schema") == parity.get("schema") == "cogni_ml.pixelumm_ffn_graph_probe/v1" and
            all(document.get(key) is True for key in ("pass", "probe_completed", "numerical_pass", "execution_pass")),
            "stage parity report did not pass")
    protocol = document["protocol"]
    F._need(protocol.get("candidate_topology") == parity.get("candidate_topology") == "stage_outputs" and
            protocol.get("stage_outputs_are_mandatory_targets") is True and
            protocol.get("output_only_front_stage_outputs") == ["graph.gate", "graph.up", "graph.silu", "graph.product"],
            "stage parity report targets a different executable")
    fixture = document["fixture"]
    for report_key, fixture_key in (("fixture_path", "path"), ("fixture_metadata_sha256", "metadata_sha256"),
                                    ("fixture_payload_sha256", "payload_sha256"),
                                    ("weight_cache_metadata_sha256", "weight_cache_metadata_sha256"),
                                    ("weight_cache_payload_sha256", "weight_cache_payload_sha256")):
        F._need(parity.get(report_key) == fixture.get(fixture_key), f"parity {report_key} differs")
    F._need(pins.get(parity["fixture_path"]) == parity["fixture_metadata_sha256"], "launch fixture pin differs")
    cache_path = launch["native_argv"][launch["native_argv"].index("--weight-cache") + 1]
    F._need(pins.get(cache_path) == parity["weight_cache_metadata_sha256"], "launch weight cache pin differs")
    repo = Path(__file__).resolve().parents[2]

    def normalized_sources(value: object) -> dict[str, Any]:
        # The correctness probe records its entry point absolutely and its
        # dependencies relatively; the timing probe canonicalizes every key.
        # Normalize both within this checkout and reject duplicate aliases.
        normalized = {}
        for source, source_digest in F._object(value, "parity source hashes").items():
            path = Path(source)
            relative = str((path if path.is_absolute() else repo / path).resolve().relative_to(repo))
            F._need(relative not in normalized, f"duplicate parity source alias: {relative}")
            normalized[relative] = source_digest
        return normalized

    sources = normalized_sources(parity.get("source_sha256"))
    F._need(len(sources) == 15 and sources == normalized_sources(document.get("source_sha256")),
            "parity source inventory differs")
    launch_sources = F._object(launch.get("sources_sha256"), "launch source hashes")
    for relative, source_digest in sources.items():
        if relative == "examples/pixelumm_ffn_graph_probe.cr" and relative not in launch_sources:
            # This is the earlier correctness executable's entry point, not a
            # dependency of the timing executable. The timing process checks
            # its current hash before Metal admission; check it again here.
            F._need(hashlib.sha256((repo / relative).read_bytes()).hexdigest() == source_digest,
                    "correctness entry point differs from the parity source pin")
            continue
        F._need(launch_sources.get(relative) == source_digest, f"launch source pin differs for {relative}")
    guard_path = Path(report["memory_guard"]["log_path"])
    F._need(artifacts.get(guard_path.name) == hashlib.sha256(guard_path.read_bytes()).hexdigest(),
            "guard completion digest differs")
    guard_events = F._parse_samples(guard_path.read_bytes())
    F._need(guard_events[-1].get("event") == "finished" and guard_events[-1].get("returncode") == 0 and
            not any(row.get("event") in ("blocked", "stopping", "cleanup_failed", "cleanup_incomplete") for row in guard_events),
            "memory guard did not complete cleanly")
    limits = [row for row in guard_events if row.get("event") == "limits"]
    starts = [row for row in guard_events if row.get("event") == "started"]
    F._need(len(limits) == len(starts) == 1 and
            report["memory_guard"].get("limits") == limits[0] and
            report["memory_guard"].get("started") == starts[0], "native guard records differ from the execution log")
    for key, expected in (("max_footprint_bytes", 8 * 1024**3), ("max_swap_growth_bytes", 64 * 1024**2),
                          ("runtime_floor_pct", 10), ("timeout_sec", 600)):
        F._need(F._integer(limits[0].get(key), f"guard.{key}") == expected, f"guard {key} differs")
    F._need(limits[0].get("memory_accounting") == "file-cache" and
            limits[0].get("runtime_floor_pct_exact") == "10" and
            F._integer(starts[0].get("pid"), "guard PID", minimum=1) ==
            F._integer(starts[0].get("pgid"), "guard PGID", minimum=1), "guard policy or owned identity differs")
    return report, launch, completion


def bind_observer_artifacts(result: dict[str, Any], completion: Mapping[str, Any],
                            summary_path: Path, samples_path: Path, raw_path: Path) -> None:
    """Legacy receipts can describe samples, but cannot admit another GPU trial."""
    hashes = F._object(completion.get("artifact_sha256"), "completion artifact hashes")
    paths = {"power/summary.json": summary_path, "power/samples.jsonl": samples_path,
             "power/powermetrics.txt": raw_path}
    present = [key in hashes for key in paths]
    F._need(all(present) or not any(present), "observer completion digest inventory is partial")
    bound = all(present)
    if bound:
        for key, path in paths.items():
            F._need(hashes[key] == hashlib.sha256(path.read_bytes()).hexdigest(),
                    f"observer completion digest differs for {key}")
    result["observer_bundle_bound_to_completion"] = bound
    result["frequency_labels_are_descriptive_only"] = not bound
    result["receipt_bound_frequency_gate_pass"] = bound and result["all_pairs_frequency_qualified"]
    result["full_block_trial_admission"]["observer_bundle_must_be_receipt_bound"] = True
    result["full_block_trial_admission"]["passed"] &= bound


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "launch", "completion", "observer-summary", "samples", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        F._need(not args.output.exists(), "output already exists")
        raw_samples = args.samples.read_bytes()
        summary = F._load_json(args.observer_summary)
        raw_capture = args.observer_summary.parent / "powermetrics.txt"
        raw = raw_capture.read_bytes()
        F._need(hashlib.sha256(raw).hexdigest() == summary["raw_capture"]["sha256"] and
                len(raw) == summary["raw_capture"]["size_bytes"], "raw capture digest/size differs")
        report, launch, completion = validate_artifacts(args.report, args.launch, args.completion)
        result = analyze_documents(report, launch, summary,
                                   F._parse_samples(raw_samples), samples_bytes=raw_samples)
        bind_observer_artifacts(result, completion, args.observer_summary, args.samples, raw_capture)
        result["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (args.report, args.launch, args.completion, args.observer_summary, args.samples)}
        result["analyzer_source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in (Path(__file__), Path(F.__file__),
                                                        Path(F._OBSERVER.__file__), Path(F._AA.__file__))}
        F._write_new(args.output, result)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output),
                      "all_pairs_frequency_qualified": result["all_pairs_frequency_qualified"],
                      "full_block_trial_admitted": result["full_block_trial_admission"]["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
