"""Hostile CPU-only tests for the fixed J5760 continuation-trace analyzer."""
from __future__ import annotations

import importlib.util
import math
import statistics
import struct
import unittest
from pathlib import Path
from typing import Any


ANALYZER_PATH = Path(__file__).parents[1] / "examples/support/analyze_pixelumm_ffn_aa_trace.py"
SPEC = importlib.util.spec_from_file_location("pixelumm_ffn_aa_trace_analysis", ANALYZER_PATH)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _digest(char: str) -> str:
    return char * 64


def _stats(count: int = 0, nested: int = 0, wall: int = 0, cpu: int = 0,
           nested_wall: int = 0, nested_cpu: int = 0, max_wall: int = 0,
           max_cpu: int = 0) -> dict[str, int]:
    return {
        "count": count,
        "wall_total_ns": wall,
        "wall_max_ns": max_wall,
        "thread_cpu_total_ns": cpu,
        "thread_cpu_max_ns": max_cpu,
        "nested_count": nested,
        "nested_wall_total_ns": nested_wall,
        "nested_thread_cpu_total_ns": nested_cpu,
        "exclusive_wall_total_ns": wall - nested_wall if count and nested else wall,
        "exclusive_thread_cpu_total_ns": cpu - nested_cpu if count and nested else cpu,
    }


def _build_report(trace_enabled: bool, report_index: int) -> dict[str, Any]:
    """A valid fixed-shape synthetic report; no model, fixture file, or GPU."""
    samples: list[dict[str, Any]] = []
    event_counts = [6] + [7] * 17
    cursor = 1  # initial root 0, then the first sample's explicit start boundary
    event_spec: list[tuple[int, bool]] = []
    sample_ranges: list[tuple[int, int, int, int]] = []
    for sample_index, internal_count in enumerate(event_counts):
        first_root = cursor
        event_spec.append((cursor - 1, False))  # start boundary
        for offset in range(internal_count):
            event_spec.append((cursor + offset, True))
        end_root = first_root + internal_count + 1
        event_spec.append((end_root - 1, False))  # end boundary
        sample_ranges.append((first_root, end_root, len(event_spec) - internal_count - 2,
                              len(event_spec)))
        cursor = end_root + 1  # the next sample has a separate start boundary

    assert len(event_spec) == 161 and cursor == 162
    origin = 1_000_000_000
    gap_ns = 2_000_000
    duration_ns = 1_000_000
    events = []
    root_start_ns: list[int] = []
    root_end_ns: list[int] = []
    root_tracked: list[int] = [origin]
    root_completion: list[int] = []
    root_status_track: list[int] = []
    previous_gpu_end = 0
    wall_by_events = cpu_by_events = nested_wall = nested_cpu = 0
    max_event_wall = max_event_cpu = 0
    for index, (before_root, nested) in enumerate(event_spec):
        begin = origin + 1_000_000 + index * gap_ns
        end = begin + duration_ns
        if index:
            root_tracked.append(events[-1]["end_mach_ticks"] + 100)
        if before_root == 0:
            gpu_start = begin + 100_000
        else:
            gpu_start = max(begin + 100_000, previous_gpu_end + 100_000)
        gpu_end = gpu_start + 3_000_000
        previous_gpu_end = gpu_end
        root_start_ns.append(gpu_start)
        root_end_ns.append(gpu_end)
        root_completion.append(gpu_end + 100_000)
        root_status_track.append(0)
        serial = index + 1
        api_serial = 2 * serial
        cpu_begin = 10_000_000_000 + index * gap_ns
        cpu_end = cpu_begin + 800_000
        event = {
            "serial": serial,
            "begin_mach_ticks": begin,
            "end_mach_ticks": end,
            "begin_thread_cpu_ns": cpu_begin,
            "end_thread_cpu_ns": cpu_end,
            "api_scope_serial": api_serial,
            "parent_graph_serial": api_serial - 1 if nested else 0,
            "before_root_sequence": before_root,
            "after_root_sequence": before_root + 1,
            "phase": 5,
            "graph_depth": 1 if nested else 0,
            "before_root_status": 0,
            "before_root_status_after_call": 4,
            "after_root_status": 0,
            "flags": 15 if nested else 14,
        }
        events.append(event)
        wall = end - begin
        cpu = cpu_end - cpu_begin
        wall_by_events += wall
        cpu_by_events += cpu
        if nested:
            nested_wall += wall
            nested_cpu += cpu
        max_event_wall = max(max_event_wall, wall)
        max_event_cpu = max(max_event_cpu, cpu)

    roots = []
    for sequence in range(162):
        tracked = root_tracked[sequence] if sequence < len(root_tracked) else events[-1]["end_mach_ticks"] + 100
        # Final root is created by the last event; all others are created at the
        # next event boundary or at trace start.
        if sequence == 0:
            tracked = origin
        elif sequence <= len(events):
            tracked = events[sequence - 1]["end_mach_ticks"] + 100
        gpu_start = root_start_ns[sequence] if sequence < len(root_start_ns) else previous_gpu_end + 100_000
        gpu_end = root_end_ns[sequence] if sequence < len(root_end_ns) else gpu_start + 3_000_000
        completion = max(root_completion[sequence] if sequence < len(root_completion) else gpu_end + 100_000,
                         tracked)
        start_seconds = gpu_start / 1e9
        end_seconds = gpu_end / 1e9
        roots.append({
            "sequence": sequence,
            "tracked_or_observed_mach_ticks": tracked,
            "completion_observed_mach_ticks": completion,
            "gpu_start_time_bits": struct.unpack("=Q", struct.pack("=d", start_seconds))[0],
            "gpu_end_time_bits": struct.unpack("=Q", struct.pack("=d", end_seconds))[0],
            "status_at_track_or_enable": 0,
            "final_status": 4,
            "has_error": 0,
            "completion_source": 1,
            "gpu_start_seconds": start_seconds,
            "gpu_end_seconds": end_seconds,
        })

    # Patch the first root's GPU interval to remain outside sample 0. It still
    # covers its own explicit start-boundary call, which the algorithm excludes.
    first_event = events[0]
    roots[0]["gpu_start_seconds"] = (first_event["begin_mach_ticks"] + 100_000) / 1e9
    roots[0]["gpu_end_seconds"] = roots[0]["gpu_start_seconds"] + 100_000 / 1e9
    roots[0]["gpu_start_time_bits"] = struct.unpack("=Q", struct.pack("=d", roots[0]["gpu_start_seconds"]))[0]
    roots[0]["gpu_end_time_bits"] = struct.unpack("=Q", struct.pack("=d", roots[0]["gpu_end_seconds"]))[0]
    roots[0]["completion_observed_mach_ticks"] = int(roots[0]["gpu_end_seconds"] * 1e9) + 100_000

    root_timing = [{
        "sequence": root["sequence"],
        "gpu_start_seconds": root["gpu_start_seconds"],
        "gpu_end_seconds": root["gpu_end_seconds"],
        "completed": True,
        "has_error": False,
    } for root in roots]

    for sample_index, (first, end_root, _event_start, _event_end) in enumerate(sample_ranges):
        selected = root_timing[first:end_root]
        gpu_union = sum(row["gpu_end_seconds"] - row["gpu_start_seconds"] for row in selected)
        samples.append({
            "index": sample_index,
            "warmup": sample_index < 2,
            "capture_raw": sample_index == 0,
            "mode": "baseline",
            "first_root": first,
            "end_root": end_root,
            "encode_wall_seconds": 0.01,
            "gpu_root_union_seconds": gpu_union,
            "outside_root_overlap_seconds": 0.0,
            "isolated_root_timing": True,
        })

    nested_count = sum(event["graph_depth"] > 0 for event in events)
    commit_stats = _stats(161, nested_count, wall_by_events, cpu_by_events,
                          nested_wall, nested_cpu, max_event_wall, max_event_cpu)
    commit_stats["exclusive_wall_total_ns"] = wall_by_events
    commit_stats["exclusive_thread_cpu_total_ns"] = cpu_by_events
    graph_wall = nested_wall + 54 * 1_000_000
    graph_cpu = nested_cpu + 54 * 500_000
    graph_stats = _stats(54, nested_count, graph_wall, graph_cpu, nested_wall,
                         nested_cpu, (graph_wall + 53) // 54, (graph_cpu + 53) // 54)
    phases = []
    names = ["unassigned", "self_prepare", "self_attention", "self_projection_residual",
             "cross_attention_projection_residual", "ffn"]
    for phase_id, name in enumerate(names):
        graph = graph_stats if phase_id == 5 else _stats()
        commit = commit_stats if phase_id == 5 else _stats()
        phases.append({"phase_id": phase_id, "phase": name,
                       "operations": {"graph_encode": graph,
                                      "commit_and_continue": commit,
                                      "t5_linear_encode": _stats()}})

    op_counts = [18, 18, 18, 18, 18, 18, 2]
    timed = [sample["gpu_root_union_seconds"] for sample in samples[2:]]
    sorted_timed = sorted(timed)
    median = (sorted_timed[7] + sorted_timed[8]) / 2
    deviations = sorted(abs(value - median) for value in timed)
    mad = (deviations[7] + deviations[8]) / 2
    first4 = statistics.median(timed[:4])
    last4 = statistics.median(timed[-4:])
    summary = {
        "schema": "cogni_ml.pixelumm_ffn_aa_baseline/v1",
        "scope": "J5760_connected_baseline_aa",
        "sample_count": 16,
        "sample_seconds": timed,
        "median_seconds": median,
        "min_seconds": min(timed),
        "max_seconds": max(timed),
        "p90_seconds": sorted_timed[13] + (sorted_timed[14] - sorted_timed[13]) * 0.5,
        "mad_seconds": mad,
        "mad_over_median": mad / median,
        "first4_median_seconds": first4,
        "last4_median_seconds": last4,
        "last4_over_first4_median": last4 / first4,
        "adjacent_pair_later_over_earlier": [timed[i + 1] / timed[i] for i in range(0, 16, 2)],
    }
    metrics = []
    for sample_index in range(18):
        metrics.extend([
            {"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 5760 * 12288},
            {"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 5760 * 4096},
            {"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 5760 * 4096},
        ])
    metrics.extend([
        {"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 5760 * 12288},
        {"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 5760 * 12288},
        {"failures": 1, "different_words": 0, "nonfinite": 1, "elements": 5760 * 4096},
    ])

    trace = {
        "info": {"owner_thread_id": 1, "mach_timebase_numer": 1, "mach_timebase_denom": 1,
                 "trace_start_mach_ticks": origin, "event_capacity": 8192,
                 "event_count": 161, "root_count": 162, "failed": 0},
        "events": events,
        "roots": roots,
    }
    resources = dict(ANALYZER.EXPECTED_NAMED_BYTES)
    return {
        "schema": "cogni_ml.pixelumm_ffn_aa_probe/v1",
        "scope": "native connected FFN A/A under fixed GPU-comparator cadence; no full generation",
        "geometry": {"tokens": 5760, "hidden": 4096, "ffn": 12288, "grid": [10, 18, 32]},
        "fixture": {"path": "/fixture/j5760", "metadata_sha256": _digest("a"),
                    "payload_sha256": _digest("b"), "weight_cache_metadata_sha256": _digest("c"),
                    "weight_cache_payload_sha256": _digest("d"), "input_window": "w0", "block": 0,
                    "normalized_input": _digest("e"), "input_provenance": "saved_initial_noise_conditioning_upstream_stem",
                    "model_inference": False, "torch_version": "synthetic", "upstream_revision": "synthetic",
                    "upstream_model_sha256": _digest("f")},
        "comparison": {"atol": 2e-5, "rtol": 2e-5},
        "protocol": {"warmups": 2, "timed_samples": 16, "identical_plan_and_buffers": True,
                     "timed_path": "separate gate/up F32-compute MPSGraph -> BF16 SiLU -> BF16 product -> F32-compute down MPSGraph",
                     "correctness_capture_copies": "raw gate/up in sample 0 warmup only",
                     "normal_checks_per_sample": 3, "raw_gate_up_checks_after_loop": 2,
                     "positive_nan_controls": 1, "cpu_diagnostics_between_samples": False,
                     "cpu_readback_between_samples": False, "gpu_checks_between_samples": True,
                     "no_candidate_or_promotion": True,
                     "continuation_trace_enabled": trace_enabled,
                     "memory_accounting": "external cache-aware guard; file-backed pages counted once; reserve 10 percent"},
        "execution": {"named_bytes": sum(resources.values()), "resources": resources,
                      "encoding_named_allocation_delta": 0, "encoding_readbacks": 0,
                      "total_readbacks": 1, "wait_count": 1, "encode_loop_wall_seconds": 0.2,
                      "terminal_wait_wall_seconds": 0.1, "total_root_count": 162,
                      "operation_counts": op_counts},
        "samples": samples,
        "summary": summary,
        "numerical_metrics": metrics,
        "numerical_failure": None,
        "root_timings": root_timing,
        "api_profile": {"phase_count": 6, "phases": phases},
        "continuation_trace": trace if trace_enabled else None,
        "source_sha256": {"source.cr": _digest("1"), "bridge.mm": _digest("2")},
        "numerical_pass": True,
        "timing_isolation_pass": True,
        "pass": True,
    }


def _four_reports() -> list[dict[str, Any]]:
    return [_build_report(enabled, index) for index, enabled in enumerate((False, True, True, False))]


class PixelUMMFFNTraceAnalysisTest(unittest.TestCase):
    def test_falsifier_smoke_requires_four_arm_contract(self) -> None:
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports([_build_report(False, 0)])

    def test_valid_four_arm_sweep_classifies_every_event(self) -> None:
        result = ANALYZER.analyze_reports(_four_reports())
        self.assertEqual([row["continuation_trace_enabled"] for row in result["arms"]],
                         [False, True, True, False])
        trace_scope = result["native_trace_scope"]
        self.assertEqual(trace_scope["event_count_across_traced_arms"], 322)
        for arm in trace_scope["per_traced_arm"]:
            samples = arm["sample_commit_totals"]
            self.assertEqual(sum(row["internal_commit_count"] for row in samples), 125)
            self.assertEqual(sum(row["explicit_boundary_commits_excluded_from_internal_totals"]["count"]
                                 for row in samples), 36)

    def test_shared_overlap_summary_excludes_current_root_and_unions_prior_roots(self) -> None:
        def summarized(begin: int, end: int, before: int,
                       roots: list[tuple[float, float]]) -> dict[str, Any]:
            trace = {
                "events": [{"serial": 1, "begin_mach_ticks": begin, "end_mach_ticks": end,
                            "before_root_sequence": before, "phase": 5, "graph_depth": 1,
                            "thread_cpu_seconds": 1e-9}],
                "roots": [{"start_seconds": start, "end_seconds": finish,
                           "completion_observed_mach_ticks": 0} for start, finish in roots],
                "host_seconds": lambda ticks: ticks / 1e9,
            }
            return ANALYZER.ORDER._continuation_summary([{"trace": trace, "index": 0,
                                                          "combined_attention": False}])[
                "largest_earlier_root_overlaps"][0]

        # Event [10,20] overlaps the current root #2, but roots #0 and #1 end
        # before it. The current root must contribute exactly zero.
        self_only = summarized(10, 20, 2, [(1e-9, 5e-9), (6e-9, 9e-9),
                                           (10e-9, 20e-9)])
        self.assertEqual(self_only["earlier_root_overlap_seconds"], 0.0)
        self.assertEqual(self_only["earlier_roots_overlapping"], 0)

        # Earlier roots #0 and #1 overlap each other and jointly cover [12,18].
        # Their separate intersections sum to 8ns; union length is 6ns.
        union = summarized(12, 18, 2, [(10e-9, 16e-9), (14e-9, 20e-9),
                                       (12e-9, 18e-9)])
        self.assertAlmostEqual(union["earlier_root_overlap_seconds"], 6e-9, places=18)
        self.assertEqual(union["earlier_roots_overlapping"], 2)

    def test_rejects_wrong_counts_missing_trace_and_failed_lifecycle(self) -> None:
        cases = []
        wrong_count = _four_reports()
        wrong_count[0]["execution"]["operation_counts"][0] = 17
        cases.append(wrong_count)
        missing_trace = _four_reports()
        missing_trace[1]["continuation_trace"] = None
        cases.append(missing_trace)
        wrong_wait = _four_reports()
        wrong_wait[3]["execution"]["wait_count"] = 2
        cases.append(wrong_wait)
        allocation = _four_reports()
        allocation[0]["execution"]["encoding_named_allocation_delta"] = 1
        cases.append(allocation)
        for index, reports in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    ANALYZER.analyze_reports(reports)

    def test_rejects_nonfinite_root_or_sample_timing(self) -> None:
        for location in ("root", "sample", "trace"):
            reports = _four_reports()
            if location == "root":
                reports[0]["root_timings"][10]["gpu_start_seconds"] = math.nan
            elif location == "sample":
                reports[0]["samples"][3]["encode_wall_seconds"] = math.inf
            else:
                reports[1]["continuation_trace"]["roots"][10]["gpu_end_seconds"] = math.nan
            with self.subTest(location=location), self.assertRaises(ValueError):
                ANALYZER.analyze_reports(reports)

    def test_rejects_root_error_and_stale_isolation_claim(self) -> None:
        failed_root = _four_reports()
        failed_root[2]["root_timings"][8]["has_error"] = True
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(failed_root)
        stale_isolation = _four_reports()
        stale_isolation[0]["samples"][2]["isolated_root_timing"] = False
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(stale_isolation)
        overlapping_outside_root = _four_reports()
        sample_root = overlapping_outside_root[0]["samples"][2]["first_root"]
        overlap_at = overlapping_outside_root[0]["root_timings"][sample_root]["gpu_start_seconds"]
        overlapping_outside_root[0]["root_timings"][0]["gpu_start_seconds"] = overlap_at - 0.001
        overlapping_outside_root[0]["root_timings"][0]["gpu_end_seconds"] = overlap_at + 0.001
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(overlapping_outside_root)
        trace_error = _four_reports()
        trace_error[1]["continuation_trace"]["roots"][8]["has_error"] = 1
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(trace_error)

    def test_rejects_arm_order_and_source_or_fixture_drift(self) -> None:
        wrong_order = _four_reports()
        wrong_order[0]["protocol"]["continuation_trace_enabled"] = True
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(wrong_order)
        source_drift = _four_reports()
        source_drift[2]["source_sha256"]["source.cr"] = _digest("9")
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(source_drift)
        fixture_drift = _four_reports()
        fixture_drift[3]["fixture"]["payload_sha256"] = _digest("9")
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(fixture_drift)

    def test_rejects_numerical_control_and_api_count_corruption(self) -> None:
        bad_nan = _four_reports()
        bad_nan[0]["numerical_metrics"][56]["failures"] = 0
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(bad_nan)
        bad_api = _four_reports()
        bad_api[2]["api_profile"]["phases"][5]["operations"]["commit_and_continue"]["count"] = 160
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports(bad_api)

    def test_rejects_raw_trace_wall_exceeding_api_aggregate(self) -> None:
        reports = _four_reports()
        commit = reports[1]["api_profile"]["phases"][5]["operations"]["commit_and_continue"]
        # Keep the aggregate internally valid, but make it too small to cover
        # the 161 one-millisecond raw trace spans (including allowed rounding).
        commit["wall_total_ns"] = 150_000_000
        commit["exclusive_wall_total_ns"] = 150_000_000
        with self.assertRaisesRegex(ValueError, "raw continuation wall/thread CPU exceeds API aggregate"):
            ANALYZER.analyze_reports(reports)

    def test_rejects_trace_nesting_outside_fixed_sample_boundaries(self) -> None:
        reports = _four_reports()
        events = reports[1]["continuation_trace"]["events"]
        # Preserve the total nested count/API count while moving one nested
        # call onto the first sample's start boundary and one off its interior.
        events[0]["graph_depth"] = 1
        events[0]["parent_graph_serial"] = events[0]["api_scope_serial"] - 1
        events[0]["flags"] = 15
        events[1]["graph_depth"] = 0
        events[1]["parent_graph_serial"] = 0
        events[1]["flags"] = 14
        with self.assertRaisesRegex(ValueError, "internal continuation or async-boundary nesting changed"):
            ANALYZER.analyze_reports(reports)

    def test_rejects_sample_ranges_that_only_preserve_cardinality(self) -> None:
        reports = _four_reports()
        reports[1]["samples"][2]["first_root"] += 1
        reports[1]["samples"][2]["end_root"] += 1
        with self.assertRaisesRegex(ValueError, "sample 2 schedule/root range changed"):
            ANALYZER.analyze_reports(reports)


if __name__ == "__main__":
    unittest.main()
