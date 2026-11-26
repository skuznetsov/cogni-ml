"""CPU-only validation tests for the PixelUMM drift report analyzer."""
from __future__ import annotations
import copy
import math
import statistics
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "support"))
import analyze_pixelumm_drift_probe as analyzer


def bits(value: float) -> int:
    return struct.unpack("=Q", struct.pack("=d", value))[0]


def metric(elements: int = 0, failures: int = 0, different_words: int = 0, nonfinite: int = 0) -> dict[str, int]:
    return {"failures": failures, "different_words": different_words, "nonfinite": nonfinite, "elements": elements}


def synthetic_profile() -> dict:
    root_count = 53
    trace_start = 1_000_000_000
    roots = []
    profile_roots = []
    for i in range(root_count):
        tracked = trace_start + i * 20_000_000
        start = tracked / 1e9 + 0.0015
        end = tracked / 1e9 + 0.0265
        completion = tracked + 40_000_000
        roots.append({"sequence": i, "tracked_or_observed_mach_ticks": tracked,
                      "completion_observed_mach_ticks": completion,
                      "gpu_start_time_bits": bits(start), "gpu_end_time_bits": bits(end),
                      "status_at_track_or_enable": 0, "final_status": 4, "has_error": 0,
                      "completion_source": 1, "gpu_start_seconds": start, "gpu_end_seconds": end})
        profile_roots.append({"sequence": i, "start_seconds": start, "end_seconds": end, "candidates": []})

    phase_ids = [1] * 12 + [2] * 8 + [3] * 4 + [5] * 28
    events = []
    for i, phase in enumerate(phase_ids):
        tracked = trace_start + i * 20_000_000
        events.append({"serial": i + 1, "begin_mach_ticks": tracked + 1_000_000,
                       "end_mach_ticks": tracked + 2_000_000,
                       "begin_thread_cpu_ns": 1_000_000 + i * 10_000,
                       "end_thread_cpu_ns": 1_000_500 + i * 10_000,
                       "api_scope_serial": i + 2, "parent_graph_serial": i + 1,
                       "before_root_sequence": i, "after_root_sequence": i + 1,
                       "phase": phase, "graph_depth": 1, "before_root_status": 0,
                       "before_root_status_after_call": 2, "after_root_status": 0, "flags": 15})

    markers_by_plan = []
    cursor = 1
    expected_candidates = [[] for _ in range(root_count)]
    wall_values = {}
    cpu_values = {}
    for pi, plan in enumerate(analyzer.PLANS):
        marks = []
        start_cursor = cursor
        for j, name in enumerate(analyzer.MARKERS):
            root_cursor = start_cursor + round(j * 13 / 10)
            marks.append({"name": name, "elapsed_seconds": j * 0.1,
                          "thread_cpu_seconds": j * 0.05, "root_count": root_cursor})
        for a, b in zip(marks, marks[1:]):
            label = f"{plan}/{a['name']}"
            wall_values[label] = b["elapsed_seconds"] - a["elapsed_seconds"]
            cpu_values[label] = b["thread_cpu_seconds"] - a["thread_cpu_seconds"]
            for ri in range(a["root_count"] - 1, b["root_count"]):
                expected_candidates[ri].append(label)
        cursor = marks[-1]["root_count"]
        markers_by_plan.append({"plan": plan, "markers": marks})
    expected_candidates[0].append("request_prelude")
    for ri in range(markers_by_plan[-1]["markers"][-1]["root_count"] - 1, root_count):
        expected_candidates[ri].append("request_final_comparators")
    candidate_sums = dict.fromkeys(analyzer.PHASE_LABELS, 0.0)
    definite_sums = dict.fromkeys(analyzer.PHASE_LABELS, 0.0)
    for i, root in enumerate(profile_roots):
        root["candidates"] = expected_candidates[i]
        duration = root["end_seconds"] - root["start_seconds"]
        phase_candidates = [label for label in root["candidates"] if label in candidate_sums]
        for label in phase_candidates:
            candidate_sums[label] += duration
        if len(phase_candidates) == 1:
            definite_sums[phase_candidates[0]] += duration

    graph_counts = {0: 0, 1: 36, 2: 12, 3: 12, 4: 0, 5: 12}
    commit_counts = {0: 0, 1: 12, 2: 8, 3: 4, 4: 0, 5: 28}
    api_phases = []
    for pid, name in analyzer.API_PHASES.items():
        count = graph_counts[pid]
        nested_count = commit_counts[pid]
        nested_wall = nested_count * 2_000_000
        nested_cpu = nested_count * 1_000
        graph_wall = count * 10_000_000
        graph_cpu = count * 10_000
        graph = {"count": count, "wall_total_ns": graph_wall, "wall_max_ns": 10_000_000 if count else 0,
                 "thread_cpu_total_ns": graph_cpu, "thread_cpu_max_ns": 10_000 if count else 0,
                 "nested_count": nested_count, "nested_wall_total_ns": nested_wall,
                 "nested_thread_cpu_total_ns": nested_cpu,
                 "exclusive_wall_total_ns": graph_wall - nested_wall,
                 "exclusive_thread_cpu_total_ns": graph_cpu - nested_cpu}
        commit = {"count": nested_count, "wall_total_ns": nested_wall, "wall_max_ns": 2_000_000 if nested_count else 0,
                  "thread_cpu_total_ns": nested_cpu, "thread_cpu_max_ns": 1_000 if nested_count else 0,
                  "nested_count": nested_count, "nested_wall_total_ns": nested_wall,
                  "nested_thread_cpu_total_ns": nested_cpu,
                  "exclusive_wall_total_ns": nested_wall, "exclusive_thread_cpu_total_ns": nested_cpu}
        empty = dict.fromkeys(analyzer.API_FIELDS, 0)
        api_phases.append({"source_phase_id": pid, "source_phase_name": name,
                           "operations": {"graph_encode": graph, "commit_and_continue": commit,
                                          "t5_linear_encode": empty}})

    intervals = [(r["start_seconds"], r["end_seconds"]) for r in profile_roots]
    union = analyzer.interval_union(intervals)
    root_sum = sum(b - a for a, b in intervals)
    envelope = max(b for _, b in intervals) - min(a for a, _ in intervals)
    trace = {"info": {"owner_thread_id": 42, "mach_timebase_numer": 1,
                       "mach_timebase_denom": 1, "trace_start_mach_ticks": trace_start,
                       "event_capacity": 8192, "event_count": len(events),
                       "root_count": len(roots), "failed": 0},
             "events": events, "roots": roots}
    return {"schema": analyzer.PROFILE,
            "markers_by_plan": markers_by_plan,
            "phase_wall_seconds": wall_values,
            "phase_thread_cpu_seconds": cpu_values,
            "phase_candidate_root_duration_sum_seconds": candidate_sums,
            "phase_definite_root_duration_sum_seconds": definite_sums,
            "root_candidate_duration_semantics": analyzer.PROFILE_SEMANTICS,
            "gpu_phase_interpretation": analyzer.GPU_INTERPRETATION,
            "roots": profile_roots,
            "root_profile": {"root_count": root_count, "root_duration_seconds": root_sum,
                "envelope_seconds": envelope, "union_seconds": union,
                "overlap_seconds": root_sum - union, "gap_seconds": max(0.0, envelope - union),
                "phase_union_seconds": {"whole_chain": union}},
            "api_profile": {"source_schema": "MPSAPIProfile WAN phase IDs", "phases": api_phases},
            "continuation_trace": trace}


def metric_slots(index: int, kind: str) -> list[dict[str, int]]:
    values = [metric() for _ in range(29)]
    for plan in range(4):
        for stage, name in enumerate(analyzer.CAPTURES):
            slot = 4 + plan * 6 + stage
            if kind == "correctness" or name == "output":
                size = 5760 * (12288 if name in ("gate_proj", "up_proj", "product") else 4096)
                # A nonzero word difference is retained and remains under tolerance.
                values[slot] = metric(size, different_words=(1 if index == 0 and stage == 0 else 0))
    if index == 0:
        values[28] = metric(5760 * 4096, failures=1, nonfinite=1)
    return values


def request_row(index: int, profile: dict, roots_duration: float) -> dict:
    if index < 2:
        kind, enabled = "correctness", bool(index)
    elif index < 6:
        kind, enabled = "warmup", (False, True, True, False)[index - 2]
    else:
        kind, enabled = "timed", analyzer.TIMED_PATTERN[(index - 6) % 8]
    base_wall = 100_000_000_000 + index * 5_000_000_000
    wait = 0.1 + index * 0.0005
    offsets = [0, 10_000_000, 20_000_000, 1_020_000_000,
               int((1.02 + wait) * 1e9), int((1.022 + wait) * 1e9), int((1.024 + wait) * 1e9)]
    cpu_offsets = [0, 1_000_000, 2_000_000, 52_000_000,
                   52_000_000 + int(wait * 0.3 * 1e9), 54_000_000 + int(wait * 0.3 * 1e9),
                   56_000_000 + int(wait * 0.3 * 1e9)]
    hosts = [{"wall_ns": base_wall + x, "thread_cpu_ns": 10_000_000_000 + index * 500_000_000 + y,
              "thermal_state": index % 4} for x, y in zip(offsets, cpu_offsets)]
    pairs = {"admission_poison": (1, 2), "encode": (2, 3), "terminal_wait": (3, 4),
             "profile_extraction": (4, 5), "metrics_release": (5, 6), "critical": (1, 4), "total": (0, 6)}
    phases = {name: {"wall_seconds": (hosts[b]["wall_ns"] - hosts[a]["wall_ns"]) / 1e9,
                     "thread_cpu_seconds": (hosts[b]["thread_cpu_ns"] - hosts[a]["thread_cpu_ns"]) / 1e9}
              for name, (a, b) in pairs.items()}
    raw_points = []
    for j in range(7):
        raw_points.append({"time_ns": 1_000_000_000 + index * 2_000_000_000 + j * 1_000_000,
            "allocated_bytes": 100_000 + index * 1_000 + j * 100, "gc_no": 1, "heap_bytes": 500_000,
            "free_bytes": 100_000, "bytes_since_gc": 0, "bytes_before_gc": 100_000,
            "collection_count": 0, "collection_total_ns": 0, "collection_max_ns": 0,
            "rendezvous_pause_count": 0, "rendezvous_pause_total_ns": 0, "rendezvous_pause_max_ns": 0,
            "stopped_interval_count": 0, "stopped_interval_total_ns": 0, "stopped_interval_max_ns": 0,
            "unmatched_event_count": 0})
    gc = {"seconds": 0.006, "allocated_bytes": 600, "collections": 0, "gc_no_delta": 0,
          "collection_seconds": 0.0, "rendezvous_pauses": 0, "rendezvous_pause_seconds": 0.0,
          "stopped_intervals": 0, "stopped_seconds": 0.0}
    packed = metric_slots(index, kind)
    return {"index": index, "profile_enabled": enabled, "kind": kind,
        "body_encode_seconds": 0.2 + index * 0.0001, "comparator_encode_seconds": 0.1,
        "gpu_elapsed_seconds": roots_duration, "root_count": len(profile["roots"]), "phases": phases, "gc": gc,
        "raw_points": raw_points, "host_points": hosts, "named_buffer_allocations": 0,
        "terminal_wait_count": 1, "preterminal_readbacks": 0, "comparator_readbacks_after_wait": 1,
        "comparator_readback_bytes": 464, "numerical_metrics": copy.deepcopy(packed),
        "capture_metrics": copy.deepcopy(packed) if kind == "correctness" else None,
        "profile": copy.deepcopy(profile) if enabled else None}


def valid_report() -> dict:
    profile = synthetic_profile()
    roots_duration = profile["root_profile"]["root_duration_seconds"]
    reqs = [request_row(i, profile, roots_duration) for i in range(102)]
    return {"schema": analyzer.REPORT,
        "comparison": {"atol": 2e-5, "rtol": 2e-5, "capture_order": list(analyzer.CAPTURES)},
        "validation": {"all_numerical_metrics_passed": True, "positive_nan_control_passed": True,
            "one_terminal_wait_and_packed_readback_per_request": True,
            "zero_named_encode_allocations_and_preterminal_readbacks": True,
            "full_generation_executed": False, "production_promoted": False},
        "schedule": {"correctness_requests": 2, "warmup_requests": 4, "timed_requests": 96,
                     "native_requests": 102, "requests": reqs},
        "setup_seconds": 12.5, "observed_loop": {"seconds": 200.0, "allocated_bytes": 100_000,
            "collections": 0, "gc_no_delta": 0, "collection_seconds": 0.0,
            "rendezvous_pauses": 0, "rendezvous_pause_seconds": 0.0,
            "stopped_intervals": 0, "stopped_seconds": 0.0}}


def valid_guard() -> list[dict]:
    return [
        {"event": "limits", "monotonic": 1.0, "memory_accounting": "file-cache",
         "headroom_formula": "printed free + file-backed (includes speculative)",
         "runtime_floor_pct": 10, "runtime_floor_pct_exact": "10",
         "max_footprint_bytes": 16 * 1024**3, "max_swap_growth_bytes": 64 * 1024**2,
         "timeout_sec": 600,
         "startup_rule": "memory_pressure -Q >= 10% and selected headroom >= 10% physical + footprint budget",
         "limitation": "sampled only", "file_cache_limitation": "reclaimability estimate"},
        {"event": "sample", "monotonic": 1.1, "physical_bytes": 100 * 1024**3,
         "headroom_bytes": 90 * 1024**3, "pressure_free_pct": 90, "free_spec_bytes": 50 * 1024**3,
         "file_backed_bytes": 50 * 1024**3, "speculative_bytes": 10 * 1024**3,
         "group_footprint_bytes": 0, "swap_used_bytes": 100, "group_live_processes": 0},
        {"event": "started", "monotonic": 1.2, "pid": 7, "pgid": 7},
        {"event": "sample", "monotonic": 1.3, "physical_bytes": 100 * 1024**3,
         "headroom_bytes": 90 * 1024**3, "pressure_free_pct": 90, "free_spec_bytes": 50 * 1024**3,
         "file_backed_bytes": 50 * 1024**3, "speculative_bytes": 10 * 1024**3,
         "group_footprint_bytes": 20_000, "swap_used_bytes": 100, "group_live_processes": 1},
        {"event": "finished", "monotonic": 1.4, "returncode": 0},
    ]


class DriftAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = valid_report()

    def test_summarizes_schedule_pairs_profiles_and_backlog_excluding_current_root(self) -> None:
        summary = analyzer.analyze(self.report, "a" * 64)
        self.assertEqual(summary["request_counts"]["profiled"], 51)
        self.assertEqual(summary["paired_adjacent_on_off"]["pair_count"], 48)
        self.assertEqual(summary["chronological_drift"]["bin_size_requests"], 8)
        self.assertEqual(len(summary["chronological_drift"]["bins"]), 12)
        self.assertEqual(summary["different_words_by_request"][0]["slots"][4], 1)
        continuations = summary["profiled_requests"]["continuations"]
        self.assertEqual(continuations["event_count"], 51 * 52)
        # Root zero starts during its own submit call. It is still excluded because no earlier root exists.
        first = continuations["events"][0]
        self.assertEqual(first["earlier_roots_overlapping"], 0)
        self.assertEqual(first["earlier_root_overlap_seconds"], 0)
        first_trace_event = self.report["schedule"]["requests"][1]["profile"]["continuation_trace"]["events"][0]
        first_root = self.report["schedule"]["requests"][1]["profile"]["roots"][0]
        call_start = first_trace_event["begin_mach_ticks"] / 1e9
        call_end = first_trace_event["end_mach_ticks"] / 1e9
        self.assertGreater(min(call_end, first_root["end_seconds"]) -
                           max(call_start, first_root["start_seconds"]), 0)
        self.assertGreater(continuations["events"][1]["earlier_root_overlap_seconds"], 0)
        phase = continuations["by_phase"]["self_prepare"]
        self.assertAlmostEqual(phase["total_wall_seconds"], sum(
            e["wall_seconds"] for e in continuations["events"] if e["source_phase"] == "self_prepare"))
        self.assertAlmostEqual(phase["total_thread_cpu_seconds"], sum(
            e["thread_cpu_seconds"] for e in continuations["events"] if e["source_phase"] == "self_prepare"))
        self.assertAlmostEqual(phase["wall_weighted_earlier_root_overlap_fraction"],
                               phase["total_earlier_root_overlap_seconds"] / phase["total_wall_seconds"])
        self.assertIn("exclusive totals", summary["profiled_requests"]["api_semantics"])
        self.assertIn("not additive GPU work", summary["profiled_requests"]["root_sum_semantics"])
        self.assertEqual(summary["arms"]["profile_on"]["host_phase_wall_and_cpu"]["encode"]["wall_seconds"]["count"], 48)
        self.assertIn("gpu_root_duration_sum_seconds", summary["arms"]["profile_on"])
        self.assertNotIn("gpu_chain_span_seconds", summary["arms"]["profile_on"])
        limits = " ".join(summary["interpretation_limits"])
        self.assertIn("root-duration sum", limits)
        self.assertIn("Interval union", limits)
        self.assertIn("envelope", limits)

    def test_wall_weighted_backlog_fraction_is_ratio_of_phase_totals(self) -> None:
        rows = [
            {"wall_seconds": 0.001, "thread_cpu_seconds": 0.00001,
             "earlier_root_overlap_seconds": 0.0009},
            {"wall_seconds": 1.0, "thread_cpu_seconds": 0.00002,
             "earlier_root_overlap_seconds": 0.1},
        ]
        result = analyzer.continuation_totals(rows)
        self.assertAlmostEqual(result["total_wall_seconds"], 1.001)
        self.assertAlmostEqual(result["total_thread_cpu_seconds"], 0.00003)
        self.assertAlmostEqual(result["wall_weighted_earlier_root_overlap_fraction"], 0.1009 / 1.001)
        self.assertNotAlmostEqual(result["wall_weighted_earlier_root_overlap_fraction"],
                                  statistics.median([0.9, 0.1]))

    def test_rejects_schedule_mutation(self) -> None:
        report = copy.deepcopy(self.report)
        report["schedule"]["requests"][6]["profile_enabled"] = True
        with self.assertRaisesRegex(ValueError, "schedule changed"):
            analyzer.analyze(report)

    def test_rejects_root_interval_mutation(self) -> None:
        report = copy.deepcopy(self.report)
        report["schedule"]["requests"][1]["profile"]["roots"][0]["end_seconds"] += 0.01
        with self.assertRaises(ValueError):
            analyzer.analyze(report)

    def test_rejects_metric_or_positive_control_mutation(self) -> None:
        report = copy.deepcopy(self.report)
        report["schedule"]["requests"][0]["numerical_metrics"][28]["failures"] = 0
        report["schedule"]["requests"][0]["capture_metrics"][28]["failures"] = 0
        with self.assertRaisesRegex(ValueError, "positive NaN control"):
            analyzer.analyze(report)

    def test_rejects_nonfinite_and_incomplete_profile_metrics(self) -> None:
        report = copy.deepcopy(self.report)
        report["schedule"]["requests"][6]["body_encode_seconds"] = math.nan
        with self.assertRaisesRegex(ValueError, "request 6: invalid body_encode_seconds"):
            analyzer.analyze(report)
        report = copy.deepcopy(self.report)
        del report["schedule"]["requests"][7]["profile"]["phase_thread_cpu_seconds"][analyzer.PHASE_LABELS[0]]
        with self.assertRaisesRegex(ValueError, "incomplete phase_thread_cpu_seconds"):
            analyzer.analyze(report)

    def test_validates_guard_floor_caps_and_lifecycle(self) -> None:
        result = analyzer.validate_guard(valid_guard(), "b" * 64)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["peak_group_footprint_bytes"], 20_000)
        self.assertEqual(result["maximum_swap_growth_bytes"], 0)
        guard = valid_guard()
        guard[3]["swap_used_bytes"] += 64 * 1024**2 + 1
        with self.assertRaisesRegex(ValueError, "swap-growth cap"):
            analyzer.validate_guard(guard)

    def test_rejects_incomplete_or_inconsistent_guard(self) -> None:
        guard = valid_guard()
        guard[3]["headroom_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "headroom fields are inconsistent"):
            analyzer.validate_guard(guard)
        guard = valid_guard()
        guard.pop()
        with self.assertRaisesRegex(ValueError, "missing limits, start, samples, or finish"):
            analyzer.validate_guard(guard)


if __name__ == "__main__":
    unittest.main()
