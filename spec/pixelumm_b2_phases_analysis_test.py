"""CPU-only falsifier tests for the B2 full36 phase report analyzer."""
from __future__ import annotations

import copy
import struct
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "examples" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))
import analyze_pixelumm_b2_phases as analyzer  # noqa: E402
import analyze_pixelumm_order_profile as common  # noqa: E402


def _root_intervals(root_count: int) -> list[tuple[float, float]]:
    return [(100.0 + index * 0.01, 100.005 + index * 0.01) for index in range(root_count)]


def _root_rows(intervals: list[tuple[float, float]], *, profiled: bool = False,
               candidates: list[list[str]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for index, (start, end) in enumerate(intervals):
        row = {"sequence": index, "completed": True, "has_error": False,
               "gpu_start_seconds": start, "gpu_end_seconds": end}
        if profiled:
            row.update({"duration_seconds": end - start,
                        "candidate_phases": candidates[index] if candidates is not None else []})
        rows.append(row)
    return rows


def _trace(intervals: list[tuple[float, float]]) -> dict[str, Any]:
    roots = []
    for index, (start, end) in enumerate(intervals):
        tracked = round((start - 1e-6) * 1e9)
        completed = round((end + 1e-6) * 1e9)
        start_bits = struct.unpack("=Q", struct.pack("=d", start))[0]
        end_bits = struct.unpack("=Q", struct.pack("=d", end))[0]
        roots.append({"sequence": index, "tracked_or_observed_mach_ticks": tracked,
            "completion_observed_mach_ticks": completed, "gpu_start_time_bits": start_bits,
            "gpu_end_time_bits": end_bits, "status_at_track_or_enable": 0, "final_status": 4,
            "has_error": 0, "completion_source": 1, "gpu_start_seconds": start,
            "gpu_end_seconds": end})

    phase_ids = [1] * 120 + [2] * 180 + [3] * 60
    events = []
    previous_end = 99_000_000_000
    for index, phase_id in enumerate(phase_ids):
        tracked = roots[index]["tracked_or_observed_mach_ticks"]
        begin = tracked + 100
        end = begin + 500
        assert begin >= previous_end
        events.append({"serial": index + 1, "begin_mach_ticks": begin, "end_mach_ticks": end,
            "begin_thread_cpu_ns": 1_000_000_000 + index * 1_000,
            "end_thread_cpu_ns": 1_000_000_500 + index * 1_000,
            "api_scope_serial": index + 1, "parent_graph_serial": 0,
            "before_root_sequence": index, "after_root_sequence": index + 1,
            "phase": phase_id, "graph_depth": 0, "before_root_status": 0,
            "before_root_status_after_call": 2, "after_root_status": 0, "flags": 14})
        previous_end = end
    return {"info": {"owner_thread_id": 1, "mach_timebase_numer": 1,
        "mach_timebase_denom": 1, "trace_start_mach_ticks": 99_000_000_000,
        "event_capacity": 8192, "event_count": len(events), "root_count": len(roots), "failed": 0},
        "events": events, "roots": roots}


def _api_profile(candidate: bool) -> dict[str, Any]:
    event_counts = {0: 0, 1: 120, 2: 180, 3: 60, 4: 0, 5: 0}
    graph_counts = {0: 5, 1: 324, 2: 3240 if candidate else 6480,
                    3: 108, 4: 0, 5: 108}

    def stats(count: int) -> dict[str, int]:
        return {"count": count, "wall_total_ns": count * 2_000,
                "wall_max_ns": 2_000 if count else 0,
                "thread_cpu_total_ns": count * 1_000,
                "thread_cpu_max_ns": 1_000 if count else 0,
                "nested_count": 0, "nested_wall_total_ns": 0,
                "nested_thread_cpu_total_ns": 0,
                "exclusive_wall_total_ns": count * 2_000,
                "exclusive_thread_cpu_total_ns": count * 1_000}

    rows = []
    for phase_id in range(6):
        operations = {"graph_encode": stats(graph_counts[phase_id]),
                      "commit_and_continue": stats(event_counts[phase_id]),
                      "t5_linear_encode": stats(0)}
        rows.append({"pixelumm_phase": common.PIXELUMM_API_PHASES[phase_id],
                     "source_phase_id": phase_id, "source_phase_name": common.API_PHASES[phase_id],
                     "operations": operations})
    return {"source_schema": "MPSAPIProfile WAN phase IDs", "phases": rows}


def _phase_profile(candidate: bool, root_count: int = 361) -> dict[str, Any]:
    intervals = _root_intervals(root_count)
    blocks = []
    expected_candidates: list[list[str]] = [[] for _ in intervals]
    aggregate_wall = dict.fromkeys(common.PHASES, 0.0)
    aggregate_cpu = dict.fromkeys(common.PHASES, 0.0)
    for slot in range(36):
        start_cursor = 1 + slot * 10
        markers = []
        for marker_index, name in enumerate(common.MARKERS):
            markers.append({"name": name, "elapsed_seconds": marker_index * 0.01,
                "thread_cpu_seconds": marker_index * 0.008,
                "root_count": start_cursor + marker_index})
        phase_wall = {phase: 0.01 for phase in common.PHASES}
        phase_cpu = {phase: 0.008 for phase in common.PHASES}
        for phase_index, phase in enumerate(common.PHASES):
            aggregate_wall[phase] += phase_wall[phase]
            aggregate_cpu[phase] += phase_cpu[phase]
            label = f"block_{slot}/{phase}"
            for root_index in range(start_cursor + phase_index - 1,
                                    start_cursor + phase_index + 1):
                expected_candidates[root_index].append(label)
        blocks.append({"slot": slot, "attention_tile_count": 60, "markers": markers,
                       "phase_wall_seconds": phase_wall,
                       "phase_thread_cpu_seconds": phase_cpu})
    for index in range(360, root_count):
        expected_candidates[index].append("head")
    duration = intervals[0][1] - intervals[0][0]
    candidate_sums = {label: 0.0 for slot in range(36) for label in
                      [f"block_{slot}/{phase}" for phase in common.PHASES]}
    candidate_sums["head"] = 0.0
    definite_sums = dict.fromkeys(candidate_sums, 0.0)
    for index, candidates in enumerate(expected_candidates):
        for label in candidates:
            candidate_sums[label] += duration
        if len(candidates) == 1:
            definite_sums[candidates[0]] += duration
    metrics = common._root_metrics(intervals)
    profile = {
        "schema": analyzer.PROFILE, "block_count": 36,
        "ffn_graph_topology": "separate_gate_up", "fused_kv_gather": False,
        "scale_softmax": False, "blocks": blocks,
        "aggregate_phase_wall_seconds": aggregate_wall,
        "aggregate_phase_thread_cpu_seconds": aggregate_cpu,
        "head_root_range": {"start_root_count": 361, "end_root_count": root_count,
            "candidate_root_indices_inclusive": [360, root_count - 1]},
        "root_candidates": expected_candidates,
        "phase_candidate_root_duration_sum_seconds": candidate_sums,
        "phase_definite_root_duration_sum_seconds": definite_sums,
        "phase_root_duration_sum_semantics": analyzer.ROOT_SEMANTICS,
        "root_profile": {"root_count": root_count,
            "root_duration_seconds": metrics["root_duration_seconds"],
            "envelope_seconds": metrics["envelope_seconds"], "union_seconds": metrics["union_seconds"],
            "overlap_seconds": metrics["overlap_seconds"],
            "phase_union_seconds": {"whole_chain": metrics["union_seconds"]}},
        "roots": _root_rows(intervals, profiled=True, candidates=expected_candidates),
        "api_profile": _api_profile(candidate),
    }
    if candidate:
        profile["attention_graph_topology"] = "paired_windows_b2"
    return profile


def _contract_fixture() -> dict[str, Any]:
    reverse = False
    rows = analyzer._expected_rows(reverse)
    return {
        "schema": analyzer.REPORT, "profile_phases": True, "generation": False,
        "production_promoted": False, "full_model_qualified": False, "passed": True,
        "cleanup_quarantined": False, "atol": 2e-5, "rtol": 2e-5,
        "block_count": 36, "includes_head": True, "grid": [6, 44, 80],
        "expected_request_count": 15, "timed": True, "reverse": reverse,
        **{key: char * 64 for key, char in zip(analyzer.FIXTURE_PIN_FIELDS, "abcde")},
        "input_sha256": "f" * 64, "alternate_input_sha256": "0" * 64,
        "capture_order": common._expected_capture_order(),
        "capture_inventory": {"block_attention": 36, "retained_block_outputs": 2,
                              "head_stages": 5, "total": 43},
        "profiling_contract": {
            "phase_profile_extraction": "post_terminal_wait",
            "root_profile_extraction": "post_terminal_wait",
            "continuation_trace_extraction": "post_terminal_wait",
            "early_phase_read_rejected_on_first_profiled_gate": True,
            "cpu_tensor_diagnostics_between_timed_requests": False,
            "only_last_measured_request_has_capture_readback": True,
            "one_existing_terminal_wait_per_request": True,
            "named_buffer_allocations_during_encode": 0,
            "activation_readbacks_before_terminal_wait": 0,
            "stage_metric_readbacks": 0, "full36_tiles_per_block": 60,
            "baseline_attention_graph_calls": 6480, "candidate_attention_graph_calls": 3240,
            "baseline_total_graph_calls": 7025, "candidate_total_graph_calls": 3785,
            "baseline_command_segments": 4579, "candidate_command_segments": 2779,
        },
        "observer_schedule": {
            "poisoned_gate_requests": [0, 1, 2, 3], "poisoned_gates_profiled": True,
            "warmup_requests": [4, 5], "warmups_profiled": False,
            "measured_requests": list(range(6, 14)),
            "measured_candidate_modes": [row["candidate"] for row in rows[6:14]],
            "measured_profiled": [row["profiled"] for row in rows[6:14]],
            "measured_timing_eligible": [row["timing_eligible"] for row in rows[6:14]],
            "tail_request": 14, "tail_candidate": True, "tail_profiled": True,
            "reverse_flips_modes_only": True,
        },
    }


class B2PhaseAnalyzerTests(unittest.TestCase):
    def test_current_chunk_inventory_is_geometry_checked_in_both_attention_modes(self) -> None:
        for selected, original in ((False, self.baseline), (True, self.candidate)):
            profile = copy.deepcopy(original)
            profile["token_count"] = 21120
            with self.assertRaisesRegex(ValueError, "Graph phase counts"):
                analyzer._validate_phase_profile(profile, 0, selected, self.intervals, self.trace)
            graph = profile["api_profile"]["phases"][1]["operations"]["graph_encode"]
            graph.update(count=1224, wall_total_ns=1224 * 2000,
                         thread_cpu_total_ns=1224 * 1000,
                         exclusive_wall_total_ns=1224 * 2000,
                         exclusive_thread_cpu_total_ns=1224 * 1000)
            parsed = analyzer._validate_phase_profile(profile, 0, selected, self.intervals, self.trace)
            self.assertEqual(sum(row["graph_encode"]["count"] for row in parsed["api"].values()),
                             4685 if selected else 7925)

    @classmethod
    def setUpClass(cls) -> None:
        cls.intervals = _root_intervals(361)
        cls.trace_raw = _trace(cls.intervals)
        cls.trace = common.validate_trace(cls.trace_raw, 0, cls.intervals)
        cls.baseline = _phase_profile(False)
        cls.candidate = _phase_profile(True)

    def test_baseline_and_b2_topologies_validate_with_expected_graph_counts(self) -> None:
        baseline = analyzer._validate_phase_profile(self.baseline, 0, False, self.intervals, self.trace)
        candidate = analyzer._validate_phase_profile(self.candidate, 0, True, self.intervals, self.trace)
        self.assertEqual(baseline["api"][2]["graph_encode"]["count"], 6480)
        self.assertEqual(candidate["api"][2]["graph_encode"]["count"], 3240)
        self.assertEqual(sum(phase["graph_encode"]["count"] for phase in baseline["api"].values()), 7025)
        self.assertEqual(sum(phase["graph_encode"]["count"] for phase in candidate["api"].values()), 3785)
        self.assertEqual(len(candidate["per_block"]), 36)

    def test_b2_topology_tag_is_required_and_baseline_json_stays_legacy_shape(self) -> None:
        candidate_missing_tag = copy.deepcopy(self.candidate)
        del candidate_missing_tag["attention_graph_topology"]
        with self.assertRaisesRegex(ValueError, "missing or unexpected fields"):
            analyzer._validate_phase_profile(candidate_missing_tag, 0, True, self.intervals, self.trace)
        baseline_with_tag = copy.deepcopy(self.baseline)
        baseline_with_tag["attention_graph_topology"] = "independent_windows"
        with self.assertRaisesRegex(ValueError, "missing or unexpected fields"):
            analyzer._validate_phase_profile(baseline_with_tag, 0, False, self.intervals, self.trace)

    def test_root_inventory_and_phase_candidate_tampering_are_rejected(self) -> None:
        fewer_roots = copy.deepcopy(self.candidate)
        fewer_roots["roots"].pop()
        with self.assertRaisesRegex(ValueError, "profile/root count mismatch"):
            analyzer._validate_phase_profile(fewer_roots, 0, True, self.intervals, self.trace)
        wrong_candidates = copy.deepcopy(self.baseline)
        wrong_candidates["root_candidates"][0] = ["head"]
        with self.assertRaisesRegex(ValueError, "candidate table disagrees"):
            analyzer._validate_phase_profile(wrong_candidates, 0, False, self.intervals, self.trace)

    def test_graph_count_and_trace_root_count_tampering_are_rejected(self) -> None:
        wrong_graphs = copy.deepcopy(self.candidate)
        wrong_graphs["api_profile"]["phases"][2]["operations"]["graph_encode"]["count"] = 3239
        with self.assertRaisesRegex(ValueError, "inconsistent graph_encode API totals|Graph phase counts"):
            analyzer._validate_phase_profile(wrong_graphs, 0, True, self.intervals, self.trace)
        wrong_trace = copy.deepcopy(self.trace_raw)
        wrong_trace["events"].pop()
        with self.assertRaisesRegex(ValueError, "continuation event/root counts"):
            common.validate_trace(wrong_trace, 0, self.intervals)

    def test_tolerance_and_contract_metadata_are_fail_closed(self) -> None:
        report = _contract_fixture()
        analyzer.validate_contract_metadata(report)
        report["rtol"] = 3e-5
        with self.assertRaisesRegex(ValueError, "tolerance changed"):
            analyzer.validate_contract_metadata(report)

    def test_continuation_localization_excludes_the_current_submission_root(self) -> None:
        detail = analyzer._continuation_prior_root_detail([{
            "index": 0, "cohort": "profiled_measured", "candidate": True, "trace": self.trace,
        }])
        first = detail["per_call"][0]
        second = detail["per_call"][1]
        self.assertEqual(first["before_root_sequence"], 0)
        self.assertEqual(first["prior_submitted_root_count"], 0)
        self.assertEqual(first["prior_root_interval_overlap_union_seconds"], 0.0)
        self.assertEqual(second["prior_submitted_root_count"], 1)
        self.assertEqual(detail["per_phase"]["self_prepare"]["continuation_count"], 120)
        self.assertAlmostEqual(detail["per_phase"]["self_prepare"]["per_call_wall_seconds"]["p50"], 5e-7)
        self.assertIn("not exact completion", detail["interpretation"])

    def test_profile_summary_composes_continuation_rows_for_candidate_detail(self) -> None:
        profile = analyzer._validate_phase_profile(
            self.candidate, 0, True, self.intervals, self.trace)
        summary = analyzer._profile_summary({"requests": [{
            "index": 0, "cohort": "profiled_measured", "candidate": True,
            "profiled": True, "timing": {}, "profile": profile, "trace": self.trace,
        }]})
        continuation = summary["continuation_per_call_quantiles_and_prior_root_overlap"]
        self.assertEqual(len(continuation["per_call"]), 360)
        self.assertTrue(all(row["candidate"] for row in continuation["per_call"]))
        self.assertEqual(continuation["per_phase"]["self_attention"]["continuation_count"], 180)

    def test_reverse_mode_schedule_and_profile_contract_are_pinned(self) -> None:
        report = _contract_fixture()
        report["reverse"] = True
        rows = analyzer._expected_rows(True)
        report["observer_schedule"]["measured_candidate_modes"] = [row["candidate"] for row in rows[6:14]]
        analyzer.validate_contract_metadata(report)
        report["observer_schedule"]["measured_timing_eligible"][2] = True
        with self.assertRaisesRegex(ValueError, "observer schedule metadata"):
            analyzer.validate_contract_metadata(report)


if __name__ == "__main__":
    unittest.main()
