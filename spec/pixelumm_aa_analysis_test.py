"""CPU-only controls for the PixelUMM fixed-baseline A/A profile analyzer."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER = load_module("pixelumm_aa_analysis", ROOT / "examples/support/analyze_pixelumm_aa_profile.py")
PREVIOUS = load_module("pixelumm_previous_order_test_fixtures", ROOT / "spec/pixelumm_order_analysis_test.py")
ORDER = ANALYZER.ORDER


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def memory_events_aa(layout: list[dict[str, int]]) -> list[dict[str, object]]:
    order = ORDER._expected_capture_order()
    events: list[dict[str, object]] = []
    for request, times in enumerate(layout):
        rows: list[tuple[str, int]] = [
            ("request_before_encode", times["before_encode"] + 1_000_000),
            ("request_after_terminal_wait", times["terminal_end"] + 1_000_000),
            ("request_after_profile_extraction", times["terminal_end"] + 2_000_000),
        ]
        tick = times["terminal_end"] + 3_000_000
        for stage in order:
            head = stage.startswith("head.")
            prefix = "integrated_head_" if head else ""
            tick += 250_000_000
            rows.append((f"before_{prefix}compare_buffers", tick))
            tick += 250_000_000
            rows.append((f"{prefix}compare_buffers_ready", tick))
            tick += 250_000_000
            rows.append((f"{prefix}compare_stage_done", tick))
        rows.extend((("request_after_diagnostic_comparisons", tick + 1_000_000),
                     ("after_chain_release", tick + 2_000_000),
                     ("before_post_diagnostic_gc", tick + 3_000_000),
                     ("after_post_diagnostic_gc", times["after_gc"])))
        events.extend(({"schema": "cogni_ml.pixelumm_memory_phase/v1", "sequence": len(events) + 1,
                        "timestamp_clock": "CLOCK_MONOTONIC", "monotonic_ns": timestamp,
                        "guard_timestamp_clock": "mach_absolute_time", "guard_monotonic_ns": timestamp,
                        "phase": phase, "request": request} for phase, timestamp in rows))
    return events


def make_spacing_and_memory(reverse: bool) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    expected = ANALYZER._expected_schedule(reverse)
    spacings = []
    layout = []
    previous_terminal = None
    previous_gc = None
    for index in range(6):
        if index == 0:
            before = 1_000_000_000
            ready = before - 100_000_000
            target = None
            sleep = 0
            actual_gap = None
            lateness = None
        else:
            ready = int(previous_gc) + 2_000_000
            target = int(previous_terminal) + ANALYZER.PAUSE_NS
            sleep = max(0, target - ready)
            before = target + 10_000_000
            actual_gap = before - int(previous_terminal)
            lateness = before - target
        terminal = before + 1_000_000_000
        after_gc = terminal + 35_000_000_000
        spacings.append({
            "clock": "mach_absolute_time_nanoseconds",
            "previous_terminal_end_ns": previous_terminal,
            "diagnostic_ready_ns": ready,
            "target_encode_start_ns": target,
            "before_encode_ns": before,
            "requested_gap_ns": ANALYZER.PAUSE_NS,
            "requested_sleep_ns": sleep,
            "actual_gap_ns": actual_gap,
            "lateness_ns": lateness,
            "allowed_lateness_ns": ANALYZER.ALLOWED_LATENESS_NS,
        })
        layout.append({"before_encode": before, "terminal_end": terminal, "after_gc": after_gc})
        previous_terminal, previous_gc = terminal, after_gc
    return spacings, memory_events_aa(layout)


def report_document(reverse: bool) -> dict[str, Any]:
    expected = ANALYZER._expected_schedule(reverse)
    spacings, memory = make_spacing_and_memory(reverse)
    prior = PREVIOUS.report_document(False)
    requests = []
    for index, arm in enumerate(expected):
        request = PREVIOUS.request_document(index, False, arm["profile"])
        request["kind"] = arm["kind"]
        request["shared_scratch_poisoned_before_request"] = arm["poisoned"]
        request["terminal_wait_end_mach_ns"] = spacings[index]["before_encode_ns"] + 1_000_000_000
        request["aa_spacing"] = spacings[index]
        requests.append(request)
    schedule = [{"kind": arm["kind"], "combined_attention": arm["combined_attention"],
                 "alternate": arm["alternate"], "poisoned": arm["poisoned"],
                 "profiling": arm["profile"]} for arm in expected]
    return {
        **prior,
        "schema": ANALYZER.REPORT,
        "profile_aa": True,
        "reverse_aa_profile": reverse,
        "profile_combined_attention": False,
        "compare_combined_attention": False,
        "profile_variance": False,
        "profile_shared_gate_up": False,
        "compare_shared_gate_up": False,
        "reverse_combined_attention": False,
        "correctness_only": False,
        "aa_pause_seconds": 50.0,
        "aa_allowed_lateness_seconds": 0.25,
        "expected_request_count": 6,
        "aa_schedule": schedule,
        "combined_attention_schedule": copy.deepcopy(schedule),
        "shared_gate_up_schedule": [{"kind": arm["kind"], "shared_gate_up": False,
                                    "profiling": arm["profile"]} for arm in expected],
        "requests": requests,
        "memory_phases": memory,
    }


class PixelUMMAAAnalysisTests(unittest.TestCase):
    def analyze_small(self, report: dict[str, Any], label: str = "fixture.json") -> dict[str, Any]:
        raw = json.dumps(report, sort_keys=True, allow_nan=True).encode()
        return ANALYZER._validate_report(report, hashlib.sha256(raw).hexdigest(), label, expected_segments=2)

    def assert_rejected(self, mutate) -> None:
        raw = copy.deepcopy(report_document(False))
        mutate(raw)
        with self.assertRaises(ValueError):
            self.analyze_small(raw, "mutant.json")

    def test_normal_and_reverse_a_a_profiles_recompute_descriptive_summary(self) -> None:
        normal = self.analyze_small(report_document(False), "normal.json")
        reverse = self.analyze_small(report_document(True), "reverse.json")
        summary = ANALYZER.analyze_reports([normal, reverse])
        self.assertEqual(summary["schema"], "cogni_ml.pixelumm_aa_profile_analysis/v1")
        self.assertEqual(summary["lifecycle_scope"]["requests_per_run"], 6)
        self.assertEqual(summary["lifecycle_scope"]["terminal_waits_per_request"], 1)
        self.assertEqual(summary["numerical_scope"]["captures_per_request"], 43)
        self.assertEqual(summary["pause_scope"]["requested_gap_seconds"], 50.0)
        self.assertTrue(summary["pause_scope"]["last_request_has_no_following_pause"])
        normal_run = summary["runs"][0]
        self.assertEqual([row["profiling"] for row in normal_run["requests"]],
                         [False, False, True, True, False, False])
        self.assertEqual(normal_run["requests"][0]["aa_spacing"]["actual_gap_ns"], None)
        self.assertEqual(normal_run["requests"][1]["aa_spacing"]["actual_gap_seconds"], 50.01)
        self.assertEqual(len(normal_run["profile_localization"]["profiled_requests"]), 2)
        self.assertEqual(normal_run["memory_event_gaps"]["comparison_stage_gap_counts"],
                         {"readback_setup": 43, "compare_and_digest": 43})
        self.assertIn("continuation_trace", normal_run["profile_localization"])
        self.assertIn("descriptive_drift", normal_run)
        self.assertIn("causal profiler effect", summary["numerical_scope"]["not_claimed"])
        self.assertIn("acceleration", summary["numerical_scope"]["not_claimed"])

    def test_reverse_profile_placement_and_fixed_baseline_schedule_are_enforced(self) -> None:
        self.assert_rejected(lambda doc: doc.__setitem__("profile_aa", False))
        self.assert_rejected(lambda doc: doc.__setitem__("profile_combined_attention", True))
        self.assert_rejected(lambda doc: doc.__setitem__("reverse_combined_attention", True))
        self.assert_rejected(lambda doc: doc["aa_schedule"][2].__setitem__("combined_attention", True))
        self.assert_rejected(lambda doc: doc["aa_schedule"][3].__setitem__("profiling", False))
        self.assert_rejected(lambda doc: doc["combined_attention_schedule"][2].__setitem__("kind", "wrong"))
        self.assert_rejected(lambda doc: doc["combined_attention_schedule"][2].__setitem__("alternate", 0))
        self.assert_rejected(lambda doc: doc["shared_gate_up_schedule"][1].__setitem__("shared_gate_up", True))
        self.assert_rejected(lambda doc: doc["requests"][0].__setitem__("combined_attention", True))
        self.assert_rejected(lambda doc: doc["requests"][2].__setitem__("kind", "wrong"))
        self.assert_rejected(lambda doc: doc["requests"][2].__setitem__("shared_scratch_poisoned_before_request", True))

    def test_external_capture_is_retained_but_excluded_from_timing_aggregates(self) -> None:
        normal = self.analyze_small(report_document(False), "normal.json")
        reverse = self.analyze_small(report_document(True), "reverse.json")
        summary = ANALYZER.analyze_reports([normal, reverse], external_captures=["reverse:1"])
        run = summary["runs"][1]
        self.assertTrue(run["requests"][1]["externally_captured"])
        self.assertEqual(run["descriptive_drift"]["timing_by_profile_selector"]["profile_on"]["request_indices"], [4])
        self.assertEqual([row["request"] for row in run["profile_localization"]["profiled_requests"]], [4])
        self.assertEqual([row["request"] for row in run["profile_localization"]["externally_captured_requests"]], [1])
        self.assertEqual(run["profile_localization"]["continuation_trace"]["event_count"], 1)
        for delta in run["descriptive_drift"]["adjacent_request_timing_deltas_seconds"]:
            self.assertNotIn(1, (delta["from_request"], delta["to_request"]))
        for annotations in (["reverse:6"], ["other:1"], ["reverse:-1"], ["reverse:1", "reverse:1"]):
            with self.assertRaises(ValueError):
                ANALYZER.analyze_reports([normal, reverse], external_captures=annotations)

    def test_spacing_links_sleep_lateness_and_clock_order_fail_closed(self) -> None:
        self.assert_rejected(lambda doc: doc["requests"][1]["aa_spacing"].__setitem__("target_encode_start_ns", 1))
        self.assert_rejected(lambda doc: doc["requests"][1]["aa_spacing"].__setitem__("requested_sleep_ns", 0))
        self.assert_rejected(lambda doc: doc["requests"][1]["aa_spacing"].__setitem__("actual_gap_ns", 1))
        self.assert_rejected(lambda doc: doc["requests"][1]["aa_spacing"].__setitem__("previous_terminal_end_ns", 1))
        self.assert_rejected(lambda doc: doc["requests"][0]["aa_spacing"].__setitem__("actual_gap_ns", 0))
        self.assert_rejected(lambda doc: doc["requests"][2]["aa_spacing"].__setitem__("diagnostic_ready_ns", 2**64))
        self.assert_rejected(lambda doc: doc["requests"][2].__setitem__("terminal_wait_end_mach_ns", 1))
        self.assert_rejected(lambda doc: doc["memory_phases"][0].__setitem__("guard_monotonic_ns", 1))
        self.assert_rejected(lambda doc: doc["memory_phases"][0].__setitem__("phase", "wrong"))

    def test_spacing_accepts_exact_lateness_boundary_and_rejects_one_nanosecond_over(self) -> None:
        spacing = {
            "clock": "mach_absolute_time_nanoseconds",
            "previous_terminal_end_ns": 1_000_000_000,
            "diagnostic_ready_ns": 1_010_000_000,
            "target_encode_start_ns": 51_000_000_000,
            "before_encode_ns": 51_250_000_000,
            "requested_gap_ns": ANALYZER.PAUSE_NS,
            "requested_sleep_ns": 49_990_000_000,
            "actual_gap_ns": 50_250_000_000,
            "lateness_ns": ANALYZER.ALLOWED_LATENESS_NS,
            "allowed_lateness_ns": ANALYZER.ALLOWED_LATENESS_NS,
        }
        parsed = ANALYZER._validate_spacing(spacing, 1, 1_000_000_000, 1_005_000_000,
                                            51_251_000_000, 52_250_000_000, 52_251_000_000)
        self.assertEqual(parsed["lateness_ns"], ANALYZER.ALLOWED_LATENESS_NS)
        spacing["before_encode_ns"] += 1
        spacing["lateness_ns"] += 1
        spacing["actual_gap_ns"] += 1
        with self.assertRaises(ValueError):
            ANALYZER._validate_spacing(spacing, 1, 1_000_000_000, 1_005_000_000,
                                       51_251_000_001, 52_250_000_000, 52_251_000_000)

    def test_root_lifecycle_trace_finite_and_input_corruptions_fail(self) -> None:
        self.assert_rejected(lambda doc: doc["requests"][0].__setitem__("command_segments", 1))
        self.assert_rejected(lambda doc: doc["requests"][2]["root_profile"].__setitem__("root_count", 1))
        self.assert_rejected(lambda doc: doc["requests"][2]["root_profile"]["roots"][0].__setitem__("completed", False))
        self.assert_rejected(lambda doc: doc["requests"][2].__setitem__("terminal_wait_count", 2))
        self.assert_rejected(lambda doc: doc["requests"][2].__setitem__("named_allocations_during_encode", 1))
        self.assert_rejected(lambda doc: doc["requests"][2]["comparisons"][0].__setitem__("finite", False))
        self.assert_rejected(lambda doc: doc["requests"][2]["comparisons"][0].__setitem__("max_abs", float("nan")))
        self.assert_rejected(lambda doc: doc["requests"][2]["continuation_trace"]["events"][0].__setitem__("graph_depth", 0))
        self.assert_rejected(lambda doc: doc["requests"][2]["phase_profile"]["api_profile"]["phases"][2]["operations"]["graph_encode"].__setitem__("count", 1))
        self.assert_rejected(lambda doc: doc["requests"][5].__setitem__("input_sha256", sha("different A")))
        self.assert_rejected(lambda doc: doc["memory_phases"].pop(2))

    def test_pair_requires_one_normal_and_one_reverse_with_shared_fixtures(self) -> None:
        normal = self.analyze_small(report_document(False))
        reverse = self.analyze_small(report_document(True))
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports([normal, normal])
        reverse["same_input_sha256"] = sha("another input")
        with self.assertRaises(ValueError):
            ANALYZER.analyze_reports([normal, reverse])

    def test_production_segment_contract_and_cli_exclusive_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected exactly 4579"):
            ANALYZER.analyze_report(report_document(False), sha("small"), "small.json")
        with tempfile.TemporaryDirectory() as directory:
            normal = Path(directory) / "normal.json"
            reverse = Path(directory) / "reverse.json"
            output = Path(directory) / "analysis.json"
            normal.write_text(json.dumps(report_document(False)), encoding="utf-8")
            reverse.write_text(json.dumps(report_document(True)), encoding="utf-8")
            expected_segments = ANALYZER.EXPECTED_COMMAND_SEGMENTS
            ANALYZER.EXPECTED_COMMAND_SEGMENTS = 2
            try:
                args = ["--report", str(normal), "--report", str(reverse), "--output", str(output)]
                self.assertEqual(ANALYZER.main(args), 0)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema"],
                                 "cogni_ml.pixelumm_aa_profile_analysis/v1")
                self.assertEqual(ANALYZER.main(args), 2)
            finally:
                ANALYZER.EXPECTED_COMMAND_SEGMENTS = expected_segments


if __name__ == "__main__":
    unittest.main()
