"""CPU-only tests for matched-frequency PixelUMM profile analysis."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples/support/pixelumm_frequency_profile.py"
SPEC = importlib.util.spec_from_file_location("pixelumm_frequency_profile", MODULE_PATH)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "pixelumm_aa_analysis_fixture", ROOT / "spec/pixelumm_aa_analysis_test.py")
assert FIXTURE_SPEC and FIXTURE_SPEC.loader
FIXTURE = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURE)


ANCHOR_MONO = 10_000_000_000
ANCHOR_UNIX = 1_800_000_000_000_000_000
BASE_GUARD = ANCHOR_MONO + 1_000_000_000


def report_document() -> dict[str, Any]:
    """Use the repository AA fixture, widened to provide telemetry interiors."""
    report = FIXTURE.report_document(False)
    report.update({"compare_cpu_diagnostics": False,
                   "reverse_cpu_diagnostics": False,
                   "align_cpu_diagnostic_lifecycle": False,
                   "reverse_tiled_gate_up": False,
                   "tiled_gate_up_rows": 0})
    previous_terminal = None
    previous_gc = None
    layout = []
    for index in range(6):
        if index == 0:
            before = BASE_GUARD
            ready = before - 100_000_000
            target = None
            sleep = 0
            gap = None
            lateness = None
        else:
            ready = int(previous_gc) + 2_000_000
            target = int(previous_terminal) + 50_000_000_000
            before = target + 10_000_000
            sleep = max(0, target - ready)
            gap = before - int(previous_terminal)
            lateness = before - target
        terminal = before + 30_000_000_000
        after_gc = terminal + 35_000_000_000
        layout.append({"before_encode": before, "terminal_end": terminal, "after_gc": after_gc})
        row = report["requests"][index]
        row["terminal_wait_end_mach_ns"] = terminal
        row["aa_spacing"] = {
            "clock": "mach_absolute_time_nanoseconds",
            "previous_terminal_end_ns": previous_terminal,
            "diagnostic_ready_ns": ready,
            "target_encode_start_ns": target,
            "before_encode_ns": before,
            "requested_gap_ns": 50_000_000_000,
            "requested_sleep_ns": sleep,
            "actual_gap_ns": gap,
            "lateness_ns": lateness,
            "allowed_lateness_ns": 250_000_000,
        }
        previous_terminal, previous_gc = terminal, after_gc
    report["memory_phases"] = FIXTURE.memory_events_aa(layout)
    return report


def launch_document(report: dict[str, Any]) -> dict[str, Any]:
    fixture_fields = (("--reference", "--reference-sha256", "reference_sha256"),
                      ("--alternate-reference", "--alternate-reference-sha256", "alternate_reference_sha256"),
                      ("--head-reference", "--head-reference-sha256", "head_reference_sha256"),
                      ("--head-alternate-reference", "--head-alternate-reference-sha256", "head_alternate_reference_sha256"),
                      ("--bf16-cache", "--bf16-cache-sha256", "bf16_cache_sha256"))
    argv = ["pixelumm_decoder_integrated_probe"]
    pins = {}
    for index, (path_option, digest_option, report_field) in enumerate(fixture_fields):
        path = f"/fixtures/input-{index}.json"
        digest = report[report_field]
        argv.extend((path_option, path, digest_option, digest))
        pins[path] = digest
    argv.append("--profile-aa")
    return {"schema": "cogni_ml.pixelumm_matched_frequency_profile/v1",
            "native_argv": argv, "fixture_pins": pins,
            "clock_anchor": {"monotonic_before_ns": ANCHOR_MONO - 500,
                             "unix_ns": ANCHOR_UNIX,
                             "monotonic_after_ns": ANCHOR_MONO + 500,
                             "implementation": "mach_absolute_time()"},
            "frequency_gate": {"max_mean_relative_difference": 0.02,
                               "minimum_interior_samples": 5,
                               "all_observed_samples_valid": True,
                               "minimum_mean_active_residency_pct": 99.0}}


def observer_inputs(report: dict[str, Any], launch: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor = launch["clock_anchor"]
    midpoint = (anchor["monotonic_before_ns"] + anchor["monotonic_after_ns"]) // 2
    extra_ms = (max(midpoint - anchor["monotonic_before_ns"],
                    anchor["monotonic_after_ns"] - midpoint) + 999_999) // 1_000_000 + 1
    bounds = {}
    for index, event in enumerate(report["memory_phases"]):
        if event["phase"] in ("request_before_encode", "request_after_terminal_wait"):
            bounds.setdefault(event["request"], {})[event["phase"]] = event["guard_monotonic_ns"]
    rows: list[dict[str, Any]] = []
    utc_bounds = []
    for request_id in range(6):
        guard = bounds[request_id]
        start = (anchor["unix_ns"] + guard["request_before_encode"] - midpoint) // 1_000_000 + extra_ms
        end = (anchor["unix_ns"] + guard["request_after_terminal_wait"] - midpoint) // 1_000_000 - extra_ms
        utc_bounds.append((start, end))
    start = min(left for left, _ in utc_bounds)
    end = max(right for _, right in utc_bounds)
    def utc(value: int) -> str:
        return datetime.fromtimestamp(value / 1_000, timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")
    for window_end in range(start + 2_000, end + 1, 2_000):
        window_start = window_end - 2_000
        rows.append({"sample_index": len(rows),
                     "timestamp_unix_ms": window_end,
                     "timestamp_utc": utc(window_end),
                     "timestamp_resolution_ms": 1_000,
                     "timestamp_is_window_end_assumption": True,
                     "sample_window_start_unix_ms": window_start,
                     "sample_window_start_utc": utc(window_start),
                     "sample_window_end_unix_ms": window_end,
                     "sample_window_end_utc": utc(window_end),
                     "elapsed_ms": 2_000.0,
                     "gpu_active_frequency_mhz": 1_350.0,
                     "gpu_active_residency_pct": 99.5,
                     "gpu_power_mw": 8_000.0, "thermal_pressure": "nominal",
                     "valid": True})
    sample_bytes = b"".join((json.dumps(row, sort_keys=True, allow_nan=False,
                                        separators=(",", ":")) + "\n").encode() for row in rows)
    first_guard = min(value["request_before_encode"] for value in bounds.values())
    last_guard = max(value["request_after_terminal_wait"] for value in bounds.values())
    pgid = 9876
    started_ms = (anchor["unix_ns"] + first_guard - 2_000_000_000 - midpoint) // 1_000_000
    finished_ms = (anchor["unix_ns"] + last_guard + 2_000_000_000 - midpoint) // 1_000_000
    summary = {"schema": "cogni_ml.pixelumm_power_observer/v1", "status": "stopped_early",
               "collector": {"sampler_stopped": True, "sampler_process_group_id": pgid,
                             "planned_sample_count": 600, "actual_returncode": 0,
                             "interrupted_by_signal": 15, "sampler_terminated_by_signal": False,
                             "timed_out": False, "launch_error": None,
                             "sample_interval_ms": 2_000,
                             "stop_attempts": [{"signal": "INT", "sent": True,
                                                "returncode": 0, "error": None}],
                             "sampler_group_checks": [{"state": "absent", "process_group_id": pgid,
                                                       "returncode": 1, "stderr_excerpt": None, "error": None}]},
               "raw_capture": {"file": "powermetrics.txt", "finalized": True,
                               "sha256": hashlib.sha256(b"raw capture").hexdigest(),
                               "size_bytes": 12},
               "clock": {"started_utc": utc(started_ms), "finished_utc": utc(finished_ms),
                         "started_monotonic_ns": first_guard - 2_000_000_000,
                         "finished_monotonic_ns": last_guard + 2_000_000_000,
                         "monotonic_elapsed_ns": last_guard - first_guard + 4_000_000_000,
                         "sample_timestamp_resolution_ms": 1_000},
               "parsed_samples": {"file": "samples.jsonl",
                                  "sha256": hashlib.sha256(sample_bytes).hexdigest(),
                                  "size_bytes": len(sample_bytes),
                                  "coverage": {"sample_count": len(rows),
                                                "valid_sample_count": len(rows),
                                                "expected_sample_count": 600,
                                                "duration_complete": False, "interrupted": True,
                                                "all_observed_samples_valid": True,
                                                "valid_sample_fraction": 1.0}}}
    return summary, rows


def refresh_sample_integrity(summary: dict[str, Any], samples: list[dict[str, Any]], *,
                             renumber: bool = False) -> None:
    if renumber:
        for index, sample in enumerate(samples):
            sample["sample_index"] = index
    for sample in samples:
        for key in ("timestamp_utc", "sample_window_start_utc", "sample_window_end_utc"):
            unix_key = {"timestamp_utc": "timestamp_unix_ms",
                        "sample_window_start_utc": "sample_window_start_unix_ms",
                        "sample_window_end_utc": "sample_window_end_unix_ms"}[key]
            sample[key] = datetime.fromtimestamp(sample[unix_key] / 1_000, timezone.utc).isoformat(
                timespec="milliseconds").replace("+00:00", "Z")
    raw = b"".join((json.dumps(row, sort_keys=True, allow_nan=False,
                               separators=(",", ":")) + "\n").encode() for row in samples)
    summary["parsed_samples"]["sha256"] = hashlib.sha256(raw).hexdigest()
    summary["parsed_samples"]["size_bytes"] = len(raw)
    summary["parsed_samples"]["coverage"]["sample_count"] = len(samples)
    summary["parsed_samples"]["coverage"]["valid_sample_count"] = len(samples)


def shift_utc(value: str, delta_ms: int) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    timestamp_ms = round(parsed.timestamp() * 1_000) + delta_ms
    return datetime.fromtimestamp(timestamp_ms / 1_000, timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


class PixelUMMFrequencyProfileTests(unittest.TestCase):
    def inputs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        report = report_document()
        launch = launch_document(report)
        summary, samples = observer_inputs(report, launch)
        return report, launch, summary, samples

    def analyze_small(self, report: dict[str, Any], launch: dict[str, Any],
                      summary: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
        return ANALYZER.analyze_documents(report, launch, summary, samples,
                                          expected_command_segments=2,
                                          expected_token_count=None)

    def test_matched_frequency_profile_returns_exclusive_union_buckets(self) -> None:
        report, launch, summary, samples = self.inputs()
        result = self.analyze_small(report, launch, summary, samples)
        self.assertEqual(result["schema"], "cogni_ml.pixelumm_frequency_profile_analysis/v1")
        self.assertTrue(result["frequency_match_gate"]["passed"])
        accounting = result["profiled_requests"][0]["gpu_interval_accounting"]
        self.assertGreater(accounting["chain_union_seconds"], 0.0)
        self.assertEqual(accounting["disjoint_membership_total_seconds"], accounting["chain_union_seconds"])
        self.assertEqual(len(result["profiled_requests"]), 2)
        self.assertTrue(result["profiled_requests"][0]["api_profile_accounting"]["available"])
        self.assertIn("not per-shader service time", result["gpu_interval_caveat"])

    def test_nearest_off_requests_are_frequency_controls(self) -> None:
        report, launch, summary, samples = self.inputs()
        result = self.analyze_small(report, launch, summary, samples)
        comparisons = result["frequency_match_gate"]["profiled_requests"][0]["comparisons"]
        self.assertEqual([row["control_request"] for row in comparisons], [1, 4])
        self.assertTrue(all(row["passed"] for row in comparisons))
        # A mismatch is a gate failure, retained as a diagnostic result.
        req2_before = next(row["guard_monotonic_ns"] for row in report["memory_phases"]
                           if row["request"] == 2 and row["phase"] == "request_before_encode")
        req2_after = next(row["guard_monotonic_ns"] for row in report["memory_phases"]
                          if row["request"] == 2 and row["phase"] == "request_after_terminal_wait")
        midpoint = launch["clock_anchor"]["monotonic_before_ns"] + 500
        req2_start_ms = (ANCHOR_UNIX + req2_before - midpoint) // 1_000_000
        req2_end_ms = (ANCHOR_UNIX + req2_after - midpoint) // 1_000_000
        for sample in samples:
            if req2_start_ms <= sample["sample_window_start_unix_ms"] < req2_end_ms:
                sample["gpu_active_frequency_mhz"] = 1_200.0
        refresh_sample_integrity(summary, samples)
        result = self.analyze_small(report, launch, summary, samples)
        self.assertFalse(result["frequency_match_gate"]["passed"])

    def test_mixed_root_membership_and_overlapping_intervals_conserve_union(self) -> None:
        roots = [
            {"gpu_start_seconds": 0.0, "gpu_end_seconds": 3.0},
            {"gpu_start_seconds": 1.0, "gpu_end_seconds": 2.0},
            {"gpu_start_seconds": 2.0, "gpu_end_seconds": 4.0},
        ]
        candidates = [["block_0/attention"], ["block_0/attention", "block_0/ffn_down"],
                      ["block_0/ffn_down"]]
        accounting = ANALYZER.interval_accounting(roots, candidates)
        self.assertEqual(accounting["chain_union_seconds"], 4.0)
        self.assertEqual(accounting["disjoint_membership_buckets"], [
            {"active_root_candidate_family_sets": [["attention"]], "seconds": 1.0},
            {"active_root_candidate_family_sets": [["attention"], ["attention", "ffn_down"]],
             "seconds": 1.0},
            {"active_root_candidate_family_sets": [["attention"], ["ffn_down"]], "seconds": 1.0},
            {"active_root_candidate_family_sets": [["ffn_down"]], "seconds": 1.0},
        ])
        self.assertEqual(accounting["disjoint_membership_total_seconds"], 4.0)

    def test_rejects_wrong_or_missing_clock_anchor_and_duplicate_boundaries(self) -> None:
        report, launch, summary, samples = self.inputs()
        wrong_clock = copy.deepcopy(launch)
        wrong_clock["clock_anchor"]["implementation"] = "CLOCK_MONOTONIC"
        with self.assertRaises(ValueError):
            self.analyze_small(report, wrong_clock, summary, samples)
        missing_clock = copy.deepcopy(report)
        missing_clock["memory_phases"].pop()
        with self.assertRaises(ValueError):
            self.analyze_small(missing_clock, launch, summary, samples)
        duplicate = copy.deepcopy(report)
        duplicate["memory_phases"].append(copy.deepcopy(duplicate["memory_phases"][0]))
        with self.assertRaises(ValueError):
            self.analyze_small(duplicate, launch, summary, samples)

    def test_production_defaults_pin_current_token_inventory_and_fixture_hashes(self) -> None:
        report, launch, summary, samples = self.inputs()
        with self.assertRaises(ValueError):
            ANALYZER._validate_report(report, expected_command_segments=2)
        wrong_frequency_gate = copy.deepcopy(launch)
        wrong_frequency_gate["frequency_gate"]["max_mean_relative_difference"] = 0.021
        with self.assertRaises(ValueError):
            self.analyze_small(report, wrong_frequency_gate, summary, samples)
        wrong_fixture = copy.deepcopy(report)
        wrong_fixture["head_reference_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.analyze_small(wrong_fixture, launch, summary, samples)

    def test_default_profile_fixture_requires_native_zero_tiled_rows(self) -> None:
        report, _, _, _ = self.inputs()
        self.assertEqual(report["tiled_gate_up_rows"], 0)
        ANALYZER._validate_report(report, expected_command_segments=2,
                                  expected_token_count=None)
        for value in (None, 1, True):
            incompatible = copy.deepcopy(report)
            incompatible["tiled_gate_up_rows"] = value
            with self.assertRaises(ValueError):
                ANALYZER._validate_report(incompatible, expected_command_segments=2,
                                          expected_token_count=None)

    def test_rejects_failed_report_nonfinite_samples_and_missing_coverage(self) -> None:
        report, launch, summary, samples = self.inputs()
        failed = copy.deepcopy(report)
        failed["passed"] = False
        with self.assertRaises(ValueError):
            self.analyze_small(failed, launch, summary, samples)
        invalid_samples = copy.deepcopy(samples)
        invalid_samples[0]["gpu_active_frequency_mhz"] = float("nan")
        with self.assertRaises(ValueError):
            self.analyze_small(report, launch, summary, invalid_samples)
        summary["parsed_samples"]["coverage"]["all_observed_samples_valid"] = False
        with self.assertRaises(ValueError):
            self.analyze_small(report, launch, summary, samples)

    def test_rejects_frequency_windows_with_insufficient_interior_samples(self) -> None:
        report, launch, summary, samples = self.inputs()
        summarize = ANALYZER._OBSERVER.summarize_utc_ranges
        def insufficient(samples_arg, ranges_arg):
            rows = summarize(samples_arg, ranges_arg)
            rows["request_2"].update({"eligible_samples": 4, "valid_samples": 4,
                                      "valid_sample_fraction": 1.0})
            return rows
        with patch.object(ANALYZER._OBSERVER, "summarize_utc_ranges", side_effect=insufficient):
            with self.assertRaisesRegex(ValueError, "request_2 lacks the required valid interior power samples"):
                self.analyze_small(report, launch, summary, samples)

    def test_observer_provenance_and_capture_interval_fail_closed(self) -> None:
        report, launch, summary, samples = self.inputs()
        requests, memory = report["requests"], report["memory_phases"]
        anchor = ANALYZER._validate_launch(launch)[0]
        ranges = ANALYZER._request_ranges(requests, memory, anchor)
        ANALYZER._validate_observer(summary, samples, ranges, anchor, None)
        original_summary = copy.deepcopy(summary)

        complete = copy.deepcopy(original_summary)
        complete["status"] = "complete"
        complete["collector"]["planned_sample_count"] = len(samples)
        complete["collector"]["interrupted_by_signal"] = None
        complete["collector"]["stop_attempts"] = []
        complete["parsed_samples"]["coverage"]["expected_sample_count"] = len(samples)
        complete["parsed_samples"]["coverage"]["duration_complete"] = True
        complete["parsed_samples"]["coverage"]["interrupted"] = False
        ANALYZER._validate_observer(complete, samples, ranges, anchor, None)

        for mutate in (
                lambda value: value["collector"].__setitem__("sampler_stopped", False),
                lambda value: value["collector"]["sampler_group_checks"][-1].__setitem__("state", "present"),
                lambda value: value["raw_capture"].__setitem__("finalized", False),
                lambda value: value["parsed_samples"]["coverage"].__setitem__("duration_complete", True)):
            broken = copy.deepcopy(summary)
            mutate(broken)
            with self.assertRaises(ValueError):
                ANALYZER._validate_observer(broken, samples, ranges, anchor, None)

        forced_kill = copy.deepcopy(original_summary)
        forced_kill["collector"]["stop_attempts"].append(
            {"signal": "KILL", "sent": True, "returncode": 0})
        with self.assertRaises(ValueError):
            ANALYZER._validate_observer(forced_kill, samples, ranges, anchor, None)

        missing_stop = copy.deepcopy(summary)
        missing_stop["collector"]["stop_attempts"] = []
        with self.assertRaisesRegex(ValueError, "no recorded sampler SIGINT stop"):
            ANALYZER._validate_observer(missing_stop, samples, ranges, anchor, None)

        for field, value in (("sent", False), ("returncode", 1), ("returncode", False),
                             ("error", "permission denied"), ("stderr_excerpt", "failed")):
            failed_stop = copy.deepcopy(summary)
            failed_stop["collector"]["stop_attempts"][0][field] = value
            with self.assertRaisesRegex(ValueError, "SIGINT attempt did not succeed"):
                ANALYZER._validate_observer(failed_stop, samples, ranges, anchor, None)

        out_of_capture = copy.deepcopy(samples)
        out_of_capture[0]["sample_window_start_unix_ms"] = 1
        refresh_sample_integrity(summary, out_of_capture)
        with self.assertRaises(ValueError):
            ANALYZER._validate_observer(summary, out_of_capture, ranges, anchor, None)

        duplicate_index = copy.deepcopy(samples)
        duplicate_index[1]["sample_index"] = duplicate_index[0]["sample_index"]
        refresh_sample_integrity(summary, duplicate_index)
        with self.assertRaises(ValueError):
            ANALYZER._validate_observer(summary, duplicate_index, ranges, anchor, None)

        missing_interval = copy.deepcopy(samples)
        missing_interval.pop(1)
        refresh_sample_integrity(summary, missing_interval, renumber=True)
        with self.assertRaises(ValueError):
            ANALYZER._validate_observer(summary, missing_interval, ranges, anchor, None)

        quantized_overlap = copy.deepcopy(samples)
        quantized_overlap[1]["elapsed_ms"] = 2_010.0
        quantized_overlap[1]["sample_window_start_unix_ms"] = (
            quantized_overlap[1]["sample_window_end_unix_ms"] - 2_010)
        refresh_sample_integrity(summary, quantized_overlap)
        ANALYZER._validate_observer(summary, quantized_overlap, ranges, anchor, None)
        too_much_overlap = copy.deepcopy(quantized_overlap)
        too_much_overlap[1]["elapsed_ms"] = 3_001.0
        too_much_overlap[1]["sample_window_start_unix_ms"] = (
            too_much_overlap[1]["sample_window_end_unix_ms"] - 3_001)
        refresh_sample_integrity(summary, too_much_overlap)
        with self.assertRaises(ValueError):
            ANALYZER._validate_observer(summary, too_much_overlap, ranges, anchor, None)

        summary = original_summary
        canonical = b"".join((json.dumps(row, sort_keys=True, allow_nan=False,
                                           separators=(",", ":")) + "\n").encode()
                             for row in samples)
        with self.assertRaises(ValueError):
            ANALYZER._validate_observer(summary, samples, ranges, anchor, b" " + canonical)

    def test_observer_utc_endpoints_are_anchored_and_elapsed_consistent(self) -> None:
        report, launch, summary, samples = self.inputs()
        anchor = ANALYZER._validate_launch(launch)[0]
        ranges = ANALYZER._request_ranges(report["requests"], report["memory_phases"], anchor)
        provenance = ANALYZER._validate_observer(summary, samples, ranges, anchor, None)
        residuals = provenance["utc_endpoint_anchor_residual_ms"]
        self.assertLessEqual(abs(residuals["started"]), 100)
        self.assertLessEqual(abs(residuals["finished"]), 100)
        self.assertLessEqual(abs(provenance["utc_wall_elapsed_residual_ms"]), 200)

        shifted_both = copy.deepcopy(summary)
        shifted_both["clock"]["started_utc"] = shift_utc(
            shifted_both["clock"]["started_utc"], 1_000)
        shifted_both["clock"]["finished_utc"] = shift_utc(
            shifted_both["clock"]["finished_utc"], 1_000)
        with self.assertRaisesRegex(ValueError, "endpoint residual exceeds the 100ms"):
            ANALYZER._validate_observer(shifted_both, samples, ranges, anchor, None)

        wall_drift = copy.deepcopy(summary)
        wall_drift["clock"]["finished_utc"] = shift_utc(
            wall_drift["clock"]["finished_utc"], 1_000)
        with self.assertRaisesRegex(ValueError, "endpoint residual exceeds the 100ms"):
            ANALYZER._validate_observer(wall_drift, samples, ranges, anchor, None)

    def test_cli_writes_only_a_fresh_output_file(self) -> None:
        report, launch, summary, samples = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"report": root / "report.json", "launch": root / "launch.json",
                     "summary": root / "summary.json", "samples": root / "samples.jsonl",
                     "output": root / "result.json"}
            paths["report"].write_text(json.dumps(report))
            paths["launch"].write_text(json.dumps(launch))
            paths["summary"].write_text(json.dumps(summary))
            paths["samples"].write_bytes(b"".join((json.dumps(row, sort_keys=True, allow_nan=False,
                                                              separators=(",", ":")) + "\n").encode()
                                                 for row in samples))
            arguments = ["--report", str(paths["report"]), "--launch", str(paths["launch"]),
                         "--observer-summary", str(paths["summary"]), "--samples", str(paths["samples"]),
                         "--output", str(paths["output"])]
            analyze = ANALYZER.analyze_documents
            def small_fixture_analyzer(report_value, launch_value, summary_value, sample_value,
                                       *, samples_bytes=None):
                return analyze(report_value, launch_value, summary_value, sample_value,
                               samples_bytes=samples_bytes, expected_command_segments=2,
                               expected_token_count=None)
            with patch.object(ANALYZER, "analyze_documents", side_effect=small_fixture_analyzer):
                self.assertEqual(ANALYZER.main(arguments), 0)
                self.assertEqual(json.loads(paths["output"].read_text())["schema"],
                                 "cogni_ml.pixelumm_frequency_profile_analysis/v1")
                with self.assertRaises(SystemExit) as rejected:
                    ANALYZER.main(arguments)
                self.assertEqual(rejected.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
