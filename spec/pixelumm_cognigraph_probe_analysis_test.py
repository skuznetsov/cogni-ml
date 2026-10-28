from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "examples/support/analyze_pixelumm_cognigraph_probe.py"
SPEC = importlib.util.spec_from_file_location("pixelumm_cognigraph_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def report_fixture() -> dict:
    requests = []
    previous = False
    for index, (enabled, kind, capture, positive, timed) in enumerate(ANALYZER.expected_requests()):
        metrics = [{"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 0}
                   for _ in range(29)]
        slots = [4 + plan * 6 + stage for plan in range(4) for stage in range(6)]
        outputs = [4 + plan * 6 + 5 for plan in range(4)]
        for slot in slots if capture else outputs:
            metrics[slot]["elements"] = 1
        if positive:
            metrics[28] = {"failures": 1, "different_words": 0, "nonfinite": 1, "elements": 1}
        scale = 0.9 if enabled else 1.0
        requests.append({
            "index": index,
            "kind": kind,
            "native_islands_enabled": enabled,
            "native_mode_toggled_before_request": index == 0 or enabled != previous,
            "capture_mode": capture,
            "positive_nan_control": positive,
            "timing_eligible": timed,
            "full_request_wall_seconds": scale,
            "host_clock_wall_seconds": scale,
            "host_thread_cpu_seconds": scale * 0.4,
            "body_encode_seconds": scale * 0.5,
            "comparator_encode_seconds": scale * 0.1,
            "terminal_wait_wall_seconds": scale * 0.2,
            "native_gpu_elapsed_seconds": scale * 0.7,
            "gpu_root_duration_sum_seconds": scale * 0.7,
            "gpu_root_union_seconds": scale * 0.6,
            "gpu_root_overlap_seconds": scale * 0.1,
            "root_count": 3,
            "terminal_wait_count": 1,
            "named_buffer_allocations": 0,
            "preterminal_readbacks": 0,
            "comparator_readbacks_after_wait": 1,
            "numerical_metrics": metrics,
        })
        previous = enabled
    stats = {
        "graphs": 75, "ops": 293, "waves": 76, "barriers": 1, "maxwidth": 4,
        "native_dispatches_perblock": 366, "baseline_encoders_perblock": 366,
        "candidate_encoders_perblock_no_captures": 149,
    }
    return {
        "schema": ANALYZER.REPORT_SCHEMA,
        "probe_process_id": 12,
        "probe_completed": True,
        "production_promoted": False,
        "full_generation_executed": False,
        "numerical_comparison": {"atol": 2e-5, "rtol": 2e-5, "result": "all comparator gates passed"},
        "input_pins": {
            "reference_manifest_sha256": "c" * 64,
            "reference_payload_sha256": "d" * 64,
            "weight_cache_manifest_sha256": ANALYZER.EXPECTED_WEIGHT_CACHE_SHA256,
            "weight_cache_payload_sha256": ANALYZER.EXPECTED_WEIGHT_CACHE_PAYLOAD_SHA256,
            "upstream_model_sha256": ANALYZER.EXPECTED_MODEL_SHA256,
            "upstream_revision": ANALYZER.EXPECTED_UPSTREAM_REVISION,
        },
        "capture_contract": {
            "request_capture_mode_changes_native_plan_capture_mask": True,
            "output_only_requests_retain_only_output_capture": True,
            "encoder_counts_are_prepared_plan_model_counters": True,
        },
        "measurement_semantics": {},
        "named_resources": {
            "named_bytes": 1500,
            "resource_bytes": {
                "workspace": 100, "inputs_and_weights": 200, "staging": 300,
                "captures": 400, "metrics": 500,
            },
            "native_plan_count": 4,
            "native_plan_handles_distinct": True,
            "opaque_native_plan_bytes": None,
            "opaque_native_plan_bytes_semantics": "unknown; excluded from named MetalBuffer inventory",
            "plan_input_binding_sha256": ["a" * 64, "b" * 64],
            "capture_slots_private": True,
            "f32_stage_buffers_shared": True,
        },
        "native_island_stats": [stats.copy() for _ in range(4)],
        "native_island_stats_interpretation": "encoder counts are prepared-plan model counters, not measured calls",
        "native_api_negative_controls": {
            "unprepared_enable_status": -2,
            "duplicate_prepare_status": -2,
            "invalid_enabled_status": -1,
            "late_prepare_after_drained_request_status": -2,
        },
        "resource_guard": {
            "max_process_footprint_mib": 16_384,
            "minimum_free_memory_percent": 10,
            "maximum_swap_growth_mib": 64,
            "enforced_by_external_memory_guard": True,
        },
        "request_count": 40,
        "setup_seconds": 5.0,
        "requests": requests,
        "summary": {
            "correctness_gates_passed": True,
            "candidate_requests": 20,
            "baseline_requests": 20,
            "timed_candidate_requests": 16,
            "timed_baseline_requests": 16,
        },
    }


def guard_fixture() -> list[dict]:
    mib = 1024 * 1024
    gib = 1024**3
    sample = {
        "physical_bytes": 100 * gib,
        "headroom_bytes": 90 * gib,
        "free_spec_bytes": 50 * gib,
        "file_backed_bytes": 50 * gib,
        "speculative_bytes": 10 * gib,
        "pressure_free_pct": 30,
    }
    return [
        {"event": "limits", "runtime_floor_pct": 10, "runtime_floor_pct_exact": "10",
         "memory_accounting": "file-cache",
         "headroom_formula": "printed free + file-backed (includes speculative)",
         "startup_rule": "memory_pressure -Q >= 10% and selected headroom >= 10% physical + footprint budget",
         "max_footprint_bytes": 16_384 * mib, "max_swap_growth_bytes": 64 * mib,
         "timeout_sec": 600, "monotonic": 1.0},
        {"event": "sample", **sample, "group_footprint_bytes": 1000,
         "swap_used_bytes": 100 * mib, "group_live_processes": 0, "monotonic": 1.1},
        {"event": "started", "pid": 12, "pgid": 12, "monotonic": 1.2},
        {"event": "sample", **{**sample, "pressure_free_pct": 29},
         "group_footprint_bytes": 2000, "swap_used_bytes": 101 * mib,
         "group_live_processes": 1, "monotonic": 1.3},
        {"event": "sample", **{**sample, "pressure_free_pct": 29},
         "group_footprint_bytes": 2000, "swap_used_bytes": 101 * mib,
         "group_live_processes": 0, "monotonic": 1.4},
        {"event": "finished", "returncode": 0, "monotonic": 1.5},
    ]


class PixelUMMCogniGraphAnalyzerTest(unittest.TestCase):
    def test_balanced_schedule_and_paired_metrics(self) -> None:
        summary = ANALYZER.validate_report(report_fixture(), "fixture-digest")
        self.assertEqual(summary["request_counts"]["timed_candidate"], 16)
        self.assertEqual(summary["request_counts"]["timed_baseline"], 16)
        for metric in ANALYZER.METRIC_NAMES:
            result = summary["paired_relative_metrics"][metric]
            self.assertEqual(result["pair_count"], 16)
            self.assertLess(result["candidate_over_baseline_median_relative"], 0.0)
        self.assertEqual(summary["timed_root_counts"]["candidate"]["total_roots"], 48)

    def test_rejects_changed_tolerance_unbalanced_pair_or_extra_readback(self) -> None:
        data = report_fixture()
        data["numerical_comparison"]["atol"] = 1e-4
        with self.assertRaises(ValueError):
            ANALYZER.validate_report(data)

        data = report_fixture()
        data["named_resources"]["native_plan_count"] = 0
        with self.assertRaises(ValueError):
            ANALYZER.validate_report(data)

        data = report_fixture()
        data["input_pins"]["weight_cache_manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            ANALYZER.validate_report(data)

        data = report_fixture()
        data["requests"][8]["native_islands_enabled"] = True
        with self.assertRaises(ValueError):
            ANALYZER.validate_report(data)

        data = report_fixture()
        data["requests"][0]["native_mode_toggled_before_request"] = False
        with self.assertRaises(ValueError):
            ANALYZER.validate_report(data)

        data = report_fixture()
        data["requests"][9]["comparator_readbacks_after_wait"] = 2
        with self.assertRaises(ValueError):
            ANALYZER.validate_report(data)

    def test_guard_enforces_footprint_floor_and_swap_cap(self) -> None:
        summary = ANALYZER.validate_guard(guard_fixture(), 12)
        self.assertEqual(summary["minimum_free_memory_percent"], 29.0)
        self.assertEqual(summary["minimum_sampled_headroom_bytes"], 90 * 1024**3)
        self.assertEqual(summary["probe_process_id"], 12)
        bad = guard_fixture()
        bad[0]["max_swap_growth_bytes"] -= 1
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)

        bad = guard_fixture()
        bad[0]["memory_accounting"] = "free-speculative"
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)

        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(guard_fixture(), 99)

        bad = guard_fixture()
        bad[1]["headroom_bytes"] -= 1
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)

    def test_guard_rejects_admission_start_and_cleanup_failures(self) -> None:
        bad = guard_fixture()
        bad[1]["headroom_bytes"] = 20 * 1024**3
        bad[1]["free_spec_bytes"] = 20 * 1024**3
        bad[1]["file_backed_bytes"] = 10 * 1024**3
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)

        bad = guard_fixture()
        bad.insert(-1, {"event": "cleanup_incomplete", "monotonic": 1.35})
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)

        bad = guard_fixture()
        bad.insert(-1, {"event": "stopping", "reason": "telemetry unavailable", "monotonic": 1.35})
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)

        bad = guard_fixture()
        bad[2]["pgid"] += 1
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)

        bad = guard_fixture()
        bad[1]["group_live_processes"] = 1
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)

        bad = guard_fixture()
        bad[3]["group_live_processes"] = 0
        with self.assertRaises(ValueError):
            ANALYZER.validate_guard(bad, 12)


if __name__ == "__main__":
    unittest.main()
