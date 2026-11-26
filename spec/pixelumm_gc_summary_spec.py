"""CPU-only validation tests for the PixelUMM GC report summarizer."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from examples.support.pixelumm_gc_summary import main, summarize


POLICIES = ("normal", "scoped_disabled")
TIMED_PATTERN = ("normal", "scoped_disabled", "scoped_disabled", "normal",
                 "scoped_disabled", "normal", "normal", "scoped_disabled")
CAPTURE_ORDER = ("post_attn_norm", "gate_proj", "up_proj", "product", "down_proj", "output")


def point_state() -> dict[str, int]:
    return {
        "time_ns": 1,
        "allocated_bytes": 0,
        "gc_no": 0,
        "heap_bytes": 4096,
        "free_bytes": 2048,
        "bytes_since_gc": 0,
        "bytes_before_gc": 1024,
        "collection_count": 0,
        "collection_total_ns": 0,
        "collection_max_ns": 0,
        "rendezvous_pause_count": 0,
        "rendezvous_pause_total_ns": 0,
        "rendezvous_pause_max_ns": 0,
        "stopped_interval_count": 0,
        "stopped_interval_total_ns": 0,
        "stopped_interval_max_ns": 0,
        "unmatched_event_count": 0,
    }


def delta(first: dict[str, int], last: dict[str, int]) -> dict[str, int | float]:
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
        "stopped_seconds": (last["stopped_interval_total_ns"] - first["stopped_interval_total_ns"]) / 1e9,
    }


def metric_array(trial_index: int) -> list[dict[str, int]]:
    metrics = [{"failures": 0, "different_words": slot % 3,
                "nonfinite": 0, "elements": 128} for slot in range(4, 28)]
    metrics = ([{"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 0}
                for _ in range(4)] + metrics)
    metrics.append({"failures": 1, "different_words": 0, "nonfinite": 1, "elements": 128}
                   if trial_index == 0 else
                   {"failures": 0, "different_words": 0, "nonfinite": 0, "elements": 0})
    return metrics


def build_report(*, normal_primary_collect: bool = True,
                 normal_tail_collect: bool = True,
                 disabled_primary_collect: bool = False) -> dict:
    state = point_state()

    def snapshot() -> dict[str, int]:
        return dict(state)

    initial = snapshot()

    def request(index: int, policy: str, kind: str, *, tail: bool,
                collect: bool, speed_ns: int) -> dict:
        points = [snapshot()]
        # Six non-overlapping phase boundaries. Give each phase measurable
        # wall and host allocation deltas, and place an ordinary collection in
        # terminal_wait when requested.
        increments = (10, 20, 30, speed_ns, 40, 50)
        allocations = (1, 2, 3, 4, 5, 6)
        for phase_index, (elapsed, allocated) in enumerate(zip(increments, allocations)):
            state["time_ns"] += elapsed
            state["allocated_bytes"] += allocated
            state["bytes_since_gc"] += allocated
            if phase_index == 3 and collect:
                state["gc_no"] += 1
                state["collection_count"] += 1
                state["collection_total_ns"] += 100
                state["collection_max_ns"] = max(state["collection_max_ns"], 100)
                state["rendezvous_pause_count"] += 1
                state["rendezvous_pause_total_ns"] += 60
                state["rendezvous_pause_max_ns"] = max(state["rendezvous_pause_max_ns"], 60)
                state["stopped_interval_count"] += 1
                state["stopped_interval_total_ns"] += 25
                state["stopped_interval_max_ns"] = max(state["stopped_interval_max_ns"], 25)
            points.append(snapshot())
        phase_points = {
            "policy_enter": (0, 1), "admission_poison": (1, 2),
            "encode_body_and_comparators": (2, 3), "terminal_wait": (3, 4),
            "policy_exit": (4, 5), "metrics_release": (5, 6),
            "critical": (1, 4), "total": (0, 6),
        }
        return {
            "index": index, "policy": policy, "kind": "tail" if tail else kind,
            "phases": {name: delta(points[first], points[last])
                       for name, (first, last) in phase_points.items()},
            "body_encode_seconds": 1e-3 + index * 1e-5,
            "comparator_encode_seconds": 1e-5 + index * 1e-7,
            "gpu_elapsed_seconds": speed_ns / 1e9,
            "named_buffer_allocations": 0,
            "terminal_wait_count": 1,
            "preterminal_readbacks": 0,
            "comparator_readbacks_after_wait": 1,
            "comparator_readback_bytes": 464,
            "raw_points": points,
            "capture_metrics": metric_array(index // 2) if kind == "correctness" and not tail else None,
        }

    trials = []
    trial_policies = list(POLICIES) + ["normal", "scoped_disabled", "scoped_disabled", "normal"]
    trial_kinds = ["correctness"] * 2 + ["warmup"] * 4
    trial_policies.extend(TIMED_PATTERN * 8)
    trial_kinds.extend(["timed"] * 64)
    for index, (policy, kind) in enumerate(zip(trial_policies, trial_kinds)):
        started = snapshot()
        primary_collect = ((normal_primary_collect and policy == "normal" and index % 3 == 0) or
                           (disabled_primary_collect and policy == "scoped_disabled" and index % 3 == 0))
        primary = request(index * 2, policy, kind, tail=False, collect=primary_collect,
                          speed_ns=500 if policy == "scoped_disabled" else 700 + index)
        tail_collect = normal_tail_collect and index % 4 == 0
        tail = request(index * 2 + 1, "normal", "tail", tail=True,
                       collect=tail_collect, speed_ns=100)
        finished = snapshot()
        trials.append({"index": index, "policy": policy, "kind": kind,
                       "inclusive": delta(started, finished),
                       "primary": primary, "tail": tail})
    final = snapshot()

    return {
        "schema": "cogni_ml.pixelumm_gc_probe/v1",
        "comparison": {"atol": 2e-5, "rtol": 2e-5, "capture_order": list(CAPTURE_ORDER)},
        "observation": {
            "normal_tail_after_every_primary": True,
            "forced_collections_or_artificial_pressure_in_gpu_run": False,
        },
        "validation": {
            "all_numerical_metrics_passed": True,
            "positive_nan_control_passed": True,
            "one_terminal_wait_and_packed_readback_per_request": True,
            "zero_named_encode_allocations_and_preterminal_readbacks": True,
            "gc_enabled_after_every_scope": True,
            "observer_events_paired": True,
            "full_generation_executed": False,
            "production_promoted": False,
        },
        "schedule": {
            "correctness_trials": 2, "warmup_trials": 4, "timed_trials": 64,
            "native_requests": 140,
            "timed_order": "ABBA/BAAB repeated eight times; A=normal B=scoped_disabled",
            "trials": trials,
        },
        "initial_point": initial,
        "final_point": final,
        "observed_loop": delta(initial, final),
    }


class PixelUMMGCReportSummarySpec(unittest.TestCase):
    def test_summarizes_balanced_pairs_phase_deltas_and_metric_different_words(self):
        report = build_report()
        summary = summarize(report, hashlib.sha256(b"report").hexdigest())
        self.assertEqual(summary["fixed_schedule"]["samples_per_arm"], 32)
        self.assertEqual(summary["paired_wall"]["primary_only"]["pairs"], 32)
        self.assertEqual(summary["paired_wall"]["primary_only"]["scoped_disabled_wins"], 32)
        self.assertEqual(summary["paired_wall"]["inclusive_with_normal_tail"]["pairs"], 32)
        self.assertEqual(summary["paired_wall"]["inclusive_with_normal_tail"]["scoped_disabled_wins"], 32)
        normal = summary["arms"]["normal"]["timed_primary"]
        self.assertEqual(normal["samples"], 32)
        self.assertEqual(normal["component_timings"]["body_encode_seconds"]["count"], 32)
        self.assertEqual(normal["component_timings"]["comparator_encode_seconds"]["count"], 32)
        self.assertEqual(normal["component_timings"]["gpu_chain_span_seconds"]["count"], 32)
        normal_inclusive = summary["arms"]["normal"]["timed_inclusive_with_normal_tail"]
        self.assertGreater(normal_inclusive["component_timings"]["body_encode_seconds"]["median"],
                           normal["component_timings"]["body_encode_seconds"]["median"])
        drift = summary["process_drift"]["component_timing_drift"]["arms"]["normal"]["primary_only"]
        for name in ("body_encode_seconds", "comparator_encode_seconds", "gpu_chain_span_seconds"):
            self.assertGreater(drift[name]["late_median_seconds"], drift[name]["early_median_seconds"])
            self.assertEqual(drift[name]["status"], "computed")
        self.assertGreater(normal["phase_timings"]["terminal_wait"]["host_allocated_bytes"]["sum"], 0)
        self.assertGreater(normal["phase_timings"]["terminal_wait"]["gc_observations"]
                           ["boehm_collection_count"]["sum"], 0)
        self.assertEqual(summary["correctness_capture_metrics"]["metrics_per_correctness_primary"], 29)
        self.assertEqual(summary["correctness_capture_metrics"]["correctness_primaries"][0]
                         ["metrics"][4]["different_words"], 1)
        self.assertEqual(summary["observer"]["all_loop"]["gc_observations"]
                         ["boehm_collection_count"], report["observed_loop"]["collections"])
        self.assertEqual(summary["observer"]["final_point_maxima_ns"]["collection_envelope"],
                         report["final_point"]["collection_max_ns"])

    def test_zero_normal_primary_collection_is_separate_from_inclusive_tail(self):
        report = build_report(normal_primary_collect=False, normal_tail_collect=True)
        summary = summarize(report)
        diagnostics = summary["pause_diagnostics"]
        self.assertFalse(diagnostics["normal_primary"]["diagnostic"])
        self.assertEqual(diagnostics["normal_primary"]["status"], "non_diagnostic_zero_collections")
        self.assertTrue(diagnostics["normal_primary_with_normal_tail"]["diagnostic"])
        self.assertGreater(diagnostics["normal_primary_with_normal_tail"]["collection_count"], 0)

        no_gc = summarize(build_report(normal_primary_collect=False, normal_tail_collect=False))
        self.assertFalse(no_gc["pause_diagnostics"]["normal_primary"]["diagnostic"])
        self.assertFalse(no_gc["pause_diagnostics"]["normal_primary_with_normal_tail"]["diagnostic"])

    def test_rejects_missing_phase_fields_nonfinite_values_and_changed_tolerance(self):
        malformed = build_report()
        del malformed["schedule"]["trials"][6]["primary"]["phases"]["encode_body_and_comparators"]
        with self.assertRaisesRegex(ValueError, "incomplete or unexpected phases"):
            summarize(malformed)

        malformed = build_report()
        malformed["schedule"]["trials"][6]["primary"]["gpu_elapsed_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid trial 6 primary gpu_elapsed_seconds"):
            summarize(malformed)

        malformed = build_report()
        malformed["comparison"]["rtol"] = 2.00001e-5
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            summarize(malformed)

    def test_validates_inclusive_windows_against_requests_and_outer_loop(self):
        malformed = build_report()
        malformed["schedule"]["trials"][0]["inclusive"].pop("seconds")
        with self.assertRaisesRegex(ValueError, "incomplete or unexpected delta fields"):
            summarize(malformed)

        malformed = build_report()
        malformed["schedule"]["trials"][0]["inclusive"]["collections"] = 0
        with self.assertRaisesRegex(ValueError, "smaller than disjoint primary and tail totals"):
            summarize(malformed)

        malformed = build_report()
        malformed["schedule"]["trials"][0]["inclusive"]["collections"] += 1
        with self.assertRaisesRegex(ValueError, "all-loop collections is smaller"):
            summarize(malformed)

    def test_rejects_nonzero_disabled_critical_gc_readbacks_and_metric_failures(self):
        with self.assertRaisesRegex(ValueError, "collected while scoped GC was disabled"):
            summarize(build_report(disabled_primary_collect=True))

        malformed = build_report()
        malformed["schedule"]["trials"][0]["primary"]["terminal_wait_count"] = 0
        with self.assertRaisesRegex(ValueError, "terminal_wait_count == 1"):
            summarize(malformed)

        malformed = build_report()
        malformed["schedule"]["trials"][0]["primary"]["capture_metrics"][5]["failures"] = 1
        with self.assertRaisesRegex(ValueError, "numerical metric 5 failed"):
            summarize(malformed)

    def test_rejects_incomplete_schedule_and_unclear_second_positive_control(self):
        malformed = build_report()
        malformed["schedule"]["trials"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete fixed GC trial schedule"):
            summarize(malformed)

        malformed = build_report()
        malformed["schedule"]["trials"][1]["primary"]["capture_metrics"][28]["nonfinite"] = 1
        malformed["schedule"]["trials"][1]["primary"]["capture_metrics"][28]["elements"] = 128
        with self.assertRaisesRegex(ValueError, "positive-control slot is not clear"):
            summarize(malformed)

    def test_cli_writes_fresh_summary_with_source_report_sha256(self):
        with tempfile.TemporaryDirectory(prefix="pixelumm-gc-summary-", dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            report_path = root / "report.json"
            summary_path = root / "summary.json"
            report_path.write_text(json.dumps(build_report()), encoding="utf-8")
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main([str(report_path), "--output", str(summary_path)]), 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["report_sha256"], digest)
            self.assertIn(f"report_sha256={digest}", stdout.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main([str(report_path), "--output", str(summary_path)])


if __name__ == "__main__":
    unittest.main()
