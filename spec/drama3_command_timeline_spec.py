#!/usr/bin/env python3
"""Focused tests for the offline DRAMA-3 command timeline analyzer."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/drama3/analyze_command_timeline.py"
SPEC = importlib.util.spec_from_file_location("analyze_command_timeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def command(
    index=0,
    *,
    wait_start=2.0,
    wait_end=8.0,
    gpu_start=3.0,
    gpu_end=7.0,
    gpu_timing_valid=True,
    **overrides,
):
    row = {
        "event": "drama3_gpu_command",
        "index": index,
        "completion_status": 0,
        "gpu_seconds": gpu_end - gpu_start if gpu_timing_valid else None,
        "gpu_timing_valid": gpu_timing_valid,
        "host_before_commit_s": 1.0,
        "host_after_commit_s": 1.1,
        "host_before_wait_s": wait_start,
        "host_after_wait_s": wait_end,
        "gpu_start_s": gpu_start if gpu_timing_valid else None,
        "gpu_end_s": gpu_end if gpu_timing_valid else None,
    }
    row.update(overrides)
    return row


RESOURCE_FIELDS = (
    "user_ns",
    "system_ns",
    "pageins",
    "disk_read_bytes",
    "faults",
    "cow_faults",
    "resident_bytes",
    "footprint_bytes",
)


def resource_snapshot(begin, end, base, **overrides):
    row = {
        "sample_begin_s": begin,
        "sample_end_s": end,
        **{name: base + offset for offset, name in enumerate(RESOURCE_FIELDS)},
    }
    row.update(overrides)
    return row


def resources():
    return {
        "encoding_begin": resource_snapshot(1.2, 1.3, 100),
        "before_commit": resource_snapshot(1.8, 1.9, 160),
        "after_commit": resource_snapshot(1.95, 2.0, 170),
        "after_wait": resource_snapshot(7.8, 7.9, 230),
    }


class CommandTimelineSpec(unittest.TestCase):
    def test_wait_partition_before_inside_and_after_gpu(self):
        cases = [
            # Wait starts before execution and spans its interval.
            (command(wait_start=2.0, wait_end=8.0, gpu_start=3.0, gpu_end=7.0), (1.0, 4.0, 1.0)),
            # Wait begins during execution and includes its completion tail.
            (command(index=1, wait_start=4.0, wait_end=8.0, gpu_start=3.0, gpu_end=7.0), (0.0, 3.0, 1.0)),
            # The existing wait call begins after GPU execution has ended.
            (command(index=2, wait_start=8.0, wait_end=9.0, gpu_start=3.0, gpu_end=7.0), (0.0, 0.0, 1.0)),
        ]

        report = analyzer.analyze_events([row for row, _ in cases])
        by_index = {row["index"]: row for row in report["commands"]}
        for source, expected in cases:
            actual = by_index[source["index"]]["wait_partition_seconds"]
            self.assertEqual(actual["pre_gpu"], expected[0])
            self.assertEqual(actual["gpu_overlap"], expected[1])
            self.assertEqual(actual["post_gpu"], expected[2])
            self.assertAlmostEqual(sum(actual.values()), by_index[source["index"]]["wait_duration_seconds"])

        totals = report["totals"]
        partitioned = totals["wait_partition_seconds"]
        self.assertAlmostEqual(
            sum(partitioned[key] for key in ("pre_gpu", "gpu_overlap", "post_gpu")),
            partitioned["partitioned_wait"],
        )
        self.assertAlmostEqual(
            partitioned["partitioned_wait"] + partitioned["unpartitioned_wait"],
            totals["wait_duration_seconds"],
        )

    def test_missing_optional_gpu_timing_is_counted_and_wait_is_unpartitioned(self):
        row = command(
            wait_start=2.0,
            wait_end=3.0,
            gpu_timing_valid=False,
            gpu_seconds=None,
            gpu_start_s=None,
            gpu_end_s=None,
        )
        report = analyzer.analyze_events([row])

        self.assertEqual(report["counts"]["gpu_timing_invalid_commands"], 1)
        self.assertEqual(report["counts"]["gpu_timing_missing_commands"], 0)
        self.assertEqual(report["counts"]["gpu_bounds_missing_commands"], 1)
        item = report["commands"][0]
        self.assertIsNone(item["gpu_execution_seconds"])
        self.assertIsNone(item["wait_partition_seconds"])
        self.assertAlmostEqual(item["unpartitioned_wait_seconds"], 1.0)
        self.assertAlmostEqual(report["totals"]["wait_partition_seconds"]["unpartitioned_wait"], 1.0)

    def test_missing_validity_flag_is_distinguished_from_invalid_gpu_timing(self):
        row = command(gpu_timing_valid=False)
        del row["gpu_timing_valid"]
        report = analyzer.analyze_events([row])
        self.assertEqual(report["counts"]["gpu_timing_missing_commands"], 1)
        self.assertEqual(report["counts"]["gpu_timing_invalid_commands"], 0)

    def test_duplicate_command_index_fails_closed(self):
        with self.assertRaisesRegex(analyzer.TimelineError, "duplicate command index 0"):
            analyzer.analyze_events([command(), command()])

    def test_nonfinite_valid_gpu_timestamp_fails_closed(self):
        with self.assertRaisesRegex(analyzer.TimelineError, "gpu_start_s.*finite"):
            analyzer.analyze_events([command(gpu_start_s=math.inf)])

    def test_redundant_gpu_duration_must_match_valid_timestamps(self):
        with self.assertRaisesRegex(analyzer.TimelineError, "gpu_seconds disagrees"):
            analyzer.analyze_events([command(gpu_seconds=99.0)])

    def test_empty_selected_request_fails_closed(self):
        with self.assertRaisesRegex(analyzer.TimelineError, "no command timeline events"):
            analyzer.analyze_events([])

    def test_invalid_host_time_order_fails_closed(self):
        with self.assertRaisesRegex(analyzer.TimelineError, "host timestamp order"):
            analyzer.analyze_events([command(host_before_wait_s=1.05)])

    def test_invalid_gpu_bounds_fails_closed_when_flag_claims_valid(self):
        with self.assertRaisesRegex(analyzer.TimelineError, "GPU timing bounds"):
            analyzer.analyze_events([command(gpu_start=5.0, gpu_end=4.0)])

    def test_phase_ranges_are_explicit_inclusive_and_unassigned_is_retained(self):
        report = analyzer.analyze_events(
            [command(index=10), command(index=11)],
            phases=[("prefill", 10, 10)],
        )
        self.assertEqual([row["phase"] for row in report["commands"]], ["prefill", "unassigned"])
        self.assertEqual(report["phases"]["prefill"]["command_count"], 1)
        self.assertEqual(report["phases"]["unassigned"]["command_count"], 1)

    def test_cpu_driver_scheduling_interval_is_separate_from_gpu_time(self):
        row = command(
            kernel_timing_valid=True,
            kernel_start_s=1.25,
            kernel_end_s=1.50,
        )
        report = analyzer.analyze_events([row])
        item = report["commands"][0]
        self.assertAlmostEqual(item["driver_scheduling_seconds"], 0.25)
        self.assertAlmostEqual(item["gpu_execution_seconds"], 4.0)
        self.assertIsNone(item.get("schedule_to_gpu_start_seconds"))
        self.assertEqual(report["counts"]["driver_scheduling_valid_commands"], 1)

    def test_unavailable_and_legacy_kernel_fields_are_not_invalid(self):
        unavailable = command(
            kernel_timing_valid=False,
            kernel_start_s=None,
            kernel_end_s=None,
        )
        legacy = command(index=1)
        report = analyzer.analyze_events([unavailable, legacy])
        self.assertEqual(report["counts"]["driver_scheduling_missing_commands"], 2)
        self.assertEqual(report["counts"]["driver_scheduling_valid_commands"], 0)
        self.assertIsNone(report["commands"][0]["driver_scheduling_seconds"])
        self.assertIsNone(report["commands"][1]["driver_scheduling_seconds"])

    def test_valid_kernel_flag_rejects_missing_nonfinite_or_reversed_times(self):
        for overrides in (
            {"kernel_start_s": None, "kernel_end_s": 1.5},
            {"kernel_start_s": math.inf, "kernel_end_s": 1.5},
            {"kernel_start_s": 1.5, "kernel_end_s": 1.4},
            {"kernel_start_s": 0.0, "kernel_end_s": 1.5},
        ):
            with self.subTest(overrides=overrides):
                row = command(kernel_timing_valid=True, **overrides)
                with self.assertRaises(analyzer.TimelineError):
                    analyzer.analyze_events([row])

    def test_unavailable_kernel_flag_cannot_carry_timestamp_values(self):
        row = command(
            kernel_timing_valid=False,
            kernel_start_s=1.2,
            kernel_end_s=1.3,
        )
        with self.assertRaisesRegex(analyzer.TimelineError, "unavailable kernel timing"):
            analyzer.analyze_events([row])

    def test_resource_counter_deltas_are_attributed_to_separate_snapshot_windows(self):
        report = analyzer.analyze_events([command(resources=resources())])
        item = report["commands"][0]
        windows = item["resource_windows"]

        encoding = windows["encoding_to_before_commit"]
        self.assertTrue(encoding["valid"])
        self.assertEqual(encoding["sample_window_seconds"], {"lower": 0.5, "upper": 0.7})
        self.assertEqual(encoding["counters"]["user_ns"], {"delta": 60, "status": "valid"})
        self.assertEqual(encoding["gauges"]["resident_bytes"], {"change": 60, "status": "valid"})

        commit = windows["before_commit_to_after_commit"]
        self.assertEqual(commit["counters"]["user_ns"]["delta"], 10)
        submitted = windows["after_commit_to_after_wait"]
        self.assertEqual(submitted["counters"]["user_ns"]["delta"], 60)
        self.assertEqual(report["totals"]["resource_counter_windows"]["encoding_to_before_commit"]["counters"]["user_ns"]["delta_distribution"], {"min": 60, "mean": 60.0, "max": 60})

    def test_overlapping_command_windows_are_distributions_not_process_totals(self):
        # Different in-flight commands may observe the same process activity.
        report = analyzer.analyze_events([
            command(index=0, resources=resources()),
            command(index=1, resources=resources()),
        ])
        summary = report["totals"]["resource_counter_windows"]["after_commit_to_after_wait"]
        self.assertIn("nonadditive", summary["scope"])
        self.assertEqual(summary["valid_window_count"], 2)
        self.assertEqual(summary["counters"]["user_ns"]["delta_distribution"], {"min": 60, "mean": 60.0, "max": 60})
        self.assertNotIn("delta_sum", summary["counters"]["user_ns"])
        self.assertNotIn("change_sum", summary["gauges"]["resident_bytes"])
        self.assertAlmostEqual(summary["sample_window_seconds"]["lower"]["mean"], 5.8)

    def test_missing_resource_fields_are_unavailable_without_dropping_other_deltas(self):
        snapshots = resources()
        snapshots["before_commit"]["pageins"] = None
        report = analyzer.analyze_events([command(resources=snapshots)])
        window = report["commands"][0]["resource_windows"]["encoding_to_before_commit"]
        self.assertEqual(window["counters"]["pageins"], {"delta": None, "status": "unavailable"})
        self.assertEqual(window["counters"]["user_ns"]["status"], "valid")

    def test_negative_task_event_value_makes_its_counter_group_unavailable(self):
        snapshots = resources()
        snapshots["before_commit"]["faults"] = -1
        report = analyzer.analyze_events([command(resources=snapshots)])
        counters = report["commands"][0]["resource_windows"]["encoding_to_before_commit"]["counters"]
        for name in ("pageins", "faults", "cow_faults"):
            self.assertEqual(counters[name], {"delta": None, "status": "unavailable"})
        self.assertEqual(counters["user_ns"]["status"], "valid")

    def test_counter_reset_is_local_and_gauge_changes_may_be_negative(self):
        snapshots = resources()
        snapshots["before_commit"]["user_ns"] = 50
        snapshots["before_commit"]["resident_bytes"] = 90
        report = analyzer.analyze_events([command(resources=snapshots)])
        window = report["commands"][0]["resource_windows"]["encoding_to_before_commit"]
        self.assertEqual(window["counters"]["user_ns"], {"delta": None, "status": "counter_reset"})
        self.assertEqual(window["gauges"]["resident_bytes"], {"change": -16, "status": "valid"})

    def test_overlapping_sample_windows_are_invalid_only_for_that_window(self):
        snapshots = resources()
        snapshots["after_commit"] = resource_snapshot(1.85, 2.0, 170)
        report = analyzer.analyze_events([command(resources=snapshots)])
        windows = report["commands"][0]["resource_windows"]
        self.assertTrue(windows["encoding_to_before_commit"]["valid"])
        self.assertFalse(windows["before_commit_to_after_commit"]["valid"])
        self.assertEqual(
            windows["before_commit_to_after_commit"]["counters"]["user_ns"],
            {"delta": None, "status": "invalid_window"},
        )
        self.assertTrue(windows["after_commit_to_after_wait"]["valid"])

    def test_invalid_sampling_clock_is_local_and_never_emits_counter_delta(self):
        snapshots = resources()
        snapshots["after_commit"]["sample_end_s"] = math.nan
        report = analyzer.analyze_events([command(resources=snapshots)])
        windows = report["commands"][0]["resource_windows"]
        self.assertFalse(windows["before_commit_to_after_commit"]["valid"])
        self.assertEqual(
            windows["before_commit_to_after_commit"]["invalid_reason"],
            "invalid_sample_timing:after_commit",
        )
        self.assertEqual(
            windows["before_commit_to_after_commit"]["counters"]["user_ns"],
            {"delta": None, "status": "invalid_window"},
        )
        self.assertTrue(windows["encoding_to_before_commit"]["valid"])
        self.assertFalse(windows["after_commit_to_after_wait"]["valid"])

    def test_malformed_optional_resource_object_does_not_break_gpu_timeline(self):
        report = analyzer.analyze_events([command(resources=["bad schema"])])
        item = report["commands"][0]
        self.assertFalse(item["resource_snapshots_missing"])
        self.assertTrue(all(not window["valid"] for window in item["resource_windows"].values()))
        self.assertEqual(item["gpu_execution_seconds"], 4.0)

    def test_legacy_events_without_resource_snapshots_remain_analyzable(self):
        report = analyzer.analyze_events([command()])
        item = report["commands"][0]
        self.assertTrue(item["resource_snapshots_missing"])
        self.assertIsNone(item["resource_windows"])
        self.assertEqual(report["counts"]["resource_snapshot_missing_commands"], 1)


if __name__ == "__main__":
    unittest.main()
