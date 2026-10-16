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


if __name__ == "__main__":
    unittest.main()
