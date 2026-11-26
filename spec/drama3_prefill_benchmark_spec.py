"""Fail-closed controls for DRAMA-3 batched-prefill timing analysis."""
import json
import unittest

from scripts.drama3.analyze_prefill_benchmark import qualify


SCHEDULE = (1, 4, 1, 4, 4, 1, 4, 1, 1, 4)
COMPONENTS = ("moe", "backbone")


def fixture(rich_timing=True):
    observations = []
    log_lines = ["[PREFLIGHT] unrelated startup diagnostic"]
    command_index = 0
    absolute_gpu_time = 100.0
    for component in COMPONENTS:
        for iteration, chunk in enumerate(SCHEDULE):
            command_count = (17 + chunk - 1) // chunk
            wall_seconds = 1.0 + iteration / 10
            log_lines.append(json.dumps({
                "event": "prefill_benchmark_begin", "component": component,
                "iteration": iteration, "chunk": chunk,
            }))
            for _ in range(command_count):
                event = {"event": "drama3_gpu_command", "index": command_index,
                         "gpu_seconds": 0.001}
                if rich_timing:
                    event.update({"completion_status": 0, "gpu_timing_valid": True,
                                  "gpu_start_s": absolute_gpu_time,
                                  "gpu_end_s": absolute_gpu_time + 0.001})
                    absolute_gpu_time += 0.003
                log_lines.append(json.dumps(event))
                command_index += 1
            log_lines.append(json.dumps({
                "event": "prefill_benchmark_end", "component": component,
                "iteration": iteration, "chunk": chunk, "seconds": wall_seconds,
            }))
            observations.append({
                "component": component, "iteration": iteration, "chunk": chunk,
                "seconds": wall_seconds,
                "graph": {"submitted": command_count, "completed": command_count,
                          "completion_waits": command_count, "pending": 0},
                "readbacks": {"control": 0, "diagnostic": 0, "terminal": 0},
            })
    benchmark = {"rows": 17, "chunk": 4, "persistent_context": True,
                 "input_sha256": "a" * 64, "observations": observations}
    return benchmark, "\n".join(log_lines) + "\n"


class PrefillBenchmarkControls(unittest.TestCase):
    def test_reports_abba_baab_warm_arms_and_keeps_gpu_span_separate(self):
        benchmark, log_text = fixture()
        report = qualify(benchmark, log_text)
        self.assertTrue(report["qualified"])
        self.assertFalse(report["quiet_host_causal_claim"])
        self.assertEqual(report["excluded_primes_per_component"], 2)
        self.assertEqual(report["components"]["moe"]["patterns"], ["ABBA", "BAAB"])
        component = report["components"]["backbone"]
        self.assertEqual(component["arms"]["scalar"]["samples"], 4)
        self.assertEqual(component["arms"]["batched"]["samples"], 4)
        self.assertGreater(component["arms"]["scalar"]["completed_gpu_interval_sum_seconds"]["median"],
                           component["arms"]["batched"]["completed_gpu_interval_sum_seconds"]["median"])
        self.assertGreater(component["arms"]["scalar"]["gpu_span_seconds"]["median"],
                           component["arms"]["scalar"]["completed_gpu_interval_sum_seconds"]["median"])
        self.assertIn("gpu_span_seconds", component["arms"]["scalar"])
        self.assertGreater(component["speedup"]["wall_scalar_over_batched"], 1.0)

    def test_accepts_depth_zero_duration_events_with_implicit_checked_status(self):
        benchmark, log_text = fixture(rich_timing=False)
        report = qualify(benchmark, log_text)
        self.assertEqual(report["components"]["moe"]["arms"]["scalar"]["gpu_span_seconds"], None)
        self.assertIn("completion-checked", report["gpu_timing_evidence"])

    def test_rejects_missing_command_and_invalid_gpu_duration(self):
        benchmark, log_text = fixture()
        lines = log_text.splitlines()
        command_pos = next(i for i, line in enumerate(lines)
                           if '"event": "drama3_gpu_command"' in line)
        del lines[command_pos]
        with self.assertRaisesRegex(ValueError, "command count"):
            qualify(benchmark, "\n".join(lines))

        for value in (0, -1, float("nan"), float("inf"), None):
            benchmark, log_text = fixture()
            lines = log_text.splitlines()
            event = next(json.loads(line) for line in lines
                         if '"event": "drama3_gpu_command"' in line)
            event["gpu_seconds"] = value
            index = next(i for i, line in enumerate(lines)
                         if '"event": "drama3_gpu_command"' in line)
            lines[index] = json.dumps(event, allow_nan=True)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid GPU time"):
                qualify(benchmark, "\n".join(lines))

    def test_rejects_nonzero_completion_status_and_invalid_interval(self):
        for mutation in (
            lambda event: event.__setitem__("completion_status", 1),
            lambda event: event.__setitem__("gpu_timing_valid", False),
            lambda event: event.__setitem__("gpu_end_s", event["gpu_start_s"]),
        ):
            benchmark, log_text = fixture()
            lines = log_text.splitlines()
            index = next(i for i, line in enumerate(lines)
                         if '"event": "drama3_gpu_command"' in line)
            event = json.loads(lines[index])
            mutation(event)
            lines[index] = json.dumps(event)
            with self.assertRaises(ValueError):
                qualify(benchmark, "\n".join(lines))

    def test_rejects_incomplete_completion_or_readback_counters(self):
        benchmark, log_text = fixture()
        benchmark["observations"][2]["graph"]["completed"] -= 1
        with self.assertRaisesRegex(ValueError, "completion counters"):
            qualify(benchmark, log_text)
        benchmark, log_text = fixture()
        benchmark["observations"][2]["graph"]["pending"] = 1
        with self.assertRaisesRegex(ValueError, "pending commands"):
            qualify(benchmark, log_text)
        benchmark, log_text = fixture()
        benchmark["observations"][2]["readbacks"]["diagnostic"] = 4
        with self.assertRaisesRegex(ValueError, "readbacks"):
            qualify(benchmark, log_text)

    def test_rejects_orphan_markers_and_mismatched_marker_identity(self):
        benchmark, log_text = fixture()
        lines = log_text.splitlines()
        del lines[max(i for i, line in enumerate(lines)
                      if '"event": "prefill_benchmark_end"' in line)]
        with self.assertRaisesRegex(ValueError, "unterminated measured scope"):
            qualify(benchmark, "\n".join(lines))

        benchmark, log_text = fixture()
        lines = log_text.splitlines()
        begin_index = next(i for i, line in enumerate(lines)
                           if '"event": "prefill_benchmark_begin"' in line)
        event = json.loads(lines[begin_index])
        event["component"] = "backbone"
        lines[begin_index] = json.dumps(event)
        with self.assertRaisesRegex(ValueError, "scope does not match benchmark observation"):
            qualify(benchmark, "\n".join(lines))

        benchmark, log_text = fixture()
        lines = log_text.splitlines()
        lines.insert(1, json.dumps({"event": "prefill_benchmark_end", "component": "moe",
                                    "iteration": 0, "chunk": 1, "seconds": 1.0}))
        with self.assertRaisesRegex(ValueError, "end marker without begin"):
            qualify(benchmark, "\n".join(lines))

    def test_does_not_count_commands_outside_measured_scope(self):
        benchmark, log_text = fixture()
        outside = json.dumps({"event": "drama3_gpu_command", "index": 999,
                              "gpu_seconds": 1000.0})
        report = qualify(benchmark, outside + "\n" + log_text + "\n" + outside)
        self.assertEqual(report["unscoped_gpu_commands_ignored"], 2)
        self.assertEqual(report["timed_gpu_commands"], sum(
            row["graph"]["submitted"] for row in benchmark["observations"]))

    def test_rejects_changed_schedule_and_nonpersistent_context(self):
        benchmark, log_text = fixture()
        benchmark["observations"][4]["chunk"] = 1
        with self.assertRaisesRegex(ValueError, "changed ABBA/BAAB schedule"):
            qualify(benchmark, log_text)
        benchmark, log_text = fixture()
        benchmark["persistent_context"] = False
        with self.assertRaisesRegex(ValueError, "persistent context"):
            qualify(benchmark, log_text)

    def test_rejects_mixed_absolute_and_duration_only_timing_modes(self):
        benchmark, log_text = fixture()
        lines = log_text.splitlines()
        in_first_scope = False
        for index, line in enumerate(lines):
            if '"event": "prefill_benchmark_begin"' in line:
                in_first_scope = True
            elif '"event": "prefill_benchmark_end"' in line and in_first_scope:
                break
            elif in_first_scope and '"event": "drama3_gpu_command"' in line:
                event = json.loads(line)
                for field in ("completion_status", "gpu_timing_valid", "gpu_start_s", "gpu_end_s"):
                    event.pop(field)
                lines[index] = json.dumps(event)
        with self.assertRaisesRegex(ValueError, "inconsistent absolute GPU timing mode"):
            qualify(benchmark, "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
