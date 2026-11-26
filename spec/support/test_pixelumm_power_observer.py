"""CPU-only tests for the external PixelUMM powermetrics observer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "support" / "pixelumm_power_observer.py"
SPEC = importlib.util.spec_from_file_location("pixelumm_power_observer", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
observer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer
SPEC.loader.exec_module(observer)


REAL_FORMAT_SAMPLE = """*** Sampled system activity (Sat Oct 31 12:58:23 2026 -0400) (1003.78ms elapsed) ***

**** Thermal pressure ****

Current pressure level: Nominal

**** GPU usage ****

GPU HW active frequency: 649 MHz
GPU HW active residency:  93.11% (444 MHz:  37% 612 MHz:  27% 808 MHz: 9.4% 968 MHz:  15% 1110 MHz: 5.1% 1236 MHz:   0% 1338 MHz:   0% 1398 MHz:   0%)
GPU Power: 2312 mW


"""


class PixelUMMPowerObserverTests(unittest.TestCase):
    def test_parses_real_powermetrics_text_and_utc_window(self) -> None:
        samples = observer.parse_powermetrics_text(REAL_FORMAT_SAMPLE)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(
            sample["timestamp_header"],
            "*** Sampled system activity (Sat Oct 31 12:58:23 2026 -0400) (1003.78ms elapsed) ***",
        )
        self.assertEqual(sample["timestamp_local"], "Sat Oct 31 12:58:23 2026 -0400")
        self.assertEqual(sample["timestamp_utc"], "2026-10-31T16:58:23Z")
        self.assertEqual(sample["sample_window_start_utc"], "2026-10-31T16:58:21.996220Z")
        self.assertEqual(sample["elapsed_ms"], 1003.78)
        self.assertEqual(sample["gpu_active_frequency_mhz"], 649.0)
        self.assertEqual(sample["sample_window_end_utc"], "2026-10-31T16:58:23Z")
        self.assertEqual(sample["gpu_active_residency_pct"], 93.11)
        self.assertEqual(sample["gpu_power_mw"], 2312.0)
        self.assertEqual(sample["thermal_pressure"], "Nominal")
        self.assertEqual(sample["gpu_frequency_states_mhz"], [
            {"frequency_mhz": 444.0, "residency_pct": 37.0},
            {"frequency_mhz": 612.0, "residency_pct": 27.0},
            {"frequency_mhz": 808.0, "residency_pct": 9.4},
            {"frequency_mhz": 968.0, "residency_pct": 15.0},
            {"frequency_mhz": 1110.0, "residency_pct": 5.1},
            {"frequency_mhz": 1236.0, "residency_pct": 0.0},
            {"frequency_mhz": 1338.0, "residency_pct": 0.0},
            {"frequency_mhz": 1398.0, "residency_pct": 0.0},
        ])
        self.assertTrue(sample["valid"])

        summary = observer.summarize_samples(samples, expected_sample_count=1)
        self.assertEqual(summary["valid_sample_count"], 1)
        self.assertEqual(summary["valid_sample_fraction"], 1.0)
        self.assertTrue(summary["duration_complete"])

    def test_missing_metrics_remain_null_and_invalidate_coverage(self) -> None:
        incomplete = """*** Sampled system activity (Sat Oct 31 12:58:23 2026 -0400) (2000.00ms elapsed) ***
GPU HW active frequency: 700 MHz
"""
        samples = observer.parse_powermetrics_text(incomplete)
        self.assertEqual(len(samples), 1)
        self.assertIsNone(samples[0]["gpu_active_residency_pct"])
        self.assertIsNone(samples[0]["gpu_power_mw"])
        self.assertIsNone(samples[0]["thermal_pressure"])
        self.assertEqual(
            samples[0]["missing_fields"],
            ["gpu_active_residency_pct", "gpu_power_mw", "thermal_pressure"],
        )
        summary = observer.summarize_samples(samples, expected_sample_count=1)
        self.assertEqual(summary["valid_sample_count"], 0)
        self.assertEqual(summary["valid_sample_fraction"], 0.0)
        self.assertFalse(summary["all_observed_samples_valid"])

    def test_explicit_zero_is_preserved_as_a_measurement(self) -> None:
        zero_sample = """*** Sampled system activity (Sat Oct 31 12:58:23 2026 -0400) (2000.00ms elapsed) ***
Current pressure level: Nominal
GPU HW active frequency: 0 MHz
GPU HW active residency: 0.00% (444 MHz: 100%)
GPU Power: 0 mW
"""
        sample = observer.parse_powermetrics_text(zero_sample)[0]
        self.assertEqual(sample["gpu_active_frequency_mhz"], 0.0)
        self.assertEqual(sample["gpu_active_residency_pct"], 0.0)
        self.assertEqual(sample["gpu_power_mw"], 0.0)
        self.assertTrue(sample["valid"])

    def test_malformed_sample_is_not_promoted_to_valid_zero_data(self) -> None:
        malformed = """*** Sampled system activity (Sat Oct 31 12:58:23 2026 -0400) (2000.00ms elapsed) ***
Current pressure level: Nominal
GPU HW active frequency: unavailable
GPU HW active residency: 0.00%
GPU Power: 0 mW
"""
        samples = observer.parse_powermetrics_text(malformed)
        self.assertEqual(len(samples), 1)
        self.assertIsNone(samples[0]["gpu_active_frequency_mhz"])
        self.assertIn("gpu_active_frequency_mhz", samples[0]["missing_fields"])
        self.assertFalse(samples[0]["valid"])
        summary = observer.summarize_samples(samples, expected_sample_count=1)
        self.assertEqual(summary["valid_sample_fraction"], 0.0)
        self.assertFalse(summary["all_observed_samples_valid"])

    def test_utc_range_summary_excludes_boundary_intervals(self) -> None:
        base = observer.parse_powermetrics_text(REAL_FORMAT_SAMPLE)[0]
        samples = []
        for index, (window_start, window_end) in enumerate([
            (99_000, 101_000),
            (101_000, 103_000),
            (103_000, 105_000),
            (115_000, 117_000),
            (117_000, 119_000),
            (119_000, 121_000),
        ]):
            samples.append({
                **base,
                "sample_index": index,
                "sample_window_start_unix_ms": window_start,
                "sample_window_end_unix_ms": window_end,
                "elapsed_ms": 2_000.0,
            })

        summary = observer.summarize_utc_ranges(samples, {"pass": (100_000, 120_000)})["pass"]
        self.assertEqual(summary["overlapping_samples"], 6)
        self.assertEqual(summary["boundary_excluded_samples"], 4)
        self.assertEqual(summary["eligible_samples"], 2)
        self.assertEqual(summary["valid_samples"], 2)
        self.assertEqual(summary["valid_sample_fraction"], 1.0)

    def test_signal_handler_only_records_stop_request(self) -> None:
        previous_signal = observer._TERMINATION_SIGNAL
        observer._TERMINATION_SIGNAL = None
        try:
            with mock.patch.object(observer.os, "killpg") as killpg:
                observer._signal_child(signal.SIGTERM, None)
            self.assertEqual(observer._TERMINATION_SIGNAL, signal.SIGTERM)
            killpg.assert_not_called()
        finally:
            observer._TERMINATION_SIGNAL = previous_signal

    def test_privileged_stop_targets_only_the_owned_process_group(self) -> None:
        args = ["/usr/bin/sudo", "-n", "/bin/kill", "-INT", "--", "-4321"]
        completed = subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
        with mock.patch.object(observer.subprocess, "run", return_value=completed) as run:
            result = observer._send_privileged_group_signal(
                "/usr/bin/sudo", 4321, signal.SIGINT
            )
        self.assertTrue(result["sent"])
        self.assertEqual(result["command"], args)
        run.assert_called_once_with(
            args,
            capture_output=True,
            timeout=observer.STOP_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )

    def test_group_presence_requires_an_unambiguous_pgrep_absence(self) -> None:
        args = ["/usr/bin/sudo", "-n", "/usr/bin/pgrep", "-g", "4321"]
        absent = subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"")
        with mock.patch.object(observer.subprocess, "run", return_value=absent) as run:
            result = observer._check_privileged_group_presence("/usr/bin/sudo", 4321)
        self.assertEqual(result["state"], "absent")
        run.assert_called_once_with(
            args,
            stdout=observer.subprocess.DEVNULL,
            stderr=observer.subprocess.PIPE,
            timeout=observer.STOP_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )

        denied = subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"a password is required")
        with mock.patch.object(observer.subprocess, "run", return_value=denied):
            result = observer._check_privileged_group_presence("/usr/bin/sudo", 4321)
        self.assertEqual(result["state"], "unknown")

        failed = subprocess.CompletedProcess(args, 3, stdout=b"", stderr=b"pgrep failed")
        with mock.patch.object(observer.subprocess, "run", return_value=failed):
            result = observer._check_privileged_group_presence("/usr/bin/sudo", 4321)
        self.assertEqual(result["state"], "unknown")

    def test_stop_escalates_when_tracked_child_exits_but_owned_group_remains(self) -> None:
        class ExitedChild:
            returncode = -signal.SIGINT

            def communicate(self, timeout=None):
                return None, b""

        child = ExitedChild()
        presence = {"state": "present", "stderr_excerpt": None}
        absent = {"state": "absent", "stderr_excerpt": None}
        checks = [presence, presence, presence, absent]
        sent = []

        def send(_sudo_path, _group_id, signum):
            sent.append(observer._signal_name(signum))
            return {"signal": observer._signal_name(signum), "sent": True}

        with mock.patch.object(observer, "_check_privileged_group_presence", side_effect=checks), \
                mock.patch.object(observer, "_send_privileged_group_signal", side_effect=send):
            result = observer._stop_sampler_group(child, 4321, "/usr/bin/sudo")

        self.assertTrue(result["stopped"])
        self.assertEqual(sent, ["INT", "TERM", "KILL"])
        self.assertEqual(result["group_checks"][-1]["state"], "absent")

    def test_unknown_owned_group_presence_never_counts_as_stopped(self) -> None:
        class ExitedChild:
            returncode = -signal.SIGINT

            def communicate(self, timeout=None):
                return None, b""

        child = ExitedChild()
        unknown = {"state": "unknown", "stderr_excerpt": "permission denied"}
        sent = []

        def send(_sudo_path, _group_id, signum):
            sent.append(observer._signal_name(signum))
            return {"signal": observer._signal_name(signum), "sent": False}

        with mock.patch.object(observer, "_check_privileged_group_presence", return_value=unknown), \
                mock.patch.object(observer, "_send_privileged_group_signal", side_effect=send):
            result = observer._stop_sampler_group(child, 4321, "/usr/bin/sudo")

        self.assertFalse(result["stopped"])
        self.assertEqual(sent, ["INT", "TERM", "KILL"])
        self.assertEqual(len(result["group_checks"]), 4)

    def test_stop_escalation_reports_failure_without_process_exit(self) -> None:
        class StuckChild:
            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired("powermetrics", timeout, stderr=b"still active")

        completed = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"denied")
        with mock.patch.object(observer, "_check_privileged_group_presence",
                               return_value={"state": "present", "stderr_excerpt": None}), \
                mock.patch.object(observer.subprocess, "run", return_value=completed) as run:
            result = observer._stop_sampler_group(StuckChild(), 4321, "/usr/bin/sudo")

        self.assertFalse(result["stopped"])
        self.assertEqual([attempt["signal"] for attempt in result["attempts"]], ["INT", "TERM", "KILL"])
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(call.args[0][-1] == "-4321" for call in run.call_args_list))

    def test_collect_does_not_finalize_when_child_exits_but_group_remains(self) -> None:
        class ExitedChild:
            pid = 4321
            returncode = -signal.SIGINT

            def communicate(self, timeout=None):
                return None, b""

        child = ExitedChild()
        present = {"state": "present", "stderr_excerpt": None}
        started_events = []
        sent_signals = []

        def send(command, **_kwargs):
            sent_signals.append(command[3])
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        with tempfile.TemporaryDirectory() as temp_root:
            output_dir = Path(temp_root) / "capture"
            with (
                mock.patch.object(observer.platform, "system", return_value="Darwin"),
                mock.patch.object(observer, "validate_output_dir", return_value=output_dir),
                mock.patch.object(observer.shutil, "which", side_effect=["/usr/bin/sudo", "/usr/bin/powermetrics"]),
                mock.patch.object(Path, "is_file", return_value=True),
                mock.patch.object(observer.subprocess, "Popen", return_value=child),
                mock.patch.object(observer.subprocess, "run", side_effect=send),
                mock.patch.object(observer, "_check_privileged_group_presence", return_value=present),
                mock.patch.object(observer, "_TERMINATION_SIGNAL", signal.SIGTERM),
            ):
                summary = observer.collect(
                    output_dir,
                    2,
                    on_started=started_events.append,
                )

        self.assertEqual(summary["status"], "stop_failed")
        self.assertFalse(summary["collector"]["sampler_stopped"])
        self.assertFalse(summary["raw_capture"]["finalized"])
        self.assertEqual(sent_signals, ["-INT", "-TERM", "-KILL"])

    def test_stop_escalation_stops_after_first_privileged_signal(self) -> None:
        class ExitingChild:
            returncode = None

            def communicate(self, timeout=None):
                returncode = self.returncode
                if returncode is None:
                    raise subprocess.TimeoutExpired("powermetrics", timeout)
                return None, b""

        child = ExitingChild()
        args = ["/usr/bin/sudo", "-n", "/bin/kill", "-INT", "--", "-4321"]

        def signal_child(command, **_kwargs):
            self.assertEqual(command, args)
            child.returncode = -signal.SIGINT
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        with mock.patch.object(observer, "_check_privileged_group_presence",
                               side_effect=[
                                   {"state": "present", "stderr_excerpt": None},
                                   {"state": "absent", "stderr_excerpt": None},
                               ]), \
                mock.patch.object(observer.subprocess, "run", side_effect=signal_child) as run:
            result = observer._stop_sampler_group(child, 4321, "/usr/bin/sudo")

        self.assertTrue(result["stopped"])
        self.assertEqual([attempt["signal"] for attempt in result["attempts"]], ["INT"])
        run.assert_called_once()

    def test_collect_reports_stop_failure_and_owned_group_without_real_process(self) -> None:
        class StuckChild:
            pid = 4321
            returncode = None

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired("powermetrics", timeout, stderr=b"still active")

        child = StuckChild()
        denied = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"denied")
        started_events = []
        with tempfile.TemporaryDirectory() as temp_root:
            output_dir = Path(temp_root) / "capture"
            with (
                mock.patch.object(observer.platform, "system", return_value="Darwin"),
                mock.patch.object(observer, "validate_output_dir", return_value=output_dir),
                mock.patch.object(observer.shutil, "which", side_effect=["/usr/bin/sudo", "/usr/bin/powermetrics"]),
                mock.patch.object(Path, "is_file", return_value=True),
                mock.patch.object(observer.subprocess, "Popen", return_value=child),
                mock.patch.object(observer.subprocess, "run", return_value=denied) as run,
                mock.patch.object(observer, "_check_privileged_group_presence",
                                  return_value={"state": "present", "stderr_excerpt": None}),
                mock.patch.object(observer, "_TERMINATION_SIGNAL", signal.SIGTERM),
            ):
                summary = observer.collect(
                    output_dir,
                    2,
                    on_started=started_events.append,
                )

        self.assertEqual(summary["status"], "stop_failed")
        self.assertFalse(summary["collector"]["sampler_stopped"])
        self.assertFalse(summary["raw_capture"]["finalized"])
        self.assertEqual(summary["collector"]["observer_pid"], observer.os.getpid())
        self.assertEqual(summary["collector"]["sampler_process_group_id"], 4321)
        self.assertEqual(started_events[0]["observer_pid"], observer.os.getpid())
        self.assertEqual(started_events[0]["sampler_process_group_id"], 4321)
        self.assertEqual(run.call_count, 3)


if __name__ == "__main__":
    unittest.main()
