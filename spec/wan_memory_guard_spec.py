"""CPU-only policy, parser, and process-group tests for the Wan guard."""

import importlib.util
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples/support/wan_memory_guard.py"
SPEC = importlib.util.spec_from_file_location("wan_memory_guard", MODULE_PATH)
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)

MIB = 1024 * 1024
GIB = 1024**3
PHYSICAL = 64 * GIB


def sample(*, pressure=80, available=40 * GIB, footprint=0, swap=10 * MIB,
           live_processes=1, speculative=None, file_backed=None):
    return guard.Telemetry(
        pressure_free_pct=pressure,
        free_spec_bytes=available,
        physical_bytes=PHYSICAL,
        swap_used_bytes=swap,
        group_footprint_bytes=footprint,
        group_live_processes=live_processes,
        speculative_bytes=speculative,
        file_backed_bytes=file_backed,
    )


class ParserSpec(unittest.TestCase):
    def test_memory_pressure_requires_one_valid_percentage(self):
        self.assertEqual(guard.parse_memory_pressure(
            "The system has 123 (1 pages with a page size of 4096).\n"
            "System-wide memory free percentage: 41%\n"), 41)
        for text in ("", "System-wide memory free percentage: 101%\n",
                     "System-wide memory free percentage: 10%\n"
                     "System-wide memory free percentage: 20%\n"):
            with self.subTest(text=text), self.assertRaises(guard.TelemetryError):
                guard.parse_memory_pressure(text)

    def test_vm_stat_counts_file_cache_without_double_counting_speculative_pages(self):
        report = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                  "Pages free: 100.\nPages active: 9999.\n"
                  "Pages inactive: 9999.\nPages purgeable: 9999.\n"
                  "Pages speculative: 25.\nFile-backed pages: 200.\n"
                  "Anonymous pages: 9999.\n")
        counts = guard.parse_vm_stat(report)
        self.assertEqual(counts.free_spec_bytes, 125 * 16384)
        self.assertEqual(counts.free_file_cache_bytes, 300 * 16384)
        self.assertEqual(counts.speculative_bytes, 25 * 16384)
        self.assertEqual(counts.file_backed_bytes, 200 * 16384)
        for text in ("", "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                     "Pages free: 2.\n", "page size of 0 bytes\nPages free: 1.\n"
                     "Pages speculative: 1.\nFile-backed pages: 2.\n",
                     report.replace("File-backed pages: 200.\n", ""),
                     report + "File-backed pages: 200.\n",
                     report.replace("File-backed pages: 200.", "File-backed pages: 24."),
                     report.replace("File-backed pages: 200.", "File-backed pages: -1.")):
            with self.subTest(text=text), self.assertRaises(guard.TelemetryError):
                guard.parse_vm_stat(text, require_file_cache=True)

    def test_strict_vm_stat_parser_does_not_require_optional_cache_telemetry(self):
        report = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                  "Pages free: 100.\nPages speculative: 25.\n")
        for suffix in ("", "File-backed pages: unknown.\n", "File-backed pages: 24.\n",
                       "File-backed pages: 100.\nFile-backed pages: 100.\n"):
            with self.subTest(suffix=suffix):
                counts = guard.parse_vm_stat(report + suffix)
                self.assertEqual(counts.free_spec_bytes, 125 * 16384)
                self.assertIsNone(counts.file_backed_bytes)
                with self.assertRaises(guard.TelemetryError):
                    _ = counts.free_file_cache_bytes

    def test_strict_collector_preserves_missing_cache_counter_compatibility(self):
        report = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                  "Pages free: 1500000.\nPages speculative: 10000.\n")
        with patch.object(guard.sys, "platform", "darwin"), \
                patch.object(guard, "_run_text", side_effect=[
                    "System-wide memory free percentage: 80%\n", report,
                    str(PHYSICAL) + "\n", "used = 10.00M"]):
            reading = guard.collect_telemetry(None)
        self.assertIsNone(guard.assess_startup(reading, 10 * GIB))
        self.assertIn("file-cache counters missing", guard.assess_startup(
            reading, 10 * GIB, memory_accounting="file-cache"))

    def test_swap_usage_parses_binary_mib_and_fails_closed(self):
        self.assertEqual(guard.parse_swap_used(
            "total = 32768.00M  used = 25.50M  free = 32742.50M (encrypted)"),
            int(25.5 * MIB))
        self.assertEqual(guard.parse_swap_used("total = 2.00G used = 1.25G free = 0.75G"),
                         int(1.25 * GIB))
        with self.assertRaises(guard.TelemetryError):
            guard.parse_swap_used("swap usage unavailable")

    def test_process_group_listing_rejects_zero_identity_and_missing_leader(self):
        with patch.object(guard, "_run_text", return_value="0 0 I\n"):
            with self.assertRaises(guard.TelemetryError):
                guard.process_group_members(101)
        with patch.object(guard, "_run_text", return_value="202 101 S\n"):
            with self.assertRaisesRegex(guard.TelemetryError, "leader missing"):
                guard._group_usage(101)

    @unittest.skipUnless(sys.platform == "darwin", "macOS libproc ABI")
    def test_rusage_v2_ctypes_layout_matches_native_footprint_field(self):
        self.assertEqual(guard._RUsageInfoV2.ri_phys_footprint.offset, 72)
        self.assertEqual(ctypes.sizeof(guard._RUsageInfoV2), 160)
        self.assertGreaterEqual(guard._proc_pid_rusage(os.getpid()), 0)

    @unittest.skipUnless(sys.platform == "darwin", "Darwin libc waitid ABI")
    def test_libc_waitid_fallback_observes_exit_without_reaping(self):
        child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"],
                                 start_new_session=True)
        try:
            deadline = time.monotonic() + 3
            with patch.object(guard.os, "waitid", None, create=True):
                while time.monotonic() < deadline:
                    status = guard._peek_child_exit(child.pid)
                    if status is not None:
                        break
                    time.sleep(0.01)
                else:
                    self.fail("libc waitid did not observe child exit")
            self.assertEqual(status, 7)
            self.assertIsNone(child.returncode, "WNOWAIT must preserve the child identity")
            self.assertEqual(child.wait(), 7)
        finally:
            if child.returncode is None:
                child.kill()
                child.wait()


class PolicySpec(unittest.TestCase):
    def test_file_cache_policy_admits_cache_heavy_host_but_default_stays_strict(self):
        cached = sample(available=10 * GIB, speculative=2 * GIB, file_backed=20 * GIB)
        self.assertIn("startup headroom", guard.assess_startup(cached, 10 * GIB))
        self.assertIsNone(guard.assess_startup(cached, 10 * GIB,
                                             memory_accounting="file-cache"))
        self.assertIsNone(guard.assess_runtime(cached, 10 * GIB, 2 * MIB, 10 * MIB,
                                             memory_accounting="file-cache"))

    def test_speculative_overlap_cannot_create_false_startup_admission(self):
        # Printed free=2 GiB; speculative=4 GiB is already in file-backed=12 GiB.
        # Correct headroom=14 GiB, not the naively summed 18 GiB. The requested
        # 8-GiB budget plus the 15%-of-RAM policy floor is 17.6 GiB.
        cached = sample(available=6 * GIB, speculative=4 * GIB, file_backed=12 * GIB)
        reason = guard.assess_startup(cached, 8 * GIB, memory_accounting="file-cache")
        self.assertIn("startup headroom estimate (free + file-backed)", reason)
        self.assertIn("15%-of-RAM policy floor", reason)

    def test_file_cache_policy_requires_consistent_complete_integer_counters(self):
        invalid_samples = [
            sample(),
            sample(speculative=0),
            sample(file_backed=1),
            sample(speculative=True, file_backed=10 * GIB),
            sample(speculative=-1, file_backed=10 * GIB),
            sample(available=1, speculative=2, file_backed=3),
            sample(speculative=2, file_backed=1),
            sample(speculative=0, file_backed=PHYSICAL),
        ]
        for invalid in invalid_samples:
            with self.subTest(sample=invalid):
                self.assertIsNotNone(guard.assess_startup(
                    invalid, 1, memory_accounting="file-cache"))
                self.assertIsNotNone(guard.assess_runtime(
                    invalid, 1, 0, 0, memory_accounting="file-cache"))

    def test_file_cache_policy_preserves_reserve_pressure_footprint_and_swap_stops(self):
        base = dict(available=4 * GIB, speculative=2 * GIB, file_backed=20 * GIB)
        for overrides, expected in [
            ({"file_backed": 7 * GIB}, "15%"),
            ({"pressure": 14}, "pressure"),
            ({"footprint": 11 * GIB}, "process-tree footprint"),
            ({"swap": 13 * MIB}, "swap growth"),
        ]:
            with self.subTest(expected=expected):
                self.assertIn(expected, guard.assess_runtime(
                    sample(**(base | overrides)), 10 * GIB, 2 * MIB, 10 * MIB,
                    memory_accounting="file-cache"))
        self.assertIn("pressure", guard.assess_startup(
            sample(**base, pressure=14), 10 * GIB, memory_accounting="file-cache"))

    def test_high_pressure_percentage_does_not_hide_low_physical_headroom(self):
        # 90% memory_pressure free is insufficient when vm_stat has <15% physical
        # headroom plus the requested 8-GiB process-tree budget.
        low_headroom = sample(pressure=90, available=17 * GIB)
        reason = guard.assess_startup(low_headroom, 8 * GIB)
        self.assertIn("startup headroom", reason)

    def test_startup_requires_both_pressure_and_budgeted_physical_headroom(self):
        self.assertIsNone(guard.assess_startup(sample(pressure=15, available=20 * GIB), 8 * GIB))
        self.assertIn("pressure", guard.assess_startup(sample(pressure=14), 8 * GIB))

    def test_impossible_free_speculative_bytes_fail_closed(self):
        impossible = guard.Telemetry(80, PHYSICAL + 1, PHYSICAL, 0, 0, 0)
        self.assertIn("exceed physical", guard.assess_startup(impossible, 1))
        self.assertIn("exceed physical", guard.assess_runtime(impossible, 1, 0, 0))

    def test_process_footprint_budget_is_not_rss(self):
        # The policy receives the summed ri_phys_footprint; RSS is not a field.
        too_large = sample(footprint=11 * GIB)
        self.assertIn("process-tree footprint", guard.assess_runtime(
            too_large, 10 * GIB, 0, 10 * MIB))

    def test_runtime_checks_floor_and_swap_growth(self):
        self.assertIn("free/speculative", guard.assess_runtime(
            sample(available=9 * GIB), 10 * GIB, 2 * MIB, 10 * MIB))
        self.assertIn("swap growth", guard.assess_runtime(
            sample(swap=13 * MIB), 10 * GIB, 2 * MIB, 10 * MIB))
        self.assertIsNone(guard.assess_runtime(
            sample(swap=12 * MIB), 10 * GIB, 2 * MIB, 10 * MIB))


class ContainmentSpec(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_with_provider(self, command, provider, *, timeout=5.0, name="guard.log",
                          memory_accounting="free-speculative"):
        return guard.run_guarded(
            command,
            max_footprint_bytes=10 * GIB,
            max_swap_growth_bytes=2 * MIB,
            timeout_sec=timeout,
            log_path=self.root / name,
            telemetry_provider=provider,
            poll_interval_sec=0.02,
            terminate_grace_sec=0.15,
            memory_accounting=memory_accounting,
        )

    def test_cache_aware_launch_logs_raw_counters_and_selected_estimate(self):
        marker = self.root / "cache-launch"
        command = [sys.executable, "-c",
                   "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ok')", str(marker)]
        result = self.run_with_provider(command, lambda _group, _deadline: sample(
            available=10 * GIB, speculative=2 * GIB, file_backed=20 * GIB,
            live_processes=0), memory_accounting="file-cache", name="cache-launch.log")
        self.assertEqual(result, 0)
        self.assertEqual(marker.read_text(), "ok")
        records = [json.loads(line) for line in (self.root / "cache-launch.log").read_text().splitlines()]
        self.assertEqual(records[0]["memory_accounting"], "file-cache")
        readings = [record for record in records if record["event"] == "sample"]
        self.assertTrue(readings)
        self.assertTrue(all(record["headroom_bytes"] == 28 * GIB for record in readings))
        self.assertTrue(all(record["free_spec_bytes"] == 10 * GIB for record in readings))
        self.assertTrue(all(record["file_backed_bytes"] == 20 * GIB for record in readings))

    def test_cache_floor_violation_stops_owned_child(self):
        pid_file = self.root / "cache-floor.pid"
        command = [sys.executable, "-c",
                   "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)",
                   str(pid_file)]
        calls = 0

        def provider(group_id, _deadline):
            nonlocal calls
            calls += 1
            return sample(available=4 * GIB, speculative=2 * GIB,
                          file_backed=(20 if group_id is None or calls < 6 else 7) * GIB)

        result = self.run_with_provider(command, provider, name="cache-floor.log",
                                        memory_accounting="file-cache")
        self.assertEqual(result, guard.EXIT_GUARD)
        self.assertTrue(pid_file.exists())
        self.assertTrue(guard.process_is_gone_or_zombie(int(pid_file.read_text())))
        self.assertIn('"reason":"file-cache headroom estimate fell below the 15%-of-RAM policy floor"',
                      (self.root / "cache-floor.log").read_text())

    def test_terminated_waitable_leader_does_not_require_getpgid(self):
        child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"],
                                 start_new_session=True)
        try:
            deadline = time.monotonic() + 3
            while guard._peek_child_exit(child.pid) is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            # Darwin returns ESRCH for the unreaped zombie. Simulate that
            # behavior on every platform while the real waitable PID is held.
            with patch.object(guard.os, "getpgid", side_effect=ProcessLookupError):
                guard._terminate_owned_group(child, 0.0)
            self.assertIsNone(child.returncode)
            self.assertEqual(child.wait(), 7)
        finally:
            if child.returncode is None:
                child.kill()
                child.wait()

    def test_reaped_or_unwaitable_leader_never_signals_a_group(self):
        for returncode in (None, 0):
            with self.subTest(returncode=returncode):
                child = SimpleNamespace(pid=101, returncode=returncode)
                with patch.object(guard, "_peek_child_exit",
                                  side_effect=guard.TelemetryError("not our waitable child")), \
                        patch.object(guard.os, "killpg") as killpg:
                    with self.assertRaises((RuntimeError, guard.TelemetryError)):
                        guard._signal_owned_group(child, signal.SIGKILL)
                    killpg.assert_not_called()

    def test_live_leader_with_changed_pgid_never_signals_a_group(self):
        child = SimpleNamespace(pid=101, returncode=None)
        with patch.object(guard, "_peek_child_exit", return_value=None), \
                patch.object(guard.os, "getpgid", return_value=202), \
                patch.object(guard.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "original process group"):
                guard._signal_owned_group(child, signal.SIGKILL)
            killpg.assert_not_called()

    def test_exited_leader_still_contains_term_resistant_descendant(self):
        self.check_exited_leader_cleanup(process_listing_unavailable=False)

    def test_exited_leader_cleanup_survives_process_listing_failure(self):
        self.check_exited_leader_cleanup(process_listing_unavailable=True)

    def check_exited_leader_cleanup(self, *, process_listing_unavailable):
        ready = self.root / "descendant.pid"
        grandchild_code = (
            "import os,pathlib,signal,sys,time; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
        )
        leader_code = (
            "import pathlib,subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
            "p=pathlib.Path(sys.argv[2]); "
            "exec('while not p.exists(): time.sleep(.01)')"
        )
        sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
                                    start_new_session=True)
        child = subprocess.Popen([sys.executable, "-c", leader_code, grandchild_code, str(ready)],
                                 start_new_session=True)
        try:
            deadline = time.monotonic() + 3
            while guard._peek_child_exit(child.pid) is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            descendant = int(ready.read_text())
            with patch.object(guard.os, "getpgid", side_effect=ProcessLookupError):
                if process_listing_unavailable:
                    with patch.object(guard, "process_group_members",
                                      side_effect=guard.TelemetryError("injected ps outage")):
                        with self.assertRaises(guard.TelemetryError):
                            guard._terminate_owned_group(child, 0.05)
                else:
                    guard._terminate_owned_group(child, 0.05)
            deadline = time.monotonic() + 3
            while not guard.process_is_gone_or_zombie(descendant):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertIsNone(sentinel.poll(), "cleanup must leave another session alone")
            self.assertIsNone(child.returncode, "do not reap before group cleanup")
            self.assertEqual(child.wait(), 0)
        finally:
            if child.returncode is None:
                # This test created and still pins this original session group.
                guard._terminate_owned_group(child, 0.05)
                child.wait(timeout=3)
            sentinel.terminate()
            sentinel.wait()

    def test_cleanup_identity_failure_is_logged_without_a_success_record(self):
        command = [sys.executable, "-c", "import time;time.sleep(30)"]
        signal_group = guard._signal_owned_group
        calls = 0

        def fail_once(child, signum):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise guard.TelemetryError("injected identity loss")
            return signal_group(child, signum)

        with patch.object(guard, "_signal_owned_group", side_effect=fail_once):
            result = self.run_with_provider(command, lambda _group, _deadline: sample(),
                                            timeout=0.1, name="identity-loss.log")
        self.assertEqual(result, guard.EXIT_TELEMETRY)
        log = (self.root / "identity-loss.log").read_text()
        self.assertIn('"event":"cleanup_failed"', log)
        self.assertNotIn('"event":"finished"', log)

    def test_persistent_identity_loss_returns_without_waiting_or_signaling(self):
        child = SimpleNamespace(pid=101, returncode=None,
                                wait=Mock(side_effect=AssertionError("must preserve identity pin")))
        with patch.object(guard.subprocess, "Popen", return_value=child), \
                patch.object(guard, "_peek_child_exit",
                             side_effect=guard.TelemetryError("persistent identity loss")), \
                patch.object(guard.os, "killpg") as killpg:
            result = self.run_with_provider(["not-launched"], lambda _group, _deadline: sample(),
                                            name="persistent-identity.log")
        self.assertEqual(result, guard.EXIT_TELEMETRY)
        killpg.assert_not_called()
        child.wait.assert_not_called()
        log = (self.root / "persistent-identity.log").read_text()
        self.assertIn('"event":"cleanup_incomplete"', log)
        self.assertIn('"phase":"group_termination"', log)
        self.assertNotIn('"event":"finished"', log)

    def test_uninterruptible_reap_never_uses_an_unbounded_wait(self):
        child = SimpleNamespace(pid=101, returncode=None,
                                wait=Mock(side_effect=subprocess.TimeoutExpired("not-launched", 2)))
        with patch.object(guard.subprocess, "Popen", return_value=child), \
                patch.object(guard, "_terminate_owned_group"), \
                patch.object(guard, "_signal_owned_group"):
            result = self.run_with_provider(["not-launched"],
                                            lambda group, _deadline: sample(
                                                available=40 * GIB if group is None else 9 * GIB),
                                            name="uninterruptible.log")
        self.assertEqual(result, guard.EXIT_TELEMETRY)
        self.assertGreater(child.wait.call_count, 0)
        self.assertTrue(all(call.kwargs.get("timeout", 0) > 0
                            for call in child.wait.call_args_list))
        log = (self.root / "uninterruptible.log").read_text()
        self.assertIn('"event":"cleanup_incomplete"', log)
        self.assertIn('"phase":"leader_reap"', log)
        self.assertNotIn('"event":"finished"', log)

    def test_live_descendant_after_sigkill_is_not_successful_cleanup(self):
        child = SimpleNamespace(pid=101, returncode=None)

        def reap_leader(*, timeout):
            child.returncode = 0
            return 0

        child.wait = Mock(side_effect=reap_leader)
        with patch.object(guard.subprocess, "Popen", return_value=child), \
                patch.object(guard, "process_group_members",
                             return_value=[guard.GroupProcess(202, "U")]), \
                patch.object(guard, "_peek_child_exit", return_value=0), \
                patch.object(guard, "_signal_owned_group") as signal_group:
            started = time.monotonic()
            result = self.run_with_provider(["not-launched"],
                                            lambda group, _deadline: sample(
                                                available=40 * GIB if group is None else 9 * GIB),
                                            name="live-after-kill.log")
        self.assertEqual(result, guard.EXIT_TELEMETRY)
        child.wait.assert_not_called()
        self.assertLess(time.monotonic() - started, 3)
        self.assertIn(signal.SIGKILL, [call.args[1] for call in signal_group.call_args_list])
        log = (self.root / "live-after-kill.log").read_text()
        self.assertIn('"event":"cleanup_incomplete"', log)
        self.assertIn('"phase":"group_termination"', log)
        self.assertNotIn('"event":"finished"', log)

    def test_low_physical_headroom_blocks_command_before_launch(self):
        marker = self.root / "should-not-exist"
        command = [sys.executable, "-c",
                   "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('launched')",
                   str(marker)]
        calls = []

        def provider(group_id, _deadline):
            calls.append(group_id)
            return sample(pressure=90, available=17 * GIB)

        result = self.run_with_provider(command, provider, name="admission.log")
        self.assertEqual(result, guard.EXIT_ADMISSION)
        self.assertEqual(calls, [None])
        self.assertFalse(marker.exists())

    def test_missing_runtime_metric_fails_closed_and_kills_only_child_group(self):
        pid_file = self.root / "child.pid"
        command = [sys.executable, "-c",
                   "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid())); time.sleep(30)",
                   str(pid_file)]
        calls = 0

        def provider(group_id, _deadline):
            nonlocal calls
            calls += 1
            if group_id is not None and calls >= 6:
                raise guard.TelemetryError("injected missing sample")
            return sample()

        result = self.run_with_provider(command, provider, name="missing.log")
        self.assertEqual(result, guard.EXIT_TELEMETRY)
        self.assertTrue(pid_file.exists())
        pid = int(pid_file.read_text())
        self.assertTrue(guard.process_is_gone_or_zombie(pid))
        log = (self.root / "missing.log").read_text()
        self.assertIn('"reason":"telemetry unavailable"', log)

    def test_timeout_kills_its_own_command(self):
        pid_file = self.root / "timeout.pid"
        command = [sys.executable, "-c",
                   "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)",
                   str(pid_file)]
        result = self.run_with_provider(command, lambda _group, _deadline: sample(), timeout=0.12,
                                        name="timeout.log")
        self.assertEqual(result, guard.EXIT_TIMEOUT)
        self.assertTrue(pid_file.exists())
        self.assertTrue(guard.process_is_gone_or_zombie(int(pid_file.read_text())))

    def test_normal_child_exit_code_is_preserved(self):
        result = self.run_with_provider(
            [sys.executable, "-c", "raise SystemExit(7)"],
            lambda _group, _deadline: sample(live_processes=0), name="normal.log")
        self.assertEqual(result, 7)
        self.assertIn('"returncode":7', (self.root / "normal.log").read_text())

    def test_wrapper_sigterm_cleans_child_and_grandchild_group(self):
        pid_file = self.root / "tree.pids"
        grandchild_code = "import time; time.sleep(30)"
        child_code = (
            "import pathlib,subprocess,sys,time; "
            "g=subprocess.Popen([sys.executable,'-c',sys.argv[2]]); "
            "pathlib.Path(sys.argv[1]).write_text(f'{__import__(\"os\").getpid()} {g.pid}'); "
            "time.sleep(30)"
        )
        helper_code = (
            "import importlib.util,json,sys; "
            "s=importlib.util.spec_from_file_location('guard',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); "
            "sample=lambda pgid,deadline: m.Telemetry(80,40*1024**3,64*1024**3,10*1024**2,0); "
            "rc=m.run_guarded([sys.executable,'-c',sys.argv[3],sys.argv[4],sys.argv[5]], "
            "max_footprint_bytes=10*1024**3,max_swap_growth_bytes=2*1024**2, "
            "timeout_sec=8,log_path=sys.argv[6],telemetry_provider=sample, "
            "poll_interval_sec=.02,terminate_grace_sec=.15); print(json.dumps({'rc':rc}),flush=True)"
        )
        helper = subprocess.Popen(
            [sys.executable, "-c", helper_code, str(MODULE_PATH), str(pid_file), child_code,
             str(pid_file), grandchild_code, str(self.root / "signal.log")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 4
            while not pid_file.exists() and time.monotonic() < deadline:
                self.assertIsNone(helper.poll(), "guard exited before launching child tree")
                time.sleep(0.01)
            self.assertTrue(pid_file.exists(), "child tree did not publish PIDs")
            child_pid, grandchild_pid = map(int, pid_file.read_text().split())
            helper.send_signal(signal.SIGTERM)
            stdout, stderr = helper.communicate(timeout=5)
            self.assertEqual(helper.returncode, 0, stderr)
            self.assertEqual(json.loads(stdout.strip())["rc"], 143)
            self.assertTrue(guard.process_is_gone_or_zombie(child_pid))
            self.assertTrue(guard.process_is_gone_or_zombie(grandchild_pid))
            log = (self.root / "signal.log").read_text()
            self.assertIn('"event":"limits"', log)
            self.assertIn("between-poll spikes", log)
            self.assertIn("setsid", log)
            self.assertIn("setpgid", log)
        finally:
            if helper.poll() is None:
                helper.kill()
                helper.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main()
