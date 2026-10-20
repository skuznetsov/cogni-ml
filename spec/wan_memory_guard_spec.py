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
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples/support/wan_memory_guard.py"
SPEC = importlib.util.spec_from_file_location("wan_memory_guard", MODULE_PATH)
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)

MIB = 1024 * 1024
GIB = 1024**3
PHYSICAL = 64 * GIB


def sample(*, pressure=80, available=40 * GIB, footprint=0, swap=10 * MIB,
           live_processes=1):
    return guard.Telemetry(
        pressure_free_pct=pressure,
        free_spec_bytes=available,
        physical_bytes=PHYSICAL,
        swap_used_bytes=swap,
        group_footprint_bytes=footprint,
        group_live_processes=live_processes,
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

    def test_vm_stat_counts_only_free_and_speculative_pages(self):
        report = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                  "Pages free: 100.\nPages active: 9999.\n"
                  "Pages speculative: 25.\n")
        self.assertEqual(guard.parse_vm_stat(report), 125 * 16384)
        for text in ("", "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                     "Pages free: 2.\n", "page size of 0 bytes\nPages free: 1.\n"
                     "Pages speculative: 1.\n"):
            with self.subTest(text=text), self.assertRaises(guard.TelemetryError):
                guard.parse_vm_stat(text)

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

    def run_with_provider(self, command, provider, *, timeout=5.0, name="guard.log"):
        return guard.run_guarded(
            command,
            max_footprint_bytes=10 * GIB,
            max_swap_growth_bytes=2 * MIB,
            timeout_sec=timeout,
            log_path=self.root / name,
            telemetry_provider=provider,
            poll_interval_sec=0.02,
            terminate_grace_sec=0.15,
        )

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
