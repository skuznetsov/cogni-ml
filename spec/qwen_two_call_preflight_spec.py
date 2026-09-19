import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

SPEC = importlib.util.spec_from_file_location(
    "preflight", Path(__file__).resolve().parents[1] / "scripts/qwen_two_call_preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def report(pct=70, physical_bytes=64 * 1024**3):
    return (f"The system has {physical_bytes} (4194304 pages with a page size of 16384).\n"
            f"System-wide memory free percentage: {pct}%\n")


class PreflightTest(unittest.TestCase):
    def test_boundary_and_observed_pressure_stop(self):
        for pct, expected in ((30, False), (61, False), (66, False), (69, False),
                              (70, True), (74, True), (75, True), (79, True), (80, True), (100, True)):
            with self.subTest(pct=pct):
                result = preflight.assess(report(pct))
                self.assertEqual(result['memory_preflight_pass'], expected)
                self.assertEqual(result['runtime_floor_pct'], 30)
                self.assertEqual(result['required_initial_pct'], 70)
                self.assertTrue(result['runtime_guard_still_required'])

    def test_other_capacity_is_not_certified(self):
        for gib in (32, 96, 128):
            with self.subTest(gib=gib), self.assertRaises(ValueError):
                preflight.assess(report(100, gib * 1024**3))

    def test_malformed_missing_duplicate_and_out_of_range_fail_closed(self):
        cases = ('', report(-1), report(101), report('nan'), report(80.5),
                 report() + report(), report().splitlines()[0],
                 report().splitlines()[1])
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                preflight.assess(text)

    def test_cli_only_queries_pressure_and_returns_admission_status(self):
        for pct, code in ((61, 75), (69, 75), (70, 0), (74, 0), (75, 0), (79, 0)):
            with self.subTest(pct=pct), patch.object(preflight.subprocess, 'run') as run:
                run.return_value = subprocess.CompletedProcess([], 0, report(pct), '')
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(preflight.main(), code)
                run.assert_called_once_with(['/usr/bin/memory_pressure', '-Q'],
                    check=True, capture_output=True, text=True, timeout=5)

    def test_query_failure_is_not_a_pass(self):
        for error in (OSError('unavailable'), subprocess.TimeoutExpired('memory_pressure', 5),
                      subprocess.CalledProcessError(1, 'memory_pressure')):
            with self.subTest(error=type(error).__name__), patch.object(
                    preflight.subprocess, 'run', side_effect=error), redirect_stdout(io.StringIO()):
                self.assertEqual(preflight.main(), 2)


if __name__ == '__main__':
    unittest.main()
