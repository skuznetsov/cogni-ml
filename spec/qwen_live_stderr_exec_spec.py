"""CPU-only qualification of live trace transport through the safety runner."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
RELAY = ROOT / 'scripts/qwen_live_stderr_exec.py'


class LiveStderrExecTest(unittest.TestCase):
    def test_rejects_existing_log_and_bad_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'trace'
            log.write_text('preserve')
            for args in ([], [str(log), '--', sys.executable, '-c', 'raise SystemExit(99)'],
                         ['relative', '--', sys.executable], [str(Path(tmp)/'new'), '--', 'relative']):
                result = subprocess.run([sys.executable, str(RELAY), *args], capture_output=True)
                self.assertEqual(result.returncode, 64)
            self.assertEqual(log.read_text(), 'preserve')
            self.assertFalse((Path(tmp)/'new').exists())

    def test_rejects_symlink_without_touching_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, link = Path(tmp)/'target', Path(tmp)/'link'
            target.write_text('preserve')
            link.symlink_to(target)
            result = subprocess.run([sys.executable, str(RELAY), str(link), '--', sys.executable,
                                     '-c', 'raise SystemExit(99)'], capture_output=True)
            self.assertEqual(result.returncode, 64)
            self.assertEqual(target.read_text(), 'preserve')

    def test_missing_executable_is_failure_in_live_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'trace'
            result = subprocess.run([sys.executable, str(RELAY), str(log), '--', str(Path(tmp)/'missing')],
                                    capture_output=True)
            self.assertEqual(result.returncode, 126)
            self.assertIn('cannot exec diagnostic target:', log.read_text())

    def test_exec_preserves_pid_arguments_stdout_and_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'trace'
            code = 'import os,sys; print(os.getpid()); print(repr(sys.argv[1:]),file=sys.stderr); sys.exit(7)'
            proc = subprocess.Popen([sys.executable, str(RELAY), str(log), '--', sys.executable,
                                     '-c', code, 'two words', '$literal'], stdout=subprocess.PIPE)
            output, _ = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 7)
            self.assertEqual(int(output), proc.pid)
            self.assertEqual(log.read_text(), "['two words', '$literal']\n")
            self.assertEqual(log.stat().st_mode & 0o777, 0o600)

    def test_marker_is_live_before_guarded_child_can_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, release = Path(tmp)/'trace', Path(tmp)/'release'
            # The child cannot exit normally until the parent observes its marker.
            code = ('import json,os,sys,time; from pathlib import Path; '
                    'print(json.dumps({"pid":os.getpid(),"pgid":os.getpgrp(),"ppid":os.getppid()}),file=sys.stderr,flush=True); '
                    'deadline=time.monotonic()+8\n'
                    'while not Path(sys.argv[1]).exists() and time.monotonic()<deadline: time.sleep(.01)\n'
                    'sys.exit(7 if Path(sys.argv[1]).exists() else 99)')
            env = dict(os.environ, RUN_SAFE_PASSTHROUGH_STDIO='1', COGNI_RUN_SAFE_MIN_FREE_PCT='30',
                       COGNI_RUN_SAFE_REQUIRE_QUIET='0', COGNI_RUN_SAFE_WAIT_QUIET_SEC='0')
            proc = subprocess.Popen([str(ROOT/'scripts/run_safe.sh'), sys.executable, '12', '128',
                                     str(RELAY), str(log), '--', sys.executable, '-c', code, str(release)],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic()+6
                while not (log.exists() and log.stat().st_size) and time.monotonic()<deadline:
                    self.assertIsNone(proc.poll(), 'runner finished before live marker')
                    time.sleep(.01)
                self.assertTrue(log.exists() and log.stat().st_size, 'marker not live')
                marker = json.loads(log.read_text())
                self.assertIsNone(proc.poll())
                self.assertEqual(marker['ppid'], proc.pid)
                self.assertEqual(marker['pid'], marker['pgid'])
            finally:
                release.touch()
                stdout, stderr = proc.communicate(timeout=18)
            self.assertEqual(proc.returncode, 7, (stdout, stderr))
            self.assertIn(b'[EXIT: 7]', stderr)


if __name__ == '__main__':
    unittest.main()
