#!/usr/bin/env python3
"""Guards for the full-decode capture's privacy and trajectory boundaries."""

import copy
import importlib.util
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from xml.etree import ElementTree as ET


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/drama3"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("capture_decode_profile", SCRIPTS / "capture_decode_profile.py")
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class DecodeCaptureSpec(unittest.TestCase):
    def test_only_allowlisted_environment_and_pinned_inference_controls(self):
        source = {"PATH": "/bin", "TMPDIR": "/external/tmp", "SECRET_TOKEN": "private",
                  "DRAMA3_TOWER_FFN_PAIR_FUSION": "1", "DRAMA3_MEMORY_PROFILE": "1",
                  "COGNI_METAL_COMMAND_TIMEOUT_MS": "180000",
                  "QWEN35_NATIVE_TOKENIZER_OFF": "1"}
        env = capture.environment(source, False)
        self.assertNotIn("SECRET_TOKEN", env)
        self.assertNotIn("QWEN35_NATIVE_TOKENIZER_OFF", env)
        self.assertEqual(env["TMPDIR"], "/external/tmp")
        self.assertEqual(env["COGNI_METAL_COMMAND_TIMEOUT_MS"], "180000")
        self.assertEqual(env["DRAMA3_TOWER_FFN_PAIR_FUSION"], "0")
        self.assertEqual(env["DRAMA3_MEMORY_PROFILE"], "0")
        self.assertEqual(env["DRAMA3_HOST_PROFILE"], "0")
        self.assertEqual(env["DRAMA3_METAL_TIMELINE"], "0")
        timed = capture.environment(source, True)
        self.assertEqual({k for k in env if env[k] != timed[k]}, {"DRAMA3_METAL_TIMELINE"})

    def test_toc_removes_nested_environments_before_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toc.xml"
            path.write_text('<trace-toc><environment>private1</environment><run>'
                            '<info><environment>private2</environment><target>worker</target></info>'
                            '<data><table schema="metal-gpu-intervals"/></data>'
                            '</run></trace-toc>')
            self.assertEqual(capture.sanitize_toc(path), ["metal-gpu-intervals"])
            text = path.read_text()
            self.assertNotIn("private", text)
            self.assertNotIn("environment", text)
            self.assertEqual(ET.parse(path).findtext(".//target"), "worker")

    def test_partial_toc_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toc.xml"
            path.write_text('<trace-toc><environment>private')
            with self.assertRaises(ET.ParseError):
                capture.sanitize_toc(path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".xml.sanitized").exists())

    def test_live_recorder_preserves_owned_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "owned.trace"
            trace.mkdir()
            payload = trace / "recording"
            payload.write_text("still being written")
            recorder = Mock()
            recorder.poll.return_value = None
            self.assertFalse(capture.cleanup_trace(trace, [recorder]))
            self.assertEqual(payload.read_text(), "still being written")

    def test_reaped_recorder_allows_owned_trace_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "owned.trace"
            trace.mkdir()
            recorder = Mock()
            recorder.poll.return_value = 0
            self.assertTrue(capture.cleanup_trace(trace, [recorder]))
            self.assertFalse(trace.exists())

    def test_live_recorder_does_not_confirm_cleanup_before_trace_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "not-created-yet.trace"
            recorder = Mock()
            recorder.poll.return_value = None
            self.assertFalse(capture.cleanup_trace(trace, [recorder]))
            self.assertFalse(trace.exists())

    def test_partial_worker_line_has_a_deadline_and_is_reaped(self):
        child = subprocess.Popen([
            sys.executable, "-c",
            "import sys,time; sys.stdout.write('{'); sys.stdout.flush(); time.sleep(60)",
        ], stdout=subprocess.PIPE, text=True)
        try:
            with self.assertRaises(TimeoutError):
                capture.receive(child, bytearray(), timeout=0.1)
        finally:
            _, forced = capture.stop_process(child, None, 0.01)
            self.assertTrue(forced)
            self.assertIsNotNone(child.poll())
            child.stdout.close()

    def test_buffered_worker_events_are_not_lost(self):
        child = subprocess.Popen([
            sys.executable, "-c",
            'print(\'{"event":"ready"}\\n{"event":"result"}\')',
        ], stdout=subprocess.PIPE, text=True)
        try:
            pending = bytearray()
            self.assertEqual(capture.receive(child, pending)["event"], "ready")
            self.assertEqual(capture.receive(child, pending)["event"], "result")
        finally:
            capture.stop_process(child, None, 1)
            child.stdout.close()

    def test_partial_recorder_start_line_has_a_deadline_and_is_reaped(self):
        child = subprocess.Popen([
            sys.executable, "-c",
            "import sys,time; sys.stdout.write('Starting'); sys.stdout.flush(); time.sleep(60)",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorders = []
            try:
                with patch.object(capture.subprocess, "Popen", return_value=child):
                    with self.assertRaisesRegex(RuntimeError, "recording start"):
                        capture.start_trace(123, root / "template", root / "trace", root,
                                            {}, startup_timeout=0.1, cleanup_grace=0.1,
                                            recorders=recorders)
                self.assertEqual(recorders, [child])
                self.assertIsNotNone(child.poll())
                self.assertIn("Starting", (root / "record-start.log").read_text())
            finally:
                capture.stop_process(child, None, 0.1)
                child.stdout.close()

    def test_invariants_detect_count_read_code_sampler_and_termination_changes(self):
        report = {"resident_allocation_stats": {"allocations": 3, "requested_bytes": 64,
                                              "allocation_ns": 0},
                  "resident_graph": {"submitted": 2, "completed": 2,
                                     "completion_waits": 2, "pending": 0,
                                     "depth": 2, "max_pending": 2, "retained_bytes": 0},
                  "resident_readbacks": {"control": 4, "terminal": 8},
                  "frames": [[1, 2]], "sampler_inputs": {"seed": 1},
                  "stopped_on_eos": True, "truncated": False,
                  "samples": 2, "sample_rate": 44100, "prompt_rows": 29}
        original = capture.trajectory(report)
        for container, key in (("resident_allocation_stats", "allocations"),
                               ("resident_allocation_stats", "requested_bytes"),
                               ("resident_graph", "completion_waits"),
                               ("resident_graph", "depth"),
                               ("resident_graph", "max_pending"),
                               ("resident_graph", "retained_bytes"),
                               ("resident_readbacks", "control"),
                               ("sampler_inputs", "seed")):
            changed = copy.deepcopy(report)
            changed[container][key] += 1
            self.assertNotEqual(capture.trajectory(changed), original)
        for key in ("stopped_on_eos", "truncated", "samples", "sample_rate", "frames"):
            changed = copy.deepcopy(report)
            changed[key] = None
            self.assertNotEqual(capture.trajectory(changed), original)
        report["resident_allocation_stats"]["allocation_ns"] = 999
        self.assertEqual(capture.trajectory(report), original)


if __name__ == "__main__":
    unittest.main()
