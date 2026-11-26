"""Regression checks for exact upstream MoE return-local capture."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "drama3" / "replay_mlx_moe.py"
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUNNER.parent))
SPEC = importlib.util.spec_from_file_location("drama3_replay_mlx_moe_capture", RUNNER)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class UpstreamMoeCaptureSpec(unittest.TestCase):
    def test_shared_weighted_stage_comes_from_return_local(self):
        capture = replay.UpstreamCapture.__new__(replay.UpstreamCapture)
        capture.stage_refs = {}
        capture.active_target = True
        capture._store_once = lambda name, value: capture.stage_refs.setdefault(name, value)

        raw_shared = object()
        weighted_shared = object()
        official_result = object()

        # First assignment has completed: this is the raw shared-expert output.
        capture._trace_target_sparse(
            SimpleNamespace(f_locals={"shared_y": raw_shared}), "line", None
        )
        self.assertIs(capture.stage_refs["shared_expert_output"], raw_shared)
        self.assertNotIn("shared_weighted_output", capture.stage_refs)

        # A line event before the gate assignment still exposes the old local.
        capture._trace_target_sparse(
            SimpleNamespace(f_locals={"shared_y": raw_shared}), "line", None
        )
        self.assertNotIn("shared_weighted_output", capture.stage_refs)

        # The return event observes the final local used by `return y + shared_y`.
        capture._trace_target_sparse(
            SimpleNamespace(f_locals={"shared_y": weighted_shared}),
            "return",
            official_result,
        )
        self.assertIs(capture.stage_refs["shared_weighted_output"], weighted_shared)
        self.assertIs(capture.stage_refs["moe_output_official"], official_result)
        self.assertFalse(capture.active_target)


if __name__ == "__main__":
    unittest.main()
