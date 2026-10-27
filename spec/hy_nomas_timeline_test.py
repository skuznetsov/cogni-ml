"""CPU-only contract tests for the offline Hy-Nomas director timeline."""

from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples/hy_nomas_timeline.py"
SPEC = importlib.util.spec_from_file_location("hy_nomas_timeline", MODULE_PATH)
timeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = timeline
SPEC.loader.exec_module(timeline)


def project(*, duration="3", changes=None, fps=24, width=1280, height=720):
    return {
        "schema": "hy_nomas_director_project_v1",
        "project_id": "two-prompts",
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": Decimal(duration),
        "prompt_changes": changes if changes is not None else [
            {"at_seconds": Decimal("0"), "prompt": "A red kite over a field."},
            {"at_seconds": Decimal("1.4"), "prompt": "The kite turns toward the sea."},
        ],
    }


class TimelineCompileSpec(unittest.TestCase):
    def test_compile_reports_requested_and_aligned_times_and_exact_upstream_counts(self):
        compiled = timeline.compile_project(project())
        self.assertEqual(compiled["status"], "compiled_only")
        self.assertEqual(compiled["fps"], 24)
        self.assertEqual(compiled["requested_duration_seconds"], "3")
        self.assertEqual(compiled["effective_duration_seconds"], "3")
        self.assertEqual(compiled["frame_count"], 72)
        self.assertEqual(compiled["latent_frames"], 18)
        self.assertEqual(compiled["requested_frame_boundaries"], [0, 33, 72])
        self.assertEqual(compiled["segment_frame_boundaries"], [0, 36, 72])
        self.assertEqual(compiled["segment_latent_boundaries"], [0, 9, 18])
        self.assertEqual(compiled["prompts"], [
            "A red kite over a field.", "The kite turns toward the sea."])
        second = compiled["segments"][1]
        self.assertEqual(second["requested_start_seconds"], "1.4")
        self.assertEqual(second["requested_start_frame"], 33)
        self.assertEqual(second["effective_start_frame"], 36)
        self.assertEqual(second["effective_start_seconds"], "1.5")

    def test_compile_emits_event_mode_and_explicit_prompt_and_boundary_overrides(self):
        compiled = timeline.compile_project(project())
        self.assertEqual(compiled["upstream_overrides"], {
            "inference.rollout_type": "seg_ar",
            "inference.video_length": "event",
            "inference.video_fps": 24,
            "inference.video_size": [720, 1280],
            "inference.num_frames": 72,
            "inference.segment_frame_boundaries": [0, 36, 72],
            "inference.max_segments": 2,
            "inference.desc_field_first": "prompt",
            "inference.desc_field_rest": "prompt",
        })
        self.assertEqual(compiled["upstream_input"][0]["segments"], [
            {"prompt": "A red kite over a field."},
            {"prompt": "The kite turns toward the sea."},
        ])
        self.assertIn("compile-only", " ".join(compiled["warnings"]).lower())

    def test_nondefault_geometry_is_bound_to_upstream_video_size_order(self):
        compiled = timeline.compile_project(project(width=640, height=1024))
        self.assertEqual((compiled["width"], compiled["height"]), (640, 1024))
        self.assertEqual(compiled["upstream_overrides"]["inference.video_size"],
                         [1024, 640])

    def test_rejects_prompt_reserved_upstream_segment_separator(self):
        with self.assertRaises(timeline.TimelineError):
            timeline.compile_project(project(changes=[
                {"at_seconds": Decimal("0"), "prompt": "first ||| second"},
            ]))

    def test_effective_boundaries_round_up_from_exact_requested_time(self):
        compiled = timeline.compile_project(project(
            duration="3.001",
            changes=[
                {"at_seconds": Decimal("0"), "prompt": "first"},
                {"at_seconds": Decimal("1.501"), "prompt": "second"},
            ]))
        self.assertEqual(compiled["requested_frame_boundaries"], [0, 36, 72])
        self.assertEqual(compiled["segment_frame_boundaries"], [0, 40, 76])
        self.assertEqual(compiled["segment_latent_boundaries"], [0, 10, 19])
        self.assertEqual(compiled["segments"][1]["effective_start_frame"], 40)
        self.assertGreater(
            Decimal(compiled["segments"][1]["effective_start_seconds"]),
            Decimal("1.501"))
        self.assertGreater(Decimal(compiled["effective_duration_seconds"]),
                           Decimal("3.001"))

    def test_requested_decimal_is_preserved_without_context_rounding(self):
        requested = Decimal("1.50000000000000000000000000001")
        compiled = timeline.compile_project(project(changes=[
            {"at_seconds": Decimal("0"), "prompt": "first"},
            {"at_seconds": requested, "prompt": "second"},
        ]))
        self.assertEqual(compiled["segments"][1]["requested_start_seconds"],
                         "1.50000000000000000000000000001")
        self.assertEqual(compiled["segments"][1]["effective_start_frame"], 40)

    def test_single_prompt_uses_upstream_single_event_field_contract(self):
        compiled = timeline.compile_project(project(changes=[
            {"at_seconds": Decimal("0"), "prompt": "A quiet forest."},
        ]))
        self.assertEqual(compiled["upstream_overrides"]["inference.max_segments"], 1)
        self.assertEqual(compiled["upstream_overrides"]["inference.desc_field_first"],
                         "prompt")
        self.assertEqual(compiled["upstream_overrides"]["inference.desc_field_rest"], "")

    def test_rejects_non_pinned_fps_bad_geometry_and_invalid_duration(self):
        for changes, fps, width, duration in (
            (None, 30, 1280, "3"),
            (None, 24, 1279, "3"),
            (None, 24, 1280, "0"),
        ):
            with self.subTest(fps=fps, width=width, duration=duration):
                candidate = project(fps=fps, duration=duration, changes=changes)
                candidate["width"] = width
                with self.assertRaises(timeline.TimelineError):
                    timeline.compile_project(candidate)

    def test_rejects_missing_zero_start_unsorted_and_overshooting_changes(self):
        invalid = (
            [{"at_seconds": Decimal("0.5"), "prompt": "late start"}],
            [
                {"at_seconds": Decimal("0"), "prompt": "first"},
                {"at_seconds": Decimal("2"), "prompt": "second"},
                {"at_seconds": Decimal("1"), "prompt": "out of order"},
            ],
            [
                {"at_seconds": Decimal("0"), "prompt": "first"},
                {"at_seconds": Decimal("3"), "prompt": "at end"},
            ],
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(timeline.TimelineError):
                timeline.compile_project(project(changes=changes))

    def test_rejects_subframe_aliases_and_changes_that_collapse_after_alignment(self):
        cases = (
            [
                {"at_seconds": Decimal("0"), "prompt": "first"},
                {"at_seconds": Decimal("0.01"), "prompt": "same latent boundary"},
                {"at_seconds": Decimal("0.02"), "prompt": "same latent boundary"},
            ],
            [
                {"at_seconds": Decimal("0"), "prompt": "first"},
                {"at_seconds": Decimal("0.5001"), "prompt": "first aligned boundary"},
                {"at_seconds": Decimal("0.51"), "prompt": "same aligned boundary"},
            ],
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(timeline.TimelineError):
                timeline.compile_project(project(changes=changes))

    def test_load_preserves_decimal_timestamp_and_cli_output_never_clobbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "project.json"
            output = root / "compiled.json"
            source.write_text(json.dumps({
                "schema": "hy_nomas_director_project_v1",
                "project_id": "decimal",
                "fps": 24,
                "width": 1280,
                "height": 720,
                "duration_seconds": 1.0,
                "prompt_changes": [{"at_seconds": 0.0, "prompt": "Still."}],
            }))
            loaded = timeline.load_project(source)
            self.assertIsInstance(loaded["duration_seconds"], Decimal)
            self.assertEqual(timeline.write_compiled(source, output), 0)
            saved = json.loads(output.read_text())
            self.assertEqual(saved["frame_count"], 24)
            with self.assertRaises(FileExistsError):
                timeline.write_compiled(source, output)
            self.assertEqual(json.loads(output.read_text()), saved)


if __name__ == "__main__":
    unittest.main()
