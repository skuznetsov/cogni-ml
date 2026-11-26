"""CPU falsifiers for the common-prefix generation replay comparator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "compare_generation_replay.py"
NATIVE_REPLAY = ROOT / "examples" / "drama3_generation_replay.cr"
SPEC = importlib.util.spec_from_file_location("drama3_generation_compare_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def trace_values(name: str, step: int) -> list[float]:
    if name.endswith(".main_logits") or ".tower_logits." in name:
        return [float(index) / 16.0 + step / 1024.0 for index in range(32)]
    return [0.125 + step / 1024.0, -0.25, 0.5]


def make_bundle(*, truncated: bool = True, frame_count: int = 4) -> dict:
    frames = [[101 + row, 1 + row, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              for row in range(frame_count)]
    frame_count = len(frames)
    steps = compare._expected_steps(frame_count)
    traces = []
    for step in steps:
        names = [
            f"generation.step{step}.backbone_hidden",
            f"generation.step{step}.main_logits",
        ]
        if step < frame_count:
            names.extend([
                f"generation.step{step}.tower_hidden",
                *(f"generation.step{step}.tower_logits.{head}" for head in range(9)),
            ])
        for name in names:
            values = trace_values(name, step)
            traces.append({"name": name, "shape": [len(values)], "dtype": "float32", "values": values})
    return {
        "format": compare.TRACE_FORMAT,
        "source_report": {"path": "source.wav.json", "sha256": hashlib.sha256(b"report").hexdigest()},
        "source_generation": {
            "elapsed_seconds": 12.5,
            "generation_seconds": 8.25,
            "truncated": truncated,
            "stopped_on_eos": not truncated,
        },
        "prompt_ids": [7, 11, 13],
        "prompt_rows": 3,
        "frames": frames,
        "frame_count": frame_count,
        "capture_policy": {
            "checkpoints": list(compare.CHECKPOINTS),
            "tail_window_frames": compare.TAIL_WINDOW_FRAMES,
            "tail_inclusive": True,
        },
        "capture_steps": steps,
        "model_config": {
            "hidden_size": 3,
            "main_vocab_size": 32,
            "tower_dim": 3,
            "tower_vocab_size": 32,
            "num_codebooks": 10,
        },
        "model_files": {
            "config.json": hashlib.sha256(b"config").hexdigest(),
            "model.safetensors.index.json": hashlib.sha256(b"index").hexdigest(),
        },
        "execution": {"backend": "native_resident", "dense_reduction": "simd"},
        "source_hashes": {"examples/drama3_generation_replay.cr": hashlib.sha256(b"native").hexdigest()},
        "terminal_step": {
            "index": frame_count,
            "measured_in_generation": not truncated,
            "interpretation": (
                "counterfactual_after_truncated_prefix" if truncated
                else "actual_terminal_prediction_after_reported_frames"
            ),
        },
        "traces": traces,
    }


def trace_map(bundle: dict) -> dict[str, dict]:
    return {trace["name"]: trace for trace in bundle["traces"]}


class GenerationReplayComparatorSpec(unittest.TestCase):
    def test_native_capture_helper_filters_checkpoints_after_frame_count(self) -> None:
        source = NATIVE_REPLAY.read_text(encoding="utf-8")
        checkpoint_match = re.search(r"^\s+CHECKPOINTS\s*=\s*\[([^]]+)\]$", source, re.MULTILINE)
        window_match = re.search(r"^\s+TAIL_WINDOW_FRAMES\s*=\s*(\d+)\s*$", source, re.MULTILINE)
        method_match = re.search(
            r"(?ms)^    def capture_steps\(frame_count : Int32\) : Array\(Int32\)\n"
            r"(?P<body>.*?)(?=^    end$)",
            source,
        )
        self.assertIsNotNone(checkpoint_match)
        self.assertIsNotNone(window_match)
        self.assertIsNotNone(method_match)
        checkpoints = [int(value.strip()) for value in checkpoint_match.group(1).split(",")]
        tail_window = int(window_match.group(1))
        body = method_match.group("body")
        self.assertRegex(body, r"CHECKPOINTS\.select\s*\{\s*\|index\|\s*index\s*<=\s*frame_count\s*\}")
        self.assertIn("(tail_start..frame_count).to_a", body)
        self.assertRegex(body, r"\.uniq\.sort\s*$")

        for frame_count in (4, 489):
            tail_start = frame_count - tail_window if frame_count > tail_window else 0
            source_steps = sorted({step for step in checkpoints if step <= frame_count}
                                  | set(range(tail_start, frame_count + 1)))
            self.assertEqual(source_steps, compare._expected_steps(frame_count))
        long_steps = sorted({step for step in checkpoints if step <= 489}
                            | set(range(489 - tail_window, 490)))
        self.assertNotIn(512, long_steps)
        self.assertNotIn(640, long_steps)
        self.assertEqual(long_steps[-1], 489)

    def test_identical_f32_json_roundtrip_passes_with_fixed_tolerance(self) -> None:
        reference = make_bundle()
        actual = json.loads(json.dumps(reference, allow_nan=False))
        # The two JSON decimals differ but round to the same declared F32 bits.
        trace_map(actual)["generation.step0.backbone_hidden"]["values"][0] = 0.1
        trace_map(reference)["generation.step0.backbone_hidden"]["values"][0] = 0.10000000149011612
        reference["execution"] = {"backend": "mlx_reference", "device": "gpu"}
        actual["execution"] = {"backend": "native_resident", "dense_reduction": "simd"}
        report = compare.compare_bundles(reference, actual)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["tolerances"], {"atol": 6e-5, "rtol": 3e-4})
        self.assertEqual(report["parity_metrics"]["out_of_tolerance_count"], 0)
        self.assertEqual(len(report["stages"]), len(reference["traces"]))
        self.assertEqual(report["reference_execution"], reference["execution"])
        self.assertEqual(report["actual_execution"], actual["execution"])
        self.assertEqual(report["reference_provenance"]["source_hashes"], reference["source_hashes"])
        self.assertEqual(report["actual_provenance"]["model_files"], actual["model_files"])
        self.assertIn("does not attest model shard bytes", report["comparison_scope"]["note"])

    def test_source_hash_and_prefix_identity_mismatches_are_rejected(self) -> None:
        cases = []
        reference = make_bundle()
        changed_hash = copy.deepcopy(reference)
        changed_hash["source_report"]["sha256"] = "0" * 64
        cases.append(("source hash", changed_hash))
        changed_frames = copy.deepcopy(reference)
        changed_frames["frames"][0][0] += 1
        cases.append(("frame prefix", changed_frames))
        changed_prompt = copy.deepcopy(reference)
        changed_prompt["prompt_ids"][0] += 1
        cases.append(("prompt prefix", changed_prompt))
        changed_steps = copy.deepcopy(reference)
        changed_steps["capture_steps"] = changed_steps["capture_steps"][:-1]
        cases.append(("capture steps", changed_steps))
        changed_policy = copy.deepcopy(reference)
        changed_policy["capture_policy"]["tail_window_frames"] = 127
        cases.append(("capture policy", changed_policy))
        changed_terminal = copy.deepcopy(reference)
        changed_terminal["terminal_step"]["measured_in_generation"] = True
        cases.append(("terminal semantics", changed_terminal))

        for label, actual in cases:
            with self.subTest(label=label), self.assertRaises(compare.InvalidTrace):
                compare.compare_bundles(reference, actual)

    def test_tail_window_schedule_covers_start_through_terminal(self) -> None:
        for frame_count in (4, 127, 128, 129, 200, 489, 768):
            with self.subTest(frame_count=frame_count):
                steps = compare._expected_steps(frame_count)
                tail_start = max(0, frame_count - 128)
                self.assertEqual(steps, sorted(set(
                    [step for step in compare.CHECKPOINTS if step <= frame_count]
                    + list(range(tail_start, frame_count + 1))
                )))
                self.assertTrue(set(range(tail_start, frame_count + 1)).issubset(steps))
                self.assertIn(tail_start, steps)
                self.assertIn(frame_count, steps)
                self.assertEqual(steps, sorted(set(steps)))
                if tail_start > 0 and tail_start - 1 not in compare.CHECKPOINTS:
                    self.assertNotIn(tail_start - 1, steps)
        self.assertEqual(compare._expected_steps(4), list(range(5)))
        self.assertEqual(compare._expected_steps(128), list(range(129)))
        self.assertEqual(compare._expected_steps(129), list(range(130)))
        long_steps = compare._expected_steps(489)
        self.assertEqual(long_steps[long_steps.index(361):], list(range(361, 490)))
        self.assertNotIn(360, long_steps)
        self.assertEqual(len(long_steps), 140)
        self.assertEqual(long_steps[-1], 489)

    def test_comparator_rejects_missing_late_tail_step(self) -> None:
        reference = make_bundle(frame_count=130)
        actual = copy.deepcopy(reference)
        late_step = 129
        actual["capture_steps"].remove(late_step)
        actual["traces"] = [
            trace for trace in actual["traces"]
            if not trace["name"].startswith(f"generation.step{late_step}.")
        ]
        with self.assertRaisesRegex(compare.InvalidTrace, "capture_steps"):
            compare.compare_bundles(reference, actual)

    def test_hidden_stage_error_fails_fixed_tolerance_and_reports_worst_value(self) -> None:
        reference = make_bundle()
        actual = copy.deepcopy(reference)
        name = "generation.step1.backbone_hidden"
        trace_map(actual)[name]["values"][1] += 0.001
        report = compare.compare_bundles(reference, actual)
        self.assertEqual(report["status"], "mismatch")
        self.assertGreater(report["parity_metrics"]["out_of_tolerance_count"], 0)
        stage = next(item for item in report["stages"] if item["name"] == name)
        self.assertEqual(stage["value_count"], 3)
        self.assertEqual(stage["out_of_tolerance_count"], 1)
        self.assertAlmostEqual(stage["max_abs"], 0.001, places=6)
        self.assertGreater(stage["rms"], 0)
        self.assertIsNotNone(stage["relative_l2"])
        self.assertEqual(stage["worst"]["coordinate"], [1])
        self.assertAlmostEqual(stage["worst"]["reference"], -0.25)
        self.assertAlmostEqual(stage["worst"]["actual"], -0.249)
        self.assertGreater(stage["worst"]["allowed_error"], 6e-5)

    def test_late_measured_step_failure_is_included(self) -> None:
        reference = make_bundle()
        actual = copy.deepcopy(reference)
        name = "generation.step2.main_logits"
        trace_map(actual)[name]["values"][31] += 0.5
        report = compare.compare_bundles(reference, actual)
        self.assertEqual(report["status"], "mismatch")
        step = next(item for item in report["steps"] if item["step"] == 2)
        self.assertTrue(step["included_in_parity"])
        self.assertGreater(step["metrics"]["out_of_tolerance_count"], 0)
        stage = next(item for item in step["stages"] if item["name"] == name)
        self.assertTrue(stage["logit_ranking"]["diagnostic_only"])
        self.assertIn("top1_equal", stage["logit_ranking"])
        self.assertIn("top30_overlap_count", stage["logit_ranking"])

    def test_missing_trace_and_nonfinite_value_are_malformed(self) -> None:
        reference = make_bundle()
        missing = copy.deepcopy(reference)
        missing["traces"].pop()
        with self.assertRaisesRegex(compare.InvalidTrace, "missing"):
            compare.compare_bundles(reference, missing)

        nonfinite = copy.deepcopy(reference)
        trace_map(nonfinite)["generation.step0.backbone_hidden"]["values"][0] = math.nan
        with self.assertRaisesRegex(compare.InvalidTrace, "non-finite"):
            compare.compare_bundles(reference, nonfinite)

    def test_identically_truncated_vectors_are_rejected_by_pinned_dimensions(self) -> None:
        reference = make_bundle()
        actual = copy.deepcopy(reference)
        name = "generation.step0.backbone_hidden"
        for bundle in (reference, actual):
            record = trace_map(bundle)[name]
            record["shape"] = [2]
            record["values"] = record["values"][:2]
        with self.assertRaisesRegex(compare.InvalidTrace, r"expected \[3\]"):
            compare.compare_bundles(reference, actual)

    def test_shared_model_dimensions_and_file_hashes_must_match(self) -> None:
        reference = make_bundle()
        changed_dimension = copy.deepcopy(reference)
        changed_dimension["model_config"]["hidden_size"] = 4
        for record in trace_map(changed_dimension).values():
            if record["name"].endswith(".backbone_hidden"):
                record["shape"] = [4]
                record["values"].append(0.0)
        with self.assertRaisesRegex(compare.InvalidTrace, "model_config"):
            compare.compare_bundles(reference, changed_dimension)

        changed_hash = copy.deepcopy(reference)
        changed_hash["model_files"]["config.json"] = "0" * 64
        with self.assertRaisesRegex(compare.InvalidTrace, "model_files"):
            compare.compare_bundles(reference, changed_hash)

        changed_index_hash = copy.deepcopy(reference)
        changed_index_hash["model_files"]["model.safetensors.index.json"] = "0" * 64
        with self.assertRaisesRegex(compare.InvalidTrace, "model_files"):
            compare.compare_bundles(reference, changed_index_hash)

    def test_tied_logit_ranking_uses_deterministic_index_order(self) -> None:
        reference = [1.0] * 35
        actual = [0.0] + [1.0] * 34
        result = compare._logit_diagnostic(reference, actual)
        self.assertFalse(result["top1_equal"])
        self.assertEqual(result["top30_overlap_count"], 29)
        self.assertTrue(result["diagnostic_only"])

    def test_counterfactual_terminal_is_reported_but_not_gated(self) -> None:
        reference = make_bundle(truncated=True)
        actual = copy.deepcopy(reference)
        name = "generation.step4.main_logits"
        trace_map(actual)[name]["values"][0] += 1.0
        report = compare.compare_bundles(reference, actual)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["parity_scope"]["counterfactual_terminal_excluded_from_pass_fail"])
        self.assertEqual(report["parity_scope"]["counterfactual_terminal_step"], 4)
        self.assertEqual(report["counterfactual_terminal_metrics"]["out_of_tolerance_count"], 1)
        terminal = next(item for item in report["steps"] if item["step"] == 4)
        self.assertFalse(terminal["measured_in_generation"])
        self.assertFalse(terminal["included_in_parity"])

        measured_reference = make_bundle(truncated=False)
        measured_actual = copy.deepcopy(measured_reference)
        trace_map(measured_actual)[name]["values"][0] += 1.0
        measured = compare.compare_bundles(measured_reference, measured_actual)
        self.assertEqual(measured["status"], "mismatch")
        self.assertTrue(next(item for item in measured["steps"] if item["step"] == 4)
                         ["included_in_parity"])

    def test_numeric_cli_failure_writes_full_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            actual_path = root / "actual.json"
            output_path = root / "comparison.json"
            reference = make_bundle()
            actual = copy.deepcopy(reference)
            trace_map(actual)["generation.step0.backbone_hidden"]["values"][0] += 0.01
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
            actual_path.write_text(json.dumps(actual), encoding="utf-8")
            code = compare.main([
                "--reference", str(reference_path), "--actual", str(actual_path),
                "--output", str(output_path),
            ])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["status"], "mismatch")

    def test_malformed_cli_input_returns_two_without_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            actual_path = root / "actual.json"
            output_path = root / "comparison.json"
            reference_path.write_text('{"format":"bad","format":"duplicate"}', encoding="utf-8")
            actual_path.write_text(json.dumps(make_bundle()), encoding="utf-8")
            code = compare.main([
                "--reference", str(reference_path), "--actual", str(actual_path),
                "--output", str(output_path),
            ])
            self.assertEqual(code, 2)
            self.assertFalse(output_path.exists())

    def test_cli_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            actual_path = root / "actual.json"
            output_path = root / "comparison.json"
            reference_path.write_text(json.dumps(make_bundle()), encoding="utf-8")
            actual_path.write_text(json.dumps(make_bundle()), encoding="utf-8")
            output_path.write_text("preserve-me\n", encoding="utf-8")
            code = compare.main([
                "--reference", str(reference_path), "--actual", str(actual_path),
                "--output", str(output_path),
            ])
            self.assertEqual(code, 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "preserve-me\n")


if __name__ == "__main__":
    unittest.main()
